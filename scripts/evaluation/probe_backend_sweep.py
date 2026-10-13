# -*- coding: utf-8 -*-
"""后端扫描：LDA 打分为何净负，以及怎样（如果能）修好。

上一轮把 LDA 的类修成 (身份组, 说话人) 之后净收益反而更差（dev −2.1、holdout −5.1），
而崩掉的场次（S_R004 组）恰好是训练里含**极小类**的那些：S_R004S02C01 的
002-F=2 窗、003-F=2 窗。`lda_fit` 只排除 <2 窗的类，于是 2 窗的类贡献了
接近奇异的判别方向。这是**实验的缺陷**，不是"后端没用"的证据。

同时在扫的轴：
  min_class_windows  —— 每个 LDA 类的纯窗下限（2 / 8 / 16 / 32）
  normalize          —— LDA 输出后是否 L2 归一化再聚类
                        False：欧氏距离 + average linkage（LDA 坐标的原生度量）
                        True ：余弦距离（与裸余弦基线同度量，只换了空间）
  训练池             —— 默认用**本 split 全部场次**（除目标场）拟合；`--meetings`
                        可收窄。**不含 holdout** —— 见 `build_pool` docstring。

零推理。

    python scripts/evaluation/probe_backend_sweep.py --split dev --seconds 600
"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from v12_aishell4 import meeting_group, split_meetings  # noqa: E402
import probe_ab_dev as AB  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_pair_oracle import Transform, lda_fit, pure_windows  # noqa: E402
from probe_sentence_attribution import (  # noqa: E402
    attribution, segment_map, sentence_dict_predictions)
from unlocked_selector import CAP, MIN_WIN, build_unlocked, pick_unlocked  # noqa: E402


def load(cache, meeting, window, hop):
    data = np.load(Path(cache) / meeting / f"w{window:g}_h{hop:g}.npz",
                   allow_pickle=False)
    return (data["emb"], np.asarray(data["times"], dtype=float),
            data["speakers"].astype(str), np.asarray(data["votes"], dtype=float))


DEFAULT_CACHE = r"D:\yinp\AISHELL-4\_v13_spike\cache"


def build_pool(meetings, cache, window, hop):
    """加载标注窗缓存，供 LDA 拟合。

    **只加载传入的场次** —— 由调用方决定范围。
    当前唯一调用方是 `main`，它传**本 split 的场次**。

    ⚠️ 曾经这里写的是 `set(dev + holdout)`，于是对 dev 的每一场，
    LDA 方向都从含 8 场 holdout 的池里学 —— 封存集为一个输出数字做了贡献，
    直接违反 spec §3.3。改成按 split 收口后池从 20 场降到 12 场。
    """
    pool = {}
    for meeting in meetings:
        try:
            pool[meeting] = load(cache, meeting, window, hop)
        except FileNotFoundError:
            continue
    return pool


def fit_backend(pool, target, min_class_windows, normalize):
    """用除目标场外的全部纯窗拟合 LDA，类 = (身份组, 说话人)。

    返回 `(fit, kept_names)` —— `kept_names` 是**保留类的名字清单**（已排序），
    不是个数。`classes` 轴的真实分辨率只有两点（见全局约束），
    只有类名清单能让产物自证用了哪批类。
    """
    Xs, ys = [], []
    for meeting, (emb, times, speakers, votes) in pool.items():
        if meeting == target:
            continue
        pe, _, ps = pure_windows(emb, times, speakers, votes)
        if len(pe):
            Xs.append(pe)
            ys.append(np.array([f"{meeting_group(meeting)}:{x}" for x in ps]))
    if not Xs:
        return None, []
    X, y = np.concatenate(Xs), np.concatenate(ys)
    counts = Counter(y.tolist())
    kept_names = sorted(c for c, n in counts.items() if n >= min_class_windows)
    if len(kept_names) < 2:
        return None, []
    fit = lda_fit(X, y, min_class_windows=min_class_windows)
    if fit is None:
        return None, kept_names
    return Transform(fit.mean, fit.weights, normalize=normalize), kept_names


def refine_labels(labels, E, times, rE, rtimes, enabled=True):
    """走 v12 的同一套两级细化，返回 (labels, times, level)。

    **两臂必须同构**：若 `none` 臂与 LDA 臂都在无细化下跑，结论是
    "后端在 AHC 那一层的差异"；而生产里细化按簇质心重新归属、
    会把簇边界错误吸掉一部分 —— **可能正好把 LDA 想修的那部分先修了**。
    不带细化的数字对"这条线有没有东西"没有回答力。

    ⚠️ `vectors` 必须传**聚类时用的那个空间**（LDA 输出），
    与 `real_path_labels` 内部取质心一致。
    """
    if not enabled:
        return labels, times, "none"
    return AB.real_path_labels(labels, E, times, rE, rtimes)


def variants_for(class_limits, norms, k_modes=("true",)):
    """枚举全部变体名。

    命名：`{none|c<类窗下限>}_{cos|euc}[_autok]`

    ⚠️ `_euc` 变体**不进 autok 臂**：`build_unlocked` 内部
    `linkage(E, method="average", metric="cosine")` 是写死的
    （`unlocked_selector.py:48`），而欧氏空间的 LDA 输出必须用欧氏距离。
    让它跑 autok 等于用余弦 silhouette 评价欧氏空间。
    给它加 metric 参数会动到生产共用实现 —— 不值得，跳过即可。
    """
    out = []
    for k_mode in k_modes:
        suffix = "_autok" if k_mode == "auto" else ""
        if k_mode == "auto":
            out.append("none" + suffix)
        elif k_mode == "true":
            out.append("none")
        for c in class_limits:
            for n in norms:
                if k_mode == "auto" and not n:
                    continue
                out.append(f"c{c}_{'cos' if n else 'euc'}{suffix}")
    # 去重保序
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def sentence_metrics(sentences, labels, times, win):
    """句级归属率，带 clean / overlap / overall 三桶。

    ⚠️ **不改 `AB.score`** —— 它只返回 `overall` 派生的 6 个键，而现有调用方
    （`probe_unlocked_sentence.run_meeting` 等）全是 `**AB.score(...)` 解包，
    给它加桶会静默改掉它们的产物 schema。这里重建 `score` 的取桶路径，
    段表构造与 `score` 逐行一致（`segment_map` → `sentence_dict_predictions`
    → `attribution`），只是不丢掉三桶。

    **弃权与错答必须分开报**：弃权是显式缺失（读者看得到），错答是静默错误
    （读者被误导而不自知）。合报一个 clean 数会让两者在不该互换的预算里互换。

    `wrong = 1 - per_sentence - abstain`：**clean 桶含弃权句且算未答对**，
    所以 `1 - per_sentence` 里混着弃权，不能直接当错答率。
    """
    end = max((s["end"] for s in sentences), default=0.0)
    w = np.full(len(labels), win)
    segs = segment_map(times, labels, w, end)
    pred = sentence_dict_predictions(sentences, segs)
    a = attribution(sentences, pred)
    out = {}
    for bucket in ("overall", "clean", "overlap"):
        b = a[bucket]
        ps, ab = b["per_sentence"], b["abstain"]
        out[bucket] = {
            "per_sentence": ps,
            "abstain": ab,
            "wrong": None if ps is None or ab is None else 1.0 - ps - ab,
            "n": b["n"],
        }
    return out


def stub_row(name, k_mode, skip=None):
    """未出数的变体行。**键集与成功行逐位对齐** —— 产物是按行读的，
    "这个键没写" 与 "这一步没走" 在 JSON 里长得一样（`json.dumps` 省略
    None vs 整个键缺失，读的人分不出）。缺 `classes` 曾让下游必须写
    `.get("classes")` 才能不崩，而 `.get` 把"键不存在"和"值为 None"混成一个结果。
    """
    row = {"accuracy": None, "classes": None, "level": None,
           "n_final": None, "n_classes": None, "k_mode": k_mode,
           "k_used": None, "sentence": None}
    if skip is not None:
        row["skip"] = skip
    return row


def k_modes_arg(text):
    """解析 `--k-modes`，未知取值直接报错。

    不这样做的话 `--k-modes True` 会静默落进 `variants_for` 的 else 兜底：
    既不产 `none` 臂，行里的 `k_modes` 又写着 `True` —— 产物对自己撒谎。
    一个存在意义就是"记录这一跑用了哪种 K 模式"的开关，不能报错数。
    """
    modes = [piece for piece in text.split(",") if piece]
    unknown = sorted(set(modes) - {"true", "auto"})
    if unknown:
        raise argparse.ArgumentTypeError(f"未知 k-mode：{unknown}（只认 true / auto）")
    return modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="显式场次列表；缺省 = 本 split 全部")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=v12.WIN_S)
    ap.add_argument("--hop", type=float, default=v12.HOP_S)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--classes", default="2,8,16,32")
    ap.add_argument("--norm", default="euclid,cosine")
    ap.add_argument("--k-modes", default="true,auto", type=k_modes_arg,
                    help="true = 真值有效人数（诊断）；auto = pick_unlocked 自选（判据）")
    ap.add_argument("--refine", default="on", choices=("on", "off"),
                    help="是否走两级细化。off 仅用于诊断，产物会标 level=none")
    ap.add_argument("--output")
    args = ap.parse_args()

    class_limits = [int(x) for x in args.classes.split(",")]
    norms = [x == "cosine" for x in args.norm.split(",")]

    root = Path(args.root)
    # `AB.load_sentences` 读的是 `AB.DATA` 下的 TextGrid，而本脚本的 `--root`
    # 才是这一跑的数据根 —— 不接上就会**静默读另一个目录**的标注
    # （或在别的机器上直接 FileNotFoundError）。
    AB.DATA = root
    meetings = args.meetings or split_meetings(root, args.split)
    # ★ 池只含本 split —— 见 build_pool docstring 与 test_backend_sweep.py
    pool = build_pool(meetings, args.cache, args.window, args.hop)

    refine_cache = {}
    for meeting in meetings:
        try:
            refine_cache[meeting] = PE.cached_stage(meeting, "refine")
        except FileNotFoundError:
            continue
    missing_refine = sorted(set(meetings) - set(refine_cache))
    if missing_refine:
        # ⚠️ 不能只 `continue`：那会让这些场次**静默**从产物里消失，
        # 而"少了几场"看起来和"本来就这么几场"一模一样 —— 本仓库
        # 反复出现的"不报错只出错数"。§R7：失败场次必须显式记录。
        print(f"⚠️ 无 refine 缓存的场次（跳过）：{missing_refine}", flush=True)

    k_modes = args.k_modes
    variants = variants_for(class_limits, norms, k_modes)
    rows = []
    for meeting in meetings:
        if meeting not in refine_cache:
            # ⚠️ 不能 `continue`：那会让这些场次**静默**从产物里消失，而"少了几场"
            # 看起来和"本来就这么几场"一模一样（§R7）。这里落存根行 —— 与下面
            # "窗数不足"那条路径对称；此前只有它落行，两条失败路径处理不一致。
            row = {"meeting": meeting, "true_k": None, "n_windows": None,
                   "variants": {n: stub_row(n, "auto" if n.endswith("_autok") else "true",
                                            "无 refine 缓存")
                                for n in variants},
                   "k_modes": k_modes}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        rE, rtimes = refine_cache[meeting]
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        # `max()` 空序列会抛 ValueError —— 空 RTTM（或所有段都被上面的 `start < seconds`
        # 滤掉）会让整场崩掉，而崩溃的场次与"本来就没这场"在产物里分不开。
        # `--seconds > 0` 时这条不参与求值，所以只在真的读空标注时才生效。
        seconds = args.seconds if args.seconds > 0 else max(
            (s["end"] for s in reference), default=0.0)
        reference = [{"start": s["start"], "end": min(s["end"], seconds),
                      "speaker": s["speaker"]} for s in reference if s["start"] < seconds]
        by_speaker = {}
        for s in reference:
            by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0.0) + s["end"] - s["start"]
        grand = sum(by_speaker.values()) or 1.0
        effective = sorted(n for n, v in by_speaker.items() if v / grand >= args.share)

        emb_all, times_all = cached_windows(args.cache, meeting, args.window, args.hop)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=seconds),
                                  args.window, v11.MIN_VAD_FRACTION)
        # 每场**只读一次** TextGrid（`load_sentences` 解的是标注文件，不是缓存），
        # 放进变体循环里会让每个变体各读一遍 —— 纯浪费。
        sentences = AB.load_sentences(meeting)
        row = {"meeting": meeting, "true_k": len(effective),
               "n_windows": int(len(emb)), "variants": {}, "k_modes": k_modes}
        if len(emb) < 2 or len(effective) < 2:
            # ⚠️ 不能 `continue` 掉整行：跳过与"本来就不跑"在产物里必须可区分。
            # 该场窗数不足/有效人数不足，逐变体记存根，行仍然落盘。
            # 与上面"无 refine 缓存"那条同理（§R7：失败场次必须显式记录）。
            reason = "窗数不足" if len(emb) < 2 else "有效人数不足"
            for name in variants:
                row["variants"][name] = stub_row(
                    name, "auto" if name.endswith("_autok") else "true", reason)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        k = min(len(effective), len(emb))
        for name in variants:
            base_name = name[:-len("_autok")] if name.endswith("_autok") else name
            k_mode = "auto" if name.endswith("_autok") else "true"
            if base_name == "none":
                vectors, metric, kept = emb, "cosine", None
                rvectors = rE                      # 裸空间：两侧同为 192 维原始嵌入
            else:
                limit = int(base_name[1:base_name.index("_")])
                normalize = base_name.endswith("cos")
                fit, kept = fit_backend(pool, meeting, limit, normalize)
                if fit is None:
                    # `level`/`k_mode`/`k_used` 都必须在**每一行**上都在：产物按行读，
                    # 缺键会让"这一步没走"与"这段代码没写"分不开。键集由 `stub_row`
                    # 统一 —— 三处失败路径各写各的曾让 `classes` 只在一处存在。
                    row["variants"][name] = stub_row(name, k_mode, "未拟合")
                    continue
                vectors, metric = fit(emb), fit.metric
                # ⚠️ 细化窗必须过**同一个** `fit`。`_refine_labels` 里是
                # `rE @ anchors.T`，而 anchors 取自 `vectors` —— 若 rE 留在
                # 原始 192 维、vectors 已在 LDA 空间（维度 = 判别方向数），
                # 这条矩阵乘直接 `ValueError`，**每个 LDA 臂都会崩**。
                # 只 `none` 臂能跑（两侧同为原始嵌入），于是崩溃被伪装成
                # "后端臂无数据"。投影后两侧同空间，与生产同构。
                rvectors = fit(rE)
            if k_mode == "auto":
                chosen = pick_unlocked(build_unlocked(vectors, CAP, MIN_WIN), mode="objective")
                if chosen is None:
                    # 空候选：与生产一致地退化，但**显式记 None 而非静默出数**
                    row["variants"][name] = stub_row(name, k_mode, "空候选")
                    continue
                k_used, clusters = int(chosen["k"]), np.asarray(chosen["labels"])
            else:
                k_used, clusters = k, cluster_labels(vectors, k, metric=metric)
            fclusters, ftimes, level = refine_labels(
                clusters, vectors, times, rvectors, rtimes, args.refine == "on")
            # `AB._win` 只认 refined/2s-only；关掉细化时该用主窗（`v12.WIN_S`）
            fwin = v12.WIN_S if level == "none" else AB._win(level)
            named = map_by_frames(reference, ftimes, fclusters, np.full(len(ftimes), fwin))
            acc = frame_accuracy(reference, ftimes, named, np.full(len(ftimes), fwin))
            # 句级指标必须和 frame 指标一样**每条臂都出**（含 `none` 臂）：
            # 少一条臂就会让"这条臂没跑"与"这条臂在后端上更好"分不开。
            # `load_sentences` 对本场返回空时记 None（而不是崩），与存根行同构。
            sent = sentence_metrics(sentences, fclusters, ftimes, fwin) if sentences else None
            row["variants"][name] = {"accuracy": acc["accuracy"],
                                     "classes": kept,
                                     "level": level, "n_final": int(len(ftimes)),
                                     "n_classes": len(kept) if kept else 0,
                                     "k_mode": k_mode, "k_used": int(k_used),
                                     "sentence": sent}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(f"\n{'会议':<16}{'K':>3}{'n':>5}" + "".join(f"{v:>11}" for v in variants))
    for row in rows:
        cells = []
        for v in variants:
            a = (row["variants"].get(v) or {}).get("accuracy")
            cells.append(f"{a * 100:10.1f}%" if a is not None else "         -")
        print(f"{row['meeting']:<16}{row['true_k']:>3}{row['n_windows']:>5}"
              + "".join(cells))

    # 基线 = 裸臂。`--k-modes auto` 时根本不存在 `none`（只有 `none_autok`），
    # 写死 `["none"]` 会 KeyError —— 新开的开关不能有值把它跑崩。
    base = {row["meeting"]: (row["variants"].get("none") or
                             row["variants"].get("none_autok") or {}).get("accuracy")
            for row in rows}
    print()
    for v in variants[1:]:
        deltas = []
        for row in rows:
            a = (row["variants"].get(v) or {}).get("accuracy")
            b = base[row["meeting"]]
            if a is not None and b is not None:
                deltas.append(a - b)
        if not deltas:
            continue
        deltas = np.array(deltas)
        wins = int((deltas > 0.01).sum())
        losses = int((deltas < -0.01).sum())
        print(f"  {v:>11}  净值 {deltas.mean() * 100:+6.1f} 点   "
              f"赢{wins}/亏{losses}/平{len(deltas) - wins - losses}")

    # 句级汇总：clean 桶的**错答与弃权分开**报。`1 - 归属率` 不是错答率
    # （clean 桶含弃权句且算未答对），硬凑成一个数会让两者互相顶替。
    print()
    print("=== 句级（clean 桶，错答与弃权分开）===")
    for v in variants:
        rows_v = [(row["variants"].get(v) or {}).get("sentence") for row in rows]
        # 无句子的场次 / 存根行 / 未拟合行都会是 None —— 逐场过滤，
        # 空集合直接跳过（不能让一个 None 把整张表打成 NaN 或抛 TypeError）。
        rows_v = [x for x in rows_v if x and x["clean"]["per_sentence"] is not None]
        if not rows_v:
            print(f"  {v:>12}  （无句级数据）")
            continue
        cw = [x["clean"]["wrong"] for x in rows_v if x["clean"]["wrong"] is not None]
        ca = [x["clean"]["abstain"] for x in rows_v if x["clean"]["abstain"] is not None]
        wrong_s = "     -" if not cw else f"{np.mean(cw) * 100:5.1f}%"
        abstain_s = "     -" if not ca else f"{np.mean(ca) * 100:5.1f}%"
        print(f"  {v:>12}  clean {np.mean([x['clean']['per_sentence'] for x in rows_v]) * 100:5.1f}%"
              f"  错答 {wrong_s}  弃权 {abstain_s}"
              f"   (n={len(rows_v)})")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-backend-sweep-v1", "meetings": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
