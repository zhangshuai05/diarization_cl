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
    ap.add_argument("--refine", default="on", choices=("on", "off"),
                    help="是否走两级细化。off 仅用于诊断，产物会标 level=none")
    ap.add_argument("--output")
    args = ap.parse_args()

    class_limits = [int(x) for x in args.classes.split(",")]
    norms = [x == "cosine" for x in args.norm.split(",")]

    root = Path(args.root)
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

    variants = ["none"] + [f"c{c}_{'cos' if n else 'euc'}"
                           for c in class_limits for n in norms]
    rows = []
    for meeting in meetings:
        if meeting not in refine_cache:
            continue
        rE, rtimes = refine_cache[meeting]
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        seconds = args.seconds if args.seconds > 0 else max(s["end"] for s in reference)
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
        if len(emb) < 2 or len(effective) < 2:
            continue
        k = min(len(effective), len(emb))
        row = {"meeting": meeting, "true_k": len(effective),
               "n_windows": int(len(emb)), "variants": {}}
        for name in variants:
            if name == "none":
                vectors, metric, kept = emb, "cosine", None
                rvectors = rE                      # 裸空间：两侧同为 192 维原始嵌入
            else:
                limit = int(name[1:name.index("_")])
                normalize = name.endswith("cos")
                fit, kept = fit_backend(pool, meeting, limit, normalize)
                if fit is None:
                    # `level` 必须在**每一行**上都在：产物是按行读的，缺键会让
                    # "这一步没走" 与 "这段代码没写" 分不开（Task 5 的汇总
                    # 直接读 `v.get('level')`，缺键显示成空而非"未拟合"）。
                    row["variants"][name] = {"accuracy": None, "level": None}
                    continue
                vectors, metric = fit(emb), fit.metric
                # ⚠️ 细化窗必须过**同一个** `fit`。`_refine_labels` 里是
                # `rE @ anchors.T`，而 anchors 取自 `vectors` —— 若 rE 留在
                # 原始 192 维、vectors 已在 LDA 空间（维度 = 判别方向数），
                # 这条矩阵乘直接 `ValueError`，**每个 LDA 臂都会崩**。
                # 只 `none` 臂能跑（两侧同为原始嵌入），于是崩溃被伪装成
                # "后端臂无数据"。投影后两侧同空间，与生产同构。
                rvectors = fit(rE)
            clusters = cluster_labels(vectors, k, metric=metric)
            fclusters, ftimes, level = refine_labels(
                clusters, vectors, times, rvectors, rtimes, args.refine == "on")
            # `AB._win` 只认 refined/2s-only；关掉细化时该用主窗（`v12.WIN_S`）
            fwin = v12.WIN_S if level == "none" else AB._win(level)
            named = map_by_frames(reference, ftimes, fclusters, np.full(len(ftimes), fwin))
            acc = frame_accuracy(reference, ftimes, named, np.full(len(ftimes), fwin))
            row["variants"][name] = {"accuracy": acc["accuracy"],
                                     "classes": kept,
                                     "level": level, "n_final": int(len(ftimes)),
                                     "n_classes": len(kept) if kept else 0}
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

    base = {row["meeting"]: row["variants"]["none"]["accuracy"] for row in rows}
    print()
    for v in variants[1:]:
        deltas = []
        for row in rows:
            a = (row["variants"].get(v) or {}).get("accuracy")
            if a is not None:
                deltas.append(a - base[row["meeting"]])
        if not deltas:
            continue
        deltas = np.array(deltas)
        wins = int((deltas > 0.01).sum())
        losses = int((deltas < -0.01).sum())
        print(f"  {v:>11}  净值 {deltas.mean() * 100:+6.1f} 点   "
              f"赢{wins}/亏{losses}/平{len(deltas) - wins - losses}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-backend-sweep-v1", "meetings": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
