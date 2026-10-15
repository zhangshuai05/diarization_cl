# -*- coding: utf-8 -*-
"""3D-Speaker pipeline 移植 · 第一步：验证谱聚类能否把 K 立起来。

**为什么要移植**：我们自己的链路在 R8007_M8010 上窗级/句级聚类全部塌陷到 K=1，
而诊断显示**纯窗上 CAM++ 留一准确率 100%**（间隔 0.377）——声纹没问题，
是"取窗"和"聚类"这两步合起来出了问题。3D-Speaker（Apache 2.0）的
`infer_diarization.py` 是这套组合的成熟版本，本脚本照它的配方做第一步：

    VAD → 1.5s/0.75s chunk → CAM++ 嵌入 → 谱聚类（k-NN 图 + eigengap 自动定 K）

与我们的旧链路相比，只有三处不同（其余尽量对齐以便归因）：
    1. chunk 1.5s/0.75s（旧链路 2.0s/1.0s）
    2. 谱聚类 + k-NN 稀疏化（旧链路 AHC + 轮廓系数选 K）
    3. 用 eigengap 自动定 K（旧链路用 objective = silhouette - λ(k-2)）

**判据**：K 能否自动接近真值、句级准确率能否超过现状（R8007_M8010 = 22.2%，
R8008_M8013 = 38.1%）。**这一步不过，post_process（匈牙利匹配多标签）就没有意义。**

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_3dspeaker_pipeline.py --meetings R8007_M8010 R8008_M8013
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "streaming_asr", HERE.parent / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, RESULTS, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)
from run_der import load_models  # noqa: E402

CHUNK_S, STEP_S = 1.5, 0.75        # 3D-Speaker 的配方
MIN_CHUNK = 0.40
MAX_K = 15                          # 3D-Speaker: max_num_spks=15
MIN_K = 1
KNN_K = 8                           # k-NN 稀疏化的邻居数
SPECTRAL_SEED = 42

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def chunk_windows(vad_segments):
    """VAD 段内滑 1.5s/0.75s 的 chunk（3D-Speaker 的 chunk()）。"""
    out = []
    for start, end in vad_segments:
        t = start
        while t + CHUNK_S < end + STEP_S:
            out.append((t, min(t + CHUNK_S, end)))
            t += STEP_S
    return [(a, b) for a, b in out if b - a >= MIN_CHUNK]


def embed_chunks(model, audio, rate, wins):
    vecs, kept = [], []
    for a, b in wins:
        chunk = audio[int(a * rate):int(b * rate)]
        if len(chunk) < int(MIN_CHUNK * rate):
            continue
        try:
            e = model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if e is None:
            continue
        if hasattr(e, "detach"):
            e = e.detach().cpu().numpy()
        vecs.append(_norm(e))
        kept.append((a, b))
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32)), kept


def eigengap_k(eigenvalues, max_k=MAX_K):
    """最大特征间隙定 K（谱聚类标准做法）。"""
    vals = np.sort(eigenvalues)[: max_k + 1]
    gaps = np.diff(vals)
    if not len(gaps):
        return 1
    return int(np.argmax(gaps) + 1)


def spectral_cluster(E, n_clusters=None, knn_k=KNN_K):
    """归一化谱聚类 + k-NN 稀疏化（纯 scipy，不引 sklearn）。

    亲和矩阵用 k-NN 截断而非全局 p-percentile：全局阈值在簇大小不均时
    会把两个簇"搭桥"连通（这正是 K 塌陷的常见机制）。
    """
    from scipy.cluster.vq import kmeans2
    from scipy.linalg import eigh

    n = len(E)
    if n < 3:
        return np.zeros(n, dtype=int), 1, np.array([0.0])
    S = np.clip(E @ E.T, 0.0, None)
    np.fill_diagonal(S, 0.0)
    k = min(knn_k, n - 1)
    W = np.zeros_like(S)
    for i in range(n):
        idx = np.argpartition(S[i], -k)[-k:]
        W[i, idx] = S[i, idx]
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    L = np.eye(n) - (W * dinv[:, None]) * dinv[None, :]
    vals, vecs = eigh(L)
    k_auto = max(MIN_K, min(eigengap_k(vals), MAX_K, n - 1))
    k_use = n_clusters if n_clusters else k_auto
    U = vecs[:, :k_use]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    try:
        _, labels = kmeans2(U, k_use, minit="++", seed=SPECTRAL_SEED)
    except Exception:
        labels = np.arange(n) % k_use
    return labels.astype(int), k_use, vals[: MAX_K + 1]


def osd_probs(sess, audio):
    """pyannote/segmentation-3.0 ONNX -> 帧级 7 类概率（10s 窗 / 5s 步长，重叠窗取平均）。"""
    W, STEP, FR = 160000, 80000, 589
    fs = (W / FR) / 16000.0
    total = len(audio)
    n = int(np.ceil(total / (W / FR))) + 8
    acc = np.zeros((n, 7), np.float64)
    cnt = np.zeros(n, np.float64)
    name = sess.get_inputs()[0].name
    for start in range(0, max(1, total), STEP):
        chunk = audio[start:start + W]
        valid = len(chunk)
        if valid < W:
            chunk = np.pad(chunk, (0, W - valid))
        out = sess.run(None, {name: chunk[None, None, :].astype(np.float32)})[0][0]
        p = np.exp(out) if out.max() <= 0 else out
        vf = min(FR, int(round(valid / (W / FR))))
        i0 = int(round(start / (W / FR)))
        i1 = min(n, i0 + vf)
        if i1 <= i0:
            continue
        acc[i0:i1] += p[: i1 - i0]
        cnt[i0:i1] += 1.0
        if start + W >= total:
            break
    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return acc, mask, fs


def post_process(chunk_labels, chunk_spans, probs, k, frame_step_s):
    """3D-Speaker 的 post_process：逐 OSD 窗匈牙利匹配，把局部 powerset 类别对到全局簇。

    为什么必须逐窗：powerset 的 class 1/2/3 是**窗内局部**说话人，跨 10s 窗会重新编号，
    所以匹配只能在窗内做（这正是它能绕开"跨窗一致性"死结的原因）。
    输出帧级多标签：class 4/5/6 是双人重叠，匹配后会激活两个全局说话人。
    """
    from scipy.optimize import linear_sum_assignment

    n = len(probs)
    cluster_frames = np.zeros((n, k), dtype=np.float32)
    for (a, b), lab in zip(chunk_spans, chunk_labels.tolist()):
        i0, i1 = int(a / frame_step_s), min(n, int(b / frame_step_s))
        if i1 > i0:
            cluster_frames[i0:i1, lab] = 1.0

    win = max(1, int(10.0 / frame_step_s))
    act = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, win):
        e = min(n, s + win)
        data, ccf = probs[s:e, :], cluster_frames[s:e, :]
        if ccf.sum() == 0:
            continue
        cost = np.full((7, k), -1.0)
        for j in range(7):
            dj = data[:, j] > 0.5
            if dj.sum():
                cost[j] = [float((dj & (ccf[:, c] > 0.5)).sum()) for c in range(k)]
        rows, cols = linear_sum_assignment(-cost)
        for r, c in zip(rows, cols):
            if cost[r, c] > 0:
                act[s:e, c] = np.maximum(act[s:e, c], data[:, r])

    # 每帧取"期望说话人数"个（powerset 期望值：单人1 + 双人2）
    exp_n = np.rint(probs[:, 1:4].sum(axis=1) + 2 * probs[:, 4:7].sum(axis=1)).astype(int)
    order = np.argsort(-act, axis=1)
    binary = np.zeros_like(act)
    for t in range(n):
        for i in range(min(k, max(1, int(exp_n[t])))):
            if act[t, order[t, i]] > 0:
                binary[t, order[t, i]] = 1.0
        if binary[t].sum() == 0 and cluster_frames[t].sum() > 0:
            binary[t] = cluster_frames[t]
    return binary


def binary_to_turns(binary, frame_step_s):
    """帧级多标签 -> 段表（每段带可能的次说话人）。"""
    out = []
    k = binary.shape[1]
    for spk in range(k):
        col = binary[:, spk] > 0.5
        start = None
        for i, flag in enumerate(col):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                out.append({"start": start * frame_step_s, "end": i * frame_step_s,
                            "speaker": "C%d" % spk})
                start = None
        if start is not None:
            out.append({"start": start * frame_step_s, "end": len(col) * frame_step_s,
                        "speaker": "C%d" % spk})
    out.sort(key=lambda x: x["start"])
    return out


def turns_multilabel(binary, turns, frame_step_s, thresh=0.25):
    """句级多标签统计：主标签 + 是否检出重叠。"""
    k = binary.shape[1]
    pred, gt, ok2, has2, cov = [], [], 0, 0, 0
    for turn in turns:
        i0 = max(0, int(turn["start"] / frame_step_s))
        i1 = min(len(binary), int(turn["end"] / frame_step_s))
        if i1 <= i0:
            pred.append(-1); gt.append(turn["speaker"]); continue
        seg = binary[i0:i1]
        share = seg.mean(axis=0)
        order = np.argsort(-share)
        pred.append(int(order[0]))
        gt.append(turn["speaker"])
        if share[order[0]] > 0:
            cov += 1
        if len(order) > 1 and share[order[1]] > thresh:
            has2 += 1
    pair = defaultdict(float)
    for g, p in zip(gt, pred):
        pair[(g, p)] += 1.0
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gt, pred) if mapping.get(g) == p)
    return {"acc": ok / max(len(gt), 1), "multilabel_rate": has2 / max(len(gt), 1),
            "coverage": cov / max(len(gt), 1)}


def class_chunks(probs, frame_step_s, chunk_s=CHUNK_S, step_s=STEP_S,
                 min_run_s=0.30):
    """在 powerset 的"同一局部说话人"区间内切 chunk，而不是从 VAD 段盲滑。

    **为什么**：实测 2.0s 网格窗有 **90%** 跨说话人边界（1851 个里只有 194 个能标上
    唯一说话人），因为 VAD 段只认"有没有语音"、不认"是谁在说"，A 说完 B 接话就落进
    同一段。powerset 的 class 1/2/3 是"局部 speaker 1/2/3 单独说话"，实测类别纯度
    77–83%，按它切分能把"窗内只属于一个人"的比例从 10.5% 提到 52%。
    跨窗编号不连续没关系 —— 聚类本来就要解决跨窗对齐。

    区间先按 min_run_s（0.30s）全收，**不能**按 chunk_s 过滤：实测那样 R8007_M8010
    只剩 668 个 chunk、覆盖掉到 45%、准确率反降 4.7 点。短于 chunk_s 的区间
    **并入前一个**，否则它塞不下一个完整窗、只能整段丢弃。
    """
    argmax = probs.argmax(axis=1)
    single = np.isin(argmax, (1, 2, 3))
    raw, start, cur = [], None, None
    for i, (flag, c) in enumerate(zip(single, argmax)):
        now = int(c)
        if flag and (start is None or now != cur):
            if start is not None and (i - start) * frame_step_s >= min_run_s:
                raw.append([start * frame_step_s, i * frame_step_s])
            start, cur = i, now
        elif not flag and start is not None:
            if (i - start) * frame_step_s >= min_run_s:
                raw.append([start * frame_step_s, i * frame_step_s])
            start, cur = None, None
    if start is not None and (len(single) - start) * frame_step_s >= min_run_s:
        raw.append([start * frame_step_s, len(single) * frame_step_s])

    merged = []
    for a, b in raw:
        if merged and (b - a) < chunk_s:
            merged[-1][1] = b                  # 短区间并入前一个，保住覆盖
        else:
            merged.append([a, b])

    wins = []
    for a, b in merged:
        t = a
        while t + chunk_s <= b + 1e-9:
            wins.append((t, t + chunk_s))
            t += step_s
    return wins, len(merged)


def chunks_to_turns(windows, labels, turns):
    """chunk 标签 -> 句级主导投票 -> 全局映射 -> 句级准确率。"""
    gt, pred, durs, cover = [], [], [], 0
    for turn in turns:
        tally = defaultdict(float)
        for (a, b), lab in zip(windows, labels.tolist()):
            ov = min(b, turn["end"]) - max(a, turn["start"])
            if ov > 0:
                tally[lab] += ov
        if not tally:
            mid = (turn["start"] + turn["end"]) / 2.0
            near = min(range(len(windows)),
                       key=lambda i: abs((windows[i][0] + windows[i][1]) / 2.0 - mid))
            tally[labels[near]] = 1.0
        else:
            cover += 1
        gt.append(turn["speaker"])
        pred.append(max(tally, key=lambda x: tally[x]))
        durs.append(turn["end"] - turn["start"])
    pair = defaultdict(float)
    for g, p, d in zip(gt, pred, durs):
        pair[(g, p)] += d
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gt, pred) if mapping.get(g) == p)
    return {"acc": ok / max(len(gt), 1), "n_pred_speakers": len(set(pred)),
            "coverage": cover / max(len(gt), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*",
                    default=["R8007_M8010", "R8008_M8013", "R8009_M8020"])
    ap.add_argument("--out", default="three_d_speaker_step1.json")
    ap.add_argument("--arm", choices=["vad", "class"], default="class",
                    help="chunk 从哪来：vad=从 VAD 段盲滑（旧做法）；"
                         "class=powerset 同一局部说话人的区间内滑（新）")
    ap.add_argument("--post", action="store_true",
                    help="启用 OSD+匈牙利匹配后处理（实测 2/3 场有害，默认关；"
                         "只有需要重叠多标签时才开）")
    args = ap.parse_args()

    import onnxruntime as ort
    import claude_a2_v11 as v11
    models = load_models()
    osd_path = Path(r"D:\yinp\pyannote_seg3\model.onnx")
    osd_sess = (ort.InferenceSession(str(osd_path), providers=["CPUExecutionProvider"])
                if osd_path.exists() else None)
    rows = []
    print("%-14s %5s %6s | %7s %7s %6s | %9s %9s"
          % ("meeting", "真值K", "chunk", "自动K", "真值K", "覆盖",
             "自动K准确", "真值K准确"), flush=True)
    for meeting in args.meetings:
        t0 = time.time()
        wav = resolve_wav(meeting)
        audio, rate = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        vad = v11._vad_segs(models["vad"], str(wav))
        # OSD 只跑一次：既用来切 chunk 的类别区间，也喂给 post_process
        probs = None
        if osd_sess is not None:
            probs_all, mask_all, fs = osd_probs(osd_sess, audio)
            probs = probs_all[mask_all]
        wins_vad = chunk_windows(vad)
        wins_cls, n_runs = (class_chunks(probs, fs) if probs is not None
                            else (wins_vad, 0))
        arm = args.arm
        wins = wins_cls if arm == "class" else wins_vad
        E, kept = embed_chunks(models["spk"], audio, rate, wins)
        true_k = len({t["speaker"] for t in turns})

        labels_auto, k_auto, eigs = spectral_cluster(E, None)
        labels_true, _, _ = spectral_cluster(E, min(true_k, len(E) - 1))
        r_auto = chunks_to_turns(kept, labels_auto, turns)
        r_true = chunks_to_turns(kept, labels_true, turns)
        pa = pt = None
        if args.post and probs is not None:        # 复用上面那一次 OSD，不重复跑
            k_t = min(true_k, len(E) - 1)
            pa = turns_multilabel(
                post_process(labels_auto, kept, probs, k_auto, fs), turns, fs)
            pt = turns_multilabel(
                post_process(labels_true, kept, probs, k_t, fs), turns, fs)
        row = {"meeting": meeting, "true_k": true_k, "n_chunks": len(E),
               "k_auto": k_auto, "eigvals": [round(float(v), 4) for v in eigs[:8]],
               "auto": r_auto, "true_k_forced": r_true,
               "post_auto": pa, "post_true": pt,
               "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        print("[%s] %-14s %5d %6d | K %2d/%2d 覆盖%3.0f%% | 谱聚类 %5.1f%%/%5.1f%% | "
              "+OSD后处理 %5.1f%%/%5.1f%% (多标签 %s)"
              % (arm, meeting, true_k, len(E), k_auto, true_k,
                 r_auto["coverage"] * 100, r_auto["acc"] * 100, r_true["acc"] * 100,
                 (pa or {}).get("acc", 0) * 100, (pt or {}).get("acc", 0) * 100,
                 ("%.0f%%" % ((pt or {}).get("multilabel_rate", 0) * 100)) if pt else "-"),
              flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"chunk_s": CHUNK_S, "step_s": STEP_S,
                                   "knn_k": KNN_K, "max_k": MAX_K,
                                   "note": "谱聚类 k-NN 图 + eigengap 定 K；纯 scipy"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n对照：现状（2.0s 网格窗 + AHC）R8007_M8010 = 22.2%，R8008_M8013 = 38.1%，"
          "R8009_M8020 = 84.2%")
    if rows:
        import statistics as st
        BASE = {"R8001_M8004": 45.0, "R8003_M8001": 42.3, "R8007_M8010": 22.2,
                "R8007_M8011": 34.8, "R8008_M8013": 38.1, "R8009_M8018": 77.1,
                "R8009_M8019": 85.7, "R8009_M8020": 84.2}
        pairs = [(r["meeting"], r["auto"]["acc"] * 100, BASE[r["meeting"]])
                 for r in rows if r["meeting"] in BASE]
        if pairs:
            print("\n=== 汇总（%d 场，臂=%s，后处理=%s）==="
                  % (len(rows), args.arm, "开" if args.post else "关"))
            print("  %-14s %8s %8s %8s" % ("meeting", "本方案", "现状", "净变化"))
            for m, a, b in pairs:
                print("  %-14s %7.1f%% %7.1f%% %+7.1f" % (m, a, b, a - b))
            print("  %-14s %7.1f%% %7.1f%% %+7.1f"
                  % ("平均", st.mean([a for _, a, _ in pairs]),
                     st.mean([b for _, _, b in pairs]),
                     st.mean([a - b for _, a, b in pairs])))
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
