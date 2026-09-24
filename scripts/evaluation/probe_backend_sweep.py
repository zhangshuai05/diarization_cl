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
  训练池             —— 默认用**全部 20 场**（除目标场）拟合，是对后端最有利的设定：
                        若这样都不赚，后端方向可以弃了。

零推理。

    python scripts/evaluation/probe_backend_sweep.py --split dev --seconds 600
"""
import argparse
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
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_pair_oracle import Transform, lda_fit, pure_windows  # noqa: E402


def load(cache, meeting, window, hop):
    data = np.load(Path(cache) / meeting / f"w{window:g}_h{hop:g}.npz",
                   allow_pickle=False)
    return (data["emb"], np.asarray(data["times"], dtype=float),
            data["speakers"].astype(str), np.asarray(data["votes"], dtype=float))


def fit_backend(pool, target, min_class_windows, normalize):
    """用除目标场外的全部纯窗拟合 LDA，类 = (身份组, 说话人)。"""
    Xs, ys = [], []
    for meeting, (emb, times, speakers, votes) in pool.items():
        if meeting == target:
            continue
        pe, _, ps = pure_windows(emb, times, speakers, votes)
        if len(pe):
            Xs.append(pe)
            ys.append(np.array([f"{meeting_group(meeting)}:{x}" for x in ps]))
    if not Xs:
        return None, 0
    X, y = np.concatenate(Xs), np.concatenate(ys)
    kept = sum(1 for c in np.unique(y) if (y == c).sum() >= min_class_windows)
    if kept < 2:
        return None, 0
    fit = lda_fit(X, y, min_class_windows=min_class_windows)
    if fit is None:
        return None, kept
    return Transform(fit.mean, fit.weights, normalize=normalize), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=v12.WIN_S)
    ap.add_argument("--hop", type=float, default=v12.HOP_S)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--classes", default="2,8,16,32")
    ap.add_argument("--norm", default="euclid,cosine")
    ap.add_argument("--output")
    args = ap.parse_args()

    class_limits = [int(x) for x in args.classes.split(",")]
    norms = [x == "cosine" for x in args.norm.split(",")]

    root = Path(args.root)
    meetings = split_meetings(root, args.split)
    every = sorted(set(split_meetings(root, "dev") + split_meetings(root, "holdout")))
    pool = {}
    for meeting in every:
        try:
            pool[meeting] = load(args.cache, meeting, args.window, args.hop)
        except FileNotFoundError:
            continue

    variants = ["none"] + [f"c{c}_{'cos' if n else 'euc'}"
                           for c in class_limits for n in norms]
    rows = []
    for meeting in meetings:
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
        win = np.full(len(times), args.window)
        if len(emb) < 2 or len(effective) < 2:
            continue
        k = min(len(effective), len(emb))
        row = {"meeting": meeting, "true_k": len(effective),
               "n_windows": int(len(emb)), "variants": {}}
        for name in variants:
            if name == "none":
                vectors, metric, kept = emb, "cosine", None
            else:
                limit = int(name[1:name.index("_")])
                normalize = name.endswith("cos")
                fit, kept = fit_backend(pool, meeting, limit, normalize)
                if fit is None:
                    row["variants"][name] = {"accuracy": None}
                    continue
                vectors, metric = fit(emb), fit.metric
            clusters = cluster_labels(vectors, k, metric=metric)
            named = map_by_frames(reference, times, clusters, win)
            acc = frame_accuracy(reference, times, named, win)
            row["variants"][name] = {"accuracy": acc["accuracy"], "classes": kept}
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
