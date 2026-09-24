# -*- coding: utf-8 -*-
"""v13 真因判别：K 门是不是主因？—— 绕开选 K，在真值 K 上看聚类长什么样。

`probe_v12_k.py` 在 L_R003S01C02 上读到 k=2 的簇是 [421, 1]、k=4 是 [70, 230, 121, 1]。
一个 1 窗的孤立簇在 `_silhouette` 里把整个 K 打成 −1.0（任一簇 <2 就返回 −1.0），
所以"放宽 MIN_CLUSTER_SHARE"这条改动在这个形态上**一分不赚**——尺寸门和分数门同时挡着，
而且 `_select_candidate` 最后还有一道 `best["score"] > 0.0`。

真正要问的是下一个问题：**绕开所有门，在真值 K 上聚类，出来的簇是不是干净？**

* 干净 → 罪在选 K 判据，改判据（v13 的决策层）。
* 脏（大簇吞人、小簇碎成渣）→ 罪在特征空间本身，改判据没用，得换后端。

这个脚本**不经过任何门、任何惩罚**，直接拿真实路径的窗嵌入做一件最朴素的事：
AHC@真K，然后量每个簇相对参考的纯度、每个参考说话人有没有自己的窗。

判据（防止"全塞进一个簇"伪装成高分）：
  * `cluster_purity` 要**逐簇**看，不看均值——一个 90% 的簇 + 四个 5% 的碎簇，
    均值可能还行，但那是失败。
  * `speakers_with_home` 是"每个有效说话人是否拥有一个以它为主的簇"，这才是
    K 选对了能不能兑现的成分。缺一个，那个人的时长就全记在别人头上。

    python scripts/evaluation/probe_truek_purity.py --split dev --seconds 600
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

FRAME = 0.01


def window_reference(reference, times, window):
    """逐窗参考多数说话人 + 该窗的参考纯度（多数占窗内语音的比例）。

    纯度低的窗是跨边界窗——"窗本身脏"的直接读数，与聚类无关。
    按**帧**计票：跨边界窗里说得久的自然占多数。
    """
    duration = max(s["end"] for s in reference) + window
    n = int(duration / FRAME) + 1
    active = [[] for _ in range(n)]
    for segment in reference:
        a = max(0, int(round(segment["start"] / FRAME)))
        b = min(n, int(round(segment["end"] / FRAME)))
        for i in range(a, b):
            active[i].append(segment["speaker"])

    labels, purity = [], []
    for start in times:
        counts = Counter()
        for i in range(int(round(start / FRAME)),
                       min(n, int(round((start + window) / FRAME)))):
            counts.update(active[i])
        if not counts:
            labels.append("")
            purity.append(0.0)
            continue
        name, hits = counts.most_common(1)[0]
        labels.append(name)
        purity.append(hits / sum(counts.values()))
    return labels, purity


def cluster_rows(vectors, reference_labels, k, times):
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(vectors, method="average", metric="cosine")
    predicted = fcluster(tree, t=k, criterion="maxclust") - 1
    truth = np.asarray(reference_labels)

    rows = []
    for cluster in range(k):
        mask = predicted == cluster
        if not mask.any():
            rows.append({"cluster": cluster, "windows": 0, "purity": 0.0,
                         "majority": "", "speakers": {}})
            continue
        counts = Counter(name for name in truth[mask].tolist() if name)
        name, hits = counts.most_common(1)[0] if counts else ("", 0)
        rows.append({
            "cluster": cluster, "windows": int(mask.sum()),
            "majority": name, "purity": round(hits / int(mask.sum()), 4),
            "speakers": dict(counts.most_common(5)),
            "span_sec": [round(float(times[mask].min()), 1),
                         round(float(times[mask].max()), 1)],
        })
    # 每个参考说话人"家"在哪：它作为多数派的窗占它全部窗的比例。
    homes = {}
    for name in sorted(set(truth.tolist()) - {""}):
        mask = truth == name
        if not mask.any():
            continue
        cluster = int(Counter(predicted[mask].tolist()).most_common(1)[0][0])
        homes[name] = {
            "windows": int(mask.sum()),
            "home_cluster": cluster,
            "share_in_home": round(float((predicted[mask] == cluster).mean()), 4),
        }
    return predicted, rows, homes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--meeting")
    ap.add_argument("--split", choices=("dev", "holdout", "pilot"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--share", type=float, default=0.05,
                    help="有效说话人门槛：占参考语音时长的比例")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(args.root)
    if args.meeting:
        meetings = [args.meeting]
    elif args.split:
        meetings = split_meetings(root, args.split)
    else:
        raise SystemExit("要给 --meeting 或 --split")

    from web_server import get_models

    models = get_models(lambda message: None, need_stream=False, need_offline=False)

    report = []
    for meeting in meetings:
        wav = root / "wav" / f"{meeting}.flac"
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        seconds = args.seconds if args.seconds > 0 else max(s["end"] for s in reference)
        reference = [s for s in reference if s["start"] < seconds]

        # 与 v12.diarize() 内部逐字同参地取窗，不经过选 K。
        info = sf.info(wav)
        stop = -1 if seconds <= 0 else min(int(seconds * info.samplerate), info.frames)
        audio, rate = sf.read(wav, dtype="float32", stop=stop)
        if audio.ndim > 1:
            audio = audio[:, 0]
        vad_segments = v11._vad_segs(models["vad"], str(wav))
        vad_segments = [(start, min(end, seconds)) for start, end in vad_segments
                        if start < seconds]
        with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
            E, times, errors = v11._extract_windows(audio, rate, vad_segments,
                                                    models["spk"])
        times = np.asarray(times)

        result = v12.diarize(str(wav), models, max_seconds=args.seconds)

        labels, purity = window_reference(reference, times, v12.WIN_S)
        seconds_by_speaker = {}
        for segment in reference:
            name = segment["speaker"]
            seconds_by_speaker[name] = (seconds_by_speaker.get(name, 0.0)
                                        + segment["end"] - segment["start"])
        grand = sum(seconds_by_speaker.values()) or 1.0
        effective = sorted(name for name, value in seconds_by_speaker.items()
                           if value / grand >= args.share)
        in_pool = {name for name in labels if name}

        item = {
            "meeting": meeting, "seconds": seconds,
            "n_windows": int(len(E)), "embedding_errors": int(errors),
            "window_seconds": v12.WIN_S,
            "reference_speakers_all": len(seconds_by_speaker),
            "effective": effective, "true_k": len(effective),
            "effective_without_window": [n for n in effective if n not in in_pool],
            "window_purity_mean": round(float(np.mean(purity)), 4) if len(purity) else None,
            "window_purity_lt_50pct": int(sum(1 for value in purity if value < 0.5)),
            "v12_selected_k": int(result["k"]),
            "v12_refined": bool(result["refined"]),
            "v12_rescued": bool(result["rescued"]),
            "ahc_true_k": None, "cluster_purity": [], "speaker_homes": {},
            "min_cluster_windows": None,
        }

        if len(E) >= 2 and len(effective) >= 2:
            k = min(len(effective), len(E))
            _, rows, homes = cluster_rows(E, labels, k, times)
            item["ahc_true_k"] = k
            item["cluster_purity"] = rows
            item["speaker_homes"] = homes
            item["min_cluster_windows"] = min((row["windows"] for row in rows),
                                              default=0)
            # 纯度的诚实汇总：按窗加权，但另报**最差簇**——一个簇吞掉大半、
            # 其余碎成渣时均值会骗人，最差簇不会。
            item["worst_cluster_purity"] = round(min(row["purity"] for row in rows), 4)
            item["speakers_with_own_cluster"] = sum(
                1 for name in effective
                if homes.get(name, {}).get("share_in_home", 0.0) >= 0.5)

        report.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-truek-purity-v1", "meetings": report},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")

    print(f"\n{'会议':<16}{'有效':>4}{'真K':>5}{'选K':>5}{'窗数':>6}{'窗纯度':>8}"
          f"{'最差簇':>8}{'有家':>5}  缺窗说话人")
    for item in report:
        homes = item.get("speakers_with_own_cluster")
        worst = item.get("worst_cluster_purity")
        print(f"{item['meeting']:<16}{len(item['effective']):>4}{item['true_k']:>5}"
              f"{item['v12_selected_k']:>5}{item['n_windows']:>6}"
              f"{(item['window_purity_mean'] or 0) * 100:>7.1f}%"
              f"{(worst or 0) * 100:>7.1f}%"
              f"{str(homes) + '/' + str(item['true_k']):>6}  "
              f"{','.join(item['effective_without_window']) or '-'}")

    if report:
        ok = [item for item in report if item.get("ahc_true_k")]
        if ok:
            print(f"\n真K 下：平均最差簇纯度 "
                  f"{np.mean([i['worst_cluster_purity'] for i in ok]) * 100:.1f}%"
                  f"，平均有家 {np.mean([i['speakers_with_own_cluster'] for i in ok]):.2f}"
                  f"/{np.mean([i['true_k'] for i in ok]):.2f} 人")
            print(f"v12 选 K 命中 {sum(1 for i in ok if i['v12_selected_k'] == i['true_k'])}"
                  f"/{len(ok)} 场")


if __name__ == "__main__":
    main()
