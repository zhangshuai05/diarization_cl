# -*- coding: utf-8 -*-
"""泳道指标：句级四类 + clean/overlap 分桶 + DER + merged（消重复簇）复算。

**规则一处都不重写**，全部复用既有实现：

* 句级四类与分桶 —— `probe_sentence_attribution.attribution`（本项目主指标的定义处）；
  段表 → 句级预测的映射仍走它的 `segment_map` / `sentence_dict_predictions`
  （整数簇号 `0` 是 falsy 的陷阱只在那两处处理，这里不做二次判断）。
* DER —— `der.der(reference, predicted, collar=0.25)`（帧级 union 分母，md-eval 同口径）。
* merged —— `probe_over_seg_gain` 的 `truth_coverage` / `classify_dup` / `merge_dups`
  + `probe_gap_decomp.window_reference`，与 `probe_oracle_k_curve` 同一 recipe；
  合并后**重走 T1 的生产段表**（`cache_path.segments_for`）再打分，
  不是只在标签上合并（口径见 spec §4.2）。

返回键是 `contract.LANE_PAYLOAD_KEYS` 的子集，不多不少：
`sentence_metrics` 出 level/overall/clean/overlap/abstain/unmapped/wrong，
`der_metrics` 出 der/miss_rate/false_alarm_rate/confusion_rate，
`with_merged` 出 merged_overall/merged_der。

**比率一律 0–1 小数**（页面再乘 100）。四个顶格比率由**精确计数**导出：

    overall = (n − 弃权 − 未映射 − 错答) / n      ← 三类互斥完备，残差就是命中数
    wrong   = n_wrong / n

两者都不是「1 − 对 − 弃权 − 未映射」那种减法链：`wrong` 若这么算，`attribution`
三处 round 的取整误差会全落到错答头上（本仓库踩过，见 `probe_oracle_k_curve.leg`）。
`clean` / `overlap` 两桶取自 `attribution` 的 4 位取整值（只用于展示，不参与恒等式）。

分母为 0 时（一场没有句子）比率返回 `None`，与探针 `_empty()` 的惯例一致——
T1 在窗数 < 2 时会返回自建 payload，这条路径必须能走通、不抛异常。
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
# 本模块可被 `import metrics`（server 以 web/ 为根起）或 `backend.metrics` 两种方式导入，
# 所以把自己的目录也放进来；其余两条路径与 cache_path.py 的兜底同一写法。
for _path in (_HERE, ROOT / "streaming_asr", ROOT / "scripts" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cache_path as CP  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
from der import der  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_over_seg_gain import classify_dup, merge_dups, truth_coverage  # noqa: E402

# `probe_oracle_k_curve` 的 merged 用法（`rows = classify_dup(truth_coverage(labels, names))`
# → `merge_dups(labels, rows)`）即由上面三个函数 + `window_reference` 组成，与它逐行同源。
# 直接从这里 import 而不 import `probe_oracle_k_curve`：后者是**未入库**的探针脚本
# （`git status` 里仍是 `??`），web 后端不能把依赖压在它身上。


def _segment_map(sentences, segments):
    """生产段表（`[{"start","end","spk","margin"}]`）→ `segment_map` 的三元组。

    名字归一化（`None` / `""` 丢弃、整数簇号 `0` 不当假值）与按起点排序都在
    `segment_map` 里，这里只做键名搬运。
    """
    end = max([s["end"] for s in sentences] + [0.0])
    return P.segment_map([s["start"] for s in segments],
                         [s["spk"] for s in segments],
                         [s["end"] - s["start"] for s in segments], end)


def _as_rttm(segments):
    """段表 → `der()` 的输入（统一成 rttm 的 `speaker` 键）。

    `reference` 侧来自 `der.read_rttm`（`speaker`），预测侧是 T1 段表（`spk`），
    两种键名都接受：DER 与句级共用同一份段表，转换只做一次、不复制段边界公式。
    """
    out = []
    for segment in segments:
        name = segment["spk"] if "spk" in segment else segment["speaker"]
        out.append({"start": float(segment["start"]), "end": float(segment["end"]),
                    "speaker": str(name)})
    return out


def sentence_metrics(sentences, segments, level):
    """句级四类 + clean/overlap 分桶（`segments` = T1 的生产段表）。

    `level` 原样带出（`refined` / `2s-only` / `rescued`），是页面上"这条读数是哪套窗"的标记。
    """
    segs = _segment_map(sentences, segments or [])
    attribution = P.attribution(sentences, P.sentence_dict_predictions(sentences, segs))
    n = attribution["n_sentences_scored"]
    if not n:
        rates = {"overall": None, "abstain": None, "unmapped": None, "wrong": None}
    else:
        abstain, unmapped, wrong = (attribution["n_abstain"], attribution["n_unmapped"],
                                    attribution["n_wrong"])
        rates = {
            # 残差 = 命中数（三类互斥完备），精确到整数；不是 attribution 的 4 位取整值。
            "overall": (n - abstain - unmapped - wrong) / n,
            "abstain": abstain / n,
            "unmapped": unmapped / n,
            # 精确计数，**不要**写成 1 − 对 − 弃权 − 未映射。
            "wrong": wrong / n,
        }
    return {
        "level": level,
        "overall": rates["overall"],
        "clean": attribution["clean"]["per_sentence"],
        "overlap": attribution["overlap"]["per_sentence"],
        "abstain": rates["abstain"],
        "unmapped": rates["unmapped"],
        "wrong": rates["wrong"],
    }


def der_metrics(reference, segments):
    """帧级 DER 与 miss/FA/confusion 分解（collar=0.25 s，与对拍靶子同口径）。"""
    result = der(reference, _as_rttm(segments), collar=0.25)
    return {key: result[key]
            for key in ("der", "miss_rate", "false_alarm_rate", "confusion_rate")}


def with_merged(meeting, payload, sentences, reference):
    """消重复簇后**重走生产段表**再打分，返回 `{"merged_overall", "merged_der"}`。

    `payload` = `cache_path.segments_for(...)` 的返回值（要 `labels` / `times` / `level`）；
    `sentences` = `probe_ab_dev.load_sentences` 的句表；`reference` = `der.read_rttm` 的 rttm 段。

    无可消重时（k < 2 / 空标签 / 无真值参照）**原样返回未合并读数**：不读缓存、不抛异常。
    T1 在窗数 < 2 时返回自建 payload（空 `labels`），窗数少的场次也没有 dup 可并。
    """
    labels = np.asarray(payload.get("labels", []), dtype=int)
    if labels.size == 0 or len(set(int(x) for x in labels)) < 2 or not reference:
        merged = payload
    else:
        times = np.asarray(payload["times"], dtype=float)
        names = window_reference(reference, times, P.WIN)
        rows = classify_dup(truth_coverage(labels, names))
        merged_labels = merge_dups(labels, rows)
        if len(set(int(x) for x in merged_labels)) == len(set(int(x) for x in labels)):
            merged = payload          # 没有 dup：重算会得到同一张段表，直接跳过
        else:
            merged = CP.segments_for(meeting, labels=merged_labels, level=payload["level"])
    return {
        "merged_overall": sentence_metrics(sentences, merged["segments"],
                                           merged["level"])["overall"],
        "merged_der": der_metrics(reference, merged["segments"])["der"],
    }
