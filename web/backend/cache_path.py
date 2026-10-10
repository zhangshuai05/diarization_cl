# -*- coding: utf-8 -*-
"""缓存嵌入 → 生产同一份段表（不跑模型）。

只做两件事：读探针缓存的三阶段窗嵌入（`probe_path_equiv.cached_stage`），
把它们喂进**生产函数**——`v11._candidate_clusters` / `_select_candidate` / `_centroids`、
`v12._refine_labels` / `_segment_windows`。段边界公式只有生产那一份，
本模块**不重写**，所以这里的段表与 `v12.diarize()` 同口径、DER 可比。

探针链（`probe_ab_dev.real_path_labels` + `probe_sentence_attribution.segment_map`）
不行：它按"窗起点 + 窗长"平铺、且 `k<2` 不救援（见 `probe_path_equiv` docstring 的
差异 1、2），段级几乎 100% 不同，只能对句级。

三阶段缓存 = 生产的三套窗（`build_probe_cache.py`）：

    main   2.0 / 1.0    `v11._extract_windows` @2.0
    refine 0.6 / 0.15   `v12._refine_windows`
    rescue 0.6 / 0.15   `v11._extract_windows` @0.6（`k==1` 救援）

`sys.path` 兜底与 `probe_path_equiv` 同一写法：本模块按路径直接引生产模块，
不依赖调用方摆好路径（T5 的 `server.py` 也直接吃这一份）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "streaming_asr", ROOT / "scripts" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
import probe_path_equiv as PE  # noqa: E402

STAGES = ("main", "refine", "rescue")

# level ↔ 生产里对应的那套窗：决定读哪个 npz、用哪个 hop 分段。
LEVEL_STAGES = {
    "2s-only": ("main", v12.HOP_S),
    "refined": ("refine", v12.REFINE_HOP_S),
    "rescued": ("rescue", v12.REFINE_HOP_S),
}


def _cached_stage(meeting, stage):
    """单阶段读取。路径不存在就**显式**抛 `FileNotFoundError`（消息含缺失路径），
    不赌 `np.load` 的报错文案。
    """
    path = PE.PROBE_CACHE / meeting / f"{stage}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"probe cache missing: {path}")
    return PE.cached_stage(meeting, stage)


def cached_meeting(meeting):
    """读三阶段窗嵌入 → `{"main"|"refine"|"rescue": {"emb", "times"}}`。

    缺 npz（含缺整个阶段）→ `FileNotFoundError`，消息含缺失文件路径；
    **不静默返回空结果**。
    """
    out = {}
    for stage in STAGES:
        emb, times = _cached_stage(meeting, stage)
        out[stage] = {"emb": emb, "times": times}
    return out


def _segments_from(times, labels, margins, hop_s):
    """直通生产的 `v12._segment_windows`——边界公式只有那一份。"""
    return [dict(segment) for segment in v12._segment_windows(times, labels, margins, hop_s)]


def _margins(E, labels):
    """与 `v12.diarize` 同式：相对所选簇质心的余弦 margin（单簇时全 0）。

    生产把这几行写在 `diarize` 体内，没有可调用的函数；数值口径必须与它逐位一致，
    所以这里照抄（见 README「复制来源」）。
    """
    centroids = v11._centroids(E, labels)
    scores = E @ centroids.T
    if scores.shape[1] < 2:
        return np.zeros(len(E))
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def rescue_labels(meeting):
    """生产 `k==1` 救援分支的等价物：救援窗嵌入重跑候选。

    分出多人 → `(labels, times)`；仍是单人或窗太少 → `None`（生产保持单人）。
    """
    E, times = _cached_stage(meeting, "rescue")
    if len(E) < 2:
        return None
    # max_speakers 必须显式传 v12 的值：v11 里它是默认参数，def 时就绑成 v11.MAX_SPEAKERS（4）。
    candidates, _ = v11._candidate_clusters(E, max_speakers=v12.MAX_SPEAKERS)
    selected = v11._select_candidate(candidates)
    if selected["k"] <= 1:
        return None
    return selected["labels"].copy(), times


def _payload(times, labels, margins, hop_s, level, k):
    return {
        "segments": _segments_from(times, labels, margins, hop_s),
        "level": level,
        "times": times,
        "labels": labels,
        "margins": margins,
        "k": int(k),
    }


def segments_for(meeting, labels=None, level=None):
    """会议 → 生产口径段表。

    `labels=None`：跑完整流程（主窗选 K → `k>=2` 细化边界 / `k==1` 救援），
    与 `v12.diarize()` 的分离部分逐分支对齐。

    给了 `labels`：只用它们重新分段，`level` 指出这批标签属于哪个阶段的窗
    （泳道换标签 / 手动改标签后复用同一条边界公式）。`level` 省略按 `"2s-only"` 算。
    """
    if labels is not None:
        level = level or "2s-only"
        stage, hop_s = _level_stage(level)
        E, times = _cached_stage(meeting, stage)
        labels = np.asarray(labels, dtype=int)
        if len(labels) != len(E):
            raise ValueError(f"labels 有 {len(labels)} 个，{stage} 阶段有 {len(E)} 个窗")
        return _payload(times, labels, _margins(E, labels), hop_s, level, len(set(labels.tolist())))

    E, times = _cached_stage(meeting, "main")
    if len(E) < 2:
        # 与 diarize 的 insufficient 分支同口径：没有候选可选。
        return {"segments": [], "level": "2s-only", "times": times,
                "labels": np.asarray([], dtype=int), "margins": np.asarray([]), "k": 0}

    candidates, _ = v11._candidate_clusters(E, max_speakers=v12.MAX_SPEAKERS)
    selected = v11._select_candidate(candidates)
    labels = selected["labels"].copy()
    k = int(selected["k"])
    level = "2s-only"

    # 长窗已定好身份（k 与质心），短窗只负责找换人边界；身份漂移不了。
    if selected["k"] >= 2:
        anchors = v11._centroids(E, labels)
        anchor_labels = sorted(set(int(value) for value in labels))
        rE, rtimes = _cached_stage(meeting, "refine")
        rlabels = v12._refine_labels(rE, anchors, anchor_labels, rtimes)
        if rlabels is not None and len(set(rlabels.tolist())) >= 2:
            E, times, labels = rE, rtimes, rlabels
            level = "refined"

    # 长窗判成单人时用密窗复核：分出更多人就**整体替换** E/times/labels（生产同款）。
    elif selected["k"] == 1:
        rescued = rescue_labels(meeting)
        if rescued is not None:
            labels, times = rescued
            E = _cached_stage(meeting, "rescue")[0]  # 只为了按救援窗重算 margin
            k = int(len(set(labels.tolist())))
            level = "rescued"

    return _payload(times, labels, _margins(E, labels), LEVEL_STAGES[level][1], level, k)


def _level_stage(level):
    if level not in LEVEL_STAGES:
        raise ValueError(f"unknown level {level!r}（可选 {sorted(LEVEL_STAGES)}）")
    return LEVEL_STAGES[level]
