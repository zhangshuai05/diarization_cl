# -*- coding: utf-8 -*-
"""泳道与旋钮：base / unlocked /（可选）oracle 三条选 K 路线，逐条 yield。

**不重写 T1 的流程**：`cache_path.segments_for` 已经把「选 K → k>=2 细化 / k==1 救援
→ 生产边界公式」串成一条与 `v12.diarize` 同口径的流水线；本模块只在调用期间替换
**选择器**（`v11` 的两个函数与 λ 常量），流水线本身一字不动：

* `base`     —— 生产选择器（`v11._candidate_clusters` + `v11._select_candidate`）。
  `pin_base=True` 时 λ 用**运行前的生产常量**（spec §6.1 的"钉住开关"：改旋钮动不了它）；
  解钉后 base 跟随 `lam` 旋钮。
* `unlocked` —— `unlocked_selector.unlocked(min_win, cap, mode)` 换掉选择器；λ 恒为旋钮值
  （`build_unlocked` 在调用时刻读 `v11.MULTI_K_PENALTY`）。与端到端对拍臂
  `probe_e2e_unlocked2.py` 同一手法、同一实现。
* `oracle`   —— 默认关；开启时 k=2..10 网格逐个强制成候选、走同一条流水线，取句级
  **双射**（`metrics.sentence_metrics(...)["overall"]`）峰值，平局取小 k（与
  `probe_oracle_k_curve.argmax_k` 同规）。**读真值**，页面据 `contract.TRUTH_DERIVED` 标徽标。

λ 用 `try/finally` 临时改 `v11.MULTI_K_PENALTY`，**无条件还原**（spec §4.3）。

payload 键 = `contract.LANE_PAYLOAD_KEYS`，不多不少；`lane` 恰为 `"base"` /
`"unlocked"` / `"oracle"`（T6 靠 `contract.TRUTH_DERIVED` 里的字面量 `oracle` 分支
渲染"读真值"徽标）。`merged_*` 两列由 `metrics.with_merged` 读真值回填；无真值场次
返回未合并读数且 DER 给 `None`——`der()` 在空参照上会把预测帧全记成 false alarm，
直接调会报出 >1 的假 DER。

T1 的 `k` **逐分支语义不同**（refined = 主窗选的 k，rescued = 去重后的簇数），这里
原样带出、不解释、不重算。
"""
import contextlib
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]

# λ / 选择器是模块级常量，交换只对单线程安全；T5 是 ThreadingHTTPServer，两个
# /api/run 并发会互相踩（oracle 一条要跑分钟级）。整条流水线在锁内算完再 yield，
# 锁不跨 yield。
_KNOB_LOCK = threading.Lock()

# 与 cache_path/metrics 同一套 `sys.path` 兜底：本模块可被 server 以 web/ 为根导入，
# 也可被测试以 backend 为根导入。
for _path in (_HERE, ROOT / "streaming_asr", ROOT / "scripts" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cache_path as CP  # noqa: E402
import contract  # noqa: E402
import metrics  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from unlocked_selector import unlocked  # noqa: E402

# 旋钮默认值（spec §4.3；`min_win`/`cap` 与 unlocked_selector 的 CAP=10/MIN_WIN=2 同值，
# `lam` = v11.MULTI_K_PENALTY 的生产值 0.005）。用户给的 knobs 按名覆盖。
KNOB_DEFAULTS = {
    "lam": 0.005, "min_win": 2, "cap": 10, "mode": "objective",
    "refine": True, "oracle": False, "pin_base": True,
}

# oracle 的 k 网格：与生产 cap=10 对齐（probe_oracle_k_curve.K_GRID 同一条）。
K_GRID = tuple(range(2, 11))

DER_KEYS = ("der", "miss_rate", "false_alarm_rate", "confusion_rate")

# 50 场缓存里 20 场在 test 划分、30 场在 train_L；按 TextGrid 是否存在定根，
# 不把 cohort 判断写死成会议名的日期前缀。
DATA_ROOTS = (P.DATA, P.DATA.parent / "train_L")


def load_truth(meeting):
    """`(sentences, reference)`：句子表与真值 rttm 段表。

    缺 TextGrid 或 rttm 时对应给空——「无门限参照」是合法场次（50 场里 38 场如此），
    不是错误；真值文件存在但读不动时照样抛。句子口径 = `probe_ab_dev.load_sentences`
    （同两个 P 函数），真值段与 `probe_e2e_unlocked2` 同口径：只留 600 s 前、末端截到 600 s。
    """
    for root in DATA_ROOTS:
        path = root / "TextGrid" / f"{meeting}.TextGrid"
        if not path.is_file():
            continue
        tiers = P.read_textgrid(path)
        sentences = P.annotate_overlap(P.sentences_from_textgrid(tiers, P.SECONDS), tiers)
        rttm = path.with_suffix(".rttm")
        reference = []
        if rttm.is_file():
            reference = [{"start": s["start"], "end": min(s["end"], P.SECONDS),
                          "speaker": s["speaker"]}
                         for s in read_rttm(rttm) if s["start"] < P.SECONDS]
        return sentences, reference
    return [], []


@contextlib.contextmanager
def _lam(value):
    """调用期间把 λ 常量换成旋钮值；退出**无条件**还原（spec §4.3）。"""
    saved = v11.MULTI_K_PENALTY
    v11.MULTI_K_PENALTY = float(value)
    try:
        yield
    finally:
        v11.MULTI_K_PENALTY = saved


@contextlib.contextmanager
def _refine(enabled):
    """`refine=False` ⇒ `v12._refine_labels` 返回 None；T1 看到 None 就留在主窗。"""
    if enabled:
        yield
        return
    original = v12._refine_labels
    v12._refine_labels = lambda *args, **kwargs: None
    try:
        yield
    finally:
        v12._refine_labels = original


@contextlib.contextmanager
def _force_k(k):
    """oracle 用：候选表压成「只有 k」一项（标签由 `P.clusters_for` 给），其余交 T1。"""
    def candidates(E, max_speakers=None):
        labels = P.clusters_for(E, k)
        sizes = [int((labels == label).sum()) for label in range(int(k))]
        return [{"k": int(k), "labels": labels, "score": 1.0, "objective": 1.0,
                 "sizes": sizes, "eligible": True}], None

    def select(cands, forced_k=None):
        return cands[0]

    saved = (v11._candidate_clusters, v11._select_candidate)
    v11._candidate_clusters, v11._select_candidate = candidates, select
    try:
        yield
    finally:
        v11._candidate_clusters, v11._select_candidate = saved


def _lane_payload(lane, meeting, base, sentences, reference):
    """T1 的段表 payload + 指标 → 页面契约的扁平行（键 = `LANE_PAYLOAD_KEYS`）。"""
    scores = metrics.sentence_metrics(sentences, base["segments"], base["level"])
    if reference:
        der = metrics.der_metrics(reference, base["segments"])
        merged = metrics.with_merged(meeting, base, sentences, reference)
    else:
        der = dict.fromkeys(DER_KEYS, None)
        # 无可消重（没真值参照）：merged 列 = 未合并读数，DER 不给数字。
        merged = {"merged_overall": scores["overall"], "merged_der": None}
    fields = {**scores, **merged, **der,
              "lane": lane, "k": int(base["k"]),
              "segments": [dict(segment) for segment in base["segments"]]}
    return {key: fields[key] for key in contract.LANE_PAYLOAD_KEYS}


def _oracle_lane(meeting, knob, sentences):
    """k=2..10 网格里双射峰值的那份 T1 payload（平局取小 k）。"""
    E, _times = CP._cached_stage(meeting, "main")
    grid = [k for k in K_GRID if k <= len(E)] or [K_GRID[0]]
    best_payload = None
    best_score = None
    for k in grid:
        with _KNOB_LOCK:
            with _refine(knob["refine"]):
                with _force_k(k):
                    payload = CP.segments_for(meeting)
        score = metrics.sentence_metrics(sentences, payload["segments"],
                                         payload["level"])["overall"]
        if score is None:                      # 没有句子：无可比，按 -1 参与（峰值退化为小 k）
            score = -1.0
        if best_score is None or score > best_score:
            best_payload, best_score = payload, score
    return best_payload


def run_lanes(meeting, knobs=None):
    """依次产出 base / unlocked /（`knobs["oracle"]` 时）oracle 三条泳道的 payload。

    每条 yield 一次；生成器被提前关闭时 `finally` 照样还原 λ。
    """
    knob = {**KNOB_DEFAULTS, **(knobs or {})}
    sentences, reference = load_truth(meeting)
    saved = v11.MULTI_K_PENALTY
    try:
        # base：钉住时用运行前的生产常量——还原点是什么，钉住的就该是什么
        # （不写死 KNOB_DEFAULTS["lam"]）。
        base_lam = saved if knob["pin_base"] else float(knob["lam"])
        with _KNOB_LOCK:
            with _refine(knob["refine"]):
                with _lam(base_lam):
                    base = CP.segments_for(meeting)
        yield _lane_payload("base", meeting, base, sentences, reference)

        with _KNOB_LOCK:
            with _refine(knob["refine"]):
                with _lam(float(knob["lam"])):
                    with unlocked(min_win=knob["min_win"], cap=knob["cap"], mode=knob["mode"]):
                        unlocked_payload = CP.segments_for(meeting)
        yield _lane_payload("unlocked", meeting, unlocked_payload, sentences, reference)

        if knob["oracle"]:
            oracle = _oracle_lane(meeting, knob, sentences)
            yield _lane_payload("oracle", meeting, oracle, sentences, reference)
    finally:
        v11.MULTI_K_PENALTY = saved
