# -*- coding: utf-8 -*-
"""`contract` + `metrics` 的指标口径与契约测试。

不碰 D:\\ 真缓存：段表用合成 npz（tmp_path），句表/真值用内存构造。
真实数据上的口径由 T4 的 DER 对拍与 T9 的实测兜底。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import cache_path  # noqa: E402
import contract  # noqa: E402
import metrics  # noqa: E402


def sent(start, end, speaker, overlapped=False):
    return {"start": start, "end": end, "speaker": speaker, "overlapped": overlapped}


def seg(start, end, spk):
    """T1/生产口径的段表条目（`spk` 是簇号，`margin` 只作占位）。"""
    return {"start": start, "end": end, "spk": spk, "margin": 1.0}


# ───────────────────── 句级四类 + 分桶 ─────────────────────

# 5 句、2 个真值人、3 个预测簇 ⇒ 匈牙利只能配 2 对，第 3 个簇必然未映射。
# 权重（句时长）让最优置换唯一：0→A(2.0s) / 1→B(2.0s)，簇 2 落空。
FOUR_SENTS = [
    sent(0.0, 2.0, "A"),          # pred 0 → 映射到 A → 对
    sent(3.0, 5.0, "B", overlapped=True),   # pred 1 → 映射到 B → 对（也让 overlap 桶非空）
    sent(6.0, 6.5, "A"),          # pred 2 → 未映射
    sent(7.0, 7.5, "B"),          # pred 0 → 映射到 A，与真值不符 → 错答
    sent(8.0, 9.0, "A"),          # 无任何段覆盖 → 弃权
]
FOUR_SEGS = [seg(0.0, 2.0, 0), seg(3.0, 5.0, 1), seg(6.0, 6.5, 2), seg(7.0, 7.5, 0)]

REF = [{"start": 0.0, "end": 10.0, "speaker": "A"}]


def test_four_buckets_sum_to_one():
    m = metrics.sentence_metrics(FOUR_SENTS, FOUR_SEGS, "2s-only")
    # 对 2/5、弃权 1/5、未映射 1/5、错答 1/5
    assert m["overall"] == pytest.approx(2 / 5, abs=1e-12)
    assert m["abstain"] == pytest.approx(1 / 5, abs=1e-12)
    assert m["unmapped"] == pytest.approx(1 / 5, abs=1e-12)
    assert m["wrong"] == pytest.approx(1 / 5, abs=1e-12)
    assert abs(m["overall"] + m["abstain"] + m["unmapped"] + m["wrong"] - 1.0) < 1e-9


def test_wrong_is_exact_count_not_rounded_residual():
    # 3 句：对 1、错答 1、弃权 1 ⇒ wrong 精确 = 1/3。写成 1-对-弃权-未映射 只会得到
    # 4 位取整后的 0.3333（本仓库踩过的坑），所以这里要求逐位等于 1/3。
    sents = [sent(0.0, 2.0, "A"), sent(3.0, 4.0, "B"), sent(5.0, 6.0, "A")]
    segs = [seg(0.0, 2.0, 0), seg(3.0, 4.0, 0)]
    m = metrics.sentence_metrics(sents, segs, "2s-only")
    assert m["wrong"] == 1 / 3
    assert m["unmapped"] == 0.0
    assert m["abstain"] == 1 / 3


def test_clean_and_overlap_buckets_split_sentences():
    # 无重叠句对、重叠句错 ⇒ clean=1.0 / overlap=0.0（两桶各自内部按句平均）
    sents = [sent(0.0, 3.0, "A", overlapped=False),
             sent(4.0, 5.0, "B", overlapped=True)]
    segs = [seg(0.0, 3.0, 0), seg(4.0, 5.0, 0)]   # 都判成簇 0，匈牙利配 0→A
    m = metrics.sentence_metrics(sents, segs, "2s-only")
    assert m["clean"] == 1.0
    assert m["overlap"] == 0.0
    assert m["overall"] == pytest.approx(0.5, abs=1e-12)


def test_empty_segments_all_abstain():
    m = metrics.sentence_metrics(FOUR_SENTS, [], "2s-only")
    assert (m["overall"], m["abstain"], m["unmapped"], m["wrong"]) == (0.0, 1.0, 0.0, 0.0)


def test_empty_sentences_yield_none_rates_not_exception():
    # T1 在窗数 < 2 时会返回自建 payload（空段表）；没有任何句子时不能抛异常，
    # 比率字段按探针惯例返回 None（分母为 0）。
    m = metrics.sentence_metrics([], [], "2s-only")
    assert m["level"] == "2s-only"
    for key in ("overall", "clean", "overlap", "abstain", "unmapped", "wrong"):
        assert m[key] is None


def test_all_rates_are_fractions():
    m = metrics.sentence_metrics(FOUR_SENTS, FOUR_SEGS, "2s-only")
    for key in ("overall", "clean", "overlap", "abstain", "unmapped", "wrong"):
        assert 0.0 <= m[key] <= 1.0


def test_empty_bucket_is_none_not_zero():
    # 全是重叠句 ⇒ clean 桶为空：探针惯例返回 None（页面显示“—”），不是 0.0
    sents = [sent(0.0, 1.0, "A", overlapped=True), sent(2.0, 3.0, "B", overlapped=True)]
    segs = [seg(0.0, 1.0, 0), seg(2.0, 3.0, 1)]
    m = metrics.sentence_metrics(sents, segs, "2s-only")
    assert m["clean"] is None
    assert m["overlap"] == 1.0


def test_values_match_probe_attribution_without_rounding():
    """同一口径：我们的精确值 = 探针 `attribution` 的 4 位取整值（差只来自取整）。"""
    import probe_sentence_attribution as P

    pred = P.sentence_dict_predictions(FOUR_SENTS,
                                       metrics._segment_map(FOUR_SENTS, FOUR_SEGS))
    probe = P.attribution(FOUR_SENTS, pred)
    m = metrics.sentence_metrics(FOUR_SENTS, FOUR_SEGS, "2s-only")
    assert m["overall"] == pytest.approx(probe["overall"]["per_sentence"], abs=5e-5)
    assert m["wrong"] == pytest.approx(probe["n_wrong"] / probe["overall"]["n"], abs=5e-5)


# ───────────────────── DER ─────────────────────

def test_der_zero_when_prediction_equals_reference():
    assert metrics.der_metrics(REF, REF)["der"] == 0.0


def test_der_accepts_production_segments_with_spk():
    assert metrics.der_metrics(REF, [seg(0.0, 10.0, 0)])["der"] == 0.0


def test_der_uses_collar_025():
    # 预测比参考短 0.2s（< collar）⇒ 缺口落在边界 collar 内，不进 miss。
    # 若 collar 写成 0（`der()` 的默认值），miss_rate 会是 0.02。
    out = metrics.der_metrics(REF, [seg(0.0, 9.8, 0)])
    assert out["der"] == 0.0
    assert out["miss_rate"] == 0.0


def test_der_charges_confusion_when_second_half_is_another_cluster():
    out = metrics.der_metrics(REF, [seg(0.0, 5.0, 0), seg(5.0, 10.0, 1)])
    assert 0.45 < out["der"] < 0.55
    assert out["miss_rate"] == 0.0
    assert out["false_alarm_rate"] == 0.0
    assert out["confusion_rate"] > 0.0
    assert abs(out["der"] - (out["miss_rate"] + out["false_alarm_rate"]
                             + out["confusion_rate"])) < 1e-3


# ───────────────────── 契约 ─────────────────────

REQUIRED_LANE_KEYS = {
    "lane", "k", "level", "overall", "clean", "overlap", "abstain", "unmapped",
    "wrong", "merged_overall", "der", "miss_rate", "false_alarm_rate",
    "confusion_rate", "segments",
}


def test_contract_covers_required_fields_without_duplicates():
    assert REQUIRED_LANE_KEYS <= set(contract.LANE_PAYLOAD_KEYS)
    assert len(contract.LANE_PAYLOAD_KEYS) == len(set(contract.LANE_PAYLOAD_KEYS))


def test_truth_derived_marks_oracle_lane_and_merged_fields():
    """`TRUTH_DERIVED` = 页面必须标「读真值」徽标的名单（contract 是唯一来源）。

    ① 恰好三项，不多不少不重复；
    ② 三项都暴露给页面：`oracle` 走泳道名（`payload["lane"]` 的取值，由 `lane` 键承载），
       `merged_overall` / `merged_der` 走字段名（直接在 `LANE_PAYLOAD_KEYS` 里）。
    """
    assert set(contract.TRUTH_DERIVED) == {"oracle", "merged_overall", "merged_der"}
    assert len(contract.TRUTH_DERIVED) == 3                     # 无重复

    keys = set(contract.LANE_PAYLOAD_KEYS)
    assert set(contract.TRUTH_DERIVED) & keys == {"merged_overall", "merged_der"}
    # `oracle` 是泳道名而非字段名：它经 `lane` 这个键暴露，别混进键名单本身
    assert set(contract.TRUTH_DERIVED) - keys == {"oracle"}
    assert "oracle" not in keys
    assert "lane" in keys


def test_returned_keys_are_in_contract():
    keys = set(contract.LANE_PAYLOAD_KEYS)
    payload = {"level": "2s-only", "k": 0, "times": np.array([1.0]),
               "labels": np.array([], dtype=int), "margins": np.array([]), "segments": []}
    assert set(metrics.sentence_metrics(FOUR_SENTS, FOUR_SEGS, "2s-only")) <= keys
    assert set(metrics.der_metrics(REF, FOUR_SEGS)) <= keys
    assert set(metrics.with_merged("M", payload, FOUR_SENTS, REF)) <= keys


# ───────────────────── merged（消重复簇）─────────────────────

MEETING = "SYNTH"


def _main_stage(root, emb, times):
    target = root / MEETING
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "main.npz", emb=np.asarray(emb, dtype=np.float32), times=times)


def _use_synth_cache(monkeypatch, tmp_path, n=12):
    monkeypatch.setattr(cache_path.PE, "PROBE_CACHE", tmp_path)
    _main_stage(tmp_path, np.zeros((n, 3)), np.arange(n, dtype=float))


def test_with_merged_pulls_duplicate_clusters_back(monkeypatch, tmp_path):
    """过分割造的假收益：簇 2 是簇 1 的重复副本（都归 B）。

    基线：句级双射只能把 1 个簇配给 B，另一个簇的句子落进「未映射」⇒ 2/3；
    消重后两个 B 句都归到主簇 ⇒ 1.0。DER 同理从 ~0.18 归零。
    （真值边界选在 4.0s，与生产段表在窗 3/4 之间的边界重合，DER 才能严格归零。）
    """
    _use_synth_cache(monkeypatch, tmp_path)
    times = np.arange(12, dtype=float)
    labels = np.array([0] * 4 + [1] * 2 + [2] * 6)
    payload = {
        "segments": cache_path._segments_from(times, labels, np.zeros(12), 1.0),
        "level": "2s-only", "times": times, "labels": labels, "k": 3,
        "margins": np.zeros(12),
    }
    sents = [sent(0.0, 4.0, "A"), sent(4.0, 6.0, "B"), sent(6.0, 12.0, "B")]
    reference = [{"start": 0.0, "end": 4.0, "speaker": "A"},
                 {"start": 4.0, "end": 12.0, "speaker": "B"}]

    base = metrics.sentence_metrics(sents, payload["segments"], "2s-only")
    assert base["overall"] == pytest.approx(2 / 3, abs=1e-12)
    assert metrics.der_metrics(reference, payload["segments"])["der"] > 0.1

    out = metrics.with_merged(MEETING, payload, sents, reference)

    assert out["merged_overall"] == 1.0
    assert out["merged_der"] == 0.0


def test_with_merged_skips_rebuild_when_nothing_to_merge(monkeypatch):
    # 没有可消重的簇（空 labels / k<2）时不许碰缓存：T1 的 insufficient payload 走这条。
    def boom(*args, **kwargs):
        raise AssertionError("无可消重时不应重建段表")

    monkeypatch.setattr(metrics.CP, "segments_for", boom)
    for labels, k in ((np.array([], dtype=int), 0), (np.array([0] * 3), 1)):
        payload = {"segments": [], "level": "2s-only", "times": np.arange(len(labels) or 1,
                   dtype=float), "labels": labels, "margins": np.array([]), "k": k}
        out = metrics.with_merged(MEETING, payload, FOUR_SENTS, REF)
        assert out["merged_overall"] == 0.0            # 空段表 ⇒ 全弃权
        assert out["merged_der"] == pytest.approx(1.0, abs=1e-6)   # 预测为空，参考全时长 miss


def test_with_merged_without_reference_returns_unmerged(monkeypatch):
    # 没有真值参照（其余 38 场）时不许抛异常，也不能假装消过重。
    def boom(*args, **kwargs):
        raise AssertionError("无参照时不应重建段表")

    monkeypatch.setattr(metrics.CP, "segments_for", boom)
    times = np.arange(12, dtype=float)
    labels = np.array([0] * 6 + [1] * 6)
    payload = {"segments": cache_path._segments_from(times, labels, np.zeros(12), 1.0),
               "level": "2s-only", "times": times, "labels": labels, "k": 2,
               "margins": np.zeros(12)}
    base = metrics.sentence_metrics(FOUR_SENTS, payload["segments"], "2s-only")
    out = metrics.with_merged(MEETING, payload, FOUR_SENTS, [])
    assert out["merged_overall"] == base["overall"]
