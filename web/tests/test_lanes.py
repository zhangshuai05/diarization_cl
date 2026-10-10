# -*- coding: utf-8 -*-
"""`lanes` 的泳道名单、旋钮接线与 λ 还原测试。

缓存全用**合成** npz（tmp_path），真值经 `lanes.load_truth` 注入，不碰 D:\\ 真缓存；
真实数据上的口径由 T4 的 DER 对拍兜底。
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
import lanes  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import unlocked_selector  # noqa: E402

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
A2 = np.array([0.8, 0.6, 0.0], dtype=np.float32)   # A 的近邻（cos .8）：k=3 时才该分开
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
MEETING = "SYNTH"


def _stage(root, stage, emb, hop):
    """合成一个阶段的缓存：行 = 窗，times = 窗起点（按该阶段 hop 铺）。"""
    emb = np.asarray(emb, dtype=np.float32)
    target = root / MEETING
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / f"{stage}.npz", emb=emb, times=np.arange(len(emb), dtype=float) * hop)


def _use_synth_cache(monkeypatch, tmp_path, main=None, refine=None, rescue=None):
    monkeypatch.setattr(cache_path.PE, "PROBE_CACHE", tmp_path)
    _stage(tmp_path, "main", main if main is not None else [A] * 6 + [B] * 6, hop=1.0)
    # 细化窗必须铺满 0–12s，否则句级覆盖不到后半段（弃权），oracle 曲线会全为 0。
    _stage(tmp_path, "refine",
           refine if refine is not None else [A] * 40 + [B] * 40, hop=0.15)
    _stage(tmp_path, "rescue", rescue if rescue is not None else [A] * 20 + [B] * 20, hop=0.15)


def _install_truth(monkeypatch, sentences, reference):
    monkeypatch.setattr(lanes, "load_truth", lambda meeting: (sentences, reference))


def _table(n, lam):
    """k=2 轮廓 0.80、k=3 轮廓 0.82：差值 0.02 落在 (0.005, 0.03) 之间 ⇒ 谁说了算由 λ 定。"""
    half, third = n // 2, n // 3
    return [
        {"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0, "objective": 0.0,
         "sizes": [n], "eligible": True},
        {"k": 2, "labels": np.array([0] * half + [1] * (n - half)), "score": 0.80,
         "objective": 0.80, "sizes": [half, n - half], "eligible": True},
        {"k": 3, "labels": np.array([0] * third + [1] * third + [2] * (n - 2 * third)),
         "score": 0.82, "objective": 0.82 - lam, "sizes": [third, third, n - 2 * third],
         "eligible": True},
    ]


def test_knob_defaults_match_brief_verbatim():
    assert lanes.KNOB_DEFAULTS == {
        "lam": 0.005, "min_win": 2, "cap": 10, "mode": "objective",
        "refine": True, "oracle": False, "pin_base": True,
    }


# ───────────────────── base 的钉住开关 ─────────────────────

def test_pin_base_ignores_lam(monkeypatch, tmp_path):
    """钉住时 base 用**生产常量**，λ 旋钮（0.005 → 0.03）逐位动不了它。"""
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    monkeypatch.setattr(v11, "_candidate_clusters",
                        lambda E, max_speakers=None: (_table(len(E), v11.MULTI_K_PENALTY), None))

    a = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                      "pin_base": True, "lam": 0.005}))[0]
    b = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                      "pin_base": True, "lam": 0.03}))[0]

    assert a["lane"] == "base"
    assert a["k"] == 3          # 生产 λ=0.005：k=3 的目标 0.815 > k=2 的 0.80
    assert a == b               # 逐位相等（若 λ 泄漏，第二次会选 k=2）


def test_unpinned_base_follows_lam(monkeypatch, tmp_path):
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    monkeypatch.setattr(v11, "_candidate_clusters",
                        lambda E, max_speakers=None: (_table(len(E), v11.MULTI_K_PENALTY), None))

    low = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                        "pin_base": False, "lam": 0.005}))[0]
    high = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                         "pin_base": False, "lam": 0.03}))[0]

    assert (low["k"], high["k"]) == (3, 2)   # 解钉后 λ=0.03 把 k=3 压到 0.79 < 0.80
    assert low != high


# ───────────────────── unlocked 的旋钮 ─────────────────────

def _fake_unlocked(seen):
    def build(E, cap, min_win):
        seen.append((cap, min_win, v11.MULTI_K_PENALTY))
        return _table(len(E), v11.MULTI_K_PENALTY)
    return build


def test_unlocked_moves_with_lam_when_unpinned(monkeypatch, tmp_path):
    """unlocked 支线读的是调用时刻的 λ；同时 cap/min_win 原样传给 build_unlocked。"""
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    seen = []
    monkeypatch.setattr(unlocked_selector, "build_unlocked", _fake_unlocked(seen))

    low = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "lam": 0.005}))[1]
    high = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                        "lam": 0.03, "min_win": 3, "cap": 8}))[1]

    assert low["lane"] == "unlocked" and high["lane"] == "unlocked"
    assert (low["k"], high["k"]) == (3, 2)
    assert seen == [(10, 2, 0.005), (8, 3, 0.03)]


def test_unlocked_largest_eligible_ignores_lam(monkeypatch, tmp_path):
    """mode=largest_eligible 时 objective 不参与选择 ⇒ 高 λ 也还是 k=3。"""
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    monkeypatch.setattr(unlocked_selector, "build_unlocked", _fake_unlocked([]))

    out = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS,
                                       "lam": 0.03, "mode": "largest_eligible"}))

    assert out[1]["k"] == 3


# ───────────────────── λ 必须还原 ─────────────────────

def test_lam_restored_after_run(monkeypatch, tmp_path):
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    monkeypatch.setattr(v11, "_candidate_clusters",
                        lambda E, max_speakers=None: (_table(len(E), v11.MULTI_K_PENALTY), None))
    monkeypatch.setattr(v11, "MULTI_K_PENALTY", 0.007)   # 生产值的替身
    before = v11.MULTI_K_PENALTY

    list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "lam": 0.03, "pin_base": False}))
    assert v11.MULTI_K_PENALTY == before

    list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "lam": 0.03, "pin_base": True}))
    assert v11.MULTI_K_PENALTY == before


def test_lam_restored_when_a_lane_raises(monkeypatch, tmp_path):
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])
    monkeypatch.setattr(v11, "MULTI_K_PENALTY", 0.007)
    before = v11.MULTI_K_PENALTY

    def boom(E, max_speakers=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(v11, "_candidate_clusters", boom)
    with pytest.raises(RuntimeError, match="boom"):
        list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "lam": 0.03, "pin_base": False}))
    assert v11.MULTI_K_PENALTY == before


# ───────────────────── payload 契约 ─────────────────────

def test_lane_names_and_exact_contract_keys(monkeypatch, tmp_path):
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])   # 无真值：DER 不给数字，但键必须齐

    out = list(lanes.run_lanes(MEETING, lanes.KNOB_DEFAULTS))

    assert [p["lane"] for p in out] == ["base", "unlocked"]
    for payload in out:
        assert tuple(payload) == contract.LANE_PAYLOAD_KEYS   # 键齐、顺序 = 契约阅读顺序
        assert payload["der"] is None and payload["merged_der"] is None
        # 无真值也不能把 DER 当 0 报（der() 在空参照上会给出 >1 的假数）
        assert payload["merged_overall"] == payload["overall"]   # 未合并读数 = 句级 overall


def test_oracle_lane_is_opt_in_and_picks_bijective_peak(monkeypatch, tmp_path):
    """oracle：k=2..10 网格取双射峰值；k=2 把 A/A2 并成一人（2/3），k=3 才 1.0。"""
    _use_synth_cache(monkeypatch, tmp_path,
                     main=[A] * 4 + [A2] * 4 + [B] * 4,
                     refine=[A] * 27 + [A2] * 27 + [B] * 26)
    sentences = [
        {"start": 0.0, "end": 3.9, "speaker": "A", "overlapped": False},
        {"start": 4.1, "end": 7.9, "speaker": "A2", "overlapped": False},
        {"start": 8.1, "end": 11.9, "speaker": "B", "overlapped": False},
    ]
    reference = [{"start": 0.0, "end": 4.0, "speaker": "A"},
                 {"start": 4.0, "end": 8.0, "speaker": "A2"},
                 {"start": 8.0, "end": 12.0, "speaker": "B"}]
    _install_truth(monkeypatch, sentences, reference)

    off = list(lanes.run_lanes(MEETING, lanes.KNOB_DEFAULTS))
    on = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "oracle": True}))

    assert [p["lane"] for p in off] == ["base", "unlocked"]       # 默认关
    assert [p["lane"] for p in on] == ["base", "unlocked", "oracle"]
    oracle = on[2]
    assert (oracle["k"], oracle["overall"]) == (3, 1.0)
    assert tuple(oracle) == contract.LANE_PAYLOAD_KEYS
    assert oracle["der"] == 0.0                                   # 真值网格 = 真值边界


def test_refine_off_keeps_main_windows(monkeypatch, tmp_path):
    _use_synth_cache(monkeypatch, tmp_path)
    _install_truth(monkeypatch, [], [])

    out = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "refine": False}))

    assert [p["level"] for p in out] == ["2s-only", "2s-only"]
    assert all(p["k"] == 2 for p in out)


# ───────────────────── 窗太少：容错产出 ─────────────────────

def test_tiny_cache_yields_lanes_without_exception(monkeypatch, tmp_path):
    """T1 在 len(E)<2 时返回自建 payload（空段表、k=0）：三条泳道都要能出，不抛异常。"""
    _use_synth_cache(monkeypatch, tmp_path, main=[A], refine=[A], rescue=[A])
    sentences = [{"start": 0.0, "end": 1.0, "speaker": "A", "overlapped": False}]
    reference = [{"start": 0.0, "end": 1.0, "speaker": "A"}]
    _install_truth(monkeypatch, sentences, reference)

    out = list(lanes.run_lanes(MEETING, {**lanes.KNOB_DEFAULTS, "oracle": True}))

    assert [p["lane"] for p in out] == ["base", "unlocked", "oracle"]
    for payload in out:
        assert payload["k"] == 0 and payload["segments"] == []
        assert (payload["overall"], payload["abstain"]) == (0.0, 1.0)   # 全弃权
        assert tuple(payload) == contract.LANE_PAYLOAD_KEYS
