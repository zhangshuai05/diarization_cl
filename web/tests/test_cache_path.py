# -*- coding: utf-8 -*-
"""`cache_path` 的边界公式与接线测试。

缓存全用**合成** npz（tmp_path），不碰 D:\\ 上的真缓存；真口径由 T4 对拍兜底。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import cache_path  # noqa: E402

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
MEETING = "SYNTH"


def _stage(root, stage, emb, hop):
    """合成一个阶段的缓存：行 = 窗，times = 窗起点（按该阶段的 hop 铺）。"""
    emb = np.asarray(emb, dtype=np.float32)
    target = root / MEETING
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / f"{stage}.npz", emb=emb, times=np.arange(len(emb), dtype=float) * hop)


def _use_synth_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_path.PE, "PROBE_CACHE", tmp_path)


def test_missing_cache_raises_with_path(tmp_path, monkeypatch):
    import cache_path
    monkeypatch.setattr(cache_path.PE, "PROBE_CACHE", tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        cache_path.cached_meeting("NOT_A_MEETING")
    assert "NOT_A_MEETING" in str(e.value)


def test_segment_boundaries_match_production_formula():
    # 窗起点 [0.0, 1.0, 2.0]，hop=1.0，标签 [0,0,1]；按生产 _segment_windows 的边界公式
    # 段起点=相邻窗重叠中点+hop/2，段末=末窗起点+hop
    import cache_path
    out = cache_path._segments_from(np.array([0.,1.,2.]), np.array([0,0,1]), np.zeros(3), 1.0)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 2.0), (2.0, 3.0)]


def test_cached_meeting_returns_all_three_stages(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)
    _stage(tmp_path, "refine", [A] * 4 + [B] * 4, hop=0.15)
    _stage(tmp_path, "rescue", [A] * 4 + [B] * 4, hop=0.15)

    out = cache_path.cached_meeting(MEETING)

    assert sorted(out) == ["main", "refine", "rescue"]
    assert out["main"]["emb"].shape == (12, 3)
    assert out["refine"]["times"][1] == pytest.approx(0.15)


def test_cached_meeting_raises_when_one_stage_missing(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)
    _stage(tmp_path, "refine", [A] * 4 + [B] * 4, hop=0.15)
    # 缺 rescue.npz

    with pytest.raises(FileNotFoundError) as e:
        cache_path.cached_meeting(MEETING)
    assert "rescue.npz" in str(e.value)


def test_segments_for_raises_when_cache_missing(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        cache_path.segments_for("NOT_A_MEETING")
    assert "NOT_A_MEETING" in str(e.value)


def test_segments_for_refines_when_base_k_is_two(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)
    _stage(tmp_path, "refine", [A] * 4 + [B] * 4, hop=0.15)

    out = cache_path.segments_for(MEETING)

    assert out["level"] == "refined"
    assert out["k"] == 2
    assert sorted(set(out["labels"].tolist())) == [0, 1]
    assert len(out["times"]) == 8          # 细化后 E/times 整体换成密窗
    assert [segment["spk"] for segment in out["segments"]] == [0, 1]
    assert all(segment["start"] < segment["end"] for segment in out["segments"])


def test_segments_for_rescues_when_base_k_is_one(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 12, hop=1.0)
    _stage(tmp_path, "rescue", [A] * 4 + [B] * 4, hop=0.15)

    out = cache_path.segments_for(MEETING)

    assert out["level"] == "rescued"
    assert out["k"] == 2
    assert sorted(set(out["labels"].tolist())) == [0, 1]
    assert len(out["times"]) == 8          # 救援是**整体替换**：times 来自救援窗（8）不是主窗（12）


def test_segments_for_falls_back_to_2s_only_when_refine_collapses(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)
    _stage(tmp_path, "refine", [A] * 8, hop=0.15)   # 密窗全判成同一人 → 细化失败

    out = cache_path.segments_for(MEETING)

    assert out["level"] == "2s-only"
    assert out["k"] == 2
    assert len(out["times"]) == 12                 # 退回主窗


def test_segments_for_accepts_injected_labels(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)

    out = cache_path.segments_for(MEETING, labels=[1] * 6 + [0] * 6, level="2s-only")

    assert out["level"] == "2s-only"
    assert out["k"] == 2
    assert [segment["spk"] for segment in out["segments"]] == [1, 0]
    assert all(segment["margin"] >= 0.0 for segment in out["segments"])


def test_segments_for_rejects_bad_level(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)

    with pytest.raises(ValueError):
        cache_path.segments_for(MEETING, labels=[0] * 12, level="nonsense")


def test_segments_for_rejects_length_mismatch(tmp_path, monkeypatch):
    _use_synth_cache(monkeypatch, tmp_path)
    _stage(tmp_path, "main", [A] * 6 + [B] * 6, hop=1.0)

    with pytest.raises(ValueError):
        cache_path.segments_for(MEETING, labels=[0] * 5, level="2s-only")
