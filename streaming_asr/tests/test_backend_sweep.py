# -*- coding: utf-8 -*-
"""锁住 `probe_backend_sweep` 的四处口径修正（2026-10-10）。

这四处都不报错、只出错数 —— 与本仓库反复出现的同一类陷阱
（见 `test_probe_silent_failures.py` 的四个先例）：

1. 训练池含 holdout —— 从封存集里学，直接违反 spec §3.3
2. k 钉在真值有效人数 —— 用了被禁假设，把最贵的一环（选 K）整个摘掉
3. 跳过两级细化 —— 与生产不同构，测的只是链条中间一层
4. 只有 frame accuracy、无句级指标 —— 而验收指标是句级错答率
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_backend_sweep as BS  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")


def test_pool_excludes_holdout_when_split_is_dev():
    """池必须只含本 split 的场次 —— 否则封存集为一个输出数字做了贡献。"""
    # 数据/缓存是机器本地的绝对路径（D:\yinp\…），新克隆、CI、同事机器上不存在。
    # pytest.ini 的 testpaths=tests 让本文件进每一次全量跑，所以缺数据时必须
    # skip 而不是 error —— 否则整个包套件在别的机器上全红。
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:\\yinp\\…）")
    dev = split_meetings(DATA, "dev")
    holdout = split_meetings(DATA, "holdout")
    assert dev and holdout, "前置：两个 split 都应非空"
    assert not (set(dev) & set(holdout)), "前置：两者身份组已隔离"

    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    # 前置：缓存缺失/改名时池会是空的，两条断言随即恒真 —— 那正是本仓库
    # 反复出现的"不报错只出错数"，所以先钉住池非空。
    assert pool, f"前置：池非空（缓存 {BS.DEFAULT_CACHE} 读数是否正常？）"
    leaked = sorted(set(pool) & set(holdout))
    assert leaked == [], f"训练池泄漏了封存集场次：{leaked}"
    assert set(pool) <= set(dev), f"池含非本 split 场次：{sorted(set(pool) - set(dev))}"


def test_refinement_is_applied_and_level_recorded():
    """变体结果必须记录细化层级 —— 否则无法区分'后端在 AHC 层的差异'
    与'整条链路的差异'，而生产里 AHC 只是中间一环。"""
    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    if not pool:
        import pytest
        pytest.skip("无标注窗缓存")

    import numpy as np
    import probe_ab_dev as AB
    import probe_path_equiv as PE

    meeting = sorted(pool)[0]
    emb, times, speakers, votes = pool[meeting]
    rE, rtimes = PE.cached_stage(meeting, "refine")
    assert len(rE) > 0, "前置：refine 缓存应非空"

    labels = np.zeros(len(emb), dtype=int)          # k=1，最稳的输入
    fl, ft, level = AB.real_path_labels(labels, emb, times, rE, rtimes)
    assert level in ("refined", "2s-only")
    # k<2 时必须返回原 times（`real_path_labels` docstring 记录的坑：
    # 用 arange 重建网格会让 600s 参考的后半段全被判成无覆盖，弃权假报 37%）
    if level == "2s-only":
        assert np.allclose(ft, times), "k<2 时不得重建时间网格"


def test_refine_labels_off_and_on_are_wired():
    """`refine_labels` 是本任务新增的接线点，且 `--refine off` 必须不崩。

    旧测试只碰 `probe_ab_dev.real_path_labels`（本任务未改），k=1 时它走
    早退分支、两个断言恒真 —— 对本任务新增的代码零覆盖。
    本测试走两段：先单元层验两个分支，再真跑一次 `main` 验行里的 `level`。
    """
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:\\yinp\\…）")
    import numpy as np
    import probe_ab_dev as AB
    import probe_path_equiv as PE

    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    assert pool, "前置：标注窗缓存应非空"
    meeting = sorted(pool)[0]
    emb, times, _, _ = pool[meeting]
    rE, rtimes = PE.cached_stage(meeting, "refine")

    labels = np.zeros(len(emb), dtype=int)

    # off：原样返回 + level="none"（不能调 AB._win，它不认 "none"）
    off_l, off_t, off_level = BS.refine_labels(labels, emb, times, rE, rtimes, False)
    assert off_level == "none"
    assert off_t is times and off_l is labels, "off 分支必须原样返回，不得重建网格"

    # on：走 real_path_labels，level 只能是这两个
    _, _, on_level = BS.refine_labels(labels, emb, times, rE, rtimes, True)
    assert on_level in ("refined", "2s-only")

    # 细化窗与锚点必须在**同一空间** —— 这是本任务唯一会崩的地方，也是
    # brief 原文里唯一真正错的地方（原文直接传 rE，而 anchors 取自 LDA 空间）。
    #
    # ⚠️ 判据不能写成"`fit` 输出的列数与输入无关"。那是**恒真**的：`Transform`
    # 是 `(X - mean) @ weights`，输出宽度只由 `weights` 定（实测 14 == 14），
    # 在 brief 原文下也照样通过。唯一会红的形式是"目标调用不抛"。
    from probe_gap_decomp import cluster_labels
    fit, _ = BS.fit_backend(pool, meeting, 8, True)
    assert fit is not None, "前置：本场应能拟合"
    vectors, rvectors = fit(emb), fit(rE)
    assert rvectors.shape[1] == vectors.shape[1], (
        f"细化窗在错误空间（{rvectors.shape[1]} 列 vs 聚类空间 {vectors.shape[1]} 列）")
    labels_k = cluster_labels(vectors, 3, metric=fit.metric)
    # brief 原文下这里是 `rE @ anchors.T`：ValueError: matmul ... size 14 is
    # different from 192 —— 会 raise，不会 assert。
    fclusters, ftimes, level = BS.refine_labels(
        labels_k, vectors, times, rvectors, rtimes, True)
    assert level in ("refined", "2s-only"), "投影后的细化窗必须能被消化"
    assert len(ftimes) == len(rvectors), "细化网格应与细化窗一一对应"

    # ── 端到端：真跑一次 `main`，把同一件事钉在整条变体循环上 ──
    # 上面几段都只调函数；"level 落进产物行"这个验收点只有穿过 `main`
    # 才能验到，`fit is None` 存根行带不带 `level` 也只在这里暴露。
    import json
    import tempfile

    # 两场：目标场之外还剩一场可拟合 ⇒ LDA 臂真的会跑（单场时池为空、
    # 只走存根，就绕过了要测的那条路）。
    two = sorted(pool)[:2]
    assert len(two) == 2, "前置：至少两场"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sweep.json"
        argv = sys.argv
        sys.argv = ["probe_backend_sweep.py", "--meetings", *two,
                    "--seconds", "120", "--classes", "8", "--norm", "cosine",
                    "--output", str(out)]
        try:
            BS.main()          # brief 原文下：这里抛 ValueError，测试红
        finally:
            sys.argv = argv
        rows = json.loads(out.read_text(encoding="utf-8"))["meetings"]
    assert len(rows) == 2, f"两场都应出行：{len(rows)}"
    for row in rows:
        assert row["variants"], f"{row['meeting']} 无变体行"
        for name, v in row["variants"].items():
            assert "level" in v, f"{row['meeting']}/{name} 缺 level 键"
            if v["accuracy"] is not None:
                assert v["level"] in ("refined", "2s-only", "none"), v["level"]
