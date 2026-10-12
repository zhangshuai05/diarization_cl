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

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_backend_sweep as BS  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")


def test_pool_excludes_holdout_when_split_is_dev():
    """池必须只含本 split 的场次 —— 否则封存集为一个输出数字做了贡献。"""
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
