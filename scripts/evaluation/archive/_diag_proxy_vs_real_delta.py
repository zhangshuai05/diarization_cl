# -*- coding: utf-8 -*-
"""决定性检验：代理的 Δ（归属率随 k 的变化）与真实路径的 Δ（DER 随 k 的变化）
在**逐场**上是否相关？

若相关 → 代理至少保留了排序信息，只是需要重新标定阈值。
若无关/反向 → 代理与真实路径量的不是同一件事，作为 K 判据不可救（§4.1e 的重演）。

用真实路径已有的两条腿（base.k → fixed.k）作对照：代理在同样的 base.k 与
fixed.k 上算两个归属率，取差；真实路径取 DER 差。逐场比。
"""
import json
import sys
from pathlib import Path

import numpy as np

# archive/ 比 scripts/evaluation/ 多一层，仓库根是 parents[3]。
# 早先用 parents[2]，搬进 archive/ 后没跟着改，指向了 scripts/ 而非仓库根。
ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_sentence_vs_realpath import sweep  # noqa: E402

rows = json.loads((ROOT / "artifacts" /
                   "v13_e2e_unlocked2_dev_mw2_objective.json").read_text(encoding="utf-8"))["rows"]

proxy_d, real_d, names = [], [], []
print(f"{'会议':<20}{'真k变化':>12}{'真实ΔDER':>10}{'代理Δ归属':>11}")
for r in sorted(rows, key=lambda x: x["meeting"]):
    bk, fk = r["base"]["k"], r["fixed"]["k"]
    if bk == fk:
        continue
    acc, _ = sweep(r["meeting"], sorted({bk, fk}))
    pd = acc.get(fk, np.nan) - acc.get(bk, np.nan)
    rd = r["base"]["der"] - r["fixed"]["der"]
    proxy_d.append(pd); real_d.append(rd); names.append(r["meeting"])
    print(f"{r['meeting']:<20}{f'{bk}->{fk}':>12}{rd*100:>9.1f}pts{100*pd:>10.1f}pts")

proxy_d, real_d = np.array(proxy_d), np.array(real_d)
print(f"\n  n={len(proxy_d)}（只有 k 变化的场次）")
if len(proxy_d) > 2:
    print(f"  Pearson  r = {np.corrcoef(proxy_d, real_d)[0,1]:+.3f}")
    print(f"  Spearman r = {np.corrcoef(np.argsort(np.argsort(proxy_d)), np.argsort(np.argsort(real_d)))[0,1]:+.3f}")
    print(f"  代理Δ 均 {100*proxy_d.mean():.1f}pts, 真实Δ 均 {100*real_d.mean():.1f}pts  "
          f"(放大 {proxy_d.mean()/real_d.mean():.1f} 倍)")

# 同时看"代理在真实路径的 base.k 与 fixed.k 上都给多少"——若 base 也高，
# 说明代理根本不认为 base 是错的，那它就没有可用的排序
print(f"\n  逐场：代理认为 base.k 有多好？")
for r in sorted(rows, key=lambda x: x["meeting"]):
    bk, fk = r["base"]["k"], r["fixed"]["k"]
    acc, _ = sweep(r["meeting"], sorted({bk, fk}))
    flag = "  ← 代理同意 base" if acc.get(bk, 0) >= acc.get(fk, 0) - 0.02 else ""
    print(f"    {r['meeting']:<20} base k={bk}: {100*acc.get(bk,float('nan')):>5.1f}%  "
          f"fixed k={fk}: {100*acc.get(fk,float('nan')):>5.1f}%{flag}")
