# -*- coding: utf-8 -*-
"""诊断：加了碎簇惩罚后，剩下的 9 场失败是什么形状？

要回答：
1. `pick_best_k` 相对 argmax 改了哪几场？为什么改/没改？
2. 真实路径选 k 上的归属率为什么那么低（M_R003S02C01 真实 k=2 只给 38.8%）？
   ——如果"真实 k 上代理也很差"，说明代理与真实路径量的不是同一件事，
   而不是代理偏高。
3. 各场归属率随 k 的**曲线形状**（单调？内点峰值？）
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# archive/ 比 scripts/evaluation/ 多一层，仓库根是 parents[3]。
# 早先用 parents[2]，搬进 archive/ 后没跟着改，指向了 scripts/ 而非仓库根。
ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_sentence_attribution as P  # noqa: E402
from probe_sentence_vs_realpath import sweep  # noqa: E402
from sentence_k_rule import pick_best_k  # noqa: E402

rows = json.loads((ROOT / "artifacts" /
                   "v13_e2e_unlocked2_dev_mw2_objective.json").read_text(encoding="utf-8"))["rows"]
real = {r["meeting"]: r for r in rows}
KS = list(range(2, 13))

for meeting in sorted(real):
    r = real[meeting]
    fk = r["fixed"]["k"]
    acc, sizes = sweep(meeting, KS)
    argmax = max(acc, key=lambda k: acc[k])
    pen = pick_best_k(acc, sizes)
    curve = " ".join(f"{k}:{100*acc[k]:.0f}" for k in KS)
    frag = {k: sum(1 for s in sizes[k] if s < 4) for k in KS}
    shape = "单调↑" if all(acc[b] >= acc[a] - 0.02 for a, b in zip(KS, KS[1:])) else "有峰"
    print(f"\n{meeting}  真K={r['eff_k']} 真实fix k={fk} ΔDER={100*(r['base']['der']-r['fixed']['der']):.1f}pts")
    print(f"  曲线 {shape}: {curve}")
    print(f"  碎簇数: " + " ".join(f"{k}:{frag[k]}" for k in KS))
    print(f"  argmax={argmax}({100*acc[argmax]:.0f}%)  pick_best_k={pen}({100*acc[pen]:.0f}%)  "
          f"真实k上={100*acc.get(fk,float('nan')):.0f}%")
