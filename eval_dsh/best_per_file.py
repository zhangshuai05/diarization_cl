# -*- coding: utf-8 -*-
"""天花板分析 v2：每个文件在所有策略中的最优成绩。
好 = n_spks==2 且两真人主标签不同 且 双方纯度>=0.8；否则按(min_pur, matched) 取 2 标签且主标签不同的最优。
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "output", "cluster", "summary.csv")

best = {}
rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
for r in rows:
    pur = {k[7:]: float(v) for k, v in r.items()
           if k.startswith("purity_") and v not in ("", None)}
    mains = {k[5:]: v for k, v in r.items()
             if k.startswith("main_") and v not in ("", None)}
    nspk = int(r["n_spks"])
    minp = min(pur.values()) if pur else 0.0
    distinct = len(set(mains.values())) > 1 if mains else False
    good = (nspk == 2 and distinct and minp >= 0.8 and len(pur) == 2)
    # 评分：好 > 2标签且主标签不同时的纯度 > 匹配数
    if nspk == 2 and distinct:
        score = (1 if good else 0, minp, int(r["matched"]))
    else:
        score = (-1, minp, int(r["matched"]))
    key = (r["file"], r["dist"])
    if key not in best or score > best[key][0]:
        best[key] = (score, r["strategy"], nspk, distinct, minp, pur, r["matched"])

print(f"{'file':<24}{'dist':<5}{'best_strategy':<16}{'spks':>5}{'distinct':>9}{'min_pur':>9}{'matched':>8}")
n_good = 0
for (f, d), (score, st, nspk, distinct, minp, pur, mat) in sorted(best.items()):
    good = score[0] == 1
    n_good += 1 if good else 0
    pur_s = " ".join(f"{k}={v:.2f}" for k, v in sorted(pur.items()))
    print(f"{f:<24}{d:<5}{st:<16}{nspk:>5}{str(distinct):>9}{minp:>9.3f}{mat:>8} {pur_s} {'GOOD' if good else ''}")
print(f"\nreachable GOOD files: {n_good}/{len(best)}")
