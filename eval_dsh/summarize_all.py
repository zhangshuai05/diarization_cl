# -*- coding: utf-8 -*-
"""
全量汇总：output/auto 下所有文件的分离结果 vs 真值，
按 距离×设备 聚合：标签数分布、每人纯度均值、匹配率。
输出: output/auto/full_summary.csv + 控制台表格
"""
import json, os, re, csv, sys
from difflib import SequenceMatcher
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
AUTO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "auto")
if not os.path.isabs(AUTO):
    AUTO = os.path.join(HERE, AUTO)


def norm(s):
    return re.sub(r"[\s,，。.！!？?、：:；;\"'“”‘’]", "", s)


def best_match(text, cands):
    t = norm(text)
    if not t:
        return None
    best, bi = 0.0, None
    for i, c in cands:
        r = SequenceMatcher(None, t, c).ratio()
        if r > best:
            best, bi = r, i
    return bi, best


def eval_file(g, jp):
    out = json.load(open(jp, encoding="utf-8"))
    si = out.get("sentence_info") or []
    if not si:
        return None
    cands = [(i, norm(s.get("text", ""))) for i, s in enumerate(si)]
    person_spks = {p: {} for p in g["speakers"]}
    matched = 0
    for t in g["turns"]:
        bi, r = best_match(t["text"], cands)
        if bi is None or r < 0.35:
            continue
        spk = str(si[bi].get("spk", "?"))
        person_spks[t["spk"]][spk] = person_spks[t["spk"]].get(spk, 0) + 1
        matched += 1
    n_spks = len(set(str(s.get("spk", "?")) for s in si))
    purities = {}
    mains = {}
    for p in g["speakers"]:
        d = person_spks[p]
        tot = sum(d.values())
        purities[p] = (max(d.values()) / tot) if tot else 0.0
        mains[p] = max(d.items(), key=lambda kv: kv[1])[0] if d else "-"
    return {"n_spks": n_spks, "matched": matched, "purities": purities, "mains": mains}


gt = json.load(open(os.path.join(OUT, "ground_truth.json"), encoding="utf-8"))
rows = []
for g in gt:
    base = g["file"].rsplit(".", 1)[0]
    jp = os.path.join(AUTO, base + "_diar.json")
    if not os.path.exists(jp):
        print("MISSING:", base)
        continue
    r = eval_file(g, jp)
    if r is None:
        continue
    pur = r["purities"]
    ps = list(pur.values())
    rows.append({
        "file": g["file"], "device": "116" if g["file"].startswith("Call") else "118",
        "dist": g["distance"], "n_spks": r["n_spks"],
        "matched": r["matched"], "gt_turns": g["n_turns"],
        "min_purity": round(min(ps), 3) if ps else 0,
        "avg_purity": round(sum(ps) / len(ps), 3) if ps else 0,
        "both_ok": all(p >= 0.8 for p in ps) and r["n_spks"] == 2 and
                   r["mains"][g["speakers"][0]] != r["mains"][g["speakers"][1]],
    })

with open(os.path.join(AUTO, "full_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 聚合
agg = defaultdict(lambda: {"n": 0, "n_spks_ok": 0, "sum_avg_pur": 0.0, "both_ok": 0})
for r in rows:
    key = (r["device"], r["dist"])
    a = agg[key]
    a["n"] += 1
    if r["n_spks"] == 2:
        a["n_spks_ok"] += 1
    a["sum_avg_pur"] += r["avg_purity"]
    if r["both_ok"]:
        a["both_ok"] += 1

print(f"{'device':<8}{'dist':<6}{'files':>6}{'spk==2':>8}{'avg_pur':>9}{'both_ok':>9}")
for key in sorted(agg.keys()):
    a = agg[key]
    print(f"{key[0]:<8}{key[1]:<6}{a['n']:>6}{a['n_spks_ok']:>8}"
          f"{a['sum_avg_pur']/max(a['n'],1):>9.3f}{a['both_ok']:>9}")
print("saved:", os.path.join(AUTO, "full_summary.csv"))
