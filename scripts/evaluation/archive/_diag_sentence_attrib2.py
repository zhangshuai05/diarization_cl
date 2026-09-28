# -*- coding: utf-8 -*-
"""量化三件事的形状，决定指标怎么修：
(1) 并集切段是否把话切碎（与单 tier mark 数对比）
(2) 2.0s 粗窗分辨率的**上界**：每句"跨越窗边界"的比例 → 粗窗腿的天花板
(3) 弃权是否系统性吃掉某个说话人（VAD 门 vs RTTM 门对照）
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

import claude_a2_v11 as v11  # noqa: E402
from probe_gap_decomp import cached_windows, gate_windows  # noqa: E402
from probe_sentence_attribution import (  # noqa: E402
    CACHE, DATA, MIN_VAD, VAD_CACHE, WIN, HOP,
    clusters_for, read_textgrid, sentences_from_textgrid)

meeting = sys.argv[1] if len(sys.argv) > 1 else "L_R003S01C02"
tg = read_textgrid(DATA / "TextGrid" / f"{meeting}.TextGrid")
marks = [(s, e, nm) for nm, ivs in tg.items() for s, e, t in ivs if t.strip()]
sentences = sentences_from_textgrid(tg, 600.0)

print(f"=== {meeting} ===")
print(f"[1] 单 tier mark 段数 = {len(marks)}; 并集切段 = {len(sentences)} "
      f"({100*len(sentences)/len(marks):.0f}%)")
d = sorted(s["end"] - s["start"] for s in sentences)
print(f"    并集段长: 中位 {d[len(d)//2]:.2f}s, <0.5s {sum(1 for x in d if x < 0.5)}/{len(d)}, "
      f"最长 {d[-1]:.1f}s")
# 相邻同说话人段合并（是否切碎了）
merged = []
for s in sentences:
    if merged and merged[-1]["speaker"] == s["speaker"] and s["start"] - merged[-1]["end"] < 1e-9:
        merged[-1]["end"] = s["end"]
    else:
        merged.append(dict(s))
print(f"    相邻同说话人合并后 = {len(merged)} 段 (减少 {len(sentences)-len(merged)})")

# ── 2. 粗窗分辨率天花板：只用一个 k=5 的标签，看"整句落在同一窗内"的比例
emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
intervals = [(a, b) for a, b in
             json.loads((VAD_CACHE / f"{meeting}.json").read_text(encoding="utf-8"))["intervals"]]
emb, times = gate_windows(emb_all, times_all, intervals, WIN, MIN_VAD)
tarr = np.asarray(times)
scored = [s for s in sentences if s["end"] - s["start"] >= 0.5]
single_window = straddle = no_window = 0
for s in scored:
    a, b = s["start"], s["end"]
    covering = [(t, t + WIN) for t in tarr if t < b and t + WIN > a]
    if not covering:
        no_window += 1
    elif len(covering) == 1:
        single_window += 1
    else:
        straddle += 1
n = len(scored)
print(f"\n[2] ≥0.5s 句 {n} 句:")
print(f"    落在**单一窗**内(粗窗腿可能判对): {single_window} = {100*single_window/n:.1f}%")
print(f"    跨**多个窗**(粗窗腿受边界污染): {straddle} = {100*straddle/n:.1f}%")
print(f"    完全无窗(必弃权): {no_window} = {100*no_window/n:.1f}%")
print(f"    → 粗窗腿的**上界约 {100*(single_window)/n:.1f}%**（跨窗句即使 k 对也被多数票稀释）")

# ── 3. 弃权按说话人 × 门
for gate, win_times in (("VAD 门", times), ("RTTM 门", times_all)):
    t = np.asarray(win_times)
    per = {}
    for s in scored:
        a, b = s["start"], s["end"]
        has = any(x < b and x + WIN > a for x in t)
        r = per.setdefault(s["speaker"], [0, 0])
        r[0] += 1
        r[1] += 0 if has else 1
    tot = sum(v[0] for v in per.values())
    ab = sum(v[1] for v in per.values())
    detail = "  ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(per.items(), key=lambda kv: -kv[1][0]))
    print(f"\n[3] {gate}: 弃权 {ab}/{tot} = {100*ab/tot:.1f}%   ({detail})")

# ── 4. 真值 K vs 各 k 的簇数（cap 是否够）
lab5 = clusters_for(emb, 5)
print(f"\n[4] 真值说话人(≥0.5s 句) = {len({s['speaker'] for s in scored})} 个; "
      f"实际出现在窗里的 = {len(set(lab5))} 簇(k=5)")
