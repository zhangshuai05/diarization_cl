# -*- coding: utf-8 -*-
"""诊断 probe_sentence_attribution 的三个可疑数：
1. 弃权 25.7% 从哪来（VAD 门覆盖？句长偏移？）
2. 主表 11.5% vs 辅表 60% 的巨大落差（是按说话人偏斜还是按句长偏斜）
3. 匈牙利映射的**上界**有多高（cap 只有 8，真值 6 人时是否够）
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
    CACHE, DATA, FRAME, MIN_VAD, VAD_CACHE, WIN, HOP,
    attribution, clusters_for, frame_map, read_textgrid, sentence_predictions,
    sentences_from_textgrid)

meeting = "L_R003S01C02"
emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
intervals = [(a, b) for a, b in json.loads((VAD_CACHE / f"{meeting}.json").read_text(encoding="utf-8"))["intervals"]]
emb, times = gate_windows(emb_all, times_all, intervals, WIN, MIN_VAD)
tg = read_textgrid(DATA / "TextGrid" / f"{meeting}.TextGrid")
sentences = sentences_from_textgrid(tg, 600.0)
end = max(s["end"] for s in sentences)

print(f"窗: RTTM 门 {len(emb_all)} → VAD 门 {len(emb)}\n"
      f"句子: {len(sentences)} 段, 并集跨度 {end:.0f}s\n")

# ── 1. VAD 门对时间轴的覆盖
speech = sum(min(e, end) - s for s, e in intervals if s < end)
print(f"[1] VAD 区间覆盖 {speech:.0f}s / {end:.0f}s = {100*speech/end:.1f}% 的时间轴")

# 窗能盖到的总时长（并集，避免重叠重复计）
wins = sorted((t, min(t + WIN, end)) for t in times if t < end)
merged = []
for a, b in wins:
    if merged and a <= merged[-1][1] + 1e-9:
        merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    else:
        merged.append((a, b))
covered = sum(b - a for a, b in merged)
print(f"    窗并集覆盖 {covered:.0f}s = {100*covered/end:.1f}%  ← 弃权的直接来源")
gaps = [(merged[i][1], merged[i+1][0]) for i in range(len(merged)-1)]
gaps = [g for g in gaps if g[1] - g[0] > 0.3]
print(f"    窗间空档 {len(gaps)} 处, 共 {sum(b-a for a,b in gaps):.0f}s, 最长 {max((b-a for a,b in gaps), default=0):.1f}s")

# ── 2. 弃权句 vs 非弃权句：句长 / 时长权重
labels5 = clusters_for(emb, 5)
grid = frame_map(times, labels5, np.full(len(times), WIN), end)
pred, abstain = sentence_predictions(sentences, grid)
ab_s = [s for s, p in zip(sentences, pred) if p is None]
ok_s = [s for s, p in zip(sentences, pred) if p is not None]
print(f"\n[2] 弃权 {abstain}/{len(sentences)} = {100*abstain/len(sentences):.1f}%")
for nm, grp in (("弃权句", ab_s), ("有预测句", ok_s)):
    if grp:
        d = sorted(s["end"] - s["start"] for s in grp)
        print(f"    {nm}: {len(grp)} 句, 时长合计 {sum(d):.0f}s, 中位句长 {d[len(d)//2]:.2f}s")

# ── 3. 按真值说话人分解（k=5）
truth = [s["speaker"] for s in sentences]
per_sp = {}
for s, p in zip(sentences, pred):
    d = s["end"] - s["start"]
    r = per_sp.setdefault(s["speaker"], [0.0, 0.0, 0, 0])
    r[0] += d
    if p is None:
        r[1] += d
        r[3] += 1
    else:
        r[2] += 1
print(f"\n[3] 按真值说话人(并集段归给谁): 总时长 / 弃权时长 / 句数")
for sp, (tot, ab, n_ok, n_ab) in sorted(per_sp.items(), key=lambda kv: -kv[1][0]):
    print(f"    {sp:<8} 总 {tot:>6.0f}s  弃权 {100*ab/tot if tot else 0:>5.1f}%  句数 {n_ok+n_ab:>4}")

# ── 4. 匈牙利上界：真值分布 vs 预测分布
print(f"\n[4] 真值说话人 {len(set(truth))} 个; k=5 时预测 {len(set(labels5))} 个簇")
for k in (5, 6, 7, 8):
    lab = clusters_for(emb, k)
    print(f"    k={k}: 簇数 {len(set(lab))}, 簇大小 {sorted(Counter(lab).values(), reverse=True)}")
res = attribution(sentences, pred)
print(f"    k=5 归属: 主表 {res['overall']['duration_weighted']} 辅表 {res['overall']['per_sentence']}")
print(f"    ≥0.5s 计入 {res['n_sentences_scored']}/{res['n_sentences_all']}, "
      f"剔除短句 {res['n_dropped_short']}")
