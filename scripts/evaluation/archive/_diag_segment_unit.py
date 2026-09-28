# -*- coding: utf-8 -*-
"""查清 515 vs 123：单 tier 的 mark 到底是"句子"还是被切碎的片段。
判据：同一 tier 内相邻非空 mark 之间的**空档长度分布**。
若大量空档极短(<0.3s)，说明标注是连续的话语被切碎，并集合并才是"句子"；
若空档普遍在秒级，说明 mark 本身就是句子，515 才对。
"""
import sys
from collections import Counter
from pathlib import Path

# archive/ 比 scripts/evaluation/ 多一层，仓库根是 parents[3]。
# 早先用 parents[2]，搬进 archive/ 后没跟着改，指向了 scripts/ 而非仓库根。
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from probe_sentence_attribution import read_textgrid, sentences_from_textgrid  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")


def main(meet):
    tg = read_textgrid(DATA / "TextGrid" / f"{meet}.TextGrid")
    print(f"=== {meet} ===\n")
    for nm, ivs in sorted(tg.items()):
        mk = sorted([(s, e) for s, e, t in ivs if t.strip()])
        if len(mk) < 2:
            continue
        gaps = [mk[i + 1][0] - mk[i][1] for i in range(len(mk) - 1)]
        gs = sorted(gaps)
        durs = sorted(e - s for s, e in mk)
        tiny = sum(1 for g in gaps if g < 0.3)
        zero = sum(1 for g in gaps if abs(g) < 1e-6)
        print(f"  tier {nm:<8} mark {len(mk):>3} 个 | 句长中位 {durs[len(durs)//2]:>5.2f}s"
              f" | 空档中位 {gs[len(gs)//2]:>5.2f}s 最小 {gs[0]:>6.2f}s 最大 {gs[-1]:>6.1f}s"
              f" | 空档<0.3s: {tiny}/{len(gaps)} 其中=0: {zero}")
    # 空档分布汇总
    allgaps = []
    for nm, ivs in tg.items():
        mk = sorted([(s, e) for s, e, t in ivs if t.strip()])
        allgaps += [mk[i + 1][0] - mk[i][1] for i in range(len(mk) - 1)]
    allgaps.sort()
    print(f"\n  全场空档 {len(allgaps)} 个:")
    for thr in (0.001, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        n = sum(1 for g in allgaps if g < thr)
        print(f"    <{thr:>5}s: {n:>4} = {100*n/len(allgaps):>5.1f}%")
    marks = sum(1 for ivs in tg.values() for s, e, t in ivs if t.strip())
    print(f"\n  单 tier 段 {marks}; 并集切段 {len(sentences_from_textgrid(tg, 600.0))}")
    # 逐个说话人的 turn 数（按空档 >0.5s 合并）
    turns = 0
    for nm, ivs in tg.items():
        mk = sorted([(s, e) for s, e, t in ivs if t.strip()])
        for i, (s, e) in enumerate(mk):
            if i == 0 or s - mk[i - 1][1] > 0.5:
                turns += 1
    print(f"  按**空档>0.5s**合并后的话语段 = {turns}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_R003S01C02")
