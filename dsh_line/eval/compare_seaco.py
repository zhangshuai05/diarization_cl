# -*- coding: utf-8 -*-
"""
分离结果 vs 人工真值：标签稳定性评估。
核心问题：同一个真人的句子被标成几个不同 spk（拆标签）？
方法：真值每一轮文本 -> 与分离分句文本做相似度对齐 -> 统计每个真人映射到的 spk 分布。
用法: python compare_seaco.py [结果目录]   (默认 output\auto)
"""
import json, os, re, sys
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


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


def main(resdir, suffix="_diar"):
    gt = json.load(open(os.path.join(OUT, "ground_truth.json"), encoding="utf-8"))
    report = []
    for g in gt:
        base = g["file"].rsplit(".", 1)[0]
        jp = os.path.join(resdir, base + suffix + ".json")
        if not os.path.exists(jp):
            print("no result for", g["file"])
            continue
        out = json.load(open(jp, encoding="utf-8"))
        si = out.get("sentence_info") or []
        if not si:
            print("empty sentence_info for", g["file"])
            continue
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

        n_spks = sorted(set(str(s.get("spk", "?")) for s in si))
        line = {
            "file": g["file"], "dist": g["distance"],
            "gt_turns": g["n_turns"], "seaco_sents": len(si),
            "spks": n_spks, "matched": matched,
        }
        print(f"{g['file']:<28} dist={g['distance']:<5} "
              f"spks={n_spks} matched={matched}/{g['n_turns']}")
        for p in g["speakers"]:
            d = person_spks[p]
            main = max(d.items(), key=lambda kv: kv[1])[0] if d else "-"
            purity = (max(d.values()) / sum(d.values())) if d else 0.0
            print(f"    {p:<5} -> {d}  主标签={main} 纯度={purity:.2f}")
            line[p] = {"main_spk": main, "spk_dist": d, "purity": purity}
        report.append(line)

    with open(os.path.join(resdir, "eval_diar.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(resdir, "eval_diar.json"))


if __name__ == "__main__":
    args = sys.argv[1:]
    resdir = os.path.join(OUT, "auto")
    suffix = "_diar"
    if args and not args[0].startswith("--"):
        resdir = args.pop(0)
    if args and args[0] == "--suffix" and len(args) > 1:
        suffix = args[1]
    main(resdir, suffix)
