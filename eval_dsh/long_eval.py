# -*- coding: utf-8 -*-
"""
长会议评估：GT 逐轮文本 -> 与分句文本相似度对齐 -> 每说话人主标签/纯度。
用法: python long_eval.py [结果目录]
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


def eval_meeting(gt, resdir, res_name):
    jp = os.path.join(resdir, res_name + ".json")
    if not os.path.exists(jp):
        print(f"  [no result] {res_name}")
        return None
    out = json.load(open(jp, encoding="utf-8"))
    si = out.get("sentence_info") or []
    cands = [(i, norm(s.get("text", ""))) for i, s in enumerate(si)]
    person_spks = {p: {} for p in gt["speakers"]}
    matched = 0
    for t in gt["turns"]:
        bi, r = best_match(t["text"], cands)
        if bi is None or r < 0.35:
            continue
        spk = str(si[bi].get("spk", "?"))
        person_spks[t["spk"]][spk] = person_spks[t["spk"]].get(spk, 0) + 1
        matched += 1
    n_spks = len(set(str(s.get("spk", "?")) for s in si))
    res = {"n_spks": n_spks, "matched": matched}
    for p in gt["speakers"]:
        d = person_spks[p]
        tot = sum(d.values())
        main = max(d.items(), key=lambda kv: kv[1])[0] if d else "-"
        pur = (max(d.values()) / tot) if tot else 0.0
        res[p] = {"main": main, "purity": pur, "n": tot}
    mains = {p: res[p]["main"] for p in gt["speakers"]}
    res["distinct"] = len(set(mains.values())) > 1 and "-" not in mains.values()
    return res


def main():
    resdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "online2")
    if not os.path.isabs(resdir):
        resdir = os.path.join(HERE, resdir)
    for name in ["L1_118_wzy_zjl", "L2_118_ash_wzy", "L3_116_zjl_wzy"]:
        gtp = os.path.join(OUT, "long", name + "_gt.json")
        if not os.path.exists(gtp):
            continue
        gt = json.load(open(gtp, encoding="utf-8"))
        print(f"===== {name}: {gt['n_turns']} turns, spks={gt['speakers']}")
        res_names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(resdir)
            if f.startswith(name + "_") and f.endswith(".json"))
        for res_name in res_names:
            r = eval_meeting(gt, resdir, res_name)
            if r is None:
                continue
            pur_s = " ".join(f"{p}={r[p]['purity']:.2f}(main{r[p]['main']})"
                             for p in gt["speakers"])
            flag = "GOOD" if (r["distinct"] and r["n_spks"] <= 2
                              and all(r[p]["purity"] >= 0.8 for p in gt["speakers"])) else ""
            print(f"  {res_name:<28} spks={r['n_spks']} matched={r['matched']}/"
                  f"{gt['n_turns']} distinct={r['distinct']} {pur_s} {flag}")


if __name__ == "__main__":
    main()
