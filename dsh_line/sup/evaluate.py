# sup/evaluate.py
import json, os, re, csv
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output")

def norm(s):
    return re.sub(r"[\s,，。.！!？?、：:；;\"'“”‘’]", "", s)

def purity_stats(gt, sentences):
    cands = [(i, norm(s.get("text", ""))) for i, s in enumerate(sentences)]
    person_spks = {p: {} for p in gt["speakers"]}
    matched = 0
    for t in gt["turns"]:
        tt = norm(t["text"])
        best, bi = 0.0, None
        for i, c in cands:
            r = SequenceMatcher(None, tt, c).ratio()
            if r > best:
                best, bi = r, i
        if bi is None or best < 0.35:
            continue
        spk = str(sentences[bi].get("spk", "?"))
        person_spks[t["spk"]][spk] = person_spks[t["spk"]].get(spk, 0) + 1
        matched += 1
    res = {"matched": matched,
           "n_spks": len(set(str(s.get("spk", "?")) for s in sentences))}
    for p in gt["speakers"]:
        d = person_spks[p]
        tot = sum(d.values())
        res[f"purity_{p}"] = round((max(d.values()) / tot), 3) if tot else 0.0
        res[f"main_{p}"] = max(d.items(), key=lambda kv: kv[1])[0] if d else "-"
        res[f"n_{p}"] = tot
    mains = [res[f"main_{p}"] for p in gt["speakers"]]
    res["distinct"] = len(set(mains)) > 1 and "-" not in mains
    res["good"] = (res["distinct"] and res["n_spks"] <= 2
                   and all(res[f"purity_{p}"] >= 0.8 for p in gt["speakers"]))
    return res

def main():
    gt = json.load(open(os.path.join(OUT, "ground_truth.json"), encoding="utf-8"))
    rows = []
    for g in gt:
        base = g["file"][:-4]
        for tag, rel in [("uis", f"output/sup/res/{base}_uis.json"),
                         ("v1", f"output/online/{base}_onl_n0.40.json"),
                         ("funasr", f"output/vadfix/{base}_diar.json")]:
            p = os.path.join(HERE, rel)
            if not os.path.exists(p):
                continue
            sentences = json.load(open(p, encoding="utf-8")).get("sentence_info") or []
            if not sentences:
                continue
            r = purity_stats(g, sentences)
            rows.append({"file": base, "dist": g["distance"], "method": tag, **r})
    os.makedirs(os.path.join(OUT, "sup"), exist_ok=True)
    if rows:
        # 必要偏离：ground_truth 不同文件说话人名字不同（wzy/zjl 与 ash/wzy），
        # verbatim 的 `keys = rows[0].keys()` 会因 extrasaction='raise' 在首个 ash 文件处
        # 抛 ValueError。改为按全表行收集字段并集，缺失字段以空串填充。
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(OUT, "sup", "eval_summary.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"saved eval_summary.csv ({len(rows)} rows)")
    for r in rows:
        if r["method"] == "uis":
            pur = " ".join(f"{k[7:]}={v}" for k, v in r.items() if k.startswith("purity_"))
            print(r["file"], r["dist"], "GOOD" if r["good"] else "", pur)

if __name__ == "__main__":
    main()
