# sup/databuild.py
import json, os, re
from difflib import SequenceMatcher
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output")

def _norm(s):
    return re.sub(r"[\s,，。.！!？?、：:；;\"'“”‘’]", "", s)

def align_turns(turns, sentences, thresh=0.35):
    cands = [(i, _norm(s.get("text", ""))) for i, s in enumerate(sentences)]
    out = []
    for t in turns:
        tt = _norm(t["text"])
        best, bi = 0.0, None
        for i, c in cands:
            r = SequenceMatcher(None, tt, c).ratio()
            if r > best:
                best, bi = r, i
        if bi is None or best < thresh:
            out.append((None, best))
        else:
            out.append((bi, best))
    return out

def label_segments(vsegs_ms, seg_ids, sentence_spk, sentences):
    lab = {}
    for si in seg_ids:
        st, ed = vsegs_ms[si]
        best, bo = -1, 0
        for sent_idx, spk in sentence_spk.items():
            s = sentences[sent_idx]
            ov = max(min(ed, float(s["end"])) - max(st, float(s["start"])), 0)
            if ov > bo:
                bo, best = ov, spk
        if bo > 0:
            lab[si] = best
    return lab

def build_file(base, models=None):
    from online_diar_v2 import extract_robust
    from online_diar import load_models
    vad, spk = models if models else load_models()
    wav = None
    for dev in ["Device116", "Device118"]:
        cand = os.path.join(HERE, "Audio", dev, base + ".wav")
        if os.path.exists(cand):
            wav = cand
            break
    assert wav, f"wav not found for {base}"
    vsegs, seg_ids, seg_embs = extract_robust(vad, spk, wav)
    gt = json.load(open(os.path.join(OUT, "ground_truth.json"), encoding="utf-8"))
    g = next(x for x in gt if x["file"] == base + ".wav")
    seaco_path = os.path.join(OUT, "vadfix", base + "_diar.json")
    sentences = []
    if os.path.exists(seaco_path):
        seaco = json.load(open(seaco_path, encoding="utf-8"))
        sentences = seaco.get("sentence_info") or []
    align = align_turns(g["turns"], sentences)
    # 说话人名 -> 数字标签（按首现顺序）
    names = []
    for t in g["turns"]:
        if t["spk"] not in names:
            names.append(t["spk"])
    sentence_spk = {}
    for (bi, r), t in zip(align, g["turns"]):
        if bi is not None:
            sentence_spk[bi] = names.index(t["spk"])
    seg_lab = label_segments(vsegs, seg_ids, sentence_spk, sentences)
    N = len(seg_ids)
    labels = np.full(N, -1, dtype=np.int64)
    mask = np.zeros(N, dtype=np.float32)
    for i, si in enumerate(seg_ids):
        if si in seg_lab:
            labels[i] = seg_lab[si]
            mask[i] = 1.0
    os.makedirs(os.path.join(OUT, "sup", "seq"), exist_ok=True)
    np.savez(os.path.join(OUT, "sup", "seq", base + ".npz"),
             embs=seg_embs, labels=labels, mask=mask)
    return {"base": base, "embs": seg_embs, "labels": labels, "mask": mask}
