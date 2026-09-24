# sup/online_uis.py
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from online_diar_v2 import extract_robust
from diar_custom import build_timeline_from_segs, assign_to_sentences
from sup.uisrnn import UISRNN

def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    m = UISRNN(**ck["args"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m

def process(vad, spk, wav, model, outdir):
    base = os.path.basename(wav).rsplit(".", 1)[0]
    vsegs, seg_ids, seg_embs = extract_robust(vad, spk, wav)
    with torch.no_grad():
        lab = model.infer(torch.from_numpy(seg_embs)).numpy()
    labmap = {seg_ids[k]: int(lab[k]) for k in range(len(seg_ids))}
    timeline = build_timeline_from_segs(vsegs, labmap)
    sen_path = os.path.join(HERE, "output", "long", base + "_diar.json")
    if not os.path.exists(sen_path):
        sen_path = os.path.join(HERE, "output", "vadfix", base + "_diar.json")
    sentences = []
    if os.path.exists(sen_path):
        sentences = json.load(open(sen_path, encoding="utf-8")).get("sentence_info") or []
    ns = assign_to_sentences(sentences, timeline)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, base + "_uis.json"), "w", encoding="utf-8") as f:
        json.dump({"key": base, "strategy": "uis", "sentence_info": ns},
                  f, ensure_ascii=False, indent=2)
    print(f"{base} uis: {len(seg_ids)} segs -> {int(lab.max())+1} speakers")
