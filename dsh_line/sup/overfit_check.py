# sup/overfit_check.py
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from collections import Counter
from sup.uisrnn import UISRNN
from sup.dataset import load_all, split_files

def permutation_acc(pred, labels, mask):
    m = mask > 0
    p, l = pred[m].numpy(), labels[m].numpy()
    mapping = {}
    for k in set(l.tolist()):
        mapping[k] = Counter(p[l == k].tolist()).most_common(1)[0][0]
    return float(np.mean([p[i] == mapping[l[i]] for i in range(len(l))]))

def main():
    seqs = load_all()
    train_keys, _ = split_files(list(seqs.keys()))
    ck = torch.load(os.path.join(HERE, "output", "sup", "ckpt", "model.pt"),
                    map_location="cpu")
    m = UISRNN(**ck["args"]); m.load_state_dict(ck["state_dict"]); m.eval()
    accs = []
    for k in train_keys:
        s = seqs[k]
        with torch.no_grad():
            pred = m.infer(torch.from_numpy(s["embs"]))
        a = permutation_acc(pred, torch.from_numpy(s["labels"]),
                            torch.from_numpy(s["mask"]))
        accs.append(a)
        print(f"{k}: acc={a:.3f}")
    print(f"mean train acc={np.mean(accs):.3f}")

if __name__ == "__main__":
    main()
