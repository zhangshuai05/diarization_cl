# sup/train.py
import json, os, sys, time
from copy import deepcopy
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sup.dataset import load_all, split_files, SequenceDataset
from sup.uisrnn import UISRNN

SEED = 42

def _remap_first_appearance(labels):
    """Remap valid labels (>=0) to dense first-appearance order 0,1,2,...

    UISRNN.forward indexes states/joint by label value and treats
    y == len(states) as a new speaker, so the feed sequence must be remapped
    this way. Block-level shuffle (SequenceDataset) permutes blocks, so this
    must be done per-sequence AFTER shuffling, not on the stored npz.
    """
    out = labels.clone()
    seen = {}
    nxt = 0
    for i in range(len(labels)):
        y = int(labels[i])
        if y < 0:
            continue
        if y not in seen:
            seen[y] = nxt
            nxt += 1
        out[i] = seen[y]
    return out

def main(epochs=150, lr=0.003, batch_size=16):
    torch.manual_seed(SEED); np.random.seed(SEED)
    seqs = load_all()
    train_keys, test_keys = split_files(list(seqs.keys()))
    print(f"train files={len(train_keys)} test files={len(test_keys)}", flush=True)
    train_ds = SequenceDataset({k: seqs[k] for k in train_keys}, shuffle_blocks=True, seed=SEED)
    # 序列长度可变：collate 返回 batch 列表，逐条前向
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda b: b)
    model = UISRNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_frame_nll = float("inf")
    best_ep = -1
    best_state = None
    for ep in range(epochs):
        tot = 0.0; frames = 0
        for batch in loader:
            loss = 0.0
            for embs, labels, mask in batch:
                labels = _remap_first_appearance(labels)
                l, _, _ = model(embs, labels, mask)
                loss = loss + l
                frames += int(mask.sum().item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        frame_nll = tot / max(frames, 1)
        if frame_nll < best_frame_nll:
            best_frame_nll = frame_nll
            best_ep = ep
            best_state = deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep}: train frame_nll={frame_nll:.4f}", flush=True)
    os.makedirs(os.path.join(HERE, "output", "sup", "ckpt"), exist_ok=True)
    torch.save({"state_dict": best_state,
                "args": {"dim": model.dim, "hidden": model.gru.hidden_size,
                         "sigma": model.sigma,
                         "transition_bias": model.transition_bias,
                         "crp_alpha": model.crp_alpha}},
               os.path.join(HERE, "output", "sup", "ckpt", "model.pt"))
    print(f"saved ckpt (best epoch {best_ep}, best frame_nll {best_frame_nll:.4f})", flush=True)

if __name__ == "__main__":
    main()
