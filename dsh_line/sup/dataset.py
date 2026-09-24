# sup/dataset.py
import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQDIR = os.path.join(HERE, "output", "sup", "seq")

TEST_FILES = ["Call20260718210454", "Call20260718211745",
              "Note20260803111436", "Note20260803112655",
              "Call20260718211938"]

def load_all(bases=None):
    out = {}
    paths = sorted(glob.glob(os.path.join(SEQDIR, "*.npz")))
    for p in paths:
        base = os.path.basename(p)[:-4]
        if bases and base not in bases:
            continue
        z = np.load(p)
        out[base] = {"embs": z["embs"], "labels": z["labels"], "mask": z["mask"]}
    return out

def split_files(keys, test_files=None, seed=42):
    test_files = test_files or TEST_FILES
    train = [k for k in keys if k not in test_files]
    test = [k for k in test_files if k in keys]
    return train, test

class SequenceDataset(Dataset):
    def __init__(self, seqs, shuffle_blocks=True, seed=42):
        self.seqs = list(seqs.values())
        self.rng = np.random.default_rng(seed)
        self.shuffle_blocks = shuffle_blocks
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        e, l, m = s["embs"], s["labels"], s["mask"]
        if self.shuffle_blocks:
            # 块级重排：以连续同标签块为单位重排（论文数据增强）
            bounds = [0]
            for t in range(1, len(l)):
                if l[t] != l[t-1]:
                    bounds.append(t)
            bounds.append(len(l))
            blocks = list(zip(bounds[:-1], bounds[1:]))
            self.rng.shuffle(blocks)
            idx = np.concatenate([np.arange(a, b) for a, b in blocks])
            e, l, m = e[idx], l[idx], m[idx]
        return (torch.from_numpy(e), torch.from_numpy(l), torch.from_numpy(m))
