# sup/test_dataset.py
import numpy as np
from sup.dataset import SequenceDataset, split_files

def test_dataset_len_and_shuffle_shape():
    seqs = {
        "a": {"embs": np.zeros((10, 192), np.float32),
              "labels": np.array([0]*5 + [1]*5, np.int64),
              "mask": np.ones(10, np.float32)},
    }
    ds = SequenceDataset(seqs, shuffle_blocks=True, seed=0)
    e, l, m = ds[0]
    assert e.shape == (10, 192) and l.shape == (10,) and m.shape == (10,)

def test_split_files_holds_out():
    train, test = split_files(["a", "b", "Call20260718210454"])
    assert "Call20260718210454" in test and "a" in train
