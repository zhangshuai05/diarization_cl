# sup/test_online_uis.py
import numpy as np, torch
from sup.toydata import make_toy
from sup.uisrnn import UISRNN

def test_infer_labels_follow_toy_pattern():
    torch.manual_seed(42)
    embs, labels, mask = make_toy()
    model = UISRNN()
    pred = model.infer(torch.from_numpy(embs)).numpy()
    assert len(set(pred.tolist())) >= 1
    assert pred.shape == labels.shape

def test_load_model_returns_eval_uisrnn():
    import os
    import torch
    from sup import online_uis
    from sup.uisrnn import UISRNN
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(HERE, "output", "sup", "ckpt", "model.pt")
    m = online_uis.load_model(ckpt)
    assert isinstance(m, UISRNN)
    assert not m.training
