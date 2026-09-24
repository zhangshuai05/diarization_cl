import numpy as np
from sup.toydata import make_toy

def test_shape_and_pattern():
    embs, labels, mask = make_toy()
    assert embs.shape[0] == 150 and embs.shape[1] == 192
    assert mask.sum() == 150
    assert set(labels.tolist()) == {0, 1}
    # 块结构：每 3 段换人
    assert (labels[0:3] == 0).all() and labels[3] == 1

def test_reproducible():
    a = make_toy()[0]
    b = make_toy()[0]
    assert np.allclose(a, b)

def test_center_separation_in_range():
    embs, labels, _ = make_toy()
    c0 = embs[labels == 0].mean(0); c0 /= np.linalg.norm(c0)
    c1 = embs[labels == 1].mean(0); c1 /= np.linalg.norm(c1)
    sim = float(c0 @ c1)
    assert 0.25 <= sim <= 0.45
