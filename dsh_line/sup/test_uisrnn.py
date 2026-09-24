# sup/test_uisrnn.py
import numpy as np
import pytest
import torch
from sup.toydata import make_toy
from sup.uisrnn import UISRNN

def _train_toy():
    torch.manual_seed(42); np.random.seed(42)
    embs, labels, mask = make_toy()
    E = torch.from_numpy(embs); L = torch.from_numpy(labels); M = torch.from_numpy(mask)
    model = UISRNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for it in range(400):
        opt.zero_grad()
        loss, _, _ = model(E, L, M)
        loss.backward()
        opt.step()
    return model, E

def test_toy_inference_accuracy():
    model, E = _train_toy()
    pred = model.infer(E).numpy()
    labels = make_toy()[1]
    # 允许标签置换：用映射后的准确率
    acc = max(
        (pred == labels).mean(),
        (pred == 1 - labels).mean(),
    )
    assert acc > 0.9, f"toy accuracy too low: {acc}"

def test_loss_decreases():
    torch.manual_seed(42); np.random.seed(42)
    embs, labels, mask = make_toy()
    E = torch.from_numpy(embs); L = torch.from_numpy(labels); M = torch.from_numpy(mask)
    model = UISRNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for it in range(200):
        opt.zero_grad()
        loss, _, _ = model(E, L, M)
        loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5

def test_forward_raises_on_out_of_range_label():
    torch.manual_seed(42); np.random.seed(42)
    embs, _, mask = make_toy()  # 2-speaker toy (labels 0/1)
    E = torch.from_numpy(embs); M = torch.from_numpy(mask)
    model = UISRNN()
    L_bad = torch.tensor([0, 1, 7])  # 7 超出密集 0..1 范围
    with pytest.raises(ValueError, match="out of range"):
        model(E[:3], L_bad, M[:3])

def test_unknown_label_guard_equals_deleted_frame():
    torch.manual_seed(42); np.random.seed(42)
    embs, labels, mask = make_toy()
    E = torch.from_numpy(embs); L = torch.from_numpy(labels); M = torch.from_numpy(mask)
    model = UISRNN()
    # 在第 10 帧插入 -1 标签（掩码 0），对照为删除该帧的序列
    L2 = torch.cat([L[:10], torch.tensor([-1]), L[10:]])
    M2 = torch.cat([M[:10], torch.tensor([0.0]), M[10:]])
    E2 = torch.cat([E[:10], E[9:10], E[10:]])  # 帧内容随意，损失只看标签路径
    loss1, _, _ = model(E, L, M)
    loss2, _, _ = model(E2, L2, M2)
    assert torch.allclose(loss1, loss2, atol=1e-5)
