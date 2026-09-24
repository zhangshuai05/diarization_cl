import numpy as np

def make_toy(n_speakers=2, per_speaker_blocks=25, block_len=3, dim=192, seed=42):
    rng = np.random.default_rng(seed)
    # 构造两两余弦约 0.35 的单位中心（模拟真实数据类间重叠度）
    centers = []
    for k in range(n_speakers):
        if k == 0:
            c = rng.standard_normal(dim)
        else:
            c = rng.standard_normal(dim)
            # 与已有中心保持目标相似度 0.35 附近的混合
            mix = sum(centers)
            c = 0.35 * (mix / (np.linalg.norm(mix) + 1e-9)) + 0.94 * (c / (np.linalg.norm(c) + 1e-9))
        centers.append(c / (np.linalg.norm(c) + 1e-9))
    embs, labels = [], []
    for i in range(n_speakers * per_speaker_blocks):
        k = i % n_speakers
        for _ in range(block_len):
            e = centers[k] + 0.25 * rng.standard_normal(dim)
            embs.append(e / (np.linalg.norm(e) + 1e-9))
            labels.append(k)
    T = len(embs)
    return (np.stack(embs).astype(np.float32),
            np.array(labels, dtype=np.int64),
            np.ones(T, dtype=np.float32))
