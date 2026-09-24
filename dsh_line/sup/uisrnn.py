# sup/uisrnn.py
# UIS-RNN (Unbounded Interleaved-State Recurrent Neural Network) — PyTorch.
#
# Hyperparameters (FINAL, after toy-data convergence check — brief Step 3):
#   dim=192, hidden=128, sigma=0.3 (changed from the brief's 0.5),
#   transition_bias=0.0, crp_alpha=1.0.
#   With sigma=0.5 the toy tests fail: the new-speaker score log(crp_alpha)=0 is
#   compared against existing-speaker scores that include the Gaussian emission
#   normalization -0.5*d*log(2*pi*sigma^2) ~ -43 nats, so "new speaker" always
#   wins and the model never separates. At sigma=0.3 that constant flips sign
#   (~+54 nats), restoring a sane scale. transition_bias=2.0 alone (sigma=0.5)
#   does NOT fix it (acc 0.0067, 150 speakers). Final: acc=1.0, loss halves.
#
# 注意事项 guard: in forward(), when labels[t] < 0 (unknown label from a
# masked-out segment), record NO loss, do NOT update any GRU state, and do NOT
# update `prev` — the state is simply carried over unchanged.
#
# forward() returns (total, nll_emission, nll_trans) as three 0-dim tensors:
# per masked frame the joint NLL -(joint[y]-Z_joint) feeds `total`; the emission
# and transition components are standalone diagnostics -(emit[y]-Z_emit) and
# -(trans[y]-Z_trans) with their OWN normalizers — NOT an additive decomposition
# of total. `total` is the joint NLL; nll_emission/nll_trans are per-component
# diagnostics only.
import math
import torch
import torch.nn as nn

class UISRNN(nn.Module):
    def __init__(self, dim=192, hidden=128, sigma=0.3,
                 transition_bias=0.0, crp_alpha=1.0):
        super().__init__()
        self.dim = dim
        self.gru = nn.GRUCell(dim, hidden)
        self.fc = nn.Linear(hidden, dim)
        self.sigma = sigma
        self.transition_bias = transition_bias
        self.crp_alpha = crp_alpha
        # 新说话人发射先验：全局高斯 N(e; prior_mean, exp(prior_logvar) * I)
        # 初始化 prior_logvar = log(sigma^2)，使新说话人分支与既有说话人同尺度
        self.prior_mean = nn.Parameter(torch.zeros(dim))
        self.prior_logvar = nn.Parameter(torch.tensor(math.log(sigma ** 2)))

    def _emission_logp(self, e, mu):
        d = self.dim
        diff = e - mu
        return -0.5 * (diff * diff).sum(-1) / (self.sigma ** 2) \
               - 0.5 * d * math.log(2 * math.pi * self.sigma ** 2)

    def _prior_emission_logp(self, e):
        """新说话人分支的全局高斯发射对数概率：N(e; prior_mean, exp(prior_logvar) * I)。"""
        var = torch.exp(self.prior_logvar)
        diff = e - self.prior_mean
        return -0.5 * (diff * diff).sum() / var \
               - 0.5 * self.dim * (self.prior_logvar + math.log(2 * math.pi))

    def _step_logp(self, e, states, counts, prev):
        """Return (joint, emit, trans) per candidate, with joint = emit + trans.

        emit[k]  = Gaussian emission logp for existing speaker k / global-Gaussian prior for new
        trans[k] = log(n_k) + transition_bias*[k==prev] for existing / log(crp_alpha) for new
        """
        joint, emit, trans = [], [], []
        for k, h in enumerate(states):
            mu = self.fc(h)
            e_lp = self._emission_logp(e, mu)
            t_lp = torch.tensor(
                math.log(counts[k]) + (self.transition_bias if k == prev else 0.0),
                dtype=e.dtype, device=e.device)
            emit.append(e_lp)
            trans.append(t_lp)
            joint.append(e_lp + t_lp)
        # 新说话人：发射先验为全局高斯 N(e; prior_mean, exp(prior_logvar) I)，转移为 log(crp_alpha)
        e_lp_new = self._prior_emission_logp(e)
        t_lp_new = torch.tensor(math.log(self.crp_alpha), dtype=e.dtype, device=e.device)
        emit.append(e_lp_new)
        trans.append(t_lp_new)
        joint.append(e_lp_new + t_lp_new)
        return joint, emit, trans

    def forward(self, embs, labels, mask):
        # 前置条件：labels 必须为密集 0..K-1 首现顺序（新说话人 = 当前 len(states)）
        T = embs.shape[0]
        states, counts, prev = [], [], -1
        total = torch.zeros((), dtype=embs.dtype, device=embs.device)
        n_emit = torch.zeros((), dtype=embs.dtype, device=embs.device)
        n_trans = torch.zeros((), dtype=embs.dtype, device=embs.device)
        for t in range(T):
            e = embs[t]
            y = int(labels[t])
            if y < 0:
                # unknown label (masked-out segment): no loss, no GRU/prev update
                continue
            if y > len(states):
                raise ValueError(f"label {y} out of range at step {t} (labels must be dense 0..K-1 in first-appearance order, new speaker = current count {len(states)})")
            joint, emit, trans = self._step_logp(e, states, counts, prev)
            # total 为联合 NLL；emit/trans 为各自归一化下的独立分量诊断（非 total 的可加分解）
            Z_joint = torch.logsumexp(torch.stack(joint), dim=0)
            Z_emit = torch.logsumexp(torch.stack(emit), dim=0)
            Z_trans = torch.logsumexp(torch.stack(trans), dim=0)
            if mask[t] > 0:
                total = total + (-(joint[y] - Z_joint))
                n_emit = n_emit + (-(emit[y] - Z_emit))
                n_trans = n_trans + (-(trans[y] - Z_trans))
            # 状态更新（训练期按真值标签走）
            if y == len(states):
                h = self.gru(e[None], torch.zeros(1, self.gru.hidden_size,
                                                  device=e.device))[0]
                states.append(h)
                counts.append(1)
            else:
                states[y] = self.gru(e[None], states[y][None])[0]
                counts[y] += 1
            prev = y
        return total, n_emit, n_trans

    @torch.no_grad()
    def infer(self, embs):
        T = embs.shape[0]
        states, counts, prev, out = [], [], -1, []
        for t in range(T):
            e = embs[t]
            joint, _, _ = self._step_logp(e, states, counts, prev)
            k = int(torch.argmax(torch.stack(joint)))
            out.append(k)
            if k == len(states):
                h = self.gru(e[None], torch.zeros(1, self.gru.hidden_size,
                                                  device=e.device))[0]
                states.append(h)
                counts.append(1)
            else:
                states[k] = self.gru(e[None], states[k][None])[0]
                counts[k] += 1
            prev = k
        return torch.tensor(out, dtype=torch.long)
