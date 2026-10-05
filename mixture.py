"""Neural LM + count-based experts, mixed by a small context-only gate."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from countmem import features


@torch.no_grad()
def token_stats(model, toks, ng, T, stride, bs=16):
    """Sliding-window per-target NLL (nats), predictive entropy and max log-prob for toks[1:]."""
    was = model.training
    model.eval()
    N = toks.numel()
    T = min(T, N - 1)
    out = torch.empty(3, N - 1)
    starts = list(range(0, N - 1 - T + 1, stride))
    if starts[-1] + T < N - 1:
        starts.append(N - 1 - T)
    covered = 0
    for i in range(0, len(starts), bs):
        sb = starts[i:i + bs]
        x = torch.stack([toks[s:s + T] for s in sb])
        y = torch.stack([toks[s + 1:s + T + 1] for s in sb])
        g = torch.stack([ng[s:s + T] for s in sb]) if ng is not None else None
        lp = F.log_softmax(model.logits(model.hidden(x, g)).float(), -1)
        nll = -lp.gather(-1, y[..., None])[..., 0]
        ent = -(lp.exp() * lp).sum(-1)
        mx = lp.max(-1).values
        for j, s in enumerate(sb):
            lo = covered - s
            out[0, covered:s + T] = nll[j, lo:]
            out[1, covered:s + T] = ent[j, lo:]
            out[2, covered:s + T] = mx[j, lo:]
            covered = s + T
    model.train(was)
    return out.numpy()


class Gate(nn.Module):
    def __init__(self, nfeat, nexp, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nfeat, hidden), nn.ReLU(), nn.Linear(hidden, nexp))
        self.register_buffer("mu", torch.zeros(nfeat))
        self.register_buffer("sd", torch.ones(nfeat))

    def logw(self, feat, avail):
        z = self.net((feat - self.mu) / self.sd)
        return F.log_softmax(z.masked_fill(~avail, -1e9), -1)

    def nll(self, feat, P, avail):
        """-log sum_e w_e P_e (nats) per token."""
        return -torch.logsumexp(self.logw(feat, avail) + torch.log(P.clamp_min(1e-30)), -1)

    def flops_per_token(self):
        n = sum(p.numel() for p in self.net.parameters())
        return 2 * n + 10 * self.net[-1].out_features


def assemble(nn_stats, cm_stats):
    """Gate inputs. Expert 0 = neural LM; then global n-gram orders; then doc-cache orders."""
    Fc, Pc, Ac = features(cm_stats)
    nll, ent, mx = nn_stats
    feat = np.concatenate([Fc, ent[:, None], mx[:, None]], 1).astype(np.float32)
    P = np.concatenate([np.exp(-nll)[:, None], Pc], 1).astype(np.float32)
    avail = np.concatenate([np.ones((len(nll), 1), dtype=bool), Ac], 1)
    return torch.from_numpy(feat), torch.from_numpy(P), torch.from_numpy(avail)


def fit_gate(feat, P, avail, steps=400, lr=0.01, hidden=64, seed=0, verbose=False):
    torch.manual_seed(seed)
    g = Gate(feat.size(1), P.size(1), hidden)
    g.mu.copy_(feat.mean(0))
    g.sd.copy_(feat.std(0).clamp_min(1e-3))
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    for i in range(steps):
        loss = g.nll(feat, P, avail).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and i % 100 == 0:
            print(i, loss.item())
    # FLOPs: full-batch forward+backward per step
    flops = steps * 3 * g.flops_per_token() * feat.size(0)
    return g, flops
