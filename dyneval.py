"""Score-first dynamic evaluation (test-time training).

The stream is processed left to right in chunks of S new tokens.  For each
chunk the model first *scores* the S targets (that log-likelihood is what we
report), and only afterwards takes one gradient step on those same, already
scored, tokens.  No prediction ever depends on its own target.

FLOP accounting: the scoring forward is evaluation (as in any eval), but the
backward pass, the optimizer update and the elementwise work are training and
are charged to the global budget.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from flops import elementwise_flops_per_token


def adapted_params(model, which):
    out = []
    for n, p in model.named_parameters():
        if which == "all":
            ok = True
        elif which == "noemb":
            ok = not (n.startswith("emb") or n.startswith("ngram") or n.startswith("ve."))
        elif which == "dense":  # everything except the sparse n-gram tables
            ok = not n.startswith("ngram")
        elif which == "head":
            ok = n.startswith("head")
        elif which.startswith("top"):  # last k blocks + head
            k = int(which[3:])
            L = model.cfg.n_layer
            ok = n.startswith("head") or any(n.startswith(f"blocks.{i}.") for i in range(L - k, L))
        else:
            raise ValueError(which)
        p.requires_grad_(ok)
        if ok:
            out.append((n, p))
    return out


def backward_flops(model, T, S):
    """Measured matmul FLOPs of backward only (fwd+bwd minus fwd) for one window, plus attention."""
    cfg = model.cfg
    idx = torch.randint(0, cfg.vocab - 1, (1, T))
    ng = None
    if cfg.ngram_orders:
        ng = torch.stack([torch.randint(0, cfg.ngram_buckets if j % 2 == 0 else 8192, (1, T))
                          for j in range(2 * len(cfg.ngram_orders))], -1)
    with FlopCounterMode(display=False) as f1:
        z = model.logits(model.hidden(idx, ng))
    with FlopCounterMode(display=False) as f2:
        z = model.logits(model.hidden(idx, ng))
        F.cross_entropy(z[0, -S:].float(), idx[0, -S:]).backward()
    model.zero_grad(set_to_none=True)
    bwd = f2.get_total_flops() - f1.get_total_flops()
    bwd += 2 * cfg.n_layer * 2 * (2 * T * T * cfg.d)  # attention backward (not seen by the counter)
    return bwd + 2 * elementwise_flops_per_token(cfg, T) // 3 * T


def dynamic_eval(model, toks, ng, T, S, lr, which="all", opt="sgd", beta2=0.99, decay=0.0,
                 budget=float("inf"), U=0):
    """Score-first dynamic evaluation.

    Targets are processed in update-chunks of U tokens.  Inside a chunk every target is scored with
    the *current* weights, using sliding windows of length T that advance by S (forward only).  The
    last window of the chunk ends exactly at the chunk end; its forward is reused for one gradient
    step on the chunk's (already scored) targets.  Thus the weights used to score target t were
    trained only on targets < t.  U = 0 -> U = S (classic dynamic evaluation).

    Returns (stats (3, N-1): nll, entropy, max log-prob per target; FLOPs charged to the budget).
    """
    U = U or S
    assert U % S == 0 and U <= T
    model.eval()  # no dropout; gradients still flow
    params = adapted_params(model, which)
    plist = [p for _, p in params]
    init = [p.detach().clone() for p in plist] if decay else None
    state = [torch.zeros_like(p) for p in plist] if opt == "rms" else None
    upd = (8 if opt == "rms" else 3) + (3 if decay else 0)
    n_dense = sum(p.numel() for n, p in params if not n.startswith("ngram"))
    per_update = backward_flops(model, T, U) + upd * n_dense
    per_update += len(model.cfg.ngram_orders) * T * model.cfg.d * 12  # touched rows of sparse tables
    N = toks.numel()
    out = torch.empty(3, N - 1)
    used, a = 0.0, 0

    def window(e):
        s0 = max(0, e - T)
        g = ng[s0:e][None] if ng is not None else None
        return s0, toks[s0:e][None], toks[s0 + 1:e + 1][None], g

    def record(z, y, lo, a_, e_):
        with torch.no_grad():
            lp = F.log_softmax(z[lo:].detach(), -1)
            out[0, a_:e_] = -lp.gather(-1, y[lo:, None])[:, 0]
            out[1, a_:e_] = -(lp.exp() * lp).sum(-1)
            out[2, a_:e_] = lp.max(-1).values

    while a < N - 1:
        e = min(a + U, N - 1)
        b = a
        while b < e:                      # score targets b .. b2-1
            b2 = min(b + S, e)
            last = b2 == e
            s0, x, y, g = window(b2)
            with torch.set_grad_enabled(last and used + per_update <= budget):
                z = model.logits(model.hidden(x, g)).float()[0]
            record(z, y[0], b - s0, b, b2)
            b = b2
        if z.requires_grad:               # one step on the chunk's already-scored targets
            lo = max(0, a - s0)
            F.cross_entropy(z[lo:], y[0, lo:]).backward()
            with torch.no_grad():
                for i, p in enumerate(plist):
                    gr = p.grad
                    if gr is None:
                        continue
                    if gr.is_sparse:
                        gr = gr.coalesce()
                        p.index_add_(0, gr.indices()[0], gr.values(), alpha=-lr)
                        continue
                    if opt == "rms":
                        state[i].mul_(beta2).addcmul_(gr, gr, value=1 - beta2)
                        p.addcdiv_(gr, state[i].sqrt().add_(1e-8), value=-lr)
                    else:
                        p.add_(gr, alpha=-lr)
                    if decay:
                        p.lerp_(init[i], decay)
            model.zero_grad(set_to_none=True)
            used += per_update
        a = e
    for p in model.parameters():
        p.requires_grad_(True)
    return out.numpy(), used
