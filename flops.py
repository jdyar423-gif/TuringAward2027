"""Conservative FLOP accounting.

Total training FLOPs =
    measured matmul FLOPs of forward+backward (torch.utils.flop_counter)
  + attention FLOPs (QK^T and AV at full T x T, i.e. *not* discounted for the causal mask;
    added analytically because the CPU flash-attention kernel is invisible to the counter)
  + elementwise ops (norms, activations, softmax, rotary, residuals, gates, loss, table lookups,
    gradient scatter-adds; generous upper-bound estimate)
  + optimizer FLOPs (Newton-Schulz / Polar Express matmuls counted exactly; per-parameter
    elementwise updates for every dense parameter; per-touched-row cost for sparse tables)
  + EMA weight averaging updates.
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def measure_matmul_flops(model, B, T):
    """Exact matmul (+attention) FLOPs of one forward+backward on a (B,T) batch."""
    cfg = model.cfg
    idx = torch.randint(0, cfg.vocab - 1, (B, T))
    tgt = torch.randint(0, cfg.vocab - 1, (B, T))
    ng = None
    if cfg.ngram_orders:
        ng = torch.stack([torch.randint(0, cfg.ngram_buckets if j % 2 == 0 else 8192, (B, T))
                          for j in range(2 * len(cfg.ngram_orders))], -1)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        loss = model(idx, tgt, ng)
        loss.backward()
    model.zero_grad(set_to_none=True)
    total = fc.get_total_flops()
    ops = {str(k) for k in fc.get_flop_counts()["Global"]}
    if not any("attention" in o for o in ops):
        total += 3 * cfg.n_layer * B * 2 * (2 * T * T * cfg.d)
    return total


def elementwise_flops_per_token(cfg, T):
    """Generous upper bound on non-matmul forward ops per token, times 3 for fwd+bwd."""
    d, h, H, L, V = cfg.d, cfg.hidden, cfg.n_head, cfg.n_layer, cfg.vocab
    per_layer = 64 * d + 4 * h + 6 * T * H + (2 * 2 * cfg.conv * d + 4 * d if cfg.conv else 0)
    tables = 1 + len(cfg.ngram_orders) + cfg.value_embeds
    fwd = L * per_layer + 10 * V + 4 * d * tables + 16 * d + (8 * d if cfg.smear else 0)
    return 3 * fwd


def muon_flops(shape, ns_steps):
    from optim import use_gram
    m, n = sorted(shape)
    if use_gram(m, n, ns_steps):
        orth = 4 * m * m * n + (8 * ns_steps - 6) * m ** 3 + ns_steps * 8 * m * m
    else:
        orth = ns_steps * (4 * m * m * n + 2 * m ** 3 + 2 * m * m + 3 * m * n)
    return orth + 25 * m * n  # momentum, nesterov, normalisation, NorMuon, wd, update


ADAM_FLOPS_PER_PARAM = 15


def sparse_table_flops(cfg, tokens):
    """Sparse row-Adam on the n-gram tables: at most `tokens` touched rows per table."""
    return len(cfg.ngram_orders) * tokens * (4 * cfg.d + 12)


def analytic_matmul_flops_per_token(cfg, T):
    """Forward matmul FLOPs per token (cross-check for the measurement)."""
    d, h, L, V, H = cfg.d, cfg.hidden, cfg.n_layer, cfg.vocab, cfg.n_head
    up = 2 * h if cfg.act == "swiglu" else h
    per = 2 * d * 3 * d + 2 * d * d + 2 * d * up + 2 * h * d + 4 * T * d
    per += {"sparse": 2 * 12 * H, "head": 2 * d * H, "elem": 2 * d * d}.get(cfg.attn_gate, 0)
    return L * per + 2 * d * V + (24 if cfg.smear else 0)
