"""Output head: final norm, factorised softcapped logits and the chunked cross-entropy.

    h = rmsnorm(sum_n lanes[n]) * g_head
    z = softcap * tanh(((h @ P) @ E_out.T) / softcap)          (float32)

Softcap and the float32 logits follow kschwethelm/looped-lm-scaling ``nanochat/gpt.py`` (MIT).
The cross-entropy ``-log softmax(z)[y]`` runs over row chunks inside ``lax.scan`` with a
checkpointed body, so only one ``[chunk, V]`` float32 logit block (and its gradient) is live at
a time; the target logit is selected by compare/select (no gather, no scatter in the backward).
The candidate tape enters afterwards through per-token terms (``bes.model.predictive``), so the
pass over the vocabulary has no candidate code path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bes.model.layers import act_dtype, hi, mm, rmsnorm, softcap


def final_hidden(lanes: jax.Array, g: jax.Array) -> jax.Array:
    """``rmsnorm(sum_n lanes[n]) * g``: lanes ``[N, B, T, d]`` -> ``float32[B, T, d]``."""
    ft = hi(lanes.dtype)
    s = lanes[0].astype(ft)
    for n in range(1, lanes.shape[0]):
        s = s + lanes[n].astype(ft)
    return rmsnorm(s, g, out_dtype=ft)


def logits(ph: dict, h: jax.Array, cfg) -> jax.Array:
    """Softcapped logits ``float32[..., V]`` of the network alone. ph: ``{"proj": [d, r],
    "out": [V, r]}``."""
    dt = act_dtype(cfg)
    hp = mm(h, ph["proj"], dt, out_dtype=dt)
    z = jnp.einsum("...r,vr->...v", hp, ph["out"].astype(dt), preferred_element_type=hi(dt))
    return softcap(z, cfg.softcap)


def chunked_ce(ph: dict, h: jax.Array, y: jax.Array, cfg) -> jax.Array:
    """Per-token cross-entropy ``-log softmax(z)[y]`` of the network: ``float32[n]``.

    h: ``[n, d]`` final hidden; y: int ``[n]`` targets. Rows are processed in chunks of
    ``cfg.loss_chunk`` (the last chunk is padded).
    """
    dt = act_dtype(cfg)
    n = h.shape[0]
    hp = mm(h, ph["proj"], dt, out_dtype=dt)                      # [n, r]
    e_out = ph["out"].astype(dt)                                  # [V, r]
    vocab = e_out.shape[0]
    chunk = max(1, min(int(cfg.loss_chunk), n))
    n_chunks = -(-n // chunk)
    pad = n_chunks * chunk - n

    def pad_rows(a, value):
        if not pad:
            return a
        widths = ((0, pad),) + ((0, 0),) * (a.ndim - 1)
        return jnp.pad(a, widths, constant_values=value)

    xs = {"hp": pad_rows(hp, 0), "y": pad_rows(y.astype(jnp.int32), 0)}
    xs = jax.tree.map(lambda a: a.reshape((n_chunks, chunk) + a.shape[1:]), xs)

    def body(carry, xc):
        z = jnp.einsum("nr,vr->nv", xc["hp"], e_out, preferred_element_type=hi(dt))
        z = softcap(z, cfg.softcap)
        lse = jax.nn.logsumexp(z, axis=-1)
        iota = jnp.arange(vocab, dtype=jnp.int32)
        tgt = jnp.sum(jnp.where(iota == xc["y"][:, None], z, 0.0), axis=-1)
        return carry, lse - tgt

    body = jax.checkpoint(body, prevent_cse=False)
    _, ce = jax.lax.scan(body, None, xs)
    return ce.reshape(-1)[:n]
