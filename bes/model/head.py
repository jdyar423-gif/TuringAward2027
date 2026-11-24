"""Output head: final norm, factorised softcapped logits and the chunked cross-entropy.

    h = rmsnorm(sum_n lanes[n]) * g_head
    z = softcap * tanh(((h @ P) @ E_out.T) / softcap)          (float32)

Softcap and the float32 logits follow kschwethelm/looped-lm-scaling ``nanochat/gpt.py`` (MIT).
The cross-entropy ``-log softmax(z)[y]`` runs over row chunks inside ``lax.scan`` with a
checkpointed body, so only one ``[chunk, V]`` float32 logit block (and its gradient) is live at
a time; the target logit is selected by compare/select (no gather, no scatter in the backward).
The speed switch :data:`FAST_CE` replaces that by a hand-written backward pass (same values to
float32 rounding; the forward writes the logits once and reads them for nothing else, the
backward never recomputes them).
The candidate tape enters afterwards through per-token terms (``bes.model.predictive``), so the
pass over the vocabulary has no candidate code path.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from bes.model.layers import act_dtype, hi, mm, rmsnorm, softcap

#: Speed switch (read at trace time; default off, i.e. the reference path below):
#: :func:`chunked_ce` through :func:`_fast_ce`, a hand-written backward pass. The log-sum-exp
#: is one pass without the row maximum (the softcap bounds ``|z| <= softcap``, so ``exp(z)``
#: neither overflows nor underflows in float32), which XLA fuses into the epilogue of the logit
#: product together with the target logit; the backward pass keeps the float32 logits
#: (``[n, V]``) and the row sums instead of recomputing the chunk, so it repeats neither the
#: logit product nor any reduction over the vocabulary. Same values to float32 rounding.
FAST_CE = False
#: largest float32 logit residual (bytes, ``4 n V``) the fast path keeps; above it the
#: reference path (one chunk live at a time) runs
FAST_CE_MAX_RESIDUAL = 1 << 31
#: largest softcap for which ``sum_v exp(z_v)`` stays finite in float32 for any vocabulary of
#: up to 2^24 ids (``exp(60) * 2^24 < 2^128``)
FAST_CE_MAX_CAP = 60.0


def final_hidden(lanes, g: jax.Array) -> jax.Array:
    """``rmsnorm(sum_n lanes[n]) * g``: lanes ``[N, B, T, d]`` or a tuple of ``N`` arrays
    ``[B, T, d]`` -> ``float32[B, T, d]``."""
    ft = hi(lanes[0].dtype)
    s = lanes[0].astype(ft)
    for n in range(1, len(lanes)):
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
    ``cfg.loss_chunk`` (the last chunk is padded). With :data:`FAST_CE` the backward pass keeps
    every chunk's float32 logits instead of recomputing them (``4 n V`` bytes, at most
    :data:`FAST_CE_MAX_RESIDUAL`; 134 MB for a training piece of 4,096 rows).
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
    cap = float(cfg.softcap or 0.0)
    if (FAST_CE and 0.0 < cap <= FAST_CE_MAX_CAP
            and 4 * n_chunks * chunk * vocab <= FAST_CE_MAX_RESIDUAL):
        ce = _fast_ce(cap, n_chunks)(xs["hp"], e_out, xs["y"])
        return ce.reshape(-1)[:n]

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


@functools.lru_cache(maxsize=None)
def _fast_ce(cap: float, n_chunks: int):
    """Cross-entropy of padded row chunks with a hand-written backward pass (:data:`FAST_CE`).

    Returns ``f(hp [C, c, r], e_out [V, r], y int32 [C, c]) -> float32 [C, c]`` (a
    ``jax.custom_vjp``; ``C = n_chunks`` static chunks, ``0 < cap <= FAST_CE_MAX_CAP``).

    Forward per chunk, as the reference body: ``z = hp e_out^T`` (float32 accumulation),
    ``t = tanh(z / cap)``, ``zc = cap t`` (``layers.softcap``, bit for bit), the target
    ``zc[y]`` by compare/select; then ``ce = log(s) - zc[y]`` with ``s = sum_v exp(zc)``
    unshifted (``|zc| <= cap``; the reference subtracts the row maximum first, which needs a
    second pass over ``zc``). Residuals: ``t`` (float32 ``[c, V]``; XLA keeps the logits
    themselves and re-applies the ``tanh``) and ``s``. Backward: the cotangent of ``zc``,
    ``dzc = where(v = y, -g, 0) + (g / s) exp(zc)``, as autodiff of the reference forms it,
    times the softcap derivative ``(1 - t)(1 + t)``; ``d e_out = dz^T hp`` and
    ``d hp = dz e_out`` as the transposes of the logit product (float32 accumulation, cast to
    the operand dtype). The reference and this path differ by the rounding of the unshifted
    sum, of ``exp(zc) / s`` and of the derivative product (autodiff multiplies by ``cap``,
    forms ``a (1 + t)`` as ``a + a t`` and divides by ``cap``), and, with more than one chunk,
    by ``d e_out``: summed over the chunks in float32 and cast once, where the transposed scan
    of the reference rounds the running sum to the operand dtype after every chunk (with
    bfloat16 activations one rounding instead of ``C``; with one chunk, i.e. up to
    ``loss_chunk`` rows as in every training piece, the same operation).
    """

    def chunk_fwd(hp_i, e_out, y_i):
        z = jnp.einsum("nr,vr->nv", hp_i, e_out, preferred_element_type=hi(hp_i.dtype))
        t = jnp.tanh(z / cap)
        zc = cap * t
        s = jnp.sum(jnp.exp(zc), axis=-1)
        iota = jnp.arange(e_out.shape[0], dtype=jnp.int32)
        tgt = jnp.sum(jnp.where(iota == y_i[:, None], zc, 0.0), axis=-1)
        return jnp.log(s) - tgt, t, s

    def forward(hp, e_out, y):
        outs = [chunk_fwd(hp[i], e_out, y[i]) for i in range(n_chunks)]
        return (jnp.stack([o[0] for o in outs]), tuple(o[1] for o in outs),
                tuple(o[2] for o in outs))

    @jax.custom_vjp
    def ce(hp, e_out, y):
        return forward(hp, e_out, y)[0]

    def ce_fwd(hp, e_out, y):
        out, ts, ss = forward(hp, e_out, y)
        return out, (hp, e_out, y, ts, ss)

    def ce_bwd(res, g):
        hp, e_out, y, ts, ss = res
        iota = jnp.arange(e_out.shape[0], dtype=jnp.int32)
        d_hp, d_e = [], None
        for i in range(n_chunks):
            t, s, g_i = ts[i], ss[i], g[i]
            dzc = (jnp.where(iota == y[i][:, None], -g_i[:, None], 0.0)
                   + (g_i / s)[:, None] * jnp.exp(cap * t))
            dz = dzc * ((1.0 - t) * (1.0 + t))
            part = jax.lax.dot_general(dz, hp[i], (((0,), (0,)), ((), ())),
                                       preferred_element_type=t.dtype)
            d_e = part if d_e is None else d_e + part
            d_hp.append(jax.lax.dot_general(dz, e_out, (((1,), (0,)), ((), ())),
                                            preferred_element_type=t.dtype).astype(hp.dtype))
        return jnp.stack(d_hp), d_e.astype(e_out.dtype), None

    ce.defvjp(ce_fwd, ce_bwd)
    return ce
