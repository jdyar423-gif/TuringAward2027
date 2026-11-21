"""Multi-head attention with exact key biases, calibrated sink and confidence dim.

Ported in part (MIT licence):
* kschwethelm/looped-lm-scaling ``nanochat/gpt.py``: bias-free q/k/v/o projections, rotary
  embedding then QK normalisation, heads handled by reshapes;
* karpathy/nanochat ``nanochat/gpt.py``: the value-embedding gate
  ``v += 3 * sigmoid(W_g x[..., :12]) * ve`` with one gate per value-embedding head.

Differences required by the specification: the content part of q and k has
``dc = head_dim - bias_dims`` dims, is L2-normalised and the query is scaled by a learned
per-head gain (logit = ``gain * cos``); the reserved dims carry exact per-key biases (see
``bes.model.sinks``); the mask is causal and same-segment.

Implementations (``cfg.attn_impl``):
* ``"xla"``: dense masked softmax, any backend (CPU tests);
* ``"splash"``: provided by the TPU track in ``bes.model.splash`` as
  ``attend(q, k, v, seg, cfg) -> y`` with ``q, k, v, y: [B, T, H, head_dim]`` and the same
  semantics (static causal mask + per-example segment ids). Imported lazily so CPU runs never
  touch it.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bes.model import sinks
from bes.model.layers import F32, act_dtype, hi, l2norm, mm
from bes.model.rope_clocks import rotate

VE_GATE_CHANNELS = 12   # nanochat: the gate reads the first 12 channels of the block input
NEG_INF = float(jnp.finfo(jnp.float32).min)


class Ctx(NamedTuple):
    """Per-forward attention context shared by all loops and blocks.

    cos, sin: ``float32[B, T, 1, P]`` rotary tables; sink: ``float32[B, T]`` sink indicator or
    None (sinks statically off); conf: 0/1 confidence switch or None (statically off);
    seg, pos: ``int32[B, T]`` segment ids and positions used by the mask.
    """
    cos: jax.Array
    sin: jax.Array
    sink: jax.Array | None
    conf: jax.Array | None
    seg: jax.Array
    pos: jax.Array


def qkv(pa: dict, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg):
    """Queries, keys and values of one block.

    pa: per-block attention parameters ``wq [d, H*dc]``, ``wk [d, H*dc]``, ``wv [d, H*hd]``,
    ``q_gain [H]`` and, when present, ``sink_logit [H]``, ``sink_value [H, hd-1]``,
    ``ve_gate [12, n_ve]``. x: ``[B, T, d]`` normalised block input (activation dtype).
    ve: ``[B, T, n_ve * hd]`` value embedding of this loop (block 0 only) or None.
    Returns ``(q, k, v)``, each ``[B, T, H, hd]`` in the activation dtype, reserved dims filled.
    """
    dt = act_dtype(cfg)
    ft = hi(dt)
    b, t, _ = x.shape
    h, hd = cfg.n_heads, cfg.head_dim
    dc = hd - cfg.bias_dims
    q = mm(x, pa["wq"], dt).reshape(b, t, h, dc)
    k = mm(x, pa["wk"], dt).reshape(b, t, h, dc)
    v = mm(x, pa["wv"], dt).reshape(b, t, h, hd)
    if ve is not None:
        n_ve = ve.shape[-1] // hd
        gate_in = x[..., :VE_GATE_CHANNELS].astype(ft)
        gate = 3.0 * jax.nn.sigmoid(gate_in @ pa["ve_gate"].astype(ft))
        add = ve.astype(ft).reshape(b, t, n_ve, hd) * gate[..., None]
        v = v + jnp.pad(add, ((0, 0), (0, 0), (0, h - n_ve), (0, 0)))
    q = l2norm(rotate(q, ctx.cos, ctx.sin)) * pa["q_gain"].astype(ft)[:, None]
    k = l2norm(rotate(k, ctx.cos, ctx.sin))
    s = ctx.sink if "sink_logit" in pa else None
    return sinks.assemble(q, k, v, s, pa.get("sink_logit"), pa.get("sink_value"), ctx.conf,
                          cfg.bias_dims, dt)


def attend_xla(q: jax.Array, k: jax.Array, v: jax.Array, seg_q: jax.Array, seg_k: jax.Array,
               pos_q: jax.Array, pos_k: jax.Array, sink_k: jax.Array | None = None):
    """Dense masked softmax attention.

    q: ``[B, Tq, H, hd]``; k, v: ``[B, Tk, H, hd]``; seg_*/pos_*: int ``[B, T*]``. A key is
    visible when ``pos_k <= pos_q`` and ``seg_k == seg_q``. sink_k: ``float32[B, Tk]`` or None.
    Logits and softmax in float32 (scale 1: the gain lives in q). Returns
    ``(y float32[B, Tq, H, hd], sink_mass float32[B, H, Tq] or None)``.
    """
    ft = hi(q.dtype)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=ft)
    mask = (pos_k[:, None, :] <= pos_q[:, :, None]) & (seg_k[:, None, :] == seg_q[:, :, None])
    logits = jnp.where(mask[:, None], logits, NEG_INF)
    p = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bkhd->bqhd", p.astype(v.dtype), v, preferred_element_type=ft)
    mass = None if sink_k is None else jnp.sum(p * sink_k.astype(ft)[:, None, None, :], axis=-1)
    return y, mass


def self_attend(q: jax.Array, k: jax.Array, v: jax.Array, ctx: Ctx, sink: jax.Array | None, cfg):
    """Full-sequence attention with the configured implementation.

    Returns ``(y float32[B, T, H, hd], sink_mass float32 scalar)``: the mean attention mass
    on sink keys over batch, heads and queries (0 when sinks are off).
    """
    if cfg.attn_impl == "xla":
        y, mass = attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos, sink)
        return y, (jnp.zeros((), F32) if mass is None else jnp.mean(mass))
    if cfg.attn_impl == "splash":
        from bes.model import splash  # TPU-only path, provided by the TPU attention track
        y = splash.attend(q, k, v, ctx.seg, cfg).astype(hi(q.dtype))
        if sink is None or ctx.conf is None:
            return y, jnp.zeros((), F32)
        # with the confidence dim on, its output is the non-sink mass
        return y, jnp.asarray(ctx.conf, F32) * jnp.mean(1.0 - y[..., -1].astype(F32))
    raise ValueError(f"unknown attn_impl {cfg.attn_impl!r}")


def heads(pa: dict, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg):
    """Per-head attention output before ``W_o``: ``(y float32[B, T, H, hd], sink_mass)``."""
    q, k, v = qkv(pa, x, ctx, ve, cfg)
    return self_attend(q, k, v, ctx, ctx.sink if "sink_logit" in pa else None, cfg)


def attention(pa: dict, wo: jax.Array, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg,
              attend_fn=None):
    """Full attention sub-layer: heads then ``W_o``.

    attend_fn: optional ``(q, k, v) -> (y [B, T, H, hd], aux)`` replacing the full-sequence
    attention (e.g. attention over a KV cache that also returns the updated cache as ``aux``).
    Returns ``(float32[B, T, d], aux)``; by default ``aux`` is the mean sink mass.
    """
    b, t, _ = x.shape
    q, k, v = qkv(pa, x, ctx, ve, cfg)
    if attend_fn is None:
        y, aux = self_attend(q, k, v, ctx, ctx.sink if "sink_logit" in pa else None, cfg)
    else:
        y, aux = attend_fn(q, k, v)
    out = mm(y.reshape(b, t, cfg.n_heads * cfg.head_dim), wo, act_dtype(cfg))
    return out, aux
