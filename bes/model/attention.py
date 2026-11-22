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

Token gains (``bes.model.model.gain_rows``): the value channels of a position are multiplied by
``1 + g_v`` (``x @ W_v``, before the value embedding is added) and its attention output channels
by ``1 + g_o`` before ``W_o``, both in ``hi(act_dtype)``.

Selective attention (Leviathan, Kalman, Matias, ICLR 2025, arXiv 2410.02703; dense path only,
:func:`selective_logits`): head 0's positive raw logits ``S[i, j]`` (zero for the query's own
slot, BOS keys and invisible or padded pairs), accumulated over the earlier queries,
``F[i, j] = sum_{i' < i} S[i', j]``, are subtracted from the logits of every head before the
softmax (``logits - sel * F``; ``sel`` is the runtime switch, ``x - 0 * F = x`` exactly).

Head-wise output gate (``attn_gate``; Qiu et al., "Gated Attention for Large Language Models",
arXiv 2505.06708; :func:`head_gate`): the output of head ``h`` at a position is multiplied by
``1 + gate * (2 sigmoid(x @ W_g[:, h] / d + b_g[r, h]) - 1)`` (``x`` the position's normalised block
input, ``r`` the loop, ``gate`` the runtime switch), after the kernel or the ``attend_fn`` hook and
the sink-mass metric, before the token gain ``g_o`` and ``W_o``; float32. ``W_g`` and ``b_g`` are
zero-initialised, so the gate is exactly 1 at initialisation (``2 sigmoid(0) - 1 = 0``).

Implementations (``cfg.attn_impl``):
* ``"xla"``: dense masked softmax, any backend (CPU tests);
* ``"splash"``: provided by ``bes.model.attention_splash`` as
  ``attend(q, k, v, seg, cfg) -> y`` with ``q, k, v, y: [B, T, H, head_dim]`` and the same
  semantics (static causal mask + per-example segment ids). Imported lazily so CPU runs never
  touch it. No selective attention.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bes.model import sinks
from bes.model.layers import F32, act_dtype, gain_product, hi, l2norm, mm
from bes.model.rope_clocks import rotate

VE_GATE_CHANNELS = 12   # nanochat: the gate reads the first 12 channels of the block input
NEG_INF = float(jnp.finfo(jnp.float32).min)


class Ctx(NamedTuple):
    """Per-forward attention context shared by all loops and blocks.

    cos, sin: ``float32[B, T, 1, P]`` rotary tables; sink: ``float32[B, T]`` sink indicator or
    None (sinks statically off); conf: 0/1 confidence switch or None (statically off);
    seg, pos: ``int32[B, T]`` segment ids and positions used by the mask; bos: ``bool[B, T]``
    the position holds BOS (keys selective attention never masks) or None; sel: ``float32[]``
    selective-attention switch, or None (statically off); gate: ``float32[]`` switch of the
    head-wise output gate, or None (statically off).
    """
    cos: jax.Array
    sin: jax.Array
    sink: jax.Array | None
    conf: jax.Array | None
    seg: jax.Array
    pos: jax.Array
    bos: jax.Array | None = None
    sel: jax.Array | None = None
    gate: jax.Array | None = None


def qkv(pa: dict, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg, gain_v=None):
    """Queries, keys and values of one block.

    pa: per-block attention parameters ``wq [d, H*dc]``, ``wk [d, H*dc]``, ``wv [d, H*hd]``,
    ``q_gain [H]`` and, when present, ``sink_logit [H]``, ``sink_value [H, hd-1]``,
    ``ve_gate [12, n_ve]``. x: ``[B, T, d]`` normalised block input (activation dtype).
    ve: ``[B, T, n_ve * hd]`` value embedding of this loop (block 0 only) or None.
    gain_v: ``[B, T, H*hd]`` per-token value gain deltas (float32 rows, or a
    ``layers.PhaseGain``) or None: ``x @ W_v`` is multiplied by ``1 + gain_v`` in
    ``hi(act_dtype)`` (``layers.gain_product``) before the value embedding is added.
    Returns ``(q, k, v)``, each ``[B, T, H, hd]`` in the activation dtype, reserved dims filled.
    """
    dt = act_dtype(cfg)
    ft = hi(dt)
    b, t, _ = x.shape
    h, hd = cfg.n_heads, cfg.head_dim
    dc = hd - cfg.bias_dims
    q = mm(x, pa["wq"], dt).reshape(b, t, h, dc)
    k = mm(x, pa["wk"], dt).reshape(b, t, h, dc)
    v = mm(x, pa["wv"], dt)
    if gain_v is not None:
        v = gain_product(v, gain_v, ft)
    v = v.reshape(b, t, h, hd)
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


def selective_logits(logits: jax.Array, visible: jax.Array, maskable_k: jax.Array,
                     q_slot: jax.Array, sel: jax.Array | float,
                     prior: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """Selective attention on dense logits (module docstring).

    logits: ``float32[B, H, Tq, Tk]`` masked logits (``NEG_INF`` where not visible); visible:
    ``bool[B, Tq, Tk]`` the attention mask; maskable_k: ``bool[B, Tk]`` keys selection may mask
    (not BOS, not padding); q_slot: int ``[Tq]`` or ``[B, Tq]`` the key index of each query's
    own slot (``arange(T)`` for self-attention; ``t0 + arange(Tq)`` for queries appended to a
    cache at slot ``t0``); sel: the ``float32[]`` switch; prior: ``float32[B, Tk]`` selection
    already accumulated by earlier queries outside ``logits`` (a key-value cache), or None.

    ``S[b, i, j] = relu(logits[b, 0, i, j])`` where ``visible & maskable_k & (j != q_slot[i])``,
    else 0 (head 0 selects, raw logits); ``F[b, i, j] = prior[b, j] + sum_{i' < i} S[b, i', j]``
    (queries in slot order); the result is ``logits - sel * F`` on every head where visible
    (``NEG_INF`` elsewhere). Gradients flow through ``F``.
    Returns ``(logits float32[B, H, Tq, Tk], s_sum float32[B, Tk])``: ``s_sum`` is
    ``sum_i S[b, i, j]`` over these queries (what a cache adds to its ``prior``).
    """
    tk = logits.shape[-1]
    own = jnp.asarray(q_slot)[..., :, None] == jnp.arange(tk, dtype=jnp.int32)
    keep = visible & maskable_k[:, None, :] & ~own
    s = jnp.where(keep, jax.nn.relu(logits[:, 0]), jnp.zeros((), logits.dtype))      # [B, Tq, Tk]
    f = jnp.cumsum(jnp.concatenate([jnp.zeros_like(s[:, :1]), s[:, :-1]], axis=1), axis=1)
    if prior is not None:
        f = f + prior[:, None, :].astype(f.dtype)
    out = jnp.where(visible[:, None], logits - jnp.asarray(sel, logits.dtype) * f[:, None], NEG_INF)
    return out, jnp.sum(s, axis=1)


def attend_xla(q: jax.Array, k: jax.Array, v: jax.Array, seg_q: jax.Array, seg_k: jax.Array,
               pos_q: jax.Array, pos_k: jax.Array, sink_k: jax.Array | None = None, *,
               sel: jax.Array | None = None, bos_k: jax.Array | None = None):
    """Dense masked softmax attention.

    q: ``[B, Tq, H, hd]``; k, v: ``[B, Tk, H, hd]``; seg_*/pos_*: int ``[B, T*]``. A key is
    visible when ``pos_k <= pos_q`` and ``seg_k == seg_q``. sink_k: ``float32[B, Tk]`` or None.
    sel: selective-attention switch ``float32[]`` or None (off); with it, self-attention only
    (``Tq == Tk``, query ``i`` in slot ``i``) and bos_k: ``bool[B, Tk]`` the BOS keys, which are
    never masked; keys with ``seg_k < 0`` (padding) are not masked either
    (:func:`selective_logits`).
    Logits and softmax in float32 (scale 1: the gain lives in q). Returns
    ``(y float32[B, Tq, H, hd], sink_mass float32[B, H, Tq] or None)``.
    """
    ft = hi(q.dtype)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=ft)
    mask = (pos_k[:, None, :] <= pos_q[:, :, None]) & (seg_k[:, None, :] == seg_q[:, :, None])
    logits = jnp.where(mask[:, None], logits, NEG_INF)
    if sel is not None:
        tq, tk = q.shape[1], k.shape[1]
        if tq != tk or bos_k is None:
            raise ValueError("selective attention in attend_xla is self-attention (Tq == Tk) "
                             "and needs bos_k")
        logits, _ = selective_logits(logits, mask, ~bos_k & (seg_k >= 0),
                                     jnp.arange(tq, dtype=jnp.int32), sel)
    p = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bkhd->bqhd", p.astype(v.dtype), v, preferred_element_type=ft)
    mass = None if sink_k is None else jnp.sum(p * sink_k.astype(ft)[:, None, None, :], axis=-1)
    return y, mass


def self_attend(q: jax.Array, k: jax.Array, v: jax.Array, ctx: Ctx, sink: jax.Array | None, cfg):
    """Full-sequence attention with the configured implementation.

    Returns ``(y float32[B, T, H, hd], sink_mass float32 scalar)``: the mean attention mass
    on sink keys over batch, heads and queries (0 when sinks are off). Selective attention
    (``ctx.sel`` not None) needs ``attn_impl == "xla"``.
    """
    if cfg.attn_impl == "xla":
        y, mass = attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos, sink, sel=ctx.sel,
                             bos_k=ctx.bos)
        return y, (jnp.zeros((), F32) if mass is None else jnp.mean(mass))
    if ctx.sel is not None:
        raise ValueError("selective attention needs the dense path (attn_impl='xla')")
    if cfg.attn_impl == "splash":
        from bes.model import attention_splash as splash  # TPU path, imported lazily
        y = splash.attend(q, k, v, ctx.seg, cfg).astype(hi(q.dtype))
        if sink is None or ctx.conf is None:
            return y, jnp.zeros((), F32)
        # with the confidence dim on, its output is the non-sink mass
        return y, jnp.asarray(ctx.conf, F32) * jnp.mean(1.0 - y[..., -1].astype(F32))
    raise ValueError(f"unknown attn_impl {cfg.attn_impl!r}")


def heads(pa: dict, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg):
    """Per-head kernel output before the output gate, the token gains and ``W_o``:
    ``(y float32[B, T, H, hd], sink_mass)``."""
    q, k, v = qkv(pa, x, ctx, ve, cfg)
    return self_attend(q, k, v, ctx, ctx.sink if "sink_logit" in pa else None, cfg)


def head_gate(pa: dict, x: jax.Array, loop: jax.Array | int, on: jax.Array | float) -> jax.Array:
    """Head-wise output gate ``1 + on * (2 sigmoid(x @ gate_w / d + gate_b[loop]) - 1)``.

    pa: per-block attention parameters with ``gate_w [d, H]`` and ``gate_b [n_loops, H]``
    (float32); x: ``[B, T, d]`` normalised block input (the input of ``W_q``, ``W_k``, ``W_v``);
    loop: int32 ``[]`` loop index (may be traced: fed through the loop scan's ``xs``); on: the
    ``gate`` switch. The gate logits are a matmul with float32 operands at the backend's
    default precision (as the value-embedding gate in :func:`qkv`); the sigmoid and the gate
    are float32. The logit is the mean over the ``d`` channels of ``x * gate_w``: Adam moves
    every entry of ``gate_w`` by about its rate per step, so an unscaled sum over ``d``
    unit-RMS channels would move the logit ``d`` times faster than a scalar of the same group
    (measured: the unscaled gate saturated within 244 steps and cost 1% of the loss in the
    micro ablation, docs/V9.md). In ``(0, 2)``; exactly 1 with zero parameters or ``on = 0``.
    Returns ``hi(x.dtype)`` (float32) ``[B, T, H]``.
    """
    ft = hi(x.dtype)
    bias = jax.lax.dynamic_index_in_dim(pa["gate_b"], loop, 0, keepdims=False).astype(ft)   # [H]
    z = (x.astype(ft) @ pa["gate_w"].astype(ft)) * (1.0 / x.shape[-1]) + bias
    return 1.0 + jnp.asarray(on, ft) * (2.0 * jax.nn.sigmoid(z) - 1.0)


def attention(pa: dict, wo: jax.Array, x: jax.Array, ctx: Ctx, ve: jax.Array | None, cfg,
              attend_fn=None, gain_v=None, gain_o=None, loop: jax.Array | int | None = None):
    """Full attention sub-layer: heads, output gate, token gain, then ``W_o``.

    attend_fn: optional ``(q, k, v) -> (y [B, T, H, hd], aux)`` replacing the full-sequence
    attention (e.g. attention over a KV cache that also returns the updated cache as ``aux``).
    gain_v, gain_o: ``[B, T, H*hd]`` per-token gain deltas (float32 rows, or
    ``layers.PhaseGain``) or None: the values (see :func:`qkv`) and the head outputs ``y`` are
    multiplied by ``1 + gain`` in ``hi(act_dtype)`` (``layers.gain_product``), the latter after
    ``attend_fn`` / the sink-mass metric and before ``W_o``.
    With ``gate_w`` / ``gate_b`` in ``pa`` (``attn_gate``) the head outputs ``y`` (of the kernel
    or of ``attend_fn``) are first multiplied by :func:`head_gate` with the switch ``ctx.gate``
    and loop: the int32 loop index (required then). Order: kernel -> sink-mass metric -> gate
    -> ``gain_o`` -> ``W_o``. Returns ``(float32[B, T, d], aux)``; by default ``aux`` is the
    mean sink mass.
    """
    b, t, _ = x.shape
    ft = hi(act_dtype(cfg))
    q, k, v = qkv(pa, x, ctx, ve, cfg, gain_v)
    if attend_fn is None:
        y, aux = self_attend(q, k, v, ctx, ctx.sink if "sink_logit" in pa else None, cfg)
    else:
        y, aux = attend_fn(q, k, v)
    if "gate_w" in pa:
        if loop is None or ctx.gate is None:
            raise ValueError("the attention output gate needs the loop index and ctx.gate "
                             "(make_ctx with attn_gate)")
        y = y.astype(ft) * head_gate(pa, x, loop, ctx.gate)[..., None]
    y = y.reshape(b, t, cfg.n_heads * cfg.head_dim)
    if gain_o is not None:
        y = gain_product(y, gain_o, ft)
    out = mm(y, wo, act_dtype(cfg))
    return out, aux
