"""TPU attention path (``cfg.attn_impl == "splash"``): the Pallas splash-attention kernels of
``jax.experimental.pallas.ops.tpu.splash_attention`` as shipped with jax 0.11.1.

Contract (identical to the dense path ``bes.model.attention.attend_xla`` for self-attention):
``q, k, v: [B, T, H, head_dim]`` arrive full width with the reserved bias dims, the sink keys
and values and the confidence dim already assembled (``bes.model.sinks``); the logit is the plain
dot product (no softmax scale: the learned gain lives in ``q``); a key is visible when it is not
later than the query and has the same segment id. The calibrated sink is an ordinary key (the
``BOS`` position, whose bias dim carries ``sink_logit``), so the kernel's own ``sinks`` argument
(a value-less extra logit per head) is not used, and neither is its logit soft cap.

What the installed kernel provides and how it is used here:

* per-example kernels ``[H, T, head_dim]`` (``make_splash_mha_single_device``), vmapped over the
  batch; a static causal mask (``CausalMask``, computed inside the kernel, fully masked blocks
  skipped) combined with per-example ``SegmentIds``;
* block sizes from the config (``splash_block_q``, ``splash_block_kv``,
  ``splash_block_kv_compute``), shrunk to the largest multiple of 128 dividing the padded
  length; the backward kernels use the same tiles (``splash_fused_bwd`` selects the fused
  dQ/dK/dV kernel);
* lengths that are not a multiple of 128 are padded with an extra segment id (padded queries
  see only padded keys, so no softmax row is empty; real queries never see them);
* the kernel's custom VJP (forward residual = log-sum-exp) gives the gradients;
* ``interpret`` mode (``splash_interpret``: ``"auto"`` = interpret unless the default backend is
  a TPU) runs the same kernels on CPU, which the tests use against the dense path.

Precision: the kernel computes ``q.k`` with float32 accumulation of the operand dtype (bfloat16 in
training), the softmax statistics in float32 and ``p @ v`` with ``p`` in float32 and ``v`` cast to
float32; the output is returned in the dtype of ``q``.

Imported lazily by ``bes.model.attention`` so CPU runs never touch Pallas.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

LANES = 128            # TPU lane width: block sizes and padded lengths are multiples of it
PAD_SEGMENT = -2       # segment id of padding positions (real ids are >= 0, cache gaps use -1)


def _sa():
    from jax.experimental.pallas.ops.tpu import splash_attention
    return splash_attention


def available() -> bool:
    """True when the installed jax ships the splash-attention kernels."""
    try:
        _sa()
    except ImportError:
        return False
    return True


def register_alias() -> None:
    """Make ``import bes.model.splash`` resolve to this module when no such module exists.

    ``bes.model.attention`` selects the splash path with ``from bes.model import splash``; this
    module is ``bes.model.attention_splash``. Until that import names this module, callers that
    run ``attn_impl="splash"`` (the compile-only checks, training) call this first; afterwards it
    is a no-op.
    """
    import importlib.util
    import sys
    if "bes.model.splash" in sys.modules or importlib.util.find_spec("bes.model.splash") is not None:
        return
    sys.modules["bes.model.splash"] = sys.modules[__name__]


def use_interpret(cfg) -> bool:
    """Interpret mode from ``cfg.splash_interpret`` (``"auto"``: not on a TPU backend)."""
    mode = str(getattr(cfg, "splash_interpret", "auto")).lower()
    if mode in ("1", "true", "yes", "on"):
        return True
    if mode in ("0", "false", "no", "off"):
        return False
    if mode != "auto":
        raise ValueError(f"splash_interpret must be auto/true/false, got {mode!r}")
    return jax.default_backend() != "tpu"


def padded_length(t: int) -> int:
    """Sequence length rounded up to a multiple of 128."""
    return -(-int(t) // LANES) * LANES


def _tile(requested: int, length: int) -> int:
    """Largest multiple of 128 that is <= ``requested`` and divides ``length``."""
    b = max(LANES, min(int(requested), length) // LANES * LANES)
    while length % b:
        b -= LANES
    return b


def block_sizes(cfg, t: int):
    """``BlockSizes`` for a padded length ``t`` (a multiple of 128), forward and backward."""
    sa = _sa()
    bq = _tile(getattr(cfg, "splash_block_q", 512), t)
    bkv = _tile(getattr(cfg, "splash_block_kv", 512), t)
    bkc = _tile(getattr(cfg, "splash_block_kv_compute", 512), bkv)
    fused = bool(getattr(cfg, "splash_fused_bwd", False))
    return sa.BlockSizes(
        block_q=bq, block_kv=bkv, block_kv_compute=bkc,
        block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkc,
        block_q_dq=None if fused else bq, block_kv_dq=None if fused else bkv,
        use_fused_bwd_kernel=fused)


@functools.lru_cache(maxsize=None)
def make_kernel(t: int, n_heads: int, bs, interpret: bool):
    """Per-example kernel for ``n_heads`` heads and length ``t`` (multiple of the blocks).

    Built under ``jax.ensure_compile_time_eval`` so that its mask-info arrays are concrete even
    when the first request comes from inside a trace (otherwise the cache would keep tracers of
    that trace and the next program using the same shape would fail)."""
    sa = _sa()
    mask = sa.MultiHeadMask([sa.CausalMask((t, t)) for _ in range(n_heads)])
    with jax.ensure_compile_time_eval():
        return sa.make_splash_mha_single_device(mask, block_sizes=bs, interpret=interpret)


def attend(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, cfg, *,
           interpret: bool | None = None) -> jax.Array:
    """Causal same-segment self-attention with the splash kernels.

    q, k, v: ``[B, T, H, head_dim]`` (any float dtype, normally the activation dtype);
    seg: int ``[B, T]`` segment ids (negative ids are allowed and only see equal ids).
    Returns ``[B, T, H, head_dim]`` in the dtype of ``q``.
    """
    qt, kt, vt = (jnp.swapaxes(a, 1, 2) for a in (q, k, v))                   # [B, H, T, hd]
    return jnp.swapaxes(attend_major(qt, kt, vt, seg, cfg, interpret=interpret), 1, 2)


def attend_major(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, cfg, *,
                 interpret: bool | None = None) -> jax.Array:
    """:func:`attend` on head-major operands: ``q, k, v: [B, H, T, head_dim]``, the layout of the
    kernels, so that a caller that builds its operands in this layout pays no transpose (on TPU a
    transpose of an activation costs about as much as the matrix product that made it:
    docs/TPU_NOTES.md). Returns ``[B, H, T, head_dim]`` in the dtype of ``q``.
    """
    sa = _sa()
    b, h, t, hd = q.shape
    tp = padded_length(t)
    seg = jnp.asarray(seg, jnp.int32)
    if tp != t:
        pad = ((0, 0), (0, 0), (0, tp - t), (0, 0))
        q, k, v = (jnp.pad(a, pad) for a in (q, k, v))
        seg = jnp.pad(seg, ((0, 0), (0, tp - t)), constant_values=PAD_SEGMENT)
    interp = use_interpret(cfg) if interpret is None else bool(interpret)
    kernel = make_kernel(tp, h, block_sizes(cfg, tp), interp)
    ids = sa.SegmentIds(q=seg, kv=seg)

    def one(q1, k1, v1, s1):
        return kernel(q1, k1, v1, segment_ids=s1)
    return jax.vmap(one)(q, k, v, ids)[:, :, :t]


def attend_like_xla(q: jax.Array, k: jax.Array, v: jax.Array, seg_q: jax.Array,
                    seg_k: jax.Array, pos_q: jax.Array, pos_k: jax.Array,
                    sink_k: jax.Array | None = None, *, cfg=None):
    """The signature of ``bes.model.attention.attend_xla`` for self-attention.

    Requires ``Tq == Tk`` with the same segments and positions ``arange(T)`` (the static causal
    mask is on slot order); positions and segments are not checked inside a trace. Returns
    ``(y float32[B, T, H, hd], None)``: the kernel does not expose the probabilities (with the
    confidence dim on, the sink mass is ``1 - y[..., -1]``).
    """
    del seg_k, pos_q, pos_k, sink_k
    if q.shape[1] != k.shape[1]:
        raise ValueError("the splash path is self-attention only (Tq == Tk)")
    y = attend(q, k, v, seg_q, cfg if cfg is not None else _DefaultCfg())
    return y.astype(jnp.promote_types(q.dtype, jnp.float32)), None


class _DefaultCfg:
    splash_block_q = splash_block_kv = splash_block_kv_compute = 512
    splash_fused_bwd = False
    splash_interpret = "auto"


def reference(q, k, v, seg) -> np.ndarray:
    """Float64 numpy reference of the same attention (tests and diagnostics)."""
    q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
    seg = np.asarray(seg)
    t = q.shape[1]
    logits = np.einsum("bqhd,bkhd->bhqk", q, k)
    mask = (np.arange(t)[None, :, None] >= np.arange(t)[None, None, :]) & \
        (seg[:, :, None] == seg[:, None, :])
    logits = np.where(mask[:, None], logits, -np.inf)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", p, v)


def flops(b: int, t: int, h: int, hd: int, causal: bool = True) -> float:
    """Forward matmul FLOPs of the attention core (``q.k`` and ``p.v``); half when causal."""
    f = 4.0 * b * h * t * t * hd
    return f / 2 if causal else f


__all__ = ["available", "register_alias", "use_interpret", "padded_length", "block_sizes", "make_kernel", "attend",
           "attend_major", "attend_like_xla", "reference", "flops", "LANES"]
