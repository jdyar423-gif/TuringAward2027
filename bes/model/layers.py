"""Basic layers: dtype policy, RMSNorm, matmul helper, ReLU^2 MLP, token neurons and initialisers.

Parts are ported from kschwethelm/looped-lm-scaling, ``nanochat/gpt.py`` (MIT licence):
RMSNorm with a learnable scale (eps 1e-6), the bias-free ReLU^2 MLP and the uniform
initialisation with bound ``sqrt(3) / sqrt(fan_in)`` (same standard deviation as a normal
initialisation, without outliers). The token neurons (:func:`token_neurons`) are a ReLU^2 layer
whose keys and values are rows of the position's token (closest published form: L3 "Large
Lookup Layers", Tseng and De Sa, ICML 2026, arXiv 2601.21461), with normalised keys.

Conventions used throughout ``bes.model``:

* every weight matrix is stored ``[..., fan_in, fan_out]`` and applied as ``x @ W``; stacked
  matrices carry their stacking axes first (``[n_blocks, d_in, d_out]``);
* parameters are float32 master copies; matmul operands are cast to the activation dtype
  (``cfg.act_dtype``) and accumulate in float32;
* normalisation statistics, logits and losses are float32.

``hi(dtype) = promote_types(dtype, float32)`` is the precision of statistics, logits and
accumulations: float32 for bfloat16 / float32 activations; float64 only when the activations
are float64 (a test-only setting that needs ``jax.enable_x64``, used for gradient checks).
"""
from __future__ import annotations

import math
import zlib
from typing import NamedTuple

import jax
import jax.numpy as jnp

F32 = jnp.float32
RMS_EPS = 1e-6   # same epsilon as the reference RMSNorm
L2_EPS = 1e-12   # epsilon of the QK L2 normalisation (squared-norm units)

#: Speed switches (read at trace time; ``False`` is the reference code path of the live run).
#: ``FAST_RELU2``: the feed-forward activation ``relu(h)^2`` has a hand-written backward
#: ``ct * 2 relu(h)`` that keeps only the activation-dtype pre-activation ``h`` (:func:`relu2`):
#: autodiff of ``square(relu(h))`` also keeps the comparison ``h > 0`` for ``relu``'s
#: ``select``, which on TPU v6e is a bit mask packed in the rematerialised ``W_up`` product and
#: unpacked in the backward one. Same values (only the sign of an exact zero can differ).
FAST_RELU2 = False


def act_dtype(cfg) -> jnp.dtype:
    """Activation dtype of a ``ModelCfg`` (``"bfloat16"``, ``"float32"``; ``"float64"`` in tests)."""
    return jnp.dtype(cfg.act_dtype)


def hi(dtype) -> jnp.dtype:
    """Statistics / logits / accumulation dtype for activations of ``dtype`` (>= float32)."""
    return jnp.promote_types(dtype, F32)


def rmsnorm(x: jax.Array, gain: jax.Array | None = None, *, out_dtype=None,
            eps: float = RMS_EPS) -> jax.Array:
    """RMS normalisation over the last axis with an optional per-channel gain.

    x: ``[..., d]`` any float dtype; gain: ``[d]`` or None.
    Statistics and the gain product are computed in ``hi(x.dtype)`` (float32); the result is
    cast to ``out_dtype`` (default: the dtype of ``x``).
    """
    xf = x.astype(hi(x.dtype))
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    if gain is not None:
        y = y * gain.astype(xf.dtype)
    return y.astype(out_dtype if out_dtype is not None else x.dtype)


def l2norm(x: jax.Array, eps: float = L2_EPS) -> jax.Array:
    """Unit-L2 normalisation over the last axis in ``hi(x.dtype)`` (a zero vector stays zero)."""
    xf = x.astype(hi(x.dtype))
    return xf * jax.lax.rsqrt(jnp.sum(xf * xf, axis=-1, keepdims=True) + eps)


def mm(x: jax.Array, w: jax.Array, dtype, out_dtype=None) -> jax.Array:
    """``x [..., k] @ w [k, n]`` with operands cast to ``dtype`` and ``hi(dtype)`` (float32)
    accumulation; the result has ``out_dtype`` (default ``hi(dtype)``)."""
    acc = hi(dtype)
    y = jnp.einsum("...k,kn->...n", x.astype(dtype), w.astype(dtype), preferred_element_type=acc)
    return y.astype(out_dtype if out_dtype is not None else acc)


class PhaseGain(NamedTuple):
    """Token gain rows of one block at a loop phase known only at run time (``gain_phases > 1``;
    ``bes.model.model.block_gain``).

    rows: ``[P, n_blocks, B, T, width]`` the gathered rows of every phase and block (closed over
    by the loop scan); phase: int32 ``[]`` (traced, fed through the scan's ``xs``); block: the
    static block index. :func:`gain_product` slices ``rows[phase, block]`` itself.
    """
    rows: jax.Array
    phase: jax.Array
    block: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``[B, T, width]`` of the selected rows."""
        return tuple(self.rows.shape[2:])


def gain_product(x: jax.Array, gain, dtype) -> jax.Array:
    """``x * (1 + gain)`` in ``dtype`` (``hi(act_dtype)``: ``1 + delta`` is not representable in
    bfloat16 for small deltas; ``x * (1 + 0) = x`` exactly).

    x: ``[..., n]``; gain: ``[..., n]`` delta rows (float32), or a :class:`PhaseGain`. For the
    latter the slice ``rows[phase, block]`` and the product form one ``jax.checkpoint`` region:
    the backward pass recomputes the slice from the closed-over rows instead of keeping a copy of
    it for every loop (without remat a traced slice is a per-loop residual of the product, with
    one phase the slice is static and loop-invariant). Same values either way.
    """
    if isinstance(gain, PhaseGain):
        i = gain.block

        def scaled(x, rows, phase):
            g = jax.lax.dynamic_index_in_dim(rows, phase, 0, keepdims=False)[i]
            return x.astype(dtype) * (1.0 + g.astype(dtype))
        return jax.checkpoint(scaled, prevent_cse=False)(x, gain.rows, gain.phase)
    return x.astype(dtype) * (1.0 + gain.astype(dtype))


@jax.custom_vjp
def relu2(h: jax.Array) -> jax.Array:
    """``relu(h)^2`` in ``hi(h.dtype)`` of a pre-activation ``h`` stored in the activation dtype:
    the values of ``jnp.square(jax.nn.relu(h.astype(hi(h.dtype))))``.

    The backward pass is written by hand, ``ct_h = (ct * (2 * relu(h))).astype(h.dtype)``, and
    keeps only ``h`` (the input). Autodiff of the reference expression computes the same
    product, ``ct * (2 * relu(h))`` (``square``'s rule), and then ``select(h > 0, ., 0)``
    (``relu``'s rule); where ``h <= 0`` the product is already a zero (``relu(h) = 0``), so the
    select only fixes the sign of that zero, and its operand ``h > 0`` is an extra residual of
    the size of ``h`` (on TPU a packed bit mask written by the rematerialised ``W_up`` product
    and read back by the backward one). Equal values for finite cotangents.
    """
    hf = h.astype(hi(h.dtype))
    return jnp.square(jax.nn.relu(hf))


def _relu2_fwd(h):
    return relu2(h), h


def _relu2_bwd(h, ct):
    hf = h.astype(hi(h.dtype))
    return ((ct * (2 * jax.nn.relu(hf))).astype(h.dtype),)


relu2.defvjp(_relu2_fwd, _relu2_bwd)


def mlp(x: jax.Array, w_up: jax.Array, w_down: jax.Array, dtype, gain=None) -> jax.Array:
    """``W_down relu(W_up x)^2`` without biases.

    x: ``[..., d]``; w_up: ``[d, d_ffn]``; w_down: ``[d_ffn, d]``. gain: optional ``[..., n]``
    multiplicative deltas (``n <= d_ffn``, float32 rows) or a :class:`PhaseGain` of that shape:
    the first ``n`` neurons are multiplied by ``1 + gain`` in ``hi(dtype)``, before the cast to
    ``dtype`` (:func:`gain_product`). Returns ``hi(dtype)`` (float32) ``[..., d]``.

    The pre-activations ``W_up x`` are stored in ``dtype`` and squared in ``hi(dtype)``: the
    ``[..., d_ffn]`` tensor is the largest of the block, and keeping it in float32 between the
    two products cost 28% of the sub-layer on TPU v6e (2.97 ms against 2.13 ms per application
    at 32,768 tokens; docs/TPU_NOTES.md). With ``dtype = float32`` nothing changes; with
    bfloat16 the pre-activation is rounded once more than before (2^-9 relative, unbiased),
    before a square whose result is rounded to bfloat16 for ``W_down`` either way. With
    :data:`FAST_RELU2` the square is :func:`relu2`: the same values, a backward pass that keeps
    only the stored pre-activation.
    """
    h = mm(x, w_up, dtype, out_dtype=dtype)
    if FAST_RELU2:
        h = relu2(h)
    else:
        h = jnp.square(jax.nn.relu(h.astype(hi(dtype))))
    if gain is not None:
        n = gain.shape[-1]
        h = (gain_product(h, gain, h.dtype) if n == h.shape[-1] else
             jnp.concatenate([gain_product(h[..., :n], gain, h.dtype), h[..., n:]], axis=-1))
    return mm(h.astype(dtype), w_down, dtype)


def token_preact(x: jax.Array, pin: jax.Array, key: jax.Array, dtype) -> jax.Array:
    """Pre-activations of the private neurons of each position's token.

    x: ``[B, T, d]`` normalised input of the feed-forward sub-layer; pin: ``[d, dk]``; key:
    ``[B, T, rho, dk]`` the token's keys, already RMS-normalised over ``dk`` (activation dtype).
    ``a[b, t, j] = <rmsnorm(x @ pin)[b, t], key[b, t, j]> / sqrt(dk)``: the projection
    accumulates in ``hi(dtype)``, is normalised to unit RMS and cast to ``dtype``; the
    contraction has ``dtype`` operands and ``hi(dtype)`` accumulation. Both factors have unit
    RMS, so ``|a| <= sqrt(dk)``, ``a`` does not depend on the scale of a key, and ``a`` is
    about ``N(0, 1)`` for random keys. Returns ``hi(dtype)`` ``[B, T, rho]``.
    """
    dk = pin.shape[-1]
    z = rmsnorm(mm(x, pin, dtype), out_dtype=dtype)
    a = jnp.einsum("...c,...jc->...j", z, key.astype(dtype), preferred_element_type=hi(dtype))
    return a * (1.0 / math.sqrt(dk))


def token_neurons(x: jax.Array, pin: jax.Array, pout: jax.Array, key: jax.Array,
                  val: jax.Array, on: jax.Array | float, dtype) -> jax.Array:
    """Token neurons: ``rho`` private ReLU^2 neurons of the position's token in a shared
    ``dk``-dimensional neuron space (``ModelCfg.tok_neurons``, ``tok_dim``).

        a = token_preact(x, pin, key)                # [B, T, rho]
        o = sum_j relu(a_j)^2 val_j                  # [B, T, dk]
        y = on * (o @ pout)                          # [B, T, d]

    x: ``[B, T, d]`` (the input of ``W_up``); pin ``[d, dk]``, pout ``[dk, d]``: the block's
    shared projections; key, val: ``[B, T, rho, dk]`` the rows of the position's token
    (``bes.model.model.token_rows``, activation dtype); on: the ``page`` switch (``0 * y`` is
    exactly zero, so the switch at 0 or ``val = 0`` adds nothing). ``relu(a)^2`` in
    ``hi(dtype)``; the value contraction and ``@ pout`` have ``dtype`` operands and ``hi(dtype)``
    accumulation. Returns ``hi(dtype)`` (float32) ``[B, T, d]``.
    """
    ft = hi(dtype)
    s = jnp.square(jax.nn.relu(token_preact(x, pin, key, dtype)))
    o = jnp.einsum("...j,...jc->...c", s.astype(dtype), val.astype(dtype), preferred_element_type=ft)
    return jnp.asarray(on, ft) * mm(o, pout, dtype)


def softcap(z: jax.Array, cap: float) -> jax.Array:
    """``cap * tanh(z / cap)`` in ``hi(z.dtype)`` (identity when ``cap <= 0``)."""
    z = z.astype(hi(z.dtype))
    if cap and cap > 0:
        z = cap * jnp.tanh(z / cap)
    return z


# --------------------------------------------------------------------------- initialisers
def leaf_key(key: jax.Array, path: str) -> jax.Array:
    """Per-leaf PRNG key derived from the leaf path, so that adding or removing feature
    leaves never changes the initial values of the other leaves."""
    return jax.random.fold_in(key, zlib.crc32(path.encode()) & 0x7FFFFFFF)


def uniform_init(key: jax.Array, shape: tuple[int, ...], fan_in: int) -> jax.Array:
    """Uniform in ``[-sqrt(3/fan_in), sqrt(3/fan_in)]`` (std ``1/sqrt(fan_in)``), float32."""
    bound = math.sqrt(3.0) / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, F32, -bound, bound)


def normal_init(key: jax.Array, shape: tuple[int, ...], std: float) -> jax.Array:
    """Normal with the given standard deviation, float32."""
    return std * jax.random.normal(key, shape, F32)
