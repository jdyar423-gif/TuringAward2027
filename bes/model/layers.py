"""Basic layers: dtype policy, RMSNorm, matmul helper, ReLU^2 MLP and initialisers.

Parts are ported from kschwethelm/looped-lm-scaling, ``nanochat/gpt.py`` (MIT licence):
RMSNorm with a learnable scale (eps 1e-6), the bias-free ReLU^2 MLP and the uniform
initialisation with bound ``sqrt(3) / sqrt(fan_in)`` (same standard deviation as a normal
initialisation, without outliers).

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

import jax
import jax.numpy as jnp

F32 = jnp.float32
RMS_EPS = 1e-6   # same epsilon as the reference RMSNorm
L2_EPS = 1e-12   # epsilon of the QK L2 normalisation (squared-norm units)


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


def mlp(x: jax.Array, w_up: jax.Array, w_down: jax.Array, dtype) -> jax.Array:
    """``W_down relu(W_up x)^2`` without biases.

    x: ``[..., d]``; w_up: ``[d, d_ffn]``; w_down: ``[d_ffn, d]``. Returns ``hi(dtype)``
    (float32) ``[..., d]``.
    """
    h = mm(x, w_up, dtype)
    h = jnp.square(jax.nn.relu(h)).astype(dtype)
    return mm(h, w_down, dtype)


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
