"""Rotary embeddings driven by the token position and by the structural clocks.

The content part of every head (``head_dim - bias_dims`` dims, even) is split into
``P = (head_dim - bias_dims) // 2`` rotary planes. Plane layout (static):

* planes ``[0, P - clock_planes)`` use the token position within the row, with the standard
  geometric frequencies ``rope_base ** (-i / n_pos)``; with ``clock_planes = 0`` this is exactly
  the RoPE of the reference implementation (``inv_freq = base ** (-2i / dims)``);
* the remaining ``clock_planes`` are split evenly between the ``N_CLOCKS`` clocks, in clock
  order; the planes of one clock carry geometric frequencies ``rope_base ** (-j / per_clock)``;
  the angle is ``clock value * frequency``.

Rotation convention ported from kschwethelm/looped-lm-scaling ``nanochat/gpt.py`` (MIT):
the vector is split into halves ``x1 = x[..., :P]``, ``x2 = x[..., P:]`` and plane ``i`` rotates
``(x1_i, x2_i)``; only relative rotations matter.

Angles are computed with element-wise float32 products (never a matmul: TPU matmuls in the
default precision would round positions to bfloat16).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import N_CLOCKS

F32 = jnp.float32


def plane_layout(cfg) -> tuple[int, int, int]:
    """Returns ``(n_planes, n_position_planes, planes_per_clock)`` for a ``ModelCfg``."""
    dc = cfg.head_dim - cfg.bias_dims
    if dc <= 0 or dc % 2:
        raise ValueError(f"content dims head_dim - bias_dims = {dc} must be positive and even")
    n_planes = dc // 2
    if cfg.clock_planes < 0 or cfg.clock_planes > n_planes:
        raise ValueError(f"clock_planes={cfg.clock_planes} must lie in [0, {n_planes}]")
    if cfg.clock_planes % N_CLOCKS:
        raise ValueError(f"clock_planes={cfg.clock_planes} must be a multiple of N_CLOCKS={N_CLOCKS}")
    return n_planes, n_planes - cfg.clock_planes, cfg.clock_planes // N_CLOCKS


def freq_table(cfg, dtype=np.float32) -> np.ndarray:
    """Frequency table ``[1 + N_CLOCKS, P]`` (float32 by default).

    Row 0 holds the frequencies of the position planes (zero elsewhere); row ``1 + c`` the
    frequencies of the planes of clock ``c``. The angle of plane ``p`` is
    ``pos * F[0, p] + sum_c clock_c * F[1 + c, p]``; each plane has exactly one non-zero row.
    """
    n_planes, n_pos, per = plane_layout(cfg)
    table = np.zeros((1 + N_CLOCKS, n_planes), np.float64)
    if n_pos:
        table[0, :n_pos] = cfg.rope_base ** (-np.arange(n_pos, dtype=np.float64) / n_pos)
    if per:
        f = cfg.rope_base ** (-np.arange(per, dtype=np.float64) / per)
        for c in range(N_CLOCKS):
            table[1 + c, n_pos + c * per: n_pos + (c + 1) * per] = f
    return table.astype(dtype)


def angles(pos: jax.Array, clock: jax.Array | None, freq: jax.Array,
           clock_on: jax.Array | float = 1.0) -> jax.Array:
    """Rotary angles ``[..., P]`` in the dtype of ``freq`` (float32).

    pos: int ``[...]`` token position within the row; clock: int ``[..., N_CLOCKS]`` or None
    (treated as zeros); freq: ``[1 + N_CLOCKS, P]`` from :func:`freq_table`;
    clock_on: 0/1 switch multiplying the clock values (0 leaves the clock planes unrotated).
    """
    freq = jnp.asarray(freq)
    dt = freq.dtype
    a = pos.astype(dt)[..., None] * freq[0]
    if clock is not None:
        c = clock.astype(dt) * jnp.asarray(clock_on, dt)
        for i in range(N_CLOCKS):
            a = a + c[..., i:i + 1] * freq[1 + i]
    return a


def cos_sin(ang: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(cos, sin)`` of ``ang [B, T, P]`` with a head axis inserted: ``[B, T, 1, P]``."""
    ang = ang[..., None, :]
    return jnp.cos(ang), jnp.sin(ang)


def rotate(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply the rotation to ``x [..., 2P]``; cos/sin broadcast to ``[..., P]``.

    Computed in ``promote_types(x.dtype, cos.dtype)`` (float32)."""
    p = cos.shape[-1]
    xf = x.astype(jnp.promote_types(x.dtype, cos.dtype))
    x1, x2 = xf[..., :p], xf[..., p:]
    return jnp.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)
