"""Loop controller: hyper-connection lanes and the scanned recurrence.

Ported from kschwethelm/looped-lm-scaling ``nanochat/gpt.py`` (MIT licence), "hyperconnect"
injection mode: ``N`` parallel lane states, per-loop aggregation ``alpha_r``, mixing ``M_r`` and
broadcast ``beta_r``; cyclic initialisation ``alpha_r = e_{r mod N}``, ``M_r = I``,
``beta_r = 1`` so that the first step equals the plain looped forward.

    lanes = (e, ..., e)
    for r:  u = sum_n alpha[r, n] lanes[n];  u = blocks(u);  s = rmsnorm(u) * g_loop[r]
            lanes = mix[r] @ lanes + beta[r][:, None] * s

The lane combinations are written as element-wise float32 products (the lane count is tiny),
never as matmuls, so the coefficients keep full precision on TPU.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from bes.model.layers import hi


def init_lanes(n_loops: int, n_lanes: int) -> dict[str, np.ndarray]:
    """Cyclic initial lane coefficients (float32 numpy): alpha ``[L, N]``, mix ``[L, N, N]``,
    beta ``[L, N]``."""
    alpha = np.zeros((n_loops, n_lanes), np.float32)
    alpha[np.arange(n_loops), np.arange(n_loops) % n_lanes] = 1.0
    mix = np.tile(np.eye(n_lanes, dtype=np.float32), (n_loops, 1, 1))
    beta = np.ones((n_loops, n_lanes), np.float32)
    return {"alpha": alpha, "mix": mix, "beta": beta}


def combine(alpha_r: jax.Array, lanes: jax.Array, dtype) -> jax.Array:
    """``sum_n alpha_r[n] * lanes[n]``: alpha_r ``[N]``, lanes ``[N, B, T, d]`` -> ``[B, T, d]``."""
    ft = hi(dtype)
    a = alpha_r.astype(ft)
    u = a[0] * lanes[0].astype(ft)
    for n in range(1, lanes.shape[0]):
        u = u + a[n] * lanes[n].astype(ft)
    return u.astype(dtype)


def update(mix_r: jax.Array, beta_r: jax.Array, lanes: jax.Array, s: jax.Array, dtype) -> jax.Array:
    """``mix_r @ lanes + beta_r[:, None] * s``: mix_r ``[N, N]``, beta_r ``[N]``,
    lanes ``[N, B, T, d]``, s ``[B, T, d]`` -> ``[N, B, T, d]``."""
    ft = hi(dtype)
    m = mix_r.astype(ft)
    bt = beta_r.astype(ft)
    n_lanes = lanes.shape[0]
    lf = [lanes[j].astype(ft) for j in range(n_lanes)]
    sf = s.astype(ft)
    out = []
    for i in range(n_lanes):
        acc = bt[i] * sf
        for j in range(n_lanes):
            acc = acc + m[i, j] * lf[j]
        out.append(acc.astype(dtype))
    return jnp.stack(out)


def run_loops(step: Callable[[jax.Array, Any], tuple[jax.Array, Any]], e: jax.Array,
              xs: Any, n_lanes: int, remat: bool) -> tuple[jax.Array, Any]:
    """Run ``step(lanes, xs_r) -> (lanes, ys_r)`` over the loop axis with ``lax.scan``.

    e: ``[B, T, d]`` initial state copied into every lane; xs: per-loop inputs with a leading
    loop axis; remat: wrap the step in ``jax.checkpoint`` (recompute in the backward pass).
    Returns ``(lanes [N, B, T, d], ys with a leading loop axis)``.
    """
    lanes0 = jnp.broadcast_to(e[None], (n_lanes,) + e.shape)
    body = jax.checkpoint(step, prevent_cse=False) if remat else step
    return jax.lax.scan(body, lanes0, xs)
