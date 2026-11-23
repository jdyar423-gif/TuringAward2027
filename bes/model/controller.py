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


def combine(alpha_r: jax.Array, lanes, dtype) -> jax.Array:
    """``sum_n alpha_r[n] * lanes[n]``: alpha_r ``[N]``, lanes ``[N, B, T, d]`` or a tuple of
    ``N`` arrays ``[B, T, d]`` -> ``[B, T, d]``."""
    ft = hi(dtype)
    a = alpha_r.astype(ft)
    u = a[0] * lanes[0].astype(ft)
    for n in range(1, len(lanes)):
        u = u + a[n] * lanes[n].astype(ft)
    return u.astype(dtype)


def update(mix_r: jax.Array, beta_r: jax.Array, lanes, s: jax.Array, dtype):
    """``mix_r @ lanes + beta_r[:, None] * s``: mix_r ``[N, N]``, beta_r ``[N]``,
    lanes ``[N, B, T, d]``, s ``[B, T, d]`` -> ``[N, B, T, d]``; with a tuple of ``N`` lanes
    ``[B, T, d]`` the result is a tuple too (no stacking)."""
    ft = hi(dtype)
    m = mix_r.astype(ft)
    bt = beta_r.astype(ft)
    n_lanes = len(lanes)
    lf = [lanes[j].astype(ft) for j in range(n_lanes)]
    sf = s.astype(ft)
    out = []
    for i in range(n_lanes):
        acc = bt[i] * sf
        for j in range(n_lanes):
            acc = acc + m[i, j] * lf[j]
        out.append(acc.astype(dtype))
    return tuple(out) if isinstance(lanes, (tuple, list)) else jnp.stack(out)


def keep_kernel_outputs(prim, *_, **__) -> bool:
    """``jax.checkpoint`` policy: keep the outputs of the Pallas kernels (the attention output
    and the log-sum-exp residual of the forward splash kernel), recompute everything else.

    Without it the backward pass of a rematerialised loop step runs the forward kernels a second
    time only to rebuild those outputs. They are one activation per block application (50 MB at
    32 x 1024 in bfloat16); the matrix products and the element-wise work around them are
    recomputed either way, because their own residuals (float32, several times larger) are what
    is cheaper to recompute than to keep (docs/TPU_NOTES.md)."""
    return getattr(prim, "name", "") == "pallas_call"


REMAT_POLICIES: dict[str, Any] = {"": None, "kernel": keep_kernel_outputs}


def run_loops(step: Callable[[jax.Array, Any], tuple[jax.Array, Any]], e: jax.Array,
              xs: Any, n_lanes: int, remat: bool, keep: str = "") -> tuple[jax.Array, Any]:
    """Run ``step(lanes, xs_r) -> (lanes, ys_r)`` over the loop axis with ``lax.scan``.

    e: ``[B, T, d]`` initial state copied into every lane; xs: per-loop inputs with a leading
    loop axis; remat: wrap the step in ``jax.checkpoint`` (recompute in the backward pass);
    keep: what that backward pass keeps instead (``ModelCfg.remat_keep``, :data:`REMAT_POLICIES`).
    The lanes are carried as a tuple of ``N`` arrays ``[B, T, d]``: slicing a stacked
    ``[N, B, T, d]`` carry and stacking it again in every loop are copies of the activations
    (1.4% of a training step on TPU v6e). Returns ``(lanes: tuple of N arrays [B, T, d], ys
    with a leading loop axis)``; :func:`combine`, :func:`update` and ``head.final_hidden`` take
    both forms.
    """
    if keep not in REMAT_POLICIES:
        raise ValueError(f"remat_keep must be one of {sorted(REMAT_POLICIES)}, got {keep!r}")
    lanes0 = tuple(e for _ in range(n_lanes))
    policy = REMAT_POLICIES[keep]
    if not remat:
        body = step
    elif policy is None:
        body = jax.checkpoint(step, prevent_cse=False)
    else:
        body = jax.checkpoint(step, prevent_cse=False, policy=policy)
    return jax.lax.scan(body, lanes0, xs)
