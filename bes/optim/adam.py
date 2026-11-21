"""Adam for the non-matrix groups (embeddings, unembedding, norm gains, scalars).

Ported from ``kschwethelm/looped-lm-scaling``, file ``nanochat/optim.py`` (MIT licence),
``adamw_step_fused`` at lines 20-49, with the float32 compute and write-back of the
``karpathy/nanochat`` version of the same function (``nanochat/optim.py`` lines 41-63). Group
settings as in looped-lm-scaling ``nanochat/gpt.py`` lines 639-649: betas (0.8, 0.95), eps 1e-10,
no weight decay.

One step at step number ``t`` (1-based):

    p <- p (1 - lr wd)                                   # line 39 (wd = 0 in this project)
    m <- lerp(m, g, 1 - b1);  v <- lerp(v, g^2, 1 - b2)   # lines 41-42
    p <- p - (lr / (1 - b1^t)) m / (sqrt(v / (1 - b2^t)) + eps)   # lines 44-49

A row of an embedding table whose gradient has been exactly zero since the first step keeps
``m = v = 0`` and therefore does not move (the update is ``0 / eps = 0``); once a row has received
gradient it keeps moving while its momentum decays, as in any dense Adam.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def init(param: jax.Array) -> dict[str, jax.Array]:
    """``{"m", "v"}``: float32 zeros of the shape of ``param``."""
    return {"m": jnp.zeros(param.shape, jnp.float32), "v": jnp.zeros(param.shape, jnp.float32)}


def update(param: jax.Array, grad: jax.Array, state: dict[str, jax.Array], lr: jax.Array | float,
           step: jax.Array | int, *, b1: float = 0.8, b2: float = 0.95, eps: float = 1e-10,
           weight_decay: jax.Array | float = 0.0) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One Adam step.

    param, grad: any shape, any float dtype (computed in float32, written back in the dtype of
    ``param``). state: from :func:`init`. lr: scalar. step: the 1-based number of this step
    (scalar int or float, may be traced). weight_decay: decoupled decay (0 for every group of
    this project; a Python zero skips the multiply). Returns ``(new_param, new_state)``.
    """
    g = grad.astype(jnp.float32)
    lr = jnp.asarray(lr, jnp.float32)
    t = jnp.asarray(step, jnp.float32)
    p = param.astype(jnp.float32)
    if not (isinstance(weight_decay, (int, float)) and weight_decay == 0):
        p = p * (1.0 - lr * jnp.asarray(weight_decay, jnp.float32))
    m = state["m"] + (1.0 - b1) * (g - state["m"])
    v = state["v"] + (1.0 - b2) * (g * g - state["v"])
    bias1 = 1.0 - jnp.power(jnp.float32(b1), t)
    bias2 = 1.0 - jnp.power(jnp.float32(b2), t)
    denom = jnp.sqrt(v / bias2) + eps
    p = p - (lr / bias1) * (m / denom)
    return p.astype(param.dtype), {"m": m, "v": v}
