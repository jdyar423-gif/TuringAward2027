"""MuonH: Muon on the Frobenius sphere ("hyperball") for stacked matrices ``[..., m, n]``.

Ported from ``kschwethelm/looped-lm-scaling``, file ``nanochat/optim.py`` (MIT licence):
``muon_step_fused`` at lines 86-132 and the norm cache of ``MuonAdamW._step_muon`` at lines
237-239. Hyper-parameters as set up in that repository's ``nanochat/gpt.py``
(``GPT.setup_optimizer``, lines 650-663: momentum 0.95, 5 steps, no weight decay).

One step, per matrix ``W`` of the stack, with cached initial norm ``c = ||W_0||_F``:

    buf <- lerp(buf, G, 1 - mu)                    # lines 104-107: Nesterov-style momentum
    U   <- lerp(G, buf, mu)
    X   <- PolarExpress_5(U / (1.02 ||U||_F + 1e-6))   # lines 109-122
    X   <- X * c / max(||X||_F, 1e-10)             # lines 124-126: update norm = c
    W   <- W - lr X                                # lines 128-130
    W   <- W * c / max(||W||_F, 1e-10)             # lines 131-132: back onto the sphere

so ``||W||_F`` stays at ``c`` for the whole run and a step turns ``W`` by an angle of at most
``asin(lr)`` (about ``lr`` radians when the update is orthogonal to ``W``). Weight decay is omitted
(a first-order no-op on the sphere, as noted in the reference).

Zero matrices. If ``c = 0`` the update is ``0 * X`` and the projection multiplies by
``0 / 1e-10``: a matrix that starts at zero stays exactly zero forever. This is why every matrix
trained by this optimiser must be initialised non-zero (``bes.optim.build`` checks it); feature
paths that must start switched off use zero-initialised tables or scalars trained by Adam.

Differences from the reference: the iteration runs in a configurable dtype (the reference always
casts to bfloat16) and every norm, the rescale and the step are computed in float32 (the reference
computes the update norm and rescale in bfloat16). The momentum buffer and the cached norm are
float32 regardless of the parameter dtype.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bes.optim.polar import frob_norm, polar_express

#: Frobenius pre-normalisation factor of the reference (``nanochat/optim.py`` line 111).
PRENORM = 1.02


def init(param: jax.Array) -> dict[str, jax.Array]:
    """Optimiser state of one stacked leaf.

    param: ``[..., m, n]``. Returns ``{"mom": float32 [..., m, n] zeros,
    "norm": float32 [..., 1, 1]}``; ``norm`` is the cached initial Frobenius norm of every matrix
    of the stack (the reference caches it at the first step, before any update, lines 237-239).
    """
    if param.ndim < 2:
        raise ValueError(f"MuonH trains matrices [..., m, n], got shape {param.shape}")
    return {"mom": jnp.zeros(param.shape, jnp.float32), "norm": frob_norm(param)}


def update(param: jax.Array, grad: jax.Array, state: dict[str, jax.Array], lr: jax.Array | float,
           momentum: jax.Array | float = 0.95, *, ns_steps: int = 5,
           ns_dtype: jnp.dtype | str = jnp.float32,
           prenorm: float = PRENORM) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One MuonH step for a stack of matrices.

    param, grad: ``[..., m, n]`` (grad may have any float dtype; it is used in float32).
    state: from :func:`init`. lr, momentum: scalars (Python or traced float32).
    Returns ``(new_param, new_state)``; ``new_param`` has the dtype of ``param``.
    """
    g = grad.astype(jnp.float32)
    mu = jnp.asarray(momentum, jnp.float32)
    # Nesterov-style momentum in lerp form (reference lines 105-107).
    mom = state["mom"] + (1.0 - mu) * (g - state["mom"])
    u = g + mu * (mom - g)
    # Polar Express (lines 109-122), then the update norm is set to the cached norm (124-126).
    x = polar_express(u, steps=ns_steps, prenorm=prenorm, dtype=ns_dtype).astype(jnp.float32)
    c = state["norm"]
    x = x * (c / jnp.maximum(frob_norm(x), 1e-10))
    # Step and projection back to the sphere of radius c (lines 128-132).
    w = param.astype(jnp.float32) - jnp.asarray(lr, jnp.float32) * x
    w = w * (c / jnp.maximum(frob_norm(w), 1e-10))
    return w.astype(param.dtype), {"mom": mom, "norm": c}
