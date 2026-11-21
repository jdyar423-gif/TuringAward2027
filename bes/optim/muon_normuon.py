"""Muon with Polar Express, NorMuon second moment and cautious weight decay, for stacked
matrices ``[..., m, n]``.

Ported from ``karpathy/nanochat``, file ``nanochat/optim.py`` (MIT licence): ``muon_step_fused``
at lines 111-180, the state shapes and reduction axis at lines 378-383 and the learning-rate shape
factor at line 393. Hyper-parameters as set up in ``nanochat/gpt.py`` (``GPT.setup_optimizer``,
lines 446-452: momentum 0.95, 5 steps, ``beta2 = 0.9``); the momentum and weight-decay schedules of
``scripts/base_train.py`` (lines 371-386) are left to the caller (see ``bes.optim.build``).

One step, per matrix of the stack (line numbers of the reference):

    buf <- lerp(buf, G, 1 - mu);  U <- lerp(G, buf, mu)              # 130-133 Nesterov
    X   <- U with every output neuron rescaled to the mean norm      # 138-141 MuonEq (optional)
    X   <- PolarExpress_5(X / (1.01 ||X||_F + 1e-6))                 # 143-156
    X   <- X * sqrt(min(m, n)) / ||X||_F                             # 158-161 Muon+ (optional)
    q   <- mean of X^2 along the smaller dimension (one value per index of the larger one)
    v   <- lerp(v, q, 1 - beta2)                                     # 163-169 NorMuon
    X   <- X / sqrt(v), rescaled so that ||X||_F is unchanged        # 170-174 norm restoration
    lr' <- lr * sqrt(max(1, d_out / d_in))                           # 393
    W   <- W - lr' X - lr' wd W [X * W >= 0]                         # 176-180 cautious decay

The decay mask uses the final update direction ``X`` (as the reference does): decay is applied
only to entries where the update and the parameter have the same sign (zero counts as agreeing),
i.e. where the step already moves the entry towards zero.

Orientation. The reference stores weights as PyTorch ``nn.Linear`` does, ``[d_out, d_in]``. Three
pieces depend on which axis is the output: the learning-rate factor ``sqrt(max(1, d_out/d_in))``
(it makes the entry RMS of the update ``1/sqrt(d_in)`` for every shape), the row equilibration
(it equalises the update norms of the *output* neurons) and the reduction axis of the second
moment for square matrices (ties reduce over the input axis). ``layout="in_out"`` (the default)
means JAX-style ``y = x @ W`` with ``W: [..., d_in, d_out]``; ``layout="out_in"`` reproduces the
reference literally. The two are exact transposes of each other (tested). Polar Express itself,
Muon+, the norm restoration and the decay are orientation-free.

Differences from the reference: the iteration runs in a configurable dtype and the norms are
accumulated in float32; the buffers are float32 regardless of the parameter dtype.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from bes.optim.polar import frob_norm, polar_express

#: Frobenius pre-normalisation factor of the reference (``nanochat/optim.py`` line 144).
PRENORM = 1.01
LAYOUTS = ("in_out", "out_in")


def _io(shape: tuple[int, ...], layout: str) -> tuple[int, int, int]:
    """``(input_axis, d_in, d_out)`` of a stacked matrix shape under ``layout``."""
    if len(shape) < 2:
        raise ValueError(f"Muon trains matrices [..., m, n], got shape {shape}")
    m, n = int(shape[-2]), int(shape[-1])
    if layout == "in_out":
        return -2, m, n
    if layout == "out_in":
        return -1, n, m
    raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def lr_shape_factor(shape: tuple[int, ...], layout: str = "in_out") -> float:
    """``sqrt(max(1, d_out / d_in))`` (reference line 393, where ``d_out / d_in = rows / cols``)."""
    _, d_in, d_out = _io(shape, layout)
    return math.sqrt(max(1.0, d_out / d_in))


def second_moment_axis(shape: tuple[int, ...], layout: str = "in_out") -> int:
    """Axis (-2 or -1) that the NorMuon second moment is averaged over.

    The reference (line 383, ``red_dim = -1 if rows >= cols else -2`` on ``[d_out, d_in]``)
    reduces over the input axis when ``d_out >= d_in`` and over the output axis otherwise, i.e.
    along the smaller dimension, keeping one statistic per index of the larger one.
    """
    in_axis, d_in, d_out = _io(shape, layout)
    out_axis = -1 if in_axis == -2 else -2
    return in_axis if d_out >= d_in else out_axis


def init(param: jax.Array, layout: str = "in_out") -> dict[str, jax.Array]:
    """Optimiser state of one stacked leaf ``[..., m, n]``.

    Returns ``{"mom": float32 [..., m, n] zeros, "v": float32 zeros}`` where ``v`` has the shape of
    ``param`` with the reduced axis set to 1 (``[..., m, 1]`` or ``[..., 1, n]``, lines 380-382).
    """
    axis = second_moment_axis(param.shape, layout)
    v_shape = list(param.shape)
    v_shape[axis] = 1
    return {"mom": jnp.zeros(param.shape, jnp.float32), "v": jnp.zeros(tuple(v_shape), jnp.float32)}


def update(param: jax.Array, grad: jax.Array, state: dict[str, jax.Array], lr: jax.Array | float,
           momentum: jax.Array | float = 0.95, beta2: jax.Array | float = 0.9,
           weight_decay: jax.Array | float = 0.0, *, ns_steps: int = 5,
           ns_dtype: jnp.dtype | str = jnp.float32, layout: str = "in_out",
           row_equilibrate: bool = True, renormalize: bool = True,
           prenorm: float = PRENORM) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One Muon step for a stack of matrices.

    param, grad: ``[..., m, n]``. state: from :func:`init` with the same ``layout``.
    lr: the group learning rate *before* the shape factor, which is applied here (as the reference
    applies it to the group lr, line 393; the decay uses the same factor, line 180).
    momentum, beta2, weight_decay: scalars (Python or traced float32).
    row_equilibrate / renormalize: the MuonEq (lines 138-141) and Muon+ (lines 158-161) steps, on
    by default as in the reference.
    Returns ``(new_param, new_state)``; ``new_param`` has the dtype of ``param``.
    """
    shape = tuple(param.shape)
    in_axis, d_in, d_out = _io(shape, layout)
    ns_dtype = jnp.dtype(ns_dtype)
    g = grad.astype(jnp.float32)
    mu = jnp.asarray(momentum, jnp.float32)
    # Nesterov-style momentum (lines 130-133).
    mom = state["mom"] + (1.0 - mu) * (g - state["mom"])
    x = (g + mu * (mom - g)).astype(ns_dtype)
    # MuonEq: every output neuron's update rescaled to the mean output-neuron norm (138-141).
    if row_equilibrate:
        x32 = x.astype(jnp.float32)
        target = frob_norm(x32) / math.sqrt(d_out)
        row = jnp.maximum(jnp.sqrt(jnp.sum(x32 * x32, axis=in_axis, keepdims=True)), 1e-6)
        x = x * (target / row).astype(ns_dtype)
    # Polar Express (143-156).
    x = polar_express(x, steps=ns_steps, prenorm=prenorm, dtype=ns_dtype).astype(jnp.float32)
    # Muon+: Frobenius norm of an exactly semi-orthogonal matrix (158-161).
    if renormalize:
        x = x * (math.sqrt(min(shape[-2], shape[-1])) / jnp.maximum(frob_norm(x), 1e-6))
    # NorMuon variance reduction with norm restoration (163-174); no bias correction.
    axis = second_moment_axis(shape, layout)
    red = shape[axis]
    q = jnp.mean(x * x, axis=axis, keepdims=True)
    q_norm = jnp.sqrt(jnp.sum(q, axis=(-2, -1), keepdims=True) * red)
    v = state["v"] + (1.0 - jnp.asarray(beta2, jnp.float32)) * (q - state["v"])
    step = jax.lax.rsqrt(jnp.maximum(v, 1e-10))
    q_norm_new = jnp.sqrt(jnp.sum((q * red) * (step * step), axis=(-2, -1), keepdims=True))
    x = x * (step * (q_norm / jnp.maximum(q_norm_new, 1e-10)))
    # Cautious weight decay and the step (176-180, lr factor of line 393).
    lr_eff = jnp.asarray(lr, jnp.float32) * lr_shape_factor(shape, layout)
    wd = jnp.asarray(weight_decay, jnp.float32)
    w = param.astype(jnp.float32)
    agree = (x * w) >= 0
    w = w - (lr_eff * x + lr_eff * wd * w * agree)
    return w.astype(param.dtype), {"mom": mom, "v": v}
