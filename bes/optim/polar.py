"""Polar Express orthogonalisation for stacked matrices ``[..., m, n]`` (shared by both spectral
optimisers).

Ported from two MIT-licensed sources, which carry identical coefficients and loop bodies:
- ``kschwethelm/looped-lm-scaling``, file ``nanochat/optim.py``: coefficients at lines 76-84,
  loop at lines 109-122 (pre-normalisation factor 1.02, line 111);
- ``karpathy/nanochat``, file ``nanochat/optim.py``: coefficients at lines 100-108, loop at
  lines 143-154 (pre-normalisation factor 1.01, line 144).
The coefficients are those published with the Polar Express paper (Amsel, Persson, Musco, Gower,
arXiv:2505.16932), computed for 5 iterations, safety factor 2e-2 and cushion 2.

What the iteration computes. Each step maps ``X -> a X + b X X^T X + c (X X^T)^2 X``, i.e. it applies
the odd polynomial ``p(s) = a s + b s^3 + c s^5`` to every singular value while keeping the singular
vectors. After the Frobenius pre-normalisation every singular value lies in ``(0, 1/prenorm]``; the
five composed polynomials send every singular value in ``[1e-3, 1]`` into ``[0.8589, 1.1411]``
(they equi-oscillate around 1 by design instead of converging to it), so the output is
``U diag(f(s)) V^T``, a cheap approximation of the polar factor ``U V^T`` of ``G = U S V^T``.
Singular values below ~1e-3 of the Frobenius norm are not lifted to the band (they are mapped
to ``f(s) < 0.86``).

Device notes. The Gram matrix is formed on the smaller side (``X^T X`` for tall inputs, ``X X^T``
otherwise, as in both references), so one iteration costs ``4 l s^2 + 2 s^3`` matmul FLOPs with
``s = min(m, n)``, ``l = max(m, n)``. The iteration runs in ``dtype`` (bfloat16 on TPU as in the
references, float32 for CPU tests); norms are always accumulated in float32. No matmul precision
is forced, so on TPU a float32 iteration runs at XLA's default matmul precision.
"""
from __future__ import annotations

import math
from typing import Sequence

import jax
import jax.numpy as jnp

#: (a, b, c) per iteration; identical in both reference files (see module docstring).
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
MAX_STEPS = len(POLAR_EXPRESS_COEFFS)


def frob_norm(x: jax.Array) -> jax.Array:
    """Frobenius norm of every matrix of a stack.

    x: ``[..., m, n]`` any float dtype. Returns float32 ``[..., 1, 1]`` (accumulated in float32).
    """
    x32 = x.astype(jnp.float32)
    return jnp.sqrt(jnp.sum(x32 * x32, axis=(-2, -1), keepdims=True))


def _check_steps(steps: int) -> None:
    if not 1 <= steps <= MAX_STEPS:
        # The references slice ``coeffs[:ns_steps]`` and silently run at most 5 steps; asking for
        # more is almost certainly a configuration error, so it is refused here.
        raise ValueError(f"Polar Express has coefficients for 1..{MAX_STEPS} steps, got {steps}")


def polar_express(g: jax.Array, steps: int = 5, prenorm: float = 1.02,
                  dtype: jnp.dtype | str = jnp.float32) -> jax.Array:
    """Approximate polar factor of every matrix of a stack.

    g: ``[..., m, n]`` (leading axes are independent matrices), any float dtype.
    steps: number of Polar Express iterations (1..5); the coefficient list is used in order.
    prenorm: Frobenius pre-normalisation factor (1.02 in the MuonH reference, 1.01 in nanochat).
    dtype: dtype of the iteration. Returns ``[..., m, n]`` in ``dtype``.

    Follows lines 110-122 (looped-lm-scaling) / 136 and 143-154 (nanochat): cast, divide by
    ``||X||_F * prenorm + 1e-6``, then 5 steps with the Gram matrix on the smaller side. The only
    numerical difference: the norm is accumulated in float32 and the division done in float32
    before the cast (the references divide in the iteration dtype).
    """
    _check_steps(steps)
    if g.ndim < 2:
        raise ValueError(f"polar_express needs [..., m, n], got shape {g.shape}")
    dtype = jnp.dtype(dtype)
    x = g.astype(dtype)
    x32 = x.astype(jnp.float32)
    x = (x32 / (frob_norm(x32) * prenorm + 1e-6)).astype(dtype)
    m, n = g.shape[-2], g.shape[-1]
    tall = m > n  # the references test size(-2) > size(-1); square inputs take the wide branch
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        if tall:
            gram = jnp.matmul(jnp.swapaxes(x, -1, -2), x)        # [..., n, n]
            poly = b * gram + c * jnp.matmul(gram, gram)
            x = a * x + jnp.matmul(x, poly)
        else:
            gram = jnp.matmul(x, jnp.swapaxes(x, -1, -2))        # [..., m, m]
            poly = b * gram + c * jnp.matmul(gram, gram)
            x = a * x + jnp.matmul(poly, x)
    return x


def scalar_map(s: float, steps: int = 5) -> float:
    """The composed polynomial applied to one singular value (for analysis and tests)."""
    _check_steps(steps)
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        s = a * s + b * s ** 3 + c * s ** 5
    return s


def polar_express_flops(shape: Sequence[int], steps: int = 5) -> int:
    """Matmul FLOPs (2 per multiply-add) of ``steps`` iterations on a stack of shape ``[..., m, n]``.

    Per iteration and matrix: Gram ``2 s^2 l``, Gram squared ``2 s^3``, product with X ``2 l s^2``,
    with ``s = min(m, n)`` and ``l = max(m, n)``. Elementwise work (about ``3 s^2 + 2 l s`` per
    iteration, plus O(m n) for momentum, norms and the step) is not included.
    """
    _check_steps(steps)
    *lead, m, n = (int(d) for d in shape)
    batch = math.prod(lead)
    s, l = min(m, n), max(m, n)
    return batch * steps * (4 * l * s * s + 2 * s ** 3)
