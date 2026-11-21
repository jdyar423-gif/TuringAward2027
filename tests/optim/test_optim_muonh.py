"""MuonH against an independent float64 numpy implementation, plus its invariants."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bes.optim import muonh

COEFFS = [  # published Polar Express coefficients (5 steps), typed independently of the module
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def np_polar(g, prenorm):
    x = g / (np.sqrt((g * g).sum()) * prenorm + 1e-6)
    for a, b, c in COEFFS:
        if x.shape[0] > x.shape[1]:
            gram = x.T @ x
            x = a * x + x @ (b * gram + c * gram @ gram)
        else:
            gram = x @ x.T
            x = a * x + (b * gram + c * gram @ gram) @ x
    return x


def np_muonh_step(w, g, buf, c, lr, mu):
    """Reference: one MuonH step, one matrix at a time (looped-lm-scaling muon_step_fused)."""
    w, buf = w.copy(), buf.copy()
    for k in range(w.shape[0]):
        buf[k] = buf[k] + (1 - mu) * (g[k] - buf[k])
        u = g[k] + mu * (buf[k] - g[k])
        x = np_polar(u, 1.02)
        x = x * (c[k] / max(np.linalg.norm(x), 1e-10))
        wk = w[k] - lr * x
        w[k] = wk * (c[k] / max(np.linalg.norm(wk), 1e-10))
    return w, buf


SHAPES = [(24, 16), (16, 24), (20, 20), (3, 12, 20), (2, 3, 20, 12)]


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_numpy_reference_step_by_step(shape):
    rng = np.random.default_rng(len(shape) * 100 + shape[-1])
    w0 = rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    m, n = shape[-2:]
    ref_w = w0.reshape(-1, m, n).astype(np.float64)
    ref_buf = np.zeros_like(ref_w)
    ref_c = np.sqrt((ref_w ** 2).sum(axis=(1, 2)))
    w, state = jnp.asarray(w0), muonh.init(jnp.asarray(w0))
    np.testing.assert_allclose(np.asarray(state["norm"]).reshape(-1), ref_c, rtol=1e-6)
    step = jax.jit(muonh.update)
    for t in range(6):
        g = (rng.standard_normal(shape) * (0.1 + t)).astype(np.float32)
        lr, mu = 0.03 * (1.0 - 0.1 * t), 0.95 if t != 3 else 0.6
        ref_w, ref_buf = np_muonh_step(ref_w, g.reshape(-1, m, n).astype(np.float64), ref_buf,
                                       ref_c, lr, mu)
        w, state = step(w, jnp.asarray(g), state, lr, mu)
        # measured max error ~3e-7 (float32 rounding) against per-step moves of ~1e-2
        np.testing.assert_allclose(np.asarray(w).reshape(-1, m, n), ref_w, atol=2e-6, rtol=0)
        np.testing.assert_allclose(np.asarray(state["mom"]).reshape(-1, m, n), ref_buf,
                                   atol=1e-5, rtol=1e-5)


def test_first_step_momentum_and_angle():
    """From zero momentum: buf = (1-mu) g, and the step rotates W by about lr radians."""
    rng = np.random.default_rng(1)
    w0 = jnp.asarray(rng.uniform(-0.5, 0.5, (32, 16)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((32, 16)).astype(np.float32))
    st = muonh.init(w0)
    w1, st1 = muonh.update(w0, g, st, lr=0.02, momentum=0.9)
    np.testing.assert_allclose(np.asarray(st1["mom"]), 0.1 * np.asarray(g), rtol=1e-6)
    cos = float(jnp.sum(w0 * w1) / (jnp.linalg.norm(w0) * jnp.linalg.norm(w1)))
    angle = float(np.arccos(min(cos, 1.0)))
    assert 0.01 < angle < 0.021  # the update has norm lr*||W0|| and is not parallel to W0


def test_norm_is_kept_on_the_sphere():
    rng = np.random.default_rng(2)
    w0 = jnp.asarray(rng.uniform(-1, 1, (4, 16, 40)).astype(np.float32) * np.arange(1, 5)[:, None, None])
    c0 = np.linalg.norm(np.asarray(w0), axis=(1, 2))
    w, st = w0, muonh.init(w0)
    step = jax.jit(muonh.update)
    for t in range(30):
        g = jnp.asarray(rng.standard_normal(w0.shape).astype(np.float32) * 10 ** rng.uniform(-3, 3))
        w, st = step(w, g, st, 0.05)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(w), axis=(1, 2)), c0, rtol=2e-6)
    assert float(jnp.max(jnp.abs(w - w0))) > 0.1  # it did move


def test_zero_matrix_stays_zero():
    """A matrix with zero initial norm never leaves zero: matrices must be initialised non-zero."""
    rng = np.random.default_rng(3)
    w0 = rng.uniform(-0.5, 0.5, (3, 8, 6)).astype(np.float32)
    w0[1] = 0.0
    w, st = jnp.asarray(w0), muonh.init(jnp.asarray(w0))
    for _ in range(5):
        w, st = muonh.update(w, jnp.asarray(rng.standard_normal(w0.shape).astype(np.float32)), st, 0.1)
    w = np.asarray(w)
    assert np.all(w[1] == 0.0)
    assert np.all(np.isfinite(w))
    assert np.abs(w[0] - w0[0]).max() > 1e-3 and np.abs(w[2] - w0[2]).max() > 1e-3


def test_zero_gradient_keeps_parameters_finite():
    w0 = jnp.asarray(np.random.default_rng(4).uniform(-1, 1, (8, 8)).astype(np.float32))
    w1, _ = muonh.update(w0, jnp.zeros_like(w0), muonh.init(w0), 0.1)
    np.testing.assert_allclose(np.asarray(w1), np.asarray(w0), rtol=1e-6, atol=1e-7)


def test_bfloat16_params_and_iteration():
    rng = np.random.default_rng(5)
    w0 = jnp.asarray(rng.uniform(-1, 1, (2, 16, 24)), jnp.bfloat16)
    st = muonh.init(w0)
    assert st["mom"].dtype == jnp.float32 and st["norm"].dtype == jnp.float32
    g = jnp.asarray(rng.standard_normal((2, 16, 24)), jnp.bfloat16)
    w1, st1 = muonh.update(w0, g, st, 0.02, ns_dtype=jnp.bfloat16)
    assert w1.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.linalg.norm(np.asarray(w1, np.float32), axis=(1, 2)),
                               np.asarray(st["norm"]).reshape(-1), rtol=1e-2)


def test_jit_and_grad_through_the_update():
    rng = np.random.default_rng(6)
    w0 = jnp.asarray(rng.uniform(-0.5, 0.5, (2, 10, 6)).astype(np.float32))
    g0 = jnp.asarray(rng.standard_normal((2, 10, 6)).astype(np.float32))
    st = muonh.init(w0)
    st = {"mom": jnp.asarray(rng.standard_normal((2, 10, 6)).astype(np.float32)), "norm": st["norm"]}
    probe = jnp.asarray(rng.standard_normal((2, 10, 6)).astype(np.float32))
    eager = muonh.update(w0, g0, st, 0.05)
    jitted = jax.jit(muonh.update)(w0, g0, st, 0.05)
    np.testing.assert_allclose(np.asarray(eager[0]), np.asarray(jitted[0]), atol=1e-6)

    def f(w, g):
        return jnp.sum(muonh.update(w, g, st, 0.05)[0] * probe)

    dw, dg = jax.jit(jax.grad(f, argnums=(0, 1)))(w0, g0)
    assert np.all(np.isfinite(np.asarray(dw))) and np.all(np.isfinite(np.asarray(dg)))
    for arg, d in ((0, dw), (1, dg)):  # central finite differences along a random direction
        v = jnp.asarray(rng.standard_normal((2, 10, 6)).astype(np.float32))
        eps = 1e-2
        args_p, args_m = [w0, g0], [w0, g0]
        args_p[arg] = args_p[arg] + eps * v
        args_m[arg] = args_m[arg] - eps * v
        fd = (float(f(*args_p)) - float(f(*args_m))) / (2 * eps)
        assert float(jnp.sum(d * v)) == pytest.approx(fd, rel=2e-2, abs=1e-3)


def test_rejects_vectors():
    with pytest.raises(ValueError):
        muonh.init(jnp.ones(5))
