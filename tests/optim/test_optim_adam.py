"""Adam against an independent numpy implementation."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bes.optim import adam


def np_adam(p, g, m, v, lr, t, b1=0.8, b2=0.95, eps=1e-10):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


@pytest.mark.parametrize("shape", [(), (7,), (5, 3), (2, 4, 6)])
def test_matches_numpy_reference_step_by_step(shape):
    rng = np.random.default_rng(len(shape))
    p0 = rng.standard_normal(shape).astype(np.float32)
    p, st = jnp.asarray(p0), adam.init(jnp.asarray(p0))
    rp, rm, rv = p0.astype(np.float64), np.zeros(shape), np.zeros(shape)
    step = jax.jit(adam.update)
    for t in range(1, 9):
        g = (rng.standard_normal(shape) * 10 ** rng.uniform(-2, 2)).astype(np.float32)
        lr = 0.01 * t
        rp, rm, rv = np_adam(rp, g.astype(np.float64), rm, rv, lr, t)
        p, st = step(p, jnp.asarray(g), st, lr, t)
        np.testing.assert_allclose(np.asarray(p), rp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(st["m"]), rm, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(st["v"]), rv, rtol=1e-5, atol=1e-9)


def test_first_step_moves_by_lr_times_sign():
    g = jnp.asarray([3.0, -0.001, 1e4, -7.0])
    p, _ = adam.update(jnp.zeros(4), g, adam.init(jnp.zeros(4)), 0.1, 1)
    np.testing.assert_allclose(np.asarray(p), -0.1 * np.sign(np.asarray(g)), rtol=1e-6)


def test_rows_with_zero_gradient_do_not_move():
    rng = np.random.default_rng(0)
    table = jnp.asarray(rng.standard_normal((10, 4)).astype(np.float32))
    g = np.zeros((10, 4), np.float32)
    g[[1, 4]] = rng.standard_normal((2, 4))
    new, st = adam.update(table, jnp.asarray(g), adam.init(table), 0.3, 1)
    untouched = [i for i in range(10) if i not in (1, 4)]
    np.testing.assert_array_equal(np.asarray(new)[untouched], np.asarray(table)[untouched])
    assert np.all(np.asarray(new)[[1, 4]] != np.asarray(table)[[1, 4]])
    # a row with zero gradient so far keeps zero moments and stays put in later steps too
    new2, _ = adam.update(new, jnp.asarray(g), st, 0.3, 2)
    np.testing.assert_array_equal(np.asarray(new2)[untouched], np.asarray(table)[untouched])


def test_dtype_and_decay():
    p = jnp.ones(3, jnp.bfloat16)
    new, st = adam.update(p, jnp.ones(3, jnp.bfloat16), adam.init(p), 0.01, 1)
    assert new.dtype == jnp.bfloat16 and st["m"].dtype == jnp.float32
    # decoupled decay (unused in this project, ported for completeness): p (1 - lr wd) first
    q, _ = adam.update(jnp.full(2, 2.0), jnp.zeros(2), adam.init(jnp.zeros(2)), 0.1, 1,
                       weight_decay=0.5)
    np.testing.assert_allclose(np.asarray(q), 2.0 * (1 - 0.05), rtol=1e-6)
