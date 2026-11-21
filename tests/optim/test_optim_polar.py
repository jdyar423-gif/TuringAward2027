"""Polar Express: published coefficients, matrix-function identity, orthogonality, FLOP count."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bes.optim import polar

# Coefficients as printed in both reference files (Polar Express, 5 iterations, safety factor 2e-2,
# cushion 2), typed here independently of the module.
PUBLISHED = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def composed(s, steps=5):
    s = np.asarray(s, np.float64)
    for a, b, c in PUBLISHED[:steps]:
        s = a * s + b * s ** 3 + c * s ** 5
    return s


def test_coefficients_are_the_published_ones():
    assert len(polar.POLAR_EXPRESS_COEFFS) == 5
    np.testing.assert_array_equal(np.array(polar.POLAR_EXPRESS_COEFFS), np.array(PUBLISHED))


def test_composed_polynomial_band():
    """Every singular value in [1e-3, 1] lands in [0.8589, 1.1412]; tiny ones are not lifted."""
    s = np.concatenate([np.logspace(-3, 0, 200001), np.linspace(1e-3, 1.0, 200001)])
    out = composed(s)
    assert out.min() > 0.8588 and out.max() < 1.1412
    # The band is reached (equi-oscillation), not a trivially loose bound.
    assert out.min() < 0.8600 and out.max() > 1.1400
    assert composed(1e-5) < 0.1
    assert polar.scalar_map(0.3) == pytest.approx(float(composed(0.3)), rel=1e-12)


def svd_reference(g, prenorm=1.02):
    """U f(s) V^T with f the composed polynomial applied to the pre-normalised singular values."""
    g = np.asarray(g, np.float64)
    out = np.empty_like(g)
    flat_g, flat_o = g.reshape((-1,) + g.shape[-2:]), out.reshape((-1,) + g.shape[-2:])
    for k in range(flat_g.shape[0]):
        u, s, vt = np.linalg.svd(flat_g[k], full_matrices=False)
        s_in = s / (np.sqrt((flat_g[k] ** 2).sum()) * prenorm + 1e-6)
        flat_o[k] = (u * composed(s_in)) @ vt
    return out


@pytest.mark.parametrize("shape", [(24, 16), (16, 24), (20, 20), (3, 12, 20), (2, 2, 20, 12),
                                   (40, 1), (1, 40)])
def test_matches_matrix_function_identity(shape):
    rng = np.random.default_rng(sum(shape))
    g = rng.standard_normal(shape).astype(np.float32) * 3.0
    got = np.asarray(polar.polar_express(jnp.asarray(g), steps=5, prenorm=1.02))
    want = svd_reference(g)
    assert got.shape == g.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=0)


@pytest.mark.parametrize("steps", [1, 3])
def test_fewer_steps_use_the_first_coefficients(steps):
    rng = np.random.default_rng(steps)
    g = rng.standard_normal((18, 10)).astype(np.float32)
    u, s, vt = np.linalg.svd(g.astype(np.float64), full_matrices=False)
    want = (u * composed(s / (np.linalg.norm(g) * 1.01 + 1e-6), steps)) @ vt
    got = np.asarray(polar.polar_express(jnp.asarray(g), steps=steps, prenorm=1.01))
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=0)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (128, 16), (2, 48, 96)])
def test_orthogonality_after_five_steps(shape):
    """Random (well-conditioned) inputs: singular values within the band, close to U V^T."""
    rng = np.random.default_rng(7)
    g = rng.standard_normal(shape).astype(np.float32)
    x = np.asarray(polar.polar_express(jnp.asarray(g)), np.float64)
    k = min(shape[-2:])
    for xm, gm in zip(x.reshape((-1,) + shape[-2:]), g.reshape((-1,) + shape[-2:])):
        sv = np.linalg.svd(xm, compute_uv=False)
        assert sv.min() > 0.85 and sv.max() < 1.15, (sv.min(), sv.max())
        gram = xm.T @ xm if shape[-2] >= shape[-1] else xm @ xm.T
        orth_err = np.linalg.norm(gram - np.eye(k)) / np.sqrt(k)
        u, _, vt = np.linalg.svd(gm.astype(np.float64), full_matrices=False)
        polar_err = np.linalg.norm(xm - u @ vt) / np.sqrt(k)
        assert orth_err < 0.25 and polar_err < 0.12, (orth_err, polar_err)


def test_zero_input_gives_zero():
    x = polar.polar_express(jnp.zeros((2, 8, 5)))
    assert np.all(np.asarray(x) == 0.0)


def test_bfloat16_iteration():
    """bf16 (as in the references) orthogonalises as well as float32. The two outputs differ by
    ~10% because each singular value lands at a different point of the +-14% band, so quality is
    checked against the exact polar factor rather than against the float32 output."""
    rng = np.random.default_rng(3)
    g = rng.standard_normal((2, 32, 48)).astype(np.float32)
    x16 = polar.polar_express(jnp.asarray(g), dtype=jnp.bfloat16)
    assert x16.dtype == jnp.bfloat16
    x16 = np.asarray(x16.astype(jnp.float32), np.float64)
    for xm, gm in zip(x16, g.astype(np.float64)):
        sv = np.linalg.svd(xm, compute_uv=False)
        assert sv.min() > 0.8 and sv.max() < 1.2, (sv.min(), sv.max())
        u, _, vt = np.linalg.svd(gm, full_matrices=False)
        p = u @ vt
        assert np.linalg.norm(xm - p) / np.sqrt(32) < 0.15
        assert (xm * p).sum() / (np.linalg.norm(xm) * np.linalg.norm(p)) > 0.99


def test_rejects_bad_step_counts_and_vectors():
    with pytest.raises(ValueError):
        polar.polar_express(jnp.ones((4, 4)), steps=0)
    with pytest.raises(ValueError):
        polar.polar_express(jnp.ones((4, 4)), steps=6)
    with pytest.raises(ValueError):
        polar.polar_express(jnp.ones(4))


def test_flop_formula_matches_xla_cost_analysis():
    """Analytic matmul count == XLA's count of the dot ops; also proves the Gram side is small."""
    for shape in [(2, 64, 256), (2, 256, 64), (3, 48, 48)]:
        lowered = jax.jit(lambda g: polar.polar_express(g, steps=5)).lower(
            jax.ShapeDtypeStruct(shape, jnp.float32))
        cost = lowered.compile().cost_analysis()
        cost = cost[0] if isinstance(cost, (list, tuple)) else cost
        xla = float(cost["flops"])
        analytic = polar.polar_express_flops(shape, 5)
        # XLA also counts elementwise ops: allow a few percent above the matmul count.
        assert analytic <= xla <= analytic * 1.03, (shape, analytic, xla)
    # Large side used for the Gram matrix would cost (4 s l^2 + 2 l^3) per step instead.
    assert polar.polar_express_flops((768, 3072)) == 5 * (4 * 3072 * 768 ** 2 + 2 * 768 ** 3)
