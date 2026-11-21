"""nanochat Muon (Polar Express + NorMuon + cautious weight decay) against an independent float64
numpy implementation written in the reference's own layout ``[d_out, d_in]``."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bes.optim import muon_normuon as mn

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


def np_muon_step(w, g, buf, v, lr, mu, beta2, wd, row_eq=True, renorm=True):
    """Reference step on matrices in PyTorch layout [K, d_out, d_in] (nanochat muon_step_fused).

    Returns (w, buf, v, margin): margin = min over steps so far of |update * w| per entry, used to
    skip entries whose decay mask is numerically ambiguous between float32 and float64.
    """
    w, buf, v = w.copy(), buf.copy(), v.copy()
    rows, cols = w.shape[-2:]
    red_dim = -1 if rows >= cols else -2
    lr_eff = lr * max(1.0, rows / cols) ** 0.5
    margin = np.empty_like(w)
    for k in range(w.shape[0]):
        buf[k] = buf[k] + (1 - mu) * (g[k] - buf[k])
        x = g[k] + mu * (buf[k] - g[k])
        if row_eq:
            target = np.sqrt((x * x).sum()) / np.sqrt(rows)
            row_norm = np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), 1e-6)
            x = x * (target / row_norm)
        x = np_polar(x, 1.01)
        if renorm:
            x = x * (np.sqrt(min(rows, cols)) / max(np.sqrt((x * x).sum()), 1e-6))
        v_mean = (x * x).mean(axis=red_dim, keepdims=True)
        size = x.shape[red_dim]
        v_norm = np.sqrt(v_mean.sum() * size)
        v[k] = v[k] + (1 - beta2) * (v_mean - v[k])
        step = 1.0 / np.sqrt(np.maximum(v[k], 1e-10))
        v_norm_new = np.sqrt(((v_mean * size) * step ** 2).sum())
        x = x * (step * (v_norm / max(v_norm_new, 1e-10)))
        mask = (x * w[k]) >= 0
        margin[k] = np.abs(x * w[k])
        w[k] = w[k] - (lr_eff * x + lr_eff * wd * w[k] * mask)
    return w, buf, v, margin


def to_torch(a, layout):
    """[..., m, n] stored under ``layout`` -> [K, d_out, d_in]."""
    a = np.asarray(a, np.float64)
    a = a.reshape((-1,) + a.shape[-2:])
    return np.swapaxes(a, -1, -2) if layout == "in_out" else a


SHAPES = [(24, 16), (16, 24), (20, 20), (3, 12, 20), (2, 3, 20, 12)]


@pytest.mark.parametrize("layout", ["in_out", "out_in"])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_numpy_reference_step_by_step(shape, layout):
    rng = np.random.default_rng(len(shape) * 100 + shape[-1] + (layout == "out_in"))
    w0 = rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    w, state = jnp.asarray(w0), mn.init(jnp.asarray(w0), layout)
    ref_w = to_torch(w0, layout)
    ref_buf = np.zeros_like(ref_w)
    rows, cols = ref_w.shape[-2:]
    ref_v = np.zeros((ref_w.shape[0], rows, 1) if rows >= cols else (ref_w.shape[0], 1, cols))
    np.testing.assert_array_equal(to_torch(np.zeros(state["v"].shape), layout).shape, ref_v.shape)
    ambiguous = np.zeros(ref_w.shape, bool)
    step = jax.jit(mn.update, static_argnames=("layout",))
    for t in range(6):
        g = (rng.standard_normal(shape) * (0.1 + t)).astype(np.float32)
        lr, mu, beta2, wd = 0.02 * (1.0 - 0.1 * t), (0.95 if t != 3 else 0.7), 0.9, 0.5
        ref_w, ref_buf, ref_v, margin = np_muon_step(ref_w, to_torch(g, layout), ref_buf, ref_v,
                                                     lr, mu, beta2, wd)
        # A float32/float64 sign disagreement needs |update * w| below ~3e-8; such entries are
        # excluded for the rest of the run (none occurred with these seeds when measured).
        ambiguous |= margin < 1e-7
        w, state = step(w, jnp.asarray(g), state, lr, mu, beta2, wd, layout=layout)
        got = to_torch(w, layout)
        # measured max error ~2e-7 (float32 rounding) against per-step moves of ~1e-2
        np.testing.assert_allclose(got[~ambiguous], ref_w[~ambiguous], atol=2e-6, rtol=0)
        np.testing.assert_allclose(to_torch(state["mom"], layout), ref_buf, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(to_torch(state["v"], layout), ref_v, rtol=1e-4, atol=1e-9)
    assert ambiguous.mean() < 0.01


@pytest.mark.parametrize("row_eq,renorm", [(False, False), (True, False), (False, True)])
def test_optional_steps_match_reference(row_eq, renorm):
    rng = np.random.default_rng(11)
    shape = (2, 14, 22)
    w0 = rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    w, state = jnp.asarray(w0), mn.init(jnp.asarray(w0))
    ref_w, ref_buf = to_torch(w0, "in_out"), np.zeros((2, 22, 14))
    ref_v = np.zeros((2, 22, 1))
    for t in range(3):
        g = rng.standard_normal(shape).astype(np.float32)
        ref_w, ref_buf, ref_v, margin = np_muon_step(ref_w, to_torch(g, "in_out"), ref_buf, ref_v,
                                                     0.02, 0.95, 0.9, 0.0, row_eq, renorm)
        w, state = mn.update(w, jnp.asarray(g), state, 0.02, 0.95, 0.9, 0.0,
                             row_equilibrate=row_eq, renormalize=renorm)
        np.testing.assert_allclose(to_torch(w, "in_out"), ref_w, atol=2e-6, rtol=0)


def test_layouts_are_exact_transposes():
    rng = np.random.default_rng(12)
    for shape in [(2, 12, 20), (2, 20, 12), (16, 16)]:
        w0 = rng.uniform(-0.3, 0.3, shape).astype(np.float32)
        g = rng.standard_normal(shape).astype(np.float32)
        a, sa = mn.update(jnp.asarray(w0), jnp.asarray(g), mn.init(jnp.asarray(w0), "in_out"),
                          0.02, weight_decay=0.3, layout="in_out")
        wt, gt = np.swapaxes(w0, -1, -2), np.swapaxes(g, -1, -2)
        b, sb = mn.update(jnp.asarray(wt), jnp.asarray(gt), mn.init(jnp.asarray(wt), "out_in"),
                          0.02, weight_decay=0.3, layout="out_in")
        np.testing.assert_allclose(np.asarray(a), np.swapaxes(np.asarray(b), -1, -2), atol=1e-6)
        np.testing.assert_allclose(np.asarray(sa["v"]), np.swapaxes(np.asarray(sb["v"]), -1, -2),
                                   rtol=1e-5)


def test_shape_rules():
    # in_out: [d_in, d_out]. Up-projection 768 -> 3072 gets sqrt(4) = 2, down-projection gets 1.
    assert mn.lr_shape_factor((2, 768, 3072), "in_out") == 2.0
    assert mn.lr_shape_factor((2, 3072, 768), "in_out") == 1.0
    assert mn.lr_shape_factor((3072, 768), "out_in") == 2.0  # reference: rows/cols = 3072/768
    assert mn.lr_shape_factor((768, 768), "in_out") == 1.0
    # Second moment: averaged along the smaller dimension; square ties reduce over the input axis.
    assert mn.second_moment_axis((768, 3072), "in_out") == -2
    assert mn.second_moment_axis((3072, 768), "in_out") == -1
    assert mn.second_moment_axis((768, 768), "in_out") == -2
    assert mn.second_moment_axis((768, 768), "out_in") == -1
    assert mn.init(jnp.zeros((2, 8, 32)))["v"].shape == (2, 1, 32)
    assert mn.init(jnp.zeros((2, 32, 8)))["v"].shape == (2, 32, 1)
    with pytest.raises(ValueError):
        mn.lr_shape_factor((4, 4), "rows_cols")


def test_norm_restoration_and_step_length():
    """Without decay every step has Frobenius norm lr * factor * sqrt(min(m, n))."""
    rng = np.random.default_rng(13)
    for shape in [(2, 16, 40), (2, 40, 16)]:
        w = jnp.asarray(rng.uniform(-0.3, 0.3, shape).astype(np.float32))
        st = mn.init(w)
        lr = 0.01
        factor = mn.lr_shape_factor(shape)
        for _ in range(4):
            g = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
            w_new, st = mn.update(w, g, st, lr)
            np.testing.assert_allclose(np.linalg.norm(np.asarray(w_new - w), axis=(1, 2)),
                                       lr * factor * np.sqrt(16), rtol=1e-4)
            w = w_new


def test_cautious_decay_only_where_update_and_parameter_agree():
    rng = np.random.default_rng(14)
    shape = (2, 12, 30)
    w0 = jnp.asarray(rng.uniform(-1, 1, shape).astype(np.float32))
    g = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
    st = mn.init(w0)
    lr, wd = 0.01, 2.0
    factor = mn.lr_shape_factor(shape)
    plain, _ = mn.update(w0, g, st, lr, weight_decay=0.0)
    decayed, _ = mn.update(w0, g, st, lr, weight_decay=wd)
    direction = (np.asarray(w0) - np.asarray(plain)) / (lr * factor)  # the update X
    agree = direction * np.asarray(w0) >= 0
    assert 0.2 < agree.mean() < 0.8
    expected = np.asarray(plain) - lr * factor * wd * np.asarray(w0) * agree
    np.testing.assert_allclose(np.asarray(decayed), expected, atol=1e-6)
    # where they disagree the decay is not applied at all
    np.testing.assert_array_equal(np.asarray(decayed)[~agree], np.asarray(plain)[~agree])


def test_zero_init_matrix_moves():
    """Unlike MuonH, this optimiser can train a zero-initialised matrix."""
    w = jnp.zeros((8, 6))
    w1, _ = mn.update(w, jnp.asarray(np.random.default_rng(15).standard_normal((8, 6)), jnp.float32),
                      mn.init(w), 0.02, weight_decay=0.1)
    assert float(jnp.linalg.norm(w1)) > 0.01


def test_jit_and_grad_through_the_update():
    rng = np.random.default_rng(16)
    shape = (2, 10, 6)
    w0 = jnp.asarray(rng.uniform(-0.5, 0.5, shape).astype(np.float32))
    g0 = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
    st = mn.init(w0)
    st = {"mom": jnp.asarray(rng.standard_normal(shape).astype(np.float32)),
          "v": jnp.asarray(rng.uniform(0.01, 0.1, st["v"].shape).astype(np.float32))}
    probe = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
    eager = mn.update(w0, g0, st, 0.05)
    jitted = jax.jit(mn.update)(w0, g0, st, 0.05)
    np.testing.assert_allclose(np.asarray(eager[0]), np.asarray(jitted[0]), atol=1e-6)

    def f(w, g):
        return jnp.sum(mn.update(w, g, st, 0.05)[0] * probe)

    dw, dg = jax.jit(jax.grad(f, argnums=(0, 1)))(w0, g0)
    assert np.all(np.isfinite(np.asarray(dw))) and np.all(np.isfinite(np.asarray(dg)))
    for arg, d in ((0, dw), (1, dg)):
        v = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
        eps = 1e-2
        args_p, args_m = [w0, g0], [w0, g0]
        args_p[arg] = args_p[arg] + eps * v
        args_m[arg] = args_m[arg] - eps * v
        fd = (float(f(*args_p)) - float(f(*args_m))) / (2 * eps)
        assert float(jnp.sum(d * v)) == pytest.approx(fd, rel=2e-2, abs=1e-3)
