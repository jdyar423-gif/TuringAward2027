"""Speed switches of the parts before the first loop and after the last one
(``bes.model.head.FAST_CE``): each one computes the values of the reference code path to the
float rounding of the dtype involved.

``FAST_CE`` (the cross-entropy head with a hand-written backward pass) is compared off against
on: ``head.chunked_ce`` alone (values and the gradients of the hidden state, the projection and
the unembedding) in float64, float32 and bfloat16 with one and with several chunks, and
``value_and_grad(loss_fn)`` (loss, metrics, every gradient leaf) on a tiny configuration with the
live run's features (splash attention in interpret mode, bfloat16 activations, rematerialised
loops, sink, candidate tape in and out, registers, value embeddings, ``table_grad="take"``) and
in float32. The only arithmetic difference is the log-sum-exp without the row maximum (and the
rounding of the softmax and softcap-derivative products built from it); with several chunks in
bfloat16 the unembedding gradient is summed over the chunks in float32 and rounded once instead
of after every chunk. Tolerances, relative to the largest entry of each array (measured
differences in brackets):

* float64: ``1e-13`` [3e-16];
* float32: ``1e-6`` [2e-7]; the whole model ``5e-6`` [1.0e-6; the loss bit for bit];
* bfloat16: the cross-entropy values ``1e-6`` (float32 either way) [1.2e-7]; gradients at most
  one bfloat16 rounding step (``2^-7`` of the entry) per entry, on at most 1% of the entries
  [the hidden state and the projection bit for bit, the unembedding 0.02% of its entries]; the
  whole model with one chunk ``1e-6`` [the loss and 58 of 59 leaves bit for bit, the
  unembedding 1.6e-7]; with ``C`` chunks the unembedding ``C`` rounding steps at the scale of
  its largest entry [2.6e-3 with two, 3.8e-3 with three chunks].
"""
from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.config import ModelCfg
from bes.model import head, model as M

#: the live run's model (configs/final.py with table_grad=take) at a tiny size
LIVE = dict(attn_impl="splash", act_dtype="bfloat16", remat=True, clock_planes=0, use_conf=False,
            table_grad="take")
BF16_STEP = 2.0 ** -7          # one rounding step of bfloat16, relative to the entry (upper bound)


@contextlib.contextmanager
def switch(module, name: str, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def _assert_rounding_steps(got, want, steps: float, what: str, share: float = 0.01):
    """bfloat16 arrays: every entry within ``steps`` rounding steps of the reference entry, and
    at most ``share`` of the entries differ at all."""
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    diff = np.abs(g - w)
    assert np.all(diff <= steps * BF16_STEP * np.abs(w) + 1e-30), (what, float(np.max(diff / (np.abs(w) + 1e-30))))
    assert np.mean(diff > 0) <= share, (what, float(np.mean(diff > 0)))


# --------------------------------------------------------------------- head.chunked_ce
def _ce_case(act, n=80, chunk=32, vocab=512, r=32, d=64, seed=0, cap=15.0):
    cfg = dataclasses.replace(ModelCfg(), vocab_size=vocab, head_rank=r, d_model=d, loss_chunk=chunk,
                              act_dtype=act, softcap=cap)
    rng = np.random.default_rng(seed)
    ft = jnp.float64 if act == "float64" else jnp.float32
    h = jnp.asarray(rng.standard_normal((n, d)), ft)
    ph = {"proj": jnp.asarray(rng.standard_normal((d, r)) / np.sqrt(d), ft),
          "out": jnp.asarray(rng.standard_normal((vocab, r)) * 2.0, ft)}
    y = jnp.asarray(rng.integers(0, vocab, n), jnp.int32)
    w = jnp.asarray(rng.random(n), ft)

    def run(fast):
        def loss(ph, h):
            ce = head.chunked_ce(ph, h, y, cfg)
            return jnp.sum(w * ce), ce
        vg = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        with switch(head, "FAST_CE", fast):
            text = str(jax.make_jaxpr(loss)(ph, h)) + str(jax.make_jaxpr(vg)(ph, h))
            (_, ce), (g_ph, g_h) = jax.jit(vg)(ph, h)
        return text, ce, {"proj": g_ph["proj"], "out": g_ph["out"], "h": g_h}

    return cfg, run(False), run(True)


@pytest.mark.parametrize("chunk", [32, 80, 1000])
@pytest.mark.parametrize("act", ["float64", "float32"])
def test_fast_ce_has_the_values_and_gradients_of_the_reference(act, chunk):
    with jax.enable_x64(act == "float64"):
        cfg, (t_ref, ce_ref, g_ref), (t_fast, ce_fast, g_fast) = _ce_case(act, chunk=chunk)
        # the fast program: a custom VJP, no row maximum, no second logit product
        assert "custom_vjp" in t_fast and "custom_vjp" not in t_ref
        assert "reduce_max" not in t_fast and "reduce_max" in t_ref
        assert ce_fast.dtype == ce_ref.dtype and ce_fast.shape == ce_ref.shape == (80,)
        tol = 1e-13 if act == "float64" else 1e-6
        assert _rel(ce_fast, ce_ref) < tol
        for k in g_ref:
            assert g_fast[k].dtype == g_ref[k].dtype and g_fast[k].shape == g_ref[k].shape
            assert _rel(g_fast[k], g_ref[k]) < tol, (k, _rel(g_fast[k], g_ref[k]))


@pytest.mark.parametrize("chunk", [32, 80])
def test_fast_ce_in_bfloat16_differs_by_rounding_steps_only(chunk):
    _, (_, ce_ref, g_ref), (_, ce_fast, g_fast) = _ce_case("bfloat16", chunk=chunk)
    n_chunks = -(-80 // chunk)
    assert _rel(ce_fast, ce_ref) < 1e-6
    for k in ("proj", "h"):                   # through the bfloat16 cotangent of h @ P
        _assert_rounding_steps(g_fast[k], g_ref[k], 1, k)
    if n_chunks == 1:
        _assert_rounding_steps(g_fast["out"], g_ref["out"], 1, "out")
    else:
        # the reference rounds the running sum over the chunks to bfloat16 after every chunk
        # (a rounding at the scale of the partial sums, so small entries can lose all their
        # digits): at most one rounding step per chunk at the scale of the largest entry
        assert _rel(g_fast["out"], g_ref["out"]) <= n_chunks * BF16_STEP


def test_fast_ce_falls_back_where_its_assumptions_fail():
    """No softcap (logits unbounded: the unshifted sum could overflow) or a residual above
    ``FAST_CE_MAX_RESIDUAL``: the reference path runs."""
    for cap, limit in ((0.0, head.FAST_CE_MAX_RESIDUAL), (15.0, 1024)):
        cfg = dataclasses.replace(ModelCfg(), vocab_size=64, head_rank=8, d_model=16, loss_chunk=16,
                                  softcap=cap)
        ph = {"proj": jnp.ones((16, 8), jnp.float32), "out": jnp.ones((64, 8), jnp.float32)}
        h, y = jnp.ones((20, 16), jnp.float32), jnp.zeros((20,), jnp.int32)
        with switch(head, "FAST_CE", True), switch(head, "FAST_CE_MAX_RESIDUAL", limit):
            text = str(jax.make_jaxpr(lambda p: jnp.sum(head.chunked_ce(p, h, y, cfg)))(ph))
        assert "custom_vjp" not in text, (cap, limit)


def test_fast_ce_unshifted_sum_is_safe_at_the_softcap():
    """Every logit at +softcap or -softcap (the extremes ``|zc| <= cap`` allows): finite values
    and gradients that match the reference."""
    cfg = dataclasses.replace(ModelCfg(), vocab_size=256, head_rank=8, d_model=8, loss_chunk=64,
                              softcap=15.0)
    rng = np.random.default_rng(3)
    h = jnp.asarray(rng.standard_normal((24, 8)) * 50.0, jnp.float32)
    ph = {"proj": jnp.eye(8, dtype=jnp.float32),
          "out": jnp.asarray(np.sign(rng.standard_normal((256, 8))) * 40.0, jnp.float32)}
    y = jnp.asarray(rng.integers(0, 256, 24), jnp.int32)
    out = {}
    for fast in (False, True):
        with switch(head, "FAST_CE", fast):
            out[fast] = jax.jit(jax.value_and_grad(lambda p, hh: jnp.sum(head.chunked_ce(p, hh, y, cfg)),
                                                   argnums=(0, 1)))(ph, h)
    (l0, g0), (l1, g1) = out[False], out[True]
    assert np.isfinite(float(l1)) and abs(float(l1) - float(l0)) <= 1e-6 * abs(float(l0))
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g0)):
        assert np.all(np.isfinite(np.asarray(a)))
        assert _rel(a, b) < 1e-5


# ------------------------------------------------------------------- the whole model
def _model_case(cfg, module, name, seed=31):
    params = random_params(cfg, seed=seed)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=40, seed=seed + 1, seg_starts=((0,), (0, 17)))

    def run(on):
        vg = jax.value_and_grad(lambda p, c, b: M.loss_fn(p, c, b, cfg), has_aux=True)
        with switch(module, name, on):
            text = str(jax.make_jaxpr(vg)(params, consts, batch))
            return jax.jit(vg)(params, consts, batch), text

    (ref, t_ref), (got, t_got) = run(False), run(True)
    assert t_got != t_ref, f"{name} did not change the traced program"
    return ref, got


def _compare_model(ref, got, loss_tol, grad_tol, what, overrides=None):
    """Loss and metrics to ``loss_tol`` (relative), every gradient leaf to ``grad_tol`` relative
    to its largest entry (``overrides``: ``{leaf path: tolerance}``)."""
    (l_ref, m_ref), g_ref = ref
    (l_got, m_got), g_got = got
    assert np.isfinite(float(l_ref))
    assert abs(float(l_got) - float(l_ref)) <= loss_tol * abs(float(l_ref)), what
    for k in m_ref:
        np.testing.assert_allclose(np.asarray(m_got[k]), np.asarray(m_ref[k]), rtol=loss_tol, atol=1e-7,
                                   err_msg=f"{what}: metric {k}")
    fr, fg = jax.tree_util.tree_flatten_with_path(g_ref)[0], jax.tree_util.tree_flatten_with_path(g_got)[0]
    assert [p for p, _ in fr] == [p for p, _ in fg]
    for (path, a), (_, b) in zip(fg, fr):
        name = jax.tree_util.keystr(path)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert _rel(a, b) <= (overrides or {}).get(name, grad_tol), (what, name, _rel(a, b))


@pytest.mark.slow
@pytest.mark.parametrize("loss_chunk", [128, 64])
@pytest.mark.parametrize("act", ["bfloat16", "float32"])
def test_fast_ce_model_loss_and_gradients_match(act, loss_chunk):
    """80 rows in one chunk (as every training piece: 4,096 rows, ``loss_chunk`` 8,192) or two.
    Measured: bfloat16 with one chunk, the loss and 58 of the 59 gradient leaves bit for bit, the
    unembedding within 1.6e-7 of its largest entry (0.01% of its entries one rounding step
    apart); with two chunks the unembedding within 2.6e-3 (the reference rounds its running sum
    to bfloat16 after every chunk), the rest bit for bit; float32, the loss bit for bit and every
    leaf within 1.0e-6 of its largest entry."""
    cfg = tiny_cfg(n_loops=3, **dict(LIVE, act_dtype=act), loss_chunk=loss_chunk)
    ref, got = _model_case(cfg, head, "FAST_CE")
    n_chunks = -(-80 // loss_chunk)
    tol = 5e-6 if act == "float32" else 1e-6
    out_tol = n_chunks * BF16_STEP if act == "bfloat16" and n_chunks > 1 else tol
    _compare_model(ref, got, 1e-6, tol, f"FAST_CE {act} chunk {loss_chunk}",
                   overrides={"['head']['out']": out_tol})
