"""Gradients: jax.test_util.check_grads on a tiny model (float64), scatter-free table gradients
equal the scatter-add gradients, and every spectral-optimiser matrix is non-zero at init."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from _model_testlib import config_model, init, make_batch, random_params, tiny_cfg, value_and_grad_fn
from bes.model import model as M, tables
from bes.roles import leaves_with_paths, role_of


def _micro_cfg(**kw):
    return tiny_cfg(vocab_size=64, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                    n_loops=2, cand_dim=8, critic_hidden=8, loss_chunk=7, **kw)


def test_check_grads_tiny_model_all_features_on():
    """Reverse-mode gradients of the full loss (registers, clocks, candidate input, the output
    law with trust features, trust biases and a src multiplier, sink, confidence, value
    embeddings, chunked loss with padding) against finite differences.

    Run in float64 (``jax.enable_x64``) so that finite differences are accurate; the step
    1e-6 keeps the O(eps) error of the ReLU^2 kink negligible. ``mass_offset = 3`` gives the
    candidate masses order one, and some targets are planted among the candidates."""
    cfg32 = _micro_cfg(mass_offset=3.0)
    params32 = random_params(cfg32, seed=21)
    batch = make_batch(cfg32, B=2, T=10, seed=22, seg_starts=((0,), (0, 6)))
    batch["cand_tok"][:, ::2, 0] = batch["tokens"][:, 1::2]
    batch["cand_feat"][:, ::2, 0, 0] = 1
    with jax.enable_x64(True):
        cfg = dataclasses.replace(cfg32, act_dtype="float64")
        params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), params32)
        consts = M.make_consts(cfg, src_5=0.5)
        f = jax.jit(lambda p: M.loss_fn(p, consts, batch, cfg)[0])
        assert f(params).dtype == jnp.float64
        met = jax.jit(lambda p: M.loss_fn(p, consts, batch, cfg)[1])(params)
        assert float(met["tape_mass"]) > 1.0 and abs(float(met["tape_gain"])) > 1e-2
        check_grads(f, (params,), order=1, modes=["rev"], eps=1e-6)


def test_onehot_table_gradients_equal_scatter_add_gradients():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=23)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=24)
    (l1, _), g1 = value_and_grad_fn(cfg)(params, consts, batch)
    (l2, _), g2 = value_and_grad_fn(dataclasses.replace(cfg, table_grad="take"))(params, consts, batch)
    assert float(l1) == float(l2)
    for (path, a), b in zip(leaves_with_paths(g1), jax.tree.leaves(g2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5, err_msg=path)


def test_compiled_loss_gradient_has_no_scatter():
    """Default path: table gradients, the output law (trust lookups, per-token mixing) and
    target selection compile without scatter (positive control: the ``take`` mode does
    produce scatter-adds)."""
    cfg = _micro_cfg()
    params = random_params(cfg, seed=25)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=10, seed=26, seg_starts=((0,), (0, 6)))
    hlo = jax.jit(jax.grad(lambda p: M.loss_fn(p, consts, batch, cfg)[0])).lower(params).compile().as_text()
    assert hlo.count(" scatter(") == 0
    idx = jnp.arange(6, dtype=jnp.int32) % 4
    ctrl = jax.jit(jax.grad(lambda t: jnp.sum(tables.big_lookup(t, idx, mode="take") ** 2))).lower(
        jnp.zeros((4, 3))).compile().as_text()
    assert ctrl.count(" scatter(") > 0


def _assert_matrices_nonzero(params):
    n = 0
    for path, leaf in leaves_with_paths(params):
        if role_of(path).group != "matrix":
            continue
        a = np.asarray(leaf)
        assert a.ndim >= 2, path
        norms = np.linalg.norm(a.reshape(-1, a.shape[-2], a.shape[-1]), axis=(-2, -1))
        assert np.all(norms > 0), (path, norms)
        n += 1
    return n


def test_spectral_matrices_nonzero_at_init():
    for cfg in (tiny_cfg(), config_model("smoke")):
        assert _assert_matrices_nonzero(init(cfg, seed=0)) >= 9


def test_spectral_matrices_nonzero_at_init_final_config():
    cfg = config_model("final")
    params = jax.jit(functools.partial(M.init_params, cfg))(jax.random.PRNGKey(0))
    assert _assert_matrices_nonzero(params) >= 9
    # zero-initialised feature tables really are zero
    for path, leaf in leaves_with_paths(params):
        if path.startswith(("emb/reg/", "cand/in_tok", "cand/in_feat/", "trust/feat/",
                            "trust/bias/")) or \
                path in ("core/attn/sink_logit", "core/attn/sink_value", "critic/w2"):
            assert float(jnp.max(jnp.abs(leaf))) == 0.0, path
