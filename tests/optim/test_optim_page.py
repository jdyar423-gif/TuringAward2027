"""The Adam ``page`` group and the groups of the other new leaves: ``page/val`` in ``page`` (its
rate with the transfer rule and as a run-time input, its state, its decoupled decay),
``page/key`` in ``embed``, ``page/pin`` / ``page/pout`` in the spectral ``matrix`` group (non-zero at
initialisation, as MuonH requires) and the attention output gate in ``scalar``."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.config import Cfg, ModelCfg, OptimCfg
from bes.optim import adam, build, group_lrs, muonh
from bes.optim.build import ADAM_GROUPS, GROUPS, batch_factor, lr_inputs, width_factor
from bes.roles import role_of


def _cfg(**optim):
    model = ModelCfg(vocab_size=96, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                     n_loops=3, clock_planes=3, cand_dim=8, critic_hidden=8, tok_neurons=3,
                     tok_dim=12, attn_gate=True, gain_o=True)
    return Cfg(model=model, optim=OptimCfg(**optim))


def _params(cfg, seed=0):
    from bes.model import model as M
    return jax.jit(functools.partial(M.init_params, cfg.model))(jax.random.PRNGKey(seed))


def _rand_like(tree, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda x: jnp.asarray(rng.standard_normal(x.shape).astype(np.float32) * scale), tree)


def test_page_group_rate_follows_the_adam_rules():
    cfg = _cfg(lr_page=0.006)
    lrs, w = group_lrs(cfg), width_factor(cfg)
    assert "page" in GROUPS and "page" in ADAM_GROUPS
    assert lrs["page"] == pytest.approx(0.006 * w)
    small = _cfg(lr_page=0.006, tokens_per_step=16384)
    assert group_lrs(small)["page"] == pytest.approx(0.006 * w * batch_factor(small))
    assert group_lrs(_cfg(lr_page=0.006, horizon_tokens=2e10))["page"] == pytest.approx(lrs["page"])
    rates = lr_inputs(cfg)
    assert isinstance(rates["page"], np.float32) and rates["page"] == np.float32(lrs["page"])
    assert OptimCfg().lr_page == 0.004 and OptimCfg().page_decay == 0.0
    assert group_lrs(Cfg())["page"] == pytest.approx(0.004 * width_factor(Cfg()))


def test_groups_of_the_page_and_gate_leaves():
    cfg = _cfg()
    params = _params(cfg)
    opt = build(cfg, params)                              # MuonH check_init: pin, pout non-zero
    groups = opt.groups
    assert groups["page/val"] == "page" and groups["page/key"] == "embed"
    assert groups["page/pin"] == groups["page/pout"] == "matrix"
    assert groups["core/attn/gate_w"] == groups["core/attn/gate_b"] == "scalar"
    assert [p for p, g in groups.items() if g == "page"] == ["page/val"]
    lrs = group_lrs(cfg)
    assert all(lp.base_lr == lrs[lp.group] for lp in opt.plan)
    state = opt.init(params)
    assert set(state.slots["page/val"]) == {"m", "v"} and set(state.slots["page/key"]) == {"m", "v"}
    assert set(state.slots["page/pin"]) == {"mom", "norm"}
    assert all(float(jnp.min(state.slots[p]["norm"])) > 0 for p in ("page/pin", "page/pout"))
    with pytest.raises(KeyError, match="page"):
        opt.update(_rand_like(params, 1), state, params, 1.0,
                   group_lr={k: v for k, v in lr_inputs(cfg).items() if k != "page"})
    # a zero-initialised projection would be refused by MuonH
    zero = {**params, "page": {**params["page"], "pin": jnp.zeros_like(params["page"]["pin"])}}
    with pytest.raises(ValueError, match="page/pin"):
        build(cfg, zero)


@pytest.mark.parametrize("decay", [0.0, 0.3])
def test_page_update_is_adam_with_the_page_decay(decay):
    """``page/val`` gets exactly ``adam.update(weight_decay=page_decay)`` with the page rate; the
    keys are the plain ``embed`` Adam step, the projections MuonH; the gain decay is separate."""
    cfg = _cfg(page_decay=decay, gain_decay=0.1)
    params = jax.tree.map(lambda a: a + 0.5, _params(cfg))           # non-zero values: decay acts
    opt = build(cfg, params)
    state = opt.init(params)
    lrs, lr_scale = group_lrs(cfg), 0.7
    for t in (1, 2):
        grads = _rand_like(params, t)
        new, state_new = opt.update(grads, state, params, lr_scale)
        want, want_slot = adam.update(params["page"]["val"], grads["page"]["val"], state.slots["page/val"],
                                      lrs["page"] * lr_scale, t, weight_decay=decay)
        np.testing.assert_allclose(np.asarray(new["page"]["val"]), np.asarray(want), rtol=1e-6, atol=1e-7)
        for k in ("m", "v"):
            np.testing.assert_allclose(np.asarray(state_new.slots["page/val"][k]), np.asarray(want_slot[k]),
                                       rtol=1e-6, atol=1e-7)
        key, _ = adam.update(params["page"]["key"], grads["page"]["key"], state.slots["page/key"],
                             lrs["embed"] * lr_scale, t)
        np.testing.assert_allclose(np.asarray(new["page"]["key"]), np.asarray(key), rtol=1e-6, atol=1e-7)
        pin, _ = muonh.update(params["page"]["pin"], grads["page"]["pin"], state.slots["page/pin"],
                              lrs["matrix"] * lr_scale, cfg.optim.momentum, ns_steps=cfg.optim.ns_steps,
                              ns_dtype=opt.ns_dtype)
        np.testing.assert_allclose(np.asarray(new["page"]["pin"]), np.asarray(pin), rtol=1e-5, atol=1e-6)
        gain, _ = adam.update(params["gain"]["o"], grads["gain"]["o"], state.slots["gain/o"],
                              lrs["gain"] * lr_scale, t, weight_decay=0.1)
        np.testing.assert_allclose(np.asarray(new["gain"]["o"]), np.asarray(gain), rtol=1e-6, atol=1e-7)
        params, state = new, state_new
    zero = jax.tree.map(jnp.zeros_like, params)                      # decay alone moves the values
    moved, _ = opt.update(zero, opt.init(params), params, 1.0)
    np.testing.assert_allclose(np.asarray(moved["page"]["val"]),
                               np.asarray(params["page"]["val"]) * (1.0 - lrs["page"] * decay), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(moved["page"]["key"]), np.asarray(params["page"]["key"]))


def test_zero_page_decay_matches_a_plain_adam_group():
    """``page_decay = 0`` gives the same update as a plain Adam group with the same rate."""
    cfg = _cfg(lr_page=0.01, lr_scalar=0.01)
    params = jax.tree.map(lambda a: a + 0.25, _params(cfg))
    grads = _rand_like(params, 3)
    as_page = build(cfg, params)
    as_scalar = build(cfg, params, lambda p: "scalar" if p == "page/val" else role_of(p))
    a, _ = as_page.update(grads, as_page.init(params), params, 0.9)
    b, _ = as_scalar.update(grads, as_scalar.init(params), params, 0.9)
    np.testing.assert_array_equal(np.asarray(a["page"]["val"]), np.asarray(b["page"]["val"]))
