"""The Adam ``gain`` group: its learning rate (with the transfer rule and as a run-time input),
its leaves (``gain/*`` of the model tree), its state and its decoupled decay towards gain 1."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.config import Cfg, ModelCfg, OptimCfg
from bes.optim import adam, build, group_lrs
from bes.optim.build import ADAM_GROUPS, GROUPS, batch_factor, lr_inputs, width_factor
from bes.roles import role_of


def _cfg(**optim):
    model = ModelCfg(vocab_size=96, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                     n_loops=3, clock_planes=3, cand_dim=8, critic_hidden=8, gain_ffn=24,
                     gain_o=True, gain_v=True, gain_phases=2)
    return Cfg(model=model, optim=OptimCfg(**optim))


def _params(cfg, seed=0):
    from bes.model import model as M
    return jax.jit(functools.partial(M.init_params, cfg.model))(jax.random.PRNGKey(seed))


def _rand_like(tree, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda x: jnp.asarray(rng.standard_normal(x.shape).astype(np.float32) * scale), tree)


def test_gain_group_rate_follows_the_adam_rules():
    cfg = _cfg(lr_gain=0.02)
    lrs, w = group_lrs(cfg), width_factor(cfg)
    assert "gain" in GROUPS and "gain" in ADAM_GROUPS
    assert lrs["gain"] == pytest.approx(0.02 * w)
    small = _cfg(lr_gain=0.02, tokens_per_step=16384)
    assert group_lrs(small)["gain"] == pytest.approx(0.02 * w * batch_factor(small))
    long = group_lrs(_cfg(lr_gain=0.02, horizon_tokens=2e10))
    assert long["gain"] == pytest.approx(lrs["gain"])                # no data correction
    rates = lr_inputs(cfg)
    assert isinstance(rates["gain"], np.float32) and rates["gain"] == np.float32(lrs["gain"])
    assert group_lrs(Cfg())["gain"] == pytest.approx(OptimCfg().lr_gain * width_factor(Cfg()))


def test_gain_tables_are_the_gain_group_with_adam_state():
    cfg = _cfg()
    params = _params(cfg)
    opt = build(cfg, params)
    lrs = group_lrs(cfg)
    gain = {lp.path: lp for lp in opt.plan if lp.group == "gain"}
    assert set(gain) == {"gain/ffn", "gain/o", "gain/v"}
    assert all(lp.base_lr == lrs["gain"] for lp in gain.values())
    assert all(lp.group != "gain" for lp in opt.plan if not lp.path.startswith("gain/"))
    state = opt.init(params)
    for path, lp in gain.items():
        slot = state.slots[path]
        assert set(slot) == {"m", "v"}
        assert all(a.shape == lp.shape and a.dtype == jnp.float32 and float(jnp.max(jnp.abs(a))) == 0
                   for a in slot.values())
    with pytest.raises(KeyError, match="gain"):
        opt.update(_rand_like(params, 1), state, params, 1.0,
                   group_lr={k: v for k, v in lr_inputs(cfg).items() if k != "gain"})


@pytest.mark.parametrize("decay", [0.0, 0.3])
def test_gain_update_is_adam_with_the_gain_decay(decay):
    """Every gain leaf gets exactly ``adam.update(weight_decay=gain_decay)`` with the gain rate;
    the other Adam groups are never decayed; ``gain_decay = 0`` is the plain Adam step."""
    cfg = _cfg(gain_decay=decay)
    params = jax.tree.map(lambda a: a + 0.5, _params(cfg))          # non-zero tables: decay acts
    opt = build(cfg, params)
    state = opt.init(params)
    lrs, lr_scale = group_lrs(cfg), 0.7
    for t in (1, 2):
        grads = _rand_like(params, t)
        new, state_new = opt.update(grads, state, params, lr_scale)
        for name in ("ffn", "o", "v"):
            path = f"gain/{name}"
            want, want_slot = adam.update(params["gain"][name], grads["gain"][name], state.slots[path],
                                          lrs["gain"] * lr_scale, t, weight_decay=decay)
            np.testing.assert_allclose(np.asarray(new["gain"][name]), np.asarray(want), rtol=1e-6, atol=1e-7)
            for k in ("m", "v"):
                np.testing.assert_allclose(np.asarray(state_new.slots[path][k]), np.asarray(want_slot[k]),
                                           rtol=1e-6, atol=1e-7)
        emb, _ = adam.update(params["emb"]["tok"], grads["emb"]["tok"], state.slots["emb/tok"],
                             lrs["embed"] * lr_scale, t)
        np.testing.assert_allclose(np.asarray(new["emb"]["tok"]), np.asarray(emb), rtol=1e-6, atol=1e-7)
        params, state = new, state_new
    # with a zero gradient the decay alone moves the tables towards zero (gain 1)
    zero = jax.tree.map(jnp.zeros_like, params)
    fresh = opt.init(params)
    moved, _ = opt.update(zero, fresh, params, 1.0)
    for name in ("ffn", "o", "v"):
        a, b = np.asarray(moved["gain"][name]), np.asarray(params["gain"][name])
        np.testing.assert_allclose(a, b * (1.0 - lrs["gain"] * decay), rtol=1e-6)


def test_zero_decay_skips_the_multiply_and_matches_the_other_adam_groups():
    """``gain_decay = 0`` gives the same update as a plain Adam group with the same rate."""
    cfg = _cfg(lr_gain=0.01, lr_scalar=0.01)
    params = jax.tree.map(lambda a: a + 0.25, _params(cfg))
    grads = _rand_like(params, 3)
    assert cfg.optim.gain_decay == 0.0
    as_gain = build(cfg, params)
    as_scalar = build(cfg, params, lambda p: "scalar" if p.startswith("gain/") else role_of(p))
    a, _ = as_gain.update(grads, as_gain.init(params), params, 0.9)
    b, _ = as_scalar.update(grads, as_scalar.init(params), params, 0.9)
    for name in ("ffn", "o", "v"):
        np.testing.assert_array_equal(np.asarray(a["gain"][name]), np.asarray(b["gain"][name]))
