"""Roles cover every parameter leaf and give the optimiser groups; the budget reproduces the
plan's ledger and its caps."""
from __future__ import annotations

import dataclasses
import functools
import os

import jax
import pytest

from _model_testlib import ROOT, config_model, init, tiny_cfg
from bes import budget, roles
from bes.config import CAND_FEATS, CAND_SLOTS, REG_FIELDS, load_config
from bes.model import model as M
from bes.optim.build import build, group_lrs, width_factor


def _shapes(cfg):
    return jax.eval_shape(functools.partial(M.init_params, cfg), jax.random.PRNGKey(0))


CONFIGS = {
    "final": lambda: config_model("final"),
    "no_library": lambda: config_model("no_library"),
    "smoke": lambda: config_model("smoke"),
    "features_off": lambda: dataclasses.replace(
        config_model("smoke"), use_sink=False, use_conf=False, use_reg=False, use_cand_in=False,
        use_cand_out=False, ve_dim=0, clock_planes=0),
}

LOOKUPS = ("emb/tok", "emb/ve", "emb/reg/", "cand/in_tok", "cand/in_feat/", "trust/feat/",
           "trust/bias/")
#: leaves of the Adam "feature" group (zero-initialised feature tables)
FEATURE = ("emb/reg/", "cand/in_tok", "cand/in_feat/", "trust/bias/")


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_leaf_has_a_role(name):
    tree = _shapes(CONFIGS[name]())
    leaves = roles.leaves_with_paths(tree)
    assert leaves
    for path, leaf in leaves:
        role = roles.role_of(path)
        assert role.kind in roles.KINDS and role.group in roles.GROUPS
        assert (role.kind == "lookup") == path.startswith(LOOKUPS), path
        if role.group == "matrix":
            assert len(leaf.shape) >= 2, path
    rt = roles.roles_tree(tree)
    assert len(jax.tree.leaves(rt, is_leaf=lambda x: isinstance(x, roles.Role))) == len(leaves)


def test_role_paths_and_unknown_leaves():
    assert roles.role_of("core/attn/wq") == roles.Role("dense", "matrix")
    assert roles.role_of(("emb", "reg", "role")) == roles.Role("lookup", "feature")
    assert roles.role_of("emb/tok") == roles.Role("lookup", "embed")
    assert roles.role_of("head/out").group == "unembed"
    assert roles.role_of("trust/feat/kind") == roles.Role("lookup", "unembed")
    assert roles.role_of("trust/bias/src") == roles.Role("lookup", "feature")
    assert roles.role_of("cand/in_tok").group == roles.role_of("cand/in_feat/mlen").group == "feature"
    assert roles.role_of("critic/w1").group == "frozen"
    with pytest.raises(KeyError):
        roles.role_of("core/attn/unknown")
    with pytest.raises(ValueError):
        roles.Role("dense", "bogus")


@pytest.mark.parametrize("name", ["final", "smoke"])
def test_optimiser_groups_of_the_model_leaves(name):
    """Every leaf of the real parameter tree gets its role's optimiser group and that group's
    rate: the feature group (register, candidate-input and trust-bias tables) gets
    ``lr_feature`` times the width factor, the trust feature tables the unembedding rate."""
    from bes.optim.build import GROUPS
    cfg = load_config(os.path.join(ROOT, "configs", f"{name}.py"))
    opt = build(cfg, _shapes(cfg.model))
    from bes.optim.build import batch_factor
    lrs, w = group_lrs(cfg), width_factor(cfg) * batch_factor(cfg)
    assert set(GROUPS) == set(roles.GROUPS)
    assert lrs["feature"] == pytest.approx(cfg.optim.lr_feature * w)
    assert lrs["unembed"] == pytest.approx(cfg.optim.lr_unembed * w)
    for lp in opt.plan:
        assert lp.group == roles.role_of(lp.path).group, lp.path
        assert lp.base_lr == lrs[lp.group], lp.path
        assert (lp.group == "feature") == lp.path.startswith(FEATURE), lp.path
        if lp.path.startswith("trust/feat/"):
            assert lp.group == "unembed", lp.path
    assert {lp.group for lp in opt.plan} == set(roles.GROUPS)


def test_final_budget_reproduces_the_plan_ledger():
    cfg = config_model("final")
    b = budget.count(_shapes(cfg), cfg, budget.PLAN_ASSETS)
    assert b.total <= 100e6 and b.active_step <= 20e6
    rows = b.ledger()
    d, V, L = cfg.d_model, cfg.vocab_size, cfg.n_loops
    dc_all = cfg.n_heads * (cfg.head_dim - cfg.bias_dims)
    # core: q/k project to content dims only, v/o to full heads, ReLU^2 MLP (plan: 14.16M)
    core = cfg.n_blocks * (2 * d * dc_all + 2 * d * cfg.n_heads * cfg.head_dim + 2 * d * cfg.d_ffn)
    assert rows["core"] == (core, core) and abs(core - 14.16e6) / 14.16e6 < 0.005
    assert rows["head"] == (V * 512 + 512 * d,) * 2 == (4_587_520,) * 2            # plan: 4.59M
    assert rows["emb"] == (V * d, d) and V * d == 6_291_456                        # plan: 6.29M / 768
    assert rows["ve"] == (L * V * 128, L * 128) == (8_388_608, 1_024)              # plan: 8.39M / 1,024
    assert rows["library"] == (60_000_000, 2 * 32_768) and rows["index"] == (4_000_000, 450)
    assert abs(rows["small"][0] - 0.35e6) < 0.1e6                                  # plan: 0.35M
    assert abs(b.total - 98.6e6) / 98.6e6 < 0.005                                  # plan: ~98.6M
    assert abs(b.active_step - 19.2e6) / 19.2e6 < 0.01                             # plan: ~19.2M
    # lookup reads: one row per read (trust biases: rows of width 1)
    reg_rows = len(REG_FIELDS) * d
    feat_rows = sum(min(CAND_SLOTS, c) for _, c in CAND_FEATS)
    cand_rows = CAND_SLOTS * cfg.cand_dim + 2 * feat_rows * cfg.cand_dim + feat_rows
    assert rows["tables"][1] == reg_rows + cand_rows
    bias = [i for i in b.items if i.name.startswith("trust/bias/")]
    assert len(bias) == len(CAND_FEATS) and all(i.width == 1 and i.kind == "lookup" for i in bias)
    assert sum(i.total for i in bias) == sum(c for _, c in CAND_FEATS)


def test_strict_count_formula():
    cfg = config_model("final")
    b = budget.count(_shapes(cfg), cfg, budget.PLAN_ASSETS)
    dense = sum(i.total for i in b.items if i.kind == "dense")
    assets = 2 * 32_768 + 450
    d, cd = cfg.d_model, cfg.cand_dim
    for T, D in ((1024, 325), (2048, 540), (2048, 2048), (8, 3)):
        reg = sum(min(card, T) for _, card in REG_FIELDS) * d
        feats = (2 * cd + 1) * sum(min(card, T * CAND_SLOTS) for _, card in CAND_FEATS)
        want = dense + D * (d + cfg.n_loops * cfg.ve_dim + cd) + reg + feats + assets
        assert b.active_strict(T, D) == want
    assert b.active_strict(1024, 100) < b.active_strict(1024, 200) < b.active_strict(2048, 400)
    with pytest.raises(ValueError):
        b.active_strict(10, 11)


def test_no_library_budget_within_caps():
    cfg = config_model("no_library")
    b = budget.count(_shapes(cfg), cfg)
    assert b.total <= 100e6 and b.active_step <= 20e6
    assert b.ledger()["ve"] == (cfg.n_loops * cfg.vocab_size * 512, cfg.n_loops * 512)
    assert b.ledger()["library"] == (0, 0)


def test_count_on_arrays_equals_count_on_shapes():
    cfg = tiny_cfg()
    a = budget.count(init(cfg), cfg, {"library_numbers": 1000, "library_read": 10})
    s = budget.count(_shapes(cfg), cfg, budget.Assets(library_numbers=1000, library_read=10))
    assert (a.total, a.active_step) == (s.total, s.active_step)
    assert a.total == sum(x.size for x in jax.tree.leaves(init(cfg))) + 1000
    assert "core" in a.report()
