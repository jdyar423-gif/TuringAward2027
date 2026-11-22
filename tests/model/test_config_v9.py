"""configs/v9.py and configs/v9_safe.py: the final allocation (the specification's part C,
section C5, without the attention gate: docs/V9.md section 8), the ledger with the ``gain`` and
``page`` rows under the caps, the same parameter tree with every switch off, and the roles and
read rules of the token pages and of the optional gate."""
import dataclasses
import functools
import os

import jax
import pytest

from _model_testlib import ROOT, tiny_cfg
from bes import budget, roles
from bes.config import load_config
from bes.model import model as M


def _cfg(name):
    return load_config(os.path.join(ROOT, "configs", f"{name}.py"))


def _shapes(model_cfg):
    return jax.eval_shape(functools.partial(M.init_params, model_cfg), jax.random.PRNGKey(0))


def test_v9_fields():
    cfg, base = _cfg("v9"), _cfg("no_library")
    m = cfg.model
    assert (m.ve_dim, m.gain_ffn, m.gain_o, m.gain_v, m.gain_phases) == (128, 0, True, False, 1)
    assert (m.tok_neurons, m.tok_dim, m.attn_gate) == (8, 192, False)
    assert (m.sel_attn, m.attn_impl, m.table_grad) == (False, "splash", "sorted")
    assert m.remat and m.act_dtype == "bfloat16"
    assert (cfg.run.name, cfg.run.out_dir) == ("v9", "runs/v9")
    assert cfg.data == base.data and not cfg.decode.use_library
    # the optimiser settings are those of the base: no decay on the page groups
    assert cfg.optim == base.optim
    assert cfg.optim.lr_page == 0.004 and cfg.optim.lr_gain == 0.01
    assert (cfg.optim.gain_decay, cfg.optim.page_decay) == (0.0, 0.0)
    M.check_cfg(m)


def test_v9_ledger_is_under_the_caps():
    """The whole ledger of configs/v9.py, and what the optional gate would add to it."""
    cfg = _cfg("v9").model
    b = budget.count(_shapes(cfg), cfg)
    print("\n" + b.report())
    assert b.within_caps() and b.total <= 100_000_000 and b.active_step <= 20_000_000
    rows = b.ledger()
    assert list(rows) == list(budget.LEDGER_ORDER)
    assert budget.LEDGER_ORDER.index("page") == budget.LEDGER_ORDER.index("gain") + 1
    v, nb, d, hw = cfg.vocab_size, cfg.n_blocks, cfg.d_model, cfg.n_heads * cfg.head_dim
    rho, dk = cfg.tok_neurons, cfg.tok_dim
    tables = 2 * v * nb * rho * dk
    proj = 2 * nb * d * dk
    gate = nb * d * cfg.n_heads + cfg.n_loops * nb * cfg.n_heads
    assert (tables, proj, v * nb * hw, gate) == (50_331_648, 589_824, 12_582_912, 9_312)
    assert rows["gain"] == (v * nb * hw, nb * hw) == (12_582_912, 1_536)
    assert rows["page"] == (tables + proj, proj + 2 * nb * rho * dk) == (50_921_472, 595_968)
    assert rows["library"] == rows["index"] == (0, 0)
    assert rows == {"core": (14_118_912, 14_118_912), "head": (4_587_520, 4_587_520),
                    "small": (306_776, 306_776), "emb": (6_291_456, 768), "ve": (8_388_608, 1_024),
                    "gain": (12_582_912, 1_536), "page": (50_921_472, 595_968),
                    "tables": (838_448, 18_736), "library": (0, 0), "index": (0, 0)}
    assert (b.total, b.active_step) == (98_036_104, 19_631_240)
    # the final model plus the pages, nothing else
    final = budget.count(_shapes(_cfg("final").model), _cfg("final").model)
    assert b.total - final.total == rows["gain"][0] + rows["page"][0]
    assert b.active_step - final.active_step == rows["gain"][1] + rows["page"][1]
    assert rows["small"] == final.ledger()["small"]
    assert not [i.name for i in b.items if i.name.startswith("core/attn/gate_")]
    page = {i.name: i for i in b.items if i.row == "page"}
    assert set(page) == {"page/pin", "page/pout", "page/key", "page/val"}
    assert page["page/pin"].kind == page["page/pout"].kind == "dense"
    for name in ("page/key", "page/val"):
        i = page[name]
        assert (i.kind, i.reads, i.strict, i.rows, i.width) == ("lookup", 1, "tokens", v, nb * rho * dk)
    # the optional gate (attn_gate=True) is dense, in the "small" row, and fits under the caps
    gated = dataclasses.replace(cfg, attn_gate=True)
    bg = budget.count(_shapes(gated), gated)
    gates = {i.name: i for i in bg.items if i.name.startswith("core/attn/gate_")}
    assert {n: (i.row, i.kind, i.total) for n, i in gates.items()} == {
        "core/attn/gate_w": ("small", "dense", nb * d * cfg.n_heads),
        "core/attn/gate_b": ("small", "dense", cfg.n_loops * nb * cfg.n_heads)}
    assert (bg.total - b.total, bg.active_step - b.active_step) == (gate, gate)
    assert bg.within_caps() and (bg.total, bg.active_step) == (98_045_416, 19_640_552)
    # strict reading: one row of every token table per distinct token of the window
    per_token = d + cfg.n_loops * cfg.ve_dim + cfg.cand_dim + nb * hw + 2 * nb * rho * dk
    assert per_token == 9_536
    assert b.active_strict(1024, 400) - b.active_strict(1024, 300) == 100 * per_token


def test_v9_safe_is_v9_with_every_switch_off():
    from bes.train.supervisor import parse_switches
    v9, safe = _cfg("v9"), _cfg("v9_safe")
    assert safe.model == v9.model and safe.optim == v9.optim and safe.data == v9.data
    assert (safe.run.name, safe.run.out_dir) == ("v9_safe", "runs/v9_safe")
    sw = parse_switches(safe.run.switches)
    assert sw == M.switches_off()
    en = M.make_consts(safe.model, **sw)["enable"]
    assert all(float(en[s]) == 0.0 for s in M.SWITCHES)
    full = M.make_consts(v9.model)["enable"]
    assert float(full["gain"]) == float(full["page"]) == 1.0
    assert float(full["sel"]) == float(full["gate"]) == 0.0    # not configured: off whatever the switch
    assert parse_switches(_cfg("safe").run.switches) == M.switches_off()


def test_gain_tables_role_read_rule_and_width_check():
    assert roles.role_of("gain/ffn") == roles.Role("lookup", "gain") == roles.role_of("gain/v")
    assert "gain" in roles.GROUPS
    cfg = tiny_cfg(gain_ffn=40, gain_o=True, gain_phases=3)
    b = budget.count(_shapes(cfg), cfg)
    hw = cfg.n_heads * cfg.head_dim
    assert b.ledger()["gain"] == (cfg.vocab_size * 3 * cfg.n_blocks * (40 + hw), 3 * cfg.n_blocks * (40 + hw))
    with pytest.raises(ValueError, match="gain/ffn"):
        budget.count(_shapes(cfg), tiny_cfg(gain_ffn=41, gain_o=True, gain_phases=3))
    assert budget.count(_shapes(tiny_cfg()), tiny_cfg()).ledger()["gain"] == (0, 0)


def test_page_and_gate_roles_read_rules_and_width_check():
    assert roles.role_of("page/key") == roles.Role("lookup", "embed")
    assert roles.role_of("page/val") == roles.Role("lookup", "page")
    assert roles.role_of("page/pin") == roles.role_of("page/pout") == roles.Role("dense", "matrix")
    assert roles.role_of("core/attn/gate_w") == roles.role_of("core/attn/gate_b") == roles.Role("dense", "scalar")
    assert "page" in roles.GROUPS
    cfg = tiny_cfg(tok_neurons=3, tok_dim=24, attn_gate=True)
    b = budget.count(_shapes(cfg), cfg)
    nb, d, w = cfg.n_blocks, cfg.d_model, cfg.n_blocks * 3 * 24
    assert b.ledger()["page"] == (2 * cfg.vocab_size * w + 2 * nb * d * 24, 2 * w + 2 * nb * d * 24)
    with pytest.raises(ValueError, match="page/"):
        budget.count(_shapes(cfg), tiny_cfg(tok_neurons=4, tok_dim=24, attn_gate=True))
    plain = budget.count(_shapes(tiny_cfg()), tiny_cfg())
    assert plain.ledger()["page"] == (0, 0)
    gate = nb * d * cfg.n_heads + cfg.n_loops * nb * cfg.n_heads
    assert b.ledger()["small"][0] - plain.ledger()["small"][0] == gate
    # strict reading: both tables one row per distinct token
    assert b.active_strict(64, 30) - b.active_strict(64, 20) - (
        plain.active_strict(64, 30) - plain.active_strict(64, 20)) == 10 * 2 * w
