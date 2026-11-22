"""The start-up benchmark: plans, the search order and the selection with stand-in timings
(including a plan that fails to compile), the canary's decisions, the host-rate and horizon
arithmetic, and one real measurement and canary on a tiny model (CPU)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bes.config import BOS, Cfg, DataCfg, ModelCfg, OptimCfg, RunCfg, load_config
from bes.train import benchmark as B
from bes.train import schedule as S

ROOT = Path(__file__).resolve().parents[2]
FINAL = load_config(str(ROOT / "configs" / "final.py"))


# ------------------------------------------------------------------------------ plans
def test_plans_are_config_overrides():
    base = B.Plan.of(FINAL)
    assert base == B.Plan("splash", True, 32, "onehot")
    assert base.key == "splash-remat1-ms32-onehot"
    off = base.flip_remat()
    assert off == B.Plan("splash", False, 16, "onehot") and off.flip_remat() == base
    assert S.shapes(off.apply(FINAL)) == ((16, 16, 1024), (16, 8, 2048))
    assert S.shapes(base.apply(FINAL)) == ((8, 32, 1024), (8, 16, 2048))
    assert base.flip_table().table_grad == "take" and base.flip_attn().attn_impl == "xla"
    cfg = off.flip_table().apply(FINAL)
    assert (cfg.model.remat, cfg.optim.micro_seqs, cfg.model.table_grad) == (False, 16, "take")
    assert B.Plan.from_dict(json.loads(json.dumps(base.to_dict()))) == base
    assert B.seq_lens(FINAL) == (1024, 2048)
    assert B.time_shares(FINAL) == pytest.approx({1024: 0.67, 2048: 0.33})     # data.late_from = 0.67
    assert B.plan_valid(FINAL, B.Plan("xla", False, 3, "onehot")).startswith("invalid shape")
    # a configuration without selective attention: the plan never mentions it
    assert base.sel is None and len(base.overrides()) == 4 and base.flip_attn().sel is None
    assert B.Plan.from_dict({k: v for k, v in base.to_dict().items() if k != "sel"}) == base
    assert base.with_table("sorted").apply(FINAL).model.table_grad == "sorted"
    assert B.TABLE_MODES == ("sorted", "onehot", "take")


def test_v9_plan_and_plans_with_selective_attention():
    v9 = load_config(str(ROOT / "configs" / "v9.py"))
    assert B.Plan.of(v9) == B.Plan("splash", True, 32, "sorted") and B.Plan.of(v9).sel is None
    assert B.plan_valid(v9, B.Plan.of(v9)) == ""
    # selective attention (dense path only): the flash plan of such a configuration runs without it
    sel_cfg = dataclasses.replace(v9, model=dataclasses.replace(v9.model, attn_impl="xla",
                                                                sel_attn=True))
    base = B.Plan.of(sel_cfg)
    assert base == B.Plan("xla", True, 32, "sorted", True) and base.key == "xla-remat1-ms32-sorted-sel"
    flash = base.flip_attn()
    assert flash == B.Plan("splash", True, 32, "sorted", False)
    assert flash.key == "splash-remat1-ms32-sorted"
    assert flash.flip_attn() == base
    m = flash.apply(sel_cfg).model
    assert (m.attn_impl, m.sel_attn) == ("splash", False) and B.plan_valid(sel_cfg, flash) == ""
    twin = base.without_sel()
    assert twin.key == "xla-remat1-ms32-sorted" and not twin.apply(sel_cfg).model.sel_attn
    assert base.apply(sel_cfg).model.sel_attn and "model.sel_attn=true" in base.overrides()
    assert B.Plan.from_dict(json.loads(json.dumps(flash.to_dict()))) == flash
    # never selective attention on the flash path, never turned on for a configuration without it
    assert "dense attention path" in B.plan_valid(sel_cfg, B.Plan("splash", True, 32, "sorted", True))
    assert "dense attention path" in B.plan_valid(sel_cfg, B.Plan("splash", True, 32, "sorted"))
    assert "no selective attention" in B.plan_valid(v9, base)


# ------------------------------------------------------------ search with stand-ins
def standin_measure(table, calls, cfg=FINAL, default=1000.0):
    """``table[(plan key, T)] = tokens/s`` or an error string (a failed compile); other plans
    run at ``default`` tokens/s."""
    def measure(plan, t):
        calls.append((plan.key, t))
        v = table.get((plan.key, t), default)
        if isinstance(v, str):
            return B.Measurement(plan, t, False, v)
        shape = S.micro_shape(plan.apply(cfg), t)
        return B.Measurement(plan, t, True, shape=shape, compile_s=60.0,
                             step_s=shape[0] * shape[1] * shape[2] / v, tokens_per_s=v, steps=20)
    return measure


def run_search(table, mode="full", platform="tpu", soft=1e9, hard=1e9, known=None, now=0.0,
               cfg=FINAL, default=1000.0):
    calls = []
    ms = B.search(cfg, standin_measure(table, calls, cfg, default), clock=lambda: now,
                  soft_deadline=soft, hard_deadline=hard, mode=mode, platform=platform, known=known)
    return ms, calls


def test_search_order_and_selection_with_a_failed_compile():
    table = {
        ("splash-remat1-ms32-onehot", 1024): 300e3, ("splash-remat1-ms32-onehot", 2048): 280e3,
        # remat off does not fit (the TPU notes: 32.9 GiB of temporaries)
        ("splash-remat0-ms16-onehot", 1024): "RESOURCE_EXHAUSTED: HLO temporaries exceed HBM",
        ("splash-remat1-ms32-take", 1024): 320e3, ("splash-remat1-ms32-take", 2048): 290e3,
        ("xla-remat1-ms32-take", 1024): 250e3,
    }
    ms, calls = run_search(table)
    # the other table modes cheapest first: "sorted" (slower here: base length only), then "take"
    assert calls == [("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-onehot", 2048),
                     ("splash-remat0-ms16-onehot", 1024),
                     ("splash-remat1-ms32-sorted", 1024),
                     ("splash-remat1-ms32-take", 1024), ("splash-remat1-ms32-take", 2048),
                     ("xla-remat1-ms32-take", 1024)]
    failed = [m for m in ms if not m.ok]
    assert len(failed) == 1 and "RESOURCE_EXHAUSTED" in failed[0].reason
    sel = B.select_plan(ms, (1024, 2048), B.time_shares(FINAL))
    assert sel["plan"] == B.Plan("splash", True, 32, "take")
    assert sel["expected_tps"] == pytest.approx(0.67 * 320e3 + 0.33 * 290e3)
    assert sel["per_seq_len"][1024]["plan"] == "splash-remat1-ms32-take"
    # the dense plan was measured at the base length only (slower there)
    assert {c["plan"] for c in sel["candidates"]} == {"splash-remat1-ms32-onehot",
                                                      "splash-remat1-ms32-take"}


def test_selection_weights_the_lengths_by_wall_time():
    p1, p2 = B.Plan("splash", True, 32, "onehot"), B.Plan("splash", False, 16, "onehot")
    ms = [B.Measurement(p1, 1024, True, tokens_per_s=400.0), B.Measurement(p1, 2048, True, tokens_per_s=200.0),
          B.Measurement(p2, 1024, True, tokens_per_s=380.0), B.Measurement(p2, 2048, True, tokens_per_s=260.0)]
    sel = B.select_plan(ms, (1024, 2048), {1024: 0.5, 2048: 0.5})
    assert sel["plan"] == p2 and sel["per_seq_len"][1024]["plan"] == p1.key
    sel = B.select_plan(ms, (1024, 2048), {1024: 0.9, 2048: 0.1})
    assert sel["plan"] == p1
    # a plan that only runs at one length is never chosen for the run
    ms2 = [B.Measurement(p1, 1024, True, tokens_per_s=900.0), B.Measurement(p1, 2048, False, "OOM"),
           B.Measurement(p2, 1024, True, tokens_per_s=100.0), B.Measurement(p2, 2048, True, tokens_per_s=100.0)]
    assert B.select_plan(ms2, (1024, 2048), {1024: 0.5, 2048: 0.5})["plan"] == p2
    with pytest.raises(B.BenchmarkError, match="OOM"):
        B.select_plan(ms2[:2], (1024, 2048), {1024: 0.5, 2048: 0.5})


def test_remat_flip_wins_and_is_measured_at_both_lengths():
    table = {("splash-remat1-ms32-onehot", 1024): 300e3, ("splash-remat1-ms32-onehot", 2048): 300e3,
             ("splash-remat0-ms16-onehot", 1024): 350e3, ("splash-remat0-ms16-onehot", 2048): 340e3}
    ms, calls = run_search(table)
    assert ("splash-remat0-ms16-onehot", 2048) in calls
    # the table and attention flips start from the best complete plan (remat off)
    assert ("splash-remat0-ms16-take", 1024) in calls and ("xla-remat0-ms16-onehot", 1024) in calls
    assert B.select_plan(ms, (1024, 2048), B.time_shares(FINAL))["plan"].key == "splash-remat0-ms16-onehot"


def test_cpu_reduced_mode_time_budget_and_cache():
    cpu = dataclasses.replace(FINAL, model=dataclasses.replace(FINAL.model, attn_impl="xla"))
    calls = []
    ms = B.search(cpu, standin_measure({}, calls), clock=lambda: 0.0, soft_deadline=1e9,
                  hard_deadline=1e9, mode="reduced", platform="cpu")
    assert calls == [("xla-remat1-ms32-onehot", 1024), ("xla-remat1-ms32-onehot", 2048),
                     ("xla-remat0-ms16-onehot", 1024)]
    splash = [m for m in ms if m.plan.attn_impl == "splash"]
    assert splash and not splash[0].ok and "needs a TPU" in splash[0].reason
    # past the soft deadline only the configured plan is measured; past the hard one nothing
    ms, calls = run_search({}, soft=0.0, hard=1e9, now=1.0)
    assert calls == [("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-onehot", 2048)]
    assert all("time budget" in m.reason for m in ms if not m.ok)
    ms, calls = run_search({}, soft=0.0, hard=0.0, now=1.0)
    assert calls == [] and all("hard cap" in m.reason for m in ms)
    # measurements recorded by an earlier attempt are reused, not repeated
    known = {("splash-remat1-ms32-onehot", 1024): B.Measurement(B.Plan.of(FINAL), 1024, True, tokens_per_s=5.0)}
    ms, calls = run_search({}, known=known)
    assert ("splash-remat1-ms32-onehot", 1024) not in calls
    assert any(m.cached for m in ms)


V9 = load_config(str(ROOT / "configs" / "v9.py"))


def test_search_challenges_the_v9_plan_with_the_other_table_modes():
    """configs/v9.py ("sorted"): the one-hot and the scatter-add modes challenge it, cheapest
    first, and a mode only wins by measured tokens/s."""
    table = {("splash-remat1-ms32-sorted", 1024): 300e3, ("splash-remat1-ms32-sorted", 2048): 280e3,
             ("splash-remat0-ms16-sorted", 1024): "RESOURCE_EXHAUSTED",
             ("splash-remat1-ms32-onehot", 1024): 250e3, ("splash-remat1-ms32-take", 1024): 200e3}
    ms, calls = run_search(table, cfg=V9)
    assert calls == [("splash-remat1-ms32-sorted", 1024), ("splash-remat1-ms32-sorted", 2048),
                     ("splash-remat0-ms16-sorted", 1024), ("splash-remat1-ms32-onehot", 1024),
                     ("splash-remat1-ms32-take", 1024), ("xla-remat1-ms32-sorted", 1024)]
    assert B.select_plan(ms, (1024, 2048), B.time_shares(V9))["plan"] == B.Plan.of(V9)
    # the one-hot mode faster at both lengths: it is measured at both and chosen
    table[("splash-remat1-ms32-onehot", 1024)] = 320e3
    table[("splash-remat1-ms32-onehot", 2048)] = 290e3
    ms, calls = run_search(table, cfg=V9)
    assert ("splash-remat1-ms32-onehot", 2048) in calls and ("xla-remat1-ms32-onehot", 1024) in calls
    sel = B.select_plan(ms, (1024, 2048), B.time_shares(V9))
    assert sel["plan"].table_grad == "onehot" and "sel_attn" not in sel


@pytest.mark.parametrize("failing, expected", [
    # the configured mode fails at the long length only: the one-hot mode is measured at both
    # lengths although the time budget is used up, and runs
    ({("splash-remat1-ms32-sorted", 2048): "RESOURCE_EXHAUSTED"},
     [("splash-remat1-ms32-sorted", 1024), ("splash-remat1-ms32-sorted", 2048),
      ("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-onehot", 2048)]),
    # every table mode fails at the base length: then the attention flip
    ({("splash-remat1-ms32-sorted", 1024): "compile error", ("splash-remat1-ms32-sorted", 2048): "x",
      ("splash-remat1-ms32-onehot", 1024): "compile error", ("splash-remat1-ms32-take", 1024): "x"},
     [("splash-remat1-ms32-sorted", 1024), ("splash-remat1-ms32-sorted", 2048),
      ("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-take", 1024),
      ("xla-remat1-ms32-sorted", 1024), ("xla-remat1-ms32-sorted", 2048)]),
    # two faults: the flash kernel and the configured mode both fail on the chip: the dense
    # attention with the one-hot mode (both flipped) runs
    ({("splash-remat1-ms32-sorted", 1024): "Mosaic error", ("splash-remat1-ms32-sorted", 2048): "x",
      ("splash-remat1-ms32-onehot", 1024): "Mosaic error", ("splash-remat1-ms32-take", 1024): "x",
      ("xla-remat1-ms32-sorted", 1024): "sorted: runtime failure"},
     [("splash-remat1-ms32-sorted", 1024), ("splash-remat1-ms32-sorted", 2048),
      ("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-take", 1024),
      ("xla-remat1-ms32-sorted", 1024), ("xla-remat1-ms32-onehot", 1024),
      ("xla-remat1-ms32-onehot", 2048)]),
])
def test_a_failing_configured_plan_makes_its_proven_flips_essential(failing, expected):
    """Past the soft deadline only essential candidates are measured: when the configured plan
    fails at some length, its table-mode flips, then its attention flip, then both flips
    together are, until one plan runs at every length, so a run configured with a new mode
    never ends without a plan while a proven combination works."""
    ms, calls = run_search(failing, cfg=V9, soft=0.0, hard=1e9, now=1.0)
    assert calls == expected
    sel = B.select_plan(ms, (1024, 2048), B.time_shares(V9))
    assert sel["plan"].key == expected[-1][0]
    skipped = [m for m in ms if (m.plan.key, m.seq_len) not in calls]
    assert skipped and all("time budget" in m.reason for m in skipped)
    # past the hard deadline not even the essential candidates are measured
    _, calls = run_search(failing, cfg=V9, soft=0.0, hard=0.0, now=1.0)
    assert calls == []
    # within the time budget the rest of the search goes on as usual
    _, calls = run_search(failing, cfg=V9)
    assert calls[:len(expected)] == expected and len(calls) > len(expected)


def test_a_challenger_is_compared_with_plans_that_can_be_chosen():
    """The configured plan is the fastest at the base length but fails at the long one: a
    challenger faster than the plans that ran at every length is measured at the long length
    too (and chosen), although it is slower than the failed plan at the base length."""
    table = {("splash-remat1-ms32-sorted", 1024): 1000.0,
             ("splash-remat1-ms32-sorted", 2048): "RESOURCE_EXHAUSTED at 2048",
             ("splash-remat1-ms32-onehot", 1024): 800.0, ("splash-remat1-ms32-onehot", 2048): 800.0,
             ("splash-remat1-ms32-take", 1024): 950.0, ("splash-remat1-ms32-take", 2048): 950.0}
    ms, calls = run_search(table, cfg=V9, default=100.0)
    assert ("splash-remat1-ms32-take", 2048) in calls
    assert B.select_plan(ms, (1024, 2048), B.time_shares(V9))["plan"].key == "splash-remat1-ms32-take"
    # a challenger slower than the complete plan is still measured at the base length only
    assert ("splash-remat0-ms16-sorted", 1024) in calls and ("splash-remat0-ms16-sorted", 2048) not in calls


def _sel_cfg() -> Cfg:
    """A tiny configuration with selective attention (dense path)."""
    cfg = tiny_cfg()
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, sel_attn=True))


@pytest.mark.parametrize("splash_tps, kept", [(300.0, True), (340.0, False)])
def test_selective_attention_is_kept_only_within_the_speed_rule(splash_tps, kept):
    """With selective attention the flash plan runs without it; the dense plan with it is kept
    iff its expected tokens/s is at least run.sel_min_speed (0.85) times the best plan without
    it, and the selection reports the decision and both numbers."""
    cfg = _sel_cfg()
    lens = B.seq_lens(cfg)
    base = B.Plan.of(cfg)
    assert base.sel and base.attn_impl == "xla" and lens == (16, 32)
    flash = base.flip_attn()
    table = {(base.key, 16): 280.0, (base.key, 32): 280.0,
             (flash.key, 16): splash_tps, (flash.key, 32): splash_tps,
             (base.without_sel().key, 16): 290.0, (base.without_sel().key, 32): 290.0}
    ms, calls = run_search(table, cfg=cfg, default=1.0)    # remat and table-mode flips: slow
    assert (flash.key, 32) in calls                         # the flash plan ran at every length
    assert not flash.apply(cfg).model.sel_attn and base.apply(cfg).model.sel_attn
    sel = B.select_plan(ms, lens, B.time_shares(cfg), cfg.run.sel_min_speed)
    dec = sel["sel_attn"]
    assert dec["kept"] is kept and dec["min_speed"] == 0.85
    assert dec["sel_plan"] == base.key and dec["sel_tps"] == pytest.approx(280.0)
    assert dec["plain_plan"] == flash.key and dec["plain_tps"] == pytest.approx(splash_tps)
    assert dec["ratio"] == pytest.approx(280.0 / splash_tps)
    assert sel["plan"] == (base if kept else flash)
    # the dense plan without selective attention is the comparison on a host without the flash
    # kernel (here 280 / 290 = 0.97: kept)
    ms, calls = run_search(table, cfg=cfg, platform="cpu", default=1.0)
    assert (base.without_sel().key, 32) in calls and not any(k == flash.key and m.ok for k, m in
                                                             ((m.plan.key, m) for m in ms))
    dec = B.select_plan(ms, lens, B.time_shares(cfg))["sel_attn"]
    assert dec["kept"] and dec["plain_plan"] == base.without_sel().key


def test_selective_attention_switched_off_on_the_dense_path_and_when_nothing_else_runs():
    cfg = _sel_cfg()
    base = B.Plan.of(cfg)
    twin = base.without_sel()
    # on a CPU host (reduced mode): the dense plan without it, measured, is much faster
    ms, calls = run_search({(base.key, 16): 100.0, (base.key, 32): 100.0, (twin.key, 16): 200.0,
                            (twin.key, 32): 200.0}, cfg=cfg, mode="reduced", platform="cpu",
                           default=1.0)
    assert (twin.key, 16) in calls and (twin.key, 32) in calls
    sel = B.select_plan(ms, (16, 32), B.time_shares(cfg))
    assert sel["plan"] == twin and not sel["sel_attn"]["kept"]
    assert not twin.apply(cfg).model.sel_attn and twin.apply(cfg).model.attn_impl == "xla"
    # no plan without it ran at every length: kept as configured
    only = [m for m in ms if m.plan == base]
    dec = B.select_plan(only, (16, 32), B.time_shares(cfg))["sel_attn"]
    assert dec["kept"] and dec["plain_plan"] is None and "kept as configured" in dec["reason"]
    # the plan with it never ran at every length: the best plan without it, said so
    fail = [B.Measurement(base, 16, False, "x"), B.Measurement(twin, 16, True, tokens_per_s=5.0),
            B.Measurement(twin, 32, True, tokens_per_s=5.0)]
    sel = B.select_plan(fail, (16, 32), B.time_shares(cfg))
    assert sel["plan"] == twin and not sel["sel_attn"]["kept"]
    # a fault of the selective path (every plan with it fails) on a CPU host (no flash kernel):
    # the essential flips end with the dense plan without it, which runs
    bug = "bug in the selective path"
    ms, calls = run_search({(base.key, 16): bug, (base.with_table("sorted").key, 16): bug,
                            (base.with_table("take").key, 16): bug, (twin.key, 16): 50.0,
                            (twin.key, 32): 50.0}, cfg=cfg, platform="cpu", soft=0.0, now=1.0)
    assert calls == [(base.key, 16), (base.key, 32), (base.with_table("sorted").key, 16),
                     (base.with_table("take").key, 16), (twin.key, 16), (twin.key, 32)]
    assert any("needs a TPU" in m.reason for m in ms if m.plan == base.flip_attn())
    assert B.select_plan(ms, (16, 32), B.time_shares(cfg))["plan"] == twin


def test_results_file_round_trip(tmp_path):
    p = tmp_path / "results.jsonl"
    m = B.Measurement(B.Plan.of(FINAL), 1024, True, shape=(8, 32, 1024), compile_s=1.0, step_s=0.5,
                      tokens_per_s=524288.0, losses=[9.0, float("nan")])
    B.append_result(p, m)
    B.append_result(p, B.Measurement(B.Plan.of(FINAL), 2048, False, "OOM"))
    with open(p, "a") as f:
        f.write("{torn")
    got = B.load_results(p)
    assert list(got) == [("splash-remat1-ms32-onehot", 1024)]
    back = got[("splash-remat1-ms32-onehot", 1024)]
    assert back.shape == (8, 32, 1024) and back.tokens_per_s == 524288.0
    assert back.losses[0] == 9.0 and math.isnan(back.losses[1])


# ------------------------------------------------------------------- canary decisions
def test_loss_drops_and_the_ladder_decision():
    assert B.loss_drops([9.0, 8.9, 8.6, 8.5, 8.4, 8.3, 8.2])
    assert not B.loss_drops([9.0, 9.0, 8.99, 9.01, 9.0, 9.0])
    assert not B.loss_drops([9.0, 8.0, float("nan"), 7.0, 6.0, 5.0])
    assert not B.loss_drops([9.0])
    # one noisy batch at the end does not decide: the means of the first and last fifths do
    assert B.loss_drops([9.0, 9.0, 8.9, 8.8, 8.6, 8.5, 8.3, 8.2, 8.1, 8.0, 7.9, 8.6])
    assert not B.loss_drops([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.7])
    assert [B.canary_lr(i) for i in (0, 4, 9, 20)] == [0.1, 0.5, 1.0, 1.0]
    assert B.choose_rung(True, False, lambda k: pytest.fail("no rung needed"), 7) == (0, [])
    with pytest.raises(B.CanaryError, match="safe configuration"):
        B.choose_rung(False, False, lambda k: True, 7)
    assert B.choose_rung(False, True, lambda k: k == 3, 7) == (3, [1, 2, 3])
    assert B.choose_rung(False, True, lambda k: False, 7) == ("safe", list(range(1, 8)))


def test_host_rate_workers_and_horizon():
    calls = []
    t = [0.0]

    class Loader:
        def __call__(self, index, stage):
            calls.append(index)
            t[0] += 0.5

    stage = S.stage_at(0.0, FINAL)
    rate = B.host_rate(Loader(), stage, start=10, n=4, warm=2, clock=lambda: t[0])
    assert calls == list(range(10, 16)) and rate["next_index"] == 16
    assert rate["rows_per_s"] == pytest.approx(4 * 8 * 32 / 2.0)
    assert B.loader_workers_needed(400.0, 4, 300.0, 12) == 4          # keeps up
    assert B.loader_workers_needed(200.0, 4, 300.0, 12) == 7          # ceil(1.15 * 300 / 50)
    assert B.loader_workers_needed(100.0, 4, 1000.0, 12) == 12        # capped by the core plan
    hz = B.horizon(350e3, 300e3, 20 * 3600)
    assert hz["tokens_per_s"] == 300e3 and hz["horizon_tokens"] == pytest.approx(300e3 * 72000)
    assert B.horizon(350e3, None, 10.0)["horizon_tokens"] == pytest.approx(3.5e6)
    assert B.memory_limit_bytes({"backend": "cpu"}) is None
    assert B.memory_limit_bytes({"backend": "tpu", "memory_limit_gb": 31.25}) == int(28 * 2 ** 30)
    assert B.memory_limit_bytes({"backend": "tpu", "memory_limit_gb": 16}) == int(0.9 * 16 * 2 ** 30)


def test_reshape_batch_keeps_the_rows():
    a0, b0, t = 4, 8, 16
    batch = {"tokens": np.arange(a0 * b0 * (t + 1)).reshape(a0, b0, t + 1),
             "reg": np.arange(a0 * b0 * t * 2).reshape(a0, b0, t, 2)}
    out = B.reshape_batch(batch, (8, 4, t))
    assert out["tokens"].shape == (8, 4, t + 1) and out["reg"].shape == (8, 4, t, 2)
    assert np.array_equal(out["tokens"].reshape(-1, t + 1), batch["tokens"].reshape(-1, t + 1))
    with pytest.raises(ValueError):
        B.reshape_batch(batch, (3, 4, t))


# ------------------------------------------------- real measurement and canary (tiny)
def tiny_cfg() -> Cfg:
    return Cfg(model=ModelCfg(vocab_size=8192, d_model=32, n_heads=1, head_dim=32, ve_dim=32,
                              head_rank=16, n_loops=1, n_blocks=1, max_seq=64, clock_planes=3,
                              cand_dim=8, critic_hidden=8, loss_chunk=256),
               optim=OptimCfg(tokens_per_step=256, micro_seqs=4, lr_width_ref=32,
                              batch_exponent=0.0, data_exponent=0.0),
               data=DataCfg(seq_len=16, seq_len_late=32),
               run=RunCfg(switches=""))


def motif_batches(cfg: Cfg, n: int, t: int, seed: int = 0):
    """Rows of a repeated 4-token motif after BOS (learnable in a few steps)."""
    a, b, _ = S.micro_shape(cfg, t)
    out = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        motif = rng.integers(0, 50, (a, b, 4))
        tok = np.tile(motif, (1, 1, (t + 1) // 4 + 1))[..., :t + 1].astype(np.int32)
        tok[..., 0] = BOS
        out.append({"tokens": tok, "w": np.ones((a, b, t), np.float32),
                    "seg": np.zeros((a, b, t), np.int32),
                    "reg": np.zeros((a, b, t, 15), np.int16), "clock": np.zeros((a, b, t, 3), np.int32),
                    "cand_tok": np.full((a, b, t, 16), -1, np.int32),
                    "cand_feat": np.zeros((a, b, t, 16, 6), np.int8)})
    return out


def test_real_measurement_and_canary_on_a_tiny_model(monkeypatch):
    pytest.importorskip("jax")
    cfg = tiny_cfg()
    lens = B.seq_lens(cfg)
    batches = {t: motif_batches(cfg, 2, t) for t in lens}
    measure = B.make_measure(cfg, batches, steps=3, warmup=1)
    m = measure(B.Plan.of(cfg), lens[0])
    assert m.ok, m.reason
    assert m.shape == (4, 4, 16) and m.tokens_per_s > 0 and m.compile_s > 0
    assert m.memory and m.peak_bytes > 0 and len(m.losses) == 3
    # a plan the model refuses fails with its reason instead of stopping the benchmark
    bad = measure(B.Plan("xla", False, 4, "bogus"), lens[0])
    assert not bad.ok and "table_grad" in bad.reason
    programs = B.plan_programs(measure.programs, B.Plan.of(cfg))
    assert list(programs) == [(4, 4, 16)]                   # kept for the canary and the trainer
    # the canary: determinism, parity (dense itself), safe and full both learn the motif at the
    # base length and run at the long one; the measured program is reused, not compiled again
    canary_batches = {lens[0]: motif_batches(cfg, 8, lens[0], seed=1),
                      lens[1]: motif_batches(cfg, 2, lens[1], seed=2)}
    can = B.canary(cfg, B.Plan.of(cfg), canary_batches, steps=8, min_drop=0.02, programs=programs)
    assert can["compiled_reused"] == {"4x4x16": True, "4x2x32": False}
    assert can["determinism"]["ok"] and can["parity"]["ok"]
    assert can["safe"]["ok"] and can["full"]["ok"], (can["safe"], can["full"])
    for run in (can["safe"], can["full"]):
        assert run["skipped"] == 0 and run["shapes"]["4x2x32"]["ok"]
        assert len(run["shapes"]["4x2x32"]["losses"]) == 2
    assert can["ladder"] == {"rung": 0, "groups": [], "switches": None, "overrides": []}
    with pytest.raises(ValueError, match="every compiled shape"):    # the long length is missing
        B.canary(cfg, B.Plan.of(cfg), canary_batches[lens[0]], steps=8)
    # the full configuration fails, the safe one passes: the first passing rung is chosen
    real = B.loss_drops
    seen = []

    def fake(losses, min_drop=0.02, tail=None):
        seen.append(1)
        return len(seen) != 2 and real(losses, min_drop, tail)     # call 2 is the full run

    monkeypatch.setattr(B, "loss_drops", fake)
    can = B.canary(cfg, B.Plan.of(cfg), canary_batches, steps=8)
    # rungs 1-4 (gate, sel, gain, page) switch off paths this model does not have: not run again
    assert can["ladder"]["rung"] == 5
    assert can["ladder"]["groups"] == ["gate", "sel", "gain", "page", "sink"]
    assert can["ladder"]["switches"] == "gate:0 sel:0 gain:0 page:0 sink:0" and can["rungs"]["5"]["ok"]
    assert all(can["rungs"][str(k)]["same_as"] == "the full configuration" for k in (1, 2, 3, 4))
    assert len(seen) == 3                          # safe, full, rung 5


def test_canary_ladder_of_a_v9_model(monkeypatch):
    """A tiny model with the v9 paths (output gains, the attention gate, token neurons, the
    "sorted" table gradient): safe and full learn the motif; when the full configuration and
    rung 1 (gate off) fail, rung 2 (selective attention, absent here) is not run again and rung 3
    (token gains off) is tried and chosen."""
    pytest.importorskip("jax")
    from bes.model import model as MM

    cfg = tiny_cfg()
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, gain_o=True, attn_gate=True, tok_neurons=2, tok_dim=8, table_grad="sorted",
        sorted_chunk=16))
    lens = B.seq_lens(cfg)
    batches = {lens[0]: motif_batches(cfg, 8, lens[0], seed=1),
               lens[1]: motif_batches(cfg, 2, lens[1], seed=2)}
    can = B.canary(cfg, B.Plan.of(cfg), batches, steps=8)
    assert can["plan"] == "xla-remat0-ms4-sorted"
    assert can["safe"]["ok"] and can["full"]["ok"] and can["ladder"]["rung"] == 0
    assert can["safe"]["switches"] == MM.switches_off()
    real = B.loss_drops
    seen = []

    def fake(losses, min_drop=0.02, tail=None):
        seen.append(1)
        return len(seen) not in (2, 3) and real(losses, min_drop, tail)   # full and rung 1 fail

    monkeypatch.setattr(B, "loss_drops", fake)
    can = B.canary(cfg, B.Plan.of(cfg), batches, steps=8)
    assert can["ladder"]["rung"] == 3 and can["ladder"]["groups"] == ["gate", "sel", "gain"]
    assert can["ladder"]["switches"] == "gate:0 sel:0 gain:0"
    assert not can["rungs"]["1"]["ok"] and "losses" in can["rungs"]["1"]          # rung 1 ran
    assert can["rungs"]["2"]["same_as"] == "rung 1" and can["rungs"]["3"]["ok"]
    assert len(seen) == 4                            # safe, full, rung 1, rung 3


def test_canary_runs_every_shape_and_fails_on_skipped_updates(monkeypatch):
    """A backward-only fault of the long program (finite loss, NaN gradient) while the sink is
    on: the canary executes the T=32 program, the full configuration fails on its skipped
    updates and the ladder switches the sink off (before: the canary ran the base length only and
    passed). The same fault in every configuration stops the run with CanaryError."""
    import jax.numpy as jnp

    from bes.model import model as MM

    cfg = tiny_cfg()
    lens = B.seq_lens(cfg)
    real = MM.loss_fn

    def faulty(params, consts, batch, model_cfg, every=False):
        loss, m = real(params, consts, batch, model_cfg)
        if batch["tokens"].shape[-1] == lens[1] + 1:          # static: the long program only
            on = 1.0 if every else consts["enable"]["sink"]
            # value + 0; gradient sqrt'(0) * 0 = NaN when on == 1, 0 otherwise
            loss = loss + 0.0 * jnp.sqrt(0.0 * jnp.sum(params["head"]["g"]) + (1.0 - on))
        return loss, m

    monkeypatch.setattr(MM, "loss_fn", faulty)
    batches = {lens[0]: motif_batches(cfg, 8, lens[0], seed=1),
               lens[1]: motif_batches(cfg, 2, lens[1], seed=2)}
    can = B.canary(cfg, B.Plan.of(cfg), batches, steps=8)
    full, safe = can["full"], can["safe"]
    assert not full["ok"] and full["skipped"] == 2 and full["shapes"]["4x2x32"]["skipped"] == 2
    assert all(isinstance(x, float) for x in full["losses"])     # the base length is fine
    assert safe["ok"] and safe["skipped"] == 0
    assert can["ladder"]["rung"] == 5
    assert can["ladder"]["switches"] == "gate:0 sel:0 gain:0 page:0 sink:0"
    monkeypatch.setattr(MM, "loss_fn", lambda *a: faulty(*a, every=True))
    with pytest.raises(B.CanaryError, match="safe configuration"):
        B.canary(cfg, B.Plan.of(cfg), batches, steps=8)
