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
    assert B.time_shares(FINAL) == {1024: 0.5, 2048: 0.5}
    assert B.plan_valid(FINAL, B.Plan("xla", False, 3, "onehot")).startswith("invalid shape")


# ------------------------------------------------------------ search with stand-ins
def standin_measure(table, calls):
    """``table[(plan key, T)] = tokens/s`` or an error string (a failed compile)."""
    def measure(plan, t):
        calls.append((plan.key, t))
        v = table.get((plan.key, t), 1000.0)
        if isinstance(v, str):
            return B.Measurement(plan, t, False, v)
        return B.Measurement(plan, t, True, shape=S.micro_shape(plan.apply(FINAL), t),
                             compile_s=60.0, step_s=262144 / v, tokens_per_s=v, steps=20)
    return measure


def run_search(table, mode="full", platform="tpu", soft=1e9, hard=1e9, known=None, now=0.0):
    calls = []
    ms = B.search(FINAL, standin_measure(table, calls), clock=lambda: now, soft_deadline=soft,
                  hard_deadline=hard, mode=mode, platform=platform, known=known)
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
    assert calls == [("splash-remat1-ms32-onehot", 1024), ("splash-remat1-ms32-onehot", 2048),
                     ("splash-remat0-ms16-onehot", 1024),
                     ("splash-remat1-ms32-take", 1024), ("splash-remat1-ms32-take", 2048),
                     ("xla-remat1-ms32-take", 1024)]
    failed = [m for m in ms if not m.ok]
    assert len(failed) == 1 and "RESOURCE_EXHAUSTED" in failed[0].reason
    sel = B.select_plan(ms, (1024, 2048), B.time_shares(FINAL))
    assert sel["plan"] == B.Plan("splash", True, 32, "take")
    assert sel["expected_tps"] == pytest.approx(0.5 * 320e3 + 0.5 * 290e3)
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
    # the canary: determinism, parity (dense itself), safe and full both learn the motif
    can = B.canary(cfg, B.Plan.of(cfg), motif_batches(cfg, 8, lens[0], seed=1), steps=8,
                   min_drop=0.02)
    assert can["determinism"]["ok"] and can["parity"]["ok"]
    assert can["safe"]["ok"] and can["full"]["ok"], (can["safe"], can["full"])
    assert can["ladder"] == {"rung": 0, "groups": [], "switches": None, "overrides": []}
    # the full configuration fails, the safe one passes: the first passing rung is chosen
    real = B.loss_drops
    seen = []

    def fake(losses, min_drop=0.02, tail=None):
        seen.append(1)
        return len(seen) != 2 and real(losses, min_drop, tail)     # call 2 is the full run

    monkeypatch.setattr(B, "loss_drops", fake)
    can = B.canary(cfg, B.Plan.of(cfg), motif_batches(cfg, 8, lens[0], seed=1), steps=8)
    assert can["ladder"]["rung"] == 1 and can["ladder"]["groups"] == ["sink"]
    assert can["ladder"]["switches"] == "sink:0" and can["rungs"]["1"]["ok"]
