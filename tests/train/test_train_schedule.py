"""Schedule: progress under a fake clock (slow, fast, interrupted runs), curves of p, shapes."""
import math
from pathlib import Path

import numpy as np
import pytest

from bes.config import Cfg, DataCfg, OptimCfg, load_config
from bes.train import schedule as S
from bes.train.ledger import Ledger

ROOT = Path(__file__).resolve().parents[2]
FINAL = load_config(str(ROOT / "configs" / "final.py"))
T0 = 1.7e9


# ------------------------------------------------------------------ progress formula
def test_progress_formula_and_edges():
    assert S.progress(0.0, 100.0, 200.0) == 0.0
    assert S.progress(50.0, 150.0, 200.0) == pytest.approx(0.5)
    assert S.progress(10.0, 200.0, 200.0) == 1.0          # window closed
    assert S.progress(0.0, 250.0, 200.0) == 1.0           # closed even without training
    assert S.progress(30.0, 190.0, 200.0) == pytest.approx(30 / 40)
    assert S.train_end_time(T0 + 24 * 3600, FINAL.run) == pytest.approx(T0 + 19.5 * 3600)


def test_progress_is_linear_while_training_and_monotone_otherwise():
    end = T0 + 1000.0
    led = Ledger.start(T0, T0 + 100.0, phase="train")
    ps = []
    for k in range(1, 9):  # training: T + R constant -> p linear in time
        ps.append(S.progress_of(led, T0 + 100.0 + 100.0 * k, end))
    assert np.allclose(np.diff(ps), ps[0])
    led.switch("eval", T0 + 900.0)  # evaluation: T frozen, R shrinks -> p still grows
    q = [S.progress_of(led, T0 + 900.0 + s, end) for s in (0.0, 30.0, 60.0)]
    assert q[0] <= q[1] <= q[2] < 1.0
    assert S.progress_of(led, end, end) == 1.0


# ------------------------------------------------------------------ run simulator
def simulate(cfg, step_s, *, setup_s=900.0, ckpt_every=1500.0, ckpt_s=20.0, eval_every=7200.0,
             eval_s=120.0, crash_at=None, downtime=0.0, restart_setup_s=600.0):
    """A training run driven only by the schedule and the ledger (no JAX).

    Returns the kept trajectory [(dispatch time, p, stage)], the final clock, the train end and the
    final ledger. A crash discards every step after the last checkpoint snapshot."""
    clock = S.FakeClock(T0)
    deadline = T0 + cfg.run.deadline_hours * 3600.0
    end = S.train_end_time(deadline, cfg.run)
    led = Ledger.start(T0)
    clock.advance(setup_s)
    led.switch("train", clock())
    snap = led.snapshot(clock())
    traj, last_ckpt, last_eval, crashed = [], clock(), clock(), False
    while True:
        now = clock()
        if now + step_s > end:
            break
        p = S.progress_of(led, now, end)
        traj.append((now, p, S.stage_at(p, cfg)))
        clock.advance(step_s)
        led.add_step(traj[-1][2].tokens_per_step)
        if crash_at is not None and not crashed and clock() >= crash_at:
            crashed = True
            clock.advance(downtime)
            traj = [x for x in traj if x[0] + step_s <= snap.mark]
            led = Ledger.from_dict(snap.to_dict()).resumed(clock())
            clock.advance(restart_setup_s)
            led.switch("train", clock())
            continue
        if clock() - last_ckpt >= ckpt_every:
            led.switch("ckpt", clock())
            snap = led.snapshot(clock())
            clock.advance(ckpt_s)
            led.switch("train", clock())
            last_ckpt = clock()
        if clock() - last_eval >= eval_every:
            led.switch("eval", clock())
            clock.advance(eval_s)
            led.switch("train", clock())
            last_eval = clock()
    return traj, clock(), end, led


def switch_points(traj, attr):
    vals = [getattr(s, attr) for _, _, s in traj]
    return [i for i in range(1, len(vals)) if vals[i] != vals[i - 1]]


@pytest.mark.parametrize("step_s", [0.45, 0.8, 2.5])  # fast, nominal, slow
def test_runs_end_on_time_whatever_the_throughput(step_s):
    traj, t_end, end, led = simulate(FINAL, step_s)
    assert end - step_s < t_end <= end + 1e-6           # last step finishes inside the window
    ps = np.array([p for _, p, _ in traj])
    assert np.all(np.diff(ps) >= 0)                       # monotone
    # last dispatch: R = end - now < 2 steps, so p = T / (T + R) > 1 - 2 step / T
    assert ps[0] == 0.0 and ps[-1] >= 1 - 2 * step_s / led.train_seconds(t_end)
    assert S.progress_of(led, end, end) == 1.0
    # every second is accounted for
    assert led.accounted(t_end) == pytest.approx(t_end - T0, abs=1e-6)
    # the length switch and the chat-heavy tail each happen exactly once, at the right p
    for attr, at in (("seq_len", FINAL.data.late_from), ("assistant_only", FINAL.data.assistant_only_from)):
        idx = switch_points(traj, attr)
        assert len(idx) == 1
        assert ps[idx[0] - 1] < at <= ps[idx[0]]


def test_slow_and_fast_runs_switch_at_the_same_wall_time():
    times = {}
    for step_s in (0.45, 2.5):
        traj, _, end, _ = simulate(FINAL, step_s)
        i = switch_points(traj, "seq_len")[0]
        times[step_s] = traj[i][0]
    assert abs(times[0.45] - times[2.5]) <= 2.5 + 1e-6


@pytest.mark.parametrize("crash_hours,downtime", [(3.0, 1800.0), (11.0, 5400.0), (18.0, 600.0)])
def test_interrupted_runs_end_on_time(crash_hours, downtime):
    step_s = 0.8
    traj, t_end, end, led = simulate(FINAL, step_s, crash_at=T0 + crash_hours * 3600.0,
                                     downtime=downtime)
    assert end - step_s < t_end <= end + 1e-6
    ps = np.array([p for _, p, _ in traj])
    assert np.all(np.diff(ps) >= 0)                       # monotone along the kept trajectory
    assert ps[-1] >= 1 - 2 * step_s / led.train_seconds(t_end)
    assert led.restarts == 1
    assert led.seconds["lost"] >= downtime                # downtime plus discarded work
    assert led.seconds["lost"] <= downtime + 1500.0 + 2 * 20.0 + 2 * step_s
    assert led.accounted(t_end) == pytest.approx(t_end - T0, abs=1e-6)
    assert len(switch_points(traj, "seq_len")) == 1


def test_interrupted_run_loses_tokens_not_schedule():
    """With lost time the run sees fewer tokens but still covers the whole schedule."""
    full, _, end, led_full = simulate(FINAL, 0.8)
    cut, _, _, led_cut = simulate(FINAL, 0.8, crash_at=T0 + 10 * 3600.0, downtime=7200.0)
    assert led_cut.tokens < led_full.tokens
    assert cut[-1][1] == pytest.approx(full[-1][1], abs=1e-3)
    assert cut[-1][2].lr == pytest.approx(FINAL.optim.final_lr_frac, abs=0.01)


# ------------------------------------------------------------------ curves of p
def ref_lr(p, warmup, warmdown, final):
    warm = np.ones_like(p) if warmup <= 0 else np.interp(p, [0.0, warmup], [0.0, 1.0])
    down = np.interp(p, [1.0 - warmdown, 1.0], [1.0, final]) if warmdown > 0 else np.ones_like(p)
    return np.minimum(warm, down)


@pytest.mark.parametrize("warmup,warmdown,final", [(0.01, 0.65, 0.1), (0.0, 1.0, 0.1),
                                                   (0.6, 0.6, 0.0), (0.05, 0.0, 0.5)])
def test_lr_factor_shape(warmup, warmdown, final):
    o = OptimCfg(warmup_frac=warmup, warmdown_frac=warmdown, final_lr_frac=final)
    grid = np.linspace(0.0, 1.0, 2001)
    got = np.array([S.lr_factor(p, o) for p in grid])
    np.testing.assert_allclose(got, ref_lr(grid, warmup, warmdown, final), atol=1e-12)
    assert S.lr_factor(1.0, o) == pytest.approx(min(final, 1.0) if warmdown > 0 else 1.0)
    assert S.lr_factor(-0.5, o) == S.lr_factor(0.0, o) and S.lr_factor(7.0, o) == S.lr_factor(1.0, o)


def test_lr_factor_final_config_landmarks():
    o = FINAL.optim                      # reference shape: short warm-up, linear decay throughout
    assert (o.warmup_frac, o.warmdown_frac, o.final_lr_frac) == (0.002, 1.0, 0.1)
    assert S.lr_factor(0.0, o) == 0.0
    assert S.lr_factor(0.001, o) == pytest.approx(0.5)
    assert S.lr_factor(0.002, o) == pytest.approx(1.0 - 0.9 * 0.002)
    assert S.lr_factor(0.5, o) == pytest.approx(0.55)
    assert S.lr_factor(1.0, o) == pytest.approx(0.1)


def test_data_curves():
    d = DataCfg()
    grid = np.linspace(0.0, 1.0, 1001)
    ref_chat = np.where(grid >= d.tail_from, d.chat_frac_tail,
                        np.interp(grid, [d.late_from, d.tail_from], [d.chat_frac, d.chat_frac_tail]))
    np.testing.assert_allclose([S.chat_frac_at(p, d) for p in grid], ref_chat, atol=1e-12)
    assert [S.seq_len_at(p, d) for p in (0.0, 0.4999, 0.5, 1.0)] == [1024, 1024, 2048, 2048]
    assert [S.assistant_only_at(p, d) for p in (0.0, 0.919, 0.92, 1.0)] == [False, False, True, True]
    assert [S.data_phase_at(p, d) for p in (0.1, 0.6, 0.95)] == ["short", "long", "tail"]
    assert S.library_fracs_at(0.3, d) == (d.library_frac_base, d.library_frac_chat)
    assert S.library_fracs_at(0.3, DataCfg(use_library=False)) == (0.0, 0.0)
    step = DataCfg(late_from=0.95, tail_from=0.9)       # no ramp when the switch comes later
    assert S.chat_frac_at(0.89, step) == step.chat_frac and S.chat_frac_at(0.9, step) == step.chat_frac_tail


def ref_nanochat_momentum(it, num_iterations, warmdown_ratio=0.65):
    """Literal transcription of ``get_muon_momentum`` from ``karpathy/nanochat``
    ``scripts/base_train.py`` (lines 371-382, MIT licence), in iterations."""
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def test_momentum_schedule_matches_reference():
    n = 10_000
    cfg = Cfg(optim=OptimCfg(optimizer="muon", warmdown_frac=0.65))   # the nanochat ratio
    for it in list(range(0, 800, 7)) + list(range(3000, n, 37)):
        assert S.momentum_at(it, it / n, cfg) == pytest.approx(ref_nanochat_momentum(it, n), abs=1e-12)
    muonh = Cfg(optim=OptimCfg(optimizer="muonh", momentum=0.95))
    assert {S.momentum_at(it, it / n, muonh) for it in (0, 50, 9000)} == {0.95}


def test_weight_decay_schedule():
    cfg = Cfg(optim=OptimCfg(optimizer="muon"))
    for p in np.linspace(0, 1, 11):  # nanochat get_weight_decay: 0.5 * (1 + cos(pi * it / N))
        assert S.wd_scale_at(p, cfg) == pytest.approx(0.5 * (1 + math.cos(math.pi * p)))
    assert S.wd_scale_at(0.3, Cfg()) == 1.0


# ------------------------------------------------------------------ shapes and stages
def test_shapes_of_the_configs():
    assert S.shapes(FINAL) == ((8, 32, 1024), (8, 16, 2048))
    smoke = load_config(str(ROOT / "configs" / "smoke.py"))
    assert S.shapes(smoke) == ((4, 8, 128), (4, 4, 256))
    micro = load_config(str(ROOT / "configs" / "micro" / "base.py"))
    assert S.shapes(micro) == ((4, 16, 256),)
    for cfg in (FINAL, smoke, micro):
        for a, b, t in S.shapes(cfg):
            assert a * b * t == cfg.optim.tokens_per_step
    assert S.shapes(Cfg(data=DataCfg(late_from=1.5))) == ((8, 32, 1024),)
    assert S.shapes(Cfg(data=DataCfg(late_from=0.0))) == ((8, 16, 2048),)
    with pytest.raises(ValueError):
        S.micro_shape(FINAL, 3000)
    with pytest.raises(ValueError):
        S.micro_shape(Cfg(optim=OptimCfg(tokens_per_step=100_000)), 1024)


def test_stage_fields_and_regimes():
    assert FINAL.data.late_from == 0.67            # the last third of the run is at 2048 tokens
    s0, s1, s2 = (S.stage_at(p, FINAL) for p in (0.2, 0.7, 0.95))
    assert s0.shape == S.stage_at(0.6, FINAL).shape == (8, 32, 1024)
    assert s1.shape == s2.shape == (8, 16, 2048)
    assert {s.tokens_per_step for s in (s0, s1, s2)} == {262144}
    assert (s0.assistant_only, s1.assistant_only, s2.assistant_only) == (False, False, True)
    assert len({s0.regime, s1.regime, s2.regime}) == 3
    assert S.stage_at(0.7, FINAL).regime == S.stage_at(0.8, FINAL).regime  # chat ramp is continuous
    assert s1.lr == S.lr_factor(0.7, FINAL.optim) and s1.chat_frac == S.chat_frac_at(0.7, FINAL.data)
    assert S.stage_at(2.0, FINAL).p == 1.0
    assert set(s1.to_dict()) >= {"p", "lr", "seq_len", "chat_frac", "assistant_only"}


def test_fake_clock():
    c = S.FakeClock(5.0)
    assert c() == 5.0 and c.advance(2.5) == 7.5
    with pytest.raises(ValueError):
        c.advance(-1.0)
    with pytest.raises(ValueError):
        c.set(1.0)


def test_horizon_estimate_feeds_the_matrix_learning_rate():
    from bes.optim.build import group_lrs
    cfg = Cfg()
    tokens = S.estimate_horizon_tokens(300_000.0, 19.5 * 3600.0)
    assert tokens == pytest.approx(2.106e10)
    assert S.estimate_horizon_tokens(-1.0, 10.0) == 0.0
    with_h = S.with_horizon(cfg, tokens)
    assert with_h.optim.horizon_tokens == tokens and cfg.optim.horizon_tokens == 0.0
    base, scaled = group_lrs(cfg), group_lrs(with_h)
    assert scaled["matrix"] == pytest.approx(base["matrix"] * (tokens / 9.83e8) ** -0.32)
    assert scaled["embed"] == base["embed"]
