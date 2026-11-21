"""Guard state machine on synthetic loss traces: spikes, NaN, recovery, exhaustion, targets."""
import math

import numpy as np
import pytest

from bes.config import RunCfg
from bes.train.guard import (Action, GuardCfg, GuardState, observe, rebase, rollback_target)

CFG = GuardCfg(spike=1.3, patience=3, nonfinite_patience=3, ema=0.9, warmup=10, skip_steps=200,
               lr_factor=0.7, max_rollbacks=3)


def run(trace, cfg=CFG, state=None, start=1):
    state = GuardState() if state is None else state
    actions = []
    for i, loss in enumerate(trace):
        state, d = observe(state, start + i, loss, cfg)
        actions.append(d)
    return state, actions


def ref_ema(losses, decay):
    """Independent reference: plain mean of the first 1/(1-decay) losses, then exponential."""
    n0 = int(round(1 / (1 - decay)))
    if len(losses) <= n0:
        return float(np.mean(losses))
    m = float(np.mean(losses[:n0]))
    for x in losses[n0:]:
        m = decay * m + (1 - decay) * x
    return m


def warm(n=20, level=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return list(level + 0.05 * rng.standard_normal(n))


def test_good_losses_build_the_moving_average():
    trace = warm(30)
    state, acts = run(trace)
    assert all(d.action is Action.OK for d in acts)
    assert state.ema == pytest.approx(ref_ema(trace, 0.9), rel=1e-12)
    assert state.count == 30 and state.healthy and state.lr_mult == 1.0
    assert math.isinf(acts[5].threshold) and acts[-1].threshold == pytest.approx(1.3 * ref_ema(trace[:-1], 0.9))


def test_three_spikes_in_a_row_trigger_a_rollback():
    base = warm(20)
    state, acts = run(base + [5.0, 5.0, 5.0, 3.0])
    kinds = [d.action for d in acts[20:]]
    assert kinds == [Action.SUSPECT, Action.SUSPECT, Action.ROLLBACK, Action.OK]
    rb = acts[22]
    assert rb.reason == "spike" and rb.first_bad == 21 and rb.skip == 200
    assert rb.lr_mult == pytest.approx(0.7)
    assert state.rollbacks == 1 and state.lr_mult == pytest.approx(0.7)
    # spikes never entered the average
    assert state.ema == pytest.approx(ref_ema(base + [3.0], 0.9), rel=1e-12)


def test_recovery_resets_the_streak():
    state, acts = run(warm(20) + [5.0, 5.0, 3.0, 5.0, 5.0, 3.0])
    assert [d.action for d in acts[20:]] == [Action.SUSPECT, Action.SUSPECT, Action.OK,
                                             Action.SUSPECT, Action.SUSPECT, Action.OK]
    assert state.rollbacks == 0 and state.healthy and state.first_bad == -1


def test_exactly_at_the_threshold_is_good():
    state, _ = run([2.0] * 20)
    state, d = observe(state, 21, 1.3 * 2.0, CFG)
    assert d.action is Action.OK
    state, d = observe(state, 22, 1.3 * state.ema * (1 + 1e-9), CFG)
    assert d.action is Action.SUSPECT


def test_no_spike_judgement_during_warmup_but_nonfinite_counts():
    state, acts = run([9.0, 3.0, 20.0, 20.0, 20.0])  # big jumps while the average is cold
    assert all(d.action is Action.OK for d in acts)
    state, acts = run([3.0, float("nan"), float("inf"), float("-inf")])
    assert [d.action for d in acts] == [Action.OK, Action.SUSPECT, Action.SUSPECT, Action.ROLLBACK]
    assert acts[-1].reason == "nonfinite" and acts[-1].first_bad == 2


def test_immediate_rollback_on_nonfinite_when_configured():
    cfg = GuardCfg(**{**CFG.__dict__, "nonfinite_patience": 1})
    state, acts = run(warm(20) + [float("nan")], cfg)
    assert acts[-1].action is Action.ROLLBACK and acts[-1].first_bad == 21
    # a mixed streak: two spikes then a NaN with patience 1 for non-finite
    state, acts = run(warm(20) + [5.0, 5.0, float("nan")], cfg)
    assert [d.action for d in acts[20:]] == [Action.SUSPECT, Action.SUSPECT, Action.ROLLBACK]
    assert acts[-1].first_bad == 21


def test_exhaustion_spikes_are_ignored_and_nan_halts():
    trace = warm(20)
    for _ in range(3):
        trace += [6.0, 6.0, 6.0] + warm(15, seed=len(trace))
    state, acts = run(trace)
    assert sum(d.action is Action.ROLLBACK for d in acts) == 3
    assert state.rollbacks == 3 and state.lr_mult == pytest.approx(0.7 ** 3)
    s_spike, acts_spike = run([6.0, 6.0, 6.0, 6.1], state=state, start=1000)
    assert [d.action for d in acts_spike] == [Action.SUSPECT, Action.SUSPECT, Action.IGNORE,
                                              Action.OK]
    assert s_spike.ema == pytest.approx(ref_ema([6.0, 6.1], 0.9))  # re-based on the new level
    assert s_spike.rollbacks == 3 and s_spike.lr_mult == pytest.approx(0.7 ** 3)
    s_nan, acts_nan = run([float("nan")] * 3, state=state, start=1000)
    assert acts_nan[-1].action is Action.HALT and acts_nan[-1].first_bad == 1000


def test_rebase_forgets_the_level_but_keeps_counters():
    state, _ = run(warm(20) + [5.0, 5.0, 5.0])
    state = rebase(state)
    assert (state.count, state.streak, state.rollbacks) == (0, 0, 1)
    state, acts = run([4.5] * 5, state=state)  # higher level after a stage switch: no alarm
    assert all(d.action is Action.OK for d in acts)


def test_rollback_target():
    assert rollback_target([0, 100, 200], 201) == 100      # 200 produced the first bad loss
    assert rollback_target([0, 100, 200], 202) == 200
    assert rollback_target([200, 0, 100, 100], 150) == 100
    assert rollback_target([50, 60], 10) == 50              # nothing trusted: the oldest
    assert rollback_target([0], 1) == 0
    assert rollback_target([], 5) is None
    # a state re-saved at a rollback stays trusted when the streak starts right after it
    assert rollback_target([0, 15, 15], 16) == 0
    assert rollback_target([0, 15, 15], 16, verified_steps=[15]) == 15
    assert rollback_target([0, 15, 40], 41, verified_steps=[15]) == 15


def test_state_serialisation_and_config():
    state, _ = run(warm(20) + [5.0, 5.0])
    assert GuardState.from_dict(state.to_dict()) == state
    g = GuardCfg.from_run(RunCfg())
    assert (g.spike, g.patience, g.skip_steps, g.lr_factor, g.max_rollbacks) == (1.3, 3, 200, 0.7, 3)
    for bad in ({"spike": 1.0}, {"patience": 0}, {"ema": 1.0}, {"lr_factor": 0.0}, {"warmup": 0}):
        with pytest.raises(ValueError):
            GuardCfg(**bad)
