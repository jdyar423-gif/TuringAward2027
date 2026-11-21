"""Ledger: every second is attributed once, survives restarts and rollbacks, round-trips exactly."""
import json

import numpy as np
import pytest

from bes.train import schedule as S
from bes.train.ledger import PHASES, Ledger, atomic_write_json

T0 = 1.7e9


def test_every_second_is_attributed_once():
    rng = np.random.default_rng(0)
    led = Ledger.start(T0)
    now = T0
    booked = dict.fromkeys(PHASES, 0.0)  # independent bookkeeping
    phase = "setup"
    for _ in range(500):
        dt = float(rng.exponential(7.0))
        booked[phase] += dt
        now += dt
        phase = str(rng.choice(PHASES))
        led.switch(phase, now)
        assert led.accounted() == pytest.approx(now - T0, rel=1e-12)
    for name in PHASES:  # durations are differences of epoch times (resolution ~0.24 us)
        assert led.seconds[name] == pytest.approx(booked[name], abs=1e-5)
    assert led.accounted(now + 5.0) == pytest.approx(now + 5.0 - T0, rel=1e-12)


def test_open_interval_counts_for_its_phase_only():
    led = Ledger.start(T0, T0 + 10.0, phase="train")
    assert led.seconds["setup"] == 10.0
    assert led.train_seconds(T0 + 25.0) == 15.0
    assert led.seconds_in("eval", T0 + 25.0) == 0.0
    led.switch("eval", T0 + 25.0)
    assert led.train_seconds(T0 + 99.0) == 15.0
    assert led.seconds_in("eval", T0 + 30.0) == 5.0


def test_clock_going_backwards_books_nothing():
    led = Ledger.start(T0, T0 + 10.0, phase="train")
    led.switch("eval", T0 + 5.0)  # clock jumped back
    assert led.seconds["train"] == 0.0 and led.mark == T0 + 10.0
    led.switch("train", T0 + 12.0)
    assert led.seconds["eval"] == 2.0


def test_snapshot_and_restart_book_lost_time():
    led = Ledger.start(T0, T0 + 100.0, phase="train")
    led.add_step(tokens=1000, steps=10)
    snap = led.snapshot(T0 + 400.0)       # checkpoint
    assert led.mark == T0 + 100.0          # the live ledger is unchanged
    assert snap.train_seconds() == 300.0
    led.add_step(tokens=500, steps=5)      # work after the checkpoint, then a crash at +500
    back = Ledger.from_dict(json.loads(json.dumps(snap.to_dict()))).resumed(T0 + 900.0)
    assert back.seconds["lost"] == 500.0   # 100 s of discarded work + 400 s of downtime
    assert back.train_seconds() == 300.0 and back.steps == 10 and back.tokens == 1000
    assert back.restarts == 1 and back.rollbacks == 0 and back.phase == "setup"
    assert back.accounted(T0 + 950.0) == pytest.approx(950.0)
    assert snap.restarts == 0              # resumed() does not modify its argument


def test_progress_after_restarts_uses_the_kept_training_seconds():
    """Three crashes: each restart recomputes p from the snapshot and the clock."""
    end = T0 + 10_000.0
    led = Ledger.start(T0, T0 + 200.0, phase="train")
    now = T0 + 200.0
    snaps = []
    for crash in range(3):
        now += 1000.0
        snap = led.snapshot(now)
        snaps.append(snap)
        p_ckpt = S.progress_of(snap, now, end)
        now += 700.0                       # work after the checkpoint, lost
        now += 300.0                       # downtime
        led = Ledger.from_dict(snap.to_dict()).resumed(now)
        p_resumed = S.progress_of(led, now, end)
        assert p_resumed == pytest.approx(snap.train_seconds() / (snap.train_seconds() + end - now))
        assert p_resumed >= p_ckpt
        led.switch("train", now)
    assert led.restarts == 3
    assert led.seconds["lost"] == pytest.approx(3 * 1000.0)
    assert led.train_seconds() == pytest.approx(3 * 1000.0)
    assert led.accounted(now) == pytest.approx(now - T0)


def test_rollback_keeps_the_breakdown_and_moves_discarded_training_to_lost():
    led = Ledger.start(T0, T0 + 50.0, phase="train")
    led.add_step(tokens=100, steps=1)
    kept = led.snapshot(T0 + 150.0)        # checkpoint after 100 s of training
    led.switch("eval", T0 + 250.0)         # 100 s more training, then 20 s eval
    led.switch("train", T0 + 270.0)
    led.add_step(tokens=200, steps=2)
    led.restarts = 1
    back = led.rolled_back(kept, T0 + 330.0, data_skipped=7)
    assert back.train_seconds() == 100.0
    assert back.seconds["lost"] == pytest.approx(160.0)   # 100 + 60 s of discarded training
    assert back.seconds["eval"] == 20.0 and back.seconds["setup"] == 50.0
    assert (back.steps, back.tokens, back.rollbacks, back.restarts, back.data_skipped) == (1, 100, 1, 1, 7)
    assert back.accounted() == pytest.approx(330.0)
    assert led.rolled_back(kept, T0 + 330.0, count=False).rollbacks == 0


def test_json_round_trip_is_exact(tmp_path):
    led = Ledger.start(T0 + 0.1, T0 + 1 / 3, phase="train")
    led.switch("ckpt", T0 + np.pi)
    led.add_step(tokens=262144, steps=1)
    path = tmp_path / "sub" / "ledger.json"
    led.save(path)
    back = Ledger.load(path)
    assert back == led
    assert back.seconds["train"] == led.seconds["train"]  # bitwise, not approx
    assert not list(path.parent.glob(".tmp-*"))
    with pytest.raises(ValueError):
        Ledger.from_dict({**led.to_dict(), "format": 99})
    with pytest.raises(ValueError):
        led.switch("", T0)


def test_atomic_write_json_replaces_and_refuses_nan(tmp_path):
    path = tmp_path / "x.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    with pytest.raises(ValueError):
        atomic_write_json(path, {"a": float("nan")})
    assert json.loads(path.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
