"""Trainer loop with the stand-in model and a fake clock: deadline, schedule, restarts, guard."""
import dataclasses
import json

import jax
import numpy as np
import pytest

import train_standin as M
from bes.train import checkpoint as C
from bes.train import schedule as S
from bes.train.train import IteratorData, Trainer

T0 = 1000.0
KEYS = ("tokens", "w")


def trainer(cfg, out, clock, data, **kw):
    return Trainer(cfg, loss_fn=M.loss_fn, init_fn=M.init_fn, role_fn=M.role_fn,
                   consts_fn=M.consts_fn, data=data, out_dir=out, clock=clock,
                   batch_keys=KEYS, **kw)


def leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)


def reference_state(cfg, tmp_path, steps, dt=1.0):
    """Independent run of ``steps`` updates on the same data and clock (no poison)."""
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path / f"ref{steps}", clock, M.Data(clock, dt), max_steps=steps)
    tr.run()
    return tr.state


@pytest.mark.parametrize("dt", [2.0, 7.0])
def test_deadline_stops_the_loop_on_time(tmp_path, dt):
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    data = M.Data(clock, dt)
    tr = trainer(cfg, tmp_path, clock, data)
    out = tr.run()
    end = T0 + 288.0
    assert out["reason"] == "deadline" and out["train_end"] == pytest.approx(end)
    assert end - dt < clock() <= end + 1e-9
    assert tr.step == int(288 // dt) == len(tr.history)
    assert out["p_end"] >= 1 - dt / 288 - 1e-12
    ps = [h[1] for h in tr.history]
    assert ps[0] == 0.0 and np.all(np.diff(ps) > 0)
    led = out["ledger"]
    assert led["seconds"]["train"] == pytest.approx(clock() - T0)
    assert led["steps"] == tr.step and led["tokens"] == tr.step * 256
    assert out["final"] is not None and C.latest(tmp_path / "ckpt").step == tr.step
    assert json.loads((tmp_path / "run.json").read_text())["deadline"] == pytest.approx(T0 + 360.0)


def test_slow_and_fast_runs_follow_the_same_schedule(tmp_path):
    cfg = M.tiny_cfg()
    runs = {}
    for dt in (1.5, 6.0):
        clock = S.FakeClock(T0)
        data = M.Data(clock, dt)
        tr = trainer(cfg, tmp_path / str(dt), clock, data)
        tr.run()
        assert T0 + 288 - dt < clock() <= T0 + 288 + 1e-9
        for step, p, lr, loss, t in tr.history:
            assert lr == S.lr_factor(p, cfg.optim)            # schedule as a function of p only
            assert t == S.seq_len_at(p, cfg.data)
        shapes = [s for _, s in data.calls]
        switches = [i for i in range(1, len(shapes)) if shapes[i] != shapes[i - 1]]
        assert len(switches) == 1 and shapes[0] == (4, 4, 16) and shapes[-1] == (4, 2, 32)
        i = switches[0]
        assert tr.history[i - 1][1] < 0.5 <= tr.history[i][1]
        runs[dt] = T0 + i * dt                                  # dispatch time of the first long step
        stage_events = [e for e in tr.events if e["event"] == "stage"]
        assert [e["seq_len"] for e in stage_events] == [16, 32, 32]  # start, long, chat-heavy tail
    assert abs(runs[1.5] - runs[6.0]) <= 6.0


def test_both_shapes_are_compiled_before_the_first_step(tmp_path):
    cfg = M.tiny_cfg()
    traces = []

    def counting_loss(*args):
        traces.append(1)
        return M.loss_fn(*args)

    clock = S.FakeClock(T0)
    data = M.Data(clock, 4.0)
    tr = Trainer(cfg, loss_fn=counting_loss, init_fn=M.init_fn, role_fn=M.role_fn, data=data,
                 out_dir=tmp_path, clock=clock, batch_keys=KEYS)
    tr.setup()
    assert set(tr.compiled) == {(4, 4, 16), (4, 2, 32)} and len(traces) == 2
    assert data.calls == []
    tr.run()
    assert len(traces) == 2 and {s for _, s in data.calls} == {(4, 4, 16), (4, 2, 32)}


def test_ten_steps_restart_ten_steps_equals_twenty_bitwise(tmp_path):
    cfg = M.tiny_cfg()
    clock_a = S.FakeClock(T0)
    a = trainer(cfg, tmp_path / "split", clock_a, M.Data(clock_a, 1.0), max_steps=10)
    out_a = a.run()
    assert out_a["reason"] == "max_steps" and a.step == 10
    clock_b = S.FakeClock(clock_a())  # restart without lost time
    b = trainer(cfg, tmp_path / "split", clock_b, M.Data(clock_b, 1.0), max_steps=20)
    out_b = b.run()
    clock_c = S.FakeClock(T0)
    c = trainer(cfg, tmp_path / "whole", clock_c, M.Data(clock_c, 1.0), max_steps=20)
    c.run()
    assert [e["event"] for e in b.events][:2] == ["restore", "setup"]
    assert b.step == c.step == 20 and b.cursor == c.cursor == 20
    leaves_equal(b.state, c.state)
    leaves_equal(b.rng, c.rng)
    assert b.history == c.history[10:]          # same p, lr and loss at steps 11..20, bit for bit
    assert a.history == c.history[:10]
    assert b.guard == c.guard
    assert out_b["ledger"]["seconds"]["lost"] == 0.0 and out_b["ledger"]["restarts"] == 1


def test_interrupted_run_resumes_and_still_ends_on_time(tmp_path):
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    first = trainer(cfg, tmp_path, clock, M.Data(clock, 2.0, crash_at=52))
    with pytest.raises(M.Crash):
        first.run()
    kept = [h for h in first.history if h[0] <= 45]   # last checkpoint before the crash: step 45
    assert C.latest(tmp_path / "ckpt").step == 45
    crash_time = clock()
    clock.advance(100.0)                                # downtime
    second = trainer(cfg, tmp_path, clock, M.Data(clock, 2.0))
    out = second.run()
    assert out["reason"] == "deadline" and T0 + 288 - 2.0 < clock() <= T0 + 288 + 1e-9
    assert second.history[0][0] == 46
    led = out["ledger"]
    assert led["restarts"] == 1
    assert led["seconds"]["lost"] == pytest.approx(crash_time + 100.0 - (T0 + 90.0))
    assert sum(led["seconds"].values()) == pytest.approx(clock() - T0)
    traj = kept + second.history
    ps = [h[1] for h in traj]
    assert np.all(np.diff(ps) >= 0) and ps[-1] >= 1 - 2 * 2.0 / led["seconds"]["train"]
    lens = [h[4] for h in traj]
    assert sum(lens[i] != lens[i - 1] for i in range(1, len(lens))) == 1
    assert len(traj) < 144                             # fewer steps, same schedule


def test_spike_rolls_back_skips_data_and_lowers_the_learning_rate(tmp_path):
    cfg = M.tiny_cfg(ckpt_every_min=0.25, guard_skip_steps=10, keep_ckpts=50)
    clock = S.FakeClock(T0)
    data = M.Data(clock, 1.0, poison={25, 26, 27})
    tr = trainer(cfg, tmp_path / "run", clock, data)
    out = tr.run()
    rb = [e for e in tr.events if e["event"] == "rollback"]
    assert len(rb) == 1
    assert rb[0]["first_bad"] == 26 and rb[0]["ckpt_step"] == 15 and rb[0]["cursor"] == 39
    assert rb[0]["reason"] == "spike" and rb[0]["lr_mult"] == pytest.approx(0.7)
    assert out["reason"] == "deadline" and out["guard"]["rollbacks"] == 1
    assert out["ledger"]["rollbacks"] == 1 and out["ledger"]["data_skipped"] == 39 - 15
    used = [i for i, _ in data.calls]
    assert used[:29] == list(range(29)) and used[29] == 39  # poison never seen again
    # after the rollback the learning rate carries the 0.7 multiplier
    after = tr.history[tr.history.index(next(h for h in tr.history if h[0] == 28)) + 1:]
    assert after[0][0] == 16
    for step, p, lr, loss, t in after:
        assert lr == pytest.approx(0.7 * S.lr_factor(p, cfg.optim), rel=1e-12)
    # the state re-saved at the rollback is exactly the state after update 15
    resaved = [c for c in C.list_checkpoints(tmp_path / "run" / "ckpt")
               if json.loads((c.path / C.MANIFEST).read_text())["meta"]["reason"] == "rollback"]
    assert len(resaved) == 1 and resaved[0].step == 15
    ref = reference_state(cfg, tmp_path, 15)
    back, _ = C.load(resaved[0], {"state": ref, "rng": tr.rng})
    leaves_equal(back["state"], ref)
    # p is computed from the restored ledger: monotone along the kept trajectory
    kept = [h for h in tr.history if h[0] <= 15] + after
    assert np.all(np.diff([h[1] for h in kept]) >= 0)


def test_nan_everywhere_exhausts_the_guard_and_keeps_a_trusted_state(tmp_path):
    cfg = M.tiny_cfg(ckpt_every_min=0.25, guard_skip_steps=5)
    clock = S.FakeClock(T0)
    data = M.Data(clock, 1.0, nan=set(range(25, 100_000)))
    tr = trainer(cfg, tmp_path / "run", clock, data)
    out = tr.run()
    kinds = [e["event"] for e in tr.events if e["event"] in ("rollback", "halt")]
    assert kinds == ["rollback"] * 3 + ["halt"]
    assert out["reason"] == "halt" and out["final"] is None and tr.guard.rollbacks == 3
    halt = [e for e in tr.events if e["event"] == "halt"][0]
    rbs = [e for e in tr.events if e["event"] == "rollback"]
    # the state restored at the first rollback (step 15) stays trusted: every retry starts from it
    # with the data moved further on, and the halt keeps it as the final state
    assert [e["ckpt_step"] for e in rbs] == [15, 15, 15] and halt["ckpt_step"] == 15
    assert [e["first_bad"] for e in rbs] == [26, 16, 16] and halt["first_bad"] == 16
    assert [e["cursor"] for e in rbs] == [34, 43, 52]
    assert halt["reason"] == "nonfinite" and tr.step == 15
    assert C.latest(tmp_path / "run" / "ckpt").step == 15  # nothing newer survives
    assert out["ledger"]["rollbacks"] == 3 and out["ledger"]["steps"] == 15
    leaves_equal(tr.state, reference_state(cfg, tmp_path, 15))


def test_eval_time_is_booked_and_moves_progress(tmp_path):
    cfg = M.tiny_cfg(eval_every_min=1.0)
    clock = S.FakeClock(T0)
    calls = []

    def eval_fn(state, step):
        calls.append(step)
        clock.advance(5.0)
        return {"val": float(step)}

    tr = trainer(cfg, tmp_path, clock, M.Data(clock, 2.0), eval_fn=eval_fn)
    out = tr.run()
    assert len(calls) >= 4
    led = out["ledger"]
    assert led["seconds"]["eval"] == pytest.approx(5.0 * len(calls))
    assert sum(led["seconds"].values()) == pytest.approx(clock() - T0)
    assert T0 + 288 - 2.0 < clock() <= T0 + 288 + 1e-9
    assert [e["result"]["val"] for e in tr.events if e["event"] == "eval"] == [float(s) for s in calls]


def test_failed_checkpoints_and_evaluations_do_not_stop_training(tmp_path, monkeypatch):
    cfg = M.tiny_cfg(eval_every_min=1.0)
    clock = S.FakeClock(T0)
    flags = {"fail": False}
    real_rename = C._rename

    def flaky_rename(src, dst):
        if flags["fail"]:
            raise OSError("disk full")
        return real_rename(src, dst)

    monkeypatch.setattr(C, "_rename", flaky_rename)

    def data(index, stage):
        flags["fail"] = 20 <= index < 60       # the interval saves at steps 30, 45, 60 fail
        return M.Data(clock, 2.0)(index, stage)

    def broken_eval(state, step):
        raise RuntimeError("eval broke")

    tr = trainer(cfg, tmp_path, clock, data, eval_fn=broken_eval)
    out = tr.run()
    assert out["reason"] == "deadline" and T0 + 288 - 2.0 < clock() <= T0 + 288 + 1e-9
    errors = [e for e in tr.events if e["event"] == "ckpt_error"]
    assert [e["step"] for e in errors] == [30, 45, 60]
    assert all("disk full" in e["reason"] for e in errors)
    evals = [e for e in tr.events if e["event"] == "eval"]
    assert len(evals) >= 4 and all("eval broke" in e["result"]["error"] for e in evals)
    saved = [e["step"] for e in tr.events if e["event"] == "ckpt"]
    assert saved[:3] == [0, 15, 75] and out["final"] is not None
    assert C.latest(tmp_path / "ckpt").step == tr.step
    assert not [p for p in (tmp_path / "ckpt").iterdir() if p.name.startswith(".tmp-")]


def test_async_checkpoints_give_the_same_run(tmp_path):
    cfg = M.tiny_cfg(ckpt_every_min=0.1)
    states = []
    for mode in (False, True):
        clock = S.FakeClock(T0)
        tr = trainer(cfg, tmp_path / str(mode), clock, M.Data(clock, 1.0), max_steps=40,
                     async_ckpt=mode)
        tr.run()
        tmpl = {"state": tr.state, "rng": tr.rng}
        back, man, info = C.restore(tmp_path / str(mode) / "ckpt", tmpl, config_hash=tr.config_hash)
        assert info.step == 40
        leaves_equal(back["state"], tr.state)
        states.append(tr.state)
    leaves_equal(states[0], states[1])


def test_resume_refuses_another_configuration(tmp_path):
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    trainer(cfg, tmp_path, clock, M.Data(clock, 1.0), max_steps=3).run()
    other = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, guard_spike=1.5))
    with pytest.raises(C.ConfigMismatchError):
        trainer(other, tmp_path, clock, M.Data(clock, 1.0), max_steps=5).setup()
    forced = trainer(other, tmp_path, clock, M.Data(clock, 1.0), max_steps=5, force=True)
    forced.run()
    assert forced.history[0][0] == 4 and forced.step == 5


def test_current_consts_replace_the_checkpointed_ones(tmp_path):
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    trainer(cfg, tmp_path, clock, M.Data(clock, 1.0), max_steps=3).run()
    traces = []

    def counting_loss(*args):
        traces.append(1)
        return M.loss_fn(*args)

    tr = Trainer(cfg, loss_fn=counting_loss, init_fn=M.init_fn, role_fn=M.role_fn,
                 consts_fn=lambda m: {"scale": jax.numpy.full((), 2.0, jax.numpy.float32)},
                 data=M.Data(clock, 1.0), out_dir=tmp_path, clock=clock, batch_keys=KEYS,
                 max_steps=4)
    tr.setup()
    assert tr.step == 3 and float(tr.state.consts["scale"]) == 2.0
    assert [e["note"] for e in tr.events if e["event"] == "consts"] == [
        "checkpointed consts replaced by the current ones"]
    params3 = jax.tree.map(np.asarray, tr.state.params)
    tr.set_consts({"scale": np.float32(0.5)})
    with pytest.raises(ValueError):
        tr.set_consts({"scale": np.zeros(2, np.float32)})
    tr.run()
    assert len(traces) == 2  # compiled once per shape at set-up, never again
    step4 = tr.history[0]
    batch = M.Data()(3, S.stage_at(step4[1], cfg))
    per = [float(M.loss_fn(params3, {"scale": np.float32(0.5)},
                           {k: v[i] for k, v in batch.items()}, None)[0]) for i in range(4)]
    assert step4[0] == 4 and step4[3] == pytest.approx(
        np.average(per, weights=[batch["w"][i].sum() for i in range(4)]), rel=1e-5)


def test_stop_request_and_restart_after_the_window(tmp_path):
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    box = {}

    def data(index, stage):
        if index == 7:
            box["tr"].request_stop()
        return M.Data(clock, 1.0)(index, stage)

    box["tr"] = tr = trainer(cfg, tmp_path, clock, data)
    out = tr.run()
    assert out["reason"] == "requested" and tr.step == 8 and C.latest(tmp_path / "ckpt").step == 8
    clock.set(T0 + 500.0)                       # restarted after the training window closed
    late = trainer(cfg, tmp_path, clock, M.Data(clock, 1.0))
    out = late.run()
    assert out["reason"] == "deadline" and late.step == 8 and out["final"] is None
    assert out["p_end"] == 1.0


def test_iterator_data_reopens_only_on_discontinuities():
    opened = []

    def open_fn(start, stage):
        opened.append((start, stage.shape))
        i = start
        while True:
            yield {"i": i}
            i += 1

    data = IteratorData(open_fn)
    cfg = M.tiny_cfg()
    short, long = S.stage_at(0.1, cfg), S.stage_at(0.6, cfg)
    assert [data(i, short)["i"] for i in range(3)] == [0, 1, 2]
    assert data(9, short)["i"] == 9 and data(10, short)["i"] == 10
    assert data(11, long)["i"] == 11
    assert opened == [(0, (4, 4, 16)), (9, (4, 4, 16)), (11, (4, 2, 32))]


def test_a_run_goes_on_with_its_micro_batches_cut_in_pieces(tmp_path):
    """The live run of 24 Nov got ``micro_split`` at a sixth of its way: the same configuration
    hash, the same checkpoints and batches; ten steps whole and ten steps in pieces end where
    twenty whole steps end, to float rounding."""
    cfg = M.tiny_cfg()                                   # shapes (4, 4, 16) and (4, 2, 32)
    clock_a = S.FakeClock(T0)
    a = trainer(cfg, tmp_path / "split", clock_a, M.Data(clock_a, 1.0), max_steps=10)
    a.run()
    clock_b = S.FakeClock(clock_a())
    b = trainer(cfg, tmp_path / "split", clock_b, M.Data(clock_b, 1.0), max_steps=20, micro_split=2)
    b.run()
    clock_c = S.FakeClock(T0)
    c = trainer(cfg, tmp_path / "whole", clock_c, M.Data(clock_c, 1.0), max_steps=20)
    c.run()
    assert [e["event"] for e in b.events][:2] == ["restore", "setup"] and b.step == c.step == 20
    assert next(e for e in b.events if e["event"] == "setup")["micro_split"] == 2
    for x, y in zip(jax.tree.leaves(b.state.params), jax.tree.leaves(c.state.params)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=5e-4, atol=5e-6)
    assert [h[0] for h in b.history] == [h[0] for h in c.history[10:]]
    np.testing.assert_allclose([h[3] for h in b.history], [h[3] for h in c.history[10:]], rtol=1e-4)
    with pytest.raises(ValueError, match="micro_split"):
        trainer(cfg, tmp_path / "bad", S.FakeClock(T0), M.Data(S.FakeClock(T0), 1.0), micro_split=4)
