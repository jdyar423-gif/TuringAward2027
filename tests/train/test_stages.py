"""The stage plan, the shedding order, the stage ledger and the supervisor's stage loop, with a
fake clock and stand-in stage functions (no JAX)."""
from __future__ import annotations

import json

import pytest

from bes.config import RunCfg
from bes.train import stages as ST
from bes.train import supervisor as SV
from bes.train.schedule import FakeClock

HOUR = 3600.0
RUN = RunCfg()                      # the 24-hour plan
BUDGET = RUN.deadline_hours * HOUR


def specs24():
    return ST.stage_specs(RUN, BUDGET)


# ------------------------------------------------------------------------------- plan
def test_plan_covers_the_budget_and_scales_the_post_window():
    sp = specs24()
    assert set(sp) == set(ST.ORDER) | {"buffer"}
    total = sum(s.planned_s for s in sp.values())
    assert total == pytest.approx(BUDGET)
    post = sum(sp[n].planned_s for n in ST.POST + ("buffer",))
    assert post == pytest.approx(RUN.post_reserve_hours * HOUR)
    # the plan's 19:30-24:00 window: 50 / 10 / 60 / 45 / 30 / 15 / 60 minutes
    assert [round(sp[n].planned_s / 60) for n in ST.POST + ("buffer",)] == [50, 10, 60, 45, 30, 15, 60]
    assert sp["benchmark"].planned_s == 15 * 60 and sp["benchmark"].hard_s == 25 * 60
    assert sp["train"].planned_s == pytest.approx(BUDGET - post - sp["preflight"].planned_s
                                                  - sp["background"].planned_s - 15 * 60)
    small = ST.stage_specs(RunCfg(post_reserve_hours=0.5), 2 * HOUR)
    assert sum(small[n].planned_s for n in ST.POST + ("buffer",)) == pytest.approx(0.5 * HOUR)


def test_on_time_nothing_is_shed():
    sp = specs24()
    t0 = 1e6
    deadline = t0 + BUDGET
    train_end = deadline - RUN.post_reserve_hours * HOUR
    al = ST.allocate(sp, list(ST.ORDER), t0, deadline, train_end)
    assert not al.skipped and not al.shrunk
    assert al.slices["train"] == pytest.approx(train_end - t0)
    for n in ST.POST:
        assert al.slices[n] == pytest.approx(sp[n].planned_s)
    al = ST.allocate(sp, list(ST.POST), train_end, deadline, train_end)
    assert not al.skipped and al.available_s == pytest.approx(RUN.post_reserve_hours * HOUR)


def test_late_runs_shed_in_the_fixed_order():
    """As the post-training window shrinks, stages shrink to their minimum and are skipped in
    exactly SHED_ORDER (buffer first, prefopt last); export is never shed."""
    sp = specs24()
    deadline = 1e6 + BUDGET
    train_end = deadline - RUN.post_reserve_hours * HOUR
    seen_skips: list[str] = []
    prev_total = None
    for late_min in range(0, 271, 3):
        now = train_end + 60.0 * late_min
        al = ST.allocate(sp, list(ST.POST), now, deadline, train_end)
        total = sum(al.slices.values())
        if prev_total is not None:
            assert total <= prev_total + 1e-6            # monotone in the lateness
        prev_total = total
        assert al.slices["export"] == pytest.approx(sp["export"].planned_s)
        assert "export" not in al.skipped
        skipped = [n for n in ST.SHED_ORDER if n in al.skipped]
        assert skipped == [n for n in ST.SHED_ORDER if n in skipped]
        # what is skipped is always a prefix of the shedding order (the buffer is never "skipped")
        assert skipped == [n for n in ST.SHED_ORDER[1:]][:len(skipped)]
        for n in al.skipped:
            if n not in seen_skips:
                seen_skips.append(n)
        # a stage is never shrunk while a stage earlier in the order still has more than its minimum
        for i, n in enumerate(ST.SHED_ORDER):
            if n in al.shrunk and n != "buffer":
                for m in ST.SHED_ORDER[:i]:
                    if m == "buffer":
                        continue
                    assert al.slices.get(m, 0.0) <= sp[m].min_s + 1e-6
        if total <= deadline - now + 1e-6:
            continue
        # only when everything sheddable is gone may the slices overrun (export alone)
        assert set(al.skipped) == set(ST.SHED_ORDER[1:])
    assert seen_skips == list(ST.SHED_ORDER[1:])


def test_shrink_before_skip_and_reasons():
    sp = specs24()
    deadline = 1e6 + BUDGET
    train_end = deadline - RUN.post_reserve_hours * HOUR
    # 70 minutes late: the 60-minute buffer is gone and judge lost 10 of its 30 minutes
    al = ST.allocate(sp, list(ST.POST), train_end + 70 * 60, deadline, train_end)
    assert al.slices["judge"] == pytest.approx(20 * 60)
    assert not al.skipped and al.shrunk["buffer"] == pytest.approx(60 * 60)
    # judge at its minimum (15% of 30 minutes) still fits at 85.5 minutes, not later
    al = ST.allocate(sp, list(ST.POST), train_end + 85.5 * 60, deadline, train_end)
    assert al.slices["judge"] == pytest.approx(sp["judge"].min_s) and not al.skipped
    al = ST.allocate(sp, list(ST.POST), train_end + 86 * 60, deadline, train_end)
    assert al.slices["judge"] == 0 and "late" in al.skipped["judge"]


def test_benchmark_and_train_windows():
    sp = specs24()
    deadline = 1e6 + BUDGET
    train_end = deadline - RUN.post_reserve_hours * HOUR
    # plenty of time: the planned 15 minutes, hard cap 25
    al = ST.allocate(sp, ["benchmark", "train"], train_end - 10 * HOUR, deadline, train_end)
    assert al.slices["benchmark"] == 15 * 60 and al.hard["benchmark"] == 25 * 60
    # 20 minutes before the end of training: at most half of it, at least the minimum (6 min)
    al = ST.allocate(sp, ["benchmark", "train"], train_end - 20 * 60, deadline, train_end)
    assert al.slices["benchmark"] == pytest.approx(10 * 60)
    al = ST.allocate(sp, ["benchmark", "train"], train_end - 7 * 60, deadline, train_end)
    assert al.slices["benchmark"] == pytest.approx(sp["benchmark"].min_s)
    # less than the benchmark's minimum left, or none: neither the benchmark nor training
    for now in (train_end - 5 * 60, train_end + 1):
        al = ST.allocate(sp, ["benchmark", "train", "sample"], now, deadline, train_end)
        assert {"benchmark", "train"} <= set(al.skipped)
        assert al.slices["train"] == 0 and al.slices["benchmark"] == 0
    assert "benchmark was skipped" in ST.allocate(sp, ["benchmark", "train"], train_end - 60,
                                                  deadline, train_end).skipped["train"]
    # once the benchmark is done, training uses whatever is left of its window
    al = ST.allocate(sp, ["train", "sample"], train_end - 60, deadline, train_end)
    assert al.slices["train"] == pytest.approx(60) and "train" not in al.skipped


# ----------------------------------------------------------------------------- ledger
def test_ledger_persists_and_keeps_one_budget(tmp_path):
    clock = FakeClock(100.0)
    path = tmp_path / "stages.json"
    led = ST.StageLedger(path, t0=100.0, deadline=200.0, clock=clock)
    assert led.pending() == list(ST.ORDER)
    led.start("preflight", led.now(), planned_s=5, slice_s=5)
    clock.advance(3)
    led.finish("preflight", led.now(), {"x": float("nan")})
    led.start("background", led.now(), planned_s=1, slice_s=1)
    clock.advance(1)
    led.skip("background", led.now(), "nothing to do")
    led.start("benchmark", led.now(), planned_s=10, slice_s=10)
    clock.advance(4)
    led.interrupt("benchmark", led.now(), "stop")
    again = ST.StageLedger(path, clock=clock)
    assert again.t0 == 100.0 and again.deadline == 200.0
    assert again.pending()[0] == "benchmark"
    assert again.stages["preflight"]["used_s"] == pytest.approx(3.0)
    assert again.stages["preflight"]["result"] == {"x": "nan"}
    assert again.stages["benchmark"]["interrupted"] == 1
    assert again.stages["benchmark"]["used_s"] == pytest.approx(4.0)
    json.loads(path.read_text())                              # strict JSON (no NaN)
    with pytest.raises(ValueError):
        ST.StageLedger(path, t0=101.0, deadline=201.0, clock=clock)
    with pytest.raises(ValueError):
        ST.StageLedger(tmp_path / "new.json", clock=clock)     # a new ledger needs a budget


def test_ledger_time_never_goes_back(tmp_path):
    clock = FakeClock(1000.0)
    led = ST.StageLedger(tmp_path / "s.json", t0=1000.0, deadline=2000.0, clock=clock)
    clock.advance(50)
    assert led.now() == 1050.0
    led.save()
    back = FakeClock(900.0)                                    # a new machine, wrong clock
    led2 = ST.StageLedger(tmp_path / "s.json", clock=back)
    assert led2.now() == 1050.0 and led2.remaining() == 950.0


def test_a_started_stage_found_by_a_new_process_counts_as_interrupted(tmp_path):
    clock = FakeClock(0.0)
    led = ST.StageLedger(tmp_path / "s.json", t0=0.0, deadline=1000.0, clock=clock)
    led.start("preflight", 0.0, planned_s=1, slice_s=1)
    clock.advance(30.0)
    led.heartbeat()                                           # the last sign of life
    clock.advance(5.0)                                        # then the process is killed
    clock.advance(60.0)                                       # ... and is down for a minute
    led2 = ST.StageLedger(tmp_path / "s.json", clock=clock)
    assert led2.pending()[0] == "preflight"
    led2.start("preflight", led2.now(), planned_s=1, slice_s=1)
    r = led2.stages["preflight"]
    assert r["attempts"] == 2 and r["interrupted"] == 1
    assert r["used_s"] == pytest.approx(30.0) and r["down_s"] == pytest.approx(65.0)
    clock.advance(10.0)
    led2.finish("preflight", led2.now())
    assert led2.stages["preflight"]["used_s"] == pytest.approx(40.0)
    assert "down" in led2.format_table()


# ------------------------------------------------------------- stage loop (stand-ins)
class Killed(BaseException):
    """A simulated SIGKILL: nothing is written, the supervisor does not catch it."""


class StandIns:
    """Stand-in stage functions on a fake clock. ``cost[name]`` seconds of work per stage (a
    stage honours its slice: it stops at ``ctx.deadline``); training advances in steps of
    ``step_s`` until ``train_end`` and keeps its step count in a file (its checkpoint)."""

    def __init__(self, clock: FakeClock, tmp, cost=None, step_s=60.0, kill_train_at=None):
        self.clock, self.tmp = clock, tmp
        self.cost = dict(cost or {})
        self.step_s = step_s
        self.kill_train_at = kill_train_at
        self.calls: list[str] = []
        self.ckpt = tmp / "train_ckpt.json"
        self.stop_at_step = None

    def make(self, name):
        def fn(ctx):
            self.calls.append(name)
            if name == "train":
                return self.train(ctx)
            work = self.cost.get(name, 0.5 * ctx.slice_s)
            dt = min(work, max(0.0, ctx.deadline - ctx.now()))
            self.clock.advance(dt)
            return {"worked_s": dt, "partial": dt < work}
        return fn

    def train(self, ctx):
        step = json.loads(self.ckpt.read_text())["step"] if self.ckpt.exists() else 0
        while ctx.now() + self.step_s <= ctx.train_end:
            if self.kill_train_at is not None and step == self.kill_train_at:
                self.kill_train_at = None
                raise Killed("killed in the middle of training")
            if ctx.stop_requested():
                self.ckpt.write_text(json.dumps({"step": step}))
                raise SV.Interrupted(f"stopped at step {step}")
            self.clock.advance(self.step_s)
            step += 1
            if self.stop_at_step is not None and step == self.stop_at_step:
                ctx.sup.request_stop("test signal")
            if step % 5 == 0:
                self.ckpt.write_text(json.dumps({"step": step}))
        self.ckpt.write_text(json.dumps({"step": step}))
        return {"step": step}

    def fns(self):
        return {n: self.make(n) for n in ST.ORDER}


def make_sup(tmp, clock, standins, run=None, budget=None, t0=0.0, **kw):
    run = run or RunCfg(deadline_hours=4.0, post_reserve_hours=1.5, preflight_min=1.0,
                        benchmark_min=10.0, benchmark_cap_min=15.0)
    budget = budget or run.deadline_hours * HOUR
    specs = ST.stage_specs(run, budget)
    sup = SV.Supervisor(tmp / "run", specs, standins.fns(), t0=t0, deadline=t0 + budget,
                        post_reserve_s=run.post_reserve_hours * HOUR, clock=clock,
                        log=lambda m: None, **kw)
    real_run = sup.run
    sup.run = lambda: real_run(heartbeat_s=0)                # no thread with a fake clock
    return sup, specs


def test_fast_run_does_everything_on_time(tmp_path):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 30.0 for n in ST.ORDER})
    sup, specs = make_sup(tmp_path, clock, si)
    res = sup.run()
    assert res["status"] == "complete"
    assert si.calls == list(ST.ORDER)
    assert all(sup.ledger.status(n) == "finished" for n in ST.ORDER)
    assert clock() <= sup.deadline
    # training ran until the end of its window
    assert sup.train_end - sup.ledger.stages["train"]["finished"] < si.step_s
    assert "train" in res["table"] and "export" in res["table"]


@pytest.mark.parametrize("overrun_min,expect_skipped", [(30, ["judge"]), (45, ["select", "judge"]),
                                                       (80, ["select", "judge"])])
def test_slow_run_sheds_in_order_and_still_exports(tmp_path, overrun_min, expect_skipped):
    """Preference optimisation overruns its slice: the stages after it are shed in the fixed
    order (the buffer first, then the judged comparison, ...), the export still happens, and the
    run ends by the deadline (plus the export itself when nothing else is left to shed)."""
    clock = FakeClock(0.0)
    run = RunCfg(deadline_hours=4.0, post_reserve_hours=1.5, preflight_min=1.0, benchmark_min=10.0)
    specs = ST.stage_specs(run, 4 * HOUR)
    si = StandIns(clock, tmp_path, cost={n: specs[n].planned_s for n in ST.ORDER})
    sup, _ = make_sup(tmp_path, clock, si, run=run)
    orig = si.make("prefopt")

    def prefopt(ctx):
        out = orig(ctx)
        clock.advance(overrun_min * 60.0)          # a stage that ignores its slice
        return out

    sup.fns["prefopt"] = prefopt
    res = sup.run()
    assert res["status"] == "complete"
    led = sup.ledger
    skipped = [n for n in ST.ORDER if led.status(n) == "skipped"]
    assert skipped == expect_skipped
    assert all("late" in led.stages[n]["reason"] for n in skipped)
    assert [n for n in ST.SHED_ORDER if n in skipped] == [n for n in ST.SHED_ORDER[1:] if n in skipped]
    assert led.status("export") == "finished" and "export" in si.calls
    end = led.stages["export"]["finished"]
    prefopt_end = led.stages["prefopt"]["finished"]
    if prefopt_end <= sup.deadline - specs["export"].planned_s:
        assert end <= sup.deadline + 1e-6
    else:      # the overrun itself passed the deadline: everything else is shed, export at once
        assert end == pytest.approx(prefopt_end + led.stages["export"]["used_s"])
    assert sup.train_end - led.stages["train"]["finished"] < si.step_s


def test_interrupted_run_resumes_the_same_budget(tmp_path):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 20.0 for n in ST.ORDER})
    si.stop_at_step = 30
    sup, specs = make_sup(tmp_path, clock, si)
    res = sup.run()
    assert res["status"] == "interrupted" and res["stage"] == "train"
    assert sup.ledger.status("train") == "pending" and sup.ledger.stages["train"]["interrupted"] == 1
    assert json.loads(si.ckpt.read_text())["step"] == 30
    deadline = sup.deadline
    clock.advance(600.0)                   # downtime belongs to the budget
    si.stop_at_step = None
    sup2, _ = make_sup(tmp_path, clock, si, t0=0.0)
    assert sup2.deadline == deadline
    res2 = sup2.run()
    assert res2["status"] == "complete"
    assert si.calls.count("train") == 2 and si.calls.count("preflight") == 1
    assert sup2.ledger.stages["train"]["attempts"] == 2
    assert clock() <= deadline


@pytest.mark.parametrize("boundary", list(ST.ORDER))
def test_kill_at_every_stage_boundary_then_resume(tmp_path, boundary):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 20.0 for n in ST.ORDER})
    armed = {"on": True}

    def kill(name):
        if armed["on"] and name == boundary:
            armed["on"] = False
            raise Killed(f"killed before {name}")

    sup, _ = make_sup(tmp_path, clock, si, on_boundary=kill)
    with pytest.raises(Killed):
        sup.run()
    before = list(si.calls)
    assert before == list(ST.ORDER[:ST.ORDER.index(boundary)])
    clock.advance(120.0)
    sup2, _ = make_sup(tmp_path, clock, si)
    res = sup2.run()
    assert res["status"] == "complete"
    assert si.calls == list(ST.ORDER)                  # every stage exactly once
    assert sup2.deadline == sup.deadline


def test_kill_in_the_middle_of_training_then_resume(tmp_path):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 20.0 for n in ST.ORDER}, kill_train_at=23)
    sup, _ = make_sup(tmp_path, clock, si)
    with pytest.raises(Killed):
        sup.run()
    assert sup.ledger.status("train") == "started"
    assert json.loads(si.ckpt.read_text())["step"] == 20      # the last checkpoint
    clock.advance(300.0)
    sup2, _ = make_sup(tmp_path, clock, si)
    res = sup2.run()
    assert res["status"] == "complete"
    rec = sup2.ledger.stages["train"]
    assert rec["status"] == "finished" and rec["attempts"] == 2 and rec["interrupted"] == 1
    assert si.calls.count("train") == 2 and si.calls.count("benchmark") == 1
    assert sup2.train_end - rec["finished"] < si.step_s


def test_failures_skips_and_the_emergency_export(tmp_path):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 20.0 for n in ST.ORDER})
    sup, _ = make_sup(tmp_path, clock, si)

    def critic(ctx):
        raise ValueError("a bug in an optional stage")

    def judge(ctx):
        raise SV.StageSkipped("module bes.eval.proxy.judge is missing")

    def benchmark(ctx):
        raise RuntimeError("no plan compiles")

    sup.fns["critic"], sup.fns["judge"] = critic, judge
    real_bench = sup.fns["benchmark"]
    sup.fns["benchmark"] = benchmark
    with pytest.raises(SV.FatalStageError):
        sup.run()
    assert sup.ledger.status("benchmark") == "failed"
    assert any(e["event"] == "emergency_export" for e in sup.ledger.events)
    assert sup.ledger.status("export") == "pending"            # out of turn: not the final export
    sup2, _ = make_sup(tmp_path, clock, si)
    sup2.fns["critic"], sup2.fns["judge"] = critic, judge
    sup2.fns["benchmark"] = real_bench                           # fixed: retried on resume
    res = sup2.run()
    assert res["status"] == "complete"
    led = sup2.ledger
    assert led.status("benchmark") == "finished" and led.status("critic") == "failed"
    assert led.status("judge") == "skipped" and "missing" in led.stages["judge"]["reason"]
    assert led.status("export") == "finished"
    # a failed optional stage is final: a further resume does not retry it
    sup3, _ = make_sup(tmp_path, clock, si)
    assert sup3.pending() == []


def test_interrupt_near_the_deadline_exports(tmp_path):
    clock = FakeClock(0.0)
    si = StandIns(clock, tmp_path, cost={n: 20.0 for n in ST.ORDER})
    sup, specs = make_sup(tmp_path, clock, si)
    exported = []

    def judge(ctx):
        clock.set(sup.deadline - 0.5 * specs["export"].planned_s)
        sup.request_stop("SIGTERM near the end")
        ctx.check()

    def export(ctx):
        exported.append(ctx.now())
        return {"ok": True}

    sup.fns["judge"], sup.fns["export"] = judge, export
    res = sup.run()
    assert res["status"] == "interrupted" and exported
