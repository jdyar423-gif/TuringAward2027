"""The stages of the 24-hour run: planned durations, the shedding order and the stage ledger.

One budget of wall time covers the whole run (``docs/PLAN.md`` section 6). It starts when the run
is created (``t0``) and ends at ``deadline = t0 + budget``; both are persisted in the stage ledger
(``<out_dir>/stages.json``), so a restarted process continues the same budget. The downtime of a
crashed process is part of the budget (wall time), and the ledger keeps the latest time it has
seen (``last_seen``): a clock that jumps backwards after a restart never gives time back.

Stages (execution order, :data:`ORDER`)
---------------------------------------

============  ===================================================  ========================
stage         what                                                 planned (24-hour run)
============  ===================================================  ========================
preflight     devices, host, manifest                              ``run.preflight_min``
background    start the supervised host work (it runs all along)   1 minute (instant)
benchmark     execution plans, canary, host rate, horizon          ``run.benchmark_min``
                                                                   (hard cap ``run.benchmark_cap_min``)
train         the wall-clock schedule until ``deadline - post_reserve``   the rest
sample        particle samples on chat / constraint / math prompts   50 of the 270 post minutes
critic        critic fit on the labelled samples                    10
prefopt       one epoch of DPO with health checks                   60
select        decode settings on the dev half (exact scores)        45
judge         judged comparison on the held-out half                30
export        final parameters, configuration, reports              15
(buffer)      slack that nothing runs in                            60
============  ===================================================  ========================

The post-training minutes are those of the plan's 19:30-24:00 window (270 minutes); they are
scaled to ``run.post_reserve_hours`` (:data:`POST_MINUTES`). ``run.post_minutes`` replaces
entries (:func:`post_minutes`); a stage planned with 0 minutes is skipped.

Lateness
--------
Training needs no shedding: it ends at ``train_end = deadline - post_reserve`` by construction
(``bes.train.schedule``), so a late start only shortens it. Everything after it shares the time
that is left. When the planned slices of the remaining post-training stages (plus the buffer) do
not fit, :func:`allocate` sheds in the fixed order :data:`SHED_ORDER`: the first entry is shrunk
to its minimum (``MIN_FRACTION`` of its planned slice), then skipped, before the next entry is
touched::

    buffer -> judge -> critic -> sample -> select -> prefopt

so the judged comparison goes first and preference optimisation last. ``export`` is never shed
(the final export always happens, overrunning the deadline if it must); ``preflight``,
``background`` and ``train`` are never shed either. The benchmark shrinks (down to its minimum)
when the training window is short: it takes at most half of the time left before ``train_end``.
A stage whose window is gone (``train`` after ``train_end``; the benchmark, and with it
training, when less than the benchmark's minimum is left before ``train_end``) is skipped with
that reason.

The allocation is recomputed whenever a stage starts, from the time left at that moment, so the
plan adapts to what actually happened.

Ledger
------
:class:`StageLedger` records per stage ``pending | started | finished | skipped | failed`` with
timestamps, planned / allotted / used seconds, attempts and a small result; it is rewritten
atomically after every change. A restarted run continues at the first stage that is neither
finished nor skipped (:meth:`StageLedger.pending`). Pure Python, no JAX.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bes.config import RunCfg
from bes.train.ledger import atomic_write_json

HOUR = 3600.0
FORMAT = 1

#: Execution order of the stages.
ORDER: tuple[str, ...] = ("preflight", "background", "benchmark", "train", "sample", "critic",
                          "prefopt", "select", "judge", "export")
PRE: tuple[str, ...] = ("preflight", "background", "benchmark")
POST: tuple[str, ...] = ("sample", "critic", "prefopt", "select", "judge", "export")
#: Stages that use the accelerator (they run in the supervisor process, one after the other).
DEVICE: frozenset[str] = frozenset({"benchmark", "train", "sample", "critic", "prefopt", "select",
                                    "judge"})
#: Minutes of the plan's post-training window (19:30-24:00, 270 minutes); scaled to
#: ``RunCfg.post_reserve_hours``. ``buffer`` is slack: no stage runs in it.
POST_MINUTES: dict[str, float] = {"sample": 50.0, "critic": 10.0, "prefopt": 60.0, "select": 45.0,
                                  "judge": 30.0, "export": 15.0, "buffer": 60.0}
#: Shedding order when the run is late: the first entry shrinks to its minimum, then is skipped,
#: before the next one is touched.
SHED_ORDER: tuple[str, ...] = ("buffer", "judge", "critic", "sample", "select", "prefopt")
#: Shortest useful slice as a fraction of the planned slice.
MIN_FRACTION: dict[str, float] = {"benchmark": 0.4, "sample": 0.25, "critic": 0.5,
                                  "prefopt": 0.5, "select": 0.3, "judge": 0.15, "buffer": 0.0}
#: Stages that are never skipped for lack of time.
NEVER_SHED: frozenset[str] = frozenset({"preflight", "background", "export"})
#: Post stages whose work grows with their slice (more prompts, questions, judged pairs). When
#: the run is ahead of the plan (an earlier stage finished early), the time it left is shared
#: by the pending ones in proportion to their planned slices; export and the buffer keep theirs.
ELASTIC: tuple[str, ...] = ("sample", "select", "judge")

STATUSES = ("pending", "started", "finished", "skipped", "failed")
DONE = ("finished", "skipped")


@dataclass(frozen=True)
class StageSpec:
    """One stage of the plan: planned seconds, the minimum useful seconds, the hard cap (the
    benchmark's), its place in the shedding order (``None``: never shed) and whether it uses the
    accelerator."""

    name: str
    planned_s: float
    min_s: float
    hard_s: float
    shed: int | None
    device: bool


def post_minutes(run: RunCfg) -> dict[str, float]:
    """:data:`POST_MINUTES` with the entries of ``run.post_minutes`` (``"stage=minutes ..."``)
    replaced. ``export`` keeps a positive share (it is never skipped)."""
    out = dict(POST_MINUTES)
    for item in str(getattr(run, "post_minutes", "") or "").replace(",", " ").split():
        name, sep, value = item.partition("=")
        if not sep or name not in out:
            raise ValueError(f"run.post_minutes: {item!r} is not <stage>=<minutes> with a stage of "
                             f"{sorted(out)}")
        minutes = float(value)
        if not math.isfinite(minutes) or minutes < 0 or (name == "export" and minutes <= 0):
            raise ValueError(f"run.post_minutes: {item!r}: minutes must be >= 0 (> 0 for export)")
        out[name] = minutes
    return out


def stage_specs(run: RunCfg, budget_s: float) -> dict[str, StageSpec]:
    """The plan of every stage (and of the buffer) for a run of ``budget_s`` seconds."""
    post_s = run.post_reserve_hours * HOUR
    pre = {"preflight": run.preflight_min * 60.0,
           "background": min(60.0, 0.01 * budget_s),
           "benchmark": run.benchmark_min * 60.0}
    specs: dict[str, StageSpec] = {}
    for name in PRE:
        planned = pre[name]
        hard = max(planned, run.benchmark_cap_min * 60.0) if name == "benchmark" else planned
        specs[name] = StageSpec(name, planned, planned * MIN_FRACTION.get(name, 1.0), hard,
                                None, name in DEVICE)
    train = max(0.0, budget_s - post_s - sum(pre.values()))
    specs["train"] = StageSpec("train", train, 0.0, train, None, True)
    minutes = post_minutes(run)
    total_post = sum(minutes.values())
    for name in POST + ("buffer",):
        planned = post_s * minutes[name] / total_post
        shed = SHED_ORDER.index(name) if name in SHED_ORDER else None
        specs[name] = StageSpec(name, planned, planned * MIN_FRACTION.get(name, 1.0), planned,
                                shed, name in DEVICE)
    return specs


@dataclass
class Allocation:
    """Seconds allotted to every stage still to run (0 = skipped), the hard limits, the
    reasons of the skips and the seconds removed from shrunk stages."""

    slices: dict[str, float]
    hard: dict[str, float]
    skipped: dict[str, str] = field(default_factory=dict)
    shrunk: dict[str, float] = field(default_factory=dict)
    available_s: float = 0.0
    needed_s: float = 0.0


def allocate(specs: Mapping[str, StageSpec], pending: Sequence[str], now: float,
             deadline: float, train_end: float) -> Allocation:
    """Slices of the ``pending`` stages (execution order) at wall time ``now`` (module docstring).

    Pre-training stages keep their plan (the benchmark at most half of the time left before
    ``train_end``, at least its minimum); ``train`` gets ``train_end - now``; the post-training
    stages share ``deadline - max(now, train_end)`` (``train_end`` while training is pending)
    and are shed in :data:`SHED_ORDER` when they do not fit."""
    slices: dict[str, float] = {}
    hard: dict[str, float] = {}
    alloc = Allocation(slices, hard)
    before_train = max(0.0, train_end - now)
    for name in pending:
        if name not in PRE and name != "train":
            continue
        spec = specs[name]
        if name == "train":
            slices[name] = before_train
            if before_train <= 0:
                alloc.skipped[name] = "the training window closed at train_end"
            elif "benchmark" in alloc.skipped:
                slices[name] = 0.0
                alloc.skipped[name] = "no execution plan: the benchmark was skipped"
        elif name == "benchmark":
            if before_train < spec.min_s:
                slices[name] = 0.0
                alloc.skipped[name] = (f"{before_train:.0f} s of training window left, less than the "
                                       f"benchmark's minimum {spec.min_s:.0f} s: no training")
            else:
                s = min(spec.planned_s, max(spec.min_s, 0.5 * before_train))
                slices[name] = s
                hard[name] = max(s, min(spec.hard_s, 0.75 * before_train))
                if s < spec.planned_s:
                    alloc.shrunk[name] = spec.planned_s - s
        else:
            slices[name] = spec.planned_s
        hard.setdefault(name, slices[name])
    post = [n for n in pending if n in POST]
    if not post:
        return alloc
    start = max(now, train_end) if "train" in pending else now
    available = max(0.0, deadline - start)
    for n in post:
        if specs[n].planned_s <= 0 and n not in NEVER_SHED:
            alloc.skipped[n] = "planned with 0 minutes (run.post_minutes)"
    sl = {n: specs[n].planned_s for n in post}
    sl["buffer"] = specs["buffer"].planned_s
    need = sum(sl.values())
    alloc.available_s, alloc.needed_s = available, need
    for name in SHED_ORDER:
        if need <= available:
            break
        if name not in sl:
            continue
        cut = min(sl[name] - specs[name].min_s, need - available)
        if cut > 0:
            sl[name] -= cut
            need -= cut
            alloc.shrunk[name] = alloc.shrunk.get(name, 0.0) + cut
        if need > available and sl[name] > 0 and name != "buffer":
            alloc.skipped[name] = (f"late: {available:.0f} s left, the remaining stages need at least "
                                   f"{need:.0f} s; shed in the order {', '.join(SHED_ORDER)}")
            need -= sl[name]
            alloc.shrunk[name] = alloc.shrunk.get(name, 0.0) + sl[name]
            sl[name] = 0.0
    ahead = available - alloc.needed_s          # > 0 only when nothing had to be shed
    grow = [n for n in post if n in ELASTIC and sl[n] > 0 and n not in alloc.skipped]
    if ahead > 0 and grow:
        base = sum(sl[n] for n in grow)
        sl.update({n: sl[n] + ahead * sl[n] / base for n in grow})
    for name in post:
        slices[name] = sl[name]
        hard[name] = sl[name]
    return alloc


# ------------------------------------------------------------------------------- ledger
def _locked(fn):
    """Run a :class:`StageLedger` method under the ledger's lock (the heartbeat thread saves)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


def _record(name: str) -> dict[str, Any]:
    return {"name": name, "status": "pending", "planned_s": None, "slice_s": None,
            "hard_s": None, "started": None, "finished": None, "used_s": 0.0, "down_s": 0.0,
            "attempts": 0, "interrupted": 0, "reason": "", "result": {}}


class StageLedger:
    """Persistent stage records and the run budget (module docstring).

    path: ``stages.json``. A new ledger needs ``t0`` and ``deadline``; an existing one keeps its
    own (passing different values raises: a run has one budget). ``clock``: wall clock.

    :meth:`heartbeat` (called every few seconds by the supervisor) keeps ``last_seen`` fresh on
    disk, so after a kill the next process knows when the killed attempt stopped: that time
    goes to the stage's ``used_s``, the gap until the restart to its ``down_s``. Thread-safe.
    """

    def __init__(self, path: str | Path, *, t0: float | None = None, deadline: float | None = None,
                 order: Sequence[str] = ORDER, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self.order = tuple(order)
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("format") != FORMAT:
                raise ValueError(f"{self.path}: unsupported stage ledger format {data.get('format')}")
            self.t0 = float(data["t0"])
            self.deadline = float(data["deadline"])
            self.last_seen = float(data.get("last_seen", self.t0))
            self.stages = {k: dict(v) for k, v in data["stages"].items()}
            self.events = list(data.get("events", []))
            if t0 is not None and abs(float(t0) - self.t0) > 1e-6:
                raise ValueError(f"{self.path}: the run started at {self.t0}, not {t0}; a run keeps "
                                 "its budget across restarts")
            if deadline is not None and abs(float(deadline) - self.deadline) > 1e-6:
                raise ValueError(f"{self.path}: the run's deadline is {self.deadline}, not {deadline}")
        else:
            if t0 is None or deadline is None:
                raise ValueError(f"{self.path} does not exist: a new ledger needs t0 and deadline")
            self.t0, self.deadline = float(t0), float(deadline)
            self.last_seen = self.t0
            self.stages = {}
            self.events = []
        for name in self.order:
            self.stages.setdefault(name, _record(name))
            self.stages[name].setdefault("down_s", 0.0)
        self.seen_at_load = self.last_seen
        self._lock = threading.RLock()
        self.save()

    # ------------------------------------------------------------------- time
    def now(self) -> float:
        """Wall time, never earlier than any time this ledger has seen (restarts included)."""
        t = max(float(self.clock()), self.last_seen)
        self.last_seen = t
        return t

    def heartbeat(self) -> None:
        """Record that the run is alive now (persists ``last_seen``)."""
        with self._lock:
            self.now()
            self.save()

    def remaining(self, now: float | None = None) -> float:
        return self.deadline - (self.now() if now is None else now)

    def elapsed(self, now: float | None = None) -> float:
        return (self.now() if now is None else now) - self.t0

    # ---------------------------------------------------------------- queries
    def status(self, name: str) -> str:
        return self.stages[name]["status"]

    def pending(self) -> list[str]:
        """Stages that are neither finished nor skipped, in execution order."""
        return [n for n in self.order if self.stages[n]["status"] not in DONE]

    def done(self, name: str) -> bool:
        return self.stages[name]["status"] in DONE

    # --------------------------------------------------------------- mutation
    def save(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:                           # during construction
            return self._write()
        with lock:
            self._write()

    def _write(self) -> None:
        atomic_write_json(self.path, {"format": FORMAT, "t0": self.t0, "deadline": self.deadline,
                                      "last_seen": self.last_seen, "stages": self.stages,
                                      "events": self.events[-200:]})

    @_locked
    def event(self, kind: str, now: float | None = None, **fields: Any) -> None:
        t = self.now() if now is None else now
        self.events.append({"event": kind, "time": round(t - self.t0, 3), **_plain(fields)})
        self.save()

    @_locked
    def start(self, name: str, now: float, *, planned_s: float, slice_s: float,
              hard_s: float | None = None) -> None:
        r = self.stages[name]
        if r["status"] == "started" and r["started"] is not None:
            # the previous attempt never ended (a kill): it ran until the last heartbeat that
            # reached the disk, and the run was down from then until this process started it
            r["interrupted"] += 1
            stop = max(r["started"], min(self.seen_at_load, now))
            r["used_s"] = round(r["used_s"] + stop - r["started"], 3)
            r["down_s"] = round(r.get("down_s", 0.0) + max(0.0, now - stop), 3)
        r.update(status="started", planned_s=round(planned_s, 3), slice_s=round(slice_s, 3),
                 hard_s=round(slice_s if hard_s is None else hard_s, 3), reason="")
        r["attempts"] += 1
        r["started"] = now
        r["finished"] = None
        self.save()

    @_locked
    def _close(self, name: str, now: float, status: str, reason: str,
               result: Mapping[str, Any] | None) -> None:
        r = self.stages[name]
        if r["started"] is not None and r["status"] == "started":
            r["used_s"] = round(r["used_s"] + max(0.0, now - r["started"]), 3)
        r.update(status=status, finished=now, reason=str(reason))
        if result is not None:
            r["result"] = _plain(dict(result))
        self.save()

    def finish(self, name: str, now: float, result: Mapping[str, Any] | None = None) -> None:
        self._close(name, now, "finished", "", result)

    def skip(self, name: str, now: float, reason: str,
             result: Mapping[str, Any] | None = None) -> None:
        self._close(name, now, "skipped", reason, result)

    def fail(self, name: str, now: float, reason: str) -> None:
        self._close(name, now, "failed", reason, None)

    @_locked
    def interrupt(self, name: str, now: float, reason: str) -> None:
        """The stage stopped before it was complete (stop request): it stays unfinished."""
        r = self.stages[name]
        if r["started"] is not None:
            r["used_s"] = round(r["used_s"] + max(0.0, now - r["started"]), 3)
        r["started"] = None
        r["interrupted"] += 1
        r["status"] = "pending"
        r["reason"] = f"interrupted: {reason}"
        self.save()

    @_locked
    def reset(self, name: str) -> None:
        """Make a stage pending again (e.g. the export after an earlier stage is re-run)."""
        r = self.stages[name]
        r.update(status="pending", started=None, finished=None, reason="")
        self.save()

    # ----------------------------------------------------------------- report
    def table(self, specs: Mapping[str, StageSpec] | None = None) -> list[dict[str, Any]]:
        """One row per stage: planned / allotted / used seconds, status, attempts, reason."""
        rows = []
        for name in self.order:
            r = self.stages[name]
            planned = r["planned_s"]
            if planned is None and specs is not None and name in specs:
                planned = round(specs[name].planned_s, 1)
            rows.append({"stage": name, "status": r["status"], "planned_s": planned,
                         "slice_s": r["slice_s"], "used_s": r["used_s"],
                         "down_s": r.get("down_s", 0.0), "attempts": r["attempts"],
                         "interrupted": r["interrupted"], "reason": r["reason"]})
        return rows

    def format_table(self, specs: Mapping[str, StageSpec] | None = None) -> str:
        lines = [f"{'stage':10s} {'status':9s} {'planned':>9s} {'allotted':>9s} {'used':>9s} "
                 f"{'down':>7s} {'tries':>5s}  reason"]
        for row in self.table(specs):
            def f(x, w=9):
                return "-".rjust(w) if x is None else f"{float(x):{w}.1f}"
            lines.append(f"{row['stage']:10s} {row['status']:9s} {f(row['planned_s'])} "
                         f"{f(row['slice_s'])} {f(row['used_s'])} {f(row['down_s'], 7)} "
                         f"{row['attempts']:5d}  {row['reason'][:100]}")
        return "\n".join(lines)


def _plain(x: Any) -> Any:
    """JSON-safe copy (non-finite floats become strings, unknown objects their repr)."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return _plain(x.item())
        if isinstance(x, np.ndarray):
            return _plain(x.tolist())
    except ImportError:  # pragma: no cover
        pass
    return repr(x)


__all__ = ["ORDER", "PRE", "POST", "DEVICE", "POST_MINUTES", "SHED_ORDER", "MIN_FRACTION",
           "NEVER_SHED", "StageSpec", "stage_specs", "Allocation", "allocate", "StageLedger"]
