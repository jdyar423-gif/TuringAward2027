"""Start-up benchmark and canary of the 24-hour run (``docs/PLAN.md`` section 6,
``docs/TPU_NOTES.md`` "First 15 minutes on the real chip").

Execution plans
---------------
A plan (:class:`Plan`) is a set of configuration overrides applied before the trainer and the
loader are built: ``model.attn_impl`` (the flash kernel ``"splash"`` or dense ``"xla"``),
``model.remat`` together with ``optim.micro_seqs`` (rematerialisation on with the configured
micro-batch, off with half of it so that it fits), and ``model.table_grad`` (``"onehot"`` or
``"take"``). ``bes.train.schedule.micro_shape`` turns ``micro_seqs`` into the compiled shapes:
with 262,144 tokens per step and ``seq_len = 1024``, ``micro_seqs = 32`` gives ``(8, 32, 1024)``
and ``(8, 16, 2048)``, ``micro_seqs = 16`` gives ``(16, 16, 1024)`` and ``(16, 8, 2048)``. Every
plan has the same tokens per step, so the loader's batches of the configured plan serve every
plan after a reshape of the two leading axes (row ``m`` of a step stays row ``m``).

Measurement (:func:`make_measure`): build the train state and the step of
``bes.train.train.make_train_step`` for the plan, lower and compile it for the step shape (compile
time), read ``memory_analysis()`` (the compiler's peak; a plan above ``memory_limit`` is skipped),
run ``warmup`` steps and then ``steps`` timed steps on real batches from the loader (step time,
tokens per second). Any exception (compile failure, out of memory at run time) skips the plan
with its reason. Measurements are appended to ``results.jsonl`` so that a restarted benchmark
does not measure a plan twice.

Search (:func:`search`), within the benchmark's time budget (no new candidate after the soft
deadline, except the configured plan, and nothing after the hard deadline):

1. the configured plan at every sequence length of the run;
2. the rematerialisation flip (off with half the micro-batch, or on with twice it);
3. the table-gradient flip of the best plan so far;
4. the attention flip of the best plan so far (the flash kernel needs a TPU: on a CPU host it
   is recorded as skipped).

A flipped candidate is measured at the base length first and at the long length only if it is
faster there. ``mode = "reduced"`` (CPU rehearsals) stops after step 2.

Selection (:func:`select_plan`): per sequence length the fastest valid plan; for the run the plan
valid at every length with the largest expected throughput ``sum_T share_T * tps_T``, where
``share_T`` is the wall-time share of length ``T`` (``late_from`` and ``1 - late_from``: progress
is linear in wall time).

Canary (:func:`canary`) on the chosen plan, with real batches:

- parity: the loss of one micro-batch under the chosen plan and under the dense attention plan
  agree within ``2e-2`` relative with bfloat16 activations (``1e-4`` in float32);
- determinism: the same batch twice gives the same loss, bit for bit;
- ``steps`` steps of the safe configuration (every feature switch off) and of the full one
  (the run's switches), each from the same initial parameters on the same distinct batches,
  must stay finite and lower the loss (:func:`loss_drops`: the mean of the last fifth at most
  ``1 - min_drop`` times the mean of the first fifth). Switches are runtime inputs of the compiled step, so every run reuses one program.
  If the full configuration fails and the safe one passes, the rungs of
  ``bes.train.supervisor.LADDER`` (feature groups off one at a time, cumulatively, in the order
  of plan section 6) are tried until one passes; that rung is used for the run. If the safe
  configuration fails the run cannot train: :class:`CanaryError`.

Host pipeline (:func:`host_rate`): loader rows per second against the rows the device consumes
(``A * B / step_s``); :func:`loader_workers_needed` raises the worker count (within the core plan)
when the loader is the bottleneck.

Horizon (:func:`horizon`): ``bes.train.schedule.estimate_horizon_tokens(tokens per second,
planned training seconds)`` with the throughput of the chosen plan, limited by the loader; the
value goes to ``optim.horizon_tokens`` (``schedule.with_horizon``) and ``LoaderCfg.horizon_tokens``
before the optimiser is built, and is reused on every resume.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bes.config import Cfg, apply_overrides
from bes.train import schedule

GIB = 2 ** 30


class BenchmarkError(RuntimeError):
    """No execution plan works on this machine."""


class CanaryError(RuntimeError):
    """The safe configuration does not train: the run cannot continue."""


# ------------------------------------------------------------------------------ plans
@dataclass(frozen=True)
class Plan:
    """One execution plan (module docstring)."""

    attn_impl: str
    remat: bool
    micro_seqs: int
    table_grad: str

    @classmethod
    def of(cls, cfg: Cfg) -> "Plan":
        return cls(cfg.model.attn_impl, bool(cfg.model.remat), int(cfg.optim.micro_seqs),
                   cfg.model.table_grad)

    @property
    def key(self) -> str:
        return f"{self.attn_impl}-remat{int(self.remat)}-ms{self.micro_seqs}-{self.table_grad}"

    def overrides(self) -> list[str]:
        return [f"model.attn_impl={self.attn_impl}", f"model.remat={str(self.remat).lower()}",
                f"optim.micro_seqs={self.micro_seqs}", f"model.table_grad={self.table_grad}"]

    def apply(self, cfg: Cfg) -> Cfg:
        return apply_overrides(cfg, self.overrides())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Plan":
        return cls(str(d["attn_impl"]), bool(d["remat"]), int(d["micro_seqs"]), str(d["table_grad"]))

    def flip_remat(self) -> "Plan":
        """Remat off with half the micro-batch, or on with twice it."""
        if self.remat:
            return dataclasses.replace(self, remat=False, micro_seqs=max(1, self.micro_seqs // 2))
        return dataclasses.replace(self, remat=True, micro_seqs=self.micro_seqs * 2)

    def flip_table(self) -> "Plan":
        return dataclasses.replace(self, table_grad="take" if self.table_grad == "onehot" else "onehot")

    def flip_attn(self) -> "Plan":
        return dataclasses.replace(self, attn_impl="xla" if self.attn_impl == "splash" else "splash")


@dataclass
class Measurement:
    """Result of one plan at one sequence length (sizes in bytes, times in seconds)."""

    plan: Plan
    seq_len: int
    ok: bool
    reason: str = ""
    shape: tuple[int, int, int] = (0, 0, 0)
    compile_s: float = 0.0
    step_s: float = 0.0
    tokens_per_s: float = 0.0
    peak_bytes: int = 0
    memory: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)
    steps: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["plan"] = self.plan.to_dict()
        d["shape"] = list(self.shape)
        d["losses"] = [x if math.isfinite(x) else str(x) for x in self.losses]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Measurement":
        kw = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d}
        kw["plan"] = Plan.from_dict(d["plan"])
        kw["shape"] = tuple(d.get("shape", (0, 0, 0)))
        kw["losses"] = [float(x) for x in d.get("losses", [])]
        return cls(**kw)


def seq_lens(cfg: Cfg) -> tuple[int, ...]:
    """Sequence lengths of the run in order of use."""
    return tuple(t for _, _, t in schedule.shapes(cfg))


def time_shares(cfg: Cfg) -> dict[int, float]:
    """Wall-time share of every sequence length (progress is linear in wall time)."""
    lens = seq_lens(cfg)
    if len(lens) == 1:
        return {lens[0]: 1.0}
    f = min(1.0, max(0.0, cfg.data.late_from))
    return {lens[0]: f, lens[1]: 1.0 - f}


def plan_valid(cfg: Cfg, plan: Plan) -> str:
    """``""`` when the plan's shapes exist for ``cfg``, else the reason."""
    try:
        schedule.shapes(plan.apply(cfg))
    except ValueError as e:
        return f"invalid shape: {e}"
    return ""


# ----------------------------------------------------------------------------- search
Measure = Callable[[Plan, int], Measurement]


def search(cfg: Cfg, measure: Measure, *, clock: Callable[[], float], soft_deadline: float,
           hard_deadline: float, mode: str = "full", platform: str = "tpu",
           known: Mapping[tuple[str, int], Measurement] | None = None,
           record: Callable[[Measurement], None] | None = None,
           log: Callable[[str], None] | None = None) -> list[Measurement]:
    """Measure candidate plans in the order of the module docstring; returns every
    measurement, skipped candidates included (``ok = False`` with the reason)."""
    say = log or (lambda msg: None)
    lens = seq_lens(cfg)
    base = Plan.of(cfg)
    results: dict[tuple[str, int], Measurement] = {}
    known = dict(known or {})

    def run(plan: Plan, t: int, essential: bool = False) -> Measurement:
        k = (plan.key, t)
        if k in results:
            return results[k]
        if k in known:
            m = known[k]
            m.cached = True
            results[k] = m
            return m
        why = plan_valid(cfg, plan)
        if not why and plan.attn_impl == "splash" and platform != "tpu":
            why = "the flash (splash) kernel needs a TPU; skipped on this host"
        now = clock()
        if not why and now >= hard_deadline:
            why = "benchmark hard cap reached"
        if not why and now >= soft_deadline and not essential:
            why = "benchmark time budget used up"
        if why:
            m = Measurement(plan, t, False, why)
            results[k] = m
            say(f"benchmark: {plan.key} T={t}: skipped ({why})")
            return m
        m = measure(plan, t)
        results[k] = m
        if record is not None:
            record(m)
        if m.ok:
            say(f"benchmark: {plan.key} T={t}: {m.tokens_per_s:,.0f} tok/s, step {m.step_s:.3f} s, "
                f"compile {m.compile_s:.1f} s, peak {m.peak_bytes / GIB:.2f} GiB")
        else:
            say(f"benchmark: {plan.key} T={t}: FAILED ({m.reason})")
        return m

    def best_at(t: int) -> Measurement | None:
        ok = [m for (key, tt), m in results.items() if tt == t and m.ok]
        return max(ok, key=lambda m: m.tokens_per_s) if ok else None

    def best_plan() -> Plan:
        # the fastest plan at the base length among those measured at every length, else the
        # fastest at the base length
        complete = [m for m in (results.get((p.key, lens[0])) for p in _plans(results))
                    if m is not None and m.ok and all(results.get((m.plan.key, t)) is not None
                                                      and results[(m.plan.key, t)].ok for t in lens)]
        if complete:
            return max(complete, key=lambda m: m.tokens_per_s).plan
        b = best_at(lens[0])
        return base if b is None else b.plan

    def challenge(plan: Plan) -> None:
        cur = best_at(lens[0])
        m = run(plan, lens[0])
        if m.ok and (cur is None or m.tokens_per_s > cur.tokens_per_s):
            for t in lens[1:]:
                run(plan, t)

    for t in lens:
        run(base, t, essential=True)
    challenge(base.flip_remat())
    if mode != "reduced":
        challenge(best_plan().flip_table())
        challenge(best_plan().flip_attn())
    else:
        alt = best_plan().flip_attn()
        if alt.attn_impl == "splash" and platform != "tpu":
            run(alt, lens[0])          # recorded as skipped (needs a TPU)
    return list(results.values())


def _plans(results: Mapping[tuple[str, int], Measurement]) -> list[Plan]:
    seen: dict[str, Plan] = {}
    for m in results.values():
        seen.setdefault(m.plan.key, m.plan)
    return list(seen.values())


def select_plan(measurements: Sequence[Measurement], lens: Sequence[int],
                shares: Mapping[int, float]) -> dict[str, Any]:
    """The plan for the run (module docstring): ``{"plan", "per_seq_len", "expected_tps",
    "candidates"}``; raises :class:`BenchmarkError` when no plan is valid at every length."""
    by: dict[str, dict[int, Measurement]] = {}
    for m in measurements:
        by.setdefault(m.plan.key, {})[m.seq_len] = m
    per_len = {}
    for t in lens:
        ok = [m for m in measurements if m.seq_len == t and m.ok]
        if ok:
            b = max(ok, key=lambda m: m.tokens_per_s)
            per_len[t] = {"plan": b.plan.key, "tokens_per_s": b.tokens_per_s, "step_s": b.step_s}
    cands = []
    for key, ms in by.items():
        if all(t in ms and ms[t].ok for t in lens):
            exp = sum(shares[t] * ms[t].tokens_per_s for t in lens)
            cands.append((exp, key, ms[lens[0]].plan))
    if not cands:
        reasons = sorted({f"{m.plan.key} T={m.seq_len}: {m.reason}" for m in measurements if not m.ok})
        raise BenchmarkError("no execution plan compiled and ran at every sequence length "
                             f"{list(lens)}: {reasons[:8]}")
    cands.sort(key=lambda c: (-c[0], c[1]))
    exp, key, plan = cands[0]
    return {"plan": plan, "plan_key": key, "per_seq_len": per_len, "expected_tps": exp,
            "candidates": [{"plan": k, "expected_tps": e} for e, k, _ in cands]}


# ------------------------------------------------------------------------ measurement
def reshape_batch(batch: Mapping[str, np.ndarray], shape: Sequence[int]) -> dict[str, np.ndarray]:
    """A batch ``[A0, B0, ...]`` as ``[A, B, ...]`` with ``A * B = A0 * B0`` (same rows)."""
    a, b, _ = (int(x) for x in shape)
    out = {}
    for k, v in batch.items():
        v = np.ascontiguousarray(v)
        if v.shape[0] * v.shape[1] != a * b:
            raise ValueError(f"{k}: {v.shape[:2]} rows cannot become ({a}, {b})")
        out[k] = v.reshape((a, b) + v.shape[2:])
    return out


def memory_limit_bytes(device_report: Mapping[str, Any] | None = None) -> int | None:
    """``min(28 GiB, 90% of the device memory)`` on a TPU (``bes.train.aot.HBM_LIMIT_GIB``),
    ``None`` (no limit) elsewhere."""
    from bes.train.aot import HBM_LIMIT_GIB

    rep = device_report or {}
    if rep.get("backend") != "tpu":
        return None
    lim = HBM_LIMIT_GIB * GIB
    dev = rep.get("memory_limit_gb")
    if dev:
        lim = min(lim, 0.9 * float(dev) * GIB)
    return int(lim)


def make_measure(cfg: Cfg, batches: Mapping[int, Sequence[Mapping[str, np.ndarray]]], *,
                 steps: int = 20, warmup: int = 2, memory_limit: int | None = None,
                 consts_switches: Mapping[str, float] | None = None, seed: int = 0) -> Measure:
    """The real :data:`Measure` on this process's first device (module docstring).

    batches: ``{seq_len: [batch, ...]}`` real loader batches of ``cfg``'s own shape at that
    length (reshaped for every plan; they are cycled)."""
    import jax
    import jax.numpy as jnp

    from bes.model import model as M
    from bes.roles import role_of
    from bes.train import train as TR
    from bes.train.aot import memory_stats

    sw = dict(consts_switches or {})

    def measure(plan: Plan, t: int) -> Measurement:
        pcfg = plan.apply(cfg)
        shape = schedule.micro_shape(pcfg, t)
        m = Measurement(plan, t, False, shape=shape)
        state = compiled = dev = None
        try:
            consts = M.make_consts(pcfg.model, **sw)
            state, opt = TR.init_state(pcfg, M.init_params, role_of, jax.random.key(seed), consts)
            step = TR.make_train_step(M.loss_fn, opt, pcfg)
            scalar = jax.ShapeDtypeStruct((), jnp.float32)
            t0 = time.perf_counter()
            compiled = step.lower(TR.abstract(state), TR.batch_spec(shape), scalar, scalar,
                                  scalar).compile()
            m.compile_s = time.perf_counter() - t0
            m.peak_bytes, m.memory = memory_stats(compiled)
            if memory_limit is not None and m.peak_bytes > memory_limit:
                m.reason = (f"compiled peak {m.peak_bytes / GIB:.2f} GiB above the limit "
                            f"{memory_limit / GIB:.2f} GiB")
                return m
            dev = [jax.device_put(reshape_batch(b, shape)) for b in batches[t]]
            lr, mom, wd = np.float32(1e-3), np.float32(0.95), np.float32(1.0)
            for i in range(max(1, warmup)):
                state, met = compiled(state, dev[i % len(dev)], lr, mom, wd)
            jax.block_until_ready(met["loss"])
            losses = []
            t1 = time.perf_counter()
            for i in range(steps):
                state, met = compiled(state, dev[i % len(dev)], lr, mom, wd)
                losses.append(met["loss"])
            jax.block_until_ready(state.params)
            dt = time.perf_counter() - t1
            m.losses = [float(x) for x in jax.device_get(losses)]
            m.steps = steps
            m.step_s = dt / max(1, steps)
            m.tokens_per_s = shape[0] * shape[1] * shape[2] / m.step_s if m.step_s > 0 else 0.0
            if not all(math.isfinite(x) for x in m.losses):
                m.reason = "non-finite loss in the timed steps"
                return m
            m.ok = True
        except Exception as e:  # noqa: BLE001 - every failure skips the plan with its reason
            text = str(e).strip().splitlines()
            m.reason = f"{type(e).__name__}: {text[0][:300] if text else ''}"
        finally:
            del state, compiled, dev
            gc.collect()
        return m

    return measure


def load_results(path: str | Path) -> dict[tuple[str, int], Measurement]:
    """Measurements already recorded in ``results.jsonl`` (successful ones only: a failure is
    measured again, the machine may have changed)."""
    p = Path(path)
    out: dict[tuple[str, int], Measurement] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            m = Measurement.from_dict(json.loads(line))
        except (ValueError, KeyError, TypeError):
            continue
        if m.ok:
            out[(m.plan.key, m.seq_len)] = m
    return out


def append_result(path: str | Path, m: Measurement) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(m.to_dict(), sort_keys=True) + "\n")
        f.flush()


# ---------------------------------------------------------------------------- canary
def loss_drops(losses: Sequence[float], min_drop: float = 0.02, tail: int | None = None) -> bool:
    """All finite and the mean of the last ``tail`` losses at most ``(1 - min_drop)`` times the
    mean of the first ``tail`` (default ``tail = max(2, ceil(n / 5))``: 6 of 30 steps), so one
    noisy batch at either end decides nothing."""
    xs = [float(x) for x in losses]
    if len(xs) < 2 or not all(math.isfinite(x) for x in xs):
        return False
    k = max(1, min(len(xs) // 2, tail if tail is not None else max(2, math.ceil(len(xs) / 5))))
    return float(np.mean(xs[-k:])) <= (1.0 - min_drop) * float(np.mean(xs[:k]))


def canary_lr(i: int, warm: int = 10) -> float:
    """Learning-rate factor of canary step ``i``: linear warm-up over ``warm`` steps, then 1."""
    return min(1.0, (i + 1) / warm)


def canary_warmup(steps: int) -> int:
    """Warm-up steps of a canary run of ``steps`` steps: a third of it, between 2 and 10."""
    return max(2, min(10, steps // 3))


def choose_rung(full_ok: bool, safe_ok: bool, try_rung: Callable[[int], bool],
                n_rungs: int) -> tuple[int | str, list[int]]:
    """The fallback decision: ``0`` when the full configuration passes; otherwise (the safe one
    must pass, else :class:`CanaryError`) the first rung ``k = 1..n_rungs`` that passes, else
    ``"safe"``. Returns ``(rung, rungs tried)``."""
    if full_ok:
        return 0, []
    if not safe_ok:
        raise CanaryError("the safe configuration (every feature switch off) does not train: "
                          "its loss is non-finite or does not fall; the run cannot continue")
    tried = []
    for k in range(1, n_rungs + 1):
        tried.append(k)
        if try_rung(k):
            return k, tried
    return "safe", tried


def canary(cfg: Cfg, plan: Plan, batches: Sequence[Mapping[str, np.ndarray]], *, steps: int = 30,
           min_drop: float = 0.02, seed: int = 0, log: Callable[[str], None] | None = None,
           clock: Callable[[], float] = time.time, deadline: float | None = None) -> dict[str, Any]:
    """The correctness canary of the module docstring on ``batches`` (real loader batches of the
    configured shape at the base length; at least ``steps`` of them, cycled otherwise).

    Returns ``{"parity", "determinism", "full", "safe", "ladder", "ok"}``; ``ladder`` holds the
    chosen ``rung`` (0 = full configuration, ``k``, or ``"safe"``), its ``groups``, ``switches``
    (``RunCfg.switches`` text) and config ``overrides``."""
    import jax

    from bes.model import model as M
    from bes.roles import role_of
    from bes.train import train as TR
    from bes.train.supervisor import LADDER, format_switches, ladder_rung, parse_switches

    say = log or (lambda msg: None)
    pcfg = plan.apply(cfg)
    mcfg = pcfg.model
    t = pcfg.data.seq_len
    shape = schedule.micro_shape(pcfg, t)
    run_sw = parse_switches(cfg.run.switches)
    key = jax.random.key(seed)
    params = M.init_params(mcfg, key)
    full_consts = M.make_consts(mcfg, **run_sw)
    b0 = reshape_batch(batches[0], shape)
    mb = {k: v[0] for k, v in b0.items()}
    out: dict[str, Any] = {"plan": plan.key, "shape": list(shape), "steps": steps}

    # parity and determinism on one micro-batch
    def loss_of(model_cfg):
        f = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, model_cfg)[0])
        return float(f(params, full_consts, mb)), float(f(params, full_consts, mb))

    l1, l2 = loss_of(mcfg)
    out["determinism"] = {"loss": l1, "repeat": l2, "ok": bool(np.float32(l1) == np.float32(l2))}
    tol = 2e-2 if mcfg.act_dtype == "bfloat16" else 1e-4
    if mcfg.attn_impl == "xla":
        out["parity"] = {"ok": True, "note": "the chosen plan uses the dense attention itself",
                         "loss": l1, "tol": tol}
    else:
        ld, _ = loss_of(dataclasses.replace(mcfg, attn_impl="xla"))
        rel = abs(l1 - ld) / max(1e-12, abs(ld))
        out["parity"] = {"ok": bool(rel <= tol), "loss": l1, "dense": ld, "rel": rel, "tol": tol}
    say(f"canary: determinism {out['determinism']['ok']}, parity {out['parity']}")

    # safe / full / ladder runs: one compiled step (the program the measurement compiled, so a
    # persistent compilation cache serves it), switches are runtime inputs; every run starts
    # from the same initial parameters and a fresh optimiser state
    import jax.numpy as jnp

    _, opt = TR.init_state(pcfg, M.init_params, role_of, key, full_consts)
    step = TR.make_train_step(M.loss_fn, opt, pcfg)
    dev = [jax.device_put(reshape_batch(b, shape)) for b in batches[:max(1, steps)]]

    def run_steps(switches: Mapping[str, float], label: str) -> dict[str, Any]:
        p0 = M.init_params(mcfg, key)
        state = TR.TrainState(step=jnp.zeros((), jnp.int32), params=p0, opt_state=opt.init(p0),
                              consts=M.make_consts(mcfg, **switches))
        losses = []
        t0 = time.perf_counter()
        warm = canary_warmup(steps)
        for i in range(steps):
            if deadline is not None and clock() > deadline:
                break
            state, met = step(state, dev[i % len(dev)], np.float32(canary_lr(i, warm)),
                              np.float32(0.95), np.float32(1.0))
            losses.append(float(met["loss"]))     # blocks: the deadline check sees real time
        vals = losses
        del state
        res = {"losses": [round(x, 5) if math.isfinite(x) else str(x) for x in vals],
               "ok": loss_drops(vals, min_drop) and len(vals) == steps,
               "seconds": round(time.perf_counter() - t0, 2), "switches": dict(switches)}
        if len(vals) < steps:
            res["note"] = f"stopped after {len(vals)} steps at the benchmark's hard cap"
        say(f"canary: {label}: first {vals[0] if vals else float('nan'):.4f} last "
            f"{vals[-1] if vals else float('nan'):.4f} ok={res['ok']}")
        return res

    safe_sw = dict(M.switches_off())
    out["safe"] = run_steps(safe_sw, "safe configuration")
    out["full"] = run_steps(run_sw, "full configuration")
    tried: dict[int, dict] = {}

    def try_rung(k: int) -> bool:
        sw, _, groups = ladder_rung(k)
        tried[k] = run_steps({**run_sw, **sw}, f"rung {k} ({'+'.join(groups)} off)")
        return tried[k]["ok"]

    rung, _ = choose_rung(out["full"]["ok"], out["safe"]["ok"], try_rung, len(LADDER))
    if rung == "safe":
        _, ov, groups = ladder_rung(len(LADDER))
        sw = safe_sw
    else:
        sw_k, ov, groups = ladder_rung(int(rung))
        sw = {**run_sw, **sw_k}
    out["rungs"] = {str(k): v for k, v in tried.items()}
    out["ladder"] = {"rung": rung, "groups": groups,
                     "switches": cfg.run.switches if rung == 0 else format_switches(sw),
                     "overrides": list(ov)}
    if rung == 0:
        out["ladder"]["switches"] = None      # nothing to change: keep the configuration as is
    out["ok"] = bool(out["determinism"]["ok"] and out["parity"]["ok"])
    del dev
    gc.collect()
    return out


# ------------------------------------------------------------------ host pipeline rate
def host_rate(loader: Callable[[int, Any], Any], stage: Any, *, start: int, n: int = 8,
              warm: int = 2, clock: Callable[[], float] = time.perf_counter) -> dict[str, float]:
    """Rows per second the loader delivers at ``stage`` (``n`` timed batches after ``warm``),
    from data index ``start`` on (indices it has not produced yet)."""
    for i in range(warm):
        loader(start + i, stage)
    t0 = clock()
    for i in range(n):
        loader(start + warm + i, stage)
    dt = max(1e-9, clock() - t0)
    a, b, _ = stage.shape
    return {"rows_per_s": n * a * b / dt, "batches": float(n), "seconds": dt,
            "next_index": float(start + warm + n)}


def loader_workers_needed(rows_per_s: float, workers: int, needed_rows_per_s: float,
                          max_workers: int, margin: float = 1.15) -> int:
    """Worker count at which the loader keeps up with the device (linear in the workers),
    between ``workers`` and ``max_workers``."""
    if rows_per_s <= 0:
        return max_workers
    if rows_per_s >= margin * needed_rows_per_s:
        return workers
    per = rows_per_s / max(1, workers)
    return int(min(max_workers, max(workers, math.ceil(margin * needed_rows_per_s / per))))


def horizon(expected_tps: float, loader_tps: float | None, train_seconds: float) -> dict[str, float]:
    """Expected training tokens: ``min(device, loader) tokens/s x planned training seconds``."""
    tps = expected_tps if not loader_tps else min(expected_tps, loader_tps)
    return {"tokens_per_s": tps, "train_seconds": train_seconds,
            "horizon_tokens": schedule.estimate_horizon_tokens(tps, train_seconds)}


__all__ = ["Plan", "Measurement", "BenchmarkError", "CanaryError", "seq_lens", "time_shares",
           "plan_valid", "search", "select_plan", "reshape_batch", "memory_limit_bytes",
           "make_measure", "load_results", "append_result", "loss_drops", "canary_lr",
           "choose_rung", "canary", "host_rate", "loader_workers_needed", "horizon"]
