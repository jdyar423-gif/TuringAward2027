"""Training step and training loop (SPEC section 5).

Step
----
:func:`make_train_step` returns one jit-compiled function

    train_step(state, batch, lr_scale, momentum=None, wd_scale=None, group_lr=None)
        -> (state, metrics)

``batch`` is a dict of SPEC section 3 arrays with a leading micro-batch axis ``A``. The step scans
(``lax.scan``) over the ``A`` micro-batches, accumulating in float32 the gradient of each
micro-batch's mean loss multiplied by its token weight ``W_i = sum(w_i)``, so the update uses the
gradient of the weighted mean over the whole step, ``sum_i W_i grad L_i / sum_i W_i`` (identical to
one big batch). One optimiser update follows (``bes.optim``; it clips if ``grad_clip > 0``).
If the loss or the gradient norm is not finite the update is skipped in-graph: parameters and
optimiser state are returned unchanged and ``metrics["skipped"] = 1``. The state is donated.
``group_lr`` (``{group: float32 []}``, :func:`bes.optim.build.lr_inputs`) makes every group's
learning rate a run-time input; the trainer always passes it, so its compiled program does not
depend on the learning rates (the horizon estimate changes the matrix rate, not the program) and
the start-up benchmark's compiled step serves the trainer. Without it the rates fixed when the
optimiser was built are used (compile-time constants).
Metrics are device arrays (the caller reads them one step late):

``loss``       float32 [], weighted mean loss of the step
``weight``     float32 [], total token weight
``grad_norm``  float32 [], global norm of the accumulated gradient (before clipping)
``skipped``    int32 [], 1 if the update was skipped
``lr_scale``   float32 []
``micro``      the loss function's metrics stacked over the micro-batches (leading axis ``A``)

Compilation: :func:`compile_step` lowers the step for one shape, fingerprints the lowered program
(:func:`program_fingerprint`: its text without debug information, the argument structure and
types, the donated arguments) and compiles it, or reuses an executable handed over with the same
fingerprint (the benchmark's, in the same process).

Interfaces taken from the model track (SPEC 4.3/4.4): ``init_fn(model_cfg, key) -> params``,
``loss_fn(params, consts, micro_batch, model_cfg) -> (loss, metrics)``, ``role_fn(path)`` for the
optimiser groups and ``consts_fn(model_cfg) -> consts`` (non-trainable inputs such as the 0/1
feature switches; they live in the train state so they can change without recompiling).

Loop
----
:class:`Trainer` owns the run: it fixes the deadline (``run.json``), builds the state and the
optimiser, compiles the step for every shape of the schedule before the first step, restores the
newest checkpoint if there is one (otherwise saves the initial state), then loops: progress ``p``
from the ledger and the clock, the :class:`~bes.train.schedule.Stage` at ``p``, the batch for the
current data index, one asynchronous step, then the previous step's metrics (one step late) go
to the guard and the log: its loss, and a skipped update (non-finite loss or gradient) as a
non-finite event (``skipped_update`` in the log), so a step whose loss is finite but whose
gradient is not cannot be dropped silently. The loop stops when the next step would end after the
training window, at ``max_steps``, on request, or when the guard halts.

Checkpoints are written on a wall-clock interval (only while the guard is healthy; the pending
metrics are read first so that the stored guard state matches the stored step). The initial,
final and rollback saves are blocking and checked (the checkpoint is complete on disk before the
trainer goes on; a failed initial save stops the set-up); interval saves may run in the
background (``async_ckpt``) and count as saved (``ckpt`` event, ``_saved``) only once their write
has succeeded. Every save protects the checkpoint a rollback would restore from the room-making of
:class:`~bes.train.checkpoint.Checkpointer` (a full disk costs old checkpoints, not the trusted
one). A final save that fails is retried once; then the parameters alone are saved under
``<out_dir>/ckpt_params/``; if that fails too, :meth:`Trainer.run` raises
:class:`~bes.train.checkpoint.CheckpointError`: training never ends normally without its final
state on disk.

A rollback restores the newest trusted checkpoint, moves the data index past everything seen
since plus ``guard_skip_steps``, lowers the learning rate and re-saves the restored state so the
decision survives a crash. When there is no checkpoint to restore, or the trusted one is more than
``guard_max_rollback_intervals`` checkpoint intervals of training behind, the current state is
kept instead: the learning-rate cut and the data skip apply, the guard's average is re-based on
the state kept, and an ``alarm`` is logged (hours of training are not thrown away for a spike).

Data: ``data(index, stage) -> batch`` (or an object with ``get(index, stage)``), a pure function of
the data index for a given stage so that restarts reproduce the same batches. The trainer keeps
the index (the data cursor) and checkpoints it; :class:`IteratorData` adapts an iterator factory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG, Cfg, config_hash
from bes.optim import build as build_optimizer
from bes.optim import global_norm
from bes.optim.build import lr_inputs
from bes.train import guard as guard_lib
from bes.train import schedule
from bes.train.checkpoint import CheckpointError, Checkpointer, CkptInfo, read_manifest
from bes.train.ledger import Ledger, atomic_write_json

F32 = jnp.float32

#: SPEC section 3: key -> (dtype, trailing dims after ``[A, B, T]``); ``tokens`` has ``T + 1``.
BATCH_FIELDS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "tokens": (np.int32, ()),
    "w": (np.float32, ()),
    "seg": (np.int32, ()),
    "reg": (np.int16, (N_REG,)),
    "clock": (np.int32, (N_CLOCKS,)),
    "cand_tok": (np.int32, (CAND_SLOTS,)),
    "cand_feat": (np.int8, (CAND_SLOTS, N_CAND_FEAT)),
}
BATCH_KEYS: tuple[str, ...] = tuple(BATCH_FIELDS)
#: Saves that are written synchronously and checked before the trainer goes on.
BLOCKING_SAVES = frozenset({"init", "final", "rollback"})
#: Directory (inside the trainer's ``out_dir``) of a parameters-only final checkpoint.
PARAMS_DIR = "ckpt_params"


def batch_spec(shape: tuple[int, int, int],
               keys: tuple[str, ...] = BATCH_KEYS) -> dict[str, jax.ShapeDtypeStruct]:
    """Abstract batch of shape ``(A, B, T)`` with the SPEC section 3 dtypes for ``keys``."""
    a, b, t = shape
    out = {}
    for k in keys:
        if k not in BATCH_FIELDS:
            raise KeyError(f"unknown batch key {k!r} (SPEC section 3 keys: {BATCH_KEYS})")
        dtype, tail = BATCH_FIELDS[k]
        out[k] = jax.ShapeDtypeStruct((a, b, t + 1 if k == "tokens" else t) + tail, dtype)
    return out


class TrainState(NamedTuple):
    """step: int32 [] updates taken (skipped ones included); params; opt_state (``OptState``);
    consts: the loss function's non-trainable inputs (may be ``{}``)."""

    step: jax.Array
    params: Any
    opt_state: Any
    consts: Any


def abstract(tree: Any) -> Any:
    """``jax.ShapeDtypeStruct`` for every leaf of ``tree``."""
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(np.shape(x), x.dtype), tree)


def init_state(cfg: Cfg, init_fn: Callable, optimizer_or_role_fn: Any, key: jax.Array,
               consts: Any = None) -> tuple[TrainState, Any]:
    """``(state, optimizer)``: parameters from ``init_fn(cfg.model, key)``, the optimiser built
    for them (``optimizer_or_role_fn`` is an ``Optimizer`` or a role function) and its state."""
    params = init_fn(cfg.model, key)
    if hasattr(optimizer_or_role_fn, "update"):
        opt = optimizer_or_role_fn
    else:
        opt = build_optimizer(cfg, params, optimizer_or_role_fn)
    state = TrainState(step=jnp.zeros((), jnp.int32), params=params, opt_state=opt.init(params),
                       consts={} if consts is None else consts)
    return state, opt


#: Metrics of the loss function that are sums over a micro-batch's tokens; the others are means
#: weighted by the tokens' loss weights (``bes.model.model.loss_fn``).
MICRO_SUMS = frozenset({"ce_sum", "weight", "tokens", "ce_by_role", "weight_by_role"})


def split_micro(batch: Mapping[str, Any], k: int) -> dict[str, Any]:
    """``[A, B, ...]`` -> ``[A * k, B // k, ...]``: every micro-batch cut into ``k`` pieces of
    whole rows, in order. The update does not change (rows are independent and the accumulation
    is token-weighted); the tensors of one ``value_and_grad`` get ``k`` times smaller."""
    out = {}
    for name, v in batch.items():
        a, b = v.shape[:2]
        if b % k:
            raise ValueError(f"{name}: {b} rows per micro-batch cannot be cut into {k} pieces")
        out[name] = v.reshape(a * k, b // k, *v.shape[2:])
    return out


def regroup_micro(micro: Any, k: int) -> Any:
    """Metrics stacked over ``A * k`` pieces as the loss function reports them for the ``A``
    micro-batches: sums (:data:`MICRO_SUMS`) are added, the others averaged with the pieces'
    token weights (plain means without a ``weight`` metric); the loss shares are recomputed."""
    if not isinstance(micro, Mapping):
        return jax.tree.map(lambda v: jnp.mean(v.reshape(-1, k, *v.shape[1:]), axis=1), micro)
    cut = {name: v.reshape(-1, k, *v.shape[1:]) for name, v in micro.items()}
    w = cut.get("weight")
    out = {}
    for name, v in cut.items():
        if name in MICRO_SUMS:
            out[name] = jnp.sum(v, axis=1)
        elif w is None:
            out[name] = jnp.mean(v, axis=1)
        else:
            ww = w.reshape(w.shape + (1,) * (v.ndim - 2)).astype(F32)
            total = jnp.sum(ww, axis=1)
            mean = jnp.sum(ww * v, axis=1) / jnp.maximum(total, jnp.asarray(1e-9, F32))
            out[name] = jnp.where(total > 0, mean, jnp.mean(v, axis=1)).astype(v.dtype)
    if "ce_by_role" in out and "ce_sum" in out and "loss_share_by_role" in out:
        out["loss_share_by_role"] = out["ce_by_role"] / jnp.maximum(out["ce_sum"][..., None], 1e-9)
    return out


def accumulate(loss_fn: Callable, params: Any, consts: Any, batch: Mapping[str, Any],
               model_cfg: Any, micro_split: int = 1) -> tuple[Any, jax.Array, jax.Array, Any]:
    """Token-weighted gradient accumulation over the leading micro-batch axis of ``batch``.

    Returns ``(grads float32 tree, loss, weight, micro_metrics)`` with
    ``grads = sum_i W_i grad L_i / W``, ``loss = sum_i W_i L_i / W``, ``W = sum_i W_i`` and
    ``W_i = sum(w_i)``; ``micro_metrics`` stacks the loss function's metrics over ``A``.

    micro_split: cut every micro-batch into this many pieces on the device
    (:func:`split_micro`) and scan over the pieces; gradients, loss and weight are the same to
    float rounding, the metrics are regrouped to the ``A`` micro-batches. On the TPU v6e a
    step is memory-bound outside its matrix products and smaller pieces are faster (4 pieces of
    a 16 x 1024 micro-batch: +10%, 24 Nov 2026).
    """
    k = int(micro_split)
    if k > 1:
        batch = split_micro(batch, k)
    vg = jax.value_and_grad(loss_fn, has_aux=True)

    def body(carry, mb):
        gsum, lsum, wsum = carry
        (loss, aux), g = vg(params, consts, mb, model_cfg)
        w = jnp.sum(jnp.asarray(mb["w"], F32))
        gsum = jax.tree.map(lambda a, gi: a + w * gi.astype(F32), gsum, g)
        return (gsum, lsum + w * jnp.asarray(loss, F32), wsum + w), aux

    zeros = jax.tree.map(lambda p: jnp.zeros(np.shape(p), F32), params)
    init = (zeros, jnp.zeros((), F32), jnp.zeros((), F32))
    (gsum, lsum, wsum), micro = jax.lax.scan(body, init, dict(batch))
    if k > 1:
        micro = regroup_micro(micro, k)
    inv = 1.0 / jnp.maximum(wsum, jnp.asarray(1e-9, F32))
    return jax.tree.map(lambda g: g * inv, gsum), lsum * inv, wsum, micro


def make_train_step(loss_fn: Callable, optimizer: Any, cfg: Cfg, *, donate: bool = True,
                    skip_nonfinite: bool = True, micro_split: int = 1) -> Callable:
    """The jit-compiled training step described in the module docstring (``micro_split``:
    :func:`accumulate`)."""
    model_cfg = cfg.model

    def train_step(state: TrainState, batch: Mapping[str, Any], lr_scale: Any,
                   momentum: Any = None, wd_scale: Any = None, group_lr: Any = None):
        grads, loss, weight, micro = accumulate(loss_fn, state.params, state.consts, batch,
                                                model_cfg, micro_split)
        gnorm = global_norm(grads)
        kw = {}
        if momentum is not None:
            kw["momentum"] = momentum
        if wd_scale is not None:
            kw["wd_scale"] = wd_scale
        if group_lr is not None:
            kw["group_lr"] = group_lr
        lr_scale = jnp.asarray(lr_scale, F32)
        params, opt_state = optimizer.update(grads, state.opt_state, state.params, lr_scale, **kw)
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        if skip_nonfinite:
            keep = lambda new, old: jnp.where(ok, new, old)  # noqa: E731
            params = jax.tree.map(keep, params, state.params)
            opt_state = jax.tree.map(keep, opt_state, state.opt_state)
        new_state = TrainState(step=state.step + 1, params=params, opt_state=opt_state,
                               consts=state.consts)
        skipped = (~ok if skip_nonfinite else jnp.zeros((), bool)).astype(jnp.int32)
        metrics = {"loss": loss, "weight": weight, "grad_norm": gnorm, "skipped": skipped,
                   "lr_scale": lr_scale, "micro": micro}
        return new_state, metrics

    return jax.jit(train_step, donate_argnums=(0,) if donate else ())


# ------------------------------------------------------------------------ compilation
def lower_step(step_fn: Callable, state: Any, shape: tuple[int, int, int],
               batch_keys: tuple[str, ...] = BATCH_KEYS, group_lr: Any = None):
    """``step_fn.lower`` for batches of ``shape``: the abstract state, the SPEC batch, three
    float32 scalars (``lr_scale``, ``momentum``, ``wd_scale``) and, if given, the group rates."""
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    args = [abstract(state), batch_spec(shape, batch_keys), scalar, scalar, scalar]
    if group_lr is not None:
        args.append(abstract(group_lr))
    return step_fn.lower(*args)


def program_fingerprint(lowered: Any) -> str:
    """Hex digest identifying a lowered program: its StableHLO text without debug information
    (no source locations), the argument tree, every argument's shape, dtype and weak type, and
    the donated arguments. Two lowerings with the same fingerprint compile to the same function
    of the same arguments, so one's executable serves the other."""
    h = hashlib.sha256(lowered.as_text().encode())
    h.update(str(lowered.in_tree).encode())
    avals = [(tuple(a.shape), str(a.dtype), bool(getattr(a, "weak_type", False)))
             for a in jax.tree.leaves(lowered.in_avals)]
    h.update(repr(avals).encode())
    h.update(repr(tuple(lowered.donate_argnums)).encode())
    return h.hexdigest()


def compile_step(step_fn: Callable, state: Any, shape: tuple[int, int, int],
                 batch_keys: tuple[str, ...] = BATCH_KEYS, group_lr: Any = None,
                 reuse: tuple[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """``(executable, info)`` of ``step_fn`` at ``shape`` (see :func:`lower_step`).

    reuse: ``(fingerprint, executable)`` compiled elsewhere (the start-up benchmark); it is
    returned instead of compiling when its fingerprint equals this lowering's. ``info``:
    ``fingerprint``, ``reused``, ``seconds`` (lowering plus compilation)."""
    t0 = time.perf_counter()
    lowered = lower_step(step_fn, state, shape, batch_keys, group_lr)
    fp = program_fingerprint(lowered)
    if reuse is not None and reuse[0] == fp:
        return reuse[1], {"fingerprint": fp, "reused": True,
                          "seconds": round(time.perf_counter() - t0, 3)}
    compiled = lowered.compile()
    return compiled, {"fingerprint": fp, "reused": False,
                      "seconds": round(time.perf_counter() - t0, 3)}


# ------------------------------------------------------------------------------ data
class IteratorData:
    """``data(index, stage)`` from an iterator factory ``open_fn(start_index, stage)``.

    The iterator is (re)opened at the requested index whenever the index is not the one after the
    previous request (start, restart, rollback skip) or the stage's shape changed; otherwise the
    next batch of the open iterator is returned."""

    def __init__(self, open_fn: Callable[[int, schedule.Stage], Iterator[Mapping[str, Any]]]):
        self.open_fn = open_fn
        self._it: Iterator | None = None
        self._next = -1
        self._shape: tuple[int, int, int] | None = None

    def __call__(self, index: int, stage: schedule.Stage) -> Mapping[str, Any]:
        if self._it is None or index != self._next or stage.shape != self._shape:
            self._it = iter(self.open_fn(index, stage))
            self._shape = stage.shape
        batch = next(self._it)
        self._next = index + 1
        return batch


# ---------------------------------------------------------------------------- logging
def _plain(x: Any) -> Any:
    """JSON-friendly copy of metrics (numpy -> Python, NaN kept as float)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.ndarray, np.generic, jax.Array)):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    return x


class JsonlLog:
    """Appends one JSON object per record to ``path``; prints a short line per record if
    ``verbose``."""

    def __init__(self, path: str | os.PathLike | None, verbose: bool = True) -> None:
        self.path = None if path is None else Path(path)
        self.verbose = verbose
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict) -> None:
        rec = _plain(record)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        if self.verbose:
            if rec.get("event") == "step":
                print(f"step {rec['step']:7d} p {rec['p']:.4f} lr {rec['lr']:.4f} "
                      f"loss {rec['loss']:.4f} gn {rec['grad_norm']:.3f} T {rec['seq_len']} "
                      f"tok/s {rec.get('tok_per_s', 0):.0f}", file=sys.stderr, flush=True)
            else:
                keys = ("event", "step", "p", "reason", "action", "path")
                print(" ".join(f"{k}={rec[k]}" for k in keys if k in rec),
                      file=sys.stderr, flush=True)


@dataclass(frozen=True)
class StepInfo:
    """Host-side record of a dispatched step (its metrics are read one step late)."""

    step: int
    p: float
    lr: float
    stage: schedule.Stage
    regime: str
    time: float


def _regime_key(stage: schedule.Stage) -> str:
    return json.dumps(stage.regime)


# ---------------------------------------------------------------------------- trainer
class Trainer:
    """The training loop of one run directory (see the module docstring).

    cfg: full configuration. loss_fn / init_fn / role_fn / consts_fn: model interfaces (SPEC 4).
    data: ``data(index, stage) -> batch`` or an object with ``get(index, stage)``.
    out_dir: run directory (``run.json``, ``ledger.json``, ``train_log.jsonl``, ``ckpt/``);
    defaults to ``cfg.run.out_dir``. clock: wall clock in seconds since the epoch (injectable).
    deadline: absolute end of the whole run (default: ``run.json``, else start +
    ``deadline_hours``). max_steps: stop after this many updates (global step count).
    batch_keys: SPEC section 3 keys the data provides and the loss function reads.
    eval_fn: ``eval_fn(state, step) -> dict | None`` called every ``eval_every_min``.
    log_fn: ``log_fn(record)``; default :class:`JsonlLog` to ``train_log.jsonl``.
    async_ckpt: write interval checkpoints from a background thread. resume: restore the newest
    checkpoint of ``out_dir`` if there is one. force: load checkpoints written under another
    configuration hash. programs: ``{shape: (fingerprint, executable)}`` compiled steps to reuse
    when the trainer's own lowering has the same fingerprint (:func:`compile_step`).

    After :meth:`run`: ``history`` holds ``(step, p, lr, loss, seq_len)`` for every step whose
    loss was read, ``events`` the non-step log records, ``compile_info`` per shape whether its
    executable was reused, ``programs`` the compiled steps with their fingerprints.
    """

    def __init__(self, cfg: Cfg, *, loss_fn: Callable, init_fn: Callable, data: Any,
                 role_fn: Callable | None = None, consts_fn: Callable | None = None,
                 out_dir: str | os.PathLike | None = None, clock: Callable[[], float] = time.time,
                 deadline: float | None = None, max_steps: int | None = None,
                 batch_keys: tuple[str, ...] = BATCH_KEYS, eval_fn: Callable | None = None,
                 log_fn: Callable[[dict], None] | None = None, verbose: bool = False,
                 async_ckpt: bool = False, resume: bool = True, force: bool = False,
                 programs: Mapping[tuple[int, int, int], tuple[str, Any]] | None = None,
                 micro_split: int = 1) -> None:
        self.cfg = cfg
        self.loss_fn, self.init_fn = loss_fn, init_fn
        self.role_fn, self.consts_fn = role_fn, consts_fn
        self._data = data if callable(data) else data.get
        self.out_dir = Path(cfg.run.out_dir if out_dir is None else out_dir)
        self.clock = clock
        self._deadline_arg = deadline
        self.max_steps = max_steps
        self.batch_keys = tuple(batch_keys)
        self.eval_fn = eval_fn
        self.log = log_fn if log_fn is not None else JsonlLog(self.out_dir / "train_log.jsonl",
                                                              verbose)
        self.async_ckpt, self.resume, self.force = async_ckpt, resume, force
        self._programs = dict(programs or {})
        self.guard_cfg = guard_lib.GuardCfg.from_run(cfg.run)
        self.config_hash = config_hash(cfg)
        self.shapes = schedule.shapes(cfg)
        self.micro_split = int(micro_split)
        if self.micro_split < 1 or any(b % self.micro_split for _, b, _ in self.shapes):
            raise ValueError(f"micro_split {micro_split} does not divide the rows per micro-batch of "
                             f"every shape {self.shapes}")
        self.history: list[tuple[int, float, float, float, int]] = []
        self.events: list[dict] = []
        self.skipped_updates = 0
        self.ready = False
        self._stop_requested = False
        self._bg: tuple[CkptInfo, tuple[int, int], str] | None = None

    @property
    def params_root(self) -> Path:
        """Root of a parameters-only final checkpoint (written only when the full one fails)."""
        return self.out_dir / PARAMS_DIR

    @property
    def programs(self) -> dict[tuple[int, int, int], tuple[str, Any]]:
        """``{shape: (fingerprint, executable)}`` of the compiled steps (after :meth:`setup`)."""
        return {s: (self.compile_info[s]["fingerprint"], self.compiled[s]) for s in self.compiled}

    # ------------------------------------------------------------------ setup
    def _run_info(self, now: float) -> tuple[float, float]:
        """``(t0, deadline)`` of the run, fixed by the first start (``run.json``); a run started
        with ``resume=False`` is a new run."""
        path = self.out_dir / "run.json"
        info = json.loads(path.read_text()) if path.exists() and self.resume else {}
        t0 = float(info.get("t0", now))
        if self._deadline_arg is not None:
            deadline = float(self._deadline_arg)
        else:
            deadline = float(info.get("deadline", t0 + self.cfg.run.deadline_hours * 3600.0))
        if info.get("t0") != t0 or info.get("deadline") != deadline:
            atomic_write_json(path, {"t0": t0, "deadline": deadline,
                                     "config_hash": self.config_hash})
        return t0, deadline

    def _event(self, event: str, **fields: Any) -> None:
        rec = {"event": event, "step": self.step, "time": self.clock() - self.t0, **fields}
        self.events.append(rec)
        self.log(rec)

    def setup(self) -> None:
        """Deadline, state, optimiser, compilation of every shape, restore or initial save."""
        start = self.clock()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0, self.deadline = self._run_info(start)
        self.train_end = schedule.train_end_time(self.deadline, self.cfg.run)
        self.ledger = Ledger.start(self.t0, start)
        self.step = 0
        self.cursor = 0
        self.guard = guard_lib.GuardState()
        self.skipped_updates = 0
        self._observed_regime: str | None = None
        self._dispatched_regime: str | None = None
        self._bg = None
        self._saved: tuple[int, int] | None = None
        self._saved_info: CkptInfo | None = None
        self._halt_restored = False
        k_init, self.rng = jax.random.split(jax.random.key(self.cfg.run.seed))
        consts = self.consts_fn(self.cfg.model) if self.consts_fn is not None else {}
        self._consts_host = jax.device_get(consts)  # the state's copy is donated every step
        self.state, self.optimizer = init_state(self.cfg, self.init_fn, self.role_fn, k_init,
                                                consts)
        self.step_fn = make_train_step(self.loss_fn, self.optimizer, self.cfg,
                                       micro_split=self.micro_split)
        self.group_lr = lr_inputs(self.cfg)
        self.compiled, self.compile_info = {}, {}
        for shape in self.shapes:
            self.compiled[shape], self.compile_info[shape] = compile_step(
                self.step_fn, self.state, shape, self.batch_keys, self.group_lr,
                reuse=self._programs.get(shape))
        self.ckpt = Checkpointer(self.out_dir / "ckpt", self.config_hash,
                                 keep=self.cfg.run.keep_ckpts, async_write=self.async_ckpt)
        restored = self.ckpt.restore(self._ckpt_tree(), force=self.force) if self.resume else None
        if restored is not None:
            tree, man, info = restored
            meta = man["meta"]
            self._install(tree, man, info)
            self.guard = guard_lib.GuardState.from_dict(meta["guard"])
            self._observed_regime = meta.get("regime")
            # [snapshot, start of this process) is lost; this process's set-up is set-up.
            self.ledger = Ledger.from_dict(meta["ledger"]).resumed(start)
            self._event("restore", path=info.name, ckpt_step=info.step,
                        lost=self.ledger.seconds["lost"])
        else:
            self._save("init", fatal=True)
        now = self.clock()
        self._last_ckpt = now
        self._last_eval = now
        self._step_est = 0.0
        self.ready = True
        compiled = {"x".join(map(str, s)): {k: v for k, v in c.items() if k != "fingerprint"}
                    for s, c in self.compile_info.items()}
        self._event("setup", shapes=[list(s) for s in self.shapes], deadline=self.deadline,
                    train_end=self.train_end, ledger=self.ledger.summary(now), compile=compiled,
                    micro_split=self.micro_split)

    def _ckpt_tree(self) -> dict:
        return {"state": self.state, "rng": self.rng}

    def _install(self, tree: dict, man: dict, info: CkptInfo | None = None) -> None:
        """Adopt a restored checkpoint; the current consts (switches) replace the stored ones."""
        state, self.rng = tree["state"], tree["rng"]
        self.step = int(man["step"])
        self.cursor = int(man["meta"]["cursor"])
        self._saved = (self.step, self.cursor)
        self._saved_info = info
        stored = jax.device_get(state.consts)
        same = jax.tree.structure(stored) == jax.tree.structure(self._consts_host) and all(
            np.array_equal(a, b) for a, b in zip(jax.tree.leaves(stored),
                                                 jax.tree.leaves(self._consts_host)))
        if not same:
            self._event("consts", note="checkpointed consts replaced by the current ones")
        self.state = state._replace(consts=jax.device_put(self._consts_host))

    def set_consts(self, consts: Any) -> None:
        """Replace the loss function's non-trainable inputs (e.g. the 0/1 switches of the fallback
        ladder) between steps without recompiling; structure, shapes and dtypes must not change.
        They hold for the rest of this process (also across rollbacks); a restarted process uses
        its own ``consts_fn``."""
        host = jax.device_get(consts)
        old = self._consts_host
        ok = jax.tree.structure(host) == jax.tree.structure(old) and all(
            np.shape(a) == np.shape(b) and np.asarray(a).dtype == np.asarray(b).dtype
            for a, b in zip(jax.tree.leaves(host), jax.tree.leaves(old)))
        if not ok:
            raise ValueError("consts must keep their structure, shapes and dtypes")
        self._consts_host = host
        self.state = self.state._replace(consts=jax.device_put(host))
        self._event("consts", note="set_consts")

    # --------------------------------------------------------------- checkpoints
    def _collect_writer(self, block: bool = True) -> None:
        """Record the background save once its write has finished: success counts the state as
        saved (``_saved``, ``ckpt`` event), a failure is logged (``ckpt_error``). With
        ``block=False`` it returns at once while the write is still running."""
        if self._bg is None:
            return
        if not block and not self.ckpt.idle():
            return
        info, saved, reason = self._bg
        self._bg = None
        try:
            self.ckpt.wait()
            read_manifest(info.path)          # complete on disk
        except CheckpointError as e:
            self._event("ckpt_error", step=saved[0], reason=f"{reason}: {e}")
            return
        self._record_save(info, saved, reason)

    def _record_save(self, info: CkptInfo, saved: tuple[int, int], reason: str) -> None:
        self._saved, self._saved_info = saved, info
        self._write_ledger(self.clock())
        job = self.ckpt.last_job
        extra = {k: job[k] for k in ("pruned", "retried") if job.get(k)}
        self._event("ckpt", step=saved[0], reason=reason, path=info.name, **extra)

    def _protect(self) -> list[int]:
        """Sequence numbers a save must not delete to make room: the checkpoint a rollback would
        restore now (for the current bad streak, or for one starting after this step)."""
        first_bad = self.guard.first_bad if not self.guard.healthy else self.step + 2
        target = self._trusted(first_bad)
        return [] if target is None else [target.seq]

    def _save(self, reason: str, *, fatal: bool = False) -> CkptInfo | None:
        """Checkpoint the current state (the caller has read its step's metrics).

        ``init``, ``final`` and ``rollback`` saves (and every save without ``async_ckpt``) are
        blocking and checked: the returned checkpoint is complete on disk. Other saves go to the
        background writer and are recorded when their write has succeeded
        (:meth:`_collect_writer`); ``None`` is returned for them. A failed write is logged
        (``ckpt_error``) and returns ``None``, or raises :class:`CheckpointError` if ``fatal``."""
        self._collect_writer()
        blocking = reason in BLOCKING_SAVES or not self.async_ckpt
        saved = (self.step, self.cursor)
        phase = self.ledger.phase
        now = self.clock()
        self.ledger.switch("ckpt", now)
        info, error = None, None
        try:
            meta = {"cursor": self.cursor, "ledger": self.ledger.snapshot(now).to_dict(),
                    "guard": self.guard.to_dict(), "regime": self._observed_regime,
                    "reason": reason, "deadline": self.deadline, "train_end": self.train_end}
            info = self.ckpt.save(self._ckpt_tree(), step=self.step, meta=meta, blocking=blocking,
                                  protect=self._protect())
            if blocking:
                read_manifest(info.path)      # checked: complete on disk
        except (CheckpointError, OSError, ValueError, TypeError) as e:
            info, error = None, e
        now = self.clock()
        self.ledger.switch(phase, now)
        self._last_ckpt = now
        if error is not None:
            self._event("ckpt_error", reason=f"{reason}: {error!r}")
            if fatal:
                raise CheckpointError(f"the {reason} checkpoint could not be written: "
                                      f"{error!r}") from error
            return None
        if not blocking:
            self._bg = (info, saved, reason)
            return None
        self._record_save(info, saved, reason)
        return info

    def _save_params_only(self) -> CkptInfo | None:
        """Blocking save of the parameters and consts alone (``{"state": {"params", "consts"}}``,
        readable by ``bes.decode.respond.load_checkpoint``) under :attr:`params_root`."""
        now = self.clock()
        meta = {"kind": "params", "reason": "final", "cursor": self.cursor,
                "ledger": self.ledger.snapshot(now).to_dict(), "guard": self.guard.to_dict(),
                "deadline": self.deadline, "train_end": self.train_end}
        tree = {"state": {"params": self.state.params, "consts": self.state.consts}}
        try:
            mgr = Checkpointer(self.params_root, self.config_hash, keep=1)
            info = mgr.save(tree, step=self.step, meta=meta, blocking=True, protect=())
            read_manifest(info.path)
        except (CheckpointError, OSError, ValueError, TypeError) as e:
            self._event("ckpt_error", reason=f"final (parameters only): {e!r}")
            return None
        self._event("ckpt", reason="final_params", path=f"{PARAMS_DIR}/{info.name}")
        return info

    def _final_save(self) -> tuple[CkptInfo, str]:
        """The final checkpoint, blocking and checked: the full state, retried once, then the
        parameters alone. Returns ``(info, kind)`` with kind ``"full"`` or ``"params"``; raises
        :class:`CheckpointError` when nothing could be written."""
        for attempt in (1, 2):
            info = self._save("final")
            if info is not None:
                return info, "full"
            self._event("alarm", reason=f"final checkpoint attempt {attempt} of 2 failed")
        info = self._save_params_only()
        if info is not None:
            self._event("alarm", reason=f"only the parameters of the final state were saved "
                                        f"({info.path}); the optimiser state is lost")
            return info, "params"
        self._event("alarm", reason="the final state could not be written at all")
        raise CheckpointError(f"training ended at step {self.step} but neither the final state nor "
                              f"its parameters could be written under {self.out_dir} (see the "
                              "ckpt_error events)")

    def _write_ledger(self, now: float) -> None:
        try:
            self.ledger.snapshot(now).save(self.out_dir / "ledger.json")
        except OSError as e:  # monitoring only; never fatal
            self._event("warning", reason=f"ledger.json not written: {e}")

    def _trusted(self, first_bad: int) -> CkptInfo | None:
        """The checkpoint a rollback for a bad streak starting at ``first_bad`` restores
        (:func:`bes.train.guard.rollback_target` over the checkpoints on disk), ``None`` if there
        is none."""
        self._collect_writer()
        ckpts = self.ckpt.list()
        verified = []
        for c in ckpts:
            try:
                if read_manifest(c.path)["meta"].get("reason") == "rollback":
                    verified.append(c.step)
            except CheckpointError:
                pass  # unreadable: restore() skips it anyway
        target = guard_lib.rollback_target([c.step for c in ckpts], first_bad, verified)
        if target is None:
            return None
        return max((c for c in ckpts if c.step == target), key=lambda c: c.seq)

    def _rollback_veto(self, target: CkptInfo | None) -> str | None:
        """Why a rollback to ``target`` keeps the current state instead, or ``None``: there is
        no checkpoint, or it holds more than ``guard_max_rollback_intervals`` checkpoint
        intervals of training less than now."""
        if target is None:
            return "there is no checkpoint to restore"
        run = self.cfg.run
        limit = run.guard_max_rollback_intervals * run.ckpt_every_min * 60.0
        if limit <= 0:
            return None
        try:
            kept = Ledger.from_dict(read_manifest(target.path)["meta"]["ledger"])
        except (CheckpointError, KeyError, TypeError, ValueError):
            return None                      # unreadable: restore() decides
        behind = self.ledger.train_seconds(self.clock()) - kept.train_seconds()
        if behind <= limit:
            return None
        return (f"the newest trusted checkpoint {target.name} is {behind:.0f} s of training "
                f"behind, more than {run.guard_max_rollback_intervals:g} checkpoint intervals "
                f"({limit:.0f} s)")

    def _restore_trusted(self, target: CkptInfo, *, count: bool, skip: int = 0) -> CkptInfo | None:
        """Restore ``target`` (or the newest loadable checkpoint before it), move the data cursor
        ``skip`` past the current one and fix the ledger; ``None`` if nothing could be
        restored. The live guard state is kept."""
        try:
            restored = self.ckpt.restore(self._ckpt_tree(), force=self.force, max_seq=target.seq)
        except CheckpointError as e:
            self._event("ckpt_error", reason=f"restore for the guard: {e}")
            return None
        if restored is None:
            return None
        tree, man, info = restored
        cursor_before = self.cursor
        self._install(tree, man, info)
        new_cursor = max(self.cursor, cursor_before + skip)
        skipped = new_cursor - self.cursor
        self.cursor = new_cursor
        kept = Ledger.from_dict(man["meta"]["ledger"])
        self.ledger = self.ledger.rolled_back(kept, self.clock(), data_skipped=skipped,
                                              count=count)
        self.ckpt.delete_newer(info.seq)
        return info

    def _act(self, decision: guard_lib.Decision) -> str | None:
        """Carry out a guard decision: ``None`` (go on), ``"rollback"`` (the state was replaced;
        steps in flight are void) or ``"halt"`` (training is over; the trusted state is kept)."""
        action = decision.action
        if action is guard_lib.Action.IGNORE:
            self._event("guard_exhausted", reason=decision.reason, first_bad=decision.first_bad)
        if action not in (guard_lib.Action.ROLLBACK, guard_lib.Action.HALT):
            return None
        self.ledger.switch("setup", self.clock())
        target = self._trusted(decision.first_bad)
        if action is guard_lib.Action.ROLLBACK:
            veto = self._rollback_veto(target)
            if veto is None:
                info = self._restore_trusted(target, count=True, skip=decision.skip)
                if info is not None:
                    self._event("rollback", reason=decision.reason, first_bad=decision.first_bad,
                                path=info.name, ckpt_step=info.step, cursor=self.cursor,
                                lr_mult=self.guard.lr_mult, rollbacks=self.guard.rollbacks)
                    if self._save("rollback") is None:  # makes the guard state and cursor durable
                        self._event("alarm", reason="the restored state could not be re-saved: "
                                                    "a crash now would replay the skipped data")
                    self.ledger.switch("train", self.clock())
                    return "rollback"
                veto = "no checkpoint could be restored"
            # keep the current state: the guard's learning-rate cut and the data skip apply, and
            # its average is re-based on the state kept (as when its rollbacks are exhausted)
            self.guard = guard_lib.rebase(self.guard)
            self.cursor += decision.skip
            self.ledger.data_skipped += int(decision.skip)
            self._event("alarm", kind="rollback_not_done", reason=(
                f"rollback for the {decision.reason} streak from step {decision.first_bad} not "
                f"done: {veto}; training goes on from the current state with the learning rate "
                f"x {self.guard.lr_mult:.4g} and {decision.skip} data steps skipped"),
                first_bad=decision.first_bad, lr_mult=self.guard.lr_mult,
                rollbacks=self.guard.rollbacks, cursor=self.cursor)
            self.ledger.switch("train", self.clock())
            return None
        info = None if target is None else self._restore_trusted(target, count=False)
        self._halt_restored = info is not None
        self._event("halt", reason=decision.reason, first_bad=decision.first_bad,
                    path=None if info is None else info.name,
                    ckpt_step=None if info is None else info.step)
        return "halt"

    # --------------------------------------------------------------------- loop
    def request_stop(self) -> None:
        """Ask the loop to stop after the current step (e.g. from a signal handler)."""
        self._stop_requested = True

    def _fetch(self, stage: schedule.Stage) -> dict:
        index = self.cursor
        batch = self._data(index, stage)
        self.cursor += 1
        out = {}
        for k, s in batch_spec(stage.shape, self.batch_keys).items():
            if k not in batch:
                raise KeyError(f"batch {index} lacks key {k!r}")
            arr = np.asarray(batch[k], dtype=s.dtype)
            if arr.shape != s.shape:
                raise ValueError(f"batch {index}: {k} has shape {arr.shape}; stage shape "
                                 f"{stage.shape} needs {s.shape}")
            out[k] = arr
        return jax.device_put(out)

    def _consume(self, info: StepInfo, metrics: Any) -> guard_lib.Decision:
        """Read a step's metrics (blocks until it is done) and pass its loss, and whether its
        update was skipped, to the guard."""
        m = jax.device_get(metrics)
        loss = float(m["loss"])
        skipped = int(m["skipped"]) != 0
        if info.regime != self._observed_regime:
            if self._observed_regime is not None:
                self.guard = guard_lib.rebase(self.guard)
            self._observed_regime = info.regime
        self.guard, decision = guard_lib.observe(self.guard, info.step, loss, self.guard_cfg,
                                                 skipped=skipped)
        self.history.append((info.step, info.p, info.lr, loss, info.stage.seq_len))
        if skipped:
            self.skipped_updates += 1
            self._event("skipped_update", step=info.step, loss=loss,
                        grad_norm=float(m["grad_norm"]), seq_len=info.stage.seq_len,
                        action=decision.action.value, reason=decision.reason,
                        total=self.skipped_updates)
        every = max(1, self.cfg.run.log_every)
        notable = decision.action is not guard_lib.Action.OK or skipped
        if info.step % every == 0 or notable:
            est = self._step_est
            self.log({"event": "step", "step": info.step, "p": info.p, "lr": info.lr,
                      "loss": loss, "loss_ema": self.guard.ema, "grad_norm": m["grad_norm"],
                      "skipped": m["skipped"], "weight": m["weight"],
                      "seq_len": info.stage.seq_len, "chat_frac": info.stage.chat_frac,
                      "assistant_only": info.stage.assistant_only, "phase": info.stage.phase,
                      "action": decision.action.value, "reason": decision.reason,
                      "step_s": est, "tok_per_s": info.stage.tokens_per_step / est if est else 0.0,
                      "time": info.time - self.t0,
                      "micro": jax.tree.map(lambda v: np.mean(np.asarray(v, np.float64), axis=0),
                                            m["micro"])})
            self._write_ledger(self.clock())
        return decision

    def _stop_reason(self, now: float) -> str | None:
        if self._stop_requested:
            return "requested"
        if self.max_steps is not None and self.step >= self.max_steps:
            return "max_steps"
        if now >= self.train_end or now + self._step_est > self.train_end:
            return "deadline"
        return None

    def _maybe_eval(self) -> None:
        if self.eval_fn is None:
            return
        now = self.clock()
        if now - self._last_eval < self.cfg.run.eval_every_min * 60.0:
            return
        self.ledger.switch("eval", now)
        try:
            result = self.eval_fn(self.state, self.step)
        except Exception as e:  # an evaluation failure never stops training
            result = {"error": repr(e)}
        now = self.clock()
        self.ledger.switch("train", now)
        self._last_eval = now
        self._event("eval", result=result or {})

    def run(self) -> dict:
        """Train until the window closes (or ``max_steps`` / a stop request / a guard halt), save
        the final state and return a summary. Raises :class:`CheckpointError` when the final
        state could not be written (see :meth:`_final_save`)."""
        if not self.ready:
            self.setup()
        clock = self.clock
        self.ledger.switch("train", clock())
        pending: tuple[StepInfo, Any] | None = None
        last_train = self.ledger.train_seconds(clock())
        reason: str | None = None
        while True:
            now = clock()
            reason = self._stop_reason(now)
            if reason is not None:
                break
            p = schedule.progress_of(self.ledger, now, self.train_end)
            stage = schedule.stage_at(p, self.cfg)
            regime = _regime_key(stage)
            if regime != self._dispatched_regime:
                self._event("stage", p=p, seq_len=stage.seq_len, phase=stage.phase,
                            assistant_only=stage.assistant_only, shape=list(stage.shape))
                self._dispatched_regime = regime
            batch = self._fetch(stage)
            lr = stage.lr * self.guard.lr_mult
            mom = schedule.momentum_at(self.step, p, self.cfg)
            wd = schedule.wd_scale_at(p, self.cfg)
            t = clock()
            train_now = self.ledger.train_seconds(t)
            dt, last_train = train_now - last_train, train_now
            self._step_est = dt if self._step_est == 0.0 else 0.8 * self._step_est + 0.2 * dt
            self.state, metrics = self.compiled[stage.shape](
                self.state, batch, np.float32(lr), np.float32(mom), np.float32(wd), self.group_lr)
            self.step += 1
            self.ledger.add_step(stage.tokens_per_step)
            info = StepInfo(self.step, p, lr, stage, regime, t)
            outcome = None
            if pending is not None:
                outcome = self._act(self._consume(*pending))
                pending = None
            if outcome is None:
                pending = (info, metrics)
                if clock() - self._last_ckpt >= self.cfg.run.ckpt_every_min * 60.0:
                    outcome = self._act(self._consume(*pending))  # read first: guard matches step
                    pending = None
                    if outcome is None and self.guard.healthy:
                        self._save("interval")
            if outcome == "halt":
                reason = "halt"
                break
            if outcome == "rollback":  # the step in flight belongs to the discarded trajectory
                self._dispatched_regime = None
                last_train = self.ledger.train_seconds(clock())
                continue
            self._collect_writer(block=False)
            self._maybe_eval()
        final, final_kind = None, None
        if pending is not None:  # the last step's loss decides whether its state is kept
            outcome = self._act(self._consume(*pending))
            pending = None
            if outcome == "halt":
                reason = "halt"
            elif outcome == "rollback":  # no time left to continue: keep the restored state
                self._event("final_rollback", ckpt_step=self.step)
        self._collect_writer()
        kept_by_halt = reason == "halt" and self._halt_restored
        if not kept_by_halt and self._saved != (self.step, self.cursor):
            final, final_kind = self._final_save()      # raises when nothing could be written
        holder = final if final is not None else self._saved_info
        now = clock()
        self.ledger.switch("done", now)
        self._write_ledger(now)
        p_end = schedule.progress_of(self.ledger, now, self.train_end)
        summary = {"reason": reason, "step": self.step, "cursor": self.cursor, "p_end": p_end,
                   "time": now, "train_end": self.train_end, "ledger": self.ledger.to_dict(),
                   "guard": self.guard.to_dict(), "final": None if final is None else final.name,
                   "final_kind": final_kind,
                   "checkpoint": None if holder is None else str(holder.path),
                   "skipped_updates": self.skipped_updates}
        self._event("end", reason=reason, p=p_end, checkpoint=summary["checkpoint"])
        return summary
