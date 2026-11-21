"""Loss-spike and NaN guard as a pure state machine (SPEC section 5).

The trainer feeds every observed training loss to :func:`observe` (one step late, in step order)
and acts on the returned :class:`Decision`:

``OK``        the loss is good; it is folded into the moving average.
``SUSPECT``   the loss is bad (non-finite, or above ``spike * average`` once the average has seen
              ``warmup`` good losses) but the streak of bad losses is still short; nothing to do.
``ROLLBACK``  the streak reached ``patience`` consecutive bad losses (``nonfinite_patience`` if the
              current loss is non-finite): restore the newest checkpoint older than the streak
              (:func:`rollback_target`), skip ``skip`` steps of data and multiply the learning rate
              by ``lr_factor`` (the returned state already carries the new multiplier).
``IGNORE``    a finite streak triggered after ``max_rollbacks`` rollbacks: keep training; the
              average is re-based on the new losses.
``HALT``      a non-finite streak triggered after ``max_rollbacks`` rollbacks: stop and keep the
              last good checkpoint.

The moving average is the arithmetic mean of the first ``1 / (1 - ema)`` good losses since the
last re-baseline and an exponential moving average afterwards (so a fresh baseline never rests
on a single sample). Bad losses never enter it, so a slow upward drift cannot hide a spike. The train
step skips updates whose loss or gradient is non-finite, so a single bad batch leaves the state
intact; therefore the default ``nonfinite_patience`` equals ``patience`` (set it to 1 for an
immediate rollback on any non-finite loss).

Step convention: the loss of step ``k`` is computed with the parameters after update ``k - 1``.
A checkpoint taken after update ``s`` is trusted once the loss of step ``s + 1`` was good, so the
rollback target for a streak starting at step ``k1`` is the newest checkpoint with
``s <= k1 - 2``; the state re-saved at a rollback stays trusted (see :func:`rollback_target`).
"""
from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

from bes.config import RunCfg


@dataclass(frozen=True)
class GuardCfg:
    """Thresholds of the guard (defaults from ``RunCfg``)."""

    spike: float = 1.3
    patience: int = 3
    nonfinite_patience: int = 3
    ema: float = 0.99
    warmup: int = 100
    skip_steps: int = 200
    lr_factor: float = 0.7
    max_rollbacks: int = 3

    def __post_init__(self) -> None:
        if self.spike <= 1.0:
            raise ValueError("spike factor must be > 1")
        if self.patience < 1 or self.nonfinite_patience < 1:
            raise ValueError("patience must be >= 1")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        if self.warmup < 1 or self.skip_steps < 0 or self.max_rollbacks < 0:
            raise ValueError("warmup >= 1, skip_steps >= 0 and max_rollbacks >= 0 required")
        if not 0.0 < self.lr_factor <= 1.0:
            raise ValueError("lr_factor must be in (0, 1]")

    @classmethod
    def from_run(cls, run: RunCfg) -> "GuardCfg":
        return cls(spike=run.guard_spike, patience=run.guard_patience,
                   nonfinite_patience=run.guard_nonfinite_patience, ema=run.guard_ema,
                   warmup=run.guard_warmup, skip_steps=run.guard_skip_steps,
                   lr_factor=run.guard_lr_factor, max_rollbacks=run.guard_max_rollbacks)


@dataclass(frozen=True)
class GuardState:
    """ema: moving average of good losses (meaningful when ``count > 0``); count: good losses
    folded in since the last re-baseline; streak: consecutive bad losses; first_bad: step of the
    first loss of the streak (-1 without a streak); rollbacks: rollbacks so far; lr_mult: product
    of the rollback learning-rate factors."""

    ema: float = 0.0
    count: int = 0
    streak: int = 0
    first_bad: int = -1
    rollbacks: int = 0
    lr_mult: float = 1.0

    @property
    def healthy(self) -> bool:
        """No bad loss pending: a checkpoint taken now is a rollback candidate."""
        return self.streak == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GuardState":
        return cls(ema=float(d["ema"]), count=int(d["count"]), streak=int(d["streak"]),
                   first_bad=int(d["first_bad"]), rollbacks=int(d["rollbacks"]),
                   lr_mult=float(d["lr_mult"]))


class Action(str, enum.Enum):
    OK = "ok"
    SUSPECT = "suspect"
    ROLLBACK = "rollback"
    IGNORE = "ignore"
    HALT = "halt"


@dataclass(frozen=True)
class Decision:
    """action; reason (``""``, ``"spike"`` or ``"nonfinite"``); first_bad: first step of the streak
    (for ROLLBACK / IGNORE / HALT, else -1); skip: data steps to skip (ROLLBACK); lr_mult: the
    learning-rate multiplier from now on; threshold: ``spike * ema`` used for the judgement
    (``inf`` while warming up)."""

    action: Action
    reason: str = ""
    first_bad: int = -1
    skip: int = 0
    lr_mult: float = 1.0
    threshold: float = math.inf


def _fold(state: GuardState, loss: float, cfg: GuardCfg) -> GuardState:
    # Arithmetic mean of the first 1/(1-ema) good losses, exponential afterwards: a fresh
    # baseline (start, stage switch) is an average of several losses, never a single sample.
    rate = max(1.0 - cfg.ema, 1.0 / (state.count + 1))
    ema = state.ema + rate * (loss - state.ema)
    return replace(state, ema=ema, count=state.count + 1, streak=0, first_bad=-1)


def observe(state: GuardState, step: int, loss: float,
            cfg: GuardCfg) -> tuple[GuardState, Decision]:
    """Judge the training loss of ``step`` (a positive loss such as a cross-entropy); returns the
    new state and what to do. A streak triggers when its length reaches the patience of the
    current loss's kind (``patience`` if finite, ``nonfinite_patience`` if not)."""
    loss = float(loss)
    finite = math.isfinite(loss)
    threshold = cfg.spike * state.ema if state.count >= cfg.warmup else math.inf
    if finite and loss <= threshold:
        new = _fold(state, loss, cfg)
        return new, Decision(Action.OK, lr_mult=new.lr_mult, threshold=threshold)
    reason = "spike" if finite else "nonfinite"
    first_bad = step if state.streak == 0 else state.first_bad
    streak = state.streak + 1
    limit = cfg.patience if finite else cfg.nonfinite_patience
    if streak < limit:
        new = replace(state, streak=streak, first_bad=first_bad)
        return new, Decision(Action.SUSPECT, reason, lr_mult=new.lr_mult, threshold=threshold)
    if state.rollbacks < cfg.max_rollbacks:
        new = replace(state, streak=0, first_bad=-1, rollbacks=state.rollbacks + 1,
                      lr_mult=state.lr_mult * cfg.lr_factor)
        return new, Decision(Action.ROLLBACK, reason, first_bad=first_bad, skip=cfg.skip_steps,
                             lr_mult=new.lr_mult, threshold=threshold)
    if finite:
        # Out of rollbacks: accept the new level and re-base the average on it.
        new = _fold(replace(state, count=0), loss, cfg)
        return new, Decision(Action.IGNORE, reason, first_bad=first_bad, lr_mult=new.lr_mult,
                             threshold=threshold)
    new = replace(state, streak=streak, first_bad=first_bad)
    return new, Decision(Action.HALT, reason, first_bad=first_bad, lr_mult=new.lr_mult,
                         threshold=threshold)


def rebase(state: GuardState) -> GuardState:
    """Forget the moving average (new loss level after a stage switch); keeps the counters."""
    return replace(state, ema=0.0, count=0, streak=0, first_bad=-1)


def rollback_target(saved_steps: Iterable[int], first_bad: int,
                    verified_steps: Iterable[int] = ()) -> int | None:
    """Step of the checkpoint to restore for a streak that started at ``first_bad``.

    Trusted are the checkpoints with ``s <= first_bad - 2`` (their parameters produced a good loss)
    and those in ``verified_steps`` (states re-saved at a previous rollback: they were trusted when
    chosen, so a bad streak right after restoring one is blamed on the data, as in a retry from the
    same checkpoint with more data skipped). Returns the newest trusted step, else the oldest
    available one (e.g. the initial checkpoint when the very first steps fail), else ``None``."""
    steps = sorted(set(int(s) for s in saved_steps))
    if not steps:
        return None
    verified = set(int(s) for s in verified_steps)
    trusted = [s for s in steps if s <= first_bad - 2 or s in verified]
    return trusted[-1] if trusted else steps[0]
