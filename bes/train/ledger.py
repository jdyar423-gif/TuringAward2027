"""Persistent time ledger of a training run (SPEC section 5).

The ledger attributes every wall-clock second since the start of the run ``t0`` to exactly one
phase: ``setup`` (downloads, compilation, restoring), ``train`` (the step loop), ``eval``,
``ckpt`` (writing checkpoints) or ``lost`` (time whose work was discarded by a restart or a
rollback, including the downtime of a crashed process). Time is attributed lazily: the interval
``[mark, now)`` belongs to the current ``phase`` and is closed by :meth:`Ledger.switch`. Hence

    sum(seconds.values()) + (now - mark) == now - t0

as long as the clock never goes backwards (a backwards jump is attributed to nothing).

The ledger also counts the optimiser steps and the token positions of the *kept* trajectory
(``steps``, ``tokens``) and the restart / rollback events. A checkpoint stores a snapshot of the
ledger; restoring it (:meth:`Ledger.resumed`) keeps the snapshot's training seconds and books
everything after the snapshot as ``lost``. The schedule computes the progress ``p`` from
``train_seconds`` of this ledger, so ``p`` survives restarts.

Pure Python (no JAX), JSON-serialisable; floats round-trip exactly through ``json``.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Phases every ledger reports (other names are accepted and kept).
PHASES: tuple[str, ...] = ("setup", "train", "eval", "ckpt", "lost")
FORMAT = 1


def fsync_dir(path: str | os.PathLike) -> None:
    """``fsync`` a directory so that a rename inside it is durable (no-op where unsupported)."""
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temporary file in the same directory, fsync, rename,
    fsync of the directory. Readers see either the old or the new content, never a mix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    """:func:`atomic_write_bytes` of ``json.dumps(obj)`` (sorted keys, finite numbers only)."""
    atomic_write_bytes(path, json.dumps(obj, sort_keys=True, indent=1, allow_nan=False).encode())


def _zero_phases() -> dict[str, float]:
    return dict.fromkeys(PHASES, 0.0)


@dataclass
class Ledger:
    """Wall-clock accounting of one run.

    t0: wall-clock start of the run (seconds since the epoch). mark: wall-clock time up to which
    time has been attributed. phase: phase of the open interval ``[mark, now)``. seconds: closed
    seconds per phase. steps / tokens: optimiser steps and token positions of the kept
    trajectory. restarts / rollbacks: number of resumptions from a checkpoint after a crash /
    after a guard rollback. data_skipped: data indices the kept trajectory never trained on
    (discarded work plus the guard's deliberate skips).
    """

    t0: float
    mark: float
    phase: str = "setup"
    seconds: dict[str, float] = field(default_factory=_zero_phases)
    steps: int = 0
    tokens: int = 0
    restarts: int = 0
    rollbacks: int = 0
    data_skipped: int = 0

    def __post_init__(self) -> None:
        for name in PHASES:
            self.seconds.setdefault(name, 0.0)
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError(f"invalid phase {self.phase!r}")

    # ------------------------------------------------------------------ construction
    @classmethod
    def start(cls, t0: float, now: float | None = None, phase: str = "setup") -> "Ledger":
        """A new ledger for a run that started at ``t0``. If ``now`` is given, ``[t0, now)`` is
        booked as ``setup`` and the open interval starts at ``now`` in ``phase``."""
        led = cls(t0=float(t0), mark=float(t0), phase="setup")
        if now is not None:
            led.switch(phase, now)
        return led

    # ---------------------------------------------------------------------- queries
    def open_seconds(self, now: float) -> float:
        """Length of the open interval ``[mark, now)`` (0 if the clock went backwards)."""
        return max(0.0, float(now) - self.mark)

    def seconds_in(self, phase: str, now: float | None = None) -> float:
        """Seconds booked to ``phase``, including the open interval if it is that phase."""
        closed = self.seconds.get(phase, 0.0)
        if now is not None and phase == self.phase:
            closed += self.open_seconds(now)
        return closed

    def train_seconds(self, now: float | None = None) -> float:
        """Training seconds of the kept trajectory (the ``T`` of the progress formula)."""
        return self.seconds_in("train", now)

    def accounted(self, now: float | None = None) -> float:
        """All booked seconds (plus the open interval if ``now`` is given)."""
        total = sum(self.seconds.values())
        return total + (self.open_seconds(now) if now is not None else 0.0)

    # -------------------------------------------------------------------- mutation
    def switch(self, phase: str, now: float) -> None:
        """Close the open interval at ``now`` (booked to the current phase) and open ``phase``."""
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"invalid phase {phase!r}")
        self.seconds[self.phase] = self.seconds.get(self.phase, 0.0) + self.open_seconds(now)
        self.mark = max(self.mark, float(now))
        self.phase = phase

    def close(self, now: float) -> None:
        """Book the open interval up to ``now`` without changing the phase."""
        self.switch(self.phase, now)

    def add_step(self, tokens: int, steps: int = 1) -> None:
        """Count ``steps`` optimiser steps that processed ``tokens`` token positions."""
        self.steps += int(steps)
        self.tokens += int(tokens)

    def snapshot(self, now: float) -> "Ledger":
        """A copy whose open interval is closed at ``now`` (what a checkpoint stores)."""
        led = self.copy()
        led.close(now)
        return led

    def resumed(self, now: float, *, rollback: bool = False, data_skipped: int = 0) -> "Ledger":
        """The ledger of a state restored from a checkpoint whose snapshot is ``self``.

        Everything after the snapshot's ``mark`` up to ``now`` is booked as ``lost`` (the work
        done in it was not kept, and a crashed process's downtime is in it too); the restart or
        rollback is counted; the open phase becomes ``setup``. ``self`` is not modified.
        """
        led = self.copy()
        led.seconds["lost"] = led.seconds.get("lost", 0.0) + led.open_seconds(now)
        led.mark = max(led.mark, float(now))
        led.phase = "setup"
        if rollback:
            led.rollbacks += 1
        else:
            led.restarts += 1
        led.data_skipped += int(data_skipped)
        return led

    def rolled_back(self, kept: "Ledger", now: float, *, data_skipped: int = 0,
                    count: bool = True) -> "Ledger":
        """The live ledger ``self`` after an in-process rollback to a checkpoint whose snapshot
        is ``kept``: the training seconds after the snapshot move to ``lost``, steps and tokens
        revert to the snapshot's, the other phases and the event counters stay as they are, the
        rollback is counted (unless ``count`` is false) and the open phase becomes ``setup``.
        ``self`` is not modified."""
        led = self.copy()
        led.close(now)
        discarded = max(0.0, led.seconds.get("train", 0.0) - kept.seconds.get("train", 0.0))
        led.seconds["train"] = kept.seconds.get("train", 0.0)
        led.seconds["lost"] = led.seconds.get("lost", 0.0) + discarded
        led.steps, led.tokens = kept.steps, kept.tokens
        led.rollbacks += int(count)
        led.data_skipped += int(data_skipped)
        led.phase = "setup"
        return led

    def copy(self) -> "Ledger":
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["format"] = FORMAT
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ledger":
        d = dict(d)
        fmt = d.pop("format", FORMAT)
        if fmt != FORMAT:
            raise ValueError(f"unsupported ledger format {fmt}")
        d["seconds"] = {str(k): float(v) for k, v in d.get("seconds", {}).items()}
        for name in ("t0", "mark"):
            d[name] = float(d[name])
        for name in ("steps", "tokens", "restarts", "rollbacks", "data_skipped"):
            d[name] = int(d.get(name, 0))
        return cls(**d)

    def save(self, path: str | os.PathLike) -> None:
        """Atomic JSON write (see :func:`atomic_write_json`)."""
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Ledger":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self, now: float | None = None) -> dict[str, float]:
        """Seconds per phase with the open interval included (for logs)."""
        out = {k: round(v, 3) for k, v in self.seconds.items()}
        if now is not None:
            out[self.phase] = round(self.seconds_in(self.phase, now), 3)
        return out
