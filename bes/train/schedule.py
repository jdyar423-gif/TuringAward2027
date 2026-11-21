"""Wall-clock driven training schedule (SPEC section 5).

Progress
--------
The run has one absolute ``deadline`` (seconds since the epoch). Training must stop at
``train_end = deadline - post_reserve_hours * 3600``. With ``T`` the training seconds of the kept
trajectory (from :class:`bes.train.ledger.Ledger`) and ``R = max(0, train_end - now)``,

    p = T / (T + R)          (p = 1 once R = 0; p = 0 while T = 0 and R > 0).

While training, ``T`` grows and ``R`` shrinks at the same rate, so ``p`` is linear in wall time and
reaches 1 exactly when the window closes, whatever the throughput. Evaluation and checkpointing
leave ``T`` unchanged and shrink ``R``, so ``p`` still increases. After a restart, ``T`` is the
value stored with the restored checkpoint and ``R`` has shrunk by the lost time, so the restored
trajectory continues at a progress no smaller than the one recorded with its checkpoint, and the
schedule compresses itself into the time that is left.

Functions of p
--------------
Everything the trainer and the loader need is a function of ``p`` (plus the step count for the
momentum warm-up of ``"muon"``): learning-rate factor, sequence length (hence the compiled shape),
chat fraction, library fractions and the assistant-only loss switch. :func:`stage_at` bundles them
in a :class:`Stage`.

The learning-rate shape (linear warm-up, constant, linear warm-down to a final fraction) follows
``karpathy/nanochat`` ``scripts/base_train.py`` ``get_lr_multiplier`` (lines 359-369, MIT
licence); momentum and weight-decay schedules of the ``"muon"`` optimiser are ported from the same
file (``get_muon_momentum`` and ``get_weight_decay``, lines 371-386). In all three the reference's
fraction of iterations is replaced by ``p``.
``"muonh"`` keeps a constant momentum and has no weight decay, as in
``kschwethelm/looped-lm-scaling`` (``nanochat/gpt.py`` ``setup_optimizer``, MIT licence).
The learning-rate transfer rule of ``kschwethelm/looped-lm-scaling`` ``scripts/base_train.py``
(lines 341-367: batch factor ``(tokens_per_step / 2**18) ** 0.558`` on every group, data factor
``(tokens / tokens_ref) ** -0.32`` on the matrix group) is applied in ``bes.optim.build.group_lrs``
from ``OptimCfg``; the expected token horizon of a wall-clock run is estimated by the supervisor
from the start-up benchmark (:func:`estimate_horizon_tokens`) and fixed for the whole run.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from bes.config import Cfg, DataCfg, OptimCfg, RunCfg

HOUR = 3600.0


# ------------------------------------------------------------------------- clocks
class FakeClock:
    """Deterministic clock for tests and dry runs: ``clock()`` returns ``t``; ``advance`` moves it."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("a fake clock only moves forward")
        self.t += float(dt)
        return self.t

    def set(self, t: float) -> None:
        if t < self.t:
            raise ValueError("a fake clock only moves forward")
        self.t = float(t)


# ----------------------------------------------------------------------- progress
def train_end_time(deadline: float, run: RunCfg) -> float:
    """Wall-clock time at which training must be over: ``deadline - post_reserve``."""
    return float(deadline) - run.post_reserve_hours * HOUR


def progress(train_seconds: float, now: float, train_end: float) -> float:
    """``p = T / (T + max(0, train_end - now))`` in ``[0, 1]`` (see the module docstring)."""
    remaining = max(0.0, float(train_end) - float(now))
    if remaining <= 0.0:
        return 1.0
    t = max(0.0, float(train_seconds))
    if t <= 0.0:
        return 0.0
    return t / (t + remaining)


def progress_of(ledger: Any, now: float, train_end: float) -> float:
    """:func:`progress` with ``T = ledger.train_seconds(now)`` (open interval included)."""
    return progress(ledger.train_seconds(now), now, train_end)


def _clip01(p: float) -> float:
    return min(1.0, max(0.0, float(p)))


# -------------------------------------------------------------- learning rate
def estimate_horizon_tokens(tokens_per_second: float, train_seconds: float) -> float:
    """Expected training tokens of a wall-clock run: measured throughput x planned training time."""
    return max(0.0, float(tokens_per_second)) * max(0.0, float(train_seconds))


def with_horizon(cfg: Cfg, horizon_tokens: float) -> Cfg:
    """``cfg`` with ``optim.horizon_tokens`` set (the data factor of the matrix learning rate).

    Call it once, before the optimiser is built, and keep the value for the whole run (resumes
    included): the optimiser's learning rates are fixed at build time."""
    from dataclasses import replace
    return replace(cfg, optim=replace(cfg.optim, horizon_tokens=float(horizon_tokens)))


def lr_factor(p: float, o: OptimCfg) -> float:
    """Learning-rate factor: linear warm-up from 0 over ``[0, warmup_frac]``, 1, then linear
    warm-down over the last ``warmdown_frac`` of progress to ``final_lr_frac`` at ``p = 1``.
    Where warm-up and warm-down overlap the smaller factor wins."""
    p = _clip01(p)
    warm = 1.0 if o.warmup_frac <= 0 else min(1.0, p / o.warmup_frac)
    start = 1.0 - o.warmdown_frac
    if o.warmdown_frac <= 0 or p <= start:
        down = 1.0
    else:
        frac = (1.0 - p) / o.warmdown_frac
        down = o.final_lr_frac + (1.0 - o.final_lr_frac) * frac
    return min(warm, down)


def momentum_at(step: int, p: float, cfg: Cfg) -> float:
    """Momentum of the matrix optimiser for update number ``step`` (0-based) at progress ``p``.

    ``"muonh"``: constant ``optim.momentum``. ``"muon"`` (nanochat ``get_muon_momentum``): linear
    from ``muon_momentum_start`` to ``muon_momentum_peak`` over the first
    ``muon_momentum_warmup_steps`` updates, then the peak, then linear down to
    ``muon_momentum_end`` over the learning-rate warm-down (progress instead of iterations).
    """
    o = cfg.optim
    if o.optimizer != "muon":
        return float(o.momentum)
    peak = o.muon_momentum_peak
    n = o.muon_momentum_warmup_steps
    if n > 0 and step < n:
        frac = step / n
        return (1.0 - frac) * o.muon_momentum_start + frac * peak
    p = _clip01(p)
    start = 1.0 - o.warmdown_frac
    if o.warmdown_frac > 0 and p >= start:
        frac = (p - start) / o.warmdown_frac
        return peak * (1.0 - frac) + o.muon_momentum_end * frac
    return float(peak)


def wd_scale_at(p: float, cfg: Cfg) -> float:
    """Factor on ``optim.weight_decay``: cosine from 1 to 0 over progress for ``"muon"``
    (nanochat ``get_weight_decay``); 1 otherwise (MuonH has no weight decay)."""
    if cfg.optim.optimizer != "muon":
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * _clip01(p)))


# ---------------------------------------------------------------------- data
def seq_len_at(p: float, d: DataCfg) -> int:
    """``seq_len`` before ``late_from``, ``seq_len_late`` from it on."""
    return d.seq_len_late if p >= d.late_from else d.seq_len


def chat_frac_at(p: float, d: DataCfg) -> float:
    """Fraction of tokens from chat sources: ``chat_frac`` until ``late_from``, then a linear ramp
    to ``chat_frac_tail`` at ``tail_from`` (the long-context phase gradually adds chat), then
    ``chat_frac_tail`` (the chat-heavy final phase). If ``late_from >= tail_from`` the change is a
    step at ``tail_from``."""
    lo, hi = d.chat_frac, d.chat_frac_tail
    if p >= d.tail_from:
        return float(hi)
    if p < d.late_from or d.late_from >= d.tail_from:
        return float(lo)
    frac = (p - d.late_from) / (d.tail_from - d.late_from)
    return float(lo + (hi - lo) * frac)


def library_fracs_at(p: float, d: DataCfg) -> tuple[float, float]:
    """``(base, chat)`` fractions of sequences that carry a retrieved library passage; constant in
    ``p`` and zero without the library. The loader keeps them at zero until the library is built."""
    del p
    if not d.use_library:
        return 0.0, 0.0
    return float(d.library_frac_base), float(d.library_frac_chat)


def assistant_only_at(p: float, d: DataCfg) -> bool:
    """From ``assistant_only_from`` on the loss covers assistant tokens only."""
    return p >= d.assistant_only_from


def data_phase_at(p: float, d: DataCfg) -> str:
    """``"short"`` (``seq_len``), ``"long"`` (``seq_len_late``) or ``"tail"`` (chat-heavy)."""
    if p >= d.tail_from:
        return "tail"
    return "long" if p >= d.late_from else "short"


# --------------------------------------------------------------------- shapes
def micro_shape(cfg: Cfg, seq_len: int) -> tuple[int, int, int]:
    """Compiled step shape ``(A, B, T)`` for sequence length ``T``.

    Every micro-batch holds ``micro_seqs * data.seq_len`` token positions (``B = micro_seqs`` at
    the base length, fewer, longer rows at ``seq_len_late``); ``A`` micro-batches make
    ``tokens_per_step``. The final configuration gives ``(8, 32, 1024)`` and ``(8, 16, 2048)``.
    """
    o, d = cfg.optim, cfg.data
    per_micro = o.micro_seqs * d.seq_len
    if seq_len <= 0 or per_micro % seq_len:
        raise ValueError(f"micro-batch of {per_micro} tokens is not a multiple of seq_len {seq_len}")
    if o.tokens_per_step % per_micro:
        raise ValueError(f"tokens_per_step {o.tokens_per_step} is not a multiple of {per_micro}")
    return o.tokens_per_step // per_micro, per_micro // seq_len, int(seq_len)


def shapes(cfg: Cfg) -> tuple[tuple[int, int, int], ...]:
    """Distinct ``(A, B, T)`` the run can use, in order of use (compile all of them up front)."""
    d = cfg.data
    lens = []
    if d.late_from > 0:
        lens.append(d.seq_len)
    if d.late_from <= 1:
        lens.append(d.seq_len_late)
    out: list[tuple[int, int, int]] = []
    for t in lens:
        s = micro_shape(cfg, t)
        if s not in out:
            out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------- stage
@dataclass(frozen=True)
class Stage:
    """What the trainer and the loader do at progress ``p``.

    lr: learning-rate factor (before the guard's multiplier). seq_len / micro_seqs / accum: the
    compiled shape ``(A, B, T) = (accum, micro_seqs, seq_len)``. chat_frac: fraction of tokens
    from chat sources. library_frac_base / library_frac_chat: fractions of base / chat sequences
    that carry a retrieved passage. assistant_only: loss on assistant tokens only. phase:
    ``"short" | "long" | "tail"``.
    """

    p: float
    lr: float
    seq_len: int
    micro_seqs: int
    accum: int
    chat_frac: float
    library_frac_base: float
    library_frac_chat: float
    assistant_only: bool
    phase: str

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(A, B, T)``: key of the compiled step."""
        return self.accum, self.micro_seqs, self.seq_len

    @property
    def tokens_per_step(self) -> int:
        return self.accum * self.micro_seqs * self.seq_len

    @property
    def regime(self) -> tuple[Any, ...]:
        """The discrete part of the stage; a change means a new loss level (guard re-baselines)."""
        return self.shape, self.assistant_only, self.phase

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage_at(p: float, cfg: Cfg) -> Stage:
    """The :class:`Stage` at progress ``p`` (clipped to ``[0, 1]``)."""
    p = _clip01(p)
    d = cfg.data
    a, b, t = micro_shape(cfg, seq_len_at(p, d))
    base, chat = library_fracs_at(p, d)
    return Stage(p=p, lr=lr_factor(p, cfg.optim), seq_len=t, micro_seqs=b, accum=a,
                 chat_frac=chat_frac_at(p, d), library_frac_base=base, library_frac_chat=chat,
                 assistant_only=assistant_only_at(p, d), phase=data_phase_at(p, d))
