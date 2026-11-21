"""The data mixture as token shares, with epoch caps, and the per-sample transform policy
(``docs/PLAN.md`` section 5; stages from ``bes.train.schedule.stage_at``).

Pure functions of explicit inputs: the loader calls :func:`plan` when it plans (start-up, a new
stage, more data on disk) and logs the returned :class:`Mixture`.

Kinds and their targets
-----------------------
Every probability here is a **token share** (rows are packed full, so a row share is a token
share too). With ``c = stage.chat_frac``, ``L = stage.library_frac_base`` when a library is
available (else 0) and ``mu`` the sub-kind weights of :class:`~bes.config.MixCfg`
(``chat_weight, rag_weight, math_weight, library_weight``) renormalised over the sub-kinds
that have data on disk:

===========  =========================  ===================================================
kind         target share               sources (weights within the kind)
===========  =========================  ===================================================
``base``     ``(1 - c)(1 - L)``         ``sources.mixture_weights("base", permissive_only)``
``library``  ``(1 - c) L + c mu_lib``   ``library_read`` / ``library_qa`` (``MixCfg.read_weight``
                                        / ``qa_weight``): the tasks of ``bes.data.rag_format``
``chat``     ``c mu_chat``              ``sources.mixture_weights("chat", ...)``
``rag``      ``c mu_rag``               ``sources.mixture_weights("rag", ...)``
``math``     ``c mu_math``              ``sources.mixture_weights("math", ...)``
===========  =========================  ===================================================

Sources without data on disk are skipped and the weights of their kind renormalised (they are
listed in ``Mixture.missing``). Rows: *base rows* carry the ``base`` kind, *library rows* the
part ``(1 - c) L`` of the library kind (the library share of the base rows), *chat rows*
everything else; ``Mixture.chat_rows`` gives the sub-kind shares inside chat rows.

Epoch caps
----------
``available[s]`` is the number of tokens of source ``s`` on disk (for a library task: one epoch
of tasks, :func:`library_available`) and ``horizon_tokens`` the number of tokens the mixture is
planned for. A source may take at most ``cap_s = max_epochs * available[s] / horizon_tokens``
of the tokens. The shares are found by water-filling (:func:`water_fill`) on two levels:

1. kinds: ``x_k = min(K_k, lam * target_k)`` with ``K_k`` the sum of the caps of the kind's
   sources and ``lam`` such that the shares sum to 1. A kind whose sources are all at their
   caps passes the rest of its target to the other kinds in proportion to their targets
   (``Mixture.moved``);
2. sources inside each kind: ``y_s = min(cap_s, lam_k * w_s)`` summing to ``x_k``: the excess
   of a capped source goes to the sources of the same kind that still have room
   (``Mixture.capped``, ``target`` versus ``share``).

When all the data together cannot fill the horizon at ``max_epochs``, the cap is raised for
every source by the same factor (``Mixture.epoch_limit``, with a note). Passing the run's total
token horizon at every stage keeps every source within its cap over the whole run (its share
is bounded at every stage), at the price of being conservative for sources whose share changes
with the stage; while downloads are still running, pass the tokens until the next re-plan.

Transform policy
----------------
:func:`transform_probs`: hindsight relabelling (``samples.apply_hindsight``) with probability
``DataCfg.hindsight_frac`` on ``chat`` samples; retrieval insertion
(``rag_format.insert_retrieval``) with probability ``stage.library_frac_chat`` on ``chat`` and
``rag`` samples when a library is available (``rag`` samples that already bring a passage pass
through unchanged); nothing on ``math``, ``library`` and ``base`` samples. :func:`apply_transforms`
runs them in that order (hindsight first, so that the retrieval sees the final user message, as
at inference).

Intended use (one row of ``T + 1`` tokens at stage ``st``; all draws from
``samples.rng_for(seed, <explicit keys>)``)
--------------------------------------------------------------------------------------------
::

    mx = plan(st, available, horizon)              # once per stage and when data arrives
    rule = WeightRule.at(st, data)                 # bes.data.samples
    row = <draw from mx.rows>                      # "base" | "library" | "chat"
    base:    s = from_tokens(seg, None, rule, source=src, kind="base")   # src ~ mx.sources["base"]
             head, rest = fit(s, room)             # head fills the row, rest opens the next one
    library: task ~ mx.sources["library"]; pid = rag_format.pick_entry(lib, task, counter, seed)
             s = rag_format.library_task(task, pid, lib, tok, rng, rule, mix)   # None: next pid
    chat:    sub ~ mx.chat_rows (token shares: draw segments with segment_probs(...) and mean
             lengths, or schedule by token deficit); src ~ mx.sources[sub] (library: as above)
             s = from_tokens(seg, roles, rule, source=src, kind=sub); s, _ = fit(s, T + 1)
             s = apply_transforms(s, draw_transforms(sub, st, rng), tok, rng, lib=lib, max_len=T + 1)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import DataCfg, MixCfg
from bes.data import sources as registry
from bes.data.rag_format import LIBRARY_TASKS, RagStats, as_view, insert_retrieval
from bes.data.samples import Sample, apply_hindsight

KIND_ORDER = ("base", "library", "chat", "rag", "math")
CHAT_SUBKINDS = ("chat", "rag", "math", "library")
ROW_KINDS = ("base", "library", "chat")
TRANSFORMS = ("hindsight", "retrieval")


# --------------------------------------------------------------------- water-filling
def water_fill(weights: Sequence[float], caps: Sequence[float], total: float) -> tuple[np.ndarray, float]:
    """Split ``total`` in proportion to ``weights`` without exceeding ``caps``.

    Returns ``(x float64[n], unplaced)`` with ``x_i = min(cap_i, lam * w_i)`` for one ``lam``
    and ``sum(x) = total - unplaced``; ``unplaced > 0`` only when the items with positive weight
    cannot hold ``total`` (they are then all at their caps). Items with weight 0 get 0.
    Exact: items are saturated in order of ``cap_i / w_i``.
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(caps, dtype=np.float64)
    if w.shape != c.shape or w.ndim != 1:
        raise ValueError("weights and caps must be 1-d of the same length")
    if (w < 0).any() or (c < 0).any() or not (np.isfinite(w).all() and np.isfinite(c).all()):
        raise ValueError("weights and caps must be finite and non-negative")
    x = np.zeros_like(w)
    active = np.flatnonzero(w > 0)
    if total <= 0 or active.size == 0:
        return x, max(0.0, float(total))
    room = float(c[active].sum())
    if room <= total:
        x[active] = c[active]
        return x, float(total) - room
    order = active[np.argsort(c[active] / w[active], kind="stable")]
    remaining, wsum, k = float(total), float(w[order].sum()), 0
    while k < order.size and c[order[k]] <= remaining * w[order[k]] / wsum:
        i = order[k]
        x[i] = c[i]
        remaining -= c[i]
        wsum -= w[i]
        k += 1
    x[order[k:]] = remaining * w[order[k:]] / wsum
    return x, 0.0


# --------------------------------------------------------------------------- mixture
@dataclass(frozen=True)
class Mixture:
    """Effective token shares of one plan (module docstring).

    rows: ``{"base", "library", "chat"}`` -> share of rows (sum 1). chat_rows: sub-kind
    (``chat, rag, math, library``) -> share inside chat rows (sum 1; empty without chat rows).
    kinds: kind -> share of all tokens (sum 1). sources: kind -> {source: share inside the kind}
    (sum 1 for every kind with a positive share). share / target: source -> share of all tokens
    after / before the epoch caps. epochs: source -> planned reads of its data over
    ``horizon_tokens``. capped: sources held at the epoch limit. kind_target / moved: kind ->
    target share and the change the kind-level water-filling made. missing: sources with a
    weight but no data on disk. epoch_limit: the cap applied (``max_epochs`` unless all data
    together is too small). notes: human-readable remarks for the log.
    """

    rows: dict[str, float]
    chat_rows: dict[str, float]
    kinds: dict[str, float]
    sources: dict[str, dict[str, float]]
    share: dict[str, float]
    target: dict[str, float]
    epochs: dict[str, float]
    capped: tuple[str, ...]
    kind_target: dict[str, float]
    moved: dict[str, float]
    missing: tuple[str, ...]
    epoch_limit: float
    horizon_tokens: float
    notes: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source_probs(self, kind: str) -> dict[str, float]:
        """Token shares of the sources inside ``kind`` (empty when the kind has no share)."""
        return dict(self.sources.get(kind, {}))


def kind_weights(kind: str, data: DataCfg = DataCfg(), mix: MixCfg = MixCfg()) -> dict[str, float]:
    """Normalised default weights of the sources of ``kind`` (``library``: the two tasks)."""
    if kind == "library":
        raw = {"library_read": mix.read_weight, "library_qa": mix.qa_weight}
        raw = {k: float(v) for k, v in raw.items() if v > 0}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()} if total > 0 else {}
    return registry.mixture_weights(kind, data.permissive_only)


def _normalise(d: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(d.values()))
    return {k: float(v) / total for k, v in d.items()} if total > 0 else {}


def plan(stage: Any, available: Mapping[str, float], horizon_tokens: float, *,
         data: DataCfg = DataCfg(), mix: MixCfg = MixCfg(), library: bool | None = None) -> Mixture:
    """The :class:`Mixture` at ``stage`` (``chat_frac``, ``library_frac_base``) for the data on
    disk (``available``: tokens per source name, library tasks included) planned over
    ``horizon_tokens`` tokens (module docstring). ``library``: whether a library can be used
    (``None``: when a library task has tokens in ``available``; never with
    ``data.use_library = False``). Raises ``ValueError`` when no kind with a positive target has
    data."""
    if horizon_tokens <= 0:
        raise ValueError("horizon_tokens must be positive")
    avail = {str(k): max(0.0, float(v)) for k, v in available.items()}
    has_lib = any(avail.get(t, 0.0) > 0 for t in LIBRARY_TASKS) if library is None else bool(library)
    has_lib = has_lib and data.use_library
    notes: list[str] = []

    weights: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for kind in KIND_ORDER:
        w = kind_weights(kind, data, mix) if (kind != "library" or has_lib) else {}
        missing += [s for s in w if avail.get(s, 0.0) <= 0]
        weights[kind] = _normalise({s: x for s, x in w.items() if avail.get(s, 0.0) > 0})

    c = min(1.0, max(0.0, float(stage.chat_frac)))
    mu = _normalise({k: v for k, v in (("chat", mix.chat_weight), ("rag", mix.rag_weight),
                                       ("math", mix.math_weight), ("library", mix.library_weight))
                     if v > 0 and weights[k]})
    if c > 0 and not mu:
        notes.append(f"no conversation data on disk: chat_frac {c:.3f} -> 0")
        c = 0.0
    lib_base = float(stage.library_frac_base) if weights["library"] else 0.0
    lib_base = min(1.0, max(0.0, lib_base))
    targets = {"base": (1 - c) * (1 - lib_base), "library": (1 - c) * lib_base + c * mu.get("library", 0.0),
               "chat": c * mu.get("chat", 0.0), "rag": c * mu.get("rag", 0.0), "math": c * mu.get("math", 0.0)}
    lib_rows_part = (1 - c) * lib_base

    cap_src = {s: mix.max_epochs * avail[s] / horizon_tokens for k in KIND_ORDER for s in weights[k]}
    cap_kind = np.array([sum(cap_src[s] for s in weights[k]) for k in KIND_ORDER])
    tgt = np.array([targets[k] for k in KIND_ORDER])
    if float(cap_kind[tgt > 0].sum()) <= 0:
        raise ValueError("no data on disk for any kind this stage asks for")
    x, unplaced = water_fill(tgt, cap_kind, 1.0)
    scale = 1.0
    if unplaced > 1e-12:
        scale = 1.0 / float(cap_kind[tgt > 0].sum())
        x, _ = water_fill(tgt, cap_kind * scale, 1.0)
        notes.append(f"all data at {mix.max_epochs:g} epochs fills only {1 - unplaced:.3f} of the "
                     f"horizon: epoch limit raised to {mix.max_epochs * scale:.2f}")
    kinds = {k: float(v) for k, v in zip(KIND_ORDER, x)}

    share: dict[str, float] = {}
    target: dict[str, float] = {}
    within: dict[str, dict[str, float]] = {}
    capped: list[str] = []
    for k in KIND_ORDER:
        names = list(weights[k])
        for s in names:
            target[s] = targets[k] * weights[k][s]
        if not names:
            continue
        caps = np.array([cap_src[s] * scale for s in names])
        y, _ = water_fill([weights[k][s] for s in names], caps, kinds[k])
        for s, v, cp in zip(names, y, caps):
            share[s] = float(v)
            if v > 0 and v >= cp * (1 - 1e-9):
                capped.append(s)
        if kinds[k] > 0:
            within[k] = {s: share[s] / kinds[k] for s in names}
    epochs = {s: share[s] * horizon_tokens / avail[s] for s in share}

    lib_total = targets["library"]
    lib_rows = kinds["library"] * (lib_rows_part / lib_total) if lib_total > 0 else 0.0
    chat_rows_total = 1.0 - kinds["base"] - lib_rows
    rows = {"base": kinds["base"], "library": lib_rows, "chat": max(0.0, chat_rows_total)}
    chat_parts = {"chat": kinds["chat"], "rag": kinds["rag"], "math": kinds["math"],
                  "library": kinds["library"] - lib_rows}
    chat_rows = _normalise({k: max(0.0, v) for k, v in chat_parts.items() if v > 1e-15})
    moved = {k: kinds[k] - targets[k] for k in KIND_ORDER}
    for k in KIND_ORDER:
        if targets[k] > 0 and kinds[k] < targets[k] * (1 - 1e-9):
            notes.append(f"{k}: target {targets[k]:.4f} -> {kinds[k]:.4f} (all its sources at the cap)")
    return Mixture(rows=rows, chat_rows=chat_rows, kinds=kinds, sources=within, share=share,
                   target=target, epochs=epochs, capped=tuple(capped), kind_target=targets,
                   moved=moved, missing=tuple(missing), epoch_limit=mix.max_epochs * scale,
                   horizon_tokens=float(horizon_tokens), notes=tuple(notes))


def segment_probs(token_shares: Mapping[str, float], mean_tokens: Mapping[str, float]) -> dict[str, float]:
    """Probabilities for drawing whole segments so that their expected token shares equal
    ``token_shares``: ``p_k ~ share_k / mean_tokens_k`` (entries without a positive mean length
    are dropped)."""
    raw = {k: float(s) / float(mean_tokens[k]) for k, s in token_shares.items()
           if s > 0 and float(mean_tokens.get(k, 0.0)) > 0}
    return _normalise(raw)


def library_tokens(n_wiki: int, n_qa: int, mix: MixCfg = MixCfg()) -> dict[str, float]:
    """``available`` entries of the two library tasks: one epoch of draws over ``n_wiki``
    passages / ``n_qa`` entries times the mean tokens per draw (``MixCfg.read_task_tokens`` /
    ``qa_task_tokens``)."""
    return {"library_read": float(n_wiki) * mix.read_task_tokens,
            "library_qa": float(n_qa) * mix.qa_task_tokens}


def library_available(lib: Any, mix: MixCfg = MixCfg()) -> dict[str, float]:
    """:func:`library_tokens` for a library (``kind_range`` sizes; ``{}`` for ``None``)."""
    if lib is None:
        return {}
    view = as_view(lib)
    return library_tokens(len(view.kind_range("wiki")), len(view.kind_range("qa")), mix)


def tokens_on_disk(root: str | Path, names: Sequence[str] | None = None) -> dict[str, float]:
    """Tokens of the committed shards of every source under ``<root>/tok/`` (manifest sums;
    all registry sources when ``names`` is ``None``; sources without shards get 0)."""
    from bes.data import shards

    out = {}
    for name in names if names is not None else [s.name for s in registry.SOURCES]:
        directory = Path(root) / "tok" / name
        entries = shards.read_manifest(directory) if directory.exists() else {}
        out[name] = float(sum(int(e.get("n_tokens", 0)) for e in entries.values()))
    return out


# ------------------------------------------------------------------- transform policy
@dataclass(frozen=True)
class TransformPlan:
    """Transforms to apply to one sample."""

    hindsight: bool = False
    retrieval: bool = False


def transform_probs(kind: str, stage: Any, data: DataCfg = DataCfg(), library: bool = True) -> dict[str, float]:
    """Probability of each transform for a sample of sub-kind ``kind`` (module docstring)."""
    retrieval = float(stage.library_frac_chat) if (library and data.use_library and kind in ("chat", "rag")) else 0.0
    return {"hindsight": float(data.hindsight_frac) if kind == "chat" else 0.0, "retrieval": retrieval}


def draw_transforms(kind: str, stage: Any, rng: np.random.Generator, data: DataCfg = DataCfg(),
                    library: bool = True) -> TransformPlan:
    """A :class:`TransformPlan` drawn with :func:`transform_probs` (two uniforms from ``rng``)."""
    p = transform_probs(kind, stage, data, library)
    u = rng.random(2)
    return TransformPlan(hindsight=bool(u[0] < p["hindsight"]), retrieval=bool(u[1] < p["retrieval"]))


def apply_transforms(sample: Sample, todo: TransformPlan, tok, rng: np.random.Generator, *,
                     lib: Any = None, data: DataCfg = DataCfg(), mix: MixCfg = MixCfg(),
                     max_len: int | None = None, stats: RagStats | None = None) -> Sample:
    """``sample`` after the planned transforms: hindsight (kept unchanged when ``relabel``
    declines; counted as ``hindsight:applied`` / ``hindsight:declined``), then retrieval
    (``rag_format.insert_retrieval``; skipped without ``lib``). Each transform draws from its
    own generator seeded from ``rng``, so one transform's draws never shift the other's."""
    seeds = rng.integers(0, 2 ** 63, size=2)
    out = sample
    if todo.hindsight:
        relabelled = apply_hindsight(out, np.random.default_rng(int(seeds[0])), tok, max_len=max_len)
        if stats is not None:
            stats.count("hindsight:applied" if relabelled is not None else "hindsight:declined")
        if relabelled is not None:
            out = relabelled
    if todo.retrieval and lib is not None:
        out = insert_retrieval(out, lib, tok, np.random.default_rng(int(seeds[1])), data, mix,
                               max_len=max_len, stats=stats)
    return out


__all__ = ["water_fill", "Mixture", "plan", "kind_weights", "segment_probs", "library_tokens",
           "library_available", "tokens_on_disk", "TransformPlan", "transform_probs",
           "draw_transforms", "apply_transforms", "KIND_ORDER", "CHAT_SUBKINDS", "ROW_KINDS",
           "TRANSFORMS"]
