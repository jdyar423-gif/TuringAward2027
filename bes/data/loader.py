"""Deterministic training loader: packed rows, exact-layer features in worker processes and a
plan journal for restarts (``docs/SPEC.md`` sections 1 and 3, ``docs/SPEC_MODEL_TRAIN.md``
section 6, ``docs/PLAN.md`` section 5).

Use
---
::

    with Loader(cfg, tok, "data_cache", library=lib, n_workers=8, seed=0) as loader:
        Trainer(cfg, ..., data=loader).run()          # loader(index, stage) -> batch
    # IteratorData(loader.open) also works, but it only passes a new stage on a shape change:
    # give the trainer the loader itself so every block is planned with the current stage.

``loader(index, stage)`` returns the batch of training step ``index`` for
``stage.shape = (A, B, T)`` as SPEC section 3 (:data:`BATCH_FIELDS`): ``tokens int32[A,B,T+1]``,
``w float32[A,B,T]`` (weight of each target, ``bes.data.samples.loss_weights``), ``seg
int32[A,B,T]``, and ``reg / clock / cand_tok / cand_feat`` = ``bes.exact.features.
exact_features(tokens[..., :T], seg, tok)`` of every micro-batch. Every row starts with ``BOS``
and packs one or more segments; padding is one more segment with weight 0
(``bes.data.packing``).

Determinism
-----------
The batch of step ``index`` is a pure function of the seed, the index, the stage parameters with
which its *block* of ``LoaderCfg.block_steps`` steps was planned, and the plan journal. The
journal (``journal.jsonl``, JSON lines appended with fsync, a torn last line ignored; see
``bes.data.shards.append_jsonl``) holds a header with every setting that decides batches and one
record per planned block: its first step, the stage (shape, chat fraction, library fractions,
assistant-only), the realised mixture (row-kind shares, source shares, transform
probabilities), the shards first seen at this block and how many shards of each source are
visible, the shard universe of every epoch, the library used, and the state of every stream at
the block start. A record is fsynced before any batch of its block is handed out; the record of
block ``b+1`` is written when block ``b`` has been produced (its end states are the start states
of ``b+1``).

Records only ever get appended. The effective plan is the chain obtained by reading records in
file order, each record cutting off the chain at its start. A request for ``index``:

- inside a planned block with the same shape: the stream states at ``index`` are replayed from
  the block's recorded start states (packing and transforms, no features) and the batch is
  reproduced bit for bit, whatever the stage passed now (its other parameters apply from the
  next block on);
- inside a planned block with another shape (the trainer crossed a shape switch, possibly after
  a restart): a new block is planned at ``index`` with the states replayed to ``index``;
- after the last planned block (a rollback skip, or a fresh journal): a new block at ``index``
  starts from the end states of the last planned block; the skipped indices consume no data.

Streams
-------
``LoaderCfg.n_streams`` virtual streams (``V``, a power of two) are dealt to the workers
(``v -> v mod W``); row ``m`` of the run belongs to stream ``m mod V`` (``packing.step_rows``),
so the output does not depend on the number of workers. A stream state is plain integers
(:class:`StreamState`): rows produced, a cursor per source (epoch, reads in the epoch, shard
universe), counters of library tasks, the document being continued in base rows, the draws in
its two look-ahead buffers, tokens drawn per source, and the deficit counters of the block.
Samples are referred to by *draw records* and rebuilt deterministically:

- source ``s`` epoch ``e`` covers the first ``U_e`` shards of ``s`` in first-seen order; its
  ``N`` segments are permuted by ``samples.permute_index(k, N, seed, "epoch", s, e)`` and dealt
  to the streams (``packing.epoch_position``). ``U_e`` is the number of shards visible in the
  first block in which any stream entered epoch ``e`` (recorded in the next record), so every
  planned block sees only the shards its record lists. A stream starts epoch ``e + 1`` when it
  has read its share of epoch ``e``, or as soon as the block's visible segments of ``s`` are
  ``1 + LoaderCfg.reepoch_growth`` times those of ``U_e``: shards committed during the run are
  read from the next block on;
- chat, rag and math samples are cut at an assistant turn boundary to ``T + 1`` tokens: a
  conversation that does not fit keeps its prefix or, with probability ``MixCfg.later_window``,
  one of its later windows (``samples.fit_draw``); then they are transformed
  (``mixture.apply_transforms``: hindsight, retrieval); all draws keyed by ``(seed, source,
  epoch, stream, read)``;
- held-out split: a conversation of a chat, rag or math source whose ``(source, shard id,
  segment index)`` hash falls below ``LoaderCfg.heldout_frac`` (:func:`is_heldout`, rule
  :data:`HELDOUT_RULE` in the journal header) is skipped by every draw of every epoch;
  :meth:`Loader.heldout` / :func:`heldout_samples` read those conversations (sampling prompts,
  probes) without workers;
- library tasks: draw ``j`` of stream ``v`` is task draw ``v + j V`` (``rag_format.pick_entry``);
  the loader prepares the library (``LibraryView.prepare_tasks``), so reading tasks are dealt
  only over the entries ``rag_format.reading_suitable`` keeps (cached next to the journal; the
  journal's library fingerprint records the rule and the number of entries).

Rows: base rows (``packing.fill_base``) from the base sources by token deficit; library rows
and chat rows by best fit from a look-ahead buffer of ``lookahead`` samples; row kinds by the
low-discrepancy rule of ``packing.row_kind`` with the block's row shares (``mixture.plan``).

Mixture: every block is planned with ``mixture.plan(stage, available, horizon)`` where
``available`` is each source's remaining epoch budget ``(max_epochs * size - consumed) /
max_epochs`` (``consumed``: tokens drawn so far, summed over the stream states) and ``horizon``
the remaining tokens of the run (``tokens / p``, before ``p = horizon_min_p``
``LoaderCfg.horizon_tokens``), so no source is read more than ``max_epochs`` times over the run.
``size`` is the visible tokens, except for a source that got new shards within the last
``growing_blocks`` blocks: then the size the registry expects after its download, so that the
first shards of a source still being downloaded do not cap it (and shift the mixture to other
kinds) early in the run. When a cap cuts a source's planned share below two thirds of its
target, the loader logs one warning for that source per run (``logging``, logger
``bes.data.loader``) and lists it in ``stats()["warnings"]``.

Workers
-------
``n_workers = 0`` runs everything in this process. Otherwise worker processes (``spawn``,
picklable inputs, each loads the tokenizer, the transducer tables and the library once) receive
commands through one pipe each and write rows and features straight into a ring of
``LoaderCfg.ring`` shared-memory batch slots; this process copies a finished batch out and frees
its slot (back-pressure: at most ``ring`` steps are in flight). Every command after a worker's
last acknowledged step is kept; a worker that dies (or stays silent ``task_timeout_s`` on
pending work; replays send a heartbeat per step) is replaced and its commands replayed, so a
step is never lost or duplicated (its rows are a pure function of the plan and the states).
Workers ask the kernel to kill them when the thread that created the loader ends (Linux
``PR_SET_PDEATHSIG``: create the loader in the main thread) and exit when their pipe closes;
``close()`` stops and joins them and unlinks the shared memory (after a SIGKILL of this process
the multiprocessing resource tracker unlinks it).

Memory bound. This process: ``ring`` slots of :func:`batch_nbytes` (53.5 MB at (8, 32, 1024)
and at (8, 16, 2048)), one batch copy, ~140 MB of its own, the journal records (2-8 kB each:
compressed stream states) and a window of ``2 * ring`` per-step stream states (~0.2 MB each).
Per worker (measured at both shapes with the library): ~85-160 MB private memory (tokenizer,
transducer tables, ``V / W`` streams with about ``lookahead + lib_lookahead`` samples of at most
``T + 1`` tokens each, the exact-layer intermediates of ``feature_rows`` rows), plus file-backed
pages of the memory-mapped library and shards (~250 MB here, page cache shared between the
workers) and the ring slots it writes (shared). Nothing grows with the length of the run except
the journal (one record per block, ~10 MB for 1500 blocks), which is only read at start-up.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
import weakref
import zlib
from collections import Counter, OrderedDict, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing import connection as mpc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bes.config import (BOS, CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG, Cfg, DataCfg, LoaderCfg,
                        MixCfg)
from bes.data import mixture as M
from bes.data import packing as P
from bes.data import shards as SH
from bes.data import sources as registry
from bes.data.rag_format import LIBRARY_TASKS, SUITABLE_RULE, RagStats, as_view, library_task, pick_entry
from bes.data.samples import (Sample, WeightRule, fit, fit_draw, from_tokens, permute_index, rng_for,
                              stock_deflection, stock_refusal)

JOURNAL_FORMAT = "bes-loader-journal/1"
SHARD_KINDS = ("base", "chat", "rag", "math")
ROW_KINDS = ("base", "library", "chat")
#: Remaining epoch budget kept for a source that has used up ``max_epochs`` (share of an epoch).
BUDGET_FLOOR = 1e-3
#: A planned share cut below this fraction of its target by the epoch caps is warned about.
CUT_WARN = 2.0 / 3.0
#: Kinds whose conversations the held-out split applies to, and its rule (journal header).
HELDOUT_KINDS = ("chat", "rag", "math")
#: Conversations in which an assistant turn opens with a stock refusal ("I'm sorry, but I can't
#: assist with that.") are skipped like held-out ones (counted as ``fail/<source>/refusal``).
#: The chat mixture holds safety refusals (Tulu 3: CoCoNot, WildJailbreak, WildGuardMix); at a
#: tenth of the training of 23-24 Nov the model gave that sentence to "write a product review of
#: running shoes". A function of the data only, so a stream stays reproducible; a journal written
#: before the rule replays its current block with it (the recorded block starts stay valid).
DROP_STOCK_REFUSALS = True
#: Chat conversations in which an assistant turn opens by saying what the assistant is or lacks
#: ("As an AI language model, ...", "I don't have access to real-time data ...", "I'm sorry for
#: any confusion, but as an AI ...") are skipped too (``fail/<source>/deflection``). Measured on
#: the tokenised sources on 24 Nov: 7% of the conversations of UltraChat, about a fifth of the
#: system-prompt chats of SmolTalk; at 0.4 of the training the model answered "My phone fell
#: into a sink full of water" with "I'm sorry, but I don't have access to the latest updates".
#: Only sources of kind "chat": in the retrieval sources "the passage does not say" is the task.
DROP_STOCK_DEFLECTIONS = True
DEFLECTION_KINDS = ("chat",)

HELDOUT_RULE = ("heldout/1: h = sm(sm(sm(blake2b64(source)) ^ shard) ^ segment), held out when "
                "h >> 11 < frac * 2^53 (sm = SplitMix64 finaliser)")
_LOG = logging.getLogger(__name__)
_PASSAGE_TAGS = ("retrieval:gated", "retrieval:ranked", "retrieval:random")


# ---------------------------------------------------------------------- held-out split
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on ``uint64`` arrays (wrapping arithmetic)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _name_key(name: str) -> np.ndarray:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8, person=b"bes-heldout").digest()
    return np.array([int.from_bytes(digest, "little")], dtype=np.uint64)


def heldout_hash(source: str, shard: int, segments: Any) -> np.ndarray:
    """``uint64[n]``: the hash of the held-out rule (:data:`HELDOUT_RULE`) of segments
    ``segments`` (indices inside shard ``shard`` of ``source``)."""
    seg = np.atleast_1d(np.asarray(segments, dtype=np.int64)).astype(np.uint64)
    base = _splitmix64(_splitmix64(_name_key(source)) ^ np.uint64(int(shard)))
    return _splitmix64(base ^ seg)


def is_heldout(source: str, shard: int, segments: Any, frac: float) -> np.ndarray:
    """``bool[n]``: which of ``segments`` of shard ``shard`` of ``source`` are kept out of
    training (about ``frac`` of them; none for ``frac <= 0``)."""
    seg = np.atleast_1d(np.asarray(segments, dtype=np.int64))
    if frac <= 0 or seg.size == 0:
        return np.zeros(seg.shape, dtype=bool)
    return (heldout_hash(source, shard, seg) >> np.uint64(11)).astype(np.float64) < float(frac) * 2.0 ** 53

#: SPEC section 3: key -> (dtype, trailing dims after ``[A, B, T]``); ``tokens`` has ``T + 1``.
#: Equal to ``bes.train.train.BATCH_FIELDS`` (not imported: that module imports jax).
BATCH_FIELDS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "tokens": (np.int32, ()),
    "w": (np.float32, ()),
    "seg": (np.int32, ()),
    "reg": (np.int16, (N_REG,)),
    "clock": (np.int32, (N_CLOCKS,)),
    "cand_tok": (np.int32, (CAND_SLOTS,)),
    "cand_feat": (np.int8, (CAND_SLOTS, N_CAND_FEAT)),
}
_ALIGN = 64


# ------------------------------------------------------------------------- batch layout
def batch_shapes(shape: Sequence[int]) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    """``{key: (dtype, full shape)}`` of a batch of ``shape = (A, B, T)``."""
    a, b, t = (int(x) for x in shape)
    return {k: (np.dtype(dt), (a, b, t + 1 if k == "tokens" else t) + tail)
            for k, (dt, tail) in BATCH_FIELDS.items()}


def _layout(shape: Sequence[int]) -> tuple[list[tuple[str, np.dtype, tuple[int, ...], int]], int]:
    out, off = [], 0
    for k, (dt, shp) in batch_shapes(shape).items():
        out.append((k, dt, shp, off))
        off += -(-int(np.prod(shp)) * dt.itemsize // _ALIGN) * _ALIGN
    return out, off


def batch_nbytes(shape: Sequence[int]) -> int:
    """Bytes of one batch of ``shape`` in a ring slot (fields 64-byte aligned)."""
    return _layout(shape)[1]


def batch_views(buf: Any, shape: Sequence[int]) -> dict[str, np.ndarray]:
    """Arrays of a batch of ``shape`` laid out in ``buf`` (a buffer of at least
    :func:`batch_nbytes` bytes; the arrays share its memory)."""
    out = {}
    for k, dt, shp, off in _layout(shape)[0]:
        out[k] = np.ndarray(shp, dtype=dt, buffer=buf, offset=off)
    return out


# ------------------------------------------------------------------------ source table
def source_table(data: DataCfg = DataCfg()) -> tuple[str, ...]:
    """Names the loader can draw from, in a fixed order (indices of the draw records): the
    registry sources of ``SHARD_KINDS`` with a positive weight, then the library tasks."""
    names: list[str] = []
    for kind in SHARD_KINDS:
        names += sorted(registry.mixture_weights(kind, data.permissive_only))
    return tuple(names) + tuple(LIBRARY_TASKS)


def source_kind(name: str) -> str:
    """Mixture sub-kind of a source: its registry kind, ``"library"`` for a library task."""
    return "library" if name in LIBRARY_TASKS else registry.get(name).kind


def library_fingerprint(lib: Any) -> dict[str, Any] | None:
    """Identity of a library for the journal: entry counts, when known the stored numbers and
    the directory, and for a view prepared for the library tasks (``LibraryView.prepare_tasks``)
    the entries each task draws from (``None`` without a library)."""
    if lib is None:
        return None
    view = as_view(lib)
    info: dict[str, Any] = {"n_passages": int(view.n_passages), "wiki": len(view.kind_range("wiki")),
                            "qa": len(view.kind_range("qa"))}
    if view.task_pids:
        info["tasks"] = {"rule": SUITABLE_RULE, **{k: len(v) for k, v in sorted(view.task_pids.items())}}
    meta = getattr(view.lib, "meta", None)
    if isinstance(meta, Mapping) and isinstance(meta.get("accounting"), Mapping):
        info["numbers"] = meta["accounting"].get("total_numbers")
    fp = getattr(view.lib, "fingerprint", None)
    if isinstance(fp, str):
        info["id"] = fp
    path = getattr(view.lib, "path", None)
    if path:
        info["path"] = str(path)
    return info


def _same_library(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    keys = (set(a) | set(b)) - {"path"}
    return all(a.get(k) == b.get(k) for k in keys)


def _resolve_library(lib: Any) -> Any:
    """``lib`` (a path or a library object) as the loader uses it. An answers library
    (``bes.library.answers``) is put in training mode: a stored question equal to the
    sample's own is skipped, and the questions are held in memory."""
    if isinstance(lib, (str, os.PathLike)):
        from bes.library.retrieve import Library

        lib = Library.load(str(lib))
    inner = getattr(lib, "lib", lib)                 # a LibraryView wraps the library
    if callable(getattr(inner, "for_training", None)) and not getattr(inner, "train", True):
        lib = inner.for_training()
    return lib


# --------------------------------------------------------------------------- stage
def stage_dict(stage: Any) -> dict[str, Any]:
    """The stage parameters the loader records (``bes.train.schedule.Stage`` or any object with
    ``shape``, ``chat_frac``, ``library_frac_base``, ``library_frac_chat``, ``assistant_only``)."""
    return {"shape": [int(x) for x in stage.shape], "p": float(getattr(stage, "p", 0.0)),
            "chat_frac": float(stage.chat_frac),
            "library_frac_base": float(getattr(stage, "library_frac_base", 0.0)),
            "library_frac_chat": float(getattr(stage, "library_frac_chat", 0.0)),
            "assistant_only": bool(getattr(stage, "assistant_only", False)),
            "phase": str(getattr(stage, "phase", ""))}


# ------------------------------------------------------------------------ stream state
@dataclass
class StreamState:
    """State of one stream (module docstring); plain integers, JSON-able with :meth:`to_list`.

    n: rows produced. cur: source index -> ``[epoch, reads in the epoch, universe shards]``.
    lib: library task index -> draws made. carry: ``[source, epoch, read, universe, pos]`` of
    the document continued in the next base row. cb / lb: chat-row and library-row buffers, each
    item a draw record plus the row number at which it was drawn. drawn: source -> tokens drawn.
    dc: deficit counters of the current block (key -> tokens).
    Draw records: ``[0, source, epoch, read, universe, cap, flags]`` (a shard segment; ``cap =
    0`` for a base document, else the length it was cut to; flags: 1 hindsight, 2 retrieval) or
    ``[1, task, draw, cap]`` (a library task).
    """

    n: int = 0
    cur: dict[int, list[int]] = field(default_factory=dict)
    lib: dict[int, int] = field(default_factory=dict)
    carry: list[int] | None = None
    cb: list[list[int]] = field(default_factory=list)
    lb: list[list[int]] = field(default_factory=list)
    drawn: dict[int, int] = field(default_factory=dict)
    dc: dict[str, int] = field(default_factory=dict)

    def to_list(self) -> list:
        return [self.n, [[k, *v] for k, v in sorted(self.cur.items())],
                [[k, v] for k, v in sorted(self.lib.items())], self.carry,
                [list(x) for x in self.cb], [list(x) for x in self.lb],
                [[k, v] for k, v in sorted(self.drawn.items())],
                [[k, v] for k, v in sorted(self.dc.items())]]

    @classmethod
    def from_list(cls, x: Sequence) -> "StreamState":
        n, cur, lib, carry, cb, lb, drawn, dc = x
        return cls(n=int(n), cur={int(c[0]): [int(v) for v in c[1:]] for c in cur},
                   lib={int(k): int(v) for k, v in lib}, carry=None if carry is None else [int(v) for v in carry],
                   cb=[[int(v) for v in it] for it in cb], lb=[[int(v) for v in it] for it in lb],
                   drawn={int(k): int(v) for k, v in drawn}, dc={str(k): int(v) for k, v in dc})


def fresh_states(n_streams: int) -> list[list]:
    """States of ``n_streams`` streams that have produced nothing."""
    return [StreamState().to_list() for _ in range(n_streams)]


def encode_states(states: Sequence[Sequence]) -> str:
    """Compact text form of stream states (JSON, zlib, base64)."""
    raw = json.dumps(list(states), separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def decode_states(text: str) -> list[list]:
    return json.loads(zlib.decompress(base64.b64decode(text)).decode("utf-8"))


def _without_deficits(states: Sequence[Sequence]) -> list[list]:
    out = []
    for s in states:
        s = list(s)
        s[7] = []
        out.append(s)
    return out


# ------------------------------------------------------------------------- block plan
@dataclass(frozen=True)
class Plan:
    """A block record as the streams use it (indices into the source table)."""

    id: int
    start: int
    rows0: int
    shape: tuple[int, int, int]
    rule: WeightRule
    cum: tuple[int, ...]
    base: tuple[tuple[int, float], ...]
    chat: tuple[tuple[str, int, float], ...]
    lib: tuple[tuple[str, int, float], ...]
    probs: dict[str, tuple[float, float]]
    n_visible: dict[int, int]
    universes: dict[int, tuple[int, ...]]
    library: bool

    @property
    def T(self) -> int:
        return self.shape[2]

    @property
    def R(self) -> int:
        return self.shape[0] * self.shape[1]

    def universe(self, src: int, epoch: int) -> int:
        """Shards of epoch ``epoch`` of source ``src``: recorded, else the visible ones."""
        u = self.universes.get(src, ())
        return u[epoch] if epoch < len(u) else self.n_visible.get(src, 0)

    @classmethod
    def from_record(cls, rec: Mapping[str, Any], index_of: Mapping[str, int],
                    data: DataCfg) -> "Plan":
        st = rec["stage"]
        rule = WeightRule(assistant_only=bool(st["assistant_only"]), passage_w=float(data.passage_loss_w))
        return cls(id=int(rec["id"]), start=int(rec["start"]), rows0=int(rec["rows0"]),
                   shape=tuple(int(x) for x in rec["shape"]), rule=rule, cum=P.thresholds(rec["rows"]),
                   base=tuple((index_of[s], float(w)) for s, w in rec["base"]),
                   chat=tuple((sub, index_of[s], float(w)) for sub, s, w in rec["chat"]),
                   lib=tuple(("library", index_of[s], float(w)) for s, w in rec["lib"]),
                   probs={k: (float(v[0]), float(v[1])) for k, v in rec["probs"].items()},
                   n_visible={index_of[s]: int(n) for s, n in rec["n_visible"].items() if s in index_of},
                   universes={index_of[s]: tuple(int(x) for x in u) for s, u in rec["universes"].items()
                              if s in index_of},
                   library=rec.get("library") is not None)


def plan_message(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a record the workers need (everything but the stream states)."""
    return {k: v for k, v in rec.items() if k != "states"}


# ------------------------------------------------------------------------- shard store
class ShardStore:
    """Shards of every source in first-seen order (``order[name] = [(id, n_segments, n_tokens,
    has_roles)]``), opened lazily and memory-mapped; segment ``p`` of the universe of the first
    ``U`` shards is segment ``p - start`` of the shard that holds it."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.order: dict[str, list[tuple[int, int, int, bool]]] = {}
        self._open: dict[tuple[str, int], SH.Shard] = {}
        self._cum: dict[tuple[str, int], np.ndarray] = {}

    def extend(self, name: str, entries: Sequence[Sequence]) -> None:
        lst = self.order.setdefault(name, [])
        have = {e[0] for e in lst}
        for e in entries:
            if int(e[0]) not in have:
                lst.append((int(e[0]), int(e[1]), int(e[2]), bool(e[3])))
                have.add(int(e[0]))

    def _cumulative(self, name: str, n_shards: int) -> np.ndarray:
        key = (name, n_shards)
        c = self._cum.get(key)
        if c is None:
            lst = self.order.get(name, [])
            if n_shards > len(lst):
                raise RuntimeError(f"{name}: universe of {n_shards} shards, {len(lst)} known")
            c = np.cumsum([e[1] for e in lst[:n_shards]], dtype=np.int64)
            self._cum[key] = c
        return c

    def n_segments(self, name: str, n_shards: int) -> int:
        if n_shards <= 0:
            return 0
        c = self._cumulative(name, n_shards)
        return int(c[-1]) if c.size else 0

    def shard(self, name: str, k: int) -> SH.Shard:
        sid, n_seg, n_tok, has_roles = self.order[name][k]
        sh = self._open.get((name, sid))
        if sh is None:
            entry = {"shard": sid, "n_tokens": n_tok, "n_segments": n_seg, "has_roles": has_roles,
                     "has_recs": False}
            sh = SH.open_shard(self.root / name, entry)
            self._open[(name, sid)] = sh
        return sh

    def locate(self, name: str, n_shards: int, p: int) -> tuple[int, int, int]:
        """``(k, shard id, index in the shard)`` of segment ``p`` of the universe of the first
        ``n_shards`` shards (``k``: the shard's place in first-seen order)."""
        c = self._cumulative(name, n_shards)
        k = int(np.searchsorted(c, p, side="right"))
        return k, int(self.order[name][k][0]), p - (int(c[k - 1]) if k else 0)

    def segment(self, name: str, n_shards: int, p: int) -> tuple[np.ndarray, np.ndarray | None]:
        """``(ids uint16[n], roles int8[n] or None)`` of segment ``p`` of the universe."""
        k, _, local = self.locate(name, n_shards, p)
        sh = self.shard(name, k)
        return sh.segment(local), sh.segment_roles(local)


def scan_new_shards(root: Path, names: Sequence[str],
                    order: Mapping[str, Sequence[Sequence]]) -> dict[str, list[list]]:
    """Committed shards of every source under ``root/tok`` that ``order`` does not list yet:
    ``{name: [[id, n_segments, n_tokens, has_roles, input_bytes], ...]}`` (ascending ids;
    ``input_bytes``: size of the parquet file it was made from, 0 when unknown)."""
    out: dict[str, list[list]] = {}
    for name in names:
        directory = root / "tok" / name
        if not directory.exists():
            continue
        known = {int(e[0]) for e in order.get(name, ())}
        new = []
        for sid, e in sorted(SH.read_manifest(directory).items()):
            if sid in known or int(e.get("n_segments", 0)) <= 0:
                continue
            size = (e.get("input") or {}).get("size") or 0
            new.append([int(sid), int(e["n_segments"]), int(e["n_tokens"]), bool(e.get("has_roles")),
                        int(size)])
        if new:
            out[name] = new
    return out


def heldout_samples(data_dir: str | os.PathLike, kind_or_source: str, n: int, seed: int = 0, *,
                    data: DataCfg = DataCfg(), frac: float = LoaderCfg().heldout_frac) -> list[Sample]:
    """Up to ``n`` held-out conversations, read straight from the committed shards under
    ``data_dir/tok`` (no loader and no worker processes needed).

    ``kind_or_source``: a registry source of kind ``chat``, ``rag`` or ``math``, or one of these
    kinds (its sources by their mixture weights, ``sources.mixture_weights(kind,
    data.permissive_only)``: weighted sampling without replacement, every held-out conversation
    of a source equally likely). Held out: the segments :func:`is_heldout` marks with ``frac``
    (``LoaderCfg.heldout_frac``; the rule is :data:`HELDOUT_RULE`, stored in the loader
    journal's header); the training stream skips exactly these, in every epoch. Returns
    ``bes.data.samples.Sample`` objects (whole stored conversations, ``tags = ("heldout",)``),
    ordered by their draw; deterministic in ``seed`` and the shards on disk (shards committed
    later add held-out conversations). Raises ``KeyError`` for an unknown name and
    ``ValueError`` for a source of another kind."""
    if kind_or_source in HELDOUT_KINDS:
        kind = kind_or_source
        weights = registry.mixture_weights(kind, data.permissive_only)
    else:
        src = registry.get(kind_or_source)
        if src.kind not in HELDOUT_KINDS:
            raise ValueError(f"{src.name} is a {src.kind} source; the held-out split covers {HELDOUT_KINDS}")
        kind, weights = src.kind, {src.name: 1.0}
    keys, refs = [], []
    seed_key = _splitmix64(np.array([int(seed) & ((1 << 64) - 1)], dtype=np.uint64))
    for name, w in sorted(weights.items()):
        directory = Path(data_dir) / "tok" / name
        if w <= 0 or not directory.exists():
            continue
        found = []
        for sid, entry in sorted(SH.read_manifest(directory).items()):
            m = int(entry.get("n_segments", 0))
            local = np.flatnonzero(is_heldout(name, sid, np.arange(m), frac)) if m > 0 else np.zeros(0, np.int64)
            if local.size:
                h = _splitmix64(_splitmix64(_splitmix64(_name_key(name) ^ seed_key) ^ np.uint64(sid))
                                ^ local.astype(np.uint64))
                found.append((sid, entry, local, ((h >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53))
        total = sum(x[2].size for x in found)
        for sid, entry, local, u in found:
            keys.append(np.log(u) * (total / w))                  # Efraimidis-Spirakis keys
            refs += [(name, sid, entry, int(x)) for x in local]
    if not refs or n <= 0:
        return []
    key = np.concatenate(keys)
    order = np.lexsort((np.arange(key.size), -key))[:int(n)]
    out, open_shards = [], {}
    for i in order.tolist():
        name, sid, entry, x = refs[i]
        sh = open_shards.get((name, sid))
        if sh is None:
            sh = open_shards[(name, sid)] = SH.open_shard(Path(data_dir) / "tok" / name, entry)
        roles = sh.segment_roles(x)
        out.append(from_tokens(np.asarray(sh.segment(x)), None if roles is None else np.asarray(roles),
                               source=name, kind=kind, tags=("heldout",)))
    return out


# ------------------------------------------------------------------------------- stats
class StepStats:
    """Counters of the rows one worker produced for one step (merged by the coordinator).

    Keys of ``c``: ``tok/<row kind>/<sub-kind>/<source>`` and ``tgt/...`` (tokens and loss
    weight placed), ``pad/<row kind>``, ``rows/<row kind>``, ``draw/<source>`` and
    ``dtok/<source>`` (samples and tokens drawn), ``fail/<source>/<reason>``, ``cut/<source>``
    (tokens of later turns dropped by ``samples.fit``), ``row/fallback`` and ``row/empty``.
    ``time``: CPU seconds by phase."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.time: Counter = Counter()
        self.errors: list[str] = []

    def as_dict(self, rag: RagStats | None = None) -> dict[str, Any]:
        out = {"c": dict(self.c), "time": dict(self.time), "errors": self.errors[:5]}
        if rag is not None:
            out["rag"] = {"counts": dict(rag.counts), "tokens": dict(rag.tokens),
                          "samples": dict(rag.samples), "passage_tokens": dict(rag.passage_tokens)}
        return out


class _NoData(RuntimeError):
    pass


_MISS = object()


@dataclass
class StreamCtx:
    """What the streams of one worker share."""

    tok: Any
    store: ShardStore
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    seed: int
    V: int
    data: DataCfg
    mix: MixCfg
    lcfg: LoaderCfg
    lib: Any = None
    rag: RagStats = field(default_factory=RagStats)
    stats: StepStats = field(default_factory=StepStats)

    def error(self, where: str, exc: BaseException) -> None:
        self.stats.c[f"fail/{where}/error"] += 1
        if len(self.stats.errors) < 5:
            self.stats.errors.append(f"{where}: {type(exc).__name__}: {exc}")


class Stream:
    """One virtual stream: its state, the samples its buffers refer to (rebuilt on demand) and
    the row production of the module docstring."""

    def __init__(self, v: int, ctx: StreamCtx):
        self.v = v
        self.ctx = ctx
        self.state = StreamState()
        self._cache: dict[tuple, Any] = {}

    def set_state(self, state: StreamState) -> None:
        self.state = state
        keep = {tuple(it[:-1]) for it in state.cb + state.lb}
        if state.carry is not None:
            keep.add(self._carry_key(state.carry))
        self._cache = {k: v for k, v in self._cache.items() if k in keep}

    @staticmethod
    def _carry_key(carry: Sequence[int]) -> tuple:
        src, e, j, u, _ = carry
        return (0, src, e, j, u, 0, 0)

    # ----------------------------------------------------------------- cursors
    def _advance(self, plan: Plan, src: int) -> tuple[int, int, int]:
        """``(epoch, read, universe)`` of the next read of source ``src``; moves the cursor.

        The next epoch starts when the stream has read its share of the current one, or as soon
        as the block's visible segments of the source are ``1 + reepoch_growth`` times those of
        the current epoch's universe (shards that arrive during the run are read from the next
        block on, not only after a long first epoch)."""
        ctx = self.ctx
        name = ctx.names[src]
        cur = self.state.cur.get(src)
        e, j, u = (0, 0, plan.universe(src, 0)) if cur is None else cur
        n = ctx.store.n_segments(name, u)
        growth = float(ctx.lcfg.reepoch_growth)
        if cur is not None and growth > 0:
            visible = ctx.store.n_segments(name, plan.n_visible.get(src, 0))
            while visible > n and visible >= (1.0 + growth) * n:   # ends: unrecorded epochs see all
                e, j = e + 1, 0
                u = plan.universe(src, e)
                n = ctx.store.n_segments(name, u)
        while j >= P.epoch_share(self.v, ctx.V, n):
            e, j = e + 1, 0
            u = plan.universe(src, e)
            n = ctx.store.n_segments(name, u)
            if n == 0:
                raise _NoData(f"{name}: no segments in epoch {e}")
        self.state.cur[src] = [e, j + 1, u]
        return e, j, u

    def _flags(self, plan: Plan, sub: str, src: int, e: int, j: int) -> int:
        """Transforms of one draw: ``mixture.draw_transforms`` with the block's probabilities
        and a generator keyed by the draw."""
        ph, pr = plan.probs.get(sub, (0.0, 0.0))
        if ph <= 0.0 and pr <= 0.0:
            return 0
        u = rng_for(self.ctx.seed, "todo", self.ctx.names[src], e, self.v, j).random(2)
        return int(u[0] < ph) | (int(u[1] < pr) << 1)

    # ---------------------------------------------------------- materialization
    def _get(self, draw: Sequence[int]) -> Any:
        key = tuple(draw)
        hit = self._cache.get(key, _MISS)
        if hit is not _MISS:
            return hit
        val = self._materialize(key)
        if val is not None:
            self._cache[key] = val
        return val

    def _materialize(self, draw: tuple) -> Any:
        ctx = self.ctx
        name = ctx.names[draw[1]]
        try:
            return self._from_shard(draw) if draw[0] == 0 else self._from_library(draw)
        except _NoData:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad sample is skipped (deterministically) and counted
            ctx.error(name, exc)
            return None

    def _from_shard(self, draw: tuple) -> Any:
        ctx = self.ctx
        _, src, e, j, u, cap, flags = draw
        name = ctx.names[src]
        t0 = time.process_time()
        n = ctx.store.n_segments(name, u)
        p = permute_index(P.epoch_position(self.v, j, ctx.V, n), n, ctx.seed, "epoch", name, e)
        kind = ctx.kinds[src]
        if cap and kind in HELDOUT_KINDS and ctx.lcfg.heldout_frac > 0:
            _, sid, local = ctx.store.locate(name, u, p)
            if bool(is_heldout(name, sid, local, ctx.lcfg.heldout_frac)[0]):    # never trained
                ctx.stats.c[f"fail/{name}/heldout"] += 1
                ctx.stats.time["read"] += time.process_time() - t0
                return None
        ids, roles = ctx.store.segment(name, u, p)
        if cap == 0:
            out = np.array(ids, dtype=np.int32)
            ctx.stats.time["read"] += time.process_time() - t0
            return out
        if DROP_STOCK_REFUSALS and roles is not None and stock_refusal(ids, ctx.tok):
            ctx.stats.c[f"fail/{name}/refusal"] += 1                            # never trained
            ctx.stats.time["read"] += time.process_time() - t0
            return None
        if (DROP_STOCK_DEFLECTIONS and roles is not None and kind in DEFLECTION_KINDS
                and stock_deflection(ids, ctx.tok)):
            ctx.stats.c[f"fail/{name}/deflection"] += 1                         # never trained
            ctx.stats.time["read"] += time.process_time() - t0
            return None
        s = from_tokens(np.asarray(ids), None if roles is None else np.asarray(roles), source=name, kind=kind)
        if len(s) > cap and s.is_conversation:      # the prefix or a later window, per draw
            head = fit_draw(s, cap, rng_for(ctx.seed, "fit", name, e, self.v, j), ctx.mix.later_window)
        else:
            head, _ = fit(s, cap)
        ctx.stats.time["read"] += time.process_time() - t0
        if head is None:
            ctx.stats.c[f"fail/{name}/too_long"] += 1
            return None
        if len(head) < len(s):
            ctx.stats.c[f"cut/{name}"] += len(s) - len(head)
        if flags:
            t1 = time.process_time()
            todo = M.TransformPlan(hindsight=bool(flags & 1), retrieval=bool(flags & 2) and ctx.lib is not None)
            head = M.apply_transforms(head, todo, ctx.tok, rng_for(ctx.seed, "tf", name, e, self.v, j),
                                      lib=ctx.lib, data=ctx.data, mix=ctx.mix, max_len=cap, stats=ctx.rag)
            phase = {1: "hindsight", 2: "retrieval", 3: "hindsight+retrieval"}[int(flags)]
            ctx.stats.time[phase] += time.process_time() - t1
        self._conv_stats(kind, head)
        return head

    def _conv_stats(self, kind: str, s: Any) -> None:
        """Counters of the realised passages and windows of a conversation of sub-kind ``kind``
        (``conv/``, ``conv_passage/``, ``conv_distractor/``, ``conv_later/``, ``conv_window/``)."""
        c = self.ctx.stats.c
        c[f"conv/{kind}"] += 1
        tags = set(s.tags)
        if "window" in tags:
            c[f"conv_window/{kind}"] += 1
        if tags.intersection(_PASSAGE_TAGS):
            c[f"conv_passage/{kind}"] += 1
            c[f"conv_distractor/{kind}"] += int("retrieval:gated" not in tags)
            c[f"conv_later/{kind}"] += int("retrieval:later_turn" in tags)

    def _from_library(self, draw: tuple) -> Any:
        ctx = self.ctx
        _, task, j, cap = draw
        name = ctx.names[task]
        if ctx.lib is None:
            ctx.stats.c[f"fail/{name}/no_library"] += 1
            return None
        t0 = time.process_time()
        c = self.v + j * ctx.V
        try:
            pid = pick_entry(ctx.lib, name, c, ctx.seed)
            s = None if pid is None else library_task(name, pid, ctx.lib, ctx.tok, rng_for(ctx.seed, "lib", name, c),
                                                      WeightRule(), ctx.mix, stats=ctx.rag)
        finally:
            ctx.stats.time["library"] += time.process_time() - t0
        if s is None:
            ctx.stats.c[f"fail/{name}/declined"] += 1
            return None
        head, _ = fit(s, cap)
        if head is None:
            ctx.stats.c[f"fail/{name}/too_long"] += 1
        return head

    # ------------------------------------------------------------------- draws
    def _dc_add(self, key: str, n: int) -> None:
        self.state.dc[key] = self.state.dc.get(key, 0) + int(n)

    def _drawn(self, src: int, n: int) -> None:
        self.state.drawn[src] = self.state.drawn.get(src, 0) + int(n)
        self.ctx.stats.c[f"draw/{self.ctx.names[src]}"] += 1
        self.ctx.stats.c[f"dtok/{self.ctx.names[src]}"] += int(n)

    def _draw_base(self, plan: Plan) -> P.Carry | None:
        """The next base document (by token deficit), ``None`` after ``max_draw_attempts``
        failures in a row."""
        keys = [f"b:{src}" for src, _ in plan.base]
        shares = [w for _, w in plan.base]
        for _ in range(max(1, self.ctx.lcfg.max_draw_attempts)):
            k = P.deficit_pick(shares, [self.state.dc.get(key, 0) for key in keys])
            src = plan.base[k][0]
            e, j, u = self._advance(plan, src)
            draw = (0, src, e, j, u, 0, 0)
            ids = self._get(draw)
            if ids is not None and len(ids) >= 2:
                self._dc_add(keys[k], len(ids))
                self._drawn(src, len(ids))
                return P.Carry(draw, ids, 1)
            self._dc_add(keys[k], 1)
            self._cache.pop(draw, None)
            self.ctx.stats.c[f"fail/{self.ctx.names[src]}/empty"] += 1
        return None

    def _draw_item(self, plan: Plan, entries: Sequence[tuple[str, int, float]], prefix: str) -> list | None:
        """The next buffered sample (by token deficit over ``entries``) as a buffer item."""
        ctx = self.ctx
        keys = [f"{prefix}:{sub}:{src}" for sub, src, _ in entries]
        shares = [w for _, _, w in entries]
        cap = plan.T + 1
        for _ in range(max(1, ctx.lcfg.max_draw_attempts)):
            k = P.deficit_pick(shares, [self.state.dc.get(key, 0) for key in keys])
            sub, src, _ = entries[k]
            if ctx.kinds[src] == "library":
                j = self.state.lib.get(src, 0)
                self.state.lib[src] = j + 1
                draw: tuple = (1, src, j, cap)
            else:
                e, j, u = self._advance(plan, src)
                draw = (0, src, e, j, u, cap, self._flags(plan, sub, src, e, j))
            s = self._get(draw)
            if s is not None:
                self._dc_add(keys[k], len(s))
                self._drawn(src, len(s))
                return [*draw, self.state.n]
            self._dc_add(keys[k], 1)
        return None

    # -------------------------------------------------------------------- rows
    def row(self, plan: Plan, n: int) -> tuple[int, list[tuple[np.ndarray, np.ndarray, tuple[str, int]]]]:
        """``(row kind, pieces)`` of this stream's row number ``n`` (``== state.n``); pieces are
        ``(ids, roles, (sub-kind, source))``."""
        st = self.state
        if n != st.n:
            raise RuntimeError(f"stream {self.v}: row {n} requested, state at row {st.n}")
        kind = P.row_kind(self.v, n, self.ctx.V, plan.cum)
        if kind == 0:
            pieces = self._base_row(plan)
        else:
            pieces = self._buffer_row(plan, library=kind == 1)
            if not pieces and plan.base:
                self.ctx.stats.c["row/fallback"] += 1
                kind, pieces = 0, self._base_row(plan)
        if not pieces:
            self.ctx.stats.c["row/empty"] += 1
            pieces = [(np.array([BOS], np.int32), np.zeros(1, np.int8), ("none", -1))]
        st.n += 1
        return kind, pieces

    def _base_row(self, plan: Plan) -> list:
        st = self.state
        carry = None
        if st.carry is not None:
            key = self._carry_key(st.carry)
            ids = self._get(key)
            if ids is None:
                raise RuntimeError(f"stream {self.v}: the continued document cannot be read")
            carry = P.Carry(key, ids, st.carry[4])
        placed, carry = P.fill_base(plan.T + 1, carry, lambda: self._draw_base(plan) if plan.base else None)
        st.carry = None if carry is None else [*carry.key[1:5], carry.pos]
        out = []
        for key, ids in placed:
            if carry is None or key != carry.key:
                self._cache.pop(key, None)
            out.append((ids, np.zeros(len(ids), np.int8), ("base", key[1])))
        return out

    def _placeable(self, item: Sequence[int], cap: int) -> Any:
        s = self._get(item[:-1])
        if s is not None and len(s) > cap:      # a buffer carried into a block with a smaller T
            s, _ = fit(s, cap)
        return s

    def _buffer_row(self, plan: Plan, library: bool) -> list:
        ctx, st = self.ctx, self.state
        entries = plan.lib if library else plan.chat
        if not entries:
            return []
        buf = st.lb if library else st.cb
        size = ctx.lcfg.lib_lookahead if library else ctx.lcfg.lookahead
        prefix = "l" if library else "c"
        cap = plan.T + 1
        buf[:] = [it for it in buf if self._placeable(it, cap) is not None]

        def refill() -> list | None:
            return self._draw_item(plan, entries, prefix)

        while len(buf) < max(1, size):
            item = refill()
            if item is None:
                break
            buf.append(item)
        if not buf:
            return []
        first = P.oldest_waiting([it[-1] for it in buf], st.n, ctx.lcfg.max_wait)
        placed = P.pack_best_fit(buf, cap, lambda it: len(self._placeable(it, cap)), refill, first)
        out = []
        for it in placed:
            s = self._placeable(it, cap)
            self._cache.pop(tuple(it[:-1]), None)
            sub = "library" if it[0] == 1 else ctx.kinds[it[1]]
            out.append((s.ids, s.roles, (sub, it[1])))
        return out


# ------------------------------------------------------------------------ the worker
@dataclass(frozen=True)
class WorkerInit:
    """Picklable start-up data of one worker (or of the in-process executor)."""

    worker: int
    streams: tuple[int, ...]
    tok: Any
    data_dir: str
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    seed: int
    n_streams: int
    data: DataCfg
    mix: MixCfg
    lcfg: LoaderCfg
    slots: tuple[str, ...] = ()
    parent_pid: int | None = None


class _Abort(Exception):
    pass


class StreamWorker:
    """The streams of one worker and the commands of the coordinator (also run in-process)."""

    def __init__(self, init: WorkerInit):
        from bes.exact import instr_table

        self.init = init
        self.index_of = {n: i for i, n in enumerate(init.names)}
        store = ShardStore(Path(init.data_dir) / "tok")
        self.ctx = StreamCtx(tok=init.tok, store=store, names=init.names, kinds=init.kinds,
                             seed=init.seed, V=init.n_streams, data=init.data, mix=init.mix,
                             lcfg=init.lcfg)
        self.streams = {v: Stream(v, self.ctx) for v in init.streams}
        self.plans: dict[int, Plan] = {}
        self.block: int | None = None
        self.step_next: int | None = None
        instr_table.load(init.tok)          # the transducer tables, once

    def set_library(self, lib: Any) -> None:
        self.ctx.lib = None if lib is None else as_view(lib)

    def add_shards(self, delta: Mapping[str, Sequence[Sequence]]) -> None:
        for name, entries in delta.items():
            self.ctx.store.extend(name, entries)

    def add_plan(self, rec: Mapping[str, Any]) -> None:
        plan = Plan.from_record(rec, self.index_of, self.init.data)
        self.plans[plan.id] = plan

    def forget(self, ids: Sequence[int]) -> None:
        for i in ids:
            if i != self.block:
                self.plans.pop(int(i), None)

    def reset(self, block: int, step: int, states: Mapping[int, Sequence]) -> None:
        for v, s in states.items():
            self.streams[int(v)].set_state(StreamState.from_list(s))
        self.block, self.step_next = int(block), int(step)

    def states(self) -> dict[int, list]:
        return {v: s.state.to_list() for v, s in self.streams.items()}

    def run(self, block: int, step: int, out: Mapping[str, np.ndarray] | None = None,
            check: Any = None) -> dict[str, Any]:
        """Produce this worker's rows of ``step`` (block ``block``) into ``out`` (batch views),
        or only advance the streams when ``out`` is ``None`` (replay). Returns the step's stats."""
        from bes.exact.features import exact_features

        plan = self.plans[block]
        if block != self.block:
            if step != plan.start:
                raise RuntimeError(f"block {block} starts at step {plan.start}, not {step}")
            for s in self.streams.values():
                s.state.dc = {}
            self.block = block
        if step != self.step_next:
            raise RuntimeError(f"step {step} requested, the streams are at step {self.step_next}")
        ctx = self.ctx
        ctx.stats, ctx.rag = StepStats(), RagStats()
        tm = ctx.stats.time
        a_, b_, t_ = plan.shape
        layout = P.step_rows(plan.rows0 + (step - plan.start) * plan.R, a_, b_, ctx.V)
        rows: list[tuple[int, int]] = []
        t_rows = time.process_time()
        for v in sorted(self.streams):
            for n, a, b in layout.get(v, ()):
                if check is not None:
                    check()
                kind, pieces = self.streams[v].row(plan, n)
                if out is None:
                    continue
                t0 = time.process_time()
                row = P.assemble_row([(ids, rl) for ids, rl, _ in pieces], t_, plan.rule)
                out["tokens"][a, b] = row.tokens
                out["w"][a, b] = row.w
                out["seg"][a, b] = row.seg
                rows.append((a, b))
                self._row_stats(kind, pieces, row)
                tm["assemble"] += time.process_time() - t0
        measured = sum(tm[k] for k in ("read", "hindsight", "retrieval", "hindsight+retrieval",
                                       "library", "assemble"))
        tm["pack"] += max(0.0, time.process_time() - t_rows - measured)
        if out is not None and rows:
            t0 = time.process_time()
            chunk = max(1, ctx.lcfg.feature_rows)
            for lo in range(0, len(rows), chunk):
                part = rows[lo:lo + chunk]
                ia = np.array([a for a, _ in part])
                ib = np.array([b for _, b in part])
                f = exact_features(out["tokens"][ia, ib, :t_], out["seg"][ia, ib], ctx.tok)
                for key in ("reg", "clock", "cand_tok", "cand_feat"):
                    out[key][ia, ib] = f[key]
            ctx.stats.time["features"] += time.process_time() - t0
        self.step_next = step + 1
        return ctx.stats.as_dict(ctx.rag)

    def _row_stats(self, kind: int, pieces: Sequence, row: P.Row) -> None:
        c = self.ctx.stats.c
        rk = ROW_KINDS[kind]
        c[f"rows/{rk}"] += 1
        c[f"pad/{rk}"] += row.n_pad
        o = 0
        for ids, _, (sub, src) in pieces:
            n = len(ids)
            name = self.ctx.names[src] if src >= 0 else "none"
            c[f"tok/{rk}/{sub}/{name}"] += n
            lo = max(o - 1, 0)
            c[f"tgt/{rk}/{sub}/{name}"] += float(row.w[lo:o + n - 1].sum())
            o += n


def _die_with_parent() -> None:
    from bes.data.tokenize_worker import _die_with_parent as die

    die()


def _worker_main(conn: Any, init: WorkerInit) -> None:
    """Entry point of a worker process (spawn-safe): load once, then serve commands."""
    _die_with_parent()
    if init.parent_pid is not None and os.getppid() != init.parent_pid:
        os._exit(1)
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the coordinator decides on shutdown
    shms: list[Any] = []
    try:
        from multiprocessing import shared_memory

        worker = StreamWorker(init)
        shms = [shared_memory.SharedMemory(name=n, track=False) for n in init.slots]
        conn.send(("ready", os.getpid()))
    except BaseException:  # noqa: BLE001 - reported, then the process ends
        with contextlib.suppress(Exception):
            conn.send(("error", traceback.format_exc()))
        return
    inbox: deque = deque()

    def check() -> None:
        while conn.poll():
            msg = conn.recv()
            inbox.append(msg)
            if msg[0] == "sync":
                raise _Abort

    try:
        while True:
            msg = inbox.popleft() if inbox else conn.recv()
            op = msg[0]
            try:
                if op == "stop":
                    break
                if op == "shards":
                    worker.add_shards(msg[1])
                elif op == "library":
                    worker.set_library(msg[1])
                elif op == "plan":
                    worker.add_plan(msg[1])
                elif op == "forget":
                    worker.forget(msg[1])
                elif op == "reset":
                    worker.reset(msg[1], msg[2], msg[3])
                elif op == "replay":
                    _, gen, block, until = msg
                    while worker.step_next < until:
                        worker.run(block, worker.step_next, None, check)
                        conn.send(("progress", gen, worker.step_next))   # a heartbeat per step
                    conn.send(("replayed", gen, block, until, worker.states()))
                elif op == "step":
                    _, gen, block, step, slot, shape = msg
                    stats = worker.run(block, step, batch_views(shms[slot].buf, shape), check)
                    conn.send(("done", gen, block, step, worker.states(), stats))
                elif op == "sync":
                    conn.send(("synced", msg[1]))
                else:
                    raise ValueError(f"unknown command {op!r}")
            except _Abort:
                keep: deque = deque()
                synced = False
                for m in inbox:            # drop the stale work queued before the sync
                    if not synced and m[0] in ("step", "replay"):
                        continue
                    synced = synced or m[0] == "sync"
                    keep.append(m)
                inbox = keep
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    except BaseException:  # noqa: BLE001
        with contextlib.suppress(Exception):
            conn.send(("error", traceback.format_exc()))
    finally:
        for shm in shms:
            with contextlib.suppress(Exception):
                shm.close()


# ------------------------------------------------------------------------- executors
class _Local:
    """Executor without processes: one :class:`StreamWorker` with every stream; commands are
    queued and run when :meth:`poll` is called (so nothing is produced that is not needed)."""

    n_workers = 1

    def __init__(self, init: WorkerInit, slot_bytes: int, n_slots: int):
        self.worker = StreamWorker(init)
        self.bufs = [np.zeros(slot_bytes, dtype=np.uint8) for _ in range(n_slots)]
        self.queue: deque = deque()
        self.restarts = 0

    def static(self, op: str, payload: Any) -> None:
        getattr(self.worker, {"shards": "add_shards", "library": "set_library", "plan": "add_plan",
                              "forget": "forget"}[op])(payload)

    def send(self, w: int, msg: tuple) -> None:
        if msg[0] == "sync":            # nothing queued before a sync needs to run
            self.queue.clear()
        self.queue.append(msg)

    def views(self, slot: int, shape: Sequence[int]) -> dict[str, np.ndarray]:
        return batch_views(self.bufs[slot], shape)

    def poll(self, timeout: float) -> list[tuple[int, tuple]]:
        while self.queue:
            msg = self.queue.popleft()
            op = msg[0]
            if op == "reset":
                self.worker.reset(msg[1], msg[2], msg[3])
            elif op == "replay":
                _, gen, block, until = msg
                while self.worker.step_next < until:
                    self.worker.run(block, self.worker.step_next, None)
                return [(0, ("replayed", gen, block, until, self.worker.states()))]
            elif op == "step":
                _, gen, block, step, slot, shape = msg
                stats = self.worker.run(block, step, self.views(slot, shape))
                return [(0, ("done", gen, block, step, self.worker.states(), stats))]
            elif op == "sync":
                self.queue.clear()
                return [(0, ("synced", msg[1]))]
        return []

    def pids(self) -> list[int]:
        return []

    def close(self) -> None:
        self.queue.clear()


def _close_pool(procs: list, conns: list, shms: list) -> None:
    for conn in conns:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.send(("stop",))
    for p in procs:
        if p is not None:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
    for conn in conns:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    for shm in shms:
        with contextlib.suppress(Exception):
            shm.close()
        with contextlib.suppress(Exception):
            shm.unlink()


class _Pool:
    """Worker processes with one pipe each and a ring of shared-memory batch slots.

    Static data (shards, library, plans) is kept and re-sent to a replacement worker. Stateful
    commands (``sync``, ``reset``, ``replay``, ``step``) are logged per worker from its last
    acknowledged state on (an acknowledged step or replay becomes a ``reset`` to the state it
    reported) and replayed to a replacement, so the coordinator never sees a death. A worker
    with unacknowledged commands that sends nothing for ``timeout`` seconds is killed and
    replaced."""

    def __init__(self, make_init: Any, n_workers: int, n_streams: int, slot_bytes: int,
                 n_slots: int, start_method: str = "spawn", timeout: float = 600.0,
                 ready_timeout: float = 300.0):
        from multiprocessing import shared_memory

        self.ctx = mp.get_context(start_method)
        self.n_workers = max(1, min(int(n_workers), n_streams))
        self.streams = [tuple(v for v in range(n_streams) if v % self.n_workers == w)
                        for w in range(self.n_workers)]
        self.shms = [shared_memory.SharedMemory(create=True, size=max(1, slot_bytes)) for _ in range(n_slots)]
        self.make_init = make_init
        self.timeout = float(timeout)
        self.ready_timeout = ready_timeout
        self.procs: list[Any] = [None] * self.n_workers
        self.conns: list[Any] = [None] * self.n_workers
        self.logs: list[deque] = [deque() for _ in range(self.n_workers)]
        self.seen = [time.monotonic()] * self.n_workers
        self.static_shards: dict[str, list] = {}
        self.static_library: Any = None
        self.static_plans: dict[int, dict] = {}
        self.restarts = 0
        self._finalizer = weakref.finalize(self, _close_pool, self.procs, self.conns, self.shms)
        try:
            for w in range(self.n_workers):
                self._start(w)
            for w in range(self.n_workers):
                self._await_ready(w)
        except BaseException:
            self._finalizer()
            raise

    # -------------------------------------------------------------- processes
    def _start(self, w: int) -> None:
        parent, child = self.ctx.Pipe(duplex=True)
        init = self.make_init(w, self.streams[w], tuple(s.name for s in self.shms), os.getpid())
        proc = self.ctx.Process(target=_worker_main, args=(child, init), daemon=True,
                                name=f"bes-loader-{w}")
        proc.start()
        child.close()
        self.procs[w], self.conns[w] = proc, parent

    def _await_ready(self, w: int) -> None:
        conn, proc = self.conns[w], self.procs[w]
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if conn.poll(0.2):
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg[0] == "ready":
                    return
                if msg[0] == "error":
                    raise RuntimeError(f"loader worker {w} failed to start:\n{msg[1]}")
            elif not proc.is_alive():
                break
        raise RuntimeError(f"loader worker {w} did not start (exit code {proc.exitcode})")

    def _respawn(self, w: int) -> None:
        old = self.procs[w]
        with contextlib.suppress(Exception):
            if old.is_alive():
                old.kill()
            old.join(timeout=5)
        with contextlib.suppress(Exception):
            self.conns[w].close()
        self.restarts += 1
        self._start(w)
        self._await_ready(w)
        conn = self.conns[w]
        conn.send(("shards", self.static_shards))
        if self.static_library is not None:
            conn.send(("library", self.static_library))
        for rec in self.static_plans.values():
            conn.send(("plan", rec))
        for msg in self.logs[w]:
            conn.send(msg)
        self.seen[w] = time.monotonic()

    # ---------------------------------------------------------------- commands
    def static(self, op: str, payload: Any) -> None:
        if op == "shards":
            for name, entries in payload.items():
                self.static_shards.setdefault(name, []).extend(entries)
        elif op == "library":
            self.static_library = payload
        elif op == "plan":
            self.static_plans[int(payload["id"])] = payload
        elif op == "forget":
            for i in payload:
                self.static_plans.pop(int(i), None)
        for conn in self.conns:
            with contextlib.suppress(Exception):
                conn.send((op, payload))

    def _busy(self, w: int) -> bool:
        return any(m[0] != "reset" for m in self.logs[w])

    def send(self, w: int, msg: tuple) -> None:
        if not self._busy(w):
            self.seen[w] = time.monotonic()        # the silence clock starts with the work
        if msg[0] == "sync":
            self.logs[w].clear()
        self.logs[w].append(msg)
        with contextlib.suppress(Exception):   # a dead worker is found by poll and replaced
            self.conns[w].send(msg)

    def _acked(self, w: int, msg: tuple) -> None:
        """A sync, step or replay finished: drop it (and what came before it) from the log; a
        step or replay becomes a reset to the state it reported."""
        log = self.logs[w]
        if msg[0] == "synced":
            match = lambda m: m[0] == "sync" and m[1] == msg[1]  # noqa: E731
            head = None
        elif msg[0] == "done":
            _, _, block, step, states, _ = msg
            match = lambda m: m[0] == "step" and m[2] == block and m[3] == step  # noqa: E731
            head = ("reset", block, step + 1, states)
        else:
            _, _, block, until, states = msg
            match = lambda m: m[0] == "replay" and m[2] == block and m[3] == until  # noqa: E731
            head = ("reset", block, until, states)
        for k, m in enumerate(log):
            if match(m):
                for _ in range(k + 1):
                    log.popleft()
                if head is not None:
                    log.appendleft(head)
                return

    def views(self, slot: int, shape: Sequence[int]) -> dict[str, np.ndarray]:
        return batch_views(self.shms[slot].buf, shape)

    def _drain(self, w: int, events: list) -> bool:
        """Read every message waiting from worker ``w``; False when its pipe is broken."""
        conn = self.conns[w]
        while True:
            try:
                if not conn.poll():
                    return True
                msg = conn.recv()
            except (EOFError, OSError):
                return False
            self.seen[w] = time.monotonic()
            if msg[0] in ("done", "replayed", "synced"):
                self._acked(w, msg)
            events.append((w, msg))

    def poll(self, timeout: float) -> list[tuple[int, tuple]]:
        """Messages that arrived within ``timeout`` seconds, as ``(worker, message)``; dead and
        stuck workers are replaced on the way."""
        now = time.monotonic()
        for w in range(self.n_workers):
            if self._busy(w) and now - self.seen[w] > self.timeout:
                with contextlib.suppress(Exception):
                    self.procs[w].kill()
        waitables = {}
        for w in range(self.n_workers):
            waitables[self.conns[w]] = w
            waitables[self.procs[w].sentinel] = w
        ready = mpc.wait(list(waitables), timeout)
        events: list[tuple[int, tuple]] = []
        for w in sorted({waitables[obj] for obj in ready}):
            ok = self._drain(w, events)
            if not ok or not self.procs[w].is_alive():
                self.procs[w].join(timeout=5)
                self._respawn(w)
        return events

    def pids(self) -> list[int]:
        return [p.pid for p in self.procs]

    def close(self) -> None:
        self._finalizer()


# --------------------------------------------------------------------------- the chain
@dataclass
class Block:
    """A block of the effective plan: record ``rec`` covering steps ``[start, end)``."""

    rec: dict
    end: int

    @property
    def id(self) -> int:
        return int(self.rec["id"])

    @property
    def start(self) -> int:
        return int(self.rec["start"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.rec["shape"])

    @property
    def R(self) -> int:
        return self.shape[0] * self.shape[1]

    def rows_at(self, step: int) -> int:
        """Rows of the run produced before ``step`` (``start <= step <= end``)."""
        return int(self.rec["rows0"]) + (step - self.start) * self.R

    def tokens_at(self, step: int) -> int:
        return int(self.rec["tokens0"]) + (step - self.start) * self.R * self.shape[2]


def build_chain(records: Sequence[Mapping[str, Any]], block_steps: int) -> list[Block]:
    """The effective plan: block records in file order, each cutting off the chain at its start."""
    chain: list[Block] = []
    for rec in records:
        if rec.get("type") != "block":
            continue
        s = int(rec["start"])
        chain = [b for b in chain if b.start < s]
        if chain and chain[-1].end > s:
            chain[-1].end = s
        chain.append(Block(rec, s + int(block_steps)))     # the record itself is never changed
    return chain


def journal_settings(seed: int, lcfg: LoaderCfg, data: DataCfg, mix: MixCfg, tok: Any,
                     names: Sequence[str]) -> dict[str, Any]:
    """Everything outside the block records that decides batches (the journal header)."""
    keep = ("n_streams", "block_steps", "lookahead", "lib_lookahead", "max_wait", "max_draw_attempts",
            "reepoch_growth", "heldout_frac")
    d = asdict(data)
    return json.loads(json.dumps({
        "format": JOURNAL_FORMAT, "seed": int(seed), "loader": {k: getattr(lcfg, k) for k in keep},
        "data": {k: d[k] for k in ("hindsight_frac", "distractor_frac", "passage_loss_w",
                                   "use_library", "permissive_only")},
        "heldout": {"rule": HELDOUT_RULE, "kinds": list(HELDOUT_KINDS)},
        "mix": asdict(mix), "tokenizer": str(getattr(tok, "sha256", "")), "names": list(names)}))


# ---------------------------------------------------------------------------- loader
@dataclass
class _Pending:
    slot: int
    block: Block
    parts: dict[int, dict] = field(default_factory=dict)
    stats: list[dict] = field(default_factory=list)


class Loader:
    """``loader(index, stage) -> batch`` (module docstring).

    cfg: the run configuration (``cfg.data`` for the weights and transforms, ``cfg.run`` for
    the seed and the journal directory, the schedule for the batch shapes). tok: the tokenizer
    (``bes.tokenizer.tok.load()``). data_dir: the directory holding ``tok/<source>/``. library:
    a ``bes.library.retrieve.Library``, its directory, an object with the same contract, or
    ``None`` (``set_library`` adds one later; it applies from the next planned block).
    n_workers: worker processes (``None``: ``LoaderCfg.workers``; 0: in-process). seed: default
    ``cfg.run.seed``. journal: path of the plan journal (default ``<cfg.run.out_dir>/
    <LoaderCfg.journal_name>``); reopening an existing journal reproduces its batches.
    """

    def __init__(self, cfg: Cfg, tok: Any, data_dir: str | os.PathLike, library: Any = None,
                 n_workers: int | None = None, seed: int | None = None, *,
                 loader_cfg: LoaderCfg | None = None, mix: MixCfg | None = None,
                 journal: str | os.PathLike | None = None, start_method: str = "spawn",
                 tokens_before: int = 0):
        from bes.config import check_profile
        from bes.train import schedule

        check_profile(cfg.data)
        self.cfg = cfg
        #: training tokens of the run before this loader's first batch: 0, except for a stream
        #: that starts in the middle of a run (a run grafted onto another configuration:
        #: scripts/graft_run.py). The remaining horizon is estimated from all tokens of the run.
        self.tokens_before = int(tokens_before)
        self.data = cfg.data
        self.lcfg = loader_cfg if loader_cfg is not None else LoaderCfg()
        self.mix = mix if mix is not None else MixCfg()
        self.tok = tok
        self.data_dir = Path(data_dir)
        self.seed = int(cfg.run.seed if seed is None else seed)
        self.V = int(self.lcfg.n_streams)
        P.stream_bits(self.V)
        self.names = source_table(self.data)
        self.index_of = {n: i for i, n in enumerate(self.names)}
        self.kinds = tuple(source_kind(n) for n in self.names)
        self.shard_names = tuple(n for n in self.names if n not in LIBRARY_TASKS)
        self.journal_path = Path(journal) if journal is not None else Path(cfg.run.out_dir) / self.lcfg.journal_name
        self._warnings: dict[str, dict[str, Any]] = {}
        self._library = self._prepared(_resolve_library(library))
        self._lib_fp = library_fingerprint(self._library)
        self.shapes = tuple(schedule.shapes(cfg))
        self.slot_bytes = max(batch_nbytes(s) for s in self.shapes)
        self.n_slots = max(2, int(self.lcfg.ring))
        self._closed = False
        self._open_journal()

        nw = self.lcfg.workers if n_workers is None else int(n_workers)

        def make_init(w: int, streams: Sequence[int], slots: Sequence[str] = (),
                      parent: int | None = None) -> WorkerInit:
            return WorkerInit(worker=w, streams=tuple(streams), tok=tok, data_dir=str(self.data_dir),
                              names=self.names, kinds=self.kinds, seed=self.seed, n_streams=self.V,
                              data=self.data, mix=self.mix, lcfg=self.lcfg, slots=tuple(slots),
                              parent_pid=parent)

        if nw <= 0:
            self.exec: Any = _Local(make_init(0, range(self.V)), self.slot_bytes, self.n_slots)
            self._streams_of = [tuple(range(self.V))]
        else:
            self.exec = _Pool(make_init, nw, self.V, self.slot_bytes, self.n_slots, start_method,
                              timeout=self.lcfg.task_timeout_s)
            self._streams_of = self.exec.streams
        self.exec.static("shards", {k: [list(e) for e in v] for k, v in self.order.items()})
        if self._library is not None:
            self.exec.static("library", self._library)
        self._sent_plans: set[int] = set()
        self._gen = 0
        self._pos: int | None = None
        self._frontier = 0
        self._pending: dict[int, _Pending] = {}
        self._done: set[int] = set()
        self._free = list(range(self.n_slots))
        self._stage: Any = None
        self._state_cache: dict[tuple[int, int], list[list]] = {}
        self._waiting: dict[str, Any] = {}
        self._block_stats: OrderedDict[int, dict] = OrderedDict()
        self._timing: Counter = Counter()
        self._resets: Counter = Counter()

    # ------------------------------------------------------------------ journal
    def _open_journal(self) -> None:
        self.settings = journal_settings(self.seed, self.lcfg, self.data, self.mix, self.tok, self.names)
        records, problems = SH.scan_jsonl(self.journal_path)
        if problems:
            raise ValueError(f"{self.journal_path}: {problems[:3]}")
        if not records:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            header = {"type": "header", "settings": self.settings, "time": time.time()}
            SH.append_jsonl(self.journal_path, header)
            records = [header]
        head = records[0]
        if head.get("type") != "header":
            raise ValueError(f"{self.journal_path}: the first record is not a loader journal header")
        if head.get("settings") != self.settings:
            old, new = head.get("settings", {}), self.settings
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{self.journal_path} was written with other settings ({diff}); use a "
                             f"new journal or the same seed, LoaderCfg, DataCfg, MixCfg and tokenizer")
        self.records = [r for r in records if r.get("type") == "block"]
        self.order: dict[str, list[list]] = {}
        for rec in self.records:
            for name, entries in rec.get("new_shards", {}).items():
                self.order.setdefault(name, []).extend(entries)
        self.chain = build_chain(self.records, self.lcfg.block_steps)
        for b in self.chain:
            self._note_cuts(b.rec)
        used = [b.rec.get("library") for b in self.chain if b.rec.get("library") is not None]
        if used and not all(_same_library(u, self._lib_fp) for u in used):
            raise ValueError(f"{self.journal_path}: its blocks used the library {used[-1]}, "
                             f"this loader has {self._lib_fp}")
        self._next_id = 1 + max((int(r["id"]) for r in self.records), default=-1)

    def _find(self, index: int) -> Block | None:
        for b in reversed(self.chain):
            if b.start <= index < b.end:
                return b
            if b.start <= index:
                return None
        return None

    def _before(self, index: int) -> Block | None:
        """The last block that starts at or before ``index``."""
        for b in reversed(self.chain):
            if b.start <= index:
                return b
        return None

    # ------------------------------------------------------------------ planning
    def _universes(self, prev: Block | None, states: Sequence[Sequence]) -> dict[str, list[int]]:
        """Epoch universes after ``prev`` produced ``states``: the recorded ones plus every epoch
        entered during ``prev`` (with ``prev``'s visible shards)."""
        table = {k: list(v) for k, v in (prev.rec["universes"].items() if prev is not None else ())}
        if prev is None:
            return table
        top: dict[str, int] = {}
        for s in states:
            for c in s[1]:
                name = self.names[int(c[0])]
                top[name] = max(top.get(name, -1), int(c[1]))
        for name, e_max in top.items():
            lst = table.setdefault(name, [])
            while len(lst) <= e_max:
                lst.append(int(prev.rec["n_visible"].get(name, 0)))
        return table

    def _growing(self, new_shards: Mapping[str, Sequence]) -> set[str]:
        """Sources with new shards in this plan or in one of the last ``growing_blocks`` records
        of the chain (downloads and tokenisation still running)."""
        out = {k for k, v in new_shards.items() if v}
        for b in self.chain[-max(0, int(self.lcfg.growing_blocks)):] if self.lcfg.growing_blocks > 0 else ():
            out.update(k for k, v in b.rec.get("new_shards", {}).items() if v)
        return out

    def _expected_tokens(self, name: str) -> float:
        """Tokens of ``name`` once its download is complete: the registry's expected parquet
        bytes (``min(approx_bytes, download cap)``) times the tokens per parquet byte of its
        committed shards; 0 when the shards do not record their input size."""
        entries = self.order.get(name, ())
        tokens = sum(int(e[2]) for e in entries)
        size = sum(int(e[4]) for e in entries if len(e) > 4)
        if tokens <= 0 or size <= 0:
            return 0.0
        src = registry.get(name)
        try:
            cap = registry.download_cap(src, self.data.permissive_only)
        except ValueError:
            return 0.0
        expected = float(src.approx_bytes) if cap is None else float(min(src.approx_bytes, cap))
        return tokens / size * expected

    def _available(self, consumed: Mapping[str, int], growing: set[str] = frozenset()) -> tuple[dict[str, float], dict[str, float]]:
        """``(available, sizes)``: remaining epoch budget per source in the units of
        ``mixture.plan``, ``(max_epochs * size - consumed) / max_epochs``, and the sizes used.
        ``size``: the visible tokens (library tasks: ``mixture.library_available``); for a source
        still ``growing``, the larger of that and :meth:`_expected_tokens` (its first shards do
        not cap it while the rest is downloaded and tokenised; ``reepoch_growth`` brings the new
        shards in). With the remaining horizon this caps every source at ``max_epochs`` reads
        over the run. A spent budget keeps a floor of ``BUDGET_FLOOR`` of one epoch, so that when
        every source is spent the plan raises the epoch limit for all of them (shares by size)
        instead of having no data."""
        me = float(self.mix.max_epochs)
        sizes = {}
        for n in self.shard_names:
            visible = float(sum(int(e[2]) for e in self.order.get(n, ())))
            sizes[n] = max(visible, self._expected_tokens(n)) if n in growing and visible > 0 else visible
        lib = self._library if self.data.use_library else None
        sizes.update({k: float(v) for k, v in M.library_available(lib, self.mix).items()})
        avail = {name: (max(me * tokens - float(consumed.get(name, 0)), BUDGET_FLOOR * tokens) / me
                        if tokens > 0 else 0.0) for name, tokens in sizes.items()}
        return avail, sizes

    def _horizon(self, stage: Mapping[str, Any], tokens0: int, block_tokens: int) -> float:
        p = float(stage["p"])
        done = float(tokens0) + float(getattr(self, "tokens_before", 0))
        total = done / p if p >= self.lcfg.horizon_min_p and done > 0 else float(self.lcfg.horizon_tokens)
        return max(float(block_tokens), total - done)

    def _new_block(self, index: int, stage: Any, states: Sequence[Sequence], prev: Block | None,
                   at: int | None = None) -> Block:
        """Plan, record (fsync) and adopt a block starting at ``index`` whose streams start from
        ``states``, produced by ``prev`` up to step ``at`` (``prev.end`` for a gap)."""
        t0 = time.perf_counter()
        st = stage_dict(stage)
        shape = tuple(st["shape"])
        if shape not in self.shapes:
            raise ValueError(f"shape {shape} is not one of the run's shapes {self.shapes}")
        if prev is not None:
            at = prev.end if at is None else at
            rows0, tokens0 = prev.rows_at(at), prev.tokens_at(at)
        else:
            rows0 = tokens0 = 0
        states = _without_deficits(states)
        consumed: Counter = Counter()
        for s in states:
            for src, n in s[6]:
                consumed[self.names[int(src)]] += int(n)
        new_shards = scan_new_shards(self.data_dir, self.shard_names, self.order)
        for name, entries in new_shards.items():
            self.order.setdefault(name, []).extend(entries)
        if new_shards:
            self.exec.static("shards", new_shards)
        lib = self._library if self.data.use_library else None
        block_tokens = shape[0] * shape[1] * shape[2] * self.lcfg.block_steps
        horizon = self._horizon(st, tokens0, block_tokens)
        ns = SimpleNamespace(**st)
        growing = self._growing(new_shards)
        avail, sizes = self._available(consumed, growing)
        try:
            mx = M.plan(ns, avail, horizon, data=self.data, mix=self.mix, library=lib is not None)
        except ValueError as exc:
            raise RuntimeError(f"no training data under {self.data_dir / 'tok'}: {exc}") from exc
        chat = [[sub, src, mx.chat_rows[sub] * share] for sub in M.CHAT_SUBKINDS
                if mx.chat_rows.get(sub, 0.0) > 0 for src, share in sorted(mx.sources.get(sub, {}).items())
                if share > 0]
        rec = {
            "type": "block", "id": self._next_id, "start": int(index), "rows0": int(rows0),
            "tokens0": int(tokens0), "shape": list(shape), "stage": st,
            "rows": [mx.rows.get(k, 0.0) for k in ROW_KINDS],
            "base": [[s, w] for s, w in sorted(mx.sources.get("base", {}).items()) if w > 0],
            "chat": chat,
            "lib": [[s, w] for s, w in sorted(mx.sources.get("library", {}).items()) if w > 0],
            "probs": {sub: [M.transform_probs(sub, ns, self.data, lib is not None)["hindsight"],
                            M.transform_probs(sub, ns, self.data, lib is not None)["retrieval"]]
                      for sub in M.CHAT_SUBKINDS},
            "new_shards": new_shards, "n_visible": {n: len(v) for n, v in self.order.items()},
            "universes": self._universes(prev, states),
            "library": self._lib_fp if lib is not None else None,
            "consumed": dict(consumed), "horizon": horizon, "growing": sorted(growing),
            "sizes": {k: v for k, v in sizes.items() if v > 0},
            "mixture": {"kinds": mx.kinds, "capped": list(mx.capped), "epochs": mx.epochs,
                        "notes": list(mx.notes), "missing": list(mx.missing),
                        "target": {s: v for s, v in sorted(mx.target.items()) if v > 0},
                        "share": {s: v for s, v in sorted(mx.share.items()) if v > 0}},
            "states": encode_states(states), "time": time.time(),
        }
        SH.append_jsonl(self.journal_path, rec)
        self._note_cuts(rec)
        self._next_id += 1
        self.records.append(rec)
        self.chain = build_chain(self.records, self.lcfg.block_steps)
        blk = self.chain[-1]
        self._state_cache[(blk.id, blk.start)] = states
        self._timing["plan_s"] += time.perf_counter() - t0
        return blk

    def _note_cuts(self, rec: Mapping[str, Any]) -> None:
        """Warn (once per source in this process; kept for :meth:`stats`) when the epoch caps
        cut a source's planned token share in block record ``rec`` below ``CUT_WARN`` of its
        target."""
        mixture = rec.get("mixture") or {}
        target, share = mixture.get("target") or {}, mixture.get("share") or {}
        for src, want in sorted(target.items()):
            got = float(share.get(src, 0.0))
            if src in self._warnings or not want > 0 or got >= CUT_WARN * float(want):
                continue
            size = (rec.get("sizes") or {}).get(src)
            msg = (f"loader: the epoch caps cut the planned token share of {src} from {float(want):.4f} "
                   f"to {got:.4f} (at most {self.mix.max_epochs:g} reads of "
                   f"{float(size or 0):.3g} tokens over a horizon of {float(rec.get('horizon', 0)):.3g} "
                   f"tokens; block {rec.get('id')} at step {rec.get('start')}); the rest of its share "
                   f"goes to other sources")
            self._warnings[src] = {"source": src, "target": float(want), "share": got,
                                   "block": rec.get("id"), "step": rec.get("start"), "message": msg}
            _LOG.warning(msg)

    def _send_plan(self, blk: Block) -> None:
        if blk.id not in self._sent_plans:
            self.exec.static("plan", plan_message(blk.rec))
            self._sent_plans.add(blk.id)
        old = sorted(i for i in self._sent_plans if i < blk.id - 8)
        if old:
            self.exec.static("forget", old)
            self._sent_plans.difference_update(old)

    def _block_states(self, blk: Block) -> list[list]:
        hit = self._state_cache.get((blk.id, blk.start))
        if hit is None:
            hit = decode_states(blk.rec["states"])
            self._state_cache[(blk.id, blk.start)] = hit
        return hit

    # ------------------------------------------------------------- production
    def _reset_workers(self, blk: Block, step: int, states: Sequence[Sequence]) -> None:
        self._send_plan(blk)
        for w, streams in enumerate(self._streams_of):
            self.exec.send(w, ("reset", blk.id, step, {v: states[v] for v in streams}))

    def _states_at(self, blk: Block, step: int) -> list[list]:
        """States of all streams at ``step`` of ``blk`` (``start <= step <= end``): cached, or
        replayed from the nearest cached state of the block. Call after :meth:`_sync`."""
        hit = self._state_cache.get((blk.id, step))
        if hit is not None:
            return hit
        if step == blk.start:
            return self._block_states(blk)
        known = [s for (b, s) in self._state_cache if b == blk.id and blk.start <= s < step]
        frm = max(known) if known else blk.start
        states = self._state_cache[(blk.id, frm)] if known else self._block_states(blk)
        t0 = time.perf_counter()
        self._reset_workers(blk, frm, states)
        for w in range(len(self._streams_of)):
            self.exec.send(w, ("replay", self._gen, blk.id, step))
        parts = self._collect("replayed", lambda m: m[2] == blk.id and m[3] == step)
        merged = self._merge(parts)
        self._state_cache[(blk.id, step)] = merged
        self._timing["replay_s"] += time.perf_counter() - t0
        self._timing["replayed_steps"] += step - frm
        return merged

    def _merge(self, parts: Mapping[int, Mapping[int, Sequence]]) -> list[list]:
        out: list[Any] = [None] * self.V
        for states in parts.values():
            for v, s in states.items():
                out[int(v)] = s
        if any(s is None for s in out):
            raise RuntimeError("a worker reported no state for some of its streams")
        return out

    def _poll(self) -> list[tuple[int, tuple]]:
        events = self.exec.poll(0.0 if isinstance(self.exec, _Local) else 0.5)
        for w, msg in events:
            if msg[0] == "error":
                raise RuntimeError(f"loader worker {w} failed:\n{msg[1]}")
        return events

    def _collect(self, kind: str, match: Any) -> dict[int, dict]:
        """Wait until every worker sent a ``kind`` message (``replayed``) that ``match``es."""
        need = len(self._streams_of)
        parts: dict[int, dict] = {}
        while len(parts) < need:
            for w, msg in self._poll():
                if msg[0] == kind and msg[1] == self._gen and match(msg):
                    parts[w] = msg[4]
                else:
                    self._handle(w, msg)
        return parts

    def _sync(self) -> None:
        """Cancel everything in flight (a new generation); afterwards every slot is free."""
        self._gen += 1
        gen = self._gen
        for w in range(len(self._streams_of)):
            self.exec.send(w, ("sync", gen))
        waiting = set(range(len(self._streams_of)))
        while waiting:
            for w, msg in self._poll():
                if msg[0] == "synced" and msg[1] == gen:
                    waiting.discard(w)
        self._pending.clear()
        self._done.clear()
        self._free = list(range(self.n_slots))

    def _handle(self, w: int, msg: tuple) -> None:
        op = msg[0]
        if op == "error":
            raise RuntimeError(f"loader worker {w} failed:\n{msg[1]}")
        if op != "done" or msg[1] != self._gen:
            return
        _, _, block, step, states, stats = msg
        pend = self._pending.get(step)
        if pend is None or pend.block.id != block:
            return
        pend.parts[w] = states
        pend.stats.append(stats)
        if len(pend.parts) == len(self._streams_of):
            self._state_cache[(block, step + 1)] = self._merge(pend.parts)
            self._done.add(step)
            self._add_stats(pend.block, step, pend.stats)

    def _fill(self) -> None:
        """Dispatch steps while slots are free (at most ``ring`` ahead of the consumer)."""
        while self._free and self._pos is not None and self._frontier < self._pos + self.n_slots:
            step = self._frontier
            blk = self._find(step)
            if blk is None:
                last = self.chain[-1] if self.chain else None
                if last is None or step != last.end:
                    return
                states = self._state_cache.get((last.id, last.end))
                if states is None:
                    return                      # the block's last step is not finished yet
                blk = self._new_block(step, self._stage, states, prev=last)
            self._send_plan(blk)
            slot = self._free.pop()
            self._pending[step] = _Pending(slot, blk)
            for w in range(len(self._streams_of)):
                self.exec.send(w, ("step", self._gen, blk.id, step, slot, blk.shape))
            self._frontier += 1

    def _reset(self, index: int, stage: Any) -> None:
        """Position the production at ``index`` (module docstring: reproduce, re-plan for a
        new shape, or plan after a gap / on a fresh journal)."""
        shape = tuple(int(x) for x in stage.shape)
        self._sync()
        blk = self._find(index)
        if blk is not None and blk.shape == shape:
            states, target, why = self._states_at(blk, index), blk, "reproduce"
        else:
            if blk is not None:
                states, why = self._states_at(blk, index), "shape"
                target = self._new_block(index, stage, states, prev=blk, at=index)
            elif (prev := self._before(index)) is not None:
                states, why = self._states_at(prev, prev.end), "gap"
                target = self._new_block(index, stage, states, prev=prev)
            else:
                why = "fresh"
                target = self._new_block(index, stage, fresh_states(self.V), prev=None)
            states = self._block_states(target)
        self._resets[why] += 1
        self._reset_workers(target, index, states)
        self._pos = self._frontier = index

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the loader is closed")

    # ------------------------------------------------------------------- public
    def __call__(self, index: int, stage: Any) -> dict[str, np.ndarray]:
        """The batch of step ``index`` at ``stage`` (module docstring)."""
        self._check_open()
        index = int(index)
        shape = tuple(int(x) for x in stage.shape)
        if shape not in self.shapes:
            raise ValueError(f"shape {shape} is not one of the run's shapes {self.shapes}")
        self._stage = stage
        blk = self._find(index)
        if self._pos != index or (blk is not None and blk.shape != shape):
            self._reset(index, stage)
        t0 = time.perf_counter()
        while index not in self._done:
            self._fill()
            for w, msg in self._poll():
                self._handle(w, msg)
        self._timing["wait_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        pend = self._pending.pop(index)
        views = self.exec.views(pend.slot, pend.block.shape)
        local = isinstance(self.exec, _Local)
        batch = {k: np.array(v) for k, v in views.items()} if self.lcfg.copy_out or local else views
        self._free.append(pend.slot)
        self._done.discard(index)
        self._pos = index + 1
        self._timing["copy_s"] += time.perf_counter() - t0
        self._timing["batches"] += 1
        horizon = index - 2 * self.n_slots
        for key in [k for k in self._state_cache if k[1] < horizon]:
            del self._state_cache[key]
        self._fill()
        return batch

    get = __call__

    def open(self, start_index: int, stage: Any) -> Iterator[dict[str, np.ndarray]]:
        """Batches ``start_index, start_index + 1, ...`` at ``stage`` (for
        ``bes.train.train.IteratorData``)."""
        i = int(start_index)
        while True:
            yield self(i, stage)
            i += 1

    def set_library(self, library: Any) -> None:
        """Use ``library`` (or none) from the next planned block on; a journal can only be
        reopened with the library its blocks used. Prepares the reading-task pre-filter
        (``rag_format.reading_suitable``, a few seconds for lib_v2, cached next to the
        journal)."""
        self._library = self._prepared(_resolve_library(library))
        self._lib_fp = library_fingerprint(self._library)
        self.exec.static("library", self._library)

    def _prepared(self, lib: Any) -> Any:
        """``lib`` as a view whose reading tasks draw only suitable entries (none without a
        library or with ``DataCfg.use_library = False``)."""
        if lib is None or not self.data.use_library:
            return lib
        return as_view(lib).prepare_tasks(self.mix, cache_dir=self.journal_path.parent)

    def heldout(self, kind_or_source: str, n: int, seed: int = 0) -> list[Sample]:
        """Up to ``n`` held-out conversations of a source or of the sources of a kind
        (``chat | rag | math``): :func:`heldout_samples` on this loader's data directory with its
        settings. The training stream never returns them; no worker is involved."""
        return heldout_samples(self.data_dir, kind_or_source, n, seed, data=self.data,
                               frac=self.lcfg.heldout_frac)

    def worker_pids(self) -> list[int]:
        """Process ids of the workers (empty in-process)."""
        return list(self.exec.pids())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.exec.close()

    def __enter__(self) -> "Loader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -------------------------------------------------------------------- stats
    def _add_stats(self, blk: Block, step: int, parts: Sequence[Mapping[str, Any]]) -> None:
        bs = self._block_stats.get(blk.id)
        if bs is None:
            bs = {"c": Counter(), "time": Counter(), "rag": RagStats(), "errors": [], "steps": 0,
                  "first": step, "last": step}
            self._block_stats[blk.id] = bs
            while len(self._block_stats) > max(1, self.lcfg.stats_blocks):
                self._block_stats.popitem(last=False)
        bs["steps"] += 1
        bs["last"] = step
        for part in parts:
            bs["c"].update(part["c"])
            bs["time"].update(part["time"])
            if "rag" in part:
                r = RagStats()
                for k in ("counts", "tokens", "samples", "passage_tokens"):
                    getattr(r, k).update(part["rag"][k])
                bs["rag"].merge(r)
            if len(bs["errors"]) < 5:
                bs["errors"] += part["errors"][:5 - len(bs["errors"])]

    def _epochs(self, states: Sequence[Sequence]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        acc: dict[str, list[float]] = {}
        for v, s in enumerate(states):
            for c in s[1]:
                name = self.names[int(c[0])]
                e, j, u = int(c[1]), int(c[2]), int(c[3])
                n = sum(int(x[1]) for x in self.order.get(name, [])[:u])
                share = P.epoch_share(v, self.V, n)
                acc.setdefault(name, []).append(e + (j / share if share else 0.0))
        for name, vals in acc.items():
            out[name] = {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals),
                         "streams": len(vals)}
        return out

    def _summary(self, blk_id: int, bs: Mapping[str, Any]) -> dict[str, Any]:
        rec = next((r for r in self.records if int(r["id"]) == blk_id), None)
        c = bs["c"]
        shape = tuple(rec["shape"]) if rec else (0, 0, 0)
        row_len = shape[2] + 1
        tok = {k[4:]: v for k, v in c.items() if k.startswith("tok/")}
        total = sum(tok.values()) or 1
        by_sub: Counter = Counter()
        by_row: Counter = Counter()
        for k, v in tok.items():
            rk, sub, _ = k.split("/", 2)
            by_sub[f"{rk}/{sub}"] += v
            by_row[rk] += v
        rows = {rk: int(c.get(f"rows/{rk}", 0)) for rk in ROW_KINDS}
        pad = {rk: int(c.get(f"pad/{rk}", 0)) for rk in ROW_KINDS}
        pad_share = {rk: pad[rk] / (rows[rk] * row_len) for rk in ROW_KINDS if rows[rk]}
        all_rows = sum(rows.values())
        out = {
            "block": blk_id, "start": rec["start"] if rec else None, "shape": list(shape),
            "steps": bs["steps"], "first": bs["first"], "last": bs["last"],
            "stage": rec["stage"] if rec else None,
            "planned_rows": dict(zip(ROW_KINDS, rec["rows"])) if rec else None,
            "rows": rows, "row_share": {rk: rows[rk] / all_rows for rk in ROW_KINDS} if all_rows else {},
            "tokens": dict(sorted(tok.items())),
            "token_share": {k: v / total for k, v in sorted(tok.items())},
            "subkind_share": {k: v / total for k, v in sorted(by_sub.items())},
            "pad_share": pad_share,
            "pad_share_all": sum(pad.values()) / (all_rows * row_len) if all_rows else 0.0,
            "targets": float(sum(v for k, v in c.items() if k.startswith("tgt/"))),
            "draws": {k[5:]: v for k, v in c.items() if k.startswith("draw/")},
            "fail": {k[5:]: v for k, v in c.items() if k.startswith("fail/")},
            "cut_tokens": {k[4:]: v for k, v in c.items() if k.startswith("cut/")},
            "fallback_rows": int(c.get("row/fallback", 0)), "empty_rows": int(c.get("row/empty", 0)),
            "passages": self._passage_shares(c),
            "retrieval": bs["rag"].as_dict(), "time_s": dict(bs["time"]), "errors": list(bs["errors"]),
            "n_visible": rec["n_visible"] if rec else None,
            "capped": rec["mixture"]["capped"] if rec else None,
            "notes": rec["mixture"]["notes"] if rec else None,
        }
        states = self._state_cache.get((blk_id, bs["last"] + 1))
        if states is not None:
            out["epochs"] = self._epochs(states)
        return out

    @staticmethod
    def _passage_shares(c: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Per conversation sub-kind: conversations built, those that carry a passage inserted by
        retrieval (share of all), the distractor share and the later-turn share of those
        passages, and the share of conversations that are a later window."""
        out = {}
        for key in sorted(k for k in c if k.startswith("conv/")):
            kind = key[5:]
            n = int(c[key])
            m = int(c.get(f"conv_passage/{kind}", 0))
            out[kind] = {"samples": n, "with_passage": m, "passage_share": m / n if n else 0.0,
                         "distractor_share": c.get(f"conv_distractor/{kind}", 0) / m if m else 0.0,
                         "later_turn_share": c.get(f"conv_later/{kind}", 0) / m if m else 0.0,
                         "window_share": c.get(f"conv_window/{kind}", 0) / n if n else 0.0}
        return out

    def stats(self) -> dict[str, Any]:
        """The trainer's log record of the loader: per block the realised mixture (tokens by
        row kind, sub-kind and source, padding share, retrieval and hindsight counters, the
        realised passage shares of the conversations, failed draws, epoch counters), the
        coordinator's timing (seconds waiting, copying, planning, replaying) and the
        ``warnings`` of this run (planned shares the epoch caps cut by more than a third)."""
        blocks = [self._summary(i, bs) for i, bs in self._block_stats.items()]
        total_time: Counter = Counter()
        for bs in self._block_stats.values():
            total_time.update(bs["time"])
        return {"block": blocks[-1] if blocks else None, "blocks": blocks,
                "warnings": list(self._warnings.values()),
                "time_s": dict(total_time), "coordinator": dict(self._timing),
                "resets": dict(self._resets), "workers": {"n": len(self._streams_of),
                                                          "restarts": self.exec.restarts,
                                                          "pids": self.worker_pids()},
                "journal": {"path": str(self.journal_path), "blocks": len(self.records),
                            "chain": [[b.id, b.start, b.end, list(b.shape)] for b in self.chain[-8:]]}}


__all__ = ["Loader", "LoaderCfg", "BATCH_FIELDS", "batch_shapes", "batch_nbytes", "batch_views",
           "source_table", "source_kind", "library_fingerprint", "StreamState", "Plan", "Block",
           "build_chain", "encode_states", "decode_states", "fresh_states", "journal_settings",
           "StreamWorker", "WorkerInit", "ShardStore", "scan_new_shards", "stage_dict",
           "heldout_samples", "heldout_hash", "is_heldout", "HELDOUT_RULE", "HELDOUT_KINDS"]
