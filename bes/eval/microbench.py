"""Micro training bench for the CPU evidence gates E2, E4 and E5 (``docs/PLAN.md`` section 7).

One *arm* is a declarative training run of the micro model (``configs/micro/bench.py``): a name,
configuration overrides (``bes.config.apply_overrides`` strings), training-time switches of
``bes.model.model.make_consts``, a seed, a token budget and a data mixture. :func:`run_arm`
trains it, evaluates it at fixed token checkpoints on fixed held-out rows, saves its parameters
and appends one JSON record to ``<out>/results.jsonl``. :func:`run_queue` runs a list of arms
one after the other and skips the finished ones. :func:`e2_report` turns the records into the
E2 document with the pre-registered decision rule.

Data
----
Token pools are built once from the small samples in ``data_cache/samples`` and cached as numpy
files under ``data_cache/micro/pools`` (memory-mapped when read):

=========  ================================================================  ==============
domain     train / held-out source                                            segments
=========  ================================================================  ==============
``web``    ClimbMix ``shard_00000`` / ``shard_00997``                          documents
``chat``   smol-smoltalk ``train-00000-of-00004`` / ``test-00000-of-00001``,  conversations
           rendered with ``bes.data.chat_format``
``wiki``   Simple English Wikipedia: records ``[0, N - 2000)`` / the last 2000   documents
``probe*`` synthetic probe tasks of ``bes.eval.probes`` (rendered the same way)   conversations
=========  ================================================================  ==============

Rows hold ``T + 1`` tokens and follow ``bes.data.packing``: documents fill rows exactly and the
document that does not fit continues in the next row behind a fresh ``BOS``
(``packing.fill_base``); conversations are cut at the end of an assistant turn
(``bes.data.samples.fit``) and packed by best fit from a look-ahead buffer
(``packing.pack_best_fit``); ``packing.assemble_row`` gives ``tokens``, ``seg`` and the loss
weights ``w`` of ``bes.data.samples.loss_weights``. Each pool also stores the role code of every
token (for the assistant-only loss). Pool arrays: ``tokens uint16 [N, T+1]``, ``seg int16
[N, T]``, ``w float32 [N, T]``, ``roles int8 [N, T+1]``.

The training loader ``bes.data.loader.Loader`` imports, but the bench does not use it: its
mixture is planned by sub-kinds and registry weights (``bes.data.mixture``), it reads every
segment of a source (no held-out split) and it has no way to mix in the synthetic probe tasks;
the bench needs a fixed web/chat ratio, fixed held-out rows and the probe mixture. The bench
reuses the loader's pure packing functions instead, so rows have the same layout.

Batches and pairing
-------------------
Global row ``r`` of a run goes to the domain whose row count lags most behind its share
(smooth weighted round robin: a fixed, stratified pattern; with 4 rows per micro-batch and
75/25 every micro-batch holds 3 web rows and 1 chat row). The ``k``-th row of domain ``d`` is
pool row ``permute_index(k mod N_d, N_d, "micro-rows", seed, d, k // N_d)`` (a new permutation
for every pass over the pool). The batch of step ``s`` is therefore a pure function of the
pools, the mixture, the seed, the step and the shape: every arm with the same seed sees
bit-identical batches, and the initial parameters come from ``jax.random.key(seed)`` (the model
derives one key per parameter path, so the leaves two arms share start equal). Features are
``exact_features(tokens[..., :T], seg)`` of the batch.

Training and evaluation
-----------------------
``bes.model.model`` (``init_params``, ``make_consts``, ``loss_fn``), ``bes.optim.build`` with the
learning-rate transfer rule at the arm's real ``tokens_per_step`` and ``horizon_tokens``, and
``bes.train.train.make_train_step`` (gradient accumulation over the ``A`` micro-batches inside one
jitted step). The learning-rate factor, the Muon momentum and the weight-decay factor are those
of ``bes.train.schedule`` with progress ``p = tokens done / token budget``. At each checkpoint
(fractions ``Bench.eval_fracs`` of the budget) the bench measures on the first
``Bench.eval_rows`` rows of each held-out pool: the per-token loss on every evaluation domain
(``web``, ``chat`` = all weighted conversation tokens, ``assistant`` = assistant tokens only),
the model's ``tape_gain``, ``tape_mass`` and ``sink_mass``, and the same losses under
evaluation-time switches (``views``, by default for the reference arm: tape off, registers off,
sink off). Per-row sums are stored so the report can compute paired standard errors.

Graft arms
----------
An arm with a :class:`Graft` (kind ``graft``) is the micro-scale copy of the live run's switch
from ``final`` to ``v9`` (``scripts/graft_run.py``, ``docs/CHANGES.md`` 19.7): it trains its
base arm's configuration (same seed, same batches, same schedule) up to the fraction
``graft.frac`` of the budget, which must be an evaluation fraction, then carries every leaf the
two configurations share, with its optimiser slots, the update count and the step, into a fresh
train state of its own configuration of the same seed (``graft_run.graft_arrays``), checks that
the loss on held-out rows is unchanged, and trains on to the end at the same schedule positions
with the batches the base would have seen. The trunk is either a state kept by the base arm
(``run_arm(keep_states=True)``) or trained inside the graft arm (:func:`run_graft_arm`); the
record says which. ``scripts/micro_graft_report.py`` compares such arms with the base and the
target trained from the first step, paired by seed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from bes.config import (BOS, PAD, Cfg, DataCfg, DecodeCfg, ModelCfg, OptimCfg, RunCfg,
                        config_hash, load_config, to_dict)

REPO = Path(__file__).resolve().parents[2]
BENCH_CONFIG = REPO / "configs" / "micro" / "bench.py"
SAMPLES_DIR = REPO / "data_cache" / "samples"
POOL_ROOT = REPO / "data_cache" / "micro" / "pools"
POOL_VERSION = 1
RESULTS = "results.jsonl"
ROLE_ASSISTANT = 3

#: default training mixture: (domain, row share)
DEFAULT_MIXTURE: tuple[tuple[str, float], ...] = (("web", 0.75), ("chat", 0.25))
#: evaluation-time switch sets of the reference arm
REF_VIEWS: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    ("tape_off", (("cand_in", 0.0), ("cand_out", 0.0))),
    ("reg_off", (("reg", 0.0),)),
    ("sink_off", (("sink", 0.0),)),
)
#: kinds of arms (they decide how the report judges them; ``add_new``: a new component added to
#: the reference, such as the v9 token pages; ``graft``: an arm with a :class:`Graft`, reported by
#: ``scripts/micro_graft_report.py``, not by the E2 rule)
KINDS = ("reference", "replicate", "remove_evidence", "remove_new", "setting", "probe", "add_new",
         "graft")

#: sample files (relative to the samples directory) and the column read
SOURCES: dict[str, dict[str, Any]] = {
    "web": {"train": ("karpathy__climbmix-400b-shuffle/shard_00000.parquet", "text", 0, None),
            "heldout": ("karpathy__climbmix-400b-shuffle/shard_00997.parquet", "text", 0, None)},
    "chat": {"train": ("HuggingFaceTB__smol-smoltalk/data/train-00000-of-00004.parquet", "messages", 0, None),
             "heldout": ("HuggingFaceTB__smol-smoltalk/data/test-00000-of-00001.parquet", "messages", 0, None)},
    "wiki": {"train": ("wikimedia__wikipedia/20231101.simple/train-00000-of-00001.parquet", "text", 0, -2000),
             "heldout": ("wikimedia__wikipedia/20231101.simple/train-00000-of-00001.parquet", "text", -2000, None)},
}
MAX_DOC_CHARS = 40_000
MAX_DOC_TOKENS = 8_192


# ============================================================================ settings
@dataclass(frozen=True)
class Bench:
    """Settings shared by every arm of a queue (part of every arm's hash).

    config: base configuration file; seq_len: ``T`` (inputs per row); micro_rows: rows per
    micro-batch; tokens_per_step: tokens per optimiser step (``A = tokens_per_step /
    (micro_rows * seq_len)``); eval_fracs: evaluation checkpoints as fractions of the budget;
    eval_rows: held-out rows per domain; eval_domains: held-out pools evaluated (``chat`` also
    gives ``assistant``); eval_chunk: rows per evaluation forward pass; pool_rows: rows built for
    each training pool (a run that needs more starts another permuted pass); pool_root /
    samples_dir: where pools are cached and read from.
    """

    config: str = str(BENCH_CONFIG)
    seq_len: int = 512
    micro_rows: int = 4
    tokens_per_step: int = 4096
    eval_fracs: tuple[float, ...] = (0.25, 0.5, 1.0)
    eval_rows: int = 32
    eval_domains: tuple[str, ...] = ("web", "chat")
    eval_chunk: int = 8
    log_every: int = 10
    pool_rows: tuple[tuple[str, int], ...] = (("web", 8192), ("chat", 4096), ("wiki", 4096),
                                              ("probe", 4096))
    pool_root: str = str(POOL_ROOT)
    samples_dir: str = str(SAMPLES_DIR)

    def rows_for(self, domain: str) -> int:
        """Rows built for the training pool of ``domain`` (probe pools share ``probe``)."""
        table = dict(self.pool_rows)
        key = "probe" if domain.startswith("probe") else domain
        return int(table.get(key, 4096))

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["config"] = _rel(self.config)
        d.pop("pool_root")
        d.pop("samples_dir")
        return d


@dataclass(frozen=True)
class Arm:
    """One declarative training run.

    overrides: ``"section.field=value"`` strings applied after the bench's own settings;
    switches: ``(name, value)`` training-time switches of ``make_consts`` (also the base of every
    evaluation view); seed: data order and initial key; tokens: budget (0 = the queue's);
    kind: one of :data:`KINDS`; component: what the arm removes or changes (report);
    mixture: ``(domain, row share)``; views: ``(name, switches)`` evaluation-time switch sets
    (empty: :data:`REF_VIEWS` for the reference kind, none otherwise); graft: a :class:`Graft`
    (kind ``graft`` only): the arm trains ``graft.base`` up to ``graft.frac`` of the budget and
    continues as its own configuration (:func:`run_graft_arm`). An arm without a graft serialises
    without a ``graft`` key, so its dict and its :func:`spec_hash` are those of the bench before
    graft arms existed.
    """

    name: str
    overrides: tuple[str, ...] = ()
    switches: tuple[tuple[str, float], ...] = ()
    seed: int = 0
    tokens: int = 0
    kind: str = "setting"
    component: str = ""
    mixture: tuple[tuple[str, float], ...] = DEFAULT_MIXTURE
    views: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    note: str = ""
    graft: "Graft | None" = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"arm {self.name!r}: unknown kind {self.kind!r} (one of {KINDS})")
        if not self.mixture or any(s < 0 for _, s in self.mixture) or sum(s for _, s in self.mixture) <= 0:
            raise ValueError(f"arm {self.name!r}: the mixture needs positive shares")
        if (self.graft is not None) != (self.kind == "graft"):
            raise ValueError(f"arm {self.name!r}: kind 'graft' and a graft spec go together")
        if self.graft is not None:
            self.graft.check(self)

    def eval_views(self) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        if self.views:
            return self.views
        return REF_VIEWS if self.kind == "reference" else ()

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "overrides": list(self.overrides),
             "switches": [list(s) for s in self.switches], "seed": self.seed,
             "tokens": self.tokens, "kind": self.kind, "component": self.component,
             "mixture": [list(m) for m in self.mixture],
             "views": [[n, [list(s) for s in sw]] for n, sw in self.eval_views()],
             "note": self.note}
        if self.graft is not None:
            d["graft"] = self.graft.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Arm":
        return cls(name=d["name"], overrides=tuple(d.get("overrides", ())),
                   switches=tuple((str(a), float(b)) for a, b in d.get("switches", ())),
                   seed=int(d.get("seed", 0)), tokens=int(d.get("tokens", 0)),
                   kind=d.get("kind", "setting"), component=d.get("component", ""),
                   mixture=tuple((str(a), float(b)) for a, b in d.get("mixture", DEFAULT_MIXTURE)),
                   views=tuple((n, tuple((str(a), float(b)) for a, b in sw))
                               for n, sw in d.get("views", ())),
                   note=d.get("note", ""),
                   graft=Graft.from_dict(d["graft"]) if d.get("graft") else None)


@dataclass(frozen=True)
class Graft:
    """The trunk of a graft arm (``docs/E2_ABLATIONS.md`` does not report these arms; see
    ``scripts/micro_graft_report.py``).

    base: the arm whose configuration (overrides, switches) trains the first ``frac`` of the
    budget; it must have the graft arm's seed, mixture and budget, so the trunk is the base arm's
    own computation and the data order continues where the base stops. Its name is where a kept
    trunk state is looked up (``<out>/arms/<base>/states``, :func:`find_trunk`). frac: the graft
    point as a fraction of the budget; it must be one of ``Bench.eval_fracs`` (checked by
    :func:`graft_plan`). target: name of the arm that trains the graft arm's configuration from
    the first step with the same seed (a label for the report).
    """

    base: Arm
    frac: float
    target: str = ""

    def check(self, arm: Arm) -> None:
        b = self.base
        if b.graft is not None:
            raise ValueError(f"arm {arm.name!r}: the base of a graft cannot itself be a graft arm")
        if not 0.0 < float(self.frac) < 1.0:
            raise ValueError(f"arm {arm.name!r}: the graft fraction must lie strictly between 0 and 1")
        if int(b.seed) != int(arm.seed):
            raise ValueError(f"arm {arm.name!r}: seed {arm.seed} differs from its base {b.name!r} "
                             f"(seed {b.seed}); the trunk must be the base arm's run")
        if tuple(b.mixture) != tuple(arm.mixture):
            raise ValueError(f"arm {arm.name!r}: the mixture differs from its base {b.name!r}")
        if int(b.tokens) != int(arm.tokens):
            raise ValueError(f"arm {arm.name!r}: the budget differs from its base {b.name!r}")

    def to_dict(self) -> dict[str, Any]:
        base = {k: v for k, v in self.base.to_dict().items() if k not in ("note", "component")}
        return {"base": base, "frac": float(self.frac), "target": self.target}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Graft":
        return cls(base=Arm.from_dict(d["base"]), frac=float(d["frac"]), target=str(d.get("target", "")))


def _rel(path: str | os.PathLike) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def spec_hash(arm: Arm, tokens: int, bench: Bench) -> str:
    """Identity of a run: the arm without its labels, the budget, the bench settings and the
    text of the bench configuration file (edits of the files it imports, such as
    ``configs/micro/base.py`` or ``bes/config.py`` defaults, are not detected: use a new run
    directory after such a change)."""
    d = arm.to_dict()
    for label in ("note", "component"):
        d.pop(label)
    b = bench.to_dict()
    b.pop("log_every")                       # logging only
    cfg_file = Path(bench.config)
    text = hashlib.sha256(cfg_file.read_bytes()).hexdigest() if cfg_file.exists() else ""
    blob = json.dumps({"arm": d, "tokens": int(arm.tokens or tokens), "bench": b,
                       "config_text": text, "pool_version": POOL_VERSION}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def arm_config(arm: Arm, tokens: int, bench: Bench = Bench()) -> Cfg:
    """The full configuration of an arm: ``bench.config`` + the bench's shape and the arm's real
    token horizon (learning-rate transfer rule) + the arm's overrides (applied last)."""
    budget = int(arm.tokens or tokens)
    base = [f"data.seq_len={bench.seq_len}", f"data.seq_len_late={bench.seq_len}",
            f"model.max_seq={bench.seq_len}", f"optim.micro_seqs={bench.micro_rows}",
            f"optim.tokens_per_step={bench.tokens_per_step}",
            f"optim.horizon_tokens={float(budget)}", f"run.seed={arm.seed}",
            f"run.name={arm.name}"]
    return load_config(bench.config, base + list(arm.overrides))


def cfg_from_dict(d: Mapping[str, Any]) -> Cfg:
    """Inverse of ``bes.config.to_dict`` for the five sections (unknown fields are dropped)."""
    def make(cls, sub):
        names = {f.name for f in dataclasses.fields(cls)}
        vals = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sub.items() if k in names}
        return cls(**vals)
    return Cfg(model=make(ModelCfg, d["model"]), optim=make(OptimCfg, d["optim"]),
               data=make(DataCfg, d["data"]), run=make(RunCfg, d["run"]),
               decode=make(DecodeCfg, d.get("decode", {})))


# ============================================================================ sources
def iter_parquet(path: str | os.PathLike, column: str, start: int = 0, stop: int | None = None
                 ) -> Iterator[Any]:
    """Values of ``column`` for records ``[start, stop)`` of a parquet file (negative bounds
    count from the end); only the row groups that hold them are read."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    lo = start + n if start < 0 else start
    hi = n if stop is None else (stop + n if stop < 0 else min(stop, n))
    off = 0
    for g in range(pf.metadata.num_row_groups):
        size = pf.metadata.row_group(g).num_rows
        if off + size <= lo:
            off += size
            continue
        if off >= hi:
            break
        values = pf.read_row_group(g, columns=[column]).column(0).to_pylist()
        for i, v in enumerate(values):
            if lo <= off + i < hi:
                yield v
        off += size


def source_texts(domain: str, split: str, samples_dir: str | os.PathLike) -> Iterator[Any]:
    """Records of a sample source: strings (documents) or message lists (conversations)."""
    rel, column, start, stop = SOURCES[domain][split]
    for v in iter_parquet(Path(samples_dir) / rel, column, start, stop):
        if column == "text":
            if isinstance(v, str) and v.strip():
                yield v[:MAX_DOC_CHARS]
        else:
            yield v


def source_descriptor(domain: str, split: str, samples_dir: str | os.PathLike) -> dict:
    rel, column, start, stop = SOURCES[domain][split]
    path = Path(samples_dir) / rel
    size = path.stat().st_size if path.exists() else -1
    return {"file": rel, "column": column, "start": start, "stop": stop, "size": size}


# ============================================================================== rows
def _empty_rows(n: int, T: int) -> dict[str, np.ndarray]:
    return {"tokens": np.full((n, T + 1), PAD, np.uint16), "seg": np.zeros((n, T), np.int16),
            "w": np.zeros((n, T), np.float32), "roles": np.zeros((n, T + 1), np.int8)}


def _put_row(out: dict, r: int, pieces: Sequence[tuple[np.ndarray, np.ndarray]], T: int, rule) -> int:
    """Assemble ``pieces`` into row ``r`` of ``out``; returns the padding positions."""
    from bes.data import packing as P

    row = P.assemble_row(pieces, T, rule)
    out["tokens"][r] = row.tokens
    out["seg"][r] = row.seg
    out["w"][r] = row.w
    o = 0
    for ids, rl in pieces:
        out["roles"][r, o:o + len(ids)] = rl
        o += len(ids)
    return row.n_pad


def document_rows(texts: Iterable[str], tok, T: int, n_rows: int,
                  max_doc_tokens: int = MAX_DOC_TOKENS) -> tuple[dict[str, np.ndarray], dict]:
    """Rows of ``T + 1`` tokens filled with consecutive documents (``packing.fill_base``).

    Returns the arrays of the module docstring (fewer rows when the texts run out) and stats.
    """
    from bes.data import packing as P
    from bes.data.samples import WeightRule

    out = _empty_rows(n_rows, T)
    it = iter(texts)
    stats = {"docs": 0, "pad": 0}
    pending: list[list[int]] = []

    def next_doc():
        while True:
            if not pending:
                chunk = [t for _, t in zip(range(256), it)]
                if not chunk:
                    return None
                pending.extend(reversed(tok.encode_batch(chunk, num_threads=1)))
            ids = pending.pop()
            if ids:
                stats["docs"] += 1
                arr = np.asarray([BOS] + ids[:max_doc_tokens - 1], np.int32)
                return P.Carry(key=stats["docs"], ids=arr)

    carry = None
    rule = WeightRule()
    r = 0
    while r < n_rows:
        placed, carry = P.fill_base(T + 1, carry, next_doc)
        if not placed:
            break
        pieces = [(ids, np.zeros(len(ids), np.int8)) for _, ids in placed]
        stats["pad"] += _put_row(out, r, pieces, T, rule)
        r += 1
    return {k: v[:r] for k, v in out.items()}, {**stats, "rows": r}


def conversation_samples(convs: Iterable[Any], tok, T: int) -> Iterator[Any]:
    """``bes.data.samples.Sample`` of every renderable conversation, cut to ``T + 1`` tokens at
    an assistant-turn boundary (conversations that do not render or fit are skipped)."""
    from bes.data.samples import conversation_sample, fit

    for msgs in convs:
        try:
            sample = conversation_sample(list(msgs), tok)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        head, _ = fit(sample, T + 1)
        if head is not None:
            yield head


def sample_rows(samples: Iterable[Any], T: int, n_rows: int, lookahead: int = 32
                ) -> tuple[dict[str, np.ndarray], dict]:
    """Rows packed by best fit (``packing.pack_best_fit``) from a look-ahead buffer of samples
    (``bes.data.samples.Sample``, each at most ``T + 1`` tokens)."""
    from bes.data import packing as P
    from bes.data.samples import WeightRule

    out = _empty_rows(n_rows, T)
    it = iter(samples)
    stats = {"samples": 0, "pad": 0}

    def refill():
        for s in it:
            if len(s) <= T + 1:
                return s
        return None

    buffer = []
    for _ in range(lookahead):
        s = refill()
        if s is None:
            break
        buffer.append(s)
    rule = WeightRule()
    r = 0
    while r < n_rows and buffer:
        placed = P.pack_best_fit(buffer, T + 1, len, refill)
        if not placed:
            break
        stats["samples"] += len(placed)
        stats["pad"] += _put_row(out, r, [(s.ids, s.roles) for s in placed], T, rule)
        r += 1
    return {k: v[:r] for k, v in out.items()}, {**stats, "rows": r}


# ============================================================================= pools
@dataclass
class Pool:
    """Packed rows of one domain and split (arrays of the module docstring) and their meta."""

    name: str
    tokens: np.ndarray
    seg: np.ndarray
    w: np.ndarray
    roles: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return int(self.tokens.shape[0])

    def take(self, idx: Sequence[int] | np.ndarray) -> dict[str, np.ndarray]:
        """Rows ``idx`` as ``tokens int32 [n, T+1]``, ``seg int32 [n, T]``, ``w float32 [n, T]``,
        ``roles int8 [n, T+1]``."""
        idx = np.asarray(idx, np.int64)
        return {"tokens": np.asarray(self.tokens[idx], np.int32),
                "seg": np.asarray(self.seg[idx], np.int32),
                "w": np.asarray(self.w[idx], np.float32),
                "roles": np.asarray(self.roles[idx], np.int8)}


_ARRAYS = ("tokens", "seg", "w", "roles")
#: domain -> builder(split, T, n_rows, tok, samples_dir) -> (arrays, stats); and descriptors
_BUILDERS: dict[str, Callable[..., tuple[dict, dict]]] = {}
_DESCRIBE: dict[str, Callable[[str, str], dict]] = {}


def register_pool(domain: str, builder: Callable[..., tuple[dict, dict]],
                  describe: Callable[[str, str], dict]) -> None:
    """Make ``domain`` available to mixtures and evaluations (used by ``bes.eval.probes``)."""
    _BUILDERS[domain] = builder
    _DESCRIBE[domain] = describe


def _builtin_builder(domain: str):
    def build(split: str, T: int, n_rows: int, tok, samples_dir) -> tuple[dict, dict]:
        records = source_texts(domain, split, samples_dir)
        if SOURCES[domain][split][1] == "text":
            return document_rows(records, tok, T, n_rows)
        return sample_rows(conversation_samples(records, tok, T), T, n_rows)

    def describe(split: str, samples_dir: str) -> dict:
        return source_descriptor(domain, split, samples_dir)
    return build, describe


for _d in SOURCES:
    register_pool(_d, *_builtin_builder(_d))


def _resolve(domain: str) -> None:
    if domain not in _BUILDERS and domain.startswith("probe"):
        import bes.eval.probes  # noqa: F401  (registers the probe pools)
    if domain not in _BUILDERS:
        raise KeyError(f"unknown pool domain {domain!r} (known: {sorted(_BUILDERS)})")


def pool_key(domain: str, split: str, T: int, n_rows: int, tok, samples_dir) -> tuple[str, dict]:
    _resolve(domain)
    meta = {"domain": domain, "split": split, "T": int(T), "rows_requested": int(n_rows),
            "version": POOL_VERSION, "tokenizer": str(getattr(tok, "sha256", "")),
            "source": _DESCRIBE[domain](split, str(samples_dir))}
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]
    return f"{domain}_{split}_T{T}_n{n_rows}_{digest}", meta


def save_pool(directory: Path, arrays: Mapping[str, np.ndarray], meta: dict) -> None:
    """Write a pool atomically (temporary directory, then rename)."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = directory.parent / f".tmp-{directory.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for k in _ARRAYS:
        np.save(tmp / f"{k}.npy", np.ascontiguousarray(arrays[k]))
    (tmp / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1))
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def open_pool(directory: Path, name: str = "") -> Pool:
    meta = json.loads((directory / "meta.json").read_text())
    arrs = {k: np.load(directory / f"{k}.npy", mmap_mode="r") for k in _ARRAYS}
    return Pool(name=name or directory.name, meta=meta, **arrs)


def get_pool(domain: str, split: str, T: int, n_rows: int, tok, *,
             root: str | os.PathLike = POOL_ROOT, samples_dir: str | os.PathLike = SAMPLES_DIR,
             log: Callable[[str], None] | None = None) -> Pool:
    """The cached pool (built on first use). ``split`` is ``"train"`` or ``"heldout"``."""
    name, meta = pool_key(domain, split, T, n_rows, tok, samples_dir)
    directory = Path(root) / name
    if (directory / "meta.json").exists():
        try:
            pool = open_pool(directory, f"{domain}_{split}")
            if {k: pool.meta.get(k) for k in meta} == meta:
                return pool
        except (OSError, ValueError):
            pass
    t0 = time.time()
    arrays, stats = _BUILDERS[domain](split, T, n_rows, tok, samples_dir)
    if arrays["tokens"].shape[0] == 0:
        raise RuntimeError(f"pool {name}: no rows could be built (missing samples?)")
    meta = {**meta, "rows": int(arrays["tokens"].shape[0]), "stats": stats,
            "tokens_nonpad": int((arrays["tokens"] != PAD).sum()),
            "targets": float(arrays["w"].sum()), "build_seconds": round(time.time() - t0, 2)}
    save_pool(directory, arrays, meta)
    if log is not None:
        log(f"pool {name}: {meta['rows']} rows, {meta['tokens_nonpad']} tokens "
            f"({meta['build_seconds']} s)")
    return open_pool(directory, f"{domain}_{split}")


# =========================================================================== batches
def row_kinds(shares: Sequence[float], n: int) -> np.ndarray:
    """``int64[n]``: domain of each global row by smooth weighted round robin (row ``r`` goes to
    the domain with the largest ``share * (r + 1) - placed``, the first one on ties)."""
    s = np.asarray(shares, np.float64)
    s = s / s.sum()
    placed = np.zeros(len(s))
    out = np.empty(n, np.int64)
    for r in range(n):
        k = int(np.argmax(s * (r + 1) - placed))
        out[r] = k
        placed[k] += 1.0
    return out


class BatchMaker:
    """Paired batches of one run (module docstring): ``batch(step)`` is a pure function of the
    pools, the mixture, the seed, the step and ``shape = (A, B, T)``.

    pools: ``domain -> Pool`` (train split); mixture: ``(domain, share)``; n_steps: steps of the
    run (the row plan is fixed for them; later steps extend it the same way).
    """

    def __init__(self, pools: Mapping[str, Pool], mixture: Sequence[tuple[str, float]],
                 seed: int, shape: tuple[int, int, int], n_steps: int, tok=None):
        from bes.data.samples import permute_index

        self.pools = dict(pools)
        self.domains = [d for d, _ in mixture]
        self.shares = [float(s) for _, s in mixture]
        self.seed = int(seed)
        self.shape = tuple(int(x) for x in shape)
        self.tok = tok
        a, b, t = self.shape
        for d in self.domains:
            if self.pools[d].tokens.shape[1] != t + 1:
                raise ValueError(f"pool {d} has rows of {self.pools[d].tokens.shape[1]} tokens, "
                                 f"the shape needs {t + 1}")
        n = int(n_steps) * a * b
        kinds = row_kinds(self.shares, n)
        plan = np.zeros((n, 2), np.int64)
        plan[:, 0] = kinds
        self.passes: dict[str, int] = {}
        for i, d in enumerate(self.domains):
            where = np.flatnonzero(kinds == i)
            size = self.pools[d].n
            plan[where, 1] = [permute_index(k % size, size, "micro-rows", self.seed, d, k // size)
                              for k in range(len(where))]
            self.passes[d] = int(math.ceil(len(where) / size)) if len(where) else 0
        self.plan = plan

    def rows(self, step: int) -> np.ndarray:
        """``int64[A*B, 2]``: (domain index, pool row) of every row of the step."""
        a, b, _ = self.shape
        lo, hi = step * a * b, (step + 1) * a * b
        if hi > len(self.plan):
            raise IndexError(f"step {step} lies beyond the planned {len(self.plan) // (a * b)} steps")
        return self.plan[lo:hi]

    def raw(self, step: int) -> dict[str, np.ndarray]:
        """Rows of the step without features: ``tokens [A*B, T+1]``, ``seg``, ``w``, ``roles``."""
        rows = self.rows(step)
        parts = []
        for i, d in enumerate(self.domains):
            sel = np.flatnonzero(rows[:, 0] == i)
            if sel.size:
                parts.append((sel, self.pools[d].take(rows[sel, 1])))
        out = {}
        n = len(rows)
        for k in _ARRAYS:
            first = parts[0][1][k]
            arr = np.empty((n,) + first.shape[1:], first.dtype)
            for sel, got in parts:
                arr[sel] = got[k]
            out[k] = arr
        return out

    def batch(self, step: int) -> dict[str, np.ndarray]:
        """SPEC section 3 batch ``[A, B, ...]`` of the step (features included)."""
        a, b, t = self.shape
        r = self.raw(step)
        return with_features(r, self.tok, (a, b))

    def token_shares(self, n_steps: int) -> dict[str, float]:
        """Share of weighted target tokens per domain over the first ``n_steps`` steps."""
        a, b, _ = self.shape
        rows = self.plan[: n_steps * a * b]
        tot = {}
        for i, d in enumerate(self.domains):
            sel = rows[rows[:, 0] == i, 1]
            w = np.asarray(self.pools[d].w)
            tot[d] = float(w[sel].sum()) if sel.size else 0.0
        s = sum(tot.values()) or 1.0
        return {d: v / s for d, v in tot.items()}


def with_features(raw: Mapping[str, np.ndarray], tok, lead: tuple[int, ...] | None = None
                  ) -> dict[str, np.ndarray]:
    """Batch dict with the exact-layer features of ``raw`` (``tokens [n, T+1]``, ``seg``, ``w``);
    reshaped to ``lead + (...)`` when ``lead`` is given (e.g. ``(A, B)``)."""
    from bes.exact.features import exact_features

    tokens = np.asarray(raw["tokens"], np.int32)
    n, t1 = tokens.shape
    seg = np.asarray(raw["seg"], np.int32)
    feats = exact_features(tokens[:, : t1 - 1], seg, tok)
    out = {"tokens": tokens, "w": np.asarray(raw["w"], np.float32), "seg": seg, **feats}
    if lead is not None:
        out = {k: v.reshape(tuple(lead) + v.shape[1:]) for k, v in out.items()}
    return out


# ========================================================================= evaluation
@dataclass
class EvalSet:
    """Fixed held-out rows of one domain with features (``batch``: ``[n, ...]`` arrays plus
    ``w_asst``, the weights of assistant targets)."""

    domain: str
    batch: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(self.batch["tokens"].shape[0])


def eval_sets(domains: Sequence[str], n_rows: int, T: int, tok, bench: Bench = Bench(),
              log: Callable[[str], None] | None = None) -> list[EvalSet]:
    """The first ``n_rows`` rows of each held-out pool (the same rows for every arm)."""
    out = []
    for d in domains:
        pool = get_pool(d, "heldout", T, max(64, n_rows), tok, root=bench.pool_root,
                        samples_dir=bench.samples_dir, log=log)
        raw = pool.take(np.arange(min(n_rows, pool.n)))
        b = with_features(raw, tok)
        b["w_asst"] = (raw["w"] * (raw["roles"][:, 1:] == ROLE_ASSISTANT)).astype(np.float32)
        out.append(EvalSet(d, b))
    return out


def _eval_rows_fn(cfg):
    """Jitted ``(params, consts, batch) -> per-row sums`` (one compilation per arm)."""
    import jax
    import jax.numpy as jnp

    from bes.model import model as M

    mcfg = cfg.model

    def fn(params, consts, batch):
        feats, y, w = M.split_batch(batch)
        h, aux = M.forward(params, consts, feats, mcfg)
        b, t, _ = h.shape
        ce, ce_base, mass = M.token_ce(params, consts, h, feats, y, mcfg, parts=True)
        ce, ce_base, mass = (x.reshape(b, t) for x in (ce, ce_base, mass))
        wa = jnp.asarray(batch["w_asst"], jnp.float32)
        return {"ce": jnp.sum(w * ce, -1), "w": jnp.sum(w, -1), "ce_a": jnp.sum(wa * ce, -1),
                "w_a": jnp.sum(wa, -1), "gain": jnp.sum(w * (ce_base - ce), -1),
                "mass": jnp.sum(w * mass, -1), "sink": aux["sink_mass"]}
    return jax.jit(fn)


def evaluate(fn, params, consts, sets: Sequence[EvalSet], chunk: int = 8) -> dict[str, dict]:
    """Per-domain losses (``loss``, per-row ``ce`` / ``w`` sums, ``tape_gain``, ``tape_mass``,
    ``sink_mass``); a ``chat`` set also gives ``assistant``."""
    out: dict[str, dict] = {}
    for es in sets:
        acc: dict[str, list] = {}
        sinks = []
        for lo in range(0, es.n, chunk):
            part = {k: v[lo:lo + chunk] for k, v in es.batch.items()}
            keep = part["tokens"].shape[0]
            if keep < chunk:                          # pad to the compiled shape, weights 0
                part = {k: np.concatenate([v, np.repeat(v[-1:], chunk - keep, 0)]) for k, v in part.items()}
                for k in ("w", "w_asst"):
                    part[k][keep:] = 0.0
            got = {k: np.asarray(v) for k, v in fn(params, consts, part).items()}
            sinks.append(got.pop("sink"))
            for k, v in got.items():
                acc.setdefault(k, []).append(v[:keep])
        r = {k: np.concatenate(v).astype(np.float64) for k, v in acc.items()}
        wsum = max(float(r["w"].sum()), 1e-9)
        out[es.domain] = {"loss": float(r["ce"].sum()) / wsum, "ce": r["ce"].tolist(),
                          "w": r["w"].tolist(), "tape_gain": float(r["gain"].sum()) / wsum,
                          "tape_mass": float(r["mass"].sum()) / wsum,
                          "sink_mass": np.mean(np.stack(sinks), 0).astype(float).tolist()}
        if float(r["w_a"].sum()) > 0:
            key = "assistant" if es.domain == "chat" else f"{es.domain}_assistant"
            out[key] = {"loss": float(r["ce_a"].sum()) / float(r["w_a"].sum()),
                        "ce": r["ce_a"].tolist(), "w": r["w_a"].tolist()}
    return out


# ============================================================================ helpers
def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (``getrusage``)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def flops_per_token(cfg: Cfg) -> dict[str, float]:
    """Analytic matmul FLOPs per token (forward; training = 3 x forward + one more core forward
    with ``remat``). Attention scores count the causal half of a row of ``data.seq_len``. The
    v9 token neurons count their two projections and the ``rho`` key products and value sums,
    the attention gate its ``d x H`` logits; token gains and selective attention are
    element-wise (no matmul)."""
    m = cfg.model
    t = cfg.data.seq_len
    d, nh, hd = m.d_model, m.n_heads, m.head_dim
    dc = hd - m.bias_dims
    block = 2 * (2 * d * nh * dc + d * nh * hd + nh * hd * d + 2 * d * m.d_ffn)
    if m.tok_neurons:
        block += 2 * (2 * d * m.tok_dim + 2 * m.tok_neurons * m.tok_dim)
    if m.attn_gate:
        block += 2 * d * nh
    attn = 2 * 2 * nh * hd * (t + 1) / 2
    core = m.n_loops * m.n_blocks * (block + attn)
    head = 2 * (d * m.head_rank + m.head_rank * m.vocab_size)
    extra = (2 * m.cand_dim * d if m.use_cand_in else 0) + (2 * d * m.cand_dim if m.use_cand_out else 0)
    fwd = core + head + extra
    return {"forward": float(fwd), "train": float(3 * fwd + (core if m.remat else 0))}


def append_jsonl(path: Path, rec: Mapping[str, Any]) -> None:
    """Append one JSON line and fsync it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_default(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Records of a JSON-lines file (a torn or invalid line is skipped)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def finished(out: str | os.PathLike, arm: Arm, tokens: int, bench: Bench = Bench()) -> dict | None:
    """The finished record of this arm (same spec hash) in ``<out>/results.jsonl``, if any."""
    h = spec_hash(arm, tokens, bench)
    for rec in reversed(read_jsonl(Path(out) / RESULTS)):
        if rec.get("spec_hash") == h and rec.get("status") == "done":
            return rec
    return None


# ============================================================================ training
def train_pools(arm: Arm, T: int, tok, bench: Bench, log=None) -> dict[str, Pool]:
    return {d: get_pool(d, "train", T, bench.rows_for(d), tok, root=bench.pool_root,
                        samples_dir=bench.samples_dir, log=log) for d, _ in arm.mixture}


def run_arm(arm: Arm, out: str | os.PathLike, tokens: int, bench: Bench = Bench(), *,
            log: Callable[[str], None] = print, tok=None, max_steps: int | None = None,
            save_every_eval: bool = True,
            batch_hook: Callable[[int, Mapping[str, np.ndarray]], None] | None = None,
            keep_states: bool = False, trunk: str = "auto",
            trunk_dirs: Sequence[str | os.PathLike] = ()) -> dict:
    """Train and evaluate one arm; append its record to ``<out>/results.jsonl`` and return it.

    The parameters go to ``<out>/arms/<name>/params`` (``bes.train.checkpoint``). At every
    evaluation checkpoint before the last, the train state is saved to ``.../state`` so that a
    killed run resumes there (same batches, same compiled step: the continuation equals an
    uninterrupted run). max_steps: stop early (tests); the record then has ``status="partial"``.
    batch_hook: ``batch_hook(step, batch)`` before every step (tests of the pairing).
    keep_states: also keep the train state of every evaluation checkpoint before the last in
    ``.../states`` (never pruned or removed), for graft arms of this base (:func:`find_trunk`);
    it does not change the run. An arm with a :class:`Graft` is run by :func:`run_graft_arm`
    (``trunk``, ``trunk_dirs``: see there; ``keep_states`` does not apply to it).
    """
    if arm.graft is not None:
        return run_graft_arm(arm, out, tokens, bench, log=log, tok=tok, max_steps=max_steps,
                             save_every_eval=save_every_eval, batch_hook=batch_hook, trunk=trunk,
                             trunk_dirs=trunk_dirs)
    import jax
    import jax.numpy as jnp

    from bes.model import model as M
    from bes.optim import build as build_opt
    from bes.optim.build import batch_factor, data_factor, group_lrs
    from bes.train import checkpoint as C
    from bes.train import schedule as S
    from bes.train.train import TrainState, abstract, make_train_step

    out = Path(out)
    if tok is None:
        from bes.tokenizer import load as load_tok
        tok = load_tok()
    budget = int(arm.tokens or tokens)
    cfg = arm_config(arm, budget, bench)
    shape = S.micro_shape(cfg, cfg.data.seq_len)
    a_, b_, t_ = shape
    n_steps = max(1, budget // cfg.optim.tokens_per_step)
    eval_steps = sorted({min(n_steps, max(1, int(round(f * n_steps)))) for f in bench.eval_fracs})
    h = spec_hash(arm, budget, bench)
    arm_dir = out / "arms" / arm.name
    state_dir = arm_dir / "state"
    log(f"[{arm.name}] {n_steps} steps of {cfg.optim.tokens_per_step} tokens, shape {shape}, "
        f"seed {arm.seed}, evaluations at steps {eval_steps}")

    t_setup = time.time()
    pools = train_pools(arm, t_, tok, bench, log)
    maker = BatchMaker(pools, arm.mixture, arm.seed, shape, n_steps, tok)
    sets = eval_sets(bench.eval_domains, bench.eval_rows, t_, tok, bench, log)
    params = M.init_params(cfg.model, jax.random.key(arm.seed))
    train_sw = dict(arm.switches)
    consts = M.make_consts(cfg.model, **train_sw)          # evaluation copy (the state's is donated)
    opt = build_opt(cfg, params)
    state = TrainState(step=jnp.zeros((), jnp.int32), params=params, opt_state=opt.init(params),
                       consts=M.make_consts(cfg.model, **train_sw))
    step_fn = make_train_step(M.loss_fn, opt, cfg)
    eval_fn = _eval_rows_fn(cfg)
    views = [(name, M.make_consts(cfg.model, **{**train_sw, **dict(sw)}))
             for name, sw in arm.eval_views()]
    n_params = int(sum(np.size(x) for x in jax.tree.leaves(params)))

    start, curve, train_log = 0, [], []
    timing = {"train_s": 0.0, "eval_s": 0.0, "first_step_s": 0.0, "timed_steps": 0,
              "setup_s": 0.0, "resumes": 0}
    skipped = 0
    if state_dir.exists():
        try:
            got = C.restore(state_dir, {"state": abstract(state)}, config_hash=h)
        except C.CheckpointError as e:
            log(f"[{arm.name}] state checkpoint not usable ({e}); starting over")
            got = None
        if got is not None:
            tree, man, _ = got
            state = tree["state"]
            meta = man["meta"]
            start, curve, train_log = int(meta["step"]), meta["curve"], meta["train_log"]
            timing.update(meta["timing"])
            timing["resumes"] += 1
            skipped = int(meta.get("skipped", 0))
            log(f"[{arm.name}] resumed at step {start}")
        else:
            shutil.rmtree(state_dir, ignore_errors=True)
    timing["setup_s"] += time.time() - t_setup

    stop = n_steps if max_steps is None else min(n_steps, max_steps)
    tps = cfg.optim.tokens_per_step
    for s in range(start, stop):
        p = s / n_steps
        lr = S.lr_factor(p, cfg.optim)
        mom = S.momentum_at(s, p, cfg)
        wd = S.wd_scale_at(p, cfg)
        batch = maker.batch(s)
        if batch_hook is not None:
            batch_hook(s, batch)
        t0 = time.time()
        state, metrics = step_fn(state, batch, np.float32(lr), np.float32(mom), np.float32(wd))
        loss = float(metrics["loss"])
        dt = time.time() - t0
        if s == start:
            timing["first_step_s"] = dt if start == 0 else timing["first_step_s"]
        else:
            timing["train_s"] += dt
            timing["timed_steps"] += 1
        skipped += int(metrics["skipped"])
        if s % bench.log_every == 0 or s == stop - 1:
            train_log.append([s, loss, round(lr, 6)])
            log(f"[{arm.name}] step {s:5d}/{n_steps} loss {loss:.4f} lr {lr:.3f} "
                f"({dt:.2f} s/step)")
        if (s + 1) in eval_steps:
            te = time.time()
            point = {"step": s + 1, "tokens": (s + 1) * tps, "train_loss": loss,
                     "eval": evaluate(eval_fn, state.params, consts, sets, bench.eval_chunk),
                     "views": {name: evaluate(eval_fn, state.params, c, sets, bench.eval_chunk)
                               for name, c in views}}
            timing["eval_s"] += time.time() - te
            curve.append(point)
            msg = " ".join(f"{d} {v['loss']:.4f}" for d, v in point["eval"].items())
            log(f"[{arm.name}] eval at {point['tokens']} tokens: {msg}")
            if keep_states and s + 1 < n_steps:
                C.save(arm_dir / STATES, {"state": state}, step=s + 1, config_hash=h, keep=0,
                       meta={"step": s + 1, "curve": curve, "train_log": train_log,
                             "timing": timing, "skipped": skipped})
            if save_every_eval and s + 1 < n_steps:
                C.save(state_dir, {"state": state}, step=s + 1, config_hash=h, keep=1,
                       meta={"step": s + 1, "curve": curve, "train_log": train_log,
                             "timing": timing, "skipped": skipped})
    done = stop == n_steps
    params_dir = arm_dir / "params"
    if done:
        shutil.rmtree(params_dir, ignore_errors=True)
        C.save(params_dir, {"params": state.params}, step=n_steps, config_hash=config_hash(cfg),
               keep=1, meta={"arm": arm.to_dict(), "tokens": budget, "spec_hash": h})
    timed = max(1, timing["timed_steps"])
    sec_step = timing["train_s"] / timed if timing["timed_steps"] else timing["first_step_s"]
    rec = {"type": "arm", "status": "done" if done else "partial", "name": arm.name,
           "arm": arm.to_dict(), "spec_hash": h, "tokens": budget, "steps": n_steps,
           "steps_run": stop, "shape": list(shape), "tokens_per_step": tps,
           "config": to_dict(cfg), "config_hash": config_hash(cfg), "bench": bench.to_dict(),
           "lrs": group_lrs(cfg), "batch_factor": batch_factor(cfg), "data_factor": data_factor(cfg),
           "n_params": n_params, "flops_per_token": flops_per_token(cfg),
           "timing": {**timing, "sec_per_step": sec_step, "sec_per_mtok": sec_step / tps * 1e6},
           "peak_rss_mb": peak_rss_mb(), "skipped_steps": skipped,
           "data": {"token_shares": maker.token_shares(n_steps), "passes": maker.passes,
                    "pools": {d: p.meta.get("rows") for d, p in pools.items()}},
           "curve": curve, "train_log": train_log,
           "params_dir": _rel(params_dir) if done else None,
           "host": {"cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "omp_threads": os.environ.get("OMP_NUM_THREADS")},
           "finished": time.time()}
    append_jsonl(out / RESULTS, rec)
    if done:
        shutil.rmtree(state_dir, ignore_errors=True)
    log(f"[{arm.name}] {rec['status']}: {sec_step:.2f} s/step, {rec['timing']['sec_per_mtok']:.0f} "
        f"s per 1M tokens, peak {rec['peak_rss_mb']:.0f} MB")
    return rec


def load_arm(out: str | os.PathLike, name: str, records: Sequence[dict] | None = None):
    """``(record, cfg, params)`` of a finished arm of ``<out>`` (the newest record of ``name``)."""
    import jax

    from bes.model import model as M
    from bes.train import checkpoint as C
    from bes.train.train import abstract

    recs = list(records) if records is not None else read_jsonl(Path(out) / RESULTS)
    rec = next((r for r in reversed(recs) if r.get("name") == name and r.get("status") == "done"), None)
    if rec is None:
        raise FileNotFoundError(f"no finished arm {name!r} in {out}")
    cfg = cfg_from_dict(rec["config"])
    template = {"params": abstract(M.init_params(cfg.model, jax.random.key(0)))}
    path = Path(out) / "arms" / name / "params"
    got = C.restore(path, template, force=True)
    if got is None:
        raise FileNotFoundError(f"no parameters for arm {name!r} in {path}")
    return rec, cfg, got[0]["params"]


# ============================================================================== grafts
#: directory inside ``<out>/arms/<name>/`` where ``run_arm(keep_states=True)`` keeps its states
STATES = "states"
#: relative tolerance of the function check at the graft point, the rule of
#: ``scripts/graft_run.py`` ``function_check``: ``|after - before| <= 1e-6 max(1, |before|)``
GRAFT_TOL = 1e-6
#: how a graft arm gets its trunk: a kept state of its base if one is found, else trained in the
#: graft arm (``auto``); always trained (``train``); only loaded, never trained (``load``)
TRUNK_MODES = ("auto", "train", "load")
#: optimiser fields that only act on a group; they may differ between base and target when the
#: base has no leaf in that group (the target's new tables train with their own rates)
_GROUP_FIELDS = {"gain": ("lr_gain", "gain_decay"), "page": ("lr_page", "page_decay")}


def _graft_run():
    """``scripts/graft_run.py``, the graft of the live run, imported from its file (the bench
    reuses its ``graft_arrays``); ``sys.path`` is restored after the import."""
    import importlib.util
    import sys

    mod = sys.modules.get("graft_run")
    if mod is None or not hasattr(mod, "graft_arrays"):
        spec = importlib.util.spec_from_file_location("graft_run", REPO / "scripts" / "graft_run.py")
        mod = importlib.util.module_from_spec(spec)
        saved = list(sys.path)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        sys.modules["graft_run"] = mod
    return mod


@dataclass(frozen=True)
class GraftPlan:
    """The checked layout of a graft arm at one budget (:func:`graft_plan`): both
    configurations, the step shape, the steps, the evaluation steps, the graft step, the base
    arm's spec hash and the parameter paths only the target has."""

    budget: int
    base_cfg: Cfg
    cfg: Cfg
    shape: tuple[int, int, int]
    n_steps: int
    eval_steps: tuple[int, ...]
    step: int
    base_hash: str
    new_params: tuple[str, ...]


def _param_shapes(cfg: Cfg) -> dict[str, tuple[tuple[int, ...], str]]:
    """``{path: (shape, dtype)}`` of the parameter tree of ``cfg`` (abstract: nothing allocated)."""
    import jax

    from bes.model import model as M
    from bes.optim.build import path_str

    tree = jax.eval_shape(lambda: M.init_params(cfg.model, jax.random.key(0)))
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {path_str(kp): (tuple(int(d) for d in x.shape), str(x.dtype)) for kp, x in flat}


def graft_plan(arm: Arm, tokens: int, bench: Bench = Bench(), *, shapes: bool = True) -> GraftPlan:
    """Check that ``arm`` (with a :class:`Graft`) can run at this budget and bench; its layout.

    Refused with ``ValueError``: a graft fraction that is not one of ``bench.eval_fracs``;
    evaluation fractions that do not fall on distinct steps (budget too small); a graft step
    outside ``(0, n_steps)``; base and target with another step shape; a parameter of the base
    that the target lacks or has with another shape or dtype; optimiser settings that differ
    (other than the rate and decay of a group the base has no leaf in), and with them any
    difference of the schedule: learning-rate factor, momentum or decay factor at any step. The
    graft changes the model only. shapes=False skips the parameter trees (no JAX import; the
    group check then assumes the base trains every group but ``gain`` and ``page``, and
    ``new_params`` is empty): for a parent process that only starts runs."""
    from bes.train import schedule as S

    g = arm.graft
    if g is None:
        raise ValueError(f"arm {arm.name!r} has no graft")
    budget = int(arm.tokens or tokens)
    base_cfg, cfg = arm_config(g.base, budget, bench), arm_config(arm, budget, bench)
    shape = S.micro_shape(cfg, cfg.data.seq_len)
    if (S.micro_shape(base_cfg, base_cfg.data.seq_len) != shape
            or base_cfg.optim.tokens_per_step != cfg.optim.tokens_per_step):
        raise ValueError(f"arm {arm.name!r}: base and target train with different step shapes")
    n_steps = max(1, budget // cfg.optim.tokens_per_step)
    fracs = [float(f) for f in bench.eval_fracs]
    if not any(abs(f - float(g.frac)) <= 1e-9 for f in fracs):
        raise ValueError(f"arm {arm.name!r}: the graft fraction {g.frac:g} is not an evaluation fraction "
                         f"(eval_fracs {tuple(fracs)}); add it to the evaluation fractions (--eval-fracs)")
    steps = [min(n_steps, max(1, int(round(f * n_steps)))) for f in fracs]
    if len(set(steps)) != len(steps):
        raise ValueError(f"arm {arm.name!r}: the evaluation fractions {tuple(fracs)} fall on the steps "
                         f"{steps} of {n_steps}; the budget is too small to tell them apart")
    step = min(n_steps, max(1, int(round(float(g.frac) * n_steps))))
    if not 0 < step < n_steps:
        raise ValueError(f"arm {arm.name!r}: graft step {step} is not inside the run of {n_steps} steps")
    shapes_b, shapes_t = (_param_shapes(base_cfg), _param_shapes(cfg)) if shapes else ({}, {})
    bad = [p for p, sd in shapes_b.items() if shapes_t.get(p) != sd]
    if bad:
        raise ValueError(f"arm {arm.name!r}: parameters of the base that the target does not have with "
                         f"the same shape and dtype: {bad[:6]}")
    groups_b: set[str] = set()
    if shapes:
        from bes.roles import role_of
        groups_b = {getattr(role_of(p), "group", role_of(p)) for p in shapes_b}
    free = {f for grp, fields in _GROUP_FIELDS.items() if grp not in groups_b for f in fields}
    ob, ot = dataclasses.asdict(base_cfg.optim), dataclasses.asdict(cfg.optim)
    diff = sorted(k for k in ob if k not in free and ob[k] != ot.get(k))
    if diff:
        raise ValueError(f"arm {arm.name!r}: optimiser settings differ from its base: {diff}")
    for s in range(n_steps):
        p = s / n_steps
        a = (S.lr_factor(p, base_cfg.optim), S.momentum_at(s, p, base_cfg), S.wd_scale_at(p, base_cfg))
        b = (S.lr_factor(p, cfg.optim), S.momentum_at(s, p, cfg), S.wd_scale_at(p, cfg))
        if a != b:
            raise ValueError(f"arm {arm.name!r}: the schedule differs from its base at step {s}: {a} vs {b}")
    return GraftPlan(budget=budget, base_cfg=base_cfg, cfg=cfg, shape=tuple(shape), n_steps=n_steps,
                     eval_steps=tuple(sorted(set(steps))), step=step,
                     base_hash=spec_hash(g.base, budget, bench),
                     new_params=tuple(sorted(p for p in shapes_t if p not in shapes_b)))


def find_trunk(base: Arm, step: int, base_hash: str, dirs: Sequence[str | os.PathLike]
               ) -> tuple[Any, list[dict]]:
    """A state of ``base`` at ``step`` kept by ``run_arm(keep_states=True)`` under the spec hash
    ``base_hash``, looked up in ``<d>/arms/<base>/states`` and ``<d>/<base>/arms/<base>/states``
    (one ``--out`` sub-directory per arm) for every run directory ``d``, newest first. Returns
    ``(CkptInfo or None, rejected)``; ``rejected`` lists the candidates at that step that were
    written under another hash or are unreadable."""
    from bes.train import checkpoint as C

    rejected: list[dict] = []
    seen: set[str] = set()
    for d in dirs:
        for root in (Path(d) / "arms" / base.name / STATES,
                     Path(d) / base.name / "arms" / base.name / STATES):
            key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            for info in reversed(C.list_checkpoints(root)):
                if info.step != int(step):
                    continue
                try:
                    man = C.read_manifest(info.path)
                except C.CheckpointError as e:
                    rejected.append({"checkpoint": _rel(info.path), "reason": str(e)})
                    continue
                if man.get("config_hash") == base_hash:
                    return info, rejected
                rejected.append({"checkpoint": _rel(info.path),
                                 "reason": f"written under spec hash {man.get('config_hash')}, not {base_hash}"})
    return None, rejected


class _Trainee:
    """One configuration of a graft arm (the trunk's or the target's): its fresh train state
    exactly as :func:`run_arm` builds that of an arm with this configuration and seed, its
    optimiser, compiled step, evaluation function, consts, views and loss function."""

    def __init__(self, arm: Arm, cfg: Cfg, views: Sequence[tuple[str, Any]] = ()):
        self.arm, self.cfg, self.views_spec = arm, cfg, tuple(views)
        self.switches = dict(arm.switches)
        self.opt = None
        self.used = False
        self._fns = None

    def fresh_state(self):
        import jax
        import jax.numpy as jnp

        from bes.model import model as M
        from bes.optim import build as build_opt
        from bes.train.train import TrainState

        params = M.init_params(self.cfg.model, jax.random.key(self.arm.seed))
        opt = build_opt(self.cfg, params)
        if self.opt is None:
            self.opt = opt
        return TrainState(step=jnp.zeros((), jnp.int32), params=params, opt_state=opt.init(params),
                          consts=M.make_consts(self.cfg.model, **self.switches))

    def fns(self):
        """``(step_fn, eval_fn, consts, views, loss_fn)``, built once."""
        if self._fns is None:
            import jax

            from bes.model import model as M
            from bes.train.train import make_train_step

            if self.opt is None:
                self.fresh_state()
            mcfg = self.cfg.model
            views = [(name, M.make_consts(mcfg, **{**self.switches, **dict(sw)}))
                     for name, sw in self.views_spec]
            self._fns = (make_train_step(M.loss_fn, self.opt, self.cfg), _eval_rows_fn(self.cfg),
                         M.make_consts(mcfg, **self.switches), views,
                         jax.jit(lambda p, c, b: M.loss_fn(p, c, b, mcfg)[0]))
        return self._fns


def _restore_graft_state(state_dir: Path, h: str, models: Mapping[str, _Trainee],
                         log: Callable[[str], None], name: str):
    """``(phase, state, meta, info)`` of the newest loadable state checkpoint of a graft arm (its
    template is the trunk's or the target's tree, by the checkpoint's ``phase``), or None."""
    from bes.train import checkpoint as C
    from bes.train.train import abstract

    for info in reversed(C.list_checkpoints(state_dir)):
        try:
            man = C.read_manifest(info.path)
        except C.CheckpointError as e:
            log(f"[{name}] skipping unreadable state checkpoint {info.name} ({e})")
            continue
        phase = (man.get("meta") or {}).get("phase")
        if man.get("config_hash") != h or phase not in models:
            log(f"[{name}] state checkpoint {info.name} not usable (hash {man.get('config_hash')}, "
                f"phase {phase}); starting over")
            return None
        try:
            tree, man = C.load(info, {"state": abstract(models[phase].fresh_state())}, config_hash=h)
        except C.CorruptCheckpointError as e:
            log(f"[{name}] skipping corrupt state checkpoint {info.name} ({e})")
            continue
        return phase, tree["state"], man["meta"], info
    return None


def run_graft_arm(arm: Arm, out: str | os.PathLike, tokens: int, bench: Bench = Bench(), *,
                  log: Callable[[str], None] = print, tok=None, max_steps: int | None = None,
                  save_every_eval: bool = True,
                  batch_hook: Callable[[int, Mapping[str, np.ndarray]], None] | None = None,
                  trunk: str = "auto", trunk_dirs: Sequence[str | os.PathLike] = ()) -> dict:
    """Train a graft arm (``arm.graft``) and append its record to ``<out>/results.jsonl``.

    Trunk, steps ``[0, g)`` (``g`` = :func:`graft_plan` ``.step``, an evaluation step): the base
    arm's own computation, i.e. its configuration, initial parameters (``jax.random.key(seed)``),
    optimiser, compiled step, schedule and batches. ``trunk="auto"`` first looks for a kept state
    of the base at step ``g`` with the base arm's spec hash (:func:`find_trunk` in ``out`` and
    ``trunk_dirs``) and loads it with the base's curve and train log up to ``g``; without one it
    trains the trunk. ``"train"`` always trains it, ``"load"`` refuses to. ``record["graft"]
    ["trunk"]["source"]`` is ``"loaded"`` or ``"trained"``.

    Graft, after the evaluation at step ``g``: ``scripts/graft_run.py`` ``graft_arrays`` copies
    every leaf of the trunk's state checkpoint (parameters, optimiser slots, update count, step)
    into a fresh train state of the arm's own configuration (the target) of the same seed; the
    target-only leaves and their slots stay at that fresh state's values, the consts are the
    target's. Function check: the training loss (``bes.model.model.loss_fn``) of the first
    ``bench.eval_chunk`` rows of the first held-out set, and the held-out loss of every evaluated
    domain, before (trunk) and after (grafted) must agree within :data:`GRAFT_TOL` relative;
    otherwise a ``status="failed"`` record is appended and ``RuntimeError`` raised.

    Target, steps ``[g, n)``: the arm's configuration at the same schedule positions
    (``p = s / n``) and with the same batches (they depend on the seed and the mixture only). Curve
    points carry ``phase`` (``trunk`` or ``target``); views are evaluated on the target only. A
    killed run resumes from ``<out>/arms/<name>/state``, before or after the graft.
    """
    import jax

    from bes.optim.build import batch_factor, data_factor, group_lrs
    from bes.train import checkpoint as C
    from bes.train import schedule as S
    from bes.train.train import abstract

    if trunk not in TRUNK_MODES:
        raise ValueError(f"trunk={trunk!r}: one of {TRUNK_MODES}")
    out = Path(out)
    if tok is None:
        from bes.tokenizer import load as load_tok
        tok = load_tok()
    plan = graft_plan(arm, tokens, bench)
    g, base = arm.graft, arm.graft.base
    budget, n_steps, gstep, shape, eval_steps = plan.budget, plan.n_steps, plan.step, plan.shape, plan.eval_steps
    h = spec_hash(arm, budget, bench)
    arm_dir = out / "arms" / arm.name
    state_dir = arm_dir / "state"
    tps = plan.cfg.optim.tokens_per_step
    log(f"[{arm.name}] {n_steps} steps of {tps} tokens, shape {shape}, seed {arm.seed}, evaluations at "
        f"steps {list(eval_steps)}; steps [0, {gstep}) as {base.name!r}, then grafted "
        f"(new leaves {list(plan.new_params)})")

    t_setup = time.time()
    pools = train_pools(arm, shape[2], tok, bench, log)
    maker = BatchMaker(pools, arm.mixture, arm.seed, shape, n_steps, tok)
    sets = eval_sets(bench.eval_domains, bench.eval_rows, shape[2], tok, bench, log)
    models = {"trunk": _Trainee(base, plan.base_cfg), "target": _Trainee(arm, plan.cfg, arm.eval_views())}

    start, curve, train_log, skipped = 0, [], [], 0
    timing: dict[str, Any] = {"train_s": 0.0, "eval_s": 0.0, "first_step_s": 0.0, "timed_steps": 0,
                              "setup_s": 0.0, "resumes": 0, "graft_s": 0.0,
                              "phases": {ph: {"first_step_s": None, "train_s": 0.0, "timed_steps": 0}
                                         for ph in models}}
    phase, state, info_graft = "trunk", None, None
    trunk_source: dict[str, Any] = {"source": "trained", "spec_hash": plan.base_hash, "rejected": []}

    def meta(step: int) -> dict:
        return {"step": step, "curve": curve, "train_log": train_log, "timing": timing,
                "skipped": skipped, "phase": phase, "graft": info_graft}

    def do_graft(trunk_info, trunk_state) -> tuple[Any, dict]:
        t0 = time.time()
        tm, gm = models["trunk"], models["target"]
        fresh = gm.fresh_state()
        arrays, records, treedef, rep = _graft_run().graft_arrays(trunk_info.path, {"state": fresh})
        if rep["mismatched"] or rep["left_behind"]:
            raise RuntimeError(f"[{arm.name}] the trunk does not fit the target: mismatched "
                               f"{rep['mismatched'][:4]}, left behind {rep['left_behind'][:4]}")
        paths, _, tdef = C.flatten_with_paths({"state": fresh})
        by_path = {r["path"]: r for r in records}
        if any(by_path[p]["stored"] != by_path[p]["dtype"] or by_path[p].get("key_impl") for p in paths):
            raise RuntimeError(f"[{arm.name}] the train state holds leaves stored in another dtype")
        new_state = jax.tree_util.tree_unflatten(tdef, [jax.device_put(arrays[p]) for p in paths])["state"]
        _, _, t_consts, _, t_loss = tm.fns()
        _, g_eval, g_consts, _, g_loss = gm.fns()
        n_rows = min(bench.eval_chunk, sets[0].n)
        rows = {k: v[:n_rows] for k, v in sets[0].batch.items()}
        before = float(t_loss(trunk_state.params, t_consts, rows))
        after = float(g_loss(new_state.params, g_consts, rows))
        held = evaluate(g_eval, new_state.params, g_consts, sets, bench.eval_chunk)
        heldout = {}
        for d, ev in held.items():
            b = curve[-1]["eval"][d]
            rows_diff = np.abs(np.asarray(ev["ce"], np.float64) - np.asarray(b["ce"], np.float64))
            heldout[d] = {"before": b["loss"], "after": ev["loss"], "difference": ev["loss"] - b["loss"],
                          "max_row_difference": float(rows_diff.max()) if rows_diff.size else 0.0}
        tol = GRAFT_TOL * max(1.0, abs(before))
        ok = (math.isfinite(before) and math.isfinite(after) and abs(after - before) <= tol
              and all(abs(v["difference"]) <= GRAFT_TOL * max(1.0, abs(v["before"])) for v in heldout.values()))
        p_g = gstep / n_steps
        info = {"base": base.name, "base_spec_hash": plan.base_hash, "target": g.target,
                "frac": float(g.frac), "step": gstep, "tokens": gstep * tps, "trunk": trunk_source,
                "check": {"batch": f"the first {n_rows} rows of held-out {sets[0].domain} "
                                   f"({n_rows} x {shape[2]} positions), bes.model.model.loss_fn",
                          "loss_before": before, "loss_after": after, "difference": after - before,
                          "tolerance": tol, "ok": bool(ok)},
                "heldout": heldout,
                "leaves": {k: rep[k] for k in ("copied", "copied_params", "copied_param_numbers",
                                                "new_params", "new_param_numbers", "new_other")},
                "carried": {"step": int(new_state.step), "count": int(new_state.opt_state.count),
                            "p": p_g, "lr_factor": S.lr_factor(p_g, plan.cfg.optim),
                            "momentum": S.momentum_at(gstep, p_g, plan.cfg)},
                "seconds": round(time.time() - t0, 2)}
        log(f"[{arm.name}] graft at step {gstep}: {rep['copied_params']} parameter leaves "
            f"({rep['copied_param_numbers']:,} numbers) carried with their optimiser slots, count "
            f"{info['carried']['count']}; new {rep['new_params']} ({rep['new_param_numbers']:,} numbers); "
            f"check loss {before:.7f} -> {after:.7f} (difference {after - before:+.3e}); held-out "
            + ", ".join(f"{d} {v['difference']:+.2e}" for d, v in heldout.items()))
        if not ok:
            append_jsonl(out / RESULTS, {"type": "arm", "status": "failed", "name": arm.name,
                                         "arm": arm.to_dict(), "spec_hash": h, "tokens": budget,
                                         "error": "function check failed at the graft point",
                                         "graft": info, "finished": time.time()})
            raise RuntimeError(f"[{arm.name}] the grafted model does not compute the trunk's function: "
                               f"loss {before!r} -> {after!r}, held-out "
                               f"{ {d: v['difference'] for d, v in heldout.items()} } (tolerance {tol:.2e})")
        timing["graft_s"] += time.time() - t0
        return new_state, info

    def save_grafted(arrays_state) -> None:
        C.save(state_dir, {"state": arrays_state}, step=gstep, config_hash=h, keep=1, meta=meta(gstep))

    got = _restore_graft_state(state_dir, h, models, log, arm.name) if state_dir.exists() else None
    if got is not None:
        phase, state, m, info = got
        start, curve, train_log = int(m["step"]), m["curve"], m["train_log"]
        timing.update(m["timing"])
        timing["resumes"] += 1
        skipped, info_graft = int(m.get("skipped", 0)), m.get("graft")
        if info_graft:
            trunk_source = info_graft["trunk"]
        log(f"[{arm.name}] resumed at step {start} ({phase})")
        if phase == "trunk" and start == gstep:            # stopped between the trunk's save and the graft
            state, info_graft = do_graft(info, state)
            phase = "target"
            save_grafted(state)
    else:
        shutil.rmtree(state_dir, ignore_errors=True)
        found, rejected = (None, []) if trunk == "train" else find_trunk(base, gstep, plan.base_hash,
                                                                         [out, *trunk_dirs])
        for r in rejected:
            log(f"[{arm.name}] trunk candidate rejected: {r['checkpoint']}: {r['reason']}")
        if found is None and trunk == "load":
            raise FileNotFoundError(f"[{arm.name}] trunk='load': no kept state of {base.name!r} at step "
                                    f"{gstep} with spec hash {plan.base_hash} in {[str(out), *map(str, trunk_dirs)]}")
        if found is not None:
            tm = models["trunk"]
            tree, man = C.load(found, {"state": abstract(tm.fresh_state())}, config_hash=plan.base_hash)
            m = man["meta"]
            curve = [{**c, "views": {}, "phase": "trunk"} for c in m["curve"] if int(c["step"]) <= gstep]
            train_log = [e for e in m["train_log"] if int(e[0]) < gstep]
            skipped = int(m.get("skipped", 0))
            if not curve or int(curve[-1]["step"]) != gstep:
                raise RuntimeError(f"[{arm.name}] {found.path}: no evaluation at the graft step in its meta")
            trunk_source = {"source": "loaded", "checkpoint": _rel(found.path), "spec_hash": plan.base_hash,
                            "rejected": rejected}
            log(f"[{arm.name}] trunk loaded from {_rel(found.path)} (spec hash {plan.base_hash})")
            state, info_graft = do_graft(found, tree["state"])
            phase, start = "target", gstep
            save_grafted(state)
        else:
            trunk_source = {"source": "trained", "spec_hash": plan.base_hash, "rejected": rejected}
            state = models["trunk"].fresh_state()
    timing["setup_s"] += time.time() - t_setup

    stop = n_steps if max_steps is None else min(n_steps, max_steps)
    for s in range(start, stop):
        tr = models[phase]
        step_fn, eval_fn, consts, views, _ = tr.fns()
        p = s / n_steps
        lr = S.lr_factor(p, tr.cfg.optim)
        mom = S.momentum_at(s, p, tr.cfg)
        wd = S.wd_scale_at(p, tr.cfg)
        batch = maker.batch(s)
        if batch_hook is not None:
            batch_hook(s, batch)
        t0 = time.time()
        state, metrics = step_fn(state, batch, np.float32(lr), np.float32(mom), np.float32(wd))
        loss = float(metrics["loss"])
        dt = time.time() - t0
        ph = timing["phases"][phase]
        if not tr.used:                                   # compilation: not a timed step
            tr.used = True
            if ph["first_step_s"] is None:
                ph["first_step_s"] = dt
            if s == 0:
                timing["first_step_s"] = dt
        else:
            ph["train_s"] += dt
            ph["timed_steps"] += 1
            timing["train_s"] += dt
            timing["timed_steps"] += 1
        skipped += int(metrics["skipped"])
        if s % bench.log_every == 0 or s == stop - 1:
            train_log.append([s, loss, round(lr, 6)])
            log(f"[{arm.name}] step {s:5d}/{n_steps} ({phase}) loss {loss:.4f} lr {lr:.3f} ({dt:.2f} s/step)")
        if (s + 1) in eval_steps:
            te = time.time()
            point = {"step": s + 1, "tokens": (s + 1) * tps, "train_loss": loss,
                     "eval": evaluate(eval_fn, state.params, consts, sets, bench.eval_chunk),
                     "views": {name: evaluate(eval_fn, state.params, c, sets, bench.eval_chunk)
                               for name, c in views},
                     "phase": phase}
            timing["eval_s"] += time.time() - te
            curve.append(point)
            msg = " ".join(f"{d} {v['loss']:.4f}" for d, v in point["eval"].items())
            log(f"[{arm.name}] eval at {point['tokens']} tokens ({phase}): {msg}")
            if phase == "trunk" and s + 1 == gstep:
                trunk_info = C.save(state_dir, {"state": state}, step=gstep, config_hash=h, keep=1,
                                    meta=meta(gstep))
                state, info_graft = do_graft(trunk_info, state)
                phase = "target"
                save_grafted(state)
            elif save_every_eval and s + 1 < n_steps:
                C.save(state_dir, {"state": state}, step=s + 1, config_hash=h, keep=1, meta=meta(s + 1))
    done = stop == n_steps
    cfg = plan.cfg
    params_dir = arm_dir / "params"
    if done:
        shutil.rmtree(params_dir, ignore_errors=True)
        C.save(params_dir, {"params": state.params}, step=n_steps, config_hash=config_hash(cfg),
               keep=1, meta={"arm": arm.to_dict(), "tokens": budget, "spec_hash": h})
    for ph in timing["phases"].values():
        ph["sec_per_step"] = ph["train_s"] / ph["timed_steps"] if ph["timed_steps"] else None
    timed = timing["timed_steps"]
    sec_step = timing["train_s"] / timed if timed else float(timing["first_step_s"] or 0.0)
    count = lambda c: int(sum(int(np.prod(sh)) for sh, _ in _param_shapes(c).values()))  # noqa: E731
    rec = {"type": "arm", "status": "done" if done else "partial", "name": arm.name,
           "arm": arm.to_dict(), "spec_hash": h, "tokens": budget, "steps": n_steps,
           "steps_run": stop, "shape": list(shape), "tokens_per_step": tps,
           "config": to_dict(cfg), "config_hash": config_hash(cfg), "bench": bench.to_dict(),
           "lrs": group_lrs(cfg), "batch_factor": batch_factor(cfg), "data_factor": data_factor(cfg),
           "n_params": count(cfg), "flops_per_token": flops_per_token(cfg),
           "timing": {**timing, "sec_per_step": sec_step, "sec_per_mtok": sec_step / tps * 1e6},
           "peak_rss_mb": peak_rss_mb(), "skipped_steps": skipped,
           "data": {"token_shares": maker.token_shares(n_steps), "passes": maker.passes,
                    "pools": {d: p.meta.get("rows") for d, p in pools.items()}},
           "curve": curve, "train_log": train_log,
           "params_dir": _rel(params_dir) if done else None,
           "graft": info_graft,
           "base": {"name": base.name, "spec_hash": plan.base_hash, "config": to_dict(plan.base_cfg),
                    "config_hash": config_hash(plan.base_cfg), "n_params": count(plan.base_cfg),
                    "flops_per_token": flops_per_token(plan.base_cfg)},
           "host": {"cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "omp_threads": os.environ.get("OMP_NUM_THREADS")},
           "finished": time.time()}
    append_jsonl(out / RESULTS, rec)
    if done:
        shutil.rmtree(state_dir, ignore_errors=True)
    log(f"[{arm.name}] {rec['status']}: {sec_step:.2f} s/step, {rec['timing']['sec_per_mtok']:.0f} "
        f"s per 1M tokens, peak {rec['peak_rss_mb']:.0f} MB, trunk {trunk_source['source']}")
    return rec


# ============================================================================== queue
def run_queue(arms: Sequence[Arm], out: str | os.PathLike, tokens: int, bench: Bench = Bench(),
              *, runner: Callable[[Arm], Any] | None = None, log: Callable[[str], None] = print
              ) -> list[str]:
    """Run the unfinished arms one after the other; returns the names that were run.

    runner: ``runner(arm)`` trains one arm (default: :func:`run_arm` in this process; the
    scripts pass one that starts a child process). An arm is finished when ``results.jsonl``
    holds a ``done`` record with its spec hash, so a restarted queue skips it.
    """
    ran = []
    for arm in arms:
        if finished(out, arm, tokens, bench) is not None:
            log(f"[{arm.name}] finished earlier: skipped")
            continue
        if runner is None:
            run_arm(arm, out, tokens, bench, log=log)
        else:
            runner(arm)
        ran.append(arm.name)
    return ran


# ============================================================================ analysis
@dataclass(frozen=True)
class Rule:
    """The pre-registered E2 rule (``docs/PLAN.md`` section 7) in relative loss units.

    Components with outside evidence are removed only if the arm without them is better than
    the reference by more than ``max(se_mult * paired SE, min_margin)``; a new component is kept
    only if removing it costs at least ``new_min_gain`` at the last checkpoint and that cost is at
    least ``new_keep_frac`` of the cost at the early checkpoint (``early_frac`` of the budget);
    an alternative setting is adopted only if it beats the reference by more than the margin.
    """

    se_mult: float = 2.0
    min_margin: float = 0.005
    new_min_gain: float = 0.01
    new_keep_frac: float = 0.5
    early_frac: float = 0.25


def paired_diff(ce_a: Sequence[float], ce_b: Sequence[float], w: Sequence[float]
                ) -> tuple[float, float]:
    """``(delta, se)`` of the per-token loss difference ``sum(a - b) / sum(w)`` of two models on
    the same held-out rows (per-row sums ``a``, ``b`` and weights ``w``); the standard error
    treats rows as independent clusters: ``sqrt(n/(n-1) * sum((a_i - b_i - delta w_i)^2)) /
    sum(w)``."""
    a, b, w = (np.asarray(x, np.float64) for x in (ce_a, ce_b, w))
    wsum = float(w.sum())
    if wsum <= 0 or a.size == 0:
        return 0.0, 0.0
    d = a - b
    delta = float(d.sum()) / wsum
    n = d.size
    if n < 2:
        return delta, 0.0
    se = math.sqrt(n / (n - 1) * float(((d - delta * w) ** 2).sum())) / wsum
    return delta, se


def decide(kind: str, g_last: float, g_early: float | None, se_last: float,
           rule: Rule = Rule()) -> tuple[str, str]:
    """``(decision, reason)`` for one arm. ``g = (L_arm - L_ref) / L_ref`` (positive: the arm
    is worse than the reference; for a removal arm that is what the component buys);
    ``se_last``: the paired standard error of ``g`` at the last checkpoint (relative)."""
    margin = max(rule.se_mult * se_last, rule.min_margin)
    if kind == "remove_evidence":
        if g_last < -margin:
            return "remove", f"without it the loss is {-g_last:.2%} lower (> margin {margin:.2%})"
        return "keep", f"removal changes the loss by {g_last:+.2%} (margin {margin:.2%})"
    if kind == "remove_new":
        if g_last < rule.new_min_gain:
            return "switch off", f"gain {g_last:+.2%} < {rule.new_min_gain:.0%} at the last checkpoint"
        if g_early is not None and g_last < rule.new_keep_frac * g_early:
            return "switch off", (f"gain fades: {g_last:+.2%} at the end < {rule.new_keep_frac:.0%} "
                                  f"of {g_early:+.2%} early")
        return "keep", (f"gain {g_last:+.2%} >= {rule.new_min_gain:.0%}"
                        + ("" if g_early is None else f" and >= {rule.new_keep_frac:.0%} of {g_early:+.2%} early"))
    if kind == "setting":
        if g_last < -margin:
            return "adopt", f"{-g_last:.2%} lower loss (> margin {margin:.2%})"
        return "keep default", f"{g_last:+.2%} vs reference (margin {margin:.2%})"
    if kind == "add_new":
        if g_last > margin:
            return "harm", f"{g_last:.2%} higher loss with it (> margin {margin:.2%}): switch it off"
        if g_last < -margin:
            return "gain", f"{-g_last:.2%} lower loss with it (> margin {margin:.2%})"
        return "not resolved", f"{g_last:+.2%} vs reference (margin {margin:.2%})"
    return "-", ""


def _ckpt(rec: dict, tokens: int) -> dict | None:
    return next((c for c in rec.get("curve", []) if int(c["tokens"]) == int(tokens)), None)


def domain_shares(ref: dict, domains: Sequence[str]) -> dict[str, float]:
    """Mixture shares of the reference restricted to the evaluated domains (``mix`` weights)."""
    mix = {d: s for d, s in ref["arm"]["mixture"]}
    w = {d: float(mix.get(d, 0.0)) for d in domains if d in mix}
    s = sum(w.values())
    return {d: v / s for d, v in w.items()} if s > 0 else {}


def compare(arm: dict, ref: dict, tokens: int, domain: str, view: str | None = None
            ) -> tuple[float, float, float] | None:
    """``(rel, se_rel, ref_loss)`` of ``arm`` against ``ref`` at a checkpoint for one domain
    (``mix``: share-weighted web/chat); ``view``: compare the reference's own view instead of
    another arm (``arm`` is then ignored)."""
    ca, cr = _ckpt(arm, tokens), _ckpt(ref, tokens)
    if ca is None or cr is None:
        return None
    if domain == "mix":
        shares = domain_shares(ref, [d for d in cr["eval"] if d != "assistant"])
        if not shares:
            return None
        l_ref = l_arm = var = 0.0
        for d, pi in shares.items():
            got = compare(arm, ref, tokens, d, view)
            if got is None:
                return None
            rel, se, lr = got
            l_ref += pi * lr
            l_arm += pi * lr * (1 + rel)
            var += (pi * se * lr) ** 2
        return (l_arm - l_ref) / l_ref, math.sqrt(var) / l_ref, l_ref
    er = cr["eval"].get(domain)
    ea = (cr["views"].get(view, {}) if view else ca["eval"]).get(domain)
    if er is None or ea is None:
        return None
    delta, se = paired_diff(ea["ce"], er["ce"], er["w"])
    return delta / er["loss"], se / er["loss"], er["loss"]


def latest_done(records: Iterable[dict], tokens: int | None = None) -> dict[str, dict]:
    """The newest finished record per arm name (optionally only budgets equal to ``tokens``)."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("type") == "arm" and r.get("status") == "done" and (tokens is None or int(r["tokens"]) == int(tokens)):
            out[r["name"]] = r
    return out


def early_tokens(ref: dict, rule: Rule = Rule()) -> int | None:
    """Token count of the checkpoint closest to ``rule.early_frac`` of the budget (not the last)."""
    pts = [int(c["tokens"]) for c in ref.get("curve", [])]
    if len(pts) < 2:
        return None
    target = rule.early_frac * int(ref["tokens"])
    return min(pts[:-1], key=lambda t: abs(t - target))


def arm_table(records: Iterable[dict], ref_name: str = "ref", replicate: str | None = "ref_s1",
              rule: Rule = Rule(), domain: str = "mix") -> list[dict]:
    """One row per finished arm against the reference: relative differences at every checkpoint,
    the paired SE (max of the seed-replicate noise and the held-out sampling SE), the margin and
    the decision. Empty when the reference has not finished."""
    done = latest_done(records)
    ref = done.get(ref_name)
    if ref is None:
        return []
    pts = [int(c["tokens"]) for c in ref["curve"]]
    last, early = pts[-1], early_tokens(ref, rule)
    rep = done.get(replicate) if replicate else None
    rows = []
    for name, rec in done.items():
        if name == ref_name or int(rec["tokens"]) != int(ref["tokens"]):
            continue
        kind = rec["arm"]["kind"]
        rels = {}
        for t in pts:
            got = compare(rec, ref, t, domain)
            rels[t] = None if got is None else got[0]
        got_last = compare(rec, ref, last, domain)
        se_eval = got_last[1] if got_last else 0.0
        noise = None
        if rep is not None and name != replicate:
            g = compare(rep, ref, last, domain)
            noise = None if g is None else abs(g[0])
        se_pair = max(se_eval, noise or 0.0)
        g_last = rels.get(last)
        g_early = rels.get(early) if early is not None else None
        if g_last is None:
            decision, reason = "-", "checkpoints do not match the reference"
        elif kind == "replicate":
            decision, reason = "-", "seed replicate: paired noise"
        else:
            decision, reason = decide(kind, g_last, g_early, se_pair, rule)
        rows.append({"name": name, "kind": kind, "component": rec["arm"].get("component", ""),
                     "rel": rels, "se_eval": se_eval, "noise": noise, "se_pair": se_pair,
                     "margin": max(rule.se_mult * se_pair, rule.min_margin),
                     "decision": decision, "reason": reason, "record": rec})
    return rows


# ============================================================================== report
LIMITS_TEXT = """\
What a micro run can say about the 20B-token run, and what it cannot:

- **Can**: show that a path trains and is wired correctly (a component that makes the loss
  worse from the first checkpoint, a feature table that never moves, a learning-rate group off
  by an order of magnitude); rank changes whose effect at equal tokens is large (several
  percent) and stable across checkpoints; measure what an exact feature gives a model that has
  not yet learnt the same thing itself (copying, counting, fences); measure the CPU cost per
  token and peak memory of each variant.
- **Cannot**: predict the size of an effect at 20B tokens. The micro model is 6x narrower
  (d=128 vs 768), sees 1-4M tokens (5,000-20,000x fewer) with a few hundred optimiser steps
  (the final run takes about 80,000), at T=512 instead of 1024/2048 (the tape's value grows
  with context length), in float32 on CPU. Early-training advantages of shortcut features
  (tape, registers) can shrink once the network learns induction and counting itself, or grow
  with longer contexts; optimiser and learning-rate comparisons after a few hundred steps say
  little about the end of a long run; the transfer rule's exponents are extrapolated over four
  orders of magnitude here; loop-count effects at d=128 need not hold at d=768. Differences
  below the margin (at least 0.5 %, see the noise table) are not evidence either way.
- The rule therefore only removes components with outside evidence when they clearly hurt,
  and only keeps new components whose gain is clear (>= 1 %) and does not fade between the
  early and the last checkpoint; everything else stays as configured, and the full-scale run
  keeps every path behind a zero-initialised switch.
"""


def report_domains(rec: dict) -> list[str]:
    """Evaluated domains of a record in display order: the bench's ``eval_domains`` first, then
    the other keys of the last checkpoint (``assistant``, ...)."""
    keys = list(rec["curve"][-1]["eval"]) if rec.get("curve") else []
    first = [d for d in rec.get("bench", {}).get("eval_domains", []) if d in keys]
    return first + sorted(k for k in keys if k not in first)


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:+.{digits}f}%"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def e2_report(records: Sequence[dict], *, arms: Sequence[Arm] = (), ref_name: str = "ref",
              replicate: str = "ref_s1", rule: Rule = Rule(), source: str = "runs/e2/results.jsonl",
              skipped: Sequence[tuple[str, str]] = ()) -> str:
    """The E2 document (Markdown) from arm records; placeholders when nothing has run yet.

    arms: the planned arms (to list the missing ones); skipped: ``(name, reason)`` of arms the
    queue does not run.
    """
    done = latest_done(records)
    head = ["# E2 — micro ablations (gate E2 of `docs/PLAN.md` section 7)", "",
            f"Generated by `python scripts/micro_ablate.py --report` from `{source}`; do not edit "
            "by hand.", ""]
    ref = done.get(ref_name)
    rule_text = (
        "## Rule (pre-registered)\n\n"
        "Plan section 7, E2: components with outside evidence are removed only if worse than "
        f"max({rule.se_mult:g} x paired standard error, {rule.min_margin:.1%}); new components must "
        f"gain >= {rule.new_min_gain:.0%} at the last checkpoint and keep at least "
        f"{rule.new_keep_frac:.0%} of their gain at the early checkpoint ({rule.early_frac:.0%} of "
        "the budget; 2.5M of 10M in the plan). Operationally, with g = (L_arm - L_ref) / L_ref "
        "on the held-out loss (`mix` = web and chat weighted by the training shares):\n\n"
        "- `remove_evidence` arms (a component with outside evidence removed): **remove** if "
        "g < -margin, else **keep**;\n"
        "- `remove_new` arms (a new component removed): **keep** if g_last >= 1% and g_last >= "
        "0.5 x g_early, else **switch off**;\n"
        "- `setting` arms (an alternative value): **adopt** if g < -margin, else **keep default**;\n"
        "- `add_new` arms (a new component added to the reference, such as the v9 token pages): "
        "**harm** if g > margin (switch it off), **gain** if g < -margin, else **not resolved**: a "
        "row of a token-keyed table is visited about 120 times on average in a 1M-token micro run "
        "(8,192 token types), so such an arm can show harm, not the benefit at scale;\n"
        "- margin = max(2 x SE_pair, 0.5%); SE_pair = max(|g| of the seed replicate against the "
        "reference, the paired held-out sampling SE of the arm). The seed replicate changes both "
        "the data order and the initialisation, so it bounds the paired noise from above; with one "
        "replicate it is a one-degree-of-freedom estimate.\n")
    if ref is None:
        body = ["## Status", "", "**Not run yet.** No finished reference arm in "
                f"`{source}`. Launch the queue (`scripts/micro_ablate.py`), then regenerate this "
                "file with `--report`.", "", rule_text, "## Limits", "", LIMITS_TEXT]
        if arms:
            body += ["## Planned arms", "", _arm_list(arms, skipped)]
        return "\n".join(head + body) + "\n"
    pts = [int(c["tokens"]) for c in ref["curve"]]
    rows = arm_table(records, ref_name, replicate, rule)
    out = head + [
        "## Setup", "",
        f"- Model `{ref['bench']['config']}`: d={ref['config']['model']['d_model']}, "
        f"{ref['config']['model']['n_loops']} loops, {ref['n_params']:,} parameters; rows of "
        f"T={ref['shape'][2]}, {ref['tokens_per_step']} tokens per step "
        f"(shape A,B,T = {tuple(ref['shape'])}), {ref['steps']} steps = {ref['tokens']:,} tokens per arm.",
        f"- Mixture (row shares): {', '.join(f'{d} {s:.0%}' for d, s in ref['arm']['mixture'])}; "
        f"realised target-token shares: "
        f"{', '.join(f'{d} {s:.1%}' for d, s in ref['data']['token_shares'].items())}.",
        f"- Held-out: the first {ref['bench']['eval_rows']} rows of each held-out pool "
        f"({', '.join(ref['bench']['eval_domains'])}); `assistant` = assistant tokens of the chat rows.",
        f"- Learning rates (transfer rule: batch factor {ref['batch_factor']:.4f}, data factor "
        f"{ref['data_factor']:.3f}): " + ", ".join(f"{g} {v:.4g}" for g, v in ref["lrs"].items() if v),
        f"- Checkpoints: {', '.join(f'{t:,}' for t in pts)} tokens; early checkpoint for the rule: "
        f"{early_tokens(ref, rule) or 'n/a'}.", "", rule_text]
    # reference losses
    out += ["## Reference", ""]
    doms = report_domains(ref)
    out.append(_md_table(["tokens"] + doms + [f"tape_gain ({doms[0]})", f"tape_mass ({doms[0]})",
                                               f"sink_mass ({doms[0]})"],
                         [[f"{c['tokens']:,}"] + [f"{c['eval'][d]['loss']:.4f}" for d in doms]
                          + [f"{c['eval'][doms[0]].get('tape_gain', 0):.4f}",
                             f"{c['eval'][doms[0]].get('tape_mass', 0):.3f}",
                             ", ".join(f"{x:.3f}" for x in c['eval'][doms[0]].get('sink_mass', []))]
                          for c in ref["curve"]]))
    out.append("")
    rep = done.get(replicate)
    out += ["## Paired noise (seed replicate)", ""]
    if rep is None:
        out += [f"Seed replicate `{replicate}` not finished yet: SE_pair is the held-out sampling SE "
                "only, which understates the run-to-run noise.", ""]
    else:
        nrows = []
        for t in pts:
            cells = [f"{t:,}"]
            for d in ["mix"] + doms:
                g = compare(rep, ref, t, d)
                cells.append("n/a" if g is None else f"{_pct(g[0])} (sampling SE {100 * g[1]:.2f}%)")
            nrows.append(cells)
        out += [_md_table(["tokens"] + ["mix"] + doms, nrows), ""]
    # main table
    out += ["## Arms against the reference", "",
            "g = relative held-out loss difference (arm - ref) / ref on `mix`; positive = the arm is "
            "worse. Per-domain values at the last checkpoint follow.", ""]
    header = ["arm", "kind", "component"] + [f"g @ {t:,}" for t in pts] + [
        "SE_pair", "margin", "decision", "reason"]
    trows = []
    order = {k: i for i, k in enumerate(("replicate", "remove_new", "remove_evidence", "setting",
                                         "add_new", "probe"))}
    for r in sorted(rows, key=lambda r: (order.get(r["kind"], 9), r["name"])):
        trows.append([f"`{r['name']}`", r["kind"], r["component"]] + [_pct(r["rel"][t]) for t in pts]
                     + [f"{100 * r['se_pair']:.2f}%", f"{100 * r['margin']:.2f}%", f"**{r['decision']}**",
                        r["reason"]])
    out += [_md_table(header, trows), ""]
    # per domain
    drow = []
    for r in sorted(rows, key=lambda r: r["name"]):
        cells = [f"`{r['name']}`"]
        for d in doms:
            g = compare(r["record"], ref, pts[-1], d)
            cells.append("n/a" if g is None else f"{_pct(g[0])} ± {100 * g[1]:.2f}%")
        drow.append(cells)
    out += ["### Per domain at the last checkpoint (± paired held-out sampling SE)", "",
            _md_table(["arm"] + doms, drow), ""]
    # views
    if ref["curve"][-1].get("views"):
        vrows = []
        for name in ref["curve"][-1]["views"]:
            for t in pts:
                cells = [f"`{name}`", f"{t:,}"]
                for d in ["mix"] + doms:
                    g = compare(ref, ref, t, d, view=name)
                    cells.append("n/a" if g is None else _pct(g[0]))
                vrows.append(cells)
        out += ["## Evaluation-time switches on the trained reference", "",
                "The reference model evaluated with a path switched off at evaluation time only "
                "(a second view: the model was trained with the path on, so this measures how much "
                "the trained model leans on it, not what training without it would give).", "",
                _md_table(["view", "tokens", "mix"] + doms, vrows), ""]
    # model metrics and cost
    crow = []
    for name, rec in sorted(done.items()):
        last_pt = rec["curve"][-1]
        e0 = last_pt["eval"].get(report_domains(rec)[0], {})
        crow.append([f"`{name}`", f"{rec['n_params']:,}", f"{rec['flops_per_token']['train'] / 1e6:.1f}",
                     f"{rec['timing']['sec_per_mtok']:.0f}", f"{rec['peak_rss_mb']:.0f}",
                     f"{e0.get('tape_gain', 0):.4f}", f"{e0.get('tape_mass', 0):.3f}",
                     ", ".join(f"{x:.3f}" for x in e0.get("sink_mass", [])),
                     rec.get("skipped_steps", 0)])
    out += [f"## Model metrics and cost (last checkpoint; tape and sink metrics on `{doms[0]}`)", "",
            "Training MFLOP per token is analytic (`microbench.flops_per_token`); seconds per 1M "
            "tokens and peak memory are measured on this container (2 threads, shared machine).", "",
            _md_table(["arm", "params", "train MFLOP/token", "s per 1M tokens", "peak MB",
                       "tape_gain", "tape_mass", "sink_mass", "skipped steps"], crow), ""]
    missing = [a for a in arms if a.name not in done]
    if missing or skipped:
        out += ["## Not run (yet)", "", _arm_list(missing, skipped), ""]
    out += ["## Limits", "", LIMITS_TEXT]
    return "\n".join(out) + "\n"


def _arm_list(arms: Sequence[Arm], skipped: Sequence[tuple[str, str]] = ()) -> str:
    lines = []
    for a in arms:
        what = ", ".join([*a.overrides, *(f"{k}={v:g}" for k, v in a.switches)]) or "(reference settings)"
        lines.append(f"- `{a.name}` ({a.kind}, seed {a.seed}): {what}" + (f" — {a.component}" if a.component else ""))
    for name, reason in skipped:
        lines.append(f"- `{name}`: skipped — {reason}")
    return "\n".join(lines) if lines else "(none)"
