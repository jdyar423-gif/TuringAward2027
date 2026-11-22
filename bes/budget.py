"""Parameter budget: stored numbers and numbers read per generated token.

Accounting rules (plan section 2): one parameter = one stored number of at most 16 bits.

* ``total``: every number shipped with the model (all parameter leaves, the compressed
  library divided by 2 bytes, the retrieval index).
* ``active_step`` (standard count, the MoE "A" convention): numbers read to produce one
  token's full distribution given the cached state. Dense leaves count fully; a lookup table
  counts one row per read (``min(reads, rows) * row_width``); the library and index count the
  numbers one retrieval reads (conservatively charged to the step that retrieves).
* ``active_strict(T, distinct_tokens)``: union of the numbers read over a context window of
  ``T`` positions holding ``distinct_tokens`` distinct token ids: vocabulary tables count one
  row per distinct token, small tables every row that ``T`` positions can touch.

``count`` works on real parameter trees and on ``jax.eval_shape`` trees (only ``.shape`` is
used), so the final configuration can be checked without allocating it.
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Any

from bes.config import CAND_SLOTS
from bes.roles import leaves_with_paths, role_of

TOTAL_CAP = 100_000_000
ACTIVE_CAP = 20_000_000
#: numbers per independent 64 kB xz library block (one number = 2 bytes)
LIBRARY_BLOCK_NUMBERS = 32_768


@dataclass(frozen=True)
class Assets:
    """Sizes of the shipped non-parameter assets, in stored numbers.

    library_numbers: compressed library bytes / 2; index_numbers: retrieval index numbers;
    library_read: numbers read by one retrieval (blocks decompressed); index_read: index
    numbers read by one retrieval.
    """
    library_numbers: int = 0
    index_numbers: int = 0
    library_read: int = 0
    index_read: int = 0

    @classmethod
    def from_any(cls, assets: Any) -> "Assets":
        if assets is None:
            return cls()
        if isinstance(assets, Assets):
            return assets
        if isinstance(assets, dict):
            return cls(**assets)
        raise TypeError(f"cannot build Assets from {type(assets).__name__}")


#: the plan's ledger: 60M-number library (1-2 blocks per retrieval), 4M-number index (~450 read)
PLAN_ASSETS = Assets(library_numbers=60_000_000, index_numbers=4_000_000,
                     library_read=2 * LIBRARY_BLOCK_NUMBERS, index_read=450)

#: The library actually built by ``scripts/build_library.py`` with the default ``LibraryCfg``
#: (docs/E1_COVERAGE.md): every stored table, and the analytical worst case of one retrieval
#: (``Library.read_bound``: index lists 132,628 + question-answer index 58,900 + blocks,
#: directory and title tables 53,695). Use ``assets_of(library)`` for a library on disk.
V2_ASSETS = Assets(library_numbers=40_268_392, index_numbers=23_499_999,
                   library_read=53_695, index_read=191_528)


def assets_of(library: Any) -> Assets:
    """Assets of a built library object (``bes.library.retrieve.Library``): its exact stored
    numbers and the analytical bound of one retrieval's read set."""
    acc = library.accounting()
    bound = library.read_bound()
    total_read = int(bound["total"] if isinstance(bound, dict) else bound)
    index = int(acc["index_numbers"])
    return Assets(library_numbers=int(acc["total_numbers"]) - index, index_numbers=index,
                  library_read=total_read, index_read=0)

#: lookup tables: (pattern, rows read per generated token, strict rule)
#:   "tokens": rows indexed by token ids (strict: one row per distinct token in the window);
#:   "window": rows indexed by small features (strict: every row T positions can touch).
_LOOKUP_READS: list[tuple[str, int, str]] = [
    ("emb/tok", 1, "tokens"),
    ("emb/ve", 1, "tokens"),            # one [n_loops * ve_dim] row per token
    ("emb/reg/*", 1, "window"),
    ("cand/in_tok", CAND_SLOTS, "tokens"),
    ("cand/in_feat/*", CAND_SLOTS, "window"),
    ("trust/feat/*", CAND_SLOTS, "window"),
    ("trust/bias/*", CAND_SLOTS, "window"),      # rows of width 1 (one scalar per feature value)
    ("gain/*", 1, "tokens"),            # one [gain_phases * n_blocks * width] row per token
    ("page/key", 1, "tokens"),          # one [n_blocks * tok_neurons * tok_dim] row per token
    ("page/val", 1, "tokens"),          # (the projections page/pin, page/pout are dense)
]

#: rows of the plan's ledger (section 4) and the leaves they hold
_LEDGER_ROWS: list[tuple[str, tuple[str, ...]]] = [
    ("core", ("core/attn/wq", "core/attn/wk", "core/attn/wv", "core/attn/wo", "core/mlp/*")),
    ("head", ("head/proj", "head/out")),
    ("emb", ("emb/tok",)),
    ("ve", ("emb/ve",)),
    ("gain", ("gain/*",)),
    ("page", ("page/*",)),
]
LEDGER_ORDER = ("core", "head", "small", "emb", "ve", "gain", "page", "tables", "library", "index")


@dataclass(frozen=True)
class Item:
    """One counted entry: a parameter leaf or an asset."""
    name: str
    row: str          # ledger row (see LEDGER_ORDER)
    kind: str         # "dense" | "lookup" | "asset"
    total: int
    step: int
    rows: int = 0
    width: int = 0
    reads: int = 0
    strict: str = ""

    def strict_count(self, T: int, distinct_tokens: int) -> int:
        """Numbers of this entry read over a window of ``T`` positions."""
        if self.kind != "lookup":
            return self.step
        if self.strict == "tokens":
            rows = min(self.rows, distinct_tokens, T * self.reads)
        else:
            rows = min(self.rows, T * self.reads)
        return rows * self.width


@dataclass(frozen=True)
class Budget:
    total: int
    active_step: int
    items: tuple[Item, ...]

    def active_strict(self, T: int, distinct_tokens: int) -> int:
        """Strict (context-window union) active count for ``T`` positions with
        ``distinct_tokens`` distinct ids (``distinct_tokens <= T``)."""
        if distinct_tokens > T:
            raise ValueError("distinct_tokens cannot exceed T")
        return sum(item.strict_count(T, distinct_tokens) for item in self.items)

    def ledger(self) -> dict[str, tuple[int, int]]:
        """``{row: (total, step)}`` in the order of the plan's ledger."""
        out = {row: [0, 0] for row in LEDGER_ORDER}
        for item in self.items:
            out[item.row][0] += item.total
            out[item.row][1] += item.step
        return {row: (t, s) for row, (t, s) in out.items()}

    def within_caps(self, total_cap: int = TOTAL_CAP, active_cap: int = ACTIVE_CAP) -> bool:
        return self.total <= total_cap and self.active_step <= active_cap

    def report(self) -> str:
        lines = [f"{'row':8s} {'total':>12s} {'per step':>12s}"]
        for row, (t, s) in self.ledger().items():
            lines.append(f"{row:8s} {t:12,d} {s:12,d}")
        lines.append(f"{'sum':8s} {self.total:12,d} {self.active_step:12,d}")
        return "\n".join(lines)


def _ledger_row(path: str, kind: str) -> str:
    for row, patterns in _LEDGER_ROWS:
        if any(fnmatch.fnmatchcase(path, p) for p in patterns):
            return row
    return "tables" if kind == "lookup" else "small"


def _lookup_rule(path: str) -> tuple[int, str]:
    for pattern, reads, rule in _LOOKUP_READS:
        if fnmatch.fnmatchcase(path, pattern):
            return reads, rule
    raise KeyError(f"lookup leaf {path!r} has no read rule in bes.budget")


def _gain_width(cfg: Any, path: str) -> int | None:
    """Row width of a gain table ``[V, gain_phases, n_blocks, width]`` (None: not a gain table)."""
    per = {"gain/ffn": cfg.gain_ffn, "gain/o": cfg.n_heads * cfg.head_dim,
           "gain/v": cfg.n_heads * cfg.head_dim}.get(path)
    return None if per is None else cfg.gain_phases * cfg.n_blocks * per


def _page_width(cfg: Any, path: str) -> int | None:
    """Row width of a token-neuron table ``[V, n_blocks, tok_neurons, tok_dim]`` (None: not one)."""
    if path not in ("page/key", "page/val"):
        return None
    return cfg.n_blocks * cfg.tok_neurons * cfg.tok_dim


def count(params: Any, cfg: Any = None, assets: Any = None) -> Budget:
    """Count a parameter tree (arrays or shape structs) plus the shipped assets.

    cfg: the ``ModelCfg`` (used for a consistency check of the value-embedding, gain and
    token-neuron tables); assets: :class:`Assets`, a dict of its fields, or None (no library,
    no index).
    """
    items: list[Item] = []
    for path, leaf in leaves_with_paths(params):
        shape = tuple(int(s) for s in leaf.shape)
        size = math.prod(shape)
        role = role_of(path)
        row = _ledger_row(path, role.kind)
        if role.kind == "dense":
            items.append(Item(path, row, "dense", size, size))
            continue
        reads, rule = _lookup_rule(path)
        rows = shape[0]
        width = size // rows
        if cfg is not None and path == "emb/ve" and width != cfg.n_loops * cfg.ve_dim:
            raise ValueError(f"emb/ve row width {width} != n_loops * ve_dim")
        if cfg is not None and path.startswith("gain/") and width != _gain_width(cfg, path):
            raise ValueError(f"{path} row width {width} does not match gain_phases, n_blocks and "
                             "the gain width of the configuration")
        if cfg is not None and path.startswith("page/") and width != _page_width(cfg, path):
            raise ValueError(f"{path} row width {width} != n_blocks * tok_neurons * tok_dim")
        items.append(Item(path, row, "lookup", size, min(reads, rows) * width,
                          rows=rows, width=width, reads=reads, strict=rule))
    a = Assets.from_any(assets)
    if a.library_numbers or a.library_read:
        items.append(Item("library", "library", "asset", int(a.library_numbers), int(a.library_read)))
    if a.index_numbers or a.index_read:
        items.append(Item("index", "index", "asset", int(a.index_numbers), int(a.index_read)))
    return Budget(total=sum(i.total for i in items), active_step=sum(i.step for i in items),
                  items=tuple(items))
