"""Gated retrieval of at most one library passage per user message.

``Library`` ties together the packed text (``bes.library.pack.Packed``) and the stored
truncated index (``bes.library.index.TruncatedIndex``). All scoring uses only the stored
index; everything a query reads is logged in a ``ReadLog`` as distinct stored numbers
(index df entries and posting rows, directory entries, compressed text/title blocks).

Gate (``GateCfg``): a passage is returned only when

* the query type is allowed (``query_type``: questions and short look-ups; no writing,
  coding, translation or arithmetic tasks),
* the top BM25 score ``s1`` is at least ``min_score``,
* the query coverage ``s1 / s_max`` is at least ``min_coverage``, where ``s_max`` sums the
  largest stored impact of every scored query bucket (the best score any passage could
  reach), and
* the relative margin ``m = (s1 - s2) / s1`` over the best passage of a *different*
  article is at least ``min_margin``, or at least ``min_margin_title`` when the top
  article's title matches the query (a fraction ``>= min_title_match`` of its title terms
  occur in the query).

The returned passage is ``title + "\\n" + body`` capped at ``max_tokens`` tokens.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from bes.config import LibraryCfg
from bes.library.clean import CountTokens, default_count_tokens, render_passage, split_sentences
from bes.library.index import IndexCfg, IndexRead, TruncatedIndex
from bes.library.pack import Packed, PackCfg, pack
from bes.library.terms import strip_disambiguator, terms

# ------------------------------------------------------------------------- query type

_LEADS = ("please", "can you", "could you", "would you", "will you", "i want you to",
          "i need you to", "i'd like you to", "i would like you to", "kindly", "pls", "hey",
          "hi", "hello", "ok", "okay", "so")
_TASK = frozenset(
    """write compose draft create generate rewrite rephrase paraphrase summarize summarise
    translate edit proofread correct fix make design brainstorm suggest imagine pretend act
    roleplay continue complete convert format classify categorize categorise sort extract
    calculate compute solve implement code debug refactor optimize optimise plan outline list
    give provide help come draw sing play respond reply recommend rank rate review analyze
    analyse evaluate improve shorten expand simplify turn change develop build prepare
    organize organise schedule find count combine split rewrite""".split()
)
_ASK = frozenset(
    """who what when where which why how whom whose is are was were do does did can could has
    have had name in on at tell explain describe define what's who's where's when's how's
    which's whats whos""".split()
)
_SMALL_TALK = frozenset(
    """hi hello hey thanks thank you thx ok okay yes no bye goodbye good morning evening night
    cool great nice sure please lol""".split()
)
_MATH_RX = re.compile(r"\d\s*[-+*/^×÷=]\s*\(?\d|\b(solve|equation|integral|derivative)\b", re.I)


def query_type(text: str) -> str:
    """Coarse type of a user message: ``question``, ``lookup``, ``task``, ``math`` or ``other``.

    ``question``: starts with a question/explanation word (after politeness leads) or ends
    with ``?``; ``lookup``: a short phrase of at most six words without a verb-like lead;
    ``task``: an instruction to produce or transform text/code (or a long/pasted message);
    ``math``: arithmetic or equation solving.
    """
    t = text.strip()
    if not t:
        return "other"
    if "```" in t or len(t) > 1000 or t.count("\n") > 6:
        return "task"
    low = re.sub(r"\s+", " ", t.lower())
    changed = True
    while changed:
        changed = False
        for lead in _LEADS:
            if low.startswith(lead + " ") or low.startswith(lead + ","):
                low = low[len(lead):].lstrip(" ,")
                changed = True
    words = re.findall(r"[a-z0-9']+", low)
    if not words or all(w in _SMALL_TALK for w in words):
        return "other"
    if _MATH_RX.search(low):
        return "math"
    first = words[0]
    if first in ("tell", "explain", "describe", "define") or low.startswith(("give me information",
                                                                               "give me facts")):
        return "question"
    if first in _TASK:
        return "task"
    if first in _ASK or t.endswith("?"):
        return "question"
    if len(words) <= 6:
        return "lookup"
    return "other"


# ------------------------------------------------------------------------- config


_L = LibraryCfg()


@dataclass(frozen=True)
class GateCfg:
    min_score: float = _L.gate_min_score            # BM25 units (de-quantised impacts)
    min_margin: float = _L.gate_min_margin
    min_margin_title: float = _L.gate_min_margin_title
    min_title_match: float = _L.gate_min_title_match
    min_coverage: float = _L.gate_min_coverage
    allowed_types: tuple[str, ...] = ("question", "lookup")
    margin_candidates: int = 64     # candidates scanned for the best different-title score


@dataclass(frozen=True)
class RetrieveCfg:
    max_tokens: float = _L.max_passage_tokens  # cap on title line + body
    gate: GateCfg = field(default_factory=GateCfg)


# ------------------------------------------------------------------------- read log


@dataclass
class ReadLog:
    """Distinct stored numbers read while answering one query."""

    index_df: int = 0
    index_offsets: int = 0
    index_postings: int = 0
    directory: int = 0
    text_blocks: int = 0
    title_blocks: int = 0
    blocks: set = field(default_factory=set, repr=False)  # text block ids read
    title_block_ids: set = field(default_factory=set, repr=False)
    _dir_seen: set = field(default_factory=set, repr=False)

    @property
    def index(self) -> int:
        return self.index_df + self.index_offsets + self.index_postings

    @property
    def total(self) -> int:
        return self.index + self.directory + self.text_blocks + self.title_blocks

    def as_dict(self) -> dict[str, int]:
        return {"index_df": self.index_df, "index_offsets": self.index_offsets,
                "index_postings": self.index_postings,
                "directory": self.directory, "text_blocks": self.text_blocks,
                "title_blocks": self.title_blocks, "n_text_blocks": len(self.blocks),
                "n_title_blocks": len(self.title_block_ids), "total": self.total}

    def touch_dir(self, array: str, idx: Iterable[int], numbers_per_entry: int) -> None:
        for i in idx:
            key = (array, int(i))
            if key not in self._dir_seen:
                self._dir_seen.add(key)
                self.directory += numbers_per_entry


# ------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Passage:
    pid: int
    title_id: int
    title: str
    body: str
    score: float      # BM25 score from the stored index (de-quantised)
    text: str         # title + "\n" + body, capped at max_tokens


def cap_tokens(title: str, body: str, max_tokens: float, count_tokens: CountTokens) -> str:
    """``render_passage(title, body)`` cut at the last sentence (else word) that fits."""
    full = render_passage(title, body)
    if count_tokens(full) <= max_tokens:
        return full
    lines = body.split("\n")
    kept: list[str] = []
    for line in lines:
        trial = render_passage(title, "\n".join(kept + [line]))
        if count_tokens(trial) <= max_tokens:
            kept.append(line)
            continue
        part = ""
        for sent in split_sentences(line) or [line]:
            cand = (part + " " + sent).strip()
            if count_tokens(render_passage(title, "\n".join(kept + [cand]))) <= max_tokens:
                part = cand
            else:
                break
        if not part:
            for w in line.split():
                cand = (part + " " + w).strip()
                if count_tokens(render_passage(title, "\n".join(kept + [cand]))) <= max_tokens:
                    part = cand
                else:
                    break
        if part:
            kept.append(part)
        break
    out = render_passage(title, "\n".join(kept))
    while count_tokens(out) > max_tokens and len(out) > 1:  # pathological title
        out = out[: int(len(out) * 0.9)]
    return out


# ------------------------------------------------------------------------- library


class Library:
    """Packed passages + stored truncated index + gate."""

    def __init__(self, packed: Packed, index: TruncatedIndex, cfg: RetrieveCfg = RetrieveCfg(),
                 count_tokens: CountTokens = default_count_tokens, meta: dict | None = None):
        if packed.n_passages != index.n_docs:
            raise ValueError("index and packed library disagree on the number of passages")
        self.packed = packed
        self.index = index
        self.cfg = cfg
        self.count_tokens = count_tokens
        self.meta = meta or {}
        self._title_map: dict[str, int] | None = None

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, passages: Sequence[tuple[str, str]], out_dir: str | None = None,
              index_cfg: IndexCfg = IndexCfg(), pack_cfg: PackCfg = PackCfg(),
              cfg: RetrieveCfg = RetrieveCfg(), count_tokens: CountTokens = default_count_tokens,
              meta: dict | None = None, prior: Sequence[float] | None = None) -> "Library":
        """Build from (title, body) pairs; passages of one title must be contiguous.

        Passage ids follow the given order; title ids are assigned in order of appearance.
        ``prior`` (one positive value per passage) steers which postings the truncated
        index keeps (``bes.library.index``); None means uniform.
        """
        titles: list[str] = []
        tids: list[int] = []
        last = None
        for title, _ in passages:
            if title != last:
                titles.append(title)
                last = title
            tids.append(len(titles) - 1)
        if len(set(titles)) != len(titles):
            raise ValueError("passages of one title must be contiguous")
        packed = pack([b for _, b in passages], tids, titles, pack_cfg)
        index = TruncatedIndex.build(passages, index_cfg, prior)
        lib = cls(packed, index, cfg, count_tokens, meta)
        if out_dir is not None:
            lib.save(out_dir)
        return lib

    def save(self, out_dir: str) -> None:
        from bes.library.pack import _atomic_write

        self.packed.save(out_dir)
        self.index.save(out_dir)
        meta = dict(self.meta)
        meta.update(retrieve_cfg=dataclasses.asdict(self.cfg), accounting=self.accounting())
        _atomic_write(os.path.join(out_dir, "library.json"), json.dumps(meta, indent=1).encode())

    @classmethod
    def load(cls, path: str, count_tokens: CountTokens = default_count_tokens,
             cfg: RetrieveCfg | None = None, mmap: bool = True) -> "Library":
        meta = json.loads(open(os.path.join(path, "library.json")).read())
        if cfg is None:
            rc = dict(meta.get("retrieve_cfg", {}))
            gate = GateCfg(**{**rc.get("gate", {}),
                              "allowed_types": tuple(rc.get("gate", {}).get("allowed_types",
                                                                         GateCfg().allowed_types))})
            cfg = RetrieveCfg(max_tokens=rc.get("max_tokens", RetrieveCfg().max_tokens), gate=gate)
        return cls(Packed.load(path, mmap), TruncatedIndex.load(path, mmap), cfg, count_tokens, meta)

    # ------------------------------------------------------------------ accounting
    def accounting(self) -> dict[str, int]:
        """Exact stored numbers: text blocks, directory, titles (library) and index."""
        a = dict(self.packed.accounting())
        a.update(self.index.accounting())
        a["total_numbers"] = a["library_numbers"] + a["index_numbers"]
        return a

    # ------------------------------------------------------------------ lookups
    @property
    def n_passages(self) -> int:
        return self.packed.n_passages

    def title_range(self, tid: int, log: ReadLog | None = None) -> range:
        """Passage ids of title ``tid`` (reads two ``title_first`` entries)."""
        if log is not None:
            log.touch_dir("title_first", (tid, tid + 1), 2)
        return range(int(self.packed.title_first[tid]), int(self.packed.title_first[tid + 1]))

    def title_id_of(self, pid: int, log: ReadLog | None = None) -> int:
        t, probes = self.packed.title_of(pid)
        if log is not None:
            log.touch_dir("title_first", probes, 2)
        return t

    def title_text(self, tid: int, log: ReadLog | None = None) -> str:
        tpb = self.packed.cfg.titles_per_block
        tb = tid // tpb
        if log is not None and tb not in log.title_block_ids:
            log.title_block_ids.add(tb)
            log.title_blocks += self.packed.title_block_numbers(tb)
        return self.packed.read_title_block(tb)[tid % tpb]

    def find_title(self, title: str) -> int | None:
        """Title id of an exact title (training-side helper; not used for scoring)."""
        if self._title_map is None:
            self._title_map = {t: i for i, t in enumerate(self.packed.all_titles())}
        return self._title_map.get(title)

    def passage(self, pid: int, score: float = 0.0, log: ReadLog | None = None) -> Passage:
        """Read one passage: one text block (+ one title block for its title)."""
        b, s, e, probes = self.packed.locate(pid)
        if log is not None:
            log.touch_dir("block_first", probes, 2)
            nxt = [pid + 1] if pid + 1 < int(self.packed.block_first[b + 1]) else []
            log.touch_dir("pas_off", [pid] + nxt, 1)
            if b not in log.blocks:
                log.blocks.add(b)
                log.text_blocks += self.packed.block_numbers(b)
        body = self.packed.read_block(b)[s:e].decode("utf-8")
        tid = self.title_id_of(pid, log)
        title = self.title_text(tid, log)
        text = cap_tokens(title, body, self.cfg.max_tokens, self.count_tokens)
        return Passage(pid, tid, title, body, score, text)

    # ------------------------------------------------------------------ scoring
    def search(self, query: str, exclude_pids: Iterable[int] = (), exclude_titles: Iterable[int] = (),
               log: ReadLog | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(pids, scores) ranked by the stored index (score desc, pid asc); scores in BM25 units.

        ``exclude_pids`` / ``exclude_titles`` remove passages (e.g. the passage a training
        text came from, or its whole article) before ranking.
        """
        return self.search_max(query, exclude_pids, exclude_titles, log)[:2]

    def search_max(self, query: str, exclude_pids: Iterable[int] = (),
                   exclude_titles: Iterable[int] = (), log: ReadLog | None = None
                   ) -> tuple[np.ndarray, np.ndarray, float]:
        """``search`` plus ``s_max`` (BM25 units), the best score any passage could reach."""
        ir = IndexRead()
        pids, sc, smax = self.index.score_with_max(query, ir)
        if log is not None:
            log.index_df += ir.df_numbers
            log.index_offsets += ir.offset_numbers
            log.index_postings += ir.posting_numbers
        ex = set(int(p) for p in exclude_pids)
        ranges = [self.title_range(int(t), log) for t in exclude_titles]
        if (ex or ranges) and pids.size:
            mask = np.ones(pids.shape[0], dtype=bool)
            if ex:
                mask &= ~np.isin(pids, np.fromiter(ex, dtype=np.int64))
            for r in ranges:
                mask &= ~((pids >= r.start) & (pids < r.stop))
            pids, sc = pids[mask], sc[mask]
        return pids, sc.astype(np.float64) * self.index.scale, float(smax) * self.index.scale

    def retrieve_topk(self, query: str, k: int, exclude_pids: Iterable[int] = (),
                      exclude_titles: Iterable[int] = (), log: ReadLog | None = None) -> list[Passage]:
        """The k best passages (no gate); for experiments."""
        pids, sc = self.search(query, exclude_pids, exclude_titles, log)
        return [self.passage(int(p), float(s), log) for p, s in zip(pids[:k], sc[:k])]

    def gate_features(self, query: str, pids: np.ndarray, scores: np.ndarray,
                      log: ReadLog | None = None, smax: float | None = None) -> dict:
        """Features of the gate for a ranked list (reads the top title)."""
        if pids.size == 0:
            return {"s1": 0.0, "s2": 0.0, "margin": 0.0, "title_match": 0.0, "coverage": 0.0,
                    "qtype": query_type(query), "tid": -1}
        s1 = float(scores[0])
        tid = self.title_id_of(int(pids[0]), log)
        r = self.title_range(tid, log)
        s2 = 0.0
        for p, s in zip(pids[: self.cfg.gate.margin_candidates], scores[: self.cfg.gate.margin_candidates]):
            if not (r.start <= int(p) < r.stop):
                s2 = float(s)
                break
        title = self.title_text(tid, log)
        tt = set(terms(strip_disambiguator(title)))
        qt = set(terms(query))
        tm = (len(tt & qt) / len(tt)) if tt else 0.0
        cov = 1.0 if smax is None else (min(1.0, s1 / smax) if smax > 0 else 0.0)
        return {"s1": s1, "s2": s2, "margin": (s1 - s2) / s1 if s1 > 0 else 0.0,
                "title_match": tm, "coverage": cov, "qtype": query_type(query), "tid": tid}

    def gate_pass(self, f: dict, gate: GateCfg | None = None) -> bool:
        g = gate or self.cfg.gate
        if f["qtype"] not in g.allowed_types or f["s1"] < g.min_score:
            return False
        if f.get("coverage", 1.0) < g.min_coverage:
            return False
        if f["margin"] >= g.min_margin:
            return True
        return f["title_match"] >= g.min_title_match and f["margin"] >= g.min_margin_title

    def retrieve(self, query: str, exclude_pids: Iterable[int] = (), exclude_titles: Iterable[int] = (),
                 log: ReadLog | None = None) -> Passage | None:
        """At most one passage for a user message, or None when the gate does not pass."""
        if query_type(query) not in self.cfg.gate.allowed_types:
            return None
        pids, sc, smax = self.search_max(query, exclude_pids, exclude_titles, log)
        f = self.gate_features(query, pids, sc, log, smax)
        if not self.gate_pass(f):
            return None
        return self.passage(int(pids[0]), float(sc[0]), log)
