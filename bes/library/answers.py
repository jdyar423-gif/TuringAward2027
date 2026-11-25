"""Answers library: stored (question, answer) pairs written by strong teachers, retrieved by
strict question-to-question matching (``docs/PLAN_M6.md``).

The stored numbers of the model that no token reads (``docs/ACCOUNTING.md``) hold text, not
trained weights: writing an answer into the library costs no training step. The core reads a
few entries per reply:

* **exemplars**: the best stored questions for the user's message, shown with their answers in
  one ``LIBRARY`` turn right before the user turn (``retrieve``); training puts them there the
  same way (``bes.data.rag_format.insert_retrieval``), so the core learns to answer by analogy
  with them;
* **verbatim**: when the stored question asks for the same thing (``verbatim``: same question
  type, nearly all terms shared in both directions and no rare term left unshared on either
  side, no contrast between the two questions, the answer meets the user's explicit
  constraints), the stored answer is the reply.

Layout (every array is counted; one stored number = 2 bytes; ``answers.json`` holds the
settings, the sources and the accounting):

entries   ``ans_blob``, ``ans_block_off``, ``ans_block_raw``, ``ans_block_crc``,
          ``ans_block_first``, ``ans_entry_off``: UTF-8 ``question + "\\n" + answer`` in
          independent xz blocks (``bes.library.pack.EntryBlocks``)
per entry ``ans_mass``  uint16 [E]   idf mass of the question's distinct term buckets in
                                     1/``MASS_UNIT`` (rounded down)
          ``ans_qtype`` uint8  [E]   question type (``bes.library.qa.QTYPES``)
          ``ans_src``   uint8  [E]   index into ``sources`` of ``answers.json``
index     ``ans_post``   uint32 [M]  entry ids grouped by term bucket (ascending in a bucket);
                                     buckets with more than ``df_cap`` entries have no list
          ``ans_counts`` uint16 [2**n_bits]   entries per bucket (document frequency, clipped)
          ``ans_coarse`` uint32 [2**(n_bits-g) + 1]   offset of every group of 2**g buckets

Matching. Terms are ``bes.library.terms.terms`` (stopwords removed, plural-stemmed). A query
reads the posting lists of its term buckets in increasing document frequency (at most
``read_postings`` postings, never a bucket over ``df_cap``), which gives every candidate a
partial matched mass. The ``verify_top`` best candidates are then read (their text blocks)
and scored exactly on the term strings, so hash collisions and unindexed common terms do not
decide a match: with ``idf(t)`` from the bucket of ``t``,

    M = sum of idf over shared terms,  cov_q = M / W(query),  cov_e = M / W(entry),
    sym = min(cov_q, cov_e).

``contrast`` (``bes.library.qa``) flags pairs that differ in numbers, order, degree, relation
or polarity words or in a capitalised name; such a candidate is never used. Candidates are
ranked by ``sym`` minus ``type_penalty`` when the question types differ.

No evaluation question is stored (the builder drops every pair whose normalised question
equals a prompt of ``bes/eval/proxy/prompts.jsonl``), and with ``train=True`` (the loader) a
stored question equal to the sample's own is skipped, so a training sample never sees its
own answer.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from bes.library.pack import EntryBlocks, PackCfg, _atomic_save_npy, _atomic_write, numbers_of, pack_entries
from bes.library.qa import contrast, contrast_mismatch, qtype, question_key
from bes.library.terms import bucket, terms

ANSWERS_JSON = "answers.json"
MASS_UNIT = 64
PASSAGE_SEP = "\n\n---\n\n"          # as bes.data.sources.PASSAGE_SEP (not imported: no cycle)
NUMBERS_ONLY = "numbers_only"       # a contrast of the loose mode: the questions differ in their numbers alone
ALL_TYPES = ("question", "lookup", "task", "math", "other")


@dataclass(frozen=True)
class AnswerCfg:
    n_bits: int = 20                 # term buckets of the index
    group_bits: int = 6              # one coarse offset per 64 buckets
    df_cap: int = 16384              # buckets with more entries have no posting list
    read_postings: int = 131072      # postings read per query at most (lowest df first)
    max_query_terms: int = 24        # distinct query terms considered at most
    min_partial: float = 0.3         # candidates need partial mass >= min_partial * W(query)
    verify_top: int = 12             # candidates whose text is read for the exact score
    block_raw_bytes: int = 65536
    type_penalty: float = 0.1        # taken off sym when the question types differ
    exemplar_min: float = 0.5        # an exemplar needs at least this ranked score
    exemplar_min_math: float = 0.75  # ... for an arithmetic message (other numbers, another problem)
    second_min: float = 0.6          # a second exemplar needs at least this
    exemplars: int = 2               # exemplars in one LIBRARY turn at most
    exemplar_max_chars: int = 1400   # an exemplar's answer is cut here (at a line or sentence end)
    verbatim_cov_q: float = 0.8      # verbatim: share of the user's term mass the entry has
    verbatim_cov_e: float = 0.55     # ... and of the entry's term mass the user has
    verbatim_max_miss: float = 5.0   # ... and no unshared term on either side is rarer than this idf
    cache_blocks: int = 64           # decompressed blocks kept per process (LRU)
    part_factor: float = 0.9         # ranked score of a match with a part of the message (sub_queries)
    read_cap: int = 700_000          # stored numbers one reply may read at most (enforced: AnswerRead)
    max_query_chars: int = 1000      # a longer message is not looked up (a document, not a question)


@dataclass(frozen=True)
class _Gate:
    allowed_types: tuple[str, ...] = ALL_TYPES


@dataclass(frozen=True)
class _ViewCfg:
    """What ``bes.data.rag_format.LibraryView`` reads from ``lib.cfg``."""

    gate: _Gate = field(default_factory=_Gate)


@dataclass(frozen=True)
class Passage:
    """The contract of ``bes.library.retrieve.Passage`` (``kind`` is always ``"qa"``)."""

    pid: int
    title_id: int
    title: str        # the stored question of the best exemplar
    body: str         # its stored answer
    score: float      # its ranked score
    text: str         # what the LIBRARY turn shows: one or two "Q: ...\nA: ..." exemplars
    kind: str = "qa"
    prob: float | None = None


@dataclass(frozen=True)
class Candidate:
    entry: int
    question: str
    answer: str
    sym: float
    cov_q: float
    cov_e: float
    type_match: bool
    contrast: str | None      # why the two questions differ (None: no contrast found)
    same_question: bool       # the normalised questions are equal
    miss_q: float = 0.0       # largest idf among the user's terms the entry lacks (0: none)
    miss_e: float = 0.0       # largest idf among the entry's terms the user lacks
    part: bool = False        # matched with a part of the message, not the whole of it
    fit_value: float | None = None   # verbatim(): the reader's fit this entry was taken with (else None)

    @property
    def rank(self) -> float:
        return self.sym


@dataclass
class AnswerRead:
    """Distinct stored numbers one reply reads (every query of the reply shares one log: the
    whole message, its parts, the verbatim rule and the exemplars). ``cap`` > 0 is the most it
    may read (``AnswerCfg.read_cap``): a posting list or a text block that would pass it is
    not read (``refused`` counts them), so the bound of ``AnswerLibrary.read_bound`` holds for
    every reply by construction."""

    index: int = 0
    per_entry: int = 0
    text_blocks: int = 0
    blocks: set = field(default_factory=set, repr=False)
    lists: set = field(default_factory=set, repr=False)
    cap: int = 0
    refused: int = 0

    @property
    def total(self) -> int:
        return self.index + self.per_entry + self.text_blocks

    def room(self, n: int) -> bool:
        """Whether ``n`` more numbers may be read."""
        return self.cap <= 0 or self.total + int(n) <= self.cap


class ReadCap(Exception):
    """A text block was not read: the reply's read budget (``AnswerRead.cap``) is used up."""


#: Index terms of format instructions (stemmed as ``bes.library.terms.terms`` stems them).
_FORMAT_TERMS = frozenset(
    """answer respond reply response use write keep make sure exactly least most one two three four
    five six seven eight nine ten word sentence paragraph line bullet point item list numbered
    lowercase uppercase letter json format start end begin single short brief briefly include
    contain avoid without comma question""".split())
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
_LEAD_CLAUSE = re.compile(r"^[^,.?!]{3,80},\s+")


def sub_queries(message: str) -> list[str]:
    """Parts of a user message that may be a stored question on their own: its sentences when
    it has several ("What are the primary colors? Answer in lowercase.") and what follows a
    leading clause that ends in a comma ("In exactly three bullet points, list ..."). A request
    that carries a format instruction then still finds the entry about its subject. Parts with
    fewer than two index terms that are not format words (``_FORMAT_TERMS``: "Use exactly
    three bullet points." has none) are left out."""
    text = " ".join(message.split())
    parts: list[str] = []
    sents = [s for s in _SENTENCE_END.split(text) if s]
    if len(sents) > 1:
        parts += sents
    m = _LEAD_CLAUSE.match(text)
    if m is not None and len(text) - m.end() >= 10:
        parts.append(text[m.end():])
    out: list[str] = []
    for p in parts:
        if p == text or p in out or len(set(terms(p)) - _FORMAT_TERMS) < 2:
            continue                                   # a format instruction, not a subject
        out.append(p)
    return out[:4]


def cut_answer(answer: str, max_chars: int) -> str:
    """``answer`` within ``max_chars``: cut at the last line end, else sentence end, else word
    end before the limit (never inside an open code fence when a line end is available)."""
    if len(answer) <= max_chars:
        return answer
    head = answer[:max_chars]
    for sep in ("\n\n", "\n", ". ", " "):
        i = head.rfind(sep)
        if i >= max_chars // 2:
            cut = head[:i + (1 if sep == ". " else 0)].rstrip()
            if cut.count("```") % 2 == 1:           # do not end inside a code block
                j = cut.rfind("```")
                cut = cut[:j].rstrip() if j >= max_chars // 4 else cut + "\n```"
            return cut
    return head.rstrip()


#: The first line of an exemplar shown with its match tag (``retrieve(tag=True)``): how near the
#: stored question is to the user's as the index measures it, which the reader of a LIBRARY turn
#: cannot see otherwise. ``close``: ranked score >= CLOSE_MIN; ``similar``: at least the library's
#: own floor; ``loose``: under it (the loose mode, and the distractors of training); ``other
#: numbers``: the questions differ in their numbers and in nothing else.
CLOSE_MIN = 0.7
MATCH_CLOSE, MATCH_SIMILAR, MATCH_LOOSE, MATCH_NUMBERS = ("Match: close", "Match: similar", "Match: loose",
                                                          "Match: other numbers")


def match_tag(sym: float, contrast: str | None, exemplar_min: float) -> str:
    if contrast == NUMBERS_ONLY:
        return MATCH_NUMBERS
    if sym >= CLOSE_MIN - 1e-12:
        return MATCH_CLOSE
    return MATCH_SIMILAR if sym >= exemplar_min - 1e-12 else MATCH_LOOSE


def render_exemplar(question: str, answer: str, max_chars: int, tag: str | None = None) -> str:
    head = f"{tag}\n" if tag else ""
    return f"{head}Q: {question}\nA: {cut_answer(answer, max_chars)}"


class AnswerLibrary:
    """Stored answers + their index (module docstring). Implements the library contract of
    ``bes.data.rag_format.LibraryView`` with ``kind_range("qa")`` = every entry."""

    PER_ENTRY = ("mass", "qtype", "src")
    INDEX = ("post", "counts", "coarse")
    HEADER_NUMBERS = 2 + 1 + 1 + 1   # n_entries (32-bit), n_bits, group_bits, df_cap
    SMALL_READS = 4096               # unchecked reads of one reply at most (new_read)

    def __init__(self, cfg: AnswerCfg, blocks: EntryBlocks, mass: np.ndarray, qtypes: np.ndarray,
                 src: np.ndarray, post: np.ndarray, counts: np.ndarray, coarse: np.ndarray,
                 meta: dict | None = None, train: bool = False, path: str | None = None):
        self.path = path
        self.acfg = cfg
        self.cfg = _ViewCfg()
        self.blocks = blocks
        self.mass = mass
        self.qtype = qtypes
        self.src = src
        self.post = post
        self.counts = counts
        self.coarse = coarse
        self.meta = dict(meta or {})
        self.train = bool(train)
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._questions: list[str] | None = None
        self._kept = None
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------ pickling (workers)
    def __getstate__(self) -> dict:
        if self.path is not None:              # a worker opens the files itself (memory maps)
            return {"_reload": (self.path, self.train, dataclasses.asdict(self.acfg), self._questions is not None)}
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_questions"] = None
        state["_kept"] = None
        state["_idf"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        if "_reload" in state:
            path, train, cfg, preloaded = state["_reload"]
            other = AnswerLibrary.load(path, train=train, **cfg)
            self.__dict__.update(other.__dict__)
            if preloaded:
                self.preload()
        else:
            self.__dict__.update(state)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, pairs: Sequence[tuple[str, str, str]], cfg: AnswerCfg = AnswerCfg(),
              meta: dict | None = None) -> "AnswerLibrary":
        """Pack and index ``pairs`` = ``(question, answer, source)`` in the order given.
        Questions must be one line; no two may share a ``question_key``."""
        if cfg.block_raw_bytes > 65536:
            raise ValueError("block_raw_bytes above 65536 does not fit the uint16 entry offsets")
        sources = sorted({p[2] for p in pairs})
        if len(sources) > 255:
            raise ValueError("more than 255 sources")
        keys = set()
        for q, a, _ in pairs:
            if "\n" in q or not q.strip() or not a.strip():
                raise ValueError(f"bad pair: {q[:60]!r}")
            k = question_key(q)
            if k in keys:
                raise ValueError(f"duplicate question: {q[:60]!r}")
            keys.add(k)
        blocks = pack_entries([(q + "\n" + a).encode("utf-8") for q, a, _ in pairs],
                              PackCfg(block_raw_bytes=cfg.block_raw_bytes))
        n, nb = len(pairs), 1 << cfg.n_bits
        ent = [sorted({bucket(t, cfg.n_bits) for t in terms(q)}) for q, _, _ in pairs]
        lens = np.fromiter((len(b) for b in ent), dtype=np.int64, count=n)
        flat_b = np.fromiter((b for bs in ent for b in bs), dtype=np.int64, count=int(lens.sum()))
        flat_e = np.repeat(np.arange(n, dtype=np.int64), lens)
        df = np.bincount(flat_b, minlength=nb).astype(np.int64)
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        mass = np.floor(np.bincount(flat_e, weights=idf[flat_b], minlength=n) * MASS_UNIT + 1e-9)
        if mass.size and mass.max() > 0xFFFF:
            raise ValueError("question mass overflows uint16")
        keep = df[flat_b] <= cfg.df_cap
        fb, fe = flat_b[keep], flat_e[keep]
        o = np.lexsort((fe, fb))
        post = fe[o].astype(np.uint32)
        kept_df = np.where(df <= cfg.df_cap, df, 0)
        csum = np.concatenate([[0], np.cumsum(kept_df)])
        coarse = csum[np.arange(0, nb + 1, 1 << cfg.group_bits)].astype(np.uint32)
        counts = np.minimum(df, 0xFFFF).astype(np.uint16)
        qt = np.fromiter((qtype(q) for q, _, _ in pairs), dtype=np.uint8, count=n)
        src = np.fromiter((sources.index(s) for _, _, s in pairs), dtype=np.uint8, count=n)
        m = dict(meta or {})
        m["sources"] = sources
        return cls(cfg, blocks, mass.astype(np.uint16), qt, src, post, counts, coarse, m)

    # ------------------------------------------------------------------ sizes / io
    @property
    def n_entries(self) -> int:
        return int(self.mass.shape[0])

    @property
    def n_passages(self) -> int:
        return self.n_entries

    def kind_range(self, kind: str) -> range:
        if kind == "qa":
            return range(0, self.n_entries)
        if kind == "wiki":
            return range(0, 0)
        raise KeyError(kind)

    def accounting(self) -> dict[str, int]:
        a = self.blocks.accounting("ans_array_")
        per = {k: numbers_of(np.asarray(getattr(self, "qtype" if k == "qtype" else k)).nbytes) for k in self.PER_ENTRY}
        idx = {k: numbers_of(np.asarray(getattr(self, k)).nbytes) for k in self.INDEX}
        out = {"text_numbers": sum(a.values()), "entry_numbers": sum(per.values()),
               "index_numbers": sum(idx.values()) + self.HEADER_NUMBERS,
               "index_header_numbers": self.HEADER_NUMBERS,
               "entries": self.n_entries, "postings": int(self.post.shape[0]), "blocks": self.blocks.n_blocks,
               "raw_bytes": int(np.asarray(self.blocks.block_raw, dtype=np.int64).sum()), **a,
               **{f"ans_array_{k}": v for k, v in per.items()}, **{f"ans_array_{k}": v for k, v in idx.items()}}
        out["total_numbers"] = out["text_numbers"] + out["entry_numbers"] + out["index_numbers"]
        return out

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.blocks.save(out_dir, "ans_")
        for k in ("mass", "qtype", "src", "post", "counts", "coarse"):
            _atomic_save_npy(os.path.join(out_dir, f"ans_{k}.npy"), np.asarray(getattr(self, k)))
        meta = {"format": "bes-answers-1", "cfg": dataclasses.asdict(self.acfg),
                "pack_cfg": dataclasses.asdict(self.blocks.cfg), "accounting": self.accounting(), **self.meta}
        _atomic_write(os.path.join(out_dir, ANSWERS_JSON), json.dumps(meta, indent=1, sort_keys=True).encode())

    @classmethod
    def load(cls, out_dir: str, mmap: bool = True, train: bool = False, **overrides: Any) -> "AnswerLibrary":
        meta = json.loads(open(os.path.join(out_dir, ANSWERS_JSON)).read())
        if meta.pop("format", None) != "bes-answers-1":
            raise ValueError(f"{out_dir} is not an answers library")
        cfg = dataclasses.replace(AnswerCfg(**meta.pop("cfg")), **overrides)
        pcfg = PackCfg(**meta.pop("pack_cfg"))
        meta.pop("accounting", None)
        mode = "r" if mmap else None
        arr = {k: np.load(os.path.join(out_dir, f"ans_{k}.npy"), mmap_mode=mode)
               for k in ("mass", "qtype", "src", "post", "counts", "coarse")}
        blocks = EntryBlocks.load(out_dir, "ans_", pcfg, mmap)
        return cls(cfg, blocks, arr["mass"], arr["qtype"], arr["src"], arr["post"], arr["counts"],
                   arr["coarse"], meta, train=train, path=str(out_dir))

    def for_training(self) -> "AnswerLibrary":
        """The same library with ``train=True`` (a stored question equal to the query's is
        skipped) and every stored question in memory (``preload``): what the loader uses."""
        other = AnswerLibrary(self.acfg, self.blocks, self.mass, self.qtype, self.src, self.post,
                              self.counts, self.coarse, self.meta, train=True, path=self.path)
        return other.preload()

    # ------------------------------------------------------------------ what the run checks
    @property
    def n_wiki(self) -> int:
        return 0

    @property
    def n_qa(self) -> int:
        return self.n_entries

    @property
    def fingerprint(self) -> str:
        """Identity of the stored content: size, entries and the checksums of the text blocks."""
        import hashlib

        h = hashlib.sha256()
        h.update(json.dumps([self.n_entries, int(self.post.shape[0]), dataclasses.asdict(self.acfg)["n_bits"]]).encode())
        h.update(np.asarray(self.blocks.block_crc).tobytes())
        return h.hexdigest()[:16]

    def read_bound(self, max_query_chars: int = 1000) -> dict[str, int]:
        """Upper bound of the stored numbers one query reads: the posting lists (at most
        ``read_postings`` postings of two numbers), their directory entries, the per-entry
        rows of the candidates and the text blocks of the ``verify_top`` candidates."""
        cfg = self.acfg
        block = max((self.blocks.block_numbers(b) for b in range(self.blocks.n_blocks)), default=0)
        index = 2 * cfg.read_postings + cfg.max_query_terms * (3 + (1 << cfg.group_bits))
        per_entry = cfg.read_postings
        text = cfg.verify_top * block
        one = index + per_entry + text
        if cfg.read_cap > 0:
            # a reply asks several queries (the message, its parts, the verbatim rule); what
            # holds for all of them together is the cap its read log enforces (new_read)
            return {"index": min(index, cfg.read_cap), "per_entry": 0, "text_blocks": cfg.read_cap - min(index, cfg.read_cap),
                    "one_query": one, "total": cfg.read_cap}
        return {"index": index, "per_entry": per_entry, "text_blocks": text, "one_query": one, "total": one}

    def new_read(self) -> AnswerRead:
        """The read log of one reply. Its cap is ``AnswerCfg.read_cap`` minus ``SMALL_READS``:
        the document frequencies of the terms of a message and of its candidates are read
        without a check (one number per distinct term; a message has at most
        ``max_query_chars`` characters, a reply at most six queries of twelve candidates:
        fewer than ``SMALL_READS`` numbers), so a reply never reads more than ``read_cap``."""
        return AnswerRead(cap=max(1, int(self.acfg.read_cap) - self.SMALL_READS))

    # ------------------------------------------------------------------ text access
    def _block(self, b: int, log: AnswerRead | None = None) -> bytes:
        if log is not None and b not in log.blocks:
            n = self.blocks.block_numbers(b)
            if not log.room(n):
                log.refused += 1
                raise ReadCap(b)
            log.blocks.add(b)
            log.text_blocks += n
        hit = self._cache.get(b)
        if hit is not None:
            self._cache.move_to_end(b)
            return hit
        raw = self.blocks.read_block(b)
        self._cache[b] = raw
        while len(self._cache) > max(1, self.acfg.cache_blocks):
            self._cache.popitem(last=False)
        return raw

    def entry(self, e: int, log: AnswerRead | None = None) -> tuple[str, str]:
        b, s, t, _ = self.blocks.locate(int(e))
        q, _, a = self._block(b, log)[s:t].decode("utf-8").partition("\n")
        return q, a

    def preload(self) -> "AnswerLibrary":
        """Read every stored question into memory (training: the exact score of a candidate
        then needs no block); the answers stay in their blocks."""
        if self._questions is None:
            qs: list[str] = []
            for b in range(self.blocks.n_blocks):
                raw = self.blocks.read_block(b)
                lo, hi = int(self.blocks.block_first[b]), int(self.blocks.block_first[b + 1])
                offs = [int(self.blocks.entry_off[e]) for e in range(lo, hi)] + [len(raw)]
                for i in range(hi - lo):
                    qs.append(raw[offs[i]:offs[i + 1]].decode("utf-8").partition("\n")[0])
            self._questions = qs
        return self

    def question(self, e: int, log: AnswerRead | None = None) -> str:
        if self._questions is not None:
            return self._questions[int(e)]
        return self.entry(e, log)[0]

    # ------------------------------------------------------------------ index
    def idf_df(self, df: int) -> float:
        n = self.n_entries
        return math.log1p((n - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        v = self._idf.get(term)
        if v is None:
            v = self.idf_df(int(self.counts[bucket(term, self.acfg.n_bits)]))
            if len(self._idf) < 1_000_000:           # per process; the same number every time
                self._idf[term] = v
        return v

    def _kept_counts(self) -> np.ndarray:
        if self._kept is None:
            c = np.asarray(self.counts, dtype=np.int64)
            self._kept = np.where(c <= self.acfg.df_cap, c, 0)
        return self._kept

    def bucket_range(self, b: int) -> tuple[int, int]:
        g = self.acfg.group_bits
        kept = self._kept_counts()
        g0 = (b >> g) << g
        start = int(self.coarse[b >> g]) + int(kept[g0:b].sum())
        return start, start + int(kept[b])

    def partial(self, query: str, exclude: Iterable[int] = (), log: AnswerRead | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(entries, partial_mass)`` of the candidates the index gives, best first: matched
        idf mass over the posting lists read (module docstring)."""
        cfg = self.acfg
        qb = list(dict.fromkeys(bucket(t, cfg.n_bits) for t in terms(query)))[:cfg.max_query_terms]
        empty = (np.zeros(0, np.int64), np.zeros(0))
        if not qb or self.n_entries == 0:
            return empty
        dfs = [int(self.counts[b]) for b in qb]
        w = [self.idf_df(d) for d in dfs]
        wq = float(sum(w))
        if log is not None:
            log.index += len(qb)
        parts_e, parts_w, read = [], [], 0
        for d, wb, b in sorted(zip(dfs, w, qb)):
            if d == 0 or d > cfg.df_cap or read + d > cfg.read_postings:
                continue
            if log is not None and b not in log.lists:
                cost = 2 * d + 2 + (1 << cfg.group_bits)
                if not log.room(cost):
                    log.refused += 1
                    continue
                log.lists.add(b)
                log.index += cost
            s, e = self.bucket_range(b)
            parts_e.append(np.asarray(self.post[s:e], dtype=np.int64))
            parts_w.append(np.full(e - s, wb))
            read += d
        if not parts_e:
            return empty
        u, inv = np.unique(np.concatenate(parts_e), return_inverse=True)
        m = np.bincount(inv, weights=np.concatenate(parts_w))
        keep = m >= cfg.min_partial * wq - 1e-12
        ex = np.fromiter((int(x) for x in exclude), dtype=np.int64)
        if ex.size:
            keep &= ~np.isin(u, ex)
        u, m = u[keep], m[keep]
        if log is not None and log.cap > 0 and not log.room(int(u.size)):
            # the stored masses of the candidates with the most matched mass only
            left = max(0, log.cap - log.total)
            top = np.lexsort((u, -m))[:left]
            top.sort()
            u, m = u[top], m[top]
            log.refused += 1
        # rank by the partial symmetric score (the entry's whole mass is stored)
        we = np.asarray(self.mass[u], dtype=np.float64) / MASS_UNIT
        part = np.minimum(m / wq, np.minimum(1.0, m / np.maximum(we, 1e-9)))
        o = np.lexsort((u, -np.round(part, 9)))
        if log is not None:
            log.per_entry += int(u.size)
        return u[o], part[o]

    def candidates(self, query: str, k: int | None = None, exclude: Iterable[int] = (),
                   log: AnswerRead | None = None, floor: float | None = None) -> list[Candidate]:
        """The best stored questions for ``query`` with their exact scores, best first (at most
        ``k``, default ``verify_top``); candidates with a contrast come last. The contrast is
        looked for only where the ranked score reaches the lowest threshold any rule uses (a
        candidate below it is neither an exemplar nor a verbatim answer whatever it holds).
        ``answer`` is left empty: the callers read the answers of the entries they pick.

        ``floor`` (the loose mode of :meth:`exemplars`): the contrast is looked for down to this
        score, and a stored question that differs from the user's in its numbers and in nothing
        else gets the contrast :data:`NUMBERS_ONLY` and is ranked with the candidates without
        one (the same problem with other numbers: a worked example, not an answer)."""
        cfg = self.acfg
        need = min(cfg.exemplar_min, cfg.second_min, cfg.verbatim_cov_q, cfg.verbatim_cov_e) - 1e-9
        if floor is not None:
            need = min(need, float(floor) - 1e-9)
        ents, _ = self.partial(query, exclude, log)
        if ents.size == 0:
            return []
        q_terms = set(terms(query))
        if not q_terms:
            return []
        wq = sum(self.idf(t) for t in q_terms)
        if log is not None:
            log.index += len(q_terms)
        qt = qtype(query)
        qkey = question_key(query)
        user_contrast = None
        out: list[Candidate] = []
        for e in ents[:cfg.verify_top].tolist():
            try:
                question = self.question(e, log)
            except ReadCap:
                break
            s_terms = set(terms(question))
            if not s_terms:
                continue
            if log is not None:
                log.index += len(s_terms - q_terms)
            m = sum(self.idf(t) for t in q_terms & s_terms)
            we = sum(self.idf(t) for t in s_terms)
            cov_q, cov_e = m / wq, m / we
            sym = min(cov_q, cov_e)
            same = question_key(question) == qkey
            if self.train and same:
                continue
            tm = int(self.qtype[e]) == qt
            ranked = sym - (0.0 if tm else cfg.type_penalty)
            mm = None
            if ranked >= need:
                if user_contrast is None:
                    user_contrast = contrast(query)
                stored_contrast = contrast(question)
                mm = contrast_mismatch(user_contrast, stored_contrast)
                if mm == "numbers" and floor is not None and contrast_mismatch(
                        dataclasses.replace(user_contrast, numbers=stored_contrast.numbers), stored_contrast) is None:
                    mm = NUMBERS_ONLY
            miss_q = max((self.idf(t) for t in q_terms - s_terms), default=0.0)
            miss_e = max((self.idf(t) for t in s_terms - q_terms), default=0.0)
            out.append(Candidate(int(e), question, "", ranked, cov_q, cov_e, tm, mm, same, miss_q, miss_e))
        out.sort(key=lambda c: (c.contrast not in (None, NUMBERS_ONLY), -round(c.sym, 9), c.entry))
        return out[:k] if k is not None else out

    def _answer(self, e: int, log: AnswerRead | None = None) -> str:
        """The stored answer of entry ``e``; ``""`` when the reply's read cap refuses its block."""
        try:
            return self.entry(e, log)[1]
        except ReadCap:
            return ""

    # ------------------------------------------------------------------ library contract
    def passage(self, pid: int, score: float = 0.0, log: AnswerRead | None = None) -> Passage:
        q, a = self.entry(int(pid), log)
        return Passage(int(pid), -1, q, a, float(score), render_exemplar(q, a, self.acfg.exemplar_max_chars))

    def title_id_of(self, pid: int) -> int:
        return -1

    def title_range(self, tid: int) -> range:
        return range(0, 0)

    def find_title(self, title: str) -> None:
        return None

    def exemplars(self, query: str, exclude_pids: Iterable[int] = (), log: AnswerRead | None = None,
                  floor: float | None = None, numbers: bool = True) -> list[Candidate]:
        """The exemplars a LIBRARY turn shows for ``query``, best first (possibly none).

        ``floor``: the loose mode. The best exemplar may have a ranked score down to ``floor``,
        whatever the kind of message (default: ``exemplar_min``, and ``exemplar_min_math`` for
        an arithmetic message), and a stored question that differs from the user's only in its
        numbers is an exemplar too (:data:`NUMBERS_ONLY`; by default every contrast excludes an
        entry; ``numbers=False`` keeps them out in the loose mode as well). A model uses such
        exemplars well only when it was trained with them (``DataCfg.exemplar_floor``)."""
        cfg = self.acfg
        if len(query) > cfg.max_query_chars:
            return []
        ok = (None, NUMBERS_ONLY) if (floor is not None and numbers) else (None,)
        best = {c.entry: c for c in self.candidates(query, None, exclude_pids, log, floor) if c.contrast in ok}
        for part in sub_queries(query):
            for c in self.candidates(part, cfg.exemplars, exclude_pids, log, floor):
                score = c.sym * cfg.part_factor
                if c.contrast in ok and (c.entry not in best or score > best[c.entry].sym):
                    best[c.entry] = dataclasses.replace(c, sym=score, part=True)
        cands = sorted(best.values(), key=lambda c: (-round(c.sym, 9), c.entry))
        if floor is not None:
            floor = float(floor)
        else:
            floor = cfg.exemplar_min
            if cfg.exemplar_min_math > floor:
                from bes.library.retrieve import query_type

                if query_type(query) == "math":
                    floor = cfg.exemplar_min_math
        if not cands or cands[0].sym < floor - 1e-12:
            return []
        picks = cands[:1] + [c for c in cands[1:cfg.exemplars] if c.sym >= cfg.second_min - 1e-12]
        picks = [c if c.answer else dataclasses.replace(c, answer=self._answer(c.entry, log)) for c in picks]
        return [c for c in picks if c.answer]         # without its answer (read cap) an entry is no exemplar

    def retrieve(self, query: str, exclude_pids: Iterable[int] = (), exclude_titles: Iterable[int] = (),
                 log: AnswerRead | None = None, kinds: Sequence[str] | None = None,
                 context: Sequence[str] | None = None, floor: float | None = None,
                 tag: bool = False, numbers: bool = True) -> Passage | None:
        """The LIBRARY turn for a user message, or ``None``: one or two exemplars, the best
        one last (next to the user turn). A later turn that leans on the conversation
        (``bes.library.retrieve.self_contained``) retrieves nothing. ``floor``: see
        :meth:`exemplars`, as ``numbers``. ``tag``: every exemplar starts with its match tag (:func:`match_tag`);
        a model reads the tags well only when it was trained with them
        (``DataCfg.exemplar_tag``)."""
        if kinds is not None and "qa" not in kinds:
            return None
        if context:
            from bes.library.retrieve import self_contained

            if not self_contained(query):
                return None
        picks = self.exemplars(query, exclude_pids, log, floor, numbers)
        if not picks:
            return None
        mc = self.acfg.exemplar_max_chars
        tags = [match_tag(c.sym, c.contrast, self.acfg.exemplar_min) if tag else None for c in picks]
        text = PASSAGE_SEP.join(render_exemplar(c.question, c.answer, mc, t)
                                for c, t in reversed(list(zip(picks, tags))))
        best = picks[0]
        return Passage(best.entry, -1, best.question, best.answer, float(best.sym), text)

    def retrieve_topk(self, query: str, k: int, exclude_pids: Iterable[int] = (),
                      exclude_titles: Iterable[int] = (), log: AnswerRead | None = None,
                      kinds: Sequence[str] | None = None) -> list[Passage]:
        if k <= 0 or (kinds is not None and "qa" not in kinds) or len(query) > self.acfg.max_query_chars:
            return []
        mc = self.acfg.exemplar_max_chars
        out = []
        for c in self.candidates(query, k, exclude_pids, log):
            q, a = c.question, c.answer or self._answer(c.entry, log)
            if a:
                out.append(Passage(c.entry, -1, q, a, float(c.sym), render_exemplar(q, a, mc)))
        return out

    def verbatim(self, query: str, spec: Any = None, context: Sequence[str] | None = None,
                 log: AnswerRead | None = None, min_rank: float | None = None, fit: Any = None,
                 min_fit: float | None = None, fit_rank: float | None = None) -> Candidate | None:
        """The stored entry whose answer can be given as the reply, or ``None`` (module
        docstring). ``spec``: the user's compiled constraints (``bes.exact.constraints``);
        an answer that violates one is not used.

        ``min_rank`` (``DecodeCfg.verbatim_rank``): a second rule for "the library holds this
        question", beside the library's own (an entry that passes either is used): the entry's
        exact overlap ``min(cov_q, cov_e)``, less ``type_penalty`` when the question types
        differ, reaches ``min_rank``. A contrast and a violated constraint exclude an entry under
        both rules. (The overlap is the entry's own: a taught entry's lifted score does not
        count.)

        ``fit`` with ``min_fit`` (``DecodeCfg.verbatim_fit``; ``bes.decode.fitness.AnswerFit``): a
        third rule, asked of the reader instead of the index. ``fit(query, stored question,
        answer)`` is how much less probable the model finds the stored answer as a reply to the
        user's question than to the question it was stored under (nats per token; 0: the same).
        It is computed for the candidates neither other rule takes whose own overlap reaches
        ``fit_rank`` (default: the exemplar threshold), and the entry with the best fit is used
        when its fit is at least ``-min_fit``. The other two rules come first: an entry either
        of them takes is returned without asking the model. ``fit_value`` of the returned
        candidate is the fit it passed with."""
        cfg = self.acfg
        if len(query) > cfg.max_query_chars:
            return None
        if context:
            from bes.library.retrieve import self_contained

            if not self_contained(query):
                return None
        ask = fit is not None and min_fit is not None
        ask_from = float(cfg.exemplar_min if fit_rank is None else fit_rank)
        # the index looks for a contrast only down to the lowest threshold its own rules use:
        # a test that starts lower has to say so
        looked = min(cfg.exemplar_min, cfg.second_min, cfg.verbatim_cov_q, cfg.verbatim_cov_e)
        floor = ask_from if (ask and ask_from < looked) else None
        best: tuple[float, Candidate] | None = None
        for c in (self.candidates(query, 3, (), log) if floor is None else self.candidates(query, 3, (), log, floor)):
            if c.contrast is not None:
                continue
            own_rule = (c.type_match and c.cov_q >= cfg.verbatim_cov_q - 1e-12 and c.cov_e >= cfg.verbatim_cov_e - 1e-12
                        and max(c.miss_q, c.miss_e) <= cfg.verbatim_max_miss + 1e-12)
            own = min(c.cov_q, c.cov_e) - (0.0 if c.type_match else cfg.type_penalty)
            by_rank = min_rank is not None and own >= float(min_rank) - 1e-12
            by_fit = ask and not (own_rule or by_rank) and own >= ask_from - 1e-12
            if not (own_rule or by_rank or by_fit):
                continue
            answer = c.answer or self._answer(c.entry, log)
            if not answer:
                continue
            if spec is not None and getattr(spec, "constraints", None):
                from bes.exact import constraints

                try:
                    if constraints.verify(spec, answer):
                        continue
                except Exception:
                    continue
            if not by_fit:
                return dataclasses.replace(c, answer=answer)
            try:
                value = fit(query, c.question, answer)
            except Exception:                       # the reader could not be asked: no third rule
                value = None
            if value is not None and float(value) >= -float(min_fit) and (best is None or float(value) > best[0]):
                best = (float(value), dataclasses.replace(c, answer=answer, fit_value=float(value)))
        return None if best is None else best[1]


__all__ = ["AnswerCfg", "AnswerLibrary", "AnswerRead", "ReadCap", "Candidate", "Passage", "ANSWERS_JSON", "NUMBERS_ONLY",
           "MATCH_CLOSE", "MATCH_SIMILAR", "MATCH_LOOSE", "MATCH_NUMBERS", "CLOSE_MIN", "match_tag",
           "cut_answer", "render_exemplar", "sub_queries", "ALL_TYPES"]
