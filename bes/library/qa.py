"""Question-answer memory layer: short human-written (question, answer) pairs retrieved by
question-to-question matching (``kind="qa"`` entries of the library).

Cleaning (``clean_pairs``): questions and answers get NFKC + whitespace normalisation;
questions lose surrounding quotes and trailing punctuation and end in exactly one ``?``. A
pair is dropped when the question has no content term, the answer is empty or longer than
``max_answer_words`` words, the question is time-sensitive (``time_sensitive``: its answer
goes stale, e.g. "who is the current ...", "when was the last time ...", "how old is ...",
"this year"), its normalised form equals a question of an evaluation split
(``question_key``; never stored), or it duplicates an earlier pair (exact string, then
normalised question). Normalised duplicates whose answers disagree are dropped together
(an ambiguous question). One canonical answer per question is kept (NQ: the first
annotation; TriviaQA: ``answer.value``).

Stored layout (every array counted; one stored number = 2 bytes):

entries   ``qa_blob``, ``qa_block_off``, ``qa_block_raw``, ``qa_block_crc``,
          ``qa_block_first``, ``qa_entry_off``: UTF-8 ``question + "\\n" + answer`` in
          independent xz blocks (``bes.library.pack.EntryBlocks``)
per entry ``qa_mass``  uint16 [E]   idf mass of the question's term buckets, in 1/``MASS_UNIT``
                                    (rounded down; ``cov_e`` is capped at 1)
          ``qa_qtype`` uint8  [E]   question type (``QTYPES``)
index     ``qa_post``   uint32 [M]  entry ids grouped by term bucket (ascending inside a bucket)
          ``qa_counts`` uint16 [2**n_bits]   postings per bucket (= document frequency)
          ``qa_coarse`` uint32 [2**(n_bits-g) + 1]   offset of every group of 2**g buckets
          header: n_entries, n_bits, group_bits (``HEADER_NUMBERS``)

Matching (``QAStore.search``) is strict and symmetric. For the distinct query term buckets
``Q`` (``bes.library.terms.terms``: stopwords removed, plural-stemmed) and a stored question
with buckets ``S``, with ``idf(b) = ln(1 + (E - df + 0.5) / (df + 0.5))``,

    M = sum_{b in Q & S} idf(b),  Wq = sum_{b in Q} idf(b),  We = sum_{b in S} idf(b),
    cov_q = M / Wq,  cov_e = M / We,  sym = M / max(Wq, We) = min(cov_q, cov_e),

so "who wrote hamlet" matches "who wrote hamlet?" (sym = 1) but not "who wrote the
screenplay of the 1996 film hamlet?" (cov_e small). ``We`` is the stored ``qa_mass``; ``M``
uses the posting lists read in increasing df order until ``read_postings`` entries were
read (a skipped, very common bucket still counts in ``Wq`` and ``We``, so a match through it
is missed and the score can only fall). Candidates are ranked by ``sym`` (ties: question type
match, then entry id). The question types of the query and the entry (who / when / where /
how many / ...) are compared because the wh-words are stopwords.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from bes.config import LibraryCfg
from bes.library.pack import EntryBlocks, PackCfg, _atomic_save_npy, _atomic_write, numbers_of, pack_entries
from bes.library.terms import bucket, terms, tokens

_L = LibraryCfg()

MASS_UNIT = 64          # qa_mass stores round(We * MASS_UNIT) (uint16: We < 1024)
QTYPES = ("other", "who", "what", "when", "where", "howmany", "how", "why")

# ------------------------------------------------------------------------------ normalisation

_WS_RX = re.compile(r"\s+")
_TRAIL_Q_RX = re.compile(r"[\s?.!,;:]+$")


def normalize_space(s: str) -> str:
    """NFKC, every whitespace run (including no-break spaces) -> one space, stripped."""
    return _WS_RX.sub(" ", unicodedata.normalize("NFKC", s or "")).strip()


def clean_question(q: str) -> str:
    """Display form of a stored question ("" when nothing is left)."""
    q = normalize_space(q)
    if len(q) >= 2 and q[0] == '"' and q[-1] == '"':
        q = q[1:-1]
    q = q.replace('""', '"').strip()
    q = _TRAIL_Q_RX.sub("", q)
    return q + "?" if q else ""


def clean_answer(a: str) -> str:
    """Display form of a stored answer: normalised spaces, no trailing ``,;:`` or spaces."""
    return normalize_space(a).rstrip(",;: ").strip()


def question_key(q: str) -> str:
    """Normalised question for duplicate and evaluation-overlap checks: lower case, accents
    and punctuation removed, words (stopwords kept) joined by single spaces."""
    return " ".join(tokens(q))


_ARTICLES = frozenset(("a", "an", "the"))


def answer_key(a: str) -> str:
    """SQuAD-style answer normalisation: lower case, punctuation and articles removed."""
    return " ".join(t for t in tokens(a) if t not in _ARTICLES)


# ------------------------------------------------------------------------------ question type

_HOWMANY = frozenset("many much long old far big tall high deep large often fast heavy wide old".split())
_WHEN_NOUNS = frozenset("year years date century decade month day time era period age".split())


def qtype(question: str) -> int:
    """Index into ``QTYPES`` of the first wh-word of the question (0 = none)."""
    toks = tokens(question)
    for i, t in enumerate(toks[:12]):
        if t in ("who", "whom", "whose"):
            return 1
        if t == "when":
            return 3
        if t == "where":
            return 4
        if t == "why":
            return 7
        if t == "how":
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            return 5 if nxt in _HOWMANY else 6
        if t in ("what", "which"):
            nxt = toks[i + 1:i + 3]
            return 3 if any(x in _WHEN_NOUNS for x in nxt) else 2
    return 0


# ------------------------------------------------------------------------------ time filter

#: (rule name, pattern) - a question matching any pattern has an answer that goes stale.
TIME_RULES: tuple[tuple[str, re.Pattern], ...] = tuple((n, re.compile(p, re.I)) for n, p in (
    ("current", r"\bcurrent(?:ly)?\b(?<!alternating current)(?<!direct current)(?<!electric current)"
                r"(?<!ocean current)(?<!rip current)(?<!air current)"),
    ("latest", r"\b(?:latest|most recent|recent(?:ly)?|upcoming|newest)\b"),
    ("now", r"\bnow\b(?! (?:known|called|named|famous|considered|regarded|referred|spelled|spelt|"
            r"recogni[sz]ed|thought|believed|extinct|defunct|demolished|dissolved|abolished|"
            r"part of the past))"),
    ("relative_time", r"\b(?:this|last|next|coming) (?:year|season|week|month|weekend|summer|"
                      r"winter|spring|fall|autumn|night|time|episode|game|match|race|tournament|"
                      r"world cup|olympics|election|super bowl|series)\b|\b(?:today|tonight|"
                      r"yesterday|tomorrow|nowadays|these days|so far|to date|as of|presently|"
                      r"at present|at the moment|right now|this year's|last year's)\b"),
    ("next", r"\bnext\b(?! generation| to\b)"),
    ("last_time", r"\blast time\b|\bthe last (?:year|season|time|team|person|player|country|one)\b"),
    ("still", r"\bstill (?:alive|living|in existence|exists?|in office|married|together|playing|"
              r"active|open|running|operating|on the air|airing|in production|being made|in print|"
              r"used|in use|on|going)\b"),
    ("present_role", r"\bthe present\b|^(?:who|what|which) (?:is|are) "
                     r"(?!(?:the )?(?:first|second|third|only|original|inaugural|founding)\b)(?:the )?(?:\w+ ){0,3}?"
                     r"(?:president|prime minister|ceo|coo|cfo|chairman|chairperson|chairwoman|"
                     r"coach|manager|chief|speaker|governor|mayor|king|queen|monarch|pope|"
                     r"secretary|leader|captain|owner|minister|chancellor|director|editor|host|"
                     r"presenter|head|commissioner|ambassador|senator|representative|premier|"
                     r"champion|champions|holder|record holder|richest|oldest living)\b"),
    ("records_counts", r"\b(?:has|have) the most\b|\bholds? the record\b|\bhow many times\b.*\b(?:has|have)\b|"
                       r"\bhow many\b.*\b(?:has|have)\s+(?:\w+\s+){0,3}(?:won|been|had|played|scored|appeared|"
                       r"made|sold|recorded|released|produced|lost|beaten|beat|reached|hosted|held|"
                       r"died|visited|joined|taken|gone|competed|entered)\b|"
                       r"\bhow old (?:is|are)\b|\bhow long (?:has|have)\b"),
    ("won_undated", r"^(?:who|which (?:team|country|player|club|nation|state|school|horse|driver|band|"
                    r"movie|film|song|album|actor|actress))\s+(?:won|wins|has won)\b"
                    r"(?!.*(?:\d|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
                    r"tenth|inaugural|original|battle|war)\b))"),
    ("personal_status", r"\b(?:is|are) .*\b(?:married to|dating|engaged to)\b|"
                        r"\bwho (?:is|are) .*'s? (?:wife|husband|girlfriend|boyfriend|partner|fiance|"
                        r"fiancee|spouse)\b"),
    ("future", r"\bwhen will\b|\bwill be\b|\bgoing to be\b|\bwho will\b|\bwhere will\b"),
    ("population", r"\bpopulation of\b|\bwhat is the population\b|\bhow many people live\b"),
    ("new_thing", r"\bnew (?:season|episode|series|album|single|song|movie|film|version|update|"
                  r"edition|coach|manager|president|leader|ceo|captain|head|show)\b"),
))


def time_sensitive(question: str) -> str | None:
    """Name of the first ``TIME_RULES`` rule the (normalised, lower-case) question matches."""
    q = normalize_space(question).lower()
    for name, rx in TIME_RULES:
        if rx.search(q):
            return name
    return None


# ------------------------------------------------------------------------------ cleaning


@dataclass(frozen=True)
class QACleanCfg:
    max_answer_words: int = 12
    max_question_words: int = 60
    max_answer_chars: int = 120
    drop_time_sensitive: bool = True


@dataclass(frozen=True)
class QAPair:
    question: str     # display form (``clean_question``)
    answer: str       # canonical answer (``clean_answer``)
    source: str       # source name (``bes.library.sources.QA_SOURCES``)
    row: int          # row index in the source file
    aliases: tuple[str, ...] = ()   # unused by the build (answers of dev rows live in ``raw``)


def clean_pairs(rows: Iterable[tuple[str, int, str, Sequence[str]]], exclude_keys: set[str] | frozenset[str],
                cfg: QACleanCfg = QACleanCfg()) -> tuple[list[QAPair], dict[str, dict[str, int]]]:
    """Clean ``(source, row, question, answers)`` rows (in order) into stored pairs.

    Returns the kept pairs (first occurrence order) and ``stats[source][filter]`` counts:
    ``in``, ``empty_question``, ``no_terms``, ``empty_answer``, ``long_answer``,
    ``long_question``, ``time:<rule>``, ``eval_overlap``, ``duplicate_exact``,
    ``duplicate_norm``, ``ambiguous`` (normalised duplicates with different answers, all
    dropped), ``kept``. ``exclude_keys`` are ``question_key`` values of evaluation questions.
    """
    stats: dict[str, dict[str, int]] = {}

    def bump(src: str, k: str, n: int = 1) -> None:
        d = stats.setdefault(src, {})
        d[k] = d.get(k, 0) + n

    seen_raw: set[str] = set()
    by_key: dict[str, int] = {}        # question key -> index in ``kept``
    kept: list[QAPair | None] = []
    answers_of: list[str] = []         # answer key of each kept slot
    for src, row, q_raw, answers in rows:
        bump(src, "in")
        raw = normalize_space(q_raw)
        if raw in seen_raw:
            bump(src, "duplicate_exact")
            continue
        seen_raw.add(raw)
        q = clean_question(q_raw)
        if not q:
            bump(src, "empty_question")
            continue
        if len(q.split()) > cfg.max_question_words:
            bump(src, "long_question")
            continue
        if not terms(q):
            bump(src, "no_terms")
            continue
        a = clean_answer(answers[0]) if answers else ""
        if not a or not answer_key(a):
            bump(src, "empty_answer")
            continue
        if len(a.split()) > cfg.max_answer_words or len(a) > cfg.max_answer_chars or "\n" in a:
            bump(src, "long_answer")
            continue
        if cfg.drop_time_sensitive:
            rule = time_sensitive(q)
            if rule is not None:
                bump(src, f"time:{rule}")
                continue
        key = question_key(q)
        if key in exclude_keys:
            bump(src, "eval_overlap")
            continue
        j = by_key.get(key)
        if j is not None:
            if kept[j] is not None and answers_of[j] != answer_key(a):
                bump(kept[j].source, "ambiguous")      # the earlier pair goes as well
                bump(kept[j].source, "kept", -1)
                kept[j] = None
                bump(src, "ambiguous")
            else:
                bump(src, "duplicate_norm")
            continue
        by_key[key] = len(kept)
        kept.append(QAPair(q, a, src, int(row)))
        answers_of.append(answer_key(a))
        bump(src, "kept")
    return [p for p in kept if p is not None], stats


def eval_keys(questions: Iterable[str]) -> frozenset[str]:
    """``question_key`` of every evaluation question (the set the store must avoid)."""
    return frozenset(k for k in (question_key(clean_question(q)) for q in questions) if k)


def assert_no_eval_overlap(pairs: Sequence[QAPair], keys: frozenset[str] | set[str]) -> None:
    """Raise ``ValueError`` if any stored question normalises to an evaluation question."""
    bad = [p.question for p in pairs if question_key(p.question) in keys]
    if bad:
        raise ValueError(f"{len(bad)} stored questions equal evaluation questions, e.g. {bad[:3]}")


# ------------------------------------------------------------------------------ ordering


def entry_order(pairs: Sequence[QAPair], how: str = "answer") -> np.ndarray:
    """Storage order of the pairs: ``"answer"`` (answer key, then question key: paraphrases
    with the same answer share a block, which also helps the compressor), ``"question"``
    (question key) or ``"source"`` (input order)."""
    n = len(pairs)
    if how == "source":
        return np.arange(n, dtype=np.int64)
    if how == "question":
        keys = [(question_key(p.question), i) for i, p in enumerate(pairs)]
    elif how == "answer":
        keys = [(answer_key(p.answer), question_key(p.question), i) for i, p in enumerate(pairs)]
    else:
        raise ValueError(how)
    return np.asarray([k[-1] for k in sorted(keys)], dtype=np.int64)


def entry_bytes(question: str, answer: str) -> bytes:
    return (question + "\n" + answer).encode("utf-8")


def parse_entry(raw: bytes) -> tuple[str, str]:
    q, _, a = raw.decode("utf-8").partition("\n")
    return q, a


def render_qa(question: str, answer: str) -> str:
    """Text shown to the model for a qa entry."""
    return f"Q: {question}\nA: {answer}"


# ------------------------------------------------------------------------------ store


@dataclass(frozen=True)
class QACfg:
    n_bits: int = _L.qa_bits          # term buckets of the qa index
    group_bits: int = 6               # one coarse offset per 64 buckets
    read_postings: int = _L.qa_read_postings  # postings read per query at most (lowest df first)
    max_query_terms: int = 16         # distinct query buckets considered at most
    min_cov_q: float = 0.34           # candidates need M >= min_cov_q * Wq (pre-filter)
    order: str = "answer"             # entry_order
    block_raw_bytes: int = 65536


@dataclass
class QARead:
    """Distinct stored numbers read by one qa query (index part)."""

    counts: int = 0
    coarse: int = 0
    postings: int = 0
    per_entry: int = 0
    _seen: set = field(default_factory=set, repr=False)

    def touch(self, name: str, idx: Iterable[int], per: float) -> int:
        n = 0.0
        for i in idx:
            k = (name, int(i))
            if k not in self._seen:
                self._seen.add(k)
                n += per
        return int(math.ceil(n))

    @property
    def total(self) -> int:
        return self.counts + self.coarse + self.postings + self.per_entry


@dataclass
class QAMatch:
    """Ranked qa candidates of one query (arrays aligned, best first)."""

    entries: np.ndarray     # int64 entry ids
    sym: np.ndarray         # float64 min(cov_q, cov_e)
    cov_q: np.ndarray
    cov_e: np.ndarray
    type_match: np.ndarray  # bool
    wq: float               # query idf mass
    n_terms: int            # distinct query buckets
    qtype: int

    @property
    def size(self) -> int:
        return int(self.entries.shape[0])


class QAStore:
    """Stored qa entries + their index (see the module docstring)."""

    PER_ENTRY = ("mass", "qtype")
    INDEX = ("post", "counts", "coarse")
    HEADER_NUMBERS = 2 + 1 + 1   # n_entries (32-bit), n_bits, group_bits

    def __init__(self, cfg: QACfg, blocks: EntryBlocks, mass: np.ndarray, qtypes: np.ndarray,
                 post: np.ndarray, counts: np.ndarray, coarse: np.ndarray, meta: dict | None = None):
        self.cfg = cfg
        self.blocks = blocks
        self.mass = mass
        self.qtype = qtypes
        self.post = post
        self.counts = counts
        self.coarse = coarse
        self.meta = meta or {}
        self._idf_cache: dict[int, float] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, pairs: Sequence[QAPair], cfg: QACfg = QACfg(), pack_cfg: PackCfg | None = None,
              meta: dict | None = None) -> "QAStore":
        """Order, pack and index ``pairs`` (entry ``e`` = ``pairs[order[e]]``)."""
        order = entry_order(pairs, cfg.order)
        ps = [pairs[int(i)] for i in order]
        pc = pack_cfg or PackCfg(block_raw_bytes=cfg.block_raw_bytes)
        blocks = pack_entries([entry_bytes(p.question, p.answer) for p in ps], pc)
        n = len(ps)
        nb = 1 << cfg.n_bits
        ent_buckets = [sorted({bucket(t, cfg.n_bits) for t in terms(p.question)}) for p in ps]
        lens = np.fromiter((len(b) for b in ent_buckets), dtype=np.int64, count=n)
        flat_b = np.fromiter((b for bs in ent_buckets for b in bs), dtype=np.int64, count=int(lens.sum()))
        flat_e = np.repeat(np.arange(n, dtype=np.int64), lens)
        df = np.bincount(flat_b, minlength=nb).astype(np.int64)
        if df.max(initial=0) > 0xFFFF:
            raise ValueError("a qa bucket holds more than 65535 entries; use more bits")
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        mass_f = np.bincount(flat_e, weights=idf[flat_b], minlength=n)
        mass = np.floor(mass_f * MASS_UNIT + 1e-9)    # floor: an identical question scores 1
        if mass.size and mass.max() > 0xFFFF:
            raise ValueError("question mass overflows uint16")
        o = np.lexsort((flat_e, flat_b))
        post = flat_e[o].astype(np.uint32)
        counts = df.astype(np.uint16)
        g = cfg.group_bits
        csum = np.concatenate([[0], np.cumsum(df)])
        coarse = csum[np.arange(0, nb + 1, 1 << g)].astype(np.uint32)
        qt = np.fromiter((qtype(p.question) for p in ps), dtype=np.uint8, count=n)
        m = dict(meta or {})
        m["sources"] = sorted({p.source for p in ps})
        return cls(cfg, blocks, mass.astype(np.uint16), qt, post, counts, coarse, m)

    # ------------------------------------------------------------------ sizes / io
    @property
    def n_entries(self) -> int:
        return int(self.mass.shape[0])

    def accounting(self) -> dict[str, int]:
        a = self.blocks.accounting("qa_array_")
        text = sum(a.values())
        per = {k: numbers_of(getattr(self, "qtype" if k == "qtype" else k).nbytes) for k in self.PER_ENTRY}
        idx = {k: numbers_of(getattr(self, k).nbytes) for k in self.INDEX}
        out = {"qa_text_numbers": text, "qa_entry_numbers": sum(per.values()),
               "qa_index_numbers": sum(idx.values()) + self.HEADER_NUMBERS,
               "qa_entries": self.n_entries, "qa_postings": int(self.post.shape[0]),
               "qa_blocks": self.blocks.n_blocks,
               "qa_raw_bytes": int(np.asarray(self.blocks.block_raw, dtype=np.int64).sum()), **a,
               **{f"qa_array_{k}": v for k, v in per.items()}, **{f"qa_array_{k}": v for k, v in idx.items()},
               "qa_header_numbers": self.HEADER_NUMBERS}
        out["qa_numbers"] = out["qa_text_numbers"] + out["qa_entry_numbers"] + out["qa_index_numbers"]
        return out

    def save(self, out_dir: str) -> None:
        self.blocks.save(out_dir, "qa_")
        for k in ("mass", "qtype", "post", "counts", "coarse"):
            _atomic_save_npy(os.path.join(out_dir, f"qa_{k}.npy"), np.asarray(getattr(self, k)))
        meta = {"qa_cfg": dataclasses.asdict(self.cfg), "pack_cfg": dataclasses.asdict(self.blocks.cfg),
                "accounting": self.accounting(), **self.meta}
        _atomic_write(os.path.join(out_dir, "qa.json"), json.dumps(meta, indent=1, sort_keys=True).encode())

    @classmethod
    def load(cls, out_dir: str, mmap: bool = True) -> "QAStore":
        meta = json.loads(open(os.path.join(out_dir, "qa.json")).read())
        cfg = QACfg(**meta.pop("qa_cfg"))
        pcfg = PackCfg(**meta.pop("pack_cfg"))
        meta.pop("accounting", None)
        mode = "r" if mmap else None
        arr = {k: np.load(os.path.join(out_dir, f"qa_{k}.npy"), mmap_mode=mode)
               for k in ("mass", "qtype", "post", "counts", "coarse")}
        blocks = EntryBlocks.load(out_dir, "qa_", pcfg, mmap)
        return cls(cfg, blocks, arr["mass"], arr["qtype"], arr["post"], arr["counts"], arr["coarse"], meta)

    # ------------------------------------------------------------------ access
    def entry(self, e: int) -> tuple[str, str]:
        return parse_entry(self.blocks.entry(e))

    def idf(self, df: int) -> float:
        n = self.n_entries
        return math.log1p((n - df + 0.5) / (df + 0.5))

    def bucket_range(self, b: int, log: QARead | None = None) -> tuple[int, int]:
        g = self.cfg.group_bits
        g0 = (b >> g) << g
        start = int(self.coarse[b >> g]) + int(np.asarray(self.counts[g0:b], dtype=np.int64).sum())
        if log is not None:
            log.coarse += log.touch("qa_coarse", [b >> g], 2)
            log.counts += log.touch("qa_counts", range(g0, b + 1), 1)
        return start, start + int(self.counts[b])

    def query_buckets(self, query: str) -> list[int]:
        """Distinct term buckets of the query, in query order (all count in ``Wq``)."""
        return list(dict.fromkeys(bucket(t, self.cfg.n_bits) for t in terms(query)))

    def search(self, query: str, exclude: Iterable[int] = (), log: QARead | None = None) -> QAMatch:
        """Ranked candidates with ``cov_q >= cfg.min_cov_q`` (module docstring). Reads one
        ``qa_counts`` entry per distinct query bucket, the lists of at most
        ``max_query_terms`` buckets (lowest df first, at most ``read_postings`` postings in
        all) and the per-entry arrays of the candidates that pass the pre-filter."""
        qb = self.query_buckets(query)
        qt = qtype(query)
        empty = QAMatch(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0),
                        np.zeros(0, bool), 0.0, len(qb), qt)
        if not qb or self.n_entries == 0:
            return empty
        if log is not None:
            log.counts += log.touch("qa_counts", qb, 1)
        dfs = [int(self.counts[b]) for b in qb]
        w = [self.idf(d) for d in dfs]
        wq = float(sum(w))
        parts_e, parts_w = [], []
        read = 0
        lists = 0
        for d, wb, b in sorted(zip(dfs, w, qb)):
            if d == 0:
                continue
            if read + d > self.cfg.read_postings or lists >= self.cfg.max_query_terms:
                continue
            lists += 1
            s, e = self.bucket_range(b, log)
            parts_e.append(np.asarray(self.post[s:e], dtype=np.int64))
            parts_w.append(np.full(e - s, wb))
            read += d
            if log is not None:
                log.postings += log.touch("qa_post_bucket", [b], 2 * (e - s))
        if not parts_e:
            return dataclasses.replace(empty, wq=wq)
        ents = np.concatenate(parts_e)
        ws = np.concatenate(parts_w)
        u, inv = np.unique(ents, return_inverse=True)
        m = np.bincount(inv, weights=ws)
        ex = np.fromiter((int(x) for x in exclude), dtype=np.int64)
        keep = m >= self.cfg.min_cov_q * wq - 1e-12
        if ex.size:
            keep &= ~np.isin(u, ex)
        u, m = u[keep], m[keep]
        if u.size == 0:
            return dataclasses.replace(empty, wq=wq)
        we = np.asarray(self.mass[u], dtype=np.float64) / MASS_UNIT
        tm = np.asarray(self.qtype[u], dtype=np.int64) == qt
        if log is not None:
            log.per_entry += log.touch("qa_mass", u, 1)
            log.per_entry += log.touch("qa_qtype", u, 0.5)
        cov_q = m / wq
        cov_e = np.where(we > 0, np.minimum(1.0, m / np.maximum(we, 1e-12)), 0.0)
        sym = np.minimum(cov_q, cov_e)
        o = np.lexsort((u, ~tm, -np.round(sym, 9)))
        return QAMatch(u[o], sym[o], cov_q[o], cov_e[o], tm[o], wq, len(qb), qt)


def brute_force_sym(query: str, questions: Sequence[str], n_bits: int) -> np.ndarray:
    """Reference ``sym`` of every stored question (no read budget, no quantisation): used by
    the tests and by the E1 measurement."""
    n = len(questions)
    sets = [{bucket(t, n_bits) for t in terms(q)} for q in questions]
    df: dict[int, int] = {}
    for s in sets:
        for b in s:
            df[b] = df.get(b, 0) + 1
    idf = {b: math.log1p((n - d + 0.5) / (d + 0.5)) for b, d in df.items()}
    q = {bucket(t, n_bits) for t in terms(query)}
    wq = sum(idf.get(b, math.log1p((n + 0.5) / 0.5)) for b in q)
    out = np.zeros(n)
    for i, s in enumerate(sets):
        m = sum(idf[b] for b in q & s)
        we = sum(idf[b] for b in s)
        if wq > 0 and we > 0:
            out[i] = min(m / wq, m / we)
    return out
