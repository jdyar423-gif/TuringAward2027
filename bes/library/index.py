"""BM25 indexes over library passages.

``TruncatedIndex`` is the index that ships with the model and is counted in the budget:

    postings  uint32 [P]                   kept postings grouped by term bucket; inside a
                                           bucket by impact (desc) then passage id (asc);
                                           entry = (passage_id << 8) | impact
    counts    uint16 [2**n_bits]           postings kept per bucket (<= max_list)
    coarse    uint32 [2**(n_bits-g) + 1]   offset of the first posting of every group of
                                           2**g buckets (the offset of bucket b is
                                           coarse[b >> g] + sum(counts[(b >> g) << g : b]))
    df        uint16 [2**n_bits]           passages containing the bucket (saturating)
    header    N, avgdl, impact scale, k1, b, title weights, n_bits, g, max_query_terms

Every word term (``bes.library.terms.terms``) is hashed into one of ``2**n_bits`` buckets; a
bucket acts as one BM25 term (colliding terms share statistics). The impact of bucket ``k``
in passage ``d`` is the standard BM25 term weight

    idf(k) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)),
    idf(k) = ln(1 + (N - df + 0.5) / (df + 0.5)),

with ``tf`` the title-weighted count of the bucket's terms in the passage (title terms count
``1 + title_weight`` times, ``1 + lead_title_weight`` times in the first passage of an
article) and ``dl`` the weighted passage length; it is quantised linearly to 1..255 with one
global scale.

Truncation (static pruning). The full postings do not fit the budget, so the index keeps
the postings with the largest utility

    u(k, d) = (impact(k, d) / max_k' impact(k', d)) ** rel_exponent * prior(d) ** prior_exponent
              * (title_bonus if k holds a title term of d else 1)

subject to at most ``max_list`` postings per bucket and ``budget_numbers`` stored numbers in
total. ``prior(d)`` is the selector's per-word benefit of the passage (article importance x
position weight x list penalty), 1 when not given. This document-centric, prior-weighted
rule was chosen on dev questions over keeping the top-K postings of every bucket, which wastes
most fixed slots (most buckets hold one to three rare-term postings) and cuts every
moderately common term to K passages (``docs/E1_COVERAGE.md``).

Scoring a query reads ``df`` for each distinct query bucket and the lists of the (at most
``max_query_terms``) lowest-df buckets, and sums the stored impacts per passage; a pruned
posting scores 0. Nothing else is derived at load time.

``ExactBM25`` is an evaluation reference (true terms, no hashing, no truncation, float
weights); it is never shipped.
"""
from __future__ import annotations

import dataclasses
import json
import os
from array import array
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from bes.config import LibraryCfg
from bes.library.terms import bucket, terms

_L = LibraryCfg()
IMPACT_BITS = 8
PID_BITS = 24
MAX_PASSAGES = 1 << PID_BITS


@dataclass(frozen=True)
class IndexCfg:
    n_bits: int = _L.index_bits                 # 2**18 term buckets
    budget_numbers: int = _L.index_numbers      # stored numbers of the whole index
    max_list: int = _L.index_max_list           # postings per bucket at most (read-set bound)
    group_bits: int = 6                         # one coarse offset per 64 buckets
    k1: float = _L.bm25_k1                      # k1, b and the title weights: best of a
    b: float = _L.bm25_b                        # 54-point grid on dev questions; the margin
    title_weight: float = _L.title_weight       # over (0.9, 0.4, 1, 1) is within noise
    lead_title_weight: float = _L.lead_title_weight  # title weight in an article's first passage
    rel_exponent: float = 0.5                   # posting utility (module docstring), chosen on dev
    prior_exponent: float = 1.5
    title_bonus: float = 2.0
    max_query_terms: int = _L.max_query_terms   # at most this many buckets per query

    def overhead_numbers(self) -> int:
        n = 1 << self.n_bits
        n_groups = (n >> self.group_bits) + 1
        return n + 2 * n_groups + n + TruncatedIndex.HEADER_NUMBERS

    def max_postings(self) -> int:
        return max(0, (self.budget_numbers - self.overhead_numbers()) // 2)


# --------------------------------------------------------------------------- postings


@dataclass
class Postings:
    """Raw (untruncated) postings: one row per (key, passage) pair."""

    key: np.ndarray    # int64 [M] bucket id or term id
    pid: np.ndarray    # int32 [M]
    tf: np.ndarray     # float32 [M] title-weighted term frequency
    dl: np.ndarray     # float32 [N] title-weighted passage length (terms)
    title: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # bool [M]

    @property
    def n_docs(self) -> int:
        return int(self.dl.shape[0])


def doc_terms(title: str, body: str, title_weight: float) -> tuple[dict[str, float], set[str]]:
    """(weighted term counts, title terms) of one passage; title terms count
    ``1 + title_weight`` times."""
    c: dict[str, float] = {}
    tw = 1.0 + title_weight
    tt = terms(title)
    for t in tt:
        c[t] = c.get(t, 0.0) + tw
    for t in terms(body):
        c[t] = c.get(t, 0.0) + 1.0
    return c, set(tt)


def lead_flags(docs: Sequence[tuple[str, str]]) -> list[bool]:
    """True for the first passage of every run of equal titles (the article's lead)."""
    out = []
    last = None
    for title, _ in docs:
        out.append(title != last)
        last = title
    return out


def collect_postings(docs: Iterable[tuple[str, str]], key_of: Callable[[str], int],
                     title_weight: float, lead_title_weight: float | None = None,
                     leads: Sequence[bool] | None = None) -> Postings:
    """Collect postings for ``docs`` = iterable of (title, body) in passage-id order.

    ``key_of(term)`` maps a term to its key (bucket or term id); terms mapping to the same
    key in one passage are summed. Title terms count ``1 + title_weight`` times, or
    ``1 + lead_title_weight`` times in passages flagged in ``leads``.
    """
    keys = array("q")
    pids = array("i")
    tfs = array("f")
    dls = array("f")
    tflag = array("b")
    cache: dict[str, int] = {}
    for pid, (title, body) in enumerate(docs):
        c: dict[int, float] = {}
        ct: set[int] = set()
        total = 0.0
        tw = title_weight
        if leads is not None and lead_title_weight is not None and leads[pid]:
            tw = lead_title_weight
        counts, tset = doc_terms(title, body, tw)
        for t, w in counts.items():
            k = cache.get(t)
            if k is None:
                k = key_of(t)
                cache[t] = k
            c[k] = c.get(k, 0.0) + w
            total += w
            if t in tset:
                ct.add(k)
        dls.append(total)
        keys.extend(c.keys())
        pids.extend([pid] * len(c))
        tfs.extend(c.values())
        tflag.extend(1 if k in ct else 0 for k in c)

    def arr(a, dt):  # zero-copy views of the array buffers (kept alive by the views)
        return np.frombuffer(a, dtype=dt) if len(a) else np.zeros(0, dt)

    return Postings(key=arr(keys, np.int64), pid=arr(pids, np.int32), tf=arr(tfs, np.float32),
                    dl=arr(dls, np.float32), title=arr(tflag, np.int8).astype(bool))


def bm25_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    df = df.astype(np.float64)
    return np.log1p((n_docs - df + 0.5) / (df + 0.5))


def bm25_impacts(p: Postings, n_keys: int, k1: float, b: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(impact float64 [M], df int64 [n_keys], avgdl) for raw postings."""
    n = p.n_docs
    df = np.bincount(p.key, minlength=n_keys).astype(np.int64)
    avgdl = float(p.dl.astype(np.float64).mean()) if n else 1.0
    avgdl = avgdl if avgdl > 0 else 1.0
    idf = bm25_idf(df, n)
    tf = p.tf.astype(np.float64)
    norm = p.dl.astype(np.float64)[p.pid]          # dl of each posting's passage
    norm *= b / avgdl
    norm += 1.0 - b
    norm *= k1
    norm += tf                                      # tf + k1 * (1 - b + b * dl / avgdl)
    imp = idf[p.key]
    imp *= tf
    imp *= k1 + 1.0
    imp /= norm
    return imp, df, avgdl


def rekey_postings(p: Postings, key_map: np.ndarray) -> Postings:
    """Postings with every key replaced by ``key_map[key]``; the tf of keys that collide
    inside one passage is summed (e.g. true terms -> hash buckets)."""
    nk = np.asarray(key_map, dtype=np.int64)[p.key]
    if nk.size and (nk.min() < 0 or nk.max() >= (1 << 32)):
        raise ValueError("keys must fit in 32 bits")
    combo = (p.pid.astype(np.int64) << 32) | nk
    u, inv = np.unique(combo, return_inverse=True)
    tf = np.bincount(inv, weights=p.tf.astype(np.float64)).astype(np.float32)
    tflag = np.zeros(u.shape[0], dtype=bool)
    if p.title.size:
        tflag = np.bincount(inv, weights=p.title.astype(np.float64)) > 0
    return Postings(key=u & 0xFFFFFFFF, pid=(u >> 32).astype(np.int32), tf=tf, dl=p.dl, title=tflag)


def term_postings(docs: Sequence[tuple[str, str]], title_weight: float = 1.0,
                  lead_title_weight: float | None = None) -> tuple[Postings, list[str]]:
    """Postings over true terms and the term list (key = index into the list)."""
    vocab: dict[str, int] = {}

    def key_of(t: str) -> int:
        k = vocab.get(t)
        if k is None:
            k = len(vocab)
            vocab[t] = k
        return k

    p = collect_postings(docs, key_of, title_weight, lead_title_weight,
                         lead_flags(docs) if lead_title_weight is not None else None)
    return p, list(vocab)


def bucket_map(term_list: Sequence[str], n_bits: int) -> np.ndarray:
    """int64 [V]: bucket of every term of ``term_list``."""
    return np.fromiter((bucket(t, n_bits) for t in term_list), dtype=np.int64, count=len(term_list))


def posting_utility(imp: np.ndarray, pid: np.ndarray, title: np.ndarray, n_docs: int,
                    prior: np.ndarray | None, cfg: IndexCfg) -> np.ndarray:
    """Utility of every posting (module docstring)."""
    maxdoc = np.zeros(n_docs, dtype=np.float64)
    np.maximum.at(maxdoc, pid, imp)
    u = imp / np.maximum(maxdoc[pid], 1e-300)       # relative impact in (0, 1]
    u **= cfg.rel_exponent
    if prior is not None:
        pr = np.asarray(prior, dtype=np.float64)
        if pr.shape[0] != n_docs or np.any(pr <= 0) or not np.all(np.isfinite(pr)):
            raise ValueError("prior must be positive and finite, one value per passage")
        u = u * pr[pid] ** cfg.prior_exponent
    if title.size:
        u = u * np.where(title, cfg.title_bonus, 1.0)
    return u


def select_postings(key: np.ndarray, pid: np.ndarray, util: np.ndarray, n_buckets: int,
                    max_list: int, max_total: int, negated: bool = False) -> np.ndarray:
    """bool [M]: postings kept - at most ``max_list`` per bucket and ``max_total`` in all,
    largest utility first (ties: lower passage id, then lower bucket). ``negated``: ``util``
    already holds the negated utilities (saves a copy of an [M] array)."""
    m = key.shape[0]
    neg = util if negated else -util
    order = np.lexsort((pid, neg, key))  # bucket asc, utility desc, pid asc
    ks = key[order]
    starts = np.searchsorted(ks, np.arange(n_buckets), side="left")
    rank = starts[ks]
    del ks
    np.subtract(np.arange(m), rank, out=rank)
    cand = order[rank < max_list]
    del order, rank
    if cand.shape[0] > max_total:
        o2 = np.lexsort((key[cand], pid[cand], neg[cand]))
        cand = cand[o2[:max_total]]
    keep = np.zeros(m, dtype=bool)
    keep[cand] = True
    return keep


# ---------------------------------------------------------------------------- read log


@dataclass
class IndexRead:
    """Distinct stored numbers read by one query from the index."""

    df_numbers: int = 0
    offset_numbers: int = 0
    posting_numbers: int = 0
    buckets: int = 0
    _seen: set = field(default_factory=set, repr=False)

    def touch(self, array_name: str, idx: Iterable[int], numbers_per_entry: int) -> int:
        n = 0
        for i in idx:
            k = (array_name, int(i))
            if k not in self._seen:
                self._seen.add(k)
                n += numbers_per_entry
        return n

    @property
    def total(self) -> int:
        return self.df_numbers + self.offset_numbers + self.posting_numbers


# ---------------------------------------------------------------------- stored index


class TruncatedIndex:
    """The stored, counted index (see the module docstring)."""

    ARRAYS = ("postings", "counts", "coarse", "df")
    #: n_docs, avgdl, scale, k1, b, title_weight, lead_title_weight (32-bit each)
    #: + n_bits, group_bits, max_query_terms (16-bit each)
    HEADER_NUMBERS = 2 * 7 + 3

    def __init__(self, cfg: IndexCfg, postings: np.ndarray, counts: np.ndarray, coarse: np.ndarray,
                 df: np.ndarray, n_docs: int, avgdl: float, scale: float,
                 full: dict[str, np.ndarray] | None = None):
        self.cfg = cfg
        self.postings = postings
        self.counts = counts
        self.coarse = coarse
        self.df = df
        self.n_docs = int(n_docs)
        self.avgdl = float(avgdl)
        self.scale = float(scale)
        self.full = full  # optional untruncated lists (evaluation only, never stored)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, docs: Sequence[tuple[str, str]], cfg: IndexCfg = IndexCfg(),
              prior: Sequence[float] | None = None, keep_full: bool = False) -> "TruncatedIndex":
        """Build from (title, body) pairs in passage-id order (passages of one title are
        contiguous; the first of each run is the lead). ``prior``: one positive value per
        passage, or None for a uniform prior."""
        p = collect_postings(docs, lambda t: bucket(t, cfg.n_bits), cfg.title_weight,
                             cfg.lead_title_weight, lead_flags(docs))
        return cls.from_postings(p, cfg, prior, keep_full)

    @classmethod
    def from_postings(cls, p: Postings, cfg: IndexCfg, prior: Sequence[float] | None = None,
                      keep_full: bool = False) -> "TruncatedIndex":
        n_buckets = 1 << cfg.n_bits
        n = p.n_docs
        if n > MAX_PASSAGES:
            raise ValueError("too many passages for 24-bit ids")
        if cfg.max_list > 0xFFFF:
            raise ValueError("max_list must fit in uint16")
        imp, df, avgdl = bm25_impacts(p, n_buckets, cfg.k1, cfg.b)
        max_imp = float(imp.max()) if imp.size else 1.0
        scale = max_imp / ((1 << IMPACT_BITS) - 1) if max_imp > 0 else 1.0
        q = imp / scale                    # in place below: one float64 temporary at a time
        np.rint(q, out=q)
        np.clip(q, 1, (1 << IMPACT_BITS) - 1, out=q)
        q = q.astype(np.int16)
        title = p.title if p.title.size == imp.size else np.zeros(imp.size, dtype=bool)
        util = posting_utility(imp, p.pid, title, n, None if prior is None else np.asarray(prior), cfg)
        np.negative(util, out=util)        # select_postings sorts by -utility
        keep = select_postings(p.key, p.pid, util, n_buckets, cfg.max_list, cfg.max_postings(),
                               negated=True)
        del util
        k_s, q_s, pid_s, imp_s = p.key[keep], q[keep].astype(np.int64), p.pid[keep].astype(np.int64), imp[keep]
        if not keep_full:
            del imp, q
        order = np.lexsort((pid_s, -imp_s, -q_s, k_s))
        k_s, q_s, pid_s = k_s[order], q_s[order], pid_s[order]
        post = ((pid_s << IMPACT_BITS) | q_s).astype(np.uint32)
        counts = np.bincount(k_s, minlength=n_buckets).astype(np.uint16)
        g = cfg.group_bits
        csum = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])
        coarse = csum[np.arange(0, n_buckets + 1, 1 << g)].astype(np.uint32)
        if coarse.shape[0] != (n_buckets >> g) + 1:
            raise ValueError("n_bits must be >= group_bits")
        df16 = np.minimum(df, 0xFFFF).astype(np.uint16)
        full = None
        if keep_full:
            o = np.lexsort((p.pid, -imp, -q, p.key))
            ks = p.key[o]
            full = {"starts": np.searchsorted(ks, np.arange(n_buckets), side="left"),
                    "ends": np.searchsorted(ks, np.arange(n_buckets), side="right"),
                    "pid": p.pid[o].astype(np.int32), "q": q[o].astype(np.int16)}
        return cls(cfg, post, counts, coarse, df16, n, avgdl, scale, full)

    # ------------------------------------------------------------------ query
    def query_buckets(self, query: str, log: IndexRead | None = None) -> list[int]:
        """Distinct buckets of the query terms that will be scored (lowest df first, at most
        ``max_query_terms``, empty buckets dropped). Reads one df entry per distinct bucket."""
        seen: dict[int, None] = {}
        for t in terms(query):
            seen.setdefault(bucket(t, self.cfg.n_bits), None)
        bs = list(seen)
        if log is not None:
            log.df_numbers += log.touch("df", bs, 1)  # uint16 = one stored number
        dfs = sorted((int(self.df[b]), b) for b in bs if int(self.df[b]) > 0)
        return [b for _, b in dfs[: self.cfg.max_query_terms]]

    def bucket_range(self, b: int, log: IndexRead | None = None) -> tuple[int, int]:
        """[start, end) of bucket ``b`` in ``postings``."""
        g = self.cfg.group_bits
        g0 = (b >> g) << g
        start = int(self.coarse[b >> g]) + int(np.asarray(self.counts[g0:b], dtype=np.int64).sum())
        end = start + int(self.counts[b])
        if log is not None:
            log.offset_numbers += log.touch("coarse", [b >> g], 2)
            log.offset_numbers += log.touch("counts", range(g0, b + 1), 1)
        return start, end

    def read_bucket(self, b: int, log: IndexRead | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(pids, quantised impacts) stored for bucket ``b``."""
        s, e = self.bucket_range(b, log)
        rows = np.asarray(self.postings[s:e]).astype(np.int64)
        if log is not None:
            log.posting_numbers += log.touch("postings_bucket", [b], 2 * (e - s))
            log.buckets += 1
        return rows >> IMPACT_BITS, rows & ((1 << IMPACT_BITS) - 1)

    def score_buckets(self, buckets: Sequence[int], log: IndexRead | None = None,
                      full: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(pids int64 [m], scores int64 [m]) of every passage with a non-zero score, sorted
        by score desc then pid asc. Scores are in impact units (multiply by ``scale``)."""
        return self._score(buckets, log, full)[:2]

    def score_with_max(self, query: str, log: IndexRead | None = None
                       ) -> tuple[np.ndarray, np.ndarray, int]:
        """``score`` plus the sum over the scored buckets of their largest stored impact (the
        first entry of each list), i.e. the best score any passage could reach."""
        return self._score(self.query_buckets(query, log), log, False)

    def _score(self, buckets: Sequence[int], log: IndexRead | None, full: bool
               ) -> tuple[np.ndarray, np.ndarray, int]:
        if not buckets:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), 0
        if full:
            if self.full is None:
                raise ValueError("index built without keep_full")
            f = self.full
            pids = np.concatenate([f["pid"][f["starts"][b]:f["ends"][b]] for b in buckets]).astype(np.int64)
            imps = np.concatenate([f["q"][f["starts"][b]:f["ends"][b]] for b in buckets]).astype(np.int64)
            smax = sum(int(f["q"][f["starts"][b]]) for b in buckets if f["ends"][b] > f["starts"][b])
        else:
            parts = [self.read_bucket(int(b), log) for b in buckets]
            pids = np.concatenate([x[0] for x in parts])
            imps = np.concatenate([x[1] for x in parts])
            smax = sum(int(x[1][0]) for x in parts if x[1].size)
        if pids.size == 0:
            return pids, imps, 0
        u, inv = np.unique(pids, return_inverse=True)
        sc = np.bincount(inv, weights=imps).astype(np.int64)
        order = np.lexsort((u, -sc))
        return u[order], sc[order], smax

    def score(self, query: str, log: IndexRead | None = None, full: bool = False
              ) -> tuple[np.ndarray, np.ndarray]:
        return self.score_buckets(self.query_buckets(query, log), log, full)

    # ------------------------------------------------------------------ accounting / io
    def accounting(self) -> dict[str, int]:
        a = {k: (int(getattr(self, k).nbytes) + 1) // 2 for k in self.ARRAYS}
        total = sum(a.values()) + self.HEADER_NUMBERS
        return {"index_postings_numbers": a["postings"], "index_counts_numbers": a["counts"],
                "index_coarse_numbers": a["coarse"], "index_df_numbers": a["df"],
                "index_header_numbers": self.HEADER_NUMBERS, "index_numbers": total,
                "index_postings": int(self.postings.shape[0])}

    def save(self, out_dir: str) -> None:
        from bes.library.pack import _atomic_save_npy, _atomic_write

        os.makedirs(out_dir, exist_ok=True)
        for k in self.ARRAYS:
            _atomic_save_npy(os.path.join(out_dir, f"index_{k}.npy"), np.asarray(getattr(self, k)))
        meta = {"index_cfg": dataclasses.asdict(self.cfg), "n_docs": self.n_docs,
                "avgdl": self.avgdl, "scale": self.scale, "accounting": self.accounting()}
        _atomic_write(os.path.join(out_dir, "index.json"), json.dumps(meta, indent=1).encode())

    @classmethod
    def load(cls, out_dir: str, mmap: bool = True) -> "TruncatedIndex":
        meta = json.loads(open(os.path.join(out_dir, "index.json")).read())
        mode = "r" if mmap else None
        arrs = {k: np.load(os.path.join(out_dir, f"index_{k}.npy"), mmap_mode=mode) for k in cls.ARRAYS}
        return cls(IndexCfg(**meta["index_cfg"]), arrs["postings"], arrs["counts"], arrs["coarse"],
                   arrs["df"], meta["n_docs"], meta["avgdl"], meta["scale"])


# ------------------------------------------------------------------- exact reference


class ExactBM25:
    """Exact BM25 over true terms (evaluation reference; never shipped)."""

    def __init__(self, p: Postings, term_list: Sequence[str], k1: float = 0.6, b: float = 0.3):
        self.vocab = {t: i for i, t in enumerate(term_list)}
        self.n_docs = p.n_docs
        imp, df, self.avgdl = bm25_impacts(p, len(term_list), k1, b)
        order = np.lexsort((p.pid, p.key))
        self.pid = p.pid[order].astype(np.int32)
        self.w = imp[order].astype(np.float32)
        key_s = p.key[order]
        self.starts = np.searchsorted(key_s, np.arange(len(term_list)), side="left")
        self.ends = np.searchsorted(key_s, np.arange(len(term_list)), side="right")
        self.df = df

    @classmethod
    def build(cls, docs: Sequence[tuple[str, str]], k1: float = 0.6, b: float = 0.3,
              title_weight: float = 1.0, lead_title_weight: float | None = None) -> "ExactBM25":
        p, tl = term_postings(docs, title_weight, lead_title_weight)
        return cls(p, tl, k1, b)

    def scores(self, query: str) -> np.ndarray:
        """float32 [N] BM25 scores of every passage."""
        s = np.zeros(self.n_docs, dtype=np.float32)
        for t in dict.fromkeys(terms(query)):
            k = self.vocab.get(t)
            if k is None:
                continue
            a, e = self.starts[k], self.ends[k]
            s[self.pid[a:e]] += self.w[a:e]
        return s

    def topk(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(pids, scores) of the k best passages (score desc, pid asc); zero scores dropped."""
        s = self.scores(query)
        nz = np.flatnonzero(s > 0)
        if nz.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        if nz.size > k:
            part = nz[np.argpartition(-s[nz], k - 1)[:k]]
            kth = s[part].min()
            part = nz[s[nz] >= kth]  # keep ties at the k-th score so the pid tie-break is exact
        else:
            part = nz
        order = np.lexsort((part, -s[part]))[:k]
        return part[order].astype(np.int64), s[part[order]]
