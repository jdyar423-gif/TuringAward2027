"""Retrieval-aware samples built with the written library (``docs/SPEC_MODEL_TRAIN.md`` section 6,
``docs/PLAN.md`` sections 4D and 5).

Builders
--------
Every builder takes the library, the tokenizer and a ``numpy.random.Generator`` (all random
choices come from it) and returns :class:`bes.data.samples.Sample` objects whose weights follow
the given :class:`~bes.data.samples.WeightRule`.

:func:`insert_retrieval`
    The retrieval of inference (``bes.decode.respond``) on one user turn of a conversation:
    the first user turn, or with probability ``MixCfg.retrieval_later_turn`` (conversations with
    several user turns) a later one drawn uniformly. When that message is question-like (the
    library's own ``query_type``, ``lib.cfg.gate.allowed_types``) and the gate opens
    (``lib.retrieve``), a ``LIBRARY`` turn goes right before that user turn, after the earlier
    exchanges, as the responder places it: the gated passage, or with probability
    ``DataCfg.distractor_frac`` a distractor instead (``MixCfg.distractor_random`` of them a
    uniformly drawn passage, the others one of the next ``MixCfg.distractor_topk`` ranked
    passages whose article is neither the top article nor the gated passage's). So passages
    appear exactly where inference would show one, and ``distractor_frac`` of them are
    distractors (on top of the gate's own errors). With ``max_len`` the sample is fitted as the
    responder fits a prompt: the turns after the served exchange go first, then the oldest
    exchanges (the system turn stays). One passage per conversation: documents, conversations
    that already have a library turn (the retrieval sources, including SQuAD v2's "not in the
    passage" questions), and conversations whose chosen user turn is not question-like or whose
    gate stays closed pass through unchanged.
:func:`reading_task`
    From a "wiki" passage, no language model involved. Lead passage: the user asks about the
    article title (neutral templates such as "Tell me about X.", "Who or what is X?", "X?");
    section passage whose first line is a known heading (``SECTION_HEADINGS``): the user asks
    about that section ("Tell me about the history of X."). The assistant answers with the
    first one to three sentences of the passage text. Variants (``MixCfg.read_*``): ``open``
    (the passage as ``LIBRARY`` turn), ``closed`` (no library turn), ``distractor`` (a passage
    of another article). A library turn only comes before a question the library's query
    typing accepts, as at inference (else the task is closed book). :func:`reading_suitable`
    marks, in one pass over the library, the passages a reading task can be built from (the
    checks above that need no random draw); a :class:`LibraryView` prepared with it
    (:meth:`LibraryView.prepare_tasks`) deals reading tasks over those entries only, so draws
    no longer read a passage only to decline it.
:func:`qa_task`
    From a "qa" entry (title = stored question, body = stored answer). The user asks the
    stored question; the assistant gives the stored answer through a few templates. Variants
    (``MixCfg.qa_*``): ``copy`` (the entry itself as ``LIBRARY`` turn), ``neighbour`` (the
    nearest *other* stored question, leave-one-out through ``exclude_pids`` and the entry's
    title group; it may or may not have the same answer), ``closed``.

No self-retrieval: a sample built from library entry ``pid`` records ``exclude_pids = (pid,)``
and ``exclude_titles = (title_id,)`` (no title for a qa entry, whose title id is -1; the
library stores no duplicate questions); distractor and neighbour passages, and any later
retrieval through :func:`insert_retrieval`, exclude them. No accidental constraint: every
generated user message is checked with ``bes.exact.constraints.compile`` (another template is
tried when it is not empty; the sample is declined when none is). Builders return ``None``
when they decline (unsuitable entry); :class:`RagStats` counts why.

Library contract (:class:`LibraryView`)
---------------------------------------
``Passage(pid, title_id, title, body, score, text, kind)`` with ``kind in {"wiki", "qa"}``
(for "qa": title = question, body = answer, text = ``"Q: <question>\\nA: <answer>"``);
``n_passages``, ``passage(pid)``, ``find_title``, ``title_range``, ``retrieve(query,
exclude_pids, exclude_titles, log, kinds)``, ``retrieve_topk(query, k, ..., kinds)`` and
``kind_range(kind)`` (contiguous pid range per kind). :func:`as_view` wraps a library that has
no ``kind_range`` yet (only wiki passages): passages get ``kind = "wiki"``,
``kind_range("wiki") = range(n_passages)``, ``kind_range("qa")`` is empty and a request for
other kinds finds nothing. When the library has ``search`` (ranked pids without reading
passages) ranked distractors read one passage instead of ``k``.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import LIBRARY, USER, DataCfg, MixCfg
from bes.data import chat_format as cf
from bes.data.samples import (ROLE_LIBRARY, Sample, WeightRule, conversation_cuts, conversation_sample,
                              exchange_starts, head_stop, permute_index, turn_spans, window)
from bes.exact import constraints as C

LIB_KINDS = ("wiki", "qa")
DEFAULT_ALLOWED_TYPES = ("question", "lookup")
#: Library tasks and the entry kind each is built from (sources of the mixture).
LIBRARY_TASKS: dict[str, str] = {"library_read": "wiki", "library_qa": "qa"}


# ------------------------------------------------------------------------- statistics
@dataclass
class RagStats:
    """Counters for the loader's log; merge the stats of several workers with :meth:`merge`.

    ``counts[key]``: events (``retrieval:calls``, ``retrieval:question_like``,
    ``retrieval:gate_open``, ``retrieval:fired``, ``retrieval:gated``,
    ``retrieval:distractor_ranked``, ``retrieval:later_turn``, ``retrieval:skip_has_library``,
    ``read:open``, ``read:declined_<reason>``, ``qa:neighbour_same_answer``, ...; the
    ``retrieval:gated`` / ``retrieval:distractor_*`` counts are of inserted passages).
    ``tokens[label]`` / ``samples[label]``: tokens and number of built samples per label
    (``read:open``, ``qa:copy``, ``retrieval:gated``, ...); ``passage_tokens[label]``: tokens of
    the inserted library turns.
    """

    counts: Counter = field(default_factory=Counter)
    tokens: Counter = field(default_factory=Counter)
    samples: Counter = field(default_factory=Counter)
    passage_tokens: Counter = field(default_factory=Counter)

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def built(self, label: str, sample: Sample, passage_tokens: int = 0) -> None:
        self.samples[label] += 1
        self.tokens[label] += len(sample)
        self.passage_tokens[label] += int(passage_tokens)

    def merge(self, other: "RagStats") -> None:
        for mine, theirs in ((self.counts, other.counts), (self.tokens, other.tokens),
                             (self.samples, other.samples),
                             (self.passage_tokens, other.passage_tokens)):
            mine.update(theirs)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict: the counters, mean tokens per label and the realised retrieval shares:
        ``fired_rate`` = passages inserted / calls, ``question_like_rate`` = question-like
        messages / calls, ``gate_rate`` = gate opened / question-like, ``distractor_share`` =
        distractors / passages inserted, ``later_turn_share`` = passages before a later user
        turn / passages inserted."""
        c = self.counts
        calls, fired = c["retrieval:calls"], c["retrieval:fired"]
        ql, opened = c["retrieval:question_like"], c["retrieval:gate_open"]
        distractors = c["retrieval:distractor_ranked"] + c["retrieval:distractor_random"]
        return {
            "counts": dict(sorted(c.items())),
            "samples": dict(sorted(self.samples.items())),
            "mean_tokens": {k: self.tokens[k] / n for k, n in sorted(self.samples.items()) if n},
            "mean_passage_tokens": {k: self.passage_tokens[k] / n
                                    for k, n in sorted(self.samples.items()) if n},
            "retrieval": {"calls": calls, "fired": fired,
                          "fired_rate": fired / calls if calls else 0.0,
                          "question_like_rate": ql / calls if calls else 0.0,
                          "gate_rate": opened / ql if ql else 0.0,
                          "distractor_share": distractors / fired if fired else 0.0,
                          "later_turn_share": c["retrieval:later_turn"] / fired if fired else 0.0},
        }


def _count(stats: RagStats | None, key: str) -> None:
    if stats is not None:
        stats.count(key)


# ---------------------------------------------------------------------------- library
@dataclass(frozen=True)
class LibPassage:
    """A passage of the extended contract (module docstring)."""

    pid: int
    title_id: int
    title: str
    body: str
    score: float
    text: str
    kind: str = "wiki"


def kind_of(passage: Any) -> str:
    """``passage.kind`` (``"wiki"`` for passages of a library without kinds)."""
    return getattr(passage, "kind", None) or "wiki"


class LibraryView:
    """The library contract the builders use, on top of ``bes.library.retrieve.Library`` or any
    object with the extended contract (module docstring). Apart from the wrapped library it
    keeps a small cache of the passages read last (results do not depend on it; not pickled)
    and, once :meth:`prepare_tasks` ran, the entries each library task draws from."""

    CACHE = 32

    def __init__(self, lib: Any):
        self.lib = lib.lib if isinstance(lib, LibraryView) else lib
        self.native = callable(getattr(self.lib, "kind_range", None))
        gate = getattr(getattr(self.lib, "cfg", None), "gate", None)
        self.allowed_types = tuple(getattr(gate, "allowed_types", DEFAULT_ALLOWED_TYPES))
        self._cache: dict[int, Any] = {}
        self.task_pids: dict[str, np.ndarray] = dict(lib.task_pids) if isinstance(lib, LibraryView) else {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    # ------------------------------------------------------------------ tasks
    def task_entries(self, task: str) -> range | np.ndarray:
        """Entries library task ``task`` draws from (``pick_entry``): every entry of its kind, or
        after :meth:`prepare_tasks` the reading-suitable ones for ``"library_read"``."""
        got = self.task_pids.get(task)
        return got if got is not None else self.kind_range(LIBRARY_TASKS[task])

    def prepare_tasks(self, mix: MixCfg = MixCfg(), cache_dir: str | os.PathLike | None = None) -> "LibraryView":
        """Deal reading tasks only over the entries :func:`reading_suitable` keeps (computed
        once per library content and rule: memoised in this process and saved under
        ``cache_dir``); returns ``self``."""
        mask = reading_suitable(self, mix, cache_dir)
        pids = (self.kind_range("wiki").start + np.flatnonzero(mask)).astype(np.int32)
        pids.setflags(write=False)
        self.task_pids["library_read"] = pids
        return self

    def _wrap(self, p: Any) -> Any:
        if not hasattr(p, "kind"):
            p = LibPassage(int(p.pid), int(p.title_id), p.title, p.body, float(p.score), p.text, "wiki")
        if len(self._cache) >= self.CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[int(p.pid)] = p
        return p

    # ------------------------------------------------------------------ structure
    @property
    def n_passages(self) -> int:
        return int(self.lib.n_passages)

    def kind_range(self, kind: str) -> range:
        """Contiguous pid range of ``kind`` (``"wiki"`` or ``"qa"``)."""
        if kind not in LIB_KINDS:
            raise KeyError(f"unknown library kind {kind!r}")
        if self.native:
            r = self.lib.kind_range(kind)
            return range(int(r.start), int(r.stop))
        n = self.n_passages
        return range(0, n) if kind == "wiki" else range(n, n)

    def kind_of_pid(self, pid: int) -> str:
        for kind in LIB_KINDS:
            if pid in self.kind_range(kind):
                return kind
        raise IndexError(f"pid {pid} is in no kind range")

    def _kinds_ok(self, kinds: Sequence[str] | None) -> bool:
        return self.native or kinds is None or "wiki" in kinds

    def passage(self, pid: int) -> Any:
        """One passage with its ``kind``."""
        hit = self._cache.get(int(pid))
        return hit if hit is not None else self._wrap(self.lib.passage(int(pid)))

    def title_id(self, pid: int) -> int:
        """Title id of ``pid`` (without reading the passage text when the library allows)."""
        fn = getattr(self.lib, "title_id_of", None)
        return int(fn(int(pid))) if callable(fn) else int(self.lib.passage(int(pid)).title_id)

    def title_range(self, tid: int) -> range:
        return self.lib.title_range(int(tid))

    def is_lead(self, pid: int) -> bool:
        """True when ``pid`` is the first passage of its title (an article's lead)."""
        return self.title_range(self.title_id(pid)).start == int(pid)

    def question_like(self, text: str) -> bool:
        """The library's own query typing: ``query_type(text)`` is an allowed gate type."""
        from bes.library.retrieve import query_type

        return query_type(text) in self.allowed_types

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, exclude_pids: Sequence[int] = (), exclude_titles: Sequence[int] = (),
                 kinds: Sequence[str] | None = None, context: Sequence[str] | None = None) -> Any:
        """Gated retrieval (``Library.retrieve``) or ``None``. ``context``: the earlier user
        turns of the conversation, passed on exactly as the responder passes them."""
        if not self._kinds_ok(kinds):
            return None
        kw = {"kinds": tuple(kinds)} if self.native and kinds is not None else {}
        if context and self.native:
            try:
                p = self.lib.retrieve(query, exclude_pids=tuple(exclude_pids),
                                      exclude_titles=tuple(exclude_titles),
                                      context=list(context), **kw)
                return None if p is None else self._wrap(p)
            except TypeError:                       # a library object without the context argument
                pass
        p = self.lib.retrieve(query, exclude_pids=tuple(exclude_pids),
                              exclude_titles=tuple(exclude_titles), **kw)
        return None if p is None else self._wrap(p)

    def retrieve_topk(self, query: str, k: int, exclude_pids: Sequence[int] = (),
                      exclude_titles: Sequence[int] = (), kinds: Sequence[str] | None = None) -> list:
        """The ``k`` best passages without the gate."""
        if k <= 0 or not self._kinds_ok(kinds):
            return []
        kw = {"kinds": tuple(kinds)} if self.native and kinds is not None else {}
        out = self.lib.retrieve_topk(query, k, exclude_pids=tuple(exclude_pids),
                                     exclude_titles=tuple(exclude_titles), **kw)
        return [self._wrap(p) for p in out]

    def ranked_pids(self, query: str, k: int, exclude_pids: Sequence[int] = (),
                    exclude_titles: Sequence[int] = (), kinds: Sequence[str] | None = None) -> list[int]:
        """Pids of the ``k`` best passages (score order, no gate). When only wiki passages are
        asked for (or the library has no kinds) and the library has ``search`` (BM25 ranking of
        wiki passages, ``bes.library.retrieve.Library``), it ranks without reading passages;
        otherwise ``retrieve_topk`` (whose passages stay in the cache)."""
        if k <= 0 or not self._kinds_ok(kinds):
            return []
        search = getattr(self.lib, "search", None)
        wiki_only = not self.native or (kinds is not None and tuple(kinds) == ("wiki",))
        if wiki_only and callable(search):
            pids, _ = search(query, tuple(exclude_pids), tuple(exclude_titles))
            return [int(p) for p in np.asarray(pids)[:k].tolist()]
        return [int(p.pid) for p in self.retrieve_topk(query, k, exclude_pids, exclude_titles, kinds)]

    def random_pid(self, rng: np.random.Generator, kinds: Sequence[str] | None = None,
                   exclude_pids: Sequence[int] = (), exclude_titles: Sequence[int] = (),
                   tries: int = 8) -> int | None:
        """A uniformly drawn pid of ``kinds`` (all passages for ``None``) that is not excluded;
        ``None`` when ``tries`` draws all hit excluded passages (or there is none)."""
        ranges = [self.kind_range(kd) for kd in (kinds if kinds is not None else LIB_KINDS)]
        starts = np.cumsum([0] + [len(r) for r in ranges])
        total = int(starts[-1])
        if total == 0:
            return None
        ex_p, ex_t = set(int(p) for p in exclude_pids), set(int(t) for t in exclude_titles)
        for _ in range(tries):
            j = int(rng.integers(total))
            r = int(np.searchsorted(starts, j, side="right")) - 1
            pid = ranges[r].start + j - int(starts[r])
            if pid in ex_p or (ex_t and self.title_id(pid) in ex_t):
                continue
            return pid
        return None


def as_view(lib: Any) -> LibraryView:
    """``lib`` behind the builders' contract (a :class:`LibraryView` is returned unchanged)."""
    return lib if isinstance(lib, LibraryView) else LibraryView(lib)


def pick_distractor(view: LibraryView, query: str, rng: np.random.Generator, mix: MixCfg = MixCfg(), *,
                    exclude_pids: Sequence[int] = (), exclude_titles: Sequence[int] = (),
                    kinds: Sequence[str] | None = None,
                    skip_top_title: bool = False) -> tuple[Any, str]:
    """``(passage, mode)``: with probability ``mix.distractor_random`` a uniformly drawn
    passage (``mode = "random"``), else one of the ``mix.distractor_topk`` best-ranked passages
    for ``query`` (``"ranked"``; with ``skip_top_title`` the article of the best passage is left
    out, since it may answer the query; a best "qa" entry, which has no article (title id -1),
    is left out alone); each mode falls back to the other. ``(None, "")`` when nothing is left."""
    first = "random" if rng.random() < mix.distractor_random else "ranked"
    for mode in (first, "ranked" if first == "random" else "random"):
        if mode == "random":
            pid = view.random_pid(rng, kinds, exclude_pids, exclude_titles)
            if pid is not None:
                return view.passage(pid), "random"
            continue
        k = max(1, int(mix.distractor_topk))
        pids = view.ranked_pids(query, k + (1 if skip_top_title else 0), exclude_pids,
                                exclude_titles, kinds)
        if skip_top_title and pids:
            top = view.title_id(pids[0])
            pids = [p for p in pids[1:] if top < 0 or view.title_id(p) != top]
        if pids:
            return view.passage(pids[int(rng.integers(len(pids)))]), "ranked"
    return None, ""


def self_exclusion(p: Any) -> dict[str, tuple[int, ...]]:
    """``exclude_pids`` / ``exclude_titles`` of a sample built from entry ``p``: the entry and
    its article (qa entries have no article: title id -1)."""
    return {"exclude_pids": (int(p.pid),),
            "exclude_titles": (int(p.title_id),) if int(p.title_id) >= 0 else ()}


def _weighted_choice(rng: np.random.Generator, options: Sequence[tuple[Any, float]]) -> Any:
    weights = np.asarray([max(0.0, float(w)) for _, w in options], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("no option has a positive weight")
    return options[int(rng.choice(len(options), p=weights / weights.sum()))][0]


def _no_constraint(text: str) -> bool:
    return C.compile(text).empty


def _weighted_order(rng: np.random.Generator, weights: Sequence[float]) -> list[int]:
    """Indices in a random order drawn by weight without replacement (keys ``u ** (1 / w)``)."""
    w = np.asarray(weights, dtype=np.float64)
    keys = rng.random(w.size) ** (1.0 / w)
    return [int(i) for i in np.argsort(-keys, kind="stable")]


def _first_ok(rng: np.random.Generator, templates: Sequence[tuple[str, float]], prefer: Any = None,
              **fmt: str) -> str | None:
    """The first template, in an order drawn by weight, whose filled-in text compiles to no
    constraint and satisfies ``prefer`` (a predicate; the first text without constraint when
    none does); ``None`` when every text compiles to a constraint."""
    fallback = None
    for i in _weighted_order(rng, [w for _, w in templates]):
        text = templates[i][0].format(**fmt)
        if not _no_constraint(text):
            continue
        if prefer is None or prefer(text):
            return text
        if fallback is None:
            fallback = text
    return fallback


def _probs(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return w / w.sum()


# ------------------------------------------------------------------ insert_retrieval
def first_user_turn(spans: Sequence[tuple[int, int, int]]) -> int | None:
    """Index in ``spans`` (``samples.turn_spans``) of the first user turn, ``None`` if none."""
    for k, (role, _, _) in enumerate(spans):
        if role == USER:
            return k
    return None


def fit_served(sample: Sample, lib_start: int, max_len: int) -> Sample | None:
    """``sample``, whose library turn starts at token ``lib_start`` right before the user turn
    it serves, within ``max_len`` tokens as ``bes.decode.respond.fit_prompt`` fits a prompt:
    the turns after the served exchange go first (``samples.fit``), then the oldest exchanges
    (``samples.window``: the system turn stays); ``None`` when not even the system turn, the
    passage and the served exchange fit (the responder would then drop the passage)."""
    cuts = conversation_cuts(sample.ids)
    served = cuts[cuts > lib_start]
    if served.size == 0:
        return None
    end = int(served[0])
    keep = head_stop(sample.ids)
    for s in exchange_starts(sample.ids):
        if s > lib_start:
            break
        if (keep + end - s if s > keep else end) <= max_len:
            return window(sample, s, max_len)
    return None


def insert_retrieval(sample: Sample, lib: Any, tok, rng: np.random.Generator,
                     cfg: DataCfg = DataCfg(), mix: MixCfg = MixCfg(), *, max_len: int | None = None,
                     stats: RagStats | None = None) -> Sample:
    """``sample`` with one ``LIBRARY`` turn where the responder would put one, or ``sample``
    unchanged (rules in the module docstring). The retrieval excludes ``sample.exclude_pids`` /
    ``exclude_titles``. With ``max_len`` the result is fitted by :func:`fit_served`; when not
    even the served exchange fits with its passage the sample is returned unchanged. Tags gain
    ``retrieval:gated``, ``retrieval:ranked`` or ``retrieval:random``, ``retrieval:later_turn``
    when a later user turn is served, and ``window`` when old exchanges were dropped."""
    view = as_view(lib)
    _count(stats, "retrieval:calls")
    if not sample.is_conversation:
        _count(stats, "retrieval:skip_not_conversation")
        return sample
    spans = turn_spans(sample.ids)
    users = [k for k, (role, _, _) in enumerate(spans) if role == USER]
    if not users:
        _count(stats, "retrieval:skip_no_user")
        return sample
    if any(role == LIBRARY for role, _, _ in spans):
        _count(stats, "retrieval:skip_has_library")
        return sample
    k, later = users[0], False
    if len(users) > 1 and rng.random() < mix.retrieval_later_turn:
        k, later = users[1 + int(rng.integers(len(users) - 1))], True
    _, start, stop = spans[k]
    query = tok.decode(sample.ids[start + 1:stop - 1].tolist())
    if not view.question_like(query):
        _count(stats, "retrieval:skip_not_question")
        return sample
    _count(stats, "retrieval:question_like")
    ex_p, ex_t = tuple(sample.exclude_pids), tuple(sample.exclude_titles)
    earlier = [tok.decode(sample.ids[a + 1:b - 1].tolist()) for u in users if u < k
               for _, a, b in (spans[u],)]
    gated = view.retrieve(query, ex_p, ex_t, context=earlier)
    if gated is None:
        _count(stats, "retrieval:gate_closed")
        return sample
    _count(stats, "retrieval:gate_open")
    passage, mode = gated, "gated"
    if rng.random() < cfg.distractor_frac:
        tid = int(gated.title_id)
        d, d_mode = pick_distractor(view, query, rng, mix, exclude_pids=ex_p + (int(gated.pid),),
                                    exclude_titles=ex_t + ((tid,) if tid >= 0 else ()),
                                    skip_top_title=True)
        if d is None:
            _count(stats, "retrieval:no_distractor")          # the gated passage stays
        else:
            passage, mode = d, d_mode
    turn, turn_roles = cf.render_turn("library", passage.text, tok)
    ids = np.concatenate([sample.ids[:start], turn, sample.ids[start:]])
    roles = np.concatenate([sample.roles[:start], turn_roles, sample.roles[start:]])
    out = sample.with_tokens(ids, roles, add_tags=(f"retrieval:{mode}",) + (("retrieval:later_turn",) if later else ()))
    if max_len is not None:
        out = fit_served(out, start, max_len)
        if out is None:
            _count(stats, "retrieval:too_long")
            return sample
    _count(stats, "retrieval:gated" if mode == "gated" else f"retrieval:distractor_{mode}")
    _count(stats, "retrieval:fired")
    if later:
        _count(stats, "retrieval:later_turn")
    if stats is not None:
        stats.built(f"retrieval:{mode}", out, len(turn))
    return out


# ----------------------------------------------------------------------- reading tasks
#: Questions about an article (lead passages): (template, weight); ``{t}`` is the title without
#: its disambiguator. Chosen to read naturally whatever the entity type.
TITLE_TEMPLATES: tuple[tuple[str, float], ...] = (
    ("Tell me about {t}.", 3.0),
    ("Who or what is {t}?", 2.0),
    ("What can you tell me about {t}?", 2.0),
    ("What do you know about {t}?", 1.5),
    ("Can you tell me about {t}?", 1.5),
    ("Give me a short summary of {t}.", 1.5),
    ("What is {t}?", 1.0),
    ("{t}?", 1.0),
    ("I'd like to know about {t}.", 1.0),
    ("Could you give me an overview of {t}?", 1.0),
    ("Tell me something about {t}.", 1.0),
    ("What should I know about {t}?", 0.8),
    ("Give me a quick introduction to {t}.", 0.8),
    ("Some information about {t}, please.", 0.5),
    ("tell me about {t}", 0.5),
    ("{t}", 0.4),
)
#: Questions about one section of an article; ``{h}`` is the noun phrase of its heading.
SECTION_TEMPLATES: tuple[tuple[str, float], ...] = (
    ("Tell me about the {h} of {t}.", 3.0),
    ("What can you tell me about the {h} of {t}?", 2.0),
    ("What do you know about the {h} of {t}?", 1.0),
    ("Describe the {h} of {t}.", 1.0),
    ("Can you summarize the {h} of {t}?", 1.0),
    ("I'd like to know about the {h} of {t}.", 0.8),
)
#: Section headings (lower case) that make a natural "the <noun phrase> of <title>" question.
SECTION_HEADINGS: dict[str, str] = {h: h for h in (
    "history", "plot", "career", "cast", "geography", "early life", "background", "life",
    "story", "climate", "reception", "personal life", "production", "later life",
    "early life and education", "demographics", "early years", "origin", "origins",
    "development", "awards", "death", "culture", "legacy", "gameplay", "life and career",
    "discography", "filmography", "name", "economy", "release", "characters", "early history",
    "music", "religion", "geology", "causes", "club career", "design", "voice cast", "members",
    "population", "education", "characteristics", "naming", "symptoms", "treatment", "diagnosis",
    "habitat", "diet", "reproduction", "uses", "properties", "structure", "transport",
    "transportation", "tourism", "politics", "government", "languages", "international career",
    "playing career", "professional career", "presidency", "philosophy", "mythology",
    "infrastructure", "life cycle", "distribution", "behaviour", "behavior", "etymology",
    "rules", "format", "personality", "biology", "anatomy", "honours", "honors", "works")}
SECTION_HEADINGS.update({"plot summary": "plot", "synopsis": "plot", "biography": "life",
                         "distribution and habitat": "habitat", "selected works": "works",
                         "storyline": "story"})
_TERMINAL = re.compile(r"[.!?][\"'”’)\]]*$")
_LIST_LINE = re.compile(r"\s*(?:[-*•]|\d+[.)])\s")
MAX_TITLE_WORDS = 12


def _passage_lines(p: Any) -> list[str] | None:
    """Lines of the passage text after its title line (``None`` if the text does not start with
    the title line)."""
    if not p.title or not p.text.startswith(p.title + "\n"):
        return None
    return p.text[len(p.title) + 1:].split("\n")


def _answer_sentences(paragraph: str, rng: np.random.Generator, mix: MixCfg) -> str | None:
    """The first one to three sentences of ``paragraph`` (count drawn with ``mix.read_one /
    read_two / read_three`` among the counts the paragraph has; cut back to the last sentence
    that ends with ``.``, ``!`` or ``?``); ``None`` when there is no such sentence."""
    from bes.library.clean import split_sentences

    if not paragraph.strip() or _LIST_LINE.match(paragraph):
        return None
    sents = split_sentences(paragraph)
    if not sents:
        return None
    weights = _answer_weights(len(sents), mix)
    k = int(rng.choice(len(weights), p=_probs(weights))) + 1
    while k > 0 and not _TERMINAL.search(sents[k - 1]):
        k -= 1
    if k == 0:
        return None
    answer = " ".join(sents[:k])
    return answer if len(answer.split()) >= 3 else None


def _reading_form(title: str, lines: list[str] | None, lead: bool,
                  mix: MixCfg) -> tuple[str, str, str, str | None] | str:
    """``(title without disambiguator, form, paragraph, heading)`` of a reading task on a
    passage (``lines``: its lines after the title line, ``None`` without a title line; ``lead``:
    the first passage of its article), or the reason why it is unsuitable. No random draw."""
    from bes.library.terms import strip_disambiguator

    if lines is None:
        return "no_title_line"
    t = strip_disambiguator(title).strip()
    if not t or len(t.split()) > MAX_TITLE_WORDS or t.lower().startswith("list of"):
        return "title"
    if lead:
        return t, "title", lines[0] if lines else "", None
    if not mix.section_tasks:
        return "section"
    heading = SECTION_HEADINGS.get(lines[0].strip().lower()) if lines else None
    if heading is None or len(lines) < 2:
        return "section"
    return t, "section", lines[1], heading


def _answer_weights(n_sentences: int, mix: MixCfg) -> list[float]:
    weights = [mix.read_one, mix.read_two, mix.read_three][:n_sentences]
    if sum(weights) <= 0:
        weights = [1.0] + [0.0] * (len(weights) - 1)
    return weights


def _answer_possible(paragraph: str, mix: MixCfg) -> bool:
    """Whether :func:`_answer_sentences` returns an answer for some draw (it draws only among
    the sentence counts with a positive weight)."""
    from bes.library.clean import split_sentences

    if not paragraph.strip() or _LIST_LINE.match(paragraph):
        return False
    sents = split_sentences(paragraph)
    for k0, w in enumerate(_answer_weights(len(sents), mix), start=1):
        if w <= 0:
            continue
        k = k0
        while k > 0 and not _TERMINAL.search(sents[k - 1]):
            k -= 1
        if k > 0 and len(" ".join(sents[:k]).split()) >= 3:
            return True
    return False


def reading_qa(p: Any, view: LibraryView, rng: np.random.Generator, mix: MixCfg = MixCfg(),
               question_like: bool = False) -> tuple[str, str, str] | str:
    """``(question, answer, form)`` of a reading task on passage ``p`` (``form`` is
    ``"title"`` or ``"section"``), or the reason (a string) why ``p`` is unsuitable. With
    ``question_like`` a question the library's query typing accepts is preferred."""
    got = _reading_form(p.title, _passage_lines(p), view.is_lead(p.pid), mix)
    if isinstance(got, str):
        return got
    title, form, paragraph, heading = got
    answer = _answer_sentences(paragraph, rng, mix)
    if answer is None:
        return "answer"
    prefer = view.question_like if question_like else None
    if form == "title":
        question = _first_ok(rng, TITLE_TEMPLATES, prefer, t=title)
    else:
        question = _first_ok(rng, SECTION_TEMPLATES, prefer, t=title, h=heading)
    if question is None:
        return "constraint"
    return question, answer, form


#: Version of the rule of :func:`reading_suitable` (part of its cache key and of the loader's
#: library fingerprint); bump when the checks change.
SUITABLE_RULE = "reading-suitable/1"
_SUITABLE_MEMO: dict[str, np.ndarray] = {}


def _suitable_key(view: LibraryView, mix: MixCfg) -> str | None:
    """Digest of the library's stored text structure (block and title-block CRCs, passage and
    title directories) and of the rule; ``None`` for a library without ``packed`` arrays."""
    packed = getattr(view.lib, "packed", None)
    names = ("block_crc", "block_raw", "tblock_crc", "title_first", "pas_off")
    if packed is None or not all(hasattr(packed, n) for n in names):
        return None
    h = hashlib.blake2b(digest_size=16)
    rule = (SUITABLE_RULE, MAX_TITLE_WORDS, sorted(SECTION_HEADINGS.items()), bool(mix.section_tasks),
            [float(x) > 0 for x in (mix.read_one, mix.read_two, mix.read_three)],
            view.kind_range("wiki").start, len(view.kind_range("wiki")))
    h.update(repr(rule).encode("utf-8"))
    for n in names:
        a = np.ascontiguousarray(np.asarray(getattr(packed, n)))
        h.update(n.encode("ascii") + str(a.dtype).encode("ascii") + np.int64(a.size).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def _suitable_scan(view: LibraryView, mix: MixCfg) -> np.ndarray:
    """``bool[n_wiki]`` by reading every wiki passage once, in id order."""
    r = view.kind_range("wiki")
    out = np.zeros(len(r), dtype=bool)
    packed = getattr(view.lib, "packed", None)
    fast = (view.native and r.start == 0 and packed is not None and callable(getattr(packed, "iter_bodies", None))
            and callable(getattr(packed, "all_titles", None)) and hasattr(packed, "title_first")
            and int(getattr(packed, "n_passages", -1)) == len(r))
    if fast:
        first = np.asarray(packed.title_first, dtype=np.int64)
        fast = first.size >= 1 and int(first[0]) == 0 and int(first[-1]) == len(r) and bool((np.diff(first) >= 0).all())
    if not fast:
        for i, pid in enumerate(r):
            p = view.passage(pid)
            got = _reading_form(p.title, _passage_lines(p), view.is_lead(pid), mix)
            out[i] = not isinstance(got, str) and _answer_possible(got[2], mix)
        return out
    titles = packed.all_titles()
    tid_of = np.repeat(np.arange(first.size - 1), np.diff(first))
    lead = np.zeros(len(r), dtype=bool)
    lead[first[:-1][first[:-1] < len(r)]] = True
    for pid, body in packed.iter_bodies():       # one block decompressed at a time
        title = titles[int(tid_of[pid])]
        got = _reading_form(title, body.split("\n") if title else None, bool(lead[pid]), mix)
        out[pid] = not isinstance(got, str) and _answer_possible(got[2], mix)
    return out


def reading_suitable(lib: Any, mix: MixCfg = MixCfg(),
                     cache_dir: str | os.PathLike | None = None) -> np.ndarray:
    """``bool[n_wiki]`` (read-only): the "wiki" entries a reading task can be built from, i.e.
    those that pass every check of :func:`reading_qa` that needs no random draw (a title line,
    the title rules, a lead passage or a known section heading, an answer of one to three
    sentences for some drawn count). A reading task on an entry outside it always declines; one
    on an entry inside it can still decline when every question template compiles to a
    constraint (rare). One pass over the library (``Packed.iter_bodies`` when the library has
    it: one block decompression per 64 kB block; else ``passage`` per entry); the result is
    memoised per library content and rule, and with ``cache_dir`` saved there as
    ``reading_suitable-<digest>.npy`` and read back by later processes."""
    view = as_view(lib)
    key = _suitable_key(view, mix)
    n = len(view.kind_range("wiki"))
    hit = _SUITABLE_MEMO.get(key) if key is not None else None
    path = Path(cache_dir) / f"reading_suitable-{key}.npy" if key is not None and cache_dir else None
    if hit is None and path is not None and path.exists():
        try:
            got = np.load(path)
            if got.dtype == np.bool_ and got.shape == (n,):
                hit = got
        except (OSError, ValueError):
            hit = None
    if hit is None:
        hit = _suitable_scan(view, mix)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                np.save(f, hit)
            os.replace(tmp, path)
    hit = np.array(hit, dtype=bool, copy=True)
    hit.setflags(write=False)
    if key is not None:
        _SUITABLE_MEMO[key] = hit
    return hit


def _variant(rng: np.random.Generator, variant: str | None, options: Sequence[tuple[str, float]]) -> str:
    if variant is not None:
        if variant not in {name for name, _ in options}:
            raise ValueError(f"unknown variant {variant!r}; known: {[n for n, _ in options]}")
        return variant
    return _weighted_choice(rng, options)


def reading_task(pid: int, lib: Any, tok, rng: np.random.Generator, rule: WeightRule = WeightRule(),
                 mix: MixCfg = MixCfg(), *, variant: str | None = None,
                 stats: RagStats | None = None) -> Sample | None:
    """Reading task on "wiki" passage ``pid`` (module docstring); ``variant`` in ``open |
    closed | distractor`` (drawn with ``mix.read_*`` when ``None``). The question of an
    ``open`` or ``distractor`` task is one the library's query typing accepts (a passage only
    appears where the gate could have put one); without such a template the task becomes
    ``closed`` (counted as ``read:<variant>_not_question``). Labels: ``source =
    "library_read"``, ``kind = "library"``, tags ``read:<variant>`` and ``read:<form>``.
    ``None`` when the passage is unsuitable (counted as ``read:declined_<reason>``)."""
    view = as_view(lib)
    p = view.passage(pid)
    if kind_of(p) != "wiki":
        _count(stats, "read:declined_kind")
        return None
    variant = _variant(rng, variant, (("open", mix.read_open), ("closed", mix.read_closed),
                                      ("distractor", mix.read_distractor)))
    got = reading_qa(p, view, rng, mix, question_like=variant != "closed")
    if isinstance(got, str):
        _count(stats, f"read:declined_{got}")
        return None
    question, answer, form = got
    if variant != "closed" and not view.question_like(question):
        _count(stats, f"read:{variant}_not_question")
        variant = "closed"
    excl = self_exclusion(p)
    messages: list[dict[str, Any]] = []
    if variant == "open":
        messages.append({"role": "library", "content": p.text})
    elif variant == "distractor":
        d, mode = pick_distractor(view, question, rng, mix, kinds=("wiki",), **excl)
        if d is None:
            _count(stats, "read:distractor_fallback")
            variant = "closed"
        else:
            _count(stats, f"read:distractor_{mode}")
            messages.append({"role": "library", "content": d.text})
    messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    sample = conversation_sample(messages, tok, rule, source="library_read", kind="library",
                                 tags=(f"read:{variant}", f"read:{form}"), **excl)
    _count(stats, f"read:{variant}")
    if stats is not None:
        stats.built(f"read:{variant}", sample, _library_tokens(sample))
    return sample


def _library_tokens(sample: Sample) -> int:
    return int(np.count_nonzero(sample.roles == ROLE_LIBRARY))


# ------------------------------------------------------------------ question-answer tasks
#: Templates of the assistant's answer around a short stored answer ``{a}`` (``{e}`` is the
#: final period when the answer has none).
QA_ANSWER_TEMPLATES: tuple[tuple[str, float], ...] = (
    ("{A}{e}", 0.45),
    ("The answer is {a}{e}", 0.35),
    ("That would be {a}{e}", 0.20),
)
QA_SHORT_WORDS = 8
_SENTENCE_INSIDE = re.compile(r"[.!?][\"'”’)\]]*\s")


def qa_answer_text(answer: str, rng: np.random.Generator) -> str:
    """The assistant's reply for a stored answer: a long answer (more than ``QA_SHORT_WORDS``
    words or several sentences) as it is, a short one through ``QA_ANSWER_TEMPLATES`` (first
    letter capitalised in the bare form when the first word has no capital letter)."""
    a = answer.strip()
    if len(a.split()) > QA_SHORT_WORDS or _SENTENCE_INSIDE.search(a):
        return a
    e = "" if a[-1:] in ".!?" else "."
    first = a.split()[0] if a.split() else a
    cap = a[:1].upper() + a[1:] if first.islower() else a
    return _weighted_choice(rng, QA_ANSWER_TEMPLATES).format(a=a, A=cap, e=e)


def _norm_answer(a: str) -> str:
    return " ".join(re.findall(r"\w+", a.lower()))


def qa_task(pid: int, lib: Any, tok, rng: np.random.Generator, rule: WeightRule = WeightRule(),
            mix: MixCfg = MixCfg(), *, variant: str | None = None,
            stats: RagStats | None = None) -> Sample | None:
    """Question-answer task on "qa" entry ``pid`` (module docstring); ``variant`` in ``copy |
    neighbour | closed`` (drawn with ``mix.qa_*`` when ``None``; ``neighbour`` falls back to
    ``closed`` when no other stored question matches, and both library variants do when the
    library's query typing does not accept the question). Labels: ``source = "library_qa"``,
    ``kind = "library"``, tag ``qa:<variant>``. ``None`` when the entry is unsuitable (empty,
    or a question that compiles to a constraint)."""
    view = as_view(lib)
    p = view.passage(pid)
    if kind_of(p) != "qa":
        _count(stats, "qa:declined_kind")
        return None
    question, answer = p.title.strip(), p.body.strip()
    if not question or not answer:
        _count(stats, "qa:declined_empty")
        return None
    if not _no_constraint(question):
        _count(stats, "qa:declined_constraint")
        return None
    variant = _variant(rng, variant, (("copy", mix.qa_copy), ("neighbour", mix.qa_neighbour),
                                      ("closed", mix.qa_closed)))
    if variant != "closed" and not view.question_like(question):
        _count(stats, f"qa:{variant}_not_question")
        variant = "closed"
    excl = self_exclusion(p)
    messages: list[dict[str, Any]] = []
    if variant == "copy":
        messages.append({"role": "library", "content": p.text})
    elif variant == "neighbour":
        pids = view.ranked_pids(question, 1, kinds=("qa",), **excl)
        if not pids:
            _count(stats, "qa:neighbour_fallback")
            variant = "closed"
        else:
            nb = view.passage(pids[0])
            if _norm_answer(nb.body) == _norm_answer(answer):
                _count(stats, "qa:neighbour_same_answer")
            messages.append({"role": "library", "content": nb.text})
    messages += [{"role": "user", "content": question},
                 {"role": "assistant", "content": qa_answer_text(answer, rng)}]
    sample = conversation_sample(messages, tok, rule, source="library_qa", kind="library",
                                 tags=(f"qa:{variant}",), **excl)
    _count(stats, f"qa:{variant}")
    if stats is not None:
        stats.built(f"qa:{variant}", sample, _library_tokens(sample))
    return sample


# ------------------------------------------------------------------------- dispatch
def pick_entry(lib: Any, task: str, counter: int, seed: int = 0) -> int | None:
    """The library entry of draw number ``counter`` of ``task`` (``"library_read"`` or
    ``"library_qa"``): every entry the task draws from (``LibraryView.task_entries``: the
    entries of its kind, or the reading-suitable ones of a prepared view) once per epoch of
    that many draws, in an order keyed by ``(seed, task, epoch)`` (``samples.permute_index``).
    ``None`` when there is no entry."""
    entries = as_view(lib).task_entries(task)
    if len(entries) == 0:
        return None
    epoch, i = divmod(int(counter), len(entries))
    return int(entries[permute_index(i, len(entries), "library_task", task, int(seed), epoch)])


def library_task(task: str, pid: int, lib: Any, tok, rng: np.random.Generator,
                 rule: WeightRule = WeightRule(), mix: MixCfg = MixCfg(), *,
                 stats: RagStats | None = None) -> Sample | None:
    """:func:`reading_task` (``task = "library_read"``) or :func:`qa_task` (``"library_qa"``)."""
    if task == "library_read":
        return reading_task(pid, lib, tok, rng, rule, mix, stats=stats)
    if task == "library_qa":
        return qa_task(pid, lib, tok, rng, rule, mix, stats=stats)
    raise KeyError(f"unknown library task {task!r}; known: {sorted(LIBRARY_TASKS)}")


__all__ = ["RagStats", "LibPassage", "LibraryView", "as_view", "kind_of", "pick_distractor", "self_exclusion",
           "first_user_turn", "fit_served", "insert_retrieval", "reading_qa", "reading_suitable",
           "reading_task", "qa_answer_text", "qa_task", "pick_entry", "library_task", "TITLE_TEMPLATES",
           "SECTION_TEMPLATES", "SECTION_HEADINGS", "QA_ANSWER_TEMPLATES", "LIBRARY_TASKS", "LIB_KINDS",
           "SUITABLE_RULE"]
