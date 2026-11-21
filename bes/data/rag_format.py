"""Retrieval-aware samples built with the written library (``docs/SPEC_MODEL_TRAIN.md`` section 6,
``docs/PLAN.md`` sections 4D and 5).

Builders
--------
Every builder takes the library, the tokenizer and a ``numpy.random.Generator`` (all random
choices come from it) and returns :class:`bes.data.samples.Sample` objects whose weights follow
the given :class:`~bes.data.samples.WeightRule`.

:func:`insert_retrieval`
    A conversation whose first user turn is question-like (the library's own
    ``query_type``, ``lib.cfg.gate.allowed_types``) gets a ``LIBRARY`` turn right before that
    user turn: with probability ``1 - DataCfg.distractor_frac`` the gated passage
    (``lib.retrieve``; nothing is inserted when the gate stays closed, as at inference), else a
    distractor (``MixCfg.distractor_random`` of them a uniformly drawn passage, the others one
    of the next ``MixCfg.distractor_topk`` ranked passages whose article is not the top
    article). Documents, conversations that already have a library turn before their first
    user turn (the retrieval sources, including SQuAD v2's "not in the passage" questions) and
    conversations without a question-like first user turn pass through unchanged.
:func:`reading_task`
    From a "wiki" passage, no language model involved. Lead passage: the user asks about the
    article title (neutral templates such as "Tell me about X.", "Who or what is X?", "X?");
    section passage whose first line is a known heading (``SECTION_HEADINGS``): the user asks
    about that section ("Tell me about the history of X."). The assistant answers with the
    first one to three sentences of the passage text. Variants (``MixCfg.read_*``): ``open``
    (the passage as ``LIBRARY`` turn), ``closed`` (no library turn), ``distractor`` (a passage
    of another article). A library turn only comes before a question the library's query
    typing accepts, as at inference (else the task is closed book).
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

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bes.config import LIBRARY, USER, DataCfg, MixCfg
from bes.data import chat_format as cf
from bes.data.samples import (ROLE_LIBRARY, Sample, WeightRule, conversation_sample, fit,
                              permute_index, turn_spans)
from bes.exact import constraints as C

LIB_KINDS = ("wiki", "qa")
DEFAULT_ALLOWED_TYPES = ("question", "lookup")
#: Library tasks and the entry kind each is built from (sources of the mixture).
LIBRARY_TASKS: dict[str, str] = {"library_read": "wiki", "library_qa": "qa"}


# ------------------------------------------------------------------------- statistics
@dataclass
class RagStats:
    """Counters for the loader's log; merge the stats of several workers with :meth:`merge`.

    ``counts[key]``: events (``retrieval:calls``, ``retrieval:fired``, ``retrieval:gated``,
    ``retrieval:distractor_ranked``, ``retrieval:skip_has_library``, ``read:open``,
    ``read:declined_<reason>``, ``qa:neighbour_same_answer``, ...). ``tokens[label]`` /
    ``samples[label]``: tokens and number of built samples per label (``read:open``,
    ``qa:copy``, ``retrieval:gated``, ...); ``passage_tokens[label]``: tokens of the inserted
    library turns.
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
        """Plain dict: the counters, mean tokens per label and the retrieval rates
        (``fired_rate`` = fired / calls, ``distractor_share`` = distractors / fired)."""
        c = self.counts
        calls, fired = c["retrieval:calls"], c["retrieval:fired"]
        distractors = c["retrieval:distractor_ranked"] + c["retrieval:distractor_random"]
        return {
            "counts": dict(sorted(c.items())),
            "samples": dict(sorted(self.samples.items())),
            "mean_tokens": {k: self.tokens[k] / n for k, n in sorted(self.samples.items()) if n},
            "mean_passage_tokens": {k: self.passage_tokens[k] / n
                                    for k, n in sorted(self.samples.items()) if n},
            "retrieval": {"calls": calls, "fired": fired,
                          "fired_rate": fired / calls if calls else 0.0,
                          "distractor_share": distractors / fired if fired else 0.0},
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
    only keeps a small cache of the passages read last (results do not depend on it)."""

    CACHE = 32

    def __init__(self, lib: Any):
        self.lib = lib.lib if isinstance(lib, LibraryView) else lib
        self.native = callable(getattr(self.lib, "kind_range", None))
        gate = getattr(getattr(self.lib, "cfg", None), "gate", None)
        self.allowed_types = tuple(getattr(gate, "allowed_types", DEFAULT_ALLOWED_TYPES))
        self._cache: dict[int, Any] = {}

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
                 kinds: Sequence[str] | None = None) -> Any:
        """Gated retrieval (``Library.retrieve``) or ``None``."""
        if not self._kinds_ok(kinds):
            return None
        kw = {"kinds": tuple(kinds)} if self.native and kinds is not None else {}
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


def insert_retrieval(sample: Sample, lib: Any, tok, rng: np.random.Generator,
                     cfg: DataCfg = DataCfg(), mix: MixCfg = MixCfg(), *, max_len: int | None = None,
                     stats: RagStats | None = None) -> Sample:
    """``sample`` with a ``LIBRARY`` turn before its first user turn, or ``sample`` unchanged
    (rules in the module docstring). The retrieval excludes ``sample.exclude_pids`` /
    ``exclude_titles``. With ``max_len`` the result is cut by ``samples.fit`` (later turns go
    first); when even the first exchange no longer fits the sample is returned unchanged.
    Tags gain ``retrieval:gated``, ``retrieval:ranked`` or ``retrieval:random``."""
    view = as_view(lib)
    _count(stats, "retrieval:calls")
    if not sample.is_conversation:
        _count(stats, "retrieval:skip_not_conversation")
        return sample
    spans = turn_spans(sample.ids)
    k = first_user_turn(spans)
    if k is None:
        _count(stats, "retrieval:skip_no_user")
        return sample
    _, start, stop = spans[k]
    if k > 0 and spans[k - 1][0] == LIBRARY:
        _count(stats, "retrieval:skip_has_library")
        return sample
    query = tok.decode(sample.ids[start + 1:stop - 1].tolist())
    if not view.question_like(query):
        _count(stats, "retrieval:skip_not_question")
        return sample
    excl = dict(exclude_pids=sample.exclude_pids, exclude_titles=sample.exclude_titles)
    if rng.random() < cfg.distractor_frac:
        passage, mode = pick_distractor(view, query, rng, mix, skip_top_title=True, **excl)
        if passage is None:
            _count(stats, "retrieval:no_distractor")
            return sample
        _count(stats, f"retrieval:distractor_{mode}")
    else:
        passage, mode = view.retrieve(query, **excl), "gated"
        if passage is None:
            _count(stats, "retrieval:gate_closed")
            return sample
        _count(stats, "retrieval:gated")
    turn, turn_roles = cf.render_turn("library", passage.text, tok)
    ids = np.concatenate([sample.ids[:start], turn, sample.ids[start:]])
    roles = np.concatenate([sample.roles[:start], turn_roles, sample.roles[start:]])
    out = sample.with_tokens(ids, roles, add_tags=(f"retrieval:{mode}",))
    if max_len is not None:
        out, _ = fit(out, max_len)
        if out is None:
            _count(stats, "retrieval:too_long")
            return sample
    _count(stats, "retrieval:fired")
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
    weights = [mix.read_one, mix.read_two, mix.read_three][:len(sents)]
    if sum(weights) <= 0:
        weights = [1.0] + [0.0] * (len(weights) - 1)
    k = int(rng.choice(len(weights), p=_probs(weights))) + 1
    while k > 0 and not _TERMINAL.search(sents[k - 1]):
        k -= 1
    if k == 0:
        return None
    answer = " ".join(sents[:k])
    return answer if len(answer.split()) >= 3 else None


def reading_qa(p: Any, view: LibraryView, rng: np.random.Generator, mix: MixCfg = MixCfg(),
               question_like: bool = False) -> tuple[str, str, str] | str:
    """``(question, answer, form)`` of a reading task on passage ``p`` (``form`` is
    ``"title"`` or ``"section"``), or the reason (a string) why ``p`` is unsuitable. With
    ``question_like`` a question the library's query typing accepts is preferred."""
    from bes.library.terms import strip_disambiguator

    lines = _passage_lines(p)
    if lines is None:
        return "no_title_line"
    title = strip_disambiguator(p.title).strip()
    if not title or len(title.split()) > MAX_TITLE_WORDS or title.lower().startswith("list of"):
        return "title"
    if view.is_lead(p.pid):
        form, paragraph, heading = "title", lines[0] if lines else "", None
    else:
        if not mix.section_tasks:
            return "section"
        heading = SECTION_HEADINGS.get(lines[0].strip().lower()) if lines else None
        if heading is None or len(lines) < 2:
            return "section"
        form, paragraph = "section", lines[1]
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
    ``"library_qa"``): every entry of the task's kind once per epoch of
    ``len(kind_range)`` draws, in an order keyed by ``(seed, task, epoch)``
    (``samples.permute_index``). ``None`` when the kind has no entries."""
    r = as_view(lib).kind_range(LIBRARY_TASKS[task])
    if len(r) == 0:
        return None
    epoch, i = divmod(int(counter), len(r))
    return r.start + permute_index(i, len(r), "library_task", task, int(seed), epoch)


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
           "first_user_turn", "insert_retrieval", "reading_qa", "reading_task", "qa_answer_text",
           "qa_task", "pick_entry", "library_task", "TITLE_TEMPLATES", "SECTION_TEMPLATES",
           "SECTION_HEADINGS", "QA_ANSWER_TEMPLATES", "LIBRARY_TASKS", "LIB_KINDS"]
