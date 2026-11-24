"""Satisficing selection among the particles of one reply (PLAN 4G, ``docs/MATH.md`` F4).

Never an arg-max over a weak scorer: failures are dropped by exact checks, then the survivor
that agrees most with the other survivors is returned (minimum Bayes risk under the particle
weights with a lexical utility). Pure Python and numpy, deterministic.

Failure checks (:func:`check`), per candidate:

- ``unfinished``: the particle did not emit ``END`` (``finish`` is ``"length"`` or ``"dead"``);
- ``violation``: ``bes.exact.constraints.verify(spec, text)`` is not empty on the complete
  visible text (every constraint of the spec, also those the potentials do not enforce);
- ``degenerate``: the detectors of ``bes.eval.exact_score.degenerate_stats`` (empty, looping
  tail, a line of five or more words repeated three times outside fences, unclosed code fence,
  leaked role or template markers, a reply that opens with a stock refusal: a particle that
  answers is preferred, and when every particle refuses the refusal stands), plus a repeated-4-gram rate above ``rep4_max`` and a
  prose reply that stops in the middle of a sentence (:func:`cut_off`; neither for code or
  short-form prompts), leaked special-token strings (:func:`special_leak`), and
  ``at_budget``: a finished particle that used its whole token budget (given ``budget``).
  Such replies can be "finished": budget-aware potentials force ``END`` at the token limit;
- ``band``: the text is outside the length band of the prompt type (:func:`length_band`).

Length band (:func:`length_band`), a small explicit policy on the last user message:

=============  ==============================================  ===============================
type           detected by                                     band
=============  ==============================================  ===============================
``code``       programming words, code fences, code syntax     none
``math``       a number and an arithmetic cue                  none (majority vote instead)
``transform``  summarise / rewrite / translate / fix ...       none (depends on the input)
``short``      poem, haiku, slogan, headline, name, "only"...  none
``factual``    a short who / when / where / which / how many   1-3 sentences (and <= 90 words)
               ... question, or a short yes/no question
``writing``    write / compose / draft a text                  80-200 words
``open``       every other question or request                 80-200 words
=============  ==============================================  ===============================

Modifiers of the word bands: "brief / short / quick / concise" -> 40-200 words ("very short",
"a few sentences", "in a nutshell" -> 15-120), "in detail / detailed / thorough /
comprehensive / long" -> 150-400. Explicit user constraints always override the band: a spec
with any count (words, sentences, items, paragraphs, lines) or a format request (JSON, code
block, one of a list of answers, several responses, repeat the request, quoted) gets no band.

Selection (:func:`select`): the levels are tried in this fixed order and the first non-empty
pool wins; ``Selection.rule`` reports which one fired:

1. ``strict``: finished, no violation, not degenerate, in the band;
2. ``relax_band``: finished, no violation, not degenerate;
3. ``relax_degenerate``: finished, no violation;
4. ``best_finished``: the finished particle with the fewest violations, then the largest
   weight, then the fewest degenerate flags (no consensus);
5. ``unfinished``: no particle finished: ``"length"`` before ``"dead"``, then the fewest
   violations, then the most tokens.

Inside levels 1-3 (:func:`consensus`): for arithmetic word problems the final number of every
pool member is read with ``bes.eval.exact_score.extract_answer`` (the scorer's own rule, with
the numbers of the user turns as context) and the pool is cut to the answer with the largest
total weight (ties: more members, then the earliest member); then every member ``i`` gets the
expected utility ``score_i = sum_j w_j U(i, j)`` over the pool (normalised weights; uniform
when they sum to 0; ``U(i, i) = 1``, so a member's own weight counts as agreement with
itself: without the self term, of two survivors the lighter one would always win). The
largest score wins; ties (within ``1e-12``) go to the larger weight, then the lower index.
Utilities: ``"f1"`` (word unigram F1), ``"f12"`` (mean of unigram and bigram F1, default),
``"lcs"`` (F-measure of the longest common word subsequence).

Guarantee (tested): the result never violates a constraint when some finished particle
satisfies all of them, and otherwise has the fewest violations among the finished particles.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from bes.config import SPECIAL_TOKENS

LEVELS = ("strict", "relax_band", "relax_degenerate", "best_finished", "unfinished")
UTILITIES = ("f1", "f12", "lcs")
REP4_MAX = 0.3
TIE = 1e-12
LENGTH_UNITS = ("words", "sentences", "items", "paragraphs", "lines")
FORMAT_FLAGS = ("json_output", "code_block", "quoted")
FORMAT_VALUES = ("one_of", "responses", "repeat")

#: special-token strings that must never be visible (ours and common chat templates')
_LEAK_STRINGS = tuple(SPECIAL_TOKENS) + ("<|im_start|>", "<|im_end|>", "<|eot_id|>",
                                         "<|endoftext|>", "<|end_of_text|>",
                                         "<|begin_of_text|>", "<|start_header_id|>",
                                         "<|end_header_id|>")


# ===================================================================== candidates & band
@dataclass(frozen=True)
class Candidate:
    """One particle as the selector sees it: the complete visible ``text`` (display text,
    stripped), its normalised ``weight`` (>= 0; 0 for unfinished and dead particles), its
    ``finish`` reason (``"stop"``, ``"length"``, ``"dead"``), the number of response tokens and
    the model log-probability sum (information only)."""

    text: str
    weight: float = 1.0
    finish: str = "stop"
    n_tokens: int = 0
    logp: float | None = None

    @property
    def finished(self) -> bool:
        return self.finish == "stop"


@dataclass(frozen=True)
class Band:
    """A length band: ``words`` and/or ``sentences`` as ``(lo, hi)`` (``hi = None``: no
    bound); ``kind`` is the prompt type it was chosen for."""

    kind: str
    words: tuple[int, int | None] | None = None
    sentences: tuple[int, int | None] | None = None

    def contains(self, text: str, measured: Mapping[str, Any] | None = None) -> bool:
        m = measured if measured is not None else _measure(text)
        words = int(m["words"])
        if self.words is not None:
            lo, hi = self.words
            if words < lo or (hi is not None and words > hi):
                return False
        if self.sentences is not None:
            lo, hi = self.sentences
            n = max(int(m["sentences"]), 1 if words else 0)    # a terminator-less answer is one
            if n < lo or (hi is not None and n > hi):
                return False
        return True

    def __str__(self) -> str:
        parts = []
        for unit in ("sentences", "words"):
            iv = getattr(self, unit)
            if iv is not None:
                parts.append(f"{unit} {iv[0]}-{'' if iv[1] is None else iv[1]}")
        return f"{self.kind}: " + ", ".join(parts)


def _measure(text: str) -> dict:
    from bes.exact import constraints

    return constraints.measure(text)


_WORD = re.compile(r"[a-z0-9']+")
_CODE = re.compile(
    r"```|\b(?:python|javascript|typescript|java|c\+\+|c#|rust|golang|kotlin|swift|php|ruby|sql|"
    r"bash|shell|powershell|html|css|regex|regexp|json|yaml|xml|api|git|github|docker|numpy|pandas|"
    r"function|functions|method|class|variable|variables|loop|loops|recursion|recursive|"
    r"algorithm|compile|compiler|debug|bug|code|coding|program|programming|script|pytest|unit tests?|"
    r"list comprehension|dictionary comprehension|data structure|linked list|binary search|"
    r"string|integer|boolean|array|tuple|docstring|type hints?)\b|\bdef |==|!=|=>|\w\(\)|;\s*$",
    re.I)
_NUM = r"(?:\d+(?:[.,]\d+)*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|" \
       r"fifteen|twenty|thirty|forty|fifty|hundred|thousand|half|double|twice)"
_NUMBERS = re.compile(rf"(?<![\w.]){_NUM}\b", re.I)
_ARITH = re.compile(
    rf"\d\s*[-+*/x×÷^]\s*\(?\d|\d\s*%|\bpercent(?:age)?\b|\b(?:squared|cubed|factorial|to the power"
    rf"|square root|divided by|multiplied by|times|plus|minus)\b|^(?:compute|calculate|solve|"
    rf"evaluate|convert {_NUM})\b|\bhow (?:many|much)\b.*{_NUM}|{_NUM}\b.*\bhow (?:many|much)\b|"
    rf"{_NUM} (?:degrees? )?(?:celsius|fahrenheit|kelvin)|\b(?:what is|what's) (?:that|this|it) in\b|"
    rf"(?<![\w.]){_NUM}\b.*\b(?:perimeter|area|volume|average|total|sum|speed|costs?|price|"
    rf"percentage|discount|interest|profit)\b|\b(?:perimeter|area|volume|average|total|sum|speed|"
    rf"costs?|price|percentage|discount|interest|profit)\b.*(?<![\w.]){_NUM}\b", re.I)
_MATH_WORD = re.compile(r"\b(?:sum|total|average|mean|median|product|ratio|fraction|decimal|cost|"
                        r"price|pay|earn|discount|tip|tax|speed|distance|area|perimeter|volume|"
                        r"percentage|how many|how much|result|answer)\b", re.I)
_LEAD_FILLER = re.compile(r"^(?:(?:now|and|so|also|then|ok|okay|great|thanks|thank you|please|"
                          r"next|finally)[,!.:]?\s+)+", re.I)
_TRANSFORM = re.compile(
    r"^(?:can you |could you |would you |help me |i want you to |i need you to )?(?:summari[sz]e|"
    r"rewrite|rephrase|paraphrase|translate|fix|correct|proofread|edit|shorten|simplify|"
    r"convert (?!\d)|turn (?:this|these|it|that|those|your|them)|make (?:this|it|that|the|your|"
    r"them|these|those)|put (?:this|it|that|them|these)|reword|condense|extract|classify|"
    r"categori[sz]e|sort|reorder|retell|add|remove|delete|replace|change|update|expand|extend|"
    r"continue|adapt|modify|improve|polish)\b|\b(?:summary of|summari[sz]e (?:this|the following))\b",
    re.I)
_SHORT_FORM = re.compile(
    r"\b(?:poem|poems|haiku|limerick|verse|slogan|tagline|headline|title|motto|tweet|caption|"
    r"riddle|joke|name for|brand name|one word|single word|a word|only the|name only|"
    r"(?:the|a) (?:number|name|answer) only|just the|only (?:the|with)|yes or no)\b", re.I)
_WRITING = re.compile(
    r"^(?:can you |could you |would you |help me |i want you to |i need you to )?(?:write|compose|"
    r"draft|create|craft|pen|prepare|produce|generate)\b", re.I)
_FACT_LEAD = re.compile(
    r"^(?:who|whom|whose|when|where|which|"
    r"how (?:many|much|far|long|old|big|tall|high|deep|heavy|fast|large|wide)|"
    r"what (?:year|date|time|day|month|century|color|colour|country|city|language|currency)|"
    r"what(?:'s| is| was| are| were) (?:the )?(?:name|capital|population|boiling|freezing|"
    r"melting|speed|distance|height|length|weight|area|largest|smallest|biggest|tallest|"
    r"longest|highest|deepest|oldest|first|last|chemical symbol|symbol|formula|currency|"
    r"official language)|name (?:the|one))\b", re.I)
_YES_NO = re.compile(r"^(?:is|are|was|were|does|do|did|can|could|should|will|would|has|"
                     r"have)\b", re.I)
_EXPLAIN = re.compile(r"\b(?:why|explain|describe|difference|differences|compare|comparison|how "
                      r"(?:does|do|did|can|could|should|would|to)|(?:and|but|or) how|tips|ways|"
                      r"steps|advice|example|examples|pros|cons|tell me more|elaborate|discuss)\b",
                      re.I)
_BRIEF = re.compile(r"\b(?:brief|briefly|short|shorter|quick|quickly|concise|concisely|succinct|"
                    r"in short)\b", re.I)
_VERY_BRIEF = re.compile(r"\b(?:very short|very brief|tiny|a few sentences|a couple of sentences|"
                         r"in a nutshell|in a few words|tl;?dr)\b", re.I)
_LONG = re.compile(r"\b(?:in detail|detailed|thorough|thoroughly|comprehensive|in depth|"
                   r"in-depth|long|lengthy|elaborate)\b", re.I)
WORD_BANDS = {"default": (80, 200), "brief": (40, 200), "very_brief": (15, 120), "long": (150, 400)}


def instruction_part(user_text: str, max_chars: int = 400) -> str:
    """The part of a message that carries the request: its first paragraph (pasted material
    usually follows a blank line), whitespace-normalised, at most ``max_chars`` characters,
    without leading fillers such as "Now," or "Thanks!"."""
    first = re.split(r"\n[ \t]*\n", str(user_text).strip(), maxsplit=1)[0]
    head = " ".join(first.split())[:max_chars]
    return _LEAD_FILLER.sub("", head)


def _is_math(head: str) -> bool:
    if _ARITH.search(head):
        return True
    return len(_NUMBERS.findall(head)) >= 2 and bool(_MATH_WORD.search(head))


def prompt_type(user_text: str, history: Sequence[str] = ()) -> str:
    """``code``, ``math``, ``transform``, ``short``, ``factual``, ``writing`` or ``open`` (the
    module docstring's table) of the last user message; ``history``: the earlier user messages
    of the conversation (a conversation about code stays code; a follow-up with a number or a
    math word to a math question is math). Only :func:`instruction_part` is read, except that
    a code fence anywhere means code."""
    head = instruction_part(user_text)
    low = head.lower()
    if not low:
        return "open"
    past = [prompt_type(h) for h in history] if history else []
    if _CODE.search(head) or "```" in str(user_text) or "code" in past:
        return "code"
    if _WRITING.match(low):
        return "short" if _SHORT_FORM.search(low) else "writing"
    if _TRANSFORM.search(low):
        return "transform"
    if _is_math(head) or ("math" in past and (_NUMBERS.search(low) or _MATH_WORD.search(low))):
        return "math"
    if _SHORT_FORM.search(low):
        return "short"
    words = _WORD.findall(low)
    single = len(re.findall(r"[.!?](?:\s|$)", head.rstrip("?!. ") + " ")) <= 1
    if single and len(words) <= 25 and not _EXPLAIN.search(low):
        if _FACT_LEAD.match(low) or (_YES_NO.match(low) and low.endswith("?") and len(words) <= 15):
            return "factual"
    return "open"


def has_explicit_length(spec) -> bool:
    """True when ``spec`` asks for any count or a format that fixes the shape of the reply."""
    if spec is None:
        return False
    if any(getattr(spec, u, None) is not None for u in LENGTH_UNITS):
        return True
    if any(bool(getattr(spec, f, False)) for f in FORMAT_FLAGS):
        return True
    return any(bool(getattr(spec, f, None)) for f in FORMAT_VALUES)


def length_band(user_text: str, spec=None, kind: str | None = None,
                history: Sequence[str] = ()) -> Band | None:
    """The band of the module docstring (``None``: no band)."""
    if has_explicit_length(spec):
        return None
    kind = prompt_type(user_text, history) if kind is None else kind
    if kind in ("code", "math", "transform", "short"):
        return None
    if kind == "factual":
        return Band("factual", words=(1, 90), sentences=(1, 3))
    low = instruction_part(user_text).lower()
    if _VERY_BRIEF.search(low):
        mod = "very_brief"
    elif _LONG.search(low):
        mod = "long"
    elif _BRIEF.search(low):
        mod = "brief"
    else:
        mod = "default"
    return Band(kind, words=WORD_BANDS[mod])


# ================================================================================ checks
def special_leak(text: str) -> bool:
    """A special-token string (ours or a common chat template's) is visible."""
    return any(s in text for s in _LEAK_STRINGS)


_END_CHARS = frozenset(".!?…\"'”’»)]}*_`:;>")
_NOT_PROSE_LINE = re.compile(r"(?:[-*+•][ \t]|\d+[.)][ \t]|#{1,6}[ \t]|\||```|>)")
CUT_OFF_MIN_WORDS = 6


def cut_off(text: str) -> bool:
    """The reply stops in the middle of a prose sentence: its last line holds at least
    ``CUT_OFF_MIN_WORDS`` words, is not a list item (a marker followed by a space or tab), a
    heading, a table row, a quote or a fence, and ends without a closing character
    (``. ! ? … " ' ) ] } * _ ` : ;`` and closing quotes). Short last lines (sign-offs, names)
    never count; a reply cut after a few words of its last line is caught only by
    ``at_budget`` (:func:`check`)."""
    t = text.rstrip()
    if not t or t[-1] in _END_CHARS:
        return False
    last = t.split("\n")[-1].strip()
    if _NOT_PROSE_LINE.match(last):
        return False
    return len(re.findall(r"\w+", last)) >= CUT_OFF_MIN_WORDS


def degenerate_flags(text: str, *, code: bool = False, prose: bool = True,
                     rep4_max: float = REP4_MAX) -> list[str]:
    """Names of the degenerate-output detectors that fire on a finished reply (``code``:
    the prompt is about code; ``prose``: the reply is expected to be prose, so ``rep4`` and
    ``cut_off`` apply)."""
    from bes.eval.exact_score import degenerate_stats

    d = degenerate_stats(text, "stop")
    out = [name for name in ("empty", "loop", "repeated_lines", "unclosed_fence", "role_leak",
                             "refusal") if d.get(name)]
    if prose and not code and d["rep4"] > rep4_max:
        out.append("rep4")
    if prose and not code and cut_off(text):
        out.append("cut_off")
    if special_leak(text) and "role_leak" not in out:
        out.append("special_leak")
    return out


@dataclass(frozen=True)
class Check:
    """Outcome of the failure checks of one candidate."""

    finished: bool
    violations: tuple[str, ...]
    degenerate: tuple[str, ...]
    in_band: bool

    @property
    def reasons(self) -> list[str]:
        out = [] if self.finished else ["unfinished"]
        out += [f"violation: {v}" for v in self.violations]
        out += [f"degenerate: {d}" for d in self.degenerate]
        if not self.in_band:
            out.append("band")
        return out


def check(c: Candidate, spec=None, band: Band | None = None, *, code: bool = False,
          prose: bool = True, rep4_max: float = REP4_MAX, budget: int | None = None) -> Check:
    """Run every failure check on one candidate (module docstring). ``budget``: the token
    budget of the particles; a finished candidate with ``n_tokens >= budget`` is degenerate
    (``at_budget``): it ended because the budget-aware potentials left nothing else."""
    from bes.exact import constraints

    text = c.text
    measured = constraints.measure(text)
    violations = tuple(constraints.verify(spec, text, measured)) if spec is not None else ()
    flags = degenerate_flags(text, code=code, prose=prose, rep4_max=rep4_max)
    if budget is not None and c.finished and c.n_tokens >= budget:
        flags.append("at_budget")
    degenerate = tuple(flags)
    in_band = band is None or band.contains(text, measured)
    return Check(c.finished, violations, degenerate, in_band)


# ============================================================================= utilities
def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _f1(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    overlap = sum((a & b).values())
    return 2.0 * overlap / (na + nb)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def utility(a: str, b: str, kind: str = "f12") -> float:
    """Lexical agreement of two texts in ``[0, 1]`` (symmetric; 1 for identical word
    sequences): ``"f1"`` unigram F1, ``"f12"`` mean of unigram and bigram F1, ``"lcs"``
    F-measure of the longest common word subsequence. Words: ``\\w+`` runs, lowercased."""
    if kind not in UTILITIES:
        raise ValueError(f"unknown utility {kind!r}")
    wa, wb = _words(a), _words(b)
    if kind == "lcs":
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        n = _lcs_len(wa, wb)
        return 2.0 * n / (len(wa) + len(wb))
    u1 = _f1(Counter(wa), Counter(wb))
    if kind == "f1":
        return u1
    u2 = _f1(Counter(zip(wa, wa[1:])), Counter(zip(wb, wb[1:])))
    return 0.5 * (u1 + u2)


# ============================================================================ consensus
def _norm_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray([max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in weights])
    s = float(w.sum())
    return np.full(len(w), 1.0 / len(w)) if s <= 0 else w / s


def majority_answer(texts: Sequence[str], weights: Sequence[float],
                    context: Sequence[str] = ()) -> tuple[Fraction | None, list[int]]:
    """Weighted majority of the final numbers (``exact_score.extract_answer`` with the numbers
    of ``context`` as context): ``(value, members)``; ``(None, all)`` when no text states a
    number. Ties: larger total weight, then more members, then the earliest member."""
    from bes.eval.exact_score import context_values, extract_answer

    ctx = context_values(list(context)) if context else None
    w = _norm_weights(weights)
    groups: dict[Fraction, list[int]] = {}
    for i, t in enumerate(texts):
        v = extract_answer(t, ctx)["value"]
        if v is not None:
            groups.setdefault(v, []).append(i)
    if not groups:
        return None, list(range(len(texts)))
    best = max(groups.items(), key=lambda kv: (round(float(sum(w[i] for i in kv[1])), 12),
                                                len(kv[1]), -kv[1][0]))
    return best[0], best[1]


def consensus(texts: Sequence[str], weights: Sequence[float], kind: str = "f12"
              ) -> tuple[int, list[float]]:
    """Minimum-Bayes-risk choice: ``(position, scores)`` with ``scores[i] = sum_j w_j
    U(i, j)`` (normalised weights, ``U(i, i) = 1``). Ties within ``TIE``: larger weight, then
    the lower position."""
    n = len(texts)
    if n == 0:
        raise ValueError("no texts")
    w = _norm_weights(weights)
    u = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u[i, j] = u[j, i] = utility(texts[i], texts[j], kind)
    scores = [float(sum(w[j] * u[i, j] for j in range(n))) for i in range(n)]
    best = 0
    for i in range(1, n):
        d = scores[i] - scores[best]
        if d > TIE or (abs(d) <= TIE and w[i] > w[best] + TIE):
            best = i
    return best, scores


# ============================================================================== select
@dataclass
class Selection:
    """Result of :func:`select`: the chosen candidate (``index``, ``text``), the level that
    fired (``rule``), the pool it was chosen from, the consensus scores of that pool, the
    majority answer (math), the band and prompt type, and the failure reasons of every
    candidate."""

    index: int
    text: str
    rule: str
    pool: list[int] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    answer: str | None = None
    band: Band | None = None
    prompt_type: str = "open"
    reasons: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_survivors(self) -> int:
        """Candidates that passed every check (the ``strict`` pool)."""
        return sum(1 for r in self.reasons.values() if not r)

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "rule": self.rule, "pool": self.pool,
                "scores": {str(k): round(v, 6) for k, v in self.scores.items()},
                "answer": self.answer, "band": None if self.band is None else str(self.band),
                "prompt_type": self.prompt_type, "n_survivors": self.n_survivors,
                "reasons": {str(k): v for k, v in self.reasons.items()}}


def _fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{float(v):.10g}"


def select(cands: Sequence[Candidate], spec=None, user_text: str = "", *,
           history: Sequence[str] = (), context: Sequence[str] | None = None,
           band: Band | None | str = "auto", kind: str | None = None,
           utility_kind: str = "f12", rep4_max: float = REP4_MAX,
           budget: int | None = None) -> Selection:
    """Choose one candidate (module docstring).

    spec: the ``ConstraintSpec`` of the last user message (``None``: no constraint);
    user_text: that message (prompt type and band); history: the earlier user messages
    (prompt type); context: texts whose numbers are context for the math answer extraction
    (default: ``history + [user_text]``, as the scorer's context); band: ``"auto"``
    (:func:`length_band`), a :class:`Band` or ``None``; kind: prompt type override; budget:
    the particles' token budget (``max_new_tokens``) for the ``at_budget`` check.
    """
    if not cands:
        raise ValueError("no candidates")
    kind = prompt_type(user_text, history) if kind is None else kind
    if band == "auto":
        band = length_band(user_text, spec, kind)
    checks = [check(c, spec, band, code=kind == "code", prose=kind != "short", rep4_max=rep4_max,
                    budget=budget) for c in cands]
    reasons = {i: ch.reasons for i, ch in enumerate(checks)}
    ctx = list(context) if context is not None else \
        [t for t in (*history, user_text) if t]
    tests = (("strict", lambda ch: ch.finished and not ch.violations and not ch.degenerate
              and ch.in_band),
             ("relax_band", lambda ch: ch.finished and not ch.violations and not ch.degenerate),
             ("relax_degenerate", lambda ch: ch.finished and not ch.violations))
    for rule, ok in tests:
        pool = [i for i, ch in enumerate(checks) if ok(ch)]
        if not pool:
            continue
        answer = None
        if kind == "math":
            value, members = majority_answer([cands[i].text for i in pool],
                                             [cands[i].weight for i in pool], ctx)
            if value is not None:
                answer = _fmt(value)
                pool = [pool[m] for m in members]
        pos, scores = consensus([cands[i].text for i in pool], [cands[i].weight for i in pool],
                                utility_kind)
        idx = pool[pos]
        return Selection(idx, cands[idx].text, rule, pool, dict(zip(pool, scores)), answer,
                         band, kind, reasons)
    finished = [i for i, ch in enumerate(checks) if ch.finished]
    if finished:
        idx = min(finished, key=lambda i: (len(checks[i].violations), -float(cands[i].weight),
                                           len(checks[i].degenerate), i))
        return Selection(idx, cands[idx].text, "best_finished", finished, {}, None, band, kind,
                         reasons)
    order = {"length": 0, "dead": 1}
    idx = min(range(len(cands)), key=lambda i: (order.get(cands[i].finish, 2),
                                                len(checks[i].violations), -cands[i].n_tokens, i))
    return Selection(idx, cands[idx].text, "unfinished", list(range(len(cands))), {}, None, band,
                     kind, reasons)


__all__ = ["Candidate", "Band", "Check", "Selection", "select", "check", "consensus",
           "majority_answer", "utility", "prompt_type", "length_band", "has_explicit_length",
           "degenerate_flags", "special_leak", "cut_off", "LEVELS", "UTILITIES", "WORD_BANDS"]
