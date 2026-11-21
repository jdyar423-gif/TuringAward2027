"""Hindsight relabelling: verifiable constraints added to existing prompts.

:func:`relabel` takes a conversation whose last message is an assistant response, measures that
response with ``bes.exact.constraints.measure`` and appends to the last user message a natural
instruction (``constraints.render``) that the existing response satisfies by construction. The
result is re-compiled and verified before it is returned, so every relabelled sample carries a
correct, machine-checkable instruction (and the countdown registers built from it reach zero
exactly where the response ends).

Sampling (``rng`` is a ``numpy.random.Generator``; every choice below is drawn from it):

1. Eligible constraint types and their weights, from the response's measured properties
   (types already constrained by the user message are skipped):

   ===============  ======  =================================================================
   type             weight  eligible when
   ===============  ======  =================================================================
   sentences        1.0     prose (no code fence, list item, heading or JSON), 1-20 sentences,
                            at least 6 words for a single sentence
   words            1.0     at least 5 words, no code fence, not JSON
   items            1.2     2-12 list items; or 0.2 for "no bullet points" when the response
                            is prose of at least 3 sentences and 40 words
   paragraphs       0.8     2-8 paragraphs, no code fence; or 0.25 for "a single paragraph"
                            when the response is one line of at least 3 sentences and 40 words
                            (short one-line answers never get it)
   lines            0.4     2-16 non-blank lines of at most 100 characters, no code fence
   keywords         0.4     content words (4-20 letters, not stop words) occur in the response
   case             3.0     the response is all lowercase (at least 3 words) or all uppercase
   end_question     1.0     the response ends with a question
   json_output      3.0     the response is a JSON object or array
   code_block       0.6     the response has a closed code block
   no_commas        0.3     no comma in a response of at least 15 words
   ===============  ======  =================================================================

   Further instruction families (each only when the response already has the property, so
   exotic formats stay as rare as they are in the data; none in JSON responses, and the ones
   about text not in responses with code fences):

   ===============  ======  =================================================================
   type             weight  eligible when
   ===============  ======  =================================================================
   list_style       0.5     2 or more list items, all bullets or all numbered (an items count
                            drawn alone also names the style with probability 0.5)
   frequencies      0.25    a content word occurs (whole word) at least twice in a response
                            of at least 20 words: at least k (k >= 2), exactly, or at most
   letters          0.05    a common letter occurs at least 3 times (at least / less than)
   capital_words    0.1     at least 2 all-capital words (at least / at most)
   first_word       0.2     2-8 paragraphs, a later one opens with a content word
   divider          2.0     *** dividers between 2-8 paragraphs (not response separators)
   sections         1.5     section lines "<Marker> 1 .. n" (Section, Part, Step, Day, ...)
   end_phrase       0.15    a last sentence of 2-8 plain words
   starts_with      0.12    the first word (not a filler opener such as "Sure"); 1.0 for
                            "Yes" / "No" openers (options "Yes" or "No")
   highlights       0.5     at least one *highlighted* span (at least k, sometimes exactly)
   placeholders     1.0     at least one [placeholder]
   title            2.0     a <<title>>
   postscript       1.5     a line starting with P.S. or P.P.S (the marker as written)
   quoted           1.0     the whole response in double quotes
   one_of           3.0     the whole response is yes / no / maybe / true / false / ...
   responses        2.0     2-4 different parts between separator lines (not ***, ---, ___)
   repeat           3.0     the response starts with the user message (drawn alone)
   ===============  ======  =================================================================

2. One type (probability 0.6) or two (0.4) are drawn without replacement by weight.
3. Counts: the relation is drawn with weights that depend on the measured count ``n``:
   ``exactly n`` (sentences: 3 if ``n <= 5``, 1 if ``n <= 10``; items and paragraphs: 3; lines:
   2; words: 0.3 and only when ``n <= 12``), ``at most`` a bound ``>= n`` (2; words round up to
   a multiple of 5/10/25/50/100), ``at least`` a bound ``<= n`` (1-1.5), ``between`` (0.5-1).
   Item instructions name the list style only when ``list_style`` is drawn (the instruction
   must compile back to the drawn constraints).
4. Keywords: 1 (0.5), 2 (0.3) or 3 (0.2) distinct words of the response, preferably (0.7)
   words that the user message does not contain.
5. The instruction is appended after one space (0.6), a blank line (0.3) or a newline (0.1);
   after a newline only when the user message does not end with ``. ! ? :`` or a closing
   quote or bracket. A ``repeat`` instruction is one of ``constraints.REPEAT_INSTRUCTIONS``;
   the request to repeat is then the original user message.

Up to ``attempts`` draws are tried; the first whose re-compiled user message equals the merge of
the original constraints with the drawn ones, and whose response passes ``verify``, is returned.
``None`` means: no assistant response to relabel, the existing request is already violated by
the response, or no draw verified.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bes.exact import constraints as C
from bes.exact.constraints import ConstraintSpec, Interval

FIELD_WEIGHTS = {"sentences": 1.0, "words": 1.0, "items": 1.2, "paragraphs": 0.8, "lines": 0.4,
                 "keywords": 0.4, "case": 3.0, "end_question": 1.0, "json_output": 3.0,
                 "code_block": 0.6, "no_commas": 0.3,
                 # further instruction families (eligible only when the response has the property)
                 "list_style": 0.5, "frequencies": 0.25, "letters": 0.05, "highlights": 0.5,
                 "placeholders": 1.0, "capital_words": 0.1, "sections": 1.5, "first_word": 0.2,
                 "divider": 2.0, "title": 2.0, "postscript": 1.5, "quoted": 1.0, "starts_with": 0.12,
                 "end_phrase": 0.15, "one_of": 3.0, "responses": 2.0, "repeat": 3.0}
NO_LIST_WEIGHT = 0.2
SINGLE_PARAGRAPH_WEIGHT = 0.25
YES_NO_START_WEIGHT = 1.0
P_TWO_CONSTRAINTS = 0.4
P_STYLE_WITH_ITEMS = 0.5
#: Lines that open a section ("Section 2", "## Part 3: ...") use one of these markers.
SECTION_MARKERS = ("Section", "SECTION", "Part", "PART", "Chapter", "Step", "Day", "Stage", "Phase",
                   "Week", "Lesson", "Module")
#: Openers that are not asked for (the "start with" instruction would teach filler).
FILLER_OPENERS = frozenset(
    "sure certainly absolutely great okay ok of here hello hi hey well so yes no thanks thank "
    "i the a an this that it in to as for on there".split())

_STOP = frozenset(
    "about above after again against also although always among another anyone anything around "
    "because been before being below between both cannot could does doing down during each "
    "either else enough even ever every from further have having here hers herself himself "
    "however into itself just least less like made make makes many might more most much must "
    "myself never only other others ours ourselves over same should since some something "
    "such than that their theirs them themselves then there therefore these they this those "
    "though through thus together under until upon very well were what whatever when where "
    "whether which while whom whose will with within without would your yours yourself "
    "yourselves often usually also really quite rather perhaps maybe almost already still "
    "across along away back done each even ever first last next once part per several "
    "thing things time times used uses using want wants year years like likely help helps "
    "include includes including example examples answer response question".split())
_WORD = re.compile(r"(?<![\w'-])[A-Za-z][a-z]{3,19}(?![\w'-])")
_ALNUM_RUN = re.compile(r"[^\W_]+")


# --------------------------------------------------------------------------- helpers
def response_text(content: Any, tok=None) -> str | None:
    """Display text of assistant content (a string, or ``bes.data.chat_format`` parts with
    calculator spans shown as ``expr = result``); ``None`` when the content is invalid."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return None
    from bes.data import chat_format as cf

    if tok is None:
        from bes.tokenizer.tok import load

        tok = load()
    try:
        ids, _ = cf.render_turn("assistant", list(content), tok)
    except (ValueError, TypeError, AttributeError):
        return None
    return cf.display_text(ids[1:-1], tok)


def _weighted(rng: np.random.Generator, options: Sequence[tuple[float, Any]]) -> Any:
    weights = np.asarray([w for w, _ in options], dtype=np.float64)
    if weights.sum() <= 0:
        return None
    return options[int(rng.choice(len(options), p=weights / weights.sum()))][1]


def _round_up(n: int, rng: np.random.Generator) -> int:
    """A natural upper bound ``>= n``: ``n`` itself or ``n`` rounded up to 5/10/25/50/100."""
    options = {n}
    for step in (5, 10, 25, 50, 100):
        if step <= max(5, n):
            options.add(-(-n // step) * step)
    return int(rng.choice(sorted(options)))


def _round_down(n: int, rng: np.random.Generator) -> int:
    """A natural lower bound in ``[1, n]``: ``n`` or ``n`` rounded down to 5/10/25/50/100."""
    options = {n}
    for step in (5, 10, 25, 50, 100):
        v = n // step * step
        if step <= n and v >= 1:
            options.add(v)
    return int(rng.choice(sorted(options)))


def _small_slack(rng: np.random.Generator) -> int:
    return int(rng.choice(3, p=[0.5, 0.3, 0.2]))


def _count_interval(unit: str, n: int, rng: np.random.Generator) -> Interval | None:
    """A request on a count of ``unit`` that ``n`` satisfies (relation weights in the module
    docstring)."""
    if unit == "words":
        options = [(2.0, "at_most"), (1.5, "at_least"), (1.0, "between")]
        if n <= 12:
            options.append((0.3, "exactly"))
    elif unit == "sentences":
        options = [(3.0 if n <= 5 else 1.0 if n <= 10 else 0.0, "exactly"), (2.0, "at_most"),
                   (1.5 if n >= 2 else 0.0, "at_least"), (1.0 if n >= 2 else 0.0, "between")]
    elif unit == "lines":
        options = [(2.0, "exactly"), (1.0, "at_most")]
    else:                                   # items, paragraphs
        options = [(3.0, "exactly"), (1.0, "at_most"), (1.0, "at_least"), (0.5, "between")]
    kind = _weighted(rng, options)
    if kind == "exactly":
        return Interval(n, n)
    if unit == "words":
        lo, hi = _round_down(n, rng), _round_up(n, rng)
    else:
        lo, hi = max(1, n - _small_slack(rng)), n + _small_slack(rng)
    if kind == "at_most":
        return Interval(0, hi)
    if kind == "at_least":
        return Interval(lo, None) if lo >= 1 else None
    if kind == "between":
        if hi == lo:
            hi = lo + 1 + _small_slack(rng)
        return Interval(lo, hi) if lo < hi else None
    return None


def _keyword_pool(text: str, user_text: str, taken: Sequence[str]) -> tuple[list[str], list[str]]:
    """Candidate keywords (first spelling in the response): ``(absent from the user message,
    present in it)``."""
    seen: dict[str, str] = {}
    for m in _WORD.finditer(text):
        w = m.group(0)
        low = w.lower()
        if low not in _STOP and low not in seen:
            seen[low] = w
    taken_l = {k.lower() for k in taken}
    user_words = set(_ALNUM_RUN.findall(user_text.lower()))
    fresh, known = [], []
    for low, w in seen.items():
        if low in taken_l or not C.valid_keyword(w):
            continue
        (known if low in user_words else fresh).append(w)
    return fresh, known


def _short_lines(text: str, limit: int = 100) -> bool:
    return all(len(line) <= limit for line in text.split("\n") if line.strip())


def _lower_bound(n: int, rng: np.random.Generator) -> int:
    """A lower bound in ``[1, n]`` for small counts: ``n``, ``n - 1`` or ``n - 2``."""
    return max(1, n - _small_slack(rng))


def _feature_interval(n: int, rng: np.random.Generator, exact_weight: float = 0.3) -> Interval:
    """"at least k <= n" mostly, sometimes exactly ``n`` (highlights, placeholders, ...)."""
    if rng.random() < exact_weight:
        return Interval(n, n)
    return Interval(_lower_bound(n, rng), None)


def _frequency_candidate(text: str, user_text: str, user_spec: ConstraintSpec,
                         rng: np.random.Generator) -> tuple[str, Interval] | None:
    """A content word of the response that occurs (whole word) at least twice, with a
    frequency request it satisfies."""
    taken = {w.lower() for w, _ in user_spec.frequencies} | {k.lower() for k in user_spec.forbidden}
    counts: dict[str, tuple[str, int]] = {}
    for m in _WORD.finditer(text):
        w = m.group(0)
        low = w.lower()
        if low in _STOP or low in taken or low in counts or not C.valid_keyword(w):
            continue
        n = C.count_keyword(text, w, whole=True)
        if n >= 2:
            counts[low] = (w, n)
    if not counts:
        return None
    word, n = list(counts.values())[int(rng.integers(len(counts)))]
    kind = _weighted(rng, [(1.0, "at_least"), (0.4, "exactly"), (0.3, "at_most")])
    if kind == "exactly":
        return word, Interval(n, n)
    if kind == "at_most":
        return word, Interval(0, n + _small_slack(rng))
    lo = _lower_bound(n, rng)
    return word, Interval(max(2, lo), None)


def _letter_candidate(text: str, user_spec: ConstraintSpec, rng: np.random.Generator):
    taken = {ch for ch, _ in user_spec.letters}
    letters = [ch for ch in "etaoinsrhldcumfpgwybvk" if ch not in taken]
    ch = letters[int(rng.integers(len(letters)))]
    n = C.letter_count(text, ch)
    if n < 3:
        return None
    if rng.random() < 0.7:
        return ch, Interval(_round_down(n, rng), None)
    return ch, Interval(0, _round_up(n + 1, rng) - 1)      # "less than N times", N > n


_PS_LINE = re.compile(r"^[ \t]*(?:[*_]+[ \t]*)?(P\.[ \t]?P\.[ \t]?S\.?|P\.[ \t]?S\.)", re.IGNORECASE | re.MULTILINE)
_SECTION_LINE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,2})?([A-Za-z]+)[ \t]*([0-9]+)(?![0-9])", re.MULTILINE)
_SEPARATOR_LINE = re.compile(r"^[ \t]*([*\-=_~#|+]{3,20})[ \t]*$", re.MULTILINE)
#: Lines that are markdown dividers or rules, never response separators.
MARKDOWN_RULES = frozenset(("***", "---", "___"))
_CLOSED_ANSWERS = (("yes", "no"), ("yes", "no", "maybe"), ("true", "false"), ("agree", "disagree"))


def _section_candidate(text: str) -> tuple[str, Interval] | None:
    """``(marker, exactly n)`` when the response's section lines are ``<marker> 1 .. n``."""
    found = _SECTION_LINE.findall(text)
    for marker in SECTION_MARKERS:
        nums = [int(n) for w, n in found if w == marker]
        if len(nums) >= 2 and nums == list(range(1, len(nums) + 1)) \
                and C.count_sections(text, marker) == len(nums):
            return marker, Interval(len(nums), len(nums))
    return None


def _end_phrase_candidate(text: str) -> str | None:
    """The last sentence of the response when it is short (2-8 words) and plain."""
    tail = text.rstrip()
    m = re.search(r"(?:^|(?<=[.!?])\s+|\n)([A-Z][^.!?\n]{2,80}[.!?])$", tail)
    if not m:
        return None
    phrase = " ".join(m.group(1).split())
    if not 2 <= len(phrase.split()) <= 8 or not C.valid_phrase(phrase) or not C.ends_with_phrase(text, phrase):
        return None
    if any(ch in phrase for ch in "*_[]()<>`#"):
        return None
    return phrase


def _start_candidate(text: str, m: Mapping[str, Any], rng: np.random.Generator):
    """``(options, weight)`` for a "start with" request: "Yes" / "No" for an answer that
    starts with one of them, "Dear" for a letter, or the first word of the response."""
    first = m["first_word"]
    head = text.lstrip(C._LEADING_MARKUP)
    word = re.match(r"[A-Za-z][a-z]+", head)
    if first in ("yes", "no") and word and word.group(0).lower() == first:
        opts = ("Yes", "No") if word.group(0)[0].isupper() else ("yes", "no")
        return opts, YES_NO_START_WEIGHT
    if not word or len(word.group(0)) < 3 or word.group(0).lower() in FILLER_OPENERS:
        return None
    w = word.group(0)
    if not C.starts_with_phrase(text, w):
        return None
    return (w,), (0.5 if w == "Dear" else 1.0) * FIELD_WEIGHTS["starts_with"]


def _first_word_candidate(m: Mapping[str, Any], rng: np.random.Generator) -> tuple[int, str] | None:
    firsts = m["paragraph_first_words"]
    options = [(i + 1, w) for i, w in enumerate(firsts) if i >= 1 and w.isalpha() and len(w) >= 3
               and w not in _STOP and w not in FILLER_OPENERS]
    if len(firsts) < 2 or not options:
        return None
    return options[int(rng.integers(len(options)))]


def _responses_candidate(text: str) -> tuple[str, int] | None:
    seps = {s for s in _SEPARATOR_LINE.findall(text)
            if C._SEPARATOR_OK.fullmatch(s) and s not in MARKDOWN_RULES}
    for sep in sorted(seps):
        parts, clean = C.split_responses(text, sep)
        if clean and 2 <= len(parts) <= 4 and len(set(parts)) == len(parts) and all(len(p.split()) >= 3 for p in parts):
            # the separator must not also be a paragraph divider or a markdown rule elsewhere
            if text.count(sep) == len(parts) - 1:
                return sep, len(parts)
    return None


def candidates(m: Mapping[str, Any], text: str, user_text: str, user_spec: ConstraintSpec,
               rng: np.random.Generator) -> list[tuple[float, str, Any]]:
    """Eligible ``(weight, field, value)`` constraints that ``text`` satisfies (one value drawn
    per field), following the module docstring."""
    out: list[tuple[float, str, Any]] = []
    prose = (m["code_fences"] == 0 and m["items"] == 0 and m["headings"] == 0
             and not m["json_valid"])
    s, w = m["sentences"], m["words"]
    if user_spec.sentences is None and prose and 1 <= s <= 20 and (s > 1 or w >= 6):
        iv = _count_interval("sentences", s, rng)
        if iv is not None:
            out.append((FIELD_WEIGHTS["sentences"], "sentences", iv))
    if user_spec.words is None and w >= 5 and m["code_fences"] == 0 and not m["json_valid"]:
        iv = _count_interval("words", w, rng)
        if iv is not None:
            out.append((FIELD_WEIGHTS["words"], "words", iv))
    n = m["items"]
    if user_spec.items is None and not m["json_valid"]:
        if 2 <= n <= 12:
            iv = _count_interval("items", n, rng)
            if iv is not None:
                out.append((FIELD_WEIGHTS["items"], "items", iv))
        elif n == 0 and prose and s >= 3 and w >= 40:
            out.append((NO_LIST_WEIGHT, "items", Interval(0, 0)))
    p = m["paragraphs"]
    if user_spec.paragraphs is None and m["code_fences"] == 0 and not m["json_valid"]:
        if 2 <= p <= 8:
            iv = _count_interval("paragraphs", p, rng)
            if iv is not None:
                out.append((FIELD_WEIGHTS["paragraphs"], "paragraphs", iv))
        elif p == 1 and m["lines"] == 1 and s >= 3 and w >= 40:
            out.append((SINGLE_PARAGRAPH_WEIGHT, "paragraphs", Interval(1, 1)))
    ln = m["lines"]
    if (user_spec.lines is None and 2 <= ln <= 16 and m["code_fences"] == 0
            and not m["json_valid"] and _short_lines(text)):
        iv = _count_interval("lines", ln, rng)
        if iv is not None:
            out.append((FIELD_WEIGHTS["lines"], "lines", iv))
    fresh, known = _keyword_pool(text, user_text, user_spec.keywords + user_spec.forbidden)
    if fresh or known:
        k = int(rng.choice(3, p=[0.5, 0.3, 0.2])) + 1
        picked: list[str] = []
        for _ in range(k):
            pool = fresh if fresh and (not known or rng.random() < 0.7) else known
            if not pool:
                break
            picked.append(pool.pop(int(rng.integers(len(pool)))))
        if picked:
            out.append((FIELD_WEIGHTS["keywords"], "keywords", tuple(picked)))
    if user_spec.case is None:
        if m["all_lower"] and w >= 3:
            out.append((FIELD_WEIGHTS["case"], "case", "lower"))
        elif m["all_upper"] and w >= 2:
            out.append((FIELD_WEIGHTS["case"], "case", "upper"))
    if not user_spec.end_question and m["ends_with_question"]:
        out.append((FIELD_WEIGHTS["end_question"], "end_question", True))
    if not user_spec.json_output and m["json_valid"]:
        out.append((FIELD_WEIGHTS["json_output"], "json_output", True))
    if not user_spec.code_block and m["code_blocks"] >= 1:
        out.append((FIELD_WEIGHTS["code_block"], "code_block", True))
    if not user_spec.no_commas and m["commas"] == 0 and w >= 15:
        out.append((FIELD_WEIGHTS["no_commas"], "no_commas", True))
    out += _further_candidates(m, text, user_text, user_spec, rng)
    return out


def _further_candidates(m: Mapping[str, Any], text: str, user_text: str, user_spec: ConstraintSpec,
                        rng: np.random.Generator) -> list[tuple[float, str, Any]]:
    """Candidates of the further instruction families (module docstring)."""
    out: list[tuple[float, str, Any]] = []
    style = _item_style(m)
    if not user_spec.list_style and style and m["items"] >= 2 and not m["json_valid"] \
            and (user_spec.items is None or user_spec.items.hi != 0):
        out.append((FIELD_WEIGHTS["list_style"], "list_style", style))
    if m["code_fences"] == 0 and not m["json_valid"]:
        if not user_spec.frequencies and m["words"] >= 20:
            fc = _frequency_candidate(text, user_text, user_spec, rng)
            if fc is not None:
                out.append((FIELD_WEIGHTS["frequencies"], "frequencies", (fc,)))
        if not user_spec.letters and m["words"] >= 10:
            lc = _letter_candidate(text, user_spec, rng)
            if lc is not None:
                out.append((FIELD_WEIGHTS["letters"], "letters", (lc,)))
        if user_spec.capital_words is None and m["capital_words"] >= 2:
            n = m["capital_words"]
            iv = (Interval(_lower_bound(n, rng), None) if rng.random() < 0.7
                  else Interval(0, n + _small_slack(rng)))
            out.append((FIELD_WEIGHTS["capital_words"], "capital_words", iv))
        p = m["paragraphs"]
        if user_spec.first_word is None and 2 <= p <= 8:
            fw = _first_word_candidate(m, rng)
            if fw is not None:
                out.append((FIELD_WEIGHTS["first_word"], "first_word", fw))
        if not user_spec.divider and m["dividers"] >= 1 and 2 <= p <= 8 and _responses_candidate(text) is None:
            out.append((FIELD_WEIGHTS["divider"], "divider", True))
        if user_spec.sections is None:
            sc = _section_candidate(text)
            if sc is not None:
                out.append((FIELD_WEIGHTS["sections"], "sections", sc))
        if not user_spec.end_phrase:
            ep = _end_phrase_candidate(text)
            if ep is not None:
                out.append((FIELD_WEIGHTS["end_phrase"], "end_phrase", ep))
        if not user_spec.starts_with:
            sw = _start_candidate(text, m, rng)
            if sw is not None:
                out.append((sw[1], "starts_with", sw[0]))
    if user_spec.highlights is None and m["highlights"] >= 1 and not m["json_valid"]:
        out.append((FIELD_WEIGHTS["highlights"], "highlights", _feature_interval(m["highlights"], rng)))
    if user_spec.placeholders is None and m["placeholders"] >= 1 and not m["json_valid"] and m["code_fences"] == 0:
        out.append((FIELD_WEIGHTS["placeholders"], "placeholders", _feature_interval(m["placeholders"], rng)))
    if not user_spec.title and m["has_title"]:
        out.append((FIELD_WEIGHTS["title"], "title", True))
    if not user_spec.postscript:
        ps = _PS_LINE.search(text)
        if ps is not None:
            marker = " ".join(ps.group(1).split())
            if C.valid_phrase(marker, 80) and C.has_postscript(text, marker):
                out.append((FIELD_WEIGHTS["postscript"], "postscript", marker))
    if not user_spec.quoted and m["quoted"] and m["words"] >= 2:
        out.append((FIELD_WEIGHTS["quoted"], "quoted", True))
    if not user_spec.one_of:
        key = C.option_key(text)
        for opts in _CLOSED_ANSWERS:
            if key in opts:
                pick = opts if rng.random() < 0.7 else next((o for o in _CLOSED_ANSWERS if key in o and o != opts), opts)
                out.append((FIELD_WEIGHTS["one_of"], "one_of", pick))
                break
    if user_spec.responses is None:
        rs = _responses_candidate(text)
        if rs is not None:
            out.append((FIELD_WEIGHTS["responses"], "responses", rs))
    request = user_text.strip()
    if (user_spec.repeat is None and request and len(user_text) <= C.SCAN_CHARS - 400
            and len(request) <= C.MAX_REPEAT and C.repeats_request(text, request)):
        out.append((FIELD_WEIGHTS["repeat"], "repeat", request))
    return out


def _item_style(m: Mapping[str, Any]) -> str | None:
    if m["items"] and m["numbered_items"] == m["items"]:
        return "numbered"
    if m["items"] and m["bullet_items"] == m["items"]:
        return "bullet"
    return None


def append_instruction(user_text: str, instruction: str, rng: np.random.Generator) -> str:
    """``user_text`` followed by ``instruction`` (separators in the module docstring)."""
    head = user_text.rstrip()
    if not head:
        return instruction
    ends = head[-1] in ".!?:\"')]”’"
    sep = _weighted(rng, [(0.6 if ends else 0.0, " "), (0.3, "\n\n"), (0.1, "\n")])
    return head + sep + instruction


# --------------------------------------------------------------------------- relabel
def relabel_with_spec(messages: Sequence[Mapping[str, Any]], rng: np.random.Generator, tok=None,
                      attempts: int = 4) -> tuple[list[dict], ConstraintSpec] | None:
    """:func:`relabel`, also returning the added constraints."""
    if not messages or not all(isinstance(x, Mapping) for x in messages):
        return None
    if messages[-1].get("role") != "assistant" or len(messages) < 2:
        return None
    user = messages[-2]
    if user.get("role") != "user" or not isinstance(user.get("content"), str):
        return None
    text = response_text(messages[-1].get("content"), tok)
    if not text or not text.strip():
        return None
    user_text = user["content"]
    user_spec = C.compile(user_text)
    m = C.measure(text)
    if C.verify(user_spec, text, m):
        return None                          # the response already breaks the request
    for _ in range(attempts):
        cands = candidates(m, text, user_text, user_spec, rng)
        if not cands:
            return None
        k = 2 if len(cands) >= 2 and rng.random() < P_TWO_CONSTRAINTS else 1
        chosen: dict[str, Any] = {}
        pool = list(cands)
        for _ in range(k):
            weights = np.asarray([wt for wt, _, _ in pool], dtype=np.float64)
            i = int(rng.choice(len(pool), p=weights / weights.sum()))
            _, field, value = pool.pop(i)
            chosen[field] = value
        if "repeat" in chosen:
            chosen = {"repeat": chosen["repeat"]}     # the request must be the user message alone
        style = _item_style(m)
        items = chosen.get("items")
        if (items is not None and items.hi != 0 and style and "list_style" not in chosen
                and not user_spec.list_style and rng.random() < P_STYLE_WITH_ITEMS):
            chosen["list_style"] = style              # "exactly 3 bullet points"
        try:
            added = ConstraintSpec(**chosen)
        except ValueError:
            continue
        expected = C.merge(user_spec, added)
        if expected is None:
            continue
        if added.repeat is not None:
            instruction = C.REPEAT_INSTRUCTIONS[int(rng.integers(len(C.REPEAT_INSTRUCTIONS)))]
        else:
            instruction = C.render(added, rng)
        new_user = append_instruction(user_text, instruction, rng)
        new_spec = C.compile(new_user)
        if new_spec != expected or C.verify(new_spec, text, m):
            continue
        out = [dict(x) for x in messages]
        out[-2] = {**dict(user), "content": new_user}
        return out, added
    return None


def relabel(messages: Sequence[Mapping[str, Any]], rng: np.random.Generator, tok=None,
            attempts: int = 4) -> list[dict] | None:
    """A copy of ``messages`` whose last user message carries an appended, verified instruction
    that the final assistant response satisfies; ``None`` when no such instruction is found
    (module docstring). ``tok`` renders assistant content given as parts (default: the
    committed tokenizer, loaded only when needed). The input is not modified."""
    res = relabel_with_spec(messages, rng, tok, attempts)
    return None if res is None else res[0]


__all__ = ["relabel", "relabel_with_spec", "candidates", "response_text", "append_instruction",
           "FIELD_WEIGHTS"]
