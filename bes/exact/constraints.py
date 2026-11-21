"""Constraints: exact response properties, a conservative compiler for explicit requests,
verifiers, natural instructions and countdown targets (``docs/SPEC.md`` section 2.2).

Everything is a deterministic function of text (or of token ids, through
``bes.data.chat_format.display_text``). Counts of words, sentences and list items are those of
the byte-level transducer (``bes.exact.transducer.measure_counts``; :func:`transducer_counts`
runs the same compiled machine faster), so a verified response and the registers fed to the
model agree exactly.

Properties (:func:`measure`)
----------------------------
``measure(text, keywords=())`` returns a dict of exact properties of one response:

- ``words``, ``sentences``, ``items`` (list items): the transducer's counts, the end of the text
  closing the turn. ``bullet_items`` / ``numbered_items`` split ``items`` by the line's marker.
- ``newlines``: number of ``\\n`` (the register's line count). ``lines``: lines with at least one
  byte that is not ASCII whitespace.
- ``paragraphs``: maximal runs of lines that are not separators; a separator is a blank line
  or a thematic break (``***``, ``---``, ``___``, spaces allowed) outside a code fence. A fenced
  block, blank lines included, belongs to the paragraph around it. **Divider mode**: when the
  text uses the markdown divider ``***`` outside fences (a line made of three or more ``*`` and
  spaces, or a run of three or more ``*`` with whitespace or the line edge on both sides),
  paragraphs are the non-empty chunks between dividers and blank lines do not separate (the
  convention of "paragraphs are separated with the markdown divider ***" requests).
- ``code_fences``: fence lines (a line that starts, after spaces or tabs, with three backticks:
  the transducer's fence toggle); ``code_blocks = code_fences // 2``; ``has_code_fence``.
- ``headings``: lines outside fences of the form ``#``..``######``, a space or tab, text (at most
  three leading spaces).
- ``chars`` (code points), ``commas``, ``question_marks``.
- ``depth``, ``open_fence``, ``open_inline``, ``open_quote``: open brackets and formatting state
  at the end (transducer).
- ``ends_with_question``: the last character, ignoring trailing whitespace and the closers
  ``" ' ) ] } * _ ” ’ »``, is ``?``. ``starts_with_capital``: the first character after leading
  whitespace and ``# * _ > " ' ( [ “ ‘ «`` is an uppercase letter.
- ``all_lower`` / ``all_upper``: :meth:`str.islower` / :meth:`str.isupper` (at least one cased
  character and none of the other case).
- ``json_valid``: the text, stripped and without one optional surrounding code fence
  (``` or ```json), parses as a JSON object or array (``NaN``/``Infinity`` rejected).
- ``keywords``: ``{keyword: occurrences}`` for the given keywords: case-insensitive matches
  that start at a word start (not after a letter or digit), so inflected forms count
  (``"dress"`` in ``"dressing"``); ``keywords_whole``: the same with a word end required too.
  Inner spaces of a keyword match any whitespace. A required keyword must have
  ``keywords >= 1``; a forbidden one must have ``keywords_whole == 0`` (both lenient towards
  the response).

Specification (:class:`ConstraintSpec`)
---------------------------------------
Counts are intervals ``Interval(lo, hi)`` (``hi = None``: no upper bound) on ``words,
sentences, items, paragraphs, lines``; ``keywords`` must each appear, ``forbidden`` must not;
``json_output`` (``json_valid``), ``code_block`` (at least one closed fenced block), ``case``
(``"lower"``: ``all_lower``, ``"upper"``: ``all_upper``), ``end_question``
(``ends_with_question``), ``no_commas`` (no ``,``). ``items = Interval(0, 0)`` means "no list".

Compiler (:func:`compile`)
--------------------------
Only explicit, checkable requests about the whole response are compiled; anything else gives
no constraint. The text is split into sentences (after ``.!?``, at newlines and ``;``) and
clauses (at ``, ``, ``: ``, `` - ``, parentheses and "but / however / unless ..."). Decisions:

- Not read as instructions: code fences, inline code, blockquote lines and the inside of quotes
  (``"..."``, ``'...'``, curly quotes, ``[...]``; quoted text is a keyword only after a keyword
  trigger). Messages longer than ``SCAN_CHARS`` are read in their first and last
  ``SCAN_CHARS // 2`` characters, where instructions are.
- Relations: ``exactly/precisely N``; ``at most / no more than / not more than / a maximum of /
  up to / as many as / within / N or fewer|less / N max`` -> ``[0, N]``; ``fewer than / less
  than / under / below N`` -> ``[0, N-1]``; ``at least / no fewer than / a minimum of / N or
  more / N+`` -> ``[N, inf)``; ``more than / over / exceed N`` -> ``[N+1, inf)``; ``between A
  and B``, ``A to B``, ``A-B``, ``A or B`` -> ``[A, B]`` (``up to A or B`` -> ``[0, B]``). A
  negation that governs the phrase flips it (``do not use more than 3 sentences``,
  ``should not exceed 100 words`` -> ``[0, N]``); any other negation within two words, a negated
  "exactly" or a negated bare count gives nothing. ``about / around / roughly / or so`` give
  nothing.
- Bare counts (no relation word) are exact for sentences, items, paragraphs and lines (``in
  three sentences``, ``a one-sentence answer``, ``a single paragraph``, ``your response must
  have 2 paragraphs``, ``write 4 lines``) and an upper bound for words (``in 50 words``, ``a
  100-word summary`` -> ``[0, N]``): word counts are budgets. Bare word counts need ``in /
  within / using / as`` or the adjectival form (``give me 10 words that rhyme`` is not a
  length).
- Licensing. A count is read in a *request*: a clause that starts with a request verb
  (possibly after "please", "can you", "make sure to", ...), or contains "should / must /
  needs to / is required", or starts with "I want / I need / I'd like", or is a predicate of a
  requirement list ("- contains at least 5 sentences"), or belongs to a request sentence. A
  relation or a bare count after ``in / into / using / within / as`` and a fragment that opens
  its clause (``Max 3 sentences.``) are also read. Weak introducers (``with / of / contain /
  have / be / write / give / use ...``) need a request; ``with`` and ``of`` also need a text
  noun before them (``a poem of 4 lines``, not ``the intersection of two lines``). Adjectival
  forms need a production verb before and a text noun after (``write a 3-sentence summary``,
  not ``describe a 5-paragraph essay structure``).
- Not requests: statements (a clause or sentence that starts with a subject, "here is", or a
  quoted span, without a request: ``I wrote a 300-word essay``, ``you edited it into 3
  sentences``); informational, hypothetical and conditional questions (``how do you ...``,
  ``is it possible ...``, ``what if ...``, ``if / when ...``); counts in sentences about code
  (``a function that takes a single word``, ``each row in a single line``); per-part counts
  (``each``, ``per``, ``every``); counts of an edit (``by removing at most three words``);
  comparisons (``one item shorter``); counts after a determiner (``in this one sentence``);
  counts in multi-step sentences (``then``, ``followed by``, ``first``, ``and explain``).
- ``list|name|give me|suggest|provide|... N <things>`` opening a request sentence gives ``items =
  [N, N]`` for ``N >= 2``, unless the message asks for an inline form (comma-separated, one
  sentence, one line), another quantity follows in the sentence (``three pros and three
  cons``) or the count is per part. The same count inside a question ("what are three reasons
  ...?") is a soft request and gives nothing.
- Keywords must be delimited by quotes, ``[...]`` or ``*...*`` after a trigger (``include /
  use / mention ... the word(s) / keyword(s) / phrase(s) / term(s)``, ``include "x"``, ``the
  word X should appear``) in a request sentence, or be single plain words in the template
  ``include (the) keyword(s) A, B and C (in the response).``. ``[a, b]`` and ``*a, b*`` are
  lists; a bracketed template variable defined in the message (``[keywords] are "a", "b"``) is
  replaced by its quoted values, and generic names (``[keyword]``, names with ``_`` or digits)
  are dropped. ``X or Y`` lists are not required keywords. Negated triggers (``do not include /
  avoid / without using / should not appear``) give ``forbidden``. Frequency requests keep only
  the implied inclusion (``at least 3 times`` -> required; ``less than 2 times`` -> nothing).
  Nothing is read in sentences about code.
- Formats: ``answer / respond / format ... in|as JSON``, ``your response must be in JSON``;
  ``in a code block`` / ``use a code block`` / ``wrap ... in triple backticks`` (in a request);
  ``all lowercase / in lowercase letters / no capital letters``; ``all caps / in capital
  letters`` (not "convert to lowercase", not "words in all caps should appear 2 times");
  ``end (your response) with a question``; ``do not use any commas / no commas``. Nothing in a
  conditional sentence.
- Several constraints on one count are intersected (``at least 3 and at most 5 sentences``);
  an empty intersection drops that count. Two count requests joined by ``or``, or a count
  followed by an ``or`` alternative (``a single sentence, a paragraph, or an essay``), are
  dropped.

Targets (:func:`targets_for_tokens`)
------------------------------------
For every assistant turn, the targets ``(sentences, items, words)`` come from the user turn
immediately before it in the same segment and are broadcast over the whole assistant turn (its
``ASSISTANT`` token, content, ``END`` and anything up to the next turn start); -1 elsewhere.
A count with an upper bound gives the countdown target ``hi`` (``exactly N`` and ``at most N``
give ``N``, ``fewer than N`` gives ``N - 1``, ``between A and B`` gives ``B``); a lower bound alone
gives none, because the register's overshoot value would misdescribe it.
"""
from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from bes.config import ASSISTANT, BOS, END, ROLE_TOKENS, USER
from bes.exact.transducer import COMPONENTS, compile_machine

COUNT_UNITS = ("words", "sentences", "items", "paragraphs", "lines")
TARGET_UNITS = ("sentences", "items", "words")       # order of the last axis of the targets
MAX_COUNT = 100_000
MAX_KEYWORDS = 8
SCAN_CHARS = 6000

# ================================================================================ measure
_ASCII_WS = " \t\n\r\x0b\x0c"
_LINE_WS = " \t\r\x0b\x0c"
_FENCE_LINE = re.compile(r"[ \t]*```")
_ITEM_LINE = re.compile(r"[ \t]*(?:([-*+•])|[0-9]+[.)])[ \t]")
_HEADING_LINE = re.compile(r" {0,3}#{1,6}[ \t].*\S")
_BREAK_LINE = re.compile(r" {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})\Z")
_TRAILING_CLOSERS = _ASCII_WS + "\"')]}*_”’»"
_LEADING_MARKUP = _ASCII_WS + "#*_>\"'([“‘«"
_JSON_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n(.*?)\n?[ \t]*```", re.S)


def _json_reject(name: str):
    raise ValueError(name)


def json_valid(text: str) -> bool:
    """``True`` when ``text`` (stripped, without one surrounding code fence) is a JSON object or
    array."""
    s = text.strip(_ASCII_WS)
    m = _JSON_FENCE.fullmatch(s)
    if m:
        s = m.group(1).strip(_ASCII_WS)
    try:
        value = json.loads(s, parse_constant=_json_reject)
    except (ValueError, RecursionError):
        return False
    return isinstance(value, (dict, list))


@functools.lru_cache(maxsize=4096)
def keyword_pattern(keyword: str, whole: bool = True) -> re.Pattern:
    """Case-insensitive pattern of ``keyword`` that starts at a word start (not after a letter
    or digit) and, when ``whole``, also ends at a word end; inner spaces match any whitespace."""
    body = r"\s+".join(re.escape(w) for w in keyword.split())
    return re.compile(r"(?<![^\W_])" + body + (r"(?![^\W_])" if whole else ""), re.IGNORECASE)


def count_keyword(text: str, keyword: str, whole: bool = True) -> int:
    """Occurrences of ``keyword`` in ``text`` (see :func:`keyword_pattern`)."""
    if not keyword.split():
        return 0
    return len(keyword_pattern(keyword, whole).findall(text))


_STAR_RULE_LINE = re.compile(r"[ \t]*(?:\*[ \t]*){3,}\Z")
_STAR_DIVIDER = re.compile(r"(?:(?<=[ \t])|^)\*{3,}(?=[ \t]|\Z)")


def _line_stats(text: str) -> dict[str, int]:
    lines = text.split("\n")
    fence = False
    stats = dict(lines=0, paragraphs=0, headings=0, code_fences=0, bullet_items=0,
                 numbered_items=0)
    in_par = False
    chunks, in_chunk, dividers = 0, False, 0      # "***"-divided chunks (divider mode)
    for line in lines:
        body = line.rstrip("\r")
        blank = not line.strip(_LINE_WS)
        if not blank:
            stats["lines"] += 1
        if not fence and (blank or _BREAK_LINE.match(body)):
            in_par = False
        elif not in_par:
            stats["paragraphs"] += 1
            in_par = True
        if not fence:
            if _HEADING_LINE.match(line):
                stats["headings"] += 1
            m = _ITEM_LINE.match(line)
            if m:
                stats["bullet_items" if m.group(1) else "numbered_items"] += 1
        # Divider mode: a line of stars, or a "***" run with whitespace (or the line edge) on
        # both sides; the text between dividers forms the chunks.
        if not fence and _STAR_RULE_LINE.match(body):
            dividers += 1
            in_chunk = False
        else:
            pieces = [body] if fence else _STAR_DIVIDER.split(body)
            for k, piece in enumerate(pieces):
                if k:
                    dividers += 1
                    in_chunk = False
                if piece.strip(_LINE_WS) and not in_chunk:
                    chunks += 1
                    in_chunk = True
        if _FENCE_LINE.match(line):
            stats["code_fences"] += 1
            fence = not fence
    if dividers:
        stats["paragraphs"] = chunks
    return stats


@functools.lru_cache(maxsize=1)
def _byte_tables() -> tuple[list[list[int]], list[int], list[int], list[list[int]]]:
    """The transducer's compiled per-byte tables as Python lists: ``code[s][b] = next_state * 64
    + events`` (bit 0 word, 1 sentence, 2 newline, 3 item, bits 4-5 bracket + 1), the flush
    state and sentence of every state, and the state components."""
    m = compile_machine()
    ev = m.events.astype(np.int64)
    packed = (ev[..., 0] | ev[..., 1] << 1 | ev[..., 2] << 2 | ev[..., 3] << 3
              | (ev[..., 4] + 1) << 4)
    code = (m.next_state.astype(np.int64) * 64 + packed).tolist()
    return code, m.flush_state.astype(np.int64).tolist(), m.flush_sentence.astype(np.int64).tolist(), \
        m.components.astype(np.int64).tolist()


_FENCE_C, _INLINE_C, _QUOTE_C = (COMPONENTS.index(k) for k in ("fence", "inline", "quote"))


def transducer_counts(text: str | bytes) -> dict[str, int]:
    """``bes.exact.transducer.measure_counts(text)`` (the end of the text closes the turn),
    computed with the compiled per-byte tables of the same machine (several times faster;
    equality is tested)."""
    code, flush_state, flush_sentence, comp = _byte_tables()
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    s = words = sentences = lines = items = wis = depth = 0
    for b in data:
        c = code[s][b]
        s, e = c >> 6, c & 63
        if e & 2:
            sentences += 1
            wis = 0
        if e & 1:
            words += 1
            wis += 1
        if e & 4:
            lines += 1
        if e & 8:
            items += 1
        bracket = (e >> 4) - 1
        if bracket:
            depth = max(0, depth + bracket)
    if flush_sentence[s]:
        sentences += 1
        wis = 0
    s = flush_state[s]
    st = comp[s]
    return {"words": words, "sentences": sentences, "lines": lines, "items": items,
            "word_in_sent": wis, "depth": depth, "fence": st[_FENCE_C], "inline": st[_INLINE_C],
            "quote": st[_QUOTE_C], "pending_sentence": 0}


def measure(text: str, keywords: Iterable[str] = ()) -> dict:
    """Exact properties of one response (definitions in the module docstring)."""
    counts = transducer_counts(text)
    stats = _line_stats(text)
    head = text.lstrip(_LEADING_MARKUP)
    out = {
        "chars": len(text),
        "words": counts["words"],
        "sentences": counts["sentences"],
        "lines": stats["lines"],
        "newlines": counts["lines"],
        "paragraphs": stats["paragraphs"],
        "items": counts["items"],
        "bullet_items": stats["bullet_items"],
        "numbered_items": stats["numbered_items"],
        "headings": stats["headings"],
        "code_fences": stats["code_fences"],
        "code_blocks": stats["code_fences"] // 2,
        "has_code_fence": stats["code_fences"] > 0,
        "commas": text.count(","),
        "question_marks": text.count("?"),
        "depth": counts["depth"],
        "open_fence": bool(counts["fence"]),
        "open_inline": bool(counts["inline"]),
        "open_quote": bool(counts["quote"]),
        "ends_with_question": text.rstrip(_TRAILING_CLOSERS).endswith("?"),
        "starts_with_capital": bool(head) and head[0].isupper(),
        "all_lower": text.islower(),
        "all_upper": text.isupper(),
        "json_valid": json_valid(text),
        "keywords": {k: count_keyword(text, k, whole=False) for k in keywords},
        "keywords_whole": {k: count_keyword(text, k, whole=True) for k in keywords},
    }
    return out


# ============================================================================ the spec
@dataclass(frozen=True)
class Interval:
    """Allowed values ``lo <= n <= hi`` of a count (``hi = None``: no upper bound)."""

    lo: int = 0
    hi: int | None = None

    def __post_init__(self):
        if not isinstance(self.lo, int) or self.lo < 0:
            raise ValueError(f"lo must be a non-negative int, got {self.lo!r}")
        if self.hi is not None and (not isinstance(self.hi, int) or self.hi < self.lo):
            raise ValueError(f"hi must be None or an int >= lo, got {self.hi!r}")

    @classmethod
    def exactly(cls, n: int) -> "Interval":
        return cls(n, n)

    @classmethod
    def at_most(cls, n: int) -> "Interval":
        return cls(0, n)

    @classmethod
    def at_least(cls, n: int) -> "Interval":
        return cls(n, None)

    @property
    def kind(self) -> str:
        """``"exactly"``, ``"at_most"``, ``"at_least"`` or ``"between"``."""
        if self.hi == self.lo:
            return "exactly"
        if self.hi is None:
            return "at_least"
        return "at_most" if self.lo == 0 else "between"

    def contains(self, n: int) -> bool:
        return self.lo <= n and (self.hi is None or n <= self.hi)

    def intersect(self, other: "Interval") -> "Interval | None":
        lo = max(self.lo, other.lo)
        his = [h for h in (self.hi, other.hi) if h is not None]
        hi = min(his) if his else None
        return None if hi is not None and hi < lo else Interval(lo, hi)

    def __str__(self) -> str:
        if self.kind == "exactly":
            return f"exactly {self.lo}"
        if self.kind == "at_least":
            return f"at least {self.lo}"
        if self.kind == "at_most":
            return f"at most {self.hi}"
        return f"between {self.lo} and {self.hi}"


def _valid_interval(unit: str, iv: Interval | None) -> bool:
    if iv is None:
        return True
    if (iv.hi if iv.hi is not None else iv.lo) > MAX_COUNT:
        return False
    if iv.hi is None:
        return iv.lo >= 1
    if iv.hi == 0:
        return unit == "items"
    return True


_KEYWORD_OK = re.compile(r"[^\W_](?:[^\"“”‘’\[\]*`<>{}|,;\n\t]{0,38}[^\W_])?")


def valid_keyword(k: str) -> bool:
    """A keyword is 1-40 characters, starts and ends with a letter or digit, has at most four
    words and none of the delimiters ``" “ ” ‘ ’ [ ] * ` < > { } | , ;``."""
    return (isinstance(k, str) and bool(_KEYWORD_OK.fullmatch(k)) and len(k.split()) <= 4
            and "  " not in k)


@dataclass(frozen=True)
class ConstraintSpec:
    """Explicit, checkable constraints on one response (semantics in the module docstring)."""

    words: Interval | None = None
    sentences: Interval | None = None
    items: Interval | None = None
    paragraphs: Interval | None = None
    lines: Interval | None = None
    keywords: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    json_output: bool = False
    code_block: bool = False
    case: str | None = None          # "lower" | "upper"
    end_question: bool = False
    no_commas: bool = False

    def __post_init__(self):
        for unit in COUNT_UNITS:
            iv = getattr(self, unit)
            if iv is not None and not isinstance(iv, Interval):
                raise TypeError(f"{unit} must be an Interval or None")
            if not _valid_interval(unit, iv):
                raise ValueError(f"{unit}: {iv} is not a checkable request")
        for name in ("keywords", "forbidden"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(valid_keyword(k) for k in value):
                raise ValueError(f"{name} must be a tuple of valid keywords, got {value!r}")
            if len({k.lower() for k in value}) != len(value):
                raise ValueError(f"{name} has duplicates: {value!r}")
        if {k.lower() for k in self.keywords} & {k.lower() for k in self.forbidden}:
            raise ValueError("a keyword cannot be both required and forbidden")
        if self.case not in (None, "lower", "upper"):
            raise ValueError(f"case must be None, 'lower' or 'upper', got {self.case!r}")

    @property
    def empty(self) -> bool:
        return not self.constraints

    @property
    def constraints(self) -> tuple[str, ...]:
        """One description per atomic constraint, e.g. ``("sentences: exactly 3",
        "keyword: 'ocean'")``."""
        out = [f"{u}: {getattr(self, u)}" for u in COUNT_UNITS if getattr(self, u) is not None]
        out += [f"keyword: {k!r}" for k in self.keywords]
        out += [f"forbidden: {k!r}" for k in self.forbidden]
        for name in ("json_output", "code_block", "end_question", "no_commas"):
            if getattr(self, name):
                out.append(name)
        if self.case:
            out.append(f"case: {self.case}")
        return tuple(out)

    def targets(self) -> tuple[int, int, int]:
        """Countdown targets ``(sentences, items, words)``: the upper bound, -1 for none."""
        return tuple(-1 if getattr(self, u) is None or getattr(self, u).hi is None
                     else getattr(self, u).hi for u in TARGET_UNITS)  # type: ignore[return-value]


EMPTY = ConstraintSpec()


def merge(a: ConstraintSpec, b: ConstraintSpec) -> ConstraintSpec | None:
    """Both specs at once (counts intersected, keyword lists concatenated without duplicates),
    or ``None`` when they conflict."""
    values: dict = {}
    for unit in COUNT_UNITS:
        x, y = getattr(a, unit), getattr(b, unit)
        if x is None or y is None:
            values[unit] = x if y is None else y
        else:
            values[unit] = x.intersect(y)
            if values[unit] is None:
                return None
    for name in ("keywords", "forbidden"):
        seen, out = set(), []
        for k in getattr(a, name) + getattr(b, name):
            if k.lower() not in seen:
                seen.add(k.lower())
                out.append(k)
        values[name] = tuple(out)
    if a.case and b.case and a.case != b.case:
        return None
    values["case"] = a.case or b.case
    for name in ("json_output", "code_block", "end_question", "no_commas"):
        values[name] = getattr(a, name) or getattr(b, name)
    try:
        return ConstraintSpec(**values)
    except ValueError:
        return None


# ========================================================================== verification
def verify(spec: ConstraintSpec, text: str, measured: dict | None = None) -> list[str]:
    """Violated constraints of ``spec`` on ``text`` (empty list: all satisfied). Each entry
    starts with the constraint's name (``"sentences: ..."``, ``"keyword: ..."``).
    ``measured``: ``measure(text)`` computed before (keyword counts are added as needed)."""
    words = spec.keywords + spec.forbidden
    if measured is None:
        m = measure(text, words)
    else:
        m = dict(measured)
        m["keywords"] = {k: count_keyword(text, k, whole=False) for k in words}
        m["keywords_whole"] = {k: count_keyword(text, k, whole=True) for k in words}
    out = []
    for unit in COUNT_UNITS:
        iv = getattr(spec, unit)
        if iv is not None and not iv.contains(m[unit]):
            out.append(f"{unit}: {m[unit]}, want {iv}")
    out += [f"keyword: {k!r} missing" for k in spec.keywords if m["keywords"][k] == 0]
    out += [f"forbidden: {k!r} present" for k in spec.forbidden if m["keywords_whole"][k] > 0]
    if spec.json_output and not m["json_valid"]:
        out.append("json_output: not a JSON object or array")
    if spec.code_block and m["code_blocks"] == 0:
        out.append("code_block: no closed code block")
    if spec.case == "lower" and not m["all_lower"]:
        out.append("case: not all lowercase")
    if spec.case == "upper" and not m["all_upper"]:
        out.append("case: not all uppercase")
    if spec.end_question and not m["ends_with_question"]:
        out.append("end_question: does not end with a question")
    if spec.no_commas and m["commas"]:
        out.append(f"no_commas: {m['commas']} commas")
    return out


# ============================================================================== numbers
_SMALL = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
          "fifteen sixteen seventeen eighteen nineteen twenty").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
NUMBER_WORDS: dict[str, int] = {w: i + 1 for i, w in enumerate(_SMALL)}
NUMBER_WORDS.update(_TENS)
for _t, _tv in _TENS.items():
    for _u in range(1, 10):
        NUMBER_WORDS[f"{_t}-{_SMALL[_u - 1]}"] = _tv + _u
NUMBER_WORDS.update({"one hundred": 100, "a hundred": 100})
WORD_OF_NUMBER: dict[int, str] = {}
for _w, _v in NUMBER_WORDS.items():
    if _v not in WORD_OF_NUMBER and not _w.startswith("a "):
        WORD_OF_NUMBER[_v] = _w

_NUM_WORDS_RE = "|".join(re.escape(w) for w in sorted(NUMBER_WORDS, key=len, reverse=True))
_NUM_BODY = (r"(?:(?<![\w.,$/])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,6})(?![0-9]|[.,][0-9]|%|/[0-9])"
             rf"|(?<![\w-])(?:{_NUM_WORDS_RE})(?![\w]))")


def _num(name: str) -> str:
    return f"(?P<{name}>{_NUM_BODY})"


def parse_count(text: str) -> int | None:
    """Value of a number as written in a request (digits, ``1,000`` or a number word)."""
    t = text.strip().lower()
    if t in NUMBER_WORDS:
        return NUMBER_WORDS[t]
    t = t.replace(",", "")
    return int(t) if t.isdigit() else None


# ============================================================================== masking
_QUOTED = re.compile(
    r'"[^"\n]{1,300}"|“[^”\n]{1,300}”|‘[^’\n]{1,120}’'
    r"|(?<![\w])'[^'\n]{1,120}'(?![\w])|`[^`\n]{1,200}`|\[[^\[\]\n]{1,120}\]")
_BLOCKQUOTE = re.compile(r"[ \t]*>")
_INERT = "\x01"


def _mask_code(text: str) -> str:
    """``text`` with fenced code (fence lines included) and blockquote lines replaced by
    spaces (same length, newlines kept)."""
    out = []
    fence = False
    for line in text.split("\n"):
        toggles = bool(_FENCE_LINE.match(line))
        if fence or toggles or _BLOCKQUOTE.match(line):
            out.append(" " * len(line))
        else:
            out.append(line)
        if toggles:
            fence = not fence
    return "\n".join(out)


def _mask_quote(span: str) -> str:
    inner = span[1:-1]
    keep = len(inner) - len(inner.rstrip(".!?"))         # '"What is X?" Your ...': keep the "?"
    return span[0] + _INERT * (len(inner) - keep) + inner[len(inner) - keep:] + span[-1]


def _mask_quotes(text: str) -> str:
    """``text`` with the inside of every quoted span replaced by an inert character, except a
    final ``.!?`` (so a quoted question still ends its sentence)."""
    return _QUOTED.sub(lambda m: _mask_quote(m.group(0)), text)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’])\s+|\s*\n\s*|;\s*|(?<=:)\s+(?=[A-Z0-9])")


def _sentences(masked: str) -> list[tuple[int, int]]:
    """Spans of sentences (split after ``.!?`` followed by whitespace, at newlines and ``;``)."""
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(masked):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(masked):
        spans.append((pos, len(masked)))
    return [(a, b) for a, b in spans if masked[a:b].strip()]


_NORM_TABLE = str.maketrans({"*": " ", "_": " ", "’": "'", "‘": "'", "“": '"',
                             "”": '"', "–": "-", "—": " - ", "\t": " "})


def _norm(s: str) -> str:
    """Lowercase, emphasis removed, dashes and curly quotes normalised, spaces collapsed."""
    return re.sub(r"\s+", " ", s.translate(_NORM_TABLE).lower()).strip()


_CLAUSE_SPLIT = re.compile(r", |: |(?<![0-9]) - (?![0-9])| \(|\) |\b(?:but|however|although|though|"
                           r"whereas|otherwise|unless|except|instead)\b ")


def _clauses(sentence: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _CLAUSE_SPLIT.finditer(sentence):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(sentence)))
    return [(a, b) for a, b in spans if sentence[a:b].strip()]


# ========================================================================== count rules
_ADJ = (r"(?:short|brief|concise|complete|full|separate|distinct|simple|clear|key|main|unique|"
        r"different|individual|coherent|informative|detailed|long|longer|descriptive|bulleted|"
        r"numbered|markdown|total|more|additional|well-written|well-formed|good|new|meaningful|"
        r"grammatical|proper|actual|english|whole|own)")
_FILL = rf"(?:{_ADJ} ){{0,2}}?"
_UNIT_ALT = (r"(?:bullet[- ]?points?|bullets?|bulleted (?:items?|points?|lines?)|"
             r"numbered (?:items?|points?|lines?)|list items?|items?|words?|sentences?|"
             r"paragraphs?|lines?(?! of (?:code|python|javascript|java|sql|html|css|output)\b))"
             r"(?![\w-])")
_UNIT_SING = (r"(?:bullet[- ]?point|bullet|(?:list|numbered|bulleted) (?:item|point)|item|word|"
              r"sentence|paragraph|line(?! of (?:code|python|javascript|java|sql|html|css|output)\b))")


def _unit(name: str = "unit") -> str:
    return f"(?P<{name}>{_UNIT_ALT})"


def unit_of(noun: str) -> str:
    """Count unit of a unit noun (``"bullet points"`` -> ``"items"``)."""
    n = noun.lower()
    if n.startswith(("bullet", "numbered", "list", "item")):
        return "items"
    for unit in ("words", "sentences", "paragraphs", "lines"):
        if n.startswith(unit[:-1]):
            return unit
    raise ValueError(noun)


_REL_WORDS = {
    "exactly": r"exactly|precisely",
    "at_most": (r"at most|at max(?:imum)?|at the most|no more than|not more than|nothing more than|"
                r"a maximum of|maximum of|max(?:imum)?(?: of)?|up to|no longer than|not longer than|"
                r"not to exceed|not exceeding|no greater than|as many as"),
    "lt": r"fewer than|less than|under|below|shorter than",
    "at_least": (r"at least|at minimum|at a minimum|no fewer than|no less than|not fewer than|"
                 r"not less than|a minimum of|minimum of|min(?:imum)?(?: of)?|no shorter than"),
    "gt": r"more than|over|longer than|greater than|in excess of",
}
_REL_OF: list[tuple[re.Pattern, str]] = [(re.compile(rf"(?:{v})\Z"), k) for k, v in _REL_WORDS.items()]
_REL_ANY = "|".join(sorted((w for v in _REL_WORDS.values() for w in v.split("|")), key=len, reverse=True))
_REL = rf"(?P<rel>{_REL_ANY})"


def _rel_kind(phrase: str) -> str:
    for pat, kind in _REL_OF:
        if pat.match(phrase):
            return kind
    raise ValueError(phrase)


def _interval(kind: str, n: int) -> Interval | None:
    if kind == "exactly":
        return Interval(n, n)
    if kind == "at_most":
        return Interval(0, n)
    if kind == "lt":
        return Interval(0, n - 1) if n >= 1 else None
    if kind == "at_least":
        return Interval(n, None)
    if kind == "gt":
        return Interval(n + 1, None)
    raise ValueError(kind)


_FLIP = {"at_most": "gt", "gt": "at_most", "lt": "at_least", "at_least": "lt", "exactly": None}
_NEG = re.compile(r"\b(?:not|never|no|don't|dont|doesn't|shouldn't|mustn't|cannot|can't|won't|"
                  r"without|avoid|nor|neither)\b")
# A negation that clearly governs the count phrase right after it ("do not use more than",
# "should not exceed", "without exceeding", "don't write your answer in more than").
_NEG_GOVERNS = re.compile(
    r"\b(?:do not|don't|dont|does not|doesn't|did not|didn't|never|not|cannot|can't|should not|"
    r"shouldn't|must not|mustn't|will not|won't|without|no need to) (?:(?:use|using|write|writing|"
    r"go|going|be|have|having|include|including|make|making|exceed|exceeding|give|giving|provide|"
    r"providing|produce|producing|put|add|adding|type|respond|answer|reply|need|contain|containing|"
    r"it|this|that|your|the|answer|response|reply|text|essay|summary|anything|any|in|with|of|a|an|"
    r"me|us|them|to) ){0,4}\Z")
# Any other negation this close to the count phrase makes it ambiguous.
_NEG_NEAR = re.compile(r"\b(?:not|never|no|don't|dont|doesn't|shouldn't|mustn't|cannot|can't|"
                       r"won't|without|avoid|nor|neither)\b(?: \S+){0,2} ?\Z")
_SUFFIX_AFTER = re.compile(r" (?:or (?:less|fewer|under|shorter|below|more|longer|above|greater)|"
                           r"max(?:imum)?|at most|at the most|tops|at least|minimum|min|at minimum|"
                           r"or so)\b")
_APPROX_BEFORE = re.compile(r"(?:\b(?:about|around|approximately|approx|roughly|circa|nearly|"
                            r"almost|some|like|maybe|perhaps|or so)|~) ?\Z")
_APPROX_AFTER = re.compile(r" ?(?:or so|give or take|more or less|roughly|approximately|"
                           r"or thereabouts|-?ish)\b")

_LEAD = (r"(?:(?:[-*•]|[0-9]+[.)]) )?(?:(?:please|kindly|also|and|then|now|so|just|finally|"
         r"additionally|lastly|first|firstly|second|secondly|next|moreover|furthermore|plus|ok|"
         r"okay|hey|hi|hello|again|instead|rather|simply|briefly|quickly|in addition|while),? )*")
_ASK = (r"(?:(?:(?:can|could|would|will) you (?:please )?(?:kindly )?|i (?:want|need|would like|'d like|"
        r"d like|wish|would prefer|prefer) (?:you to )?|i'm looking for |im looking for |"
        r"make sure (?:to |that |you )?|making sure (?:to |that )?|be sure to |try to |remember to |"
        r"feel free to |you (?:should|must|need to|have to|will) |"
        r"let's |lets |help me |go ahead and |i'd appreciate it if you |please )){0,2}")
_REQUEST_VERBS = (r"(?:write|compose|draft|create|generate|produce|give|provide|make|prepare|craft|"
                  r"summari[sz]e|explain|describe|answer|respond|reply|tell|list|name|rewrite|"
                  r"rephrase|paraphrase|translate|keep|limit|use|format|structure|present|break|"
                  r"split|divide|output|return|express|put|outline|suggest|recommend|share|offer|"
                  r"state|define|discuss|elaborate|enumerate|identify|mention|include|end|finish|"
                  r"start|begin|add|show|convert|turn|condense|shorten|expand|organi[sz]e|"
                  r"come up with|brainstorm|develop|design|plan|compile|detail|highlight|note|"
                  r"cover|construct|jot|restrict|cap|reduce|trim|cut|do|try|think of|wrap|"
                  r"deliver|send|type|print|draw|sum up|introduce|analy[sz]e|compare|"
                  r"evaluate|review|argue|justify|teach|clarify|interpret|predict|imagine|"
                  r"pretend|act|play|tweet|post|email|reword|edit|fix|correct|improve|polish|"
                  r"avoid|refrain|exclude|omit|steer|never|don't|dont|do not|ensure|ensuring|"
                  r"making sure|make sure|be sure|aim|stick)")
# A predicate of a requirement list ("- contains at least 5 sentences", "and contain exactly 4
# bullet points"): the request's subject was given before.
_PREDICATE_START = re.compile(rf"^{_LEAD}(?:contains?|includes?|has|have|uses?|is|are|be|ends?|"
                              r"starts?|begins?|mentions?|features?|consists? of|should|must|"
                              r"needs to|totals?|spans?)\b")
_REQUEST_START = re.compile(rf"^{_LEAD}{_ASK}{_REQUEST_VERBS}\b")
_MODAL = re.compile(r"\b(?:should|must|shall|needs? to|has to|have to|ought to|is to|are to|"
                    r"is required|are required|required to|is expected to|are expected to)\b")
_I_WANT = re.compile(rf"^{_LEAD}(?:i (?:want|need|would like|'d like|d like|would prefer|prefer|"
                     r"don't want|do not want|dont want)|i'm looking for|im looking for)\b")
# A modal near the start of the clause ("your response should ...", "there must be ...").
_MODAL_EARLY = re.compile(rf"^{_LEAD}(?:\S+ ){{0,7}}?(?:should|must|shall|needs? to|has to|have to|"
                          r"ought to|is to|are to|is required to|are required to|will need to)\b")
# A clause that starts with a subject ("you edited it into 3 sentences", "this line of code does
# two things in one line", "i wrote a 300-word essay") states a fact unless it is a request.
_SUBJECT_START = re.compile(
    rf"^{_LEAD}(?:(?:i|you|we|he|she|they|it|this|that|these|those|there|my|your|our|his|her|their|"
    r"its|the|someone|people|everyone|i've|i'm|you've|you're|we've|it's|that's|there's|here|"
    r"here's|(?:a|an)(?! (?:maximum|minimum|max|min|total|single|couple)\b)(?! \S+-(?:word|"
    r"sentence|paragraph|line|item|bullet)))\b|\")")
# Informational and hypothetical questions ("how do you decide ...", "what if i ...", "is it
# possible ...", "how could they be merged ...").
_QSUBJ = r"(?:i|we|they|it|this|that|these|those|he|she|the|one|there)"
_NOT_A_REQUEST = re.compile(
    r"^(?:(?:how|what|why|where|which|who|whose) (?:do|does|did|is|are|was|were|has|have|had)\b|"
    rf"how (?:can|could|would|should|might|may|will) {_QSUBJ}\b|"
    r"how (?:can|could|would|should|might|may|will|does|do|did) \w+ing\b|"
    rf"(?:is|are|was|were|does|do|did|has|have|can|could|would|will|should|might) {_QSUBJ}\b|"
    r"what if\b|what about\b|how about\b|if\b|when\b|whenever\b|unless\b|in case\b|once\b|"
    r"whether\b|because\b|since\b|as long as\b)")
# A multi-step task ("write a paragraph, then improve it", "write a sentence and explain it"):
# whole-response counts do not apply.
_SEQUENCE = re.compile(r"\b(?:then|followed by|after that|afterwards|before that|subsequently)\b"
                       r"|(?<=\S )first\b|\band (?:also )?(?:explain|describe|discuss|justify|"
                       r"analy[sz]e|comment on|elaborate|show)\b")


def _is_request(clause: str) -> bool:
    """A loose test (licensing bare counts): a request verb first, a modal anywhere, "I want"."""
    return bool(_REQUEST_START.match(clause) or _MODAL.search(clause) or _I_WANT.match(clause))


def _is_strict_request(clause: str) -> bool:
    """A request verb first, "I want ...", or a modal among the first words."""
    return bool(_REQUEST_START.match(clause) or _I_WANT.match(clause) or _MODAL_EARLY.match(clause))


def _is_statement(clause: str) -> bool:
    if _NOT_A_REQUEST.match(clause):
        return True
    if clause.startswith('"'):
        return True                      # '"alot" should be spelled as two words': about the text
    return bool(_SUBJECT_START.match(clause)) and not _is_strict_request(clause)


_STRONG_INTRO = r"in|into|using|within|as"
_WEAK_INTRO = (r"with|of|contain|contains|containing|have|has|having|be|is|are|write|writing|"
               r"give|giving|provide|providing|produce|compose|generate|make|create|include|"
               r"includes|including|use|list|offer|share|need|want|consist of|consisting of|"
               r"keep|take|takes|span|spanning|total|totaling|totalling")
_OBJ = r"(?:(?:me|us|it|this|that|them|me with|us with) )?"

_P_BETWEEN = re.compile(rf"\bbetween {_num('a')} (?:{_UNIT_ALT} )?and {_num('b')} {_FILL}{_unit()}")
_P_FROMTO = re.compile(rf"(?:\bfrom )?{_num('a')} ?(?:-|to) ?{_num('b')} {_FILL}{_unit()}")
_P_OR = re.compile(rf"{_num('a')} or {_num('b')} {_FILL}{_unit()}")
_P_REL2 = re.compile(rf"\b(?P<rel1>{_REL_ANY}) {_num('a')} (?:{_FILL}(?P<unit1>{_UNIT_ALT}) )?(?:and|but) "
                     rf"(?P<rel2>{_REL_ANY}) {_num('b')} {_FILL}{_unit()}")
_P_REL = re.compile(rf"\b{_REL} (?:only |just )?{_num('num')} {_FILL}{_unit()}")
_P_SUFFIX = re.compile(rf"{_num('num')} {_FILL}{_unit()} (?P<suf>or (?:less|fewer|under|shorter|"
                       r"below|more|longer|above|greater)|max(?:imum)?|at most|at the most|tops|"
                       r"at least|minimum|min|at minimum|exactly|or so)\b")
_P_PLUS = re.compile(rf"{_num('num')}\+ {_FILL}{_unit()}")
_P_BARE = re.compile(rf"\b(?P<intro>{_STRONG_INTRO}|{_WEAK_INTRO}) {_OBJ}(?:only |just )?"
                     rf"{_num('num')} {_FILL}{_unit()}")
_P_ADJ = re.compile(rf"\b(?:a|an) (?:(?:short|brief|concise|quick|simple|single|full|complete) )?"
                    rf"(?P<num>{_NUM_BODY}|single)[- ](?P<unit>{_UNIT_SING})(?![\w-]) (?P<noun>[a-z][\w-]*)"
                    r"(?: (?P<noun2>[a-z][\w-]*))?")
# "write a 3-sentence summary" (not "describe a 5-paragraph essay structure", "use a single-line
# conditional expression"): a production verb before, a noun of a text after.
_PRODUCE = re.compile(r"\b(?:write|compose|draft|create|generate|produce|give|provide|craft|prepare|"
                      r"make|send|share|offer|deliver|return|output|reply|respond|answer|summari[sz]e|"
                      r"rewrite|rephrase|reword|need|want|like|prefer|looking for|turn|convert|"
                      r"condense|shorten|reduce|format|put|keep|limit|construct|develop|compile|jot|"
                      r"type|post|tweet|email|edit|revise|translate|draw up|come up with)\b")
_ADJ_NOUNS = frozenset(
    "answer response reply summary essay story poem paragraph description explanation "
    "introduction intro conclusion list version statement bio biography post email letter "
    "overview speech article report review synopsis abstract tweet caption message note piece "
    "critique criticism rewording rewrite example definition followup follow-up recap outline "
    "guide narrative account analysis argument commentary dialogue entry memo pitch proposal "
    "quote slogan tagline thesis tribute tutorial update blurb sentence text passage poem rhyme "
    "limerick haiku sonnet verse stanza song lyrics script".split())
# As _UNIT_SING without the plain "bullet" ("a single bullet wound").
_UNIT_SING_SINGLE = (r"(?:bullet[- ]?point|(?:list|numbered|bulleted) (?:item|point)|item|word|"
                     r"sentence|paragraph|line(?! of (?:code|python|javascript|java|sql|html|css|output)\b))")
_P_SINGLE = re.compile(rf"\b(?:(?P<prep>in|as|into|with|using) )?(?:a |one )?single (?P<unit>{_UNIT_SING_SINGLE})(?![\w-])")
_P_A_ONE = re.compile(r"\b(?:explain|describe|summari[sz]e|answer|respond|reply|tell me|state|"
                      r"express|define)(?: (?:it|this|that|them|everything|me|us))? in a "
                      r"(?P<unit>sentence|paragraph)(?P<two> or two)?(?![\w-])")
# Nouns of texts: "a poem of 4 lines", "a story with 5 paragraphs" (not "the intersection of two lines").
_TEXT_NOUNS = frozenset(
    "poem story essay summary list paragraph response answer text letter email description piece "
    "post speech limerick haiku song total maximum minimum explanation article report review "
    "outline overview draft version message note bio introduction conclusion verse stanza rhyme "
    "tweet caption blurb synopsis abstract series reply respond write passage paragraphs".split())
_LIST_CONTEXT = re.compile(r"\b(?:list|lists|bullet|bullets|bulleted|numbered|points?|answer|response|"
                           r"reply|format|markdown|output)\b")
_P_LIMIT = re.compile(rf"\b(?:keep|limit|restrict|cap|constrain|shorten|condense|reduce|trim|cut)"
                      rf"(?: [\w']+){{0,4}}? (?:to|at) (?:a maximum of |at most |only |just |no more than )?"
                      rf"{_num('num')} {_FILL}{_unit()}")
_P_EXCEED = re.compile(rf"\b(?:exceed|exceeds|exceeding|surpass|surpassing|go over|go beyond|going over)"
                       rf"(?: the)?(?: limit of)? {_num('num')} {_FILL}{_unit()}")
_P_FRAGMENT = re.compile(rf"^(?:please )?(?:only |just )?{_num('num')} {_FILL}{_unit()}"
                         r"(?: (?:please|only|total|each|in total|is enough|will do))?$")
_P_NO_ITEMS = re.compile(r"\b(?:do not|don't|dont|never|avoid|without|no)(?: (?:use|using|include|"
                         r"including|add|adding|any|the|a|bulleted|numbered|markdown|format(?:ting)?"
                         r"|with))* (?:bullet[- ]?points?|bullets|bulleted lists?|numbered lists?|"
                         r"lists?|list items|list formatting)\b"
                         r"(?= ?(?:$|[.,;:!?)]|or\b|and\b|in\b|please\b|at all\b|anywhere\b))")

_LIST_VERBS = (r"(?:list|name|give(?: me| us)?|provide(?: me with| us with| me| us)?|suggest|"
               r"recommend|share|offer|enumerate|mention|outline|brainstorm|come up with|"
               r"tell me|write down|write out|jot down|think of)")
_P_LIST_N = re.compile(rf"^{_LEAD}{_ASK}{_LIST_VERBS}(?: the)?(?: top)?(?: exactly)? {_num('num')} "
                       r"(?P<noun>[a-z][a-z'-]*)")
# Measures, not listed things ("give me 5 minutes"), and words that cannot head a list noun.
_LIST_BLOCK_NOUNS = frozenset(
    "minute minutes hour hours second seconds day days week weeks month months year years time "
    "times percent dollars dollar cents euros pounds kg km miles meters metres feet inches "
    "degrees or to and".split())
_NUMBER_ANYWHERE = re.compile(_NUM_BODY)
_INLINE_FORM = re.compile(r"comma[- ]separated|separated by commas|in (?:a |one )(?:single )?"
                          r"(?:sentence|line|paragraph)\b|on (?:a |one )(?:single )?line\b|\binline\b|"
                          r"in prose\b|as a sentence\b|paragraph form\b|one-liner")


@dataclass(frozen=True)
class _Cand:
    start: int
    end: int
    unit: str | None
    interval: Interval | None
    list_n: bool = False          # from "list N <things>" (dropped when an inline form is asked)


def _bare_interval(unit: str, n: int) -> Interval:
    return Interval(0, n) if unit == "words" else Interval(n, n)


_STRONG_PREPS = frozenset(("in", "into", "using", "within", "as"))
# A determiner before the number ("in this one sentence", "the two paragraphs below").
_DET_BEFORE = re.compile(r"\b(?:this|that|these|those|the|my|your|his|her|our|their|its|each|every) \Z")
# The count is about an edit, not about the response ("by removing at most three words").
_EDIT_BEFORE = re.compile(r"\b(?:remove|removes|removing|removed|delete|deleting|deleted|cut|cutting|"
                          r"add|adding|added|change|changing|replace|replacing|drop|dropping|omit|"
                          r"omitting|trim|trimming|shorten (?:it |this )?by|reduce (?:it |this )?by)"
                          r"(?: [\w']+){0,2} \Z")
_REL_TAIL = re.compile(rf"(?:^| )({_REL_ANY}) \Z")
# "explain each concept in less than 50 words", "3 sentences per paragraph": per-part counts.
_EACH = re.compile(r"\b(?:each|per|every)\b")
# "one item shorter than the input": a comparison, not a count.
_COMPARATIVE_AFTER = re.compile(r" (?:shorter|longer|more|less|fewer|bigger|smaller|larger|greater|than)\b")


def _count_candidates(c: str, first_clause: bool = True, in_request: bool = False) -> list[_Cand]:
    """All count requests found in one normalised clause ``c`` (overlaps unresolved). A
    candidate with ``interval = None`` is a recognised but ambiguous phrase: it blocks shorter
    readings of the same words and yields no constraint. ``first_clause``: ``c`` opens its
    sentence (the "list N <things>" form is read only there); ``in_request``: the sentence is a
    request ("your response should contain ..., and contain exactly 4 bullet points")."""
    cands: list[_Cand] = []
    if _is_statement(c):
        return cands
    request = _is_request(c) or in_request or bool(_PREDICATE_START.match(c))

    def licensed_at(start: int) -> bool:
        """A relation or range counts in a request, as a fragment (clause start) or after
        "in / into / using / within / as"; never about an edit or after a determiner."""
        before = c[:start]
        if _EDIT_BEFORE.search(before) or _DET_BEFORE.search(before):
            return False
        words = [w for w in before.split() if w not in ("only", "just")] or [""]
        return request or start == 0 or words[-1] in _STRONG_PREPS

    def licensed(m: re.Match) -> bool:
        return licensed_at(m.start())

    def negation(start: int) -> str | None:
        """``"flip"`` (a negation governs the phrase), ``"drop"`` (one is too close to tell)
        or ``None``."""
        before = c[:start]
        if _NEG_GOVERNS.search(before):
            return "flip"
        return "drop" if _NEG_NEAR.search(before) else None

    def approx(m: re.Match, num_group: str) -> bool:
        return bool(_APPROX_BEFORE.search(c[: m.start(num_group)]) or _APPROX_AFTER.match(c, m.end()))

    list_context = bool(_LIST_CONTEXT.search(c))

    per_part = bool(_EACH.search(c))

    def add(m: re.Match, unit_text: str | None, iv: Interval | None) -> None:
        if unit_text in ("item", "items") and not list_context:
            return                       # "orders of 100 items or more": not list items
        if per_part or _COMPARATIVE_AFTER.match(c, m.end()):
            iv = None
        unit = unit_of(unit_text) if unit_text else None
        cands.append(_Cand(m.start(), m.end(), unit, iv))

    def relation(kind: str, start: int) -> str | None:
        """The relation after negation: flipped, or ``None`` when it cannot be decided."""
        state = negation(start)
        if state == "flip":
            return _FLIP[kind]
        return None if state == "drop" else kind

    def plain(m: re.Match, num_group: str = "num") -> bool:
        """No negation near the phrase and no approximation around its number."""
        return negation(m.start()) is None and not approx(m, num_group)

    for pat in (_P_BETWEEN, _P_FROMTO, _P_OR):
        for m in pat.finditer(c):
            a, b = parse_count(m.group("a")), parse_count(m.group("b"))
            iv = None
            if a is not None and b is not None and a < b and plain(m, "a"):
                tail = _REL_TAIL.search(c, 0, m.start())     # "up to 200 or 300 words"
                if tail is None:
                    iv = Interval(a, b) if licensed(m) else None
                elif licensed_at(tail.start(1)) and negation(tail.start(1)) is None:
                    kind = _rel_kind(tail.group(1))
                    iv = {"at_most": Interval(0, b), "lt": Interval(0, b - 1) if b > 1 else None,
                          "at_least": Interval(a, None), "gt": Interval(a + 1, None)}.get(kind)
            add(m, m.group("unit"), iv)
    for m in _P_REL2.finditer(c):
        if m.group("unit1") and unit_of(m.group("unit1")) != unit_of(m.group("unit")):
            continue                     # "at least 200 words and at least 3 sentences"
        a, b = parse_count(m.group("a")), parse_count(m.group("b"))
        iv = None
        if a is not None and b is not None and plain(m, "a") and licensed(m):
            i1 = _interval(_rel_kind(m.group("rel1")), a)
            i2 = _interval(_rel_kind(m.group("rel2")), b)
            iv = i1.intersect(i2) if i1 and i2 else None
        add(m, m.group("unit"), iv)
    for m in _P_REL.finditer(c):
        n = parse_count(m.group("num"))
        kind = relation(_rel_kind(m.group("rel")), m.start())
        ok = n is not None and kind is not None and not approx(m, "num") and licensed(m)
        add(m, m.group("unit"), _interval(kind, n) if ok else None)
    for m in _P_SUFFIX.finditer(c):
        n, suf = parse_count(m.group("num")), m.group("suf")
        if suf == "or so" or n is None or approx(m, "num") or not licensed(m):
            add(m, m.group("unit"), None)
            continue
        if suf == "exactly":
            kind = "exactly"
        elif suf.startswith("or "):
            kind = "at_most" if suf.split()[1] in ("less", "fewer", "under", "shorter", "below") else "at_least"
        else:
            kind = "at_least" if suf in ("at least", "minimum", "min", "at minimum") else "at_most"
        kind = relation(kind, m.start())
        add(m, m.group("unit"), _interval(kind, n) if kind else None)
    for m in _P_PLUS.finditer(c):
        n = parse_count(m.group("num"))
        ok = n is not None and plain(m) and licensed(m)
        add(m, m.group("unit"), Interval(n, None) if ok else None)
    for m in _P_LIMIT.finditer(c):
        n = parse_count(m.group("num"))
        add(m, m.group("unit"), Interval(0, n) if n is not None and plain(m) else None)
    for m in _P_EXCEED.finditer(c):
        n = parse_count(m.group("num"))
        kind = relation("gt", m.start())
        ok = n is not None and kind is not None and not approx(m, "num")
        add(m, m.group("unit"), _interval(kind, n) if ok else None)
    for m in _P_BARE.finditer(c):
        if _SUFFIX_AFTER.match(c, m.end()):
            continue                     # "in 3 sentences or more": the suffix form decides
        n, intro, unit = parse_count(m.group("num")), m.group("intro"), unit_of(m.group("unit"))
        strong = intro in ("in", "into", "using", "within", "as")
        if not (strong or (unit != "words" and (request or m.start() == 0))):
            continue                     # not a request here ("give me 10 words that rhyme")
        if intro in ("of", "with") and m.start() > 0:
            prev = c[: m.start()].split()[-1]
            if prev not in _TEXT_NOUNS:
                continue                 # "the intersection of two lines"
        if n is None or not plain(m) or _EDIT_BEFORE.search(c, 0, m.start()):
            add(m, m.group("unit"), None)
            continue
        add(m, m.group("unit"), Interval(0, n) if intro == "within" else _bare_interval(unit, n))
    for m in _P_ADJ.finditer(c):
        raw = m.group("num")
        n = 1 if raw == "single" else parse_count(raw)
        nouns = {m.group("noun"), m.group("noun2")}
        produced = bool(_PRODUCE.search(c, 0, m.start()) or _I_WANT.match(c))
        if not (produced and nouns & _ADJ_NOUNS):
            continue                     # "describe a 5-paragraph essay structure"
        ok = n is not None and request and plain(m)
        add(m, m.group("unit"), _bare_interval(unit_of(m.group("unit")), n) if ok else None)
    for m in _P_SINGLE.finditer(c):
        ok = (m.group("prep") is not None or request) and negation(m.start()) is None
        add(m, m.group("unit"), _bare_interval(unit_of(m.group("unit")), 1) if ok else None)
    for m in _P_A_ONE.finditer(c):
        ok = request and negation(m.start()) is None
        add(m, m.group("unit"), (Interval(1, 2) if m.group("two") else Interval(1, 1)) if ok else None)
    m = _P_FRAGMENT.match(c)
    if m:
        n = parse_count(m.group("num"))
        unit = unit_of(m.group("unit"))
        add(m, m.group("unit"), _bare_interval(unit, n) if n is not None else None)
    for m in _P_NO_ITEMS.finditer(c):
        cands.append(_Cand(m.start(), m.end(), "items", Interval(0, 0)))
    m = _P_LIST_N.match(c) if first_clause and not _EACH.search(c) else None
    if m and m.group("noun") not in _LIST_BLOCK_NOUNS:
        n = parse_count(m.group("num"))
        noun_is_unit = re.fullmatch(_UNIT_ALT, m.group("noun")) is not None
        another = _NUMBER_ANYWHERE.search(c, m.end())    # "three pros and three cons"
        if n is not None and n >= 2 and not noun_is_unit and not approx(m, "num") and not another:
            cands.append(_Cand(m.start("num"), m.end(), "items", Interval(n, n), list_n=True))
    return cands


def _resolve(cands: list[_Cand]) -> list[_Cand]:
    """Non-overlapping candidates: earliest start first, then the longest."""
    out: list[_Cand] = []
    end = -1
    for cand in sorted(cands, key=lambda x: (x.start, -(x.end - x.start))):
        if cand.start >= end:
            out.append(cand)
            end = cand.end
    return out


_COUNT_PHRASE_AT = re.compile(rf"{_NUM_BODY}\+? {_FILL}{_UNIT_ALT}")


def _another_quantity(sentence: str, pos: int) -> bool:
    """A number after ``pos`` that is not itself the count of a unit ("at most 6 words")."""
    for m in _NUMBER_ANYWHERE.finditer(sentence, pos):
        if not _COUNT_PHRASE_AT.match(sentence, m.start()):
            return True
    return False


_OR_WORD = re.compile(r"\bor\b")
_OR_ALTERNATIVE = re.compile(rf" or (?:[\w'-]+ ){{0,4}}?{_UNIT_ALT}")
_OR_LIST = re.compile(r"(?:, (?!and\b)[^,.;:]{1,30}){1,3}, or\b")   # "a sentence, a paragraph, or ..."


def _sentence_counts(sentence: str, inline_form: bool) -> list[tuple[str, Interval]]:
    """Count requests of one normalised sentence, as ``(unit, interval)`` (none in a multi-step
    sentence such as "write a single paragraph, then improve it")."""
    if _SEQUENCE.search(sentence):
        return []
    statement = _is_statement(sentence)
    in_request = not statement and _is_strict_request(sentence)
    # "a function that takes a single word": counts of code, not of the response (except
    # "list N <things>", which is about the response: "give me three reasons to learn python").
    code = bool(_CODE_TOPIC.search(sentence)) and not _RESPONSE_SUBJ.search(sentence)
    found: list[_Cand] = []
    for a, b in _clauses(sentence):
        clause = sentence[a:b].strip()
        offset = a + (len(sentence[a:b]) - len(sentence[a:b].lstrip()))
        if statement and not _is_strict_request(clause):
            continue                     # a clause of a statement: "..., divided into 10 lines"
        for cand in _resolve(_count_candidates(clause, first_clause=a == 0, in_request=in_request)):
            if code and not cand.list_n:
                continue
            if cand.list_n and _another_quantity(sentence, offset + cand.end):
                continue                 # "list 9 discoveries, and also 4 breakthroughs"
            found.append(_Cand(cand.start + offset, cand.end + offset, cand.unit, cand.interval,
                               cand.list_n))
    found.sort(key=lambda x: x.start)
    drop = set()
    for i in range(len(found) - 1):          # "3 sentences or 2 paragraphs": alternatives
        if _OR_WORD.search(sentence[found[i].end:found[i + 1].start]):
            drop.update((i, i + 1))
    for i, cand in enumerate(found):
        if _OR_ALTERNATIVE.match(sentence, cand.end) or _OR_LIST.match(sentence, cand.end):
            drop.add(i)
    out = []
    for i, cand in enumerate(found):
        if i in drop or cand.interval is None or cand.unit is None:
            continue
        if inline_form and cand.list_n:
            continue
        out.append((cand.unit, cand.interval))
    return out


# ======================================================================== keyword rules
_KW_NOUN = r"(?:key[ -]?words?|key[ -]?phrases?|words?|phrases?|terms?|expressions?)"
_KW_VERB = (r"(?:include|includes|including|included|use|uses|using|used|mention|mentions|"
            r"mentioning|mentioned|contain|contains|containing|incorporate|incorporates|"
            r"incorporating|incorporated|feature|features|featuring|insert|add|adding|have|"
            r"has|having|put)")
_KW_STRONG = r"(?:include|includes|mention|mentions|incorporate|incorporates)"
_FORBID_VERB = (r"(?:avoid|avoids|avoiding|exclude|excludes|excluding|omit|omits|omitting|"
                r"refrain from(?: using)?|steer clear of)")
_DET = r"(?:(?:the|these|those|following|all|both|each|every|of|exact|specific|any|using|use) )*"
_ATOM_START = "(?=[\"“‘'\\[*])"
_K_LIST = re.compile(rf"\b{_KW_VERB} {_DET}{_KW_NOUN} ?:? ?{_ATOM_START}", re.I)
_K_STRONG = re.compile(rf"\b{_KW_STRONG} {_ATOM_START}", re.I)
_K_FORBID = re.compile(rf"\b{_FORBID_VERB} {_DET}(?:{_KW_NOUN} ?:? ?)?{_ATOM_START}", re.I)
_K_BARE = re.compile(rf"\b{_KW_VERB} (?:the )?(?:key[ -]?words?|keyword)(?: below)? ?:? (?=[a-z])", re.I)
_ATOM = re.compile(
    r'"(?P<a>[^"\n]{1,60})"|“(?P<b>[^”\n]{1,60})”|‘(?P<c>[^’\n]{1,60})’'
    r"|(?<![\w])'(?P<d>[^'\n]{1,60})'(?![\w])|\[(?P<f>[^\[\]\n]{1,80})\]"
    r"|\*\*(?P<g>[^*\n]{1,60})\*\*|\*(?P<h>[^*\n]{1,60})\*")
_K_SUBJ = re.compile(
    rf"\b(?:the )?{_KW_NOUN} (?:(?P<atom>{_ATOM.pattern})|(?P<bare>[a-z][a-z'-]*)) "
    r"(?:should|must|needs to|has to|is to|will|shall|is required to)(?: also)? (?P<neg>not )?"
    r"(?:appear|be (?:included|used|mentioned|present)|occur|show up|feature)\b", re.I)
# "[keywords] are "a", "b"": a bracketed template variable defined elsewhere in the message.
_PLACEHOLDER_DEF = re.compile(r"\[([^\[\]\n]{1,40})\] ?(?:is|are|=|:|-|will be|should be) ?:? ?", re.I)
# Bracketed names that are unfilled template variables, never keywords.
_TEMPLATE_NAME = re.compile(r"(?:key ?words?|key ?phrases?|words?|phrases?|terms?|.*_.*|.*[0-9].*)", re.I)
_LIST_SEP = re.compile(r" ?(?:, ?(?:and |& )?|and |& |as well as |plus )", re.I)
_LIST_OR = re.compile(r" ?,? ?(?:or|nor) ", re.I)
_BARE_WORD = re.compile(r"[a-z][a-z'-]{0,30}", re.I)
_BARE_END = re.compile(r" ?(?:in (?:the|your) (?:response|answer|reply|text|essay|story|poem|"
                       r"output|email|letter|post|summary|description|message))? ?[.!?;:]* ?\Z", re.I)
_KW_STOP = frozenset(
    "that which who whom whose related relevant such like about from some any the a an to for of "
    "in on with and or as is are be your you it this these those following below above "
    "appropriate specific important several many few other more most naturally throughout into "
    "at by when where how what why if not no all each every their its our my his her them they "
    "we he she there here than then also only just very well".split())
_KW_NEG_BEFORE = re.compile(r"(?:\b(?:do not|don't|dont|never|not|without|cannot|can't|should not|"
                            r"shouldn't|must not|mustn't|no|avoid|avoids|avoiding)) (?:[\w']+ ){0,2}\Z",
                            re.I)
_FREQ_NONE = re.compile(r" ?(?:less than|fewer than|at most|no more than|not more than|under|"
                        r"a maximum of|up to|maximum|zero times|0 times|no times)\b", re.I)


_ATOM_KIND = {"a": "quote", "b": "quote", "c": "quote", "d": "quote", "f": "bracket",
              "g": "star", "h": "star"}
_INNER_COMMA_SEP = re.compile(r" (?:and |& )?", re.I)     # '"a," "b," and "c"' (commas inside)


def _atom_value(m: re.Match) -> list[tuple[str, str]]:
    """``(keyword, delimiter kind)`` of a delimited atom; ``[a, b]`` and ``*a, b*`` are lists.
    Trailing punctuation inside the delimiters is dropped (``"fire."`` -> ``fire``)."""
    for g in _ATOM_KIND:
        v = m.group(g)
        if v is not None:
            kind = _ATOM_KIND[g]
            parts = [" ".join(p.split()) for p in (v.split(",") if kind in ("bracket", "star") else [v])]
            if len(parts) > 1 and any(p.lower().startswith("or ") for p in parts):
                return []                # "*a, b, or c*": alternatives
            parts = [re.sub(r"^(?:and|&) ", "", p, flags=re.I) if len(parts) > 1 else p for p in parts]
            out = [(p.rstrip(".,;:!?"), kind) for p in parts]
            return [(w, kind) for w, kind in out if w]
    return []


def _parse_atom_list(s: str, pos: int) -> tuple[list[tuple[str, str]], bool, int]:
    """Keywords of a delimited list starting at ``s[pos]``: ``(keywords, has_or, end)``."""
    out, has_or = [], False
    m = _ATOM.match(s, pos)
    if not m:
        return [], False, pos
    out += _atom_value(m)
    end = m.end()
    while True:
        inner_comma = len(m.group(0)) >= 3 and m.group(0)[-2] in ",;"
        sep = _LIST_SEP.match(s, end) or (_INNER_COMMA_SEP.match(s, end) if inner_comma else None)
        alt = _LIST_OR.match(s, end)
        nxt = None
        if alt:
            nxt = _ATOM.match(s, alt.end())
            if nxt:
                has_or = True
        if nxt is None and sep:
            nxt = _ATOM.match(s, sep.end())
        if nxt is None:
            return out, has_or, end
        out += _atom_value(nxt)
        m, end = nxt, nxt.end()


def _placeholder_table(text: str) -> dict[str, list[str] | None]:
    """Template variables ``[name]`` defined in ``text`` (``[keywords] are "a", "b"``): the
    quoted values of the definition, or ``None`` when it is not a quoted list."""
    table: dict[str, list[str] | None] = {}
    flat = " ".join(text.split())
    for m in _PLACEHOLDER_DEF.finditer(flat):
        words, has_or, _ = _parse_atom_list(flat, m.end())
        quoted = [w for w, kind in words if kind in ("quote", "code")]
        value = quoted if quoted and len(quoted) == len(words) and not has_or else None
        table.setdefault(m.group(1).strip().lower(), value)
    return table


def _parse_bare_list(s: str, pos: int) -> list[str] | None:
    """Plain single words ``A, B and C`` ending the sentence (``in the response`` allowed)."""
    words, end = [], pos
    m = _BARE_WORD.match(s, end)
    while m and len(words) < 6:
        words.append(m.group(0))
        end = m.end()
        sep = _LIST_SEP.match(s, end)
        if not sep:
            break
        m = _BARE_WORD.match(s, sep.end())
        if m:
            end = sep.end()
        else:
            break
    if not words or not _BARE_END.fullmatch(s, end):
        return None
    if any(w.lower() in _KW_STOP or w.lower().endswith("'") for w in words):
        return None
    return words


def _sentence_keywords(s: str, placeholders: dict[str, list[str] | None],
                       triggers: bool = True) -> tuple[list[str], list[str]]:
    """Required and forbidden keywords of one sentence (original case, quotes intact).
    ``placeholders`` maps template variables to their quoted values (see
    :func:`_placeholder_table`); a ``[name]`` keyword that is a variable is replaced by its
    values, or dropped when they are not a quoted list. Without ``triggers`` (the sentence is
    not a request) only the explicit "the word X should appear" form is read."""
    required: list[str] = []
    forbidden: list[str] = []

    def negated(start: int) -> bool:
        return bool(_KW_NEG_BEFORE.search(s[:start]))

    def freq_kills(end: int) -> bool:
        return bool(_FREQ_NONE.match(s, end))

    def values(atoms: list[tuple[str, str]]) -> list[str]:
        out = []
        for word, kind in atoms:
            key = word.lower()
            if kind == "bracket" and key in placeholders:
                out += placeholders[key] or []
            elif not (kind == "bracket" and _TEMPLATE_NAME.fullmatch(key)):
                out.append(word)
        return out

    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for m in _K_FORBID.finditer(s) if triggers else ():
        atoms, _, end = _parse_atom_list(s, m.end())
        if atoms and free(m.start(), end):
            forbidden += values(atoms)
            taken.append((m.start(), end))
    for pat in (_K_LIST, _K_STRONG) if triggers else ():
        for m in pat.finditer(s):
            atoms, has_or, end = _parse_atom_list(s, m.end())
            if not atoms or not free(m.start(), end):
                continue
            taken.append((m.start(), end))
            if negated(m.start()):
                forbidden += values(atoms)
            elif not has_or and not freq_kills(end):
                required += values(atoms)
    for m in _K_BARE.finditer(s) if triggers else ():
        if not free(m.start(), m.end()):
            continue
        words = _parse_bare_list(s, m.end())
        if words:
            taken.append((m.start(), len(s)))
            (forbidden if negated(m.start()) else required).extend(words)
    for m in _K_SUBJ.finditer(s):
        if not free(m.start(), m.end()):
            continue
        if m.group("atom"):
            am = _ATOM.match(m.group("atom"))
            words = values(_atom_value(am)) if am else []
        else:
            bare = m.group("bare")
            words = [] if bare is None or bare.lower() in _KW_STOP else [bare]
        if not words:
            continue
        taken.append((m.start(), m.end()))
        if m.group("neg") or negated(m.start()):
            forbidden += words
        elif not freq_kills(m.end()):
            required += words
    return required, forbidden


# ========================================================================= format rules
_J_VERB = (r"(?:answer|respond|reply|output|return|format|write|give|provide|present|structure|"
           r"wrap|put|express|deliver|generate|produce|display|show|send|encode|represent)")
_J_OBJ = (r"(?:it|this|that|them|me|us|back|the|your|my|its|entire|whole|full|final|complete|"
          r"answer|response|output|reply|result|results|data|information|info|list|summary|"
          r"everything|all|only|just|text|content|solution|values|items|details)")
_J_DET = (r"(?:a|an|the|valid|proper|strict|pure|plain|raw|correct|well-formed|properly|"
          r"formatted|structured|standard|single)")
_F_JSON = [
    re.compile(rf"(?:^|, |\band |\bthen |\bplease |\balso ){_LEAD}{_ASK}{_J_VERB}(?: {_J_OBJ})*? "
               rf"(?:in|as|using|with|into)(?: {_J_DET})* json\b"),
    re.compile(r"(?:^|\b(?:your|the) )(?:(?:entire|whole|full|complete|final) )?(?:response|answer|"
               r"output|reply|result)(?: [\w']+){0,2}? (?:should|must|needs? to|has to|have to|is to|"
               r"will|shall)(?: (?:only|also|strictly|always|entirely))? be(?: (?:in|a|an|the|valid|"
               r"proper|strict|formatted|written|wrapped|given|provided|returned|output|presented|"
               r"structured|as|in the form of|using|entirely|completely|pure))* json\b"),
    re.compile(r"^(?:(?:respond|answer|reply|output|return) )?(?:(?:in|with|as) )?(?:(?:valid|pure|"
               r"raw|strict) )?json(?: (?:format|object|output|response))?(?: only)?$"),
]
_CB_NOUN = (r"(?:(?:markdown |fenced |python |code )?code ?blocks?|code fences?|"
            r"fenced code(?: blocks?)?|triple backticks|backticks)")
_CB_VERB = (r"(?:put|wrap|place|format|write|give|provide|return|present|enclose|show|output|"
            r"respond|answer|reply|include|share|display|surround|deliver|print|list|send)")
_CB_OBJ = (r"(?:it|this|that|the|your|code|answer|response|output|result|solution|function|script|"
           r"program|query|snippet|everything|entire|whole|final|full|complete|me|all|of|example|"
           r"examples|commands?|program|implementation)")
_F_CODE = [
    re.compile(rf"(?:^|, |\band |\bthen |\bplease |\balso ){_LEAD}{_ASK}{_CB_VERB}(?: {_CB_OBJ})*? "
               rf"(?:in|inside|within|into|using|with|as)(?: (?:a|an|one|single|the|your|its|their|"
               rf"own|separate|proper|properly|formatted|markdown|python|triple|fenced))* {_CB_NOUN}\b"),
    re.compile(rf"(?:^|, |\band |\bthen |\bplease |\balso ){_LEAD}{_ASK}use (?:a |an |one |single )*"
               rf"(?:fenced |markdown )?(?:code ?blocks?|code fences?|triple backticks)\b"),
    re.compile(rf"(?:(?:your|the) )?(?:(?:entire|whole|full|complete|final) )?(?:code|answer|response|"
               rf"output|solution|script|function|reply)(?: [\w']+){{0,2}}? (?:should|must|needs? to|"
               rf"has to|is to|will) be (?:(?:in|inside|within|wrapped in|formatted as|formatted in|"
               rf"enclosed in|placed in|put in) )(?:a |an |one |single |the )*(?:fenced |markdown )?"
               rf"{_CB_NOUN}\b"),
]
_CODE_TOPIC = re.compile(r"\b(?:function|functions|program|script|code|method|class|regex|query|"
                         r"variable|variables|string|strings|sql|python|javascript|java|c\+\+|html|"
                         r"css|api|parser|converter|command|algorithm|array|arrays|dictionary|"
                         r"tuple|integer|integers|parameter|parameters|argument|arguments|file|"
                         r"files)\b")
# The response itself as the subject ("your entire response", "the answer"; not "the output" of a
# program or "the beginning of the text").
_RESPONSE_SUBJ = re.compile(r"\byour (?:entire |whole |full |complete )?(?:response|answer|reply|output|"
                            r"message|text)\b|\bthe (?:entire |whole |full |complete )?(?:response|"
                            r"answer|reply)\b")
_F_LOWER = [
    re.compile(r"\bin (?:all |only |entirely |completely |strictly )?lower ?case(?: letters| characters)?\b"),
    re.compile(r"\b(?:all|only|entirely|completely|strictly) lower ?case\b"),
    re.compile(r"\blower ?case (?:letters |characters )?only\b"),
    re.compile(r"\b(?:use|using|with|write|written|type|typed) (?:only |just |all )?lower ?case(?: letters| characters)?\b"),
]
_F_LOWER_NEG = [
    re.compile(r"\bno (?:capital|capitalized|uppercase|upper case) (?:letters|characters)\b|\bno capitals\b|"
               r"\bno capitalization\b"),
    re.compile(r"\b(?:do not|don't|dont|never|without) (?:use |using )?(?:any )?(?:capital|uppercase|upper case) "
               r"(?:letters|characters)\b|\b(?:do not|don't|dont|never) capitali[sz]e\b"),
]
_F_UPPER = [
    re.compile(r"\bin (?:all )?(?:capital letters|caps|upper ?case(?: letters)?|capitals|block letters)\b"),
    re.compile(r"\ball (?:caps|capital letters|capitals|upper ?case)\b"),
    re.compile(r"\b(?:capital|upper ?case) letters only\b|\bonly (?:capital|upper ?case) letters\b"),
    re.compile(r"\b(?:use|using|with|write|written|type|typed) (?:only |just |all )?(?:capital letters|"
               r"caps|upper ?case(?: letters)?)\b"),
]
_F_UPPER_NEG = [re.compile(r"\bno lower ?case (?:letters|characters)\b")]
_F_QUESTION = [
    re.compile(r"\b(?:end|finish|conclude|close|wrap up)(?: (?:it|this|your|the|entire|whole|response|"
               r"answer|reply|message|text|output|email|letter|post|story|poem))*(?: off)? (?:with|on|"
               r"by asking) (?:a |an |one )?(?:[\w-]+ )?question\b"),
    re.compile(r"\b(?:your|the) (?:response|answer|reply|message|text|output)(?: [\w']+)? (?:should|"
               r"must|needs to|has to) (?:end|finish|conclude) (?:with|on) (?:a |an )?(?:[\w-]+ )?question\b"),
    re.compile(r"\blast sentence (?:should|must) be a question\b"),
]
_F_NO_COMMAS = [
    re.compile(r"\b(?:do not|don't|dont|never|without|avoid|refrain from)(?: (?:using|use|the use of|"
               r"any|the|of))* commas?\b"),
    re.compile(r"\bno commas?\b(?! (?:after|before|between|around|inside|within|in the|in your|"
               r"in this|on)\b)"),
    re.compile(r"\bcommas? (?:are|is) (?:not allowed|forbidden|prohibited|not permitted)\b"),
]


_CONDITIONAL = re.compile(r"^(?:if|when|whenever|unless|in case|once|while)\b")
_CONVERSION = re.compile(r"\b(?:to|into|convert|converts|converting|lowercased|uppercased) \Z")
_CASE_FREQUENCY = re.compile(r"\bwords? (?:with|in|written in|that are in|using) (?:all )?(?:capital|"
                             r"caps|upper|lower)|\b(?:capital(?:ized)?|upper ?case|lower ?case|"
                             r"all[- ]caps) words\b|\b(?:appear|appears|occur|occurs)\b")


def _negated_before(s: str, start: int) -> bool:
    return bool(_NEG.search(s[max(0, start - 40):start].split(",")[-1]))


def _sentence_formats(s: str) -> dict:
    """Format requests of one normalised sentence, read in the whole sentence and in each of
    its clauses ("i have a question: use only lowercase letters"); none in a conditional
    sentence such as "when a response needs to be in JSON, ...")."""
    if _CONDITIONAL.match(s):
        return {}
    sentence_request = _is_request(s)
    code_topic = bool(_CODE_TOPIC.search(s)) and not _RESPONSE_SUBJ.search(s)
    out = _formats_of(s, sentence_request, code_topic, whole=True)
    for a, b in _clauses(s):
        clause = s[a:b].strip()
        if clause != s and not _CONDITIONAL.match(clause):
            part = _formats_of(clause, sentence_request or _is_request(clause), code_topic, whole=False)
            out["case"] = out.get("case", set()) | part.pop("case", set())
            out.update(part)
    if not out.get("case"):
        out.pop("case", None)
    return out


def _formats_of(s: str, request: bool, code_topic: bool, whole: bool) -> dict:
    """Format requests of one normalised sentence or clause ``s`` (``code_topic``: the sentence
    is about code, so case words describe the code's output; ``whole``: ``s`` is a whole
    sentence, the only place where the fragment "JSON only." is read)."""
    out: dict = {}
    json_patterns = () if code_topic else _F_JSON if whole else _F_JSON[:2]   # not a program's output
    for pat in json_patterns:
        m = pat.search(s)
        if m and not _negated_before(s, m.start()):
            out["json_output"] = True
    if request or _RESPONSE_SUBJ.search(s):
        for pat in _F_CODE:
            m = pat.search(s)
            if m and not _negated_before(s, m.start()):
                out["code_block"] = True
    cases = set()
    if not code_topic and not _CASE_FREQUENCY.search(s):
        if request or _RESPONSE_SUBJ.search(s) or len(s.split()) <= 6:
            for case, patterns in (("lower", _F_LOWER), ("upper", _F_UPPER)):
                for pat in patterns:
                    m = pat.search(s)
                    if (m and not _negated_before(s, m.start())
                            and not _CONVERSION.search(s, 0, m.start())):
                        cases.add(case)
        if any(p.search(s) for p in _F_LOWER_NEG):
            cases.add("lower")
        if any(p.search(s) for p in _F_UPPER_NEG):
            cases.add("upper")
    if cases:
        out["case"] = cases
    for pat in _F_QUESTION:
        m = pat.search(s)
        if m and not _negated_before(s, m.start()):
            out["end_question"] = True
    if any(p.search(s) for p in _F_NO_COMMAS):
        out["no_commas"] = True
    return out


# ============================================================================== compile
def _scan_region(text: str) -> str:
    if len(text) <= SCAN_CHARS:
        return text
    half = SCAN_CHARS // 2
    return text[:half] + "\n" + text[-half:]


@functools.lru_cache(maxsize=16384)
def _compile_cached(text: str) -> ConstraintSpec:
    text = _scan_region(text)
    code_masked = _mask_code(text)
    masked = _mask_quotes(code_masked)
    inline_form = bool(_INLINE_FORM.search(_norm(masked)))
    placeholders = _placeholder_table(code_masked)
    counts: dict[str, list[Interval]] = {}
    required: list[str] = []
    forbidden: list[str] = []
    flags: dict = {}
    cases: set[str] = set()
    for a, b in _sentences(masked):
        sentence = _norm(masked[a:b])
        for unit, iv in _sentence_counts(sentence, inline_form):
            counts.setdefault(unit, []).append(iv)
        code_task = _CODE_TOPIC.search(sentence) and not _RESPONSE_SUBJ.search(sentence)
        if not code_task and not _NOT_A_REQUEST.match(sentence):
            req, forb = _sentence_keywords(" ".join(code_masked[a:b].split()), placeholders,
                                           triggers=_is_strict_request(sentence))
            required += req
            forbidden += forb
        fmt = _sentence_formats(sentence)
        cases |= fmt.pop("case", set())
        flags.update(fmt)

    values: dict = {}
    for unit, ivs in counts.items():
        acc: Interval | None = ivs[0]
        for iv in ivs[1:]:
            acc = acc.intersect(iv) if acc is not None else None
        if acc is not None and _valid_interval(unit, acc):
            values[unit] = acc
    req_l, forb_l = {k.lower() for k in required}, {k.lower() for k in forbidden}
    both = req_l & forb_l

    def clean(words: list[str]) -> tuple[str, ...]:
        seen, out = set(), []
        for k in words:
            if k.lower() in both or k.lower() in seen or not valid_keyword(k):
                continue
            seen.add(k.lower())
            out.append(k)
        return tuple(out[:MAX_KEYWORDS])

    values["keywords"], values["forbidden"] = clean(required), clean(forbidden)
    if len(cases) == 1:
        values["case"] = cases.pop()
    values.update(flags)
    return ConstraintSpec(**values)


def compile(text: str) -> ConstraintSpec:  # noqa: A001  (the name is part of the spec)
    """Explicit, checkable constraints requested by a user message (rules in the module
    docstring); the empty spec when there are none."""
    if not isinstance(text, str) or not text.strip():
        return EMPTY
    return _compile_cached(text)


# =============================================================================== render
def _choice(rng: np.random.Generator, options: Sequence):
    return options[int(rng.integers(len(options)))]


def _num_text(n: int, rng: np.random.Generator, words_ok: bool = True) -> str:
    if words_ok and n in WORD_OF_NUMBER and (n <= 20 and rng.random() < 0.5 or n > 20 and rng.random() < 0.2):
        return WORD_OF_NUMBER[n]
    return str(n)


_ITEM_NOUNS = {
    None: (("list item", "list items"), ("bullet point", "bullet points"), ("bulleted item", "bulleted items")),
    "bullet": (("bullet point", "bullet points"), ("bullet point", "bullets"), ("bulleted item", "bulleted items")),
    "numbered": (("numbered item", "numbered items"), ("numbered point", "numbered points"),
                 ("list item", "list items")),
}


def _noun(unit: str, n: int, rng: np.random.Generator, style: str | None, plural: bool | None = None) -> str:
    plural = (n != 1) if plural is None else plural
    if unit == "items":
        sing, plur = _choice(rng, _ITEM_NOUNS[style if style in _ITEM_NOUNS else None])
    else:
        sing, plur = unit[:-1], unit
    return plur if plural else sing


def _adj_noun(unit: str, style: str | None, rng: np.random.Generator) -> str:
    if unit == "items":
        return _choice(rng, ("item", "bullet")) if style != "numbered" else "item"
    return unit[:-1]


_NP_FRAMES = ("Answer in {np}.", "Your response should contain {np}.", "Use {np}.",
              "Respond with {np}.", "Write {np}.", "Your answer must have {np}.",
              "Please answer with {np}.", "The response should be {np} long.", "{NP}, please.",
              "Make sure your answer has {np}.")
_STRONG_FRAMES = ("Answer in {np}.", "Respond in {np}.", "Please reply in {np}.",
                  "Explain it in {np}.", "Give your answer in {np}.")
_ADJ_FRAMES = ("Give a {adj} answer.", "Write a {adj} response.", "Provide a {adj} reply.",
               "Please give me a {adj} answer.")


def _np_exactly(unit: str, n: int, rng, style) -> list[str]:
    nt = _num_text(n, rng)
    noun = _noun(unit, n, rng, style)
    out = [f"exactly {nt} {noun}", f"precisely {nt} {noun}"]
    return out


def _render_count(unit: str, iv: Interval, rng: np.random.Generator, style: str | None) -> str:
    kind = iv.kind
    sentences: list[str] = []
    if unit == "items" and iv.hi == 0:
        return _choice(rng, ("Do not use any bullet points.", "Don't use bullet points or lists.",
                             "Avoid bullet points.", "Write in prose without bullet points.",
                             "No bullet points, please.", "Do not use lists."))
    if kind == "exactly":
        n = iv.lo
        nps = _np_exactly(unit, n, rng, style)
        sentences += [_choice(rng, _NP_FRAMES).replace("{np}", np_).replace("{NP}", np_.capitalize())
                      for np_ in nps]
        if unit != "words":
            bare = f"{_num_text(n, rng)} {_noun(unit, n, rng, style)}"
            sentences.append(_choice(rng, _NP_FRAMES).replace("{np}", bare).replace("{NP}", bare.capitalize()))
            sentences.append(_choice(rng, _STRONG_FRAMES).replace("{np}", bare))
            adj = f"{_num_text(n, rng) if n <= 20 else n}-{_adj_noun(unit, style, rng)}"
            frame = _choice(rng, _ADJ_FRAMES)
            if unit == "items":
                frame = _choice(rng, ("Answer with a {adj} list.", "Format your answer as a {adj} list."))
            sentences.append(frame.replace("{adj}", adj))
            if n == 1:
                single = f"a single {_noun(unit, 1, rng, style)}"
                sentences.append(_choice(rng, ("Answer in {s}.", "Use {s}.", "Respond with {s}.",
                                               "Write {s}.")).replace("{s}", single))
    elif kind == "at_most":
        n = iv.hi
        nt = _num_text(n, rng)
        noun = _noun(unit, n, rng, style)
        nps = [f"at most {nt} {noun}", f"no more than {nt} {noun}", f"not more than {nt} {noun}",
               f"a maximum of {nt} {noun}", f"up to {nt} {noun}", f"{nt} {noun} or fewer",
               f"{nt} {noun} or less", f"{nt} {noun} max", f"within {nt} {noun}"]
        if n + 1 >= 2:
            nt1 = _num_text(n + 1, rng)
            noun1 = _noun(unit, n + 1, rng, style)
            nps += [f"fewer than {nt1} {noun1}", f"less than {nt1} {noun1}", f"under {nt1} {noun1}"]
        np_ = _choice(rng, nps)
        sentences.append(_choice(rng, _NP_FRAMES).replace("{np}", np_).replace("{NP}", np_.capitalize()))
        sentences.append(_choice(rng, ("Keep your answer to {nt} {noun}.", "Limit your response to {nt} {noun}.",
                                       "Do not exceed {nt} {noun}.", "Don't write more than {nt} {noun}.",
                                       "Do not use more than {nt} {noun}.", "Your answer should not exceed {nt} {noun}."))
                         .replace("{nt}", nt).replace("{noun}", noun))
        if unit == "words":
            sentences.append(_choice(rng, _STRONG_FRAMES).replace("{np}", f"{nt} {noun}"))
            sentences.append(_choice(rng, _ADJ_FRAMES).replace("{adj}", f"{n}-word"))
    elif kind == "at_least":
        n = iv.lo
        nt = _num_text(n, rng)
        noun = _noun(unit, n, rng, style)
        nps = [f"at least {nt} {noun}", f"no fewer than {nt} {noun}", f"no less than {nt} {noun}",
               f"a minimum of {nt} {noun}", f"{nt} {noun} or more"]
        if n >= 2:
            nt1 = _num_text(n - 1, rng)
            noun1 = _noun(unit, n - 1, rng, style)
            nps += [f"more than {nt1} {noun1}", f"over {nt1} {noun1}"]
        np_ = _choice(rng, nps)
        sentences.append(_choice(rng, _NP_FRAMES).replace("{np}", np_).replace("{NP}", np_.capitalize()))
        sentences.append(_choice(rng, ("Do not write fewer than {nt} {noun}.", "Don't use less than {nt} {noun}.",
                                       "Write {n}+ {noun}."))
                         .replace("{nt}", nt).replace("{noun}", noun).replace("{n}", str(n)))
    else:
        a, b = iv.lo, iv.hi
        noun = _noun(unit, b, rng, style, plural=True)
        at, bt = _num_text(a, rng), _num_text(b, rng)
        nps = [f"between {at} and {bt} {noun}", f"{at} to {bt} {noun}", f"{a}-{b} {noun}",
               f"from {at} to {bt} {noun}", f"at least {at} and at most {bt} {noun}",
               f"no fewer than {at} and no more than {bt} {noun}"]
        if b == a + 1:
            nps.append(f"{at} or {bt} {noun}")
        np_ = _choice(rng, nps)
        sentences.append(_choice(rng, _NP_FRAMES).replace("{np}", np_).replace("{NP}", np_.capitalize()))
    return _choice(rng, sentences)


def _quote(k: str, rng: np.random.Generator) -> str:
    styles = ['"{}"', "'{}'", "“{}”"]
    if "'" in k:
        styles = ['"{}"', "“{}”"]
    return _choice(rng, styles).format(k)


def _join(items: list[str], rng: np.random.Generator, conj: str = "and") -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    last = _choice(rng, (f", {conj} ", f" {conj} "))
    return ", ".join(items[:-1]) + last + items[-1]


def _render_keywords(words: tuple[str, ...], rng: np.random.Generator) -> str:
    q = [_quote(k, rng) for k in words]
    lst = _join(q, rng)
    if len(words) == 1:
        frames = ("Include the word {l}.", "Make sure to use the word {l}.", "Mention {l} in your answer.",
                  "Your response must include the term {l}.", "Be sure to include {l}.",
                  "The word {l} should appear in your answer.", "Use the keyword {l} at least once.")
    else:
        frames = ("Include the keywords {l}.", "Use the words {l} in your answer.", "Mention {l} in your response.",
                  "Make sure to include the terms {l}.", "Your answer must contain the words {l}.")
    return _choice(rng, frames).replace("{l}", lst)


def _render_forbidden(words: tuple[str, ...], rng: np.random.Generator) -> str:
    q = [_quote(k, rng) for k in words]
    if len(words) == 1:
        frames = ("Do not use the word {l}.", "Avoid the word {l}.", "Don't mention {l}.",
                  "The word {l} should not appear in your answer.", "Never use the word {l}.",
                  "Do not include the keyword {l}.")
        return _choice(rng, frames).replace("{l}", q[0])
    frames = (("Do not use the words {l}.", "and"), ("Avoid the words {l}.", "or"),
              ("Don't mention {l}.", "or"), ("Do not include the keywords {l}.", "and"))
    frame, conj = _choice(rng, frames)
    return frame.replace("{l}", _join(q, rng, conj))


_JSON_T = ("Answer in JSON.", "Format your entire response as valid JSON.", "Respond only with a JSON object.",
           "Your entire output should be in JSON format.", "Return your answer as JSON.",
           "Please reply in valid JSON.")
_CODE_T = ("Put your answer in a code block.", "Wrap the code in a markdown code block.", "Use a fenced code block.",
           "Format the code inside a code block.", "Present your answer within a code block.")
_LOWER_T = ("Write your entire response in lowercase letters.", "Use only lowercase letters.",
            "Your answer must be in all lowercase, with no capital letters.", "Respond in all lowercase.",
            "No capital letters are allowed.")
_UPPER_T = ("Write your entire response in capital letters.", "Respond in all caps.", "Use only uppercase letters.",
            "Your entire answer should be in English and in all capital letters.")
_QUESTION_T = ("End your response with a question.", "Finish your answer with a question.",
               "Your reply should end with a question.", "Conclude with a follow-up question.")
_COMMAS_T = ("Do not use any commas.", "Avoid commas in your response.", "Write without using commas.",
             "Refrain from the use of any commas.", "No commas are allowed.")


def render(spec: ConstraintSpec, rng: np.random.Generator | None = None, *,
           item_style: str | None = None) -> str:
    """Natural instruction text for ``spec``: one sentence per constraint (keywords grouped),
    chosen among several templates and shuffled; ``compile(render(spec, rng)) == spec``.

    item_style: ``"bullet"``, ``"numbered"`` or ``None`` (neutral nouns) for item counts.
    """
    rng = np.random.default_rng() if rng is None else rng
    pieces = [_render_count(u, getattr(spec, u), rng, item_style)
              for u in COUNT_UNITS if getattr(spec, u) is not None]
    if spec.keywords:
        pieces.append(_render_keywords(spec.keywords, rng))
    if spec.forbidden:
        pieces.append(_render_forbidden(spec.forbidden, rng))
    for flag, options in (("json_output", _JSON_T), ("code_block", _CODE_T),
                          ("end_question", _QUESTION_T), ("no_commas", _COMMAS_T)):
        if getattr(spec, flag):
            pieces.append(_choice(rng, options))
    if spec.case:
        pieces.append(_choice(rng, _LOWER_T if spec.case == "lower" else _UPPER_T))
    order = rng.permutation(len(pieces))
    return " ".join(pieces[i] for i in order)


# ============================================================================== targets
def _display(ids: np.ndarray, tok) -> str:
    from bes.data.chat_format import display_text

    return display_text(ids, tok)


def targets_of_text(text: str) -> tuple[int, int, int]:
    """Countdown targets ``(sentences, items, words)`` requested by one user message."""
    return compile(text).targets()


def _row_targets(row: np.ndarray, new_seg: np.ndarray, tok, out: np.ndarray) -> None:
    n = row.shape[0]
    is_role = np.isin(row, ROLE_TOKENS)
    bound = is_role | (row == BOS) | new_seg
    bpos = np.flatnonzero(bound)
    for a in np.flatnonzero(row == ASSISTANT).tolist():
        k = int(np.searchsorted(bpos, a))          # bpos[k] == a
        stop = int(bpos[k + 1]) if k + 1 < bpos.size else n
        if new_seg[a] or k == 0:
            continue
        p = int(bpos[k - 1])
        if row[p] != USER:
            continue
        ends = np.flatnonzero(row[p + 1:a] == END)
        content = row[p + 1: p + 1 + int(ends[0])] if ends.size else row[p + 1:a]
        tgt = targets_of_text(_display(content, tok))
        if tgt != (-1, -1, -1):
            out[a:stop] = tgt


def targets_for_tokens(tokens, seg=None, tok=None) -> np.ndarray:
    """Countdown targets of a batch: ``int32[B, T, 3]`` (sentences, items, words; -1 = none).

    tokens: ``int[B, T]``; seg: ``int[B, T]`` segment ids or ``None`` (segments split by
    ``BOS`` only, as in ``registers.compute``); tok: the tokenizer used to decode the user
    turns (default: the committed one). Inside every assistant turn (from its ``ASSISTANT``
    token to the next ``BOS``, role token or segment start) the targets are those of the user
    turn right before it (its content up to its ``END``), when that user turn is in the same
    segment; -1 everywhere else.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be [B, T]")
    n_b, n_t = tokens.shape
    out = np.full((n_b, n_t, 3), -1, np.int32)
    if n_t == 0:
        return out
    if tok is None:
        from bes.tokenizer.tok import load

        tok = load()
    new_seg = np.zeros((n_b, n_t), bool)
    new_seg[:, 0] = True
    if seg is not None:
        seg = np.asarray(seg)
        if seg.shape != tokens.shape:
            raise ValueError(f"seg has shape {seg.shape}, tokens {tokens.shape}")
        new_seg[:, 1:] = seg[:, 1:] != seg[:, :-1]
    for b in range(n_b):
        if np.any(tokens[b] == ASSISTANT):
            _row_targets(tokens[b].astype(np.int64), new_seg[b], tok, out[b])
    return out


def compile_from_tokens(tokens, seg=None, tok=None) -> np.ndarray:
    """Alias of :func:`targets_for_tokens` (the name used in ``docs/SPEC.md``)."""
    return targets_for_tokens(tokens, seg, tok)


class IncrementalTargets:
    """The rows of :func:`targets_for_tokens` one token at a time, for ``batch_size`` streams.

    ``step(tokens, new_segment=None)`` consumes one token per stream (``int[B]``, or a scalar
    when ``batch_size == 1``) and returns the targets at its position, ``int32[B, 3]`` (or
    ``int32[3]``). The first token of a stream starts a segment; ``new_segment`` (bool ``[B]``)
    starts one before the token (a change of ``seg``). ``copy`` and ``select`` support particle
    decoding.
    """

    def __init__(self, batch_size: int = 1, tok=None):
        if tok is None:
            from bes.tokenizer.tok import load

            tok = load()
        self.tok = tok
        self.batch_size = int(batch_size)
        self._role = [0] * self.batch_size          # role token of the current turn (0: none)
        self._user: list[list[int] | None] = [None] * self.batch_size
        self._closed = [False] * self.batch_size    # the current user turn has seen its END
        self._target = np.full((self.batch_size, 3), -1, np.int32)
        self._fresh = [True] * self.batch_size

    def copy(self) -> "IncrementalTargets":
        return self.select(range(self.batch_size))

    def select(self, index) -> "IncrementalTargets":
        """New tracker whose streams are ``index`` of this one."""
        index = [int(i) for i in index]
        out = IncrementalTargets.__new__(IncrementalTargets)
        out.tok, out.batch_size = self.tok, len(index)
        out._role = [self._role[i] for i in index]
        out._user = [None if self._user[i] is None else list(self._user[i]) for i in index]
        out._closed = [self._closed[i] for i in index]
        out._target = self._target[index].copy()
        out._fresh = [self._fresh[i] for i in index]
        return out

    def _reset(self, b: int) -> None:
        self._role[b], self._user[b], self._closed[b] = 0, None, False
        self._target[b] = -1

    def step(self, tokens, new_segment=None) -> np.ndarray:
        tok_arr = np.asarray(tokens, dtype=np.int64)
        scalar = tok_arr.ndim == 0
        tok_arr = tok_arr.reshape(-1)
        if tok_arr.shape[0] != self.batch_size:
            raise ValueError(f"expected {self.batch_size} tokens, got {tok_arr.shape[0]}")
        fresh_seg = (np.zeros(self.batch_size, bool) if new_segment is None
                     else np.asarray(new_segment, dtype=bool).reshape(-1))
        out = np.full((self.batch_size, 3), -1, np.int32)
        for b, v in enumerate(tok_arr.tolist()):
            if self._fresh[b] or fresh_seg[b]:
                self._reset(b)
                self._fresh[b] = False
            if v == BOS:
                self._reset(b)
            elif v in ROLE_TOKENS:
                prev = self._role[b]
                self._target[b] = -1
                if v == ASSISTANT and prev == USER and self._user[b] is not None:
                    content = np.asarray(self._user[b], dtype=np.int64)
                    self._target[b] = targets_of_text(_display(content, self.tok))
                self._role[b] = v
                self._user[b], self._closed[b] = ([], False) if v == USER else (None, False)
            elif self._role[b] == USER and not self._closed[b]:
                if v == END:
                    self._closed[b] = True
                else:
                    self._user[b].append(v)
            if self._role[b] == ASSISTANT:
                out[b] = self._target[b]
        if scalar:
            if self.batch_size != 1:
                raise ValueError("a scalar token needs batch_size == 1")
            return out[0]
        return out


__all__ = ["COUNT_UNITS", "TARGET_UNITS", "Interval", "ConstraintSpec", "EMPTY", "measure",
           "json_valid", "count_keyword", "keyword_pattern", "valid_keyword", "merge", "compile",
           "verify", "render", "parse_count", "unit_of", "targets_of_text", "targets_for_tokens",
           "compile_from_tokens", "IncrementalTargets", "NUMBER_WORDS", "WORD_OF_NUMBER"]
