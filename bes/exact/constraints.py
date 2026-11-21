"""Constraints: exact response properties, a conservative compiler for explicit requests,
verifiers, natural instructions, countdown targets and token-local enforcement data
(``docs/SPEC.md`` section 2.2).

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
  byte that is not ASCII whitespace (a title or a postscript line counts like any other).
- ``paragraphs``: maximal runs of lines that are not separators; a separator is a blank line
  or a thematic break (``***``, ``---``, ``___``, spaces allowed) outside a code fence. A fenced
  block, blank lines included, belongs to the paragraph around it. **Divider mode**: when the
  text uses the markdown divider ``***`` outside fences (a line made of three or more ``*`` and
  spaces, or a run of three or more ``*`` with whitespace or the line edge on both sides),
  paragraphs are the non-empty chunks between dividers and blank lines do not separate.
  :func:`paragraph_texts` returns the paragraphs themselves; ``dividers`` counts the dividers,
  ``blank_paragraphs`` the paragraphs of blank-line mode, ``paragraph_first_words`` the first
  word of each paragraph and ``first_word`` that of the text (:func:`first_word`: after
  leading whitespace and ``# * _ > " ' ( [ “ ‘ «``, the first run of letters and digits,
  lowercased).
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
  Inner spaces of a keyword match any whitespace.
- ``highlights`` (:func:`count_highlights`): ``*text*`` / ``**text**`` spans on one line whose
  text has no ``*`` and does not start or end with whitespace (a ``* item`` bullet is not one).
  ``placeholders``: ``[text]`` spans on one line with non-blank text. ``capital_words``: words
  (letters and digits, inner apostrophes allowed) that are :meth:`str.isupper` (``"I"``
  counts). ``has_title``: a ``<<title>>`` on one line. ``quoted``: the stripped text starts and
  ends with ``"``.

Checkers used by :func:`verify` (all case-insensitive): :func:`letter_count` (an ASCII letter
in either case); :func:`starts_with_phrase` (after leading whitespace and markup, and after a
leading ``<<title>>`` or heading line, the phrase follows, not continued by a letter or digit);
:func:`ends_with_phrase` (the end, ignoring trailing whitespace, quotes and ``*`` / ``_``, is
the phrase, not preceded by a letter or digit; nothing else may follow); :func:`matches_option`
(the whole text equals an option once punctuation is removed and whitespace collapsed);
:func:`has_postscript` (the marker starts a line or a sentence; a space may follow each inner
``.``: ``P.S.`` matches ``P. S.``); :func:`count_sections` (lines starting with ``<marker> <n>``,
optionally after a heading ``#`` run or emphasis); :func:`split_responses` (non-blank parts
between separators; blank parts only at the ends); :func:`repeats_request` (the text starts with
the request, whitespace collapsed).

Specification (:class:`ConstraintSpec`)
---------------------------------------
Counts are intervals ``Interval(lo, hi)`` (``hi = None``: no upper bound) on ``words,
sentences, items, paragraphs, lines``; ``keywords`` must each appear (word-start match),
``forbidden`` must not (whole-word match: lenient towards the response in both cases);
``json_output`` (``json_valid``), ``code_block`` (at least one closed fenced block), ``case``
(``"lower"``: ``all_lower``, ``"upper"``: ``all_upper``), ``end_question``
(``ends_with_question``), ``no_commas`` (no ``,``). ``items = Interval(0, 0)`` means "no list".
Further instruction families:

==================  ===========================================================================
field               satisfied when
==================  ===========================================================================
``list_style``      ``"bullet"`` / ``"numbered"``: there is a list item and every one has that
                    marker (``- * + •`` or ``N.`` / ``N)``)
``frequencies``     ``((word, Interval), ...)``: whole-word occurrences in the interval (the
                    project's eval scorer counts the same way); "at least once" is a keyword
``letters``         ``((letter, Interval), ...)``: occurrences of the letter in either case
                    (``Interval(0, 0)``: "without the letter e")
``highlights``,     ``Interval`` on ``measure()`` of the same name
``placeholders``,
``capital_words``
``sections``        ``(marker, Interval)``: lines starting with ``<marker> <number>``
``first_word``      ``(n, word)``: paragraph ``n`` starts with ``word`` (lowercase)
``divider``         paragraph breaks are ``***`` dividers (a text of several blank-line
                    paragraphs without a divider fails)
``title``           a ``<<title>>``
``postscript``      a postscript marker, e.g. ``"P.S."``, at a line or sentence start
``quoted``          the whole response is in double quotes
``starts_with``     options; the response starts with one of them
``end_phrase``      the response ends with the phrase exactly (case and markup aside)
``one_of``          options; the whole response is one of them (punctuation and case aside)
``responses``       ``(separator, n)``: exactly ``n`` different non-blank parts between
                    separators
``repeat``          the request (the rest of the user message); the response starts with it
==================  ===========================================================================

Word and letter pairs are kept sorted by word (canonical order), start options carry no final
sentence punctuation, and every phrase is one line without double quotes. :func:`merge`
intersects counts and options; conflicting markers, phrases or styles make it return ``None``.

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
  "exactly" or a negated bare count gives nothing. ``about / around / roughly / or so /
  hopefully / if necessary`` give nothing.
- Bare counts (no relation word) are exact for sentences, items, paragraphs and lines (``in
  three sentences``, ``a one-sentence answer``, ``a single paragraph``, ``your response must
  have 2 paragraphs``, ``write 4 lines``). For words, a bare count up to ``BARE_EXACT_WORDS``
  (5) is exact (``a single word``, ``in three words``, ``answer with one word``) and a larger
  one is a budget (``in 50 words``, ``a 100-word summary`` -> ``[0, N]``). Bare word counts
  need ``in / within / using / as``, ``answer / reply / respond with`` (small counts) or the
  adjectival form (``give me 10 words that rhyme`` is not a length).
- Steps are list items (``explain it in 4 steps``, ``in exactly three numbered steps``, ``give
  me 5 steps``) in a request with ``in / into / using / with`` or a relation. A list style is
  read from the count's noun (``bullet points`` -> bullet, ``numbered items / steps`` ->
  numbered) and from format phrases (``as a numbered list``, ``in bullet points``, ``use the
  markdown bullet points``); both styles, or a style alternative, give none.
- Licensing. A count is read in a *request*: a clause that starts with a request verb
  (possibly after "please", "can you", "make sure to", ...), or contains "should / must /
  needs to / is required", or starts with "I want / I need / I'd like", or is a predicate of a
  requirement list ("- contains at least 5 sentences"), or belongs to a request sentence. A
  relation or a bare count after ``in / into / using / within / as`` and a fragment that opens
  its clause (``Max 3 sentences.``, ``One word.``) are also read. Weak introducers (``with / of
  / contain / have / be / write / give / use ...``) need a request; ``with`` and ``of`` also need
  a text noun before them (``a poem of 4 lines``, not ``the intersection of two lines``).
  Adjectival forms need a production verb before and a text noun after (``write a 3-sentence
  summary``, not ``describe a 5-paragraph essay structure``). "I should / we need to" are the
  user's own deliberation, not modals about the reply.
- Not requests: statements (a clause or sentence that starts with a subject, "here is", "I have
  ...", "I need to ...", or a quoted span, without a request: ``I wrote a 300-word essay``);
  informational, hypothetical and conditional questions (``how do you ...``, ``do you have
  ...``, ``what if ...``, ``if / when ...``) and examples (``for example ...``); counts in
  sentences about code and all counts and formats except code blocks in a message that asks
  for code (``write a function ... provide 2 solutions``); per-part counts (``each``, ``per``,
  ``every``, ``for each``, ``sentences must have fewer than 10 words``); counts of an edit or of
  a text being worked on (``by removing at most three words``, ``while editing this paragraph
  into 3 sentences``, ``help on improving a single sentence``, ``with at most 4 words added``,
  ``replace X with a single word``); counts of a part of the reply (``begin your answer with a
  1 sentence summary``, ``the revised passage``, ``an ending sentence``, ``the second
  sentence``, ``a 5-word title``); comparisons (``one item shorter``); counts after a
  determiner (``in this one sentence``); counts inside an embedded question (``decide if ...
  in one paragraph``); counts in multi-step sentences (``then``, ``followed by``, ``first``,
  ``and explain``), which also block that unit for the whole message, as do additive counts
  (``in 120 more words``) and two bare counts of one unit in a sentence (``4 clear sentences
  and four unclear sentences``); a conditional clause with its own count drops the sentence's
  counts.
- ``list|name|give me|suggest|provide|... N <things>`` (2 <= N <= ``MAX_LIST_N``) opening a
  request sentence gives ``items = [N, N]``; so do ``a list of N <things>`` and, with a list
  format in the sentence, ``pick / choose / select N (of them)``. Not when the message asks
  for an inline form (comma-separated, one sentence, one line), another quantity follows in the
  sentence (``three pros and three cons``), the count is per part or the noun is a measure or a
  count of responses (``give two different responses``). The same count inside a question
  ("what are three reasons ...?") is a soft request and gives nothing.
- Keywords must be delimited by quotes, ``[...]`` or ``*...*`` after a trigger (``include /
  use / mention ... the word(s) / keyword(s) / phrase(s) / term(s)``, ``include "x"``, ``the
  word X should appear``, ``the words "a" and "b" should not appear``) in a request sentence,
  or be plain words or comma-separated phrases of up to three words in the template ``include
  (the) keyword(s) A, B and C (in the response).``. ``[a, b]`` and ``*a, b*`` are lists; a
  bracketed template variable defined in the message (``[keywords] are "a", "b"`` or ``[name]
  are: a, b, c``) is replaced by its values, and generic names (``[keyword]``, names with
  ``_`` or digits) are dropped. ``X or Y`` lists are not required keywords. Negated triggers
  (``do not include / avoid / without using / never say / should not appear``) give
  ``forbidden`` (``or`` lists included). ``the use of the word X`` and ``you used the word X``
  are not triggers. Nothing is read in sentences about code.
- Formats: ``answer / respond / format ... in|as JSON``, ``your response must be in JSON``,
  ``give me a JSON object ...``, ``output only the JSON``, ``JSON only``; ``in a code block`` /
  ``use a code block`` / ``wrap ... in triple backticks`` (in a request); ``all lowercase / in
  lowercase letters / no capital letters``; ``all caps / in capital letters`` (not "convert to
  lowercase", not "named in all caps", not a short statement such as "this is all lower
  case"); ``end (your response) with a question``, ``close by asking me a question``, ``make
  sure the last sentence is a question``; ``do not use any commas / no commas / must not
  contain commas / comma-free`` (not in talk about comma rules). Nothing in a conditional
  sentence.
- Further families (claimed spans are masked before the count, keyword and format rules read
  the sentence, so "highlight at least 2 words" is not a word count): ``the word X should
  appear / use the word X / mention "X" <relation> N times | once | twice`` (a negation flips
  the relation; ``less than once`` is ``forbidden``); ``the letter X should appear ... N
  times``, ``use the letter X ...``, ``without the letter X``; ``highlight <relation> N
  sections ... (with markdown)``, ``N highlighted sections``, ``N highlights using markdown``
  (a bare count is a lower bound; ``highlight X using markdown`` is at least one); ``<relation>
  N placeholders`` with square brackets mentioned; ``words with all capital letters should
  appear ... N times``, ``use N words in all caps`` (bare: at least); ``must have N sections``
  together with ``mark the beginning of each section with <Marker> X``; ``Paragraph N must
  start with word W``, ``the second paragraph should start with the word "W"``; ``paragraphs
  are separated with the markdown divider: ***``; ``title ... double angular brackets / <<``;
  ``postscript (starting with M / should be "...")``, ``add a P.S.`` (default marker
  ``P.S.``); ``wrap your entire response in double quotation marks`` and other forms with the
  response (or a text noun) as object or subject; ``start / begin your response with "X" (or
  "Y")``, ``it must start with the word "X"``, ``start with yes or no`` (not "let's start
  with", not "sentences that start with", not in a multi-step sentence unless the response is
  named); ``finish your response with this exact phrase X`` (quoted, bracketed or a final
  ``?`` / ``!`` phrase), ``end your answer with the phrase "X"``; ``answer with one of the
  following options: ...`` (quoted, bracketed, parenthesised, ``- a - b`` or plain lists, a
  template variable resolved), ``answer with yes or no only``, ``a one-word answer: yes, no,
  or maybe`` (a restriction word is required); ``give two different responses ... separated
  by ******``; ``first repeat the request word for word without change, then give your answer
  (...)`` (the request is the message without that instruction). Template phrases that are
  undefined variables (``[ender]``) or contain placeholders give nothing.
- Several constraints on one count are intersected (``at least 3 and at most 5 sentences``);
  an empty intersection drops that count. Two count requests joined by ``or``, or a count
  followed by an ``or`` alternative (``a single sentence, a paragraph, or an essay``), are
  dropped. Inconsistent families are dropped newest first.

Rendering (:func:`render`)
--------------------------
One sentence (two for a "between" frequency or count of a feature) per constraint, from several
templates, shuffled; ``compile(render(spec)) == spec`` (property-tested). Item nouns follow
``spec.list_style`` ("bullet points", "numbered items"; "list items" without a style). A spec
with ``repeat`` renders as its request followed by the repeat instruction, so its other
constraints must be the request's own.

Token-local enforcement (:meth:`ConstraintSpec.token_local`)
------------------------------------------------------------
``spec.token_local()`` returns a :class:`TokenLocal`: what a decoder can enforce at every token
and what only the complete text can show.

- ``counts``: ``(counter, lo, hi)`` for ``words``, ``sentences`` and ``items``: the raw counters
  of ``bes.exact.registers.Incremental`` (``raw()[counter]``), the very counts ``verify`` uses.
  They never decrease inside a turn, so a candidate token whose increment
  (``table.inc[tok, s, INC_WORDS | INC_SENT | INC_ITEMS]`` with ``s = raw()["state"]``) takes the
  counter above ``hi`` can be masked; ``END`` is allowed only when ``lo <= count <= hi`` after
  its flush (``END`` adds one sentence when ``table.components[s, 5] == 1``). Exact counts are
  both.
- ``lines``: ``(lo, hi)`` on non-blank lines: not a register (the register counts ``\\n``) but
  a byte count with one bit of state, :func:`nonblank_line_step`, tabulable per token and
  entry bit.
- ``banned_bytes`` (``b","`` for no commas) and ``case`` (``"lower"`` bans tokens with an ASCII
  capital, ``"upper"`` tokens with an ASCII lowercase letter): static token masks; non-ASCII
  letters and "at least one cased letter" are checked at the end.
- ``forbidden``: lowercase UTF-8 forms of the forbidden words; :func:`has_forbidden_bytes`
  finds a complete occurrence (ASCII-case-insensitive, no word byte before or after: exact for
  ASCII neighbours). ``letters``: ``(letter bytes, lo, hi)``, a per-token byte count.
- ``complete``: the descriptions (``ConstraintSpec.constraints``) of everything else:
  paragraphs, keywords, frequencies, list style, JSON, code blocks, questions, highlights,
  placeholders, capital words, sections, first words, dividers, titles, postscripts, quotes,
  start and end phrases, options, separated responses, repetition. Start phrases, options and
  the repeated request are prefixes a decoder may also check early on the decoded text.

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

import dataclasses
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


def _line_stats(text: str) -> dict:
    """Line-level properties; ``par_texts`` lists the paragraphs (``paragraphs`` of them) in
    the mode the text uses (blank lines, or ``***`` dividers when there is one), ``dividers``
    counts the dividers and ``blank_paragraphs`` the paragraphs of blank-line mode."""
    lines = text.split("\n")
    fence = False
    stats: dict = dict(lines=0, paragraphs=0, headings=0, code_fences=0, bullet_items=0,
                       numbered_items=0)
    in_par = False
    pars: list[list[str]] = []                    # blank-line mode
    chunks: list[list[str]] = []                  # "***"-divided chunks (divider mode)
    in_chunk, dividers = False, 0
    for line in lines:
        body = line.rstrip("\r")
        blank = not line.strip(_LINE_WS)
        if not blank:
            stats["lines"] += 1
        if not fence and (blank or _BREAK_LINE.match(body)):
            in_par = False
        else:
            if not in_par:
                pars.append([])
                in_par = True
            pars[-1].append(body)
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
                    chunks.append([])
                    in_chunk = True
                if in_chunk:
                    chunks[-1].append(piece)
        if _FENCE_LINE.match(line):
            stats["code_fences"] += 1
            fence = not fence
    stats["blank_paragraphs"] = len(pars)
    stats["dividers"] = dividers
    stats["par_texts"] = ["\n".join(p) for p in (chunks if dividers else pars)]
    stats["paragraphs"] = len(stats["par_texts"])
    return stats


def paragraph_texts(text: str) -> list[str]:
    """The paragraphs counted by ``measure(text)["paragraphs"]``, in order (blank-line mode,
    or divider mode when the text has a ``***`` divider; module docstring)."""
    return _line_stats(text)["par_texts"]


# ------------------------------------------------------------- spans, words and phrases
_HIGHLIGHT = re.compile(r"(?<!\*)(\*\*?)(?![\s*])[^*\n]*?(?<![\s*])\1(?!\*)")
_PLACEHOLDER = re.compile(r"\[[^\[\]\n]*[^\s\[\]][^\[\]\n]*\]")
_TITLE = re.compile(r"<<[^<>\n]*[^\s<>][^<>\n]*>>")
_WORD_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_FIRST_WORD = re.compile(r"[^\W_]+")
_PUNCT = re.compile(r"[^\w\s]")
_TRAILING_END = _ASCII_WS + "\"'*_”’"


def count_highlights(text: str) -> int:
    """Markdown highlights: ``*text*`` or ``**text**`` spans on one line whose text does not
    start or end with whitespace and has no ``*`` (``"* item"`` bullets are not spans)."""
    return len(_HIGHLIGHT.findall(text))


def count_placeholders(text: str) -> int:
    """Square-bracket placeholders ``[like this]`` on one line with non-blank content."""
    return len(_PLACEHOLDER.findall(text))


def has_title(text: str) -> bool:
    """A title in double angular brackets ``<<like this>>`` (one line, non-blank)."""
    return _TITLE.search(text) is not None


def is_quoted(text: str) -> bool:
    """The text, stripped of ASCII whitespace, starts and ends with ``"`` (two at least)."""
    t = text.strip(_ASCII_WS)
    return len(t) >= 2 and t[0] == '"' and t[-1] == '"'


def capital_word_count(text: str) -> int:
    """Words (runs of letters and digits, inner apostrophes allowed) that are all capitals
    (:meth:`str.isupper`: at least one cased letter, none lowercase; ``"I"`` counts)."""
    return sum(1 for w in _WORD_TOKEN.findall(text) if w.isupper())


def letter_count(text: str, letter: str) -> int:
    """Occurrences of an ASCII letter in either case."""
    return text.count(letter.lower()) + text.count(letter.upper())


def first_word(text: str) -> str:
    """First word of ``text`` lowercased: after leading whitespace and the markup ``# * _ >
    " ' ( [ “ ‘ «``, the first run of letters and digits (``""`` when none)."""
    m = _FIRST_WORD.match(text.lstrip(_LEADING_MARKUP))
    return m.group(0).lower() if m else ""


def _ws(s: str) -> str:
    return " ".join(s.split())


@functools.lru_cache(maxsize=4096)
def _phrase_pattern(phrase: str) -> re.Pattern:
    body = r"\s+".join(re.escape(w) for w in phrase.split())
    tail = r"(?![^\W_])" if phrase[-1:].isalnum() else ""
    return re.compile(body + tail, re.IGNORECASE)


_HEAD_LINES = re.compile(r"(?:[ \t]*(?:<<[^<>\n]*>>|#{1,6}[ \t][^\n]*)[ \t]*(?:\n|\Z)[ \t\n]*)+")


def starts_with_phrase(text: str, phrase: str) -> bool:
    """``text`` begins with ``phrase`` (case-insensitive, inner whitespace flexible, not
    followed by a letter or digit when it ends with one) after leading whitespace and markup
    (as :func:`first_word`); a leading ``<<title>>`` or markdown heading line is skipped (the
    start of the body counts)."""
    if not phrase.split():
        return False
    pattern = _phrase_pattern(phrase)
    head = text.lstrip(_ASCII_WS)
    if pattern.match(head.lstrip(_LEADING_MARKUP)):
        return True
    m = _HEAD_LINES.match(head)
    return bool(m) and m.end() < len(head) and pattern.match(head[m.end():].lstrip(_LEADING_MARKUP)) is not None


def ends_with_phrase(text: str, phrase: str) -> bool:
    """``text`` ends with ``phrase``: both casefolded with whitespace runs collapsed and the
    trailing whitespace, quotes and emphasis ``" ' * _ ” ’`` removed; the text may not have a
    letter or digit right before the phrase when the phrase starts with one."""
    t = _ws(text).casefold().rstrip(_TRAILING_END)
    p = _ws(phrase).casefold().rstrip(_TRAILING_END)
    if not p or not t.endswith(p):
        return False
    before = t[: len(t) - len(p)]
    return not (p[0].isalnum() and before[-1:].isalnum())


def option_key(s: str) -> str:
    """Comparison form of an answer option: casefolded, punctuation removed, whitespace
    collapsed (``"My answer is yes."`` -> ``"my answer is yes"``)."""
    return _ws(_PUNCT.sub("", s.casefold()))


def matches_option(text: str, options: Sequence[str]) -> bool:
    """The whole text is one of ``options`` (compared by :func:`option_key`)."""
    key = option_key(text)
    return bool(key) and key in {option_key(o) for o in options}


@functools.lru_cache(maxsize=1024)
def _marker_pattern(marker: str) -> str:
    out = []
    for i, ch in enumerate(marker):
        if ch.isspace():
            if not out or out[-1] != r"\s+":
                out.append(r"\s+")
        else:
            out.append(re.escape(ch))
            if ch == "." and i < len(marker) - 1 and not marker[i + 1].isspace():
                out.append(r"[ \t]?")
    return "".join(out)


@functools.lru_cache(maxsize=1024)
def _postscript_re(marker: str) -> re.Pattern:
    return re.compile(r"(?:^[ \t]*|(?<=[.!?\"”)])[ \t]+)(?:[*_>#]+[ \t]*)?" + _marker_pattern(marker),
                      re.IGNORECASE | re.MULTILINE)


def has_postscript(text: str, marker: str) -> bool:
    """``marker`` at the start of a line (after spaces and the markup ``* _ > #``) or of a
    sentence (after ``. ! ? " ” )`` and a space), case-insensitive, where a space may follow
    every inner ``.`` (``"P.S."`` matches ``"P. S."``, ``"p.s."``)."""
    return bool(marker.strip()) and _postscript_re(marker.strip()).search(text) is not None


@functools.lru_cache(maxsize=1024)
def _section_re(marker: str) -> re.Pattern:
    return re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,2})?" + _marker_pattern(marker)
                      + r"[ \t]*[0-9]+(?![0-9])", re.IGNORECASE | re.MULTILINE)


def count_sections(text: str, marker: str) -> int:
    """Lines that start (after spaces, an optional heading ``#`` run and ``*``/``_``
    emphasis) with ``marker`` (case-insensitive) and a number: ``"Section 2"``,
    ``"## SECTION 2: ..."``, ``"**Part 3**"``."""
    return len(_section_re(marker).findall(text)) if marker.strip() else 0


def split_responses(text: str, separator: str) -> tuple[list[str], bool]:
    """``(parts, clean)``: the non-blank parts of ``text`` between ``separator`` (stripped)
    and whether blank parts occur only at the two ends."""
    parts = text.split(separator)
    clean = all(p.strip() for p in parts[1:-1])
    return [p.strip() for p in parts if p.strip()], clean


def repeats_request(text: str, request: str) -> bool:
    """``text`` starts with ``request`` (whitespace runs collapsed, casefolded)."""
    r = _ws(request).casefold()
    return bool(r) and _ws(text).casefold().startswith(r)


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
        "dividers": stats["dividers"],
        "blank_paragraphs": stats["blank_paragraphs"],
        "paragraph_first_words": [first_word(p) for p in stats["par_texts"]],
        "first_word": first_word(text),
        "highlights": count_highlights(text),
        "placeholders": count_placeholders(text),
        "capital_words": capital_word_count(text),
        "has_title": has_title(text),
        "quoted": is_quoted(text),
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


FLAG_FIELDS = ("json_output", "code_block", "end_question", "no_commas", "divider", "title",
               "quoted")
FEATURE_COUNTS = ("highlights", "placeholders", "capital_words")
LIST_STYLES = ("bullet", "numbered")
MAX_PHRASE = 120
MAX_REPEAT = 2000
_PHRASE_OK = re.compile(r"[^\n\r\t]*[^\W_][^\n\r\t]*")
_MARKER_OK = re.compile(r"[A-Za-z][A-Za-z0-9 .]{0,19}")
_SEPARATOR_OK = re.compile(r"[*\-=_~#|+]{3,20}")


def valid_phrase(p: str, limit: int = MAX_PHRASE) -> bool:
    """A phrase (start, end, option, postscript marker): one line of 1-``limit`` characters
    with a letter or digit, no leading or trailing whitespace and no double space."""
    return (isinstance(p, str) and 0 < len(p) <= limit and p == p.strip() and "  " not in p
            and bool(_PHRASE_OK.fullmatch(p)) and not any(q in p for q in "\"“”"))


def _check_feature(name: str, iv) -> None:
    if iv is None:
        return
    if not isinstance(iv, Interval):
        raise TypeError(f"{name} must be an Interval or None")
    if iv.hi is None and iv.lo < 1 or iv.hi == 0 or (iv.hi if iv.hi is not None else iv.lo) > MAX_COUNT:
        raise ValueError(f"{name}: {iv} is not a checkable request")


def _check_word_counts(name: str, value, *, letter: bool) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of (word, Interval) pairs")
    seen = set()
    for pair in value:
        if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], Interval)):
            raise ValueError(f"{name}: bad entry {pair!r}")
        w, iv = pair
        ok = (isinstance(w, str) and len(w) == 1 and "a" <= w <= "z") if letter else valid_keyword(w)
        if not ok:
            raise ValueError(f"{name}: bad word {w!r}")
        if iv.hi is None and iv.lo < 1 or (iv.hi == 0 and not letter):
            raise ValueError(f"{name}: {iv} for {w!r} is not a checkable request")
        if not letter and iv == Interval(1, None):
            raise ValueError(f"{name}: 'at least once' is a required keyword ({w!r})")
        if (iv.hi if iv.hi is not None else iv.lo) > MAX_COUNT:
            raise ValueError(f"{name}: {iv} is too large")
        if w.lower() in seen:
            raise ValueError(f"{name} has duplicates: {value!r}")
        seen.add(w.lower())


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
    list_style: str | None = None    # "bullet" | "numbered": every list item has that marker
    frequencies: tuple[tuple[str, Interval], ...] = ()   # (word, whole-word occurrences)
    letters: tuple[tuple[str, Interval], ...] = ()       # (ascii letter, occurrences)
    highlights: Interval | None = None
    placeholders: Interval | None = None
    capital_words: Interval | None = None
    sections: tuple[str, Interval] | None = None         # (marker, lines "<marker> <n>")
    first_word: tuple[int, str] | None = None            # (paragraph number, lowercase word)
    divider: bool = False            # paragraphs are separated by *** dividers
    title: bool = False              # a <<title>>
    postscript: str | None = None    # marker of a postscript line, e.g. "P.S."
    quoted: bool = False             # the whole response in double quotes
    starts_with: tuple[str, ...] = ()   # the response starts with one of these
    end_phrase: str | None = None
    one_of: tuple[str, ...] = ()        # the whole response is one of these
    responses: tuple[str, int] | None = None   # (separator, n different responses)
    repeat: str | None = None        # the request to repeat word for word first

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
        forbidden = {k.lower() for k in self.forbidden}
        if {k.lower() for k in self.keywords} & forbidden:
            raise ValueError("a keyword cannot be both required and forbidden")
        if self.case not in (None, "lower", "upper"):
            raise ValueError(f"case must be None, 'lower' or 'upper', got {self.case!r}")
        if self.list_style not in (None, *LIST_STYLES):
            raise ValueError(f"list_style must be None, 'bullet' or 'numbered', got {self.list_style!r}")
        if self.list_style and self.items is not None and self.items.hi == 0:
            raise ValueError("a list style needs list items")
        _check_word_counts("frequencies", self.frequencies, letter=False)
        _check_word_counts("letters", self.letters, letter=True)
        for name in ("frequencies", "letters"):       # canonical order: by word
            object.__setattr__(self, name, tuple(sorted(getattr(self, name), key=lambda p: (p[0].lower(), p[0]))))
        if {w.lower() for w, _ in self.frequencies} & forbidden:
            raise ValueError("a word cannot have a frequency and be forbidden")
        for name in FEATURE_COUNTS:
            _check_feature(name, getattr(self, name))
        if self.sections is not None:
            ok = (isinstance(self.sections, tuple) and len(self.sections) == 2
                  and isinstance(self.sections[0], str) and bool(_MARKER_OK.fullmatch(self.sections[0]))
                  and self.sections[0] == self.sections[0].strip()
                  and isinstance(self.sections[1], Interval) and self.sections[1].hi != 0)
            if not ok:
                raise ValueError(f"sections must be (marker, Interval), got {self.sections!r}")
            _check_feature("sections", self.sections[1])
        if self.first_word is not None:
            fw = self.first_word
            ok = (isinstance(fw, tuple) and len(fw) == 2 and isinstance(fw[0], int)
                  and 1 <= fw[0] <= 50 and isinstance(fw[1], str)
                  and bool(_FIRST_WORD.fullmatch(fw[1])) and fw[1] == fw[1].lower() and len(fw[1]) <= 30)
            if not ok:
                raise ValueError(f"first_word must be (paragraph number, lowercase word), got {fw!r}")
        for name in FLAG_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.postscript is not None and not (valid_phrase(self.postscript, 80) and self.postscript[0].isalpha()):
            raise ValueError(f"bad postscript marker {self.postscript!r}")
        if self.end_phrase is not None and not valid_phrase(self.end_phrase):
            raise ValueError(f"bad end phrase {self.end_phrase!r}")
        for name, top in (("starts_with", 4), ("one_of", 10)):
            opts = getattr(self, name)
            if not isinstance(opts, tuple) or len(opts) > top or not all(valid_phrase(o) for o in opts):
                raise ValueError(f"{name} must be a tuple of at most {top} phrases, got {opts!r}")
            keys = [option_key(o) if name == "one_of" else _ws(o).casefold() for o in opts]
            if len(set(keys)) != len(keys) or not all(keys):
                raise ValueError(f"{name} has duplicate or empty options: {opts!r}")
            if name == "starts_with" and any(o[-1] in ".,;:…" for o in opts):
                raise ValueError(f"starts_with options end without sentence punctuation: {opts!r}")
        if self.responses is not None:
            r = self.responses
            ok = (isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str)
                  and bool(_SEPARATOR_OK.fullmatch(r[0])) and isinstance(r[1], int) and 2 <= r[1] <= 4)
            if not ok:
                raise ValueError(f"responses must be (separator, 2..4), got {r!r}")
        if self.repeat is not None and not (isinstance(self.repeat, str) and self.repeat.strip()
                                            and self.repeat == self.repeat.strip()
                                            and len(self.repeat) <= MAX_REPEAT):
            raise ValueError("repeat must be a non-empty, stripped request text")

    @property
    def empty(self) -> bool:
        return not self.constraints

    @property
    def constraints(self) -> tuple[str, ...]:
        """One description per atomic constraint, e.g. ``("sentences: exactly 3",
        "keyword: 'ocean'")``; it starts with the name used in :func:`verify` messages."""
        out = [f"{u}: {getattr(self, u)}" for u in COUNT_UNITS if getattr(self, u) is not None]
        out += [f"keyword: {k!r}" for k in self.keywords]
        out += [f"forbidden: {k!r}" for k in self.forbidden]
        for name in ("json_output", "code_block", "end_question", "no_commas"):
            if getattr(self, name):
                out.append(name)
        if self.case:
            out.append(f"case: {self.case}")
        if self.list_style:
            out.append(f"list_style: {self.list_style}")
        out += [f"frequency: {w!r} {iv}" for w, iv in self.frequencies]
        out += [f"letter: {w!r} {iv}" for w, iv in self.letters]
        out += [f"{name}: {getattr(self, name)}" for name in FEATURE_COUNTS if getattr(self, name) is not None]
        if self.sections:
            out.append(f"sections: {self.sections[0]!r} {self.sections[1]}")
        if self.first_word:
            out.append(f"first_word: paragraph {self.first_word[0]} {self.first_word[1]!r}")
        for name in ("divider", "title", "quoted"):
            if getattr(self, name):
                out.append(name)
        if self.postscript:
            out.append(f"postscript: {self.postscript!r}")
        if self.starts_with:
            out.append("starts_with: " + " | ".join(repr(o) for o in self.starts_with))
        if self.end_phrase:
            out.append(f"end_phrase: {self.end_phrase!r}")
        if self.one_of:
            out.append("one_of: " + " | ".join(repr(o) for o in self.one_of))
        if self.responses:
            out.append(f"responses: {self.responses[1]} separated by {self.responses[0]!r}")
        if self.repeat:
            out.append("repeat")
        return tuple(out)

    def targets(self) -> tuple[int, int, int]:
        """Countdown targets ``(sentences, items, words)``: the upper bound, -1 for none."""
        return tuple(-1 if getattr(self, u) is None or getattr(self, u).hi is None
                     else getattr(self, u).hi for u in TARGET_UNITS)  # type: ignore[return-value]

    def token_local(self) -> "TokenLocal":
        """What a decoder can enforce token by token and what needs the complete text
        (module docstring, "Token-local enforcement")."""
        return token_local(self)


EMPTY = ConstraintSpec()


def _merge_pairs(x: tuple, y: tuple) -> tuple | None:
    out: dict[str, tuple[str, Interval]] = {}
    for w, iv in x + y:
        key = w.lower()
        if key in out:
            both = out[key][1].intersect(iv)
            if both is None:
                return None
            out[key] = (out[key][0], both)
        else:
            out[key] = (w, iv)
    return tuple(out.values())


def _same_or_none(x, y):
    """``(value, ok)``: the one that is set, or ``x`` when both are equal."""
    if x is None or x == () or x is False:
        return y, True
    if y is None or y == () or y is False:
        return x, True
    return x, x == y


def merge(a: ConstraintSpec, b: ConstraintSpec) -> ConstraintSpec | None:
    """Both specs at once (counts intersected, keyword lists concatenated without duplicates,
    start / answer options intersected), or ``None`` when they conflict."""
    values: dict = {}
    for unit in COUNT_UNITS + FEATURE_COUNTS:
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
    for name in ("frequencies", "letters"):
        values[name] = _merge_pairs(getattr(a, name), getattr(b, name))
        if values[name] is None:
            return None
    if a.sections and b.sections:
        if a.sections[0].lower() != b.sections[0].lower():
            return None
        both = a.sections[1].intersect(b.sections[1])
        if both is None:
            return None
        values["sections"] = (a.sections[0], both)
    else:
        values["sections"] = a.sections or b.sections
    for name, key in (("starts_with", lambda o: _ws(o).casefold()), ("one_of", option_key)):
        x, y = getattr(a, name), getattr(b, name)
        if x and y:
            ys = {key(o) for o in y}
            values[name] = tuple(o for o in x if key(o) in ys)
            if not values[name]:
                return None
        else:
            values[name] = x or y
    for name in ("case", "list_style", "first_word", "postscript", "end_phrase", "responses", "repeat"):
        values[name], ok = _same_or_none(getattr(a, name), getattr(b, name))
        if not ok:
            return None
    for name in FLAG_FIELDS:
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
    if spec.list_style:
        n_style = m["bullet_items" if spec.list_style == "bullet" else "numbered_items"]
        if m["items"] == 0 or n_style != m["items"]:
            out.append(f"list_style: {n_style} of {m['items']} list items are {spec.list_style}")
    for w, iv in spec.frequencies:
        n = count_keyword(text, w, whole=True)
        if not iv.contains(n):
            out.append(f"frequency: {w!r} occurs {n} times, want {iv}")
    for ch, iv in spec.letters:
        n = letter_count(text, ch)
        if not iv.contains(n):
            out.append(f"letter: {ch!r} occurs {n} times, want {iv}")
    for name in FEATURE_COUNTS:
        iv = getattr(spec, name)
        if iv is not None and not iv.contains(m[name]):
            out.append(f"{name}: {m[name]}, want {iv}")
    if spec.sections:
        n = count_sections(text, spec.sections[0])
        if not spec.sections[1].contains(n):
            out.append(f"sections: {n} lines start with {spec.sections[0]!r} N, want {spec.sections[1]}")
    if spec.first_word:
        k, word = spec.first_word
        firsts = m["paragraph_first_words"]
        got = firsts[k - 1] if k <= len(firsts) else None
        if got != word:
            out.append(f"first_word: paragraph {k} starts with {got!r}, want {word!r}")
    if spec.divider and m["dividers"] == 0 and m["blank_paragraphs"] > 1:
        out.append("divider: paragraphs are not separated by ***")
    if spec.title and not m["has_title"]:
        out.append("title: no <<title>>")
    if spec.postscript and not has_postscript(text, spec.postscript):
        out.append(f"postscript: no line starts with {spec.postscript!r}")
    if spec.quoted and not m["quoted"]:
        out.append("quoted: not wrapped in double quotes")
    if spec.starts_with and not any(starts_with_phrase(text, o) for o in spec.starts_with):
        out.append("starts_with: starts with none of " + ", ".join(repr(o) for o in spec.starts_with))
    if spec.end_phrase and not ends_with_phrase(text, spec.end_phrase):
        out.append(f"end_phrase: does not end with {spec.end_phrase!r}")
    if spec.one_of and not matches_option(text, spec.one_of):
        out.append("one_of: not one of " + ", ".join(repr(o) for o in spec.one_of))
    if spec.responses:
        sep, n = spec.responses
        parts, clean = split_responses(text, sep)
        if not clean or len(parts) != n or len(set(parts)) != len(parts):
            out.append(f"responses: {len(parts)} parts separated by {sep!r}, want {n} different ones")
    if spec.repeat and not repeats_request(text, spec.repeat):
        out.append("repeat: does not start with the request")
    return out


# ================================================================ token-local enforcement
@dataclass(frozen=True)
class TokenLocal:
    """The part of a spec a decoder can enforce token by token (module docstring).

    - ``counts``: ``(counter, lo, hi)`` with ``counter`` in ``"words", "sentences", "items"``,
      the raw counters of ``bes.exact.registers.Incremental`` (``raw()[counter]``; per-token
      increments ``table.inc[:, s, INC_WORDS | INC_SENT | INC_ITEMS]``). A token whose
      increment takes the counter above ``hi`` can never be part of a valid response; ``END``
      is valid only when ``lo <= count <= hi`` after its flush (``END`` adds one sentence when
      ``table.components[s, 5] == 1``).
    - ``lines``: ``(lo, hi)`` on non-blank lines, a byte-level count with one bit of state
      (:func:`nonblank_line_step`; not the register's newline counter).
    - ``banned_bytes``: bytes no token of the response may contain (``b","`` for no commas).
    - ``case``: ``"lower"`` bans tokens with an ASCII capital, ``"upper"`` tokens with an ASCII
      lowercase letter; non-ASCII letters and "at least one cased letter" are checked on the
      complete text.
    - ``forbidden``: lowercase UTF-8 forms of the forbidden words; an occurrence is the byte
      string, ASCII-case-insensitive, with no ASCII letter or digit (or byte >= 0x80) right
      before or after it (:func:`has_forbidden_bytes`; exact for ASCII neighbours).
    - ``letters``: ``(letter bytes, lo, hi)``: every byte of the letter in either case counts.
    - ``complete``: descriptions (as in ``ConstraintSpec.constraints``) of the constraints that
      can only be verified on the complete text (``verify``).
    """

    counts: tuple[tuple[str, int, int | None], ...] = ()
    lines: tuple[int, int | None] | None = None
    banned_bytes: bytes = b""
    case: str | None = None
    forbidden: tuple[bytes, ...] = ()
    letters: tuple[tuple[bytes, int, int | None], ...] = ()
    complete: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.counts or self.lines or self.banned_bytes or self.case or self.forbidden
                    or self.letters)


#: Constraint names (``ConstraintSpec.constraints`` prefixes) that are token-local.
TOKEN_LOCAL_NAMES = frozenset(("words", "sentences", "items", "lines", "no_commas", "case",
                               "forbidden", "letter"))


def token_local(spec: ConstraintSpec) -> TokenLocal:
    """The token-local part of ``spec`` and the descriptions of the rest (:class:`TokenLocal`)."""
    counts = tuple((u, getattr(spec, u).lo, getattr(spec, u).hi)
                   for u in ("words", "sentences", "items") if getattr(spec, u) is not None)
    lines = None if spec.lines is None else (spec.lines.lo, spec.lines.hi)
    letters = tuple((bytes([ord(ch), ord(ch.upper())]), iv.lo, iv.hi) for ch, iv in spec.letters)
    complete = tuple(c for c in spec.constraints if c.split(":")[0] not in TOKEN_LOCAL_NAMES)
    return TokenLocal(counts=counts, lines=lines, banned_bytes=b"," if spec.no_commas else b"",
                      case=spec.case, forbidden=tuple(k.lower().encode("utf-8") for k in spec.forbidden),
                      letters=letters, complete=complete)


_BLANK_BYTES = frozenset(b" \t\r\x0b\x0c")


def nonblank_line_step(data: bytes, open_line: bool) -> tuple[int, bool]:
    """Consume ``data``: ``(lines that became non-blank, open_line after)``. ``open_line``
    says whether the current line was already counted; a line counts at its first byte that
    is neither ``\\n`` nor ASCII whitespace, and ``\\n`` starts a new line. Summed over a
    response (from ``open_line=False``) this is ``measure(text)["lines"]``."""
    n = 0
    for b in data:
        if b == 0x0A:
            open_line = False
        elif not open_line and b not in _BLANK_BYTES:
            n += 1
            open_line = True
    return n, open_line


def _word_byte(b: int) -> bool:
    return b >= 0x80 or 0x30 <= b <= 0x39 or 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A


def has_forbidden_bytes(data: bytes, words: Sequence[bytes], final: bool = True) -> bool:
    """A complete occurrence of a forbidden word (``TokenLocal.forbidden``) in ``data``: the
    bytes, ASCII-case-insensitive, not preceded and not followed by a word byte (ASCII letter
    or digit, or any byte >= 0x80). An occurrence that ends ``data`` counts only when
    ``final`` (the next byte could still extend it)."""
    low = data.lower()
    for w in words:
        if not w:
            continue
        start = low.find(w)
        while start >= 0:
            end = start + len(w)
            before_ok = start == 0 or not _word_byte(data[start - 1])
            after_ok = (end == len(data) and final) or (end < len(data) and not _word_byte(data[end]))
            if before_ok and after_ok:
                return True
            start = low.find(w, start + 1)
    return False


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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’])\s+(?!(?:or|and|nor)\b)|\s*\n\s*|;\s*|"
                           r"(?<!phrase:)(?<!words:)(?<!sentence:)(?<!greeting:)(?<=:)\s+(?=[A-Z0-9])")


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
        r"grammatical|proper|actual|english|whole|own|easy|quick|basic|practical|concrete|"
        r"actionable|logical)")
_FILL = rf"(?:{_ADJ} ){{0,2}}?"
_UNIT_ALT = (r"(?:bullet[- ]?points?|bullets?|bulleted (?:items?|points?|lines?)|"
             r"numbered (?:items?|points?|lines?)|list items?|items?|words?|sentences?|"
             r"paragraphs?|steps|lines?(?! of (?:code|python|javascript|java|sql|html|css|output)\b))"
             r"(?![\w-])")
_UNIT_SING = (r"(?:bullet[- ]?point|bullet|(?:list|numbered|bulleted) (?:item|point)|item|word|"
              r"sentence|paragraph|line(?! of (?:code|python|javascript|java|sql|html|css|output)\b))")
#: A bare word count up to this is exact ("in three words", "a single word"); larger bare
#: word counts are budgets ("in 50 words" -> at most 50).
BARE_EXACT_WORDS = 5


def _unit(name: str = "unit") -> str:
    return f"(?P<{name}>{_UNIT_ALT})"


def unit_of(noun: str) -> str:
    """Count unit of a unit noun (``"bullet points"`` -> ``"items"``, ``"steps"`` -> ``"items"``)."""
    n = noun.lower()
    if n.startswith(("bullet", "numbered", "list", "item", "step")):
        return "items"
    for unit in ("words", "sentences", "paragraphs", "lines"):
        if n.startswith(unit[:-1]):
            return unit
    raise ValueError(noun)


def style_of(phrase: str) -> str | None:
    """List style named in an item phrase: ``"bullet"`` (bullet points, bulleted items),
    ``"numbered"`` (numbered items or steps) or ``None`` (list items, items, steps)."""
    p = phrase.lower()
    if "bullet" in p:
        return None if "numbered" in p else "bullet"
    return "numbered" if "numbered" in p else None


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
                           r"or thereabouts|-?ish|hopefully|if (?:necessary|needed|possible|you can|applicable)|"
                           r"(?:or|and) (?:so on|similar))\b")

_LEAD = (r"(?:(?:[-*•]|[0-9]+[.)]) )?(?:(?:please|kindly|also|and|then|now|so|just|finally|"
         r"additionally|lastly|first|firstly|second|secondly|next|moreover|furthermore|plus|ok|"
         r"okay|hey|hi|hello|again|instead|rather|simply|briefly|quickly|in addition|while|"
         r"explicitly|clearly|always|carefully|however|but|well|actually|yet|still|though|honestly|"
         r"anyway|besides),? )*")
_ASK = (r"(?:(?:(?:can|could|would|will) you (?:please )?(?:kindly )?|i (?:want|need|would like|'d like|"
        r"d like|wish|would prefer|prefer) (?:you to )?|i'm looking for |im looking for |"
        r"make sure (?:to |that |you )?|making sure (?:to |that )?|be sure to |try to |remember to |"
        r"feel free to |you (?:should|must|need to|have to|will|should not|shouldn't|must not|mustn't|"
        r"cannot|can't|can not|may not|are not allowed to) |"
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
                  r"making sure|make sure|be sure|aim|stick|say|repeat|retell|restate|redo|recast|"
                  r"mark|label|number|pick|choose|select|separate|conclude|close|wrap up)")
# A predicate of a requirement list ("- contains at least 5 sentences", "and contain exactly 4
# bullet points"): the request's subject was given before.
_PREDICATE_START = re.compile(rf"^{_LEAD}(?:contains?|includes?|has|have|uses?|is|are|be|ends?|"
                              r"starts?|begins?|mentions?|features?|consists? of|should|must|"
                              r"needs to|totals?|spans?)\b")
_REQUEST_START = re.compile(rf"^{_LEAD}{_ASK}{_REQUEST_VERBS}\b")
_MODAL = re.compile(r"\b(?:(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))should|must|shall|needs to|(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))need to|has to|"
                    r"(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))have to|"
                    r"ought to|is to|are to|"
                    r"is required|are required|required to|is expected to|are expected to)\b")
_I_WANT = re.compile(rf"^{_LEAD}(?:i (?:want|need|would like|'d like|d like|would prefer|prefer|"
                     r"don't want|do not want|dont want)|i'm looking for|im looking for)\b")
# A modal near the start of the clause ("your response should ...", "there must be ...").
_MODAL_EARLY = re.compile(rf"^{_LEAD}(?:\S+ ){{0,7}}?(?:(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))should|must|shall|needs to|"
                          r"(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))need to|has to|"
                          r"(?:(?<=letter i )|(?<!\bi )(?<!\bwe ))have to|"
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
    r"^(?:and |but |so |also |or )?(?:(?:how|what|why|where|which|who|whose) (?:do|does|did|is|are|was|were|has|have|had)\b|"
    rf"how (?:can|could|would|should|might|may|will) {_QSUBJ}\b|"
    r"how (?:can|could|would|should|might|may|will|does|do|did) \w+ing\b|"
    rf"(?:is|are|was|were|does|do|did|has|have|can|could|would|will|should|might) {_QSUBJ}\b|"
    r"what if\b|what about\b|how about\b|if\b|when\b|whenever\b|unless\b|in case\b|once\b|"
    r"whether\b|because\b|since\b|as long as\b|for example\b|for instance\b|e\.g\.|"
    r"do you (?:have|know|think|mean|see|agree|remember|understand)\b|have you\b)")
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


_WITHOUT_START = re.compile(r"^(?:without|avoiding|never|not) (?:using|use|mentioning|saying|including)\b")


def _strict_somewhere(sentence: str) -> bool:
    """The sentence or one of its clauses is a strict request ("without using the word "x",
    describe ...")."""
    if _is_strict_request(sentence):
        return True
    return any(_is_strict_request(sentence[a:b].strip()) for a, b in _clauses(sentence))


_OWN_TASK = re.compile(rf"^{_LEAD}i (?:(?:need|have|am trying|plan|am going|am planning|was asked|am supposed|"
                       r"was told|am required) to\b(?! (?:get|have|receive|see|read)\b)|have (?!to\b)(?:a|an|the|"
                       r"this|these|two|three|four|five|some|[0-9]+)\b)")


def _is_statement(clause: str) -> bool:
    if _NOT_A_REQUEST.match(clause) or _OWN_TASK.match(clause):
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
                         r"(?: (?:please|only|total|each|in total|is enough|will do))?[.!]?$")
_P_NO_ITEMS = re.compile(r"\b(?:do not|don't|dont|never|avoid|without|no)(?: (?:use|using|include|"
                         r"including|add|adding|any|the|a|bulleted|numbered|markdown|format(?:ting)?"
                         r"|with))* (?:bullet[- ]?points?|bullets|bulleted lists?|numbered lists?|"
                         r"lists?|list items|list formatting)\b"
                         r"(?= ?(?:$|[.,;:!?)]|or\b|and\b|in\b|please\b|at all\b|anywhere\b))")

_LIST_VERBS = (r"(?:list|name|give(?: me| us)?|provide(?: me with| us with| me| us)?|suggest|"
               r"recommend|share|offer|enumerate|mention|outline|brainstorm|come up with|"
               r"tell me|write down|write out|jot down|think of)")
_P_LIST_N = re.compile(rf"^{_LEAD}{_ASK}{_LIST_VERBS}(?: the)?(?: top)?(?: exactly)? {_num('num')} "
                       r"(?:(?:different|distinct|separate|alternative|unique|other|possible|new) )?"
                       r"(?P<noun>[a-z][a-z'-]*)")
# Measures, not listed things ("give me 5 minutes"), and words that cannot head a list noun.
_LIST_BLOCK_NOUNS = frozenset(
    "minute minutes hour hours second seconds day days week weeks month months year years time "
    "times percent dollars dollar cents euros pounds kg km miles meters metres feet inches "
    "degrees or to and responses response answers versions replies drafts variations variants "
    "alternatives as in of for at by on".split())
#: "list N <things>" is read for 2 <= N <= this ("think of 2015 as ..." is not a list).
MAX_LIST_N = 50
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
    style: str | None = None      # list style named by the phrase ("3 bullet points")


def _bare_interval(unit: str, n: int) -> Interval:
    return Interval(0, n) if unit == "words" and n > BARE_EXACT_WORDS else Interval(n, n)


# Steps count as list items after "in / into / using / with" or a relation ("break it into 4 steps").
_STEPS_OK = re.compile(rf"\b(?:in|into|using|with|{_REL_ANY}) (?:only |just )?{_NUM_BODY} {_FILL}steps\b")
# "answer with one word", "reply with just two words": a small bare word count after "with".
_ANSWER_WITH = re.compile(r"\b(?:answer|respond|reply|answering|responding|replying)"
                          r"(?: (?:only|just|simply|me|us))? \Z")
# "pick three of them and present them as a numbered list": a count of the listed things.
_P_PICK_N = re.compile(rf"^{_LEAD}{_ASK}(?:pick|choose|select|take)(?: out)?(?: the)?(?: top| best)? "
                       rf"{_num('num')}(?: (?:of (?:them|these|those|the \w+|your \w+|my \w+)|"
                       r"(?P<noun>[a-z][a-z'-]*s)))?(?![\w-])")
# "give me a list of 5 tips", "write a numbered list of ten ideas".
_P_LIST_OF = re.compile(rf"\b(?:a|an) (?:(?:short|brief|simple|quick|numbered|bulleted|bullet|markdown) )*"
                        rf"list of (?:exactly |only |just )?{_num('num')} (?P<noun>[a-z][a-z'-]*)")


_STRONG_PREPS = frozenset(("in", "into", "using", "within", "as"))
# A determiner before the number ("in this one sentence", "the two paragraphs below").
_DET_BEFORE = re.compile(r"\b(?:this|that|these|those|the|my|your|his|her|our|their|its|each|every) \Z")
# The count is about an edit, not about the response ("by removing at most three words").
_EDIT_BEFORE = re.compile(r"\b(?:remove|removes|removing|removed|delete|deleting|deleted|cut|cutting|"
                          r"add|adding|added|change|changing|replace|replacing|drop|dropping|omit|"
                          r"omitting|trim|trimming|shorten (?:it |this )?by|reduce (?:it |this )?by)"
                          r"(?: [\w']+){0,2} \Z"
                          # what was done ("while you were editing this paragraph into 3 sentences")
                          r"|\b(?:edited|editing|condensed|shortened|rewrote|rewritten|rephrased|combined|"
                          r"merged|converted|reduced|trimmed|broke|broken)(?: [\w'-]+){0,5} \Z"
                          # a part of the reply ("begin your answer with a 1 sentence summary")
                          r"|\b(?:begin|begins|start|starts|open|opens|end|ends|finish|finishes|conclude|"
                          r"concludes|close|closes)(?: [\w']+){0,3} with (?:a |an |one )?(?:single )?\Z"
                          r"|\b(?:improve|edit|revise|refine|fix|check|review|proofread|polish|correct) "
                          r"(?:a |an |the |this |my |your )?\Z"
                          r"|\b(?:changing|combining|merging|turning|converting|condensing|shortening|"
                          r"splitting)(?: [\w'\"\x01-]+){0,8} \Z"
                          # a text being worked on ("help on improving a single sentence")
                          r"|\b(?:improving|editing|revising|rewriting|refining|fixing|checking|reviewing|"
                          r"proofreading|polishing|correcting)(?: [\w']+){0,2} \Z")
# "I want the revised passage to be no longer than 3 sentences": a part of the reply.
_SCOPED_TEXT = re.compile(r"\b(?:the|your) (?:revised|edited|rewritten|new|improved|shortened|condensed|"
                          r"corrected|updated|polished) (?:passage|paragraph|text|version|sentence|essay|draft|"
                          r"story|section|letter|email|abstract)\b|\b(?:ending|opening|closing|concluding|"
                          r"introductory) (?:sentence|paragraph|line)\b|\b(?:first|second|third|fourth|fifth|last|"
                          r"final|next|previous) (?:and \w+ )?(?:sentence|paragraph|line)s?\b")
# "a brief 5-7 word title": the length of that element.
_PART_NOUN_AFTER = re.compile(r" (?:title|headline|name|slogan|tagline|caption|subject line|hashtag|motto|"
                              r"phrase|label)s?\b")
# "or even a single sentence": one of several alternatives.
_OR_BEFORE = re.compile(r"\bor (?:even |just |perhaps |maybe |possibly )?(?:a |an |in |as |into )?\Z")
# "decide if the characters in one paragraph are continued": a count inside an embedded question.
_EMBEDDED_IF = re.compile(r"\b(?:if|whether)(?: [\w'-]+)+ \Z|\b(?:decide|decides|deciding|determine|determining|"
                          r"check|checking|see|know|tell|ask|asking|wonder|sure|unsure)\s+(?:if|whether)\b"
                          r"[^.;:?!]* \Z|\bwhether\b[^.;:?!]* \Z")
# "generated sentences must have fewer than 10 words": a per-part count.
_PLURAL_SUBJECT = re.compile(r"\b(?:sentences|lines|paragraphs|items|bullets|bullet points|points|steps|tips|"
                             r"answers|titles|questions|examples) (?:must|should|need to|have to|are to|will|can)\b")
_EDIT_AFTER = re.compile(r" (?:added|removed|deleted|changed|replaced|inserted|altered|modified|cut)\b")
_REL_TAIL = re.compile(rf"(?:^| )({_REL_ANY}) \Z")
# "replace the phrasal verb with a single word": the count is of the replacement.
_REPLACE = re.compile(r"\b(?:replace|replacing|substitute|substituting|swap|swapping|exchange)\b")
# "explain each concept in less than 50 words", "3 sentences per paragraph": per-part counts.
_EACH = re.compile(r"\b(?:each|per|every)\b")
# "one item shorter than the input": a comparison, not a count.
_COMPARATIVE_AFTER = re.compile(r" (?:shorter|longer|more|less|fewer|bigger|smaller|larger|greater|than)\b")


def _count_candidates(c: str, first_clause: bool = True, in_request: bool = False,
                      list_format: bool = False) -> list[_Cand]:
    """All count requests found in one normalised clause ``c`` (overlaps unresolved). A
    candidate with ``interval = None`` is a recognised but ambiguous phrase: it blocks shorter
    readings of the same words and yields no constraint. ``first_clause``: ``c`` opens its
    sentence (the "list N <things>" form is read only there); ``in_request``: the sentence is a
    request ("your response should contain ..., and contain exactly 4 bullet points");
    ``list_format``: the sentence asks for a list format ("as a numbered list"), which makes
    "pick N of them" a count of list items."""
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
        if unit_text == "steps" and not (list_context or (_REQUEST_START.match(c) and _STEPS_OK.search(c, 0, m.end()))):
            return                       # "solve it in 3 steps", "all I have to do is take five steps"
        if (per_part or _COMPARATIVE_AFTER.match(c, m.end()) or _EDIT_AFTER.match(c, m.end())
                or _SCOPED_TEXT.search(c, 0, m.start()) or _OR_BEFORE.search(c, 0, m.start())
                or _EMBEDDED_IF.search(c, 0, m.start()) or _PLURAL_SUBJECT.search(c, 0, m.start())
                or (unit_text and unit_text.startswith("word") and _PART_NOUN_AFTER.match(c, m.end()))):
            iv = None
        unit = unit_of(unit_text) if unit_text else None
        style = style_of(m.group(0)) if unit == "items" else None
        cands.append(_Cand(m.start(), m.end(), unit, iv, style=style))

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
        if strong and re.match(r"\S+ (?:this|that) ", m.group(0)):
            continue                     # "refining the clarity in this one sentence"
        answer_with = (unit == "words" and intro == "with" and n is not None
                       and n <= BARE_EXACT_WORDS and _ANSWER_WITH.search(c, 0, m.start()) is not None)
        if answer_with:
            if not plain(m):
                add(m, m.group("unit"), None)
                continue
            add(m, m.group("unit"), Interval(n, n))   # "answer with one word"
            continue
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
        if _EDIT_BEFORE.search(c, 0, m.start()):
            continue                     # "begin your answer with a 1 sentence summary"
        raw = m.group("num")
        n = 1 if raw == "single" else parse_count(raw)
        nouns = {m.group("noun"), m.group("noun2")}
        produced = bool(_PRODUCE.search(c, 0, m.start()) or _I_WANT.match(c))
        if not (produced and nouns & _ADJ_NOUNS):
            continue                     # "describe a 5-paragraph essay structure"
        ok = n is not None and request and plain(m)
        add(m, m.group("unit"), _bare_interval(unit_of(m.group("unit")), n) if ok else None)
    for m in _P_SINGLE.finditer(c) if not _REPLACE.search(c) and not re.search(r"\bfor (?:a |one )?single\b", c) else ():
        ok = ((m.group("prep") is not None or request) and negation(m.start()) is None
              and not _EDIT_BEFORE.search(c, 0, m.start()) and not _EDIT_BEFORE.search(c, 0, m.start("unit")))
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
        noun_is_unit = m.group("noun") != "steps" and re.fullmatch(_UNIT_ALT, m.group("noun")) is not None
        another = _NUMBER_ANYWHERE.search(c, m.end())    # "three pros and three cons"
        if (n is not None and 2 <= n <= MAX_LIST_N and not noun_is_unit and not approx(m, "num")
                and not another):
            cands.append(_Cand(m.start("num"), m.end(), "items", Interval(n, n), list_n=True))
    m = _P_PICK_N.match(c) if first_clause and list_format and not _EACH.search(c) else None
    if m and (m.group("noun") or "") not in _LIST_BLOCK_NOUNS:
        n = parse_count(m.group("num"))
        if n is not None and 2 <= n <= MAX_LIST_N and not approx(m, "num") and not _NUMBER_ANYWHERE.search(c, m.end()):
            cands.append(_Cand(m.start("num"), m.end(), "items", Interval(n, n), list_n=True))
    for m in _P_LIST_OF.finditer(c) if request and not _EACH.search(c) else ():
        n, noun = parse_count(m.group("num")), m.group("noun")
        if (n is not None and 2 <= n <= MAX_LIST_N and noun not in _LIST_BLOCK_NOUNS and noun.endswith("s")
                and re.fullmatch(_UNIT_ALT, noun) is None and plain(m)
                and not _NUMBER_ANYWHERE.search(c, m.end())):
            cands.append(_Cand(m.start(), m.end(), "items", Interval(n, n), list_n=True,
                               style=style_of(m.group(0))))
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
_MORE_UNITS = re.compile(rf"{_NUM_BODY} (?:more|additional|extra|further) (?P<u>{_UNIT_ALT})")
_COUNT_UNIT_AT = re.compile(rf"{_NUM_BODY}\+?[- ]{_FILL}(?P<u>{_UNIT_ALT})")


def _another_quantity(sentence: str, pos: int) -> bool:
    """A number after ``pos`` that is not itself the count of a unit ("at most 6 words")."""
    for m in _NUMBER_ANYWHERE.finditer(sentence, pos):
        if not _COUNT_PHRASE_AT.match(sentence, m.start()):
            return True
    return False


_OR_WORD = re.compile(r"\bor\b")
_OR_ALTERNATIVE = re.compile(rf" or (?:[\w'-]+ ){{0,4}}?{_UNIT_ALT}")
_OR_LIST = re.compile(r"(?:, (?!and\b)[^,.;:]{1,30}){1,3}, or\b")   # "a sentence, a paragraph, or ..."


# A request for code: the things listed or counted in it are code.
_CODE_REQUEST = re.compile(r"\b(?:write|writes|writing|implement|implementing|create|build|develop|generate)\b"
                           r"(?: [\w+#'-]+){0,6} (?:function|functions|program|script|code(?! ?blocks?\b| fences?\b)|"
                           r"class|method|query|regex|algorithm|unit tests?)\b|\bunit tests?\b|\bimplement\b")


def _sentence_counts(sentence: str, inline_form: bool, list_format: bool = False,
                     code_message: bool = False) -> list[tuple[str, Interval, str | None]]:
    """Count requests of one normalised sentence, as ``(unit, interval, list style)`` (none in
    a multi-step sentence such as "write a single paragraph, then improve it", nor when a
    conditional clause carries its own count: "use one paragraph, but when providing the edited
    text use 2 paragraphs")."""
    if _SEQUENCE.search(sentence):
        return []
    for a, b in _clauses(sentence):
        clause = sentence[a:b].strip()
        if _CONDITIONAL.match(clause) and a > 0 and _COUNT_PHRASE_AT.search(clause):
            return []
    statement = _is_statement(sentence)
    in_request = not statement and _is_strict_request(sentence)
    # "a function that takes a single word": counts of code, not of the response (except
    # "list N <things>", which is about the response: "give me three reasons to learn python").
    code = bool(_CODE_TOPIC.search(sentence)) and not _RESPONSE_SUBJ.search(sentence)
    code_request = code and bool(_CODE_REQUEST.search(sentence))
    per_sentence = bool(re.search(r"\bfor each\b|\beach (?:one|item|point|of them)\b", sentence))
    found: list[_Cand] = []
    for a, b in _clauses(sentence):
        clause = sentence[a:b].strip()
        offset = a + (len(sentence[a:b]) - len(sentence[a:b].lstrip()))
        if statement and not _is_strict_request(clause):
            continue                     # a clause of a statement: "..., divided into 10 lines"
        for cand in _resolve(_count_candidates(clause, first_clause=a == 0, in_request=in_request,
                                               list_format=list_format)):
            if code and (not cand.list_n or code_request) or (code_message and not _RESPONSE_SUBJ.search(sentence)):
                continue                 # counts of code; "write a function ... provide 2 solutions"
            if cand.list_n and _another_quantity(sentence, offset + cand.end):
                continue                 # "list 9 discoveries, and also 4 breakthroughs"
            before = sentence[: offset + cand.start]
            if cand.interval is not None and (_EMBEDDED_IF.search(before) or (
                    _PLURAL_SUBJECT.search(before) and cand.unit in ("words", "sentences", "lines"))):
                continue                 # "decide if ..., settings in one paragraph", "sentences must have 10 words"
            found.append(_Cand(cand.start + offset, cand.end + offset, cand.unit, cand.interval,
                               cand.list_n, cand.style))
    found.sort(key=lambda x: x.start)
    drop = set()
    for i in range(len(found) - 1):          # "3 sentences or 2 paragraphs": alternatives
        if _OR_WORD.search(sentence[found[i].end:found[i + 1].start]):
            drop.update((i, i + 1))
    for i, cand in enumerate(found):
        if _OR_ALTERNATIVE.match(sentence, cand.end) or _OR_LIST.match(sentence, cand.end):
            drop.add(i)
    bare = [unit_of(m.group("u")) for m in _COUNT_UNIT_AT.finditer(sentence)
            if not _REL_TAIL.search(sentence, 0, m.start()) and not _P_LIST_N.match(sentence)]
    added = {u for u in bare if bare.count(u) > 1}   # "4 clear sentences and four unclear sentences"
    out = []
    for i, cand in enumerate(found):
        if i in drop or cand.interval is None or cand.unit is None or cand.unit in added:
            continue
        if inline_form and cand.list_n:
            continue
        if per_sentence and not cand.list_n:
            continue                     # "list 10 plots with a description (1 sentence) for each"
        out.append((cand.unit, cand.interval, cand.style))
    return out


# ======================================================================== keyword rules
_KW_NOUN = r"(?:key[ -]?words?|key[ -]?phrases?|words?|phrases?|terms?|expressions?)"
_KW_VERB = (r"(?:include|includes|including|included|use|uses|using|used|mention|mentions|"
            r"mentioning|mentioned|contain|contains|containing|incorporate|incorporates|"
            r"incorporating|incorporated|feature|features|featuring|insert|add|adding|have|"
            r"has|having|put)")
_KW_STRONG = r"(?:include|includes|mention|mentions|incorporate|incorporates)"
_FORBID_VERB = (r"(?:avoid|avoids|avoiding|exclude|excludes|excluding|omit|omits|omitting|"
                r"refrain from(?: using)?|steer clear of|(?:never|don't|dont|do not|you cannot|you can't|"
                r"you may not|you must not) (?:say|use|write|type|mention|include))")
_DET = r"(?:(?:the|these|those|following|all|both|each|every|of|exact|specific|any|using|use) )*"
_ATOM_START = "(?=[\"“‘'\\[*])"
_K_LIST = re.compile(rf"\b{_KW_VERB} {_DET}{_KW_NOUN} ?:? ?{_ATOM_START}", re.I)
_K_STRONG = re.compile(rf"\b{_KW_STRONG} {_ATOM_START}", re.I)
_K_FORBID = re.compile(rf"\b{_FORBID_VERB} {_DET}(?:{_KW_NOUN} ?:? ?)?{_ATOM_START}", re.I)
_K_BARE = re.compile(rf"\b{_KW_VERB} (?:the )?(?:key[ -]?words?|keyword)(?:\*\*)?(?: below)? ?:? (?=[a-z])", re.I)
_ATOM = re.compile(
    r'"(?P<a>[^"\n]{1,60})"|“(?P<b>[^”\n]{1,60})”|‘(?P<c>[^’\n]{1,60})’'
    r"|(?<![\w])'(?P<d>[^'\n]{1,60})'(?![\w])|\[(?P<f>[^\[\]\n]{1,200})\]"
    r"|\*\*(?P<g>[^*\n]{1,60})\*\*|\*(?P<h>[^*\n]{1,60})\*")
_K_SUBJ = re.compile(
    rf"\b(?:the )?{_KW_NOUN} (?:(?P<atom>{_ATOM.pattern})|(?P<bare>[a-z][a-z'-]*)) "
    r"(?:should|must|needs to|has to|is to|shall|is required to)(?: also)? (?P<neg>not )?"
    r"(?:appear|be (?:included|used|mentioned|present)|occur|show up|feature)\b", re.I)
_K_SUBJ_LIST = re.compile(rf"\b(?:the )?(?:key[ -]?words|words|phrases|terms|expressions) (?={_ATOM_START})", re.I)
_K_SUBJ_TAIL = re.compile(r" (?:should|must|needs to|need to|has to|have to|are to|will|shall|are required to)"
                          r"(?: also)? (?P<neg>not )?(?:appear|be (?:included|used|mentioned|present)|occur|"
                          r"show up|feature)\b", re.I)
# "[keywords] are "a", "b"": a bracketed template variable defined elsewhere in the message.
_PLACEHOLDER_DEF = re.compile(r"\[([^\[\]\n]{1,40})\] ?(?:is|are|=|:|-|will be|should be|can be|include|"
                              r"includes) ?:? ?", re.I)
# Bracketed names that are unfilled template variables, never keywords.
_TEMPLATE_NAME = re.compile(r"(?:key ?words?|key ?phrases?|words?|phrases?|terms?|.*_.*|.*[0-9].*)", re.I)
_LIST_SEP = re.compile(r" ?(?:, ?(?:and |& )?|and |& |as well as |plus )", re.I)
_LIST_OR = re.compile(r" ?,? ?(?:or|nor) ", re.I)
_BARE_WORD = re.compile(r"[a-z][a-z'-]{0,30}", re.I)
# A plain keyword of one to three words ("health benefits", "anti-inflammatory") in a comma list.
_BARE_PHRASE = re.compile(r"[a-z][a-z'-]{0,30}(?: [a-z][a-z'-]{0,30}){0,2}(?=,| and | or |\.|$| in )", re.I)
_BARE_OR = re.compile(r" ?(?:, ?(?:or |nor )?|or |nor )", re.I)
_BARE_END = re.compile(r" ?(?:in (?:the|your) (?:response|answer|reply|text|essay|story|poem|"
                       r"output|email|letter|post|summary|description|message))? ?[.!?;:]* ?\Z", re.I)
_KW_STOP = frozenset(
    "that which who whom whose related relevant such like about from some any the a an to for of "
    "in on with and or as is are be your you it this these those following below above "
    "appropriate specific important several many few other more most naturally throughout into "
    "at by when where how what why if not no all each every their its our my his her them they "
    "we he she there here than then also only just very well".split())
_PAST_KW = re.compile(r"(?:used|included|mentioned|incorporated|added|inserted|featured|had)\b", re.I)
_PRONOUN_BEFORE = re.compile(r"\b(?:i|you|we|they|he|she|it|you've|i've|we've|they've)\s+(?:have\s+|had\s+|just\s+|already\s+)?\Z", re.I)
_NOUN_USE = re.compile(r"\b(?:the|its|their|this|that|his|her|my|your|our) use \Z", re.I)
_FORBID_START = re.compile(rf"\b{_FORBID_VERB}\b", re.I)
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
        if value is None and not words:
            value = _plain_definition(flat, m.end())     # "[forbidden_words] are: error, mistake."
        table.setdefault(m.group(1).strip().lower(), value)
    return table


def _parse_bare_list(s: str, pos: int, allow_or: bool = False) -> list[str] | None:
    """Plain keywords ``A, B and C`` ending the sentence (``in the response`` allowed): single
    words, or phrases of up to three words when the list has commas; ``allow_or``: ``or`` also
    separates (a negated list, "do not include A, B, or C"). No stop words."""
    for pattern in (_BARE_WORD, _BARE_PHRASE):
        words, end = [], pos
        m = pattern.match(s, end)
        while m and len(words) < MAX_KEYWORDS:
            words.append(m.group(0))
            end = m.end()
            sep = (_BARE_OR.match(s, end) if allow_or else None) or _LIST_SEP.match(s, end)
            if not sep:
                break
            m = pattern.match(s, sep.end())
            if m:
                end = sep.end()
            else:
                break
        if pattern is _BARE_PHRASE and ("," not in s[pos:end] or len(words) < 2):
            continue
        if not words or not _BARE_END.fullmatch(s, end):
            continue
        if any(w.lower() in _KW_STOP or w.lower().endswith("'") for p in words for w in p.split()):
            continue
        return words
    return None


def _plain_definition(s: str, pos: int) -> list[str] | None:
    """An unquoted definition list ``[name] are: a, b and c.`` (keywords of one to three words,
    no stop words) running to the end of the sentence."""
    rest = s[pos:]
    end = re.search(r"[.!?;](?:\s|$)|\n|$", rest)
    body = rest[: end.start()].strip()
    parts = [p.strip() for p in re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+", body) if p.strip()]
    if not parts or len(parts) > MAX_KEYWORDS:
        return None
    for p in parts:
        ws = p.split()
        if not 1 <= len(ws) <= 3 or not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", p) or any(w.lower() in _KW_STOP for w in ws):
            return None
    return parts


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
            if _NOUN_USE.search(s, 0, m.start() + 4) and s[m.start():m.start() + 4].lower() == "use ":
                continue                   # "explain the use of the word X"
            if _PAST_KW.match(s, m.start()) and _PRONOUN_BEFORE.search(s, 0, m.start()):
                continue                   # "why you used the word X"
            taken.append((m.start(), end))
            if negated(m.start()):
                forbidden += values(atoms)
            elif not has_or and not freq_kills(end):
                required += values(atoms)
    for m in _K_BARE.finditer(s) if triggers else ():
        if not free(m.start(), m.end()):
            continue
        neg = negated(m.start()) or bool(_FORBID_START.match(s, m.start()))
        words = _parse_bare_list(s, m.end(), allow_or=neg)
        if words:
            taken.append((m.start(), len(s)))
            (forbidden if neg else required).extend(words)
    for m in _K_SUBJ_LIST.finditer(s):                 # 'the words "a" and "b" should not appear'
        if not free(m.start(), m.end()):
            continue
        atoms, has_or, end = _parse_atom_list(s, m.end())
        tail = _K_SUBJ_TAIL.match(s, end)
        if len(atoms) < 2 or tail is None:
            continue
        taken.append((m.start(), tail.end()))
        if tail.group("neg") or negated(m.start()):
            forbidden += values(atoms)
        elif not has_or and not freq_kills(tail.end()):
            required += values(atoms)
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
               r"raw|strict) )?json(?: (?:format|object|output|response))?(?: only)?[.!]?$"
               r"|^(?:just|only) (?:the |a |an )?(?:(?:valid|pure|raw) )?json(?: (?:format|object|output|"
               r"response))?(?:,? please)?[.!]?$"),
    # "give me a json object that describes ...", "return a json array of three colours"
    re.compile(rf"(?:^|, |\band |\bthen |\bplease |\balso ){_LEAD}{_ASK}(?:give|provide|return|output|"
               r"generate|create|write|produce|send|make|build|prepare|compose|reply with|respond with|"
               r"answer with)(?: me| us)? (?:(?:a|an|one|the|single|valid|simple|small|short|sample|"
               r"complete|proper|well-formed|properly formatted|json-formatted|only|just) )*json "
               r"(?:object|array|list|document|dictionary|structure|record|response|output|string|"
               r"representation|blob)s?\b"),
    # "output only the json", "only return json", "nothing but json"
    re.compile(r"\b(?:output|return|give|provide|respond with|reply with|answer with|print|send|show|"
               r"write)(?: me| us)? (?:only|just|nothing but|solely|strictly|exclusively) (?:(?:the|a|an|"
               r"valid|raw|pure|plain|resulting|final) )*json\b"
               r"|\b(?:only|just) (?:output|return|give|provide|respond with|reply with|answer with|print|"
               r"send|show|write) (?:(?:the|a|an|valid|raw|pure|plain) )*json\b"
               r"|\bnothing (?:but|except|other than|besides) (?:(?:the|a|an|valid|raw) )*json\b"),
]
_F_JSON_NEW = frozenset((3, 4))
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
# Further "end with a question" forms, read only outside informational questions.
_F_QUESTION_NEW = [
    re.compile(r"\b(?:end|finish|conclude|close|wrap up)(?: (?:it|this|your|the|entire|whole|response|"
               r"answer|reply|message|text|output|email|letter|post|story|poem|note|paragraph))*(?: off)? "
               r"by asking (?:me |us |the reader |the user |them |him |her |the audience )?(?:a |an |one )?"
               r"(?:[\w-]+ )?question\b"),
    re.compile(r"\b(?:ending|finishing|concluding|closing) (?:it |this |(?:your|the) (?:response|answer|reply|"
               r"message|text|email|letter|post) )?(?:with|on) (?:a |an |one )?(?:[\w-]+ )?question\b"),
    re.compile(r"\b(?:the|your) (?:last|final|closing) (?:sentence|line)(?: of (?:your|the) (?:response|answer|"
               r"reply|message|text))? (?:should|must|has to|needs to|will|shall) be (?:a|an) (?:[\w-]+ )?question\b"),
    re.compile(r"\bmake sure (?:that )?(?:the|your) (?:last|final) (?:sentence|line) is (?:a|an) (?:[\w-]+ )?question\b"),
    re.compile(r"\b(?:your|the) (?:response|answer|reply|message|text|output)(?: [\w']+)? (?:should|must|needs to|"
               r"has to|will) (?:end|finish|conclude|close) (?:by asking|in) (?:a |an )?(?:[\w-]+ )?question\b"),
]
_STYLE_INTRO = (r"\b(?:as|in|into|using|use|with|in the form of|formatted as|"
                r"format(?: it| them| this| that| your answer| your response| the answer)? as|"
                r"present(?:ed)?(?: it| them| this| these| those)? as|organi[sz]ed?(?: it| them)? (?:as|into)|"
                r"write(?: it| them)? as|list(?: them| it)? (?:as|in)|give(?: me| us)?(?: it| them)? (?:as|in)|"
                r"provide(?: it| them)? (?:as|in)|answer with|respond with|reply with)")
_F_NUMBERED = re.compile(
    rf"{_STYLE_INTRO} (?:a |an |one )?(?:(?:single|short|simple|clear|brief|concise|markdown|vertical) )*"
    r"numbered (?:list|format|form|points|items|steps|lines|outline)\b"
    r"|\bnumber (?:each|every|the|your|all the) (?:items?|points?|steps?|entries|entry|lines?|ideas?|tips?)\b")
_STYLE_ALTERNATIVE = re.compile(r"\bbullet(?:ed|s)?(?: points?| list)? (?:or|and/or) (?:an? )?numbered\b"
                                r"|\bnumbered (?:list|points|items)? ?(?:or|and/or) (?:an? )?bullet")
_F_BULLETED = re.compile(
    rf"{_STYLE_INTRO} (?:a |an |the )?(?:(?:short|simple|clear|brief|concise|markdown) )*"
    r"(?:bullet(?:ed)?[- ]?(?:points?|list|format|form)|bullets)\b")
_F_NO_COMMAS = [
    re.compile(r"\b(?:do not|don't|dont|never|without|avoid|refrain from)(?: (?:using|use|the use of|"
               r"any|the|of))* commas?\b"),
    re.compile(r"\bno commas?\b(?! (?:after|before|between|around|inside|within|in the|in your|"
               r"in this|on)\b)"),
    re.compile(r"\bcommas? (?:are|is) (?:not allowed|forbidden|prohibited|not permitted)\b"),
]
# Further "no commas" forms, read only outside informational questions.
_F_NO_COMMAS_NEW = [
    re.compile(r"\b(?:do not|don't|dont|never|without|avoid|cannot|can't|must not|mustn't|should not|"
               r"shouldn't|may not)(?: (?:using|use|the use of|any|the|of|include|including|put|putting|add|"
               r"adding|write|writing|type|typing|insert|inserting|contain|containing|have|having|a|"
               r"a single|single|even one|one))* commas?\b(?! (?:after|before|between|around|inside|"
               r"within|on|at|in this)\b)(?! in (?:the|your) (?!(?:entire |whole )?(?:response|answer|reply|"
               r"output|text)\b))"),
    re.compile(r"\b(?:no|zero) commas?\b(?! (?:after|before|between|around|inside|within|on|at|in this)\b)"
               r"(?! in (?:the|your) (?!(?:entire |whole )?(?:response|answer|reply|output|text)\b))"),
    re.compile(r"\bcommas? (?:are|is) (?:banned|not to be used|not acceptable)\b|\bcomma[- ]free\b"),
]


_CONDITIONAL = re.compile(r"^(?:if|when|whenever|unless|in case|once|while)\b")
# "give an example of a sentence where it would be better to use no comma": about comma usage.
_COMMA_TOPIC = re.compile(r"\b(?:example|rule|rules|where it would|would be better|when to use|how to use)\b")
# "a json array / object / string" names the response format, not a programming topic.
_JSON_NOUNS = re.compile(r"\bjson (?:objects?|arrays?|lists?|strings?|files?|documents?|format)\b")
_CONVERSION = re.compile(r"\b(?:to|into|convert|converts|converting|lowercased|uppercased|named|labell?ed|"
                         r"called|spelled|spelt|titled|written as|capitali[sz]ed)(?: in)?(?: all)? \Z")
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
    code_topic = bool(_CODE_TOPIC.search(_JSON_NOUNS.sub("json", s))) and not _RESPONSE_SUBJ.search(s)
    out = _formats_of(s, sentence_request, code_topic, whole=True)
    for a, b in _clauses(s):
        clause = s[a:b].strip()
        if clause != s and not _CONDITIONAL.match(clause):
            part = _formats_of(clause, sentence_request or _is_request(clause), code_topic, whole=False)
            out["case"] = out.get("case", set()) | part.pop("case", set())
            out["list_style"] = out.get("list_style", set()) | part.pop("list_style", set())
            out.update(part)
    for key in ("case", "list_style"):
        if not out.get(key):
            out.pop(key, None)
    return out


def _formats_of(s: str, request: bool, code_topic: bool, whole: bool) -> dict:
    """Format requests of one normalised sentence or clause ``s`` (``code_topic``: the sentence
    is about code, so case words describe the code's output; ``whole``: ``s`` is a whole
    sentence, the only place where the fragment "JSON only." is read)."""
    out: dict = {}
    question = bool(_NOT_A_REQUEST.match(s))
    json_patterns = () if code_topic else [     # not a program's output
        p for i, p in enumerate(_F_JSON) if (whole or i != 2) and not (question and i in _F_JSON_NEW)]
    for pat in json_patterns:
        m = pat.search(s)
        if m and not _negated_before(s, m.start()):
            out["json_output"] = True
    if (request or _RESPONSE_SUBJ.search(s)) and not code_topic:
        styles = set()
        for style, pat in (("numbered", _F_NUMBERED), ("bullet", _F_BULLETED)):
            m = pat.search(s)
            if m and not _negated_before(s, m.start()) and not question:
                styles.add(style)
        if styles and _STYLE_ALTERNATIVE.search(s):
            styles = {"bullet", "numbered"}              # "bullet points or a numbered list"
        if styles:
            out["list_style"] = styles
    if request or _RESPONSE_SUBJ.search(s):
        for pat in _F_CODE:
            m = pat.search(s)
            if m and not _negated_before(s, m.start()):
                out["code_block"] = True
    cases = set()
    if not code_topic and not _CASE_FREQUENCY.search(s):
        if request or _RESPONSE_SUBJ.search(s) or (len(s.split()) <= 6 and not _is_statement(s)):
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
    for pat in _F_QUESTION + ([] if question else _F_QUESTION_NEW):
        m = pat.search(s)
        if m and not _negated_before(s, m.start()):
            out["end_question"] = True
    if (any(p.search(s) for p in _F_NO_COMMAS + ([] if question else _F_NO_COMMAS_NEW))
            and not _COMMA_TOPIC.search(s)):
        out["no_commas"] = True
    return out


# ================================================================ further instruction families
def _rx(pattern: str) -> re.Pattern:
    """Case-insensitive pattern in which every literal space matches any whitespace run."""
    return re.compile(pattern.replace(" ", r"\s+"), re.IGNORECASE)


_RELW = (r"(?P<rel>at least|at most|exactly|precisely|no more than|not more than|no less than|"
         r"no fewer than|not less than|not fewer than|less than|fewer than|more than|over|under|"
         r"a minimum of|a maximum of|minimum of|maximum of|up to|a total of|only|just)")
_REL_KIND = {"at least": "at_least", "no less than": "at_least", "no fewer than": "at_least",
             "not less than": "at_least", "not fewer than": "at_least", "a minimum of": "at_least",
             "minimum of": "at_least", "at most": "at_most", "no more than": "at_most",
             "not more than": "at_most", "a maximum of": "at_most", "maximum of": "at_most",
             "up to": "at_most", "exactly": "exactly", "precisely": "exactly", "a total of": "exactly",
             "only": "exactly", "just": "exactly", "less than": "lt", "fewer than": "lt",
             "under": "lt", "more than": "gt", "over": "gt"}
_SUF_KIND = {"or more": "at_least", "or fewer": "at_most", "or less": "at_most", "at least": "at_least",
             "at most": "at_most", "at the most": "at_most", "max": "at_most", "maximum": "at_most",
             "min": "at_least", "minimum": "at_least"}
_TIMES = rf"(?:(?P<num>{_NUM_BODY}) (?:times?|x)(?![\w-])|(?P<nw>once|twice|thrice)(?![\w-]))"
_TSUF = r"(?: (?P<suf>or more|or fewer|or less|at least|at most|at the most|max(?:imum)?|min(?:imum)?)(?![\w-]))?"
_MODAL_V = (r"(?:should|must|needs to|need to|has to|have to|is to|are to|will|shall|is required to|"
            r"are required to|ought to)(?: also| only| still)?")
_APPEAR = r"(?:appear|be used|be included|be mentioned|be repeated|occur|show up|feature|be present)"
_IN_RESPONSE = (r"(?: (?:in|throughout|within) (?:the|your|this) (?:entire |whole )?(?:response|answer|"
                r"reply|text|output|story|poem|essay|paragraph|email|letter|note|message))?")
_W_ATOM = (r'(?:"(?P<wq>[^"\n]{1,40})"|“(?P<wc>[^”\n]{1,40})”|‘(?P<ws>[^’\n]{1,40})’'
           r"|(?<![\w])'(?P<wa>[^'\n]{1,40})'(?![\w])|\[(?P<wb>[^\[\]\n]{1,40})\]"
           r"|\*\*?(?P<wt>[^*\n]{1,40})\*\*?|(?P<wbare>[A-Za-z][A-Za-z'’-]{0,30}))")
_KW_NOUNS1 = r"(?:key ?word|word|term|phrase|keyword|name)"
_FREQ_RULES = (
    # "In your response, the word X should appear at least 3 times."
    _rx(rf"\b(?:the )?{_KW_NOUNS1} {_W_ATOM} {_MODAL_V}(?P<neg> not)? {_APPEAR}{_IN_RESPONSE} "
        rf"(?:{_RELW} )?{_TIMES}{_TSUF}"),
    # "use the word X at least twice", "that uses the word "wheel" exactly twice"
    _rx(rf"\b(?:use|uses|using|include|includes|including|mention|mentions|mentioning|repeat|repeats|"
        r"repeating|say|says|saying|feature|features|featuring|contain|contains|containing|"
        r"incorporate|incorporates|incorporating|insert|put|have|has|having)(?: the)?"
        rf"(?P<noun> {_KW_NOUNS1})? {_W_ATOM}{_IN_RESPONSE} (?:{_RELW} )?{_TIMES}{_TSUF}"),
    # '"X" should appear twice' (a delimited word only)
    _rx(rf"(?:^|(?<=[\s,(])){_W_ATOM} {_MODAL_V}(?P<neg> not)? {_APPEAR}{_IN_RESPONSE} "
        rf"(?:{_RELW} )?{_TIMES}{_TSUF}"),
    # "(make sure) the word X appears at least 3 times"
    _rx(rf"\b(?:the )?{_KW_NOUNS1} {_W_ATOM} (?:appears|occurs|is used|is mentioned|is repeated|shows up)"
        rf"{_IN_RESPONSE} (?:{_RELW} )?{_TIMES}{_TSUF}"),
)
_L_ATOM = (r"(?:\"(?P<lq>[A-Za-z])\"|“(?P<lc>[A-Za-z])”|‘(?P<ls>[A-Za-z])’|(?<![\w])'(?P<la>[A-Za-z])'"
           r"(?![\w])|\[(?P<lb>[A-Za-z])\]|\*(?P<lt>[A-Za-z])\*|(?P<lbare>[A-Za-z])(?![\w'’-]))")
_LETTER_RULES = (
    _rx(rf"\b(?:the )?letter {_L_ATOM} {_MODAL_V}(?P<neg> not)? {_APPEAR}{_IN_RESPONSE}"
        rf"(?: (?:{_RELW} )?{_TIMES}{_TSUF})?"),
    _rx(rf"\b(?:use|uses|using|include|includes|including|contain|contains|containing|feature|"
        rf"features|have|has)(?: the)? letter {_L_ATOM}{_IN_RESPONSE} (?:{_RELW} )?{_TIMES}{_TSUF}"),
    _rx(rf"\b(?:the )?letter {_L_ATOM} (?:appears|occurs|is used|shows up){_IN_RESPONSE} "
        rf"(?:{_RELW} )?{_TIMES}{_TSUF}"),
)
_LIPOGRAM = _rx(r"\b(?:do not|don't|dont|never|without|avoid|avoiding|cannot|can't|must not|should not|"
                r"shouldn't)(?: (?:use|using|include|including|type|typing|write|writing))?(?: (?:the|any))?"
                rf" letter {_L_ATOM}(?![\w'])")
_CAPS = (r"(?:all )?(?:capital letters|capitals|caps|block letters|uppercase(?: letters)?|"
         r"upper case(?: letters)?|upper-case(?: letters)?)")
_CAPW_RULES = (
    # "In your response, words with all capital letters should appear at least 3 times."
    _rx(rf"\b(?:the )?(?:number of )?words? (?:with|in|written in|that are in|using|consisting of|made of) "
        rf"{_CAPS} {_MODAL_V}(?P<neg> not)? (?:appear|be used|occur|be included|show up|be present)"
        rf"{_IN_RESPONSE} (?:{_RELW} )?{_TIMES}{_TSUF}"),
    # "use at least 5 words in all caps", "include 3 all-caps words"
    _rx(rf"\b(?:use|uses|using|include|includes|including|contain|contains|containing|have|has|with|"
        rf"write|writing)(?: {_RELW})? (?P<num2>{_NUM_BODY})(?: \w+)? words? (?:in|with|written in) {_CAPS}\b"),
    _rx(rf"\b(?:use|uses|using|include|includes|including|contain|contains|containing|have|has|with|"
        rf"write)(?: {_RELW})? (?P<num2>{_NUM_BODY}) (?:all[- ]caps|all[- ]capitals?|fully capitali[sz]ed|"
        r"uppercase|upper-case) words\b"),
)
_H_NUM = rf"(?P<num>{_NUM_BODY}|a single|a|an)"
_HL_RULES = (
    _rx(rf"\bhighlight(?:s|ed|ing)?(?: {_RELW})? {_H_NUM} (?:(?:key|important|main|different|relevant|"
        r"separate|distinct|specific|crucial|essential|short) )?(?:sections?|parts?|phrases?|words?|"
        r"points?|portions?|passages?|sentences?|lines?|ideas?|terms?|items?|areas?|segments?|"
        r"key phrases?)(?![\w-])(?:[^.;\n]{0,60}?\bwith markdown\b)?"),
    _rx(rf"(?:{_RELW} )?\b{_H_NUM} (?:markdown[- ])?highlighted (?:sections?|parts?|phrases?|words?|"
        r"portions?|passages?|sentences?|points?|lines?|areas?|segments?)(?![\w-])"),
    _rx(rf"(?:{_RELW} )?\b{_H_NUM} (?:markdown )?highlights? (?:using|with|in) markdown\b"),
)
# "highlight the benefits using markdown": at least one highlight.
_HL_ANY = _rx(r"\bhighlight\b[^.;\n]{1,60}?\b(?:using|with|in) markdown\b")
_PH_RULE = _rx(rf"(?:{_RELW} )?\b(?P<num>{_NUM_BODY}|a single|a|an|one)(?: \w+){{0,2}}? placeholders?(?![\w-])")
_SEC_COUNT = _rx(rf"\b(?:have|has|having|contain|contains|containing|include|includes|consist of|consists of|"
                 r"be divided into|divided into|split into|divide (?:it |this |your (?:response|answer) )?into|"
                 r"organi[sz]ed? into|structured? into|in|into|with) (?:(?P<rel>exactly|at least|at most|"
                 rf"no more than|no fewer than|precisely) )?(?P<num>{_NUM_BODY}) (?:\w+ )?"
                 r"(?P<noun>sections?|parts?|chapters?)\b")
_SEC_MARKER = (r"[\"“‘'\[*]*(?P<marker>[A-Za-z][A-Za-z]{1,19})[\"”’'\]*]*(?:\s*(?:#|[0-9]+|\[[^\]\n]{1,20}\]|"
               r"\{[^}\n]{1,20}\})|\s+(?:X|N)|,?\s+(?:followed by|and then|then|and)\s+(?:a\s+|the\s+)?"
               r"(?:number|numbers|numeral|section number)|,?\s+and\s+number\s+them)(?![\w])")
_SEC_MARK_RULES = (
    _rx(r"\b(?:mark|label|title|start|begin|head|denote|indicate|introduce|precede|prefix)(?:s|ed)?"
        r"(?: the (?:beginning|start) of)? (?:each|every) (?P<noun>section|part|chapter)(?: of (?:your|the) "
        r"(?:response|answer))? (?:with|as|by|using)(?: the (?:word|label|keyword|marker|heading|header|"
        rf"prefix))?(?: of)? {_SEC_MARKER}"),
    _rx(r"\b(?:each|every) (?P<noun>section|part|chapter) (?:should|must|needs to|has to|will|shall) "
        r"(?:start|begin|be marked|be labell?ed|be introduced) (?:with|as|by)(?: the (?:word|label|keyword|"
        rf"marker|heading|header|prefix))? {_SEC_MARKER}"),
)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10}
_ORD_RE = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|[1-9](?:st|nd|rd|th)|10th)"
_FW_ATOM = (r"(?:\"(?P<fq>[A-Za-z][\w'’-]{0,29})[.,!?]?\"|“(?P<fc>[A-Za-z][\w'’-]{0,29})[.,!?]?”"
            r"|(?<![\w])'(?P<fa>[A-Za-z][\w-]{0,29})'(?![\w])|\[(?P<fb>[A-Za-z][\w'’-]{0,29})\]"
            r"|\*\*?(?P<ft>[A-Za-z][\w'’-]{0,29})\*\*?|(?P<fbare>[A-Za-z][\w'’-]{0,29})(?=\s*(?:[.,;:!?)]|$)))")
_FW_RULES = (
    _rx(r"\bparagraph (?:number )?(?P<n>[0-9]{1,2}|one|two|three|four|five|six|seven|eight|nine|ten) "
        r"(?:must|should|has to|needs to|will|shall|is to)(?: also)? (?:start|begin) with(?: the)?"
        rf"(?: (?:word|term))?:?\s*{_FW_ATOM}"),
    _rx(rf"\b(?:the )?(?P<ord>{_ORD_RE}) paragraph(?: of (?:your|the) (?:response|answer|reply|text|essay|"
        r"story))? (?:must|should|has to|needs to|will|shall|is to)(?: also)? (?:start|begin) with(?: the)?"
        rf"(?: (?:word|term))?:?\s*{_FW_ATOM}"),
    _rx(rf"\b(?:start|begin) (?:the|your) (?P<ord>{_ORD_RE}) paragraph with(?: the)?(?: (?:word|term))?:?\s*"
        rf"{_FW_ATOM}"),
    _rx(rf"\bthe first word of (?:the|your) (?P<ord>{_ORD_RE}) paragraph (?:must|should|has to|needs to|"
        rf"will|shall) be:?\s*{_FW_ATOM}"),
)
_DIVIDER_RULES = (
    _rx(r"\bparagraphs (?:are|should be|must be|will be|need to be|needs to be|have to be|shall be) "
        r"separated (?:from each other |with each other )?(?:with|by|using)(?: the| a)?(?: markdown)?"
        r"(?: (?:divider|separator|horizontal rule|line))?\s*:?\s*\*\*\*(?!\*)"),
    _rx(r"\bseparate (?:the |your |each |all |all the )?paragraphs (?:with|by|using)(?: the| a)?"
        r"(?: markdown)?(?: (?:divider|separator|horizontal rule))?\s*:?\s*\*\*\*(?!\*)"),
    _rx(r"\buse (?:the (?:markdown )?divider )?\*\*\*(?!\*) (?:to separate|between) (?:the |your )?paragraphs\b"),
)
_TITLE_RULES = (
    _rx(r"\btitle\b[^.;\n]{0,80}?(?:double angular brackets|double angle brackets|<<)"),
    _rx(r"(?:double angular brackets|double angle brackets)[^.;\n]{0,40}?\btitle\b"),
)
_PHRASE_ATOM = (r"(?:\"(?P<pq>[^\"\n]{1,120})\"|“(?P<pc>[^”\n]{1,120})”|‘(?P<ps>[^’\n]{1,120})’"
                r"|(?<![\w])'(?P<pa>[^'\n]{1,120})'(?![\w])|\[(?P<pb>[^\[\]\n]{1,120})\]"
                r"|\*\*(?P<pt2>[^*\n]{1,120})\*\*|\*(?P<pt>[^*\n]{1,120})\*)")
_PHRASE_RE = re.compile(_PHRASE_ATOM)
_PS_MARK = r"(?P<pm>P\.\s?P\.\s?S\.?|P\.\s?S\.?)"
_PS_RULES = (
    _rx(rf"\bpost-?script\b(?:(?:[^.;\n]{{0,50}}?\b(?:start(?:ing|s)?|begin(?:ning|s)?|marked|opening) with|"
        rf" (?:should|must|will|shall) be| is|:)\s*:?\s*(?:{_PHRASE_ATOM}|{_PS_MARK}))?"),
    _rx(rf"\b(?:add|include|end with|finish with|append|write|put|close with)(?: a| an)? {_PS_MARK}"
        r"(?=[\s,;:!?)]|$)"),
)
_QUOTE_OBJ = (r"(?:response|answer|reply|output|text|message|note|story|paragraph|poem|email|letter|"
              r"essay|summary|description|speech|post|tweet)")
_QUOTE_MARKS = r"(?:a pair of |a set of )?(?:double (?:quotation marks|quotes|quote marks)|quotation marks)\b"
_QUOTED_RULES = (
    _rx(rf"\b(?:wrap|enclose|put|place|surround|present|write)(?: your| the| this| my)?(?: entire| whole| full| "
        rf"complete)? {_QUOTE_OBJ}(?: [\w']+){{0,2}}? (?:with|in|inside|within|between|using) {_QUOTE_MARKS}"),
    _rx(rf"\b(?:(?:your|the|this)(?: entire| whole| full| complete)? {_QUOTE_OBJ}(?: [\w']+){{0,2}}?|it) "
        r"(?:should|must|needs to|has to|will|shall|is|are)(?: also| always| then)?(?: be)? (?:wrapped|enclosed|"
        rf"put|placed|contained|surrounded|written) (?:in|with|within|inside|between|by) {_QUOTE_MARKS}"),
    _rx(rf"\b(?:that|which) (?:is|are) (?:wrapped|enclosed|put|placed|contained|surrounded) (?:in|with|within|"
        rf"inside|between|by) {_QUOTE_MARKS}"),
    _rx(rf"\b{_QUOTE_OBJ}\b[^.;\n]{{0,80}}?\b(?:is|be) (?:wrapped|enclosed|put|placed|contained|surrounded) "
        rf"(?:in|with|within|inside|between|by) {_QUOTE_MARKS}"),
)
_TEXT_OBJ = (r"(?:response|answer|reply|message|text|email|letter|story|poem|essay|note|post|speech|"
             r"output|introduction|intro|summary|description|paragraph|tweet|review)")
_START_HEADS = (
    _rx(rf"\b(?:start|begin|open)(?:s)?(?: (?:your|the|this|my))?(?: (?:entire|whole))?(?: {_TEXT_OBJ})? "
        r"(?:with|by saying|by writing)(?: the (?:exact )?(?:word|phrase|words|sentence|greeting|line)s?)?"
        r"\s*:?\s*"),
    _rx(rf"\b(?:your|the|this)(?: (?:entire|whole))? {_TEXT_OBJ} (?:must|should|has to|needs to|will|shall)"
        r"(?: (?:always|also))? (?:start|begin|open) with(?: the (?:exact )?(?:word|phrase|words|sentence)s?)?"
        r"\s*:?\s*"),
    _rx(r"\bit (?:must|should|has to|needs to|will) (?:start|begin|open) with(?: the (?:exact )?(?:word|phrase|"
        r"words|sentence)s?)?\s*:?\s*"),
    _rx(r"\b(?:the )?first word (?:of (?:your|the) (?:response|answer|reply|message|text) )?(?:must|should|"
        r"has to|needs to|will|shall) be\s*:?\s*"),
)
_END_HEADS = (
    _rx(rf"\b(?:end|finish|conclude|close|complete|wrap up)(?: (?:your|the|this))?(?: (?:entire|whole))?"
        rf"(?: {_TEXT_OBJ})? (?:with|using|on)(?: (?:this|the|these|that|following))?(?: (?:exact|"
        r"following|precise|specific|same))?(?: (?:phrase|sentence|words?|line|statement|quote|quotation|"
        r"expression)s?)?\s*:?\s*"),
    _rx(rf"\b(?:your|the|this)(?: (?:entire|whole))? {_TEXT_OBJ} (?:must|should|has to|needs to|will|shall)"
        r"(?: (?:always|also))? (?:end|finish|conclude|close) with(?: (?:this|the|these|that|following))?"
        r"(?: (?:exact|following|precise|specific))?(?: (?:phrase|sentence|words?|line|statement)s?)?\s*:?\s*"),
    _rx(r"\b(?:the )?(?:last|final|closing) (?:sentence|line|words?|phrase) (?:of (?:your|the) (?:response|"
        r"answer|reply|message|text) )?(?:must|should|has to|needs to|will|shall) be(?: (?:the|this) "
        r"(?:exact )?(?:phrase|sentence|words?))?\s*:?\s*"),
)
_CLOSED_OPTION = r"(?:yes|no|maybe|true|false|agree|disagree|correct|incorrect|right|wrong)"
_CLOSED_LIST = re.compile(rf"(?P<opts>{_CLOSED_OPTION}(?:(?:\s*,\s*(?:or\s+)?|\s+or\s+|\s*/\s*)"
                          rf"{_CLOSED_OPTION})+)(?![\w-])", re.IGNORECASE)
_ONEOF_TEMPLATE = _rx(r"\b(?:answer|respond|reply|choose|pick|select)(?: (?:with|using))?(?: only)? "
                      r"(?:one of|from) the following(?: (?:options|choices|answers|responses|phrases))?\s*:?\s*")
_ONEOF_HEADS = (
    _rx(r"\b(?:answer|respond|reply|say)\b(?: (?:only|just|simply))?(?: (?:with|using))?(?: (?:only|just|"
        r"simply))?(?: (?:a|the) (?:single |one )?words?)?(?: (?:a|an))?(?: (?:simple|plain|straight))?"
        r"\s*:?\s*"),
    _rx(r"\b(?:a|an) (?:one-word|single-word|single word|one word) (?:answer|response|reply)\s*:?\s*"),
    _rx(r"\bwith (?:a single word|one word|only one word|just one word)\s*:?\s*"),
)
_ONEOF_ONLY = re.compile(r"\b(?:only|just|simply|nothing else|nothing more|no other|no explanation|"
                         r"without (?:any )?explanation|single word|one word|one-word|single-word)\b", re.I)
_RESP_COUNT = _rx(r"\b(?:give|provide|write|generate|offer|produce|create|draft|compose)(?: me| us)? "
                  r"(?P<n>two|2|three|3|four|4) (?:different |distinct |separate |alternative |unique )?"
                  r"(?:responses|answers|versions|replies|alternatives|drafts|variations|variants)\b")
_RESP_SEP = _rx(r"\bseparated (?:by|with|using)(?:[\w\s]{0,40}?)?\s*:?\s*(?P<sep>[*\-=_~#|+]{3,})")
_REPEAT_RE = re.compile(
    r"(?:\bfirst,?\s+|\bbefore (?:you )?(?:answer|respond|reply)(?:ing)?(?:\s+(?:it|this))?,?\s+)?"
    r"(?:please\s+)?\brepeat\s+(?:the|my|this|your)\s+(?:request|question|prompt|instruction|query)"
    r"(?:\s+(?:above|below))?(?:\s+(?:word\s+for\s+word|verbatim|exactly(?:\s+as\s+(?:it\s+is|written|"
    r"given))?|without\s+(?:any\s+)?changes?)){1,3}[^.()\n]{0,160}(?:\([^()]{0,400}\))?[ \t]*[.!]?",
    re.IGNORECASE)
_PARTS_THAT = re.compile(r"\b(?:sentences|lines|paragraphs|items|words|points|bullets|bullet points|names|"
                         r"questions|titles|verses|stanzas)\s+(?:that|which|each|all)?\s*\Z", re.I)
_RESPONSE_OBJ = re.compile(r"\b(?:your|the) (?:entire |whole )?(?:response|answer|reply|message)\b", re.I)
_LETS = re.compile(r"\b(?:let's|lets|let us|we(?:'ll| will| can| could| should)?|i(?:'ll| will| can| would)?)\s+\Z", re.I)
_NEG_RAW = re.compile(r"\b(?:not|never|no|don't|dont|doesn't|shouldn't|mustn't|cannot|can't|won't|without|"
                      r"avoid|avoiding)\b", re.IGNORECASE)
_TEMPLATE_VAR = re.compile(r"[a-z][a-z0-9_]{0,30}")
# The head of a template-variable definition, at a line start or after a sentence end.
_DEF_HEAD = re.compile(r"(?:^[ \t]*(?:[-*•][ \t]+)?|(?<=[.!?\"”])[ \t]+)\[([^\[\]\n]{1,40})\][ \t]*"
                       r"(?:is|are|=|:|-|–|—|will be|should be|can be|means)[ \t]*:?[ \t]*(?=\S)",
                       re.IGNORECASE | re.MULTILINE)
_EXTRA_GATE = ("time", "twice", "once", "thrice", "letter", "highlight", "placeholder", "capital", "caps",
               "upper", "section", "part", "chapter", "paragraph", "***", "title", "<<", "angular",
               "angle", "p.s", "p. s", "postscript", "post-script", "quot", "start", "begin", "first word",
               "open", "end", "finish", "conclude", "close", "complete", "wrap", "last", "final",
               "option", "yes", "true", "agree", "correct", "right", "wrong", "word", "separat", "only",
               "just", "simply")
_EXTRA_GATE_RE = re.compile("|".join(re.escape(k) for k in _EXTRA_GATE))


@dataclass(frozen=True)
class _Tables:
    """Template variables of a message: ``[name] are "a", "b"`` (quoted lists) and
    ``[name] - value`` lines."""

    quoted: dict
    lines: dict


def _template_tables(text: str) -> _Tables:
    """Definitions ``[name] is "value"`` (the quoted value, punctuation kept) and ``[name] -
    value`` (the rest of the line)."""
    lines = {}
    for m in _DEF_HEAD.finditer(text):
        q = _PHRASE_RE.match(text, m.end())
        if q and q.group(0)[0] in "\"“‘'":
            value = next(g for g in q.groups() if g is not None)
        else:
            value = text[m.end():].split("\n", 1)[0].strip()
        if value:
            lines.setdefault(m.group(1).strip().lower(), " ".join(value.split()))
    return _Tables(_placeholder_table(text), lines)


def _resolve_var(name: str, tables: _Tables, phrase: bool = False) -> str | None:
    """The value of a template variable ``[name]`` (one quoted value or a definition line;
    ``phrase``: the definition line first, which keeps its punctuation)."""
    key = name.strip().lower()
    if phrase and key in tables.lines:
        return tables.lines[key]
    quoted = tables.quoted.get(key)
    if quoted and len(quoted) == 1:
        return quoted[0]
    if quoted is None and key in tables.lines:
        return tables.lines[key]
    return None


def _atom_word(m: re.Match, tables: _Tables, prefix: str = "w") -> str | None:
    """The keyword of a word atom (quoted, bracketed, starred or bare); ``None`` when it is a
    stop word or an unfilled template variable."""
    groups = {k: m.group(prefix + k) for k in ("q", "c", "s", "a", "b", "t")}
    bare = m.group(prefix + "bare")
    if groups["b"] is not None:
        inner = groups["b"].strip()
        key = inner.lower()
        if key in tables.quoted or key in tables.lines:
            word = _resolve_var(key, tables)
        elif _TEMPLATE_NAME.fullmatch(key):
            return None
        else:
            word = inner
    elif bare is not None:
        if bare.lower() in _KW_STOP:
            return None
        word = bare
    else:
        word = next((v for v in groups.values() if v is not None), None)
    if word is None:
        return None
    word = " ".join(word.split()).strip(".,;:!?")
    return word if valid_keyword(word) else None


def _phrase_value(m: re.Match, tables: _Tables) -> str | None:
    """The phrase of a delimited atom (punctuation kept); a bracketed single lowercase name is a
    template variable, resolved or ``None``."""
    for g in ("pq", "pc", "ps", "pa", "pt2", "pt"):
        v = m.group(g)
        if v is not None:
            return " ".join(v.split())
    inner = " ".join(m.group("pb").split())
    if _TEMPLATE_VAR.fullmatch(inner):
        return _resolve_var(inner, tables, phrase=True)
    return inner


def _times_interval(m: re.Match, negated: bool = False, num_group: str = "num") -> Interval | None:
    """Interval of a "<relation> N times" phrase (``once / twice / thrice``; a bare count is
    exact); ``negated`` flips the relation ("should not appear more than twice")."""
    nw = m.groupdict().get("nw")
    n = {"once": 1, "twice": 2, "thrice": 3}[nw.lower()] if nw else parse_count(m.group(num_group) or "")
    if n is None:
        return None
    rel, suf = m.groupdict().get("rel"), m.groupdict().get("suf")
    if rel and suf:
        return None
    kind = _REL_KIND[" ".join(rel.lower().split())] if rel else (_SUF_KIND[" ".join(suf.lower().split())]
                                                              if suf else "exactly")
    if negated:
        kind = _FLIP[kind]
        if kind is None:
            return None
    return _interval(kind, n)


def _count_interval_of(m: re.Match, bare_kind: str, num_group: str = "num") -> Interval | None:
    """Interval of "<relation> N" (``a`` / ``an`` / ``a single`` = 1; a bare count gives
    ``bare_kind``)."""
    raw = (m.group(num_group) or "").lower()
    n = 1 if raw in ("a", "an", "a single", "one") else parse_count(raw)
    if n is None:
        return None
    rel = m.groupdict().get("rel")
    kind = _REL_KIND[" ".join(rel.lower().split())] if rel else bare_kind
    return _interval(kind, n)


def _neg_raw(raw: str, start: int) -> bool:
    """A negation shortly before ``start`` in the same clause of a raw sentence."""
    seg = re.split(r"[,;:.!?]\s", raw[max(0, start - 40):start] + " ")[-1]
    return bool(_NEG_RAW.search(seg))


def _instruction(norm: str) -> bool:
    """A sentence that instructs the assistant about its reply: not a conditional, an
    informational question, a statement or a sentence about code; a request verb, a modal or
    "I want" in the sentence or in one of its clauses."""
    if not norm or _CONDITIONAL.match(norm) or _NOT_A_REQUEST.match(norm):
        return False
    if _CODE_TOPIC.search(norm) and not _RESPONSE_SUBJ.search(norm):
        return False
    if _is_request(norm) and not _is_statement(norm):
        return True
    for a, b in _clauses(norm):
        clause = norm[a:b].strip()
        if _is_request(clause) and not _is_statement(clause):
            return True
    return False


def _phrase_list(s: str, pos: int, tables: _Tables, closed_ok: bool) -> tuple[list[str], bool]:
    """Options at ``s[pos]``: delimited phrases joined by ``or`` / commas (``("Yes", "No")``)
    or, with ``closed_ok``, plain words of a closed set ("yes, no, or maybe"). Returns
    ``(options, has_and)``."""
    out: list[str] = []
    m = _PHRASE_RE.match(s, pos)
    if m:
        while m:
            v = _phrase_value(m, tables)
            if v is None:
                return [], False
            out.append(v)
            sep = re.compile(r"\s*(?:,\s*(?:or\s+)?|or\s+|/\s*)", re.I).match(s, m.end())
            nxt = _PHRASE_RE.match(s, sep.end()) if sep else None
            if nxt is None:
                tail = re.compile(r"\s*(?:,\s*)?(?:and|&)\s+", re.I).match(s, m.end())
                return out, bool(tail and _PHRASE_RE.match(s, tail.end()))
            m = nxt
        return out, False
    if closed_ok:
        c = _CLOSED_LIST.match(s, pos)
        if c:
            return [w for w in re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+|\s*/\s*", c.group("opts"), flags=re.I) if w], False
    return [], False


def _sentence_extras(raw: str, norm: str, tables: _Tables) -> tuple[dict, list[tuple[int, int]]]:
    """Requests of the further instruction families in one sentence: ``raw`` (code masked,
    quotes intact) and its normalised, quote-masked form ``norm`` (for licensing). Returns the
    values found and the spans of ``raw`` they claim (masked before the count, keyword and
    format rules read the sentence)."""
    low = raw.lower()
    out: dict = {}
    spans: list[tuple[int, int]] = []
    if not _EXTRA_GATE_RE.search(low):
        return out, spans
    per_part = bool(_EACH.search(norm))
    if not _instruction(norm):
        # The format description "Paragraphs are separated with the markdown divider: ***"
        # opening its sentence is an instruction too.
        m = _DIVIDER_RULES[0].match(raw, len(raw) - len(raw.lstrip()))
        if m and not _NOT_A_REQUEST.match(norm) and not _CONDITIONAL.match(norm):
            return {"divider": True}, [(m.start(), m.end())]
        return out, spans

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in spans)

    def claim(m: re.Match) -> None:
        spans.append((m.start(), m.end()))

    if (("time" in low or "twice" in low or "once" in low or "thrice" in low)) and not per_part:
        if "letter" in low:
            for i, pat in enumerate(_LETTER_RULES):
                for m in pat.finditer(raw):
                    if not free(m.start(), m.end()):
                        continue
                    letter = next((m.group("l" + k) for k in ("q", "c", "s", "a", "b", "t", "bare")
                                   if m.group("l" + k) is not None), None)
                    neg = bool(m.groupdict().get("neg")) or (i == 1 and _neg_raw(raw, m.start()))
                    if m.group("num") is None and m.group("nw") is None:
                        iv = Interval(0, 0) if neg else Interval(1, None)
                    else:
                        iv = _times_interval(m, neg)
                    claim(m)
                    if letter and iv is not None:
                        out.setdefault("letters", []).append((letter.lower(), iv))
        if "capital" in low or "caps" in low or "upper" in low:
            for i, pat in enumerate(_CAPW_RULES):
                for m in pat.finditer(raw):
                    if not free(m.start(), m.end()):
                        continue
                    claim(m)
                    if i == 0:
                        iv = _times_interval(m, bool(m.group("neg")))
                    else:
                        iv = None if _neg_raw(raw, m.start()) else _count_interval_of(m, "at_least", "num2")
                    if iv is not None:
                        out["capital_words"] = iv
        for i, pat in enumerate(_FREQ_RULES):
            for m in pat.finditer(raw):
                if not free(m.start(), m.end()):
                    continue
                if i == 1 and m.group("wbare") is not None and not m.group("noun"):
                    continue                 # "repeat the process twice": not a word
                if i == 2 and m.group("wbare") is not None:
                    continue
                word = _atom_word(m, tables)
                neg = bool(m.groupdict().get("neg")) or (i in (1, 3) and _neg_raw(raw, m.start()))
                iv = _times_interval(m, neg)
                claim(m)
                if word is None or iv is None:
                    continue
                if iv.hi == 0:
                    out.setdefault("forbidden", []).append(word)
                else:
                    out.setdefault("frequencies", []).append((word, iv))
    if "letter" in low and not per_part:
        for m in _LETTER_RULES[0].finditer(raw):          # "the letter z should not appear"
            if free(m.start(), m.end()):
                letter = next((m.group("l" + k) for k in ("q", "c", "s", "a", "b", "t", "bare")
                               if m.group("l" + k) is not None), None)
                claim(m)
                if letter:
                    out.setdefault("letters", []).append(
                        (letter.lower(), Interval(0, 0) if m.group("neg") else Interval(1, None)))
        for m in _LIPOGRAM.finditer(raw):
            if free(m.start(), m.end()):
                letter = next((m.group("l" + k) for k in ("q", "c", "s", "a", "b", "t", "bare")
                               if m.group("l" + k) is not None), None)
                claim(m)
                if letter:
                    out.setdefault("letters", []).append((letter.lower(), Interval(0, 0)))
    if ("capital" in low or "caps" in low or "upper" in low) and "capital_words" not in out:
        for pat in _CAPW_RULES[1:]:
            for m in pat.finditer(raw):
                if free(m.start(), m.end()):
                    claim(m)
                    iv = None if _neg_raw(raw, m.start()) else _count_interval_of(m, "at_least", "num2")
                    if iv is not None:
                        out["capital_words"] = iv
    if "highlight" in low:
        for pat in _HL_RULES:
            for m in pat.finditer(raw):
                if free(m.start(), m.end()):
                    claim(m)
                    iv = None if _neg_raw(raw, m.start()) else _count_interval_of(m, "at_least")
                    if iv is not None and _valid_feature(iv):
                        out["highlights"] = iv
        m = _HL_ANY.search(raw) if "highlights" not in out else None
        if m and free(m.start(), m.end()) and not _neg_raw(raw, m.start()):
            claim(m)
            out["highlights"] = Interval(1, None)
    if "placeholder" in low and ("square bracket" in low or "[" in raw):
        for m in _PH_RULE.finditer(raw):
            if free(m.start(), m.end()):
                claim(m)
                iv = None if _neg_raw(raw, m.start()) else _count_interval_of(m, "at_least")
                if iv is not None and _valid_feature(iv):
                    out["placeholders"] = iv
    if "section" in low or "part" in low or "chapter" in low:
        for pat in _SEC_MARK_RULES:
            for m in pat.finditer(raw):
                if free(m.start(), m.end()):
                    claim(m)
                    out.setdefault("section_markers", []).append((m.group("noun").lower(), m.group("marker")))
        for m in _SEC_COUNT.finditer(raw):
            if free(m.start(), m.end()) and not _neg_raw(raw, m.start()):
                claim(m)
                iv = _count_interval_of(m, "exactly")
                if iv is not None and iv.hi != 0:
                    out.setdefault("section_counts", []).append((m.group("noun").lower().rstrip("s"), iv))
    if "paragraph" in low and ("start" in low or "begin" in low or "first word" in low):
        for pat in _FW_RULES:
            for m in pat.finditer(raw):
                if not free(m.start(), m.end()):
                    continue
                k = m.groupdict().get("n") or m.groupdict().get("ord")
                k = k.lower()
                n = (int(k) if k.isdigit() else NUMBER_WORDS.get(k) if k in NUMBER_WORDS
                     else _ORDINALS.get(k) or (int(k[:-2]) if k[:-2].isdigit() else None))
                word = next((m.group("f" + g) for g in ("q", "c", "a", "b", "t", "bare")
                             if m.group("f" + g) is not None), None)
                claim(m)
                if n and word:
                    fw = _FIRST_WORD.match(word)
                    if fw and fw.group(0) == word.rstrip("'’-"):
                        out["first_word"] = (n, fw.group(0).lower())
    if "***" in raw:
        for pat in _DIVIDER_RULES:
            m = pat.search(raw)
            if m and free(m.start(), m.end()):
                claim(m)
                out["divider"] = True
    if "title" in low and ("<<" in raw or "angular" in low or "angle bracket" in low):
        for pat in _TITLE_RULES:
            m = pat.search(raw)
            if m and not _neg_raw(raw, m.start()):
                out["title"] = True
    if "p.s" in low or "p. s" in low or "postscript" in low or "post-script" in low:
        for pat in _PS_RULES:
            for m in pat.finditer(raw):
                if not free(m.start(), m.end()) or _neg_raw(raw, m.start()):
                    continue
                marker = None
                if m.groupdict().get("pm"):
                    marker = " ".join(m.group("pm").split())
                elif any(m.groupdict().get(g) for g in ("pq", "pc", "ps", "pa", "pb", "pt2", "pt")):
                    marker = _phrase_value(m, tables)
                    if marker is not None:
                        marker = marker.strip("*_ ").rstrip(",;:")
                claim(m)
                if marker and valid_phrase(marker, 80) and marker[0].isalpha():
                    out["postscript"] = marker
                else:
                    out.setdefault("postscript", "P.S.")
    if "quot" in low:
        for pat in _QUOTED_RULES:
            m = pat.search(raw)
            if m and not _neg_raw(raw, m.start()):
                claim(m)
                out["quoted"] = True
    if "start" in low or "begin" in low or "open" in low or "first word" in low:
        sequence = bool(_SEQUENCE.search(norm))
        for pat in _START_HEADS:
            for m in pat.finditer(raw):
                if not free(m.start(), m.end()) or _neg_raw(raw, m.start()):
                    continue
                if sequence and not _RESPONSE_OBJ.search(m.group(0)):
                    continue                         # "a sentence that starts with X, then ..."
                if _LETS.search(raw, 0, m.start()) or _PARTS_THAT.search(raw, 0, m.start()):
                    continue                         # "let's start with ...", "sentences that start with"
                opts, has_and = _phrase_list(raw, m.end(), tables, closed_ok=True)
                opts = [re.sub(r"(?:\.\.\.|…|[.,;:])+\Z", "", o).strip() for o in opts]
                if (opts and not has_and and all(valid_phrase(o) for o in opts) and len(opts) <= 4
                        and not any(_PLACEHOLDER.search(o) for o in opts)):
                    claim(m)
                    out.setdefault("starts_with", []).append(tuple(opts))
    if any(k in low for k in ("end", "finish", "conclude", "close", "complete", "wrap", "last", "final")):
        for pat in _END_HEADS:
            for m in pat.finditer(raw):
                if not free(m.start(), m.end()) or _neg_raw(raw, m.start()):
                    continue
                pm = _PHRASE_RE.match(raw, m.end())
                phrase = _phrase_value(pm, tables) if pm else None
                if pm is None and "exact phrase" in m.group(0).lower():
                    rest = raw[m.end():].strip()       # "... this exact phrase Any other questions?."
                    if rest.endswith(".") and rest[-2:-1] in ("?", "!"):
                        rest = rest[:-1]
                    if rest[-1:] in ("?", "!") and len(rest.split()) <= 12:
                        phrase = rest
                if phrase and valid_phrase(phrase) and not _PLACEHOLDER.search(phrase):
                    spans.append((m.start(), pm.end() if pm else len(raw)))
                    out.setdefault("end_phrase", []).append(phrase)
    if any(k in low for k in ("option", "yes", "true", "false", "agree", "correct", "right", "wrong",
                              "one word", "one-word", "single word", "single-word", "only", "just",
                              "simply", "nothing else", "no other")):
        if _ONEOF_TEMPLATE.search(raw) is None:      # the template is read on the whole message
            restricted = bool(_ONEOF_ONLY.search(raw))
            for i, pat in enumerate(_ONEOF_HEADS):
                for m in pat.finditer(raw):
                    plain_answer = re.search(r"\b(?:simple|plain|straight)\s*:?\s*\Z", m.group(0), re.I)
                    if _neg_raw(raw, m.start()) or (i == 0 and not (restricted or plain_answer)):
                        continue
                    opts, has_and = _phrase_list(raw, m.end(), tables, closed_ok=True)
                    if len(opts) >= 2 and not has_and and all(valid_phrase(o) for o in opts):
                        out.setdefault("one_of", []).append(tuple(opts))
                        break
    return out, spans


def _valid_feature(iv: Interval) -> bool:
    return (not (iv.hi is None and iv.lo < 1) and iv.hi != 0
            and (iv.hi if iv.hi is not None else iv.lo) <= MAX_COUNT)


def _parse_options(rest: str, tables: _Tables) -> list[str]:
    """Options listed after "one of the following options:": delimited phrases (possibly in
    parentheses) separated by spaces, commas or "or", or plain phrases separated by commas or
    sentence ends. Plain options must look like a fixed set: at most 8 words each and the same
    first word, or at most 3 words each."""
    rest = rest.strip()
    if rest.startswith("("):
        close = rest.find(")")
        rest = rest[1:close] if close > 0 else rest[1:]
    rest = rest.strip()
    out: list[str] = []
    pos = 0
    while True:
        m = _PHRASE_RE.match(rest, pos)
        if m is None:
            break
        v = _phrase_value(m, tables)
        if v is None:
            return []
        out.append(v)
        sep = re.compile(r"\s*(?:,\s*(?:or\s+)?|or\s+|/\s*|\s+)", re.I).match(rest, m.end())
        if sep is None or sep.end() == len(rest):
            break
        pos = sep.end()
    if out:
        return out
    plain = rest.rstrip()
    if plain.startswith("- "):                       # "- She cleans dishes. - not enough information"
        parts = re.split(r"\s+-\s+", plain[2:])
    elif "," in plain or " or " in plain:
        parts = re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+", plain)
    else:
        parts = re.split(r"(?<=[.!?])\s+", plain)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2 or not all(valid_phrase(p) and len(p.split()) <= 8 for p in parts):
        return []
    same_start = len({p.split()[0].lower() for p in parts}) == 1
    return parts if same_start or all(len(p.split()) <= 3 for p in parts) else []


def _message_one_of(raw: str, tables: _Tables) -> tuple[str, ...] | None:
    """Options of "Answer with one of the following options: ..." (the IFEval template): the
    rest of the line, a template variable first (defined elsewhere, or followed by the
    options)."""
    if "following" not in raw.lower():
        return None
    m = _ONEOF_TEMPLATE.search(raw)
    if m is None or _neg_raw(raw, m.start()):
        return None
    end = raw.find("\n", m.end())
    rest = raw[m.end(): end if end >= 0 else len(raw)].strip()
    var = re.match(r"\[([a-z][a-z0-9_]{0,30})\]\s*\.?\s*", rest)
    if var:
        value = _resolve_var(var.group(1), tables, phrase=True)
        rest = value if value is not None else rest[var.end():]
    opts = _parse_options(rest, tables)
    if not 1 <= len(opts) <= 10 or not all(valid_phrase(o) for o in opts):
        return None
    if len({option_key(o) for o in opts}) != len(opts) or not all(option_key(o) for o in opts):
        return None
    return tuple(opts)


def _message_responses(masked: str, raw: str) -> tuple[str, int] | None:
    """``(separator, n)`` of a "give two different responses ... separated by ******" request."""
    low = raw.lower()
    if "separat" not in low:
        return None
    count = next((m for m in _RESP_COUNT.finditer(raw) if not _neg_raw(raw, m.start())), None)
    # not "paragraphs are separated with the markdown divider: ***"
    sep = next((m for m in _RESP_SEP.finditer(raw)
                if "paragraph" not in re.split(r"[.!?\n]\s", raw[max(0, m.start() - 60):m.start()])[-1].lower()),
               None)
    if count is None or sep is None:
        return None
    n = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}[count.group("n").lower()]
    return sep.group("sep"), n


def _message_repeat(text: str, masked: str) -> tuple[str, tuple[int, int]] | None:
    """The request to repeat (the message without the repeat instruction) and the span of the
    instruction, for "First repeat the request word for word without change, then give your
    answer (...)"."""
    if "repeat" not in masked.lower():
        return None
    m = _REPEAT_RE.search(masked)
    if m is None or _neg_raw(masked, m.start()):
        return None
    request = (text[: m.start()].strip() + " " + text[m.end():].strip()).strip()
    if not request or len(request) > MAX_REPEAT:
        return None
    return request, (m.start(), m.end())


def _blank(s: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return s
    chars = list(s)
    for a, b in spans:
        for i in range(max(0, a), min(len(chars), b)):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


# ============================================================================== compile
def _scan_region(text: str) -> str:
    if len(text) <= SCAN_CHARS:
        return text
    half = SCAN_CHARS // 2
    return text[:half] + "\n" + text[-half:]


def _intersect_all(ivs: Sequence[Interval]) -> Interval | None:
    acc: Interval | None = ivs[0]
    for iv in ivs[1:]:
        acc = acc.intersect(iv) if acc is not None else None
    return acc


def _one(values: Sequence, key=lambda v: v):
    """The value when every entry agrees (by ``key``), else ``None``."""
    if not values:
        return None
    keys = {key(v) for v in values}
    return values[0] if len(keys) == 1 else None


@functools.lru_cache(maxsize=16384)
def _compile_cached(text: str) -> ConstraintSpec:
    full = text
    text = _scan_region(text)
    code_masked = _mask_code(text)
    masked = _mask_quotes(code_masked)
    repeat = _message_repeat(text, masked) if len(full) <= SCAN_CHARS else None
    if repeat is not None:                          # the instruction is not read as a request
        code_masked = _blank(code_masked, [repeat[1]])
        masked = _blank(masked, [repeat[1]])
    inline_form = bool(_INLINE_FORM.search(_norm(masked)))
    placeholders = _placeholder_table(code_masked)
    tables = _template_tables(code_masked) if "[" in code_masked else _Tables({}, {})
    counts: dict[str, list[Interval]] = {}
    required: list[str] = []
    forbidden: list[str] = []
    flags: dict = {}
    cases: set[str] = set()
    styles: set[str] = set()
    extras: dict[str, list] = {}
    spans_all = _sentences(masked)
    norms = [_norm(masked[a:b]) for a, b in spans_all]
    code_message = any(_CODE_TOPIC.search(n) and _CODE_REQUEST.search(n) for n in norms)
    blocked: set[str] = set()                       # units counted inside multi-step sentences
    for n in norms:
        if _SEQUENCE.search(n):
            blocked |= {unit_of(m.group("u")) for m in _COUNT_UNIT_AT.finditer(n)}
        blocked |= {unit_of(m.group("u")) for m in _MORE_UNITS.finditer(n)}   # "in 120 more words"
    for (a, b), sentence in zip(spans_all, norms):
        raw, msk = code_masked[a:b], masked[a:b]
        found, spans = _sentence_extras(raw, sentence, tables)
        for key, value in found.items():
            if isinstance(value, list):
                extras.setdefault(key, []).extend(value)
            else:
                extras.setdefault(key, []).append(value)
        if spans:
            raw = _blank(raw, spans)
            full_sentence, sentence = sentence, _norm(_blank(msk, spans))
        else:
            full_sentence = sentence
        list_format = bool(_F_NUMBERED.search(sentence) or _F_BULLETED.search(sentence))
        for unit, iv, style in _sentence_counts(sentence, inline_form, list_format, code_message):
            counts.setdefault(unit, []).append(iv)
            if style:
                styles.add(style)
        code_task = _CODE_TOPIC.search(full_sentence) and not _RESPONSE_SUBJ.search(full_sentence)
        if not code_task and not _NOT_A_REQUEST.match(full_sentence):
            req, forb = _sentence_keywords(" ".join(raw.split()), placeholders,
                                           triggers=_is_strict_request(full_sentence)
                                           or _is_strict_request(sentence)
                                           or (bool(_WITHOUT_START.match(full_sentence))
                                               and _strict_somewhere(full_sentence)))
            required += req
            forbidden += forb
        fmt = _sentence_formats(sentence)
        if code_message and not _RESPONSE_SUBJ.search(full_sentence):
            fmt = {k: v for k, v in fmt.items() if k == "code_block"}   # the code's output, not the reply
        cases |= fmt.pop("case", set())
        styles |= fmt.pop("list_style", set())
        flags.update(fmt)

    values: dict = {}
    for unit, ivs in counts.items():
        acc = _intersect_all(ivs)
        if acc is not None and _valid_interval(unit, acc) and unit not in blocked:
            values[unit] = acc
    forbidden += extras.pop("forbidden", [])
    freq: dict[str, list] = {}
    for w, iv in extras.pop("frequencies", []):
        freq.setdefault(w.lower(), []).append((w, iv))
    for key, pairs in list(freq.items()):
        acc = _intersect_all([iv for _, iv in pairs])
        if acc == Interval(1, None):              # "at least once": a required keyword
            required.append(pairs[0][0])
            del freq[key]
    req_l, forb_l = {k.lower() for k in required}, {k.lower() for k in forbidden}
    both = (req_l & forb_l) | (set(freq) & forb_l)

    def clean(words: list[str]) -> tuple[str, ...]:
        seen, out = set(), []
        for k in words:
            if k.lower() in both or k.lower() in seen or not valid_keyword(k):
                continue
            seen.add(k.lower())
            out.append(k)
        return tuple(out[:MAX_KEYWORDS])

    values["keywords"], values["forbidden"] = clean(required), clean(forbidden)
    values["frequencies"] = tuple(
        (pairs[0][0], acc) for key, pairs in freq.items() if key not in both
        and (acc := _intersect_all([iv for _, iv in pairs])) is not None)[:MAX_KEYWORDS]
    letters: dict[str, list[Interval]] = {}
    for ch, iv in extras.pop("letters", []):
        letters.setdefault(ch, []).append(iv)
    values["letters"] = tuple((ch, acc) for ch, ivs in letters.items()
                              if (acc := _intersect_all(ivs)) is not None)[:MAX_KEYWORDS]
    for name in FEATURE_COUNTS:
        ivs = extras.pop(name, [])
        acc = _intersect_all(ivs) if ivs else None
        if acc is not None and _valid_feature(acc):
            values[name] = acc
    markers = extras.pop("section_markers", [])
    sec_counts = extras.pop("section_counts", [])
    marker = _one(markers, key=lambda v: (v[0], v[1].lower()))
    if marker is not None:
        ivs = [iv for noun, iv in sec_counts if noun == marker[0]]
        acc = _intersect_all(ivs) if ivs else None
        if acc is not None and acc.hi != 0 and _valid_feature(acc):
            values["sections"] = (marker[1], acc)
    for name in ("first_word", "end_phrase"):
        v = _one(extras.pop(name, []), key=lambda x: x if name == "first_word" else _ws(x).casefold())
        if v is not None:
            values[name] = v
    ps = extras.pop("postscript", [])
    explicit = [p for p in ps if p != "P.S."] or ps
    v = _one(explicit, key=lambda x: _ws(x).casefold())
    if v is not None:
        values["postscript"] = v
    one_of = _message_one_of(code_masked, tables)
    if one_of is not None:
        extras.setdefault("one_of", []).append(one_of)
    for name, key in (("starts_with", lambda o: tuple(_ws(x).casefold() for x in o)),
                      ("one_of", lambda o: tuple(sorted(option_key(x) for x in o)))):
        v = _one(extras.pop(name, []), key=key)
        if v is not None and len({key((x,)) for x in v}) == len(v):
            values[name] = tuple(v)
    for name in ("divider", "title", "quoted"):
        if extras.pop(name, None):
            values[name] = True
    responses = _message_responses(masked, code_masked)
    if responses is not None and _SEPARATOR_OK.fullmatch(responses[0]):
        values["responses"] = responses
    if repeat is not None:
        values["repeat"] = repeat[0]
    if len(cases) == 1:
        values["case"] = cases.pop()
    if len(styles) == 1:
        items = values.get("items")
        if items is not None and items.hi == 0:
            values.pop("items")                # "no bullet points; use a numbered list"
        values["list_style"] = styles.pop()
    values.update(flags)
    try:
        return ConstraintSpec(**values)
    except ValueError:
        return _drop_conflicts(values)


def _drop_conflicts(values: dict) -> ConstraintSpec:
    """The spec without the fields that make ``values`` inconsistent (dropped one at a time,
    newest families first)."""
    order = ("repeat", "responses", "one_of", "end_phrase", "starts_with", "quoted", "postscript",
             "title", "divider", "first_word", "sections", "capital_words", "placeholders",
             "highlights", "letters", "frequencies", "list_style")
    values = dict(values)
    for name in order:
        values.pop(name, None)
        try:
            return ConstraintSpec(**values)
        except ValueError:
            continue
    return EMPTY


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
    None: (("list item", "list items"),),
    "bullet": (("bullet point", "bullet points"), ("bullet point", "bullets"), ("bulleted item", "bulleted items")),
    "numbered": (("numbered item", "numbered items"), ("numbered point", "numbered points")),
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
        return "bullet" if style == "bullet" else "item"
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
        if unit != "words" or n <= BARE_EXACT_WORDS:
            bare = f"{_num_text(n, rng)} {_noun(unit, n, rng, style)}"
            if unit != "words":
                sentences.append(_choice(rng, _NP_FRAMES).replace("{np}", bare).replace("{NP}", bare.capitalize()))
            sentences.append(_choice(rng, _STRONG_FRAMES).replace("{np}", bare))
            adj = f"{_num_text(n, rng) if n <= 20 else n}-{_adj_noun(unit, style, rng)}"
            frame = _choice(rng, _ADJ_FRAMES)
            if unit == "items":
                frame = _choice(rng, ("Answer with a {adj} list.", "Format your answer as a {adj} list.")
                                if style != "numbered" else ("Answer with a {adj} numbered list.",))
            sentences.append(frame.replace("{adj}", adj))
            if unit == "words":
                sentences.append(_choice(rng, ("Answer with {b}.", "Reply with just {b}.", "Respond with {b}."))
                                 .replace("{b}", bare))
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
        if unit == "words" and n > BARE_EXACT_WORDS:
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


def _times_text(kind: str, n: int, rng: np.random.Generator) -> str:
    """"at least 3 times", "exactly twice", "fewer than 4 times", ..."""
    if kind == "exactly" and n <= 3 and rng.random() < 0.5:
        word = ("once", "twice", "thrice")[n - 1] if n < 3 or rng.random() < 0.3 else f"{_num_text(n, rng)} times"
        return _choice(rng, ("exactly ", "")) + word
    count = (("once", "twice")[n - 1] if n <= 2 and rng.random() < 0.5
             else f"{_num_text(n, rng)} time{'s' if n != 1 else ''}")
    rel = {"exactly": ("exactly ", "precisely ", ""), "at_least": ("at least ", "no fewer than "),
           "at_most": ("at most ", "no more than "), "lt": ("less than ", "fewer than ")}[kind]
    return _choice(rng, rel) + count


def _interval_phrases(iv: Interval) -> list[tuple[str, int]]:
    """``(relation, n)`` phrases whose conjunction is ``iv`` ("between" needs two)."""
    if iv.kind == "exactly":
        return [("exactly", iv.lo)]
    if iv.kind == "at_least":
        return [("at_least", iv.lo)]
    if iv.kind == "at_most":
        return [("at_most", iv.hi)] if iv.hi > 1 else [("lt", iv.hi + 1)] if iv.hi == 0 else [("at_most", 1)]
    return [("at_least", iv.lo), ("at_most", iv.hi)]


def _render_frequency(word: str, iv: Interval, rng: np.random.Generator) -> str:
    out = []
    for kind, n in _interval_phrases(iv):
        if kind == "at_most" and rng.random() < 0.4:
            kind, n = "lt", n + 1
        q = _quote(word, rng)
        frame = _choice(rng, ("In your response, the word {q} should appear {t}.", "Use the word {q} {t}.",
                              "Make sure the word {q} appears {t}.", "The word {q} must appear {t} in your answer.",
                              "Mention {q} {t}."))
        out.append(frame.replace("{q}", q).replace("{t}", _times_text(kind, n, rng)))
    return " ".join(out)


def _render_letter(letter: str, iv: Interval, rng: np.random.Generator) -> str:
    q = _choice(rng, (f"'{letter}'", f'"{letter}"', f"[{letter}]", letter))
    if iv.hi == 0:
        return _choice(rng, ("Do not use the letter {q}.", "Write without using the letter {q}.",
                             "The letter {q} should not appear in your response.")).replace("{q}", q)
    out = []
    for kind, n in _interval_phrases(iv):
        frame = _choice(rng, ("In your response, the letter {q} should appear {t}.", "Use the letter {q} {t}.",
                              "Make sure the letter {q} appears {t}."))
        out.append(frame.replace("{q}", q).replace("{t}", _times_text(kind, n, rng)))
    return " ".join(out)


def _rel_text(kind: str, n: int, rng: np.random.Generator) -> str:
    rel = {"exactly": ("exactly ",), "at_least": ("at least ", "no fewer than "),
           "at_most": ("at most ", "no more than "), "lt": ("fewer than ", "less than ")}[kind]
    return _choice(rng, rel) + _num_text(n, rng)


def _render_feature(name: str, iv: Interval, rng: np.random.Generator) -> str:
    out = []
    for kind, n in _interval_phrases(iv):
        r = _rel_text(kind, n, rng)
        noun_n = n
        if name == "highlights":
            what = _choice(rng, ("sections", "key phrases", "parts")) if noun_n != 1 else _choice(rng, ("section", "key phrase", "part"))
            frame = _choice(rng, ("Highlight {r} {w} in your answer with markdown, i.e. *highlighted section*.",
                                  "Use markdown to highlight {r} {w}, like *this*.",
                                  "Your answer must contain {r} highlighted {w}, such as *highlighted section*."))
            if "highlighted {w}" in frame and what.startswith("key"):
                what = "sections" if noun_n != 1 else "section"
        elif name == "placeholders":
            what = "placeholders" if noun_n != 1 else "placeholder"
            frame = _choice(rng, ("The response must contain {r} {w} represented by square brackets, such as [address].",
                                  "Include {r} {w} in square brackets, like [name].",
                                  "Your answer should contain {r} {w} in square brackets, such as [date]."))
        else:
            what = "words" if noun_n != 1 else "word"
            if kind == "exactly" or rng.random() < 0.5:
                frame = "In your response, words with all capital letters should appear {t}."
                out.append(frame.replace("{t}", _times_text(kind, n, rng)))
                continue
            frame = _choice(rng, ("Use {r} {w} in all capital letters.", "Include {r} {w} written in all caps."))
        out.append(frame.replace("{r}", r).replace("{w}", what))
    return " ".join(out)


def _render_sections(marker: str, iv: Interval, rng: np.random.Generator) -> str:
    noun = "section"
    counts = [f"{_rel_text(kind, n, rng) if kind != 'exactly' or rng.random() < 0.3 else _num_text(n, rng)} "
              f"{noun}{'s' if n != 1 else ''}" for kind, n in _interval_phrases(iv)]
    count = " and ".join(counts) if len(counts) == 1 else f"{counts[0]}. Your response must have {counts[1]}"
    head = _choice(rng, ("Your response must have {c}.", "Divide your answer into {c}.",
                         "The response should have {c}.")).replace("{c}", count)
    mark = _choice(rng, ("Mark the beginning of each section with {m} X, such as: {m} 1.",
                         "Start each section with {m} X.", "Label each section with {m} N."))
    return head + " " + mark.replace("{m}", marker)


_ORDINAL_WORDS = {v: k for k, v in _ORDINALS.items()}


def _render_first_word(n: int, word: str, rng: np.random.Generator) -> str:
    w = _choice(rng, (word, f'"{word}"', f"'{word}'"))
    if n in _ORDINAL_WORDS and rng.random() < 0.6:
        return _choice(rng, ("The {o} paragraph must start with the word {w}.",
                             "Start the {o} paragraph with the word {w}.")).replace("{o}", _ORDINAL_WORDS[n]).replace("{w}", w)
    return f"Paragraph {n} must start with word {w}."


def _render_phrase_list(options: tuple[str, ...], rng: np.random.Generator) -> str:
    q = ['"' + o + '"' if '"' not in o else "“" + o + "”" for o in options]
    return _join(q, rng, "or") if len(q) > 1 else q[0]


_JSON_T = ("Answer in JSON.", "Format your entire response as valid JSON.", "Respond only with a JSON object.",
           "Your entire output should be in JSON format.", "Return your answer as JSON.",
           "Please reply in valid JSON.", "Output only the JSON.")
_CODE_T = ("Put your answer in a code block.", "Wrap the code in a markdown code block.", "Use a fenced code block.",
           "Format the code inside a code block.", "Present your answer within a code block.")
_LOWER_T = ("Write your entire response in lowercase letters.", "Use only lowercase letters.",
            "Your answer must be in all lowercase, with no capital letters.", "Respond in all lowercase.",
            "No capital letters are allowed.")
_UPPER_T = ("Write your entire response in capital letters.", "Respond in all caps.", "Use only uppercase letters.",
            "Your entire answer should be in English and in all capital letters.")
_QUESTION_T = ("End your response with a question.", "Finish your answer with a question.",
               "Your reply should end with a question.", "Conclude with a follow-up question.",
               "Make sure the last sentence is a question.")
_COMMAS_T = ("Do not use any commas.", "Avoid commas in your response.", "Write without using commas.",
             "Refrain from the use of any commas.", "No commas are allowed.")
_STYLE_T = {"numbered": ("Use a numbered list.", "Format your answer as a numbered list.",
                         "Present your answer as a numbered list."),
            "bullet": ("Use bullet points.", "Format your answer as a bulleted list.", "Answer in bullet points.",
                       "Use the markdown bullet points such as: * This is point 1.")}
_DIVIDER_T = ("Paragraphs are separated with the markdown divider: ***", "Separate paragraphs with the markdown divider ***.",
              "Use *** to separate paragraphs.")
_TITLE_T = ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
            "Include a title wrapped in double angular brackets, like <<title>>.",
            "Give your response a title in double angular brackets, such as <<my title>>.")
_QUOTED_T = ("Wrap your entire response with double quotation marks.", "Your entire answer should be wrapped in double quotes.",
             "Put your whole reply in double quotation marks.")
REPEAT_INSTRUCTIONS = _REPEAT_T = ("First repeat the request word for word without change, then give your answer (1. do not say any words "
             "or characters before repeating the request; 2. the request you need to repeat does not include this "
             "sentence)",
             "First, repeat the request without change, then give your answer (do not say anything before repeating "
             "the request; the request you need to repeat does not include this sentence)",
             "Before answering, repeat the request word for word.")


def _render_postscript(marker: str, rng: np.random.Generator) -> str:
    plain = re.fullmatch(r"P\.(?:P\.)?S\.", marker, re.I) is not None
    m = marker if plain and rng.random() < 0.7 else f'"{marker}"'
    return _choice(rng, ("At the end of your response, please explicitly add a postscript starting with {m}",
                         "Add a postscript that starts with {m}", "Include a postscript starting with {m}")
                   ).replace("{m}", m) + ("" if m.endswith(".") else ".")


def _render_start(options: tuple[str, ...], rng: np.random.Generator) -> str:
    lst = _render_phrase_list(options, rng)
    single_word = len(options) == 1 and len(options[0].split()) == 1
    frames = ["Begin your answer with {l}.", "Your response must start with {l}.", "Start your reply with {l}."]
    if single_word:
        frames += ["Start your response with the word {l}.", "The first word of your response should be {l}."]
    elif len(options) == 1:
        frames += ["Start your response with the phrase {l}."]
    return _choice(rng, frames).replace("{l}", lst)


def _render_end(phrase: str, rng: np.random.Generator) -> str:
    q = '"' + phrase + '"' if '"' not in phrase else "“" + phrase + "”"
    return _choice(rng, ("Finish your response with this exact phrase {q}. No other words should follow this phrase.",
                         "End your answer with the phrase {q}.", "Your response must end with the words {q}.",
                         "Conclude your reply with {q}.")).replace("{q}", q)


def _render_one_of(options: tuple[str, ...], rng: np.random.Generator) -> str:
    q = ['"' + o + '"' if '"' not in o else "“" + o + "”" for o in options]
    if len(options) == 1:
        return f"Answer with one of the following options: ({q[0]})."
    return _choice(rng, ("Answer with one of the following options: ({l}).",
                         "Respond with only {o}.", "Reply with just {o}.")
                   ).replace("{l}", ", ".join(q)).replace("{o}", _join(q, rng, "or"))


def _render_responses(sep: str, n: int, rng: np.random.Generator) -> str:
    word = {2: "two", 3: "three", 4: "four"}[n]
    if sep == "******" and n == 2 and rng.random() < 0.6:
        return ("Give two different responses. Responses and only responses should be separated by 6 asterisk "
                "symbols: ******.")
    return _choice(rng, ("Give {w} different responses, separated by {s}", "Provide {w} different answers "
                         "separated by {s}")).replace("{w}", word).replace("{s}", sep) + "."


def render(spec: ConstraintSpec, rng: np.random.Generator | None = None, *,
           item_style: str | None = None) -> str:
    """Natural instruction text for ``spec``: one sentence (or two) per constraint (keywords
    grouped), chosen among several templates and shuffled; ``compile(render(spec, rng)) ==
    spec``.

    The list style of item counts comes from ``spec.list_style`` ("bullet points", "numbered
    items"; neutral "list items" without one). ``item_style`` is accepted for compatibility and
    ignored. A spec with ``repeat`` renders as the request itself followed by the repeat
    instruction, so its other constraints must be those of the request (``ValueError``
    otherwise).
    """
    del item_style
    rng = np.random.default_rng() if rng is None else rng
    if spec.repeat is not None:
        rest = dataclasses.replace(spec, repeat=None)
        if compile(spec.repeat) != rest:
            raise ValueError("a spec with `repeat` must have the constraints of its request")
        return spec.repeat + _choice(rng, (" ", "\n\n")) + _choice(rng, _REPEAT_T)
    style = spec.list_style
    pieces = [_render_count(u, getattr(spec, u), rng, style if u == "items" else None)
              for u in COUNT_UNITS if getattr(spec, u) is not None]
    if style and spec.items is None:
        pieces.append(_choice(rng, _STYLE_T[style]))
    if spec.keywords:
        pieces.append(_render_keywords(spec.keywords, rng))
    if spec.forbidden:
        pieces.append(_render_forbidden(spec.forbidden, rng))
    for flag, options in (("json_output", _JSON_T), ("code_block", _CODE_T),
                          ("end_question", _QUESTION_T), ("no_commas", _COMMAS_T),
                          ("divider", _DIVIDER_T), ("title", _TITLE_T), ("quoted", _QUOTED_T)):
        if getattr(spec, flag):
            pieces.append(_choice(rng, options))
    if spec.case:
        pieces.append(_choice(rng, _LOWER_T if spec.case == "lower" else _UPPER_T))
    pieces += [_render_frequency(w, iv, rng) for w, iv in spec.frequencies]
    pieces += [_render_letter(ch, iv, rng) for ch, iv in spec.letters]
    pieces += [_render_feature(name, getattr(spec, name), rng) for name in FEATURE_COUNTS
               if getattr(spec, name) is not None]
    if spec.sections:
        pieces.append(_render_sections(spec.sections[0], spec.sections[1], rng))
    if spec.first_word:
        pieces.append(_render_first_word(spec.first_word[0], spec.first_word[1], rng))
    if spec.postscript:
        pieces.append(_render_postscript(spec.postscript, rng))
    if spec.starts_with:
        pieces.append(_render_start(spec.starts_with, rng))
    if spec.end_phrase:
        pieces.append(_render_end(spec.end_phrase, rng))
    if spec.one_of:
        pieces.append(_render_one_of(spec.one_of, rng))
    if spec.responses:
        pieces.append(_render_responses(spec.responses[0], spec.responses[1], rng))
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
           "compile_from_tokens", "IncrementalTargets", "NUMBER_WORDS", "WORD_OF_NUMBER",
           "TokenLocal", "token_local", "TOKEN_LOCAL_NAMES", "nonblank_line_step", "has_forbidden_bytes",
           "paragraph_texts", "count_highlights", "count_placeholders", "has_title", "is_quoted",
           "capital_word_count", "letter_count", "first_word", "starts_with_phrase", "ends_with_phrase",
           "option_key", "matches_option", "has_postscript", "count_sections", "split_responses",
           "repeats_request", "valid_phrase", "style_of", "LIST_STYLES", "FLAG_FIELDS", "FEATURE_COUNTS",
           "BARE_EXACT_WORDS", "REPEAT_INSTRUCTIONS"]
