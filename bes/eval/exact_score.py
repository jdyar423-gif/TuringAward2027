"""Objective scoring of replies that needs no judge.

- **Math**: :func:`extract_answer` reads the final stated number of a reply (a ``\\boxed{}``
  value, else the number after the last "answer" keyword, else the last bold number, else the
  last number; number words are used only when the reply has no digits) and
  :func:`math_correct` compares it with the reference exactly, tolerant to formatting (thousands
  separators, currency and percent signs, units, LaTeX fractions, trailing zeros, rounding to
  the reference's precision).
- **Constraints**: :func:`check_constraint` verifies one machine-checkable constraint of the
  prompt set (types in :data:`CONSTRAINT_TYPES`). Counts of words, sentences, lines and list
  items come from the project's exact layer when it is importable
  (``bes.exact.constraints.measure``, else ``bes.exact.transducer.measure_counts``), otherwise
  from :func:`local_counts`, a small implementation of the same definitions
  (``docs/SPEC.md`` section 2.1). :func:`count_backend` says which one is in use. When
  ``bes.exact.constraints`` provides ``compile`` and ``verify``, :func:`project_check` also runs
  the project's own constraint compiler on the user message, as a cross-check.
- **Degenerate output**: :func:`degenerate_stats` (empty, cut off at the token limit, looping
  tail, repeated lines, repeated 4-gram rate, unclosed code fence, leaked role markers, length).

Everything is plain Python; numbers are compared as exact fractions.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

# ------------------------------------------------------------------------------- numbers
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

# A number: optional sign, digits (with optional thousands groups) or a leading-dot decimal,
# optional decimal part, optional exponent, optional "/denominator", optional percent sign.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-+])?\$?\s?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)?(?P<frac>\.\d+)?"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
    r"(?:\s?/\s?(?P<den>\d+(?:\.\d+)?))?"
    r"(?P<pct>\s?%)?"
)


def _normalise(text: str) -> str:
    """Make numbers easy to find: unicode minus, LaTeX fractions and spacing, ``{,}``."""
    t = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u00a0", " ")
    t = t.replace("{,}", ",").replace("\\,", "").replace("\\!", "").replace("\\%", "%")
    t = t.replace("\\$", "$").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)/\(\s*(\d+(?:\.\d+)?)\s*\)", r"\1/\2", t)
    return t


def _to_fraction(m: re.Match) -> Fraction | None:
    whole, frac = m.group("int"), m.group("frac")
    if whole is None and frac is None:
        return None
    try:
        value = Fraction((whole or "0").replace(",", "") + (frac or ""))
        if m.group("exp"):
            exp = int(m.group("exp"))
            if abs(exp) > 30:
                return None
            value *= Fraction(10) ** exp
        if m.group("den"):
            den = Fraction(m.group("den"))
            if den == 0:
                return None
            value /= den
    except (ValueError, ZeroDivisionError):
        return None
    return -value if m.group("sign") == "-" else value


def find_numbers(text: str) -> list[tuple[Fraction, int, int, bool]]:
    """Every number in ``text`` as ``(value, start, end, percent)`` in order of appearance.
    A ``-`` counts as a sign only at the start of a token (so ``2-step`` holds ``2``)."""
    out = []
    for m in _NUMBER.finditer(text):
        value = _to_fraction(m)
        if value is None:
            continue
        start = m.start()
        if m.group("sign") == "-" and start > 0 and (text[start - 1].isalnum() or text[start - 1] == ")"):
            value = -value  # "3-2": the minus is an operator, not a sign
            start += 1
        out.append((value, start, m.end(), bool(m.group("pct"))))
    return out


def parse_number(text: str) -> Fraction | None:
    """The single number in ``text`` (``"1,234.50"``, ``"-3/4"``, ``"25%"``, ``"$12"``,
    ``"twelve"``), else ``None``."""
    nums = find_numbers(_normalise(text))
    if len(nums) == 1:
        return nums[0][0]
    if not nums:
        return parse_number_words(text)
    return None


def parse_number_words(text: str) -> Fraction | None:
    """The last English number phrase in ``text`` (``"twenty-five"``, ``"one hundred and
    two"``, ``"three thousand"``), else ``None``. Integers only."""
    words = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    best = None
    i = 0
    while i < len(words):
        if words[i] not in _NUM_WORDS and words[i] not in _SCALE_WORDS:
            i += 1
            continue
        total, current, j, seen = 0, 0, i, False
        while j < len(words):
            w = words[j]
            if w in _NUM_WORDS:
                current += _NUM_WORDS[w]
                seen = True
            elif w in _SCALE_WORDS and seen:
                if _SCALE_WORDS[w] == 100:
                    current *= 100
                else:
                    total += current * _SCALE_WORDS[w]
                    current = 0
            elif w == "and" and seen and j + 1 < len(words) and words[j + 1] in _NUM_WORDS:
                pass
            else:
                break
            j += 1
        if seen:
            best = Fraction(total + current)
        i = max(j, i + 1)
    return best


def _boxed_contents(text: str) -> list[str]:
    out, i = [], 0
    while (k := text.find("\\boxed", i)) >= 0:
        j = text.find("{", k)
        if j < 0:
            break
        depth, m = 0, j
        while m < len(text):
            if text[m] == "{":
                depth += 1
            elif text[m] == "}":
                depth -= 1
                if depth == 0:
                    break
            m += 1
        out.append(text[j + 1:m])
        i = m + 1
    return out


_LABEL_BEFORE = re.compile(r"(?i)\b(?:step|part|method|case|option|example|question|problem|no\.)\s*$")


def _is_label(text: str, start: int, end: int) -> bool:
    """A number that names something rather than states a value: ``Step 2``, ``Part 3``, or a
    numbered-list marker (digits at the start of a line followed by ``.`` or ``)`` and a space)."""
    if _LABEL_BEFORE.search(text[max(0, start - 12):start]):
        return True
    line_start = text.rfind("\n", 0, start) + 1
    return (not text[line_start:start].strip() and end < len(text) and text[end] in ".)"
            and (end + 1 == len(text) or text[end + 1] in " \t"))


#: "answer is", "answer:", "answer =", "final answer would be" (a statement of the answer).
_ANSWER_KEY = re.compile(r"(?i)\banswer\b\s*(?:is|would\s+be|will\s+be|[:=])\s*(?:is\s*)?")
_BOLD = re.compile(r"\*\*([^*\n]{1,60})\*\*")
_HEADING_WORDS = re.compile(r"(?i)^\s*(?:step|part|method|case|option|example|question|problem)\b")
#: Words that open or mark a concluding statement.
_CONCLUSION = re.compile(
    r"(?i)\b(?:so|therefore|thus|hence|in total|altogether|overall|in summary|in conclusion|finally|"
    r"that means|this means|which means|confirms|indeed|the result|the total|that gives|which gives|"
    r"this gives|comes to|we get|you get)\b")
#: A sentence that checks an earlier result ("Check: 12 x 80 = 960.", "To verify, ...").
_CHECK = re.compile(
    r"(?i)^\W*(?:(?:let'?s|let us|we can|now|and)\s+)?(?:check|checking|verify|verifying|verification|"
    r"double[- ]check|sanity check|confirm|confirming|to (?:check|verify|confirm)|plugging|substituting)\b")
#: Sentence ends: whitespace after a run of ``.?!``, except after a factorial (``7! = 5040``).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?<!\d!)\s+|\n+")


def context_values(texts: Sequence[str]) -> set[Fraction]:
    """Numbers (digits or single number words) that appear in ``texts`` (the user turns and the
    model's own earlier answers): restating them is not answering."""
    out: set[Fraction] = set()
    for text in texts:
        out.update(v for v, *_ in find_numbers(_normalise(text)))
        out.update(Fraction(_NUM_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in _NUM_WORDS)
    return out


def extract_answer(text: str, context: set[Fraction] | None = None) -> dict[str, Any]:
    """The final stated number of a reply.

    Returns ``{"value": Fraction | None, "values": [Fraction], "method": ..., "percent": bool}``,
    trying in order:

    1. ``boxed``: the last ``\\boxed{}`` holding a number (its last number);
    2. ``answer_key``: the first number within 40 characters after the last "answer is",
       "answer:" or "answer =";
    3. ``bold``: the last number of the last bold span holding one (step headings skipped);
    4. / 5. one sentence (sentences end at whitespace after ``.!?``, never after a factorial
       such as ``7!``, and at line breaks; numbers that are labels, ``Step 2`` or list markers,
       are ignored). A sentence states a *new* number when it holds one that is not in the
       context (without ``context`` every number is new). Among the sentences that state a new
       number: the last one (``last``), except that a later sentence that only checks a result
       (it opens with check, verify, to confirm, plugging in, ...) never overrides the last
       sentence with a concluding word (so, therefore, thus, in total, ...), which is then taken
       (``conclusion``). So a setup sentence ("So the train travels 3 hours at 60 km/h.") is not
       an answer, and neither is a conclusion followed by the final computation ("Therefore 3
       people eat 6 slices. Slices left: 8 - 6 = 2."); "So 15% of 80 is 12. Check: 12 x 80 =
       960." still reads 12. When no sentence states a new number, the last concluding
       sentence, else the last sentence. In the chosen sentence the candidate ``values`` are,
       with ``context``, its new numbers (all of them if none is new), otherwise its last number;
       when it holds an equation the number after its last ``=`` is the stated result: the only
       candidate of a ``last`` sentence, an added candidate of a ``conclusion`` (whose main
       clause may come first: "the total is $43, which includes 180 x 0.10 = 18");
    6. ``words``: number words, when the reply has no digits.

    ``value`` is the stated result (after ``=``) when there is one, else the last candidate; a
    reply counts as correct if any candidate of the decisive location matches (one location, so
    listing many numbers elsewhere does not help).
    """
    t = _normalise(text)

    def one(v: Fraction, method: str, pct: bool) -> dict[str, Any]:
        return {"value": v, "values": [v], "method": method, "percent": pct}

    for box in reversed(_boxed_contents(t)):
        nums = find_numbers(box)
        if nums:
            return one(nums[-1][0], "boxed", nums[-1][3])
    for key in reversed(list(_ANSWER_KEY.finditer(t))):
        nums = find_numbers(t[key.end():key.end() + 40])
        if nums:
            return one(nums[0][0], "answer_key", nums[0][3])
    for bold in reversed(_BOLD.findall(t)):
        if _HEADING_WORDS.match(bold) or bold.rstrip().endswith(":"):
            continue  # "**Step 2**", "**Multiply the result by 10:**" are labels, not answers
        nums = find_numbers(bold)
        if nums:
            return one(nums[-1][0], "bold", nums[-1][3])
    sentences = []
    for s in _SENTENCE_SPLIT.split(t):
        nums = [n for n in find_numbers(s) if not _is_label(s, n[1], n[2])]
        if nums:
            sentences.append((s, nums))
    if sentences:
        new = [i for i, (_, nums) in enumerate(sentences)
               if context is None or any(n[0] not in context for n in nums)]
        if new:
            cues = [i for i in new if _CONCLUSION.search(sentences[i][0])]
            if not cues:
                idx, method = new[-1], "last"
            else:
                later = [i for i in new if i > cues[-1] and not _CHECK.match(sentences[i][0])]
                idx, method = (later[-1], "last") if later else (cues[-1], "conclusion")
        else:
            cues = [i for i, (s, _) in enumerate(sentences) if _CONCLUSION.search(s)]
            idx, method = (cues[-1], "conclusion") if cues else (len(sentences) - 1, "last")
        s, nums = sentences[idx]
        eq = s.rfind("=")
        stated = next((n for n in nums if n[1] > eq), None) if eq >= 0 else None
        if stated is not None and method == "last":
            return one(stated[0], method, stated[3])
        cands = nums[-1:] if context is None else ([n for n in nums if n[0] not in context] or nums)
        if stated is not None and all(n[0] != stated[0] for n in cands):
            cands = cands + [stated]
        top = stated if stated is not None else cands[-1]
        return {"value": top[0], "values": [n[0] for n in cands], "method": method, "percent": top[3]}
    v = parse_number_words(t)
    if v is not None:
        return one(v, "words", False)
    return {"value": None, "values": [], "method": "none", "percent": False}


def _decimals(ref_text: str) -> int:
    m = re.search(r"\.(\d+)", ref_text.replace(",", ""))
    return len(m.group(1)) if m else 0


def numbers_match(value: Fraction, reference: str, tol: float | None = None) -> bool:
    """``value`` equals the reference string exactly, within ``tol``, or after rounding (half
    up) to the reference's number of decimals when the reference has decimals."""
    ref = parse_number(reference)
    if ref is None:
        raise ValueError(f"reference {reference!r} is not a number")
    if value == ref:
        return True
    if tol is not None and abs(value - ref) <= Fraction(str(tol)):
        return True
    d = _decimals(reference)
    if d > 0:
        scale = Fraction(10) ** d
        rounded = Fraction(math.floor(abs(value) * scale + Fraction(1, 2))) / scale
        if (rounded if value >= 0 else -rounded) == ref:
            return True
    return False


def math_correct(reply: str, reference: str, tol: float | None = None,
                 context: Sequence[str] | None = None) -> dict[str, Any]:
    """``{"correct": bool, "extracted": str | None, "candidates": [str], "method": str,
    "any_match": bool}``. ``context`` is the text the reply may restate without answering (the
    user turns so far and the model's own earlier answers, see :func:`extract_answer`).
    ``any_match`` (the reference appears anywhere as a number) is a lenient upper bound kept
    for diagnosing the extraction, never used as the score."""
    got = extract_answer(reply, context_values(context) if context is not None else None)
    correct = any(numbers_match(v, reference, tol) for v in got["values"])
    any_match = correct or any(numbers_match(v, reference, tol) for v, *_ in find_numbers(_normalise(reply)))
    return {"correct": bool(correct), "any_match": bool(any_match), "method": got["method"],
            "extracted": None if got["value"] is None else _fmt(got["value"]),
            "candidates": [_fmt(v) for v in got["values"]]}


def _fmt(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.10g}"


# ------------------------------------------------------------------------ counting backend
def _alnum(ch: str) -> bool:
    return ord(ch) >= 0x80 or ch.isascii() and ch.isalnum()


def local_counts(text: str) -> dict[str, int]:
    """Words, sentences, lines and list items of one complete turn, following the definitions
    of ``docs/SPEC.md`` section 2.1 (used only when the project's exact layer is missing):

    - word: an alphanumeric character (non-ASCII counts as alphanumeric) whose predecessor is
      neither alphanumeric nor an apostrophe that follows an alphanumeric character;
    - sentence: a run of ``.!?`` after a non-space character, followed (after any ``*``/``_``)
      by whitespace, a closing quote or bracket, or the end of the text; not inside a code
      fence; the ``.`` of a numbered-list marker does not count;
    - lines: number of ``\\n``; list item: a line starting, after spaces or tabs, with ``-``,
      ``*``, ``+``, ``•`` or digits and ``.``/``)``, then a space or tab; not inside a fence;
    - a fence flips at the third backtick of a backtick run that begins a line (after optional
      spaces or tabs); the rest of that line is already on the new side.
    """
    words = sentences = items = 0
    in_fence = False
    prev = " "
    prev_apos_after_alnum = False
    lines = text.split("\n")
    for li, line in enumerate(lines):
        stripped = line.lstrip(" \t")
        # the fence flips at the third backtick of a line-initial run (index flip_at)
        flip_at = len(line) - len(stripped) + 2 if stripped.startswith("```") else None
        marker = re.match(r"[ \t]*(?:[-*+\u2022]|\d+[.)])[ \t]", line)
        if not in_fence and flip_at is None and marker:
            items += 1
        list_dot = None
        num = re.match(r"[ \t]*\d+[.)]", line)
        if num:
            list_dot = num.end() - 1
        i, n = 0, len(line)
        while i < n:
            ch = line[i]
            fenced = in_fence != (flip_at is not None and i > flip_at)
            if _alnum(ch):
                if not _alnum(prev) and not prev_apos_after_alnum:
                    words += 1
                prev_apos_after_alnum = False
            else:
                prev_apos_after_alnum = ch == "'" and _alnum(prev)
            if ch in ".!?" and not fenced and i != list_dot:
                j = i
                while j < n and line[j] in ".!?":
                    j += 1
                before = line[i - 1] if i > 0 else ("\n" if li > 0 else " ")
                k = j
                while k < n and line[k] in "*_":
                    k += 1
                after = line[k] if k < n else ("\n" if li < len(lines) - 1 else "")
                if not before.isspace() and (after == "" or after.isspace() or after in "\"')]}"):
                    sentences += 1
                for c in line[i:j]:
                    prev = c
                prev_apos_after_alnum = False
                i = j
                continue
            prev = ch
            i += 1
        if flip_at is not None:
            in_fence = not in_fence
        prev = "\n"
        prev_apos_after_alnum = False
    return {"words": words, "sentences": sentences, "lines": text.count("\n"), "items": items}


_BACKEND_CACHE: dict[str, Any] = {}


def count_backend() -> str:
    """``"constraints.measure"``, ``"transducer.measure_counts"`` or ``"local"``."""
    if "name" not in _BACKEND_CACHE:
        name, fn = "local", None
        try:
            from bes.exact import constraints as project_constraints  # type: ignore[attr-defined]
            probe = project_constraints.measure("One two. Three four.\n- five\n")
            if all(k in probe for k in ("words", "sentences")) and any(
                    k in probe for k in ("items", "list_items")):
                name, fn = "constraints.measure", project_constraints.measure
        except Exception:
            pass
        if fn is None:
            try:
                from bes.exact import transducer
                probe = transducer.measure_counts("One two. Three four.\n- five\n")
                if {"words", "sentences", "lines", "items"} <= set(probe):
                    name, fn = "transducer.measure_counts", transducer.measure_counts
            except Exception:
                pass
        _BACKEND_CACHE["name"], _BACKEND_CACHE["fn"] = name, fn
    return _BACKEND_CACHE["name"]


def counts(text: str) -> dict[str, int]:
    """``{"words", "sentences", "lines", "items"}`` from the active counting backend."""
    count_backend()
    fn = _BACKEND_CACHE["fn"]
    if fn is None:
        return local_counts(text)
    raw = fn(text)
    return {"words": int(raw["words"]), "sentences": int(raw["sentences"]),
            "lines": int(raw.get("lines", text.count("\n"))),
            "items": int(raw.get("items", raw.get("list_items", 0)))}


# ---------------------------------------------------------------------------- constraints
_RELATIONS = ("exactly", "at_most", "at_least", "between")
_COUNT_TYPES = ("words", "sentences", "paragraphs", "lines", "bullets", "numbered", "items",
                "sections", "separated_parts")
CONSTRAINT_TYPES: dict[str, str] = {
    "words": "number of words (relation, n[, n2])",
    "sentences": "number of sentences",
    "paragraphs": "number of paragraphs (blocks separated by blank lines)",
    "lines": "number of non-empty lines",
    "bullets": "number of bullet lines (-, *, +, \u2022)",
    "numbered": "number of numbered-list lines (1. or 1)); without any, number of 'Step N' lines",
    "items": "number of list items of any kind",
    "sections": "number of lines starting with '<marker> <number>'",
    "separated_parts": "number of non-empty parts between separators '<marker>'",
    "include": "every word of 'words' appears (case-insensitive, word start)",
    "exclude": "no word of 'words' appears (case-insensitive, whole word)",
    "frequency": "'word' occurs (whole word) with relation and n",
    "case": "'value' lower or upper: no letters of the other case",
    "json": "the reply (optionally inside one code fence) parses as JSON",
    "json_keys": "the reply is a JSON object containing all 'keys'",
    "starts_with": "the reply starts with one of 'options' (case-insensitive)",
    "ends_with": "the reply ends with 'text' (case-insensitive, trailing markup ignored)",
    "no_commas": "no comma anywhere",
    "quoted": "the whole reply is wrapped in double quotes",
    "title": "contains a title in double angular brackets <<like this>>",
    "postscript": "contains a line starting with 'marker' (default P.S.)",
    "one_of": "the reply is exactly one of 'options' (case and punctuation ignored)",
    "placeholders": "at least n placeholders in square brackets [like this]",
    "ends_with_question": "the reply ends with a question mark",
    "code_block": "contains a closed ``` code block",
    "highlights": "at least n *highlighted* spans",
}


def validate_constraint(c: Any, n_turns: int = 1) -> list[str]:
    """Problems with a constraint description (empty list when well formed)."""
    if not isinstance(c, Mapping):
        return ["constraint must be an object"]
    kind = c.get("type")
    if kind not in CONSTRAINT_TYPES:
        return [f"unknown constraint type {kind!r}"]
    errs = []
    turn = c.get("turn", n_turns - 1)
    if not isinstance(turn, int) or not 0 <= turn < n_turns:
        errs.append(f"{kind}: turn {turn!r} out of range")
    if kind in _COUNT_TYPES or kind == "frequency":
        if c.get("relation") not in _RELATIONS:
            errs.append(f"{kind}: relation must be one of {_RELATIONS}")
        if not isinstance(c.get("n"), int) or c["n"] < 0:
            errs.append(f"{kind}: n must be a non-negative integer")
        if c.get("relation") == "between" and (not isinstance(c.get("n2"), int) or c["n2"] < c.get("n", 0)):
            errs.append(f"{kind}: between needs n2 >= n")
    if kind in ("sections", "separated_parts") and not isinstance(c.get("marker"), str):
        errs.append(f"{kind}: needs a marker string")
    if kind in ("include", "exclude") and (not isinstance(c.get("words"), list) or not c["words"]
                                           or not all(isinstance(w, str) and w for w in c["words"])):
        errs.append(f"{kind}: needs a non-empty list of words")
    if kind == "frequency" and not (isinstance(c.get("word"), str) and c["word"]):
        errs.append("frequency: needs a word")
    if kind == "case" and c.get("value") not in ("lower", "upper"):
        errs.append("case: value must be lower or upper")
    if kind == "json_keys" and (not isinstance(c.get("keys"), list) or not c["keys"]):
        errs.append("json_keys: needs a list of keys")
    if kind in ("starts_with", "one_of") and (not isinstance(c.get("options"), list) or not c["options"]):
        errs.append(f"{kind}: needs a list of options")
    if kind == "ends_with" and not (isinstance(c.get("text"), str) and c["text"]):
        errs.append("ends_with: needs text")
    if kind in ("placeholders", "highlights") and (not isinstance(c.get("n"), int) or c["n"] < 1):
        errs.append(f"{kind}: n must be a positive integer")
    return errs


def _relation_ok(value: int, c: Mapping[str, Any]) -> bool:
    n, rel = c["n"], c["relation"]
    if rel == "exactly":
        return value == n
    if rel == "at_most":
        return value <= n
    if rel == "at_least":
        return value >= n
    return n <= value <= c["n2"]


def _fence_lines(text: str) -> list[int]:
    return [i for i, line in enumerate(text.split("\n")) if line.lstrip(" \t").startswith("```")]


def _outside_fences(text: str) -> list[str]:
    out, inside = [], False
    for line in text.split("\n"):
        if line.lstrip(" \t").startswith("```"):
            inside = not inside
            continue
        if not inside:
            out.append(line)
    return out


def paragraphs(text: str) -> int:
    return sum(1 for block in re.split(r"\n[ \t]*\n", text.strip()) if block.strip())


def _unfence(text: str) -> str:
    t = text.strip()
    m = re.fullmatch(r"```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _word_pattern(word: str, whole: bool) -> re.Pattern:
    tail = r"(?![A-Za-z0-9])" if whole else ""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + tail, re.IGNORECASE)


_LEAD_MARKUP = " \t\n*#>_\"'\u201c\u2018"
_TRAIL_MARKUP = " \t\n*_\"'\u201d\u2019"


def measure_for(kind: str, text: str, c: Mapping[str, Any]) -> Any:
    """The measured quantity a constraint of type ``kind`` looks at."""
    if kind in ("words", "sentences", "items"):
        return counts(text)[kind]
    if kind == "paragraphs":
        return paragraphs(text)
    if kind == "lines":
        return sum(1 for line in text.split("\n") if line.strip())
    if kind == "bullets":
        return sum(1 for line in _outside_fences(text) if re.match(r"[ \t]*[-*+\u2022][ \t]", line))
    if kind == "numbered":
        lines = _outside_fences(text)
        listed = sum(1 for line in lines if re.match(r"[ \t]*\d+[.)][ \t]", line))
        if listed:
            return listed
        # no list markers: "Step 1: ..." lines (optionally as headings or bold) are numbered steps
        return sum(1 for line in lines if re.match(r"[ \t]*(?:#+[ \t]*)?(?:\*\*)?step[ \t]+\d+\b", line, re.IGNORECASE))
    if kind == "sections":
        pat = re.compile(r"[ \t]*(?:#+[ \t]*)?(?:\*\*)?" + re.escape(c["marker"]) + r"[ \t]+\d+", re.IGNORECASE)
        return sum(1 for line in text.split("\n") if pat.match(line))
    if kind == "separated_parts":
        return sum(1 for part in text.split(c["marker"]) if part.strip())
    if kind == "frequency":
        return len(_word_pattern(c["word"], whole=True).findall(text))
    raise KeyError(kind)


def check_constraint(text: str, c: Mapping[str, Any]) -> dict[str, Any]:
    """``{"type", "ok": bool, "measured": ...}`` for one constraint on one reply."""
    kind = c["type"]
    measured: Any = None
    if kind in _COUNT_TYPES or kind == "frequency":
        measured = measure_for(kind, text, c)
        ok = _relation_ok(measured, c)
    elif kind == "include":
        missing = [w for w in c["words"] if not _word_pattern(w, whole=False).search(text)]
        measured, ok = missing, not missing
    elif kind == "exclude":
        found = [w for w in c["words"] if _word_pattern(w, whole=True).search(text)]
        measured, ok = found, not found
    elif kind == "case":
        letters = [ch for ch in text if ch.isalpha()]
        if c["value"] == "lower":
            ok = bool(letters) and not any(ch.isupper() for ch in letters)
        else:
            ok = bool(letters) and not any(ch.islower() for ch in letters)
    elif kind in ("json", "json_keys"):
        try:
            parsed = json.loads(_unfence(text))
            ok = True
        except (ValueError, TypeError):
            parsed, ok = None, False
        if kind == "json_keys":
            missing = [k for k in c["keys"] if not isinstance(parsed, dict) or k not in parsed]
            measured, ok = missing, ok and not missing
    elif kind == "starts_with":
        head = text.lstrip(_LEAD_MARKUP).lower()
        ok = any(head.startswith(o.lower()) for o in c["options"])
    elif kind == "ends_with":
        tail = text.rstrip(_TRAIL_MARKUP).lower()
        want = c["text"].lower().rstrip(_TRAIL_MARKUP)
        ok = tail.endswith(want)
    elif kind == "no_commas":
        measured = text.count(",")
        ok = measured == 0
    elif kind == "quoted":
        t = text.strip()
        ok = len(t) >= 2 and t[0] in "\"\u201c" and t[-1] in "\"\u201d"
    elif kind == "title":
        ok = re.search(r"<<[^<>\n]+>>", text) is not None
    elif kind == "postscript":
        marker = c.get("marker", "P.S.")
        flexible = re.escape(marker).replace(r"\.", r"\.?\s?")
        ok = re.search(r"(?im)^[ \t*_]*" + flexible, text) is not None
    elif kind == "one_of":
        norm = re.sub(r"[^\w\s]", "", text).strip().lower()
        ok = norm in {re.sub(r"[^\w\s]", "", o).strip().lower() for o in c["options"]}
    elif kind == "placeholders":
        measured = len(re.findall(r"\[[^\[\]\n]+\]", text))
        ok = measured >= c["n"]
    elif kind == "ends_with_question":
        ok = text.rstrip(_TRAIL_MARKUP).endswith("?")
    elif kind == "code_block":
        fences = _fence_lines(text)
        ok = len(fences) >= 2
    elif kind == "highlights":
        measured = len(re.findall(r"(?<!\*)\*\*?[^*\n]+\*\*?(?!\*)", text))
        ok = measured >= c["n"]
    else:
        raise KeyError(f"unknown constraint type {kind!r}")
    return {"type": kind, "ok": bool(ok), "measured": measured}


def project_check(user_text: str, reply: str) -> dict[str, Any] | None:
    """The project's own compiler and verifier (``bes.exact.constraints.compile`` /
    ``verify``) on one exchange, or ``None`` when that module is not importable.
    Returns ``{"violations": [...], "n_constraints": int | None}``."""
    try:
        from bes.exact import constraints as project_constraints  # type: ignore[attr-defined]
        spec = project_constraints.compile(user_text)
        violations = list(project_constraints.verify(spec, reply))
    except Exception:
        return None
    n = None
    for attr in ("constraints", "items"):
        if hasattr(spec, attr):
            try:
                n = len(getattr(spec, attr))
            except TypeError:
                n = None
            break
    return {"violations": [str(v) for v in violations], "n_constraints": n}


# ------------------------------------------------------------------------ degenerate output
_ROLE_LEAK = re.compile(
    r"<\|(?:im_start|im_end|eot_id|start_header_id|end_header_id|user|assistant|system|end|endoftext)\|>"
    r"|(?m:^[ \t]*(?:#+[ \t]*)?(?:user|human|assistant)[ \t]*:)", re.IGNORECASE)
_TOKENS = re.compile(r"\w+|[^\w\s]")


def loop_tail(tokens: Sequence[str], max_period: int = 120, min_span: int = 12) -> tuple[int, int]:
    """``(period, span)`` of the longest periodic tail: the last ``span`` tokens repeat with
    period ``period`` (``tokens[i] == tokens[i + period]``) and ``span >= 3 * period`` and
    ``span >= min_span``; ``(0, 0)`` when there is none."""
    n = len(tokens)
    best = (0, 0)
    for p in range(1, min(max_period, n // 3) + 1):
        span = p
        j = n - 1 - p
        while j >= 0 and tokens[j] == tokens[j + p]:
            span += 1
            j -= 1
        if span >= 3 * p and span >= min_span and span > best[1]:
            best = (p, span)
    return best


def degenerate_stats(text: str, finish_reason: str | None = None) -> dict[str, Any]:
    """Diagnostics of one reply.

    - ``words``: word count (counting backend); ``empty``: no letters or digits at all;
    - ``truncated``: generation stopped at the token limit (``None`` when unknown);
    - ``loop``: the reply ends in a periodic span covering at least three periods and at least
      12 tokens (``loop_period``, ``loop_span`` in tokens); ``repeated_lines``: some line of at
      least five words outside code fences occurs three or more times;
    - ``rep4``: share of word 4-grams that repeat an earlier 4-gram of the reply;
    - ``unclosed_fence``: an odd number of ``` fence lines; ``role_leak``: role or template
      markers in the text (the model continued the conversation by itself);
    - ``degenerate``: any of empty, truncated, loop, repeated_lines, unclosed_fence, role_leak.
    """
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    rep4 = 1.0 - len(set(grams)) / len(grams) if grams else 0.0
    period, span = loop_tail(_TOKENS.findall(text.lower()))
    line_counts = Counter(line.strip().lower() for line in _outside_fences(text)
                          if len(re.findall(r"\w+", line)) >= 5)  # code may repeat lines legitimately
    repeated_lines = any(v >= 3 for v in line_counts.values())
    empty = not re.search(r"[A-Za-z0-9]", text) and not re.search(r"[^\x00-\x7f]", text.strip())
    truncated = None if finish_reason is None else finish_reason == "length"
    unclosed = len(_fence_lines(text)) % 2 == 1
    leak = _ROLE_LEAK.search(text) is not None
    flags = [empty, bool(truncated), period > 0, repeated_lines, unclosed, leak]
    return {"words": counts(text)["words"] if text else 0, "empty": empty, "truncated": truncated,
            "loop": period > 0, "loop_period": period, "loop_span": span,
            "repeated_lines": repeated_lines, "rep4": round(rep4, 4), "unclosed_fence": unclosed,
            "role_leak": leak, "degenerate": any(flags)}


# ------------------------------------------------------------------------------- per prompt
def score_conversation(prompt: Any, replies: Sequence[str],
                       finish_reasons: Sequence[str | None] | None = None,
                       with_project: bool = True) -> dict[str, Any]:
    """Exact scores of one model's replies to one prompt (``prompt`` is a
    ``bes.eval.proxy.prompts.Prompt``; ``replies[t]`` answers ``prompt.turns[t]``)."""
    finish_reasons = list(finish_reasons or [None] * len(replies))
    out: dict[str, Any] = {"prompt_id": prompt.id, "category": prompt.category,
                           "multi_turn": prompt.multi_turn, "turns": []}
    earlier_answers: list[str] = []
    for t, reply in enumerate(replies):
        row: dict[str, Any] = {"degenerate": degenerate_stats(reply, finish_reasons[t])}
        ref = prompt.answer_for_turn(t)
        context = list(prompt.turns[:t + 1]) + earlier_answers
        if ref is not None:
            row["math"] = math_correct(reply, ref, prompt.tol, context=context)
        if prompt.category == "math":
            got = extract_answer(reply, context_values(context))
            if got["value"] is not None:
                earlier_answers.append(_fmt(got["value"]))
        cons = prompt.constraints_for_turn(t)
        if cons:
            row["constraints"] = [check_constraint(reply, c) for c in cons]
            row["constraints_ok"] = all(r["ok"] for r in row["constraints"])
        if with_project and cons:
            row["project"] = project_check(prompt.turns[t], reply)
        out["turns"].append(row)
    last = out["turns"][-1] if out["turns"] else {}
    out["math_correct"] = last.get("math", {}).get("correct") if "math" in last else None
    with_cons = [r for r in out["turns"] if "constraints_ok" in r]
    out["constraints_ok"] = all(r["constraints_ok"] for r in with_cons) if with_cons else None
    out["n_constraints"] = sum(len(r.get("constraints", [])) for r in out["turns"])
    out["n_constraints_ok"] = sum(sum(c["ok"] for c in r.get("constraints", [])) for r in out["turns"])
    return out


def _share(values: Sequence[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate :func:`score_conversation` results (over all turns for the degenerate rates)."""
    math_items = [s for s in scores if s.get("math_correct") is not None]
    cons_items = [s for s in scores if s.get("constraints_ok") is not None]
    turns = [t for s in scores for t in s["turns"]]
    degen = [t["degenerate"] for t in turns]
    by_type: dict[str, list[bool]] = {}
    for t in turns:
        for c in t.get("constraints", []):
            by_type.setdefault(c["type"], []).append(c["ok"])
    words = sorted(d["words"] for d in degen)
    project = [t["project"] for t in turns if t.get("project") is not None]
    answers = [t["math"]["correct"] for t in turns if "math" in t]
    out = {
        "n_prompts": len(scores), "n_turns": len(turns), "count_backend": count_backend(),
        "math": {"n": len(math_items),
                 "correct": _share([bool(s["math_correct"]) for s in math_items]),
                 "n_answers": len(answers), "answers_correct": _share(answers),
                 "any_match": _share([bool(s["turns"][-1]["math"]["any_match"]) for s in math_items]),
                 "methods": dict(Counter(s["turns"][-1]["math"]["method"] for s in math_items))},
        "constraints": {"n_prompts": len(cons_items),
                        "prompts_all_ok": _share([bool(s["constraints_ok"]) for s in cons_items]),
                        "n_constraints": sum(s["n_constraints"] for s in cons_items),
                        "constraints_ok": _share([c for v in by_type.values() for c in v]),
                        "by_type": {k: {"n": len(v), "ok": _share(v)} for k, v in sorted(by_type.items())}},
        "degenerate": {k: _share([bool(d[k]) for d in degen if d[k] is not None])
                       for k in ("degenerate", "empty", "truncated", "loop", "repeated_lines",
                                 "unclosed_fence", "role_leak")},
        "rep4_mean": round(sum(d["rep4"] for d in degen) / len(degen), 4) if degen else None,
        "words": {"mean": round(sum(words) / len(words), 1) if words else None,
                  "median": words[len(words) // 2] if words else None,
                  "p90": words[min(len(words) - 1, int(0.9 * len(words)))] if words else None},
    }
    if project:
        out["project_checker"] = {"n": len(project),
                                  "no_violation": _share([not p["violations"] for p in project])}
    by_cat: dict[str, list] = {}
    for s in scores:
        by_cat.setdefault(s["category"], []).append(s)
    out["by_category"] = {cat: {
        "n": len(items),
        "math_correct": _share([bool(s["math_correct"]) for s in items if s.get("math_correct") is not None]),
        "constraints_ok": _share([bool(s["constraints_ok"]) for s in items if s.get("constraints_ok") is not None]),
        "degenerate": _share([bool(t["degenerate"]["degenerate"]) for s in items for t in s["turns"]]),
        "words_mean": round(sum(t["degenerate"]["words"] for s in items for t in s["turns"])
                            / max(1, sum(len(s["turns"]) for s in items)), 1),
    } for cat, items in sorted(by_cat.items())}
    return out
