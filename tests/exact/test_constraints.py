"""Tests of bes.exact.constraints: measured properties, the compiler, verifiers, rendering and
the countdown targets.

Every measured property is checked against a reference written here with different means
(line-by-line string operations, a character scan, ``str`` methods); the transducer-based counts
are checked against ``bes.exact.transducer.measure_counts``, and the line-level fence and list
rules against the transducer's own state. Targets are checked against a brute force that walks
back from every position, and the countdown registers against the counts at every token.

The further instruction families (highlights, placeholders, titles, capital words, paragraph
texts and first words, start / end phrases) are checked against character scans written here;
the compiler against hand-labelled prompts, including every constrained dev-half prompt of the
proxy set and false positives found on natural chat prompts; ``render`` by a property-tested
round trip over random specs of every family; the token-local metadata against the register
counters, ``measure`` and ``verify``.
"""
from __future__ import annotations

import random
import re
import time

import numpy as np
import pytest

from bes import config
from bes.config import ASSISTANT, BOS, END, LIBRARY, SYSTEM, USER
from bes.exact import constraints as C
from bes.exact import transducer as T
from bes.exact.constraints import ConstraintSpec, Interval

LINE_WS = " \t\r\x0b\x0c"


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


def ex(n):
    return Interval(n, n)


def le(n):
    return Interval(0, n)


def ge(n):
    return Interval(n, None)


def bt(a, b):
    return Interval(a, b)


def S(**kw) -> ConstraintSpec:
    return ConstraintSpec(**kw)


# ===================================================================== reference measures
def ref_fence_flags(lines: list[str]) -> list[bool]:
    """Inside a fence at the start of each line; a line whose first non-blank characters are
    three backticks toggles."""
    flags, inside = [], False
    for line in lines:
        flags.append(inside)
        if line.lstrip(" \t").startswith("```"):
            inside = not inside
    return flags


def ref_is_break(line: str) -> bool:
    body = line.rstrip("\r")
    lead = len(body) - len(body.lstrip(" "))
    rest = body.lstrip(" ")
    chars = rest.replace(" ", "").replace("\t", "")
    return lead <= 3 and len(chars) >= 3 and len(set(chars)) == 1 and rest[:1] in ("*", "-", "_")


def ref_dividers(body: str) -> list[str] | None:
    """Pieces of a line split at "***" dividers (runs of >= 3 stars with whitespace or the line
    edge on both sides); ``None`` for a line made only of stars and blanks."""
    chars = body.replace(" ", "").replace("\t", "")
    if len(chars) >= 3 and set(chars) == {"*"}:
        return None
    pieces, start, i = [], 0, 0
    while i < len(body):
        if body[i] == "*":
            j = i
            while j < len(body) and body[j] == "*":
                j += 1
            left_ok = i == 0 or body[i - 1] in " \t"
            right_ok = j == len(body) or body[j] in " \t"
            if j - i >= 3 and left_ok and right_ok:
                pieces.append(body[start:i])
                start = j
            i = j
        else:
            i += 1
    pieces.append(body[start:])
    return pieces


def ref_paragraphs(text: str) -> int:
    lines = text.split("\n")
    flags = ref_fence_flags(lines)
    # divider mode
    tokens, any_div = [], False
    for line, fenced in zip(lines, flags):
        body = line.rstrip("\r")
        pieces = [body] if fenced else ref_dividers(body)
        if pieces is None:
            tokens.append(None)
            any_div = True
            continue
        for k, piece in enumerate(pieces):
            if k:
                tokens.append(None)
                any_div = True
            tokens.append(piece)
    if any_div:
        count, has = 0, False
        for t in tokens + [None]:
            if t is None:
                count += has
                has = False
            elif t.strip(LINE_WS):
                has = True
        return count
    count, prev_sep = 0, True
    for line, fenced in zip(lines, flags):
        sep = not fenced and (not line.strip(LINE_WS) or ref_is_break(line))
        if not sep and prev_sep:
            count += 1
        prev_sep = sep
    return count


def ref_item_kind(line: str) -> str | None:
    rest = line.lstrip(" \t")
    if len(rest) >= 2 and rest[0] in "-*+•" and rest[1] in " \t":
        return "bullet"
    k = 0
    while k < len(rest) and rest[k].isascii() and rest[k].isdigit():
        k += 1
    if k and len(rest) > k + 1 and rest[k] in ".)" and rest[k + 1] in " \t":
        return "numbered"
    return None


def ref_heading(line: str) -> bool:
    lead = len(line) - len(line.lstrip(" "))
    rest = line.lstrip(" ")
    h = len(rest) - len(rest.lstrip("#"))
    after = rest[h:]
    return lead <= 3 and 1 <= h <= 6 and after[:1] in (" ", "\t") and bool(after.strip())


def ref_ends_q(text: str) -> bool:
    i = len(text)
    while i and text[i - 1] in " \t\n\r\x0b\x0c\"')]}*_”’»":
        i -= 1
    return i > 0 and text[i - 1] == "?"


def ref_starts_cap(text: str) -> bool:
    for ch in text:
        if ch in " \t\n\r\x0b\x0c#*_>\"'([“‘«":
            continue
        return ch.isupper()
    return False


def ref_keyword_count(text: str, kw: str, whole: bool) -> int:
    words = kw.split()
    low = text.lower()
    n = 0
    for i in range(len(text)):
        if i and text[i - 1].isalnum():
            continue
        pos, ok = i, True
        for k, w in enumerate(words):
            if k:
                j = pos
                while j < len(text) and text[j].isspace():
                    j += 1
                if j == pos:
                    ok = False
                    break
                pos = j
            if low[pos:pos + len(w)] != w.lower():
                ok = False
                break
            pos += len(w)
        if ok and (not whole or pos == len(text) or not text[pos].isalnum()):
            n += 1
    return n


PIECES = ["```", "```python", "`", "- ", "* ", "+ ", "1. ", "12) ", "• ", "# ", "## T", "Hello",
          " world", "word", ".", "!", "?", "...", " ", "  ", "\n", "\n\n", "\t", "\"", "'", "(", ")",
          "[", "]", "{", "}", "don't", "3.14", "**Done.**", " *** ", "***", "---", "* * *", "é",
          "x", ",", "\r", "   ", "ABC", "def"]


def random_text(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(0, n)))


@pytest.mark.parametrize("seed", range(3))
def test_measure_matches_references(seed):
    rng = random.Random(seed)
    for _ in range(1500):
        t = random_text(rng)
        m = C.measure(t)
        lines = t.split("\n")
        flags = ref_fence_flags(lines)
        mc = T.measure_counts(t)
        assert C.transducer_counts(t) == mc
        for key in ("words", "sentences", "items", "depth"):
            assert m[key] == mc[key], (key, t)
        assert m["newlines"] == mc["lines"] == t.count("\n")
        assert (m["open_fence"], m["open_inline"], m["open_quote"]) == (bool(mc["fence"]), bool(mc["inline"]), bool(mc["quote"]))
        assert m["lines"] == sum(1 for line in lines if line.strip(LINE_WS))
        assert m["paragraphs"] == ref_paragraphs(t), t
        kinds = [None if f else ref_item_kind(line) for line, f in zip(lines, flags)]
        assert m["bullet_items"] == kinds.count("bullet") and m["numbered_items"] == kinds.count("numbered")
        assert m["bullet_items"] + m["numbered_items"] == m["items"], t      # same rule as the machine
        fences = sum(1 for line in lines if line.lstrip(" \t").startswith("```"))
        assert m["code_fences"] == fences and m["code_blocks"] == fences // 2
        assert m["open_fence"] == (fences % 2 == 1), t                       # same rule as the machine
        assert m["has_code_fence"] == (fences > 0)
        assert m["headings"] == sum(1 for line, f in zip(lines, flags) if not f and ref_heading(line))
        assert m["ends_with_question"] == ref_ends_q(t)
        assert m["starts_with_capital"] == ref_starts_cap(t)
        assert m["all_lower"] == (any(c.isalpha() and c.lower() != c.upper() for c in t)
                                  and not any(c.isupper() for c in t)) == t.islower()
        assert m["all_upper"] == t.isupper()
        assert m["chars"] == len(t) and m["commas"] == t.count(",") and m["question_marks"] == t.count("?")


def test_no_byte_counts_both_a_word_and_a_sentence():
    """transducer_counts applies the sentence event before the word event of a byte, as
    transducer.run does; the order is irrelevant because no byte carries both."""
    m = T.compile_machine()
    assert not ((m.events[..., 0] > 0) & (m.events[..., 1] > 0)).any()


def test_transducer_counts_on_real_text():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    for path in sorted((root / "docs").glob("*.md"))[:4]:
        text = path.read_text()
        assert C.transducer_counts(text) == T.measure_counts(text)


@pytest.mark.parametrize("text,key,want", [
    ("One. Two! Three?", "sentences", 3), ("Done", "sentences", 0), ("Dr. Who.", "sentences", 2),
    ("don't stop", "words", 2), ("well-being", "words", 2), ("", "words", 0),
    ("- a\n- b\n* c\n1. d\n2) e", "items", 5), ("- a\n- b\n* c\n1. d\n2) e", "bullet_items", 3),
    ("- a\n- b\n* c\n1. d\n2) e", "numbered_items", 2), ("```\n- a\n```\n- b", "items", 1),
    ("a\n\nb\n\n\nc", "paragraphs", 3), ("a\nb\nc", "paragraphs", 1), ("a\n---\nb", "paragraphs", 2),
    ("```\na\n\nb\n```", "paragraphs", 1), ("<<title>>\n\np1\n\n***\n\np2\n\nP.S. x", "paragraphs", 2),
    ("p1. ***\n\np2. *** p3", "paragraphs", 3), ("p1\n***\np2", "paragraphs", 2), ("", "paragraphs", 0),
    ("   \n\t\n", "paragraphs", 0), ("**bold** text\n\nmore ***x***", "paragraphs", 2),
    ("a\n\n b \n\n", "lines", 2), ("a\n\n b \n\n", "newlines", 4), ("# T\n## U\n#no\n    # code", "headings", 2),
    ("```\n# not\n```", "headings", 0), ("```py\nx\n```\n```\ny", "code_fences", 3),
    ("```py\nx\n```\n```\ny", "code_blocks", 1), ("```py\nx\n```\n```\ny", "open_fence", True),
    ("Why?\n", "ends_with_question", True), ("Is it (really?)*", "ends_with_question", True),
    ("What? No.", "ends_with_question", False), ("**Paris** is", "starts_with_capital", True),
    ("1. Apple", "starts_with_capital", False), ("hello world 42", "all_lower", True),
    ("hello World", "all_lower", False), ("42", "all_lower", False), ("HELLO 42!", "all_upper", True),
    ('{"a": [1, 2]}', "json_valid", True), ('```json\n{"a": 1}\n```', "json_valid", True),
    ("[1, 2]", "json_valid", True), ("42", "json_valid", False), ('{"a": NaN}', "json_valid", False),
    ('Here: {"a": 1}', "json_valid", False), ("(a [b {c", "depth", 3), ("a, b, c", "commas", 2),
])
def test_measure_hand_labelled(text, key, want):
    assert C.measure(text)[key] == want


@pytest.mark.parametrize("seed", range(2))
def test_keyword_counts_match_scan(seed):
    rng = random.Random(seed)
    alphabet = ["cat", "Cat", "cats", "dog", " ", "  ", "\n", ".", ",", "-", "_", "x", "é", "1",
                "carbon", "dioxide", "Carbon", "'s"]
    keywords = ["cat", "dog", "carbon dioxide", "x", "é", "1", "cat 1"]
    for _ in range(400):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        m = C.measure(t, keywords)
        for k in keywords:
            assert m["keywords"][k] == ref_keyword_count(t, k, whole=False), (t, k)
            assert m["keywords_whole"][k] == ref_keyword_count(t, k, whole=True), (t, k)


# ============================================================================ spec types
def test_interval_and_spec_validation():
    assert Interval(3, 3).kind == "exactly" and Interval(0, 5).kind == "at_most"
    assert Interval(2, None).kind == "at_least" and Interval(2, 5).kind == "between"
    assert Interval(2, 5).contains(5) and not Interval(2, 5).contains(6) and Interval(2).contains(10**9)
    assert Interval(1, 5).intersect(Interval(3, None)) == Interval(3, 5)
    assert Interval(1, 2).intersect(Interval(3, None)) is None
    for bad in ((-1, None), (3, 2), (1.5, None)):
        with pytest.raises(ValueError):
            Interval(*bad)
    for kw in ({"words": Interval(0, None)}, {"sentences": Interval(0, 0)}, {"keywords": ("a b c d e",)},
               {"keywords": ("x", "X")}, {"keywords": ("x",), "forbidden": ("x",)}, {"case": "title"},
               {"keywords": ('"q"',)}, {"keywords": ("a,b",)}, {"words": ge(10**6)}):
        with pytest.raises((ValueError, TypeError)):
            ConstraintSpec(**kw)
    assert ConstraintSpec(items=Interval(0, 0)).targets() == (-1, 0, -1)
    spec = S(sentences=ex(3), items=ge(2), words=bt(10, 50), keywords=("a",), json_output=True)
    assert spec.targets() == (3, -1, 50)
    assert len(spec.constraints) == 5 and not spec.empty and C.EMPTY.empty


def test_merge():
    a = S(sentences=ge(3), keywords=("x", "y"), case="lower")
    b = S(sentences=le(5), keywords=("Y", "z"), end_question=True)
    assert C.merge(a, b) == S(sentences=bt(3, 5), keywords=("x", "y", "z"), case="lower", end_question=True)
    assert C.merge(S(sentences=ge(6)), S(sentences=le(5))) is None
    assert C.merge(S(case="lower"), S(case="upper")) is None
    assert C.merge(S(keywords=("x",)), S(forbidden=("x",))) is None
    assert C.merge(C.EMPTY, a) == a


# ============================================================================== compile
CASES = [
    # counts with relations
    ("Explain photosynthesis in exactly three sentences.", S(sentences=ex(3))),
    ("Your response should contain at least 5 sentences.", S(sentences=ge(5))),
    ("Your response should contain less than 150 words.", S(words=le(149))),
    ("Answer with at least 150 words.", S(words=ge(150))),
    ("Answer with no more than 50 words.", S(words=le(50))),
    ("Use fewer than 4 paragraphs.", S(paragraphs=le(3))),
    ("Write more than 2 paragraphs.", S(paragraphs=ge(3))),
    ("Summarize this in under 50 words.", S(words=le(49))),
    ("Keep it under 100 words, please.", S(words=le(99))),
    ("Max 3 sentences.", S(sentences=le(3))),
    ("3 sentences max.", S(sentences=le(3))),
    ("50 words or less.", S(words=le(50))),
    ("Answer in 3 sentences or more.", S(sentences=ge(3))),
    ("Write 100+ words.", S(words=ge(100))),
    ("Respond with a minimum of 8 bullet points.", S(items=ge(8), list_style="bullet")),
    ("Use up to 4 bullet points.", S(items=le(4), list_style="bullet")),
    ("As many as 4 bullet points.", S(items=le(4), list_style="bullet")),
    ("Your answer must contain exactly 3 bullet points. Use the markdown bullet points such as: * This is point 1.",
     S(items=ex(3), list_style="bullet")),
    ("Use precisely 7 lines.", S(lines=ex(7))),
    ("Answer in 1,000 words or fewer.", S(words=le(1000))),
    ("Use exactly twenty-one sentences.", S(sentences=ex(21))),
    # ranges and combinations
    ("Write between 3 and 5 sentences.", S(sentences=bt(3, 5))),
    ("Use 2-3 sentences.", S(sentences=bt(2, 3))),
    ("Answer in 2 to 4 paragraphs.", S(paragraphs=bt(2, 4))),
    ("Describe it in one or two sentences.", S(sentences=bt(1, 2))),
    ("Describe it in a sentence or two.", S(sentences=bt(1, 2))),
    ("Now, in just 2 - 3 sentences, describe what you did.", S(sentences=bt(2, 3))),
    ("You can write up to 200 or 300 words.", S()),
    ("Please write up to 200 or 300 words.", S(words=le(300))),
    ("Your response should contain at least 3 sentences and no more than 5 sentences.", S(sentences=bt(3, 5))),
    ("Answer in at least 3 and at most 5 sentences.", S(sentences=bt(3, 5))),
    ("Your response should contain at least 200 words and at least 3 sentences.", S(words=ge(200), sentences=ge(3))),
    ("Your response should contain at least 500 words, include a postscript, and contain exactly 4 bullet points.",
     S(words=ge(500), items=ex(4), list_style="bullet", postscript="P.S.")),
    ("Use at least 6 sentences. Use at most 2 sentences.", S()),
    ("Write 3 sentences or 2 paragraphs.", S()),
    ("Answer in a single sentence, a short paragraph, or a list.", S()),
    # negation
    ("Do not use more than 3 sentences.", S(sentences=le(3))),
    ("Don't write fewer than 4 bullet points.", S(items=ge(4), list_style="bullet")),
    ("Your response should not exceed 100 words.", S(words=le(100))),
    ("Answer the question without exceeding 80 words.", S(words=le(80))),
    ("It must not be longer than 50 words.", S(words=le(50))),
    ("Don't answer in 3 sentences.", S()),
    ("Explain why the sky is not blue in 3 sentences.", S()),
    ("Don't limit yourself to 3 sentences.", S()),
    # bare counts
    ("Explain photosynthesis in three sentences.", S(sentences=ex(3))),
    ("In 3 sentences, describe the sun.", S(sentences=ex(3))),
    ("Give me a one-sentence answer.", S(sentences=ex(1))),
    ("Write a single paragraph about dogs.", S(paragraphs=ex(1))),
    ("Answer in a single sentence.", S(sentences=ex(1))),
    ("There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***",
     S(paragraphs=ex(2), divider=True)),
    ("Your response must have 3 paragraphs.", S(paragraphs=ex(3))),
    ("Write a poem with 4 lines.", S(lines=ex(4))),
    ("Write a poem of between 12 and 20 lines about a pirate.", S(lines=bt(12, 20))),
    ("Write three sentences to describe a beach.", S(sentences=ex(3))),
    ("Three bullet points, please.", S(items=ex(3), list_style="bullet")),
    ("Write a 500-word essay on climate change.", S(words=le(500))),
    ("Describe the ocean in 50 words.", S(words=le(50))),
    ("Answer in one word.", S(words=ex(1))),
    ("Write an 8-sentence paragraph about editing.", S(sentences=ex(8))),
    ("Format your answer as a 5-item list.", S(items=ex(5))),
    ("Explain it in a sentence.", S(sentences=ex(1))),
    ("Describe it in about 100 words.", S()),
    ("Write roughly 3 paragraphs.", S()),
    ("Summarize in 3 sentences or so.", S()),
    ("answer with [relation] 20 words.", S()),
    # list N
    ("List 5 fruits.", S(items=ex(5))),
    ("Give me three reasons to learn Python.", S(items=ex(3))),
    ("Can you list 5 ideas for a party?", S(items=ex(5))),
    ("Name two other types of renewable energy besides solar.", S(items=ex(2))),
    ("Now list 2 colours of the sea, at most 6 words.", S(items=ex(2), words=le(6))),
    ("What are three reasons the Roman empire fell?", S()),
    ("Name three countries, comma-separated.", S()),
    ("List 5 fruits in one sentence.", S(sentences=ex(1))),
    ("Name three advantages and three disadvantages of buying a car.", S()),
    ("Suggest three alternatives for each sentence.", S()),
    ("Give me 5 minutes to think.", S()),
    ("Give me 1 example.", S()),
    ("Give me 10 words that rhyme with cat.", S()),
    # statements, questions, code, edits, per-part and multi-step: nothing
    ("I read a 500-word article yesterday.", S()),
    ("My essay has 5 paragraphs. Can you shorten it?", S()),
    ("I wrote a 300-word essay arguing that school should start later.", S()),
    ("You edited the original paragraph into 3 sentences.", S()),
    ("I have added one or two sentences and made some changes.", S()),
    ("This line of code does two things in one line.", S()),
    ("Here is a brief overview of it in one sentence.", S()),
    ('"alot" should be spelled as two separate words.', S()),
    ("How do you decide if a sentence should be broken into two sentences?", S()),
    ("Is it possible for the revised sentence to be a single sentence?", S()),
    ("What if I combined the 3rd and 4th sentences into one sentence?", S()),
    ("If the file contains fewer than two lines, return an empty string.", S()),
    ("Write a Python function that takes in a single word as input.", S()),
    ("Write a function to convert kilometers. The function should be defined in a single line.", S()),
    ("Write a Python function to find the intersection of two lines.", S()),
    ("Write 10 lines of code to sort a list.", S()),
    ("Make the paragraph shorter by removing at most three words.", S()),
    ("Explain each of these 4 concepts in less than 50 words.", S()),
    ("Use at most 3 sentences per paragraph.", S()),
    ("Write a single paragraph, then improve it.", S()),
    ("Write a single sentence that needs editing and explain that.", S()),
    ("Can you give me an example of refining this one sentence?", S()),
    ("Describe a 5-paragraph essay structure.", S()),
    ("Use a single-line conditional expression to handle this.", S()),
    ("The victim was found with a single bullet wound.", S()),
    ("A company has a discount for bulk orders of 100 items or more.", S()),
    ("Make the list one item shorter than the input.", S()),
    ("Don't use list comprehension for this task.", S()),
    ("Is there a length limit? Submissions of 1-2 paragraphs tend to work well.", S()),
    ("Rewrite this text: \"The essay must have at least 5 paragraphs.\"", S()),
    ("```\nUse at most 3 sentences.\n```\nWhat does this code do?", S()),
    ("> Answer in two sentences.\nWhat do you think of this quote?", S()),
    # no list
    ("Don't use bullet points.", S(items=ex(0))),
    ("Write in prose without bullet points.", S(items=ex(0))),
    ("No bullet points, please.", S(items=ex(0))),
    ("Do not use lists.", S(items=ex(0))),
    ("Avoid lists of more than 5 items.", S()),
    # keywords
    ("Include the word 'ocean'.", S(keywords=("ocean",))),
    ('Include keywords [book], [library], and [borrow] in the response.', S(keywords=("book", "library", "borrow"))),
    ("Include keywords *chlorophyll*, *sunlight*, *carbon dioxide* in the response.",
     S(keywords=("chlorophyll", "sunlight", "carbon dioxide"))),
    ("Include keywords [love, joy, happiness] in the response.", S(keywords=("love", "joy", "happiness"))),
    ('In your response, the word "joy" should appear at least 3 times.', S(frequencies=(("joy", ge(3)),))),
    ("In your response, the word travel should appear at least 3 times.", S(frequencies=(("travel", ge(3)),))),
    ('The word "joy" should appear less than 2 times.', S(frequencies=(("joy", le(1)),))),
    ("Include keywords travel, adventure in the response.", S(keywords=("travel", "adventure"))),
    ("Include the keywords: research, dress, questions.", S(keywords=("research", "dress", "questions"))),
    ("Include keywords related to SEO in your response.", S()),
    ('It should also include the keywords: "enchanted," "brave," and "whispering."',
     S(keywords=("enchanted", "brave", "whispering"))),
    ('Mention "Paris" and "Rome" in your answer.', S(keywords=("Paris", "Rome"))),
    ("Use the word 'cat' or 'dog'.", S()),
    ('Write a story that uses the phrase "stardust in your hair".', S(keywords=("stardust in your hair",))),
    ('Do not include keywords [sad] in the response.', S(forbidden=("sad",))),
    ('Avoid the words "however" and "therefore".', S(forbidden=("however", "therefore"))),
    ("Don't mention 'cats' or 'dogs'.", S(forbidden=("cats", "dogs"))),
    ('The word "very" should not appear in your answer.', S(forbidden=("very",))),
    ('Write exactly 100 words, include the keyword "information", and avoid the use of the word "you".',
     S(words=ex(100), keywords=("information",), forbidden=("you",))),
    ('Include keywords [keywords] in the response. [keywords] are "sunny day", "fresh air".',
     S(keywords=("sunny day", "fresh air"))),
    ("Do not include keywords [forbidden_words] in the response. [forbidden_words] are: error, mistake.",
     S(forbidden=("error", "mistake"))),
    ("In your response, the letter [e] should appear at least 15 times.", S(letters=(("e", ge(15)),))),
    ('If I wanted to say "a lot" in the sentence how would I modify it to include "a lot"?', S()),
    ('The first array, "x", should contain the phrase "Hello" in this C program.', S()),
    ("Write a function that takes an `index` and returns items up to and including `index`.", S()),
    ('She is the author of several books, including "Off the Clock" and "168 Hours".', S()),
    ("The story should include the keywords: *envelope, secret, and discovery*.",
     S(keywords=("envelope", "secret", "discovery"))),
    ("Include keywords *red, blue, or green* in the response.", S()),
    ("Write a Python function to extract the metadata and return it as a JSON string.", S()),
    ("Create a Python function to read the XML file and format it into a JSON document.", S()),
    ("How would condensing the information into one sentence impact the meaning?", S()),
    ("Write a Python function to read a header (a single line of text) at the beginning of the text.", S()),
    # formats
    ("Answer in JSON.", S(json_output=True)),
    ("Format your entire response as valid JSON.", S(json_output=True)),
    ("Entire output should be wrapped in JSON format.", S(json_output=True)),
    ("Respond only with a JSON object.", S(json_output=True)),
    ("Write a function that returns a JSON object.", S()),
    ("Convert this XML to JSON.", S()),
    ("When a response needs to be in JSON format, use double quotes.", S()),
    ("Don't answer in JSON.", S()),
    ("Put the code in a code block.", S(code_block=True)),
    ("Wrap your answer in triple backticks.", S(code_block=True)),
    ("Use a fenced code block.", S(code_block=True)),
    ("Explain how code blocks work in markdown.", S()),
    ("Your entire response should be in English, and in all lowercase letters. no capital letters are allowed.",
     S(case="lower")),
    ("Respond in all caps.", S(case="upper")),
    ("Your entire answer should be in English and in all capital letters.", S(case="upper")),
    ("Write a Python function that converts a string to lowercase.", S()),
    ("Convert this substring to all lowercase and sort it.", S()),
    ("In your response, words with all capital letters should appear at least 2 times.", S(capital_words=ge(2))),
    ("Write in all lowercase but use all caps for the title.", S()),
    ("End your response with a question.", S(end_question=True)),
    ("Conclude with a follow-up question.", S(end_question=True)),
    ("How do I end an email with a question?", S()),
    ("Do not use any commas.", S(no_commas=True)),
    ("In your entire response, refrain from the use of any commas.", S(no_commas=True)),
    ("No commas after the greeting, please.", S()),
    # mixed IFEval-style prompts
    ("What are the benefits of solar energy? Your response should contain at least 5 sentences. "
     'Highlight at least 2 sections with markdown. Include keywords "cost", "clean".',
     S(sentences=ge(5), keywords=("cost", "clean"), highlights=ge(2))),
    ('Provide a response to the following question: "What are the key elements of a plan?" '
     "Your response should contain at least 5 sentences and include a postscript starting with P.S.",
     S(sentences=ge(5), postscript="P.S.")),
    ("Please ensure the note is less than 60 words and includes the keywords \"gift\" and \"thanks\".",
     S(words=le(59), keywords=("gift", "thanks"))),
    ("Summarize the meeting, ensuring that your response:\n- Contains at least 5 sentences.\n- Is in all lowercase.",
     S(sentences=ge(5), case="lower")),
    ("Can you help me create a note that is at least 50 words long?", S(words=ge(50))),
]


@pytest.mark.parametrize("text,want", CASES, ids=[c[0][:50] for c in CASES])
def test_compile_hand_labelled(text, want):
    assert C.compile(text) == want


def test_compile_edge_inputs():
    assert C.compile("") == C.EMPTY and C.compile("   ") == C.EMPTY
    assert C.compile(None) == C.EMPTY  # type: ignore[arg-type]
    long_doc = ("The committee must have at least 5 members. " * 400) + "Summarize the text above in two sentences."
    assert C.compile(long_doc).sentences == ex(2)          # only the head and the tail are read
    assert C.compile("Answer in three sentences.") is C.compile("Answer in three sentences.")   # cached


# =============================================================================== verify
@pytest.mark.parametrize("spec,text,violated", [
    (S(sentences=ex(2)), "One. Two.", []),
    (S(sentences=ex(2)), "One. Two. Three.", ["sentences"]),
    (S(words=le(3)), "one two three", []),
    (S(words=le(3)), "one two three four", ["words"]),
    (S(words=ge(3)), "one two", ["words"]),
    (S(items=ex(2)), "- a\n- b", []),
    (S(items=ex(0)), "Prose only.", []),
    (S(items=ex(0)), "- a", ["items"]),
    (S(paragraphs=bt(2, 3)), "a\n\nb", []),
    (S(paragraphs=bt(2, 3)), "a", ["paragraphs"]),
    (S(lines=ex(2)), "a\n\nb\n", []),
    (S(keywords=("dress",)), "Dressing well matters.", []),
    (S(keywords=("dress",)), "Address the issue.", ["keyword"]),
    (S(keywords=("carbon dioxide",)), "Carbon\ndioxide is a gas.", []),
    (S(forbidden=("cat",)), "Cats are fine.", []),
    (S(forbidden=("cat",)), "A cat.", ["forbidden"]),
    (S(json_output=True), '{"a": 1}', []),
    (S(json_output=True), "a: 1", ["json_output"]),
    (S(code_block=True), "```\nx\n```", []),
    (S(code_block=True), "```\nx", ["code_block"]),
    (S(case="lower"), "all lower 1", []),
    (S(case="lower"), "Not lower", ["case"]),
    (S(case="upper"), "ALL UPPER", []),
    (S(end_question=True), "Right?", []),
    (S(end_question=True), "Right.", ["end_question"]),
    (S(no_commas=True), "a b", []),
    (S(no_commas=True), "a, b", ["no_commas"]),
    (S(sentences=ex(1), words=le(2), no_commas=True), "a, b c.", ["words", "no_commas"]),
])
def test_verify(spec, text, violated):
    got = C.verify(spec, text)
    assert [v.split(":")[0] for v in got] == violated
    assert C.verify(spec, text, C.measure(text)) == got


# ============================================================================== render
VOCAB = ["ocean", "carbon dioxide", "Paris", "machine learning", "json", "lowercase", "question",
         "commas", "words", "three sentences", "bullet points", "don't", "e-mail", "C3PO", "x",
         "naïve", "New York City", "rock and roll", "at least", "include", "not", "keywords", "never"]


def random_spec(rng: np.random.Generator) -> ConstraintSpec:
    tops = {"words": 600, "sentences": 25, "items": 15, "paragraphs": 8, "lines": 20}
    values: dict = {}
    for unit, top in tops.items():
        if rng.random() < 0.3:
            k, n = int(rng.integers(4)), int(rng.integers(1, top + 1))
            values[unit] = [ex(n), le(n), ge(n), bt(n, int(rng.integers(n + 1, top + 3)))][k]
    if rng.random() < 0.08:
        values["items"] = ex(0)
    words = [str(w) for w in rng.permutation(VOCAB)]
    if rng.random() < 0.3:
        values["keywords"] = tuple(words[: int(rng.integers(1, 4))])
    if rng.random() < 0.2:
        values["forbidden"] = tuple(words[5: 5 + int(rng.integers(1, 3))])
    for flag in ("json_output", "code_block", "end_question", "no_commas"):
        if rng.random() < 0.15:
            values[flag] = True
    if rng.random() < 0.2:
        values["case"] = "lower" if rng.random() < 0.5 else "upper"
    add_further_families(values, rng, words)
    return ConstraintSpec(**values)


PHRASES = ["Yes", "No", "Dear Sir", "Once upon a time", "Any other questions?", "Best wishes",
           "Is there anything else I can help with?", "Have a nice day!", "My answer is yes.",
           "My answer is no.", "My answer is maybe.", "Welcome", "In summary", "Thank you",
           "See you soon.", "Hope this helps!", "naïve start", "C3PO says hi", "true", "false"]
SECTION_MARKERS = ["Section", "SECTION", "Part", "Chapter", "Day"]
PS_MARKERS = ["P.S.", "P.P.S", "p.s.", "PS", "Note", "P. S."]
FIRST_WORDS = ["however", "finally", "moreover", "historically", "dogs", "summer", "the", "a"]
SEPARATORS = ["******", "***", "---", "===", "~~~", "+++"]


def small_interval(rng, top: int = 6, allow_zero: bool = False, exclude_ge1: bool = False) -> Interval:
    while True:
        k, n = int(rng.integers(4)), int(rng.integers(1, top + 1))
        iv = [ex(n), le(n), ge(n), bt(n, int(rng.integers(n + 1, top + 3)))][k]
        if allow_zero and rng.random() < 0.15:
            iv = ex(0)
        if exclude_ge1 and iv == ge(1):
            continue
        return iv


def add_further_families(values: dict, rng, words: list[str]) -> None:
    """The instruction families beyond counts, keywords and simple formats."""
    items = values.get("items")
    if rng.random() < 0.15 and (items is None or items.hi != 0):
        values["list_style"] = "bullet" if rng.random() < 0.5 else "numbered"
    forbidden = {w.lower() for w in values.get("forbidden", ())}
    if rng.random() < 0.15:
        pool = [w for w in words if w.lower() not in forbidden]
        k = int(rng.integers(1, 3))
        values["frequencies"] = tuple((w, small_interval(rng, exclude_ge1=True)) for w in pool[-k:])
    if rng.random() < 0.1:
        letters = rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=int(rng.integers(1, 3)), replace=False)
        values["letters"] = tuple((str(ch), small_interval(rng, 30, allow_zero=True)) for ch in letters)
    for name in ("highlights", "placeholders", "capital_words"):
        if rng.random() < 0.1:
            values[name] = small_interval(rng)
    if rng.random() < 0.08:
        values["sections"] = (SECTION_MARKERS[int(rng.integers(len(SECTION_MARKERS)))], small_interval(rng))
    if rng.random() < 0.08:
        values["first_word"] = (int(rng.integers(1, 6)), FIRST_WORDS[int(rng.integers(len(FIRST_WORDS)))])
    for flag in ("divider", "title", "quoted"):
        if rng.random() < 0.08:
            values[flag] = True
    if rng.random() < 0.1:
        values["postscript"] = PS_MARKERS[int(rng.integers(len(PS_MARKERS)))]
    phrases = [str(p) for p in rng.permutation(PHRASES)]
    if rng.random() < 0.1:
        values["starts_with"] = tuple(p.rstrip(".") for p in phrases[: int(rng.integers(1, 3))])
    if rng.random() < 0.1:
        values["end_phrase"] = phrases[5]
    if rng.random() < 0.08:
        values["one_of"] = tuple(phrases[8: 8 + int(rng.integers(1, 4))])
    if rng.random() < 0.06:
        values["responses"] = (SEPARATORS[int(rng.integers(len(SEPARATORS)))], int(rng.integers(2, 5)))


PREFIXES = ["", "Tell me about the history of Rome.", "What is photosynthesis?",
            "Write a short story about a dragon who loves tea", "Hi!\n\nI have a question:",
            "Summarize the following text:\n\nThe city was founded long ago and grew quickly."]


@pytest.mark.parametrize("seed", range(4))
def test_render_round_trip(seed):
    rng = np.random.default_rng(seed)
    for i in range(1500):
        spec = random_spec(rng)
        style = (None, "bullet", "numbered")[i % 3]
        text = C.render(spec, rng, item_style=style)
        assert C.compile(text) == spec, text
        prefix = PREFIXES[i % len(PREFIXES)]
        sep = "\n\n" if not prefix or prefix[-1] not in ".?!:" else " "
        assert C.compile(prefix + sep + text) == spec, prefix + sep + text


def test_render_uses_several_templates():
    rng = np.random.default_rng(0)
    for spec in (S(sentences=ex(3)), S(words=le(50)), S(items=ge(2)), S(keywords=("ocean",)),
                 S(json_output=True), S(case="lower"), S(forbidden=("cat",))):
        texts = {C.render(spec, rng) for _ in range(60)}
        assert len(texts) >= 4, spec
    assert C.render(C.EMPTY, rng) == ""


# ============================================================================== targets
def ref_targets(row: np.ndarray, seg: np.ndarray | None, tok) -> np.ndarray:
    """Walk back from every position: its turn start, that turn's role, the turn before it."""
    from bes.data.chat_format import display_text

    n = row.shape[0]
    out = np.full((n, 3), -1, np.int32)
    roles = set(config.ROLE_TOKENS)

    def seg_start(t):
        s = t
        while s > 0 and (seg is None or seg[s] == seg[s - 1]):
            s -= 1
        return s

    for t in range(n):
        lo = seg_start(t)
        a = t
        while a > lo and row[a] not in roles and row[a] != BOS:
            a -= 1
        if row[a] != ASSISTANT:
            continue
        if a == lo:
            continue                       # the assistant token starts the segment
        p = a - 1
        while p > lo and row[p] not in roles and row[p] != BOS:
            p -= 1
        if row[p] != USER:
            continue
        content = []
        for q in range(p + 1, a):
            if row[q] == END:
                break
            content.append(int(row[q]))
        spec = C.compile(display_text(np.asarray(content, np.int64), tok))
        out[t] = spec.targets()
    return out


INSTRUCTIONS = ["Answer in exactly three sentences.", "Use at most 12 words.", "List 3 fruits.",
                "Write between 2 and 4 bullet points.", "Hello there, how are you?",
                "Use at least 5 sentences.", "Write a 20-word summary.", "Do not use bullet points."]


def random_rows(tok, seed: int, n_rows: int = 3, n_t: int = 260):
    rng = np.random.default_rng(seed)
    pieces = [tok.encode(s) for s in INSTRUCTIONS] + [tok.encode(" One. Two."), tok.encode("\n- a\n- b")]
    toks = np.empty((n_rows, n_t), np.int32)
    seg = np.empty((n_rows, n_t), np.int32)
    for b in range(n_rows):
        row, sg, s = [], [], 0
        while len(row) < n_t:
            if rng.random() < 0.8:
                row.append(BOS)
                sg.append(s)
            for _ in range(int(rng.integers(1, 6))):
                role = [SYSTEM, USER, ASSISTANT, LIBRARY, USER, ASSISTANT][int(rng.integers(6))]
                body = list(pieces[int(rng.integers(len(pieces)))]) if rng.random() < 0.8 else []
                body += [int(x) for x in rng.integers(0, 3000, int(rng.integers(0, 4)))]
                turn = [role, *body] + ([END] if rng.random() < 0.9 else [])
                row += turn
                sg += [s] * len(turn)
            s += 1 if rng.random() < 0.7 else 0
        toks[b], seg[b] = row[:n_t], sg[:n_t]
    return toks, seg


@pytest.mark.parametrize("seed", range(4))
def test_targets_match_brute_force(tok, seed):
    toks, seg = random_rows(tok, seed)
    for use_seg in (True, False):
        s = seg if use_seg else None
        got = C.targets_for_tokens(toks, s, tok)
        assert got.shape == toks.shape + (3,) and got.dtype == np.int32
        for b in range(toks.shape[0]):
            want = ref_targets(toks[b], None if s is None else s[b], tok)
            np.testing.assert_array_equal(got[b], want)
        assert (got >= 0).any()                    # the rows exercise the targets
    assert np.array_equal(C.compile_from_tokens(toks, seg, tok), C.targets_for_tokens(toks, seg, tok))


@pytest.mark.parametrize("seed", range(3))
def test_incremental_targets_equal_batch(tok, seed):
    toks, seg = random_rows(tok, 10 + seed, n_rows=2)
    want = C.targets_for_tokens(toks, seg, tok)
    inc = C.IncrementalTargets(toks.shape[0], tok)
    rows = []
    for t in range(toks.shape[1]):
        new = seg[:, t] != seg[:, t - 1] if t else np.zeros(toks.shape[0], bool)
        rows.append(inc.step(toks[:, t], new_segment=new))
    np.testing.assert_array_equal(np.stack(rows, axis=1), want)
    # copy and select reproduce the continuation
    half = toks.shape[1] // 2
    inc2 = C.IncrementalTargets(1, tok)
    for t in range(half):
        inc2.step(toks[0, t], new_segment=t > 0 and seg[0, t] != seg[0, t - 1])
    fork, picked = inc2.copy(), inc2.select([0, 0])
    for t in range(half, toks.shape[1]):
        new = bool(seg[0, t] != seg[0, t - 1])
        a = fork.step(toks[0, t], new_segment=new)
        b = picked.step(np.array([toks[0, t]] * 2), new_segment=np.array([new, new]))
        np.testing.assert_array_equal(a, want[0, t])
        np.testing.assert_array_equal(b, np.stack([want[0, t]] * 2))


def test_targets_shapes_and_errors(tok):
    assert C.targets_for_tokens(np.zeros((2, 0), np.int32), None, tok).shape == (2, 0, 3)
    with pytest.raises(ValueError):
        C.targets_for_tokens(np.zeros(5, np.int32), None, tok)
    with pytest.raises(ValueError):
        C.targets_for_tokens(np.zeros((1, 5), np.int32), np.zeros((1, 4), np.int32), tok)
    with pytest.raises(ValueError):
        C.IncrementalTargets(2, tok).step(np.array([1, 2, 3]))


def conversation_ids(tok, user: str, reply, extra=()):
    from bes.data import chat_format as cf

    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}, *extra]
    return cf.render_conversation(msgs, tok)[0]


@pytest.mark.parametrize("user,reply,field,count", [
    ("Write exactly three sentences about the sea.", "The sea is vast. It is salty. It covers most of Earth.",
     "sentences", 3),
    ("Answer with exactly 2 bullet points.", "- blue\n- green\n", "items", 2),
    ("Use exactly 7 words.", "Seven words make up this whole answer", "words", 7),
    ("Answer in at most 4 sentences.", "One. Two. Three. Four.", "sentences", 4),
])
def test_countdown_reaches_zero_at_the_right_token(tok, user, reply, field, count):
    """The targets feed registers.compute; inside the assistant turn the countdown equals
    2 + target - (count so far), so it is 2 (nothing left) from the token that completes the
    count to END, and larger before."""
    from bes.exact import registers as R

    ids = conversation_ids(tok, user, reply)
    targets = C.targets_for_tokens(ids[None], None, tok)
    regs = R.compute(ids[None], None, targets)
    names = [n for n, _ in config.REG_FIELDS]
    reg_name = {"sentences": "rem_sent", "items": "rem_item", "words": "rem_word"}[field]
    rem = regs.reg[0, :, names.index(reg_name)]
    raw = regs.raw[field][0]
    a = ids.tolist().index(ASSISTANT)
    end = len(ids) - 1
    assert ids[end] == END
    col = C.TARGET_UNITS.index(field)
    assert (targets[0, :a, col] == -1).all() and (targets[0, a:, col] == count).all()
    assert (rem[:a] == 0).all()
    for t in range(a, end + 1):
        k = count - raw[t]
        assert rem[t] == (2 + k if k < 16 else rem[t]), t
    first_done = next(t for t in range(a, end + 1) if raw[t] == count)
    assert (rem[first_done:end + 1] == 2).all() and (rem[a:first_done] > 2).all()
    assert raw[end] == count


def test_targets_only_from_the_preceding_user_turn(tok):
    from bes.data import chat_format as cf

    msgs = [{"role": "system", "content": "Always answer in exactly two sentences."},
            {"role": "user", "content": "Use at most 10 words."},
            {"role": "assistant", "content": "Short answer here."},
            {"role": "user", "content": "And now?"},
            {"role": "assistant", "content": "Still short."}]
    ids = cf.render_conversation(msgs, tok)[0]
    tg = C.targets_for_tokens(ids[None], None, tok)[0]
    a1, a2 = [i for i, v in enumerate(ids.tolist()) if v == ASSISTANT]
    u2 = [i for i, v in enumerate(ids.tolist()) if v == USER][1]
    assert (tg[a1:u2] == [-1, -1, 10]).all()          # system prompts are not read
    assert (tg[:a1] == -1).all() and (tg[u2:] == -1).all()
    # a trailing generation prompt already carries the target (what IncrementalFeatures reads)
    gen = cf.render_conversation(msgs[1:2], tok, add_generation_prompt=True)[0]
    assert C.targets_for_tokens(gen[None], None, tok)[0, -1].tolist() == [-1, -1, 10]


def test_features_with_real_constraints(tok):
    """bes.exact.features uses targets_for_tokens; its incremental path equals the batch one."""
    from bes.exact import features

    toks, seg = random_rows(tok, 99, n_rows=2, n_t=200)
    head = conversation_ids(tok, "Answer with exactly 2 bullet points.", "- blue\n- green\n")
    toks[:, : len(head)] = head                    # a user -> assistant pair with a target
    seg[:, : len(head)] = -1
    want = features.exact_features(toks, seg, tok)
    names = [n for n, _ in config.REG_FIELDS]
    rem = [names.index(n) for n in ("rem_sent", "rem_item", "rem_word")]
    assert (want["reg"][..., rem] > 0).any()
    for b in range(toks.shape[0]):
        got = features.IncrementalFeatures(tok).extend(toks[b], seg[b])
        for key in features.FEATURE_KEYS:
            np.testing.assert_array_equal(got[key], want[key][b], err_msg=key)


def test_targets_throughput(tok):
    from bes.data import chat_format as cf

    rng = random.Random(0)
    convs = []
    for i in range(300):
        user = rng.choice(INSTRUCTIONS) + " " + " ".join(rng.choice(["alpha", "beta", "gamma"]) for _ in range(30))
        reply = " ".join(rng.choice(["One.", "two", "three", "- item\n"]) for _ in range(80))
        convs.append(cf.render_conversation([{"role": "user", "content": user + str(i)},
                                             {"role": "assistant", "content": reply}], tok)[0])
    flat = np.concatenate(convs)
    rows = np.resize(flat, (32, 1024)).astype(np.int32)
    C._compile_cached.cache_clear()
    t0 = time.perf_counter()
    C.targets_for_tokens(rows, None, tok)
    assert 32 / (time.perf_counter() - t0) > 60          # SPEC 2.3 budget for the whole exact layer


# ================================================================ further instruction families
# ------------------------------------------------------------------ independent references
def ref_highlights(text: str) -> int:
    """Left-to-right scan: an opening run of one or two stars (not after a star), a first
    character that is neither whitespace nor a star, text up to the next star or newline, a
    last character that is not whitespace and a closing run of the same length not followed by
    a star."""
    n, i = 0, 0
    while i < len(text):
        if text[i] == "*" and (i == 0 or text[i - 1] != "*"):
            k = 2 if text[i:i + 2] == "**" else 1
            j = i + k
            if j < len(text) and not text[j].isspace() and text[j] != "*":
                e = j
                while e < len(text) and text[e] not in "*\n":
                    e += 1
                if (e < len(text) and text[e:e + k] == "*" * k and not text[e - 1].isspace()
                        and (e + k == len(text) or text[e + k] != "*")):
                    n += 1
                    i = e + k
                    continue
        i += 1
    return n


def ref_placeholders(text: str) -> int:
    n, i = 0, 0
    while i < len(text):
        if text[i] == "[":
            j = i + 1
            while j < len(text) and text[j] not in "[]\n":
                j += 1
            if j < len(text) and text[j] == "]" and text[i + 1:j].strip():
                n += 1
                i = j + 1
                continue
        i += 1
    return n


def ref_title(text: str) -> bool:
    for i in range(len(text) - 1):
        if text[i:i + 2] == "<<":
            j = i + 2
            while j < len(text) and text[j] not in "<>\n":
                j += 1
            if text[j:j + 2] == ">>" and text[i + 2:j].strip():
                return True
    return False


def ref_capital_words(text: str) -> int:
    words, cur = [], ""
    for i, ch in enumerate(text):
        if ch.isalnum():
            cur += ch
        elif ch in "'’" and cur and i + 1 < len(text) and text[i + 1].isalnum():
            cur += ch
        else:
            if cur:
                words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    return sum(1 for w in words if any(c.isupper() for c in w) and not any(c.islower() for c in w))


def ref_paragraph_texts(text: str) -> list[str]:
    lines = text.split("\n")
    flags = ref_fence_flags(lines)
    tokens, any_div = [], False
    for line, fenced in zip(lines, flags):
        body = line.rstrip("\r")
        pieces = [body] if fenced else ref_dividers(body)
        if pieces is None:
            tokens.append(None)
            any_div = True
            continue
        for k, piece in enumerate(pieces):
            if k:
                tokens.append(None)
                any_div = True
            tokens.append(piece)
    out: list[list[str]] = []
    cur = None
    if any_div:
        for t in tokens:
            if t is None:
                cur = None
            elif cur is None:
                if t.strip(LINE_WS):
                    cur = [t]
                    out.append(cur)
            else:
                cur.append(t)
        return ["\n".join(c) for c in out]
    for line, fenced in zip(lines, flags):
        if not fenced and (not line.strip(LINE_WS) or ref_is_break(line)):
            cur = None
        else:
            if cur is None:
                cur = []
                out.append(cur)
            cur.append(line.rstrip("\r"))
    return ["\n".join(c) for c in out]


MARKUP = " \t\n\r\x0b\x0c#*_>\"'([“‘«"


def ref_first_word(text: str) -> str:
    i = 0
    while i < len(text) and text[i] in MARKUP:
        i += 1
    j = i
    while j < len(text) and text[j].isalnum():
        j += 1
    return text[i:j].lower()


FAMILY_PIECES = PIECES + ["*", "**", "*x*", "**bold**", "* item", "[", "]", "[name]", "[ ]", "<<", ">>",
                          "<<Title>>", "NASA", "I", "don't", "DON'T", "Ünited", "ÉTÉ", "3D", "e", "E",
                          "P.S.", "Section 1", "\n\n", "***"]


def family_text(rng: random.Random, n: int = 30) -> str:
    return "".join(rng.choice(FAMILY_PIECES) for _ in range(rng.randint(0, n)))


@pytest.mark.parametrize("seed", range(3))
def test_further_measures_match_references(seed):
    rng = random.Random(100 + seed)
    for _ in range(1500):
        t = family_text(rng)
        m = C.measure(t)
        assert m["highlights"] == ref_highlights(t) == C.count_highlights(t), t
        assert m["placeholders"] == ref_placeholders(t), t
        assert m["has_title"] == ref_title(t), t
        assert m["capital_words"] == ref_capital_words(t), t
        assert m["quoted"] == (len(t.strip(" \t\n\r\x0b\x0c")) >= 2 and t.strip(" \t\n\r\x0b\x0c")[0] == '"'
                               and t.strip(" \t\n\r\x0b\x0c")[-1] == '"')
        pars = C.paragraph_texts(t)
        assert pars == ref_paragraph_texts(t), t
        assert len(pars) == m["paragraphs"] == ref_paragraphs(t)
        assert m["paragraph_first_words"] == [ref_first_word(p) for p in pars]
        assert m["first_word"] == ref_first_word(t)
        assert m["dividers"] >= 0 and (m["dividers"] > 0) == any(
            ref_dividers(line.rstrip("\r")) is None or len(ref_dividers(line.rstrip("\r"))) > 1
            for line, f in zip(t.split("\n"), ref_fence_flags(t.split("\n"))) if not f)
        for ch in "eaz":
            assert C.letter_count(t, ch) == sum(1 for c in t if c in (ch, ch.upper()))


@pytest.mark.parametrize("text,key,want", [
    ("This is *very* important, **really**.", "highlights", 2), ("* item\n* item", "highlights", 0),
    ("a * b * c", "highlights", 0), ("***x***", "highlights", 0), ("*a*b*", "highlights", 1),
    ("**Note:** read *this*", "highlights", 2), ("*multi\nline*", "highlights", 0),
    ("Dear [Name], see you on [Date].", "placeholders", 2), ("[ ] and []", "placeholders", 0),
    ("[a [b] c]", "placeholders", 1), ("<<My Poem>>\nText", "has_title", True), ("<< >>", "has_title", False),
    ("<<a\nb>>", "has_title", False), ('  "Quoted answer."  ', "quoted", True), ('"Quoted".', "quoted", False),
    ('"', "quoted", False), ("“Curly”", "quoted", False), ("I think NASA is GREAT. ok", "capital_words", 3),
    ("DON'T stop", "capital_words", 1), ("3D and 42", "capital_words", 1), ("ÉTÉ été", "capital_words", 1),
    ("**Hello** world\n\nNext one", "paragraph_first_words", ["hello", "next"]), ("# Title", "first_word", "title"),
    ("- Item one", "first_word", ""), ('"Yes," she said', "first_word", "yes"), ("", "first_word", ""),
    ("p1\n\n***\n\np2", "dividers", 1), ("p1\n\np2", "dividers", 0), ("a *** b", "dividers", 1),
])
def test_further_measures_hand_labelled(text, key, want):
    assert C.measure(text)[key] == want


@pytest.mark.parametrize("text,phrase,start,end", [
    ("Yes, it is.", "Yes", True, False), ("Yesterday it rained.", "Yes", False, False),
    ("**Yes**, of course.", "yes", True, False), ('"Welcome to our site!"', "Welcome", True, False),
    ("Dear  Sir,\nthanks", "Dear Sir", True, False), ("Is there anything else I can help with?", "help with?", False, True),
    ("Bye. Any other questions? ", "Any other questions?", False, True), ("... any other questions?**", "Any other questions?", False, True),
    ("Many other questions?", "any other questions?", False, False), ("Hope this helps!\n", "Hope this helps!", True, True),
    ("Hope this helps.", "Hope this helps", True, False), ("Best wishes", "Best wishes", True, True),
])
def test_phrase_checks(text, phrase, start, end):
    assert C.starts_with_phrase(text, phrase) == start
    assert C.ends_with_phrase(text, phrase) == end


def _ref_starts(text: str, phrase: str) -> bool:
    i = 0
    while i < len(text) and text[i] in MARKUP:
        i += 1
    head = " ".join(text[i:].split())
    p = " ".join(phrase.split())
    if not head.lower().startswith(p.lower()):
        return False
    rest = head[len(p):]
    return not (p[-1].isalnum() and rest[:1].isalnum())


def ref_starts_with(text: str, phrase: str) -> bool:
    """The text or its body after leading <<title>> / markdown heading lines starts with the
    phrase."""
    if _ref_starts(text, phrase):
        return True
    lines = text.lstrip(" \t\n\r\x0b\x0c").split("\n")
    k = 0
    while k < len(lines) and (lines[k].strip(" \t").startswith("<<") and lines[k].strip(" \t").endswith(">>")
                              and "<" not in lines[k].strip(" \t")[2:-2] and ">" not in lines[k].strip(" \t")[2:-2]
                              or re.match(r"[ \t]*#{1,6}[ \t]", lines[k])):
        k += 1
        while k < len(lines) and not lines[k].strip(" \t"):
            k += 1
    body = "\n".join(lines[k:])
    return 0 < k and body.strip() != "" and _ref_starts(body, phrase)


def ref_ends_with(text: str, phrase: str) -> bool:
    t = " ".join(text.split()).lower().rstrip(" \t\n\r\x0b\x0c\"'*_”’")
    p = " ".join(phrase.split()).lower().rstrip(" \t\n\r\x0b\x0c\"'*_”’")
    if not p or not t.endswith(p):
        return False
    before = t[: len(t) - len(p)]
    return not (p[0].isalnum() and before[-1:].isalnum())


def test_phrase_checks_match_references():
    rng = random.Random(7)
    alphabet = ["yes", "Yes", "YES", "no", " ", "  ", "\n", ",", ".", "!", "?", "*", "**", '"', "'", "#",
                "help", "with", "x", "s", "es", "<<T>>", "<<", ">>", "# ", "\n\n"]
    phrases = ["yes", "Yes no", "help with?", "with!", "x", "s es"]
    for _ in range(3000):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        for p in phrases:
            assert C.starts_with_phrase(t, p) == ref_starts_with(t, p), (t, p)
            assert C.ends_with_phrase(t, p) == ref_ends_with(t, p), (t, p)


@pytest.mark.parametrize("text,marker,want", [
    ("Hi.\n\nP.S. See you.", "P.S.", True), ("Hi.\n\np. s. see you", "P.S.", True), ("Hi. P.S. inline", "P.S.", True), ("Hi P.S. inline", "P.S.", False),
    ("x.P.S. y", "P.S.", False),
    ("**P.S.** bold", "P.S.", True), ("PS: no dots", "P.S.", False), ("P.P.S also", "P.P.S", True),
    ("P.S. only", "P.P.S", False), ("> P.S. quoted", "P.S.", True), ("P.S. Remember the sunscreen!", "P.S. Remember", True),
])
def test_has_postscript(text, marker, want):
    assert C.has_postscript(text, marker) == want


@pytest.mark.parametrize("text,marker,want", [
    ("Section 1\na\nSection 2\nb", "Section", 2), ("SECTION 1: x\n## SECTION 2\n**SECTION 3**", "Section", 3),
    ("See Section 2 below", "Section", 0), ("Section one", "Section", 0), ("Part 1\nPart 2", "Section", 0),
    ("  Day 1\n  Day 12\nDay x", "Day", 2),
])
def test_count_sections(text, marker, want):
    assert C.count_sections(text, marker) == want


def test_split_responses_repeat_and_options():
    assert C.split_responses("a ****** b", "******") == (["a", "b"], True)
    assert C.split_responses("****** a ****** b ******", "******") == (["a", "b"], True)
    assert C.split_responses("a ****** ****** b", "******") == (["a", "b"], False)
    assert C.repeats_request("Write a poem.\n\nRoses are red.", "write a  poem.")
    assert not C.repeats_request("Sure! Write a poem.", "Write a poem.")
    assert C.option_key("  My answer is **yes**. ") == "my answer is yes"
    assert C.matches_option("Yes.", ("yes", "no")) and not C.matches_option("Yes, it is.", ("yes", "no"))
    assert not C.matches_option("", ("yes",))


# --------------------------------------------------------------------------- spec validation
def test_further_spec_validation():
    bad = [{"list_style": "dashes"}, {"list_style": "bullet", "items": ex(0)},
           {"frequencies": (("x", ge(1)),)}, {"frequencies": (("x", le(0)),)}, {"frequencies": (("x", ge(0)),)},
           {"frequencies": (("x", ex(2)), ("X", ex(3)))}, {"frequencies": (("x", ex(2)),), "forbidden": ("x",)},
           {"letters": (("ab", ge(2)),)}, {"letters": (("E", ge(2)),)}, {"highlights": ex(0)},
           {"placeholders": ge(0)}, {"sections": ("1st", ge(2))}, {"sections": ("Section", le(0))},
           {"first_word": (0, "x")}, {"first_word": (1, "Hello")}, {"first_word": (1, "two words")},
           {"postscript": "1PS"}, {"end_phrase": 'say "hi"'}, {"end_phrase": " padded"}, {"starts_with": ("a",) * 2},
           {"starts_with": tuple("abcde")}, {"one_of": ("Yes.", "yes")}, {"responses": ("ab", 2)},
           {"responses": ("***", 1)}, {"repeat": "  "}, {"repeat": " padded "}, {"divider": 1}]
    for kw in bad:
        with pytest.raises((ValueError, TypeError)):
            ConstraintSpec(**kw)
    spec = S(frequencies=(("zeta", ex(2)), ("Alpha", ge(3))), letters=(("z", le(3)), ("a", ex(0))))
    assert spec.frequencies == (("Alpha", ge(3)), ("zeta", ex(2))) and spec.letters == (("a", ex(0)), ("z", le(3)))
    assert spec == S(frequencies=(("Alpha", ge(3)), ("zeta", ex(2))), letters=(("z", le(3)), ("a", ex(0))))
    full = S(list_style="numbered", items=ex(3), highlights=ge(2), sections=("Part", ex(2)), first_word=(2, "however"),
             divider=True, title=True, postscript="P.S.", quoted=True, starts_with=("Yes", "No"),
             end_phrase="Bye.", one_of=("yes", "no"), responses=("******", 2), repeat="Write it.")
    assert len(full.constraints) == 14 and full.targets() == (-1, 3, -1)


def test_further_merge():
    a = S(frequencies=(("x", ge(2)),), starts_with=("Yes", "No"), one_of=("yes", "no", "maybe"), title=True)
    b = S(frequencies=(("X", le(4)),), starts_with=("no",), one_of=("No", "maybe"), letters=(("e", ge(3)),))
    m = C.merge(a, b)
    assert m == S(frequencies=(("x", bt(2, 4)),), starts_with=("No",), one_of=("no", "maybe"), title=True,
                  letters=(("e", ge(3)),))
    assert C.merge(S(list_style="bullet"), S(list_style="numbered")) is None
    assert C.merge(S(starts_with=("Yes",)), S(starts_with=("No",))) is None
    assert C.merge(S(frequencies=(("x", le(1)),)), S(frequencies=(("x", ge(3)),))) is None
    assert C.merge(S(sections=("Part", ex(2))), S(sections=("Section", ex(2)))) is None
    assert C.merge(S(end_phrase="Bye."), S(end_phrase="Bye.")) == S(end_phrase="Bye.")
    assert C.merge(S(frequencies=(("x", ex(2)),)), S(forbidden=("x",))) is None


# ------------------------------------------------------------------------------- compile
FURTHER_CASES = [
    # list styles and steps
    ("Present them as a numbered list.", S(list_style="numbered")),
    ("Pick three of them and present them as a numbered list.", S(items=ex(3), list_style="numbered")),
    ("Now explain it in exactly three numbered steps.", S(items=ex(3), list_style="numbered")),
    ("Name five musical instruments as a numbered list, with no other text.", S(items=ex(5), list_style="numbered")),
    ("List four tips for staying focused while studying, as bullet points.", S(items=ex(4), list_style="bullet")),
    ("Turn those tips into exactly four bullet points.", S(items=ex(4), list_style="bullet")),
    ("Explain how to plant a tree in 4 steps.", S(items=ex(4))),
    ("Give me a list of 5 tips for sleeping.", S(items=ex(5))),
    ("Use bullet points or a numbered list.", S()),
    ("Don't use bullet points; use a numbered list instead.", S(list_style="numbered")),
    ("Solve it in 3 steps.", S()),
    ("I like numbered lists.", S()),
    # single words and small word counts
    ("What is the opposite of \"hot\"? Answer with a single word.", S(words=ex(1))),
    ("Reply with one word.", S(words=ex(1))),
    ("Answer with one word only.", S(words=ex(1))),
    ("What color is the sky? One word.", S(words=ex(1))),
    ("Describe yourself in three words.", S(words=ex(3))),
    ("Describe it in 10 words.", S(words=le(10))),
    ("Give me 3 words that describe you.", S()),
    ("Give me a one-word answer: yes, no, or maybe.", S(words=ex(1), one_of=("yes", "no", "maybe"))),
    # follow-up turns, JSON, questions
    ("Now say it again in all capital letters.", S(case="upper")),
    ("Give me a JSON object that describes a fictional person, with the keys \"name\", \"age\" and \"city\". "
     "Output only the JSON.", S(json_output=True)),
    ("Return a JSON array of three colors.", S(json_output=True)),
    ("Just the JSON, please.", S(json_output=True)),
    ("Write a Python function that returns a JSON array.", S()),
    ("What is a JSON object?", S()),
    ("Close by asking me a question.", S(end_question=True)),
    ("Make sure the last sentence is a question.", S(end_question=True)),
    ("What is a good way of ending with a question?", S()),
    # forbidden words and commas
    ("Never say 'basically'.", S(forbidden=("basically",))),
    ('Without using the word "love", describe your mother.', S(forbidden=("love",))),
    ('The words "cat" and "dog" should not appear.', S(forbidden=("cat", "dog"))),
    ('You cannot use the word "the".', S(forbidden=("the",))),
    ("Do not mention apples.", S()),
    ("Do not include any commas in your answer.", S(no_commas=True)),
    ("Your response must not contain any commas.", S(no_commas=True)),
    ("There should be no commas in your response.", S(no_commas=True)),
    ("Write a comma-free paragraph.", S(no_commas=True)),
    ("Don't put commas between the adjectives.", S()),
    ("There are no commas in this sentence.", S()),
    ("Is this sentence comma-free?", S()),
    # keyword and letter frequency
    ('Rewrite the story so that it uses the word "honey" at least twice.', S(frequencies=(("honey", ge(2)),))),
    ('Write a short paragraph about bicycles that uses the word "wheel" exactly twice.',
     S(frequencies=(("wheel", ex(2)),))),
    ('Do not use the word "very" more than twice.', S(frequencies=(("very", le(2)),))),
    ("In your response, the word [success] should appear at least 2 times.", S(frequencies=(("success", ge(2)),))),
    ('Use the word "x" at least once.', S(keywords=("x",))),
    ("the word love should not appear more than 3 times", S(frequencies=(("love", le(3)),))),
    ("Repeat the process twice.", S()),
    ("Say it three times.", S()),
    ("Use the letter q at least 3 times.", S(letters=(("q", ge(3)),))),
    ("Write a sentence without the letter e.", S(letters=(("e", ex(0)),))),
    ("In your response, the letter e should appear less than 5 times.", S(letters=(("e", le(4)),))),
    ("How many times does the letter e appear in this sentence?", S()),
    # highlights, placeholders, capital words
    ("Write a short motivational message, and highlight at least two key phrases with markdown, like *this*.",
     S(highlights=ge(2))),
    ("Highlight at least 1 section in your answer with markdown, i.e. *highlighted section*.", S(highlights=ge(1))),
    ("Use markdown to highlight two sections.", S(highlights=ge(2))),
    ("Highlight the most important points in bold.", S()),
    ("Do not highlight more than 2 words.", S()),
    ("The response must contain at least 2 placeholders represented by square brackets, such as [address].",
     S(placeholders=ge(2))),
    ("Use placeholders where needed.", S()),
    ("Use at least 5 words in all caps.", S(capital_words=ge(5))),
    ("Include no more than 2 all-caps words.", S(capital_words=le(2))),
    # sections, first words, dividers, titles, postscripts, quotes
    ("Your response must have 3 sections. Mark the beginning of each section with SECTION X, such as:\nSECTION 1\n"
     "[content of section 1]\nSECTION 2\n[content of section 2]", S(sections=("SECTION", ex(3)))),
    ("The response must have 2 sections. Mark the beginning of each section with Section 1 and Section 2.",
     S(sections=("Section", ex(2)))),
    ("Write a report with 3 sections.", S()),
    ("There should be 4 paragraphs. Paragraphs and only paragraphs are separated with each other by two new lines "
     "as if it was '\\n\\n' in python. Paragraph 2 must start with word historically.",
     S(paragraphs=ex(4), first_word=(2, "historically"))),
    ('The second paragraph should start with the word "Moreover".', S(first_word=(2, "moreover"))),
    ("Start the second paragraph with a question.", S()),
    ("Separate paragraphs with the markdown divider ***.", S(divider=True)),
    ("Write a title for a blog post about saving money. Wrap the title in double angular brackets, like <<title>>.",
     S(title=True)),
    ("Add a title to your essay.", S()),
    ("Write a short note to a friend about your holiday plans. At the end, add a postscript that starts with P.S.",
     S(postscript="P.S.")),
    ("At the end of your response, please explicitly add a postscript starting with *P.S.*", S(postscript="P.S.")),
    ("At the end of your response, please explicitly add a postscript starting with P.P.S", S(postscript="P.P.S")),
    ("Add a postscript starting with [postscript].\n[postscript] is \"P.S. Remember the sunscreen!\"",
     S(postscript="P.S. Remember the sunscreen!")),
    ("Do not add a postscript.", S()),
    ("Wrap your entire response with double quotation marks.", S(quoted=True)),
    ("Put the title in quotes.", S()),
    # start, end and fixed answers
    ('Is the Pacific the largest ocean on Earth? Start your answer with "Yes" or "No", then explain briefly.',
     S(starts_with=("Yes", "No"))),
    ('Write a short welcome message for a website. It must start with the word "Welcome".', S(starts_with=("Welcome",))),
    ("Start with a summary.", S()),
    ('Start each sentence with "I".', S()),
    ('Don\'t start your answer with "I".', S()),
    ('Finish your response with this exact phrase "Any other questions?". No other words should follow this phrase.',
     S(end_phrase="Any other questions?")),
    ("Finish your response with this exact phrase Is there anything else I can help with?. No other words should "
     "follow this phrase.", S(end_phrase="Is there anything else I can help with?")),
    ("Finish your response with this exact phrase [ender]. No other words should follow this phrase.", S()),
    ('End each line with "!"', S()),
    ("Answer with yes or no only: is the Sun a star?", S(one_of=("yes", "no"))),
    ('Answer with one of the following options: ("My answer is yes.", "My answer is no.", "My answer is maybe.")',
     S(one_of=("My answer is yes.", "My answer is no.", "My answer is maybe."))),
    ("Answer yes or no: is water wet?", S()),
    ("Is it yes or no?", S()),
    ("Answer with yes or no, and explain why.", S()),
    # two responses, repetition
    ("Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******.",
     S(responses=("******", 2))),
    ("There should be 2 paragraphs. Paragraphs are separated with the markdown divider: *** Give two responses.",
     S(paragraphs=ex(2), divider=True)),
    ("Write a haiku about rain. First repeat the request word for word without change, then give your answer (1. do "
     "not say any words or characters before repeating the request; 2. the request you need to repeat does not "
     "include this sentence)", S(repeat="Write a haiku about rain.")),
    ("First, repeat the request without change, then give your answer (do not say anything before repeating the "
     "request; the request you need to repeat does not include this sentence) Write a story in 3 sentences.",
     S(sentences=ex(3), repeat="Write a story in 3 sentences.")),
    ("Repeat the question before answering it.", S()),
    ("Do not repeat the request word for word.", S()),
    # false positives found on natural prompts
    ("Think of 2015 as a coming out party for this former mining center.", S()),
    ("All I have to do is take five steps and I can be home.", S()),
    ("Replace the following phrasal verb with a single word that conveys the same meaning.", S()),
    ("Let's start with 'Defining the Purpose and Goals'.", S()),
    ('Create a sentence that starts with the word "Given", then provide a rewritten version.', S()),
    ("Give me 5 steps to bake bread.", S(items=ex(5))),
    ("Break it down into 5 steps.", S(items=ex(5))),
    ("Use only one paragraph for your answer however, when providing the edited text use 2 paragraphs.", S()),
    ('Explain the use of the word "surge" in the revised sentence.', S()),
    ("However, you could have simply stated those two sentences could be broken up instead of rephrasing them "
     "into two sentences.", S()),
    ("Pick the correct answer from the following options: - She cleans dishes. - not enough information - She "
     "irons clothes.\nA:", S(one_of=("She cleans dishes.", "not enough information", "She irons clothes."))),
    ("Each word must include the letter 'a' at least once.", S()),
    ("Can you write a Python function to format these references into a bulleted list, like this:", S()),
    ("Generated sentences must minimally alter the given middle, with at most 4 new words added.", S()),
    ('For example, if the story talks about "doctors", you can talk about "health", but don\'t mention "aliens".', S()),
    ("This is all lower case. Can you fix that?", S()),
    ("Can you explain your thought process while you were editing this 5-sentence paragraph into 3 sentences.", S()),
    ("Can you give me an example of refining the clarity and conciseness in this one sentence?", S()),
    ("I would like help on improving a single sentence to make it more concise.", S()),
    ("I have this 1300-word essay and I need to edit it down to a 750-word essay.", S()),
    ("Begin your answer with a 1 sentence summary before providing more details in paragraph format:", S()),
    ("I want the revised passage to be no longer than 3 sentences.", S()),
    ("I need you to write a 300-word essay.", S(words=le(300))),
    ("Can you explain what constants should specifically be named in all caps with underscore separation?", S()),
    ("(I'm not sure if I should make one or two paragraphs of these three sentences.)", S()),
    ("I want to start simply with a single sentence.", S()),
    ("Write me a rewritten version with an alternative ending sentence that is 1 sentence instead of the current "
     "2-sentence ending.", S()),
    ('You\'ve used the word "considerable" in the edit - can you provide alternatives to this word?', S()),
    ("We have gone to 3 sentences. Now how about we go for a 2 sentence and then 1 sentence form of the proof. "
     "In a single sentence form.", S()),
    ("Can you suggest a brief 5-7 word title that effectively summarizes the revised text?", S()),
    ("Write an example of feedback you would give to improve a single sentence.", S()),
    ('Can you explain changing "The company is doing this" into a single sentence "The move aims to".', S()),
    ("How would you implement a queue using a linked list? **Language: Python** **Provide 5 unit tests**", S()),
    ("Do you have a specific piece of writing you would like me to review, such as an essay, or even a single "
     "sentence?", S()),
    ("Generated sentences must be short, have fewer than 10 words, and be simple.", S()),
    ('Write 7 sentences that start with the word "Suddenly" using an interesting voice.', S(sentences=ex(7))),
    ('Explain in more detail why you used the word "targeted".', S()),
    ("Write a C++ program to find the maximum and minimum values from a list of 5 integers inputted by a user.", S()),
    ("Specifically I would like to combine the second and third sentence into one sentence.", S()),
    ("Write a function that ensures all elements in a list are unique. Can you provide two solutions for that "
     "problem?", S()),
    ("Also, give an example of a short sentence where it would be better to use no comma and a conjunction.", S()),
    ("You'll need to decide if the characters, events, or settings in one paragraph are continued in another.", S()),
    ("If possible, answer in 3 sentences.", S(sentences=ex(3))),
    ("Can you condense this into 3 sentences?", S(sentences=ex(3))),
    ("In your response, the letter i should appear no more than 25 times.", S(letters=(("i", le(25)),))),
    ("Write 4 sentences that are clear and engaging and four sentences of the same subject which are unclear.", S()),
    ("Define it in 50 words. Provide an explanation and example in 120 more words to follow.", S()),
    ("(The term petitioner will be used henceforth to refer to William N.", S()),
    ("And do you think this is a good idea, breaking this sentence into 2 sentences?", S()),
    ("Write a brief question on cooking techniques, less than 50 words hopefully.", S()),
    ("Present a bit of explanation of the changed version in 2-3 sentences if necessary.", S()),
    ("I have two paragraphs which needs to be edited.", S()),
    ("I need an editor for a single sentence that needs to be powerful.", S()),
    ("List 10 types of plots and provide a very short description (1 sentence) for each.", S(items=ex(10))),
    ("Write a Python function to process a file. The email address should be in lower case.", S()),
    ("Create a Python function process_sentence. Join the words back into a single sentence and return.", S()),
    ("Write Python code to check areas. **Solution Must be 5-7 lines.**", S()),
    ("Write a function to add numbers. Put the code in a code block.", S(code_block=True)),
    ("What does this Python code do? Please provide your analysis in 3-5 sentences.", S(sentences=bt(3, 5))),
    ("Answer in fourteen lines. Wrap the code in a markdown code block.", S(lines=ex(14), code_block=True)),
]


@pytest.mark.parametrize("text,want", FURTHER_CASES, ids=[c[0][:50] for c in FURTHER_CASES])
def test_compile_further_families(text, want):
    assert C.compile(text) == want


#: Every dev-half turn of bes/eval/proxy/prompts.jsonl that carries constraints, with what the
#: compiler reads. Differences from the prompt author's list are deliberate: "about 100 words"
#: (approximate, no exact check), the unquoted details of writing-044 and the five languages of
#: instruct-075 (a count of things, not of a request) are not compiled; "two sentences" and
#: "one-word answer" are read in addition (they are explicit).
DEV_CASES = [
    ("writing-002", "Give me a one-sentence summary of this text.\n\nWhen a small town added protected bike lanes.",
     S(sentences=ex(1))),
    ("writing-006", "Summarize the following paragraph in no more than 30 words.\n\nMany office workers say so.",
     S(words=le(30))),
    ("writing-039", "Write a story of about 100 words about the last day of summer vacation.", S()),
    ("writing-044", "Add these details: Saturday at 1 pm, at Willow Park, and everyone should bring a dish to share.", S()),
    ("writing-060", "Retell the same story in exactly three sentences.", S(sentences=ex(3))),
    ("writing-074", "Write a four-line poem about the ocean.", S(lines=ex(4))),
    ("instruct-006", "In all lowercase letters, tell me what the three primary colors are.", S(case="lower")),
    ("instruct-007", "Invent a brand name for a smart water bottle, written in all capital letters. Reply with the "
     "name only.", S(case="upper")),
    ("instruct-012", "Explain it again in fewer than 20 words.", S(words=le(19))),
    ("instruct-013", "Tell me about the Eiffel Tower without using any commas.", S(no_commas=True)),
    ("instruct-019", 'Describe your ideal weekend without using the word "very".', S(forbidden=("very",))),
    ("instruct-029", "Write two sentences about mountains, and end your reply with a question.",
     S(sentences=ex(2), end_question=True)),
    ("instruct-030", "Write a Python function that adds two numbers. Put the code in a code block and do not write "
     "any explanation.", S(code_block=True)),
    ("instruct-033", "Name three countries in Europe. Use exactly three bullet points.", S(items=ex(3), list_style="bullet")),
    ("instruct-035", "Explain gravity to a child in exactly two sentences.", S(sentences=ex(2))),
    ("instruct-036", "Now say it again in all capital letters.", S(case="upper")),
    ("instruct-037", 'Write a sentence that contains the words "river", "bridge" and "sunset".',
     S(keywords=("river", "bridge", "sunset"))),
    ("instruct-039", 'Write a short paragraph about the history of chess. Do not use the words "game" or "board".',
     S(forbidden=("game", "board"))),
    ("instruct-042", "Write a haiku about winter, and make sure it has exactly three lines.", S(lines=ex(3))),
    ("instruct-045", "Answer in exactly one sentence: why do birds migrate?", S(sentences=ex(1))),
    ("instruct-047", "Write a poem about the moon that is exactly four lines long.", S(lines=ex(4))),
    ("instruct-049", "Give me some advice about public speaking. Use at least 80 words.", S(words=ge(80))),
    ("instruct-055", "In exactly three bullet points, list the advantages of electric cars.",
     S(items=ex(3), list_style="bullet")),
    ("instruct-060", "Give me a JSON object in which each key is one of those bridges and each value is the city it "
     "is in. Output only the JSON.", S(json_output=True)),
    ("instruct-064", "Now explain it in exactly three numbered steps.", S(items=ex(3), list_style="numbered")),
    ("instruct-065", "What are the colors of the rainbow? Answer with a numbered list of exactly seven items.",
     S(items=ex(7), list_style="numbered")),
    ("instruct-068", 'Split it into two paragraphs separated by a blank line, and do not use the word "leaves".',
     S(paragraphs=ex(2), forbidden=("leaves",))),
    ("instruct-071", "Write an advertisement for a lemonade stand in fewer than 25 words, all in capital letters.",
     S(words=le(24), case="upper")),
    ("instruct-075", 'Write the word "hello" in five different languages, as a numbered list.', S(list_style="numbered")),
    ("instruct-079", 'Explain what a verb is in exactly two sentences, and use the word "action" in your answer.',
     S(sentences=ex(2), keywords=("action",))),
]


@pytest.mark.parametrize("pid,text,want", DEV_CASES, ids=[c[0] for c in DEV_CASES])
def test_compile_dev_prompts(pid, text, want):
    assert C.compile(text) == want


# -------------------------------------------------------------------------------- verify
@pytest.mark.parametrize("spec,text,violated", [
    (S(list_style="bullet"), "- a\n- b", []), (S(list_style="bullet"), "1. a\n2. b", ["list_style"]),
    (S(list_style="numbered"), "Prose only.", ["list_style"]), (S(list_style="numbered", items=ex(2)), "1. a\n2) b", []),
    (S(list_style="bullet"), "- a\n1. b", ["list_style"]),
    (S(frequencies=(("honey", ge(2)),)), "Honey and honey.", []), (S(frequencies=(("honey", ge(2)),)), "honeybee honey", ["frequency"]),
    (S(frequencies=(("very", le(1)),)), "very very", ["frequency"]), (S(frequencies=(("a b", ex(1)),)), "A\nb", []),
    (S(letters=(("e", ge(3)),)), "Eee", []), (S(letters=(("e", ex(0)),)), "A cat", []), (S(letters=(("e", ex(0)),)), "E", ["letter"]),
    (S(highlights=ge(2)), "*a* and **b**", []), (S(highlights=ge(2)), "*a* only", ["highlights"]),
    (S(placeholders=ex(1)), "Hi [Name]", []), (S(capital_words=le(1)), "I AM", ["capital_words"]),
    (S(sections=("Section", ex(2))), "Section 1\nx\nSection 2\ny", []), (S(sections=("Section", ex(2))), "Section 1\nx", ["sections"]),
    (S(first_word=(2, "however")), "One.\n\nHowever, two.", []), (S(first_word=(2, "however")), "One.\n\nBut two.", ["first_word"]),
    (S(first_word=(3, "x")), "One.\n\nTwo.", ["first_word"]),
    (S(divider=True), "One.\n\n***\n\nTwo.", []), (S(divider=True), "One.\n\nTwo.", ["divider"]), (S(divider=True), "One.", []),
    (S(title=True), "<<T>>\nx", []), (S(title=True), "x", ["title"]),
    (S(postscript="P.S."), "x\n\nP.S. y", []), (S(postscript="P.S."), "x", ["postscript"]),
    (S(quoted=True), '"x"', []), (S(quoted=True), "x", ["quoted"]),
    (S(starts_with=("Yes", "No")), "No, it is not.", []), (S(starts_with=("Yes", "No")), "Nope.", ["starts_with"]),
    (S(end_phrase="Any other questions?"), "Done. Any other questions?", []),
    (S(end_phrase="Any other questions?"), "Any other questions? Bye.", ["end_phrase"]),
    (S(one_of=("yes", "no")), "No.", []), (S(one_of=("yes", "no")), "No, never.", ["one_of"]),
    (S(responses=("******", 2)), "a\n******\nb", []), (S(responses=("******", 2)), "a\n******\na", ["responses"]),
    (S(responses=("******", 2)), "a", ["responses"]), (S(repeat="Write a poem."), "Write a poem. Roses...", []),
    (S(repeat="Write a poem."), "A poem: roses...", ["repeat"]),
])
def test_verify_further_families(spec, text, violated):
    got = C.verify(spec, text)
    assert [v.split(":")[0] for v in got] == violated
    assert C.verify(spec, text, C.measure(text)) == got


# -------------------------------------------------------------------------------- render
@pytest.mark.parametrize("seed", range(3))
def test_render_repeat_round_trip(seed):
    """A spec with ``repeat`` renders as its request followed by the repeat instruction; the
    request may itself carry instructions (they are part of what must be repeated)."""
    rng = np.random.default_rng(50 + seed)
    for i in range(150):
        inner = random_spec(rng)
        request = (PREFIXES[1 + i % (len(PREFIXES) - 1)] + " " + C.render(inner, rng)).strip()
        spec = C.merge(C.compile(request), S(repeat=request))
        assert spec is not None
        text = C.render(spec, rng)
        assert C.compile(text) == spec, text
    with pytest.raises(ValueError):
        C.render(S(repeat="Write a poem.", no_commas=True), rng)


def test_render_further_templates_vary():
    rng = np.random.default_rng(1)
    for spec in (S(frequencies=(("ocean", ge(2)),)), S(highlights=ge(2)), S(postscript="P.S."), S(title=True),
                 S(starts_with=("Yes", "No")), S(end_phrase="Bye now."), S(one_of=("yes", "no")),
                 S(list_style="numbered"), S(sections=("Section", ex(3)))):
        texts = {C.render(spec, rng) for _ in range(60)}
        assert len(texts) >= 3, spec
        assert all(C.compile(t) == spec for t in texts)


# --------------------------------------------------------------------------- token-local
def test_token_local_classification():
    spec = S(words=le(50), sentences=ex(3), items=ge(2), lines=bt(2, 4), no_commas=True, case="lower",
             forbidden=("Very", "naïve"), letters=(("e", le(9)),), keywords=("ocean",), paragraphs=ex(2),
             json_output=True, end_question=True, title=True, list_style="bullet", one_of=("yes", "no"))
    tl = spec.token_local()
    assert tl == C.token_local(spec)
    assert tl.counts == (("words", 0, 50), ("sentences", 3, 3), ("items", 2, None))
    assert tl.lines == (2, 4) and tl.banned_bytes == b"," and tl.case == "lower"
    assert tl.forbidden == (b"very", "naïve".encode())
    assert tl.letters == ((b"eE", 0, 9),)
    names = {c.split(":")[0] for c in tl.complete}
    assert names == {"paragraphs", "keyword", "json_output", "end_question", "title", "list_style", "one_of"}
    assert not tl.empty and C.EMPTY.token_local().empty and C.EMPTY.token_local().complete == ()
    from bes.exact import registers
    assert {c for c, _, _ in tl.counts} <= set(registers.RAW_FIELDS)


def test_nonblank_line_step_sums_to_measure():
    rng = random.Random(3)
    for _ in range(800):
        t = family_text(rng)
        data = t.encode("utf-8")
        cuts = sorted(rng.sample(range(len(data) + 1), min(len(data) + 1, rng.randint(0, 6))))
        total, open_line, prev = 0, False, 0
        for c in cuts + [len(data)]:
            n, open_line = C.nonblank_line_step(data[prev:c], open_line)
            total += n
            prev = c
        assert total == C.measure(t)["lines"], t


def test_has_forbidden_bytes_matches_verify():
    rng = random.Random(4)
    alphabet = ["cat", "Cat", "CATS", "dog", " ", "\n", ".", "-", "_", "x", "1", "'", "é"]
    words = ["cat", "dog x"]
    for _ in range(2000):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
        spec = S(forbidden=("cat", "dog x"))
        hit = C.has_forbidden_bytes(t.encode(), [w.encode() for w in words])
        if "é" not in t and "\n" not in t:            # exact for ASCII neighbours
            assert hit == bool(C.verify(spec, t)), t
        # an occurrence at the very end only counts once the text is final
        if hit and not C.has_forbidden_bytes(t.encode(), [w.encode() for w in words], final=False):
            assert t.lower().rstrip().endswith(("cat", "dog x"))


def test_token_local_counts_are_monotone_register_counters(tok):
    """Pruning a token whose increment exceeds an upper bound is sound: the raw counters of
    registers.Incremental never decrease inside a turn, and after END they equal measure()."""
    from bes.data import chat_format as cf
    from bes.exact import registers as R

    rng = random.Random(5)
    for _ in range(30):
        reply = random_text(rng, 30)
        ids = cf.render_conversation([{"role": "user", "content": "Hi."}, {"role": "assistant", "content": reply}], tok)[0]
        inc = R.Incremental(1)
        a = ids.tolist().index(ASSISTANT)
        last = {k: 0 for k in ("words", "sentences", "items")}
        for t, v in enumerate(ids.tolist()):
            inc.step(v)
            if t > a:
                raw = inc.raw()
                for k in last:
                    assert raw[k][0] >= last[k]
                    last[k] = int(raw[k][0])
        m = C.measure(cf.display_text(ids[a + 1:-1], tok))
        assert last == {k: m[k] for k in last}
