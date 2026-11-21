"""Tests of bes.exact.constraints: measured properties, the compiler, verifiers, rendering and
the countdown targets.

Every measured property is checked against a reference written here with different means
(line-by-line string operations, a character scan, ``str`` methods); the transducer-based counts
are checked against ``bes.exact.transducer.measure_counts``, and the line-level fence and list
rules against the transducer's own state. Targets are checked against a brute force that walks
back from every position, and the countdown registers against the counts at every token.
"""
from __future__ import annotations

import random
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
    ("Respond with a minimum of 8 bullet points.", S(items=ge(8))),
    ("Use up to 4 bullet points.", S(items=le(4))),
    ("As many as 4 bullet points.", S(items=le(4))),
    ("Your answer must contain exactly 3 bullet points. Use the markdown bullet points such as: * This is point 1.",
     S(items=ex(3))),
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
     S(words=ge(500), items=ex(4))),
    ("Use at least 6 sentences. Use at most 2 sentences.", S()),
    ("Write 3 sentences or 2 paragraphs.", S()),
    ("Answer in a single sentence, a short paragraph, or a list.", S()),
    # negation
    ("Do not use more than 3 sentences.", S(sentences=le(3))),
    ("Don't write fewer than 4 bullet points.", S(items=ge(4))),
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
    ("There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***", S(paragraphs=ex(2))),
    ("Your response must have 3 paragraphs.", S(paragraphs=ex(3))),
    ("Write a poem with 4 lines.", S(lines=ex(4))),
    ("Write a poem of between 12 and 20 lines about a pirate.", S(lines=bt(12, 20))),
    ("Write three sentences to describe a beach.", S(sentences=ex(3))),
    ("Three bullet points, please.", S(items=ex(3))),
    ("Write a 500-word essay on climate change.", S(words=le(500))),
    ("Describe the ocean in 50 words.", S(words=le(50))),
    ("Answer in one word.", S(words=le(1))),
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
    ('In your response, the word "joy" should appear at least 3 times.', S(keywords=("joy",))),
    ("In your response, the word travel should appear at least 3 times.", S(keywords=("travel",))),
    ('The word "joy" should appear less than 2 times.', S()),
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
    ("Do not include keywords [forbidden_words] in the response. [forbidden_words] are: error, mistake.", S()),
    ("In your response, the letter [e] should appear at least 15 times.", S()),
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
    ("In your response, words with all capital letters should appear at least 2 times.", S()),
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
     S(sentences=ge(5), keywords=("cost", "clean"))),
    ('Provide a response to the following question: "What are the key elements of a plan?" '
     "Your response should contain at least 5 sentences and include a postscript starting with P.S.",
     S(sentences=ge(5))),
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
    return ConstraintSpec(**values)


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
