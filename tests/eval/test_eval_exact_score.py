"""Objective scoring: math extraction, constraint checks, counting definitions, degenerate output."""
from __future__ import annotations

import random
import re
from fractions import Fraction as F

import pytest

from bes.eval import exact_score as E
from bes.eval.proxy.prompts import Prompt


# ---------------------------------------------------------------------------------- math
@pytest.mark.parametrize("reply, ref, ok", [
    ("15% of 80 is 12.", "12", True),
    ("So 15% of 80 is 12. To verify, 12 x 80 = 960, so 15% of 80 is indeed 960.", "12", False),
    ("The answer is $1,234.50.", "1234.5", True),
    ("Total: 12,600 people", "12600", True),
    ("We get \\boxed{\\frac{3}{4}}.", "0.75", True),
    ("Therefore the result is \\boxed{1{,}024}", "1024", True),
    ("\\(\\boxed{\\$75}\\)", "75", True),
    ("It is 11/8, that is 1.375.", "1.375", True),
    ("The new price is **$78**. You save $42.", "78", True),
    ("**Step 1:** 3 x 24 = 72\n**Step 2:** done", "72", True),
    ("She got 86% of the questions right.", "86", True),
    ("She got 0.86 of the questions right.", "86", False),
    ("The temperature is −5 degrees.", "-5", True),
    ("About 42.16 km.", "42.2", True),
    ("Roughly 42 km.", "42.2", False),
    ("It comes to 3276 dollars per month.", "3276", True),
    ("The answer: 40.50", "40.5", True),
    ("The answer is 7, since 3-4 days pass twice.", "7", True),
    ("That would be twenty-five dollars.", "25", True),
    ("One hundred and two people came.", "102", True),
    ("I am not sure.", "12", False),
    ("", "12", False),
    ("Answer: 12 eggs. Explanation: 3 boxes x 4 = 12.", "12", True),
    ("The final answer is 1.75 dollars per notebook.", "1.75", True),
    ("It takes 215 minutes (3 hours and 35 minutes).", "215", False),
    ("It takes 3 hours and 35 minutes, which is 215 minutes.", "215", True),
])
def test_math_correct(reply, ref, ok):
    assert E.math_correct(reply, ref)["correct"] is ok, E.extract_answer(reply)


def test_math_any_match_and_methods():
    r = E.math_correct("15% of 80 is 12. Check: 12 x 80 = 960.", "12")
    assert r["correct"] is False and r["any_match"] is True and r["method"] == "last"
    r = E.math_correct("So 15% of 80 is 12. Check: 12 x 80 = 960.", "12")
    assert r["correct"] is True and r["method"] == "conclusion"
    assert E.extract_answer("\\boxed{5}")["method"] == "boxed"
    assert E.extract_answer("The answer is 5. Then 6.")["method"] == "answer_key"
    assert E.extract_answer("nothing here")["method"] == "none"
    assert E.math_correct("x = 3.333", "3.33")["correct"]
    assert E.math_correct("about 3.3", "3.33", tol=0.05)["correct"]
    assert not E.math_correct("about 3.2", "3.33", tol=0.05)["correct"]


@pytest.mark.parametrize("reply, ref, context, ok", [
    # restating numbers of the question after the answer
    ("So, there are 2400 milliliters in 2.4 liters.", "2400", ["How many milliliters are in 2.4 liters?"], True),
    ("Thus, the worker earns $720 over a 5-day work week.", "720",
     ["A worker earns $18 per hour. How much in an 8-hour day?", "How much is that over a 5-day work week?", "144"], True),
    ("So, it would take 180 minutes to drive 225 km at 75 km/h.", "180",
     ["I drove 150 km in 2 hours. What was my speed?", "How many minutes would a 225 km drive take?", "75"], True),
    ("In total, 3 shirts cost $90.", "90", ["If I buy three shirts at that price, how much do I pay?"], True),
    # an intermediate result after the answer in the same sentence
    ("So the total cost is $43, which includes 180 x 0.10 = 18 for texts.", "43",
     ["A plan costs $25 plus $0.10 per text. What do I pay for 180 texts?"], True),
    # answer first, explanation after
    ("The new price is $78. A 35% discount on $120 is $42, and $120 - $42 = $78.", "78",
     ["A jacket costs $120 and is discounted by 35%. What is the new price?"], True),
    # a model that contradicts itself is judged by its last conclusion
    ("So, 15% of 80 is 12. To verify, 12 x 80 = 960. This confirms that 15% of 80 is indeed 960.", "12",
     ["What is 15% of 80?"], False),
    ("Therefore, an 18% tip is $75.52.", "11.52", ["Our bill is $64. How much is an 18% tip?"], False),
    # the answer may coincide with a number of the question
    ("So 7 times 1 is 7.", "7", ["What is 7 times 1?"], True),
    # a bold label is not an answer; the conclusion is
    ("1. **Multiply the result by 10:** 4 x 10 = 40\n\nTherefore, 2^{10} = 1024.", "1024",
     ["What is 2 to the power of 10?"], True),
    # review eval-8: what a reader takes as the final answer
    # a setup sentence with a concluding word states no new number: the final line answers
    ("So the train travels for 3 hours at 60 km per hour.\n\n60 * 3 = 180\n\nIt goes 180 km.", "180",
     ["A train travels 60 km per hour for 3 hours. How far does it go?"], True),
    # a conclusion followed by the final computation line: the computation's result
    ("Therefore 3 people eat 3 x 2 = 6 slices.\nSlices left: 8 - 6 = 2.", "2",
     ["A pizza is cut into 8 slices and 3 people eat 2 slices each. How many slices are left?"], True),
    # "7!" does not end the sentence
    ("Therefore, 7! = 5040.", "5040", ["What is 7 factorial (7!)?"], True),
    # the stated final answer is 5, so the earlier sentence holding the reference does not count
    ("The discount is 25% of $20, so the discount is $15. The sale price is $5.", "15",
     ["A shirt costs $20 and is 25% off. What is the sale price?"], False),
    ("So Tom has 3 x 8 = 24 apples. After eating 2 he has 24 - 2 = 23 apples left.", "23",
     ["Tom has 3 boxes with 8 apples each. He eats 2. How many apples are left?"], True),
    ("So Tom has 3 x 8 = 24 apples. After eating 2 he has 24 - 2 = 23 apples left.", "24",
     ["Tom has 3 boxes with 8 apples each. He eats 2. How many apples are left?"], False),
    ("So we multiply the cans per pack by the number of packs. 6 x 4 = 24 cans.", "24",
     ["A pack has 6 cans. How many cans are in 4 packs?"], True),
    # a check after the conclusion still does not override it; a conclusion whose result follows "="
    ("So, 15% of 80 is 12. Let's check: 12 / 80 = 0.15, which is 15%.", "12", ["What is 15% of 80?"], True),
    ("So 8 - 6 = 2 slices are left.", "2", ["A pizza has 8 slices; 3 people eat 2 each. How many are left?"], True),
])
def test_math_correct_with_context(reply, ref, context, ok):
    assert E.math_correct(reply, ref, context=context)["correct"] is ok


def test_factorial_does_not_split_a_sentence_and_the_result_follows_the_equals_sign():
    got = E.extract_answer("Therefore, 7! = 5040.")
    assert got["value"] == 5040 and got["method"] == "conclusion"
    assert E.extract_answer("Slices left: 8 - 6 = 2.")["values"] == [2]
    assert E.extract_answer("Wow! It is 3. Then 4!")["value"] == 4          # "!" after a word still ends one
    assert E.extract_answer("So 15% of 80 is 12. Check: 12 x 80 = 960.")["value"] == 12


def test_context_values_reads_digits_and_number_words():
    assert E.context_values(["Buy three shirts at $40, 25% off.", "30"]) >= {F(3), F(40), F(25), F(30)}


def test_find_and_parse_numbers():
    assert [v for v, *_ in E.find_numbers("a 2-step plan")] == [2]
    assert [v for v, *_ in E.find_numbers("3-2 and -3")] == [3, 2, -3]
    assert [v for v, *_ in E.find_numbers("1e3 or 2.5E-1")] == [1000, F(1, 4)]
    assert E.find_numbers("H2O and v2") == []
    assert E.parse_number("1,234.50") == F(246900, 200)
    assert E.parse_number("-3/4") == F(-3, 4)
    assert E.parse_number("25%") == 25
    assert E.parse_number("twelve") == 12
    assert E.parse_number("1 and 2") is None
    assert E.parse_number_words("three thousand and four") == 3004
    assert E.parse_number_words("no numbers") is None


def test_numbers_match_rounding_rules():
    assert E.numbers_match(F(421558, 10000), "42.2")
    assert E.numbers_match(F(4215, 100), "42.2")          # half up
    assert not E.numbers_match(F(4214, 100), "42.2")
    assert not E.numbers_match(F(124, 10), "12")           # integer references are exact
    assert E.numbers_match(F(12), "12.0")
    assert E.numbers_match(F(12), "twelve")              # number words parse in references too
    for bad in ("apples", "1 and 2"):
        with pytest.raises(ValueError):
            E.numbers_match(F(1), bad)


# ----------------------------------------------------------------- counting definitions
def _alnum(ch: str) -> bool:
    return ord(ch) >= 0x80 or (ch.isascii() and ch.isalnum())


def reference_counts(text: str) -> dict[str, int]:
    """Independent implementation of the SPEC 2.1 counts (regex and index based)."""
    words = 0
    for i, ch in enumerate(text):
        if not _alnum(ch):
            continue
        prev = text[i - 1] if i else " "
        apos_after_alnum = prev == "'" and i >= 2 and _alnum(text[i - 2])
        if not _alnum(prev) and not apos_after_alnum:
            words += 1
    fence_spans, inside, pos = [], False, 0
    items = 0
    for line in text.split("\n"):
        lead = len(line) - len(line.lstrip(" \t"))
        is_fence = line[lead:].startswith("```")
        if not inside and not is_fence and re.match(r"[ \t]*(?:[-*+•]|\d+[.)])[ \t]", line):
            items += 1
        end = pos + len(line) + 1
        third = pos + lead + 3  # one past the third backtick
        if is_fence and not inside:
            fence_spans.append((third, end))      # opening: the rest of the line is inside
        elif is_fence and inside:
            fence_spans.append((pos, third))      # closing: inside up to the third backtick
        elif inside:
            fence_spans.append((pos, end))
        if is_fence:
            inside = not inside
        pos = end
    sentences = 0
    for m in re.finditer(r"[.!?]+", text):
        s, e = m.start(), m.end()
        if any(a <= s < b for a, b in fence_spans):
            continue
        line_start = text.rfind("\n", 0, s) + 1
        if re.fullmatch(r"[ \t]*\d+", text[line_start:s]) and text[s] == ".":
            # the dot of a numbered-list marker; the rest of the run may still count
            if e - s == 1:
                continue
            s += 1
        if s == 0 or text[s - 1].isspace():
            continue
        k = e
        while k < len(text) and text[k] in "*_":
            k += 1
        if k == len(text) or text[k].isspace() or text[k] in "\"')]}":
            sentences += 1
    return {"words": words, "sentences": sentences, "lines": text.count("\n"), "items": items}


CORPUS = [
    "Hello world. This is it!",
    "I met Dr. Smith. Done.",
    "- a\n- b\n1. c\n\n```\nx. y. z.\n- no\n```\nEnd.",
    "don't stop 3.5 e-mail 'yes'",
    "**Done.** Next one",
    "He said \"hi.\" Then (he left.) Ok",
    "What?! Really... yes",
    "  1. First step.\n  2) Second step!\n* star item\n+ plus item\n• bullet item",
    "Café au lait coûte 3€. Merci.",
    "A line\nanother line\n\n\nlast",
    "```python\nprint('x.')\n```\nAfter the code. Fine.",
    ". starts with a dot. And ends",
    "Numbers like 2023. Are odd.\n2023. At line start.",
    "",
]


@pytest.mark.parametrize("text", CORPUS)
def test_local_counts_match_independent_reference(text):
    assert E.local_counts(text) == reference_counts(text)


def test_local_counts_on_random_texts():
    rng = random.Random(0)
    alphabet = list("ab .!?-*1\n'\"`)") + ["``` ", "1. ", "- ", "é"]
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert E.local_counts(text) == reference_counts(text), repr(text)


def test_project_counting_backend_agrees_when_present():
    transducer = pytest.importorskip("bes.exact.transducer")
    for text in CORPUS:
        got = transducer.measure_counts(text)
        assert {k: got[k] for k in ("words", "sentences", "lines", "items")} == E.local_counts(text), repr(text)
    assert E.count_backend() in ("constraints.measure", "transducer.measure_counts", "local")


def test_local_counts_fuzz_against_project_transducer():
    transducer = pytest.importorskip("bes.exact.transducer")
    rng = random.Random(5)
    alphabet = list("ab .!?-*1\n'\"`)_(]") + ["``` ", "1. ", "- ", "é", "2) ", "\t", "**"]
    try:
        from bes.exact import constraints as project_constraints
    except ImportError:
        project_constraints = None
    for _ in range(1500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        got = transducer.measure_counts(text)
        local = E.local_counts(text)
        assert {k: got[k] for k in ("words", "sentences", "lines", "items")} == local, repr(text)
        if project_constraints is not None and hasattr(project_constraints, "measure"):
            m = project_constraints.measure(text)
            assert (m["words"], m["sentences"], m["items"]) == (local["words"], local["sentences"], local["items"]), repr(text)


def test_counts_fall_back_to_local(monkeypatch):
    monkeypatch.setitem(E._BACKEND_CACHE, "name", "local")
    monkeypatch.setitem(E._BACKEND_CACHE, "fn", None)
    assert E.count_backend() == "local"
    assert E.counts("One. Two.") == E.local_counts("One. Two.")


# ---------------------------------------------------------------------------- constraints
def ok(reply, **c):
    return E.check_constraint(reply, c)["ok"]


def test_count_constraints():
    assert ok("One. Two. Three.", type="sentences", relation="exactly", n=3)
    assert not ok("One. Two.", type="sentences", relation="exactly", n=3)
    assert ok("a b c", type="words", relation="at_most", n=3)
    assert not ok("a b c d", type="words", relation="at_most", n=3)
    assert ok("a b c d", type="words", relation="at_least", n=4)
    assert ok("a b c d e", type="words", relation="between", n=4, n2=6)
    assert ok("First para.\n\nSecond para.", type="paragraphs", relation="exactly", n=2)
    assert not ok("First para.\nSecond line.", type="paragraphs", relation="exactly", n=2)
    poem = "Roses are red\nViolets are blue\n\nSugar is sweet\nAnd so are you"
    assert ok(poem, type="lines", relation="exactly", n=4)
    bullets = "Here you go:\n- apple\n- kiwi\n* orange\n+ lemon\n• lime"
    assert ok(bullets, type="bullets", relation="exactly", n=5)
    assert ok("1. one\n2) two\n3. three", type="numbered", relation="exactly", n=3)
    assert ok("### Step 1: dig\n**Step 2:** sow\nStep 3: water", type="numbered", relation="exactly", n=3)
    assert ok("Step 1\n1. a\n2. b", type="numbered", relation="exactly", n=2)   # list markers win
    assert ok("1. one\n- two", type="items", relation="exactly", n=2)
    assert ok("```\n- not a bullet\n```\n- real", type="bullets", relation="exactly", n=1)
    assert ok("Section 1\nA\n## Section 2\nB\n**Section 3**\nC", type="sections", relation="exactly", n=3, marker="Section")
    assert ok("one\n***\ntwo\n***\nthree", type="separated_parts", relation="exactly", n=3, marker="***")
    assert ok("Green trees, green grass, GREEN light.", type="frequency", relation="at_least", n=3, word="green")
    assert not ok("evergreen greens", type="frequency", relation="at_least", n=1, word="green")


def test_content_constraints():
    assert ok("The morning aroma filled my cups.", type="include", words=["morning", "aroma", "cup"])
    assert not ok("The aroma of my cup.", type="include", words=["morning", "aroma", "cup"])
    assert ok("Every day is nice.", type="exclude", words=["very"])
    assert not ok("It was Very nice.", type="exclude", words=["very"])
    assert ok("red, yellow and blue.", type="case", value="lower")
    assert not ok("Red and blue.", type="case", value="lower")
    assert ok("AQUAFLOW 3000!", type="case", value="upper")
    assert not ok("123", type="case", value="upper")
    assert ok('```json\n{"name": "Ann", "age": 30, "city": "Oslo"}\n```', type="json_keys", keys=["name", "age", "city"])
    assert not ok('{"name": "Ann"}', type="json_keys", keys=["name", "age"])
    assert ok('["camel", "fox", "lizard"]', type="json")
    assert not ok("Here is the JSON: [1, 2]", type="json")
    assert ok("**Yes**, it is.", type="starts_with", options=["Yes", "No"])
    assert not ok("Well, yes.", type="starts_with", options=["Yes", "No"])
    assert ok("Recycle more. *Every bit helps.*", type="ends_with", text="Every bit helps.")
    assert not ok("Every bit helps. Thanks!", type="ends_with", text="Every bit helps.")
    assert ok("No commas here at all.", type="no_commas")
    assert not ok("Paris, France", type="no_commas")
    assert ok("“Sunny weather, because it is warm.”", type="quoted")
    assert not ok("Sunny weather.", type="quoted")
    assert ok("<<Saving Money Made Simple>>\nText", type="title")
    assert ok("Hi!\n\nP.S. See you soon.", type="postscript", marker="P.S.")
    assert ok("Hi!\n\nPS: see you", type="postscript", marker="P.S.")
    assert not ok("This has no postscript.", type="postscript", marker="P.S.")
    assert ok("Yes.", type="one_of", options=["yes", "no", "maybe"])
    assert not ok("Yes, it is a mammal.", type="one_of", options=["yes", "no", "maybe"])
    assert ok("Meet on [date] at [time] in [room].", type="placeholders", n=3)
    assert ok("What do you think?", type="ends_with_question")
    assert not ok("I think so.", type="ends_with_question")
    assert ok("```python\ndef add(a, b):\n    return a + b\n```", type="code_block")
    assert not ok("```python\ndef add(a, b):", type="code_block")
    assert ok("Keep *going* and **never stop**.", type="highlights", n=2)


def test_validate_constraint_messages():
    assert E.validate_constraint({"type": "words", "relation": "exactly", "n": 3}) == []
    assert E.validate_constraint({"type": "nope"})
    assert E.validate_constraint({"type": "words", "relation": "exactly", "n": 3, "turn": 1}, n_turns=1)
    assert E.validate_constraint({"type": "include", "words": []})
    assert E.validate_constraint({"type": "words", "relation": "between", "n": 5, "n2": 3})
    assert set(E.CONSTRAINT_TYPES) >= {"words", "sentences", "bullets", "json", "case", "include"}


# ------------------------------------------------------------------------ degenerate output
def brute_loop_tail(tokens, max_period=120, min_span=12):
    best = (0, 0)
    n = len(tokens)
    for p in range(1, min(max_period, n // 3) + 1):
        for span in range(n, 0, -1):
            tail = tokens[n - span:]
            if all(tail[i] == tail[i + p] for i in range(span - p)):
                if span >= 3 * p and span >= min_span and span > best[1]:
                    best = (p, span)
                break
    return best


def test_loop_tail_against_brute_force():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(0, 40)
        base = [rng.choice("ab") for _ in range(rng.randint(1, 4))]
        tokens = [rng.choice("abc") for _ in range(n)] + base * rng.randint(0, 8)
        assert E.loop_tail(tokens) == brute_loop_tail(tokens), tokens


def test_degenerate_stats():
    clean = E.degenerate_stats("Paris is the capital of France. It is known for the Eiffel Tower.", "stop")
    assert not clean["degenerate"] and clean["words"] == 13 and clean["truncated"] is False
    loop = E.degenerate_stats("Sure. I like it and then" + " and then" * 30, "length")
    assert loop["loop"] and loop["truncated"] and loop["degenerate"] and loop["rep4"] > 0.5
    assert E.degenerate_stats("", "stop")["empty"]
    assert E.degenerate_stats("...", None)["empty"]
    assert E.degenerate_stats("Here:\n```python\nprint(1)\n", "stop")["unclosed_fence"]
    assert E.degenerate_stats("Done.<|im_end|>\n<|im_start|>user\nmore", "stop")["role_leak"]
    assert E.degenerate_stats("Answer.\nUser: and now?", "stop")["role_leak"]
    lines = "\n".join(["This exact line has five words"] * 3 + ["end"])
    assert E.degenerate_stats(lines, "stop")["repeated_lines"]
    code = "\n".join(["```python", *["for i in range(1, 6):\n    print(i)"] * 3, "```"])
    assert not E.degenerate_stats(code, "stop")["repeated_lines"]
    assert E.degenerate_stats("ok", None)["truncated"] is None


# ------------------------------------------------------------------------------ per prompt
def test_score_conversation_and_summary():
    math = Prompt(id="math-1", category="math", turns=("What is 2+3?", "Double it."), answer=("5", "10"))
    inst = Prompt(id="instruct-1", category="instruct", turns=("Say two sentences.",),
                  constraints=({"type": "sentences", "relation": "exactly", "n": 2},
                               {"type": "no_commas"}))
    s1 = E.score_conversation(math, ["It is 5.", "That makes 10."], ["stop", "stop"])
    assert s1["math_correct"] is True and s1["turns"][0]["math"]["correct"]
    s2 = E.score_conversation(inst, ["One, two. Three."], ["length"])
    assert s2["constraints_ok"] is False and s2["n_constraints"] == 2 and s2["n_constraints_ok"] == 1
    rate = Prompt(id="math-2", category="math", turns=("I drove 150 km in 2 hours. What was my speed in km/h?",
                                                       "At that speed, how many minutes would a 225 km drive take?"),
                  answer=("75", "180"))
    s3 = E.score_conversation(rate, ["Your speed was 75 km/h.", "So, it would take 180 minutes to drive 225 km at 75 km/h."])
    assert [t["math"]["correct"] for t in s3["turns"]] == [True, True]
    summary = E.summarize([s1, s2])
    assert summary["math"]["correct"] == 1.0
    assert summary["constraints"]["prompts_all_ok"] == 0.0
    assert summary["constraints"]["by_type"]["no_commas"]["ok"] == 0.0
    assert summary["degenerate"]["truncated"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["by_category"]["math"]["math_correct"] == 1.0


def test_project_check_optional():
    result = E.project_check("Answer in exactly two sentences.", "One. Two.")
    try:
        import bes.exact.constraints  # noqa: F401
    except ImportError:
        assert result is None
    else:
        assert result is None or isinstance(result["violations"], list)
