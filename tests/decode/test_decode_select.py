"""Satisficing selection (bes.decode.select) on constructed particle sets.

References written here: a word-count F1 and a memoised longest-common-subsequence, a brute
force expected-utility arg-max, a brute force weighted vote over the scorer's own answer
extraction, a repeated-4-gram rate, and sentence / word counts of texts built with known
counts.
"""
from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from fractions import Fraction
from functools import lru_cache

import pytest

from bes.decode import select as S
from bes.decode.select import Band, Candidate
from bes.eval.exact_score import context_values, extract_answer
from bes.exact import constraints as C
from bes.exact.constraints import ConstraintSpec, Interval

WORDS = ("river stone light cloud field music garden winter paper window bridge forest "
         "silver morning ocean candle market letter mountain harbor").split()


def sentence(rng: random.Random, n_words: int) -> str:
    ws = [rng.choice(WORDS) for _ in range(n_words)]
    return " ".join(ws).capitalize() + "."


def text_with(n_sent: int, words_per: int = 8, seed: int = 0) -> str:
    rng = random.Random(seed)
    return " ".join(sentence(rng, words_per) for _ in range(n_sent))


# ------------------------------------------------------------------ independent references
def ref_f1(a: str, b: str) -> float:
    wa, wb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    ca, cb = Counter(wa), Counter(wb)
    overlap = sum(min(ca[w], cb[w]) for w in ca)
    p, r = overlap / len(wa), overlap / len(wb)
    return 0.0 if overlap == 0 else 2 * p * r / (p + r)


def ref_lcs(a: str, b: str) -> float:
    wa, wb = tuple(re.findall(r"\w+", a.lower())), tuple(re.findall(r"\w+", b.lower()))

    @lru_cache(maxsize=None)
    def lcs(i: int, j: int) -> int:
        if i == len(wa) or j == len(wb):
            return 0
        if wa[i] == wb[j]:
            return 1 + lcs(i + 1, j + 1)
        return max(lcs(i + 1, j), lcs(i, j + 1))

    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    n = lcs(0, 0)
    return 0.0 if n == 0 else 2 * n / (len(wa) + len(wb))


def ref_rep4(text: str) -> float:
    w = re.findall(r"\w+", text.lower())
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams) if grams else 0.0


def ref_mbr(texts, weights, utility) -> int:
    tot = sum(weights)
    w = [x / tot for x in weights] if tot > 0 else [1 / len(texts)] * len(texts)
    scores = [sum(w[j] * (1.0 if i == j else utility(texts[i], texts[j])) for j in range(len(texts)))
              for i in range(len(texts))]
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if abs(s - best) <= 1e-12]
    return min(tied, key=lambda i: (-w[i], i))


# ----------------------------------------------------------------------------- utilities
def test_utilities_match_independent_references():
    rng = random.Random(1)
    for _ in range(200):
        a = " ".join(rng.choice(WORDS[:8]) for _ in range(rng.randint(0, 12)))
        b = " ".join(rng.choice(WORDS[:8]) for _ in range(rng.randint(0, 12)))
        assert S.utility(a, b, "f1") == pytest.approx(ref_f1(a, b), abs=1e-12)
        assert S.utility(a, b, "lcs") == pytest.approx(ref_lcs(a, b), abs=1e-12)
        bigram = lambda t: " ".join("_".join(p) for p in zip(re.findall(r"\w+", t.lower()),
                                                              re.findall(r"\w+", t.lower())[1:]))
        wa, wb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
        f2 = 1.0 if len(wa) < 2 and len(wb) < 2 else ref_f1(bigram(a), bigram(b))
        assert S.utility(a, b, "f12") == pytest.approx(0.5 * (ref_f1(a, b) + f2), abs=1e-12)
        for kind in S.UTILITIES:
            assert S.utility(a, b, kind) == pytest.approx(S.utility(b, a, kind), abs=1e-15)
            assert 0.0 <= S.utility(a, b, kind) <= 1.0
    for kind in S.UTILITIES:
        assert S.utility("The river runs.", "the RIVER runs", kind) == 1.0
        assert S.utility("river stone", "cloud field", kind) == 0.0
    with pytest.raises(ValueError):
        S.utility("a", "b", "bleu")


# ----------------------------------------------------------------------------- consensus
def test_consensus_is_the_weighted_expected_utility_argmax():
    rng = random.Random(2)
    for trial in range(60):
        n = rng.randint(1, 6)
        texts = [" ".join(rng.choice(WORDS[:6]) for _ in range(rng.randint(1, 8))) for _ in range(n)]
        weights = [rng.choice([0.0, 0.1, 0.5, 1.0, 2.0]) for _ in range(n)]
        for kind, ref in (("f1", ref_f1), ("lcs", ref_lcs)):
            pos, scores = S.consensus(texts, weights, kind)
            assert pos == ref_mbr(texts, weights, ref), (texts, weights, kind)


def test_consensus_picks_the_cluster_and_respects_weights():
    cluster = ["The bridge over the river is made of old grey stone.",
               "The bridge over the river is built of old grey stone.",
               "This bridge over the river is made of grey stone."]
    outlier = "Paper candles glow in the winter market at night."
    pos, _ = S.consensus(cluster + [outlier], [1, 1, 1, 1])
    assert pos in (0, 1, 2)
    # the self term: a heavy outlier wins over a light cluster
    pos, _ = S.consensus(cluster + [outlier], [0.03, 0.03, 0.03, 0.91])
    assert pos == 3
    # two survivors: the heavier one wins (without the self term the lighter one would)
    a, b = "Stone and light.", "Cloud and paper."
    assert S.consensus([a, b], [0.8, 0.2])[0] == 0
    assert S.consensus([a, b], [0.2, 0.8])[0] == 1


def test_ties_are_deterministic():
    same = ["Garden music in the morning."] * 4
    assert S.consensus(same, [1, 1, 1, 1])[0] == 0
    assert S.consensus(same, [1, 3, 3, 1])[0] == 1           # larger weight wins a tie
    cands = [Candidate("Silver ocean light, a quiet harbor.", 0.25) for _ in range(4)]
    first = S.select(cands, None, "Describe the sea.")
    again = S.select(cands, None, "Describe the sea.")
    assert first.index == again.index == 0 and first.as_dict() == again.as_dict()


def test_selection_does_not_depend_on_candidate_order():
    rng = random.Random(3)
    texts = [text_with(2, 9, seed=s) for s in range(5)]
    weights = [0.1, 0.3, 0.2, 0.25, 0.15]
    base = S.select([Candidate(t, w) for t, w in zip(texts, weights)], None, "Tell me a story.",
                    band=None)
    for _ in range(10):
        order = list(range(5))
        rng.shuffle(order)
        got = S.select([Candidate(texts[i], weights[i]) for i in order], None, "Tell me a story.",
                       band=None)
        assert got.text == base.text


# ------------------------------------------------------------------------- failure checks
def test_unfinished_and_dead_particles_are_dropped():
    good = text_with(2)
    cands = [Candidate(text_with(2, seed=5), 0.0, "length", 40), Candidate(good, 0.6),
             Candidate(text_with(2, seed=6), 0.0, "dead", 12)]
    sel = S.select(cands, None, "Tell me about rivers.", band=None)
    assert sel.index == 1 and sel.rule == "strict"
    assert "unfinished" in sel.reasons[0] and "unfinished" in sel.reasons[2]
    assert sel.reasons[1] == []


def test_violations_are_checked_on_the_complete_text():
    spec = ConstraintSpec(sentences=Interval.exactly(2), keywords=("harbor",))
    texts = [text_with(1, seed=1) + " A harbor.", text_with(3, seed=2) + " The harbor.",
             text_with(1, seed=3) + " Near the harbor.", text_with(2, seed=4)]
    # the reference: count sentences and keywords independently
    def ok(t):
        n_sent = len(re.findall(r"[.!?](?=\s|$)", t))
        return n_sent == 2 and re.search(r"(?<![A-Za-z0-9])harbor", t, re.I) is not None
    assert [ok(t) for t in texts] == [True, False, True, False]
    sel = S.select([Candidate(t, 0.25) for t in texts], spec, "Two sentences, mention harbor.")
    assert sel.index in (0, 2) and not C.verify(spec, sel.text)
    assert any(r.startswith("violation") for r in sel.reasons[1])
    assert any(r.startswith("violation") for r in sel.reasons[3])


@pytest.mark.parametrize("text,flag", [
    ("", "empty"),
    ("...", "empty"),
    ("Let me tell you. " + "and then we go " * 12, "loop"),
    ("Intro line here.\n" + "this exact line has six words\n" * 3 + "End.", "repeated_lines"),
    ("Here is code:\n```python\nx = 1\n", "unclosed_fence"),
    ("Sure, here it is.\nUser: what about tomorrow?", "role_leak"),
    ("The answer is near <|calc|> here.", "special_leak"),
    ("This reply stops in the middle of a long and winding", "cut_off"),
])
def test_each_degenerate_detector_fires(text, flag):
    assert flag in S.degenerate_flags(text), S.degenerate_flags(text)
    ok = text_with(3, seed=9)
    assert S.degenerate_flags(ok) == []


def test_repeated_four_gram_rate_matches_a_reference():
    rng = random.Random(4)
    for _ in range(50):
        base = [rng.choice(WORDS[:5]) for _ in range(rng.randint(4, 40))]
        text = " ".join(base) + "."
        rep = ref_rep4(text)
        flags = S.degenerate_flags(text, rep4_max=0.3)
        assert ("rep4" in flags) == (rep > 0.3) or "loop" in flags or "repeated_lines" in flags
    code_like = "x = x + 1\n" * 2 + "y = y + 1 and x = x + 1 and x = x + 1"
    assert "rep4" not in S.degenerate_flags(code_like, code=True)


def test_cut_off_ignores_short_lines_lists_and_closed_endings():
    assert not S.cut_off("Thanks for your help.\n\nBest regards,\nSarah")
    assert not S.cut_off("Steps:\n- mix the flour and the sugar with care")
    assert not S.cut_off("He said it was done.")
    assert not S.cut_off("Is it raining in the mountains today?")
    assert not S.cut_off("Line one of the poem\nLine two of the poem")   # short-form: not prose
    assert S.cut_off("The river carries the stone towards the sea and the")
    # bold markup is not a list marker (a marker needs a following space)
    assert S.cut_off("Intro.\n**Energy**: renewable energy is concerned with resources")
    assert not S.cut_off("Intro.\n**Energy**: renewable energy is concerned with resources.")
    assert not S.cut_off("Intro.\n* renewable energy is concerned with many resources")
    assert not S.cut_off("Intro.\n### 4. **Scientific")                  # a heading
    assert "cut_off" not in S.degenerate_flags("a b c d e f g", prose=False)


def test_a_particle_that_used_its_whole_budget_is_degenerate():
    q = "Why do leaves change color?"
    natural = Candidate(text_with(12, 10, seed=1), 0.4, "stop", n_tokens=180)
    forced = Candidate(text_with(12, 10, seed=2), 0.6, "stop", n_tokens=320)
    sel = S.select([forced, natural], None, q, budget=320)
    assert sel.index == 1 and "degenerate: at_budget" in sel.reasons[0]
    assert S.select([forced, natural], None, q).index == 0          # without a budget: weight
    only = S.select([forced], None, q, budget=320)
    assert only.rule == "relax_degenerate" and only.index == 0
    # unfinished particles are not "at budget" (they are unfinished)
    cut = S.check(Candidate("A b c.", 0.0, "length", n_tokens=320), budget=320)
    assert "at_budget" not in cut.degenerate and not cut.finished


# --------------------------------------------------------------------------- length band
@pytest.mark.parametrize("prompt,history,kind", [
    ("Who wrote the play \"Romeo and Juliet\"?", (), "factual"),
    ("What is the capital of Australia?", (), "factual"),
    ("Is a tomato a fruit?", (), "factual"),
    ("Why is the sky blue?", (), "open"),
    ("What is the difference between a cold and the flu?", (), "open"),
    ("Name a few famous inventors and what they invented.", (), "open"),
    ("Write a short story about a girl and her kite.", (), "writing"),
    ("Now write a negative review of the same movie.", (), "writing"),
    ("Write a haiku about autumn rain.", (), "short"),
    ("Invent a brand name. Reply with the name only.", (), "short"),
    ("Summarize this paragraph in one sentence.\n\nIt cost 5 dollars and 3 cents per day.", (),
     "transform"),
    ("Make it shorter and more casual.", (), "transform"),
    ("Write a Python function that reverses a string.", (), "code"),
    ("Change it so that the names are sorted by length.",
     ("Write a Python function that sorts a list of names.",), "code"),
    ("What is 389 + 476 - 158?", (), "math"),
    ("A shirt costs $40 and is on sale for 25% off. What is the sale price?", (), "math"),
    ("Convert 25 degrees Celsius to degrees Fahrenheit.", (), "math"),
    ("And what is its area in square centimeters?",
     ("A square has sides of 9 cm. What is its perimeter in centimeters?",), "math"),
    ("Which tip works best for someone who only has 15 minutes a day?", (), "factual"),
    ("Write an email offering a 10% discount on the next order.", (), "writing"),
])
def test_prompt_types(prompt, history, kind):
    assert S.prompt_type(prompt, history) == kind


def test_bands_by_type_and_modifiers():
    assert S.length_band("Who painted the Mona Lisa?") == Band("factual", (1, 90), (1, 3))
    assert S.length_band("Why do leaves change color?") == Band("open", S.WORD_BANDS["default"])
    assert S.length_band("Briefly, why do leaves change color?").words == S.WORD_BANDS["brief"]
    assert S.length_band("Write a very short story about rain.").words == S.WORD_BANDS["very_brief"]
    assert S.length_band("Explain in detail how vaccines work.").words == S.WORD_BANDS["long"]
    for none in ("Write a Python function that adds two numbers.", "What is 2 + 2?",
                 "Rewrite this sentence in the passive voice: \"We won.\"", "Write a limerick."):
        assert S.length_band(none) is None, none
    # explicit constraints always override the band
    for spec in (ConstraintSpec(words=Interval.at_most(30)), ConstraintSpec(items=Interval(3, 3)),
                 ConstraintSpec(json_output=True), ConstraintSpec(paragraphs=Interval(2, 2))):
        assert S.length_band("Why do leaves change color?", spec) is None
    assert S.length_band("Why do leaves change color?", ConstraintSpec(keywords=("red",))) is not None


def test_band_membership_counts_words_and_sentences():
    factual = Band("factual", (1, 90), (1, 3))
    assert factual.contains(text_with(2))
    assert not factual.contains(text_with(5))
    assert factual.contains("Canberra")                       # no terminator: one sentence
    assert not factual.contains(text_with(2, words_per=60))    # 120 words
    open_band = Band("open", (80, 200))
    assert open_band.contains(text_with(12, 10))              # 120 words
    assert not open_band.contains(text_with(3, 10))           # 30 words
    assert not open_band.contains(text_with(30, 10))          # 300 words


def test_band_is_a_filter_and_relaxed_when_nothing_fits():
    q = "What is the capital of France?"
    short, long_ = "The capital of France is Paris.", text_with(6, seed=3)
    sel = S.select([Candidate(long_, 0.7), Candidate(short, 0.3)], None, q)
    assert sel.text == short and sel.rule == "strict" and "band" in sel.reasons[0]
    sel = S.select([Candidate(long_, 0.5), Candidate(text_with(7, seed=4), 0.5)], None, q)
    assert sel.rule == "relax_band"


# ----------------------------------------------------------------------------- fallbacks
def test_fallback_order_and_reported_rule():
    q = "Tell me about the harbor."
    spec = ConstraintSpec(sentences=Interval.exactly(2))
    # all degenerate but satisfying the spec -> relax_degenerate
    rep = "harbor lights shine on the water\n" * 3
    loops = [Candidate(rep + "The harbor is calm. The end.", 0.5),
             Candidate(rep + "The boats are home. Good night.", 0.5)]
    assert all("repeated_lines" in S.degenerate_flags(c.text) for c in loops)
    assert all(not C.verify(spec, c.text) for c in loops)
    sel = S.select(loops, spec, q)
    assert sel.rule == "relax_degenerate" and not C.verify(spec, sel.text)
    # every finished particle violates: fewest violations, then weight
    spec2 = ConstraintSpec(sentences=Interval.exactly(2), keywords=("zebra",))
    viol = [Candidate(text_with(1, seed=1), 0.9), Candidate(text_with(2, seed=2), 0.1),
            Candidate(text_with(3, seed=3), 0.0, "length")]
    sel = S.select(viol, spec2, q)
    assert sel.rule == "best_finished" and sel.index == 1      # one violation beats two
    # nothing finished: "length" before "dead", then fewer violations, then more tokens
    none = [Candidate(text_with(1, seed=1), 0.0, "dead", 30),
            Candidate(text_with(2, seed=2), 0.0, "length", 10),
            Candidate(text_with(2, seed=3), 0.0, "length", 20)]
    sel = S.select(none, spec, q)
    assert sel.rule == "unfinished" and sel.index == 2
    assert S.select([Candidate("", 0.0, "dead")], None, q).rule == "unfinished"
    with pytest.raises(ValueError):
        S.select([], None, q)


# ---------------------------------------------------------------------------------- math
def test_math_majority_uses_the_scorers_extraction_and_weights():
    q = "Sam has 3 boxes with 24 pencils in each box. How many pencils does he have?"
    replies = ["3 boxes times 24 pencils is 72 pencils. So the answer is 72.",
               "Each box has 24 pencils, so 3 * 24 = 72. The total is 72.",
               "He has 3 boxes and 24 pencils, so he has 27 pencils in total."]
    ctx = context_values([q])
    vals = [extract_answer(r, ctx)["value"] for r in replies]
    assert vals == [Fraction(72), Fraction(72), Fraction(27)]      # the reference rule
    sel = S.select([Candidate(r, 1 / 3) for r in replies], None, q)
    assert sel.answer == "72" and sel.index in (0, 1) and sel.prompt_type == "math"
    sel = S.select([Candidate(r, w) for r, w in zip(replies, (0.2, 0.2, 0.6))], None, q)
    assert sel.answer == "27" and sel.index == 2                    # weighted vote
    # brute-force weighted vote on random sets
    rng = random.Random(5)
    for _ in range(40):
        ans = [rng.choice([12, 15, 18]) for _ in range(rng.randint(1, 6))]
        ws = [rng.choice([0.1, 0.2, 0.3]) for _ in ans]
        texts = [f"First we add the numbers. The answer is {a}." for a in ans]
        tot: dict[int, float] = {}
        for a, w in zip(ans, ws):
            tot[a] = tot.get(a, 0.0) + w
        cnt = Counter(ans)
        best = max(tot, key=lambda a: (round(tot[a], 12), cnt[a], -ans.index(a)))
        sel = S.select([Candidate(t, w) for t, w in zip(texts, ws)], None, "What is 7 + 8 - 3?")
        assert sel.answer == str(best)


# ------------------------------------------------------------------------- the guarantee
SPECS = [ConstraintSpec(sentences=Interval.exactly(2)), ConstraintSpec(words=Interval.at_most(12)),
         ConstraintSpec(keywords=("harbor",)), ConstraintSpec(sentences=Interval(1, 2),
                                                               keywords=("stone",)),
         ConstraintSpec(end_question=True), ConstraintSpec(no_commas=True)]
POOL = [text_with(n, w, seed=s) for n in (1, 2, 3) for w in (4, 9) for s in range(2)] + [
    "Is the harbor open today?", "A stone, a harbor, a light.", "The harbor stone is old. It is wet.",
    "Stone walls stand. They are tall and grey and old and cold and wet.", "Harbor."]


def test_never_violates_a_constraint_that_a_finished_particle_satisfied():
    rng = random.Random(6)
    for trial in range(300):
        spec = rng.choice(SPECS)
        k = rng.randint(1, 6)
        cands = [Candidate(rng.choice(POOL), rng.random(), rng.choice(["stop", "stop", "length",
                                                                       "dead"]))
                 for _ in range(k)]
        sel = S.select(cands, spec, rng.choice(["Describe the harbor.", "Who built the harbor?",
                                                 "Write a story about stone."]))
        finished = [c for c in cands if c.finished]
        n_viol = [len(C.verify(spec, c.text)) for c in finished]
        if finished and min(n_viol) == 0:
            assert not C.verify(spec, sel.text), (spec, sel.text, sel.rule)
            assert cands[sel.index].finished
        elif finished:
            assert len(C.verify(spec, sel.text)) == min(n_viol)
        else:
            assert sel.rule == "unfinished"


def test_selection_report_is_serialisable():
    import json

    sel = S.select([Candidate(text_with(2), 0.5), Candidate("", 0.5)], ConstraintSpec(
        sentences=Interval.exactly(2)), "Tell me two things.")
    d = sel.as_dict()
    json.dumps(d)
    assert d["rule"] == "strict" and d["n_survivors"] == 1 and d["index"] == 0
