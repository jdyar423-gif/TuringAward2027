"""The proxy prompt set: composition, schema, answers, constraints and the dev / held-out split."""
from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction as F

import pytest

from bes.eval.exact_score import parse_number, validate_constraint
from bes.eval.proxy import prompts as P

# Independent recomputation of every math answer, keyed by the first user turn (two-turn items
# give one value per turn). Written separately from the prompt file on purpose.
MATH_REFERENCE = {
    "What is 47 times 23?": 47 * 23,
    "Compute 1,250 divided by 25.": F(1250, 25),
    "What is 389 + 476 - 158?": 389 + 476 - 158,
    "What is 12 squared minus 5 cubed?": 12 ** 2 - 5 ** 3,
    "What is the sum of all the whole numbers from 1 to 20?": sum(range(21)),
    "What is 3/4 + 5/8? Give the answer as a decimal.": F(3, 4) + F(5, 8),
    "What is 0.6 multiplied by 0.15?": F(6, 10) * F(15, 100),
    "What is the average of 14, 22, 9 and 35?": F(14 + 22 + 9 + 35, 4),
    "What is 2 to the power of 10?": 2 ** 10,
    "What is 7 factorial (7!)?": 7 * 6 * 5 * 4 * 3 * 2,
    "What is 15% of 240?": F(15 * 240, 100),
    "A price went up from $80 to $92. By what percentage did it increase?": F(100 * 12, 80),
    "What percentage of 32 is 8?": F(800, 32),
    "A jacket costs $120 and is discounted by 35%. What is the new price in dollars?": 120 - F(35 * 120, 100),
    "If 40% of a number is 18, what is the number?": F(18 * 100, 40),
    "A town of 12,000 people grows by 5% in one year. What is the new population?": 12000 + 600,
    "Our restaurant bill is $64. How much is an 18% tip, in dollars?": F(64 * 18, 100),
    "A test has 50 questions and Maya got 43 of them right. What percentage did she get right?": F(4300, 50),
    "The price of a laptop fell from $900 to $765. What was the percentage decrease?": F(100 * 135, 900),
    "A monthly salary of $3,000 is raised by 4%, and a year later by another 5%. What is the final monthly salary in dollars?": F(3000 * 104 * 105, 100 * 100),
    "How many seconds are there in 3.5 hours?": F(7, 2) * 60 * 60,
    "Convert 25 degrees Celsius to degrees Fahrenheit.": F(25 * 9, 5) + 32,
    "How many centimeters are there in 4 feet? Use 1 inch = 2.54 cm.": 48 * F(254, 100),
    "A recipe needs 750 grams of flour. How many kilograms is that?": F(3, 4),
    "Convert 98.6 degrees Fahrenheit to degrees Celsius.": (F(986, 10) - 32) * F(5, 9),
    "How many minutes are there in one week?": 60 * 24 * 7,
    "A car travels at 90 kilometers per hour. How many meters does it travel each second?": F(90 * 1000, 60 * 60),
    "How many milliliters are in 2.4 liters?": 2400,
    "If 1 mile is 1.609 kilometers, how many kilometers is 26.2 miles? Round to one decimal place.": F(422, 10),
    "A room is 5 meters long and 4 meters wide. What is its floor area in square centimeters?": 500 * 400,
    "Sam has 3 boxes with 24 pencils in each box. How many pencils does he have in total?": 72,
    "A train leaves at 9:40 and arrives at 13:15 on the same day. How many minutes does the trip take?": 20 + 3 * 60 + 15,
    "Concert tickets cost $12.50 each. How much do 6 tickets cost, in dollars?": 75,
    "A garden is 15 meters long and 8 meters wide. What is its perimeter in meters?": 15 + 8 + 15 + 8,
    "Five notebooks cost $8.75 in total. How much does one notebook cost, in dollars?": F(875, 500),
    "A water tank holds 1,200 liters when full and is 3/8 full right now. How many liters of water are in it?": 150 * 3,
    "Lena reads 35 pages every day. How many pages does she read in two weeks?": 35 * 7 * 2,
    "A cyclist rides 18 km in 45 minutes. What is her average speed in kilometers per hour?": F(18 * 60, 45),
    "A bakery makes 240 muffins. It sells 3/5 of them in the morning and 40 more in the afternoon. How many muffins are left at the end of the day?": 240 - 144 - 40,
    "Ana buys 4 shirts at $15 each and 2 pairs of jeans at $38 each. She pays with a $200 bill. How much change does she get?": 200 - 60 - 76,
    "A phone plan costs $25 per month plus $0.10 per text message. How much do I pay for a month in which I send 180 texts?": 25 + 18,
    "A rectangle has a perimeter of 36 cm and a length of 11 cm. What is its area in square centimeters?": 11 * 7,
    "Three friends split a $96 dinner bill equally, and each of them also leaves a $4 tip. How much does each friend pay in total?": 32 + 4,
    "A car uses 6 liters of fuel per 100 km, and fuel costs $1.50 per liter. How much does the fuel cost for a 450 km trip, in dollars?": F(27 * 3, 2),
    "A school has 18 classes with 28 students each. If 3 out of every 7 students walk to school, how many students walk to school?": 18 * 4 * 3,
    # two-turn items: (first turn answer, second turn answer)
    "A shirt costs $40 and is on sale for 25% off. What is the sale price?": (30, 90),
    "What is 18 multiplied by 7?": (126, 42),
    "A rectangle is 12 meters long and 5 meters wide. What is its area in square meters?": (60, 240),
    "I drove 150 km in 2 hours. What was my average speed in km/h?": (75, 180),
    "Convert 5 kilometers to meters.": (5000, F(25, 2)),
    "A recipe for 4 people uses 300 grams of rice. How much rice is that per person?": (75, 750),
    "What is 20% of 150?": (30, 6),
    "I have $250 and spend $85 on groceries. How much money is left?": (165, 110),
    "A box holds 12 eggs. How many eggs are in 7 boxes?": (84, F(893, 10)),
    "The temperature is 20 degrees Celsius. What is that in degrees Fahrenheit?": (68, 86),
    "A book has 320 pages and I have read 40% of it. How many pages have I read?": (128, 8),
    "What is the sum of 345 and 678?": (1023, 341),
    "A worker earns $18 per hour. How much does she earn in an 8-hour day?": (144, 720),
    "A square has sides of 9 cm. What is its perimeter in centimeters?": (36, 81),
    "A bag has 5 red, 3 blue and 2 green marbles. What fraction of the marbles are blue? Give the answer as a decimal.": (F(3, 10), 50),
}

_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve"]


@pytest.fixture(scope="module")
def prompts():
    return P.load()


def test_composition(prompts):
    counts = Counter(p.category for p in prompts)
    assert dict(counts) == P.TARGET_COUNTS
    assert len(prompts) == 400
    multi = sum(p.multi_turn for p in prompts)
    assert 90 <= multi <= 110  # about a quarter
    for cat in P.CATEGORIES:
        share = sum(p.multi_turn for p in prompts if p.category == cat) / counts[cat]
        assert 0.15 <= share <= 0.35, cat


def test_ids_unique_and_named(prompts):
    ids = [p.id for p in prompts]
    assert len(set(ids)) == len(ids)
    for p in prompts:
        assert re.fullmatch(rf"{p.category}-\d{{3}}", p.id)
    firsts = [p.turns[0] for p in prompts]
    assert len(set(firsts)) == len(firsts), "two prompts share the same first turn"


def test_validate_reports_problems():
    good = {"id": "math-900", "category": "math", "turns": ["1+1?"], "answer": "2", "constraints": [], "notes": ""}
    assert P.validate([good]) == []
    bad = [dict(good, answer="two apples"), dict(good, id="x", category="poetry"),
           dict(good, id="m2", turns=[]), dict(good, id="m3", answer=None),
           dict(good, id="m4", constraints=[{"type": "words", "relation": "about", "n": 3}]),
           {"id": "i1", "category": "instruct", "turns": ["Say hi."], "constraints": []}]
    problems = P.validate(bad)
    assert len(problems) >= 6


def test_math_answers_independent(prompts):
    math = [p for p in prompts if p.category == "math"]
    assert len(math) == len(MATH_REFERENCE)
    for p in math:
        want = MATH_REFERENCE[p.turns[0]]
        if p.multi_turn:
            assert isinstance(p.answer, tuple) and len(p.answer) == 2
            assert [parse_number(a) for a in p.answer] == [F(w) for w in want], p.id
        else:
            assert parse_number(p.answer) == F(want), p.id
        assert p.answer_for_turn(len(p.turns) - 1) is not None


def test_constraints_well_formed_and_stated(prompts):
    for p in prompts:
        for c in p.constraints:
            assert validate_constraint(c, len(p.turns)) == [], (p.id, c)
            text = p.turns[int(c.get("turn", len(p.turns) - 1))].lower()
            if c["type"] in ("include", "exclude"):
                for w in c["words"]:
                    assert w.lower() in text, (p.id, w)
            if "relation" in c and c["relation"] != "between" and c.get("n", 0) > 2:
                n = c["n"]
                spelled = {str(n), _NUMBER_WORDS[n] if n < len(_NUMBER_WORDS) else str(n)}
                if c["relation"] == "at_most":
                    spelled |= {str(n + 1)}
                assert any(re.search(rf"\b{s}\b", text) for s in spelled), (p.id, c)
    assert all(p.constraints for p in prompts if p.category == "instruct")


def test_split_balanced_and_deterministic(prompts):
    for cat in P.CATEGORIES:
        for turns in (1, 2):
            stratum = [p for p in prompts if p.category == cat and len(p.turns) == turns]
            dev = sum(p.split == "dev" for p in stratum)
            assert abs(2 * dev - len(stratum)) <= 1
        in_cat = [p for p in prompts if p.category == cat]
        assert abs(2 * sum(p.split == "dev" for p in in_cat) - len(in_cat)) <= 2
    assert abs(sum(p.split == "dev" for p in prompts) - 200) <= 5
    again = P.load()
    assert [p.split for p in again] == [p.split for p in prompts]


def test_split_stable_under_additions_elsewhere():
    records = P.load_records()
    before = P.assign_splits(records)
    extra = [{"id": f"math-9{i:02d}", "category": "math", "turns": ["a", "b"], "answer": ["1", "2"]} for i in range(3)]
    after = P.assign_splits(records + extra)
    changed = [pid for pid in before if before[pid] != after[pid]]
    assert all(pid.startswith("math-") for pid in changed)
    multi_math = {r["id"] for r in records if r["category"] == "math" and len(r["turns"]) == 2}
    assert set(changed) <= multi_math


def test_every_prefix_of_a_split_covers_every_category_in_proportion(prompts):
    """A run that stops at a deadline answers a prefix of `select` (review eval-6: in file order
    the first 86 held-out prompts were chat, math and code only). Every prefix holds each
    category, and inside it each number of turns, within one prompt of its share."""
    for split in ("dev", "test", "all"):
        order = P.select(prompts, split)
        assert sorted(p.id for p in order) == sorted(p.id for p in prompts if split == "all" or p.split == split)
        assert [p.id for p in P.select(prompts, split)] == [p.id for p in order]          # deterministic
        size = Counter(p.category for p in order)
        turns = Counter((p.category, len(p.turns)) for p in order)
        seen, seen_turns = Counter(), Counter()
        for n, p in enumerate(order, 1):
            seen[p.category] += 1
            seen_turns[(p.category, len(p.turns))] += 1
            for c in size:
                assert abs(seen[c] - n * size[c] / len(order)) < 1, (split, n, c)
            m = seen[p.category]
            for (c, t), k in turns.items():
                if c == p.category:
                    assert abs(seen_turns[(c, t)] - m * k / size[c]) < 1, (split, n, c, t)
        for limit in (5, 17, 60):
            assert [p.id for p in P.select(prompts, split, limit=limit)] == [p.id for p in order[:limit]]
    head = P.select(prompts, "test", limit=10)
    assert {p.category for p in head} == set(P.CATEGORIES)


def test_interleave_is_proportional_for_any_sizes():
    import itertools

    for k in range(1, 4):
        for sizes in itertools.product(range(7), repeat=k):
            groups = [[(g, i) for i in range(s)] for g, s in enumerate(sizes)]
            out = P.interleave(groups)
            assert sorted(out) == sorted(x for g in groups for x in g)
            taken = [0] * k
            for n, (g, i) in enumerate(out, 1):
                assert i == taken[g]                       # each group keeps its own order
                taken[g] += 1
                assert all(abs(taken[h] - n * sizes[h] / sum(sizes)) < 1 for h in range(k))


def test_select_and_hash(prompts):
    dev = P.select(prompts, "dev")
    test = P.select(prompts, "test")
    assert len(dev) + len(test) == len(prompts)
    assert not {p.id for p in dev} & {p.id for p in test}
    assert len(P.select(prompts, "all", ["math"], limit=5)) == 5
    p = prompts[0]
    import dataclasses
    assert dataclasses.replace(p, split="test").content_hash == p.content_hash
    assert dataclasses.replace(p, turns=(p.turns[0] + "?",)).content_hash != p.content_hash
    with pytest.raises(ValueError):
        P.select(prompts, "train")
