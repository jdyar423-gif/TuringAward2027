"""Statistics of blind comparisons against hand computation and brute force."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from bes.eval.proxy import score as S


def test_wilson_hand_value():
    # k = 60, n = 100, z = 1.959964: centre = (0.6 + z^2/200) / (1 + z^2/100) = 0.596307,
    # half = z * sqrt(0.0024 + z^2/40000) / (1 + z^2/100) = 0.094304
    lo, hi = S.wilson(60, 100)
    assert lo == pytest.approx(0.502003, abs=2e-5)
    assert hi == pytest.approx(0.690611, abs=2e-5)


@pytest.mark.parametrize("k, n", [(0, 10), (3, 10), (10, 10), (17.5, 40), (412, 800), (1, 1)])
def test_wilson_is_root_of_score_equation(k, n):
    # The Wilson bounds are the roots of (p_hat - p)^2 = z^2 p (1 - p) / n.
    z = S.z_value(0.95)
    p_hat = k / n
    a = 1 + z * z / n
    b = -(2 * p_hat + z * z / n)
    c = p_hat * p_hat
    roots = sorted(np.roots([a, b, c]).real)
    lo, hi = S.wilson(k, n)
    assert lo == pytest.approx(max(0.0, roots[0]), abs=1e-9)
    assert hi == pytest.approx(min(1.0, roots[1]), abs=1e-9)


def test_wilson_half_width_rule_of_thumb():
    # half-width = 1.96 * 0.5 / sqrt(n), shrunk slightly by Wilson at small n: "about 1/sqrt(n)"
    for n in (100, 400, 800):
        lo, hi = S.wilson(n / 2, n)
        assert (hi - lo) / 2 == pytest.approx(1 / math.sqrt(n), rel=0.05)


def test_sign_test_hand_and_brute_force():
    assert S.sign_test(9, 1) == pytest.approx(22 / 1024)
    assert S.sign_test(5, 5) == 1.0
    assert S.sign_test(0, 0) == 1.0
    assert S.sign_test(0, 6) == pytest.approx(2 / 64)
    for n in range(1, 13):
        for k in range(n + 1):
            # brute force: probability of an outcome at least as unbalanced as k vs n - k
            extreme = abs(2 * k - n)
            count = sum(1 for bits in itertools.product((0, 1), repeat=n) if abs(2 * sum(bits) - n) >= extreme)
            assert S.sign_test(k, n - k) == pytest.approx(min(1.0, count / 2 ** n)), (k, n)


def test_binom_cdf_half_exact():
    assert S.binom_cdf_half(1, 10) == Fraction(11, 1024)
    assert S.binom_cdf_half(-1, 5) == 0 and S.binom_cdf_half(5, 5) == 1


def test_n_required_hand_value_and_power():
    za, zb = 1.959964, 0.841621
    hand = ((za * 0.5 + zb * math.sqrt(0.55 * 0.45)) / 0.05) ** 2  # 782.5
    assert S.n_required(0.05) == math.ceil(hand) == 783
    assert S.n_required(0.05, tie_rate=0.5) == math.ceil(hand / 0.5)
    assert S.n_required(0.10) < S.n_required(0.05) < S.n_required(0.03)
    # exact power of the exact sign test at n = 783 decisive comparisons with p = 0.55
    n = 783
    crit = next(k for k in range(n // 2, n + 1) if S.sign_test(k, n - k) <= 0.05)
    p = Fraction(55, 100)
    power = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(crit, n + 1))
    power += sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(0, n - crit + 1))
    assert 0.75 < float(power) < 0.85
    with pytest.raises(ValueError):
        S.n_required(0.6)


def test_tally_counts_ties_as_half():
    t = S.tally(["win"] * 6 + ["loss"] * 2 + ["tie"] * 1 + ["both_bad"] * 1)
    assert (t["n"], t["wins"], t["losses"], t["ties"], t["both_bad"], t["decisive"]) == (10, 6, 2, 1, 1, 8)
    assert t["score"] == pytest.approx(0.7)
    assert t["win_rate_decisive"] == pytest.approx(0.75)
    assert t["score_ci95"] == [round(v, 4) for v in S.wilson(7, 10)]
    assert t["win_rate_decisive_ci95"] == [round(v, 4) for v in S.wilson(6, 8)]
    assert t["sign_test_p"] == pytest.approx(S.sign_test(6, 2), abs=1e-6)
    assert S.tally([])["score"] is None
    with pytest.raises(ValueError):
        S.tally(["draw"])


def test_paired_sign_test_by_hand():
    a = {"p1": 1.0, "p2": 0.5, "p3": 1.0, "p4": 0.0, "p5": 1.0}
    b = {"p1": 0.0, "p2": 0.5, "p3": 0.5, "p4": 0.0, "p6": 1.0}
    r = S.paired_sign_test(a, b)
    assert (r["n"], r["a_better"], r["b_better"], r["same"]) == (4, 2, 0, 2)
    assert r["mean_score_diff"] == pytest.approx((1.0 + 0 + 0.5 + 0) / 4)
    assert r["p"] == pytest.approx(0.5)


def _pairset():
    pairs, assignment = [], {}
    cats = ["chat", "math", "code", "writing"]
    for i in range(16):
        pid = f"p{i:02d}"
        x_is_a = i % 2 == 0
        x_text, y_text = ("x " * (5 if i < 8 else 20)).strip(), ("y " * 10).strip()
        a, b = (x_text, y_text) if x_is_a else (y_text, x_text)
        pairs.append({"pair_id": pid, "prompt_id": f"q{i}", "category": cats[i % 4], "multi_turn": i % 4 == 3,
                      "turns": [{"user": "u", "A": a, "B": b}]})
        assignment[pid] = "x" if x_is_a else "y"
    key = {"name": "t", "x": "ours/run", "y": "opp/run", "assignment": assignment}
    return pairs, key


def test_summarize_unblinds_and_slices():
    pairs, key = _pairset()
    # x wins on even pairs (x shown as A, verdict A), loses on 1 and 5, ties elsewhere
    verdicts = {}
    for i, p in enumerate(pairs):
        if i % 2 == 0:
            verdicts[p["pair_id"]] = "A"
        elif i in (1, 5):
            verdicts[p["pair_id"]] = "A"   # y is A on odd pairs -> loss for x
        else:
            verdicts[p["pair_id"]] = "tie" if i % 3 else "both_bad"
    s = S.summarize(pairs, key, verdicts)
    o = s["overall"]
    assert (o["wins"], o["losses"], o["ties"] + o["both_bad"]) == (8, 2, 6)
    assert o["score"] == pytest.approx((8 + 3) / 16)
    assert set(s["by_category"]) == {"chat", "math", "code", "writing"}
    assert s["by_turns"]["multi_turn"]["n"] == 4
    assert s["by_length"]["x_shorter"]["n"] == 8 and s["by_length"]["x_longer"]["n"] == 8
    assert s["per_prompt_score"]["q0"] == 1.0 and s["per_prompt_score"]["q1"] == 0.0
    md = S.to_markdown(s, "judge-test")
    assert "| overall | 16 | 8/2/6 |" in md and "Length control" in md


def test_the_resolution_note_states_what_n_comparisons_can_resolve():
    """The held-out half gives at most 200 comparisons per opponent arm: about +/-7 points by
    hand (1.96 * 0.5 / sqrt(200) = 0.0693), and even a 10-point difference needs more than 200
    comparisons at a 30% tie rate (review eval-7: the model card and the runbook quoted 'about
    5 points on 400 prompts' under the 200-prompt table)."""
    r = S.resolution(200, tie_rate=0.3)
    assert r["half_width"] == pytest.approx(1.959964 * 0.5 / math.sqrt(200), abs=2e-3)
    assert r["needed"] == {"0.03": S.n_required(0.03, tie_rate=0.3), "0.05": S.n_required(0.05, tie_rate=0.3),
                           "0.10": S.n_required(0.10, tie_rate=0.3)} and r["needed"]["0.10"] > 200
    note = S.resolution_note(200, 0.3)
    assert "n = 200" in note and "±7 points" in note and "277" in note and "400" not in note
    assert "±5 points" in S.resolution_note(400) and "±10 points" in S.resolution_note(100)
    # the comparison report carries the note for its own n and tie rate
    summary = {"pairset": "p", "x": "x", "y": "y", "n_rated": 200, "n_pairs": 200,
               "overall": S.tally(["win"] * 80 + ["loss"] * 60 + ["tie"] * 60), "by_category": {},
               "by_turns": {}, "by_length": {}, "power": {"delta_0.05": 1}}
    assert S.resolution_note(200, 0.3) in S.to_markdown(summary)
