"""Statistics of blind comparisons: win rates, intervals, tests, sample sizes, reports.

Outcomes are from the point of view of run ``x`` (ours): ``win``, ``loss``, ``tie``,
``both_bad`` (both bad counts as a tie in every rate).

- ``score`` = (wins + ties/2) / n, with the Wilson interval computed as if the score were a
  binomial proportion over n comparisons. The variance of a {0, 1/2, 1} outcome is at most
  that of a {0, 1} outcome with the same mean, so this interval is conservative.
- ``win_rate_decisive`` = wins / (wins + losses) with its Wilson interval; with n decisive
  comparisons the 95% half-width is about ``1/sqrt(n)``.
- ``sign_test_p``: exact two-sided binomial test of wins against losses (ties dropped).
- :func:`paired_sign_test`: two arms of ours against the same opponent on the same prompts;
  per prompt the outcome scores (1, 1/2, 0) are compared and ties of the difference dropped.
- :func:`n_required`: comparisons needed to detect a score of ``0.5 + delta`` against 0.5.

No scipy: quantiles come from ``statistics.NormalDist`` and binomial tails are exact sums.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from statistics import NormalDist
from typing import Any

OUTCOME_SCORE = {"win": 1.0, "loss": 0.0, "tie": 0.5, "both_bad": 0.5}


def z_value(confidence: float = 0.95) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson(k: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` (``k`` may be fractional)."""
    if n <= 0:
        return (0.0, 1.0)
    z = z_value(confidence)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_cdf_half(k: int, n: int) -> Fraction:
    """``P(X <= k)`` for ``X ~ Binomial(n, 1/2)``, exactly."""
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    return Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)


def sign_test(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign test p-value (``min(1, 2 * P(X <= min(n_pos, n_neg)))``)."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    return float(min(Fraction(1), 2 * binom_cdf_half(min(n_pos, n_neg), n)))


def n_required(delta: float, alpha: float = 0.05, power: float = 0.8, tie_rate: float = 0.0,
               p0: float = 0.5) -> int:
    """Comparisons needed to detect a true decisive win rate ``p0 + delta`` against ``p0`` with
    a two-sided level-``alpha`` test of the given power (normal approximation):
    ``n_dec = ((z_{1-alpha/2} sqrt(p0 q0) + z_power sqrt(p1 q1)) / delta)^2``, inflated by
    ``1 / (1 - tie_rate)`` because ties carry no information."""
    if not 0 < delta < 1 - p0 or not 0 <= tie_rate < 1:
        raise ValueError("need 0 < delta < 1 - p0 and 0 <= tie_rate < 1")
    p1 = p0 + delta
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    n_dec = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / delta) ** 2
    return math.ceil(n_dec / (1 - tie_rate))


def resolution(n: int, tie_rate: float = 0.0, confidence: float = 0.95) -> dict[str, Any]:
    """What ``n`` comparisons can resolve: ``half_width``, half the width of the Wilson interval
    of a score near one half (``wilson(n / 2, n)``; ties make a score's interval a little
    narrower, so this is the conservative width), and ``needed``: the comparisons
    :func:`n_required` asks for to detect a difference of 3, 5 and 10 points at ``tie_rate``."""
    lo, hi = wilson(n / 2, n, confidence) if n > 0 else (0.0, 1.0)
    t = min(max(float(tie_rate), 0.0), 0.95)
    return {"n": int(n), "half_width": round((hi - lo) / 2, 4), "tie_rate": round(t, 4),
            "needed": {f"{d:.2f}": n_required(d, tie_rate=t) for d in (0.03, 0.05, 0.10)}}


def resolution_note(n: int, tie_rate: float = 0.0) -> str:
    """One sentence for a report under a table of ``n`` comparisons (per row), e.g. the
    held-out half (at most 200 per opponent arm): the interval half-width in points and the
    comparisons a 5- and a 10-point difference need at the given tie rate."""
    r = resolution(n, tie_rate)
    return (f"With n = {r['n']} comparisons the 95% interval of a score is about "
            f"±{100 * r['half_width']:.0f} points; at a tie rate of {r['tie_rate']:.0%}, detecting a "
            f"difference of 10 points needs {r['needed']['0.10']} comparisons and 5 points "
            f"{r['needed']['0.05']} (two-sided 5% test, 80% power), so a smaller lead on n = {r['n']} "
            "is no evidence either way.")


def tally(outcomes: Iterable[str]) -> dict[str, Any]:
    """Counts, score, decisive win rate, intervals and the sign test of a list of outcomes."""
    c = Counter(outcomes)
    bad = set(c) - set(OUTCOME_SCORE)
    if bad:
        raise ValueError(f"unknown outcomes {sorted(bad)}")
    n = sum(c.values())
    wins, losses = c["win"], c["loss"]
    ties = c["tie"] + c["both_bad"]
    decisive = wins + losses
    score = (wins + 0.5 * ties) / n if n else None
    return {
        "n": n, "wins": wins, "losses": losses, "ties": c["tie"], "both_bad": c["both_bad"],
        "score": None if score is None else round(score, 4),
        "score_ci95": [round(v, 4) for v in wilson(wins + 0.5 * ties, n)] if n else None,
        "decisive": decisive,
        "win_rate_decisive": round(wins / decisive, 4) if decisive else None,
        "win_rate_decisive_ci95": [round(v, 4) for v in wilson(wins, decisive)] if decisive else None,
        "sign_test_p": round(sign_test(wins, losses), 6),
    }


def paired_sign_test(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    """Arm a against arm b on the prompts both have (``prompt_id -> score in {0, .5, 1}``)."""
    common = sorted(set(a) & set(b))
    pos = sum(1 for k in common if a[k] > b[k])
    neg = sum(1 for k in common if a[k] < b[k])
    mean_diff = sum(a[k] - b[k] for k in common) / len(common) if common else None
    return {"n": len(common), "a_better": pos, "b_better": neg, "same": len(common) - pos - neg,
            "mean_score_diff": None if mean_diff is None else round(mean_diff, 4),
            "p": round(sign_test(pos, neg), 6)}


# --------------------------------------------------------------------------------- reports
def _words(text: str) -> int:
    return len(text.split())


def length_buckets(pairs: Sequence[Mapping[str, Any]], key: Mapping[str, Any],
                   outcomes: Mapping[str, str]) -> dict[str, Any]:
    """Outcomes split by the length ratio of x's replies to y's (words over all turns): x
    shorter (< 2/3), similar, x longer (> 3/2). A judge that rewards length shows a rising
    score across the buckets that the request does not justify."""
    rows: dict[str, list[str]] = {"x_shorter": [], "similar": [], "x_longer": []}
    for pair in pairs:
        pid = pair["pair_id"]
        if pid not in outcomes:
            continue
        a = sum(_words(t["A"]) for t in pair["turns"])
        b = sum(_words(t["B"]) for t in pair["turns"])
        x, y = (a, b) if key["assignment"][pid] == "x" else (b, a)
        ratio = (x + 1) / (y + 1)
        bucket = "x_shorter" if ratio < 2 / 3 else "x_longer" if ratio > 1.5 else "similar"
        rows[bucket].append(outcomes[pid])
    return {k: tally(v) if v else {"n": 0} for k, v in rows.items()}


def summarize(pairs: Sequence[Mapping[str, Any]], key: Mapping[str, Any], verdicts: Mapping[str, str],
              judge_records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Full summary of one pair set rated by one source."""
    from bes.eval.proxy.make_ab import unblind

    outcomes = {pid: unblind(v, pid, key) for pid, v in verdicts.items() if pid in key["assignment"]}
    by_id = {p["pair_id"]: p for p in pairs}
    out: dict[str, Any] = {"x": key["x"], "y": key["y"], "pairset": key["name"],
                           "n_pairs": len(pairs), "n_rated": len(outcomes),
                           "overall": tally(outcomes.values())}
    cats: dict[str, list[str]] = {}
    turns: dict[str, list[str]] = {"single_turn": [], "multi_turn": []}
    for pid, o in outcomes.items():
        p = by_id.get(pid)
        if p is None:
            continue
        cats.setdefault(p["category"], []).append(o)
        turns["multi_turn" if p.get("multi_turn") else "single_turn"].append(o)
    out["by_category"] = {c: tally(v) for c, v in sorted(cats.items())}
    out["by_turns"] = {k: tally(v) for k, v in turns.items() if v}
    out["by_length"] = length_buckets(pairs, key, outcomes)
    ties = out["overall"]["ties"] + out["overall"]["both_bad"]
    tie_rate = ties / out["overall"]["n"] if out["overall"]["n"] else 0.0
    out["power"] = {f"delta_{d:.2f}": n_required(d, tie_rate=min(tie_rate, 0.95))
                    for d in (0.03, 0.05, 0.10)}
    out["per_prompt_score"] = {by_id[pid]["prompt_id"]: OUTCOME_SCORE[o]
                               for pid, o in outcomes.items() if pid in by_id}
    if judge_records:
        recs = [r for r in judge_records if r.get("pair_id") in outcomes]
        if recs:
            out["judge"] = {
                "orders_agree": round(sum(bool(r["agree"]) for r in recs) / len(recs), 4),
                "first_position_pref_mean": round(sum(r["first_position_pref"] for r in recs) / len(recs), 4),
                "format_mass_mean": round(sum((r["order1"]["mass"] + r["order2_swapped"]["mass"]) / 2
                                              for r in recs) / len(recs), 4),
                "clipped": sum(bool(r.get("clipped")) for r in recs),
            }
    return out


def _row(name: str, t: Mapping[str, Any]) -> str:
    if not t.get("n"):
        return f"| {name} | 0 | | | | | | |"
    ci = t["score_ci95"]
    dec = t.get("win_rate_decisive")
    dci = t.get("win_rate_decisive_ci95")
    return (f"| {name} | {t['n']} | {t['wins']}/{t['losses']}/{t['ties'] + t['both_bad']} | "
            f"{t['score']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{'' if dec is None else f'{dec:.3f}'} | "
            f"{'' if dci is None else f'[{dci[0]:.3f}, {dci[1]:.3f}]'} | {t['sign_test_p']:.4f} |")


def to_markdown(summary: Mapping[str, Any], source: str = "") -> str:
    """A Markdown report of :func:`summarize` output."""
    lines = [f"# Blind comparison `{summary['pairset']}`" + (f" rated by `{source}`" if source else ""),
             "", f"- x (ours): `{summary['x']}`", f"- y (opponent): `{summary['y']}`",
             f"- rated pairs: {summary['n_rated']} of {summary['n_pairs']}", "",
             "Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). "
             "Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.", "",
             "| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |",
             "|---|---|---|---|---|---|---|---|", _row("overall", summary["overall"])]
    for name, t in summary["by_category"].items():
        lines.append(_row(name, t))
    for name, t in summary["by_turns"].items():
        lines.append(_row(name, t))
    lines += ["", "Length control (ratio of x's words to y's words):", "",
              "| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |",
              "|---|---|---|---|---|---|---|---|"]
    for name, t in summary["by_length"].items():
        lines.append(_row(name, t))
    lines += ["", "Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): "
              + ", ".join(f"{k.replace('delta_', '+')}: {v}" for k, v in summary["power"].items())]
    o = summary["overall"]
    if o.get("n"):
        lines += ["", resolution_note(o["n"], (o["ties"] + o["both_bad"]) / o["n"])]
    if "judge" in summary:
        j = summary["judge"]
        lines += ["", f"Judge diagnostics: both orders agree on {j['orders_agree']:.1%} of pairs; mean "
                  f"preference for the first-shown reply {j['first_position_pref_mean']:+.3f}; mean "
                  f"probability mass on A/B/C {j['format_mass_mean']:.3f}; replies clipped in {j['clipped']} pairs."]
    return "\n".join(lines) + "\n"


def exact_markdown(run_id: str, s: Mapping[str, Any]) -> str:
    """Markdown summary of ``bes.eval.exact_score.summarize`` output for one run."""
    d = s["degenerate"]
    lines = [f"# Exact scores of `{run_id}`", "",
             f"- prompts: {s['n_prompts']} ({s['n_turns']} replies); counting backend: `{s['count_backend']}`",
             f"- math: {s['math']['n']} prompts, last answer correct {s['math']['correct']} "
             f"(all {s['math'].get('n_answers')} answers incl. first turns: {s['math'].get('answers_correct')}; "
             f"lenient, reference appears anywhere: {s['math']['any_match']}), extraction {s['math']['methods']}",
             f"- constraints: {s['constraints']['n_prompts']} prompts, all satisfied {s['constraints']['prompts_all_ok']}, "
             f"single constraints satisfied {s['constraints']['constraints_ok']} of {s['constraints']['n_constraints']}",
             f"- degenerate replies {d['degenerate']}: empty {d['empty']}, cut at the token limit {d['truncated']}, "
             f"looping tail {d['loop']}, repeated lines {d['repeated_lines']}, unclosed code fence "
             f"{d['unclosed_fence']}, role markers {d['role_leak']}; mean repeated-4-gram rate {s['rep4_mean']}",
             f"- length in words: mean {s['words']['mean']}, median {s['words']['median']}, p90 {s['words']['p90']}",
             "", "| constraint type | n | satisfied |", "|---|---|---|"]
    for k, v in s["constraints"]["by_type"].items():
        lines.append(f"| {k} | {v['n']} | {v['ok']} |")
    lines += ["", "| category | n | math correct | constraints ok | degenerate | mean words |",
              "|---|---|---|---|---|---|"]
    for k, v in s["by_category"].items():
        lines.append(f"| {k} | {v['n']} | {v['math_correct']} | {v['constraints_ok']} | {v['degenerate']} | {v['words_mean']} |")
    if "project_checker" in s:
        lines += ["", f"Project constraint compiler cross-check: {s['project_checker']}"]
    return "\n".join(lines) + "\n"
