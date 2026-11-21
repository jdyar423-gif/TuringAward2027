"""Judge: order-swap consistency, stand-in judges, prompt rendering, resumable runs, scoring."""
from __future__ import annotations

import itertools

import pytest

from bes.eval.proxy import judge as J
from bes.eval.proxy import make_ab, score
from bes.eval.proxy.common import read_jsonl
from bes.eval.proxy.prompts import Prompt


def test_combine_truth_table():
    swap = {"A": "B", "B": "A", "C": "C"}
    for first, second in itertools.product("ABC", repeat=2):
        got = J.combine(first, second)
        if first in "AB" and swap[second] == first:
            assert got == first
        else:
            assert got == "tie"
    assert J.combine("A", "B") == "A"      # A in the blinded order, still A after the swap
    assert J.combine("A", "A") == "tie"    # always the first-shown reply: position bias


def _pair(a, b, pid="p1", turns=1):
    return {"pair_id": pid, "prompt_id": "q", "category": "chat", "multi_turn": turns > 1,
            "turns": [{"user": f"user {t}", "A": a, "B": b} for t in range(turns)]}


def test_position_biased_judge_gives_ties():
    j = J.StandInJudge("first")
    for a, b in [("short", "a much longer reply"), ("x", "y")]:
        rec = J.judge_pair(j, _pair(a, b))
        assert rec["verdict"] == "tie" and not rec["agree"]
        assert rec["first_position_pref"] > 0.4


def test_consistent_judge_picks_the_same_reply_in_both_orders():
    j = J.StandInJudge("longer")
    rec = J.judge_pair(j, _pair("a long and detailed reply", "short"))
    assert rec["verdict"] == "A" and rec["agree"]
    assert rec["order1"]["verdict"] == "A" and rec["order2_swapped"]["verdict"] == "B"
    assert rec["soft"]["A"] > rec["soft"]["B"]
    rec = J.judge_pair(j, _pair("short", "a long and detailed reply"))
    assert rec["verdict"] == "B"
    assert J.judge_pair(j, _pair("same", "same"))["verdict"] == "tie"
    kw = J.StandInJudge("keyword", keyword="Paris")
    assert J.judge_pair(kw, _pair("It is Paris.", "It is Lyon."))["verdict"] == "A"
    assert J.judge_pair(kw, _pair("It is Paris.", "Paris, I think."))["verdict"] == "tie"


def test_render_and_parse_roundtrip():
    msgs, clipped = J.render_messages([{"user": "Q?", "A": "first reply", "B": ""}])
    assert msgs[0]["role"] == "system" and not clipped
    assert "(empty reply)" in msgs[1]["content"] and msgs[1]["content"].endswith(J.QUESTION)
    assert J.parse_rendered(msgs) == ("first reply", "(empty reply)")
    multi, _ = J.render_messages([{"user": "Q1", "A": "a1", "B": "b1"}, {"user": "Q2", "A": "a2\nmore", "B": "b2"}])
    a, b = J.parse_rendered(multi)
    assert "a1" in a and "a2\nmore" in a and "b1" in b and "b2" in b and "a1" not in b
    assert "Assistant A:" in multi[1]["content"] and "whole conversation" in multi[1]["content"]
    long, clipped = J.render_messages([{"user": "Q", "A": "x" * (J.MAX_REPLY_CHARS + 50), "B": "y"}])
    assert clipped and "continues for 50 more characters" in long[1]["content"]
    assert "Length is not a merit" in J.JUDGE_INSTRUCTIONS


def _pairset(tmp_path, n=12):
    ps = [Prompt(id=f"chat-{i:03d}", category="chat", turns=(f"q{i}",)) for i in range(n)]
    x = {p.id: {"prompt_hash": p.content_hash, "turns": [{"user": p.turns[0], "reply": "a long detailed answer " * 3}]} for p in ps}
    y = {p.id: {"prompt_hash": p.content_hash, "turns": [{"user": p.turns[0], "reply": "short"}]} for p in ps}
    make_ab.make_pairs(ps, x, y, "ours/run", "opp/run", "js", seed=5, root=tmp_path)
    return make_ab.load_pairs(tmp_path, "js")[0]


def test_run_judge_resumable_and_scored(tmp_path):
    pairs = _pairset(tmp_path)
    j = J.StandInJudge("longer")
    r1 = J.run_judge(pairs, j, "js", tmp_path, limit=5, log=None)
    assert (r1["judged_now"], r1["already"], r1["remaining"]) == (5, 0, 7)
    r2 = J.run_judge(pairs, j, "js", tmp_path, log=None)
    assert (r2["judged_now"], r2["already"]) == (7, 5)
    calls = j.calls
    r3 = J.run_judge(pairs, j, "js", tmp_path, log=None)
    assert r3["judged_now"] == 0 and j.calls == calls
    verdicts = J.load_verdicts(tmp_path, "js", "judge-standin-longer")
    assert len(verdicts) == len(pairs)
    key = make_ab.load_key(tmp_path, "js")
    records = read_jsonl(J.ratings_path(tmp_path, "js", "judge-standin-longer"))
    summary = score.summarize(pairs, key, verdicts, records)
    assert summary["overall"]["wins"] == len(pairs)          # ours is always the longer reply
    assert summary["judge"]["orders_agree"] == 1.0
    # a position-biased judge cannot produce wins
    jb = J.StandInJudge("first", key="biased")
    J.run_judge(pairs, jb, "js", tmp_path, log=None)
    biased = score.summarize(pairs, key, J.load_verdicts(tmp_path, "js", "judge-biased"))
    assert biased["overall"]["wins"] == biased["overall"]["losses"] == 0


def test_run_judge_deadline(tmp_path):
    pairs = _pairset(tmp_path, 4)
    r = J.run_judge(pairs, J.StandInJudge(), "js", tmp_path, minutes=1e-9, log=None)
    assert r["judged_now"] == 0 and r["remaining"] == 4


def test_load_verdicts_last_record_wins(tmp_path):
    from bes.eval.proxy.common import append_jsonl
    path = J.ratings_path(tmp_path, "s", "human-bo")
    append_jsonl(path, [{"pair_id": "a", "verdict": "A"}, {"pair_id": "b", "verdict": "B"},
                        {"pair_id": "a", "verdict": "retracted"}, {"pair_id": "b", "verdict": "tie"},
                        {"pair_id": "c", "verdict": "weird"}])
    assert J.load_verdicts(tmp_path, "s", "human-bo") == {"b": "tie"}


def test_protocol_document_quotes_the_judge_prompt():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[2] / "docs" / "EVAL_PROTOCOL.md").read_text(encoding="utf-8")
    for line in [J.JUDGE_SYSTEM, J.QUESTION, *J.JUDGE_INSTRUCTIONS.split("\n")]:
        assert line in doc, line


def test_custom_rule_probabilities_are_normalised():
    j = J.StandInJudge(lambda a, b: (2.0, 1.0, 1.0))
    msgs, _ = J.render_messages([{"user": "q", "A": "a", "B": "b"}])
    p = j.probs(msgs)
    assert p["A"] == pytest.approx(0.5) and sum(p[o] for o in J.OPTIONS) == pytest.approx(1.0)
