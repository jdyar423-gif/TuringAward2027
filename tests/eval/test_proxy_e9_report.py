"""The E9 stand-in report (scripts/e9_standin.py): replies cut by their budget and the constraint
compiler's coverage (review eval-2 and eval-5). Synthetic prompts only; the held-out half is
never read here."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bes.eval import exact_score
from bes.eval.proxy import prompts as P
from bes.eval.proxy.prompts import Prompt

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def e9():
    spec = importlib.util.spec_from_file_location("e9_standin_script", REPO / "scripts" / "e9_standin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_reply_that_used_its_whole_budget_counts_as_cut(e9):
    """Records written before bes.decode.respond reported forced endings as "length": a reply
    with every budget token used is cut, whatever its stored finish reason."""
    assert e9.cut_by_budget({"finish_reason": "stop", "completion_tokens": 320}, 320)
    assert e9.cut_by_budget({"finish_reason": "length", "completion_tokens": 100}, 320)
    assert not e9.cut_by_budget({"finish_reason": "stop", "completion_tokens": 319}, 320)
    dev = next(p for p in P.select(P.load(), "dev") if p.category == "chat" and not p.multi_turn)
    rec = {"settings": {"max_new_tokens": 12}, "prompt_id": dev.id,
           "turns": [{"reply": "Here is a list of ideas:\n- one\n- two", "finish_reason": "stop",
                      "completion_tokens": 12}]}
    deg = e9._rescore({"potentials": {dev.id: rec}})["potentials"][dev.id]["score"]["turns"][0]["degenerate"]
    assert deg["truncated"] is True and deg["degenerate"] is True
    early = {**rec, "turns": [{**rec["turns"][0], "completion_tokens": 11}]}
    deg = e9._rescore({"a": {dev.id: early}})["a"][dev.id]["score"]["turns"][0]["degenerate"]
    assert deg["truncated"] is False and deg["degenerate"] is False


def test_compiler_coverage_counts_what_the_compiler_reads(e9):
    ps = [
        Prompt(id="t-1", category="instruct", turns=("Answer in exactly 3 sentences: why is the sky blue?",),
               constraints=({"type": "sentences", "relation": "exactly", "n": 3},)),
        # an approximate word count is (deliberately) not compiled; the comma ban is
        Prompt(id="t-2", category="instruct", turns=("Write roughly a hundred words about rain, do not use commas.",),
               constraints=({"type": "words", "relation": "between", "n": 90, "n2": 110}, {"type": "no_commas"})),
        Prompt(id="t-3", category="chat", turns=("Tell me a joke.", "Now in all lowercase letters."),
               constraints=({"type": "case", "value": "lower", "turn": 1},)),
        Prompt(id="t-4", category="chat", turns=("Hello!",)),
    ]
    cov = e9.compiler_coverage(ps)
    assert cov == {"prompts": 3, "prompts_all_read": 2, "constraints": 4, "read": 3, "missed_types": {"words": 1}}
    assert set(e9.SPEC_FIELDS) == set(exact_score.CONSTRAINT_TYPES)     # every annotated type is mapped


def test_the_report_gives_both_halves_but_details_for_dev_only(e9):
    row = {k: 0 for k in ("constraints_ok", "n_constrained", "unfinished", "n_turns", "degenerate",
                          "degenerate_no_trunc", "at_budget", "at_budget_finished", "cut_off", "n_prose_finished",
                          "particles_finished", "particles", "survivors", "resampled_turns", "resample_events")}
    row.update(dict.fromkeys(("constraints_ci", "unfinished_ci", "degenerate_ci", "degenerate_no_trunc_ci",
                              "at_budget_ci", "cut_off_ci", "particles_finished_ci", "survivors_ci", "words",
                              "words_constrained", "logprob_per_token", "logprob_vs_plain",
                              "same_reply_as_plain", "tokens_per_turn", "math"), "-"))
    row.update({"logprob_pooled": 0.0, "steps_mean": 0.0, "seconds_mean": 0.0, "rules": {}, "by_type": {}})
    summary = {"arms": {a: dict(row) for a in e9.ARM_ORDER}, "paired": {}, "compiler": {
        "dev": {"prompts": 47, "prompts_all_read": 45, "constraints": 50, "read": 48,
                "missed_types": {"include": 1, "words": 1}},
        "test": {"prompts": 48, "prompts_all_read": 41, "constraints": 56, "read": 49}}}
    text = e9.table(summary)
    assert "45/47 = 96%" in text and "(not read: include 1, words 1)" in text
    assert "aggregate only, 41/48 = 85%, 49/56." in text
    assert "an END forced into the last slot counts as cut" in text
