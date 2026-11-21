"""Tests of bes.data.hindsight: relabelled prompts carry instructions that the existing response
satisfies.

Every output is checked twice: the re-compiled user message must verify on the response
(``constraints.verify``), and every added constraint is re-checked here with other means (the
transducer's ``measure_counts`` for counts, ``json``, ``str`` methods and a regular expression
for the rest).
"""
from __future__ import annotations

import collections
import copy
import json
import pathlib
import re

import numpy as np
import pytest

from bes import config
from bes.config import ASSISTANT, END
from bes.data import hindsight as H
from bes.exact import constraints as C
from bes.exact import transducer as T
from bes.exact.constraints import ConstraintSpec, Interval

LONG_LINE = ("The library opened in 1902 and has grown ever since. It now holds more than two million "
             "books, maps and recordings, and its reading rooms are open to everyone. Volunteers run "
             "weekly tours that explain the history of the building and its collections.")
RESPONSES = {
    "prose": "The sea covers most of Earth. It is salty and deep. Many creatures live there, from "
             "tiny plankton to giant whales. Tides are driven by the Moon.",
    "short": "Paris.",
    "long_line": LONG_LINE,
    "bullets": "Here are some tips:\n\n- Drink water.\n- Sleep well.\n- Walk daily.\n- Eat vegetables.",
    "numbered": "1. Preheat the oven.\n2. Mix the flour and sugar.\n3. Bake for 20 minutes.",
    "paragraphs": "First paragraph here. It has two sentences.\n\nSecond paragraph follows. It also has "
                  "two.\n\nThird one ends it.",
    "poem": "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you",
    "json": '{"name": "Ada", "languages": ["English", "French"]}',
    "code": "Here is the function:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nIt returns the sum.",
    "lower": "sure thing, here is my answer in lowercase only, as you like it.",
    "upper": "THIS IS A VERY LOUD ANSWER.",
    "question": "I can help with that. Which city are you visiting?",
    "no_commas": "This answer has no commas at all and it still reads well enough for most people who see it.",
}
USERS = ["Tell me something.", "What do you think?", "Help me with this", "Write about it:\n\nsome topic"]


def conv(response, user="Tell me something.", system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}, {"role": "assistant", "content": response}]


def independent_check(added: ConstraintSpec, text: str) -> None:
    """Re-check every added constraint without bes.exact.constraints.verify."""
    counts = T.measure_counts(text)
    for unit in ("sentences", "items", "words"):
        iv = getattr(added, unit)
        if iv is not None:
            n = counts[unit]
            assert iv.lo <= n and (iv.hi is None or n <= iv.hi), (unit, iv, n)
    if added.lines is not None:
        n = sum(1 for line in text.split("\n") if line.strip(" \t\r"))
        assert added.lines.contains(n)
    for k in added.keywords:
        assert re.search(r"(?<![^\W_])" + re.escape(k), text, re.IGNORECASE), k
    if added.json_output:
        assert isinstance(json.loads(text), (dict, list))
    if added.case == "lower":
        assert text == text.lower() and text != text.upper()
    if added.case == "upper":
        assert text == text.upper() and text != text.lower()
    if added.end_question:
        assert text.rstrip(" \n\"')*_").endswith("?")
    if added.code_block:
        assert text.count("```") >= 2
    if added.no_commas:
        assert "," not in text


def check_output(original, out, added):
    assert out is not None
    assert len(out) == len(original)
    assert out[:-2] == [dict(m) for m in original[:-2]] and out[-1] == dict(original[-1])
    old_user, new_user = original[-2]["content"], out[-2]["content"]
    assert new_user.startswith(old_user.rstrip()) and len(new_user) > len(old_user.rstrip())
    text = H.response_text(out[-1]["content"])
    spec = C.compile(new_user)
    assert spec == C.merge(C.compile(old_user), added)
    assert C.verify(spec, text) == []
    assert not added.empty
    independent_check(added, text)


@pytest.mark.parametrize("name", sorted(RESPONSES))
def test_relabel_outputs_are_verified(name):
    rng = np.random.default_rng(0)
    produced = 0
    for i in range(40):
        original = conv(RESPONSES[name], USERS[i % len(USERS)])
        res = H.relabel_with_spec(original, rng)
        if res is None:
            continue
        produced += 1
        out, added = res
        check_output(original, out, added)
    assert produced >= 30, name


def test_relabel_does_not_mutate_and_is_deterministic():
    original = conv(RESPONSES["prose"], system="Be helpful.")
    frozen = copy.deepcopy(original)
    a = H.relabel(original, np.random.default_rng(5))
    b = H.relabel(original, np.random.default_rng(5))
    assert original == frozen and a == b and a is not None
    assert a[0] == original[0] and a is not original


def test_sampling_matches_the_response():
    rng = np.random.default_rng(1)
    used = collections.Counter()
    for name in RESPONSES:
        for _ in range(60):
            res = H.relabel_with_spec(conv(RESPONSES[name]), rng)
            if res:
                for field in res[1].constraints:
                    used[(name, field.split(":")[0])] += 1
    fields = {f for _, f in used}
    assert {"sentences", "words", "items", "paragraphs", "lines", "keyword", "case", "end_question",
            "json_output", "code_block", "no_commas"} <= fields
    assert used[("json", "json_output")] > 20 and used[("lower", "case")] > 20
    assert used[("upper", "case")] > 20 and used[("code", "code_block")] > 10
    assert used[("bullets", "items")] > 10 and used[("numbered", "items")] > 10
    assert used[("question", "end_question")] > 5
    for name in ("json", "code"):          # no counts on JSON or fenced code
        assert not any(used[(name, f)] for f in ("sentences", "words", "paragraphs", "lines"))


def test_single_paragraph_rule():
    """'A single paragraph' is never requested for a short one-line answer, sometimes for a
    long one-line paragraph, and lists are never asked to have no list items."""
    rng = np.random.default_rng(2)
    short, long_line = collections.Counter(), collections.Counter()
    for _ in range(200):
        for name, counter in (("short", short), ("long_line", long_line)):
            res = H.relabel_with_spec(conv(RESPONSES[name]), rng)
            if res:
                counter[res[1].paragraphs] += 1
                if name == "short":            # one word: no sentence or word count is natural
                    assert res[1].sentences is None and res[1].words is None
    assert short[Interval(1, 1)] == 0
    assert 0 < long_line[Interval(1, 1)] < 0.3 * sum(long_line.values())
    for _ in range(100):
        res = H.relabel_with_spec(conv(RESPONSES["bullets"]), rng)
        assert res is None or res[1].items != Interval(0, 0)


def test_existing_request_is_kept_and_not_duplicated():
    user = "Describe the sea. Answer in exactly 4 sentences."
    rng = np.random.default_rng(3)
    for _ in range(40):
        res = H.relabel_with_spec(conv(RESPONSES["prose"], user), rng)
        assert res is not None
        out, added = res
        assert added.sentences is None
        assert C.compile(out[-2]["content"]).sentences == Interval(4, 4)
        check_output(conv(RESPONSES["prose"], user), out, added)


def test_violated_request_or_bad_input_gives_none():
    rng = np.random.default_rng(4)
    assert H.relabel(conv(RESPONSES["prose"], "Answer in exactly 2 sentences."), rng) is None
    assert H.relabel([{"role": "user", "content": "Hi"}], rng) is None
    assert H.relabel([], rng) is None
    assert H.relabel(conv(""), rng) is None and H.relabel(conv("   "), rng) is None
    assert H.relabel([{"role": "user", "content": ["x"]}, {"role": "assistant", "content": "Hi."}], rng) is None
    assert H.relabel([{"role": "assistant", "content": "Hi."}], rng) is None
    assert H.relabel(conv([{"type": "bogus"}]), rng) is None


def test_unverifiable_instruction_is_not_returned():
    """A user message that ends inside an open code fence swallows any appended instruction
    (fenced text is not read), so no draw verifies and nothing is returned."""
    user = "Why does this fail?\n```python\nprint(1"
    rng = np.random.default_rng(11)
    for _ in range(20):
        assert H.relabel(conv(RESPONSES["prose"], user), rng) is None


def test_bounds_and_intervals_contain_the_count():
    rng = np.random.default_rng(12)
    nice = (5, 10, 25, 50, 100)
    for n in range(1, 400):
        up, down = H._round_up(n, rng), H._round_down(n, rng)
        assert up >= n and (up == n or any(up % s == 0 for s in nice))
        assert 1 <= down <= n and (down == n or any(down % s == 0 for s in nice))
        for unit in ("words", "sentences", "items", "paragraphs", "lines"):
            iv = H._count_interval(unit, n, rng)
            assert iv is None or iv.contains(n), (unit, n, iv)


def test_multi_turn_only_last_user_message_changes():
    msgs = [{"role": "user", "content": "First question?"}, {"role": "assistant", "content": "First answer."},
            {"role": "library", "content": "Title\n\nSome passage."},
            {"role": "user", "content": "Second question?"}, {"role": "assistant", "content": RESPONSES["prose"]}]
    rng = np.random.default_rng(6)
    for _ in range(20):
        res = H.relabel_with_spec(msgs, rng)
        assert res is not None
        out, added = res
        assert out[:3] == msgs[:3] and out[4] == msgs[4]
        check_output(msgs, out, added)


def test_calculator_parts_use_the_display_text():
    from bes.tokenizer.tok import load

    tok = load()
    parts = [{"type": "text", "text": "She sold "}, {"type": "calc", "expr": "48/2", "result": "24"},
             {"type": "text", "text": " clips in May. That is half of April."}]
    assert H.response_text(parts, tok) == "She sold 48/2 = 24 clips in May. That is half of April."
    rng = np.random.default_rng(7)
    for _ in range(20):
        res = H.relabel_with_spec(conv(parts, "How many clips?"), rng, tok)
        assert res is not None
        check_output(conv(parts, "How many clips?"), *res)


def test_append_instruction_separators():
    rng = np.random.default_rng(8)
    seps = collections.Counter()
    for _ in range(300):
        out = H.append_instruction("Tell me about cats.", "Use two sentences.", rng)
        seps[out[len("Tell me about cats."):-len("Use two sentences.")]] += 1
        no_punct = H.append_instruction("Tell me about cats", "Use two sentences.", rng)
        assert no_punct[len("Tell me about cats"):-len("Use two sentences.")] in ("\n\n", "\n")
    assert set(seps) == {" ", "\n\n", "\n"} and seps[" "] > seps["\n"]
    assert H.append_instruction("", "Use two sentences.", rng) == "Use two sentences."


def test_relabelled_countdown_reaches_zero_at_end():
    """A relabelled exact sentence count drives the rem_sent register to its zero value (2)
    exactly at the END of the response."""
    from bes.data import chat_format as cf
    from bes.exact import registers as R
    from bes.tokenizer.tok import load

    tok = load()
    rng = np.random.default_rng(9)
    names = [n for n, _ in config.REG_FIELDS]
    hits = 0
    for _ in range(200):
        res = H.relabel_with_spec(conv(RESPONSES["prose"]), rng)
        if res is None or res[1].sentences is None or res[1].sentences.kind != "exactly":
            continue
        hits += 1
        ids = cf.render_conversation(res[0], tok)[0]
        regs = R.compute(ids[None], None, C.targets_for_tokens(ids[None], None, tok))
        rem = regs.reg[0, :, names.index("rem_sent")]
        a = ids.tolist().index(ASSISTANT)
        assert ids[-1] == END and rem[-1] == 2 and (rem[a:-1] > 2).all()
    assert hits >= 5


def test_real_conversations():
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "data_cache/samples/HuggingFaceTB__smol-smoltalk/data/test-00000-of-00001.parquet"
    if not path.exists():
        pytest.skip("smol-smoltalk sample not downloaded")
    import pyarrow.parquet as pq

    convs = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[:300]
    rng = np.random.default_rng(10)
    tried = done = 0
    for c in convs:
        msgs = [m for m in c if m["role"] != "system"]
        k = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
        if k is None or k == 0:
            continue
        tried += 1
        res = H.relabel_with_spec(msgs[: k + 1], rng)
        if res is not None:
            done += 1
            check_output(msgs[: k + 1], *res)
    assert tried > 200 and done > 0.9 * tried
