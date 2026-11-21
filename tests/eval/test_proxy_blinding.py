"""Blind pairs: balanced seeded assignment, hidden key with commitment, no leaks; the rater tool."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from bes.eval.proxy import make_ab
from bes.eval.proxy.common import read_jsonl
from bes.eval.proxy.judge import load_verdicts, ratings_path
from bes.eval.proxy.prompts import Prompt
from bes.eval.proxy.rate import rate_session

X_RUN, Y_RUN = "ours-ckpt-7/default-aaaa-s0", "falcon-e6003438/recommended-bbbb-s0"


def _prompts(n=24):
    cats = ["chat", "math", "code", "writing", "instruct"]
    out = []
    for i in range(n):
        turns = (f"question {i}?",) if i % 4 else (f"question {i}?", f"follow up {i}?")
        out.append(Prompt(id=f"{cats[i % 5]}-{i:03d}", category=cats[i % 5], turns=turns,
                          answer=("1", "2") if (cats[i % 5] == "math" and len(turns) == 2) else ("1" if cats[i % 5] == "math" else None)))
    return out


def _records(prompts, who):
    return {p.id: {"prompt_id": p.id, "prompt_hash": p.content_hash,
                   "turns": [{"user": t, "reply": f"{who} reply to {t}"} for t in p.turns]} for p in prompts}


@pytest.fixture()
def pairset(tmp_path):
    ps = _prompts()
    meta = make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "t1", seed=7, root=tmp_path)
    return tmp_path, ps, meta


def test_assign_sides_balanced_and_seeded():
    items = [(f"id{i}", ["a", "b", "c"][i % 3]) for i in range(101)]
    s1 = make_ab.assign_sides(items, seed=1)
    assert s1 == make_ab.assign_sides(items, seed=1)
    assert s1 != make_ab.assign_sides(items, seed=2)
    total = Counter(s1.values())
    assert abs(total["x"] - total["y"]) <= 1
    for cat in "abc":
        c = Counter(s1[i] for i, k in items if k == cat)
        assert abs(c["x"] - c["y"]) <= 1, cat
    # over many seeds, every item is on each side about equally often (no fixed pattern)
    counts = Counter()
    for seed in range(200):
        counts.update(i for i, side in make_ab.assign_sides(items, seed).items() if side == "x")
    assert min(counts.values()) > 60 and max(counts.values()) < 140


def test_pairs_file_reveals_nothing(pairset):
    root, ps, meta = pairset
    text = (root / "pairs" / "t1" / "pairs.jsonl").read_text() + (root / "pairs" / "t1" / "meta.json").read_text()
    for secret in (X_RUN, Y_RUN, "ours", "falcon", '"seed"', '"salt"', "assignment"):
        assert secret not in text
    pairs = read_jsonl(root / "pairs" / "t1" / "pairs.jsonl")
    assert len(pairs) == meta["n_pairs"] == len(ps)
    for p in pairs:
        assert set(p) == {"pair_id", "prompt_id", "category", "multi_turn", "turns"}
        assert p["prompt_id"] not in p["pair_id"]
    # file order is shuffled, not the prompt order
    assert [p["prompt_id"] for p in pairs] != [q.id for q in ps]


def test_key_needed_and_consistent(pairset):
    root, ps, meta = pairset
    pairs, _ = make_ab.load_pairs(root, "t1")
    key = make_ab.load_key(root, "t1")
    assert key["x"] == X_RUN and key["y"] == Y_RUN
    sides = Counter(key["assignment"].values())
    assert abs(sides["x"] - sides["y"]) <= 1
    for p in pairs:
        x_side = "A" if key["assignment"][p["pair_id"]] == "x" else "B"
        for t in p["turns"]:
            assert t[x_side].startswith("XX reply") and t["B" if x_side == "A" else "A"].startswith("YY reply")
        assert make_ab.unblind(x_side, p["pair_id"], key) == "win"
        assert make_ab.unblind("tie", p["pair_id"], key) == "tie"
    # without the key nothing can be unblinded
    make_ab.key_path(root, "t1").unlink()
    with pytest.raises(FileNotFoundError):
        make_ab.load_key(root, "t1")


def test_tampered_key_detected(pairset):
    root, _, _ = pairset
    path = make_ab.key_path(root, "t1")
    key = json.loads(path.read_text())
    first = sorted(key["assignment"])[0]
    key["assignment"][first] = "y" if key["assignment"][first] == "x" else "x"
    path.write_text(json.dumps(key))
    with pytest.raises(ValueError):
        make_ab.load_key(root, "t1")


def test_same_seed_same_pairs(tmp_path):
    ps = _prompts()
    for name in ("a", "b"):
        make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "same", seed=3, root=tmp_path / name)
    a = (tmp_path / "a" / "keys" / "same.json").read_text()
    b = (tmp_path / "b" / "keys" / "same.json").read_text()
    assert json.loads(a)["assignment"] == json.loads(b)["assignment"]
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "same", seed=4, root=tmp_path / "c")
    c = json.loads((tmp_path / "c" / "keys" / "same.json").read_text())
    assert c["assignment"] != json.loads(a)["assignment"]


def test_stale_outputs_are_excluded(tmp_path):
    ps = _prompts(10)
    x = _records(ps, "XX")
    y = _records(ps, "YY")
    x[ps[0].id]["prompt_hash"] = "stale"
    del y[ps[1].id]
    meta = make_ab.make_pairs(ps, x, y, X_RUN, Y_RUN, "s", seed=0, root=tmp_path)
    assert meta["n_pairs"] == 8
    with pytest.raises(ValueError):
        make_ab.make_pairs(ps[:1], x, y, X_RUN, Y_RUN, "none", seed=0, root=tmp_path)


# ------------------------------------------------------------------------------ rater tool
def _scripted(answers):
    it = iter(answers)
    return lambda _prompt: next(it)


def test_rate_session_resumable_with_undo_and_skip(pairset):
    root, ps, _ = pairset
    pairs, _ = make_ab.load_pairs(root, "t1")
    make_ab.key_path(root, "t1").unlink()   # the rater never needs the key
    shown = []
    stats = rate_session(pairs, "t1", "alice", root, input_fn=_scripted(["?", "zz", "a", "s", "b", "u", "t", "x", "q"]),
                         print_fn=shown.append, width=60)
    assert stats == {"rated": 4, "skipped": 1, "undone": 1}   # actions: a, b, t, x; b undone
    screen = "\n".join(shown)
    for secret in (X_RUN, Y_RUN, "assignment"):
        assert secret not in screen
    verdicts = load_verdicts(root, "t1", "human-alice")
    assert sorted(verdicts.values()) == ["A", "both_bad", "tie"]
    log = read_jsonl(ratings_path(root, "t1", "human-alice"))
    assert sum(r["verdict"] == "retracted" for r in log) == 1
    # a second session only shows the pairs that are still unrated
    seen = []
    rate_session(pairs, "t1", "alice", root, input_fn=_scripted(["a"] * len(pairs)), print_fn=seen.append, width=60)
    assert len(load_verdicts(root, "t1", "human-alice")) == len(pairs)
    assert sum("=== pair" in s for s in seen) == len(pairs) - 3


def test_raters_get_different_orders(pairset):
    root, _, _ = pairset
    pairs, _ = make_ab.load_pairs(root, "t1")
    first_screens = set()
    for rater in ("r1", "r2", "r3"):
        shown = []
        rate_session(pairs, "t1", rater, root, input_fn=_scripted(["q"]), print_fn=shown.append, width=60)
        first_screens.add(next(s for s in shown if s.startswith("=== pair")))
    assert len(first_screens) > 1
