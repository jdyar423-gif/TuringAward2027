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


# ------------------------------------------------------------ growing pair sets (review eval-1)
def _long_short(prompts, who):
    """x always writes the longer reply, so a length-preferring judge prefers x every time."""
    text = "a long and detailed reply " * 6 if who == "x" else "short"
    return {p.id: {"prompt_id": p.id, "prompt_hash": p.content_hash,
                   "turns": [{"user": t, "reply": text} for t in p.turns]} for p in prompts}


def _dev():
    from bes.eval.proxy import prompts as P
    return P.select(P.load(), "dev")


def _files(root, name):
    return {p: p.read_bytes() for p in (make_ab.key_path(root, name), root / "pairs" / name / "pairs.jsonl",
                                         root / "pairs" / name / "meta.json")}


def test_growing_a_pair_set_keeps_every_published_side_and_verdict(tmp_path):
    """A partial run is paired and judged, then the run grows and the pair set is rebuilt under
    the same name (a rerun of `all`, the 24-hour judge stage after a resume): every published
    pair keeps its side, so the stored verdicts still unblind to wins (before: 55 of 60 sides
    flipped and the score fell from 1.0 to 0.54)."""
    from bes.eval.proxy import score
    from bes.eval.proxy.__main__ import score_report
    from bes.eval.proxy.judge import StandInJudge, run_judge

    dev = _dev()
    first, more = dev[:60], dev[:120]
    judge = StandInJudge("longer")
    m1 = make_ab.make_pairs(first, _long_short(first, "x"), _long_short(first, "y"), X_RUN, Y_RUN, "g", 0, tmp_path)
    pairs1, _ = make_ab.load_pairs(tmp_path, "g")
    run_judge(pairs1, judge, "g", tmp_path, log=None)
    key1 = make_ab.load_key(tmp_path, "g")
    assert score_report(tmp_path, "g", "judge-standin-longer")["overall"]["wins"] == 60
    m2 = make_ab.make_pairs(more, _long_short(more, "x"), _long_short(more, "y"), X_RUN, Y_RUN, "g", 0, tmp_path)
    key2 = make_ab.load_key(tmp_path, "g")
    assert all(key2["assignment"][pid] == side for pid, side in key1["assignment"].items())
    assert m2["added"] == [60, 60] and m2["commitments"][0] == m1["commitment"] != m2["commitment"]
    pairs2, _ = make_ab.load_pairs(tmp_path, "g")
    assert pairs2[:60] == pairs1                         # published pairs are kept as they were
    r = run_judge(pairs2, judge, "g", tmp_path, log=None)
    assert (r["judged_now"], r["already"]) == (60, 60)   # only the new pairs are judged
    overall = score_report(tmp_path, "g", "judge-standin-longer")["overall"]
    assert (overall["n"], overall["wins"], overall["losses"], overall["score"]) == (120, 120, 0, 1.0)
    # the new sides keep every category balanced, given the old ones
    cats = {p["pair_id"]: p["category"] for p in pairs2}
    per = Counter((cats[pid], side) for pid, side in key2["assignment"].items())
    assert all(abs(per[(c, "x")] - per[(c, "y")]) <= 1 for c in set(cats.values()))
    # every verdict carries the commitment it was made under and the digest of its pair
    recs = read_jsonl(make_ab.ratings_dir(tmp_path, "g") / "judge-standin-longer.jsonl")
    assert [r["commitment"] for r in recs] == [m1["commitment"]] * 60 + [m2["commitment"]] * 60
    digests = {p["pair_id"]: make_ab.pair_digest(p) for p in pairs2}
    assert all(r["pair_digest"] == digests[r["pair_id"]] for r in recs)
    # rebuilding with nothing new changes nothing
    before = _files(tmp_path, "g")
    assert make_ab.make_pairs(more, _long_short(more, "x"), _long_short(more, "y"), X_RUN, Y_RUN, "g", 0,
                              tmp_path)["commitment"] == m2["commitment"]
    assert _files(tmp_path, "g") == before
    assert score.summarize(pairs2, key2, {})["n_rated"] == 0


def test_a_rebuild_that_would_reassign_rated_pairs_is_refused(tmp_path):
    from bes.eval.proxy.judge import StandInJudge, run_judge

    ps = _prompts()
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "r", seed=1, root=tmp_path)
    run_judge(make_ab.load_pairs(tmp_path, "r")[0], StandInJudge("longer"), "r", tmp_path, limit=3, log=None)
    before = _files(tmp_path, "r")
    for change in ({"y_run": "other/run"}, {"x_run": "ours/other"}, {"seed": 2}):
        kw = {"x_run": X_RUN, "y_run": Y_RUN, "seed": 1, **change}
        with pytest.raises(make_ab.PairSetConflict):
            make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), kw["x_run"], kw["y_run"], "r",
                               seed=kw["seed"], root=tmp_path)
        assert _files(tmp_path, "r") == before          # nothing was written
    # a published pair whose replies would change is refused as well
    changed = _records(ps, "XX")
    changed[ps[0].id]["turns"][0]["reply"] = "a different reply"
    with pytest.raises(make_ab.PairSetConflict):
        make_ab.make_pairs(ps, changed, _records(ps, "YY"), X_RUN, Y_RUN, "r", seed=1, root=tmp_path)
    # the key deleted by hand: an identical rebuild reproduces the same chain, so the stored
    # verdicts verify; a rebuild of a grown set from scratch does not, and is refused
    make_ab.key_path(tmp_path, "r").unlink()
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "r", seed=1, root=tmp_path)
    assert _files(tmp_path, "r").keys() == before.keys()
    assert make_ab.key_path(tmp_path, "r").read_bytes() == before[make_ab.key_path(tmp_path, "r")]
    small, big = ps[:16], ps
    make_ab.make_pairs(small, _records(small, "XX"), _records(small, "YY"), X_RUN, Y_RUN, "g", seed=1, root=tmp_path)
    run_judge(make_ab.load_pairs(tmp_path, "g")[0], StandInJudge("longer"), "g", tmp_path, limit=3, log=None)
    make_ab.key_path(tmp_path, "g").unlink()
    with pytest.raises(make_ab.PairSetConflict):
        make_ab.make_pairs(big, _records(big, "XX"), _records(big, "YY"), X_RUN, Y_RUN, "g", seed=1, root=tmp_path)
    # without stored verdicts a pair set may be rebuilt freely
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "free", seed=1, root=tmp_path)
    meta = make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, "other/run", "free", seed=1,
                              root=tmp_path)
    assert make_ab.load_key(tmp_path, "free")["y"] == "other/run" and meta["added"] == [len(ps)]


def test_records_under_a_foreign_commitment_refuse_the_score(tmp_path):
    from bes.eval.proxy.__main__ import score_report
    from bes.eval.proxy.common import append_jsonl
    from bes.eval.proxy.judge import StandInJudge, load_verdicts, run_judge

    ps = _prompts()
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "f", seed=1, root=tmp_path)
    pairs, meta = make_ab.load_pairs(tmp_path, "f")
    judge = StandInJudge("longer")
    run_judge(pairs, judge, "f", tmp_path, log=None)
    assert score_report(tmp_path, "f", "judge-standin-longer")["n_rated"] == len(ps)
    good = read_jsonl(ratings_path(tmp_path, "f", "judge-standin-longer"))[0]
    for forged in ({**good, "commitment": "0" * 64}, {k: v for k, v in good.items() if k != "commitment"},
                   {**good, "pair_digest": "0" * 16}):
        path = ratings_path(tmp_path, "f", "human-eve")
        path.unlink(missing_ok=True)
        append_jsonl(path, [forged])
        with pytest.raises(ValueError, match="Refused"):
            score_report(tmp_path, "f", "human-eve")
        with pytest.raises(ValueError, match="Refused"):
            rate_session(pairs, "f", "eve", tmp_path, input_fn=_scripted(["q"]), print_fn=lambda m: None)
        with pytest.raises(ValueError, match="Refused"):
            load_verdicts(tmp_path, "f", "human-eve", pub=make_ab.published(tmp_path, "f"))
    append_jsonl(ratings_path(tmp_path, "f", "judge-eve"), [{**good, "commitment": "1" * 64}])
    with pytest.raises(ValueError, match="Refused"):
        run_judge(pairs, StandInJudge("longer", key="eve"), "f", tmp_path, log=None)
    # a pairs file edited after the key was written is refused too
    path = tmp_path / "pairs" / "f" / "pairs.jsonl"
    edited = read_jsonl(path)
    edited[0]["turns"][0]["A"] = "edited"
    path.write_text("".join(json.dumps(p) + "\n" for p in edited))
    with pytest.raises(ValueError):
        score_report(tmp_path, "f", "judge-standin-longer")


def test_a_pair_set_of_the_first_key_format_is_never_scored_unchecked(tmp_path):
    """A pair set written before verdicts carried commitments (one hash of the whole key):
    its records cannot be checked, so scoring, judging and rating refuse it, and a rebuild under
    its name is refused while it holds verdicts."""
    from bes.eval.proxy.__main__ import score_report
    from bes.eval.proxy.common import append_jsonl, stable_hash, write_json_atomic, write_jsonl_atomic
    from bes.eval.proxy.judge import StandInJudge, run_judge

    ps = _prompts(6)
    salt = stable_hash(["salt", 0, "old"], 32)
    pid = {p.id: "p" + stable_hash([salt, p.id], 10) for p in ps}
    key = {"name": "old", "seed": 0, "salt": salt, "x": X_RUN, "y": Y_RUN,
           "assignment": {pid[p.id]: "x" for p in ps}, "prompt_of": {pid[p.id]: p.id for p in ps}}
    pairs = [{"pair_id": pid[p.id], "prompt_id": p.id, "category": p.category, "multi_turn": p.multi_turn,
              "turns": [{"user": t, "A": f"XX reply to {t}", "B": f"YY reply to {t}"} for t in p.turns]} for p in ps]
    write_json_atomic(make_ab.key_path(tmp_path, "old"), key)
    write_jsonl_atomic(tmp_path / "pairs" / "old" / "pairs.jsonl", pairs)
    write_json_atomic(tmp_path / "pairs" / "old" / "meta.json", {"name": "old", "n_pairs": len(ps),
                                                                 "commitment": make_ab.commitment(key)})
    assert make_ab.load_key(tmp_path, "old")["legacy"]
    append_jsonl(ratings_path(tmp_path, "old", "judge-x"), [{"pair_id": pid[ps[0].id], "verdict": "A"}])
    with pytest.raises(ValueError, match="Refused"):
        score_report(tmp_path, "old", "judge-x")
    with pytest.raises(ValueError, match="earlier version"):
        run_judge(pairs, StandInJudge(), "old", tmp_path, log=None)
    with pytest.raises(make_ab.PairSetConflict):
        make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "old", seed=0, root=tmp_path)


def test_new_sides_keep_every_category_balanced():
    import random

    rng = random.Random(3)
    cats = ["chat", "math", "code", "writing", "instruct"]
    for trial in range(200):
        existing: list[tuple[str, str, str]] = []
        n = 0
        for _step in range(rng.randint(1, 5)):
            items = [(f"i{n + j}", rng.choice(cats)) for j in range(rng.randint(0, 9))]
            n += len(items)
            sides = make_ab.assign_sides(items, seed=trial, existing=existing)
            assert set(sides) == {i for i, _ in items}
            assert sides == make_ab.assign_sides(items, seed=trial, existing=existing)   # seeded
            existing += [(i, c, sides[i]) for i, c in items]
            per = Counter((c, s) for _, c, s in existing)
            assert all(abs(per[(c, "x")] - per[(c, "y")]) <= 1 for c in cats)
            total = Counter(s for *_, s in existing)
            assert abs(total["x"] - total["y"]) <= len(cats)


# ------------------------------------------------------------------------------ rater tool
def meta_of(root, name):
    return json.loads((root / "pairs" / name / "meta.json").read_text())


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
    # every record, the retraction included, carries the published commitment and its pair's digest
    digests = {p["pair_id"]: make_ab.pair_digest(p) for p in pairs}
    assert all(r["commitment"] == meta_of(root, "t1")["commitment"] for r in log)
    assert all(r["pair_digest"] == digests[r["pair_id"]] for r in log)
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


def test_a_rebuild_with_nothing_new_still_refuses_foreign_verdicts(tmp_path):
    from bes.eval.proxy.common import append_jsonl
    from bes.eval.proxy.judge import StandInJudge, run_judge

    ps = _prompts(8)
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "n", seed=1, root=tmp_path)
    run_judge(make_ab.load_pairs(tmp_path, "n")[0], StandInJudge("longer"), "n", tmp_path, log=None)
    make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "n", seed=1, root=tmp_path)
    append_jsonl(ratings_path(tmp_path, "n", "human-mallory"), [{"pair_id": "p0", "verdict": "A"}])
    with pytest.raises(make_ab.PairSetConflict):
        make_ab.make_pairs(ps, _records(ps, "XX"), _records(ps, "YY"), X_RUN, Y_RUN, "n", seed=1, root=tmp_path)
