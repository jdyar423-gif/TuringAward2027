"""The blind arena (bes/eval/arena.py, scripts/arena.py) with scripted input and stand-in
generators: blinding (no identity on screen before a vote, a fresh label order per prompt that a
resumed session reproduces), every reply ready before anything is shown, hidden reasoning never
shown, votes and Wilson tallies after every vote (checked against an independent formula),
resuming, undo / skip / quit, multi-turn conversations, failures, and the opponents' settings
taken from the proxy harness."""
from __future__ import annotations

import importlib.util
import json
import math
import threading
import time
from pathlib import Path

import pytest

from bes.eval import arena as A
from bes.eval.proxy import opponents
from bes.eval.proxy.common import GenSettings

REPO = Path(__file__).resolve().parents[2]
SECRET = ("SECRET-ONE", "SECRET-TWO", "SECRET-THREE")


def scripted(answers):
    it = iter(answers)

    def ask(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    return ask


class Screen:
    """Collects printed text with the time it was printed."""

    def __init__(self):
        self.lines: list[tuple[float, str]] = []

    def __call__(self, msg: str) -> None:
        self.lines.append((time.monotonic(), str(msg)))

    @property
    def text(self) -> str:
        return "\n".join(m for _, m in self.lines)


def standin(tag: str, delay: float = 0.0, log: list | None = None):
    """A generator whose reply names no contestant; records its calls."""
    def generate(messages, settings, seed):
        t0 = time.monotonic()
        time.sleep(delay)
        if log is not None:
            log.append({"tag": tag, "messages": [dict(m) for m in messages], "start": t0,
                        "end": time.monotonic()})
        user = messages[-1]["content"]
        return {"text": f"reply {tag.lower()[-3:]} to: {user}", "finish_reason": "stop"}
    generate.key = f"key-{tag}"
    return generate


def contestants(n=3, delays=(0.0, 0.0, 0.0), log=None, gens=None):
    out = []
    for i in range(n):
        g = (gens or {}).get(i) or standin(SECRET[i], delays[i] if i < len(delays) else 0.0, log)
        out.append(A.generator_contestant(SECRET[i], g, key=f"key-{SECRET[i]}"))
    return out


def prompts_file(tmp_path, n=6) -> Path:
    p = tmp_path / "prompts.txt"
    p.write_text("# a comment\n\n" + "\n".join(f"Question number {i}?" for i in range(n)) + "\n")
    return p


def wilson_ref(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ================================================================== blinding and timing
def test_no_identity_before_the_vote_and_reveal_only_after(tmp_path):
    screen = Screen()
    prompts = A.file_prompts(prompts_file(tmp_path, 3))
    asked = []

    def ask(prompt):
        asked.append(screen.text)               # everything shown before this question
        return "a"
    arena = A.Arena(contestants(), tmp_path / "ar", reveal=True, print_fn=screen, input_fn=ask)
    arena.run(prompts)
    assert len(asked) == 3
    for i, before in enumerate(asked):
        # the reveal of earlier prompts is on screen, but the current item never is
        shown_now = before.rsplit("=== prompt", 1)[1]
        assert not any(s in shown_now for s in SECRET), shown_now
        assert not any(f"key-{s}" in before for s in SECRET)
    reveals = [m for _, m in screen.lines if m.startswith("revealed:")]
    assert len(reveals) == 3 and all(all(s in r for s in SECRET) for r in reveals)
    # without reveal no identity is ever printed
    screen2 = Screen()
    A.Arena(contestants(), tmp_path / "ar2", print_fn=screen2, input_fn=scripted(["b", "t", "x"])).run(prompts)
    assert not any(s in screen2.text for s in SECRET)


def test_replies_are_shown_only_when_every_contestant_is_ready(tmp_path):
    log: list = []
    screen = Screen()
    arena = A.Arena(contestants(delays=(0.3, 0.0, 0.15), log=log), tmp_path / "ar", parallel=True,
                    print_fn=screen, input_fn=scripted(["a"]))
    arena.run(A.file_prompts(prompts_file(tmp_path, 1)))
    shown = next(t for t, m in screen.lines if m.startswith("=== prompt"))
    assert len(log) == 3 and max(c["end"] for c in log) <= shown
    # the only message between the prompt and the replies is a neutral one
    gen_msgs = [m for _, m in screen.lines if m.startswith("generating")]
    assert gen_msgs == ["generating 3 replies ..."]
    # in parallel, the two slow contestants overlap in time
    slow, mid = (next(c for c in log if c["tag"] == SECRET[i]) for i in (0, 2))
    assert slow["start"] < mid["end"] and mid["start"] < slow["end"]


def test_label_order_is_random_per_prompt_and_reproduced_on_resume(tmp_path):
    prompts = A.file_prompts(prompts_file(tmp_path, 12))
    arena = A.Arena(contestants(), tmp_path / "ar", seed=4, print_fn=lambda m: None,
                    input_fn=scripted(["a"] * 5 + ["q"]))
    arena.run(prompts)
    orders = {p.id: arena.order_for(p) for p in prompts}
    assert len({tuple(o) for o in orders.values()}) > 1                  # not one fixed order
    key = A.load_key(json.loads(l) for l in (tmp_path / "ar" / "key.jsonl").read_text().splitlines())
    again = A.Arena(contestants(), tmp_path / "ar", seed=4, print_fn=lambda m: None, input_fn=scripted([]))
    assert all(again.order_for(p) == orders[p.id] for p in prompts)
    assert all(key[pid] == orders[pid] for pid in key)
    other_seed = A.Arena(contestants(), tmp_path / "ar3", seed=5, print_fn=lambda m: None,
                         input_fn=scripted([]))
    assert any(other_seed.order_for(p) != orders[p.id] for p in prompts)


def test_reasoning_is_never_shown(tmp_path):
    def thinker(messages, settings, seed):
        return {"text": "<think>first I plan the answer</think>The answer is 4.", "finish_reason": "stop"}

    def unfinished(messages, settings, seed):
        return {"text": "<think>still thinking about it", "finish_reason": "length"}

    screen = Screen()
    cs = [A.generator_contestant("one", thinker, key="k1"), A.generator_contestant("two", unfinished, key="k2")]
    A.Arena(cs, tmp_path / "ar", print_fn=screen, input_fn=scripted(["a"])).run(
        [A.typed_prompt(1, "What is 2+2?")])
    assert "The answer is 4." in screen.text and "(empty reply)" in screen.text
    assert "plan the answer" not in screen.text and "still thinking" not in screen.text
    assert "<think>" not in screen.text


def test_opponent_settings_are_the_harness_presets():
    for name, preset in (("falcon", "recommended"), ("mobilellm", "recommended"), ("mobilellm", "neutral")):
        if not opponents.MODELS[name].path().exists():
            pytest.skip(f"{name} GGUF not present")
        c = A.opponent_contestant(f"{name}={preset}", download=False)
        assert c.settings == opponents.presets(name)[preset]
        assert c.key.startswith(opponents.MODELS[name].key) and c._backend is None   # loaded lazily
    assert A.opponent_contestant("mobilellm=recommended", download=False).settings.strip_think
    sc = A.opponent_contestant("standin-plain")
    assert sc.settings == opponents.common_settings() and sc.name == "standin-plain"


def test_a_real_gguf_opponent_answers_through_the_arena(tmp_path):
    if not opponents.MODELS["falcon"].path().exists():
        pytest.skip("falcon GGUF not present")
    c = A.opponent_contestant("falcon=recommended", download=False, n_threads=2,
                              overrides=["max_tokens=12"])
    screen = Screen()
    arena = A.Arena([c, A.generator_contestant("other", standin("x"), key="kx")], tmp_path / "ar",
                    print_fn=screen, input_fn=scripted(["b"]))
    arena.run([A.typed_prompt(1, "Say hello.")])
    rec = json.loads((tmp_path / "ar" / "votes.jsonl").read_text().splitlines()[0])
    assert rec["vote"] == "B"
    replies = [m for _, m in screen.lines if m.startswith("=== prompt")][0]
    assert "(no reply)" not in replies and "falcon" not in screen.text


# ================================================================== votes and tallies
def test_votes_and_wilson_tallies_are_saved_after_every_vote(tmp_path):
    prompts = A.file_prompts(prompts_file(tmp_path, 5))
    votes = ["a", "b", "t", "x", "c"]
    seen = []

    class Ask:
        i = 0

        def __call__(self, prompt):
            if self.i:
                seen.append(json.loads((tmp_path / "ar" / "tallies.json").read_text())["votes"])
            v = votes[self.i]
            self.i += 1
            return v
    arena = A.Arena(contestants(), tmp_path / "ar", print_fn=lambda m: None, input_fn=Ask())
    arena.run(prompts)
    assert seen == [1, 2, 3, 4]                                  # rewritten after each vote
    t = json.loads((tmp_path / "ar" / "tallies.json").read_text())
    key = A.load_key(json.loads(l) for l in (tmp_path / "ar" / "key.jsonl").read_text().splitlines())
    recs = [json.loads(l) for l in (tmp_path / "ar" / "votes.jsonl").read_text().splitlines()]
    assert [r["vote"] for r in recs] == ["A", "B", A.TIE, A.ALL_BAD, "C"]
    assert all(not any(s in json.dumps(r) for s in SECRET) for r in recs)   # votes hold no identity
    # independent recount
    best = {s: 0 for s in SECRET}
    for r in recs:
        if r["vote"] in "ABC":
            best[key[r["prompt_id"]]["ABC".index(r["vote"])]] += 1
    for s in SECRET:
        row = t["contestants"][s]
        assert row["shown"] == 5 and row["best"] == best[s] and row["ties"] == 1 and row["all_bad"] == 1
        lo, hi = wilson_ref(best[s], 5)
        assert row["best_rate_ci95"] == [round(lo, 4), round(hi, 4)]
    assert sum(best.values()) == 3
    assert (tmp_path / "ar" / "tallies.md").read_text().count("|") > 20


def test_pair_tallies_leave_out_third_party_wins():
    key = {"p1": ["x", "y", "z"], "p2": ["y", "x", "z"], "p3": ["x", "y", "z"], "p4": ["z", "x", "y"]}
    votes = {"p1": {"vote": "A"}, "p2": {"vote": "A"}, "p3": {"vote": "C"}, "p4": {"vote": A.TIE}}
    t = A.tallies(votes, key, ["x", "y", "z"])
    xy = next(p for p in t["pairs"] if (p["a"], p["b"]) == ("x", "y"))
    assert (xy["n"], xy["wins"], xy["losses"], xy["ties"]) == (3, 1, 1, 1)     # p3 left out
    xz = next(p for p in t["pairs"] if (p["a"], p["b"]) == ("x", "z"))
    assert (xz["n"], xz["wins"], xz["losses"], xz["ties"]) == (3, 1, 1, 1)     # p2 left out
    lo, hi = wilson_ref(1.5, 3)
    assert xy["score"] == 0.5 and xy["score_ci95"] == [round(lo, 4), round(hi, 4)]
    assert t["contestants"]["z"]["best"] == 1 and t["contestants"]["y"]["lost"] == 2


def test_resume_skips_voted_prompts_and_continues_typed_numbering(tmp_path):
    pf = prompts_file(tmp_path, 4)
    A.Arena(contestants(), tmp_path / "ar", print_fn=lambda m: None,
            input_fn=scripted(["a", "q"])).run(A.file_prompts(pf))
    log: list = []
    screen = Screen()
    arena = A.Arena(contestants(log=log), tmp_path / "ar", print_fn=screen,
                    input_fn=scripted(["b", "b", "b"]))
    arena.run(A.file_prompts(pf))
    votes = A.effective_votes(json.loads(l) for l in (tmp_path / "ar" / "votes.jsonl").read_text().splitlines())
    assert len(votes) == 4 and "3 prompts to rate (1 votes so far)" in screen.text
    # the prompt answered but not voted in the first session came from the cache: no new call
    assert {c["messages"][-1]["content"] for c in log} == {"Question number 2?", "Question number 3?"}
    # typed prompts continue the numbering
    A.Arena(contestants(), tmp_path / "ar", print_fn=lambda m: None,
            input_fn=scripted(["first typed", "a", ""])).run()
    A.Arena(contestants(), tmp_path / "ar", print_fn=lambda m: None,
            input_fn=scripted(['"""', "two", "lines", '"""', "t"])).run()
    key = A.load_key(json.loads(l) for l in (tmp_path / "ar" / "key.jsonl").read_text().splitlines())
    typed = sorted(k for k in key if k.startswith("typed-"))
    assert [k[:10] for k in typed] == ["typed-0001", "typed-0002"]
    lines = [json.loads(l) for l in (tmp_path / "ar" / "key.jsonl").read_text().splitlines()]
    assert next(r for r in lines if r["prompt_id"] == typed[1])["turns"] == ["two\nlines"]


def test_undo_skip_quit_and_invalid_keys(tmp_path):
    screen = Screen()
    prompts = A.file_prompts(prompts_file(tmp_path, 3))
    # two contestants: 'c' is not a label; 'u' on the first item has nothing to undo; the 'u'
    # on the second item brings the first back at once ('b' re-votes it), the second follows
    # ('s'), then the third ('q')
    arena = A.Arena(contestants(n=2), tmp_path / "ar", print_fn=screen,
                    input_fn=scripted(["c", "z", "u", "a", "u", "b", "s", "q"]))
    stats = arena.run(prompts)
    headers = [m.split(" (")[0] for _, m in screen.lines if m.startswith("=== prompt")]
    assert headers == ["=== prompt 1 of 3", "=== prompt 2 of 3", "=== prompt 1 of 3",
                       "=== prompt 2 of 3", "=== prompt 3 of 3"]
    assert stats == {"voted": 1, "skipped": 1, "undone": 1}
    assert screen.text.count(A.HELP) >= 3 and "nothing to undo" in screen.text
    recs = [json.loads(l) for l in (tmp_path / "ar" / "votes.jsonl").read_text().splitlines()]
    assert [r.get("vote", "retracted") for r in recs] == ["A", "retracted", "B"]
    votes = A.effective_votes(recs)
    assert list(votes) == [prompts[0].id] and votes[prompts[0].id]["vote"] == "B"
    assert json.loads((tmp_path / "ar" / "tallies.json").read_text())["votes"] == 1
    # with reveal the last vote cannot be undone
    screen2 = Screen()
    A.Arena(contestants(n=2), tmp_path / "ar2", reveal=True, print_fn=screen2,
            input_fn=scripted(["a", "u", "q"])).run(prompts)
    assert "votes are final once revealed" in screen2.text


def test_multi_turn_prompts_feed_each_contestant_its_own_reply(tmp_path):
    p = tmp_path / "two.jsonl"
    p.write_text(json.dumps({"id": "conv-1", "turns": ["Name a fruit.", "And its colour?"]}) + "\n")
    log: list = []
    screen = Screen()
    A.Arena(contestants(log=log), tmp_path / "ar", print_fn=screen, input_fn=scripted(["a"])).run(
        A.file_prompts(p))
    second = [c for c in log if len(c["messages"]) == 3]
    assert len(second) == 3
    for c in second:
        tag = c["tag"].lower()[-3:]
        assert c["messages"][1] == {"role": "assistant", "content": f"reply {tag} to: Name a fruit."}
    assert screen.text.count("##### turn") == 2


def test_a_failing_contestant_shows_no_reply_and_is_logged(tmp_path):
    def broken(messages, settings, seed):
        raise RuntimeError("SECRET-THREE exploded")

    screen = Screen()
    cs = contestants(n=3, gens={2: broken})
    A.Arena(cs, tmp_path / "ar", print_fn=screen, input_fn=scripted(["x"])).run([A.typed_prompt(1, "Hi")])
    assert "(no reply)" in screen.text and "exploded" not in screen.text
    assert "SECRET-THREE exploded" in (tmp_path / "ar" / "errors.log").read_text()


def test_an_existing_arena_refuses_other_contestants(tmp_path):
    A.Arena(contestants(), tmp_path / "ar", print_fn=lambda m: None, input_fn=scripted([]))
    with pytest.raises(A.ArenaError):
        A.Arena(contestants(n=2), tmp_path / "ar", print_fn=lambda m: None, input_fn=scripted([]))
    with pytest.raises(A.ArenaError):
        A.Arena(contestants(), tmp_path / "ar", seed=9, print_fn=lambda m: None, input_fn=scripted([]))
    with pytest.raises(A.ArenaError):
        A.Arena(contestants(n=1), tmp_path / "ar1", print_fn=lambda m: None, input_fn=scripted([]))


def test_prompt_file_formats(tmp_path):
    txt = A.file_prompts(prompts_file(tmp_path, 2))
    assert [p.turns for p in txt] == [("Question number 0?",), ("Question number 1?",)]
    j = tmp_path / "p.jsonl"
    j.write_text(json.dumps({"prompt": "One?"}) + "\n" + json.dumps({"id": "x", "turns": ["A", "B"]}) + "\n")
    got = A.file_prompts(j)
    assert got[0].turns == ("One?",) and got[1].id == "x" and got[1].multi_turn
    dev = A.file_prompts("proxy:dev")
    assert len(dev) == 200 and all(p.split == "dev" for p in dev)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"turns": []}\n')
    with pytest.raises(A.ArenaError):
        A.file_prompts(bad)
    with pytest.raises(A.ArenaError):
        A.file_prompts(tmp_path / "missing.txt")


def test_the_command_line_with_stand_ins(tmp_path):
    spec = importlib.util.spec_from_file_location("arena_cli", REPO / "scripts" / "arena.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out: list[str] = []
    pf = prompts_file(tmp_path, 2)
    rc = cli.main(["--ours", "standin-plain", "--opponent", "standin-verbose", "--opponent", "standin-loop",
                   "--prompts", str(pf), "--root", str(tmp_path / "root"), "--reveal"],
                  input_fn=scripted(["a", "b"]), print_fn=out.append)
    assert rc == 0
    text = "\n".join(out)
    assert text.count("revealed:") == 2 and "# Arena tallies of revealed sessions (NOT blind" in text
    assert "# Blind arena tallies" not in text
    # a revealed session has an arena of its own, never the blind one's
    assert not (tmp_path / "root" / "default").exists()
    t = json.loads((tmp_path / "root" / "default-revealed" / "tallies_revealed.json").read_text())
    assert t["votes"] == 2 and t["blind"] is False
    assert set(t["contestants"]) == {"ours (standin-plain)", "standin-verbose", "standin-loop"}
    with pytest.raises(SystemExit):
        cli.main(["--prompts", str(pf), "--root", str(tmp_path / "r2")], input_fn=scripted([]),
                 print_fn=out.append)


# ============================================ blind and revealed sessions (review eval-3)
def _votes(d: Path) -> list[dict]:
    return [json.loads(l) for l in (d / "votes.jsonl").read_text().splitlines()]


def test_a_revealed_session_never_joins_a_blind_arena(tmp_path):
    """A blind session, then a revealed one and one with ours at another temperature, in the same
    directory (before: all five votes were pooled into the 'Blind arena tallies', no vote said
    whether it was revealed, and arena.json kept the first session's settings)."""
    d = tmp_path / "default"
    prompts = [A.typed_prompt(i, f"Question {i}?") for i in range(1, 6)]
    blind = A.Arena(contestants(n=2), d, print_fn=lambda m: None, input_fn=scripted(["a", "b"]))
    blind.run(prompts[:2])
    with pytest.raises(A.ArenaError, match="reveal"):
        A.Arena(contestants(n=2), d, reveal=True, print_fn=lambda m: None, input_fn=scripted(["a"]))
    hot = contestants(n=2)
    hot[0].settings = hot[0].settings.replace(temperature=1.5)          # same cache key, other settings
    with pytest.raises(A.ArenaError, match="contestants"):
        A.Arena(hot, d, print_fn=lambda m: None, input_fn=scripted(["b"]))
    recs = _votes(d)
    assert len(recs) == 2 and all(r["reveal"] is False for r in recs)
    assert json.loads((d / "tallies.json").read_text())["votes"] == 2
    setup = json.loads((d / "arena.json").read_text())
    assert setup["reveal"] is False and all(c["settings_hash"] for c in setup["contestants"])
    # the same session again is welcome
    A.Arena(contestants(n=2), d, print_fn=lambda m: None, input_fn=scripted(["t"])).run(prompts[2:3])
    assert json.loads((d / "tallies.json").read_text())["votes"] == 3


def test_blind_tallies_count_blind_votes_only(tmp_path):
    d = tmp_path / "ar"
    prompts = A.file_prompts(prompts_file(tmp_path, 3))
    A.Arena(contestants(), d, print_fn=lambda m: None, input_fn=scripted(["a", "b"])).run(prompts[:2])
    # a vote that says it was made with identities revealed, or that does not say, is never blind
    from bes.eval.proxy.common import append_jsonl
    append_jsonl(d / "votes.jsonl", [{"prompt_id": prompts[2].id, "vote": "C", "labels": ["A", "B", "C"],
                                      "reveal": True},
                                     {"prompt_id": "typed-0009-x", "vote": "A", "labels": ["A", "B", "C"]}])
    arena = A.Arena(contestants(), d, print_fn=lambda m: None, input_fn=scripted([]))
    t = arena.save_tallies()
    assert t["votes"] == 2 and t["blind"] is True
    assert sum(r["best"] for r in t["contestants"].values()) == 2
    votes = A.effective_votes(_votes(d))
    assert set(A.blind_votes(votes)) == {prompts[0].id, prompts[1].id}
    assert set(A.blind_votes(votes, reveal=True)) == {prompts[2].id}


def test_an_arena_of_the_first_format_is_refused(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    (d / "arena.json").write_text(json.dumps({"format": 1, "seed": 0, "contestants": [
        {"name": c.name, "key": c.key} for c in contestants(n=2)]}))
    with pytest.raises(A.ArenaError, match="revealed"):
        A.Arena(contestants(n=2), d, print_fn=lambda m: None, input_fn=scripted([]))


def test_the_command_line_keeps_blind_and_revealed_sessions_apart(tmp_path):
    spec = importlib.util.spec_from_file_location("arena_cli2", REPO / "scripts" / "arena.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    pf = prompts_file(tmp_path, 4)
    common = ["--ours", "standin-plain", "--opponent", "standin-verbose", "--prompts", str(pf),
              "--root", str(tmp_path / "root"), "--name", "final"]
    assert cli.main(common, input_fn=scripted(["a", "b", "q"]), print_fn=lambda m: None) == 0
    assert cli.main(common + ["--reveal"], input_fn=scripted(["a", "a", "a", "a"]), print_fn=lambda m: None) == 0
    blind = json.loads((tmp_path / "root" / "final" / "tallies.json").read_text())
    revealed = json.loads((tmp_path / "root" / "final-revealed" / "tallies_revealed.json").read_text())
    assert (blind["votes"], blind["blind"]) == (2, True) and (revealed["votes"], revealed["blind"]) == (4, False)
    assert all(not r["reveal"] for r in _votes(tmp_path / "root" / "final"))
    assert all(r["reveal"] for r in _votes(tmp_path / "root" / "final-revealed"))
    # another setting for ours in the blind arena is refused instead of pooled
    with pytest.raises(A.ArenaError):
        cli.main(common + ["--temperature", "1.5"], input_fn=scripted(["a"]), print_fn=lambda m: None)
    with pytest.raises(SystemExit):
        cli.main(common[:-1] + ["final-revealed"], input_fn=scripted([]), print_fn=lambda m: None)
