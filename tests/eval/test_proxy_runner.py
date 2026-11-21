"""Runner: cache and resumability, multi-turn protocol, candidate selection, settings, pool."""
from __future__ import annotations

import json

import pytest

from bes.eval.proxy import runner
from bes.eval.proxy.common import (GenSettings, Reply, append_jsonl, derive_seed, read_jsonl,
                                   select_candidate, strip_think)
from bes.eval.proxy.prompts import Prompt


class Recorder:
    """Deterministic test backend that records every call."""

    key = "recorder"

    def __init__(self, fail_after=None, loop_best=False):
        self.calls = []
        self.fail_after = fail_after
        self.loop_best = loop_best

    def generate(self, messages, settings, seed):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.append(([dict(m) for m in messages], seed))
        lp = -1.0 - (seed % 7)
        text = f"reply to {messages[-1]['content']!r} seed {seed % 1000}"
        if self.loop_best and seed % 7 == 0:
            text = "best but " + "looping again " * 30
        return Reply(text=text, raw_text=text, completion_tokens=4, logprob_sum=lp * 4, logprob_count=4,
                     finish_reason="stop")


PROMPTS = [Prompt(id=f"chat-{i:03d}", category="chat",
                  turns=(f"q{i}",) if i % 3 else (f"q{i}", f"follow-up {i}"), split="dev")
           for i in range(9)]


def spec(settings=None, label="t", seed=0):
    return runner.RunSpec(backend={"kind": "test"}, key="recorder", settings=settings or GenSettings(),
                          label=label, seed=seed)


def n_generations(prompts, samples=1):
    return sum(len(p.turns) for p in prompts) * samples


def test_run_writes_outputs_and_resumes_from_cache(tmp_path):
    rec = Recorder()
    r1 = runner.run(PROMPTS, spec(), tmp_path, backend=rec, log=None)
    assert r1.done == len(PROMPTS) and r1.new_generations == n_generations(PROMPTS) == len(rec.calls)
    records, meta = runner.load_outputs(tmp_path, r1.run_id)
    assert set(records) == {p.id for p in PROMPTS} and meta["n_prompts"] == len(PROMPTS)
    assert len(read_jsonl(runner.cache_path(tmp_path, "recorder"))) == n_generations(PROMPTS)
    rec2 = Recorder()
    r2 = runner.run(PROMPTS, spec(), tmp_path, backend=rec2, log=None)
    assert rec2.calls == [] and r2.reused == len(PROMPTS) and r2.new_generations == 0
    assert runner.load_outputs(tmp_path, r2.run_id)[0] == records
    assert r1.run_id in runner.list_runs(tmp_path)


def test_crash_then_resume_reuses_finished_work(tmp_path):
    r1 = runner.run(PROMPTS, spec(), tmp_path, backend=Recorder(fail_after=5), log=None)
    assert r1.failed and r1.new_generations == 5
    healthy = Recorder()
    r2 = runner.run(PROMPTS, spec(), tmp_path, backend=healthy, log=None)
    assert r2.done == len(PROMPTS) and not r2.failed
    assert len(healthy.calls) == n_generations(PROMPTS) - 5
    # identical to an uninterrupted run
    fresh = tmp_path / "fresh"
    r3 = runner.run(PROMPTS, spec(), fresh, backend=Recorder(), log=None)
    assert runner.load_outputs(fresh, r3.run_id)[0] == runner.load_outputs(tmp_path, r2.run_id)[0]


def test_deadline_stops_new_generations(tmp_path):
    r = runner.run(PROMPTS, spec(), tmp_path, backend=Recorder(), minutes=1e-9, log=None)
    assert r.done == 0 and r.out_of_time == len(PROMPTS)
    assert runner.load_outputs(tmp_path, r.run_id)[0] == {}
    r2 = runner.run(PROMPTS, spec(), tmp_path, backend=Recorder(), log=None)
    assert r2.done == len(PROMPTS)


def test_truncated_cache_line_is_skipped(tmp_path):
    runner.run(PROMPTS[:2], spec(), tmp_path, backend=Recorder(), log=None)
    path = runner.cache_path(tmp_path, "recorder")
    with open(path, "a") as f:
        f.write('{"key": "half-written')
    append_jsonl(path, [{"key": "k", "reply": {"text": "x"}}])
    rows = read_jsonl(path)
    assert rows[-1]["key"] == "k" and all("half" not in r["key"] for r in rows)


def test_each_model_sees_its_own_previous_answer(tmp_path):
    rec = Recorder()
    settings = GenSettings(system_prompt="Be brief.")
    r = runner.run(PROMPTS[:1], spec(settings), tmp_path, backend=rec, log=None)
    first, second = rec.calls
    assert [m["role"] for m in first[0]] == ["system", "user"]
    record = runner.load_outputs(tmp_path, r.run_id)[0][PROMPTS[0].id]
    assert second[0] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q0"},
                         {"role": "assistant", "content": record["turns"][0]["reply"]},
                         {"role": "user", "content": "follow-up 0"}]
    assert first[1] == derive_seed(0, PROMPTS[0].id, 0, 0) and second[1] == derive_seed(0, PROMPTS[0].id, 1, 0)


def test_best_of_k_by_logprob(tmp_path):
    settings = GenSettings(samples=4, select="logprob_mean")
    rec = Recorder()
    r = runner.run(PROMPTS, spec(settings), tmp_path, backend=rec, log=None)
    assert len(rec.calls) == n_generations(PROMPTS, 4)
    records = runner.load_outputs(tmp_path, r.run_id)[0]
    for p in PROMPTS:
        seeds = [derive_seed(0, p.id, 0, k) for k in range(4)]
        best = max(range(4), key=lambda k: (-1.0 - seeds[k] % 7, -k))
        assert records[p.id]["turns"][0]["selected"] == best
    # a K=1 run of the same arm reuses sample 0 of the K=4 run (same seeds, same cache keys)
    rec1 = Recorder()
    runner.run(PROMPTS, spec(GenSettings(), label="k1"), tmp_path, backend=rec1, log=None)
    single_turn_calls = [c for c in rec1.calls if len(c[0]) == 1]
    assert single_turn_calls == []


def test_filtered_selection_skips_degenerate(tmp_path):
    cands = [Reply(text="fine answer.", logprob_sum=-4.0, logprob_count=4),
             Reply(text="x " + "again and again " * 30, logprob_sum=-0.4, logprob_count=4),
             Reply(text="cut", logprob_sum=-0.1, logprob_count=4, finish_reason="length")]
    from bes.eval.proxy.runner import _is_degenerate
    assert select_candidate(cands, "logprob_mean") == 1
    assert select_candidate(cands, "filtered_logprob_mean", _is_degenerate) == 0
    assert select_candidate(cands, "first") == 0
    tie = [Reply(text="a", logprob_sum=-1.0, logprob_count=1), Reply(text="b", logprob_sum=-1.0, logprob_count=1)]
    assert select_candidate(tie, "logprob_mean") == 0
    nolp = [Reply(text="a"), Reply(text="b", logprob_sum=-9.0, logprob_count=1)]
    assert select_candidate(nolp, "logprob_mean") == 1
    with pytest.raises(ValueError):
        GenSettings(select="best")


CONTROL_CASES = [
    # (user message, candidates with their mean log-probability; candidate 0 is the most likely,
    #  the index a reader must keep under our selector)
    ("Answer in exactly one sentence: why do birds migrate?",
     [("Birds migrate to find food. They also look for safe places to breed.", -0.30),
      ("Birds migrate to reach places with more food and better weather for raising their young.", -0.55)], 1),
    ('Describe your ideal weekend without using the word "very".',
     [("My ideal weekend is very relaxing: a long walk, a good book and dinner with friends.", -0.35),
      ("My ideal weekend is calm: a long walk, a good book and dinner with friends.", -0.60)], 1),
    ("A pack has 6 cans. How many cans are in 4 packs?",
     [("4 packs have 6 + 4 = 10 cans.", -0.20),
      ("Each pack has 6 cans, so 4 packs have 6 x 4 = 24 cans.", -0.40),
      ("There are 6 x 4 = 24 cans in 4 packs.", -0.45)], 1),
]


@pytest.mark.parametrize("user, cands, want", CONTROL_CASES, ids=["one-sentence", "forbidden-word", "math-vote"])
def test_the_selector_matched_control_selects_as_our_decoder_does(user, cands, want):
    """`decoder_select` keeps what our selector keeps: the compliant candidate (exactly one
    sentence; no forbidden word) and the majority answer; `filtered_logprob_mean` drops only
    degenerate candidates, so it keeps the more likely candidate 0 (review eval-4)."""
    from bes.decode import select as S
    from bes.eval.proxy.runner import _is_degenerate
    from bes.exact import constraints as C

    replies = [Reply(text=t, finish_reason="stop", completion_tokens=20, logprob_sum=lp * 20, logprob_count=20)
               for t, lp in cands]
    assert select_candidate(replies, "filtered_logprob_mean", _is_degenerate) == 0
    assert select_candidate(replies, "decoder_select", _is_degenerate, user=user) == want
    direct = S.select([S.Candidate(t, 1.0, "stop", 20) for t, _ in cands], C.compile(user), user)
    assert direct.index == want


class Scripted:
    """K scripted candidates per user message, in sampling order (candidate k = k-th call)."""

    key = "scripted"

    def __init__(self, script):
        self.script, self.calls = script, {}

    def generate(self, messages, settings, seed):
        user = messages[-1]["content"]
        k = self.calls.get(user, 0)
        self.calls[user] = k + 1
        text, lp = self.script[user][k % len(self.script[user])]
        return Reply(text=text, finish_reason="stop", completion_tokens=10, logprob_sum=lp * 10, logprob_count=10)


def test_decoder_select_arm_shares_the_candidates_and_is_part_of_the_run_id(tmp_path):
    ps = [Prompt(id=f"instruct-{i:03d}", category="instruct", turns=(user,), split="dev")
          for i, (user, _, _) in enumerate(CONTROL_CASES)]
    script = {user: cands for user, cands, _ in CONTROL_CASES}
    k = max(len(c) for c in script.values())
    lp_run = runner.run(ps, spec(GenSettings(samples=k, select="logprob_mean"), label="ctl"), tmp_path,
                        backend=Scripted(script), log=None)
    again = Scripted(script)
    sel_run = runner.run(ps, spec(GenSettings(samples=k, select="decoder_select"), label="ctl"), tmp_path,
                         backend=again, log=None)
    assert again.calls == {} and sel_run.run_id != lp_run.run_id      # same cached candidates
    lp_out = runner.load_outputs(tmp_path, lp_run.run_id)[0]
    sel_out = runner.load_outputs(tmp_path, sel_run.run_id)[0]
    for p, (_, _, want) in zip(ps, CONTROL_CASES):
        assert lp_out[p.id]["turns"][0]["selected"] == 0
        assert sel_out[p.id]["turns"][0]["selected"] == want


class FakeClock:
    """``time`` for the runner module: a clock that moves only when the backend generates."""

    def __init__(self):
        import time as real
        self.real, self.now = real, 1_000_000.0

    def time(self):
        return self.now

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_a_run_stopped_by_its_deadline_covers_every_category(tmp_path, monkeypatch):
    """The 24-hour run's select and judge stages answer the prompts of `select` until their
    deadline (review eval-6: before, a sliced held-out run answered chat, math and a little code
    only). Each generation takes one second of a fake clock; 60 seconds cover about a quarter of
    the held-out half, every category in proportion (within one prompt)."""
    from collections import Counter

    from bes.eval.proxy import prompts as P

    clock = FakeClock()
    monkeypatch.setattr(runner, "time", clock)

    class OneSecond(Recorder):
        def generate(self, messages, settings, seed):
            clock.now += 1.0
            return super().generate(messages, settings, seed)

    ps = P.load()
    todo = P.select(ps, "test")
    r = runner.run(todo, spec(label="sliced"), tmp_path, backend=OneSecond(), minutes=1.0, all_prompts=ps, log=None)
    assert 40 <= r.done < len(todo) and r.out_of_time > 0
    done = runner.load_outputs(tmp_path, r.run_id)[0]
    assert set(done) == {p.id for p in todo[:r.done]}                 # a prefix of the order
    size = Counter(p.category for p in todo)
    got = Counter(rec["category"] for rec in done.values())
    assert r.done_by_category == dict(sorted(got.items())) and r.of_by_category == dict(sorted(size.items()))
    for c in size:
        assert abs(got[c] - r.done * size[c] / len(todo)) < 1, (c, got)


def test_settings_hashing_and_overrides():
    s = GenSettings(extra={"particles": 6, "use_library": True})
    assert s.extra == (("particles", 6), ("use_library", True))
    assert s.hash() == GenSettings(extra={"use_library": True, "particles": 6}).hash()
    assert s.hash() != s.replace(temperature=0.7).hash()
    assert "samples" not in s.core_dict() and "select" not in s.core_dict()
    o = s.with_overrides(["temperature=0.3", "max_tokens=64", "extra.particles=2", "system_prompt=none", "strip_think=true"])
    assert (o.temperature, o.max_tokens, o.extra_dict["particles"], o.system_prompt, o.strip_think) == (0.3, 64, 2, None, True)
    with pytest.raises(KeyError):
        s.with_overrides(["temp=1"])
    assert GenSettings.from_dict(json.loads(json.dumps(s.as_dict()))) == s
    msgs = [{"role": "user", "content": "hi"}]
    k1 = runner.gen_key("b", msgs, s, 5)
    assert k1 == runner.gen_key("b", msgs, s.replace(samples=3, select="logprob_mean"), 5)
    assert k1 != runner.gen_key("b", msgs, s, 6) and k1 != runner.gen_key("c", msgs, s, 5)


def test_derive_seed_and_strip_think():
    a = derive_seed(0, "chat-001", 0, 0)
    assert a == derive_seed(0, "chat-001", 0, 0) and 0 <= a < 2 ** 31
    assert len({derive_seed(0, "chat-001", t, k) for t in range(2) for k in range(4)}) == 8
    assert derive_seed(1, "chat-001", 0, 0) != a
    # "<think>hmm</think>" is 7 + 3 + 8 = 18 characters
    assert strip_think("<think>hmm</think> The answer is 4.") == ("The answer is 4.", {"think_blocks": 1, "think_unclosed": False, "think_chars": 18})
    text, info = strip_think("<think>still thinking about it")
    assert text == "" and info["think_unclosed"]
    assert strip_think("reasoning</think>Answer")[0] == "Answer"
    assert strip_think("No reasoning here.")[0] == "No reasoning here."


def test_process_pool_matches_in_process(tmp_path):
    backend = {"kind": "callable", "factory": "bes.eval.proxy.run_ours:standin", "kwargs": {"style": "plain"},
               "key": "standin-plain"}
    s = runner.RunSpec(backend=backend, key="standin-plain", settings=GenSettings(samples=2, select="logprob_mean"),
                       label="pool")
    r_pool = runner.run(PROMPTS, s, tmp_path / "pool", workers=2, log=None)
    r_seq = runner.run(PROMPTS, s, tmp_path / "seq", workers=0, log=None)
    assert r_pool.done == r_seq.done == len(PROMPTS)

    def strip(records):
        return {pid: [{k: v for k, v in t.items() if k != "latency_s"} for t in r["turns"]] for pid, r in records.items()}

    assert strip(runner.load_outputs(tmp_path / "pool", r_pool.run_id)[0]) == strip(runner.load_outputs(tmp_path / "seq", r_seq.run_id)[0])
