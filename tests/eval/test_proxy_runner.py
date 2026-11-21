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
