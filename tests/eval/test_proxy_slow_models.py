"""Real models on CPU (slow; skipped when the GGUF files are not in data_cache/opponents)."""
from __future__ import annotations

import pytest

from bes.eval.proxy import opponents as O
from bes.eval.proxy import prompts as P
from bes.eval.proxy import runner

pytestmark = pytest.mark.slow


def _need(name):
    entry = O.MODELS[name]
    if not entry.source_path().exists():
        pytest.skip(f"{entry.filename} not present (python -m bes.eval.proxy fetch {name})")
    pytest.importorskip("llama_cpp")
    return O.fetch(name, download=False, log=lambda _m: None)


def test_falcon_three_prompts(tmp_path):
    from bes.eval.proxy.run_gguf import GGUFBackend, gguf_spec

    path = _need("falcon")
    backend = GGUFBackend(path, n_ctx=2048, n_threads=2)
    prompt_text = backend.render([{"role": "user", "content": "Hi"}])
    assert prompt_text.endswith("<|im_start|>assistant\n")
    by_id = {p.id: p for p in P.load()}
    chosen = [by_id["chat-001"], by_id["math-001"], by_id["chat-004"]]   # chat-004 has two turns
    assert chosen[2].multi_turn
    settings = O.presets("falcon")["recommended"].replace(max_tokens=160)
    spec = runner.RunSpec(backend=gguf_spec(path, n_ctx=2048, n_threads=2, key="falcon-test"),
                          key="falcon-test", settings=settings, label="slow")
    result = runner.run(chosen, spec, tmp_path, backend=backend, log=None)
    assert result.done == 3 and not result.failed
    records, _ = runner.load_outputs(tmp_path, result.run_id)
    for p in chosen:
        for turn in records[p.id]["turns"]:
            assert turn["reply"].strip()
            assert turn["completion_tokens"] <= 160
            assert -10 < turn["logprob_mean"] < 0
            assert turn["finish_reason"] in ("stop", "length")
    # the second turn was generated from the model's own first reply
    two = records["chat-004"]
    msgs = [{"role": "user", "content": chosen[2].turns[0]},
            {"role": "assistant", "content": two["turns"][0]["reply"]},
            {"role": "user", "content": chosen[2].turns[1]}]
    from bes.eval.proxy.common import derive_seed
    key = runner.gen_key("falcon-test", msgs, settings, derive_seed(0, "chat-004", 1, 0))
    assert two["turns"][1]["gen_key"] == key
    # same seed, same reply
    again = backend.generate([{"role": "user", "content": chosen[0].turns[0]}], settings,
                             derive_seed(0, "chat-001", 0, 0))
    assert again.text == records["chat-001"]["turns"][0]["reply"]


def test_mobilellm_budget_forcing():
    from bes.eval.proxy.run_gguf import THINK_CLOSE, GGUFBackend

    path = _need("mobilellm")
    backend = GGUFBackend(path, n_ctx=2048, n_threads=2)
    base = O.presets("mobilellm")["recommended"]
    forced = backend.generate([{"role": "user", "content": "Why is the sky blue?"}],
                              base.replace(max_tokens=96, extra={"think_budget": 48}), 11)
    assert forced.extra.get("forced_answer") and THINK_CLOSE.strip() in forced.raw_text
    assert forced.text and "<think>" not in forced.text
    assert forced.completion_tokens <= 96 + len(backend.tokenize(THINK_CLOSE))
    plain = backend.generate([{"role": "user", "content": "Why is the sky blue?"}],
                             base.replace(max_tokens=48, extra={}), 11)
    assert plain.finish_reason == "length" and plain.text == "" and plain.extra["think_unclosed"]
    # the first 48 tokens are the same sample in both calls
    assert forced.raw_text.startswith(plain.raw_text)


def test_small_judge_probabilities_and_prefix_reuse():
    from bes.eval.proxy.judge import GGUFJudge, judge_pair, render_messages

    path = _need("qwen2.5-0.5b")
    judge = GGUFJudge(path, n_ctx=4096, n_threads=2)
    pair = {"pair_id": "p", "prompt_id": "q", "category": "math", "multi_turn": False,
            "turns": [{"user": "What is 2 + 2?", "A": "2 + 2 = 4.", "B": "2 + 2 = 5."}]}
    m1, _ = render_messages(pair["turns"])
    fresh = judge.probs(m1)
    other, _ = render_messages([{"user": "Name a color.", "A": "Blue.", "B": "Seven."}])
    judge.probs(other)                      # shares the instruction prefix with m1
    reused = judge.probs(m1)
    for opt in ("A", "B", "C"):
        assert reused[opt] == pytest.approx(fresh[opt], abs=2e-3)
    assert sum(fresh[o] for o in ("A", "B", "C")) == pytest.approx(1.0)
    assert 0 < fresh["mass"] <= 1.0
    rec = judge_pair(judge, pair)
    assert rec["verdict"] in ("A", "B", "tie")
