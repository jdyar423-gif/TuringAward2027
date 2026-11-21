"""Tests of the E4 probes and E5 metrics (``bes/eval/probes.py``) and of the probe script's
report logic. Task generators are checked against independent checkers: the count probe with
``bes.exact.constraints.measure``, the arithmetic probe with ``bes.exact.calc`` and Python
integers; decoding against ``bes.model.generate.generate``."""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from bes.config import ASSISTANT, BOS, CALC, CALC_END, END, EQ, USER, ModelCfg
from bes.eval import probes as P
from bes.tokenizer import load as load_tok

REPO = Path(__file__).resolve().parents[2]
TINY = ModelCfg(d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16, n_loops=2, max_seq=256,
                clock_planes=3, cand_dim=8, critic_hidden=8, table_grad="take")


@pytest.fixture(scope="module")
def tok():
    return load_tok()


def _sub(seq, part) -> bool:
    n = len(part)
    return any(list(seq[i:i + n]) == list(part) for i in range(len(seq) - n + 1))


# -------------------------------------------------------------------- generators
def test_copy_items(tok):
    items = P.eval_items("copy", 24, tok)
    for it in items:
        code = it.meta["code_ids"]
        assert len(code) == P.COPY_LEN == len(set(code))
        assert it.answer_ids[-P.COPY_LEN:] == code and _sub(it.prompt_ids, code)
        assert it.prompt_ids[0] == BOS and it.prompt_ids[1] == USER and it.prompt_ids[-1] == ASSISTANT
        assert P.score(it, it.answer_ids + [END], tok)
        assert not P.score(it, it.answer_ids, tok)                         # no END: not done
        wrong = list(it.answer_ids)
        wrong[-3], wrong[-2] = wrong[-2], wrong[-3]
        assert not P.score(it, wrong + [END], tok)
        assert not P.score(it, it.answer_ids + it.answer_ids[-1:] + [END], tok)
    assert len({tuple(it.meta["code_ids"]) for it in items}) == len(items)


def test_count_items_against_measure(tok):
    from bes.exact import constraints as C

    items = P.eval_items("count", 32, tok)
    assert sorted({it.meta["n"] for it in items}) == list(range(1, 9))
    assert {it.meta["unit"] for it in items} == {"items", "sentences"}
    for it in items:
        n, unit = it.meta["n"], it.meta["unit"]
        text = P.display(it.answer_ids, tok)
        m = C.measure(text)                                   # the independent count
        assert m["items" if unit == "items" else "sentences"] == n
        spec = C.compile(it.user)                             # the user turn compiles to N
        iv = spec.items if unit == "items" else spec.sentences
        assert iv is not None and iv.lo == iv.hi == n
        assert P.score(it, it.answer_ids + [END], tok)
        if unit == "items":                                    # one item more or less fails
            more = text + "\n- extra"
            assert C.measure(more)["items"] == n + 1
            assert not P.score(it, tok.encode(more) + [END], tok)
            if n > 1:
                fewer = text.rsplit("\n", 1)[0]
                assert not P.score(it, tok.encode(fewer) + [END], tok)
        else:
            more = text + " The end is near."
            assert not P.score(it, tok.encode(more) + [END], tok)


def test_fence_items_against_an_independent_checker(tok):
    items = P.eval_items("fence", 40, tok)
    assert {it.meta["blocks"] for it in items} == {1, 2}
    tags = set()
    for it in items:
        text = P.display(it.answer_ids, tok)
        fences = [ln for ln in text.split("\n") if ln.startswith("```")]
        tags |= {ln[3:] for ln in fences}
        assert len(fences) == 2 * it.meta["blocks"]
        assert P.score(it, it.answer_ids + [END], tok)
        head = text[: text.rindex("```")].rstrip("\n")            # drop the last closing fence
        assert not P.score(it, tok.encode(head) + [END], tok)
    assert "" in tags and "python" in tags          # some fences carry no language tag


def test_arith_items_against_calc_and_python(tok):
    from bes.exact import calc

    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    for calc_on in (True, False):
        for it in P.eval_items("arith", 30, tok, calc=calc_on):
            a, op, b = it.meta["expr"].split()
            assert it.meta["expected"] == str(ops[op](int(a), int(b))) == calc.evaluate(it.meta["expr"])
            assert 10 <= int(a) < 10000 and 10 <= int(b) < 10000
            text = P.display(it.answer_ids, tok)
            assert text == f"The answer is {it.meta['expr']} = {it.meta['expected']}."
            assert (CALC in it.answer_ids) == calc_on and (EQ in it.answer_ids) == calc_on
            assert P.score(it, it.answer_ids + [END], tok)
            bad = text.replace(f"= {it.meta['expected']}", f"= {int(it.meta['expected']) + 1}")
            assert not P.score(it, tok.encode(bad) + [END], tok)
            if calc_on:   # the result tokens and CALC_END are what the host forces
                i = it.answer_ids.index(EQ)
                j = it.answer_ids.index(CALC_END)
                assert it.answer_ids[i + 1:j] == tok.encode(it.meta["expected"])


def test_probe_pools_mix_all_tasks(tok, tmp_path):
    from bes.eval import microbench as mb

    calc_pool = mb.get_pool("probe_calc", "train", 256, 8, tok, root=tmp_path)
    plain_pool = mb.get_pool("probe_plain", "train", 256, 8, tok, root=tmp_path)
    text = tok.decode([int(t) for t in np.asarray(calc_pool.tokens).ravel()])
    for marker in ("Remember this code:", "bullet point", "```", "What is"):
        assert marker in text
    assert (np.asarray(calc_pool.tokens) == CALC).any() and not (np.asarray(plain_pool.tokens) == CALC).any()
    # forced result tokens carry no loss weight; the expression does
    row = np.asarray(calc_pool.tokens[0], np.int64)
    w = np.asarray(calc_pool.w[0])
    eqs = np.flatnonzero(row[1:] == EQ)
    if eqs.size:
        k = eqs[0]                                    # target position of EQ
        assert w[k] == 1.0 and w[k + 1] == 0.0


# ------------------------------------------------------------------ statistics
def test_wilson_interval_known_values():
    assert P.wilson(0, 10) == pytest.approx((0.0, 0.27754), abs=1e-4)
    assert P.wilson(5, 10) == pytest.approx((0.23659, 0.76341), abs=1e-4)
    assert P.wilson(10, 10) == pytest.approx((0.72246, 1.0), abs=1e-4)
    assert P.wilson(0, 0) == (0.0, 1.0)


def test_probe_verdict_rule():
    def s(acc):
        return {"acc": acc}
    assert P.probe_verdict(s(0.95), s(0.30)) == "pass"
    assert P.probe_verdict(s(0.90), s(0.30)).startswith("fail")          # on must exceed 90%
    assert P.probe_verdict(s(0.95), s(0.50)).startswith("fail")          # off must stay below 50%
    assert P.probe_verdict(None, s(0.1)) == "incomplete"
    results = [{"ok": True, "meta": {"n": 1}}, {"ok": False, "meta": {"n": 1}}, {"ok": True, "meta": {"n": 2}}]
    summ = P.probe_summary(results)
    assert summ["correct"] == 2 and summ["by_n"] == {"1": [1, 2], "2": [1, 1]}


# ---------------------------------------------------------------- E5 metrics
def _rep_brute(x, n=4):
    total = len(x) - n + 1
    if total <= 0:
        return 0.0
    rep = sum(1 for i in range(total) if tuple(x[i:i + n]) in {tuple(x[j:j + n]) for j in range(i)})
    return rep / total


def _copy_brute(gen, prompt, k):
    covered = [False] * len(gen)
    for s in range(len(gen)):
        for e in range(s + k, len(gen) + 1):
            if _sub(prompt, gen[s:e]):
                for i in range(s, e):
                    covered[i] = True
    return sum(covered) / len(gen) if gen else 0.0


def test_repeated_ngram_rate_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, 3, int(rng.integers(0, 30))).tolist()
        assert P.repeated_ngram_rate(x) == pytest.approx(_rep_brute(x))
    assert P.repeated_ngram_rate([1, 2, 3, 4] * 5) == pytest.approx(13 / 17)   # 17 4-grams, 4 distinct


def test_copied_share_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(150):
        prompt = rng.integers(0, 3, int(rng.integers(0, 16))).tolist()
        gen = rng.integers(0, 3, int(rng.integers(0, 14))).tolist()
        k = int(rng.integers(1, 5))
        assert P.copied_share(gen, prompt, k) == pytest.approx(_copy_brute(gen, prompt, k))
    assert P.copied_share(list(range(10)) + [99], list(range(20)), 8) == pytest.approx(10 / 11)
    assert P.copied_share(list(range(7)), list(range(20)), 8) == 0.0


def test_loop_detector_on_constructed_sequences():
    head = list(range(100, 140))
    loop3 = head + [1, 2, 3] * 12                       # 36 tokens of period 3 at the end
    assert P.ends_in_loop(loop3, stopped=False)
    assert not P.ends_in_loop(loop3, stopped=True)      # it ended: not absorbing
    assert not P.ends_in_loop(head, stopped=False)       # no repetition
    assert not P.ends_in_loop(loop3 + [7], stopped=False)   # escaped at the very end
    assert P.ends_in_loop(head + [5] * 40, stopped=False)   # period 1
    period20 = list(range(200, 220))
    assert P.ends_in_loop(head + period20 * 2, stopped=False)        # two full periods of 20
    assert not P.ends_in_loop(head + period20 + period20[:15], stopped=False)  # < 2 periods
    period33 = list(range(300, 333))
    assert not P.ends_in_loop(head + period33 * 3, stopped=False)    # period above the maximum
    assert not P.ends_in_loop([1, 2] * 10, stopped=False)            # shorter than the minimum tail
    m = P.self_generation_metrics(loop3 + [END], head, stopped=True)
    assert m["loop"] is False and m["n"] == len(loop3) and m["copy"] == pytest.approx(40 / 76)  # head copied


# ---------------------------------------------------------------- decoding
def _tiny_model(seed: int = 0):
    import jax

    from bes.model import model as M

    params = M.init_params(TINY, jax.random.key(seed))
    keys = iter(jax.random.split(jax.random.key(seed + 1), 200))
    # perturb every leaf so the zero-initialised feature tables matter
    params = jax.tree.map(lambda x: x + 0.3 * jax.random.normal(next(keys), x.shape, x.dtype), params)
    return params, M.make_consts(TINY)


def test_decode_batch_matches_single_stream_generation(tok):
    from bes.exact.features import IncrementalFeatures
    from bes.model import generate as G

    params, consts = _tiny_model()
    items = P.eval_items("arith", 2, tok) + P.eval_items("copy", 1, tok) + P.eval_items("count", 1, tok)
    prompts = [it.prompt_ids for it in items]
    assert len({len(p) for p in prompts}) > 1                 # ragged: left padding is exercised
    got = P.decode_batch(params, consts, TINY, prompts, max_new=10, tok=tok)
    for prompt, d in zip(prompts, got):
        ref = G.generate(params, consts, TINY, prompt, IncrementalFeatures(tok), max_new_tokens=10)
        assert d.tokens == ref.tokens and d.stopped == ref.stopped
    rngs = [np.random.default_rng(i) for i in range(len(prompts))]
    s1 = P.decode_batch(params, consts, TINY, prompts, max_new=8, tok=tok, temperature=0.8, rngs=rngs)
    rngs = [np.random.default_rng(i) for i in range(len(prompts))]
    s2 = P.decode_batch(params, consts, TINY, prompts, max_new=8, tok=tok, temperature=0.8, rngs=rngs)
    assert [d.tokens for d in s1] == [d.tokens for d in s2]           # reproducible sampling


class _Script:
    """Stand-in decoder: every row's highest score is the scripted token of the step."""

    def __init__(self, script, vocab):
        self.script, self.vocab, self.i = script, vocab, 0

    def _z(self, b):
        z = np.zeros((b, self.vocab), np.float32)
        z[:, self.script[min(self.i, len(self.script) - 1)]] = 10.0
        return z

    def prefill(self, params, consts, cache, f):
        return self._z(f["tokens"].shape[0]), None, cache

    def decode_step(self, params, consts, cache, tok, f):
        self.i += 1
        return self._z(len(tok)), None, cache


def test_calculator_results_are_forced(tok):
    expr = tok.encode("12 * 3")
    forced = tok.encode("36") + [CALC_END]
    junk = tok.encode(" banana")[:1] * len(forced)
    dot = tok.encode(".")
    script = [CALC] + expr + [EQ] + junk + dot + [END]
    prompt = P.eval_items("arith", 1, tok)[0].prompt_ids
    with_f = P.decode_batch(None, None, TINY, [prompt, prompt], max_new=30, tok=tok, force_calc=True,
                            decoder=_Script(script, 8192))
    assert with_f[0].tokens == [CALC] + expr + [EQ] + forced + dot + [END]
    assert with_f[0].forced == len(forced) and with_f[1].tokens == with_f[0].tokens
    plain = P.decode_batch(None, None, TINY, [prompt], max_new=30, tok=tok, force_calc=False,
                           decoder=_Script(script, 8192))
    assert plain[0].tokens == script and plain[0].forced == 0
    # the decoder's tracker and the local fallback force the same tokens
    tracker, name = P.make_forcer(1, 30, tok)
    local = P.CalcForcer(1, tok)
    for t in [CALC] + expr + [EQ]:
        a, b = tracker.advance([t], [True]), local.advance([t], [True])
    assert name.startswith("bes.decode") and a == b == [forced]
    bad = P.CalcForcer(1, tok)
    for t in [CALC] + tok.encode("2 +") + [EQ]:
        out = bad.advance([t])
    assert out == [[CALC_END]]


def test_run_probe_scores_scripted_answers(tok):
    it = P.eval_items("arith", 1, tok)[0]
    script = it.answer_ids + [END]
    res = P.run_probe(None, None, TINY, [it], tok, force_calc=True, decoder=_Script(script, 8192))
    assert res[0]["ok"] and res[0]["forced"] == len(tok.encode(it.meta["expected"])) + 1
    assert re.search(r"= -?\d+\.$", res[0]["text"])


# ------------------------------------------------------------ script logic
def _rp():
    sys.path.insert(0, str(REPO / "scripts"))
    import run_probes
    return run_probes


def test_e5_verdicts_compare_against_the_baseline():
    rp = _rp()

    def rec(setting, loops, rep=0.1, model="ref"):
        samples = [{"rep4": rep, "loop": i < loops, "copy": 0.0, "n": 50, "stopped": i >= loops}
                   for i in range(20)]
        return {"domain": "chat", "setting": setting, "temperature": 0.5, "samples": samples, "model": model}
    recs = [rec("tape_off", 2), rec("tape_on", 10), rec("src5_0.25", 2), rec("src5_0.5", 3, rep=0.104)]
    v = rp.e5_verdicts(recs)
    assert v[("chat", "tape_on")][0] is False and "loop" in v[("chat", "tape_on")][1][0]
    assert v[("chat", "src5_0.25")] == (True, [])
    assert v[("chat", "src5_0.5")][0] is True          # +0.05 loops within 2 x SE, +0.004 rep4 < tol
    # the trained no-tape arm, when present, is the baseline
    v2 = rp.e5_verdicts(recs + [rec("no_tape_arm", 12, model="no_tape_arm")])
    assert v2[("chat", "tape_on")][0] is True


def test_reports_have_placeholders_before_any_run(tmp_path):
    rp = _rp()
    assert "Not run yet" in rp.e4_report(tmp_path)
    assert "Not run yet" in rp.e5_report(tmp_path)
    names = {a.name for a in rp.E4_ARMS}
    assert names == {"probe_ref", "probe_no_tape", "probe_no_reg", "probe_no_calc"}
    assert set(rp.PLAN) == set(P.PROBES)
    for _, (_, on, off, view) in rp.PLAN.items():
        assert on in names and off in names and view in rp.VIEWS


@pytest.mark.skipif(not (REPO / "data_cache" / "samples").exists(), reason="needs data_cache/samples")
def test_held_out_prompts(tok):
    chat = P.chat_prompts(3, tok)
    web = P.web_prompts(2, tok, prefix=32)
    assert len(chat) == 3 and all(p[0] == BOS and p[-1] == ASSISTANT and len(p) <= 384 for p in chat)
    assert len(web) == 2 and all(p[0] == BOS and len(p) == 33 for p in web)
    assert all(a != b for a, b in itertools.combinations(chat, 2))


def test_e4_report_on_synthetic_records(tmp_path):
    from bes.eval import microbench as mb

    rp = _rp()
    accs = {("probe_ref", "default"): {"copy": 98, "count": 95, "fence": 97, "arith": 99},
            ("probe_no_tape", "default"): {"copy": 20, "count": 90, "fence": 90, "arith": 99},
            ("probe_no_reg", "default"): {"copy": 97, "count": 60, "fence": 30, "arith": 99},
            ("probe_no_calc", "default"): {"copy": 97, "count": 95, "fence": 95, "arith": 3},
            ("probe_ref", "tape_off"): {"copy": 10}, ("probe_ref", "reg_off"): {"count": 40, "fence": 45},
            ("probe_ref", "no_forcing"): {"arith": 0}}
    for a in rp.E4_ARMS:
        mb.append_jsonl(tmp_path / mb.RESULTS, {
            "type": "arm", "status": "done", "name": a.name, "spec_hash": a.name, "tokens": 1000,
            "shape": [2, 4, 512], "n_params": 10, "bench": {"config": "configs/micro/bench.py"},
            "data": {"token_shares": {"web": 0.5, "chat": 0.2, "probe_calc": 0.3}}, "arm": a.to_dict()})
    for (arm, view), per in accs.items():
        for probe, k in per.items():
            res = [{"ok": i < k, "meta": {"n": 1 + i % 8} if probe == "count" else {}} for i in range(100)]
            mb.append_jsonl(tmp_path / rp.PROBES_FILE, {
                "type": "probe", "arm": arm, "spec_hash": arm, "view": view, "probe": probe,
                "n_test": 100, "summary": P.probe_summary(res), "failures": [], "examples": []})
    text = rp.e4_report(tmp_path)
    row = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| **")}
    assert "**pass**" in row["**copy**"] and "**pass**" in row["**fence**"] and "**pass**" in row["**arith**"]
    assert "fail (off 60% >= 50%)" in row["**count**"]
    assert "98.0% [" in row["**copy**"] and "(`tape_off`)" in row["**copy**"]
    assert "### count by N (`probe_no_reg`)" in text
