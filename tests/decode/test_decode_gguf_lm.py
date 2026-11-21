"""The llama.cpp adapter (bes.decode.gguf_lm).

The vocabulary mapping is checked on a synthetic vocabulary. On the stand-in model file the
references are independent: llama-cpp-python's own high-level ``Llama`` (a separate context,
one sequence, ``logits_all=True``) evaluates every particle's full history from scratch, its
``detokenize`` gives the visible text, and ``bes.eval.proxy.run_gguf.GGUFBackend`` renders the
prompt the way the harness does.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bes.decode import potentials as P
from bes.decode import sampling as S
from bes.decode import smc
from bes.decode.gguf_lm import GGUFVocab
from bes.exact import constraints as C
from bes.exact import instr_table as it
from bes.exact.constraints import ConstraintSpec, Interval

MODEL = Path(__file__).resolve().parents[2] / "data_cache" / "opponents" / \
    "Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf"


# ------------------------------------------------------------------ synthetic vocabulary
def synthetic():
    pieces = [b"", b"Hello", b" world", b".", b"\n", b"x" * 200, b"", b"", b"", b" a", b"", b"?"]
    control = [True, False, False, False, False, False, True, True, True, False, True, False]
    eog = [False] * 12
    eog[8] = True
    texts = ["<pad>", "Hello", " world", ".", "\n", "x" * 200, "<|im_end|>", "<|im_start|>",
             "<|end_of_text|>", " a", "<|end|>", "?"]
    return GGUFVocab(pieces, control, eog, bos=None, end=6, texts=texts, tag="synthetic")


def test_vocabulary_mapping():
    v = synthetic()
    sp = v.special_tokens
    assert v.n_model == 12 and v.vocab_size == 12 + 4          # calc, eq, calc_end, bos
    assert sp["<|end|>"] == 6 and v.end == 6
    assert sp["<|calc|>"] == 12 and sp["<|eq|>"] == 13 and sp["<|/calc|>"] == 14
    assert sp["<|bos|>"] == 15 and v.bos is None
    assert sp["<|im_start|>"] == 7 and sp["<|end_of_text|>"] == 8 and sp["<pad>"] == 0
    assert sp["<|gguf_10|>"] == 10                              # a control token named <|end|>
    assert sp["<|gguf_long_5|>"] == 5                           # longer than 127 bytes
    assert v.eog[6] and v.eog[8] and not v.eog[1]
    tb = v.all_token_bytes()
    assert len(tb) == v.vocab_size and tb[5] == b"" and tb[1] == b"Hello" and tb[6] == b""
    assert v.decode_bytes([1, 2, 5, 3]) == b"Hello world."     # specials carry no exact bytes
    assert v.text([1, 2, 5, 3, 6, 7]) == "Hello world" + "x" * 200 + "."
    other = GGUFVocab([b"", b"Hello", b" World", b".", b"\n", b"x" * 200, b"", b"", b"", b" a", b"",
                       b"?"], [True, False, False, False, False, False, True, True, True, False, True,
                               False], [False] * 12, None, 6, tag="synthetic")
    assert synthetic().sha256 == v.sha256 and other.sha256 != v.sha256
    with pytest.raises(ValueError):
        v.encode("Hello")                                       # no tokenizer attached


def test_the_exact_layer_accepts_the_mapped_vocabulary():
    v = synthetic()
    table = it.build(v)
    assert table.vocab_size == v.vocab_size
    assert table.kind[6] == it.KIND_END and table.kind[15] == it.KIND_BOS
    assert [table.kind[i] for i in (12, 13, 14)] == [it.KIND_CALC, it.KIND_EQ, it.KIND_CALC_END]
    assert all(table.kind[i] == it.KIND_OTHER for i in (0, 5, 7, 8, 10))
    assert all(table.kind[i] == it.KIND_ORDINARY for i in (1, 2, 3, 4, 9, 11))
    spec = ConstraintSpec(sentences=Interval.exactly(1))
    prog = P.compile(spec, v)
    tr = prog.tracker(1, 6)
    m = tr.allowed()[0]
    assert not m[[0, 5, 6, 7, 8, 10, 12, 13, 14, 15]].any()   # specials, long token, END now
    assert m[1]                                                # "Hello" can start the reply
    tr.advance(np.array([1]))
    tr.advance(np.array([3]))                                  # "Hello."
    assert tr.allowed()[0][6]                                  # END now completes 1 sentence


# ----------------------------------------------------------------------- the real model
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="stand-in GGUF not downloaded")


@pytest.fixture(scope="module")
def falcon():
    pytest.importorskip("llama_cpp")
    if not MODEL.exists():
        pytest.skip("stand-in GGUF not downloaded")
    import llama_cpp

    from bes.decode.gguf_lm import GGUFChat, GGUFLM, GGUFModel

    model = GGUFModel(MODEL)
    # the adapter's default context enables llama.cpp's flash-attention kernel: the reference
    # uses the same kernel (the plain kernel is a different approximation, tested below)
    ref = llama_cpp.Llama(model_path=str(MODEL), n_ctx=1024, n_threads=1, n_threads_batch=1,
                          logits_all=True, flash_attn=True, verbose=False)
    lm = GGUFLM(model, n_threads=1, max_new_tokens=80)
    yield model, GGUFChat(model), lm, ref
    lm.close()


def _lsm(z) -> np.ndarray:
    z = np.asarray(z, np.float64)
    return z - (z.max() + math.log(np.exp(z - z.max()).sum()))


def ref_rows(ref, prompt, tokens) -> list[np.ndarray]:
    """log-softmax rows (float64) of the reference model after ``prompt + tokens[:i]`` for
    ``i = 0 .. len(tokens)``, evaluated from scratch the way a decoder feeds them: the prompt
    in one batch, then one token at a time. (llama.cpp's results for this hybrid model depend
    on the batch shape: one batch of the whole sequence differs from token-by-token
    evaluation by up to about 2 in a tail log-probability, so the shape must match.)"""
    ref.reset()
    ref.eval(list(prompt))
    rows = [_lsm(ref.scores[len(prompt) - 1])]
    for i, t in enumerate(tokens):
        ref.eval([int(t)])
        rows.append(_lsm(ref.scores[len(prompt) + i]))
    return rows


def ref_logp(ref, prompt, tokens=()) -> np.ndarray:
    """The reference row after ``prompt + tokens``."""
    return ref_rows(ref, prompt, tokens)[-1]


def _cmp(row: np.ndarray, want: np.ndarray, model) -> None:
    n = model.vocab.n_model
    other_eog = [t for t in np.flatnonzero(model.vocab.eog) if t != model.vocab.end]
    keep = np.ones(n, bool)
    keep[np.flatnonzero(model.vocab.eog)] = False
    np.testing.assert_allclose(row[:n][keep], want[keep], atol=1e-4)
    e = want[np.flatnonzero(model.vocab.eog)]
    assert row[model.vocab.end] == pytest.approx(np.log(np.exp(e).sum()), abs=1e-4)
    assert np.all(np.isneginf(row[other_eog])) and np.all(np.isneginf(row[n:]))
    assert np.log(np.exp(row.astype(np.float64)).sum()) == pytest.approx(0.0, abs=1e-4)


@needs_model
def test_the_chat_format_is_the_harness_rendering(falcon):
    from bes.eval.proxy.run_gguf import GGUFBackend

    model, chat, lm, ref = falcon
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Name a colour."}]
    backend = GGUFBackend(MODEL, n_ctx=512, n_threads=1)
    assert chat.render(msgs) == backend.tokenize(backend.render(msgs))
    assert chat.end_id == model.vocab.special_tokens["<|end|>"]
    assert model.vocab.text([chat.end_id]) == ""
    ids = model.vocab.encode("Hello world, 42 times.")
    assert ref.detokenize(ids).decode() == "Hello world, 42 times."


@needs_model
def test_particles_match_a_fresh_evaluation_through_steps_and_resampling(falcon):
    model, chat, lm, ref = falcon
    prompt = chat.render([{"role": "user", "content": "Write one line about rain."}])
    rows = lm.prefill(prompt, 3)
    want = ref_logp(ref, prompt)
    for j in range(3):
        _cmp(rows[j], want, model)
    hist = [list(prompt) for _ in range(3)]
    rng = np.random.default_rng(0)
    plan = [(np.array([True, True, True]), None), (np.array([True, False, True]), None),
            (np.array([True, True, True]), np.array([2, 2, 0])),
            (np.array([False, True, True]), None), (np.array([True, True, True]), np.array([1, 0, 1]))]
    for active, anc in plan:
        if anc is not None:
            lm.reorder(anc)
            hist = [list(hist[a]) for a in anc]
        toks = rng.integers(600, 30000, size=3)
        out = lm.step(toks, active)
        for j in np.flatnonzero(active):
            hist[j].append(int(toks[j]))
            _cmp(out[j], ref_logp(ref, prompt, hist[j][len(prompt):]), model)
    assert lm.state_copies >= 2 + 2


@needs_model
def test_the_plain_attention_kernel_matches_the_harness_reference(falcon):
    import llama_cpp

    from bes.decode.gguf_lm import GGUFLM

    model, chat, lm, _ = falcon
    plain_ref = llama_cpp.Llama(model_path=str(MODEL), n_ctx=512, n_threads=1, n_threads_batch=1,
                                logits_all=True, verbose=False)          # flash_attn=False
    plain = GGUFLM(model, n_threads=1, max_new_tokens=8, flash_attn=False)
    prompt = chat.render([{"role": "user", "content": "Name a colour."}])
    rows = plain.prefill(prompt, 2)
    hist = list(prompt)
    _cmp(rows[1], ref_logp(plain_ref, prompt), model)
    for t in (2000, 7000, 12000):
        out = plain.step(np.array([t, 600]), np.array([True, False]))
        hist.append(t)
        _cmp(out[0], ref_logp(plain_ref, prompt, hist[len(prompt):]), model)
    plain.close()


@needs_model
def test_decoder_end_to_end_on_the_stand_in(falcon):
    model, chat, lm, ref = falcon
    v = model.vocab
    user = "Explain in exactly two sentences why the sea is salty."
    spec = C.compile(user)
    assert spec.sentences == Interval.exactly(2)
    prompt = chat.render([{"role": "user", "content": user}])
    st = S.SamplingSettings(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05)
    res = smc.decode(lm, prompt, n_particles=3, max_new_tokens=80, seed=11, settings=st,
                     program=P.compile(spec, v), tok=v)
    finished = [p for p in res.particles if p.finish == "stop"]
    assert finished
    for p in finished:
        text = chat.display(p.tokens).strip()
        assert not C.verify(spec, text), text
        assert text == ref.detokenize(p.tokens[:-1]).decode("utf-8", "replace").strip()
    p = finished[0]
    rows = ref_rows(ref, prompt, p.tokens[:-1])
    for t, lp, row in zip(p.tokens, p.token_logp, rows):
        e = row[np.flatnonzero(v.eog)]
        expect = math.log(np.exp(e).sum()) if t == v.end else row[t]
        assert lp == pytest.approx(expect, abs=1e-4)
    # without potentials, particle 0 of three is the single-particle sample, draw for draw
    one = smc.decode(lm, prompt, n_particles=1, max_new_tokens=40, seed=5, settings=st, tok=v)
    three = smc.decode(lm, prompt, n_particles=3, max_new_tokens=40, seed=5, settings=st, tok=v)
    assert one.particles[0].tokens == three.particles[0].tokens
