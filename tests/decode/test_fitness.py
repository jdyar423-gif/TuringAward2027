"""The reader's own test of a stored answer (``bes.decode.fitness.AnswerFit``).

A smoke-sized model on the committed tokenizer. Checked here: the fit of an answer under its own
question is exactly zero; the two passes are the model's teacher-forced losses of the answer's
tokens (against ``bes.model.model.loss_fn`` on a row built by hand); an answer too long for the
window is cut at the same token for both passes; nothing fits when too little would be left."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest

from bes.config import load_config
from bes.decode.fitness import AnswerFit
from bes.model import model as M

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def scorer():
    from bes.tokenizer import load

    tok = load()
    cfg = load_config(str(REPO / "configs" / "smoke.py")).model
    cfg = dataclasses.replace(cfg, n_loops=2, max_seq=256)
    params = jax.tree.map(np.array, M.init_params(cfg, jax.random.PRNGKey(0)))
    rng = np.random.default_rng(0)
    params["head"]["out"] = params["head"]["out"] + rng.standard_normal(
        params["head"]["out"].shape).astype(np.float32) * 0.5
    return AnswerFit(params, M.make_consts(cfg), cfg, tok, max_tokens=96), tok, cfg, params


def test_the_fit_of_an_answer_under_its_own_question_is_zero(scorer):
    fit, _, _, _ = scorer
    q, a = "Why does ice float on water?", "Ice is less dense than liquid water, so it floats."
    assert fit(q, q, a) == 0.0
    other = fit("What is the capital of France?", q, a)
    assert other is not None and np.isfinite(other) and other != 0.0
    # antisymmetric in the two questions
    assert fit(q, "What is the capital of France?", a) == pytest.approx(-other, abs=1e-6)


def test_a_pass_is_the_models_loss_of_the_answers_tokens(scorer):
    from bes.data import chat_format as cf
    from bes.exact.features import exact_features

    fit, tok, cfg, params = scorer
    q, a = "Name three countries in Europe.", "France, Germany and Spain."
    got = fit.nll(q, a)
    ids, roles = cf.render_conversation([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tok)
    T = fit.T
    tokens = np.full((1, T + 1), ids[0] * 0, np.int32)
    from bes.config import PAD

    tokens[:] = PAD
    tokens[0, :len(ids)] = ids
    seg = np.zeros((1, T), np.int32)
    seg[0, len(ids):] = 1
    n_answer = len(tok.encode(a)) + 1                      # the answer and its END
    w = np.zeros((1, T), np.float32)
    start = len(ids) - n_answer - 1                        # targets are shifted by one
    w[0, start:start + n_answer] = 1.0
    batch = {"tokens": tokens, "w": w, "seg": seg, **exact_features(tokens[:, :T], seg, tok)}
    m = M.loss_fn(params, M.make_consts(cfg), batch, cfg)[1]
    assert got is not None and got[1] == n_answer
    assert got[0] == pytest.approx(float(m["ce_sum"]) / float(m["weight"]), rel=1e-5)


def test_a_long_answer_is_cut_the_same_for_both_passes(scorer):
    fit, tok, _, _ = scorer
    long = " ".join(f"Sentence number {i} of a very long answer." for i in range(40))
    short_q, long_q = "Why?", "Could you please explain to me in detail why the sky is blue during the day?"
    cut = fit.cut((short_q, long_q), long)
    assert cut is not None and long.startswith(cut[:40]) and len(cut) < len(long)
    for q in (short_q, long_q):
        assert len(fit._render(q, cut)[0]) <= fit.T + 1
    assert fit(short_q, long_q, long) is not None and fit(long_q, long_q, long) == 0.0
    # nothing sensible is left of the answer: no verdict
    assert fit.cut(("word " * 200,), "a short answer") is None
    assert fit("word " * 200, short_q, "a short answer") is None
