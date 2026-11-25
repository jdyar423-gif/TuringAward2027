"""Row writes (``bes.learn.write``): theorem 1 of docs/PLAN_M6V.md as tests.

* only the allowed rows of the allowed tables change, bit for bit;
* the logits of an input without a written token are bit-identical after the write, those of an
  input with one are not;
* the taught loss goes down;
* two writes whose texts do not contain each other's written tokens commute exactly."""
from __future__ import annotations

import os
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))

from _model_testlib import make_batch, random_params, tiny_cfg  # noqa: E402

from bes.learn import write as W  # noqa: E402
from bes.model import model as M  # noqa: E402
from bes.train.checkpoint import flatten_with_paths  # noqa: E402

CFG = tiny_cfg(gain_o=True, tok_neurons=3, tok_dim=24)
N_CONTENT = CFG.vocab_size - 16


def _np(tree):
    paths, leaves, _ = flatten_with_paths(tree)
    return dict(zip(paths, [np.asarray(x) for x in jax.device_get(leaves)]))


def _batch(seed, lo, hi, B=2, T=24):
    """A batch whose ordinary tokens (inputs, targets, candidates) lie in ``[lo, hi)``."""
    b = make_batch(CFG, B=B, T=T, seed=seed)
    bos = int(b["tokens"][0, 0])
    span = hi - lo
    tok = b["tokens"].copy()
    tok[tok != bos] = lo + tok[tok != bos] % span
    cand = b["cand_tok"].copy()
    cand[cand >= 0] = lo + cand[cand >= 0] % span
    return {**b, "tokens": tok, "cand_tok": cand}


@pytest.fixture(scope="module")
def model():
    params = random_params(CFG, seed=1)
    consts = M.make_consts(CFG)
    logits = jax.jit(lambda p, f: M.logits_fn(p, consts, f, CFG))
    return params, consts, logits


@pytest.mark.parametrize("tables", [W.WRITE_TABLES, W.TOKEN_TABLES])
def test_a_write_changes_only_its_rows_and_only_inputs_that_read_them(model, tables):
    params, consts, logits = model
    taught = _batch(0, 0, 60)                              # tokens 0..59
    rows = W.rows_of(taught, CFG.vocab_size, n_content=N_CONTENT)
    rows[:20] = False                                      # the 20 most "frequent" ids are not written
    assert rows.sum() > 5 and not rows[N_CONTENT:].any()
    w = W.Writer(CFG, consts, tables, W.reference_norms(params, tables, N_CONTENT))
    new, rep = w.write(params, taught, rows, steps=4, lr=0.2)
    assert rep["loss_after"] < rep["loss_before"] and rep["rows"] == int(rows.sum())
    a, b = _np(params), _np(new)
    for path in a:
        if path in tables:
            assert np.array_equal(a[path][~rows], b[path][~rows]), path          # other rows: the same bits
            assert not np.array_equal(a[path][rows], b[path][rows]), path
        else:
            assert np.array_equal(a[path], b[path]), path                         # every other leaf
    # an input with none of the written tokens: bit-identical logits
    other = _batch(5, 100, 400)
    assert not rows[W.input_tokens(other)].any()
    f_other = M.split_batch(other)[0]
    assert np.array_equal(np.asarray(logits(params, f_other)), np.asarray(logits(new, f_other)))
    # an input that reads a written row: not identical
    f_taught = M.split_batch(taught)[0]
    assert not np.array_equal(np.asarray(logits(params, f_taught)), np.asarray(logits(new, f_taught)))
    # tokens the taught text shares with it but whose rows were not written change nothing either
    shared = _batch(6, 0, 20)
    f_shared = M.split_batch(shared)[0]
    assert np.array_equal(np.asarray(logits(params, f_shared)), np.asarray(logits(new, f_shared)))


def test_writes_with_disjoint_texts_commute_exactly(model):
    params, consts, _ = model
    one, two = _batch(1, 20, 80), _batch(2, 200, 260)
    r1 = W.rows_of(one, CFG.vocab_size, n_content=N_CONTENT)
    r2 = W.rows_of(two, CFG.vocab_size, n_content=N_CONTENT)
    assert not (r1 & r2).any()
    w = W.Writer(CFG, consts, W.WRITE_TABLES, W.reference_norms(params, n_content=N_CONTENT))
    ab, _ = w.write(w.write(params, one, r1, steps=3)[0], two, r2, steps=3)
    ba, _ = w.write(w.write(params, two, r2, steps=3)[0], one, r1, steps=3)
    x, y = _np(ab), _np(ba)
    assert all(np.array_equal(x[p], y[p]) for p in x)
    # and the first write's effect on its own text is untouched by the second
    a_only, _ = w.write(params, one, r1, steps=3)
    assert w.loss(a_only, one) == w.loss(ab, one)


def test_rows_of_selects_by_rank_and_leaves_control_tokens_out():
    b = _batch(3, 0, 300)
    rank = np.arange(CFG.vocab_size)                       # id = rank
    rows = W.rows_of(b, CFG.vocab_size, rank=rank, min_rank=128, n_content=N_CONTENT)
    ids = W.input_tokens(b)
    assert rows.sum() == int(((ids >= 128) & (ids < N_CONTENT)).sum()) and not rows[:128].any()
    assert not rows[N_CONTENT:].any()                      # BOS is in the batch and is not selected
    with pytest.raises(ValueError):
        W.Writer(CFG, {}, ("core/attn/q",))
