"""Tests of bes.exact.features: the batch features combine registers, constraints and the tape,
and IncrementalFeatures reproduces them one token at a time.

``bes.exact.constraints`` may not exist yet; most tests therefore install a stub whose
``targets_for_tokens`` follows the specified semantics (an assistant turn's targets come from
the user turn before it) and use the real ``bes.exact.registers``. One test uses the real
constraints module and is skipped while it is missing.
"""
from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

from bes import config
from bes.config import ASSISTANT, BOS, END, LIBRARY, PAD, ROLE_TOKENS, SYSTEM, USER
from bes.exact import features, tape

registers = pytest.importorskip("bes.exact.registers")
ROLES = tuple(ROLE_TOKENS)


def stub_targets_for_tokens(tokens, seg, tok):
    """int32[B, T, 3]: inside an assistant turn (its ASSISTANT token included) the targets are
    (L % 5, L % 3, L) with L the length of the last complete user turn of the segment; -1
    elsewhere."""
    tokens = np.asarray(tokens)
    out = np.full(tokens.shape + (3,), -1, np.int32)
    for b in range(tokens.shape[0]):
        user_len, cur_role, cur_len, target = None, None, 0, None
        for t in range(tokens.shape[1]):
            v = int(tokens[b, t])
            new_seg = t == 0 or v == BOS or (seg is not None and seg[b, t] != seg[b, t - 1])
            if new_seg:
                user_len, cur_role, cur_len, target = None, None, 0, None
            if v in ROLES:
                if cur_role == USER:
                    user_len = cur_len
                cur_role, cur_len = v, 0
                target = None if v != ASSISTANT or user_len is None else (user_len % 5, user_len % 3, user_len)
            else:
                cur_len += 1
            if cur_role == ASSISTANT and target is not None:
                out[b, t] = target
    return out


@pytest.fixture
def stub_constraints(monkeypatch):
    real = features._optional

    def optional(name):
        if name == "constraints":
            return types.SimpleNamespace(targets_for_tokens=stub_targets_for_tokens)
        return real(name)

    monkeypatch.setattr(features, "_optional", optional)


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


def conversation_rows(tok, n_t=320):
    from bes.data import chat_format as cf

    convs = [
        [{"role": "system", "content": "Be brief."},
         {"role": "user", "content": "Write exactly three sentences about the sea."},
         {"role": "assistant", "content": "The sea is vast. It is salty. It covers most of Earth."},
         {"role": "user", "content": "Now list 2 colours of the sea:"},
         {"role": "assistant", "content": "- blue\n- green\n"}],
        [{"role": "library", "content": "Paris\n\nParis is the capital of France."},
         {"role": "user", "content": "What is the capital of France? Answer in one word."},
         {"role": "assistant", "content": [{"type": "text", "text": "Paris. "},
                                           {"type": "calc", "expr": "2 * (3 + 4)", "result": "14"},
                                           {"type": "text", "text": " \"quoted\" `code` (x)."}]},
         {"role": "user", "content": "And of Italy?"}],
    ]
    segs = [cf.render_conversation(convs[0], tok)[0],
            cf.render_conversation(convs[1], tok, add_generation_prompt=True)[0],
            cf.render_document("```python\nprint('a')\n```\nThe end. The end.", tok)[0]]
    rows, segids = [], []
    for shift in range(3):
        row, sg = [], []
        for i in range(3):
            ids = segs[(i + shift) % 3].tolist()
            row += ids
            sg += [i] * len(ids)
        row, sg = row[:n_t], sg[:n_t]
        pad = n_t - len(row)
        rows.append(row + [PAD] * pad)
        segids.append(sg + [sg[-1] + 1] * pad)
    return np.asarray(rows, np.int32), np.asarray(segids, np.int32)


def random_chat_rows(seed, n_rows=2, n_t=300):
    rng = np.random.default_rng(seed)
    toks = np.empty((n_rows, n_t), np.int32)
    seg = np.empty((n_rows, n_t), np.int32)
    words = [rng.integers(0, 4000) for _ in range(30)] + [13, 46, 10, 33]   # incl. ".", "\n"
    for b in range(n_rows):
        row, sg, s = [], [], 0
        while len(row) < n_t:
            row.append(BOS)
            sg.append(s)
            for _ in range(int(rng.integers(1, 5))):
                r = [SYSTEM, USER, ASSISTANT, LIBRARY][int(rng.integers(0, 4))]
                body = [int(words[i]) for i in rng.integers(0, len(words), int(rng.integers(0, 15)))]
                turn = [r, *body, END]
                row += turn
                sg += [s] * len(turn)
            s += 1
        toks[b], seg[b] = row[:n_t], sg[:n_t]
    return toks, seg


def assert_incremental_equals_batch(toks, seg, tok, want):
    for b in range(toks.shape[0]):
        inc = features.IncrementalFeatures(tok)
        got = inc.extend(toks[b], seg[b])
        for key in features.FEATURE_KEYS:
            np.testing.assert_array_equal(got[key], want[key][b], err_msg=f"{key} row {b}")


# ----------------------------------------------------------------------------- tests
def test_batch_features_shapes_and_sources(tok, stub_constraints):
    toks, seg = conversation_rows(tok)
    out = features.exact_features(toks, seg, tok)
    n_b, n_t = toks.shape
    assert set(out) == set(features.FEATURE_KEYS)
    assert out["reg"].shape == (n_b, n_t, config.N_REG) and out["reg"].dtype == np.int16
    assert out["clock"].shape == (n_b, n_t, config.N_CLOCKS) and out["clock"].dtype == np.int32
    assert out["cand_tok"].shape == (n_b, n_t, config.CAND_SLOTS) and out["cand_tok"].dtype == np.int32
    assert out["cand_feat"].shape == (n_b, n_t, config.CAND_SLOTS, config.N_CAND_FEAT)
    assert out["cand_feat"].dtype == np.int8
    regs = registers.compute(toks, seg, stub_targets_for_tokens(toks, seg, tok))
    np.testing.assert_array_equal(out["reg"], regs.reg)
    np.testing.assert_array_equal(out["clock"], regs.clock)
    cand = tape.compute(toks, seg)
    np.testing.assert_array_equal(out["cand_tok"], cand.cand_tok)
    np.testing.assert_array_equal(out["cand_feat"], cand.cand_feat)
    # the countdowns are actually exercised by these rows
    rem = [i for i, (name, _) in enumerate(config.REG_FIELDS) if name.startswith("rem_")]
    assert (out["reg"][..., rem] > 0).any()


def test_explicit_targets_override(tok, stub_constraints):
    toks, seg = conversation_rows(tok)
    tgt = np.full(toks.shape + (3,), -1, np.int32)
    out = features.exact_features(toks, seg, tok, targets=tgt)
    np.testing.assert_array_equal(out["reg"], registers.compute(toks, seg, tgt).reg)


def test_incremental_equals_batch_conversations(tok, stub_constraints):
    toks, seg = conversation_rows(tok)
    assert_incremental_equals_batch(toks, seg, tok, features.exact_features(toks, seg, tok))


@pytest.mark.parametrize("seed", range(3))
def test_incremental_equals_batch_random_chats(tok, stub_constraints, seed):
    toks, seg = random_chat_rows(seed)
    assert_incremental_equals_batch(toks, seg, tok, features.exact_features(toks, seg, tok))


def test_incremental_segments_without_bos(tok, stub_constraints):
    """Segments that start with an ordinary token (a change of seg only) and repeat content."""
    rng = np.random.default_rng(9)
    toks = rng.integers(0, 3, (2, 200)).astype(np.int32)
    toks[:, ::37] = USER
    toks[:, 5::41] = ASSISTANT
    toks[:, 0] = 1
    seg = np.cumsum(rng.random((2, 200)) < 0.05, axis=1).astype(np.int32)
    want = features.exact_features(toks, seg, tok)
    assert (want["cand_tok"] >= 0).any()
    assert_incremental_equals_batch(toks, seg, tok, want)


def test_incremental_fork(tok, stub_constraints):
    toks, seg = conversation_rows(tok)
    row = toks[0].tolist()
    cut = row.index(ASSISTANT, row.index(ASSISTANT) + 1) + 1           # inside the 2nd assistant turn
    prefix = row[:cut]
    cont_a = row[cut:cut + 40]
    cont_b = tok.encode(" Blue and green. Done.") + [END, USER] + tok.encode("More?") + [END, ASSISTANT]
    inc = features.IncrementalFeatures(tok)
    inc.extend(prefix)
    other = inc.fork()
    got_a = [inc.step(v) for v in cont_a]
    got_b = [other.step(v) for v in cont_b]
    for cont, got in ((cont_a, got_a), (cont_b, got_b)):
        full = np.asarray([prefix + cont], np.int32)
        want = features.exact_features(full, None, tok)
        for i, rowf in enumerate(got):
            for key in features.FEATURE_KEYS:
                np.testing.assert_array_equal(rowf[key], want[key][0, len(prefix) + i], err_msg=key)


def test_missing_constraints_means_no_targets(tok, monkeypatch):
    real = features._optional
    monkeypatch.setattr(features, "_optional", lambda name: None if name == "constraints" else real(name))
    monkeypatch.setattr(features, "_warned", set())
    toks, seg = conversation_rows(tok)
    with pytest.warns(RuntimeWarning):
        out = features.exact_features(toks, seg, tok)
    np.testing.assert_array_equal(out["reg"], registers.compute(toks, seg, None).reg)
    assert not features.has_constraints()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # warned only once
        inc = features.IncrementalFeatures(tok)
    got = inc.extend(toks[1], seg[1])
    np.testing.assert_array_equal(got["reg"], out["reg"][1])


def test_missing_registers_raises(tok, monkeypatch):
    real = features._optional
    monkeypatch.setattr(features, "_optional", lambda name: None if name == "registers" else real(name))
    with pytest.raises(RuntimeError):
        features.exact_features(np.asarray([[BOS, 1, 2]]), None, tok)


def test_with_real_constraints_module(tok):
    if not features.has_constraints():
        pytest.skip("bes.exact.constraints.targets_for_tokens is not available yet")
    toks, seg = conversation_rows(tok)
    want = features.exact_features(toks, seg, tok)
    assert_incremental_equals_batch(toks, seg, tok, want)
