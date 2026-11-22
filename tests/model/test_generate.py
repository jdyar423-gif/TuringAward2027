"""Generation with a key-value cache: prefill + stepwise decode equals the full-sequence forward
(all features on, several split points, several segments, chunked and ragged prefill), causality,
masking of unused slots, cache reordering, the host loops and determinism. The same for the v9
write-side paths (token gains, head-wise output gate, token neurons) and selective attention,
whose selection the cache carries (``sel``, ``bos``): chunkings, ragged batches, garbage slots,
reordering, particles, a generated BOS, bfloat16, the switches off, the final allocation, the
cache shapes and donation."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import jitted, make_batch, random_params, tiny_cfg
from bes.config import CAND_FEATS, END, special_id
from bes.model import generate as G
from bes.model import model as M

ATOL = 1e-4
TMAX = 40
KEYS = ("tokens", "seg", "reg", "clock", "cand_tok", "cand_feat")


@pytest.fixture(scope="module")
def setup():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=3)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=24, seed=4, seg_starts=((0,), (0, 11)))
    feats, _, _ = M.split_batch(batch)
    feats = {k: np.asarray(v) for k, v in feats.items()}
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    hidden = np.asarray(jitted("forward_hidden", cfg)(params, consts, feats))
    return cfg, params, consts, feats, full, hidden


def _slice(feats: dict, sl) -> dict:
    return {k: np.asarray(feats[k])[:, sl] for k in KEYS if k in feats}


def _left_pad(f: dict, n_pad: int, pad_id: int) -> dict:
    """Left-pad a chunk with ``n_pad`` padding positions (seg -1, zero features)."""
    b = f["tokens"].shape[0]
    out = {}
    for k, a in f.items():
        if k == "tokens":
            fill = np.full((b, n_pad), pad_id, a.dtype)
        elif k == "seg":
            fill = np.full((b, n_pad), -1, a.dtype)
        elif k == "cand_tok":
            fill = np.full((b, n_pad) + a.shape[2:], -1, a.dtype)
        else:
            fill = np.zeros((b, n_pad) + a.shape[2:], a.dtype)
        out[k] = np.concatenate([fill, a], axis=1)
    return out


def incremental(dec, params, consts, cfg, feats, split, chunk=8, tmax=TMAX, cache=None,
                hidden_out=None):
    """Prefill positions ``[0, split)`` in left-padded chunks of ``chunk``, then decode every
    later position one token at a time. Returns ``(logits [B, T - split + 1, V], cache)`` for
    positions ``split - 1 .. T - 1`` (their hidden states are appended to ``hidden_out``)."""
    b, t = feats["tokens"].shape
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    cache = G.init_cache(cfg, b, tmax) if cache is None else cache
    pre = _slice(feats, slice(0, split))
    n_pad = (-split) % chunk
    pre = _left_pad(pre, n_pad, pad_id)
    pre["pos"] = np.concatenate([np.zeros((b, n_pad), np.int32),
                                 np.broadcast_to(np.arange(split, dtype=np.int32), (b, split))], 1)
    out = []
    for lo in range(0, split + n_pad, chunk):
        z, h, cache = dec.prefill(params, consts, cache, {k: a[:, lo:lo + chunk] for k, a in pre.items()})
    out.append(np.asarray(z))
    hs = [np.asarray(h)]
    for j in range(split, t):
        row = {k: np.asarray(feats[k])[:, j] for k in KEYS if k != "tokens"}
        z, h, cache = dec.decode_step(params, consts, cache, feats["tokens"][:, j], row)
        out.append(np.asarray(z))
        hs.append(np.asarray(h))
    if hidden_out is not None:
        hidden_out.append(np.stack(hs, axis=1))
    return np.stack(out, axis=1), cache


@pytest.mark.parametrize("split", [1, pytest.param(6, marks=pytest.mark.slow), 11, 12,
                                   pytest.param(17, marks=pytest.mark.slow), 24])
def test_prefill_then_decode_equals_full_forward(setup, split):
    """All features on (registers, clocks, candidate input and the output law, sink,
    confidence, value embeddings); row 1 holds two segments (BOS at 0 and 11). Scores and
    hidden states."""
    cfg, params, consts, feats, full, hidden = setup
    hs = []
    z, cache = incremental(G.jitted(cfg), params, consts, cfg, feats, split, hidden_out=hs)
    np.testing.assert_allclose(z, full[:, split - 1:], atol=ATOL, rtol=0)
    np.testing.assert_allclose(hs[0], hidden[:, split - 1:], atol=ATOL, rtol=0)
    assert int(cache["t"]) == feats["tokens"].shape[1] + (-split) % 8
    np.testing.assert_array_equal(np.asarray(cache["next_pos"]), [24, 24])
    np.testing.assert_array_equal(np.asarray(cache["cur_seg"]), [0, 1])


@pytest.mark.slow
def test_one_chunk_prefill_and_unpadded_chunks(setup):
    cfg, params, consts, feats, full, hid = setup
    dec = G.jitted(cfg)
    cache = G.init_cache(cfg, 2, TMAX)
    z, h, cache = dec.prefill(params, consts, cache, _slice(feats, slice(0, 24)))
    np.testing.assert_allclose(np.asarray(z), full[:, 23], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(h), hid[:, 23], atol=ATOL, rtol=0)
    # two unpadded chunks [0, 9) and [9, 20) with default positions
    cache = G.init_cache(cfg, 2, TMAX)
    _, _, cache = dec.prefill(params, consts, cache, _slice(feats, slice(0, 9)))
    sub = _slice(feats, slice(9, 20))
    z, _, cache = dec.prefill(params, consts, cache, sub)
    np.testing.assert_allclose(np.asarray(z), full[:, 19], atol=ATOL, rtol=0)


def test_ragged_left_padded_batch_equals_single_rows(setup):
    """Two prompts of different lengths in one left-padded prefill give each row's logits."""
    cfg, params, consts, feats, full, _ = setup
    dec = G.jitted(cfg)
    lens = (9, 14)
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    rows = []
    for b, n in enumerate(lens):
        one = {k: a[b:b + 1, :n] for k, a in _slice(feats, slice(0, 24)).items()}
        one = _left_pad(one, 16 - n, pad_id)
        one["pos"] = np.concatenate([np.zeros((1, 16 - n), np.int32), np.arange(n, dtype=np.int32)[None]], 1)
        rows.append(one)
    both = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    cache = G.init_cache(cfg, 2, TMAX)
    z, _, cache = dec.prefill(params, consts, cache, {k: a[:, :8] for k, a in both.items()})
    z, _, cache = dec.prefill(params, consts, cache, {k: a[:, 8:] for k, a in both.items()})
    np.testing.assert_allclose(np.asarray(z[0]), full[0, lens[0] - 1], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(z[1]), full[1, lens[1] - 1], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.asarray(cache["next_pos"]), lens)
    # each row continues with its own next token
    nxt = np.array([feats["tokens"][0, lens[0]], feats["tokens"][1, lens[1]]])
    row = {k: np.stack([feats[k][0, lens[0]], feats[k][1, lens[1]]]) for k in KEYS if k != "tokens"}
    z, _, _ = dec.decode_step(params, consts, cache, nxt, row)
    np.testing.assert_allclose(np.asarray(z[0]), full[0, lens[0]], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(z[1]), full[1, lens[1]], atol=ATOL, rtol=0)


def test_a_chunk_ending_in_padding_sets_the_default_position_and_segment(setup):
    """A fixed-size prefill chunk whose last slots are padding (right-padded, ragged rows): the
    defaults of the next call (``next_pos``, ``cur_seg``) follow each row's last valid slot, so a
    decode step without explicit positions continues every row; a row whose chunk is all padding
    keeps its defaults."""
    cfg, params, consts, feats, full, _ = setup
    dec = G.jitted(cfg, donate=False)
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    lens = (6, 14)                                       # row 1: two segments (BOS at 0 and 11)
    chunk = {k: np.array(a[:, :16]) for k, a in _slice(feats, slice(0, 24)).items()}
    for b, n in enumerate(lens):                         # right padding after the prompt
        chunk["tokens"][b, n:] = pad_id
        chunk["seg"][b, n:] = -1
    chunk["pos"] = np.broadcast_to(np.arange(16, dtype=np.int32), (2, 16)).copy()
    _, _, cache = dec.prefill(params, consts, G.init_cache(cfg, 2, TMAX), chunk)
    np.testing.assert_array_equal(np.asarray(cache["next_pos"]), lens)
    np.testing.assert_array_equal(np.asarray(cache["cur_seg"]), [0, 1])
    nxt = np.array([feats["tokens"][0, lens[0]], feats["tokens"][1, lens[1]]])
    row = {k: np.stack([feats[k][0, lens[0]], feats[k][1, lens[1]]])
           for k in KEYS if k not in ("tokens", "seg")}
    z, _, after = dec.decode_step(params, consts, cache, nxt, row)
    np.testing.assert_allclose(np.asarray(z[0]), full[0, lens[0]], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(z[1]), full[1, lens[1]], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.asarray(after["next_pos"]), np.asarray(lens) + 1)
    # an all-padding chunk changes no default
    pad = {k: np.zeros_like(a[:, :4]) for k, a in chunk.items()}
    pad["tokens"][:] = pad_id
    pad["seg"][:] = -1
    _, _, same = dec.prefill(params, consts, cache, pad)
    for name in ("next_pos", "cur_seg"):
        np.testing.assert_array_equal(np.asarray(same[name]), np.asarray(cache[name]))


def _perturb(feats: dict, cfg, j: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: np.array(v) for k, v in feats.items()}
    out["tokens"][:, j] = (out["tokens"][:, j] + 1 + rng.integers(0, 50)) % (cfg.vocab_size - 16)
    out["reg"][:, j] = (out["reg"][:, j] + 1) % 2
    out["clock"][:, j] += 7
    out["cand_tok"][:, j] = rng.integers(0, cfg.vocab_size - 16, size=out["cand_tok"][:, j].shape)
    out["cand_feat"][:, j, :, 1:] = (out["cand_feat"][:, j, :, 1:] + 1) % 8
    return out


@pytest.mark.parametrize("j", [5, 15])
def test_perturbation_causality(setup, j):
    """Perturbing token j (and its feature row) leaves every logit before j and every cached
    key/value before j bit-identical, changes the logits from j on within j's segment and
    leaves the positions of other segments bit-identical."""
    cfg, params, consts, feats, _, _ = setup
    dec = G.jitted(cfg, donate=False)
    pert = _perturb(feats, cfg, j)
    za, _ = incremental(dec, params, consts, cfg, feats, split=1)
    zb, _ = incremental(dec, params, consts, cfg, pert, split=1)
    np.testing.assert_array_equal(zb[:, :j], za[:, :j])      # column c is position c
    diff = np.max(np.abs(zb - za), axis=-1)                    # [B, T]
    same_seg = feats["seg"] == feats["seg"][:, j:j + 1]
    later = np.arange(diff.shape[1])[None] >= j
    assert np.min(diff[later & same_seg]) > 1e-4
    np.testing.assert_array_equal(diff[later & ~same_seg], 0.0)
    ca = dec.prefill(params, consts, G.init_cache(cfg, 2, TMAX), _slice(feats, slice(0, 24)))[2]
    cb = dec.prefill(params, consts, G.init_cache(cfg, 2, TMAX), _slice(pert, slice(0, 24)))[2]
    for name in ("k", "v"):
        a, b = np.asarray(ca[name]), np.asarray(cb[name])
        np.testing.assert_array_equal(b[..., :j, :], a[..., :j, :])
        assert np.max(np.abs(b[..., j, :] - a[..., j, :])) > 0


def test_unused_and_future_slots_are_invisible(setup):
    """Garbage in the unused slots (huge keys/values, same segment, later positions or segment
    -1) does not change any logit: the mask uses positions and segments, not slot order."""
    cfg, params, consts, feats, full, _ = setup
    dec = G.jitted(cfg)
    rng = np.random.default_rng(1)
    cache = G.init_cache(cfg, 2, TMAX)
    shape = np.asarray(cache["k"]).shape
    cache["k"] = np.asarray(1e3 * rng.standard_normal(shape), np.float32)
    cache["v"] = np.asarray(1e3 * rng.standard_normal(shape), np.float32)
    seg = np.zeros((2, TMAX), np.int32)
    seg[:, ::3] = -1
    cache["seg"] = seg
    cache["pos"] = np.full((2, TMAX), 10_000, np.int32)
    z, _ = incremental(dec, params, consts, cfg, feats, split=6, cache=cache)
    np.testing.assert_allclose(z, full[:, 5:], atol=ATOL, rtol=0)


def test_reorder_cache_rows(setup):
    """Row j of a reordered cache continues exactly like row idx[j] of the original."""
    cfg, params, consts, _, _, _ = setup
    batch = make_batch(cfg, B=3, T=12, seed=21, seg_starts=((0,), (0, 5), (0,)))
    feats, _, _ = M.split_batch(batch)
    feats = {k: np.asarray(v) for k, v in feats.items()}
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    dec = G.jitted(cfg, donate=False)
    cache = G.init_cache(cfg, 3, TMAX)
    _, _, cache = dec.prefill(params, consts, cache, _slice(feats, slice(0, 8)))
    idx = np.array([2, 0, 0, 1], np.int32)
    re = dec.reorder(cache, idx)
    assert np.asarray(re["k"]).shape[2] == 4 and int(re["t"]) == int(cache["t"])
    for name in ("k", "v"):
        np.testing.assert_array_equal(np.asarray(re[name]), np.asarray(cache[name])[:, :, idx])
    for name in ("seg", "pos", "next_pos", "cur_seg"):
        np.testing.assert_array_equal(np.asarray(re[name]), np.asarray(cache[name])[idx])
    for j in range(8, 12):
        row = {k: feats[k][idx, j] for k in KEYS if k != "tokens"}
        z, _, re = dec.decode_step(params, consts, re, feats["tokens"][idx, j], row)
        np.testing.assert_allclose(np.asarray(z), full[idx, j], atol=ATOL, rtol=0)


@pytest.mark.slow
def test_static_feature_flags_off():
    """Without sinks, confidence, registers, candidates, value embeddings and clock planes."""
    cfg = tiny_cfg(use_sink=False, use_conf=False, use_reg=False, use_cand_in=False,
                   use_cand_out=False, ve_dim=0, clock_planes=0)
    params = random_params(cfg, seed=8)
    consts = M.make_consts(cfg)
    feats, _, _ = M.split_batch(make_batch(cfg, B=2, T=16, seed=9))
    feats = {k: np.asarray(v) for k, v in feats.items()}
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, _ = incremental(G.jitted(cfg), params, consts, cfg, feats, split=5, tmax=32)
    np.testing.assert_allclose(z, full[:, 4:], atol=ATOL, rtol=0)


def test_runtime_switches_are_honoured(setup):
    cfg, params, _, feats, _, _ = setup
    consts = M.make_consts(cfg, sink=0.0, conf=0.0, cand_out=0.0, kind_2=0.0, clock=0.0)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, _ = incremental(G.jitted(cfg), params, consts, cfg, feats, split=9)
    np.testing.assert_allclose(z, full[:, 8:], atol=ATOL, rtol=0)


def _logsumexp(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    top = z.max(-1, keepdims=True)
    return (top + np.log(np.exp(z - top).sum(-1, keepdims=True)))[..., 0]


def test_down_weighted_self_evidence_in_decoding(setup):
    """``src_5 = 0.25`` (the decoder's weight for evidence from the text being generated):
    prefill + decode equals the full forward, the scores are normalised log-probabilities,
    and the multiplier changes them."""
    cfg, params, _, feats, full_default, _ = setup
    consts = M.make_consts(cfg, src_5=0.25)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, _ = incremental(G.jitted(cfg), params, consts, cfg, feats, split=9)
    np.testing.assert_allclose(z, full[:, 8:], atol=ATOL, rtol=0)
    assert np.max(np.abs(_logsumexp(z))) < 1e-5 and np.max(np.abs(_logsumexp(full))) < 1e-5
    src = [name for name, _ in CAND_FEATS].index("src")
    assert np.any(feats["cand_feat"][..., src] == 5)
    assert np.max(np.abs(full - full_default)) > 1e-4


def _np_tempered_law(cfg, params, hidden, cand, m, temperature):
    """log p_T from hidden states (numpy float64): the head logits written here, the network
    tempered, the masses added untempered: log((softmax(z / T) + add) / (1 + M))."""
    from _model_testlib import to_numpy

    P = to_numpy(params)
    h = np.asarray(hidden, np.float64)
    z = (h @ P["head"]["proj"]) @ P["head"]["out"].T
    z = cfg.softcap * np.tanh(z / cfg.softcap)
    zt = z / temperature
    q = np.exp(zt - zt.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    m = np.asarray(m, np.float64)
    cand = np.asarray(cand)
    valid = (cand >= 0) & (cand < cfg.vocab_size)
    m = np.where(valid, m, 0.0)
    add = np.zeros_like(q)
    for idx in np.ndindex(cand.shape[:-1]):
        for k in range(cand.shape[-1]):
            if valid[idx + (k,)]:
                add[idx + (int(cand[idx + (k,)]),)] += m[idx + (k,)]
    return np.log((q + add) / (1.0 + m.sum(-1))[..., None])


@pytest.mark.parametrize("temperature", [0.5, 0.8])
def test_decoding_at_a_temperature_tempers_the_network_only(setup, temperature):
    """Prefill + stepwise decode at temperature T: the scores are log p_T with the network
    tempered and the tape's masses untouched (independent numpy law from the hidden states and
    the masses); one compiled program serves every temperature (default 1 included); without
    a tape the scores are z / T."""
    cfg, params, consts, feats, full, hidden = setup
    dec = G.jitted(cfg, donate=False)
    hs = []
    z, _ = incremental(dec, params, consts, cfg, feats, 9, hidden_out=hs)     # temperature 1

    def n_compiled():
        return tuple(getattr(f, "compiled", f)._cache_size() for f in (dec.prefill, dec.decode_step))
    sizes = n_compiled()
    zt = _incremental_t(dec, params, consts, cfg, feats, 9, temperature)
    tail = {k: feats[k][:, 8:] for k in ("cand_tok", "cand_feat")}
    cand, m = jitted("tape_masses", cfg)(params, consts, hs[0], tail)
    want = _np_tempered_law(cfg, params, hs[0], cand, m, temperature)
    np.testing.assert_allclose(zt, want, atol=2e-4, rtol=0)
    assert np.max(np.abs(_logsumexp(zt))) < 1e-5
    mixture = z / temperature                           # tempering the whole law: different
    mixture = mixture - _logsumexp(mixture)[..., None]
    assert np.max(np.abs(zt - mixture)) > 1e-2
    off = M.make_consts(cfg, cand_out=0.0)
    zo = _incremental_t(dec, params, off, cfg, feats, 9, temperature)
    plain = np.asarray(jitted("logits_fn", cfg)(params, off, feats))[:, 8:] / temperature
    np.testing.assert_allclose(zo - _logsumexp(zo)[..., None],
                               plain - _logsumexp(plain)[..., None], atol=2e-4, rtol=0)
    _incremental_t(dec, params, consts, cfg, feats, 9, 1.3)
    # no temperature, the default one or any other: the same compiled programs
    assert n_compiled() == sizes


def _incremental_t(dec, params, consts, cfg, feats, split, temperature, chunk=8):
    """:func:`incremental` with the temperature passed to every call."""
    b, t = feats["tokens"].shape
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    cache = G.init_cache(cfg, b, TMAX)
    pre = _left_pad(_slice(feats, slice(0, split)), (-split) % chunk, pad_id)
    n_pad = (-split) % chunk
    pre["pos"] = np.concatenate([np.zeros((b, n_pad), np.int32),
                                 np.broadcast_to(np.arange(split, dtype=np.int32), (b, split))], 1)
    for lo in range(0, split + n_pad, chunk):
        z, _, cache = dec.prefill(params, consts, cache, {k: a[:, lo:lo + chunk] for k, a in pre.items()},
                                  temperature)
    out = [np.asarray(z)]
    for j in range(split, t):
        row = {k: np.asarray(feats[k])[:, j] for k in KEYS if k != "tokens"}
        z, _, cache = dec.decode_step(params, consts, cache, feats["tokens"][:, j], row, temperature)
        out.append(np.asarray(z))
    return np.stack(out, axis=1)


def test_generate_loop_samples_the_tempered_law(setup):
    """The host loop at a temperature records the scores of the full forward at that
    temperature (``logits_fn(..., temperature)``)."""
    cfg, params, consts, _, _, _ = setup
    bos = M.bos_id(cfg)
    prompt = [bos] + list(range(30, 40))
    n_new = 6
    rows = _random_rows(cfg, len(prompt) + n_new, seed=16)
    out = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows), G.greedy,
                     max_new_tokens=n_new, stop_ids=(), chunk=8, record=True, temperature=0.6)
    seq = prompt + out.tokens
    f = {"tokens": np.asarray(seq[:-1], np.int32)[None],
         "seg": G.prompt_segments(seq[:-1], bos)[None],
         **{k: a[None, :len(seq) - 1] for k, a in rows.items()}}
    ref = np.asarray(jax.jit(functools.partial(M.logits_fn, cfg=cfg))(params, consts, f,
                                                                       temperature=np.float32(0.6)))[0]
    np.testing.assert_allclose(out.logits, ref[len(prompt) - 1:], atol=ATOL, rtol=0)


def test_decode_is_deterministic(setup):
    cfg, params, consts, feats, _, _ = setup
    dec = G.jitted(cfg)
    a, ca = incremental(dec, params, consts, cfg, feats, split=4)
    b, cb = incremental(dec, params, consts, cfg, feats, split=4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.asarray(ca["k"]), np.asarray(cb["k"]))


class ReplayFeatures:
    """Provider replaying precomputed rows (one per ``step`` call)."""

    def __init__(self, rows: dict, start: int = 0):
        self.rows, self.i = rows, start

    def step(self, token, new_segment=False):
        out = {k: a[self.i] for k, a in self.rows.items()}
        self.i += 1
        return out

    def fork(self):
        return ReplayFeatures(self.rows, self.i)


def _forward_rows(cfg, params, consts, tokens, rows):
    f = {"tokens": np.asarray(tokens, np.int32)[None],
         "seg": G.prompt_segments(tokens, M.bos_id(cfg))[None],
         **{k: a[None, :len(tokens)] for k, a in rows.items()}}
    return np.asarray(jitted("logits_fn", cfg)(params, consts, f))[0]


def _random_rows(cfg, n, seed):
    b = make_batch(cfg, B=1, T=n, seed=seed)
    return {k: b[k][0] for k in ("reg", "clock", "cand_tok", "cand_feat")}


@pytest.mark.parametrize("provider_kind", ["zero", "replay"])
def test_generate_loop_matches_the_full_forward(setup, provider_kind):
    cfg, params, consts, _, _, _ = setup
    bos = M.bos_id(cfg)
    rng = np.random.default_rng(5)
    prompt = [bos] + rng.integers(0, cfg.vocab_size - 16, 10).tolist() + [bos] + [7, 9, 11]
    n_new = 9
    if provider_kind == "zero":
        provider, rows = G.ZeroFeatures(), G.empty_rows(len(prompt) + n_new)
    else:
        rows = _random_rows(cfg, len(prompt) + n_new, seed=6)
        provider = ReplayFeatures(rows)
    out = G.generate(params, consts, cfg, prompt, provider, G.greedy, max_new_tokens=n_new,
                     stop_ids=(), chunk=8, record=True)
    assert len(out.tokens) == n_new and not out.stopped
    seq = prompt + out.tokens
    full = _forward_rows(cfg, params, consts, seq[:-1], rows)
    ref = full[len(prompt) - 1:]
    np.testing.assert_allclose(out.logits, ref, atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.argmax(ref, -1), out.tokens)
    again = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows) if provider_kind == "replay"
                       else G.ZeroFeatures(), G.greedy, max_new_tokens=n_new, stop_ids=(), chunk=8)
    assert again.tokens == out.tokens


def test_generate_stops_at_a_stop_token(setup):
    cfg, params, consts, _, _, _ = setup
    bos = M.bos_id(cfg)
    calls = []

    def sampler(z, step):
        calls.append(step)
        return END if step == 3 else int(np.argmax(z))
    out = G.generate(params, consts, cfg, [bos, 5, 6], None, sampler, max_new_tokens=20, chunk=8)
    assert out.stopped and out.tokens[-1] == END and len(out.tokens) == 4 and calls == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        G.generate(params, consts, cfg, [bos] * 30, None, G.greedy, max_new_tokens=20, tmax=32, chunk=8)


def test_particles_match_single_streams_and_resampling(setup):
    """K particles: particle j takes the rank-j token at step 0 and is greedy afterwards; at
    step 2 the population is resampled. Every final history must be greedy-consistent under the
    full forward, and its first token must be the one chosen by its ancestor."""
    cfg, params, consts, _, _, _ = setup
    bos = M.bos_id(cfg)
    prompt = [bos, 17, 23, 29, 31]
    k, n_new = 4, 6
    rows = _random_rows(cfg, len(prompt) + n_new, seed=7)
    resample = np.array([1, 1, 3, 0])
    first = {}

    def sampler(z, step, done):
        assert z.shape == (k, cfg.vocab_size)
        if step == 0:
            order = np.argsort(-z[0])
            first.update({j: int(order[j]) for j in range(k)})
            assert np.allclose(z, z[:1], atol=0)          # identical particles after the prompt
            return order[:k]
        toks = np.argmax(z, axis=-1)
        if step == 2:
            return toks[resample], resample
        return toks
    res = G.generate_particles(params, consts, cfg, prompt, ReplayFeatures(rows), sampler, k,
                               max_new_tokens=n_new, stop_ids=(), chunk=8)
    assert len(res.tokens) == k and all(len(t) == n_new for t in res.tokens)
    np.testing.assert_array_equal(res.ancestors[2], resample)
    for j, hist in enumerate(res.tokens):
        assert hist[0] == first[int(resample[j])]
        full = _forward_rows(cfg, params, consts, prompt + hist[:-1], rows)
        np.testing.assert_array_equal(np.argmax(full[len(prompt):], -1), hist[1:])
    # greedy particles all equal the single-stream greedy generation
    single = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows), G.greedy,
                        max_new_tokens=n_new, stop_ids=(), chunk=8)
    same = G.generate_particles(params, consts, cfg, prompt, ReplayFeatures(rows),
                                lambda z, s, d: np.argmax(z, -1), k, max_new_tokens=n_new,
                                stop_ids=(), chunk=8)
    assert all(t == single.tokens for t in same.tokens)


@pytest.mark.slow
def test_bf16_decode_runs_and_stays_close():
    cfg = tiny_cfg(act_dtype="bfloat16")
    params = random_params(cfg, seed=11, scale=0.1)
    consts = M.make_consts(cfg)
    feats, _, _ = M.split_batch(make_batch(cfg, B=2, T=16, seed=12))
    feats = {k: np.asarray(v) for k, v in feats.items()}
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, cache = incremental(G.jitted(cfg), params, consts, cfg, feats, split=6, tmax=32)
    assert np.asarray(cache["k"]).dtype.name == "bfloat16"
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z - full[:, 5:])) < 0.15


def test_a_generated_bos_starts_a_new_segment(setup):
    """The host loop segments the stream like the training rows (a new segment at every BOS),
    so a forced BOS mid-stream gives the same logits as the full forward."""
    cfg, params, consts, _, _, _ = setup
    bos = M.bos_id(cfg)
    prompt = [bos, 40, 41, 42, 43]
    n_new = 6
    rows = _random_rows(cfg, len(prompt) + n_new, seed=13)

    def sampler(z, step):
        return bos if step == 2 else int(np.argmax(z[:cfg.vocab_size - 16]))
    out = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows), sampler,
                     max_new_tokens=n_new, stop_ids=(), chunk=8, record=True)
    seq = prompt + out.tokens
    assert seq[len(prompt) + 2] == bos
    full = _forward_rows(cfg, params, consts, seq[:-1], rows)
    np.testing.assert_allclose(out.logits, full[len(prompt) - 1:], atol=ATOL, rtol=0)


@pytest.mark.slow
def test_generation_with_the_exact_layer_provider():
    """End to end with ``bes.exact.features.IncrementalFeatures`` (registers, clocks and the
    candidate tape computed token by token): the logits of every step equal the full forward
    on the batch features of the final sequence."""
    feats_mod = pytest.importorskip("bes.exact.features")
    tok_mod = pytest.importorskip("bes.tokenizer.tok")
    from bes.config import ASSISTANT, BOS, USER, VOCAB_SIZE
    try:
        tok = tok_mod.load()
        provider = feats_mod.IncrementalFeatures(tok)
    except Exception as exc:  # noqa: BLE001 - missing tokenizer or exact-layer assets
        pytest.skip(f"exact layer unavailable: {exc}")
    cfg = tiny_cfg(vocab_size=VOCAB_SIZE)
    params = random_params(cfg, seed=14)
    consts = M.make_consts(cfg)
    prompt = [BOS, USER] + tok.encode("Name two colours in one sentence.") + [END, ASSISTANT]
    n_new = 10
    ordinary = VOCAB_SIZE - 16
    out = G.generate(params, consts, cfg, prompt, provider,
                     lambda z, s: int(np.argmax(z[:ordinary])), max_new_tokens=n_new,
                     stop_ids=(), chunk=16, record=True)
    seq = np.asarray(prompt + out.tokens[:-1], np.int32)[None]
    batch = feats_mod.exact_features(seq, np.zeros_like(seq), tok)
    f = {"tokens": seq, "seg": np.zeros_like(seq), **batch}
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, f))[0]
    np.testing.assert_allclose(out.logits, full[len(prompt) - 1:], atol=ATOL, rtol=0)
    assert np.any(batch["cand_tok"] >= 0)          # the tape was exercised


# ------------------------------------------- v9 write-side paths and selective attention
#: every v9 path in tiny form, next to every older feature path: token gains on part of the FFN
#: neurons, on the values and on the head outputs (two phases over three loops), selective
#: attention, the head-wise output gate (per-loop bias) and token neurons (sorted table gradient)
V9 = dict(gain_ffn=50, gain_o=True, gain_v=True, gain_phases=2, sel_attn=True, attn_gate=True,
          tok_neurons=3, tok_dim=24, table_grad="sorted", sorted_chunk=16)
#: the allocation of configs/v9.py in tiny form with the optional output gate on (no selective
#: attention: no cache entries)
FINAL_V9 = dict(gain_o=True, attn_gate=True, tok_neurons=3, tok_dim=24, table_grad="sorted",
                sorted_chunk=16)
V9_OFF = {"gain": 0.0, "sel": 0.0, "gate": 0.0, "page": 0.0}


def _v9_batch(cfg, B=2, T=24, seed=32):
    """Row 1 holds two segments (BOS at 0 and 11); row 0 has one more BOS inside its segment
    (position 6): a key selection never masks and a query whose selection reaches later ones."""
    batch = make_batch(cfg, B=B, T=T, seed=seed, seg_starts=((0,), (0, 11)))
    batch["tokens"][0, 6] = M.bos_id(cfg)
    feats, _, _ = M.split_batch(batch)
    return {k: np.asarray(v) for k, v in feats.items()}


@pytest.fixture(scope="module")
def setup_v9():
    cfg = tiny_cfg(**V9)
    params = random_params(cfg, seed=31)
    consts = M.make_consts(cfg)
    feats = _v9_batch(cfg)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    hidden = np.asarray(jitted("forward_hidden", cfg)(params, consts, feats))
    return cfg, params, consts, feats, full, hidden


@pytest.mark.parametrize("split,chunk", [(24, 24), (24, 5), (11, 8), (1, 8)])
def test_v9_prefill_then_decode_equals_full_forward(setup_v9, split, chunk):
    """One chunk, many chunks (boundaries inside segments, one left-padding slot), a padded
    prefill then single steps, and single steps only: scores and hidden states of the full
    forward, with every v9 path on (the selection of the earlier queries carried in the cache)."""
    cfg, params, consts, feats, full, hidden = setup_v9
    hs = []
    z, cache = incremental(G.jitted(cfg), params, consts, cfg, feats, split, chunk=chunk,
                           hidden_out=hs)
    np.testing.assert_allclose(z, full[:, split - 1:], atol=ATOL, rtol=0)
    np.testing.assert_allclose(hs[0], hidden[:, split - 1:], atol=ATOL, rtol=0)
    n_pad = (-split) % chunk
    sel, bos = np.asarray(cache["sel"]), np.asarray(cache["bos"])
    assert sel.shape == (cfg.n_loops, cfg.n_blocks, 2, TMAX) and sel.dtype == np.float32
    assert bos.shape == (2, TMAX) and bos.dtype == np.bool_
    written = slice(n_pad, n_pad + 24)
    np.testing.assert_array_equal(bos[:, written], feats["tokens"] == M.bos_id(cfg))
    assert not bos[:, :n_pad].any() and not bos[:, n_pad + 24:].any()
    # the selection is non-negative, zero at BOS and empty slots, positive somewhere
    assert np.all(sel >= 0) and np.all(sel[:, :, bos] == 0) and sel.max() > 0.1
    np.testing.assert_array_equal(sel[..., n_pad + 24:], 0.0)
    np.testing.assert_array_equal(sel[..., :n_pad], 0.0)


def test_v9_selection_accumulated_step_by_step_equals_one_chunk(setup_v9):
    """The cache's ``sel`` after 24 single steps equals the ``sel`` of one 24-position chunk
    (the recurrence ``F = sel + earlier queries of the chunk``, ``sel += sum S``), and the
    selection matters: with the ``sel`` switch at 0 the scores differ."""
    cfg, params, consts, feats, full, _ = setup_v9
    dec = G.jitted(cfg, donate=False)
    _, _, one = dec.prefill(params, consts, G.init_cache(cfg, 2, TMAX), _slice(feats, slice(0, 24)))
    _, steps = incremental(dec, params, consts, cfg, feats, 1, chunk=1)
    np.testing.assert_allclose(np.asarray(steps["sel"]), np.asarray(one["sel"]), rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(steps["bos"]), np.asarray(one["bos"]))
    off = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, sel=0.0), feats))
    assert np.max(np.abs(off - full)) > 1e-2


def test_v9_ragged_left_padded_batch_equals_single_rows(setup_v9):
    """Two prompts of different lengths in one left-padded prefill (padding keys are neither
    seen nor selected) give each row's scores; each row then continues with its own token."""
    cfg, params, consts, feats, full, _ = setup_v9
    dec = G.jitted(cfg)
    lens = (9, 14)
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    rows = []
    for b, n in enumerate(lens):
        one = {k: a[b:b + 1, :n] for k, a in _slice(feats, slice(0, 24)).items()}
        one = _left_pad(one, 16 - n, pad_id)
        one["pos"] = np.concatenate([np.zeros((1, 16 - n), np.int32), np.arange(n, dtype=np.int32)[None]], 1)
        rows.append(one)
    both = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    cache = G.init_cache(cfg, 2, TMAX)
    z, _, cache = dec.prefill(params, consts, cache, {k: a[:, :8] for k, a in both.items()})
    z, _, cache = dec.prefill(params, consts, cache, {k: a[:, 8:] for k, a in both.items()})
    np.testing.assert_allclose(np.asarray(z[0]), full[0, lens[0] - 1], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(z[1]), full[1, lens[1] - 1], atol=ATOL, rtol=0)
    for b, n in enumerate(lens):               # padding slots hold no selection
        np.testing.assert_array_equal(np.asarray(cache["sel"])[:, :, b, :16 - n], 0.0)
    nxt = np.array([feats["tokens"][0, lens[0]], feats["tokens"][1, lens[1]]])
    row = {k: np.stack([feats[k][0, lens[0]], feats[k][1, lens[1]]]) for k in KEYS if k != "tokens"}
    z, _, _ = dec.decode_step(params, consts, cache, nxt, row)
    np.testing.assert_allclose(np.asarray(z[0]), full[0, lens[0]], atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.asarray(z[1]), full[1, lens[1]], atol=ATOL, rtol=0)


def test_v9_unused_and_future_slots_are_invisible(setup_v9):
    """Garbage in every unused slot, the selection and BOS entries included, changes nothing:
    the slots of a chunk start with a zero selection and their BOS flags are written."""
    cfg, params, consts, feats, full, _ = setup_v9
    rng = np.random.default_rng(2)
    cache = G.init_cache(cfg, 2, TMAX)
    for name in ("k", "v", "sel"):
        cache[name] = np.asarray(1e3 * np.abs(rng.standard_normal(np.shape(cache[name]))), np.float32)
    cache["bos"] = rng.random((2, TMAX)) < 0.5
    seg = np.zeros((2, TMAX), np.int32)
    seg[:, ::3] = -1
    cache["seg"] = seg
    cache["pos"] = np.full((2, TMAX), 10_000, np.int32)
    z, _ = incremental(G.jitted(cfg), params, consts, cfg, feats, split=6, cache=cache)
    np.testing.assert_allclose(z, full[:, 5:], atol=ATOL, rtol=0)


def test_v9_reorder_cache_rows(setup_v9):
    """Reordering gathers the selection and BOS entries with the other rows; row j of the
    reordered cache continues exactly like row idx[j] of the original."""
    cfg, params, consts, _, _, _ = setup_v9
    feats = _v9_batch(cfg, B=3, T=12, seed=33)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    dec = G.jitted(cfg, donate=False)
    _, _, cache = dec.prefill(params, consts, G.init_cache(cfg, 3, TMAX), _slice(feats, slice(0, 8)))
    idx = np.array([2, 0, 0, 1], np.int32)
    re = dec.reorder(cache, idx)
    assert set(re) == set(cache)
    np.testing.assert_array_equal(np.asarray(re["sel"]), np.asarray(cache["sel"])[:, :, idx])
    np.testing.assert_array_equal(np.asarray(re["bos"]), np.asarray(cache["bos"])[idx])
    for j in range(8, 12):
        row = {k: feats[k][idx, j] for k in KEYS if k != "tokens"}
        z, _, re = dec.decode_step(params, consts, re, feats["tokens"][idx, j], row)
        np.testing.assert_allclose(np.asarray(z), full[idx, j], atol=ATOL, rtol=0)


def test_v9_particles_with_resampling(setup_v9):
    """K particles of one prompt (rank-j token first, then greedy, resampled at step 2): every
    history is greedy-consistent under the full forward; greedy particles equal one stream."""
    cfg, params, consts, _, _, _ = setup_v9
    bos = M.bos_id(cfg)
    prompt = [bos, 17, 23, 29, 31, 37]
    k, n_new = 4, 6
    rows = _random_rows(cfg, len(prompt) + n_new, seed=34)
    resample = np.array([3, 1, 1, 0])
    first = {}

    def sampler(z, step, done):
        if step == 0:
            order = np.argsort(-z[0])
            first.update({j: int(order[j]) for j in range(k)})
            return order[:k]
        toks = np.argmax(z, axis=-1)
        return (toks[resample], resample) if step == 2 else toks
    res = G.generate_particles(params, consts, cfg, prompt, ReplayFeatures(rows), sampler, k,
                               max_new_tokens=n_new, stop_ids=(), chunk=4)
    np.testing.assert_array_equal(res.ancestors[2], resample)
    for j, hist in enumerate(res.tokens):
        assert len(hist) == n_new and hist[0] == first[int(resample[j])]
        full = _forward_rows(cfg, params, consts, prompt + hist[:-1], rows)
        np.testing.assert_array_equal(np.argmax(full[len(prompt):], -1), hist[1:])
    single = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows), G.greedy,
                        max_new_tokens=n_new, stop_ids=(), chunk=4)
    same = G.generate_particles(params, consts, cfg, prompt, ReplayFeatures(rows),
                                lambda z, s, d: np.argmax(z, -1), k, max_new_tokens=n_new,
                                stop_ids=(), chunk=4)
    assert all(t == single.tokens for t in same.tokens)


def test_v9_generated_bos_starts_a_new_segment(setup_v9):
    """The host loop with every v9 path: a BOS forced mid-stream starts a new segment and the
    recorded scores equal the full forward of the final sequence."""
    cfg, params, consts, _, _, _ = setup_v9
    bos = M.bos_id(cfg)
    prompt = [bos, 40, 41, 42, 43, 44, 45]
    n_new = 7
    rows = _random_rows(cfg, len(prompt) + n_new, seed=35)

    def sampler(z, step):
        return bos if step == 3 else int(np.argmax(z[:cfg.vocab_size - 16]))
    out = G.generate(params, consts, cfg, prompt, ReplayFeatures(rows), sampler,
                     max_new_tokens=n_new, stop_ids=(), chunk=4, record=True)
    seq = prompt + out.tokens
    assert seq[len(prompt) + 3] == bos
    full = _forward_rows(cfg, params, consts, seq[:-1], rows)
    np.testing.assert_allclose(out.logits, full[len(prompt) - 1:], atol=ATOL, rtol=0)


def test_v9_switches_off_decode_like_the_model_without_the_paths(setup_v9):
    """With ``gain``, ``sel``, ``gate`` and ``page`` at 0 the cache path equals the full forward
    at those switches and, bit for bit, the cache path of the model without the v9 flags (same
    other leaves)."""
    cfg, params, _, feats, _, _ = setup_v9
    off = M.make_consts(cfg, **V9_OFF)
    full = np.asarray(jitted("logits_fn", cfg)(params, off, feats))
    z, _ = incremental(G.jitted(cfg), params, off, cfg, feats, split=9)
    np.testing.assert_allclose(z, full[:, 8:], atol=ATOL, rtol=0)
    base = tiny_cfg(table_grad="sorted", sorted_chunk=16)
    plain = {k: v for k, v in params.items() if k not in ("gain", "page")}
    plain["core"] = {**params["core"], "attn": {k: v for k, v in params["core"]["attn"].items()
                                                 if k not in ("gate_w", "gate_b")}}
    zb, _ = incremental(G.jitted(base), plain, M.make_consts(base), base, feats, split=9)
    np.testing.assert_array_equal(z, zb)


@pytest.mark.parametrize("gate", [False, True], ids=["v9", "v9+gate"])
def test_v9_final_allocation_decodes_like_the_full_forward(gate):
    """configs/v9.py in tiny form (output gains, token neurons, no selection), and the same with
    the optional output gate: the cache has no selection entries and prefill + decode equals the
    full forward."""
    cfg = tiny_cfg(**{**FINAL_V9, "attn_gate": gate})
    params = random_params(cfg, seed=36)
    consts = M.make_consts(cfg)
    feats = _v9_batch(cfg, seed=37)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, cache = incremental(G.jitted(cfg), params, consts, cfg, feats, split=9)
    np.testing.assert_allclose(z, full[:, 8:], atol=ATOL, rtol=0)
    assert set(cache) == {"k", "v", "seg", "pos", "t", "next_pos", "cur_seg"}


def test_init_cache_allocates_nothing_for_paths_that_are_off():
    """configs/v9.py: the cache has the entries and shapes of the same model without any v9
    path, plus ``sel`` (float32 also under bfloat16 activations) and ``bos`` exactly when
    selective attention is on (shapes only, nothing allocated)."""
    from _model_testlib import config_model

    v9 = config_model("v9")
    plain = dataclasses.replace(v9, gain_ffn=0, gain_o=False, gain_v=False, gain_phases=1,
                                sel_attn=False, attn_gate=False, tok_neurons=0)

    def shapes(cfg):
        tree = jax.eval_shape(lambda: G.init_cache(cfg, 4, 2048))
        return {k: (a.shape, a.dtype) for k, a in tree.items()}
    got, want = shapes(v9), shapes(plain)
    assert set(got) == set(want) | ({"sel", "bos"} if v9.sel_attn else set())
    assert {k: got[k] for k in want} == want
    sel = shapes(dataclasses.replace(v9, sel_attn=True, attn_impl="xla"))
    nl, nb = v9.n_loops, v9.n_blocks
    assert {k: v for k, v in sel.items() if k not in ("sel", "bos")} == want
    assert sel["sel"] == ((nl, nb, 4, 2048), jnp.float32) and sel["bos"] == ((4, 2048), jnp.bool_)
    assert want["k"] == ((nl, nb, 4, v9.n_heads, 2048, v9.head_dim), jnp.bfloat16)


def test_v9_decoder_donates_every_cache_entry(setup_v9):
    """The jitted decoder consumes the whole input cache, the selection entries included."""
    cfg, params, consts, feats, _, _ = setup_v9
    cache = G.init_cache(cfg, 2, TMAX)
    _, _, new = G.jitted(cfg).prefill(params, consts, cache, _slice(feats, slice(0, 8)))
    assert all(cache[name].is_deleted() for name in cache)
    assert set(new) == set(cache) and not new["sel"].is_deleted()


@pytest.mark.slow
def test_v9_bf16_decode_runs_and_stays_close():
    cfg = tiny_cfg(act_dtype="bfloat16", **V9)
    params = random_params(cfg, seed=38, scale=0.1)
    consts = M.make_consts(cfg)
    feats = _v9_batch(cfg, T=16, seed=39)
    full = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    z, cache = incremental(G.jitted(cfg), params, consts, cfg, feats, split=6, tmax=32)
    assert np.asarray(cache["k"]).dtype.name == "bfloat16" and np.asarray(cache["sel"]).dtype == np.float32
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z - full[:, 5:])) < 0.15


def test_v9_perturbation_causality(setup_v9):
    """Perturbing token j (and its feature row) leaves every score before j bit-identical and
    changes the later scores of its segment."""
    cfg, params, consts, feats, _, _ = setup_v9
    dec = G.jitted(cfg, donate=False)
    j = 9
    pert = _perturb(feats, cfg, j)
    za, _ = incremental(dec, params, consts, cfg, feats, split=1)
    zb, _ = incremental(dec, params, consts, cfg, pert, split=1)
    np.testing.assert_array_equal(zb[:, :j], za[:, :j])
    diff = np.max(np.abs(zb - za), axis=-1)
    same_seg = feats["seg"] == feats["seg"][:, j:j + 1]
    later = np.arange(diff.shape[1])[None] >= j
    assert np.min(diff[later & same_seg]) > 1e-4
    np.testing.assert_array_equal(diff[later & ~same_seg], 0.0)
