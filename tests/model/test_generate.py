"""Generation with a key-value cache: prefill + stepwise decode equals the full-sequence forward
(all features on, several split points, several segments, chunked and ragged prefill), causality,
masking of unused slots, cache reordering, the host loops and determinism."""
from __future__ import annotations

import numpy as np
import pytest

from _model_testlib import jitted, make_batch, random_params, tiny_cfg
from bes.config import END, special_id
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
    """All features on (registers, clocks, candidate input and tilt, sink, confidence, value
    embeddings); row 1 holds two segments (BOS at 0 and 11). Logits and hidden states."""
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
