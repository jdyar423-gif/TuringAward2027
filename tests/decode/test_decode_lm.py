"""The project's model behind the decoder's interface (``bes.decode.lm.OurLM``).

A smoke-sized model on the committed tokenizer with the exact-layer features. Checked here:
the decode consts (the self-evidence multiplier set, every other switch kept), the temperature
on the network's distribution only (against the full forward of ``bes.model.model`` and the
tempered law written here from its parts), and the fixed cache length that lets every reply of
a process share one compiled decoder.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest

from bes.config import ASSISTANT, BOS, END, USER, DecodeCfg, load_config
from bes.decode import sampling as S
from bes.decode import smc
from bes.decode.lm import OurLM, TableLM
from bes.model import generate as G
from bes.model import model as M

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model():
    from bes.tokenizer import load

    tok = load()
    cfg = load_config(str(REPO / "configs" / "smoke.py")).model
    cfg = dataclasses.replace(cfg, n_loops=2, max_seq=256)
    params = jax.tree.map(np.array, M.init_params(cfg, jax.random.PRNGKey(0)))
    rng = np.random.default_rng(0)
    params["head"]["out"] = params["head"]["out"] + rng.standard_normal(
        params["head"]["out"].shape).astype(np.float32) * 0.5
    params["trust"]["bias"]["src"][5] = 9.0          # strong evidence from the turn itself
    params["trust"]["bias"]["kind"][1:5] = 2.0
    return tok, cfg, params


def _self_repeating_prompt(tok):
    return [BOS, USER] + tok.encode("count") + [END, ASSISTANT] + \
        tok.encode("red blue green red blue green red")


def test_decode_consts_set_the_self_evidence_multiplier_and_keep_every_other_switch(model):
    from bes.decode.lm import decode_consts

    _, cfg, _ = model
    consts = M.make_consts(cfg, src_3=0.5, kind_2=0.0, sink=0.0)
    before = {k: np.asarray(v).copy() for k, v in consts["enable"].items()}
    out = decode_consts(consts, 0.25)
    src = np.asarray(out["enable"]["src"])
    want = np.ones(8, np.float32)
    want[3], want[5] = 0.5, 0.25
    np.testing.assert_array_equal(src, want)
    assert src.dtype == np.float32
    for k, v in before.items():                       # the input is not modified
        np.testing.assert_array_equal(np.asarray(consts["enable"][k]), v)
        if k != "src":
            np.testing.assert_array_equal(np.asarray(out["enable"][k]), v)
    np.testing.assert_array_equal(np.asarray(out["rope_freq"]), np.asarray(consts["rope_freq"]))
    # make_consts(cfg, src_5=w) semantics on given consts
    ref = M.make_consts(cfg, src_3=0.5, kind_2=0.0, sink=0.0, src_5=0.25)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        decode_consts(consts, -0.5)


def test_ourlm_decodes_with_the_configured_self_evidence_weight(model):
    """By default OurLM decodes with ``DecodeCfg().self_evidence_weight`` (the checkpoint's
    consts carry 1), a per-request weight replaces it, and the weight changes the rows where
    the tape holds evidence from the turn being generated."""
    tok, cfg, params = model
    consts = M.make_consts(cfg)                       # what a checkpoint stores: src_5 = 1
    prompt = _self_repeating_prompt(tok)
    lm = OurLM(params, consts, cfg, tok, max_steps=8)
    assert float(np.asarray(lm.consts["enable"]["src"])[5]) == DecodeCfg().self_evidence_weight == 0.25
    assert float(np.asarray(consts["enable"]["src"])[5]) == 1.0
    rows = {}
    for w in (1.0, 0.25, 0.0):
        assert lm.set_self_evidence_weight(w) is True
        assert float(np.asarray(lm.consts["enable"]["src"])[5]) == w
        rows[w] = lm.prefill(prompt, 1)[0]
    assert np.max(np.abs(rows[1.0] - rows[0.25])) > 0.1
    assert np.max(np.abs(rows[0.25] - rows[0.0])) > 0.01
    explicit = OurLM(params, consts, cfg, tok, max_steps=8, self_evidence_weight=1.0)
    np.testing.assert_allclose(explicit.prefill(prompt, 1)[0], rows[1.0], atol=1e-6)
    no_tape = OurLM(params, M.make_consts(cfg, cand_out=0.0), cfg, tok, max_steps=8)
    assert no_tape.set_self_evidence_weight(0.5) is False        # nothing to weight
    # the DecodeCfg-configured entry point hands its weight to the model
    smc.run(explicit, prompt, tok=tok, cfg=DecodeCfg(particles=1, max_new_tokens=2,
                                                     self_evidence_weight=0.5), potentials=False)
    assert float(np.asarray(explicit.consts["enable"]["src"])[5]) == 0.5


def test_ourlm_tempers_the_network_only(model):
    """``set_temperature(T)``: the rows are log p_T of the full forward at temperature T (the
    tape's masses untempered), not the tempered mixture; the decoder then reports log p_T and
    applies no second temperature."""
    tok, cfg, params = model
    lm = OurLM(params, M.make_consts(cfg), cfg, tok, max_steps=8)
    prompt = _self_repeating_prompt(tok)
    base = lm.prefill(prompt, 1)[0].astype(np.float64)
    assert lm.set_temperature(0.5) is True
    rows = lm.prefill(prompt, 1)[0].astype(np.float64)
    from bes.exact.features import exact_features

    seq = np.asarray(prompt, np.int32)[None]
    feats = {k: np.asarray(v) for k, v in exact_features(seq, None, tok).items()}
    feats["tokens"] = seq
    full = np.asarray(jax.jit(M.logits_fn, static_argnames="cfg")(
        params, lm.consts, feats, cfg=cfg, temperature=np.float32(0.5)))[0, -1]
    np.testing.assert_allclose(rows, S.log_softmax(full), atol=1e-4)
    mixture = S.log_softmax(base / 0.5)
    assert np.max(np.abs(rows - mixture)) > 0.1
    res = smc.decode(lm, prompt, n_particles=2, max_new_tokens=4, seed=1, tok=tok,
                     settings=S.SamplingSettings(temperature=0.5))
    assert res.extra == {"temperature": 0.5, "tempered_by": "model"} and lm.temperature == 0.5
    with pytest.raises(ValueError):
        lm.set_temperature(0.0)


def _sizes(dec):
    return tuple(getattr(f, "compiled", f)._cache_size() for f in (dec.prefill, dec.decode_step,
                                                                   dec.reorder))


def test_every_reply_shares_one_compiled_decoder(model):
    """Two replies with prompts of very different lengths, different temperatures and two
    OurLM objects: the cache length does not depend on the prompt, so no program is compiled
    again for the second reply; the step budget is ``max_steps``."""
    tok, cfg, params = model
    consts = M.make_consts(cfg)
    short = [BOS, USER] + tok.encode("Hi") + [END, ASSISTANT]
    long = [BOS, USER] + tok.encode(" ".join(["word"] * 190)) + [END, ASSISTANT]
    assert len(long) > 150 and len(long) <= cfg.max_seq
    settings = S.SamplingSettings(temperature=0.7)
    lm = OurLM(params, consts, cfg, tok, max_steps=16)
    smc.decode(lm, short, n_particles=3, max_new_tokens=8, seed=0, tok=tok, settings=settings,
               resample_at="token", ess_frac=1.01)
    sizes = _sizes(lm.dec)
    lm2 = OurLM(params, consts, cfg, tok, max_steps=16)
    assert lm2.dec is lm.dec                          # one decoder per configuration
    smc.decode(lm2, long, n_particles=3, max_new_tokens=8, seed=1, tok=tok, resample_at="token",
               ess_frac=1.01, settings=S.SamplingSettings(temperature=1.3))
    assert _sizes(lm2.dec) == sizes
    from bes.decode.lm import cache_slots

    assert G.cache_capacity(lm2.cache) == G.cache_capacity(lm.cache) == cache_slots(cfg.max_seq, 16)
    assert lm.capacity == lm2.capacity == 16


def test_ourlm_with_the_v9_paths_matches_the_full_forward(model):
    """A model with every v9 path (token gains, selective attention, whose selection the cache
    carries and reordering copies, the head-wise output gate, token neurons) behind OurLM: the
    rows of the prefill, of every step and after a resampling are the log-probabilities of the
    full forward over each particle's history (exact-layer features of the final sequence), and
    the decode consts keep every v9 switch."""
    from bes.decode.lm import decode_consts
    from bes.exact.features import exact_features

    tok, base_cfg, base_params = model
    cfg = dataclasses.replace(base_cfg, gain_ffn=64, gain_o=True, gain_v=True, gain_phases=2,
                              sel_attn=True, attn_gate=True, tok_neurons=4, tok_dim=32)
    params = jax.tree.map(np.array, M.init_params(cfg, jax.random.PRNGKey(0)))
    params["head"], params["trust"] = base_params["head"], base_params["trust"]
    rng = np.random.default_rng(2)
    for leaf in (*params["gain"].values(), params["page"]["val"], params["core"]["attn"]["gate_w"],
                 params["core"]["attn"]["gate_b"]):
        leaf += 0.3 * rng.standard_normal(leaf.shape).astype(np.float32)
    consts = M.make_consts(cfg)
    lm = OurLM(params, consts, cfg, tok, max_steps=8)
    for name in ("gain", "sel", "gate", "page"):
        assert float(lm.consts["enable"][name]) == 1.0
    np.testing.assert_array_equal(np.asarray(decode_consts(M.make_consts(cfg, sel=0.0), 0.5)["enable"]["sel"]), 0.0)
    prompt = _self_repeating_prompt(tok)
    k = 3
    logp = lm.prefill(prompt, k)
    assert set(lm.cache) >= {"sel", "bos"}
    hist, rows = [[] for _ in range(k)], [[logp[j]] for j in range(k)]
    first = np.argsort(-logp[0])[:k]
    for s in range(4):
        toks = first if s == 0 else np.argmax(logp, -1)
        for j in range(k):
            hist[j].append(int(toks[j]))
        logp = lm.step(toks)
        for j in range(k):
            rows[j].append(logp[j])
        if s == 1:                                   # resample: particle j continues anc[j]
            anc = np.array([2, 0, 0])
            lm.reorder(anc)
            hist, rows, logp = [list(hist[a]) for a in anc], [list(rows[a]) for a in anc], logp[anc]
    fwd = jax.jit(M.logits_fn, static_argnames="cfg")
    n = len(prompt)
    for j in range(k):
        seq = np.asarray(list(prompt) + hist[j], np.int32)[None]
        feats = {kk: np.asarray(v) for kk, v in exact_features(seq, None, tok).items()}
        feats["tokens"] = seq
        want = S.log_softmax(np.asarray(fwd(params, lm.consts, feats, cfg=cfg))[0])[n - 1:]
        np.testing.assert_allclose(np.stack(rows[j]), want, atol=1e-4)
    off = M.make_consts(cfg, sel=0.0)
    seq = np.asarray(list(prompt) + hist[0], np.int32)[None]
    feats = {kk: np.asarray(v) for kk, v in exact_features(seq, None, tok).items()}
    feats["tokens"] = seq
    assert np.max(np.abs(np.asarray(fwd(params, decode_consts(off, 0.25), feats, cfg=cfg))
                         - np.asarray(fwd(params, lm.consts, feats, cfg=cfg)))) > 1e-3


def test_stand_in_models_keep_the_decoder_temperature():
    tok_size = 20
    lm = TableLM.random(tok_size, order=1, seed=0)
    assert not hasattr(lm, "set_temperature")
    assert smc._lm_tempers(lm, 0.7) is False
