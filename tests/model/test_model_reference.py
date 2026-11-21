"""With every feature switch off the model equals a plain looped transformer with value
embeddings, written here independently in numpy (float64).

With a candidate tape in the inputs the model's scores are the output law's normalised
log-probabilities (``softmax`` of them is the distribution), so distributions are compared
through log-softmax."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from _model_testlib import jitted, make_batch, random_params, tiny_cfg, to_numpy
from bes.model import model as M


def _rms(x, g=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return y if g is None else y * g


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def plain_looped_reference(P: dict, tokens: np.ndarray, seg: np.ndarray, cfg) -> np.ndarray:
    """Plain looped transformer (hyper-connection lanes, value embeddings in block 0, RoPE on
    position planes, unrotated clock planes, QK-norm with per-head gain, causal same-segment
    mask, ReLU^2 MLP, factorised softcapped head). Returns logits ``[B, T, V]``."""
    B, T = tokens.shape
    H, hd, nb, L, N = cfg.n_heads, cfg.head_dim, cfg.n_blocks, cfg.n_loops, cfg.n_lanes
    dc = hd - cfg.bias_dims
    n_planes = dc // 2
    n_pos = n_planes - cfg.clock_planes
    pos = np.arange(T)
    ang = np.zeros((T, n_planes))
    ang[:, :n_pos] = pos[:, None] * cfg.rope_base ** (-np.arange(n_pos) / n_pos)[None]
    cos, sin = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]

    def rope(x):
        x1, x2 = x[..., :n_planes], x[..., n_planes:]
        return np.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)

    mask = (pos[None, :, None] >= pos[None, None, :]) & (seg[:, :, None] == seg[:, None, :])
    a, m, bt = P["lanes"]["alpha"], P["lanes"]["mix"], P["lanes"]["beta"]
    at, ml, nm = P["core"]["attn"], P["core"]["mlp"], P["core"]["norm"]
    ve_all = P["emb"]["ve"][tokens]                     # [B, T, L, ve]
    n_ve = cfg.ve_dim // hd
    e = _rms(P["emb"]["tok"][tokens])
    lanes = [e.copy() for _ in range(N)]
    for r in range(L):
        u = sum(a[r, n] * lanes[n] for n in range(N))
        for b in range(nb):
            x = _rms(u, nm["g1"][b])
            q = (x @ at["wq"][b]).reshape(B, T, H, dc)
            k = (x @ at["wk"][b]).reshape(B, T, H, dc)
            v = (x @ at["wv"][b]).reshape(B, T, H, hd)
            if b == 0:
                gate = 3.0 / (1.0 + np.exp(-(x[..., :12] @ at["ve_gate"])))
                v[:, :, :n_ve] += gate[..., None] * ve_all[:, :, r].reshape(B, T, n_ve, hd)
            q, k = rope(q), rope(k)
            q = q / np.linalg.norm(q, axis=-1, keepdims=True) * at["q_gain"][b][None, None, :, None]
            k = k / np.linalg.norm(k, axis=-1, keepdims=True)
            logits = np.einsum("bqhd,bkhd->bhqk", q, k)
            logits = np.where(mask[:, None], logits, -np.inf)
            y = np.einsum("bhqk,bkhd->bqhd", _softmax(logits), v).reshape(B, T, H * hd)
            u = u + y @ at["wo"][b]
            x2 = _rms(u, nm["g2"][b])
            u = u + np.maximum(x2 @ ml["w_up"][b], 0.0) ** 2 @ ml["w_down"][b]
        s = _rms(u, P["loop"]["g"][r])
        lanes = [sum(m[r, i, j] * lanes[j] for j in range(N)) + bt[r, i] * s for i in range(N)]
    h = _rms(sum(lanes), P["head"]["g"])
    z = (h @ P["head"]["proj"]) @ P["head"]["out"].T
    return cfg.softcap * np.tanh(z / cfg.softcap)


@pytest.mark.parametrize("clock_planes", [0, 3])
def test_all_switches_off_equals_plain_looped_transformer(clock_planes):
    cfg = tiny_cfg(clock_planes=clock_planes)
    params = random_params(cfg, seed=1)
    batch = make_batch(cfg, B=2, T=20, seed=2, seg_starts=((0,), (0, 9)))
    feats, _, _ = M.split_batch(batch)
    consts = M.make_consts(cfg, **M.switches_off())
    got = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    want = plain_looped_reference(to_numpy(params), np.asarray(feats["tokens"]), batch["seg"], cfg)
    assert got.shape == want.shape == (2, 20, cfg.vocab_size)
    np.testing.assert_allclose(_log_softmax(got), _log_softmax(want), atol=5e-5, rtol=0)
    # the feature paths are really switched off: turning them on changes the output
    on = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    assert np.max(np.abs(_log_softmax(on) - _log_softmax(got))) > 1e-2


def test_static_flags_off_equal_runtime_switches_off():
    """Static removal of every optional path gives the same logits as the runtime switches."""
    cfg = tiny_cfg()
    params = random_params(cfg, seed=3)
    feats, _, _ = M.split_batch(make_batch(cfg, seed=4))
    off = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, **M.switches_off()), feats))
    cfg_s = dataclasses.replace(cfg, use_sink=False, use_conf=False, use_reg=False,
                                use_cand_in=False, use_cand_out=False)
    p_s = {k: v for k, v in params.items() if k not in ("cand", "trust")}
    p_s["emb"] = {k: v for k, v in params["emb"].items() if k != "reg"}
    p_s["core"] = dict(params["core"])
    p_s["core"]["attn"] = {k: v for k, v in params["core"]["attn"].items()
                           if k not in ("sink_logit", "sink_value")}
    stat = np.asarray(jitted("logits_fn", cfg_s)(p_s, M.make_consts(cfg_s, clock=0.0), feats))
    np.testing.assert_allclose(_log_softmax(stat), _log_softmax(off), atol=1e-5, rtol=1e-5)
