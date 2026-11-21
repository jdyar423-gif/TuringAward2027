"""Loss: chunked float32 cross-entropy of the output law, its metrics and its gradients."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from _model_testlib import (jitted, make_batch, random_params, tiny_cfg, to_numpy,
                            value_and_grad_fn)
from bes.config import CAND_FEATS
from bes.model import head, model as M, predictive
from bes.roles import leaves_with_paths

FEATS = [name for name, _ in CAND_FEATS]
KIND, SRC = FEATS.index("kind"), FEATS.index("src")


def test_zero_masses_give_the_plain_cross_entropy_bit_for_bit():
    """The cand_out switch off, every kind switched off, or every src multiplier at zero: the
    loss equals that of the model without the tape code path, bit for bit, and the tape
    metrics are zero; with the tape on the loss differs."""
    cfg = tiny_cfg()
    cfg_none = dataclasses.replace(cfg, use_cand_out=False)
    batch = make_batch(cfg, seed=11)
    params = random_params(cfg, seed=12)              # trained-looking trust tables
    loss_none, _ = jitted("loss_fn", cfg_none)(params, M.make_consts(cfg_none), batch)
    # candidate input off in these runs so that only the output side is compared (the kind
    # switches act on both sides)
    ref, _ = jitted("loss_fn", cfg_none)(params, M.make_consts(cfg_none, cand_in=0.0), batch)
    for switch in ({"cand_out": 0.0}, {"kind": 0.0}, {"src": 0.0}):
        consts = M.make_consts(cfg, cand_in=0.0, **switch)
        loss, met = jitted("loss_fn", cfg)(params, consts, batch)
        assert np.asarray(loss).tobytes() == np.asarray(ref).tobytes(), switch
        assert float(met["tape_mass"]) == 0.0 and float(met["tape_gain"]) == 0.0, switch
    loss_off, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg, cand_out=0.0), batch)
    assert np.asarray(loss_off).tobytes() == np.asarray(loss_none).tobytes()
    loss_on, met = jitted("loss_fn", cfg)(params, M.make_consts(cfg), batch)
    assert abs(float(loss_on) - float(loss_none)) > 1e-4
    assert float(met["tape_mass"]) > 0.0


def test_chunked_loss_equals_unchunked():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=13)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=24, seed=14)
    assert cfg.loss_chunk >= 2 * 24                     # the reference runs as a single chunk
    (l_big, _), g_big = value_and_grad_fn(cfg)(params, consts, batch)
    # 7 rows per chunk: 7 chunks, the last one padded
    (l_c, _), g_c = value_and_grad_fn(dataclasses.replace(cfg, loss_chunk=7))(params, consts, batch)
    np.testing.assert_allclose(float(l_c), float(l_big), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(g_c), jax.tree.leaves(g_big)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5)


def _np_log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _plant_hits(batch, cfg):
    """Make the target a candidate at some positions (slot 0, and twice at a few)."""
    y = batch["tokens"][:, 1:]
    batch["cand_tok"][:, ::2, 0] = y[:, ::2]
    batch["cand_tok"][:, ::5, 7] = y[:, ::5]
    batch["cand_feat"][:, ::2, 0, KIND] = 1
    batch["cand_feat"][:, ::5, 7, KIND] = 4
    batch["cand_tok"][:, 1::6, 3] = cfg.vocab_size + 2          # ids beyond the vocabulary
    return batch


def test_loss_metrics_and_scores_match_an_independent_reference():
    """numpy float64 from the hidden states: network logits, masses of the trust tables with
    a kind switched off and a src multiplier, the law p, the weighted loss, the tape metrics,
    the per-role sums and the scores of ``logits_fn`` (log p)."""
    cfg = tiny_cfg(mass_offset=3.0)
    params = random_params(cfg, seed=15)
    consts = M.make_consts(cfg, kind_2=0.0, src_5=0.25)
    batch = _plant_hits(make_batch(cfg, B=2, T=24, seed=16), cfg)
    feats, y, w = M.split_batch(batch)
    loss, met = jitted("loss_fn", cfg)(params, consts, batch)
    hidden = np.asarray(jitted("forward_hidden", cfg)(params, consts, feats), np.float64)
    P = to_numpy(params)
    V = cfg.vocab_size
    z = (hidden @ P["head"]["proj"]) @ P["head"]["out"].T
    z = cfg.softcap * np.tanh(z / cfg.softcap)
    logq = _np_log_softmax(z)
    cand, feat = batch["cand_tok"], batch["cand_feat"].astype(np.int64)
    c = sum(P["trust"]["feat"][name][feat[..., i]] for i, name in enumerate(FEATS))
    beta = sum(P["trust"]["bias"][name][feat[..., i]] for i, name in enumerate(FEATS))
    a = beta + np.einsum("btc,btkc->btk", hidden @ P["trust"]["proj"], c) - cfg.mass_offset
    kind_on, src = np.ones(8), np.ones(8)
    kind_on[2], src[5] = 0.0, 0.25
    weight = (cand >= 0) * (cand < V) * kind_on[feat[..., KIND]] * src[feat[..., SRC]]
    m = np.exp(np.minimum(a, cfg.mass_log_cap)) * weight
    p = np.exp(logq)
    for bi in range(cand.shape[0]):
        for t in range(cand.shape[1]):
            for k in range(cand.shape[2]):
                if 0 <= cand[bi, t, k] < V:
                    p[bi, t, cand[bi, t, k]] += m[bi, t, k]
    p /= (1.0 + m.sum(-1))[..., None]
    yy = np.asarray(y)
    ce = -np.log(np.take_along_axis(p, yy[..., None], -1)[..., 0])
    ce_base = -np.take_along_axis(logq, yy[..., None], -1)[..., 0]
    wn = batch["w"].astype(np.float64)
    np.testing.assert_allclose(float(loss), (ce * wn).sum() / wn.sum(), rtol=2e-5)
    np.testing.assert_allclose(float(met["weight"]), wn.sum())
    np.testing.assert_allclose(float(met["tokens"]), (wn > 0).sum())
    np.testing.assert_allclose(float(met["tape_mass"]), (wn * m.sum(-1)).sum() / wn.sum(), rtol=1e-4)
    gain = (wn * (ce_base - ce)).sum() / wn.sum()
    np.testing.assert_allclose(float(met["tape_gain"]), gain, rtol=1e-3, atol=1e-5)
    assert abs(gain) > 1e-2
    # logits_fn gives the law's log-probabilities
    got = np.asarray(jitted("logits_fn", cfg)(params, consts, feats), np.float64)
    np.testing.assert_allclose(got, np.log(p), atol=5e-5)
    # loss share by role sums to one and matches the per-role sums
    role = batch["reg"][..., 0]
    by_role = np.array([(ce * wn)[role == r].sum() for r in range(6)])
    np.testing.assert_allclose(np.asarray(met["ce_by_role"]), by_role, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(met["loss_share_by_role"]).sum(), 1.0, rtol=1e-5)


def dense_reference_loss(params, consts, batch, cfg):
    """The output law written densely, for autodiff: unchunked logits, gathered trust rows,
    one-hot candidates and an explicitly normalised p (shares only the trunk with the model)."""
    feats, y, w = M.split_batch(batch)
    h = M.forward_hidden(params, consts, feats, cfg)
    z = (h @ params["head"]["proj"]) @ params["head"]["out"].T
    z = cfg.softcap * jnp.tanh(z / cfg.softcap)
    q = jax.nn.softmax(z, axis=-1)
    ct = jnp.asarray(batch["cand_tok"])
    cf = jnp.asarray(batch["cand_feat"]).astype(jnp.int32)
    tr = params["trust"]
    c = sum(tr["feat"][name][cf[..., i]] for i, name in enumerate(FEATS))
    beta = sum(tr["bias"][name][cf[..., i]] for i, name in enumerate(FEATS))
    a = beta + jnp.einsum("btc,btkc->btk", h @ tr["proj"], c) - cfg.mass_offset
    en = consts["enable"]
    weight = (((ct >= 0) & (ct < cfg.vocab_size)).astype(jnp.float32) * en["cand_out"]
              * en["kind"][cf[..., KIND]] * en["src"][cf[..., SRC]])
    m = jnp.exp(jnp.minimum(a, cfg.mass_log_cap)) * weight
    onehot = jax.nn.one_hot(ct, cfg.vocab_size, dtype=jnp.float32)   # -1 and >= V: zero rows
    p = (q + jnp.einsum("btk,btkv->btv", m, onehot)) / (1.0 + jnp.sum(m, -1))[..., None]
    ce = -jnp.log(jnp.take_along_axis(p, y[..., None], -1)[..., 0])
    return jnp.sum(w * ce) / jnp.sum(w)


def test_loss_and_all_gradients_match_a_dense_autodiff_reference():
    """Every parameter gradient of the production loss (chunks of 7 rows with padding, the
    compare/select target pick, the multi-hot trust lookup, per-token mixing) equals plain
    autodiff of the dense reference."""
    cfg = tiny_cfg(mass_offset=3.0, loss_chunk=7)
    params = random_params(cfg, seed=17)
    consts = M.make_consts(cfg, kind_3=0.0, src_5=0.5)
    batch = _plant_hits(make_batch(cfg, B=2, T=24, seed=18), cfg)
    (loss, _), grads = value_and_grad_fn(cfg)(params, consts, batch)
    ref, ref_grads = jax.jit(jax.value_and_grad(
        lambda p: dense_reference_loss(p, consts, batch, cfg)))(params)
    np.testing.assert_allclose(float(loss), float(ref), rtol=1e-5)
    seen = set()
    for (path, g), r in zip(leaves_with_paths(grads), jax.tree.leaves(ref_grads)):
        g, r = np.asarray(g), np.asarray(r)
        scale = float(np.max(np.abs(r))) if r.size else 0.0
        np.testing.assert_allclose(g, r, rtol=1e-3, atol=2e-4 * scale + 1e-9, err_msg=path)
        if scale > 0:
            seen.add(path.split("/")[0])
    assert {"trust", "head", "core", "cand", "emb"} <= seen


def test_chunked_ce_memory_stays_per_chunk():
    """The compiled per-token loss (chunked cross-entropy plus the tape's per-token mixing)
    keeps O(chunk * V) temporaries, never [n, K, V] or [n, V] for all n."""
    cfg = tiny_cfg(vocab_size=4096, loss_chunk=256)
    n, d, K = 4096, cfg.d_model, 16
    ph = {"proj": jnp.zeros((d, cfg.head_rank)), "out": jnp.zeros((cfg.vocab_size, cfg.head_rank))}

    def f(ph, h, y, cand, m):
        ce_base = head.chunked_ce(ph, h, y, cfg)
        return jnp.sum(predictive.mix_ce(ce_base, y, cand, m, cfg.vocab_size))

    args = (ph, jnp.zeros((n, d)), jnp.zeros((n,), jnp.int32), jnp.zeros((n, K), jnp.int32),
            jnp.zeros((n, K)))
    compiled = jax.jit(jax.grad(f, argnums=(0, 1, 4))).lower(*args).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    full_logits = n * cfg.vocab_size * 4
    assert temp < full_logits / 2, (temp, full_logits)
