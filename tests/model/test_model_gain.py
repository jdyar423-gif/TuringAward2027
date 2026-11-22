"""Token gain pages (``gain/{ffn,o,v}``) and selective attention in the full-sequence forward:
bit-identity with zero tables and with the switch off, an independent numpy reference (phases,
partial ``gain_ffn``, selective attention with several segments and BOS keys), float64
gradients of every gain leaf, bfloat16 products in float32, remat, causality."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from _model_testlib import init, jitted, make_batch, random_params, tiny_cfg, to_numpy, value_and_grad_fn
from bes.model import layers, model as M

GAINS = dict(gain_ffn=100, gain_o=True, gain_v=True, gain_phases=2)   # tiny_cfg: d_ffn 256, 3 loops


def _without_gain(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "gain"}


def _zero_gain(params: dict) -> dict:
    out = dict(params)
    out["gain"] = jax.tree.map(jnp.zeros_like, params["gain"])
    return out


def _feats(cfg, seed, **kw):
    feats, _, _ = M.split_batch(make_batch(cfg, seed=seed, **kw))
    return {k: np.asarray(v) for k, v in feats.items()}


def _on(cfg, **sw):
    """Switches of the old feature paths off, gain and selection on (unless overridden)."""
    return M.make_consts(cfg, **{**M.switches_off(), "gain": 1.0, "sel": 1.0, **sw})


# ------------------------------------------------------------------------- parameters
def test_gain_tables_shapes_roles_and_switch_defaults():
    cfg = tiny_cfg(**GAINS)
    params = init(cfg)
    hw = cfg.n_heads * cfg.head_dim
    assert {k: v.shape for k, v in params["gain"].items()} == {
        "ffn": (cfg.vocab_size, 2, cfg.n_blocks, 100), "o": (cfg.vocab_size, 2, cfg.n_blocks, hw),
        "v": (cfg.vocab_size, 2, cfg.n_blocks, hw)}
    assert all(v.dtype == jnp.float32 and float(jnp.max(jnp.abs(v))) == 0.0 for v in params["gain"].values())
    assert M.gain_widths(tiny_cfg(gain_o=True)) == {"o": hw}
    assert "gain" not in init(tiny_cfg())
    # every other leaf is the same as without the flags (per-leaf keys)
    plain = init(tiny_cfg())
    for a, b in zip(jax.tree.leaves(_without_gain(params)), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(M.make_consts(cfg)["enable"]["gain"]) == 1.0
    assert float(M.make_consts(tiny_cfg())["enable"]["gain"]) == 0.0
    assert float(M.make_consts(tiny_cfg(sel_attn=True))["enable"]["sel"]) == 1.0
    assert float(M.make_consts(tiny_cfg())["enable"]["sel"]) == 0.0
    off = M.make_consts(cfg, **M.switches_off())
    assert float(off["enable"]["gain"]) == float(off["enable"]["sel"]) == 0.0


def test_check_cfg_bounds():
    M.check_cfg(tiny_cfg(gain_ffn=256, gain_phases=3))
    for bad in (dict(gain_ffn=257), dict(gain_ffn=-1), dict(gain_phases=0), dict(gain_phases=4),
                dict(sel_attn=True, attn_impl="splash")):
        with pytest.raises(ValueError):
            M.check_cfg(tiny_cfg(**bad))


def test_loop_phases():
    assert M.loop_phases(tiny_cfg(gain_phases=2)).tolist() == [0, 0, 1]          # 3 loops
    assert M.loop_phases(tiny_cfg(gain_phases=3)).tolist() == [0, 1, 2]
    assert M.loop_phases(tiny_cfg(n_loops=8, gain_phases=3)).tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert M.loop_phases(tiny_cfg(n_loops=8)).tolist() == [0] * 8


# ------------------------------------------------------------------------ exactness
@pytest.mark.parametrize("flags", [GAINS, dict(gain_ffn=256), dict(gain_o=True, gain_phases=3),
                                   dict(gain_v=True, sel_attn=True)])
def test_zero_gain_tables_are_bit_identical_to_the_model_without_them(flags):
    """Same other leaves, gain tables at their zero initialisation: logits and loss equal the
    model without the gain flags bit for bit."""
    cfg = tiny_cfg(**flags)
    base = tiny_cfg(sel_attn=cfg.sel_attn)
    params = _zero_gain(random_params(cfg, seed=1))
    batch = make_batch(cfg, seed=2)
    feats, _, _ = M.split_batch(batch)
    a = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    b = np.asarray(jitted("logits_fn", base)(_without_gain(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(a, b)
    la, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg), batch)
    lb, _ = jitted("loss_fn", base)(_without_gain(params), M.make_consts(base), batch)
    assert float(la) == float(lb)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gain_switch_off_is_bit_identical_to_gains_ignored(dtype):
    cfg = tiny_cfg(act_dtype=dtype, **GAINS)
    base = tiny_cfg(act_dtype=dtype)
    params = random_params(cfg, seed=3, scale=0.1 if dtype == "bfloat16" else 0.3)
    feats = _feats(cfg, 4)
    off = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, gain=0.0), feats))
    ign = np.asarray(jitted("logits_fn", base)(_without_gain(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(off, ign)
    on = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    assert np.max(np.abs(on - off)) > 1e-2


def test_equal_phase_rows_equal_one_phase():
    """Two phases holding the same rows give exactly the single-phase model."""
    cfg2, cfg1 = tiny_cfg(**GAINS), tiny_cfg(**{**GAINS, "gain_phases": 1})
    p1 = random_params(cfg1, seed=5)
    p2 = dict(p1)
    p2["gain"] = {k: jnp.concatenate([v, v], axis=1) for k, v in p1["gain"].items()}
    feats = _feats(cfg1, 6)
    a = np.asarray(jitted("logits_fn", cfg2)(p2, M.make_consts(cfg2), feats))
    b = np.asarray(jitted("logits_fn", cfg1)(p1, M.make_consts(cfg1), feats))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    # the rows of phase 1 (the last loop only) matter: changing them changes the output
    p3 = dict(p2)
    p3["gain"] = {k: v.at[:, 1].add(0.5) for k, v in p2["gain"].items()}
    c = np.asarray(jitted("logits_fn", cfg2)(p3, M.make_consts(cfg2), feats))
    assert np.max(np.abs(c - a)) > 1e-3


def test_gain_rows_layout_and_dtype():
    cfg = tiny_cfg(act_dtype="bfloat16", **GAINS)
    params = random_params(cfg, seed=7)
    tokens = jnp.asarray(_feats(cfg, 8)["tokens"])
    rows = M.gain_rows(params, M.make_consts(cfg), tokens, cfg)
    b, t = tokens.shape
    for name, w in M.gain_widths(cfg).items():
        a = rows[name]
        assert a.shape == (2, cfg.n_blocks, b, t, w) and a.dtype == jnp.float32
        table = np.asarray(params["gain"][name])
        np.testing.assert_array_equal(np.asarray(a[1, 0, 1, 3]), table[int(tokens[1, 3]), 1, 0])
    half = M.gain_rows(params, M.make_consts(cfg, gain=0.5), tokens, cfg)
    np.testing.assert_array_equal(np.asarray(half["o"]), 0.5 * np.asarray(rows["o"]))
    assert M.gain_rows(init(tiny_cfg()), M.make_consts(tiny_cfg()), tokens, tiny_cfg()) is None
    blk = M.block_gain(rows, 1, jnp.asarray(1, jnp.int32))      # phase 1, block 1
    assert blk["ffn"].shape == rows["ffn"][1, 1].shape
    x = jnp.asarray(np.random.default_rng(9).normal(size=blk["ffn"].shape), jnp.float32)
    np.testing.assert_array_equal(np.asarray(layers.gain_product(x, blk["ffn"], jnp.float32)),
                                  np.asarray(x * (1.0 + rows["ffn"][1, 1])))
    assert set(M.block_gain(rows, 0)) == {"ffn", "o", "v"} and M.block_gain(None, 0) is None


# ------------------------------------------------------------------- numpy reference
def _rms(x, g=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return y if g is None else y * g


def _selection_offsets(logits0, mask, bos):
    """``F[b, i, j] = sum_{i' < i} S[b, i', j]`` with ``S = relu(head-0 logit)`` on visible pairs
    whose key is neither the query's own slot nor a BOS token (loops, float64)."""
    b_n, t_n, _ = logits0.shape
    s = np.zeros_like(logits0)
    for b in range(b_n):
        for i in range(t_n):
            for j in range(t_n):
                if mask[b, i, j] and i != j and not bos[b, j]:
                    s[b, i, j] = max(logits0[b, i, j], 0.0)
    f = np.zeros_like(s)
    for i in range(1, t_n):
        f[:, i] = f[:, i - 1] + s[:, i - 1]
    return f


def looped_reference(P: dict, tokens: np.ndarray, seg: np.ndarray, cfg, sel: bool) -> np.ndarray:
    """Plain looped transformer (every old feature path off, plain RoPE) with token gains in
    phase ``r * gain_phases // n_loops`` of loop ``r`` and, if ``sel``, selective attention.
    Written from the formulas of the specification; returns logits ``[B, T, V]``."""
    B, T = tokens.shape
    H, hd, nb, L, N = cfg.n_heads, cfg.head_dim, cfg.n_blocks, cfg.n_loops, cfg.n_lanes
    dc = hd - cfg.bias_dims
    n_planes = dc // 2
    ang = np.arange(T)[:, None] * cfg.rope_base ** (-np.arange(n_planes) / n_planes)[None]
    cos, sin = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]

    def rope(x):
        x1, x2 = x[..., :n_planes], x[..., n_planes:]
        return np.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)

    pos = np.arange(T)
    mask = (pos[None, :, None] >= pos[None, None, :]) & (seg[:, :, None] == seg[:, None, :])
    bos = tokens == M.bos_id(cfg)
    rows = {name: P["gain"][name][tokens] for name in P.get("gain", {})}     # [B, T, P, nb, w]
    a, m, bt = P["lanes"]["alpha"], P["lanes"]["mix"], P["lanes"]["beta"]
    at, ml, nm = P["core"]["attn"], P["core"]["mlp"], P["core"]["norm"]
    ve_all = P["emb"]["ve"][tokens]
    n_ve = cfg.ve_dim // hd
    lanes = [_rms(P["emb"]["tok"][tokens]) for _ in range(N)]
    for r in range(L):
        ph = r * cfg.gain_phases // L
        u = sum(a[r, n] * lanes[n] for n in range(N))
        for b in range(nb):
            x = _rms(u, nm["g1"][b])
            q = rope((x @ at["wq"][b]).reshape(B, T, H, dc))
            k = rope((x @ at["wk"][b]).reshape(B, T, H, dc))
            v = x @ at["wv"][b]
            if "v" in rows:
                v = v * (1.0 + rows["v"][:, :, ph, b])
            v = v.reshape(B, T, H, hd)
            if b == 0:
                gate = 3.0 / (1.0 + np.exp(-(x[..., :12] @ at["ve_gate"])))
                v[:, :, :n_ve] += gate[..., None] * ve_all[:, :, r].reshape(B, T, n_ve, hd)
            q = q / np.linalg.norm(q, axis=-1, keepdims=True) * at["q_gain"][b][None, None, :, None]
            k = k / np.linalg.norm(k, axis=-1, keepdims=True)
            logits = np.einsum("bqhd,bkhd->bhqk", q, k)
            if sel:
                logits = logits - _selection_offsets(logits[:, 0], mask, bos)[:, None]
            logits = np.where(mask[:, None], logits, -np.inf)
            p = np.exp(logits - logits.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            y = np.einsum("bhqk,bkhd->bqhd", p, v).reshape(B, T, H * hd)
            if "o" in rows:
                y = y * (1.0 + rows["o"][:, :, ph, b])
            u = u + y @ at["wo"][b]
            hh = np.maximum(_rms(u, nm["g2"][b]) @ ml["w_up"][b], 0.0) ** 2
            if "ffn" in rows:
                n = rows["ffn"].shape[-1]
                hh[..., :n] = hh[..., :n] * (1.0 + rows["ffn"][:, :, ph, b])
            u = u + hh @ ml["w_down"][b]
        s = _rms(u, P["loop"]["g"][r])
        lanes = [sum(m[r, i, j] * lanes[j] for j in range(N)) + bt[r, i] * s for i in range(N)]
    h = _rms(sum(lanes), P["head"]["g"])
    z = (h @ P["head"]["proj"]) @ P["head"]["out"].T
    return cfg.softcap * np.tanh(z / cfg.softcap)


@pytest.mark.parametrize("sel", [False, True])
def test_gains_and_selective_attention_match_the_numpy_reference(sel):
    """Two phases over three loops, a partial FFN gain, value and output gains, three segments
    per row (BOS at every segment start and one more BOS inside a segment)."""
    cfg = tiny_cfg(clock_planes=0, sel_attn=sel, **GAINS)
    params = random_params(cfg, seed=9)
    batch = make_batch(cfg, B=2, T=20, seed=10, seg_starts=((0, 7, 13), (0, 4)))
    batch["tokens"][0, 10] = M.bos_id(cfg)              # a BOS token that starts no segment
    tokens, seg = batch["tokens"][:, :-1], batch["seg"]
    got = np.asarray(jitted("logits_fn", cfg)(params, _on(cfg), {"tokens": tokens, "seg": seg}))
    want = looped_reference(to_numpy(params), tokens, seg, cfg, sel)
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)
    # the reference really depends on the gains and on the selection
    plain = looped_reference(to_numpy(_without_gain(params)), tokens, seg, cfg, False)
    assert np.max(np.abs(want - plain)) > 1e-2


# ------------------------------------------------------------------------- gradients
@pytest.mark.parametrize("table_grad", ["onehot", "sorted"])
def test_check_grads_of_every_gain_leaf_float64(table_grad):
    """Reverse-mode gradients of the loss with respect to every gain table (two phases, a
    partial FFN gain, selective attention on) against finite differences, float64."""
    cfg32 = tiny_cfg(vocab_size=64, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                     n_loops=2, cand_dim=8, critic_hidden=8, loss_chunk=7, gain_ffn=50,
                     gain_o=True, gain_v=True, gain_phases=2, sel_attn=True,
                     table_grad=table_grad, sorted_chunk=3)
    params32 = random_params(cfg32, seed=11)
    batch = make_batch(cfg32, B=2, T=10, seed=12, seg_starts=((0,), (0, 6)))
    with jax.enable_x64(True):
        cfg = dataclasses.replace(cfg32, act_dtype="float64")
        params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), params32)
        consts = M.make_consts(cfg)
        rest = {k: v for k, v in params.items() if k != "gain"}
        f = jax.jit(lambda g: M.loss_fn({**rest, "gain": g}, consts, batch, cfg)[0])
        assert f(params["gain"]).dtype == jnp.float64
        grads = jax.grad(f)(params["gain"])
        assert all(float(jnp.max(jnp.abs(g))) > 0 for g in jax.tree.leaves(grads))
        check_grads(f, (params["gain"],), order=1, modes=["rev"], eps=1e-6)


def _grad_temp_bytes(cfg, b: int = 4, t: int = 64) -> int:
    """Compiler temp bytes of the loss gradient (compile only, no arrays)."""
    import functools
    params = jax.eval_shape(functools.partial(M.init_params, cfg), jax.random.PRNGKey(0))
    consts = M.make_consts(cfg)
    batch = {"tokens": jax.ShapeDtypeStruct((b, t + 1), jnp.int32), "w": jax.ShapeDtypeStruct((b, t), jnp.float32)}
    fn = jax.jit(jax.grad(lambda p, x: M.loss_fn(p, consts, x, cfg)[0]))
    return fn.lower(params, batch).compile().memory_analysis().temp_size_in_bytes


def test_gain_rows_are_not_copied_per_loop_under_remat():
    """The gathered rows are closed over by the loop scan: with remat the gains' extra memory
    depends on the number of phases, not on the number of loops (one copy per loop of the
    ``[n_blocks, B, T, 320]`` rows here would add about 0.66 MB per loop)."""
    for p in (1, 2):
        extra = []
        for loops in (2, 6):
            base = tiny_cfg(n_loops=loops, remat=True, use_cand_in=False, use_cand_out=False, use_reg=False)
            gained = dataclasses.replace(base, gain_ffn=256, gain_o=True, gain_phases=p)
            extra.append(_grad_temp_bytes(gained) - _grad_temp_bytes(base))
        assert extra[1] <= 1.05 * extra[0] + 65536, (p, extra)


def test_phase_rows_are_not_copied_per_loop_without_remat():
    """Without remat (a plan of the start-up benchmark) the backward pass keeps every loop's
    activations, but no copy of the phase-selected gain rows: two phases cost what one phase
    costs per loop. A float32 ``[B, T, width]`` slice per loop, block and table (the phase index
    is traced) would add 0.75 MiB per loop here."""
    b, t = 4, 64
    copy = 2 * b * t * (256 + 2 * 64) * 4                     # 2 blocks, gain_ffn + gain_o + gain_v
    slope = {}
    for p in (1, 2):
        temp = []
        for loops in (2, 6):
            cfg = tiny_cfg(n_loops=loops, remat=False, use_cand_in=False, use_cand_out=False,
                           use_reg=False, gain_ffn=256, gain_o=True, gain_v=True, gain_phases=p)
            temp.append(_grad_temp_bytes(cfg, b, t))
        slope[p] = (temp[1] - temp[0]) / 4
    assert slope[2] <= slope[1] + 0.1 * copy, (slope, copy)


def test_remat_gives_the_same_loss_and_gradients_with_gains_and_selection():
    cfg = tiny_cfg(sel_attn=True, **GAINS)
    params = random_params(cfg, seed=13)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=14)
    out = {remat: value_and_grad_fn(dataclasses.replace(cfg, remat=remat))(params, consts, batch)
           for remat in (False, True)}
    np.testing.assert_allclose(float(out[True][0][0]), float(out[False][0][0]), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(out[True][1]), jax.tree.leaves(out[False][1])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5)
    assert all(float(jnp.max(jnp.abs(g))) > 0 for g in jax.tree.leaves(out[True][1]["gain"]))


# --------------------------------------------------------------------------- bfloat16
def test_bfloat16_gain_products_are_computed_in_float32():
    """A delta below the bfloat16 resolution of 1 (1 + 1e-3 rounds to 1 in bfloat16) still
    changes the result: the product is formed in float32 before the cast."""
    rng = np.random.default_rng(15)
    x = jnp.asarray(rng.normal(size=(4, 16)), jnp.bfloat16)
    w_up = jnp.asarray(rng.normal(size=(16, 32)) / 4, jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(size=(32, 16)) / 6, jnp.bfloat16)
    small = jnp.full((4, 32), 1e-3, jnp.float32)
    assert float(jnp.asarray(1.0 + 1e-3, jnp.bfloat16)) == 1.0
    plain = np.asarray(layers.mlp(x, w_up, w_down, jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(layers.mlp(x, w_up, w_down, jnp.bfloat16, gain=0 * small)), plain)
    assert np.max(np.abs(np.asarray(layers.mlp(x, w_up, w_down, jnp.bfloat16, gain=small)) - plain)) > 0
    cfg = tiny_cfg(act_dtype="bfloat16", **GAINS)
    params = _zero_gain(random_params(cfg, seed=16, scale=0.1))
    nudged = dict(params)
    nudged["gain"] = jax.tree.map(lambda a: a + 1e-3, params["gain"])
    feats = _feats(cfg, 17)
    f = jitted("logits_fn", cfg)
    a = np.asarray(f(params, M.make_consts(cfg), feats))
    b = np.asarray(f(nudged, M.make_consts(cfg), feats))
    assert np.max(np.abs(a - b)) > 0


# --------------------------------------------------------------------------- causality
def test_causality_and_segment_isolation_with_gains_and_selection():
    cfg = tiny_cfg(sel_attn=True, **GAINS)
    params = random_params(cfg, seed=18)
    consts = M.make_consts(cfg)
    feats = _feats(cfg, 19, B=1, T=24, seg_starts=((0, 9),))
    f = jitted("logits_fn", cfg)
    base = np.asarray(f(params, consts, feats))
    for j in (3, 12, 23):
        pert = {k: v.copy() for k, v in feats.items()}
        pert["tokens"][:, j] = (pert["tokens"][:, j] + 5) % (cfg.vocab_size - 16)
        out = np.asarray(f(params, consts, pert))
        np.testing.assert_array_equal(out[:, :j], base[:, :j])
        assert np.max(np.abs(out[:, j] - base[:, j])) > 1e-4
    # the second segment alone at the start of a row: the same logits (selection is per segment)
    alone = {k: v[:, 9:] for k, v in feats.items()}
    alone["seg"] = np.zeros_like(alone["seg"])
    np.testing.assert_allclose(np.asarray(f(params, consts, alone)), base[:, 9:], atol=2e-4, rtol=1e-4)
