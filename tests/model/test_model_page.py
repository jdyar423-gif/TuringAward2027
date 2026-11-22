"""Token neurons (``tok_neurons``, ``tok_dim``; part C of the v9 specification): initialisation,
the prepared rows (gathered, keys normalised, cast once outside the loop), bit-identity with zero
values and with the switch off, an independent numpy reference (also with the output gate and the
token gains), key-scale invariance, unit-variance pre-activations at initialisation, float64
gradients of the four leaves, remat, bfloat16, the rows kept in the activation dtype inside the
loop, the block interface that generation reuses (``attend_fn`` hook, loop index, rows) and the
final allocation on the splash path."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import init, jitted, make_batch, random_params, tiny_cfg, to_numpy, value_and_grad_fn
from _v9_testlib import check_directional_grads, reference_logits, rms, switches_on, tree_f64
from bes.model import attention, controller, head, layers, model as M

PAGE = dict(tok_neurons=3, tok_dim=24)
#: the allocation of configs/v9.py in tiny form (output token gains, token neurons) with the
#: optional output gate on
FINAL = dict(gain_o=True, attn_gate=True, table_grad="sorted", sorted_chunk=16, **PAGE)


def _without_page(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "page"}


def _zero_val(params: dict) -> dict:
    return {**params, "page": {**params["page"], "val": jnp.zeros_like(params["page"]["val"])}}


def _feats(cfg, seed, **kw):
    feats, _, _ = M.split_batch(make_batch(cfg, seed=seed, **kw))
    return {k: np.asarray(v) for k, v in feats.items()}


# ------------------------------------------------------------------------- parameters
def test_page_leaves_initialisation_and_switch_defaults():
    cfg = tiny_cfg(tok_neurons=4, tok_dim=48)
    params = init(cfg)
    page = params["page"]
    v, nb, d = cfg.vocab_size, cfg.n_blocks, cfg.d_model
    assert {k: a.shape for k, a in page.items()} == {
        "pin": (nb, d, 48), "pout": (nb, 48, d), "key": (v, nb, 4, 48), "val": (v, nb, 4, 48)}
    assert all(a.dtype == jnp.float32 for a in page.values())
    for name, fan_in in (("pin", d), ("pout", 48)):              # layers.uniform_init(fan_in)
        a = np.asarray(page[name])
        assert np.max(np.abs(a)) <= np.sqrt(3.0 / fan_in) and abs(a.std() * np.sqrt(fan_in) - 1) < 0.05
        assert np.all(np.linalg.norm(a, axis=(1, 2)) > 0)       # non-zero matrices (MuonH)
    key = np.asarray(page["key"])
    assert abs(key.mean()) < 0.01 and abs(key.std() - 1.0) < 0.01
    assert float(jnp.max(jnp.abs(page["val"]))) == 0.0
    plain = init(tiny_cfg())
    assert "page" not in plain
    assert jax.tree.structure(_without_page(params)) == jax.tree.structure(plain)
    for a, b in zip(jax.tree.leaves(_without_page(params)), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(M.make_consts(cfg)["enable"]["page"]) == 1.0
    assert float(M.make_consts(tiny_cfg())["enable"]["page"]) == 0.0
    assert float(M.make_consts(cfg, **M.switches_off())["enable"]["page"]) == 0.0
    M.check_cfg(cfg)
    for bad in (dict(tok_neurons=-1), dict(tok_neurons=2, tok_dim=0)):
        with pytest.raises(ValueError):
            M.check_cfg(tiny_cfg(**bad))


def test_token_rows_layout_dtype_and_chunks():
    cfg = tiny_cfg(act_dtype="bfloat16", **PAGE)
    params = random_params(cfg, seed=1)
    consts = M.make_consts(cfg, page=0.5)
    tokens = jnp.asarray(_feats(cfg, 2)["tokens"])
    rows = M.token_rows(params, consts, tokens, cfg)
    assert len(rows) == cfg.n_blocks
    tok = np.asarray(tokens)
    key_t = np.asarray(params["page"]["key"], np.float64)[tok]                # [B, T, nb, rho, dk]
    val_t = jnp.asarray(params["page"]["val"])[tok]
    for i, r in enumerate(rows):
        assert set(r) == {"key", "val", "on"} and float(r["on"]) == 0.5
        assert r["key"].shape == r["val"].shape == tok.shape + (3, 24)
        assert r["key"].dtype == r["val"].dtype == jnp.bfloat16
        k = np.asarray(r["key"], np.float64)
        np.testing.assert_allclose(k, rms(key_t[:, :, i]), rtol=1e-2, atol=1e-2)     # bfloat16 rounding
        np.testing.assert_allclose(np.sqrt(np.mean(k * k, axis=-1)), 1.0, atol=1e-2)
        np.testing.assert_array_equal(np.asarray(r["val"]), np.asarray(val_t[:, :, i].astype(jnp.bfloat16)))
    # the rows of a chunk of positions are the rows of those positions (no state across chunks)
    part = M.token_rows(params, consts, tokens[:, 5:9], cfg)
    for r, p in zip(rows, part):
        for name in ("key", "val"):
            np.testing.assert_array_equal(np.asarray(p[name]), np.asarray(r[name][:, 5:9]))
    assert M.token_rows(init(tiny_cfg()), M.make_consts(tiny_cfg()), tokens, tiny_cfg()) is None


# ------------------------------------------------------------------------ exactness
@pytest.mark.parametrize("flags", [PAGE, FINAL])
def test_zero_values_are_bit_identical_and_only_the_values_get_a_gradient(flags):
    """Random keys and projections, values at their zero initialisation: logits and loss equal
    the model without token neurons bit for bit; the gradient of ``page/key``, ``page/pin`` and
    ``page/pout`` is exactly zero, that of ``page/val`` is not."""
    cfg = tiny_cfg(**flags)
    base = tiny_cfg(**{k: v for k, v in flags.items() if k not in PAGE})
    params = _zero_val(random_params(cfg, seed=3))
    batch = make_batch(cfg, seed=4)
    feats, _, _ = M.split_batch(batch)
    a = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    b = np.asarray(jitted("logits_fn", base)(_without_page(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(a, b)
    (la, _), grads = value_and_grad_fn(cfg)(params, M.make_consts(cfg), batch)
    lb, _ = jitted("loss_fn", base)(_without_page(params), M.make_consts(base), batch)
    assert float(la) == float(lb)
    page = grads["page"]
    assert all(float(jnp.max(jnp.abs(page[n]))) == 0.0 for n in ("key", "pin", "pout"))
    assert float(jnp.max(jnp.abs(page["val"]))) > 0.0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_page_switch_off_is_bit_identical_to_the_token_neurons_ignored(dtype):
    cfg = tiny_cfg(act_dtype=dtype, **PAGE)
    base = tiny_cfg(act_dtype=dtype)
    params = random_params(cfg, seed=5, scale=0.1 if dtype == "bfloat16" else 0.3)
    feats = _feats(cfg, 6)
    off = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, page=0.0), feats))
    ign = np.asarray(jitted("logits_fn", base)(_without_page(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(off, ign)
    on = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    assert np.max(np.abs(on - off)) > 1e-2


# ------------------------------------------------------------------- numpy reference
@pytest.mark.parametrize("flags", [PAGE, FINAL, dict(gain_ffn=100, gain_v=True, gain_phases=2, **FINAL)])
def test_token_neurons_match_the_numpy_reference(flags):
    cfg = tiny_cfg(clock_planes=0, **flags)
    params = random_params(cfg, seed=7)
    batch = make_batch(cfg, B=2, T=20, seed=8, seg_starts=((0, 7, 13), (0,)))
    tokens, seg = batch["tokens"][:, :-1], batch["seg"]
    got = np.asarray(jitted("logits_fn", cfg)(params, switches_on(cfg), {"tokens": tokens, "seg": seg}))
    want = reference_logits(to_numpy(params), tokens, seg, cfg)
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)
    plain = reference_logits(to_numpy(_without_page(params)), tokens, seg, cfg)
    assert np.max(np.abs(want - plain)) > 1e-2


def test_key_scale_invariance():
    """Multiplying a neuron's key by a positive constant changes nothing (the keys are used
    RMS-normalised); a sign flip does change the output."""
    cfg = tiny_cfg(**FINAL)
    params = random_params(cfg, seed=9)
    key = params["page"]["key"]
    rng = np.random.default_rng(10)
    c = jnp.asarray(rng.choice([0.5, 2.0, 7.5, 40.0], size=key.shape[:-1] + (1,)), jnp.float32)
    consts = M.make_consts(cfg)
    feats = _feats(cfg, 11)
    f = jitted("logits_fn", cfg)
    a = np.asarray(f(params, consts, feats))
    b = np.asarray(f({**params, "page": {**params["page"], "key": key * c}}, consts, feats))
    np.testing.assert_allclose(b, a, rtol=2e-5, atol=2e-5)
    flipped = np.asarray(f({**params, "page": {**params["page"], "key": -key}}, consts, feats))
    assert np.max(np.abs(flipped - a)) > 1e-2


def test_preactivations_are_unit_variance_at_initialisation():
    """With the initial keys and projections and a unit-RMS input, the pre-activations
    ``<rmsnorm(key), rmsnorm(x @ pin)> / sqrt(dk)`` have mean 0 and variance 1 (statistically,
    16,384 values) and ``|a| <= sqrt(dk)``."""
    cfg = tiny_cfg(tok_neurons=8)                                    # tok_dim 192 (default)
    params = init(cfg)
    rng = np.random.default_rng(12)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size - 16, size=(16, 64)), jnp.int32)
    x = layers.rmsnorm(jnp.asarray(rng.standard_normal((16, 64, cfg.d_model)), jnp.float32))
    rows = M.token_rows(params, M.make_consts(cfg), tokens, cfg)
    a = np.concatenate([np.asarray(layers.token_preact(x, params["page"]["pin"][i], r["key"], jnp.float32)).ravel()
                        for i, r in enumerate(rows)])
    assert a.size == cfg.n_blocks * 16 * 64 * 8
    assert abs(a.mean()) < 0.03 and abs(a.var() - 1.0) < 0.05, (a.mean(), a.var())
    assert np.max(np.abs(a)) <= np.sqrt(cfg.tok_dim) * (1 + 1e-5)


# ------------------------------------------------------------------------- gradients
@pytest.mark.parametrize("table_grad", ["onehot", "sorted"])
def test_page_gradients_float64(table_grad):
    """Reverse-mode gradients of the loss with respect to the four token-neuron leaves at a
    random non-zero point (output gate and token gain on) against central differences in
    float64, with both scatter-free table gradients."""
    cfg32 = tiny_cfg(vocab_size=64, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                     n_loops=2, cand_dim=8, critic_hidden=8, loss_chunk=7, tok_neurons=3, tok_dim=8,
                     attn_gate=True, gain_o=True, table_grad=table_grad, sorted_chunk=3)
    params32 = random_params(cfg32, seed=13)
    batch = make_batch(cfg32, B=2, T=10, seed=14, seg_starts=((0,), (0, 6)))
    with jax.enable_x64(True):
        cfg = dataclasses.replace(cfg32, act_dtype="float64")
        params = tree_f64(params32)
        consts = M.make_consts(cfg)
        rest = _without_page(params)
        f = jax.jit(lambda pg: M.loss_fn({**rest, "page": pg}, consts, batch, cfg)[0])
        grad_f = jax.jit(jax.grad(f))
        assert f(params["page"]).dtype == jnp.float64
        d = check_directional_grads(f, grad_f, params["page"], seed=15)
        assert set(d) == {"pin", "pout", "key", "val"} and all(abs(x) > 0 for x in d.values())


# ------------------------------------------------------------------- remat, bfloat16
def test_remat_and_bfloat16_with_the_final_allocation():
    cfg = tiny_cfg(**FINAL)
    params = random_params(cfg, seed=16, scale=0.1)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=17)
    out = {remat: value_and_grad_fn(dataclasses.replace(cfg, remat=remat))(params, consts, batch)
           for remat in (False, True)}
    loss = float(out[False][0][0])
    np.testing.assert_allclose(float(out[True][0][0]), loss, rtol=1e-6)
    for a, b in zip(jax.tree.leaves(out[True][1]), jax.tree.leaves(out[False][1])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5)
    grads = out[True][1]
    new = list(grads["page"].values()) + [grads["core"]["attn"]["gate_w"], grads["core"]["attn"]["gate_b"]]
    assert all(float(jnp.max(jnp.abs(g))) > 0 for g in new)
    (loss_bf, _), grads_bf = value_and_grad_fn(dataclasses.replace(cfg, act_dtype="bfloat16", remat=True))(
        params, consts, batch)
    assert abs(float(loss_bf) - loss) < 0.02 * loss
    assert all(g.dtype == jnp.float32 and np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grads_bf))
    for name in ("pin", "pout", "key", "val"):
        a, b = np.asarray(grads_bf["page"][name]).ravel(), np.asarray(grads["page"][name]).ravel()
        assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.95, name


def _scan_operands(closed) -> list:
    """Avals of the operands and results of every ``scan`` in a closed jaxpr (nested included)."""
    found = []

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "scan":
                found.extend(v.aval for v in list(eqn.invars) + list(eqn.outvars))
            for p in eqn.params.values():
                for sub in p if isinstance(p, (tuple, list)) else (p,):
                    if hasattr(sub, "eqns"):
                        visit(sub)
                    elif hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                        visit(sub.jaxpr)

    visit(closed.jaxpr)
    return found


@pytest.mark.parametrize("remat", [False, True])
def test_rows_enter_the_loop_once_in_the_activation_dtype(remat):
    """The loop scan closes over one key and one value array per block, ``[B, T, rho, dk]`` in
    bfloat16; no scan of the forward or of the gradient carries a float32 copy or a per-loop
    stack of them (the cotangent accumulators are bfloat16 too)."""
    cfg = tiny_cfg(act_dtype="bfloat16", remat=remat, **FINAL)
    b, t = 2, 16
    params = init(cfg)
    consts = M.make_consts(cfg)
    batch = {"tokens": jnp.zeros((b, t + 1), jnp.int32), "w": jnp.ones((b, t), jnp.float32)}
    feats, _, _ = M.split_batch(batch)
    rho, dk = cfg.tok_neurons, cfg.tok_dim
    fwd = _scan_operands(jax.make_jaxpr(lambda p: M.forward(p, consts, feats, cfg)[0])(params))
    rows = [a for a in fwd if a.shape[-2:] == (rho, dk)]
    assert sorted((a.shape, str(a.dtype)) for a in rows) == [((b, t, rho, dk), "bfloat16")] * (2 * cfg.n_blocks)
    grad = jax.make_jaxpr(jax.grad(lambda p: M.loss_fn(p, consts, batch, cfg)[0]))(params)
    rows = [a for a in _scan_operands(grad) if len(a.shape) >= 2 and a.shape[-2:] == (rho, dk)]
    assert rows and all(a.dtype == jnp.bfloat16 and a.shape == (b, t, rho, dk) for a in rows), \
        sorted({(a.shape, str(a.dtype)) for a in rows})


def _grad_temp_bytes(cfg, b: int = 4, t: int = 64) -> int:
    """Compiler temp bytes of the loss gradient (compile only, no arrays)."""
    params = jax.eval_shape(functools.partial(M.init_params, cfg), jax.random.PRNGKey(0))
    consts = M.make_consts(cfg)
    batch = {"tokens": jax.ShapeDtypeStruct((b, t + 1), jnp.int32), "w": jax.ShapeDtypeStruct((b, t), jnp.float32)}
    fn = jax.jit(jax.grad(lambda p, x: M.loss_fn(p, consts, x, cfg)[0]))
    return fn.lower(params, batch).compile().memory_analysis().temp_size_in_bytes


def test_rows_are_not_copied_per_loop_under_remat():
    """With remat the token neurons' extra memory does not grow with the number of loops (one
    copy per loop of the keys and values, ``[B, T, n_blocks, 8, 64]`` float32 each, would add
    about 2 MB per loop here)."""
    extra = []
    for loops in (2, 6):
        base = tiny_cfg(n_loops=loops, remat=True, use_cand_in=False, use_cand_out=False, use_reg=False)
        paged = dataclasses.replace(base, tok_neurons=8, tok_dim=64)
        extra.append(_grad_temp_bytes(paged) - _grad_temp_bytes(base))
    assert extra[1] <= 1.05 * extra[0] + 65536, extra


# ----------------------------------------------------------------- generation interface
def test_block_interface_with_a_hook_reproduces_the_forward():
    """The pieces generation reuses: ``token_rows``, ``gain_rows`` / ``block_gain`` and
    ``block(..., attend_fn=hook, loop=r, tok=rows[i])`` in a Python loop over the loops give the
    forward's hidden state (output gate, token gains with two phases, token neurons)."""
    cfg = tiny_cfg(gain_ffn=50, gain_phases=2, **FINAL)
    params = random_params(cfg, seed=18)
    consts = M.make_consts(cfg)
    feats = _feats(cfg, 19)
    want = np.asarray(jitted("forward_hidden", cfg)(params, consts, feats))
    dt = layers.act_dtype(cfg)

    def manual(params, consts, feats):
        tokens = jnp.asarray(feats["tokens"])
        ctx = M.make_ctx(consts, feats, cfg)
        e = M.embed(params, consts, feats, cfg)
        ve_all = M.value_embeddings(params, tokens, cfg)
        gains = M.gain_rows(params, consts, tokens, cfg)
        tok = M.token_rows(params, consts, tokens, cfg)
        blocks = M.block_params(params, cfg)
        phases = M.loop_phases(cfg)
        lp = params["lanes"]
        lanes = jnp.broadcast_to(e[None], (cfg.n_lanes,) + e.shape)

        def hook(q, k, v):
            y, _ = attention.attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos, ctx.sink)
            return y, None

        for r in range(cfg.n_loops):
            u = controller.combine(lp["alpha"][r], lanes, dt)
            for i, pb in enumerate(blocks):
                u, _ = M.block(pb, u, ctx, ve_all[r] if i == 0 else None, cfg, attend_fn=hook,
                               gain=M.block_gain(gains, i, jnp.int32(phases[r])), loop=jnp.int32(r),
                               tok=tok[i])
            s = layers.rmsnorm(u, params["loop"]["g"][r], out_dtype=dt)
            lanes = controller.update(lp["mix"][r], lp["beta"][r], lanes, s, dt)
        return head.final_hidden(lanes, params["head"]["g"])

    got = np.asarray(jax.jit(manual)(params, consts, feats))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    ctx = M.make_ctx(consts, feats, cfg)
    pb = M.block_params(params, cfg)[0]
    u = jnp.zeros(feats["tokens"].shape + (cfg.d_model,), dt)
    with pytest.raises(ValueError, match="token-neuron rows"):
        M.block(pb, u, ctx, None, cfg, loop=0)


@pytest.mark.slow
def test_final_allocation_on_the_splash_path():
    """configs/v9.py uses the flash kernel: the token gain (and the optional output gate) act on
    its output as on the dense path's, next to the token neurons (Pallas interpret mode on CPU;
    the third case, the gate switched off, is the function of configs/v9.py)."""
    S = pytest.importorskip("bes.model.attention_splash")
    if not S.available():
        pytest.skip("splash attention not installed")
    base = tiny_cfg(n_loops=2, **FINAL)
    params = random_params(base, seed=20)
    batch = make_batch(base, B=2, T=40, seed=21, seg_starts=((0,), (0, 17)))
    loss = {}
    for impl, sw in (("xla", {}), ("splash", {}), ("splash", {"gate": 0.0})):
        cfg = dataclasses.replace(base, attn_impl=impl)
        f = jax.jit(functools.partial(M.loss_fn, cfg=cfg))
        loss[impl, bool(sw)] = float(f(params, M.make_consts(cfg, **sw), batch)[0])
    np.testing.assert_allclose(loss["splash", False], loss["xla", False], rtol=1e-5)
    assert abs(loss["splash", True] - loss["splash", False]) > 1e-4
