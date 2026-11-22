"""build(): group dispatch, learning rates, frozen leaves, checks, jit/grad, toy convergence."""
import dataclasses
import sys
import types

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bes.config import Cfg, ModelCfg, OptimCfg, load_config
from bes.optim import build, global_norm, group_lrs
from bes.optim import adam, muon_normuon, muonh
from bes.optim.build import ADAM_GROUPS, GROUPS, OptState, default_role_fn, lr_inputs, path_str

ROLES = {"core/attn/wq": "matrix", "core/mlp/w_up": "matrix", "core/norm/g1": "norm",
         "emb/tok": "embed", "head/out": "unembed", "lanes/beta": "scalar", "fz/table": "frozen",
         "cand/in_tok": "feature"}


def make_params(seed=0, d=16, v=12):
    rng = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(rng.uniform(-0.5, 0.5, s).astype(np.float32))
    return {"core": {"attn": {"wq": f(2, d, d)}, "mlp": {"w_up": f(2, d, 4 * d)},
                     "norm": {"g1": jnp.ones((2, d))}},
            "emb": {"tok": f(v, d)}, "head": {"out": f(v, d)}, "lanes": {"beta": jnp.ones((4, 2))},
            "fz": {"table": f(5, d)}, "cand": {"in_tok": f(v, 4)}}


def make_cfg(kind="muonh", d_model=768, **optim):
    return Cfg(model=ModelCfg(d_model=d_model, act_dtype="float32"),
               optim=OptimCfg(optimizer=kind, **optim))


def rand_like(tree, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda x: jnp.asarray(rng.standard_normal(x.shape).astype(np.float32) * scale),
                        tree)


def test_path_str():
    tree = {"a": {"b": [jnp.zeros(1), (jnp.zeros(2),)]}, "c/d": jnp.zeros(3)}
    paths = [path_str(k) for k, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]
    assert paths == ["a/b/0", "a/b/1/0", "c/d"]


def test_group_learning_rates():
    w = 1.0 / np.sqrt(768 / 640)
    lrs = group_lrs(make_cfg())
    assert lrs["matrix"] == pytest.approx(0.014)  # no width factor on the matrix group by default
    assert lrs["embed"] == pytest.approx(0.3 * w)
    assert lrs["unembed"] == pytest.approx(0.004 * w)
    assert lrs["norm"] == pytest.approx(0.005 * w)
    assert lrs["scalar"] == pytest.approx(0.01 * w)
    assert lrs["feature"] == pytest.approx(0.01 * w)             # 30x below the embedding rate
    assert lrs["gain"] == pytest.approx(0.01 * w) and lrs["page"] == pytest.approx(0.004 * w)
    assert set(lrs) == set(GROUPS) and set(ADAM_GROUPS) < set(lrs)
    assert group_lrs(make_cfg(width_scale_matrix=True))["matrix"] == pytest.approx(0.014 * w)
    assert group_lrs(make_cfg(d_model=128, lr_width_ref=512))["embed"] == pytest.approx(0.3 * 2.0)
    assert group_lrs(make_cfg(d_model=128, lr_width_ref=512, lr_feature=0.02))["feature"] == \
        pytest.approx(0.02 * 2.0)


def test_learning_rate_transfer_rule():
    """Reference rule: every group x (B / B_ref) ** 0.558, matrix group x (T / T_ref) ** -0.32."""
    base = group_lrs(make_cfg())
    assert group_lrs(make_cfg(horizon_tokens=0.0)) == base            # no horizon: no correction
    at_ref = group_lrs(make_cfg(horizon_tokens=9.83e8))
    assert at_ref == pytest.approx(base)                               # exactly 1 at the reference
    long = group_lrs(make_cfg(horizon_tokens=2.0e10))
    factor = (2.0e10 / 9.83e8) ** -0.32                                # independent arithmetic
    assert factor == pytest.approx(0.3814, abs=2e-4)
    assert long["matrix"] == pytest.approx(0.014 * factor)
    for g in ADAM_GROUPS:                                              # Adam groups: untouched
        assert long[g] == pytest.approx(base[g])
    small = group_lrs(make_cfg(tokens_per_step=16384))
    bf = (16384 / 262144) ** 0.558
    assert bf == pytest.approx(0.2128, abs=2e-4)
    for g in ("matrix",) + ADAM_GROUPS:                                # every group scales with the batch
        assert small[g] == pytest.approx(base[g] * bf)
    both = group_lrs(make_cfg(tokens_per_step=16384, horizon_tokens=4.0e6))
    assert both["matrix"] == pytest.approx(0.014 * bf * (4.0e6 / 9.83e8) ** -0.32)
    off = group_lrs(make_cfg(tokens_per_step=16384, horizon_tokens=4.0e6, batch_exponent=0.0,
                             data_exponent=0.0))
    assert off == pytest.approx(base)
    assert group_lrs(make_cfg())["frozen"] == 0.0


@pytest.mark.parametrize("kind", ["muonh", "muon"])
def test_runtime_group_rates_equal_the_built_ones_bitwise(kind):
    """``update(..., group_lr=lr_inputs(cfg))`` gives exactly the default update, eager and
    jitted; the rates act (a doubled Adam rate doubles Adam's first step); a missing group is
    refused."""
    cfg = make_cfg(kind, weight_decay=0.1, grad_clip=1.0, horizon_tokens=3e9)
    params = make_params()
    opt = build(cfg, params, ROLES.__getitem__)
    st = opt.init(params)
    grads = rand_like(params, 4)
    rates = lr_inputs(cfg)
    assert set(rates) == set(GROUPS) - {"frozen"}
    assert all(isinstance(v, np.float32) and v == np.float32(group_lrs(cfg)[g]) for g, v in rates.items())
    for fn in (opt.update, jax.jit(opt.update)):
        a = fn(grads, st, params, 0.7)
        b = fn(grads, st, params, 0.7, group_lr=rates)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
    doubled = dict(rates, embed=np.float32(2 * rates["embed"]))
    new, _ = opt.update(grads, st, params, 0.5, group_lr=doubled)
    np.testing.assert_allclose(np.abs(np.asarray(new["emb"]["tok"] - params["emb"]["tok"])),
                               2 * rates["embed"] * 0.5, rtol=1e-4)
    with pytest.raises(KeyError, match="matrix"):
        opt.update(grads, st, params, 0.7, group_lr={k: v for k, v in rates.items() if k != "matrix"})


def test_runtime_rates_make_the_program_independent_of_the_horizon():
    """With the rates as inputs, optimisers built for different horizons lower to one program;
    with the built-in rates the horizon is a constant of the program (a recompilation)."""
    params = make_params()
    spec = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    texts = {}
    for h in (0.0, 2e10):
        cfg = make_cfg(horizon_tokens=h)
        opt = build(cfg, params, ROLES.__getitem__)
        state = jax.eval_shape(opt.init, spec)
        rates = jax.tree.map(lambda v: jax.ShapeDtypeStruct((), jnp.float32), lr_inputs(cfg))
        runtime = jax.jit(lambda g, s, p, r, o=opt: o.update(g, s, p, 1.0, group_lr=r))
        builtin = jax.jit(lambda g, s, p, o=opt: o.update(g, s, p, 1.0))
        texts[h] = (runtime.lower(spec, state, spec, rates).as_text(),
                    builtin.lower(spec, state, spec).as_text())
    assert texts[0.0][0] == texts[2e10][0]
    assert texts[0.0][1] != texts[2e10][1]


@pytest.mark.parametrize("kind", ["muonh", "muon"])
def test_dispatch_matches_the_group_optimisers(kind):
    """Every leaf gets exactly the update of its group's optimiser with the group lr * lr_scale."""
    cfg = make_cfg(kind, weight_decay=0.1)
    params = make_params()
    opt = build(cfg, params, ROLES.__getitem__)
    assert opt.groups == ROLES
    state = opt.init(params)
    lrs, lr_scale = group_lrs(cfg), 0.7
    for t in range(1, 4):
        grads = rand_like(params, t)
        new_params, state_new = opt.update(grads, state, params, lr_scale)
        for path, group in ROLES.items():
            keys = path.split("/")
            get = lambda tree: tree[keys[0]][keys[1]] if len(keys) == 2 else tree[keys[0]][keys[1]][keys[2]]
            p, g, slot = get(params), get(grads), state.slots[path]
            lr = lrs[group] * lr_scale
            if group == "frozen":
                want, want_slot = p, {}
            elif group == "matrix" and kind == "muonh":
                want, want_slot = muonh.update(p, g, slot, lr, 0.95)
            elif group == "matrix":
                want, want_slot = muon_normuon.update(p, g, slot, lr, 0.95, 0.9, 0.1)
            else:
                want, want_slot = adam.update(p, g, slot, lr, t)
            np.testing.assert_allclose(np.asarray(get(new_params)), np.asarray(want), atol=1e-7)
            jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                                                 atol=1e-7),
                         state_new.slots[path], want_slot)
        assert int(state_new.count) == t
        params, state = new_params, state_new


def test_first_adam_step_size_per_group():
    """First Adam step moves every entry by lr_group * lr_scale (m_hat / sqrt(v_hat) = sign g)."""
    cfg = make_cfg()
    params = make_params()
    opt = build(cfg, params, ROLES.__getitem__)
    new, _ = opt.update(rand_like(params, 5), opt.init(params), params, 0.5)
    lrs = group_lrs(cfg)
    for path, group in ROLES.items():
        if group in ("embed", "unembed", "feature", "norm", "scalar"):
            leaf_new, leaf_old = new, params
            for k in path.split("/"):
                leaf_new, leaf_old = leaf_new[k], leaf_old[k]
            np.testing.assert_allclose(np.abs(np.asarray(leaf_new - leaf_old)), lrs[group] * 0.5,
                                       rtol=1e-4)


def test_frozen_leaves_never_move():
    params = make_params()
    for kind in ("muonh", "muon"):
        opt = build(make_cfg(kind, grad_clip=1.0), params, ROLES.__getitem__)
        p, st = params, opt.init(params)
        assert st.slots["fz/table"] == {}
        step = jax.jit(opt.update)
        for t in range(5):
            p, st = step(rand_like(params, t, scale=100.0), st, p, 1.0)
        np.testing.assert_array_equal(np.asarray(p["fz"]["table"]), np.asarray(params["fz"]["table"]))
        assert float(jnp.abs(p["emb"]["tok"] - params["emb"]["tok"]).max()) > 0


def test_unread_embedding_rows_do_not_move_in_the_first_step():
    params = make_params()
    ids = jnp.asarray([0, 3, 3, 7])

    def loss(p):
        h = p["emb"]["tok"][ids] @ p["core"]["attn"]["wq"][0]
        return jnp.sum(jnp.tanh(h) ** 2) + jnp.sum(p["head"]["out"] ** 2)

    for kind in ("muonh", "muon"):
        opt = build(make_cfg(kind), params, ROLES.__getitem__)
        grads = jax.grad(loss)(params)
        new, _ = jax.jit(opt.update)(grads, opt.init(params), params, 1.0)
        unread = [i for i in range(12) if i not in (0, 3, 7)]
        np.testing.assert_array_equal(np.asarray(new["emb"]["tok"])[unread],
                                      np.asarray(params["emb"]["tok"])[unread])
        assert np.all(np.asarray(new["emb"]["tok"])[[0, 3, 7]] != np.asarray(params["emb"]["tok"])[[0, 3, 7]])


def test_spherical_matrices_keep_their_norm_through_build():
    cfg = load_config("configs/smoke.py")  # d_model 128: matrices [2,128,128] and [2,128,512]
    rng = np.random.default_rng(0)
    params = {"core": {"attn": {"wq": jnp.asarray(rng.uniform(-0.15, 0.15, (2, 128, 128)), jnp.float32)},
                       "mlp": {"w_up": jnp.asarray(rng.uniform(-0.15, 0.15, (2, 128, 512)), jnp.float32),
                               "w_down": jnp.asarray(rng.uniform(-0.08, 0.08, (2, 512, 128)), jnp.float32)}}}
    opt = build(cfg, params, lambda path: "matrix")
    assert opt.kind == "muonh" and opt.ns_dtype == "float32"
    p, st = params, opt.init(params)
    step = jax.jit(opt.update)
    for t in range(3):
        p, st = step(rand_like(params, t), st, p, 1.0)
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(params)):
        np.testing.assert_allclose(np.linalg.norm(np.asarray(a), axis=(1, 2)),
                                   np.linalg.norm(np.asarray(b), axis=(1, 2)), rtol=2e-6)


def test_role_objects_and_lazy_default(monkeypatch):
    params = make_params()
    role = types.SimpleNamespace  # anything with a .group attribute
    opt = build(make_cfg(), params, lambda path: role(kind="dense", group=ROLES[path]))
    assert opt.groups == ROLES
    fake = types.ModuleType("bes.roles")
    fake.role_of = lambda path: role(kind="dense", group=ROLES[path])
    monkeypatch.setitem(sys.modules, "bes.roles", fake)
    assert build(make_cfg(), params).groups == ROLES
    assert default_role_fn("emb/tok").group == "embed"


def test_build_refuses_bad_configurations():
    params = make_params()
    with pytest.raises(ValueError, match="unknown optimiser group"):
        build(make_cfg(), params, lambda path: "matrixx")
    with pytest.raises(ValueError, match="not a \\(stack of\\) matrices"):
        build(make_cfg(), {"g": jnp.ones(16), "w": jnp.ones((4, 4))}, lambda path: "matrix")
    with pytest.raises(ValueError, match="optim.optimizer"):
        build(make_cfg("sgd"), params, ROLES.__getitem__)
    with pytest.raises(ValueError, match="matrix_layout"):
        build(make_cfg(matrix_layout="rows"), params, ROLES.__getitem__)
    with pytest.raises(ValueError, match="dtype"):
        build(make_cfg(ns_dtype="float16"), params, ROLES.__getitem__)


def test_zero_initialised_matrix_refused_by_muonh_only():
    params = make_params()
    params["core"]["attn"]["wq"] = params["core"]["attn"]["wq"].at[1].set(0.0)
    with pytest.raises(ValueError, match="core/attn/wq"):
        build(make_cfg("muonh"), params, ROLES.__getitem__)
    build(make_cfg("muonh"), params, ROLES.__getitem__, check_init=False)
    build(make_cfg("muon"), params, ROLES.__getitem__)


def test_ns_dtype_resolution():
    params = make_params()
    cfg16 = Cfg(model=ModelCfg(act_dtype="bfloat16"), optim=OptimCfg())
    assert build(cfg16, params, ROLES.__getitem__).ns_dtype == "bfloat16"
    cfg32 = dataclasses.replace(cfg16, optim=OptimCfg(ns_dtype="float32"))
    assert build(cfg32, params, ROLES.__getitem__).ns_dtype == "float32"


def test_structure_mismatch_is_refused():
    params = make_params()
    opt = build(make_cfg(), params, ROLES.__getitem__)
    state = opt.init(params)
    smaller = dict(params)
    del smaller["fz"]
    with pytest.raises(ValueError, match="tree structure"):
        opt.update(rand_like(smaller, 0), state, smaller, 1.0)


def test_gradient_clipping_scales_the_whole_tree_jointly():
    params = make_params()
    g1, g2 = rand_like(params, 1, scale=50.0), rand_like(params, 2, scale=0.01)
    g1["fz"]["table"] = g1["fz"]["table"] * 1e6  # frozen gradients do not count
    for kind in ("muonh", "muon"):
        clip = build(make_cfg(kind, grad_clip=1.0), params, ROLES.__getitem__)
        plain = build(make_cfg(kind), params, ROLES.__getitem__)
        live = {k: v for k, v in jax.tree_util.tree_flatten_with_path(g1)[0]}
        norm = float(np.sqrt(sum(float(jnp.sum(v ** 2)) for k, v in live.items()
                                 if path_str(k) != "fz/table")))
        factor = 1.0 / (norm + 1e-6)
        pa, sa = clip.update(g1, clip.init(params), params, 1.0)
        pa, sa = clip.update(g2, sa, pa, 1.0)  # small gradient: not clipped
        pb, sb = plain.update(jax.tree.map(lambda g: g * factor, g1), plain.init(params), params, 1.0)
        pb, sb = plain.update(g2, sb, pb, 1.0)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                                             rtol=2e-5, atol=1e-6), pa, pb)
    assert float(global_norm({"a": jnp.full(4, 2.0), "b": jnp.full(5, 0.0)})) == pytest.approx(4.0)


def test_momentum_and_decay_overrides():
    params = make_params()
    grads = rand_like(params, 3)
    opt = build(make_cfg("muonh"), params, ROLES.__getitem__)
    st = opt.init(params)
    new, _ = opt.update(grads, st, params, 1.0, momentum=0.5)
    want, _ = muonh.update(params["core"]["attn"]["wq"], grads["core"]["attn"]["wq"],
                           st.slots["core/attn/wq"], 0.014, 0.5)
    np.testing.assert_allclose(np.asarray(new["core"]["attn"]["wq"]), np.asarray(want), atol=1e-7)
    opt = build(make_cfg("muon", weight_decay=0.2), params, ROLES.__getitem__)
    st = opt.init(params)
    new, _ = jax.jit(opt.update)(grads, st, params, 1.0, momentum=0.85, wd_scale=0.5)
    want, _ = muon_normuon.update(params["core"]["mlp"]["w_up"], grads["core"]["mlp"]["w_up"],
                                  st.slots["core/mlp/w_up"], 0.014, 0.85, 0.9, 0.1)
    np.testing.assert_allclose(np.asarray(new["core"]["mlp"]["w_up"]), np.asarray(want), atol=1e-7)


def test_abstract_build_init_and_lowering():
    """Compile-only use: build and lower from shapes alone (no arrays)."""
    params = make_params()
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    for kind in ("muonh", "muon"):
        opt = build(make_cfg(kind), abstract, ROLES.__getitem__)
        state = jax.eval_shape(opt.init, abstract)
        assert state.slots["core/attn/wq"]["mom"].shape == (2, 16, 16)
        lowered = jax.jit(opt.update).lower(abstract, state, abstract, 1.0)
        assert lowered.compile() is not None


def test_state_is_a_plain_pytree():
    params = make_params()
    opt = build(make_cfg(), params, ROLES.__getitem__)
    st = opt.init(params)
    leaves, treedef = jax.tree_util.tree_flatten(st)
    restored = jax.tree_util.tree_unflatten(treedef, [np.asarray(x) for x in leaves])
    assert isinstance(restored, OptState)
    new_a, _ = opt.update(rand_like(params, 0), st, params, 1.0)
    new_b, _ = opt.update(rand_like(params, 0), restored, params, 1.0)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), new_a, new_b)


@pytest.mark.parametrize("kind", ["muonh", "muon"])
def test_jit_and_grad_through_the_whole_update(kind):
    params = make_params()
    opt = build(make_cfg(kind), params, ROLES.__getitem__)
    st = opt.init(params)
    st, grads = opt.update(rand_like(params, 0), st, params, 1.0)[1], rand_like(params, 1)
    eager = opt.update(grads, st, params, 0.3)
    jitted = jax.jit(opt.update)(grads, st, params, 0.3)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6),
                 eager, jitted)
    probe = rand_like(params, 2)

    def f(p, g, scale):
        new, _ = opt.update(g, st, p, scale)
        return sum(jnp.sum(a * b) for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(probe)))

    dp, dg, ds = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(params, grads, 0.3)
    for leaf in jax.tree.leaves((dp, dg, ds)):
        assert np.all(np.isfinite(np.asarray(leaf)))
    eps = 1e-3  # d/d(lr_scale) by central differences
    fd = (float(f(params, grads, 0.3 + eps)) - float(f(params, grads, 0.3 - eps))) / (2 * eps)
    assert float(ds) == pytest.approx(fd, rel=2e-2, abs=1e-2)


@pytest.mark.parametrize("kind", ["muonh", "muon", "adam"])
def test_toy_regression_converges(kind):
    """y = x W* + E*[t] + 0.3 fitted by gain * (x W) + E[t] + b + c (c frozen)."""
    rng = np.random.default_rng(0)
    n, d_in, d_out, vocab = 256, 16, 8, 12
    x = rng.standard_normal((n, d_in)).astype(np.float32)
    t = rng.integers(0, vocab, n)
    w_star = (rng.standard_normal((d_in, d_out)) * 3 / np.sqrt(d_in)).astype(np.float32)
    y = x @ w_star + (rng.standard_normal((vocab, d_out)) * 0.5).astype(np.float32)[t] + 0.3
    s = np.sqrt(3 / d_in)
    params = {"core": {"w": jnp.asarray(rng.uniform(-s, s, (d_in, d_out)).astype(np.float32)),
                       "gain": jnp.ones(())},
              "emb": {"tok": jnp.zeros((vocab, d_out))}, "head": {"b": jnp.zeros(d_out)},
              "fz": {"c": jnp.full(d_out, 0.1)}}
    roles = {"core/w": "embed" if kind == "adam" else "matrix", "core/gain": "scalar",
             "emb/tok": "embed", "head/b": "norm", "fz/c": "frozen"}
    cfg = Cfg(model=ModelCfg(d_model=512, act_dtype="float32"),
              optim=OptimCfg(optimizer="muon" if kind == "muon" else "muonh", lr_matrix=0.05,
                             lr_embed=0.1, lr_norm=0.05, lr_scalar=0.05))
    opt = build(cfg, params, roles.__getitem__)

    def loss(p):
        pred = p["core"]["gain"] * (x @ p["core"]["w"]) + p["emb"]["tok"][t] + p["head"]["b"] + p["fz"]["c"]
        return jnp.mean((pred - y) ** 2)

    steps = 300

    @jax.jit
    def train_step(p, st, i):
        g = jax.grad(loss)(p)
        return opt.update(g, st, p, 1.0 - i / steps)

    p, st = params, opt.init(params)
    for i in range(steps):
        p, st = train_step(p, st, jnp.float32(i))
    first, last = float(loss(params)), float(loss(p))
    assert last < 1e-5 * first, (first, last)
    np.testing.assert_array_equal(np.asarray(p["fz"]["c"]), np.full(d_out, 0.1, np.float32))
    if kind == "muonh":
        assert float(jnp.linalg.norm(p["core"]["w"])) == pytest.approx(
            float(jnp.linalg.norm(params["core"]["w"])), rel=1e-5)
