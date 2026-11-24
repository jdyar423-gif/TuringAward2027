"""Training step: token-weighted accumulation over A micro-batches equals one big batch, donation,
non-finite skip, scalar inputs reach the optimiser, SPEC batch layout."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import train_standin as M
from bes.config import CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG
from bes.optim.build import lr_inputs
from bes.train.train import (BATCH_KEYS, TrainState, abstract, accumulate, batch_spec,
                             init_state, lower_step, make_train_step, program_fingerprint)


def make_batch(a=4, b=3, t=8, seed=0):
    rng = np.random.default_rng(seed)
    tokens = M.pattern_tokens(rng, (a, b, t + 1), noise=0.3)
    w = (rng.random((a, b, t)) < 0.7).astype(np.float32) * rng.uniform(0.5, 2.0, (a, b, t))
    w[1] = 0.0                     # a micro-batch with no weight at all
    w[2, 0] = 0.0                  # unequal weights per micro-batch
    return {"tokens": jnp.asarray(tokens), "w": jnp.asarray(w, jnp.float32)}


def big_batch_loss(params, consts, batch):
    """Independent reference: weighted mean cross-entropy over all rows of all micro-batches."""
    tok = batch["tokens"].reshape(-1, batch["tokens"].shape[-1])
    w = batch["w"].reshape(-1, batch["w"].shape[-1])
    z = M.logits(params, consts, tok[:, :-1])
    logp = jax.nn.log_softmax(z, axis=-1)
    nll = -jnp.take_along_axis(logp, tok[:, 1:, None], axis=-1)[..., 0]
    return jnp.sum(w * nll) / jnp.sum(w)


def setup(optimizer="muonh", seed=0, **optim):
    cfg = M.tiny_cfg(optimizer)
    if optim:
        cfg = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, **optim))
    state, opt = init_state(cfg, M.init_fn, M.role_fn, jax.random.key(seed), M.consts_fn(None))
    return cfg, state, opt


def test_accumulation_equals_the_big_batch_gradient():
    cfg, state, _ = setup()
    batch = make_batch()
    grads, loss, weight, micro = jax.jit(
        lambda p, c, b: accumulate(M.loss_fn, p, c, b, None))(state.params, state.consts, batch)
    ref_loss, ref_grads = jax.value_and_grad(big_batch_loss)(state.params, state.consts, batch)
    assert float(weight) == pytest.approx(float(np.sum(np.asarray(batch["w"], np.float64))), rel=1e-6)
    assert float(loss) == pytest.approx(float(ref_loss), rel=1e-5)
    for k in ref_grads:
        np.testing.assert_allclose(grads[k], ref_grads[k], rtol=1e-4, atol=1e-7)
    assert micro["ce_sum"].shape == (4,) and micro["ce_rows"].shape == (4, 3)
    assert float(micro["weight"][1]) == 0.0


@pytest.mark.parametrize("optimizer", ["muonh", "muon"])
def test_step_with_A_micro_batches_equals_one_big_micro_batch(optimizer):
    extra = {"weight_decay": 0.1} if optimizer == "muon" else {}
    cfg, state, opt = setup(optimizer, **extra)
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    s_a, s_1 = state, state
    for k in range(3):
        batch = make_batch(seed=k)
        big = {key: v.reshape((1, -1) + v.shape[2:]) for key, v in batch.items()}
        s_a, m_a = step(s_a, batch, 0.7, 0.9, 0.5)
        s_1, m_1 = step(s_1, big, 0.7, 0.9, 0.5)
        assert float(m_a["loss"]) == pytest.approx(float(m_1["loss"]), rel=1e-5)
        assert float(m_a["grad_norm"]) == pytest.approx(float(m_1["grad_norm"]), rel=1e-4)
    for k in s_a.params:
        np.testing.assert_allclose(s_a.params[k], s_1.params[k], rtol=2e-4, atol=2e-6)
    assert int(s_a.step) == 3 and int(s_a.opt_state.count) == 3


def test_metrics_and_donation():
    cfg, state, opt = setup()
    step = make_train_step(M.loss_fn, opt, cfg)
    batch = make_batch()
    ref_loss, ref_grads = jax.value_and_grad(big_batch_loss)(state.params, state.consts, batch)
    ref_norm = np.sqrt(sum(float(np.sum(np.asarray(g, np.float64) ** 2)) for g in ref_grads.values()))
    old = state
    new, m = step(state, batch, np.float32(1.0), np.float32(0.95), np.float32(1.0))
    assert set(m) == {"loss", "weight", "grad_norm", "skipped", "lr_scale", "micro"}
    assert all(isinstance(v, jax.Array) for v in jax.tree.leaves(m))  # read one step late
    assert float(m["loss"]) == pytest.approx(float(ref_loss), rel=1e-5)
    assert float(m["grad_norm"]) == pytest.approx(ref_norm, rel=1e-4)
    assert int(m["skipped"]) == 0 and int(new.step) == 1
    assert old.params["w1"].is_deleted() and old.opt_state.count.is_deleted()  # donated
    assert jax.tree.structure(new) == jax.tree.structure(abstract(new))
    for a, b in zip(jax.tree.leaves(abstract(new)), jax.tree.leaves(abstract(init_state(
            cfg, M.init_fn, M.role_fn, jax.random.key(0), M.consts_fn(None))[0]))):
        assert a.shape == b.shape and a.dtype == b.dtype  # same avals: no recompilation


def test_nonfinite_update_is_skipped():
    cfg, state, opt = setup()
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    batch = make_batch()
    batch["w"] = batch["w"].at[3, 0, 0].set(jnp.nan)
    new, m = step(state, batch, 1.0)
    assert int(m["skipped"]) == 1 and not np.isfinite(float(m["loss"]))
    assert int(new.step) == 1
    for a, b in zip(jax.tree.leaves((state.params, state.opt_state)),
                    jax.tree.leaves((new.params, new.opt_state))):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_lr_scale_momentum_and_weight_decay_reach_the_optimiser():
    cfg, state, opt = setup()
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    batch = make_batch()
    zero, _ = step(state, batch, 0.0)
    for k in ("emb", "g", "b"):  # Adam leaves do not move at lr 0
        assert np.array_equal(np.asarray(zero.params[k]), np.asarray(state.params[k]))
    one, _ = step(state, batch, 1.0)
    half, _ = step(state, batch, 0.5)
    for k in ("emb", "g", "b"):
        d1 = np.asarray(one.params[k]) - np.asarray(state.params[k])
        d5 = np.asarray(half.params[k]) - np.asarray(state.params[k])
        np.testing.assert_allclose(d5, 0.5 * d1, rtol=1e-4, atol=1e-7)
    default, _ = step(state, batch, 1.0)
    explicit, _ = step(state, batch, 1.0, np.float32(cfg.optim.momentum), np.float32(1.0))
    for a, b in zip(jax.tree.leaves(default), jax.tree.leaves(explicit)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    cfg_m, state_m, opt_m = setup("muon", weight_decay=0.5)
    step_m = make_train_step(M.loss_fn, opt_m, cfg_m, donate=False)
    s = state_m
    for _ in range(2):  # momentum only matters once the buffer is non-zero
        s, _ = step_m(s, batch, 1.0, np.float32(0.95), np.float32(1.0))
    a, _ = step_m(s, batch, 1.0, np.float32(0.5), np.float32(1.0))
    b, _ = step_m(s, batch, 1.0, np.float32(0.95), np.float32(1.0))
    c, _ = step_m(s, batch, 1.0, np.float32(0.95), np.float32(0.0))
    assert not np.allclose(a.params["w1"], b.params["w1"])
    assert not np.allclose(b.params["w1"], c.params["w1"])


def test_group_rates_as_inputs_equal_the_built_in_rates():
    """The step with ``group_lr = lr_inputs(cfg)`` is bit-identical to the step without it."""
    cfg, state, opt = setup(horizon_tokens=5e8, data_exponent=-0.32)
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    for k in range(2):
        batch = make_batch(seed=k)
        a, ma = step(state, batch, np.float32(0.7), np.float32(0.9), np.float32(1.0))
        b, mb = step(state, batch, np.float32(0.7), np.float32(0.9), np.float32(1.0),
                     lr_inputs(cfg))
        for x, y in zip(jax.tree.leaves((a, ma)), jax.tree.leaves((b, mb))):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        state = a


def test_program_fingerprint_ignores_the_rates_and_tells_programs_apart():
    cfg, state, opt = setup()
    fps = {}
    for name, extra in (("base", {}), ("horizon", dict(horizon_tokens=1e6, data_exponent=-0.32)),
                        ("muon", dict(optimizer="muon"))):
        c, s, o = setup(**extra)
        step = make_train_step(M.loss_fn, o, c)
        fps[name] = program_fingerprint(lower_step(step, s, (4, 3, 8), ("tokens", "w"),
                                                   lr_inputs(c)))
        fps[name + "/builtin"] = program_fingerprint(lower_step(step, s, (4, 3, 8), ("tokens", "w")))
    assert fps["base"] == fps["horizon"] != fps["muon"]
    assert fps["base/builtin"] != fps["horizon/builtin"]      # rates baked in: another program
    step = make_train_step(M.loss_fn, opt, cfg)
    assert program_fingerprint(lower_step(step, state, (2, 6, 8), ("tokens", "w"),
                                          lr_inputs(cfg))) != fps["base"]     # another shape


def test_same_shape_does_not_retrace():
    cfg, state, opt = setup()
    traces = []

    def counting_loss(*args):
        traces.append(1)
        return M.loss_fn(*args)

    step = make_train_step(counting_loss, opt, cfg, donate=False)
    for k in range(3):
        state, _ = step(state, make_batch(seed=k), np.float32(1.0), np.float32(0.95),
                        np.float32(1.0))
    assert len(traces) == 1
    step(state, make_batch(b=2, t=16), np.float32(1.0), np.float32(0.95), np.float32(1.0))
    assert len(traces) == 2


def test_batch_spec_follows_spec_section_3():
    spec = batch_spec((8, 32, 1024))
    assert set(spec) == set(BATCH_KEYS)
    expect = {"tokens": ((8, 32, 1025), np.int32), "w": ((8, 32, 1024), np.float32),
              "seg": ((8, 32, 1024), np.int32), "reg": ((8, 32, 1024, N_REG), np.int16),
              "clock": ((8, 32, 1024, N_CLOCKS), np.int32),
              "cand_tok": ((8, 32, 1024, CAND_SLOTS), np.int32),
              "cand_feat": ((8, 32, 1024, CAND_SLOTS, N_CAND_FEAT), np.int8)}
    for k, (shape, dtype) in expect.items():
        assert spec[k].shape == shape and spec[k].dtype == dtype
    assert set(batch_spec((8, 16, 2048), ("tokens", "w"))) == {"tokens", "w"}
    with pytest.raises(KeyError):
        batch_spec((1, 1, 1), ("bogus",))


def test_init_state():
    cfg, state, opt = setup()
    assert isinstance(state, TrainState)
    assert state.step.dtype == jnp.int32 and int(state.step) == 0
    assert opt.groups == M.ROLES
    assert set(state.consts) == {"scale"}
    state2, opt2 = init_state(cfg, M.init_fn, opt, jax.random.key(0))
    assert opt2 is opt and state2.consts == {}


# ------------------------------------------------------------------ micro-batches cut on the device
def test_cutting_micro_batches_into_pieces_changes_no_gradient():
    """accumulate(..., micro_split=k) scans over k times smaller pieces (faster on the TPU v6e:
    the passes over large tensors are memory-bound) and returns the same gradients, loss and
    weight; the metrics come back per micro-batch (sums added, means weighted by the tokens)."""
    cfg, state, _ = setup()
    batch = make_batch(a=4, b=4, t=8)                 # micro-batch 1 has no weight at all
    run = lambda k: jax.jit(lambda p, c, b: accumulate(M.loss_fn, p, c, b, None, k))(  # noqa: E731
        state.params, state.consts, batch)
    g1, l1, w1, m1 = run(1)
    for k in (2, 4):
        g, loss, w, m = run(k)
        assert float(w) == pytest.approx(float(w1), rel=1e-6)
        assert float(loss) == pytest.approx(float(l1), rel=1e-6)
        for name in g1:
            np.testing.assert_allclose(g[name], g1[name], rtol=2e-5, atol=1e-7)
        assert m["ce_sum"].shape == (4,) and m["weight"].shape == (4,)
        np.testing.assert_allclose(m["ce_sum"], m1["ce_sum"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m["weight"], m1["weight"], rtol=1e-6)
        assert float(m["weight"][1]) == 0.0 and np.all(np.isfinite(np.asarray(m["ce_sum"])))
    with pytest.raises(ValueError, match="cannot be cut"):
        run(3)


def test_a_step_on_pieces_equals_the_step_on_whole_micro_batches():
    cfg, state, opt = setup()
    whole = make_train_step(M.loss_fn, opt, cfg, donate=False)
    pieces = make_train_step(M.loss_fn, opt, cfg, donate=False, micro_split=2)
    s_a, s_b = state, state
    for k in range(3):
        batch = make_batch(a=4, b=4, t=8, seed=k)
        s_a, m_a = whole(s_a, batch, 0.7, 0.9, 0.5)
        s_b, m_b = pieces(s_b, batch, 0.7, 0.9, 0.5)
        assert float(m_a["loss"]) == pytest.approx(float(m_b["loss"]), rel=1e-6)
        assert float(m_a["grad_norm"]) == pytest.approx(float(m_b["grad_norm"]), rel=1e-5)
        assert m_b["micro"]["ce_sum"].shape == m_a["micro"]["ce_sum"].shape
    for name in s_a.params:
        np.testing.assert_allclose(s_a.params[name], s_b.params[name], rtol=2e-4, atol=2e-6)


def test_pieces_of_the_real_loss_function_regroup_to_its_metrics():
    """The model's own loss on a small configuration with every feature of the batch: the same
    loss, gradients and metrics whether a micro-batch is one piece, two or four."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import tpu_tune
    from bes.config import load_config
    from bes.model import model as BM
    from bes.train import supervisor as SV

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(str(root / "configs" / "smoke.py"), [])
    mc = cfg.model
    consts = BM.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = BM.init_params(mc, jax.random.key(0))
    one = tpu_tune.synth_batch(mc, 8, 32)
    rng = np.random.default_rng(3)
    one["w"] = (rng.random(one["w"].shape) < 0.8).astype(np.float32)
    one["w"][2:4] = 0.0                                  # a piece without weight when cut in four
    batch = {k: jnp.asarray(np.stack([v, v[::-1]])) for k, v in one.items()}       # A = 2
    run = lambda k: jax.jit(lambda p, b: accumulate(BM.loss_fn, p, consts, b, mc, k))(params, batch)  # noqa: E731
    g1, l1, w1, m1 = run(1)
    for k in (2, 4):
        g, loss, w, m = run(k)
        assert float(loss) == pytest.approx(float(l1), rel=2e-6) and float(w) == float(w1)
        worst = max(float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-12))
                    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g1)))
        assert worst < 2e-4, worst
        assert set(m) == set(m1)
        for name in ("ce_sum", "weight", "tokens", "ce_by_role", "weight_by_role"):
            np.testing.assert_allclose(m[name], m1[name], rtol=2e-5, atol=1e-4, err_msg=name)
        for name in ("loss", "tape_mass", "tape_gain", "loss_share_by_role"):
            np.testing.assert_allclose(m[name], m1[name], rtol=2e-4, atol=2e-5, err_msg=name)
        assert m["sink_mass"].shape == m1["sink_mass"].shape


def test_pieces_can_differ_by_sequence_length():
    from bes.train.train import parse_micro_split, split_for

    assert parse_micro_split("4") == 4 and parse_micro_split("16:4, 32:2") == {16: 4, 32: 2}
    assert split_for(4, 99) == 4 and split_for({16: 4}, 16) == 4 and split_for({16: 4}, 32) == 1
    with pytest.raises(ValueError):
        parse_micro_split("four")
    cfg, state, _ = setup()
    batch = make_batch(a=4, b=4, t=8)
    run = lambda k: jax.jit(lambda p, c, b: accumulate(M.loss_fn, p, c, b, None, k))(  # noqa: E731
        state.params, state.consts, batch)
    g1, l1, _, m1 = run(1)
    g, loss, _, m = run({8: 2, 16: 4})                # the batch has length 8: two pieces
    assert float(loss) == pytest.approx(float(l1), rel=1e-6) and m["ce_sum"].shape == m1["ce_sum"].shape
    for name in g1:
        np.testing.assert_allclose(g[name], g1[name], rtol=2e-5, atol=1e-7)
    g, loss, _, m = run({16: 4})                      # no entry for length 8: whole micro-batches
    for name in g1:
        np.testing.assert_array_equal(np.asarray(g[name]), np.asarray(g1[name]))
