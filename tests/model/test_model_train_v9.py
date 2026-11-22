"""A few training steps of a tiny model with every v9 path on (token gains with two phases and a
partial FFN gain, selective attention, the sorted table gradient), through the real training
step (``bes.train.train.make_train_step``, run-time group rates): finite losses, no skipped
update, non-zero gradients for every gain table, tables that move. The same for the final
allocation of configs/v9.py (token neurons and output gains), without and with the optional
attention output gate."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, tiny_cfg
from bes.config import Cfg, OptimCfg
from bes.model import model as M
from bes.optim.build import lr_inputs
from bes.roles import role_of
from bes.train.train import accumulate, init_state, make_train_step


def _step_batch(cfg, seed: int, a: int = 2) -> dict:
    """``A`` micro-batches stacked on a leading axis (SPEC section 3 layout)."""
    parts = [make_batch(cfg, B=2, T=16, seed=seed * 10 + i) for i in range(a)]
    return {k: jnp.asarray(np.stack([p[k] for p in parts])) for k in parts[0]}


def test_train_steps_with_gains_selection_and_sorted_tables():
    mcfg = tiny_cfg(gain_ffn=100, gain_o=True, gain_v=True, gain_phases=2, sel_attn=True,
                    table_grad="sorted", sorted_chunk=8, remat=True)
    cfg = Cfg(model=mcfg, optim=OptimCfg(tokens_per_step=64, micro_seqs=2, lr_gain=0.05))
    state, opt = init_state(cfg, M.init_params, role_of, jax.random.key(0), M.make_consts(mcfg))
    assert {lp.path for lp in opt.plan if lp.group == "gain"} == {"gain/ffn", "gain/o", "gain/v"}
    assert all(float(jnp.max(jnp.abs(t))) == 0.0 for t in state.params["gain"].values())
    batch = _step_batch(mcfg, 0)
    grads, loss, _, _ = jax.jit(lambda p, c, b: accumulate(M.loss_fn, p, c, b, mcfg))(
        state.params, state.consts, batch)
    assert np.isfinite(float(loss))
    assert all(float(jnp.max(jnp.abs(g))) > 0.0 for g in grads["gain"].values())   # at zero tables
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    group_lr = lr_inputs(cfg)
    losses = []
    for i in range(4):
        state, met = step(state, _step_batch(mcfg, i + 1), np.float32(1.0), None, None, group_lr)
        losses.append(float(met["loss"]))
        assert int(met["skipped"]) == 0
    assert all(np.isfinite(losses)), losses
    assert all(float(jnp.max(jnp.abs(t))) > 0.0 for t in state.params["gain"].values())
    grads, _, _, _ = jax.jit(lambda p, c, b: accumulate(M.loss_fn, p, c, b, mcfg))(
        state.params, state.consts, batch)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grads))
    assert all(float(jnp.max(jnp.abs(g))) > 0.0 for g in grads["gain"].values())


@pytest.mark.parametrize("gate", [False, True], ids=["v9", "v9+gate"])
def test_train_steps_with_the_final_allocation(gate):
    """configs/v9.py in tiny form (token neurons, output gains, sorted tables, remat), and the
    same with the optional output gate and a decay on both tables: at initialisation the token
    neurons' values (and the gate) get a gradient while the keys and projections get exactly
    none (the values are zero); after a few steps every new leaf has moved, the losses are
    finite and no update was skipped."""
    mcfg = tiny_cfg(gain_o=True, attn_gate=gate, tok_neurons=4, tok_dim=16, table_grad="sorted",
                    sorted_chunk=8, remat=True)
    decay = 0.1 if gate else 0.0
    cfg = Cfg(model=mcfg, optim=OptimCfg(tokens_per_step=64, micro_seqs=2, lr_page=0.02,
                                         gain_decay=decay, page_decay=decay))
    state, opt = init_state(cfg, M.init_params, role_of, jax.random.key(0), M.make_consts(mcfg))
    groups = opt.groups
    assert (groups["page/val"], groups["page/key"], groups["page/pin"], groups["page/pout"]) == (
        "page", "embed", "matrix", "matrix")
    gate_leaves = ("core/attn/gate_w", "core/attn/gate_b") if gate else ()
    assert all(groups[n] == "scalar" for n in gate_leaves)
    assert ("gate_w" in state.params["core"]["attn"]) == gate
    start = state.params
    batch = _step_batch(mcfg, 0)
    grads, loss, _, _ = jax.jit(lambda p, c, b: accumulate(M.loss_fn, p, c, b, mcfg))(
        state.params, state.consts, batch)
    assert np.isfinite(float(loss))
    assert float(jnp.max(jnp.abs(grads["page"]["val"]))) > 0.0
    assert all(float(jnp.max(jnp.abs(grads["page"][n]))) == 0.0 for n in ("key", "pin", "pout"))
    assert all(float(jnp.max(jnp.abs(grads["core"]["attn"][n.rsplit("/", 1)[1]]))) > 0.0
               for n in gate_leaves)
    step = make_train_step(M.loss_fn, opt, cfg, donate=False)
    group_lr = lr_inputs(cfg)
    for i in range(4):
        state, met = step(state, _step_batch(mcfg, i + 1), np.float32(1.0), None, None, group_lr)
        assert np.isfinite(float(met["loss"])) and int(met["skipped"]) == 0
    for path in ("page/val", "page/key", "page/pin", "page/pout", "gain/o") + gate_leaves:
        a, b = state.params, start
        for k in path.split("/"):
            a, b = a[k], b[k]
        assert float(jnp.max(jnp.abs(a - b))) > 0.0, path


def test_training_with_the_sorted_table_gradient_follows_the_other_modes():
    """The micro ablations train with the scatter-add table gradient ("take", the fast mode on
    CPU), configs/v9.py with "sorted": real training steps of the final allocation in tiny form
    (a chunk of 8 rows, so the token runs span several chunks), from the same state on the same
    batches, give the same losses and tables in the three modes up to float32 round-off."""
    out = {}
    for mode in ("take", "onehot", "sorted"):
        mcfg = tiny_cfg(gain_o=True, tok_neurons=4, tok_dim=16, table_grad=mode,
                        sorted_chunk=8, remat=True)
        cfg = Cfg(model=mcfg, optim=OptimCfg(tokens_per_step=64, micro_seqs=2, lr_page=0.02))
        state, opt = init_state(cfg, M.init_params, role_of, jax.random.key(0), M.make_consts(mcfg))
        step = make_train_step(M.loss_fn, opt, cfg, donate=False)
        group_lr = lr_inputs(cfg)
        losses = []
        for i in range(5):
            state, met = step(state, _step_batch(mcfg, i + 1), np.float32(1.0), None, None, group_lr)
            assert int(met["skipped"]) == 0
            losses.append(float(met["loss"]))
        out[mode] = (losses, state.params)
    ref_losses, ref = out["take"]
    assert float(jnp.max(jnp.abs(ref["page"]["val"]))) > 0.0           # the tables trained
    for mode in ("onehot", "sorted"):
        losses, params = out[mode]
        np.testing.assert_allclose(losses, ref_losses, rtol=1e-5, err_msg=mode)
        for name in ("emb/tok", "page/key", "page/val", "gain/o"):
            a, b = params, ref
            for k in name.split("/"):
                a, b = a[k], b[k]
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5,
                                       err_msg=f"{mode} {name}")
