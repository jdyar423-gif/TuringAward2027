"""A few training steps of the real model (configs/smoke.py) on synthetic SPEC batches."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.config import BOS, CAND_FEATS, CAND_SLOTS, N_CLOCKS, REG_FIELDS, load_config
from bes.train import checkpoint as C
from bes.train import schedule as S
from bes.train.train import Trainer

M = pytest.importorskip("bes.model.model")
roles = pytest.importorskip("bes.roles")

ROOT = Path(__file__).resolve().parents[2]


def synthetic_batch(index, stage):
    """SPEC section 3 batch with valid ranges: rows start with BOS and repeat an 8-token motif."""
    a, b, t = stage.shape
    rng = np.random.default_rng(index)
    motif = rng.integers(0, 2000, (a, b, 8))
    tokens = np.tile(motif, (1, 1, (t + 1) // 8 + 1))[..., :t + 1].astype(np.int32)
    tokens[..., 0] = BOS
    w = np.ones((a, b, t), np.float32)
    w[:, :, :4] = 0.0  # some masked targets
    return {"tokens": tokens, "w": w, "seg": np.zeros((a, b, t), np.int32),
            "reg": np.stack([rng.integers(0, c, (a, b, t)) for _, c in REG_FIELDS], -1).astype(np.int16),
            "clock": rng.integers(0, 30, (a, b, t, N_CLOCKS)).astype(np.int32),
            "cand_tok": rng.integers(-1, 2000, (a, b, t, CAND_SLOTS)).astype(np.int32),
            "cand_feat": np.stack([rng.integers(0, c, (a, b, t, CAND_SLOTS)) for _, c in CAND_FEATS],
                                  -1).astype(np.int8)}


def test_real_model_trains_through_both_shapes(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    clock = S.FakeClock(1.0e6)
    window = (cfg.run.deadline_hours - cfg.run.post_reserve_hours) * 3600.0
    seen = []

    def data(index, stage):
        clock.advance(window / 4)  # four steps fill the window: p = 0, 0.25, 0.5, 0.75
        seen.append(stage.shape)
        return synthetic_batch(index, stage)

    tr = Trainer(cfg, loss_fn=M.loss_fn, init_fn=M.init_params, role_fn=roles.role_of,
                 consts_fn=M.make_consts, data=data, out_dir=tmp_path, clock=clock, max_steps=4)
    tr.setup()
    assert set(tr.compiled) == {(4, 8, 128), (4, 4, 256)} and seen == []
    params0 = jax.tree.map(np.asarray, tr.state.params)
    consts = M.make_consts(cfg.model)
    out = tr.run()
    assert out["reason"] == "max_steps" and tr.step == 4
    assert seen == [(4, 8, 128), (4, 8, 128), (4, 4, 256), (4, 4, 256)]
    losses = [h[3] for h in tr.history]
    assert all(np.isfinite(losses)) and [h[4] for h in tr.history] == [128, 128, 256, 256]
    # the step's loss is the token-weighted mean of the model's own loss over the micro-batches
    first = synthetic_batch(0, S.stage_at(0.0, cfg))
    loss_one = jax.jit(lambda p, mb: M.loss_fn(p, consts, mb, cfg.model))
    per, wts = [], []
    for i in range(first["tokens"].shape[0]):
        mb = {k: jnp.asarray(v[i]) for k, v in first.items()}
        per.append(float(loss_one(params0, mb)[0]))
        wts.append(float(first["w"][i].sum()))
    assert losses[0] == pytest.approx(np.average(per, weights=wts), rel=1e-5)
    assert losses[0] == pytest.approx(np.log(cfg.model.vocab_size), abs=0.1)
    now = dict(roles.leaves_with_paths(tr.state.params))
    before = dict(roles.leaves_with_paths(params0))
    moved = {p for p in now if not np.array_equal(np.asarray(now[p]), before[p])}
    frozen = {p for p in now if roles.role_of(p).group == "frozen"}
    assert {"core/attn/wq", "core/mlp/w_up", "emb/tok", "head/out"} <= moved
    assert frozen and not (frozen & moved)  # the critic is not trained by the main optimiser
    back, man, info = C.restore(tmp_path / "ckpt", {"state": tr.state, "rng": tr.rng},
                                config_hash=tr.config_hash)
    assert info.step == 4
    for a, b in zip(jax.tree.leaves(back["state"]), jax.tree.leaves(tr.state)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
