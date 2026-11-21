"""Failure modes of the training loop found in review, with the stand-in model and a fake clock:
updates skipped for a non-finite gradient reach the guard; checkpoints on a full disk (room is
made from the oldest, ENOSPC is retried, the rollback distance is bounded); blocking and checked
init / final / rollback saves, background saves counted only once written, a loud final save;
the compiled step does not depend on the learning rates and is reused across trainers."""
import dataclasses
import errno
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import train_standin as M
from bes.optim.build import lr_inputs
from bes.train import checkpoint as C
from bes.train import schedule as S
from bes.train import train as TR
from bes.train.train import PARAMS_DIR, Trainer

T0 = 1000.0
KEYS = ("tokens", "w")


def trainer(cfg, out, clock, data, loss_fn=None, **kw):
    return Trainer(cfg, loss_fn=loss_fn or M.loss_fn, init_fn=M.init_fn, role_fn=M.role_fn,
                   consts_fn=M.consts_fn, data=data, out_dir=out, clock=clock,
                   batch_keys=KEYS, **kw)


def leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def reference_params(cfg, tmp_path, steps):
    clock = S.FakeClock(T0)
    ref = trainer(cfg, tmp_path / f"ref{steps}", clock, M.Data(clock, 1.0), max_steps=steps)
    ref.run()
    return ref.state.params


def events(tr, *kinds):
    return [e for e in tr.events if e["event"] in kinds]


# ------------------------------------------------------- skipped updates reach the guard
def grad_trap_loss(params, consts, batch, model_cfg):
    """The stand-in loss; on micro-batches whose first weight is 2.0 its value is unchanged and
    its gradient is NaN (``sqrt'(0) * 0``): a backward-only fault, the update is skipped."""
    loss, m = M.loss_fn(params, consts, batch, model_cfg)
    flag = (batch["w"][0, 0] == 2.0).astype(jnp.float32)
    return loss + 0.0 * jnp.sqrt(0.0 * jnp.sum(params["b"]) + (1.0 - flag)), m


class GradTrapData(M.Data):
    def __init__(self, clock, dt, bad):
        super().__init__(clock, dt)
        self.bad = bad

    def __call__(self, index, stage):
        b = super().__call__(index, stage)
        if self.bad(index):
            b = dict(b, w=b["w"].copy())
            b["w"][0, 0] = 2.0              # a legitimate weight: only the gradient is affected
        return b


def test_an_isolated_skipped_update_is_logged_and_tolerated(tmp_path):
    cfg = M.tiny_cfg(ckpt_every_min=0.25)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path, clock, GradTrapData(clock, 1.0, lambda i: i == 40),
                 loss_fn=grad_trap_loss, max_steps=60)
    out = tr.run()
    skips = events(tr, "skipped_update")
    assert [e["step"] for e in skips] == [41] and skips[0]["action"] == "suspect"
    assert skips[0]["reason"] == "nonfinite_grad" and np.isfinite(skips[0]["loss"])
    assert out["reason"] == "max_steps" and out["skipped_updates"] == 1
    assert not events(tr, "rollback", "halt")
    setup = events(tr, "setup")[0]                       # compilation is reported
    assert set(setup["compile"]) == {"4x4x16", "4x2x32"}
    assert all(v["reused"] is False and v["seconds"] > 0 for v in setup["compile"].values())


def test_persistent_skipped_updates_roll_back_and_halt(tmp_path):
    """From data index 30 on every gradient is NaN while the loss stays finite: the guard rolls
    back three times and halts on the trusted state (before: nothing reacted and every update
    of the remaining run was dropped silently)."""
    cfg = M.tiny_cfg(ckpt_every_min=0.25, guard_skip_steps=5)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path / "run", clock, GradTrapData(clock, 1.0, lambda i: i >= 30),
                 loss_fn=grad_trap_loss)
    out = tr.run()
    acts = events(tr, "rollback", "halt")
    assert [e["event"] for e in acts] == ["rollback"] * 3 + ["halt"]
    assert all(e["reason"] == "nonfinite_grad" for e in acts)
    assert acts[0]["first_bad"] == 31 and acts[0]["ckpt_step"] == 15
    assert out["reason"] == "halt" and out["skipped_updates"] == 12 == len(events(tr, "skipped_update"))
    assert tr.step == 15 and out["checkpoint"] and C.latest(tmp_path / "run" / "ckpt").step == 15
    leaves_equal(tr.state.params, reference_params(cfg, tmp_path, 15))


def test_a_backward_fault_of_the_long_shape_alone_is_not_silent(tmp_path):
    """Only the T=32 program has a non-finite gradient (losses finite): the run halts at the
    shape switch instead of training nothing for its second half."""
    def late_fault(params, consts, batch, model_cfg):
        loss, m = M.loss_fn(params, consts, batch, model_cfg)
        if batch["tokens"].shape[-1] == 33:             # static: the long program only
            loss = loss + 0.0 * jnp.sqrt(0.0 * jnp.sum(params["b"]))
        return loss, m

    cfg = M.tiny_cfg(ckpt_every_min=0.5)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path, clock, M.Data(clock, 2.0), loss_fn=late_fault)
    out = tr.run()
    first_long = next(h[0] for h in tr.history if h[4] == 32)     # the switch at p = 0.5
    assert events(tr, "skipped_update")[0]["step"] == first_long
    assert out["reason"] == "halt" and len(events(tr, "rollback")) == 3
    assert all(np.isfinite(h[3]) for h in tr.history)


# ------------------------------------------------------------- checkpoints on a full disk
def test_enospc_prunes_the_oldest_and_retries(tmp_path, monkeypatch):
    """A disk that holds three checkpoints but not four: every later save frees the oldest and
    succeeds; the final checkpoint is the last step (before: every save after the third failed
    and the newest checkpoint stayed at step 30)."""
    real_write = C._write_npz

    def full_disk(path, arrays):
        if len(C.list_checkpoints(path.parent.parent)) >= 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(path, arrays)

    monkeypatch.setattr(C, "_write_npz", full_disk)
    cfg = M.tiny_cfg(ckpt_every_min=0.25, keep_ckpts=3)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path, clock, M.Data(clock, 1.0), max_steps=100)
    out = tr.run()
    assert not events(tr, "ckpt_error")
    saved = events(tr, "ckpt")
    assert [e["step"] for e in saved] == [0, 15, 30, 45, 60, 75, 90, 100]
    assert all(e.get("retried") and e.get("pruned") for e in saved[3:])
    assert out["final"] == C.latest(tmp_path / "ckpt").name and C.latest(tmp_path / "ckpt").step == 100


def test_free_space_check_prunes_before_writing_and_keeps_the_trusted(tmp_path, monkeypatch):
    """The file system reports room for about two checkpoints: room is made before each write
    by deleting the oldest one, never the trusted (newest existing) one, and no write fails."""
    root = tmp_path / "ckpt"

    def used():
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0

    one = {}

    def free(path):
        if not one:                           # size of one checkpoint, measured on the first
            return 1 << 40
        return int(2.5 * one["size"]) - used()

    monkeypatch.setattr(C, "free_bytes", free)
    cfg = M.tiny_cfg(ckpt_every_min=0.25, keep_ckpts=5)
    clock = S.FakeClock(T0)
    data = M.Data(clock, 1.0)
    tr = trainer(cfg, tmp_path, clock, data, max_steps=80)
    tr.setup()
    one["size"] = used()
    out = tr.run()
    saved = events(tr, "ckpt")
    assert not events(tr, "ckpt_error") and len(saved) >= 6
    pruned = [e for e in saved if e.get("pruned")]
    assert pruned and not any(e.get("retried") for e in saved)
    # every save kept its predecessor (the trusted checkpoint) and at most two fit
    names = [e["path"] for e in saved]
    for e in pruned:
        assert names[names.index(e["path"]) - 1] not in e["pruned"]
    assert len(C.list_checkpoints(root)) <= 2 and C.latest(root).step == out["step"] == 80


def test_a_far_rollback_keeps_the_current_state(tmp_path, monkeypatch):
    """Saves fail (EIO) from data index 40 to 120, so at the spike (index 100) the newest trusted
    checkpoint is step 30, 70 s of training old with 15 s checkpoint intervals (bound 45 s): the
    rollback is not done; training goes on from the current state with the learning-rate cut and
    an alarm (before: 71 steps of training were discarded). With no bound the rollback is done."""
    real_rename = C._rename
    flags = {"fail": False}

    def flaky(src, dst):
        if flags["fail"]:
            raise OSError(errno.EIO, "I/O error")
        return real_rename(src, dst)

    monkeypatch.setattr(C, "_rename", flaky)

    class Data(M.Data):
        def __call__(self, index, stage):
            flags["fail"] = 40 <= index < 120
            return super().__call__(index, stage)

    for bound, rolled in ((3.0, False), (0.0, True)):
        cfg = M.tiny_cfg(ckpt_every_min=0.25, guard_max_rollback_intervals=bound,
                         guard_skip_steps=10)
        clock = S.FakeClock(T0)
        tr = trainer(cfg, tmp_path / f"b{bound}", clock, Data(clock, 1.0, poison={100, 101, 102}),
                     max_steps=140)
        out = tr.run()
        rb, alarms = events(tr, "rollback"), events(tr, "alarm")
        assert out["guard"]["rollbacks"] == 1 and out["reason"] == "max_steps"
        if rolled:   # saves still fail: the restored state cannot be re-saved, which is loud
            assert rb and rb[0]["ckpt_step"] == 30
            assert [a["reason"] for a in alarms] == [
                "the restored state could not be re-saved: a crash now would replay the skipped data"]
            continue
        assert not rb and alarms[0]["kind"] == "rollback_not_done" and "behind" in alarms[0]["reason"]
        steps = [h[0] for h in tr.history]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)   # nothing discarded
        after = [h for h in tr.history if h[0] > alarms[0]["step"]]
        assert after and all(lr == pytest.approx(0.7 * S.lr_factor(p, cfg.optim), rel=1e-12)
                             for _, p, lr, _, _ in after)
        assert alarms[0]["cursor"] == alarms[0]["step"] + 10      # the guard's data skip
        assert out["final"] is not None and C.latest(tmp_path / f"b{bound}" / "ckpt").step == tr.step


# ------------------------------------------------------- blocking, checked, recorded saves
def test_a_rollback_with_nothing_to_restore_keeps_training(tmp_path):
    """Every checkpoint disappears before a NaN streak: the rollback has nothing to restore, so
    training goes on with the learning-rate cut and an alarm (before: the run halted)."""
    cfg = M.tiny_cfg(ckpt_every_min=10.0)        # no interval save in the window
    clock = S.FakeClock(T0)
    root = tmp_path / "ckpt"

    class Data(M.Data):
        def __call__(self, index, stage):
            if index == 50:
                for c in C.list_checkpoints(root):
                    shutil.rmtree(c.path)
            return super().__call__(index, stage)

    tr = trainer(cfg, tmp_path, clock, Data(clock, 1.0, nan={60, 61, 62}), max_steps=90)
    out = tr.run()
    alarm = events(tr, "alarm")
    assert not events(tr, "rollback", "halt") and alarm and "no checkpoint" in alarm[0]["reason"]
    assert out["reason"] == "max_steps" and out["guard"]["lr_mult"] == pytest.approx(0.7)
    assert out["final"] and C.latest(root).step == tr.step


def rename_failing(monkeypatch, when):
    real_rename = C._rename

    def rename(src, dst):
        if when(dst):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_rename(src, dst)

    monkeypatch.setattr(C, "_rename", rename)


def test_background_saves_count_only_once_written(tmp_path, monkeypatch):
    """async_ckpt: background writes started between data indices 100 and 160 fail; none of them
    is reported as saved, the final save is blocking and on disk (before: a failed background
    write was booked as saved and summary['final'] could name a missing checkpoint)."""
    state = {"fail": False}
    rename_failing(monkeypatch, lambda dst: state["fail"])

    class Data(M.Data):
        def __call__(self, index, stage):
            state["fail"] = 100 <= index < 160
            return super().__call__(index, stage)

    cfg = M.tiny_cfg(ckpt_every_min=0.25)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path, clock, Data(clock, 1.0), async_ckpt=True)
    out = tr.run()
    saved = {e["step"]: e["path"] for e in events(tr, "ckpt")}
    errors = [e["step"] for e in events(tr, "ckpt_error")]
    assert errors and not set(errors) & set(saved)
    for step, name in saved.items():
        if step == out["step"]:
            assert (tmp_path / "ckpt" / name).exists()
    assert out["final"] == C.latest(tmp_path / "ckpt").name and (tmp_path / "ckpt" / out["final"]).exists()
    assert out["checkpoint"] == str(tmp_path / "ckpt" / out["final"])


def test_the_initial_save_is_blocking_and_fatal(tmp_path, monkeypatch):
    """async_ckpt as the 24-hour run uses it: a failing initial save stops the set-up (before: it
    only started a thread, training began without a checkpoint and the first rollback halted)."""
    rename_failing(monkeypatch, lambda dst: True)
    cfg = M.tiny_cfg()
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path, clock, M.Data(clock, 1.0), async_ckpt=True)
    with pytest.raises(C.CheckpointError, match="init"):
        tr.setup()
    assert C.latest(tmp_path / "ckpt") is None


def test_a_failed_final_save_falls_back_to_the_parameters_then_fails_loudly(tmp_path, monkeypatch):
    """Writes into ckpt/ fail from data index 31 on: the final save (step 40) is tried twice, then
    the parameters alone go to ckpt_params/ (summary final_kind 'params'). When those fail too,
    run() raises (before: it returned normally with no final checkpoint)."""
    state = {"fail": False}
    rename_failing(monkeypatch, lambda dst: state["fail"] and PARAMS_DIR not in str(dst))

    class Data(M.Data):
        def __call__(self, index, stage):
            state["fail"] = index >= 31
            return super().__call__(index, stage)

    cfg = M.tiny_cfg(ckpt_every_min=0.25)
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path / "a", clock, Data(clock, 1.0), async_ckpt=True, max_steps=40)
    out = tr.run()
    assert out["final_kind"] == "params" and PARAMS_DIR in out["checkpoint"]
    alarms = [e["reason"] for e in events(tr, "alarm")]
    assert any("attempt 2 of 2" in a for a in alarms) and any("only the parameters" in a for a in alarms)
    tree, man = C.load(out["checkpoint"], None, device=False)
    assert man["meta"]["kind"] == "params" and man["step"] == tr.step == 40
    leaves_equal(tree["state"]["params"], tr.state.params)
    assert C.latest(tmp_path / "a" / "ckpt").step < 40         # the full final is not there
    # nothing at all can be written: the run does not return normally
    rename_failing(monkeypatch, lambda dst: state["fail"])
    state["fail"] = False
    clock = S.FakeClock(T0)
    tr = trainer(cfg, tmp_path / "b", clock, Data(clock, 1.0), max_steps=40)
    with pytest.raises(C.CheckpointError, match="could be written"):
        tr.run()


# ---------------------------------------------------------- compiled-step reuse
def horizon_cfg(horizon):
    """The stand-in configuration with the transfer rule's data factor on (horizon 0 = off)."""
    cfg = M.tiny_cfg()
    o = dataclasses.replace(cfg.optim, data_exponent=-0.32, horizon_tokens=float(horizon))
    return dataclasses.replace(cfg, optim=o)


def test_the_trainer_program_does_not_depend_on_the_learning_rates(tmp_path):
    """Two configurations that differ only in the horizon (the matrix rate differs 9x) compile to
    the same program: trainer B runs on trainer A's executables and ends bit-identical to
    trainer C, which compiled its own; a different optimiser is not reused."""
    cfg_a, cfg_b = horizon_cfg(0.0), horizon_cfg(1e6)
    clock = S.FakeClock(T0)
    a = trainer(cfg_a, tmp_path / "a", clock, M.Data(clock, 1.0), max_steps=12)
    a.setup()
    clock = S.FakeClock(T0)
    b = trainer(cfg_b, tmp_path / "b", clock, M.Data(clock, 1.0), max_steps=12,
                programs=a.programs)
    b.setup()
    assert all(c["reused"] for c in b.compile_info.values())
    ratio = float(b.group_lr["matrix"]) / float(a.group_lr["matrix"])
    assert ratio == pytest.approx((1e6 / 9.83e8) ** -0.32, rel=1e-5) and ratio > 9
    b.run()
    clock = S.FakeClock(T0)
    c = trainer(cfg_b, tmp_path / "c", clock, M.Data(clock, 1.0), max_steps=12)
    c.run()
    assert not any(v["reused"] for v in c.compile_info.values())
    leaves_equal(b.state, c.state)
    assert b.history == c.history
    a.run()
    assert not np.array_equal(np.asarray(a.state.params["w1"]), np.asarray(b.state.params["w1"]))
    other = dataclasses.replace(cfg_b, optim=dataclasses.replace(cfg_b.optim, optimizer="muon"))
    state, opt = TR.init_state(other, M.init_fn, M.role_fn, jax.random.key(0), M.consts_fn(None))
    step = TR.make_train_step(M.loss_fn, opt, other)
    for shape, (fp, _) in a.programs.items():          # another optimiser: another program
        assert TR.program_fingerprint(TR.lower_step(step, state, shape, KEYS,
                                                    lr_inputs(other))) != fp


def test_restart_bitwise_with_reused_programs(tmp_path):
    """10 steps, a restart that reuses the first trainer's executables, 10 more: bit-identical
    to 20 steps in one go."""
    cfg = M.tiny_cfg()
    clock_a = S.FakeClock(T0)
    a = trainer(cfg, tmp_path / "split", clock_a, M.Data(clock_a, 1.0), max_steps=10)
    a.run()
    clock_b = S.FakeClock(clock_a())
    b = trainer(cfg, tmp_path / "split", clock_b, M.Data(clock_b, 1.0), max_steps=20,
                programs=a.programs)
    b.run()
    assert all(v["reused"] for v in b.compile_info.values())
    clock_c = S.FakeClock(T0)
    c = trainer(cfg, tmp_path / "whole", clock_c, M.Data(clock_c, 1.0), max_steps=20)
    c.run()
    leaves_equal(b.state, c.state)
    assert b.history == c.history[10:]

