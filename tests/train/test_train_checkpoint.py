"""Checkpoints: bitwise round trip, key-path npz, atomicity under a crash, corruption fallback,
configuration hash, pruning, background writes, no pickles."""
import errno
import json
import os
import struct
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.train import checkpoint as C

ROOT = Path(__file__).resolve().parents[2]


class Pair(NamedTuple):
    first: object
    second: object


def sample_tree(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "state": Pair(first=jnp.asarray(rng.standard_normal((3, 4)), jnp.float32),
                      second={"bf": jnp.asarray(rng.standard_normal(5), jnp.bfloat16),
                              "i8": jnp.asarray(rng.integers(-100, 100, 7), jnp.int8),
                              "i16": jnp.asarray(rng.integers(-900, 900, (2, 3)), jnp.int16),
                              "u32": jnp.asarray(rng.integers(0, 2**31, 4), jnp.uint32),
                              "flag": jnp.asarray([True, False, True]),
                              "scalar": jnp.asarray(3, jnp.int32),
                              "empty": jnp.zeros((0, 3), jnp.float32)}),
        "slots": {"core/attn/wq": {"mom": jnp.asarray(rng.standard_normal((2, 2)), jnp.float32)},
                  "odd%name": [jnp.ones(2, jnp.float32), jnp.full((1,), -0.0, jnp.float32)]},
        "rng": jax.random.key(seed),
        "raw_key": jax.random.PRNGKey(seed),
    }


def bits(x):
    if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        return np.asarray(jax.random.key_data(x)), str(jax.random.key_impl(x))
    a = np.asarray(x)
    return np.ascontiguousarray(a).reshape(-1).view(np.uint8), a.dtype.name


def assert_bitwise_equal(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        assert x.shape == y.shape and x.dtype == y.dtype
        (bx, nx), (by, ny) = bits(x), bits(y)
        assert nx == ny and np.array_equal(bx, by)


def test_round_trip_is_bitwise(tmp_path):
    tree = sample_tree()
    info = C.save(tmp_path, tree, step=7, config_hash="abc", meta={"cursor": 12, "x": [1.5, None]})
    assert info.name == "ckpt-000001-step-000000007" and info.step == 7 and info.seq == 1
    back, man = C.load(info, tree, config_hash="abc")
    assert_bitwise_equal(tree, back)
    assert isinstance(back["state"], Pair)
    assert man["meta"] == {"cursor": 12, "x": [1.5, None]} and man["step"] == 7
    assert isinstance(back["state"].first, jax.Array)
    host, _ = C.load(info, tree, device=False)
    assert isinstance(host["state"].first, np.ndarray)
    assert jax.dtypes.issubdtype(back["rng"].dtype, jax.dtypes.prng_key)
    # the restored key generates the same numbers
    assert np.array_equal(jax.random.normal(back["rng"], (3,)), jax.random.normal(tree["rng"], (3,)))


def test_float64_round_trip_on_host(tmp_path):
    tree = {"x": np.linspace(0, 1, 7), "y": np.asarray(2.5)}
    info = C.save(tmp_path, tree, step=0, config_hash="h")
    back, _ = C.load(info, tree, device=False)
    assert back["x"].dtype == np.float64 and np.array_equal(back["x"], tree["x"])


def test_npz_members_are_flattened_key_paths(tmp_path):
    info = C.save(tmp_path, sample_tree(), step=1, config_hash="h")
    with zipfile.ZipFile(info.path / C.ARRAYS) as zf:
        names = set(zf.namelist())
    assert "state/first.npy" in names and "state/second/bf.npy" in names
    assert "slots/core%2Fattn%2Fwq/mom.npy" in names and "slots/odd%25name/1.npy" in names
    man = json.loads((info.path / C.MANIFEST).read_text())
    recs = {r["path"]: r for r in man["leaves"]}
    assert recs["state/second/bf"]["dtype"] == "bfloat16" and recs["state/second/bf"]["stored"] == "uint16"
    assert recs["rng"]["key_impl"] == "threefry2x32" and recs["rng"]["dtype"] == "uint32"
    assert man["config_hash"] == "h" and man["format"] == C.FORMAT
    nested, _ = C.load(info)  # without a template: nested dicts keyed by the unescaped names
    assert set(nested["slots"]) == {"core/attn/wq", "odd%name"}
    assert np.array_equal(nested["state"]["first"], sample_tree()["state"].first)


def test_template_mismatches_raise(tmp_path):
    tree = {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4, jnp.int32)}
    info = C.save(tmp_path, tree, step=0, config_hash="h")
    for bad in ({"a": jnp.zeros((3, 2)), "b": tree["b"]},
                {"a": tree["a"], "b": jnp.zeros(4, jnp.float32)},
                {"a": tree["a"]},
                {"a": tree["a"], "b": tree["b"], "c": jnp.zeros(1)}):
        with pytest.raises(C.StructureMismatchError):
            C.load(info, bad)
    back, _ = C.load(info, jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree))
    assert_bitwise_equal(tree, back)
    with pytest.raises(ValueError):  # two leaves with the same path
        C.save(tmp_path, {"n": SameNames(jnp.zeros(1), jnp.ones(1))}, step=1, config_hash="h")
    # escaping keeps a "/" inside a key distinct from nesting
    info = C.save(tmp_path, {"a/b": jnp.zeros(1), "a": {"b": jnp.ones(1)}}, step=1, config_hash="h")
    back, _ = C.load(info, {"a/b": jnp.zeros(1), "a": {"b": jnp.zeros(1)}})
    assert float(back["a/b"][0]) == 0.0 and float(back["a"]["b"][0]) == 1.0


@jax.tree_util.register_pytree_with_keys_class
class SameNames:
    """A pytree node whose two children report the same key name."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def tree_flatten_with_keys(self):
        key = jax.tree_util.GetAttrKey("x")
        return ((key, self.a), (key, self.b)), None

    def tree_flatten(self):
        return (self.a, self.b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def test_configuration_hash_is_checked(tmp_path):
    tree = {"a": jnp.arange(3.0)}
    C.save(tmp_path, tree, step=3, config_hash="old")
    with pytest.raises(C.ConfigMismatchError):
        C.restore(tmp_path, tree, config_hash="new")
    back, man, info = C.restore(tmp_path, tree, config_hash="new", force=True)
    assert man["config_hash"] == "old" and info.step == 3
    assert C.restore(tmp_path, tree, config_hash="old") is not None


def test_keeps_the_last_three_and_restores_the_newest(tmp_path):
    for step in range(5):
        C.save(tmp_path, {"a": jnp.full(2, float(step))}, step=10 * step, config_hash="h", keep=3)
    ckpts = C.list_checkpoints(tmp_path)
    assert [(c.seq, c.step) for c in ckpts] == [(3, 20), (4, 30), (5, 40)]
    tree, man, info = C.restore(tmp_path, {"a": jnp.zeros(2)}, config_hash="h")
    assert info.seq == 5 and float(tree["a"][0]) == 4.0
    _, _, older = C.restore(tmp_path, {"a": jnp.zeros(2)}, config_hash="h", max_step=35)
    assert older.step == 30
    assert C.delete_newer(tmp_path, 3) and [c.seq for c in C.list_checkpoints(tmp_path)] == [3]
    assert C.next_seq(tmp_path) == 4
    assert C.restore(tmp_path / "missing", {"a": jnp.zeros(2)}) is None
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_crash_between_write_and_rename_leaves_the_previous_checkpoint(tmp_path):
    """A real crash: the child process exits inside the final rename of its second save."""
    script = textwrap.dedent("""
        import os, sys
        import numpy as np
        from bes.train import checkpoint as C
        root = sys.argv[1]
        C.save(root, {"a": np.arange(6, dtype=np.float32), "b": np.ones((2, 2), np.int32)},
               step=1, config_hash="h")
        C._rename = lambda src, dst: os._exit(17)
        C.save(root, {"a": np.arange(6, dtype=np.float32) + 100, "b": np.zeros((2, 2), np.int32)},
               step=2, config_hash="h")
        os._exit(0)
    """)
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env, cwd=ROOT,
                         capture_output=True, timeout=120)
    assert res.returncode == 17, res.stderr.decode()[-2000:]
    tmps = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert len(tmps) == 1 and (tmps[0] / C.ARRAYS).exists() and (tmps[0] / C.MANIFEST).exists()
    assert [(c.seq, c.step) for c in C.list_checkpoints(tmp_path)] == [(1, 1)]
    tmpl = {"a": np.zeros(6, np.float32), "b": np.zeros((2, 2), np.int32)}
    tree, man, info = C.restore(tmp_path, tmpl, config_hash="h", device=False)
    assert info.step == 1 and np.array_equal(tree["a"], np.arange(6, dtype=np.float32))
    mgr = C.Checkpointer(tmp_path, "h")  # a new writer removes the leftover
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert mgr.save(tmpl, step=3).seq == 2


def test_failed_rename_in_process_cleans_up(tmp_path, monkeypatch):
    C.save(tmp_path, {"a": jnp.ones(3)}, step=1, config_hash="h")

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(C, "_rename", boom)
    with pytest.raises(OSError):
        C.save(tmp_path, {"a": jnp.zeros(3)}, step=2, config_hash="h")
    assert [c.step for c in C.list_checkpoints(tmp_path)] == [1]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]


def _flip_last_data_byte(npz: Path, member: str) -> None:
    with zipfile.ZipFile(npz) as zf:
        info = zf.getinfo(member)
    with open(npz, "r+b") as f:
        f.seek(info.header_offset)
        header = f.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        pos = info.header_offset + 30 + name_len + extra_len + info.compress_size - 1
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 0xFF]))


def test_corrupt_newest_checkpoint_falls_back_to_the_previous(tmp_path):
    tmpl = {"a": jnp.zeros(1000), "b": jnp.zeros(3, jnp.int32)}
    for step in (1, 2, 3):
        C.save(tmp_path, {"a": jnp.full(1000, float(step)), "b": jnp.full(3, step, jnp.int32)},
               step=step, config_hash="h")
    newest = C.list_checkpoints(tmp_path)[-1]
    _flip_last_data_byte(newest.path / C.ARRAYS, "a.npy")
    with pytest.raises(C.CorruptCheckpointError):
        C.load(newest, tmpl)
    with pytest.warns(UserWarning, match="corrupt"):
        tree, _, info = C.restore(tmp_path, tmpl, config_hash="h")
    assert info.step == 2 and float(tree["a"][0]) == 2.0
    (C.list_checkpoints(tmp_path)[1].path / C.MANIFEST).unlink()  # torn second checkpoint
    with pytest.warns(UserWarning):
        _, _, info = C.restore(tmp_path, tmpl, config_hash="h")
    assert info.step == 1
    with open(C.list_checkpoints(tmp_path)[0].path / C.ARRAYS, "r+b") as f:
        f.truncate(100)
    with pytest.warns(UserWarning), pytest.raises(C.CorruptCheckpointError):
        C.restore(tmp_path, tmpl, config_hash="h")


def test_background_write_snapshots_the_arrays(tmp_path):
    mgr = C.Checkpointer(tmp_path, "h", async_write=True)
    host = np.arange(10, dtype=np.float32)
    dev = jnp.arange(10, dtype=jnp.float32) * 2
    info = mgr.save({"host": host, "dev": dev}, step=5, meta={"k": 1})
    host[:] = -1.0  # mutate / donate right after save() returns
    consumed = jax.jit(lambda x: x + 1, donate_argnums=0)(dev)
    jax.block_until_ready(consumed)
    mgr.wait()
    back, man = C.load(info, {"host": host, "dev": consumed}, config_hash="h", device=False)
    assert np.array_equal(back["host"], np.arange(10, dtype=np.float32))
    assert np.array_equal(back["dev"], np.arange(10, dtype=np.float32) * 2)
    assert mgr.latest().seq == info.seq


def test_background_write_errors_are_reraised(tmp_path, monkeypatch):
    mgr = C.Checkpointer(tmp_path, "h", async_write=True)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(C, "_rename", boom)
    mgr.save({"a": jnp.ones(2)}, step=1)
    with pytest.raises(C.CheckpointError, match="disk full"):
        mgr.wait()
    mgr.wait()  # the error is reported once
    monkeypatch.setattr(C, "_rename", os.rename)
    assert mgr.save({"a": jnp.ones(2)}, step=2, blocking=True).seq == 1


def _arrays(n=4096, value=0.0):
    return {"a": jnp.full(n, value, jnp.float32)}


def test_checkpointer_makes_room_from_the_oldest_and_keeps_the_protected(tmp_path, monkeypatch):
    """The free space covers less than one more checkpoint: the oldest unprotected ones are
    deleted before the write, never a protected one (default: the newest existing)."""
    mgr = C.Checkpointer(tmp_path, "h", keep=10)
    for step in range(4):
        mgr.save(_arrays(value=step), step=step)
    arrays, records, _ = C.to_host(_arrays())
    need = C.checkpoint_bytes(arrays, records)
    assert need > sum(a.nbytes for a in arrays.values())
    removed = []
    budget = {"free": need - 1}                   # one byte short; each deletion frees enough
    monkeypatch.setattr(C, "free_bytes", lambda path: budget["free"] + need * len(removed))
    real_remove = C._remove
    monkeypatch.setattr(C, "_remove", lambda info: (removed.append(info.step), real_remove(info)))
    mgr.save(_arrays(value=9), step=4, protect=[1])               # seq 1 = step 0 is protected
    assert removed == [1]                                          # the oldest unprotected one
    assert [c.step for c in C.list_checkpoints(tmp_path)] == [0, 2, 3, 4]
    assert mgr.last_job["pruned"] == ["ckpt-000002-step-000000001"] and not mgr.last_job["retried"]
    removed.clear()
    budget["free"] = -10 * need                   # far too little: everything unprotected goes
    mgr.save(_arrays(value=10), step=5)                            # default: the newest is kept
    assert removed == [0, 2, 3] and C.latest(tmp_path).step == 5
    assert [c.step for c in C.list_checkpoints(tmp_path)] == [4, 5]
    # with every other checkpoint protected nothing is deleted (the write may then fail)
    removed.clear()
    mgr.save(_arrays(value=11), step=6, protect=[c.seq for c in C.list_checkpoints(tmp_path)])
    assert removed == []


def test_checkpointer_retries_once_after_enospc(tmp_path, monkeypatch):
    mgr = C.Checkpointer(tmp_path, "h", keep=5)
    for step in range(3):
        mgr.save(_arrays(value=step), step=step)
    real_write = C._write_npz
    calls = []

    def flaky(path, arrays):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(path, arrays)

    monkeypatch.setattr(C, "_write_npz", flaky)
    info = mgr.save(_arrays(value=7), step=3)
    assert len(calls) == 2 and mgr.last_job["retried"] and mgr.last_job["pruned"] == [
        "ckpt-000001-step-000000000"]                              # one oldest checkpoint freed
    back, _ = C.load(info, _arrays(), device=False)
    assert float(back["a"][0]) == 7.0
    assert not [p for p in tmp_path.iterdir() if p.name.startswith((".tmp-", ".trash-"))]
    # a second ENOSPC is not retried again; another error is never retried
    monkeypatch.setattr(C, "_write_npz", lambda path, arrays: (_ for _ in ()).throw(
        OSError(errno.ENOSPC, "still full")))
    with pytest.raises(C.CheckpointError, match="still full"):
        mgr.save(_arrays(), step=4)
    monkeypatch.setattr(C, "_write_npz", lambda path, arrays: (_ for _ in ()).throw(
        OSError(errno.EIO, "broken")))
    before = [c.name for c in C.list_checkpoints(tmp_path)]
    with pytest.raises(C.CheckpointError, match="broken"):
        mgr.save(_arrays(), step=5)
    assert [c.name for c in C.list_checkpoints(tmp_path)] == before


def test_prune_never_deletes_protected(tmp_path):
    for step in range(5):
        C.save(tmp_path, _arrays(value=step), step=step, config_hash="h", keep=0)
    gone = C.prune(tmp_path, 2, protect=[1, 2])
    assert [c.seq for c in gone] == [3]
    assert [c.seq for c in C.list_checkpoints(tmp_path)] == [1, 2, 4, 5]


def test_no_pickles(tmp_path):
    with pytest.raises(TypeError):
        C.save(tmp_path, {"a": np.array([{"x": 1}], dtype=object)}, step=0, config_hash="h")
    info = C.save(tmp_path, {"a": jnp.ones(2)}, step=0, config_hash="h")
    np.savez(info.path / C.ARRAYS, a=np.array([object()], dtype=object))  # forged pickle member
    with pytest.raises(C.CorruptCheckpointError):
        C.load(info, {"a": jnp.ones(2)})


def test_optimizer_state_round_trip_gives_identical_updates(tmp_path):
    from bes.config import Cfg, ModelCfg
    from bes.optim import build

    params = {"w": jnp.asarray(np.random.default_rng(1).standard_normal((8, 4)), jnp.float32),
              "g": jnp.ones(4, jnp.float32)}
    roles = {"w": "matrix", "g": "norm"}
    opt = build(Cfg(model=ModelCfg(d_model=8)), params, lambda p: roles[p])
    state = opt.init(params)
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    params, state = opt.update(grads, state, params, 1.0)
    info = C.save(tmp_path, {"params": params, "opt": state}, step=1, config_hash="h")
    back, _ = C.load(info, {"params": params, "opt": state})
    assert_bitwise_equal({"params": params, "opt": state}, back)
    a = opt.update(grads, state, params, 0.5)
    b = opt.update(grads, back["opt"], back["params"], 0.5)
    assert_bitwise_equal(a, b)
