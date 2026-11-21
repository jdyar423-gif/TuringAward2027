"""Atomic checkpoints of pytrees of arrays plus JSON metadata (SPEC section 5).

Layout: one directory per checkpoint, ``<root>/ckpt-<seq:06d>-step-<step:09d>/`` with

* ``arrays.npz``    an uncompressed zip of ``.npy`` members, one per leaf, named by the leaf's
  key path (``state/params/core/attn/wq``; ``/`` and ``%`` inside a key are escaped as ``%2F``
  and ``%25``). No pickles: object arrays are refused and loading uses ``allow_pickle=False``.
* ``manifest.json`` format, sequence number, step, configuration hash, creation time, one record
  per leaf ``{path, dtype, shape, stored, key_impl}`` and the caller's JSON metadata (data
  cursor, ledger, guard state, ...).

Dtypes numpy cannot store natively (``bfloat16``, ``float8_*``) are stored as unsigned integers of
the same width and viewed back on load; typed PRNG keys are stored as their key data plus the
implementation name. The zip format keeps a CRC-32 per member, verified when a member is read, so
a corrupted array file is detected.

Writing protocol (crash safe): everything is written into a fresh hidden temporary directory,
each file is flushed and fsynced, the temporary directory is fsynced, then renamed to its final
name (atomic on POSIX) and the parent directory is fsynced. A crash at any point leaves either no
new checkpoint or a complete one; leftover temporary directories are ignored by readers and
removed by the next :class:`Checkpointer`. Sequence numbers only grow, so names never collide;
"latest" means the highest sequence number. Old checkpoints are pruned to the newest ``keep`` by
first renaming them to a hidden trash name.

Disk space (:class:`Checkpointer`): before a write the free space under the root is compared with
the size of the new checkpoint (arrays plus a small allowance); when it is short, the oldest
checkpoints are deleted first until it fits, never the protected ones (the caller's trusted
checkpoint; by default the newest existing one). A write that still fails with ``ENOSPC`` /
``EDQUOT`` prunes once more and is retried once. So a full disk costs old checkpoints, not every
later save.

A checkpoint written under another configuration hash is refused unless ``force=True``.
"""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import threading
import time
import uuid
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from bes.train.ledger import fsync_dir

FORMAT = 1
ARRAYS = "arrays.npz"
MANIFEST = "manifest.json"
_NAME_RE = re.compile(r"^ckpt-(\d+)-step-(\d+)$")
_TMP = ".tmp-"
_TRASH = ".trash-"
_NATIVE_KINDS = "biufc"
_ROOT_PATH = "__root__"

#: Rename used for the final, atomic step of a save (a hook for crash tests).
_rename = os.rename
#: Errors of a write that ran out of space.
_NO_SPACE = (errno.ENOSPC, errno.EDQUOT)


def free_bytes(path: str | os.PathLike) -> int:
    """Free bytes of the file system holding ``path`` (a hook for disk-full tests)."""
    return int(shutil.disk_usage(path).free)


def checkpoint_bytes(arrays: dict[str, np.ndarray], records: list[dict]) -> int:
    """Disk bytes a checkpoint of these host arrays needs, with a small allowance: the array
    bytes plus 1 KiB per member (zip and ``.npy`` headers, its manifest record), plus 2 % and
    1 MiB."""
    payload = sum(int(a.nbytes) for a in arrays.values())
    return payload + payload // 50 + 1024 * (len(arrays) + len(records)) + (1 << 20)


class CheckpointError(RuntimeError):
    """Base class of checkpoint errors."""


class ConfigMismatchError(CheckpointError):
    """The checkpoint was written under another configuration hash."""


class StructureMismatchError(CheckpointError):
    """The checkpoint's leaves do not match the template (paths, shapes or dtypes)."""


class CorruptCheckpointError(CheckpointError):
    """A file of the checkpoint is missing, truncated or fails its checksum."""


@dataclass(frozen=True, order=True)
class CkptInfo:
    """A complete checkpoint on disk (ordered by sequence number)."""

    seq: int
    step: int
    path: Path = field(compare=False)

    @property
    def name(self) -> str:
        return self.path.name


def ckpt_name(seq: int, step: int) -> str:
    return f"ckpt-{seq:06d}-step-{step:09d}"


# ------------------------------------------------------------------------ key paths
def _escape(s: str) -> str:
    return s.replace("%", "%25").replace("/", "%2F")


def _unescape(s: str) -> str:
    return s.replace("%2F", "/").replace("%25", "%")


def _entry_name(entry: Any) -> str:
    tu = jax.tree_util
    if isinstance(entry, tu.DictKey):
        name = str(entry.key)
    elif isinstance(entry, tu.SequenceKey):
        name = str(entry.idx)
    elif isinstance(entry, tu.GetAttrKey):
        name = str(entry.name)
    elif isinstance(entry, tu.FlattenedIndexKey):
        name = str(entry.key)
    else:
        name = str(entry)
    return _escape(name)


def leaf_path(keypath: Iterable[Any]) -> str:
    """``/``-joined, escaped name of a ``jax.tree_util`` key path (``__root__`` for a bare leaf)."""
    parts = [_entry_name(e) for e in keypath]
    return "/".join(parts) if parts else _ROOT_PATH


def flatten_with_paths(tree: Any) -> tuple[list[str], list[Any], Any]:
    """``(paths, leaves, treedef)``; raises ``ValueError`` if two leaves get the same path."""
    flat, treedef = jax.tree_util.tree_flatten_with_path(tree)
    paths = [leaf_path(kp) for kp, _ in flat]
    if len(set(paths)) != len(paths):
        dup = sorted({p for p in paths if paths.count(p) > 1})
        raise ValueError(f"leaf paths are not unique: {dup[:5]}")
    return paths, [leaf for _, leaf in flat], treedef


# --------------------------------------------------------------------------- dtypes
def _is_key(x: Any) -> bool:
    dt = getattr(x, "dtype", None)
    return dt is not None and jax.dtypes.issubdtype(dt, jax.dtypes.prng_key)


def _storable(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        raise TypeError("object arrays cannot be checkpointed (no pickles)")
    if arr.dtype.kind in _NATIVE_KINDS:
        return arr
    if arr.dtype.itemsize not in (1, 2, 4, 8):
        raise TypeError(f"cannot store dtype {arr.dtype}")
    return arr.view(np.dtype(f"u{arr.dtype.itemsize}"))


def to_host(tree: Any, *, copy: bool = False) -> tuple[dict[str, np.ndarray], list[dict], str]:
    """Device-get ``tree`` into ``({path: storable array}, leaf records, treedef string)``.

    copy: force private host copies (needed when the write happens after the device buffers may
    have been donated, e.g. in a background thread)."""
    paths, leaves, treedef = flatten_with_paths(tree)
    impls = [str(jax.random.key_impl(x)) if _is_key(x) else None for x in leaves]
    raw = [jax.random.key_data(x) if impl else x for x, impl in zip(leaves, impls)]
    host = jax.device_get(raw)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for path, x, impl in zip(paths, host, impls):
        arr = np.array(x, copy=True) if copy else np.asarray(x)
        stored = _storable(arr)
        arrays[path] = stored
        rec = {"path": path, "dtype": arr.dtype.name, "shape": list(arr.shape),
               "stored": stored.dtype.name}
        if impl:
            rec["key_impl"] = impl
        records.append(rec)
    return arrays, records, str(treedef)


def _from_stored(arr: np.ndarray, rec: dict, device: bool) -> Any:
    if rec["stored"] != rec["dtype"]:
        arr = arr.view(jnp.dtype(rec["dtype"]))
    if rec.get("key_impl"):
        return jax.random.wrap_key_data(jnp.asarray(arr), impl=rec["key_impl"])
    return jax.device_put(arr) if device else arr


# ---------------------------------------------------------------------------- listing
def list_checkpoints(root: str | os.PathLike) -> list[CkptInfo]:
    """Complete checkpoints under ``root`` sorted by sequence number (temporary and trash
    directories are ignored). Contents are validated only on load."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        m = _NAME_RE.match(p.name)
        if m and p.is_dir():
            out.append(CkptInfo(int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def latest(root: str | os.PathLike) -> CkptInfo | None:
    ckpts = list_checkpoints(root)
    return ckpts[-1] if ckpts else None


def cleanup_stale(root: str | os.PathLike) -> list[str]:
    """Remove temporary and trash directories left by crashed writers (single-writer roots)."""
    root = Path(root)
    removed = []
    if root.is_dir():
        for p in root.iterdir():
            if p.name.startswith((_TMP, _TRASH)):
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p.name)
    return removed


def _remove(info: CkptInfo) -> None:
    trash = info.path.parent / f"{_TRASH}{info.name}-{uuid.uuid4().hex[:8]}"
    os.rename(info.path, trash)
    fsync_dir(info.path.parent)
    shutil.rmtree(trash, ignore_errors=True)


def prune(root: str | os.PathLike, keep: int, protect: Iterable[int] = ()) -> list[CkptInfo]:
    """Delete all but the newest ``keep`` checkpoints, never those whose sequence number is in
    ``protect``; returns the deleted ones."""
    ckpts = list_checkpoints(root)
    keep_seqs = set(int(s) for s in protect)
    doomed = [c for c in (ckpts[:-keep] if keep > 0 else ckpts) if c.seq not in keep_seqs]
    for info in doomed:
        _remove(info)
    return doomed


def delete_newer(root: str | os.PathLike, seq: int) -> list[CkptInfo]:
    """Delete checkpoints with a sequence number above ``seq`` (used after a rollback)."""
    doomed = [c for c in list_checkpoints(root) if c.seq > seq]
    for info in doomed:
        _remove(info)
    return doomed


def next_seq(root: str | os.PathLike) -> int:
    ckpts = list_checkpoints(root)
    return ckpts[-1].seq + 1 if ckpts else 1


# ---------------------------------------------------------------------------- writing
def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as f:
        with zipfile.ZipFile(f, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, arr in arrays.items():
                with zf.open(name + ".npy", mode="w", force_zip64=True) as member:
                    # np.asarray(order="C") keeps 0-d arrays 0-d (ascontiguousarray would not)
                    np.lib.format.write_array(member, np.asarray(arr, order="C"),
                                              allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write(root: str | os.PathLike, arrays: dict[str, np.ndarray], records: list[dict],
          treedef: str, *, seq: int, step: int, config_hash: str,
          meta: dict | None = None) -> CkptInfo:
    """Write host arrays as checkpoint ``seq`` atomically (see the module docstring)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = ckpt_name(seq, step)
    final = root / name
    if final.exists():
        raise CheckpointError(f"checkpoint {final} already exists")
    manifest = {"format": FORMAT, "seq": int(seq), "step": int(step),
                "config_hash": str(config_hash), "created": time.time(), "treedef": treedef,
                "leaves": records, "meta": meta or {}}
    blob = json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False).encode()
    tmp = root / f"{_TMP}{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        _write_npz(tmp / ARRAYS, arrays)
        _write_bytes(tmp / MANIFEST, blob)
        fsync_dir(tmp)
        _rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    fsync_dir(root)
    return CkptInfo(int(seq), int(step), final)


def save(root: str | os.PathLike, tree: Any, *, step: int, config_hash: str,
         meta: dict | None = None, keep: int = 3) -> CkptInfo:
    """Synchronous save of ``tree`` (any pytree of arrays) plus JSON ``meta``; prunes to ``keep``."""
    arrays, records, treedef = to_host(tree)
    info = write(root, arrays, records, treedef, seq=next_seq(root), step=step,
                 config_hash=config_hash, meta=meta)
    if keep > 0:
        prune(root, keep)
    return info


# ---------------------------------------------------------------------------- reading
def read_manifest(path: str | os.PathLike) -> dict:
    try:
        man = json.loads((Path(path) / MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise CorruptCheckpointError(f"{path}: unreadable manifest ({e})") from e
    if not isinstance(man, dict) or man.get("format") != FORMAT:
        raise CorruptCheckpointError(f"{path}: unknown manifest format")
    return man


def _nest(paths: list[str], leaves: list[Any]) -> dict:
    out: dict = {}
    for path, leaf in zip(paths, leaves):
        parts = [_unescape(p) for p in path.split("/")]
        node = out
        for k in parts[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise StructureMismatchError(f"path {path!r} conflicts with a leaf")
        node[parts[-1]] = leaf
    return out


def _check_leaf(path: str, value: Any, tmpl: Any) -> None:
    shape = tuple(getattr(tmpl, "shape", np.shape(tmpl)))
    if tuple(value.shape) != shape:
        raise StructureMismatchError(f"{path}: shape {tuple(value.shape)} != template {shape}")
    t_dtype = getattr(tmpl, "dtype", None)
    if t_dtype is None:
        t_dtype = np.asarray(tmpl).dtype
    if _is_key(value) != jax.dtypes.issubdtype(t_dtype, jax.dtypes.prng_key):
        raise StructureMismatchError(f"{path}: key / non-key mismatch with the template")
    if value.dtype != t_dtype:
        raise StructureMismatchError(f"{path}: dtype {value.dtype} != template {t_dtype}")


def load(ckpt: CkptInfo | str | os.PathLike, template: Any = None, *,
         config_hash: str | None = None, force: bool = False,
         device: bool = True) -> tuple[Any, dict]:
    """Load one checkpoint; returns ``(tree, manifest)``.

    template: a pytree (arrays or ``jax.ShapeDtypeStruct``) giving the structure; leaves are matched
    by key path and must agree in shape and dtype; missing or extra leaves raise
    :class:`StructureMismatchError`. Without a template the tree is rebuilt as nested dicts.
    config_hash: if given, must equal the stored hash unless ``force``. device: return device
    arrays (``jax.device_put``) instead of numpy arrays (typed keys are always JAX arrays).
    """
    path = ckpt.path if isinstance(ckpt, CkptInfo) else Path(ckpt)
    man = read_manifest(path)
    if config_hash is not None and man.get("config_hash") != config_hash and not force:
        raise ConfigMismatchError(f"{path.name} was written under configuration "
                                  f"{man.get('config_hash')}, not {config_hash} (use force=True)")
    records = {r["path"]: r for r in man["leaves"]}
    if template is None:
        paths, tmpls, treedef = list(records), [None] * len(records), None
    else:
        paths, tmpls, treedef = flatten_with_paths(template)
        missing = [p for p in paths if p not in records]
        extra = [p for p in records if p not in set(paths)]
        if missing or extra:
            raise StructureMismatchError(f"{path.name}: missing leaves {missing[:5]}, "
                                         f"unexpected leaves {extra[:5]}")
    leaves = []
    try:
        with np.load(path / ARRAYS, allow_pickle=False) as z:
            for p, tmpl in zip(paths, tmpls):
                rec = records[p]
                arr = z[p]
                if list(arr.shape) != rec["shape"] or arr.dtype.name != rec["stored"]:
                    raise CorruptCheckpointError(f"{path.name}: {p} does not match its record")
                value = _from_stored(arr, rec, device)
                if tmpl is not None:
                    _check_leaf(p, value, tmpl)
                leaves.append(value)
    except CheckpointError:
        raise
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as e:
        raise CorruptCheckpointError(f"{path.name}: unreadable arrays ({type(e).__name__}: {e})") from e
    tree = _nest(paths, leaves) if treedef is None else jax.tree_util.tree_unflatten(treedef, leaves)
    return tree, man


def restore(root: str | os.PathLike, template: Any = None, *, config_hash: str | None = None,
            force: bool = False, max_step: int | None = None, max_seq: int | None = None,
            device: bool = True) -> tuple[Any, dict, CkptInfo] | None:
    """Load the newest loadable checkpoint (optionally with ``step <= max_step`` /
    ``seq <= max_seq``). A corrupt checkpoint is skipped with a warning in favour of the previous
    one; configuration or structure mismatches raise. Returns ``None`` if there is no candidate;
    raises :class:`CorruptCheckpointError` if every candidate is corrupt."""
    candidates = [c for c in list_checkpoints(root)
                  if (max_step is None or c.step <= max_step)
                  and (max_seq is None or c.seq <= max_seq)]
    errors = []
    for info in reversed(candidates):
        try:
            tree, man = load(info, template, config_hash=config_hash, force=force, device=device)
            return tree, man, info
        except CorruptCheckpointError as e:
            warnings.warn(f"skipping corrupt checkpoint: {e}")
            errors.append(str(e))
    if errors:
        raise CorruptCheckpointError("no loadable checkpoint: " + "; ".join(errors))
    return None


# ------------------------------------------------------------------------- manager
class Checkpointer:
    """Checkpoints of one run directory: sequence numbers, pruning, optional background writes.

    With ``async_write`` the arrays are copied to host memory synchronously (so the device buffers
    may be donated right after :meth:`save` returns) and written by one background thread; the
    next save, :meth:`wait`, :meth:`list` and :meth:`restore` wait for it. A failed background
    write is re-raised (as :class:`CheckpointError`) by the next of these calls; :meth:`idle` tells
    without blocking whether the write has finished.

    Every write first makes room (module docstring): the oldest checkpoints are deleted while the
    free space is below :func:`checkpoint_bytes`, except the protected ones (``save(protect=...)``,
    default the newest existing checkpoint); after an ``ENOSPC`` it prunes once more and retries
    once. ``last_job`` describes the latest finished write (``pruned`` names, ``retried``).
    """

    def __init__(self, root: str | os.PathLike, config_hash: str, *, keep: int = 3,
                 async_write: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        cleanup_stale(self.root)
        self.config_hash = config_hash
        self.keep = int(keep)
        self.async_write = bool(async_write)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.last_job: dict[str, Any] = {}

    def _make_room(self, need: int, protect: frozenset[int],
                   at_least_one: bool = False) -> list[str]:
        """Delete the oldest unprotected checkpoints until ``need`` bytes are free (and at least
        one when ``at_least_one``); returns their names."""
        removed: list[str] = []
        cleanup_stale(self.root)                 # single writer: leftovers of crashed writes
        for info in list_checkpoints(self.root):
            if info.seq in protect:
                continue
            if free_bytes(self.root) >= need and (removed or not at_least_one):
                break
            _remove(info)
            removed.append(info.name)
        return removed

    def _job(self, arrays, records, treedef, seq, step, meta, protect=None) -> None:
        try:
            if protect is None:                  # default: the newest existing checkpoint
                newest = list_checkpoints(self.root)[-1:]
                protect = [c.seq for c in newest]
            keep = frozenset(int(s) for s in protect)
            need = checkpoint_bytes(arrays, records)
            job: dict[str, Any] = {"need_bytes": need, "pruned": [], "retried": False}
            if free_bytes(self.root) < need:
                job["pruned"] += self._make_room(need, keep)
            kw = dict(seq=seq, step=step, config_hash=self.config_hash, meta=meta)
            try:
                write(self.root, arrays, records, treedef, **kw)
            except OSError as e:
                if e.errno not in _NO_SPACE:
                    raise
                job["pruned"] += self._make_room(need, keep, at_least_one=True)
                job["retried"] = True
                write(self.root, arrays, records, treedef, **kw)
            self.last_job = job
            if self.keep > 0:
                prune(self.root, self.keep, keep)
        except BaseException as e:  # re-raised in the caller's thread by wait()
            self._error = e

    def wait(self) -> None:
        """Block until a background write has finished; re-raise its error."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise CheckpointError(f"background checkpoint write failed: {err!r}") from err

    def idle(self) -> bool:
        """No background write is running (a finished one may still have to be collected by
        :meth:`wait`)."""
        return self._thread is None or not self._thread.is_alive()

    def save(self, tree: Any, *, step: int, meta: dict | None = None,
             blocking: bool | None = None, protect: Iterable[int] | None = None) -> CkptInfo:
        """Save ``tree`` with JSON ``meta`` as the newest checkpoint; returns its (future) info.

        protect: sequence numbers that making room and pruning never delete (default: the newest
        existing checkpoint). A blocking save raises :class:`CheckpointError` (or the ``OSError``)
        when the write failed; a background one reports it through :meth:`wait`."""
        self.wait()
        blocking = (not self.async_write) if blocking is None else blocking
        seq = next_seq(self.root)
        arrays, records, treedef = to_host(tree, copy=not blocking)
        protect = None if protect is None else tuple(int(s) for s in protect)
        if blocking:
            self._job(arrays, records, treedef, seq, step, meta, protect)
            self.wait()
        else:
            self._thread = threading.Thread(target=self._job, name="checkpoint-writer",
                                            args=(arrays, records, treedef, seq, step, meta,
                                                  protect),
                                            daemon=False)
            self._thread.start()
        return CkptInfo(seq, int(step), self.root / ckpt_name(seq, step))

    def list(self) -> list[CkptInfo]:
        self.wait()
        return list_checkpoints(self.root)

    def latest(self) -> CkptInfo | None:
        ckpts = self.list()
        return ckpts[-1] if ckpts else None

    def restore(self, template: Any = None, *, force: bool = False,
                max_step: int | None = None, max_seq: int | None = None,
                device: bool = True) -> tuple[Any, dict, CkptInfo] | None:
        self.wait()
        return restore(self.root, template, config_hash=self.config_hash, force=force,
                       max_step=max_step, max_seq=max_seq, device=device)

    def load(self, info: CkptInfo, template: Any = None, *, force: bool = False,
             device: bool = True) -> tuple[Any, dict]:
        self.wait()
        return load(info, template, config_hash=self.config_hash, force=force, device=device)

    def delete_newer(self, seq: int) -> list[CkptInfo]:
        self.wait()
        return delete_newer(self.root, seq)
