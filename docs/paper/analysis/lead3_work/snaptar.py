"""Read a checkpoint out of a snapshot directory (run.tar.000, run.tar.001, ... = one tar split in
parts) without extracting it: a seekable view over the concatenated parts, the tar member's byte
range, and np.load on that range. Only the leaves asked for are read into memory.

Used by the second-pass lead author because the disk has about 3.7 GB free and a v9 checkpoint
is 1.1 GB; nothing is written to disk.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np


class MultiPart(io.RawIOBase):
    """Read-only, seekable concatenation of files."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [os.fstat(f.fileno()).st_size for f in self.files]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = sum(self.sizes)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.total + off
        return self.pos

    def readinto(self, b):
        want = len(b)
        got = 0
        while got < want and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            n = min(want - got, self.sizes[i] - (self.pos - self.starts[i]))
            chunk = f.read(n)
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self):
        for f in self.files:
            f.close()
        super().close()


class Window(io.RawIOBase):
    """A byte range [start, start + size) of a seekable stream, itself seekable."""

    def __init__(self, base, start, size):
        self.base, self.start, self.size, self.pos = base, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        if n == 0:
            return 0
        self.base.seek(self.start + self.pos)
        data = self.base.read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_snapshot(snapdir):
    parts = sorted(Path(snapdir).glob("run.tar.[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(f"{snapdir}: no run.tar.NNN parts")
    mp = MultiPart(parts)
    buf = io.BufferedReader(mp, buffer_size=1 << 20)
    tf = tarfile.open(fileobj=buf, mode="r:")
    return mp, buf, tf


def find_checkpoint(tf):
    """(manifest dict, arrays member) of the checkpoint in the snapshot (the newest one)."""
    members = tf.getmembers()
    arrays = [m for m in members if m.name.endswith("/arrays.npz") and "/train/ckpt/" in m.name]
    if not arrays:
        raise FileNotFoundError("no train/ckpt/*/arrays.npz in the snapshot")
    arrays.sort(key=lambda m: m.name)
    am = arrays[-1]
    mm = tf.getmember(am.name.rsplit("/", 1)[0] + "/manifest.json")
    man = json.loads(tf.extractfile(mm).read())
    return man, am


def read_member_bytes(tf, name):
    return tf.extractfile(tf.getmember(name)).read()


def load_leaves(snapdir, prefix="state/params/", extra=("state/consts",)):
    """{leaf path: np.ndarray (stored dtype viewed as the record's dtype)} for the leaves whose
    path starts with ``prefix`` (and with any of ``extra``), the checkpoint manifest, and the
    run manifest of the snapshot."""
    mp, buf, tf = open_snapshot(snapdir)
    try:
        man, am = find_checkpoint(tf)
        run_man = json.loads(read_member_bytes(tf, "run/manifest.json"))
        records = {r["path"]: r for r in man["leaves"]}
        win = Window(buf, am.offset_data, am.size)
        out = {}
        with np.load(io.BufferedReader(win, buffer_size=1 << 20), allow_pickle=False) as z:
            for p, rec in records.items():
                if p.startswith(prefix) or any(p.startswith(e) for e in extra):
                    arr = np.array(z[p], copy=True)
                    if rec["stored"] != rec["dtype"]:
                        import jax.numpy as jnp

                        arr = arr.view(jnp.dtype(rec["dtype"]))
                    out[p] = arr
        return out, man, run_man
    finally:
        tf.close()
        mp.close()


def load_dir_leaves(ckpt_dir, prefix="state/params/", extra=("state/consts",)):
    """The same for an extracted checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)
    man = json.loads((ckpt_dir / "manifest.json").read_text())
    records = {r["path"]: r for r in man["leaves"]}
    out = {}
    with np.load(ckpt_dir / "arrays.npz", allow_pickle=False) as z:
        for p, rec in records.items():
            if p.startswith(prefix) or any(p.startswith(e) for e in extra):
                arr = np.array(z[p], copy=True)
                if rec["stored"] != rec["dtype"]:
                    import jax.numpy as jnp

                    arr = arr.view(jnp.dtype(rec["dtype"]))
                out[p] = arr
    return out, man
