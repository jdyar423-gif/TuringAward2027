"""Extract the small record files of a run snapshot (one tar split in parts) into a directory.

Usage:  python -I extract_records.py SNAPSHOT_DIR OUT_DIR

A snapshot directory holds parts.json and run.tar.000, run.tar.001, ...  The parts are opened as
one seekable stream, so only the tar headers and the members on the allow-list below are read
(a 1.1 GB snapshot is read in about a second; weights, checkpoint arrays and the library are
skipped without being read). Each kept member is written to a path computed here, never the path
stored in the tar, so the extraction cannot write outside OUT_DIR. parts.json is copied as is.

Before reading, the part files are checked against parts.json (names and byte counts): a snapshot
that is still being copied is refused with exit code 3 and nothing is written.
"""
import io
import json
import os
import re
import shutil
import sys
import tarfile

KEEP = [
    r"run/manifest\.json",
    r"run/stages\.json",
    r"run/stage_table\.json",
    r"run/train/ledger\.json",
    r"run/train/run\.json",
    r"run/train/train_log\.jsonl",
    r"run/benchmark/results\.jsonl",
    r"run/logs/[A-Za-z0-9_.-]+\.log",
    r"run/health/[A-Za-z0-9_.-]+\.json",
    r"run/final/reports/[A-Za-z0-9_.-]+\.json",
    r"run/final/(config|weights|export)\.json",
    r"run/final/MODEL_CARD\.md",
    r"run/train/ckpt/[A-Za-z0-9_.-]+/manifest\.json",
    r"run\.log",
]
KEEP_RE = [re.compile("^" + k + "$") for k in KEEP]


class Parts(io.RawIOBase):
    """Read-only, seekable concatenation of the part files."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [os.fstat(f.fileno()).st_size for f in self.files]
        self.starts = []
        s = 0
        for n in self.sizes:
            self.starts.append(s)
            s += n
        self.total = s
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def readinto(self, b):
        want, got = len(b), 0
        while got < want and self.pos < self.total:
            i = max(k for k, st in enumerate(self.starts) if st <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(want - got, self.sizes[i] - (self.pos - self.starts[i])))
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


def check_parts(snap):
    """The part files named in parts.json, in order, after checking that they are complete."""
    meta = json.load(open(os.path.join(snap, "parts.json")))
    names = sorted(n for n in os.listdir(snap) if re.match(r"^run\.tar\.\d{3}$", n))
    want = [p["name"] for p in meta["parts"]]
    if names != want:
        sys.stderr.write(f"{snap}: parts on disk {len(names)} != parts.json {len(want)} (copy in progress?)\n")
        sys.exit(3)
    for p in meta["parts"]:
        size = os.path.getsize(os.path.join(snap, p["name"]))
        if size != p["bytes"]:
            sys.stderr.write(f"{snap}: {p['name']} has {size} bytes, parts.json says {p['bytes']} (copy in progress?)\n")
            sys.exit(3)
    return [os.path.join(snap, n) for n in want], meta


def main(snap, out, quiet=False):
    paths, meta = check_parts(snap)
    os.makedirs(out, exist_ok=True)
    kept = []
    raw = Parts(paths)
    buf = io.BufferedReader(raw, buffer_size=1 << 16)
    with tarfile.open(fileobj=buf, mode="r:") as tf:
        for m in tf:                       # headers only; member data is skipped by seeking
            if not m.isfile() or not any(r.match(m.name) for r in KEEP_RE):
                continue
            parts = m.name.split("/")
            if any(p in ("", ".", "..") for p in parts):
                continue
            dest = os.path.join(out, *parts)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as g:
                shutil.copyfileobj(src, g)
            kept.append((m.name, m.size))
    raw.close()
    shutil.copyfile(os.path.join(snap, "parts.json"), os.path.join(out, "parts.json"))
    with open(os.path.join(out, "EXTRACTED.json"), "w") as g:
        json.dump({"snapshot": os.path.abspath(snap), "snapshot_created": meta.get("created"),
                   "snapshot_steps": meta.get("steps"), "snapshot_tokens": meta.get("tokens"),
                   "members": kept}, g, indent=1)
    if not quiet:
        for name, size in kept:
            print(f"{size:>12,d}  {name}")
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
