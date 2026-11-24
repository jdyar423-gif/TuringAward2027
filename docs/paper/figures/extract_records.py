"""Extract the small record files of a run snapshot (tar split in parts) into a directory.

Usage:  python -I extract_records.py SNAPSHOT_DIR OUT_DIR

A snapshot directory holds parts.json and run.tar.000, run.tar.001, ...  The tar is read as one
stream (the parts in order); only members on the allow-list below are written, each to a path
computed here (never the path stored in the tar), so the extraction cannot write outside OUT_DIR.
Weights, checkpoint arrays and the library are skipped.  parts.json is copied as is.
"""
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


class Parts:
    """A read-only file object over the parts, in order."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.f = None

    def read(self, n=-1):
        out = b""
        while n < 0 or len(out) < n:
            if self.f is None:
                if not self.paths:
                    break
                self.f = open(self.paths.pop(0), "rb")
            chunk = self.f.read(-1 if n < 0 else n - len(out))
            if not chunk:
                self.f.close()
                self.f = None
                continue
            out += chunk
        return out


def main(snap, out):
    names = sorted(n for n in os.listdir(snap) if re.match(r"^run\.tar\.\d{3}$", n))
    if not names:
        sys.exit(f"no run.tar.NNN parts in {snap}")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(os.path.join(snap, "parts.json"), os.path.join(out, "parts.json"))
    kept = []
    with tarfile.open(fileobj=Parts(os.path.join(snap, n) for n in names), mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not any(r.match(m.name) for r in KEEP_RE):
                continue
            parts = m.name.split("/")
            if any(p in ("", ".", "..") for p in parts):
                continue
            dest = os.path.join(out, *parts)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            src = tf.extractfile(m)
            with open(dest, "wb") as g:
                shutil.copyfileobj(src, g)
            kept.append((m.name, m.size))
    with open(os.path.join(out, "EXTRACTED.json"), "w") as g:
        json.dump({"snapshot": os.path.abspath(snap), "members": kept}, g, indent=1)
    for name, size in kept:
        print(f"{size:>12,d}  {name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
