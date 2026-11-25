"""The parameters of a run that continues an exported model (``RunCfg.init_from``).

``RunCfg.init_from`` names an export directory (``weights16.npz`` next to ``config.py``), a
directory that holds one (``<dir>/export``), or a directory with the tar parts of one
(``release/<name>/<stem>.tar.000 ...`` and ``parts.json``, as ``scripts/pack_dir`` wrote them).
Parts are unpacked once into ``<data_dir>/init/<name>`` (every part and the whole tar are checked
against ``parts.json`` first).

:func:`make_init_fn` returns a function with the signature of ``bes.model.model.init_params``:
the trainer builds its parameter tree with it, so a fresh run starts from the exported 16-bit
weights with a new optimiser state, and a resumed run restores its checkpoint over the same tree
as before. Every leaf of the run's model configuration must be in the export with the same
shape, and the export may hold no other leaf (``bes.export.build_tree``): a continuation keeps
the architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = "weights16.npz"
PARTS = "parts.json"


class InitFromError(RuntimeError):
    """The export a run should start from is missing, damaged or of another architecture."""


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def find_export(root: Path) -> Path | None:
    """The export directory at or under ``root`` (the one that holds ``weights16.npz``)."""
    if (root / WEIGHTS).is_file():
        return root
    if (root / "export" / WEIGHTS).is_file():
        return root / "export"
    hits = sorted(p.parent for p in root.glob(f"*/{WEIGHTS}")) + sorted(p.parent for p in root.glob(f"*/*/{WEIGHTS}"))
    return hits[0] if hits else None


def unpack_parts(parts_dir: Path, out_dir: Path, log: Callable[[str], None] | None = None) -> Path:
    """Unpack the tar parts of ``parts_dir`` (``parts.json``) under ``out_dir`` and return the
    export directory inside. Done once: a finished unpack leaves ``out_dir/.unpacked`` with the
    tar's SHA-256, and a second call with the same parts returns at once."""
    meta_path = parts_dir / PARTS
    if not meta_path.is_file():
        raise InitFromError(f"{parts_dir} holds neither {WEIGHTS} nor {PARTS}")
    meta = json.loads(meta_path.read_text())
    want = str(meta.get("sha256", ""))
    marker = out_dir / ".unpacked"
    if marker.is_file() and marker.read_text().strip() == want:
        found = find_export(out_dir)
        if found is not None:
            return found
    names = [str(p["name"]) for p in meta.get("parts", [])]
    if not names:
        raise InitFromError(f"{meta_path} lists no parts")
    whole = hashlib.sha256()
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tar_path = tmp / "_whole.tar"
    with tar_path.open("wb") as out:
        for rec in meta["parts"]:
            part = parts_dir / str(rec["name"])
            if not part.is_file():
                raise InitFromError(f"{part} is missing")
            if rec.get("sha256") and _sha256(part) != rec["sha256"]:
                raise InitFromError(f"{part}: SHA-256 differs from {PARTS}")
            with part.open("rb") as f:
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    whole.update(b)
                    out.write(b)
    if want and whole.hexdigest() != want:
        raise InitFromError(f"{parts_dir}: the joined parts do not have the SHA-256 of {PARTS}")
    with tarfile.open(tar_path) as tar:
        base = tmp.resolve()
        for m in tar.getmembers():                      # no path may leave the directory
            dest = (tmp / m.name).resolve()
            if not (m.isfile() or m.isdir()) or (dest != base and base not in dest.parents):
                raise InitFromError(f"{parts_dir}: unexpected tar member {m.name!r}")
        try:
            tar.extractall(tmp, filter="data")
        except TypeError:                               # a Python without extraction filters
            tar.extractall(tmp)
    tar_path.unlink()
    (tmp / ".unpacked").write_text(want + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    found = find_export(out_dir)
    if found is None:
        raise InitFromError(f"{parts_dir}: the parts hold no {WEIGHTS}")
    if log is not None:
        log(f"init_from: unpacked {len(names)} parts of {parts_dir} into {out_dir}")
    return found


def export_dir(path: str | os.PathLike, data_dir: str | os.PathLike = "data_cache",
               log: Callable[[str], None] | None = None) -> Path:
    """The export directory ``RunCfg.init_from`` means (module docstring)."""
    root = _resolve(path)
    if not root.is_dir():
        raise InitFromError(f"init_from: {root} is not a directory")
    found = find_export(root)
    if found is not None:
        return found
    return unpack_parts(root, _resolve(data_dir) / "init" / root.name, log)


def load_params(path: str | os.PathLike, model_cfg, data_dir: str | os.PathLike = "data_cache",
                log: Callable[[str], None] | None = None) -> tuple[Any, dict]:
    """``(params, info)``: the float32 parameter tree of the export's 16-bit weights in the
    structure of ``init_params(model_cfg)``, and ``{"export", "sha256", "leaves", "numbers"}``."""
    import numpy as np

    from bes import export as E

    d = export_dir(path, data_dir, log)
    try:
        params = E.load_weights(d, model_cfg, "16")
    except E.ExportError as e:
        raise InitFromError(f"init_from: {d} does not fit the run's model configuration: {e}") from e
    import jax

    leaves = jax.tree_util.tree_leaves(params)
    info = {"export": str(d), "sha256": _sha256(d / WEIGHTS), "leaves": len(leaves),
            "numbers": int(sum(int(np.prod(np.shape(x))) for x in leaves))}
    if log is not None:
        log(f"init_from: {info['leaves']} leaves, {info['numbers']:,} numbers from {d} "
            f"(weights16 sha256 {info['sha256'][:16]})")
    return params, info


def make_init_fn(path: str | os.PathLike, data_dir: str | os.PathLike = "data_cache",
                 log: Callable[[str], None] | None = None) -> Callable[[Any, Any], Any]:
    """``init_fn(model_cfg, key) -> params`` that returns the export's weights (``key`` unused).
    Every call reads the file again and returns new arrays: the trainer donates the buffers of
    its state to the compiled step, so a tree handed out once must not be handed out twice.
    ``init_fn.info`` holds what the last call loaded."""

    def init_fn(model_cfg, key=None):
        params, init_fn.info = load_params(path, model_cfg, data_dir, log)
        return params

    init_fn.info = {}                                   # type: ignore[attr-defined]
    return init_fn


__all__ = ["InitFromError", "find_export", "unpack_parts", "export_dir", "load_params", "make_init_fn"]
