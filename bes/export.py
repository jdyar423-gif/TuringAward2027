"""Export of a finished run: 16-bit parameters, configuration, tokenizer, library, hashes and a
model card written from measured facts only.

    python scripts/export.py --run runs/final                    # -> runs/final/final/
    python scripts/export.py --verify runs/final/final           # SHA-256 of every file
    from bes.export import load_export                             # inference (float32 tree)
    python -m bes.eval.proxy run --model ours --generator bes.export:export_generator \\
        --generator-arg export=runs/final/final --split dev

``scripts/run_24h.py`` calls :func:`export_run` in its last stage (``export_run(out_dir,
checkpoint=..., config=..., overrides=..., horizon_tokens=..., library=..., final_dir=...,
reports=..., decode=..., cfg=...)``; ``out_dir`` is the run directory, ``final_dir`` the export
directory, default ``<out_dir>/final``).

Directory layout (:data:`FORMAT` 1)
-----------------------------------

====================  ==========================================================================
``weights16.npz``     every parameter leaf as a 16-bit number (``float16`` stored natively,
                      ``bfloat16`` stored as its ``uint16`` bit pattern), one ``.npy`` member per
                      leaf path (``core/attn/wq``); no pickles
``weights.json``      one record per leaf: path, shape, chosen dtype, relative L2 error of both
                      16-bit formats, largest absolute error
``weights32.npz``     the float32 master copy, kept **only** when the held-out check failed or
                      could not run (then the model card says so at the top)
``config.json``       the effective configuration (``bes.config.to_dict``), its hash, the
                      overrides, the horizon and the decoding choice of the run
``config.py``         ``load_config("<export>/config.py")`` rebuilds that configuration
``tokenizer/``        ``tokenizer.json`` and ``meta.json`` (``bes.tokenizer.load(<dir>)``)
``library/``          the built library's counted tables, the JSON files that configure them and
                      its plain-text attribution (:func:`library_files`; ``library_mode="copy"``),
                      or
``library_pointer.json``  its path, per-file SHA-256 and digest (``library_mode="pointer"``)
``reports/``          the run's stage ledger, manifest, training ledger, post-training and
                      evaluation reports that exist
``export.json``       everything measured at export time (check, accounting, read-set audit,
                      run facts, licences, errors)
``MODEL_CARD.md``     generated from ``export.json``
``SHA256SUMS``        ``sha256sum -c`` format, every other file of the directory
====================  ==========================================================================

Storage rule (``docs/ACCOUNTING.md``): one stored number has at most 16 bits, and the export holds
nothing numeric that the accounting does not count. The library ships only what
``Library.load`` reads and ``Library.accounting`` counts (each table at ``ceil(bytes / 2)``
numbers, so a ``uint32`` table counts two numbers per element) plus its attribution text; its
provenance arrays and build reports stay behind, and the export checks that the shipped tables
add up to the accounted total exactly (:func:`library_numbers`). Each leaf is stored
in the format that reproduces its float32 values with the smaller relative L2 error
``||x - q(x)|| / ||x||`` (:func:`choose_16bit`; a ``float16`` overflow counts as an infinite
error, so leaves outside the ``float16`` range go to ``bfloat16``; ties go to ``float16``). The
loader (:func:`load_weights`) restores a float32 tree for inference.

Held-out check (:func:`check_16bit`): the mean cross-entropy (nats per weighted token, the
training loss of ``bes.model.model.loss_fn``) of the 16-bit weights and of the float32 weights on
held-out rows, computed with float32 activations, dense attention and ``highest`` matmul precision
so that only the storage differs. It passes when ``|loss16 - loss32| < 0.002`` nats. Held-out rows
(:func:`heldout_batches`): ClimbMix ``shard_00997.parquet`` (beyond the run's download cap: shards
are fetched in listing order up to about 325), the smol-smoltalk ``test`` split (the registry reads
the train files only) and the project's 400 proxy prompts (written for this project, in no data
source). Each file is looked up in ``<data_dir>/samples/`` and downloaded there when missing (pinned
revision); a file that also appears among the run's training downloads is flagged.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import hashlib
import json
import math
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FORMAT = 1
#: largest allowed |held-out loss of the 16-bit weights - loss of the float32 weights| (nats)
TOLERANCE_NATS = 0.002
DTYPES = ("float16", "bfloat16")          # preference order when both errors are equal
WEIGHTS16 = "weights16.npz"
WEIGHTS32 = "weights32.npz"
WEIGHTS_JSON = "weights.json"
EXPORT_JSON = "export.json"
CARD = "MODEL_CARD.md"
SUMS = "SHA256SUMS"
POINTER = "library_pointer.json"
#: Plain-text files of a built library that ship with its counted tables: the attribution the
#: sources' licences require (never read by the model, so not counted).
LIBRARY_TEXT_FILES = ("ATTRIBUTION.md", "ATTRIBUTION.tsv", "ATTRIBUTION_QA.tsv")
#: The supervisor's record of the settings a library was built with
#: (``bes.train.supervisor.LIBRARY_SETTINGS``; read here, never shipped).
LIBRARY_SETTINGS = "build_settings.json"
#: version of the export code; part of the generator cache key
EXPORT_VERSION = "1"

Log = Callable[[str], None]


class ExportError(RuntimeError):
    """The export cannot produce a usable model (non-finite weights, a broken directory)."""


def _bf16():
    import ml_dtypes

    return np.dtype(ml_dtypes.bfloat16)


def _say(log: Log | None, msg: str) -> None:
    (log or (lambda m: print(m, flush=True)))(msg)


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    return repr(x)


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ===================================================================== 16-bit storage
@dataclass(frozen=True)
class LeafChoice:
    """The stored format of one parameter leaf and the measured errors of both formats.

    rel_err: ``{"float16": e, "bfloat16": e}`` with ``e = ||x - q(x)||_2 / ||x||_2`` (``inf`` on
    an overflow; 0 for an all-zero leaf); max_abs_err: ``max |x - q(x)|`` of the chosen format."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    stored: str
    rel_err: dict[str, float]
    max_abs_err: float

    @property
    def numbers(self) -> int:
        return int(math.prod(self.shape))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "shape": list(self.shape), "dtype": self.dtype,
                "stored": self.stored, "numbers": self.numbers,
                "rel_err": {k: (v if math.isfinite(v) else "inf") for k, v in self.rel_err.items()},
                "max_abs_err": self.max_abs_err}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LeafChoice":
        return cls(str(d["path"]), tuple(int(s) for s in d["shape"]), str(d["dtype"]),
                   str(d["stored"]), {k: float(v) for k, v in d["rel_err"].items()},
                   float(d["max_abs_err"]))


def choose_16bit(x: Any, path: str = "") -> tuple[np.ndarray, LeafChoice]:
    """``(stored array, choice)`` for one float32 leaf (module docstring). The stored array is
    ``float16``, or ``uint16`` holding ``bfloat16`` bit patterns. Raises :class:`ExportError` for
    non-finite values."""
    a = np.asarray(x)
    if a.dtype != np.float32:
        if not np.issubdtype(a.dtype, np.floating):
            raise ExportError(f"{path}: parameter leaf of dtype {a.dtype}, expected float32")
        a = a.astype(np.float32)
    if not np.all(np.isfinite(a)):
        raise ExportError(f"{path}: {int(np.size(a) - np.isfinite(a).sum())} non-finite values; "
                          "the weights are broken and are not exported")
    x64 = a.astype(np.float64)
    norm = float(np.sqrt(np.sum(x64 * x64)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cands = {"float16": a.astype(np.float16), "bfloat16": a.astype(_bf16())}
    errs: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    for name, q in cands.items():
        back = q.astype(np.float64)
        if not np.all(np.isfinite(back)):
            errs[name], max_abs[name] = math.inf, math.inf
            continue
        d = back - x64
        e = float(np.sqrt(np.sum(d * d)))
        errs[name] = e / norm if norm > 0 else (0.0 if e == 0.0 else math.inf)
        max_abs[name] = float(np.max(np.abs(d))) if d.size else 0.0
    best = min(DTYPES, key=lambda n: (errs[n], DTYPES.index(n)))
    if not math.isfinite(errs[best]):
        raise ExportError(f"{path}: no 16-bit format holds this leaf (values beyond the bfloat16 range)")
    q = cands[best]
    stored = q if best == "float16" else q.view(np.uint16)
    return stored, LeafChoice(path, tuple(int(s) for s in a.shape), best, stored.dtype.name, errs,
                              max_abs[best])


def flat_params(params: Any) -> tuple[list[str], list[np.ndarray]]:
    """Leaf paths (``bes.train.checkpoint.flatten_with_paths``) and host arrays of a tree."""
    import jax

    from bes.train.checkpoint import flatten_with_paths

    paths, leaves, _ = flatten_with_paths(params)
    return paths, [np.asarray(v) for v in jax.device_get(leaves)]


def quantize(params: Any) -> tuple[dict[str, np.ndarray], list[LeafChoice]]:
    """Every leaf of a float32 parameter tree in its 16-bit format: ``({path: stored}, choices)``."""
    arrays: dict[str, np.ndarray] = {}
    choices: list[LeafChoice] = []
    for path, leaf in zip(*flat_params(params)):
        stored, ch = choose_16bit(leaf, path)
        arrays[path] = stored
        choices.append(ch)
    return arrays, choices


def stored_to_float32(stored: np.ndarray, dtype: str) -> np.ndarray:
    """The float32 values of one stored 16-bit array (``dtype`` "float16" | "bfloat16" |
    "float32")."""
    a = np.asarray(stored)
    if dtype == "float16":
        if a.dtype != np.float16:
            raise ExportError(f"float16 leaf stored as {a.dtype}")
        return a.astype(np.float32)
    if dtype == "bfloat16":
        if a.dtype != np.uint16:
            raise ExportError(f"bfloat16 leaf stored as {a.dtype}")
        return a.view(_bf16()).astype(np.float32)
    if dtype == "float32":
        return a.astype(np.float32)
    raise ExportError(f"unknown stored dtype {dtype!r}")


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Uncompressed zip of ``.npy`` members (no pickles), written to a temporary name, fsynced
    and renamed."""
    tmp = path.with_name(f".tmp-{path.name}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as f:
            with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for name, arr in arrays.items():
                    with zf.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asarray(arr, order="C"),
                                                  allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {name: np.asarray(z[name]) for name in z.files}


def params_template(model_cfg) -> Any:
    """``jax.eval_shape`` tree of ``bes.model.model.init_params(model_cfg)``."""
    import jax

    from bes.model import model as M

    return jax.eval_shape(lambda: M.init_params(model_cfg, jax.random.PRNGKey(0)))


def build_tree(values: Mapping[str, np.ndarray], model_cfg) -> Any:
    """A float32 parameter tree (``jax.numpy`` arrays) from ``{path: float32 array}``, in the
    structure of ``init_params(model_cfg)``; paths and shapes must match exactly."""
    import jax
    import jax.numpy as jnp

    from bes.train.checkpoint import flatten_with_paths

    paths, tmpl, treedef = flatten_with_paths(params_template(model_cfg))
    missing = [p for p in paths if p not in values]
    extra = sorted(set(values) - set(paths))
    if missing or extra:
        raise ExportError(f"the stored leaves do not match the model configuration: missing "
                          f"{missing[:5]}, unexpected {extra[:5]}")
    leaves = []
    for p, t in zip(paths, tmpl):
        v = np.asarray(values[p], np.float32)
        if tuple(v.shape) != tuple(t.shape):
            raise ExportError(f"{p}: stored shape {tuple(v.shape)}, the configuration wants "
                              f"{tuple(t.shape)}")
        leaves.append(jnp.asarray(v))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def dequantize(arrays: Mapping[str, np.ndarray], choices: Sequence[LeafChoice], model_cfg) -> Any:
    """The float32 tree of quantised leaves (what the loader restores)."""
    by = {c.path: c for c in choices}
    return build_tree({p: stored_to_float32(a, by[p].dtype) for p, a in arrays.items()}, model_cfg)


def load_weights(export_dir: str | Path, model_cfg, which: str = "16",
                 expect_sha256: str | None = None) -> Any:
    """Float32 parameter tree of an export: ``which="16"`` (the shipped 16-bit weights) or
    ``"32"`` (the float32 copy, present only when the check failed)."""
    d = Path(export_dir)
    path = d / (WEIGHTS16 if which == "16" else WEIGHTS32)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist" + (
            " (the float32 copy is kept only when the 16-bit check failed)" if which == "32" else ""))
    if expect_sha256 and sha256_file(path) != expect_sha256:
        raise ExportError(f"{path}: SHA-256 differs from the export record (file changed)")
    arrays = read_npz(path)
    if which == "32":
        return build_tree({p: stored_to_float32(a, "float32") for p, a in arrays.items()}, model_cfg)
    meta = _read_json(d / WEIGHTS_JSON)
    if not meta:
        raise ExportError(f"{d / WEIGHTS_JSON} is missing or unreadable")
    choices = [LeafChoice.from_dict(r) for r in meta["leaves"]]
    for c in choices:
        a = arrays.get(c.path)
        if a is None or a.dtype.name != c.stored or tuple(a.shape) != c.shape:
            raise ExportError(f"{c.path}: stored array does not match its record in {WEIGHTS_JSON}")
    rows = meta.get("rows")
    if rows:
        # token rows were evicted (bes.evict): the tables hold the kept rows and a map says which;
        # the model of the configuration has rows of zeros for the others
        from bes import evict

        m = arrays.get(str(rows["map"]))
        if m is None or m.dtype != np.uint16 or m.shape != (int(rows["vocab"]),) or \
                int((m != evict.NO_ROW).sum()) != int(rows["kept"]):
            raise ExportError(f"{rows['map']}: the stored row map does not match its record in {WEIGHTS_JSON}")
        try:
            arrays = evict.expand(arrays, tuple(rows["tables"]), str(rows["map"]))
        except ValueError as e:
            raise ExportError(str(e)) from e
        choices = [dataclasses.replace(c, shape=tuple(int(x) for x in arrays[c.path].shape))
                   if c.path in rows["tables"] else c for c in choices]
    return dequantize(arrays, choices, model_cfg)


def dtype_summary(choices: Sequence[LeafChoice]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in DTYPES:
        sel = [c for c in choices if c.dtype == name]
        out[name] = {"leaves": len(sel), "numbers": sum(c.numbers for c in sel),
                     "max_rel_err": max((c.rel_err[name] for c in sel), default=0.0)}
    out["bits_per_number"] = 16
    out["numbers"] = sum(c.numbers for c in choices)
    return out


# ===================================================================== held-out rows
@dataclass(frozen=True)
class HeldoutFile:
    """One held-out parquet file of a Hugging Face dataset (module docstring)."""

    domain: str
    repo: str
    path: str
    column: str
    registry: str                  # registry entry of the same dataset (pinned revision)
    why: str

    @property
    def slug(self) -> str:
        return self.repo.replace("/", "__")


HELDOUT_FILES: tuple[HeldoutFile, ...] = (
    HeldoutFile("web", "karpathy/climbmix-400b-shuffle", "shard_00997.parquet", "text", "climbmix",
                "beyond the run's ClimbMix download cap (files are taken in listing order up to the cap)"),
    HeldoutFile("chat", "HuggingFaceTB/smol-smoltalk", "data/test-00000-of-00001.parquet",
                "messages", "smol_smoltalk", "the test split; the registry reads the train files only"),
)


@dataclass
class HeldoutBatch:
    """Rows of one held-out domain: ``tokens int32[n, T+1]``, ``w float32[n, T]``,
    ``seg int32[n, T]`` and the exact-layer features of the inputs; ``probe``: the longest
    single record read for the rows, rendered and cut to ``model.max_seq`` tokens (the read-set
    audit's sequence), or ``None``."""

    domain: str
    arrays: dict[str, np.ndarray]
    probe: np.ndarray | None = None

    @property
    def rows(self) -> int:
        return int(self.arrays["tokens"].shape[0])

    @property
    def weight(self) -> float:
        return float(np.sum(self.arrays["w"]))


class _Watch:
    """Iterates ``items`` and remembers the largest one by ``size(item)``."""

    def __init__(self, items: Iterable[Any], size: Callable[[Any], int]) -> None:
        self.items = iter(items)
        self.size = size
        self.best: Any = None
        self.best_size = -1

    def __iter__(self):
        for x in self.items:
            try:
                s = int(self.size(x))
            except Exception:  # noqa: BLE001 - an odd record is never the probe
                s = -1
            if s > self.best_size:
                self.best, self.best_size = x, s
            yield x


def _conv_chars(msgs: Any) -> int:
    return sum(len(str(m.get("content", ""))) for m in msgs) if isinstance(msgs, (list, tuple)) else -1


def _probe_ids(kind: str, record: Any, tok, max_len: int) -> np.ndarray | None:
    """A record rendered as one segment (``BOS`` first) and cut to ``max_len`` tokens."""
    from bes.config import BOS
    from bes.data import chat_format as cf

    try:
        if kind == "text":
            ids = np.asarray([BOS] + tok.encode(str(record))[:max_len - 1], np.int32)
        else:
            ids = np.asarray(cf.render_conversation(list(record), tok)[0][:max_len], np.int32)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return ids if ids.size > 1 else None


def heldout_path(hf: HeldoutFile, data_dir: str | Path, *, download: bool = True,
                 log: Log | None = None) -> tuple[Path | None, str]:
    """``(local file, note)``: ``<data_dir>/samples/<org>__<name>/<path>``, downloaded at the
    registry's pinned revision when missing and ``download``."""
    local = Path(data_dir) / "samples" / hf.slug / hf.path
    if local.exists():
        return local, "found on disk"
    if not download:
        return None, "missing (downloads disabled)"
    try:
        from huggingface_hub import hf_hub_download

        from bes.data import sources

        rev = sources.get(hf.registry).revision
        _say(log, f"export: downloading held-out file {hf.repo}/{hf.path}")
        got = hf_hub_download(hf.repo, hf.path, repo_type="dataset", revision=rev,
                              local_dir=str(Path(data_dir) / "samples" / hf.slug))
        return Path(got), f"downloaded at revision {rev[:12]}"
    except Exception as e:  # noqa: BLE001 - no network or no such file: the domain is skipped
        return None, f"download failed: {type(e).__name__}: {e}"


def in_training_downloads(hf: HeldoutFile, data_dir: str | Path,
                          tok_dirs: Sequence[str | Path] = ()) -> bool | None:
    """True when the file is among the run's training files: in its registry entry's download
    manifest, or the input of a committed token shard under ``tok_dirs`` (``<root>/tok``; raw
    downloads may be deleted once tokenised). ``None`` when that cannot be told."""
    try:
        from bes.data import download, shards

        man = download.repo_dir(hf.registry, data_dir) / download.MANIFEST
    except Exception:  # noqa: BLE001
        return None
    if man.exists():
        for line in man.read_text().splitlines():
            try:
                if json.loads(line).get("path") == hf.path:
                    return True
            except ValueError:
                continue
    for root in tok_dirs:
        d = Path(root) / hf.registry
        if not d.is_dir():
            continue
        try:
            for e in shards.read_manifest(d).values():
                inp = e.get("input") or {}
                if inp.get("path") == hf.path and inp.get("repo") == hf.repo:
                    return True
        except Exception:  # noqa: BLE001 - an unreadable manifest says nothing
            return None
    return False


def _pad_rows(arrays: dict[str, np.ndarray], multiple: int) -> dict[str, np.ndarray]:
    """Append zero-weight copies of row 0 so the row count is a multiple of ``multiple``."""
    n = arrays["tokens"].shape[0]
    extra = (-n) % multiple
    if not extra or n == 0:
        return arrays
    out = {}
    for k, v in arrays.items():
        rep = np.repeat(v[:1], extra, axis=0)
        if k == "w":
            rep = np.zeros_like(rep)
        out[k] = np.concatenate([v, rep], axis=0)
    return out


def _finish_rows(domain: str, rows: Mapping[str, np.ndarray], T: int, tok,
                 chunk: int, probe: np.ndarray | None = None) -> HeldoutBatch | None:
    from bes.exact.features import exact_features

    if rows["tokens"].shape[0] == 0:
        return None
    arrays = {"tokens": np.asarray(rows["tokens"], np.int32),
              "w": np.asarray(rows["w"], np.float32),
              "seg": np.asarray(rows["seg"], np.int32)}
    arrays.update(exact_features(arrays["tokens"][:, :T], arrays["seg"], tok))
    return HeldoutBatch(domain, _pad_rows(arrays, chunk), probe)


def proxy_samples(tok, rule=None) -> list[Any]:
    """Every user turn of the proxy prompt set as a one-turn conversation sample
    (``BOS USER text END``)."""
    from bes.data import chat_format as cf
    from bes.data.samples import WeightRule, make_sample
    from bes.eval.proxy import prompts as prompt_set

    out = []
    for p in prompt_set.load():
        for turn in p.turns:
            ids, roles = cf.render_conversation([{"role": "user", "content": turn}], tok)
            out.append(make_sample(ids, roles, rule or WeightRule(), source="proxy"))
    return out


def heldout_batches(cfg, tok, *, data_dir: str | Path | None = None, rows: int = 16,
                    seq_len: int | None = None, domains: Sequence[str] = ("web", "chat", "proxy"),
                    download: bool = True, records: Mapping[str, Sequence[Any]] | None = None,
                    chunk: int = 8, tok_dirs: Sequence[str | Path] = (),
                    log: Log | None = None) -> tuple[list[HeldoutBatch], dict]:
    """Held-out rows of ``T = seq_len`` (default ``min(data.seq_len, model.max_seq)``) inputs,
    ``rows`` per domain, packed as the training loader packs (``bes.eval.microbench``'s row
    builders over ``bes.data.packing``). ``records`` replaces the built-in sources of a domain
    (``{"web": [text, ...], "chat": [messages, ...]}``; tests); ``tok_dirs``: the run's
    ``<root>/tok`` directories, checked for shards made from a held-out file. Returns
    ``(batches, info)``; a domain without data is skipped and ``info`` says why."""
    from bes.eval import microbench as MB

    T = int(seq_len or min(cfg.data.seq_len, cfg.model.max_seq))
    data_dir = Path(data_dir if data_dir is not None else REPO / cfg.run.data_dir)
    info: dict[str, Any] = {"seq_len": T, "rows_per_domain": rows, "domains": {}}
    batches: list[HeldoutBatch] = []
    records = dict(records or {})
    by_domain = {hf.domain: hf for hf in HELDOUT_FILES}
    max_len = int(cfg.model.max_seq)
    for dom in domains:
        note: dict[str, Any] = {}
        probe = None
        try:
            if dom == "proxy":
                built, stats = MB.sample_rows(proxy_samples(tok), T, rows)
                note.update(source="bes/eval/proxy/prompts.jsonl (user turns)", held_out=True)
            elif dom in records or dom in by_domain:
                if dom in records:
                    values: Iterable[Any] = list(records[dom])
                    kind = "text" if dom == "web" else "messages"
                    note.update(source="records given by the caller", held_out=None)
                else:
                    hf = by_domain[dom]
                    path, how = heldout_path(hf, data_dir, download=download, log=log)
                    note.update(source=f"{hf.repo}/{hf.path}", why=hf.why, file=how)
                    if path is None:
                        info["domains"][dom] = {**note, "rows": 0, "skipped": how}
                        continue
                    trained = in_training_downloads(hf, data_dir, tok_dirs)
                    note["in_training_downloads"] = trained
                    note["held_out"] = None if trained is None else not trained
                    values, kind = MB.iter_parquet(path, hf.column), hf.column
                if kind == "text":
                    watch = _Watch((v[:MB.MAX_DOC_CHARS] for v in values if isinstance(v, str) and v.strip()),
                                   len)
                    built, stats = MB.document_rows(watch, tok, T, rows)
                else:
                    watch = _Watch(values, _conv_chars)
                    built, stats = MB.sample_rows(MB.conversation_samples(watch, tok, T), T, rows)
                if watch.best is not None:
                    probe = _probe_ids(kind, watch.best, tok, max_len)
            else:
                info["domains"][dom] = {"rows": 0, "skipped": "unknown domain"}
                continue
            batch = _finish_rows(dom, built, T, tok, chunk, probe)
            if batch is None:
                info["domains"][dom] = {**note, "rows": 0, "skipped": "no rows could be built"}
                continue
            batches.append(batch)
            info["domains"][dom] = {**note, "rows": int(built["tokens"].shape[0]),
                                    "weighted_tokens": batch.weight, "stats": stats}
        except Exception as e:  # noqa: BLE001 - one bad domain does not stop the check
            info["domains"][dom] = {**note, "rows": 0, "skipped": f"{type(e).__name__}: {e}"}
    info["weighted_tokens"] = sum(b.weight for b in batches)
    return batches, info


# ===================================================================== held-out loss
def check_model_cfg(model_cfg):
    """The model configuration of the check: float32 activations, dense attention, no remat."""
    return dataclasses.replace(model_cfg, act_dtype="float32", attn_impl="xla", remat=False)


@functools.lru_cache(maxsize=4)
def _metrics_fn(mcfg):
    """One jitted ``loss_fn`` metrics function per (frozen) model configuration, so both
    weight sets of the check share one compilation."""
    import jax

    from bes.model import model as M

    return jax.jit(lambda p, c, b: M.loss_fn(p, c, b, mcfg)[1])


def heldout_loss(params: Any, consts: Any, batches: Sequence[HeldoutBatch], model_cfg, *,
                 chunk: int = 8) -> dict[str, Any]:
    """Weighted mean cross-entropy (nats) over ``batches``, overall and per domain, with
    float32 activations, dense attention and ``highest`` matmul precision."""
    import jax

    mcfg = check_model_cfg(model_cfg)
    sums: dict[str, list[float]] = {}
    with jax.default_matmul_precision("highest"):
        fn = _metrics_fn(mcfg)
        for b in batches:
            acc = sums.setdefault(b.domain, [0.0, 0.0])
            n = b.rows
            for lo in range(0, n, chunk):
                sub = {k: v[lo:lo + chunk] for k, v in b.arrays.items()}
                m = jax.device_get(fn(params, consts, sub))
                acc[0] += float(m["ce_sum"])
                acc[1] += float(m["weight"])
    total_ce = sum(v[0] for v in sums.values())
    total_w = sum(v[1] for v in sums.values())
    return {"loss": total_ce / total_w if total_w > 0 else None, "weight": total_w,
            "by_domain": {d: {"loss": v[0] / v[1] if v[1] > 0 else None, "weight": v[1]}
                          for d, v in sums.items()}}


def check_16bit(params32: Any, params16: Any, consts: Any, batches: Sequence[HeldoutBatch],
                model_cfg, *, tol: float = TOLERANCE_NATS, chunk: int = 8) -> dict[str, Any]:
    """The held-out check of the module docstring. ``ok`` is ``None`` when there are no
    held-out rows (nothing measured)."""
    t0 = time.time()
    if not batches or sum(b.weight for b in batches) <= 0:
        return {"ok": None, "tol": tol, "reason": "no held-out rows", "seconds": 0.0}
    l32 = heldout_loss(params32, consts, batches, model_cfg, chunk=chunk)
    l16 = heldout_loss(params16, consts, batches, model_cfg, chunk=chunk)
    delta = l16["loss"] - l32["loss"]
    by = {}
    for d in l32["by_domain"]:
        a, b = l32["by_domain"][d]["loss"], l16["by_domain"][d]["loss"]
        by[d] = {"loss32": a, "loss16": b, "delta": None if a is None or b is None else b - a,
                 "weight": l32["by_domain"][d]["weight"]}
    ok = bool(math.isfinite(delta) and abs(delta) < tol)
    return {"ok": ok, "tol": tol, "loss32": l32["loss"], "loss16": l16["loss"], "delta": delta,
            "abs_delta": abs(delta), "weighted_tokens": l32["weight"], "by_domain": by,
            "precision": "float32 activations, dense attention, highest matmul precision",
            "seconds": round(time.time() - t0, 2)}


# ===================================================================== accounting
def library_files(path: str | Path) -> dict[str, list[str]]:
    """The files of a built library that an export ships (paths relative to ``path``):

    ``arrays``: every table ``Library.load`` reads, which ``Library.accounting`` counts (the
    packed text, the BM25 index, the title look-up, the question-answer layer and the gate, as
    their classes name them); ``meta``: the JSON files that configure those tables (and hold the
    index and qa header numbers the accounting counts); ``text``: the attribution files of
    :data:`LIBRARY_TEXT_FILES` that exist. Nothing else is shipped: ``provenance/`` (int64 /
    float64 build data, not counted), build reports, the supervisor's settings record.
    Raises ``FileNotFoundError`` when a file the loader needs is missing."""
    from bes.library.index import TruncatedIndex
    from bes.library.pack import EntryBlocks, Packed
    from bes.library.qa import QAStore
    from bes.library.retrieve import TitleTable

    root = Path(path)
    if (root / "answers.json").is_file():
        # an answers library (bes.library.answers): its tables and the JSON that configures them
        from bes.library.answers import AnswerLibrary

        arrays = [f"ans_{k}.npy" for k in (*EntryBlocks.ARRAYS, *AnswerLibrary.PER_ENTRY, *AnswerLibrary.INDEX)]
        missing = [f for f in arrays if not (root / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{root} is not a complete answers library: missing {missing}")
        return {"arrays": sorted(arrays), "meta": ["answers.json"],
                "text": [f for f in LIBRARY_TEXT_FILES if (root / f).is_file()]}
    arrays = [f"pack_{k}.npy" for k in Packed.ARRAYS] + [f"index_{k}.npy" for k in TruncatedIndex.ARRAYS]
    meta = ["library.json", "pack.json", "index.json"]
    if (root / "titlekey_hash.npy").exists():                   # as Library.load decides
        arrays += [f"titlekey_{k}.npy" for k in TitleTable.ARRAYS]
    if (root / "qa.json").exists():
        meta.append("qa.json")
        arrays += [f"qa_{k}.npy" for k in (*EntryBlocks.ARRAYS, *QAStore.PER_ENTRY, *QAStore.INDEX)]
    if (root / "gate_coef.npy").exists():
        arrays.append("gate_coef.npy")
        if (root / "gate.json").exists():
            meta.append("gate.json")
    missing = [f for f in arrays + meta if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is not a complete library: missing {missing}")
    return {"arrays": sorted(arrays), "meta": sorted(meta),
            "text": [f for f in LIBRARY_TEXT_FILES if (root / f).is_file()]}


def _library_marker(path: str | Path) -> Path:
    """The JSON file that marks a built library (``bes.library.retrieve.library_marker``)."""
    from bes.library.retrieve import library_marker

    f = library_marker(path)
    if not f:
        raise FileNotFoundError(f"{path} holds no built library (neither library.json nor answers.json)")
    return Path(f)


def library_file_list(ship: Mapping[str, Sequence[str]]) -> list[str]:
    """Every path of :func:`library_files`, sorted."""
    return sorted({f for part in ("arrays", "meta", "text") for f in ship.get(part, ())})


def library_numbers(path: str | Path, ship: Mapping[str, Sequence[str]],
                    accounting: Mapping[str, Any]) -> dict[str, Any]:
    """The shipped tables against the library's own accounting. Each table holds
    ``ceil(bytes / 2)`` numbers of 16 bits (a ``uint32`` table two per element, as
    ``Library.accounting`` counts it); their sum plus the header numbers the accounting adds
    (``*header_numbers``: scalars kept in the JSON files) must equal ``total_numbers``.
    ``ok`` is False otherwise: a shipped array the accounting does not count, or a counted one
    that would not ship."""
    root = Path(path)
    tables: dict[str, dict[str, Any]] = {}
    for rel in ship.get("arrays", ()):
        a = np.load(root / rel, mmap_mode="r", allow_pickle=False)
        n = (int(a.nbytes) + 1) // 2
        tables[rel] = {"dtype": a.dtype.name, "shape": [int(s) for s in a.shape], "numbers": n,
                       "bits_per_number": round(8.0 * a.nbytes / n, 3) if n else 0.0}
    header = sum(int(v) for k, v in accounting.items() if str(k).endswith("header_numbers"))
    counted = sum(t["numbers"] for t in tables.values())
    total = int(accounting["total_numbers"])
    problems = []
    if counted + header != total:
        problems.append(f"the shipped tables hold {counted:,} numbers plus {header} header numbers, the "
                        f"accounting counts {total:,}")
    problems += [f"{k}: {t['bits_per_number']} bits per counted number" for k, t in tables.items()
                 if t["bits_per_number"] > 16]
    return {"ok": not problems, "problems": problems, "table_numbers": counted,
            "header_numbers": header, "total_numbers": total, "tables": tables,
            "wide_tables": sorted(k for k, t in tables.items() if np.dtype(t["dtype"]).itemsize > 2)}


def library_digest(path: str | Path, files: Sequence[str] | None = None) -> dict[str, Any]:
    """Per-file SHA-256 of a library and their digest (SHA-256 of the sorted
    ``relpath<TAB>sha256`` lines), over ``files`` (default: the shipped files of
    :func:`library_files`; every file when ``path`` is not a complete library)."""
    root = Path(path)
    if files is None:
        try:
            files = library_file_list(library_files(root))
        except (FileNotFoundError, ImportError):
            files = [str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file()]
    hashes = {rel: sha256_file(root / rel) for rel in sorted(files)}
    blob = "".join(f"{k}\t{v}\n" for k, v in sorted(hashes.items())).encode()
    return {"digest": hashlib.sha256(blob).hexdigest(), "files": hashes,
            "bytes": sum((root / k).stat().st_size for k in hashes)}


def library_qa_sources(path: str | Path) -> list[dict[str, Any]]:
    """The question-answer sources a built library stores (``qa.json`` ``sources``), each with
    its registry entry (dataset, config, split, revision, licence, permissive) and the number of
    stored entries (rows of ``ATTRIBUTION_QA.tsv``; ``None`` without that file)."""
    root = Path(path)
    ans = _read_json(root / "answers.json")
    if ans:
        # an answers library records its sources itself (scripts/build_answers.py)
        out = []
        for name, s in sorted(((ans.get("build") or {}).get("sources") or {}).items()):
            lic = str(s.get("licence") or "unknown")
            out.append({"source": name, "repo": s.get("repo"), "config": s.get("files"), "split": "train",
                        "revision": s.get("commit") or s.get("revision"), "license": lic,
                        "permissive": lic.split()[0].lower() in ("apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-3.0",
                                                                 "odc-by-1.0", "llama3.1"),
                        "entries": s.get("stored")})
        return out
    qa = _read_json(root / "qa.json")
    if not qa:
        return []
    counts: dict[str, int] | None = None
    tsv = root / "ATTRIBUTION_QA.tsv"
    if tsv.is_file():
        counts = {}
        with open(tsv, encoding="utf-8", errors="replace") as f:
            header = True
            for line in f:
                if line.startswith("#"):
                    continue
                if header:                      # the column names: entry, source, row
                    header = False
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    counts[parts[1]] = counts.get(parts[1], 0) + 1
    try:
        from bes.library.sources import QA_SOURCES
    except ImportError:  # pragma: no cover - the library package is part of the repository
        QA_SOURCES = {}
    out = []
    for name in sorted(qa.get("sources") or []):
        s = QA_SOURCES.get(name)
        out.append({"source": name, "repo": getattr(s, "repo", None), "config": getattr(s, "config", None),
                    "split": getattr(s, "split", None), "revision": getattr(s, "revision", None),
                    "license": getattr(s, "license", "unknown (not in bes.library.sources)"),
                    "permissive": bool(getattr(s, "permissive", False)),
                    "entries": None if counts is None else counts.get(name, 0)})
    return out


def accounting_report(params: Any, model_cfg, lib: Any = None) -> dict[str, Any]:
    """``bes.budget.count`` of the parameter tree plus the real library: the library and qa
    layer, title table and gate (stored numbers) with the worst-case read of one retrieval
    (``Library.read_bound``) minus its BM25-index part, and the BM25 index with that part."""
    from bes import budget

    assets = None
    lib_info = None
    if lib is not None:
        acc = lib.accounting()
        bound = lib.read_bound()
        assets = budget.Assets(library_numbers=int(acc["total_numbers"]) - int(acc["index_numbers"]),
                               index_numbers=int(acc["index_numbers"]),
                               library_read=int(bound["total"]) - int(bound.get("index", 0)),
                               index_read=int(bound.get("index", 0)))
        lib_info = {"accounting": {k: int(v) for k, v in acc.items() if isinstance(v, (int, np.integer))},
                    "read_bound": {k: int(v) for k, v in bound.items()},
                    "n_wiki": int(lib.n_wiki), "n_qa": int(lib.n_qa)}
    b = budget.count(params, model_cfg, assets)
    return {"total": int(b.total), "active_step": int(b.active_step),
            "caps": {"total": budget.TOTAL_CAP, "active": budget.ACTIVE_CAP},
            "within_caps": bool(b.within_caps()),
            "rows": [{"row": r, "total": int(t), "step": int(s)} for r, (t, s) in b.ledger().items()],
            "library": lib_info}


def probe_sequences(batches: Sequence[HeldoutBatch]) -> list[tuple[str, np.ndarray]]:
    """Read-set probes: the longest held-out record of every domain that has one (web and chat;
    the proxy prompts are short). Without any, the proxy rows' first row (several segments)."""
    out = [(f"{b.domain} (held-out record)", b.probe) for b in batches if b.probe is not None]
    if not out and batches:
        b = batches[0]
        out = [(f"{b.domain} row (several segments)", b.arrays["tokens"][0, :-1])]
    return out


def readset_report(params: Any, consts: Any, model_cfg, tok, probes: Sequence[tuple[str, Any]],
                   *, lib: Any = None, queries: Sequence[Any] = (), windows: Sequence[int] = (),
                   include_critic: bool = False) -> dict[str, Any]:
    """Distinct stored numbers read per generated token (``bes.readset.ReadLogger``, both
    conventions) on probe sequences with real exact-layer features, and the numbers one library
    retrieval reads (``ReadLog``) over ``queries`` (user turns, or ``(turn, earlier user
    turns)`` pairs retrieved with that context, as ``bes.decode.respond`` does)."""
    from bes.exact.features import exact_features
    from bes.readset import ReadLogger

    out: dict[str, Any] = {"probes": [], "include_critic": include_critic}
    for name, ids in probes:
        arr = np.asarray(ids, np.int32)[None, :]
        seg = np.zeros_like(arr)
        feats = exact_features(arr, seg, tok)
        rl = ReadLogger(params, model_cfg, consts, include_critic=include_critic)
        counts = rl.log_sequence({"tokens": arr[0], "seg": seg[0], **{k: v[0] for k, v in feats.items()}})
        row = {"probe": name, "positions": len(counts), "distinct_tokens": int(len(set(arr[0].tolist()))),
               "dense": int(counts[0].dense) if counts else 0,
               "step_mean": float(np.mean([c.step for c in counts])) if counts else None,
               "step_max": int(max(c.step for c in counts)) if counts else None,
               "strict_last": int(counts[-1].strict) if counts else None,
               "strict_max": int(max(c.strict for c in counts)) if counts else None,
               "strict_at": {}}
        for w in windows:
            if 0 < w <= len(counts):
                row["strict_at"][str(w)] = {"strict": int(counts[w - 1].strict),
                                            "distinct_tokens": int(len(set(arr[0, :w].tolist())))}
        out["probes"].append(row)
    if lib is not None and queries:
        from bes.library.retrieve import ReadLog

        reads, n_ret, n_verb = [], 0, 0
        new_read = getattr(lib, "new_read", None)       # an answers library: one capped log per reply
        for q in queries:
            # a query is a user turn, or (turn, earlier user turns) as the responder retrieves
            text, context = (q, None) if isinstance(q, str) else (q[0], list(q[1]) or None)
            log = new_read() if callable(new_read) else ReadLog()
            try:
                if callable(new_read) and lib.verbatim(text, None, context=context, log=log) is not None:
                    n_verb += 1                         # the stored answer is the reply: nothing else is read
                    p = None
                else:
                    p = lib.retrieve(text, log=log, context=context) if context else lib.retrieve(text, log=log)
            except Exception:  # noqa: BLE001 - a query the library cannot handle reads nothing
                continue
            if p is not None:
                n_ret += 1
            reads.append(int(log.total))
        bound = int(lib.read_bound()["total"])
        out["library"] = {"queries": len(queries), "retrieved": n_ret,
                          "read_mean": float(np.mean(reads)) if reads else None,
                          "read_max": int(max(reads)) if reads else None, "read_bound": bound}
        if callable(new_read):
            out["library"]["verbatim"] = n_verb
    steps = [p["step_max"] for p in out["probes"] if p["step_max"] is not None]
    lib_max = (out.get("library") or {}).get("read_max") or 0
    out["step_max_with_retrieval"] = (max(steps) + lib_max) if steps else None
    return out


# ===================================================================== run facts
def _stage_rows(stages: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    from bes.train.stages import ORDER

    if not stages:
        return []
    recs = stages.get("stages", {})
    rows = []
    for name in ORDER:
        r = recs.get(name)
        if r is None:
            continue
        rows.append({"stage": name, "status": r.get("status"), "planned_s": r.get("planned_s"),
                     "slice_s": r.get("slice_s"), "used_s": r.get("used_s"),
                     "attempts": r.get("attempts"), "interrupted": r.get("interrupted"),
                     "reason": r.get("reason") or ""})
    return rows


def run_facts(out_dir: str | Path, cfg=None, reports: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What the run directory records: budget, code, plan, ladder rung, horizon, stage table,
    training ledger, post-training and evaluation results (missing parts are ``None``)."""
    out = Path(out_dir)
    reports = dict(reports or {})
    man = _read_json(reports.get("manifest") or out / "manifest.json") or {}
    stages = _read_json(reports.get("stages") or out / "stages.json")
    ledger = _read_json(out / "train" / "ledger.json")
    bench = man.get("benchmark") or {}
    facts: dict[str, Any] = {
        "run_dir": str(out), "has_manifest": bool(man),
        "config": man.get("config"), "overrides": man.get("overrides"),
        "config_hash": man.get("config_hash"), "code": man.get("code"),
        "budget": man.get("budget"), "resumes": len(man.get("resumes") or []),
        "devices": man.get("devices"), "host": {k: (man.get("host") or {}).get(k) for k in
                                                ("cores", "ram_gb", "disk_total_gb", "python", "platform")},
        "preflight_warnings": man.get("preflight_warnings"),
        "benchmark": None, "stages": _stage_rows(stages), "ledger": None,
        "prefopt": man.get("prefopt"), "decode": man.get("decode"), "judged": man.get("judged"),
        "effective": man.get("effective"),
    }
    if bench:
        can = bench.get("canary") or {}
        facts["benchmark"] = {
            "plan": bench.get("plan_key"), "plan_overrides": bench.get("plan_overrides"),
            "expected_tps": bench.get("expected_tps"), "tokens_per_s": bench.get("tokens_per_s"),
            "horizon_tokens": bench.get("horizon_tokens"),
            "train_seconds_planned": bench.get("train_seconds_planned"),
            "per_seq_len": bench.get("per_seq_len"), "backend": bench.get("backend"),
            "ladder": bench.get("ladder"), "warnings": bench.get("warnings"),
            "canary": {k: (can.get(k) or {}).get("ok") for k in ("parity", "determinism", "safe", "full")},
            "host": bench.get("host"),
            "measurements": [{k: m.get(k) for k in ("plan", "seq_len", "ok", "reason", "compile_s",
                                                    "step_s", "tokens_per_s", "peak_bytes")}
                             for m in bench.get("measurements") or []]}
        if bench.get("sel_attn"):               # the selective-attention decision, when there was one
            facts["benchmark"]["sel_attn"] = bench["sel_attn"]
    if ledger:
        secs = ledger.get("seconds") or {}
        facts["ledger"] = {"tokens": ledger.get("tokens"), "steps": ledger.get("steps"),
                           "seconds": secs, "train_hours": (secs.get("train") or 0.0) / 3600.0,
                           "restarts": ledger.get("restarts"), "rollbacks": ledger.get("rollbacks"),
                           "data_skipped": ledger.get("data_skipped")}
    post = Path(reports.get("post") or out / "post")
    facts["critic"] = _read_json(post / "critic_report.json")
    samples = post / "samples.jsonl"
    if samples.exists():
        n = 0
        for line in samples.read_text().splitlines():
            n += bool(line.strip())
        facts["samples"] = {"prompts": n}
    health = Path(reports.get("health") or out / "health")
    facts["health"] = _read_json(health / "latest.json")
    return facts


def _proxy_root(cfg, out_dir: Path, reports: Mapping[str, str]) -> Path | None:
    if reports.get("proxy"):
        return Path(reports["proxy"]).parent
    try:
        from bes.train.supervisor import proxy_root

        return proxy_root(cfg, out_dir)
    except Exception:  # noqa: BLE001
        return None


def report_files(facts: Mapping[str, Any], proxy_root: Path | None) -> list[Path]:
    """Exact-score and judged-comparison reports this run's manifest refers to."""
    if proxy_root is None:
        return []
    rep = proxy_root / "reports"
    files: list[Path] = []
    dec = facts.get("decode") or {}
    for run in (dec.get("runs") or {}).values():
        rid = run.get("run_id") if isinstance(run, Mapping) else None
        if rid:
            files += sorted((rep / "exact").glob(f"{rid.replace('/', '__')}-*"))
    judged = facts.get("judged") or {}
    for arm in (judged.get("opponents") or {}).values():
        ps = arm.get("pairset") if isinstance(arm, Mapping) else None
        if ps:
            files += sorted((rep / "ab").glob(f"{ps}--*"))
    if judged.get("ours"):
        files += sorted((rep / "exact").glob(f"{judged['ours'].replace('/', '__')}-*"))
    factual = ((dec.get("library") or {}).get("factual") or {}).get("report")
    if factual:                                       # the library decision's measurement
        files += [Path(factual).with_suffix(".md"), Path(factual).with_suffix(".json")]
    return [f for f in dict.fromkeys(files) if f.is_file()]


# ===================================================================== licences
def data_licences(cfg, out_dir: Path, manifest: Mapping[str, Any] | None,
                  library_sources: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Training data sources with token shards on disk (registry licences), the tokenizer's
    training data, the question-answer sources of the shipped library
    (:func:`library_qa_sources`) and the opponent and judge models used for evaluation only.
    ``permissive_violations``: stored qa sources without a clear licence in a run that asked for
    permissive data only (``data.permissive_only``)."""
    from bes.data import sources

    rows = []
    tok_dir = None
    try:
        from bes.train.supervisor import tok_root

        tok_dir = tok_root(cfg, out_dir) / "tok"
    except Exception:  # noqa: BLE001
        pass
    present = set()
    if tok_dir is not None and tok_dir.is_dir():
        present = {p.name for p in tok_dir.iterdir() if (p / "manifest.jsonl").exists()}
    planned: list[str] = []
    try:
        from bes.train.supervisor import download_plan

        planned = list(dict.fromkeys(n for n, _, _ in download_plan(cfg)))
    except Exception:  # noqa: BLE001
        pass
    for name in dict.fromkeys(planned + sorted(present)):
        try:
            s = sources.get(name)
        except KeyError:
            continue
        rows.append({"source": name, "repo": s.repo, "kind": s.kind, "license": s.license,
                     "nonpermissive": bool(s.nonpermissive), "revision": s.revision,
                     "tokenized_in_run": name in present})
    tok_meta = _read_json(REPO / "assets" / "tokenizer" / "meta.json") or {}
    tok_data = [{"domain": k, "repo": v.get("repo_id"), "revision": v.get("revision"),
                 "files": v.get("files")} for k, v in (tok_meta.get("training", {}).get("domains") or {}).items()]
    evaluated = []
    try:
        from bes.eval.proxy import opponents

        names = []
        for item in str(cfg.run.eval_opponents).replace(",", " ").split():
            names.append(item.partition("=")[0])
        names.append(cfg.run.eval_judge)
        for n in dict.fromkeys(names):
            e = opponents.MODELS.get(n)
            if e is not None:
                evaluated.append({"name": n, "role": e.role, "upstream": e.upstream, "repo": e.repo,
                                  "file": e.filename, "license": e.license})
    except Exception:  # noqa: BLE001
        pass
    qa = [dict(s) for s in (library_sources or [])]
    violations = [s["source"] for s in qa if not s.get("permissive")] if cfg.data.permissive_only else []
    return {"training_sources": rows, "tokenizer_training": tok_data, "evaluation_models": evaluated,
            "permissive_only": bool(cfg.data.permissive_only), "library_qa_sources": qa,
            "library_permissive": all(s.get("permissive") for s in qa) if qa else None,
            "permissive_violations": violations}


# ===================================================================== configuration files
CONFIG_PY = '''"""Configuration of an exported model (written by bes/export.py; do not edit).

``bes.config.load_config("<export>/config.py")`` returns the configuration the run trained with
(the effective configuration: execution plan, fallback-ladder switches and horizon applied).
"""
import json
from pathlib import Path

from bes.config import Cfg, DataCfg, DecodeCfg, ModelCfg, OptimCfg, RunCfg

_D = json.loads(Path(__file__).with_name("config.json").read_text())["cfg"]
CFG = Cfg(model=ModelCfg(**_D["model"]), optim=OptimCfg(**_D["optim"]), data=DataCfg(**_D["data"]),
          run=RunCfg(**_D["run"]), decode=DecodeCfg(**_D["decode"]))
'''


def load_export_config(path: str | Path):
    """``load_config`` of an export's ``config.py`` without writing a ``__pycache__`` next to it
    (the export directory is hashed)."""
    import sys

    from bes.config import load_config

    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return load_config(str(path))
    finally:
        sys.dont_write_bytecode = old


def _hashed_files(d: Path) -> list[Path]:
    """Every file of an export directory that ``SHA256SUMS`` covers (``__pycache__`` is not)."""
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.name != SUMS
            and "__pycache__" not in p.relative_to(d).parts]


def default_decode_overrides(cfg) -> list[str]:
    """Harness settings that make an arm of ours decode with the run's own ``DecodeCfg`` (the
    same list ``scripts/run_24h.py`` gives the decode selection)."""
    d = cfg.decode
    return [f"extra.particles={d.particles}", f"extra.use_library={str(d.use_library).lower()}",
            f"max_tokens={d.max_new_tokens}", f"temperature={d.temperature!r}", f"top_p={d.top_p!r}"]


def decode_settings(overrides: Sequence[str]) -> dict[str, Any]:
    """The plain settings dict our generator receives from the harness (``ours`` preset
    ``default`` with ``overrides``), as ``bes.eval.proxy`` would build it."""
    from bes.eval.proxy.__main__ import ours_presets
    from bes.eval.proxy.run_ours import settings_for_generator

    return settings_for_generator(ours_presets()["default"].with_overrides(list(overrides)))


def enable_switches(consts: Mapping[str, Any]) -> dict[str, Any]:
    """``make_consts`` keyword arguments that rebuild ``consts["enable"]``."""
    en = consts["enable"]
    out: dict[str, Any] = {}
    for k, v in en.items():
        a = np.asarray(v, np.float32)
        out[k] = float(a) if a.ndim == 0 else [float(x) for x in a.reshape(-1)]
    return out


def consts_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    import jax

    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    return ta == tb and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


# ===================================================================== model card
def _fmt_int(x: Any) -> str:
    return f"{int(x):,}" if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) else str(x)


def _fmt_f(x: Any, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else str(x)


def _fmt_signed(x: Any, nd: int = 6) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else str(x)


def _secs(x: Any) -> str:
    if not isinstance(x, (int, float)):
        return "-"
    return f"{x / 3600:.2f} h" if x >= 3600 else f"{x / 60:.1f} min" if x >= 60 else f"{x:.0f} s"


#: The model's runtime switches (``bes.model.model.SWITCHES``) as the card names them.
_FEATURE_NAMES = (("reg", "registers"), ("cand_in", "candidate-tape input features (n-gram count heads)"),
                  ("cand_out", "candidate-tape masses in the output law"),
                  ("sink", "calibrated attention sinks"), ("conf", "the confidence channel"),
                  ("clock", "structural clocks"), ("gain", "per-token gains"),
                  ("sel", "selective attention"), ("gate", "the head-wise attention output gate"),
                  ("page", "token neurons"))
#: Switches of the paths that exist only when a ``ModelCfg`` flag is on (the v9 paths,
#: ``docs/V9.md``): the card names them only for a model that has them.
_PATH_FLAGS = {"gain": ("gain_ffn", "gain_o", "gain_v"), "sel": ("sel_attn",),
               "gate": ("attn_gate",), "page": ("tok_neurons",)}


def _has_path(model: Mapping[str, Any], key: str) -> bool:
    flags = _PATH_FLAGS.get(key)
    return flags is None or any(bool(model.get(f)) for f in flags)


def _pages_text(m: Mapping[str, Any]) -> str:
    """The v9 token pages and the attention gate of a model configuration, or ``""``."""
    parts = []
    if m.get("tok_neurons"):
        parts.append(f"{m['tok_neurons']} private feed-forward neurons per block in a "
                     f"{m.get('tok_dim')}-dimensional space shared by all tokens")
    gains = [name for name, flag in (("feed-forward neurons", m.get("gain_ffn")),
                                     ("attention outputs", m.get("gain_o")),
                                     ("values", m.get("gain_v"))) if flag]
    if gains:
        parts.append("gains on the " + ", ".join(gains) + " of the positions that hold it")
    s = (" Every vocabulary id owns a page of " + " and ".join(parts) + ", read once per token "
         "(`docs/V9.md`).") if parts else ""
    if m.get("attn_gate"):
        s += " A head-wise gate scales the output of every attention head."
    return s


def _switch_on(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return any(float(x) != 0.0 for x in v)
    try:
        return float(v) != 0.0
    except (TypeError, ValueError):
        return None


def what_this_is(rec: Mapping[str, Any]) -> str:
    """The card's first paragraph, from the export record only: the architecture, the feature
    paths the shipped runtime switches leave on and off (and the fallback-ladder rung), whether
    a library ships, and how replies are decoded."""
    m = (((rec.get("config") or {}).get("cfg") or {}).get("model")) or {}
    ffn = m.get("ffn_width") or (m.get("d_model") or 0) * (m.get("ffn_mult") or 0)
    s = (f"A looped transformer: d_model {m.get('d_model')}, {m.get('n_blocks')} shared blocks applied "
         f"{m.get('n_loops')} times, {m.get('n_heads')} heads of {m.get('head_dim')}, feed-forward width "
         f"{ffn}, vocabulary {m.get('vocab_size')}, context {m.get('max_seq')} tokens (`docs/PLAN.md`).")
    s += _pages_text(m)
    en = (rec.get("consts") or {}).get("enable") or {}
    on = [name for key, name in _FEATURE_NAMES if _has_path(m, key) and _switch_on(en.get(key))]
    off = [name for key, name in _FEATURE_NAMES
           if _has_path(m, key) and _switch_on(en.get(key)) is False]
    if not en:
        s += " The runtime switches of the weights were not recorded."
    if on:
        s += " Feature paths on: " + ", ".join(on) + "."
    if off:
        s += " Switched off: " + ", ".join(off) + "."
    unset = set((rec.get("consts") or {}).get("defaults") or ())
    filled = [name for key, name in _FEATURE_NAMES if _has_path(m, key) and key in unset]
    if filled:
        s += (" The checkpoint stored no switch value for " + ", ".join(filled)
              + " (written before the switch existed, or without its switches): the configuration's "
                "default is used.")
    lad = ((rec.get("run") or {}).get("benchmark") or {}).get("ladder") or {}
    if lad.get("groups"):
        s += f" The canary put the run on fallback ladder rung {lad.get('rung')} ({', '.join(lad['groups'])} off)."
    lib = rec.get("library") or {}
    settings = (rec.get("decode_choice") or {}).get("settings") or {}
    if lib.get("mode") not in (None, "none") and lib.get("kind") == "answers":
        s += (" A written answers library ships with it (questions with the answers teacher models wrote, "
              "stored as text, not trained): a question it holds is answered with the stored answer, and "
              "similar entries are shown to the model as exemplars before the user's turn; "
              f"{'on' if settings.get('use_library', True) else 'off'} by default.")
    elif lib.get("mode") not in (None, "none"):
        s += (" A written library (Simple English Wikipedia passages and question-answer pairs) ships with "
              f"it behind a calibrated retrieval gate, {'on' if settings.get('use_library', True) else 'off'} "
              "by default.")
    else:
        s += " No written library ships with it" + (f" ({lib['not_shipped']})" if lib.get("not_shipped") else "") + "."
    k = settings.get("particles")
    if isinstance(k, int) and not isinstance(k, bool) and k > 1:
        s += (f" Replies are decoded with K = {k} particles steered by exact potentials and chosen by a "
              "satisficing rule.")
    elif k == 1:
        s += " Replies are decoded with one sample (K = 1)."
    else:
        s += " The decoding settings were not recorded."
    return s


def library_decision_text(dec: Mapping[str, Any] | None) -> str:
    """One line on the decode selection's library decision (``manifest.json`` ``decode.library``):
    shipped or not, why, and the numbers it rests on."""
    if not dec:
        return "no library decision recorded"
    out = [("shipped" if dec.get("ship") else "not shipped") + f": {dec.get('reason')}"]
    fa = dec.get("factual") or {}
    if fa:
        out.append(f"factual questions: {fa.get('n_paired')} of {fa.get('n_questions')} answered with the "
                   f"library on and off, gate fired on {fa.get('fired')}; helped {fa.get('helped')}, hurt "
                   f"{fa.get('hurt')} (sign test p = {fa.get('p')}); right with the library {fa.get('lib_acc')}, "
                   f"without {fa.get('nolib_acc')}" + (" (stopped early)" if fa.get("stopped_early") else ""))
    elif dec.get("factual_error"):
        out.append(f"factual measurement: {dec['factual_error']}")
    dev = dec.get("dev") or {}
    if dev:
        out.append(f"dev exact score with the library {dev.get('on')}, without {dev.get('off')} over "
                   f"{dev.get('n_common')} prompts")
    return "; ".join(out)


def model_card(rec: Mapping[str, Any]) -> str:
    """``MODEL_CARD.md`` from an ``export.json`` record: measured facts only; anything missing
    is reported as missing."""
    L: list[str] = []
    chk = rec.get("check") or {}
    facts = rec.get("run") or {}
    src = rec.get("source") or {}
    L.append(f"# Model card: BES-100M-A20M, export of run `{Path(str(facts.get('run_dir') or '?')).name}`")
    L.append("")
    L.append(f"Written by `bes/export.py` (export format {rec.get('format')}) on {rec.get('created')} "
             "from the files of this run and measurements made during the export. Nothing here is "
             "a projection; a fact the run did not record is marked as missing.")
    L.append("")
    if chk.get("ok") is not True:
        why = ("the check could not be run: " + str(chk.get("reason") or chk.get("error"))) if chk.get("ok") is None \
            else (f"|loss16 - loss32| = {_fmt_f(chk.get('abs_delta'), 5)} nats is not below "
                  f"{chk.get('tol')} nats")
        L += ["> **WARNING: the 16-bit weights did NOT pass the held-out check** (" + why + "). Both "
              f"`{WEIGHTS16}` and the float32 `{WEIGHTS32}` are kept; compare them before shipping.", ""]
    if rec.get("errors"):
        L += ["> **Parts of the export failed** (details in `export.json`, key `errors`): "
              + "; ".join(f"{e.get('part')}: {e.get('error')}" for e in rec["errors"]), ""]
    L += ["## What this is", ""]
    L.append(what_this_is(rec))
    L.append("")
    L.append(f"- checkpoint: `{src.get('checkpoint')}` (step {src.get('step')}, {src.get('weights')} weights)")
    L.append(f"- configuration hash {rec.get('config', {}).get('config_hash')}; configuration file "
             f"`{rec.get('config', {}).get('config')}`, overrides `{rec.get('config', {}).get('overrides')}`")
    code = facts.get("code") or {}
    L.append(f"- code commit {code.get('commit')} (uncommitted changes: {code.get('dirty')})")
    L.append("")

    # ---------------------------------------------------------------- weights
    w = rec.get("weights") or {}
    L += ["## Stored weights (at most 16 bits per number)", ""]
    L.append("| format | leaves | numbers | largest relative L2 error of a leaf |")
    L.append("|---|---|---|---|")
    for name in DTYPES:
        s = w.get(name) or {}
        L.append(f"| {name} | {s.get('leaves', 0)} | {_fmt_int(s.get('numbers', 0))} | "
                 f"{_fmt_f(s.get('max_rel_err', 0.0), 6)} |")
    L.append(f"| total | {sum((w.get(n) or {}).get('leaves', 0) for n in DTYPES)} | "
             f"{_fmt_int(w.get('numbers', 0))} | |")
    L.append("")
    L.append("Each leaf is stored in the format with the smaller measured relative error "
             "(`weights.json` lists both errors per leaf); the loader restores float32.")
    L.append("")
    L += ["### Held-out check", ""]
    if chk.get("ok") is None:
        L.append(f"Not measured: {chk.get('reason') or chk.get('error')}.")
    else:
        L.append(f"Mean cross-entropy on {_fmt_int(chk.get('weighted_tokens'))} held-out tokens "
                 f"({chk.get('precision')}): float32 weights {_fmt_f(chk.get('loss32'), 5)} nats, "
                 f"16-bit weights {_fmt_f(chk.get('loss16'), 5)} nats, difference "
                 f"{_fmt_signed(chk.get('delta'))} nats (tolerance {chk.get('tol')}): "
                 f"**{'passed' if chk.get('ok') else 'FAILED'}**.")
        L.append("")
        L.append("| domain | weighted tokens | float32 | 16-bit | difference |")
        L.append("|---|---|---|---|---|")
        for d, v in (chk.get("by_domain") or {}).items():
            L.append(f"| {d} | {_fmt_int(v.get('weight'))} | {_fmt_f(v.get('loss32'), 5)} | "
                     f"{_fmt_f(v.get('loss16'), 5)} | {_fmt_signed(v.get('delta'))} |")
    ho = rec.get("heldout") or {}
    if ho.get("domains"):
        L.append("")
        L.append("Held-out data:")
        for d, v in ho["domains"].items():
            flag = ""
            if v.get("in_training_downloads"):
                flag = " **(this file is also among the run's training downloads: not held out)**"
            L.append(f"- {d}: {v.get('source')}, {v.get('rows', 0)} rows of {ho.get('seq_len')} tokens"
                     + (f"; skipped: {v['skipped']}" if v.get("skipped") else "")
                     + (f" ({v['why']})" if v.get("why") else "") + flag)
    L.append("")

    # ---------------------------------------------------------------- accounting
    acc = rec.get("accounting") or {}
    L += ["## Accounting (`bes.budget`, real library sizes)", ""]
    if acc.get("rows"):
        L.append("| row | stored numbers | numbers read per generated token |")
        L.append("|---|---|---|")
        for r in acc["rows"]:
            L.append(f"| {r['row']} | {_fmt_int(r['total'])} | {_fmt_int(r['step'])} |")
        L.append(f"| **total** | **{_fmt_int(acc.get('total'))}** | **{_fmt_int(acc.get('active_step'))}** |")
        L.append("")
        caps = acc.get("caps") or {}
        L.append(f"Caps {_fmt_int(caps.get('total'))} stored / {_fmt_int(caps.get('active'))} read per "
                 f"token: {'within both' if acc.get('within_caps') else '**EXCEEDED**'}. The per-token "
                 "column is the standard convention of `docs/ACCOUNTING.md`: every dense leaf, one row "
                 "per lookup read, and the worst-case read of one library retrieval charged to the step "
                 "that retrieves.")
        lib = acc.get("library")
        if lib and (rec.get("library") or {}).get("kind") == "answers":
            la = lib.get("accounting") or {}
            L.append("")
            L.append(f"Answers library: {_fmt_int(lib.get('n_qa'))} question-answer entries; "
                     f"{_fmt_int(la.get('total_numbers'))} stored numbers (compressed text and its directory "
                     f"{_fmt_int(la.get('text_numbers'))}, per-entry rows {_fmt_int(la.get('entry_numbers'))}, index "
                     f"{_fmt_int(la.get('index_numbers'))}); one reply reads at most "
                     f"{_fmt_int((lib.get('read_bound') or {}).get('total'))} of them (a cap the read log enforces).")
        elif lib:
            la = lib.get("accounting") or {}
            L.append("")
            L.append(f"Library: {_fmt_int(lib.get('n_wiki'))} wiki passages and {_fmt_int(lib.get('n_qa'))} "
                     f"question-answer entries; {_fmt_int(la.get('total_numbers'))} stored numbers "
                     f"(text blocks, directory and titles {_fmt_int(la.get('library_numbers'))}, BM25 index "
                     f"{_fmt_int(la.get('index_numbers'))}, title look-up {_fmt_int(la.get('title_table_numbers'))}, "
                     f"qa layer {_fmt_int(la.get('qa_numbers'))}, gate {_fmt_int(la.get('gate_numbers'))}); "
                     f"worst-case read of one retrieval {_fmt_int((lib.get('read_bound') or {}).get('total'))}.")
            lr = rec.get("library") or {}
            nb = lr.get("numbers") or {}
            if nb:
                L.append("")
                wide = nb.get("wide_tables") or []
                L.append(f"The export's library holds exactly these tables: {len(nb.get('tables') or {})} arrays "
                         f"of {_fmt_int(nb.get('table_numbers'))} numbers plus {nb.get('header_numbers')} header "
                         f"numbers in their JSON files = {_fmt_int(nb.get('total_numbers'))}, at most 16 bits per "
                         "counted number" + (f" (the 32-bit tables {', '.join(f'`{w}`' for w in wide)} count two "
                                            "numbers per element)" if wide else "")
                         + ", plus the attribution text."
                         + (f" Left out (not counted, never read by the model): "
                            f"{', '.join(f'`{x}`' for x in lr['left_out'])}." if lr.get("left_out") else ""))
        else:
            L.append("")
            L.append("No library is shipped with this export"
                     + (f": {(rec.get('library') or {}).get('not_shipped')}." if (rec.get("library") or {}).get("not_shipped") else "."))
    else:
        L.append(f"Not computed: {acc.get('error', 'missing')}.")
    L.append("")
    rs = rec.get("readset") or {}
    L += ["## Read-set audit (`bes.readset.ReadLogger`)", ""]
    if rs.get("probes"):
        L.append("Distinct stored numbers read for each generated token on held-out sequences with "
                 "their real exact-layer features. *step*: the standard convention (given the cache); "
                 "*strict*: the union over the visible context window.")
        L.append("")
        L.append("| probe | positions | distinct tokens | step mean | step max | strict at the end | strict max |")
        L.append("|---|---|---|---|---|---|---|")
        for p in rs["probes"]:
            L.append(f"| {p['probe']} | {p['positions']} | {p['distinct_tokens']} | "
                     f"{_fmt_int(round(p['step_mean'])) if p.get('step_mean') is not None else '-'} | "
                     f"{_fmt_int(p.get('step_max'))} | {_fmt_int(p.get('strict_last'))} | {_fmt_int(p.get('strict_max'))} |")
        for p in rs["probes"]:
            for wlen, v in (p.get("strict_at") or {}).items():
                L.append(f"- {p['probe']}, window of {wlen} positions: strict {_fmt_int(v['strict'])} "
                         f"({v['distinct_tokens']} distinct tokens)")
        lib = rs.get("library")
        if lib:
            L.append("")
            L.append(f"Library retrievals over {lib['queries']} proxy-prompt turns: {lib['retrieved']} "
                     f"returned a passage; numbers read per retrieval mean {_fmt_f(lib.get('read_mean'), 0)}, "
                     f"max {_fmt_int(lib.get('read_max'))} (bound {_fmt_int(lib.get('read_bound'))}). Largest "
                     f"step count plus the largest measured retrieval: {_fmt_int(rs.get('step_max_with_retrieval'))}.")
        L.append(f"The critic head is {'included' if rs.get('include_critic') else 'not read (the decoder does not use it)'}.")
    else:
        L.append(f"Not measured: {rs.get('error', 'missing')}.")
    L.append("")

    # ---------------------------------------------------------------- training
    L += ["## Training run", ""]
    if not facts.get("has_manifest"):
        L.append("The export did not come from a run directory with a manifest: no run facts.")
    budget = facts.get("budget") or {}
    if budget:
        L.append(f"- budget {_secs(budget.get('budget_s'))} of wall time from "
                 f"{_dt.datetime.fromtimestamp(budget['t0'], _dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if budget.get('t0') else '?'}; "
                 f"{facts.get('resumes', 0)} resumes")
    dev = facts.get("devices") or {}
    if dev:
        L.append(f"- devices: {[d.get('kind') for d in dev.get('devices', [])]} (jax {dev.get('jax')}, "
                 f"backend {dev.get('backend')})")
    led = facts.get("ledger")
    if led:
        secs = led.get("seconds") or {}
        L.append(f"- trained {_fmt_int(led.get('tokens'))} tokens in {_fmt_int(led.get('steps'))} optimiser "
                 f"steps; {led.get('train_hours', 0):.2f} h in the step loop; set-up {_secs(secs.get('setup'))}, "
                 f"checkpoints {_secs(secs.get('ckpt'))}, lost to restarts and rollbacks {_secs(secs.get('lost'))}; "
                 f"{led.get('restarts')} restarts, {led.get('rollbacks')} rollbacks")
    else:
        L.append("- training ledger: missing")
    b = facts.get("benchmark")
    if b:
        lad = b.get("ladder") or {}
        L.append(f"- execution plan {b.get('plan')} ({_fmt_int(round(b.get('expected_tps') or 0))} tokens/s "
                 f"expected), horizon {b.get('horizon_tokens'):.3e} tokens" if b.get("horizon_tokens") else
                 f"- execution plan {b.get('plan')}")
        L.append(f"- fallback ladder rung {lad.get('rung')}"
                 + (f": {', '.join(lad.get('groups') or [])} switched off" if lad.get("groups") else " (full configuration)"))
        sa = b.get("sel_attn")
        if sa:
            tps = {k: _fmt_int(round(sa.get(k) or 0)) for k in ("sel_tps", "plain_tps")}
            L.append(f"- selective attention {'kept' if sa.get('kept') else 'switched off'} by the "
                     f"start-up benchmark: {sa.get('sel_plan')} {tps['sel_tps']} tokens/s, without it "
                     f"{sa.get('plain_plan')} {tps['plain_tps']} tokens/s ({sa.get('reason')})")
        can = b.get("canary") or {}
        L.append(f"- canary: parity {can.get('parity')}, determinism {can.get('determinism')}, "
                 f"safe run {can.get('safe')}, full run {can.get('full')}")
    if facts.get("stages"):
        L.append("")
        L.append("| stage | status | planned | used | note |")
        L.append("|---|---|---|---|---|")
        for r in facts["stages"]:
            note = (r.get("reason") or "")[:120].replace("|", "/")
            if r.get("interrupted"):
                note = (note + f" interrupted {r['interrupted']}x").strip()
            L.append(f"| {r['stage']} | {r['status']} | {_secs(r.get('planned_s'))} | {_secs(r.get('used_s'))} | {note} |")
    L.append("")

    # ---------------------------------------------------------------- post-training / evaluation
    L += ["## Post-training and evaluation", ""]
    po = facts.get("prefopt")
    if po:
        L.append(f"- preference optimisation: {'accepted' if po.get('accepted') else 'not accepted'} "
                 f"(lr {po.get('lr')}, {po.get('n_train_pairs')} training pairs, {po.get('n_heldout_pairs')} held out)"
                 + (f"; {po.get('reason')}" if po.get("reason") and not po.get("accepted") else ""))
    else:
        L.append("- preference optimisation: no record")
    cr = facts.get("critic")
    if cr:
        L.append(f"- critic head: held-out rank correlation {cr.get('rho_heldout')}, gate passed "
                 f"{cr.get('use_critic')} (the decoder has no critic path)")
    dec = facts.get("decode")
    if dec:
        L.append(f"- decoding choice on the dev split: `{dec.get('chosen')}` by exact scores "
                 f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (dec.get('scores') or {}).items()} } "
                 f"over {dec.get('n_common')} common prompts of {dec.get('n_dev')}")
        if "library" in dec:
            L.append(f"- library decision ({(dec.get('library') or {}).get('rule')}): "
                     f"{library_decision_text(dec.get('library'))}")
    else:
        L.append("- decoding choice: no record (the run's own `DecodeCfg` is the default)")
    jd = facts.get("judged")
    if jd and jd.get("opponents"):
        L.append("")
        L.append(f"Judged blind comparison on the held-out (test) split, our run `{jd.get('ours')}` "
                 f"({jd.get('ours_done')} of {jd.get('n_test')} prompts answered):")
        L.append("")
        L.append("| opponent arm | rated by | n | score | 95% interval | decisive win rate | partial |")
        L.append("|---|---|---|---|---|---|---|")
        for arm, v in jd["opponents"].items():
            if v.get("skipped"):
                L.append(f"| {arm} | - | - | - | - | - | skipped: {v['skipped']} |")
                continue
            for srcname, s in (v.get("scores") or {}).items():
                ci = s.get("ci95") or ["-", "-"]
                L.append(f"| {arm} | {srcname} | {s.get('n')} | {s.get('score')} | [{ci[0]}, {ci[1]}] | "
                         f"{s.get('win_rate_decisive')} | {s.get('partial')} |")
        L.append("")
        L.append("Score = (wins + ties/2) / n from our side. With n comparisons the interval is about "
                 "plus or minus 1/sqrt(n); differences under about 5 points are not resolvable on 400 "
                 "prompts (`docs/EVAL_PROTOCOL.md` section 8). A stand-in judge (`judge-standin-*`) "
                 "measures nothing about quality; it only exercises the plumbing.")
    else:
        L.append("- judged comparison: no record")
    files = rec.get("reports") or []
    if files:
        L.append("")
        L.append("Reports copied into `reports/`: " + ", ".join(f"`{f}`" for f in files))
    L.append("")

    # ---------------------------------------------------------------- disclosures
    disc = rec.get("disclosures") or {}
    L += ["## Disclosures (`docs/PLAN.md` section 2)", ""]
    L.append(f"- **Library**: {disc.get('library')}")
    L.append(f"- **Inference-time compute**: {disc.get('particles')}")
    L.append(f"- **External models**: {disc.get('external_models')}")
    L.append(f"- **Preference data**: {disc.get('preference_data')}")
    L.append("")

    # ---------------------------------------------------------------- licences
    lic = rec.get("licences") or {}
    L += ["## Data, licences and attribution", ""]
    if lic.get("training_sources"):
        L.append("| training source | dataset | licence (dataset card) | non-permissive | tokenised in this run |")
        L.append("|---|---|---|---|---|")
        for r in lic["training_sources"]:
            L.append(f"| {r['source']} | {r['repo']} | {r['license']} | {r['nonpermissive']} | {r['tokenized_in_run']} |")
        L.append("")
        L.append("Caveats per source (synthetic outputs of third-party models, upstream terms): "
                 "`docs/DATA_SOURCES.md` section 4.")
    if lic.get("tokenizer_training"):
        L.append("")
        L.append("Tokenizer trained on: " + "; ".join(f"{t['repo']} {t.get('files')}" for t in lic["tokenizer_training"]))
    libsrc = rec.get("library") or {}
    if libsrc.get("mode") not in (None, "none"):
        L.append("")
        if libsrc.get("kind") == "answers":
            L.append(f"Answers library ({libsrc.get('mode')}, digest {str(libsrc.get('digest'))[:16]}...): the "
                     "question-answer pairs below, from datasets whose cards state a licence; attribution in "
                     "`ATTRIBUTION.md` of the library directory"
                     + (" (copied next to this card as `LIBRARY_ATTRIBUTION.md`)." if libsrc.get("attribution") else "."))
        else:
            L.append(f"Library ({libsrc.get('mode')}, digest {str(libsrc.get('digest'))[:16]}...): Simple English "
                     "Wikipedia text (CC BY-SA 3.0 / GFDL) and the question-answer pairs below; per-article and "
                     "per-entry attribution in `ATTRIBUTION.md`, `ATTRIBUTION.tsv` and `ATTRIBUTION_QA.tsv` of the "
                     "library directory" + (" (copied next to this card as `LIBRARY_ATTRIBUTION.md`)." if libsrc.get("attribution") else "."))
        qa = lic.get("library_qa_sources") or libsrc.get("qa_sources") or []
        if qa:
            L.append("")
            L.append("| question-answer source (stored in the library) | dataset | config / split | licence (dataset card) | permissive | entries |")
            L.append("|---|---|---|---|---|---|")
            for s in qa:
                L.append(f"| {s.get('source')} | {s.get('repo')} | {s.get('config')} / {s.get('split')} | "
                         f"{s.get('license')} | {s.get('permissive')} | {_fmt_int(s.get('entries')) if s.get('entries') is not None else '-'} |")
        elif libsrc.get("qa_sources") == []:
            L.append("")
            L.append("The library stores no question-answer pairs.")
        bs = libsrc.get("build_settings") or {}
        L.append("")
        L.append(f"Build settings recorded with the library: permissive only {bs.get('permissive_only')}, "
                 f"question-answer sources asked for {bs.get('qa_sources')}, tokenizer sha256 "
                 f"{str(bs.get('tokenizer_sha256'))[:16]}..." if bs else
                 "No build-settings record came with the library (built outside the run's library task): its "
                 "tokenizer and `LibraryCfg` were not checked against the run's.")
    if lic.get("permissive_violations"):
        L.append("")
        L.append(f"**The run asked for permissive data only (`data.permissive_only`), but the library stores "
                 f"question-answer sources without a clear licence: {', '.join(lic['permissive_violations'])}.**")
    if lic.get("evaluation_models"):
        L.append("")
        L.append("Models used for evaluation only (never trained on, never shipped):")
        for e in lic["evaluation_models"]:
            L.append(f"- {e['name']} ({e['role']}): {e['upstream']}, file {e['repo']}/{e['file']}, licence {e['license']}")
    L.append("")

    if rec.get("eviction"):
        from bes import evict

        L += [evict.card_section(rec["eviction"])]

    # ---------------------------------------------------------------- files
    L += ["## Files", ""]
    L.append(f"SHA-256 of every file is in `{SUMS}` (`sha256sum -c {SUMS}` or "
             "`python scripts/export.py --verify <dir>`). Main files:")
    L.append("")
    L.append("| file | bytes | sha256 |")
    L.append("|---|---|---|")
    for f in rec.get("main_files") or []:
        L.append(f"| `{f['path']}` | {_fmt_int(f['bytes'])} | `{f['sha256'][:16]}...` |")
    L.append("")
    L.append("Load for inference: `from bes.export import load_export; m = load_export(dir)`; chat: "
             "`python scripts/chat.py --export <dir>`; blind arena: `python scripts/arena.py --export <dir>`.")
    return "\n".join(L) + "\n"


# ===================================================================== export
def _latest_checkpoint(out: Path, man: Mapping[str, Any]) -> tuple[Path, str]:
    """The weights ``scripts/run_24h.py`` exports (``Run.latest_params``): the accepted post-DPO
    checkpoint, else the newer of ``train/ckpt`` and the parameters-only final save
    ``train/ckpt_params`` (on a tie the full checkpoint)."""
    from bes.decode import respond
    from bes.train import checkpoint as CK

    post = out / "post" / "ckpt"
    if (man.get("prefopt") or {}).get("accepted") and CK.latest(post) is not None:
        return respond.find_checkpoint(post), "prefopt"
    best = None
    for rank, root in ((1, out / "train" / "ckpt"), (0, out / "train" / "ckpt_params")):
        info = CK.latest(root)
        if info is not None and (best is None or (info.step, rank) > best[0]):
            best = ((info.step, rank), info.path)
    return (best[1] if best is not None else respond.find_checkpoint(out / "train")), "train"


def resolve_cfg(out: Path, man: Mapping[str, Any] | None, config: str | None = None,
                overrides: Sequence[str] | None = None):
    """The effective configuration of a run (``bes.train.supervisor.effective_config`` once the
    benchmark is complete, else its base configuration), or ``load_config(config, overrides)``."""
    from bes.config import load_config

    if man:
        from bes.train import supervisor as SV

        if (man.get("benchmark") or {}).get("complete"):
            return SV.effective_config(man)
        return SV.base_config(man)
    if not config:
        raise ExportError(f"{out} holds no run manifest: give the configuration (config= or --config)")
    return load_config(str(config), list(overrides or []))


def _copy_reports(out: Path, files: Sequence[Path], dst: Path, reports: Mapping[str, str]) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    pairs = [(Path(reports.get("manifest") or out / "manifest.json"), "manifest.json"),
             (Path(reports.get("stages") or out / "stages.json"), "stages.json"),
             (out / "stage_table.json", "stage_table.json"),
             (out / "train" / "ledger.json", "train_ledger.json"),
             (out / "train" / "run.json", "train_run.json"),
             (Path(reports.get("post") or out / "post") / "prefopt_report.json", "prefopt_report.json"),
             (Path(reports.get("post") or out / "post") / "critic_report.json", "critic_report.json"),
             (Path(reports.get("health") or out / "health") / "latest.json", "health_latest.json")]
    pairs += [(f, f"proxy/{f.parent.name}/{f.name}") for f in files]
    for src, name in pairs:
        if src.is_file():
            target = dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(f"reports/{name}")
    return copied


def write_sums(d: Path) -> dict[str, str]:
    """Write ``SHA256SUMS`` for every file under ``d`` (itself excluded); returns the map."""
    sums = {str(p.relative_to(d)): sha256_file(p) for p in _hashed_files(d)}
    (d / SUMS).write_text("".join(f"{h}  {k}\n" for k, h in sorted(sums.items())))
    return sums


def verify_export(export_dir: str | Path, *, library: bool = True) -> list[str]:
    """Problems of an export directory (empty list = intact): every file of ``SHA256SUMS``
    present with its hash, no unlisted file, and (pointer mode) the pointed library unchanged."""
    d = Path(export_dir)
    problems = []
    sums_path = d / SUMS
    if not sums_path.exists():
        return [f"{sums_path} is missing"]
    listed = {}
    for line in sums_path.read_text().splitlines():
        if not line.strip():
            continue
        h, _, rel = line.partition("  ")
        listed[rel] = h
    for rel, h in listed.items():
        p = d / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_file(p) != h:
            problems.append(f"{rel}: SHA-256 differs")
    for p in _hashed_files(d):
        if str(p.relative_to(d)) not in listed:
            problems.append(f"{p.relative_to(d)}: not listed in {SUMS}")
    if library and (d / POINTER).exists():
        ptr = _read_json(d / POINTER) or {}
        path = Path(str(ptr.get("path", "")))
        files = list(ptr["files"]) if isinstance(ptr.get("files"), Mapping) else None
        if not path.is_dir():
            problems.append(f"library pointer: {path} does not exist")
        elif files is not None and any(not (path / f).is_file() for f in files):
            problems.append(f"library pointer: {path} changed since the export (files missing)")
        elif library_digest(path, files)["digest"] != ptr.get("digest"):
            problems.append(f"library pointer: {path} changed since the export (digest differs)")
    return problems


def export_run(out_dir: str | Path, *, checkpoint: str | Path | None = None,
               config: str | None = None, overrides: Sequence[str] | None = None,
               horizon_tokens: float | None = None, library: str | Path | None = None,
               final_dir: str | Path | None = None, reports: Mapping[str, str] | None = None,
               decode: Mapping[str, Any] | None = None, cfg=None, library_mode: str = "copy",
               tol: float = TOLERANCE_NATS, heldout_rows: int = 16, download_heldout: bool = True,
               data_dir: str | Path | None = None,
               heldout_records: Mapping[str, Sequence[Any]] | None = None,
               heldout_domains: Sequence[str] = ("web", "chat", "proxy"),
               log: Log | None = None) -> dict[str, Any]:
    """Export a run (module docstring) into ``final_dir`` (default ``<out_dir>/final``); returns
    a summary. ``out_dir`` is the run directory; the other arguments default to what its manifest
    records (``scripts/run_24h.py`` passes them). ``library_mode``: ``"copy"`` (self-contained),
    ``"pointer"`` (path and digest) or ``"none"``. The essential parts (weights, configuration,
    tokenizer, hashes) raise on failure; every measured part records its error instead."""
    from bes.config import config_hash, to_dict
    from bes.decode import respond
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    if library_mode not in ("copy", "pointer", "none"):
        raise ValueError(f"library_mode must be copy, pointer or none, not {library_mode!r}")
    t_start = time.time()
    out = Path(out_dir)
    reports = {k: str(v) for k, v in (reports or {}).items()}
    man = _read_json(Path(reports.get("manifest") or out / "manifest.json"))
    errors: list[dict[str, str]] = []

    def failed(part: str, e: BaseException) -> None:
        errors.append({"part": part, "error": f"{type(e).__name__}: {e}"})
        _say(log, f"export: WARNING {part} failed: {type(e).__name__}: {e}")

    if cfg is None:
        cfg = resolve_cfg(out, man, config, overrides)
    if checkpoint is None:
        if not man:
            raise ExportError(f"{out} holds no run manifest: give the checkpoint")
        ckpt, source = _latest_checkpoint(out, man)
    else:
        ckpt = respond.find_checkpoint(checkpoint)
        source = "prefopt" if "post" in Path(ckpt).parts else "train"
    final = Path(final_dir) if final_dir is not None else out / "final"
    if library is None and man:
        mk = _read_json(out / "state" / "library.json") or {}
        library = mk.get("path") or None
    if library is not None and not cfg.data.use_library and not cfg.decode.use_library:
        library = None
    if decode is None and man:
        decode = man.get("decode")
    # the decode selection's library decision (scripts/run_24h.py stage_select): a library that
    # did not pass its rule is neither shipped nor counted
    lib_decision = dict((decode or {}).get("library") or {}) if isinstance(decode, Mapping) else {}
    not_shipped = None
    if library is None and lib_decision.get("ship"):
        errors.append({"part": "library", "error": "the decode selection shipped the library, but no built library "
                                                   "was found for this export (a run continued on a machine whose "
                                                   "library task had not finished?): exported without it"})
    if library is not None and lib_decision.get("ship") is False:
        not_shipped = f"the decode selection's library rule did not pass: {lib_decision.get('reason')}"
        _say(log, f"export: the library is not shipped ({not_shipped})")
        library = None
    _say(log, f"export: {ckpt} ({source} weights) -> {final}")

    # ------------------------------------------------------------- weights (essential)
    params32, consts, info = respond.load_checkpoint(ckpt, cfg.model)
    if info.get("consts_filled") or info.get("consts_dropped"):
        _say(log, f"export: runtime switches of the checkpoint ({info['consts']}): at their defaults "
                  f"{info.get('consts_filled')}, ignored {info.get('consts_dropped')}")
    arrays16, choices = quantize(params32)
    params16 = dequantize(arrays16, choices, cfg.model)
    rebuilt = M.make_consts(cfg.model, **enable_switches(consts))
    consts_ok = consts_equal(rebuilt, consts)
    tok = load_tokenizer()

    # ------------------------------------------------------------- held-out check
    heldout_info: dict[str, Any] = {}
    batches: list[HeldoutBatch] = []
    try:
        tok_dirs = []
        try:
            from bes.train.supervisor import tok_root

            tok_dirs = [tok_root(cfg, out) / "tok"]
        except Exception:  # noqa: BLE001
            pass
        batches, heldout_info = heldout_batches(cfg, tok, data_dir=data_dir, rows=heldout_rows,
                                                domains=heldout_domains, download=download_heldout,
                                                records=heldout_records, tok_dirs=tok_dirs, log=log)
    except Exception as e:  # noqa: BLE001
        failed("held-out rows", e)
    try:
        check = check_16bit(params32, params16, consts, batches, cfg.model, tol=tol)
    except Exception as e:  # noqa: BLE001
        failed("held-out check", e)
        check = {"ok": None, "tol": tol, "error": f"{type(e).__name__}: {e}"}
    if check.get("ok") is True:
        _say(log, f"export: held-out check passed: loss32 {check['loss32']:.5f}, loss16 "
                  f"{check['loss16']:.5f}, |delta| {check['abs_delta']:.2e} < {tol} nats")
    elif check.get("ok") is False:
        _say(log, "export: " + "!" * 70)
        _say(log, f"export: HELD-OUT CHECK FAILED: |loss16 - loss32| = {check['abs_delta']:.5f} nats "
                  f"is not below {tol}; the float32 weights are kept next to the 16-bit ones")
        _say(log, "export: " + "!" * 70)
    else:
        _say(log, f"export: WARNING the held-out check could not run "
                  f"({check.get('reason') or check.get('error')}); the float32 weights are kept too")

    # ------------------------------------------------------------- library, accounting, audit
    lib = None
    lib_rec: dict[str, Any] = {"mode": "none"}
    if not_shipped:
        lib_rec["not_shipped"] = not_shipped
    lib_digest_files: dict[str, str] = {}
    lib_ship: list[str] = []
    if library is not None and library_mode != "none":
        try:
            from bes.library.retrieve import Library

            lib = Library.load(str(library))
            ship = library_files(library)
            lib_ship = library_file_list(ship)
            numbers = library_numbers(library, ship, lib.accounting())
            if not numbers["ok"]:
                raise ExportError(f"the library's tables do not match its accounting: {numbers['problems']}")
            dig = library_digest(library, lib_ship)
            every = {str(p.relative_to(library)) for p in Path(library).rglob("*") if p.is_file()}
            lib_rec = {"mode": library_mode, "source_path": str(Path(library).resolve()),
                       "digest": dig["digest"], "bytes": dig["bytes"], "files": len(dig["files"]),
                       "meta_sha256": sha256_file(_library_marker(library)),
                       "kind": "answers" if (Path(library) / "answers.json").is_file() else "passages",
                       "shipped": ship, "numbers": numbers, "left_out": sorted(every - set(lib_ship)),
                       "qa_sources": library_qa_sources(library),
                       "build_settings": _read_json(Path(library) / LIBRARY_SETTINGS),
                       "decision": lib_decision or None}
            lib_digest_files = dig["files"]
        except Exception as e:  # noqa: BLE001
            failed("library", e)
            lib, lib_rec = None, {"mode": "none", "error": f"{type(e).__name__}: {e}"}
            lib_ship = []
    try:
        accounting = accounting_report(params16, cfg.model, lib)
    except Exception as e:  # noqa: BLE001
        failed("accounting", e)
        accounting = {"error": f"{type(e).__name__}: {e}"}
    try:
        from bes.eval.proxy import prompts as prompt_set

        probes = probe_sequences(batches)
        # every user turn with the earlier turns of its prompt (bes.decode.respond retrieves so)
        queries = [(t, tuple(p.turns[:i])) for p in prompt_set.load() for i, t in enumerate(p.turns)]
        readset = readset_report(params16, consts, cfg.model, tok, probes, lib=lib, queries=queries,
                                 windows=sorted({cfg.data.seq_len, cfg.data.seq_len_late}),
                                 include_critic=bool(cfg.decode.use_critic))
    except Exception as e:  # noqa: BLE001
        failed("read-set audit", e)
        readset = {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------- run facts
    try:
        facts = run_facts(out, cfg, reports)
    except Exception as e:  # noqa: BLE001
        failed("run facts", e)
        facts = {"run_dir": str(out), "error": f"{type(e).__name__}: {e}"}
    if isinstance(decode, Mapping):                  # the choice this export applies (card, reports)
        facts["decode"] = dict(decode)
    try:
        licences = data_licences(cfg, out, man, lib_rec.get("qa_sources") if lib is not None else None)
        if licences.get("permissive_violations"):
            errors.append({"part": "licences", "error": "the run asked for permissive data only, the shipped "
                                                        "library stores question-answer sources without a clear "
                                                        f"licence: {licences['permissive_violations']}"})
    except Exception as e:  # noqa: BLE001
        failed("licences", e)
        licences = {"error": f"{type(e).__name__}: {e}"}
    proxy = _proxy_root(cfg, out, reports)
    try:
        rep_files = report_files(facts, proxy)
    except Exception as e:  # noqa: BLE001
        failed("report files", e)
        rep_files = []

    # ------------------------------------------------------------- decoding choice
    chosen = list((decode or {}).get("overrides") or default_decode_overrides(cfg))
    try:
        settings = decode_settings(chosen)
    except Exception as e:  # noqa: BLE001
        failed("decode settings", e)
        settings = {}
    k = settings.get("particles", cfg.decode.particles)
    use_lib = bool(settings.get("use_library", cfg.decode.use_library)) and lib is not None
    disclosures = {
        "library": ("shipped and on by default" if use_lib else
                    "shipped but off by default" if lib is not None else
                    f"not shipped ({not_shipped})" if not_shipped else "not shipped")
        + (f"; frozen and hashed before any prompt (digest {lib_rec.get('digest', '')[:16]}...), offline, "
           "counted in the stored numbers above" if lib is not None else "")
        + (f"; library decision: {library_decision_text(lib_decision)}" if lib_decision else "")
        + (f"; decode selection variants measured on dev: {sorted(((decode or {}).get('scores') or {}).keys())}"
           if decode else "; no decode selection recorded"),
        "particles": f"K = {k} samples of this model per reply (particle decoding with exact potentials, "
                     "then satisficing selection); the decode selection compares K = 1 when K > 1",
        "external_models": (f"scoring only: the judge `{cfg.run.eval_judge}` rates blinded pairs and the "
                            f"opponents ({cfg.run.eval_opponents}) are only compared against; no training "
                            "step reads another model's logits (no distillation); "
                            f"run.use_external_models = {cfg.run.use_external_models}"),
        "preference_data": (f"DPO pairs by source: {(facts.get('prefopt') or {}).get('pairs_by_source')}; "
                            "sampled pairs are labelled by the exact verifiers; UltraFeedback pairs (when "
                            "used) carry labels produced by third-party models (dataset card)")
        if facts.get("prefopt") else "no preference optimisation record",
    }

    # ------------------------------------------------------------- write (temporary dir)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".tmp-{final.name}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir()
    try:
        write_npz(tmp / WEIGHTS16, arrays16)
        keep32 = check.get("ok") is not True
        if keep32:
            paths, leaves = flat_params(params32)
            write_npz(tmp / WEIGHTS32, {p: np.asarray(v, np.float32) for p, v in zip(paths, leaves)})
        _write_json(tmp / WEIGHTS_JSON, {"format": FORMAT, "rule": "smaller relative L2 error; ties float16",
                                        "leaves": [c.to_dict() for c in choices]})
        cfg_rec = {"cfg": to_dict(cfg), "config_hash": config_hash(cfg), "config": config or (man or {}).get("config"),
                   "overrides": list(overrides) if overrides is not None else (man or {}).get("overrides"),
                   "run_overrides": ((man or {}).get("effective") or {}).get("overrides"),
                   "horizon_tokens": horizon_tokens if horizon_tokens is not None
                   else ((man or {}).get("benchmark") or {}).get("horizon_tokens"),
                   "decode_choice": {"chosen": (decode or {}).get("chosen", "default"),
                                     "overrides": chosen, "settings": settings}}
        _write_json(tmp / "config.json", cfg_rec)
        (tmp / "config.py").write_text(CONFIG_PY)
        if config_hash(load_export_config(tmp / "config.py")) != config_hash(cfg):
            raise ExportError("config.py does not rebuild the configuration (hash differs)")
        tdir = tmp / "tokenizer"
        tdir.mkdir()
        from bes.tokenizer.tok import ASSET_DIR, ASSET_FILE, META_FILE

        for name in (ASSET_FILE, META_FILE):
            if (ASSET_DIR / name).exists():
                shutil.copy2(ASSET_DIR / name, tdir / name)
        if load_tokenizer(tdir).sha256 != tok.sha256:
            raise ExportError("the copied tokenizer differs from the one the model was trained with")
        if lib is not None:
            if library_mode == "copy":
                for rel in lib_ship:            # the counted tables, their JSON, attribution
                    (tmp / "library" / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(library) / rel, tmp / "library" / rel)
                from bes.library.retrieve import Library

                if Library.load(str(tmp / "library")).accounting() != lib.accounting() or \
                        library_digest(tmp / "library")["digest"] != lib_rec["digest"]:
                    raise ExportError("the copied library differs from the built one")
                lib_rec["path"] = "library"
            else:
                _write_json(tmp / POINTER, {"path": lib_rec["source_path"], "digest": lib_rec["digest"],
                                            "files": lib_digest_files})
                lib_rec["path"] = lib_rec["source_path"]
            attr = Path(library) / "ATTRIBUTION.md"
            if attr.exists():
                shutil.copy2(attr, tmp / "LIBRARY_ATTRIBUTION.md")
                lib_rec["attribution"] = "LIBRARY_ATTRIBUTION.md"
        copied = []
        try:
            copied = _copy_reports(out, rep_files, tmp / "reports", reports)
        except Exception as e:  # noqa: BLE001
            failed("copying reports", e)
        rec: dict[str, Any] = {
            "format": FORMAT, "export_version": EXPORT_VERSION,
            "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source": {"checkpoint": str(ckpt), "step": info.get("step"), "seq": info.get("seq"),
                       "config_hash": info.get("config_hash"), "weights": source,
                       "fingerprint": respond.checkpoint_fingerprint(ckpt)},
            "config": {k2: cfg_rec[k2] for k2 in ("cfg", "config_hash", "config", "overrides",
                                                  "run_overrides", "horizon_tokens")},
            "decode_choice": cfg_rec["decode_choice"],
            "weights": {**dtype_summary(choices), "sha256": sha256_file(tmp / WEIGHTS16),
                        "float32_kept": keep32,
                        "sha256_float32": sha256_file(tmp / WEIGHTS32) if keep32 else None},
            "consts": {"enable": enable_switches(consts), "rebuilt_exactly": consts_ok,
                       "source": info.get("consts"), "defaults": list(info.get("consts_filled") or []),
                       "ignored": list(info.get("consts_dropped") or [])},
            "tokenizer": {"sha256": tok.sha256, "path": "tokenizer"},
            "library": lib_rec, "check": check, "heldout": heldout_info,
            "accounting": accounting, "readset": readset, "run": facts, "licences": licences,
            "disclosures": disclosures, "reports": copied, "errors": errors,
        }
        if not consts_ok:
            errors.append({"part": "consts", "error": "the checkpoint's runtime switches are not rebuilt "
                                                      "exactly from the configuration; enable values stored"})
        if info.get("consts_dropped"):
            errors.append({"part": "consts", "error": "runtime switches stored in the checkpoint that this "
                                                      f"configuration does not have were ignored: "
                                                      f"{info['consts_dropped']}"})
        main = [WEIGHTS16] + ([WEIGHTS32] if keep32 else []) + [WEIGHTS_JSON, "config.json",
                                                                "tokenizer/tokenizer.json"]
        rec["main_files"] = [{"path": f, "bytes": (tmp / f).stat().st_size, "sha256": sha256_file(tmp / f)}
                             for f in main if (tmp / f).exists()]
        rec["seconds"] = round(time.time() - t_start, 1)
        _write_json(tmp / EXPORT_JSON, rec)
        (tmp / CARD).write_text(model_card(json.loads((tmp / EXPORT_JSON).read_text())))
        write_sums(tmp)
        problems = verify_export(tmp, library=False)
        if problems:
            raise ExportError(f"the written export does not verify: {problems[:5]}")
        # swap into place: an earlier (fallback or emergency) export is replaced
        old = None
        if final.exists():
            old = final.with_name(f".old-{final.name}-{uuid.uuid4().hex[:8]}")
            os.replace(final, old)
        os.replace(tmp, final)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if lib_rec.get("mode") == "copy":
        lib_rec["path"] = str(final / "library")
    summary = {"export_dir": str(final), "checkpoint": str(ckpt), "weights": source,
               "check_ok": check.get("ok"), "abs_delta": check.get("abs_delta"), "tol": tol,
               "float32_kept": check.get("ok") is not True, "total_numbers": accounting.get("total"),
               "active_step": accounting.get("active_step"), "library": lib_rec.get("mode"),
               "errors": len(errors), "seconds": round(time.time() - t_start, 1)}
    _say(log, f"export: done in {summary['seconds']} s: {final} (check {check.get('ok')}, "
              f"{len(errors)} parts failed)")
    return summary


# ===================================================================== loading
@dataclass
class Exported:
    """A loaded export: float32 parameters, consts, configuration, tokenizer, library."""

    path: Path
    cfg: Any
    params: Any
    consts: Any
    tok: Any
    library: Any
    record: dict
    weights: str = "16"
    notes: list[str] = field(default_factory=list)

    @property
    def step(self) -> Any:
        return (self.record.get("source") or {}).get("step")

    def default_overrides(self) -> list[str]:
        """Harness overrides of the run's decoding choice (``config.json``)."""
        ch = (self.record.get("decode_choice") or {}).get("overrides")
        return list(ch) if ch else default_decode_overrides(self.cfg)

    def settings(self, overrides: Sequence[str] = ()) -> dict[str, Any]:
        """Plain settings for :meth:`bes.decode.respond.Responder.reply`: the run's decoding
        choice plus harness-style ``overrides`` (``"extra.particles=1"``, ``"temperature=0.7"``)."""
        return decode_settings(self.default_overrides() + list(overrides))

    def responder(self, *, decoder: Any = None):
        """A :class:`bes.decode.respond.Responder` on this model (one compiled decoder shared by
        every reply)."""
        from bes.decode.lm import OurLM
        from bes.decode.respond import Responder
        from bes.model import generate as G

        dec = decoder if decoder is not None else G.jitted(self.cfg.model)

        def lm_factory(max_new_tokens: int):
            return OurLM(self.params, self.consts, self.cfg.model, self.tok,
                         max_steps=4 * int(max_new_tokens), decoder=dec)

        return Responder(lm_factory, self.tok, self.cfg, library=self.library)

    def key(self) -> str:
        """Cache key of the generator: weights, configuration, decoding code, library."""
        from bes.config import config_hash
        from bes.decode.respond import RESPOND_VERSION, code_hash

        blob = {"v": RESPOND_VERSION, "export": EXPORT_VERSION,
                "weights": (self.record.get("weights") or {}).get(
                    "sha256" if self.weights == "16" else "sha256_float32"),
                "which": self.weights, "cfg": config_hash(self.cfg), "code": code_hash(),
                "library": (self.record.get("library") or {}).get("digest") if self.library is not None else "none"}
        digest = hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()
        return f"ours-export-step{self.step}-{digest[:16]}"

    def generator(self) -> Callable[..., dict[str, Any]]:
        """``generate(messages, settings, seed) -> dict`` (the ``bes/eval/proxy/run_ours.py``
        contract) with ``key``, ``responder`` and ``export`` attributes."""
        resp = self.responder()

        def generate(messages, settings, seed):
            return resp.reply(messages, settings, seed)

        generate.key = self.key()              # type: ignore[attr-defined]
        generate.responder = resp              # type: ignore[attr-defined]
        generate.export = self                 # type: ignore[attr-defined]
        return generate


def load_export(export_dir: str | Path, *, weights: str = "16", library: Any = "auto",
                verify: bool = True, act_dtype: str | None = None, log: Log | None = None) -> Exported:
    """Load an export for inference.

    weights: ``"16"`` (shipped) or ``"32"`` (the float32 copy of a failed check); library:
    ``"auto"`` (the exported library, if any), ``None`` / ``False`` (none) or a library path;
    verify: check every file against ``SHA256SUMS`` first (raises :class:`ExportError`);
    act_dtype: activation dtype of decoding (default: the configuration's on a TPU, ``float32``
    elsewhere: bfloat16 matmuls are slow on CPUs)."""
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    d = Path(export_dir)
    rec = _read_json(d / EXPORT_JSON)
    if not rec or rec.get("format") != FORMAT:
        raise ExportError(f"{d} is not an export directory of format {FORMAT} (no {EXPORT_JSON})")
    if verify:
        problems = verify_export(d, library=library == "auto")
        if problems:
            raise ExportError(f"{d} does not verify: {problems[:5]}")
    cfg = load_export_config(d / "config.py")
    notes = []
    if act_dtype is None:
        import jax

        if jax.default_backend() != "tpu" and cfg.model.act_dtype != "float32":
            act_dtype = "float32"
            notes.append(f"decoding with float32 activations on {jax.default_backend()} "
                         f"(the configuration says {cfg.model.act_dtype})")
    if act_dtype:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, act_dtype=act_dtype))
    tok = load_tokenizer(d / "tokenizer")
    want = (rec.get("tokenizer") or {}).get("sha256")
    if want and tok.sha256 != want:
        raise ExportError(f"tokenizer sha256 {tok.sha256} differs from the export record {want}")
    if want and load_tokenizer().sha256 != want:
        notes.append("the repository's tokenizer differs from the exported one; library passages are "
                     "cut with the repository's")
    w = rec.get("weights") or {}
    params = load_weights(d, cfg.model, weights)
    enable = (rec.get("consts") or {}).get("enable") or {}
    consts = M.make_consts(cfg.model, **enable)
    lib = None
    if library == "auto":
        lr = rec.get("library") or {}
        if lr.get("mode") == "copy" and (d / "library").is_dir():
            lib_path = d / "library"
        elif lr.get("mode") == "pointer":
            lib_path = Path(str((_read_json(d / POINTER) or {}).get("path", "")))
        else:
            lib_path = None
        if lib_path is not None:
            from bes.library.retrieve import Library

            lib = Library.load(str(lib_path))
    elif library:
        from bes.library.retrieve import Library

        lib = Library.load(str(library))
    if weights == "16" and w.get("float32_kept"):
        notes.append("the 16-bit weights did not pass the held-out check at export time "
                     "(load with weights='32' to compare)")
    for n in notes:
        _say(log, f"export: note: {n}")
    return Exported(d, cfg, params, consts, tok, lib, rec, weights, notes)


def export_generator(export: str, overrides: str = "", key: str = "", library: str = "auto",
                     weights: str = "16", verify: str = "true") -> Callable[..., dict[str, Any]]:
    """Harness factory (``--generator bes.export:export_generator --generator-arg export=DIR``;
    string keyword arguments). overrides: comma-separated ``decode.*`` configuration overrides;
    library: ``auto`` | ``none`` | a path; weights: ``16`` | ``32``."""
    from bes.config import apply_overrides

    lib: Any = {"auto": "auto", "none": None, "": None}.get(library, library)
    ex = load_export(export, weights=weights, library=lib,
                     verify=str(verify).lower() in ("1", "true", "yes", "on"))
    items = [x.strip() for x in str(overrides).split(",") if x.strip()]
    if items:
        bad = [o for o in items if not o.startswith("decode.")]
        if bad:
            raise ValueError(f"only decode.* overrides change an exported model's replies, not {bad}")
        ex.cfg = apply_overrides(ex.cfg, items)
    gen = ex.generator()
    if key:
        gen.key = key                          # type: ignore[attr-defined]
    return gen


__all__ = ["FORMAT", "TOLERANCE_NATS", "ExportError", "LeafChoice", "choose_16bit", "quantize",
           "dequantize", "load_weights", "HeldoutFile", "HELDOUT_FILES", "heldout_batches",
           "heldout_loss", "check_16bit", "accounting_report", "readset_report", "run_facts",
           "data_licences", "model_card", "export_run", "verify_export", "load_export",
           "Exported", "export_generator", "library_digest", "library_files", "library_file_list",
           "library_numbers", "library_qa_sources", "library_decision_text", "what_this_is",
           "decode_settings", "default_decode_overrides"]
