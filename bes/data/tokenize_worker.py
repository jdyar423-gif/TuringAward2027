"""Downloaded parquet files -> token shards, with a pool of worker processes.

One input file of a source becomes one shard, ``tok/<source>/<n>.*`` with ``n`` the file's
``LocalFile.index`` (its position in the source's cached listing). Records are read in batches
(``download.iter_records``), mapped by the source adapter (``bes.data.sources``) and rendered by
``bes.data.chat_format``: documents with ``render_document`` (no role array), conversations with
``render_conversation`` (role array), preference pairs as two consecutive conversation segments
(chosen, rejected) with the same ``rec``. Conversations are stored whole; truncation at turn
boundaries (``chat_format.truncate_to_turns``) is left to the loader.

Determinism: a shard is a pure function of (input file, adapter, tokenizer, ``max_records``),
so the output does not depend on the number of workers or on scheduling.

Crash safety: workers only write shard files (unique temporary names, fsync, rename); the
coordinator alone appends manifest lines, one per finished shard, after the files are in place.
A restart skips committed shards and redoes the rest, so nothing is lost or committed twice.
Workers ask the kernel to kill them when the coordinator dies (Linux ``PR_SET_PDEATHSIG``); a
worker that dies (OOM kill, failed start-up) makes its files fail instead of hanging the run.
Under the ``spawn`` start method, a script that calls ``run`` needs an
``if __name__ == "__main__":`` guard.

The tokenizer is injected: an object with ``encode(text) -> list[int]`` and the
``special_tokens`` mapping that ``chat_format`` checks, or a spec string resolved in every
worker: ``"bytes"`` (``ByteTokenizer``, one id per UTF-8 byte) or ``"module:attr"`` (called if
callable; default ``"bes.tokenizer.tok:load"``).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bes.config import SPECIAL_TOKENS, VOCAB_SIZE, special_id
from bes.data import download, shards, sources

DEFAULT_TOKENIZER = "bes.tokenizer.tok:load"
TOK_DIR = "tok"
#: A file whose adapter raises on more than this fraction of its records (and on more than
#: ``MAX_ERRORS_ABS`` records) is not committed: that is a bug, not noise.
MAX_ERROR_FRAC = 0.01
MAX_ERRORS_ABS = 5


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- tokenizers
class ByteTokenizer:
    """Stand-in tokenizer: one id per UTF-8 byte (ids ``0..255`` are the raw bytes of the real
    vocabulary, so its output is valid input for the model, only unmerged)."""

    fingerprint = "bytes/1"

    def __init__(self) -> None:
        self.vocab_size = VOCAB_SIZE
        self.special_tokens = {name: special_id(name) for name in SPECIAL_TOKENS}

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        return bytes(int(i) for i in ids if 0 <= int(i) < 256)

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


def resolve_tokenizer(spec: Any) -> Any:
    """Tokenizer object from ``spec`` (see the module docstring)."""
    if isinstance(spec, str):
        if spec == "bytes":
            return ByteTokenizer()
        module, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(module), attr or "load")
        return obj() if callable(obj) and not hasattr(obj, "encode") else obj
    if not hasattr(spec, "encode"):
        raise TypeError(f"tokenizer spec {spec!r} has no encode()")
    return spec


_PROBE = ("The quick brown fox jumps over the lazy dog. 3.14159 = pi?\n\tdef f(x): return x**2\n"
          "Ünïcödé, 中文, emoji \U0001F600, <|end|> <|bos|>")


def tokenizer_fingerprint(tok: Any) -> str:
    """Identity of a tokenizer: its asset hash when it has one, else a hash of probe encodings."""
    if isinstance(getattr(tok, "fingerprint", None), str):
        return tok.fingerprint
    if isinstance(getattr(tok, "sha256", None), str):
        return f"sha256:{tok.sha256}"
    blob = json.dumps([int(getattr(tok, "vocab_size", 0)), [int(i) for i in tok.encode(_PROBE)]])
    return "probe:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def code_fingerprint() -> str:
    """Hash of the code that decides shard contents (adapters, chat format, this module)."""
    from bes.data import chat_format

    h = hashlib.sha256()
    for mod in (sources, chat_format, sys.modules[__name__]):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------ one shard
@dataclass(frozen=True)
class Job:
    source: str
    shard: int
    local: str
    input: dict[str, Any]
    out_dir: str
    max_records: int | None = None
    batch_rows: int = 1024


def _content_bytes(content: Any) -> int:
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    total = 0
    for p in content:
        for key in ("text", "expr", "result"):
            if isinstance(p.get(key), str):
                total += len(p[key].encode("utf-8"))
    return total


def tokenize_file(job: Job, tok: Any, *, code: str | None = None) -> dict[str, Any]:
    """Tokenise one input file into shard ``job.shard``; returns its manifest entry.

    The files are in place when this returns; committing (``shards.append_manifest``) is the
    caller's job. Raises if the adapter fails on too many records (nothing is committed).
    """
    from bes.data import chat_format

    src = sources.get(job.source)
    t0, c0 = time.perf_counter(), time.process_time()
    st = {"records": 0, "segments": 0, "skipped": 0, "rejected": 0, "errors": 0,
          "tokens": 0, "text_bytes": 0}
    first_errors: list[str] = []
    writer = shards.ShardWriter(job.out_dir, job.shard, with_roles=src.fmt != "text")
    try:
        row = -1
        for batch in download.iter_records(job.local, src.columns, batch_rows=job.batch_rows):
            for record in batch:
                row += 1
                if job.max_records is not None and row >= job.max_records:
                    break
                st["records"] += 1
                try:
                    item = sources.adapt(src, record)
                except Exception as exc:  # noqa: BLE001 - counted, and fatal above a threshold
                    st["errors"] += 1
                    if len(first_errors) < 3:
                        first_errors.append(f"row {row}: {type(exc).__name__}: {exc}")
                    continue
                if item is None:
                    st["skipped"] += 1
                    continue
                try:
                    if isinstance(item, str):
                        ids, _ = chat_format.render_document(item, tok)
                        rendered = [(ids, None)]
                        st["text_bytes"] += len(item.encode("utf-8"))
                    else:
                        convs = [item.chosen, item.rejected] if isinstance(item, sources.Pair) else [item]
                        rendered = [chat_format.render_conversation(c, tok) for c in convs]
                        st["text_bytes"] += sum(_content_bytes(m["content"]) for c in convs for m in c)
                except ValueError:
                    st["rejected"] += 1
                    continue
                for ids, roles in rendered:
                    writer.add(ids, roles, rec=row)
                    st["segments"] += 1
                    st["tokens"] += int(len(ids))
            if job.max_records is not None and row + 1 >= job.max_records:
                break
        if st["errors"] > MAX_ERRORS_ABS and st["errors"] > MAX_ERROR_FRAC * max(st["records"], 1):
            raise RuntimeError(f"{src.name} shard {job.shard}: adapter failed on {st['errors']} of "
                               f"{st['records']} records, e.g. {first_errors}")
        entry = writer.close()
    except BaseException:
        writer.abort()
        raise
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    entry.update({
        "source": src.name, "kind": src.kind, "fmt": src.fmt, "pairs": src.fmt == "pair",
        "input": dict(job.input), "tokenizer": tokenizer_fingerprint(tok),
        "code": code or code_fingerprint(), "max_records": job.max_records, "stats": st,
        "first_errors": first_errors,
        "timing": {"wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                   "tokens_per_cpu_s": round(st["tokens"] / cpu, 1) if cpu > 0 else None},
    })
    return entry


# ------------------------------------------------------------------------ worker pool
_WORKER: dict[str, Any] = {}


def _die_with_parent() -> None:
    """Linux: deliver SIGKILL to this process when its parent dies (no orphaned workers)."""
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGKILL))   # PR_SET_PDEATHSIG = 1
    except (OSError, AttributeError):  # pragma: no cover - not Linux
        pass


def _init_worker(tokenizer_spec: Any, code: str, parent_pid: int | None = None) -> None:
    _die_with_parent()
    # A worker still starting up when the coordinator was killed set the death signal too late;
    # it would then wait forever for work (a spawned worker holds both ends of its task queue).
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(1)
    _WORKER["tok"] = resolve_tokenizer(tokenizer_spec)
    _WORKER["code"] = code


def _work(job: Job) -> dict[str, Any]:
    try:
        return tokenize_file(job, _WORKER["tok"], code=_WORKER["code"])
    except Exception as exc:  # noqa: BLE001 - reported to the coordinator, which decides
        return {"error": f"{type(exc).__name__}: {exc}", "shard": job.shard, "source": job.source}


@dataclass
class RunStats:
    """Counters of one ``run`` call (``tokens_per_cpu_s`` ~ tokens per second per core)."""

    source: str
    files: int = 0
    already_done: int = 0
    committed: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)
    records: int = 0
    segments: int = 0
    skipped: int = 0
    rejected: int = 0
    errors: int = 0
    tokens: int = 0
    text_bytes: int = 0
    wall_s: float = 0.0
    worker_cpu_s: float = 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def tokens_per_cpu_s(self) -> float:
        return self.tokens / self.worker_cpu_s if self.worker_cpu_s > 0 else 0.0

    @property
    def bytes_per_token(self) -> float:
        return self.text_bytes / self.tokens if self.tokens else 0.0

    def add(self, entry: dict[str, Any]) -> None:
        for k in ("records", "segments", "skipped", "rejected", "errors", "tokens", "text_bytes"):
            setattr(self, k, getattr(self, k) + int(entry["stats"][k]))
        self.worker_cpu_s += float(entry["timing"]["cpu_s"])
        self.committed += 1

    def summary(self) -> str:
        return (f"{self.source}: {self.committed} shards committed ({self.already_done} already done, "
                f"{len(self.failed)} failed) {self.tokens / 1e6:.2f}M tokens, {self.segments} segments, "
                f"{self.tokens_per_s / 1e6:.2f}M tok/s wall, {self.tokens_per_cpu_s / 1e6:.2f}M tok/s/core, "
                f"{self.bytes_per_token:.2f} bytes/token")


def tok_dir(source: str | sources.Source, root: str | Path | None = None) -> Path:
    """Shard directory ``<root>/tok/<source>``."""
    root = Path(root) if root is not None else download.default_root()
    return root / TOK_DIR / sources.get(source).name


def run(source: str | sources.Source, root: str | Path | None = None, *,
        files: Sequence[download.LocalFile] | None = None, out_dir: str | Path | None = None,
        workers: int = 2, tokenizer: Any = DEFAULT_TOKENIZER, max_records: int | None = None,
        batch_rows: int = 1024, start_method: str = "spawn", strict: bool = True,
        log: Callable[[str], None] = _log_stderr,
        on_commit: Callable[[dict[str, Any]], None] | None = None) -> RunStats:
    """Tokenise every not-yet-committed input file of ``source``.

    ``files`` defaults to ``download.completed_files(source, root)``; ``out_dir`` to
    ``tok_dir(source, root)``. ``workers=0`` runs in this process. A failed file is logged and
    left uncommitted (it is retried by the next run); with ``strict`` the run raises at the end
    if any file failed. ``on_commit(entry)`` is called after every manifest append.
    """
    src = sources.get(source)
    if src.kind == "library":
        raise ValueError(f"{src.name} is raw text for the library build, not training shards")
    files = list(files) if files is not None else download.completed_files(src, root)
    out = Path(out_dir) if out_dir is not None else tok_dir(src, root)
    out.mkdir(parents=True, exist_ok=True)
    tok = resolve_tokenizer(tokenizer)
    fp, code = tokenizer_fingerprint(tok), code_fingerprint()
    done = shards.read_manifest(out)
    for e in done.values():
        if e.get("tokenizer") != fp:
            raise RuntimeError(f"{out}: shard {e['shard']} was made with tokenizer {e.get('tokenizer')}, "
                               f"now {fp}; use a fresh directory")
        if e.get("max_records") != max_records:
            raise RuntimeError(f"{out}: shard {e['shard']} was made with max_records="
                               f"{e.get('max_records')}, now {max_records}")
    if any(e.get("code") != code for e in done.values()):
        log(f"[tok] {src.name}: some committed shards were made by other code versions")
    stats = RunStats(source=src.name, files=len(files))
    seen: set[int] = set()
    jobs = []
    for f in sorted(files, key=lambda f: f.index):
        if f.index in seen:
            raise ValueError(f"two input files with index {f.index}")
        seen.add(f.index)
        if f.index in done:
            stats.already_done += 1
            continue
        shards.remove_temporaries(out, f.index)
        meta = {"repo": f.repo, "commit": f.commit, "path": f.path, "size": f.size,
                "sha256": f.sha256, "index": f.index}
        jobs.append(Job(src.name, f.index, f.local, meta, str(out), max_records, batch_rows))
    t0 = time.perf_counter()

    def commit(entry: dict[str, Any]) -> None:
        if "error" in entry:
            stats.failed.append(entry)
            log(f"[tok] {src.name} shard {entry['shard']}: FAILED {entry['error']}")
            return
        shards.append_manifest(out, entry)
        stats.add(entry)
        stats.wall_s = time.perf_counter() - t0
        log(f"[tok] {src.name} shard {entry['shard']:06d}: {entry['n_tokens'] / 1e6:.2f}M tokens, "
            f"{entry['timing']['tokens_per_cpu_s'] or 0:.0f} tok/s/core "
            f"({stats.committed + stats.already_done}/{stats.files} done)")
        if on_commit is not None:
            on_commit(entry)

    if jobs and workers <= 0:
        for job in jobs:
            try:
                entry = tokenize_file(job, tok, code=code)
            except Exception as exc:  # noqa: BLE001
                entry = {"error": f"{type(exc).__name__}: {exc}", "shard": job.shard}
            commit(entry)
    elif jobs:
        # A process pool that notices dead workers (killed by the OOM killer, or failing to start,
        # e.g. a ``spawn`` child re-running an unguarded main script): their files are reported
        # as failed instead of hanging the run, as ``multiprocessing.Pool`` would.
        ex = ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=mp.get_context(start_method),
                                 initializer=_init_worker, initargs=(tokenizer, code, os.getpid()))
        try:
            futures = {ex.submit(_work, job): job for job in jobs}
            for fut in as_completed(futures):
                try:
                    entry = fut.result()
                except BrokenProcessPool as exc:
                    entry = {"error": f"worker process died: {exc}", "shard": futures[fut].shard}
                commit(entry)
        finally:
            # on an error in the coordinator: drop queued files, let running ones finish
            # (their shards stay uncommitted and are redone by the next run)
            ex.shutdown(wait=True, cancel_futures=True)
    stats.wall_s = time.perf_counter() - t0
    log(f"[tok] {stats.summary()}")
    if strict and stats.failed:
        raise RuntimeError(f"{src.name}: {len(stats.failed)} file(s) failed: "
                           f"{[e['error'] for e in stats.failed][:3]}")
    return stats


# ------------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tokenise downloaded sources into token shards.")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--root", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="run the consistency checker afterwards")
    args = ap.parse_args(argv)
    status = 0
    for name in args.sources:
        stats = run(name, args.root, workers=args.workers, tokenizer=args.tokenizer,
                    max_records=args.max_records, strict=False)
        print(json.dumps({**asdict(stats), "tokens_per_cpu_s": stats.tokens_per_cpu_s}, default=str))
        if args.check:
            rep = shards.check(tok_dir(name, args.root))
            print(f"{name}: check ok={rep.ok} errors={rep.errors[:5]} warnings={len(rep.warnings)}")
            status |= 0 if rep.ok else 1
        status |= 1 if stats.failed else 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
