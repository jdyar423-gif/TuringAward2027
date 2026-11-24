"""Resumable downloads of individual parquet files from the Hugging Face Hub, and a
bounded-memory streaming reader.

Layout under ``root`` (default ``bes.config.RunCfg().data_dir``)::

    raw/<org>__<name>/<rev>/<path in repo>       completed files (atomic rename)
    raw/<org>__<name>/<rev>/manifest.jsonl       one JSON line per completed file
    raw/<org>__<name>/<rev>/listing.<key>.json   cached file listing and resolved commit
    raw/<org>__<name>/<rev>/.staging/            partial downloads (resumed by huggingface_hub)

``<rev>`` is the registry ``revision`` made filename-safe. Registry entries that share a
repository and revision (for example ``synth`` and ``synth_rag``) share the files.

The first listing of a source resolves its revision to a commit and is cached, so the file
list -- and therefore the index of every file, which ``tokenize_worker`` uses as the shard id --
stays fixed for the life of ``data_cache``; every download then asks for that exact commit.

A file counts as downloaded when it is at its final path with the listed size *and* has a
manifest line. Each download goes to a staging directory, is checked (size, and SHA-256 when the
Hub reports one), fsynced, renamed into place, and only then recorded. Per-file locks make
concurrent downloaders (threads or processes) safe.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bes.config import RunCfg
from bes.data import sources
from bes.data.shards import append_jsonl, atomic_write_bytes, file_lock, fsync_dir, scan_jsonl

RAW_DIR = "raw"
MANIFEST = "manifest.jsonl"
#: ``max_bytes`` value meaning "use the registry's ``cap_bytes``".
REGISTRY_CAP = "registry"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def default_root() -> Path:
    return Path(RunCfg().data_dir)


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class RemoteFile:
    """A file of a repository at the listed commit; ``sha256`` is the LFS hash when known."""

    path: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class LocalFile:
    """A downloaded file of one source. ``index`` is its position among the source's matching
    files at the cached commit, and is the shard id it becomes."""

    source: str
    index: int
    repo: str
    commit: str
    path: str
    local: str
    size: int
    sha256: str | None = None


Lister = Callable[[str, str, str], tuple[str, list[RemoteFile]]]
Fetcher = Callable[[str, str, str, Path], str | Path]


# ------------------------------------------------------------------------ Hub access
def hub_lister(repo: str, revision: str, prefix: str) -> tuple[str, list[RemoteFile]]:
    """``(commit, files under prefix)`` of a dataset repository, listed recursively."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    commit = revision if _SHA_RE.match(revision) else api.dataset_info(repo, revision=revision).sha
    out = []
    for e in api.list_repo_tree(repo, path_in_repo=prefix or None, recursive=True,
                                repo_type="dataset", revision=commit):
        if isinstance(e, RepoFile):
            out.append(RemoteFile(e.path, int(e.size), e.lfs.sha256 if e.lfs else None))
    return commit, out


def hub_fetch(repo: str, path: str, commit: str, staging: Path) -> Path:
    """Download one file into ``staging`` (huggingface_hub resumes a partial download)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset",
                                revision=commit, local_dir=str(staging)))


class CorruptDownload(Exception):
    """Size or checksum mismatch of a downloaded file (retried)."""


def is_permanent(exc: BaseException) -> bool:
    """True for errors that a retry cannot fix (missing repository, revision or file)."""
    try:
        from huggingface_hub import errors as hf_errors
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, hf_errors.LocalEntryNotFoundError):   # offline / connection failure
        return False
    return isinstance(exc, (hf_errors.EntryNotFoundError, hf_errors.RepositoryNotFoundError,
                            hf_errors.RevisionNotFoundError, hf_errors.DisabledRepoError))


def retry(fn: Callable[[], Any], *, what: str, retries: int = 6, backoff: float = 2.0,
          max_delay: float = 60.0, log: Callable[[str], None] = _log_stderr,
          sleep: Callable[[float], None] = time.sleep) -> Any:
    """Call ``fn`` up to ``retries`` times with exponential backoff (jittered, capped)."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - every failure is either permanent or retried
            if is_permanent(exc) or attempt == retries - 1:
                raise
            delay = min(backoff * 2 ** attempt, max_delay) * (0.5 + 0.5 * random.random())
            log(f"[download] {what}: {type(exc).__name__}: {str(exc)[:200]}; "
                f"retry {attempt + 1}/{retries - 1} in {delay:.1f}s")
            sleep(delay)
    raise RuntimeError("unreachable")


# -------------------------------------------------------------------------- listing
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "root"


def repo_dir(source: str | sources.Source, root: str | Path | None = None) -> Path:
    """Directory of the raw files of the source's repository and revision."""
    src = sources.get(source)
    root = Path(root) if root is not None else default_root()
    return root / RAW_DIR / src.repo.replace("/", "__") / _slug(src.revision)


def static_prefix(pattern: str) -> str:
    """Directory part of ``pattern`` before its first wildcard (``""`` = repository root)."""
    head = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
    return head.rsplit("/", 1)[0] if "/" in head else ""


def get_listing(repo: str, revision: str, prefix: str, directory: Path, *,
                lister: Lister | None = None, refresh: bool = False, retries: int = 6,
                log: Callable[[str], None] = _log_stderr,
                sleep: Callable[[float], None] = time.sleep,
                fallback: str = "main") -> tuple[str, list[RemoteFile]]:
    """Cached ``(commit, files)`` of ``repo`` at ``revision`` under ``prefix``.

    The cache ``listing.<prefix>.json`` is written atomically on the first call; later calls
    read it (no network) unless ``refresh``. If a pinned commit no longer exists on the Hub,
    ``fallback`` (default: the default branch) is listed instead and a warning is logged.
    """
    cache = directory / f"listing.{_slug(prefix)}.json"
    if cache.exists() and not refresh:
        d = json.loads(cache.read_text())
        return d["commit"], [RemoteFile(**f) for f in d["files"]]
    lister = lister or hub_lister

    def call(rev: str) -> tuple[str, list[RemoteFile]]:
        return retry(lambda: lister(repo, rev, prefix), what=f"list {repo}@{rev}/{prefix}",
                     retries=retries, log=log, sleep=sleep)

    try:
        commit, files = call(revision)
    except Exception as exc:  # noqa: BLE001
        from huggingface_hub import errors as hf_errors

        if not (isinstance(exc, hf_errors.RevisionNotFoundError) and _SHA_RE.match(revision)):
            raise
        log(f"[download] {repo}: pinned commit {revision} not found; listing {fallback}")
        commit, files = call(fallback)
    files = sorted(files, key=lambda f: f.path)
    blob = {"repo": repo, "revision": revision, "commit": commit, "prefix": prefix,
            "files": [asdict(f) for f in files]}
    atomic_write_bytes(cache, json.dumps(blob, indent=0).encode())
    return commit, files


def source_listing(source: str | sources.Source, root: str | Path | None = None, *,
                   lister: Lister | None = None, refresh: bool = False, retries: int = 6,
                   log: Callable[[str], None] = _log_stderr,
                   sleep: Callable[[float], None] = time.sleep) -> tuple[str, list[RemoteFile]]:
    """``(commit, files)``: the source's files (matching its patterns), sorted by path.

    The position of a file in this list is its shard id.
    """
    src = sources.get(source)
    rdir = repo_dir(src, root)
    # a ref (refs/convert/parquet) can move between the machines of one run, and only the cached
    # listing of a machine pins it: list the commit the registry verified when there is one
    rev = src.revision
    if not _SHA_RE.match(rev) and _SHA_RE.match(src.verified_sha or ""):
        rev = src.verified_sha
    commits, matched = set(), {}
    for prefix in sorted({static_prefix(p) for p in src.files}):
        commit, files = get_listing(src.repo, rev, prefix, rdir, lister=lister,
                                    refresh=refresh, retries=retries, log=log, sleep=sleep,
                                    fallback=src.revision if rev != src.revision else "main")
        commits.add(commit)
        for f in files:
            if any(fnmatch.fnmatchcase(f.path, pat) for pat in src.files):
                matched[f.path] = f
    if len(commits) > 1:
        raise RuntimeError(f"{src.name}: cached listings disagree on the commit {sorted(commits)}")
    return commits.pop(), [matched[p] for p in sorted(matched)]


def select_files(files: Sequence[RemoteFile], max_files: int | None = None,
                 max_bytes: int | None = None) -> list[RemoteFile]:
    """Longest prefix of ``files`` with at most ``max_files`` files and ``max_bytes`` bytes.

    The first file is always included (unless ``max_files == 0``), even when it alone exceeds
    ``max_bytes``, so a capped source never ends up empty.
    """
    out: list[RemoteFile] = []
    total = 0
    for f in files:
        if max_files is not None and len(out) >= max_files:
            break
        if max_bytes is not None and out and total + f.size > max_bytes:
            break
        out.append(f)
        total += f.size
    return out


# -------------------------------------------------------------------------- download
def _manifest_map(directory: Path) -> dict[str, dict[str, Any]]:
    entries, _ = scan_jsonl(directory / MANIFEST)
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        if isinstance(e.get("path"), str):
            out.setdefault(e["path"], e)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, rf: RemoteFile, verify: bool) -> bool:
    if not path.exists() or path.stat().st_size != rf.size:
        return False
    return not (verify and rf.sha256) or _sha256(path) == rf.sha256


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _record(directory: Path, repo: str, commit: str, rf: RemoteFile) -> dict[str, Any]:
    entry = {"repo": repo, "commit": commit, "path": rf.path, "size": rf.size,
             "sha256": rf.sha256, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    append_jsonl(directory / MANIFEST, entry)
    return entry


def is_complete(directory: Path, rf: RemoteFile) -> bool:
    """True when ``rf`` is recorded in the manifest and present with the right size."""
    final = directory / rf.path
    return rf.path in _manifest_map(directory) and final.exists() and final.stat().st_size == rf.size


def download_file(repo: str, commit: str, rf: RemoteFile, directory: Path, *,
                  fetch: Fetcher | None = None, retries: int = 6, backoff: float = 2.0,
                  verify: bool = True, log: Callable[[str], None] = _log_stderr,
                  sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Make ``directory/rf.path`` a complete, recorded copy of ``rf``; returns its manifest entry.

    Idempotent and safe to interrupt at any point: a file renamed into place but not yet
    recorded is verified and recorded on the next call; a partial download in the staging
    directory is resumed by huggingface_hub.
    """
    fetch = fetch or hub_fetch
    final = directory / rf.path
    if is_complete(directory, rf):
        return _manifest_map(directory)[rf.path]
    lock = directory / ".locks" / (hashlib.sha1(rf.path.encode()).hexdigest() + ".lock")
    with file_lock(lock):
        if is_complete(directory, rf):                 # finished by a concurrent downloader
            return _manifest_map(directory)[rf.path]
        if final.exists():
            if _matches(final, rf, verify):            # renamed before a crash, not recorded
                return _record(directory, repo, commit, rf)
            final.unlink()
        staging = directory / ".staging"

        def attempt() -> dict[str, Any]:
            tmp = Path(fetch(repo, rf.path, commit, staging))
            if not _matches(tmp, rf, verify):
                tmp.unlink(missing_ok=True)
                raise CorruptDownload(f"{rf.path}: size or SHA-256 differs from the listing")
            _fsync_file(tmp)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, final)
            fsync_dir(final.parent)
            return _record(directory, repo, commit, rf)

        return retry(attempt, what=f"{repo}/{rf.path}", retries=retries, backoff=backoff,
                     log=log, sleep=sleep)


def download_source(source: str | sources.Source, root: str | Path | None = None, *,
                    max_files: int | None = None, max_bytes: int | str | None = REGISTRY_CAP,
                    threads: int = 1, fetch: Fetcher | None = None, lister: Lister | None = None,
                    retries: int = 6, backoff: float = 2.0, verify: bool = True,
                    log: Callable[[str], None] = _log_stderr,
                    sleep: Callable[[float], None] = time.sleep) -> list[LocalFile]:
    """Download the first files of a source (in listing order) up to the caps.

    ``max_bytes`` defaults to the registry's ``cap_bytes`` (``None`` = no cap). Returns the
    ``LocalFile`` of every selected file, all complete when the call returns.
    """
    src = sources.get(source)
    rdir = repo_dir(src, root)
    cap = src.cap_bytes if max_bytes == REGISTRY_CAP else max_bytes
    commit, files = source_listing(src, root, lister=lister, retries=retries, log=log, sleep=sleep)
    chosen = select_files(files, max_files, cap)
    index = {f.path: i for i, f in enumerate(files)}

    def one(rf: RemoteFile) -> LocalFile:
        download_file(src.repo, commit, rf, rdir, fetch=fetch, retries=retries,
                      backoff=backoff, verify=verify, log=log, sleep=sleep)
        return LocalFile(src.name, index[rf.path], src.repo, commit, rf.path,
                         str((rdir / rf.path).resolve()), rf.size, rf.sha256)

    if threads <= 1:
        return [one(rf) for rf in chosen]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        return list(ex.map(one, chosen))


def completed_files(source: str | sources.Source, root: str | Path | None = None, *,
                    lister: Lister | None = None) -> list[LocalFile]:
    """Every downloaded file of the source (recorded and present), in shard-id order."""
    src = sources.get(source)
    rdir = repo_dir(src, root)
    commit, files = source_listing(src, root, lister=lister)
    done = _manifest_map(rdir)
    out = []
    for i, f in enumerate(files):
        p = rdir / f.path
        if f.path in done and p.exists() and p.stat().st_size == f.size:
            out.append(LocalFile(src.name, i, src.repo, done[f.path].get("commit", commit), f.path,
                                 str(p.resolve()), f.size, f.sha256))
    return out


# --------------------------------------------------------------------------- reading
def iter_batches(path: str | Path, columns: Sequence[str] | None = None, *,
                 batch_rows: int = 1024, buffer_bytes: int = 4 << 20) -> Iterator[Any]:
    """``pyarrow.RecordBatch`` es of at most ``batch_rows`` rows, in file order.

    Memory stays bounded even for files made of one huge row group (PleIAs/SYNTH stores ~795 MB
    per row group): column chunks are read through a ``buffer_bytes`` stream instead of being
    pre-buffered, decoding is single-threaded, and only ``columns`` are decoded.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path), buffer_size=buffer_bytes, pre_buffer=False)
    try:
        names = pf.schema_arrow.names
        if columns is not None:
            missing = [c for c in columns if c not in names]
            if missing:
                raise KeyError(f"{path}: missing columns {missing}; the file has {names}")
        yield from pf.iter_batches(batch_size=batch_rows, columns=list(columns) if columns else None,
                                   use_threads=False)
    finally:
        pf.close()


def iter_records(path: str | Path, columns: Sequence[str] | None = None, *,
                 batch_rows: int = 1024, buffer_bytes: int = 4 << 20) -> Iterator[list[dict[str, Any]]]:
    """Rows of a parquet file as lists of dicts, one list per batch (see ``iter_batches``)."""
    for batch in iter_batches(path, columns, batch_rows=batch_rows, buffer_bytes=buffer_bytes):
        yield batch.to_pylist()


def num_rows(path: str | Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(str(path)).metadata.num_rows


# ------------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download registry sources from the Hugging Face Hub.")
    ap.add_argument("sources", nargs="*", help="registry names (default: none; see --list)")
    ap.add_argument("--root", default=None, help="data directory (default: RunCfg.data_dir)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-bytes", type=float, default=None, help="cap in bytes (default: registry cap)")
    ap.add_argument("--no-cap", action="store_true", help="ignore the registry cap")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    args = ap.parse_args(argv)
    if args.list or not args.sources:
        for s in sources.SOURCES:
            flag = " NONPERMISSIVE" if s.nonpermissive else ""
            print(f"{s.name:24s} {s.kind:8s} w={s.weight:<5} {s.repo}{flag}")
        return 0
    cap: int | str | None = REGISTRY_CAP
    if args.no_cap:
        cap = None
    elif args.max_bytes is not None:
        cap = int(args.max_bytes)
    for name in args.sources:
        got = download_source(name, args.root, max_files=args.max_files, max_bytes=cap,
                              threads=args.threads)
        print(f"{name}: {len(got)} files, {sum(f.size for f in got) / 1e9:.3f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
