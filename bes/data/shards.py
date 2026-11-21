"""Token shards (``docs/SPEC_MODEL_TRAIN.md`` section 6): writer, memory-mapped reader, checker.

A source's shards live in ``tok/<source>/``. Shard ``n`` (``stem = f"{n:06d}"``) is

- ``<stem>.u16``  : ``uint16[n_tokens]`` little-endian token ids, the concatenation of its
  segments; every segment starts with ``BOS`` and ``BOS`` occurs nowhere else;
- ``<stem>.idx``  : ``int64[n_segments + 1]`` segment offsets: segment ``i`` is
  ``tokens[idx[i]:idx[i+1]]`` (``idx[0] == 0``, ``idx[-1] == n_tokens``, strictly increasing);
- ``<stem>.role`` : optional ``int8[n_tokens]``, the role code of every token as produced by
  ``chat_format`` (codes of ``REG_FIELDS["role"]``: 0 raw text, 1 system, 2 user, 3 assistant,
  4 library); present for conversations, absent for raw documents;
- ``<stem>.rec``  : optional ``int64[n_segments]``, the row of the input file each segment came
  from (provenance; a preference pair gives two segments with the same row).

``manifest.jsonl`` holds one JSON line per *committed* shard. A shard is committed only after
all of its files were written to temporary names, fsynced and renamed into place (and the
directory fsynced); the manifest line is appended (and fsynced) after that, by a single writer.
Files that no manifest line refers to are leftovers of an interrupted run and are ignored by the
reader (the checker lists them as warnings). A torn last manifest line (crash during the append)
is ignored by readers and cut off by the next append.
"""
from __future__ import annotations

import bisect
import contextlib
import fcntl
import json
import os
import re
import uuid
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import BOS, REG_FIELDS, VOCAB_SIZE

SHARD_FORMAT = "bes-shard/1"
TOKEN_DTYPE = np.dtype("<u2")
OFFSET_DTYPE = np.dtype("<i8")
ROLE_DTYPE = np.dtype("i1")
REC_DTYPE = np.dtype("<i8")
EXT = {"tokens": ".u16", "offsets": ".idx", "roles": ".role", "recs": ".rec"}
MANIFEST = "manifest.jsonl"
N_ROLE_CODES = dict(REG_FIELDS)["role"]
_TMP_MARK = ".tmp."
_STEM_RE = re.compile(r"^(\d{6,})\.(u16|idx|role|rec)$")
_CHUNK = 1 << 24  # tokens per chunk in the deep checks (32 MB of uint16)


def stem(shard: int) -> str:
    """File stem of shard ``n``: zero-padded to six digits."""
    if shard < 0:
        raise ValueError("shard ids are non-negative")
    return f"{shard:06d}"


# ------------------------------------------------------------------ durable file helpers
def fsync_dir(path: str | Path) -> None:
    """fsync a directory so that renames inside it are durable."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def file_lock(path: str | Path) -> Iterator[None]:
    """Exclusive advisory lock (``flock``) on ``path`` (created if missing) for the block."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file, fsync and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_MARK + uuid.uuid4().hex[:12])
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    """Append one JSON line to ``path`` and fsync it.

    Under an exclusive lock, a torn last line (no trailing newline, left by a crash during an
    earlier append) is first cut off, so the file always consists of complete lines.
    """
    path = Path(path)
    line = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with file_lock(path.with_name(path.name + ".lock")):
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size:
                os.lseek(fd, size - 1, os.SEEK_SET)
                if os.read(fd, 1) != b"\n":
                    _truncate_torn_tail(fd, size)
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)


def _truncate_torn_tail(fd: int, size: int) -> None:
    keep, pos = 0, size
    while pos > 0 and keep == 0:
        step = min(1 << 16, pos)
        pos -= step
        os.lseek(fd, pos, os.SEEK_SET)
        block = os.read(fd, step)
        nl = block.rfind(b"\n")
        if nl >= 0:
            keep = pos + nl + 1
    os.ftruncate(fd, keep)


def scan_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a JSON-lines file: ``(objects, problems)``.

    A last line without newline is a torn append and is skipped silently; any other line that
    is not a JSON object is reported in ``problems``.
    """
    path = Path(path)
    if not path.exists():
        return [], []
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        lines.pop()          # torn tail
    out, problems = [], []
    for k, raw in enumerate(lines):
        try:
            obj = json.loads(raw)
        except ValueError:
            problems.append(f"{path.name}: line {k + 1} is not valid JSON")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{path.name}: line {k + 1} is not a JSON object")
            continue
        out.append(obj)
    return out, problems


def read_manifest(directory: str | Path) -> dict[int, dict[str, Any]]:
    """Committed shards of ``directory``: ``{shard_id: manifest entry}`` (first entry wins)."""
    entries, _ = scan_jsonl(Path(directory) / MANIFEST)
    out: dict[int, dict[str, Any]] = {}
    for e in entries:
        if isinstance(e.get("shard"), int) and e["shard"] not in out:
            out[e["shard"]] = e
    return out


def append_manifest(directory: str | Path, entry: dict[str, Any]) -> None:
    """Commit a shard whose files ``ShardWriter.close`` has already put in place."""
    append_jsonl(Path(directory) / MANIFEST, entry)


# ------------------------------------------------------------------------------ writer
class ShardWriter:
    """Stream segments into one shard; ``close()`` makes the files durable and returns the
    manifest entry (the caller appends it with ``append_manifest``).

    Temporary files carry a unique suffix, so two writers of the same shard (for instance an
    orphaned worker of a killed run and its replacement) never write into the same file; the
    content is a pure function of the input, so whichever rename lands last is correct.
    """

    def __init__(self, directory: str | Path, shard: int, *, with_roles: bool,
                 with_recs: bool = True, vocab_size: int = VOCAB_SIZE, bos: int = BOS,
                 buffer_bytes: int = 1 << 20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard, self.stem = shard, stem(shard)
        self.with_roles, self.with_recs = with_roles, with_recs
        self.vocab_size, self.bos = vocab_size, bos
        self._suffix = f"{_TMP_MARK}{os.getpid()}.{uuid.uuid4().hex[:8]}"
        kinds = ["tokens", "offsets"] + (["roles"] if with_roles else []) + (["recs"] if with_recs else [])
        self._tmp = {k: self.directory / (self.stem + EXT[k] + self._suffix) for k in kinds}
        self._tok_f = open(self._tmp["tokens"], "wb", buffering=buffer_bytes)
        self._role_f = open(self._tmp["roles"], "wb", buffering=buffer_bytes) if with_roles else None
        self._crc = {"tokens": 0, "roles": 0}
        self._offsets: list[int] = [0]
        self._recs: list[int] = []
        self._closed = False

    @property
    def n_tokens(self) -> int:
        return self._offsets[-1]

    @property
    def n_segments(self) -> int:
        return len(self._offsets) - 1

    def add(self, ids: Sequence[int] | np.ndarray, roles: Sequence[int] | np.ndarray | None = None,
            rec: int = -1) -> None:
        """Append one segment: ``ids`` (int, ``[n]``, ``ids[0] == BOS``, no other ``BOS``, all in
        ``[0, vocab_size)``) and, for a shard with roles, ``roles`` (int, ``[n]``, codes in
        ``[0, N_ROLE_CODES)``). ``rec`` is the input row (``-1`` if unknown)."""
        if self._closed:
            raise RuntimeError("writer is closed")
        a = np.asarray(ids)
        if a.ndim != 1 or a.size == 0:
            raise ValueError("a segment is a non-empty 1-d sequence of ids")
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"token ids must be integers, got {a.dtype}")
        if int(a[0]) != self.bos:
            raise ValueError(f"segment must start with BOS={self.bos}, got {int(a[0])}")
        if int(a.min()) < 0 or int(a.max()) >= self.vocab_size:
            raise ValueError(f"token id outside [0, {self.vocab_size})")
        if a.size > 1 and bool((a[1:] == self.bos).any()):
            raise ValueError("BOS inside a segment")
        if self.with_roles:
            if roles is None:
                raise ValueError("this shard stores roles; pass roles=")
            r = np.asarray(roles)
            if r.shape != a.shape:
                raise ValueError(f"roles shape {r.shape} != ids shape {a.shape}")
            if not np.issubdtype(r.dtype, np.integer) or int(r.min()) < 0 or int(r.max()) >= N_ROLE_CODES:
                raise ValueError(f"role codes must be integers in [0, {N_ROLE_CODES})")
            rb = r.astype(ROLE_DTYPE).tobytes()
            self._crc["roles"] = zlib.crc32(rb, self._crc["roles"])
            self._role_f.write(rb)
        elif roles is not None:
            raise ValueError("this shard was opened with with_roles=False")
        tb = a.astype(TOKEN_DTYPE).tobytes()
        self._crc["tokens"] = zlib.crc32(tb, self._crc["tokens"])
        self._tok_f.write(tb)
        self._offsets.append(self._offsets[-1] + int(a.size))
        self._recs.append(int(rec))

    def close(self) -> dict[str, Any]:
        """fsync, rename into place, fsync the directory; return the manifest entry fields."""
        if self._closed:
            raise RuntimeError("writer is closed")
        files: dict[str, dict[str, int]] = {}
        for f in (self._tok_f, self._role_f):
            if f is not None:
                f.flush()
                os.fsync(f.fileno())
                f.close()
        files["tokens"] = {"bytes": self.n_tokens * TOKEN_DTYPE.itemsize, "crc32": self._crc["tokens"]}
        if self.with_roles:
            files["roles"] = {"bytes": self.n_tokens * ROLE_DTYPE.itemsize, "crc32": self._crc["roles"]}
        blobs = {"offsets": np.asarray(self._offsets, dtype=OFFSET_DTYPE).tobytes()}
        if self.with_recs:
            blobs["recs"] = np.asarray(self._recs, dtype=REC_DTYPE).tobytes()
        for kind, blob in blobs.items():
            with open(self._tmp[kind], "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            files[kind] = {"bytes": len(blob), "crc32": zlib.crc32(blob)}
        for kind, tmp in self._tmp.items():
            os.replace(tmp, self.directory / (self.stem + EXT[kind]))
        fsync_dir(self.directory)
        self._closed = True
        return {
            "format": SHARD_FORMAT, "shard": self.shard, "n_tokens": self.n_tokens,
            "n_segments": self.n_segments, "has_roles": self.with_roles,
            "has_recs": self.with_recs, "files": {EXT[k][1:]: v for k, v in files.items()},
        }

    def abort(self) -> None:
        """Close and delete the temporary files (nothing is renamed)."""
        if self._closed:
            return
        for f in (self._tok_f, self._role_f):
            if f is not None and not f.closed:
                f.close()
        for tmp in self._tmp.values():
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        self._closed = True

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


def remove_temporaries(directory: str | Path, shard: int | None = None) -> int:
    """Delete leftover temporary files (of ``shard`` only, when given); returns how many."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    prefix = stem(shard) + "." if shard is not None else ""
    n = 0
    for p in directory.iterdir():
        if _TMP_MARK in p.name and p.name.startswith(prefix) and not p.name.endswith(".lock"):
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
                n += 1
    return n


# ------------------------------------------------------------------------------ reader
def _map(path: Path, dtype: np.dtype, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(count,))


@dataclass(eq=False)
class Shard:
    """One committed shard, memory-mapped. ``tokens``: ``uint16[n_tokens]``; ``offsets``:
    ``int64[n_segments+1]``; ``roles``: ``int8[n_tokens]`` or None; ``recs``: ``int64[n_segments]``
    or None."""

    directory: Path
    entry: dict[str, Any]
    tokens: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    roles: np.ndarray | None = field(repr=False, default=None)
    recs: np.ndarray | None = field(repr=False, default=None)

    @property
    def shard(self) -> int:
        return int(self.entry["shard"])

    @property
    def n_tokens(self) -> int:
        return int(self.entry["n_tokens"])

    def __len__(self) -> int:
        return int(self.entry["n_segments"])

    def segment(self, i: int) -> np.ndarray:
        """Token ids of segment ``i`` (a read-only ``uint16`` view)."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def segment_roles(self, i: int) -> np.ndarray | None:
        if not 0 <= i < len(self):
            raise IndexError(i)
        return None if self.roles is None else self.roles[self.offsets[i]:self.offsets[i + 1]]

    def segment_lengths(self) -> np.ndarray:
        """``int64[n_segments]`` length of every segment."""
        return np.diff(self.offsets)


def open_shard(directory: str | Path, entry: dict[str, Any]) -> Shard:
    """Memory-map the files of a committed shard (sizes are checked against ``entry``)."""
    directory = Path(directory)
    s = stem(int(entry["shard"]))
    n_tok, n_seg = int(entry["n_tokens"]), int(entry["n_segments"])
    paths = {k: directory / (s + EXT[k]) for k in EXT}
    expect = {"tokens": n_tok * TOKEN_DTYPE.itemsize, "offsets": (n_seg + 1) * OFFSET_DTYPE.itemsize}
    if entry.get("has_roles"):
        expect["roles"] = n_tok * ROLE_DTYPE.itemsize
    if entry.get("has_recs"):
        expect["recs"] = n_seg * REC_DTYPE.itemsize
    for k, size in expect.items():
        actual = paths[k].stat().st_size
        if actual != size:
            raise ValueError(f"{paths[k]}: {actual} bytes, manifest implies {size}")
    offsets = np.fromfile(paths["offsets"], dtype=OFFSET_DTYPE)
    return Shard(
        directory=directory, entry=entry,
        tokens=_map(paths["tokens"], TOKEN_DTYPE, n_tok), offsets=offsets,
        roles=_map(paths["roles"], ROLE_DTYPE, n_tok) if entry.get("has_roles") else None,
        recs=np.fromfile(paths["recs"], dtype=REC_DTYPE) if entry.get("has_recs") else None,
    )


class ShardSet:
    """All committed shards of one directory (ascending shard id) with global segment indexing.

    Segment ``j`` of the set is segment ``j - start[k]`` of the ``k``-th shard. The set is a
    snapshot of the manifest at construction; call ``ShardSet(directory)`` again to see shards
    committed later.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest = read_manifest(self.directory)
        self.shards: list[Shard] = [open_shard(self.directory, manifest[k]) for k in sorted(manifest)]
        self.starts: list[int] = []
        total = 0
        for sh in self.shards:
            self.starts.append(total)
            total += len(sh)
        self.n_segments = total
        self.n_tokens = sum(sh.n_tokens for sh in self.shards)

    def __len__(self) -> int:
        return self.n_segments

    def locate(self, j: int) -> tuple[int, int]:
        """``(shard position, segment index within the shard)`` of global segment ``j``."""
        if not 0 <= j < self.n_segments:
            raise IndexError(j)
        k = bisect.bisect_right(self.starts, j) - 1
        return k, j - self.starts[k]

    def segment(self, j: int) -> np.ndarray:
        k, i = self.locate(j)
        return self.shards[k].segment(i)

    def segment_roles(self, j: int) -> np.ndarray | None:
        k, i = self.locate(j)
        return self.shards[k].segment_roles(i)


# ----------------------------------------------------------------------------- checker
@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_shards: int = 0
    n_segments: int = 0
    n_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _crc_file(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            crc = zlib.crc32(block, crc)
    return crc


def _check_tokens(sh: Shard, vocab_size: int, bos: int, errors: list[str], name: str) -> None:
    n = sh.n_tokens
    bos_positions = []
    for a in range(0, n, _CHUNK):
        chunk = np.asarray(sh.tokens[a:a + _CHUNK])
        if chunk.size and int(chunk.max()) >= vocab_size:
            errors.append(f"{name}: token id >= vocab_size {vocab_size}")
            return
        bos_positions.append(a + np.flatnonzero(chunk == bos))
    pos = np.concatenate(bos_positions) if bos_positions else np.zeros(0, dtype=np.int64)
    starts = sh.offsets[:-1]
    if pos.size != starts.size or not np.array_equal(pos, starts):
        errors.append(f"{name}: BOS positions do not match the segment offsets "
                      f"({pos.size} BOS for {starts.size} segments)")
    if sh.roles is not None:
        for a in range(0, n, _CHUNK):
            r = np.asarray(sh.roles[a:a + _CHUNK])
            if r.size and (int(r.min()) < 0 or int(r.max()) >= N_ROLE_CODES):
                errors.append(f"{name}: role code outside [0, {N_ROLE_CODES})")
                return


def check(directory: str | Path, *, deep: bool = True, vocab_size: int = VOCAB_SIZE,
          bos: int = BOS) -> CheckReport:
    """Verify the shards of ``directory`` against their manifest.

    Always: manifest lines parse, no shard id is committed twice, every referenced file exists
    with the size the entry implies, offsets start at 0, increase strictly and end at
    ``n_tokens``, a preference-pair shard has an even number of segments. With ``deep``: CRC-32
    of every file, all ids ``< vocab_size``, ``BOS`` exactly at the segment starts, role codes in
    range. Files that no entry refers to (temporaries, uncommitted shards) are warnings.
    """
    directory = Path(directory)
    rep = CheckReport()
    entries, problems = scan_jsonl(directory / MANIFEST)
    rep.errors += problems
    seen: dict[int, dict[str, Any]] = {}
    for e in entries:
        sid = e.get("shard")
        if not isinstance(sid, int):
            rep.errors.append(f"manifest entry without an integer shard id: {str(e)[:80]}")
            continue
        if sid in seen:
            rep.errors.append(f"shard {sid} committed more than once")
            continue
        seen[sid] = e
    for sid in sorted(seen):
        e = seen[sid]
        name = stem(sid)
        if e.get("format") != SHARD_FORMAT:
            rep.errors.append(f"{name}: unknown format {e.get('format')!r}")
            continue
        try:
            sh = open_shard(directory, e)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rep.errors.append(f"{name}: {exc}")
            continue
        off = sh.offsets
        increasing = off.size < 2 or bool((np.diff(off) > 0).all())
        if off.size != len(sh) + 1 or off[0] != 0 or off[-1] != sh.n_tokens or not increasing:
            rep.errors.append(f"{name}: offsets are not 0 < ... < n_tokens strictly increasing")
            continue
        if e.get("pairs") and len(sh) % 2:
            rep.errors.append(f"{name}: preference shard with an odd number of segments")
        if deep:
            for key, info in (e.get("files") or {}).items():
                path = directory / (name + "." + key)
                if _crc_file(path) != info.get("crc32"):
                    rep.errors.append(f"{name}.{key}: CRC-32 mismatch")
            _check_tokens(sh, vocab_size, bos, rep.errors, name)
        rep.n_shards += 1
        rep.n_segments += len(sh)
        rep.n_tokens += sh.n_tokens
    committed = {stem(s) for s in seen}
    for p in sorted(directory.iterdir()) if directory.exists() else []:
        if p.name in (MANIFEST, MANIFEST + ".lock"):
            continue
        m = _STEM_RE.match(p.name)
        if m is None or m.group(1) not in committed:
            rep.warnings.append(f"not committed: {p.name}")
    return rep
