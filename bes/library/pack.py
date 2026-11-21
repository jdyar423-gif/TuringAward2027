"""Packing of passages into independent xz (raw LZMA2) blocks with a compact directory.

Stored layout (every array below is shipped and counted; one stored number = 2 bytes, an
array of ``nbytes`` bytes counts ``ceil(nbytes / 2)`` numbers):

text blocks
    ``blob``        uint8  [C]        concatenated compressed blocks (raw LZMA2 streams)
    ``block_off``   uint32 [B + 1]    byte offset of each compressed block in ``blob``
    ``block_raw``   uint32 [B]        decompressed length of each block (<= ``block_raw_bytes``)
    ``block_crc``   uint32 [B]        CRC-32 of each decompressed block
    ``block_first`` uint32 [B + 1]    id of the first passage of each block (last = P)
passage directory
    ``pas_off``     uint16 [P]        byte offset of each passage inside its decompressed block
    ``title_first`` uint32 [T + 1]    id of the first passage of each title (last = P)
title list (titles joined by ``\\n``, ``titles_per_block`` titles per block)
    ``tblob``, ``tblock_off`` [TB + 1], ``tblock_raw`` [TB], ``tblock_crc`` [TB]

Passages are stored in id order, grouped by title, and never straddle a block, so the
directory entry ``(block, offset, length, title id)`` of passage ``p`` is implicit:
``block = max{b : block_first[b] <= p}``, ``offset = pas_off[p]``, ``length`` = next
offset in the same block (or ``block_raw[block]``) minus ``offset``, and
``title = max{t : title_first[t] <= p}``. Reading one passage reads exactly one text block;
rendering its title reads one title block (a few kB).
"""
from __future__ import annotations

import dataclasses
import json
import lzma
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from bes.config import LibraryCfg


class CorruptBlockError(ValueError):
    """A block failed to decompress or its CRC-32 does not match."""


@dataclass(frozen=True)
class PackCfg:
    block_raw_bytes: int = LibraryCfg().block_raw_bytes  # max decompressed bytes per block
    preset: int = 6                # LZMA2 preset; 6 compresses 64 kB blocks as well as 9e
    lc: int = 3
    lp: int = 0
    pb: int = 0                    # text is byte aligned: pb=0 is 0.3% smaller than pb=2
    titles_per_block: int = 256
    threads: int = 2

    def filters(self) -> list[dict]:
        dict_size = max(4096, 1 << max(12, math.ceil(math.log2(max(self.block_raw_bytes, 4096)))))
        return [{"id": lzma.FILTER_LZMA2, "preset": self.preset, "dict_size": dict_size,
                 "lc": self.lc, "lp": self.lp, "pb": self.pb}]


def numbers_of(nbytes: int) -> int:
    """Stored numbers needed for ``nbytes`` bytes (one number = 2 bytes)."""
    return (int(nbytes) + 1) // 2


def compress_block(raw: bytes, cfg: PackCfg) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=cfg.filters())


def decompress_block(comp: bytes, raw_len: int, crc: int, cfg: PackCfg) -> bytes:
    """Decompress one block and verify its length and CRC-32 (raises CorruptBlockError)."""
    try:
        d = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=cfg.filters())
        raw = d.decompress(bytes(comp), max_length=raw_len + 1)
    except lzma.LZMAError as e:
        raise CorruptBlockError(f"lzma error: {e}") from e
    if len(raw) != raw_len:
        raise CorruptBlockError(f"length {len(raw)} != {raw_len}")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != int(crc):
        raise CorruptBlockError("crc mismatch")
    return raw


def _compress_all(raws: Sequence[bytes], cfg: PackCfg) -> list[bytes]:
    if cfg.threads <= 1 or len(raws) < 4:
        return [compress_block(r, cfg) for r in raws]
    with ThreadPoolExecutor(max_workers=cfg.threads) as ex:  # lzma releases the GIL
        return list(ex.map(lambda r: compress_block(r, cfg), raws, chunksize=8))


def _blocks_to_arrays(raws: Sequence[bytes], cfg: PackCfg) -> dict[str, np.ndarray]:
    comps = _compress_all(raws, cfg)
    off = np.zeros(len(comps) + 1, dtype=np.uint32)
    if comps:
        off[1:] = np.cumsum([len(c) for c in comps], dtype=np.uint64).astype(np.uint32)
    blob = np.frombuffer(b"".join(comps), dtype=np.uint8).copy()
    raw_len = np.array([len(r) for r in raws], dtype=np.uint32)
    crc = np.array([zlib.crc32(r) & 0xFFFFFFFF for r in raws], dtype=np.uint32)
    return {"blob": blob, "off": off, "raw": raw_len, "crc": crc}


@dataclass
class Packed:
    """In-memory (or memory-mapped) packed library. See the module docstring for fields."""

    cfg: PackCfg
    blob: np.ndarray
    block_off: np.ndarray
    block_raw: np.ndarray
    block_crc: np.ndarray
    block_first: np.ndarray
    pas_off: np.ndarray
    title_first: np.ndarray
    tblob: np.ndarray
    tblock_off: np.ndarray
    tblock_raw: np.ndarray
    tblock_crc: np.ndarray

    ARRAYS = ("blob", "block_off", "block_raw", "block_crc", "block_first", "pas_off",
              "title_first", "tblob", "tblock_off", "tblock_raw", "tblock_crc")

    # ------------------------------------------------------------------ sizes
    @property
    def n_passages(self) -> int:
        return int(self.pas_off.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.block_raw.shape[0])

    @property
    def n_titles(self) -> int:
        return int(self.title_first.shape[0]) - 1

    def accounting(self) -> dict[str, int]:
        """Exact stored numbers per component (one number = 2 bytes)."""
        a = {k: numbers_of(getattr(self, k).nbytes) for k in self.ARRAYS}
        text = a["blob"] + a["block_off"] + a["block_raw"] + a["block_crc"] + a["block_first"]
        directory = a["pas_off"] + a["title_first"]
        titles = a["tblob"] + a["tblock_off"] + a["tblock_raw"] + a["tblock_crc"]
        return {
            "text_numbers": text, "directory_numbers": directory, "title_numbers": titles,
            "library_numbers": text + directory + titles, "text_blob_numbers": a["blob"],
            "raw_text_bytes": int(np.asarray(self.block_raw, dtype=np.int64).sum()),
            "n_passages": self.n_passages, "n_blocks": self.n_blocks, "n_titles": self.n_titles,
            **{f"array_{k}": v for k, v in a.items()},
        }

    # ------------------------------------------------------------------ access
    def read_block(self, b: int) -> bytes:
        lo, hi = int(self.block_off[b]), int(self.block_off[b + 1])
        return decompress_block(self.blob[lo:hi].tobytes(), int(self.block_raw[b]),
                                int(self.block_crc[b]), self.cfg)

    def block_numbers(self, b: int) -> int:
        """Stored numbers read to fetch text block ``b`` (compressed bytes + its table row)."""
        comp = int(self.block_off[b + 1]) - int(self.block_off[b])
        return numbers_of(comp) + 2 * 4  # off[b], off[b+1], raw[b], crc[b] are uint32

    def read_title_block(self, tb: int) -> list[str]:
        lo, hi = int(self.tblock_off[tb]), int(self.tblock_off[tb + 1])
        raw = decompress_block(self.tblob[lo:hi].tobytes(), int(self.tblock_raw[tb]),
                               int(self.tblock_crc[tb]), self.cfg)
        return raw.decode("utf-8").split("\n")

    def title_block_numbers(self, tb: int) -> int:
        comp = int(self.tblock_off[tb + 1]) - int(self.tblock_off[tb])
        return numbers_of(comp) + 2 * 4

    def locate(self, pid: int) -> tuple[int, int, int, list[int]]:
        """(block, start, end, probes): byte range of passage ``pid`` in its decompressed
        block; ``probes`` are the indices of ``block_first`` read by the search."""
        if not 0 <= pid < self.n_passages:
            raise IndexError(pid)
        b, probes = bsearch_right(self.block_first, pid)
        b -= 1
        start = int(self.pas_off[pid])
        if pid + 1 < int(self.block_first[b + 1]):
            end = int(self.pas_off[pid + 1])
        else:
            end = int(self.block_raw[b])
        return b, start, end, probes

    def title_of(self, pid: int) -> tuple[int, list[int]]:
        """(title id, indices of ``title_first`` probed)."""
        t, probes = bsearch_right(self.title_first, pid)
        return t - 1, probes

    def passage_body(self, pid: int) -> str:
        b, s, e, _ = self.locate(pid)
        return self.read_block(b)[s:e].decode("utf-8")

    def title(self, tid: int) -> str:
        tpb = self.cfg.titles_per_block
        return self.read_title_block(tid // tpb)[tid % tpb]

    def all_titles(self) -> list[str]:
        out: list[str] = []
        for tb in range(int(self.tblock_raw.shape[0])):
            out.extend(self.read_title_block(tb))
        return out[: self.n_titles]

    def iter_bodies(self) -> Iterable[tuple[int, str]]:
        """All (passage id, body) in id order, one block decompressed at a time."""
        pid = 0
        for b in range(self.n_blocks):
            raw = self.read_block(b)
            hi = int(self.block_first[b + 1])
            while pid < hi:
                s = int(self.pas_off[pid])
                e = int(self.pas_off[pid + 1]) if pid + 1 < hi else len(raw)
                yield pid, raw[s:e].decode("utf-8")
                pid += 1

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for k in self.ARRAYS:
            _atomic_save_npy(os.path.join(out_dir, f"pack_{k}.npy"), np.asarray(getattr(self, k)))
        meta = {"pack_cfg": dataclasses.asdict(self.cfg), "accounting": self.accounting()}
        _atomic_write(os.path.join(out_dir, "pack.json"), json.dumps(meta, indent=1).encode())

    @classmethod
    def load(cls, out_dir: str, mmap: bool = True) -> "Packed":
        meta = json.loads(open(os.path.join(out_dir, "pack.json")).read())
        cfg = PackCfg(**meta["pack_cfg"])
        arrs = {k: np.load(os.path.join(out_dir, f"pack_{k}.npy"), mmap_mode="r" if mmap else None)
                for k in cls.ARRAYS}
        return cls(cfg=cfg, **arrs)


def bsearch_right(arr: np.ndarray, x: int) -> tuple[int, list[int]]:
    """``np.searchsorted(arr, x, 'right')`` by explicit bisection; also returns the indices
    of the entries probed (for the read-set log)."""
    lo, hi = 0, int(arr.shape[0])
    probes: list[int] = []
    while lo < hi:
        mid = (lo + hi) // 2
        probes.append(mid)
        if int(arr[mid]) <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo, probes


def pack(bodies: Sequence[str], title_ids: Sequence[int], titles: Sequence[str],
         cfg: PackCfg = PackCfg()) -> Packed:
    """Pack passage bodies into blocks.

    ``title_ids[i]`` is the title of passage ``i``; it must be non-decreasing and every
    title ``0..len(titles)-1`` must own at least one passage (titles are numbered in pack
    order). Bodies are stored as UTF-8 without separators.
    """
    n = len(bodies)
    tids = np.asarray(title_ids, dtype=np.int64)
    if tids.shape[0] != n:
        raise ValueError("title_ids and bodies differ in length")
    if n and (np.any(np.diff(tids) < 0) or tids[0] != 0 or tids[-1] != len(titles) - 1
              or np.any(np.diff(tids) > 1)):
        raise ValueError("title ids must be non-decreasing, start at 0 and cover every title")
    for t in titles:
        if "\n" in t:
            raise ValueError("titles must not contain newlines")
    raws: list[bytes] = []
    first: list[int] = [0]
    offs = np.zeros(n, dtype=np.uint16)
    cur = bytearray()
    for i, body in enumerate(bodies):
        bb = body.encode("utf-8")
        if len(bb) > cfg.block_raw_bytes:
            raise ValueError(f"passage {i} longer than a block")
        if len(cur) + len(bb) > cfg.block_raw_bytes:
            raws.append(bytes(cur))
            first.append(i)
            cur = bytearray()
        if len(cur) > 0xFFFF:
            raise ValueError("block offset does not fit in uint16")
        offs[i] = len(cur)
        cur += bb
    if cur or not raws:
        raws.append(bytes(cur))
    if n == 0:
        raws = []
        first = []
    first.append(n)
    text = _blocks_to_arrays(raws, cfg)
    title_first = np.zeros(len(titles) + 1, dtype=np.uint32)
    if n:
        starts = np.flatnonzero(np.r_[True, np.diff(tids) != 0])
        title_first[:-1] = starts.astype(np.uint32)
    title_first[-1] = n
    tpb = cfg.titles_per_block
    traws = ["\n".join(titles[k:k + tpb]).encode("utf-8") for k in range(0, len(titles), tpb)]
    tarr = _blocks_to_arrays(traws, cfg)
    return Packed(
        cfg=cfg, blob=text["blob"], block_off=text["off"], block_raw=text["raw"],
        block_crc=text["crc"], block_first=np.asarray(first, dtype=np.uint32), pas_off=offs,
        title_first=title_first, tblob=tarr["blob"], tblock_off=tarr["off"],
        tblock_raw=tarr["raw"], tblock_crc=tarr["crc"],
    )


# ------------------------------------------------------------------------------ io utils


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
