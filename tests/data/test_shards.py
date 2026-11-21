"""Token shard format, reader and consistency checker (``bes/data/shards.py``).

The reference for the on-disk bytes is built independently with ``struct``/numpy from lists of
segments; corruption tests edit files directly and re-sign the manifest where needed, so that
each check is exercised on its own.
"""
from __future__ import annotations

import json
import struct
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

from bes.config import BOS, VOCAB_SIZE
from bes.data import shards as SH


def random_segments(rng: np.random.Generator, n: int, max_len: int = 40) -> list[list[int]]:
    ordinary = np.setdiff1d(np.arange(VOCAB_SIZE), [BOS])
    return [[BOS] + rng.choice(ordinary, size=int(rng.integers(0, max_len))).tolist() for _ in range(n)]


def random_roles(rng: np.random.Generator, segs: list[list[int]]) -> list[list[int]]:
    return [rng.integers(0, SH.N_ROLE_CODES, size=len(s)).tolist() for s in segs]


def write_shard(directory: Path, shard: int, segs, roles=None, recs=None, commit=True) -> dict:
    w = SH.ShardWriter(directory, shard, with_roles=roles is not None)
    for i, s in enumerate(segs):
        w.add(s, None if roles is None else roles[i], rec=i if recs is None else recs[i])
    entry = w.close()
    entry["source"] = "test"
    if commit:
        SH.append_manifest(directory, entry)
    return entry


def ref_bytes(segs, roles=None, recs=None) -> dict[str, bytes]:
    """Independent reference of the four files, with struct."""
    flat = [t for s in segs for t in s]
    offs = [0]
    for s in segs:
        offs.append(offs[-1] + len(s))
    out = {"u16": struct.pack(f"<{len(flat)}H", *flat), "idx": struct.pack(f"<{len(offs)}q", *offs),
           "rec": struct.pack(f"<{len(segs)}q", *(recs if recs is not None else range(len(segs))))}
    if roles is not None:
        out["role"] = struct.pack(f"<{len(flat)}b", *[r for rs in roles for r in rs])
    return out


# --------------------------------------------------------------------------- round trip
@pytest.mark.parametrize("with_roles", [False, True])
def test_round_trip_bytes_and_reader(tmp_path, with_roles):
    rng = np.random.default_rng(0)
    segs = random_segments(rng, 300)
    roles = random_roles(rng, segs) if with_roles else None
    entry = write_shard(tmp_path, 7, segs, roles)
    for ext, blob in ref_bytes(segs, roles).items():
        assert (tmp_path / f"000007.{ext}").read_bytes() == blob, ext
        assert entry["files"][ext] == {"bytes": len(blob), "crc32": zlib.crc32(blob)}
    assert entry["n_tokens"] == sum(map(len, segs)) and entry["n_segments"] == len(segs)
    assert not (tmp_path / "000007.role").exists() or with_roles
    ss = SH.ShardSet(tmp_path)
    assert len(ss) == len(segs) and ss.n_tokens == entry["n_tokens"]
    for j, s in enumerate(segs):
        assert ss.segment(j).tolist() == s
        r = ss.segment_roles(j)
        assert (r is None) == (not with_roles)
        if with_roles:
            assert r.tolist() == roles[j]
    sh = ss.shards[0]
    assert sh.recs.tolist() == list(range(len(segs)))
    assert sh.segment_lengths().tolist() == [len(s) for s in segs]
    rep = SH.check(tmp_path)
    assert rep.ok and not rep.warnings and rep.n_segments == len(segs), rep


def test_empty_shard_and_global_indexing(tmp_path):
    rng = np.random.default_rng(1)
    write_shard(tmp_path, 3, [])                       # every record filtered out
    a, b = random_segments(rng, 5), random_segments(rng, 9)
    write_shard(tmp_path, 10, b)                       # committed out of order
    write_shard(tmp_path, 4, a)
    ss = SH.ShardSet(tmp_path)
    assert [s.shard for s in ss.shards] == [3, 4, 10]
    ref = a + b                                        # ascending shard id order
    assert len(ss) == len(ref)
    assert [ss.segment(j).tolist() for j in range(len(ref))] == ref
    assert ss.locate(5) == (2, 0)
    with pytest.raises(IndexError):
        ss.segment(len(ref))
    assert SH.check(tmp_path).ok


def test_writer_rejects_invalid_segments(tmp_path):
    w = SH.ShardWriter(tmp_path, 0, with_roles=True)
    bad = [
        ([], [], "non-empty"), ([5, 6], [0, 0], "BOS"), ([BOS, 1, BOS], [0, 0, 0], "BOS inside"),
        ([BOS, VOCAB_SIZE], [0, 0], "outside"), ([BOS, -1], [0, 0], "outside"),
        ([BOS, 1], [0], "shape"), ([BOS, 1], [0, SH.N_ROLE_CODES], "role codes"),
        ([BOS, 1.5], [0, 0], "integers"),
    ]
    for ids, roles, msg in bad:
        with pytest.raises(ValueError, match=msg):
            w.add(ids, roles)
    with pytest.raises(ValueError, match="roles="):
        w.add([BOS, 1])
    w.add([BOS, 1], [0, 3])
    w.abort()
    assert list(tmp_path.iterdir()) == []
    w2 = SH.ShardWriter(tmp_path, 1, with_roles=False)
    with pytest.raises(ValueError, match="with_roles=False"):
        w2.add([BOS, 1], [0, 0])
    w2.abort()


# -------------------------------------------------------------- crashes and the manifest
def test_crash_leftovers_are_ignored_and_cleaned(tmp_path):
    rng = np.random.default_rng(2)
    good = random_segments(rng, 4)
    write_shard(tmp_path, 0, good)
    # a writer that died before close: temporaries only
    w = SH.ShardWriter(tmp_path, 1, with_roles=False)
    w.add([BOS, 1, 2])
    w._tok_f.flush()
    # a shard renamed into place but never committed (crash before the manifest append)
    write_shard(tmp_path, 2, random_segments(rng, 3), commit=False)
    ss = SH.ShardSet(tmp_path)
    assert [s.shard for s in ss.shards] == [0] and len(ss) == 4
    rep = SH.check(tmp_path)
    assert rep.ok
    assert any("000001.u16.tmp" in x for x in rep.warnings)
    assert any("000002.u16" in x for x in rep.warnings)
    assert SH.remove_temporaries(tmp_path, 1) == 1       # only the .u16 temporary exists before close
    w._tok_f.close()
    assert not any(".tmp." in p.name for p in tmp_path.iterdir())


def test_torn_manifest_line_is_ignored_then_repaired(tmp_path):
    rng = np.random.default_rng(3)
    e0 = write_shard(tmp_path, 0, random_segments(rng, 2))
    man = tmp_path / SH.MANIFEST
    with open(man, "ab") as f:
        f.write(b'{"format": "bes-shard/1", "shard": 1, "n_tok')     # crash mid-append
    assert list(SH.read_manifest(tmp_path)) == [0]
    assert SH.check(tmp_path).ok                                    # torn tail is not an error
    e1 = write_shard(tmp_path, 1, random_segments(rng, 3))
    lines = man.read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3
    assert [json.loads(x)["shard"] for x in lines[:2]] == [0, 1]
    assert SH.read_manifest(tmp_path) == {0: e0, 1: e1}


def test_concurrent_appends_keep_lines_whole(tmp_path):
    path = tmp_path / "m.jsonl"

    def worker(k: int) -> None:
        for i in range(80):
            SH.append_jsonl(path, {"shard": k * 1000 + i, "pad": "x" * ((k + 1) * 37 * (i % 9))})

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    objs, problems = SH.scan_jsonl(path)
    assert not problems and sorted(o["shard"] for o in objs) == sorted(k * 1000 + i for k in range(2) for i in range(80))


# ------------------------------------------------------------------- checker: corruption
def _sign(directory: Path, shard: int) -> None:
    """Rewrite the manifest so its CRCs and sizes match the files on disk again."""
    man = directory / SH.MANIFEST
    entries = [json.loads(x) for x in man.read_text().splitlines()]
    for e in entries:
        if e["shard"] == shard:
            for ext in e["files"]:
                blob = (directory / f"{SH.stem(shard)}.{ext}").read_bytes()
                e["files"][ext] = {"bytes": len(blob), "crc32": zlib.crc32(blob)}
    man.write_text("".join(json.dumps(e) + "\n" for e in entries))


@pytest.fixture
def good_shard(tmp_path):
    rng = np.random.default_rng(4)
    segs = random_segments(rng, 50, max_len=20)
    roles = random_roles(rng, segs)
    write_shard(tmp_path, 0, segs, roles)
    assert SH.check(tmp_path).ok
    return tmp_path, segs


def _patch(path: Path, offset: int, data: bytes) -> None:
    blob = bytearray(path.read_bytes())
    blob[offset:offset + len(data)] = data
    path.write_bytes(bytes(blob))


def test_checker_detects_flipped_token_byte(good_shard):
    d, segs = good_shard
    _patch(d / "000000.u16", 2 * 5 + 1, b"\x7f")
    rep = SH.check(d)
    assert not rep.ok and any("CRC" in e for e in rep.errors)
    assert SH.check(d, deep=False).ok          # the cheap check cannot see a same-size flip


def test_checker_detects_bos_inside_segment_and_large_ids(good_shard):
    d, segs = good_shard
    pos = len(segs[0]) + 1                       # second token of the second segment
    assert len(segs[1]) > 1
    _patch(d / "000000.u16", 2 * pos, struct.pack("<H", BOS))
    _sign(d, 0)
    rep = SH.check(d)
    assert any("BOS positions" in e for e in rep.errors) and not any("CRC" in e for e in rep.errors)
    _patch(d / "000000.u16", 2 * pos, struct.pack("<H", VOCAB_SIZE + 3))
    _sign(d, 0)
    assert any("vocab_size" in e for e in SH.check(d).errors)


def test_checker_detects_bad_offsets_and_roles(good_shard):
    d, segs = good_shard
    _patch(d / "000000.idx", 8 * 2, struct.pack("<q", 0))        # offsets no longer increasing
    _sign(d, 0)
    assert any("offsets" in e for e in SH.check(d).errors)


def test_checker_detects_bad_role_codes(good_shard):
    d, _ = good_shard
    _patch(d / "000000.role", 3, struct.pack("<b", 9))
    _sign(d, 0)
    rep = SH.check(d)
    assert any("role code" in e for e in rep.errors) and not any("CRC" in e for e in rep.errors)


def test_checker_detects_truncated_or_missing_files(good_shard):
    d, _ = good_shard
    blob = (d / "000000.idx").read_bytes()
    (d / "000000.idx").write_bytes(blob[:-8])
    assert any("bytes" in e for e in SH.check(d).errors)
    (d / "000000.idx").write_bytes(blob)
    assert SH.check(d).ok
    (d / "000000.role").unlink()
    assert not SH.check(d).ok


def test_checker_detects_duplicate_and_invalid_manifest_lines(good_shard):
    d, _ = good_shard
    man = d / SH.MANIFEST
    line = man.read_text().splitlines()[0]
    man.write_text(line + "\n" + line + "\n")
    assert any("more than once" in e for e in SH.check(d).errors)
    man.write_text("not json\n" + line + "\n")
    assert any("not valid JSON" in e for e in SH.check(d).errors)
    e = json.loads(line)
    e["format"] = "other/9"
    man.write_text(json.dumps(e) + "\n")
    assert any("unknown format" in x for x in SH.check(d).errors)


def test_checker_pairs_need_even_segment_count(tmp_path):
    rng = np.random.default_rng(5)
    w = SH.ShardWriter(tmp_path, 0, with_roles=True)
    for s in random_segments(rng, 3):
        w.add(s, [0] * len(s))
    e = w.close()
    e["pairs"] = True
    SH.append_manifest(tmp_path, e)
    assert any("odd number" in x for x in SH.check(tmp_path).errors)


def test_atomic_write_and_scan(tmp_path):
    p = tmp_path / "a" / "b.json"
    SH.atomic_write_bytes(p, b"{}")
    SH.atomic_write_bytes(p, b'{"x": 1}')
    assert p.read_bytes() == b'{"x": 1}' and [q.name for q in p.parent.iterdir()] == ["b.json"]
    assert SH.scan_jsonl(tmp_path / "missing.jsonl") == ([], [])
