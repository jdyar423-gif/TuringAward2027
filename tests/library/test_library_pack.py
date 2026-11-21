"""Packing into independent xz blocks (bes.library.pack): round trip, CRC, random access,
exact accounting of stored numbers."""
from __future__ import annotations

import lzma
import os
import random
import zlib

import numpy as np
import pytest

from bes.library.pack import CorruptBlockError, PackCfg, Packed, bsearch_right, pack


def _corpus(seed: int, n_titles: int = 60, block: int = 4096):
    rng = random.Random(seed)
    alphabet = "abcdefghij klmnop Ωé漢字 \n-"
    titles, bodies, tids = [], [], []
    for t in range(n_titles):
        titles.append(f"Title {t} " + "".join(rng.choice("xyzé") for _ in range(rng.randint(0, 5))))
        for _ in range(rng.randint(1, 6)):
            n = rng.randint(1, 900)
            bodies.append("".join(rng.choice(alphabet) for _ in range(n)))
            tids.append(t)
    cfg = PackCfg(block_raw_bytes=block, titles_per_block=7, threads=2)
    return titles, bodies, tids, cfg


@pytest.mark.parametrize("seed", range(3))
def test_round_trip_and_directory(seed):
    titles, bodies, tids, cfg = _corpus(seed)
    pk = pack(bodies, tids, titles, cfg)
    assert pk.n_passages == len(bodies) and pk.n_titles == len(titles)
    for pid, body in enumerate(bodies):
        assert pk.passage_body(pid) == body
        tid, _ = pk.title_of(pid)
        assert tid == tids[pid] and pk.title(tid) == titles[tids[pid]]
    assert pk.all_titles() == titles
    assert [b for _, b in pk.iter_bodies()] == bodies
    # blocks hold whole passages and respect the raw size limit
    assert int(np.asarray(pk.block_raw).max()) <= cfg.block_raw_bytes
    for pid in range(len(bodies)):
        b, s, e, _ = pk.locate(pid)
        assert 0 <= s < e <= int(pk.block_raw[b])
        assert int(pk.block_first[b]) <= pid < int(pk.block_first[b + 1])


def test_random_access_reads_exactly_one_block(monkeypatch):
    titles, bodies, tids, cfg = _corpus(7)
    pk = pack(bodies, tids, titles, cfg)
    calls = []
    real = pk.read_block

    def counting(b):
        calls.append(b)
        return real(b)

    monkeypatch.setattr(pk, "read_block", counting)
    rng = random.Random(1)
    for pid in rng.sample(range(len(bodies)), 25):
        calls.clear()
        assert pk.passage_body(pid) == bodies[pid]
        b, _, _, _ = pk.locate(pid)
        assert calls == [b]


def test_blocks_are_independent_raw_lzma2_streams():
    titles, bodies, tids, cfg = _corpus(3)
    pk = pack(bodies, tids, titles, cfg)
    for b in range(pk.n_blocks):
        comp = pk.blob[int(pk.block_off[b]):int(pk.block_off[b + 1])].tobytes()
        raw = lzma.decompress(comp, format=lzma.FORMAT_RAW, filters=cfg.filters())
        assert len(raw) == int(pk.block_raw[b])
        assert zlib.crc32(raw) & 0xFFFFFFFF == int(pk.block_crc[b])


def test_crc_detects_corruption():
    titles, bodies, tids, cfg = _corpus(4)
    pk = pack(bodies, tids, titles, cfg)
    b = pk.n_blocks // 2
    lo, hi = int(pk.block_off[b]), int(pk.block_off[b + 1])
    rng = random.Random(0)
    detected = 0
    for trial in range(20):
        blob = pk.blob.copy()
        pos = rng.randrange(lo, hi)
        blob[pos] ^= 1 << rng.randrange(8)
        bad = Packed(**{**pk.__dict__, "blob": blob})
        try:
            raw = bad.read_block(b)
        except CorruptBlockError:
            detected += 1
            continue
        pytest.fail(f"corruption at byte {pos} not detected (decoded {len(raw)} bytes)")
    assert detected == 20
    # a wrong CRC entry or raw length is detected as well
    crc = np.asarray(pk.block_crc).copy()
    crc[b] ^= 1
    with pytest.raises(CorruptBlockError):
        Packed(**{**pk.__dict__, "block_crc": crc}).read_block(b)
    raw_len = np.asarray(pk.block_raw).copy()
    raw_len[b] -= 1
    with pytest.raises(CorruptBlockError):
        Packed(**{**pk.__dict__, "block_raw": raw_len}).read_block(b)
    # other blocks are unaffected by corruption of block b
    blob = pk.blob.copy()
    blob[lo] ^= 0xFF
    bad = Packed(**{**pk.__dict__, "blob": blob})
    other = (b + 1) % pk.n_blocks
    assert bad.read_block(other) == pk.read_block(other)


def test_exact_accounting_matches_files(tmp_path):
    titles, bodies, tids, cfg = _corpus(5)
    pk = pack(bodies, tids, titles, cfg)
    acc = pk.accounting()
    # independent count: payload bytes of every stored array, one number per 2 bytes
    expect = 0
    for name in Packed.ARRAYS:
        nbytes = np.asarray(getattr(pk, name)).nbytes
        expect += (nbytes + 1) // 2
    assert acc["library_numbers"] == expect
    assert acc["text_numbers"] + acc["directory_numbers"] + acc["title_numbers"] == expect
    assert acc["text_blob_numbers"] == (int(pk.block_off[-1]) + 1) // 2
    assert acc["array_pas_off"] == len(bodies)            # uint16 per passage
    assert acc["array_title_first"] == 2 * (len(titles) + 1)  # uint32 per title + 1
    assert acc["raw_text_bytes"] == sum(len(b.encode()) for b in bodies)
    pk.save(str(tmp_path))
    on_disk = 0
    for name in Packed.ARRAYS:
        arr = np.load(os.path.join(tmp_path, f"pack_{name}.npy"))
        on_disk += (arr.nbytes + 1) // 2
    assert on_disk == expect
    re = Packed.load(str(tmp_path))
    assert re.accounting() == acc
    assert [re.passage_body(i) for i in range(len(bodies))] == bodies


def test_pack_validates_title_order():
    with pytest.raises(ValueError):
        pack(["a", "b", "c"], [0, 1, 0], ["x", "y"])
    with pytest.raises(ValueError):
        pack(["a", "b"], [0, 2], ["x", "y", "z"])
    with pytest.raises(ValueError):
        pack(["a"], [0], ["bad\ntitle"])


def test_empty_pack():
    pk = pack([], [], [])
    assert pk.n_passages == 0 and pk.n_blocks == 0 and pk.n_titles == 0
    assert pk.accounting()["library_numbers"] >= 0


def test_bsearch_right_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        arr = np.sort(rng.integers(0, 50, size=rng.integers(1, 40))).astype(np.uint32)
        x = int(rng.integers(-1, 52))
        idx, probes = bsearch_right(arr, x)
        assert idx == int(np.searchsorted(arr, x, side="right"))
        assert len(probes) <= int(np.ceil(np.log2(arr.shape[0] + 1)))
