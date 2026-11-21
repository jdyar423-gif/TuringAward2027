"""Tests of bes.tokenizer.train_bpe against a naive BPE trainer written inside this file.

The reference counts every adjacent pair of every regex chunk (weighted by chunk frequency),
merges the most frequent pair (ties: the smallest pair ``(left, right)``) everywhere left to
right, and repeats. It is quadratic and only used on small corpora.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAYON_NUM_THREADS", "2")  # before rustbpe starts its thread pool

import collections
import json
import random

import pytest
import regex

from bes import config
from bes.tokenizer import tok as tokmod
from bes.tokenizer import train_bpe
from bes.tokenizer.tok import Tokenizer


def ref_train(texts: list[str], n_merges: int, pattern: str) -> list[tuple[int, int]]:
    words = collections.Counter()
    for text in texts:
        words.update(regex.findall(pattern, text))
    state = [[list(w.encode("utf-8")), n] for w, n in words.items()]
    merges = []
    for k in range(n_merges):
        pairs = collections.Counter()
        for ids, n in state:
            for p in zip(ids, ids[1:]):
                pairs[p] += n
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        for entry in state:
            ids, out, i = entry[0], [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    out.append(256 + k)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            entry[0] = out
    return merges


def ref_encode(text: str, merges, pattern: str) -> list[int]:
    rank = {tuple(m): 256 + i for i, m in enumerate(merges)}
    out = []
    for chunk in regex.findall(pattern, text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            cands = [(rank[p], i) for i, p in enumerate(zip(ids, ids[1:])) if p in rank]
            if not cands:
                break
            new, i = min(cands)
            ids[i:i + 2] = [new]
        out.extend(ids)
    return out


def corpus(seed: int = 0, n: int = 600) -> list[str]:
    rng = random.Random(seed)
    words = ["the", "cat", "sat", "on", "mat", "hello", "world", "token", "tokens", "byte",
             "pair", "encoding", "naïve", "café", "日本", "語", "über", "aaa", "zz", "I'm", "don't"]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 25)):
            r = rng.random()
            if r < 0.7:
                parts.append(rng.choice(words) if rng.random() < 0.9 else rng.choice(words).upper())
            elif r < 0.8:
                parts.append(str(rng.randint(0, 10 ** rng.randint(1, 8))))
            elif r < 0.9:
                parts.append(rng.choice([".", ",", "!", "?", "--", "...", "(x)", "🙂", "🙂🙂"]))
            else:
                parts.append(rng.choice(["\n", "\n\n", "\t", "  "]))
        out.append(" ".join(parts))
    out += ["a" * 40, "aaaa aaaaa aaaaaaa", "abababab ab", "1111 111111", "zzz" * 9, ""]
    return out


@pytest.fixture(scope="module")
def trained():
    texts = corpus()
    merges, trainer = train_bpe.train_merges(texts, 150)
    return texts, merges, trainer


def test_pattern_is_the_two_digit_variant():
    pat = train_bpe.SPLIT_PATTERN
    assert r"\p{N}{1,2}" in pat and r"\p{N}{1,3}" not in pat
    assert regex.findall(pat, "abc 12345 don't  \n\nx") == ["abc", " ", "12", "34", "5", " don", "'t",
                                                            "  \n\n", "x"]
    assert train_bpe.N_MERGES == 7920 == config.VOCAB_SIZE - config.N_SPECIAL - 256


def test_rustbpe_merges_equal_reference(trained):
    texts, merges, _ = trained
    assert len(merges) == 150
    assert merges == ref_train(texts, 150, train_bpe.SPLIT_PATTERN)


def test_rustbpe_merges_equal_reference_on_unicode_corpus():
    rng = random.Random(5)
    alphabet = "abcdé日本語🙂 \n'0123456789.,"
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80))) for _ in range(400)]
    merges, _ = train_bpe.train_merges(texts, 120)
    assert merges == ref_train(texts, 120, train_bpe.SPLIT_PATTERN)


def test_rustbpe_merges_equal_reference_on_sample_text():
    """Real chat text, when data_cache/samples/ has been downloaded."""
    from bes.tokenizer import samples

    if not all(f.local_path.exists() for f in samples.files_for("chat", "heldout")):
        pytest.skip("samples not downloaded (python scripts/build_tokenizer.py)")
    texts = list(samples.iter_texts("chat", "heldout", 120_000))
    merges, _ = train_bpe.train_merges(texts, 120)
    assert merges == ref_train(texts, 120, train_bpe.SPLIT_PATTERN)


def test_training_is_deterministic(trained):
    texts, merges, _ = trained
    again, _ = train_bpe.train_merges(list(reversed(texts)), 150)
    assert again == merges  # counts do not depend on the order of the texts


def test_training_stops_when_pairs_run_out():
    merges, _ = train_bpe.train_merges(["ab ab", "abc"], 10)
    assert merges == ref_train(["ab ab", "abc"], 10, train_bpe.SPLIT_PATTERN) == [(97, 98), (32, 256), (256, 99)]


def test_merges_from_ranks_round_trip(trained):
    _, merges, _ = trained
    table = tokmod.token_bytes_from_merges(merges)
    ranks = {b: i for i, b in enumerate(table)}
    assert train_bpe.merges_from_ranks(ranks) == merges


def test_merges_from_ranks_rejects_invalid_ranks(trained):
    _, merges, _ = trained
    table = tokmod.token_bytes_from_merges(merges)
    ranks = {b: i for i, b in enumerate(table)}
    gap = dict(ranks)
    gap[table[-1]] = len(table) + 3
    with pytest.raises(ValueError, match="consecutive"):
        train_bpe.merges_from_ranks(gap)
    swapped = dict(ranks)
    swapped[b"a"], swapped[b"b"] = ranks[b"b"], ranks[b"a"]
    with pytest.raises(ValueError, match="single byte"):
        train_bpe.merges_from_ranks(swapped)
    not_a_merge = {bytes([i]): i for i in range(256)}
    not_a_merge[b"xyz"] = 256  # needs two merges, cannot be the first one
    with pytest.raises(ValueError, match="not the merge"):
        train_bpe.merges_from_ranks(not_a_merge)


def test_asset_round_trip_and_agreement(tmp_path, trained):
    texts, merges, trainer = trained
    digest = train_bpe.write_asset(merges, tmp_path, meta={"note": "test"})
    raw_text = (tmp_path / tokmod.ASSET_FILE).read_text()
    raw = json.loads(raw_text)
    assert raw["merges"] == [list(m) for m in merges] and raw["sha256"] == digest
    assert raw_text.count("\n    [") == len(merges)  # one merge per line
    meta = json.loads((tmp_path / tokmod.META_FILE).read_text())
    assert meta == {"note": "test", "sha256": digest}
    tok = Tokenizer.from_asset(tmp_path)
    assert tok.sha256 == digest
    assert tok.vocab_size == 256 + 150 + config.N_SPECIAL
    assert tok.bos == tok.vocab_size - config.N_SPECIAL + 1
    assert tok.special_tokens == {n: config.special_id(n, tok.vocab_size) for n in config.SPECIAL_TOKENS}
    probe = texts[:200] + ["unseen words: zebra quokka 9876543210 ünïcödé 🙂🙂🙂 <|bos|>"]
    for text in probe:
        ids = tok.encode(text)
        assert ids == list(trainer.encode(text)) == ref_encode(text, merges, train_bpe.SPLIT_PATTERN)
        assert tok.decode(ids) == text
    assert not (tmp_path / (tokmod.ASSET_FILE + ".tmp")).exists()


def test_dump_asset_rejects_forward_references():
    with pytest.raises(ValueError):
        train_bpe.dump_asset([(97, 98), (300, 97)])
