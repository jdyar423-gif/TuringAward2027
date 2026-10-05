"""Data preparation: WikiText-2 -> byte-level BPE token streams.

Prefers the raw WikiText-2 (wikitext-2-raw-v1) if it is present under
data/wikitext-2-raw/ (wiki.{train,valid,test}.raw); otherwise uses the
standard WikiText-2 (v1) release mirrored in pytorch/examples.

The tokenizer is a lossless byte-level BPE trained on the TRAIN split only.
BPB is always computed against the exact UTF-8 bytes of the split file, so
the choice of tokenizer cannot change the denominator.

Usage: python prepare.py --vocab 8192
"""
import argparse
import heapq
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MIRROR = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/{}.txt"
SPLITS = ("train", "valid", "test")

# GPT-2 style pre-tokenizer (no \p{L} in `re`, so [^\W\d_] stands in for letters).
# The trailing catch-all guarantees the chunks always re-concatenate to the input.
PAT = re.compile(r""" ?<unk>|'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+|.""", re.S)


def load_splits():
    raw_dir = os.path.join(DATA, "wikitext-2-raw")
    if all(os.path.exists(os.path.join(raw_dir, f"wiki.{s}.raw")) for s in SPLITS):
        name = "wikitext-2-raw-v1"
        texts = {s: open(os.path.join(raw_dir, f"wiki.{s}.raw"), "rb").read() for s in SPLITS}
    else:
        name = "wikitext-2-v1"
        d = os.path.join(DATA, "wikitext-2")
        os.makedirs(d, exist_ok=True)
        texts = {}
        for s in SPLITS:
            p = os.path.join(d, f"{s}.txt")
            if not os.path.exists(p):
                urllib.request.urlretrieve(MIRROR.format(s), p)
            texts[s] = open(p, "rb").read()
    return name, {s: t.decode("utf-8") for s, t in texts.items()}


def chunks(text):
    out = PAT.findall(text)
    assert "".join(out) == text
    return out


def train_bpe(text, vocab_size, n_special=1):
    """Byte-level BPE. Returns list of merges ((a, b) -> new id), ids 0..255 are bytes."""
    words = Counter(chunks(text))
    seqs = [list(w.encode("utf-8")) for w in words]
    freqs = [words[w] for w in words]
    pair_cnt = defaultdict(int)
    where = defaultdict(set)
    for i, s in enumerate(seqs):
        for a, b in zip(s, s[1:]):
            pair_cnt[(a, b)] += freqs[i]
            where[(a, b)].add(i)
    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)
    merges = []
    n_merges = vocab_size - 256 - n_special
    while len(merges) < n_merges and heap:
        negc, p = heapq.heappop(heap)
        if pair_cnt.get(p, 0) != -negc or -negc <= 0:
            continue  # stale entry
        new = 256 + len(merges)
        merges.append(p)
        touched = defaultdict(int)
        for i in list(where[p]):
            s, f = seqs[i], freqs[i]
            for a, b in zip(s, s[1:]):  # remove old pairs
                pair_cnt[(a, b)] -= f
                touched[(a, b)] += 0
            j, out = 0, []
            while j < len(s):
                if j + 1 < len(s) and s[j] == p[0] and s[j + 1] == p[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            seqs[i] = out
            for a, b in zip(out, out[1:]):
                pair_cnt[(a, b)] += f
                where[(a, b)].add(i)
                touched[(a, b)] += 0
        del pair_cnt[p]
        where.pop(p, None)
        for q in touched:
            if q in pair_cnt and pair_cnt[q] > 0:
                heapq.heappush(heap, (-pair_cnt[q], q))
    return merges


class BPE:
    def __init__(self, merges, n_special=1):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges) + n_special
        self.bos = self.vocab_size - 1
        self.bytes_of = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.bytes_of.append(self.bytes_of[a] + self.bytes_of[b])
        self.bytes_of.append(b"")  # BOS
        self.cache = {}

    def encode_chunk(self, w):
        if w in self.cache:
            return self.cache[w]
        s = list(w.encode("utf-8"))
        while len(s) > 1:
            best, bi = None, -1
            for i, pr in enumerate(zip(s, s[1:])):
                r = self.rank.get(pr)
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            new = 256 + best
            pr = self.merges[best]
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == pr[0] and s[j + 1] == pr[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            s = out
        self.cache[w] = s
        return s

    def encode(self, text):
        ids = []
        for w in chunks(text):
            ids.extend(self.encode_chunk(w))
        return ids

    def decode(self, ids):
        return b"".join(self.bytes_of[i] for i in ids).decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()
    name, texts = load_splits()
    out = os.path.join(DATA, f"bpe{args.vocab}")
    os.makedirs(out, exist_ok=True)
    merges = train_bpe(texts["train"], args.vocab)
    tok = BPE(merges)
    meta = {"dataset": name, "vocab_size": tok.vocab_size, "bos": tok.bos, "merges": merges}
    for s in SPLITS:
        ids = tok.encode(texts[s])
        assert tok.decode(ids) == texts[s], f"BPE round-trip failed on {s}"
        arr = np.array([tok.bos] + ids, dtype=np.uint16)  # BOS is context only, never scored
        arr.tofile(os.path.join(out, f"{s}.bin"))
        nbytes = len(texts[s].encode("utf-8"))
        meta[f"{s}_bytes"] = nbytes
        meta[f"{s}_tokens"] = len(ids)
        print(f"{name} {s}: {nbytes} bytes, {len(ids)} tokens, {nbytes/len(ids):.3f} bytes/token")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
