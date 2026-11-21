"""Tests of the committed tokenizer (assets/tokenizer/) and of bes.tokenizer.tok.

The reference encoder below is written independently of the implementation: it splits text with
the ``regex`` module and applies the stored merge pairs in rank order (plain pairwise BPE),
while the implementation runs tiktoken on byte-string ranks.
"""
from __future__ import annotations

import json
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import regex

from bes import config
from bes.tokenizer import tok as tokmod
from bes.tokenizer.tok import Tokenizer, load

ASSET = Path(tokmod.ASSET_DIR)
N_ORDINARY = config.VOCAB_SIZE - config.N_SPECIAL


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return load()


@pytest.fixture(scope="module")
def raw_asset() -> dict:
    return json.loads((ASSET / tokmod.ASSET_FILE).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ independent reference
def ref_encode(text: str, merges: list[list[int]], pattern: str) -> list[int]:
    """Pairwise BPE: per regex chunk, repeatedly merge the adjacent pair of lowest merge index."""
    rank = {(a, b): 256 + i for i, (a, b) in enumerate(merges)}
    out: list[int] = []
    for chunk in regex.findall(pattern, text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best = min(range(len(ids) - 1), key=lambda i: rank.get((ids[i], ids[i + 1]), 1 << 30))
            new = rank.get((ids[best], ids[best + 1]))
            if new is None:
                break
            ids[best:best + 2] = [new]
        out.extend(ids)
    return out


def ref_token_bytes(merges: list[list[int]]) -> list[bytes]:
    table = [bytes([i]) for i in range(256)]
    for a, b in merges:
        table.append(table[a] + table[b])
    return table


# --------------------------------------------------------------------------- test corpus
SPECIAL_STRINGS = list(config.SPECIAL_TOKENS)

FIXED_TEXTS = [
    "",
    " ",
    "a",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "\n", "\n\n\n", "\t\tindented\n", "line one\r\nline two\r\n", "trailing newline\n",
    "naïve café résumé — “curly quotes” ‘single’ … ½ ² ³ € £ ¥ © ®",
    "日本語のテキストと中文混合、한국어 텍스트。",
    "Ελληνικά κείμενα, русский текст, العربية, हिन्दी, עברית, ไทย",
    "emoji 👍🏽 👨‍👩‍👧‍👦 🇹🇼 🏳️‍🌈 ❤️ 🤖🤖🤖",
    "combining e\u0301 a\u0308 \u0915\u094d\u0937 zero\u200bwidth\u200d joiner \ufeff bom",
    "controls \x00\x01\x02\x7f escape \x1b[31mred\x1b[0m bell\x07",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n\n\tif x:\n\t\tpass\n",
    '{"key": "value", "n": 12345, "list": [1.5e-10, -3, null, true]}',
    "<html><body class=\"x\">&amp; &lt;tag&gt;</body></html>",
    r"regex: ^\s*(\d{1,3})[.)]\s+(?P<name>\w+)$",
    "3141592653589793238462643383279502884197169399375105820974944592",
    "1,234,567.89 0x1F 0b1010 1e-10 -42 +7 2024-11-21 13:45:00 v1.2.3",
    "I'm don't they're we've you'll he'd I'M DON'T THEY'RE",
    "a" * 300, " " * 300, "=" * 300, "\n" * 100, "ab" * 150, "1" * 301, "😀" * 50,
    "".join(SPECIAL_STRINGS),
    "text <|user|> more <|end|> and <|bos|><|bos|>",
    "partial <|end <|/calc| |> <||> <|eq|",
]

LONG_TEXTS = ["a" * 20000, " " * 5000 + "x", "x\n" * 5000, "1234567890" * 2000]

_BLOCKS = [(0x20, 0x7E), (0xA0, 0xFF), (0x100, 0x17F), (0x391, 0x3C9), (0x410, 0x44F),
           (0x5D0, 0x5EA), (0x627, 0x64A), (0x905, 0x939), (0x3041, 0x3093), (0x4E00, 0x4FFF),
           (0xAC00, 0xAD00), (0x1F300, 0x1F64F), (0x2000, 0x206F), (0x30, 0x39)]


def random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = []
        for _ in range(rng.randrange(0, 120)):
            r = rng.random()
            if r < 0.15:
                chars.append(rng.choice([" ", "\n", "\t", "  ", "\r\n", "'"]))
            else:
                lo, hi = rng.choice(_BLOCKS)
                chars.append(chr(rng.randint(lo, hi)))
        out.append("".join(chars))
    return out


ALL_TEXTS = FIXED_TEXTS + random_texts(300, seed=1)


# ----------------------------------------------------------------------------- the asset
def test_asset_files_are_small_and_consistent(raw_asset):
    for name in (tokmod.ASSET_FILE, tokmod.META_FILE):
        assert (ASSET / name).stat().st_size < 1_000_000
    assert raw_asset["format"] == tokmod.ASSET_FORMAT
    assert raw_asset["vocab_size"] == config.VOCAB_SIZE
    assert raw_asset["n_bytes"] == 256
    assert raw_asset["special_tokens"] == list(config.SPECIAL_TOKENS)
    assert len(raw_asset["merges"]) == config.VOCAB_SIZE - config.N_SPECIAL - 256 == 7920
    for i, (a, b) in enumerate(raw_asset["merges"]):
        assert 0 <= a < 256 + i and 0 <= b < 256 + i
    meta = json.loads((ASSET / tokmod.META_FILE).read_text(encoding="utf-8"))
    assert meta["sha256"] == raw_asset["sha256"]
    assert meta["vocab_size"] == config.VOCAB_SIZE and meta["n_merges"] == 7920
    assert set(meta["training"]["domains"]) == {"web", "chat", "wiki"}


def test_hash_is_checked(tmp_path, raw_asset):
    content = tokmod.read_asset(ASSET)
    assert content["sha256"] == raw_asset["sha256"] == tokmod.content_hash(
        tokmod.canonical_content(raw_asset["merges"], raw_asset["pattern"], raw_asset["special_tokens"]))
    bad = dict(raw_asset, merges=[list(m) for m in raw_asset["merges"]])
    bad["merges"][100] = [bad["merges"][100][1], bad["merges"][100][0]]
    (tmp_path / tokmod.ASSET_FILE).write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        tokmod.read_asset(tmp_path)
    (tmp_path / tokmod.ASSET_FILE).write_text(json.dumps(dict(raw_asset, format="other")), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        tokmod.read_asset(tmp_path)
    (tmp_path / tokmod.ASSET_FILE).write_text(json.dumps(dict(raw_asset, vocab_size=9000)), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        tokmod.read_asset(tmp_path)
    (tmp_path / tokmod.ASSET_FILE).write_text(json.dumps(dict(raw_asset, merges=[[1, 2, 3]])), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        tokmod.read_asset(tmp_path)


def test_load_from_copied_asset_only(tmp_path, tok):
    shutil.copy(ASSET / tokmod.ASSET_FILE, tmp_path / tokmod.ASSET_FILE)
    other = Tokenizer.from_asset(tmp_path)
    assert other.sha256 == tok.sha256 and other is not tok
    for text in FIXED_TEXTS:
        assert other.encode(text) == tok.encode(text)


def test_loading_needs_neither_training_code_nor_data():
    code = (
        "import sys\n"
        "from bes.tokenizer.tok import load\n"
        "t = load()\n"
        "ids = t.encode('Loading check: 12345 <|end|> \u00e9')\n"
        "assert t.decode(ids) == 'Loading check: 12345 <|end|> \u00e9'\n"
        "bad = [m for m in ('rustbpe', 'pyarrow', 'huggingface_hub', 'bes.tokenizer.samples',\n"
        "                   'bes.tokenizer.train_bpe') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print(t.sha256, ids)\n"
    )
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         cwd="/", timeout=120)
    assert res.returncode == 0, res.stderr
    tok = load()
    sha, ids = res.stdout.split(" ", 1)
    assert sha == tok.sha256
    assert json.loads(ids) == tok.encode("Loading check: 12345 <|end|> \u00e9")


# ------------------------------------------------------------------------ ids and bytes
def test_special_ids_equal_config(tok):
    assert tok.vocab_size == config.VOCAB_SIZE
    assert tok.n_ordinary == N_ORDINARY
    for k, name in enumerate(config.SPECIAL_TOKENS):
        assert tok.special_tokens[name] == config.special_id(name) == N_ORDINARY + k
        assert tok.special_name(N_ORDINARY + k) == name
        assert tok.token_bytes(N_ORDINARY + k) == b""
        assert tok.decode([N_ORDINARY + k]) == name
    assert (tok.pad, tok.bos, tok.system, tok.user, tok.assistant, tok.library, tok.end,
            tok.calc, tok.eq, tok.calc_end) == (config.PAD, config.BOS, config.SYSTEM, config.USER,
                                                config.ASSISTANT, config.LIBRARY, config.END,
                                                config.CALC, config.EQ, config.CALC_END)
    assert tok.special_ids == frozenset(range(N_ORDINARY, config.VOCAB_SIZE))
    assert not tok.is_special(0) and not tok.is_special(N_ORDINARY - 1) and tok.is_special(config.BOS)


def test_token_bytes(tok, raw_asset):
    table = tok.all_token_bytes()
    assert len(table) == config.VOCAB_SIZE
    assert table[:256] == [bytes([i]) for i in range(256)]
    assert table[:N_ORDINARY] == ref_token_bytes(raw_asset["merges"])
    assert all(table[i] for i in range(N_ORDINARY))
    assert len(set(table[:N_ORDINARY])) == N_ORDINARY
    assert all(tok.token_bytes(i) == table[i] for i in range(0, config.VOCAB_SIZE, 37))
    lengths = tok.token_byte_lengths()
    assert lengths.dtype == np.int32 and lengths.shape == (config.VOCAB_SIZE,)
    assert lengths.tolist() == [len(b) for b in table]
    with pytest.raises(ValueError):
        tok.token_bytes(config.VOCAB_SIZE)
    with pytest.raises(ValueError):
        tok.token_bytes(-1)


# ------------------------------------------------------------------------------ encoding
def test_round_trip(tok):
    for text in ALL_TEXTS + LONG_TEXTS:
        ids = tok.encode(text)
        assert all(isinstance(i, int) and 0 <= i < N_ORDINARY for i in ids), repr(text[:80])
        assert tok.decode(ids) == text, repr(text[:80])
        assert tok.decode_bytes(ids) == text.encode("utf-8")
    assert tok.encode("") == []


def test_matches_reference_encoder(tok, raw_asset):
    merges, pattern = raw_asset["merges"], raw_asset["pattern"]
    for text in ALL_TEXTS:
        assert tok.encode(text) == ref_encode(text, merges, pattern), repr(text)


def test_matches_reference_encoder_on_samples(tok, raw_asset):
    """Held-out sample text, when data_cache/samples/ has been downloaded."""
    from bes.tokenizer import samples

    texts = []
    for domain in samples.DOMAINS:
        if not all(f.local_path.exists() for f in samples.files_for(domain, "heldout")):
            pytest.skip("samples not downloaded (python scripts/build_tokenizer.py)")
        texts += [t[:3000] for t in samples.iter_texts(domain, "heldout", 150_000)]
    merges, pattern = raw_asset["merges"], raw_asset["pattern"]
    for text in texts:
        assert tok.encode(text) == ref_encode(text, merges, pattern)
        assert tok.decode(tok.encode(text)) == text


def test_special_strings_never_give_special_ids(tok):
    rng = random.Random(7)
    texts = []
    for s in SPECIAL_STRINGS:
        texts += [s, s * 3, f"x{s}y", f" {s} ", f"\n{s}\n", f"{s}{rng.choice(SPECIAL_STRINGS)}"]
    texts.append(" ".join(SPECIAL_STRINGS))
    for text in texts:
        ids = tok.encode(text)
        assert not any(i in tok.special_ids for i in ids), text
        assert max(ids) < N_ORDINARY
        assert tok.decode(ids) == text
    for ids in tok.encode_batch(texts):
        assert max(ids) < N_ORDINARY


def test_encode_batch_equals_encode(tok):
    texts = ALL_TEXTS[:120]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]
    assert tok.encode_batch(texts, num_threads=1) == tok.encode_batch(iter(texts), num_threads=2)
    assert tok.encode_batch([]) == []


def test_numbers_split_in_groups_of_at_most_two_digits(tok):
    rng = random.Random(3)
    for _ in range(200):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 60)))
        ids = tok.encode(digits)
        pieces = [tok.token_bytes(i) for i in ids]
        assert b"".join(pieces) == digits.encode()
        assert all(1 <= len(p) <= 2 and p.isdigit() for p in pieces)
        assert len(ids) >= (len(digits) + 1) // 2
    ids = tok.encode("abc12345def")
    assert all(tok.token_bytes(i).isdigit() or not any(c in b"0123456789" for c in tok.token_bytes(i))
               for i in ids)


def test_decode_behaviour(tok):
    hello = tok.encode("Hello")
    assert tok.decode([config.BOS, config.USER, *hello, config.END]) == "<|bos|><|user|>Hello<|end|>"
    assert tok.decode_bytes([config.BOS, *hello, config.END]) == b"Hello"
    assert tok.decode(np.array(hello, dtype=np.int32)) == "Hello"
    assert tok.decode([]) == ""
    assert tok.decode([0xE2]) == "\ufffd"  # an incomplete UTF-8 sequence is replaced
    with pytest.raises(ValueError):
        tok.decode([config.VOCAB_SIZE])
    with pytest.raises(ValueError):
        tok.decode([-1])
    with pytest.raises(ValueError):
        tok.decode(np.zeros((2, 2), dtype=np.int32))


def test_determinism(tok, raw_asset):
    a = Tokenizer.from_asset(ASSET)
    b = Tokenizer(raw_asset["merges"], raw_asset["pattern"])
    c = pickle.loads(pickle.dumps(tok))
    assert a.sha256 == b.sha256 == c.sha256 == tok.sha256 == raw_asset["sha256"]
    for text in ALL_TEXTS[:80]:
        ref = tok.encode(text)
        assert tok.encode(text) == ref == a.encode(text) == b.encode(text) == c.encode(text)
    assert load() is tok
    with pytest.raises(ValueError):
        Tokenizer(raw_asset["merges"], raw_asset["pattern"], sha256="0" * 64)


def test_bytes_per_token_is_sensible(tok):
    text = ("The library contains short passages from an encyclopedia. Each passage starts with "
            "its title, and the model reads at most one passage before the question it serves. ") * 4
    assert len(text.encode()) / len(tok.encode(text)) > 4.0
