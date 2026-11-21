"""Byte-level BPE tokenizer with 8192 ids, built on ``tiktoken.Encoding``.

Id layout (``docs/SPEC.md`` section 1): ids ``0..255`` are the raw bytes (id ``b`` is the single
byte ``b``), ids ``256..8175`` are merges in rank order, and the last ``N_SPECIAL = 16`` ids
are the special tokens of ``bes.config.SPECIAL_TOKENS``. Ordinary encoding never produces a
special id, even when the text contains the string of a special token.

The asset ``assets/tokenizer/tokenizer.json`` stores the split pattern, the special token names
and the merges as pairs ``[left, right]`` (merge ``i`` creates id ``256 + i`` with bytes
``bytes(left) + bytes(right)``), plus a SHA-256 of that content which ``load`` verifies.
Loading needs nothing but this file.
"""
from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from bes.config import N_SPECIAL, SPECIAL_TOKENS, VOCAB_SIZE, special_id

ASSET_FORMAT = "bes-bpe/1"
N_BYTES = 256
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "tokenizer"
ASSET_FILE = "tokenizer.json"
META_FILE = "meta.json"

Merge = tuple[int, int]


# ----------------------------------------------------------------------------- asset format
def canonical_content(merges: Sequence[Merge], pattern: str,
                      special_tokens: Sequence[str] = SPECIAL_TOKENS) -> dict:
    """The hashed part of the asset as a plain dict (lists of ints and strings only)."""
    merges = [[int(a), int(b)] for a, b in merges]
    return {
        "format": ASSET_FORMAT,
        "vocab_size": N_BYTES + len(merges) + len(special_tokens),
        "n_bytes": N_BYTES,
        "pattern": pattern,
        "special_tokens": list(special_tokens),
        "merges": merges,
    }


def content_hash(content: dict) -> str:
    """SHA-256 (hex) of the canonical JSON serialisation of ``canonical_content(...)``."""
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def token_bytes_from_merges(merges: Iterable[Merge]) -> list[bytes]:
    """Byte string of every ordinary id: ``256`` single bytes followed by one entry per merge."""
    out = [bytes([i]) for i in range(N_BYTES)]
    for a, b in merges:
        if not (0 <= a < len(out) and 0 <= b < len(out)):
            raise ValueError(f"merge {len(out) - N_BYTES} refers to an id that does not exist yet")
        out.append(out[a] + out[b])
    return out


def read_asset(asset_dir: str | Path = ASSET_DIR) -> dict:
    """Read and validate ``tokenizer.json``; raises ``ValueError`` on any inconsistency."""
    path = Path(asset_dir) / ASSET_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != ASSET_FORMAT:
        raise ValueError(f"{path}: unknown format")
    try:
        merges = [(int(a), int(b)) for a, b in raw["merges"]]
        content = canonical_content(merges, str(raw["pattern"]), [str(s) for s in raw["special_tokens"]])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{path}: malformed asset ({err})") from err
    for key in ("vocab_size", "n_bytes"):
        if raw.get(key) != content[key]:
            raise ValueError(f"{path}: field {key}={raw.get(key)} disagrees with the merges")
    digest = content_hash(content)
    if raw.get("sha256") != digest:
        raise ValueError(f"{path}: content hash mismatch (file says {raw.get('sha256')}, content is {digest})")
    content["sha256"] = digest
    return content


# ------------------------------------------------------------------------------- tokenizer
class Tokenizer:
    """Encoder/decoder for one BPE vocabulary.

    ``merges[i] = (left, right)`` creates id ``256 + i``; specials follow the merges, so the
    special id of ``SPECIAL_TOKENS[k]`` is ``vocab_size - N_SPECIAL + k`` (equal to the
    constants of ``bes.config`` for the 8192-id asset).
    """

    def __init__(self, merges: Sequence[Merge], pattern: str,
                 special_tokens: Sequence[str] = SPECIAL_TOKENS, sha256: str | None = None):
        import tiktoken

        if tuple(special_tokens) != SPECIAL_TOKENS:
            raise ValueError("special tokens must be bes.config.SPECIAL_TOKENS in that order")
        self.merges: tuple[Merge, ...] = tuple((int(a), int(b)) for a, b in merges)
        self.pattern = pattern
        content = canonical_content(self.merges, pattern, special_tokens)
        self.sha256 = content_hash(content)
        if sha256 is not None and sha256 != self.sha256:
            raise ValueError("content hash does not match the given merges and pattern")
        self.vocab_size: int = content["vocab_size"]
        self.n_ordinary: int = self.vocab_size - N_SPECIAL

        ordinary = token_bytes_from_merges(self.merges)
        ranks = {tb: i for i, tb in enumerate(ordinary)}
        if len(ranks) != len(ordinary):
            raise ValueError("two ids have the same byte string; the merges are not a valid BPE")
        self.special_tokens: dict[str, int] = {
            name: special_id(name, self.vocab_size) for name in SPECIAL_TOKENS
        }
        self._special_names: dict[int, str] = {i: n for n, i in self.special_tokens.items()}
        self.special_ids: frozenset[int] = frozenset(self._special_names)
        self._bytes: tuple[bytes, ...] = tuple(ordinary) + (b"",) * N_SPECIAL
        self._enc = tiktoken.Encoding(
            name=f"bes-bpe-{self.vocab_size}",
            pat_str=pattern,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
            explicit_n_vocab=self.vocab_size,
        )
        (self.pad, self.bos, self.system, self.user, self.assistant, self.library, self.end,
         self.calc, self.eq, self.calc_end) = (self.special_tokens[n] for n in SPECIAL_TOKENS[:10])

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_asset(cls, asset_dir: str | Path = ASSET_DIR) -> "Tokenizer":
        content = read_asset(asset_dir)
        return cls(content["merges"], content["pattern"], content["special_tokens"], content["sha256"])

    def __reduce__(self):
        return (Tokenizer, (self.merges, self.pattern, SPECIAL_TOKENS, self.sha256))

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, sha256={self.sha256[:12]})"

    # -- encoding -----------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Ordinary encoding of ``text``: ids in ``[0, n_ordinary)``, never a special id."""
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: Iterable[str], num_threads: int = 2) -> list[list[int]]:
        """``[encode(t) for t in texts]`` computed with ``num_threads`` threads."""
        return self._enc.encode_ordinary_batch(list(texts), num_threads=num_threads)

    def decode(self, ids: Iterable[int]) -> str:
        """Concatenated bytes of ``ids`` decoded as UTF-8 (invalid sequences become U+FFFD).

        A special id is rendered as its name, e.g. ``"<|end|>"``; ``bes.data.chat_format.
        display_text`` gives the user-facing text instead.
        """
        arr = np.asarray(ids if isinstance(ids, np.ndarray) else list(ids), dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("decode expects a 1-D sequence of ids")
        if arr.size and (arr.min() < 0 or arr.max() >= self.vocab_size):
            raise ValueError(f"ids outside [0, {self.vocab_size})")
        return self._enc.decode(arr.tolist(), errors="replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Concatenated byte strings of ``ids`` (specials contribute nothing)."""
        return b"".join(self.token_bytes(i) for i in ids)

    # -- vocabulary ---------------------------------------------------------------------
    def token_bytes(self, token_id: int) -> bytes:
        """Byte string of one id; ``b""`` for special ids."""
        token_id = int(token_id)
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"id {token_id} outside [0, {self.vocab_size})")
        return self._bytes[token_id]

    def all_token_bytes(self) -> list[bytes]:
        """Byte strings of all ``vocab_size`` ids in id order (``b""`` for specials)."""
        return list(self._bytes)

    def token_byte_lengths(self) -> np.ndarray:
        """``int32[vocab_size]`` number of bytes of every id (0 for specials), for bits/byte."""
        return np.array([len(b) for b in self._bytes], dtype=np.int32)

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self.special_ids

    def special_name(self, token_id: int) -> str | None:
        """Name of a special id (``"<|bos|>"``…) or ``None`` for an ordinary id."""
        return self._special_names.get(int(token_id))


@functools.lru_cache(maxsize=8)
def _load_cached(asset_dir: str) -> Tokenizer:
    return Tokenizer.from_asset(asset_dir)


def load(asset_dir: str | Path | None = None) -> Tokenizer:
    """Load the committed tokenizer (``assets/tokenizer/`` by default); cached per directory.

    The default asset must have exactly ``bes.config.VOCAB_SIZE`` ids.
    """
    path = Path(asset_dir) if asset_dir is not None else ASSET_DIR
    tok = _load_cached(str(path.resolve()))
    if asset_dir is None and tok.vocab_size != VOCAB_SIZE:
        raise ValueError(f"committed tokenizer has {tok.vocab_size} ids, config says {VOCAB_SIZE}")
    return tok
