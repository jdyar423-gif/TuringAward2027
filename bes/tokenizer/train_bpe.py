"""Training of the byte-level BPE and writing of the tokenizer asset.

Follows karpathy/nanochat ``nanochat/tokenizer.py`` (MIT licence): ``rustbpe`` learns the merges
from a stream of texts, ``tiktoken`` does the encoding. The split pattern is nanochat's: the
pattern of tiktoken's ``cl100k_base`` encoding with numbers cut into groups of at most two digits
(``\\p{N}{1,2}`` instead of ``\\p{N}{1,3}``).

rustbpe reports its result as tiktoken-style ranks (token bytes -> id). The merge pair of each
id is recovered by running byte-pair encoding on the id's bytes with the earlier ranks only (the
functions ``bpe`` and ``recover_merges`` of the karpathy/minbpe repository, MIT licence), which
is exactly how tiktoken will segment those bytes. ``scripts/build_tokenizer.py`` checks on held-out text that rustbpe's
own encoder and the tiktoken encoder built from the recovered merges agree.

rustbpe parallelises with rayon: set ``RAYON_NUM_THREADS`` before the first call to bound the
number of threads.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from bes.config import N_SPECIAL, SPECIAL_TOKENS, VOCAB_SIZE
from bes.tokenizer.tok import (
    ASSET_DIR,
    ASSET_FILE,
    META_FILE,
    N_BYTES,
    Merge,
    canonical_content,
    content_hash,
    token_bytes_from_merges,
)

# From karpathy/nanochat, nanochat/tokenizer.py (MIT licence).
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

#: Merges of the committed vocabulary: 8192 = 256 bytes + 7920 merges + 16 special tokens.
N_MERGES = VOCAB_SIZE - N_SPECIAL - N_BYTES


def _bpe_split(ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """Byte-pair encode ``token`` using only ranks below ``max_rank`` (lowest rank first)."""
    parts = [bytes([b]) for b in token]
    while True:
        best: tuple[int, int] | None = None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            return parts
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]


def merges_from_ranks(ranks: dict[bytes, int]) -> list[Merge]:
    """Recover ``merges[i] = (left, right)`` (creating id ``256 + i``) from tiktoken-style ranks.

    ``ranks`` must map the 256 single bytes to their own value and the other byte strings to
    the consecutive ids ``256, 257, …``. Raises ``ValueError`` when some id is not the merge of
    two earlier ids under byte-pair encoding.
    """
    by_id = sorted(ranks.items(), key=lambda kv: kv[1])
    if [i for _, i in by_id] != list(range(len(by_id))):
        raise ValueError("ranks are not the consecutive ids 0..n-1")
    for b in range(min(N_BYTES, len(by_id))):
        if by_id[b][0] != bytes([b]):
            raise ValueError(f"id {b} is not the single byte {b}")
    merges: list[Merge] = []
    for token, rank in by_id[N_BYTES:]:
        parts = _bpe_split(ranks, token, rank)
        if len(parts) != 2:
            raise ValueError(f"id {rank} ({token!r}) is not the merge of two earlier ids")
        merges.append((ranks[parts[0]], ranks[parts[1]]))
    return merges


def train_merges(texts: Iterable[str], n_merges: int = N_MERGES, pattern: str = SPLIT_PATTERN,
                 buffer_size: int = 8192):
    """Learn up to ``n_merges`` merges from ``texts`` with rustbpe.

    Returns ``(merges, trainer)``: the merge pairs (fewer than ``n_merges`` only when the text
    runs out of pairs) and the trained ``rustbpe.Tokenizer`` (for agreement checks).
    """
    import rustbpe

    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(iter(texts), N_BYTES + n_merges, buffer_size=buffer_size,
                                pattern=pattern)
    ranks: dict[bytes, int] = {}
    for token, rank in trainer.get_mergeable_ranks():
        token = bytes(token)
        if token in ranks:
            raise ValueError(f"rustbpe produced the byte string {token!r} twice")
        ranks[token] = int(rank)
    merges = merges_from_ranks(ranks)
    if token_bytes_from_merges(merges) != [t for t, _ in sorted(ranks.items(), key=lambda kv: kv[1])]:
        raise AssertionError("recovered merges do not rebuild the rustbpe vocabulary")
    return merges, trainer


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dump_asset(merges: Sequence[Merge], pattern: str = SPLIT_PATTERN,
               special_tokens: Sequence[str] = SPECIAL_TOKENS) -> tuple[str, str]:
    """Serialise an asset as JSON text with one merge per line; returns ``(text, sha256)``."""
    content = canonical_content(merges, pattern, special_tokens)
    token_bytes_from_merges(content["merges"])  # validates that every merge refers to earlier ids
    digest = content_hash(content)
    lines = ["{"]
    for key in ("format", "vocab_size", "n_bytes", "pattern", "special_tokens"):
        lines.append(f"  {json.dumps(key)}: {json.dumps(content[key], ensure_ascii=True)},")
    lines.append(f'  "sha256": "{digest}",')
    lines.append('  "merges": [')
    lines.append(",\n".join(f"    [{a}, {b}]" for a, b in content["merges"]))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n", digest


def write_asset(merges: Sequence[Merge], out_dir: str | Path = ASSET_DIR,
                pattern: str = SPLIT_PATTERN, meta: dict | None = None) -> str:
    """Write ``tokenizer.json`` (and ``meta.json`` if ``meta`` is given) atomically.

    Returns the content hash. The committed asset must have exactly ``N_MERGES`` merges; other
    sizes are accepted for experiments and tests.
    """
    text, digest = dump_asset(merges, pattern)
    _atomic_write(Path(out_dir) / ASSET_FILE, text)
    if meta is not None:
        write_meta(out_dir, {**meta, "sha256": digest})
    return digest


def write_meta(out_dir: str | Path, meta: dict) -> None:
    _atomic_write(Path(out_dir) / META_FILE, json.dumps(meta, indent=2, sort_keys=False) + "\n")
