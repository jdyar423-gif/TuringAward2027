"""Term extraction shared by the index, the retriever, the gate and the tests.

``terms(text)`` = Unicode NFKD, combining marks removed, lower case, runs of letters/digits,
English stopwords removed, then a light plural stemmer (Harman's "S" stemmer). Terms are
mapped to ``2**n_bits`` buckets by the first 8 bytes of BLAKE2b (stable across processes
and platforms, unlike Python's ``hash``).
"""
from __future__ import annotations

import hashlib
import unicodedata
from functools import lru_cache

import regex

_TOKEN_RX = regex.compile(r"[\p{L}\p{N}]+")

#: Function words removed from queries and documents (they carry no retrieval signal and,
#: in a truncated index, would only occupy buckets).
STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all also am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does doesn doing
    don down during each few for from further had hadn has hasn have haven having he her here
    hers herself him himself his how i if in into is isn it its itself just let me more most
    mustn my myself no nor not now of off on once only or other ought our ours ourselves out over
    own same shan she should shouldn so some such than that the their theirs them themselves
    then there these they this those through to too under until up upon very was wasn we were
    weren what when where which while who whom whose why with would wouldn you your yours
    yourself yourselves s t d ll m re ve y ain many much might shall yes get got gets tell
    please""".split()
)
# Kept as terms on purpose: "us" (US), "may" (the month), "will" (Will Smith), "won", "one".


_MARKS_RX = regex.compile(r"\p{M}+")


def normalize(text: str) -> str:
    """NFKD, drop combining marks, lower case (``"Utqiaġvik"`` -> ``"utqiagvik"``)."""
    if text.isascii():
        return text.lower()
    return _MARKS_RX.sub("", unicodedata.normalize("NFKD", text)).lower()


def tokens(text: str) -> list[str]:
    """Lower-case letter/digit runs of the normalised text (stopwords kept)."""
    return _TOKEN_RX.findall(normalize(text))


@lru_cache(maxsize=1 << 20)
def stem(tok: str) -> str:
    """Harman's S-stemmer: ies->y, es->e, s->'' with the usual exceptions."""
    if len(tok) > 3 and tok.endswith("ies") and not tok.endswith(("eies", "aies")):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("es") and not tok.endswith(("aes", "ees", "oes")):
        return tok[:-1]
    if len(tok) > 2 and tok.endswith("s") and not tok.endswith(("us", "ss")):
        return tok[:-1]
    return tok


def terms(text: str) -> list[str]:
    """Index terms of ``text`` in order (with repetitions)."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


@lru_cache(maxsize=1 << 21)
def term_hash(term: str) -> int:
    """64-bit stable hash of a term."""
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "little")


def bucket(term: str, n_bits: int) -> int:
    """Bucket id in ``[0, 2**n_bits)``."""
    return term_hash(term) & ((1 << n_bits) - 1)


def strip_disambiguator(title: str) -> str:
    """``"Mint (candy)"`` -> ``"Mint"``; used for title matching."""
    i = title.find(" (")
    return title[:i] if i > 0 and title.endswith(")") else title
