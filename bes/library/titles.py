"""Exact title look-up with primary topics (``TitleTable``) and person articles.

A head-entity question names an article by a key, the content terms of its object ("Who was
Galileo?" -> "galileo"). The table maps a key to the article a reader most likely means, or
to nothing when the key stays ambiguous (no exact-title credit then):

1. A **plain** title owns its key: no disambiguator, no leading article, no stopword dropped
   and no plural stemmed away ("Black hole" owns "black hole": Wikipedia gives the plain
   title to the primary topic). A disambiguation page owns nothing.
2. Otherwise the most important article whose title contains the key owns it, when it can
   stand for the key and leads every other such article by ``title_margin`` (the importance
   computed at build time, ``bes.library.select.importance``). It can stand for the key when
   it is a **variant** of it (disambiguated "Galileo (spacecraft)", "The Pointer", stemmed
   "Bachs", stopword-dropped "While loop") or a **person** article whose name the key is
   ("Charles Darwin" for "darwin"). Disambiguated siblings of a non-disambiguated title count
   as that title ("The Beatles (album)" does not compete with "The Beatles").
3. **Who-frames** ("Who was Lincoln?") look among persons: the person with that surname or
   common single name who leads every other person with it by ``person_margin``. Here a
   disambiguated person article ("Jefferson (footballer, born 1981)") is ranked without the
   bare-name mentions it shares with its namesakes (``name_importance``). A who-frame never
   gets a non-person owner when some person bears the name ("Who was May?" is not the month).

Person articles: the article uses he/she/his/her at least twice, does not open with
"A/An/The" (a kind, not a person), and its lead has a parenthesis with a year within the
first 150 characters and before the first "is/was" (the subject's life span, not the birth
year of someone in the predicate), or a first sentence "X was/is a ... <occupation>". A
person's names: the last name word (surname) and every other name word the article uses on
its own at least twice ("Galileo" for Galileo Galilei); honorifics, particles, numerals and
initials are not names.

Keys: content terms (``bes.library.terms``) with the S-stemmer applied only to common words.
A title word that the article never writes in lower case is a name and keeps its plural
("Bachs", "The Jeffersons", "Britney Spears"), so "Bach" and "Bachs" stay apart; in a query a
word written with a capital letter is not stemmed (``object_keys``).

Stored (counted): ``hash`` uint32 [K] sorted, ``ref`` uint32 [K] = title id | variant << 31 |
person << 30 | named << 29 (``named``: some person bears this one-word key). Who-frame
entries use a salted hash and exist only where the who-frame answer differs from the general
owner; title id ``DENY`` there means "no person". A hit is verified against the stored title
text, so a hash collision cannot return a wrong article.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import regex

from bes.config import LibraryCfg
from bes.library.pack import bsearch_right, numbers_of
from bes.library.terms import STOPWORDS, stem, strip_disambiguator, term_hash, terms, tokens

_L = LibraryCfg()

TID_MASK = (1 << 29) - 1
DENY = TID_MASK
VARIANT_BIT, PERSON_BIT, NAMED_BIT = 1 << 31, 1 << 30, 1 << 29

#: Words dropped from the object of a query before its key is formed ("what does X mean",
#: "facts about X"); terms of ``bes.library.terms.terms`` (already stemmed).
DEF_WORDS = frozenset("explain describe define definition mean meaning know information info "
                      "overview summary fact give learn".split())
_ARTICLES = frozenset(("a", "an", "the"))
#: Words of a person's title that are not names.
NOT_NAMES = frozenset("""sir lord lady dame saint st pope king queen prince princess emperor empress duke
    duchess count countess baron baroness president general admiral captain dr doctor mr mrs ms miss rev
    reverend father brother sister mother uncle aunt grand great the of de da di del della van von der den
    le la du dos das do bin ibn al el ben jr sr junior senior ii iii iv v vi vii viii ix x xi xii xiii xiv
    xv xvi tsar czar sultan shah khan imam rabbi bishop cardinal archbishop""".split())
_OCCUPATIONS = (
    "philosopher|poet|writer|author|novelist|playwright|dramatist|king|queen|emperor|empress|pharaoh|"
    "prince|princess|sultan|caliph|prophet|saint|pope|bishop|priest|monk|nun|general|admiral|soldier|"
    "warrior|explorer|navigator|inventor|scientist|mathematician|physicist|chemist|biologist|astronomer|"
    "economist|historian|psychologist|engineer|architect|painter|sculptor|artist|composer|musician|"
    "singer|songwriter|rapper|guitarist|pianist|drummer|actor|actress|comedian|director|producer|"
    "filmmaker|dancer|model|journalist|politician|statesman|stateswoman|leader|president|minister|lawyer|"
    "judge|businessman|businesswoman|entrepreneur|athlete|footballer|player|boxer|wrestler|swimmer|"
    "cyclist|golfer|coach|ruler|chief|monarch|revolutionary|activist|thinker|theologian|scholar|teacher|"
    "physician|doctor|surgeon|astronaut|aviator|spy|criminal|gangster|pirate|cartoonist|illustrator|"
    "photographer|designer|chef|personality|socialite|aristocrat|nobleman|noblewoman|duke|duchess|mystic|"
    "missionary|apostle|tsar|czar|shah|khan|chancellor|dictator|conqueror|man|woman|person|figure|"
    "lawgiver|educator|reformer|preacher|evangelist|consort|regent|governor|senator|mayor|diplomat|"
    "commander|officer")
_FIRST_OCC_RX = re.compile(r"^[^.]{0,160}?\b(?:was|is|were)\b(?:\s+[\w'-]+,?){0,7}?\s+(?:" + _OCCUPATIONS
                           + r")s?\b", re.I)
_SPAN_RX = re.compile(r"\(([^()]*)\)")
_YEARISH_RX = re.compile(r"\b\d{3,4}\b|\b\d{1,4}\s*(?:BC|BCE|AD|CE)\b|\bborn\b")
_PRONOUN_RX = re.compile(r"\b(?:he|she|his|her|him|himself|herself)\b", re.I)
_PREDICATE_RX = re.compile(r"\b(?:is|was|are|were|has|had)\b")
_GENERIC_RX = re.compile(r"^(?:A|An|The)\s")
_DISAMBIGUATION_RX = re.compile(r"\b(?:may|can|might|could) (?:refer to|mean)\b|\bhas (?:more than one|several|"
                                r"many|multiple|two|different) meanings?\b|\bdisambiguation\b", re.I)
_CASE_TOKEN_RX = regex.compile(r"[\p{L}\p{N}]+")
_MARKS_RX = regex.compile(r"\p{M}+")


def case_tokens(text: str) -> list[str]:
    """Letter/digit runs of ``text`` after NFKD and mark removal, case kept."""
    if not text.isascii():
        text = _MARKS_RX.sub("", unicodedata.normalize("NFKD", text))
    return _CASE_TOKEN_RX.findall(text)


def key_hash(key: str) -> int:
    return term_hash("\x00title\x00" + key) & 0xFFFFFFFF


def who_hash(key: str) -> int:
    return term_hash("\x00who\x00" + key) & 0xFFFFFFFF


def title_key(title: str, names: frozenset[str] | set[str] = frozenset()) -> str:
    """Content terms of a title without its disambiguator ("Mercury (planet)" -> "mercury");
    words in ``names`` keep their plural."""
    return " ".join(w if w in names else stem(w) for w in tokens(strip_disambiguator(title))
                    if w not in STOPWORDS)


def raw_title_key(title: str) -> str:
    """``title_key`` without stemming."""
    return " ".join(w for w in tokens(strip_disambiguator(title)) if w not in STOPWORDS)


def surface(title: str) -> list[str]:
    """Tokens of a title without its disambiguator and a leading article (stopwords kept)."""
    toks = tokens(strip_disambiguator(title))
    return toks[1:] if len(toks) > 1 and toks[0] in _ARTICLES else toks


def object_keys(text: str) -> list[str]:
    """Keys to try for the object of a query, best first: words written with a capital letter
    unstemmed and the others stemmed, then every word unstemmed. Definition words are dropped."""
    content = []
    for w in case_tokens(text):
        lw = w.lower()
        if lw in STOPWORDS:
            continue
        s = stem(lw)
        if s in DEF_WORDS or lw in DEF_WORDS:
            continue
        content.append((lw, s, w[:1].isupper()))
    mixed = " ".join(lw if cap else s for lw, s, cap in content)
    raw = " ".join(lw for lw, _, _ in content)
    return [k for k in dict.fromkeys((mixed, raw)) if k]


def article_texts(passages: Sequence[tuple[str, str]], n_passages: int = 3) -> list[str]:
    """The first ``n_passages`` bodies of every title, joined by newlines (passages of one
    title contiguous, titles in order of appearance)."""
    out: list[list[str]] = []
    last = None
    for title, body in passages:
        if title != last:
            out.append([])
            last = title
        if len(out[-1]) < n_passages:
            out[-1].append(body)
    return ["\n".join(x) for x in out]


# ------------------------------------------------------------------ article analysis


@dataclass(frozen=True)
class TitleInfo:
    """What the build knows about one title (from its title and the text of its article)."""

    key: str
    plain: bool
    person: bool
    disambiguation_page: bool
    names: tuple[str, ...]   # a person's names (surname first); () for other articles


def is_person(text: str) -> bool:
    """Whether an article (its first passages) is about one person (module docstring): the
    year parenthesis must belong to the subject ("Galileo Galilei (1564 – 1642) was ..."),
    not to someone named in the predicate ("Gravity is a song by Sara Bareilles (born 1979)")."""
    if not text or len(_PRONOUN_RX.findall(text)) < 2 or _GENERIC_RX.match(text):
        return False
    for m in _SPAN_RX.finditer(text[:400]):
        if m.start() > 150 or _PREDICATE_RX.search(text[:m.start()]):
            break
        if _YEARISH_RX.search(m.group(1)):
            return True
    first = text.split("\n", 1)[0]
    return bool(_FIRST_OCC_RX.search(first))


def lowercase_words(text: str) -> set[str]:
    """Words the text writes in lower case (accents removed)."""
    return {w for w in case_tokens(text) if w[:1].islower()}


def person_names(title: str, text: str) -> tuple[str, ...]:
    """The surname (last name word) and the other name words the article uses on their own
    (not next to another name word) at least twice, the most used first (ties: surname)."""
    words = [w for w in tokens(strip_disambiguator(title))
             if w not in NOT_NAMES and len(w) > 1 and not any(c.isdigit() for c in w)]
    if not words:
        return ()
    others = set(words)
    body = [w.lower() for w in case_tokens(text)]
    alone = dict.fromkeys(words, 0)
    for i, t in enumerate(body):
        if t in alone and not (i + 1 < len(body) and body[i + 1] in others) and not (i > 0 and body[i - 1] in others):
            alone[t] += 1
    surname = words[-1]
    names = [surname] + [w for w in dict.fromkeys(words[:-1]) if w != surname and alone[w] >= 2]
    return tuple(sorted(names, key=lambda w: (-alone[w], w != surname)))


def analyse(title: str, text: str | None) -> TitleInfo:
    """``TitleInfo`` of one title; ``text`` is the article's first passages (None: unknown,
    then nothing is a person and every word is stemmed)."""
    person = bool(text) and is_person(text)
    dpage = bool(text) and bool(_DISAMBIGUATION_RX.search(text[:200]))
    names: set[str] = set()
    if text:
        lower = lowercase_words(text)
        for w in tokens(strip_disambiguator(title)):
            if w not in STOPWORDS and stem(w) != w and (person or w not in lower):
                names.add(w)
    key = title_key(title, names)
    toks = tokens(strip_disambiguator(title))
    plain = strip_disambiguator(title) == title and all(w not in STOPWORDS for w in toks) and " ".join(toks) == key
    return TitleInfo(key, plain, person, dpage, person_names(title, text) if person else ())


def analyse_library(lib, n_passages: int = 3) -> list[TitleInfo]:
    """``analyse`` of every title of a built library (``bes.library.retrieve.Library``) from
    its stored passages, as the build analysed them."""
    titles = lib.packed.all_titles()
    tf = np.asarray(lib.packed.title_first, dtype=np.int64)
    first = {int(tf[t]): t for t in range(len(titles))}
    texts: list[list[str]] = [[] for _ in titles]
    tid = -1
    for pid, body in lib.packed.iter_bodies():
        tid = first.get(pid, tid)
        if len(texts[tid]) < n_passages:
            texts[tid].append(body)
    return [analyse(t, "\n".join(x)) for t, x in zip(titles, texts)]


# ------------------------------------------------------------------ the table


class TitleTable:
    """Exact title look-up with primary topics (module docstring)."""

    ARRAYS = ("hash", "ref")

    def __init__(self, hashes: np.ndarray, refs: np.ndarray):
        self.hash = hashes
        self.ref = refs

    @classmethod
    def build(cls, titles: Sequence[str], importance: Sequence[float] | None = None,
              texts: Sequence[str] | None = None, name_importance: Sequence[float] | None = None,
              title_margin: float = _L.title_margin, person_margin: float = _L.person_margin,
              infos: Sequence[TitleInfo] | None = None) -> "TitleTable":
        """Keys of every title (``texts[tid]``: the article's first passages; None: titles
        only). ``importance`` ranks articles for general keys, ``name_importance`` (default
        ``importance``) ranks persons for who-frames."""
        n = len(titles)
        imp = np.ones(n) if importance is None else np.asarray(importance, dtype=np.float64)
        nimp = imp if name_importance is None else np.asarray(name_importance, dtype=np.float64)
        if infos is None:
            infos = [analyse(t, None if texts is None else texts[i]) for i, t in enumerate(titles)]
        by_key: dict[str, list[int]] = defaultdict(list)
        contains: dict[str, set[int]] = defaultdict(set)
        named: dict[str, list[int]] = defaultdict(list)
        nondis = {t: i for i, t in enumerate(titles)
                  if strip_disambiguator(t) == t and not infos[i].disambiguation_page}
        collapsed = np.array([strip_disambiguator(t) != t and strip_disambiguator(t) in nondis
                              for t in titles], dtype=bool)
        for tid, info in enumerate(infos):
            if info.key:
                by_key[info.key].append(tid)
            if info.disambiguation_page or collapsed[tid]:
                continue
            for w in set(info.key.split()):
                if info.person and w not in info.names:
                    continue                     # a person stands only for its names
                contains[w].add(tid)
            for w in info.names:
                named[w].append(tid)
        entries: dict[int, tuple[float, int, int]] = {}     # hash -> (importance, ref)

        def put(h: int, tid: int, flags: int) -> None:
            cur = entries.get(h)
            val = (float(imp[tid]) if tid != DENY else -1.0, -tid, tid | flags)
            if cur is None or val > cur:                       # a 32-bit collision: keep one
                entries[h] = val

        general: dict[str, int] = {}
        for k in sorted(set(by_key) | set(named)):
            owner = cls._owner(k, by_key.get(k, ()), infos, contains, named, imp, title_margin)
            is_named = len(k.split()) == 1 and bool(named.get(k))
            if owner is not None:
                info = infos[owner]
                flags = (0 if (info.plain and info.key == k) else VARIANT_BIT) | \
                        (PERSON_BIT if info.person else 0) | (NAMED_BIT if is_named else 0)
                put(key_hash(k), owner, flags)
                general[k] = owner
            if is_named:
                g = general.get(k)
                if g is not None and infos[g].person and infos[g].plain:
                    lead = g                     # "Aristotle", "Rihanna": the plain title is the person
                else:
                    ps = sorted(set(named[k]), key=lambda t: (-nimp[t], t))
                    lead = ps[0] if len(ps) == 1 or nimp[ps[0]] - nimp[ps[1]] >= person_margin else DENY
                default = g if g is not None and infos[g].person else DENY
                if lead != default:
                    put(who_hash(k), lead, PERSON_BIT)
        hs = np.array(sorted(entries), dtype=np.uint32)
        refs = np.array([entries[int(h)][2] for h in hs], dtype=np.uint32)
        return cls(hs, refs)

    @staticmethod
    def _owner(k: str, same_key: Sequence[int], infos, contains, named, imp, margin: float) -> int | None:
        plains = [t for t in same_key if infos[t].plain and not infos[t].disambiguation_page]
        if len(plains) == 1:
            return plains[0]
        words = k.split()
        if len(plains) > 1:
            pool = set(plains)
        else:
            sets = [contains.get(w, set()) for w in words]
            pool = set.intersection(*sets) if sets else set()
        if not pool:
            return None
        ranked = sorted(pool, key=lambda t: (-imp[t], t))
        best = ranked[0]
        if len(ranked) > 1 and imp[best] - imp[ranked[1]] < margin:
            return None
        info = infos[best]
        if info.key == k or (len(words) == 1 and k in info.names):
            return best
        return None

    def numbers(self) -> int:
        return numbers_of(self.hash.nbytes) + numbers_of(self.ref.nbytes)

    def _find(self, h: int, log) -> int | None:
        if self.hash.shape[0] == 0:
            return None
        lo, probes = bsearch_right(self.hash, h)
        if log is not None:
            log.title_lookup += 2 * len(set(probes))
        i = lo - 1
        if i < 0 or int(self.hash[i]) != h:
            return None
        if log is not None:
            log.title_lookup += 2
        return int(self.ref[i])

    def lookup(self, key: str, log=None, who: bool = False) -> tuple[int, bool, bool] | None:
        """``(title id, variant, person)`` of a key, None when the table has no entry for it
        (binary search; reads the probed hashes). Title id -1: the key has an entry but no
        answer (``who``: a person bears this name and no person article leads, or the owner
        is not a person)."""
        if not key:
            return None
        if who:
            r = self._find(who_hash(key), log)
            if r is not None:
                tid = r & TID_MASK
                return (-1, True, True) if tid == DENY else (tid, True, True)
        r = self._find(key_hash(key), log)
        if r is None:
            return None
        tid, variant, person, is_named = r & TID_MASK, bool(r & VARIANT_BIT), bool(r & PERSON_BIT), bool(r & NAMED_BIT)
        if who and is_named and not person:
            return -1, variant, person       # a who-frame: a person bears this name, the owner is not one
        return tid, variant, person

    @staticmethod
    def lookups_per_query() -> int:
        """Binary searches one query can make: who-frame and general keys, two forms each."""
        return 4


def verify(key: str, title: str) -> bool:
    """Whether ``key`` can name ``title`` (rejects hash collisions): every key term is a term of
    the title, stemmed or not."""
    words = set(title_key(title).split()) | set(raw_title_key(title).split())
    return bool(key) and set(key.split()) <= words
