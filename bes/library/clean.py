"""Cleaning of Simple English Wikipedia article text and splitting into passages.

The input is the ``text`` column of the Hugging Face dataset ``wikimedia/wikipedia``
(config ``20231101.simple``): plain text in which paragraphs are separated by blank lines,
section headings are short lines of their own, list items start with a space, and the
category names of the article follow at the very end. The extraction left some debris:
raw templates and tables, ``<ref>`` tags, ``'''`` bold marks, empty or half-empty
parentheses where pronunciation templates were removed (``Austria (, ;  ), officially``),
holes where unit templates were removed (``The trail is about  long.``) and the
"References / Other websites / Related pages" tails.

The rules below were chosen by measuring the whole dump (counts in ``docs/E1_COVERAGE.md``):

* markup: HTML comments, ``<ref>`` / ``<math>`` / ``<gallery>`` blocks, other tags, balanced
  ``{{templates}}`` and ``{| tables |}``, wiki links (``[[a|b]]`` -> ``b``; file, image and
  category links dropped), external links, bold/italic quote runs, HTML entities; lines
  that are table rows (start with ``|``, ``!``, ``{|``, or contain ``||`` and are mostly
  non-letters) and image caption remnants (``thumb|200px|...``) are dropped;
* the last line group is dropped when every line looks like a category name (it is kept
  separately: the first category is a good sort key that puts similar articles next to
  each other, which helps the block compressor);
* everything from the first tail heading (``References``, ``Other websites``,
  ``Related pages``, ``Sources``, ``Notes``, ...) to the end is dropped;
* parentheses: every ``;``/``,``-separated part that is empty, a bare label (``Irish:``),
  contains IPA symbols or a pronunciation marker is removed; a parenthesis left empty is
  removed with its leading space;
* reference marks (``[1]``, ``[12]``, ``[citation needed]``) are removed; spaces before
  ``,`` ``.`` ``;`` ``:`` and inside brackets are removed; runs of spaces collapse;
* a sentence that ends in a dangling function word (``... is .``, ``... is a .``) is a
  removed template and is dropped;
* a sentence (or list item) with a hole left by a removed unit/number template
  (``It is about long.``, ``an area of, making it``, ``from to 2020``, ``between and wide``)
  is dropped (``hole_repair``); an approximator stranded before a compass direction
  (``about north of Munich``) is removed instead;
* list items of fewer than ``min_item_words`` words (navigation lines) are dropped; when
  list words exceed ``list_heavy_ratio`` times the prose words the lists are dropped
  altogether (date, year and month pages);
* dropped pages: ``(disambiguation)`` titles and pages whose lead ends in "may refer to:" /
  "can mean:" with mostly list content; ``List of ...`` pages with more list than prose;
  pages left with fewer than ``min_article_words`` words or without one complete sentence.

Passages (``make_passages``) never exceed ``max_words`` words and ``max_tokens`` tokens
(title line included, measured with the given ``count_tokens``; by default the model's BPE
tokenizer, ``bpe_count_tokens``), start at a paragraph or sentence boundary, keep every
source line as its own line (list items as ``- item``) and begin with the section heading
when they are not part of the lead.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

from bes.config import LibraryCfg

_L = LibraryCfg()

# ------------------------------------------------------------------------------ helpers


def proxy_count_tokens(text: str) -> float:
    """Token-count proxy used before the BPE tokenizer existed: UTF-8 bytes / 3.3."""
    return len(text.encode("utf-8")) / 3.3


_ENCODE: Callable[[str], list[int]] | None = None


def bpe_count_tokens(text: str) -> int:
    """Number of ids of ``bes.tokenizer.load().encode(text)`` (the model's own tokenizer).

    The tokenizer is loaded on the first call in each process (about 0.5 s, cached), so
    importing this module stays cheap; the function is picklable (module level)."""
    global _ENCODE
    if _ENCODE is None:
        from bes.tokenizer import load

        _ENCODE = load().encode
    return len(_ENCODE(text))


#: Token counter used by default everywhere in the library (passage cap, rendering).
default_count_tokens = bpe_count_tokens

CountTokens = Callable[[str], float]


def count_tokens_name(count_tokens: "CountTokens") -> str:
    """Stable name of a token counter for metadata (``"bpe"``, ``"utf8_bytes/3.3"``)."""
    if count_tokens is bpe_count_tokens:
        return "bpe"
    if count_tokens is proxy_count_tokens:
        return "utf8_bytes/3.3"
    return getattr(count_tokens, "__name__", "custom")

#: Headings whose section and everything after it are dropped (lower case, no colon).
TAIL_HEADINGS: frozenset[str] = frozenset(
    {
        "references", "reference", "other websites", "other website", "related pages",
        "related page", "external links", "external link", "see also", "sources", "source",
        "notes", "footnotes", "citations", "further reading", "notes and references",
        "references and notes", "official website", "official websites", "official site",
        "gallery", "bibliography", "other websites and references", "references and sources",
        "websites", "links", "other links", "related articles", "more reading", "notes and sources",
        "sources and references", "references and further reading", "further information",
    }
)

# IPA letters and suprasegmentals that never occur in ordinary English prose.
_IPA_CHARS = (
    "ɐɑɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ"
    "ǀǁǂˈˌːˑ̃̈"
)
_IPA_RX = re.compile("[" + re.escape(_IPA_CHARS) + "]")
_PRON_RX = re.compile(
    r"\b(pronounced|pronunciation|pron\.|IPA|listen|say:|said as|rhymes with)\b", re.I
)
_SLASH_IPA_RX = re.compile(r"(^|\s)/[^/\s][^/]*/(\s|$)")

_COMMENT_RX = re.compile(r"<!--.*?-->", re.S)
_REF_SELF_RX = re.compile(r"<ref\b[^<>]*/>", re.I)
_REF_BLOCK_RX = re.compile(r"<ref\b[^<>]*>.*?</ref\s*>", re.I | re.S)
_REF_OPEN_RX = re.compile(r"<ref\b[^\n]*", re.I)
_DROP_BLOCK_RX = re.compile(
    r"<(math|gallery|timeline|score|chem|ce|syntaxhighlight|source|imagemap|graph|mapframe)"
    r"\b[^<>]*>.*?</\1\s*>",
    re.I | re.S,
)
_TAG_RX = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>\n]*)?/?>")
_FILE_LINK_START_RX = re.compile(r"\[\[\s*(File|Image|Category|Media|Datei|Fichier)\s*:", re.I)
_LINK_RX = re.compile(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]")
_LINK_PLAIN_RX = re.compile(r"\[\[([^\[\]|]*)\]\]")
_EXT_LINK_RX = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_URL_RX = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_QUOTES_RX = re.compile(r"'{2,}")
_MAGIC_RX = re.compile(r"__[A-Z]+__")
_ENTITY_RX = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_EQ_HEADING_RX = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
_REFMARK_RX = re.compile(
    r"\[\s*(\d{1,3}|[a-z]|note\s*\d+|nb\s*\d+|citation needed|clarification needed|"
    r"when\?|who\?|which\?|according to whom\?|dubious[^\]]{0,30}|update needed|"
    r"failed verification|better source needed|verification needed|page needed|"
    r"needs update|not in citation given)\s*\]",
    re.I,
)
_SPACE_BEFORE_PUNCT_RX = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN_RX = re.compile(r"([(\[])\s+")
_MULTISPACE_RX = re.compile(r"[ \t  -   　]+")
_DUP_PUNCT_RX = re.compile(r"([,;:])(\s*[,;:])+")
_ZERO_WIDTH_RX = re.compile("[​‌‍‎‏⁠﻿­]")
_BROKEN_END_RX = re.compile(
    r"(?:^|\s)(the|an|of|is|was|are|were|than|approximately|about|between|and|or|at|from|by"
    r"|(?:is|was|are|were) an?|of only|weighs|weighing|measures|measuring|covering"
    r"|totaling|totalling|reaching|with an? (?:area|length|height|elevation) of)\s*[.;]$"
)
# Holes left by removed unit/number templates ("It is about long.", "an area of, making",
# "from to 2020", "between and wide"): the sentence is dropped. Measured on the whole dump
# (``docs/E1_COVERAGE.md``); every pattern needs a measurement word directly after a word
# that requires a number, so ordinary prose ("unemployment is high") is not matched.
_MEASURE = (r"(?:long|high|tall|wide|deep|thick|across|away|in (?:length|diameter|area|size|"
            r"height|width|depth)|above sea level|below sea level)")
_HOLE_RXS = (
    re.compile(r"\b(?:about|approximately|around|nearly|almost|roughly|between|anywhere from)\s+"
               + _MEASURE + r"\b"),
    re.compile(r"(?:^|(?<=[.;] ))(?:It|it|This|Its \w+|The \w+(?: \w+)?|They|Which|which)\s+"
               r"(?:is|was|are|were)\s+(?:long|tall|wide|deep|thick|above sea level)\s*(?:[.;]|$)"),
    re.compile(r"(?:^|(?<=[.;] ))(?:It|it)\s+(?:is|was)\s+(?:high|across)\s*(?:[.;]|$)"),
    re.compile(r"\b(?:It|it|which|that)\s+(?:is|was)\s+(?:long|tall|wide|deep)\s+and\b"),
    re.compile(r"\b(?:area|elevation|height|length|altitude|depth|distance|width|diameter|radius|"
               r"wingspan|weight|mass|size|volume|capacity|span|draft|draught) of\s*"
               r"(?:[.,;)]|$|(?:and|or|with|which|in|at|making|but|is|was|about)\b)"),
    re.compile(r"\bfrom\s+to\b|\bbetween\s+and\b|\bfrom\s*[;:]\s*$|\branges? from\s*[.,;]"),
)
_HOLE_RX = re.compile("|".join(f"(?:{rx.pattern})" for rx in _HOLE_RXS))
# Repaired instead of dropped: an approximator left in front of a compass direction whose
# distance was removed ("about north of Munich" -> "north of Munich").
_APPROX_DIR_RX = re.compile(
    r"\b(?:about|approximately|around|roughly|nearly|some)\s+(?=(?:north|south|east|west|"
    r"northeast|northwest|southeast|southwest|north-east|north-west|south-east|south-west)"
    r"(?:ern)?\b)")


def hole_repair(sentence: str) -> tuple[str | None, int]:
    """(repaired sentence or None to drop it, number of in-place repairs).

    Drops a sentence with a hole left by a removed unit template (``_HOLE_RXS``) and removes
    an approximator stranded before a compass direction (``_APPROX_DIR_RX``)."""
    if _HOLE_RX.search(sentence):
        return None, 0
    return _APPROX_DIR_RX.subn("", sentence)
_LIST_MARK_RX = re.compile(r"^\s*(?:[*#•·]+|[-–—](?=\s)|\d{1,3}[.)](?=\s))\s*")
_WORD_RX = re.compile(r"\S+")
_TERMINAL_RX = re.compile(r"[.!?][\"'”’)\]]*$")
_ENDS_RX = re.compile(r"[.!?:][\"'”’)\]]*$")  # sentence end or a colon introducing a list


def n_words(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


# ------------------------------------------------------------------------- markup layer


def _remove_balanced(text: str, open_tok: str, close_tok: str) -> str:
    """Remove every balanced ``open_tok ... close_tok`` span (nesting allowed).

    An unclosed opener removes the rest of its paragraph (up to the next blank line).
    """
    if open_tok not in text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    lo, lc = len(open_tok), len(close_tok)
    while i < n:
        j = text.find(open_tok, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth = 0
        k = j
        while k < n:
            if text.startswith(open_tok, k):
                depth += 1
                k += lo
            elif text.startswith(close_tok, k):
                depth -= 1
                k += lc
                if depth == 0:
                    break
            else:
                k += 1
        if depth != 0:  # unclosed: drop to the end of the paragraph
            end = text.find("\n\n", j)
            k = n if end < 0 else end
        i = k
    return "".join(out)


def _remove_file_links(text: str) -> str:
    """Remove ``[[File:...]]``, ``[[Image:...]]`` and ``[[Category:...]]`` (nested links ok)."""
    while True:
        m = _FILE_LINK_START_RX.search(text)
        if m is None:
            return text
        depth = 0
        k = m.start()
        n = len(text)
        while k < n:
            if text.startswith("[[", k):
                depth += 1
                k += 2
            elif text.startswith("]]", k):
                depth -= 1
                k += 2
                if depth == 0:
                    break
            else:
                k += 1
        if depth != 0:
            end = text.find("\n", m.start())
            k = n if end < 0 else end
        text = text[: m.start()] + text[k:]


def strip_markup(text: str) -> str:
    """Remove wiki/HTML debris and normalise whitespace characters (keeps line structure)."""
    text = text.replace("\r", "")
    text = _ZERO_WIDTH_RX.sub("", text)
    text = text.replace("\t", " ").replace(" ", " ")
    if "<" in text:
        text = _COMMENT_RX.sub("", text)
        text = _REF_SELF_RX.sub("", text)
        text = _REF_BLOCK_RX.sub("", text)
        text = _REF_OPEN_RX.sub("", text)
        text = _DROP_BLOCK_RX.sub("", text)
        text = _TAG_RX.sub("", text)
    if "{{" in text:
        text = _remove_balanced(text, "{{", "}}")
    if "{|" in text:
        text = _remove_balanced(text, "{|", "|}")
    if "[[" in text:
        text = _remove_file_links(text)
        for _ in range(3):  # nested plain links
            text = _LINK_RX.sub(r"\2", text)
            text = _LINK_PLAIN_RX.sub(r"\1", text)
    if "[" in text:
        text = _EXT_LINK_RX.sub(lambda m: m.group(1) or "", text)
    if "//" in text or "www." in text:
        text = _URL_RX.sub("", text)
    if "''" in text:
        text = _QUOTES_RX.sub("", text)
    if "__" in text:
        text = _MAGIC_RX.sub("", text)
    if "&" in text and _ENTITY_RX.search(text):
        text = html.unescape(text).replace(" ", " ")
    return text


def _is_table_or_image_line(s: str) -> bool:
    """Table rows / image caption remnants left by the extraction (``s`` is stripped)."""
    if not s:
        return False
    if s[0] in "|!" or s.startswith("{|") or s.startswith("|}"):
        return True
    if "||" in s and sum(c.isalpha() for c in s) < 0.5 * len(s):  # digit-heavy table row
        return True
    low = s.lower()
    if low.startswith(("thumb|", "thumbnail|", "frame|", "frameless|", "upright")):
        return True
    if re.search(r"\b\d+\s*px\s*\|", low) or re.search(r"\|\s*\d+\s*px\b", low):
        return True
    if ("style=" in low or "class=" in low or "colspan=" in low or "rowspan=" in low
            or "align=" in low or "width=" in low):
        return True
    return False


# ------------------------------------------------------------------------- inline layer


def _drop_part(part: str) -> bool:
    """Whether one ``;``/``,`` separated part of a parenthesis is pronunciation debris."""
    p = part.strip().strip(",;:").strip()
    if not p:
        return True
    if p.lower() in ("or", "and", "also", "or also", "also called", "lit.", "lit", "listen"):
        return True
    if part.strip().endswith(":"):  # a label whose value was removed, e.g. "Irish:"
        return True
    if _IPA_RX.search(p) or _PRON_RX.search(p) or _SLASH_IPA_RX.search(" " + p + " "):
        return True
    return False


def _clean_paren_content(content: str) -> str:
    """Return the cleaned content of one parenthesis (may be empty -> remove it)."""
    # split on '; ' and ', ' (with optional surrounding spaces) but not inside numbers
    pieces = re.split(r"(\s*;\s*|\s*,\s+|^\s*,\s*|\s*,\s*$)", content)
    parts = pieces[0::2]
    seps = pieces[1::2]
    kept: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        if _drop_part(part):
            continue
        sep = seps[i] if i < len(seps) else ""
        kept.append((part.strip(), sep))
    if not kept:
        return ""
    out = []
    for i, (part, sep) in enumerate(kept):
        out.append(part)
        if i < len(kept) - 1:
            s = sep.strip() if sep.strip() else ","
            out.append(s + " ")
    return "".join(out).strip()


_PAREN_RX = re.compile(r"(\s?)\(([^()]*)\)")


def clean_parentheses(text: str) -> str:
    """Remove pronunciation/IPA parts and empty parentheses (innermost first)."""
    if "(" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        lead, content = m.group(1), m.group(2)
        c = _clean_paren_content(content)
        if not c:
            return ""
        return f"{lead}\x00{c}\x01"  # protect from re-matching; restored below

    prev = None
    while prev != text:
        prev = text
        text = _PAREN_RX.sub(repl, text)
    return text.replace("\x00", "(").replace("\x01", ")")


def clean_inline(text: str) -> str:
    """Inline cleaning of one line of prose or one list item."""
    text = clean_parentheses(text)
    if "[" in text:
        text = _REFMARK_RX.sub("", text)
    text = _MULTISPACE_RX.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RX.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RX.sub(r"\1", text)
    text = _DUP_PUNCT_RX.sub(r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"^[\s,;:.)]+", "", text)  # debris at the start of a line
    return text.strip()


# -------------------------------------------------------------------- sentence splitter

_ABBREV = frozenset(
    """mr mrs ms dr prof sr jr st mt ft no nos vol vols vs etc e.g i.e eg ie inc ltd co corp
    jan feb mar apr jun jul aug sep sept oct nov dec u.s u.k u.n a.m p.m approx est ca c
    gen col lt sgt capt cmdr adm gov sen rep rev hon pres ave blvd rd fig figs al op ed eds
    pp p b d fl r c.f cf dept univ assn bros mme mlle messrs esq lb lbs oz in.no km cm mm kg
    no.1 sq ch pt pts ps viz approx. mgr msgr fr br sec""".split()
)
_SENT_RX = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences at ``[.!?]`` + space + capital/digit.

    Abbreviations (``Dr.``, ``U.S.``, ``Jan.``) and single-letter initials (``A. S. Neill``)
    do not end a sentence. The concatenation ``" ".join(result)`` equals the paragraph with
    its inter-sentence whitespace collapsed to single spaces.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    out: list[str] = []
    start = 0
    for m in _SENT_RX.finditer(paragraph):
        end = m.end(1)
        # the word that carries the terminator
        k = m.start(1)
        w0 = k
        while w0 > start and not paragraph[w0 - 1].isspace():
            w0 -= 1
        word = paragraph[w0:k].strip("\"'“‘([").lower()
        if m.group(1)[0] == "." and len(m.group(1).rstrip("\"'”’)]")) == 1:
            if word in _ABBREV or (len(word) == 1 and word.isalpha()):
                continue
            if re.fullmatch(r"(?:[a-z]\.)+[a-z]", word):  # u.s, e.g, a.s
                continue
        if end - start <= 1:
            continue
        out.append(paragraph[start:end].strip())
        start = m.end()
    tail = paragraph[start:].strip()
    if tail:
        out.append(tail)
    return out


# ---------------------------------------------------------------------- article layer


@dataclass
class Paragraph:
    """One prose paragraph (``items`` empty) or one list (``sentences`` empty)."""

    sentences: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)

    @property
    def is_list(self) -> bool:
        return bool(self.items)

    def words(self) -> int:
        return sum(n_words(s) for s in self.sentences) + sum(n_words(s) for s in self.items)


@dataclass
class Section:
    heading: str  # "" for the lead section
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class CleanArticle:
    """Result of ``clean_article``. ``drop_reason`` is None when the article is kept."""

    title: str
    sections: list[Section]
    categories: list[str]
    drop_reason: str | None
    stats: dict[str, int]

    def prose_words(self) -> int:
        return sum(p.words() for s in self.sections for p in s.paragraphs if not p.is_list)

    def list_words(self) -> int:
        return sum(p.words() for s in self.sections for p in s.paragraphs if p.is_list)

    def text(self) -> str:
        """Plain rendering (headings and paragraphs on their own lines, lists as ``- item``)."""
        lines: list[str] = []
        for s in self.sections:
            if s.heading:
                lines.append(s.heading)
            for p in s.paragraphs:
                if p.is_list:
                    lines.extend("- " + it for it in p.items)
                else:
                    lines.append(" ".join(p.sentences))
        return "\n".join(lines)


@dataclass(frozen=True)
class CleanCfg:
    min_item_words: int = 3        # shorter list items are navigation lines
    list_heavy_ratio: float = 2.0  # list words > ratio * prose words -> drop the lists
    min_article_words: int = 5     # prose words needed to keep an article
    max_heading_words: int = 10
    max_category_words: int = 12
    repair_holes: bool = True      # drop/repair sentences with removed-unit holes (hole_repair)


def _norm_heading(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(":").strip()).lower()


def _is_category_line(s: str, max_words: int) -> bool:
    s = s.strip()
    if not s:
        return True
    if _is_table_or_image_line(s) or s[0] in "{=*#:" or "=" in s:
        return False
    if len(s) > 100 or n_words(s) > max_words:
        return False
    if s[-1] in ".!?;:," or ". " in s:
        return False
    return True


def _heading_like(raw: str, max_words: int) -> bool:
    """A short line without terminal punctuation (the dump's rendering of a heading)."""
    s = raw.strip()
    if not s or raw.startswith((" ", "\t")) or _LIST_MARK_RX.match(raw) and not raw[0].isalnum():
        return False
    if _EQ_HEADING_RX.match(s):
        return True
    w = n_words(s)
    if w == 0 or w > max_words or len(s) > 80:
        return False
    if s[-1] in ".,;:!":
        return False
    if s[-1] == "?" and not (raw.endswith(" ") and w <= 6):
        return False
    if s[-1] == ")" and w > 6:
        return False
    return True


_DISAMBIG_RX = re.compile(
    r"\b(may|can|could|might|will|also|sometimes)\s+(also\s+)?(refer to|mean|be|stand for|"
    r"be one of|be a name|have several meanings)\b[^.\n]*:\s*$",
    re.I,
)


def clean_article(title: str, text: str, cfg: CleanCfg = CleanCfg()) -> CleanArticle:
    """Clean one article. Never raises on odd input; see the module docstring for rules."""
    stats: dict[str, int] = {"raw_chars": len(text)}
    title = re.sub(r"\s+", " ", title).strip()
    if "(disambiguation)" in title.lower():
        return CleanArticle(title, [], [], "disambiguation", stats)
    text = strip_markup(text)
    lines = text.split("\n")
    # -- categories: the last line group when every line looks like a category name
    while lines and not lines[-1].strip():
        lines.pop()
    categories: list[str] = []
    j = len(lines)
    while j > 0 and lines[j - 1].strip():
        j -= 1
    last_group = lines[j:]
    if last_group and j > 0 and all(
        _is_category_line(x, cfg.max_category_words) and not x.startswith(" ") for x in last_group
    ):
        categories = [x.strip() for x in last_group if x.strip()]
        lines = lines[:j]
    # -- tail cut at the first tail heading
    for i, raw in enumerate(lines):
        s = raw.strip()
        m = _EQ_HEADING_RX.match(s)
        h = m.group(2) if m else s
        if h and _norm_heading(h) in TAIL_HEADINGS and (m or _heading_like(raw, cfg.max_heading_words)):
            stats["tail_cut_lines"] = len(lines) - i
            lines = lines[:i]
            break
    # -- line classification into sections / paragraphs
    sections: list[Section] = [Section("")]
    cur_list: list[str] | None = None
    n_dropped_items = 0
    n_dropped_lines = 0
    n_broken = 0
    n_holes = 0
    n_hole_repairs = 0

    def close_list() -> None:
        nonlocal cur_list
        if cur_list:
            sections[-1].paragraphs.append(Paragraph(items=cur_list))
        cur_list = None

    def repair(sent: str) -> str | None:
        nonlocal n_holes, n_hole_repairs
        if not cfg.repair_holes:
            return sent
        fixed, nrep = hole_repair(sent)
        if fixed is None:
            n_holes += 1
        n_hole_repairs += nrep
        return fixed

    def add_item(text: str) -> None:
        nonlocal cur_list, n_dropped_items
        item = repair(clean_inline(text)) or ""
        if n_words(item) < cfg.min_item_words:
            n_dropped_items += 1
            if cur_list is None:
                cur_list = []
            return
        if cur_list is None:
            cur_list = []
        cur_list.append(item)

    def indented(k: int) -> bool:
        return 0 <= k < nlines and lines[k][:1] in (" ", "\t") and bool(lines[k].strip())

    prev_blank = True
    nlines = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            close_list()
            prev_blank = True
            continue
        if _is_table_or_image_line(s):
            n_dropped_lines += 1
            continue
        m = _EQ_HEADING_RX.match(s)
        nxt = lines[i + 1].strip() if i + 1 < nlines else ""
        nxt2 = lines[i + 2].strip() if i + 2 < nlines else ""
        if m or (prev_blank and _heading_like(raw, cfg.max_heading_words) and (nxt or nxt2)):
            close_list()
            sections.append(Section(clean_inline(m.group(2) if m else s)))
            prev_blank = False
            continue
        marked = bool(_LIST_MARK_RX.match(raw)) and not s[0].isalnum()
        if indented(i) or marked:
            long_prose = (n_words(s) >= 12 and bool(_TERMINAL_RX.search(s))
                          and not indented(i - 1) and not indented(i + 1))
            is_item = not long_prose
        elif cur_list is not None and not _ENDS_RX.search(s):
            is_item = True  # continuation of a list run
        elif (not prev_blank and not _ENDS_RX.search(s) and nxt and not _ENDS_RX.search(nxt)
              and n_words(s) <= 15):
            is_item = True  # first line of an unindented run of short lines
        else:
            is_item = False
        if is_item:
            add_item(_LIST_MARK_RX.sub("", s))
        else:
            close_list()
            sents = []
            for sent in split_sentences(clean_inline(s)):
                if _BROKEN_END_RX.search(sent):
                    n_broken += 1
                    continue
                fixed = repair(sent)
                if fixed is not None:
                    sents.append(fixed)
            if sents:  # every source line becomes its own paragraph
                sections[-1].paragraphs.append(Paragraph(sentences=sents))
        prev_blank = False
    close_list()
    # -- drop empty sections (headings without content)
    sections = [s for k, s in enumerate(sections) if s.paragraphs or k == 0]
    art = CleanArticle(title, sections, categories, None, stats)
    prose, lst = art.prose_words(), art.list_words()
    stats.update(
        prose_words=prose, list_words=lst, dropped_items=n_dropped_items,
        dropped_lines=n_dropped_lines, broken_sentences=n_broken, hole_sentences=n_holes,
        hole_repairs=n_hole_repairs,
    )
    # -- disambiguation pages
    firsts = [p for s in sections for p in s.paragraphs][:2]
    for p in firsts:
        if not p.is_list and p.sentences and _DISAMBIG_RX.search(p.sentences[-1]) \
                and lst >= prose * 0.5:
            art.drop_reason = "disambiguation"
            return art
    if title.lower().startswith(("list of ", "lists of ")) and lst > prose:
        art.drop_reason = "list_page"
        return art
    # -- list-heavy pages keep their prose only
    if lst > cfg.list_heavy_ratio * prose:
        for s in sections:
            s.paragraphs = [p for p in s.paragraphs if not p.is_list]
        art.sections = [s for k, s in enumerate(sections) if s.paragraphs or k == 0]
        stats["lists_dropped"] = lst
        if prose < cfg.min_article_words:
            art.drop_reason = "list_only"
            return art
    has_sentence = any(_TERMINAL_RX.search(x) for s in art.sections for p in s.paragraphs
                       for x in p.sentences)
    if art.prose_words() + art.list_words() < cfg.min_article_words or not has_sentence:
        art.drop_reason = "empty"
        return art
    return art


# ---------------------------------------------------------------------- passage layer


@dataclass(frozen=True)
class RawPassage:
    """A passage before packing. ``body`` excludes the title; ``render`` adds it."""

    title: str
    body: str
    pos: int          # index of the passage inside its article (0 = lead)
    section: str      # heading of the section the passage starts in ("" = lead)
    n_words: int

    def render(self) -> str:
        return render_passage(self.title, self.body)


def render_passage(title: str, body: str) -> str:
    """Text shown to the model: the title line, then the body."""
    return f"{title}\n{body}"


def _split_long(unit: str, fits: Callable[[str], bool]) -> list[str]:
    """Split one over-long sentence/item into pieces that fit (word boundaries first,
    characters for a single word that is longer than a passage)."""
    pieces: list[str] = []
    cur = ""
    for w in unit.split():
        trial = w if not cur else cur + " " + w
        if fits(trial):
            cur = trial
            continue
        if cur:
            pieces.append(cur)
        cur = ""
        if fits(w):
            cur = w
            continue
        while w:  # a single word that does not fit: cut by characters
            k = len(w)
            while k > 1 and not fits(w[:k]):
                k -= 1
            pieces.append(w[:k])
            w = w[k:]
    if cur:
        pieces.append(cur)
    return pieces


class _NotLineAdditive(Exception):
    """A passage counted line by line exceeded the cap when counted as a whole."""


def make_passages(
    art: CleanArticle,
    max_words: int = _L.max_passage_words,
    max_tokens: float = _L.max_passage_tokens,
    count_tokens: CountTokens = default_count_tokens,
    flush_frac: float = 0.6,
    line_wise: bool | None = None,
) -> list[RawPassage]:
    """Split a cleaned article into passages.

    Each passage satisfies ``n_words(body) <= max_words`` and
    ``count_tokens(title + "\\n" + body) <= max_tokens``. Passages start at a paragraph or
    sentence boundary (only a single sentence longer than a passage is cut inside). A
    passage that starts inside a non-lead section starts with that section's heading line;
    a heading line is only ever emitted together with content of its section. At a section
    boundary the current passage is closed when it is at least ``flush_frac`` full (in words
    or tokens); otherwise the next section continues in the same passage. A heading that
    takes more than half of ``max_tokens`` on its own is left out.

    ``line_wise`` (default: on for ``bpe_count_tokens``) counts a candidate passage as
    ``count(title + "\\n") + sum(count(line + "\\n")) + count(last line)`` with every line
    counted once: in the BPE split pattern a newline followed by a non-space character always
    ends a pre-token, so the sum equals the count of the whole text (verified on 20,000 pool
    passages). Every finished passage is still counted as a whole; if that ever exceeds the
    cap, the article is split again with whole-text counts, so the cap holds in any case.
    """
    if line_wise is None:
        line_wise = count_tokens is bpe_count_tokens
    if line_wise:
        try:
            return _make_passages(art, max_words, max_tokens, count_tokens, flush_frac, True)
        except _NotLineAdditive:
            pass
    return _make_passages(art, max_words, max_tokens, count_tokens, flush_frac, False)


def _make_passages(art: CleanArticle, max_words: int, max_tokens: float, count_tokens: CountTokens,
                   flush_frac: float, line_wise: bool) -> list[RawPassage]:
    title = art.title
    out: list[RawPassage] = []
    if count_tokens(render_passage(title, "x")) > max_tokens:
        return out  # the title alone does not fit a passage
    lines: list[str] = []
    words = 0
    section0 = ""
    line_open = False  # whether the last line is a prose line that accepts more sentences
    memo_nl: dict[str, float] = {}
    memo_last: dict[str, float] = {}
    head_n = count_tokens(title + "\n") if line_wise else 0.0

    def ntok(ls: list[str]) -> float:
        if not line_wise:
            return count_tokens(render_passage(title, "\n".join(ls)))
        n = head_n
        for ln in ls[:-1]:
            v = memo_nl.get(ln)
            if v is None:
                v = memo_nl[ln] = count_tokens(ln + "\n")
            n += v
        v = memo_last.get(ls[-1])
        if v is None:
            v = memo_last[ls[-1]] = count_tokens(ls[-1])
        return n + v

    def fits(ls: list[str], w: int) -> bool:
        return w <= max_words and ntok(ls) <= max_tokens

    def candidate(pending: str, text: str, uw: int, join: bool) -> tuple[list[str], int]:
        new = list(lines)
        w = words
        if pending:
            new.append(pending)
            w += n_words(pending)
            join = False
        if join and new:
            new[-1] = new[-1] + " " + text
        else:
            new.append(text)
        return new, w + uw

    def flush() -> None:
        nonlocal lines, words, line_open
        if lines and words > 0:
            body = "\n".join(lines)
            if line_wise and count_tokens(render_passage(title, body)) > max_tokens:
                raise _NotLineAdditive(title)
            out.append(RawPassage(title, body, len(out), section0, words))
        lines, words, line_open = [], 0, False

    def fill() -> float:
        if not lines:
            return 0.0
        return max(words / max_words, ntok(lines) / max_tokens)

    for sec in art.sections:
        if not sec.paragraphs:
            continue
        head = sec.heading
        if head and count_tokens(render_passage(title, head)) > max_tokens / 2:
            head = ""   # a heading that fills half a passage (e.g. a long non-Latin heading) is dropped
        if lines and fill() >= flush_frac:
            flush()
        pending = head  # emitted right before the first unit of the section
        for para in sec.paragraphs:
            line_open = False
            for unit in (para.items if para.is_list else para.sentences):
                text = ("- " + unit) if para.is_list else unit
                uw = n_words(text)  # words as rendered (the list marker counts)
                join = line_open and not para.is_list
                cand, cw = candidate(pending, text, uw, join)
                if not fits(cand, cw) and lines:
                    flush()
                    pending = head  # a continuation passage repeats the heading
                    cand, cw = candidate(pending, text, uw, False)
                if fits(cand, cw):
                    if not lines:
                        section0 = head
                    lines, words = cand, cw
                    pending = ""
                    line_open = not para.is_list
                    continue
                # a single unit longer than a passage (lines is empty here)
                prefix = "- " if para.is_list else ""
                head_lines = [pending] if pending else []
                hw = n_words(pending)
                if head_lines and not fits(head_lines + [prefix + unit[:1]], hw + 1):
                    head_lines, hw = [], 0      # not even one character fits after the heading

                def piece_fits(piece: str) -> bool:
                    return fits(head_lines + [prefix + piece], hw + n_words(prefix + piece))

                pieces = _split_long(unit, piece_fits)
                for k, piece in enumerate(pieces):
                    section0 = head
                    lines = head_lines + [prefix + piece]
                    words = hw + n_words(prefix + piece)
                    if k < len(pieces) - 1:
                        flush()
                pending = ""
                line_open = not para.is_list
    flush()
    return out


def clean_and_split(title: str, text: str, cfg: CleanCfg = CleanCfg(),
                    max_words: int = _L.max_passage_words,
                    max_tokens: float = _L.max_passage_tokens,
                    count_tokens: CountTokens = default_count_tokens) -> tuple[CleanArticle, list[RawPassage]]:
    """``clean_article`` followed by ``make_passages`` (no passages for dropped articles)."""
    art = clean_article(title, text, cfg)
    if art.drop_reason is not None:
        return art, []
    return art, make_passages(art, max_words, max_tokens, count_tokens)


# -------------------------------------------------------------------------- streaming


def iter_articles(parquet_path: str, columns: Sequence[str] = ("id", "title", "text"),
                  row_groups: Iterable[int] | None = None) -> Iterator[dict]:
    """Stream rows of the parquet file one row group at a time (bounded memory)."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(parquet_path)
    groups = range(f.metadata.num_row_groups) if row_groups is None else row_groups
    for g in groups:
        t = f.read_row_group(g, columns=list(columns))
        cols = {c: t.column(c).to_pylist() for c in columns}
        for i in range(t.num_rows):
            yield {c: cols[c][i] for c in columns}
        del t, cols
