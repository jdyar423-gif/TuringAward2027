"""Gated retrieval of at most one library entry per user message.

``Library`` ties together the packed wiki passages (``bes.library.pack.Packed``), the stored
truncated BM25 index (``bes.library.index.TruncatedIndex``), the title lookup table
(``TitleTable``), the optional question-answer layer (``bes.library.qa.QAStore``) and the
calibrated gate (``bes.library.gate.GateModel``). Every number a query reads is logged in a
``ReadLog`` as distinct stored numbers.

Entries have contiguous passage ids per kind: wiki passages ``0 .. n_wiki - 1``, then qa
entries (``kind_range``). A qa entry is a ``Passage`` with ``kind="qa"``, ``title`` = the
stored question, ``body`` = the stored answer, ``text = "Q: <question>\\nA: <answer>"`` and
``title_id = -1``.

Candidates of one query:

* wiki: for an identity request (``identity_frame``: "Who was X?", "What is X?", "Tell me
  about X.", a bare name) the lead passage of the primary topic of X's key
  (``bes.library.titles.TitleTable``; "Who was Napoleon?" -> "Napoleon", "Who was Galileo?"
  -> "Galileo Galilei"), else the best BM25 passage;
* qa: the best stored question by symmetric overlap (``bes.library.qa``) among the first
  ``qa_read_entries`` that pass the contrast check, when its ``sym >= qa_min_sym``.

Gate: the turn type must be allowed (``turn_type``: questions and short look-ups; no
writing, coding, translation or arithmetic tasks, no chat; with ``context`` (the earlier
user turns) no later turn that leans on them). With a fitted ``GateModel`` the candidate
with the larger calibrated probability is returned when that probability reaches the
model's threshold. Without one (tests, old libraries) the hand-set ``GateCfg`` rule applies
to the BM25 candidate: top score ``s1 >= min_score``, coverage ``s1 / s_max >=
min_coverage``, and margin ``(s1 - s2) / s1 >= min_margin`` against the best passage of a
different article, or ``>= min_margin_title`` when ``>= min_title_match`` of the title terms
occur in the query.

A wiki passage is rendered as ``title + "\\n" + body`` capped at ``max_tokens`` tokens
(``bes.library.clean.default_count_tokens``, the BPE tokenizer, loaded lazily).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from bes.config import LibraryCfg
from bes.library.clean import CountTokens, default_count_tokens, render_passage, split_sentences
from bes.library.index import IndexCfg, IndexRead, TruncatedIndex
from bes.library.pack import Packed, PackCfg, _atomic_save_npy, _atomic_write, numbers_of, pack
from bes.library.terms import STOPWORDS, strip_disambiguator, terms, tokens
from bes.library.titles import (DEF_WORDS, TitleTable, case_tokens, key_hash, object_keys,  # noqa: F401
                                surface, title_key, verify)

# ------------------------------------------------------------------------- query type

_LEADS = ("please", "can you", "could you", "would you", "will you", "i want you to",
          "i need you to", "i'd like you to", "i would like you to", "kindly", "pls", "hey",
          "hi", "hello", "ok", "okay", "so", "now", "then", "and", "but", "also", "well")
_TASK = frozenset(
    """write compose draft create generate rewrite rephrase paraphrase summarize summarise
    translate edit proofread correct fix make design brainstorm suggest imagine pretend act
    roleplay continue complete convert format classify categorize categorise sort extract
    calculate compute solve implement code debug refactor optimize optimise plan outline list
    give provide help come draw sing play respond reply recommend rank rate review analyze
    analyse evaluate improve shorten expand simplify turn change develop build prepare
    organize organise schedule find count combine split rewrite""".split()
)
_ASK = frozenset(
    """who what when where which why how whom whose is are was were do does did can could has
    have had name in on at tell explain describe define what's who's where's when's how's
    which's whats whos""".split()
)
_SMALL_TALK = frozenset(
    """hi hello hey thanks thank you thx ok okay yes no bye goodbye good morning evening night
    cool great nice sure please lol""".split()
)
_MATH_RX = re.compile(r"\d\s*[-+*/^×÷=%]\s*\(?\d|\b(solve|equation|integral|derivative)\b|"
                      r"\d\s*(?:squared|cubed|factorial|!)|\b(?:minus|plus|multiplied by|divided by|times|"
                      r"to the power of)\s+\d|\b(?:square|cube) root of\s+\d|\bhow many \w+ (?:are )?(?:there )?"
                      r"in \d|\d\s*(?:degrees? )?(?:celsius|fahrenheit|kelvin)\b|\b(?:perimeter|circumference)\b|"
                      r"\bwhat fraction\b|"
                      r"\d+(?:\.\d+)?\s*%\s*of\b|\bpercent(?:age)? of\b|\bconvert \d|"
                      r"\b(?:sum|product|average|mean|median|square root|factorial) of \d", re.I)
# Word problems: at least two numbers and an arithmetic cue ("43 of 50 ... what percentage").
_NUM_RX = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_WORD_PROBLEM_RX = re.compile(r"\b(?:percentage|percent|average|total|altogether|in all|left|remain|"
                              r"remaining|each|per|cost|costs|price|speed|rate|ratio|times|sum|difference|"
                              r"product|twice|half|double|split|share|change|profit|discount|interest|"
                              r"km/h|mph|how much|how many|how long|how far)\b")
# Code: an unambiguous programming word, or characters and constructs that only code uses.
_CODE_RX = re.compile(r"\b(?:javascript|typescript|c\+\+|c#|golang|sql|html|css|regex|regexp|json|"
                      r"yaml|xml|api|apis|compiler|recursion|recursive|boolean|programming|coding|"
                      r"source code|big o|time complexity|linked list|binary search|hash ?map|"
                      r"data structure|unit test|git|github|docker|kubernetes|numpy|pandas|"
                      r"dataframe|python (?:function|code|script|program|list|dict|class|module|"
                      r"library|package|loop)|in python|python's|bash script|shell script|"
                      r"stack overflow|null pointer|segfault|runtime error|syntax error)\b"
                      r"|[{}]|;\s*$|; |==|!=|=>|->|`|\b[a-z_]\w*\([^()]*\)|\bdef |\breturn\b", re.I)
# First- or second-person messages (the asker or the assistant as subject) are advice,
# opinion or chat, not factual look-ups. Quoted text and everything after a word that
# introduces a title ("who sang I will always love you", "what genre is My Way") is ignored,
# and generic "you" ("where would you find ...") does not count.
_TITLE_CTX_RX = re.compile(r"\b(?:sings?|sang|sung|singer|wrote|written|song|songs|album|movie|film|book|"
                           r"novel|play|show|poem|lyrics|title|titled|called|named|genre|single|track|"
                           r"episode|series|musical|opera|catchphrase|motto|slogan|saying|quote|phrase|"
                           r"line|words)\b")
_QUOTED_RX = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<![a-z])'[^']*'(?![a-z])")
_PERSONAL_RX = re.compile(
    r"^(?:i|i'm|im|i've|i'd|i'll|my|we|we're|we've|our|let's)\b"
    r"|\b(?:should|can|could|would|do|did|will|shall|must|may|might|am|have)\s+(?:i|we)\b"
    r"|\b(?:i|we)\s+(?:am|was|were|have|had|want|need|feel|think|like|love|hate|got|get|go|went|drove|"
    r"bought|made|did|do|will|would|can|could|should|must|just|really|also|always|never|tried|keep|live|"
    r"work)\b"
    r"|\b(?:my|mine|myself|our|ours|ourselves)\b"
    r"|\b(?:do|did|have|are|were|would|could|can|will)\s+you\s+(?:think|like|prefer|recommend|believe|"
    r"feel|want|enjoy|love|hate|suggest|advise|have|remember|consider|personally)\b"
    r"|\byour\s+(?:opinion|view|views|favou?rite|name|thoughts|ideal|advice|recommendation|experience|"
    r"feelings|best|own|take)\b"
    r"|\byourself\b|^(?:you|you're|are you|who are you)\b")
_ABOUT_ME_OK = re.compile(r"^(?:(?:can|could|would|will) you |please )?tell me (?:about|who|what|when|"
                          r"where|why|how|which)\b|^(?:what|who|where|when) (?:do|did|can) you know about\b|"
                          r"^do you know (?:about|who|what|when|where|why|how|which)\b|"
                          r"^(?:can|could) you (?:tell|explain|describe)\b", re.I)
_WRITING_RX = re.compile(r"\b(?:story|poem|haiku|limerick|essay|letter|email|e-mail|speech|toast|"
                         r"song|lyrics|slogan|tagline|tweet|caption|joke|riddle|dialogue|script|"
                         r"paragraph|sentence|sentences|words|bullet|bullets|list)\b", re.I)
_WH = frozenset("who what when where which why how whom whose".split())


def personal(low: str) -> bool:
    """Whether a (lower-case, lead-stripped) message is about the asker or the assistant."""
    t = _QUOTED_RX.sub(" ", low)
    m = _TITLE_CTX_RX.search(t)
    if m is not None:
        t = t[:m.start()]
    return bool(_PERSONAL_RX.search(t.strip()))


# Greetings, wishes, thanks, acknowledgements and requests to go on: chat, not a look-up.
_ACK = (r"thanks|thank you|thank u|thx|ty|many thanks|cheers|ok|okay|k|kk|cool|great|good|nice|awesome|amazing|"
        r"excellent|perfect|sure|of course|yes|yeah|yep|yup|yes please|no|nope|nah|no thanks|no thank you|"
        r"interesting|wow|oh|ah|hmm|huh|really|indeed|right|correct|exactly|true|wrong|lol|haha|got it|"
        r"i see|i understand|understood|makes sense|that makes sense|sounds good|sounds great|never ?mind|"
        r"no problem|no worries|my pleasure|you'?re welcome|well done|good job|great job|nice job|good work|"
        r"great work|nice work|great answer|good answer|nice one|that'?s (?:it|all|fine|ok|okay|great|good|"
        r"cool|nice|right|correct|wrong|true|not true|incorrect|interesting|funny|helpful|awesome|perfect|"
        r"amazing)|that (?:is|was) (?:it|all|great|good|nice|right|correct|wrong|true|not true|incorrect|"
        r"interesting|funny|helpful|awesome|perfect|amazing|very helpful)|you(?:'re| are) (?:wrong|right|"
        r"great|awesome|funny|amazing|the best|welcome)|agreed|fair enough|fine|alright|all right|bye|"
        r"goodbye|good ?bye|see you|see you later|see ya|later|take care|good ?night|good morning|"
        r"good afternoon|good evening|good day|hello|hi|hey|hiya|yo|greetings|hello there|hi there|hey there|"
        r"what'?s up|sup|how are you|how are you doing|how are you today|how are you doing today|"
        r"how'?s it going|how do you do|merry christmas|happy (?:new year|birthday|holidays|easter|"
        r"thanksgiving|halloween|diwali|eid|hanukkah|valentine'?s day)|congrats|congratulations|good luck|"
        r"best wishes|all the best|try again|again|go on|go ahead|continue|keep going|carry on|more|"
        r"tell me more|say more|one more|another one|next|next one|please continue|please go on|"
        r"and|so|why|what|really|ok so|i love you|love you|sorry|my bad|oops|wait|stop|enough|so much|"
        r"very much|a lot|lots|too")
_CHAT_RX = re.compile(r"^(?:(?:" + _ACK + r")\b[\s,.!?:;)(-]*)+$")


def chit_chat(text: str) -> bool:
    """Greetings, wishes, thanks, acknowledgements and requests to retry or go on: one or
    more such phrases and nothing else ("ok great", "cool, thanks!"), or one followed by a
    short remark about the answer ("thanks, that was helpful")."""
    low = re.sub(r"[\"“”`*_~]", " ", text.strip().lower().replace("’", "'"))
    low = re.sub(r"\s+", " ", low).strip()
    if not low or len(low) > 80:
        return False
    if _CHAT_RX.match(low):
        return True
    # "thanks, that was helpful" / "cool, thanks!": an acknowledgement opening a short remark
    m = re.match(r"^(?:" + _ACK + r")\b[\s,.!?:;-]+(.*)$", low)
    if m and m.group(1):
        rest = m.group(1)
        return bool(re.fullmatch(r"(?:that|this|it|you)(?:'s| is| was| were| are)? ?(?:very |so |really |"
                                 r"super |quite )?(?:helpful|useful|great|good|nice|clear|interesting|cool|"
                                 r"perfect|amazing|awesome|right|correct|fine|enough)[\s.!?]*|"
                                 r"(?:" + _ACK + r")\b[\s,.!?]*", rest))
    return False


# Back-references a later turn may lean on: pronouns, demonstratives, "the second one".
_BACKREF_RX = re.compile(
    r"\b(?:it|its|itself|they|them|their|theirs|themselves|he|him|his|himself|she|her|hers|herself|"
    r"this|that|these|those)\b|"
    r"\bthe (?:first|second|third|fourth|fifth|last|next|previous|other|same|former|latter|above|"
    r"following|latest|earlier|later|original|new|old)(?: \w+)?\b|"
    r"\b(?:the )?(?:other|same|former|latter) ones?\b|\bwhich ones?\b|\b(?:option|choice|number|item|step|"
    r"point) (?:[a-d]|\d+|one|two|three)\b|\b(?:both|either|neither)\b")
_OPENER_RX = re.compile(r"^(?:(?:and|but|or|so|also|then|now|ok|okay|well|oh|hmm|alright|right)\b[\s,]*)+")
_ELLIPSIS_RX = re.compile(r"^(?:what|how) about\b|^(?:what|how) (?:else|so)\b|^why (?:not|so)\b|"
                          r"^same (?:for|with|about)\b|^(?:in|for|with|without|on|at|by|to|from|of) [\w' -]+\W*$|"
                          r"^\W*$")


def capitalised_names(text: str) -> list[str]:
    """Words written with a capital letter that do not start the message or a sentence
    (lower case; an acronym of two or more capitals counts anywhere)."""
    out = []
    initial = True
    for w in re.findall(r"[\w'’-]+|[.!?;:]", text):
        if w in ".!?;:":
            initial = True
            continue
        w = w.strip("'’-")
        if w and ((len(w) > 1 and w.isupper()) or (w[:1].isupper() and not initial)):
            out.append(w.lower())
        initial = False
    return out


def self_contained(text: str) -> bool:
    """Whether a later user turn can be looked up on its own: no elliptical opening ("what
    about X?", "and in France?"), no back-reference (it / its / he / she / they / that /
    "the second one") unless the message names something itself (a capitalised name or at
    least three content terms), and at least two content terms or a capitalised name."""
    low = _OPENER_RX.sub("", re.sub(r"\s+", " ", text.strip().lower()))
    names = [w for w in capitalised_names(text) if w not in STOPWORDS]
    content = [t for t in terms(text) if t not in DEF_WORDS]
    if _ELLIPSIS_RX.search(low):
        return False
    t = _QUOTED_RX.sub(" ", low)
    m = _TITLE_CTX_RX.search(t)              # "who sang it's a long way to the top": a title
    head = t[:m.start()] if m is not None else t
    if _BACKREF_RX.search(head) and not names and len(content) < 3:
        return False
    return len(content) >= 2 or bool(names)


def turn_type(text: str, context: Sequence[str] | None = None) -> str:
    """``query_type`` of a user message, or ``"followup"`` for a later turn (``context``: the
    earlier user turns) that is not ``self_contained``."""
    if context and not self_contained(text):
        return "followup"
    return query_type(text)


# Identity requests: the frames under which an exact title match counts.
_FRAME_RX = re.compile(
    r"^(?:(?:and|so|ok|okay|well|hey|hi|hello|please|now)[ ,]+)*(?:(?:can|could|would|will) you (?:please )?)?"
    r"(?P<f>what(?:'s| is) (?:the )?definition of|(?:the )?definition of|"
    r"(?:what|who)(?:'s| is| are| was| were)|whats|whos|"
    r"tell me (?:something |more |all |a (?:little|bit) )?about|(?:what|how much) do you know about|"
    r"do you know (?:about|anything about)|(?:give|show) me (?:some )?(?:information|info|facts|an overview|"
    r"a summary) (?:about|on|of)|define|explain|describe|"
    r"i want to (?:know|learn) (?:more )?about|information (?:about|on)|"
    r"facts (?:about|on)|an? overview of)\s+(?P<x>.+?)[\s?.!]*$", re.I)
_MEAN_RX = re.compile(r"^(?:what|whats|what's) (?:does|do) (?P<x>.+?) mean[\s?.!]*$", re.I)
# An object that belongs to something else ("its capital", "her husband") names no article.
_RELATIVE_OBJECT_RX = re.compile(r"^(?:its|his|her|their|my|your|our|this|that|these|those|it|he|she|they|"
                                 r"him|them)\b", re.I)


def identity_frame(query: str) -> tuple[str, str] | None:
    """``(kind, object)`` of an identity request: ``who`` ("Who was X?"), ``what`` ("What is
    X?"), ``neutral`` ("Tell me about X.", "Define X.", "What does X mean?") or ``bare`` (the
    message is the name itself, at most six words); None for every other message ("Why is the
    sky blue?", "What does len do?", "try again")."""
    t = re.sub(r"\s+", " ", query.strip())
    m = _MEAN_RX.match(t)
    if m:
        out = ("neutral", m.group("x"))
    elif (m := _FRAME_RX.match(t)) is not None:
        f = m.group("f").lower()
        kind = "who" if f.startswith("who") else "what" if f.startswith("what") and "definition" not in f \
            else "neutral"
        out = (kind, m.group("x"))
    elif not re.search(r"[?!;:]|\.\s", t) and len(t.split()) <= 6 and re.search(r"\w", t):
        out = ("bare", t.rstrip("."))
    else:
        return None
    if _RELATIVE_OBJECT_RX.match(out[1]):
        return None                          # "What is its capital?", "Who is her husband?"
    return out


def query_type(text: str) -> str:
    """Coarse type of a user message: ``question``, ``lookup``, ``task``, ``math`` or ``other``.

    ``question``: starts with a question/explanation word (after politeness leads), ends
    with ``?`` or has a wh-word among its first words; ``lookup``: a phrase of at most twelve
    words without sentence punctuation; ``task``: an instruction to produce or transform
    text/code, a message about code, a writing request phrased as a question ("Can you write
    a poem ...?"), or a long/pasted message; ``math``: arithmetic, percentages, unit
    conversions with numbers, equations; ``other``: small talk, messages about the asker or
    the assistant (``personal``), longer statements.
    """
    t = text.strip()
    if not t:
        return "other"
    if "```" in t or len(t) > 1000 or t.count("\n") > 6:
        return "task"
    low = re.sub(r"\s+", " ", t.lower())
    changed = True
    while changed:
        changed = False
        for lead in _LEADS:
            if low.startswith(lead + " ") or low.startswith(lead + ","):
                low = low[len(lead):].lstrip(" ,")
                changed = True
    words = re.findall(r"[a-z0-9']+", low)
    if not words or all(w in _SMALL_TALK for w in words) or chit_chat(t):
        return "other"
    if _MATH_RX.search(low) or (len(set(_NUM_RX.findall(low))) >= 2 and _WORD_PROBLEM_RX.search(low)):
        return "math"
    if _CODE_RX.search(t):
        return "task"
    if personal(low) and not _ABOUT_ME_OK.search(t.strip().lower()):
        return "other"
    first = words[0]
    if first in _TASK:
        return "task"
    if first in ("tell", "explain", "describe", "define") or low.startswith(("give me information",
                                                                               "give me facts")):
        return "task" if _WRITING_RX.search(low) and first != "define" and "about" not in words[:3] else "question"
    if first in _ASK or t.endswith("?"):
        if len(words) > 2 and words[1] in ("you", "u") and words[2] in _TASK:
            return "task"            # "can you write ...?"
        return "question"
    if any(w in _WH for w in words[1:12]):
        return "question"            # "india's first arctic research station ... what is the name"
    if len(words) <= 12 and not re.search(r"[.!;:]\s", t):
        return "lookup"              # a phrase: "nearest metro station to gandhi nagar in delhi"
    return "other"


# ------------------------------------------------------------------------- config


_L = LibraryCfg()


@dataclass(frozen=True)
class GateCfg:
    """Hand-set rule gate (used only when the library has no fitted ``GateModel``)."""

    min_score: float = _L.gate_min_score            # BM25 units (de-quantised impacts)
    min_margin: float = _L.gate_min_margin
    min_margin_title: float = _L.gate_min_margin_title
    min_title_match: float = _L.gate_min_title_match
    min_coverage: float = _L.gate_min_coverage
    allowed_types: tuple[str, ...] = ("question", "lookup")
    margin_candidates: int = 64     # candidates scanned for the best different-title score


@dataclass(frozen=True)
class RetrieveCfg:
    max_tokens: float = _L.max_passage_tokens  # cap on title line + body
    gate: GateCfg = field(default_factory=GateCfg)
    qa_min_sym: float = _L.qa_min_sym          # qa candidates below this overlap are ignored
    agree_min_sym: float = 0.5                 # an "other" qa entry counts for agree/disagree
    cache_blocks: int = 64                     # decompressed blocks kept per process (LRU)
    qa_read_entries: int = _L.qa_read_entries  # qa candidates read (contrast check) per query at most


# ------------------------------------------------------------------------- read log


@dataclass
class ReadLog:
    """Distinct stored numbers read while answering one query."""

    index_df: int = 0
    index_offsets: int = 0
    index_postings: int = 0
    directory: int = 0
    text_blocks: int = 0
    title_blocks: int = 0
    title_lookup: int = 0
    qa_index: int = 0
    qa_blocks: int = 0
    blocks: set = field(default_factory=set, repr=False)  # text block ids read
    title_block_ids: set = field(default_factory=set, repr=False)
    qa_block_ids: set = field(default_factory=set, repr=False)
    _dir_seen: set = field(default_factory=set, repr=False)

    @property
    def index(self) -> int:
        return self.index_df + self.index_offsets + self.index_postings

    @property
    def total(self) -> int:
        return (self.index + self.directory + self.text_blocks + self.title_blocks
                + self.title_lookup + self.qa_index + self.qa_blocks)

    def as_dict(self) -> dict[str, int]:
        return {"index_df": self.index_df, "index_offsets": self.index_offsets,
                "index_postings": self.index_postings,
                "directory": self.directory, "text_blocks": self.text_blocks,
                "title_blocks": self.title_blocks, "title_lookup": self.title_lookup,
                "qa_index": self.qa_index, "qa_blocks": self.qa_blocks,
                "n_text_blocks": len(self.blocks), "n_title_blocks": len(self.title_block_ids),
                "n_qa_blocks": len(self.qa_block_ids), "total": self.total}

    def touch_dir(self, array: str, idx: Iterable[int], numbers_per_entry: int) -> None:
        for i in idx:
            key = (array, int(i))
            if key not in self._dir_seen:
                self._dir_seen.add(key)
                self.directory += numbers_per_entry


# ------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Passage:
    pid: int
    title_id: int     # -1 for qa entries
    title: str        # article title; the stored question for qa entries
    body: str         # passage body; the stored answer for qa entries
    score: float      # BM25 score (wiki) or symmetric overlap (qa) from the stored index
    text: str         # wiki: title + "\n" + body capped at max_tokens; qa: "Q: ...\nA: ..."
    kind: str = "wiki"
    prob: float | None = None  # calibrated gate probability (None without a gate model)


def cap_tokens(title: str, body: str, max_tokens: float, count_tokens: CountTokens) -> str:
    """``render_passage(title, body)`` cut at the last sentence (else word) that fits."""
    full = render_passage(title, body)
    if count_tokens(full) <= max_tokens:
        return full
    lines = body.split("\n")
    kept: list[str] = []
    for line in lines:
        trial = render_passage(title, "\n".join(kept + [line]))
        if count_tokens(trial) <= max_tokens:
            kept.append(line)
            continue
        part = ""
        for sent in split_sentences(line) or [line]:
            cand = (part + " " + sent).strip()
            if count_tokens(render_passage(title, "\n".join(kept + [cand]))) <= max_tokens:
                part = cand
            else:
                break
        if not part:
            for w in line.split():
                cand = (part + " " + w).strip()
                if count_tokens(render_passage(title, "\n".join(kept + [cand]))) <= max_tokens:
                    part = cand
                else:
                    break
        if part:
            kept.append(part)
        break
    out = render_passage(title, "\n".join(kept))
    while count_tokens(out) > max_tokens and len(out) > 1:  # pathological title
        out = out[: int(len(out) * 0.9)]
    return out


# ------------------------------------------------------------------------- title table
# ``TitleTable`` (primary topics, person names, who-frames) lives in ``bes.library.titles``.


def query_title_key(query: str) -> str:
    """Content terms of a whole query without definition words (the look-up key before the
    identity frames of ``identity_frame``; kept for callers that key whole messages)."""
    return " ".join(t for t in terms(query) if t not in DEF_WORDS)


# ------------------------------------------------------------------------- library

KINDS = ("wiki", "qa")


@dataclass
class _Cand:
    kind: str
    pid: int
    score: float
    feats: dict
    prob: float = 0.0


#: the JSON file that marks a built library directory (this module's, or ``bes.library.answers``)
LIBRARY_MARKERS = ("library.json", "answers.json")


def library_marker(path: str | os.PathLike) -> str:
    """Path of the JSON file that marks ``path`` as a built library: ``library.json`` (this
    module) or ``answers.json`` (``bes.library.answers``); ``""`` when it holds neither."""
    for name in LIBRARY_MARKERS:
        f = os.path.join(str(path), name)
        if os.path.isfile(f):
            return f
    return ""


class Library:
    """Packed passages + stored truncated index + title table + qa layer + gate."""

    def __init__(self, packed: Packed, index: TruncatedIndex, cfg: RetrieveCfg = RetrieveCfg(),
                 count_tokens: CountTokens = default_count_tokens, meta: dict | None = None,
                 qa=None, titles: TitleTable | None = None, gate=None, path: str | None = None):
        if packed.n_passages != index.n_docs:
            raise ValueError("index and packed library disagree on the number of passages")
        self.packed = packed
        self.index = index
        self.cfg = cfg
        self.count_tokens = count_tokens
        self.meta = meta or {}
        self.qa = qa
        self.titles = titles
        self.gate = gate
        self.path = path
        self._title_map: dict[str, int] | None = None
        self._cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------ pickling (workers)
    def __getstate__(self) -> dict:
        if self.path is not None:
            return {"_reload": (self.path, self.count_tokens, self.cfg)}
        d = dict(self.__dict__)
        d["_cache"] = OrderedDict()
        d["_title_map"] = None
        return d

    def __setstate__(self, state: dict) -> None:
        if "_reload" in state:
            path, ct, cfg = state["_reload"]
            self.__dict__.update(Library.load(path, ct, cfg).__dict__)
        else:
            self.__dict__.update(state)

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, passages: Sequence[tuple[str, str]], out_dir: str | None = None,
              index_cfg: IndexCfg = IndexCfg(), pack_cfg: PackCfg = PackCfg(),
              cfg: RetrieveCfg = RetrieveCfg(), count_tokens: CountTokens = default_count_tokens,
              meta: dict | None = None, prior: Sequence[float] | None = None,
              qa=None, title_importance: Sequence[float] | None = None, gate=None,
              name_importance: Sequence[float] | None = None) -> "Library":
        """Build from (title, body) pairs; passages of one title must be contiguous.

        Passage ids follow the given order; title ids are assigned in order of appearance.
        ``prior`` (one positive value per passage) steers which postings the truncated
        index keeps (``bes.library.index``); None means uniform. ``qa`` is an optional
        ``QAStore``; ``title_importance`` (one value per title) decides which article is the
        primary topic of a shared title key, ``name_importance`` (default the same) which
        person a who-frame means (``bes.library.titles``); ``gate`` an optional fitted
        ``GateModel``.
        """
        from bes.library.titles import article_texts

        titles: list[str] = []
        tids: list[int] = []
        last = None
        for title, _ in passages:
            if title != last:
                titles.append(title)
                last = title
            tids.append(len(titles) - 1)
        if len(set(titles)) != len(titles):
            raise ValueError("passages of one title must be contiguous")
        packed = pack([b for _, b in passages], tids, titles, pack_cfg)
        index = TruncatedIndex.build(passages, index_cfg, prior)
        table = TitleTable.build(titles, title_importance, article_texts(passages),
                                 name_importance=name_importance)
        lib = cls(packed, index, cfg, count_tokens, meta, qa=qa, titles=table, gate=gate)
        if out_dir is not None:
            lib.save(out_dir)
        return lib

    def save(self, out_dir: str) -> None:
        self.packed.save(out_dir)
        self.index.save(out_dir)
        if self.titles is not None:
            for k in TitleTable.ARRAYS:
                _atomic_save_npy(os.path.join(out_dir, f"titlekey_{k}.npy"), np.asarray(getattr(self.titles, k)))
        if self.qa is not None:
            self.qa.save(out_dir)
        if self.gate is not None:
            self.gate.save(out_dir)
        meta = dict(self.meta)
        meta.update(retrieve_cfg=dataclasses.asdict(self.cfg), accounting=self.accounting(),
                    has_qa=self.qa is not None, has_gate=self.gate is not None,
                    has_titles=self.titles is not None)
        _atomic_write(os.path.join(out_dir, "library.json"), json.dumps(meta, indent=1, sort_keys=True).encode())

    @classmethod
    def load(cls, path: str, count_tokens: CountTokens = default_count_tokens,
             cfg: RetrieveCfg | None = None, mmap: bool = True) -> "Library":
        """Open a saved library (arrays memory-mapped; nothing is decompressed or derived
        until a query or ``passage`` call needs it). A directory that holds an answers library
        (``bes.library.answers``, marked by ``answers.json``) opens as an ``AnswerLibrary``,
        which serves the same contract."""
        from bes.library.gate import GateModel
        from bes.library.qa import QAStore

        if os.path.exists(os.path.join(path, "answers.json")):
            from bes.library.answers import AnswerLibrary

            return AnswerLibrary.load(path, mmap)          # type: ignore[return-value]
        meta = json.loads(open(os.path.join(path, "library.json")).read())
        if cfg is None:
            rc = dict(meta.get("retrieve_cfg", {}))
            g = dict(rc.pop("gate", {}))
            g["allowed_types"] = tuple(g.get("allowed_types", GateCfg().allowed_types))
            known = {f.name for f in dataclasses.fields(RetrieveCfg)}
            cfg = RetrieveCfg(**{k: v for k, v in rc.items() if k in known}, gate=GateCfg(**g))
        titles = None
        if os.path.exists(os.path.join(path, "titlekey_hash.npy")):
            mode = "r" if mmap else None
            titles = TitleTable(*(np.load(os.path.join(path, f"titlekey_{k}.npy"), mmap_mode=mode)
                                  for k in TitleTable.ARRAYS))
        qa = QAStore.load(path, mmap) if os.path.exists(os.path.join(path, "qa.json")) else None
        gate = GateModel.load(path) if os.path.exists(os.path.join(path, "gate_coef.npy")) else None
        return cls(Packed.load(path, mmap), TruncatedIndex.load(path, mmap), cfg, count_tokens, meta,
                   qa=qa, titles=titles, gate=gate, path=path)

    # ------------------------------------------------------------------ accounting
    def accounting(self) -> dict[str, int]:
        """Exact stored numbers of every table: wiki text blocks, directory, titles, index,
        title look-up, qa layer, gate; ``total_numbers`` is their sum."""
        a = dict(self.packed.accounting())
        a.update(self.index.accounting())
        a["title_table_numbers"] = self.titles.numbers() if self.titles is not None else 0
        a["qa_numbers"] = 0
        if self.qa is not None:
            a.update(self.qa.accounting())
        a["gate_numbers"] = self.gate.numbers() if self.gate is not None else 0
        a["wiki_numbers"] = a["library_numbers"] + a["index_numbers"] + a["title_table_numbers"]
        a["total_numbers"] = a["wiki_numbers"] + a["qa_numbers"] + a["gate_numbers"]
        return a

    def read_bound(self, max_query_chars: int = 1000) -> dict[str, int]:
        """Worst-case stored numbers one ``retrieve()`` can read, from the configuration and
        the largest stored blocks (a query of more than ``max_query_chars`` characters is a
        task and reads nothing, so it has at most ``max_query_chars // 2`` distinct terms)."""
        import math as _m

        def lg(n: int) -> int:
            return max(1, _m.ceil(_m.log2(n + 1)))

        nt = max_query_chars // 2
        ic = self.index.cfg
        g = 1 << ic.group_bits
        b = {"index": nt + ic.max_query_terms * (2 * ic.max_list + g + 2)}
        if self.titles is not None:
            b["title_lookup"] = self.titles.lookups_per_query() * (2 * lg(int(self.titles.hash.shape[0])) + 2)
        comp_t = np.diff(np.asarray(self.packed.tblock_off, dtype=np.int64))
        max_tb = int(numbers_of(int(comp_t.max())) + 8) if comp_t.size else 0
        comp = np.diff(np.asarray(self.packed.block_off, dtype=np.int64))
        max_b = int(numbers_of(int(comp.max())) + 8) if comp.size else 0
        b["title_blocks"] = 2 * max_tb
        b["text_blocks"] = max_b
        b["directory"] = 2 * 2 * lg(self.packed.n_titles) + 2 * 2 * 2 + 2 * lg(self.packed.n_blocks) + 2
        if self.qa is not None:
            qc = self.qa.cfg
            qg = 1 << qc.group_bits
            r = qc.read_postings
            b["qa_index"] = nt + qc.max_query_terms * (qg + 2) + 2 * r + r + (r + 1) // 2
            qcomp = np.diff(np.asarray(self.qa.blocks.block_off, dtype=np.int64))
            max_qb = int(numbers_of(int(qcomp.max())) + 8) if qcomp.size else 0
            n_read = max(2, int(self.cfg.qa_read_entries))
            b["qa_blocks"] = n_read * max_qb
            b["qa_directory"] = n_read * (2 * lg(self.qa.blocks.n_blocks) + 2)
        b["total"] = sum(b.values())
        return b

    # ------------------------------------------------------------------ ids
    @property
    def n_wiki(self) -> int:
        return self.packed.n_passages

    @property
    def n_qa(self) -> int:
        return 0 if self.qa is None else self.qa.n_entries

    @property
    def n_passages(self) -> int:
        return self.n_wiki + self.n_qa

    def kind_range(self, kind: str) -> range:
        """Contiguous passage ids of one kind (``"wiki"`` or ``"qa"``)."""
        if kind == "wiki":
            return range(0, self.n_wiki)
        if kind == "qa":
            return range(self.n_wiki, self.n_wiki + self.n_qa)
        raise ValueError(f"unknown kind {kind!r}")

    def kind_of(self, pid: int) -> str:
        if 0 <= pid < self.n_wiki:
            return "wiki"
        if self.n_wiki <= pid < self.n_passages:
            return "qa"
        raise IndexError(pid)

    # ------------------------------------------------------------------ blocks (LRU)
    def _block(self, kind: str, b: int) -> bytes:
        key = (kind, b)
        raw = self._cache.get(key)
        if raw is not None:
            self._cache.move_to_end(key)
            return raw
        raw = self.packed.read_block(b) if kind == "wiki" else self.qa.blocks.read_block(b)
        self._cache[key] = raw
        if len(self._cache) > self.cfg.cache_blocks:
            self._cache.popitem(last=False)
        return raw

    # ------------------------------------------------------------------ lookups
    def title_range(self, tid: int, log: ReadLog | None = None) -> range:
        """Passage ids of title ``tid`` (reads two ``title_first`` entries); empty for -1."""
        if tid < 0:
            return range(0)
        if log is not None:
            log.touch_dir("title_first", (tid, tid + 1), 2)
        return range(int(self.packed.title_first[tid]), int(self.packed.title_first[tid + 1]))

    def title_id_of(self, pid: int, log: ReadLog | None = None) -> int:
        """Title id of a wiki passage (-1 for a qa entry)."""
        if pid >= self.n_wiki:
            return -1
        t, probes = self.packed.title_of(pid)
        if log is not None:
            log.touch_dir("title_first", probes, 2)
        return t

    def title_text(self, tid: int, log: ReadLog | None = None) -> str:
        tpb = self.packed.cfg.titles_per_block
        tb = tid // tpb
        if log is not None and tb not in log.title_block_ids:
            log.title_block_ids.add(tb)
            log.title_blocks += self.packed.title_block_numbers(tb)
        return self.packed.read_title_block(tb)[tid % tpb]

    def find_title(self, title: str) -> int | None:
        """Title id of an exact title (training-side helper; builds a dict on first use)."""
        if self._title_map is None:
            self._title_map = {t: i for i, t in enumerate(self.packed.all_titles())}
        return self._title_map.get(title)

    def passage(self, pid: int, score: float = 0.0, log: ReadLog | None = None) -> Passage:
        """Read one entry: one text block (+ one title block for a wiki title), or one qa
        block. Decompressed blocks are kept in a small per-process LRU cache."""
        pid = int(pid)
        if pid >= self.n_wiki:
            return self._qa_passage(pid - self.n_wiki, score, log)
        b, s, e, probes = self.packed.locate(pid)
        if log is not None:
            log.touch_dir("block_first", probes, 2)
            nxt = [pid + 1] if pid + 1 < int(self.packed.block_first[b + 1]) else []
            log.touch_dir("pas_off", [pid] + nxt, 1)
            if b not in log.blocks:
                log.blocks.add(b)
                log.text_blocks += self.packed.block_numbers(b)
        body = self._block("wiki", b)[s:e].decode("utf-8")
        tid = self.title_id_of(pid, log)
        title = self.title_text(tid, log)
        text = cap_tokens(title, body, self.cfg.max_tokens, self.count_tokens)
        return Passage(pid, tid, title, body, score, text, "wiki")

    def _qa_entry(self, e: int, log: ReadLog | None) -> tuple[str, str]:
        from bes.library.qa import parse_entry

        blocks = self.qa.blocks
        b, s, t, probes = blocks.locate(e)
        if log is not None:
            log.touch_dir("qa_block_first", probes, 2)
            nxt = [e + 1] if e + 1 < int(blocks.block_first[b + 1]) else []
            log.touch_dir("qa_entry_off", [e] + nxt, 1)
            if b not in log.qa_block_ids:
                log.qa_block_ids.add(b)
                log.qa_blocks += blocks.block_numbers(b)
        return parse_entry(self._block("qa", b)[s:t])

    def _qa_passage(self, e: int, score: float, log: ReadLog | None) -> Passage:
        from bes.library.qa import render_qa

        if self.qa is None or not 0 <= e < self.qa.n_entries:
            raise IndexError(e + self.n_wiki)
        q, a = self._qa_entry(e, log)
        return Passage(self.n_wiki + e, -1, q, a, score, render_qa(q, a), "qa")

    # ------------------------------------------------------------------ scoring
    def search(self, query: str, exclude_pids: Iterable[int] = (), exclude_titles: Iterable[int] = (),
               log: ReadLog | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(pids, scores) of wiki passages ranked by the stored index (score desc, pid asc);
        scores in BM25 units. ``exclude_pids`` / ``exclude_titles`` remove passages (e.g. the
        passage a training text came from, or its whole article) before ranking."""
        return self.search_max(query, exclude_pids, exclude_titles, log)[:2]

    def search_max(self, query: str, exclude_pids: Iterable[int] = (),
                   exclude_titles: Iterable[int] = (), log: ReadLog | None = None
                   ) -> tuple[np.ndarray, np.ndarray, float]:
        """``search`` plus ``s_max`` (BM25 units), the best score any passage could reach."""
        ir = IndexRead()
        pids, sc, smax = self.index.score_with_max(query, ir)
        if log is not None:
            log.index_df += ir.df_numbers
            log.index_offsets += ir.offset_numbers
            log.index_postings += ir.posting_numbers
        ex = set(int(p) for p in exclude_pids)
        ranges = [self.title_range(int(t), log) for t in exclude_titles if int(t) >= 0]
        if (ex or ranges) and pids.size:
            mask = np.ones(pids.shape[0], dtype=bool)
            if ex:
                mask &= ~np.isin(pids, np.fromiter(ex, dtype=np.int64))
            for r in ranges:
                mask &= ~((pids >= r.start) & (pids < r.stop))
            pids, sc = pids[mask], sc[mask]
        return pids, sc.astype(np.float64) * self.index.scale, float(smax) * self.index.scale

    def gate_features(self, query: str, pids: np.ndarray, scores: np.ndarray,
                      log: ReadLog | None = None, smax: float | None = None) -> dict:
        """Rule-gate features of the BM25 top passage (reads its title)."""
        if pids.size == 0:
            return {"s1": 0.0, "s2": 0.0, "margin": 0.0, "title_match": 0.0, "coverage": 0.0,
                    "qtype": query_type(query), "tid": -1}
        f = self._wiki_feats(query, int(pids[0]), float(scores[0]), pids, scores, smax, log)
        return {k: f[k] for k in ("s1", "s2", "margin", "title_match", "coverage", "qtype", "tid")}

    def gate_pass(self, f: dict, gate: GateCfg | None = None) -> bool:
        """The hand-set rule gate on ``gate_features``."""
        g = gate or self.cfg.gate
        if f["qtype"] not in g.allowed_types or f["s1"] < g.min_score:
            return False
        if f.get("coverage", 1.0) < g.min_coverage:
            return False
        if f["margin"] >= g.min_margin:
            return True
        return f["title_match"] >= g.min_title_match and f["margin"] >= g.min_margin_title

    def _wiki_feats(self, query: str, pid: int, s1: float, pids: np.ndarray, scores: np.ndarray,
                    smax: float | None, log: ReadLog | None, exact: bool = False,
                    variant: bool = False, person: bool = False) -> dict:
        tid = self.title_id_of(pid, log)
        r = self.title_range(tid, log)
        s2 = 0.0
        m = self.cfg.gate.margin_candidates
        for p, s in zip(pids[:m], scores[:m]):
            if not (r.start <= int(p) < r.stop):
                s2 = float(s)
                break
        title = self.title_text(tid, log)
        tt = set(terms(strip_disambiguator(title)))
        qt = set(terms(query))
        tm = (len(tt & qt) / len(tt)) if tt else 0.0
        cov = 1.0 if smax is None else (min(1.0, s1 / smax) if smax > 0 else 0.0)
        margin = (s1 - s2) / s1 if s1 > 0 else (-1.0 if s2 > 0 else 0.0)
        return {"s1": s1, "s2": s2, "margin": max(-1.0, margin), "title_match": tm, "coverage": cov,
                "qtype": query_type(query), "tid": tid, "exact_title": exact, "variant": variant,
                "person": person, "lead": pid == r.start, "n_terms": len(set(terms(query))), "cross": False}

    def title_hit(self, query: str, log: ReadLog | None = None) -> tuple[int, bool, bool] | None:
        """``(title id, variant, person)`` of the exact-title look-up, or None. Only identity
        requests look up (``identity_frame``; who-frames ask the person entries first); a
        one-term key needs an explicit frame or a capitalised name ("What is a pointer?",
        "Galileo", not "try again"); an object with inner stopwords must equal the owner's
        title ("for loop" is not "While loop"); the hit is verified against the stored title
        (hash collisions). The first key form with an entry decides."""
        if self.titles is None:
            return None
        frame = identity_frame(query)
        if frame is None:
            return None
        kind, obj = frame
        keys = object_keys(obj)
        if not keys:
            return None
        if kind == "bare" and len(keys[0].split()) == 1 and not any(
                w[:1].isupper() for w in case_tokens(obj) if w.lower() not in STOPWORDS):
            return None
        who = kind == "who"
        hit, key = None, ""
        for key in (keys[::-1] if who else keys):        # person names are stored unstemmed
            hit = self.titles.lookup(key, log, who=who)
            if hit is not None:
                break
        if hit is None or not 0 <= hit[0] < self.packed.n_titles:
            return None
        title = self.title_text(hit[0], log)
        if not verify(key, title):
            return None
        xs = surface(obj)
        if any(w in STOPWORDS for w in xs) and xs != surface(title):
            return None
        return hit

    # ------------------------------------------------------------------ candidates
    def _wiki_candidates(self, query: str, exclude_pids, exclude_titles, log: ReadLog | None,
                         k: int) -> list[_Cand]:
        pids, sc, smax = self.search_max(query, exclude_pids, exclude_titles, log)
        out: list[_Cand] = []
        hit = self.title_hit(query, log)
        ex_p = set(int(p) for p in exclude_pids)
        ex_t = set(int(t) for t in exclude_titles)
        if hit is not None and hit[0] not in ex_t:
            tid, variant, person = hit
            lead = self.title_range(tid, log).start
            if lead not in ex_p:
                hitpos = np.flatnonzero(pids == lead)
                s = float(sc[hitpos[0]]) if hitpos.size else 0.0
                out.append(_Cand("wiki", lead, s, self._wiki_feats(
                    query, lead, s, pids, sc, smax, log, exact=True, variant=variant, person=person)))
        for p, s in zip(pids[: k + 1], sc[: k + 1]):
            if len(out) >= k:
                break
            if any(c.pid == int(p) for c in out):
                continue
            out.append(_Cand("wiki", int(p), float(s), self._wiki_feats(query, int(p), float(s), pids, sc,
                                                                       smax, log)))
        return out[:k]

    def title_key_of(self, tid: int, log: ReadLog | None = None) -> str:
        return title_key(self.title_text(tid, log))

    def _qa_candidates(self, query: str, exclude_pids, log: ReadLog | None, k: int) -> list[_Cand]:
        """Up to ``k`` qa candidates. The best stored questions by ``sym`` are read in order
        (at most ``cfg.qa_read_entries``); one that asks for a different fact than the user's
        question (``bes.library.qa.contrast_mismatch``: other numbers, order, degree,
        relation or polarity words, or names missing on either side) is dropped."""
        from bes.library.qa import QARead, answer_key, contrast, contrast_mismatch

        if self.qa is None:
            return []
        ex = [int(p) - self.n_wiki for p in exclude_pids if int(p) >= self.n_wiki]
        qr = QARead()
        m = self.qa.search(query, ex, qr)
        if log is not None:
            log.qa_index += qr.total
        lo = self.cfg.qa_min_sym - 1e-12
        if m.size == 0 or m.sym[0] < lo:
            return []
        user = contrast(query)
        kept: list[tuple[int, str, str]] = []          # (rank, answer, answer key)
        n_read = min(m.size, max(1, int(self.cfg.qa_read_entries)))
        for j in range(n_read):
            if m.sym[j] < min(lo, self.cfg.agree_min_sym - 1e-12):
                break
            question, answer = self._qa_entry(int(m.entries[j]), log)
            if contrast_mismatch(user, contrast(question)) is None:
                kept.append((j, answer, answer_key(answer)))
        out = []
        for j, answer, akey in [x for x in kept if m.sym[x[0]] >= lo][:k]:
            others = [x for x in kept if x[0] != j]
            if others:
                o_sym = float(m.sym[others[0][0]])
            else:                                       # the best candidate not read, if any
                o_sym = float(m.sym[n_read]) if m.size > n_read else 0.0
            o = others[0] if others and m.sym[others[0][0]] >= self.cfg.agree_min_sym - 1e-12 else None
            f = {"sym": float(m.sym[j]), "cov_q": float(m.cov_q[j]), "cov_e": float(m.cov_e[j]),
                 "type_match": bool(m.type_match[j]), "no_wh": m.qtype == 0,
                 "sym_margin": float(m.sym[j]) - o_sym, "agree": bool(o is not None and o[2] == akey),
                 "disagree": bool(o is not None and o[2] != akey), "wq": m.wq, "n_terms": m.n_terms,
                 "cross": False, "answer": answer}
            out.append(_Cand("qa", self.n_wiki + int(m.entries[j]), float(m.sym[j]), f))
        return out

    def _cross(self, wiki: list[_Cand], qa: list[_Cand], log: ReadLog | None) -> None:
        """Mark wiki/qa candidates whose qa answer occurs in the wiki passage text."""
        if not wiki or not qa:
            return
        texts = {}
        for c in wiki:
            texts[c.pid] = " " + " ".join(tokens(self.passage(c.pid, c.score, log).text)) + " "
        for q in qa:
            ak = " ".join(tokens(q.feats["answer"]))
            if not ak:
                continue
            for c in wiki:
                if (" " + ak + " ") in texts[c.pid]:
                    c.feats["cross"] = True
                    if c is wiki[0]:
                        q.feats["cross"] = True

    def candidates(self, query: str, k: int = 1, exclude_pids: Iterable[int] = (),
                   exclude_titles: Iterable[int] = (), log: ReadLog | None = None,
                   kinds: Sequence[str] | None = None, cross: bool | None = None) -> list[_Cand]:
        """Up to ``k`` candidates of each allowed kind with their gate features and
        probabilities (probability 0 without a gate model), best first. ``cross`` (default:
        when a gate model is present) computes the answer-in-passage features, which reads
        the wiki candidates' text blocks."""
        kinds = KINDS if kinds is None else tuple(kinds)
        for kd in kinds:
            if kd not in KINDS:
                raise ValueError(f"unknown kind {kd!r}")
        exclude_pids = [int(p) for p in exclude_pids]
        exclude_titles = [int(t) for t in exclude_titles]
        wiki = self._wiki_candidates(query, exclude_pids, exclude_titles, log, k) if "wiki" in kinds else []
        qa = self._qa_candidates(query, exclude_pids, log, k) if "qa" in kinds else []
        if cross or (cross is None and self.gate is not None):
            self._cross(wiki, qa, log)
        if self.gate is not None:
            for c in wiki + qa:
                c.prob = self.gate.prob(c.kind, c.feats)
        cands = wiki + qa
        cands.sort(key=lambda c: (-c.prob, KINDS.index(c.kind), -c.score, c.pid))
        return cands

    # ------------------------------------------------------------------ public api
    def retrieve(self, query: str, exclude_pids: Iterable[int] = (), exclude_titles: Iterable[int] = (),
                 log: ReadLog | None = None, kinds: Sequence[str] | None = None,
                 context: Sequence[str] | None = None) -> Passage | None:
        """At most one entry for a user message, or None when the gate does not pass.

        ``kinds=None`` considers both kinds; ``("wiki",)`` or ``("qa",)`` restricts.
        ``context``: the earlier user turns of the conversation (oldest first; None or empty
        for a first turn). A later turn that is not self-contained (``turn_type``:
        "What is its capital?", "And cats?", "Who is the author?") retrieves nothing; neither
        does chat ("merry christmas", "thanks, that was helpful") in any turn."""
        if turn_type(query, context) not in self.cfg.gate.allowed_types:
            return None
        if self.gate is None:              # hand-set rule gate on the BM25 passage
            if kinds is not None and "wiki" not in kinds:
                return None
            pids, sc, smax = self.search_max(query, exclude_pids, exclude_titles, log)
            f = self.gate_features(query, pids, sc, log, smax)
            if not self.gate_pass(f):
                return None
            return self.passage(int(pids[0]), float(sc[0]), log)
        cands = self.candidates(query, 1, exclude_pids, exclude_titles, log, kinds)
        if not cands or cands[0].prob < self.gate.threshold:
            return None
        c = cands[0]
        return dataclasses.replace(self.passage(c.pid, c.score, log), prob=c.prob)

    def retrieve_topk(self, query: str, k: int, exclude_pids: Iterable[int] = (),
                      exclude_titles: Iterable[int] = (), log: ReadLog | None = None,
                      kinds: Sequence[str] | None = None) -> list[Passage]:
        """The k best entries without the gate. Without a gate model: wiki passages in BM25
        order. With one: up to k candidates of each allowed kind, merged by calibrated
        probability (``Passage.prob``)."""
        if k <= 0:
            return []
        if self.gate is None:
            if kinds is not None and "wiki" not in kinds:
                cands = self.candidates(query, k, exclude_pids, exclude_titles, log, kinds)
                return [self.passage(c.pid, c.score, log) for c in cands[:k]]
            pids, sc = self.search(query, exclude_pids, exclude_titles, log)
            return [self.passage(int(p), float(s), log) for p, s in zip(pids[:k], sc[:k])]
        cands = self.candidates(query, k, exclude_pids, exclude_titles, log, kinds)
        return [dataclasses.replace(self.passage(c.pid, c.score, log), prob=c.prob) for c in cands[:k]]
