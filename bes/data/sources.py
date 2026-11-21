"""Registry of training datasets on the Hugging Face Hub (``docs/SPEC_MODEL_TRAIN.md`` section 6).

Every entry was checked against the Hub on ``VERIFIED_ON`` (dataset API, repository tree and the
datasets-server ``info`` / ``first-rows`` endpoints): repository id, config, split, file layout,
column names, licence and size. ``docs/DATA_SOURCES.md`` records what was checked and what did
not check out.

An entry says *where* the data is (``repo``, ``revision``, ``files`` patterns inside the
repository, ``columns`` to read) and *how* to read one record: ``adapter(record) -> Item`` maps
one parquet row (a ``dict`` of the listed columns) to

- ``str``                      for ``fmt == "text"``      (a raw document),
- ``list[Message]``            for ``fmt == "messages"``  (a conversation),
- ``Pair``                     for ``fmt == "pair"``      (a preference pair),
- ``None``                     when the record is filtered out.

Messages use the format of ``bes.data.chat_format``: ``{"role": r, "content": c}`` with ``r`` in
``ROLES`` and turn order ``[system] ([library] user assistant)+``. A ``library`` turn holds the
retrieved passage(s) -- title first, several passages joined by ``PASSAGE_SEP`` -- and comes
immediately before the user turn it serves (``docs/SPEC.md`` section 1). ``c`` is a string,
except that assistant content with calculator spans is a list of parts
``{"type": "text", "text": ...}`` / ``{"type": "calc", "expr": ..., "result": ...}``; such
content is built from the textual markup ``<<expr=result>>`` (``CALC_OPEN``, ``CALC_SEP``,
``CALC_CLOSE``) by ``calc_parts``.

``weight`` is the default mixture weight *within* the entry's ``kind`` (weights of a kind sum to
1; ``library`` entries are not mixed and have weight 0). ``nonpermissive`` marks licences that
restrict commercial use; ``mixture_weights(kind, permissive_only=True)`` drops those entries
and hands their weight to ``fallback``.
"""
from __future__ import annotations

import re
import urllib.parse
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Union

VERIFIED_ON = "2026-11-21"

KINDS = ("base", "chat", "rag", "math", "prefs", "library")
FORMATS = ("text", "messages", "pair")
ROLES = ("system", "user", "assistant", "library")

#: Calculator-span markup inside assistant content: ``<<expr=result>>``.
CALC_OPEN, CALC_SEP, CALC_CLOSE = "<<", "=", ">>"

#: At most this many passages per library turn, each at most this many characters.
MAX_PASSAGES = 3
MAX_PASSAGE_CHARS = 6000
#: Separator between the passages of one library turn.
PASSAGE_SEP = "\n\n---\n\n"

Message = dict[str, Any]   # {"role": str, "content": str | list[part]}


@dataclass(frozen=True)
class Pair:
    """A preference pair: two complete conversations that share their prompt."""

    chosen: list[Message]
    rejected: list[Message]


Item = Union[str, list[Message], Pair, None]
Adapter = Callable[[dict[str, Any]], Item]


@dataclass(frozen=True)
class Source:
    """One dataset (or one filtered view of a dataset) in the registry."""

    name: str                       # registry key; used in paths ``tok/<name>/``
    repo: str                       # Hugging Face dataset id
    config: str | None              # datasets-server config name (documentation only)
    split: str                      # split the ``files`` patterns select
    kind: str                       # one of KINDS
    fmt: str                        # one of FORMATS: what ``adapter`` returns
    files: tuple[str, ...]          # fnmatch patterns of repository paths at ``revision``
    columns: tuple[str, ...]        # parquet columns the adapter reads
    adapter: Adapter
    license: str                    # licence as stated by the card (plus upstream notes)
    nonpermissive: bool             # True when commercial use is restricted
    weight: float                   # default weight within ``kind``
    revision: str = "main"          # commit sha (pinned) or a branch / convert ref
    verified_sha: str = ""          # commit seen at verification
    n_files: int = 0                # matching files at verification
    approx_rows: int = 0            # rows of the split at verification
    approx_bytes: int = 0           # parquet bytes of the matching files at verification
    cap_bytes: int | None = None    # default download cap for the 24-hour run (None = all)
    fallback: str | None = None     # entry that inherits ``weight`` when this one is excluded
    notes: str = ""


# ------------------------------------------------------------------------------ helpers
_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}
_WS_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
#: Runs of spaces *between* two non-space characters (left by removed citation tags); leading
#: indentation is never touched.
_MULTI_SPACE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def _text(value: Any) -> str:
    """``value`` as a stripped string with ``\\r\\n`` normalised to ``\\n`` ('' for None)."""
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def stable_bucket(key: str, n: int) -> int:
    """Deterministic bucket in ``[0, n)`` of a string (CRC-32; independent of the process)."""
    return zlib.crc32(key.encode("utf-8")) % n


def cap_chars(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """``text`` cut to at most ``limit`` characters at a whitespace boundary (adds ``" …"``)."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit - 2)
    if cut < limit // 2:
        cut = limit - 2
    return text[:cut].rstrip() + " …"


def passage(title: str, body: str) -> str:
    """Content of a library turn: the title on the first line, then the passage."""
    title, body = _text(title), cap_chars(_text(body))
    return f"{title}\n\n{body}" if title else body


def wiki_title(url: str | None) -> str:
    """Article title of a Wikipedia / Wikibooks URL (``.../wiki/New_South_Wales`` -> ``New South Wales``)."""
    if not url:
        return ""
    path = urllib.parse.urlsplit(url.strip().split()[0]).path
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return urllib.parse.unquote(last).replace("_", " ").strip()


#: Allowed predecessor of each role; the turn grammar is ``[system] ([library] user assistant)+``.
_ALLOWED_AFTER = {"system": ("start",), "library": ("start", "system", "assistant"),
                  "user": ("start", "system", "library", "assistant"), "assistant": ("user",)}


def _clean_parts(content: Any) -> list[dict[str, str]] | None:
    """Validated assistant content parts (empty text parts removed); ``None`` if invalid."""
    parts = []
    for p in content:
        if not isinstance(p, dict):
            return None
        if p.get("type") == "text" and isinstance(p.get("text"), str):
            if p["text"]:
                parts.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "calc" and isinstance(p.get("expr"), str) and isinstance(p.get("result"), str):
            if not p["expr"].strip() or not p["result"].strip():
                return None
            parts.append({"type": "calc", "expr": p["expr"], "result": p["result"]})
        else:
            return None
    return parts or None


def is_valid_conversation(messages: Sequence[Message]) -> bool:
    """True when the roles follow ``[system] ([library] user assistant)+``."""
    state = "start"
    for m in messages:
        role = m.get("role")
        if role not in _ALLOWED_AFTER or state not in _ALLOWED_AFTER[role]:
            return False
        state = role
    return state == "assistant"


def clean_messages(messages: Iterable[Any] | None) -> list[Message] | None:
    """Normalise a conversation; ``None`` when it is unusable.

    - roles are lower-cased and aliases mapped (``human`` -> ``user``, ``gpt`` -> ``assistant``);
      any other role outside ``ROLES`` (tool, function, ...) rejects the conversation;
    - string contents are stripped; empty system / library turns are dropped, an empty user or
      assistant turn rejects the conversation; assistant content may be a list of parts
      (``_clean_parts``);
    - turns after the last assistant turn are dropped;
    - the result must follow ``[system] ([library] user assistant)+`` (``is_valid_conversation``,
      the grammar ``chat_format.check_structure`` enforces), otherwise it is rejected.
    """
    if messages is None:
        return None
    out: list[Message] = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        raw_role = str(m.get("role", "")).strip().lower()
        role = _ROLE_ALIASES.get(raw_role, raw_role)
        if role not in ROLES:
            return None
        raw = m.get("content")
        if role == "assistant" and isinstance(raw, (list, tuple)):
            content: Any = _clean_parts(raw)
        else:
            content = _text(raw) if raw is None or isinstance(raw, str) else None
        if not content:
            if role in ("system", "library") and isinstance(raw, (str, type(None))):
                continue
            return None
        out.append({"role": role, "content": content})
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out if is_valid_conversation(out) else None


def _qa(question: Any, answer: Any, passages: Sequence[str] = ()) -> list[Message] | None:
    """``[library, user, assistant]`` (one library turn holding ``passages`` joined by
    ``PASSAGE_SEP``; none when there are no passages), cleaned by ``clean_messages``."""
    passages = [p for p in passages if p]
    msgs = [{"role": "library", "content": PASSAGE_SEP.join(passages)}] if passages else []
    msgs += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return clean_messages(msgs)


def select_passages(n: int, supporting: Iterable[int], max_passages: int = MAX_PASSAGES,
                    n_distractors: int = 1) -> list[int]:
    """Indices (ascending, i.e. native order) of the passages to keep out of ``n``.

    The supporting passages come first (the first ``max_passages`` of them); if room is left,
    the first ``n_distractors`` non-supporting passages are added.
    """
    sup = sorted({i for i in supporting if 0 <= i < n})[:max_passages]
    room = max(0, min(n_distractors, max_passages - len(sup)))
    chosen = set(sup)
    distractors = [i for i in range(n) if i not in chosen][:room]
    return sorted(sup + distractors)


_WORD = re.compile(r"[a-z0-9]{4,}")


def overlap_rank(docs: Sequence[str], answer: str, k: int = MAX_PASSAGES) -> list[int]:
    """Indices (ascending) of the ``k`` documents sharing most distinct >=4-char words with ``answer``.

    Ties are broken by position, so the result is deterministic.
    """
    words = set(_WORD.findall(answer.lower()))
    scores = [len(words & set(_WORD.findall(d.lower()))) for d in docs]
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


# ------------------------------------------------------------------------------ adapters
def text_column(r: dict[str, Any]) -> str | None:
    """``record["text"]`` (ClimbMix, FineWeb-Edu)."""
    text = _text(r.get("text"))
    return text or None


def wiki_article(r: dict[str, Any]) -> str | None:
    """``title`` then a blank line then ``text`` (wikimedia/wikipedia)."""
    text = _text(r.get("text"))
    if not text:
        return None
    title = _text(r.get("title"))
    return f"{title}\n\n{text}" if title else text


def chat_messages(r: dict[str, Any]) -> list[Message] | None:
    """``record["messages"]`` (SmolTalk family, UltraChat)."""
    return clean_messages(r.get("messages"))


#: Tulu 3 subsets left out: non-commercial (No Robots, CC BY-NC 4.0), non-English (Aya), and
#: hard-coded identity answers that name another model.
TULU_EXCLUDED = frozenset({
    "ai2-adapt-dev/no_robots_converted",
    "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10",
})


def tulu_messages(r: dict[str, Any]) -> list[Message] | None:
    """Tulu 3 SFT mixture without the subsets in ``TULU_EXCLUDED``."""
    if r.get("source") in TULU_EXCLUDED:
        return None
    return clean_messages(r.get("messages"))


#: SYNTH exercises used as closed-book chat. "editing" is left out (some answers are cut off
#: after the first sentence) and so is "math mcq" (the answer is a bare option letter).
SYNTH_CHAT_EXERCISES = frozenset({"memorization", "mcq", "math exercise", "constrained writing",
                                  "creative writing", "cooking"})
_SYNTH_SOURCE = re.compile(r"<source_(\d+)>\s*(.*?)\s*</source_\1>", re.S)
_SYNTH_REF = re.compile(r"<ref\b[^>]*>.*?</ref>", re.S)
_SYNTH_REF_NAME = re.compile(r"<ref\s+name=\"source_(\d+)\"")
#: Rows generated from the dataset authors' hand-written "self-knowledge" seeds (``query_seed_url``
#: holds that label instead of a URL) answer in the first person as the authors' own models, a
#: foreign identity; they are dropped.
SYNTH_SELF_SEED = "self-knowledge"


def _synth_usable(r: dict[str, Any]) -> bool:
    """English and not generated from the self-knowledge seeds."""
    return r.get("language") == "en" and SYNTH_SELF_SEED not in (r.get("query_seed_url") or "").lower()


def _synth_open_book_half(synth_id: str) -> bool:
    """Memorization rows are split in two disjoint halves by a stable hash of ``synth_id``."""
    return stable_bucket(synth_id, 2) == 1


def synth_closed_book(r: dict[str, Any]) -> list[Message] | None:
    """PleIAs/SYNTH as chat: usable rows (``_synth_usable``) of ``SYNTH_CHAT_EXERCISES``,
    ``query`` -> ``synthetic_answer`` (the reasoning draft is dropped).

    Memorization rows of the open-book half (see ``synth_open_book``) are skipped.
    """
    if not _synth_usable(r) or r.get("exercise") not in SYNTH_CHAT_EXERCISES:
        return None
    if r["exercise"] == "memorization" and _synth_open_book_half(str(r.get("synth_id", ""))):
        return None
    return _qa(r.get("query"), r.get("synthetic_answer"))


def _strip_refs(answer: str) -> str:
    out = _SYNTH_REF.sub("", answer)
    out = _WS_BEFORE_PUNCT.sub(r"\1", out)
    return _MULTI_SPACE.sub(" ", out).strip()


def synth_open_book(r: dict[str, Any]) -> list[Message] | None:
    """PleIAs/SYNTH in retrieval format: library passage(s), question, answer; usable rows only
    (``_synth_usable``).

    - ``memorization`` rows of the open-book half with a seed text: the seed section (title
      from ``query_seed_url``) is the passage;
    - ``rag`` rows: the ``<source_k>`` blocks of ``constraints`` cited by the answer (plus one
      uncited distractor if room) are the passages; ``<ref ...>...</ref>`` citations are removed
      from the answer. Only ``source_1`` has a known title (the seed URL).
    """
    if not _synth_usable(r):
        return None
    exercise = r.get("exercise")
    answer = _text(r.get("synthetic_answer"))
    if exercise == "memorization":
        if not _synth_open_book_half(str(r.get("synth_id", ""))) or not _text(r.get("query_seed_text")):
            return None
        passages = [passage(wiki_title(r.get("query_seed_url")), r.get("query_seed_text"))]
    elif exercise == "rag":
        blocks = {int(k): body for k, body in _SYNTH_SOURCE.findall(r.get("constraints") or "")}
        if not blocks:
            return None
        keys = sorted(blocks)
        cited = {int(k) for k in _SYNTH_REF_NAME.findall(answer)}
        keep = select_passages(len(keys), [i for i, k in enumerate(keys) if k in cited])
        title1 = wiki_title(r.get("query_seed_url"))
        passages = [passage(title1 if keys[i] == 1 else "", blocks[keys[i]]) for i in keep]
        answer = _strip_refs(answer)
    else:
        return None
    return _qa(r.get("query"), answer, passages)


_GLAIVE_DOC = re.compile(r"Document:\s*(\d+)\s*\n")
_GLAIVE_BLOCK = re.compile(r"\s*Title:\s*(.*?)\s*\n\s*Text:\s*(.*)", re.S)
_GLAIVE_ANSWER = re.compile(r"\s*Cited Documents:\s*([^\n]*)\n\s*Answer:\s*(.*)", re.S)
_GLAIVE_CO = re.compile(r"</?co\b[^>]*>")


def glaive_documents(text: str) -> list[tuple[int, str, str]]:
    """``[(doc_id, title, body)]`` parsed from glaive's ``Document:k / Title: / Text:`` blocks."""
    parts = _GLAIVE_DOC.split(text or "")
    out = []
    for key, block in zip(parts[1::2], parts[2::2]):
        m = _GLAIVE_BLOCK.match(block)
        title, body = (m.group(1), m.group(2)) if m else ("", block)
        out.append((int(key), _text(title), _text(body)))
    return out


def glaive_rag(r: dict[str, Any]) -> list[Message] | None:
    """glaiveai/RAG-v1: the cited documents (+1 distractor if room) as library passages; the
    answer without its ``Cited Documents:`` header, ``Answer:`` prefix and ``<co:k>`` citation
    tags. The dataset's system prompt (identical on every row) is dropped."""
    m = _GLAIVE_ANSWER.match(r.get("answer") or "")
    docs = glaive_documents(r.get("documents") or "")
    if not m or not docs:
        return None
    cited = {int(x) for x in re.findall(r"\d+", m.group(1))}
    answer = _MULTI_SPACE.sub(" ", _GLAIVE_CO.sub("", m.group(2))).strip()
    keep = select_passages(len(docs), [i for i, (k, _, _) in enumerate(docs) if k in cited])
    return _qa(r.get("question"), answer, [passage(docs[i][1], docs[i][2]) for i in keep])


def rag_instruct(r: dict[str, Any]) -> list[Message] | None:
    """FreedomIntelligence/RAG-Instruct: of the 10 untitled documents keep the ``MAX_PASSAGES``
    with most word overlap with the answer (the dataset does not mark the supporting ones)."""
    docs = [_text(d) for d in (r.get("documents") or []) if _text(d)]
    answer = _text(r.get("answer"))
    if not docs:
        return None
    keep = overlap_rank(docs, answer)
    return _qa(r.get("question"), answer, [passage("", docs[i]) for i in keep])


def neural_bridge_rag(r: dict[str, Any]) -> list[Message] | None:
    """neural-bridge/rag-dataset-12000: ``context`` -> library, ``question``, ``answer``."""
    if not _text(r.get("context")):
        return None
    return _qa(r.get("question"), r.get("answer"), [passage("", r.get("context"))])


#: Answer used for SQuAD v2 questions that the passage cannot answer.
SQUAD_UNANSWERABLE = "The passage does not contain the answer to this question."


def squad_v2(r: dict[str, Any]) -> list[Message] | None:
    """rajpurkar/squad_v2: ``title`` + ``context`` -> library; first answer span, or
    ``SQUAD_UNANSWERABLE`` for the unanswerable questions."""
    answers = (r.get("answers") or {}).get("text") or []
    answer = _text(answers[0]) if answers else SQUAD_UNANSWERABLE
    title = _text(r.get("title")).replace("_", " ")
    return _qa(r.get("question"), answer, [passage(title, r.get("context"))])


def hotpot_distractor(r: dict[str, Any]) -> list[Message] | None:
    """hotpotqa/hotpot_qa (distractor): supporting paragraphs plus one distractor, in their
    native order, each ``title`` then the joined sentences; the short answer."""
    ctx = r.get("context") or {}
    titles, sentences = list(ctx.get("title") or []), list(ctx.get("sentences") or [])
    if not titles or len(titles) != len(sentences):
        return None
    support = set((r.get("supporting_facts") or {}).get("title") or [])
    keep = select_passages(len(titles), [i for i, t in enumerate(titles) if t in support])
    paras = [passage(titles[i], "".join(sentences[i])) for i in keep]
    return _qa(r.get("question"), r.get("answer"), paras)


def dolly(r: dict[str, Any]) -> list[Message] | None:
    """databricks/databricks-dolly-15k: ``context`` (when present) -> library turn."""
    ctx = _text(r.get("context"))
    return _qa(r.get("instruction"), r.get("response"), [passage("", ctx)] if ctx else [])


_GSM_CALC = re.compile(re.escape(CALC_OPEN) + r"([^<>]*)" + re.escape(CALC_CLOSE))
_GSM_FINAL = re.compile(r"\n?####\s*(.*?)\s*$", re.S)
_MATH_CHARS = set("0123456789.,+-−*/x×÷()% $")


def _norm_expr(s: str) -> str:
    s = re.sub(r"[\s$,]", "", s)
    return s.replace("×", "*").replace("x", "*").replace("÷", "/").replace("−", "-")


def _thousands(result: str) -> str | None:
    """``"22500"`` -> ``"22,500"`` (None when ``result`` is not a plain number)."""
    m = re.fullmatch(r"(-?)(\d+)(\.\d+)?", result)
    if not m:
        return None
    return f"{m.group(1)}{int(m.group(2)):,}{m.group(3) or ''}"


def _visible_copy(text: str, pos: int, result: str) -> str | None:
    """The copy of ``result`` written at ``text[pos:]`` (as is or with thousands commas), only if
    the number ends there (``<<10/2=5>>5.00`` has no copy of ``5``); else ``None``."""
    for v in (result, _thousands(result)):
        if v and text.startswith(v, pos):
            nxt = text[pos + len(v):pos + len(v) + 2]
            if not (nxt[:1].isdigit() or (nxt[:1] in ".," and nxt[1:2].isdigit())):
                return v
    return None


def gsm8k_calc_text(text: str) -> str:
    """Rewrite GSM8K's ``a op b = <<a op b=c>>c`` into one span ``<<a op b=c>>``.

    For every annotation ``<<expr=result>>``:
    1. the visible copy of ``result`` right after it (also ``22,500`` for ``22500``; the number
       must end there) is removed; if there is none, the annotation is dropped and the visible
       text kept as it is;
    2. a visible ``lhs =`` (optionally followed by ``$``) right before it is removed when ``lhs``
       equals ``expr`` up to spaces, ``$``, thousands commas and ``x``/``×``/``÷`` for ``*``/``/``;
       otherwise the visible text before the span is kept.
    Annotations whose inside is not exactly ``expr=result`` are dropped.
    """
    out: list[str] = []
    pos = 0
    for m in _GSM_CALC.finditer(text):
        before = text[pos:m.start()]
        inner = m.group(1)
        after = m.end()
        if inner.count(CALC_SEP) != 1:
            out.append(before)
            pos = after
            continue
        expr, result = (s.strip() for s in inner.split(CALC_SEP))
        visible = _visible_copy(text, after, result) if result else None
        if not expr or not result or visible is None:
            out.append(before)          # keep the visible text, drop the annotation
            pos = after
            continue
        head = before.rstrip()
        dollar = head.endswith("$")
        if dollar:
            head = head[:-1].rstrip()
        if head.endswith("="):
            lhs_end = len(head) - 1
            start = lhs_end
            while start > 0 and head[start - 1] in _MATH_CHARS:
                start -= 1
            while start < lhs_end and head[start] == " ":      # spaces stay with the prose
                start += 1
            # an opening parenthesis that the left-hand side does not close belongs to the prose
            while head[start:lhs_end].count("(") > head[start:lhs_end].count(")"):
                start = head.index("(", start) + 1
            lhs = head[start:lhs_end]
            if lhs.strip() and _norm_expr(lhs) == _norm_expr(expr):
                before = head[:start] + ("$" if dollar else "")
        out.append(before + f"{CALC_OPEN}{expr}{CALC_SEP}{result}{CALC_CLOSE}")
        pos = after + len(visible)
    out.append(text[pos:])
    return "".join(out)


def gsm8k_solution(answer: str) -> str:
    """GSM8K solution with ``<<expr=result>>`` spans and ``#### N`` replaced by ``The answer is N.``"""
    answer = _text(answer)
    m = _GSM_FINAL.search(answer)
    final = m.group(1).strip() if m else ""
    body = answer[:m.start()] if m else answer
    body = gsm8k_calc_text(body).rstrip()
    return f"{body}\nThe answer is {final}." if final else body


def _trim_zeros(result: str) -> str:
    """``"144.00"`` -> ``"144"``, ``"2.50"`` -> ``"2.5"`` (other strings unchanged)."""
    if re.fullmatch(r"-?\d+\.\d*", result):
        result = result.rstrip("0").rstrip(".")
    return result


def calc_parts(text: str) -> str | list[dict[str, str]]:
    """Content parts for ``chat_format``: every ``<<expr=result>>`` becomes
    ``{"type": "calc", "expr": expr, "result": result}`` (trailing decimal zeros of the result
    trimmed), the text around it ``{"type": "text", "text": ...}``. Returns ``text`` itself when
    it has no well-formed span."""
    parts: list[dict[str, str]] = []
    pos = 0
    for m in _GSM_CALC.finditer(text):
        inner = m.group(1)
        if inner.count(CALC_SEP) != 1:
            continue
        expr, result = (s.strip() for s in inner.split(CALC_SEP))
        if not expr or not result:
            continue
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        parts.append({"type": "calc", "expr": expr, "result": _trim_zeros(result)})
        pos = m.end()
    if not parts:
        return text
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    return parts


def gsm8k(r: dict[str, Any]) -> list[Message] | None:
    """openai/gsm8k (main): question -> solution whose assistant content is a list of text and
    calculator parts (see ``gsm8k_calc_text`` and ``calc_parts``)."""
    return _qa(r.get("question"), calc_parts(gsm8k_solution(r.get("answer") or "")))


def tinygsm(r: dict[str, Any]) -> list[Message] | None:
    """TinyGSM/TinyGSM: question -> the Python program as a fenced code block (not executed)."""
    code = _text(r.get("code"))
    return _qa(r.get("question"), f"```python\n{code}\n```" if code else "")


def orca_math(r: dict[str, Any]) -> list[Message] | None:
    """microsoft/orca-math-word-problems-200k: question -> answer."""
    return _qa(r.get("question"), r.get("answer"))


def ultrafeedback_pair(r: dict[str, Any]) -> Pair | None:
    """HuggingFaceH4/ultrafeedback_binarized (train_prefs): ``chosen`` vs ``rejected``.

    Pairs with ``score_chosen <= score_rejected`` (ties) or with different prompts are dropped.
    """
    try:
        if not float(r.get("score_chosen")) > float(r.get("score_rejected")):
            return None
    except (TypeError, ValueError):
        return None
    chosen, rejected = clean_messages(r.get("chosen")), clean_messages(r.get("rejected"))
    if chosen is None or rejected is None:
        return None
    def first_prompt(conv: list[Message]) -> list[Message]:
        return [m for m in conv if m["role"] != "assistant"][:1]

    if first_prompt(chosen) != first_prompt(rejected):
        return None
    return Pair(chosen=chosen, rejected=rejected)


# ------------------------------------------------------------------------------ registry
_SMOLTALK_SHA = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
_SMOLTALK_LICENSE = "apache-2.0 (card text; the YAML header has no licence field)"
_WIKI_SHA = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
_SYNTH_SHA = "0d6813a2966662c39f22f0b9af28a0c1c9f7a437"
_SYNTH_COLUMNS = ("synth_id", "language", "exercise", "query", "query_seed_url",
                  "query_seed_text", "constraints", "synthetic_answer")
_SYNTH_LICENSE = "cc-by-4.0 (card); seed text CC BY-SA 4.0 / CC0 (column seed_license)"
_CONVERT = "refs/convert/parquet"
_GB = 10 ** 9


def _smoltalk(name: str, config: str, weight: float, n_files: int, rows: int, size: int,
              notes: str) -> Source:
    return Source(
        name=name, repo="HuggingFaceTB/smoltalk", config=config, split="train", kind="chat",
        fmt="messages", files=(f"data/{config}/train-*.parquet",), columns=("messages",),
        adapter=chat_messages, license=_SMOLTALK_LICENSE, nonpermissive=False, weight=weight,
        revision=_SMOLTALK_SHA, verified_sha=_SMOLTALK_SHA, n_files=n_files, approx_rows=rows,
        approx_bytes=size, notes=notes)


SOURCES: tuple[Source, ...] = (
    # ---------------------------------------------------------------------- base text
    Source(
        name="climbmix", repo="karpathy/climbmix-400b-shuffle", config="default", split="train",
        kind="base", fmt="text", files=("shard_*.parquet",), columns=("text",),
        adapter=text_column,
        license="card: mit; upstream nvidia/Nemotron-ClimbMix: cc-by-nc-4.0",
        nonpermissive=True, weight=0.85, fallback="fineweb_edu",
        revision="915333b4f8b8684f39aeaafea600fea6f43fb703",
        verified_sha="915333b4f8b8684f39aeaafea600fea6f43fb703",
        n_files=6543, approx_rows=6543 * 86016, approx_bytes=599_758_882_644,
        cap_bytes=30 * _GB,
        notes="Re-shuffled ClimbMix; ~92 MB per shard, 84 row groups of 1024 rows. The card says "
              "MIT but the upstream dataset is CC BY-NC 4.0; treated as non-commercial."),
    Source(
        name="fineweb_edu", repo="karpathy/fineweb-edu-100b-shuffle", config="default",
        split="train", kind="base", fmt="text", files=("shard_*.parquet",), columns=("text",),
        adapter=text_column, license="odc-by", nonpermissive=False, weight=0.10,
        revision="4c8f30d6756da75362432a4d5569e1b229263b71",
        verified_sha="4c8f30d6756da75362432a4d5569e1b229263b71",
        n_files=1823, approx_rows=97_230_848, approx_bytes=171_425_076_443, cap_bytes=4 * _GB,
        notes="Permissive alternative to climbmix (inherits its weight with permissive_only); "
              "raise cap_bytes to the climbmix cap in that mode."),
    Source(
        name="simplewiki", repo="wikimedia/wikipedia", config="20231101.simple", split="train",
        kind="base", fmt="text", files=("20231101.simple/train-*.parquet",),
        columns=("title", "text"), adapter=wiki_article,
        license="cc-by-sa-3.0 and gfdl (card)", nonpermissive=False, weight=0.05,
        revision=_WIKI_SHA, verified_sha=_WIKI_SHA, n_files=1, approx_rows=241_787,
        approx_bytes=156_885_218,
        notes="Simple English Wikipedia dump of 2023-11-01: title, blank line, article text."),
    # --------------------------------------------------------------------------- chat
    Source(
        name="smol_smoltalk", repo="HuggingFaceTB/smol-smoltalk", config="default", split="train",
        kind="chat", fmt="messages", files=("data/train-*.parquet",), columns=("messages",),
        adapter=chat_messages, license="apache-2.0", nonpermissive=False, weight=0.30,
        revision="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        verified_sha="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        n_files=4, approx_rows=460_341, approx_bytes=922_475_727,
        notes="SmolTalk subset for models below 1B parameters."),
    _smoltalk("smoltalk_magpie_ultra", "smol-magpie-ultra", 0.20, 6, 409_537, 1_396_096_926,
              "Multi-turn, synthetic: generated by a third-party model whose licence sets "
              "conditions on the use of its outputs (see the card)."),
    _smoltalk("smoltalk_constraints", "smol-constraints", 0.05, 1, 34_424, 16_862_908,
              "Responses that follow explicit constraints; decontaminated against IFEval."),
    _smoltalk("smoltalk_rewrite", "smol-rewrite", 0.03, 1, 53_342, 38_927_074,
              "System prompt + rewriting task."),
    _smoltalk("smoltalk_summarize", "smol-summarize", 0.03, 1, 96_356, 119_245_090,
              "System prompt + email / news summarisation."),
    _smoltalk("smoltalk_everyday", "everyday-conversations", 0.02, 1, 2_260, 946_449,
              "Upstream HuggingFaceTB/everyday-conversations-llama3.1-2k (apache-2.0); 13% of "
              "the dialogues end with a user turn, which clean_messages drops."),
    _smoltalk("smoltalk_systemchats", "systemchats-30k", 0.02, 1, 34_133, 47_871_128,
              "Upstream cognitivecomputations/SystemChat-2.0 (apache-2.0); system prompts."),
    Source(
        name="tulu3_sft", repo="allenai/tulu-3-sft-mixture", config="default", split="train",
        kind="chat", fmt="messages", files=("data/train-*.parquet",),
        columns=("messages", "source"), adapter=tulu_messages,
        license="odc-by-1.0 (subsets carry their own licences; the CC BY-NC No Robots subset "
                "is excluded)",
        nonpermissive=False, weight=0.15,
        revision="b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        verified_sha="b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        n_files=6, approx_rows=939_343, approx_bytes=1_412_954_868,
        notes="Excludes TULU_EXCLUDED (No Robots: non-commercial; Aya: non-English; hard-coded "
              "identity prompts) = 109,740 rows. Contains outputs of third-party models."),
    Source(
        name="ultrachat", repo="HuggingFaceH4/ultrachat_200k", config="default", split="train_sft",
        kind="chat", fmt="messages", files=("data/train_sft-*.parquet",), columns=("messages",),
        adapter=chat_messages, license="mit", nonpermissive=False, weight=0.08,
        revision="8049631c405ae6576f93f445c6b8166f76f5505a",
        verified_sha="8049631c405ae6576f93f445c6b8166f76f5505a",
        n_files=3, approx_rows=207_865, approx_bytes=732_028_119,
        notes="Synthetic dialogues from a third-party model (see the card); truecased."),
    Source(
        name="synth", repo="PleIAs/SYNTH", config="default", split="train", kind="chat",
        fmt="messages", files=("synth_*.parquet",), columns=_SYNTH_COLUMNS,
        adapter=synth_closed_book, license=_SYNTH_LICENSE, nonpermissive=False, weight=0.12,
        revision=_SYNTH_SHA, verified_sha=_SYNTH_SHA, n_files=500, approx_rows=79_648_272,
        approx_bytes=236_335_243_511, cap_bytes=2 * _GB,
        notes="English rows of SYNTH_CHAT_EXERCISES, query -> synthetic_answer (the reasoning "
              "draft is dropped); rows from the self-knowledge seeds (~0.5%, first-person identity "
              "of the authors' own models) are dropped. Each ~473 MB file is ONE row group of "
              "~795 MB, read in streaming batches. Synthetic (third-party models, see the card)."),
    # ---------------------------------------------------------------------------- rag
    Source(
        name="synth_rag", repo="PleIAs/SYNTH", config="default", split="train", kind="rag",
        fmt="messages", files=("synth_*.parquet",), columns=_SYNTH_COLUMNS,
        adapter=synth_open_book, license=_SYNTH_LICENSE, nonpermissive=False, weight=0.30,
        revision=_SYNTH_SHA, verified_sha=_SYNTH_SHA, n_files=500, approx_rows=79_648_272,
        approx_bytes=236_335_243_511, cap_bytes=2 * _GB,
        notes="Same files as synth: the other half of the memorization rows with the seed "
              "section as passage, plus the 'rag' exercise rows (cited <source_k> blocks)."),
    Source(
        name="glaive_rag", repo="glaiveai/RAG-v1", config="default", split="train", kind="rag",
        fmt="messages", files=("default/train/*.parquet",),
        columns=("question", "documents", "answer"), adapter=glaive_rag, license="apache-2.0",
        nonpermissive=False, weight=0.15, revision=_CONVERT,
        verified_sha="03eb928590e5c957ecf36d420c4ee8043f718121",
        n_files=1, approx_rows=51_354, approx_bytes=133_532_368,
        notes="The main branch holds one 270 MB JSON file; the datasets-server parquet "
              "conversion (refs/convert/parquet) is used instead."),
    Source(
        name="rag_instruct", repo="FreedomIntelligence/RAG-Instruct", config="default",
        split="train", kind="rag", fmt="messages", files=("default/train/*.parquet",),
        columns=("question", "answer", "documents"), adapter=rag_instruct, license="apache-2.0",
        nonpermissive=False, weight=0.15, revision=_CONVERT,
        verified_sha="05323227f08cd6f8c4fcc2d4c1e0a2e1c4a0fca5",
        n_files=1, approx_rows=40_541, approx_bytes=177_556_026,
        notes="Main branch is JSON; parquet conversion used. Synthesised by a third-party "
              "model from Wikipedia; 10 untitled documents per question."),
    Source(
        name="neural_bridge_rag", repo="neural-bridge/rag-dataset-12000", config="default",
        split="train", kind="rag", fmt="messages", files=("data/train-*.parquet",),
        columns=("context", "question", "answer"), adapter=neural_bridge_rag,
        license="apache-2.0 (the card also asks users to abide by the terms of use of the web "
                "corpus it was drawn from)",
        nonpermissive=False, weight=0.07,
        revision="586c0d0ddeb022fccd909c7b415cc2ca8660baa4",
        verified_sha="586c0d0ddeb022fccd909c7b415cc2ca8660baa4",
        n_files=1, approx_rows=9_600, approx_bytes=23_101_821,
        notes="Web-page contexts (up to ~6k characters)."),
    Source(
        name="squad_v2", repo="rajpurkar/squad_v2", config="squad_v2", split="train", kind="rag",
        fmt="messages", files=("squad_v2/train-*.parquet",),
        columns=("id", "title", "context", "question", "answers"), adapter=squad_v2,
        license="cc-by-sa-4.0", nonpermissive=False, weight=0.15,
        revision="3ffb306f725f7d2ce8394bc1873b24868140c412",
        verified_sha="3ffb306f725f7d2ce8394bc1873b24868140c412",
        n_files=1, approx_rows=130_319, approx_bytes=16_369_982,
        notes="About a third of the train questions are unanswerable -> SQUAD_UNANSWERABLE."),
    Source(
        name="hotpot_qa", repo="hotpotqa/hotpot_qa", config="distractor", split="train",
        kind="rag", fmt="messages", files=("distractor/train-*.parquet",),
        columns=("question", "answer", "supporting_facts", "context"), adapter=hotpot_distractor,
        license="cc-by-sa-4.0", nonpermissive=False, weight=0.10,
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
        verified_sha="1908d6afbbead072334abe2965f91bd2709910ab",
        n_files=2, approx_rows=90_447, approx_bytes=331_786_656,
        notes="Distractor setting: 10 paragraphs per question, 2 supporting."),
    Source(
        name="dolly", repo="databricks/databricks-dolly-15k", config="default", split="train",
        kind="rag", fmt="messages", files=("default/train/*.parquet",),
        columns=("instruction", "context", "response", "category"), adapter=dolly,
        license="cc-by-sa-3.0", nonpermissive=False, weight=0.08, revision=_CONVERT,
        verified_sha="721815b9fa637f55fa82de54f99195bbec3cd0ce",
        n_files=1, approx_rows=15_011, approx_bytes=7_747_823,
        notes="Main branch is one JSONL file; parquet conversion used. About a third of the "
              "rows (closed_qa, information_extraction, summarization) have a context passage."),
    # --------------------------------------------------------------------------- math
    Source(
        name="gsm8k", repo="openai/gsm8k", config="main", split="train", kind="math",
        fmt="messages", files=("main/train-*.parquet",), columns=("question", "answer"),
        adapter=gsm8k, license="mit", nonpermissive=False, weight=0.35,
        revision="740312add88f781978c0658806c59bc2815b9866",
        verified_sha="740312add88f781978c0658806c59bc2815b9866",
        n_files=1, approx_rows=7_473, approx_bytes=2_306_545,
        notes="Each <<expr=result>> annotation becomes one calculator part of the assistant "
              "content (gsm8k_calc_text, calc_parts); results as annotated, trailing decimal "
              "zeros trimmed, not re-evaluated."),
    Source(
        name="tinygsm", repo="TinyGSM/TinyGSM", config="default", split="train", kind="math",
        fmt="messages", files=("data/train-*.parquet",), columns=("question", "code"),
        adapter=tinygsm, license="mit", nonpermissive=False, weight=0.0,
        revision="5049a7840fa546985eaae4c698d5daf6fdf7b47d",
        verified_sha="5049a7840fa546985eaae4c698d5daf6fdf7b47d",
        n_files=17, approx_rows=11_846_109, approx_bytes=3_750_847_774, cap_bytes=1 * _GB,
        notes="Python programs (def simple_math_problem() -> int); weight 0 until a "
              "program-to-calculator-span converter exists, because the model has a "
              "calculator, not an interpreter."),
    Source(
        name="orca_math", repo="microsoft/orca-math-word-problems-200k", config="default",
        split="train", kind="math", fmt="messages", files=("data/train-*.parquet",),
        columns=("question", "answer"), adapter=orca_math, license="mit", nonpermissive=False,
        weight=0.65, revision="29255d1770cc4eac66e5e7fa378cba542c026350",
        verified_sha="29255d1770cc4eac66e5e7fa378cba542c026350",
        n_files=1, approx_rows=200_035, approx_bytes=84_248_748,
        notes="Answers generated by a third-party model (see the card); no calculator "
              "annotations."),
    # -------------------------------------------------------------------------- prefs
    Source(
        name="ultrafeedback", repo="HuggingFaceH4/ultrafeedback_binarized", config="default",
        split="train_prefs", kind="prefs", fmt="pair", files=("data/train_prefs-*.parquet",),
        columns=("chosen", "rejected", "score_chosen", "score_rejected"),
        adapter=ultrafeedback_pair, license="mit", nonpermissive=False, weight=1.0,
        revision="3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
        verified_sha="3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
        n_files=1, approx_rows=61_135, approx_bytes=225_891_836,
        notes="Ties (score_chosen == score_rejected) are dropped. Completions come from open "
              "and proprietary models, scored by a model (see the card)."),
    # ------------------------------------------------------------------------ library
    Source(
        name="library_simplewiki", repo="wikimedia/wikipedia", config="20231101.simple",
        split="train", kind="library", fmt="text", files=("20231101.simple/train-*.parquet",),
        columns=("title", "text"), adapter=wiki_article, license="cc-by-sa-3.0 and gfdl (card)",
        nonpermissive=False, weight=0.0, revision=_WIKI_SHA, verified_sha=_WIKI_SHA, n_files=1,
        approx_rows=241_787, approx_bytes=156_885_218,
        notes="Raw text for the library build (not tokenised into training shards)."),
    Source(
        name="library_enwiki", repo="wikimedia/wikipedia", config="20231101.en", split="train",
        kind="library", fmt="text", files=("20231101.en/train-*.parquet",),
        columns=("title", "text"), adapter=wiki_article, license="cc-by-sa-3.0 and gfdl (card)",
        nonpermissive=False, weight=0.0, revision=_WIKI_SHA, verified_sha=_WIKI_SHA, n_files=41,
        approx_rows=6_407_814, approx_bytes=11_630_929_031,
        notes="English Wikipedia 2023-11-01 for lead paragraphs of the library."),
)

REGISTRY: dict[str, Source] = {s.name: s for s in SOURCES}
assert len(REGISTRY) == len(SOURCES), "duplicate source names"


def get(name: str | Source) -> Source:
    """The registry entry called ``name`` (a ``Source`` is returned unchanged)."""
    if isinstance(name, Source):
        return name
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown data source {name!r}; known: {sorted(REGISTRY)}") from None


def by_kind(kind: str) -> list[Source]:
    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}")
    return [s for s in SOURCES if s.kind == kind]


def mixture_weights(kind: str, permissive_only: bool = False) -> dict[str, float]:
    """Normalised default weights of ``kind`` (entries with weight 0 are omitted).

    With ``permissive_only`` every ``nonpermissive`` entry is removed and its weight goes to its
    ``fallback`` (which must be permissive and of the same kind) before normalising.
    """
    weights: dict[str, float] = {}
    for s in by_kind(kind):
        if permissive_only and s.nonpermissive:
            if s.fallback is not None:
                weights[s.fallback] = weights.get(s.fallback, 0.0) + s.weight
            continue
        weights[s.name] = weights.get(s.name, 0.0) + s.weight
    weights = {k: v for k, v in weights.items() if v > 0}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else {}


def download_cap(source: str | Source, permissive_only: bool = False) -> int | None:
    """Default download cap of an entry in bytes (``None`` = everything).

    With ``permissive_only`` an entry that is the ``fallback`` of excluded entries also takes
    their caps (``fineweb_edu`` then gets the ClimbMix cap on top of its own). Asking for the cap
    of an excluded entry in that mode is an error: it should not be downloaded at all.
    """
    src = get(source)
    if permissive_only and src.nonpermissive:
        raise ValueError(f"{src.name} is excluded with permissive_only")
    caps = [src.cap_bytes]
    if permissive_only:
        caps += [s.cap_bytes for s in SOURCES if s.nonpermissive and s.fallback == src.name]
    return None if any(c is None for c in caps) else sum(caps)


def adapt(source: str | Source, record: dict[str, Any]) -> Item:
    """``source.adapter(record)`` with the output type checked against ``source.fmt``."""
    src = get(source)
    item = src.adapter(record)
    if item is None:
        return None
    expected = {"text": str, "messages": list, "pair": Pair}[src.fmt]
    if not isinstance(item, expected):
        raise TypeError(f"{src.name}: adapter returned {type(item).__name__}, expected {src.fmt}")
    return item
