"""Training samples: token ids, roles and per-token loss weights, and the sample-level
transforms applied before packing (``docs/SPEC.md`` sections 1 and 3).

Sample
------
A :class:`Sample` is one segment -- a document or a conversation -- ready to be packed:

========  ===============  ==================================================================
field     dtype, shape     meaning
========  ===============  ==================================================================
``ids``   int32 ``[n]``    token ids; ``ids[0] == BOS`` and no other ``BOS``
``roles`` int8 ``[n]``     role code of every token (``bes.data.chat_format``: 0 raw text,
                           1 system, 2 user, 3 assistant, 4 library)
``w``     float32 ``[n]``  ``w[i]`` = loss weight of *predicting* ``ids[i]``; ``w[0] = 0``
========  ===============  ==================================================================

plus labels (``source``: registry name or library-task name, ``kind``: mixture sub-kind
``base | chat | rag | math | library``, ``tags``: variants and transforms applied), the
:class:`WeightRule` that produced ``w``, and ``exclude_pids`` / ``exclude_titles``: library
entries the sample was built from, which no later retrieval may return.

A loader that writes a sample at row offset ``o`` (``tokens[o:o + n] = ids``) sets the
target-aligned batch weight ``w_batch[o + i - 1] = w[i]`` for ``i >= 1`` (input position
``o + i - 1`` predicts ``ids[i]``). ``w[0]`` is the weight of predicting the sample's ``BOS``
from the end of the previous segment: always 0.

Loss weights (:func:`loss_weights`, one pure function of ``ids``, ``roles`` and the rule)
------------------------------------------------------------------------------------------
=====================================================  =======================================
target token                                           weight
=====================================================  =======================================
``BOS``, ``PAD``                                       0
raw text (role 0)                                      1 (in every stage)
assistant turn: content and its ``END``                1
``ASSISTANT`` right after the ``END`` of a user turn   0 (the chat grammar allows nothing else)
user and system turns (role token, content, ``END``)   1; 0 in the assistant-only stage
library turns (role token, passage, ``END``)           ``passage_w`` (``DataCfg.passage_loss_w``);
                                                       0 in the assistant-only stage
calculator ``CALC``, the expression, ``EQ``            1 (assistant content)
calculator result tokens and ``CALC_END``              0 (the host forces them at inference)
=====================================================  =======================================

Role tokens and ``END`` carry the role of their turn (``chat_format``), so they follow the
weight of their turn. A calculator span that is not closed has its tokens after ``EQ`` (up to
the next special token) at weight 0, as ``bes.data.prefs.forced_mask``.

Transforms
----------
- :func:`tokens_to_messages`: exact inverse of ``chat_format.render_conversation``.
- :func:`fit`: the longest admissible prefix within ``max_len`` tokens. Conversations are cut
  only after the ``END`` of an assistant turn (never inside a turn, never after a user turn);
  the cut-off turns are dropped. Documents are cut anywhere; the rest continues as a new
  sample that starts with a fresh ``BOS``.
- :func:`window`: the conversation without its oldest exchanges (``BOS``, the system turn, then
  the exchanges from a given one on), the truncation the responder applies at inference when
  the context overflows (``bes.decode.respond.fit_prompt``). An *exchange* is ``[library] user
  assistant``; a window always starts at one (:func:`exchange_starts`), never at an assistant
  turn or at a library turn without its user turn.
- :func:`fit_windows` / :func:`fit_draw`: a conversation longer than ``max_len`` as consecutive
  windows (the prefix of :func:`fit`, then a window from the first exchange after it, ...);
  a training draw takes the prefix or, with probability ``later``, one of the later windows,
  so that the later turns of long conversations are trained too.
- :func:`apply_hindsight`: tokens -> messages -> ``bes.data.hindsight.relabel`` -> tokens.

Apply content transforms (hindsight, retrieval) before :func:`fit`, or pass ``max_len`` to them.

Determinism
-----------
Nothing here keeps state between calls. Every random choice comes from a ``numpy.random.
Generator`` passed by the caller; :func:`rng_for` builds one from a hash of explicit keys and
:func:`permute_index` gives a keyed bijection of ``range(n)`` (epoch orders without storing a
permutation).
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, N_SPECIAL, PAD,
                        ROLE_TOKENS, SYSTEM, USER, VOCAB_SIZE, DataCfg)
from bes.data import chat_format as cf

FIRST_SPECIAL = VOCAB_SIZE - N_SPECIAL
ROLE_RAW, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_LIBRARY = (
    cf.ROLE_RAW, cf.ROLE_SYSTEM, cf.ROLE_USER, cf.ROLE_ASSISTANT, cf.ROLE_LIBRARY)
#: Number of role codes (``REG_FIELDS["role"]``; code 5 is reserved and gets weight 0).
N_ROLE_CODES = 6
ROLE_NAME_OF_TOKEN: dict[int, str] = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant",
                                      LIBRARY: "library"}
CALC_TOKENS = (CALC, EQ, CALC_END)
KINDS = ("base", "chat", "rag", "math", "library")


# ------------------------------------------------------------------------- weight rule
@dataclass(frozen=True)
class WeightRule:
    """Stage-dependent part of the loss-weight rule (module docstring).

    assistant_only: the stage trains on assistant turns only (user, system and library turns
    get weight 0; raw text keeps weight 1). passage_w: weight of library-turn tokens otherwise.
    """

    assistant_only: bool = False
    passage_w: float = DataCfg().passage_loss_w

    @classmethod
    def at(cls, stage: Any, data: DataCfg = DataCfg()) -> "WeightRule":
        """The rule of a ``bes.train.schedule.Stage`` (anything with ``assistant_only``)."""
        return cls(assistant_only=bool(stage.assistant_only), passage_w=float(data.passage_loss_w))

    def role_table(self) -> np.ndarray:
        """``float32[N_ROLE_CODES]``: weight of a token by the role code of its turn."""
        talk = 0.0 if self.assistant_only else 1.0
        lib = 0.0 if self.assistant_only else float(self.passage_w)
        return np.array([1.0, talk, talk, 1.0, lib, 0.0], dtype=np.float32)


def loss_weights(ids: Sequence[int] | np.ndarray, roles: Sequence[int] | np.ndarray,
                 rule: WeightRule = WeightRule()) -> np.ndarray:
    """``float32[n]``: loss weight of predicting each of ``ids int[n]`` (rule in the module
    docstring); ``roles int[n]`` are the role codes of ``chat_format``."""
    a = np.asarray(ids, dtype=np.int64)
    r = np.asarray(roles, dtype=np.int64)
    if a.shape != r.shape or a.ndim != 1:
        raise ValueError(f"ids {a.shape} and roles {r.shape} must be 1-d of the same length")
    if a.size and (int(r.min()) < 0 or int(r.max()) >= N_ROLE_CODES):
        raise ValueError(f"role codes must lie in [0, {N_ROLE_CODES})")
    w = rule.role_table()[r]
    w[(a == BOS) | (a == PAD)] = 0.0
    if a.size > 1:
        after_user_end = np.zeros(a.size, dtype=bool)
        after_user_end[1:] = (a[:-1] == END) & (r[:-1] == ROLE_USER)
        w[(a == ASSISTANT) & after_user_end] = 0.0
    if a.size and bool(np.isin(a, CALC_TOKENS).any()):
        for span in cf.find_calc_spans(a):
            if span.eq >= 0:
                w[span.eq + 1:(span.end + 1 if span.end >= 0 else span.stop)] = 0.0
    return w


# ------------------------------------------------------------------------------ sample
@dataclass(frozen=True, eq=False)
class Sample:
    """One segment with its loss weights (shapes and meaning in the module docstring).

    Build samples with :func:`make_sample`, :func:`document_sample`, :func:`conversation_sample`
    or :func:`from_tokens`, which validate the arrays and compute ``w``; the arrays are
    read-only.
    """

    ids: np.ndarray
    roles: np.ndarray
    w: np.ndarray
    rule: WeightRule = WeightRule()
    source: str = ""
    kind: str = ""
    tags: tuple[str, ...] = ()
    exclude_pids: tuple[int, ...] = ()
    exclude_titles: tuple[int, ...] = ()

    def __len__(self) -> int:
        return int(self.ids.size)

    @property
    def is_conversation(self) -> bool:
        """True when the segment has turns (any role code other than raw text)."""
        return bool(self.roles.size) and bool((self.roles != ROLE_RAW).any())

    @property
    def n_targets(self) -> float:
        """Sum of the loss weights."""
        return float(self.w.sum())

    def key(self) -> bytes:
        """16-byte digest of ``ids``, ``roles`` and ``w`` (determinism checks)."""
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.ids, self.roles, self.w):
            h.update(np.int64(arr.size).tobytes())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.digest()

    def with_tokens(self, ids: Sequence[int] | np.ndarray, roles: Sequence[int] | np.ndarray,
                    *, add_tags: Sequence[str] = (), **changes: Any) -> "Sample":
        """A copy with new ``ids`` / ``roles`` (``w`` recomputed with the same rule) and
        ``add_tags`` appended; ``changes`` override other fields (``rule``, ``source``, ...)."""
        fields = dict(rule=self.rule, source=self.source, kind=self.kind,
                      tags=self.tags + tuple(add_tags), exclude_pids=self.exclude_pids,
                      exclude_titles=self.exclude_titles)
        fields.update(changes)
        return make_sample(ids, roles, **fields)

    def with_rule(self, rule: WeightRule) -> "Sample":
        """The same tokens weighted by ``rule``."""
        return replace(self, rule=rule, w=_readonly(loss_weights(self.ids, self.roles, rule),
                                                     np.float32))

    def prefix(self, stop: int) -> "Sample":
        """``ids[:stop]`` (and roles), weights recomputed."""
        return self.with_tokens(self.ids[:stop], self.roles[:stop])


def _readonly(a: Any, dtype: Any) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def make_sample(ids: Sequence[int] | np.ndarray, roles: Sequence[int] | np.ndarray,
                rule: WeightRule = WeightRule(), *, source: str = "", kind: str = "",
                tags: Sequence[str] = (), exclude_pids: Sequence[int] = (),
                exclude_titles: Sequence[int] = ()) -> Sample:
    """A validated :class:`Sample` with ``w = loss_weights(ids, roles, rule)``.

    Raises ``ValueError`` unless ``ids`` is a non-empty 1-d integer sequence in
    ``[0, VOCAB_SIZE)`` that starts with ``BOS`` and has no other ``BOS``, and ``roles`` has the
    same length with codes in ``[0, N_ROLE_CODES)``.
    """
    a = np.asarray(ids)
    r = np.asarray(roles)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("a sample is a non-empty 1-d sequence of ids")
    if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(r.dtype, np.integer):
        raise ValueError("ids and roles must be integers")
    if int(a[0]) != BOS or bool((a[1:] == BOS).any()):
        raise ValueError("a sample starts with BOS and has no other BOS")
    if int(a.min()) < 0 or int(a.max()) >= VOCAB_SIZE:
        raise ValueError(f"token id outside [0, {VOCAB_SIZE})")
    if r.shape != a.shape:
        raise ValueError(f"roles shape {r.shape} != ids shape {a.shape}")
    if int(r.min()) < 0 or int(r.max()) >= N_ROLE_CODES:
        raise ValueError(f"role codes must lie in [0, {N_ROLE_CODES})")
    ids32, roles8 = _readonly(a, np.int32), _readonly(r, np.int8)
    w = _readonly(loss_weights(ids32, roles8, rule), np.float32)
    return Sample(ids=ids32, roles=roles8, w=w, rule=rule, source=source, kind=kind,
                  tags=tuple(tags), exclude_pids=tuple(int(p) for p in exclude_pids),
                  exclude_titles=tuple(int(t) for t in exclude_titles))


def from_tokens(ids: Sequence[int] | np.ndarray, roles: Sequence[int] | np.ndarray | None = None,
                rule: WeightRule = WeightRule(), **labels: Any) -> Sample:
    """A sample from a shard segment: ``roles`` from the ``.role`` file, or ``None`` for a raw
    document (all codes 0; the ids must then contain no role token)."""
    a = np.asarray(ids)
    if roles is None:
        if a.size and bool(np.isin(a, ROLE_TOKENS).any()):
            raise ValueError("a segment with role tokens needs its roles")
        roles = np.zeros(a.shape, dtype=np.int8)
    return make_sample(a, roles, rule, **labels)


def document_sample(text: str, tok, rule: WeightRule = WeightRule(), **labels: Any) -> Sample:
    """``BOS text`` as a sample (``chat_format.render_document``)."""
    ids, roles = cf.render_document(text, tok)
    return make_sample(ids, roles, rule, **labels)


def conversation_sample(messages: Sequence[Mapping[str, Any]], tok, rule: WeightRule = WeightRule(),
                        **labels: Any) -> Sample:
    """A rendered conversation as a sample (``chat_format.render_conversation``)."""
    ids, roles = cf.render_conversation(messages, tok)
    return make_sample(ids, roles, rule, **labels)


# --------------------------------------------------------------- tokens -> messages
def _decode_exact(run: Sequence[int], tok) -> str | None:
    """The string whose ordinary encoding is exactly ``run``, or ``None`` when there is none
    (invalid UTF-8, or a run that is not the encoding of its own text)."""
    try:
        text = tok.decode_bytes(run).decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if tok.encode(text) == list(run) else None


def _split_exact(run: Sequence[int], tok) -> list[str] | None:
    """Strings whose encodings concatenate to ``run`` (longest pieces first, backtracking);
    ``None`` when no such split exists. Only reached for runs that rendered several adjacent
    text parts, which the data adapters never produce."""
    run = list(run)
    n = len(run)
    dead: set[int] = set()

    def solve(i: int) -> list[str] | None:
        if i == n:
            return []
        if i in dead:
            return None
        for j in range(n, i, -1):
            text = _decode_exact(run[i:j], tok)
            if text is None or not text:
                continue
            rest = solve(j)
            if rest is not None:
                return [text, *rest]
        dead.add(i)
        return None

    return solve(0)


def _text_parts(run: Sequence[int], tok) -> list[str]:
    if not run:
        return []
    text = _decode_exact(run, tok)
    if text is not None:
        return [text]
    pieces = _split_exact(run, tok)
    if pieces is None:
        raise ValueError("assistant text that no rendering of content parts produces")
    return pieces


def _assistant_content(content: list[int], tok) -> str | list[dict[str, str]]:
    """Assistant content of a turn's ids: a string, or parts with calculator spans."""
    specials = [i for i, t in enumerate(content) if t >= FIRST_SPECIAL]
    if not specials:
        parts = _text_parts(content, tok)
        if len(parts) <= 1:
            return parts[0] if parts else ""
        return [{"type": "text", "text": p} for p in parts]
    # The specials must form complete spans: (CALC, EQ, CALC_END), (CALC, EQ, CALC_END), ...
    if len(specials) % 3 or any(content[s] != CALC_TOKENS[k % 3] for k, s in enumerate(specials)):
        raise ValueError("assistant content with a special token outside a complete calculator span")
    out: list[dict[str, str]] = []
    pos = 0
    for k in range(0, len(specials), 3):
        c, e, z = specials[k], specials[k + 1], specials[k + 2]
        out += [{"type": "text", "text": p} for p in _text_parts(content[pos:c], tok)]
        expr, result = _decode_exact(content[c + 1:e], tok), _decode_exact(content[e + 1:z], tok)
        if expr is None or result is None:
            raise ValueError("calculator expression or result that no string encodes to")
        out.append({"type": "calc", "expr": expr, "result": result})
        pos = z + 1
    out += [{"type": "text", "text": p} for p in _text_parts(content[pos:], tok)]
    return out


def stock_refusal(ids: Sequence[int] | np.ndarray, tok, head: int = 28) -> bool:
    """Some assistant turn of the conversation ``ids`` opens with a stock refusal
    (``bes.eval.exact_score.canned_refusal`` on the first ``head`` tokens of the turn). The
    training stream leaves such conversations out (``bes.data.loader.DROP_STOCK_REFUSALS``): a
    model of this size cannot tell which requests they belong to and gives the same sentence to
    harmless ones."""
    from bes.eval.exact_score import canned_refusal

    arr = np.asarray(ids)
    for pos in np.flatnonzero(arr == ASSISTANT):
        turn = arr[pos + 1: pos + 1 + head]
        special = np.flatnonzero(turn >= FIRST_SPECIAL)
        if special.size:
            turn = turn[:special[0]]
        if turn.size and canned_refusal(tok.decode(turn)):
            return True
    return False


def tokens_to_messages(ids: Sequence[int] | np.ndarray, tok,
                       add_generation_prompt: bool = False) -> list[dict[str, Any]]:
    """Messages ``m`` with ``chat_format.render_conversation(m, tok, add_generation_prompt)``
    equal to ``ids`` (exact inverse of the renderer).

    ``ids``: ``BOS`` then turns ``ROLE content END`` (and a final bare ``ASSISTANT`` when
    ``add_generation_prompt``). Content is decoded exactly: user, system and library content
    is the string whose encoding is the turn's ids; assistant content is a string, or a list of
    ``{"type": "text"}`` / ``{"type": "calc", "expr", "result"}`` parts when it holds calculator
    spans. Rendering ``m`` again gives ``ids`` token for token, and ``tokens_to_messages`` of a
    rendering returns the messages with empty text parts dropped and a parts list without
    calculator spans given as one string. Raises ``ValueError`` on ids that no conversation
    renders to (no ``BOS``, a missing ``END``, stray special tokens, a broken turn order, ...).
    """
    arr = [int(t) for t in np.asarray(ids, dtype=np.int64).tolist()]
    if not arr or arr[0] != BOS:
        raise ValueError("a conversation starts with BOS")
    stop = len(arr)
    if add_generation_prompt:
        if stop < 2 or arr[-1] != ASSISTANT:
            raise ValueError("add_generation_prompt needs a final bare ASSISTANT")
        stop -= 1
    messages: list[dict[str, Any]] = []
    pos = 1
    while pos < stop:
        role = ROLE_NAME_OF_TOKEN.get(arr[pos])
        if role is None:
            raise ValueError(f"position {pos}: expected a role token, got {arr[pos]}")
        try:
            end = arr.index(END, pos + 1, stop)
        except ValueError:
            raise ValueError(f"position {pos}: {role} turn without END") from None
        content = arr[pos + 1:end]
        if role == "assistant":
            value: Any = _assistant_content(content, tok)
        else:
            if any(t >= FIRST_SPECIAL for t in content):
                raise ValueError(f"position {pos}: special token inside a {role} turn")
            value = _decode_exact(content, tok)
            if value is None:
                raise ValueError(f"position {pos}: {role} content that no string encodes to")
        messages.append({"role": role, "content": value})
        pos = end + 1
    cf.check_structure([m["role"] for m in messages], add_generation_prompt)
    return messages


def canonical_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What :func:`tokens_to_messages` returns for a rendering of ``messages`` whose assistant
    parts never place two text parts side by side: empty text parts dropped, a parts list
    without calculator spans as one string (``""`` when nothing is left)."""
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, (list, tuple)):
            parts = [dict(p) for p in content if not (p.get("type") == "text" and p.get("text") == "")]
            if not any(p.get("type") == "calc" for p in parts):
                content = "".join(p["text"] for p in parts)
            else:
                content = parts
        out.append({"role": m["role"], "content": content})
    return out


# -------------------------------------------------------------------------------- fit
def turn_spans(ids: Sequence[int] | np.ndarray) -> list[tuple[int, int, int]]:
    """``[(role_token, start, stop)]`` of every turn: ``ids[start]`` is the role token and
    ``ids[stop - 1]`` its ``END`` (``stop = len(ids)`` for a turn left open)."""
    a = np.asarray(ids, dtype=np.int64)
    starts = np.flatnonzero(np.isin(a, ROLE_TOKENS)).tolist()
    out = []
    for k, s in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else a.size
        ends = np.flatnonzero(a[s + 1:nxt] == END)
        out.append((int(a[s]), s, s + 2 + int(ends[0]) if ends.size else int(a.size)))
    return out


def conversation_cuts(ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """``int64[k]``: admissible lengths of a conversation prefix, i.e. the positions just after
    every ``END`` that closes an assistant turn (``chat_format.assistant_turn_ends``)."""
    return cf.assistant_turn_ends(ids)


def fit(sample: Sample, max_len: int) -> tuple[Sample | None, Sample | None]:
    """``(head, rest)``: the longest admissible prefix of at most ``max_len`` tokens.

    Conversation: ``head`` ends at the ``END`` of an assistant turn (the whole sample when it
    fits and ends that way; the turns after the cut are dropped, ``rest = None``); ``head`` is
    ``None`` when not even the first exchange fits. Document: ``head = sample`` when it fits;
    otherwise the first ``max_len`` tokens, and ``rest`` = ``BOS`` + the remaining tokens (a
    new sample, weights recomputed); with ``max_len < 2`` nothing fits (``(None, sample)``).
    """
    n = len(sample)
    if sample.is_conversation:
        cuts = conversation_cuts(sample.ids)
        cuts = cuts[cuts <= max_len]
        if cuts.size == 0:
            return None, None
        cut = int(cuts[-1])
        return (sample if cut == n else sample.prefix(cut)), None
    if n <= max_len:
        return sample, None
    if max_len < 2:
        return None, sample
    rest_ids = np.concatenate([[BOS], sample.ids[max_len:]])
    rest_roles = np.concatenate([[ROLE_RAW], sample.roles[max_len:]])
    return sample.prefix(max_len), sample.with_tokens(rest_ids, rest_roles)


def exchange_starts(ids: Sequence[int] | np.ndarray) -> list[int]:
    """Token positions at which the exchanges (``[library] user assistant``) of a conversation
    start, in order: every user turn, or the library turn right before it."""
    spans = turn_spans(ids)
    out = []
    for k, (role, start, _) in enumerate(spans):
        if role == USER:
            out.append(spans[k - 1][1] if k > 0 and spans[k - 1][0] == LIBRARY else start)
    return out


def head_stop(ids: Sequence[int] | np.ndarray) -> int:
    """End of what every window keeps: ``BOS`` and the system turn when the conversation opens
    with one."""
    spans = turn_spans(ids)
    return spans[0][2] if spans and spans[0][0] == SYSTEM else 1


def window(sample: Sample, start: int, max_len: int | None = None) -> Sample | None:
    """The conversation without the exchanges before token position ``start`` (one of
    :func:`exchange_starts`): ``BOS``, the system turn when there is one, then ``ids[start:]``
    (``bes.decode.respond.fit_prompt`` drops the oldest exchanges first and keeps the system
    turn); tags gain ``"window"``. The first exchange start gives the whole sample. With
    ``max_len`` the window is cut by :func:`fit` (its longest admissible prefix; ``None`` when
    not even its first exchange fits). Raises ``ValueError`` when ``start`` is not an exchange
    start."""
    if int(start) not in exchange_starts(sample.ids):
        raise ValueError(f"position {start} does not start an exchange")
    keep = head_stop(sample.ids)
    out = sample
    if start > keep:
        out = sample.with_tokens(np.concatenate([sample.ids[:keep], sample.ids[start:]]),
                                 np.concatenate([sample.roles[:keep], sample.roles[start:]]),
                                 add_tags=("window",))
    if max_len is not None:
        out, _ = fit(out, max_len)
    return out


def _pieces(sample: Sample, max_len: int) -> tuple[Sample | None, list[Sample]]:
    """``(prefix, later windows)`` of :func:`fit_windows` (``prefix`` is ``fit``'s head)."""
    head, _ = fit(sample, max_len)
    if head is sample:
        return head, []
    starts = exchange_starts(sample.ids)
    keep = head_stop(sample.ids)
    later: list[Sample] = []
    i, end = 0, (len(head) if head is not None else 0)
    if head is None:
        i = 1
    while i < len(starts) and starts[i] < end:
        i += 1
    while i < len(starts):
        w = window(sample, starts[i], max_len) if starts[i] > keep else None
        if w is None:
            i += 1
            continue
        later.append(w)
        end = starts[i] + len(w) - keep                 # end of the window in ``sample``
        while i < len(starts) and starts[i] < end:
            i += 1
    return head, later


def fit_windows(sample: Sample, max_len: int) -> list[Sample]:
    """A conversation as consecutive pieces of at most ``max_len`` tokens: ``[sample]`` when it
    fits (and ends with an assistant turn); otherwise its prefix (:func:`fit`; left out when not
    even the first exchange fits), then the :func:`window` that starts at the first exchange
    after the prefix, then the one after that window, and so on. An exchange that does not fit
    on its own (with the system turn) starts no window and is skipped; ``[]`` when nothing fits.
    Every piece is a conversation that ends with an assistant turn."""
    if not sample.is_conversation:
        raise ValueError("fit_windows takes a conversation")
    head, later = _pieces(sample, max_len)
    return ([head] if head is not None else []) + later


def fit_draw(sample: Sample, max_len: int, rng: np.random.Generator, later: float = 0.5) -> Sample | None:
    """What one training draw keeps of ``sample`` within ``max_len`` tokens (deterministic in
    ``rng``). A document: the head of :func:`fit`. A conversation that fits: itself. Otherwise
    one piece of :func:`fit_windows`: the prefix, or with probability ``later`` one of the later
    windows, drawn uniformly (a later window whenever the prefix does not exist, the prefix when
    there is no later window). ``None`` when nothing fits."""
    if not sample.is_conversation:
        return fit(sample, max_len)[0]
    head, rest = _pieces(sample, max_len)
    if not rest:
        return head
    if head is None or rng.random() < float(later):
        return rest[int(rng.integers(len(rest)))]
    return head


# ------------------------------------------------------------------------- hindsight
def apply_hindsight(sample: Sample, rng: np.random.Generator, tok,
                    max_len: int | None = None) -> Sample | None:
    """The conversation with a verified constraint appended to its last user message
    (``bes.data.hindsight.relabel`` on :func:`tokens_to_messages`); ``None`` when the sample is
    not a conversation ending with an assistant turn, when ``relabel`` declines, or when the
    result is longer than ``max_len``. Only the last user turn changes: its tokens are replaced
    by the encoding of the new message, every other token is kept. Tags gain ``"hindsight"``.
    Raises ``ValueError`` (from :func:`tokens_to_messages`) when the tokens are not a rendered
    conversation: a data bug, not a sample to skip.
    """
    from bes.data import hindsight

    if not sample.is_conversation:
        return None
    messages = tokens_to_messages(sample.ids, tok)
    new = hindsight.relabel(messages, rng, tok)
    if new is None:
        return None
    spans = [s for s in turn_spans(sample.ids) if s[0] == USER]
    _, start, stop = spans[-1]
    turn, turn_roles = cf.render_turn("user", new[-2]["content"], tok)
    ids = np.concatenate([sample.ids[:start], turn, sample.ids[stop:]])
    roles = np.concatenate([sample.roles[:start], turn_roles, sample.roles[stop:]])
    if max_len is not None and ids.size > max_len:
        return None
    return sample.with_tokens(ids, roles, add_tags=("hindsight",))


# ---------------------------------------------------------------------- determinism
def _key_bytes(key: Any) -> bytes:
    if isinstance(key, (bool, np.bool_)):
        return b"b1" if key else b"b0"
    if isinstance(key, (int, np.integer)):
        body = str(int(key)).encode()
        return b"i" + len(body).to_bytes(2, "little") + body
    if isinstance(key, (float, np.floating)):
        body = float(key).hex().encode()
        return b"f" + len(body).to_bytes(2, "little") + body
    if isinstance(key, str):
        body = key.encode("utf-8")
        return b"s" + len(body).to_bytes(4, "little") + body
    if isinstance(key, (bytes, bytearray)):
        return b"y" + len(key).to_bytes(4, "little") + bytes(key)
    if isinstance(key, (tuple, list)):
        return b"t" + len(key).to_bytes(4, "little") + b"".join(_key_bytes(k) for k in key)
    if key is None:
        return b"n"
    raise TypeError(f"unsupported key type {type(key).__name__}")


def key_digest(*keys: Any, size: int = 32) -> bytes:
    """BLAKE2b digest of explicit keys (ints, floats, strings, bytes, None, tuples); stable
    across processes and platforms."""
    h = hashlib.blake2b(digest_size=size, person=b"bes-samples")
    h.update(_key_bytes(tuple(keys)))
    return h.digest()


def rng_for(*keys: Any) -> np.random.Generator:
    """A ``numpy.random.Generator`` seeded by a hash of ``keys`` (counter-based: the same keys
    give the same stream in any process, independently of what was drawn before)."""
    words = np.frombuffer(key_digest(*keys), dtype="<u8")
    return np.random.default_rng([int(x) for x in words])


_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64 finaliser (a bijective 64-bit mixer)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def permute_index(i: int, n: int, *keys: Any) -> int:
    """Position of ``i`` under a keyed bijection of ``range(n)`` (a 4-round Feistel network on
    the next even power of two, with cycle walking). Distinct ``i`` give distinct results; the
    permutation depends only on ``n`` and ``keys``."""
    if not 0 <= i < n:
        raise IndexError(f"{i} outside range({n})")
    if n <= 1:
        return 0
    bits = max(2, (n - 1).bit_length())
    bits += bits & 1
    half = bits // 2
    mask = (1 << half) - 1
    seed = key_digest("permute_index", n, *keys)
    round_keys = [int.from_bytes(seed[8 * r:8 * r + 8], "little") for r in range(4)]
    x = int(i)
    while True:
        left, right = x >> half, x & mask
        for k in round_keys:
            left, right = right, left ^ (_mix64(right ^ k) & mask)
        x = (left << half) | right
        if x < n:
            return x


__all__ = ["Sample", "WeightRule", "loss_weights", "make_sample", "from_tokens", "document_sample",
           "conversation_sample", "tokens_to_messages", "canonical_messages", "turn_spans",
           "conversation_cuts", "fit", "exchange_starts", "head_stop", "window", "fit_windows",
           "fit_draw", "apply_hindsight", "rng_for", "key_digest", "permute_index", "KINDS"]
