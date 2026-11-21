"""Automatic pairwise judge: an instruction-tuned GGUF model on CPU, both orders, strict agreement.

For every blinded pair the judge sees the conversation twice, once with the replies in their
blinded order and once swapped. Each time it is asked for one letter, ``A``, ``B`` or ``C``
(tie / equally good / both poor), and the verdict is read from the next-token distribution
restricted to those three tokens (one forward pass, no generation). The final verdict is a side
only when both orders pick that same reply; any disagreement or a ``C`` is a tie. The averaged
probabilities over the two orders are kept as a soft verdict, and the per-order verdicts as
diagnostics of position bias.

The prompt (:data:`JUDGE_INSTRUCTIONS`) asks for correctness first, then following every
explicit instruction, then helpfulness, and says that length is not a merit beyond what the
request needs. Known biases of this kind of judge are listed in ``docs/EVAL_PROTOCOL.md``.

Judges are objects with a ``key`` and ``probs(messages) -> {"A": p, "B": p, "C": p,
"mass": p}``; :class:`GGUFJudge` is the real one, :class:`StandInJudge` the test double.
"""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from bes.eval.proxy.common import HARNESS_VERSION, append_jsonl, default_root, read_jsonl, safe_name

JUDGE_SYSTEM = "You are a careful, impartial judge of replies written by chat assistants."

JUDGE_INSTRUCTIONS = """Compare two assistant replies to the same user request and decide which one serves the user better.

Judge, in this order of importance:
1. Correctness: facts, arithmetic, code and reasoning must be right. A wrong final answer is a serious flaw, even if it is well written.
2. Following the request: every explicit instruction must be obeyed (length, number of sentences, items or words, format, required or forbidden words, tone).
3. Helpfulness: the reply addresses what was asked, clearly and completely, without padding.

Rules:
- Length is not a merit. Extra length only counts when the request needs it; prefer the shorter reply when both are equally correct and complete.
- Ignore the order in which the replies are shown and the labels A and B.
- Repetition, unfinished text, gibberish, or a reply that writes the user's next message are serious flaws.
- Answer C if the replies are about equally good or both are poor."""

QUESTION = "Which reply is better? Answer with a single letter: A, B or C."
OPTIONS = ("A", "B", "C")
MAX_REPLY_CHARS = 4000


def _clip(text: str, limit: int = MAX_REPLY_CHARS) -> tuple[str, bool]:
    if not text.strip():
        return "(empty reply)", False
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[... reply cut here; it continues for {len(text) - limit} more characters]", True


def render_messages(turns: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], bool]:
    """Judge chat messages for one ordering of a pair. ``turns = [{"user", "A", "B"}]``.
    Returns ``(messages, clipped)``; ``clipped`` is True when a reply was cut to
    :data:`MAX_REPLY_CHARS` characters."""
    clipped = False
    parts = [JUDGE_INSTRUCTIONS, ""]
    if len(turns) == 1:
        a, ca = _clip(turns[0]["A"])
        b, cb = _clip(turns[0]["B"])
        clipped = ca or cb
        parts += ["[User request]", turns[0]["user"], "[End of user request]", "",
                  "[Reply A]", a, "[End of reply A]", "", "[Reply B]", b, "[End of reply B]"]
    else:
        parts.append("The conversation has several turns. Judge the assistant's replies over the "
                     "whole conversation; the user's later messages are the same for both.")
        parts.append("")
        for side in ("A", "B"):
            parts.append(f"[Conversation with assistant {side}]")
            for t in turns:
                text, c = _clip(t[side])
                clipped = clipped or c
                parts += [f"User: {t['user']}", f"Assistant {side}: {text}"]
            parts += [f"[End of conversation with assistant {side}]", ""]
    parts += ["", QUESTION]
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "\n".join(parts)}], clipped


_SINGLE = re.compile(r"\[Reply A\]\n(.*)\n\[End of reply A\]\n\n\[Reply B\]\n(.*)\n\[End of reply B\]", re.DOTALL)


def parse_rendered(messages: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    """Inverse of :func:`render_messages` for the replies (used by stand-in judges): the text of
    reply A and reply B (all assistant turns joined for multi-turn conversations)."""
    text = messages[-1]["content"]
    m = _SINGLE.search(text)
    if m:
        return m.group(1), m.group(2)
    out = []
    for side in ("A", "B"):
        block = re.search(rf"\[Conversation with assistant {side}\]\n(.*?)\n\[End of conversation with assistant {side}\]",
                          text, re.DOTALL)
        body = block.group(1) if block else ""
        out.append("\n".join(re.findall(rf"^Assistant {side}: (.*?)(?=^User: |\Z)", body, re.DOTALL | re.MULTILINE)))
    return out[0], out[1]


class Judge(Protocol):
    key: str

    def probs(self, messages: Sequence[Mapping[str, str]]) -> dict[str, float]:
        ...


class GGUFJudge:
    """A GGUF instruct model used as a judge.

    ``assistant_prefix`` is appended after the chat template's generation prompt (for thinking
    models, e.g. ``"<think>\\n\\n</think>\\n\\n"`` to answer directly). The fixed instruction
    prefix of consecutive prompts is reused from the KV cache when the model is a plain
    transformer (not for recurrent or hybrid models)."""

    def __init__(self, path: str | Path, n_ctx: int = 8192, n_threads: int = 4,
                 chat_template: str | None = None, assistant_prefix: str = "",
                 key: str | None = None, verbose: bool = False) -> None:
        from bes.eval.proxy.run_gguf import GGUFBackend

        self.backend = GGUFBackend(path, n_ctx=n_ctx, n_threads=n_threads, chat_template=chat_template,
                                   verbose=verbose)
        self.assistant_prefix = assistant_prefix
        self.key = key or f"judge-{self.backend.key}"
        self.option_ids: dict[str, int] = {}
        for opt in OPTIONS:
            ids = self.backend.llm.tokenize(opt.encode("utf-8"), add_bos=False, special=False)
            if not ids:
                raise ValueError(f"cannot tokenize option {opt!r}")
            self.option_ids[opt] = ids[0]
        if len(set(self.option_ids.values())) != len(OPTIONS):
            raise ValueError("the options A, B, C do not map to distinct first tokens")
        llm = self.backend.llm
        self._reuse = not (getattr(llm, "_is_recurrent", False) or getattr(llm, "_is_hybrid", False))
        self._last: list[int] = []

    def probs(self, messages: Sequence[Mapping[str, str]]) -> dict[str, float]:
        text = self.backend.render(messages) + self.assistant_prefix
        tokens = self.backend.tokenize(text)
        if len(tokens) >= self.backend.n_ctx:
            raise ValueError(f"judge prompt of {len(tokens)} tokens exceeds n_ctx={self.backend.n_ctx}")
        llm = self.backend.llm
        common = 0
        if self._reuse:
            for a, b in zip(self._last, tokens):
                if a != b:
                    break
                common += 1
            common = min(common, len(tokens) - 1)
        if common == 0:
            llm.reset()
        llm.n_tokens = common
        llm.eval(tokens[common:])
        self._last = list(tokens)
        z = self.backend.last_logits().astype(np.float64)
        m = float(z.max())
        logp = z - m - math.log(float(np.exp(z - m).sum()))
        raw = {opt: float(np.exp(logp[i])) for opt, i in self.option_ids.items()}
        mass = sum(raw.values())
        out = {opt: (raw[opt] / mass if mass > 0 else 1.0 / len(OPTIONS)) for opt in OPTIONS}
        out["mass"] = mass
        out["prompt_tokens"] = float(len(tokens))
        return out


class StandInJudge:
    """Test double. ``rule``:
    - ``"longer"``: prefers the longer reply (consistent under swapping);
    - ``"first"``: always prefers whatever is shown as A (pure position bias);
    - ``"keyword"``: prefers the reply containing ``keyword``;
    - a callable ``(a_text, b_text) -> (pA, pB, pC)``.
    """

    def __init__(self, rule: str | Callable[[str, str], tuple[float, float, float]] = "longer",
                 keyword: str = "", key: str | None = None) -> None:
        self.rule, self.keyword = rule, keyword
        self.key = key or f"standin-{rule if isinstance(rule, str) else 'custom'}"
        self.calls = 0

    def probs(self, messages: Sequence[Mapping[str, str]]) -> dict[str, float]:
        self.calls += 1
        a, b = parse_rendered(messages)
        if callable(self.rule):
            pa, pb, pc = self.rule(a, b)
        elif self.rule == "longer":
            pa, pb, pc = (0.8, 0.1, 0.1) if len(a) > len(b) else (0.1, 0.8, 0.1) if len(b) > len(a) else (0.1, 0.1, 0.8)
        elif self.rule == "first":
            pa, pb, pc = 0.7, 0.2, 0.1
        elif self.rule == "keyword":
            ka, kb = self.keyword in a, self.keyword in b
            pa, pb, pc = (0.8, 0.1, 0.1) if ka and not kb else (0.1, 0.8, 0.1) if kb and not ka else (0.15, 0.15, 0.7)
        else:
            raise ValueError(f"unknown stand-in rule {self.rule!r}")
        s = pa + pb + pc
        return {"A": pa / s, "B": pb / s, "C": pc / s, "mass": 1.0}


def standin_judge(rule: str = "longer", keyword: str = "", key: str | None = None) -> StandInJudge:
    """Factory for ``--judge py:bes.eval.proxy.judge:standin_judge``."""
    return StandInJudge(rule=rule, keyword=keyword, key=key)


# ---------------------------------------------------------------------------- combination
def _argmax(p: Mapping[str, float]) -> str:
    return max(OPTIONS, key=lambda o: (p[o], -OPTIONS.index(o)))


def combine(first: str, second_swapped: str) -> str:
    """Final verdict from the two orders. ``first`` is the verdict on the blinded order,
    ``second_swapped`` the verdict on the swapped order *in the swapped labels*. A side wins only
    when both orders choose the same reply; otherwise (including any ``C``) it is a tie."""
    back = {"A": "B", "B": "A", "C": "C"}[second_swapped]
    if first == back and first in ("A", "B"):
        return first
    return "tie"


def judge_pair(judge: Judge, pair: Mapping[str, Any]) -> dict[str, Any]:
    """Both orders of one blinded pair -> verdict record (labels of the blinded order)."""
    turns = pair["turns"]
    swapped = [{"user": t["user"], "A": t["B"], "B": t["A"]} for t in turns]
    m1, c1 = render_messages(turns)
    m2, c2 = render_messages(swapped)
    t0 = time.perf_counter()
    p1 = judge.probs(m1)
    p2 = judge.probs(m2)
    v1, v2 = _argmax(p1), _argmax(p2)
    soft = {"A": (p1["A"] + p2["B"]) / 2, "B": (p1["B"] + p2["A"]) / 2, "C": (p1["C"] + p2["C"]) / 2}
    return {"pair_id": pair["pair_id"], "verdict": combine(v1, v2),
            "order1": {"verdict": v1, "probs": {o: round(p1[o], 5) for o in OPTIONS},
                       "mass": round(p1.get("mass", 1.0), 5)},
            "order2_swapped": {"verdict": v2, "probs": {o: round(p2[o], 5) for o in OPTIONS},
                               "mass": round(p2.get("mass", 1.0), 5)},
            "agree": v1 == {"A": "B", "B": "A", "C": "C"}[v2],
            "soft": {o: round(v, 5) for o, v in soft.items()},
            "first_position_pref": round(((p1["A"] - p1["B"]) + (p2["A"] - p2["B"])) / 2, 5),
            "clipped": c1 or c2, "judge": judge.key, "seconds": round(time.perf_counter() - t0, 3),
            "v": HARNESS_VERSION}


def ratings_path(root: Path, pairset: str, source: str) -> Path:
    return Path(root) / "ratings" / safe_name(pairset) / f"{safe_name(source)}.jsonl"


def run_judge(pairs: Sequence[Mapping[str, Any]], judge: Judge, pairset: str, root: Path | None = None,
              minutes: float | None = None, limit: int | None = None,
              log: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Judge every pair not yet judged by this judge (resumable); appends to
    ``ratings/<pairset>/judge-<key>.jsonl``."""
    root = Path(root) if root is not None else default_root()
    log = log or (lambda _m: None)
    path = ratings_path(root, pairset, f"judge-{judge.key}")
    done = {r["pair_id"] for r in read_jsonl(path)}
    pending = [p for p in pairs if p["pair_id"] not in done]
    todo = pending[:limit] if limit is not None else pending
    deadline = time.time() + 60 * minutes if minutes else None
    n = 0
    t0 = time.time()
    for pair in todo:
        if deadline is not None and time.time() > deadline:
            break
        append_jsonl(path, [judge_pair(judge, pair)])
        n += 1
        if n % 10 == 0:
            log(f"[judge {judge.key}] {n}/{len(todo)} pairs, {(time.time() - t0) / 60:.1f} min")
    return {"path": str(path), "judged_now": n, "already": len(done), "remaining": len(pending) - n}


def load_verdicts(root: Path, pairset: str, source: str) -> dict[str, str]:
    """``pair_id -> verdict`` of a rating source (``judge-<key>`` or ``human-<rater>``); the
    last record of a pair wins and ``retracted`` removes it."""
    out: dict[str, str] = {}
    for r in read_jsonl(ratings_path(root, pairset, source)):
        v = r.get("verdict")
        if v == "retracted":
            out.pop(r["pair_id"], None)
        elif v in ("A", "B", "tie", "both_bad"):
            out[r["pair_id"]] = v
    return out
