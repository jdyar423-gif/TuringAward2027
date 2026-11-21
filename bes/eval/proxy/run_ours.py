"""Run the project's own model through the harness.

The generator is written by other tracks (``bes/decode``, ``bes/model/generate.py``); the
harness only needs a callable with this contract (:class:`OurGenerator`)::

    generator(messages, settings, seed) -> str | dict

- ``messages``: ``[{"role": "system" | "user" | "assistant", "content": str}, ...]`` ending with
  a user message. Earlier assistant messages are the generator's own earlier replies exactly as
  it returned them (visible text). No library passage is included: retrieval, if enabled, is
  the generator's business (and must only use the frozen offline library).
- ``settings``: a plain dict with ``temperature``, ``top_p``, ``top_k``, ``min_p``,
  ``repeat_penalty``, ``max_tokens``, ``system_prompt`` and every key of the arm's ``extra``
  options (for example ``particles``, ``use_library``, ``use_critic``, named like the fields of
  ``bes.config.DecodeCfg``). ``max_tokens`` counts our tokens.
- ``seed``: the generator must be deterministic given ``(messages, settings, seed)``.
- returns the reply text, or a dict with ``text`` (required) and optionally
  ``prompt_tokens``, ``completion_tokens``, ``finish_reason`` (``"stop"`` / ``"length"``),
  ``logprob_sum``, ``logprob_count`` and ``extra`` (a JSON-serialisable dict, for example
  ``{"particles": 6, "library": "Title of passage", "calc_spans": 1}``).

The harness measures latency itself and caches replies by a ``key`` that must identify the
checkpoint and decoding code: the generator object may carry a ``key`` attribute, or the
command line passes ``--key``. A factory ``"module:function"`` builds the generator inside each
worker process from string keyword arguments (``--generator-arg name=value``).

Stand-ins (:func:`standin`) produce deterministic text for tests and rehearsals;
:func:`gguf_generator` wraps any GGUF chat model behind this same contract (useful for an
end-to-end rehearsal before our checkpoint exists).
"""
from __future__ import annotations

import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from bes.eval.proxy.common import GenSettings, Message, Reply


class OurGenerator(Protocol):
    def __call__(self, messages: Sequence[Message], settings: Mapping[str, Any],
                 seed: int) -> str | Mapping[str, Any]:
        ...


def settings_for_generator(settings: GenSettings) -> dict[str, Any]:
    """The plain dict a generator receives (see the module docstring)."""
    out = {"temperature": settings.temperature, "top_p": settings.top_p, "top_k": settings.top_k,
           "min_p": settings.min_p, "repeat_penalty": settings.repeat_penalty,
           "max_tokens": settings.max_tokens, "system_prompt": settings.system_prompt}
    out.update(settings.extra_dict)
    return out


class CallableBackend:
    """Adapts an :class:`OurGenerator` to the harness backend interface."""

    def __init__(self, generator: OurGenerator, key: str | None = None) -> None:
        self.generator = generator
        key = key or getattr(generator, "key", None)
        if not key:
            raise ValueError("our generator needs a cache key: give it a 'key' attribute or pass --key")
        self.key = str(key)

    def generate(self, messages: Sequence[Message], settings: GenSettings, seed: int) -> Reply:
        t0 = time.perf_counter()
        out = self.generator([dict(m) for m in messages], settings_for_generator(settings), int(seed))
        latency = time.perf_counter() - t0
        if isinstance(out, str):
            return Reply(text=out.strip(), raw_text=out, latency_s=latency)
        if not isinstance(out, Mapping) or not isinstance(out.get("text"), str):
            raise TypeError("a generator must return a str or a dict with a 'text' string")
        return Reply(text=out["text"].strip(), raw_text=out.get("raw_text", out["text"]),
                     prompt_tokens=int(out.get("prompt_tokens", 0)),
                     completion_tokens=int(out.get("completion_tokens", 0)),
                     latency_s=latency, finish_reason=str(out.get("finish_reason", "stop")),
                     logprob_sum=out.get("logprob_sum"), logprob_count=int(out.get("logprob_count", 0)),
                     extra=dict(out.get("extra", {})))


# ------------------------------------------------------------------------------ stand-ins
_OPENERS = ("Sure.", "Here is an answer.", "Good question.", "Happy to help.")


class StandIn:
    """A deterministic stand-in generator (no model).

    ``style``:
    - ``"plain"``: a short, varied reply that restates the request; for arithmetic it evaluates
      the first ``a op b`` it finds, so math scoring has something to grade;
    - ``"verbose"``: the plain reply padded with filler sentences (length-bias probes);
    - ``"loop"``: a reply that degenerates into a repeated phrase;
    - ``"empty"``: an empty reply.
    The reply depends only on ``(messages, settings, seed)``.
    """

    def __init__(self, style: str = "plain", key: str | None = None) -> None:
        if style not in ("plain", "verbose", "loop", "empty"):
            raise ValueError(f"unknown stand-in style {style!r}")
        self.style = style
        self.key = key or f"standin-{style}"
        self.calls = 0

    def __call__(self, messages: Sequence[Message], settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.calls += 1
        rng = random.Random(f"{seed}:{messages[-1]['content']}:{len(messages)}")
        user = messages[-1]["content"]
        if self.style == "empty":
            return {"text": "", "completion_tokens": 0, "finish_reason": "stop",
                    "logprob_sum": -1.0, "logprob_count": 1}
        words = re.findall(r"[A-Za-z']+", user)[:10]
        text = f"{rng.choice(_OPENERS)} You asked about {' '.join(words).lower()}."
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([-+*/x×])\s*(-?\d+(?:\.\d+)?)", user)
        if m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            value = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "×": a * b,
                     "/": a / b if b else float("nan")}[op]
            text += f" The answer is {value:g}."
        if self.style == "verbose":
            text += " " + " ".join(rng.choice(["This is worth noting.", "There is more to say.",
                                               "Many people wonder about this."]) for _ in range(12))
        if self.style == "loop":
            text += " and then" * 40
        n_tok = len(text.split())
        finish = "length" if self.style == "loop" else "stop"
        return {"text": text, "completion_tokens": n_tok, "prompt_tokens": sum(len(m["content"].split()) for m in messages),
                "finish_reason": finish, "logprob_sum": -0.5 * n_tok - rng.random(), "logprob_count": n_tok + 1,
                "extra": {"style": self.style}}


def standin(style: str = "plain", key: str | None = None) -> StandIn:
    """Factory for ``--generator bes.eval.proxy.run_ours:standin --generator-arg style=plain``."""
    return StandIn(style=style, key=key)


def gguf_generator(path: str, n_ctx: str | int = 4096, n_threads: str | int = 1,
                   key: str | None = None) -> OurGenerator:
    """Wrap a GGUF chat model behind the generator contract (rehearsals only)."""
    from bes.eval.proxy.run_gguf import GGUFBackend

    backend = GGUFBackend(path, n_ctx=int(n_ctx), n_threads=int(n_threads), key=key)
    known = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty", "max_tokens", "system_prompt"}

    def generate(messages: Sequence[Message], settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
        s = GenSettings(**{k: v for k, v in settings.items() if k in known})
        r = backend.generate(messages, s, seed)
        return {"text": r.text, "raw_text": r.raw_text, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason,
                "logprob_sum": r.logprob_sum, "logprob_count": r.logprob_count, "extra": r.extra}

    generate.key = f"ours-as-{backend.key}"  # type: ignore[attr-defined]
    return generate
