"""Run any GGUF chat model through llama.cpp (``llama_cpp`` 0.3.37) on CPU.

:class:`GGUFBackend` renders the conversation with the model's own chat template (the
``tokenizer.chat_template`` stored in the GGUF, through the same Jinja formatter llama-cpp-python
uses for ``create_chat_completion``; a template can be supplied for files that lack one), then
samples with llama.cpp's own sampler chain (penalties, top-k, typical, top-p, min-p,
temperature) seeded per call, and stops at any end-of-generation token or at ``max_tokens``.
While sampling it reads the raw logits of every step to accumulate the log-probability of each
sampled token under the model's unmodified distribution (no ``logits_all`` buffer needed);
this feeds the best-of-K control arm.

Each call starts from an empty context (no KV reuse across calls), so a reply depends only on
``(model, messages, settings, seed)`` and the thread count; outputs are reproducible across runs
on the same machine and ``n_threads``.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bes.eval.proxy.common import GenSettings, Message, Reply, file_sha256, stable_hash, strip_think

#: Appended to an unfinished reasoning block when its budget is spent (budget forcing).
THINK_CLOSE = "\n</think>\n\n"


class GGUFBackend:
    """A GGUF chat model. ``key`` defaults to ``gguf-<first 12 hex of the file's SHA-256>``
    (plus the template hash when a template override is given)."""

    def __init__(self, path: str | Path, n_ctx: int = 4096, n_threads: int = 1, n_batch: int = 512,
                 chat_template: str | None = None, key: str | None = None, verbose: bool = False,
                 **_ignored: Any) -> None:
        import llama_cpp
        from llama_cpp import llama_chat_format

        self._lib = llama_cpp
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.n_ctx = int(n_ctx)
        self.llm = llama_cpp.Llama(model_path=str(self.path), n_ctx=self.n_ctx, n_threads=n_threads,
                                   n_threads_batch=n_threads, n_batch=n_batch, logits_all=False,
                                   verbose=verbose)
        self.n_vocab = int(self.llm.n_vocab())
        template = chat_template or self.llm.metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(f"{self.path.name} has no chat template; pass chat_template=")
        self.template = template
        model = self.llm._model
        eos, bos = self.llm.token_eos(), self.llm.token_bos()
        self.formatter = llama_chat_format.Jinja2ChatFormatter(
            template=template, eos_token=model.token_get_text(eos) if eos >= 0 else "",
            bos_token=model.token_get_text(bos) if bos >= 0 else "", add_generation_prompt=True)
        self._vocab = model.vocab
        suffix = f"-t{stable_hash(template, 8)}" if chat_template else ""
        self.key = key or f"gguf-{file_sha256(self.path)[:12]}{suffix}"

    # ----------------------------------------------------------------------------- helpers
    def render(self, messages: Sequence[Message], **template_kwargs: Any) -> str:
        """The prompt text the chat template produces (generation prompt appended)."""
        return self.formatter(messages=[dict(m) for m in messages], **template_kwargs).prompt

    def tokenize(self, text: str) -> list[int]:
        """Token ids of a rendered prompt (special tokens parsed, no extra BOS: templates add
        their own)."""
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def is_eog(self, token: int) -> bool:
        return bool(self._lib.llama_vocab_is_eog(self._vocab, token))

    def last_logits(self) -> np.ndarray:
        """``float32[n_vocab]`` raw logits of the last evaluated position (a view)."""
        ptr = self._lib.llama_get_logits_ith(self.llm.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))

    def _logprob(self, token: int) -> float:
        z = self.last_logits().astype(np.float64)
        m = float(z.max())
        return float(z[token] - m - math.log(float(np.exp(z - m).sum())))

    # -------------------------------------------------------------------------- generation
    def _sample(self, context: list[int], settings: GenSettings, seed: int,
                limit: int) -> tuple[list[int], float, int, bool]:
        """Sample up to ``limit`` tokens after ``context`` from an empty KV state with llama.cpp's
        sampler chain. Returns ``(tokens, logprob_sum, logprob_count, stopped)``; the end-of-turn
        token is not returned but its log-probability is counted."""
        llm = self.llm
        llm.reset()
        llm.set_seed(int(seed))
        out: list[int] = []
        lp_sum, lp_n, stopped = 0.0, 0, False
        gen = llm.generate(context, top_k=settings.top_k, top_p=settings.top_p, min_p=settings.min_p,
                           typical_p=settings.typical_p, temp=settings.temperature,
                           repeat_penalty=settings.repeat_penalty,
                           frequency_penalty=settings.frequency_penalty,
                           presence_penalty=settings.presence_penalty, reset=True)
        try:
            for token in gen:
                lp_sum += self._logprob(token)
                lp_n += 1
                if self.is_eog(token):
                    stopped = True
                    break
                out.append(int(token))
                if len(out) >= limit:
                    break
        finally:
            gen.close()
        return out, lp_sum, lp_n, stopped

    def generate(self, messages: Sequence[Message], settings: GenSettings, seed: int) -> Reply:
        """One reply. ``settings.extra["think_budget"] = n`` enables budget forcing for reasoning
        models: if the reply is still inside ``<think>`` after ``n`` tokens, :data:`THINK_CLOSE` is
        appended and the model writes its answer with the rest of ``max_tokens`` (the forced
        tokens get no log-probability)."""
        prompt_text = self.render(messages)
        tokens = self.tokenize(prompt_text)
        budget = min(settings.max_tokens, self.n_ctx - len(tokens) - 1)
        if budget <= 0:
            raise ValueError(f"prompt of {len(tokens)} tokens does not fit n_ctx={self.n_ctx}")
        think_budget = int(settings.extra_dict.get("think_budget") or 0)
        t0 = time.perf_counter()
        first = min(budget, think_budget) if think_budget > 0 else budget
        out, lp_sum, lp_n, stopped = self._sample(tokens, settings, int(seed), first)
        extra: dict[str, Any] = {"template": stable_hash(self.template, 8)}
        if think_budget > 0 and not stopped and len(out) >= first and len(out) < budget:
            partial = self.llm.detokenize(out, prev_tokens=tokens).decode("utf-8", errors="replace")
            if "<think>" in partial and "</think>" not in partial:
                # budget forcing: end the reasoning and let the model answer in the remaining budget
                close = self.tokenize(THINK_CLOSE)
                room = min(budget - len(out), self.n_ctx - len(tokens) - len(out) - len(close) - 1)
                if room > 0:
                    more, lp2, n2, stopped = self._sample(tokens + out + close, settings,
                                                          (int(seed) + 1) & 0x7FFFFFFF, room)
                    out = out + close + more
                    lp_sum, lp_n = lp_sum + lp2, lp_n + n2
                    extra.update({"forced_answer": True, "think_tokens": first})
        finish = "stop" if stopped else "length"
        latency = time.perf_counter() - t0
        raw = self.llm.detokenize(out, prev_tokens=tokens).decode("utf-8", errors="replace")
        text = raw
        if settings.strip_think:
            text, info = strip_think(raw)
            extra.update(info)
        return Reply(text=text.strip(), raw_text=raw, prompt_tokens=len(tokens),
                     completion_tokens=len(out), latency_s=latency,
                     finish_reason=finish, logprob_sum=lp_sum, logprob_count=lp_n, extra=extra)


def gguf_spec(path: str | Path, n_ctx: int = 4096, n_threads: int = 1,
              chat_template: str | None = None, key: str | None = None) -> dict[str, Any]:
    """Picklable backend spec for :func:`bes.eval.proxy.common.make_backend`. The key is
    computed here (in the parent) so that the cache file is known before workers start."""
    if key is None:
        suffix = f"-t{stable_hash(chat_template, 8)}" if chat_template else ""
        key = f"gguf-{file_sha256(path)[:12]}{suffix}"
    spec: dict[str, Any] = {"kind": "gguf", "path": str(path), "n_ctx": n_ctx, "n_threads": n_threads,
                            "key": key}
    if chat_template:
        spec["chat_template"] = chat_template
    return spec
