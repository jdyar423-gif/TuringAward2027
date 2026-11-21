"""The particle decoder over a llama.cpp model (llama-cpp-python 0.3.37, low-level calls).

Three pieces, all CPU:

- :class:`GGUFVocab`: the model's vocabulary seen as a tokenizer by the exact layer
  (``bes.exact.instr_table.build`` and ``bes.decode.potentials.compile``): ``vocab_size``,
  ``special_tokens``, ``sha256``, ``all_token_bytes``, ``token_bytes``, ``decode_bytes``,
  ``encode``. Ids ``0 .. n_model - 1`` are the model's tokens; three *virtual* ids follow
  (``<|calc|>``, ``<|eq|>``, ``<|/calc|>``: the exact layer needs exactly one id of each kind,
  stand-ins have no calculator and the model never produces them). Mapping of the model's
  tokens:

  - the template's end-of-turn token (``<|im_end|>`` for ChatML templates) is ``<|end|>``;
    the model's BOS is ``<|bos|>`` (a fourth virtual id when the model has none);
  - every other control token is a special token of its own (kind "other": never sampled
    under the structural potentials, no bytes for the transducer), named by its text, or
    ``<|gguf_<id>|>`` when that text is taken;
  - tokens of more than 127 bytes (the transducer tables store byte counts in ``int8``) are
    special too (``<|gguf_long_<id>|>``): the potentials never sample them; their real bytes
    are still shown by :meth:`GGUFVocab.text`;
  - every other token (normal and user-defined) is ordinary, with the bytes of
    ``llama_token_to_piece(..., special=False)``.

- :class:`GGUFChat`: the model's own chat template (``tokenizer.chat_template`` through the
  Jinja formatter llama-cpp-python uses, as in ``bes.eval.proxy.run_gguf``), tokenised with
  special tokens parsed and no extra BOS; ``display(tokens)`` is the visible text.

- :class:`GGUFLM`: the ``bes.decode.lm.LanguageModel`` protocol. ``K`` particles are ``K``
  sequences of one ``llama_context`` (``n_seq_max = K``, unified KV cache); one ``llama_decode``
  per decoding step evaluates every active particle (a batch of one token per sequence).
  Resampling copies whole sequence states with ``llama_state_seq_get_data`` /
  ``llama_state_seq_set_data`` (KV cache *and* recurrent state). ``llama_memory_seq_cp`` is
  not used: on the hybrid attention + SSM opponent it shares a recurrent cell lazily, and
  after such copies in the middle of decoding the sequences involved gave log-probabilities
  that differ from a fresh evaluation of the same tokens by up to 0.8 (measured; the
  sequences not involved, and every sequence after explicit state copies, matched the same
  reference exactly). The tests compare every particle, through steps, inactive steps and
  resampling, with llama-cpp-python's high-level ``Llama`` fed the same way (the prompt in one
  batch, then one token at a time): agreement within 1e-6. Two properties of llama.cpp to
  know: its results for this model depend on the batch shape (one batch of a whole sequence
  differs from token-by-token evaluation by up to about 2 in a tail log-probability), and on
  the attention kernel (see ``GGUFLM``'s ``flash_attn``). Log-probabilities are ``log_softmax`` of the raw
  logits in float64 (returned as float32 ``[K, vocab_size]``, ``-inf`` for the virtual ids);
  with ``merge_eog`` (default) the mass of every end-of-generation token
  (``llama_vocab_is_eog``) is moved onto ``<|end|>``, so ending the turn means what it means
  for llama.cpp's own sampling loop.

No candidate tape and no calculator for stand-ins: the decoder sees only the log-probabilities.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Byte limit of a token for the transducer tables (``instr_table`` stores counts in int8).
MAX_TOKEN_BYTES = 127
#: End-of-turn markers looked for in a chat template, in order of preference.
END_MARKERS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "</s>")
VIRTUAL = ("<|calc|>", "<|eq|>", "<|/calc|>")
_RESERVED = frozenset(("<|bos|>", "<|end|>", "<|calc|>", "<|eq|>", "<|/calc|>", "<|system|>",
                       "<|user|>", "<|assistant|>", "<|library|>"))


# ============================================================================ vocabulary
class GGUFVocab:
    """A llama.cpp vocabulary as the exact layer's tokenizer (module docstring).

    pieces: bytes of every model token (``b""`` for control tokens); control: ``bool[n]``
    control tokens; eog: ``bool[n]`` end-of-generation tokens; bos / end: model ids (``bos``
    may be ``None``); texts: the token texts (names of control tokens); tokenize: optional
    ``text -> ids`` (ordinary tokenisation, special tokens not parsed); tag: identifies the
    model file in ``sha256``.
    """

    def __init__(self, pieces: Sequence[bytes], control: Sequence[bool], eog: Sequence[bool],
                 bos: int | None, end: int, texts: Sequence[str] | None = None,
                 tokenize=None, tag: str = ""):
        n = len(pieces)
        if not (len(control) == len(eog) == n):
            raise ValueError("pieces, control and eog must have one entry per token")
        if not 0 <= int(end) < n:
            raise ValueError("the end-of-turn id must be a model token")
        self.n_model = n
        self.pieces = [bytes(p) for p in pieces]
        self.control = np.asarray(control, bool).copy()
        self.eog = np.asarray(eog, bool).copy()
        self.eog[int(end)] = True
        self.end = int(end)
        self._tokenize = tokenize
        texts = list(texts) if texts is not None else [p.decode("utf-8", "replace") for p in pieces]
        special: dict[str, int] = {"<|end|>": self.end}
        extra = list(VIRTUAL)
        if bos is None or not 0 <= int(bos) < n:
            extra.append("<|bos|>")
            self.bos = None
        else:
            self.bos = int(bos)
            special["<|bos|>"] = self.bos
        for i, name in enumerate(extra):
            special[name] = n + i
        self.vocab_size = n + len(extra)
        self.long = np.array([len(p) > MAX_TOKEN_BYTES for p in self.pieces], bool)
        for t in range(n):
            if t in (self.end, self.bos):
                continue
            if self.control[t] or self.long[t] or not self.pieces[t]:
                name = texts[t] if self.control[t] else f"<|gguf_long_{t}|>"
                if not name or name in special or name in _RESERVED:
                    name = f"<|gguf_{t}|>"
                special[name] = t
        self.special_tokens = special
        self.special_ids = frozenset(special.values())
        self._bytes = [b"" if t in self.special_ids else self.pieces[t] for t in range(n)] \
            + [b""] * len(extra)
        h = hashlib.sha256(b"bes-gguf-vocab/1\x00" + tag.encode("utf-8") + b"\x00")
        h.update(json.dumps(sorted(special.items())).encode("utf-8"))
        for p in self.pieces:
            h.update(len(p).to_bytes(2, "little") + p)
        self.sha256 = h.hexdigest()

    # -- tokenizer protocol of the exact layer ------------------------------------------
    def all_token_bytes(self) -> list[bytes]:
        return list(self._bytes)

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[int(token_id)]

    def decode_bytes(self, ids) -> bytes:
        """Exact-layer bytes of ``ids`` (special and virtual ids contribute nothing)."""
        return b"".join(self._bytes[int(i)] for i in ids)

    def encode(self, text: str) -> list[int]:
        """Ordinary tokenisation of ``text`` (special tokens are not parsed)."""
        if self._tokenize is None:
            raise ValueError("this vocabulary has no tokenizer")
        return [int(t) for t in self._tokenize(text)]

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self.special_ids

    # -- display --------------------------------------------------------------------------
    def text(self, ids) -> str:
        """Visible text of ``ids``: the real bytes of every non-control model token (long
        tokens included), decoded as UTF-8 (invalid bytes become U+FFFD)."""
        out = []
        for i in ids:
            i = int(i)
            if i < self.n_model and not self.control[i]:
                out.append(self.pieces[i])
        return b"".join(out).decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"GGUFVocab(n_model={self.n_model}, vocab_size={self.vocab_size}, sha256={self.sha256[:12]})"


def _end_token(template: str, texts: Sequence[str], control: np.ndarray, eot: int, eos: int) -> int:
    for marker in END_MARKERS:
        if marker in template:
            ids = [t for t, s in enumerate(texts) if s == marker and control[t]]
            if ids:
                return ids[0]
    if eot is not None and eot >= 0:
        return int(eot)
    if eos is not None and eos >= 0:
        return int(eos)
    raise ValueError("cannot find the end-of-turn token of this model")


# ================================================================================ model
class GGUFModel:
    """One GGUF file loaded once (weights, vocabulary, chat template).

    path: the GGUF file; chat_template: override (default: the file's
    ``tokenizer.chat_template``); end_token: override of the end-of-turn token text;
    verbose: llama.cpp logging (default off).
    """

    def __init__(self, path: str | Path, *, chat_template: str | None = None,
                 end_token: str | None = None, verbose: bool = False):
        import llama_cpp
        from llama_cpp import _internals

        if not verbose:
            logging.getLogger("llama-cpp-python").setLevel(logging.ERROR)
        self._lib = llama_cpp
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        params = llama_cpp.llama_model_default_params()
        self.model = _internals.LlamaModel(path_model=str(self.path), params=params,
                                           verbose=verbose)
        lib, vocab = llama_cpp, self.model.vocab
        n = int(self.model.n_vocab())
        buf = ctypes.create_string_buffer(1024)
        pieces, texts = [], []
        for t in range(n):
            k = lib.llama_token_to_piece(vocab, t, buf, len(buf), 0, False)
            if k < 0:
                raise RuntimeError(f"token {t}: piece longer than {len(buf)} bytes")
            pieces.append(bytes(buf.raw[:k]))
            raw = lib.llama_vocab_get_text(vocab, t)
            texts.append(raw.decode("utf-8", "replace") if raw else "")
        control = np.array([bool(lib.llama_vocab_is_control(vocab, t)) for t in range(n)])
        eog = np.array([bool(lib.llama_vocab_is_eog(vocab, t)) for t in range(n)])
        meta = self.model.metadata()
        template = chat_template or meta.get("tokenizer.chat_template")
        if not template:
            raise ValueError(f"{self.path.name} has no chat template; pass chat_template=")
        self.template = template
        if end_token is not None:
            ids = [t for t, s in enumerate(texts) if s == end_token]
            if not ids:
                raise ValueError(f"no token {end_token!r}")
            end = ids[0]
        else:
            end = _end_token(template, texts, control, self.model.token_eot(),
                             self.model.token_eos())
        bos = self.model.token_bos()
        self.vocab = GGUFVocab(pieces, control, eog, bos if bos >= 0 else None, end, texts,
                               tokenize=self._encode_ordinary, tag=self.path.name)

    def _encode_ordinary(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False)

    def tokenize_prompt(self, text: str) -> list[int]:
        """Ids of a rendered prompt: special tokens parsed, no extra BOS (templates add their own)."""
        return self.model.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def close(self) -> None:
        self.model.close()


# ================================================================================= chat
class GGUFChat:
    """Chat format of a GGUF model: its own template, its end-of-turn token, visible text."""

    def __init__(self, model: GGUFModel):
        from llama_cpp import llama_chat_format

        self.model = model
        self.vocab = model.vocab
        lm = model.model
        eos, bos = lm.token_eos(), lm.token_bos()
        self.formatter = llama_chat_format.Jinja2ChatFormatter(
            template=model.template, eos_token=lm.token_get_text(eos) if eos >= 0 else "",
            bos_token=lm.token_get_text(bos) if bos >= 0 else "", add_generation_prompt=True)
        self.end_id = model.vocab.end

    def render_text(self, messages: Sequence[Mapping[str, Any]]) -> str:
        msgs = []
        for m in messages:
            if m["role"] not in ("system", "user", "assistant"):
                raise ValueError(f"the GGUF chat format has no {m['role']!r} role")
            msgs.append({"role": m["role"], "content": m["content"]})
        return self.formatter(messages=msgs).prompt

    def render(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """Prompt ids ending with the template's generation prompt."""
        return self.model.tokenize_prompt(self.render_text(messages))

    def display(self, tokens: Sequence[int]) -> str:
        """Visible text of a response (up to ``END``)."""
        toks = list(tokens)
        if self.end_id in toks:
            toks = toks[:toks.index(self.end_id)]
        return self.vocab.text(toks)


# ================================================================================== LM
def _log_softmax64(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    m = float(np.max(z))
    return z - (m + np.log(np.sum(np.exp(z - m))))


class GGUFLM:
    """``K`` particles of one prompt on a llama.cpp model (module docstring).

    model: a :class:`GGUFModel`; n_threads: llama.cpp threads; n_batch: prefill chunk;
    n_ctx: KV cells of the context (default: enough for ``K`` sequences of prompt +
    ``max_new_tokens``, rounded up; the context is recreated when a prompt needs more);
    max_new_tokens: tokens per particle the context must hold after the prompt;
    merge_eog: move the probability of every end-of-generation token onto ``<|end|>``;
    flash_attn: llama.cpp's attention kernel (``None``: the library default, which enables
    flash attention for this model on CPU; ``False``: the plain kernel that
    ``llama_cpp.Llama`` uses by default, e.g. in ``bes.eval.proxy.run_gguf``). The two
    kernels are different numerical approximations of the same model: on the stand-in they
    differ by a total variation of about 0.003 per next-token distribution (measured), while
    the same kernel gives identical log-probabilities to about 1e-6 in any context.

    Counters for cost reports: ``decode_calls``, ``tokens_evaluated`` (prompt tokens
    included), ``state_copies``.
    """

    def __init__(self, model: GGUFModel, *, n_threads: int = 1, n_batch: int = 512,
                 n_ctx: int | None = None, max_new_tokens: int = 512, merge_eog: bool = True,
                 flash_attn: bool | None = None):
        self.model = model
        self.vocab = model.vocab
        self.vocab_size = int(model.vocab.vocab_size)
        self.n_model = int(model.vocab.n_model)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        self.n_ctx = None if n_ctx is None else int(n_ctx)
        self.max_new_tokens = int(max_new_tokens)
        self.merge_eog = bool(merge_eog)
        self.flash_attn = flash_attn
        self._lib = model._lib
        self._ctx = None
        self._batch = None
        self._n_seq = 0
        self._cells = 0
        eog = np.flatnonzero(model.vocab.eog)
        self._eog_other = np.array([t for t in eog if t != model.vocab.end], np.int64)
        self._eog = eog
        self.decode_calls = 0
        self.tokens_evaluated = 0
        self.state_copies = 0
        self.k = 0

    # -- context ----------------------------------------------------------------------
    def _make_context(self, n_seq: int, cells: int) -> None:
        from llama_cpp import _internals

        lib = self._lib
        self._free()
        cp = lib.llama_context_default_params()
        cp.n_ctx = int(cells)
        cp.n_batch = self.n_batch
        cp.n_ubatch = self.n_batch
        cp.n_seq_max = int(n_seq)
        cp.n_threads = self.n_threads
        cp.n_threads_batch = self.n_threads
        cp.kv_unified = True
        if self.flash_attn is not None:
            cp.flash_attn_type = (lib.LLAMA_FLASH_ATTN_TYPE_ENABLED if self.flash_attn
                                  else lib.LLAMA_FLASH_ATTN_TYPE_DISABLED)
        self._ctx = _internals.LlamaContext(model=self.model.model, params=cp, verbose=False)
        self._batch = _internals.LlamaBatch(n_tokens=max(self.n_batch, n_seq), embd=0,
                                            n_seq_max=1, verbose=False)
        self._n_seq, self._cells = int(n_seq), int(cells)

    def _ensure(self, n_seq: int, need: int) -> None:
        cells = max(need, self.n_ctx or 0)
        cells = -(-cells // 256) * 256
        if self._ctx is None or self._n_seq != n_seq or self._cells < need:
            self._make_context(n_seq, cells)

    def _free(self) -> None:
        if self._batch is not None:
            self._batch.close()
            self._batch = None
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def _decode(self, entries: list[tuple[int, int, int, bool]]) -> None:
        """One ``llama_decode`` of ``(token, pos, seq, want_logits)`` entries."""
        b = self._batch.batch
        for i, (tok, pos, seq, want) in enumerate(entries):
            b.token[i] = tok
            b.pos[i] = pos
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = seq
            b.logits[i] = bool(want)
        b.n_tokens = len(entries)
        rc = self._lib.llama_decode(self._ctx.ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc} (context of {self._cells} cells full?)")
        self.decode_calls += 1
        self.tokens_evaluated += len(entries)

    def _logp(self, i: int) -> np.ndarray:
        ptr = self._lib.llama_get_logits_ith(self._ctx.ctx, i)
        z = np.ctypeslib.as_array(ptr, shape=(self.n_model,))
        lp = _log_softmax64(z)
        out = np.full(self.vocab_size, -np.inf)
        out[:self.n_model] = lp
        if self.merge_eog and self._eog.size > 1:
            e = lp[self._eog]
            m = float(np.max(e))
            out[self.vocab.end] = m + np.log(np.sum(np.exp(e - m))) if np.isfinite(m) else -np.inf
            out[self._eog_other] = -np.inf
        return out.astype(np.float32)

    def _copy_state(self, src: int, dst: int, cache: dict) -> None:
        lib, ctx = self._lib, self._ctx.ctx
        if src not in cache:
            n = lib.llama_state_seq_get_size(ctx, src)
            buf = (ctypes.c_uint8 * n)()
            got = lib.llama_state_seq_get_data(ctx, buf, n, src)
            if got <= 0:
                raise RuntimeError(f"could not read the state of sequence {src}")
            cache[src] = (buf, got)
        buf, got = cache[src]
        lib.llama_memory_seq_rm(self._ctx.memory, dst, -1, -1)
        if lib.llama_state_seq_set_data(ctx, buf, got, dst) != got:
            raise RuntimeError(f"could not copy the state of sequence {src} to {dst}")
        self.state_copies += 1

    # -- LanguageModel protocol ---------------------------------------------------------
    def prefill(self, prompt_ids: Sequence[int], n_particles: int) -> np.ndarray:
        prompt = [int(t) for t in prompt_ids]
        k = int(n_particles)
        if not prompt:
            raise ValueError("the prompt must hold at least one token")
        if k < 1:
            raise ValueError("n_particles must be >= 1")
        if min(prompt) < 0 or max(prompt) >= self.n_model:
            raise ValueError("prompt ids outside the model's vocabulary")
        self._ensure(k, k * (len(prompt) + self.max_new_tokens + 1))
        self._lib.llama_memory_clear(self._ctx.memory, True)
        n = len(prompt)
        for lo in range(0, n, self.n_batch):
            chunk = prompt[lo:lo + self.n_batch]
            self._decode([(t, lo + i, 0, lo + i == n - 1) for i, t in enumerate(chunk)])
        logp = self._logp(len(prompt[(n - 1) // self.n_batch * self.n_batch:]) - 1)
        cache: dict = {}
        for s in range(1, k):
            self._copy_state(0, s, cache)
        self.k = k
        self._slot = np.arange(k, dtype=np.int64)
        self._pos = np.full(k, n, np.int64)
        return np.repeat(logp[None], k, axis=0)

    def step(self, tokens: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        tok = np.asarray(tokens, np.int64).reshape(-1)
        if tok.shape[0] != self.k:
            raise ValueError(f"expected {self.k} tokens, got {tok.shape[0]}")
        act = np.ones(self.k, bool) if active is None else np.asarray(active, bool).reshape(-1)
        out = np.zeros((self.k, self.vocab_size), np.float32)
        rows = np.flatnonzero(act)
        if rows.size == 0:
            return out
        if np.any(tok[rows] < 0) or np.any(tok[rows] >= self.n_model):
            raise ValueError("a particle was fed a token the model does not have")
        self._decode([(int(tok[j]), int(self._pos[j]), int(self._slot[j]), True) for j in rows])
        for i, j in enumerate(rows):
            out[j] = self._logp(i)
        self._pos[rows] += 1
        return out

    def reorder(self, ancestors: np.ndarray) -> None:
        anc = np.asarray(ancestors, np.int64).reshape(-1)
        if anc.shape[0] != self.k or (self.k and (anc.min() < 0 or anc.max() >= self.k)):
            raise ValueError("ancestors must hold K indices in [0, K)")
        if np.array_equal(anc, np.arange(self.k)):
            return
        new_slot = np.full(self.k, -1, np.int64)
        claimed: set[int] = set()
        for j, a in enumerate(anc):
            if int(a) not in claimed:
                claimed.add(int(a))
                new_slot[j] = self._slot[a]
        free = sorted(set(range(self.k)) - set(int(s) for s in new_slot if s >= 0))
        cache: dict = {}
        for j in np.flatnonzero(new_slot < 0):
            dst = free.pop(0)
            self._copy_state(int(self._slot[anc[j]]), dst, cache)
            new_slot[j] = dst
        self._slot = new_slot
        self._pos = self._pos[anc].copy()

    def close(self) -> None:
        self._free()
        self.k = 0

    def __del__(self):
        try:
            self._free()
        except Exception:
            pass


__all__ = ["GGUFVocab", "GGUFModel", "GGUFChat", "GGUFLM", "MAX_TOKEN_BYTES", "END_MARKERS"]
