"""Generation with a key-value cache (SPEC 4.4).

Each position's hyper-connection lanes are local to that position, so only attention needs a
cache: for every loop ``r`` and block ``i`` the assembled full-width keys and values of the
earlier positions (``head_dim`` wide, reserved bias dims, sink and confidence dims included).

Cache (a dict of arrays, static shapes, compiled once per ``(B, Tmax)``)::

    k, v      [n_loops, n_blocks, B, H, Tmax, head_dim]   activation dtype
    seg       int32 [B, Tmax]   segment id of each slot; -1 = empty or padding (never visible)
    pos       int32 [B, Tmax]   position of each slot (the rotary position used by the mask)
    t         int32 []          next free slot (shared by all rows)
    next_pos  int32 [B]         default position of the next token of each row
    cur_seg   int32 [B]         default segment id of the next token of each row

and, only with selective attention (``cfg.sel_attn``)::

    sel       float32 [n_loops, n_blocks, B, Tmax]   selection accumulated by every query so far
              (``sum_i S[i, j]`` of ``bes.model.attention.selective_logits`` per key slot ``j``)
    bos       bool [B, Tmax]    the slot holds a BOS token (a key selection never masks)

A key is visible to a query when ``pos_k <= pos_q`` and ``seg_k == seg_q``; slots are only an
address, so gaps of padding or empty slots are harmless. Every call appends ``Tq`` slots at
``t`` (``prefill``: a chunk ``[B, Tq]``; ``decode_step``: one token per row) and returns the
next-token scores of the chunk's last position (``model.logits_from_hidden``: ``softmax`` of the
scores is the output law ``p_T`` at the sampling temperature ``T``, which tempers the network's
distribution only; with a candidate tape the scores are its normalised log-probabilities, so
``logsumexp = 0``). The temperature is a traced float32 scalar (default 1: the law itself), so
:func:`jitted` compiles one program for every temperature. Positions whose ``seg`` is negative
are padding: they are written with ``seg = -1`` and never seen by real queries, which gives
left-padded ragged prompts and fixed-size prefill chunks (one compilation for any prompt
length).

The v9 write-side paths need no cache state of their own: the value gain ``gain/v`` is inside
the cached values; the output gain ``gain/o``, the FFN gain ``gain/ffn``, the token neurons
(rows of the chunk's tokens, ``model.token_rows``) and the head-wise output gate (the loop index
is fed through the loop scan) act at the new positions only. Selective attention carries the
selection of the earlier queries in ``sel``: for queries at slots ``t .. t + Tq - 1``,
``F[i, j] = sel[j] + sum_{i' < i in the chunk} S[i', j]``, then ``sel[j] += sum_i S[i, j]``
(the new slots start at 0). Prefill in any chunking followed by single-token steps reproduces
the full-sequence forward as long as positions strictly increase from slot to slot within a
segment (as in every stream the host loops build; with two equal positions the earlier query
sees the later key in the full forward, but in the cache that key does not exist yet).

Feature rows (``reg``, ``clock``, ``cand_tok``, ``cand_feat``) follow ``docs/SPEC.md`` 2: the row
of position ``t`` describes the state after consuming token ``t`` and its candidate tape lists
candidates for token ``t + 1``; their masses enter the output law at the last position with that
row (``consts["enable"]["src"]`` scales them per ``src`` value, e.g. ``src_5 < 1`` for evidence
from the text being generated).

Host-side loops: :func:`generate` (one stream) and :func:`generate_particles` (``K`` rows, each
an independent particle of the same prompt, with optional resampling through
:func:`reorder_cache`). Feature providers are objects with ``step(token) -> {"reg", "clock",
"cand_tok", "cand_feat"}`` rows (``bes.exact.features.IncrementalFeatures``; :class:`ZeroFeatures`
is a stand-in) and, for particles, ``fork()`` (otherwise ``copy.deepcopy`` is used).
"""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_SLOTS, END, N_CAND_FEAT, N_CLOCKS, N_REG, special_id
from bes.model import controller, head
from bes.model import model as M
from bes.model.attention import NEG_INF, selective_logits
from bes.model.layers import act_dtype, hi, rmsnorm

FEATURE_KEYS = ("reg", "clock", "cand_tok", "cand_feat")
_POSITION_KEYS = ("seg", "pos")
#: default prefill chunk length (one compilation per (B, chunk, Tmax) covers every prompt)
DEFAULT_CHUNK = 64
#: default granularity of the cache length chosen by the host loops
TMAX_ROUND = 128


# ------------------------------------------------------------------------------ cache
def init_cache(cfg, batch: int, tmax: int, dtype=None) -> dict:
    """Empty cache for ``batch`` rows and ``tmax`` slots (see the module docstring).

    dtype: dtype of the keys and values (default: the activation dtype of ``cfg``). The
    selective-attention entries ``sel`` (``hi(act_dtype)``: float32, the dtype of the logits it
    accumulates) and ``bos`` exist only with ``cfg.sel_attn``; no other v9 path has cache
    state."""
    dt = jnp.dtype(dtype) if dtype is not None else act_dtype(cfg)
    shape = (cfg.n_loops, cfg.n_blocks, batch, cfg.n_heads, tmax, cfg.head_dim)
    cache = {"k": jnp.zeros(shape, dt), "v": jnp.zeros(shape, dt),
             "seg": jnp.full((batch, tmax), -1, jnp.int32),
             "pos": jnp.zeros((batch, tmax), jnp.int32),
             "t": jnp.zeros((), jnp.int32),
             "next_pos": jnp.zeros((batch,), jnp.int32),
             "cur_seg": jnp.zeros((batch,), jnp.int32)}
    if cfg.sel_attn:
        cache["sel"] = jnp.zeros((cfg.n_loops, cfg.n_blocks, batch, tmax), hi(act_dtype(cfg)))
        cache["bos"] = jnp.zeros((batch, tmax), jnp.bool_)
    return cache


def cache_capacity(cache: dict) -> int:
    """Number of slots ``Tmax`` of a cache."""
    return int(cache["seg"].shape[1])


def reorder_cache(cache: dict, idx) -> dict:
    """Rows ``idx`` of a cache: row ``j`` of the result is row ``idx[j]`` of ``cache``.

    idx: int ``[B_out]`` (any length: duplicates a row, e.g. ``zeros(K)`` turns a one-row
    prefill into ``K`` particles, or resamples ``K`` particles). A gather on the batch axis
    of every entry (``sel`` and ``bos`` too, when present).
    """
    idx = jnp.asarray(idx, jnp.int32).reshape(-1)
    out = {name: jnp.take(cache[name], idx, axis=2, mode="clip")
           for name in ("k", "v", "sel") if name in cache}
    for name in ("seg", "pos", "next_pos", "cur_seg", "bos"):
        if name in cache:
            out[name] = jnp.take(cache[name], idx, axis=0, mode="clip")
    out["t"] = cache["t"]
    return out


# -------------------------------------------------------------------------- attention
def attend_cache(q: jax.Array, kc: jax.Array, vc: jax.Array, seg_q: jax.Array,
                 seg_k: jax.Array, pos_q: jax.Array, pos_k: jax.Array, *,
                 sel: jax.Array | None = None, maskable_k: jax.Array | None = None,
                 q_slot: jax.Array | None = None, prior: jax.Array | None = None):
    """Dense masked softmax of new queries over cached keys (same semantics as
    ``attention.attend_xla``).

    q: ``[B, Tq, H, hd]``; kc, vc: ``[B, H, Tk, hd]`` (the cache layout); seg_*/pos_*: int
    ``[B, T*]``. Logits and softmax in float32 (scale 1: the gain lives in q).
    sel: selective-attention switch ``float32[]`` or None (off); with it, maskable_k:
    ``bool[B, Tk]`` the slots selection may mask (not BOS, not padding or empty), q_slot: int
    ``[Tq]`` the slot of each query (its own key), prior: ``float32[B, Tk]`` the selection of
    the earlier queries (``cache["sel"]`` of this loop and block, zero at the new slots);
    ``attention.selective_logits`` subtracts ``sel * (prior + selection of the earlier queries
    of the chunk)`` from the logits.
    Returns ``(y float32[B, Tq, H, hd], s_sum float32[B, Tk] or None)``: ``s_sum`` is the
    selection ``sum_i S[i, j]`` of these queries (what the cache adds to ``prior``).
    """
    ft = hi(q.dtype)
    logits = jnp.einsum("bqhd,bhkd->bhqk", q, kc, preferred_element_type=ft)
    mask = (pos_k[:, None, :] <= pos_q[:, :, None]) & (seg_k[:, None, :] == seg_q[:, :, None])
    logits = jnp.where(mask[:, None], logits, NEG_INF)
    s_sum = None
    if sel is not None:
        logits, s_sum = selective_logits(logits, mask, maskable_k, q_slot, sel, prior)
    p = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bqhd", p.astype(vc.dtype), vc, preferred_element_type=ft), s_sum


# ---------------------------------------------------------------------------- extend
def _features(feats: dict, names: Sequence[str]) -> dict:
    return {k: jnp.asarray(feats[k]) for k in names if feats.get(k) is not None}


def extend(params: dict, consts: dict, cache: dict, feats: dict, cfg, temperature=1.0):
    """Append a chunk of ``Tq`` positions per row to the cache.

    feats: ``tokens int[B, Tq]`` and optionally ``seg [B, Tq]`` (default: each row's
    ``cur_seg``; negative = padding), ``pos [B, Tq]`` (default: ``next_pos + arange(Tq)``),
    ``reg [B, Tq, N_REG]``, ``clock [B, Tq, N_CLOCKS]``, ``cand_tok [B, Tq, K]``,
    ``cand_feat [B, Tq, K, N_CAND_FEAT]``. The slots ``[t, t + Tq)`` must fit in the cache
    (checked by the host loops; ``dynamic_update_slice`` would clamp the start index). The
    cache's defaults for the next call are set from each row's last valid slot (``next_pos`` its
    position + 1, ``cur_seg`` its segment), so a chunk may end in padding; a row whose chunk is all
    padding keeps them.
    temperature: sampling temperature ``T > 0`` of the scores (a float or a float32 scalar;
    it tempers the network's distribution only, see ``model.logits_from_hidden``).
    The chunk's tokens read their token gain rows (``model.gain_rows``, phase of loop ``r`` from
    ``model.loop_phases``) and token-neuron rows (``model.token_rows``) and every block gets the
    loop index (output gate); with ``sel_attn`` the slots of the chunk get ``bos`` and a zero
    selection before the loop, and every loop and block updates its ``sel`` row (module
    docstring).
    Returns ``(logits float32[B, V], hidden float32[B, d], cache)`` at the chunk's last
    position: ``logits`` are the scores of ``model.logits_from_hidden`` (``softmax`` = the
    output law ``p_T``, with the masses of that position's candidate row).
    """
    M.check_cfg(cfg)
    dt = act_dtype(cfg)
    tokens = jnp.asarray(feats["tokens"]).astype(jnp.int32)
    b, tq = tokens.shape
    t0 = jnp.asarray(cache["t"], jnp.int32)
    pos = feats.get("pos")
    if pos is None:
        pos = cache["next_pos"][:, None] + jnp.arange(tq, dtype=jnp.int32)[None]
    pos = jnp.asarray(pos).astype(jnp.int32)
    seg = feats.get("seg")
    seg = jnp.broadcast_to(cache["cur_seg"][:, None], (b, tq)) if seg is None else jnp.asarray(seg)
    seg = seg.astype(jnp.int32)
    valid = seg >= 0
    full = {"tokens": tokens, "pos": pos, "seg": seg, **_features(feats, FEATURE_KEYS)}

    ctx = M.make_ctx(consts, full, cfg)
    e = M.embed(params, consts, full, cfg)
    ve_all = M.value_embeddings(params, tokens, cfg)
    gains = M.gain_rows(params, consts, tokens, cfg)
    tok_rows = M.token_rows(params, consts, tokens, cfg)
    blocks = M.block_params(params, cfg)
    seg_k = jax.lax.dynamic_update_slice(cache["seg"], jnp.where(valid, seg, -1), (0, t0))
    pos_k = jax.lax.dynamic_update_slice(cache["pos"], pos, (0, t0))
    names = ("k", "v", "sel") if cfg.sel_attn else ("k", "v")
    carried = {name: cache[name] for name in names}          # updated by every loop and block
    if cfg.sel_attn:
        bos_k = jax.lax.dynamic_update_slice(cache["bos"], ctx.bos, (0, t0))
        maskable = ~bos_k & (seg_k >= 0)
        q_slot = t0 + jnp.arange(tq, dtype=jnp.int32)
        fresh = jnp.zeros(carried["sel"].shape[:3] + (tq,), carried["sel"].dtype)
        carried["sel"] = jax.lax.dynamic_update_slice(carried["sel"], fresh, (0, 0, 0, t0))
    lp = params["lanes"]
    xs: dict[str, Any] = {"alpha": lp["alpha"], "mix": lp["mix"], "beta": lp["beta"],
                          "g": params["loop"]["g"],
                          "r": jnp.arange(cfg.n_loops, dtype=jnp.int32)}
    if ve_all is not None:
        xs["ve"] = ve_all
    if gains is not None and cfg.gain_phases > 1:
        xs["phase"] = jnp.asarray(M.loop_phases(cfg))

    def step(carry, x):
        lanes, box = carry[0], dict(carry[1])
        r = x["r"]
        u = controller.combine(x["alpha"], lanes, dt)
        for i, pb in enumerate(blocks):
            def hook(q, k, v, i=i):
                zero = jnp.zeros((), jnp.int32)
                blk = jnp.asarray(i, jnp.int32)
                start = (r, blk, zero, zero, t0, zero)
                for name, new in (("k", k), ("v", v)):
                    upd = jnp.swapaxes(new, 1, 2).astype(box[name].dtype)[None, None]
                    box[name] = jax.lax.dynamic_update_slice(box[name], upd, start)
                kc = jax.lax.dynamic_index_in_dim(box["k"], r, 0, keepdims=False)[i]
                vc = jax.lax.dynamic_index_in_dim(box["v"], r, 0, keepdims=False)[i]
                if not cfg.sel_attn:
                    return attend_cache(q, kc, vc, ctx.seg, seg_k, ctx.pos, pos_k)[0], None
                prior = jax.lax.dynamic_index_in_dim(box["sel"], r, 0, keepdims=False)[i]
                y, s_sum = attend_cache(q, kc, vc, ctx.seg, seg_k, ctx.pos, pos_k, sel=ctx.sel,
                                        maskable_k=maskable, q_slot=q_slot, prior=prior)
                upd = (prior + s_sum.astype(prior.dtype))[None, None]
                box["sel"] = jax.lax.dynamic_update_slice(box["sel"], upd, (r, blk, zero, zero))
                return y, None
            u, _ = M.block(pb, u, ctx, x.get("ve") if i == 0 else None, cfg, attend_fn=hook,
                           gain=M.block_gain(gains, i, x.get("phase")), loop=r,
                           tok=None if tok_rows is None else tok_rows[i])
        s = rmsnorm(u, x["g"], out_dtype=dt)
        return (controller.update(x["mix"], x["beta"], lanes, s, dt), box), None

    lanes0 = jnp.broadcast_to(e[None], (cfg.n_lanes,) + e.shape)
    (lanes, carried), _ = jax.lax.scan(step, (lanes0, carried), xs)
    hidden = head.final_hidden(lanes, params["head"]["g"])[:, -1]           # [B, d] float32
    tail = {k: a[:, -1] for k, a in _features(full, ("cand_tok", "cand_feat")).items()}
    logits = M.logits_from_hidden(params, consts, hidden, tail, cfg, temperature)
    # the defaults of the next call follow each row's last valid slot (a chunk may end in padding)
    slot = jnp.arange(tq, dtype=jnp.int32)
    last = jnp.max(jnp.where(valid, slot[None], -1), axis=1)                 # -1: all padding
    at = slot[None] == last[:, None]
    new_cache = {**carried, "seg": seg_k, "pos": pos_k, "t": t0 + tq,
                 "next_pos": jnp.where(last >= 0, jnp.sum(jnp.where(at, pos, 0), axis=1) + 1,
                                       cache["next_pos"]),
                 "cur_seg": jnp.where(last >= 0, jnp.sum(jnp.where(at, seg, 0), axis=1),
                                      cache["cur_seg"])}
    if cfg.sel_attn:
        new_cache["bos"] = bos_k
    return logits, hidden, new_cache


def prefill(params: dict, consts: dict, cache: dict, feats: dict, cfg, temperature=1.0):
    """Process a prompt chunk ``feats["tokens"] int[B, Tq]`` (see :func:`extend`; prompts may
    be fed in several chunks). Returns ``(logits float32[B, V], hidden float32[B, d], cache)``
    of the chunk's last position (scores at ``temperature``, see :func:`extend`)."""
    return extend(params, consts, cache, feats, cfg, temperature)


def decode_step(params: dict, consts: dict, cache: dict, tok, feats: dict | None, cfg,
                temperature=1.0):
    """One token per row.

    tok: int ``[B]``; feats: rows ``reg [B, N_REG]``, ``clock [B, N_CLOCKS]``,
    ``cand_tok [B, K]``, ``cand_feat [B, K, N_CAND_FEAT]`` (each optional) and optionally
    ``seg [B]`` / ``pos [B]`` (defaults: ``cur_seg`` / ``next_pos`` of the cache);
    temperature: as in :func:`extend`.
    Returns ``(logits float32[B, V], hidden float32[B, d], cache)``.
    """
    tok = jnp.asarray(tok).astype(jnp.int32).reshape(-1)
    f = {"tokens": tok[:, None]}
    if feats is not None:
        for name, a in _features(feats, FEATURE_KEYS + _POSITION_KEYS).items():
            f[name] = a[:, None]
    return extend(params, consts, cache, f, cfg, temperature)


class Decoder(NamedTuple):
    """Jit-compiled entry points for one model configuration (SPEC 4.4 signatures). The
    temperature (default 1) is always passed to the compiled program as a float32 scalar, so
    one compilation per input shape serves every temperature."""
    prefill: Callable      # (params, consts, cache, feats, temperature=1) -> (logits, hidden, cache)
    decode_step: Callable  # (params, consts, cache, tok, feats, temperature=1) -> (logits, hidden, cache)
    reorder: Callable      # (cache, idx) -> cache


def as_temperature(temperature) -> np.ndarray:
    """``float32[]`` sampling temperature for the compiled programs (raises ``ValueError``
    unless it is a positive finite number)."""
    t = np.asarray(temperature, np.float32).reshape(())
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
    return t


@functools.lru_cache(maxsize=None)
def jitted(cfg, donate: bool = True) -> Decoder:
    """Compiled :func:`prefill`, :func:`decode_step` and :func:`reorder_cache` for ``cfg``
    (one compilation per input shape, whatever the temperature; cached per ``(cfg, donate)``,
    so every caller in a process shares the compiled programs). With ``donate`` the input
    cache is consumed."""
    dn = (2,) if donate else ()

    def _prefill(params, consts, cache, feats, temperature):
        return prefill(params, consts, cache, feats, cfg, temperature)

    def _decode_step(params, consts, cache, tok, feats, temperature):
        return decode_step(params, consts, cache, tok, feats, cfg, temperature)

    pre = jax.jit(_prefill, donate_argnums=dn)
    step = jax.jit(_decode_step, donate_argnums=dn)

    def prefill_fn(params, consts, cache, feats, temperature=1.0):
        return pre(params, consts, cache, feats, as_temperature(temperature))

    def decode_step_fn(params, consts, cache, tok, feats, temperature=1.0):
        return step(params, consts, cache, tok, feats, as_temperature(temperature))

    prefill_fn.compiled = pre                    # the jitted programs (cache sizes, lowering)
    decode_step_fn.compiled = step
    return Decoder(prefill=prefill_fn, decode_step=decode_step_fn, reorder=jax.jit(reorder_cache))


# ------------------------------------------------------------------ feature providers
def empty_rows(n: int) -> dict[str, np.ndarray]:
    """``n`` all-zero feature rows (registers 0, clocks 0, an empty candidate tape)."""
    return {"reg": np.zeros((n, N_REG), np.int16), "clock": np.zeros((n, N_CLOCKS), np.int32),
            "cand_tok": np.full((n, CAND_SLOTS), -1, np.int32),
            "cand_feat": np.zeros((n, CAND_SLOTS, N_CAND_FEAT), np.int8)}


class ZeroFeatures:
    """Feature-provider stand-in: every row is :func:`empty_rows` (tests, feature-less runs)."""

    def step(self, token: int, new_segment: bool = False) -> dict[str, np.ndarray]:
        del token, new_segment
        return {k: a[0] for k, a in empty_rows(1).items()}

    def fork(self) -> "ZeroFeatures":
        return ZeroFeatures()


def fork_provider(provider):
    """``provider.fork()`` when available, else a deep copy."""
    return provider.fork() if hasattr(provider, "fork") else copy.deepcopy(provider)


def _row(provider, token: int) -> dict[str, np.ndarray]:
    out = provider.step(int(token))
    return {k: np.asarray(out[k]) for k in FEATURE_KEYS}


def _stack(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        return empty_rows(0)
    return {k: np.stack([r[k] for r in rows]) for k in FEATURE_KEYS}


def prompt_segments(prompt_ids: Sequence[int], bos_id: int) -> np.ndarray:
    """Segment ids of a prompt: a new segment starts at every ``BOS`` after position 0."""
    ids = np.asarray(prompt_ids, np.int64).reshape(-1)
    starts = (ids == bos_id).astype(np.int32)
    if starts.size:
        starts[0] = 0
    return np.cumsum(starts).astype(np.int32)


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _t_args(temperature) -> tuple:
    """Extra positional argument of a decoder call: none at temperature 1 (any decoder with
    the plain SPEC 4.4 signature works), else the temperature."""
    return () if float(temperature) == 1.0 else (temperature,)


def _layout(n_prompt: int, max_new: int, chunk: int, tmax: int | None) -> tuple[int, int]:
    n_pad = (-n_prompt) % chunk
    need = n_pad + n_prompt + max_new
    tmax = _round_up(max(need, 1), TMAX_ROUND) if tmax is None else int(tmax)
    if need > tmax:
        raise ValueError(f"cache of {tmax} slots cannot hold {need} (padding + prompt + new)")
    return n_pad, tmax


def _prefill_prompt(dec: Decoder, params, consts, cfg, prompt_ids, rows, tmax, chunk, n_pad,
                    temperature=1.0):
    """Left-pad the prompt to a multiple of ``chunk`` (padding: ``seg = -1``) and prefill a
    one-row cache chunk by chunk. Returns ``(logits [1, V], hidden [1, d], cache)`` (scores at
    ``temperature``)."""
    n = len(prompt_ids)
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    tokens = np.concatenate([np.full(n_pad, pad_id, np.int32), np.asarray(prompt_ids, np.int32)])
    seg = np.concatenate([np.full(n_pad, -1, np.int32), prompt_segments(prompt_ids, M.bos_id(cfg))])
    pos = np.concatenate([np.zeros(n_pad, np.int32), np.arange(n, dtype=np.int32)])
    feat = {k: np.concatenate([empty_rows(n_pad)[k], rows[k]]) for k in FEATURE_KEYS}
    cache = init_cache(cfg, 1, tmax)
    logits = hidden = None
    for lo in range(0, n_pad + n, chunk):
        sl = slice(lo, lo + chunk)
        f = {"tokens": tokens[None, sl], "seg": seg[None, sl], "pos": pos[None, sl],
             **{k: a[None, sl] for k, a in feat.items()}}
        logits, hidden, cache = dec.prefill(params, consts, cache, f, *_t_args(temperature))
    return logits, hidden, cache


def greedy(logits: np.ndarray, step: int) -> int:
    """Sampling callback: the arg-max token."""
    del step
    return int(np.argmax(logits))


@dataclass
class Generation:
    """Result of :func:`generate`: new tokens (the stop token included when hit), whether a
    stop token ended the stream, and, with ``record``, the logits ``[n, V]`` and hidden
    states ``[n, d]`` from which each new token was sampled."""
    tokens: list[int]
    stopped: bool
    logits: np.ndarray | None = None
    hidden: np.ndarray | None = None


def generate(params: dict, consts: dict, cfg, prompt_ids: Sequence[int], provider=None,
             sample_fn: Callable[[np.ndarray, int], int] = greedy, *, max_new_tokens: int = 64,
             stop_ids: Sequence[int] = (END,), tmax: int | None = None,
             chunk: int = DEFAULT_CHUNK, record: bool = False,
             decoder: Decoder | None = None, temperature: float = 1.0) -> Generation:
    """Host-side generation loop for one stream.

    prompt_ids: the prompt (normally ``BOS ... ASSISTANT``); provider: feature provider
    (default :class:`ZeroFeatures`), stepped through the prompt and every new token;
    sample_fn: ``(logits float32[V], step) -> token`` where ``softmax(logits)`` is the output
    law at ``temperature`` (the network tempered, the tape untouched: sample from
    ``softmax(logits)`` itself); tmax: cache slots (default: enough for the padded prompt and
    ``max_new_tokens``, rounded up to ``TMAX_ROUND``). As in training rows, every ``BOS`` (in
    the prompt or generated) starts a new segment.
    """
    if not len(prompt_ids):
        raise ValueError("the prompt must hold at least one token (normally BOS)")
    provider = ZeroFeatures() if provider is None else provider
    dec = decoder if decoder is not None else jitted(cfg)
    n_pad, tmax = _layout(len(prompt_ids), max_new_tokens, chunk, tmax)
    rows = _stack([_row(provider, tok) for tok in prompt_ids])
    logits, hidden, cache = _prefill_prompt(dec, params, consts, cfg, prompt_ids, rows, tmax,
                                            chunk, n_pad, temperature)
    bos = M.bos_id(cfg)
    seg = int(prompt_segments(prompt_ids, bos)[-1])
    stops = set(int(s) for s in stop_ids)
    out: list[int] = []
    rec_z, rec_h = [], []
    stopped = False
    for step in range(max_new_tokens):
        z = np.asarray(logits[0])
        if record:
            rec_z.append(z)
            rec_h.append(np.asarray(hidden[0]))
        tok = int(sample_fn(z, step))
        out.append(tok)
        if tok in stops:
            stopped = True
            break
        if step == max_new_tokens - 1:
            break
        seg += int(tok == bos)
        row = {k: a[None] for k, a in _row(provider, tok).items()}
        row["seg"] = np.asarray([seg], np.int32)
        logits, hidden, cache = dec.decode_step(params, consts, cache, np.asarray([tok], np.int32),
                                                row, *_t_args(temperature))
    return Generation(out, stopped, np.stack(rec_z) if record and rec_z else None,
                      np.stack(rec_h) if record and rec_h else None)


@dataclass
class ParticleGeneration:
    """Result of :func:`generate_particles`: per particle its new tokens and whether it hit a
    stop token; ``ancestors[s]`` is the ancestor index vector applied at step ``s`` (identity
    when the callback did not resample)."""
    tokens: list[list[int]]
    done: np.ndarray
    ancestors: list[np.ndarray]


def _parse_sample(out, k: int) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(out, tuple):
        tokens, anc = out
    else:
        tokens, anc = out, None
    tokens = np.asarray(tokens, np.int64).reshape(-1)
    if tokens.shape != (k,):
        raise ValueError(f"sample_fn returned {tokens.shape[0]} tokens for {k} particles")
    if anc is not None:
        anc = np.asarray(anc, np.int64).reshape(-1)
        if anc.shape != (k,) or anc.min() < 0 or anc.max() >= k:
            raise ValueError("ancestor indices must be a permutation-with-repeats of range(K)")
    return tokens, anc


def generate_particles(params: dict, consts: dict, cfg, prompt_ids: Sequence[int], provider,
                       sample_fn: Callable, n_particles: int, *, max_new_tokens: int = 64,
                       stop_ids: Sequence[int] = (END,), tmax: int | None = None,
                       chunk: int = DEFAULT_CHUNK, decoder: Decoder | None = None,
                       temperature: float = 1.0) -> ParticleGeneration:
    """``K`` batched particles of one prompt (each row an independent continuation).

    The prompt is prefilled once in a one-row cache that is then duplicated into ``K`` rows;
    the provider is forked once per particle. sample_fn: ``(logits float32[K, V], step,
    done bool[K]) -> tokens int[K]`` or ``(tokens, ancestors int[K])`` (``softmax(logits)``
    is the output law at ``temperature``, as in :func:`generate`). With ancestors,
    particle ``j`` continues the history (cache row, provider, tokens, done flag) of old
    particle ``ancestors[j]`` and then appends ``tokens[j]``. Finished particles keep their
    tokens; their later rows are fed as padding. Every ``BOS`` starts a new segment.
    """
    k = int(n_particles)
    provider = ZeroFeatures() if provider is None else provider
    dec = decoder if decoder is not None else jitted(cfg)
    n_pad, tmax = _layout(len(prompt_ids), max_new_tokens, chunk, tmax)
    rows = _stack([_row(provider, tok) for tok in prompt_ids])
    logits, _, cache = _prefill_prompt(dec, params, consts, cfg, prompt_ids, rows, tmax, chunk,
                                       n_pad, temperature)
    cache = dec.reorder(cache, np.zeros(k, np.int32))
    logits = np.repeat(np.asarray(logits), k, axis=0)
    bos = M.bos_id(cfg)
    seg = np.full(k, prompt_segments(prompt_ids, bos)[-1], np.int64)
    providers = [fork_provider(provider) for _ in range(k)]
    histories: list[list[int]] = [[] for _ in range(k)]
    done = np.zeros(k, bool)
    stops = set(int(s) for s in stop_ids)
    pad_id = special_id("<|pad|>", cfg.vocab_size)
    ancestry: list[np.ndarray] = []
    for step in range(max_new_tokens):
        tokens, anc = _parse_sample(sample_fn(np.asarray(logits), step, done.copy()), k)
        if anc is not None and not np.array_equal(anc, np.arange(k)):
            cache = dec.reorder(cache, anc.astype(np.int32))
            providers = [fork_provider(providers[a]) for a in anc]
            histories = [list(histories[a]) for a in anc]
            done, seg = done[anc], seg[anc]
        ancestry.append(np.arange(k) if anc is None else anc)
        for j in range(k):
            if not done[j]:
                histories[j].append(int(tokens[j]))
                done[j] = int(tokens[j]) in stops
        if done.all() or step == max_new_tokens - 1:
            break
        feed = np.where(done, pad_id, tokens).astype(np.int32)
        seg = seg + ((tokens == bos) & ~done)
        rows_k = [empty_rows(1) if done[j] else {n: a[None] for n, a in
                                                 _row(providers[j], int(tokens[j])).items()}
                  for j in range(k)]
        f = {n: np.concatenate([r[n] for r in rows_k]) for n in FEATURE_KEYS}
        f["seg"] = np.where(done, -1, seg).astype(np.int32)
        logits, _, cache = dec.decode_step(params, consts, cache, feed, f, *_t_args(temperature))
    return ParticleGeneration(histories, done, ancestry)


__all__ = ["init_cache", "cache_capacity", "reorder_cache", "attend_cache", "extend", "prefill",
           "decode_step", "Decoder", "jitted", "as_temperature", "empty_rows", "ZeroFeatures",
           "fork_provider", "prompt_segments", "greedy", "Generation", "generate",
           "ParticleGeneration", "generate_particles", "FEATURE_KEYS", "DEFAULT_CHUNK"]
