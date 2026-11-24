"""BES looped transformer: parameters, constants, forward pass and loss (SPEC section 4).

    e      = rmsnorm(E_tok[x] + sum_f E_reg_f[reg_f] + W_cin @ sum_k mask_k (E_ctok[cand_k] + sum_f E_cfeat_f[feat_kf]))
    lanes  = (e, ..., e)
    for r in 0..n_loops-1:                                   # lax.scan, optional jax.checkpoint
        u     = sum_n alpha[r, n] lanes[n]
        u     = block_0(u; ve = E_ve[x, r]);  u = block_1(u); ...
        s     = rmsnorm(u) * g_loop[r]
        lanes = mix[r] @ lanes + beta[r][:, None] * s
    h      = rmsnorm(sum_n lanes[n]) * g_head
    q      = softmax(softcap * tanh(((h @ P) @ E_out.T) / softcap))
    a_k    = sum_f beta_f[feat_kf] + <h A_t, c_k> - mass_offset,      c_k = sum_f E_tfeat_f[feat_kf]
    m_k    = exp(min(a_k, mass_log_cap)) * mask_k * src_scale[src_k]
    p(v)   = (q(v) + sum_k m_k [cand_k = v]) / (1 + sum_k m_k)          # bes.model.predictive

Core design and initialisation ported from kschwethelm/looped-lm-scaling ``nanochat/gpt.py``
("hyperconnect" mode) and the value-embedding gate from karpathy/nanochat ``nanochat/gpt.py``
(both MIT licence). Every new feature path starts from zero-initialised tables or scalars, so
at initialisation the register and candidate-input paths contribute exactly zero and every
active candidate slot carries the small mass ``exp(-mass_offset)``; the sink and confidence
paths are active from the start (sink logit 0, prior mean 0) and are removed exactly by their
switches. ``make_consts`` holds the runtime switches so one compiled graph covers every rung
of the fallback ladder.

Input dictionaries (``feats``): ``tokens`` int ``[B, T]`` (required) and optionally ``seg``
``[B, T]`` (default 0), ``pos`` ``[B, T]`` (default ``arange(T)``), ``reg`` ``[B, T, N_REG]``,
``clock`` ``[B, T, N_CLOCKS]``, ``cand_tok`` ``[B, T, K]`` and ``cand_feat``
``[B, T, K, N_CAND_FEAT]``. A missing optional feature behaves like a disabled path.

Token gain pages (``gain_ffn``, ``gain_o``, ``gain_v``; zero-initialised tables ``gain/*`` of
shape ``[V, P, n_blocks, width]``, ``P = gain_phases``): the input token of a position reads one
row per table (:func:`gain_rows`, like ``emb/ve``); in loop ``r`` (phase ``p = r P // n_loops``)
block ``i`` multiplies the value channels of the position (``x @ W_v``), its attention output
channels before ``W_o`` and its first ``gain_ffn`` FFN neurons (after ``relu(.)^2``) by
``1 + enable["gain"] * row[p, i]``, in ``hi(act_dtype)``. Selective attention (``sel_attn``):
``bes.model.attention.selective_logits`` with the switch ``enable["sel"]``.

Head-wise attention output gate (``attn_gate``; zero-initialised ``core/attn/gate_w
[n_blocks, d, H]`` and ``core/attn/gate_b [n_loops, n_blocks, H]``): in loop ``r`` block ``i``
multiplies the output of head ``h`` by ``1 + enable["gate"] (2 sigmoid(x W_g[i][:, h] / d +
b_g[r, i, h]) - 1)`` (``bes.model.attention.head_gate``; the loop index reaches the block through
the scan's ``xs``). Token neurons (``tok_neurons = rho``, ``tok_dim = dk``): every vocabulary id
owns ``rho`` private ReLU^2 neurons per block in a ``dk``-dimensional space shared by all tokens
(dense ``page/pin [n_blocks, d, dk]``, ``page/pout [n_blocks, dk, d]``); the keys ``page/key`` and
the zero-initialised values ``page/val`` (``[V, n_blocks, rho, dk]``) of the input token of a
position are gathered once per forward (:func:`token_rows`) and block ``i`` adds
``enable["page"] * layers.token_neurons(x, ...)`` to its feed-forward output.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_FEATS, REG_FIELDS, special_id
from bes.model import (attention, controller, head, oracle_io, predictive, rope_clocks, sinks,
                       tables)
from bes.model import layers as _layers
from bes.model.critic import critic  # noqa: F401  (re-exported for other tracks)
from bes.model.layers import (F32, PhaseGain, act_dtype, hi, leaf_key, mlp, normal_init, rmsnorm,
                              token_neurons, uniform_init)

N_KINDS = dict(CAND_FEATS)["kind"]
N_SRC = dict(CAND_FEATS)["src"]
ROLE_FIELD = [name for name, _ in REG_FIELDS].index("role")
N_ROLES = dict(REG_FIELDS)["role"]
#: names of the scalar runtime switches in ``consts["enable"]`` (plus the vectors ``kind``
#: and ``src``)
SWITCHES = ("reg", "cand_in", "cand_out", "conf", "sink", "clock", "gain", "sel", "gate", "page")
#: gain tables in the order of :func:`gain_widths`
GAIN_TABLES = ("ffn", "o", "v")
#: Execution-level speed switch (not configuration; default off; ``rows`` in
#: ``bes.model.fast``), for the token-neuron rows of :func:`token_rows`:
#: (a) the key preparation (gather, RMS norm, cast) is not rematerialised in the backward pass
#: under ``cfg.remat``: the forward keeps the norm's residuals (the float32 gathered keys and
#: their reciprocal RMS) instead of gathering and normalising again;
#: (b) :func:`forward` closes the loop scan over lane-dense views of the rows (``[..., m, n] ->
#: [..., m * n]`` when ``n`` is not a multiple of 128 lanes and ``m * n`` is, :func:`_lane_dense`)
#: and each loop step reshapes them back, so the cotangents the scan sums over the loops are
#: carried in the unpadded layout the gather's transpose reads (an ``[8, 192]`` block of a token
#: sits in padded ``(16, 256)`` bfloat16 tiles). Same operations on the same values: residuals
#: kept instead of recomputed and reshapes only, so every value is the same bit for bit
#: (tests/model/test_fast_rows.py). TPU v6e compile of configs/v9.py (take, 4 x 1024 pieces,
#: with ``flat``): -3.57 M estimated cycles per piece, about -119 ms per step (wf v9_rows report).
FAST_ROWS = False


def bos_id(cfg) -> int:
    """Id of the BOS token for the configured vocabulary size."""
    return special_id("<|bos|>", cfg.vocab_size)


def check_cfg(cfg) -> None:
    """Validate the shape constraints of a ``ModelCfg`` (raises ``ValueError``)."""
    rope_clocks.plane_layout(cfg)
    if cfg.bias_dims < 1:
        raise ValueError("bias_dims must be >= 1 (reserved dims carry the per-key biases)")
    if cfg.ve_dim < 0 or cfg.ve_dim % cfg.head_dim or cfg.ve_dim > cfg.n_heads * cfg.head_dim:
        raise ValueError(f"ve_dim={cfg.ve_dim} must be a multiple of head_dim and <= n_heads*head_dim")
    if cfg.n_lanes < 1 or cfg.n_loops < 1 or cfg.n_blocks < 1:
        raise ValueError("n_lanes, n_loops and n_blocks must be >= 1")
    if cfg.attn_impl not in ("xla", "splash"):
        raise ValueError(f"unknown attn_impl {cfg.attn_impl!r}")
    if cfg.table_grad not in ("onehot", "take", "sorted"):
        raise ValueError(f"unknown table_grad {cfg.table_grad!r}")
    if cfg.sorted_chunk < 1:
        raise ValueError(f"sorted_chunk={cfg.sorted_chunk} must be >= 1")
    if not 0 <= cfg.gain_ffn <= cfg.d_ffn:
        raise ValueError(f"gain_ffn={cfg.gain_ffn} must be in [0, d_ffn={cfg.d_ffn}]")
    if not 1 <= cfg.gain_phases <= cfg.n_loops:
        raise ValueError(f"gain_phases={cfg.gain_phases} must be in [1, n_loops={cfg.n_loops}]")
    if cfg.sel_attn and cfg.attn_impl != "xla":
        raise ValueError("sel_attn needs the dense attention path (attn_impl='xla')")
    if cfg.tok_neurons < 0 or cfg.tok_dim < 1:
        raise ValueError(f"tok_neurons={cfg.tok_neurons} must be >= 0 and tok_dim={cfg.tok_dim} >= 1")
    if cfg.remat_keep not in controller.REMAT_POLICIES:
        raise ValueError(f"remat_keep={cfg.remat_keep!r} must be one of {sorted(controller.REMAT_POLICIES)}")


# ------------------------------------------------------------------------- parameters
def init_params(cfg, key: jax.Array) -> dict:
    """Initial parameter tree (float32) with the stable names of SPEC 4.3.

    Reference initialisation for the core: token embedding N(0, 1), unembedding N(0, 0.001),
    every matrix uniform with std ``1/sqrt(fan_in)`` (all non-zero), RMSNorm gains 1, cyclic
    lanes, value embeddings like ``W_v``, value gate uniform(0, 0.02), query gain
    ``sqrt(dc)`` (the reference scale ``q.k / sqrt(dims)`` of RMS-normalised vectors).
    Feature tables (registers, candidate tables, trust features and trust biases), the token
    gain tables ``gain/{ffn,o,v}`` ``[V, gain_phases, n_blocks, width]`` (only those whose flag is
    on, :func:`gain_widths`) and the sink scalars are zero; the critic output weights are zero.
    With ``attn_gate`` the gate parameters ``core/attn/gate_w [n_blocks, d, H]`` and
    ``core/attn/gate_b [n_loops, n_blocks, H]`` are zero (gate exactly 1). With ``tok_neurons``
    the projections ``page/pin [n_blocks, d, dk]`` and ``page/pout [n_blocks, dk, d]`` are
    uniform (fan-in ``d`` and ``dk``; non-zero, as MuonH needs), the keys ``page/key
    [V, n_blocks, rho, dk]`` N(0, 1) (used RMS-normalised) and the values ``page/val`` (same
    shape) zero, so the token neurons add exactly zero at initialisation.
    """
    check_cfg(cfg)
    v, d, nh, hd = cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.head_dim
    nb, nl, cd = cfg.n_blocks, cfg.n_loops, cfg.cand_dim
    dc = hd - cfg.bias_dims

    def uni(path: str, shape: tuple[int, ...], fan_in: int) -> jax.Array:
        return uniform_init(leaf_key(key, path), shape, fan_in)

    def zeros(shape: tuple[int, ...]) -> jax.Array:
        return jnp.zeros(shape, F32)

    emb: dict[str, Any] = {"tok": normal_init(leaf_key(key, "emb/tok"), (v, d), 1.0)}
    if cfg.ve_dim:
        emb["ve"] = uni("emb/ve", (v, nl, cfg.ve_dim), d)
    if cfg.use_reg:
        emb["reg"] = {name: zeros((card, d)) for name, card in REG_FIELDS}
    params: dict[str, Any] = {"emb": emb}
    if cfg.use_cand_in:
        params["cand"] = {
            "in_tok": zeros((v, cd)),
            "in_feat": {name: zeros((card, cd)) for name, card in CAND_FEATS},
            "in_proj": uni("cand/in_proj", (cd, d), cd),
        }
    attn: dict[str, Any] = {
        "wq": uni("core/attn/wq", (nb, d, nh * dc), d),
        "wk": uni("core/attn/wk", (nb, d, nh * dc), d),
        "wv": uni("core/attn/wv", (nb, d, nh * hd), d),
        "wo": uni("core/attn/wo", (nb, nh * hd, d), nh * hd),
        "q_gain": jnp.full((nb, nh), math.sqrt(dc), F32),
    }
    if cfg.use_sink:
        attn["sink_logit"] = zeros((nb, nh))
        attn["sink_value"] = zeros((nb, nh, hd - 1))
    if cfg.ve_dim:
        attn["ve_gate"] = jax.random.uniform(
            leaf_key(key, "core/attn/ve_gate"),
            (min(attention.VE_GATE_CHANNELS, d), cfg.ve_dim // hd), F32, 0.0, 0.02)
    if cfg.attn_gate:
        attn["gate_w"] = zeros((nb, d, nh))
        attn["gate_b"] = zeros((nl, nb, nh))
    params["core"] = {
        "attn": attn,
        "mlp": {"w_up": uni("core/mlp/w_up", (nb, d, cfg.d_ffn), d),
                "w_down": uni("core/mlp/w_down", (nb, cfg.d_ffn, d), cfg.d_ffn)},
        "norm": {"g1": jnp.ones((nb, d), F32), "g2": jnp.ones((nb, d), F32)},
    }
    params["loop"] = {"g": jnp.ones((nl, d), F32)}
    params["lanes"] = {k: jnp.asarray(a) for k, a in controller.init_lanes(nl, cfg.n_lanes).items()}
    params["head"] = {
        "g": jnp.ones((d,), F32),
        "proj": uni("head/proj", (d, cfg.head_rank), d),
        "out": normal_init(leaf_key(key, "head/out"), (v, cfg.head_rank), 0.001),
    }
    if cfg.use_cand_out:
        params["trust"] = {"proj": uni("trust/proj", (d, cd), d),
                           "feat": {name: zeros((card, cd)) for name, card in CAND_FEATS},
                           "bias": {name: zeros((card,)) for name, card in CAND_FEATS}}
    params["critic"] = {"w1": uni("critic/w1", (d, cfg.critic_hidden), d),
                        "w2": zeros((cfg.critic_hidden,))}
    widths = gain_widths(cfg)
    if widths:
        params["gain"] = {name: zeros((v, cfg.gain_phases, nb, w)) for name, w in widths.items()}
    if cfg.tok_neurons:
        rho, dk = cfg.tok_neurons, cfg.tok_dim
        params["page"] = {"pin": uni("page/pin", (nb, d, dk), d),
                          "pout": uni("page/pout", (nb, dk, d), dk),
                          "key": normal_init(leaf_key(key, "page/key"), (v, nb, rho, dk), 1.0),
                          "val": zeros((v, nb, rho, dk))}
    return params


def gain_widths(cfg) -> dict[str, int]:
    """``{table: width per block}`` of the gain tables whose flag is on, in :data:`GAIN_TABLES`
    order: ``ffn`` (``gain_ffn``), ``o`` and ``v`` (``n_heads * head_dim``)."""
    hw = cfg.n_heads * cfg.head_dim
    flags = {"ffn": (cfg.gain_ffn > 0, cfg.gain_ffn), "o": (cfg.gain_o, hw), "v": (cfg.gain_v, hw)}
    return {name: flags[name][1] for name in GAIN_TABLES if flags[name][0]}


def make_consts(cfg, **switches) -> dict:
    """Non-trainable inputs: rotary frequencies and the runtime feature switches.

    Keyword switches (0/1 floats or bools): ``reg``, ``cand_in``, ``cand_out``, ``conf``,
    ``sink``, ``clock`` (multiplies the clock values; 0 leaves the clock planes unrotated),
    ``gain`` (multiplies the token gain rows: 0 gives gains of exactly 1), ``sel`` (multiplies
    the selective-attention offsets), ``gate`` (multiplies the deviation of the attention output
    gate from 1), ``page`` (multiplies the output of the token neurons);
    ``kind`` (a scalar or a sequence of ``N_KINDS`` per candidate kind) or ``kind_<i>``.
    ``src`` (a scalar or a sequence of ``N_SRC``) or ``src_<i>``: non-negative multipliers of
    the output-law masses of the candidates whose ``src`` feature is ``i`` (default 1; e.g.
    ``src_5=0.25`` down-weights evidence from the text being generated).
    Defaults: on when the static ``cfg.use_*`` flag (``clock_planes > 0`` for ``clock``, any
    gain table for ``gain``, ``sel_attn`` for ``sel``, ``attn_gate`` for ``gate``,
    ``tok_neurons > 0`` for ``page``) is on. A switch never re-enables a path whose static flag
    is off.
    Returns ``{"rope_freq": float32[1 + N_CLOCKS, P], "enable": {name: float32 scalar,
    "kind": float32[N_KINDS], "src": float32[N_SRC]}}``.
    """
    values = {"reg": cfg.use_reg, "cand_in": cfg.use_cand_in, "cand_out": cfg.use_cand_out,
              "conf": cfg.use_conf, "sink": cfg.use_sink, "clock": cfg.clock_planes > 0,
              "gain": bool(gain_widths(cfg)), "sel": cfg.sel_attn, "gate": cfg.attn_gate,
              "page": cfg.tok_neurons > 0}
    vectors = {"kind": np.ones(N_KINDS, np.float32), "src": np.ones(N_SRC, np.float32)}
    for name, value in switches.items():
        stem, _, idx = name.partition("_")
        if name in values:
            values[name] = value
        elif name in vectors:
            vectors[name] = np.broadcast_to(np.asarray(value, np.float32),
                                            vectors[name].shape).copy()
        elif stem in vectors and idx.isdigit() and int(idx) < vectors[stem].size:
            vectors[stem][int(idx)] = float(value)
        else:
            raise KeyError(f"unknown switch {name!r}")
    if not np.all(vectors["src"] >= 0):
        raise ValueError(f"src multipliers must be non-negative, got {vectors['src'].tolist()}")
    enable = {name: jnp.asarray(float(values[name]), F32) for name in SWITCHES}
    enable.update({name: jnp.asarray(v) for name, v in vectors.items()})
    freq = rope_clocks.freq_table(cfg, np.dtype(hi(act_dtype(cfg))))
    return {"rope_freq": jnp.asarray(freq), "enable": enable}


def switches_off() -> dict:
    """Keyword arguments for :func:`make_consts` that disable every feature path."""
    return {**{name: 0.0 for name in SWITCHES}, "kind": 0.0}


# ---------------------------------------------------------------------------- forward
def _feat(feats: dict, name: str):
    value = feats.get(name)
    return None if value is None else jnp.asarray(value)


def make_ctx(consts: dict, feats: dict, cfg) -> attention.Ctx:
    """Rotary tables, sink indicator, confidence switch, segments and positions; with
    ``sel_attn`` also the BOS positions (``tokens == bos_id``, whatever the sink switch) and
    the selective-attention switch; with ``attn_gate`` the switch of the output gate; with
    ``attn_impl == "splash"`` the rotary tables of the head-major path
    (``attention.rotary_major``)."""
    tokens = jnp.asarray(feats["tokens"])
    b, t = tokens.shape
    pos = _feat(feats, "pos")
    pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t)) if pos is None else pos
    seg = _feat(feats, "seg")
    seg = jnp.zeros((b, t), jnp.int32) if seg is None else seg
    en = consts["enable"]
    clock = _feat(feats, "clock") if cfg.clock_planes else None
    cos, sin = rope_clocks.cos_sin(rope_clocks.angles(pos, clock, consts["rope_freq"], en["clock"]))
    sink = sinks.sink_indicator(tokens, bos_id(cfg), en["sink"]) if cfg.use_sink else None
    conf = en["conf"] if cfg.use_conf else None
    bos = tokens == bos_id(cfg) if cfg.sel_attn else None
    sel = jnp.asarray(en["sel"], F32) if cfg.sel_attn else None
    gate = jnp.asarray(en["gate"], F32) if cfg.attn_gate else None
    rot_c, rot_s = (attention.rotary_major(cos, sin, cfg.bias_dims) if cfg.attn_impl == "splash"
                    else (None, None))
    return attention.Ctx(cos, sin, sink, conf, seg.astype(jnp.int32), pos.astype(jnp.int32),
                         bos, sel, gate, rot_c, rot_s)


def _cand(feats: dict):
    cand_tok = _feat(feats, "cand_tok")
    if cand_tok is None:
        return None, None
    cand_feat = _feat(feats, "cand_feat")
    if cand_feat is None:
        cand_feat = jnp.zeros(cand_tok.shape + (len(CAND_FEATS),), jnp.int32)
    return cand_tok, cand_feat


def embed(params: dict, consts: dict, feats: dict, cfg) -> jax.Array:
    """Normalised input embedding ``e [B, T, d]`` in the activation dtype."""
    dt = act_dtype(cfg)
    en = consts["enable"]
    tokens = jnp.asarray(feats["tokens"])
    pre = tables.big_lookup(params["emb"]["tok"], tokens, mode=cfg.table_grad, dtype=dt,
                            chunk=cfg.sorted_chunk)
    reg = _feat(feats, "reg")
    if cfg.use_reg and reg is not None:
        pre = pre + tables.reg_embed(params["emb"]["reg"], reg, en["reg"], dt)
    cand_tok, cand_feat = _cand(feats)
    if cfg.use_cand_in and cand_tok is not None:
        m = oracle_io.cand_mask(cand_tok, cand_feat, en["kind"], en["cand_in"])
        pre = pre + oracle_io.cand_input(params["cand"], cand_tok, cand_feat, m, cfg)
    return rmsnorm(pre, out_dtype=dt)


def value_embeddings(params: dict, tokens: jax.Array, cfg) -> jax.Array | None:
    """Per-loop value embeddings ``E_ve[x, r]`` as ``[n_loops, B, T, ve_dim]`` (activation dtype)."""
    if not cfg.ve_dim:
        return None
    table = params["emb"]["ve"]
    nv, nl, w = table.shape
    rows = tables.big_lookup(table.reshape(nv, nl * w), tokens, mode=cfg.table_grad,
                             dtype=act_dtype(cfg), chunk=cfg.sorted_chunk)
    rows = rows.reshape(tokens.shape + (nl, w))
    return jnp.moveaxis(rows, -2, 0).astype(act_dtype(cfg))


def gain_rows(params: dict, consts: dict, tokens: jax.Array, cfg) -> dict[str, jax.Array] | None:
    """Token gain rows of the input tokens, scaled by the ``gain`` switch, or None when no gain
    table is configured.

    tokens: int ``[B, T]``. Returns ``{table: [P, n_blocks, B, T, width]}`` in ``hi(act_dtype)``
    (float32; float64 only in the float64 tests) for the tables of :func:`gain_widths`: one
    lookup per token and table (``tables.big_lookup`` of the ``[V, P * n_blocks * width]``
    view, ``cfg.table_grad``). The rows are loop-invariant: the loop scan closes over them and
    :func:`block_gain` picks a phase and block inside the step, so their cotangents accumulate
    over the loops in this dtype.
    """
    widths = gain_widths(cfg)
    if not widths:
        return None
    ft = hi(act_dtype(cfg))
    on = jnp.asarray(consts["enable"]["gain"], ft)
    tokens = jnp.asarray(tokens)
    out = {}
    for name in widths:
        table = params["gain"][name]
        nv, p, nb, w = table.shape
        rows = tables.big_lookup(table.reshape(nv, p * nb * w), tokens, mode=cfg.table_grad,
                                 dtype=act_dtype(cfg), chunk=cfg.sorted_chunk)
        rows = rows.astype(ft).reshape(tokens.shape + (p, nb, w))
        out[name] = on * jnp.moveaxis(rows, (-3, -2), (0, 1))
    return out


def loop_phases(cfg) -> np.ndarray:
    """``int32[n_loops]``: the gain phase ``(r * gain_phases) // n_loops`` of every loop ``r``."""
    return (np.arange(cfg.n_loops) * cfg.gain_phases // cfg.n_loops).astype(np.int32)


def block_gain(gains: dict[str, jax.Array] | None, i: int,
               phase: jax.Array | None = None) -> dict[str, Any] | None:
    """Gain rows of block ``i`` at one loop phase: ``{table: [B, T, width]}`` or None.

    gains: from :func:`gain_rows`; phase: ``int32[]`` (traced, e.g. :func:`loop_phases` fed
    through the scan's ``xs``), or None for phase 0 (``gain_phases = 1``: a static slice of the
    closed-over rows, loop-invariant). With a phase the entries are ``layers.PhaseGain`` (the
    closed-over rows, the phase and ``i``): the product selects the rows itself
    (``layers.gain_product``), so the backward pass keeps no per-loop copy of them.
    """
    if not gains:
        return None
    if phase is None:
        return {name: a[0, i] for name, a in gains.items()}
    return {name: PhaseGain(a, phase, i) for name, a in gains.items()}


def token_rows(params: dict, consts: dict, tokens: jax.Array, cfg) -> list[dict] | None:
    """Prepared token-neuron rows of the input tokens, or None when ``tok_neurons`` is 0.

    tokens: int ``[B, T]``. One lookup per token and table (``tables.big_lookup`` of the
    ``[V, n_blocks * rho * dk]`` views of ``page/key`` and ``page/val``, ``cfg.table_grad``);
    then per block ``i`` the keys are RMS-normalised over ``dk`` (``layers.rmsnorm``, unit RMS)
    and keys and values are cast to the activation dtype. Returns one dict per block:
    ``{"key": [B, T, rho, dk], "val": [B, T, rho, dk], "on": float32 []}`` (``on`` is the
    ``page`` switch, which :func:`block` applies to the token neurons' output). The rows are
    loop-invariant: the loop scan closes over them, so they are gathered, normalised and cast
    once per forward and no float32 copy is alive inside the loop (one array per block: a block
    reads its rows without a slice, and their cotangents accumulate over the loops in the
    activation dtype, as for every closed-over value of the scan). With ``cfg.remat`` the key
    preparation is rematerialised in the backward pass, so the float32 gathered keys (the input
    the normalisation's gradient needs) are not kept from the forward to the backward; with
    :data:`FAST_ROWS` they are kept (50 MB per 4 x 1024 piece of configs/v9.py) and the backward
    pass gathers and normalises no second time. With the
    speed switch ``layers.FAST_TOKN`` key and val are ``[B, rho, dk, T]`` (tokens in lanes; the
    layout ``layers.token_neurons`` then reads).
    """
    if not cfg.tok_neurons:
        return None
    dt = act_dtype(cfg)
    tokens = jnp.asarray(tokens)

    def gather(table, tokens):                    # [V, nb, rho, dk] -> [B, T, nb, rho, dk]
        nv, nb, rho, dk = table.shape
        flat = tables.big_lookup(table.reshape(nv, nb * rho * dk), tokens, mode=cfg.table_grad,
                                 dtype=dt, chunk=cfg.sorted_chunk)
        return flat.reshape(tokens.shape + (nb, rho, dk))

    def keys_of(table, tokens):
        rows = gather(table, tokens)
        return [rmsnorm(rows[:, :, i], out_dtype=dt) for i in range(cfg.n_blocks)]

    remat_keys = cfg.remat and not FAST_ROWS              # FAST_ROWS (a): keep the residuals
    keys = (jax.checkpoint(keys_of) if remat_keys else keys_of)(params["page"]["key"], tokens)
    vals = gather(params["page"]["val"], tokens)
    on = jnp.asarray(consts["enable"]["page"], F32)
    rows = [{"key": keys[i], "val": vals[:, :, i].astype(dt), "on": on} for i in range(cfg.n_blocks)]
    if _layers.FAST_TOKN:
        # tokens in lanes for layers.token_neurons: [B, T, rho, dk] -> [B, rho, dk, T], once
        # per forward (the prepared rows, outside the loop scan)
        rows = [{**r, "key": jnp.moveaxis(r["key"], 1, -1), "val": jnp.moveaxis(r["val"], 1, -1)}
                for r in rows]
    return rows


def _lane_dense(a: jax.Array) -> jax.Array:
    """``[..., m, n] -> [..., m * n]`` when ``n`` is not a multiple of 128 (the lane width of a
    TPU tile) and ``m * n`` is; any other array (scalars, lane-dense rows) as it is. A reshape:
    the carrier :data:`FAST_ROWS` hands the loop scan, whatever layout :func:`token_rows` gives."""
    s = np.shape(a)
    if len(s) >= 3 and s[-1] % 128 and (s[-2] * s[-1]) % 128 == 0:
        return a.reshape(s[:-2] + (s[-2] * s[-1],))
    return a


def major_gains(gains: dict[str, jax.Array] | None, cfg) -> dict[str, jax.Array] | None:
    """The attention gain rows (``o``, ``v``) of every block in the head-major layout of the
    splash path: ``{table: [n_blocks, B, H, T, head_dim]}`` from phase 0 of :func:`gain_rows`
    (``gain_phases = 1``), or None when there is nothing to convert (no such table, or another
    attention implementation). Made once per forward, outside the loop scan: inside it the
    transpose would be repeated for every block application (``attention.MajorGain``)."""
    if not gains or cfg.attn_impl != "splash":
        return None
    h, hd = cfg.n_heads, cfg.head_dim
    out = {}
    for name in ("o", "v"):
        if name in gains:
            rows = gains[name][0]                                    # [n_blocks, B, T, H*hd]
            out[name] = jnp.swapaxes(rows.reshape(rows.shape[:3] + (h, hd)), 2, 3)
    return out or None


def block_params(params: dict, cfg) -> list[dict]:
    """Per-block parameter dicts with matrices cast to the activation dtype (outside the scan).

    With ``attn_gate`` the attention entry also holds ``gate_w [d, H]`` and ``gate_b
    [n_loops, H]`` (float32, indexed by the loop inside the scan); with ``tok_neurons`` the
    block holds the projections ``pin [d, dk]`` and ``pout [dk, d]``; with ``attn_impl ==
    "splash"`` the attention entry also holds the matrices of the head-major path
    (``attention.major_weights``).
    """
    dt = act_dtype(cfg)
    core = params["core"]
    pa_all, pm, pn = core["attn"], core["mlp"], core["norm"]
    out = []
    for i in range(cfg.n_blocks):
        pa = {"wq": pa_all["wq"][i].astype(dt), "wk": pa_all["wk"][i].astype(dt),
              "wv": pa_all["wv"][i].astype(dt), "q_gain": pa_all["q_gain"][i]}
        if "sink_logit" in pa_all:
            pa["sink_logit"] = pa_all["sink_logit"][i]
            pa["sink_value"] = pa_all["sink_value"][i]
        if i == 0 and "ve_gate" in pa_all:
            pa["ve_gate"] = pa_all["ve_gate"]
        if cfg.attn_gate:
            pa["gate_w"] = pa_all["gate_w"][i]
            pa["gate_b"] = pa_all["gate_b"][:, i]
        if cfg.attn_impl == "splash":
            pa.update(attention.major_weights(pa, cfg))
        pb = {"attn": pa, "wo": pa_all["wo"][i].astype(dt),
              "w_up": pm["w_up"][i].astype(dt), "w_down": pm["w_down"][i].astype(dt),
              "g1": pn["g1"][i], "g2": pn["g2"][i]}
        if cfg.tok_neurons:
            pb["pin"] = params["page"]["pin"][i].astype(dt)
            pb["pout"] = params["page"]["pout"][i].astype(dt)
        out.append(pb)
    return out


def block(pb: dict, u: jax.Array, ctx: attention.Ctx, ve: jax.Array | None, cfg,
          attend_fn=None, gain: dict[str, Any] | None = None,
          loop: jax.Array | int | None = None, tok: dict[str, jax.Array] | None = None):
    """``u += attn(rmsnorm(u) * g1); u += mlp(x) + token_neurons(x)`` with ``x = rmsnorm(u) * g2``.

    pb: one entry of :func:`block_params`; u: ``[B, T, d]`` (activation dtype); ve: this loop's
    value embedding for block 0, else None; attend_fn: optional attention override (see
    ``attention.attention``; the output gate and the token gains apply to its result too);
    gain: this block's token gain rows at this loop's phase (:func:`block_gain`: ``{"ffn":
    [B, T, gain_ffn], "o": [B, T, H*hd], "v": [B, T, H*hd]}``, the tables present) or None;
    loop: int32 ``[]`` loop index (traced, fed through the scan's ``xs``), required with
    ``attn_gate``; tok: this block's token-neuron rows (entry ``i`` of :func:`token_rows`),
    required with ``tok_neurons``. Returns ``(u, aux)`` (by default aux is the mean sink mass).
    """
    dt = act_dtype(cfg)
    g = gain or {}
    x = rmsnorm(u, pb["g1"], out_dtype=dt)
    a, aux = attention.attention(pb["attn"], pb["wo"], x, ctx, ve, cfg, attend_fn,
                                 gain_v=g.get("v"), gain_o=g.get("o"), loop=loop)
    u = (u.astype(hi(dt)) + a).astype(dt)
    x = rmsnorm(u, pb["g2"], out_dtype=dt)
    f = mlp(x, pb["w_up"], pb["w_down"], dt, gain=g.get("ffn"))
    if cfg.tok_neurons:
        if tok is None:
            raise ValueError("tok_neurons > 0: block needs its token-neuron rows (token_rows)")
        f = f + token_neurons(x, pb["pin"], pb["pout"], tok["key"], tok["val"], tok["on"], dt)
    u = (u.astype(hi(dt)) + f).astype(dt)
    return u, aux


def forward(params: dict, consts: dict, feats: dict, cfg) -> tuple[jax.Array, dict]:
    """Final hidden state ``float32[B, T, d]`` and auxiliaries ``{"sink_mass": float32[n_blocks]}``
    (mean attention mass on the sink per block, averaged over loops)."""
    check_cfg(cfg)
    dt = act_dtype(cfg)
    ctx = make_ctx(consts, feats, cfg)
    e = embed(params, consts, feats, cfg)
    ve_all = value_embeddings(params, jnp.asarray(feats["tokens"]), cfg)
    gains = gain_rows(params, consts, jnp.asarray(feats["tokens"]), cfg)
    tok = token_rows(params, consts, jnp.asarray(feats["tokens"]), cfg)
    tok_shapes = None
    if FAST_ROWS and tok is not None:
        # the scan closes over lane-dense views; each step reshapes them back (FAST_ROWS (b))
        tok_shapes = jax.tree.map(np.shape, tok)
        tok = jax.tree.map(_lane_dense, tok)
    blocks = block_params(params, cfg)
    lp = params["lanes"]
    xs: dict[str, Any] = {"alpha": lp["alpha"], "mix": lp["mix"], "beta": lp["beta"],
                          "g": params["loop"]["g"]}
    if ve_all is not None:
        xs["ve"] = ve_all
    if gains is not None and cfg.gain_phases > 1:
        xs["phase"] = jnp.asarray(loop_phases(cfg))
    if cfg.attn_gate:
        xs["loop"] = jnp.arange(cfg.n_loops, dtype=jnp.int32)
    major = major_gains(gains, cfg) if "phase" not in xs else None

    def tok_of(i):
        if tok is None or tok_shapes is None:
            return None if tok is None else tok[i]
        return jax.tree.map(lambda a, s: a.reshape(s), tok[i], tok_shapes[i])

    def step(lanes, x):
        u = controller.combine(x["alpha"], lanes, dt)
        masses = []
        for i, pb in enumerate(blocks):
            g = block_gain(gains, i, x.get("phase"))
            if major:
                g = {**g, **{name: attention.MajorGain(rows[i]) for name, rows in major.items()}}
            u, m = block(pb, u, ctx, x.get("ve") if i == 0 else None, cfg, gain=g,
                         loop=x.get("loop"), tok=tok_of(i))
            masses.append(m)
        s = rmsnorm(u, x["g"], out_dtype=dt)
        return controller.update(x["mix"], x["beta"], lanes, s, dt), jnp.stack(masses)

    lanes, masses = controller.run_loops(step, e, xs, cfg.n_lanes, cfg.remat, cfg.remat_keep)
    h = head.final_hidden(lanes, params["head"]["g"])
    return h, {"sink_mass": jnp.mean(masses, axis=0)}


def forward_hidden(params: dict, consts: dict, feats: dict, cfg) -> jax.Array:
    """Final hidden state ``float32[B, T, d]`` (input of the head and of the critic)."""
    return forward(params, consts, feats, cfg)[0]


def tape_masses(params: dict, consts: dict, hidden: jax.Array, feats: dict, cfg):
    """``(cand_tok int[..., K], m float32[..., K])``: the candidates and their masses in the
    output law (``bes.model.predictive.masses``), or ``(None, None)`` when the tape path is
    statically off or no tape is given.

    hidden: ``[..., d]``; feats: ``cand_tok [..., K]`` and optionally ``cand_feat [..., K, F]``
    with the same leading shape. The runtime switches enter through the slot mask
    (``cand_out``, ``kind``) and the ``src`` multipliers.
    """
    cand_tok, cand_feat = _cand(feats)
    if not cfg.use_cand_out or cand_tok is None:
        return None, None
    en = consts["enable"]
    mask = oracle_io.cand_mask(cand_tok, cand_feat, en["kind"], en["cand_out"])
    src = en["src"] if "src" in en else jnp.ones((N_SRC,), F32)
    return cand_tok, predictive.masses(params["trust"], hidden, cand_tok, cand_feat, mask, src, cfg)


def logits_from_hidden(params: dict, consts: dict, hidden: jax.Array, feats: dict, cfg,
                       temperature=1.0) -> jax.Array:
    """Final next-token scores ``s float32[..., V]`` from a hidden state ``[..., d]``, with
    ``softmax(s) = p_T``: the output law at sampling temperature ``T``, which tempers the
    network's distribution only (``predictive.mix_logprobs``; the tape's masses are untouched).

    feats may hold ``cand_tok [..., K]`` / ``cand_feat [..., K, F]`` with the same leading
    shape as ``hidden`` (e.g. ``[B]`` for one decoding step). When the tape path is statically
    on and a tape is given, ``s = log p_T`` (normalised log-probabilities); otherwise
    ``s = z / T`` (the softcapped logits ``z``). temperature: ``T > 0``, a Python float or a
    float32 scalar (traced under ``jit``: one compiled program serves every temperature); the
    default ``1`` gives the law ``p`` itself and leaves ``z`` untouched. Callers rely only on
    ``softmax(s) = p_T``.
    """
    z = head.logits(params["head"], hidden, cfg)
    cand_tok, m = tape_masses(params, consts, hidden, feats, cfg)
    unit = isinstance(temperature, (int, float)) and float(temperature) == 1.0
    if cand_tok is None:
        return z if unit else z / jnp.asarray(temperature, z.dtype)
    return predictive.mix_logprobs(z, cand_tok, m, temperature)


def logits_fn(params: dict, consts: dict, feats: dict, cfg, temperature=1.0) -> jax.Array:
    """Unchunked scores ``float32[B, T, V]`` of :func:`logits_from_hidden` (tests and
    generation references; ``temperature`` as there)."""
    return logits_from_hidden(params, consts, forward_hidden(params, consts, feats, cfg), feats, cfg,
                              temperature)


def token_ce(params: dict, consts: dict, hidden: jax.Array, feats: dict, y: jax.Array, cfg, *,
             parts: bool = False):
    """Per-token cross-entropy of the output law, tape included: ``float32[n]``.

    hidden: ``[..., d]`` final hidden states; y: int ``[...]`` targets (same leading shape,
    flattened to ``n`` rows); feats: the tape as in :func:`tape_masses`. The network part
    ``-log q(y)`` comes from ``head.chunked_ce`` (chunks of ``cfg.loss_chunk`` rows); the tape
    adds per-token terms only (``predictive.mix_ce``). With ``parts`` returns
    ``(ce, ce_base, mass)``: ``ce_base = -log q(y)`` and ``mass = sum_k m_k`` (``float32[n]``,
    zeros without a tape).
    """
    d = hidden.shape[-1]
    yy = jnp.asarray(y).reshape(-1)
    ce_base = head.chunked_ce(params["head"], hidden.reshape(-1, d), yy, cfg)
    cand_tok, m = tape_masses(params, consts, hidden, feats, cfg)
    if cand_tok is None:
        ce, mass = ce_base, jnp.zeros_like(ce_base)
    else:
        k = cand_tok.shape[-1]
        cand, m = cand_tok.reshape(-1, k), m.reshape(-1, k)
        ce = predictive.mix_ce(ce_base, yy, cand, m, cfg.vocab_size)
        mass = jnp.sum(m, axis=-1)
    return (ce, ce_base, mass) if parts else ce


# ------------------------------------------------------------------------------- loss
def split_batch(batch: dict) -> tuple[dict, jax.Array, jax.Array]:
    """Micro-batch (SPEC section 3 without the ``A`` axis) -> ``(feats, targets [B, T], w [B, T])``."""
    tokens = jnp.asarray(batch["tokens"])
    feats = {"tokens": tokens[..., :-1]}
    for name in ("seg", "pos", "reg", "clock", "cand_tok", "cand_feat"):
        if batch.get(name) is not None:
            feats[name] = batch[name]
    return feats, tokens[..., 1:], jnp.asarray(batch["w"], F32)


def loss_fn(params: dict, consts: dict, batch: dict, cfg) -> tuple[jax.Array, dict]:
    """Weighted mean cross-entropy of one micro-batch and metrics.

    batch: ``tokens int[B, T+1]``, ``w float32[B, T]`` and optional ``seg``, ``reg``,
    ``clock``, ``cand_tok``, ``cand_feat`` (``[B, T, ...]``). The cross-entropy is that of the
    output law (:func:`token_ce`); logits and log-sum-exp are float32 and computed in chunks of
    ``cfg.loss_chunk`` rows.
    Metrics: ``loss``, ``ce_sum`` (sum of w * ce), ``weight`` (sum of w), ``tokens`` (count of
    w > 0), ``tape_mass`` (w-weighted mean of the total tape mass ``M = sum_k m_k``),
    ``tape_gain`` (w-weighted mean of ``-log q(y) - ce`` in nats: what the tape buys; it can be
    negative), ``sink_mass`` (``[n_blocks]``), ``ce_by_role`` / ``weight_by_role`` /
    ``loss_share_by_role`` (``[6]``, by the role register of the input position; zeros
    without registers).
    """
    feats, y, w = split_batch(batch)
    h, aux = forward(params, consts, feats, cfg)
    b_, t_, _ = h.shape
    n = b_ * t_
    ce, ce_base, mass = token_ce(params, consts, h, feats, y, cfg, parts=True)
    w = w.reshape(n)
    weight = jnp.sum(w)
    wce = w * ce
    ce_sum = jnp.sum(wce)
    denom = jnp.maximum(weight, 1e-9)
    loss = ce_sum / denom
    metrics = {"loss": loss, "ce_sum": ce_sum, "weight": weight,
               "tokens": jnp.sum((w > 0).astype(F32)), "sink_mass": aux["sink_mass"],
               "tape_mass": jnp.sum(w * mass) / denom,
               "tape_gain": jnp.sum(w * (ce_base - ce)) / denom}
    reg = _feat(feats, "reg")
    if reg is None:
        zeros = jnp.zeros((N_ROLES,), F32)
        metrics.update(ce_by_role=zeros, weight_by_role=zeros, loss_share_by_role=zeros)
    else:
        role = jnp.clip(reg[..., ROLE_FIELD].astype(jnp.int32), 0, N_ROLES - 1).reshape(n)
        onehot = (role[:, None] == jnp.arange(N_ROLES, dtype=jnp.int32)).astype(F32)
        ce_by_role = jnp.sum(wce[:, None] * onehot, axis=0)
        metrics["ce_by_role"] = ce_by_role
        metrics["weight_by_role"] = jnp.sum(w[:, None] * onehot, axis=0)
        metrics["loss_share_by_role"] = ce_by_role / jnp.maximum(ce_sum, 1e-9)
    return loss, metrics
