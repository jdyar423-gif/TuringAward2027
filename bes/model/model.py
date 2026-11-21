"""BES looped transformer: parameters, constants, forward pass and loss (SPEC section 4).

    e      = rmsnorm(E_tok[x] + sum_f E_reg_f[reg_f] + W_cin @ sum_k m_k (E_ctok[cand_k] + sum_f E_cfeat_f[feat_kf]))
    lanes  = (e, ..., e)
    for r in 0..n_loops-1:                                   # lax.scan, optional jax.checkpoint
        u     = sum_n alpha[r, n] lanes[n]
        u     = block_0(u; ve = E_ve[x, r]);  u = block_1(u); ...
        s     = rmsnorm(u) * g_loop[r]
        lanes = mix[r] @ lanes + beta[r][:, None] * s
    h      = rmsnorm(sum_n lanes[n]) * g_head
    z      = softcap * tanh(((h @ P) @ E_out.T) / softcap) + sum_k b_k onehot(cand_k)

Core design and initialisation ported from kschwethelm/looped-lm-scaling ``nanochat/gpt.py``
("hyperconnect" mode) and the value-embedding gate from karpathy/nanochat ``nanochat/gpt.py``
(both MIT licence). Every new feature path starts from zero-initialised tables or scalars, so
at initialisation the register, candidate-input and tilt paths contribute exactly zero; the
sink and confidence paths are active from the start (sink logit 0, prior mean 0) and are
removed exactly by their switches. ``make_consts`` holds the runtime 0/1 switches so one
compiled graph covers every rung of the fallback ladder.

Input dictionaries (``feats``): ``tokens`` int ``[B, T]`` (required) and optionally ``seg``
``[B, T]`` (default 0), ``pos`` ``[B, T]`` (default ``arange(T)``), ``reg`` ``[B, T, N_REG]``,
``clock`` ``[B, T, N_CLOCKS]``, ``cand_tok`` ``[B, T, K]`` and ``cand_feat``
``[B, T, K, N_CAND_FEAT]``. A missing optional feature behaves like a disabled path.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_FEATS, REG_FIELDS, special_id
from bes.model import attention, controller, head, oracle_io, rope_clocks, sinks, tables
from bes.model.critic import critic  # noqa: F401  (re-exported for other tracks)
from bes.model.layers import (F32, act_dtype, hi, leaf_key, mlp, normal_init, rmsnorm,
                              uniform_init)

N_KINDS = dict(CAND_FEATS)["kind"]
ROLE_FIELD = [name for name, _ in REG_FIELDS].index("role")
N_ROLES = dict(REG_FIELDS)["role"]
#: names of the scalar runtime switches in ``consts["enable"]`` (plus the vector ``kind``)
SWITCHES = ("reg", "cand_in", "cand_out", "conf", "sink", "clock")


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
    if cfg.table_grad not in ("onehot", "take"):
        raise ValueError(f"unknown table_grad {cfg.table_grad!r}")


# ------------------------------------------------------------------------- parameters
def init_params(cfg, key: jax.Array) -> dict:
    """Initial parameter tree (float32) with the stable names of SPEC 4.3.

    Reference initialisation for the core: token embedding N(0, 1), unembedding N(0, 0.001),
    every matrix uniform with std ``1/sqrt(fan_in)`` (all non-zero), RMSNorm gains 1, cyclic
    lanes, value embeddings like ``W_v``, value gate uniform(0, 0.02), query gain
    ``sqrt(dc)`` (the reference scale ``q.k / sqrt(dims)`` of RMS-normalised vectors).
    Feature tables (registers, candidate tables, trust features) and the sink scalars are
    zero; the critic output weights are zero.
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
                           "feat": {name: zeros((card, cd)) for name, card in CAND_FEATS}}
    params["critic"] = {"w1": uni("critic/w1", (d, cfg.critic_hidden), d),
                        "w2": zeros((cfg.critic_hidden,))}
    return params


def make_consts(cfg, **switches) -> dict:
    """Non-trainable inputs: rotary frequencies and the runtime feature switches.

    Keyword switches (0/1 floats or bools): ``reg``, ``cand_in``, ``cand_out``, ``conf``,
    ``sink``, ``clock`` (multiplies the clock values; 0 leaves the clock planes unrotated);
    ``kind`` (a scalar or a sequence of ``N_KINDS`` per candidate kind) or ``kind_<i>``.
    Defaults: on when the static ``cfg.use_*`` flag (``clock_planes > 0`` for ``clock``) is
    on. A switch never re-enables a path whose static flag is off.
    Returns ``{"rope_freq": float32[1 + N_CLOCKS, P], "enable": {name: float32 scalar,
    "kind": float32[N_KINDS]}}``.
    """
    values = {"reg": cfg.use_reg, "cand_in": cfg.use_cand_in, "cand_out": cfg.use_cand_out,
              "conf": cfg.use_conf, "sink": cfg.use_sink, "clock": cfg.clock_planes > 0}
    kind = np.ones(N_KINDS, np.float32)
    for name, value in switches.items():
        if name in values:
            values[name] = value
        elif name == "kind":
            kind = np.broadcast_to(np.asarray(value, np.float32), (N_KINDS,)).copy()
        elif name.startswith("kind_") and name[5:].isdigit() and int(name[5:]) < N_KINDS:
            kind[int(name[5:])] = float(value)
        else:
            raise KeyError(f"unknown switch {name!r}")
    enable = {name: jnp.asarray(float(values[name]), F32) for name in SWITCHES}
    enable["kind"] = jnp.asarray(kind)
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
    """Rotary tables, sink indicator, confidence switch, segments and positions."""
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
    return attention.Ctx(cos, sin, sink, conf, seg.astype(jnp.int32), pos.astype(jnp.int32))


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
    pre = tables.big_lookup(params["emb"]["tok"], tokens, mode=cfg.table_grad, dtype=dt)
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
                             dtype=act_dtype(cfg))
    rows = rows.reshape(tokens.shape + (nl, w))
    return jnp.moveaxis(rows, -2, 0).astype(act_dtype(cfg))


def block_params(params: dict, cfg) -> list[dict]:
    """Per-block parameter dicts with matrices cast to the activation dtype (outside the scan)."""
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
        out.append({"attn": pa, "wo": pa_all["wo"][i].astype(dt),
                    "w_up": pm["w_up"][i].astype(dt), "w_down": pm["w_down"][i].astype(dt),
                    "g1": pn["g1"][i], "g2": pn["g2"][i]})
    return out


def block(pb: dict, u: jax.Array, ctx: attention.Ctx, ve: jax.Array | None, cfg,
          attend_fn=None):
    """``u += attn(rmsnorm(u) * g1); u += mlp(rmsnorm(u) * g2)``.

    pb: one entry of :func:`block_params`; u: ``[B, T, d]`` (activation dtype); ve: this loop's
    value embedding for block 0, else None; attend_fn: optional attention override (see
    ``attention.attention``). Returns ``(u, aux)`` (by default aux is the mean sink mass).
    """
    dt = act_dtype(cfg)
    x = rmsnorm(u, pb["g1"], out_dtype=dt)
    a, aux = attention.attention(pb["attn"], pb["wo"], x, ctx, ve, cfg, attend_fn)
    u = (u.astype(hi(dt)) + a).astype(dt)
    x = rmsnorm(u, pb["g2"], out_dtype=dt)
    u = (u.astype(hi(dt)) + mlp(x, pb["w_up"], pb["w_down"], dt)).astype(dt)
    return u, aux


def forward(params: dict, consts: dict, feats: dict, cfg) -> tuple[jax.Array, dict]:
    """Final hidden state ``float32[B, T, d]`` and auxiliaries ``{"sink_mass": float32[n_blocks]}``
    (mean attention mass on the sink per block, averaged over loops)."""
    check_cfg(cfg)
    dt = act_dtype(cfg)
    ctx = make_ctx(consts, feats, cfg)
    e = embed(params, consts, feats, cfg)
    ve_all = value_embeddings(params, jnp.asarray(feats["tokens"]), cfg)
    blocks = block_params(params, cfg)
    lp = params["lanes"]
    xs: dict[str, Any] = {"alpha": lp["alpha"], "mix": lp["mix"], "beta": lp["beta"],
                          "g": params["loop"]["g"]}
    if ve_all is not None:
        xs["ve"] = ve_all

    def step(lanes, x):
        u = controller.combine(x["alpha"], lanes, dt)
        masses = []
        for i, pb in enumerate(blocks):
            u, m = block(pb, u, ctx, x.get("ve") if i == 0 else None, cfg)
            masses.append(m)
        s = rmsnorm(u, x["g"], out_dtype=dt)
        return controller.update(x["mix"], x["beta"], lanes, s, dt), jnp.stack(masses)

    lanes, masses = controller.run_loops(step, e, xs, cfg.n_lanes, cfg.remat)
    h = head.final_hidden(lanes, params["head"]["g"])
    return h, {"sink_mass": jnp.mean(masses, axis=0)}


def forward_hidden(params: dict, consts: dict, feats: dict, cfg) -> jax.Array:
    """Final hidden state ``float32[B, T, d]`` (input of the head and of the critic)."""
    return forward(params, consts, feats, cfg)[0]


def tilt_terms(params: dict, consts: dict, hidden: jax.Array, feats: dict, cfg):
    """``(cand_tok [..., K], b float32[..., K], m float32[..., K])`` of the output tilt, or
    ``(None, None, None)`` when the tilt path is statically off or no tape is given."""
    cand_tok, cand_feat = _cand(feats)
    if not cfg.use_cand_out or cand_tok is None:
        return None, None, None
    en = consts["enable"]
    m = oracle_io.cand_mask(cand_tok, cand_feat, en["kind"], en["cand_out"])
    return cand_tok, oracle_io.trust(params["trust"], hidden, cand_feat, m), m


def logits_from_hidden(params: dict, consts: dict, hidden: jax.Array, feats: dict, cfg) -> jax.Array:
    """Tilted logits ``float32[..., V]`` from a hidden state ``[..., d]``.

    feats may hold ``cand_tok [..., K]`` / ``cand_feat [..., K, F]`` with the same leading
    shape as ``hidden`` (e.g. ``[B]`` for one decoding step); without them there is no tilt.
    """
    z = head.logits(params["head"], hidden, cfg)
    cand_tok, b, _ = tilt_terms(params, consts, hidden, feats, cfg)
    if cand_tok is not None:
        z = oracle_io.add_tilt(z, cand_tok, b)
    return z


def logits_fn(params: dict, consts: dict, feats: dict, cfg) -> jax.Array:
    """Unchunked tilted logits ``float32[B, T, V]`` (tests and generation references)."""
    return logits_from_hidden(params, consts, forward_hidden(params, consts, feats, cfg), feats, cfg)


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
    ``clock``, ``cand_tok``, ``cand_feat`` (``[B, T, ...]``). Logits and log-sum-exp are
    float32 and computed in chunks of ``cfg.loss_chunk`` rows.
    Metrics: ``loss``, ``ce_sum`` (sum of w * ce), ``weight`` (sum of w), ``tokens`` (count of
    w > 0), ``trust_abs`` (mean |b| over active candidate slots), ``sink_mass``
    (``[n_blocks]``), ``ce_by_role`` / ``weight_by_role`` / ``loss_share_by_role`` (``[6]``,
    by the role register of the input position; zeros without registers).
    """
    feats, y, w = split_batch(batch)
    h, aux = forward(params, consts, feats, cfg)
    b_, t_, d = h.shape
    n = b_ * t_
    cand_tok, b, m = tilt_terms(params, consts, h, feats, cfg)
    ce = head.chunked_ce(params["head"], h.reshape(n, d), y.reshape(n),
                         None if cand_tok is None else cand_tok.reshape(n, -1),
                         None if b is None else b.reshape(n, -1), cfg)
    w = w.reshape(n)
    weight = jnp.sum(w)
    wce = w * ce
    ce_sum = jnp.sum(wce)
    loss = ce_sum / jnp.maximum(weight, 1e-9)
    metrics = {"loss": loss, "ce_sum": ce_sum, "weight": weight,
               "tokens": jnp.sum((w > 0).astype(F32)), "sink_mass": aux["sink_mass"]}
    if b is None:
        metrics["trust_abs"] = jnp.zeros((), F32)
    else:
        metrics["trust_abs"] = jnp.sum(jnp.abs(b)) / jnp.maximum(jnp.sum(m), 1.0)
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
