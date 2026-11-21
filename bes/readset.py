"""Read-set audit of the active-parameter count (SPEC 4.3, plan section 7 gate E7).

The claim audited is "at most 20M numbers read per generated token". The distribution of the
token after position ``t`` is computed at position ``t``; ``docs/ACCOUNTING.md`` defines two
conventions for the numbers it reads:

* **step** (standard, the MoE "A" convention): given the cached keys and values of the earlier
  positions, every dense leaf (each number once, however many loops reuse it) plus the distinct
  rows that position ``t`` indexes in every lookup table;
* **strict**: every number the distribution functionally depends on: the dense leaves, the rows
  indexed at every earlier position that ``t`` can see (same segment, ``j <= t``) and the
  output-side rows of ``t`` itself. A sink position (``BOS`` with the sink on) contributes no
  rows to later positions: its key and value are ``sink_logit`` / ``sink_value`` at every loop
  and block, whatever its token, registers or candidates.

Rows read by one position (indices clipped into the table, as the model does):

=====================  ===========================================================  ==========
table                  row                                                          side
=====================  ===========================================================  ==========
``emb/tok``            the token                                                    input
``emb/ve``             the token (one ``[n_loops, ve_dim]`` row)                    input
``emb/reg/<f>``        the register value of field ``f`` (if the ``reg`` switch)    input
``cand/in_tok``        the candidate of every active slot (``mask_k != 0``)         input
``cand/in_feat/<f>``   feature ``f`` of every active slot                           input
``trust/feat/<f>``     feature ``f`` of every slot with output mass                 output
``trust/bias/<f>``     feature ``f`` of every slot with output mass (width 1)       output
=====================  ===========================================================  ==========

A slot has output mass (``bes.model.predictive``) when its ``cand_out`` slot mask is non-zero,
its id lies in the vocabulary and the ``src`` multiplier of its source is non-zero. Dense
leaves of switched-off paths are not read (``cand/in_proj`` without ``cand_in``,
``trust/proj`` without ``cand_out``, the sink scalars without ``sink``); the critic is read only
by decoders that use it (``include_critic``).

Contents:

* :func:`read_rows` - the read set of the last position of a sequence (either convention);
* :class:`RowStore` - the dense leaves plus *only* the read rows; any other row raises KeyError;
* :func:`reference_logits` - a numpy (float64) forward of the last position that sees nothing
  but a :class:`RowStore`, from scratch (strict) or from a key-value cache (step);
* :func:`poison` - large finite noise on every lookup row outside a read set;
* :class:`ReadLogger` - distinct stored numbers read per generated token, both conventions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from bes.config import CAND_FEATS, REG_FIELDS, special_id
from bes.roles import leaves_with_paths, role_of

RMS_EPS = 1e-6      # bes.model.layers.RMS_EPS
L2_EPS = 1e-12      # bes.model.layers.L2_EPS
VE_GATE_CHANNELS = 12
KIND_FIELD = [name for name, _ in CAND_FEATS].index("kind")
N_KINDS = dict(CAND_FEATS)["kind"]
SRC_FIELD = [name for name, _ in CAND_FEATS].index("src")
N_SRC = dict(CAND_FEATS)["src"]
SWITCHES = ("reg", "cand_in", "cand_out", "conf", "sink", "clock")
#: dense leaves that a runtime switch removes from the computation
_DENSE_SWITCH = {"cand/in_proj": "cand_in", "trust/proj": "cand_out",
                 "core/attn/sink_logit": "sink", "core/attn/sink_value": "sink"}


# ---------------------------------------------------------------------------- helpers
def _enable(consts: Mapping | None, cfg) -> dict[str, Any]:
    """Runtime switches as floats (``kind`` as ``float64[N_KINDS]``, ``src`` as
    ``float64[N_SRC]``); defaults follow the static flags (as ``bes.model.model.make_consts``)."""
    static = {"reg": cfg.use_reg, "cand_in": cfg.use_cand_in, "cand_out": cfg.use_cand_out,
              "conf": cfg.use_conf, "sink": cfg.use_sink, "clock": cfg.clock_planes > 0}
    en = {} if consts is None else consts["enable"]
    out: dict[str, Any] = {n: float(np.asarray(en[n])) if n in en else float(static[n])
                           for n in SWITCHES}
    for name, size in (("kind", N_KINDS), ("src", N_SRC)):
        out[name] = (np.asarray(en[name], np.float64) if name in en
                     else np.ones(size, np.float64))
    return out


def _row_feats(feats: Mapping) -> dict[str, np.ndarray]:
    """One sequence: accepts ``[T, ...]`` or ``[1, T, ...]`` arrays; returns numpy ``[T, ...]``."""
    out = {}
    tok = np.asarray(feats["tokens"])
    squeeze = tok.ndim == 2
    if squeeze and tok.shape[0] != 1:
        raise ValueError("the audit works on one sequence (batch size 1)")
    for k, a in feats.items():
        if a is None:
            continue
        a = np.asarray(a)
        out[k] = a[0] if squeeze else a
    return out


def _slot_mask(cand_tok_row: np.ndarray, cand_feat_row: np.ndarray, kind_on: np.ndarray,
               path_on: float) -> np.ndarray:
    """``mask_k`` of ``bes.model.oracle_io.cand_mask`` for one position: float64 ``[K]``."""
    kind = np.clip(cand_feat_row[:, KIND_FIELD].astype(np.int64), 0, N_KINDS - 1)
    return (cand_tok_row >= 0).astype(np.float64) * kind_on[kind] * float(path_on)


def _out_weight(cfg, en: Mapping, cand_tok_row: np.ndarray, cand_feat_row: np.ndarray) -> np.ndarray:
    """Constant factor of the output-law mass of each slot of one position: the ``cand_out``
    slot mask times the ``src`` multiplier, zero for ids outside the vocabulary. float64 ``[K]``."""
    src = np.clip(cand_feat_row[:, SRC_FIELD].astype(np.int64), 0, N_SRC - 1)
    return (_slot_mask(cand_tok_row, cand_feat_row, en["kind"], en["cand_out"])
            * (cand_tok_row < cfg.vocab_size) * en["src"][src])


def position_reads(cfg, en: Mapping, token: int, reg_row=None, cand_tok_row=None,
                   cand_feat_row=None, *, inputs: bool = True,
                   outputs: bool = True) -> dict[str, set[int]]:
    """Rows read by one position: ``{table path: {row, ...}}`` (see the module table).

    en: switches from :func:`_enable`; inputs / outputs select the input-side and the
    output-side (output-law) rows.
    """
    v = cfg.vocab_size
    out: dict[str, set[int]] = defaultdict(set)
    if inputs:
        x = int(np.clip(int(token), 0, v - 1))
        out["emb/tok"].add(x)
        if cfg.ve_dim:
            out["emb/ve"].add(x)
        if cfg.use_reg and reg_row is not None and en["reg"] != 0:
            for f, (name, card) in enumerate(REG_FIELDS):
                out[f"emb/reg/{name}"].add(int(np.clip(int(reg_row[f]), 0, card - 1)))
    has_cand = cand_tok_row is not None
    if has_cand:
        cand_tok_row = np.asarray(cand_tok_row).astype(np.int64)
        cand_feat_row = (np.zeros(cand_tok_row.shape + (len(CAND_FEATS),), np.int64)
                         if cand_feat_row is None else np.asarray(cand_feat_row).astype(np.int64))
    for side, flag, use, switch, tok_table, prefixes in (
            ("in", inputs, cfg.use_cand_in, "cand_in", "cand/in_tok", ("cand/in_feat",)),
            ("out", outputs, cfg.use_cand_out, "cand_out", None, ("trust/feat", "trust/bias"))):
        if not (flag and use and has_cand and en[switch] != 0):
            continue
        if side == "out":                  # no mass outside the vocabulary or at src scale 0
            m = _out_weight(cfg, en, cand_tok_row, cand_feat_row)
        else:
            m = _slot_mask(cand_tok_row, cand_feat_row, en["kind"], en[switch])
        for k in np.flatnonzero(m):
            if tok_table is not None:
                out[tok_table].add(int(np.clip(cand_tok_row[k], 0, v - 1)))
            for f, (name, card) in enumerate(CAND_FEATS):
                row = int(np.clip(cand_feat_row[k, f], 0, card - 1))
                for prefix in prefixes:
                    out[f"{prefix}/{name}"].add(row)
    return out


def _merge(into: dict[str, set[int]], other: Mapping[str, Iterable[int]]) -> None:
    for p, rows in other.items():
        into.setdefault(p, set()).update(int(r) for r in rows)


def _is_sink(cfg, en, token: int) -> bool:
    return bool(cfg.use_sink and en["sink"] != 0 and int(token) == special_id("<|bos|>", cfg.vocab_size))


def visible_positions(feats: Mapping, t: int | None = None) -> np.ndarray:
    """Positions ``j <= t`` in the segment of ``t`` (default: the last position)."""
    f = _row_feats(feats)
    n = f["tokens"].shape[0]
    t = n - 1 if t is None else int(t)
    seg = f.get("seg", np.zeros(n, np.int64))
    pos = f.get("pos", np.arange(n))
    return np.flatnonzero((seg == seg[t]) & (pos <= pos[t]))


def read_rows(cfg, consts: Mapping | None, feats: Mapping,
              convention: str = "strict") -> dict[str, np.ndarray]:
    """Read set of the last position of one sequence: ``{table path: sorted int64 rows}``.

    feats: ``tokens [T]`` (or ``[1, T]``) and optional ``seg``, ``pos``, ``reg``,
    ``cand_tok``, ``cand_feat``; convention: ``"step"`` or ``"strict"`` (module docstring).
    """
    if convention not in ("step", "strict"):
        raise ValueError(f"unknown convention {convention!r}")
    f = _row_feats(feats)
    en = _enable(consts, cfg)
    t = f["tokens"].shape[0] - 1

    def at(j: int, inputs: bool, outputs: bool):
        return position_reads(cfg, en, f["tokens"][j], _get(f, "reg", j), _get(f, "cand_tok", j),
                              _get(f, "cand_feat", j), inputs=inputs, outputs=outputs)

    rows: dict[str, set[int]] = {}
    _merge(rows, at(t, True, True))
    if convention == "strict":
        for j in visible_positions(f, t):
            if j != t and not _is_sink(cfg, en, f["tokens"][j]):
                _merge(rows, at(int(j), True, False))
    return {p: np.asarray(sorted(r), np.int64) for p, r in sorted(rows.items())}


def _get(f: Mapping, key: str, j: int):
    a = f.get(key)
    return None if a is None else a[j]


def dense_paths(params: Any, cfg, consts: Mapping | None = None, *,
                include_critic: bool = False) -> list[str]:
    """Dense leaves read for every token under the given switches."""
    en = _enable(consts, cfg)
    out = []
    for p, _ in leaves_with_paths(params):
        if role_of(p).kind != "dense":
            continue
        if p.startswith("critic/") and not include_critic:
            continue
        if p in _DENSE_SWITCH and en[_DENSE_SWITCH[p]] == 0:
            continue
        out.append(p)
    return out


def lookup_paths(params: Any) -> list[str]:
    return [p for p, _ in leaves_with_paths(params) if role_of(p).kind == "lookup"]


# --------------------------------------------------------------------------- row store
class RowStore:
    """The numbers a forward may use: dense leaves and only the listed rows of lookup tables.

    params: the parameter tree; reads: ``{table path: rows}`` (e.g. from :func:`read_rows`);
    dense: dense paths to hold (default: every dense leaf). Values are float64 copies.
    """

    def __init__(self, params: Any, reads: Mapping[str, Iterable[int]],
                 dense: Iterable[str] | None = None):
        flat = dict(leaves_with_paths(params))
        want = None if dense is None else set(dense)
        self._dense = {p: np.asarray(a, np.float64) for p, a in flat.items()
                       if role_of(p).kind == "dense" and (want is None or p in want)}
        self._rows: dict[str, dict[int, np.ndarray]] = {}
        for p, rows in reads.items():
            if p not in flat or role_of(p).kind != "lookup":
                raise KeyError(f"{p!r} is not a lookup table of the parameter tree")
            table = np.asarray(flat[p])
            self._rows[p] = {int(i): np.asarray(table[int(i)], np.float64) for i in rows}
        self.widths = {p: int(np.prod(np.shape(a)[1:])) for p, a in flat.items()
                       if role_of(p).kind == "lookup"}

    def has(self, path: str) -> bool:
        return path in self._dense or path in self._rows

    def dense(self, path: str) -> np.ndarray:
        if path not in self._dense:
            raise KeyError(f"dense leaf {path!r} is not in the store")
        return self._dense[path]

    def row(self, path: str, i: int) -> np.ndarray:
        try:
            return self._rows[path][int(i)]
        except KeyError:
            raise KeyError(f"row {int(i)} of {path!r} is not in the read set") from None

    def numbers(self) -> int:
        """Stored numbers held (dense sizes plus read rows times their widths)."""
        n = sum(a.size for a in self._dense.values())
        return n + sum(len(r) * self.widths[p] for p, r in self._rows.items())


# ------------------------------------------------------------------- numpy reference
def _rms(x: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)
    return y if g is None else y * g


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + L2_EPS)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _freq_table(cfg) -> np.ndarray:
    """Rotary frequency table ``[1 + N_CLOCKS, P]`` (written independently of the model)."""
    from bes.config import N_CLOCKS
    dc = cfg.head_dim - cfg.bias_dims
    n_planes = dc // 2
    n_pos = n_planes - cfg.clock_planes
    per = cfg.clock_planes // N_CLOCKS
    table = np.zeros((1 + N_CLOCKS, n_planes), np.float64)
    if n_pos:
        table[0, :n_pos] = cfg.rope_base ** (-np.arange(n_pos) / n_pos)
    for c in range(N_CLOCKS if per else 0):
        table[1 + c, n_pos + c * per: n_pos + (c + 1) * per] = cfg.rope_base ** (-np.arange(per) / per)
    return table.astype(np.float32).astype(np.float64)   # the model keeps float32 frequencies


def _embed(store: RowStore, cfg, en, f, j: int) -> np.ndarray:
    pre = store.row("emb/tok", int(np.clip(f["tokens"][j], 0, cfg.vocab_size - 1))).copy()
    rows = position_reads(cfg, en, f["tokens"][j], _get(f, "reg", j), _get(f, "cand_tok", j),
                          _get(f, "cand_feat", j), inputs=True, outputs=False)
    if cfg.use_reg and _get(f, "reg", j) is not None and en["reg"] != 0:
        for k, (name, card) in enumerate(REG_FIELDS):
            pre += en["reg"] * store.row(f"emb/reg/{name}", int(np.clip(f["reg"][j][k], 0, card - 1)))
    if "cand/in_tok" in rows:
        ct = np.asarray(f["cand_tok"][j]).astype(np.int64)
        cf = _cand_feat(f, j, ct)
        m = _slot_mask(ct, cf, en["kind"], en["cand_in"])
        acc = np.zeros(store.dense("cand/in_proj").shape[0])
        for k in np.flatnonzero(m):
            vec = store.row("cand/in_tok", int(np.clip(ct[k], 0, cfg.vocab_size - 1))).copy()
            for i, (name, card) in enumerate(CAND_FEATS):
                vec += store.row(f"cand/in_feat/{name}", int(np.clip(cf[k, i], 0, card - 1)))
            acc += m[k] * vec
        pre += acc @ store.dense("cand/in_proj")
    return _rms(pre)


def _cand_feat(f: Mapping, j: int, ct: np.ndarray) -> np.ndarray:
    cf = _get(f, "cand_feat", j)
    return np.zeros(ct.shape + (len(CAND_FEATS),), np.int64) if cf is None else np.asarray(cf).astype(np.int64)


def _angles(cfg, en, f, idx: np.ndarray, freq: np.ndarray) -> np.ndarray:
    pos = f.get("pos", np.arange(f["tokens"].shape[0]))[idx].astype(np.float64)
    a = pos[:, None] * freq[0]
    clock = f.get("clock")
    if cfg.clock_planes and clock is not None:
        c = np.asarray(clock)[idx].astype(np.float64) * en["clock"]
        for i in range(c.shape[-1]):
            a = a + c[:, i:i + 1] * freq[1 + i]
    return a


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    p = cos.shape[-1]
    x1, x2 = x[..., :p], x[..., p:]
    return np.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)


def reference_logits(store: RowStore, consts: Mapping | None, feats: Mapping, cfg,
                     kv: Mapping | None = None) -> np.ndarray:
    """Next-token scores ``float64[V]`` of the last position of one sequence, computed in
    numpy from a :class:`RowStore` only: with the tape path on, the log-probabilities of the
    output law ``p = (q + sum_k m_k onehot(cand_k)) / (1 + sum_k m_k)``, else the softcapped
    logits (compare them through :func:`max_logprob_diff`).

    Without ``kv`` (strict): every position the last one can see is recomputed from scratch.
    With ``kv`` (step): ``kv = {"k", "v": [n_loops, n_blocks, S, H, head_dim], "seg", "pos":
    int [S]}`` holds the cached keys and values of the earlier positions (e.g.
    :func:`kv_from_cache`; ``seg < 0`` marks empty slots) and only the last position is
    computed. Positions default to ``arange(T)``, segments to 0.
    """
    f = _row_feats(feats)
    en = _enable(consts, cfg)
    n = f["tokens"].shape[0]
    f.setdefault("pos", np.arange(n))
    f.setdefault("seg", np.zeros(n, np.int64))
    last = n - 1
    idx = np.array([last]) if kv is not None else visible_positions(f, last)
    hd, h_n, nb = cfg.head_dim, cfg.n_heads, cfg.n_blocks
    dc = hd - cfg.bias_dims
    freq = _freq_table(cfg)
    ang = _angles(cfg, en, f, idx, freq)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    bos = special_id("<|bos|>", cfg.vocab_size)
    on = cfg.use_sink and en["sink"] != 0
    sink = (f["tokens"][idx] == bos).astype(np.float64) * en["sink"] if on else None
    conf = en["conf"] if cfg.use_conf else None
    q_n = len(idx)
    # a sink position other than the last one only feeds later positions through its key and
    # value, which are the sink parameters: its rows are never read (zero placeholders)
    dummy = [int(j) != last and _is_sink(cfg, en, f["tokens"][j]) for j in idx]
    d_model = store.dense("head/g").shape[0]
    e = np.stack([np.zeros(d_model) if dm else _embed(store, cfg, en, f, int(j))
                  for j, dm in zip(idx, dummy)])                                    # [Q, d]
    n_ve = cfg.ve_dim // hd
    ve = None
    if cfg.ve_dim:
        ve = np.stack([np.zeros((cfg.n_loops, cfg.ve_dim)) if dm else
                       store.row("emb/ve", int(np.clip(f["tokens"][j], 0, cfg.vocab_size - 1)))
                       for j, dm in zip(idx, dummy)])                               # [Q, L, ve]
    pos_q, seg_q = f["pos"][idx], f["seg"][idx]
    if kv is not None:
        pos_k = np.concatenate([np.asarray(kv["pos"]), pos_q])
        seg_k = np.concatenate([np.asarray(kv["seg"]), seg_q])
    else:
        pos_k, seg_k = pos_q, seg_q
    mask = (pos_k[None, :] <= pos_q[:, None]) & (seg_k[None, :] == seg_q[:, None])  # [Q, S+Q]

    D = store.dense
    wq, wk, wv, wo = D("core/attn/wq"), D("core/attn/wk"), D("core/attn/wv"), D("core/attn/wo")
    q_gain = D("core/attn/q_gain")
    w_up, w_down = D("core/mlp/w_up"), D("core/mlp/w_down")
    g1, g2 = D("core/norm/g1"), D("core/norm/g2")
    alpha, mix, beta = D("lanes/alpha"), D("lanes/mix"), D("lanes/beta")
    lanes = [e.copy() for _ in range(cfg.n_lanes)]
    for r in range(cfg.n_loops):
        u = sum(alpha[r, m] * lanes[m] for m in range(cfg.n_lanes))
        for i in range(nb):
            x = _rms(u, g1[i])
            q = (x @ wq[i]).reshape(q_n, h_n, dc)
            k = (x @ wk[i]).reshape(q_n, h_n, dc)
            v = (x @ wv[i]).reshape(q_n, h_n, hd)
            if i == 0 and ve is not None:
                gate = 3.0 * _sigmoid(x[:, :VE_GATE_CHANNELS] @ D("core/attn/ve_gate"))  # [Q, n_ve]
                v[:, :n_ve] += ve[:, r].reshape(q_n, n_ve, hd) * gate[..., None]
            q = _l2(_rotate(q, cos, sin)) * q_gain[i][None, :, None]
            k = _l2(_rotate(k, cos, sin))
            q = np.concatenate([q, np.ones((q_n, h_n, 1)), np.zeros((q_n, h_n, cfg.bias_dims - 1))], -1)
            v_body, v_last = v[..., :-1], v[..., -1:]
            if conf is not None:
                v_last = conf + (1.0 - conf) * v_last
            k_bias = np.zeros((q_n, h_n, 1))
            if sink is not None:
                s = sink[:, None, None]
                k = k * (1.0 - s)
                k_bias = s * D("core/attn/sink_logit")[i][None, :, None]
                v_body = v_body * (1.0 - s) + s * D("core/attn/sink_value")[i][None]
                v_last = v_last * (1.0 - s)
            k = np.concatenate([k, k_bias, np.zeros((q_n, h_n, cfg.bias_dims - 1))], -1)
            v = np.concatenate([v_body, v_last], -1)
            if kv is not None:
                k = np.concatenate([np.asarray(kv["k"][r, i], np.float64), k], 0)
                v = np.concatenate([np.asarray(kv["v"][r, i], np.float64), v], 0)
            logits = np.einsum("qhd,khd->hqk", q, k)
            logits = np.where(mask[None], logits, -np.inf)
            p = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            p = p / np.sum(p, axis=-1, keepdims=True)
            y = np.einsum("hqk,khd->qhd", p, v).reshape(q_n, h_n * hd)
            u = u + y @ wo[i]
            x2 = _rms(u, g2[i])
            u = u + np.maximum(x2 @ w_up[i], 0.0) ** 2 @ w_down[i]
        s_r = _rms(u, D("loop/g")[r])
        lanes = [sum(mix[r, a, b] * lanes[b] for b in range(cfg.n_lanes)) + beta[r, a] * s_r
                 for a in range(cfg.n_lanes)]
    h = _rms(sum(lanes), D("head/g"))[-1]                                            # [d]
    z = (h @ D("head/proj")) @ D("head/out").T
    if cfg.softcap and cfg.softcap > 0:
        z = cfg.softcap * np.tanh(z / cfg.softcap)
    ct = _get(f, "cand_tok", last)
    if cfg.use_cand_out and ct is not None and en["cand_out"] != 0:
        ct = np.asarray(ct).astype(np.int64)
        cf = _cand_feat(f, last, ct)
        weight = _out_weight(cfg, en, ct, cf)
        ht = h @ D("trust/proj")
        add, total = np.zeros_like(z), 0.0
        for k in np.flatnonzero(weight):
            rows = [(name, int(np.clip(cf[k, i], 0, card - 1)))
                    for i, (name, card) in enumerate(CAND_FEATS)]
            c = sum(store.row(f"trust/feat/{name}", r) for name, r in rows)
            bias = sum(float(store.row(f"trust/bias/{name}", r)) for name, r in rows)
            a = bias + float(ht @ c) - cfg.mass_offset
            mass = math.exp(min(a, cfg.mass_log_cap)) * float(weight[k])
            add[ct[k]] += mass
            total += mass
        logq = log_softmax(z)
        hit = add > 0
        z = np.where(hit, np.log(np.where(hit, np.exp(logq) + add, 1.0)), logq) - math.log1p(total)
    return z


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))


def kv_from_cache(cache: Mapping, row: int = 0) -> dict[str, np.ndarray]:
    """One row of a ``bes.model.generate`` cache as the ``kv`` argument of
    :func:`reference_logits` (all slots; empty slots carry ``seg = -1``)."""
    k = np.asarray(cache["k"], np.float64)[:, :, row]          # [L, nb, H, S, hd]
    v = np.asarray(cache["v"], np.float64)[:, :, row]
    return {"k": np.swapaxes(k, 2, 3), "v": np.swapaxes(v, 2, 3),
            "seg": np.asarray(cache["seg"])[row], "pos": np.asarray(cache["pos"])[row]}


def jax_last_logits(params: Any, consts: Mapping, feats: Mapping, cfg) -> np.ndarray:
    """Scores ``float64[V]`` of the JAX model (``logits_fn``) at the last position of one
    sequence."""
    import jax

    from bes.model import model as M
    f = {k: np.asarray(a)[None] for k, a in _row_feats(feats).items()}
    z = jax.jit(lambda p, c, x: M.logits_fn(p, c, x, cfg))(params, consts, f)
    return np.asarray(z, np.float64)[0, -1]


def max_logprob_diff(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """``max_v |log p_a(v) - log p_b(v)|``."""
    return float(np.max(np.abs(log_softmax(z_a) - log_softmax(z_b))))


# ------------------------------------------------------------------------------ poison
def poison(params: Any, reads: Mapping[str, Iterable[int]], scale: float = 1e3,
           seed: int = 0) -> Any:
    """Copy of ``params`` (numpy float32 leaves) where every row of every lookup table that is
    not in ``reads`` gets ``scale * N(0, 1)`` noise. Dense leaves are unchanged."""
    import jax

    rng = np.random.default_rng(seed)
    keep = {p: set(int(i) for i in rows) for p, rows in reads.items()}

    def leaf(path, a):
        from bes.roles import path_str
        p = path_str(path)
        a = np.array(a, np.float32)
        if role_of(p).kind != "lookup":
            return a
        bad = np.ones(a.shape[0], bool)
        bad[list(keep.get(p, ()))] = False
        noise = rng.standard_normal(a.shape).astype(np.float32) * np.float32(scale)
        a[bad] += noise[bad]
        return a
    return jax.tree_util.tree_map_with_path(leaf, params)


# ------------------------------------------------------------------------------ logger
@dataclass
class ReadCount:
    """Distinct stored numbers read for one generated token.

    step: standard convention; strict: union over the visible context window; dense: the dense
    part of both; ``*_rows``: distinct rows per lookup table; extra: asset numbers (library,
    index) charged to this step.
    """
    step: int
    strict: int
    dense: int
    step_rows: dict[str, int] = field(default_factory=dict)
    strict_rows: dict[str, int] = field(default_factory=dict)
    extra: int = 0


class ReadLogger:
    """Counts distinct stored numbers read per generated token under both conventions.

    params: parameter tree or ``jax.eval_shape`` tree (only shapes are used); consts: runtime
    switches (default: the static flags). Call :meth:`log` once per position with the token and
    its feature row (as a feature provider returns it); the count is for the distribution
    computed at that position. A ``BOS`` (or ``new_segment=True``) starts a new window.
    """

    def __init__(self, params: Any, cfg, consts: Mapping | None = None, *,
                 include_critic: bool = False):
        self.cfg = cfg
        self.en = _enable(consts, cfg)
        leaves = dict(leaves_with_paths(params))
        self.dense = sum(math.prod(np.shape(leaves[p])) for p in
                         dense_paths(params, cfg, consts, include_critic=include_critic))
        self.widths = {p: math.prod(np.shape(a)[1:]) for p, a in leaves.items()
                       if role_of(p).kind == "lookup"}
        self.history: list[ReadCount] = []
        self.reset()

    def reset(self) -> None:
        """Start a new context window (segment)."""
        self._union: dict[str, set[int]] = {}
        self._extra_total = 0

    def log(self, token: int, row: Mapping | None = None, *, new_segment: bool = False,
            extra_numbers: int = 0) -> ReadCount:
        """Count the step that feeds ``token`` (with its feature ``row``: ``reg``,
        ``cand_tok``, ``cand_feat``) and produces the next distribution. extra_numbers: asset
        numbers read at this step (e.g. a library retrieval)."""
        row = {} if row is None else row
        sink = _is_sink(self.cfg, self.en, token)
        if new_segment or int(token) == special_id("<|bos|>", self.cfg.vocab_size):
            self.reset()
        here = position_reads(self.cfg, self.en, token, row.get("reg"), row.get("cand_tok"),
                              row.get("cand_feat"), inputs=True, outputs=False)
        out = position_reads(self.cfg, self.en, token, row.get("reg"), row.get("cand_tok"),
                             row.get("cand_feat"), inputs=False, outputs=True)
        step_rows: dict[str, set[int]] = {}
        _merge(step_rows, here)
        _merge(step_rows, out)
        strict_rows = {p: set(r) for p, r in self._union.items()}
        _merge(strict_rows, step_rows)
        if not sink:                       # a sink position is invisible to later positions
            _merge(self._union, here)
        self._extra_total += int(extra_numbers)
        n_step = {p: len(r) for p, r in step_rows.items()}
        n_strict = {p: len(r) for p, r in strict_rows.items()}
        count = ReadCount(
            step=self.dense + sum(n * self.widths[p] for p, n in n_step.items()) + int(extra_numbers),
            strict=self.dense + sum(n * self.widths[p] for p, n in n_strict.items()) + self._extra_total,
            dense=self.dense, step_rows=n_step, strict_rows=n_strict, extra=int(extra_numbers))
        self.history.append(count)
        return count

    def log_sequence(self, feats: Mapping) -> list[ReadCount]:
        """Log every position of one sequence (``tokens [T]`` plus feature rows; a change of
        ``seg`` starts a new window)."""
        f = _row_feats(feats)
        seg = f.get("seg")
        out = []
        for j in range(f["tokens"].shape[0]):
            new = seg is not None and j > 0 and seg[j] != seg[j - 1]
            row = {k: f[k][j] for k in ("reg", "cand_tok", "cand_feat") if k in f}
            out.append(self.log(int(f["tokens"][j]), row, new_segment=bool(new)))
        return out

    def summary(self) -> dict[str, float]:
        """Mean and maximum of both counts over the logged steps."""
        if not self.history:
            return {}
        s = np.array([c.step for c in self.history], np.float64)
        t = np.array([c.strict for c in self.history], np.float64)
        return {"steps": float(len(s)), "step_mean": float(s.mean()), "step_max": float(s.max()),
                "strict_mean": float(t.mean()), "strict_max": float(t.max()),
                "dense": float(self.dense)}


__all__ = ["position_reads", "visible_positions", "read_rows", "dense_paths", "lookup_paths",
           "RowStore", "reference_logits", "log_softmax", "kv_from_cache", "jax_last_logits",
           "max_logprob_diff", "poison", "ReadCount", "ReadLogger"]
