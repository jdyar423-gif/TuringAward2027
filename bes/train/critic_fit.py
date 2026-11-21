"""Fitting the critic head on the frozen core (plan section 4G; ``docs/MATH.md`` F3).

The critic ``critic(h) = relu(h @ W1) @ w2`` (``bes.model.critic``; leaves ``critic/w1 [d, H]``
and ``critic/w2 [H]``) reads the final hidden state of the frozen core. Once the hidden states
are known the fit is cheap, so the stage has two parts:

1. :func:`collect_hidden`: the hidden state ``forward_hidden(...)[b, pos_b]`` of every sampled
   conversation (``prompt + ASSISTANT + response + END``, see ``bes.data.prefs.row_tokens``) at
   its final token (any position can be requested), computed in fixed-shape batches with the
   exact-layer features. The result (:class:`HiddenCache`) stays in memory for all the fitting
   steps and can be saved with a fingerprint of the rows, the positions, the switches and the
   frozen (non-critic) parameters, so a stale file is never reused.
2. :func:`fit_critic`: a few hundred Adam steps on the cached states that change only the
   critic leaves: logistic regression on scalar qualities ``q in [0, 1]`` (binary cross-entropy
   with soft targets, ``softplus(s) - q s``) and/or a pairwise logistic ranking loss on
   preferences between two samples of the same prompt (``softplus(s_loser - s_winner)``), both
   supported at once, plus an L2 penalty on both critic matrices.

Report: the within-prompt Spearman rank correlation between critic scores and labels on
held-out prompts (a prompt is held out by a hash of its id and the seed). For each prompt with
at least two labelled samples whose labels are not all equal, rho is the Pearson correlation of
the average ranks (tied values share their mean rank); a prompt whose scores are all equal
counts rho = 0 (no ranking ability). The reported value is the unweighted mean over those
prompts. With pairwise labels only, a sample's label is its win rate among its comparisons.
The project uses the critic only when the held-out rho reaches ``CriticCfg.min_rho`` (0.3,
``report["use_critic"]``); the fitted parameters are returned either way.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import BOS, PAD, CriticCfg
from bes.model import model as M
from bes.model.critic import critic as critic_fn
from bes.optim import adam

F32 = jnp.float32
CRITIC = "critic"


def _model_cfg(cfg):
    return cfg.model if hasattr(cfg, "model") else cfg


# -------------------------------------------------------------------- hidden states
@dataclass
class HiddenCache:
    """``h float32[N, d]`` hidden states, ``valid bool[N]`` (False: the row did not fit or the
    position was outside it; its ``h`` is zero) and the ``fingerprint`` they were computed for."""

    h: np.ndarray
    valid: np.ndarray
    fingerprint: str

    def save(self, path: str | os.PathLike) -> None:
        """Write ``path`` (``.npz``) through a temporary file and a rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, h=self.h, valid=self.valid, fingerprint=np.asarray(self.fingerprint))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "HiddenCache":
        with np.load(path, allow_pickle=False) as z:
            return cls(np.asarray(z["h"], np.float32), np.asarray(z["valid"], bool),
                       str(z["fingerprint"]))


def params_fingerprint(params: dict, consts: dict | None = None) -> str:
    """Hash of every non-critic leaf (path, shape, float32 sum and sum of squares) and of the
    switches in ``consts``: a cheap identity of the frozen core."""
    h = hashlib.sha256()
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in flat:
        name = "/".join(str(getattr(p, "key", p)) for p in path)
        if name.split("/", 1)[0] == CRITIC:
            continue
        x = jnp.asarray(leaf, F32)
        s = np.asarray(jax.device_get(jnp.stack([jnp.sum(x), jnp.sum(x * x)])), np.float32)
        h.update(f"{name}:{tuple(np.shape(leaf))}:".encode())
        h.update(s.tobytes())
    if consts is not None:
        for leaf in jax.tree.leaves(jax.device_get(consts)):
            h.update(np.asarray(leaf, np.float32).tobytes())
    return h.hexdigest()[:32]


def rows_fingerprint(rows: Sequence[np.ndarray], positions: np.ndarray, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for r, p in zip(rows, positions):
        h.update(np.int64(len(r)).tobytes())
        h.update(np.asarray(r).astype("<i4").tobytes())
        h.update(np.int64(p).tobytes())
    return h.hexdigest()[:32]


@functools.lru_cache(maxsize=8)
def _final_hidden_fn(cfg) -> Callable:
    """jit ``(params, consts, feats, pos int32[B]) -> float32[B, d]``."""
    def fn(params, consts, feats, pos):
        h = M.forward_hidden(params, consts, feats, cfg)
        return jnp.take_along_axis(h, pos[:, None, None].astype(jnp.int32), axis=1)[:, 0]

    return jax.jit(fn)


def _default_features(tokens: np.ndarray, seg: np.ndarray, tok: Any) -> dict:
    from bes.exact.features import exact_features

    return dict(exact_features(tokens, seg, tok))


def collect_hidden(params: dict, consts: dict, rows: Sequence[Any], cfg, *,
                   seq_len: int = CriticCfg.seq_len, batch_rows: int = CriticCfg.batch_rows,
                   positions: Sequence[int] | None = None, tok: Any = None,
                   features: bool = True, feature_fn: Callable | None = None,
                   cache_path: str | os.PathLike | None = None) -> HiddenCache:
    """Final hidden states of ``rows`` (int token sequences, each starting with ``BOS``).

    positions: the position to read in each row (default: its last token). A row is invalid
    when it is empty, does not start with ``BOS``, is longer than ``seq_len`` or its position
    lies outside it. Rows are padded with ``PAD`` in a segment of their own (causal attention:
    padding never changes the state of a real position). cfg: ``ModelCfg`` or ``Cfg``.
    cache_path: reuse this file when its fingerprint matches, otherwise compute and write it.
    """
    mcfg = _model_cfg(cfg)
    arrs = [np.asarray(r).astype(np.int32).reshape(-1) for r in rows]
    n = len(arrs)
    lens = np.asarray([a.size for a in arrs], np.int64)
    pos = (lens - 1 if positions is None else np.asarray(positions, np.int64).reshape(-1))
    if pos.shape != (n,):
        raise ValueError("one position per row is required")
    first = np.asarray([int(a[0]) if a.size else -1 for a in arrs], np.int64)
    valid = (lens >= 1) & (lens <= seq_len) & (first == BOS) & (pos >= 0) & (pos < lens)
    extra = json.dumps({"cfg": asdict(mcfg), "seq_len": seq_len, "features": bool(features)},
                       sort_keys=True, default=str)
    fp = rows_fingerprint(arrs, pos, extra + params_fingerprint(params, consts))
    if cache_path is not None and Path(cache_path).exists():
        cached = HiddenCache.load(cache_path)
        if cached.fingerprint == fp and cached.h.shape == (n, mcfg.d_model):
            return cached
    if features and tok is None and feature_fn is None:
        from bes.tokenizer.tok import load

        tok = load()
    feature_fn = _default_features if feature_fn is None else feature_fn
    fn = _final_hidden_fn(mcfg)
    h = np.zeros((n, mcfg.d_model), np.float32)
    pending: list[tuple[np.ndarray, Any]] = []

    def drain(item):
        idx, out = item
        h[idx] = np.asarray(jax.device_get(out), np.float32)[:idx.size]

    for start in range(0, n, batch_rows):
        idx = np.arange(start, min(n, start + batch_rows))
        tokens = np.full((batch_rows, seq_len), PAD, np.int32)
        seg = np.ones((batch_rows, seq_len), np.int32)
        tokens[:, 0], seg[:, 0] = BOS, 0
        p = np.zeros(batch_rows, np.int32)
        for j, i in enumerate(idx.tolist()):
            if valid[i]:
                tokens[j, :lens[i]] = arrs[i]
                seg[j, :lens[i]] = 0
                p[j] = pos[i]
        feats: dict[str, Any] = {"tokens": tokens, "seg": seg}
        if features:
            f = feature_fn(tokens, seg, tok)
            feats.update(reg=np.asarray(f["reg"], np.int16), clock=np.asarray(f["clock"], np.int32),
                         cand_tok=np.asarray(f["cand_tok"], np.int32),
                         cand_feat=np.asarray(f["cand_feat"], np.int8))
        pending.append((idx, fn(params, consts, feats, p)))
        if len(pending) > 2:
            drain(pending.pop(0))
    for item in pending:
        drain(item)
    h[~valid] = 0.0
    cache = HiddenCache(h, valid, fp)
    if cache_path is not None:
        cache.save(cache_path)
    return cache


# ------------------------------------------------------------------------ statistics
def average_ranks(x: Any) -> np.ndarray:
    """``float64[n]`` ranks ``1..n`` of ``x``; tied values share the mean of their ranks."""
    x = np.asarray(x, np.float64).reshape(-1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(np.float64)
    return ((ends - counts + 1 + ends) / 2.0)[inv.reshape(-1)]


def spearman(x: Any, y: Any) -> float:
    """Spearman rank correlation with average ranks for ties (NaN if either side is constant
    or there are fewer than two values)."""
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.size < 2 or rx.size != ry.size:
        return float("nan")
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def within_prompt_spearman(scores: Any, labels: Any, groups: Any,
                           mask: Any = None) -> tuple[float, int, dict[Any, float]]:
    """Mean within-group Spearman correlation (module docstring rules).

    scores, labels: ``[N]`` (NaN label = unlabelled); groups: ``[N]`` group ids; mask: ``bool[N]``
    samples to use. Returns ``(mean rho or NaN, number of groups used, {group: rho})``.
    """
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(labels, np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    m = np.ones(s.size, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    m = m & np.isfinite(y)
    sel = np.flatnonzero(m)
    if sel.size == 0:
        return float("nan"), 0, {}
    order = sel[np.argsort(g[sel], kind="stable")]
    gs = g[order]
    cuts = np.flatnonzero(gs[1:] != gs[:-1]) + 1
    per: dict[Any, float] = {}
    for part in np.split(order, cuts):
        if part.size < 2 or np.all(y[part] == y[part[0]]):
            continue
        key = g[part[0]].item() if hasattr(g[part[0]], "item") else g[part[0]]
        per[key] = 0.0 if np.all(s[part] == s[part[0]]) else spearman(s[part], y[part])
    if not per:
        return float("nan"), 0, {}
    return float(np.mean(list(per.values()))), len(per), per


def win_rates(pairs: np.ndarray, n: int) -> np.ndarray:
    """``float64[n]``: wins / comparisons of every sample in ``pairs [P, 2]`` (winner, loser);
    NaN for samples without comparisons."""
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    wins = np.bincount(pairs[:, 0], minlength=n).astype(np.float64)
    games = wins + np.bincount(pairs[:, 1], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(games > 0, wins / np.maximum(games, 1), np.nan)


def heldout_groups(groups: Any, heldout_frac: float, seed: int) -> set:
    """Group ids held out: a group is held out when the BLAKE2b hash of ``(seed, id)`` falls
    below ``heldout_frac``; with two or more groups both sides are kept non-empty."""
    keys = {(k.item() if hasattr(k, "item") else k) for k in np.unique(np.asarray(groups))}
    if heldout_frac <= 0 or not keys:
        return set()

    def bucket(k) -> float:
        d = hashlib.blake2b(f"{seed}:{k!r}".encode(), digest_size=8).digest()
        return int.from_bytes(d, "little") / 2.0 ** 64

    b = {k: bucket(k) for k in keys}
    held = {k for k, v in b.items() if v < heldout_frac}
    if len(keys) >= 2 and not held:
        held = {min(b, key=b.__getitem__)}
    elif len(keys) >= 2 and len(held) == len(keys):
        held.discard(max(b, key=b.__getitem__))
    return held


# ------------------------------------------------------------------------------- fit
@functools.lru_cache(maxsize=16)
def _objective(has_scalar: bool, has_pairs: bool, pair_weight: float, l2: float) -> Callable:
    def objective(crit, h, si, q, pw, pl):
        total = jnp.zeros((), F32)
        if has_scalar:
            s = critic_fn(crit, h[si])
            total = total + jnp.mean(jax.nn.softplus(s) - q * s)
        if has_pairs:
            delta = critic_fn(crit, h[pw]) - critic_fn(crit, h[pl])
            total = total + pair_weight * jnp.mean(jax.nn.softplus(-delta))
        if l2:
            total = total + l2 * sum(jnp.sum(jnp.square(v)) for v in jax.tree.leaves(crit))
        return total

    return objective


@functools.lru_cache(maxsize=16)
def _fit_fns(has_scalar: bool, has_pairs: bool, pair_weight: float, l2: float, b1: float,
             b2: float, eps: float) -> tuple[Callable, Callable]:
    objective = _objective(has_scalar, has_pairs, pair_weight, l2)

    def step(crit, state, t, lr, h, si, q, pw, pl):
        loss, g = jax.value_and_grad(objective)(crit, h, si, q, pw, pl)
        new, new_state = {}, {}
        for k in crit:
            new[k], new_state[k] = adam.update(crit[k], g[k], state[k], lr, t, b1=b1, b2=b2,
                                               eps=eps)
        return new, new_state, loss

    return jax.jit(step), jax.jit(objective)


def _scores(crit: dict, h: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = [np.asarray(jax.device_get(critic_fn(crit, jnp.asarray(h[i:i + chunk]))), np.float64)
           for i in range(0, h.shape[0], chunk)]
    return np.concatenate(out) if out else np.zeros(0, np.float64)


def _batch(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    if size <= 0 or n <= size:
        return np.arange(n, dtype=np.int32)
    return np.sort(rng.choice(n, size=size, replace=False)).astype(np.int32)


def fit_critic(params: dict, hidden: Any, *, groups: Any, quality: Any = None, pairs: Any = None,
               valid: Any = None, ccfg: CriticCfg | None = None) -> tuple[dict, dict]:
    """Fit ``params["critic"]`` on cached hidden states; every other leaf is returned as is.

    hidden: ``float32[N, d]``; groups: ``[N]`` prompt id of every sample; quality: ``[N]`` scalar
    labels in ``[0, 1]`` (NaN = unlabelled) or None; pairs: ``int[P, 2]`` (winner, loser) sample
    indices of the same prompt or None; valid: ``bool[N]`` samples to use (default all).
    Returns ``(params, report)``; see the module docstring for the report's correlation.
    """
    ccfg = CriticCfg() if ccfg is None else ccfg
    h = np.asarray(hidden, np.float32)
    if h.ndim != 2:
        raise ValueError("hidden must be [N, d]")
    n, d = h.shape
    crit0 = {k: jnp.asarray(v, F32) for k, v in params[CRITIC].items()}
    if crit0["w1"].shape[0] != d:
        raise ValueError(f"hidden width {d} does not match critic/w1 {crit0['w1'].shape}")
    g = np.asarray(groups).reshape(-1)
    if g.shape != (n,):
        raise ValueError("one group id per sample is required")
    ok = np.ones(n, bool) if valid is None else np.asarray(valid, bool).reshape(-1)
    q = np.full(n, np.nan) if quality is None else np.asarray(quality, np.float64).reshape(-1)
    if q.shape != (n,):
        raise ValueError("one quality per sample is required")
    fin = np.isfinite(q)
    if np.any((q[fin] < 0) | (q[fin] > 1)):
        raise ValueError("scalar qualities must lie in [0, 1]")
    pr = np.zeros((0, 2), np.int64) if pairs is None else np.asarray(pairs, np.int64).reshape(-1, 2)
    if pr.size and (pr.min() < 0 or pr.max() >= n):
        raise ValueError("pair index outside the samples")
    good = (pr[:, 0] != pr[:, 1]) & (g[pr[:, 0]] == g[pr[:, 1]]) & ok[pr[:, 0]] & ok[pr[:, 1]]
    n_bad_pairs = int((~good).sum())
    pr = pr[good]
    held_ids = heldout_groups(g[ok], ccfg.heldout_frac, ccfg.seed)
    held = np.isin(g, np.asarray(sorted(held_ids, key=repr), dtype=g.dtype)) if held_ids else np.zeros(n, bool)
    tr_mask, ho_mask = ok & ~held, ok & held
    s_idx = np.flatnonzero(tr_mask & fin)
    p_tr = pr[tr_mask[pr[:, 0]] & tr_mask[pr[:, 1]]]
    p_ho = pr[ho_mask[pr[:, 0]] & ho_mask[pr[:, 1]]]
    has_s, has_p = s_idx.size > 0, p_tr.shape[0] > 0
    if not (has_s or has_p):
        raise ValueError("no training labels: give scalar qualities or pairs on training prompts")

    step, objective = _fit_fns(bool(has_s), bool(has_p), float(ccfg.pair_weight), float(ccfg.l2),
                               float(ccfg.adam_b1), float(ccfg.adam_b2), float(ccfg.adam_eps))
    hd = jnp.asarray(h)
    q32 = np.nan_to_num(q).astype(np.float32)
    dummy = np.zeros(1, np.int32)

    def args(si, pi):
        return (hd, jnp.asarray(s_idx[si] if has_s else dummy, jnp.int32),
                jnp.asarray(q32[s_idx[si]] if has_s else np.zeros(1, np.float32)),
                jnp.asarray(p_tr[pi, 0] if has_p else dummy, jnp.int32),
                jnp.asarray(p_tr[pi, 1] if has_p else dummy, jnp.int32))

    all_s = np.arange(s_idx.size, dtype=np.int32)
    all_p = np.arange(p_tr.shape[0], dtype=np.int32)
    loss_start = float(objective(crit0, *args(all_s, all_p)))
    rng = np.random.default_rng(ccfg.seed)
    crit = dict(crit0)
    state = {k: adam.init(v) for k, v in crit.items()}
    lr = np.float32(ccfg.lr)
    for t in range(1, ccfg.steps + 1):
        si = _batch(rng, s_idx.size, ccfg.batch) if has_s else dummy
        pi = _batch(rng, p_tr.shape[0], ccfg.pair_batch) if has_p else dummy
        crit, state, _ = step(crit, state, np.float32(t), lr, *args(si, pi))
    loss_end = float(objective(crit, *args(all_s, all_p)))

    scores = _scores(crit, h)
    labels = q if fin.any() else win_rates(pr, n)
    rho_ho, n_ho, per_ho = within_prompt_spearman(scores, labels, g, ho_mask)
    rho_tr, n_tr, _ = within_prompt_spearman(scores, labels, g, tr_mask)

    def pair_acc(p):
        if p.shape[0] == 0:
            return float("nan")
        diff = scores[p[:, 0]] - scores[p[:, 1]]
        return float(np.mean((diff > 0) + 0.5 * (diff == 0)))

    finite = bool(np.isfinite(scores[ok]).all()) and np.isfinite(loss_end)
    use = bool(finite and np.isfinite(rho_ho) and rho_ho >= ccfg.min_rho)
    report = {
        "rho_heldout": rho_ho, "n_heldout_groups": n_ho, "rho_train": rho_tr,
        "n_train_groups": n_tr, "use_critic": use, "min_rho": ccfg.min_rho,
        "pair_acc_heldout": pair_acc(p_ho), "pair_acc_train": pair_acc(p_tr),
        "n_heldout_pairs": int(p_ho.shape[0]), "n_train_pairs": int(p_tr.shape[0]),
        "n_scalar_train": int(s_idx.size), "n_dropped_pairs": n_bad_pairs,
        "labels": "scalar+pairs" if (fin.any() and pr.size) else ("scalar" if fin.any() else "pairs"),
        "rho_labels": "quality" if fin.any() else "win_rate",
        "loss_start": loss_start, "loss_end": loss_end, "steps": ccfg.steps, "finite": finite,
        "heldout_groups": sorted(per_ho, key=repr), "rho_per_heldout_group": per_ho,
    }
    new_params = dict(params)
    new_params[CRITIC] = {k: jnp.asarray(v, params[CRITIC][k].dtype) for k, v in crit.items()}
    return new_params, report


def run_critic_fit(params: dict, consts: dict, rows: Sequence[Any], cfg, *, groups: Any,
                   quality: Any = None, pairs: Any = None, positions: Sequence[int] | None = None,
                   ccfg: CriticCfg | None = None, tok: Any = None, features: bool = True,
                   feature_fn: Callable | None = None,
                   cache_path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    """:func:`collect_hidden` (cached) followed by :func:`fit_critic`; rows that do not fit
    ``ccfg.seq_len`` are left out (``report["n_invalid_rows"]``)."""
    ccfg = CriticCfg() if ccfg is None else ccfg
    cache = collect_hidden(params, consts, rows, cfg, seq_len=ccfg.seq_len,
                           batch_rows=ccfg.batch_rows, positions=positions, tok=tok,
                           features=features, feature_fn=feature_fn, cache_path=cache_path)
    new_params, report = fit_critic(params, cache.h, groups=groups, quality=quality, pairs=pairs,
                                    valid=cache.valid, ccfg=ccfg)
    report["n_invalid_rows"] = int((~cache.valid).sum())
    return new_params, report
