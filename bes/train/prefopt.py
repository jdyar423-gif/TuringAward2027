"""Preference optimisation: one epoch of DPO against a frozen reference, with health checks.

Objective (DPO; Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model"). For a pair (prompt x, chosen y_c, rejected y_r), policy parameters
theta and the frozen reference theta0::

    margin = (log pi_theta(y_c|x) - log pi_theta0(y_c|x)) - (log pi_theta(y_r|x) - log pi_theta0(y_r|x))
    loss   = -log sigmoid(beta * margin) = softplus(-beta * margin)

``log pi(y|x)`` is the weighted sum of the response-token log-probabilities under the full model,
the candidate tape's output law included (the weights come from ``bes.data.prefs``), see
:func:`row_logps`. The
reference terms are computed once before training (:func:`compute_ref_logps`) and stored with
the pairs (``PairBatches.ref``), so a training step runs the policy only. At theta = theta0
every margin is 0 and the loss is ``log 2``.

Step (:func:`make_dpo_step`): one jit-compiled function that scans (``lax.scan``) over the ``A``
micro-batches of a batch, accumulating the gradient of the *summed* pair losses, divides by the
number of real pairs (the gradient of the mean loss over the whole batch) and applies one
optimiser update; a non-finite loss or gradient skips the update in-graph. Parameters and
optimiser state are donated.

Optimiser (:func:`build_optimizer`): plain Adam with one learning rate for every trainable leaf,
expressed with the project's builder (``bes.optim.build``) and a role function that maps every
trainable leaf to the Adam ``scalar`` group and every critic leaf (``critic/...``) or leaf whose
role group is ``frozen`` to the ``frozen`` group (never updated). The group rate is made exactly
1 (``lr_scalar = 1``, ``lr_width_ref = d_model``) and the actual rate arrives as the traced
``lr_scale``, so one compiled step serves every candidate learning rate and the schedule. Betas,
epsilon and the global-norm clip come from ``PrefCfg``. The spectral optimiser is not used: the
published recipe at this scale is Adam on all weights at 1e-6..3e-6 for one epoch, which moves
any matrix norm negligibly.

Schedule (:func:`lr_factor`): one epoch in a fixed shuffled order; linear warm-up over
``warmup_frac`` of the steps, then linear decay towards ``final_lr_frac``. More than one epoch
degrades quality at this scale, so there is no epoch count.

Health checks and selection (:func:`run_prefopt`). Every candidate learning rate starts from the
same parameters, with a fresh optimiser state and the same data order, and is accepted only if

(a) the held-out preference accuracy improved: the share of held-out pairs whose margin is
    positive (ties count one half) exceeds the start value plus ``min_acc_gain``. At the start
    the policy is the reference, every margin is 0 and the value is exactly 0.5, so the check
    asks that the trained policy ranks unseen pairs better than chance in the labelled direction;
(b) the held-out assistant-token loss from ``heldout_loss_fn(params) -> float`` rose by at most
    ``max_loss_rise`` (3 %) over the start value;
(c) the sampling statistics from ``sample_stats_fn(params) -> {"mean_len", "rep4"}`` stay in
    range: ``len_ratio_min <= mean_len / start <= len_ratio_max`` (0.7..1.6) and
    ``rep4 <= rep_ratio_max * start`` (1.2); :func:`response_stats` computes both from sampled
    token lists.

Checks run in this order and stop at the first failure unless ``evaluate_all`` (sampling is the
expensive one). The accepted candidate with the best held-out preference accuracy is returned
(ties: the earlier learning rate); if none is accepted the starting parameters are returned and
the report says why.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import Cfg, PrefCfg
from bes.data.prefs import PairBatches, PrefSplit, shuffled_order
from bes.model import model as M
from bes.optim.build import build as _build_optimizer
from bes.optim.build import global_norm, path_str
from bes.roles import role_of

F32 = jnp.float32
STAT_LEN = "mean_len"
STAT_REP = "rep4"
#: batch keys the device functions read (``index`` stays on the host)
DEVICE_KEYS = ("tokens", "w", "seg", "pos", "reg", "clock", "cand_tok", "cand_feat", "ref",
               "pair_mask")


# ------------------------------------------------------------------- log-probabilities
def token_logps(params: dict, consts: dict, batch: Mapping[str, Any], cfg) -> jax.Array:
    """``float32[B, T]``: log-probability of every target ``tokens[:, 1:]`` under the full model.

    batch: one micro-batch in the SPEC section 3 layout without the ``A`` axis (``tokens
    [B, T+1]`` and the optional features). Uses the model's public pieces exactly as
    ``loss_fn`` does -- ``forward_hidden`` and ``token_ce`` (one implementation of the
    per-token cross-entropy) -- so the vocabulary projection runs over row chunks of
    ``cfg.loss_chunk`` (one float32 ``[chunk, V]`` block live at a time, in the forward and the
    backward) and the candidate tape's output law is included.
    """
    feats, y, _ = M.split_batch(dict(batch))
    h = M.forward_hidden(params, consts, feats, cfg)
    return -M.token_ce(params, consts, h, feats, y, cfg).reshape(y.shape)


def row_logps(params: dict, consts: dict, batch: Mapping[str, Any], cfg) -> jax.Array:
    """``float32[B]``: ``sum_t w[b, t] * log p(tokens[b, t+1] | tokens[b, :t+1])`` (see
    :func:`token_logps` for the batch layout and the chunking)."""
    w = jnp.asarray(batch["w"], F32)
    return jnp.sum(w * token_logps(params, consts, batch, cfg), axis=-1)


def _device_part(batch: Mapping[str, Any]) -> dict:
    return {k: batch[k] for k in DEVICE_KEYS if k in batch and batch[k] is not None}


@functools.lru_cache(maxsize=16)
def make_logp_fn(cfg) -> Callable:
    """jit ``(params, consts, batch [A, B, ...]) -> float32[A, B]``: :func:`row_logps` of every
    micro-batch (``lax.map`` over ``A``). Cached per model configuration."""
    def fn(params, consts, batch):
        return jax.lax.map(lambda mb: row_logps(params, consts, mb, cfg), batch)

    jitted = jax.jit(fn)
    return lambda params, consts, batch: jitted(params, consts, _device_part(batch))


# ---------------------------------------------------------------------------- DPO terms
def dpo_terms(logp: jax.Array, ref: jax.Array, beta: jax.Array | float):
    """Pair losses and margins from interleaved rows (chosen ``2p``, rejected ``2p + 1``).

    logp, ref: ``float32[..., 2P]`` policy and reference row log-probabilities.
    Returns ``(loss float32[..., P], margin float32[..., P])``.
    """
    margin = (logp[..., 0::2] - ref[..., 0::2]) - (logp[..., 1::2] - ref[..., 1::2])
    return jax.nn.softplus(-jnp.asarray(beta, F32) * margin), margin


def _won(margin):
    """1 for a positive margin, 1/2 for a tie, 0 otherwise (numpy or jax)."""
    return (margin > 0) * 1.0 + (margin == 0) * 0.5


def dpo_sums(params: dict, consts: dict, mb: Mapping[str, Any], beta, cfg):
    """Summed DPO loss of the real pairs of one micro-batch and auxiliary sums
    (``n`` pairs, ``loss``, ``won`` (accuracy numerator), ``margin``)."""
    logp = row_logps(params, consts, mb, cfg)
    loss, margin = dpo_terms(logp, jnp.asarray(mb["ref"], F32), beta)
    mask = jnp.asarray(mb["pair_mask"], F32)
    total = jnp.sum(mask * loss)
    aux = {"n": jnp.sum(mask), "loss": total, "won": jnp.sum(mask * _won(margin)),
           "margin": jnp.sum(mask * margin)}
    return total, aux


def dpo_grads(params: dict, consts: dict, batch: Mapping[str, Any], beta, cfg):
    """Gradient of the mean pair loss over all micro-batches of ``batch`` (``[A, B, ...]``).

    Returns ``(grads float32 tree, sums)`` where ``sums`` adds the auxiliaries of
    :func:`dpo_sums` over the micro-batches; ``grads = sum_i grad(L_i) / n`` with ``L_i`` the
    summed loss of micro-batch ``i`` and ``n`` the number of real pairs.
    """
    vg = jax.value_and_grad(dpo_sums, has_aux=True)

    def body(gsum, mb):
        (_, aux), g = vg(params, consts, mb, beta, cfg)
        return jax.tree.map(lambda a, b: a + b.astype(F32), gsum, g), aux

    zeros = jax.tree.map(lambda p: jnp.zeros(np.shape(p), F32), params)
    gsum, aux = jax.lax.scan(body, zeros, dict(batch))
    sums = jax.tree.map(jnp.sum, aux)
    inv = 1.0 / jnp.maximum(sums["n"], 1.0)
    return jax.tree.map(lambda g: g * inv, gsum), sums


# --------------------------------------------------------------------------- optimiser
def pref_role_fn(base_role_fn: Callable[[str], Any] | None = None) -> Callable[[str], str]:
    """Role function for the DPO optimiser: ``"frozen"`` for ``critic/...`` and for every leaf
    whose group under ``base_role_fn`` (default ``bes.roles.role_of``) is ``frozen``, ``"scalar"``
    (one Adam group) for every other leaf."""
    base = role_of if base_role_fn is None else base_role_fn

    def fn(path: str) -> str:
        role = base(path)
        group = getattr(role, "group", role)
        if path.split("/", 1)[0] == "critic" or group == "frozen":
            return "frozen"
        return "scalar"

    return fn


def build_optimizer(cfg: Cfg, params: Any, pref: PrefCfg | None = None,
                    role_fn: Callable[[str], Any] | None = None):
    """``bes.optim.Optimizer`` doing plain Adam (``PrefCfg`` betas, epsilon, clip) with group
    rate exactly 1 on every trainable leaf; pass the learning rate as ``lr_scale``."""
    pref = PrefCfg() if pref is None else pref
    optim = dataclasses.replace(cfg.optim, lr_scalar=1.0, lr_width_ref=cfg.model.d_model,
                                adam_b1=pref.adam_b1, adam_b2=pref.adam_b2,
                                adam_eps=pref.adam_eps, grad_clip=pref.grad_clip)
    return _build_optimizer(dataclasses.replace(cfg, optim=optim), params,
                            pref_role_fn(role_fn), check_init=False)


def trainable_paths(params: Any, role_fn: Callable[[str], Any] | None = None) -> list[str]:
    """Paths of the leaves the DPO optimiser updates."""
    fn = pref_role_fn(role_fn)
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return [p for p in (path_str(k) for k, _ in flat) if fn(p) != "frozen"]


@functools.lru_cache(maxsize=16)
def make_dpo_step(cfg, optimizer, *, donate: bool = True, skip_nonfinite: bool = True) -> Callable:
    """jit ``step(params, opt_state, consts, batch, lr, beta) -> (params, opt_state, metrics)``.

    cfg: ``ModelCfg``; optimizer: from :func:`build_optimizer`; batch: ``[A, B, ...]`` with
    ``ref`` and ``pair_mask`` (``bes.data.prefs.PairBatches``); lr: this step's learning rate
    (traced); beta: DPO temperature (traced). Metrics (device scalars): ``loss`` (mean pair loss
    at the parameters *before* the update), ``acc`` (share of positive margins, ties 1/2),
    ``margin`` (mean), ``pairs``, ``grad_norm`` (of the leaves the optimiser updates, before
    clipping), ``skipped`` (1 when the update was skipped because the loss or that gradient was
    not finite). Cached per (cfg, optimizer).
    """
    updated = tuple(lp.group != "frozen" for lp in optimizer.plan)   # tree-leaf order

    def step(params, opt_state, consts, batch, lr, beta):
        grads, sums = dpo_grads(params, consts, batch, beta, cfg)
        inv = 1.0 / jnp.maximum(sums["n"], 1.0)
        loss = sums["loss"] * inv
        gnorm = global_norm([g for g, u in zip(jax.tree.leaves(grads), updated) if u])
        new_params, new_state = optimizer.update(grads, opt_state, params, jnp.asarray(lr, F32))
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        if skip_nonfinite:
            new_params = jax.tree.map(lambda a, b: jnp.where(ok, a, b), new_params, params)
            new_state = jax.tree.map(lambda a, b: jnp.where(ok, a, b), new_state, opt_state)
        metrics = {"loss": loss, "acc": sums["won"] * inv, "margin": sums["margin"] * inv,
                   "pairs": sums["n"], "grad_norm": gnorm,
                   "skipped": (~ok if skip_nonfinite else jnp.zeros((), bool)).astype(jnp.int32)}
        return new_params, new_state, metrics

    jitted = jax.jit(step, donate_argnums=(0, 1) if donate else ())

    def run(params, opt_state, consts, batch, lr, beta):
        return jitted(params, opt_state, consts, _device_part(batch), np.float32(lr),
                      np.float32(beta))

    return run


def lr_factor(step: int, n_steps: int, warmup_frac: float = PrefCfg.warmup_frac,
              final_frac: float = PrefCfg.final_lr_frac) -> float:
    """Learning-rate multiplier of step ``step`` (0-based) of an epoch of ``n_steps``: linear
    warm-up to 1 over ``ceil(warmup_frac * n_steps)`` steps, then linear decay that would reach
    ``final_frac`` one step after the last (so no step has a zero rate unless warm-up is 0)."""
    if n_steps <= 0:
        return 0.0
    warm = int(math.ceil(warmup_frac * n_steps)) if warmup_frac > 0 else 0
    if step < warm:
        return (step + 1) / warm
    span = n_steps - warm
    if span <= 0:
        return 1.0
    return 1.0 - (1.0 - final_frac) * (step - warm) / span


# ------------------------------------------------------------------------- data passes
def pair_logps(params: dict, consts: dict, data: PairBatches, cfg,
               logp_fn: Callable | None = None, in_flight: int = 2) -> np.ndarray:
    """``float32[n_pairs, 2]`` (chosen, rejected) log-probabilities of every pair of ``data``.

    At most ``in_flight`` batches are dispatched ahead of the one being read, which bounds the
    device memory taken by queued inputs while the host builds the next batch.
    """
    logp_fn = make_logp_fn(cfg) if logp_fn is None else logp_fn
    out = np.full((data.n_pairs, 2), np.nan, np.float32)
    pending: list[tuple[np.ndarray, Any]] = []

    def drain(item):
        idx, lp = item
        lp = np.asarray(jax.device_get(lp), np.float32)        # [A, B]
        ok = idx >= 0
        out[idx[ok], 0] = lp[:, 0::2][ok]
        out[idx[ok], 1] = lp[:, 1::2][ok]

    for k in range(len(data)):
        batch = data.get(k)
        pending.append((batch["index"], logp_fn(params, consts, batch)))
        if len(pending) > in_flight:
            drain(pending.pop(0))
    for item in pending:
        drain(item)
    return out


def compute_ref_logps(params: dict, consts: dict, data: PairBatches, cfg,
                      logp_fn: Callable | None = None) -> np.ndarray:
    """Reference log-probabilities of every pair, computed once and stored in ``data``
    (:meth:`PairBatches.set_ref`). Returns them (``float32[n_pairs, 2]``)."""
    ref = pair_logps(params, consts, data, cfg, logp_fn)
    data.set_ref(ref)
    return ref


def preference_metrics(logp: np.ndarray, ref: np.ndarray, counts: np.ndarray | None = None,
                       beta: float = PrefCfg.beta) -> dict[str, float]:
    """Held-out metrics of policy log-probabilities against the stored reference.

    logp, ref: ``[n, 2]`` (chosen, rejected); counts: ``[n, 2]`` weighted targets per row.
    ``acc``: share of positive margins (ties 1/2; the selection metric); ``loss``: mean DPO
    loss; ``margin``: mean margin; ``policy_acc``: share of pairs whose chosen row has the
    higher policy log-probability (ties 1/2); ``policy_acc_per_token``: the same per target
    token (with ``counts``); ``n``: pairs.
    """
    lp, rf = np.asarray(logp, np.float64), np.asarray(ref, np.float64)
    if lp.shape != rf.shape or lp.ndim != 2 or lp.shape[1] != 2:
        raise ValueError("logp and ref must both have shape [n, 2]")
    n = lp.shape[0]
    out: dict[str, float] = {"n": float(n)}
    if n == 0:
        return {**out, "acc": float("nan"), "loss": float("nan"), "margin": float("nan"),
                "policy_acc": float("nan"), "policy_acc_per_token": float("nan")}
    margin = (lp[:, 0] - rf[:, 0]) - (lp[:, 1] - rf[:, 1])
    out["acc"] = float(np.mean(_won(margin)))
    out["loss"] = float(np.mean(np.logaddexp(0.0, -beta * margin)))
    out["margin"] = float(np.mean(margin))
    out["policy_acc"] = float(np.mean(_won(lp[:, 0] - lp[:, 1])))
    if counts is not None:
        c = np.maximum(np.asarray(counts, np.float64), 1.0)
        out["policy_acc_per_token"] = float(np.mean(_won(lp[:, 0] / c[:, 0] - lp[:, 1] / c[:, 1])))
    else:
        out["policy_acc_per_token"] = float("nan")
    return out


def _tree_finite(tree: Any) -> bool:
    leaves = jax.tree.leaves(tree)
    return bool(all(bool(jax.device_get(jnp.all(jnp.isfinite(x)))) for x in leaves))


def train_epoch(params: dict, consts: dict, data: PairBatches, cfg: Cfg, lr: float,
                pref: PrefCfg | None = None, *, optimizer=None, step_fn: Callable | None = None,
                role_fn: Callable[[str], Any] | None = None,
                log: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """One DPO epoch over ``data`` (which must hold its reference log-probabilities) from
    ``params`` (not modified: training runs on a copy). Returns ``(params, report)`` with the
    per-epoch means of the step metrics, the number of steps and of skipped updates;
    ``report["finite"]`` is False when any update was skipped or a parameter is not finite
    (a candidate that met a non-finite loss is not trusted). At most two steps run ahead of the
    host, so queued input batches cannot pile up on the device."""
    pref = PrefCfg() if pref is None else pref
    if data.ref is None:
        raise ValueError("the reference log-probabilities are not set; call compute_ref_logps first")
    optimizer = build_optimizer(cfg, params, pref, role_fn) if optimizer is None else optimizer
    step_fn = make_dpo_step(cfg.model, optimizer) if step_fn is None else step_fn
    t0 = time.time()
    params = jax.tree.map(jnp.copy, params)       # donation must not consume the caller's buffers
    opt_state = optimizer.init(params)
    n_steps = len(data)
    history = []
    for k in range(n_steps):
        factor = lr_factor(k, n_steps, pref.warmup_frac, pref.final_lr_frac)
        params, opt_state, m = step_fn(params, opt_state, consts, data.get(k), lr * factor,
                                       pref.beta)
        history.append(m)
        if k >= 2:
            jax.block_until_ready(history[k - 2]["loss"])
        if log is not None and (k + 1) % 50 == 0:
            log(f"dpo lr={lr:g} step {k + 1}/{n_steps}")
    host = jax.device_get(history)
    report: dict[str, Any] = {"lr": float(lr), "steps": n_steps, "time_s": 0.0}
    if host:
        pairs = np.asarray([h["pairs"] for h in host], np.float64)
        wsum = max(float(pairs.sum()), 1.0)
        for key in ("loss", "acc", "margin"):
            vals = np.asarray([h[key] for h in host], np.float64)
            report[f"train_{key}"] = float(np.sum(vals * pairs) / wsum)
        report["first_loss"] = float(host[0]["loss"])
        report["last_loss"] = float(host[-1]["loss"])
        report["max_grad_norm"] = float(max(float(h["grad_norm"]) for h in host))
        report["skipped"] = int(sum(int(h["skipped"]) for h in host))
    else:
        report.update(skipped=0)
    report["finite"] = bool(_tree_finite(params)) and report["skipped"] == 0
    report["time_s"] = time.time() - t0
    return params, report


# ----------------------------------------------------------------------- health checks
def response_stats(responses: Sequence[Sequence[int]]) -> dict[str, float]:
    """Sampling statistics of token-id responses: ``mean_len`` (mean number of tokens) and
    ``rep4`` (repeated 4-gram rate: among all 4-gram occurrences, the share that repeats a
    4-gram seen earlier in the same response; 0 when no response has 4 tokens)."""
    lengths, repeated, total = [], 0, 0
    for r in responses:
        ids = [int(t) for t in r]
        lengths.append(len(ids))
        seen: set[tuple[int, ...]] = set()
        for i in range(len(ids) - 3):
            g = tuple(ids[i:i + 4])
            total += 1
            if g in seen:
                repeated += 1
            else:
                seen.add(g)
    return {STAT_LEN: float(np.mean(lengths)) if lengths else 0.0,
            STAT_REP: repeated / total if total else 0.0}


def check_accuracy(start: Mapping[str, Any], cand: Mapping[str, Any],
                   pref: PrefCfg) -> list[tuple[str, str]]:
    a0, a1 = float(start["pref"]["acc"]), float(cand["pref"]["acc"])
    if a1 > a0 + pref.min_acc_gain:
        return []
    return [("accuracy", f"held-out preference accuracy {a1:.4f} does not exceed the start "
                         f"value {a0:.4f} + {pref.min_acc_gain:g}")]


def check_loss(start: Mapping[str, Any], cand: Mapping[str, Any],
               pref: PrefCfg) -> list[tuple[str, str]]:
    l0, l1 = float(start["heldout_loss"]), float(cand["heldout_loss"])
    if math.isfinite(l0) and math.isfinite(l1) and l1 <= l0 * (1.0 + pref.max_loss_rise):
        return []
    return [("loss", f"held-out assistant loss {l1:.5f} is more than "
                     f"{100 * pref.max_loss_rise:g}% above the start value {l0:.5f}")]


def check_stats(start: Mapping[str, Any], cand: Mapping[str, Any],
                pref: PrefCfg) -> list[tuple[str, str]]:
    s0, s1 = start["stats"], cand["stats"]
    out = []
    n0, n1 = float(s0[STAT_LEN]), float(s1[STAT_LEN])
    if not pref.len_ratio_min * n0 <= n1 <= pref.len_ratio_max * n0:
        out.append(("length", f"mean response length {n1:.2f} outside "
                              f"[{pref.len_ratio_min:g}, {pref.len_ratio_max:g}] x {n0:.2f}"))
    r0, r1 = float(s0[STAT_REP]), float(s1[STAT_REP])
    if not r1 <= pref.rep_ratio_max * r0:
        out.append(("repetition", f"repeated 4-gram rate {r1:.4f} above "
                                  f"{pref.rep_ratio_max:g} x {r0:.4f}"))
    return out


def check_candidate(start: Mapping[str, Any], cand: Mapping[str, Any],
                    pref: PrefCfg | None = None) -> list[tuple[str, str]]:
    """All failed checks ``[(code, message)]`` of a fully evaluated candidate (empty: accept).

    start / cand: ``{"pref": preference_metrics(...), "heldout_loss": float, "stats":
    {"mean_len", "rep4"}}``; cand may carry ``"finite": False``.
    """
    pref = PrefCfg() if pref is None else pref
    if not cand.get("finite", True):
        return [("nonfinite", "training produced non-finite parameters or losses")]
    return check_accuracy(start, cand, pref) + check_loss(start, cand, pref) + check_stats(
        start, cand, pref)


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> int | None:
    """Index of the accepted candidate with the best held-out preference accuracy (the first
    one on ties), or ``None`` when no candidate was accepted."""
    best = None
    for i, c in enumerate(candidates):
        if c.get("accepted") and (best is None or c["pref"]["acc"] > candidates[best]["pref"]["acc"]):
            best = i
    return best


def make_heldout_loss_fn(batches: Sequence[Mapping[str, Any]], consts: dict, cfg) -> Callable:
    """``fn(params) -> float``: weighted mean cross-entropy (``model.loss_fn``) over ``batches``
    -- SPEC section 3 micro-batches ``[B, T+1]`` or batches with the ``A`` axis -- whose ``w``
    should select the assistant tokens of held-out conversations."""
    one = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, cfg)[1])
    many = jax.jit(lambda p, c, b: jax.lax.map(lambda mb: M.loss_fn(p, c, mb, cfg)[1], b))
    keys = ("tokens", "w", "seg", "pos", "reg", "clock", "cand_tok", "cand_feat")

    def fn(params) -> float:
        ce, weight = 0.0, 0.0
        for b in batches:
            part = {k: b[k] for k in keys if k in b and b[k] is not None}
            m = (many if np.ndim(part["tokens"]) == 3 else one)(params, consts, part)
            m = jax.device_get(m)
            ce += float(np.sum(m["ce_sum"]))
            weight += float(np.sum(m["weight"]))
        return ce / weight if weight > 0 else float("nan")

    return fn


# ----------------------------------------------------------------------------- driver
def _as_split(pairs: Any):
    if isinstance(pairs, PrefSplit):
        return list(pairs.train), list(pairs.heldout)
    train, heldout = pairs
    return list(train), list(heldout)


def _stats(values: Mapping[str, Any]) -> dict[str, float]:
    """Sampling statistics as floats; ``mean_len`` and ``rep4`` are required."""
    values = dict(values)
    missing = [k for k in (STAT_LEN, STAT_REP) if k not in values]
    if missing:
        raise KeyError(f"sample_stats_fn must return {STAT_LEN!r} and {STAT_REP!r}; missing {missing}")
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float, np.number))
            or np.ndim(v) == 0}


def run_prefopt(params: dict, pairs: Any, cfg: Cfg, *, lrs: Sequence[float] = (1e-6, 3e-6),
                heldout_loss_fn: Callable[[Any], float],
                sample_stats_fn: Callable[[Any], Mapping[str, float]],
                pref: PrefCfg | None = None, consts: dict | None = None, tok: Any = None,
                features: bool = True, feature_fn: Callable | None = None,
                role_fn: Callable[[str], Any] | None = None, evaluate_all: bool = False,
                log: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """DPO with health checks and selection (module docstring).

    params: starting parameters (also the frozen reference). pairs: a
    ``bes.data.prefs.PrefSplit`` or ``(train_pairs, heldout_pairs)`` (filtered, so that every
    row fits ``pref.seq_len``). cfg: the full configuration (``cfg.model`` for the model,
    ``cfg.optim`` as the base of the optimiser). lrs: candidate peak learning rates, each tried
    from ``params``. heldout_loss_fn / sample_stats_fn: injected health measurements (see
    :func:`make_heldout_loss_fn`, :func:`response_stats`). consts: feature switches (default
    ``make_consts(cfg.model)``). tok / features / feature_fn: exact features of the rows (see
    ``PairBatches``). role_fn: base role function (default ``bes.roles.role_of``).

    Returns ``(params, report)``: the selected candidate's parameters or, when none passes, the
    starting parameters themselves. ``report``: ``accepted``, ``lr``, ``reason``, ``start``,
    ``candidates`` (per learning rate: ``lr``, ``accepted``, ``reasons`` (codes), ``messages``,
    ``checked``, ``pref`` metrics, ``heldout_loss``, ``stats``, ``train``), data sizes and time.
    """
    pref = PrefCfg() if pref is None else pref
    t0 = time.time()
    say = log if log is not None else (lambda msg: None)
    consts = M.make_consts(cfg.model) if consts is None else consts
    train_pairs, held_pairs = _as_split(pairs)
    report: dict[str, Any] = {"accepted": False, "lr": None, "beta": pref.beta,
                              "lrs": [float(x) for x in lrs], "n_train_pairs": len(train_pairs),
                              "n_heldout_pairs": len(held_pairs), "candidates": []}
    if not train_pairs or not held_pairs or not lrs:
        missing = ("training pairs" if not train_pairs else
                   "held-out pairs" if not held_pairs else "learning rates")
        report.update(reason=f"no {missing}: the starting parameters are kept",
                      time_s=time.time() - t0)
        return params, report
    cache = int(pref.cache_gb * 2 ** 30)
    common = dict(tok=tok, features=features, feature_fn=feature_fn,
                  mask_forced=pref.mask_forced, cache_bytes=cache)
    train = PairBatches(train_pairs, pref.seq_len, pref.pairs_per_micro, pref.n_micro,
                        order=shuffled_order(len(train_pairs), pref.seed), **common)
    held = PairBatches(held_pairs, pref.seq_len, pref.pairs_per_micro, pref.n_micro, **common)
    report["steps_per_epoch"] = len(train)
    logp_fn = make_logp_fn(cfg.model)
    say(f"dpo: reference log-probabilities of {train.n_pairs} + {held.n_pairs} pairs")
    compute_ref_logps(params, consts, train, cfg.model, logp_fn)
    compute_ref_logps(params, consts, held, cfg.model, logp_fn)
    counts = held.target_counts()
    start = {"pref": preference_metrics(held.ref, held.ref, counts, pref.beta),
             "heldout_loss": float(heldout_loss_fn(params)),
             "stats": _stats(sample_stats_fn(params))}
    report["start"] = start
    optimizer = build_optimizer(cfg, params, pref, role_fn)
    step_fn = make_dpo_step(cfg.model, optimizer)
    best_params = None
    for lr in lrs:
        cand_params, train_rep = train_epoch(params, consts, train, cfg, lr, pref,
                                             optimizer=optimizer, step_fn=step_fn, log=log)
        cand: dict[str, Any] = {"lr": float(lr), "train": train_rep, "finite": train_rep["finite"],
                                "checked": []}
        problems: list[tuple[str, str]] = []
        if not cand["finite"]:
            problems = [("nonfinite", "training produced non-finite parameters or losses")]
            cand["checked"].append("nonfinite")
        else:
            cand["checked"].append("nonfinite")
            logp = pair_logps(cand_params, consts, held, cfg.model, logp_fn)
            cand["pref"] = preference_metrics(logp, held.ref, counts, pref.beta)
            problems += check_accuracy(start, cand, pref)
            cand["checked"].append("accuracy")
            if evaluate_all or not problems:
                cand["heldout_loss"] = float(heldout_loss_fn(cand_params))
                problems += check_loss(start, cand, pref)
                cand["checked"].append("loss")
            if evaluate_all or not problems:
                cand["stats"] = _stats(sample_stats_fn(cand_params))
                problems += check_stats(start, cand, pref)
                cand["checked"] += ["length", "repetition"]
        cand["accepted"] = not problems
        cand["reasons"] = [code for code, _ in problems]
        cand["messages"] = [msg for _, msg in problems]
        report["candidates"].append(cand)
        say(f"dpo: lr={lr:g} accepted={cand['accepted']} reasons={cand['reasons']}")
        best = select_candidate(report["candidates"])
        if best == len(report["candidates"]) - 1:
            best_params = cand_params
        del cand_params
    best = select_candidate(report["candidates"])
    if best is None:
        lines = [f"lr={c['lr']:g}: " + "; ".join(c["messages"]) for c in report["candidates"]]
        report.update(accepted=False, lr=None,
                      reason="no candidate passed the health checks; the starting parameters "
                             "are kept (" + " | ".join(lines) + ")",
                      time_s=time.time() - t0)
        return params, report
    chosen = report["candidates"][best]
    report.update(accepted=True, lr=chosen["lr"],
                  reason=f"lr={chosen['lr']:g} accepted with held-out preference accuracy "
                         f"{chosen['pref']['acc']:.4f} (start {start['pref']['acc']:.4f})",
                  time_s=time.time() - t0)
    return best_params, report
