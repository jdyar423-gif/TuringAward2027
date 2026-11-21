"""Calibrated gate: one logistic regression per entry kind on cheap retrieval features.

``p(kind) = sigmoid(w_kind . x_kind + c_kind)`` with the feature vectors below (fixed
transforms of the raw features computed by ``bes.library.retrieve.Library``):

wiki (``WIKI_FEATURES``)
    log1p(s1), margin (s1 - s2) / s1 against the best passage of another article, title
    match (share of title terms in the query), coverage s1 / s_max, exact title (an identity
    request whose key has a primary topic, ``bes.library.titles``), that topic is a variant
    title (not the plain title of the key), it is a person article, lead passage, lookup
    query, log1p(distinct query terms), qa answer found in the passage
qa (``QA_FEATURES``)
    sym = min(cov_q, cov_e), cov_q, cov_e, question type match, query without wh-word,
    sym1 - sym2, top-2 entries agree on the answer (sym2 >= 0.5), disagree, log1p(Wq),
    log1p(distinct query terms), answer found in the wiki candidate

The gate returns the candidate with the larger probability when it reaches ``threshold``
(at most one entry, always). Coefficients and the threshold are stored numbers (float32,
counted by ``numbers``). ``fit_logistic`` is a deterministic Newton (IRLS) fit with an L2
penalty on the weights (not the intercept); ``choose_threshold`` takes the smallest
threshold whose pooled dev precision reaches the target (largest coverage).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

WIKI_FEATURES = ("log_s1", "margin", "title_match", "coverage", "exact_title", "variant", "person",
                 "lead", "lookup", "log_terms", "cross")
QA_FEATURES = ("sym", "cov_q", "cov_e", "type_match", "no_wh", "sym_margin", "agree", "disagree",
               "log_wq", "log_terms", "cross")
KINDS = ("wiki", "qa")


def wiki_vector(f: dict) -> np.ndarray:
    """float64 [len(WIKI_FEATURES)] from the raw wiki features of one candidate."""
    return np.array([np.log1p(max(f["s1"], 0.0)), f["margin"], f["title_match"], f["coverage"],
                     float(f["exact_title"]), float(f["variant"]), float(f["person"]), float(f["lead"]),
                     float(f["qtype"] == "lookup"), np.log1p(f["n_terms"]), float(f["cross"])])


def qa_vector(f: dict) -> np.ndarray:
    """float64 [len(QA_FEATURES)] from the raw qa features of one candidate."""
    return np.array([f["sym"], f["cov_q"], f["cov_e"], float(f["type_match"]), float(f["no_wh"]),
                     f["sym_margin"], float(f["agree"]), float(f["disagree"]), np.log1p(f["wq"]),
                     np.log1p(f["n_terms"]), float(f["cross"])])


VECTOR = {"wiki": wiki_vector, "qa": qa_vector}
FEATURES = {"wiki": WIKI_FEATURES, "qa": QA_FEATURES}


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, l2: float = 1e-2,
                 iters: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Weighted L2-penalised logistic regression by Newton steps.

    ``x`` float64 [n, d], ``y`` in {0, 1} [n], ``w`` sample weights [n] (default 1).
    Minimises ``sum_i w_i * logloss_i / sum_i w_i + l2/2 * |beta[:d]|^2``; returns
    ``beta`` [d + 1] (weights then intercept). Deterministic; converges for separable data
    because of the penalty.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    sw = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    sw = sw / sw.sum()
    xa = np.hstack([x, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = sigmoid(xa @ beta)
        g = xa.T @ (sw * (p - y)) + reg * beta
        h = (xa * (sw * p * (1 - p))[:, None]).T @ xa + np.diag(reg) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        beta -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def pr_curve(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> list[dict]:
    """Precision / coverage of ``p >= t`` for every distinct ``t`` (descending); ``p`` = -inf
    means no candidate. Coverage and precision use the weights ``w`` (default 1)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    tot = w.sum()
    o = np.argsort(-p, kind="stable")
    ps, ys, ws = p[o], y[o], w[o]
    cw = np.cumsum(ws)
    cy = np.cumsum(ws * ys)
    out = []
    i = 0
    n = ps.shape[0]
    while i < n:
        if not np.isfinite(ps[i]):
            break
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        out.append({"t": float(ps[i]), "coverage": float(cw[j] / tot), "precision": float(cy[j] / cw[j])})
        i = j + 1
    return out


def choose_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                     target: float = 0.5, min_t: float = 0.0) -> tuple[float, dict]:
    """Smallest threshold ``t >= min_t`` with pooled precision of ``p >= t`` at least
    ``target`` (= the largest coverage at that precision). Returns ``(t, point)``; when no
    threshold reaches the target, ``t = inf`` (the gate never fires)."""
    best = (float("inf"), {"t": float("inf"), "coverage": 0.0, "precision": 0.0})
    for pt in pr_curve(p, y, w):
        if pt["t"] >= min_t and pt["precision"] >= target:
            best = (pt["t"], pt)
    return best


def choose_threshold_sets(p: np.ndarray, y: np.ndarray, groups, target: float = 0.5,
                          min_t: float = 0.0, negative=(), max_negative_rate: float = 0.0
                          ) -> tuple[float, dict]:
    """Smallest threshold ``t >= min_t`` at which *every* group (dev set) whose firing set is
    non-empty has precision of ``p >= t`` at least ``target``: the largest coverage subject
    to the precision floor on each dev set separately (a pooled floor would let an easy set
    hide a hard one). A group in ``negative`` has no right answers (its precision is 0 as
    soon as it fires): it qualifies when at most ``max_negative_rate`` of its rows fire.
    Returns ``(t, {"coverage": pooled with equal set weights over the other groups,
    "per_set": {group: (coverage, precision)}})``; ``t = inf`` when no threshold qualifies."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(list(groups))
    names = sorted(set(g.tolist()))
    neg = set(negative)
    size = {s: int((g == s).sum()) for s in names}
    cand = sorted({float(v) for v in p[np.isfinite(p)] if v >= min_t})
    best_t = float("inf")
    for t in reversed(cand):           # descending; keep the smallest qualifying threshold
        sel = p >= t
        ok = True
        for s in names:
            m = sel & (g == s)
            if s in neg:
                if m.sum() > max_negative_rate * size[s] + 1e-12:
                    ok = False
                    break
            elif m.any() and y[m].mean() < target:
                ok = False
                break
        if ok:
            best_t = t
    per = {}
    covs = []
    for s in names:
        m = g == s
        f = m & (p >= best_t)
        per[s] = (float(f.sum() / max(1, m.sum())), float(y[f].mean()) if f.any() else 0.0)
        if s not in neg:
            covs.append(per[s][0])
    return best_t, {"t": best_t, "coverage": float(np.mean(covs)) if covs else 0.0, "per_set": per}


@dataclass
class GateModel:
    """Coefficients per kind (weights + intercept) and the decision threshold."""

    coef: dict[str, np.ndarray]
    threshold: float

    def prob(self, kind: str, feats: dict) -> float:
        b = self.coef[kind]
        x = VECTOR[kind](feats)
        return float(sigmoid(float(x @ b[:-1] + b[-1])))

    def array(self) -> np.ndarray:
        """The stored form: float32 [wiki coefs, qa coefs, threshold]."""
        return np.concatenate([self.coef["wiki"], self.coef["qa"], [self.threshold]]).astype(np.float32)

    def numbers(self) -> int:
        return (self.array().nbytes + 1) // 2

    @classmethod
    def from_array(cls, a: np.ndarray) -> "GateModel":
        a = np.asarray(a, dtype=np.float64)
        nw, nq = len(WIKI_FEATURES) + 1, len(QA_FEATURES) + 1
        if a.shape[0] != nw + nq + 1:
            raise ValueError("gate array has the wrong length")
        return cls({"wiki": a[:nw], "qa": a[nw:nw + nq]}, float(a[-1]))

    def save(self, out_dir: str, extra: dict | None = None) -> None:
        from bes.library.pack import _atomic_save_npy, _atomic_write

        _atomic_save_npy(os.path.join(out_dir, "gate_coef.npy"), self.array())
        meta = {"wiki_features": list(WIKI_FEATURES), "qa_features": list(QA_FEATURES),
                "threshold": float(np.float32(self.threshold)), **(extra or {})}
        _atomic_write(os.path.join(out_dir, "gate.json"), json.dumps(meta, indent=1, sort_keys=True).encode())

    @classmethod
    def load(cls, out_dir: str) -> "GateModel":
        """The saved model; a model fitted on other features (an older library) is refused."""
        meta_path = os.path.join(out_dir, "gate.json")
        if os.path.exists(meta_path):
            meta = json.loads(open(meta_path).read())
            for kd, names in FEATURES.items():
                got = tuple(meta.get(f"{kd}_features", names))
                if got != tuple(names):
                    raise ValueError(f"the library's gate was fitted on the {kd} features {got}, this code "
                                     f"computes {tuple(names)}: rebuild the library")
        return cls.from_array(np.load(os.path.join(out_dir, "gate_coef.npy")))

    @classmethod
    def stored(cls, coef: dict[str, np.ndarray], threshold: float) -> "GateModel":
        """The model exactly as it is stored (float32 round trip), so that build-time and
        load-time decisions agree bit for bit."""
        return cls.from_array(cls(coef, threshold).array())
