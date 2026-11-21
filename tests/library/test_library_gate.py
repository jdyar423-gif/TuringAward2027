"""Calibrated gate (bes.library.gate): logistic fit, probabilities, precision/coverage curve
and threshold against direct computations written in the test."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bes.library import gate as G


def direct_loss(beta, x, y, w, l2):
    """Weighted mean log-loss + l2/2 |weights|^2 (intercept not penalised)."""
    w = w / w.sum()
    tot = 0.0
    for xi, yi, wi in zip(x, y, w):
        z = float(np.dot(beta[:-1], xi) + beta[-1])
        p = 1.0 / (1.0 + math.exp(-z))
        p = min(max(p, 1e-300), 1 - 1e-16)
        tot += wi * -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
    return tot + 0.5 * l2 * float(np.dot(beta[:-1], beta[:-1]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fit_logistic_is_the_penalised_optimum(seed):
    rng = np.random.default_rng(seed)
    n, d = 300, 4
    x = rng.normal(size=(n, d))
    true = rng.normal(size=d)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x @ true - 0.3)))).astype(float)
    w = rng.uniform(0.2, 2.0, size=n)
    l2 = 1e-2
    beta = G.fit_logistic(x, y, w, l2=l2)
    f0 = direct_loss(beta, x, y, w, l2)
    # numerical gradient of the directly computed objective vanishes at the solution
    eps = 1e-6
    for j in range(d + 1):
        e = np.zeros(d + 1)
        e[j] = eps
        g = (direct_loss(beta + e, x, y, w, l2) - direct_loss(beta - e, x, y, w, l2)) / (2 * eps)
        assert abs(g) < 1e-6
    # and random perturbations never improve it
    for _ in range(20):
        assert direct_loss(beta + rng.normal(scale=0.05, size=d + 1), x, y, w, l2) >= f0 - 1e-12


def test_fit_logistic_separable_data_stays_finite_and_deterministic():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    a = G.fit_logistic(x, y, l2=1e-3)
    b = G.fit_logistic(x, y, l2=1e-3)
    assert np.all(np.isfinite(a)) and np.array_equal(a, b)
    assert a[0] > 0


def feats_wiki(**kw):
    base = {"s1": 12.0, "margin": 0.3, "title_match": 1.0, "coverage": 0.8, "exact_title": True,
            "ambiguous": False, "lead": True, "qtype": "question", "n_terms": 2, "cross": False}
    base.update(kw)
    return base


def feats_qa(**kw):
    base = {"sym": 0.9, "cov_q": 1.0, "cov_e": 0.9, "type_match": True, "no_wh": False,
            "sym_margin": 0.2, "agree": True, "disagree": False, "wq": 14.0, "n_terms": 3, "cross": True}
    base.update(kw)
    return base


def test_gate_probability_equals_direct_computation(tmp_path):
    rng = np.random.default_rng(5)
    coef = {"wiki": rng.normal(size=len(G.WIKI_FEATURES) + 1), "qa": rng.normal(size=len(G.QA_FEATURES) + 1)}
    m = G.GateModel.stored(coef, 0.37)
    fw = feats_wiki(qtype="lookup", s1=7.5)
    x = [math.log1p(7.5), 0.3, 1.0, 0.8, 1.0, 0.0, 1.0, 1.0, math.log1p(2), 0.0]
    cw = np.asarray(coef["wiki"], dtype=np.float32).astype(np.float64)
    z = sum(a * b for a, b in zip(cw[:-1], x)) + cw[-1]
    assert m.prob("wiki", fw) == pytest.approx(1 / (1 + math.exp(-z)), rel=1e-12)
    fq = feats_qa(agree=False, disagree=True)
    xq = [0.9, 1.0, 0.9, 1.0, 0.0, 0.2, 0.0, 1.0, math.log1p(14.0), math.log1p(3), 1.0]
    cq = np.asarray(coef["qa"], dtype=np.float32).astype(np.float64)
    zq = sum(a * b for a, b in zip(cq[:-1], xq)) + cq[-1]
    assert m.prob("qa", fq) == pytest.approx(1 / (1 + math.exp(-zq)), rel=1e-12)
    # stored form: float32 coefficients + threshold, counted as stored numbers
    arr = m.array()
    assert arr.dtype == np.float32 and arr.shape[0] == len(G.WIKI_FEATURES) + len(G.QA_FEATURES) + 3
    assert m.numbers() == arr.nbytes // 2
    m.save(str(tmp_path))
    r = G.GateModel.load(str(tmp_path))
    assert r.threshold == m.threshold and r.prob("wiki", fw) == m.prob("wiki", fw)


def brute_curve(p, y, w):
    out = []
    tot = w.sum()
    for t in sorted(set(p[np.isfinite(p)].tolist()), reverse=True):
        sel = p >= t
        out.append((t, w[sel].sum() / tot, (w[sel] * y[sel]).sum() / w[sel].sum()))
    return out


@pytest.mark.parametrize("seed", [0, 3])
def test_curve_and_threshold_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 200
    p = np.round(rng.uniform(size=n), 2)          # ties on purpose
    p[rng.uniform(size=n) < 0.1] = -np.inf         # no candidate
    y = (rng.uniform(size=n) < np.clip(p, 0, 1)).astype(float)
    w = rng.uniform(0.5, 1.5, size=n)
    got = G.pr_curve(p, y, w)
    ref = brute_curve(p, y, w)
    assert len(got) == len(ref)
    for a, (t, c, pr) in zip(got, ref):
        assert a["t"] == t and a["coverage"] == pytest.approx(c) and a["precision"] == pytest.approx(pr)
    for target in (0.5, 0.7, 0.9, 1.01):
        t, pt = G.choose_threshold(p, y, w, target)
        ok = [r for r in ref if r[2] >= target]
        if not ok:
            assert t == float("inf") and pt["coverage"] == 0.0
        else:
            best = min(ok, key=lambda r: r[0])     # smallest threshold = largest coverage
            assert t == best[0] and pt["coverage"] == pytest.approx(best[1])


def test_sigmoid_is_stable():
    z = np.array([-1000.0, -30.0, 0.0, 30.0, 1000.0])
    s = G.sigmoid(z)
    assert np.all(np.isfinite(s)) and s[2] == 0.5 and s[0] == 0.0 and s[-1] == 1.0
    assert s[1] == pytest.approx(1 / (1 + math.exp(30.0)))
