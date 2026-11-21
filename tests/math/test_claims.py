"""Numerical checks of the propositions in docs/MATH.md (plan section 3).

Each test is an independent, deterministic numpy computation. They do not import the package:
they pin down the mathematics the design relies on, so that a change of design that breaks an
assumption is noticed here first.
"""
import itertools
import math

import numpy as np


def lse(v):
    m = v.max()
    return m + math.log(np.exp(v - m).sum())


def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    keep = p > 0
    return float((p[keep] * (np.log(p[keep]) - np.log(q[keep]))).sum())


# ----------------------------------------------------------------------------- scale one
def test_b1_softmax_with_sink_is_dirichlet_posterior():
    """B1: one-hot keys, sharp kernel, sink logit beta + log(lambda) -> add-lambda/V estimator."""
    rng = np.random.default_rng(0)
    vocab, alpha, beta = 12, 0.4, 40.0
    lam = vocab * alpha
    for _ in range(20):
        x = rng.integers(vocab, size=60)
        a = x[-1]
        prev, nxt = x[:-1], x[1:]
        counts = np.bincount(nxt[prev == a], minlength=vocab).astype(float)
        n_a = counts.sum()
        bayes = (counts + alpha) / (n_a + lam)
        # attention: query = one-hot(a); keys = one-hot(prev); values = one-hot(next); sink value = uniform
        logits = np.concatenate([[beta + math.log(lam)], beta * (prev == a).astype(float)])
        w = np.exp(logits - lse(logits))
        values = np.concatenate([np.full((1, vocab), 1.0 / vocab), np.eye(vocab)[nxt]])
        out = w @ values
        assert np.abs(out - bayes).max() < 1e-9
        conf = 1.0 - w[0]                      # mass outside the sink
        assert abs(conf - n_a / (n_a + lam)) < 1e-9


def _kern(a, b, beta):
    return np.exp(beta * (a @ b.T - 1.0))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_b2_one_cholesky_gives_every_prefix_posterior():
    rng = np.random.default_rng(1)
    t, dk, dv, beta, lam = 48, 12, 5, 3.0, 0.2
    keys, queries, vals = _unit(rng.normal(size=(t, dk))), _unit(rng.normal(size=(t, dk))), rng.normal(size=(t, dv))
    gram = _kern(keys, keys, beta) + lam * np.eye(t)
    cross = _kern(keys, queries, beta)                     # cross[j, i] = k(key_j, query_i)
    chol = np.linalg.cholesky(gram)
    u = np.linalg.solve(chol, cross)
    vt = np.linalg.solve(chol, vals)
    mask = np.triu(np.ones((t, t)))                        # j <= i
    mean = (mask * u).T @ vt
    var = 1.0 - ((mask * u) ** 2).sum(0)
    for i in range(t):
        g = gram[: i + 1, : i + 1]
        c = cross[: i + 1, i]
        assert np.abs(mean[i] - c @ np.linalg.solve(g, vals[: i + 1])).max() < 1e-10
        assert abs(var[i] - (1.0 - c @ np.linalg.solve(g, c))) < 1e-10


def test_b2_gp_weights_differ_from_sink_weights_by_key_correlations():
    rng = np.random.default_rng(2)
    n, dk, beta, lam = 20, 6, 2.0, 0.7
    keys, q = _unit(rng.normal(size=(n, dk))), _unit(rng.normal(size=(1, dk)))
    g = _kern(keys, keys, beta)
    c = _kern(keys, q, beta)[:, 0]
    a = g + lam * np.eye(n)
    w_gp = np.linalg.solve(a, c)
    w_sink = c / (lam + c.sum())                           # softmax with a sink of weight lam
    rhs = np.linalg.solve(a, (c.sum() * c - g @ c)) / (lam + c.sum())
    assert np.abs((w_gp - w_sink) - rhs).max() < 1e-12


# ----------------------------------------------------------------------------- scale two
def test_t1_value_of_candidates_is_a_coarsened_kl():
    rng = np.random.default_rng(3)
    v, m = 30, 4
    for _ in range(50):
        q = rng.dirichlet(np.ones(v))
        p = rng.dirichlet(0.3 * np.ones(v))
        cand = rng.choice(v, size=m, replace=False)
        rest = np.setdiff1d(np.arange(v), cand)
        p_u = np.append(p[cand], p[rest].sum())
        q_u = np.append(q[cand], q[rest].sum())

        def gain(g):
            z = np.log(q).copy()
            z[cand] += g
            q2 = np.exp(z - lse(z))
            q2_u = np.append(q2[cand], q2[rest].sum())
            direct = float((p * (np.log(q2) - np.log(q))).sum())
            assert abs(direct - (kl(p_u, q_u) - kl(p_u, q2_u))) < 1e-10
            return direct

        g_star = np.log(p_u[:m] / q_u[:m]) - math.log(p_u[m] / q_u[m])
        best = gain(g_star)
        assert abs(best - kl(p_u, q_u)) < 1e-10
        for _ in range(20):                                # concavity: no perturbation beats it
            assert gain(g_star + rng.normal(0, 0.5, m)) <= best + 1e-12
        assert abs(gain(np.zeros(m))) < 1e-12


def test_t2_copy_pointer_cost_telescopes_to_log_m_over_occ():
    rng = np.random.default_rng(4)
    lib = rng.integers(6, size=4000)
    start, length = 1234, 25
    span = lib[start:start + length]
    cost, occ_prev = 0.0, len(lib) - length                # positions that can still be extended
    positions = np.arange(len(lib) - length)
    alive = np.ones(len(positions), bool)
    for k, tok in enumerate(span):
        cont = alive & (lib[positions + k] == tok)
        cost += -math.log2(cont.sum() / alive.sum())
        alive = cont
    assert abs(cost - math.log2(occ_prev / alive.sum())) < 1e-9


def test_t3_tilt_identity_holds_but_complement_form_is_the_stable_one():
    rng = np.random.default_rng(5)
    v = 8192
    z = rng.normal(0, 3, v)
    idx = rng.choice(v, size=16, replace=False)
    s = rng.normal(0, 3, 16)
    dense = z.copy()
    dense[idx] += s
    q = np.exp(z - lse(z))
    assert abs(lse(z) + math.log1p((q[idx] * np.expm1(s)).sum()) - lse(dense)) < 1e-12
    # float32 adversarial case: almost all mass on tokens that receive a large negative correction
    z32 = np.full(v, -20.0, np.float32)
    z32[0] = 0.0
    corr = np.zeros(v, np.float32)
    corr[0] = -25.0
    exact = lse(z32.astype(np.float64) + corr.astype(np.float64))
    q32 = np.exp(z32 - np.float32(lse(z32.astype(np.float64)))).astype(np.float32)
    identity = np.float32(lse(z32.astype(np.float64))) + np.log1p(np.float32((q32[:1] * np.expm1(corr[:1])).sum()))
    y = (z32 + corr).astype(np.float32)
    m = y.max()
    complement = m + np.log(np.exp(y - m).sum(dtype=np.float32))
    assert abs(float(complement) - exact) < 1e-4
    assert abs(float(identity) - exact) > 10 * abs(float(complement) - exact)


# --------------------------------------------------------------------------- scale three
def _toy_lm(rng, vocab, order_states):
    return rng.dirichlet(np.ones(vocab), size=order_states)


def test_f1_f2_smc_normaliser_is_unbiased_and_mask_weight_is_allowed_mass():
    """Bigram model over 4 symbols, length 5, constraint: symbol 3 appears exactly once."""
    rng = np.random.default_rng(6)
    vocab, length, k = 4, 5, 8
    trans = _toy_lm(rng, vocab, vocab)
    start = rng.dirichlet(np.ones(vocab))
    z_exact = 0.0
    for seq in itertools.product(range(vocab), repeat=length):
        if seq.count(3) != 1:
            continue
        p = start[seq[0]]
        for a, b in zip(seq, seq[1:]):
            p *= trans[a, b]
        z_exact += p

    def allowed(count3, steps_left_after):
        # token-local reachability: never a second '3'; force the '3' when it is the last chance
        ok = np.ones(vocab, bool)
        if count3 == 1:
            ok[3] = False
        elif steps_left_after == 0:
            ok[:3] = False
        return ok

    estimates = []
    for _ in range(4000):
        logw = np.zeros(k)
        prev = np.full(k, -1)
        c3 = np.zeros(k, int)
        for t in range(length):
            for i in range(k):
                p = start if prev[i] < 0 else trans[prev[i]]
                ok = allowed(c3[i], length - t - 1)
                mass = p[ok].sum()                         # incremental weight = allowed mass
                logw[i] += math.log(mass)
                tok = rng.choice(vocab, p=np.where(ok, p, 0) / mass)
                prev[i], c3[i] = tok, c3[i] + (tok == 3)
        estimates.append(np.exp(logw).mean())
    est = np.mean(estimates)
    se = np.std(estimates) / math.sqrt(len(estimates))
    assert abs(est - z_exact) < 4 * se + 1e-12


def test_f4_best_of_k_tilt_is_bounded_by_log_k():
    rng = np.random.default_rng(7)
    n, k = 200, 8
    base = rng.dirichlet(np.ones(n))
    order = np.argsort(rng.normal(size=n))                 # a strict ranking by some score
    cdf_below = np.zeros(n)
    acc = 0.0
    for idx in order:                                      # ascending score
        cdf_below[idx] = acc
        acc += base[idx]
    best = (cdf_below + base) ** k - cdf_below ** k        # law of the best of k draws
    assert abs(best.sum() - 1.0) < 1e-12
    assert kl(best, base) <= math.log(k) - (k - 1) / k + 1e-9


# ------------------------------------------------------------------------ time and memory
def test_all_recurrent_allocation_is_optimal_under_the_loop_law():
    t, c = 2.0e7, 1.3e8
    for phi in (0.46, 0.55, 0.65):
        def n_eff(n_once):
            n_rec = t - n_once
            r = (c - n_once) / n_rec
            return n_once + r ** phi * n_rec
        grid = np.linspace(0, 0.95 * t, 400)
        values = np.array([n_eff(x) for x in grid])
        assert values.argmax() == 0
        assert abs(values[0] - t ** (1 - phi) * c ** phi) / values[0] < 1e-12


def test_two_price_water_filling_closed_form_is_the_optimum():
    rng = np.random.default_rng(8)
    energy = np.array([3.0, 1.2, 0.8, 0.9, 0.6, 0.4])
    cells = np.array([1.0, 8, 64, 512, 4096, 32768])
    alpha, n0, active, total = 0.3, 1e-3, 20.0, 100.0

    def alloc(la, ln):
        return n0 * np.maximum((alpha * energy / (n0 * (la + ln * cells))) ** (1 / (1 + alpha)) - 1, 0)

    def solve_la(ln):
        lo, hi = 1e-15, 1e6
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if alloc(mid, ln).sum() > active else (lo, mid)
        return hi

    lo, hi = 1e-15, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if (cells * alloc(solve_la(mid), mid)).sum() > total else (lo, mid)
    n = alloc(solve_la(hi), hi)
    risk = lambda x: (energy * (1 + x / n0) ** (-alpha)).sum()
    assert n.sum() <= active + 1e-6 and (cells * n).sum() <= total + 1e-6
    for _ in range(20000):
        m = np.maximum(n * (1 + rng.normal(0, 0.03, 6)) + rng.normal(0, 1e-4, 6), 0)
        m = m / max(m.sum() / active, (cells * m).sum() / total, 1.0)
        assert risk(m) >= risk(n) - 1e-9


def test_loops_per_scale_follow_the_allocation_law():
    k, levels, gamma = 4, 4, 0.1
    scale = k ** np.arange(levels).astype(float)
    weight = 1 / scale - np.append(1 / scale[1:], 0.0)
    for beta in (0.3, 0.5):
        energy = scale ** (-beta)
        lo, hi = 1e-12, 1e12
        for _ in range(200):
            mu = math.sqrt(lo * hi)
            r = (gamma * energy / (mu * weight)) ** (1 / (1 + gamma))
            lo, hi = (mu, hi) if (weight * r).sum() > 8.0 else (lo, mu)
        slope = np.polyfit(np.log(scale[:-1]), np.log(r[:-1]), 1)[0]
        assert abs(slope - (1 - beta) / (1 + gamma)) < 1e-6
        plain = (energy * 8.0 ** (-gamma)).sum()
        multi = (energy * r ** (-gamma)).sum()
        assert 0 < 1 - multi / plain < 0.05               # the gain exists and is small


def test_clamp_shift_registers_form_a_monoid():
    rng = np.random.default_rng(9)

    def apply(f, x):
        a, lo, hi = f
        return min(hi, max(lo, x + a))

    def compose(f, g):                                     # f first, then g
        a1, lo1, hi1 = f
        return (a1 + g[0], apply(g, lo1), apply(g, hi1))

    def rand():
        lo, hi = sorted((int(rng.integers(-6, 6)), int(rng.integers(-6, 70))))
        return (int(rng.integers(-3, 4)), lo, hi)

    for _ in range(3000):
        fs = [rand() for _ in range(int(rng.integers(2, 30)))]
        x = int(rng.integers(-40, 100))
        y = x
        for f in fs:
            y = apply(f, y)
        total = fs[0]
        for f in fs[1:]:
            total = compose(total, f)
        assert apply(total, x) == y
