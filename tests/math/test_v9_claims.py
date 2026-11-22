"""Numerical checks of the propositions in docs/V9.md, section 3.

Independent, deterministic numpy computations that do not import the package: the dimension
counts of the token-page families (rank of the Jacobian on small instances), the water-filling
allocation, and the three-piece assembly of the "sorted" table gradient.
"""
import numpy as np


def _jac_rank(f, theta, eps=1e-6, tol=1e-7):
    y0 = f(theta)
    jac = np.empty((y0.size, theta.size))
    for i in range(theta.size):
        hi, lo = theta.copy(), theta.copy()
        hi[i] += eps
        lo[i] -= eps
        jac[:, i] = (f(hi) - f(lo)).ravel() / (2 * eps)
    s = np.linalg.svd(jac, compute_uv=False)
    return int((s > tol * s[0]).sum())


def test_token_neuron_family_dimension():
    """f(x, h) = P_out^T sum_j relu(<k_xj, P_in^T h>)^2 v_xj has generic dimension
    V rho (2 dk - 1) + 2 d dk - 2 dk^2 (one scaling per neuron, one GL(dk) on each side)."""
    rng = np.random.default_rng(1)
    for vocab, rho, d, dk in [(4, 2, 6, 3), (5, 3, 7, 4), (3, 4, 6, 4)]:
        probes = rng.normal(size=(60, d))
        sizes = [d * dk, dk * d, vocab * rho * dk, vocab * rho * dk]
        offs = np.cumsum([0] + sizes)

        def f(th):
            pin = th[offs[0]:offs[1]].reshape(d, dk)
            pout = th[offs[1]:offs[2]].reshape(dk, d)
            key = th[offs[2]:offs[3]].reshape(vocab, rho, dk)
            val = th[offs[3]:offs[4]].reshape(vocab, rho, dk)
            act = np.maximum(np.einsum("pk,vjk->vpj", probes @ pin, key), 0) ** 2
            return np.einsum("vpj,vjk->vpk", act, val) @ pout

        rank = _jac_rank(f, rng.normal(size=offs[-1]))
        assert rank == vocab * rho * (2 * dk - 1) + 2 * d * dk - 2 * dk * dk


def test_gain_family_dimensions():
    """Token gains on FFN neurons: N (V + 2d - 2); destination and source gains on the m channels
    between W_v and W_o: m (2V + 2d - 3)."""
    rng = np.random.default_rng(0)
    for vocab, n, d in [(3, 4, 5), (5, 6, 4)]:
        probes = rng.normal(size=(40, d))

        def f(th):
            wu = th[:n * d].reshape(n, d)
            wd = th[n * d:2 * n * d].reshape(d, n)
            g = th[2 * n * d:].reshape(vocab, n)
            return np.einsum("pn,vn,dn->vpd", np.maximum(probes @ wu.T, 0) ** 2, g, wd)

        assert _jac_rank(f, rng.normal(size=2 * n * d + vocab * n)) == n * (vocab + 2 * d - 2)
    for vocab, m, d in [(3, 4, 6), (4, 3, 5)]:
        def f(th):
            wo = th[:d * m].reshape(d, m)
            wv = th[d * m:2 * d * m].reshape(m, d)
            go = th[2 * d * m:2 * d * m + vocab * m].reshape(vocab, m)
            gv = th[2 * d * m + vocab * m:].reshape(vocab, m)
            return np.einsum("dm,am,bm,me->abde", wo, go, gv, wv)

        assert _jac_rank(f, rng.normal(size=2 * d * m + 2 * vocab * m)) == m * (2 * vocab + 2 * d - 3)


def test_water_filling_is_the_optimum_of_the_log_law():
    """max sum_x f_x ln(1 + rho_x / n0) s.t. sum rho_x = R, rho >= 0 is attained by
    rho_x = max(0, f_x (R + |A| n0) / F_A - n0): no feasible perturbation improves it."""
    rng = np.random.default_rng(3)
    f = rng.zipf(1.3, size=200).astype(float)
    f = np.sort(f / f.sum())[::-1]
    n0, total = 0.2, 150.0
    active = np.ones(f.size, bool)
    for _ in range(f.size):
        level = (total + active.sum() * n0) / f[active].sum()
        rho = np.where(active, f * level - n0, 0.0)
        if (rho[active] > 0).all():
            break
        active &= rho > 0
    assert abs(rho.sum() - total) < 1e-9 and (rho >= 0).all() and 0 < active.sum() < f.size

    def gain(r):
        return float((f * np.log1p(r / n0)).sum())

    best = gain(rho)
    assert best > gain(np.full(f.size, total / f.size))           # better than uniform
    for _ in range(200):                                          # random feasible moves never help
        i, j = rng.choice(f.size, size=2, replace=False)
        step = min(rho[i], 0.05 * total) * rng.random()
        moved = rho.copy()
        moved[i] -= step
        moved[j] += step
        assert gain(moved) <= best + 1e-12


def test_sorted_table_gradient_assembly():
    """The three-piece assembly of docs/V9.md (run ranks inside chunks, chunk totals through a
    prefix sum) equals the scatter-add, for runs that span several chunks and absent ids."""
    rng = np.random.default_rng(5)
    for rows, n, c, width in [(7, 50, 4, 3), (5, 64, 8, 2), (11, 23, 4, 4), (3, 40, 4, 2)]:
        probs = rng.dirichlet(np.full(rows, 0.3))
        ids = rng.choice(rows, size=n, p=probs)
        ids[ids == rows - 1] = 0                                  # the last id never occurs
        g = rng.normal(size=(n, width))
        want = np.zeros((rows, width))
        np.add.at(want, ids, g)

        order = np.argsort(ids, kind="stable")
        ids_s, gs = ids[order], g[order]
        n_chunks = -(-n // c)
        pad = n_chunks * c - n
        ids_s = np.concatenate([ids_s, np.full(pad, rows)])
        gs = np.concatenate([gs, np.zeros((pad, width))])
        rank = np.concatenate([[0], np.cumsum(ids_s[1:] != ids_s[:-1])])
        local = rank.reshape(n_chunks, c) - rank.reshape(n_chunks, c)[:, :1]
        onehot = (local[:, :, None] == np.arange(c)).astype(float)
        part = np.einsum("ncr,ncw->nrw", onehot, gs.reshape(n_chunks, c, width))
        pref = np.cumsum(part.sum(1), axis=0)
        got = np.zeros((rows, width))
        for v in range(rows):
            first = np.searchsorted(ids_s, v, "left")
            last = np.searchsorted(ids_s, v, "right") - 1
            if last < first:
                continue
            c1, c2 = first // c, last // c
            got[v] = part[c1, rank[first] - rank[c1 * c]]
            if c2 > c1:
                got[v] += part[c2, 0] + pref[c2 - 1] - pref[c1]
        np.testing.assert_allclose(got, want, atol=1e-12)
