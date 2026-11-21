"""Read-set audit (plan gate E7): a numpy forward that sees only the read rows agrees with the
JAX model, poisoning every unread row changes nothing, the read set equals the support of the
gradient (an independent definition of functional dependence), and the logger's counts."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import config_model, jitted, make_batch, random_params, tiny_cfg
from bes import budget
from bes import readset as R
from bes.config import BOS, CAND_FEATS, CAND_SLOTS, N_CLOCKS, REG_FIELDS
from bes.model import generate as G
from bes.model import model as M
from bes.roles import leaves_with_paths, role_of

TOL = 1e-4
VARIANTS = {
    "all_on": {},
    "switched": {"reg": 0.0, "cand_in": 0.0, "kind_3": 0.0, "clock": 0.0, "src_5": 0.0,
                 "src_2": 0.5},
    "no_sink": {"sink": 0.0, "conf": 0.0, "cand_out": 0.0},
}


def _seq(cfg, T=20, seed=0, seg_starts=(0, 8)) -> dict:
    """One sequence ``[T, ...]``; the last position lies in the second segment."""
    b = make_batch(cfg, B=1, T=T, seed=seed, seg_starts=(seg_starts,))
    feats, _, _ = M.split_batch(b)
    return {k: np.asarray(v)[0] for k, v in feats.items()}


def _batch1(f: dict) -> dict:
    return {k: np.asarray(v)[None] for k, v in f.items()}


def _jax_last(cfg, params, consts, f) -> np.ndarray:
    return np.asarray(jitted("logits_fn", cfg)(params, consts, _batch1(f)), np.float64)[0, -1]


@pytest.fixture(scope="module")
def tiny():
    cfg = tiny_cfg()
    return cfg, random_params(cfg, seed=1)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_reference_from_the_strict_read_set_matches_jax(tiny, variant):
    cfg, params = tiny
    consts = M.make_consts(cfg, **VARIANTS[variant])
    f = _seq(cfg, seed=2)
    reads = R.read_rows(cfg, consts, f, "strict")
    store = R.RowStore(params, reads, R.dense_paths(params, cfg, consts))
    z_ref = R.reference_logits(store, consts, f, cfg)
    assert R.max_logprob_diff(z_ref, _jax_last(cfg, params, consts, f)) <= TOL
    # rows of the first segment (positions 0..7) and of the BOS sink are not needed
    assert store.numbers() < sum(np.size(a) for _, a in leaves_with_paths(params))
    if variant == "switched":
        assert not any(p.startswith(("emb/reg/", "cand/in")) for p in reads)
    # dropping one needed row makes the reference fail (it really sees only the store)
    smaller = dict(reads)
    smaller["emb/tok"] = reads["emb/tok"][reads["emb/tok"] != f["tokens"][12]]
    with pytest.raises(KeyError):
        R.reference_logits(R.RowStore(params, smaller), consts, f, cfg)


@pytest.mark.parametrize("variant", ["all_on", "no_sink"])
def test_reference_step_from_the_cache_matches_jax(tiny, variant):
    """Standard convention: the jax cache of positions < T-1 plus the rows read at T-1."""
    cfg, params = tiny
    consts = M.make_consts(cfg, **VARIANTS[variant])
    f = _seq(cfg, seed=3)
    t = len(f["tokens"])
    dec = G.jitted(cfg, donate=False)
    _, _, cache = dec.prefill(params, consts, G.init_cache(cfg, 1, 32),
                              {k: v[None, :t - 1] for k, v in f.items()})
    step = R.read_rows(cfg, consts, f, "step")
    store = R.RowStore(params, step, R.dense_paths(params, cfg, consts))
    z_ref = R.reference_logits(store, consts, f, cfg, kv=R.kv_from_cache(cache, 0))
    assert R.max_logprob_diff(z_ref, _jax_last(cfg, params, consts, f)) <= TOL
    strict = R.read_rows(cfg, consts, f, "strict")
    assert all(set(rows) <= set(strict[p]) for p, rows in step.items())
    assert sum(map(len, step.values())) < sum(map(len, strict.values()))


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_poisoning_unread_rows_leaves_the_output_unchanged(tiny, variant):
    cfg, params = tiny
    consts = M.make_consts(cfg, **VARIANTS[variant])
    f = _seq(cfg, seed=4)
    reads = R.read_rows(cfg, consts, f, "strict")
    bad = R.poison(params, reads, scale=1e3, seed=5)
    np.testing.assert_array_equal(_jax_last(cfg, bad, consts, f), _jax_last(cfg, params, consts, f))
    ref = R.reference_logits(R.RowStore(bad, reads, R.dense_paths(bad, cfg, consts)), consts, f, cfg)
    assert R.max_logprob_diff(ref, _jax_last(cfg, params, consts, f)) <= TOL
    # positive control: poisoning one row that is read (token of a visible position) matters
    worse = R.poison(params, {**reads, "emb/tok": reads["emb/tok"][reads["emb/tok"] != f["tokens"][12]]},
                     scale=1e3, seed=5)
    assert np.max(np.abs(_jax_last(cfg, worse, consts, f) - _jax_last(cfg, params, consts, f))) > 1e-3
    # the trust biases are rows of the read set too: raising one read row changes the output
    # law, raising every unread one changes nothing (the poisoning above covered them as well)
    if VARIANTS[variant].get("cand_out", 1.0):
        bias = {p: r for p, r in reads.items() if p.startswith("trust/bias/")}
        assert len(bias) == len(CAND_FEATS)
        name = "count"
        moved = jax.tree.map(np.array, params)
        moved["trust"]["bias"][name][int(bias[f"trust/bias/{name}"][0])] += 20.0
        assert R.max_logprob_diff(_jax_last(cfg, moved, consts, f), _jax_last(cfg, params, consts, f)) > 1e-3
        unread = jax.tree.map(np.array, params)
        for nm, card in CAND_FEATS:
            rows = np.setdiff1d(np.arange(card), bias[f"trust/bias/{nm}"])
            unread["trust"]["bias"][nm][rows] += 20.0
        np.testing.assert_array_equal(_jax_last(cfg, unread, consts, f), _jax_last(cfg, params, consts, f))
    else:
        assert not any(p.startswith("trust/") for p in reads)


def test_poisoning_outside_the_step_set_with_a_cached_state(tiny):
    """Given the cache, only the rows of the last position matter; recomputing from scratch
    with the same poisoned tables does change the output (the strict set is larger)."""
    cfg, params = tiny
    consts = M.make_consts(cfg)
    f = _seq(cfg, seed=6)
    t = len(f["tokens"])
    dec = G.jitted(cfg, donate=False)
    _, _, cache = dec.prefill(params, consts, G.init_cache(cfg, 1, 32),
                              {k: v[None, :t - 1] for k, v in f.items()})
    bad = R.poison(params, R.read_rows(cfg, consts, f, "step"), scale=1e3, seed=7)
    row = {k: v[None, t - 1] for k, v in f.items() if k != "tokens"}
    tok = f["tokens"][None, t - 1]
    z0 = np.asarray(dec.decode_step(params, consts, cache, tok, row)[0])
    z1 = np.asarray(dec.decode_step(bad, consts, cache, tok, row)[0])
    np.testing.assert_array_equal(z1, z0)
    assert np.max(np.abs(_jax_last(cfg, bad, consts, f) - _jax_last(cfg, params, consts, f))) > 1e-3


def _support(grads) -> dict[str, set[int]]:
    out = {}
    for p, g in leaves_with_paths(grads):
        if role_of(p).kind == "lookup":
            g = np.asarray(g).reshape(np.shape(g)[0], -1)
            rows = set(np.flatnonzero(np.any(g != 0, axis=1)).tolist())
            if rows:
                out[p] = rows
    return out


def _as_sets(reads: dict) -> dict[str, set[int]]:
    return {p: set(r.tolist()) for p, r in reads.items() if len(r)}


@pytest.mark.slow
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_read_sets_equal_the_gradient_support(tiny, variant):
    """Independent check: a table row is in the strict (step) read set exactly when the
    gradient of the last position's log-probabilities (of the decode step with the cache held
    constant) is non-zero on it."""
    cfg, params = tiny
    _check_gradient_support(cfg, params, M.make_consts(cfg, **VARIANTS[variant]), _seq(cfg, seed=8))


@pytest.mark.slow
def test_out_of_vocabulary_candidates(tiny):
    """A candidate id >= V still reads its (clipped) input row, but it carries no mass in the
    output law, so its trust rows (features and biases) are not part of the read set."""
    cfg, params = tiny
    f = _seq(cfg, seed=11)
    f["cand_tok"][-1, :2] = cfg.vocab_size + 5
    f["cand_feat"][-1, :2, 0] = 1
    f["cand_feat"][-1, 2:, 0] = 0
    f["cand_tok"][-1, 2:] = -1
    consts = M.make_consts(cfg)
    reads = R.read_rows(cfg, consts, f, "step")
    assert cfg.vocab_size - 1 in reads["cand/in_tok"].tolist()
    assert not any(p.startswith("trust/") for p in reads)
    _check_gradient_support(cfg, params, consts, f)


@functools.lru_cache(maxsize=None)
def _grad_fns(cfg):
    """Jitted gradients (w.r.t. the parameters) of a random projection of the last position's
    log-probabilities: from scratch, and of one decode step with the cache held constant."""
    def strict_fn(p, consts, feats, proj):
        z = M.logits_fn(p, consts, feats, cfg)[0, -1]
        return jnp.sum(proj * jax.nn.log_softmax(z))

    def step_fn(p, consts, cache, tok, row, proj):
        z, _, _ = G.decode_step(p, consts, cache, tok, row, cfg)
        return jnp.sum(proj * jax.nn.log_softmax(z[0]))
    return jax.jit(jax.grad(strict_fn)), jax.jit(jax.grad(step_fn))


def _check_gradient_support(cfg, params, consts, f):
    t = len(f["tokens"])
    proj = jnp.asarray(np.random.default_rng(9).standard_normal(cfg.vocab_size), jnp.float32)
    g_strict, g_step = _grad_fns(cfg)
    g = g_strict(params, consts, _batch1(f), proj)
    assert _support(g) == _as_sets(R.read_rows(cfg, consts, f, "strict"))

    _, _, cache = G.jitted(cfg, donate=False).prefill(
        params, consts, G.init_cache(cfg, 1, 32), {k: v[None, :t - 1] for k, v in f.items()})
    row = {k: v[None, t - 1] for k, v in f.items() if k != "tokens"}
    g = g_step(params, consts, cache, f["tokens"][None, t - 1], row, proj)
    assert _support(g) == _as_sets(R.read_rows(cfg, consts, f, "step"))


def test_logger_counts_follow_the_read_sets(tiny):
    cfg, params = tiny
    consts = M.make_consts(cfg)
    f = _seq(cfg, T=24, seed=10, seg_starts=(0, 9))
    lg = R.ReadLogger(params, cfg, consts)
    counts = lg.log_sequence(f)
    # dense part counted independently: every dense leaf except the critic
    dense = sum(np.size(a) for p, a in leaves_with_paths(params)
                if role_of(p).kind == "dense" and not p.startswith("critic/"))
    assert lg.dense == dense
    widths = {p: int(np.prod(np.shape(a)[1:])) for p, a in leaves_with_paths(params)}
    for t, c in enumerate(counts):
        sub = {k: v[:t + 1] for k, v in f.items()}
        for conv, got in (("step", c.step), ("strict", c.strict)):
            rows = R.read_rows(cfg, consts, sub, conv)
            assert got == dense + sum(len(r) * widths[p] for p, r in rows.items()), (t, conv)
        assert c.step <= c.strict
    # a new segment restarts the window
    assert counts[9].strict == counts[9].step                   # BOS of a new segment
    s = lg.summary()
    assert s["steps"] == 24 and s["step_max"] <= s["strict_max"]
    with_critic = R.ReadLogger(params, cfg, consts, include_critic=True)
    assert with_critic.dense - lg.dense == np.size(params["critic"]["w1"]) + np.size(params["critic"]["w2"])
    assert with_critic.log(5, {}, extra_numbers=1000).step == with_critic.dense + widths["emb/tok"] \
        + widths["emb/ve"] + 1000


def _synthetic_sequence(cfg, T: int, n_types: int, seed: int) -> dict:
    """Zipf-distributed tokens over ``n_types`` ids, candidates drawn from earlier tokens, random
    registers and clocks: a rough stand-in for real text."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(cfg.vocab_size - 16, n_types, replace=False)
    p = 1.0 / np.arange(1, n_types + 1)
    tokens = ids[rng.choice(n_types, T, p=p / p.sum())].astype(np.int32)
    tokens[0] = BOS
    reg = np.stack([rng.integers(0, c, T) for _, c in REG_FIELDS], -1).astype(np.int16)
    cand_tok = np.full((T, CAND_SLOTS), -1, np.int32)
    cand_feat = np.zeros((T, CAND_SLOTS, len(CAND_FEATS)), np.int8)
    for t in range(1, T):
        n = rng.integers(0, CAND_SLOTS + 1)
        cand_tok[t, :n] = tokens[rng.integers(0, t, n)]
        cand_feat[t, :n, 0] = rng.integers(1, 5, n)
        cand_feat[t, :n, 1:] = rng.integers(0, 8, (n, len(CAND_FEATS) - 1))
    return {"tokens": tokens, "seg": np.zeros(T, np.int32), "reg": reg,
            "clock": rng.integers(0, 50, (T, N_CLOCKS)).astype(np.int32),
            "cand_tok": cand_tok, "cand_feat": cand_feat}


@pytest.mark.parametrize("name", ["final", pytest.param("no_library", marks=pytest.mark.slow)])
def test_counts_of_the_shipped_configurations(name):
    """Shapes only (no allocation): per-step counts stay under 20M and under the budget's
    standard count; the window union stays under the budget's strict count."""
    cfg = config_model(name)
    shapes = jax.eval_shape(functools.partial(M.init_params, cfg), jax.random.PRNGKey(0))
    consts = M.make_consts(cfg)
    b = budget.count(shapes, cfg)                 # parameters only (no library or index)
    for T, n_types in ((1024, 400), (2048, 700)):
        f = _synthetic_sequence(cfg, T, n_types, seed=T)
        lg = R.ReadLogger(shapes, cfg, consts, include_critic=True)
        counts = lg.log_sequence(f)
        distinct = len(set(f["tokens"].tolist()))
        assert lg.dense == sum(i.total for i in b.items if i.kind == "dense")
        assert max(c.step for c in counts) <= b.active_step <= budget.ACTIVE_CAP
        assert max(c.strict for c in counts) <= b.active_strict(T, distinct)
        # the input-side union only grows (the output-law rows of each position come on top)
        union = [sum(n for p, n in c.strict_rows.items() if not p.startswith("trust/"))
                 for c in counts]
        assert all(a <= b_ for a, b_ in zip(union, union[1:]))
        assert counts[-1].strict_rows["emb/tok"] == distinct - 1     # BOS is a sink
