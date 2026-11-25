"""DPO (``bes/train/prefopt.py``): log-probabilities, reference terms, the step, frozen leaves,
health checks and selection.

References are written in the tests: per-row log-probabilities from the unchunked scores of
``model.logits_fn`` (the output law, candidate tape included) with an explicit
``log_softmax`` / target pick / weighted sum,
rows built by list concatenation, the DPO loss and its gradient recomputed micro-batch by
micro-batch without ``lax.scan``. Every test shares one tiny model, one batch shape and one
optimiser (a role function that also freezes ``head/g``), so the DPO step compiles once.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.config import ASSISTANT, BOS, END, PAD, USER, PrefCfg, load_config
from bes.data import prefs as P
from bes.exact.features import exact_features
from bes.model import model as M
from bes.roles import Role, role_of
from bes.train import prefopt as D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T, PPM, A = 24, 2, 2
FROZEN_EXTRA = "head/g"


def extra_frozen_role_fn(path: str):
    """``bes.roles.role_of`` plus one extra frozen leaf (the critic is frozen there already)."""
    return Role("dense", "frozen") if path == FROZEN_EXTRA else role_of(path)


@pytest.fixture(scope="module")
def cfg():
    base = load_config(os.path.join(ROOT, "configs", "smoke.py"))
    model = dataclasses.replace(base.model, d_model=32, n_heads=1, head_dim=32, ve_dim=32,
                                head_rank=16, n_loops=2, max_seq=64, clock_planes=3, cand_dim=8,
                                critic_hidden=8, loss_chunk=20)
    return dataclasses.replace(base, model=model)


@pytest.fixture(scope="module")
def pref():
    return PrefCfg(seq_len=T, pairs_per_micro=PPM, n_micro=A, beta=0.1, warmup_frac=0.0,
                   cache_gb=1.0)


@pytest.fixture(scope="module")
def params(cfg):
    """Initial parameters with every zero-initialised table perturbed (trained-looking trust
    tables of the output law and candidate input), a non-trivial unembedding and a non-zero
    critic."""
    p = jax.jit(functools.partial(M.init_params, cfg.model))(jax.random.PRNGKey(0))
    rng = np.random.default_rng(1)
    flat, treedef = jax.tree_util.tree_flatten_with_path(p)
    out = []
    for path, leaf in flat:
        name = "/".join(k.key for k in path)
        a = np.asarray(leaf)
        noise = rng.standard_normal(a.shape).astype(np.float32)
        if name.startswith(("emb/reg/", "cand/in_feat/", "trust/feat/", "trust/bias/")) or name in (
                "cand/in_tok", "critic/w2"):
            a = a + 0.3 * noise
        elif name == "head/out":
            a = a + 0.5 * noise
        out.append(jnp.asarray(a, jnp.float32))
    return jax.tree_util.tree_unflatten(treedef, out)


@pytest.fixture(scope="module")
def consts(cfg):
    return M.make_consts(cfg.model)


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


def toy_pair(rng: np.random.Generator) -> P.Pair:
    """A learnable preference: chosen responses contain token 100, rejected ones token 101."""
    a, b, c = rng.integers(200, 230, size=3).tolist()
    prompt = [BOS, USER, a, b, c, a, b, END]
    return P.make_pair(prompt, [a, b, 100, c, 100], [a, 101, b, 101])


@pytest.fixture(scope="module")
def pairs():
    rng = np.random.default_rng(7)
    return [toy_pair(rng) for _ in range(24)]


def batches(pairs, tok, order=None):
    return P.PairBatches(pairs, T, PPM, A, order=order, tok=tok, cache_bytes=1 << 28)


@pytest.fixture(scope="module")
def optimizer(cfg, params, pref):
    return D.build_optimizer(cfg, params, pref, extra_frozen_role_fn)


@pytest.fixture(scope="module")
def step_fn(cfg, optimizer):
    return D.make_dpo_step(cfg.model, optimizer)


# ------------------------------------------------------------------------ references
def ref_row_logp(params, consts, cfg, tok, prompt, resp, *, tape=True):
    """Unchunked reference: one row built by concatenation, its features computed on their
    own, scores of the output law from ``logits_fn``, explicit log-softmax, weighted sum of
    the targets (``tape=False``: the network alone, ``cand_out`` switched off)."""
    content = list(prompt) + [ASSISTANT] + list(resp) + [END]
    L = len(content)
    tokens = np.array(content + [PAD] * (T + 1 - L), np.int32)[None]
    seg = np.array([[0 if t < L else 1 for t in range(T)]], np.int32)
    feats = {"tokens": tokens[:, :T], "seg": seg, **exact_features(tokens[:, :T], seg, tok)}
    c = consts if tape else M.make_consts(cfg.model, cand_out=0.0)
    z = np.asarray(_logits(cfg.model)(params, c, feats), np.float64)[0]         # [T, V]
    logp = z - np.log(np.sum(np.exp(z - z.max(-1, keepdims=True)), -1, keepdims=True)) - z.max(
        -1, keepdims=True)
    total = 0.0
    for t in range(len(prompt), L - 1):              # targets: response tokens and END
        total += logp[t, content[t + 1]]
    return total


@functools.lru_cache(maxsize=4)
def _logits(model_cfg):
    return jax.jit(lambda p, c, f: M.logits_fn(p, c, f, model_cfg))


# -------------------------------------------------------------------- log-probabilities
def test_row_logps_equal_the_unchunked_reference(params, consts, cfg, tok, pairs):
    data = batches(pairs[:4], tok)
    batch = data.get(0)
    got = np.asarray(D.make_logp_fn(cfg.model)(params, consts, batch))          # [A, B]
    rows = 2 * PPM * T                                  # several chunks, the last one padded
    assert cfg.model.loss_chunk < rows and rows % cfg.model.loss_chunk
    for a in range(A):
        for p in range(PPM):
            pair = pairs[int(batch["index"][a, p])]
            for r, resp in zip((2 * p, 2 * p + 1), (pair.chosen, pair.rejected)):
                want = ref_row_logp(params, consts, cfg, tok, pair.prompt.tolist(), resp.tolist())
                assert got[a, r] == pytest.approx(want, rel=1e-5, abs=1e-4)
    # the candidate tape's output law is part of the value
    pair = pairs[int(batch["index"][0, 0])]
    without = ref_row_logp(params, consts, cfg, tok, pair.prompt.tolist(), pair.chosen.tolist(),
                           tape=False)
    assert abs(without - got[0, 0]) > 1e-2
    # and the sum over rows is the loss function's weighted cross-entropy
    mb = {k: v[0] for k, v in batch.items() if k in D.DEVICE_KEYS}
    _, metrics = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, cfg.model))(params, consts, mb)
    assert float(np.sum(got[0])) == pytest.approx(-float(metrics["ce_sum"]), rel=1e-5)


def test_reference_logps_equal_a_direct_computation(params, consts, cfg, tok, pairs):
    sub = pairs[:7]                                            # 2 batches, the last one padded
    data = batches(sub, tok, order=P.shuffled_order(len(sub), 3))
    assert data.ref is None
    ref = D.compute_ref_logps(params, consts, data, cfg.model)
    assert ref.shape == (7, 2) and np.array_equal(data.ref, ref)
    for i, pair in enumerate(sub):
        for j, resp in enumerate((pair.chosen, pair.rejected)):
            want = ref_row_logp(params, consts, cfg, tok, pair.prompt.tolist(), resp.tolist())
            assert ref[i, j] == pytest.approx(want, rel=1e-5, abs=1e-4)
    # stored per row in every later batch
    batch = data.get(1)
    for a in range(A):
        for p in range(PPM):
            i = int(batch["index"][a, p])
            want = ref[i] if i >= 0 else np.zeros(2)
            np.testing.assert_array_equal(batch["ref"][a, 2 * p:2 * p + 2], want)


# ------------------------------------------------------------------------- DPO loss
def test_dpo_loss_is_log2_when_policy_equals_reference(params, consts, cfg, tok, pairs, pref,
                                                       optimizer, step_fn):
    logp = jnp.asarray(np.random.default_rng(0).normal(size=(3, 8)), jnp.float32)
    loss, margin = D.dpo_terms(logp, logp, 0.37)
    np.testing.assert_allclose(np.asarray(loss), math.log(2.0), rtol=0, atol=1e-7)
    assert np.all(np.asarray(margin) == 0)
    data = batches(pairs[:8], tok)
    ref = D.compute_ref_logps(params, consts, data, cfg.model)
    stats = D.preference_metrics(ref, ref, data.target_counts(), pref.beta)
    assert stats["loss"] == pytest.approx(math.log(2.0), abs=1e-12) and stats["acc"] == 0.5
    # the step reports the loss at its input parameters (= the reference)
    state = optimizer.init(params)
    p_copy = jax.tree.map(jnp.copy, params)
    _, _, m = step_fn(p_copy, state, consts, data.get(0), 0.0, pref.beta)
    assert float(m["loss"]) == pytest.approx(math.log(2.0), abs=1e-5)
    assert float(m["pairs"]) == A * PPM and int(m["skipped"]) == 0


def test_dpo_terms_match_the_formula():
    rng = np.random.default_rng(2)
    logp, ref = rng.normal(size=6), rng.normal(size=6)
    beta = 0.25
    loss, margin = D.dpo_terms(jnp.asarray(logp, jnp.float32), jnp.asarray(ref, jnp.float32), beta)
    for p in range(3):
        m = (logp[2 * p] - ref[2 * p]) - (logp[2 * p + 1] - ref[2 * p + 1])
        assert float(margin[p]) == pytest.approx(m, rel=1e-5, abs=1e-6)
        assert float(loss[p]) == pytest.approx(-math.log(1.0 / (1.0 + math.exp(-beta * m))),
                                               rel=1e-5)


def test_accumulated_gradient_equals_the_pairwise_mean(params, consts, cfg, tok, pairs, pref):
    data = batches(pairs[:3], tok)                       # A=2 micro-batches, one filler pair
    D.compute_ref_logps(params, consts, data, cfg.model)
    batch = {k: v for k, v in data.get(0).items() if k in D.DEVICE_KEYS}
    beta = 0.5
    grads, sums = jax.jit(lambda p, b: D.dpo_grads(p, consts, b, beta, cfg.model))(params, batch)

    def reference_loss(p):
        total, n = 0.0, 0.0
        for a in range(A):                               # no scan: one micro-batch at a time
            mb = {k: v[a] for k, v in batch.items()}
            logp = D.row_logps(p, consts, mb, cfg.model)
            for q in range(PPM):
                if mb["pair_mask"][q] == 0:
                    continue
                m = (logp[2 * q] - mb["ref"][2 * q]) - (logp[2 * q + 1] - mb["ref"][2 * q + 1])
                total = total - jax.nn.log_sigmoid(beta * m)
                n += 1
        return total / n

    want = jax.jit(jax.grad(reference_loss))(params)
    assert float(sums["n"]) == 3
    for (path, g), w in zip(jax.tree_util.tree_flatten_with_path(grads)[0], jax.tree.leaves(want)):
        w = np.asarray(w)
        scale = float(np.max(np.abs(w))) if w.size else 0.0   # float32 summation-order noise
        np.testing.assert_allclose(np.asarray(g), w, rtol=2e-4, atol=1e-4 * scale + 1e-9,
                                   err_msg=str(path))


def test_one_small_step_increases_the_margin(params, consts, cfg, tok, pairs, pref, optimizer,
                                             step_fn):
    data = batches(pairs[:4], tok)
    ref = D.compute_ref_logps(params, consts, data, cfg.model)
    new, _, m = step_fn(jax.tree.map(jnp.copy, params), optimizer.init(params), consts,
                        data.get(0), 1e-4, pref.beta)
    after = D.pair_logps(new, consts, data, cfg.model)
    before_margin = ref[:, 0] - ref[:, 1]
    after_margin = after[:, 0] - after[:, 1]
    assert np.sum(after_margin) > np.sum(before_margin)
    assert np.all(after_margin > before_margin)          # every pair moves the right way here


def test_frozen_and_critic_leaves_do_not_move(params, consts, cfg, tok, pairs, pref, optimizer,
                                              step_fn):
    data = batches(pairs[:8], tok)
    D.compute_ref_logps(params, consts, data, cfg.model)
    host_before = [np.array(x) for x in jax.tree.leaves(params)]
    new, report = D.train_epoch(params, consts, data, cfg, 1e-2, pref, optimizer=optimizer,
                                step_fn=step_fn)
    assert report["steps"] == 2 and report["finite"]
    flat_old = dict((D.path_str(k), v) for k, v in jax.tree_util.tree_flatten_with_path(params)[0])
    flat_new = dict((D.path_str(k), v) for k, v in jax.tree_util.tree_flatten_with_path(new)[0])
    trainable = set(D.trainable_paths(params, extra_frozen_role_fn))
    frozen = set(flat_old) - trainable
    assert frozen == {"critic/w1", "critic/w2", FROZEN_EXTRA}
    for path in frozen:
        np.testing.assert_array_equal(np.asarray(flat_new[path]), np.asarray(flat_old[path]))
    for path in ("head/out", "core/attn/wq", "core/mlp/w_up", "trust/proj", "emb/tok"):
        assert not np.array_equal(np.asarray(flat_new[path]), np.asarray(flat_old[path])), path
    # the caller's parameters are untouched (training ran on a copy, buffers not donated)
    for before, after in zip(host_before, jax.tree.leaves(params)):
        np.testing.assert_array_equal(before, np.asarray(after))
    default_frozen = set(flat_old) - set(D.trainable_paths(params))
    assert default_frozen == {"critic/w1", "critic/w2"}


def test_optimizer_is_plain_adam_with_one_rate(cfg, params, pref, optimizer):
    groups = set(optimizer.groups.values())
    assert groups == {"scalar", "frozen"}
    assert {lp.base_lr for lp in optimizer.plan if lp.group == "scalar"} == {1.0}
    assert optimizer.cfg.adam_b1 == pref.adam_b1 and optimizer.cfg.adam_b2 == pref.adam_b2
    assert optimizer.cfg.adam_eps == pref.adam_eps and optimizer.cfg.grad_clip == pref.grad_clip


def test_lr_factor_schedule():
    f = [D.lr_factor(k, 10, 0.2, 0.0) for k in range(10)]
    assert f[:2] == [0.5, 1.0]
    assert f[2] == 1.0 and all(a > b for a, b in zip(f[2:], f[3:])) and f[-1] > 0
    assert f[-1] == pytest.approx(1.0 - 7 / 8)
    assert D.lr_factor(0, 1, 0.1, 0.0) == 1.0
    assert [D.lr_factor(k, 4, 0.0, 0.5) for k in range(4)] == [1.0, 0.875, 0.75, 0.625]


# ---------------------------------------------------------------------- health checks
def start_metrics(acc=0.5, loss=2.0, mean_len=100.0, rep=0.05):
    return {"pref": {"acc": acc}, "heldout_loss": loss,
            "stats": {D.STAT_LEN: mean_len, D.STAT_REP: rep}}


def test_check_candidate_accepts_a_healthy_candidate():
    pref = PrefCfg()
    cand = start_metrics(acc=0.6, loss=2.0 * 1.029, mean_len=159.9, rep=0.0599)
    assert D.check_candidate(start_metrics(), cand, pref) == []
    cand = start_metrics(acc=0.51, loss=1.5, mean_len=70.1, rep=0.0)
    assert D.check_candidate(start_metrics(), cand, pref) == []


@pytest.mark.parametrize("change, code", [
    (dict(acc=0.5), "accuracy"),                     # not an improvement
    (dict(acc=0.45), "accuracy"),
    (dict(loss=2.0 * 1.031), "loss"),                # more than 3 % above
    (dict(loss=float("nan")), "loss"),
    (dict(mean_len=69.9), "length"),                 # below 0.7 x
    (dict(mean_len=160.1), "length"),                # above 1.6 x
    (dict(rep=0.0701), "repetition"),                # above 1.2 x + the 0.01 absolute margin
])
def test_check_candidate_rejects_each_reason(change, code):
    cand = start_metrics(acc=0.6, loss=2.0, mean_len=100.0, rep=0.05)
    for key, value in change.items():
        if key == "acc":
            cand["pref"]["acc"] = value
        elif key == "loss":
            cand["heldout_loss"] = value
        else:
            cand["stats"][D.STAT_LEN if key == "mean_len" else D.STAT_REP] = value
    assert [c for c, _ in D.check_candidate(start_metrics(), cand, PrefCfg())] == [code]


def test_check_candidate_nonfinite_and_min_gain():
    cand = {**start_metrics(acc=0.9), "finite": False}
    assert [c for c, _ in D.check_candidate(start_metrics(), cand, PrefCfg())] == ["nonfinite"]
    cand = start_metrics(acc=0.52)
    codes = [c for c, _ in D.check_candidate(start_metrics(), cand, PrefCfg(min_acc_gain=0.05))]
    assert codes == ["accuracy"]


def test_select_candidate_prefers_the_best_accepted_accuracy():
    c = [{"accepted": True, "pref": {"acc": 0.6}}, {"accepted": False, "pref": {"acc": 0.9}},
         {"accepted": True, "pref": {"acc": 0.7}}, {"accepted": True, "pref": {"acc": 0.7}}]
    assert D.select_candidate(c) == 2
    assert D.select_candidate([dict(x, accepted=False) for x in c]) is None
    assert D.select_candidate([]) is None


def test_response_stats_against_a_direct_count():
    responses = [[1, 2, 3, 4, 1, 2, 3, 4, 5], [7, 7, 7, 7, 7], [9, 9]]
    # response 0: 4-grams 1234 2341 3412 4123 1234* 2345 -> 1 repeat of 6
    # response 1: 7777 7777* -> 1 repeat of 2; response 2: no 4-gram -> rate 0
    s = D.response_stats(responses, max_new=9)
    assert s["len"] == [9.0, 5.0, 2.0] and s["rep"] == pytest.approx([1 / 6, 1 / 2, 0.0])
    assert s[D.STAT_LEN] == pytest.approx((9 + 5 + 2) / 3)
    assert s[D.STAT_REP] == pytest.approx((1 / 6 + 1 / 2) / 3)        # mean of per-response rates
    assert s["mean_len_se"] == pytest.approx(np.std([9, 5, 2], ddof=1) / np.sqrt(3))
    assert s["rep4_se"] == pytest.approx(np.std([1 / 6, 1 / 2, 0], ddof=1) / np.sqrt(3))
    assert s["n"] == 3 and s["max_new"] == 9
    empty = D.response_stats([])
    assert (empty[D.STAT_LEN], empty[D.STAT_REP], empty["n"]) == (0.0, 0.0, 0)


# ------------------------------------------------- noise-aware checks, simulated statistics
def draw_replies(rng, n):
    """Per-response statistics of ``n`` replies: lengths ~ Gamma(3, 40) tokens (mean 120), repeated
    4-gram rates 0 for 40 % of the replies and Beta(1.2, 18) otherwise (mean ~0.04)."""
    lengths = np.maximum(1.0, np.round(rng.gamma(3.0, 40.0, n)))
    rates = np.where(rng.random(n) < 0.4, 0.0, rng.beta(1.2, 18.0, n))
    return lengths, rates


def sim(lengths, rates, max_new=None):
    s = {D.STAT_LEN: float(np.mean(lengths)), D.STAT_REP: float(np.mean(rates)),
         "len": [float(x) for x in lengths], "rep": [float(x) for x in rates]}
    if max_new is not None:
        s["max_new"] = max_new
    return {"stats": s}


@pytest.mark.parametrize("n", [8, 64])
def test_an_unchanged_policy_passes_the_sampling_checks(n):
    """Two independent samples of the same policy: rejected in about 1 % of 2000 trials with 8
    prompts and never with 64 (before, the ratio rules rejected 49 % and 22 %)."""
    rng = np.random.default_rng(n)
    pref = PrefCfg()
    rejected = sum(bool(D.check_stats(sim(*draw_replies(rng, n)), sim(*draw_replies(rng, n)), pref))
                   for _ in range(2000))
    assert rejected / 2000 <= (0.03 if n == 8 else 0.002)


def test_clearly_degraded_candidates_are_still_rejected():
    """Paired samples (the same prompts and seeds, as the run draws them) of 64 prompts: twice as
    long, repetition doubled, half as long: each rejected for its reason in at least 99 % of the
    trials; a candidate within the bands passes; a held-out loss 10 % higher is rejected."""
    rng = np.random.default_rng(1)
    pref = PrefCfg()
    hits = {"longer": 0, "repeats": 0, "shorter": 0, "within": 0}
    for _ in range(500):
        lengths, rates = draw_replies(rng, 64)
        start = sim(lengths, rates)
        cases = {"longer": sim(2 * lengths, rates), "repeats": sim(lengths, 2 * rates),
                 "shorter": sim(np.maximum(1.0, lengths // 2), rates),
                 "within": sim(1.2 * lengths, 1.05 * rates)}
        want = {"longer": ["length"], "repeats": ["repetition"], "shorter": ["length"], "within": []}
        for name, cand in cases.items():
            hits[name] += [c for c, _ in D.check_stats(start, cand, pref)] == want[name]
    assert all(v >= 0.99 * 500 for v in hits.values()), hits
    base = start_metrics()
    assert [c for c, _ in D.check_loss(base, dict(base, heldout_loss=1.1 * base["heldout_loss"]),
                                       pref)] == ["loss"]


def test_a_reply_cut_at_the_limit_bounds_the_candidate():
    """Replies cut at ``max_new`` (most of them at a short limit): a twice-as-long candidate can
    only be seen on the replies that were not cut; with the limit known it is rejected, without
    it the bound 1.6 x (cut length) is never reached."""
    rng = np.random.default_rng(2)
    pref = PrefCfg()
    cap, seen, blind = 128, 0, 0
    for _ in range(300):
        lengths = np.maximum(1.0, np.round(rng.gamma(3.0, 40.0, 64)))
        r = np.zeros(64)
        l0, l1 = np.minimum(cap, lengths), np.minimum(cap, 2 * lengths)
        seen += bool(D.check_stats(sim(l0, r, cap), sim(l1, r, cap), pref))
        blind += bool(D.check_stats(sim(l0, r), sim(l1, r), pref))
    assert seen == 300 and blind == 0


def test_a_zero_repetition_start_can_be_passed():
    pref = PrefCfg()
    start = sim([100.0] * 16, [0.0] * 16)
    assert D.check_stats(start, sim([100.0] * 16, [0.0] * 15 + [0.02]), pref) == []
    means_only = {"stats": {D.STAT_LEN: 60.0, D.STAT_REP: 0.0}}
    assert D.check_stats(means_only, {"stats": {D.STAT_LEN: 60.0, D.STAT_REP: 0.001}}, pref) == []
    assert [c for c, _ in D.check_stats(means_only, {"stats": {D.STAT_LEN: 60.0, D.STAT_REP: 0.05}},
                                        pref)] == ["repetition"]


def test_the_accuracy_gain_must_exceed_its_standard_error():
    """64 held-out pairs. A useless candidate (each margin positive with probability 1/2) passed
    in 44 % of the trials before (any accuracy above 0.5); now in about 13 %. A real gain (3 of 4
    pairs ranked right) passes."""
    rng = np.random.default_rng(3)
    pref = PrefCfg()
    n = 64
    start = {"pref": D.preference_metrics(np.zeros((n, 2)), np.zeros((n, 2)))}
    assert start["pref"]["acc"] == 0.5 and start["pref"]["acc_se"] == 0.0

    def cand(p_right):
        margin = np.where(rng.random(n) < p_right, 1.0, -1.0) * rng.uniform(0.1, 1.0, n)
        return {"pref": D.preference_metrics(np.stack([margin, np.zeros(n)], 1), np.zeros((n, 2)))}

    useless = sum(not D.check_accuracy(start, cand(0.5), pref) for _ in range(2000)) / 2000
    good = sum(not D.check_accuracy(start, cand(0.75), pref) for _ in range(500)) / 500
    assert useless <= 0.2 and good >= 0.99
    # acc_se against a direct computation: 5 wins, 2 losses, 1 tie
    margins = np.asarray([1.0, 2.0, 0.5, 3.0, 0.1, -1.0, -0.2, 0.0])
    m = D.preference_metrics(np.stack([margins, np.zeros(8)], 1), np.zeros((8, 2)))
    won = np.asarray([1, 1, 1, 1, 1, 0, 0, 0.5])
    assert m["acc"] == pytest.approx(won.mean())
    assert m["acc_se"] == pytest.approx(np.sqrt(np.sum((won - won.mean()) ** 2) / 7 / 8))


def test_heldout_loss_helper_matches_the_loss_function(params, consts, cfg, tok, pairs):
    batch = {k: v for k, v in batches(pairs[:4], tok).get(0).items() if k in D.DEVICE_KEYS}
    fn = D.make_heldout_loss_fn([batch], consts, cfg.model)
    ce, weight = 0.0, 0.0
    for a in range(A):
        mb = {k: v[a] for k, v in batch.items()}
        _, m = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, cfg.model))(params, consts, mb)
        ce, weight = ce + float(m["ce_sum"]), weight + float(m["weight"])
    assert fn(params) == pytest.approx(ce / weight, rel=1e-5)


# ------------------------------------------------------------------- run_prefopt
class Scripted:
    """Stand-in health measurement: returns the scripted values in call order."""

    def __init__(self, values):
        self.values, self.calls = list(values), []

    def __call__(self, params):
        self.calls.append(params)
        return self.values[len(self.calls) - 1]


def stats(mean_len, rep):
    return {D.STAT_LEN: mean_len, D.STAT_REP: rep}


def split(pairs):
    return pairs[:16], pairs[16:]


def run(params, pairs, cfg, pref, tok, lrs, loss_values, stat_values, **kw):
    loss_fn, stats_fn = Scripted(loss_values), Scripted(stat_values)
    out, report = D.run_prefopt(params, split(pairs), cfg, lrs=lrs, heldout_loss_fn=loss_fn,
                                sample_stats_fn=stats_fn, pref=pref, tok=tok,
                                role_fn=extra_frozen_role_fn, **kw)
    return out, report, loss_fn, stats_fn


def test_run_prefopt_accepts_and_selects_the_best_candidate(params, cfg, pref, tok, pairs):
    out, rep, loss_fn, stats_fn = run(
        params, pairs, cfg, pref, tok, (3e-3, 1e-2), [2.0, 2.01, 2.02],
        [stats(10, 0.1), stats(11, 0.1), stats(12, 0.11)])
    assert rep["accepted"] and len(rep["candidates"]) == 2
    assert all(c["accepted"] and c["reasons"] == [] for c in rep["candidates"])
    assert rep["start"]["pref"]["acc"] == 0.5 and rep["start"]["heldout_loss"] == 2.0
    accs = [c["pref"]["acc"] for c in rep["candidates"]]
    assert all(a > 0.5 for a in accs)
    best = int(np.argmax(accs))                          # first one on ties
    assert rep["lr"] == (3e-3, 1e-2)[best]
    assert loss_fn.calls[0] is params and stats_fn.calls[0] is params
    assert len(loss_fn.calls) == 3 and len(stats_fn.calls) == 3
    # the returned parameters are the selected candidate, the start is untouched
    assert out is not params
    moved = [not np.array_equal(np.asarray(a), np.asarray(b))
             for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(params))]
    assert any(moved)
    assert rep["n_train_pairs"] == 16 and rep["n_heldout_pairs"] == 8 and rep["steps_per_epoch"] == 4
    chosen = rep["candidates"][best]["train"]
    assert chosen["first_loss"] == pytest.approx(math.log(2.0), abs=1e-5)
    assert chosen["skipped"] == 0 and chosen["train_acc"] > 0.5


def test_run_prefopt_rejects_a_loss_rise_and_keeps_the_other(params, cfg, pref, tok, pairs):
    out, rep, loss_fn, stats_fn = run(
        params, pairs, cfg, pref, tok, (3e-3, 1e-2), [2.0, 2.01, 2.5],
        [stats(10, 0.1), stats(11, 0.1)])
    first, second = rep["candidates"]
    assert first["accepted"] and not second["accepted"] and second["reasons"] == ["loss"]
    assert "stats" not in second and second["checked"] == ["nonfinite", "accuracy", "loss"]
    assert rep["accepted"] and rep["lr"] == 3e-3
    assert len(stats_fn.calls) == 2                      # sampling skipped after the failure


def test_run_prefopt_falls_back_to_the_start(params, cfg, pref, tok, pairs):
    # lr 0 leaves the policy at the reference (accuracy stays 0.5), nan makes it non-finite,
    # the third candidate trains but its samples get too long
    out, rep, loss_fn, stats_fn = run(
        params, pairs, cfg, pref, tok, (0.0, float("nan"), 3e-3), [2.0, 2.0],
        [stats(10, 0.1), stats(17, 0.1)])
    assert out is params and not rep["accepted"] and rep["lr"] is None
    codes = [c["reasons"] for c in rep["candidates"]]
    assert codes == [["accuracy"], ["nonfinite"], ["length"]]
    assert rep["candidates"][0]["pref"]["acc"] == 0.5
    assert "no candidate passed" in rep["reason"]
    # evaluate_all runs every check of every finite candidate
    out, rep, loss_fn, stats_fn = run(
        params, pairs, cfg, pref, tok, (0.0,), [2.0, 2.5], [stats(10, 0.1), stats(10, 0.5)],
        evaluate_all=True)
    assert rep["candidates"][0]["reasons"] == ["accuracy", "loss", "repetition"]


def test_run_prefopt_without_pairs_keeps_the_start(params, cfg, pref, pairs):
    calls = []
    out, rep = D.run_prefopt(params, (pairs[:4], []), cfg, heldout_loss_fn=calls.append,
                             sample_stats_fn=calls.append, pref=pref)
    assert out is params and not rep["accepted"] and "held-out" in rep["reason"] and not calls


def test_run_prefopt_requires_the_sampling_statistics(params, cfg, pref, tok, pairs):
    with pytest.raises(KeyError):
        D.run_prefopt(params, split(pairs), cfg, lrs=(1e-3,), heldout_loss_fn=lambda p: 1.0,
                      sample_stats_fn=lambda p: {"length": 3}, pref=pref, tok=tok,
                      role_fn=extra_frozen_role_fn)


def test_run_prefopt_with_feature_processes_gives_the_same_result(params, cfg, pref, tok, pairs):
    """``PrefCfg.feature_workers``: the rows' exact features come from a pool of processes;
    the candidate and its report are the ones of the run that computes them itself."""
    script = ([2.0, 2.01], [stats(10, 0.1), stats(11, 0.1)])
    out0, rep0, _, _ = run(params, pairs, cfg, pref, tok, (3e-3,), *script)
    out1, rep1, _, _ = run(params, pairs, cfg, dataclasses.replace(pref, feature_workers=2), tok, (3e-3,), *script)
    assert "feature_warm" not in rep0
    assert rep1["feature_warm"]["workers"] == 2 and rep1["feature_warm"]["batches"] == 4 + 2
    assert rep1["accepted"] == rep0["accepted"] and rep1["candidates"][0]["pref"] == rep0["candidates"][0]["pref"]
    for a, b in zip(jax.tree.leaves(out0), jax.tree.leaves(out1)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_broken_feature_pool_is_reported_as_such(params, cfg, pref, tok, pairs, monkeypatch):
    def boom(self, workers, log=None):
        raise OSError("no processes")

    monkeypatch.setattr(P.PairBatches, "warm", boom)
    with pytest.raises(D.FeatureWarmError, match="no processes"):
        run(params, pairs, cfg, dataclasses.replace(pref, feature_workers=2), tok, (3e-3,),
            [2.0, 2.01], [stats(10, 0.1), stats(11, 0.1)])
