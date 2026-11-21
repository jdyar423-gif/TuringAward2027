"""DPO (``bes/train/prefopt.py``): log-probabilities, reference terms, the step, frozen leaves,
health checks and selection.

References are written in the tests: per-row log-probabilities from the unchunked tilted
logits of ``model.logits_fn`` with an explicit ``log_softmax`` / target pick / weighted sum,
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
    """Initial parameters with every zero-initialised table perturbed (an active tilt and
    candidate input), a non-trivial unembedding and a non-zero critic."""
    p = jax.jit(functools.partial(M.init_params, cfg.model))(jax.random.PRNGKey(0))
    rng = np.random.default_rng(1)
    flat, treedef = jax.tree_util.tree_flatten_with_path(p)
    out = []
    for path, leaf in flat:
        name = "/".join(k.key for k in path)
        a = np.asarray(leaf)
        noise = rng.standard_normal(a.shape).astype(np.float32)
        if name.startswith(("emb/reg/", "cand/in_feat/", "trust/feat/")) or name in (
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
def ref_row_logp(params, consts, cfg, tok, prompt, resp, *, tilt=True):
    """Unchunked reference: one row built by concatenation, its features computed on their
    own, tilted logits from ``logits_fn``, explicit log-softmax, weighted sum of the targets."""
    content = list(prompt) + [ASSISTANT] + list(resp) + [END]
    L = len(content)
    tokens = np.array(content + [PAD] * (T + 1 - L), np.int32)[None]
    seg = np.array([[0 if t < L else 1 for t in range(T)]], np.int32)
    feats = {"tokens": tokens[:, :T], "seg": seg, **exact_features(tokens[:, :T], seg, tok)}
    c = consts if tilt else M.make_consts(cfg.model, cand_out=0.0)
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
    # the candidate tilt is part of the value
    pair = pairs[int(batch["index"][0, 0])]
    untilted = ref_row_logp(params, consts, cfg, tok, pair.prompt.tolist(), pair.chosen.tolist(),
                            tilt=False)
    assert abs(untilted - got[0, 0]) > 1e-2
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
    (dict(rep=0.0601), "repetition"),                # above 1.2 x
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
    # response 1: 7777 7777* -> 1 repeat of 2; response 2: none
    s = D.response_stats(responses)
    assert s[D.STAT_LEN] == pytest.approx((9 + 5 + 2) / 3)
    assert s[D.STAT_REP] == pytest.approx(2 / 8)
    assert D.response_stats([]) == {D.STAT_LEN: 0.0, D.STAT_REP: 0.0}


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
