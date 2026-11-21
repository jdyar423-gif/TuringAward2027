"""Tests of the micro training bench (``bes/eval/microbench.py``) and the E2 queue script.

A tiny model (d=32, 2 loops, rows of 64 tokens) trains on synthetic in-memory pools, so the
tests do not depend on ``data_cache``; the real micro configuration is only checked for valid
arm configurations.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from bes.config import BOS, END, PAD
from bes.data.samples import loss_weights
from bes.eval import microbench as mb
from bes.tokenizer import load as load_tok

REPO = Path(__file__).resolve().parents[2]
WORDS = ("river stone light house garden window market silver morning letter music orange "
         "forest bridge winter summer paper table story number planet animal school friend "
         "doctor island mountain coffee cloud train").split()

TINY_CFG = '''
from bes.config import Cfg, DataCfg, ModelCfg, OptimCfg, RunCfg
CFG = Cfg(model=ModelCfg(d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16, n_loops=2,
                         max_seq=64, clock_planes=3, cand_dim=8, critic_hidden=8,
                         table_grad="take"),
          optim=OptimCfg(tokens_per_step=256, micro_seqs=2),
          data=DataCfg(seq_len=64, seq_len_late=64), run=RunCfg(name="tiny"))
'''


def _docs(seed: int):
    rng = np.random.default_rng(seed)
    while True:
        yield " ".join(rng.choice(WORDS, int(rng.integers(10, 90)))) + "."


def _convs(seed: int):
    rng = np.random.default_rng(seed)
    while True:
        q = " ".join(rng.choice(WORDS, int(rng.integers(3, 12))))
        a = " ".join(rng.choice(WORDS, int(rng.integers(3, 30))))
        yield [{"role": "user", "content": q + "?"}, {"role": "assistant", "content": a + "."}]


def _build_web(split, T, n_rows, tok, samples_dir):
    return mb.document_rows(_docs(0 if split == "train" else 1), tok, T, n_rows)


def _build_chat(split, T, n_rows, tok, samples_dir):
    return mb.sample_rows(mb.conversation_samples(_convs(2 if split == "train" else 3), tok, T), T, n_rows)


mb.register_pool("tweb", _build_web, lambda split, sd: {"test": "tweb", "split": split})
mb.register_pool("tchat", _build_chat, lambda split, sd: {"test": "tchat", "split": split})
MIX = (("tweb", 0.75), ("tchat", 0.25))


@pytest.fixture(scope="module", autouse=True)
def _compile_cache(tmp_path_factory):
    """Persistent XLA cache for this module: the same tiny arm trained again compiles once."""
    import jax
    from jax.experimental.compilation_cache import compilation_cache as cc

    old = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    jax.config.update("jax_compilation_cache_dir", str(tmp_path_factory.mktemp("jax_cache")))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    cc.reset_cache()
    yield
    jax.config.update("jax_compilation_cache_dir", old[0])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", old[1])
    cc.reset_cache()


@pytest.fixture(scope="module")
def tok():
    return load_tok()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    cfg_path = root / "tiny_cfg.py"
    cfg_path.write_text(TINY_CFG)
    bench = mb.Bench(config=str(cfg_path), seq_len=64, micro_rows=2, tokens_per_step=256,
                     eval_fracs=(0.5, 1.0), eval_rows=4, eval_chunk=4, eval_domains=("tweb", "tchat"),
                     pool_rows=(("tweb", 48), ("tchat", 24)), pool_root=str(root / "pools"), log_every=1)
    return bench, root


# ------------------------------------------------------------------------ data
def test_row_kinds_follow_the_shares():
    kinds = mb.row_kinds([0.75, 0.25], 400)
    for n in range(1, 401):                         # every prefix within one row of its share
        assert abs((kinds[:n] == 1).sum() - 0.25 * n) <= 1
    assert all((kinds[i:i + 4] == 0).sum() == 3 for i in range(0, 400, 4))  # 3 + 1 per micro-batch
    three = mb.row_kinds([0.5, 0.3, 0.2], 1000)
    assert [(three == k).sum() for k in range(3)] == [500, 300, 200]


def test_document_rows_follow_the_packing_rules(tok):
    docs = [" ".join(WORDS[i % len(WORDS)] for i in range(k, k + 40 + 7 * k)) for k in range(12)]
    arrays, stats = mb.document_rows(docs, tok, 64, 100)
    toks = arrays["tokens"].astype(np.int64)
    n = toks.shape[0]
    assert stats["docs"] == 12 and n == stats["rows"]
    assert (toks[:, 0] == BOS).all()
    # the rows hold the documents' tokens in order; a cut document continues behind a fresh BOS
    flat = [t for row in toks for t in row if t not in (BOS, PAD)]
    assert flat == [t for d in docs for t in tok.encode(d)]
    for r in range(n):
        expected, k = np.empty(64, np.int64), -1     # segment id: pieces in order, padding last
        for t in range(64):
            k += int(toks[r, t] == BOS)
            expected[t] = k
        expected[toks[r, :64] == PAD] = int((toks[r] == BOS).sum())
        assert np.array_equal(arrays["seg"][r], expected)
        assert np.array_equal(arrays["w"][r], loss_weights(toks[r], np.zeros(65, np.int64))[1:])
    assert (toks[:-1] != PAD).all()                  # only the last row may be padded


def test_conversation_rows_keep_whole_conversations(tok):
    convs = [c for _, c in zip(range(30), _convs(5))]
    arrays, stats = mb.sample_rows(mb.conversation_samples(convs, tok, 64), 64, 100)
    toks, roles = arrays["tokens"].astype(np.int64), arrays["roles"].astype(np.int64)
    assert stats["samples"] == 30
    for r in range(toks.shape[0]):
        assert toks[r, 0] == BOS
        assert np.array_equal(arrays["w"][r], loss_weights(toks[r], roles[r])[1:])
        live = toks[r][toks[r] != PAD]
        assert live[-1] == END                      # every piece ends with the assistant END
    from bes.data import chat_format as cf
    rendered = sorted((tuple(i.tolist()), tuple(r.tolist())) for i, r in
                      (cf.render_conversation(c, tok) for c in convs))
    pieces = []
    for r in range(toks.shape[0]):                   # split every row at its BOS tokens
        starts = np.flatnonzero(toks[r] == BOS).tolist() + [int((toks[r] != PAD).sum())]
        pieces += [(tuple(toks[r, a:b].tolist()), tuple(roles[r, a:b].tolist()))
                   for a, b in zip(starts[:-1], starts[1:])]
    assert sorted(pieces) == rendered                # best fit reorders, never cuts or drops


def test_pools_are_built_once_and_cached(tok, tmp_path):
    calls = []

    def build(split, T, n_rows, tok_, sd):
        calls.append(split)
        return mb.document_rows(_docs(7), tok_, T, n_rows)

    mb.register_pool("tcache", build, lambda split, sd: {"test": "tcache"})
    a = mb.get_pool("tcache", "train", 64, 10, tok, root=tmp_path)
    b = mb.get_pool("tcache", "train", 64, 10, tok, root=tmp_path)
    assert calls == ["train"] and a.n == b.n == 10
    for k in ("tokens", "seg", "w", "roles"):
        assert np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k)))
    mb.get_pool("tcache", "train", 64, 12, tok, root=tmp_path)      # another size: another pool
    assert calls == ["train", "train"]


def test_batches_are_a_pure_function_of_seed_and_step(tiny, tok):
    bench, _ = tiny
    pools = {d: mb.get_pool(d, "train", 64, bench.rows_for(d), tok, root=bench.pool_root) for d, _ in MIX}
    one = mb.BatchMaker(pools, MIX, 0, (2, 2, 64), 30, tok)
    two = mb.BatchMaker(pools, MIX, 0, (2, 2, 64), 30, tok)
    other = mb.BatchMaker(pools, MIX, 1, (2, 2, 64), 30, tok)
    for s in (0, 7, 29):
        x, y = one.batch(s), two.batch(s)
        assert all(np.array_equal(x[k], y[k]) for k in x)
        assert x["tokens"].shape == (2, 2, 65) and x["cand_feat"].shape == (2, 2, 64, 16, 6)
    assert not np.array_equal(one.batch(0)["tokens"], other.batch(0)["tokens"])
    # 30 steps x 4 rows = 120 rows: 90 web rows over a pool of 48 = two passes, each a permutation
    web = one.plan[one.plan[:, 0] == 0, 1]
    assert one.passes["tweb"] == 2 and sorted(web[:48]) == list(range(48))


# ---------------------------------------------------------------- training
def _digest(batch) -> str:
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(k.encode())
        h.update(np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def queue_run(tiny, tok):
    """Two tiny arms through the in-process queue (4 steps each), with the batch digests."""
    bench, root = tiny
    out = root / "queue"
    seen: dict[str, list[str]] = {}
    arms = [mb.Arm("ref", kind="reference", mixture=MIX),
            mb.Arm("no_tape", ("model.use_cand_in=false", "model.use_cand_out=false"),
                   kind="remove_new", mixture=MIX)]

    def runner(arm):
        seen[arm.name] = []
        mb.run_arm(arm, out, 1024, bench, log=lambda m: None, tok=tok,
                   batch_hook=lambda s, b: seen[arm.name].append(_digest(b)))

    ran = mb.run_queue(arms, out, 1024, bench, runner=runner, log=lambda m: None)
    return out, arms, seen, ran


def test_two_arms_train_on_bit_identical_batches(queue_run):
    out, arms, seen, ran = queue_run
    assert ran == ["ref", "no_tape"]
    assert len(seen["ref"]) == 4 and seen["ref"] == seen["no_tape"]
    recs = mb.latest_done(mb.read_jsonl(out / mb.RESULTS))
    assert set(recs) == {"ref", "no_tape"}
    for r in recs.values():
        assert r["steps"] == 4 and [c["tokens"] for c in r["curve"]] == [512, 1024]
        assert all(math.isfinite(c["eval"][d]["loss"]) for c in r["curve"] for d in ("tweb", "tchat", "tchat_assistant"))
        assert r["config"]["optim"]["horizon_tokens"] == 1024.0
        assert r["config"]["optim"]["tokens_per_step"] == 256 and r["shape"] == [2, 2, 64]
        assert (out / "arms" / r["name"] / "params").exists()
        assert r["peak_rss_mb"] > 0 and r["timing"]["sec_per_mtok"] > 0
    ref = recs["ref"]
    assert set(ref["curve"][-1]["views"]) == {"tape_off", "reg_off", "sink_off"}
    assert recs["no_tape"]["curve"][-1]["eval"]["tweb"]["tape_mass"] == 0.0
    assert ref["curve"][-1]["eval"]["tweb"]["tape_mass"] > 0.0
    # the reference with the tape switched off at evaluation equals neither its own loss nor nan
    off = ref["curve"][-1]["views"]["tape_off"]["tweb"]["loss"]
    assert math.isfinite(off) and off != ref["curve"][-1]["eval"]["tweb"]["loss"]


def test_queue_skips_finished_arms(queue_run, tiny, tok):
    out, arms, _, _ = queue_run
    bench, _ = tiny
    calls = []
    assert mb.run_queue(arms, out, 1024, bench, runner=calls.append, log=lambda m: None) == []
    assert calls == []
    third = mb.Arm("no_reg", ("model.use_reg=false",), kind="remove_new", mixture=MIX)
    assert mb.run_queue(arms + [third], out, 1024, bench, runner=calls.append, log=lambda m: None) == ["no_reg"]
    assert [a.name for a in calls] == ["no_reg"]
    # another budget is another run
    assert mb.finished(out, arms[0], 2048, bench) is None


def test_loaded_parameters_match_the_record(queue_run):
    out, _, _, _ = queue_run
    rec, cfg, params = mb.load_arm(out, "ref")
    assert cfg.model.d_model == 32 and rec["name"] == "ref"
    assert params["emb"]["tok"].shape == (8192, 32)


def test_a_killed_arm_resumes_to_the_uninterrupted_result(queue_run, tiny, tok):
    bench, root = tiny
    out, _, _, _ = queue_run                       # holds the uninterrupted reference run
    arm = mb.Arm("ref", kind="reference", mixture=MIX)
    quiet = dict(log=lambda m: None, tok=tok)
    full = mb.latest_done(mb.read_jsonl(out / mb.RESULTS))["ref"]
    part = mb.run_arm(arm, root / "killed", 1024, bench, max_steps=3, **quiet)   # killed after step 3
    assert part["status"] == "partial" and (root / "killed" / "arms" / "ref" / "state").exists()
    resumed = mb.run_arm(arm, root / "killed", 1024, bench, **quiet)
    assert resumed["status"] == "done" and resumed["timing"]["resumes"] == 1
    assert not (root / "killed" / "arms" / "ref" / "state").exists()
    for a, b in zip(full["curve"], resumed["curve"]):
        assert a["eval"]["tweb"]["ce"] == b["eval"]["tweb"]["ce"]
    _, _, p_full = mb.load_arm(out, "ref")
    _, _, p_res = mb.load_arm(root / "killed", "ref")
    import jax
    for x, y in zip(jax.tree.leaves(p_full), jax.tree.leaves(p_res)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# --------------------------------------------------------------- arms, configs
def test_arm_round_trip_and_hash(tiny):
    bench, _ = tiny
    a = mb.Arm("x", ("model.n_loops=4",), (("reg", 0.0),), seed=3, kind="setting", mixture=MIX)
    assert mb.Arm.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    h = mb.spec_hash(a, 1000, bench)
    assert h == mb.spec_hash(mb.Arm.from_dict(a.to_dict()), 1000, bench)
    assert h != mb.spec_hash(a, 2000, bench)
    assert h != mb.spec_hash(mb.Arm("x", ("model.n_loops=5",), (("reg", 0.0),), seed=3, mixture=MIX), 1000, bench)
    labelled = mb.Arm("x", ("model.n_loops=4",), (("reg", 0.0),), seed=3, kind="setting", mixture=MIX,
                      component="label", note="note")
    assert mb.spec_hash(labelled, 1000, bench) == h                 # labels do not change the run


def test_e2_arms_resolve_to_valid_configs():
    sys.path.insert(0, str(REPO / "scripts"))
    import micro_ablate as MA
    from bes.model.model import check_cfg
    from bes.optim.build import group_lrs
    from bes.train.schedule import micro_shape

    names = [a.name for a in MA.E2_ARMS]
    assert len(names) == len(set(names)) and names[:2] == ["ref", "ref_s1"]
    assert {"no_tape", "tape_out_only", "tape_in_only", "no_reg", "no_clock", "no_sink", "no_conf",
            "ve_0", "ve_wide", "muon", "loops_4", "loops_12", "lr_feature_x0.3", "lr_feature_x3",
            "mass_offset_4", "mass_offset_8", "no_transfer_rule"} <= set(names)
    lrs = {}
    for a in MA.E2_ARMS:
        cfg = mb.arm_config(a, 1_000_000, mb.Bench())
        check_cfg(cfg.model)
        assert micro_shape(cfg, cfg.data.seq_len) == (2, 4, 512)
        assert cfg.optim.horizon_tokens == 1e6
        lrs[a.name] = group_lrs(cfg)
    # the transfer rule is active for the reference and absent in its arm
    assert lrs["ref"]["embed"] < 0.2 * lrs["no_transfer_rule"]["embed"]
    assert lrs["ref"]["matrix"] != lrs["no_transfer_rule"]["matrix"]
    assert {n for n, _ in MA.SKIPPED} == {"sink_prior"}


# ------------------------------------------------------------- decision rule
def test_paired_diff_by_hand():
    delta, se = mb.paired_diff([2.0, 4.0], [1.0, 1.0], [1.0, 2.0])
    assert delta == pytest.approx(4 / 3) and se == pytest.approx(2 / 9)
    assert mb.paired_diff([3.0, 3.0], [3.0, 3.0], [1.0, 1.0]) == (0.0, 0.0)


@pytest.mark.parametrize("kind,g_last,g_early,se,expected", [
    ("remove_new", 0.012, 0.020, 0.001, "keep"),          # 1.2% >= 1% and >= 0.5 x 2%
    ("remove_new", 0.012, 0.030, 0.001, "switch off"),    # fades: 1.2% < 0.5 x 3%
    ("remove_new", 0.008, 0.004, 0.001, "switch off"),    # below 1%
    ("remove_new", 0.015, -0.01, 0.001, "keep"),          # no early gain: only the 1% bar
    ("remove_evidence", -0.004, None, 0.001, "keep"),     # better by 0.4% < margin 0.5%
    ("remove_evidence", -0.006, None, 0.001, "remove"),   # better by 0.6% > margin 0.5%
    ("remove_evidence", -0.006, None, 0.004, "keep"),     # margin 2 x 0.4% = 0.8%
    ("remove_evidence", 0.050, None, 0.001, "keep"),
    ("setting", -0.010, None, 0.002, "adopt"),
    ("setting", -0.010, None, 0.006, "keep default"),     # margin 1.2%
    ("setting", 0.020, None, 0.000, "keep default"),
])
def test_decision_rule_hand_cases(kind, g_last, g_early, se, expected):
    assert mb.decide(kind, g_last, g_early, se)[0] == expected


def _record(name, kind, losses, seed=0, tokens=4000, views=None, mixture=(("web", 0.75), ("chat", 0.25))):
    """A synthetic finished record: ``losses[i] = (web, chat)`` per-token losses at the
    checkpoints 1000 and 4000; two rows of weight 2 per domain (rows equal, so the sampling SE
    is 0 and SE_pair is the replicate noise)."""
    curve = []
    for t, (lw, lc) in zip((1000, 4000), losses):
        ev = {"web": {"loss": lw, "ce": [2 * lw, 2 * lw], "w": [2.0, 2.0], "tape_gain": 0.1,
                      "tape_mass": 0.5, "sink_mass": [0.1, 0.1]},
              "chat": {"loss": lc, "ce": [2 * lc, 2 * lc], "w": [2.0, 2.0]}}
        curve.append({"tokens": t, "step": t // 1000, "eval": ev, "views": views or {}})
    arm = mb.Arm(name, kind=kind, seed=seed, mixture=mixture)
    return {"type": "arm", "status": "done", "name": name, "arm": arm.to_dict(), "spec_hash": name,
            "tokens": tokens, "steps": 4, "shape": [2, 4, 512], "tokens_per_step": 1000,
            "config": {"model": {"d_model": 128, "n_loops": 8}}, "bench": {"config": "configs/micro/bench.py",
                                                                        "eval_rows": 2, "eval_domains": ["web", "chat"]},
            "n_params": 1000, "lrs": {"matrix": 0.01}, "batch_factor": 0.1, "data_factor": 9.0,
            "flops_per_token": {"train": 3e7}, "timing": {"sec_per_mtok": 1200.0}, "peak_rss_mb": 1500.0,
            "data": {"token_shares": {"web": 0.78, "chat": 0.22}}, "curve": curve, "skipped_steps": 0}


def test_e2_report_applies_the_rule_to_synthetic_records():
    ref = _record("ref", "reference", [(5.0, 4.0), (4.0, 3.0)],
                  views={"tape_off": {"web": {"loss": 4.4, "ce": [8.8, 8.8], "w": [2.0, 2.0]},
                                      "chat": {"loss": 3.3, "ce": [6.6, 6.6], "w": [2.0, 2.0]}}})
    for c in ref["curve"]:
        c["views"] = ref["curve"][-1]["views"]
    recs = [
        ref,
        _record("ref_s1", "replicate", [(5.01, 4.0), (4.004, 3.003)], seed=1),         # +0.1%
        _record("keep_me", "remove_new", [(5.15, 4.12), (4.08, 3.06)]),                 # +3% early, +2% last
        _record("fading", "remove_new", [(5.15, 4.12), (4.048, 3.036)]),                # +3% early, +1.2% last
        _record("evid", "remove_evidence", [(5.0, 4.0), (3.96, 2.97)]),                 # -1% last
        _record("evid_small", "remove_evidence", [(5.0, 4.0), (3.984, 2.988)]),         # -0.4% last
        _record("better", "setting", [(5.0, 4.0), (3.96, 2.97)]),                       # -1%
    ]
    rows = {r["name"]: r for r in mb.arm_table(recs)}
    mix_last = {n: r["rel"][4000] for n, r in rows.items()}
    assert mix_last["keep_me"] == pytest.approx(0.02) and rows["keep_me"]["rel"][1000] == pytest.approx(0.03)
    assert mix_last["fading"] == pytest.approx(0.012)
    assert rows["keep_me"]["noise"] == pytest.approx(0.001) and rows["keep_me"]["se_pair"] == pytest.approx(0.001)
    assert rows["keep_me"]["margin"] == pytest.approx(0.005)
    assert rows["keep_me"]["decision"] == "keep"
    assert rows["fading"]["decision"] == "switch off"
    assert rows["evid"]["decision"] == "remove"
    assert rows["evid_small"]["decision"] == "keep"
    assert rows["better"]["decision"] == "adopt"
    assert rows["ref_s1"]["decision"] == "-"
    text = mb.e2_report(recs)
    assert "| `fading` | remove_new |" in text and "**switch off**" in text and "**adopt**" in text
    assert "Evaluation-time switches" in text and "+10.00%" in text      # tape_off view: 4.4 vs 4.0
    assert "What a micro run can say" in text


def test_e2_report_placeholder_without_results():
    text = mb.e2_report([], arms=[mb.Arm("ref", kind="reference")], skipped=[("sink_prior", "no hook")])
    assert "Not run yet" in text and "`sink_prior`: skipped" in text and "## Rule" in text


def test_editing_the_bench_config_invalidates_finished_arms(tiny, tmp_path):
    bench, _ = tiny
    cfg = tmp_path / "edited_cfg.py"
    cfg.write_text(Path(bench.config).read_text())
    edited = mb.Bench(**{**{f: getattr(bench, f) for f in bench.__dataclass_fields__}, "config": str(cfg)})
    arm = mb.Arm("ref", kind="reference", mixture=MIX)
    before = mb.spec_hash(arm, 1024, edited)
    cfg.write_text(cfg.read_text().replace("n_loops=2", "n_loops=3"))
    assert mb.spec_hash(arm, 1024, edited) != before
