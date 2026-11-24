"""Graft arms of the micro bench (``bes.eval.microbench.run_graft_arm``), the arms of the graft
experiment in ``scripts/micro_ablate.py`` and ``scripts/micro_graft_report.py``.

The tiny model of ``tests/eval/test_microbench.py`` (d=32, 2 loops, rows of 64 tokens) trains on
synthetic in-memory pools: 8 steps of 256 tokens, checkpoints after 2, 4, 6 and 8 steps. The base
is the tiny reference, the target adds the v9 page (output gains and 2 token neurons in a neuron
space of 8), the graft arm switches at 0.5 (step 4).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from bes.eval import microbench as mb
from bes.tokenizer import load as load_tok

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import micro_ablate as MA  # noqa: E402
import micro_graft_report as GR  # noqa: E402

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


mb.register_pool("gweb", lambda split, T, n, tok, sd: mb.document_rows(_docs(10 if split == "train" else 11), tok, T, n),
                 lambda split, sd: {"test": "gweb", "split": split})
mb.register_pool("gchat", lambda split, T, n, tok, sd: mb.sample_rows(
    mb.conversation_samples(_convs(12 if split == "train" else 13), tok, T), T, n),
    lambda split, sd: {"test": "gchat", "split": split})
MIX = (("gweb", 0.75), ("gchat", 0.25))
PAGES = ("model.gain_o=true", "model.tok_neurons=2", "model.tok_dim=8")
NEW = ("gain/o", "page/key", "page/pin", "page/pout", "page/val")
TOKENS = 2048                                  # 8 steps of 256 tokens
BASE = mb.Arm("ref", kind="reference", mixture=MIX)
TARGET = mb.Arm("pages", PAGES, kind="add_new", mixture=MIX)
GRAFT = mb.Arm("graft_50", PAGES, kind="graft", mixture=MIX, graft=mb.Graft(BASE, 0.5, "pages"))
QUIET = dict(log=lambda m: None)


@pytest.fixture(scope="module", autouse=True)
def _compile_cache(tmp_path_factory):
    """Persistent XLA cache for this module: every program compiles once."""
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
    root = tmp_path_factory.mktemp("graft")
    (root / "tiny_cfg.py").write_text(TINY_CFG)
    bench = mb.Bench(config=str(root / "tiny_cfg.py"), seq_len=64, micro_rows=2, tokens_per_step=256,
                     eval_fracs=(0.25, 0.5, 0.75, 1.0), eval_rows=4, eval_chunk=4,
                     eval_domains=("gweb", "gchat"), pool_rows=(("gweb", 48), ("gchat", 24)),
                     pool_root=str(root / "pools"), log_every=1)
    return bench, root


def _digest(batch) -> str:
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(k.encode())
        h.update(np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def runs(tiny, tok):
    """``a/``: the base (keeping its states) and the target; ``b/``: the graft arm with a trained
    trunk; the graft arm again in ``a/``, where it finds the base's state at step 4."""
    bench, root = tiny
    seen: dict[str, list[str]] = {}

    def run(arm, out, key, **kw):
        seen[key] = []
        return mb.run_arm(arm, root / out, TOKENS, bench, tok=tok,
                          batch_hook=lambda s, b: seen[key].append(_digest(b)), **QUIET, **kw)

    recs = {"base": run(BASE, "a", "base", keep_states=True), "target": run(TARGET, "a", "target"),
            "trained": run(GRAFT, "b", "trained", trunk="train"), "loaded": run(GRAFT, "a", "loaded")}
    return {"recs": recs, "digests": seen, "root": root, "bench": bench}


def _flat(tree) -> dict[str, np.ndarray]:
    from bes.train import checkpoint as C

    paths, leaves, _ = C.flatten_with_paths(tree)
    return {p: np.asarray(x) for p, x in zip(paths, leaves)}


# ---------------------------------------------------------------- the trunk
def test_the_graft_arm_is_the_base_arm_up_to_the_graft_point(runs):
    base, g = runs["recs"]["base"], runs["recs"]["trained"]
    assert g["status"] == "done" and g["graft"]["trunk"]["source"] == "trained"
    assert g["graft"]["step"] == 4 and g["graft"]["tokens"] == 1024
    assert [c["phase"] for c in g["curve"]] == ["trunk", "trunk", "target", "target"]
    assert [c["tokens"] for c in g["curve"]] == [c["tokens"] for c in base["curve"]] == [512, 1024, 1536, 2048]
    # train losses of steps 0-3 and held-out evaluations at 2 and 4 steps: exactly the base's
    assert [e for e in g["train_log"] if e[0] < 4] == [e for e in base["train_log"] if e[0] < 4]
    for cg, cb in zip(g["curve"][:2], base["curve"][:2]):
        assert cg["eval"] == cb["eval"] and cg["train_loss"] == cb["train_loss"]
    # afterwards it is another model
    assert g["curve"][-1]["eval"]["gweb"]["ce"] != base["curve"][-1]["eval"]["gweb"]["ce"]
    # the same batches at every step, before and after the graft: the data order continues
    d = runs["digests"]
    assert len(d["trained"]) == 8 and d["trained"] == d["base"] == d["target"]


def test_the_function_is_preserved_at_the_graft_point(runs):
    from bes.train import schedule as S

    g = runs["recs"]["trained"]
    chk = g["graft"]["check"]
    assert chk["ok"] and math.isfinite(chk["loss_before"])
    assert chk["difference"] == chk["loss_after"] - chk["loss_before"]
    assert abs(chk["difference"]) <= chk["tolerance"] == mb.GRAFT_TOL * max(1.0, abs(chk["loss_before"]))
    assert chk["batch"].startswith("the first 4 rows of held-out gweb")
    base_at_4 = runs["recs"]["base"]["curve"][1]["eval"]
    for d, v in g["graft"]["heldout"].items():
        assert v["before"] == base_at_4[d]["loss"]
        assert abs(v["difference"]) <= mb.GRAFT_TOL * max(1.0, abs(v["before"])), d
    assert set(g["graft"]["heldout"]) == {"gweb", "gchat", "gchat_assistant"}
    carried = g["graft"]["carried"]
    assert carried["step"] == 4 and carried["count"] == 4 and carried["p"] == 0.5
    cfg = mb.cfg_from_dict(g["config"])
    assert carried["lr_factor"] == S.lr_factor(0.5, cfg.optim)
    leaves = g["graft"]["leaves"]
    assert tuple(leaves["new_params"]) == tuple(f"state/params/{p}" for p in NEW)
    assert leaves["new_param_numbers"] > 0 and leaves["copied_params"] > 0


def test_the_grafted_state_is_the_trunk_with_a_fresh_target_page(runs, tok):
    """Stopped right after the graft: the state checkpoint holds the base's state at step 4 (every
    shared leaf, optimiser slot, count and step) and, for the new leaves, the values and slots of
    a fresh tree of the target with the same seed; resumed, it ends as the uninterrupted run."""
    from bes.model import model as M
    from bes.optim import build as build_opt
    from bes.train import checkpoint as C
    from bes.train.train import abstract

    bench, root = runs["bench"], runs["root"]
    part = mb.run_arm(GRAFT, root / "c", TOKENS, bench, tok=tok, max_steps=4, trunk="train", **QUIET)
    assert part["status"] == "partial" and part["graft"]["check"]["ok"]
    info = C.latest(root / "c" / "arms" / GRAFT.name / "state")
    man = C.read_manifest(info.path)
    assert info.step == 4 and man["meta"]["phase"] == "target" and len(C.list_checkpoints(info.path.parent)) == 1

    cfg_t = mb.arm_config(TARGET, TOKENS, bench)
    assert cfg_t.model == mb.arm_config(GRAFT, TOKENS, bench).model
    params = M.init_params(cfg_t.model, jax.random.key(TARGET.seed))
    opt = build_opt(cfg_t, params)
    fresh = {"state": mb._Trainee(TARGET, cfg_t).fresh_state()}
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(fresh["state"].params), jax.tree.leaves(params)))
    got, _ = C.load(info, abstract(fresh), config_hash=mb.spec_hash(GRAFT, TOKENS, bench))
    grafted, fresh_flat = _flat(got), _flat(fresh)
    assert set(grafted) == set(fresh_flat)
    # the base's kept state at step 4 (the trunk): every one of its leaves is carried
    kept = [c for c in C.list_checkpoints(root / "a" / "arms" / BASE.name / mb.STATES) if c.step == 4][0]
    cfg_b = mb.arm_config(BASE, TOKENS, bench)
    trunk_t = {"state": mb._Trainee(BASE, cfg_b).fresh_state()}
    trunk, _ = C.load(kept, abstract(trunk_t))
    trunk_flat = {p: v for p, v in _flat(trunk).items() if not p.startswith("state/consts")}
    for p, v in trunk_flat.items():
        assert np.array_equal(grafted[p], v), p
    assert int(grafted["state/step"]) == 4 and int(grafted["state/opt_state/count"]) == 4
    new = set(grafted) - set(trunk_flat)
    assert {p for p in new if p.startswith("state/params/")} == {f"state/params/{p}" for p in NEW}
    assert {p.split("/")[3].replace("%2F", "/") for p in new if p.startswith("state/opt_state/slots/")} == set(NEW)
    for p in new:                                        # new leaves, their slots, the target's consts
        assert np.array_equal(grafted[p], fresh_flat[p]), p
    assert float(np.abs(grafted["state/params/page/val"]).max()) == 0.0
    assert float(np.abs(grafted["state/params/gain/o"]).max()) == 0.0
    assert float(np.abs(grafted["state/params/page/key"]).max()) > 0.0           # random keys of the seed
    # resumed to the end: the uninterrupted graft arm
    done = mb.run_arm(GRAFT, root / "c", TOKENS, bench, tok=tok, trunk="train", **QUIET)
    assert done["status"] == "done" and done["timing"]["resumes"] == 1
    assert done["graft"]["trunk"]["source"] == "trained"
    ref = runs["recs"]["trained"]
    assert [c["eval"] for c in done["curve"]] == [c["eval"] for c in ref["curve"]]
    _, _, p_res = mb.load_arm(root / "c", GRAFT.name)
    _, _, p_ref = mb.load_arm(root / "b", GRAFT.name)
    for x, y in zip(jax.tree.leaves(p_res), jax.tree.leaves(p_ref)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert not (root / "c" / "arms" / GRAFT.name / "state").exists()


def test_a_run_stopped_at_the_graft_resumes_with_the_graft(runs, tok, monkeypatch):
    """Killed after the trunk's state at step 4 was saved and before the graft was written: the
    next run grafts first instead of training the trunk on."""
    from bes.train import checkpoint as C

    bench, root = runs["bench"], runs["root"]
    real = mb._graft_run

    def killed():
        raise KeyboardInterrupt("killed at the graft")

    monkeypatch.setattr(mb, "_graft_run", killed)
    with pytest.raises(KeyboardInterrupt):
        mb.run_arm(GRAFT, root / "f", TOKENS, bench, tok=tok, trunk="train", **QUIET)
    info = C.latest(root / "f" / "arms" / GRAFT.name / "state")
    assert info.step == 4 and C.read_manifest(info.path)["meta"]["phase"] == "trunk"
    monkeypatch.setattr(mb, "_graft_run", real)
    done = mb.run_arm(GRAFT, root / "f", TOKENS, bench, tok=tok, trunk="train", **QUIET)
    assert done["status"] == "done" and done["timing"]["resumes"] == 1 and done["graft"]["check"]["ok"]
    assert [c["eval"] for c in done["curve"]] == [c["eval"] for c in runs["recs"]["trained"]["curve"]]


# ---------------------------------------------------------- a kept trunk
def test_a_kept_trunk_is_loaded_and_gives_the_same_run(runs):
    bench, root = runs["bench"], runs["root"]
    loaded, trained, base = runs["recs"]["loaded"], runs["recs"]["trained"], runs["recs"]["base"]
    src = loaded["graft"]["trunk"]
    assert src["source"] == "loaded" and src["spec_hash"] == mb.spec_hash(BASE, TOKENS, bench)
    assert Path(src["checkpoint"]).name.endswith("step-000000004") and "/arms/ref/states/" in src["checkpoint"]
    assert loaded["timing"]["phases"]["trunk"]["timed_steps"] == 0
    assert runs["digests"]["loaded"] == runs["digests"]["base"][4:]     # only the target's steps ran
    for cl, ct, cb in zip(loaded["curve"], trained["curve"], base["curve"]):
        assert cl["eval"] == ct["eval"]
    for cl, cb in zip(loaded["curve"][:2], base["curve"][:2]):
        assert cl["eval"] == cb["eval"]
    assert loaded["graft"]["check"] == {**trained["graft"]["check"]}
    _, _, p_l = mb.load_arm(root / "a", GRAFT.name)
    _, _, p_t = mb.load_arm(root / "b", GRAFT.name)
    for x, y in zip(jax.tree.leaves(p_l), jax.tree.leaves(p_t)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    # the base keeps its states at the evaluation checkpoints before the last (never pruned)
    states = sorted(p.name for p in (root / "a" / "arms" / "ref" / mb.STATES).iterdir())
    assert [int(n.split("-")[-1]) for n in states] == [2, 4, 6]
    assert not (root / "a" / "arms" / "ref" / "state").exists()


def test_trunk_lookup_and_its_refusals(runs, tok, tmp_path):
    bench, root = runs["bench"], runs["root"]
    h = mb.spec_hash(BASE, TOKENS, bench)
    found, rejected = mb.find_trunk(BASE, 4, h, [root / "a"])
    assert found is not None and found.step == 4 and rejected == []
    none, rejected = mb.find_trunk(BASE, 4, "0" * 16, [root / "a"])
    assert none is None and len(rejected) == 1 and "not 0000" in rejected[0]["reason"]
    assert mb.find_trunk(BASE, 3, h, [root / "a"]) == (None, [])
    # one --out sub-directory per arm: <dir>/<base>/arms/<base>/states
    (tmp_path / "layout").mkdir()
    (tmp_path / "layout" / "ref").symlink_to(root / "a")
    assert mb.find_trunk(BASE, 4, h, [tmp_path / "layout"])[0].step == 4
    # trunk="load" without a kept state: refused before anything is trained or written
    with pytest.raises(FileNotFoundError, match="no kept state"):
        mb.run_arm(GRAFT, tmp_path / "d", TOKENS, bench, tok=tok, trunk="load", **QUIET)
    assert not (tmp_path / "d" / mb.RESULTS).exists()
    with pytest.raises(ValueError, match="trunk="):
        mb.run_graft_arm(GRAFT, tmp_path / "d", TOKENS, bench, tok=tok, trunk="sometimes", **QUIET)


# ------------------------------------------------------------------ refusals
def test_a_graft_fraction_must_be_an_evaluation_fraction(tiny):
    bench, _ = tiny
    plan = mb.graft_plan(GRAFT, TOKENS, bench)
    assert (plan.step, plan.n_steps, plan.eval_steps) == (4, 8, (2, 4, 6, 8)) and plan.new_params == NEW
    with pytest.raises(ValueError, match="not an evaluation fraction"):
        mb.graft_plan(GRAFT, TOKENS, dataclasses.replace(bench, eval_fracs=(0.25, 0.75, 1.0)))
    with pytest.raises(ValueError, match="too small"):
        mb.graft_plan(GRAFT, 512, bench)                 # 2 steps: 0.25 and 0.5 fall on one step
    lr = mb.Arm("g_lr", PAGES + ("optim.lr_matrix=0.02",), kind="graft", mixture=MIX, graft=mb.Graft(BASE, 0.5))
    with pytest.raises(ValueError, match="optimiser settings differ"):
        mb.graft_plan(lr, TOKENS, bench)
    lost = mb.Arm("g_noreg", PAGES + ("model.use_reg=false",), kind="graft", mixture=MIX, graft=mb.Graft(BASE, 0.5))
    with pytest.raises(ValueError, match="parameters of the base"):
        mb.graft_plan(lost, TOKENS, bench)
    with pytest.raises(ValueError, match="seed"):
        mb.Arm("g_seed", PAGES, seed=1, kind="graft", mixture=MIX, graft=mb.Graft(BASE, 0.5))
    with pytest.raises(ValueError, match="strictly between"):
        mb.Arm("g_end", PAGES, kind="graft", mixture=MIX, graft=mb.Graft(BASE, 1.0))
    with pytest.raises(ValueError, match="go together"):
        mb.Arm("g_kind", PAGES, kind="add_new", mixture=MIX, graft=mb.Graft(BASE, 0.5))
    rt = mb.Arm.from_dict(json.loads(json.dumps(GRAFT.to_dict())))
    assert rt.graft.base.name == "ref" and rt.graft.frac == 0.5 and rt.graft.target == "pages"
    assert mb.spec_hash(rt, TOKENS, bench) == mb.spec_hash(GRAFT, TOKENS, bench)
    assert mb.spec_hash(GRAFT, TOKENS, bench) != mb.spec_hash(
        dataclasses.replace(GRAFT, graft=mb.Graft(BASE, 0.25, "pages")), TOKENS, bench)


def test_a_graft_that_changes_the_function_fails_loudly(runs, tok, tmp_path):
    """A target that switches the tape's output masses off computes another function from the
    same parameters: the check refuses it and the record says so."""
    bench = runs["bench"]
    bad = mb.Arm("graft_bad", PAGES, (("cand_out", 0.0),), kind="graft", mixture=MIX, graft=mb.Graft(BASE, 0.5))
    with pytest.raises(RuntimeError, match="does not compute the trunk's function"):
        mb.run_arm(bad, tmp_path, TOKENS, bench, tok=tok, **QUIET)
    rec = mb.read_jsonl(tmp_path / mb.RESULTS)[-1]
    assert rec["status"] == "failed" and not rec["graft"]["check"]["ok"]
    assert abs(rec["graft"]["check"]["difference"]) > rec["graft"]["check"]["tolerance"]
    assert mb.finished(tmp_path, bad, TOKENS, bench) is None


# ------------------------------------------------------- the experiment's arms
#: spec hash and configuration hash of every E2 arm at HEAD 8a86bcb, before graft arms existed
#: (default bench, 1M tokens)
HEAD_HASHES = {
    "attn_gate": ("011504cbe5f1f2c4", "5f98c1614dc91884"),
    "gain_o": ("11b16504c67afffb", "d1f959a27ca9632d"),
    "loops_12": ("159b832c6529c5d2", "99e79f129cda154a"),
    "loops_4": ("18cc4cb3c5f0b6c7", "3915859b7406fb74"),
    "lr_feature_x0.3": ("9c30bc9ddf2cae78", "88281a22bc0f3d9b"),
    "lr_feature_x3": ("fd2986828d4b49d8", "e2a549ba645b8c6a"),
    "mass_offset_4": ("4d8f128accb884f5", "c83b45b7b6b23672"),
    "mass_offset_8": ("1651ac799e0ee1d6", "b752041f180055f4"),
    "matrix_lr_x0.33": ("7166119eca9ba8c5", "333c99a121094af3"),
    "matrix_lr_x3": ("de061430e20de6d5", "0f8092946773915e"),
    "muon": ("a0f835ad8590e9a5", "3b50d94a9aab047e"),
    "no_clock": ("c18a7752ae035397", "ecdae8e8c3027bae"),
    "no_conf": ("59e5ddbeac08221c", "3b7438ac521c586b"),
    "no_reg": ("bae8e62018c37e6b", "b9aabfeee5a2244e"),
    "no_sink": ("aed48e75b7a8ec8e", "d5fb2d1dd68b5b8d"),
    "no_tape": ("00950b0f945b2f8d", "a944aa38f5358181"),
    "ref": ("9da1f480edabb87d", "627d0074266ef2fc"),
    "ref_s1": ("166cb3b3c450f336", "aa00b17c51102921"),
    "sel_attn": ("470ec073db8f8c03", "ba6e5cb90eef109a"),
    "src5_train_0": ("c7eedfcbe4b48a76", "cd923b7280a5378c"),
    "src5_train_025": ("5d81e9b274d531c8", "bac7359bad38ec47"),
    "tape_in_only": ("4816c330f305669d", "f216a58820d6d817"),
    "tape_out_only": ("8c7cb263b044c7d0", "153e0da005e9b3d9"),
    "tok_neurons": ("3ee3221fff945e0f", "178af7acc2bdec4e"),
    "transfer_rule": ("fe7f7a3894ef95b7", "c212529f28b7805e"),
    "v9_pages": ("e127207ff5611c71", "7d20af87db3e1aa5"),
    "v9_pages_decay": ("00ca776e2171c9c2", "577e403aa70fcc68"),
    "v9_pages_gate": ("8af24680fce130d4", "51e7ee04c55369b2"),
    "v9_pages_lr03": ("3c9d9e65c1eca115", "53226d6dad3430ef"),
    "ve_0": ("bc8da3ad65de0795", "4a977279860c0135"),
    "ve_wide": ("43d8c937a1b370f0", "4a95cfcfb1472fb4"),
}
#: the same at 1M tokens with the experiment's checkpoints 0.25, 0.5, 0.75, 1.0
HEAD_HASHES_FRACS4 = {"ref": "954e31b4357f27e0", "ref_s1": "191639c5aacb4b5d", "v9_pages": "be345a315ba4bde1"}
E2_NAMES_AT_HEAD = (
    "ref", "ref_s1", "no_tape", "transfer_rule", "tape_out_only", "tape_in_only", "no_reg", "no_clock",
    "no_sink", "no_conf", "ve_0", "ve_wide", "muon", "loops_4", "loops_12", "lr_feature_x0.3",
    "lr_feature_x3", "mass_offset_4", "mass_offset_8", "matrix_lr_x3", "matrix_lr_x0.33",
    "src5_train_025", "src5_train_0", "gain_o", "tok_neurons", "attn_gate", "sel_attn", "v9_pages",
    "v9_pages_gate", "v9_pages_decay", "v9_pages_lr03")


def test_the_e2_arms_keep_their_hashes_and_their_queue():
    from bes.config import config_hash

    assert tuple(a.name for a in MA.E2_ARMS) == E2_NAMES_AT_HEAD
    assert set(HEAD_HASHES) == set(E2_NAMES_AT_HEAD)
    for a in MA.E2_ARMS:
        assert "graft" not in a.to_dict() and a.graft is None
        assert (mb.spec_hash(a, 1_000_000, mb.Bench()),
                config_hash(mb.arm_config(a, 1_000_000, mb.Bench()))) == HEAD_HASHES[a.name], a.name
    four = mb.Bench(eval_fracs=(0.25, 0.5, 0.75, 1.0))
    by = {a.name: a for a in MA.E2_ARMS}
    for name, h in HEAD_HASHES_FRACS4.items():
        assert mb.spec_hash(by[name], 1_000_000, four) == h
    # roots of the pools and samples are not part of an arm's identity
    moved = mb.Bench(pool_root="/elsewhere/pools", samples_dir="/elsewhere/samples")
    assert mb.spec_hash(by["ref"], 1_000_000, moved) == HEAD_HASHES["ref"][0]


def test_the_graft_experiment_arms():
    from bes.train import schedule as S

    names = MA.GRAFT_EXPERIMENT
    assert len(names) == len(set(names)) == 15
    assert not {a.name for a in MA.GRAFT_ARMS} & set(E2_NAMES_AT_HEAD)
    by = {a.name: a for a in MA.ALL_ARMS}
    assert set(names) <= set(by)
    assert [a.name for a in MA.select(MA.ALL_ARMS, "graft_50_s1,ref_s2")] == ["graft_50_s1", "ref_s2"]
    four = mb.Bench(eval_fracs=(0.25, 0.5, 0.75, 1.0))
    v9 = mb.arm_config(by["v9_pages"], 1_000_000, four)
    refc = mb.arm_config(by["ref"], 1_000_000, four)
    for seed in (0, 1, 2):
        sfx = f"_s{seed}" if seed else ""
        assert by[f"ref{sfx}"].seed == by[f"v9_pages{sfx}"].seed == seed
        assert by[f"v9_pages{sfx}"].overrides == by["v9_pages"].overrides
        for frac in (0.25, 0.5, 0.75):
            a = by[f"graft_{round(100 * frac)}{sfx}"]
            assert a.kind == "graft" and a.seed == seed and a.graft.frac == frac
            assert a.graft.base is by[f"ref{sfx}"] and a.graft.target == f"v9_pages{sfx}"
            cfg = mb.arm_config(a, 1_000_000, four)
            assert cfg.model == v9.model and cfg.optim == v9.optim == refc.optim
            plan = mb.graft_plan(a, 1_000_000, four)
            assert plan.step == {0.25: 61, 0.5: 122, 0.75: 183}[frac] and plan.n_steps == 244
            assert plan.new_params == ("gain/o", "page/key", "page/pin", "page/pout", "page/val")
            assert plan.base_hash == mb.spec_hash(by[f"ref{sfx}"], 1_000_000, four)
            assert S.micro_shape(cfg, cfg.data.seq_len) == (2, 4, 512)
    # the default checkpoints (0.25, 0.5, 1.0) have no 0.75
    with pytest.raises(ValueError, match="not an evaluation fraction"):
        mb.graft_plan(by["graft_75"], 1_000_000, mb.Bench(), shapes=False)
    assert mb.graft_plan(by["graft_50"], 1_000_000, mb.Bench(), shapes=False).step == 122
    # 120,000 tokens (29 steps) still has four distinct checkpoints, 16,000 does not
    assert mb.graft_plan(by["graft_50"], 120_000, four, shapes=False).eval_steps == (7, 14, 22, 29)
    with pytest.raises(ValueError, match="too small"):
        mb.graft_plan(by["graft_50"], 12_288, four, shapes=False)


# ------------------------------------------------------------------- report
def _rec(name: str, seed: int, losses: dict[int, tuple[float, float, float]], graft: dict | None = None,
         tokens: int = 4000) -> dict:
    curve = [{"tokens": t, "step": t // 1000, "eval": {"web": {"loss": w}, "chat": {"loss": c},
                                                      "assistant": {"loss": a}}}
             for t, (w, c, a) in sorted(losses.items())]
    arm = {"name": name, "seed": seed, "mixture": [["web", 0.75], ["chat", 0.25]],
           "kind": "graft" if graft else "setting"}
    if graft:
        arm["graft"] = {"frac": graft["frac"]}
    return {"type": "arm", "status": "done", "name": name, "arm": arm, "tokens": tokens, "steps": 4,
            "spec_hash": name, "curve": curve, "finished": 1.0 + seed, "graft": graft}


def test_the_report_on_synthetic_results(tmp_path, capsys):
    """Five arms x three seeds with chosen losses: the table, the paired differences after the
    graft and the share of the page gain recovered."""
    T = (1000, 2000, 3000, 4000)
    off = {0: 0.0, 1: 0.01, 2: -0.02}                        # seed offsets, shared by all arms

    def L(base_w, s, shift=0.0):
        return {t: (base_w - 0.1 * i + off[s] + shift, 4.0 - 0.1 * i + off[s] + shift, 3.9 - 0.1 * i + off[s])
                for i, t in enumerate(T)}
    gap = {0: -0.04, 1: -0.05, 2: -0.03}                     # v9_pages - ref at the end, by seed
    recs = []
    for s in (0, 1, 2):
        sfx = f"_s{s}" if s else ""
        ref = L(5.0, s)
        v9 = {t: tuple(x + gap[s] * (i + 1) / 4 if k < 2 else x for k, x in enumerate(v))
              for i, (t, v) in enumerate(ref.items())}
        recs += [_rec(f"ref{sfx}", s, ref), _rec(f"v9_pages{sfx}", s, v9)]
        for frac, keep in ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25)):
            gt = int(frac * 4000)
            g = {t: (v9[t] if t <= gt else tuple(r + keep * (x - r) if k < 2 else r
                                                  for k, (x, r) in enumerate(zip(v9[t], ref[t]))))
                 for t in T}
            g = {t: (ref[t] if t <= gt else g[t]) for t in T}
            info = {"frac": frac, "tokens": gt, "step": gt // 1000, "trunk": {"source": "trained"},
                    "check": {"loss_before": 5.0, "loss_after": 5.0, "difference": 0.0, "ok": True}}
            recs.append(_rec(f"graft_{round(100 * frac)}{sfx}", s, g, info))
    for r in recs:                                           # one --out sub-directory per arm
        d = tmp_path / "runs" / r["name"]
        d.mkdir(parents=True)
        (d / mb.RESULTS).write_text(json.dumps(r) + "\n")
    stale = dict(recs[0], status="partial")
    (tmp_path / "runs" / "ref" / mb.RESULTS).open("a").write(json.dumps(stale) + "\n")

    out_json = tmp_path / "rep.json"
    assert GR.main([str(tmp_path / "runs"), "--json", str(out_json), "--merge", str(tmp_path / "all.jsonl")]) == 0
    text = capsys.readouterr().out
    rep = json.loads(out_json.read_text())
    assert rep["budget"] == 4000 and rep["checkpoints"] == list(T) and rep["grafts"] == ["graft_25", "graft_50", "graft_75"]
    assert len(rep["inputs"]) == 15 and len((tmp_path / "all.jsonl").read_text().splitlines()) == 15
    # (a) the reference's mix at the end: 0.75 * web + 0.25 * chat, mean and sd over the seed offsets
    row = next(r for r in rep["table"] if r["arm"] == "ref" and r["tokens"] == 4000)
    mixes = [0.75 * (4.7 + off[s]) + 0.25 * (3.7 + off[s]) for s in (0, 1, 2)]
    assert row["mix"]["mean"] == pytest.approx(sum(mixes) / 3) and row["mix"]["n"] == 3
    assert row["mix"]["sd"] == pytest.approx(float(np.std(mixes, ddof=1)))
    assert row["assistant"]["mean"] == pytest.approx(3.6 + sum(off.values()) / 3)
    # (b) v9_pages - ref at the end: the gap per seed, its mean and sd; graft_50 only after 2000
    pv = next(r for r in rep["paired"] if r["comparison"] == "v9_pages - ref" and r["tokens"] == 4000)
    assert pv["web"]["mean"] == pytest.approx(-0.04) and pv["web"]["sd"] == pytest.approx(0.01)
    assert pv["mix"]["per_seed"] == pytest.approx({"0": -0.04, "1": -0.05, "2": -0.03})
    assert pv["assistant"]["mean"] == pytest.approx(0.0, abs=1e-12)
    g50 = [r for r in rep["paired"] if r["comparison"] == "graft_50 - ref"]
    assert [r["tokens"] for r in g50] == [3000, 4000]
    assert g50[-1]["web"]["mean"] == pytest.approx(0.5 * -0.04) and g50[-1]["web"]["n"] == 3
    gv = next(r for r in rep["paired"] if r["comparison"] == "graft_75 - v9_pages" and r["tokens"] == 4000)
    assert gv["mix"]["mean"] == pytest.approx(-0.75 * -0.04)
    assert [r["tokens"] for r in rep["paired"] if r["comparison"] == "graft_75 - ref"] == [4000]
    # (c) recovered share: 0.75, 0.5, 0.25 of the gap in every seed
    rec = {r["arm"]: r for r in rep["recovered"]}
    for name, share in (("graft_25", 0.75), ("graft_50", 0.5), ("graft_75", 0.25)):
        assert rec[name]["mix"]["mean"] == pytest.approx(share) and rec[name]["mix"]["sd"] == pytest.approx(0.0, abs=1e-9)
        assert rec[name]["web"]["ratio_of_means"] == pytest.approx(share)
        assert rec[name]["assistant"]["per_seed"] == {"0": None, "1": None, "2": None}   # no gap there
    assert rep["families"]["graft_50"]["graft"]["trunk"] == {"0": "trained", "1": "trained", "2": "trained"}
    assert "(c) Share of the page gain recovered at 4,000 tokens" in text and "| graft_50 - ref | 4,000 (last) |" in text
    # mixed budgets are refused unless one is chosen
    extra = tmp_path / "other"
    extra.mkdir()
    (extra / mb.RESULTS).write_text(json.dumps(_rec("ref", 0, L(5.0, 0), tokens=8000) | {"finished": 99.0}) + "\n")
    with pytest.raises(SystemExit, match="several budgets"):
        GR.main([str(tmp_path / "runs"), str(extra)])
    assert GR.main([str(tmp_path / "runs"), str(extra), "--tokens", "4000"]) == 0
    assert GR.family_of("graft_50_s12") == "graft_50" and GR.family_of("v9_pages") == "v9_pages"
