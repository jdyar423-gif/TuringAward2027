"""scripts/graft_run.py: a smoke-sized run directory grafted into a v9-like configuration (token
pages and output gains, no library), its refusals, and a trainer that continues the result."""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pytest

from bes.config import config_hash
from bes.train import checkpoint as CK
from bes.train import schedule as S
from bes.train import supervisor as SV
from bes.train.benchmark import Plan
from bes.train.guard import GuardState
from bes.train.ledger import Ledger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import graft_run as G  # noqa: E402

SMOKE = str(ROOT / "configs" / "smoke.py")
T0 = 1.0e6
STEP, COUNT, CURSOR = 456, 455, 77
NEW = {"state/params/gain/o", "state/params/page/key", "state/params/page/val", "state/params/page/pin",
       "state/params/page/pout"}

V9_LIKE = '''import dataclasses
from bes.config import load_config

_B = load_config({smoke!r})
CFG = dataclasses.replace(
    _B, model=dataclasses.replace(_B.model, gain_o=True, tok_neurons=2, tok_dim=16),
    data=dataclasses.replace(_B.data, use_library=False, library_frac_base=0.0, library_frac_chat=0.0),
    decode=dataclasses.replace(_B.decode, use_library=False),
    run=dataclasses.replace(_B.run, name="smoke_v9"))
'''


def _write_config(path: Path, text: str) -> str:
    path.write_text(text)
    return str(path)


def _manifest(config: str, overrides: list[str]) -> dict:
    base = SV.base_config({"config": config, "overrides": overrides})
    plan = Plan.of(base).with_table("take")
    man = {"format": SV.FORMAT, "config": config, "config_abs": config, "overrides": list(overrides),
           "budget": {"t0": T0, "deadline": T0 + 720.0, "budget_s": 720.0}, "created": T0,
           "resumes": [T0 + 1.0], "micro_split": 2, "post_plan": "sample=1 export=1 buffer=1",
           "fast": "", "jax_cache": {"dir": "/somewhere/jax_cache", "enabled": True},
           "benchmark": {"complete": True, "plan": plan.to_dict(), "plan_key": plan.key,
                         "plan_overrides": plan.overrides(), "horizon_tokens": 1.0e9,
                         "tokens_per_s": 1000.0, "loader_workers": 2,
                         "ladder": {"rung": 0, "groups": [], "overrides": [], "switches": None}}}
    man["config_hash"] = config_hash(base)
    man["effective"] = {"overrides": SV.run_overrides(man), "config_hash": config_hash(SV.effective_config(man))}
    return man


def _perturbed(tree, seed: int):
    """Every float leaf moved off its initialisation (a trained state)."""
    rng = np.random.default_rng(seed)

    def move(x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) or not jax.numpy.issubdtype(x.dtype, np.floating):
            return x
        return x + jax.numpy.asarray(rng.normal(0.0, 0.01, x.shape), x.dtype)

    state = tree["state"]
    params, opt = jax.tree.map(move, (state.params, state.opt_state))
    opt = opt._replace(count=jax.numpy.asarray(COUNT, opt.count.dtype))
    step = jax.numpy.asarray(STEP, state.step.dtype)
    return {"state": state._replace(step=step, params=params, opt_state=opt), "rng": tree["rng"]}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A stopped smoke run: manifest with a complete benchmark, stage ledger, markers, and one
    final checkpoint of a trained-looking state with ledger, guard and cursor in its meta."""
    d = tmp_path_factory.mktemp("graft")
    src = d / "src"
    man = _manifest(SMOKE, ["run.deadline_hours=0.2"])
    SV.write_manifest(src, man)
    ecfg = SV.effective_config(man)
    tree = _perturbed(G.fresh_tree(ecfg), seed=1)
    arrays, records, treedef = CK.to_host(tree)
    led = Ledger.start(T0, T0 + 5.0, phase="train")
    led.add_step(4096, steps=STEP)
    led = led.snapshot(T0 + 300.0)
    guard = GuardState(ema=7.25, count=300, rollbacks=1, lr_mult=0.5)
    meta = {"cursor": CURSOR, "ledger": led.to_dict(), "guard": guard.to_dict(), "regime": "R",
            "reason": "final", "deadline": T0 + 720.0, "train_end": T0 + 450.0}
    root = src / "train" / "ckpt"
    CK.write(root, arrays, records, treedef, seq=3, step=STEP - 50, config_hash=config_hash(ecfg),
             meta=dict(meta, reason="interval"))
    CK.write(root, arrays, records, treedef, seq=4, step=STEP, config_hash=config_hash(ecfg), meta=meta)
    (src / "train" / "run.json").write_text(json.dumps({"t0": T0, "deadline": T0 + 720.0,
                                                         "config_hash": config_hash(ecfg)}))
    (src / "train" / "train_log.jsonl").write_text('{"event": "step"}\n')
    (src / "train" / "loader_journal.jsonl").write_text('{"type": "block"}\n')
    (src / "stages.json").write_text(json.dumps({"stages": {"train": {"status": "interrupted"}}}))
    (src / "state").mkdir()
    (src / "state" / "tokenize.json").write_text("{}")
    (src / "benchmark").mkdir()
    (src / "benchmark" / "results.jsonl").write_text("{}\n")
    (src / "micro_split").write_text("2")
    cfg = _write_config(d / "v9_like.py", V9_LIKE.format(smoke=SMOKE))
    return {"dir": d, "src": src, "man": man, "ecfg": ecfg, "tree": tree, "meta": meta, "config": cfg}


def _graft(source, name: str, *extra: str) -> Path:
    dst = source["dir"] / name
    assert G.main([str(source["src"]), str(dst), "--config", source["config"], *extra]) == 0
    return dst


def _host(tree) -> dict:
    arrays, records, _ = CK.to_host(tree)
    return arrays


def test_graft_carries_the_trained_leaves_and_adds_the_new_ones_at_their_initialisation(source, capsys):
    dst = _graft(source, "dst", "--override", "run.seed=1", "--check", "--check-shape", "2x32")
    out = capsys.readouterr().out
    res = json.loads(out.strip().splitlines()[-1].split(" ", 1)[1])
    assert res["ok"] and res["check"]["ok"] and abs(res["check"]["difference"]) <= 1e-6
    assert set(res["new_params"]) == NEW and res["source_step"] == STEP and res["left_behind"] == []
    assert not list(source["dir"].glob("dst.graft-*"))            # the temporary directory was renamed
    man = SV.read_manifest(dst)
    ecfg = SV.effective_config(man)
    # hashes as run_24h.py checks them (Run.effective) and as the trainer checks the checkpoint
    assert man["effective"]["config_hash"] == config_hash(ecfg)
    assert man["effective"]["overrides"] == SV.run_overrides(man)
    assert man["config_hash"] == config_hash(SV.base_config(man))
    assert man["overrides"] == ["run.deadline_hours=0.2", "run.seed=1"] and ecfg.run.seed == 1
    assert ecfg.model.gain_o and ecfg.model.tok_neurons == 2 and not ecfg.data.use_library
    assert ecfg.optim.horizon_tokens == 1.0e9                     # the source's benchmark horizon
    assert "jax_cache" not in man and man["benchmark"] == source["man"]["benchmark"]
    assert man["created"] == source["man"]["created"]             # the Drive mirror's identity of the run
    assert man["graft"]["step"] == STEP and man["graft"]["from_config_hash"] == config_hash(source["ecfg"])
    for rel in ("stages.json", "state/tokenize.json", "benchmark/results.jsonl", "micro_split", "train/run.json"):
        assert (dst / rel).is_file(), rel
    for rel in ("train/train_log.jsonl", "train/loader_journal.jsonl", "train/ledger.json"):
        assert not (dst / rel).exists(), rel                      # a new journal and log
    assert json.loads((dst / "train" / "run.json").read_text())["config_hash"] == config_hash(ecfg)

    info = CK.latest(dst / "train" / "ckpt")
    assert (info.seq, info.step) == (0, STEP) and len(CK.list_checkpoints(dst / "train" / "ckpt")) == 1
    fresh = G.fresh_tree(ecfg)
    got, cman = CK.load(info, fresh, config_hash=config_hash(ecfg))   # the trainer's structure check
    assert cman["meta"]["cursor"] == 0 and cman["meta"]["reason"] == "graft"
    assert cman["meta"]["ledger"] == source["meta"]["ledger"] and cman["meta"]["guard"] == source["meta"]["guard"]
    assert cman["meta"]["regime"] == "R"
    a, s, f = _host(got), _host(source["tree"]), _host(fresh)
    carried = {p for p in s if not p.startswith("state/consts")}
    assert carried <= set(a)
    for p in carried:                                             # parameters, slots, count, rng
        assert np.array_equal(a[p], s[p]), p
    new = set(a) - carried
    assert {p for p in new if p.startswith("state/params/")} == NEW
    assert any(p.startswith("state/opt_state/") for p in new)           # their optimiser slots
    for p in new:                                                 # at the new configuration's init
        assert np.array_equal(a[p], f[p]), p
    assert int(got["state"].opt_state.count) == COUNT
    assert int(got["state"].step) == STEP
    assert float(np.abs(np.asarray(got["state"].params["page"]["val"])).max()) == 0.0
    assert float(np.abs(np.asarray(got["state"].params["gain"]["o"])).max()) == 0.0


def test_plan_override_changes_only_the_table_gradient(source):
    dst = _graft(source, "dst_sorted", "--plan-override", "model.table_grad=sorted")
    man = SV.read_manifest(dst)
    bench, old = man["benchmark"], source["man"]["benchmark"]
    assert bench["plan"]["table_grad"] == "sorted" and bench["grafted_plan"] is True
    assert bench["plan_key"].endswith("-sorted") and "model.table_grad=sorted" in bench["plan_overrides"]
    assert {k: v for k, v in bench["plan"].items() if k != "table_grad"} == \
        {k: v for k, v in old["plan"].items() if k != "table_grad"}
    ecfg = SV.effective_config(man)
    assert ecfg.model.table_grad == "sorted" and man["effective"]["config_hash"] == config_hash(ecfg)
    assert CK.read_manifest(CK.latest(dst / "train" / "ckpt").path)["config_hash"] == config_hash(ecfg)
    with pytest.raises(SystemExit, match="only model.table_grad"):
        G.main([str(source["src"]), str(source["dir"] / "dst_remat"), "--config", source["config"],
                "--plan-override", "model.remat=false"])
    assert not (source["dir"] / "dst_remat").exists()


def test_refusals_write_nothing(source, tmp_path):
    src, cfg = source["src"], source["config"]
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        G.main([str(src), str(existing), "--config", cfg])
    # no checkpoint
    empty = tmp_path / "empty"
    SV.write_manifest(empty, source["man"])
    with pytest.raises(SystemExit, match="no checkpoint"):
        G.main([str(empty), str(tmp_path / "d1"), "--config", cfg])
    # the newest checkpoint was written under another configuration than the manifest's
    other = tmp_path / "other"
    SV.write_manifest(other, source["man"])
    arrays, records, treedef = CK.to_host(source["tree"])
    CK.write(other / "train" / "ckpt", arrays, records, treedef, seq=1, step=STEP, config_hash="0000",
             meta=source["meta"])
    with pytest.raises(SystemExit, match="written under configuration 0000"):
        G.main([str(other), str(tmp_path / "d2"), "--config", cfg])
    # a leaf of both configurations with another shape
    bad = _write_config(tmp_path / "wider.py", V9_LIKE.format(smoke=SMOKE).replace(
        "gain_o=True,", "gain_o=True, ve_dim=64,"))
    with pytest.raises(SystemExit, match="differ in shape"):
        G.main([str(src), str(tmp_path / "d3"), "--config", bad])
    for n in ("d1", "d2", "d3"):
        assert not (tmp_path / n).exists() and not list(tmp_path.glob(f"{n}.graft-*"))


def test_trainer_continues_the_grafted_checkpoint(source):
    """The trainer of the grafted run directory restores the graft checkpoint under its own hash
    check (the structure of a fresh v9-like state), carries step, ledger and guard, starts the data
    stream at cursor 0, takes a finite step with the real model and checkpoints it."""
    M = pytest.importorskip("bes.model.model")
    roles = pytest.importorskip("bes.roles")
    from test_train_integration import synthetic_batch

    from bes.train.train import Trainer

    dst = _graft(source, "dst_train", "--override", "run.seed=1")
    man = SV.read_manifest(dst)
    ecfg = SV.effective_config(man)
    clock = S.FakeClock(T0 + 400.0)
    seen = []

    def data(index, stage):
        seen.append((index, stage.p))
        return synthetic_batch(index, stage)

    tr = Trainer(ecfg, loss_fn=M.loss_fn, init_fn=M.init_params, role_fn=roles.role_of,
                 consts_fn=SV.make_consts_fn(ecfg.run.switches), data=data, out_dir=dst / "train",
                 clock=clock, deadline=float(man["budget"]["deadline"]), max_steps=STEP + 1,
                 train_end=T0 + 450.0 + 3600.0)
    tr.setup()
    assert tr.step == STEP and tr.cursor == 0
    assert any(e["event"] == "restore" for e in tr.events)
    assert tr.guard == GuardState.from_dict(source["meta"]["guard"])
    assert tr.ledger.steps == STEP and tr.ledger.restarts == 1
    assert tr.ledger.train_seconds() == pytest.approx(Ledger.from_dict(source["meta"]["ledger"]).train_seconds())
    p = S.progress_of(tr.ledger, clock(), tr.train_end)
    assert 0.0 < p < 1.0
    out = tr.run()
    assert out["reason"] == "max_steps" and tr.step == STEP + 1
    assert seen[0][0] == 0 and seen[0][1] == pytest.approx(p, rel=1e-6)
    assert np.isfinite(tr.history[-1][3])
    assert int(tr.state.opt_state.count) == COUNT + 1
    # the new value table took its first update (its gradient is not zero once the step ran)
    assert float(np.abs(np.asarray(tr.state.params["page"]["val"])).max()) > 0.0
    last = CK.latest(dst / "train" / "ckpt")
    assert last.seq == 1 and last.step == STEP + 1
