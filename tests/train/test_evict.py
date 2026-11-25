"""Evicted token rows (bes/evict.py): the compact storage, the loader, the accounting and the
transformation of an export directory."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bes import budget, evict
from bes import export as E
from bes.config import load_config
from bes.model import model as M
from bes.train.checkpoint import flatten_with_paths

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "smoke.py"
OVERRIDES = ["model.n_loops=2", "model.gain_o=true", "model.tok_neurons=2", "model.tok_dim=8",
             "decode.particles=2", "decode.max_new_tokens=8"]
WEB = [("The river runs through the old town. Bridges cross it at three places, and the market "
        "sits on the east bank. ") * 6,
       "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen. " * 8]
CHAT = [[{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "Give me two tips for sleeping well."},
         {"role": "assistant", "content": "Keep a regular schedule. Avoid screens before bed."}]] * 3
PAIRS = [("Why does ice float on water?", "Ice is less dense than liquid water, so it floats.", "a"),
         ("What is the capital of Australia?", "The capital of Australia is Canberra.", "b"),
         ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet.", "c"),
         ("How many days are in a leap year?", "A leap year has 366 days.", "a")]


def _leaves(tree):
    paths, leaves, _ = flatten_with_paths(tree)
    return dict(zip(paths, [np.asarray(x) for x in jax.device_get(leaves)]))


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """``(export_dir, cfg)``: a real export of a smoke-sized model with token pages whose
    zero-initialised tables were made non-zero."""
    from bes.config import config_hash
    from bes.train import checkpoint as ck
    from bes.train.ledger import Ledger

    base = tmp_path_factory.mktemp("evict")
    cfg = load_config(str(CONFIG), OVERRIDES)
    rng = np.random.default_rng(3)
    params = jax.tree.map(lambda a: np.asarray(a) + rng.normal(0, 0.05, np.shape(a)).astype(np.float32),
                          M.init_params(cfg.model, jax.random.PRNGKey(3)))
    params["head"]["out"] = params["head"]["out"] * 300.0
    run = base / "run"
    tree = {"state": {"step": np.int32(7), "params": params, "opt_state": {},
                      "consts": M.make_consts(cfg.model)}, "rng": jax.random.PRNGKey(1)}
    ck.save(run / "train" / "ckpt", tree, step=7, config_hash=config_hash(cfg))
    led = Ledger.start(t0=1000.0, now=1000.0)
    led.switch("train", 1060.0); led.add_step(tokens=4096, steps=7); led.close(1901.0)
    led.save(run / "train" / "ledger.json")
    (run / "manifest.json").write_text(json.dumps({
        "format": 1, "config": str(CONFIG), "config_abs": str(CONFIG), "overrides": list(OVERRIDES),
        "budget": {"t0": 1000.0, "deadline": 4600.0, "budget_s": 3600.0},
        "code": {"commit": "test", "dirty": False}, "config_hash": config_hash(cfg), "resumes": []}))
    (run / "stages.json").write_text(json.dumps({"format": 1, "stages": {}}))
    res = E.export_run(run, library=None, heldout_rows=4, download_heldout=False,
                       heldout_records={"web": WEB, "chat": CHAT})
    return Path(res["export_dir"]), cfg


def _keep(cfg, n=300, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 1000, cfg.model.vocab_size)
    return evict.keep_mask(counts, n, always=(cfg.model.vocab_size - 1, cfg.model.vocab_size - 2)), counts


def test_keep_mask_takes_the_most_frequent_tokens_and_the_ones_always_kept():
    counts = np.array([5, 9, 9, 0, 1, 7])
    assert evict.ranks(counts).tolist() == [3, 0, 1, 5, 4, 2]            # ties by id
    assert evict.keep_mask(counts, 3).tolist() == [False, True, True, False, False, True]
    assert evict.keep_mask(counts, 1, always=(3,)).tolist() == [False, True, False, True, False, False]


def test_compact_and_expand_are_inverse_up_to_the_evicted_rows():
    rng = np.random.default_rng(0)
    v = 40
    arrays = {"gain/o": rng.normal(size=(v, 1, 2, 6)).astype(np.float16),
              "page/key": rng.integers(1, 60000, (v, 2, 3, 4)).astype(np.uint16),     # bfloat16 bit patterns
              "page/val": rng.normal(size=(v, 2, 3, 4)).astype(np.float16),
              "emb/tok": rng.normal(size=(v, 8)).astype(np.float16)}
    keep = rng.random(v) < 0.4
    c = evict.compact(arrays, keep)
    k = int(keep.sum())
    assert c[evict.MAP_LEAF].dtype == np.uint16 and c[evict.MAP_LEAF].shape == (v,)
    assert all(c[p].shape[0] == k for p in evict.ROW_TABLES) and c["emb/tok"] is arrays["emb/tok"]
    assert np.array_equal(evict.kept_of(c), keep)
    back = evict.expand(c)
    assert evict.MAP_LEAF not in back
    for p in evict.ROW_TABLES:
        assert back[p].dtype == arrays[p].dtype and back[p].shape == arrays[p].shape
        np.testing.assert_array_equal(back[p][keep], arrays[p][keep])
        assert not back[p][~keep].any()
    with pytest.raises(ValueError):
        evict.compact(c, keep)                                           # evicted twice
    bad = dict(c); bad[evict.MAP_LEAF] = c[evict.MAP_LEAF][::-1].copy()
    with pytest.raises(ValueError):
        evict.expand(bad)


def test_an_evicted_export_loads_to_the_masked_source_and_counts_what_it_stores(source, tmp_path):
    src, cfg = source
    keep, _ = _keep(cfg)
    batches, _ = E.heldout_batches(cfg, __import__("bes.tokenizer.tok", fromlist=["load"]).load(),
                                   rows=4, download=False, records={"web": WEB, "chat": CHAT},
                                   domains=("web", "chat"))
    logs = []
    rec = evict.evict_export(src, tmp_path / "shipped", keep, rule="the 300 most frequent of a test count",
                             heldout=batches, log=logs.append)
    dst = tmp_path / "shipped"
    assert E.verify_export(dst) == []
    v, kept = cfg.model.vocab_size, int(keep.sum())
    # the weights: exactly the source's with the evicted rows at zero
    got = _leaves(E.load_weights(dst, cfg.model))
    want = _leaves(evict.mask_params(E.load_weights(src, cfg.model), keep))
    full = _leaves(E.load_weights(src, cfg.model))
    assert set(got) == set(want)
    for p in want:
        np.testing.assert_array_equal(got[p], want[p], err_msg=p)
    for p in evict.ROW_TABLES:
        assert full[p][keep].any() and not got[p][~keep].any() and full[p][~keep].any()
    # the stored file holds the kept rows and the map only
    stored = E.read_npz(dst / E.WEIGHTS16)
    assert stored[evict.MAP_LEAF].shape == (v,) and all(stored[p].shape[0] == kept for p in evict.ROW_TABLES)
    # the ledger: the source's minus the evicted rows plus the map; one more number read per token
    a0 = json.loads((src / E.EXPORT_JSON).read_text())["accounting"]
    width = sum(int(np.prod(full[p].shape[1:])) for p in evict.ROW_TABLES)
    assert rec["accounting"]["total"] == a0["total"] - (v - kept) * width + v
    assert rec["accounting"]["active_step"] == a0["active_step"] + 1
    assert rec["eviction"]["numbers_freed"] == (v - kept) * width and rec["eviction"]["kept"] == kept
    assert rec["weights"]["numbers"] == rec["accounting"]["total"]
    assert sum(a.size for a in stored.values()) == rec["accounting"]["total"]
    # the held-out record is the loss of the two models
    consts = M.make_consts(E.check_model_cfg(cfg.model))
    before = E.heldout_loss(E.load_weights(src, cfg.model), consts, batches, cfg.model)["loss"]
    after = E.heldout_loss(E.load_weights(dst, cfg.model), consts, batches, cfg.model)["loss"]
    assert rec["eviction"]["heldout"]["before"] == pytest.approx(before)
    assert rec["eviction"]["heldout"]["after"] == pytest.approx(after) and before != after
    card = (dst / E.CARD).read_text()
    assert "## Evicted token rows" in card and f"{v - kept:,} of the {v:,}" in card
    # the export loads for inference like any other
    m = E.load_export(dst)
    assert m.library is None and set(_leaves(m.params)) == set(want)
    with pytest.raises(ValueError):
        evict.evict_export(dst, tmp_path / "again", keep)


def test_the_library_is_attached_and_counted(source, tmp_path):
    from bes.library.answers import AnswerCfg, AnswerLibrary

    src, cfg = source
    keep, _ = _keep(cfg, n=200, seed=1)
    lib_dir = tmp_path / "answers"
    AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=12, read_cap=5000)).save(str(lib_dir))
    rec = evict.evict_export(src, tmp_path / "shipped", keep, library=lib_dir)
    dst = tmp_path / "shipped"
    assert E.verify_export(dst) == [] and rec["library"]["mode"] == "copy" and rec["library"]["kind"] == "answers"
    m = E.load_export(dst)
    assert m.library is not None and m.library.n_entries == len(PAIRS)
    acc = m.library.accounting()
    shapes = evict._shape_tree(cfg.model, int(keep.sum()))
    core = budget.count(shapes, cfg.model)
    assert rec["accounting"]["total"] == core.total + cfg.model.vocab_size + int(acc["total_numbers"])
    assert rec["accounting"]["active_step"] == core.active_step + 1 + 5000       # the reply's read cap
    assert "## Evicted token rows" in (dst / E.CARD).read_text()


def test_the_library_switch_is_written_into_the_decoding_choice(source, tmp_path):
    """A source exported without a library may record ``use_library=false``; the shipped model
    says what it does without an override."""
    from bes.library.answers import AnswerCfg, AnswerLibrary

    src, cfg = source
    keep, _ = _keep(cfg, n=200, seed=1)
    lib_dir = tmp_path / "answers"
    AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=12, read_cap=5000)).save(str(lib_dir))
    before = (json.loads((src / E.EXPORT_JSON).read_text()).get("decode_choice") or {}).get("overrides") or []
    kept = evict.evict_export(src, tmp_path / "kept", keep, library=lib_dir)
    assert ((kept.get("decode_choice") or {}).get("overrides") or []) == before          # None: untouched
    for want in (True, False):
        dst = tmp_path / f"ship{int(want)}"
        rec = evict.evict_export(src, dst, keep, library=lib_dir, use_library=want)
        flag = f"extra.use_library={str(want).lower()}"
        over = rec["decode_choice"]["overrides"]
        assert over.count(flag) == 1 and sum(o.startswith("extra.use_library=") for o in over) == 1
        assert rec["decode_choice"]["settings"]["use_library"] is want
        assert json.loads((dst / "config.json").read_text())["decode_choice"] == rec["decode_choice"]
        assert [o for o in over if o != flag] == [o for o in before if not o.startswith("extra.use_library=")] \
            or not before
        assert E.verify_export(dst) == []
        m = E.load_export(dst)
        assert m.settings()["use_library"] is want
        assert ("on by default" if want else "off by default") in (dst / E.CARD).read_text()
    with pytest.raises(E.ExportError):
        evict.evict_export(src, tmp_path / "nolib", keep, use_library=True)


def test_a_damaged_row_map_is_refused(source, tmp_path):
    src, cfg = source
    keep, _ = _keep(cfg)
    evict.evict_export(src, tmp_path / "shipped", keep)
    dst = tmp_path / "shipped"
    arrays = E.read_npz(dst / E.WEIGHTS16)
    arrays[evict.MAP_LEAF] = arrays[evict.MAP_LEAF].copy()
    i = int(np.flatnonzero(arrays[evict.MAP_LEAF] != evict.NO_ROW)[0])
    arrays[evict.MAP_LEAF][i] = evict.NO_ROW                               # one kept row lost
    E.write_npz(dst / E.WEIGHTS16, arrays)
    with pytest.raises(E.ExportError):
        E.load_weights(dst, cfg.model)


def test_an_overlay_is_reserved_in_the_ledger_and_loads_with_the_export(source, tmp_path):
    from bes.library.answers import AnswerCfg, AnswerLibrary
    from bes.library.overlay import LifelongAnswers

    src, cfg = source
    keep, _ = _keep(cfg, n=200, seed=1)
    lib_dir = tmp_path / "answers"
    AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=12, read_cap=5000)).save(str(lib_dir))
    plain = evict.evict_export(src, tmp_path / "plain", keep, library=lib_dir)
    rec = evict.evict_export(src, tmp_path / "shipped", keep, library=lib_dir, overlay_numbers=300)
    assert rec["accounting"]["total"] == plain["accounting"]["total"] + 300
    assert rec["accounting"]["active_step"] == plain["accounting"]["active_step"]      # inside the read cap
    assert rec["library"]["overlay_capacity_numbers"] == 300
    with pytest.raises(E.ExportError):
        evict.evict_export(src, tmp_path / "nolib", keep, overlay_numbers=300)
    mem = tmp_path / "memory.jsonl"
    m = E.load_export(tmp_path / "shipped", memory=mem)
    assert isinstance(m.library, LifelongAnswers) and m.library.overlay.capacity_numbers == 300
    assert m.library.new_read().cap == E.load_export(tmp_path / "plain").library.new_read().cap - 300
    resp = m.responder()
    r = resp.teach("What is my dog's name?", "Your dog's name is Pemberly.")
    assert r["numbers"] > 0 and mem.is_file()
    again = E.load_export(tmp_path / "shipped", memory=mem)                # a restart: still there
    assert again.library.retrieve("Can you remind me what my dog is called?") is not None
    assert again.responder().forget("What is my dog's name?")
    assert E.load_export(tmp_path / "shipped", memory=mem).library.retrieve("what is my dog called") is None
    with pytest.raises(RuntimeError):
        E.load_export(tmp_path / "plain").responder().teach("a question here", "an answer")
