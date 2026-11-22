"""Export (bes/export.py, scripts/export.py), the chat (scripts/chat.py), the Colab notebooks,
requirements-colab.txt and the user-facing documents.

A tiny run directory is written here: a smoke-sized checkpoint in the trainer's format, a
manifest, a stage ledger, a training ledger and a tiny library built with
``bes.library.retrieve.Library.build``. The 16-bit rounding is checked against an independent
integer implementation of round-to-nearest-even; the held-out loss against one direct call of
``loss_fn``; the accounting against ``bes.budget.count``.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import shutil
import signal
from pathlib import Path

import numpy as np
import pytest

from bes import export as E
from bes.config import load_config

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "smoke.py"
OVERRIDES = ["model.n_loops=2", "decode.particles=2", "decode.max_new_tokens=8"]
NOTEBOOKS = [REPO / "notebooks" / "colab_v6e_24h.ipynb", REPO / "notebooks" / "colab_v5e_smoke.ipynb"]

WEB = [("The river runs through the old town. Bridges cross it at three places, and the market "
        "sits on the east bank. ") * 6,
       "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen. " * 8,
       "A recipe: mix flour and water, knead for ten minutes, rest the dough, then bake it. " * 7]
CHAT = [[{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "Give me two tips for sleeping well."},
         {"role": "assistant", "content": "Keep a regular schedule. Avoid screens before bed."},
         {"role": "user", "content": "Thanks!"},
         {"role": "assistant", "content": "You are welcome."}]] * 3


# ============================================================================ fixtures
def _bf16_bits_reference(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns, round to nearest even, by integer arithmetic."""
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _cfg():
    return load_config(str(CONFIG), OVERRIDES)


def _params(cfg, seed: int = 3):
    import jax

    from bes.model import model as M

    p = M.init_params(cfg.model, jax.random.PRNGKey(seed))
    p = jax.tree.map(np.asarray, p)
    p["head"]["out"] = p["head"]["out"] * 300.0          # informative logits, not uniform
    p["critic"]["w1"] = p["critic"]["w1"] * 1e7          # beyond the float16 range (unused by the loss)
    return p


def _tiny_library(path: Path) -> Path:
    from bes.library.index import IndexCfg
    from bes.library.pack import PackCfg
    from bes.library.retrieve import GateCfg, Library, RetrieveCfg

    docs = [("Napoleon", "Napoleon was a French military leader and emperor."),
            ("Paris", "Paris is the capital and largest city of France."),
            ("Alan Turing", "Alan Turing was an English mathematician and computer scientist."),
            ("Hamlet", "Hamlet is a tragedy written by William Shakespeare.")]
    path.mkdir(parents=True, exist_ok=True)
    Library.build(docs, out_dir=str(path), index_cfg=IndexCfg(budget_numbers=10**6),
                  pack_cfg=PackCfg(block_raw_bytes=256, titles_per_block=4),
                  cfg=RetrieveCfg(max_tokens=160, gate=GateCfg(min_score=0.5)),
                  count_tokens=lambda s: len(s.split()))
    (path / "ATTRIBUTION.md").write_text("# Attribution\n\nTest passages written for this test.\n")
    return path


def _write_run(root: Path, params, cfg) -> Path:
    import jax

    from bes.config import config_hash
    from bes.model import model as M
    from bes.train import checkpoint as ck
    from bes.train.ledger import Ledger

    tree = {"state": {"step": np.int32(7), "params": params,
                      "opt_state": {"m": jax.tree.map(np.zeros_like, params["head"])},
                      "consts": M.make_consts(cfg.model)}, "rng": jax.random.PRNGKey(1)}
    ck.save(root / "train" / "ckpt", tree, step=7, config_hash=config_hash(cfg))
    led = Ledger.start(t0=1000.0, now=1000.0)
    led.switch("train", 1060.0)
    led.add_step(tokens=4096, steps=7)
    led.switch("ckpt", 1900.0)
    led.close(1901.0)
    led.save(root / "train" / "ledger.json")
    man = {"format": 1, "config": str(CONFIG), "config_abs": str(CONFIG), "overrides": OVERRIDES,
           "budget": {"t0": 1000.0, "deadline": 4600.0, "budget_s": 3600.0},
           "code": {"commit": "test", "dirty": False}, "config_hash": config_hash(cfg), "resumes": []}
    (root / "manifest.json").write_text(json.dumps(man))
    stages = {"format": 1, "stages": {
        "preflight": {"status": "finished", "planned_s": 15.0, "slice_s": 15.0, "used_s": 1.0, "attempts": 1},
        "train": {"status": "finished", "planned_s": 900.0, "slice_s": 900.0, "used_s": 901.0, "attempts": 1},
        "judge": {"status": "skipped", "planned_s": 30.0, "slice_s": 0.0, "used_s": 0.0, "attempts": 0,
                  "reason": "no opponent replies"}}}
    (root / "stages.json").write_text(json.dumps(stages))
    return root


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """``(run_dir, library_dir, params, cfg)``."""
    base = tmp_path_factory.mktemp("export")
    cfg = _cfg()
    params = _params(cfg)
    lib = _tiny_library(base / "lib")
    run = _write_run(base / "run", params, cfg)
    return run, lib, params, cfg


@pytest.fixture(scope="module")
def exported(tiny):
    run, lib, params, cfg = tiny
    logs = []
    res = E.export_run(run, library=lib, library_mode="copy", heldout_rows=4, download_heldout=False,
                       heldout_records={"web": WEB, "chat": CHAT}, log=logs.append)
    return Path(res["export_dir"]), res, logs


# ======================================================================= 16-bit storage
def test_choose_16bit_takes_the_format_with_the_smaller_measured_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096).astype(np.float32)
    for scale, want in ((1.0, "float16"), (1e6, "bfloat16"), (1e-7, "bfloat16"), (1e-3, "float16")):
        v = (x * scale).astype(np.float32)
        stored, ch = E.choose_16bit(v, "leaf")
        assert stored.dtype.itemsize == 2 and stored.dtype.name in ("float16", "uint16")
        # independent references for both formats
        with np.errstate(over="ignore"):
            h = v.astype(np.float16)
        e16 = (np.linalg.norm(h.astype(np.float64) - v) / np.linalg.norm(v.astype(np.float64))
               if np.isfinite(h).all() else math.inf)
        bits = _bf16_bits_reference(v)
        bf_vals = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        ebf = np.linalg.norm(bf_vals - v) / np.linalg.norm(v.astype(np.float64))
        assert ch.rel_err["bfloat16"] == pytest.approx(ebf, rel=1e-12)
        assert ch.rel_err["float16"] == pytest.approx(e16, rel=1e-12) or (math.isinf(e16) and math.isinf(ch.rel_err["float16"]))
        assert ch.dtype == want == ("float16" if e16 <= ebf else "bfloat16"), (scale, e16, ebf)
        if ch.dtype == "bfloat16":
            np.testing.assert_array_equal(stored, bits)          # exact RNE bit patterns
        back = E.stored_to_float32(stored, ch.dtype)
        assert back.dtype == np.float32
        assert ch.max_abs_err == pytest.approx(float(np.max(np.abs(back.astype(np.float64) - v))))


def test_choose_16bit_edge_cases():
    stored, ch = E.choose_16bit(np.zeros((3, 4), np.float32), "zeros")
    assert ch.dtype == "float16" and ch.rel_err == {"float16": 0.0, "bfloat16": 0.0}
    with pytest.raises(E.ExportError):
        E.choose_16bit(np.array([1.0, np.nan], np.float32), "nan")
    with pytest.raises(E.ExportError):
        E.choose_16bit(np.array([1, 2], np.int32), "int")
    # 1e5 overflows float16, so the leaf is bfloat16; the other values are exact ties of
    # bfloat16 and round to the even neighbour (1.0 -> 0x3F80, 1.0234375 -> 0x3F82, ...)
    x = np.array([1e5, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.0 - 2 ** -7], np.float32)
    s, c = E.choose_16bit(x, "ties")
    assert c.dtype == "bfloat16" and math.isinf(c.rel_err["float16"])
    np.testing.assert_array_equal(s, _bf16_bits_reference(x))
    assert s[1:].tolist() == [0x3F80, 0x3F82, 0xC000]


def test_quantize_round_trip_restores_the_stored_values_as_float32(tmp_path):
    import jax

    from bes.roles import leaves_with_paths

    cfg = _cfg()
    params = _params(cfg)
    arrays, choices = E.quantize(params)
    E.write_npz(tmp_path / "w.npz", arrays)
    back = E.read_npz(tmp_path / "w.npz")
    assert set(back) == set(arrays)
    tree = E.dequantize(back, choices, cfg.model)
    by = {c.path: c for c in choices}
    assert by["critic/w1"].dtype == "bfloat16" and by["emb/tok"].dtype == "float16"
    for path, leaf in leaves_with_paths(tree):
        orig = dict(leaves_with_paths(params))[path]
        assert leaf.dtype == np.float32 and leaf.shape == orig.shape
        if by[path].dtype == "float16":
            ref = np.asarray(orig, np.float32).astype(np.float16).astype(np.float32)
        else:
            ref = (_bf16_bits_reference(orig).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(np.asarray(leaf), ref)
    assert jax.tree.structure(tree) == jax.tree.structure(params)
    with pytest.raises(E.ExportError):
        E.dequantize({k: v for k, v in back.items() if k != "emb/tok"}, choices, cfg.model)


# ======================================================================= held-out loss
def test_heldout_rows_and_loss_match_one_direct_loss_call(tiny):
    import jax

    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    _, _, params, cfg = tiny
    tok = load_tokenizer()
    batches, info = E.heldout_batches(cfg, tok, rows=3, download=False,
                                      records={"web": WEB, "chat": CHAT}, chunk=2)
    assert [b.domain for b in batches] == ["web", "chat", "proxy"]
    assert all(b.rows % 2 == 0 for b in batches)                       # padded to the chunk
    assert info["domains"]["proxy"]["held_out"] is True
    web = batches[0]
    assert web.probe is not None and web.probe[0] == M.bos_id(cfg.model)
    consts = M.make_consts(cfg.model)
    jp = jax.tree.map(np.asarray, params)
    got = E.heldout_loss(jp, consts, batches, cfg.model, chunk=2)
    ce = w = 0.0
    mcfg = E.check_model_cfg(cfg.model)
    for b in batches:                                                   # one call per domain
        m = jax.device_get(M.loss_fn(jp, consts, b.arrays, mcfg)[1])
        ce += float(m["ce_sum"])
        w += float(m["weight"])
    assert got["loss"] == pytest.approx(ce / w, rel=1e-5)
    assert got["weight"] == pytest.approx(w)


# ======================================================================= export
def test_export_writes_a_self_contained_verified_directory(exported, tiny):
    d, res, logs = exported
    run, lib, params, cfg = tiny
    for name in ("weights16.npz", "weights.json", "config.json", "config.py", "tokenizer/tokenizer.json",
                 "tokenizer/meta.json", "library/library.json", "LIBRARY_ATTRIBUTION.md", "export.json",
                 "MODEL_CARD.md", "SHA256SUMS", "reports/manifest.json", "reports/stages.json",
                 "reports/train_ledger.json"):
        assert (d / name).is_file(), name
    assert not (d / "weights32.npz").exists()                     # the check passed
    assert not list(d.rglob("__pycache__"))
    assert E.verify_export(d) == []
    sums = dict(reversed(line.split("  ", 1)) for line in (d / "SHA256SUMS").read_text().splitlines())
    assert all(E.sha256_file(d / rel) == h for rel, h in sums.items())
    assert len(sums) == sum(1 for p in d.rglob("*") if p.is_file()) - 1
    rec = json.loads((d / "export.json").read_text())
    chk = rec["check"]
    assert chk["ok"] is True and chk["abs_delta"] < E.TOLERANCE_NATS and res["check_ok"] is True
    assert chk["abs_delta"] == pytest.approx(abs(chk["loss16"] - chk["loss32"]))
    assert chk["abs_delta"] > 0                                   # something was actually measured
    assert set(chk["by_domain"]) == {"web", "chat", "proxy"}
    assert any("held-out check passed" in m for m in logs)
    # configuration and switches rebuild exactly
    from bes.config import config_hash

    assert config_hash(E.load_export_config(d / "config.py")) == config_hash(cfg) == rec["config"]["config_hash"]
    assert rec["consts"]["rebuilt_exactly"] is True
    # accounting = bes.budget with the real library
    from bes import budget
    from bes.library.retrieve import Library

    L = Library.load(str(lib))
    acc, bound = L.accounting(), L.read_bound()
    ref = budget.count(params, cfg.model, budget.Assets(
        library_numbers=acc["total_numbers"] - acc["index_numbers"], index_numbers=acc["index_numbers"],
        library_read=bound["total"] - bound["index"], index_read=bound["index"]))
    assert rec["accounting"]["total"] == ref.total and rec["accounting"]["active_step"] == ref.active_step
    assert rec["weights"]["numbers"] == sum(int(np.size(v)) for v in
                                            __import__("jax").tree.leaves(params))
    # read-set audit on the held-out probes, both conventions
    rs = rec["readset"]
    assert rs["probes"] and all(p["strict_max"] >= p["step_max"] >= p["dense"] > 0 for p in rs["probes"])
    assert rs["library"]["queries"] > 0 and rs["library"]["read_bound"] == bound["total"]
    # library copied with its digest
    assert rec["library"]["mode"] == "copy" and rec["library"]["digest"] == E.library_digest(lib)["digest"]


def test_model_card_states_the_measured_numbers(exported):
    d, _, _ = exported
    card = (d / "MODEL_CARD.md").read_text()
    rec = json.loads((d / "export.json").read_text())
    for section in ("## Stored weights", "### Held-out check", "## Accounting", "## Read-set audit",
                    "## Training run", "## Post-training and evaluation", "## Disclosures",
                    "## Data, licences and attribution", "## Files"):
        assert section in card, section
    chk = rec["check"]
    assert f"{chk['delta']:+.6f}" in card and "**passed**" in card and "WARNING" not in card
    assert f"{rec['accounting']['total']:,}" in card
    assert "**Library**" in card and "**Inference-time compute**" in card and "K = 2" in card
    assert "**External models**" in card and "scoring only" in card
    assert "4,096 tokens" in card and "7 optimiser steps" in card          # the training ledger
    assert "| judge | skipped |" in card                                     # the stage table
    assert "judged comparison: no record" in card                           # nothing invented
    assert not re.search(r"\b(beats|better than|wins against)\b", card)


def test_load_export_restores_float32_and_replies_deterministically(exported, tiny):
    import jax

    d, _, _ = exported
    _, _, params, cfg = tiny
    ex = E.load_export(d)
    assert all(np.asarray(v).dtype == np.float32 for v in jax.tree.leaves(ex.params))
    arrays = E.read_npz(d / "weights16.npz")
    choices = [E.LeafChoice.from_dict(r) for r in json.loads((d / "weights.json").read_text())["leaves"]]
    ref = E.dequantize(arrays, choices, cfg.model)
    for a, b in zip(jax.tree.leaves(ex.params), jax.tree.leaves(ref)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert ex.library is not None and ex.settings()["particles"] == 2
    gen = ex.generator()
    msgs = [{"role": "user", "content": "Who was Napoleon?"}]
    s = ex.settings(["max_tokens=6"])
    out = gen(msgs, s, 5)
    assert isinstance(out["text"], str) and out["completion_tokens"] <= 6
    assert out == gen(msgs, s, 5)
    assert out["extra"]["library"] == "Napoleon"                 # the exported library is used
    assert gen.key == E.load_export(d).generator().key and gen.key.startswith("ours-export-step7-")
    g2 = E.export_generator(export=str(d), overrides="decode.particles=1", library="none")
    assert g2.key != gen.key and g2(msgs, {"max_tokens": 4}, 1)["extra"]["particles"] == 1
    with pytest.raises(ValueError):
        E.export_generator(export=str(d), overrides="model.n_loops=1")


def test_tampering_is_detected(exported, tmp_path):
    d, _, _ = exported
    c = tmp_path / "copy"
    shutil.copytree(d, c)
    assert E.verify_export(c) == []
    w = c / "weights16.npz"
    data = bytearray(w.read_bytes())
    data[len(data) // 2] ^= 0xFF
    w.write_bytes(bytes(data))
    (c / "extra.txt").write_text("not listed")
    problems = E.verify_export(c)
    assert any("weights16.npz: SHA-256 differs" in p for p in problems)
    assert any("extra.txt: not listed" in p for p in problems)
    with pytest.raises(E.ExportError):
        E.load_export(c)


def test_failed_check_keeps_both_weight_files_and_says_so(tiny, tmp_path):
    import jax

    run, lib, params, cfg = tiny
    res = E.export_run(run, final_dir=tmp_path / "fail", library=lib, library_mode="none", tol=0.0,
                       heldout_rows=2, download_heldout=False, heldout_records={"web": WEB, "chat": CHAT},
                       log=lambda m: None)
    d = Path(res["export_dir"])
    assert res["check_ok"] is False and res["float32_kept"] is True
    assert (d / "weights32.npz").is_file() and (d / "weights16.npz").is_file()
    card = (d / "MODEL_CARD.md").read_text()
    assert card.splitlines()[4].startswith("> **WARNING: the 16-bit weights did NOT pass") and "**FAILED**" in card
    assert "No library is shipped" in card
    ex = E.load_export(d, weights="32", library=None)
    for a, b in zip(jax.tree.leaves(ex.params), jax.tree.leaves(params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b, np.float32))
    assert E.load_export(d, library=None).notes                       # loading 16-bit warns


def test_no_heldout_rows_means_not_verified_and_both_files(tiny, tmp_path):
    run, lib, _, _ = tiny
    res = E.export_run(run, final_dir=tmp_path / "none", library=lib, library_mode="pointer",
                       heldout_domains=(), log=lambda m: None)
    d = Path(res["export_dir"])
    rec = json.loads((d / "export.json").read_text())
    assert rec["check"]["ok"] is None and (d / "weights32.npz").exists()
    assert "could not be run" in (d / "MODEL_CARD.md").read_text()
    ptr = json.loads((d / "library_pointer.json").read_text())
    assert ptr["digest"] == E.library_digest(lib)["digest"]
    assert E.verify_export(d) == []


def test_pointer_mode_detects_a_changed_library(tiny, tmp_path):
    run, lib, _, _ = tiny
    lib2 = tmp_path / "lib2"
    shutil.copytree(lib, lib2)
    res = E.export_run(run, final_dir=tmp_path / "ptr", library=lib2, library_mode="pointer",
                       heldout_rows=2, download_heldout=False, heldout_records={"web": WEB, "chat": CHAT},
                       log=lambda m: None)
    d = Path(res["export_dir"])
    assert E.verify_export(d) == []
    (lib2 / "ATTRIBUTION.md").write_text("changed")
    assert any("changed since the export" in p for p in E.verify_export(d))
    with pytest.raises(E.ExportError):
        E.load_export(d)


def test_export_replaces_an_earlier_export_atomically(tiny, tmp_path):
    run, lib, _, _ = tiny
    final = tmp_path / "final"
    (final / "ckpt").mkdir(parents=True)
    (final / "EXPORT_NOTE.txt").write_text("fallback export")
    E.export_run(run, final_dir=final, library=lib, library_mode="none", heldout_rows=2,
                 download_heldout=False, heldout_records={"web": WEB, "chat": CHAT}, log=lambda m: None)
    assert not (final / "EXPORT_NOTE.txt").exists() and E.verify_export(final) == []
    assert not [p for p in tmp_path.iterdir() if p.name.startswith((".tmp-", ".old-"))]


# ======================================================================= the shipped library
QA_ROWS = [("Who wrote Hamlet?", "William Shakespeare"), ("What is the capital of France?", "Paris"),
           ("Who was Alan Turing?", "an English mathematician"), ("Who was Napoleon?", "a French emperor")]


def _qa_library(path: Path, sources=("nq_open_train", "trivia_qa_train"), permissive: bool = False) -> Path:
    """A tiny library with a question-answer layer whose entries come from registry sources, a
    gate, and the attribution files of a real build."""
    from bes.library import gate as G
    from bes.library import qa as Q
    from bes.library.index import IndexCfg
    from bes.library.pack import PackCfg
    from bes.library.retrieve import GateCfg, Library, RetrieveCfg

    docs = [("Napoleon", "Napoleon was a French military leader and emperor."),
            ("Paris", "Paris is the capital and largest city of France."),
            ("DNA", "DNA is the molecule that carries the genetic information of living things."),
            ("Hamlet", "Hamlet is a tragedy written by William Shakespeare.")]
    pairs = [Q.QAPair(q, a, sources[i % len(sources)], i) for i, (q, a) in enumerate(QA_ROWS)]
    store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0), PackCfg(block_raw_bytes=128),
                            meta={"permissive_only": permissive})
    gate = G.GateModel.stored({"wiki": np.zeros(len(G.WIKI_FEATURES) + 1),
                               "qa": np.zeros(len(G.QA_FEATURES) + 1)}, 0.5)
    path.mkdir(parents=True, exist_ok=True)
    Library.build(docs, out_dir=str(path), index_cfg=IndexCfg(budget_numbers=10**6),
                  pack_cfg=PackCfg(block_raw_bytes=256, titles_per_block=4),
                  cfg=RetrieveCfg(max_tokens=160, gate=GateCfg(min_score=0.5)),
                  count_tokens=lambda s: len(s.split()), meta={"build": {"permissive_only": permissive}},
                  qa=store, gate=gate)
    (path / "ATTRIBUTION.md").write_text("# Attribution\n\nTest passages written for this test.\n")
    (path / "ATTRIBUTION_QA.tsv").write_text(
        "# question-answer layer: sources\n" + "".join(f"# {s}\n" for s in sorted(set(sources)))
        + "entry\tsource\trow\n" + "".join(f"{e}\t{p.source}\t{p.row}\n" for e, p in enumerate(pairs)))
    return path


def _files_the_loader_reads(lib: Path, monkeypatch) -> tuple[set, object]:
    """Independent reference of 'what the library is': every file under ``lib`` that
    ``Library.load``, ``accounting``, ``read_bound`` and a few retrievals open."""
    import builtins

    from bes.library.retrieve import Library

    seen: set[str] = set()
    root = lib.resolve()

    def note(f):
        if isinstance(f, (str, os.PathLike)):
            p = Path(os.fspath(f)).resolve()
            if root in p.parents:
                seen.add(str(p.relative_to(root)))

    real_open, real_load = builtins.open, np.load
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", lambda f, *a, **k: (note(f), real_open(f, *a, **k))[1])
        m.setattr(np, "load", lambda f, *a, **k: (note(f), real_load(f, *a, **k))[1])
        src = Library.load(str(lib))
        src.accounting(), src.read_bound()
        for q, _ in QA_ROWS:
            src.retrieve(q)
    return seen, src


def test_export_ships_only_the_counted_library_tables(tiny, tmp_path, monkeypatch):
    """tools-2: a library directory as a build leaves it (64-bit provenance arrays, a build
    report, the library task's settings record) exports as exactly the files the library's own
    loader reads, every one counted by its accounting, plus the attribution text. Before, the
    whole directory was copied: 3,080,684 uncounted 16-bit numbers of provenance for the real
    library, 101.33M shipped while the card said 'within both'."""
    from bes.library.index import TruncatedIndex
    from bes.library.qa import QAStore
    from bes.library.retrieve import Library

    run, _, _, _ = tiny
    lib = _qa_library(tmp_path / "lib")
    (lib / "provenance").mkdir()
    np.save(lib / "provenance" / "pool_idx.npy", np.arange(1000, dtype=np.int64))
    np.save(lib / "provenance" / "title_importance.npy", np.linspace(0.0, 1.0, 5))
    (lib / "qa_clean.json").write_text(json.dumps({"kept": len(QA_ROWS)}))
    (lib / "build_settings.json").write_text(json.dumps({"permissive_only": False, "tokenizer_sha256": "t"}))
    loader_files, src = _files_the_loader_reads(lib, monkeypatch)
    acc = src.accounting()
    assert "qa.json" in loader_files and "gate_coef.npy" in loader_files       # the fixture is complete

    res = E.export_run(run, final_dir=tmp_path / "x", library=lib, library_mode="copy", heldout_domains=(),
                       log=lambda m: None)
    d = Path(res["export_dir"])
    shipped = {str(p.relative_to(d / "library")) for p in (d / "library").rglob("*") if p.is_file()}
    assert shipped == loader_files | {"ATTRIBUTION.md", "ATTRIBUTION_QA.tsv"}
    sums = (d / "SHA256SUMS").read_text()
    assert "provenance" not in sums and "qa_clean" not in sums and "build_settings" not in sums
    # every shipped number is counted: ceil(bytes / 2) of each table plus the header numbers the
    # index and the qa layer keep in their JSON files is the library's total, and none is wider
    # than 32 bits (a 32-bit table counts two 16-bit numbers)
    tables = [np.load(d / "library" / f) for f in shipped if f.endswith(".npy")]
    header = TruncatedIndex.HEADER_NUMBERS + QAStore.HEADER_NUMBERS
    assert sum((a.nbytes + 1) // 2 for a in tables) + header == acc["total_numbers"]
    assert all(a.dtype.itemsize <= 4 for a in tables)
    rec = json.loads((d / "export.json").read_text())
    assert rec["library"]["numbers"]["ok"] and rec["library"]["numbers"]["total_numbers"] == acc["total_numbers"]
    assert rec["library"]["left_out"] == ["build_settings.json", "provenance/pool_idx.npy",
                                          "provenance/title_importance.npy", "qa_clean.json"]
    assert rec["accounting"]["library"]["accounting"]["total_numbers"] == acc["total_numbers"]
    # the shipped copy loads, counts and answers as the built library does
    copy = Library.load(str(d / "library"))
    assert copy.accounting() == acc
    for q, _ in QA_ROWS:
        a, b = src.retrieve(q), copy.retrieve(q)
        assert (a is None) == (b is None) and (a is None or a.text == b.text)
    assert E.verify_export(d) == []
    # the card: the qa sources with their licences, what was left out
    card = (d / "MODEL_CARD.md").read_text()
    assert "| trivia_qa_train | mandarjoshi/trivia_qa | rc.nocontext / train | unknown" in card
    assert "| nq_open_train | google-research-datasets/nq_open | nq_open / train | cc-by-sa-3.0" in card
    assert "Left out (not counted, never read by the model)" in card and "`provenance/pool_idx.npy`" in card
    # pointer mode digests the shipped files only: provenance may change, a table may not
    res = E.export_run(run, final_dir=tmp_path / "p", library=lib, library_mode="pointer", heldout_domains=(),
                       log=lambda m: None)
    p = Path(res["export_dir"])
    assert set(json.loads((p / "library_pointer.json").read_text())["files"]) == shipped
    np.save(lib / "provenance" / "pool_idx.npy", np.arange(7, dtype=np.int64))
    assert E.verify_export(p) == []
    (lib / "ATTRIBUTION_QA.tsv").write_text("changed\n")
    assert any("changed since the export" in x for x in E.verify_export(p))


def test_readset_audit_retrieves_later_turns_with_their_context():
    """The read-set audit asks the library what the responder asks it: a later turn of a proxy
    prompt is retrieved with the earlier user turns as its context (Library.retrieve(...,
    context=...), as bes.decode.respond does)."""
    calls = []

    class FakeLib:
        def retrieve(self, q, log=None, context=None):
            calls.append((q, context))
            return None

        def read_bound(self):
            return {"total": 10}

    queries = [("Who was Napoleon?", ()), ("And his wife?", ("Who was Napoleon?",)), "Tell me about Paris."]
    rs = E.readset_report(None, None, None, None, [], lib=FakeLib(), queries=queries)
    assert calls == [("Who was Napoleon?", None), ("And his wife?", ["Who was Napoleon?"]),
                     ("Tell me about Paris.", None)]
    assert rs["library"]["queries"] == 3 and rs["library"]["read_bound"] == 10


def test_export_leaves_out_a_library_the_decode_selection_did_not_keep(tiny, tmp_path):
    """The library decision of the decode selection (scripts/run_24h.py): a library without a
    significant factual gain is neither shipped nor counted, and the card says why with the
    numbers (before: the export always shipped and counted a built library)."""
    run, lib, _, _ = tiny
    report = tmp_path / "proxy" / "reports" / "factual" / "run-factual.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Factual questions\n")
    report.with_suffix(".json").write_text("{}")
    decision = {"ship": False, "rule": "ship the library only if ...",
                "reason": "no significant factual gain (helped 2, hurt 1, p 1.0)",
                "factual": {"n_questions": 20, "n_paired": 20, "fired": 7, "helped": 2, "hurt": 1, "p": 1.0,
                            "lib_acc": 0.35, "nolib_acc": 0.3, "stopped_early": False, "report": str(report)},
                "dev": {"on": 0.5, "off": 0.5, "n_common": 4}}
    dec = {"chosen": "nolib", "overrides": E.default_decode_overrides(load_config(str(CONFIG), OVERRIDES))
           + ["extra.use_library=false"], "scores": {"default": 0.5, "nolib": 0.5}, "n_common": 4,
           "n_dev": 4, "library": decision}
    res = E.export_run(run, final_dir=tmp_path / "x", library=lib, decode=dec, heldout_domains=(),
                       reports={"proxy": str(report.parent.parent)}, log=lambda m: None)
    d = Path(res["export_dir"])
    assert res["library"] == "none" and not (d / "library").exists()
    assert (d / "reports" / "proxy" / "factual" / "run-factual.md").is_file()    # the measurement
    rec = json.loads((d / "export.json").read_text())
    assert rec["accounting"]["library"] is None
    assert rec["library"]["not_shipped"].startswith("the decode selection's library rule did not pass")
    card = (d / "MODEL_CARD.md").read_text()
    assert "No library is shipped with this export: the decode selection's library rule did not pass" in card
    assert "helped 2, hurt 1 (sign test p = 1.0)" in card and "No written library ships with it" in card


def test_model_card_opening_paragraph_is_built_from_the_record():
    """tools-8: the first paragraph states what the shipped model has: no library, K = 1 and the
    switched-off paths for a safe-rung, library-less, single-sample record (before: a fixed text
    naming registers, the candidate tape, a library and particle decoding for every export)."""
    model = {"d_model": 768, "n_blocks": 2, "n_loops": 8, "n_heads": 12, "head_dim": 64, "ffn_width": 3072,
             "vocab_size": 8192, "max_seq": 2048}
    off = {k: 0.0 for k in ("reg", "cand_in", "cand_out", "conf", "sink", "clock")}
    rec = {"format": 1, "config": {"cfg": {"model": model}}, "check": {"ok": True},
           "consts": {"enable": {**off, "kind": [0.0, 0.0], "src": [1.0] * 6}},
           "run": {"benchmark": {"canary": {}, "ladder": {"rung": "safe", "groups": [
               "sink", "conf", "ngram", "library", "tilt", "registers", "calc"]}}},
           "library": {"mode": "none"}, "decode_choice": {"settings": {"particles": 1, "use_library": False}}}
    para = E.model_card(rec).split("## What this is")[1].split("\n- checkpoint")[0]
    assert "No written library ships with it" in para and "K = 1" in para
    assert "particles steered" not in para and "retrieval gate" not in para and "Feature paths on" not in para
    assert "Switched off: registers, candidate-tape input features" in para and "rung safe" in para
    full = dict(rec, consts={"enable": {k: 1.0 for k in off}}, run={},
                library={"mode": "copy", "digest": "ab" * 32},
                decode_choice={"settings": {"particles": 6, "use_library": True}})
    para = E.model_card(full).split("## What this is")[1].split("\n- checkpoint")[0]
    assert "K = 6 particles steered by exact potentials" in para
    assert "ships with it behind a calibrated retrieval gate, on by default" in para
    assert "Switched off" not in para and "Feature paths on: registers" in para


def test_export_cli_verify_and_exit_codes(exported, tmp_path):
    spec = importlib.util.spec_from_file_location("export_cli", REPO / "scripts" / "export.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    d, _, _ = exported
    assert cli.main(["--verify", str(d)]) == 0
    c = tmp_path / "c"
    shutil.copytree(d, c)
    (c / "MODEL_CARD.md").write_text("edited")
    assert cli.main(["--verify", str(c)]) == 1
    assert cli.main(["--run", str(tmp_path / "no-run"), "--final-dir", str(tmp_path / "x")]) == 1


# ======================================================================= chat
def _chat_module():
    spec = importlib.util.spec_from_file_location("bes_chat_test", REPO / "scripts" / "chat.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_lm_follows_reorders_and_finds_calculator_spans():
    from bes.config import CALC, CALC_END, EQ
    from bes.decode.lm import TableLM
    from bes.tokenizer import load as load_tokenizer

    chat = _chat_module()
    lm = chat.TraceLM(TableLM.random(5, order=2, seed=1))
    lm.prefill([0, 1], 3)
    lm.step(np.array([2, 3, 4]), np.array([True, True, False]))
    lm.reorder(np.array([1, 1, 0]))
    lm.step(np.array([0, 1, 2]))
    assert lm.rows == [[3, 0], [3, 1], [2, 2]]
    tok = load_tokenizer()
    ids = tok.encode("So ") + [CALC] + tok.encode("12*7") + [EQ] + tok.encode("84") + [CALC_END] + tok.encode(" ok")
    assert chat.calc_spans(ids, tok) == ["12*7 = 84"]


def test_chat_session_with_scripted_input(exported):
    d, _, _ = exported
    chat = _chat_module()
    lines = iter(["Hello there", "/diag", "What is 2+3?", "/reset", '"""', "line one", "line two", '"""',
                  "/seed 9", "/quit"])
    out: list[str] = []
    assert chat.main(["--export", str(d), "--show-diagnostics", "--max-tokens", "6"],
                     input_fn=lambda prompt: next(lines), print_fn=out.append) == 0
    replies = [o for o in out if o.startswith("bes> ")]
    assert len(replies) == 3
    first = out.index(replies[0])
    assert out[first + 1].startswith("  [library]") and any(o.startswith("  [particles] K=2") for o in out)
    second = out.index(replies[1])
    assert not out[second + 1].startswith("  [")                     # /diag switched them off
    assert "(new conversation)" in out and "(seed 9)" in out and "(diagnostics off)" in out
    # one-shot mode and EOF ends a session
    out2: list[str] = []
    assert chat.main(["--export", str(d), "--prompt", "Hi", "--particles", "1", "--no-library",
                      "--max-tokens", "4"], print_fn=out2.append) == 0
    assert sum(o.startswith("bes> ") for o in out2) == 1 and "library off, K=1" in out2[0]

    def eof(prompt):
        raise EOFError

    assert chat.chat_session(lambda *a: ({}, None), {}, input_fn=eof, print_fn=lambda m: None) == 0


# ======================================================================= notebooks
def _code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


def _as_python(src: str) -> str:
    """IPython shell and magic lines (``!cmd``, ``%magic``) become ``pass`` at their indentation."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(line[:len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_nbformat4_without_outputs_or_secrets(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 4
    assert nb["metadata"]["kernelspec"]["language"] == "python"
    assert nb["cells"] and {c["cell_type"] for c in nb["cells"]} <= {"markdown", "code"}
    for c in nb["cells"]:
        assert isinstance(c["source"], (list, str)) and isinstance(c.get("metadata"), dict)
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)                # no token in the notebook
    assert "getpass" in text and "requirements-colab.txt" in text
    assert '"pip", "install"' in text and "constraints-jax.txt" in text   # jax held fixed
    assert not re.search(r"\bjax(\[|==|>=)", text)                     # never installed or pinned


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_code_cell_is_valid_python_or_a_shell_line(path):
    nb = json.loads(path.read_text())
    for i, src in enumerate(_code_cells(nb)):
        ast.parse(_as_python(src), filename=f"{path.name}[cell {i}]")


def test_notebooks_run_the_right_commands():
    def src(path):
        return "\n".join("".join(c["source"]) for c in json.loads(path.read_text())["cells"])

    v6e, smoke = src(NOTEBOOKS[0]), src(NOTEBOOKS[1])
    for needle in ("scripts/run_24h.py", 'CONFIG = "configs/final.py"', '"--resume"', '"chat.py"',
                   '"arena.py"', "stages.json", "train_log.jsonl", '"v6" in k', "start_new_session=True",
                   "restore_from_drive()", "--verify"):
        assert needle in v6e, needle
    for needle in ('CONFIG = "configs/smoke.py"', "BUDGET_MINUTES = 30", '"--budget-minutes"',
                   "scripts/run_24h.py", '"--resume"'):
        assert needle in smoke, needle
    # the kernel never imports JAX before the run: the device check runs in a subprocess
    first_run = v6e.index("start_background(argv)")
    assert not re.search(r"^\s*(import jax|from jax)", v6e[:first_run], re.M)


def _cell_namespace(path: Path, params: dict) -> dict:
    """Run the parameters cell and the helper cell of a notebook with ``params`` overriding."""
    nb = json.loads(path.read_text())
    cells = _code_cells(nb)
    ns: dict = {}
    exec(compile(_as_python(cells[0]), "params", "exec"), ns)
    ns.update(params)
    exec(compile(_as_python(cells[1]), "helpers", "exec"), ns)
    return ns


def test_notebook_progress_cell_reads_a_run_directory(exported, tiny, capsys):
    run, _, _, _ = tiny
    ns = _cell_namespace(NOTEBOOKS[0], {"RUN_DIR": str(run), "REPO_DIR": str(REPO), "USE_DRIVE": False})
    assert ns["pid_alive"]() is False
    ns["show_progress"]()
    out = capsys.readouterr().out
    assert "run NOT running" in out and "preflight    finished" in out and "judge        skipped" in out
    assert "training ledger: 4,096 tokens, 7 steps" in out


def _nb_cell(path: Path, number: str) -> str:
    """The code cell whose first line starts with ``# <number>.``."""
    for src in _code_cells(json.loads(path.read_text())):
        if src.startswith(f"# {number}."):
            return src
    raise KeyError(number)


def _ckpt(root: Path, name: str, size: int = 10) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "arrays.npz").write_bytes(b"x" * size)
    (root / name / "manifest.json").write_text("{}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_mirror_keeps_one_complete_checkpoint_and_reports_its_state(path, tmp_path, monkeypatch, capsys):
    """tools-4: Drive receives the run's records, logs and the newest complete checkpoint, never
    data (shards, caches, the proxy outputs); while training runs a new checkpoint is copied at
    most once an hour; a failing Drive keeps the last complete checkpoint, and the status file,
    shown by the progress cell, says so. Before: every checkpoint went to Drive and was deleted
    there again (about 15 GB through the mount), errors reached only a log nobody read."""
    ns = _cell_namespace(path, {"RUN_DIR": str(tmp_path / "run"), "REPO_DIR": str(REPO), "USE_DRIVE": True,
                                "DRIVE_DIR": str(tmp_path / "drive")})
    mirror: dict = {"__name__": "mirror"}
    exec(compile(ns["MIRROR_CODE"], "mirror", "exec"), mirror)
    src, dst = tmp_path / "run", tmp_path / "drive"
    ck = src / "train" / "ckpt"
    _ckpt(ck, "ckpt-000001-step-000000010")
    _ckpt(ck, "ckpt-000002-step-000000020")
    (src / "manifest.json").write_text(json.dumps({"created": 1.0}))
    (src / "stages.json").write_text(json.dumps({"stages": {"train": {"status": "started"}}}))
    (src / "train" / "ledger.json").write_text("{}")
    for rel in ("jax_cache/big", "data/tok/climbmix/shard.bin", "proxy/cache/reply.json",
                "train/ckpt/.tmp-ckpt-000003/arrays.npz", "train/reading_cache.npy", "logs/library.log"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"y")
    Path(ns["LOG"]).write_text("run log\n")
    pairs = [(str(src), str(dst)), (ns["LOG"], str(dst) + ".log")]
    status: dict = {}
    mirror["mirror_pass"](pairs, status, 3600.0, now=1000.0)
    on_drive = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
    assert on_drive == ["logs/library.log", "manifest.json", "stages.json",
                        "train/ckpt/ckpt-000002-step-000000020/arrays.npz",
                        "train/ckpt/ckpt-000002-step-000000020/manifest.json", "train/ledger.json"]
    assert Path(str(dst) + ".log").read_text() == "run log\n"
    assert status["errors_last_pass"] == 0 and status["last_good_pass"] == 1000.0
    assert status["checkpoints"] == {"train/ckpt": "ckpt-000002-step-000000020"}
    names = lambda: sorted(p.name for p in (dst / "train" / "ckpt").iterdir())      # noqa: E731
    _ckpt(ck, "ckpt-000003-step-000000030")
    mirror["mirror_pass"](pairs, status, 3600.0, now=1600.0)           # within the hour: the old one stays
    assert names() == ["ckpt-000002-step-000000020"]
    mirror["mirror_pass"](pairs, status, 3600.0, now=4700.0)           # an hour later: replaced
    assert names() == ["ckpt-000003-step-000000030"]
    # Drive over quota once training is over (no hourly limit then): the complete one stays
    _ckpt(ck, "ckpt-000004-step-000000040")
    (src / "stages.json").write_text(json.dumps({"stages": {"train": {"status": "finished"}}}))

    def quota(*a, **k):
        raise OSError(122, "Disk quota exceeded")

    with monkeypatch.context() as m:
        m.setattr(shutil, "copy2", quota)
        m.setattr(shutil, "copytree", quota)
        mirror["mirror_pass"](pairs, status, 3600.0, now=4800.0)
    assert names() == ["ckpt-000003-step-000000030"]
    assert status["errors_last_pass"] >= 1 and "quota" in status["last_error"] and status["last_good_pass"] == 4700.0
    Path(ns["MIRROR_STATUS"]).write_text(json.dumps(status))
    report = "\n".join(ns["mirror_report"](now=4900.0))
    assert "NOT running" in report and "Disk quota exceeded" in report and "WARNING" in report
    assert "ckpt-000003-step-000000030" in report
    ns["show_progress"]()
    assert "Drive mirror NOT running" in capsys.readouterr().out
    mirror["mirror_pass"](pairs, status, 3600.0, now=5000.0)           # Drive works again
    assert names() == ["ckpt-000004-step-000000040"] and status["last_good_pass"] == 5000.0
    # a Drive directory holding another run is never touched
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"created": 2.0}))
    st2: dict = {}
    mirror["mirror_pass"]([(str(src), str(other))], st2, 0.0, now=6000.0)
    assert "holds another run" in st2["last_error"] and sorted(p.name for p in other.iterdir()) == ["manifest.json"]
    # restore on a recycled machine: the records and the newest checkpoint come back
    shutil.rmtree(src)
    Path(ns["LOG"]).unlink()
    ns["restore_from_drive"]()
    assert (src / "manifest.json").exists() and (ck / "ckpt-000004-step-000000040" / "arrays.npz").is_file()
    assert Path(ns["LOG"]).read_text() == "run log\n"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_start_cell_refuses_a_new_run_when_drive_holds_one(path, tmp_path):
    """tools-3: on a recycled machine the local run directory is empty but Drive holds the run:
    cell 6 refuses and points to cell 8 (before: it started a new run with a new budget whose
    mirror replaced the old run on Drive, checkpoints included)."""
    ns = _cell_namespace(path, {"RUN_DIR": str(tmp_path / "run"), "REPO_DIR": str(REPO), "USE_DRIVE": True,
                                "DRIVE_DIR": str(tmp_path / "drive")})
    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "manifest.json").write_text("{}")
    started: list = []
    ns["start_background"] = lambda *a, **k: started.append(a) or 1
    ns["start_mirror"] = lambda *a, **k: started.append("mirror")
    cell = _as_python(_nb_cell(path, "6"))
    with pytest.raises(RuntimeError, match="cell 8"):
        exec(compile(cell, "start", "exec"), ns)
    assert started == []
    (tmp_path / "drive" / "manifest.json").unlink()                   # nothing on Drive: it starts
    exec(compile(cell, "start", "exec"), ns)
    assert len(started) == 2 and started[-1] == "mirror"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_stop_cell_needs_its_explicit_flag(path, tmp_path, monkeypatch):
    """tools-3: 'Run all' runs the stop cell too; it sends SIGTERM only when STOP_THE_RUN is set
    in it (before: it stopped the run that cell 6 had just started)."""
    ns = _cell_namespace(path, {"RUN_DIR": str(tmp_path / "run"), "REPO_DIR": str(REPO), "USE_DRIVE": False})
    Path(ns["PID_FILE"]).parent.mkdir(parents=True, exist_ok=True)
    Path(ns["PID_FILE"]).write_text(str(os.getpid()))                 # a live process
    sent: list = []
    real_kill = os.kill
    monkeypatch.setattr(os, "kill", lambda pid, sig: real_kill(pid, 0) if sig == 0 else sent.append(sig))
    cell = _as_python(_nb_cell(path, "9"))
    assert ns["pid_alive"]()
    exec(compile(cell, "stop", "exec"), ns)
    assert sent == []
    exec(compile(cell.replace("STOP_THE_RUN = False", "STOP_THE_RUN = True"), "stop", "exec"), ns)
    assert sent == [signal.SIGTERM]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_the_code_from_the_local_disk(path, tmp_path, monkeypatch):
    """tools-1: a copy of the repository on Drive is copied to the local REPO_DIR without its data
    and runs there; REPO_DIR itself on Drive is refused. Before: the copy on Drive was used as it
    was, so every download, token shard, the library and the GGUF files went onto Drive."""
    copy = tmp_path / "drive_copy"
    (copy / "bes").mkdir(parents=True)
    (copy / "bes" / "__init__.py").write_text("")
    (copy / "data_cache" / "raw").mkdir(parents=True)
    (copy / "data_cache" / "raw" / "big.parquet").write_bytes(b"z")
    (copy / "runs" / "old").mkdir(parents=True)
    local = tmp_path / "local"
    ns = _cell_namespace(path, {"RUN_DIR": str(tmp_path / "run"), "REPO_DIR": str(local), "CODE_COPY": str(copy),
                                "USE_DRIVE": False})
    cell = _as_python(_nb_cell(path, "3"))
    monkeypatch.chdir(tmp_path)
    exec(compile(cell, "code", "exec"), ns)
    assert Path(os.getcwd()).resolve() == local.resolve() and (local / "bes" / "__init__.py").is_file()
    assert not (local / "data_cache").exists() and not (local / "runs").exists()
    ns = _cell_namespace(path, {"RUN_DIR": str(tmp_path / "run"), "CODE_COPY": "", "USE_DRIVE": False,
                                "REPO_DIR": "/content/drive/MyDrive/TuringAward2027"})
    with pytest.raises(RuntimeError, match="local disk"):
        exec(compile(cell, "code", "exec"), ns)


def _colab_link(repo_url: str, branch: str, name: str) -> str:
    from urllib.parse import quote

    owner_repo = repo_url.removeprefix("https://github.com/").strip("/")
    return f"https://colab.research.google.com/github/{owner_repo}/blob/{quote(branch, safe='')}/notebooks/{name}"


@pytest.mark.parametrize("path, tpu", list(zip(NOTEBOOKS, ("V6E1", "V5E1"))),
                         ids=lambda v: v.name if isinstance(v, Path) else v)
def test_notebook_opens_from_its_colab_link_without_edits(path, tpu):
    """A notebook opened from its "open in Colab" link clones the repository and branch the link
    loaded it from, on the TPU it is written for. Colab reads the branch as one path segment and
    percent-decodes it, so a branch name with a slash is written with %2F; the runtime it offers
    comes from ``metadata.accelerator`` and ``metadata.colab.gpuType`` (V6E1: v6e-1, V5E1: v5e-1)."""
    nb = json.loads(path.read_text())
    ns: dict = {}
    exec(compile(_as_python(_code_cells(nb)[0]), "params", "exec"), ns)
    assert re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", ns["REPO_URL"]) and ns["BRANCH"]
    assert ns["CODE_COPY"] == ""                                        # the clone is what runs
    links = {p.name: _colab_link(ns["REPO_URL"], ns["BRANCH"], p.name) for p in NOTEBOOKS}
    link = links[path.name]
    assert "/" not in link.split("/blob/")[1].split("/notebooks/")[0]   # the branch: one segment
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown" and f'href="{link}"' in "".join(first["source"])
    assert nb["metadata"]["accelerator"] == "TPU" and nb["metadata"]["colab"]["gpuType"] == tpu
    for doc in ("README.md", "RUNBOOK.md"):
        text = (REPO / doc).read_text()
        assert f"]({link})" in text, (doc, link)
        # every Colab link of the document opens a notebook of this repository at this branch
        found = set(re.findall(r"https://colab\.research\.google\.com/github/[^\s)\"'<>]+", text))
        assert found <= set(links.values()), found


# ======================================================================= documents
def test_requirements_pin_the_colab_dependencies_and_never_jax():
    lines = [ln.split("#")[0].strip() for ln in (REPO / "requirements-colab.txt").read_text().splitlines()]
    pkgs = {ln.split("==")[0].lower(): ln for ln in lines if ln}
    assert pkgs["llama-cpp-python"] == "llama-cpp-python==0.3.37" and pkgs["optax"] == "optax==0.2.9"
    for name in ("tiktoken", "rustbpe", "pyarrow", "huggingface_hub", "regex"):
        assert name in pkgs, name
    assert not any(k.startswith(("jax", "libtpu", "torch", "tensorflow")) for k in pkgs)


def test_readme_states_the_accounting_of_the_code():
    """tools-5: the README's headline numbers are what bes.budget gives for the shipped
    configurations with the built library (``budget.V2_ASSETS``), and its strict-convention claim
    carries its condition (before: 19.10M, the plan's placeholder ledger, and 'keeps both under
    20M' without the distinct-token condition)."""
    import functools

    import jax

    from bes import budget
    from bes.model import model as M

    def count(name):
        cfg = load_config(str(REPO / "configs" / f"{name}.py"))
        shapes = jax.eval_shape(functools.partial(M.init_params, cfg.model), jax.random.PRNGKey(0))
        return budget.count(shapes, cfg.model, budget.V2_ASSETS)

    v2 = budget.V2_ASSETS
    final, strict = count("final"), count("strict")
    read = v2.library_read + v2.index_read
    text = (REPO / "README.md").read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| `configs/final.py`"))
    assert f"{final.total:,} stored numbers" in row
    assert f"{final.active_step / 1e6:.2f}M numbers read per generated token" in row
    assert f"{(final.active_step - read) / 1e6:.2f}M on every other step" in row
    first_over = next(d for d in range(400, 2049) if strict.active_strict(2048, d) > 20e6)
    assert strict.active_strict(2048, 840) <= 20e6 < final.active_strict(2048, 840)
    assert f"up to about {first_over - 1} distinct tokens" in row
    assert f"{strict.active_strict(2048, 840) / 1e6:.2f}M at 840" in row
    lib_row = next(ln for ln in text.splitlines() if ln.startswith("| Library `lib_v2`"))
    assert f"{v2.library_numbers + v2.index_numbers:,} stored numbers" in lib_row and f"{read:,}" in lib_row
    assert "19.10M" not in text and "5% less dense core" not in (REPO / "RUNBOOK.md").read_text()


@pytest.mark.parametrize("name", ["RUNBOOK.md", "README.md"])
def test_documents_link_only_to_files_that_exist(name):
    text = (REPO / name).read_text()
    refs = set(re.findall(r"`((?:docs|scripts|configs|bes|notebooks|tests)/[A-Za-z0-9_./-]+\.(?:md|py|ipynb|jsonl))`", text))
    refs |= set(re.findall(r"\]\(((?:docs|scripts|configs|bes|notebooks)/[A-Za-z0-9_./-]+)\)", text))
    assert refs, "the document points nowhere"
    missing = [r for r in sorted(refs) if not (REPO / r).exists()]
    assert not missing, missing
