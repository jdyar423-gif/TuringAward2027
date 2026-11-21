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
import re
import shutil
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


def test_notebook_mirror_copies_and_prunes_checkpoints(tmp_path):
    ns = _cell_namespace(NOTEBOOKS[0], {"RUN_DIR": str(tmp_path / "run"), "REPO_DIR": str(REPO),
                                        "USE_DRIVE": True, "DRIVE_DIR": str(tmp_path / "drive")})
    mirror: dict = {"__name__": "mirror"}
    exec(compile(ns["MIRROR_CODE"], "mirror", "exec"), mirror)
    src, dst = tmp_path / "run", tmp_path / "drive"
    for name in ("ckpt-000001-step-000000010", "ckpt-000002-step-000000020"):
        (src / "train" / "ckpt" / name).mkdir(parents=True)
        (src / "train" / "ckpt" / name / "arrays.npz").write_bytes(b"x" * 10)
    (src / "jax_cache").mkdir()
    (src / "jax_cache" / "big").write_bytes(b"y")
    (src / "train" / "ckpt" / ".tmp-ckpt-000003").mkdir()
    (src / "manifest.json").write_text("{}")
    mirror["sync"](str(src), str(dst))
    assert (dst / "manifest.json").exists() and not (dst / "jax_cache").exists()
    assert sorted(p.name for p in (dst / "train" / "ckpt").iterdir()) == [
        "ckpt-000001-step-000000010", "ckpt-000002-step-000000020"]
    shutil.rmtree(src / "train" / "ckpt" / "ckpt-000001-step-000000010")      # pruned locally
    mirror["sync"](str(src), str(dst))
    assert [p.name for p in (dst / "train" / "ckpt").iterdir()] == ["ckpt-000002-step-000000020"]
    # restore after a recycled machine
    shutil.rmtree(src)
    ns["restore_from_drive"]()
    assert (src / "manifest.json").exists() and (src / "train" / "ckpt" / "ckpt-000002-step-000000020").is_dir()


# ======================================================================= documents
def test_requirements_pin_the_colab_dependencies_and_never_jax():
    lines = [ln.split("#")[0].strip() for ln in (REPO / "requirements-colab.txt").read_text().splitlines()]
    pkgs = {ln.split("==")[0].lower(): ln for ln in lines if ln}
    assert pkgs["llama-cpp-python"] == "llama-cpp-python==0.3.37" and pkgs["optax"] == "optax==0.2.9"
    for name in ("tiktoken", "rustbpe", "pyarrow", "huggingface_hub", "regex"):
        assert name in pkgs, name
    assert not any(k.startswith(("jax", "libtpu", "torch", "tensorflow")) for k in pkgs)


@pytest.mark.parametrize("name", ["RUNBOOK.md", "README.md"])
def test_documents_link_only_to_files_that_exist(name):
    text = (REPO / name).read_text()
    refs = set(re.findall(r"`((?:docs|scripts|configs|bes|notebooks|tests)/[A-Za-z0-9_./-]+\.(?:md|py|ipynb|jsonl))`", text))
    refs |= set(re.findall(r"\]\(((?:docs|scripts|configs|bes|notebooks)/[A-Za-z0-9_./-]+)\)", text))
    assert refs, "the document points nowhere"
    missing = [r for r in sorted(refs) if not (REPO / r).exists()]
    assert not missing, missing
