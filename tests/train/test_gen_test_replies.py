"""scripts/gen_test_replies.py: our export's replies to proxy prompts in parallel worker processes.

A tiny export (smoke configuration, no library) is written with ``bes.export.export_run``. Two
worker processes answer a one-turn and a two-turn prompt; their conversations must equal those of
the single-process harness (``bes.eval.proxy.runner.run`` with the export's own generator and
fixed cache, built from the run's recorded backend spec), a second call must generate nothing,
``--merge-into`` must give another harness root the same run, and a snapshot must unpack into the
run it holds. The elastic cache: the prefill state is the plain one's used part bit for bit, the
calibration keeps only lengths whose real-state checks pass, a reply on them is the plain reply,
a corrupted growth is caught; the parent keeps speculative replies only when the calibration covers
their (length, span) usage. A prompt that raises is recorded and skipped (exit 0 when anything was
produced, 1 when nothing was).
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "smoke.py"
OVERRIDES = ["model.n_loops=2", "decode.particles=2", "decode.max_new_tokens=12"]
IDS = ["chat-025", "math-012"]                  # math-012 has two turns

WEB = ["The river runs through the old town. Bridges cross it at three places. " * 6,
       "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen. " * 8]
CHAT = [[{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}]] * 4


def _script():
    spec = importlib.util.spec_from_file_location("gen_test_replies", REPO / "scripts" / "gen_test_replies.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


G = _script()


@pytest.fixture(scope="module")
def run_and_export(tmp_path_factory):
    """``(run_dir, export_dir)``: a smoke-sized checkpoint, its manifest and its export."""
    import jax

    from bes import export as E
    from bes.config import config_hash, load_config
    from bes.model import model as M
    from bes.train import checkpoint as ck

    base = tmp_path_factory.mktemp("testgen")
    cfg = load_config(str(CONFIG), OVERRIDES)
    params = jax.tree.map(np.asarray, M.init_params(cfg.model, jax.random.PRNGKey(3)))
    params["head"]["out"] = params["head"]["out"] * 300.0          # informative logits
    run = base / "run"
    tree = {"state": {"step": np.int32(7), "params": params,
                      "opt_state": {"m": jax.tree.map(np.zeros_like, params["head"])},
                      "consts": M.make_consts(cfg.model)}, "rng": jax.random.PRNGKey(1)}
    ck.save(run / "train" / "ckpt", tree, step=7, config_hash=config_hash(cfg))
    (run / "manifest.json").write_text(json.dumps({
        "format": 1, "config": str(CONFIG), "config_abs": str(CONFIG), "overrides": OVERRIDES,
        "budget": {"t0": 1000.0, "deadline": 4600.0, "budget_s": 3600.0},
        "code": {"commit": "test", "dirty": False}, "config_hash": config_hash(cfg), "resumes": []}))
    res = E.export_run(run, library_mode="none", heldout_rows=2, download_heldout=False,
                       heldout_records={"web": WEB, "chat": CHAT}, log=lambda m: None)
    return run, Path(res["export_dir"])


def _settings(**extra):
    from bes.eval.proxy.common import GenSettings

    return GenSettings(temperature=0.6, top_p=0.9, max_tokens=12,
                       extra={"particles": 2, "use_library": False, **extra})


def _strip(records):
    """Conversation records without the measured latencies (everything else must be equal)."""
    out = {}
    for pid, r in records.items():
        out[pid] = {**r, "turns": [{k: v for k, v in t.items() if k != "latency_s"} for t in r["turns"]]}
    return out


# ===================================================================== cores and sources
def test_core_plan_keeps_siblings_for_whole_cores_and_spreads_single_threads(tmp_path):
    groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    cpus = list(range(8))
    assert G.plan_cores(cpus, 4, 2, groups) == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert G.plan_cores(cpus, 4, 1, groups) == [[0], [1], [2], [3]]          # one thread per core first
    assert G.plan_cores(cpus, 8, 1, groups) == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert G.plan_cores(cpus, 2, 4, groups) == [[0, 4, 1, 5], [2, 6, 3, 7]]
    assert G.plan_cores([2, 3], 2, 1, [[2], [3]]) == [[2], [3]]
    with pytest.raises(ValueError):
        G.plan_cores(cpus, 5, 2, groups)
    assert G.parse_cpu_list("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}
    for c, sib in ((0, "0,2"), (1, "1,3"), (2, "0,2"), (3, "1,3")):
        d = tmp_path / f"cpu{c}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(sib + "\n")
    assert G.sibling_groups([0, 1, 2, 3], sysfs=str(tmp_path)) == [[0, 2], [1, 3]]
    assert G.sibling_groups([1, 2], sysfs=str(tmp_path)) == [[1], [2]]      # only allowed siblings


def test_snapshot_unpacks_into_its_run_and_a_current_final_export_is_used(run_and_export, tmp_path):
    run, export = run_and_export
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w") as tar:
        tar.add(str(run), arcname="run")
    data = blob.getvalue()
    snap = tmp_path / "snap"
    snap.mkdir()
    parts, size = [], max(1, len(data) // 3 + 1)
    for i in range(0, len(data), size):
        name = f"run.tar.{len(parts):03d}"
        (snap / name).write_bytes(data[i:i + size])
        parts.append({"name": name, "bytes": len(data[i:i + size]),
                      "sha256": hashlib.sha256(data[i:i + size]).hexdigest()})
    (snap / "parts.json").write_text(json.dumps({"parts": parts, "sha256": hashlib.sha256(data).hexdigest(),
                                                 "steps": 7}))
    logs = []
    got = G.unpack_snapshot(snap, tmp_path / "view", logs.append)
    assert got == tmp_path / "view" / "run" and (got / "manifest.json").is_file()
    assert G.unpack_snapshot(snap, tmp_path / "view", logs.append) == got and len(logs) == 1   # reused

    class A:
        make_export, library, no_download, data_dir = False, None, True, None

    path, how = G.export_of_run(got, tmp_path / "out", A(), logs.append)
    assert path == got / "final" and how["how"] == "the run's final export"
    (snap / parts[1]["name"]).write_bytes(b"x" + data[size + 1:2 * size])          # a damaged part
    with pytest.raises(SystemExit, match="does not match"):
        G.unpack_snapshot(snap, tmp_path / "view2", logs.append)


# ===================================================================== elastic cache
def test_elastic_cache_lengths_calibration_and_replies(run_and_export, monkeypatch):
    from bes.eval.proxy.run_ours import settings_for_generator
    from bes.export import export_generator

    _, export = run_and_export
    gen = export_generator(export=str(export))
    # candidate lengths and the spans they are checked at
    assert G.candidate_lengths(5248, 256) == [256, 512, 768, 1024, 2048, 3072]
    assert G.candidate_lengths(384, 64) == [64, 128, 192, 256, 320]
    assert G.check_spans(3072) == list(range(192, 3009, 256)) and G.check_spans(256) == [192]
    assert G.span_limit(1024) == 1024 - 64 + G.CHECK_STEPS - 1
    # 12 new tokens: the export's own cache has 384 slots; every candidate is checked on real
    # states of one prompt against it, the class end to end on a short prompt
    chk = G.calibrate_elastic(gen, _settings(), 64)
    assert chk["reference_slots"] == 384 and chk["candidates"] == [128, 192, 256, 320]
    lengths = {int(t): v for t, v in chk["lengths"].items()}
    assert set(lengths) <= {128, 192, 256, 320} and all(v == G.span_limit(t) for t, v in lengths.items())
    assert all(c in G.check_spans(t) and t not in lengths for t, c in chk["failed"])
    assert chk["elastic_ok"] == bool(lengths) and chk["checks"] >= len(lengths)
    if lengths:
        assert chk["end_to_end"]["ok"]
    warm = G.warm_elastic(gen, _settings(), 64)
    assert warm["compiled_slots"] == [128, 192]
    # a reply on the calibrated lengths is the plain reply: text, tokens, log-probabilities, extra
    s = settings_for_generator(_settings())
    msgs = [{"role": "user", "content": "Tell me about the river and the bridges of the old town."}]
    plain = gen(msgs, s, 11)
    orig = gen.responder.lm_factory
    made = []

    def elastic(n):
        made.append(G.make_elastic(orig(n), 64, lengths))
        return made[-1]

    gen.responder.lm_factory = elastic
    assert gen(msgs, s, 11) == plain
    lm = made[-1]
    assert lm.full_slots == 384 and lm.capacity == min(lm.max_steps, lm.full_slots - lm.base_slot)
    assert G.usage_ok(lm.usage, lengths, 384) and lm.usage
    gen.responder.lm_factory = orig
    with pytest.raises(ValueError):
        G.make_elastic(orig(12), 100)
    # lengths: the shortest one whose span limit holds the step, never shorter than the cache is;
    # past them the plain length
    el = G.make_elastic(orig(12), 64, {"128": 67, 256: 195})
    el._tmax(10)
    assert (el.start_slots, el.length_for(100), el.length_for(200, at_least=129), el.full_slots) == (128, 256, 384, 384)
    assert G.usage_ok({128: 67, 384: 300}, {128: 67}, 384) and not G.usage_ok({128: 68}, {128: 67}, 384)
    assert not G.usage_ok({192: 10}, {128: 67}, 384)
    # the prompt is prefilled on the plain length: after the prefill the elastic state is the
    # plain state's used part, bit for bit, also for a prompt of several prefill chunks
    ids = G.long_prompt(gen.responder.chat, 150)
    assert len(ids) > 128
    a, b = orig(12), G.make_elastic(orig(12), 64)
    ra, rb = a.prefill(ids, 2), b.prefill(ids, 2)
    assert np.array_equal(ra, rb) and b.slots == b.start_slots == 256 < a.cache["seg"].shape[1]
    for name in ("k", "v"):
        assert np.array_equal(np.asarray(a.cache[name])[:, :, :, :, :b.slots], np.asarray(b.cache[name]))
        assert not np.asarray(a.cache[name])[:, :, :, :, b.slots:].any()                 # nothing cut off
    for name in ("seg", "pos", "t", "next_pos", "cur_seg"):
        x, y = np.asarray(a.cache[name]), np.asarray(b.cache[name])
        assert np.array_equal(x[:, :b.slots] if x.ndim == 2 else x, y)
    # a growth that changes the cache is found by the calibration's end-to-end check
    if lengths:
        cls = G.elastic_class()
        grow_to = cls.grow_to

        def bad_grow(self, new):
            grow_to(self, new)
            self.cache = {**self.cache, "k": self.cache["k"] * 1.001}

        monkeypatch.setattr(cls, "grow_to", bad_grow)
        bad = G.calibrate_elastic(gen, _settings(), 64)
        assert not bad["elastic_ok"] and bad["lengths"] == {} and not bad["end_to_end"]["ok"]


def test_the_parent_decides_the_cache_from_worker_0s_calibration(tmp_path):
    import argparse
    import types

    from bes.eval.proxy import runner
    from bes.eval.proxy.common import GenSettings

    spec = runner.RunSpec(backend={"kind": "test"}, key="k", settings=GenSettings(), label="t")

    def gen(cache="elastic", root="root"):
        args = argparse.Namespace(minutes=None, grace_minutes=3.0, cache=cache, progress_s=30.0, restarts=2,
                                  max_rss_gb=0.0, start_timeout=900.0, nice=0)
        return G.Generation(args, spec, tmp_path, [], [], tmp_path / root, tmp_path, lambda m: None, 0.0)

    ok = {"role": "calibrate", "elastic_ok": True, "lengths": {"256": 195, "512": 451}, "candidates": [256, 512, 768],
          "failed": [[768, 704]], "checks": 4, "reference_slots": 5248, "particles": 1, "slot_bytes": 98304}
    ready = {"type": "ready", "backend": "cpu", "platforms": ["cpu"], "rss_mb": 1000}
    g = gen()
    assert g.mode is None and g.guess == {"mode": "elastic", "lengths": None}
    g.handle(G.Worker(1, [1]), {**ready, "warmup": {"role": "warm"}})
    assert g.mode is None                                  # only worker 0's calibration decides
    g.handle(G.Worker(0, [0]), {**ready, "warmup": ok})
    assert g.mode == {"mode": "elastic", "lengths": {256: 195, 512: 451}} and not g.stopping
    bad = {**ok, "elastic_ok": False, "lengths": {}, "particles": 6}
    g = gen()
    g.handle(G.Worker(0, [0]), {**ready, "warmup": bad})
    assert g.mode == {"mode": "full"} and 1 <= g.max_busy < 10 ** 9 and not g.stopping     # exact, fewer at once
    g = gen("elastic-always")
    g.handle(G.Worker(0, [0]), {**ready, "warmup": bad})
    assert g.mode == {"mode": "elastic", "lengths": None}
    assert gen("full").mode == {"mode": "full"}
    for backend, platforms in (("tpu", ["tpu"]), ("cpu", ["cpu", "tpu"])):
        g = gen()
        g.handle(G.Worker(0, [0]), {**ready, "backend": backend, "platforms": platforms, "warmup": ok})
        assert "not on the CPU only" in g.setup_error and g.stopping == g.setup_error and g.mode is None
    # speculative replies (handed out before the calibration) report the (length, span) pairs
    # they stepped on: kept when the calibration covers all of them, else dropped and handed out again
    line = {"key": "g1", "prompt_id": "chat-001", "turn": 0,
            "reply": {"text": "hi", "latency_s": 1.0, "completion_tokens": 1, "extra": {"steps": 3, "particles": 2}}}
    p = types.SimpleNamespace(id="chat-001", turns=("hello",))
    cases = (({"256": 100}, True), ({"256": 100, "5248": 3000}, True), ({"768": 700}, False), ({"256": 300}, False))
    for i, (usage, kept) in enumerate(cases):
        g = gen(root=f"spec-{i}")
        w1 = G.Worker(1, [1])
        w1.task, w1.spec = p, True
        g.handle(w1, {"type": "line", "prompt_id": "chat-001", "line": line, "usage": usage})
        g.handle(w1, {"type": "done", "prompt_id": "chat-001", "record": {"turns": []}, "seconds": 1.0})
        assert len(g.held) == 2 and not g.records and w1.task is None
        g.handle(G.Worker(0, [0]), {**ready, "warmup": ok})
        assert g.held == [] and g.speculation["kept_replies"] == int(kept)
        assert ("chat-001" in g.records) is kept and (list(g.requeue) == [p]) is (not kept)
        assert ("g1" in g.cached) is kept
    # a task still in flight at the decision: its later replies are checked as they come
    g = gen(root="spec-flight")
    w1 = G.Worker(1, [1])
    w1.task, w1.spec = p, True
    g.workers = [G.Worker(0, [0]), w1]
    g.handle(g.workers[0], {**ready, "warmup": ok})
    assert not w1.spec and g.speculation["in_flight"] == 1
    g.handle(w1, {"type": "line", "prompt_id": "chat-001", "line": line, "usage": {"512": 600}})
    g.handle(w1, {"type": "done", "prompt_id": "chat-001", "record": {"turns": []}, "seconds": 1.0})
    assert "g1" not in g.cached and not g.records and list(g.requeue) == [p] and not w1.dropped


# ===================================================================== the script
def test_two_workers_reply_as_the_single_process_harness_then_resume_and_merge(run_and_export, tmp_path,
                                                                             capsys):
    from bes.eval.proxy import prompts as prompt_set
    from bes.eval.proxy import runner
    from bes.eval.proxy.common import GenSettings, read_jsonl

    _, export = run_and_export
    out = tmp_path / "out"
    n_workers = min(2, len(os.sched_getaffinity(0)))
    argv = [str(export), "--out", str(out), "--ids", ",".join(IDS), "--workers", str(n_workers),
            "--cores-per-worker", "1", "--progress-s", "0", "--cache-step", "64"]
    assert G.main(argv) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("TESTGEN ")
    short = json.loads(lines[-1][len("TESTGEN "):])
    assert short["status"] == "complete" and short["done"] == 2 and short["new_replies"] == 3
    summary = json.loads((out / "testgen.json").read_text())
    assert summary["workers_ready"] == n_workers
    assert summary["decode_mode"] == ({"mode": "elastic", "lengths": summary["calibration"]["lengths"]}
                                      if summary["calibration"]["lengths"] else {"mode": "full"})
    assert summary["worker_details"][0]["warmup"]["role"] == "calibrate"
    for w in summary["worker_details"]:                     # JAX on the CPU only, in every worker
        assert (w["backend"], w["platforms"], w["jax_platforms"]) == ("cpu", ["cpu"], "cpu")
    assert summary["overrides"][-1] == "extra.use_library=false"            # the export has no library
    # the single-process harness, from the run's own recorded backend spec (export_generator)
    meta = json.loads(Path(summary["files"]["meta"]).read_text())
    assert meta["backend"]["factory"] == "bes.export:export_generator"
    spec = runner.RunSpec(backend=meta["backend"], key=meta["backend_key"],
                          settings=GenSettings.from_dict(meta["settings"]), label=meta["label"],
                          seed=int(meta["seed"]))
    assert spec.run_id == summary["run_id"]
    ps = prompt_set.load()
    todo = [p for p in prompt_set.select(ps, "all") if p.id in IDS]
    ref = runner.run(todo, spec, tmp_path / "ref", workers=0, all_prompts=ps, log=None)
    assert ref.done == 2 and ref.new_generations == 3
    ours, _ = runner.load_outputs(out / "proxy", spec.run_id)
    theirs, _ = runner.load_outputs(tmp_path / "ref", spec.run_id)
    assert _strip(ours) == _strip(theirs)
    assert len(ours["math-012"]["turns"]) == 2
    rows = read_jsonl(out / "replies.jsonl")
    assert [r["id"] for r in rows] == [p.id for p in todo]
    two = next(r for r in rows if r["id"] == "math-012")
    assert two["turns"] == list(next(p for p in todo if p.id == "math-012").turns)
    assert two["replies"] == [t["reply"] for t in ours["math-012"]["turns"]] and len(two["finish"]) == 2
    with tarfile.open(out / "testgen.tar.gz") as tar:
        names = set(tar.getnames())
    assert {"replies.jsonl", "testgen.json", "testgen.log"} <= names
    assert f"proxy/outputs/{spec.run_id}.jsonl" in names and f"proxy/cache/{spec.key}.jsonl" in names
    # a second call generates nothing
    assert G.main(argv) == 0
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1][len("TESTGEN "):])
    assert again["reused"] == 2 and again["new_replies"] == 0 and again["workers"] == 0
    assert runner.load_outputs(out / "proxy", spec.run_id)[0] == ours
    # merged into another root (where the opponents' runs live)
    merged = tmp_path / "merged"
    assert G.main([str(out), "--merge-into", str(merged)]) == 0
    assert "run id: " + spec.run_id in capsys.readouterr().out
    assert runner.load_outputs(merged, spec.run_id)[0] == ours
    assert G.main([str(out), "--merge-into", str(merged)]) == 0              # idempotent
    assert len(read_jsonl(runner.cache_path(merged, spec.key))) == 3


def test_a_prompt_that_raises_is_recorded_and_skipped(run_and_export, tmp_path, capsys):
    _, export = run_and_export
    out = tmp_path / "out"
    argv = [str(export), "--out", str(out), "--workers", "1", "--cache", "full", "--no-warmup", "--progress-s", "0",
            "--ids", ",".join(IDS), "--fail-ids", "math-012"]
    assert G.main(argv) == 0                                # something was produced: exit 0
    short = json.loads(capsys.readouterr().out.strip().splitlines()[-1][len("TESTGEN "):])
    assert (short["status"], short["done"], short["failed"], short["missing"]) == ("partial", 1, 1, 1)
    summary = json.loads((out / "testgen.json").read_text())
    assert "injected" in summary["failed"]["math-012"] and summary["missing_ids"] == ["math-012"]
    assert "math-012 failed on worker 0" in (out / "testgen.log").read_text()
    from bes.eval.proxy.common import read_jsonl

    assert [r["id"] for r in read_jsonl(out / "replies.jsonl")] == ["chat-025"]
    # nothing produced: non-zero
    argv = [str(export), "--out", str(tmp_path / "out2"), "--workers", "1", "--cache", "full", "--no-warmup",
            "--ids", "math-012", "--fail-ids", "math-012"]
    assert G.main(argv) == 1
    short = json.loads(capsys.readouterr().out.strip().splitlines()[-1][len("TESTGEN "):])
    assert (short["status"], short["done"], short["failed"]) == ("failed", 0, 1)


def test_a_missing_source_fails_cleanly(tmp_path, capsys):
    assert G.main([str(tmp_path / "nothing"), "--out", str(tmp_path / "o")]) == 2
    last = capsys.readouterr().out.strip().splitlines()[-1]
    short = json.loads(last[len("TESTGEN "):])
    assert short["status"] == "failed" and "neither an export" in short["error"]
