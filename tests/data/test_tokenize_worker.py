"""Download layer, streaming reader and the tokenisation pool (``bes/data/download.py``,
``bes/data/tokenize_worker.py``).

Token ids are checked against an independent reference renderer of ``docs/SPEC.md`` section 1
written below (with the byte-level stand-in tokenizer); downloads run against a fake Hub that
injects failures. Tests marked ``net`` talk to huggingface.co.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bes.config import ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, SYSTEM, USER
from bes.data import download as D
from bes.data import shards as SH
from bes.data import sources as S
from bes.data import tokenize_worker as W

REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ reference rendering
REF_ROLE_TOKEN = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT, "library": LIBRARY}
REF_ROLE_CODE = {"system": 1, "user": 2, "assistant": 3, "library": 4}


def ref_conversation(msgs) -> tuple[list[int], list[int]]:
    ids, roles = [BOS], [0]
    for m in msgs:
        if isinstance(m["content"], str):
            body = list(m["content"].encode("utf-8"))
        else:
            body = []
            for p in m["content"]:
                if p["type"] == "text":
                    body += list(p["text"].encode("utf-8"))
                else:
                    body += [CALC, *p["expr"].encode("utf-8"), EQ, *p["result"].encode("utf-8"), CALC_END]
        turn = [REF_ROLE_TOKEN[m["role"]], *body, END]
        ids += turn
        roles += [REF_ROLE_CODE[m["role"]]] * len(turn)
    return ids, roles


def ref_segments(source: str, rows: list[dict]) -> list[tuple[list[int], list[int] | None, int]]:
    out = []
    for i, r in enumerate(rows):
        item = S.adapt(source, {c: r.get(c) for c in S.get(source).columns})
        if item is None:
            continue
        if isinstance(item, str):
            out.append(([BOS, *item.encode("utf-8")], None, i))
        else:
            for conv in ([item.chosen, item.rejected] if isinstance(item, S.Pair) else [item]):
                ids, roles = ref_conversation(conv)
                out.append((ids, roles, i))
    return out


# ------------------------------------------------------------------------ test inputs
def text_rows(rng: np.random.Generator, n: int) -> list[dict]:
    words = ["alpha", "beta", "Gamma", "δέλτα", "ünï", "42", "x=y+1;", "\n", "中文", "end."]
    rows = [{"text": " ".join(rng.choice(words, size=int(rng.integers(1, 60))).tolist())} for _ in range(n)]
    rows[1]["text"] = "   "                                  # skipped by the adapter
    return rows


def chat_rows(rng: np.random.Generator, n: int) -> list[dict]:
    rows = []
    for k in range(n):
        msgs = []
        if k % 3 == 0:
            msgs.append({"role": "system", "content": f"Be brief #{k}."})
        for t in range(1 + k % 3):
            msgs += [{"role": "user", "content": f"Question {k}.{t}: {rng.integers(1e6)}?"},
                     {"role": "assistant", "content": f"Answer {k}.{t} <|end|> ü"}]
        if k % 4 == 1:
            msgs.append({"role": "user", "content": "trailing question"})   # dropped by the adapter
        if k % 7 == 5:
            msgs.insert(1, {"role": "tool", "content": "{}"})                # rejected by the adapter
        rows.append({"messages": msgs, "source": "synthetic"})
    return rows


def pref_rows(n: int) -> list[dict]:
    rows = []
    for k in range(n):
        u = {"role": "user", "content": f"prompt {k}"}
        rows.append({"chosen": [u, {"role": "assistant", "content": f"good {k}"}],
                     "rejected": [u, {"role": "assistant", "content": f"bad {k}"}],
                     "score_chosen": 8.0, "score_rejected": 8.0 if k % 3 == 2 else 3.5})
    return rows


def gsm_rows() -> list[dict]:
    return [{"question": "Q1?", "answer": "She has 2+3 = <<2+3=5>>5 apples.\n#### 5"},
            {"question": "Q2?", "answer": "It costs $<<4*2.5=10.0>>10.0 in total.\n#### 10"}]


def write_parquet(path: Path, rows: list[dict], row_group_size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


def local_files(tmp: Path, source: str, row_chunks: list[list[dict]], start_index: int = 0) -> list[D.LocalFile]:
    out = []
    for k, rows in enumerate(row_chunks):
        p = tmp / "raw" / source / f"part-{k:03d}.parquet"
        write_parquet(p, rows)
        out.append(D.LocalFile(source, start_index + k, "test/repo", "0" * 40, p.name, str(p),
                               p.stat().st_size, None))
    return out


INPUTS = {
    "fineweb_edu": lambda rng: text_rows(rng, 40),
    "smol_smoltalk": lambda rng: chat_rows(rng, 30),
    "ultrafeedback": lambda rng: pref_rows(12),
    "gsm8k": lambda rng: gsm_rows(),
}


def split(rows: list[dict], k: int) -> list[list[dict]]:
    step = -(-len(rows) // k)
    return [rows[i:i + step] for i in range(0, len(rows), step)]


def quiet(msg: str) -> None:
    pass


def shard_state(directory: Path) -> tuple[dict[str, bytes], list[dict]]:
    """All committed shard file bytes and the manifest entries without timing."""
    files = {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if SH._STEM_RE.match(p.name)}
    entries = sorted((SH.read_manifest(directory)).values(), key=lambda e: e["shard"])
    return files, [{k: v for k, v in e.items() if k != "timing"} for e in entries]


# ------------------------------------------------------------------- tokenisation tests
@pytest.mark.parametrize("source", sorted(INPUTS))
def test_shards_match_reference(tmp_path, source):
    rows = INPUTS[source](np.random.default_rng(0))
    chunks = split(rows, 2)
    files = local_files(tmp_path, source, chunks, start_index=5)
    out = tmp_path / "tok"
    stats = W.run(source, files=files, out_dir=out, workers=1, tokenizer="bytes", log=quiet)
    manifest = SH.read_manifest(out)
    assert sorted(manifest) == [5, 6] and stats.committed == 2
    ss = SH.ShardSet(out)
    ref = [seg for chunk in chunks for seg in ref_segments(source, chunk)]
    assert len(ss) == len(ref)
    for j, (ids, roles, rec) in enumerate(ref):
        assert ss.segment(j).tolist() == ids
        got_roles = ss.segment_roles(j)
        assert (got_roles is None) == (roles is None)
        if roles is not None:
            assert got_roles.tolist() == roles
        k, i = ss.locate(j)
        assert int(ss.shards[k].recs[i]) == rec
    for e, chunk in zip([manifest[5], manifest[6]], chunks):
        n_items = sum(1 for r in chunk if S.adapt(source, {c: r.get(c) for c in S.get(source).columns}) is not None)
        assert e["stats"]["records"] == len(chunk)
        assert e["stats"]["skipped"] == len(chunk) - n_items
        assert e["pairs"] == (source == "ultrafeedback") and e["has_roles"] == (source != "fineweb_edu")
        assert e["tokenizer"] == "bytes/1" and e["input"]["index"] == e["shard"]
    assert stats.tokens == ss.n_tokens == sum(len(x[0]) for x in ref)
    assert stats.tokens_per_cpu_s > 0 and stats.text_bytes > 0
    rep = SH.check(out)
    assert rep.ok and not rep.warnings, rep


def test_gsm8k_calculator_spans_reach_the_shard(tmp_path):
    files = local_files(tmp_path, "gsm8k", [gsm_rows()])
    out = tmp_path / "tok"
    W.run("gsm8k", files=files, out_dir=out, workers=0, tokenizer="bytes", log=quiet)
    seg = SH.ShardSet(out).segment(1).tolist()
    expr, result = list(b"4*2.5"), list(b"10")              # "10.0" trimmed to "10"
    i = seg.index(CALC)
    assert seg[i:i + len(expr) + len(result) + 3] == [CALC, *expr, EQ, *result, CALC_END]


def test_output_is_independent_of_worker_count(tmp_path):
    rng = np.random.default_rng(1)
    states = []
    for workers in (0, 1, 2):
        out = tmp_path / f"w{workers}"
        for source in ("fineweb_edu", "smol_smoltalk"):
            rows = INPUTS[source](np.random.default_rng(7))
            files = local_files(tmp_path / f"in{workers}", source, split(rows, 3))
            W.run(source, files=files, out_dir=out / source, workers=workers, tokenizer="bytes", log=quiet)
        states.append({s: shard_state(out / s) for s in ("fineweb_edu", "smol_smoltalk")})
    assert states[0] == states[1] == states[2]
    del rng


class Boom(Exception):
    pass


def test_restart_after_crash_in_the_coordinator(tmp_path):
    rows = INPUTS["smol_smoltalk"](np.random.default_rng(2))
    files = local_files(tmp_path, "smol_smoltalk", split(rows, 5))
    clean = tmp_path / "clean"
    W.run("smol_smoltalk", files=files, out_dir=clean, workers=2, tokenizer="bytes", log=quiet)
    out = tmp_path / "crashy"

    def crash(entry):
        raise Boom()

    with pytest.raises(Boom):
        W.run("smol_smoltalk", files=files, out_dir=out, workers=2, tokenizer="bytes", log=quiet, on_commit=crash)
    assert len(SH.read_manifest(out)) == 1
    # a torn line from a crash during an append, and stale temporaries of a dead writer
    with open(out / SH.MANIFEST, "ab") as f:
        f.write(b'{"shard": 3, "n_tok')
    (out / "000004.u16.tmp.99999.deadbeef").write_bytes(b"\x00\x01")
    stats = W.run("smol_smoltalk", files=files, out_dir=out, workers=2, tokenizer="bytes", log=quiet)
    assert stats.already_done == 1 and stats.committed == 4
    assert shard_state(out) == shard_state(clean)
    objs, problems = SH.scan_jsonl(out / SH.MANIFEST)
    assert not problems and sorted(o["shard"] for o in objs) == [0, 1, 2, 3, 4]
    rep = SH.check(out)
    assert rep.ok and not rep.warnings, rep
    again = W.run("smol_smoltalk", files=files, out_dir=out, workers=2, tokenizer="bytes", log=quiet)
    assert again.committed == 0 and again.already_done == 5


KILL_SCRIPT = """
import json, multiprocessing, os, signal, sys
from bes.data import tokenize_worker as W
from bes.data.download import LocalFile
files = [LocalFile(**d) for d in json.loads(sys.argv[2])]
def die(entry):
    with open(sys.argv[4], "w") as f:
        json.dump([p.pid for p in multiprocessing.active_children()], f)
    os.kill(os.getpid(), signal.SIGKILL)
W.run(sys.argv[1], files=files, out_dir=sys.argv[3], workers=2, tokenizer="bytes",
      log=lambda m: None, on_commit=die)
"""


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/status").read_text().split("State:")[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


def test_restart_after_sigkill(tmp_path):
    rows = INPUTS["fineweb_edu"](np.random.default_rng(3)) * 4
    files = local_files(tmp_path, "fineweb_edu", split(rows, 6))
    clean = tmp_path / "clean"
    W.run("fineweb_edu", files=files, out_dir=clean, workers=1, tokenizer="bytes", log=quiet)
    out, pids_file = tmp_path / "killed", tmp_path / "pids.json"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with open(tmp_path / "child.log", "wb") as log:     # files, not pipes: nothing can block on them
        proc = subprocess.run([sys.executable, "-c", KILL_SCRIPT, "fineweb_edu",
                               json.dumps([asdict(f) for f in files]), str(out), str(pids_file)],
                              cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode == -9, (tmp_path / "child.log").read_text()[-2000:]
    workers = json.loads(pids_file.read_text())
    assert len(workers) == 2
    deadline = time.monotonic() + 10
    while any(_alive(p) for p in workers) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(_alive(p) for p in workers), "workers outlived the killed coordinator"
    committed = SH.read_manifest(out)
    assert len(committed) == 1
    stats = W.run("fineweb_edu", files=files, out_dir=out, workers=2, tokenizer="bytes", log=quiet)
    assert stats.already_done == 1 and stats.committed == 5
    assert shard_state(out) == shard_state(clean)
    rep = SH.check(out)
    assert rep.ok and not rep.warnings, rep


class SuicidalTokenizer(W.ByteTokenizer):
    """Same ids and fingerprint as ``bytes``, but its process dies on a marker (an OOM kill)."""

    def encode(self, text):
        if "KILL-ME" in text:
            os.kill(os.getpid(), signal.SIGKILL)
        return super().encode(text)


def test_a_dead_worker_fails_its_files_instead_of_hanging(tmp_path):
    rows = INPUTS["fineweb_edu"](np.random.default_rng(11))
    chunks = split(rows, 4)
    chunks[2] = chunks[2] + [{"text": "KILL-ME please"}]
    files = local_files(tmp_path, "fineweb_edu", chunks)
    clean = tmp_path / "clean"
    W.run("fineweb_edu", files=files, out_dir=clean, workers=1, tokenizer="bytes", log=quiet)
    out = tmp_path / "tok"
    t0 = time.monotonic()
    stats = W.run("fineweb_edu", files=files, out_dir=out, workers=2, tokenizer=SuicidalTokenizer(),
                  strict=False, log=quiet)
    assert time.monotonic() - t0 < 60
    assert stats.failed and any("died" in e["error"] for e in stats.failed)
    assert 2 not in SH.read_manifest(out)
    assert stats.committed + len(stats.failed) == len(files)
    again = W.run("fineweb_edu", files=files, out_dir=out, workers=2, tokenizer="bytes", log=quiet)
    assert again.committed == len(stats.failed)
    assert shard_state(out) == shard_state(clean)
    assert SH.check(out).ok and not SH.check(out).warnings


class OtherTokenizer(W.ByteTokenizer):
    fingerprint = "other/1"


def test_tokenizer_and_max_records_must_match_existing_shards(tmp_path):
    files = local_files(tmp_path, "fineweb_edu", split(INPUTS["fineweb_edu"](np.random.default_rng(4)), 2))
    out = tmp_path / "tok"
    W.run("fineweb_edu", files=files[:1], out_dir=out, workers=0, tokenizer="bytes", log=quiet)
    with pytest.raises(RuntimeError, match="tokenizer"):
        W.run("fineweb_edu", files=files, out_dir=out, workers=0, tokenizer=OtherTokenizer(), log=quiet)
    with pytest.raises(RuntimeError, match="max_records"):
        W.run("fineweb_edu", files=files, out_dir=out, workers=0, tokenizer="bytes", max_records=3, log=quiet)
    assert W.tokenizer_fingerprint(OtherTokenizer()) == "other/1"


def test_max_records_and_counters(tmp_path):
    rows = INPUTS["smol_smoltalk"](np.random.default_rng(5))
    files = local_files(tmp_path, "smol_smoltalk", split(rows, 2))
    out = tmp_path / "tok"
    stats = W.run("smol_smoltalk", files=files, out_dir=out, workers=0, tokenizer="bytes", max_records=4, log=quiet)
    ref = [seg for chunk in split(rows, 2) for seg in ref_segments("smol_smoltalk", chunk[:4])]
    ss = SH.ShardSet(out)
    assert [ss.segment(j).tolist() for j in range(len(ss))] == [r[0] for r in ref]
    assert stats.records == 8 and stats.segments == len(ref) and stats.tokens == sum(len(r[0]) for r in ref)
    kept = [S.adapt("smol_smoltalk", r) for chunk in split(rows, 2) for r in chunk[:4]]
    assert stats.text_bytes == sum(len(m["content"].encode()) for c in kept if c for m in c)
    assert stats.skipped == sum(c is None for c in kept)


def test_bad_input_file_is_not_committed(tmp_path):
    files = local_files(tmp_path, "fineweb_edu", split(INPUTS["fineweb_edu"](np.random.default_rng(6)), 2))
    Path(files[1].local).write_bytes(b"this is not parquet")
    out = tmp_path / "tok"
    stats = W.run("fineweb_edu", files=files, out_dir=out, workers=1, tokenizer="bytes", strict=False, log=quiet)
    assert len(stats.failed) == 1 and sorted(SH.read_manifest(out)) == [0]
    assert SH.check(out).ok and not SH.check(out).warnings
    with pytest.raises(RuntimeError, match="failed"):
        W.run("fineweb_edu", files=files, out_dir=out, workers=1, tokenizer="bytes", log=quiet)


def test_adapter_failures_above_threshold_abort_the_file(tmp_path, monkeypatch):
    files = local_files(tmp_path, "fineweb_edu", [INPUTS["fineweb_edu"](np.random.default_rng(8))])
    src = S.get("fineweb_edu")

    def explode(record):
        raise KeyError("boom")

    monkeypatch.setitem(S.REGISTRY, "fineweb_edu", S.Source(**{**src.__dict__, "adapter": explode}))
    stats = W.run("fineweb_edu", files=files, out_dir=tmp_path / "tok", workers=0, tokenizer="bytes",
                  strict=False, log=quiet)
    assert len(stats.failed) == 1 and "adapter failed" in stats.failed[0]["error"]
    assert SH.read_manifest(tmp_path / "tok") == {}
    assert not any(".tmp." in p.name for p in (tmp_path / "tok").iterdir())


def test_library_sources_are_not_tokenised(tmp_path):
    with pytest.raises(ValueError, match="library"):
        W.run("library_simplewiki", files=[], out_dir=tmp_path, workers=0, tokenizer="bytes", log=quiet)


def test_resolve_tokenizer_specs():
    assert isinstance(W.resolve_tokenizer("bytes"), W.ByteTokenizer)
    t = W.ByteTokenizer()
    assert W.resolve_tokenizer(t) is t
    with pytest.raises(TypeError):
        W.resolve_tokenizer(object())
    fp = W.tokenizer_fingerprint(type("T", (), {"vocab_size": 9, "encode": lambda self, s: [1, 2]})())
    assert fp.startswith("probe:")


# -------------------------------------------------------------------- streaming reader
def _stream(p: Path) -> tuple[int, int, int]:
    """(peak Arrow allocation while streaming, rows, CRC of the texts in order)."""
    base = pa.total_allocated_bytes()
    peak, n, crc = 0, 0, 0
    for batch in D.iter_records(p, ["text"], batch_rows=64, buffer_bytes=1 << 20):
        peak = max(peak, pa.total_allocated_bytes() - base)
        for r in batch:
            assert set(r) == {"text"}
            crc = zlib.crc32(r["text"].encode(), crc)
            n += 1
    return peak, n, crc


def test_iter_records_streams_one_huge_row_group(tmp_path):
    """Files made of ONE row group (like PleIAs/SYNTH): memory must not grow with its size."""
    rng = np.random.default_rng(9)
    peaks = []
    for mult in (1, 4):
        texts = [rng.bytes(4000).hex() for _ in range(2500 * mult)]     # 20 MB, then 80 MB of text
        p = tmp_path / f"big{mult}.parquet"
        pq.write_table(pa.table({"text": texts, "other": list(range(len(texts)))}), p,
                       row_group_size=len(texts), compression="zstd")
        md = pq.ParquetFile(p).metadata
        assert md.num_row_groups == 1
        column_bytes = md.row_group(0).column(0).total_uncompressed_size
        peak, n, crc = _stream(p)
        assert n == len(texts) == D.num_rows(p) and crc == zlib.crc32("".join(texts).encode())
        peaks.append(peak)
        del texts
    assert peaks[1] < column_bytes / 2, (peaks, column_bytes)       # far below the 80 MB column
    assert peaks[1] <= peaks[0] + (4 << 20), peaks                  # does not grow with the row group
    with pytest.raises(KeyError, match="missing columns"):
        next(D.iter_records(p, ["text", "nope"]))


# ---------------------------------------------------------------------- download tests
class FakeHub:
    """In-memory repository with failure injection for ``fetch``."""

    def __init__(self, files: dict[str, bytes], commit: str = "c" * 40):
        self.files, self.commit = files, commit
        self.list_calls = 0
        self.fetch_calls: list[str] = []
        self.plan: dict[str, list] = {}       # path -> exceptions / "corrupt" to apply in order

    def lister(self, repo, revision, prefix):
        self.list_calls += 1
        return self.commit, [D.RemoteFile(p, len(b), hashlib.sha256(b).hexdigest())
                             for p, b in self.files.items() if p.startswith(prefix)]

    def fetch(self, repo, path, commit, staging):
        assert commit == self.commit
        self.fetch_calls.append(path)
        action = self.plan.get(path, []).pop(0) if self.plan.get(path) else None
        if isinstance(action, BaseException):
            raise action
        target = Path(staging) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"garbage" if action == "corrupt" else self.files[path])
        return target


def climbmix_hub() -> FakeHub:
    sizes = [100, 200, 300, 400, 500]
    files = {f"shard_{i:05d}.parquet": bytes([i]) * s for i, s in enumerate(sizes)}
    files["README.md"] = b"readme"
    return FakeHub(files)


def test_select_files_matches_brute_force():
    files = [D.RemoteFile(f"f{i}", s) for i, s in enumerate([5, 1, 7, 3, 3, 9])]
    for max_files in (None, 0, 1, 2, 4, 10):
        for max_bytes in (None, 0, 1, 5, 6, 13, 16, 100):
            ref = []
            for f in files:
                if max_files is not None and len(ref) >= max_files:
                    break
                if max_bytes is not None and ref and sum(x.size for x in ref) + f.size > max_bytes:
                    break
                ref.append(f)
            assert D.select_files(files, max_files, max_bytes) == ref


def test_listing_is_cached_and_filtered(tmp_path):
    hub = climbmix_hub()
    commit, files = D.source_listing("climbmix", tmp_path, lister=hub.lister)
    assert commit == hub.commit and [f.path for f in files] == [f"shard_{i:05d}.parquet" for i in range(5)]
    hub.files["shard_00009.parquet"] = b"new upstream file"           # the cache pins the listing
    assert D.source_listing("climbmix", tmp_path, lister=hub.lister) == (commit, files)
    assert hub.list_calls == 1
    gsm = FakeHub({"main/train-00000-of-00001.parquet": b"t", "main/test-00000-of-00001.parquet": b"x",
                   "socratic/train-00000-of-00001.parquet": b"s"})
    _, gfiles = D.source_listing("gsm8k", tmp_path, lister=gsm.lister)
    assert [f.path for f in gfiles] == ["main/train-00000-of-00001.parquet"]
    assert D.static_prefix("main/train-*.parquet") == "main" and D.static_prefix("shard_*.parquet") == ""


def test_download_caps_retries_and_manifest(tmp_path):
    hub = climbmix_hub()
    hub.plan["shard_00001.parquet"] = [ConnectionError("reset"), TimeoutError("slow"), "corrupt"]
    sleeps, logs = [], []
    got = D.download_source("climbmix", tmp_path, max_bytes=650, fetch=hub.fetch, lister=hub.lister,
                            backoff=1.0, log=logs.append, sleep=sleeps.append)
    assert [f.index for f in got] == [0, 1, 2]
    rdir = D.repo_dir("climbmix", tmp_path)
    for f in got:
        assert Path(f.local).read_bytes() == hub.files[f.path]
    assert hub.fetch_calls.count("shard_00001.parquet") == 4 and len(sleeps) == 3
    assert all(0.5 * 2 ** k <= s <= 2 ** k for k, s in enumerate(sleeps)) and len(logs) == 3
    man, problems = SH.scan_jsonl(rdir / D.MANIFEST)
    assert not problems and [m["path"] for m in man] == [f.path for f in got]
    assert not list((rdir / ".staging").rglob("*.parquet"))
    again = D.download_source("climbmix", tmp_path, max_bytes=650, fetch=hub.fetch, lister=hub.lister, log=quiet)
    assert again == got and len(hub.fetch_calls) == 6                 # nothing fetched again
    assert D.completed_files("climbmix", tmp_path, lister=hub.lister) == got
    assert D.download_source("climbmix", tmp_path, max_files=1, max_bytes=None, fetch=hub.fetch,
                             lister=hub.lister, log=quiet) == got[:1]


def test_permanent_errors_are_not_retried(tmp_path):
    from huggingface_hub import errors as hf_errors

    hub = climbmix_hub()
    hub.plan["shard_00000.parquet"] = [hf_errors.EntryNotFoundError("gone")]
    sleeps = []
    with pytest.raises(hf_errors.EntryNotFoundError):
        D.download_source("climbmix", tmp_path, max_files=1, fetch=hub.fetch, lister=hub.lister,
                          log=quiet, sleep=sleeps.append)
    assert sleeps == [] and D.completed_files("climbmix", tmp_path, lister=hub.lister) == []
    hub.plan["shard_00000.parquet"] = [ConnectionError("x")] * 3
    with pytest.raises(ConnectionError):
        D.download_source("climbmix", tmp_path, max_files=1, fetch=hub.fetch, lister=hub.lister,
                          retries=3, log=quiet, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_interrupted_download_leaves_no_trace_and_resumes(tmp_path):
    hub = climbmix_hub()
    hub.plan["shard_00002.parquet"] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        D.download_source("climbmix", tmp_path, max_files=3, fetch=hub.fetch, lister=hub.lister, log=quiet)
    rdir = D.repo_dir("climbmix", tmp_path)
    assert not (rdir / "shard_00002.parquet").exists()
    assert [f.index for f in D.completed_files("climbmix", tmp_path, lister=hub.lister)] == [0, 1]
    got = D.download_source("climbmix", tmp_path, max_files=3, fetch=hub.fetch, lister=hub.lister, log=quiet)
    assert [f.index for f in got] == [0, 1, 2] and hub.fetch_calls.count("shard_00000.parquet") == 1


def test_renamed_but_unrecorded_file_is_recorded_without_refetch(tmp_path):
    hub = climbmix_hub()
    D.source_listing("climbmix", tmp_path, lister=hub.lister)
    rdir = D.repo_dir("climbmix", tmp_path)
    (rdir / "shard_00003.parquet").write_bytes(hub.files["shard_00003.parquet"])     # crash after rename
    (rdir / "shard_00004.parquet").write_bytes(b"half-written")                      # stale garbage
    rf = {f.path: f for f in D.source_listing("climbmix", tmp_path, lister=hub.lister)[1]}
    D.download_file("r", hub.commit, rf["shard_00003.parquet"], rdir, fetch=hub.fetch, log=quiet)
    D.download_file("r", hub.commit, rf["shard_00004.parquet"], rdir, fetch=hub.fetch, log=quiet)
    assert hub.fetch_calls == ["shard_00004.parquet"]
    assert (rdir / "shard_00004.parquet").read_bytes() == hub.files["shard_00004.parquet"]
    assert [f.index for f in D.completed_files("climbmix", tmp_path, lister=hub.lister)] == [3, 4]


def test_threaded_downloads_record_each_file_once(tmp_path):
    hub = climbmix_hub()
    got = D.download_source("climbmix", tmp_path, max_bytes=None, threads=2, fetch=hub.fetch,
                            lister=hub.lister, log=quiet)
    assert [f.index for f in got] == [0, 1, 2, 3, 4]
    man, _ = SH.scan_jsonl(D.repo_dir("climbmix", tmp_path) / D.MANIFEST)
    assert sorted(m["path"] for m in man) == sorted(f.path for f in got)


def test_missing_pinned_commit_falls_back_to_main(tmp_path):
    from huggingface_hub import errors as hf_errors

    class Resp:
        headers: dict = {}
        status_code = 404
        url = "https://huggingface.co/x"
        text = ""
        request = None

        def json(self):
            return {}

    hub = climbmix_hub()
    seen = []

    def lister(repo, revision, prefix):
        seen.append(revision)
        if revision != "main":
            raise hf_errors.RevisionNotFoundError("no such commit", response=Resp())
        return hub.lister(repo, revision, prefix)

    logs = []
    commit, files = D.source_listing("climbmix", tmp_path, lister=lister, log=logs.append)
    assert seen == [S.get("climbmix").revision, "main"] and len(files) == 5
    assert any("not found" in m for m in logs)


# ---------------------------------------------------------------------------- network
def _tokenizer_spec():
    try:
        from bes.tokenizer import tok

        tok.load()
        return W.DEFAULT_TOKENIZER
    except Exception:  # noqa: BLE001 - assets not built yet
        return "bytes"


@pytest.mark.net
def test_end_to_end_download_and_tokenise_one_real_file(tmp_path):
    files = D.download_source("gsm8k", tmp_path, max_files=1, log=quiet)
    assert len(files) == 1 and files[0].path == "main/train-00000-of-00001.parquet"
    assert files[0].sha256 and hashlib.sha256(Path(files[0].local).read_bytes()).hexdigest() == files[0].sha256
    assert D.completed_files("gsm8k", tmp_path) == files
    stats = W.run("gsm8k", tmp_path, workers=1, tokenizer=_tokenizer_spec(), log=quiet)
    out = W.tok_dir("gsm8k", tmp_path)
    rep = SH.check(out)
    assert rep.ok and not rep.warnings, rep
    n_rows = D.num_rows(files[0].local)
    assert stats.records == n_rows == S.get("gsm8k").approx_rows and stats.segments == n_rows
    ss = SH.ShardSet(out)
    seg = ss.segment(0)
    assert int(seg[0]) == BOS and CALC in seg.tolist() and int(seg[-1]) == END
    assert int(np.count_nonzero(np.isin(ss.shards[0].tokens, [CALC]))) > n_rows    # several spans per solution


@pytest.mark.net
def test_registry_listings_match_the_hub(tmp_path):
    """Every entry: file count and bytes at its revision equal what was verified."""
    for s in S.SOURCES:
        commit, files = D.source_listing(s, tmp_path)
        assert len(files) == s.n_files, s.name
        if s.revision != "refs/convert/parquet":
            assert commit == s.verified_sha and sum(f.size for f in files) == s.approx_bytes, s.name
