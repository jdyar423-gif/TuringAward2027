"""Throughput of the training loader (``bes.data.loader``) on real token shards.

Builds (``--build``) small real shards from the parquet samples in ``data_cache/samples`` with
``bes.data.tokenize_worker`` into a separate data root (default ``data_cache/loader_bench``), then
measures rows per second per core for base rows and chat rows at T=1024 and T=2048 (one stream
set, in this process, so CPU seconds are per core), the padding share of chat rows, and where
the CPU time goes (shard reads and cutting, hindsight, retrieval, library tasks, packing, row
assembly, exact-layer features). ``--workers N`` also runs the process pool and reports the
wall-clock rate and the coordinator's copy and wait time. Prints one JSON document.

    python scripts/bench_loader.py --build
    python scripts/bench_loader.py --steps 6 --library data_cache/library/lib_v2 --workers 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = {   # source -> (sample parquet relative to data_cache/samples, max records)
    "climbmix": ("karpathy__climbmix-400b-shuffle/shard_00000.parquet", 12000),
    "simplewiki": ("wikimedia__wikipedia/20231101.simple/train-00000-of-00001.parquet", 6000),
    "smol_smoltalk": ("HuggingFaceTB__smol-smoltalk/data/train-00000-of-00004.parquet", 30000),
}


def build(out_root: Path, samples_root: Path, workers: int) -> dict:
    """Tokenise the sample files into ``out_root/tok/<source>/`` (skips committed shards)."""
    from bes.data import download, shards, tokenize_worker

    out = {}
    for name, (rel, max_records) in SAMPLES.items():
        path = samples_root / rel
        if not path.exists():
            out[name] = "missing sample"
            continue
        lf = download.LocalFile(source=name, index=0, repo="local-sample", commit="sample", path=rel,
                                local=str(path), size=path.stat().st_size)
        directory = out_root / "tok" / name
        stats = tokenize_worker.run(name, files=[lf], out_dir=directory, workers=workers,
                                    max_records=max_records, strict=True, log=lambda m: None)
        man = shards.read_manifest(directory)
        out[name] = {"shards": len(man), "tokens": sum(e["n_tokens"] for e in man.values()),
                     "segments": sum(e["n_segments"] for e in man.values()),
                     "built_now": stats.committed}
    return out


def small_cfg(seq_len: int, micro_seqs: int, accum: int):
    """The final configuration with a smaller step (``accum`` micro-batches of ``micro_seqs``
    rows of ``seq_len``); the late length stays ``2 * seq_len``."""
    from bes.config import Cfg

    cfg = Cfg()
    return dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, tokens_per_step=accum * micro_seqs * seq_len,
                                                            micro_seqs=micro_seqs),
                               data=dataclasses.replace(cfg.data, seq_len=seq_len, seq_len_late=2 * seq_len))


def stage(cfg, p: float, chat_frac: float, lib_base: float, lib_chat: float):
    from bes.train.schedule import stage_at

    return dataclasses.replace(stage_at(p, cfg), chat_frac=chat_frac, library_frac_base=lib_base,
                               library_frac_chat=lib_chat)


def totals(stats: dict) -> dict[str, Counter]:
    """Counters summed over the blocks of ``Loader.stats()``."""
    out = {"rows": Counter(), "pad": Counter(), "sub": Counter(), "time": Counter(), "fail": Counter(),
           "cut": Counter(), "retrieval": Counter(), "coord": Counter(stats["coordinator"])}
    for b in stats["blocks"]:
        row_len = b["shape"][2] + 1
        out["rows"].update(b["rows"])
        for rk, share in b["pad_share"].items():
            out["pad"][rk] += int(round(share * b["rows"][rk] * row_len))
        for k, v in b["tokens"].items():
            rk, sub, _ = k.split("/", 2)
            out["sub"][f"{rk}/{sub}"] += v
        out["time"].update(b["time_s"])
        out["fail"].update(b["fail"])
        out["cut"].update(b["cut_tokens"])
        out["retrieval"].update(b["retrieval"]["counts"])
    return out


def run_case(name: str, cfg, st, data_root: Path, tok, library, steps: int, n_workers: int,
             lcfg) -> dict:
    """``steps`` batches after one warm-up batch; CPU time of this process (n_workers = 0,
    everything runs here) or wall time (pool)."""
    from bes.data.loader import Loader

    journal = Path(tempfile.mkdtemp()) / "journal.jsonl"
    a, bsz, t = st.shape
    # warm-up until every stream produced 3 rows: its look-ahead buffers are full (steady state)
    warm = max(1, -(-3 * lcfg.n_streams // (a * bsz)))
    with Loader(cfg, tok, data_root, library=library, n_workers=n_workers, seed=0,
                loader_cfg=lcfg, journal=journal) as loader:
        for i in range(warm):
            loader(i, st)
        before = totals(loader.stats())
        c0, w0 = time.process_time(), time.perf_counter()
        for i in range(warm, warm + steps):
            loader(i, st)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        after = totals(loader.stats())
    d = {k: after[k] - before[k] for k in after}
    n_rows = steps * a * bsz
    rows = d["rows"]
    sub_total = max(1, sum(d["sub"].values()))
    out = {"case": name, "shape": list(st.shape), "workers": n_workers, "warmup_steps": warm,
           "steps": steps, "rows": n_rows, "rows_by_kind": dict(rows),
           "pad_share": {k: d["pad"][k] / (v * (t + 1)) for k, v in rows.items() if v},
           "token_share_by_subkind": {k: round(v / sub_total, 4) for k, v in sorted(d["sub"].items())},
           "worker_cpu_s_by_phase": {k: round(v, 3) for k, v in sorted(d["time"].items())},
           "worker_ms_per_row_by_phase": {k: round(1e3 * v / n_rows, 3) for k, v in sorted(d["time"].items())},
           "coordinator": {k: round(v, 4) for k, v in d["coord"].items()},
           "fail": dict(d["fail"]), "cut_tokens": dict(d["cut"]), "transforms": dict(d["retrieval"])}
    time_s = d["time"]
    measured = sum(time_s.values())
    if n_workers == 0:
        out["cpu_s"] = round(cpu, 3)
        out["rows_per_cpu_s"] = round(n_rows / cpu, 1)
        out["rows1024_per_cpu_s"] = round(n_rows * t / 1024 / cpu, 1)
        out["unattributed_cpu_s"] = round(max(0.0, cpu - measured), 3)   # copy-out, coordination
    else:
        out["wall_s"] = round(wall, 3)
        out["rows_per_s"] = round(n_rows / wall, 1)
        out["rows1024_per_s"] = round(n_rows * t / 1024 / wall, 1)
        out["worker_cpu_s"] = round(measured, 3)
        out["coordinator_cpu_s"] = round(cpu, 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data_cache" / "loader_bench"))
    ap.add_argument("--samples", default=str(ROOT / "data_cache" / "samples"))
    ap.add_argument("--build", action="store_true", help="tokenise the samples first")
    ap.add_argument("--build-workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4, help="timed batches per case")
    ap.add_argument("--micro", type=int, default=2, help="micro-batches per step (A)")
    ap.add_argument("--library", default=None, help="library directory (retrieval and library rows)")
    ap.add_argument("--workers", default="", help="also run the process pool, e.g. '2,3'")
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--lookahead", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=1e7,
                    help="LoaderCfg.horizon_tokens; the default keeps the epoch caps of these small "
                         "shards from binding, so the stage decides the mixture")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case; the fastest is kept")
    ap.add_argument("--cases", default="base1024,chat1024,base2048,chat2048,mix1024,mix2048")
    args = ap.parse_args(argv)
    from bes.config import LoaderCfg
    from bes.tokenizer.tok import load

    data_root = Path(args.data)
    report: dict = {"machine": {"cpus": os.cpu_count(), "load_avg_start": os.getloadavg()}}
    if args.build:
        report["build"] = build(data_root, Path(args.samples), args.build_workers)
    tok = load()
    library = None
    if args.library:
        from bes.library.retrieve import Library

        library = Library.load(args.library)
        report["library"] = {"dir": args.library, "wiki": library.n_wiki, "qa": library.n_qa}
    lcfg = LoaderCfg(n_streams=args.streams, horizon_tokens=args.horizon,
                     **({"lookahead": args.lookahead} if args.lookahead else {}))
    lib_chat = 0.4 if library is not None else 0.0
    # base: base rows only; chat: chat rows only (no library); lib: library rows only (reading
    # tasks only over rag_format.reading_suitable entries, as the loader prepares the library);
    # mix: chat_frac 0.25 (T=1024) / 0.5 (T=2048) with library rows at library_frac_base 0.035,
    # more than the epoch cap leaves them in the real run with lib_v2 (about 0.6 % of the tokens
    # at chat_frac 0.25), so the mix is on the expensive side
    stages = {"base": (0.0, 0.0, 0.0), "chat": (1.0, 0.0, 0.0), "lib": (0.0, 1.0, 0.0),
              "mix": (None, 0.035 if library is not None else 0.0, lib_chat)}
    cases = []
    for case in args.cases.split(","):
        kind, t = case[:-4], int(case[-4:])
        b = 32 if t == 1024 else 16
        cfg = small_cfg(1024, 32, args.micro)
        p = 0.1 if t == 1024 else 0.6
        chat, lib_base, lib_c = stages[kind]
        st = stage(cfg, p, (0.25 if t == 1024 else 0.5) if chat is None else chat, lib_base, lib_c)
        assert st.shape == (args.micro, b, t), st.shape
        cases.append((case, cfg, st))
    report["cases"] = []
    pools = [int(x) for x in args.workers.split(",") if x.strip()]
    for case, cfg, st in cases:
        for nw in [0] + pools:
            runs = [run_case(case, cfg, st, data_root, tok, library, args.steps, nw, lcfg)
                    for _ in range(max(1, args.repeat))]
            best = min(runs, key=lambda r: r.get("cpu_s", r.get("wall_s")))
            best["repeats"] = len(runs)
            report["cases"].append(best)
    report["machine"]["load_avg_end"] = os.getloadavg()
    report["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
