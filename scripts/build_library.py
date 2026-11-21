"""Build the written library v2 (wiki passages + stored BM25 index + title look-up + qa
layer + calibrated gate).

    python scripts/build_library.py --out data_cache/library/lib_v2

Steps: download the sources once (data_cache/library/raw/), clean and split every article
with BPE token counts (cached in data_cache/library/pool_bpe/), compute selection signals
(cached), select under the budget of stored numbers (text blocks + directory + titles),
pack, build the index and the title table, clean and store the question-answer pairs
(evaluation questions excluded), fit the gate on dev questions, save. Prints the exact
accounting as JSON (and, with --digest, the SHA-256 of every output file). The output is
byte-identical between runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bes.config import LibraryCfg  # noqa: E402
from bes.library.build import BuildCfg, build_v2, dir_digest  # noqa: E402
from bes.library.clean import bpe_count_tokens, proxy_count_tokens  # noqa: E402
from bes.library.index import IndexCfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    L = LibraryCfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data_cache/library")
    ap.add_argument("--out", default=L.library_dir)
    ap.add_argument("--budget", type=int, default=L.budget_numbers,
                    help="stored numbers for wiki text blocks + directory + titles")
    ap.add_argument("--index-budget", type=int, default=IndexCfg().budget_numbers,
                    help="stored numbers of the truncated BM25 index")
    ap.add_argument("--index-max-list", type=int, default=IndexCfg().max_list)
    ap.add_argument("--n-bits", type=int, default=IndexCfg().n_bits)
    ap.add_argument("--no-qa", action="store_true", help="build without the question-answer layer")
    ap.add_argument("--permissive-only", action="store_true", default=L.qa_permissive_only,
                    help="qa layer from sources with a clear licence only (NQ-open, CC BY-SA 3.0)")
    ap.add_argument("--count-tokens", choices=("bpe", "proxy"), default="bpe",
                    help="token counter for the 160-token cap: the BPE tokenizer, or UTF-8 bytes / 3.3")
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--rebuild-signals", action="store_true")
    ap.add_argument("--digest", action="store_true", help="print the SHA-256 of every output file")
    a = ap.parse_args(argv)
    t0 = time.time()

    import resource

    def log(msg: str) -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{time.time() - t0:7.1f}s {rss:5.0f} MB] {msg}", flush=True)

    count_tokens = bpe_count_tokens if a.count_tokens == "bpe" else proxy_count_tokens
    icfg = dataclasses.replace(IndexCfg(), budget_numbers=a.index_budget, max_list=a.index_max_list,
                               n_bits=a.n_bits)
    cfg = BuildCfg(budget_numbers=a.budget, index=icfg, qa=not a.no_qa, permissive_only=a.permissive_only)
    lib, report = build_v2(a.cache_dir, a.out, cfg, count_tokens, log, a.rebuild_pool, a.rebuild_signals)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out = {"out": a.out, "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(peak_mb),
           "gate": report["gate"], **lib.accounting()}
    if a.digest:
        out["digest"] = dir_digest(a.out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
