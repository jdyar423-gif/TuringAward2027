"""Build the written library (packed passages + stored truncated index) for one budget.

    python scripts/build_library.py --budget 60000000 --out data_cache/library/lib_60M

Steps: download the source once (data_cache/library/raw/), clean and split every article
(cached in data_cache/library/pool/), compute selection signals (cached), select under the
budget of stored numbers (text blocks + directory + titles), pack, build the index, save.
Prints the exact accounting as JSON. Measured on the whole Simple English Wikipedia (2
threads): about 5 minutes and 2.2 GB peak from scratch, 1.5 minutes and 2.1 GB with the
cached pool; the output is byte-identical between runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bes.library.build import build_for_budget, prepare_pool, prepare_signals  # noqa: E402
from bes.library.clean import default_count_tokens  # noqa: E402
from bes.library.index import IndexCfg  # noqa: E402
from bes.library.pack import PackCfg  # noqa: E402
from bes.library.retrieve import RetrieveCfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data_cache/library")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=60_000_000,
                    help="stored numbers for text blocks + directory + titles")
    ap.add_argument("--index-budget", type=int, default=IndexCfg().budget_numbers,
                    help="stored numbers of the truncated index")
    ap.add_argument("--n-bits", type=int, default=IndexCfg().n_bits)
    ap.add_argument("--max-tokens", type=float, default=RetrieveCfg().max_tokens)
    ap.add_argument("--count-tokens", choices=("proxy", "bpe"), default="proxy",
                    help="token counter for the passage cap: UTF-8 bytes / 3.3, or the BPE tokenizer "
                         "(bes.tokenizer.tok); changing it requires --rebuild-pool")
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--rebuild-signals", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.cache_dir, f"lib_{a.budget // 1_000_000}M")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    count_tokens, ct_name = default_count_tokens, "utf8_bytes/3.3"
    if a.count_tokens == "bpe":
        from bes.tokenizer import tok

        t = tok.load()
        enc = t.encode if hasattr(t, "encode") else tok.encode
        count_tokens, ct_name = (lambda s: len(enc(s))), "bpe"
    pool = prepare_pool(a.cache_dir, rebuild=a.rebuild_pool, max_tokens=a.max_tokens,
                        count_tokens=count_tokens, log=log)
    if pool.meta.get("count_tokens", "utf8_bytes/3.3") != ct_name:
        raise SystemExit(f"the cached pool was built with {pool.meta.get('count_tokens', 'utf8_bytes/3.3')} "
                         f"token counts; rerun with --rebuild-pool")
    pack_cfg = PackCfg()
    sig = prepare_signals(pool, a.cache_dir, pack_cfg, rebuild=a.rebuild_signals or a.rebuild_pool, log=log)
    index_cfg = dataclasses.replace(IndexCfg(), budget_numbers=a.index_budget, n_bits=a.n_bits)
    lib, _ = build_for_budget(pool, sig, a.budget, index_cfg, pack_cfg,
                              retrieve_cfg=RetrieveCfg(max_tokens=a.max_tokens), count_tokens=count_tokens,
                              out_dir=out, log=log, cache_dir=a.cache_dir, count_tokens_name=ct_name)
    import resource

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"out": out, "peak_rss_mb": round(peak_mb), **lib.accounting()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
