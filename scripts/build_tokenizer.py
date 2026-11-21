#!/usr/bin/env python
"""Train the committed byte-level BPE (8192 ids) and write ``assets/tokenizer/``.

    python scripts/build_tokenizer.py                # download samples if needed, train, evaluate
    python scripts/build_tokenizer.py --eval-only    # held-out bytes/token of the committed asset

The training sample is a fixed prefix of pinned parquet files (``bes/tokenizer/samples.py``)
downloaded into ``data_cache/samples/``. After training, the script reloads the written asset
and measures on held-out text of every domain: bytes per token, the round trip
``decode(encode(t)) == t`` and the agreement of the tiktoken encoder with rustbpe's own encoder.
The results go to ``assets/tokenizer/meta.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
from importlib import metadata
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web-mb", type=float, default=260.0, help="MB of web text (climbmix)")
    ap.add_argument("--chat-mb", type=float, default=120.0, help="MB of chat message contents (smol-smoltalk)")
    ap.add_argument("--wiki-mb", type=float, default=50.0, help="MB of Simple English Wikipedia")
    ap.add_argument("--doc-cap", type=int, default=10_000, help="characters kept per training text")
    ap.add_argument("--heldout-mb", type=float, default=8.0, help="MB of held-out text per domain")
    ap.add_argument("--threads", type=int, default=2, help="threads for rustbpe and tiktoken")
    ap.add_argument("--out", default=None, help="output directory (default: assets/tokenizer)")
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading samples")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the existing asset, write nothing")
    return ap.parse_args(argv)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def training_stream(budgets: dict[str, float], doc_cap: int, stats: dict):
    """Texts of the training mix, domain after domain; fills ``stats[domain]`` as it goes."""
    from bes.tokenizer import samples

    for domain, mb in budgets.items():
        rec = stats.setdefault(domain, {"texts": 0, "bytes": 0})
        for text in samples.iter_texts(domain, "train", int(mb * 1e6), doc_cap=doc_cap):
            rec["texts"] += 1
            rec["bytes"] += len(text.encode("utf-8"))
            yield text


_WIKI_TABLE = re.compile(r"^(\{\||\|-|\|\}|\|\||!)", re.M)


def has_wiki_table(text: str) -> bool:
    """True for pages that still carry wiki-table markup (lists of minor planets and the like)."""
    return bool(_WIKI_TABLE.search(text)) or text.count("||") >= 3


def evaluate(tok, heldout_mb: float, threads: int, trainer=None) -> dict:
    """Bytes per token, round-trip failures and (optionally) rustbpe agreement per domain.

    ``wiki_without_tables`` repeats the wiki measurement without pages that carry wiki-table
    markup: the library cleaner removes such markup, so this is closer to what the model reads.
    """
    from bes.tokenizer import samples

    out = {}
    groups = []
    for domain in samples.DOMAINS:
        texts = list(samples.iter_texts(domain, "heldout", int(heldout_mb * 1e6)))
        groups.append((domain, texts))
        if domain == "wiki":
            groups.append(("wiki_without_tables", [t for t in texts if not has_wiki_table(t)]))
    for domain, texts in groups:
        t0 = time.perf_counter()
        encoded = tok.encode_batch(texts, num_threads=threads)
        seconds = time.perf_counter() - t0
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tokens = sum(len(ids) for ids in encoded)
        rec = {
            "texts": len(texts),
            "bytes": n_bytes,
            "chars": sum(len(t) for t in texts),
            "tokens": n_tokens,
            "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
            "encode_mb_per_s": round(n_bytes / max(seconds, 1e-9) / 1e6, 1),
            "roundtrip_failures": sum(tok.decode(ids) != t for ids, t in zip(encoded, texts)),
            "special_ids_emitted": sum(any(i >= tok.n_ordinary for i in ids) for ids in encoded),
        }
        if trainer is not None:
            reference = trainer.batch_encode(texts)
            rec["rustbpe_mismatches"] = sum(list(a) != list(b) for a, b in zip(encoded, reference))
        out[domain] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))  # read when rayon starts

    from bes.config import VOCAB_SIZE
    from bes.tokenizer import samples
    from bes.tokenizer.tok import ASSET_DIR, Tokenizer, load
    from bes.tokenizer.train_bpe import N_MERGES, SPLIT_PATTERN, train_merges, write_asset, write_meta

    out_dir = Path(args.out) if args.out else ASSET_DIR
    domains = samples.DOMAINS
    needed = [f for d in domains for s in ("train", "heldout") for f in samples.files_for(d, s)]
    missing = [f for f in needed if not f.local_path.exists()]
    if missing:
        if args.no_download:
            print("missing samples:", *[str(f.local_path) for f in missing], sep="\n  ")
            return 1
        print(f"downloading {len(missing)} sample files into {samples.SAMPLE_DIR}")
        samples.download(missing)

    if args.eval_only:
        tok = load(out_dir) if args.out else load()
        print(json.dumps({"sha256": tok.sha256, "heldout": evaluate(tok, args.heldout_mb, args.threads)}, indent=2))
        return 0

    budgets = {"web": args.web_mb, "chat": args.chat_mb, "wiki": args.wiki_mb}
    stats: dict = {}
    print(f"training {N_MERGES} merges on {sum(budgets.values()):.0f} MB "
          f"({', '.join(f'{d} {mb:.0f}' for d, mb in budgets.items())})", flush=True)
    t0 = time.perf_counter()
    merges, trainer = train_merges(training_stream(budgets, args.doc_cap, stats), N_MERGES, SPLIT_PATTERN)
    train_seconds = time.perf_counter() - t0
    train_rss = _peak_rss_mb()
    if len(merges) != N_MERGES:
        print(f"only {len(merges)} merges learned; need {N_MERGES}")
        return 1
    print(f"trained in {train_seconds:.1f} s, peak RSS {train_rss:.0f} MB", flush=True)

    digest = write_asset(merges, out_dir, SPLIT_PATTERN)
    tok = Tokenizer.from_asset(out_dir)  # reload from disk: checks the hash and the format
    assert tok.sha256 == digest and tok.vocab_size == VOCAB_SIZE
    heldout = evaluate(tok, args.heldout_mb, args.threads, trainer=trainer)

    total_bytes = sum(rec["bytes"] for rec in stats.values())
    files = {d: [f.filename for f in samples.files_for(d, "train")] for d in domains}
    meta = {
        "sha256": digest,
        "vocab_size": tok.vocab_size,
        "n_bytes": 256,
        "n_merges": len(merges),
        "n_special": tok.vocab_size - tok.n_ordinary,
        "split_pattern": "cl100k_base pattern with numbers in groups of at most 2 digits",
        "training": {
            "doc_cap_chars": args.doc_cap,
            "total_bytes": total_bytes,
            "domains": {
                d: {
                    "repo_id": samples.files_for(d, "train")[0].repo_id,
                    "revision": samples.files_for(d, "train")[0].revision,
                    "files": files[d],
                    "texts": stats.get(d, {}).get("texts", 0),
                    "bytes": stats.get(d, {}).get("bytes", 0),
                    "share": round(stats.get(d, {}).get("bytes", 0) / max(total_bytes, 1), 4),
                }
                for d in domains
            },
            "wiki_rows": f"row index % {samples.WIKI_HELDOUT_EVERY} != 0",
            "seconds": round(train_seconds, 1),
            "peak_rss_mb": round(train_rss),
            "threads": int(os.environ["RAYON_NUM_THREADS"]),
            "cpu": platform.processor() or platform.machine(),
            "versions": {"python": platform.python_version(), "rustbpe": _version("rustbpe"),
                         "tiktoken": _version("tiktoken")},
        },
        "heldout": {
            "selection": {
                "web": [f.filename for f in samples.files_for("web", "heldout")],
                "chat": [f.filename for f in samples.files_for("chat", "heldout")],
                "wiki": f"row index % {samples.WIKI_HELDOUT_EVERY} == 0 of the same file",
                "wiki_without_tables": "the wiki rows without wiki-table markup lines",
            },
            "max_bytes_per_domain": int(args.heldout_mb * 1e6),
            **heldout,
        },
    }
    write_meta(out_dir, meta)
    print(json.dumps({"sha256": digest, "seconds": meta["training"]["seconds"],
                      "peak_rss_mb": meta["training"]["peak_rss_mb"],
                      "bytes_per_token": {d: r["bytes_per_token"] for d, r in heldout.items()},
                      "rustbpe_mismatches": {d: r["rustbpe_mismatches"] for d, r in heldout.items()},
                      "roundtrip_failures": {d: r["roundtrip_failures"] for d, r in heldout.items()}},
                     indent=2))
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
