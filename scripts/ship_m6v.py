"""From the chip's export to the model that ships (``docs/PLAN_M6V.md``, proposition 5).

    BES_PROFILE=m6 python scripts/ship_m6v.py --export runs/m6v/final --curve            # the price of rows
    BES_PROFILE=m6 python scripts/ship_m6v.py --export runs/m6v/final --keep 5120 \\
        --library data_cache/library/answers_v1 --credit credit.npy --out runs/m6v/shipped

The chip trains and exports the whole ``configs/v9.py`` model (98.04M stored numbers, no
library). What ships shares the dormant numbers between the page and gain rows of the tokens
that keep them and written entries:

1. **token counts** (``--counts FILE.npy``, written when missing): how often each vocabulary id
   occurs in the questions and answers of the full answers library, the text the shipped
   model is asked to produce. The role, end and calculator tokens of the chat format are
   always kept (they are in every conversation and in no stored answer);
2. **the curve** (``--curve``): the held-out loss of the export with the rows of every token
   of rank >= K set to zero, for a list of K: what an eviction costs before anything adapts;
3. **the library that fits**: the stored numbers left under the cap after the kept rows, the
   row map and the overlay's capacity go to the best-credited entries of the full library
   (``scripts/subset_answers.py``), with ``--read-cap`` as the most one reply reads;
4. **the shipped export** (``bes.evict.evict_export``): rows evicted, the library attached, the
   overlay reserved, the ledger recounted and checked against the caps, the held-out loss
   before and after recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def special_ids(vocab: int) -> list[int]:
    from bes.config import N_SPECIAL

    return list(range(vocab - N_SPECIAL, vocab))


def token_counts(library: str, tok, vocab: int, log=print) -> np.ndarray:
    """``int64[V]``: occurrences of every id in the questions and answers of ``library``."""
    from bes.library.answers import AnswerLibrary

    lib = AnswerLibrary.load(library)
    counts = np.zeros(vocab, np.int64)
    t0 = time.time()
    for e in range(lib.n_entries):
        q, a = lib.entry(e)
        counts += np.bincount(np.asarray(tok.encode(q) + tok.encode(a), np.int64), minlength=vocab)
        if e % 50000 == 49999:
            log(f"counts: {e + 1:,} of {lib.n_entries:,} entries ({time.time() - t0:.0f} s)")
    return counts


def curve(export: Path, counts: np.ndarray, ks, rows: int = 8, log=print) -> list[dict]:
    """Held-out loss of the export with the rows of rank >= K evicted, for every K of ``ks``."""
    from bes import evict
    from bes import export as E
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    rec = json.loads((export / E.EXPORT_JSON).read_text())
    cfg = E.load_export_config(export / "config.py")
    tok = load_tokenizer(export / "tokenizer")
    params = E.load_weights(export, cfg.model)
    consts = M.make_consts(E.check_model_cfg(cfg.model), **((rec.get("consts") or {}).get("enable") or {}))
    batches, _ = E.heldout_batches(cfg, tok, rows=rows, seq_len=1024, domains=("web", "chat"), download=True)
    v = cfg.model.vocab_size
    width = sum(int(np.prod(np.shape(x)[1:])) for p, x in zip(*E.flat_params(params)) if p in evict.ROW_TABLES)
    share = counts / max(1, counts.sum())
    out = []
    for k in ks:
        keep = np.ones(v, bool) if k is None else evict.keep_mask(counts, k, always=special_ids(v))
        r = E.heldout_loss(params if k is None else evict.mask_params(params, keep), consts, batches, cfg.model, chunk=4)
        row = {"keep": int(keep.sum()), "rank": k, "loss": r["loss"],
               "by_domain": {d: x["loss"] for d, x in r["by_domain"].items()},
               "occurrences_evicted": float(share[~keep].sum()), "numbers_freed": int((~keep).sum()) * width}
        if out:
            row["cost"] = row["loss"] - out[0]["loss"]
            row["per_occurrence"] = row["cost"] / row["occurrences_evicted"] if row["occurrences_evicted"] else None
        out.append(row)
        log(f"curve: keep {row['keep']:5d}  loss {row['loss']:.4f}  " +
            "  ".join(f"{d} {x:.4f}" for d, x in row["by_domain"].items()) +
            (f"  cost {row['cost']:+.4f}  occurrences evicted {row['occurrences_evicted']:.4f}  "
             f"freed {row['numbers_freed'] / 1e6:.1f}M" if "cost" in row else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--library", default=str(REPO / "data_cache" / "library" / "answers_v1"))
    ap.add_argument("--counts", default=None, help=".npy of token counts (computed and written when missing)")
    ap.add_argument("--curve", action="store_true", help="print the price of evicting rows and stop")
    ap.add_argument("--ks", default="7168,6144,5120,4096,3072,2048")
    ap.add_argument("--keep", type=int, default=None, help="tokens that keep their rows (by rank)")
    ap.add_argument("--credit", default=None, help=".npy of the library entries' credit (subset_answers.py)")
    ap.add_argument("--read-cap", type=int, default=360_000)
    ap.add_argument("--overlay", type=int, default=60_000, help="numbers reserved for taught entries")
    ap.add_argument("--total-cap", type=int, default=100_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subset-out", default=None, help="keep the subset library here (default: a temporary directory)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    from bes import budget, evict
    from bes import export as E
    from bes.tokenizer import load as load_tokenizer

    export = Path(a.export)
    cfg = E.load_export_config(export / "config.py")
    v = cfg.model.vocab_size
    tok = load_tokenizer(export / "tokenizer")
    if a.counts and Path(a.counts).is_file():
        counts = np.load(a.counts)
    else:
        counts = token_counts(a.library, tok, v)
        if a.counts:
            np.save(a.counts, counts)
    if counts.shape != (v,):
        raise SystemExit(f"token counts of {counts.shape[0]} ids for a vocabulary of {v}")
    if a.curve:
        rows = curve(export, counts, [None] + [int(k) for k in a.ks.split(",") if k])
        print(json.dumps(rows, indent=1))
        return 0
    if a.keep is None or a.out is None:
        raise SystemExit("--keep K and --out DIR are needed (or --curve)")
    keep = evict.keep_mask(counts, a.keep, always=special_ids(v))
    kept = int(keep.sum())
    core = evict.accounting(cfg.model, kept)                    # compact weights and the map
    room = a.total_cap - core["total"] - int(a.overlay)
    print(f"ship: {kept} of {v} tokens keep their rows; weights and map {core['total']:,}, overlay "
          f"{a.overlay:,}, left for the library {room:,} (cap {a.total_cap:,})")
    if room < 1_500_000:
        raise SystemExit("too little is left for a library: keep fewer rows")
    import subset_answers as SA

    cr = np.load(a.credit) if a.credit and Path(a.credit).is_file() else None
    lib, sub = SA.subset(a.library, room, read_cap=a.read_cap, cr=cr, workers=a.workers)
    holder = None
    if a.subset_out:
        sub_dir = Path(a.subset_out)
    else:
        holder = tempfile.TemporaryDirectory()
        sub_dir = Path(holder.name) / "answers"
    lib.save(str(sub_dir))
    from build_answers import attribution

    stats = {"sources": {k: s for k, s in (lib.meta.get("build") or {}).get("sources", {}).items() if s.get("stored")}}
    (sub_dir / "ATTRIBUTION.md").write_text(
        attribution(stats) + f"\nThis library is a subset of a larger build ({sub['parent_entries']:,} entries): "
        f"the {sub['kept']:,} entries kept were chosen by the rule recorded in `answers.json` (`build.subset`).\n")
    batches, _ = E.heldout_batches(cfg, tok, rows=16, seq_len=1024, domains=("web", "chat"), download=True)
    share = counts / max(1, counts.sum())
    rule = (f"the {a.keep} most frequent ids in the questions and answers of the full answers library, and the "
            f"{len(special_ids(v))} special tokens of the chat format; the evicted ids are {share[~keep].sum():.2%} "
            "of those occurrences")
    rec = evict.evict_export(export, a.out, keep, library=sub_dir, rule=rule, heldout=batches,
                             overlay_numbers=a.overlay, log=print)
    acc = rec["accounting"]
    print(json.dumps({"total": acc["total"], "active_step": acc["active_step"], "within_caps": acc["within_caps"],
                      "rows_kept": acc["rows_kept"], "library_entries": sub["kept"], "library_numbers": sub["numbers"],
                      "heldout": rec["eviction"]["heldout"]}, indent=1))
    if acc["total"] > a.total_cap or acc["active_step"] > budget.ACTIVE_CAP:
        raise SystemExit("THE SHIPPED MODEL IS OVER A CAP")
    if holder is not None:
        holder.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
