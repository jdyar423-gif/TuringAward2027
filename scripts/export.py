"""Export a run (``bes/export.py``): 16-bit weights, configuration, tokenizer, library, hashes and
a model card written from measured facts.

    python scripts/export.py --run runs/final                       # -> runs/final/final/
    python scripts/export.py --run runs/final --final-dir /content/drive/MyDrive/bes_final
    python scripts/export.py --run runs/smoke --library-mode pointer --no-download
    python scripts/export.py --checkpoint DIR --config configs/smoke.py --final-dir OUT
    python scripts/export.py --verify runs/final/final                # SHA-256 of every file

Exit status: 0 when the export verifies and the 16-bit weights passed the held-out check;
2 when the check failed or could not run (both weight files are kept: read MODEL_CARD.md);
1 on an error (nothing is replaced: an earlier export stays as it was).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory (scripts/run_24h.py --out)")
    ap.add_argument("--final-dir", default=None, help="export directory (default <run>/final)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint directory or root (default: the run's post-DPO checkpoint when "
                         "accepted, else its newest training checkpoint)")
    ap.add_argument("--config", default=None, help="configs/<name>.py (only without a run manifest)")
    ap.add_argument("--override", action="append", default=[], help="section.field=value (with --config)")
    ap.add_argument("--library", default=None, help="library directory (default: the run's library)")
    ap.add_argument("--library-mode", default="copy", choices=["copy", "pointer", "none"],
                    help="copy the library into the export, store a pointer and its digest, or none")
    ap.add_argument("--tol", type=float, default=None, help="held-out tolerance in nats (default 0.002)")
    ap.add_argument("--heldout-rows", type=int, default=16, help="held-out rows per domain")
    ap.add_argument("--data-dir", default=None, help="where held-out files live (default run.data_dir)")
    ap.add_argument("--no-download", action="store_true", help="never download held-out files")
    ap.add_argument("--verify", default=None, metavar="DIR", help="only verify an export directory")
    args = ap.parse_args(argv)

    from bes import export as EX

    if args.verify:
        problems = EX.verify_export(args.verify)
        if problems:
            print(f"{args.verify}: {len(problems)} problems", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        rec = json.loads((Path(args.verify) / EX.EXPORT_JSON).read_text())
        chk = rec.get("check") or {}
        print(f"{args.verify}: every file matches {EX.SUMS}; held-out check "
              f"{'passed' if chk.get('ok') else 'FAILED' if chk.get('ok') is False else 'not run'}"
              + (f" (|delta| {chk['abs_delta']:.2e} nats)" if isinstance(chk.get("abs_delta"), float) else ""))
        return 0 if chk.get("ok") else 2
    if not args.run and not args.checkpoint:
        ap.error("give --run (a run directory) or --checkpoint with --config")
    out = Path(args.run) if args.run else Path(args.checkpoint)
    final = args.final_dir
    if final is None and not args.run:
        ap.error("--final-dir is needed without --run")
    try:
        res = EX.export_run(out, checkpoint=args.checkpoint, config=args.config,
                            overrides=args.override or None, library=args.library,
                            final_dir=final, library_mode=args.library_mode,
                            tol=EX.TOLERANCE_NATS if args.tol is None else args.tol,
                            heldout_rows=args.heldout_rows, download_heldout=not args.no_download,
                            data_dir=args.data_dir,
                            log=lambda m: print(m, file=sys.stderr, flush=True))
    except Exception as e:  # noqa: BLE001 - the message is the report
        print(f"export FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1))
    return 0 if res.get("check_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
