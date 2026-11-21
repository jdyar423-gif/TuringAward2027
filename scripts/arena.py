"""Blind arena for the person who judges the models (``bes/eval/arena.py``).

    python scripts/arena.py --export runs/final/final                       # type prompts
    python scripts/arena.py --export runs/final/final --prompts my_prompts.txt --reveal
    python scripts/arena.py --export runs/final/final --prompts proxy:test --name test-split
    python scripts/arena.py --export DIR --opponent falcon=recommended --opponent mobilellm=neutral

Our exported model and the opponents (default ``falcon=recommended`` and
``mobilellm=recommended``, the harness's arms: GGUF files through llama.cpp with the settings of
``bes/eval/proxy/opponents.py``) answer every prompt; the replies are shown together, in a random
order under the labels A, B, C, once all are ready. Vote ``a`` / ``b`` / ``c`` (best), ``t``
(tie), ``x`` (all bad), ``s`` (skip), ``u`` (undo), ``q`` (quit). Votes and tallies with 95%
Wilson intervals are saved after every vote in ``data_cache/arena/<name>/``; running the same
command again continues the arena. ``--reveal`` shows which model wrote which reply after each
vote (votes are then final).

In a Colab cell (``input()`` works there)::

    import importlib.util
    spec = importlib.util.spec_from_file_location("bes_arena", "scripts/arena.py")
    arena = importlib.util.module_from_spec(spec); spec.loader.exec_module(arena)
    arena.main(["--export", "runs/final/final", "--reveal"])

Opponent files are downloaded and verified on first use (``python -m bes.eval.proxy fetch falcon
mobilellm`` does it ahead of time). While the 24-hour run still trains, pass ``--device cpu``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_OPPONENTS = ("falcon=recommended", "mobilellm=recommended")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=None, help="our export directory (scripts/export.py)")
    ap.add_argument("--ours", default=None,
                    help="instead of --export: a stand-in for our model (standin-plain, gguf:PATH); "
                         "rehearsals only")
    ap.add_argument("--opponent", action="append", default=None,
                    help="NAME[=PRESET] of the harness registry, gguf:PATH or standin-<style> "
                         f"(repeatable; default {' '.join(DEFAULT_OPPONENTS)})")
    ap.add_argument("--prompts", default=None,
                    help="prompt file (.txt one per line, .jsonl with 'turns' or 'prompt') or "
                         "proxy:dev / proxy:test / proxy:all; default: type them")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of the file")
    ap.add_argument("--name", default="default", help="arena name (directory under --root)")
    ap.add_argument("--root", default=None, help="default data_cache/arena")
    ap.add_argument("--reveal", action="store_true", help="show which model wrote which reply after each vote")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=None, help="our particles K (default: the run's choice)")
    ap.add_argument("--no-library", action="store_true", help="our model without the library")
    ap.add_argument("--temperature", type=float, default=None, help="our base temperature")
    ap.add_argument("--threads", type=int, default=None, help="llama.cpp threads per opponent")
    ap.add_argument("--sequential", action="store_true", help="generate one contestant after the other")
    ap.add_argument("--no-download", action="store_true", help="never download opponent files")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="cpu: run our model on the host (needed while the training run holds the TPU)")
    ap.add_argument("--show-tallies", action="store_true",
                    help="print the tallies at the end of the session (they reveal the models)")
    return ap.parse_args(argv)


def build_contestants(args: argparse.Namespace):
    from bes.eval import arena as A

    opponents = list(args.opponent or DEFAULT_OPPONENTS)
    threads = args.threads or max(1, min(8, (os.cpu_count() or 2) // max(1, len(opponents))))
    out = []
    if args.export:
        over = []
        if args.particles is not None:
            over.append(f"extra.particles={int(args.particles)}")
        if args.no_library:
            over.append("extra.use_library=false")
        if args.temperature is not None:
            over.append(f"temperature={float(args.temperature)!r}")
        out.append(A.ours_contestant(args.export, over, library=None if args.no_library else "auto"))
    elif args.ours:
        c = A.opponent_contestant(args.ours, n_threads=threads, seed=args.seed,
                                  download=not args.no_download)
        c.name = f"ours ({c.name})"
        out.append(c)
    else:
        raise SystemExit("give --export (our model) or --ours (a stand-in, rehearsals only)")
    for spec in opponents:
        out.append(A.opponent_contestant(spec, n_threads=threads, seed=args.seed,
                                         download=not args.no_download))
    return out


def main(argv: Sequence[str] | None = None, *, input_fn: Callable[[str], str] = input,
         print_fn: Callable[[str], None] = print) -> int:
    args = parse_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    from bes.eval import arena as A

    root = Path(args.root) if args.root else REPO / "data_cache" / "arena"
    contestants = build_contestants(args)
    arena = A.Arena(contestants, root / args.name, seed=args.seed, reveal=args.reveal,
                    parallel=not args.sequential, print_fn=print_fn, input_fn=input_fn)
    prompts = None
    if args.prompts:
        prompts = A.file_prompts(args.prompts)
        if args.limit is not None:
            prompts = prompts[:args.limit]
    arena.run(prompts)
    t = arena.save_tallies()
    if args.show_tallies or args.reveal:
        print_fn(A.tallies_markdown(t, f"({args.name})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
