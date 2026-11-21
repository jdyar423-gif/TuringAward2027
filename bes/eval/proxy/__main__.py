"""Command line of the proxy blind-test harness: ``python -m bes.eval.proxy <command> ...``.

Commands (every step is resumable; see ``docs/EVAL_PROTOCOL.md``):

    models                          registry of opponents and judges, with local availability
    fetch NAME [NAME ...]           download + verify (+ convert) registry models
    prompts [--split dev]           validate the prompt set and print its composition
    run --model M [--preset P]      generate one arm (M: falcon | mobilellm | gguf:PATH | ours
                                    | standin-<style>); prints the run id
    runs                            list finished runs
    exact RUN_ID [--split dev]      exact scores of a run (math, constraints, degenerate output)
    pairs --x RUN --y RUN --name N  blind pairs + hidden key
    judge --pairs N --judge J       automatic judge (J: registry name | gguf:PATH | standin-<rule>)
    rate --pairs N --rater ID       human rating in the terminal
    score --pairs N [--source S]    win rates, intervals, tests (Markdown + JSON report)
    compare --a N1 --b N2 --source S   paired sign test between two arms of ours
    power --delta D [--tie-rate T]  comparisons needed to detect a difference
    all --ours ... --opponent ...   the whole protocol end to end
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bes.config import DecodeCfg
from bes.eval import exact_score
from bes.eval.proxy import make_ab, opponents, prompts as prompt_set, runner, score
from bes.eval.proxy.common import GenSettings, default_root, models_dir, write_json_atomic, write_text_atomic
from bes.eval.proxy.judge import GGUFJudge, load_verdicts, ratings_path, run_judge, standin_judge
from bes.eval.proxy.run_gguf import gguf_spec


# ------------------------------------------------------------------------------- settings
def ours_presets() -> dict[str, GenSettings]:
    """Arms of our model: our decoder defaults (``bes.config.DecodeCfg``), one particle, and the
    library switched off."""
    d = DecodeCfg()
    base = opponents.common_settings().replace(
        max_tokens=d.max_new_tokens,
        extra={"particles": d.particles, "use_library": d.use_library, "use_critic": d.use_critic})
    return {"default": base,
            "k1": base.replace(extra={**base.extra_dict, "particles": 1}),
            "nolib": base.replace(extra={**base.extra_dict, "use_library": False})}


def _kv(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[k.strip()] = v.strip()
    return out


def build_spec(model: str, preset: str | None, overrides: list[str], label: str | None, seed: int,
               samples: int | None, select: str | None, n_threads: int, n_ctx: int | None,
               generator: str | None = None, generator_args: dict[str, str] | None = None,
               key: str | None = None, base: Path | None = None, download: bool = True) -> runner.RunSpec:
    """Resolve a model name and preset into a :class:`runner.RunSpec`."""
    if model in opponents.MODELS and opponents.MODELS[model].role == "opponent":
        entry = opponents.MODELS[model]
        path = opponents.fetch(model, base, download=download)
        settings_map = opponents.presets(model)
        preset = preset or "recommended"
        if preset not in settings_map:
            raise SystemExit(f"unknown preset {preset!r} for {model}; choose from {sorted(settings_map)}")
        settings = settings_map[preset]
        suffix = "-qwen3" if entry.convert else ""
        spec = gguf_spec(path, n_ctx=n_ctx or entry.n_ctx, n_threads=n_threads, key=entry.key + suffix)
    elif model.startswith("gguf:"):
        path = Path(model[5:])
        preset = preset or "common"
        settings = opponents.common_settings()
        spec = gguf_spec(path, n_ctx=n_ctx or 4096, n_threads=n_threads)
    elif model.startswith("standin-"):
        preset = preset or "common"
        settings = opponents.common_settings()
        spec = {"kind": "callable", "factory": "bes.eval.proxy.run_ours:standin",
                "kwargs": {"style": model[len("standin-"):]}, "key": model}
    elif model == "ours":
        if not generator:
            raise SystemExit("--model ours needs --generator module:factory (and usually --key)")
        presets = ours_presets()
        preset = preset or "default"
        if preset not in presets:
            raise SystemExit(f"unknown preset {preset!r} for ours; choose from {sorted(presets)}")
        settings = presets[preset]
        if key is None:
            from bes.eval.proxy.common import resolve_callable
            key = getattr(resolve_callable(generator)(**(generator_args or {})), "key", None)
            if not key:
                raise SystemExit("our generator has no 'key' attribute; pass --key")
        spec = {"kind": "callable", "factory": generator, "kwargs": dict(generator_args or {}), "key": key}
    else:
        raise SystemExit(f"unknown model {model!r}")
    settings = settings.with_overrides(overrides)
    if samples is not None:
        settings = settings.replace(samples=samples)
    if select is not None:
        settings = settings.replace(select=select)
    return runner.RunSpec(backend=spec, key=spec["key"], settings=settings,
                          label=label or (model if model == "ours" else preset), seed=seed)


# --------------------------------------------------------------------------------- reports
def exact_report(root: Path, run_id: str, split: str = "all") -> dict[str, Any]:
    prompts = {p.id: p for p in prompt_set.load()}
    records, _ = runner.load_outputs(root, run_id)
    scores = [exact_score.score_conversation(prompts[pid], [t["reply"] for t in r["turns"]],
                                             [t.get("finish_reason") for t in r["turns"]])
              for pid, r in records.items()
              if pid in prompts and (split == "all" or prompts[pid].split == split)]
    summary = exact_score.summarize(scores)
    base = root / "reports" / "exact" / f"{run_id.replace('/', '__')}-{split}"
    write_json_atomic(base.with_suffix(".json"), {"run_id": run_id, "split": split, "summary": summary,
                                                  "per_prompt": scores})
    write_text_atomic(base.with_suffix(".md"), score.exact_markdown(f"{run_id} ({split})", summary))
    return summary


def score_report(root: Path, pairset: str, source: str) -> dict[str, Any]:
    pairs, _meta = make_ab.load_pairs(root, pairset)
    key = make_ab.load_key(root, pairset)
    verdicts = load_verdicts(root, pairset, source)
    from bes.eval.proxy.common import read_jsonl
    judge_records = read_jsonl(ratings_path(root, pairset, source)) if source.startswith("judge-") else None
    summary = score.summarize(pairs, key, verdicts, judge_records)
    base = root / "reports" / "ab" / f"{pairset}--{source}"
    write_json_atomic(base.with_suffix(".json"), summary)
    write_text_atomic(base.with_suffix(".md"), score.to_markdown(summary, source))
    return summary


def sources_of(root: Path, pairset: str) -> list[str]:
    d = root / "ratings" / pairset
    return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []


def make_judge(spec: str, n_threads: int, n_ctx: int | None, base: Path | None, download: bool):
    if spec.startswith("standin-"):
        return standin_judge(spec[len("standin-"):])
    if spec.startswith("gguf:"):
        return GGUFJudge(spec[5:], n_ctx=n_ctx or 8192, n_threads=n_threads)
    if spec in opponents.MODELS and opponents.MODELS[spec].role in ("judge", "test-judge"):
        entry = opponents.MODELS[spec]
        path = opponents.fetch(spec, base, download=download)
        return GGUFJudge(path, n_ctx=n_ctx or entry.n_ctx, n_threads=n_threads, key=f"{entry.name}-{entry.sha256[:8]}")
    raise SystemExit(f"unknown judge {spec!r}")


# ------------------------------------------------------------------------------------ CLI
def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--categories", default=None, help="comma-separated subset of categories")
    p.add_argument("--limit", type=int, default=None, help="first N prompts of the split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="worker processes (0 = in this process)")
    p.add_argument("--threads", type=int, default=1, help="llama.cpp threads per worker")
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget for new generations")
    p.add_argument("--n-ctx", type=int, default=None)
    p.add_argument("--no-download", action="store_true")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bes.eval.proxy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="output directory (default data_cache/proxy)")
    ap.add_argument("--models-dir", default=None, help="GGUF directory (default data_cache/opponents)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models")
    p = sub.add_parser("fetch")
    p.add_argument("names", nargs="+")
    p = sub.add_parser("prompts")
    p.add_argument("--split", default="all", choices=["dev", "test", "all"])
    sub.add_parser("runs")

    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", default=None)
    p.add_argument("--set", action="append", default=[], help="settings override key=value")
    p.add_argument("--label", default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--select", default=None, choices=["first", "logprob_mean", "filtered_logprob_mean"])
    p.add_argument("--generator", default=None, help="module:factory of our generator")
    p.add_argument("--generator-arg", action="append", default=[])
    p.add_argument("--key", default=None, help="cache key of our generator (checkpoint + code)")
    _add_run_args(p)

    p = sub.add_parser("exact")
    p.add_argument("run_id")
    p.add_argument("--split", default="all", choices=["dev", "test", "all"])

    p = sub.add_parser("pairs")
    p.add_argument("--x", required=True, help="run id of ours")
    p.add_argument("--y", required=True, help="run id of the opponent")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])

    p = sub.add_parser("judge")
    p.add_argument("--pairs", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-ctx", type=int, default=None)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-download", action="store_true")

    p = sub.add_parser("rate")
    p.add_argument("--pairs", required=True)
    p.add_argument("--rater", required=True)

    p = sub.add_parser("score")
    p.add_argument("--pairs", required=True)
    p.add_argument("--source", default=None, help="judge-<key> or human-<rater> (default: all)")

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--source", required=True)

    p = sub.add_parser("power")
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--tie-rate", type=float, default=0.0)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser("all")
    p.add_argument("--ours", default=None, help="module:factory of our generator (or standin-<style>)")
    p.add_argument("--ours-arg", action="append", default=[])
    p.add_argument("--ours-key", default=None)
    p.add_argument("--ours-preset", default="default")
    p.add_argument("--opponent", action="append", default=None,
                   help="NAME[=PRESET], repeatable (default: falcon=recommended, mobilellm=recommended)")
    p.add_argument("--judge", default=None)
    p.add_argument("--judge-threads", type=int, default=4)
    p.add_argument("--judge-minutes", type=float, default=None)
    _add_run_args(p)

    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else default_root()
    base = Path(args.models_dir) if args.models_dir else models_dir()

    if args.cmd == "models":
        for row in opponents.describe(base=base):
            print(json.dumps(row))
    elif args.cmd == "fetch":
        for name in args.names:
            print(opponents.fetch(name, base))
    elif args.cmd == "prompts":
        ps = prompt_set.load()
        print(json.dumps(prompt_set.describe(prompt_set.select(ps, args.split)), indent=2))
    elif args.cmd == "runs":
        for r in runner.list_runs(root):
            print(r)
    elif args.cmd == "run":
        spec = build_spec(args.model, args.preset, args.set, args.label, args.seed, args.samples, args.select,
                          args.threads, args.n_ctx, args.generator, _kv(args.generator_arg), args.key, base,
                          not args.no_download)
        ps = prompt_set.load()
        todo = prompt_set.select(ps, args.split, args.categories.split(",") if args.categories else None, args.limit)
        res = runner.run(todo, spec, root, workers=args.workers, n_threads=args.threads,
                         minutes=args.minutes, all_prompts=ps)
        print(res.run_id)
        return 0 if not res.failed else 1
    elif args.cmd == "exact":
        summary = exact_report(root, args.run_id, args.split)
        print(score.exact_markdown(f"{args.run_id} ({args.split})", summary))
    elif args.cmd == "pairs":
        ps = prompt_set.select(prompt_set.load(), args.split)
        x, _ = runner.load_outputs(root, args.x)
        y, _ = runner.load_outputs(root, args.y)
        print(json.dumps(make_ab.make_pairs(ps, x, y, args.x, args.y, args.name, args.seed, root), indent=2))
    elif args.cmd == "judge":
        pairs, _ = make_ab.load_pairs(root, args.pairs)
        judge = make_judge(args.judge, args.threads, args.n_ctx, base, not args.no_download)
        print(json.dumps(run_judge(pairs, judge, args.pairs, root, args.minutes, args.limit)))
    elif args.cmd == "rate":
        from bes.eval.proxy.rate import rate_session
        pairs, _ = make_ab.load_pairs(root, args.pairs)
        rate_session(pairs, args.pairs, args.rater, root)
    elif args.cmd == "score":
        for source in [args.source] if args.source else sources_of(root, args.pairs):
            print(score.to_markdown(score_report(root, args.pairs, source), source))
    elif args.cmd == "compare":
        a = score_report(root, args.a, args.source)["per_prompt_score"]
        b = score_report(root, args.b, args.source)["per_prompt_score"]
        print(json.dumps(score.paired_sign_test(a, b), indent=2))
    elif args.cmd == "power":
        print(score.n_required(args.delta, args.alpha, args.power, args.tie_rate))
    elif args.cmd == "all":
        return run_all(args, root, base)
    return 0


def run_all(args: argparse.Namespace, root: Path, base: Path) -> int:
    """Opponents, ours, pairs, judge, reports; each step reuses what is already on disk."""
    ps = prompt_set.load()
    todo = prompt_set.select(ps, args.split, args.categories.split(",") if args.categories else None, args.limit)
    opp_list = args.opponent or ["falcon=recommended", "mobilellm=recommended"]
    lines = [f"# Proxy blind test ({args.split}, {len(todo)} prompts)", ""]
    opp_runs = []
    for item in opp_list:
        name, _, preset = item.partition("=")
        spec = build_spec(name, preset or None, [], None, args.seed, None, None, args.threads, args.n_ctx,
                          base=base, download=not args.no_download)
        res = runner.run(todo, spec, root, workers=args.workers, n_threads=args.threads,
                         minutes=args.minutes, all_prompts=ps)
        opp_runs.append((name, preset or spec.label, res.run_id))
        ex = exact_report(root, res.run_id, args.split)
        lines.append(f"- opponent `{res.run_id}`: math {ex['math']['correct']}, constraints "
                     f"{ex['constraints']['prompts_all_ok']}, degenerate {ex['degenerate']['degenerate']}, "
                     f"mean words {ex['words']['mean']}")
    if not args.ours:
        lines.append("- no generator for ours given (--ours): opponents only")
        print("\n".join(lines))
        return 0
    if args.ours.startswith("standin-"):
        ours = build_spec(args.ours, None, [], "ours", args.seed, None, None, args.threads, args.n_ctx)
    else:
        ours = build_spec("ours", args.ours_preset, [], f"ours-{args.ours_preset}", args.seed, None, None,
                          args.threads, args.n_ctx, args.ours, _kv(args.ours_arg), args.ours_key)
    res = runner.run(todo, ours, root, workers=args.workers, n_threads=args.threads, minutes=args.minutes,
                     all_prompts=ps)
    ex = exact_report(root, res.run_id, args.split)
    lines.append(f"- ours `{res.run_id}`: math {ex['math']['correct']}, constraints "
                 f"{ex['constraints']['prompts_all_ok']}, degenerate {ex['degenerate']['degenerate']}, "
                 f"mean words {ex['words']['mean']}")
    judge = make_judge(args.judge, args.judge_threads, None, base, not args.no_download) if args.judge else None
    x_records, _ = runner.load_outputs(root, res.run_id)
    lines += ["", "| pair set | source | n | score | 95% | decisive | p |", "|---|---|---|---|---|---|---|"]
    for name, preset, run_id in opp_runs:
        pairset = f"{ours.label}-vs-{name}-{preset}-{args.split}"
        y_records, _ = runner.load_outputs(root, run_id)
        make_ab.make_pairs(todo, x_records, y_records, res.run_id, run_id, pairset, args.seed, root)
        if judge is not None:
            pairs, _ = make_ab.load_pairs(root, pairset)
            run_judge(pairs, judge, pairset, root, args.judge_minutes)
        for source in sources_of(root, pairset):
            s = score_report(root, pairset, source)["overall"]
            if s["n"]:
                lines.append(f"| {pairset} | {source} | {s['n']} | {s['score']:.3f} | "
                             f"[{s['score_ci95'][0]:.3f}, {s['score_ci95'][1]:.3f}] | "
                             f"{s['win_rate_decisive']} | {s['sign_test_p']:.4f} |")
    text = "\n".join(lines) + "\n"
    write_text_atomic(root / "reports" / f"summary-{args.split}.md", text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
