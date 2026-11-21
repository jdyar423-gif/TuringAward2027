"""Run one model (one *arm*) over prompts: multi-turn conversations, candidate sampling and
selection, a resumable generation cache, a process pool and a wall-clock deadline.

Conversation protocol: for each user turn the model sees the system message (if the arm has
one), all earlier user turns and **its own** earlier visible replies. With ``samples = K``,
``K`` candidates are drawn per turn with seeds ``derive_seed(seed, prompt_id, turn, k)`` and
one is kept (``bes.eval.proxy.common.select_candidate``); the kept reply is what the next turn
sees and what the blind comparison shows.

On disk (``root`` defaults to ``data_cache/proxy``):

- ``cache/<backend_key>.jsonl``: one line per generation, keyed by a hash of (backend key,
  messages, generation settings, seed, harness version). Appended with fsync by the parent
  process only, so an interrupted run loses at most the prompts in flight.
- ``outputs/<backend_key>/<label>-<settings_hash>-s<seed>.jsonl``: the assembled conversations
  of every prompt that is complete in the cache (rewritten atomically after each run), with a
  ``.meta.json`` next to it. This is what ``make_ab`` and the exact scorer read.

A run identifier is ``<backend_key>/<label>-<settings_hash>-s<seed>``.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bes.eval.proxy.common import (HARNESS_VERSION, GenSettings, Reply, append_jsonl,
                                   default_root, derive_seed, make_backend, read_json,
                                   read_jsonl, safe_name, select_candidate, stable_hash,
                                   write_json_atomic, write_jsonl_atomic)
from bes.eval.proxy.prompts import Prompt


@dataclass(frozen=True)
class RunSpec:
    """One arm: a backend (picklable spec + cache key), its settings, a label and a seed."""

    backend: Mapping[str, Any]
    key: str
    settings: GenSettings
    label: str
    seed: int = 0

    @property
    def run_id(self) -> str:
        return f"{self.key}/{safe_name(self.label)}-{self.settings.hash()}-s{self.seed}"


def gen_key(backend_key: str, messages: Sequence[Mapping[str, str]], settings: GenSettings,
            seed: int) -> str:
    """Cache key of one generation."""
    return stable_hash({"v": HARNESS_VERSION, "backend": backend_key,
                        "messages": [dict(m) for m in messages], "settings": settings.core_dict(),
                        "seed": int(seed)}, 24)


def cache_path(root: Path, backend_key: str) -> Path:
    return Path(root) / "cache" / f"{safe_name(backend_key)}.jsonl"


def outputs_path(root: Path, spec: RunSpec) -> Path:
    return Path(root) / "outputs" / f"{spec.run_id}.jsonl"


def load_cache(root: Path, backend_key: str) -> dict[str, dict[str, Any]]:
    """``gen_key -> cache line`` (later lines win)."""
    return {r["key"]: r for r in read_jsonl(cache_path(root, backend_key)) if "key" in r and "reply" in r}


def _is_degenerate(reply: Reply) -> bool:
    from bes.eval.exact_score import degenerate_stats
    return bool(degenerate_stats(reply.text, reply.finish_reason)["degenerate"])


class _OutOfTime(Exception):
    pass


class _NotCached(Exception):
    pass


def converse(prompt: Prompt, settings: GenSettings, seed: int, backend_key: str,
             generate: Callable[[list[dict[str, str]], int], Reply],
             cached: Mapping[str, Mapping[str, Any]],
             on_new: Callable[[dict[str, Any]], None] | None = None,
             deadline: float | None = None) -> dict[str, Any]:
    """Run (or replay from ``cached``) one conversation; returns the output record.

    ``generate(messages, seed)`` produces one reply; every new generation is reported to
    ``on_new`` as a cache line. Raises ``_OutOfTime`` when ``deadline`` passes before a new
    generation would start.
    """
    messages: list[dict[str, str]] = []
    if settings.system_prompt is not None:
        messages.append({"role": "system", "content": settings.system_prompt})
    turns = []
    for t, user in enumerate(prompt.turns):
        messages.append({"role": "user", "content": user})
        cands: list[Reply] = []
        keys = []
        for k in range(settings.samples):
            s = derive_seed(seed, prompt.id, t, k)
            key = gen_key(backend_key, messages, settings, s)
            keys.append(key)
            if key in cached:
                cands.append(Reply.from_dict(cached[key]["reply"]))
                continue
            if deadline is not None and time.time() > deadline:
                raise _OutOfTime()
            reply = generate([dict(m) for m in messages], s)
            cands.append(reply)
            if on_new is not None:
                on_new({"key": key, "backend": backend_key, "prompt_id": prompt.id,
                        "prompt_hash": prompt.content_hash, "turn": t, "sample": k, "seed": s,
                        "settings_hash": stable_hash(settings.core_dict(), 10),
                        "reply": reply.to_dict(), "time": round(time.time(), 1)})
        idx = select_candidate(cands, settings.select, _is_degenerate)
        chosen = cands[idx]
        turns.append({"user": user, "reply": chosen.text, "finish_reason": chosen.finish_reason,
                      "selected": idx, "n_candidates": len(cands),
                      "prompt_tokens": chosen.prompt_tokens,
                      "completion_tokens": chosen.completion_tokens,
                      "latency_s": round(sum(c.latency_s for c in cands), 3),
                      "logprob_mean": chosen.logprob_mean, "gen_key": keys[idx],
                      "extra": chosen.extra})
        messages.append({"role": "assistant", "content": chosen.text})
    return {"prompt_id": prompt.id, "prompt_hash": prompt.content_hash, "category": prompt.category,
            "split": prompt.split, "multi_turn": prompt.multi_turn, "turns": turns}


# ------------------------------------------------------------------------- worker process
_WORKER: dict[str, Any] = {}


def _init_worker(backend_spec: Mapping[str, Any], n_threads: int | None) -> None:
    if n_threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(n_threads)
    _WORKER["backend"] = make_backend(backend_spec)


def _work(task: Mapping[str, Any]) -> dict[str, Any]:
    backend = _WORKER["backend"]
    prompt = Prompt(**task["prompt"])
    settings = GenSettings.from_dict(task["settings"])
    new: list[dict[str, Any]] = []
    out: dict[str, Any] = {"prompt_id": prompt.id, "new": new, "record": None, "error": None}
    try:
        out["record"] = converse(prompt, settings, task["seed"], task["backend_key"],
                                 lambda msgs, s: backend.generate(msgs, settings, s),
                                 task["cached"], new.append, task["deadline"])
    except _OutOfTime:
        out["error"] = "deadline"
    except Exception as err:  # keep the pool alive; the prompt stays incomplete
        out["error"] = f"{type(err).__name__}: {err}"
    return out


def _prompt_task(prompt: Prompt) -> dict[str, Any]:
    return {"id": prompt.id, "category": prompt.category, "turns": prompt.turns,
            "answer": prompt.answer, "tol": prompt.tol, "constraints": prompt.constraints,
            "notes": prompt.notes, "split": prompt.split}


# ------------------------------------------------------------------------------------ run
@dataclass
class RunResult:
    run_id: str
    outputs: Path
    done: int = 0
    reused: int = 0
    new_generations: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    out_of_time: int = 0
    seconds: float = 0.0


def run(prompts: Sequence[Prompt], spec: RunSpec, root: Path | None = None, workers: int = 0,
        n_threads: int | None = None, minutes: float | None = None,
        all_prompts: Sequence[Prompt] | None = None, log: Callable[[str], None] | None = print,
        backend: Any = None) -> RunResult:
    """Generate every missing conversation of ``prompts`` for ``spec`` and rewrite its outputs
    file (covering every prompt of ``all_prompts`` that is complete in the cache).

    ``workers = 0`` runs in this process (``backend`` may be passed in to reuse a loaded model);
    ``workers >= 1`` starts that many spawned processes, each building its own backend from
    ``spec.backend``. ``minutes`` is a wall-clock budget: no new generation starts after it,
    and work in flight is kept in the cache so a rerun resumes.
    """
    root = Path(root) if root is not None else default_root()
    t0 = time.time()
    deadline = t0 + 60.0 * minutes if minutes else None
    log = log or (lambda _msg: None)
    cpath = cache_path(root, spec.key)
    cached = load_cache(root, spec.key)
    by_prompt: dict[str, dict[str, Any]] = {}
    for key, line in cached.items():
        by_prompt.setdefault(line.get("prompt_id", ""), {})[key] = line
    result = RunResult(spec.run_id, outputs_path(root, spec))
    records: dict[str, dict[str, Any]] = {}

    def finish(out: Mapping[str, Any]) -> None:
        if out["new"]:
            append_jsonl(cpath, out["new"])
            for line in out["new"]:
                cached[line["key"]] = line
                by_prompt.setdefault(line["prompt_id"], {})[line["key"]] = line
            result.new_generations += len(out["new"])
        if out["record"] is not None:
            records[out["prompt_id"]] = out["record"]
            result.done += 1
            if not out["new"]:
                result.reused += 1
        elif out["error"] == "deadline":
            result.out_of_time += 1
        else:
            result.failed[out["prompt_id"]] = str(out["error"])
        n = result.done + result.out_of_time + len(result.failed)
        if n % 10 == 0 or n == len(prompts):
            elapsed = time.time() - t0
            log(f"[{spec.run_id}] {n}/{len(prompts)} prompts, {result.new_generations} new "
                f"generations, {len(result.failed)} failed, {elapsed / 60:.1f} min")

    tasks = [{"prompt": _prompt_task(p), "settings": spec.settings.as_dict(), "seed": spec.seed,
              "backend_key": spec.key, "cached": by_prompt.get(p.id, {}), "deadline": deadline}
             for p in prompts]
    if workers <= 0:
        if backend is None:
            backend = make_backend(spec.backend)
        _WORKER["backend"] = backend
        try:
            for task in tasks:
                finish(_work(task))
        finally:
            _WORKER.pop("backend", None)
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                    initargs=(dict(spec.backend), n_threads)) as pool:
            futures = [pool.submit(_work, task) for task in tasks]
            try:
                for fut in cf.as_completed(futures):
                    finish(fut.result())
            except KeyboardInterrupt:
                for fut in futures:
                    fut.cancel()
                raise
    result.seconds = time.time() - t0
    assemble(spec, root, all_prompts if all_prompts is not None else prompts, cached)
    log(f"[{spec.run_id}] finished: {result.done} complete ({result.reused} from cache), "
        f"{result.new_generations} new generations, {result.out_of_time} not started before the "
        f"deadline, {len(result.failed)} failed, {result.seconds / 60:.1f} min")
    for pid, err in list(result.failed.items())[:5]:
        print(f"  failed {pid}: {err}", file=sys.stderr)
    return result


def assemble(spec: RunSpec, root: Path, prompts: Sequence[Prompt],
             cached: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Rewrite the outputs file of ``spec`` from the cache (complete conversations only)."""
    root = Path(root)
    cached = cached if cached is not None else load_cache(root, spec.key)

    def missing(_msgs: list[dict[str, str]], _seed: int) -> Reply:
        raise _NotCached()

    records = []
    for p in prompts:
        try:
            records.append(converse(p, spec.settings, spec.seed, spec.key, missing, cached))
        except _NotCached:
            continue
    path = outputs_path(root, spec)
    write_jsonl_atomic(path, records)
    write_json_atomic(path.with_suffix(".meta.json"), {
        "run_id": spec.run_id, "backend_key": spec.key, "label": spec.label, "seed": spec.seed,
        "settings": spec.settings.as_dict(), "settings_hash": spec.settings.hash(),
        "backend": {k: (stable_hash(v, 8) if k == "chat_template" else v) for k, v in spec.backend.items()},
        "harness_version": HARNESS_VERSION, "n_prompts": len(records),
        "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    return path


def load_outputs(root: Path, run_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``(prompt_id -> record, meta)`` of a finished run."""
    path = Path(root) / "outputs" / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no outputs for run {run_id!r} ({path})")
    meta_path = path.with_suffix(".meta.json")
    meta = read_json(meta_path) if meta_path.exists() else {}
    return {r["prompt_id"]: r for r in read_jsonl(path)}, meta


def list_runs(root: Path | None = None) -> list[str]:
    root = Path(root) if root is not None else default_root()
    base = root / "outputs"
    if not base.exists():
        return []
    return sorted(str(p.relative_to(base).with_suffix("")) for p in base.glob("*/*.jsonl"))
