"""Gate E9 on a stand-in: our particle decoder driving an opponent GGUF chat model on CPU.

    python scripts/e9_standin.py run [--workers 2] [--deadline-min 34] [--max-new 320]
    python scripts/e9_standin.py report [--doc docs/E9_DECODING.md]

Prompts: the dev half of ``bes/eval/proxy/prompts.jsonl`` (never the held-out half): every
dev prompt that carries a machine-checkable constraint, plus a fixed sample of single-turn dev
prompts without one (``OTHERS`` per category, the first ones in hash order).

Arms (one ``bes.decode.respond.Responder`` over ``bes.decode.gguf_lm`` each; the model's
recommended llama.cpp settings: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, no
repetition penalty; no n-gram blocking and no copy cap, so the arms differ only in what the
gate asks about):

- ``plain``: one particle, no potentials (ordinary sampling, draw for draw);
- ``filter``: six particles, no potentials, failure filter + length band + consensus;
- ``potentials``: six particles, exact potentials of the compiled constraints (structure,
  counts, reachability, budget-aware), resampling at sentence boundaries, the same selection.

Every turn is seeded with ``derive_seed(SEED, prompt id, turn, 0)`` in every arm (common random
numbers: without potentials the first particle of a six-particle arm is the plain arm's
sample). In a two-turn prompt whose first turn carries no constraint, the first reply is the
plain arm's in every arm (the arms then decode the same constrained turn after the same
conversation); statistics of length, finishing, degeneracy, fluency and cost use only the
turns an arm decoded itself.

Output (resumable, one JSON line per (arm, prompt), per worker): ``data_cache/e9_standin/
records.w<k>.jsonl``; ``report`` scores the saved replies again with the current
``bes.eval.exact_score`` and writes the table between the ``<!-- e9:table -->`` markers of
``docs/E9_DECODING.md``. A reply that used its whole token budget counts as cut by the budget
(as ``bes.decode.respond`` now reports it, finish reason ``"length"``), also in records written
before that change, whose finish reason said ``"stop"`` when the budget-aware potentials had
forced ``END`` into the last token slot. Limits: at most two worker processes, one llama.cpp
thread and one numpy thread each, and a wall-clock deadline after which no new prompt starts.

The report also states how much of the annotated constraints the constraint compiler reads
(:func:`compiler_coverage`): per prompt on the dev half (the 47 constrained prompts measured
here were also the compiler's development set, ``docs/CHANGES.md`` section 7) and, as an
aggregate count only, on the held-out half; no held-out prompt or miss is ever printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL = REPO / "data_cache" / "opponents" / "Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf"
OUT = REPO / "data_cache" / "e9_standin"
DOC = REPO / "docs" / "E9_DECODING.md"
SEED = 20261121
OTHERS = {"chat": 4, "writing": 4, "code": 3, "math": 5}
SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}
COMMON = {**SAMPLING, "no_repeat_ngram": 0, "max_copy_run": 0, "use_library": False}
ARMS = {
    "plain": {**COMMON, "particles": 1, "potentials": False, "band": False, "resample_at": "token"},
    "filter": {**COMMON, "particles": 6, "potentials": False, "band": True, "resample_at": "token"},
    "potentials": {**COMMON, "particles": 6, "potentials": True, "band": True,
                   "resample_at": "sentence"},
}
ARM_ORDER = ("plain", "filter", "potentials")
VERSION = "e9-standin-1"


# ================================================================================ prompts
def select_prompts():
    from bes.eval.proxy import prompts as P

    dev = P.select(P.load(), "dev")
    constrained = [p for p in dev if p.constraints]
    others = []
    for cat, n in OTHERS.items():
        pool = [p for p in dev if p.category == cat and not p.constraints and not p.multi_turn]
        pool.sort(key=lambda p: P._split_rank(p.id))
        others += pool[:n]
    return constrained, others


# ================================================================================ worker
class Runner:
    """One worker: the model, one GGUFLM shared by all arms, one Responder per arm."""

    def __init__(self, max_new: int):
        from bes.decode.gguf_lm import GGUFChat, GGUFLM, GGUFModel
        from bes.decode.respond import Responder
        from bes.config import DecodeCfg

        self.model = GGUFModel(MODEL)
        self.chat = GGUFChat(self.model)
        self.lm = GGUFLM(self.model, n_threads=1, max_new_tokens=max_new)
        self.max_new = max_new

        def factory(max_new_tokens: int):
            self.lm.max_new_tokens = int(max_new_tokens)
            return self.lm

        cfg = DecodeCfg(max_new_tokens=max_new)
        self.responder = Responder(factory, self.model.vocab, cfg, chat=self.chat, context=4096)

    def settings(self, arm: str) -> dict:
        return {**ARMS[arm], "max_new_tokens": self.max_new}

    def run_prompt(self, prompt, shared: dict | None) -> dict[str, dict]:
        """All arms on one prompt: ``{arm: record}``."""
        from bes.eval.exact_score import score_conversation
        from bes.eval.proxy.common import derive_seed

        out = {}
        for arm in ARM_ORDER:
            t_arm = time.perf_counter()
            msgs, turns = [], []
            for t, user in enumerate(prompt.turns):
                msgs.append({"role": "user", "content": user})
                last = t == len(prompt.turns) - 1
                reuse = arm != "plain" and not last and not prompt.constraints_for_turn(t)
                if reuse:
                    rec = dict(out["plain"]["turns"][t])
                    rec["own"] = False
                else:
                    seed = derive_seed(SEED, prompt.id, t, 0)
                    t0 = time.perf_counter()
                    reply = self.responder.reply(msgs, self.settings(arm), seed)
                    rec = {"user": user, "reply": reply["text"],
                           "finish_reason": reply["finish_reason"],
                           "completion_tokens": reply["completion_tokens"],
                           "prompt_tokens": reply["prompt_tokens"],
                           "logprob_sum": reply["logprob_sum"],
                           "logprob_count": reply["logprob_count"], "extra": reply["extra"],
                           "seconds": round(time.perf_counter() - t0, 2), "seed": seed,
                           "own": True}
                turns.append(rec)
                msgs.append({"role": "assistant", "content": rec["reply"]})
            score = score_conversation(prompt, [r["reply"] for r in turns],
                                       [r["finish_reason"] for r in turns])
            out[arm] = {"arm": arm, "prompt_id": prompt.id, "category": prompt.category,
                        "constrained": bool(prompt.constraints), "turns": turns, "score": score,
                        "seconds": round(time.perf_counter() - t_arm, 2), "version": VERSION,
                        "settings": self.settings(arm)}
        return out


def _done(out_dir: Path) -> set[tuple[str, str]]:
    from bes.eval.proxy.common import read_jsonl

    done = set()
    for path in sorted(out_dir.glob("records.w*.jsonl")):
        for r in read_jsonl(path):
            if r.get("version") == VERSION:
                done.add((r["arm"], r["prompt_id"]))
    return done


def worker(index: int, n_workers: int, deadline: float, max_new: int, out_dir: str,
           limit: int | None) -> None:
    from bes.eval.proxy.common import append_jsonl

    out = Path(out_dir)
    constrained, others = select_prompts()
    prompts = (constrained + others)[:limit] if limit else constrained + others
    done = _done(out)
    mine = [p for i, p in enumerate(prompts) if i % n_workers == index
            and not all((a, p.id) in done for a in ARM_ORDER)]
    if not mine:
        return
    runner = Runner(max_new)
    path = out / f"records.w{index}.jsonl"
    for p in mine:
        if time.time() > deadline:
            print(f"[w{index}] deadline: stopping before {p.id}", flush=True)
            break
        t0 = time.perf_counter()
        try:
            recs = runner.run_prompt(p, None)
        except Exception as err:            # keep going; the prompt is simply missing
            import traceback

            traceback.print_exc()
            append_jsonl(out / f"errors.w{index}.jsonl", [{"prompt_id": p.id, "error": repr(err),
                                                           "version": VERSION}])
            continue
        append_jsonl(path, [recs[a] for a in ARM_ORDER])
        print(f"[w{index}] {p.id} {time.perf_counter() - t0:.0f}s "
              + " ".join(f"{a}={'ok' if recs[a]['score'].get('constraints_ok') else '-'}"
                         for a in ARM_ORDER), flush=True)


def cmd_run(args) -> None:
    import multiprocessing as mp

    from bes.eval.proxy.common import append_jsonl

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 60.0 * args.deadline_min
    n = max(1, min(2, int(args.workers)))
    t0 = time.time()
    if n == 1:
        worker(0, 1, deadline, args.max_new, str(out), args.limit)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(i, n, deadline, args.max_new, str(out), args.limit))
                 for i in range(n)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    wall = time.time() - t0
    append_jsonl(out / "runs.jsonl", [{"version": VERSION, "start": round(t0, 1),
                                       "wall_s": round(wall, 1), "workers": n,
                                       "max_new": args.max_new, "deadline_min": args.deadline_min}])
    print(f"wall time {wall:.0f}s", flush=True)


# ================================================================================ report
def _load(out_dir: Path) -> dict[str, dict[str, dict]]:
    from bes.eval.proxy.common import read_jsonl

    recs: dict[str, dict[str, dict]] = {a: {} for a in ARM_ORDER}
    for path in sorted(out_dir.glob("records.w*.jsonl")):
        for r in read_jsonl(path):
            if r.get("version") == VERSION and r.get("arm") in recs:
                recs[r["arm"]][r["prompt_id"]] = r
    common = set.intersection(*(set(v) for v in recs.values())) if recs else set()
    return {a: {k: v for k, v in recs[a].items() if k in common} for a in ARM_ORDER}


def _ci(k: float, n: int) -> str:
    from bes.eval.proxy.score import wilson

    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def _quantiles(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return f"{sum(s) / len(s):.0f} / {q(0.5):.0f} / {q(0.1):.0f}-{q(0.9):.0f}"


def _mean_se(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return f"{m:.3f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{m:.3f} ± {sd / math.sqrt(len(xs)):.3f}"


#: annotated constraint type (``bes.eval.exact_score.CONSTRAINT_TYPES``) -> the
#: ``bes.exact.constraints.ConstraintSpec`` fields that read it (any one set counts)
SPEC_FIELDS = {
    "words": ("words",), "sentences": ("sentences",), "paragraphs": ("paragraphs",), "lines": ("lines",),
    "bullets": ("items", "list_style"), "numbered": ("items", "list_style"), "items": ("items",),
    "sections": ("sections",), "separated_parts": ("responses", "divider"), "include": ("keywords",),
    "exclude": ("forbidden",), "frequency": ("frequencies",), "case": ("case",), "json": ("json_output",),
    "json_keys": ("json_output",), "starts_with": ("starts_with", "first_word"), "ends_with": ("end_phrase",),
    "no_commas": ("no_commas",), "quoted": ("quoted",), "title": ("title",), "postscript": ("postscript",),
    "one_of": ("one_of",), "placeholders": ("placeholders",), "ends_with_question": ("end_question",),
    "code_block": ("code_block",), "highlights": ("highlights",),
}


def compiler_coverage(prompts) -> dict:
    """How many annotated constraints ``bes.exact.constraints.compile`` reads from the turns
    that carry them: ``{"prompts", "prompts_all_read", "constraints", "read", "missed_types"}``
    (``missed_types``: counts by annotated type). A constraint is read when the compiled spec
    of its turn sets one of the :data:`SPEC_FIELDS` of its type. Callers print only the counts
    for the held-out half."""
    from bes.exact import constraints as C

    out = {"prompts": 0, "prompts_all_read": 0, "constraints": 0, "read": 0, "missed_types": Counter()}
    for p in prompts:
        if not p.constraints:
            continue
        every = True
        for t in range(len(p.turns)):
            cons = p.constraints_for_turn(t)
            if not cons:
                continue
            spec = C.compile(p.turns[t])
            for c in cons:
                ok = any(getattr(spec, f, None) not in (None, False, (), "") for f in SPEC_FIELDS[c["type"]])
                out["constraints"] += 1
                out["read"] += ok
                if not ok:
                    out["missed_types"][c["type"]] += 1
                    every = False
        out["prompts"] += 1
        out["prompts_all_read"] += every
    out["missed_types"] = dict(sorted(out["missed_types"].items()))
    return out


def cut_by_budget(turn: dict, budget: int) -> bool:
    """The reply did not end on its own: finish reason other than ``"stop"``, or every token of
    the budget used (an ``END`` forced into the last slot by the budget-aware potentials)."""
    return turn["finish_reason"] != "stop" or int(turn["completion_tokens"]) >= int(budget)


def _rescore(recs: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Copies of the records with ``score`` recomputed by the current exact scorer, a reply cut
    by its budget scored as cut (:func:`cut_by_budget`)."""
    from bes.eval.exact_score import score_conversation
    from bes.eval.proxy import prompts as P

    prompts = {p.id: p for p in P.load()}
    out: dict[str, dict[str, dict]] = {}
    for arm, rows in recs.items():
        out[arm] = {}
        for pid, r in rows.items():
            budget = int(r["settings"]["max_new_tokens"])
            fins = ["length" if cut_by_budget(t, budget) else "stop" for t in r["turns"]]
            out[arm][pid] = {**r, "score": score_conversation(prompts[pid], [t["reply"] for t in r["turns"]], fins)}
    return out


def summarize(recs: dict[str, dict[str, dict]]) -> dict:
    from bes.decode.select import cut_off, prompt_type
    from bes.eval.exact_score import degenerate_stats
    from bes.eval.proxy import prompts as P
    from bes.eval.proxy.score import sign_test

    recs = _rescore(recs)
    out: dict = {"arms": {}, "n_prompts": len(next(iter(recs.values()))) if recs else 0}
    all_prompts = P.load()
    out["compiler"] = {"dev": compiler_coverage(P.select(all_prompts, "dev")),
                       "test": {k: v for k, v in compiler_coverage(P.select(all_prompts, "test")).items()
                                if k != "missed_types"}}
    # per-turn statistics use the turns every arm decoded itself (the plain arm also decodes
    # the shared first turns of two-turn prompts; those are left out for every arm)
    shared = {pid: [t for t in range(len(recs[ARM_ORDER[0]][pid]["turns"]))
                    if all(recs[a][pid]["turns"][t]["own"] for a in ARM_ORDER)]
              for pid in recs[ARM_ORDER[0]]}
    out["n_turns"] = sum(len(v) for v in shared.values())
    for arm in ARM_ORDER:
        rows = list(recs[arm].values())
        cons = [r for r in rows if r["constrained"]]
        ok = [bool(r["score"]["constraints_ok"]) for r in cons]
        own = [r["turns"][t] for r in rows for t in shared[r["prompt_id"]]]
        own_c = [r["turns"][t] for r in cons for t in shared[r["prompt_id"]]]
        budget = int(rows[0]["settings"]["max_new_tokens"]) if rows else 0
        unfinished = [cut_by_budget(t, budget) for t in own]
        degen = [degenerate_stats(t["reply"], "length" if c else "stop")["degenerate"]
                 for t, c in zip(own, unfinished)]
        degen_nt = [degenerate_stats(t["reply"], None)["degenerate"] for t in own]
        words = [degenerate_stats(t["reply"], None)["words"] for t in own]
        lp = [t["logprob_sum"] / t["logprob_count"] for t in own if t["logprob_count"]]
        pooled = (sum(t["logprob_sum"] for t in own) / max(1, sum(t["logprob_count"] for t in own)))
        cost = [t["extra"]["tokens_all"] for t in own]
        steps = [t["extra"]["steps"] for t in own]
        secs = [t["seconds"] for t in own]
        math_rows = [r for r in rows if r["score"].get("math_correct") is not None]
        fin = [t for t in own if t["finish_reason"] == "stop"
               and prompt_type(t["user"]) not in ("code", "short")]
        cut = [cut_off(t["reply"]) for t in fin]
        p_fin = sum(t["extra"]["n_finished"] for t in own)
        p_all = sum(t["extra"]["particles"] for t in own)
        resampled = [t["extra"]["resampled"] for t in own]
        survivors = sum(t["extra"]["n_survivors"] for t in own)
        at_budget = [t["completion_tokens"] >= budget for t in own]
        at_budget_fin = [t["completion_tokens"] >= budget and t["finish_reason"] == "stop"
                         for t in own]
        out["arms"][arm] = {
            "at_budget": sum(at_budget), "at_budget_ci": _ci(sum(at_budget), len(own)),
            "at_budget_finished": sum(at_budget_fin),
            "survivors": survivors, "survivors_ci": _ci(survivors, p_all),
            "resampled_turns": sum(1 for x in resampled if x), "resample_events": sum(resampled),
            "cut_off": sum(cut), "n_prose_finished": len(fin), "cut_off_ci": _ci(sum(cut), len(fin)),
            "particles_finished": p_fin, "particles": p_all,
            "particles_finished_ci": _ci(p_fin, p_all),
            "n_constrained": len(cons), "constraints_ok": sum(ok),
            "constraints_ci": _ci(sum(ok), len(cons)),
            "n_turns": len(own), "unfinished": sum(unfinished),
            "unfinished_ci": _ci(sum(unfinished), len(own)),
            "degenerate": sum(degen), "degenerate_ci": _ci(sum(degen), len(own)),
            "degenerate_no_trunc": sum(degen_nt),
            "degenerate_no_trunc_ci": _ci(sum(degen_nt), len(own)),
            "words": _quantiles(words),
            "words_constrained": _quantiles([degenerate_stats(t["reply"], None)["words"] for t in own_c]),
            "logprob_per_token": _mean_se(lp), "logprob_pooled": round(pooled, 4),
            "tokens_per_turn": _quantiles(cost), "tokens_mean": sum(cost) / max(1, len(cost)),
            "steps_mean": sum(steps) / max(1, len(steps)),
            "seconds_mean": sum(secs) / max(1, len(secs)),
            "math": f"{sum(bool(r['score']['math_correct']) for r in math_rows)}/{len(math_rows)}",
            "rules": dict(Counter(t["extra"]["rule"] for t in own)),
            "by_type": {},
        }
        by: dict[str, list[bool]] = {}
        for r in cons:
            for t in r["score"]["turns"]:
                for c in t.get("constraints", []):
                    by.setdefault(c["type"], []).append(bool(c["ok"]))
        out["arms"][arm]["by_type"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())}
    pairs = {}
    for a, b in (("filter", "plain"), ("potentials", "plain"), ("potentials", "filter")):
        ids = sorted(k for k, r in recs[a].items() if r["constrained"])
        pos = sum(1 for k in ids if recs[a][k]["score"]["constraints_ok"]
                  and not recs[b][k]["score"]["constraints_ok"])
        neg = sum(1 for k in ids if recs[b][k]["score"]["constraints_ok"]
                  and not recs[a][k]["score"]["constraints_ok"])
        pairs[f"{a} vs {b}"] = {"better": pos, "worse": neg, "p": round(sign_test(pos, neg), 4)}
    out["paired"] = pairs

    def lp_of(t):
        return t["logprob_sum"] / t["logprob_count"] if t["logprob_count"] else None

    for arm in ARM_ORDER[1:]:
        diffs, same = [], 0
        for pid, ts in shared.items():
            for t in ts:
                a, b = lp_of(recs[arm][pid]["turns"][t]), lp_of(recs["plain"][pid]["turns"][t])
                if a is not None and b is not None:
                    diffs.append(a - b)
                    same += recs[arm][pid]["turns"][t]["reply"] == recs["plain"][pid]["turns"][t]["reply"]
        out["arms"][arm]["logprob_vs_plain"] = _mean_se(diffs)
        out["arms"][arm]["same_reply_as_plain"] = f"{same}/{len(diffs)}"
    out["arms"]["plain"]["logprob_vs_plain"] = "0"
    out["arms"]["plain"]["same_reply_as_plain"] = "-"
    return out


def table(summary: dict) -> str:
    arms = summary["arms"]
    rows = [
        ("prompts with every constraint satisfied", "constraints_ci",
         lambda a: f"{a['constraints_ok']}/{a['n_constrained']} = {a['constraints_ci']}"),
        ("replies cut by the token budget or dead (an END forced into the last slot counts as cut)",
         "unfinished_ci", lambda a: f"{a['unfinished']}/{a['n_turns']} = {a['unfinished_ci']}"),
        ("degenerate replies (exact_score; cut replies counted)", "degenerate_ci",
         lambda a: f"{a['degenerate']}/{a['n_turns']} = {a['degenerate_ci']}"),
        ("degenerate replies, truncation not counted", "degenerate_no_trunc_ci",
         lambda a: f"{a['degenerate_no_trunc']}/{a['n_turns']} = {a['degenerate_no_trunc_ci']}"),
        ("chosen replies that used the whole token budget (of them finished)", "at_budget_ci",
         lambda a: f"{a['at_budget']}/{a['n_turns']} = {a['at_budget_ci']} ({a['at_budget_finished']})"),
        ("finished prose replies that stop mid-sentence (select.cut_off)", "cut_off_ci",
         lambda a: f"{a['cut_off']}/{a['n_prose_finished']} = {a['cut_off_ci']}"),
        ("particles that finished", "particles_finished_ci",
         lambda a: f"{a['particles_finished']}/{a['particles']} = {a['particles_finished_ci']}"),
        ("particles passing every check of the selector (rest rejected)", "survivors_ci",
         lambda a: f"{a['survivors']}/{a['particles']} = {a['survivors_ci']}"),
        ("words per reply: mean / median / p10-p90", "words", lambda a: a["words"]),
        ("words, constrained turns", "words_constrained", lambda a: a["words_constrained"]),
        ("mean log-prob per token (model, T = 1), mean ± s.e.", "logprob_per_token",
         lambda a: a["logprob_per_token"]),
        ("pooled log-prob per token", "logprob_pooled", lambda a: f"{a['logprob_pooled']:.3f}"),
        ("paired: log-prob per token minus the plain arm's, same turn (mean ± s.e.)",
         "logprob_vs_plain", lambda a: a["logprob_vs_plain"]),
        ("same reply text as the plain arm", "same_reply_as_plain",
         lambda a: a["same_reply_as_plain"]),
        ("response tokens per turn, all particles: mean / median / p10-p90", "tokens_per_turn",
         lambda a: a["tokens_per_turn"]),
        ("turns that resampled / resampling events", "resampled_turns",
         lambda a: f"{a['resampled_turns']} / {a['resample_events']}"),
        ("decoder steps per turn (mean)", "steps_mean", lambda a: f"{a['steps_mean']:.0f}"),
        ("seconds per turn on one CPU thread (mean)", "seconds_mean",
         lambda a: f"{a['seconds_mean']:.1f}"),
        ("math answers correct (sampled math prompts)", "math", lambda a: a["math"]),
        ("selection rule that fired", "rules",
         lambda a: ", ".join(f"{k} {v}" for k, v in sorted(a["rules"].items()))),
    ]
    head = "| measure | " + " | ".join(ARM_ORDER) + " |\n|---|" + "---|" * len(ARM_ORDER) + "\n"
    body = "".join(f"| {name} | " + " | ".join(fn(arms[a]) for a in ARM_ORDER) + " |\n"
                   for name, _, fn in rows)
    types = sorted({k for a in ARM_ORDER for k in arms[a]["by_type"]})
    body2 = "\n| constraint type | " + " | ".join(ARM_ORDER) + " |\n|---|" + "---|" * len(ARM_ORDER) \
        + "\n" + "".join(f"| {t} | " + " | ".join(arms[a]["by_type"].get(t, "-") for a in ARM_ORDER)
                         + " |\n" for t in types)
    pairs = "\n| paired (constrained prompts) | better | worse | sign test p |\n|---|---|---|---|\n" + \
        "".join(f"| {k} | {v['better']} | {v['worse']} | {v['p']} |\n"
                for k, v in summary["paired"].items())
    comp = ""
    if "compiler" in summary:
        d, t = summary["compiler"]["dev"], summary["compiler"]["test"]
        comp = ("\nConstraint compiler coverage (prompts whose every annotated constraint "
                "`bes.exact.constraints.compile` reads; constraints read): dev, the prompts above and the "
                f"compiler's development set, {d['prompts_all_read']}/{d['prompts']} = "
                f"{d['prompts_all_read'] / max(1, d['prompts']):.0%}, {d['read']}/{d['constraints']} "
                f"(not read: {', '.join(f'{k} {v}' for k, v in d['missed_types'].items()) or 'none'}); "
                f"held-out half, counted in aggregate only, {t['prompts_all_read']}/{t['prompts']} = "
                f"{t['prompts_all_read'] / max(1, t['prompts']):.0%}, {t['read']}/{t['constraints']}.\n")
    return head + body + body2 + pairs + comp


def cmd_report(args) -> None:
    out = Path(args.out)
    recs = _load(out)
    summary = summarize(recs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    text = table(summary)
    print(f"{summary['n_prompts']} prompts in every arm\n")
    print(text)
    doc = Path(args.doc)
    if doc.exists():
        src = doc.read_text()
        a, b = "<!-- e9:table -->", "<!-- /e9:table -->"
        if a in src and b in src:
            pre, rest = src.split(a, 1)
            _, post = rest.split(b, 1)
            doc.write_text(pre + a + "\n" + text + b + post)
            print(f"table written to {doc}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--deadline-min", type=float, default=34.0)
    r.add_argument("--max-new", type=int, default=320)
    r.add_argument("--limit", type=int, default=None, help="first N prompts only (pilot)")
    r.add_argument("--out", default=str(OUT))
    s = sub.add_parser("report")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--doc", default=str(DOC))
    args = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
