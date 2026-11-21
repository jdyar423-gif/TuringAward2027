"""Gates E4 (step-function probes) and E5 (self-generation) of ``docs/PLAN.md`` section 7.

E4 trains four micro arms with about 30 % synthetic probe conversations in the training rows
(``bes.eval.probes``), one child process at a time (resumable: finished arms and finished
probe evaluations are skipped), then decodes the probes greedily with each arm and, on the
reference arm, with the evaluation-time switch of each mechanism::

    nohup python scripts/run_probes.py --tokens 1000000 --out runs/e4 > runs/e4/queue.log 2>&1 &
    python scripts/run_probes.py --report --out runs/e4               # writes docs/E4_PROBES.md

E5 samples continuations of held-out chat and web prompts from the trained E2 reference arm
(and the E2 ``no_tape`` arm when it exists) at temperatures 0.5, 0.7 and 0.8 with the tape off
(``tape_off``), with the tape on and the whole output law tempered (the earlier sampler,
self-evidence 1: ``mix_w1``), and with the tape on and the temperature on the network's
distribution only (what the decoder does) at self-evidence scale ``src_5 = w`` for w in
{1, 0.5, 0.25, 0} (``net_w<w>``)::

    nohup python scripts/run_probes.py --e5 --from runs/e2 --out runs/e5 > runs/e5/e5.log 2>&1 &
    python scripts/run_probes.py --e5 --report --out runs/e5          # writes docs/E5_SELF_GENERATION.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _limit_threads(argv: list[str]) -> None:
    """Before numpy / jax load (same rule as ``micro_ablate._limit_threads``)."""
    n, cpus = 2, None
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a == "--cpus" and i + 1 < len(argv) and argv[i + 1]:
            cpus = [int(c) for c in argv[i + 1].split(",")]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        chosen = [c for c in (cpus or allowed[-n:]) if c in allowed] or allowed[:n]
        os.sched_setaffinity(0, chosen)


if __name__ == "__main__":
    _limit_threads(sys.argv)
sys.path.insert(0, str(REPO / "scripts"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import micro_ablate as MA  # noqa: E402  (bench options, arm selection, child launching)
from bes.eval import microbench as mb  # noqa: E402
from bes.eval.microbench import Arm  # noqa: E402

PROBE_MIX = (("web", 0.525), ("chat", 0.175), ("probe_calc", 0.30))
PLAIN_MIX = (("web", 0.525), ("chat", 0.175), ("probe_plain", 0.30))
E4_ARMS: tuple[Arm, ...] = (
    Arm("probe_ref", kind="probe", mixture=PROBE_MIX, component="every mechanism on"),
    Arm("probe_no_tape", ("model.use_cand_in=false", "model.use_cand_out=false"), kind="probe",
        mixture=PROBE_MIX, component="trained without the candidate tape (copy off)"),
    Arm("probe_no_reg", ("model.use_reg=false",), kind="probe", mixture=PROBE_MIX,
        component="trained without register embeddings (count and fence off; clocks kept)"),
    Arm("probe_no_calc", kind="probe", mixture=PLAIN_MIX,
        component="arithmetic answers in plain digits, no calculator span (arith off)"),
)
#: probe -> (mechanism, arm trained with it, arm trained without it, evaluation-time view)
PLAN = {
    "copy": ("candidate tape", "probe_ref", "probe_no_tape", "tape_off"),
    "count": ("countdown registers", "probe_ref", "probe_no_reg", "reg_off"),
    "fence": ("registers (fence state)", "probe_ref", "probe_no_reg", "reg_off"),
    "arith": ("calculator span, forced result", "probe_ref", "probe_no_calc", "no_forcing"),
}
#: evaluation-time views of the reference: name -> (make_consts switches, force_calc, probes)
VIEWS = {
    "tape_off": ({"cand_in": 0.0, "cand_out": 0.0}, True, ("copy",)),
    "reg_off": ({"reg": 0.0}, True, ("count", "fence")),
    "no_forcing": ({}, False, ("arith",)),
}
PROBES_FILE = "probes.jsonl"
E5_FILE = "e5.jsonl"
E5_TEMPS = (0.5, 0.7, 0.8)
E5_W = (1.0, 0.5, 0.25, 0.0)
E5_TOL = 0.005
#: version of the E5 settings: 2 = tape off / mixture tempering / network-only tempering at
#: every self-evidence scale (records of other versions are ignored)
E5_VERSION = 2
#: setting names in report order (``no_tape_arm``: the model trained without the tape)
E5_ORDER = ("mix_w1",) + tuple(f"net_w{w:g}" for w in E5_W) + ("tape_off", "no_tape_arm")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


# ================================================================================ E4
def eval_arm(args, name: str) -> None:
    """Child mode: decode every probe with arm ``name`` (and the reference's views)."""
    from bes.eval import probes as P
    from bes.model import model as M
    from bes.tokenizer import load as load_tok

    out = _abs(args.out)
    tok = load_tok()
    rec, cfg, params = mb.load_arm(out, name)
    done = {(r["view"], r["probe"]) for r in mb.read_jsonl(out / PROBES_FILE)
            if r.get("spec_hash") == rec["spec_hash"] and r.get("n_test") == args.n_test}
    train_sw = dict(tuple(s) for s in rec["arm"]["switches"])
    jobs = [("default", {}, True, P.PROBES)]
    if name == "probe_ref":
        jobs += [(v, sw, force, probes) for v, (sw, force, probes) in VIEWS.items()]
    forcer_name = P.make_forcer(1, 8, tok)[1]
    for view, sw, force, probes in jobs:
        consts = M.make_consts(cfg.model, **{**train_sw, **sw})
        for probe in probes:
            if (view, probe) in done:
                log(f"[{name}] {view}/{probe} done earlier")
                continue
            items = P.eval_items(probe, args.n_test, tok, calc=True)
            t0 = time.time()
            res = P.run_probe(params, consts, cfg.model, items, tok, force_calc=force, batch=args.batch)
            summ = P.probe_summary(res)
            mb.append_jsonl(out / PROBES_FILE, {
                "type": "probe", "arm": name, "spec_hash": rec["spec_hash"], "view": view,
                "probe": probe, "n_test": args.n_test, "force_calc": force, "switches": sw,
                "summary": summ, "decoder": forcer_name if force else "none",
                "examples": [{"ok": r["ok"], "text": r["text"][:300], "meta": r["meta"]} for r in res[:6]],
                "failures": [{"text": r["text"][:300], "meta": r["meta"]} for r in res if not r["ok"]][:6],
                "seconds": time.time() - t0, "tokens": rec["tokens"]})
            log(f"[{name}] {view}/{probe}: {summ['correct']}/{summ['n']} = {summ['acc']:.1%} "
                f"({time.time() - t0:.0f} s)")


def e4_report(out: Path, arms=E4_ARMS) -> str:
    from bes.eval import probes as P

    recs = mb.read_jsonl(out / mb.RESULTS)
    done = mb.latest_done(recs)
    prs = mb.read_jsonl(out / PROBES_FILE)
    latest: dict = {}
    for r in prs:
        arm = done.get(r["arm"])
        if arm is not None and r.get("spec_hash") == arm["spec_hash"]:
            latest[(r["arm"], r["view"], r["probe"])] = r
    head = ["# E4 — step-function probes (gate E4 of `docs/PLAN.md` section 7)", "",
            f"Generated by `python scripts/run_probes.py --report` from `{mb._rel(out)}`; do not "
            "edit by hand.", "",
            "## Rule (pre-registered)", "",
            "Each exact mechanism must turn a task the plain network cannot do reliably at this size "
            "into one it does: exact-match accuracy **> 90 % with the mechanism and < 50 % without** "
            "(point estimates; 95 % Wilson intervals shown). The primary comparison is between "
            "models trained with and without the mechanism (`trained` columns); the evaluation-time "
            "switch on the trained reference is a second view (`switch`). Consequence of a fail "
            "(plan): the mechanism is switched off.", "",
            "Tasks (`bes/eval/probes.py`): **copy** a string of 20 random single-token words given "
            "earlier in the conversation; **count** exactly N bullet items or sentences, N = 1..8 "
            "(checked with `constraints.verify` of the compiled user turn); **fence** close every "
            "code fence before ending (one or two blocks, half without a language tag); **arith** "
            "`a op b` with 2-4 digit operands and op in + - *, answered through a calculator span "
            "whose result the host forces, versus the model's own digits. About 30 % of the "
            "training rows are probe conversations (the four tasks in equal numbers), the rest "
            "web 52.5 % and chat 17.5 %. Greedy decoding with the incremental exact features.", ""]
    if not latest:
        return "\n".join(head + ["## Status", "", "**Not run yet.** Launch `scripts/run_probes.py "
                                 "--tokens N --out runs/e4`, then regenerate this file with `--report`.",
                                 "", "## Arms", "", mb._arm_list(arms)]) + "\n"

    def cell(key):
        r = latest.get(key)
        if r is None:
            return None, "n/a"
        s = r["summary"]
        return s, f"{s['acc']:.1%} [{s['ci'][0]:.0%}, {s['ci'][1]:.0%}] ({s['correct']}/{s['n']})"

    ref = done.get("probe_ref")
    lines = head + ["## Setup", ""]
    if ref:
        lines += [f"- {ref['tokens']:,} training tokens per arm, shape {tuple(ref['shape'])}, "
                  f"model `{ref['bench']['config']}` ({ref['n_params']:,} parameters); realised "
                  "target-token shares " + ", ".join(f"{d} {s:.1%}" for d, s in ref['data']['token_shares'].items()) + ".", ""]
    rows, verdicts = [], {}
    for probe, (mech, on_arm, off_arm, view) in PLAN.items():
        on_s, on_c = cell((on_arm, "default", probe))
        off_s, off_c = cell((off_arm, "default", probe))
        sw_s, sw_c = cell((on_arm, view, probe))
        v1 = P.probe_verdict(on_s, off_s)
        v2 = P.probe_verdict(on_s, sw_s)
        verdicts[probe] = v1
        rows.append([f"**{probe}**", mech, on_c, off_c, f"**{v1}**", f"{sw_c} (`{view}`)", v2])
    lines += ["## Results", "",
              mb._md_table(["probe", "mechanism", "on (trained, `probe_ref`)", "off (trained without)",
                            "verdict (trained)", "off (switch on `probe_ref`)", "verdict (switch)"], rows), ""]
    # every arm on every probe
    arows = []
    for a in arms:
        cells = [f"`{a.name}`"]
        for probe in P.PROBES:
            cells.append(cell((a.name, "default", probe))[1])
        arows.append(cells)
    lines += ["## Every arm on every probe (default decoding)", "",
              mb._md_table(["arm"] + list(P.PROBES), arows), ""]
    # count by N
    for arm_name in ("probe_ref", "probe_no_reg"):
        r = latest.get((arm_name, "default", "count"))
        if r:
            by = r["summary"]["by_n"]
            lines += [f"### count by N (`{arm_name}`)", "",
                      mb._md_table(["N"] + list(by), [["correct"] + [f"{c}/{t}" for c, t in by.values()]]), ""]
    # examples of failures
    lines += ["## Failure examples", ""]
    for key, r in sorted(latest.items()):
        if r.get("failures"):
            ex = r["failures"][0]
            lines.append(f"- `{key[0]}` {key[1]}/{key[2]}: `{json.dumps(ex['text'])[:200]}` {ex['meta']}")
    lines += ["", "## Arms", "", mb._arm_list(arms), "",
              "## Notes", "",
              "- The `no_forcing` view decodes the reference without host forcing: the model never "
              "trained on the result tokens (weight 0 under the loss rule), so this view shows what "
              "forcing replaces, not what a model trained on its own digits does (`probe_no_calc`).",
              "- `probe_no_reg` keeps the structural clocks (line and sentence indices as rotary "
              "positions); only the register embeddings, including the countdowns, are removed.",
              "- Micro scale (d=128, a few hundred steps): a mechanism that passes here is a step "
              "function at this size; the plain network of the final model is larger and trains "
              "far longer, so the `off` accuracies there will be higher."]
    return "\n".join(lines) + "\n"


# ================================================================================ E5
def e5_settings(cfg, train_sw: dict) -> list[tuple[str, dict, str]]:
    """``(name, make_consts switches, temper)`` of every E5 setting of the reference arm
    (``temper`` as in ``bes.eval.probes.decode_batch``): the tape off; the tape on with the
    whole mixture tempered at self-evidence 1 (the earlier sampler); the tape on with the
    temperature on the network only at every self-evidence scale of ``E5_W``."""
    del cfg
    out = [("tape_off", {**train_sw, "cand_in": 0.0, "cand_out": 0.0}, "network"),
           ("mix_w1", {**train_sw, "src_5": 1.0}, "mixture")]
    out += [(f"net_w{w:g}", {**train_sw, "src_5": w}, "network") for w in E5_W]
    return out


def run_e5(args) -> None:
    import numpy as np

    from bes.data.samples import rng_for
    from bes.eval import probes as P
    from bes.model import model as M
    from bes.tokenizer import load as load_tok

    src = _abs(args.src)
    out = _abs(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok()
    rec, cfg, params = mb.load_arm(src, args.ref_arm)
    models = [("ref", rec, cfg, params)]
    try:
        models.append(("no_tape_arm",) + mb.load_arm(src, args.no_tape_arm))
    except FileNotFoundError:
        log(f"no finished arm {args.no_tape_arm!r} in {src}: the baseline is the evaluation-time switch")
    prompts = {"chat": P.chat_prompts(args.n_prompts, tok), "web": P.web_prompts(args.n_prompts, tok)}
    done = {(r["domain"], r["setting"], r["temperature"]) for r in mb.read_jsonl(out / E5_FILE)
            if r.get("ref_hash") == rec["spec_hash"] and r.get("n_prompts") == args.n_prompts
            and r.get("e5_version") == E5_VERSION}
    for label, mrec, mcfg, mparams in models:
        train_sw = dict(tuple(s) for s in mrec["arm"]["switches"])
        settings = e5_settings(mcfg, train_sw) if label == "ref" else \
            [("no_tape_arm", train_sw, "network")]
        for domain, plist in prompts.items():
            max_new = args.max_new_chat if domain == "chat" else args.max_new_web
            for setting, sw, temper in settings:
                if domain == "web" and setting.startswith("net_w") and setting != "net_w1":
                    continue          # no assistant turn: the self-evidence scale has no effect
                consts = M.make_consts(mcfg.model, **sw)
                for temp in E5_TEMPS:
                    if (domain, setting, temp) in done:
                        continue
                    t0 = time.time()
                    samples = []
                    for lo in range(0, len(plist), args.batch):
                        part = plist[lo:lo + args.batch]
                        rngs = [rng_for("e5", args.seed, domain, lo + i, temp) for i in range(len(part))]
                        got = P.decode_batch(mparams, consts, mcfg.model, part, max_new=max_new,
                                             tok=tok, temperature=temp, rngs=rngs, temper=temper)
                        for pr, d in zip(part, got):
                            samples.append(P.self_generation_metrics(d.tokens, pr, d.stopped))
                    mean = {k: float(np.mean([s[k] for s in samples])) for k in ("rep4", "loop", "copy", "n", "stopped")}
                    mb.append_jsonl(out / E5_FILE, {
                        "type": "e5", "e5_version": E5_VERSION, "domain": domain, "setting": setting,
                        "temperature": temp, "temper": temper, "model": label, "arm": mrec["name"],
                        "ref_hash": rec["spec_hash"], "from": mb._rel(src), "n_prompts": args.n_prompts,
                        "max_new": max_new, "switches": sw, "samples": samples, "mean": mean,
                        "seconds": time.time() - t0, "tokens": mrec["tokens"]})
                    log(f"E5 {domain} {setting} ({temper}) T={temp}: rep4 {mean['rep4']:.3f} loop "
                        f"{mean['loop']:.2f} copy {mean['copy']:.3f} len {mean['n']:.0f} "
                        f"({time.time() - t0:.0f} s)")


def _paired(a: list[float], b: list[float]) -> tuple[float, float]:
    import numpy as np

    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2:
        return float(d.mean()) if d.size else 0.0, 0.0
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))


def e5_verdicts(recs: list[dict], tol: float = E5_TOL) -> dict:
    """``{(domain, setting): (ok, [reasons])}``: a setting is *not worse* than the baseline when,
    at every temperature and for each of rep4, loop and copy, the paired mean difference is at
    most ``max(2 * paired SE, tol)``. Baseline: the model trained without the tape when present,
    else the evaluation-time switch ``tape_off``."""
    by = {(r["domain"], r["setting"], r["temperature"]): r for r in recs}
    out = {}
    for (domain, setting, temp), r in by.items():
        if setting in ("tape_off", "no_tape_arm"):
            continue
        base = by.get((domain, "no_tape_arm", temp)) or by.get((domain, "tape_off", temp))
        if base is None:
            continue
        ok, why = out.get((domain, setting), (True, []))
        for metric in ("rep4", "loop", "copy"):
            a = [float(s[metric]) for s in r["samples"]]
            b = [float(s[metric]) for s in base["samples"]]
            delta, se = _paired(a, b)
            if delta > max(2 * se, tol):
                ok = False
                why = why + [f"T={temp} {metric} +{delta:.3f} (SE {se:.3f}) vs {base['setting']}"]
        out[(domain, setting)] = (ok, why)
    return out


def e5_report(out: Path) -> str:
    import numpy as np

    from bes.eval import probes as P

    recs = [r for r in mb.read_jsonl(out / E5_FILE) if r.get("e5_version") == E5_VERSION]
    latest = {}
    for r in recs:
        latest[(r["domain"], r["setting"], r["temperature"])] = r
    recs = list(latest.values())
    head = ["# E5 — self-generation (gate E5 of `docs/PLAN.md` section 7)", "",
            f"Generated by `python scripts/run_probes.py --e5 --report` from `{mb._rel(out)}`; do not "
            "edit by hand.", "",
            "## Rule (pre-registered)", "",
            "Plan section 7, E5: with the candidate tape on, sampling at temperatures 0.5-0.8 must "
            "not be worse than without it (the model trained without the count heads) on the "
            "repeated 4-gram rate, the loop rate and the share copied verbatim; otherwise the "
            "self-evidence is down-weighted or the tape switched off at decoding. Operationally: a "
            f"setting is *not worse* when every paired difference (same prompts, same random "
            f"numbers) is at most max(2 x paired SE, {E5_TOL}). Baseline: the E2 `no_tape` arm "
            "when it exists, else the reference with the tape switched off at evaluation time.", "",
            "Settings: `tape_off` (the tape switched off at evaluation time); `mix_w1` (the tape on, "
            "the whole output law tempered, `softmax(log p / T)`: the earlier sampler, self-evidence "
            "1); `net_w<w>` (the tape on, the temperature on the network's distribution only, "
            "`p_T = (softmax(z / T) + masses) / (1 + M)`, as the decoder samples, with the masses of "
            "candidates whose source is the turn being generated scaled by w: "
            "`make_consts(cfg, src_5=w)`; web prompts have no such turn, so only w = 1 is run).", "",
            "Definitions (`bes/eval/probes.py`): **rep4** = share of the 4-gram occurrences of a "
            "sample that repeat an earlier 4-gram of the same sample; **loop** = the sample did not "
            f"emit END within its budget and its last max(2p, {P.LOOP_MIN_TAIL}) tokens are exactly "
            f"periodic with a period p <= {P.LOOP_MAX_PERIOD} (an absorbing repetition loop); "
            f"**copy** = share of generated tokens inside a run of >= {P.COPY_MIN_RUN} tokens that "
            "occurs verbatim in the prompt.", ""]
    if not recs:
        return "\n".join(head + ["## Status", "", "**Not run yet.** Run `scripts/run_probes.py --e5 "
                                 "--from runs/e2 --out runs/e5` after the E2 reference has finished, "
                                 "then regenerate this file with `--e5 --report`."]) + "\n"
    r0 = next((r for r in recs if r["model"] == "ref"), recs[0])
    base = "the E2 `no_tape` arm" if any(r["model"] == "no_tape_arm" for r in recs) else \
        "the evaluation-time switch `tape_off` (no `no_tape` arm found)"
    lines = head + ["## Setup", "",
                    f"- Model: arm `{r0['arm']}` of `{r0['from']}` ({r0['tokens']:,} training tokens); "
                    f"baseline: {base}; {r0['n_prompts']} held-out prompts per domain "
                    "(chat: system + first user turn of smol-smoltalk test conversations; web: the first "
                    "96 tokens of ClimbMix shard_00997 documents).", ""]
    verdicts = e5_verdicts(recs)
    for domain in ("chat", "web"):
        drecs = [r for r in recs if r["domain"] == domain]
        if not drecs:
            continue
        lines += [f"## {domain}", ""]
        rows = []
        for temp in E5_TEMPS:
            for setting in E5_ORDER:
                r = latest.get((domain, setting, temp))
                if r is None:
                    continue
                s = r["samples"]
                n = len(s)
                k = sum(1 for x in s if x["loop"])
                lo, hi = P.wilson(k, n)
                rep = np.asarray([x["rep4"] for x in s], float)
                se = float(rep.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                rows.append([f"{temp}", f"`{setting}`", f"{r['mean']['rep4']:.3f} ± {se:.3f}",
                             f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]", f"{r['mean']['copy']:.3f}",
                             f"{r['mean']['n']:.0f}", f"{r['mean']['stopped']:.0%}"])
        lines += [mb._md_table(["T", "setting", "rep4 (± SE)", "loop", "copy", "mean length",
                                "stopped"], rows), ""]
        vrows = [[f"`{st}`", "**not worse**" if ok else "**worse**", "; ".join(why) or "-"]
                 for (d, st), (ok, why) in sorted(verdicts.items()) if d == domain]
        if vrows:
            lines += [f"### Verdicts ({domain})", "", mb._md_table(["setting", "verdict", "violations"], vrows), ""]
    chat_ok = [w for w in E5_W if verdicts.get(("chat", f"net_w{w:g}"), (False,))[0]]
    lines += ["## Decision", ""]
    if not chat_ok:
        lines.append("No self-evidence scale passes on chat prompts with the temperature on the "
                     "network: switch the tape off at decoding (or down-weight below the tested "
                     "values) until a retrained model passes.")
    else:
        lines.append(f"Largest passing self-evidence scale on chat prompts (temperature on the network, "
                     f"as the decoder samples): **src_5 = {max(chat_ok):g}** (passing: "
                     f"{', '.join(f'{w:g}' for w in chat_ok)}; the decoder default is "
                     "`DecodeCfg.self_evidence_weight`).")
    for domain in ("chat", "web"):
        for setting, what in (("net_w1", "the temperature on the network"),
                              ("mix_w1", "the whole mixture tempered")):
            ok = verdicts.get((domain, setting), (None,))[0]
            if ok is not None:
                lines.append(f"{domain.capitalize()} prompts, tape on, self-evidence 1, {what}: "
                             f"{'not worse' if ok else 'worse'} than the baseline.")
    lines += ["", "## Limits", "",
              "- Micro model (d=128, 1-4M tokens): its unconditional samples are far less fluent than "
              "the final model's, so loop and repetition rates are high for every setting; the gate "
              "compares settings on the same model and prompts, not absolute rates.",
              "- One sample per prompt and setting; the paired SE is over prompts."]
    return "\n".join(lines) + "\n"


# ============================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=1_000_000, help="training tokens per E4 arm")
    ap.add_argument("--out", default=None, help="run directory (default runs/e4, or runs/e5 with --e5)")
    ap.add_argument("--arms", default=None, help="comma-separated subset of the E4 arms")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--n-test", type=int, default=96, help="E4 items per probe")
    ap.add_argument("--batch", type=int, default=24, help="streams decoded at once")
    ap.add_argument("--e5", action="store_true", help="E5 instead of E4")
    ap.add_argument("--from", dest="src", default="runs/e2", help="E5: run directory with the trained arms")
    ap.add_argument("--ref-arm", default="ref")
    ap.add_argument("--no-tape-arm", default="no_tape")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--max-new-chat", type=int, default=160)
    ap.add_argument("--max-new-web", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpus", default="")
    ap.add_argument("--seq-len", type=int, default=mb.Bench.seq_len)
    ap.add_argument("--micro-rows", type=int, default=mb.Bench.micro_rows)
    ap.add_argument("--tokens-per-step", type=int, default=mb.Bench.tokens_per_step)
    ap.add_argument("--eval-rows", type=int, default=mb.Bench.eval_rows)
    ap.add_argument("--eval-fracs", default=",".join(str(f) for f in mb.Bench.eval_fracs))
    ap.add_argument("--run-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--eval-arm", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = str(_abs(args.out or ("runs/e5" if args.e5 else "runs/e4")))
    out = Path(args.out)
    bench = MA.bench_from(args)

    if args.e5:
        if args.report:
            doc = Path(args.doc or REPO / "docs" / "E5_SELF_GENERATION.md")
            doc.write_text(e5_report(out))
            log(f"wrote {doc}")
            return 0
        run_e5(args)
        return 0
    if args.report:
        doc = Path(args.doc or REPO / "docs" / "E4_PROBES.md")
        doc.write_text(e4_report(out))
        log(f"wrote {doc}")
        return 0
    if args.run_arm:
        MA.die_with_parent()
        mb.run_arm(MA.select(E4_ARMS, args.run_arm)[0], out, args.tokens, bench, log=log)
        return 0
    if args.eval_arm:
        MA.die_with_parent()
        eval_arm(args, args.eval_arm)
        return 0

    arms = MA.select(E4_ARMS, args.arms)
    for a in arms:
        state = "done" if mb.finished(out, a, args.tokens, bench) else "todo"
        print(f"  {a.name:16s} {state:5s} {a.component}", flush=True)
    if args.list:
        return 0
    failed = []
    extra = ["--n-test", str(args.n_test), "--batch", str(args.batch)]

    def runner(arm: Arm) -> None:
        log(f"start {arm.name}")
        if MA.child(args, ["--run-arm", arm.name] + extra, __file__) != 0:
            failed.append(arm.name)
            log(f"{arm.name} failed; continuing")

    mb.run_queue(arms, out, args.tokens, bench, runner=runner, log=log)
    for a in arms:
        if mb.finished(out, a, args.tokens, bench) is None:
            continue
        log(f"probe evaluation of {a.name}")
        if MA.child(args, ["--eval-arm", a.name] + extra, __file__) != 0:
            failed.append(f"eval {a.name}")
    log("finished" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
