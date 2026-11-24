"""The micro graft experiment, recomputed from its records (N/micro_graft/6M/results_all.jsonl; and,
if given, the restart arms' records of the 3M directory) without the experiment's own report code:

* every held-out loss is recomputed from the per-row sums (sum of ce / sum of weights) and checked
  against the record's `loss`; `mix` = the arms' row shares of the mixture (web 0.75, chat 0.25)
  applied to the web and chat losses, as `bes.eval.microbench.domain_shares` does;
* mean and sample standard deviation over the three seeds; differences paired by seed;
* share of the page gain recovered, (ref - graft) / (ref - v9_pages), per seed and as the ratio of
  the mean differences;
* the checks: a graft arm's evaluations before its graft equal ref's of the same seed (per row),
  and the graft's function check and held-out before/after;
* parameter counts, steps, graft steps, carried schedule, timing.

    python -I -B micrograft.py [RESTART_DIR]
"""
import json
import statistics as st
import sys
from pathlib import Path

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in open(N / "micro_graft/6M/results_all.jsonl")]
by = {r["name"]: r for r in recs}
SEEDS = {0: "", 1: "_s1", 2: "_s2"}
FAM = ["ref", "v9_pages", "graft_25", "graft_50", "graft_75"]
DOMS = ["mix", "web", "chat", "assistant"]


def name(fam, seed):
    return fam + SEEDS[seed]


def evals(rec):
    """{tokens: {domain: loss}} recomputed from the per-row sums; mix from the mixture's row shares."""
    mixture = dict((d, s) for d, s in rec["arm"]["mixture"])
    out = {}
    for c in rec["curve"]:
        row = {}
        for d, e in c["eval"].items():
            ce, w = e["ce"], e["w"]
            loss = sum(ce) / sum(w)
            assert abs(loss - e["loss"]) < 1e-9, (rec["name"], c["tokens"], d, loss, e["loss"])
            row[d] = loss
        sh = {d: mixture[d] for d in ("web", "chat")}
        s = sum(sh.values())
        row["mix"] = sum(sh[d] / s * row[d] for d in sh)
        out[c["tokens"]] = row
    return out


E = {r["name"]: evals(r) for r in recs}
T = sorted(E["ref"])
last = T[-1]


def ms(xs):
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None, "n": len(xs), "values": xs}


table = {}
for fam in FAM:
    for t in T:
        table[f"{fam}@{t}"] = {d: ms([E[name(fam, s)][t][d] for s in SEEDS]) for d in DOMS}


def paired(a, b, t):
    return {d: ms([E[name(a, s)][t][d] - E[name(b, s)][t][d] for s in SEEDS]) for d in DOMS}


diffs = {}
for t in T:
    diffs[f"v9_pages-ref@{t}"] = paired("v9_pages", "ref", t)
for fam, g in (("graft_25", 0.25), ("graft_50", 0.5), ("graft_75", 0.75)):
    for t in T:
        if t > g * T[-1] + 1:
            diffs[f"{fam}-ref@{t}"] = paired(fam, "ref", t)
            diffs[f"{fam}-v9_pages@{t}"] = paired(fam, "v9_pages", t)
share = {}
for fam in ("graft_25", "graft_50", "graft_75"):
    share[fam] = {}
    for d in DOMS:
        num = [E[name("ref", s)][last][d] - E[name(fam, s)][last][d] for s in SEEDS]
        den = [E[name("ref", s)][last][d] - E[name("v9_pages", s)][last][d] for s in SEEDS]
        share[fam][d] = {"per_seed": [n / m for n, m in zip(num, den)], "ratio_of_means": st.mean(num) / st.mean(den),
                         "mean_of_ratios": st.mean([n / m for n, m in zip(num, den)])}

# checks
checks = {}
for fam in ("graft_25", "graft_50", "graft_75"):
    for s in SEEDS:
        r = by[name(fam, s)]
        g = r["graft"]
        ref = by[name("ref", s)]
        before = [c for c in r["curve"] if c["tokens"] <= g["tokens"]]
        eq = []
        for c in before:
            cr = [x for x in ref["curve"] if x["tokens"] == c["tokens"]][0]
            eq.append(all(c["eval"][d]["ce"] == cr["eval"][d]["ce"] and c["eval"][d]["w"] == cr["eval"][d]["w"]
                          for d in c["eval"]))
        checks[name(fam, s)] = {"graft_step": g["step"], "graft_tokens": g["tokens"], "carried": g["carried"],
                                "trunk_source": g["trunk"]["source"],
                                "check_difference": g["check"]["difference"], "check_loss": g["check"]["loss_before"],
                                "heldout_difference": {d: g["heldout"][d]["difference"] for d in g["heldout"]},
                                "heldout_max_row_difference": {d: g["heldout"][d]["max_row_difference"] for d in g["heldout"]},
                                "evals_before_graft_equal_ref_per_row": all(eq) and len(eq) > 0, "n_evals_before": len(eq),
                                "copied_params": g["leaves"]["copied_params"], "copied_param_numbers": g["leaves"]["copied_param_numbers"],
                                "new_params": g["leaves"]["new_params"], "new_param_numbers": g["leaves"]["new_param_numbers"],
                                "copied_leaves_total": g["leaves"]["copied"]}
meta = {
    "n_params": {fam: by[name(fam, 0)]["n_params"] for fam in FAM},
    "steps": by["ref"]["steps"], "tokens_per_step": by["ref"]["tokens_per_step"], "budget_tokens": by["ref"]["tokens"],
    "checkpoints_tokens": T, "seeds": list(SEEDS), "mixture": by["ref"]["arm"]["mixture"],
    "token_shares": by["ref"]["data"]["token_shares"], "pools_rows": by["ref"]["data"]["pools"],
    "passes_started": by["ref"]["data"]["passes"], "eval_rows": by["ref"]["bench"]["eval_rows"],
    "seq_len": by["ref"]["bench"]["seq_len"], "d_model": by["ref"]["config"]["model"]["d_model"],
    "n_loops": by["ref"]["config"]["model"]["n_loops"], "n_blocks": by["ref"]["config"]["model"]["n_blocks"],
    "tok_neurons": by["v9_pages"]["config"]["model"]["tok_neurons"], "tok_dim": by["v9_pages"]["config"]["model"]["tok_dim"],
    "gain_o": by["v9_pages"]["config"]["model"]["gain_o"],
    "train_flops_per_token": {fam: by[name(fam, 0)]["flops_per_token"]["train"] for fam in FAM},
    "sec_per_step": {r["name"]: round(r["timing"]["sec_per_step"], 3) for r in recs},
    "train_s": {r["name"]: round(r["timing"]["train_s"]) for r in recs},
    "peak_rss_mb": {r["name"]: round(r["peak_rss_mb"]) for r in recs},
    "lr_schedule": {k: by["ref"]["config"]["optim"][k] for k in ("final_lr_frac", "warmup_frac", "warmdown_frac", "horizon_tokens")},
    "optimizer": by["ref"]["config"]["optim"]["optimizer"],
    "status": {r["name"]: r["status"] for r in recs},
}
# rows read per domain by the mixture's row plan (arithmetic on the configuration): 8 rows per step
rows_total = meta["steps"] * 8
meta["rows_total"] = rows_total

restart = None
if len(sys.argv) > 1:
    rd = Path(sys.argv[1])
    rr = {}
    for f in sorted(rd.rglob("results.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "done":
                rr[(r["name"], r["tokens"])] = r
    restart = {"files": [str(f) for f in sorted(rd.rglob("results.jsonl"))], "arms": {}}
    for (nm, tok), r in sorted(rr.items()):
        ev = evals(r)
        lt = max(ev)
        restart["arms"][nm] = {"tokens_budget": tok, "steps": r["steps"], "last_tokens": lt, "loss": ev[lt],
                               "seed": r["config"]["run"]["seed"], "n_params": r["n_params"],
                               "horizon_tokens": r["config"]["optim"]["horizon_tokens"],
                               "sec_per_step": round(r["timing"]["sec_per_step"], 3),
                               "curve_tokens": sorted(ev)}
    pairs = {}
    for s in SEEDS:
        nm = name("v9_pages", s)
        if nm in restart["arms"]:
            L = restart["arms"][nm]["loss"]
            pairs[s] = {d: {"restart": L[d], "graft_50": E[name("graft_50", s)][last][d],
                            "ref_6M": E[name("ref", s)][last][d], "ref_3M": E[name("ref", s)][T[1]][d],
                            "restart_minus_graft_50": L[d] - E[name("graft_50", s)][last][d],
                            "restart_minus_ref_6M": L[d] - E[name("ref", s)][last][d]} for d in DOMS}
    restart["paired_by_seed"] = pairs
    if len(pairs) == 3:
        restart["summary"] = {d: {"restart": ms([pairs[s][d]["restart"] for s in SEEDS]),
                                  "restart_minus_graft_50": ms([pairs[s][d]["restart_minus_graft_50"] for s in SEEDS]),
                                  "restart_minus_ref_6M": ms([pairs[s][d]["restart_minus_ref_6M"] for s in SEEDS])}
                              for d in DOMS}

out = {"table": table, "diffs": diffs, "share": share, "checks": checks, "meta": meta, "restart": restart}
(HERE / "micrograft.json").write_text(json.dumps(out, indent=1))


def f(x, sign=False):
    return (f"{x:+.4f}" if sign else f"{x:.4f}")


print("budget", meta["budget_tokens"], "steps", meta["steps"], "checkpoints", T, "params", meta["n_params"])
print("mixture", meta["mixture"], "token shares", meta["token_shares"], "pools", meta["pools_rows"], "passes", meta["passes_started"])
for fam in FAM:
    for t in T:
        r = table[f"{fam}@{t}"]
        print(f"{fam:9s} {t:>9,} " + "  ".join(f"{d} {f(r[d]['mean'])} ± {r[d]['sd']:.4f}" for d in DOMS))
for k, v in diffs.items():
    print(f"{k:28s} " + "  ".join(f"{d} {f(v[d]['mean'], True)} ± {v[d]['sd']:.4f}" for d in DOMS)
          + "   per seed mix " + ", ".join(f"{x:+.4f}" for x in v["mix"]["values"]))
for fam, v in share.items():
    print(fam, "share recovered:", {d: (round(x["ratio_of_means"], 3), [round(y, 3) for y in x["per_seed"]]) for d, x in v.items()})
for k, v in checks.items():
    print(k, {kk: vv for kk, vv in v.items() if kk not in ("new_params",)})
print("sec_per_step", meta["sec_per_step"])
print("train_s", meta["train_s"])
print("flops", meta["train_flops_per_token"])
if restart:
    print(json.dumps(restart, indent=1)[:4000])
