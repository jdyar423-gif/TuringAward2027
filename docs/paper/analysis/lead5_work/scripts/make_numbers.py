"""Every derived number the final paper cites from the train logs, the stage ledger, the TPU ledger and
the review's arithmetic, recomputed from the raw files and written to lead5_work/numbers.json.
Arithmetic on measured inputs; each entry names its inputs.

Fifth pass (lead5_work, copied from lead3_work/make_numbers.py): the v9 records are those of the last
packed snapshot N/snap_final (created 20:16 UTC on 24 November; extracted by extract_snap.py into
lead5_work/snap_final_x/, byte-identical to N/records/unpacked/); training has ended (step 51,812). Kept from
the third pass: every live-rate stretch, the ratios, the FLOP arithmetic, the power arithmetic, the
three-point line and the projections (as projections, now set against the ledger). New: the 2,048-token
stretch to the end of phase "long" (p < 0.92) and the stretch of phase "tail" (p >= 0.92, loss on
assistant tokens only); the trainer's ledger at the end; the TPU ledger closed (three intervals); the
post-training stages from the stage ledger (UTC start, finish, used seconds) and their results;
power at the tie rates of the test readings.

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B make_numbers.py [V9_TRAIN_LOG]
"""
import datetime as dt
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parent
N = HERE.parents[2]
FLOG = N / "paper/lead_work/final_snap/run/train/train_log.jsonl"
SNAPX = W / "snap_final_x"
VLOG = Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPX / "run/train/train_log.jsonl"
VLED = VLOG.parent / "ledger.json"
STAGES = SNAPX / "run/stages.json"
MAN = SNAPX / "run/manifest.json"
PREF = SNAPX / "run/post/prefopt_report.json"
TPU = Path("/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/tpu_intervals.txt")
T0 = 1795455991.9461858          # train t0 of run main16 (ledger 't0'; 23 Nov 17:46:31.946 UTC)
out = {}


def put(key, value, how):
    out[key] = {"value": value, "how": how}


def steps(path):
    return [d for d in map(json.loads, open(path)) if d.get("event") == "step"]


def events(path, kind):
    return [d for d in map(json.loads, open(path)) if d.get("event") == kind]


f = steps(FLOG)
v = steps(VLOG)


def utc(t):
    return dt.datetime.fromtimestamp(T0 + t, dt.timezone.utc).strftime("%H:%M:%S")


def utc_abs(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%d %b %H:%M:%S")


# ---- live rates: (a) the trainer's own per-step rate tok_per_s; (b) achieved between records
def stretch(recs, lo, hi):
    """records with lo < step <= hi (step records are every 50 steps)"""
    return [d for d in recs if lo < d["step"] <= hi]


def achieved(recs):
    """262,144 x steps / wall time between consecutive records of the same stretch (no restart
    in between: the step gap is 50 and the time gap is under 200 s)."""
    r = []
    for a, b in zip(recs, recs[1:]):
        ds, dt_ = b["step"] - a["step"], b["time"] - a["time"]
        if ds == 50 and 0 < dt_ < 200:
            r.append(262144 * ds / dt_)
    return r


def rate_entry(s):
    rates = [d["tok_per_s"] for d in s]
    ach = achieved(s)
    return {"median_tok_per_s": round(st.median(rates)), "n": len(rates),
            "min": round(min(rates)), "max": round(max(rates)),
            "achieved_median": round(st.median(ach)) if ach else None, "achieved_n": len(ach),
            "first_step": s[0]["step"], "last_step": s[-1]["step"],
            "first_utc": utc(s[0]["time"]), "last_utc": utc(s[-1]["time"])}


stretches = {
    "rt2_16rows": (f, 0, 5075, "runtime 2, final, 16-row micro-batches, splash kernel"),
    "rt3_same": (f, 5075, 9332, "runtime 3, the same program, before the pieces"),
    "pieces": (f, 9332, 22613, "4 pieces per micro-batch"),
    "kernel_to_deflection": (f, 22613, 25280, "one-tile kernel + ReLU^2 backward, to the deflection rule"),
    "kernel_to_graft": (f, 22613, 29722, "one-tile kernel + ReLU^2 backward, to the graft"),
    "prekernel_21000": (f, 21000, 22613, "the 1,600 steps before the kernel switch"),
    "final_25300_29700": (f, 25250, 29700, "final, steps 25,300-29,700 (review item 26)"),
    "v9_take": (v, 29722, 31912, "v9 as grafted (take), before the three switches"),
    "v9_switches_1024": (v, 31912, 10 ** 9, "v9 with flat, rows, tokn at 1,024 tokens"),
}
for k, (recs, lo, hi, what) in stretches.items():
    s = [d for d in stretch(recs, lo, hi) if d.get("seq_len", 1024) == 1024]
    if not s:
        continue
    put(f"rate.{k}", rate_entry(s),
        f"{what}: median of the train log's per-step tok_per_s over the step records with "
        f"{lo} < step <= {hi} at 1,024 tokens; achieved = 262,144 x 50 / time between consecutive records")
s2048 = [d for d in v if d.get("seq_len") == 2048 and d.get("phase", "long") == "long"]
put("rate.v9_switches_2048", rate_entry(s2048),
    "v9 with the switches at 2,048 tokens, trainer phase 'long' (p from 0.67 to 0.92)")
s_tail = [d for d in v if d.get("phase") == "tail"]
put("rate.v9_tail_2048", rate_entry(s_tail),
    "v9 at 2,048 tokens, trainer phase 'tail' (p >= 0.92: chat share 0.70, loss on assistant tokens only)")
s_all2048 = [d for d in v if d.get("seq_len") == 2048]
put("rate.v9_all_2048", rate_entry(s_all2048), "v9 at 2,048 tokens, phases 'long' and 'tail' together")
phases = sorted({d.get("phase") for d in v})
put("v9.phases_seen", {ph: sum(d.get("phase") == ph for d in v) for ph in phases},
    "trainer phases in the v9 step records, with record counts (short = 1,024 tokens, long = 2,048, tail = p >= 0.92)")

r = lambda k: out[f"rate.{k}"]["value"]["median_tok_per_s"]  # noqa: E731
put("ratio.kernel_gain_live", round(r("kernel_to_deflection") / r("prekernel_21000") - 1, 4),
    "median after the kernel switch (to 25,280) / median of the 1,600 steps before, minus 1")
put("ratio.final_over_v9_take_live", round(r("final_25300_29700") / r("v9_take"), 4),
    "final 25,300-29,700 median / v9 (take) median")
put("ratio.final_over_v9_switches_live", round(r("final_25300_29700") / r("v9_switches_1024"), 4),
    "final 25,300-29,700 median / v9 with the switches median (1,024 tokens)")
put("ratio.v9_switches_gain_live", round(r("v9_switches_1024") / r("v9_take") - 1, 4),
    "v9 with the switches / v9 as grafted, live medians, minus 1")
put("ratio.v9_2048_vs_1024", round(r("v9_switches_2048") / r("v9_switches_1024") - 1, 4),
    "v9 at 2,048 tokens (phase long) / at 1,024, live medians, minus 1")
put("ratio.reference_to_kernel_live", [round(r("final_25300_29700") / 116718, 3), round(r("final_25300_29700") / 115394, 3)],
    "final's live median after the kernel (steps 25,300-29,700) / the two monitor readings of the reference "
    "implementation on runtime 1 (116,718 and 115,394 tokens/s, S/colab_bes24h_final.log 04:17:59 and 04:28:00)")
gap = {}
for k in list(out):
    if k.startswith("rate.") and out[k]["value"].get("achieved_median"):
        val = out[k]["value"]
        gap[k] = round(abs(val["median_tok_per_s"] - val["achieved_median"]) / val["median_tok_per_s"] * 100, 3)
put("rate.largest_gap_live_achieved_pct", {"largest": max(gap.values()), "by_stretch": gap},
    "|live median - achieved median| / live median, in percent, per stretch")

# ---- the trainer's ledger at the end, and the TPU ledger (closed)
led = json.loads(VLED.read_text())
put("v9.final_ledger", {"steps": led["steps"], "tokens": led["tokens"], "mark_utc": utc(led["mark"] - T0),
                        "phase": led["phase"], "train_s": round(led["seconds"]["train"], 1),
                        "train_h": round(led["seconds"]["train"] / 3600, 3), "ckpt_s": round(led["seconds"]["ckpt"], 1),
                        "setup_s": round(led["seconds"]["setup"], 1), "lost_s": round(led["seconds"]["lost"], 1),
                        "lost_h": round(led["seconds"]["lost"] / 3600, 2),
                        "restarts": led["restarts"], "rollbacks": led["rollbacks"],
                        "v9_tokens_since_graft": led["tokens"] - 7791443968, "v9_steps_since_graft": led["steps"] - 29722,
                        "tokens_check": led["tokens"] == 262144 * led["steps"]},
    "the trainer's time ledger in the last snapshot (run/train/ledger.json): steps and tokens trained, seconds by phase")
end = events(VLOG, "end")[-1]
last = v[-1]
put("v9.end", {"step": end["step"], "p": end["p"], "reason": end["reason"], "end_utc": utc(end["time"]),
               "last_step_record": last["step"], "last_record_utc": utc(last["time"]), "last_loss": round(last["loss"], 4),
               "last_phase": last.get("phase"), "last_lr": round(last["lr"], 4), "last_chat_frac": last.get("chat_frac"),
               "assistant_only": last.get("assistant_only")},
    "the end event of the train log and its last step record")
train_end = events(VLOG, "setup")[-1]["train_end"]
put("v9.train_end_setup_event", utc(train_end - T0), "the training end of the run's setup event (train_end, epoch)")
iv = [line.split()[:2] for line in open(TPU) if line.strip()]
secs = [int(b) - int(a) for a, b in iv]
put("tpu.final", {"intervals_utc": [[utc_abs(int(a)), utc_abs(int(b))] for a, b in iv], "seconds_by_runtime": secs,
                  "total_s": sum(secs), "total_h": round(sum(secs) / 3600, 3), "limit_s": 86400,
                  "unused_s": 86400 - sum(secs), "train_s_ledger": round(led["seconds"]["train"], 1),
                  "train_share": round(led["seconds"]["train"] / sum(secs), 4),
                  "runtime3_train_share": None},
    "TPU seconds held per runtime (S/tpu_intervals.txt, all three intervals closed; the third given back by hand "
    "at its end epoch) and the share in training steps of the kept trajectory (ledger 'train' seconds)")

# ---- the post-training stages (stage ledger) and their results
sg = json.loads(STAGES.read_text())
stage_rows = {}
for name in ("benchmark", "train", "sample", "critic", "prefopt", "select", "judge", "export"):
    srec = sg["stages"][name]
    stage_rows[name] = {"status": srec["status"], "planned_s": srec["planned_s"], "slice_s": srec["slice_s"],
                        "used_s": srec["used_s"], "down_s": srec["down_s"], "attempts": srec["attempts"],
                        "started_utc": utc_abs(srec["started"]) if srec.get("started") else None,
                        "finished_utc": utc_abs(srec["finished"]) if srec.get("finished") else None}
put("stages", {"deadline_utc": utc_abs(sg["deadline"]), "supervisor_end_utc": utc_abs(sg["t0"] + [
    e for e in sg["events"] if e["event"] == "supervisor_end"][-1]["time"]), "rows": stage_rows,
    "post_used_s": round(sum(stage_rows[k]["used_s"] for k in ("sample", "prefopt", "select", "export")), 1),
    "slack_before_deadline_s": round(sg["deadline"] - sg["stages"]["export"]["finished"], 1)},
    "run/stages.json: each stage's status, planned, slice and used seconds and its UTC start and finish")
st_res = {k: sg["stages"][k]["result"] for k in ("sample", "prefopt", "select", "export")}
put("post.results", st_res, "the stage ledger's result record of each post-training stage")
pref = json.loads(PREF.read_text())
cand = []
for c in pref["candidates"]:
    cand.append({"lr": c["lr"], "accepted": c["accepted"], "reasons": c["reasons"], "checked": c["checked"],
                 "acc": round(c["pref"]["acc"], 4), "acc_se": round(c["pref"]["acc_se"], 4),
                 "acc_gain_needed": round(math.hypot(pref["start"]["pref"]["acc_se"], c["pref"]["acc_se"]), 4),
                 "heldout_loss": round(c["heldout_loss"], 5),
                 "heldout_loss_rise_pct": round((c["heldout_loss"] / pref["start"]["heldout_loss"] - 1) * 100, 2),
                 "mean_len": round(c["stats"]["mean_len"], 1), "mean_len_ratio": round(c["stats"]["mean_len"] / pref["start"]["stats"]["mean_len"], 3),
                 "rep4": round(c["stats"]["rep4"], 4), "rep4_ratio": round(c["stats"]["rep4"] / pref["start"]["stats"]["rep4"], 3),
                 "n_sampled": c["stats"]["n"], "steps": c["train"]["steps"], "train_time_s": round(c["train"]["time_s"], 1),
                 "train_acc": round(c["train"]["train_acc"], 4), "max_grad_norm": round(c["train"]["max_grad_norm"], 1),
                 "skipped": c["train"]["skipped"], "policy_acc": round(c["pref"]["policy_acc"], 4)})
put("prefopt", {"beta": pref["beta"], "lrs": pref["lrs"], "chosen_lr": pref["lr"], "reason": pref["reason"],
                "n_train_pairs": pref["n_train_pairs"], "n_heldout_pairs": pref["n_heldout_pairs"],
                "steps_per_epoch": pref["steps_per_epoch"], "time_s": round(pref["time_s"], 1),
                "start": {"acc": pref["start"]["pref"]["acc"], "heldout_loss": round(pref["start"]["heldout_loss"], 5),
                          "mean_len": round(pref["start"]["stats"]["mean_len"], 1), "rep4": round(pref["start"]["stats"]["rep4"], 4),
                          "policy_acc": round(pref["start"]["pref"]["policy_acc"], 4)},
                "candidates": cand,
                "pairs_by_source": sg["stages"]["prefopt"]["result"]["pairs_by_source"]},
    "run/post/prefopt_report.json (bes/train/prefopt.py): the start and each candidate learning rate, the "
    "acceptance checks (accuracy gain > 1 x its standard error; held-out assistant loss +3% at most; length "
    "0.7-1.6 x and repetition 1.2 x + 0.01 of the start, 2 standard errors) and the choice (best accuracy)")
man = json.loads(MAN.read_text())
dec = man["decode"]
put("select", {"chosen": dec["chosen"], "scores": dec["scores"], "n_common": dec["n_common"], "vs_default": dec["vs_default"],
               "rule": dec["rule"], "weights": dec.get("weights"), "checkpoint": dec.get("checkpoint"),
               "summaries": {k: {"maths_correct": rr["summary_math"]["correct"], "maths_n": rr["summary_math"]["n"],
                                 "constraints_all_ok": rr["summary_constraints"]["prompts_all_ok"],
                                 "constrained_n": rr["summary_constraints"]["n_prompts"],
                                 "degenerate": rr["summary_degenerate"]["degenerate"],
                                 "truncated": rr["summary_degenerate"]["truncated"], "run_id": rr["run_id"]}
                             for k, rr in dec["runs"].items()},
               "library": dec["library"]["reason"]},
    "run/manifest.json 'decode' (scripts/run_24h.py stage_select): exact scores of the three variants on the 200 dev "
    "prompts (mean of maths correct, all constraints met, not degenerate), paired sign tests against the default")

# ---- losses
lv34 = [d["loss"] for d in v if d["step"] <= 34350]
put("loss.v9_to_34350", {"mean": round(st.mean(lv34), 4), "sd": round(st.stdev(lv34), 4), "n": len(lv34)},
    "v9 step losses from the graft to step 34,350 (all at 1,024 tokens)")
ll = [d["loss"] for d in s2048]
put("loss.v9_long", {"n": len(ll), "first5": round(st.mean(ll[:5]), 4), "last5": round(st.mean(ll[-5:]), 4),
                     "first_step": s2048[0]["step"], "last_step": s2048[-1]["step"],
                     "lr_first": round(s2048[0]["lr"], 4), "lr_last": round(s2048[-1]["lr"], 4),
                     "chat_frac_first": s2048[0].get("chat_frac"), "chat_frac_last": s2048[-1].get("chat_frac")},
    "v9 step losses of phase long (2,048 tokens, p 0.67-0.92): means of the first and last five records")
lt = [d["loss"] for d in s_tail]
put("loss.v9_tail", {"n": len(lt), "first5": round(st.mean(lt[:5]), 4), "last5": round(st.mean(lt[-5:]), 4),
                     "first_step": s_tail[0]["step"], "last_step": s_tail[-1]["step"],
                     "lr_first": round(s_tail[0]["lr"], 4), "lr_last": round(s_tail[-1]["lr"], 4)},
    "v9 step losses of phase tail (p >= 0.92; loss on assistant tokens only, not comparable with earlier records)")
lf = [d["loss"] for d in f if 0.45 <= d["p"] < 0.50]
put("loss.final_p045_050", {"mean": round(st.mean(lf), 4), "sd_sample": round(st.stdev(lf), 4),
                            "n": len(lf), "min": round(min(lf), 4), "max": round(max(lf), 4)},
    "final step losses with 0.45 <= p < 0.50 (sample sd)")
first = [d["loss"] for d in f if d["p"] < 0.01]
put("loss.final_p000_001", {"mean": round(st.mean(first), 3), "sd_sample": round(st.stdev(first), 3),
                            "sd_population": round(st.pstdev(first), 3), "n": len(first)},
    "final step losses with p < 0.01")
put("records.counts", {"final": len(f), "v9": len(v), "v9_first_step": v[0]["step"], "v9_last_step": v[-1]["step"]},
    "step records of the two logs (one every 50 steps)")

# ---- p and learning-rate factor at the graft and after; the tail's stage event
g_end = events(FLOG, "end")[-1]
v_stage = events(VLOG, "stage")
put("p.graft", {"final_end_event": round(g_end["p"], 4), "v9_stage_event": round(v_stage[0]["p"], 4),
                "last_final_step_record": round(f[-1]["p"], 4), "first_v9_step_record": round(v[0]["p"], 4)},
    "p at the stop (end event of step 29,722), at the restore (stage event), and of the nearest step records")
put("v9.stage_events", [{"step": e["step"], "p": round(e["p"], 4), "phase": e.get("phase"), "utc": utc(e["time"]),
                         "assistant_only": e.get("assistant_only")} for e in v_stage],
    "stage events of the v9 log (restores and the phase changes at p = 0.67 and 0.92)")

# ---- token projections of the third pass (from the record of 16:27), set against the ledger
num3 = json.loads((N / "paper/lead3_work/numbers.json").read_text())
pv3 = num3["proj.v9_total_from_newest"]["value"]
pf3 = num3["proj.final_total"]["value"]
put("proj.third_pass_vs_ledger", {"projected_tokens": pv3["total_tokens"], "projected_steps": pv3["total_steps"],
                                  "ledger_tokens": led["tokens"], "ledger_steps": led["steps"],
                                  "error_tokens": led["tokens"] - pv3["total_tokens"],
                                  "error_share": round((led["tokens"] - pv3["total_tokens"]) / led["tokens"], 5),
                                  "final_counterfactual_tokens": pf3["total_tokens"],
                                  "deficit_vs_final_counterfactual": float(f"{pf3['total_tokens'] - led['tokens']:.3g}"),
                                  "deficit_share": round((pf3["total_tokens"] - led["tokens"]) / pf3["total_tokens"], 3)},
    "the third pass's projection of the run's tokens (lead3_work/numbers.json proj.v9_total_from_newest) against the "
    "ledger; final's counterfactual total (proj.final_total, a projection with a scaled 2,048-token rate) minus the ledger")
put("proj.v9_scratch_at_graft", float(f"{30969 * 0.67 * r('v9_switches_1024') + 30969 * 0.33 * r('v9_switches_2048'):.3g}"),
    "a v9 started from scratch in the 30,969 s left at the graft, at the rates with the switches "
    "(optimistic: the switches did not exist at 09:45)")
put("tokens.gap_to_opponents", {"falcon_factor": round(8e11 / led["tokens"]), "mobilellm_factor": round(4.2e12 / led["tokens"])},
    "Falcon-H1-Tiny-90M-Instruct's 8e11 and MobileLLM-R1's 4.2e12 pre-training tokens over ours (ledger)")

# ---- statistics of the comparisons
from statistics import NormalDist  # noqa: E402


def n_required(delta, tie, alpha=0.05, power=0.8, p0=0.5):
    p1 = p0 + delta
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    n_dec = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / delta) ** 2
    return math.ceil(n_dec / (1 - tie))


def mde(n, tie):
    lo, hi = 1e-4, 0.4999
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_required(mid, tie) > n:
            lo = mid
        else:
            hi = mid
    return hi


for name, tie in (("falcon_dev", 15 / 60), ("mobilellm_dev", 37 / 60), ("protocol_30pct", 0.30),
                  ("falcon_dev_v9_two_readings", 18 / 60), ("mobilellm_dev_v9_two_readings", 33 / 60),
                  ("falcon_dev_v9_qwen_judge", 43 / 60), ("mobilellm_dev_v9_qwen_judge", 51 / 60),
                  ("falcon_test_opus", 57 / 200), ("mobilellm_test_opus", 69 / 200), ("prefopt_vs_trained_test_opus", 107 / 200),
                  # the registered judge on test, full passes: against Falcon 127 of 200 undecided (22:11 UTC),
                  # against MobileLLM 136 of 200 (22:35 UTC)
                  ("falcon_test_qwen", 127 / 200), ("mobilellm_test_qwen", 136 / 200)):
    put(f"power.{name}", {"tie_rate": round(tie, 4), "n_for_10pt": n_required(0.10, tie),
                          "n_for_5pt": n_required(0.05, tie),
                          "mde_decisive_at_n200": round(mde(200, tie), 3),
                          "mde_decisive_at_n189": round(mde(189, tie), 3)},
        "smallest true decisive win rate above 0.5 detected with 80% power by a two-sided 5% test "
        "(bes.eval.proxy.score.n_required, normal approximation, inflated by 1/(1 - tie rate))")
put("power.score_halfwidth_n200", round(1.96 * math.sqrt(0.25 / 200), 3), "normal half-width of a score near 0.5 at n = 200")

# ---- the held-out three-point extrapolation (review item 35; midrun_eval_1.json, heldout_v9.jsonl)
x = [494.7, 519.7, 538.7]
y = [2.2679438350868595, 2.2624340930463993, 2.25984169013143]
xm, ym = sum(x) / 3, sum(y) / 3
sxx = sum((a - xm) ** 2 for a in x)
b = sum((a - xm) * (c - ym) for a, c in zip(x, y)) / sxx
a0 = ym - b * xm
res = [c - (a0 + b * a) for a, c in zip(x, y)]
s = math.sqrt(sum(e * e for e in res) / 1)
x0 = 592.5
pred = a0 + b * x0
sep = s * math.sqrt(1 + 1 / 3 + (x0 - xm) ** 2 / sxx)
put("heldout.three_point", {"slope_per_min": round(b, 6), "pred_at_592.5": round(pred, 4), "v9_at_592.5": 2.2337,
                            "gap": round(pred - 2.2337, 4), "pi95_halfwidth_t1": round(12.706 * sep, 3)},
    "least-squares line through final's three held-out losses (all rows; training minutes 494.7, 519.7, 538.7), "
    "95% prediction interval with t(1 dof) = 12.706 (review1_work/stats.py); kept as the old computation, not cited")

# ---- FLOP per token (review item 25), conventions of lead_work/scripts/flops.py
V, d, nb, R, H, hd, bias, dff, rank, T = 8192, 768, 2, 8, 6, 128, 2, 3072, 512, 1024
apps = nb * R
head = d * rank + rank * V
mlp = 2 * d * dff
attn_an = d * H * (hd - bias) * 2 + d * H * hd + H * hd * d
attn_ex = 2 * (2 * d * H * hd) + d * H * hd + H * hd * d
full_sq = 2 * T * hd * H
causal = 2 * ((T + 1) / 2) * hd * H
analytic = 3 * (apps * (attn_an + mlp + causal) + head)
onetile = 0.625 * full_sq
ex = {m: 4 * apps * (attn_ex + mlp) + (1 + m) * apps * onetile + 3 * head for m in (2.0, 2.5)}
rate = r("final_25300_29700")
put("flop.final_1024", {"analytic_GFLOP_per_token": round(2 * analytic / 1e9, 3),
                        "executed_GFLOP_per_token": [round(2 * ex[2.0] / 1e9, 3), round(2 * ex[2.5] / 1e9, 3)],
                        "live_rate": rate,
                        "analytic_TFLOP_s": round(2 * analytic * rate / 1e12),
                        "executed_TFLOP_s": [round(2 * ex[2.0] * rate / 1e12), round(2 * ex[2.5] * rate / 1e12)],
                        "executed_share_of_814": [round(2 * ex[2.0] * rate / 814e12, 3), round(2 * ex[2.5] * rate / 814e12, 3)],
                        "analytic_share_of_814": round(2 * analytic * rate / 814e12, 3)},
    "analytic: forward + backward = 3 x forward multiply-adds, causal attention, no recomputation; "
    "executed: the loop body's products 4 x (forward, rematerialised forward, two backward), the rotation's "
    "q and k as two products each, attention over the 62.5% of the score square the one-tile strips cover, "
    "its forward once (save) and its backward 2.0 or 2.5 times the forward, the head 3 x; FLOP = 2 x MAC")
put("loops.time", {"one_loop_ms": 42.9, "eight_loops_ms": 289.0,
                   "per_loop_ms": round((289.0 - 42.9) / 7, 1),
                   "non_block_ms_estimate": round(42.9 - (289.0 - 42.9) / 7, 1)},
    "TPU_NOTES l.202: one loop 42.9 ms against 289 ms for eight, 32 x 1,024 micro-batch, reference code")
(W / "numbers.json").write_text(json.dumps(out, indent=1))
for k, xv in out.items():
    print(k, json.dumps(xv["value"])[:1200])
