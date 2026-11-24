"""Every derived number the third draft of paper.md cites from the train logs, the chip outputs,
the dev reports and the review's arithmetic, recomputed from the raw files and written to
numbers.json (review item 50). Arithmetic on measured inputs; each entry names its inputs.

Third pass (lead3_work, copied from lead2_work/make_numbers.py): the default v9 log is the one
extracted from the snapshot of 16:28 UTC (v9_snap_1628/, newest step record 46,350); the 2,048-token
stretch is restricted to the trainer's phase "long" and gets its achieved median and last step;
new entries: v9.newest_ledger, proj.v9_total_from_newest (projection from the newest record to the
training end of the run's own setup event), proj.final_total (final's counterfactual total recomputed
from its live median instead of the first draft's hard-coded 1.588e10).

    python3 make_numbers.py [V9_TRAIN_LOG]      (default: the newest extracted v9 log)
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = HERE.parents[1]
FLOG = N / "paper/lead_work/final_snap/run/train/train_log.jsonl"
VLOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "v9_snap_1628/run/train/train_log.jsonl"
VLED = VLOG.parent / "ledger.json"
T0 = 1795455991.9461858          # train t0 of run main16 (ledger 't0'; 23 Nov 17:46:31.946 UTC)
out = {}


def put(key, value, how):
    out[key] = {"value": value, "how": how}


def steps(path):
    return [d for d in map(json.loads, open(path)) if d.get("event") == "step"]


f = steps(FLOG)
v = steps(VLOG)
allsteps = f + v


def utc(t):
    import datetime as dt
    return dt.datetime.fromtimestamp(T0 + t, dt.timezone.utc).strftime("%H:%M:%S")


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
    rates = [d["tok_per_s"] for d in s]
    ach = achieved(s)
    if not rates:
        continue
    put(f"rate.{k}", {"median_tok_per_s": round(st.median(rates)), "n": len(rates),
                      "min": round(min(rates)), "max": round(max(rates)),
                      "achieved_median": round(st.median(ach)) if ach else None, "achieved_n": len(ach),
                      "first_step": s[0]["step"], "last_step": s[-1]["step"],
                      "first_utc": utc(s[0]["time"]), "last_utc": utc(s[-1]["time"])},
        f"{what}: median of the train log's per-step tok_per_s over the step records with "
        f"{lo} < step <= {hi} at 1,024 tokens; achieved = 262,144 x 50 / time between consecutive records")
s2048 = [d for d in v if d.get("seq_len") == 2048 and d.get("phase", "long") == "long"]
if s2048:
    rates = [d["tok_per_s"] for d in s2048]
    ach = achieved(s2048)
    put("rate.v9_switches_2048", {"median_tok_per_s": round(st.median(rates)), "n": len(rates),
                                  "min": round(min(rates)), "max": round(max(rates)),
                                  "achieved_median": round(st.median(ach)) if ach else None, "achieved_n": len(ach),
                                  "first_step": s2048[0]["step"], "first_utc": utc(s2048[0]["time"]),
                                  "last_step": s2048[-1]["step"], "last_utc": utc(s2048[-1]["time"]),
                                  "drop_vs_1024": None},
        "v9 with the switches at 2,048 tokens (from p = 0.67; trainer phase 'long' only)")
phases = sorted({d.get("phase") for d in v})
put("v9.phases_seen", phases, "trainer phases in the v9 step records (short = 1,024 tokens, long = 2,048, tail = p >= 0.92)")

r = lambda k: out[f"rate.{k}"]["value"]["median_tok_per_s"]  # noqa: E731
put("ratio.kernel_gain_live", round(r("kernel_to_deflection") / r("prekernel_21000") - 1, 4),
    "median after the kernel switch (to 25,280) / median of the 1,600 steps before, minus 1")
put("ratio.final_over_v9_take_live", round(r("final_25300_29700") / r("v9_take"), 4),
    "final 25,300-29,700 median / v9 (take) median")
put("ratio.final_over_v9_switches_live", round(r("final_25300_29700") / r("v9_switches_1024"), 4),
    "final 25,300-29,700 median / v9 with the switches median (1,024 tokens)")
put("ratio.v9_switches_gain_live", round(r("v9_switches_1024") / r("v9_take") - 1, 4),
    "v9 with the switches / v9 as grafted, live medians, minus 1")
if s2048:
    out["rate.v9_switches_2048"]["value"]["drop_vs_1024"] = round(r("v9_switches_2048") / r("v9_switches_1024") - 1, 4)
put("ratio.reference_to_kernel_live", [round(r("final_25300_29700") / 116718, 3), round(r("final_25300_29700") / 115394, 3)],
    "final's live median after the kernel (steps 25,300-29,700) / the two monitor readings of the reference "
    "implementation on runtime 1 (116,718 and 115,394 tokens/s, S/colab_bes24h_final.log 04:17:59 and 04:28:00); "
    "third pass: the second pass divided by 117,000 and 112,000 (TPU_NOTES' 112-117k, which mixes the start-up "
    "benchmark 111,893 with the monitor readings)")
# TPU time at the newest ledger mark (S/tpu_intervals.txt: runtime 1 and 2 closed, runtime 3 open)
iv = [l.split() for l in open("/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/tpu_intervals.txt")]
led_ = json.loads((VLOG.parent / "ledger.json").read_text())
secs = []
for parts in iv:
    a = float(parts[0]); b = float(parts[1]) if parts[1] != "-" else led_["mark"]
    secs.append(round(b - a))
put("tpu.at_newest", {"seconds_by_runtime": secs, "total_s": sum(secs), "total_h": round(sum(secs) / 3600, 2),
                      "train_s_ledger": round(led_["seconds"]["train"], 1),
                      "train_h": round(led_["seconds"]["train"] / 3600, 2),
                      "train_share": round(led_["seconds"]["train"] / sum(secs), 3), "at_utc": utc(led_["mark"] - T0)},
    "TPU seconds held per runtime (S/tpu_intervals.txt; the open third interval counted to the ledger's mark) and "
    "the share in training steps of the kept trajectory (ledger 'train' seconds)")

# ---- losses
lv34 = [d["loss"] for d in v if d["step"] <= 34350]
put("loss.v9_to_34350", {"mean": round(st.mean(lv34), 4), "sd": round(st.stdev(lv34), 4), "n": len(lv34)},
    "v9 step losses from the graft to step 34,350 (all at 1,024 tokens)")
lv = [d["loss"] for d in v]
put("loss.v9_all", {"mean": round(st.mean(lv), 4), "sd": round(st.stdev(lv), 4), "n": len(lv),
                    "first_step": v[0]["step"], "last_step": v[-1]["step"]},
    "v9 step losses (each the mean over a 262,144-token step), every logged step after the graft")
lf = [d["loss"] for d in f if 0.45 <= d["p"] < 0.50]
put("loss.final_p045_050", {"mean": round(st.mean(lf), 4), "sd_sample": round(st.stdev(lf), 4),
                            "n": len(lf), "min": round(min(lf), 4), "max": round(max(lf), 4)},
    "final step losses with 0.45 <= p < 0.50 (sample sd)")
first = [d["loss"] for d in f if d["p"] < 0.01]
put("loss.final_p000_001", {"mean": round(st.mean(first), 3), "sd_sample": round(st.stdev(first), 3),
                            "sd_population": round(st.pstdev(first), 3), "n": len(first)},
    "final step losses with p < 0.01")

# ---- p and learning-rate factor at the graft and after
g_end = [d for d in map(json.loads, open(FLOG)) if d.get("event") == "end"][-1]
v_stage = [d for d in map(json.loads, open(VLOG)) if d.get("event") == "stage"]
put("p.graft", {"final_end_event": round(g_end["p"], 4), "v9_stage_event": round(v_stage[0]["p"], 4),
                "last_final_step_record": round(f[-1]["p"], 4), "first_v9_step_record": round(v[0]["p"], 4)},
    "p at the stop (end event of step 29,722), at the restore (stage event), and of the nearest step records")
put("v9.newest_step", {"step": v[-1]["step"], "p": round(v[-1]["p"], 4), "utc": utc(v[-1]["time"]),
                       "loss": round(v[-1]["loss"], 4), "seq_len": v[-1].get("seq_len"), "phase": v[-1].get("phase"),
                       "lr": round(v[-1]["lr"], 4)}, "the newest step record of the extracted v9 log")
led = json.loads(VLED.read_text())
put("v9.newest_ledger", {"steps": led["steps"], "tokens": led["tokens"], "mark_utc": utc(led["mark"] - T0),
                         "train_s": round(led["seconds"]["train"], 1), "ckpt_s": round(led["seconds"]["ckpt"], 1),
                         "setup_s": round(led["seconds"]["setup"], 1), "lost_s": round(led["seconds"]["lost"], 1),
                         "restarts": led["restarts"], "rollbacks": led["rollbacks"]},
    "the trainer's time ledger in the same snapshot (steps and tokens trained, seconds by phase)")

# ---- token projections (review item 28): from the 10:57 restart
slice_s, p0, late = 27420.0, [d for d in v_stage if d["step"] == 31912][0]["p"], 0.67   # run.log 'start train: slice 27420 s'
t2048 = slice_s * (1 - late) / (1 - p0)
t1024 = slice_s - t2048
live1024 = r("v9_switches_1024")
syn1024, syn2048_take, syn1024_take = 1287.1, 1510.2, 1433.1      # ms per step, v9_measD / v9_measA
# assumption (not measured): the switches save the same fraction at 2,048 as at 1,024
syn2048 = syn2048_take * syn1024 / syn1024_take
live2048_assumed = 262144 / (syn2048 / 1000) * live1024 / (262144 / (syn1024 / 1000))
if s2048:
    live2048_meas = out["rate.v9_switches_2048"]["value"]["median_tok_per_s"]
else:
    live2048_meas = None
done = 31912 * 262144
rem = t1024 * live1024 + t2048 * (live2048_meas or live2048_assumed)
final_total = 1.588e10
put("proj.v9_total", {"t1024_s": round(t1024), "t2048_s": round(t2048), "live1024": live1024,
                      "live2048_used": round(live2048_meas or live2048_assumed),
                      "live2048_is_measured": live2048_meas is not None,
                      "remaining_tokens": float(f"{rem:.4g}"), "total_tokens": float(f"{done + rem:.4g}"),
                      "final_projected_total": final_total,
                      "deficit": float(f"{final_total - done - rem:.3g}"),
                      "deficit_share": round((final_total - done - rem) / final_total, 3)},
    "tokens at 18:33:49 projected from the 10:57:50 restart (step 31,912, p = 0.5651, slice 27,420 s, "
    "p linear in time, 2,048 tokens from p = 0.67) at the live v9 median at 1,024; at 2,048 the measured "
    "live median if it exists, else 1,510.2 x (1,287.1 / 1,433.1) ms synthetic scaled by live/synthetic at "
    "1,024; final's projected total 1.588e10 is the first draft's (lead_work/scripts/projections.py)")
# final's counterfactual total, recomputed: same slice from the graft restart (30,969 s at p = 0.5112),
# its live median at 1,024 (steps 25,300-29,700) and its 2,048 rate scaled from the synthetic steps
# (266,432 at 2,048 against 296,371 at 1,024, both with ce on; chip_kernels_0745 / tables.md T3b)
fslice, fp0 = 30969.0, 0.5112049
ft2048 = fslice * (1 - late) / (1 - fp0); ft1024 = fslice - ft2048
f1024 = r("final_25300_29700"); f2048 = 266432.0 * f1024 / 296371.0
ftot = 7791443968 + ft1024 * f1024 + ft2048 * f2048
put("proj.final_total", {"t1024_s": round(ft1024), "t2048_s": round(ft2048), "live1024": f1024,
                         "live2048_scaled": round(f2048), "total_tokens": float(f"{ftot:.4g}")},
    "final continued from the graft restart without the switch: slice 30,969 s, p linear in time, 2,048 tokens "
    "from p = 0.67; 2,048 rate = synthetic 266,432 x live/synthetic at 1,024 (a scaled estimate, not measured)")
# projection from the newest record: ledger tokens + (training end - ledger mark) x live median at 2,048
train_end = [d for d in map(json.loads, open(VLOG)) if d.get("event") == "setup"][-1]["train_end"]
rem_s = train_end - led["mark"]
vtot = led["tokens"] + rem_s * (live2048_meas or live2048_assumed)
put("proj.v9_total_from_newest", {"train_end_utc": utc(train_end - T0), "ledger_tokens": led["tokens"],
                                  "remaining_s": round(rem_s, 1), "rate_used": live2048_meas,
                                  "total_tokens": float(f"{vtot:.4g}"), "total_steps": round(vtot / 262144),
                                  "deficit_vs_final": float(f"{ftot - vtot:.3g}"),
                                  "deficit_share": round((ftot - vtot) / ftot, 3)},
    "tokens at the training end (setup event 'train_end') = the ledger's tokens at its mark + the remaining wall "
    "time x the live median at 2,048 tokens (assumes the rate holds through the p = 0.92 tail and no pause); "
    "deficit against proj.final_total")
# from-scratch v9 at the graft (first draft's question, with the rates now known)
put("proj.v9_scratch_at_graft", float(f"{30969 * 0.67 * live1024 + 30969 * 0.33 * (live2048_meas or live2048_assumed):.3g}"),
    "a v9 started from scratch in the 30,969 s left at the graft, at the rates with the switches "
    "(optimistic: the switches did not exist at 09:45)")

# ---- statistics of the dev comparisons (review item 32)
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
                  ("falcon_dev_v9_qwen_judge", 43 / 60), ("mobilellm_dev_v9_qwen_judge", 51 / 60)):
    put(f"power.{name}", {"tie_rate": round(tie, 4), "n_for_10pt": n_required(0.10, tie),
                          "n_for_5pt": n_required(0.05, tie),
                          "mde_decisive_at_n200": round(mde(200, tie), 3),
                          "mde_decisive_at_n189": round(mde(189, tie), 3)},
        "smallest true decisive win rate above 0.5 detected with 80% power by a two-sided 5% test "
        "(bes.eval.proxy.score.n_required, normal approximation, inflated by 1/(1 - tie rate))")
lo = (100 - 1.96 * math.sqrt(50)) / 200
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
    "least-squares line through final's three held-out losses (training minutes 494.7, 519.7, 538.7), "
    "95% prediction interval with t(1 dof) = 12.706 (review1_work/stats.py)")

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

# ---- the loop count (review item 41): prediction, not measurement
put("loops.time", {"one_loop_ms": 42.9, "eight_loops_ms": 289.0,
                   "per_loop_ms": round((289.0 - 42.9) / 7, 1),
                   "non_block_ms_estimate": round(42.9 - (289.0 - 42.9) / 7, 1)},
    "TPU_NOTES l.202: one loop 42.9 ms against 289 ms for eight, 32 x 1,024 micro-batch, reference code")
(HERE / "numbers.json").write_text(json.dumps(out, indent=1))
for k, x in out.items():
    print(k, json.dumps(x["value"]))
