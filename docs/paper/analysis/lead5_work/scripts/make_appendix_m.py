"""Fifth pass: writes Appendix M of paper.md (results on the test split and at the end of the run) from the
recomputed files of lead5_work/ only (final_results.json, paper_values.json, exact_counts.json, numbers.json,
heldout5.json) and the test generations' summaries (N/testgen/<tag>/out/testgen.json) and the official
export's export.json. No number is typed: rerun after final_results.py / paper_values.py when a reading
advances (the registered judge's pass is partial), then splice with --splice.

    python3 make_appendix_m.py [--splice]
"""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
N = W.parents[1]
P = W.parent
FR = json.loads((W / "final_results.json").read_text())["sets"]
PV = json.loads((W / "paper_values.json").read_text())
EX = json.loads((W / "exact_counts.json").read_text())
NU = json.loads((W / "numbers.json").read_text())
HO = json.loads((W / "heldout5.json").read_text())
EXP = json.loads((N / "packs/official_export/x/final/export.json").read_text())
TG = {t: json.loads((N / f"testgen/{t}/out/testgen.json").read_text()) for t in ("prefopt", "trained")}
Q = "judge-qwen3-4b-2507-3605803b"
O = "judge-opus"
QTIME = json.loads((W / "final_results.json").read_text()).get("qwen_snapshot_utc", "the time of writing")


def q_full(set_):
    """True when the registered judge has rated all 200 pairs of the set."""
    e = PV.get(set_, {}).get(Q)
    return bool(e) and e["all"]["n"] == 200


def q_lab(set_, form):
    full = q_full(set_)
    return {"m1": "`qwen3-4b-2507` (registered), " + ("full pass" if full else "partial"),
            "col": "`qwen3-4b-2507`" + ("" if full else " (partial)"),
            "short": "`qwen3-4b-2507`" + ("" if full else ", partial")}[form]


_QF, _QM = q_full("final-prefopt-vs-falcon"), q_full("final-prefopt-vs-mobilellm")
QNOTE = ("The registered judge's pass is complete on both official sets: its rows give the pairs rated in its "
         "files as read at " + QTIME + ".") if (_QF and _QM) else ("The registered judge's pass is " + ("complete on the Falcon-H1-Tiny set" if _QF else "partial on the Falcon-H1-Tiny set")
         + (" and complete on the MobileLLM-R1.5 set" if _QM else " and partial on the MobileLLM-R1.5 set")
         + ": its rows give the pairs rated in its files as read at " + QTIME
         + ("" if (_QF and _QM) else ", and its full pass on the " + ("MobileLLM-R1.5 set" if _QF else "Falcon-H1-Tiny set" if _QM else "two sets")
            + " is [TBD: running on CPU runtimes since 20:41 UTC; about 22:40 UTC by the operator's estimate]")
         + ".")


def v(set_, src):
    return PV.get(set_, {}).get(src)


def wlt(e):
    return e["W/L/T"] if e else "–"


def m1_row(label, reader, set_, src, partial=False):
    e = v(set_, src)
    if not e:
        return None
    a = e["all"]
    n = f"{a['n']} of 200" if partial else str(a["n"])
    return (f"| {label} | {reader} | {n} | {a['W/L/T']} | {a['score']} {a['score_ci']} | "
            f"{a.get('decisive', '–')} {a.get('decisive_ci', '')} | {a['p']} | {e.get('orders_agree_pct', '–')}% | "
            f"{e.get('first_pos', '–').replace('-', '−')} |")


L = []
add = L.append
add("## Appendix M. Results on the test split and at the end of the run")
add("")
add("""The shells of the third draft, filled. Every value is recomputed from the records named for its table by
the scripts of `lead5_work/scripts/` (this appendix is written by `make_appendix_m.py` from their outputs; no
number is typed). Conventions as in Table 7: W / L / T from our side (T includes "both bad" and pairs whose
two orders disagree), score (wins + ties/2)/n with its 95% Wilson interval, the decisive win rate wins /
(wins + losses) with its interval, exact two-sided sign test; values rounded once from full precision, a value
exactly on a half to even. R = `N/study/proxy/reports/ab/`. Our official run is
`ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0` (TAG = `prefopt`: its export has the
run's own `weights16.npz` and decoding choice), the opponents' runs `falcon-e60034381f4a/recommended-5ccec0259b-s0`
and `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`. """ + QNOTE + """ The second registered judge
(`gemma4-e4b`) was not run on the test pairs, so its rows of the shells are gone (Section 7.2, deviation 1).""")
add("")
add("**Table M1 (R1). The test split, 200 prompts: each opponent and each reader** (`R/<set>--<source>.json`, "
    "fields `overall` and `judge`, reproduced by `final_results.py`; sets `final-prefopt-vs-falcon`, "
    "`-vs-falcon-k6lp`, `-vs-falcon-k6ds`, `-vs-mobilellm`, `ref-falcon-vs-mobilellm-test`).")
add("")
add("| opponent | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders agree | first-position preference |")
add("|---|---|---|---|---|---|---|---|---|")
rows = [
    m1_row("Falcon-H1-Tiny-90M-Instruct", q_lab("final-prefopt-vs-falcon", "m1"), "final-prefopt-vs-falcon", Q, not q_full("final-prefopt-vs-falcon")),
    m1_row("Falcon-H1-Tiny-90M-Instruct", "Opus, two readings", "final-prefopt-vs-falcon", O),
    m1_row("Falcon-H1-Tiny-90M-Instruct", "Opus, two readings, neutral titles", "final-prefopt-vs-falcon", "judge-opus-neutral"),
    m1_row("Falcon-H1-Tiny, K = 6 kept by its own log-probability", "Opus, two readings", "final-prefopt-vs-falcon-k6lp", O),
    m1_row("Falcon-H1-Tiny, K = 6 kept by our selector", "Opus, two readings", "final-prefopt-vs-falcon-k6ds", O),
    m1_row("MobileLLM-R1.5-140M", q_lab("final-prefopt-vs-mobilellm", "m1"), "final-prefopt-vs-mobilellm", Q, not q_full("final-prefopt-vs-mobilellm")),
    m1_row("MobileLLM-R1.5-140M", "Opus, two readings", "final-prefopt-vs-mobilellm", O),
    m1_row("MobileLLM-R1.5-140M", "Opus, two readings, neutral titles", "final-prefopt-vs-mobilellm", "judge-opus-neutral"),
]
for r in rows:
    if r:
        add(r)
add("| MobileLLM-R1.5, K = 6 (both rules) | – | [TBD: replies in generation since 19:17 UTC, 40 of 200 prompts by 21:13; expected after midnight] | | | | | | |")
k1f = v("final-k1-vs-falcon", O)
k1m = v("final-k1-vs-mobilellm", O)
add(m1_row("Falcon-H1-Tiny, against our K = 1 arm", "Opus, two readings", "final-k1-vs-falcon", O) if k1f else
    "| Falcon-H1-Tiny, against our K = 1 arm | Opus, two readings | [TBD: readings started at 21:23 UTC] | | | | | | |")
add(m1_row("MobileLLM-R1.5, against our K = 1 arm", "Opus, two readings", "final-k1-vs-mobilellm", O) if k1m else
    "| MobileLLM-R1.5, against our K = 1 arm | Opus, two readings | [TBD: readings started at 21:23 UTC] | | | | | | |")
add(m1_row("reference: Falcon-H1-Tiny vs MobileLLM-R1.5", "Opus, two readings", "ref-falcon-vs-mobilellm-test", O))
add("")
# M2
add("**Table M2 (R2). By category and by turns** (W / L / T; the same files, fields `by_category` and `by_turns`; "
    "prompts of the `test` split).")
add("")
cols = [("Falcon-H1-Tiny: Opus", "final-prefopt-vs-falcon", O), ("Falcon-H1-Tiny: " + q_lab("final-prefopt-vs-falcon", "col"), "final-prefopt-vs-falcon", Q),
        ("Falcon, K = 6 own log-prob.: Opus", "final-prefopt-vs-falcon-k6lp", O), ("Falcon, K = 6 our selector: Opus", "final-prefopt-vs-falcon-k6ds", O),
        ("MobileLLM-R1.5: Opus", "final-prefopt-vs-mobilellm", O), ("MobileLLM-R1.5: " + q_lab("final-prefopt-vs-mobilellm", "col"), "final-prefopt-vs-mobilellm", Q),
        ("ours K = 1 vs Falcon: Opus", "final-k1-vs-falcon", O), ("ours K = 1 vs MobileLLM: Opus", "final-k1-vs-mobilellm", O)]
cols = [c for c in cols if v(c[1], c[2])]
add("| slice (prompts) | " + " | ".join(c[0] for c in cols) + " |")
add("|---|" + "---|" * len(cols))
for key, lab in (("chat", "chat (50)"), ("code", "code (30)"), ("instruct", "instruction-following (40)"), ("math", "maths (30)"),
                 ("writing", "writing (50)")):
    add(f"| {lab} | " + " | ".join(wlt((v(s, src) or {}).get("by_category", {}).get(key)) for _, s, src in cols) + " |")
for key, lab in (("single_turn", "single-turn (148)"), ("multi_turn", "two-turn (52)")):
    add(f"| {lab} | " + " | ".join(wlt((v(s, src) or {}).get("by_turns", {}).get(key)) for _, s, src in cols) + " |")
add("")
# M3
add("**Table M3 (R3). Length buckets and reader diagnostics** (the same files, fields `by_length` (ratio of our "
    "words to the opponent's, +1 each: shorter below 2/3, longer above 3/2) and `judge`; the Opus records carry "
    "no probability mass and no clipping, since the readers read the chunk files).")
add("")
add("| row | " + " | ".join(c[0] for c in cols) + " |")
add("|---|" + "---|" * len(cols))
for key, lab in (("x_shorter", "ours shorter: n; W / L / T"), ("similar", "similar: n; W / L / T"), ("x_longer", "ours longer: n; W / L / T")):
    cells = []
    for _, s, src in cols:
        e = (v(s, src) or {}).get("by_length", {}).get(key)
        cells.append(f"{e['n']}; {e['W/L/T']}" if e and e.get("n") else "0")
    add(f"| {lab} | " + " | ".join(cells) + " |")
add("| orders agree | " + " | ".join(f"{(v(s, src) or {}).get('orders_agree_pct', '–')}%" for _, s, src in cols) + " |")
add("| first-position preference | " + " | ".join((v(s, src) or {}).get("first_pos", "–").replace("-", "−") for _, s, src in cols) + " |")
add("| probability mass on A, B, C (field `judge.format_mass_mean`) | " + " | ".join(
    (f"{FR[s]['sources'][src]['judge']['format_mass_mean']:.3f}" if src == Q else "–") for _, s, src in cols) + " |")
add("| pairs with a reply cut for the judge (over 4,000 characters) | " + " | ".join(
    (str(FR[s]['sources'][src]['judge']['clipped']) if src == Q else "–") for _, s, src in cols) + " |")
add("")
# M4
add("**Table M4 (R4). Exact scores on the 200 test prompts (252 turns), in full** (`bes.eval.exact_score` on every "
    "reply, by `lead4_work/scripts/exact_split.py --split test` with every run below, `lead5_work/exact_test.json`; "
    "equal field by field to the run's report in `study/proxy/reports/exact/` where one exists, which is every run "
    "but MobileLLM-R1.5's `neutral` arm; counts from `lead5_work/exact_counts.json`).")
add("")
runs = [("ours", "ours, official"), ("ours_k1", "ours, K = 1"), ("trained", "ours, trained weights"), ("falcon", "Falcon-H1-Tiny"),
        ("falcon_k6lp", "Falcon K = 6, own log-prob."), ("falcon_k6ds", "Falcon K = 6, our selector"),
        ("mobilellm_recommended", "MobileLLM-R1.5 `recommended`"), ("mobilellm_neutral", "MobileLLM-R1.5 `neutral`")]
E = EX["200"]
add("| measure | " + " | ".join(l for _, l in runs) + " |")
add("|---|" + "---|" * len(runs))
add("| maths answers correct (of 30) | " + " | ".join(str(E[k]["maths_correct"]) for k, _ in runs) + " |")
add("| constrained prompts with every constraint met (of 48) | " + " | ".join(str(E[k]["constraints_all_ok"]) for k, _ in runs) + " |")
add("| single constraints met (of 56) | " + " | ".join(str(E[k]["single_ok"]) for k, _ in runs) + " |")
add("| degenerate replies (of 252) | " + " | ".join(str(E[k]["degenerate"]) for k, _ in runs) + " |")
add("| … cut at the token limit | " + " | ".join(str(E[k]["truncated"]) for k, _ in runs) + " |")
add("| … looping tail; repeated lines | " + " | ".join(f"{E[k]['loop']}; {E[k]['repeated_lines']}" for k, _ in runs) + " |")
add("| … empty; unclosed code fence; role markers | " + " | ".join(f"{E[k]['empty']}; {E[k]['unclosed_fence']}; {E[k]['role_leak']}" for k, _ in runs) + " |")
add("| stock refusals; stock deflections (of 252) | " + " | ".join(f"{E[k]['stock']['refusal']}; {E[k]['stock']['deflection']}" for k, _ in runs) + " |")
add("| words per reply: median; mean | " + " | ".join(f"{E[k]['words']['median']}; {E[k]['words']['mean']}" for k, _ in runs) + " |")
add("| mean repeated-4-gram rate | " + " | ".join(f"{E[k]['rep4_mean']:.3f}" for k, _ in runs) + " |")
add("")
add("A reply can be degenerate in several ways at once, so the sub-rows do not add up.")
add("")
# M5
pt = v("final-prefopt-vs-trained", O)
add("**Table M5 (R5). The trained weights against the weights after preference optimisation** (the direct pair "
    "set `final-prefopt-vs-trained`, both readings, `R/final-prefopt-vs-trained--judge-opus.json`; exact scores "
    "from Table M4; held-out losses from `lead5_work/heldout5.json`; the in-run checks from Table C5). The pair "
    "sets `final-trained-vs-falcon` and `-vs-mobilellm` were published at 19:07 UTC and not read: the direct set "
    "answers the question on the same prompts.")
add("")
add("| measure | after preference optimisation (official) | trained weights | difference |")
add("|---|---|---|---|")
add(f"| blind, Opus, two readings, 200 prompts: W / L / T (official's side) | {pt['all']['W/L/T']} | | score {pt['all']['score']} {pt['all']['score_ci']}, p = {pt['all']['p']} |")
add(f"| … the 189 untouched prompts | {pt['189']['W/L/T']} | | score {pt['189']['score']} {pt['189']['score_ci']}, p = {pt['189']['p']} |")
add("| … by category: chat; code; instruction-following; maths; writing | " +
    "; ".join(pt["by_category"][c]["W/L/T"] for c in ("chat", "code", "instruct", "math", "writing")) + " | | |")
for lab, key in (("maths answers correct (of 30)", "maths_correct"), ("every constraint met (of 48)", "constraints_all_ok"),
                 ("single constraints met (of 56)", "single_ok"), ("degenerate replies (of 252)", "degenerate")):
    add(f"| {lab} | {E['ours'][key]} | {E['trained'][key]} | {E['ours'][key] - E['trained'][key]:+d} |")
add(f"| words per reply, mean | {E['ours']['words']['mean']} | {E['trained']['words']['mean']} | |")
pp = HO["paired"]["post - train (51812)"]
add(f"| held-out loss, web + chat (64 + 64 rows) | {HO['local']['v9_51812_post_on']['web_chat']:.4f} | {HO['local']['v9_51812_train_on']['web_chat']:.4f} | "
    f"{pp['web+chat']['estimate']:+.4f} [{pp['web+chat']['ci95'][0]:.4f}, {pp['web+chat']['ci95'][1]:.4f}] |")
add(f"| held-out loss, chat assistant tokens | {HO['local']['v9_51812_post_on']['chat_assistant']:.4f} | {HO['local']['v9_51812_train_on']['chat_assistant']:.4f} | "
    f"{pp['chat_assistant']['estimate']:+.4f} [{pp['chat_assistant']['ci95'][0]:.4f}, {pp['chat_assistant']['ci95'][1]:.4f}] |")
pref = NU["prefopt"]["value"]
c3 = [c for c in pref["candidates"] if c["lr"] == 3e-6][0]
add(f"| in-run: held-out preference accuracy (1,826 pairs); assistant-token loss of the stage's own rows | {c3['acc']:.3f}; {c3['heldout_loss']:.4f} | "
    f"{pref['start']['acc']:.3f}; {pref['start']['heldout_loss']:.4f} | accuracy {c3['acc'] - pref['start']['acc']:+.3f}; loss {c3['heldout_loss_rise_pct']:+.2f}% |")
add("")
# M6
add("**Table M6 (R6). Held-out losses at the end of training and of the export** (nats per token). (a) The "
    "comparison set of Section 6.5 (64 web and 64 chat rows): the runtime's evaluation of the last checkpoint "
    "(`N/heldout_v9.jsonl`) and our CPU evaluations of both weights of step 51,812 from `N/snap_final` "
    "(`heldout_eval5.py`; bits per byte as `lead4_work/scripts/bpb_web.py` computes them, in `heldout5.py`). (b) "
    "The export's own 16-bit check on 16 rows per domain (`check` of `final/export.json`). The proxy rows are in "
    "neither part (Section 7.2, deviation 3).")
add("")
r51, L5 = HO["runtime"]["51812"], HO["local"]
add("| (a) 64 rows per domain | trained weights, step 51,812 (runtime) | after preference optimisation (local) | `final` 29,722 (the graft) |")
add("|---|---|---|---|")
for lab, k in (("web", "web"), ("chat", "chat"), ("chat, assistant tokens", "chat_assistant"), ("web + chat (comparison set)", "web_chat")):
    add(f"| {lab} | {r51[k]:.4f} | {L5['v9_51812_post_on'][k]:.4f} | {L5['final_29722'][k]:.4f} |")
add(f"| bits per byte, web rows (opponents: 1.570 and 1.500) | {HO['bpb_web']['v9_51812_train_on']:.3f} | {HO['bpb_web']['v9_51812_post_on']:.3f} | {HO['bpb_web']['final_29722']:.3f} |")
add("")
ck = EXP["check"]
add("| (b) export check, 16 rows per domain | float32 weights | 16-bit weights | difference |")
add("|---|---|---|---|")
for d in ("web", "chat"):
    b = ck["by_domain"][d]
    add(f"| {d} ({b['weight']:,.0f} weighted tokens) | {b['loss32']:.5f} | {b['loss16']:.5f} | {b['delta']:+.6f} |")
add(f"| all three domains, proxy included ({ck['weighted_tokens']:,.0f}) | {ck['loss32']:.5f} | {ck['loss16']:.5f} | {ck['delta']:+.6f} (tolerance {ck['tol']}) |")
add("")
# M7, M8
add("**Table M7 (R7). What the preference-optimisation stage did** (`run/stages.json`, `run/post/prefopt_report.json` "
    "and the run log in `N/snap_final`; Table C5 gives the candidates in full).")
add("")
stg = NU["stages"]["value"]["rows"]
add("| quantity | value |")
add("|---|---|")
add(f"| minutes used of 60 | {stg['prefopt']['used_s'] / 60:.1f} ({stg['prefopt']['started_utc'][7:]}–{stg['prefopt']['finished_utc'][7:]} UTC) |")
pb = pref["pairs_by_source"]
add(f"| preference pairs used | {pref['n_train_pairs']:,} for training and {pref['n_heldout_pairs']:,} held out by prompt: "
    f"{pb['ultrafeedback']['kept']:,} UltraFeedback pairs and {pb['sampled']['kept']} of our own sampled pairs |")
add(f"| learning rates tried and steps run at each | 1 × 10⁻⁶ and 3 × 10⁻⁶, {pref['steps_per_epoch']} steps each (one epoch), "
    f"{pref['candidates'][0]['train_time_s']:.0f} s and {pref['candidates'][1]['train_time_s']:.0f} s of training |")
add(f"| held-out preference accuracy before / after (standard error) | {pref['start']['acc']:.3f} / {pref['candidates'][0]['acc']:.3f} "
    f"({pref['candidates'][0]['acc_se']:.3f}) and {pref['candidates'][1]['acc']:.3f} ({pref['candidates'][1]['acc_se']:.3f}) |")
add(f"| held-out assistant loss change (limit +3%) | {pref['candidates'][0]['heldout_loss_rise_pct']:+.2f}% and {pref['candidates'][1]['heldout_loss_rise_pct']:+.2f}% |")
add(f"| sampled length and repetition against their bands | length {pref['candidates'][0]['mean_len_ratio']:.2f} and "
    f"{pref['candidates'][1]['mean_len_ratio']:.2f} × the start's (band 0.7–1.6), repeated 4-grams {pref['candidates'][0]['rep4_ratio']:.2f} and "
    f"{pref['candidates'][1]['rep4_ratio']:.2f} × (at most 1.2 × + 0.01) |")
add(f"| accepted (weights kept) or rejected | both accepted; kept {pref['chosen_lr']:g}, the higher accuracy (\"{pref['reason']}\") |")
add("")
sel = NU["select"]["value"]
add("**Table M8 (R8). What the decoding selection chose** (the manifest's `decode` record; `run/stages.json`; "
    "the export's `decode_choice`).")
add("")
add("| quantity | value |")
add("|---|---|")
add("| settings compared with the default | one particle (`k1`); no length band (`noband`); no library (none was built) |")
vk, vn = sel["vs_default"]["k1"], sel["vs_default"]["noband"]
add(f"| development prompts used; paired sign test of each against the default | the {sel['n_common']} `dev` prompts; "
    f"mean exact score default {sel['scores']['default']:.3f}, `k1` {sel['scores']['k1']:.3f}, `noband` {sel['scores']['noband']:.3f}; "
    f"`k1` better on {vk['a_better']}, worse on {vk['b_better']} (p = {vk['p']:.1e}); `noband` better on {vn['a_better']}, worse on "
    f"{vn['b_better']} (p = {vn['p']:.2f}) |")
add(f"| setting kept | `{sel['chosen']}` (6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library) |")
add(f"| minutes used of 35 | {stg['select']['used_s'] / 60:.1f}, in a slice of {stg['select']['slice_s'] / 60:.1f} (the stage was given the time "
    f"preference optimisation left) |")
add("")
# M9
led = NU["v9.final_ledger"]["value"]
tpu = NU["tpu.final"]["value"]
tail = NU["rate.v9_tail_2048"]["value"]
add("**Table M9 (R9). Training totals** (the trainer's ledger `run/train/ledger.json` and train log in `N/snap_final`; "
    "TPU seconds from `S/tpu_intervals.txt`, all three intervals closed; `lead5_work/numbers.json`).")
add("")
add("| quantity | value |")
add("|---|---|")
add(f"| final step; tokens trained (`v9` part since step 29,722) | {led['steps']:,}; {led['tokens']:,} ({led['v9_tokens_since_graft']:,} in {led['v9_steps_since_graft']:,} steps) |")
add(f"| training seconds of the kept trajectory; checkpoint seconds; set-up seconds | {led['train_s']:,.1f}; {led['ckpt_s']:,.1f}; {led['setup_s']:,.1f} |")
add(f"| TPU seconds: runtime 1, runtime 2, runtime 3, total | {tpu['seconds_by_runtime'][0]:,}; {tpu['seconds_by_runtime'][1]:,}; "
    f"{tpu['seconds_by_runtime'][2]:,}; {tpu['total_s']:,} of {tpu['limit_s']:,} |")
add(f"| share of TPU time in training steps of the kept trajectory | {tpu['train_share'] * 100:.1f}% |")
add(f"| live rate in the last stretch (p ≥ 0.92, conversations train on assistant tokens only), n | {tail['median_tok_per_s']:,} tokens/s, "
    f"n = {tail['n']} (steps {tail['first_step']:,}–{tail['last_step']:,}) |")
add("")
# M10
w = EXP["weights"]
rs = EXP["readset"]["probes"]
mf = {f["path"]: f for f in EXP["main_files"]}
add("**Table M10 (R10). The 16-bit check of the export** (`final/export.json`: `check`, `weights`, `main_files`, "
    "`readset`; `SHA256SUMS`).")
add("")
add("| quantity | value |")
add("|---|---|")
add(f"| largest absolute loss difference, 16-bit against float32, web and chat (the proxy rows' difference enters only the check's pass or fail) | "
    f"{max(abs(ck['by_domain'][d]['delta']) for d in ('web', 'chat')):.1e} nats (web); passed |")
add(f"| leaves stored in bfloat16 / float16 (rule: the smaller relative L2 error; ties float16) | {w['bfloat16']['leaves']} / {w['float16']['leaves']} "
    f"({w['numbers']:,} numbers; largest relative L2 error of a leaf {w['float16']['max_rel_err']:.2e}) |")
add(f"| export size (bytes) and the SHA-256 of the weights | `weights16.npz` {mf['weights16.npz']['bytes']:,} bytes, "
    f"`{mf['weights16.npz']['sha256']}`; the packed export 196,956,160 bytes in 12 parts |")
add(f"| read-set audit of the export: numbers read per token, standard reading | ledger {EXP['accounting']['active_step']:,}; measured on held-out "
    f"sequences {rs[0]['step_mean']:,.0f} (web, mean over {rs[0]['positions']:,} positions) and {rs[1]['step_mean']:,.0f} (chat), at most "
    f"{EXP['readset']['step_max_with_retrieval']:,}; strict reading {rs[0]['strict_at']['1024']['strict']:,} at 1,024 positions and "
    f"{rs[0]['strict_at']['2048']['strict']:,} at 2,048 (web) |")
add("")
# M11
add("**Table M11 (R11). Tables M1 and M4 on the 189 test prompts no mid-run sample touched** (240 turns, 28 maths "
    "and 44 constrained prompts; the 11 seen ids of Section 7.2 left out; readers from each report's "
    "`per_prompt_score` as `lead4_work/scripts/restrict_report.py` does, in `final_results.py`; exact scores by "
    "`exact_split.py --exclude`, `lead5_work/exact_test189.json`).")
add("")
add("| opponent | reader | n | W / L / T | score [95%] | sign test p |")
add("|---|---|---|---|---|---|")
for lab, rdr, s, src in (("Falcon-H1-Tiny", q_lab("final-prefopt-vs-falcon", "short"), "final-prefopt-vs-falcon", Q),
                         ("Falcon-H1-Tiny", "Opus, two readings", "final-prefopt-vs-falcon", O),
                         ("Falcon-H1-Tiny", "Opus, neutral titles", "final-prefopt-vs-falcon", "judge-opus-neutral"),
                         ("Falcon-H1-Tiny, K = 6 own log-probability", "Opus, two readings", "final-prefopt-vs-falcon-k6lp", O),
                         ("Falcon-H1-Tiny, K = 6 our selector", "Opus, two readings", "final-prefopt-vs-falcon-k6ds", O),
                         ("MobileLLM-R1.5", q_lab("final-prefopt-vs-mobilellm", "short"), "final-prefopt-vs-mobilellm", Q),
                         ("MobileLLM-R1.5", "Opus, two readings", "final-prefopt-vs-mobilellm", O),
                         ("MobileLLM-R1.5", "Opus, neutral titles", "final-prefopt-vs-mobilellm", "judge-opus-neutral"),
                         ("our trained weights", "Opus, two readings", "final-prefopt-vs-trained", O),
                         ("Falcon-H1-Tiny, against our K = 1 arm", "Opus, two readings", "final-k1-vs-falcon", O),
                         ("MobileLLM-R1.5, against our K = 1 arm", "Opus, two readings", "final-k1-vs-mobilellm", O),
                         ("reference: Falcon-H1-Tiny vs MobileLLM-R1.5", "Opus, two readings", "ref-falcon-vs-mobilellm-test", O)):
    e = v(s, src)
    if e:
        b = e["189"]
        add(f"| {lab} | {rdr} | {b['n']} | {b['W/L/T']} | {b['score']} {b['score_ci']} | {b['p']} |")
E1 = EX["189"]
add("| exact scores: maths correct (of 28); constrained prompts met (of 44) | – | – | " + "; ".join(
    f"{l} {E1[k]['maths_correct']} and {E1[k]['constraints_all_ok']}" for k, l in runs) + " | – | – |")
add("")
# M12
t = TG["prefopt"]
c = t["cost"]
tt = TG["trained"]["cost"]
add("**Table M12 (R12). Decoding throughput of the export on CPU** (the official test generation's summary "
    "`N/testgen/prefopt/out/testgen.json` and its log; the trained weights' generation for comparison, "
    "`N/testgen/trained/out/testgen.json`; on `dev`, 75 replies took 1,798 s with 16 workers, `FACTS_2.md`).")
add("")
add("| quantity | official (preference-optimised) | trained weights |")
add("|---|---|---|")
add(f"| workers × cores per worker; host | {t['workers']} × {t['cores_per_worker']}; the TPU runtime's host (CPUs 2–21, 24–43 of 44) | "
    f"{TG['trained']['workers']} × {TG['trained']['cores_per_worker']}; the same |")
add(f"| replies; wall time, export and calibration included | {t['new_replies']}; {t['wall_s']:,.1f} s ({t['started'][11:]}–{t['finished'][11:]}) | "
    f"{TG['trained']['new_replies']}; {TG['trained']['wall_s']:,.1f} s ({TG['trained']['started'][11:]}–{TG['trained']['finished'][11:]}) |")
add(f"| seconds per reply (median); new tokens per reply (median) | {c['reply_seconds']['median']:.1f}; {c['tokens']['median']:.1f} | "
    f"{tt['reply_seconds']['median']:.1f}; {tt['tokens']['median']:.1f} |")
add(f"| seconds per decoding step, 6 particles, one core (median; mean) | {c['ms_per_step']['median'] / 1000:.3f}; {c['ms_per_step']['mean'] / 1000:.3f} | "
    f"{tt['ms_per_step']['median'] / 1000:.3f}; {tt['ms_per_step']['mean'] / 1000:.3f} |")
add("")
# M13: paired comparisons of our per-prompt score (the CLI's `compare`)
PR = json.loads((W / "final_results.json").read_text())["paired"]
add("**Table M13. Paired comparisons of our per-prompt score on the same prompts** (`final_results.py`, "
    "`bes.eval.proxy.score.paired_sign_test`, as `python -m bes.eval.proxy compare --a A --b B --source judge-opus`; "
    "a prompt is better when our score against the opponent is higher in A than in B).")
add("")
add("| A | B | prompts better in A; in B; same | mean difference | sign test p | on the 189 untouched prompts |")
add("|---|---|---|---|---|---|")
for k, lab in (("final-prefopt-vs-falcon | final-prefopt-vs-falcon-k6lp", ("official vs released Falcon", "official vs Falcon K = 6, own log-probability")),
               ("final-prefopt-vs-falcon | final-prefopt-vs-falcon-k6ds", ("official vs released Falcon", "official vs Falcon K = 6, our selector")),
               ("final-prefopt-vs-falcon | final-k1-vs-falcon", ("official (K = 6) vs Falcon", "ours K = 1 vs Falcon")),
               ("final-prefopt-vs-mobilellm | final-k1-vs-mobilellm", ("official (K = 6) vs MobileLLM-R1.5", "ours K = 1 vs MobileLLM-R1.5"))):
    e = PR.get(k, {})
    if "all" not in e:
        add(f"| {lab[0]} | {lab[1]} | [TBD: a set not yet read] | | | |")
        continue
    a, b = e["all"], e["189"]
    add(f"| {lab[0]} | {lab[1]} | {a['a_better']}; {a['b_better']}; {a['same']} | {a['mean_score_diff']:+.4f} | {a['p']:.4f} | "
        f"{b['a_better']}; {b['b_better']}; {b['same']}, p = {b['p']:.4f} |")
PS = json.loads((W / "final_results.json").read_text()).get("paired_sources", {})
for k, lab in (("final-prefopt-vs-falcon: judge-opus | judge-opus-neutral", "vs Falcon-H1-Tiny"),
               ("final-prefopt-vs-mobilellm: judge-opus | judge-opus-neutral", "vs MobileLLM-R1.5")):
    e = PS.get(k, {})
    if "all" in e:
        a = e["all"]
        add(f"| official {lab}, first readings (titles named the set) | the same pairs, neutral titles | "
            f"{a['a_better']}; {a['b_better']}; {a['same']} | {a['mean_score_diff']:+.4f} | {a['p']:.4f} | – |")
add("")
text = "\n".join(L) + "\n"
(W / "appendix_M.md").write_text(text)
if "--splice" in sys.argv:
    p = (P / "paper.md").read_text()
    a = p.index("## Appendix M.")
    b = p.index("## Appendix N.")
    (P / "paper.md").write_text(p[:a] + text + "\n" + p[b:])
    print("spliced into paper.md")
print(text[:2500])
