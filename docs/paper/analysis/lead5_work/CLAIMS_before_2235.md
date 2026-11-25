# CLAIMS: every quantitative or factual claim of `paper.md`, with its source

Second pass: state of the sources 24 November 2026, about 13:15 UTC; repository at commit `8a86bcb`.
Third pass (C400–C448, at the end of this file): sources to 16:28 UTC (newest step record 46,350);
repository at commit `b1e9140`. Entries below that the third pass withdrew or changed are marked in
their row and listed at the end. After the programme-committee check (`REVIEW_2.md`), 18:10–18:55 UTC:
entries C449–C465 and the changes listed under "Changed after the check" at the end; same live cut-off;
repository at commit `190e0b2`; `L4/` = `N/paper/lead4_work/` (scripts in `L4/scripts/`, each run as
below).

Final pass (C500–C546 and the list "Withdrawn or changed in the final pass", at the end of this file): the
run's last records (`N/snap_final`, 20:16 UTC; training ended at step 51,812), the official export, the test
replies and every reading of the test pairs, recomputed by `lead5_work/scripts/`; repository at `34f12ac`.

Abbreviations used in the source column:

- `R/` = `/home/user/turingaward2027/` (the repository; `docs/` is `R/docs/`).
- `N/` = `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/`.
- `S/` = `/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/` (operator logs, chip outputs, TPU-time ledger).
- `W/` = `N/wf/` (workflow reports). `L/` = `N/paper/lead_work/` (first-pass extracts and scripts).
  `L2/` = `N/paper/lead2_work/` (second-pass scripts and outputs). `RW1/` = `N/paper/review1_work/`.
- `FLOG` = `L/final_snap/run/train/train_log.jsonl` (the stopped `final` run); `VLOG` =
  `L2/v9_snap_latest/run/train/train_log.jsonl` (the `v9` run, read from the snapshot `N/snap_main` of
  13:08 UTC without extracting it; newest step record 37,650); `FMAN`, `VMAN` = the runs' manifests;
  `FLED`, `VLED` = their ledgers. (`L2/v9_snap_1152/` is the earlier extract of 11:52, newest step 34,350; the
  numbers below that depend on the v9 log were recomputed from `VLOG` and are unchanged by the later
  records unless stated.)
- `NUM` = `L2/numbers.json` (written by `L2/make_numbers.py` from FLOG, VLOG and typed-in chip values);
  `PRED` = `L2/predictor_test.json`; `HA` = `L2/heldout_analysis.json`; `EXD` = `L2/exact_dev.json`;
  `GG` = `L2/graft_grad.json`; `BPB` = `L2/opp_bpb_web.json`; `DL` = `L2/deflection_labels.json`.
- "computed" = arithmetic by the second-pass author on the cited inputs, by the named script.
- Ids C001–C237 are those of the first pass; a claim whose value changed says so and names the
  review item (`REVIEW_1.md`). New claims start at C240.

Claims marked **[TBD]** have no value yet. A claim without a source is not in the paper.

## Status, abstract, introduction

| id | § | claim | value | source |
|---|---|---|---|---|
| C240 | status | **changed in the third pass → C400.** newest record read: step, time, p, tokens | step 37,650 at 13:05:57; p 0.6873; ledger 9,869,983,744 tokens | VLOG (last step record); `L2/v9_snap_latest/run/train/ledger.json` |
| C003 | abs, 2.1 | the budget | ≤100M stored numbers of ≤16 bits, ≤20M read per token, 24 h of one TPU v6e, from scratch, blind pairwise comparison | `docs/PLAN.md` l.5; `docs/ACCOUNTING.md` l.5–10 |
| C016 | abs, 1, 4.2 | extra reads of `v9` | 597,504 (3.1%), of which 589,824 dense projections | computed from `docs/V9.md` l.160–161 |
| C005 | abs, 1, 5.3 | analytic training FLOP per token at 1,024 tokens | `final` 0.781 G, `v9` 0.809 G (+3.6%) | `L/scripts/flops.py`; NUM `flop.final_1024` |
| C241 | abs, 1, 5.3 | step-time ratio at the switch and after the three switches (synthetic, 64 × 4 × 1,024) | 1,433.1 / 884.5 = 1.620; 1,287.1 / 884.5 = 1.455 (changed: item 1, 27) | `S/v9_measA.txt`; `S/v9_measD.txt`; `W/chip_kernels_0745.txt` |
| C242 | abs, 1, 5.4 | bytes predictor: leave-one-out errors on six configurations | mean 5.7%, largest 11.1%; rank correlation 1.0 | PRED (`loo` "estimated HBM bytes") |
| C243 | abs, 1, 5.4 | FLOP and reads cannot separate five of the six (all `v9` variants have the same counts) | 0.809 G and 19,631,240 for every `v9` row | PRED `rows` |
| C010 | abs, 1, 6 | progress at the graft | p = 0.5049 (stop event) | FLOG end event of step 29,722 (0.50489) |
| C012 | abs, 6 | tokens trained at the graft | 7,791,443,968 | FLED `tokens` |
| C244 | abs, 6.4 | grafted loss equals the source's bit for bit at the live source checkpoint (re-run) | 11.3726320 both; difference 0.0; hash `f29bfeb3ebf353a8` | `RW1/graftcheck/graft_check_29722.log` (item 21) |
| C245 | abs, 7.4 | dev blind comparison, Falcon-H1-Tiny | 2 / 43 / 15 (43 of 45 decisive pairs lost) | `N/study/proxy/reports/ab/s1-default-vs-falcon--opus-1.json` |
| C246 | abs, 7.4 | the compared checkpoint was at 0.44 of training | `final` step 25,280, p 0.4396 (end event), 6.63 × 10⁹ tokens | FLOG end event of step 25,280; 25,280 × 262,144 |
| C247 | 1 | ledger totals | `final` 98,300,111 / 19,033,736 (19,278,959 worst case); `v9` 98,036,104 / 19,631,240 | `docs/ACCOUNTING.md` l.24–26; `docs/V9.md` l.163 |
| C248 | 1, 5.3 | at least 27% of the extra time was layout | 146.0 of 548.6 ms = 26.6% | computed from C241 |
| C249 | 1, 5.2 | **changed in the third pass → C403.** live rate of `final` from 112–117 k to 281 k tokens/s | 112–117 k (`docs/TPU_NOTES.md` l.187–189); 281,389 (n = 142) | NUM `rate.kernel_to_graft` |
| C250 | 1, 5.5 | 16 block applications per token are counted once and charged 16 times | 2 blocks × 8 loops | `R/bes/config.py` ModelCfg |
| C251 | 1, 6.1 | the user asked for `v9` at p = 0.505; training it from scratch would have seen fewer tokens | 09:16 UTC; about 5.4e9 (rates at the switch) or 6.0e9 (6.02e9, with the later switches and the measured 2,048 rate) < 7.79e9 | C136; C137; NUM `proj.v9_scratch_at_graft` |
| C020 | 1 | internal review's novelty rating of the design mathematics | 2–3 out of 10 | `docs/MATH.md` l.5 |

## Setting (Section 2)

| id | § | claim | value | source |
|---|---|---|---|---|
| C021 | 2.1 | definitions of stored total, standard and strict readings | as stated | `docs/ACCOUNTING.md` l.7–10 |
| C022 | 2.1 | the compute rule as read by the operator, reported to the user | allocated time, not waits | `docs/CHANGES.md` l.258 |
| C023 | 2.2 | device memory and versions (changed: item 30, GiB) | 31.24 GiB (`bytes_limit` / 2³⁰); 44 host cores; jax/jaxlib 0.11.2; libtpu 0.0.50 | FMAN `devices`, `host`; `R/bes/train/supervisor.py` l.398 |
| C025 | 2.2, 5.2, 5.6 | the two feed-forward products alone, forward | 814 TFLOP/s | `docs/TPU_NOTES.md` l.215 |
| C252 | 2.2 | definitions: micro-batch, step, piece; exact; equal to rounding; live rate; synthetic | 16 × 1,024 rows (8 × 2,048 from p 0.67); 262,144 tokens per step; 4 pieces per micro-batch | `R/configs/final.py`; `R/bes/config.py` OptimCfg; `<run>/micro_split` (VMAN) |
| C253 | 2.2 | **changed in the third pass → C402.** achieved rate within 0.1% of the per-step rate | e.g. 281,389 against 281,207; 178,087 against 178,009 | NUM `rate.*` (`median_tok_per_s`, `achieved_median`) |
| C026–C028 | 2.3 | opponents' files, decoding, two arms, budget forcing | as stated | `docs/EVAL_PROTOCOL.md` l.71–110 |
| C234 | 2.3, 8 | opponents' training tokens | Falcon-H1-Tiny-90M-Instruct 8 × 10¹¹; MobileLLM-R1 pre-training 4.2 × 10¹² | `N/paper/refs.md` [11], [7], [9] |
| C254 | 2.3, 6.3 | **changed in the third pass → C408.** our projected total | about 1.36 × 10¹⁰ (1.360e10 at the measured rates) | NUM `proj.v9_total` |
| C030–C032 | 2.4, C | runtimes 1–3 | 7,325 s; 8,220 s; from 01:05:55 | `S/tpu_intervals.txt`; `docs/CHANGES.md` |
| C255 | 2.4, I | runtime 2 taken back about 19:29, about 2 h 12 min after its allocation (about 17:17); the ledger counts 17:13:00–19:30:00 | as stated | `docs/CHANGES.md` l.266, l.287 |
| C034 | 2.4 | TPU time at the graft and share in training steps | 12.98 h; 8.98 h (69%) | computed; FLED |
| C256 | 2.4, C1 | **not in the third draft (see C410).** TPU time at 12:00:00 | 7,325 + 8,220 + 39,245 = 54,790 s = 15.22 h | computed from C030–C032 |
| C036 | 2.4 | deadline, end of training, post-training minutes | 20:31:49; 18:33:49; 118 = 60 + 35 + 3 + 5 + 15 | FLOG setup events; VMAN `post_plan` |

## Model and ledgers (Section 4, Appendix A, B)

| id | § | claim | value | source |
|---|---|---|---|---|
| C056–C062 | 4.1, B | tokenizer, exact layer, core shapes, equations, precision, attention | as stated | `docs/SPEC.md`; `docs/SPEC_MODEL_TRAIN.md`; `R/bes/config.py`; `docs/TPU_NOTES.md` |
| C065, C066 | 4.1, B.3 | output law, its four properties, initial mass e⁻⁶ | as stated | `docs/MATH.md` l.35–46; `docs/SPEC_MODEL_TRAIN.md` l.25–49 |
| C068–C071 | 4.1, 6.2 | token-neuron equations, table shapes, initialisation U = G = 0, exact reduction | as stated | `R/bes/model/layers.py`; `R/bes/model/model.py`; `docs/V9.md` l.106–143 |
| C038, C039, C041 | 4.2, A | ledger rows; library 63,768,391; worst retrieval read 245,223; shared 34,531,720 | as in Tables 1, A1 | `docs/V9.md` l.153–163; `docs/ACCOUNTING.md` l.16–26; `tables.md` T1 |
| C042 | 4.2 | audit checks | \|Δ log p\| ≤ 1e−4; poisoned rows; reads logged | `docs/ACCOUNTING.md` l.48–55 |
| C044, C045 | 4.2 | `v9` audit and the strict reading | 19.43M (19.63M with critic); 21.2/23.8/24.6M and 22.4/26.7/28.2M; 9,536 per distinct token | `docs/ACCOUNTING.md` l.41–46; `docs/V9.md` l.166, l.307–310 |
| C043 | A | `final` strict reading | 19.62–19.81M (1,024), 20.07–20.49M (2,048), 6 documents | `docs/ACCOUNTING.md` l.30–35 |
| C046 | 4.3 | page row per token | 3,072 numbers (2 × 8 × 192) | `R/configs/v9.py` |
| C063, C064 | B.2 | sink identity; sink removal −0.07% | as stated | `docs/MATH.md` l.9–12; `docs/E2_ABLATIONS.md` l.51 |
| C257 | B.4 | library coverage / precision before → after, all rows (changed: item 17) | NQ-open 35.0/54.1 → 6.0/76.9; PopQA 6.5/40.3 → 0.5/75.0; head-entity 82.3/98.7 → 74.4/99.1; common-name 24.3/35.6 → 81.3/99.9; NQ without near-duplicate 18.5/37.3 → 1.0/43.5 | `docs/E1_COVERAGE.md` l.24–33 |
| C048 | B.4 | the gate opens on 8 of 200 dev first turns | 8 | `docs/EVAL_PROTOCOL.md` l.324–326 |
| C049–C054 | B.4 | laboratory and micro-ablation evidence for pages | as stated | `docs/V9.md` l.12–79, l.209–230 |
| C258 | B.4 | the design's own scaling-law prediction (new: item 18) | `final` 2.507, Falcon-H1-Tiny-90M 2.358, `v9` about 2.45 (2.41–2.48) nats/token; "narrows … does not close" | `docs/V9.md` l.188–205 (§7) |
| C076–C079 | B.5 | Table B1 and the two deviations | as in the table; registers 79.2% vs 92.7% | `docs/E2_ABLATIONS.md`; `docs/E4_PROBES_2M.md` l.20 |
| C072–C075 | B.6 | optimisers, learning rates, transfer rule, horizon 1.3117e10 | as stated | `R/bes/config.py`; `docs/SPEC_MODEL_TRAIN.md`; FMAN `benchmark.horizon_tokens` |
| C212 | B.6, H | transfer rule at micro scale | +14.44% | `docs/E2_ABLATIONS.md` l.64 |

## Where the time goes (Section 5, Appendix F)

| id | § | claim | value | source |
|---|---|---|---|---|
| C259 | 5.1 | five chip pauses, from the stop record to the restore record (changed: items 27, 29) | 5:54, 9:25, 9:21, 13:10, 5:14; 43:04 in all | `tables.md` T5b (train-log stop and restore records) |
| C260 | 5.1, F.5 | compile-only with jax 0.11.1, chip with 0.11.2 (new: item 15) | as stated | `W/v9_speed.md` (Method); FMAN |
| C099 | 5.1 | cost-model calibration: two shapes of one program family | 0.521 ms per M cycles | `W/speed_mlp_block.md` l.86–91 |
| C008 | 5.2 | live rates on 23 Nov and the plan's assumption | 112–117 k; `v9` 88–92 k; plan 300–450 k | `docs/TPU_NOTES.md` l.187–189; `docs/PLAN.md` l.149 |
| C101 | 5.2 | inside a block: 15.4 ms; products about 2.4; kernel 3.4; the other 9.6 (62%), outside both, incl. element-wise work (changed: item 20) | as stated | `docs/TPU_NOTES.md` l.208–223 |
| C261 | 5.2 | one 768 × 768 product: 0.08 ms forward, 0.23 ms with recomputation and backward (changed: item 20) | as stated | `docs/TPU_NOTES.md` l.216 |
| C102–C106 | 5.2, F1 | the rewrites, 289.3 → 162.0 ms; 28 tests; bfloat16 pre-activation 2⁻⁹, 28% | as in Table F1 | `docs/TPU_NOTES.md` l.229–236; `docs/CHANGES.md` l.238–251 |
| C112, C113 | 5.2, F | one-tile kernel design; executed-bundle ratios 0.34 / 0.42 | as stated | `docs/TPU_NOTES.md` l.296–318 |
| C115 | 5.2, F2 | piece times of the kernel candidates | as in Table F2 | `W/chip_kernels_0745.txt` |
| C262 | 5.2, F2 | equality of the adopted kernel on the chip (changed: items 13, 22) | Δloss ≤ 9.5e−7; gradients 2.8e−2/2.6e−2 (1,024), 7.9e−2/5.7e−2 (2,048); graph-only band 1.1–1.4e−2 / 3.9–4.7e−2; `io` 0 | `W/chip_kernels_0745.txt` (KERNELS lines); `tables.md` T3a |
| C263 | 5.2 | the comparison used freshly initialised parameters and a synthetic batch | `M.init_params(mc, key(0))`; three segments per row | `R/scripts/tpu_kernels.py` l.11–12, l.170 |
| C119 | 5.2 | CPU interpret-mode checks | float32 within 1e−5; float64 dense reference; segment boundaries | `docs/CHANGES.md` l.361; `W/kernel.md` l.71–96 |
| C264 | 5.2, Table 2 | live medians by stretch (changed: item 26) | 222,031 (101); 216,548 (85); 240,131 (266); 281,389 (142); 178,087 (44); 196,874 (99, steps 31,950–36,850) | NUM `rate.rt2_16rows`, `rate.rt3_same`, `rate.pieces`, `rate.kernel_to_graft`, `rate.v9_take`, `rate.v9_switches_1024` |
| C265 | Table 2 | benchmark and synthetic values | 111,893; 201,877; 216,681; 246,538; 296,371; 182,919; 203,665 | `tables.md` T2, T3b; `docs/CHANGES.md` §19.1, 19.3, 19.5–19.7 |
| C266 | 5.2 | **changed in the third pass → C404.** 2.40–2.51× includes the 32 → 16 rows change (+8%) (new: item 39) | 281,399 / 117,000 and / 112,000 | NUM `ratio.reference_to_kernel_live`; C107 |
| C267 | 5.2 | kernel live gain on medians (changed: item 26) | 240,143 (n 32, steps 21,050–22,600) → 281,387 (n 53, 22,650–25,250): +17.2% | NUM `ratio.kernel_gain_live` |
| C268 | 5.2 | live about 5% below synthetic; main thread 2.9% of a core | as stated | `docs/CHANGES.md` §19.6 |
| C269 | 5.3 | the three switches and their exactness | `flat` exact; `rows` exact; `tokn` float32 order | `W/v9_speed.md`; `W/v9_rows.md`; `W/v9_tokn.md`; `docs/CHANGES.md` §19.7 |
| C270 | 5.3, F3 | `v9` steps and pieces with the switches | 1,433.1 / 1,406.4 / 1,295.2 / 1,287.1 ms; pieces 22.79 / 22.59 / 20.44 / 20.01 ms | `S/v9_measA.txt`; `S/v9_measD.txt`; `FACTS_2.md` §1 |
| C271 | 5.3 | live ratios before / after the switches (changed: item 26) | 281,399 / 178,087 = 1.580; / 196,874 = 1.429 | NUM `ratio.final_over_v9_take_live`, `ratio.final_over_v9_switches_live` |
| C128, C129 | 5.3 | cost-model attribution and padding | token neurons' `[8, 192]` contractions padded 8 → 16, 192 → 256; gain table 0.17 M cycles | `W/v9_speed.md` l.74–96 (compile-only) |
| C130 | 5.3 | extra seconds per step then and now | about 0.6 s (23 Nov); 0.55 s (1,433.1 − 884.5) | `docs/TPU_NOTES.md` l.188–189; computed |
| C272 | 5.4, Table 4 | static counts of six configurations | cycles 1,944.7 … 2,825.6 M; HBM GB 515 … 1,130 | `L2/estimate_out.txt` (from `W/v9_*_work/estimate.py` on the existing HLOs); PRED |
| C273 | 5.4 | **WITHDRAWN in the third pass (fit undefined without `final`) → C407.** leave-one-out errors and rank correlations | FLOP mean 19.6% (max 66.4%); cycles 11.7% (40%); bytes 5.7% (11.1%); Spearman 0.655 / 1.0 / 1.0 | PRED `loo` |
| C274 | 5.4 | line through all six (bytes) | 492.5 ms + 0.7313 ms per GB (1.37 TB/s); residuals −110 … +118 ms | `L2/predictor_test.py` output |
| C275 | 5.4, Table 5 | cost model against the chip, by change (new: item 38) | ratios 1.14, 0.36, 0.85, 2.95, 1.72, 1.07, 7.85; signs 7 of 7 | PRED `diffs`; `W/v9_speed.md`; `W/v9_rows.md`; `W/v9_tokn.md`; `docs/CHANGES.md` §19.5 |
| C276 | 5.4 | "5% high" was 14% high on the difference (changed: item 24) | +623.7 vs +548.6 ms | `W/v9_speed.md`; C241 |
| C123 | 5.4 | carry of the next loop's input: estimated only | +4.87% | `W/speed_loops.md` l.103 |
| C132 | 5.4 | `sorted` needs 16× fewer multiply-adds than one-hot | as stated | `docs/V9.md` l.111–117 |
| C277 | 5.5 | loop arithmetic (new: item 41) | 42.9 vs 289 ms; ≈35 ms per loop; ≈8 ms block and 34 ms outside after the rewrite; 4 loops 1.65–1.95× the tokens | `docs/TPU_NOTES.md` l.202, l.247–249; NUM `loops.time`; computed |
| C278 | 5.5 | micro ablation: 4 loops −0.12%, 12 loops +0.26% | as stated | `docs/E2_ABLATIONS.md` |
| C279 | 5.5 | 2^0.65 = 1.57 under the iso-depth law with two lanes | φ = 0.65 | `N/paper/refs.md` [21] |
| C280 | 5.6 | FLOP recount (changed: item 25) | analytic 0.781 G → 220 TFLOP/s (27%); executed 1.179–1.195 G → 332–336 TFLOP/s (41%) at 281,399 tokens/s | NUM `flop.final_1024`; `RW1/flops_check.py` |
| C121, C122 | 5.7, F4 | 13 changes not adopted; two contradicted the written expectation (changed wording: item 20) | as in Table F4 | `docs/TPU_NOTES.md`; `docs/CHANGES.md` §19.3–19.7; `S/v9_meas*.txt` |
| C281 | F | `v9` switches' equality on the chip | Δloss 0; gradients 9.5e−3 / 1.0e−2 against `flat` | `FACTS_2.md` §1; `tables.md` T3c |

## The graft (Section 6, Appendix G)

| id | § | claim | value | source |
|---|---|---|---|---|
| C136 | 6.1 | the user asked for `v9` at 09:16 UTC | 09:16 | the workflow brief; commit `9a4d652` message |
| C017 | 6.1 | training time left at the restart | 30,969 s | `S/runlog_peek.txt`; VLOG run.log l.862 |
| C138 | 6.1, C.5 | the pre-registered comparison and why it was not run | as stated | `docs/CHANGES.md` l.212–226, l.232, l.259 |
| C139, C140 | 6.2 | neutral initialisation and its consequences; no gradient clipping | derivation; `grad_clip = 0.0` | `R/bes/model/layers.py`, `model.py`; `R/bes/config.py` l.171 |
| C141 | 6.2 | Adam transient | 0.894 (t = 1), 1.439 (t = 7), 1.041 (t = 50) | computed |
| C011, C142 | 6.3 | what the tool carries and refuses; 59 leaves (34,531,720), 5 new (63,504,384) | as stated | `R/scripts/graft_run.py`; `RW1/graftcheck/graft_check_29722.log` |
| C084 | 6.3, C.1 | p across the graft (events; definition stated: item 29) | 0.5049 → 0.5112 | FLOG end event; VLOG stage event |
| C153 | 6.3 | learning-rate factor after the graft | 0.5394 at step 29,750 | VLOG |
| C144 | 6.3 | horizon 40 times too small without the correction | 1.9 × 10⁸ against 8.2 × 10⁹ | commit `9a4d652` message; `docs/CHANGES.md` §19.7 |
| C282 | 6.3 | the loader's plan with the correction, against the projection (changed: item 19) | 7.45 × 10⁹ planned; about 5.8 × 10⁹ remain (0.574e9 to step 31,912 + 5.23e9 projected) | VLOG run.log l.875–878; NUM `proj.v9_total`; computed |
| C143 | 6.3 | reads left per source | 8 × (1 − 0.5049) = 3.96 | VLOG run.log ("at most 3.96093 reads") |
| C283 | 6.3 | GSM8K capped before and after the graft (new: item 45) | 0.0126 → 0.0007 after; 0.0114 → 0.0008 before | VLOG run.log l.42, l.876 |
| C284 | 6.3 | **changed in the third pass → C408.** token projection from the 10:57:50 restart (changed: item 28) | total about 1.36e10 (1.360e10); deficit about 2.3e9 (2.28e9, 14.4%) against `final`'s projected 1.588e10; at the measured live rates 196,874 (1,024) and 188,980 (2,048) | NUM `proj.v9_total`; C157 (final's projection, whose 2,048 rate is scaled) |
| C285 | 6.3 | the 2,048-token phase started at step 36,867 (p = 0.67), first record 12:48:36 | as stated | VLOG stage event and step records |
| C330 | 6.3, Table 2 | **changed in the third pass → C405.** live rate of `v9` at 2,048 tokens (new: item 43) | 188,980 tokens/s (n 16, steps 36,900–37,650, range 188,566–189,858), 4.0% below 196,874 | NUM `rate.v9_switches_2048`; VLOG |
| C146 | 6.4 | Table 6 rows 26,890, 28,501, 14 | 11.2808027; 11.3348608; 9.2843390 | `S/graft_test.log`; `S/graft_dry_out.txt`; `N/graft_rehearsal/graft.log` |
| C286 | 6.4 | Table 6 row 29,722: a re-run of the same source checkpoint under the live hash (changed: item 21) | 11.3726320 / 11.3726320 / 0.0; hash `f29bfeb3ebf353a8`; HEAD `8a86bcb` | `RW1/graftcheck/graft_check_29722.log` |
| C147 | 6.4 | the saved output of the live graft has no check record | `S/graft_out.txt` without `--check`, into `main16_v9_sorted`, hash `f1243cb29610103a` | `S/graft_out.txt`; `REVIEW_1.md` item 3 |
| C287 | 6.4 | the tool writes the graft record before the check and does not store the check | `graft_run.py` l.275, l.281, l.287–291 | `R/scripts/graft_run.py` |
| C288 | 6.4 | gradient check on real rows, CPU float32 (new: item 40) | losses 2.9320383 (web, 2 rows, 2,043 weighted tokens) and 1.5048378 (chat, 2 rows, 2,013) equal; 59 of 59 shared gradients bit-identical; K, P_in, P_out gradients exactly 0; U max 7.37e−3 / 9.03e−3, G max 1.005e−3 / 1.214e−3 | GG |
| C150 | 6.4 | the graft review | four tests, no blocker, two medium risks | `W/graft_review.md`; `R/tests/train/test_graft_run.py` (commit `769f240`) |
| C289 | 6.5 | held-out rows (changed: item 23) | web 64, chat 64, proxy 12 rows; 136,974 weighted tokens | `N/midrun_eval_1.json` (`heldout`) |
| C290 | 6.5 | the proxy rows hold user turns of all 400 prompts (new: item 9c) | `proxy_samples` calls `prompt_set.load()` | `R/bes/export.py` |
| C291 | 6.5 | runtime held-out values | `final` 26,890 / 28,501 / 29,722; `v9` 30,742 / 31,762 / 31,912 / 33,038 / 34,166 / 35,294 | `N/midrun_eval_1.json`; `N/heldout_v9.jsonl` (copy of 12:30) |
| C292 | 6.5 | **changed in the third pass → C419.** local per-position evaluations reproduce the runtime's values | 29,722, 34,166 and 35,294: differences below 1e−8 (**not so, found after the check**: web and chat differ by up to 2.7 × 10⁻⁸ there; C419) | HA `reproduction` |
| C293 | 6.5, Table 7, G1 | paired differences, extrapolation, pages-off decomposition, frequency strata | as in Table 7 and the text of §6.5 (e.g. −0.0499 [−0.0541, −0.0457]; pages off +0.0473 [0.0436, 0.0511]; quartiles 0.0655 / 0.0540 / 0.0387 / 0.0316) | HA `paired`, `extrapolation`, `frequency`; extrapolation by `L2/heldout_analysis.py` with t(1) = 12.706 |
| C294 | 6.5 | **WITHDRAWN in the third pass (a line through all 140 rows; not cited).** three-point line: 95% prediction interval at 592.5 min (changed: item 35) | ±0.025 (t with 1 dof); gap 0.0158 | NUM `heldout.three_point`; `RW1/stats.py` |
| C295 | 6.6 | the new leaves train from a learning-rate factor of 0.54 down to 0.1 | 0.5394 at step 29,750; 0.1 at p = 1 | VLOG; `R/bes/config.py` OptimCfg |
| C328 | 8, 9 | token gap to the opponents | 8e11 / 1.355e10 ≈ 59; 4.2e12 / 1.355e10 ≈ 310 | computed from C234, C254 |
| C329 | 6.5, I | local copies of `final` checkpoints 5,075, 22,613, 26,890 (and the extracted 29,722) deleted at 12:10–12:14 to free the disk; 29,722 read from the kept tar | as stated | `N/OPERATIONS.md` ("Since 12:10"); `L2/run_heldout2.log` (missing directories at 12:39); `N/snap_final_main16` |

## Evaluation (Section 7, Appendix E)

| id | § | claim | value | source |
|---|---|---|---|---|
| C231 | 7.1 | the protocol is in the root commit and unchanged since; it predates every TPU run and full-scale reply, not the micro runs (changed: item 10) | `5ff33c5`, 23 Nov 00:35:15; first runtime 02:58:32 | `git log`; `git diff 5ff33c5 HEAD -- docs/EVAL_PROTOCOL.md` (empty) |
| C188, C189, C193, C194, C196 | 7.1 | prompts, split, blinding, two readings, exact scores | as stated | `docs/EVAL_PROTOCOL.md` |
| C296 | 7.2 | pre-registered judges (new: item 9a) | `qwen3-4b-2507`, second judge `gemma4-e4b` | `docs/EVAL_PROTOCOL.md` judge table (l.225–226) |
| C297 | 7.2 | readers in this run are Opus 5.5 agents | dev study and planned final readings | `N/OPERATIONS.md` ("Endgame (v9)"); `N/study/proxy/reports/ab/*--opus-1.md` |
| C298 | 7.2 | eleven test prompts were seen; math-011 is dev; 189 untouched (changed: item 14) | chat-002, chat-017, math-034, instruct-034, instruct-062, writing-100, code-026, math-001, code-002, writing-001, instruct-001 | `bes.eval.proxy.prompts.load()` (checked 12:3x UTC); ids in `N/midrun_sample_step*.md` |
| C299 | 7.2 | MobileLLM's neutral arm has replies on dev, not yet compared blind | `neutral-48fefc857c-s0` | `N/study/proxy/outputs/mobilellm-f680f0533368-qwen3/` |
| C300 | 7.3 | tie rates on dev | `final`, one reading: 15 and 37 of 60 (25%, 62%); `v9`, two readings: 18 and 33 of 60 (30%, 55%) | `N/study/proxy/reports/ab/s1-default-vs-*--opus-1.json`, `s2-default-vs-*--judge-opus.json` |
| C301 | 7.3 | smallest detectable decisive-rate difference (changed: item 32) | at 30% / 55% ties: n = 200: 0.117 / 0.146; n = 189: 0.121 / 0.150; at 25% / 62%: 0.113 / 0.158 (n = 200) | NUM `power.*` (`bes.eval.proxy.score.n_required`) |
| C302 | 7.3 | score half-width at n = 200 | ±0.069 | NUM `power.score_halfwidth_n200` |
| C303 | 7.4 | dev blind comparison values | as in Table 8 | `N/study/proxy/reports/ab/s1-default-vs-*--opus-1.json` |
| C304 | 7.4 | exact scores on the 60 dev prompts (new: item 34) | `final` / `v9` / Falcon / MobileLLM (neutral): maths 0, 1, 4, 4 (1) of 9; constraints 12, 12, 9, 2 (1) of 14; cut 2, 0, 0, 42 (54) of 75; no stock refusal or deflection; median words 93, 96, 188, 532 (698) | EXD |
| C325 | abs, 1, 7.4 | blind dev comparison of `v9` at step 31,912, two readings on exchanged sides (new) | vs Falcon 3 / 39 / 18, 0.200 [0.118, 0.318], p < 0.0001; vs MobileLLM 20 / 7 / 33, 0.608 [0.482, 0.722], p = 0.019; orders agree 93.3% / 98.3%; first-position preference +0.000 / +0.008; MobileLLM slices instruct 9 / 0 / 3, two-turn 7 / 0 / 8, maths 0 / 5 / 4; Falcon two-turn 0 / 14 / 1 | `N/study/proxy/reports/ab/s2-default-vs-*--judge-opus.{md,json}` |
| C326 | 7.4, E.4 | `v9` decoding study: 3-gram blocking against the default | 4 / 8 / 48, p = 0.39; constraints 13 / 14 against 12; cut 1 against 0 | `N/study/proxy/reports/ab/s2-ngram3-vs-default--opus-1.json`; EXD |
| C327 | 7.4 | our replies shorter than the opponent's | 38 to 45 of 60 pairs (40, 45 for `final`; 38, 44 for `v9`) | reports `by_length` (`x_shorter`) |
| C306 | 7.4 | two-turn pairs against Falcon-H1-Tiny (`final`) | 0 / 15 / 0 | `s1-default-vs-falcon--opus-1.json` `by_turns` |
| C307 | 7.4 | bits per byte on the web rows (new: item 31) | ours 1.0172 (`final` 29,722), 1.0009 (`v9` 34,166), 0.9975 (`v9` 35,294) over 258,817 bytes; Falcon-H1-Tiny 1.5695, MobileLLM-R1.5 1.4998 over 258,814 bytes (UTF-8 cuts at row ends dropped); MobileLLM's tokenizer adds no BOS, so its first token per row is context only | BPB; HA `bpb_web_ours` |
| C324 | 7.4 | ClimbMix share of `final`'s training tokens | 62.4% (summed token counts of the 594 logged loader blocks) | FLOG `loader_block.tokens` (computed) |
| C308 | 7.4, E.4 | decoding studies | W/L/T as in Table E1: ties 49–58 of 60 per `final` setting (46–55 "both bad"), 48 for `v9` | `N/study/proxy/reports/ab/s1-*-vs-default--opus-1.json`; EXD |
| C202 | 7.5 | R1–R12 | **[TBD]** | — |
| C310 | E.2 | selection bounds (changed: item 20) | log 6 = 1.79 nats for any rule selecting one of K samples; log K − (K−1)/K = 0.96 nats is the best-of-K KL under a continuous score and an upper bound in general [69] | `docs/MATH.md` F4; `N/paper/refs.md` [69] |
| C176–C179, C182–C184, C135 | E | E5, E9, decoding defaults, decoding on the chip | as stated | `docs/E5_SELF_GENERATION.md`; `docs/E9_DECODING.md`; `R/bes/config.py` DecodeCfg; `docs/CHANGES.md` l.279 |

## Training under a wall clock, data (Appendices C and D)

| id | § | claim | value | source |
|---|---|---|---|---|
| C080–C083, C085 | C.1 | schedule, curriculum, the moved end (events: 0.2415 → 0.2190; factor 0.783 → 0.803) | as stated | `R/bes/train/schedule.py`; FLOG events |
| C311 | C.1 | step-record values across both jumps (definition: item 29) | 0.2413 → 0.2196; 0.5045 → 0.5118 | `FIGURES.md` F1 reading; NUM `p.graft` |
| C086–C088 | C.2 | checkpoints, guard, new-machine errors | as stated | `R/bes/config.py` RunCfg; `docs/CHANGES.md` l.299–302 |
| C312 | C.2, C1 | runtime 3 idle before training resumed; restore record | 48 min 19 s; 01:54:14 | `tables.md` T5b |
| C313 | C2 | event times (stop → restore records) | as in Table C2 | `FIGURES.md` interventions table; `tables.md` T5b |
| C091 | C2 | mean training rate to the graft | 241,065 tokens/s | computed from FLED |
| C314 | C3 | Table C3 with sample sd; tape gain weighted (changed: item 30) | first window sd 0.535 (population 0.507); tape gain 0.319 | `RW1/win.py` on FLOG; NUM `loss.final_p000_001` |
| C315 | C.3 | `v9` step losses after the graft (changed: item 27) | mean 2.4675, sd 0.0399, n 93 (steps 29,750–34,350) | NUM `loss.v9_to_34350` |
| C094 | C.3 | tape switched off at evaluation, micro ablation | +10.49% | `docs/E2_ABLATIONS.md` l.108 |
| C316 | C.3 | held-out tape gain at 29,722 | web 0.0127, chat 0.0286 | `N/midrun_eval_1.json` |
| C095, C096 | C.4 | DPO settings and data | as stated | `R/bes/config.py` PrefCfg; `R/bes/train/prefopt.py` |
| C317 | C.4 | test replies about 41 minutes on 42 workers (plan from a dev timing) | as stated | `N/OPERATIONS.md` ("Since 12:10") |
| C159–C161, C164 | D.1 | registry, weights, exclusions, hindsight 12% | as stated | `docs/DATA_SOURCES.md`; `R/bes/config.py` l.221 |
| C318 | D.1 | realised shares, plain means over 594 blocks (changed: item 30) | 75.04 / 14.46 / 5.24 / 3.04 / 1.32 / 0.89% | `RW1/mix.py` on FLOG |
| C319 | D.1 | shares in the first 48 blocks after the graft | 75.5–76.0 / 14.8–17.1 / 3.8–6.2 / 2.7–4.4% | `RW1/mix.py` on the 11:02 `v9` snapshot |
| C165–C168 | D.2 | the refusal sample, the Tulu 3 subsets, the rule from step 6,647 | as stated | `N/midrun_sample_step5075_p6.md`; `docs/CHANGES.md` l.310; commit `566517a` |
| C320 | D.2 | chronology: rule live at 02:28:59, scan at 08:31 (changed: item 11) | as stated | `tables.md` T5b; file times of `N/deflection_check2.json` and `S/deflection_out.txt` |
| C167, C169–C172, C174 | D.2–D.5 | openings and deflection-rule shares; UltraFeedback 2.44% | as stated | `N/openings.json`; `N/deflection_check2.json`; `S/deflection_out.txt`; `S/openings_out.txt` |
| C321 | D.4 | hand labels of 60 dropped openings (new: items 12, 42) | P 8, H 45, C 3, S 2, A 1, L 1; all P in the system chats | DL |
| C322 | D.5 | no stock refusal or deflection in `final`'s 75 dev replies | 0 / 0 | EXD |

## Reproducibility, licences (Appendices J, K)

| id | § | claim | value | source |
|---|---|---|---|---|
| C218–C222 | J | commits, configuration hashes, command line, per-run files | as stated; `fast` of `v9` with `flat=1 rows=1 tokn=1` | FMAN; VMAN; `git log`; `N/OPERATIONS.md` |
| C323 | J | **changed in the third pass → C437.** test functions at `8a86bcb` (changed: item 27) | 1,356 | `grep -rc "def test_" tests` in `R/` |
| C223, C224 | J | determinism | as stated | VMAN `benchmark.canary.determinism`; `docs/SPEC_MODEL_TRAIN.md` l.140–142 |
| C227–C229 | K | ClimbMix licence; third-party outputs; credential handling | as stated | `docs/DATA_SOURCES.md`; `docs/CHANGES.md` l.189 |

## Negative results (Appendix H)

| id | claim | value | source |
|---|---|---|---|
| C203 | GP attention (cost now "estimated": item 16) | 1e−15; ≤ 2e−4; 0.003–0.008; 63.6% vs 5.0%; 4.5–9× (estimated) | `docs/NEGATIVE_RESULTS.md` l.9–16 (adversarial review, real-text experiments and a cost model; nothing measured on the chip) |
| C204–C211, C213–C215 | other rows of Table H1 | as in the table | `docs/NEGATIVE_RESULTS.md`; `docs/E2_ABLATIONS.md`; `docs/V9.md`; `docs/E4_PROBES.md`; `docs/E9_DECODING.md`; `docs/PLAN.md` |

## Claims of the first draft that were withdrawn or changed

| first-draft claim | now | review item |
|---|---|---|
| title "Parameters Read Are Not Bytes Moved"; thesis that bytes "predict" the cost | narrowed to this implementation; tested post hoc on six configurations (Table 4) | 1, 4, 37 |
| time ratio "1.54–1.62" | 1.620 at the switch, 1.455 after the switches (synthetic); 1.580 / 1.429 live (medians) | 1, 26, 27 |
| "the cost is layout, not arithmetic" | 27% shown by removal; the rest is the cost model's attribution | 6 |
| graft check at 29,722 claimed with no record | a re-run of the same checkpoint and hash, stated as such | 3, 21 |
| adopted kernel's gradient differences 1.4e−2 / 1.5e−2, 7.5e−2 / 5.4e−2 | 2.8e−2 / 2.6e−2, 7.9e−2 / 5.7e−2; about twice the graph-only band | 13, 22 |
| seven test prompts seen, 193 untouched | eleven, 189 | 14 |
| held-out "144 rows", "proxy 16" | 140 rows, proxy 12 | 23 |
| cost model "5% high" | 14% high on the difference; Table 5 | 24, 38 |
| 1.132 GFLOP executed, 318 TFLOP/s, 39% | 1.18–1.19 GFLOP, 332–336 TFLOP/s, 41% | 25 |
| live +17.5% and 1.570 from single log lines | medians with n: +17.2%, 1.580 | 26 |
| "four pauses" | five, with train-log durations | 27, 29 |
| 1,349 tests; uncommitted `flat` and graft tests | 1,356; both in `769f240` | 27 |
| held-out and decoding study "[TBD]" | reported (Sections 6.5, 7.4; Appendix E) | 27 |
| 1.31e10 tokens, 2.8e9 fewer | about 1.36e10, 2.3e9 fewer (assumption stated) | 28 |
| "±7 points" resolves the test | n = 200 detects ≥ 11 and ≥ 16 points at the dev tie rates | 32 |
| "a protocol that cannot be tuned after the fact" | removed; deviations listed (Section 7.2) | 9 |
| "written before any run of ours had trained" | predates every TPU run and full-scale reply; micro runs earlier | 10 |
| informal reading of 7 prompts as the standing | replaced by the blind dev comparison and exact scores | 2, 20 |
| "entity look-ups stayed high" | full before/after table | 17 |
| loader horizon "right" | about 1.3 times too long | 19 |
| "13 measured non-improvements" | 13 changes measured and not adopted | 20 |
| selection bound log K − (K−1)/K "for any selection" | log K in general; the smaller value for best-of-K under a continuous score | 20 |
| GP attention cost "on the TPU" | estimated | 16 |
| "31.24 GB" | 31.24 GiB | 30 |

## Numbers that are projections or derivations, not measurements

C005, C016, C034, C041, C091, C141, C248, C251, C256, C266, C273, C274, C277, C279, C280, C282, C284,
C285, C294, C301 are arithmetic on measured or configured inputs. C128, C129, C272 (cycles and bytes)
and the predicted side of C275 are compile-only estimates of the compiler. C258 is the design's own
prediction.

## Open items ([TBD] in the paper)

Final pass (22:10 UTC): authors; the registered judge's full pass on the test pairs (155 and 92 of 200 rated);
MobileLLM-R1.5's two K = 6 control arms (in generation). Everything else listed below is closed (C500–C546). The text below is the fourth
pass's.


Updated after the check (18:40 UTC). R1–R12 (Appendix M; C202, C444, C455–C457); authors; the tokens and
final step at 18:33:49 (Section 6.3, Tables C2, M9); what each post-training stage did (C.4, M7, M8);
decoding throughput of the export (E.5, M12); the share of flagged test replies (D.5); the licence review
(K); the test comparison in the abstract, Section 7.4 / Figure 7 and Section 9. Closed since the second
pass: the registered judges on `dev` (C421–C425) and the live `v9` rate at 2,048 tokens (C405). No
longer open but recorded as deviations: the `neutral` arm of MobileLLM-R1.5 on `dev` (C451) and the
protocol's K = 1 result and matched controls (C452).

## Third pass: new and changed claims (C400–C448)

Written 24 November 2026, 17:30–18:15 UTC, for the third draft. `L3/` = `N/paper/lead3_work/`.
`NUM3` = `L3/numbers.json` (written by `L3/make_numbers.py`, a copy of the second pass's script reading
`L3/v9_snap_1628/`, the train log, ledger, run log and manifest extracted from the snapshot `N/snap_main`
created 16:28:12 UTC; newest step record 46,350). `VLOG3` = `L3/v9_snap_1628/run/train/train_log.jsonl`.
`RD` = `L3/readers.json` (`L3/readers.py`: the repository's own key check and tally over
`N/study/proxy/ratings/`). `MG` = `L3/micrograft.json` (`L3/micrograft.py` on
`N/micro_graft/6M/results_all.jsonl`). `H3` = `L3/heldout3.json`; `H46` = `L3/heldout46.json`. `MAL` =
`L4/make_all_1731.log`, a copy of `N/paper/figures/_work/make_all.log` of the rebuild `make_all.sh
--cutoff 46350` at 17:31–17:32 UTC (this entry first named the 16:57:45 log, which the 17:31 rebuild
overwrote; every MAL line cited below is in the 17:31 copy, checked after the check). `MAL4` =
`L4/make_all_1825.log`, the rebuild of 18:25 (`make_all.sh --no-extract`, default cut-off now taken from
NUM3; by `diff` it differs from MAL only in the cut-off lines, the 2,048-token MISMATCH, which is gone,
the count of mismatches (2 → 1) and Figure 5's 46,178 points).
"Recompute" gives the command; every script runs with
`taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B <script>` from `L3/`.

Section, table and figure numbers below are the third draft's. Renumbering against the second draft:
Figure 2 (kernels) → Figure F1; Figure 3 (waterfall) → Figure F2; Figure 4 (throughput against time) →
Figure C2; Figures 5, 6, 7, 8, 9 → Figures 2, 3, 4, 5, 7; new Figure 6 (micro graft, `F12_micrograft`);
Table 5 (cost model by change) → Table F5; Table 6 (function check) → Table G1; Table 7 → Table 5 (rows at
34,166 → Table G3); new Table 6 (micro graft); new Table 7 (readers); Table 8's exact scores → Table 8;
Table G1 → Table G2; old §3 → Appendix N, old §5.6 → F.6, old §5.7 → F.4. The first- and second-pass ids
above keep their section column as written then.

| id | § | claim | value | source; how to recompute |
|---|---|---|---|---|
| C400 | status, 2.4, C | newest record read; snapshot; newest held-out record | step 46,350 at 16:27:21 UTC, p 0.8794, loss 1.9463; ledger 46,351 steps, 12,150,636,544 tokens at 16:27:24; snapshot created 16:28:12; held-out record of step 46,178 written 16:25:50 | VLOG3 (last step record); `L3/v9_snap_1628/run/train/ledger.json`; `L3/v9_snap_1628/snapshot_meta.json`; `N/heldout_v9.jsonl` (last line, `time`); NUM3 `v9.newest_step`, `v9.newest_ledger` |
| C401 | status, 2.4, 6.3, C2 | training ends at 18:33:49 UTC | setup event `train_end` 1795545229 = 18:33:49; the operator's notes and the task text say 18:33:45 (a summary; the raw record is used) | VLOG3 setup events; NUM3 `proj.v9_total_from_newest.train_end_utc`; `N/OPERATIONS.md` ("Endgame (v9)") |
| C402 | 2.2 | the logged rate's definition; achieved rate within 0.13% of the live median in every stretch | 262,144 ÷ running average of the step time (weight 0.2), training time of the ledger; largest gap 0.124% (2,048-token stretch: 189,074 live, 188,840 achieved) | `R/bes/train/train.py` l.903, 965 (as `FIGURES.md` cites them); MAL "largest gap"; NUM3 `rate.*` (`achieved_median`) |
| C403 | 5.2, Table 2, F.3, H1 | the reference implementation's first rates (changed: was "112–117 k (monitor readings)") | two monitor readings 115,394 and 116,718 tokens/s (04:28:00, 04:17:59 on 23 Nov); start-up benchmark 111,893; `docs/TPU_NOTES.md` summarises 112–117 k and `v9` 88–92 k; `v9`'s benchmark on disk 88,146 | `S/colab_bes24h_final.log` l.73–74; `docs/TPU_NOTES.md` l.187–189; `docs/CHANGES.md` §19.1; `FIGURES.md` (F9 and F2 readings) |
| C404 | 5.2 | live rate of `final` rose 2.41–2.44× from the reference to the one-tile kernel (changed: was 2.40–2.51×) | 281,399 / 116,718 = 2.411; 281,399 / 115,394 = 2.439 | NUM3 `ratio.reference_to_kernel_live` |
| C405 | Table 2, 5.2, 6.3, Figs 2, F2, C2 | `v9` at 2,048 tokens (changed: was 188,980, n = 16) | live median 189,074 tokens/s, n = 190 (steps 36,900–46,350, 12:48:36–16:27:21), range 188,467–193,886; achieved 188,841 (n 189); 3.96% below 196,874 at 1,024 | NUM3 `rate.v9_switches_2048`; MAL (`v9_2048`) |
| C406 | Figure 2 | the 2,048-token live tick | 262,144 ÷ 189,074 = 1,386.5 ms (n = 190) | MAL (`fig4_v9cost.py`: "v9 switches 2048 1386.5") |
| C407 | Table 4, Figure 3a | leave-one-out error of FLOP and of reads (changed: FLOP was "20% (66%)") | not defined without `final` (the other five have equal counts); on the five `v9` configurations 10.3% (21.4%), identical for FLOP and reads | MAL ("leave-one-out flop: mean 10.3%, largest 21.4% over 5 configurations"; the MISMATCH line on `predictor_test.json`); `FIGURES.md` known discrepancy 1 |
| C408 | 6.3, 6.7, Table C2, M9, H1, G.3 | token projections (changed: were 1.360e10 and 2.28e9 / 14.4% from the 10:57 restart against 1.588e10) | `v9` run total at 18:33:49: 1.358 × 10¹⁰ (51,821 steps) = ledger 12,150,636,544 + 7,584.4 s × 189,074; `final` continued: 1.591 × 10¹⁰ (slice 30,969 s, 281,399 at 1,024, 252,972 scaled at 2,048); deficit 2.33 × 10⁹ (14.6%); remaining after the graft about 5.8 × 10⁹ (13.58 − 7.79) | NUM3 `proj.v9_total_from_newest`, `proj.final_total`; projections, not measurements |
| C409 | 6.1 | `v9` from scratch in the slice left at the graft | about 6.0 × 10⁹ (6.02e9) with the later switches; 5.4 × 10⁹ at the rates of the switch (C251) | NUM3 `proj.v9_scratch_at_graft` |
| C410 | 2.4, C.2, Table C1 | TPU time at the cut-off | runtime 3: 55,290 s to 16:27:24; total 70,835 s = 19.68 h; training seconds of the kept trajectory 55,310.2 = 15.36 h (78.1%) | NUM3 `tpu.at_newest` (`S/tpu_intervals.txt`; ledger) |
| C411 | 2.4, 6.5, C.1, C.3, Fig C1 | the chat ramp from p = 0.67 | chat share 0.25 until p = 0.67, linear to 0.70 at p = 0.92; base-text rows 75% in the loader block of step 36,850 and 38.4% (38.45% planned) in that of step 46,350; logged chat share 0.6269 at step 46,350 | `R/bes/train/schedule.py` `chat_frac_at`; `R/bes/config.py` DataCfg (`chat_frac`, `chat_frac_tail`, `tail_from`); VLOG3 `loader_block.row_share`, `chat_frac` |
| C412 | C.3 | training loss over the 2,048-token stretch | 190 records; mean of the first 5 2.404, of the last 5 1.954; learning-rate factor 0.3964 (step 36,900) → 0.2086 (step 46,350) | MAL (fig1: "stretch ('v9', 3) … first 5 2.4038, last 5 1.9539"); VLOG3 `lr` |
| C413 | Figure C1 caption | record counts at the cut-off | 594 records of `final` (50–29,700) and 333 of `v9` (29,750–46,350); a line is now drawn at 2,048 tokens | MAL (fig1 "records: final 594, v9 333") |
| C414 | Figure D1 caption, D.3 | the live loader's skips to the cut-off | e.g. system chats 29.82% of 44,541 reads, UltraChat 9.33% of 77,492, Tulu 3 6.30% of 257,801, SYNTH 1.22% of 254,236; scan within 0.25 points | MAL (`fig5_openings.py` lines) |
| C415 | 6.5, Table 5, Table G2, Fig 5 | runtime held-out evaluations to step 46,178 (new rows 36,422 … 46,178) | e.g. 46,178: web 2.6867, chat 1.3790, chat assistant 1.2651, proxy 2.4614, web + chat 2.0588, all 2.0914; 917.9 training minutes; 16 `v9` checkpoints and 3 `final` | H3 `rows` (from `N/midrun_eval_1.json`, `N/heldout_v9.jsonl`); recompute `heldout3.py` |
| C416 | 6.5, Table 5 | comparisons at 46,178 (bootstrap over rows, seed 20261124, 10,000 resamples) | since the graft −0.1679 [−0.1828, −0.1533]; web −0.1019 [−0.1106, −0.0937]; chat −0.2393 [−0.2560, −0.2235]; pages off − on +0.2166 [0.2019, 0.2320]; pages off − `final` 29,722 +0.0487 [0.0394, 0.0582] (web +0.0623, chat +0.0340); 46,178 − 35,294 −0.1073 | H46 `paired`; recompute: `heldout46.py` from the saved per-position losses `L3/heldout/v9_46178_on.npz` and `v9_46178_pagesoff.npz`, now the only local copy of this evaluation: the 16:28 snapshot that held checkpoint 46,178 is gone (the snapshots rotate; at 18:29 UTC `N/snap_main.prev/parts.json` holds the snapshot of 17:39:40 with checkpoint `ckpt-000019-step-000049424`, and `N/snap_main` that of 18:05:14 with `ckpt-000020-step-000050506`); the npz files were written by `heldout_eval.py --manifest L3/v9_snap_1628/run/manifest.json --ckpt <that snapshot> --variant on: --variant pagesoff:page=0,gain=0` |
| C417 | 6.5, Table 5, Fig 5 | `final`'s three-point line at the newest checkpoints (**changed after the check**: log-token line recomputed with the rates of 6.3, and which checkpoints lie inside each interval) | minutes: 2.1470, gap −0.088 ± 0.171 at 917.9 min, inside at every checkpoint; log tokens at equal time, `final` at 281,399 tokens/s for 10,061 s after the graft restart and then 252,972 (NUM3 `proj.final_total`): `final` 1.383 × 10¹⁰ tokens, gap −0.115 ± 0.107 at 46,178, outside from step 41,850 on (the third pass ran `final` at 281,220 throughout: 1.419 × 10¹⁰, −0.112 ± 0.111, also outside); at equal tokens −0.127 ± 0.084, outside from step 34,166 on (at 35,294: −0.044 ± 0.040); at 35,294 the first two unchanged (−0.033 ± 0.066, −0.038 ± 0.052), inside | `L4/heldout4.json` (`L4/scripts/heldout4.py N L4/heldout4.json`); H3 for the old log-token line; MAL4 (fig8 lines) |
| C418 | 6.5 (iii), G.5 | input-frequency quarters at 46,178 (section reference corrected after the check: was "6.5 (iv)") | pages: 0.299, 0.241, 0.170, 0.159 nats (rarest first); since the graft: −0.152, −0.153, −0.167, −0.199 | H46 `frequency` |
| C419 | 6.5, Table G2 | local evaluations reproduce the runtime's (**changed after the check**: was "within 10⁻⁷", which the proxy rows at 46,178 exceed) | web and chat rows within 3 × 10⁻⁸ at all four (largest 2.96 × 10⁻⁸, web at 46,178; 2.72 × 10⁻⁸, web at 29,722); proxy rows 2.20 × 10⁻⁷ at 46,178, 8.4 × 10⁻⁸ at 35,294; all rows 2.2 × 10⁻⁹ at 46,178 | `L4/heldout4.json` `local_vs_runtime` (the four `*_on.npz` of `L2/heldout/` and `L3/heldout/` against `N/midrun_eval_1.json`, `N/heldout_v9.jsonl`) |
| C420 | Table 8, 7.4 | bits per byte of ours on the web rows at 46,178 | 0.9801 over 258,817 bytes | H46 `bpb_web` |
| C421 | Table 7, abstract, 1, 7.4, 9 | every reader on the four development pair sets | as in Table 7 (e.g. `v9` vs Falcon-H1-Tiny: Opus 3 / 39 / 18; `qwen3-4b-2507` 1 / 16 / 43, 0.375 [0.264, 0.502] (**changed after the check**: the upper bound is 0.501515, printed 0.501 by rounding 0.5015 a second time), p 0.0003, orders 60.0%, first −0.114; `gemma4-e4b` 0 / 3 / 57, 0.475 [0.354, 0.599], p 0.25, 83.3%, −0.065) | RD `sources`; equal to every `N/study/proxy/reports/ab/*.json` of these sets; recompute `readers.py` |
| C422 | abstract, 1, 9 | at step 31,912 (p 0.56 when the checkpoint was written): Falcon-H1-Tiny preferred in 39 of 42 (Opus) and 16 of 17 (registered judge) decisive pairs; ours won 20 of 27 (sign test p = 0.019) and 7 of 9 (p = 0.18; 51 of 60 pairs tied) against MobileLLM-R1.5 (**changed after the check**: p values, ties and checkpoint added) | p: exact two-sided sign test 0.019157 (20/7) and 0.179688 (7/2); the end event of step 31,912 at p 0.5623, the restore's stage event at 0.5651 | RD; `L4/readers4.json` (`L4/scripts/readers4.py N L4/readers4.json`); VLOG3 events `end` and `stage` of step 31,912 |
| C423 | 7.4 | pair-by-pair agreement, `v9` sets | Opus (two readings) and `qwen3-4b-2507`: Falcon-H1-Tiny both a side 16 (same 14, opposite 2), Opus side / judge tie 26, judge side / Opus tie 1, both tie 17; MobileLLM-R1.5 6 (6 same), 21, 3, 30; κ (win, loss, tie) 0.209 and 0.183; `gemma4-e4b`'s 3 and 3 sides all agree with Opus | RD `agreement` |
| C424 | 7.4 | the second judge gave no side in 93–97% of pairs; both automatic judges clipped 30 of the 60 MobileLLM-R1.5 pairs | ties 56, 58, 57, 57 of 60; `clipped` 30 (s1 and s2, both judges) | RD |
| C425 | 7.3 | tie rates of the registered judge and what n = 200 / 189 resolves at them | 43 and 51 of 60 (71.7%, 85.0%); 0.5 + 0.182 and 0.246 (n = 200), 0.187 and 0.252 (n = 189) | RD; NUM3 `power.*_qwen_judge` |
| C426 | Table 7, 7.4, abstract, 1, 9 | the reference: Falcon-H1-Tiny against MobileLLM-R1.5 on the 200 test prompts, Opus two readings | 144 / 6 / 50; 0.845 [0.788, 0.889]; decisive 0.960 [0.915, 0.982]; orders 93.5%; first +0.0125; chat 37 / 1 / 12, code 17 / 2 / 11, instruct 31 / 0 / 9, maths 15 / 3 / 12, writing 44 / 0 / 6; two-turn 39 / 1 / 12 | RD `ref-falcon-vs-mobilellm-test`; `N/study/proxy/reports/ab/ref-falcon-vs-mobilellm-test--judge-opus.json` |
| C427 | 7.4 | the readers' notes on the reference pairs describe MobileLLM-R1.5's reply as looping or repeating, and mention a boxed answer, often nonsense or looping (**changed after the check**: was "a bare boxed answer", which the count does not support) | loop 87 and 83, box 46 and 49 of 200 (first and second reading; pairs); of the box pairs, 32 and 40 have the same clause also matching loop, repeat, nonsense, degenerate, fragment or incoherent | `L3/ref_notes.json`; `L4/ref_notes4.json` (`L4/scripts/ref_notes4.py N L4/ref_notes4.json`, same clause attribution; a keyword count) |
| C428 | 7.4 | in absolute terms | s2 (`v9` 31,912) first reading: both bad in 44 of 60 pairs of 3-gram against default (notes: wrong facts or arithmetic 20, incoherent 14); 31 of 60 against MobileLLM-R1.5; 14 of 60 against Falcon-H1-Tiny; s1 variants 46–55 of 60 | `L3/ours_notes.json`; RD (`opus-1` `raw_verdicts`) |
| C429 | 6.6, G.4, Table 6 | micro design | d 128; 2 blocks × 8 loops; rows of 512; 4,096 tokens per step; 1,464 steps; 5,996,544 tokens trained; `ref` 4,402,372 numbers, `v9_pages` 14,904,516, new leaves 10,502,144 (5 leaves), 59 shared leaves (4,402,372), 175 arrays copied; grafts at steps 366, 732, 1,098; evaluations at 1,499,136, 2,998,272, 4,497,408, 5,996,544 tokens; token neurons 8 in 32 dimensions; mixture 0.75 / 0.25 rows; token shares 0.780 / 0.220; pools 8,192 / 4,096 rows; held-out 32 + 32 rows | MG `meta`, `checks`; record fields `config`, `bench`, `data`, `graft.leaves` |
| C430 | G.4 | passes over the pools | 1.07 (web) and 0.71 (chat): 0.75 × 11,712 = 8,784 of 8,192 rows and 2,928 of 4,096 (11,712 = 1,464 steps × 8 rows) | arithmetic on MG `meta`; `R/bes/eval/microbench.py` (`BatchMaker`, `passes` = 2 and 1 started) |
| C431 | Table 6, 6.6, 6.7, abstract, 1, 9, Fig 6 | the micro results | as in Table 6 and G.4 (e.g. `v9_pages` − `ref` −0.0519 ± 0.0107, −1.24%; shares 1.06 / 0.69 / 0.24, per seed 0.86, 1.33, 1.08 / 0.62, 0.87, 0.64 / 0.20, 0.29, 0.24; graft − `ref` per seed −0.0539, −0.0551, −0.0558 / −0.0386, −0.0359, −0.0327 / −0.0127, −0.0120, −0.0122) | MG `table`, `diffs`, `share`; equal to `N/micro_graft/6M/graft_report.txt` in every printed digit (compared line by line; the figures agent's check finds its 1,013 numbers equal) |
| C432 | 6.6, G.4 | the checks inside the micro runs | every graft arm's 1, 2 or 3 evaluations before its graft equal `ref`'s of the same seed row for row (18 of 18); the graft check's difference 0.0 at 9 of 9 grafts; held-out before/after difference 0 per domain and per row; check losses 5.0309, 5.0072, 5.0118 / 4.7603, 4.7661, 4.7617 / 4.6126, 4.6162, 4.6532 | MG `checks` |
| C433 | 6.6 | learning-rate factor at the graft | 0.775, 0.55, 0.325 | MG `checks.*.carried.lr_factor` |
| C434 | 6.6, G.4 | the live run's rate during the micro runs | 189,054 tokens/s (n = 73) before 14:12:51; 189,086 (n = 117) from 14:12:51 to the cut-off 16:27:21; the runs' last 14 minutes (to 16:41:01) and the restart arms (16:42:47–17:29:56) are after the cut-off and not covered (**changed after the check**: the coverage is now stated) | `L3/rate_during_micro.json` (`rate_during_micro.py`; `last_record_utc` 16:27:21) |
| C435 | 8 | `ref`'s spread over three seeds (**changed after the check**: "spread by 0.18%" was the sd) | sd 0.0076 nats = 0.18% of 4.1912; range 4.1830–4.1979, 0.0149 = 0.36% | MG `table`; `L4/micro4.json` `ref_spread` |
| C436 | 6.6, 6.7 | the micro arms that never shared a state differ by about 0.01 between seeds | sd of `v9_pages` − `ref` 0.0107; of `graft_*` − `v9_pages` 0.0096–0.0113 | MG `diffs` |
| C437 | J | test functions at `b1e9140` and `190e0b2` (changed: 1,356 at `8a86bcb`) | 1,367 | `grep -r "def test_" tests | wc -l` in `R/` (at `190e0b2`, after the check) |
| C438 | J | the controlled experiment's commits (**changed after the check**: the restart records added) | `b04478e` (graft arms), `b1e9140` (results of the 15 runs), `190e0b2` (restart arms' records; HEAD) | `git log` in `R/` |
| C439 | Figure 5 caption | arrow label at the newest checkpoint | −0.088 ± 0.171; the label is cut at the right edge of `F8_heldout.png` in this build | MAL (fig8); the figure file |
| C440 | Figure 6 caption | values in the micro figure | as C431; drawn by `fig12_micrograft.py` | `data/micro_graft_summary.csv` |
| C441 | C, Table C2 | step 36,867 is at 9.66 × 10⁹ tokens | 36,867 × 262,144 = 9,664,462,848 | arithmetic |
| C442 | 6.6 Table 6 restart row, 6.7, G.4 | restart arms (`v9_pages` from scratch on 3 × 10⁶ tokens, seeds 0–2) | restart minus `graft_50` +0.2342 ± 0.0075 (per seed +0.2348, +0.2415, +0.2265); minus `ref` (6M) +0.1985 ± 0.0062 (+0.1962, +0.2055, +0.1938); minus `v9_pages` (6M) +0.2504 ± 0.0074; loss 4.3897 ± 0.0017 (4.3891, 4.3885, 4.3917); minus `ref` at 3M (mid-schedule) −0.0844, −0.0835, −0.0836; 732 steps, 2,998,272 tokens; the figures agent's `fig12_micrograft.py` prints the same (MAL, lines 'F12: restart …') | `N/micro_graft/3M/results.tar.gz` unpacked into `L3/micro3M/`; `micrograft.py L3/micro3M` |
| C446 | 6.5 (i) | since the phase change the chat rows fall faster than the web rows; candidate causes | assistant turns 1.4355 (36,422) → 1.2651 (46,178), −0.170; web 2.7254 → 2.6867, −0.039; loader epochs at step 46,350 (mean over streams) above 1 for smoltalk_systemchats 1.55, smoltalk_rewrite 1.48, smoltalk_constraints 2.56, synth 1.32 (and non-chat sources), below 1 for smol_smoltalk 0.91, magpie 0.47, Tulu 3 0.38, UltraChat 0.46 | H3 `rows`; VLOG3 `loader_block.epochs`; `N/FACTS_2.md` (16:52) names the candidate causes except the chat ramp |
| C447 | G.4 | the restart's own schedule against `v9_pages` at the same token count | −0.0509 ± 0.0086 (per seed −0.0411, −0.0555, −0.0563); `v9_pages` at 2,998,272 tokens 4.4301, 4.4440, 4.4479 | MG `table`, `restart` (inline check in `RESPONSE_2.md`); `N/FACTS_2.md` (17:35) gives the same |
| C448 | G.4 | the restart reads the first half of the 6M runs' data again | the row plan `row_kinds` is a prefix-stable smooth round robin and the row index depends on the seed and the count only | `R/bes/eval/microbench.py` l.607–656; `N/FACTS_2.md` (17:35) |
| C443 | 5.1 | whole steps are single timings (changed: "we report medians" applied to all) | mean of `--step-calls` back-to-back steps after a warm call, default 2, count used not recorded | `R/scripts/tpu_kernels.py`; `FIGURES.md` ("How a whole step was timed") |
| C444 | Appendix M | the files the shells name | as in Appendix M | `N/OPERATIONS.md` ("Endgame (v9)", "Final blind evaluation: the mechanics", "Since 12:10", "State at 16:50"); `S/scripts/final_pairs.sh`, `final_score.sh`, `endgame_testgen.sh`, `cmp_exports.py`, `fetch_records.sh`; `R/bes/export.py` (`export.json`) |
| C445 | 6.6 | the micro code was reviewed adversarially before the runs | trunk equality, function preservation, carried optimiser state, schedule and data order, unchanged E2 hashes, report arithmetic | `N/FACTS_2.md` (16:50); `W/micro_graft.md`; `W/micro_graft_work/review/` |

| C449 | abstract, 1, 6.6, 6.7, 9, Table 6, Fig 6 | the micro shares and the graft at a quarter, with their spread (after the check) | `graft_25` − `v9_pages` per seed +0.0088, −0.0138, −0.0043, mean −0.0031 ± 0.0113 (sd); 95% t interval of the mean (2 df, t = 4.3027) ±0.028 = 54% of the page gain 0.0519; per-seed shares 0.86–1.33 (`graft_25`), 0.62–0.87 (`graft_50`), 0.20–0.29 (`graft_75`) | `L4/micro4.json` (`L4/scripts/micro4.py N/micro_graft L4/micro3M_raw L4/micro4.json`; `results_all.jsonl` checked equal to every arm's own records) |
| C450 | 6.6, 6.7, 9, Table 6, G.4 | the restart arms against the 6M arms (after the check) | restart − `graft_50` per seed +0.2348, +0.2415, +0.2265 (range +0.227 to +0.241), mean +0.2342 = 4.5 × the page gain; − `ref` +0.1985 (3.8 ×); − `v9_pages` +0.2504 (4.8 ×); `graft_50` − `ref` −0.0358; the restart: 732 steps, 2,998,272 tokens, budget and horizon 3,000,000, learning rate decaying to 0.1 of its peak by its end (`final_lr_frac` 0.1), last record finished 17:29:56 | `L4/micro4.json`; the archive `N/micro_graft/3M/results.tar.gz`, unpacked into `L4/micro3M_raw/`, equals the copies in `N/micro_graft/3M/<arm>/`; record fields `config.optim`, `finished` |
| C451 | 7.2 (deviation 4), 7.1, M | the `neutral` arm of MobileLLM-R1.5 is not compared blind on `dev` before the test readings; the exact scores on `dev` favour the `recommended` arm | no such comparison in `N/OPERATIONS.md` ("Endgame (v9)", "Final blind evaluation: the mechanics", "State at 16:50") and no `neutral` pair set in `N/study/proxy/pairs/`; maths 4 against 1 of 9, constrained prompts met 2 against 1 of 14, replies cut 42 against 54 of 75 | `R/docs/EVAL_PROTOCOL.md` §2 ("credited with the better arm on `dev`, decided before `test` is looked at"); EXD (Table 8); `L4/exact_split_dev_check.json` |
| C452 | 7.1, 7.2 (deviation 6), 8.6, 9, M | the protocol's K = 1 result, library on/off and K-matched and selector-matched controls are not run | the protocol requires them (§3); the opponents' outputs hold one run per preset (`falcon…/recommended`, `mobilellm…/recommended`, `…/neutral`, 400 conversations each, no `--samples`); `N/OPERATIONS.md` schedules none; `v9` ships no library (E.2); the decoding selection may ship one particle (`N/OPERATIONS.md`, "Since 12:10": default, k1, noband); selection changes the distribution by at most log 6 = 1.79 nats (E.2); one particle against six on `final` 25,280: 1 / 3 / 56, p 0.63 (Table E1) | `R/docs/EVAL_PROTOCOL.md` §3 and §9; `N/study/proxy/outputs/` (file listing and line counts) |
| C453 | 6.5, M6, M10 | the held-out rows and the export's check | runtime evaluations: `heldout_batches` with 64 rows per domain (`rows` 64 in `N/midrun_eval_1.json` and `N/heldout_v9.jsonl`; web 64, chat 64, proxy 12 rows, T = 1,024); the export's check: `heldout_rows` = 16 by default in `export_run`, not passed by `run_24h.py`'s export stage; `check_16bit` stores per domain (web, chat, proxy) `loss32`, `loss16`, `delta`, `weight`, no assistant-token loss and no bits per byte; T = min(`data.seq_len` 1,024, `model.max_seq` 2,048) | `R/bes/export.py` l.550–660, 1701–1798; `R/scripts/run_24h.py` l.1460–1480; `R/scripts/midrun_eval.py` (`--rows`, default 64); `R/bes/config.py` |
| C454 | M4, M11 | the test split's counts (prompt metadata only; no reply opened) | 200 prompts, 252 turns, 52 two-turn, 30 maths, 48 constrained (`dev`: 47); without the 11 seen: 189 prompts, 240 turns, 28 maths, 44 constrained | `L4/exact_test_counts.json`, `L4/exact_test189_counts.json` (`L4/scripts/exact_split.py --count-only --split test [--exclude …]`) |
| C455 | M4, M11 | `exact_split.py` reproduces the `dev` exact scores | its summaries equal `EXD`'s for `v9` 31,912 default, Falcon-H1-Tiny, MobileLLM-R1.5 recommended and neutral, on the same 60 prompts | `L4/exact_split_dev_check.json` against `L2/exact_dev.json` (compared field by field) |
| C456 | M11 | `restrict_report.py` reproduces a report's `overall` from `per_prompt_score` | s2 vs MobileLLM-R1.5 (`judge-opus`) 20 / 7 / 33; s2 vs Falcon-H1-Tiny (`qwen3-4b-2507`) 1 / 16 / 43; s1 vs Falcon-H1-Tiny (`opus-1`) 2 / 43 / 15 with both bad among the ties | `L4/scripts/restrict_report.py R/<report>.json` (printed) |
| C457 | M6 | `bpb_web.py` reproduces our bits per byte | 1.0172 (29,722), 0.9975 (35,294), 0.9801 (46,178) over 258,817 bytes | `L4/bpb_web.log` |
| C458 | Fig 5 | what Figure 5 draws | rings at 29,722, 34,166, 35,294 and 46,178; pages off 2.2241, 2.2305 and 2.2754; arrow −0.0882 ± 0.1714, label inside the axes | MAL4 (fig8 lines); `L4/heldout_local46178.json` (`L4/scripts/local46178.py`) |
| C459 | Table E1, 7.4 | both bad in the `v9` decoding pair set (3-gram against default), first reading | 44 of 60 (A 5, B 7, tie 4) | `N/study/proxy/ratings/s2-ngram3-vs-default/opus-1.jsonl`; `L4/readers4.json` `raw_first_reading` |
| C460 | 7.2 (deviation 1), Table 7 | readings of the development sets by Opus | `s1-*` (`final`): `opus-1` only, one reading; `s2-*` (`v9`) opponent sets: `opus-1` and `judge-opus` (two readings) | `L4/readers4.json` `reading_files` (the files in `N/study/proxy/ratings/<set>/`) |
| C461 | Fig 7 caption | decisive pairs of the automatic judges on the `v9` sets | `qwen3-4b-2507` 17 (Falcon-H1-Tiny) and 9 (MobileLLM-R1.5); `gemma4-e4b` 3 and 3 | RD (Table 7) |
| C462 | 6.6 | the gain over `ref` in the 1.5 × 10⁶ tokens after each graft, and the learning-rate factor at the graft | 0.0192, 0.0161, 0.0123 at 0.775, 0.55, 0.325; `v9_pages` in its first 1.5 × 10⁶ tokens 0.0164 | `L4/micro4.json` `gain_after_graft` |
| C463 | 6.6 (Checks) | the per-row check covers the held-out domains only | `graft.heldout.<domain>.max_row_difference` = 0 for web, chat, assistant at all 9 grafts; `graft.check` (8 held-out web rows, `bes.model.model.loss_fn`) stores one aggregate `difference` = 0 | `L4/micro4.json` `graft_checks` |
| C464 | Table 7, abstract, 9 | the p of a development checkpoint is the progress at its stop event | 25,280: 0.4396 (0.44; restore 0.4402); 31,912: 0.5623 (0.56; restore 0.5651) | FLOG and VLOG3 events `end` and `stage` |
| C465 | 7.3 | the protocol's power table | columns "with 30% ties" and "with 50% ties" | `R/docs/EVAL_PROTOCOL.md` §7 |

### Withdrawn or changed in the third pass

| id | was | now |
|---|---|---|
| C240 | newest record step 37,650 at 13:05:57 | C400 (step 46,350 at 16:27:21) |
| C249 | "112–117 k" live rate of the reference | C403: 115,394 and 116,718 (two monitor readings); 111,893 is the start-up benchmark |
| C253 | achieved within 0.1% | C402: within 0.13% (0.124% at 2,048 tokens) |
| C254, C284 | about 1.36 × 10¹⁰ (1.360e10), deficit 2.28e9 (14.4%) against 1.588e10 | C408: 1.358 × 10¹⁰, 2.33 × 10⁹ (14.6%) against 1.591 × 10¹⁰ |
| C256 | TPU time at 12:00:00, 15.22 h | not in the third draft; C410 (19.68 h at 16:27:24) |
| C266 | 2.40–2.51× | C404: 2.41–2.44× |
| C273 | FLOP leave-one-out mean 19.6% (max 66.4%) | **withdrawn**: the fit is undefined without `final`; C407 |
| C292 | local reproduces runtime to 10⁻⁸ | C419: within 10⁻⁷ (still to 10⁻⁸ for 29,722, 34,166 and 35,294) |
| C294 | ±0.025 at 592.5 min for the three-point line | **withdrawn**: it belonged to a line through all 140 rows (`NUM3 heldout.three_point`, kept only as the old computation); the comparison set's line gives ±0.035 there (H3); no longer cited |
| C323 | 1,356 tests | C437: 1,367 at `b1e9140` |
| C330 | 188,980 (n 16) | C405: 189,074 (n 190) |
| C034 | 12.98 h at the graft, 8.98 h in training (69%) | still valid; no longer in the main text |
| C293 (34,166 rows) | Table 7 | moved to Table G3, values unchanged |


### Changed after the check (`REVIEW_2.md`), 18:10–18:55 UTC

| id | was | now |
|---|---|---|
| C417 | log-token line −0.112 ± 0.111, said to be inside "95% prediction intervals of ±0.05 to ±0.17" | −0.115 ± 0.107 with the rates of 6.3, outside from step 41,850 on; equal tokens outside from 34,166 on; the minutes line inside everywhere; "not resolved" rests on the line, not on the intervals |
| C292, C419 | to 10⁻⁸ (second pass), within 10⁻⁷ (third pass) | within 3 × 10⁻⁸ on web and chat; proxy 2.2 × 10⁻⁷ at 46,178 |
| C421 | [0.264, 0.501] | [0.264, 0.502] (also in `FACTS_2.md`, 13:52, which this pass does not edit) |
| C422 | no p values, no checkpoint | step 31,912 (0.56); p = 0.019 and 0.18; 51 of 60 tied |
| C427 | "a bare boxed answer" | "mention a boxed answer (often nonsense or looping)", with 32 and 40 of 46 and 49 |
| C434 | "during" | "during, to the cut-off at 16:27:21" |
| C435 | "spread by 0.18%" | sd 0.18%, range 0.36% |
| C438 | `b1e9140` | adds `190e0b2` |
| C442 | restart "worth more than four times" (0.234 against 0.052) | four to five times (0.199–0.250, C450); per-seed range +0.227 to +0.241 |
| C431 | "settles … about 0.7 at half" | 0.69 (0.62–0.87 by seed), on the CPU in float32 (C449) |
| abstract, 1, 9 | micro sizes without reach; Opus readers unnamed | width 128, 6 × 10⁶ tokens, equal tokens, page 1.24%, sizes do not transfer; Opus 5.5 agents, an addition (C449, C422) |

## Final pass: new and changed claims (C500–C550)

Written 24 November 2026, 21:10–22:15 UTC, for the final paper. `L5/` = `N/paper/lead5_work/` (scripts in
`L5/scripts/`, every one run as `taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B <script>` from `L5/scripts/`
unless stated). `SX` = `L5/snap_final_x/`, the records of the last snapshot `N/snap_final` (20:16 UTC) extracted
by `extract_snap.py` and checked byte-identical to `N/records/unpacked/`, `N/main16.final.log` and the official
export's copies (`L5/extract_snap.log`). `NUM5` = `L5/numbers.json` (`make_numbers.py`, a copy of
`lead3_work/make_numbers.py` reading `SX`); `FR` = `L5/final_results.json` (`final_results.py`); `PV` =
`L5/paper_values.json` (`paper_values.py`, values rounded once from full precision, exact halves up); `EXC` =
`L5/exact_counts.json` (`exact_check.py` on `L5/exact_test.json` and `L5/exact_test189.json`, written by
`lead4_work/scripts/exact_split.py --split test` with every run, with and without `--exclude` of the 11 seen ids);
`H5` = `L5/heldout5.json` (`heldout5.py`, after `run_heldout5.sh`, which runs `heldout_eval5.py` on both
checkpoints of `N/snap_final`); `XP` = `N/packs/official_export/x/final/export.json`; `TG` =
`N/testgen/<tag>/out/testgen.json`. The registered judge's values are those of the files as read at 21:49 UTC
(155 and 92 records, last written 21:34:37; `FR` `qwen_snapshot_utc`), the reports re-scored with `python -m
bes.eval.proxy score --pairs <set> --source judge-qwen3-4b-2507-3605803b` at 21:58 (155 and 92 rated pairs).

| id | § | claim | value | source; how to recompute |
|---|---|---|---|---|
| C500 | status, abs, 2.4, 6.3, C2, M9 | training ended at the deadline at step 51,812 with 13,582,204,928 tokens; `v9` part 22,090 steps, 5,790,760,960 tokens | end event `reason=deadline`, p 1.0, 18:33:54 (final checkpoint); setup event `train_end` 18:33:49; ledger steps 51,812, tokens 13,582,204,928 = 262,144 × steps | `SX/run/train/train_log.jsonl` (events `end`, `setup`); `SX/run/train/ledger.json`; NUM5 `v9.end`, `v9.final_ledger`, `v9.train_end_setup_event` |
| C501 | abs, 2.4, 9, C1, M9 | TPU time by runtime and in all; share in training steps | 7,325 + 8,220 + 70,509 = 86,054 s of 86,400 (346 unused); runtime 3 01:05:55–20:41:04; training seconds of the kept trajectory 62,893.1 (17.47 h) = 73.1% | `S/tpu_intervals.txt` (three closed intervals); ledger `seconds.train`; NUM5 `tpu.final` |
| C502 | C2, M9 | the ledger's other seconds and counters | checkpoint 23.2 s, set-up 279.6 s, lost 26,046.6 s, 10 restarts, 0 rollbacks | ledger; NUM5 `v9.final_ledger` |
| C503 | 2.4, C.4, C4, M7, M8 | the stages after training (UTC, slice, used) | sample 18:33:55–18:37:07 (180; 192.5 s); prefopt 18:37:07–19:06:51 (3,600; 1,783.8); select 19:06:51–20:01:47 (3,897.3; 3,295.3); export 20:01:47–20:02:22 (300; 35.0); critic and judge skipped; deadline 20:31:49, 29 min left (1,767 s) | `SX/run/stages.json`; NUM5 `stages` |
| C504 | 1 | 88 minutes after training | 18:33:55.4 to 20:02:22.0 = 5,306.6 s | `stages.json` (train `finished`, export `finished`) |
| C505 | C.4, C4, C5, 7.4, M7 | preference optimisation: data, candidates, checks, choice | β 0.1; 35,201 training and 1,826 held-out pairs (37,007 UltraFeedback after dropping 306 identical, 8,366 length ratio, 7,995 too long; 20 sampled, 1 too long); two candidates of 551 steps; held-out accuracy 0.500 → 0.6884 (se 0.0108) and 0.6944 (0.0108), needed gain 0.0108; held-out assistant loss 1.50427 → 1.51721 (+0.86%) and 1.52079 (+1.10%); length 284.9 → 282.7 (0.99×) and 270.0 (0.95×); rep4 0.2166 → 0.1856 (0.86×) and 0.1889 (0.87×); training 485.3 s and 423.8 s, training accuracy 0.677 and 0.685; both accepted, 3e-6 kept | `SX/run/post/prefopt_report.json`; `stages.json` prefopt result; `R/bes/train/prefopt.py` (checks), `R/bes/config.py` PrefCfg (bands); NUM5 `prefopt` |
| C506 | C.4, C4 | the sample stage | 17 held-out prompts of 12 sources (all `heldout: true`), about 28 candidate pairs, pool 3,996 | `SX/run/post/samples.jsonl` (17 lines); run log l.1010; `stages.json` sample result |
| C507 | 7.2, 7.4, C4, M8 | decoding selection | dev 200 prompts with prefopt weights; scores default 0.91, k1 0.85, noband 0.915; k1 better 1, worse 18, p 7.6e-5; noband better 2, worse 0, p 0.5; rule p < 0.05; chosen default; no library built; 19.6, 4.0, 31.3 min | `SX/run/manifest.json` `decode` (`vs_default`, `scores`, `rule`); run log l.1042–1105; `R/scripts/run_24h.py` `stage_select`; NUM5 `select` |
| C508 | 7.4, C4, M10, J | the export | `final/` from post/ckpt-000001-step-000051812 (prefopt); 64 leaves, 98,036,104 numbers in float16, largest relative L2 error 2.84e-4; check 2.34440 / 2.34443 over 43,138 tokens, delta +2.19e-5, tol 0.002; web 2.69236 / 2.69243 (+7.5e-5), chat 1.31725 / 1.31721 (−3.9e-5), proxy 3.28416 / 3.28419; `weights16.npz` 196,088,804 bytes, SHA-256 2da7cbd2e2f1fe49e805cb9bde62b15751f41ae92e71f00a3b0ba69f241afcf4 (recomputed with `sha256sum`) | XP (`check`, `weights`, `main_files`, `source`); `MODEL_CARD.md`; run log l.1109–1111 |
| C509 | M10 | read-set audit of the export | ledger 19,631,240; measured step mean 19,427,809 (web, 2,048 positions) and 19,427,956 (chat, 1,480); max 19,429,996; strict 24,352,024 (1,024) and 27,929,250 (2,048) on web, 23,363,294 on chat at 1,024 | XP `readset`, `accounting` |
| C510 | 7.4, C.4, J, M12 | the test generations | official: export of post/ckpt (fingerprint 8d19db7f…, same SHA-256 and decode choice as the run's own), 19:31:22–20:06:32, 2,109.8 s, 40 workers × 1 core, 252 replies; trained: 18:36:25–19:07:06, 1,840.4 s; official median 206.7 s and 155.5 tokens per reply, 0.567 s per step (mean 0.595); trained 159.4 s, 151.5 tokens, 0.550 s | TG `prefopt`, `trained` (`started`, `finished`, `wall_s`, `workers`, `cost`, `export`); `testgen.log` |
| C511 | J | the release | 12 parts, 196,956,160 bytes, whole-tar SHA-256 f82784404ab0dc46…; every part matches `parts.json` and equals the fetched part; the 21 files of the copy unpacked from the committed parts match `SHA256SUMS`; commits 99ba0e7…994b3a1 at 20:22:37–20:23:25; HEAD 34f12ac | `L5/scripts/verify_release.sh` (`L5/verify_release.log`); `git log` in `R/` |
| C512 | Table 2, 5.2, 6.3, Fig 2, F2, C2 | live rates at 2,048 tokens to the end (changed: was 189,074, n = 190, to the cut-off 46,350) | phase long 189,067 (n 226, steps 36,900–48,150), achieved 188,841; tail 189,035 (n 73, 48,200–51,800), achieved 188,851; 3.97% below 196,874; largest live/achieved gap 0.12% | NUM5 `rate.v9_switches_2048`, `rate.v9_tail_2048`, `ratio.v9_2048_vs_1024`, `rate.largest_gap_live_achieved_pct`; `L5/make_all_final.log` |
| C513 | C.1, 6.5, C2 | the tail from p = 0.92 | stage event step 48,190, p 0.9200, 17:09:57; weight by role at step 48,150: raw 5,127, system 52, user 1,781, assistant 7,545, library 419 (sum 14,924); at 48,200 and 51,800 system, user and library 0 | `SX/.../train_log.jsonl` (stage event; step records `micro.weight_by_role`); NUM5 `v9.stage_events` |
| C514 | C.3 | training losses to the end | phase long: first 5 2.4038, last 5 1.8149, lr 0.3964 → 0.1728, chat share 0.251 → 0.698; tail: 1.7783 → 1.7331, lr 0.1718 → 0.1003; records 594 + 442 | NUM5 `loss.v9_long`, `loss.v9_tail`, `records.counts`; `L5/make_all_final.log` (fig1) |
| C515 | 6.3 | the third draft's projection against the ledger; deficit against `final`'s counterfactual | projected 51,821 steps (1.358e10), actual 51,812 (9 fewer, 0.02%); deficit 1.591e10 − 1.3582e10 = 2.33e9 (14.6%), `final` still a projection | NUM5 `proj.third_pass_vs_ledger` (reads `lead3_work/numbers.json`) |
| C516 | 2.3, 8, 9 | token gap | 8e11 / 13,582,204,928 = 59; 4.2e12 / … = 309 | NUM5 `tokens.gap_to_opponents` |
| C517 | 6.5, Table 5, G2, G5 | runtime held-out records to 51,812 | 47,260 2.0840 (all) / 2.0501 (web + chat); 48,342 proxy 2.6651; 49,424 proxy 3.0764; 50,506 3.1946; 51,588 web + chat 2.0438, proxy 3.2241; 51,812 web 2.6690, chat 1.3668, assistant 1.1983, proxy 3.2580, web + chat 2.0438, all 2.1422, 1,048.2 min | `N/heldout_v9.jsonl`; H5 `runtime` |
| C518 | 6.5, G5 | local CPU evaluations of step 51,812 reproduce the runtime's | web −3.98e-8, chat +4.68e-8, proxy +2.32e-7, all +2.04e-8; within 5e-8 on web and chat at all five | H5 `local_minus_runtime_51812`; earlier steps C419 |
| C519 | 6.5, Table 5 | comparisons at 51,812 (bootstrap over rows, seed 20261124, 10,000 resamples) | 51,812 − 29,722: −0.1829 [−0.2002, −0.1663], web −0.1196 [−0.1286, −0.1112], chat −0.2514 [−0.2759, −0.2274], assistant −0.3181; 51,812 − 46,178: −0.0150 [−0.0208, −0.0091]; pages off − on +0.2631 [0.2442, 0.2829]; pages off − 29,722 +0.0802 [0.0674, 0.0938] (web +0.0741, chat +0.0869); prefopt − trained +0.0124 [0.0112, 0.0136] (assistant +0.0112) | H5 `paired` |
| C520 | 6.5, Table 5, Fig 5 | `final`'s line at every checkpoint to 51,812 | minutes −0.076 ± 0.227 (inside everywhere); log tokens −0.117 ± 0.129, `final` 1.581e10 tokens at equal time, outside 41,850–48,342, inside from 49,424; equal tokens −0.131 ± 0.103, outside from 34,166 | H5 `lines`, `outside` |
| C521 | 6.5 (iv) | what rose at the end | all-row minimum 2.0840 at 47,260; proxy 2.4683 → 3.2580; proxy weighted roles user 10,594, raw 500 of 11,094; chat non-assistant positions 1.8841 (46,178) → 2.1144 (51,812), assistant 1.2651 → 1.1983 | H5 `runtime`, `local` (`chat_not_assistant`), `proxy_weighted_roles`, `all_rows_min_v9` |
| C522 | 6.5, G5, E3, M6 | bits per byte on the 64 web rows | 51,812 trained 0.9736, prefopt 0.9778 (258,817 bytes); 46,178 0.9801; 29,722 1.0172 | H5 `bpb_web` |
| C523 | G5 | quarters by input-token frequency at 51,812 | page gain 0.356, 0.295, 0.213, 0.193; since the graft −0.171, −0.183, −0.140, −0.236 | H5 `frequency` |
| C524 | G.4 | the live rate during the micro runs, to the end | before 189,054 (n 73); during the 6M runs 189,077 (n 128); restart window before the tail 189,050 (n 23) | `L5/rate_during_micro5.json` (`rate_during_micro5.py`) |
| C525 | abs, 1, 7.5, 9, Table 7, M1, Fig 7 | official vs Falcon-H1-Tiny, Opus, two readings | 35 / 108 / 57; score 0.3175 (printed 0.318) [0.257, 0.385]; decisive 0.245 [0.182, 0.321]; p < 0.0001; orders 90.5%; first +0.005; categories chat 10/24/16 (p 0.024), code 10/13/7 (0.68), instruct 7/21/12 (0.013), maths 4/18/8 (0.0043), writing 4/32/14; two-turn 7/30/15; length shorter 93: 26/41/26 (0.419, p 0.086), similar 72: 7/40/25, longer 35: 2/27/6 (0.143); 189: 32 / 101 / 56, 0.317 [0.255, 0.387] | FR, PV `final-prefopt-vs-falcon` `judge-opus`; equal to `R/final-prefopt-vs-falcon--judge-opus.json` (all fields) |
| C526 | abs, 1, 7.5, 9, Table 7, M1, Fig 7 | official vs MobileLLM-R1.5, Opus | 116 / 15 / 69; score 0.7525 (printed 0.753; the figure prints 0.752) [0.688, 0.807]; decisive 0.885 [0.820, 0.929]; p < 0.0001; orders 96.0%; first −0.005; maths 7/12/11 (p 0.36); 189: 109 / 14 / 66, 0.751 [0.685, 0.808] | FR, PV `final-prefopt-vs-mobilellm`; equal to the report |
| C527 | abs, 7.5, Table 7, M1 | Falcon-H1-Tiny's control arms, Opus | K-matched (`2dcb1c92a6`): 35 / 112 / 53, 0.3075 (0.308) [0.248, 0.375], decisive 0.238 [0.176, 0.313], orders 93.5%, first +0.020; 189: 34 / 106 / 49, 0.310; selector-matched (`d1f2a63f82`): 29 / 116 / 55, 0.2825 (0.283) [0.225, 0.349], decisive 0.200 [0.143, 0.272], orders 92.5%, first +0.018 (0.0175); 189: 27 / 110 / 52, 0.280 | FR, PV `final-prefopt-vs-falcon-k6lp`, `-k6ds`; equal to their reports |
| C528 | 7.5 | paired comparison of our per-prompt score, released Falcon against each control arm | K-matched 29 / 25 / 146, mean +0.010, p 0.683; selector-matched 34 / 21 / 145, +0.035, p 0.105; on 189: 26 / 23 (p 0.78), 32 / 19 (p 0.092) | FR `paired` (`bes.eval.proxy.score.paired_sign_test`, as the CLI's `compare`) |
| C529 | 7.5, 7.7, Table 7, M5 | official vs trained weights, Opus | 59 / 34 / 107; 0.5625 (0.563) [0.493, 0.629]; decisive 0.634 [0.533, 0.725]; p 0.0124; orders 91.5%; first +0.010; chat 16/8/26, code 10/1/19 (p 0.012), instruct 14/8/18, maths 5/5/20, writing 14/12/24; 189: 55 / 33 / 101, 0.558, p 0.025; 4 prompts identical in every turn (instruct-014, instruct-025, writing-029, writing-033), 2 in one turn | FR, PV; `N/study/proxy/pairs/final-prefopt-vs-trained/pairs.jsonl` (A = B count, inline check) |
| C530 | Table 7, M1, M11 | the reference set on the 189 | 137 / 5 / 47, 0.849 [0.791, 0.893], decisive 0.965 | FR, PV `ref-falcon-vs-mobilellm-test` |
| C531 | 7.2, 7.5, Table 7, M1–M3, M11 | the registered judge, partial | Falcon 155: 10 / 43 / 102, 0.394 [0.320, 0.472], decisive 0.189 [0.106, 0.314], p < 0.0001, orders 60.7%, first −0.139, mass 0.985, clipped 1; 189-subset 147: 10 / 41 / 96, 0.395; MobileLLM 92: 22 / 3 / 67, 0.603 [0.501, 0.697], decisive 0.880 [0.700, 0.958], p 0.0002, orders 72.8%, first +0.011, mass 0.992, clipped 38; 87: 20 / 3 / 64, 0.598; tie rates 65.8% and 72.8% | FR, PV (files as read at 21:49); the re-scored reports of 21:58 |
| C532 | 7.5 | agreement of the Opus readers and the registered judge, pairs both rated | Falcon 155: both a side 44 (same 40, opposite 4), Opus side / judge tie 66, judge side / Opus tie 9, both tie 36, κ 0.217; MobileLLM 92: 22 same, 0 opposite, 33, 3, 34, κ 0.333 | FR `agreement` |
| C533 | 7.5 | the restriction to the 189 moves every score by at most 0.006 | largest 0.0056 (registered judge vs MobileLLM-R1.5: 0.6033 → 0.5977) | FR, PV (`all` against `189`) |
| C534 | 7.3 | resolution at the test tie rates | ties 28.5%, 34.5%, 53.5%; MDE 0.116, 0.121, 0.143 (n 200), 0.119, 0.125, 0.147 (n 189) | NUM5 `power.*_test_opus` |
| C535 | 7.8 | raw "both bad" per reading | prefopt vs trained 89 and 89 (A/B/tie: 55/45/11 and 50/56/5); vs Falcon 44 (71/81/4) and 41 (84/72/3); vs MobileLLM 66 (65/69/0) and 64 (69/67/0); reference 38 (78/79/5) and 38 (81/75/6); k6lp 42 and 45; k6ds 43 and 39; each reading's file counts equal the order1 / order2 records of `judge-opus.jsonl` | FR `readings`, `readings_match_ratings` (from `N/final_eval/<group>/<set>/r1|r2/chunk_*.verdicts.json`) |
| C536 | 7.6, Table 8, M4, M11, D.5 | exact scores of every run on test (200 / 189) | maths of 30 (of 28): ours 6 (6), K = 1 5 (5), trained 6 (6), Falcon 15 (14), K6 lp 17 (16), K6 selector 18 (17), MobileLLM rec 12 (11), neutral 11 (10); all constraints of 48 (of 44): 38 (34), 37 (33), 41 (37), 33 (29), 35 (32), 42 (38), 8 (8), 9 (8); single of 56: 46, 44, 48, 41, 42, 50, 10, 11; degenerate of 252: 5, 17, 3, 2, 1, 0, 158, 148 (truncated 5, 17, 3, 1, 0, 0, 150, 146; loops 14 and 17, repeated lines 54 and 50 for MobileLLM; empty 2, 1; fences 1, 1, 7, 10); stock refusals and deflections 0 for all; median words 101, 105, 99, 155, 147, 145, 395, 517 (means 130.4, 149.7, 119.2, 173.4, 174.3, 168.4, 474.1, 533.7) | EXC; every field equal to `N/study/proxy/reports/exact/<run>-test.json` where it exists (all but the neutral arm) |
| C537 | 7.6, abs, 1, 9 | what the selector-matched arm and the K = 1 arm show about our constraint score | Falcon K6 selector 42 > ours 38 > ours K = 1 37 > Falcon released 33 (of 48): our lead over the released preset is our decoder's (mostly its potentials), not our model's | C536 |
| C538 | E.6, Table E3, 7.9 | the official model on the first 60 dev prompts (its select-stage replies) | maths 0 of 9; constraints 13 of 14; cut 1 of 75; median words 98; on 200 dev: 4 of 30, 41 of 47, degenerate 3 of 248 (k1: 3, 38, 16; noband 4, 43, 3), equal to the run's exact reports | `L5/exact_dev_final.json` (`exact_dev_final.py` on `N/records/unpacked/proxy/outputs/ours-step51812-c04f3f6b3fca71f1/`) |
| C539 | 7.9, Table 9 | decisive win rates along the trajectory | Falcon: 2/45 = 0.044, 3/42 = 0.071, 35/143 = 0.245; MobileLLM: 15/23 = 0.652, 20/27 = 0.741, 116/131 = 0.885 | arithmetic on C421, C525, C526 |
| C540 | I | the last day's incidents | disk full 12:10 (C329); worker restarts 19:08–19:30 and 19:32–20:13 (22 and 41 min), local loops down, run unaffected; registered judge: TPU host until 20:40 (113 Falcon pairs, mean 9.8 s per pair), CPU runtime from 20:41 (63.8 s with 4 threads beside a second stream, pairs 114–136; 137.6 s with 8 threads, 137–141), restarted 21:22 as two 4-thread streams (49.8 s, 142–155); fourth CPU runtime refused; TPU given back 20:41:04 | `FACTS_2.md` (20:32, 21:00, 21:27); `N/OPERATIONS.md` ("AFTER THE TPU"); seconds per pair recomputed from the `seconds` field of `ratings/final-prefopt-vs-falcon/judge-qwen3-4b-2507-3605803b.jsonl` (inline check) |
| C541 | J | commands of the test replies, controls, pair sets and readings | as in Appendix J | `S/scripts/endgame_driver.py`, `final_pairs.sh`, `final_score.sh`, `blind_eval.py`; the runs' `meta.json` (settings hashes 9a89782942, f93654208f, 2dcb1c92a6, d1f2a63f82); `N/final_eval/JUDGE_INSTRUCTIONS.md` |
| C542 | J | tests | 1,367 `def test_` at `34f12ac` | `grep -r "def test_" tests | wc -l` in `R/` |
| C543 | K | the release without a separate licence review; README's research-only note | as stated | `R/release/main16-v9/README.md` (last section); `MODEL_CARD.md` licence table |
| C544 | Fig C2 | the figure's note and the script change | "Records to step 51,800 (18:33:30), the last step record. Training stopped at its deadline, 18:33:49; the TPU was given back at 20:41:04." | `figures/fig2_throughput.py` (fifth-pass branch); `L5/orig_fig2_throughput.py`; the PNG |
| C545 | Fig 5 | what Figure 5 draws at the final cut-off | runtime checkpoints to 51,588 (1,043.0 min), 51,812 left out (after the cut-off); gap −0.0771 ± 0.2246 | `L5/make_all_final.log` (fig8 lines, "held-out evaluations of checkpoints after the cut-off left out: steps [51812]") |
| C546 | 4.2 note | the main text's length | 9,818 words of letter or digit tokens from the abstract to the conclusion, 1,005 of them in table rows (the third draft, counted the same way, 9,287) | `L5/scripts/wordcount.py paper.md` |

| C547 | abs, 7.2, 7.5, 7.10, Table 7, M1–M3, M11 | our K = 1 arm (the official export with one particle, run `final-k1-f93654208f-s0`), Opus, two readings | vs Falcon-H1-Tiny 22 / 122 / 56, 0.250 [0.195, 0.314], decisive 0.153 [0.103, 0.220], orders 92.5%, first +0.015; 189: 20 / 116 / 53, 0.246; vs MobileLLM-R1.5 99 / 16 / 85, 0.7075 (0.708) [0.641, 0.766], decisive 0.861 [0.786, 0.913], orders 91.5%, first +0.0075 (+0.008); 189: 92 / 14 / 83, 0.706; raw both bad 45 and 41, 74 and 74 | FR, PV `final-k1-vs-falcon`, `final-k1-vs-mobilellm`; equal to `R/final-k1-vs-*--judge-opus.json`; readers' files in `N/final_eval/k1/` |
| C548 | 7.5, M13 | what the six particles are worth (paired, same prompts) | official K = 6 vs K = 1 against Falcon-H1-Tiny: 31 / 12 / 157, mean +0.0675, p 0.0054 (189: 30 / 11, p 0.0043); against MobileLLM-R1.5: 28 / 13 / 159, +0.045, p 0.0275 (189: 26 / 11, p 0.020) | FR `paired` |
| C549 | 7.6, Table 8, M4 | K = 1 exact scores on test | maths 5 of 30 (5 of 28), all constraints 37 of 48 (33 of 44), single 44 of 56, degenerate 17 of 252, all cut at the limit; median words 105 (mean 149.7) | EXC `ours_k1`; equal to its exact report |
| C550 | status, 7.2, M | what stays [TBD] | the registered judge's full pass (155 and 92 of 200 rated when read); MobileLLM-R1.5's K = 6 control arms (in generation); the author list | `N/study/proxy/ratings/`; `FACTS_2.md` (21:27, 21:40) |

### Withdrawn or changed in the final pass

| id | was | now |
|---|---|---|
| C202, C444, C455–C457 | R1–R12 [TBD]; the shells' files | filled (Appendix M) from C500–C550; the second registered judge's rows dropped (not run on test); M5's `final-trained-vs-*` rows replaced by the direct set (C529); M13 added (C528, C548); the full pass of the registered judge and MobileLLM-R1.5's control arms stay [TBD] |
| C405, C406, C413, C414 | to the cut-off step 46,350 | C512, C514, the final-pass section of `FIGURES.md` (cut-off 51,800) |
| C408 | projections of the end (1.358e10, 51,821 steps) | C500 (ledger) and C515 (projection against the ledger); the deficit against `final` stays a projection |
| C410 | TPU time at 16:27:24 (70,835 s, 78%) | C501 (86,054 s, 73.1%) |
| C415–C419 | held-out to 46,178 | C517–C523 (to 51,812); the 46,178 and 35,294 comparisons moved to Table G3 |
| C421–C428 | dev readers in the main text (Table 7) | unchanged values, moved to Appendix E.6 (Tables E2, E3) |
| C422 | the abstract's development claims (39 of 42, 16 of 17, 20 of 27, 7 of 9) | withdrawn from the abstract, replaced by the test results (C525–C531); kept in E.6 |
| C434 | the rate during the micro runs to the cut-off | C524 |
| C437 | 1,367 tests at `190e0b2` | C542 (also at `34f12ac`) |
| C452 | K = 1 and controls "not run" | Falcon's control arms run and read (C527–C528); our K = 1 arm generated, scored and read (C536, C547–C549); MobileLLM-R1.5's arms [TBD] |
| FACTS_2 (21:00) | "the constraint score in which our model leads … is the selector's doing" | withdrawn: with one particle (no selection) our model meets 37 of 48 (C537) |
| FACTS_2 (20:32) | "it is why the total rose after 46,178 (2.0914 → 2.1422)" | the all-row loss fell to 2.0840 at 47,260 and rose from the tail on (C521) |
| FACTS_2 (21:08, 21:12, 21:27) | scores 0.752, 0.562, 0.307, 0.282 and k6ds intervals [0.225, 0.348], [0.143, 0.273] | exact values 0.7525, 0.5625, 0.3075, 0.2825; printed once-rounded 0.753, 0.563, 0.308, 0.283 and [0.225, 0.349], [0.143, 0.272] (PV) |
