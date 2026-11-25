## Appendix M. Results on the test split and at the end of the run

The shells of the third draft, filled. Every value is recomputed from the records named for its table by
the scripts of `lead5_work/scripts/` (this appendix is written by `make_appendix_m.py` from their outputs; no
number is typed). Conventions as in Table 7: W / L / T from our side (T includes "both bad" and pairs whose
two orders disagree), score (wins + ties/2)/n with its 95% Wilson interval, the decisive win rate wins /
(wins + losses) with its interval, exact two-sided sign test; values rounded once from full precision, a value
exactly on a half to even. R = `N/study/proxy/reports/ab/`. Our official run is
`ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0` (TAG = `prefopt`: its export has the
run's own `weights16.npz` and decoding choice), the opponents' runs `falcon-e60034381f4a/recommended-5ccec0259b-s0`
and `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`. The registered judge's pass is complete on both official sets: its rows give the pairs rated in its files as read at 22:38 UTC (final-prefopt-vs-falcon: 200 records, last written 22:11:18, final-prefopt-vs-mobilellm: 200 records, last written 22:35:20). The second registered judge
(`gemma4-e4b`) was not run on the test pairs, so its rows of the shells are gone (Section 7.2, deviation 1).

**Table M1 (R1). The test split, 200 prompts: each opponent and each reader** (`R/<set>--<source>.json`, fields `overall` and `judge`, reproduced by `final_results.py`; sets `final-prefopt-vs-falcon`, `-vs-falcon-k6lp`, `-vs-falcon-k6ds`, `-vs-mobilellm`, `ref-falcon-vs-mobilellm-test`).

| opponent | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders agree | first-position preference |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | `qwen3-4b-2507` (registered), full pass | 200 | 12 / 61 / 127 | 0.378 [0.313, 0.446] | 0.164 [0.097, 0.266] | < 0.0001 | 60.5% | −0.134 |
| Falcon-H1-Tiny-90M-Instruct | Opus, two readings | 200 | 35 / 108 / 57 | 0.318 [0.257, 0.385] | 0.245 [0.182, 0.321] | < 0.0001 | 90.5% | +0.005 |
| Falcon-H1-Tiny-90M-Instruct | Opus, two readings, neutral titles | 200 | 33 / 109 / 58 | 0.310 [0.250, 0.377] | 0.232 [0.171, 0.308] | < 0.0001 | 92.5% | +0.010 |
| Falcon-H1-Tiny, K = 6 kept by its own log-probability | Opus, two readings | 200 | 35 / 112 / 53 | 0.308 [0.248, 0.375] | 0.238 [0.176, 0.313] | < 0.0001 | 93.5% | +0.020 |
| Falcon-H1-Tiny, K = 6 kept by our selector | Opus, two readings | 200 | 29 / 116 / 55 | 0.282 [0.225, 0.349] | 0.200 [0.143, 0.272] | < 0.0001 | 92.5% | +0.018 |
| MobileLLM-R1.5-140M | `qwen3-4b-2507` (registered), full pass | 200 | 59 / 5 / 136 | 0.635 [0.566, 0.699] | 0.922 [0.830, 0.966] | < 0.0001 | 71.5% | −0.035 |
| MobileLLM-R1.5-140M | Opus, two readings | 200 | 116 / 15 / 69 | 0.752 [0.688, 0.807] | 0.885 [0.820, 0.929] | < 0.0001 | 96.0% | −0.005 |
| MobileLLM-R1.5-140M | Opus, two readings, neutral titles | 200 | 112 / 15 / 73 | 0.742 [0.678, 0.798] | 0.882 [0.814, 0.927] | < 0.0001 | 96.0% | −0.010 |
| MobileLLM-R1.5, K = 6 (both rules) | – | [TBD: replies in generation since 19:17 UTC, 40 of 200 prompts by 21:13; expected after midnight] | | | | | | |
| Falcon-H1-Tiny, against our K = 1 arm | Opus, two readings | 200 | 22 / 122 / 56 | 0.250 [0.195, 0.314] | 0.153 [0.103, 0.220] | < 0.0001 | 92.5% | +0.015 |
| MobileLLM-R1.5, against our K = 1 arm | Opus, two readings | 200 | 99 / 16 / 85 | 0.708 [0.641, 0.766] | 0.861 [0.786, 0.913] | < 0.0001 | 91.5% | +0.008 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 200 | 144 / 6 / 50 | 0.845 [0.788, 0.889] | 0.960 [0.915, 0.982] | < 0.0001 | 93.5% | +0.012 |

**Table M2 (R2). By category and by turns** (W / L / T; the same files, fields `by_category` and `by_turns`; prompts of the `test` split).

| slice (prompts) | Falcon-H1-Tiny: Opus | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon, K = 6 own log-prob.: Opus | Falcon, K = 6 our selector: Opus | MobileLLM-R1.5: Opus | MobileLLM-R1.5: `qwen3-4b-2507` | ours K = 1 vs Falcon: Opus | ours K = 1 vs MobileLLM: Opus |
|---|---|---|---|---|---|---|---|---|
| chat (50) | 10 / 24 / 16 | 3 / 10 / 37 | 8 / 24 / 18 | 7 / 25 / 18 | 28 / 1 / 21 | 12 / 0 / 38 | 9 / 31 / 10 | 28 / 2 / 20 |
| code (30) | 10 / 13 / 7 | 3 / 4 / 23 | 11 / 13 / 6 | 11 / 12 / 7 | 16 / 2 / 12 | 13 / 0 / 17 | 7 / 13 / 10 | 13 / 4 / 13 |
| instruction-following (40) | 7 / 21 / 12 | 4 / 17 / 19 | 9 / 21 / 10 | 7 / 22 / 11 | 29 / 0 / 11 | 17 / 0 / 23 | 2 / 25 / 13 | 24 / 0 / 16 |
| maths (30) | 4 / 18 / 8 | 0 / 12 / 18 | 4 / 19 / 7 | 3 / 20 / 7 | 7 / 12 / 11 | 5 / 4 / 21 | 3 / 19 / 8 | 6 / 10 / 14 |
| writing (50) | 4 / 32 / 14 | 2 / 18 / 30 | 3 / 35 / 12 | 1 / 37 / 12 | 36 / 0 / 14 | 12 / 1 / 37 | 1 / 34 / 15 | 28 / 0 / 22 |
| single-turn (148) | 28 / 78 / 42 | 10 / 45 / 93 | 26 / 80 / 42 | 21 / 84 / 43 | 83 / 11 / 54 | 45 / 5 / 98 | 16 / 85 / 47 | 67 / 13 / 68 |
| two-turn (52) | 7 / 30 / 15 | 2 / 16 / 34 | 9 / 32 / 11 | 8 / 32 / 12 | 33 / 4 / 15 | 14 / 0 / 38 | 6 / 37 / 9 | 32 / 3 / 17 |

**Table M3 (R3). Length buckets and reader diagnostics** (the same files, fields `by_length` (ratio of our words to the opponent's, +1 each: shorter below 2/3, longer above 3/2) and `judge`; the Opus records carry no probability mass and no clipping, since the readers read the chunk files).

| row | Falcon-H1-Tiny: Opus | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon, K = 6 own log-prob.: Opus | Falcon, K = 6 our selector: Opus | MobileLLM-R1.5: Opus | MobileLLM-R1.5: `qwen3-4b-2507` | ours K = 1 vs Falcon: Opus | ours K = 1 vs MobileLLM: Opus |
|---|---|---|---|---|---|---|---|---|
| ours shorter: n; W / L / T | 93; 26 / 41 / 26 | 93; 11 / 14 / 68 | 94; 26 / 38 / 30 | 89; 19 / 37 / 33 | 140; 77 / 9 / 54 | 140; 49 / 1 / 90 | 81; 15 / 45 / 21 | 140; 71 / 9 / 60 |
| similar: n; W / L / T | 72; 7 / 40 / 25 | 72; 1 / 25 / 46 | 76; 8 / 51 / 17 | 82; 10 / 55 / 17 | 28; 17 / 5 / 6 | 28; 4 / 3 / 21 | 79; 5 / 46 / 28 | 23; 6 / 5 / 12 |
| ours longer: n; W / L / T | 35; 2 / 27 / 6 | 35; 0 / 22 / 13 | 30; 1 / 23 / 6 | 29; 0 / 24 / 5 | 32; 22 / 1 / 9 | 32; 6 / 1 / 25 | 40; 2 / 31 / 7 | 37; 22 / 2 / 13 |
| orders agree | 90.5% | 60.5% | 93.5% | 92.5% | 96.0% | 71.5% | 92.5% | 91.5% |
| first-position preference | +0.005 | −0.134 | +0.020 | +0.018 | −0.005 | −0.035 | +0.015 | +0.008 |
| probability mass on A, B, C (field `judge.format_mass_mean`) | – | 0.983 | – | – | – | 0.995 | – | – |
| pairs with a reply cut for the judge (over 4,000 characters) | – | 1 | – | – | – | 88 | – | – |

**Table M4 (R4). Exact scores on the 200 test prompts (252 turns), in full** (`bes.eval.exact_score` on every reply, by `lead4_work/scripts/exact_split.py --split test` with every run below, `lead5_work/exact_test.json`; equal field by field to the run's report in `study/proxy/reports/exact/` where one exists, which is every run but MobileLLM-R1.5's `neutral` arm; counts from `lead5_work/exact_counts.json`).

| measure | ours, official | ours, K = 1 | ours, trained weights | Falcon-H1-Tiny | Falcon K = 6, own log-prob. | Falcon K = 6, our selector | MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |
|---|---|---|---|---|---|---|---|---|
| maths answers correct (of 30) | 6 | 5 | 6 | 15 | 17 | 18 | 12 | 11 |
| constrained prompts with every constraint met (of 48) | 38 | 37 | 41 | 33 | 35 | 42 | 8 | 9 |
| single constraints met (of 56) | 46 | 44 | 48 | 41 | 42 | 50 | 10 | 11 |
| degenerate replies (of 252) | 5 | 17 | 3 | 2 | 1 | 0 | 158 | 148 |
| … cut at the token limit | 5 | 17 | 3 | 1 | 0 | 0 | 150 | 146 |
| … looping tail; repeated lines | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 14; 54 | 17; 50 |
| … empty; unclosed code fence; role markers | 0; 0; 0 | 0; 0; 0 | 0; 0; 0 | 0; 1; 0 | 0; 1; 0 | 0; 0; 0 | 2; 7; 0 | 1; 10; 0 |
| stock refusals; stock deflections (of 252) | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| words per reply: median; mean | 101; 130.4 | 105; 149.7 | 99; 119.2 | 155; 173.4 | 147; 174.3 | 145; 168.4 | 395; 474.1 | 517; 533.7 |
| mean repeated-4-gram rate | 0.034 | 0.054 | 0.038 | 0.020 | 0.026 | 0.021 | 0.258 | 0.290 |

A reply can be degenerate in several ways at once, so the sub-rows do not add up.

**Table M5 (R5). The trained weights against the weights after preference optimisation** (the direct pair set `final-prefopt-vs-trained`, both readings, `R/final-prefopt-vs-trained--judge-opus.json`; exact scores from Table M4; held-out losses from `lead5_work/heldout5.json`; the in-run checks from Table C5). The pair sets `final-trained-vs-falcon` and `-vs-mobilellm` were published at 19:07 UTC and not read: the direct set answers the question on the same prompts.

| measure | after preference optimisation (official) | trained weights | difference |
|---|---|---|---|
| blind, Opus, two readings, 200 prompts: W / L / T (official's side) | 59 / 34 / 107 | | score 0.562 [0.493, 0.629], p = 0.012 |
| … the 189 untouched prompts | 55 / 33 / 101 | | score 0.558 [0.487, 0.627], p = 0.025 |
| … by category: chat; code; instruction-following; maths; writing | 16 / 8 / 26; 10 / 1 / 19; 14 / 8 / 18; 5 / 5 / 20; 14 / 12 / 24 | | |
| maths answers correct (of 30) | 6 | 6 | +0 |
| every constraint met (of 48) | 38 | 41 | -3 |
| single constraints met (of 56) | 46 | 48 | -2 |
| degenerate replies (of 252) | 5 | 3 | +2 |
| words per reply, mean | 130.4 | 119.2 | |
| held-out loss, web + chat (64 + 64 rows) | 2.0562 | 2.0438 | +0.0124 [0.0112, 0.0136] |
| held-out loss, chat assistant tokens | 1.2095 | 1.1983 | +0.0112 [0.0097, 0.0129] |
| in-run: held-out preference accuracy (1,826 pairs); assistant-token loss of the stage's own rows | 0.694; 1.5208 | 0.500; 1.5043 | accuracy +0.194; loss +1.10% |

**Table M6 (R6). Held-out losses at the end of training and of the export** (nats per token). (a) The comparison set of Section 6.5 (64 web and 64 chat rows): the runtime's evaluation of the last checkpoint (`N/heldout_v9.jsonl`) and our CPU evaluations of both weights of step 51,812 from `N/snap_final` (`heldout_eval5.py`; bits per byte as `lead4_work/scripts/bpb_web.py` computes them, in `heldout5.py`). (b) The export's own 16-bit check on 16 rows per domain (`check` of `final/export.json`). The proxy rows are in neither part (Section 7.2, deviation 3).

| (a) 64 rows per domain | trained weights, step 51,812 (runtime) | after preference optimisation (local) | `final` 29,722 (the graft) |
|---|---|---|---|
| web | 2.6690 | 2.6804 | 2.7886 |
| chat | 1.3668 | 1.3803 | 1.6183 |
| chat, assistant tokens | 1.1983 | 1.2095 | 1.5164 |
| web + chat (comparison set) | 2.0438 | 2.0562 | 2.2267 |
| bits per byte, web rows (opponents: 1.570 and 1.500) | 0.974 | 0.978 | 1.017 |

| (b) export check, 16 rows per domain | float32 weights | 16-bit weights | difference |
|---|---|---|---|
| web (16,354 weighted tokens) | 2.69236 | 2.69243 | +0.000075 |
| chat (15,690 weighted tokens) | 1.31725 | 1.31721 | -0.000039 |
| all three domains, proxy included (43,138) | 2.34440 | 2.34443 | +0.000022 (tolerance 0.002) |

**Table M7 (R7). What the preference-optimisation stage did** (`run/stages.json`, `run/post/prefopt_report.json` and the run log in `N/snap_final`; Table C5 gives the candidates in full).

| quantity | value |
|---|---|
| minutes used of 60 | 29.7 (18:37:07–19:06:51 UTC) |
| preference pairs used | 35,201 for training and 1,826 held out by prompt: 37,007 UltraFeedback pairs and 20 of our own sampled pairs |
| learning rates tried and steps run at each | 1 × 10⁻⁶ and 3 × 10⁻⁶, 551 steps each (one epoch), 485 s and 424 s of training |
| held-out preference accuracy before / after (standard error) | 0.500 / 0.688 (0.011) and 0.694 (0.011) |
| held-out assistant loss change (limit +3%) | +0.86% and +1.10% |
| sampled length and repetition against their bands | length 0.99 and 0.95 × the start's (band 0.7–1.6), repeated 4-grams 0.86 and 0.87 × (at most 1.2 × + 0.01) |
| accepted (weights kept) or rejected | both accepted; kept 3e-06, the higher accuracy ("lr=3e-06 accepted with held-out preference accuracy 0.6944 (start 0.5000)") |

**Table M8 (R8). What the decoding selection chose** (the manifest's `decode` record; `run/stages.json`; the export's `decode_choice`).

| quantity | value |
|---|---|
| settings compared with the default | one particle (`k1`); no length band (`noband`); no library (none was built) |
| development prompts used; paired sign test of each against the default | the 200 `dev` prompts; mean exact score default 0.910, `k1` 0.850, `noband` 0.915; `k1` better on 1, worse on 18 (p = 7.6e-05); `noband` better on 2, worse on 0 (p = 0.50) |
| setting kept | `default` (6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library) |
| minutes used of 35 | 54.9, in a slice of 65.0 (the stage was given the time preference optimisation left) |

**Table M9 (R9). Training totals** (the trainer's ledger `run/train/ledger.json` and train log in `N/snap_final`; TPU seconds from `S/tpu_intervals.txt`, all three intervals closed; `lead5_work/numbers.json`).

| quantity | value |
|---|---|
| final step; tokens trained (`v9` part since step 29,722) | 51,812; 13,582,204,928 (5,790,760,960 in 22,090 steps) |
| training seconds of the kept trajectory; checkpoint seconds; set-up seconds | 62,893.1; 23.2; 279.6 |
| TPU seconds: runtime 1, runtime 2, runtime 3, total | 7,325; 8,220; 70,509; 86,054 of 86,400 |
| share of TPU time in training steps of the kept trajectory | 73.1% |
| live rate in the last stretch (p ≥ 0.92, conversations train on assistant tokens only), n | 189,035 tokens/s, n = 73 (steps 48,200–51,800) |

**Table M10 (R10). The 16-bit check of the export** (`final/export.json`: `check`, `weights`, `main_files`, `readset`; `SHA256SUMS`).

| quantity | value |
|---|---|
| largest absolute loss difference, 16-bit against float32, web and chat (the proxy rows' difference enters only the check's pass or fail) | 7.5e-05 nats (web); passed |
| leaves stored in bfloat16 / float16 (rule: the smaller relative L2 error; ties float16) | 0 / 64 (98,036,104 numbers; largest relative L2 error of a leaf 2.84e-04) |
| export size (bytes) and the SHA-256 of the weights | `weights16.npz` 196,088,804 bytes, `2da7cbd2e2f1fe49e805cb9bde62b15751f41ae92e71f00a3b0ba69f241afcf4`; the packed export 196,956,160 bytes in 12 parts |
| read-set audit of the export: numbers read per token, standard reading | ledger 19,631,240; measured on held-out sequences 19,427,809 (web, mean over 2,048 positions) and 19,427,956 (chat), at most 19,429,996; strict reading 24,352,024 at 1,024 positions and 27,929,250 at 2,048 (web) |

**Table M11 (R11). Tables M1 and M4 on the 189 test prompts no mid-run sample touched** (240 turns, 28 maths and 44 constrained prompts; the 11 seen ids of Section 7.2 left out; readers from each report's `per_prompt_score` as `lead4_work/scripts/restrict_report.py` does, in `final_results.py`; exact scores by `exact_split.py --exclude`, `lead5_work/exact_test189.json`).

| opponent | reader | n | W / L / T | score [95%] | sign test p |
|---|---|---|---|---|---|
| Falcon-H1-Tiny | `qwen3-4b-2507` | 189 | 11 / 57 / 121 | 0.378 [0.312, 0.449] | < 0.0001 |
| Falcon-H1-Tiny | Opus, two readings | 189 | 32 / 101 / 56 | 0.317 [0.255, 0.387] | < 0.0001 |
| Falcon-H1-Tiny | Opus, neutral titles | 189 | 31 / 102 / 56 | 0.312 [0.250, 0.381] | < 0.0001 |
| Falcon-H1-Tiny, K = 6 own log-probability | Opus, two readings | 189 | 34 / 106 / 49 | 0.310 [0.248, 0.379] | < 0.0001 |
| Falcon-H1-Tiny, K = 6 our selector | Opus, two readings | 189 | 27 / 110 / 52 | 0.280 [0.221, 0.348] | < 0.0001 |
| MobileLLM-R1.5 | `qwen3-4b-2507` | 189 | 56 / 5 / 128 | 0.635 [0.564, 0.700] | < 0.0001 |
| MobileLLM-R1.5 | Opus, two readings | 189 | 109 / 14 / 66 | 0.751 [0.685, 0.808] | < 0.0001 |
| MobileLLM-R1.5 | Opus, neutral titles | 189 | 106 / 14 / 69 | 0.743 [0.677, 0.800] | < 0.0001 |
| our trained weights | Opus, two readings | 189 | 55 / 33 / 101 | 0.558 [0.487, 0.627] | 0.025 |
| Falcon-H1-Tiny, against our K = 1 arm | Opus, two readings | 189 | 20 / 116 / 53 | 0.246 [0.190, 0.312] | < 0.0001 |
| MobileLLM-R1.5, against our K = 1 arm | Opus, two readings | 189 | 92 / 14 / 83 | 0.706 [0.638, 0.767] | < 0.0001 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 189 | 137 / 5 / 47 | 0.849 [0.791, 0.893] | < 0.0001 |
| exact scores: maths correct (of 28); constrained prompts met (of 44) | – | – | ours, official 6 and 34; ours, K = 1 5 and 33; ours, trained weights 6 and 37; Falcon-H1-Tiny 14 and 29; Falcon K = 6, own log-prob. 16 and 32; Falcon K = 6, our selector 17 and 38; MobileLLM-R1.5 `recommended` 11 and 8; MobileLLM-R1.5 `neutral` 10 and 8 | – | – |

**Table M12 (R12). Decoding throughput of the export on CPU** (the official test generation's summary `N/testgen/prefopt/out/testgen.json` and its log; the trained weights' generation for comparison, `N/testgen/trained/out/testgen.json`; on `dev`, 75 replies took 1,798 s with 16 workers, `FACTS_2.md`).

| quantity | official (preference-optimised) | trained weights |
|---|---|---|
| workers × cores per worker; host | 40 × 1; the TPU runtime's host (CPUs 2–21, 24–43 of 44) | 40 × 1; the same |
| replies; wall time, export and calibration included | 252; 2,109.8 s (19:31:22–20:06:32) | 252; 1,840.4 s (18:36:25–19:07:06) |
| seconds per reply (median); new tokens per reply (median) | 206.7; 155.5 | 159.4; 151.5 |
| seconds per decoding step, 6 particles, one core (median; mean) | 0.567; 0.595 | 0.550; 0.575 |

**Table M13. Paired comparisons of our per-prompt score on the same prompts** (`final_results.py`, `bes.eval.proxy.score.paired_sign_test`, as `python -m bes.eval.proxy compare --a A --b B --source judge-opus`; a prompt is better when our score against the opponent is higher in A than in B).

| A | B | prompts better in A; in B; same | mean difference | sign test p | on the 189 untouched prompts |
|---|---|---|---|---|---|
| official vs released Falcon | official vs Falcon K = 6, own log-probability | 29; 25; 146 | +0.0100 | 0.6835 | 26; 23; 140, p = 0.7754 |
| official vs released Falcon | official vs Falcon K = 6, our selector | 34; 21; 145 | +0.0350 | 0.1048 | 32; 19; 138, p = 0.0919 |
| official (K = 6) vs Falcon | ours K = 1 vs Falcon | 31; 12; 157 | +0.0675 | 0.0054 | 30; 11; 148, p = 0.0043 |
| official (K = 6) vs MobileLLM-R1.5 | ours K = 1 vs MobileLLM-R1.5 | 28; 13; 159 | +0.0450 | 0.0275 | 26; 11; 152, p = 0.0201 |
| official vs Falcon-H1-Tiny, first readings (titles named the set) | the same pairs, neutral titles | 6; 3; 191 | +0.0075 | 0.5078 | – |
| official vs MobileLLM-R1.5, first readings (titles named the set) | the same pairs, neutral titles | 7; 3; 190 | +0.0100 | 0.3438 | – |

