
## Third pass: new and changed claims (C400–C445)

Written 24 November 2026, 17:30–18:15 UTC, for the third draft. `L3/` = `N/paper/lead3_work/`.
`NUM3` = `L3/numbers.json` (written by `L3/make_numbers.py`, a copy of the second pass's script reading
`L3/v9_snap_1628/`, the train log, ledger, run log and manifest extracted from the snapshot `N/snap_main`
created 16:28:12 UTC; newest step record 46,350). `VLOG3` = `L3/v9_snap_1628/run/train/train_log.jsonl`.
`RD` = `L3/readers.json` (`L3/readers.py`: the repository's own key check and tally over
`N/study/proxy/ratings/`). `MG` = `L3/micrograft.json` (`L3/micrograft.py` on
`N/micro_graft/6M/results_all.jsonl`). `H3` = `L3/heldout3.json`; `H46` = `L3/heldout46.json`. `MAL` =
`N/paper/figures/_work/make_all.log` of the rebuild `make_all.sh --cutoff 46350` (16:57:45 UTC).
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
| C416 | 6.5, Table 5 | comparisons at 46,178 (bootstrap over rows, seed 20261124, 10,000 resamples) | since the graft −0.1679 [−0.1828, −0.1533]; web −0.1019 [−0.1106, −0.0937]; chat −0.2393 [−0.2560, −0.2235]; pages off − on +0.2166 [0.2019, 0.2320]; pages off − `final` 29,722 +0.0487 [0.0394, 0.0582] (web +0.0623, chat +0.0340); 46,178 − 35,294 −0.1073 | H46 `paired`; recompute: `heldout_eval.py --manifest L3/v9_snap_1628/run/manifest.json --ckpt N/snap_main --variant on: --variant pagesoff:page=0,gain=0` (the 16:28 snapshot, ckpt 46,178, now in `N/snap_main.prev`), then `heldout46.py` |
| C417 | 6.5, Table 5, Fig 5 | `final`'s three-point line at the newest checkpoints | minutes: 2.1470, gap −0.088 ± 0.171 at 917.9 min; log tokens at equal time: gap −0.112 ± 0.111; at equal tokens: −0.127 ± 0.084; at 35,294 unchanged (−0.033 ± 0.066, −0.038 ± 0.052, −0.044 ± 0.040) | H3 `extrapolation_final_three_points`; MAL (fig8 lines) |
| C418 | 6.5 (iv), G.5 | input-frequency quarters at 46,178 | pages: 0.299, 0.241, 0.170, 0.159 nats (rarest first); since the graft: −0.152, −0.153, −0.167, −0.199 | H46 `frequency` |
| C419 | 6.5 | local evaluations reproduce the runtime's (changed: was "to 10⁻⁸") | within 10⁻⁷: at 46,178 all-row loss differs by 2.2 × 10⁻⁹, web by 3.0 × 10⁻⁸ | `L3/heldout/v9_46178_on.npz` against `N/heldout_v9.jsonl` (inline check, logged in `RESPONSE_2.md`) |
| C420 | Table 8, 7.4 | bits per byte of ours on the web rows at 46,178 | 0.9801 over 258,817 bytes | H46 `bpb_web` |
| C421 | Table 7, abstract, 1, 7.4, 9 | every reader on the four development pair sets | as in Table 7 (e.g. `v9` vs Falcon-H1-Tiny: Opus 3 / 39 / 18; `qwen3-4b-2507` 1 / 16 / 43, 0.375 [0.264, 0.501], p 0.0003, orders 60.0%, first −0.114; `gemma4-e4b` 0 / 3 / 57, 0.475 [0.354, 0.599], p 0.25, 83.3%, −0.065) | RD `sources`; equal to every `N/study/proxy/reports/ab/*.json` of these sets; recompute `readers.py` |
| C422 | abstract | Falcon-H1-Tiny preferred in 39 of 42 (Opus) and 16 of 17 (registered judge) decisive pairs; ours won 20 of 27 and 7 of 9 against MobileLLM-R1.5 | as stated (`v9` 31,912) | RD |
| C423 | 7.4 | pair-by-pair agreement, `v9` sets | Opus (two readings) and `qwen3-4b-2507`: Falcon-H1-Tiny both a side 16 (same 14, opposite 2), Opus side / judge tie 26, judge side / Opus tie 1, both tie 17; MobileLLM-R1.5 6 (6 same), 21, 3, 30; κ (win, loss, tie) 0.209 and 0.183; `gemma4-e4b`'s 3 and 3 sides all agree with Opus | RD `agreement` |
| C424 | 7.4 | the second judge gave no side in 93–97% of pairs; both automatic judges clipped 30 of the 60 MobileLLM-R1.5 pairs | ties 56, 58, 57, 57 of 60; `clipped` 30 (s1 and s2, both judges) | RD |
| C425 | 7.3 | tie rates of the registered judge and what n = 200 / 189 resolves at them | 43 and 51 of 60 (71.7%, 85.0%); 0.5 + 0.182 and 0.246 (n = 200), 0.187 and 0.252 (n = 189) | RD; NUM3 `power.*_qwen_judge` |
| C426 | Table 7, 7.4, abstract, 1, 9 | the reference: Falcon-H1-Tiny against MobileLLM-R1.5 on the 200 test prompts, Opus two readings | 144 / 6 / 50; 0.845 [0.788, 0.889]; decisive 0.960 [0.915, 0.982]; orders 93.5%; first +0.0125; chat 37 / 1 / 12, code 17 / 2 / 11, instruct 31 / 0 / 9, maths 15 / 3 / 12, writing 44 / 0 / 6; two-turn 39 / 1 / 12 | RD `ref-falcon-vs-mobilellm-test`; `N/study/proxy/reports/ab/ref-falcon-vs-mobilellm-test--judge-opus.json` |
| C427 | 7.4 | the readers' notes on the reference pairs describe MobileLLM-R1.5's reply as looping or repeating / a bare boxed answer | 87 and 83 / 46 and 49 of 200 (first and second reading; keyword count of clauses naming its reply) | `L3/ref_notes.json` (`ref_notes.py` over `N/final_eval/ref/ref-falcon-vs-mobilellm-test/r1|r2/chunk_*.notes.json`; a heuristic, stated as a keyword count) |
| C428 | 7.4 | in absolute terms | s2 (`v9` 31,912) first reading: both bad in 44 of 60 pairs of 3-gram against default (notes: wrong facts or arithmetic 20, incoherent 14); 31 of 60 against MobileLLM-R1.5; 14 of 60 against Falcon-H1-Tiny; s1 variants 46–55 of 60 | `L3/ours_notes.json`; RD (`opus-1` `raw_verdicts`) |
| C429 | 6.6, G.4, Table 6 | micro design | d 128; 2 blocks × 8 loops; rows of 512; 4,096 tokens per step; 1,464 steps; 5,996,544 tokens trained; `ref` 4,402,372 numbers, `v9_pages` 14,904,516, new leaves 10,502,144 (5 leaves), 59 shared leaves (4,402,372), 175 arrays copied; grafts at steps 366, 732, 1,098; evaluations at 1,499,136, 2,998,272, 4,497,408, 5,996,544 tokens; token neurons 8 in 32 dimensions; mixture 0.75 / 0.25 rows; token shares 0.780 / 0.220; pools 8,192 / 4,096 rows; held-out 32 + 32 rows | MG `meta`, `checks`; record fields `config`, `bench`, `data`, `graft.leaves` |
| C430 | G.4 | passes over the pools | 1.07 (web) and 0.71 (chat): 0.75 × 11,712 = 8,784 of 8,192 rows and 2,928 of 4,096 (11,712 = 1,464 steps × 8 rows) | arithmetic on MG `meta`; `R/bes/eval/microbench.py` (`BatchMaker`, `passes` = 2 and 1 started) |
| C431 | Table 6, 6.6, 6.7, abstract, 1, 9, Fig 6 | the micro results | as in Table 6 and G.4 (e.g. `v9_pages` − `ref` −0.0519 ± 0.0107, −1.24%; shares 1.06 / 0.69 / 0.24, per seed 0.86, 1.33, 1.08 / 0.62, 0.87, 0.64 / 0.20, 0.29, 0.24; graft − `ref` per seed −0.0539, −0.0551, −0.0558 / −0.0386, −0.0359, −0.0327 / −0.0127, −0.0120, −0.0122) | MG `table`, `diffs`, `share`; equal to `N/micro_graft/6M/graft_report.txt` in every printed digit (compared line by line; the figures agent's check finds its 1,013 numbers equal) |
| C432 | 6.6, G.4 | the checks inside the micro runs | every graft arm's 1, 2 or 3 evaluations before its graft equal `ref`'s of the same seed row for row (18 of 18); the graft check's difference 0.0 at 9 of 9 grafts; held-out before/after difference 0 per domain and per row; check losses 5.0309, 5.0072, 5.0118 / 4.7603, 4.7661, 4.7617 / 4.6126, 4.6162, 4.6532 | MG `checks` |
| C433 | 6.6 | learning-rate factor at the graft | 0.775, 0.55, 0.325 | MG `checks.*.carried.lr_factor` |
| C434 | 6.6, G.4 | the live run's rate during the micro runs | 189,054 tokens/s (n = 73) before 14:12:51; 189,086 (n = 117) from 14:12:51 to the cut-off | `L3/rate_during_micro.json` (`rate_during_micro.py`) |
| C435 | 8 | `ref`'s spread over three seeds | sd 0.0076 nats = 0.18% of 4.1912 | MG `table` |
| C436 | 6.6, 6.7 | the micro arms that never shared a state differ by about 0.01 between seeds | sd of `v9_pages` − `ref` 0.0107; of `graft_*` − `v9_pages` 0.0096–0.0113 | MG `diffs` |
| C437 | J | test functions at `b1e9140` (changed: 1,356 at `8a86bcb`) | 1,367 | `grep -r "def test_" tests | wc -l` in `R/` |
| C438 | J | the controlled experiment's commits | `b04478e` (graft arms), `b1e9140` (results) | `git log` in `R/` |
| C439 | Figure 5 caption | arrow label at the newest checkpoint | −0.088 ± 0.171; the label is cut at the right edge of `F8_heldout.png` in this build | MAL (fig8); the figure file |
| C440 | Figure 6 caption | values in the micro figure | as C431; drawn by `fig12_micrograft.py` | `data/micro_graft_summary.csv` |
| C441 | C, Table C2 | step 36,867 is at 9.66 × 10⁹ tokens | 36,867 × 262,144 = 9,664,462,848 | arithmetic |
| C442 | 6.6 Table 6 restart row, 6.7, G.4 | restart arms (`v9_pages` from scratch on 3 × 10⁶ tokens, seeds 0–2) | RESTART_CLAIM | `N/micro_graft/3M/results.tar.gz` unpacked into `L3/micro3M/`; `micrograft.py L3/micro3M` |
| C443 | 5.1 | whole steps are single timings (changed: "we report medians" applied to all) | mean of `--step-calls` back-to-back steps after a warm call, default 2, count used not recorded | `R/scripts/tpu_kernels.py`; `FIGURES.md` ("How a whole step was timed") |
| C444 | Appendix M | the files the shells name | as in Appendix M | `N/OPERATIONS.md` ("Endgame (v9)", "Final blind evaluation: the mechanics", "Since 12:10", "State at 16:50"); `S/scripts/final_pairs.sh`, `final_score.sh`, `endgame_testgen.sh`, `cmp_exports.py`, `fetch_records.sh`; `R/bes/export.py` (`export.json`) |
| C445 | 6.6 | the micro code was reviewed adversarially before the runs | trunk equality, function preservation, carried optimiser state, schedule and data order, unchanged E2 hashes, report arithmetic | `N/FACTS_2.md` (16:50); `W/micro_graft.md`; `W/micro_graft_work/review/` |

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
