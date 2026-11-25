# Facts since the first draft was started (written by the orchestrator, 24 Nov 2026, 11:10 UTC)

Every number below has its source file. Use them in the next revision; do not round away the caveats.

## 1. v9 step on the chip: three switches (commit 769f240; docs/CHANGES.md 19.7)

Whole accumulation step, 64 pieces of 4 x 1024 tokens, table_grad=take, kernel onetile:save, relu2
(`scratchpad(old)/v9_measA.txt`, `v9_measD.txt`; old scratchpad = /tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad):

| setting | piece (ms) | step (ms) | tok/s (synthetic) |
|---|---|---|---|
| v9 as grafted | 22.79 | 1433.1 | 182,919 |
| + flat | 22.59 | 1406.4 | 186,393 |
| + flat + rows | 20.44 | 1295.2 | 202,403 |
| + flat + rows + tokn | 20.01 | 1287.1 | 203,665 |
| final (same chip, 07:45) | 14.64 | 884.5 (with ce: measured combination) | 296,371 |

Compile-only estimates were -46, -119 and -64 ms per step; measured -26.7, -111.2 and -8.1 ms. The
estimate for `tokn` was eight times too large. On the chip the loss difference to the `flat` entry is 0
and the gradient differences are 9.5e-3 (max) / 1.0e-2 (L2), the level a change of the compiled graph
gives (19.6). Live rate: 177.8k tok/s before, 196.7k tok/s after (10:59 UTC, step 31,950). Pause 320 s.

## 2. Held-out loss across the graft (scripts/midrun_eval.py on the runtime's CPU; same rows, float32,
dense attention; `scratchpad(new)/midrun_eval_1.json`; new scratchpad = /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad)

| run | checkpoint step | training tokens | train min | loss | web | chat | proxy | chat assistant |
|---|---|---|---|---|---|---|---|---|
| final | 26,890 | 7.049e9 | 494.7 | 2.2679 | 2.7952 | 1.6300 | 2.6332 | 1.5269 |
| final | 28,501 | 7.471e9 | 519.7 | 2.2624 | 2.7905 | 1.6217 | 2.6382 | 1.5191 |
| final | 29,722 (graft point) | 7.791e9 | 538.7 | 2.2598 | 2.7886 | 1.6183 | 2.6361 | 1.5164 |
| v9 | 30,742 | 8.059e9 | 563.7 | 2.2437 | 2.7739 | 1.6022 | 2.6112 | 1.4994 |
| v9 | 31,762 | 8.326e9 | 588.8 | 2.2370 | 2.7666 | 1.5950 | 2.6105 | 1.4918 |
| v9 | 31,912 | 8.366e9 | 592.5 | 2.2337 | 2.7662 | 1.5865 | 2.6187 | 1.4830 |

Rows: web 64, chat 64, proxy 16 (the export check's held-out rows). In the 53.8 training minutes after the
graft the loss fell by 0.0261; final's last 44 minutes before the graft fell by 0.0081 (0.0109 per 1e9
tokens, 0.00018 per minute). At final's trend the same 53.8 minutes give about -0.0099: v9 is about
0.016 nats ahead at equal training time, although it saw 0.575e9 tokens where final would have seen
about 0.91e9. Caveats to state: one run; the control is a linear extrapolation of three points, not a
continued run; the data stream changed at the graft (no library rows, another seed); 144 rows. The web
rows improved 4.4 times faster per token than before the graft, and the mixture change does not touch
them (library_frac_base was 0.02), so the gain is not a mixture artefact there. A loop on the runtime
(`/content/bes_cli/heldout_v9.jsonl`) evaluates every new checkpoint until training ends [TBD: the curve].

## 3. First blind comparison (mid-run, BEFORE the graft): final at step 25,280 (6.6e9 tokens), default
decoding (6 particles, T 0.6, top-p 0.9), the first 60 dev prompts (15 chat, 15 writing, 12 instruct,
9 code, 9 math; 15 are two-turn), judged blind by Opus 5.5 agents, ONE reading per pair (sides randomised
per pair by the harness; the key was not readable by the judges). Reports under
`scratchpad(new)/study/proxy/reports/ab/`.

| opponent | wins / losses / ties-or-both-bad | score (wins + ties/2)/n | 95% Wilson | sign test p |
|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct (recommended preset) | 2 / 43 / 15 | 0.158 | [0.087, 0.271] | < 0.0001 |
| MobileLLM-R1.5-140M (recommended preset) | 15 / 8 / 37 | 0.558 | [0.433, 0.677] | 0.21 |

Raw verdicts: Falcon set: 45 decisive, 14 both_bad, 1 tie; MobileLLM set: 23 decisive, 37 both_bad. By
category against MobileLLM: instruct 7/0/5, writing 4/0/11, chat 3/0/12, code 1/1/7, math 0/7/2.
Against Falcon: chat 0/10/5, code 2/6/1, instruct 0/9/3, math 0/6/3, writing 0/12/3; multi-turn 0/15/0.
The judges' reasons for our losses: wrong facts and arithmetic, incoherent text, loops.

Decode settings against the default on the same 60 prompts (ours against ours; W/L/T for the variant):
particles 1: 1/3/56; no_repeat_ngram 3: 6/3/51; temperature 0.4: 5/6/49; max_tokens 320: 1/1/58;
temperature 0.8: 2/7/51. Most pairs are "both bad" (46 to 55 of 60), so the study has
no power to separate the settings; the default was kept.

## 4. What must not be claimed
- That the model beats Falcon-H1-Tiny: at 0.42 of training it lost 43 of 45 decisive comparisons.
- That the graft is better than training v9 from scratch or than finishing final: neither exists.
- That the cost model predicts chip time for small contractions (see tokn).

## Added 12:25 UTC (24 Nov)
- Held-out loss of the v9 run after the graft, same 64 rows (web 65,442 / chat 60,438 / proxy
  11,094 weighted tokens), CPU, per checkpoint (step, tokens, training hours, total, web, chat,
  chat assistant turns, proxy):
  - 31,912  8.366e9   9.874 h  2.2337  2.7662  1.5865  1.4830  2.6187
  - 33,038  8.661e9  10.291 h  2.2204  2.7541  1.5752  1.4713  2.5867
  - 34,166  8.956e9  10.708 h  2.2111  2.7437  1.5630  1.4569  2.6000
  - 35,294  9.252e9  11.126 h  2.2001  2.7344  1.5508  1.4463  2.5849
  Before the graft (final): 26,890: 2.2679; 28,501: 2.2624; 29,722 (8.978 h): 2.2598.
  The control (final continued) does not exist after 8.978 h: any "v9 is ahead by x" is against
  an extrapolation and must be worded as such. The learning rate also decays over this stretch.
- Generation cost on host CPU for v9 (export's own decoder, 16 workers x 1 core): 60 dev prompts
  = 75 replies in 1,798 s wall including export and cache calibration.
- Incident 12:10: the container's disk filled (20 GB of scratch written by research agents); no
  process was lost; cleaned in 4 minutes.

## Added 12:52 UTC (24 Nov): second blind development comparison (study s2)
- Model: v9 at step 31,912 (p = 0.56, 8.366e9 tokens), the export's default decoding; 60 dev prompts
  (the same 60 as s1); judges: Opus sub-agents, blind, same instructions as s1.
- Against Falcon-H1-Tiny-90M-Instruct, first reading: 3 wins / 41 losses / 16 ties, score 0.183
  [0.106, 0.299]. With a second reading on exchanged sides (rating source judge-opus, the two
  readings combined per pair): 3/39/18, score 0.200 [0.118, 0.318]; the two orders agree on 93.3%
  of pairs; mean preference for the first-shown reply +0.000. By category (combined): chat 0/10/5,
  code 1/6/2, instruct 2/7/3, math 0/5/4, writing 0/11/4; two-turn prompts 0/14/1.
- Against MobileLLM-R1.5-140M (its recommended settings), first reading: 20/7/33, score 0.608
  [0.482, 0.722], sign test p = 0.019. By category: chat 5/0/10, code 1/2/6, instruct 9/0/3,
  math 0/5/4, writing 5/0/10; two-turn 7/0/8. With the second reading on exchanged sides
  (source judge-opus): 20/7/33, score 0.608 [0.482, 0.722], sign test p = 0.019; the two orders
  agree on 98.3% of pairs; mean preference for the first-shown reply +0.008.
- s1 for comparison (final at step 25,280, one reading): Falcon 2/43/15 (0.158), MobileLLM
  15/8/37 (0.558). The change from s1 to s2 is inside the intervals; it is not evidence that v9
  is better than final.
- no_repeat_ngram=3 against the default decoding at step 31,912: 4/8/48 (score 0.467): not adopted.
- What the judges say in absolute terms (raw verdicts of the first reading): in the
  ngram3-vs-default set (both replies ours) 44 of 60 pairs were "both_bad" (4 ties, 12 with a
  winner); against MobileLLM 31 of 60 were "both_bad"; against Falcon 14 of 60. At p = 0.56 about
  three quarters of our replies to these prompts are judged unusable (fluent, on the topic, wrong
  or empty of content); the report must say so.

## Added 12:53 UTC: the T=2048 phase
- The run entered the 2,048-token phase at p = 0.67 without a pause: log line of step 36,950 at
  154.89 min of this process (the previous, step 36,450, at 143.72 min): 500 steps in 11.17 min,
  p 0.672, loss 2.4183, T 2048. First full 500 steps at 2,048: step 37,450 at 166.46 min, i.e.
  500 steps in 11.57 min = 1.388 s per step = 188.8 thousand tokens per second (at 1,024 the
  same run made 500 steps in 11.10-11.17 min = 196.6 thousand); loss 2.3542, p 0.683.
  (Synthetic-batch measurement before the switches: 1,510.2 ms at 2 x 2048 against 1,433.1 ms at
  4 x 1024, +5.4%; live with the three switches: +4.2%.)

## Added 13:32 UTC: the two opponents against each other (reference for the scale)
- Pair set `ref-falcon-vs-mobilellm-test`: Falcon-H1-Tiny-90M-Instruct (x) against
  MobileLLM-R1.5-140M (y, recommended preset, qwen3 template), the 200 TEST prompts (52 two-turn),
  Opus readers, two readings on exchanged sides (source judge-opus): Falcon 144 wins / 6 losses /
  50 ties-or-both-bad, score 0.845 [0.788, 0.889], decisive 0.960 [0.915, 0.982]; the two orders
  agree on 93.5% of pairs; first-position preference +0.013. By category: chat 37/1/12, code
  17/2/11, instruct 31/0/9, math 15/3/12, writing 44/0/6; two-turn 39/1/12.
  Report: `N/study/proxy/reports/ab/ref-falcon-vs-mobilellm-test--judge-opus.{md,json}`.
- Reading: the two opponents are far apart on this prompt set. MobileLLM-R1.5 is a reasoning
  model; the readers' notes describe many of its replies as looping meta-reasoning with boxed
  answers. "Ahead of MobileLLM-R1.5" on these prompts is therefore a low bar, and the paper must
  say so; "behind Falcon-H1-Tiny" is the comparison that matters.
- This set involves no reply of ours: it does not open the test split for our model.

## Added 13:52 UTC: the pre-registered automatic judge on the development pairs
- Judge `qwen3-4b-2507` (Qwen3-4B-Instruct-2507, Q4_K_M GGUF, SHA-256 3605803b..., the default
  judge of docs/EVAL_PROTOCOL.md section 5), `python -m bes.eval.proxy judge`, both orders, a side
  only when both orders pick the same reply; run on the runtime's idle host cores (12 threads per
  set, 13.1 to 17.2 s per pair for both orders), rating source `judge-qwen3-4b-2507-3605803b`:

  | pair set | W/L/T | score [95%] | decisive | sign p | orders agree | first-position pref | clipped |
  |---|---|---|---|---|---|---|---|
  | s1 (final, step 25,280) vs Falcon | 2/16/42 | 0.383 [0.271, 0.510] | 0.111 | 0.0013 | 63.3% | -0.096 | 0 |
  | s1 vs MobileLLM | 7/4/49 | 0.525 [0.401, 0.646] | 0.636 | 0.55 | 78.3% | -0.000 | 30 pairs |
  | s2 (v9, step 31,912) vs Falcon | 1/16/43 | 0.375 [0.264, 0.501] | 0.059 | 0.0003 | 60.0% | -0.114 | 0 |
  | s2 vs MobileLLM | 7/2/51 | 0.542 [0.417, 0.661] | 0.778 | 0.18 | 80.0% | -0.000 | 30 pairs |

- Against the Opus readers on the same pairs (s2): vs Falcon both gave a side in 16 pairs, the
  same side in 14 and opposite sides in 2; in 26 pairs the Opus readers gave a side where the
  4B judge's two orders disagreed or said C; 1 the other way; 17 both tie. vs MobileLLM: both a
  side in 6 pairs, all 6 the same side; 21 Opus side / judge tie; 3 the other way; 30 both tie.
- Reading: the pre-registered judge points the same way (behind Falcon, 1 or 2 wins to 16 losses;
  no significant difference to MobileLLM) with far fewer decisive pairs: its two orders agree on
  only 60 to 63% of the Falcon pairs and it prefers the second-shown reply there (-0.10, -0.11).
  The final report gives both: the registered judge as registered, the Opus readers as the
  stronger reading, named as a deviation.
- Housekeeping: six files `ratings/s1-*/judge-opus.jsonl` held single-reading records under the
  wrong source name (my import mistake at 10:20; identical verdicts to `opus-1.jsonl`); moved to
  `N/study/stale_ratings/` and removed from the repository copy.

## Added 14:12 UTC: the second registered judge on the development pairs
- Judge `gemma4-e4b` (gemma-4-E4B-it Q4_K_M GGUF, SHA-256 85a896a0..., the second judge of
  docs/EVAL_PROTOCOL.md "from another family"), same procedure, runtime host cores (7.6 to 9.3 s
  per pair for both orders at 12 threads), rating source `judge-gemma4-e4b-85a896a0`:

  | pair set | W/L/T | score [95%] | orders agree | first-position pref |
  |---|---|---|---|---|
  | s1 (final, step 25,280) vs Falcon | 0/4/56 | 0.467 [0.346, 0.591] | 80.0% | -0.087 |
  | s1 vs MobileLLM | 2/0/58 | 0.517 [0.393, 0.638] | 91.7% | -0.008 |
  | s2 (v9, step 31,912) vs Falcon | 0/3/57 | 0.475 [0.354, 0.599] | 83.3% | -0.065 |
  | s2 vs MobileLLM | 2/1/57 | 0.508 [0.385, 0.630] | 93.3% | +0.008 |

- Reading: this judge answers C (tie or both poor) on 93 to 97% of the pairs; where it takes a
  side it agrees in direction with the other readers (0 wins against Falcon, 3 or 4 losses). As a
  measuring instrument on these pairs it has almost no resolution; say so, do not average it in.
- Three readers on the same 60 pairs of s2 vs Falcon, decisive pairs (wins:losses): Opus two
  readings 3:39; qwen3-4b-2507 1:16; gemma4-e4b 0:3.

## Added 16:50 UTC: the micro graft experiment (controlled; CPU; the repository's model at the micro width)
- Code: commit b04478e (`bes/eval/microbench.py` graft arms, `scripts/micro_ablate.py`,
  `scripts/micro_graft_report.py`, `tests/eval/test_micro_graft.py`, `docs/MICRO_GRAFT.md`);
  written by one agent, reviewed adversarially by a second (trunk equality, function
  preservation, carried optimiser state, schedule and data order, unchanged E2 hashes, report
  arithmetic: `W/micro_graft.md`).
- Run: 15 arms in parallel on the idle host cores of the training runtime (AMD EPYC 9B14; one
  physical core = 2 threads per arm; CPU only, `JAX_PLATFORMS=cpu`; a separate clone of the
  repository at ef5649c plus the patch), 24 Nov 14:12:51 to 16:41:01 UTC; the training rate of the
  main run did not change (11.57 and 11.59 min per 500 steps before and during).
  Arms: `ref` (the micro reference, 4,402,372 numbers), `v9_pages` (the v9 page from the first
  step: output gains and 8 token neurons per token and block in a d/4 space; 14,904,516 numbers),
  `graft_25`, `graft_50`, `graft_75` (`ref` up to 0.25 / 0.5 / 0.75 of the budget, then grafted into
  `v9_pages`: the 59 shared leaves with their optimiser state, count and step carried, the 5 new
  leaves, 10,502,144 numbers, as a fresh `v9_pages` tree of the same seed). Seeds 0, 1, 2 each.
  Budget 6,000,000 tokens = 1,464 steps of 4,096 tokens (rows of 512), d_model 128; data: web
  (77%) and chat (23%) pools of 4.20M and 1.79M tokens, so about one pass; held-out: 32 web rows
  and 32 chat rows (other files). Same data order and schedule in all arms of a seed.
- Checks inside the run: every graft arm's evaluations before its graft equal `ref`'s of the same
  seed exactly; the loss on held-out rows before and after each of the 9 grafts differs by
  exactly 0 (also per row).
- Results at the end (5,996,544 tokens), held-out loss, mix of web and chat (nats per token),
  mean ± sd over 3 seeds: ref 4.1912 ± 0.0076; v9_pages 4.1394 ± 0.0083; graft_25 4.1363 ± 0.0075;
  graft_50 4.1555 ± 0.0091; graft_75 4.1789 ± 0.0074.
  Paired by seed (mean ± sd of the three differences):
  - v9_pages - ref: -0.0519 ± 0.0107 (-1.24%); web -0.0464 ± 0.0118; chat -0.0683 ± 0.0089;
    assistant turns -0.0689 ± 0.0107. Over training: -0.0164 at 1.5M, -0.0329 at 3.0M,
    -0.0494 at 4.5M tokens.
  - graft_25 - ref: -0.0550 ± 0.0010; graft_25 - v9_pages: -0.0031 ± 0.0113 (no difference).
  - graft_50 - ref: -0.0358 ± 0.0030; graft_50 - v9_pages: +0.0161 ± 0.0096.
  - graft_75 - ref: -0.0123 ± 0.0004; graft_75 - v9_pages: +0.0396 ± 0.0103.
  Share of the page gain recovered, (ref - graft) / (ref - v9_pages), ratio of the mean
  differences (per seed in brackets): graft_25 1.06 (0.86, 1.33, 1.08); graft_50 0.69 (0.62, 0.87,
  0.64); graft_75 0.24 (0.20, 0.29, 0.24). By domain the shares are the same within 0.04.
  One checkpoint after a graft (1.5M tokens later): graft_25 has -0.0192 of ref (v9_pages had
  -0.0164 after its first 1.5M tokens); graft_50 -0.0161; graft_75 -0.0123.
- What it says, and for what: at the micro width a graft never did worse than not grafting; a
  graft at half the budget recovered about 0.7 of what the page gives when it is there from the
  start, a graft at a quarter all of it, a graft at three quarters about a quarter. The gain of a
  graft after its first 1.5M tokens is about the gain the from-scratch page had after its first
  1.5M tokens (-0.012 to -0.019 against -0.016): the page learns at about the same rate whenever
  it is added, and what a late graft loses is time.
- Limits (say them): d_model 128 against 768; 6M tokens against 1.36e10; one pass over a small
  pool; three seeds (2 degrees of freedom; the per-seed shares of graft_25 range from 0.86 to
  1.33 because the denominator is a difference of two runs); the schedule is in tokens, not wall
  clock; the data stream does not change at the graft (the live one did); the page's effect is
  1.2% here. It is evidence about the mechanism, not about the size of the effect in the live run.
- Being run now (16:43 to about 17:40): "restart" arms, `v9_pages` from scratch on a budget of
  3,000,000 tokens with its own schedule (three seeds): what a restart with the remaining half
  would have given, to set against graft_50.
- Files: `N/micro_graft/6M/graft_report.txt`, `graft_report.json`, `results_all.jsonl` (15
  records with curves, graft checks, timing), `<arm>.log`, `progress.log`.

## Added 16:52 UTC: held-out loss through the 2,048-token phase
- `N/heldout_v9.jsonl` (downloaded 16:50; one record per checkpoint; evaluated by the loop on the
  runtime host, CPU, float32, rows of 1,024 tokens as before; total / web / chat / chat assistant
  turns / proxy):
  - 36,422   9.548e9  11.543 h  2.1881  2.7254  1.5413  1.4355  2.5430
  - 37,522   9.836e9  11.960 h  2.1764  2.7137  1.5225  1.4167  2.5700
  - 38,604  10.120e9  12.377 h  2.1653  2.7066  1.5042  1.3967  2.5742
  - 39,686  10.403e9  12.794 h  2.1527  2.7035  1.4838  1.3761  2.5473
  - 40,768  10.687e9  13.212 h  2.1398  2.6997  1.4627  1.3546  2.5258
  - 41,850  10.971e9  13.629 h  2.1299  2.6977  1.4443  1.3339  2.5157
  - 42,932  11.254e9  14.046 h  2.1178  2.6922  1.4230  1.3111  2.5151
  - 44,014  11.538e9  14.463 h  2.1095  2.6910  1.4112  1.2965  2.4841
  - 45,096  11.822e9  14.880 h  2.0997  2.6857  1.3942  1.2795  2.4868
  - 46,178  12.105e9  15.298 h  2.0914  2.6867  1.3790  1.2651  2.4614
- Observation to explain, not to celebrate: since the phase change (step 36,867) the chat rows
  fall much faster than the web rows (assistant turns 1.4355 -> 1.2651, -0.170; web 2.7254 ->
  2.6867, -0.039). Candidate causes that this run cannot separate: the learning-rate decay; the
  chat sources being read for a second or third time (the stream's `max_epochs`), with held-out
  conversations that resemble training ones (templated synthetic chats), so that memorisation of
  the source looks like generalisation on this held-out set; the longer rows. The blind test
  prompts are independent of the chat sources; they, not this number, say whether replies got
  better. The second blind study was made at step 31,912 (assistant-turn loss 1.4830).
- Training log at 16:44: step 46,950, p 0.893, loss 1.8527 (rows of 2,048), 189.0k tokens/s.

## Added 17:35 UTC: the "restart" arms of the micro graft experiment
- Three runs of `v9_pages` from scratch on a budget of 3,000,000 tokens (732 steps, the first half
  of the 6M budget's data order, with the schedule of a 3M budget), seeds 0, 1, 2; same machine
  and layout, 16:42:47 to 17:29:56 UTC. Records: `N/micro_graft/3M/<arm>/results.jsonl`
  (`results.tar.gz`), logs `N/micro_graft/3M/<arm>.log`. (The driver's report step exits 1 there
  by design: the report needs all five arm families.)
- Final held-out loss (2,998,272 tokens), mix = 0.75 web + 0.25 chat as in the report:
  4.3891, 4.3885, 4.3917 (seeds 0, 1, 2); mean 4.3897 ± 0.0017. Web 4.6317 ± 0.0043, chat
  3.6637 ± 0.0074, assistant turns 3.6167 ± 0.0106.
- Against the 6M-budget arms at their end, paired by seed (mean ± sd of three differences, mix):
  - graft_50 - restart: -0.2342 ± 0.0075 (web -0.2152 ± 0.0065, chat -0.2915 ± 0.0220);
  - ref (no page, the whole budget) - restart: -0.1985 ± 0.0062;
  - v9_pages (page from the start, the whole budget) - restart: -0.2504 ± 0.0074.
  The restart's own schedule is worth -0.0509 ± 0.0086 against `v9_pages` at the same token count
  inside the 6M-budget run (4.4407): a run that ends at 3M tokens anneals; the comparison above
  already includes that advantage.
- Reading: with half of the budget spent on the page-less configuration, continuing with a graft
  ended 0.234 nats below starting the page configuration again with the half that was left, and
  0.016 above having had the page from the first step. Even not adding the page at all ended
  0.199 below the restart: at this width and budget the tokens already trained are worth about
  four times the whole gain of the page (0.052). This is the comparison the live decision at
  p = 0.505 rested on (a restart "would have seen fewer tokens than the run already had").
  Limits as for the main experiment; in addition the restart here reads the same first half of
  the data again instead of the unseen second half (the pools are sampled without regard to
  what was seen; about one pass in all), which should not matter and was not tested.

## Added 20:32 UTC: the end of the run (primary records: `N/main16.final.log`, `N/records/unpacked/`,
## `N/packs/official_export/x/final/`, `N/snap_final/`)
- Training ended 18:33:49 UTC at step 51,812 (`train: end step=51812 reason=deadline p=1.0`),
  13.582e9 tokens (ledger; exact count in `records/unpacked/run/train/ledger.json`); the train
  stage's own clock: 65,208 s. Final training checkpoint `ckpt-000022-step-000051812`.
- Post-training stages (run clock in minutes; UTC = 18:33:49 at 1499.01):
  - sample: 193 s; 17 prompts, about 28 pairs (5.9 prompts per minute on the chip).
  - critic: skipped (0 minutes planned). judge: skipped (0 minutes planned).
  - prefopt (DPO, beta 0.1): reference log-probabilities of 35,201 + 1,826 pairs (UltraFeedback
    and sampled; 1,826 held out); two candidates of 551 steps each, lr 1e-6 (accepted) and 3e-6
    (accepted; kept): "prefopt: accepted (lr 3e-06); post-DPO weights saved"; 1,784 s of a
    3,600 s slice. Details in `reports/prefopt_report.json` of the export (held-out preference
    accuracy, held-out loss, length and repetition statistics per candidate).
  - select: 3,295 s (slice 3,897 s); three decoding variants on the 200 dev prompts, exact
    scores: default 0.91, k1 0.85, noband 0.915; chosen: default ("library not shipped: no
    library was built"). Their replies are in `records/unpacked/proxy/outputs/ours-step51812-c04f3f6b3fca71f1/`.
  - export: 35 s; `/content/runs/main16/post/ckpt/ckpt-000001-step-000051812` (prefopt weights)
    -> `final/`; 16-bit check passed: float32 2.34440, 16-bit 2.34443 nats on 43,138 held-out
    tokens (chat 1.31725 / 1.31721, proxy 3.28416 / 3.28419, web 2.69236 / 2.69243).
  - The run finished at 1587.46 min of its clock (about 20:02 UTC), 29 minutes before its
    deadline. Stage table: train used 65,208.3 s (15 tries, 20,488 s down), sample 192.5,
    prefopt 1,783.8, select 3,295.3, export 35.0.
- The official model: weights after preference optimisation, default decoding (6 particles,
  temperature 0.6, top-p 0.9, 768 tokens, no library). 98,036,104 stored numbers in 16 bits,
  19,631,240 read per token (model card). `weights16.npz` SHA-256
  2da7cbd2e2f1fe49e805cb9bde62b15751f41ae92e71f00a3b0ba69f241afcf4. In the repository:
  `release/main16-v9/` (12 parts of 16 MB, commits 99ba0e7..994b3a1), rebuilt and verified from
  the committed parts (`scripts/export.py --verify`: every file matches, check passed).
- Test replies (CPU, host cores of the runtime, 40 workers, the export's own decoder):
  - trained weights (step 51,812, before preference optimisation): 18:36:17 to 19:07:06,
    200 prompts / 252 replies, 1,840 s; run `ours-export-step51812-ef5599caa3974edc/final-trained-9a89782942-s0`.
  - weights after preference optimisation: 19:31:12 to 20:06:32, 2,110 s; run
    `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`. Its export has the
    same source checkpoint, the same fingerprint (8d19db7f...), the same `weights16.npz` SHA-256
    and the same decode choice as the run's own export: these are the official model's replies.
  - Pair sets (test split, 200 pairs each, 52 two-turn): `final-prefopt-vs-falcon`,
    `final-prefopt-vs-mobilellm` (official), `final-trained-vs-falcon`,
    `final-trained-vs-mobilellm`.
- Exact scores on the 200 test prompts (`python -m bes.eval.proxy exact RUN --split test`;
  reports `N/study/proxy/reports/exact/*-test.{md,json}`): maths, 30 prompts, share correct:
  official 0.200, trained 0.200, Falcon 0.500, MobileLLM 0.400. Constraints, 48 prompts, all
  constraints of the prompt met: official 0.792, trained 0.854, Falcon 0.688, MobileLLM 0.167
  (single constraints, 56: 0.821 / 0.857 / 0.732 / 0.179).
- Held-out loss at the end (`N/heldout_v9.jsonl`, the training checkpoint; total / web / chat /
  assistant turns / proxy): 49,424: 2.1335 2.6815 1.3671 1.2148 3.0764; 50,506: 2.1411 2.6762
  1.3684 1.2071 3.1946; 51,588: 2.1394 2.6719 1.3638 1.1981 3.2241; 51,812: 2.1422 2.6690 1.3668
  1.1983 3.2580. The proxy rows (user turns of the prompt set, 12 rows) ROSE from 2.4614 at
  46,178 to 3.2580 while web and chat fell: with the chat share at 0.70 and user turns carrying
  no loss weight in chat rows the model predicts user-turn text worse and worse. It is why the
  total rose after 46,178 (2.0914 -> 2.1422). Say it; do not hide it in the total.
- Operations at the end: the session's worker was restarted twice (19:08-19:30 and
  19:32-20:13 UTC) when a tool call waited for an approval that nobody could give; the local
  loops (watch, presence, budget guard) were down for those 22 and 41 minutes and were restarted
  by hand. The run on the runtime was not affected. No Opus reading of the test pairs has been
  made yet for that reason [TBD: after the TPU is given back].

## Added 21:00 UTC: the TPU is given back; first official test results
- The TPU runtime was stopped at 20:41:04 UTC (`colab_cli.py send release`, `down --yes`); the
  ledger `S/tpu_intervals.txt` is closed: 7,325 + 8,220 + 70,509 = 86,054 s of v6e-1 in all (the
  limit was 86,400 s; 346 s unused). Third interval 1795482355 to 1795552864.
- Official model against Falcon-H1-Tiny-90M-Instruct, TEST split, 200 prompts (52 two-turn), Opus
  5.5 readers as background agents (the same written instructions as the development studies,
  `N/final_eval/JUDGE_INSTRUCTIONS.md`), two readings on exchanged sides, rating source
  `judge-opus` (`N/study/proxy/reports/ab/final-prefopt-vs-falcon--judge-opus.{md,json}`):
  35 wins / 108 losses / 57 ties-or-both-bad; score 0.318 [0.257, 0.385]; decisive 0.245
  [0.182, 0.321]; sign test p < 0.0001; the two orders agree on 90.5% of pairs; first-position
  preference +0.005. By category: chat 10/24/16 (0.360), code 10/13/7 (0.450, p 0.68),
  instruct 7/21/12 (0.325), math 4/18/8 (0.267), writing 4/32/14 (0.220); single-turn 28/78/42,
  two-turn 7/30/15. By length (our words / theirs): ours shorter 26/41/26 (0.419, p 0.086),
  similar 7/40/25 (0.271), ours longer 2/27/6 (0.143).
- The registered judge (qwen3-4b-2507, both orders), PARTIAL at 20:40 on the TPU host, resumed on
  a CPU runtime at 20:41: against Falcon 113 of 200 pairs: 7/24/82, score 0.425 [0.338, 0.517],
  decisive 0.226, p 0.0033, orders agree 57.5%, first-position preference -0.170; against
  MobileLLM-R1.5 73 of 200: 18/3/52, score 0.603 [0.488, 0.707], decisive 0.857, p 0.0015, orders
  agree 74.0%, 32 pairs clipped. [TBD: all 200 pairs of each, about 22:00 UTC.]
- Degenerate replies on the test split (exact scorer): official 2.0% (all "truncated": the reply
  used its 768 tokens), trained 1.2%, Falcon 0.8%, MobileLLM-R1.5 62.7% (59.5% truncated, 21.4%
  repeated lines, 5.6% loops). Mean words per reply: official 130, trained 119, Falcon 173,
  MobileLLM 474.
- Registered control arms of Falcon on the test split (CPU runtime, 18:42-19:17 UTC;
  `python -m bes.eval.proxy run --model falcon --preset recommended --split test --samples 6
  --select ...`): runs `falcon-e60034381f4a/recommended-2dcb1c92a6-s0` (K = 6, kept by its own
  mean log-probability) and `.../recommended-d1f2a63f82-s0` (K = 6, kept by OUR decoder's
  selector). Exact scores (maths correct of 30 / all constraints met of 48): Falcon as released
  0.500 / 0.688; K = 6 log-probability 0.567 / 0.729; K = 6 our selector 0.600 / 0.875. Ours
  (official, K = 6 particles with that selector): 0.200 / 0.792. So the constraint score in which
  our model leads the released Falcon (0.792 against 0.688) is the selector's doing: with the same
  selector over the same number of samples Falcon scores 0.875. Pair sets against the official
  model: `final-prefopt-vs-falcon-k6lp`, `final-prefopt-vs-falcon-k6ds` [TBD: judged].
  MobileLLM's K = 6 arms are being generated since 19:17 [TBD].
- K = 1 arm of ours on the test split (`--set extra.particles=1`, the export rebuilt from the
  parts committed in `release/main16-v9/`): started 20:57 on a CPU runtime [TBD].

## Added 21:08 UTC: the official model against MobileLLM-R1.5 (Opus readers, both readings)
- Pair set `final-prefopt-vs-mobilellm`, TEST split, 200 prompts (52 two-turn), Opus 5.5 readers as
  background agents, two readings on exchanged sides, source `judge-opus`
  (`N/study/proxy/reports/ab/final-prefopt-vs-mobilellm--judge-opus.{md,json}`):
  116 wins / 15 losses / 69 ties-or-both-bad; score 0.752 [0.688, 0.807]; decisive 0.885
  [0.820, 0.929]; sign test p < 0.0001; the two orders agree on 96.0% of pairs; first-position
  preference -0.005. By category: chat 28/1/21 (0.770), code 16/2/12 (0.733), instruct 29/0/11
  (0.863), math 7/12/11 (0.417, p 0.36: MobileLLM ahead, not significantly), writing 36/0/14
  (0.860); single-turn 83/11/54, two-turn 33/4/15. By length: ours shorter 77/9/54 (0.743),
  similar 17/5/6 (0.714), ours longer 22/1/9 (0.828).
- Headline for the report: against MobileLLM-R1.5-140M the official model is preferred (116 to 15);
  against Falcon-H1-Tiny-90M-Instruct it is not (35 to 108). Both with the caveats on record:
  MobileLLM-R1.5's replies are degenerate in 63% of test prompts under its recommended preset in
  this harness; the readers are Opus 5.5 agents (an addition to the registered protocol; the
  registered judge's full pass is running); 11 of the 200 prompts were seen during the run.
- A direct pair set of the two weights, `final-prefopt-vs-trained` (after against before
  preference optimisation, same decoding, 200 test prompts; 4 prompts have identical replies), is
  being read the same way [TBD].

## Added 21:12 UTC: after against before preference optimisation (direct blind comparison)
- Pair set `final-prefopt-vs-trained` (x: the official weights after preference optimisation;
  y: the trained weights before it; same decoding; TEST split, 200 prompts; 4 prompts have
  identical replies), Opus readers, two readings on exchanged sides, source `judge-opus`
  (`N/study/proxy/reports/ab/final-prefopt-vs-trained--judge-opus.{md,json}`): 59 / 34 / 107;
  score 0.562 [0.493, 0.629]; decisive 0.634 [0.533, 0.725]; sign test p = 0.0124; orders agree
  91.5%; first-position preference +0.010. By category: chat 16/8/26, code 10/1/19 (p 0.012),
  instruct 14/8/18, math 5/5/20, writing 14/12/24; single-turn 45/25/78, two-turn 14/9/29.
  Reading: the 30 minutes of preference optimisation (551 steps at 3e-6 on 35,201 pairs) moved
  blind preference in its favour, by a modest margin that is significant on 200 prompts; exact
  scores did not improve (maths 0.200 both; constraints 0.792 after, 0.854 before).
- Raw verdicts of the two readings (A / B / tie / both_bad of 200), counted from
  `N/final_eval/prefopt/<set>/r1|r2/chunk_*.verdicts.json`:
  - `final-prefopt-vs-trained` (both replies ours): r1 55 / 45 / 11 / 89; r2 50 / 56 / 5 / 89:
    "both_bad" in 44.5% of pairs in each reading. At step 31,912 the same kind of set (two decodes
    of one model, 60 dev prompts) had 73% "both_bad". On these prompts the official model's reply
    is judged unusable together with its sibling's a little under half of the time.
  - `final-prefopt-vs-falcon`: r1 71 / 81 / 4 / 44 (22.0% both_bad); r2 84 / 72 / 3 / 41 (20.5%).
  - `final-prefopt-vs-mobilellm`: r1 65 / 69 / 0 / 66 (33.0%); r2 69 / 67 / 0 / 64 (32.0%).
  (A and B are blinded sides; the key decides which is ours.)

## Added 21:27 UTC: Falcon's registered control arms (Opus readers); K = 1 replies made; the judge's threads
- **Falcon K-matched control** (`final-prefopt-vs-falcon-k6lp`: the official model against Falcon
  with 6 samples per reply, kept by Falcon's own mean log-probability; TEST, 200 prompts; Opus
  readers, two readings on exchanged sides, source `judge-opus`;
  `N/study/proxy/reports/ab/final-prefopt-vs-falcon-k6lp--judge-opus.{md,json}`): ours **35 / 112 / 53**;
  score 0.307 [0.248, 0.375]; decisive 0.238 [0.176, 0.313]; p < 0.0001; orders agree 93.5%;
  first-position preference +0.020. Categories: chat 8/24/18, code 11/13/6, instruct 9/21/10,
  math 4/19/7, writing 3/35/12; single-turn 26/80/42, two-turn 9/32/11. Length buckets (ours
  shorter / similar / longer): 26/38/30, 8/51/17, 1/23/6.
- **Falcon selector-matched control** (`final-prefopt-vs-falcon-k6ds`: Falcon with 6 samples per
  reply, kept by OUR decoder's selector): ours **29 / 116 / 55**; score 0.282 [0.225, 0.348];
  decisive 0.200 [0.143, 0.273]; p < 0.0001; orders agree 92.5%; first-position preference +0.018.
  Categories: chat 7/25/18, code 11/12/7, instruct 7/22/11, math 3/20/7, writing 1/37/12;
  single-turn 21/84/43, two-turn 8/32/12. Length buckets: 19/37/33, 10/55/17, 0/24/5.
- Paired comparison of our per-prompt score against released Falcon and against each control arm
  (`python -m bes.eval.proxy compare`, same 200 prompts, source `judge-opus`): released against
  K-matched: 29 prompts better against released, 25 better against the control, 146 the same,
  mean difference +0.010, p = 0.68; released against selector-matched: 34 / 21 / 145, mean
  difference +0.035, p = 0.10. Reading: giving Falcon the same K and even our selector moves the
  blind score by 1 to 3.5 points, not significantly on 200 prompts; the loss to Falcon is not an
  artefact of comparing our 6-particle decode with a single Falcon sample, and Falcon does not
  need our selector to win. (Exact scores, already recorded: constraints all met 0.688 released,
  0.729 K-matched, 0.875 selector-matched; maths 0.500 / 0.567 / 0.600.)
- Raw verdicts (A / B / tie / both_bad of 200) from
  `N/final_eval/controls/<set>/r1|r2/chunk_*.verdicts.json`: k6lp r1 84 / 72 / 2 / 42, r2 75 / 79 /
  1 / 45; k6ds r1 70 / 79 / 8 / 43, r2 86 / 70 / 5 / 39.
- **K = 1 arm generated** (registered disclosure "results with K = 1"): the official export,
  rebuilt on a CPU runtime from the 12 parts committed under `release/main16-v9/` and verified,
  `extra.particles=1`, every other setting the export's; run id
  `ours-export-step51812-47fd99c151a50a58/final-k1-f93654208f-s0` (same weights key as the
  official replies); 200 / 200 prompts, 252 replies, 1,356 s on 8 workers. Pair sets
  `final-k1-vs-falcon` and `final-k1-vs-mobilellm` published at 21:23; Opus readings started.
- **Registered judge speed on the 8-vCPU CPU runtimes** (seconds per pair, both orders, from the
  rating records): 9.8 s on the TPU host's idle cores (16 threads, first 113 pairs of the Falcon
  set); 63.8 s with 4 threads while a second 4-thread stream ran beside it (pairs 114 to 136);
  137.6 s with 8 threads alone (pairs 137 to 141: 8 threads on 4 physical cores is slower than 4).
  At 21:22 the judge was restarted as two 4-thread streams, one per official set (tag `testq3`,
  runtime `bes-cpu2`), resuming from 142 Falcon ratings and 89 MobileLLM ratings.
- **MobileLLM control arms**: generation of the 6-sample replies runs at about 0.35 prompts per
  minute on one 8-vCPU runtime (40 of 200 prompts after 115.6 minutes: MobileLLM's replies often
  run to the token limit). A fourth CPU runtime was refused by Colab ("too many active
  sessions": three is the limit of this account). From 21:23 a second runtime generates the same
  arm from the end of the prompt order backwards (`controls_rev_driver.py`; the generation cache
  is keyed by backend, messages, settings and seed, so the two caches are concatenated and the
  outputs assembled from the merged cache).

## Added 21:40 UTC: the K = 1 arm (Opus readers, both readings); 189-prompt subsets; judge shards
- **K = 1 against Falcon** (`final-k1-vs-falcon`: the official weights with one particle, every
  other setting the export's; TEST, 200 prompts; Opus readers, two readings on exchanged sides,
  source `judge-opus`; `N/study/proxy/reports/ab/final-k1-vs-falcon--judge-opus.{md,json}`):
  **22 / 122 / 56**; score 0.250 [0.195, 0.314]; decisive 0.153 [0.103, 0.221]; p < 0.0001; orders
  agree 92.5%; first-position preference +0.015. Categories: chat 9/31/10, code 7/13/10 (p 0.26),
  instruct 2/25/13, math 3/19/8, writing 1/34/15; single-turn 16/85/47, two-turn 6/37/9. Length
  buckets (ours shorter / similar / longer): 15/45/21, 5/46/28, 2/31/7.
- **K = 1 against MobileLLM-R1.5** (`final-k1-vs-mobilellm`): **99 / 16 / 85**; score 0.708
  [0.641, 0.766]; decisive 0.861 [0.786, 0.912]; p < 0.0001; orders agree 91.5%; first-position
  preference +0.007. Categories: chat 28/2/20, code 13/4/13 (p 0.049), instruct 24/0/16, math
  6/10/14 (p 0.45), writing 28/0/22; single-turn 67/13/68, two-turn 32/3/17. Length buckets:
  71/9/60, 6/5/12, 22/2/13.
- **What the 6 particles are worth** (paired, same 200 prompts, same opponent, source
  `judge-opus`; `python -m bes.eval.proxy compare --a final-prefopt-vs-X --b final-k1-vs-X`):
  against Falcon the official K = 6 decode scores better than K = 1 on 31 prompts, worse on 12,
  the same on 157; mean per-prompt difference +0.0675; p = 0.0054. Against MobileLLM: 28 / 13 /
  159; +0.045; p = 0.028. So the particle decoder with its selector is worth 4.5 to 7 points of
  blind score at 6 times the decoding compute; without it the model loses to Falcon 22 to 122.
- K = 1 exact scores on the test split: maths correct 0.167 (5 of 30; K = 6: 0.200), all
  constraints met 0.771 of 48 (K = 6: 0.792), single constraints 0.786 of 56 (0.821), degenerate
  replies 6.75% (all cut at the 768-token limit; K = 6: 1.98%), mean words 149.7 (130.4).
- Raw verdicts of the K = 1 readings (A / B / tie / both_bad of 200): against Falcon r1 76 / 75 /
  4 / 45, r2 79 / 74 / 6 / 41; against MobileLLM r1 61 / 63 / 2 / 74 (37.0% both_bad), r2 64 / 59 /
  3 / 74.
- On the 189 prompts never looked at during the run (`lead4_work/scripts/restrict_report.py
  --exclude` the 11 seen ids): K = 1 against Falcon 20 / 116 / 53, score 0.246 [0.190, 0.312];
  K = 1 against MobileLLM 92 / 14 / 83, 0.706 [0.638, 0.767]; official against Falcon K6
  log-probability 34 / 106 / 49, 0.310 [0.248, 0.379]; official against Falcon K6 our selector
  27 / 110 / 52, 0.280 [0.221, 0.348].
- **The registered judge is slow on the MobileLLM set**: with two 4-thread streams on one runtime
  it took 39.9 s per pair on the Falcon set and 177.7 s per pair on the MobileLLM set (MobileLLM's
  replies are long: most are cut at the judge's 4,000-character limit). At 21:36 the reverse
  generation of MobileLLM's control arm on `bes-cpu3` was stopped (11 minutes in; no prompt was
  complete) and that runtime was given to the judge: two more streams on index ranges
  [146, 173) and [173, 200) of the MobileLLM pair file (`judge_shard.py`: the registered
  `judge_pair` unchanged, the same model file, records with the pair set's commitment and the
  pair's digest; only the scheduling differs from `python -m bes.eval.proxy judge`). The first
  runtime keeps generating MobileLLM's control arm forwards.

## Added 22:13 UTC: the registered judge's FULL pass on the official set against Falcon
- `final-prefopt-vs-falcon`, source `judge-qwen3-4b-2507-3605803b` (the pre-registered automatic
  judge, GGUF Q4_K_M on CPU, each pair in both orders, strict agreement), all 200 pairs
  (`N/study/proxy/reports/ab/final-prefopt-vs-falcon--judge-qwen3-4b-2507-3605803b.{md,json}`):
  **12 / 61 / 127**; score 0.378 [0.313, 0.446]; decisive 0.164 [0.097, 0.266]; sign test
  p < 0.0001; the two orders agree on 60.5% of pairs (a disagreement is a tie); mean preference
  for the first-shown reply -0.134 (it prefers the SECOND reply); format mass 0.983; 1 pair
  clipped. Categories: chat 3/10/37, code 3/4/23 (p 1.0), instruct 4/17/19, math 0/12/18,
  writing 2/18/30; single-turn 10/45/93, two-turn 2/16/34. Length buckets (ours shorter /
  similar / longer): 11/14/68 (0.484, p 0.69), 1/25/46, 0/22/13.
- The pass was made in three places with the same model file and code: pairs 1 to 113 on the TPU
  host's idle cores (20:1x to 20:40), 114 to 142 on a CPU runtime (20:41 to 21:21, two thread
  settings), 143 to 200 on the same runtime after a restart (21:22 to 22:11).
- Agreement of the two instruments on that set, prompt by prompt (rows: Opus readers' outcome for
  us; columns: the registered judge's outcome; win / tie / loss): win 8 / 24 / 3; tie 3 / 46 / 8;
  loss 1 / 57 / 50. On the 62 prompts where both are decisive they agree on 58 (93.5%); the
  registered judge calls a tie on 127 prompts (63.5%), the readers on 57 (28.5%). On the 189
  prompts never looked at during the run the registered judge gives 11 / 57 / 121, score 0.378.

## CORRECTIONS (22:25 UTC) -- these SUPERSEDE earlier statements in this file and in the paper
An independent audit (`N/audit_19_10.md`: every blind-test count, category, p-value and the TPU
ledger were recomputed from the raw records and hold) found the following errors in my notes
above. Wherever the paper says the same, correct it.
1. **Why the proxy rows' held-out loss rose (section "Added 20:32", last bullet but one) was
   WRONG.** User turns kept loss weight 1 until the assistant-only "tail" stage began at **step
   48,190 (p = 0.92)** (`train_log.jsonl`: event `stage`, phase `tail`, `assistant_only: true`;
   `bes/config.py` `assistant_only_from = 0.92`). During the chat-share ramp (0.25 at p = 0.67 to
   0.70 at p = 0.92) the proxy loss FELL: 2.5430 (36,422) -> 2.4614 (46,178). It rose from the
   switch: 2.4683 (47,260) -> 2.6651 (48,342) -> 3.0764 (49,424) -> 3.1946 -> 3.2241 -> 3.2580.
   The total has its minimum 2.0840 at step 47,260 (not 46,178) and ends at 2.1422; the rise of
   the total comes from the 12 proxy rows. Web falls to the end (2.6847 -> 2.6690 from 47,260);
   the assistant-turn loss falls to the end (1.2475 -> 1.1983); the chat rows as a whole (user
   turns included) are flat after 48,342 (1.3612, 1.3671, 1.3684, 1.3638, 1.3668).
   Held-out rows (total / web / chat / assistant / proxy): 46,178: 2.0914 2.6867 1.3790 1.2651
   2.4614; 47,260: 2.0840 2.6847 1.3630 1.2475 2.4683; 48,342: 2.0984 2.6833 1.3612 1.2319 2.6651.
2. **"The constraint lead over Falcon is the selector's doing" was UNSUPPORTED.** With one
   particle (nothing to select) our model already meets all constraints on 0.771 (37 of 48)
   against released Falcon's 0.688 (33 of 48); K = 6 adds one prompt (38 of 48 = 0.792). What
   holds: the same selector over 6 samples lifts Falcon from 0.688 to 0.875 (33 -> 42 of 48), so
   under matched K and selector Falcon leads on this metric too.
3. **The readers were not blind to WHICH TWO SYSTEMS a set compared.** Every chunk file of the
   readings r1/r2 begins with "# Blind pairs 1-15 of 200 (set final-prefopt-vs-falcon)" (the pair
   set's name) and sits in a directory of that name; which side was which stayed hidden (a coin
   per pair). The instructions' "two unnamed systems" is therefore not true of those readings.
   Mitigations on record: (a) the registered judge never sees these names and agrees in
   direction (below); (b) a replication of the two official sets with neutral titles and
   directory names (`N/final_eval/neutral/x1|x2/a|b`, rating source `judge-opus-neutral`) is being
   read by 28 further Opus agents [TBD: result]. `scripts/blind_eval.py` (now in the repository)
   writes neutral titles by default; `--title-with-set` reproduces the files of 24 November.
4. Rounding: the K-matched Falcon control's score is 0.3075 exactly: print **0.308** (not
   0.307). The selector-matched control's interval is [0.225, **0.349**] (upper bound 0.348539).
   K = 1 degenerate replies are 17 of 252 = **6.7%**. The official score against Falcon on the
   189 untouched prompts is 60/189 = **0.317**. Convention for scores that fall exactly on a
   half at the third decimal: round half to even (0.7525 -> 0.752, 0.2825 -> 0.282, 0.5625 ->
   0.562, 0.3175 -> 0.318, 0.7075 -> 0.708, 0.3075 -> 0.308, 0.3775 -> 0.378), and say so once.
5. "both_bad in 44.5% of pairs" for `final-prefopt-vs-trained` is PER READING (89 of 200 in each);
   both readings say both_bad on 83 prompts (41.5%), at least one on 95 (47.5%).
6. End of training: the recorded end is **18:33:54** UTC (18:33:49 was the planned deadline). The
   run ended 20:02:22, deadline 20:31:49.
7. The sample stage: 17 prompts; "28 pairs" was the stage's upper bound; preference optimisation
   received **21** sampled pairs and used 20.
8. Preference optimisation helped blind preference only (59 / 34 / 107, p = 0.0124). The exact
   scores of the same replies did not improve: all constraints met 0.854 -> 0.792 (41 -> 38 of
   48), degenerate replies 1.2% -> 2.0%, maths 0.200 both.
9. The endgame incident, from the loop logs: one outage of the local loops of about 18 to 20
   minutes (about 19:11 to 19:31); the loops then ran until about 20:05; the dev container
   rebooted at 20:12:56 and the loops were back at 20:13 to 20:16 (a gap of 8 to 11 minutes). My
   own inaction after the two worker restarts lasted about 23 and 41 minutes; the loops' outage
   did not (the earlier "22 and 41 minutes" was wrong). The budget guard has no log.
10. Judge speed on the CPU runtimes (AMD EPYC 7B12, 4 cores, 8 threads: `N/peek/lscpu_*.txt`),
    two 4-thread streams: Falcon set 55.2 s per pair over 36 pairs (63.8 s on 23, 39.9 s on 13);
    MobileLLM set about 106 s per pair over 19 pairs (91.9 s on 16; 177.7 s on 3). 8 threads
    alone: 144.4 s per pair (n = 6, not 137.6 on 5). MobileLLM replies over the judge's 4,000
    characters: 94 of 252 (37.3%), in 88 of 200 pairs (NOT "most").
11. The fourth CPU runtime: the refusal says "too many active sessions, or a temporary usage or
    capacity limit"; a limit of three is my inference, not a record.
12. MobileLLM-R1.5's 62.7% degenerate is of REPLIES (158 of 252); 139 of 200 prompts (69.5%) have
    at least one degenerate reply.
13. The readers ran as `claude-opus-5-5` (the model field of a reader's transcript).

## Added 22:28 UTC: the neutral-title replication of the official set against Falcon
- 14 further Opus 5.5 readers per reading (28 agents for the two readings; the same instructions;
  chunk files identical to the first readings' except for the first line, which no longer names
  the pair set, in directories `N/final_eval/neutral/x1/a` (as published) and `.../x1/b` (sides
  exchanged); the mapping x1 = `final-prefopt-vs-falcon` was kept outside the directories the
  readers were given). Rating source `judge-opus-neutral`
  (`N/study/proxy/reports/ab/final-prefopt-vs-falcon--judge-opus-neutral.{md,json}`):
  **33 / 109 / 58**; score 0.310 [0.250, 0.377]; decisive 0.232 [0.171, 0.308]; p < 0.0001;
  orders agree 92.5%; first-position preference +0.010. Categories: chat 9/24/17, code 9/13/8
  (p 0.52), instruct 8/21/11, math 4/18/8, writing 3/33/14; single-turn 25/80/43, two-turn
  8/29/15.
- Against the first pair of readings (named titles: 35 / 108 / 57, 0.318): the same outcome on
  191 of 200 prompts; cross-table (rows first readings, columns neutral readings; win / tie /
  loss): win 32 / 3 / 0; tie 1 / 53 / 3; loss 0 / 2 / 106. On the 138 prompts where both are
  decisive they agree on all 138. Per-prompt score higher with named titles on 6 prompts, lower
  on 3, mean difference +0.0075, sign test p = 0.51. Reading: knowing which two systems were
  compared did not move the result; the replication is also a test-retest of the instrument
  (four independent readings per pair in all): 95.5% of prompts get the same outcome.
- Neutral readings against the registered judge: both decisive on 62 prompts, agreeing on 58.

## Added 22:31 UTC: the neutral-title replication of the official set against MobileLLM-R1.5
- Same procedure as for the Falcon set (directories `N/final_eval/neutral/x2/a|b`; source
  `judge-opus-neutral`; `N/study/proxy/reports/ab/final-prefopt-vs-mobilellm--judge-opus-neutral.{md,json}`):
  **112 / 15 / 73**; score 0.7425 (0.742 with halves rounded to even) [0.678, 0.798]; decisive
  0.882 [0.814, 0.927]; p < 0.0001; orders agree 96.0%; first-position preference -0.010.
  Categories: chat 28/1/21, code 16/2/12, instruct 28/0/12, math 6/12/12 (p 0.24), writing
  34/0/16; single-turn 80/11/57, two-turn 32/4/16.
- Against the first pair of readings (named titles: 116 / 15 / 69, 0.752): the same outcome on
  190 of 200 prompts; cross-table (rows first readings, columns neutral; win / tie / loss): win
  109 / 7 / 0; tie 3 / 66 / 0; loss 0 / 0 / 15. Both decisive on 124 prompts, agreeing on all 124.
  Per-prompt score higher with named titles on 7 prompts, lower on 3, mean difference +0.010,
  sign test p = 0.34.
- Both official sets together: 400 prompt-level outcomes, 381 the same in the replication
  (95.2%); where both pairs of readings are decisive (262) they never disagree. The title leak
  (correction 3) did not move either result beyond one point of score.

## Added 22:38 UTC: the registered judge's FULL pass on the official set against MobileLLM-R1.5
- `final-prefopt-vs-mobilellm`, source `judge-qwen3-4b-2507-3605803b`, all 200 pairs
  (`N/study/proxy/reports/ab/final-prefopt-vs-mobilellm--judge-qwen3-4b-2507-3605803b.{md,json}`):
  **59 / 5 / 136**; score 0.635 [0.566, 0.699]; decisive 0.922 [0.830, 0.966]; sign test
  p < 0.0001; the two orders agree on 71.5% of pairs; first-position preference -0.035; format
  mass 0.995; 88 pairs clipped (a reply longer than 4,000 characters). Categories: chat 12/0/38,
  code 13/0/17, instruct 17/0/23, math 5/4/21 (p 1.0), writing 12/1/37; single-turn 45/5/98,
  two-turn 14/0/38. Length buckets (ours shorter / similar / longer): 49/1/90, 4/3/21, 6/1/25.
  On the 189 untouched prompts: 56 / 5 / 128, score 0.635.
- Against the Opus readers' first readings, prompt by prompt (rows readers, columns registered
  judge; win / tie / loss): win 51 / 64 / 1; tie 8 / 61 / 0; loss 0 / 11 / 4. Both decisive on 56
  prompts, agreeing on 55 (98.2%). The registered judge calls a tie on 136 prompts (68.0%), the
  readers on 69 (34.5%). Against the neutral-title readings: both decisive on 56 prompts,
  agreeing on 55.
- How the pass was made: pairs 1 to 73 on the TPU host's idle cores (15.1 s per pair), pairs 74
  to 128 by the standard `judge` command going forwards on CPU runtimes, and five index ranges
  ([125,128), [128,136), [136,146), [146,173), [173,200)) by `judge_shard.py` streams on two CPU
  runtimes; 3 pairs (indices 125 to 127) were judged twice (forward stream and a shard) and the
  forward stream's record is kept (`judge_shard_merge.py`: the main file first). The 127 pairs
  judged on CPU runtimes took 105.2 s each on average (median 105.6 s). Finished 22:33 UTC.
- So the registered instrument, complete on both official sets: against Falcon 12 / 61 / 127
  (0.378), against MobileLLM-R1.5 59 / 5 / 136 (0.635). Same directions as the readers (0.318 and
  0.752), closer to 0.5 because it ties about two thirds of the prompts.
- MobileLLM's control arms: from 22:36 three CPU runtimes generate disjoint slices of the prompt
  order ([0,100) continuing the first run's cache, [100,136), [136,172)), each with both select
  rules in turn (`controls_slice_driver.py ... both SLICE`), and this container generates from
  the end backwards (3 cores). [TBD: results after midnight.]

## Added 00:20 UTC (25 Nov): the user's two instructions; MobileLLM's K-matched control arm generated
- At 00:07 UTC the user asked (a) that the weights before preference optimisation be kept as well
  and (b) that no Dynamic Workflows or the like (sub-agents) be used from then on.
  (a) `release/main16-v9-before-dpo/` (commits `9db8cff`..`5f6884d`): the export of the last
  training checkpoint `ckpt-000022-step-000051812` made at 18:37:02 UTC for the "before" test
  replies (packed on the runtime at 18:37; 12 parts, 196,352,000 bytes, whole-tar SHA-256
  `ba891c13…`); `weights16.npz` SHA-256
  `634230225867f197ed75593680c6dce3a3828e3d404787573559f2a510a9073a`; 16-bit check float32 2.32877,
  16-bit 2.32881 nats on 43,138 tokens (|delta| 4.29e-05); 98,036,104 stored, 19,631,240 read;
  rebuilt from the committed parts and verified with `scripts/export.py --verify`.
  (b) From 00:07 no new agent is launched. Consequence: MobileLLM's two K = 6 control arms get no
  Opus reading; they are judged by the registered automatic judge on the CPU runtimes and by the
  exact scorers. (All other test pair sets had been read before the instruction; the last
  translator of the Chinese version, launched at 23:50, finished its part afterwards.)
- **MobileLLM-R1.5, K = 6, kept by its own mean log-probability** (run
  `mobilellm-f680f0533368-qwen3/recommended-32f0c6d864-s0`): all 200 test prompts (252 replies,
  1,512 generations) complete at 00:12 UTC. Generated on three CPU runtimes (prompt indices 0 to
  169 of the test order) and, for indices 170 to 199, in the development container (3 cores);
  generations with the same cache key made on two Colab runtimes were identical in 48 of 48
  cases checked; the container's differ from the runtimes' for the same key (another CPU), so
  which machine made a sample is not neutral there, only irrelevant to its validity.
  Exact scores (test): maths correct 0.400 (12 of 30; released preset 0.400), all constraints met
  0.1875 (9 of 48; released 0.1667 = 8 of 48), single constraints 0.2143 of 56 (released 0.1786),
  degenerate replies 25.4% (64 of 252; released 62.7%: cut at the token limit 19.8% against
  59.5%, repeated lines 6.0% against 21.4%, looping tail 3.6% against 5.6%), mean words 238.6
  (released 474.1), median 108 (395). So six samples and its own log-probability more than halve
  MobileLLM-R1.5's degenerate replies; its maths and constraint scores barely move.
  Pair set `final-prefopt-vs-mobilellm-k6lp` published 00:14 UTC (200 pairs; 29 with a reply
  over 4,000 characters against 88 in the released-preset set). [TBD: the registered judge.]
- MobileLLM-R1.5, K = 6, kept by OUR selector (run `.../recommended-aca173fa2c-s0`): 181 of 200
  prompts complete at 00:12; the second turns of 19 two-turn prompts whose first reply differs
  under our selector are being generated on the three CPU runtimes. [TBD]

## Added 00:40 UTC (25 Nov): both MobileLLM-R1.5 control arms generated; exact scores; judge started
- The selector-matched arm (K = 6, kept by OUR selector; run
  `mobilellm-f680f0533368-qwen3/recommended-aca173fa2c-s0`) is complete on all 200 prompts since
  00:25 UTC (the second turns of 19 two-turn prompts needed new generations; three CPU runtimes).
  Both arms were assembled from the merged cache of all four machines with `ctl_assemble.py`
  (`runner.assemble`: no generation); the re-assembled K-matched arm has the same replies and
  selected candidates on all 200 prompts as the copy published at 00:14 (two records differ in
  latency metadata only).
- Exact scores on the test split, 200 prompts / 252 replies (in brackets the 189 untouched
  prompts: 28 maths, 44 constrained; `python -m bes.eval.proxy exact RUN --split test`, counts
  from `per_prompt`):
  | arm | maths correct | all constraints met | single constraints of 56 | degenerate replies (cut at the limit) | words: upper median, mean | repeated-4-gram rate |
  | released preset | 12 (11) | 8 (8) | 0.1786 | 158 (150); empty 2, loop 14, repeated lines 54, fence 7 | 395, 474.1 | 0.2576 |
  | K = 6, own log-probability | 12 (11) | 9 (8) | 0.2143 | 64 (50); empty 1, loop 9, repeated lines 15, fence 5 | 108, 238.6 | 0.1676 |
  | K = 6, our selector | 11 (10) | 8 (7) | 0.2143 | 51 (47); empty 0, loop 4, repeated lines 7, fence 2 | 105, 217.6 | 0.1355 |
  Degenerate shares: 62.7%, 25.4%, 20.2%. Prompts with at least one degenerate reply: 139, 60, 48.
  Reading: six samples take away most of MobileLLM-R1.5's degenerate replies; our selector does
  not raise its constraint score (8 of 48, as released), unlike Falcon-H1-Tiny's (33 -> 42 of 48).
- Pair sets `final-prefopt-vs-mobilellm-k6lp` (00:14) and `final-prefopt-vs-mobilellm-k6ds`
  (00:27): 200 pairs each, 52 two-turn; 29 and 28 pairs contain a reply over 4,000 characters
  (88 in the released-preset set); mean length of both replies per pair 2,689 and 2,559
  characters (4,451 released; 2,390 in the Falcon set).
- The registered judge on these two sets: six `judge_shard.py` streams (two per CPU runtime,
  4 threads each; index ranges [0,67), [67,134), [134,200) of each set), started 00:31 UTC.
  No Opus reading of these two sets (the user's instruction of 00:07). (Result: the section added below.)

## Added 02:00 UTC (25 Nov): the registered judge's FULL pass on MobileLLM-R1.5's two control arms
- Source `judge-qwen3-4b-2507-3605803b`, both orders, strict agreement; 200 records in each rating file, each carrying its
  pair set's commitment; streams started 00:28:25 UTC, both sets complete at 01:54 UTC
  (9 shard files on three CPU runtimes: six forward streams over [0,67), [67,134), [134,200)
  of each set, then from 01:39 three reverse helpers on the cores that came free); 71.8 s per pair
  on average over the 400 kept records; 24 pairs judged by two streams (a helper on another
  runtime does not see that runtime's records): the two records give the same verdict on all
  24 (identical probabilities in both orders on 19); the record merged first was kept
  (13 from a forward stream, 11 from a helper). The three CPU runtimes were given back at
  01:56 UTC (the streams still running would only have re-judged rated pairs). No Opus reading of
  these two sets.
- K = 6 kept by its own log-probability (`final-prefopt-vs-mobilellm-k6lp`): **39 / 8 / 153**, score
  0.578 [0.508, 0.644], decisive win rate 0.830 [0.699, 0.911], sign test p < 0.0001; orders agree
  71.0%, first-position preference −0.054, 29 pairs clipped; categories
  chat 10 / 2 / 38, code 8 / 3 / 19, instruct 10 / 0 / 30, math 6 / 3 / 21, writing 5 / 0 / 45; 189 prompts 37 / 8 / 144, score 0.577 [0.505, 0.645], p < 0.0001.
- K = 6 kept by our selector (`final-prefopt-vs-mobilellm-k6ds`): **31 / 8 / 161**, score 0.558 [0.488, 0.625],
  decisive win rate 0.795 [0.645, 0.892], sign test p = 0.0003; orders agree 68.5%,
  first-position preference −0.054, 28 pairs clipped; categories
  chat 6 / 2 / 42, code 9 / 2 / 19, instruct 6 / 0 / 34, math 4 / 4 / 22, writing 6 / 0 / 44; 189 prompts 29 / 7 / 153, score 0.558 [0.487, 0.627], p = 0.0003.
- The same judge on the released preset: 59 / 5 / 136, score 0.635 [0.566, 0.699] (88 pairs clipped).
  Paired, per prompt, our score against the released preset minus our score against the arm
  (`mob_ctl_results.py`; the same test as `python -m bes.eval.proxy compare`): own log-probability
  33 higher, 9 lower, 158 same, mean +0.0575, sign test p = 0.0003 (189 prompts: 32; 9;
  148, p = 0.0004); our selector 38, 8, 154, mean +0.0775, p < 0.0001 (189: 37; 8; 144,
  p < 0.0001).
- Reading: given six samples per reply MobileLLM-R1.5 degenerates far less (64 and 51 of 252 replies
  against 158) and the registered judge still decides more pairs for our model, by less: part of
  the lead over MobileLLM-R1.5 is a lead over its single sample. These rows have no Opus reading,
  so they compare with the judge's 59 / 5 / 136, not with the readers' 116 / 15 / 69.
- The development container restarted at 01:33 UTC (scratch directories and credentials kept; the
  keep-awake task restarted at 01:34); the judge's streams on the runtimes were not touched.
- The paper (English and Chinese) is patched by `patch_mobctl.py` from `mob_ctl_results.json`,
  `mob_ctl_exact.json` and `judge_pass_facts.json` (in `evidence/N/peek/`); only the author list
  stays [TBD].
