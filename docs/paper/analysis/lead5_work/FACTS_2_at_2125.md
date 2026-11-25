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
