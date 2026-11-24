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
