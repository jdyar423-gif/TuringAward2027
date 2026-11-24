"""Build the back matter of the third draft from the second draft's (back_v2.md) and the blocks moved out
of the second draft's main text (old_blocks.json, moved_fig_waterfall.md). Every replacement is asserted
to match exactly once. Writes back_v3.md."""
import json

B = json.load(open('old_blocks.json'))
t = open('back_v2.md').read()
wf = open('moved_fig_waterfall.md').read()


def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, (n, old[:100])
    t = t.replace(old, new)


# ---------------- Appendix A: the strict reading of v9 (moved from Section 4.2)
rep("""so neither
configuration meets the strict reading at 2,048 positions.""", """so neither
configuration meets the strict reading at 2,048 positions; `v9` read 21.2 / 23.8 / 24.6 million
(minimum / median / maximum over 40 documents) at 1,024 positions and 22.4 / 26.7 / 28.2 million at
2,048 (`docs/ACCOUNTING.md`, `docs/V9.md` §10). The audit's forward pass also logs the distinct numbers
read for every generated token.""")

# ---------------- Appendix C
rep("""the chat share of the mixture (0.25; 0.7 from p = 0.92), loss on
assistant tokens only from p = 0.92, and `final`'s library shares are all functions of p.""", """the chat share of the mixture (0.25 until p = 0.67, then rising linearly
to 0.70 at p = 0.92, then 0.70; `chat_frac_at` in `bes/train/schedule.py`), loss on assistant tokens only
from p = 0.92, and `final`'s library shares are all functions of p. The ramp shows in the loader:
base-text rows were 75% of the rows of the loader block of step 36,850 (p = 0.670) and 38.4% in that of
step 46,350 (p = 0.879), where the trainer logged a chat share of 0.627. By itself it lowers the loss of a
training step and the held-out chat loss of any configuration.""")
rep("""and the learning-rate factor rose from 0.783 to 0.803.""",
    """and the learning-rate factor rose from 0.783 to 0.803 (Figure C1d, line 4).""")
rep("""before both had read them. Together they left the third runtime's TPU idle for 48 min 19 s before
training resumed (commits `e096dff`, `77305dc`).""", """before both had read them. Together they left the third runtime's TPU idle for 48 min 19 s before
training resumed (commits `e096dff`, `77305dc`; Figure C2).""")
rep("""| 3 | from 24 Nov 01:05:55 | 39,245 at 12:00:00 | training resumed at step 5,075 at 01:54:14; in progress |""",
    """| 3 | from 24 Nov 01:05:55 | 55,290 at 16:27:24 (the trainer's ledger at the cut-off) | training resumed at step 5,075 at 01:54:14; in progress |""")
rep("""| – | [TBD: final step, at 18:34 UTC] | [TBD: final token count, at 18:34 UTC] | 18:33:49 | end of training |""",
    """| P1 | 36,867 | 9.66 × 10⁹ | no stop (stage event 12:47:51) | rows of 2,048 tokens; the chat share starts to rise (schedule) |
| – | [TBD: final step, at 18:34 UTC; projected about 51,800] | [TBD: final token count, at 18:34 UTC; projected about 1.358 × 10¹⁰] | 18:33:49 | end of training |""")
fig_c2 = B['fig_throughput'].replace('![Figure 4](figures/F2_throughput.svg)', '![Figure C2](figures/F2_throughput.svg)')
fig_c2 = fig_c2.replace('**Figure 4.**', '**Figure C2.**')
fig_c2 = fig_c2.replace('(Table 2, Figure 3)', '(Table 2, Figure F2)')
old_cut = """Records to step 37,650 (24 November, 13:05:57 UTC), the newest record of the
paper's live numbers."""
assert fig_c2.count(old_cut) == 1, fig_c2
fig_c2 = fig_c2.replace(old_cut, """Records to step 46,350 (24 November, 16:27:21 UTC), the newest record of the
paper's live numbers.""")
rep("""Averaged over its 32,320.8 training seconds, the trajectory up to the graft trained at 241,065
tokens/s (7,791,443,968 tokens; trainer's ledger).
""", """Averaged over its 32,320.8 training seconds, the trajectory up to the graft trained at 241,065
tokens/s (7,791,443,968 tokens; trainer's ledger). At the newest record (16:27:24) the ledger holds
46,351 steps, 12,150,636,544 tokens and 55,310 training seconds; the TPU had been held for 70,835 s
(19.68 h), 78% of it in training steps of the kept trajectory.

""" + fig_c2)
rep("""**C.3 The training curve.** Table C3 summarises the step losses logged every 50 steps (each the mean
over the step's 262,144 tokens) in windows of progress;""", """**C.3 The training curve.** Table C3 summarises the step losses logged every 50 steps (each the mean
over the step's 262,144 tokens) in windows of progress (the whole curve is in Figure C1);""")
rep("""After the graft the 93 logged steps to 34,350 average 2.4675
(sd 0.0399); the stream changed at the graft, so the two numbers are not comparable.""", """After the graft the 93 logged steps to 34,350 average 2.4675
(sd 0.0399); the stream changed at the graft, so the two numbers are not comparable. The 190 logged
steps at 2,048 tokens (36,900–46,350) fall from 2.404 (mean of the first 5) to 1.954 (the last 5) while
the chat share of the stream rises from 0.25 to 0.63 and the learning-rate factor falls from 0.40 to
0.21: their level is not comparable with earlier ones either.""")
rep("""tokens), logged every 50th step: 594 records of `final` (steps 50–29,700) and 159 of `v9` (steps
29,750–37,650; records to 13:05:57 UTC on 24 November, the newest record of the paper's live numbers).""",
    """tokens), logged every 50th step: 594 records of `final` (steps 50–29,700) and 333 of `v9` (steps
29,750–46,350; records to 16:27:21 UTC on 24 November, the newest record of the paper's live numbers).""")
rep("""least 15 records (the 16 records at 2,048 tokens are too few for a line).""", """least 15 records.""")
rep("""as logged. The loss mixes 22 sources whose shares change with p and the data rules, and the stream
changed at the graft: levels on either side of a restart do not compare configurations.""", """as logged. The loss mixes 22 sources whose shares change with p and the data rules; the stream
changed at the graft, and from P1 its chat share rises linearly (Appendix C.1), which by itself lowers the
loss of a step: levels on either side of a restart or of P1 do not compare configurations.""")
rep("""weights against float32 on held-out rows. Our replies to the test prompts are then generated on the
runtime's host cores (`scripts/gen_test_replies.py`; about 41 minutes on 42 workers, measured on the
development prompts). [TBD: what each stage did, from the stage ledger at about 20:32 UTC.]""",
    """weights against float32 on held-out rows. Our replies to the 200 test prompts are generated on the
runtime's host cores next to these stages (`scripts/gen_test_replies.py` through `endgame_testgen.sh`, 40
workers): from the trained weights at about 18:36 and, if preference optimisation keeps its weights, again
at about 19:40; the official replies are those whose weights and decoding equal the run's own export
(`cmp_exports.py`), else they are generated again on a CPU runtime after the TPU is released
(`docs/CHANGES.md` §19.9; `N/OPERATIONS.md`). [TBD: what each stage did, from the stage ledger at about
20:32 UTC; Appendix M, R7–R10.]""")

# ---------------- Appendix D
rep("""refusal or deflection (Table 8). [TBD: the share""", """refusal or deflection (Table 9). [TBD: the share""")
rep("""after the
test replies are generated.]""", """after the
test replies are generated; Appendix M, R4.]""")
rep("""actually skipped for either rule from step 25,280 to step 37,650 (13:05:57 UTC on 24 November, the
newest record of the paper's live numbers; `final` and `v9` together), with the number of reads.""",
    """actually skipped for either rule from step 25,280 to step 46,350 (16:27:21 UTC on 24 November, the
newest record of the paper's live numbers; `final` and `v9` together), with the number of reads.""")

# ---------------- Appendix E
rep("""[TBD: decoding
throughput of the final export, measured during the test-reply generation.]""", """[TBD: decoding
throughput of the final export, measured during the test-reply generation; Appendix M, R12.]""")

# ---------------- Appendix F
rep("""## Appendix F. On-chip measurements in full

""", """## Appendix F. On-chip measurements in full

**F.1 The rewrites of the dense core.** One 768 × 768 product took 0.08 ms forward and 0.23 ms with its
recomputation and backward pass (`docs/TPU_NOTES.md`).

""")
fig_f1 = B['fig_kernels'].replace('![Figure 2](figures/F3_kernels.svg)', '![Figure F1](figures/F3_kernels.svg)')
fig_f1 = fig_f1.replace('**Figure 2.**', '**Figure F1.**')
eq = B['equality'].replace('**Equality, stated honestly.**', '**F.2 Equality of the adopted kernel on the chip.**')
rep("""the kernel (`wf/kernel.md`).

**Table F3.""", """the kernel (`wf/kernel.md`).

""" + eq + fig_f1 + """**F.3 Whole steps and the `v9` switches.** `flat` also accumulates the table gradients in the `[V, W]`
shape; `rows` also skips one recomputation of a normalisation; `tokn`'s float32 sums run in another order.
The cost model attributes part of `v9`'s extra cycles to relayouts of the table gradients and almost
nothing to the attention gain table (`wf/v9_speed.md`). On runtime 1 `docs/TPU_NOTES.md` summarises the
first rates as `final` 112–117 thousand and `v9` 88–92 thousand tokens/s; on disk are the start-up
benchmarks 111,893 (`final`) and 88,146 (`v9`) and `final`'s two monitor readings 115,394 and 116,718.

**Table F3.""")
fig_f2 = wf.replace('![Figure 2](figures/F9_waterfall.svg)', '![Figure F2](figures/F9_waterfall.svg)').replace('**Figure 2.**', '**Figure F2.**')
rep("""pieces but not itself a measurement (`wf/speed_loops.md`).

**Table F4.""", """pieces but not itself a measurement (`wf/speed_loops.md`).

""" + fig_f2 + """**F.4 What did not help.** Thirteen changes were measured on the chip and not adopted (Table F4). Two
were faster by less than the adoption bar or the spread (one loss chunk per micro-batch, 1.2%; the
hand-written cross-entropy backward, 0.6% at 2,048 tokens and 1.2% slower at 1,024), and two
contradicted our own written expectation (keeping the kernel's outputs through recomputation; raising
the compiler's scoped on-chip memory limit).

**Table F4.""")
flop = B['flop'].replace('### 5.6 How far from the arithmetic\n\n', '**F.6 How far from the arithmetic.** ')
t5 = B['table5']
t5 = t5.replace("""Differences are harder than totals. For every change with both a compile-only prediction and a chip
measurement, Table 5 compares the two:""", """**F.7 The cost model against the chip, by change.** For every change with both a compile-only
prediction and a chip measurement, Table F5 compares the two:""")
t5 = t5.replace('**Table 5. Compiler cost model against the chip, by change**', '**Table F5. Compiler cost model against the chip, by change**')
t5 = t5.replace("chip files as in Table 3 and `docs/CHANGES.md` §19.5).", "chip files as in Table 3 and `docs/CHANGES.md` §19.5; Figure 4b).")
assert 'Table F5' in t5 and 'F.7' in t5
rep("""update counting the update (`wf/v9_speed_work/hlo_bytes.py`).
""", """update counting the update (`wf/v9_speed_work/hlo_bytes.py`). The line through all six configurations
of Table 4 has a slope of 0.73 ms per GB, a marginal 1.37 TB/s.

""" + flop + """**The loop count.** In the reference implementation one loop instead of eight took 42.9 against 289 ms
per micro-batch (`docs/TPU_NOTES.md`): about 35 ms per loop and 8 ms outside the loops; after the
rewrites a block application took about 8 ms and the rest of the micro-batch 34 ms, so at equal wall clock
4 loops would have trained on about 1.65 (rewritten code) to 1.95 (reference) times the tokens of 8.

""" + t5)

# ---------------- Appendix G
rep("""## Appendix G. The graft in full

`scripts/graft_run.py SRC DST""", """## Appendix G. The graft in full

**G.1 The tool.** `scripts/graft_run.py SRC DST""")
table_g1 = B['table6'].replace("**Table 6. The graft's function check**", "**Table G1. The graft's function check**")
rep("""the checkpoint was written under another configuration hash.
""", """the checkpoint was written under another configuration hash.

**G.2 The function check and the gradients.** The check of Section 6.4 at four source checkpoints:

""" + table_g1 + B['check_defects'] + """On real held-out rows (Section 6.4) the
gradients of U reached 7.4 × 10⁻³ and 9.0 × 10⁻³ (largest entry; web and chat rows) and those of G
1.0 × 10⁻³ and 1.2 × 10⁻³ (`lead2_work/graft_grad.json`). For a constant gradient, Adam with the carried
update count makes the new leaves' update 0.894 times a fresh run's at the first step, 1.44 times at the
seventh (the maximum) and 1.04 times at the fiftieth (β = 0.8, 0.95; our arithmetic).

**G.3 The data stream after the graft.** """ + B['loader'].replace('of which about 5.8 × 10⁹ will in fact be\ntrained (a projection)', 'of which about 5.8 × 10⁹ will in fact be\ntrained (a projection from the newest record)') + """
**G.4 The controlled experiment in detail** (`docs/MICRO_GRAFT.md`, `wf/micro_graft.md`,
`micro_graft/6M/`; recomputed by `lead3_work/micrograft.py`). Code: graft arms in `bes/eval/microbench.py`
(`run_graft_arm`), `scripts/micro_ablate.py` (`GRAFT_ARMS`), `scripts/micro_graft_report.py` and 11 tests in
`tests/eval/test_micro_graft.py` (commit `b04478e`; results in `b1e9140`); written by one agent and
reviewed by a second before the runs (trunk equality, function preservation, carried optimiser state,
schedule and data order, unchanged hashes of the 31 E2 arms, report arithmetic). The runs used a separate
clone of the repository with the patch, on the idle host cores of the training runtime (one physical
core, two threads, per run; CPU only), 14:12:51 to 16:41:01 UTC; the main run's live rate at 2,048 tokens
was 189,054 tokens/s over 73 step records before 14:12:51 and 189,086 over 117 during. A graft is refused
before training when a base leaf is missing or reshaped in the target, when an optimiser setting or the
schedule differs, or when the graft fraction is not an evaluation fraction. The carried update count
makes the page tables' updates 1, 2, 5 and 50 after a graft 0.89, 1.15, 1.41 and 1.04 times those of a
fresh start with the same moments, as in the live graft. Training pools of 8,192 web rows (ClimbMix) and
4,096 chat rows (smol-smoltalk) of 512 tokens; the mixture draws rows 3:1, 8,784 web and 2,928 chat rows
of 11,712 (1.07 and 0.71 passes), 78.0% and 22.0% of the weighted tokens. Held-out: 32 web and 32 chat
rows from other files. The learning-rate factor decays linearly to 0.1 over the steps, as 1 − 0.9p in the
live run.

Per checkpoint (mix, mean ± sd over three seeds), `v9_pages` − `ref`: −0.0164 ± 0.0062 at 1.50 × 10⁶
tokens, −0.0329 ± 0.0091 at 3.00, −0.0494 ± 0.0100 at 4.50, −0.0519 ± 0.0107 at 6.00; at the end by
domain, web −0.0464 ± 0.0118, chat −0.0683 ± 0.0089, assistant turns −0.0689 ± 0.0107. Graft − `v9_pages`
after each graft: `graft_25` +0.0137 ± 0.0086, +0.0029 ± 0.0101, −0.0031 ± 0.0113 (3.00, 4.50, 6.00 ×
10⁶); `graft_50` +0.0333 ± 0.0085, +0.0161 ± 0.0096 (4.50, 6.00); `graft_75` +0.0396 ± 0.0103 (6.00).
Per seed at the end (mix): graft − `ref` −0.0539, −0.0551, −0.0558 (`graft_25`), −0.0386, −0.0359, −0.0327
(`graft_50`), −0.0127, −0.0120, −0.0122 (`graft_75`); `graft_50` − `v9_pages` +0.0241, +0.0054, +0.0188.
Shares recovered by domain (ratio of the mean differences; web, chat, assistant turns): `graft_25` 1.07,
1.04, 1.04; `graft_50` 0.70, 0.66, 0.67; `graft_75` 0.24, 0.24, 0.23. The function check's loss on 8
held-out web rows, seeds 0, 1, 2, identical before and after each graft: 5.0309, 5.0072, 5.0118
(`graft_25`); 4.7603, 4.7661, 4.7617 (`graft_50`); 4.6126, 4.6162, 4.6532 (`graft_75`). A graft's
difference from `ref` varies little between seeds (sd 0.0004–0.0030), because the two share their trunk;
arms that never shared a state differ by about 0.01 between seeds. The restart arms
(`v9_pages` from scratch with a budget of 3,000,000 tokens and its own schedule, seeds 0–2; 732 steps,
2,998,272 tokens; the same host, 16:42:47–17:29:56 UTC; `micro_graft/3M/results.tar.gz`, unpacked into
`lead3_work/micro3M/`) end at 4.3891, 4.3885 and 4.3917 (mix; web 4.6317 ± 0.0043, chat 3.6637 ± 0.0074);
minus `graft_50` at 6 × 10⁶ tokens, +0.2348, +0.2415, +0.2265; minus `ref` at 6 × 10⁶, +0.1962, +0.2055,
+0.1938 (4.7% of `ref`'s loss); minus `v9_pages` at 6 × 10⁶, +0.2504 ± 0.0074; minus `ref`'s own evaluation
at 3 × 10⁶ tokens (mid-schedule, learning-rate factor 0.55), −0.0844, −0.0835, −0.0836; minus `v9_pages`
at 3 × 10⁶ tokens inside its 6 × 10⁶ run, −0.0509 ± 0.0086, the advantage of a schedule that ends there,
which the comparisons above include. The restart's batches are the first half of the 6 × 10⁶ runs' batches
of the same seed (the row plan is a prefix-stable round robin, `microbench.row_kinds`): it reads the first
half of the data again, not the unseen second half; with about one pass over the pools in all this should
matter little, and it was not tested.

**G.5 Held-out loss of every evaluated checkpoint.**
""")
rep("""**Table G1. Held-out loss of every evaluated checkpoint** (nats per token; "web + chat" is the comparison set of 128
rows and 125,880 weighted tokens; local evaluations reproduce the runtime's values to 10⁻⁸;
`lead2_work/heldout_analysis.json`).""", """**Table G2. Held-out loss of every evaluated checkpoint** (nats per token; "web + chat" is the comparison
set of 128 rows and 125,880 weighted tokens; local evaluations reproduce the runtime's values to within
10⁻⁷; `lead2_work/heldout_analysis.json`, `lead3_work/heldout3.json`, `lead3_work/heldout46.json`; from
step 37,522 the training stream's chat share is rising, Appendix C.1).""")
rep("""| `v9` 35,294 | 9.25 × 10⁹ | 667.5 | 2.7344 | 1.5508 | 1.4463 | 2.1661 | runtime and local |
""", """| `v9` 35,294 | 9.25 × 10⁹ | 667.5 | 2.7344 | 1.5508 | 1.4463 | 2.1661 | runtime and local |
| `v9` 36,422 | 9.55 × 10⁹ | 692.6 | 2.7254 | 1.5413 | 1.4355 | 2.1569 | runtime |
| `v9` 37,522 | 9.84 × 10⁹ | 717.6 | 2.7137 | 1.5225 | 1.4167 | 2.1418 | runtime |
| `v9` 38,604 | 1.012 × 10¹⁰ | 742.6 | 2.7066 | 1.5042 | 1.3967 | 2.1293 | runtime |
| `v9` 39,686 | 1.040 × 10¹⁰ | 767.7 | 2.7035 | 1.4838 | 1.3761 | 2.1179 | runtime |
| `v9` 40,768 | 1.069 × 10¹⁰ | 792.7 | 2.6997 | 1.4627 | 1.3546 | 2.1058 | runtime |
| `v9` 41,850 | 1.097 × 10¹⁰ | 817.7 | 2.6977 | 1.4443 | 1.3339 | 2.0959 | runtime |
| `v9` 42,932 | 1.125 × 10¹⁰ | 842.8 | 2.6922 | 1.4230 | 1.3111 | 2.0828 | runtime |
| `v9` 44,014 | 1.154 × 10¹⁰ | 867.8 | 2.6910 | 1.4112 | 1.2965 | 2.0765 | runtime |
| `v9` 45,096 | 1.182 × 10¹⁰ | 892.8 | 2.6857 | 1.3942 | 1.2795 | 2.0656 | runtime |
| `v9` 46,178 | 1.211 × 10¹⁰ | 917.9 | 2.6867 | 1.3790 | 1.2651 | 2.0588 | runtime and local |
""")
rep("""| `v9` 35,294, pages off | | | 2.7849 | 1.6301 | 1.5248 | 2.2305 | local |
""", """| `v9` 35,294, pages off | | | 2.7849 | 1.6301 | 1.5248 | 2.2305 | local |
| `v9` 46,178, pages off | | | 2.8508 | 1.6523 | 1.5386 | 2.2754 | local |

**Table G3. Comparisons at step 34,166** (the second draft's Table 7 rows; web + chat;
`lead2_work/heldout_analysis.json`).

| comparison | difference (nats) | 95% interval |
|---|---|---|
| `v9` 34,166 − `final` 29,722 (same rows, 103.8 more training minutes) | −0.0499 | [−0.0541, −0.0457] (bootstrap over rows) |
| `v9` 34,166 − `final`'s line at equal training time (line in minutes / in log tokens) | −0.028 / −0.031 | ±0.055 / ±0.045 (prediction, 1 degree of freedom) |
| pages off − on; token neurons only; gains only | +0.0473; +0.0188; +0.0244 | [0.0436, 0.0511]; [0.0168, 0.0209]; [0.0221, 0.0267] |
| `v9` with pages off − `final` 29,722 | −0.0026 | [−0.0064, 0.0014] |

At 34,166 the pages carried 0.066 nats on the rarest quarter of the held-out positions (1–16 occurrences
of the input token) against 0.032 on the most frequent quarter (497–4,425), and the improvement since
the graft was the same in every quarter (−0.049 to −0.051). At 46,178 the pages carry 0.299, 0.241, 0.170
and 0.159 nats on the four quarters, from the rarest, and the improvement since the graft is −0.152,
−0.153, −0.167 and −0.199; with the pages off at 46,178 the network is worse than at the graft point by
0.0623 nats on the web rows and 0.0340 on the chat rows (`lead3_work/heldout46.json`).
""")

# ---------------- Appendix H
rep("""| the plan's throughput | 300–450 thousand tokens/s assumed; 112–117 thousand measured | planned for 16–25 × 10⁹ tokens; the run will see about 1.36 × 10¹⁰ (projection) |""",
    """| the plan's throughput | 300–450 thousand tokens/s assumed; 111,893 (start-up benchmark) and 115–117 thousand (two monitor readings) measured | planned for 16–25 × 10⁹ tokens; the run will see about 1.358 × 10¹⁰ (projection) |""")
rep("""size off by factors 0.36–7.9 (Table 5) | used to rank, never as a result |""", """size off by factors 0.36–7.9 (Table F5) | used to rank, never as a result |""")

# ---------------- Appendix J
rep("""  export); documentation commits `ba0db12`, `d9f0ce4`, `1b9d1b8`, `8a86bcb`. This draft describes
  `8a86bcb`.""", """  export); documentation commits `ba0db12`, `d9f0ce4`, `1b9d1b8`, `8a86bcb`. The controlled graft
  experiment adds `b04478e` (graft arms of the micro bench) and `b1e9140` (its results); the repository
  is at `b1e9140` for this draft.""")
rep("""- **This paper's own analyses** (`N/paper/lead2_work/`): `make_numbers.py` (live medians, ratios,
  projections, power, FLOP), `predictor_test.py` (Tables 4 and 5), `heldout_eval.py` and
  `heldout_analysis.py` (Section 6.5), `graft_grad.py` (Section 6.4), `exact_dev.py` (Table 8),
  `opp_bpb.py` (Table 8), `deflection_labels.py` (Appendix D.4).""", """- **This paper's own analyses** (`N/paper/lead2_work/`, second pass): `make_numbers.py`,
  `predictor_test.py` (Table 4, Table F5), `heldout_eval.py` and `heldout_analysis.py` (Section 6.5),
  `graft_grad.py` (Section 6.4), `exact_dev.py` and `opp_bpb.py` (Table 9), `deflection_labels.py`
  (Appendix D.4). Third pass (`N/paper/lead3_work/`): `make_numbers.py` (a copy reading the snapshot of
  16:28 UTC: live medians, ratios, projections, power, FLOP), `heldout3.py` (every runtime evaluation and
  the extrapolations), `heldout_eval.py` (a copy) and `heldout46.py` (step 46,178 with pages on and off),
  `micrograft.py` (Section 6.6), `readers.py` (Table 8), `ref_notes.py` and `ours_notes.py` (keyword
  counts over the readers' notes). The figures: `figures/make_all.sh --cutoff 46350`.""")
rep("""- **Tests.** 1,356 test functions in `tests/` at `8a86bcb` (static count of `def test_`).""",
    """- **Tests.** 1,367 test functions in `tests/` at `b1e9140` (static count of `def test_`; 1,356 at
  `8a86bcb`).""")

# ---------------- Appendix L
rep("""We report this because it is the evidence that the
numbers here were checked against their sources, and because the checks were not complete.""", """We report this because it is the evidence that the
numbers here were checked against their sources, and because the checks were not complete. A third pass
(this draft) added the controlled graft experiment, the registered judges' readings and the opponents'
reference comparison, moved the live numbers to the records of 16:27 UTC, recomputed every new number
from the raw records (`lead3_work/`), and corrected the first row of Table 2, Table 4's FLOP cell, the
definition of the live rate and the description of whole-step timings (`RESPONSE_2.md`).""")

# ---------------- Appendix M (replaced below) and N
i = t.index('## Appendix M. Results to come')
t = t[:i]
open('back_v3_pre_M.md', 'w').write(t)
print('ok', len(t.split()))
