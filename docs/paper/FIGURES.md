# Figures: how to rebuild them, captions, readings and sources

Second figures pass of 24 November 2026, 14:40–15:30 UTC, after the figure check. Numbers in the
"Reading" paragraphs are those of the rebuild at 15:17 UTC: newest snapshot of the live run created
15:11:42 UTC (43,051 steps), read **to the paper's cut-off, step 37,650 (13:05:57 UTC, p = 0.6873)**,
the newest record of the paper's own live numbers (`lead2_work/numbers.json`; 108 newer step records
left out); held-out records to step 35,294. Times are UTC. N = the session scratchpad
(`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`), S = the old
scratchpad (`/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`), R = the
repository (`/home/user/turingaward2027`, read only).

## How to rebuild (one command)

```
bash /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_all.sh
```

This reads the newest snapshot of the live run, rebuilds every CSV in `paper/data/` (`make_data.py`) and
then every figure (`F1`–`F11`, each as `.svg` with outlined text and `.png` at 200 dpi). It takes about
20 s on CPU cores 2 and 3 and keeps the printed checks of every script in
`paper/figures/_work/make_all.log`.

**The cut-off.** Every CSV, and so every figure, uses the run's records up to one fixed step, written to
`data/cutoff.csv` and printed in the figures that show live data (3, 4, 5, C1, D1). The default
(`--cutoff paper`) is the newest step of `lead2_work/numbers.json` (`v9.newest_step`), the file the
text's live numbers (Table 2, §6.3) come from; so the figures and the text use the same records, and the
rebuild checks every live rate against `numbers.json` (a `MISMATCH` line if one differs). When the lead
author refreshes `numbers.json` (and the text) from a newer log, the figures follow on the next rebuild;
then the cut-off stated in the captions of Figures 3, 4, 5, C1 and D1 (step, time, and the counts that
depend on it: Figure 5's "steps 36,900–37,650", Figure C1's "159 of `v9` … 16 records at 2,048 tokens")
must be updated from the printed checks. `--cutoff newest` uses every record of the snapshot (and prints
a `MISMATCH` for each live rate that then differs from the text), `--cutoff STEP` any step record.

After the final blind test, add the two test-split reports by name:

```
bash /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_all.sh --blind final-<tag>-vs-falcon,final-<tag>-vs-mobilellm
```

A name is the report's file name in `N/study/proxy/reports/ab/` without `.json`; a name without `--` gets
`--judge-opus` appended. Other options: `--no-extract` (rebuild from the records read last time, without
reading the snapshot), `--tables` (also rewrite `paper/tables.md` with `make_tables.py`; off by default
because `tables.md` is not this role's file). The CSVs alone, with the same options:

```
taskset -c 2,3 nice -n 5 /tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python -I -B /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_data.py
```

**Exit codes** (`make_data.py`, passed on by `make_all.sh`): 0 done; 2 a bad argument or a `--blind` name
that does not exist; 3 the snapshot is still being copied (part sizes differ from `parts.json`; a run a
minute later works); 4 the cut-off is newer than the snapshot; 1 a figure script failed. With 2, 3 and 4
nothing is written (tested: a truncated snapshot gives exit 3 and leaves every file as it was; a cut-off
of step 99,999 gives exit 4). What `make_all.sh` prints: `extracted … snapshot of <time>` names the
snapshot; `cut-off: step … at …` the cut-off; a line starting with `MISMATCH` a failed cross-check
against the paper's own analysis files (one is expected: Known discrepancies, item 1); `largest gap
between the live and the achieved median` the check of §2.2's "at most 0.1%"; a line `test-split
reports on disk that were NOT added` lists `final-*` reports that exist but were not passed with
`--blind`.

**Where it writes.** Outputs follow the scripts, not a fixed path: `paths.py` takes the paper directory
to be the parent of the directory that holds it, so a copy of `figures/` elsewhere writes its own
`data/`, `data/_src/`, figures and `_work/` and never the paper's (tested: a copy of the scripts, run
unchanged, rebuilt every CSV and figure byte for byte and left the paper's `data/`, `data/_src/` and
`figures/` unchanged). `PAPER_FIGURES_DATA` moves `data/` and the extracted records with it,
`PAPER_FIGURES_OUT` the figure files, `PAPER_SNAP_LIVE` the snapshot read (tests only). The paper's own
analyses (`lead2_work/`, `review1_work/`) are read from the copy's paper directory if it has them, else
from the paper's (read only).

**Python and matplotlib.** The interpreter is the venv's Python 3.13
(`S/venv/bin/python`). **matplotlib is not installed in the venv** (`No module named 'matplotlib'`); the
figures use a private install in `figures/_work/pylib` (matplotlib 3.11.2, Pillow, fontTools, 136 MB),
which `style.py` appends to the path. It cannot be rebuilt without the network, so it must not be
deleted; nothing was installed in this pass. A copy of `figures/` finds it at its own `_work/pylib`, else
at the paper's, or at `PAPER_FIGURES_PYLIB`. The font is Inter from `/usr/share/fonts`.

**Inputs** (all paths in `figures/paths.py`):

| input | what is read | CSV |
|---|---|---|
| `N/snap_main/` (`parts.json`, `run.tar.NNN`) | newest snapshot of the live run: train log, run log, manifests, ledger, checkpoint manifests only (seeking past weights; part sizes checked against `parts.json`) | `data/_src/live/` → `train_steps`, `run_events`, `interventions`, `throughput_stages`, `f1_loss_smoothed`, `f2_segments`, `loader_drops_live`, `ledger_leaves`, `graft_facts` |
| `N/snap_final_main16/` | the stopped `final` run (fixed; extracted once into `data/_src/final/`) | the same |
| `paper/lead2_work/numbers.json` | the cut-off (`v9.newest_step`) and the text's live rates (cross-check) | `cutoff` |
| `N/midrun_eval_1.json`, `N/heldout_v9.jsonl` | the runtime's held-out evaluations (`scripts/midrun_eval.py`; web 64, chat 64, proxy 12 rows); checkpoints after the cut-off left out | `heldout_loss`, `heldout_trend` |
| `paper/lead2_work/heldout_analysis.json` | local CPU evaluations, pages switched off (Table 7, G1); also a cross-check of Table 7's line | `heldout_loss` |
| `paper/lead2_work/predictor_test.json` | Tables 4 and 5 | `costmodel_steps`, `costmodel_differences` |
| `paper/lead2_work/graft_grad.json`; `S/graft_test.log`, `S/graft_dry_out.txt`, `paper/review1_work/graftcheck/graft_check_29722.log` | gradient check at the graft; the tool's `--check` outputs with their batch shape ("check on 2 x 256 synthetic tokens") (Table 6) | `graft_facts` |
| `data/_src/live/run.log`, `data/_src/live/run/manifest.json` | the loader's read cap before the graft (first loader line, "at most 8 reads") and after it (3.96093), the speed switches the run started with after the graft (`fast:` line), the data seed (`run.seed=1` in `overrides`), the `graft` record | `graft_facts` |
| `N/study/proxy/reports/ab/*.json` | blind comparisons: `s1-default-vs-{falcon,mobilellm}--opus-1`, `s2-default-vs-{falcon,mobilellm}--judge-opus`, and any `--blind` names | `blind_comparisons` |
| `N/wf/chip_kernels_0745.txt`, `S/v9_meas{A,B_sorted,C_onehot,D}.txt`, `S/micro_out.txt`, `S/flags_out.txt`, `S/flags_out2.txt`, `R/docs/tpu_profile/tpu_tune.v6e.txt` | chip measurements; how a whole step was timed from `R/scripts/tpu_kernels.py` and `R/scripts/tpu_micro.py` (column `timing`) | `chip_*` |
| `S/v9.log`, `S/final.log`, `S/colab_bes24h_final.log`, `R/docs/tpu_profile/final_b.run.log`, `data/_src/final/run.log` | runtime-1 benchmarks and single log lines | `throughput_log_points` |
| `S/tpu_intervals.txt` | the TPU ledger (the open runtime counted to the cut-off) | `tpu_intervals` |
| `N/openings.json`, `N/deflection_check2.json` | data-rule scans | `openings_scan`, `deflection_check2` |
| `R/bes/budget.py`, `R/bes/roles.py`, `R/bes/data/sources.py` (imported read-only) | ledger rules, source kinds | `ledger_leaves`, `openings_scan` |

## Conventions

One font (Inter, at least 6.5 pt at the printed size), one palette: blue `#2a78d6` is always
`configs/final.py` (in Figures 1 and 7: what only `final` has), orange `#eb6834` always `configs/v9.py`
(what only `v9` has), grey is context or what both share. Where no configuration is meant, violet
`#6250d6` marks the one series a figure is about (F5, F10). Every pair that carries meaning differs in
grey as well: configurations are named in text in every figure; F2's "no TPU" bands are hatched, not a
second grey; F10 separates its series by marker shape (and the two kinds of change in (b) by triangle
direction); in F1 (a) `v9`'s light dots, as light as `final`'s, are named directly. "Tokens" are token
positions of the training batches, 262,144 per optimiser step, padding included. The **live rate** of a
stretch is the median of the rates the trainer logs with its step records (one record every 50 steps);
the logged rate is 262,144 ÷ the trainer's running average of the time per step (weight 0.2 on the
newest step; `R/bes/train/train.py` lines 965 and 903), and that time is the time ledger's training time,
so the time the trainer books to checkpoint writes (ledger phase `ckpt`, `_save`) is not part of it. The
**achieved rate** is 262,144 × steps between two records ÷ the wall time between them, checkpoint writes
included. At the cut-off the two medians differ by at most 0.078% in every stage (2,048-token stage);
with every record of the 14:46 snapshot the gap there is 0.125%, so §2.2's "at most 0.1%" holds only to
the cut-off. Print widths: F1, F2, F5, F6, F7, F9, F10, F11 both columns (7.0 in); F3, F4, F8 one column
(3.4 in). The CSVs are the table view of every figure.

**Restarts and phase changes** (`data/interventions.csv`; stop and restore times from the train log,
numbered alike in F1, F2 and Table C2):

| # | step | stopped → restored | what |
|---|---|---|---|
| 1 | 5,075 | runtime 2 lost about 19:29 (23 Nov) → 01:54:14 (24 Nov), 4 restore attempts | resume on runtime 3 from the step-5,075 checkpoint |
| 2 | 6,647 | 02:26:02 → 02:28:59 | stock-refusal rule (`566517a`) |
| M1 | 8,659 | 03:09:43 → 03:15:37 | pause: micro-batch pieces timed |
| 3 | 9,332 | 03:29:17 → 03:31:24 | 4 pieces per micro-batch (`b05bfe7`) |
| M2 | 10,242 | 03:48:04 → 03:57:28 | pause: on-chip memory flags timed |
| 4 | 12,062 | 04:30:39 → 04:32:03 | training end moved to 18:33:49 (`782be85`) |
| 5 | 22,613 | 07:44:14 → 07:53:35 | one-tile kernel + ReLU² backward (`6bf9b6e`) |
| 6 | 25,280 | 08:35:05 → 08:36:31 | deflection rule (`4f4ef45`) |
| 7 | 29,722 | 09:45:34 → 09:58:45 | graft to `v9` (`9a4d652`) |
| 8 | 31,912 | 10:52:36 → 10:57:50 | `v9` speed switches (`769f240`) |
| P1 | 36,867 | no stop; stage event 12:47:51, p = 0.6700 | rows of 2,048 tokens (schedule) |
| P2 | – | to come at p = 0.92 | chat-heavy tail, loss on assistant tokens only (schedule); added automatically when the train log has its stage event before the cut-off |

## Figure numbers in paper.md

| paper.md | files | script | changed in this pass |
|---|---|---|---|
| Figure 1 | `F7_model.svg/png` | `fig7_model.py` | "[TBD]" line removed; the 15 small leaves taken out of the core box and named as in Figure 7; caption names the source of the counts |
| Figure 2 | `F3_kernels.svg/png` | `fig3_kernels.py` | invisible whiskers dropped; caption gives the ranges, the overlapping pairs and the vertical line |
| Figure 3 | `F9_waterfall.svg/png` | `fig9_waterfall.py` | cut-off: 2,048-token stage = Table 2's 188,980 (n = 16); the two monitor readings drawn apart; stage steps read from the data; caption defines the logged rate |
| Figure 4 | `F2_throughput.svg/png` | `fig2_throughput.py` | cut-off; n under every live rate; "no TPU" hatched; caption describes runtime 1's discarded runs |
| Figure 5 | `F4_v9cost.svg/png` | `fig4_v9cost.py` | cut-off; n for every live tick; caption says each bar is one timing |
| Figure 6 | `F10_costmodel.svg/png` | `fig10_costmodel.py` | (a) redrawn as a dot plot, no marker hidden; (b) triangles; errors in the key |
| Figure 7 | `F6_graft.svg/png` | `fig6_graft.py` | colours as in Figure 1; small leaves named as in Figure 1; batch shape, read cap, seed and switches read from files |
| Figure 8 | `F8_heldout.svg/png` | `fig8_heldout.py` | no change (checkpoints after the cut-off would be left out; there are none) |
| Figure 9 | `F11_blind.svg/png` | `fig11_blind.py` | no change |
| Figure C1 | `F1_loss.svg/png` | `fig1_loss.py` | cut-off; panels c and d lettered; labels moved off the curves; `v9` named in (a); n in the caption |
| Figure D1 | `F5_openings.svg/png` | `fig5_openings.py` | "<0.1" instead of "0" for small shares; live share and read count per source; cut-off |

**No sentence of the running text cites Figures 3–9 or C1** (only their own captions do; Figure 3 is
named inside Figure 4's caption). That is a defect of the text for the lead author to fix; the sentences
that should cite each figure are listed at the end.

---

## Figure 1 (`F7_model`). The model at one token position

**Script and inputs.** `fig7_model.py`; every count from `data/ledger_leaves.csv` (both checkpoints'
manifests with the read rules of `R/bes/budget.py`) and `data/ledger_final_export.csv` (`final`'s
export accounting: library and index); architecture text from `R/docs/SPEC_MODEL_TRAIN.md` 4.1–4.3,
`R/docs/V9.md` 4, `R/bes/model/model.py`, `R/bes/config.py`.

**Caption** (as in `paper.md`). BES-100M-A20M at one token position, with stored numbers / numbers read
per generated token (standard reading) for each part, read from the two checkpoints' manifests and
`final`'s export accounting (`data/ledger_leaves.csv`, `data/ledger_final_export.csv`), as in Table 1.
Gray: shared by both configurations; blue: `final` only (the written library); orange: `v9` only
(attention-output gains and private token neurons in each block). The 15 small dense leaves (306,776
numbers: norm gains, attention scalars, the sink, lane weights, the projections of the candidate tape and
of the trust features, the critic) are counted in both totals and not drawn. The looped core applies two
weight-shared blocks eight times; the output law adds the masses of up to 16 candidate next tokens to the
network's distribution and renormalises in closed form. A schematic: only the counts are data.

**Reading.** Core 14,118,912 (the attention and feed-forward matrices of both blocks), head 4,587,520,
input embedding 6,291,456 / 768, value embedding 8,388,608 / 1,024, register, candidate and trust tables
838,448 / 18,736, small dense leaves 306,776 / all; `final`'s library and index 63,768,391 / ≤ 245,223;
`v9`'s gains 12,582,912 / 1,536 and neurons 50,921,472 / 595,968 (tables 50,331,648 + projections
589,824, as in Table 1). Totals: `final` 98,300,111 stored, 19,278,959 read (worst-case retrieval); `v9`
98,036,104 stored, 19,631,240 read.

**Changed in this pass.** The last line of the image read "… the setting used in the evaluation is chosen
after training [TBD]"; the decoding line is gone (decoding is not a per-position fact, Appendix E has
it). The core box said "norms, lanes, sink, critic: 306,776 / all", inside the core; that ledger row
also holds `cand/in_proj` and `trust/proj` (49,152 each) and is not part of the core, so it is now a line
below the totals with the name Figure 7 uses (`common.small_group`, computed from the leaves). The
caption names the source of the counts.

## Figure 2 (`F3_kernels`). Attention-kernel candidates on the chip

**Script and inputs.** `fig3_kernels.py`; `data/chip_kernels_0745.csv` from `N/wf/chip_kernels_0745.txt`
(the `KERNELS` line; medians recomputed from the 30 timings agree to 10⁻³ ms, column `median_check`;
`min`, `max`, `n`).

**Caption** (as in `paper.md`). One value-and-gradient of a piece of 4,096 tokens (left 4 × 1,024, right
2 × 2,048) of `configs/final.py` on the chip, eight kernel configurations measured in one pause (24
November 07:44–07:52; `wf/chip_kernels_0745.txt`). Dots: median of 30 timings. The 30 timings of a dot
span 0.09–0.47 ms from minimum to maximum, less than the width of a dot, so the ranges are not drawn; they
overlap for the adopted configuration and the same with `ce` (both lengths) and, at 2,048 tokens, for
one-tile + save against both of these, for splash + save against one-tile and for the configured splash
kernel against splash + io: those orderings are not resolved by the measurement. Vertical line: median
of the configured splash kernel; right: speed-up against it. Blue, ringed: the configuration adopted at
step 22,613. `save`: forward kernel outputs kept through recomputation; `relu2`, `ce`: hand-written
backward passes of ReLU² and of the cross-entropy head; `io`: splash accumulators in on-chip memory.

**Reading.** The adopted configuration took 14.639 ms (×1.212) at 1,024 and 16.275 ms (×1.219) at 2,048;
equality metrics in Table F2. Ranges (min–max of 30): 0.092 to 0.475 ms. Overlapping ranges, as printed:
1,024: adopted [14.605, 14.697] and + `ce` [14.489, 14.964]; 2,048: splash [19.783, 19.95] and + io
[19.724, 19.974]; splash + save [17.872, 18.228] and one-tile [17.537, 17.934]; one-tile + save [16.335,
16.737] against adopted [16.22, 16.356] and + `ce` [16.073, 16.375]; adopted and + `ce`.

**Changed in this pass.** The caption promised whiskers for minimum and maximum, but at 0.9 in per 9 ms
every range was smaller than its dot (radius about 0.29 ms), so no whisker could be seen. Whiskers this
small cannot be drawn visibly at this size, so they are dropped and the caption states the ranges and
which orderings they leave unresolved (printed by the script). The vertical line is explained.

## Figure 3 (`F9_waterfall`). Throughput by cause

**Script and inputs.** `fig9_waterfall.py`; `data/throughput_stages.csv` (`make_data.py`: live and
achieved medians per stage from `train_steps.csv`, the step of the change that starts each stage
(`from_step`), Table 2's synthetic values from `chip_whole_steps.csv`, benchmarks and single log lines
from `throughput_log_points.csv`), `data/interventions.csv` (P1), `data/cutoff.csv`. The live rates are
checked against `lead2_work/numbers.json` on every rebuild: at the cut-off all eight are equal, the
2,048-token stage included.

**Caption** (as in `paper.md`). Training throughput by cause (Table 2), the changes in the order they
happened. Upper part, before run main16 (runtime 1, 32-row micro-batches, training runs discarded):
start-up benchmarks (hollow bars) and single log lines of training (dots; the reference row's two dots
are the two monitor readings on disk, 116,718 and 115,394 tokens/s, drawn one above the other). Lower
part, run main16: each bar spans the live rate before and after one change, and the first bar starts at
zero. The live rate is the median, over the step records (every 50th step) in the window printed under
the stage (n records, first and last step, UTC), of the rate the trainer logs with each record: 262,144
tokens ÷ its running average of the time per step (weight 0.2 on the newest step; time the trainer books
to checkpoint writes is not counted). Records to step 37,650 (24 November, 13:05:57 UTC), the newest
record of the paper's live numbers; the 2,048-token stage was still running then. Diamonds: Table 2's
synthetic or start-up-benchmark value of the same stage (`final`'s synthetic step timed with the `ce`
backward on). The change from 32 to 16 rows per micro-batch falls between the two parts (benchmark
against live rate). A bar is a before-and-after difference of medians, not the effect of one change
alone: the step from runtime 2 to runtime 3 changed no code, and the graft changed the configuration.

**Reading** (15:17 rebuild, cut-off 13:05:57). Runtime 1: benchmark 111,893 → 201,877 tokens/s (+80.4%),
monitor lines 116,718 (step 72, 04:17) and 115,394 (step 301, 04:28), and one log line after the rewrite,
204,699 (step 50, 04:48). Run main16, live: 222,031 (n = 101, steps 50–5,050) → 216,548 (−2.5%, n = 85)
→ 240,131 (+10.9%, n = 266) → 281,389 (+17.2%, n = 142) → 178,087 (−36.7%, n = 44) → 196,874 (+10.5%,
n = 99) → 188,980 at 2,048 tokens (−4.0%, n = 16, steps 36,900–37,650, 12:48–13:05), the values of
Table 2 and §6.3. Synthetic or benchmark values: 216,681; –; 246,538; 296,371; 182,919; 203,665; –.

**Changed in this pass.** The 2,048-token bar showed 189.1k (n = 59, to 13:55), Table 2 above it 188,980
(n = 16): the figures read whatever snapshot was newest. They now stop at the paper's cut-off and agree
with Table 2 and §6.3 (with every record of the 14:46 snapshot the stage would read 189,069, n = 102).
The two monitor readings, 1.3k apart, rendered as one dot; they are now drawn one above the other and
named "monitor lines" on the right. The stage labels' steps ("step 9,332" …) were typed; they are read
from `throughput_stages.csv` (`from_step`, from `interventions.csv`). The caption defines the logged
rate as the trainer computes it.

## Figure 4 (`F2_throughput`). Throughput against wall-clock time

**Script and inputs.** `fig2_throughput.py`; `data/train_steps.csv` (`interval_tok_per_s`),
`data/throughput_stages.csv` (printed numbers and n), `data/throughput_log_points.csv`,
`data/run_events.csv`, `data/interventions.csv`, `data/tpu_intervals.csv`, `data/cutoff.csv`; runtime-1
measurement windows 03:34–03:57 and 04:08–04:17 from `R/docs/CHANGES.md` 19.1.

**Caption** (as in `paper.md`). Training throughput against UTC time on the three runtimes (axis broken
where no runtime was held). Runtime 1 (23 November) holds runs before run main16, all discarded: start-up
benchmarks (diamonds) of `final` before and after the glue rewrite and of `v9` before it (orange), and
single log lines of `final`'s training (dots). Runtimes 2 and 3, run main16: lines, the rate achieved
between consecutive step records (tokens ÷ wall time, checkpoint writes included); numbers, the live
rate of each stretch in 10³ tokens/s with its number of step records n (Table 2, Figure 3); in runtime
2, the start-up benchmark (diamond) and the first log line (dot). Light gray bands: TPU held but not
training; hatched: no TPU. Tags: restore times of the restarts of Table C2; P1 (dashed): rows of 2,048
tokens from p = 0.67, without a stop. Records to step 37,650 (24 November, 13:05:57 UTC), the newest
record of the paper's live numbers. A change of level at a tag is a before-and-after difference, not the
effect of that change alone.

**Reading.** 222.0 (n = 101) → 216.5 (85) → 240.1 (266) → 281.4 (142) → 178.1 (44) → 196.9 (99) → 189.0
(16) (10³ tokens/s). Runtime 1's points: `v9` benchmark 88,146 (03:13), `final` 111,893 (03:27), monitor
lines 116,718 and 115,394, `final` after the rewrite 201,877, 201,918 and 204,006 (benchmarks, 04:36–04:42)
and 204,699 (log line, 04:48). Runtime 3 spent 48 min 19 s restoring and fixing two data errors before
training resumed. The note in runtime 3 is computed from the train log's set-up events, the TPU ledger
and `cutoff.csv`: records to step 37,650 (13:05:57); to come: training to 18:33:49, 118 min of post
stages, run deadline 20:31:49; the 24 h of TPU time end at 20:46:50.

**Changed in this pass.** The two band greys (`#f0efec` and `#e6e5e0`, 3.5 L* apart) were hard to tell
apart in print; "no TPU" is now a hatch. Every live rate carries its n. The caption said "run main16"
for a figure whose runtime 1 shows only discarded runs, a `v9` benchmark among them; it now says so. The
data stop at the cut-off and the time axis ends at 13:30.

## Figure 5 (`F4_v9cost`). Cost of the v9 step against final's

**Script and inputs.** `fig4_v9cost.py`; `data/chip_whole_steps.csv` (`N/wf/chip_kernels_0745.txt`,
`S/v9_measA.txt`, `S/v9_measB_sorted.txt`, `S/v9_measC_onehot.txt`, `S/v9_measD.txt`; column `timing`),
`data/throughput_stages.csv` (live rates with n), `data/cutoff.csv`.

**Caption** (as in `paper.md`). One accumulation step of 262,144 tokens on a synthetic batch, timed on
the chip, by configuration, table-gradient mode and piece size; ratios against `final`'s fastest step of
the same length. Each bar is one timing, the mean of a few whole steps run back to back after a warm call
(`scripts/tpu_kernels.py`, 2 by default; the count used was not recorded), so no spread can be shown;
`final` was timed at 07:44–07:52, `v9` at the switch (09:45–09:58) and with the three switches added one
by one (10:53–10:56). Black ticks: 262,144 ÷ the live rate of the run, with its number of step records
n: Table 3's windows (`final` steps 25,300–29,700; `v9` before and after the switches) and, at 2,048
tokens, steps 36,900–37,650 (to 13:05:57 UTC on 24 November, the newest record of the paper's live
numbers), where `v9` runs with the switches, which were not timed there on a synthetic batch, so only its
live tick is shown. `final` was timed with the hand-written cross-entropy backward on, which the run does
not use (0.177 ms slower per piece at 1,024 tokens).

**Reading.** Synthetic: 884.5 ms (`final`) against 1,433.1 ms (`v9`, ×1.62), with the switches 1,287.1
ms (×1.46). Live: 931.6 ms (`final`, 281,399 tokens/s, n = 89), 1,472.0 (`v9`, n = 44), 1,331.5 (with the
switches, n = 99), 1,387.2 at 2,048 tokens (n = 16).

**How a whole step was timed.** `R/scripts/tpu_kernels.py`: one call compiles, one warm call, then
`--step-calls` calls back to back, and `ms` is their mean (default 2; `wf/v9_speed.md`, `wf/v9_rows.md`
and `wf/v9_tokn.md` plan `--step-calls 2`, but the commands that produced the five output files were not
recorded). `S/micro_out.txt` (Figure 3's "4 pieces" synthetic value) comes from `R/scripts/tpu_micro.py`,
the mean of `--calls` steps (default 3). So no whole step has a spread.

**Changed in this pass.** n for every live tick (it was given for the 2,048-token tick only); the caption
says what a bar is; the 2,048-token tick stops at the cut-off (1,387.2 ms = 262,144 ÷ 188,980).

## Figure 6 (`F10_costmodel`). The compiler's cost model against the chip

**Script and inputs.** `fig10_costmodel.py`; `data/costmodel_steps.csv` and
`data/costmodel_differences.csv` (`make_data.py` from `paper/lead2_work/predictor_test.json`: the six
configurations' counts and chip times, and Table 5's predicted and measured differences). The
leave-one-out predictions are recomputed from the json's rows and checked against its errors.

**Caption** (as in `paper.md`). The compiler's estimates against the chip. (a) The six configurations
with a compiled program and a whole-step time on the chip (Table 4), one row each, sorted by the measured
step (bar; one timing each, as in Figure 5). Markers: each static count's leave-one-out prediction of the
step (a straight line fitted to the other five configurations), joined to the bar by its error; the key
gives the mean and the largest absolute error of each count. FLOP and reads per token take only two
values here and give the same predictions: for a `v9` configuration the mean of the other four `v9`
steps; for `final` the fit is undefined and no marker is drawn. (b) The seven changes with a compile-only
prediction and a chip measurement (Table 5): predicted against measured change of the step, absolute
values on logarithmic axes; triangles up: the change made the step slower, down: faster; the number after
each label: predicted ÷ measured; the sign was right for all seven. Six configurations, five of them
layout variants of one model, chosen after the measurements: the figure shows how the counts did here,
not that bytes moved predict step time in general.

**Reading.** Leave-one-out error, mean (largest): bytes 6% (11%), cycles 12% (40%, `final`), FLOP or reads
10% (21%) on the five `v9` configurations. Predictions (bytes / cycles / FLOP or reads, ms): `final`
802.4 / 531.1 / –; `v9` + `flat` + `rows` 1,324.1 / 1,331.5 / 1,530.0; + `flat` 1,395.1 / 1,458.6 / 1,502.2;
as grafted 1,424.9 / 1,513.3 / 1,495.5; 8-row pieces 1,657.4 / 1,567.4 / 1,480.9; `sorted` 1,611.7 /
1,558.5 / 1,406.6. The FLOP-or-reads predictions run against the chip: the slowest step (`sorted`,
1,789.0 ms) gets the lowest prediction, the fastest `v9` step (1,295.2 ms) the highest. Differences,
predicted ÷ measured: 1.14 (`v9` instead of `final`), 0.36 (`sorted`), 0.85 and 2.95 (8-row pieces, `v9`
and `final`), 1.72 (`flat`), 1.07 (`rows`), 7.85 (`tokn`).

**Assumptions and limits.** The cycles of (b) are the cost model's (converted at 0.521 ms per million
cycles and anchored on `final`'s measured step; `wf/v9_speed.md`, Method); `tokn` was predicted on `flat`
and measured on `flat` + `rows`. Panel (a) is the paper's predictor test; it has no prediction of its own.

**Changed in this pass.** (a) was a predicted-against-measured scatter in which markers hid one another:
the bytes point of `v9` + `flat` + `rows` (1,324.1) lay under the cycles square (1,331.5), the "+" marks of
the measured steps 3 and 4 under the bytes dots, and the label "3" stood next to configuration 2's FLOP
cross. A dot plot with one row per configuration and one line per count cannot hide a marker, needs no
numbering, and shows each error as a line. The open square meant "estimated cycles" in (a) and "faster"
in (b); (b) now uses triangles (up: slower, down: faster), and its ratios sit next to the labels.

## Figure 7 (`F6_graft`). The graft

**Script and inputs.** `fig6_graft.py`; `data/ledger_leaves.csv` (both checkpoints' manifests),
`data/ledger_final_export.csv`, `data/run_events.csv` (stop, restore, set-up: deadline, training end,
pieces per micro-batch), `data/interventions.csv`, `data/graft_facts.csv` (the three `--check` outputs of
Table 6 with their batch shape; `lead2_work/graft_grad.json`; the loss guard of the
`ckpt-000026-step-000029722` manifest; the live manifest's `graft` record and `run.seed`; from the live
`run.log`, the loader's cap before the graft, the first loader plan after it and the speed switches of
the start that continued the graft). No number is typed in the script.

**Caption** (as in `paper.md`). Stored numbers of the two configurations at step 29,722, to scale, from
both checkpoints' manifests and `final`'s export accounting (`data/ledger_leaves.csv`,
`data/ledger_final_export.csv`). Colours as in Figure 1. Gray: the 59 parameter leaves both share
(34,531,720 numbers), carried with their optimiser slots; blue: `final`'s written library and index
(63,768,391), which `v9` does not have; orange: `v9`'s five new leaves (63,504,384) at initialisation
(`gain/o` and `page/val` zero, `page/key` normal, `page/pin` and `page/pout` uniform), with fresh optimiser
slots. Right: what else was carried, what is new, and the checks (Table 6; gradients, Section 6.4), each
read from the run's logs and manifests or the check outputs (`data/graft_facts.csv`).

**Reading.** Source and grafted losses equal on 2 × 256 synthetic tokens at 26,890 (11.2808027), 28,501
(11.3348608) and, in the re-run under the live hash `f29bfeb3ebf353a8`, 29,722 (11.3726320); on 2 web and
2 chat held-out rows the gradients of all 59 shared leaves are bit-identical, those of `page/key`,
`page/pin`, `page/pout` exactly zero. p 0.5049 at the stop, 0.5112 at the restore; the loss guard's moving
average 2.478. Carried: 4 pieces per micro-batch and the kernel switches `kernel=onetile:save relu2=1`
(the run's `fast:` line at the 09:57 start; `v9`'s own switches came at step 31,912). New: data seed 1;
a source may be read at most 3.96 more times (3.96093 in the run log = 8, the cap before the graft,
× (1 − 0.504884)), over a horizon of 7.45 × 10⁹ tokens.

**Changed in this pass.** Figure 1 used grey for what both share and blue for `final`'s library, this
figure the opposite; both now use grey = shared, blue = `final` only, orange = `v9` only. The ledger row
of 306,776 numbers was "norms, lanes, sink, critic" in Figure 1 and "norms, lanes, sink, trust, critic"
here; both now print the same name, computed from the leaves. "Pieces and speed switches" read like the
`v9` switches of step 31,912; the figure now names what the run started with after the graft. "2 × 256",
"8 × (1 − p)" and "seed 1" were typed; they are read from the check logs, the run log and the manifest.

## Figure 8 (`F8_heldout`). Held-out loss across the graft

**Script and inputs.** `fig8_heldout.py`; `data/heldout_loss.csv` (`N/midrun_eval_1.json`,
`N/heldout_v9.jsonl`; local evaluations from `lead2_work/heldout_analysis.json`; checkpoints after the
cut-off left out, none so far), `data/heldout_trend.csv` (the line and band, computed by `make_data.py`
and checked against Table 7's values in `heldout_analysis.json` on every rebuild).

**Caption** (as in `paper.md`, unchanged). Held-out loss of the checkpoints around the graft on the
comparison set of Table 7 (64 web and 64 chat rows of 1,024 tokens, 125,880 weighted tokens; float32,
dense attention), against training minutes of the kept trajectory. The 12 proxy rows (11,094 tokens; they
hold user turns of the test prompts) are left out; with them the runtime evaluated 140 rows, 136,974
weighted tokens. Filled points: the runtime's evaluations (`scripts/midrun_eval.py`); rings: checkpoints
evaluated again on the CPU of the development container, which agree to 10⁻⁸; open diamonds: `v9` with
its pages (token neurons and gains) switched off at evaluation. Gray line: the least-squares line through
`final`'s three points, extended (Table 7, line in minutes); band: its 95% prediction interval with one
residual degree of freedom (t = 12.71). The arrow gives `v9`'s distance from the line at the newest
checkpoint, rounded. Up to step 35,294 every `v9` point lies inside the band: the figure does not show
that `v9` is ahead of a continued `final`, and the data stream also changed at the graft.

**Reading.** `final`'s line falls 2.09 × 10⁻⁴ nats per training minute (residual sd 1.05 × 10⁻³). `v9`
minus the line: −0.010 ± 0.025 (563.7 min), −0.015 ± 0.035 (592.5), −0.022 ± 0.045 (617.5), −0.028 ± 0.055
(642.5, step 34,166) and −0.033 ± 0.066 (667.5, step 35,294), the last two as in Table 7. With the pages
off, 2.2241 and 2.2305, above the line (2.2045 and 2.1993) and inside the band.

**Assumptions and limits.** Training minutes are rounded to 0.1 as in Table G1 (with unrounded minutes
the half-widths at 642.5 and 667.5 min are 0.0551 and 0.0654). The axis shows the band at the newest
checkpoint unless it is wider than the points' own range; later checkpoints far from the three `final`
points will let the band leave the plot.

## Figure 9 (`F11_blind`). Blind comparisons

**Script and inputs.** `fig11_blind.py`; `data/blind_comparisons.csv` (`make_data.py` from
`N/study/proxy/reports/ab/s1-default-vs-falcon--opus-1.json`, `s1-default-vs-mobilellm--opus-1.json`,
`s2-default-vs-falcon--judge-opus.json`, `s2-default-vs-mobilellm--judge-opus.json`, and the reports
passed with `--blind`). Our checkpoint's step is read from each report's `x` field and its configuration
from the step (after 29,722: `v9`); "two readings" is taken from the report's judge block (orders agree).

**Caption** (as in `paper.md`, unchanged). Blind pairwise comparisons on the first 60 development prompts
(Table 8): `final` at step 25,280 read once per pair, `v9` at step 31,912 read twice on exchanged sides (a
side wins only when both readings choose it; ties include "both bad" and readings that disagree). Bars:
pairs our reply won, ties or "both bad", and pairs the opponent won; dots and whiskers: the score
(wins + ties/2)/n with its 95% Wilson interval; vertical line: 0.5. Readers: Opus 5.5 agents, not the
pre-registered judge (Section 7.2). With 60 pairs a score's 95% interval spans about ±12 points; the two
studies differ in checkpoint, configuration and number of readings, so the change between them is not an
effect of `v9`. The test-split comparison is [TBD: after the evaluation of 24–25 November 2026].

**Reading.** Against Falcon-H1-Tiny: 2 / 15 / 43 (score 0.158 [0.087, 0.271]) and 3 / 18 / 39 (0.200
[0.118, 0.318]). Against MobileLLM-R1.5: 15 / 37 / 8 (0.558 [0.433, 0.677]) and 20 / 33 / 7 (0.608 [0.482,
0.722]).

**For the final test.** Pass the two `final-<tag>-vs-*` names to `make_all.sh --blind`; their rows appear
under each opponent after the development rows, labelled with their split (`test`) and n. Then replace
the last sentence of the caption with the test result and its n.

## Figure C1 (`F1_loss`). Training loss, restarts and schedule

**Script and inputs.** `fig1_loss.py`; `data/f1_loss_smoothed.csv` (`make_data.py`: raw step losses and
the centred average, with its effective record count, half-width and window centre per record),
`data/train_steps.csv`, `data/interventions.csv`, `data/cutoff.csv`.

**Caption** (as in `paper.md`). Training loss of run main16 against training tokens: (a) the whole run on
logarithmic tokens; (b) from 10⁹ tokens on, linear. Dots: the loss of one optimiser step (the mean over
its 262,144 tokens), logged every 50th step: 594 records of `final` (steps 50–29,700) and 159 of `v9`
(steps 29,750–37,650; records to 13:05:57 UTC on 24 November, the newest record of the paper's live
numbers). Lines: a centred average over log₁₀(tokens), computed separately within each stretch between
restarts and phase changes: each record is weighted by a Gaussian (σ = 0.03 decades, cut at 3σ) times
1/tokens, so that the window is centred on the record (to within 0.001 decades), and near the end of a
stretch the window shrinks symmetrically instead of becoming one-sided; the line is drawn where it
averages at least 15 records (the 16 records at 2,048 tokens are too few for a line). A step at a restart
would show as a break between two lines; near the ends of a stretch the line rests on fewer records and
wanders by about ±0.01 nats. Vertical lines: the restarts of Table C2 (solid; M1, M2: measurement pauses)
and the switch to rows of 2,048 tokens at p = 0.67 (P1, dashed). (c, d) The learning-rate factor and
progress p as logged. The loss mixes 22 sources whose shares change with p and the data rules, and the
stream changed at the graft: levels on either side of a restart do not compare configurations.

**Reading.** A step's loss falls from 4.858 at step 50; the average reads 2.736 at step 3,100 (where it is
first drawn), 2.690 at 3,800 (10⁹ tokens), 2.584 at 11,450, 2.511 at 22,200 and 2.493 at 29,000. Across
the graft: 2.487 at step 29,300 (last `final` point drawn) and 2.490 at 30,150 (first `v9` point); the last
21 logged `final` steps average 2.492 (sd 0.039), the first 21 `v9` steps 2.497 (sd 0.045). The 2,048-token
stretch has 16 records to the cut-off, from 2.404 (mean of the first 5) to 2.351 (last 5, to step
37,650); no line. The scatter of one step's loss around the line, where the window is full, is 0.037 nats
(sd, 91 records). Largest centre offset of a drawn point: 0.000999 decades.

**Changed in this pass.** The caption referred to panels "(c, d)", which had no letters; they are now
titled c and d. The white box of the label "4: p and the factor jump" hid the p curve between about 5.0
and 6.6 × 10⁹ tokens; the label now stands between the two curves without a box, and the label of
p = 0.67 moved below its line (above it, its box cut the factor curve). In (a) `v9`'s light dots have the
lightness of `final`'s (L* 79.5 against 78.3); the legend now says "from line 7" and the cloud is
named. The caption gives the number of records and the cut-off.

**After p = 0.92.** From p = 0.92 the trainer logs the loss of assistant tokens only (stage `tail`); those
records are left out of (a) and (b) and drawn in an extra panel (e) on their own scale, with P2 marked.
Add to the caption: "(e) From p = 0.92 the loss covers assistant tokens only and is drawn on its own scale;
it is not comparable with (a) and (b)." (Only once the cut-off is past the stage event.)

## Figure D1 (`F5_openings`). Stock openings and what the loader leaves out

**Script and inputs.** `fig5_openings.py`; `data/openings_scan.csv` (`N/openings.json`),
`data/deflection_check2.csv` (`N/deflection_check2.json`), `data/loader_drops_live.csv` (train logs, to
the cut-off), `data/cutoff.csv`. Shares are computed from the counts in the CSVs and rounded half up in
decimal (0.15% prints as 0.2).

**Caption** (as in `paper.md`). (a) Share of each source's assistant turns whose opening (first 44
tokens) matches a stock pattern, from the scan of the tokenised sources (24 November 08:20; all segments
of small sources, 40,000 sampled otherwise); "Sure …" is a control that no rule removes; a turn can match
several patterns. (b) Share of each chat source's conversations that the stock-refusal rule (light gray,
live since step 6,647) and, in addition, the deflection rule (violet, live since step 25,280) leave out,
measured with the loader's functions on 30,000 sampled conversations per source (2,260 for
smoltalk_everyday, all it has). Black ticks and the column on the right: share of reads the live loader
actually skipped for either rule from step 25,280 to step 37,650 (13:05:57 UTC on 24 November, the newest
record of the paper's live numbers; `final` and `v9` together), with the number of reads. In both panels
"<0.1" means above 0 and below 0.05%, and "0" means none.

**Reading** (cut-off 13:05:57). The live loader skipped 30.3% of 22,591 reads of the system chats (scan
29.8%), 9.5% of 42,948 of UltraChat (9.5%), 6.2% of 154,413 of Tulu 3 (6.1%), 1.2% of 110,977 of SYNTH
(1.2%); smoltalk_constraints <0.1% of 34,970 (scan: 11 of 30,000 conversations, 0.04%); smoltalk_everyday
has no reads in a logged loader block.

**Changed in this pass.** smoltalk_constraints was labelled "0" in (b) although the rule removes 11 of
30,000 of its conversations; small non-zero shares now print "<0.1" in both panels, "0" only for none.
The live ticks now carry their share and number of reads (right column). Shares are rounded from the
counts (`smoltalk_magpie_ultra`'s 45 of 30,000 printed as 0.1 from its binary float, now 0.2).

---

## Known discrepancies, for the lead author (text outside the figure blocks; not changed here)

1. **Table 4, FLOP cell.** "20% (66%)" is an artefact of `lead2_work/predictor_test.py`: with `final`
   left out, the five `v9` configurations all have 0.809 GFLOP per token, the fit is undefined (in exact
   arithmetic the sum of squares is 0), but the floating-point sum of squares is 6.2 × 10⁻³², so a
   meaningless slope gives the 66%. Like the reads column it should read "not defined without `final`";
   on the five `v9` configurations FLOP and reads give identical predictions, 10% (21%), which Figure 6
   shows. (`make_data.py` prints this as its one `MISMATCH`.)
2. **"112–117 k (monitor readings)".** Table 2's first row: the only monitor readings on disk are 116,718
   and 115,394 tokens/s (`S/colab_bes24h_final.log`, 04:17:59 and 04:28:00), both drawn in Figure 3; 112k
   is the start-up benchmark (111,893), and `R/docs/CHANGES.md` §19.1 calls 112–117k start-up benchmarks.
   The row could read "115,394 and 116,718 (two monitor readings)". The same "112–117 thousand … live"
   appears in §1 (contribution 1), §5.2 (first paragraph), §5.3 (last paragraph) and Table H1.
3. **§2.2 "the step's own rate".** The rate in a step record is 262,144 ÷ the trainer's running average of
   the time per step (weight 0.2 on the newest step; `bes/train/train.py` lines 965 and 903), measured in
   the time ledger's training time, so the time booked to checkpoint writes is left out; the Figure 3
   caption now says so. §2.2's "differs by at most 0.1%" holds to the cut-off (largest gap 0.078%), not
   for all records of the 14:46 snapshot (0.125% in the 2,048-token stage).
4. **§5.1 "we report medians".** True of single pieces (30 calls, 6 for a `v9` piece); the whole steps of
   Tables 3–5 and Figures 3, 5 and 6 are single timings, each the mean of a few back-to-back steps (count
   not recorded, 2 by default).
5. **The cut-off of the text.** Table 2, §6.3 and the figures now use the same records (to step 37,650,
   13:05:57). When the text is brought up to date after the run ends, refresh `numbers.json`, rebuild,
   and change the cut-off in the five captions named under "How to rebuild".
6. **The old Figure 6 caption's "±0.025 at 592.5 minutes"** belonged to a line through the loss of all 140
   rows (`numbers.json` `heldout.three_point`). Table 7's line (web + chat) gives ±0.035 there; Figure 8
   uses Table 7's set.
7. **Table 7's intervals** use training minutes rounded to 0.1; unrounded they are ±0.0551 and ±0.0654
   instead of ±0.0554 and ±0.0657. Immaterial; the figure follows Table 7.

## Sentences of paper.md that should cite the figures

No running text cites Figures 3–9 or C1. One place for each:

- **Figure 3 (waterfall).** §5.2: "Three kinds of change followed (Table 2; details in Appendix F)." →
  "(Table 2, Figure 3; …)"; §6.3: "Since p = 0.67 (step 36,867, 12:48 UTC) `v9` trains on 2,048-token rows
  at 188,980 tokens/s live (n = 16)…" → "(Figure 3)".
- **Figure 4 (throughput against time).** §2.4: "The 24 hours were spread over three runtimes (Appendix C,
  Table C1)" → "(Appendix C, Table C1, Figure 4)"; Appendix C.2: "Together they left the third runtime's
  TPU idle for 48 min 19 s before training resumed" → "(Figure 4)".
- **Figure 5 (v9 cost).** §5.3: "Table 3 compares the two configurations on the same chip …" → "(Table 3,
  Figure 5)"; "So 3.1% more reads and 3.6% more analytic arithmetic cost 62% more time per step at the
  switch." → "(Figure 5)".
- **Figure 6 (cost model).** §5.4: "Table 4 lists four static counts for each …" → "(Table 4, Figure 6a)";
  "For every change with both a compile-only prediction and a chip measurement, Table 5 compares the
  two …" → "(Table 5, Figure 6b)"; §5.3: "The switch aimed at the contractions, `tokn`, was predicted to
  save 64 ms and saved 8".
- **Figure 7 (graft).** §6.3: "`scripts/graft_run.py` (commit `9a4d652`; Appendix G) carries the
  parameters and optimiser slots of the 59 shared leaves (34,531,720 numbers) …" → "(Figure 7)".
- **Figure 8 (held-out).** §6.5: "The runtime evaluated nine checkpoints next to the training …" →
  "(Figure 8)"; reading (ii): "Its only estimate is `final`'s trend through three checkpoints, extended:
  `v9` lies 0.03 nats below it …" → "(Figure 8)".
- **Figure 9 (blind comparisons).** §7.4: "Table 8 collects everything measured so far …" → "(Table 8,
  Figure 9)"; "Both blind readings are unambiguous against Falcon-H1-Tiny …" → "(Figure 9)".
- **Figure C1 (loss).** Appendix C.3: "Table C3 summarises the step losses logged every 50 steps …" →
  "(Table C3; the whole curve in Figure C1)"; C.1: "… p fell from 0.2415 (stop event) to 0.2190 (restore
  event) and the learning-rate factor rose from 0.783 to 0.803" → "(Figure C1d, line 4)".

## Not made, or made differently from the request

- **Figure 2's whiskers.** Minimum-to-maximum ranges of 0.09–0.47 ms cannot be drawn visibly on a 9 ms
  axis 0.9 in wide; they are stated in the caption with the comparisons they leave unresolved.
- **Table 4's FLOP cell, Table 2's first row, §2.2 and the missing citations** are outside the figure
  blocks; they are listed above with the evidence and the wording to use.
- **matplotlib in the venv.** The task's note says it is installed there; it is not. Nothing was
  installed; the figures use the private copy in `figures/_work/pylib` (see "Python and matplotlib").
- **F1's window.** RESPONSE_1 item 48 asks for "a causal window on both sides"; the task asks for a window
  that does not shift the curve. A causal (trailing) window lags the curve, so the line is a centred window
  that never crosses a restart and shrinks symmetrically near one.
- **The waterfall's first two stages** have no live rate (runtime 1's training runs were discarded); they
  are drawn as benchmarks, apart from the live part, as Table 2 labels them.
- **The calibration plot** uses the six configurations and seven changes on disk; the ~15-configuration
  version of the predictor test needs new compiles and does not exist.
- **`tables.md`** was not regenerated: it is outside this role's files. `make_tables.py` still runs with
  `make_all.sh --tables`; its T2 reads `data/f2_segments.csv`, which keeps its stretch names and holds
  achieved-rate medians (at the cut-off 178,009, 196,879 and, at 2,048 tokens, 188,832, where Table 2 and
  Figure 3 give the live rates 178,087, 196,874 and 188,980).
- **No dark-mode or interactive versions:** these are print figures; the CSVs are the table view.
