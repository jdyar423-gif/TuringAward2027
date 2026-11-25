# Figures: how to rebuild them, captions, readings and sources

Second figures pass of 24 November 2026, 14:40–15:30 UTC, after the figure check; brought up to the
third draft by the lead author after the second programme-committee check (`REVIEW_2.md`, items 10–13),
18:20–18:55 UTC; rebuilt for the final paper at the final cut-off (section "Final pass" below; the captions
are the final ones, copied by `lead5_work/scripts/sync_figures_md.py`; the "Reading" paragraphs further down
keep the 18:25 build's numbers). Figure numbers and captions are now the third draft's: each caption below is copied
verbatim from `paper.md` (`lead4_work/scripts/sync_figures_md.py`), so the two files hold one caption per
figure. Numbers in the "Reading" paragraphs are those of the rebuild at 18:25 UTC (`make_all.sh
--no-extract`, log kept as `lead4_work/make_all_1825.log`): records of the snapshot created 17:14:35 UTC
(48,351 steps; extracted at 17:31), read **to the paper's cut-off, step 46,350 (16:27:21 UTC,
p = 0.8794)**, the newest record of the paper's own live numbers (`lead3_work/numbers.json`; 40 newer step
records left out); held-out records to step 46,178. Paragraphs headed "Changed in this pass" describe the
second figures pass and keep its numbers. Times are UTC. N = the session scratchpad
(`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`), S = the old
scratchpad (`/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`), R = the
repository (`/home/user/turingaward2027`, read only).

Changed at 18:20–18:55 UTC (lead author): `paths.py` takes the default cut-off and the live-rate
cross-check from `lead3_work/numbers.json` (a plain `make_all.sh` now uses the cut-off 46,350; run at
18:25 with `--no-extract`, the snapshot then being rotated, it printed that cut-off and the 2,048-token
MISMATCH was gone; the predictor-test MISMATCH stays, as intended, because
`lead2_work/predictor_test.json` still carries the artefact of known discrepancy 1); `make_data.py` also
reads the local CPU evaluation of step 46,178 (`lead4_work/heldout_local46178.json`, written by
`lead4_work/scripts/local46178.py` from `lead3_work/heldout/v9_46178_{on,pagesoff}.npz`), so Figure 5 rings
all four checkpoints evaluated again and draws the pages-off value at 46,178; `fig8_heldout.py` moves the
arrow right of the newest checkpoint and widens the x axis so that its label is no longer cut;
`fig2_throughput.py` sets the runtime-3 note in narrow lines between tags 4 and 5, clear of the 178.1
label and of tags 7 and 8. No other script changed; the originals are kept as `lead4_work/orig_*.py`.

Figure 6 (F12, the micro graft experiment) does not read the live run, so the cut-off does not apply to
it; its numbers come from the experiment's 15 records and the 3 restart records.

## Final pass (24 November, 21:35–22:50 UTC, lead author)

The figures of the final paper were rebuilt at 21:35 UTC with

```
bash /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_all.sh --cutoff newest --blind final-prefopt-vs-falcon,final-prefopt-vs-mobilellm
```

(log kept as `lead5_work/make_all_final.log`), and again at 22:29 with the same command after the change to
Figure C2's note below (log `lead5_work/make_all_final2.log`; same data, every check line unchanged; the only
difference in the log is the list of test-split reports on disk not drawn, which now includes the K = 1, control
and neutral-title readings). The snapshot read is `N/snap_main` of 18:44:26 UTC, which holds
the whole train log (51,812 steps); its train log is byte-identical to that of the last snapshot `N/snap_final`
(20:16), which the text's numbers come from (`lead5_work/scripts/extract_snap.py`). **Cut-off: step 51,800 at
18:33:30 UTC, the last step record** (the end event of step 51,812 follows 24 s later, so it and the held-out
evaluation of checkpoint 51,812 are after the cut-off and not drawn; `--cutoff 51812` is refused, exit 4,
because no step record has that step). The rebuild prints two `MISMATCH` lines, both expected: the 2,048-token
live rate (189,067, n = 226, here; 189,074, n = 190, in `lead3_work/numbers.json`, which `paths.py` still
reads; the text now uses `lead5_work/numbers.json`, equal to the figures), and the predictor-test artefact of
known discrepancy 1. `paths.py` was not changed.

**One script changed** (`fig2_throughput.py`, Figure C2; the original is `lead5_work/orig_fig2_throughput.py`):
at the final cut-off the runtime-3 note still read "To come: training to 18:33:49, 118 min of post stages, …
the 24 h of TPU time end at 20:46:50", false once the run had ended (the script's own branch for an ended run
needs the end event, which the cut-off excludes), and the box of the "5 h 35 min without a runtime" note hid
the last digit of runtime 2's "n = 101". The note now says, when every TPU interval is closed, "Records to step
51,800 (18:33:30), the last step record. Training's deadline was 18:33:49; the TPU was given back at
20:41:04." (from the setup event and `data/tpu_intervals.csv`; until 22:28 it said "Training stopped at its
deadline, 18:33:49", but the end was recorded at 18:33:54, `FACTS_2.md` CORRECTIONS item 6), and the two gap
notes stand above the axes. Checked on the PNG. No other script changed.

**Readings at the final cut-off.** Figure F2 / C2: live rates 222.0 (n = 101), 216.5 (85), 240.1 (266),
281.4 (142), 178.1 (44), 196.9 (99), 189.1 (226, steps 36,900–48,150, rows of 2,048 tokens) and 189.0 (73,
steps 48,200–51,800, the assistant-only tail); largest gap between live and achieved median 0.120%. Figure 2:
the 2,048-token tick 1,386.5 ms (n = 226). Figure C1: 594 records of `final`, 442 of `v9` (29,750–51,800);
panel (e) draws the 73 tail records; the 2,048-token stretch falls from 2.4038 (first 5) to 1.8149 (last 5),
the tail from 1.7783 to 1.7331. Figure D1: the live loader skipped 30.01% of 64,924 reads of the system chats
(scan 29.77%), 9.29% of 110,540 of UltraChat (9.47%), 6.27% of 358,375 of Tulu 3 (6.07%) and 1.20% of 396,447
of SYNTH (1.17%) from step 25,280 to 51,800. Figure 5: runtime checkpoints to 51,588 (1,043.0 minutes), the
line's gap there −0.0771 ± 0.2246; the CPU evaluations of 51,812 (`lead5_work/heldout5.json`) are not inputs
of the figure. Figure 7: the official model's test rows, 35 / 57 / 108 (score printed 0.318) and 116 / 69 / 15
(printed 0.752, as the text prints it: exact halves are rounded to even); the official model is labelled "v9, step 51,812". Figure 6 (F12) is unchanged.

**Known cosmetic defects, not fixed** (no script failed on them): in Figure C1 the list of vertical lines in
panel (a) runs into the `v9` points near 10¹⁰ tokens, and in panel (d) the label "p = 0.67 (P1)" overlaps the
label of line 4. Both stay legible.

## How to rebuild (one command)

```
bash /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_all.sh
```

This reads the newest snapshot of the live run, rebuilds every CSV in `paper/data/` (`make_data.py`) and
then every figure (`F1`–`F12`, each as `.svg` with outlined text and `.png` at 200 dpi). It takes about
20 s on CPU cores 2 and 3 and keeps the printed checks of every script in
`paper/figures/_work/make_all.log`.

**F12 and the micro graft records.** `fig12_micrograft.py` does not read the snapshot: it reads the
records of the micro graft experiment (`N/micro_graft/6M/results_all.jsonl`), recomputes every number,
writes its own two CSVs (`data/micro_graft_curve.csv`, `data/micro_graft_summary.csv`) and draws from
them; the cut-off does not apply to it. `make_all.sh` prints two of its lines: `F12: graft_report.json:
… compared with the recomputation; largest difference …` (every number of the experiment's own report
against the recomputation; a `MISMATCH` line for each that differs) and `F12: restart arms with half the
budget: drawn in (a): …` with the numbers, or `… left out of (a): <why>`. The restart arms (`v9_pages`
from scratch on a budget of 3,000,000 tokens, seeds 0–2) exist as `N/micro_graft/3M/results.tar.gz`
(17:30 UTC); every rebuild since 17:31 unpacks it into `data/_src/micro_graft_3M` and draws them in (a)
(the 18:25 rebuild prints "drawn in (a): … +0.1985 ± 0.0062 (per seed +0.1962, +0.2055, +0.1938)", the
numbers of the caption). F12 alone (about 1.5 s; it writes only its two CSVs, the unpacked restart
records and `F12_micrograft.svg/png`):

```
taskset -c 2,3 nice -n 5 /tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python -I -B /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/fig12_micrograft.py
```

**The cut-off.** Every CSV, and so every figure, uses the run's records up to one fixed step, written to
`data/cutoff.csv` and printed in the figures that show live data (F2, C2, 2, C1, D1). The default
(`--cutoff paper`) is the newest step of `lead3_work/numbers.json` (`v9.newest_step`, step 46,350), the
file the text's live numbers (Table 2, §6.3) come from; so the figures and the text use the same
records, and the rebuild checks every live rate against `numbers.json` (a `MISMATCH` line if one
differs). When the lead author refreshes `numbers.json` (and the text) from a newer log, and points
`paths.NUMBERS` at it, the figures follow on the next rebuild; then the cut-off stated in the captions of
Figures F2, C2, 2, C1 and D1 (step, time, and the counts that depend on it: Figure 2's "steps
36,900–46,350 … n = 190, 1,386.5 ms", Figure C1's "333 of `v9`") must be updated from the printed checks. `--cutoff newest` uses every record of the snapshot (and prints
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
between the live and the achieved median` the check of §2.2's "at most 0.13%"; a line `test-split
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
| `paper/lead3_work/numbers.json` | the cut-off (`v9.newest_step`, 46,350) and the text's live rates (cross-check) | `cutoff` |
| `N/midrun_eval_1.json`, `N/heldout_v9.jsonl` | the runtime's held-out evaluations (`scripts/midrun_eval.py`; web 64, chat 64, proxy 12 rows); checkpoints after the cut-off left out | `heldout_loss`, `heldout_trend` |
| `paper/lead2_work/heldout_analysis.json` | local CPU evaluations of 29,722, 34,166 and 35,294, pages switched off (Tables 5, G2, G3); also a cross-check of Table 5's line in minutes | `heldout_loss` |
| `paper/lead4_work/heldout_local46178.json` | local CPU evaluation of 46,178, pages on and off (from `lead3_work/heldout/v9_46178_*.npz`; Tables 5, G2) | `heldout_loss` |
| `paper/lead2_work/predictor_test.json` | Tables 4 and 5 | `costmodel_steps`, `costmodel_differences` |
| `paper/lead2_work/graft_grad.json`; `S/graft_test.log`, `S/graft_dry_out.txt`, `paper/review1_work/graftcheck/graft_check_29722.log` | gradient check at the graft; the tool's `--check` outputs with their batch shape ("check on 2 x 256 synthetic tokens") (Table 6) | `graft_facts` |
| `data/_src/live/run.log`, `data/_src/live/run/manifest.json` | the loader's read cap before the graft (first loader line, "at most 8 reads") and after it (3.96093), the speed switches the run started with after the graft (`fast:` line), the data seed (`run.seed=1` in `overrides`), the `graft` record | `graft_facts` |
| `N/study/proxy/reports/ab/*.json` | blind comparisons: `s1-default-vs-{falcon,mobilellm}--opus-1`, `s2-default-vs-{falcon,mobilellm}--judge-opus`, and any `--blind` names | `blind_comparisons` |
| `N/wf/chip_kernels_0745.txt`, `S/v9_meas{A,B_sorted,C_onehot,D}.txt`, `S/micro_out.txt`, `S/flags_out.txt`, `S/flags_out2.txt`, `R/docs/tpu_profile/tpu_tune.v6e.txt` | chip measurements; how a whole step was timed from `R/scripts/tpu_kernels.py` and `R/scripts/tpu_micro.py` (column `timing`) | `chip_*` |
| `S/v9.log`, `S/final.log`, `S/colab_bes24h_final.log`, `R/docs/tpu_profile/final_b.run.log`, `data/_src/final/run.log` | runtime-1 benchmarks and single log lines | `throughput_log_points` |
| `S/tpu_intervals.txt` | the TPU ledger (the open runtime counted to the cut-off) | `tpu_intervals` |
| `N/openings.json`, `N/deflection_check2.json` | data-rule scans | `openings_scan`, `deflection_check2` |
| `R/bes/budget.py`, `R/bes/roles.py`, `R/bes/data/sources.py` (imported read-only) | ledger rules, source kinds | `ledger_leaves`, `openings_scan` |
| `N/micro_graft/6M/results_all.jsonl` (read by `fig12_micrograft.py`; compared with each arm's own `results.jsonl` beside it) | the 15 finished records of the micro graft experiment (`R/docs/MICRO_GRAFT.md`): held-out losses at four checkpoints, the graft blocks and their checks | `micro_graft_curve`, `micro_graft_summary` |
| `N/micro_graft/6M/graft_report.json` | the experiment's own report (`R/scripts/micro_graft_report.py`): compared with, not plotted | – |
| `N/micro_graft/3M/results.tar.gz` (since 17:30; unpacked into `data/_src/micro_graft_3M`) | the restart arms: `v9_pages` from scratch on a budget of 3,000,000 tokens, seeds 0–2 | `micro_graft_curve`, `micro_graft_summary` |

## Conventions

One font (Inter, at least 6.5 pt at the printed size), one palette: blue `#2a78d6` is always
`configs/final.py` (in Figures 1 and 7: what only `final` has), orange `#eb6834` always `configs/v9.py`
(what only `v9` has), grey is context or what both share. Where no configuration is meant, violet
`#6250d6` marks the one series a figure is about (F5, F10). In F12, whose arms are micro models, orange
marks the arms that carry `v9`'s page, darker the longer the page trained (`style.PAGE_RAMP`, an ordinal
ramp of the orange's hue, checked with the palette validator's ordinal checks), and black the reference
arm. Every pair that carries meaning differs in grey as well: configurations are named in text in every
figure; F2's "no TPU" bands are hatched, not a second grey; F10 separates its series by marker shape (and
the two kinds of change in (b) by triangle direction); in F1 (a) `v9`'s light dots, as light as
`final`'s, are named directly; F12's four oranges are four greys and its arms are named at their line
ends. "Tokens" are token positions of the training batches, 262,144 per optimiser step, padding included
(in F12, the micro bench's 4,096 per step). The **live rate** of a
stretch is the median of the rates the trainer logs with its step records (one record every 50 steps);
the logged rate is 262,144 ÷ the trainer's running average of the time per step (weight 0.2 on the
newest step; `R/bes/train/train.py` lines 965 and 903), and that time is the time ledger's training time,
so the time the trainer books to checkpoint writes (ledger phase `ckpt`, `_save`) is not part of it. The
**achieved rate** is 262,144 × steps between two records ÷ the wall time between them, checkpoint writes
included. At the cut-off of step 46,350 the two medians differ by at most 0.124% in every stage (the
2,048-token stage: live 189,074, n = 190; achieved 188,840, n = 189), which is §2.2's "at most 0.13%" (at
the second draft's cut-off, step 37,650, the largest gap was 0.078%). Print widths: F1, F2, F5, F6, F7, F9, F10, F11, F12 both columns (7.0 in); F3, F4, F8 one
column (3.4 in). The CSVs are the table view of every figure.

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
| P2 | 48,190 | no stop; stage event 17:09:57, p = 0.9200 | chat share 0.70, loss on assistant tokens only (schedule); added automatically from the train log's stage event |

## Figure numbers in paper.md (third draft)

| paper.md | files | script | changed after `REVIEW_2` (18:20–18:55) |
|---|---|---|---|
| Figure 1 | `F7_model.svg/png` | `fig7_model.py` | no change |
| Figure 2 | `F4_v9cost.svg/png` | `fig4_v9cost.py` | no change (the 2,048-token tick reads the cut-off 46,350: 1,386.5 ms, n = 190) |
| Figure 3 | `F10_costmodel.svg/png` | `fig10_costmodel.py` | no change |
| Figure 4 | `F6_graft.svg/png` | `fig6_graft.py` | no change |
| Figure 5 | `F8_heldout.svg/png` | `fig8_heldout.py` | the local evaluation of 46,178 added (ring; pages off, 2.2754); arrow moved right and x axis widened so that its label is inside the axes |
| Figure 6 | `F12_micrograft.svg/png` | `fig12_micrograft.py` | no change (restart arms drawn since 17:31); caption reconciled with `paper.md` |
| Figure 7 | `F11_blind.svg/png` | `fig11_blind.py` | no change; the caption now says why only the Opus readers are drawn |
| Figure C1 | `F1_loss.svg/png` | `fig1_loss.py` | no change |
| Figure C2 | `F2_throughput.svg/png` | `fig2_throughput.py` | the runtime-3 note set in narrow lines between tags 4 and 5 (it ran into the 178.1 label and tags 7–8) |
| Figure D1 | `F5_openings.svg/png` | `fig5_openings.py` | no change |
| Figure F1 | `F3_kernels.svg/png` | `fig3_kernels.py` | no change |
| Figure F2 | `F9_waterfall.svg/png` | `fig9_waterfall.py` | no change |

Second-draft numbers, for the history paragraphs below: Figures 2–9 of the second draft (F3, F9, F2, F4,
F10, F6, F8, F11) are now Figures F1, F2, C2, 2, 3, 4, 5 and 7; Figure 6 (F12) is new. Every figure of
the main text is cited in the running text (`lead3_work/check_refs.py`).

---

## Figure 1 (`F7_model`). The model at one token position

**Script and inputs.** `fig7_model.py`; every count from `data/ledger_leaves.csv` (both checkpoints'
manifests with the read rules of `R/bes/budget.py`) and `data/ledger_final_export.csv` (`final`'s
export accounting: library and index); architecture text from `R/docs/SPEC_MODEL_TRAIN.md` 4.1–4.3,
`R/docs/V9.md` 4, `R/bes/model/model.py`, `R/bes/config.py`.

**Caption** (as in `paper.md`, Figure 1; copied by `lead5_work/scripts/sync_figures_md.py`). The model at one token position, with stored numbers / numbers read per generated token
(standard reading) for each part, as in Table 1 (`data/ledger_leaves.csv`, `data/ledger_final_export.csv`).
Gray: shared by both configurations; blue: `final` only (the written library); orange: `v9` only (gains and
token neurons). The 15 small dense leaves (306,776 numbers) are counted and not drawn. A schematic: only the
counts are data.

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

## Figure F1 (`F3_kernels`). Attention-kernel candidates on the chip

**Script and inputs.** `fig3_kernels.py`; `data/chip_kernels_0745.csv` from `N/wf/chip_kernels_0745.txt`
(the `KERNELS` line; medians recomputed from the 30 timings agree to 10⁻³ ms, column `median_check`;
`min`, `max`, `n`).

**Caption** (as in `paper.md`, Figure F1; copied by `lead5_work/scripts/sync_figures_md.py`). One value-and-gradient of a piece of 4,096 tokens (left 4 × 1,024, right 2 × 2,048) of
`configs/final.py` on the chip, eight kernel configurations measured in one pause (24 November
07:44–07:52; `wf/chip_kernels_0745.txt`). Dots: median of 30 timings. The 30 timings of a dot span
0.09–0.47 ms from minimum to maximum, less than the width of a dot, so the ranges are not drawn; they
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

## Figure F2 (`F9_waterfall`). Throughput by cause

**Script and inputs.** `fig9_waterfall.py`; `data/throughput_stages.csv` (`make_data.py`: live and
achieved medians per stage from `train_steps.csv`, the step of the change that starts each stage
(`from_step`), Table 2's synthetic values from `chip_whole_steps.csv`, benchmarks and single log lines
from `throughput_log_points.csv`), `data/interventions.csv` (P1), `data/cutoff.csv`. The live rates are
checked against `lead2_work/numbers.json` on every rebuild: at the cut-off all eight are equal, the
2,048-token stage included.

**Caption** (as in `paper.md`, Figure F2; copied by `lead5_work/scripts/sync_figures_md.py`). Training throughput by cause (Table 2), in the order the changes happened. Upper part,
runtime 1 (32-row micro-batches, training runs discarded): start-up benchmarks (hollow bars) and single
log lines of training (dots; the reference row's two dots are the two monitor readings, 116,718 and
115,394 tokens/s). Lower part, run main16: each bar spans the live rate (Section 2.2) before and after one
change, with the window of step records printed under the stage (n, first and last step, UTC); records
to step 51,800 (24 November, 18:33:30 UTC), the last step record, the last stage being the assistant-only
tail from p = 0.92. Diamonds: Table 2's synthetic or benchmark value of the same stage. A bar is a
before-and-after difference of medians, not the effect of one change alone: the step from runtime 2 to
runtime 3 changed no code, and the graft changed the configuration.

**Reading** (final rebuild, cut-off 18:33:30). Runtime 1: benchmark 111,893 → 201,877 tokens/s (+80.4%),
monitor lines 116,718 (step 72, 04:17) and 115,394 (step 301, 04:28), and one log line after the rewrite,
204,699 (step 50, 04:48). Run main16, live: 222,031 (n = 101, steps 50–5,050) → 216,548 (−2.5%, n = 85)
→ 240,131 (+10.9%, n = 266) → 281,389 (+17.2%, n = 142) → 178,087 (−36.7%, n = 44) → 196,874 (+10.5%,
n = 99) → 189,067 at 2,048 tokens (−4.0%, n = 226, steps 36,900–48,150) → 189,035 in the assistant-only tail
(−0.0%, n = 73, steps 48,200–51,800), the values of Table 2 and §6.3. Synthetic or benchmark values: 216,681;
–; 246,538; 296,371; 182,919; 203,665; –; –.

**Changed in this pass.** The 2,048-token bar showed 189.1k (n = 59, to 13:55), Table 2 above it 188,980
(n = 16): the figures read whatever snapshot was newest. They now stop at the paper's cut-off and agree
with Table 2 and §6.3 (with every record of the 14:46 snapshot the stage would read 189,069, n = 102).
The two monitor readings, 1.3k apart, rendered as one dot; they are now drawn one above the other and
named "monitor lines" on the right. The stage labels' steps ("step 9,332" …) were typed; they are read
from `throughput_stages.csv` (`from_step`, from `interventions.csv`). The caption defines the logged
rate as the trainer computes it.

## Figure C2 (`F2_throughput`). Throughput against wall-clock time

**Script and inputs.** `fig2_throughput.py`; `data/train_steps.csv` (`interval_tok_per_s`),
`data/throughput_stages.csv` (printed numbers and n), `data/throughput_log_points.csv`,
`data/run_events.csv`, `data/interventions.csv`, `data/tpu_intervals.csv`, `data/cutoff.csv`; runtime-1
measurement windows 03:34–03:57 and 04:08–04:17 from `R/docs/CHANGES.md` 19.1.

**Caption** (as in `paper.md`, Figure C2; copied by `lead5_work/scripts/sync_figures_md.py`). Training throughput against UTC time on the three runtimes (axis broken where no runtime
was held). Runtime 1 (23 November) holds runs before run main16, all discarded: start-up benchmarks
(diamonds) of `final` before and after the glue rewrite and of `v9` before it (orange), and single log
lines of `final`'s training (dots). Runtimes 2 and 3, run main16: lines, the rate achieved between
consecutive step records (tokens ÷ wall time, checkpoint writes included); numbers, the live rate of
each stretch in 10³ tokens/s with its number of step records n (Table 2, Figure F2); in runtime 2, the
start-up benchmark (diamond) and the first log line (dot). Light gray bands: TPU held but not training;
hatched: no TPU (its length printed above the axis). Tags: restore times of the restarts of Table C2; P1
and P2 (dashed): rows of 2,048 tokens from p = 0.67 and the assistant-only tail from p = 0.92, without a stop.
Records to step 51,800 (24 November, 18:33:30 UTC), the last step record; the end of training (deadline 18:33:49,
recorded at 18:33:54), the stages after it (18:33:55–20:02:22) and the hold to 20:41:04 are not drawn. A change of level at a tag is
a before-and-after difference, not the effect of that change alone.

**Reading** (final rebuild, 22:29). 222.0 (n = 101) → 216.5 (85) → 240.1 (266) → 281.4 (142) → 178.1 (44) →
196.9 (99) → 189.1 (226) → 189.0 (73, the tail) (10³ tokens/s). Runtime 1's points: `v9` benchmark 88,146 (03:13), `final` 111,893 (03:27), monitor
lines 116,718 and 115,394, `final` after the rewrite 201,877, 201,918 and 204,006 (benchmarks, 04:36–04:42)
and 204,699 (log line, 04:48). Runtime 3 spent 48 min 19 s restoring and fixing two data errors before
training resumed. The note in runtime 3 is computed from the train log's set-up events, the TPU ledger
and `cutoff.csv`; with every TPU interval closed it reads "Records to step 51,800 (18:33:30), the last step
record. Training's deadline was 18:33:49; the TPU was given back at 20:41:04." (the script's other branch,
for an open interval, is not used). Since 18:25 the note is set in narrow lines between tags 4 and 5, under
the 240.1 line; at 17:31 it ran into the 178.1 label and the lines of tags 7 and 8 (`REVIEW_2` item 13).

**Changed in the third pass.** The two band greys (`#f0efec` and `#e6e5e0`, 3.5 L* apart) were hard to tell
apart in print; "no TPU" is now a hatch. Every live rate carries its n. The caption said "run main16"
for a figure whose runtime 1 shows only discarded runs, a `v9` benchmark among them; it now says so. The
data stop at the cut-off, now the last step record (18:33:30; last tick 18:00), not 13:30 as then.

## Figure 2 (`F4_v9cost`). Cost of the v9 step against final's

**Script and inputs.** `fig4_v9cost.py`; `data/chip_whole_steps.csv` (`N/wf/chip_kernels_0745.txt`,
`S/v9_measA.txt`, `S/v9_measB_sorted.txt`, `S/v9_measC_onehot.txt`, `S/v9_measD.txt`; column `timing`),
`data/throughput_stages.csv` (live rates with n), `data/cutoff.csv`.

**Caption** (as in `paper.md`, Figure 2; copied by `lead5_work/scripts/sync_figures_md.py`). One accumulation step of 262,144 tokens on a synthetic batch, timed on the chip, by
configuration, table-gradient mode and piece size; ratios against `final`'s fastest step of the same length.
Each bar is one timing (Section 5.1). Black ticks: 262,144 ÷ the live rate, with n: Table 3's windows and, at
2,048 tokens, steps 36,900–48,150 (n = 226, 1,386.5 ms), not timed on a synthetic batch. `final` was timed
with the hand-written cross-entropy backward on (0.177 ms slower per piece; the run does not use it).

**Reading.** Synthetic: 884.5 ms (`final`) against 1,433.1 ms (`v9`, ×1.62), with the switches 1,287.1
ms (×1.46). Live: 931.6 ms (`final`, 281,399 tokens/s, n = 89), 1,472.0 (`v9`, n = 44), 1,331.5 (with the
switches, n = 99), 1,386.5 at 2,048 tokens (n = 190, to the cut-off 46,350; `fig4_v9cost.py` in the 18:25
build).

**How a whole step was timed.** `R/scripts/tpu_kernels.py`: one call compiles, one warm call, then
`--step-calls` calls back to back, and `ms` is their mean (default 2; `wf/v9_speed.md`, `wf/v9_rows.md`
and `wf/v9_tokn.md` plan `--step-calls 2`, but the commands that produced the five output files were not
recorded). `S/micro_out.txt` (Figure 3's "4 pieces" synthetic value) comes from `R/scripts/tpu_micro.py`,
the mean of `--calls` steps (default 3). So no whole step has a spread.

**Changed in this pass.** n for every live tick (it was given for the 2,048-token tick only); the caption
says what a bar is; the 2,048-token tick stops at the cut-off (1,387.2 ms = 262,144 ÷ 188,980).

## Figure 3 (`F10_costmodel`). The compiler's cost model against the chip

**Script and inputs.** `fig10_costmodel.py`; `data/costmodel_steps.csv` and
`data/costmodel_differences.csv` (`make_data.py` from `paper/lead2_work/predictor_test.json`: the six
configurations' counts and chip times, and Table 5's predicted and measured differences). The
leave-one-out predictions are recomputed from the json's rows and checked against its errors.

**Caption** (as in `paper.md`, Figure 3; copied by `lead5_work/scripts/sync_figures_md.py`). The compiler's estimates against the chip. (a) The six configurations of Table 4, sorted by
the measured step (bar). Markers: each static count's leave-one-out prediction, joined to the bar by its
error; FLOP and reads give the same predictions, undefined for `final`. (b) The seven changes of Table F5:
predicted against measured change of the step, absolute values on logarithmic axes; triangles up: slower,
down: faster; after each label, predicted ÷ measured. Six configurations chosen after the measurements: the
figure shows how the counts did here, not that bytes moved predict step time in general.

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

## Figure 4 (`F6_graft`). The graft

**Script and inputs.** `fig6_graft.py`; `data/ledger_leaves.csv` (both checkpoints' manifests),
`data/ledger_final_export.csv`, `data/run_events.csv` (stop, restore, set-up: deadline, training end,
pieces per micro-batch), `data/interventions.csv`, `data/graft_facts.csv` (the three `--check` outputs of
Table 6 with their batch shape; `lead2_work/graft_grad.json`; the loss guard of the
`ckpt-000026-step-000029722` manifest; the live manifest's `graft` record and `run.seed`; from the live
`run.log`, the loader's cap before the graft, the first loader plan after it and the speed switches of
the start that continued the graft). No number is typed in the script.

**Caption** (as in `paper.md`, Figure 4; copied by `lead5_work/scripts/sync_figures_md.py`). Stored numbers of the two configurations at step 29,722, to scale; colours as in Figure 1.
Gray: the 59 shared leaves, carried with their optimiser slots; blue: `final`'s library and index; orange:
`v9`'s five new leaves, with fresh slots. Right: what else was carried, what is new, and the checks of
Section 6.4 (`data/graft_facts.csv`, `data/ledger_leaves.csv`).

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

## Figure 5 (`F8_heldout`). Held-out loss across the graft

**Script and inputs.** `fig8_heldout.py`; `data/heldout_loss.csv` (`N/midrun_eval_1.json`,
`N/heldout_v9.jsonl`; local evaluations from `lead2_work/heldout_analysis.json` (29,722, 34,166, 35,294)
and `lead4_work/heldout_local46178.json` (46,178); checkpoints after the cut-off left out, none so far),
`data/heldout_trend.csv` (the line and band in training minutes, computed by `make_data.py` and checked
against the second draft's values in `heldout_analysis.json` on every rebuild). The figure draws the line
in minutes only; the line in log tokens and the comparison at equal tokens are in Table 5
(`lead4_work/heldout4.json`).

**Caption** (as in `paper.md`, Figure 5; copied by `lead5_work/scripts/sync_figures_md.py`). Held-out loss on the comparison set of Table 5 against training minutes, to step 51,588 (the
figures stop at the last step record, 51,800; the final checkpoint 51,812, at the same 2.0438, is in Table 5).
Filled points: the runtime's evaluations; rings: CPU re-evaluations (29,722, 34,166, 35,294, 46,178); open
diamonds: `v9` with its pages switched off (34,166, 35,294, 46,178). Gray line and band: the least-squares
line through `final`'s three points, extended, with its 95% prediction interval (one residual degree of
freedom); arrow: `v9`'s distance from it at 51,588, −0.077 ± 0.225. Every `v9` point lies inside the band; in
log tokens or at equal tokens later checkpoints lie outside (Table 5). The figure does not show that `v9` is
ahead of a continued `final` (reading ii).

**Reading** (18:25 rebuild, to step 46,178; the final rebuild draws the runtime's checkpoints to 51,588,
where the gap is −0.0771 ± 0.2246, section "Final pass" above and Table 5). `final`'s line falls
2.088 × 10⁻⁴ nats per training minute (residual sd 1.05 × 10⁻³). `v9` minus the line: −0.0096 ± 0.0249 (563.7 min), −0.0152 ± 0.0354 (592.5, step 31,912),
−0.0277 ± 0.0554 (642.5, step 34,166), −0.0332 ± 0.0657 (667.5, step 35,294), −0.0720 ± 0.1289 (817.7,
step 41,850) and −0.0882 ± 0.1714 (917.9, step 46,178): every point inside the band. With the pages off,
2.2241, 2.2305 and 2.2754 at 34,166, 35,294 and 46,178 (above the line, 2.2045, 2.1993 and 2.1470). The
y axis runs from 1.9716 to 2.3224, the band at 917.9 minutes from 1.9756 to 2.3184. Rings at the four
local evaluations: 29,722, 34,166, 35,294 and 46,178.

**Assumptions and limits.** Training minutes are rounded to 0.1 as in Table G2 (with unrounded minutes
the half-widths at 642.5 and 667.5 min are 0.0551 and 0.0654, and at 917.9 min 0.1706,
`lead3_work/heldout3.json`). The axis shows the band at the newest
checkpoint unless it is wider than the points' own range; later checkpoints far from the three `final`
points will let the band leave the plot.

## Figure 7 (`F11_blind`). Blind comparisons

**Script and inputs.** `fig11_blind.py`; `data/blind_comparisons.csv` (`make_data.py` from
`N/study/proxy/reports/ab/s1-default-vs-falcon--opus-1.json`, `s1-default-vs-mobilellm--opus-1.json`,
`s2-default-vs-falcon--judge-opus.json`, `s2-default-vs-mobilellm--judge-opus.json`, and the reports
passed with `--blind`). Our checkpoint's step is read from each report's `x` field and its configuration
from the step (after 29,722: `v9`); "two readings" is taken from the report's judge block (orders agree).

**Caption** (as in `paper.md`, Figure 7; copied by `lead5_work/scripts/sync_figures_md.py`). Blind pairwise comparisons by the Opus readers (Tables 7, 9): `final` at step 25,280 on the
first 60 development prompts, read once; `v9` at step 31,912 on the same prompts and the official model
(labelled `v9`, step 51,812) on the 200 test prompts, read twice. Bars: wins, ties or "both bad", losses;
dots and whiskers: the score with its 95% Wilson interval. Only this added reader is drawn, because it decides most pairs;
the registered judge points the same way where it decides (Tables 7, E2).

**Reading** (final rebuild; wins / ties or "both bad" / losses, as the bars are drawn). Against
Falcon-H1-Tiny on `dev`: 2 / 15 / 43 (score 0.158 [0.087, 0.271]) and 3 / 18 / 39 (0.200 [0.118, 0.318]); the
official model on `test`: 35 / 57 / 108 (0.318 [0.257, 0.385]). Against MobileLLM-R1.5 on `dev`: 15 / 37 / 8
(0.558 [0.433, 0.677]) and 20 / 33 / 7 (0.608 [0.482, 0.722]); on `test`: 116 / 69 / 15 (0.752 [0.688, 0.807]).

**The final test.** The two official test sets are passed with `--blind
final-prefopt-vs-falcon,final-prefopt-vs-mobilellm` (rebuilds of 21:35 and 22:29); their rows stand under each
opponent after the development rows, labelled with their split (`test`) and n. The caption's last sentence
was kept: it says why only the Opus readers are drawn.

## Figure C1 (`F1_loss`). Training loss, restarts and schedule

**Script and inputs.** `fig1_loss.py`; `data/f1_loss_smoothed.csv` (`make_data.py`: raw step losses and
the centred average, with its effective record count, half-width and window centre per record),
`data/train_steps.csv`, `data/interventions.csv`, `data/cutoff.csv`.

**Caption** (as in `paper.md`, Figure C1; copied by `lead5_work/scripts/sync_figures_md.py`). Training loss of run main16 against training tokens: (a) the whole run on logarithmic
tokens; (b) from 10⁹ tokens on, linear. Dots: the loss of one optimiser step (the mean over its 262,144
tokens), logged every 50th step: 594 records of `final` (steps 50–29,700) and 442 of `v9` (steps
29,750–51,800, the last step record, 18:33:30 UTC on 24 November).
Lines: a centred average over log₁₀(tokens), computed separately within each stretch between restarts
and phase changes: each record is weighted by a Gaussian (σ = 0.03 decades, cut at 3σ) times 1/tokens,
so that the window is centred on the record (to within 0.001 decades), and near the end of a stretch
the window shrinks symmetrically instead of becoming one-sided; the line is drawn where it averages at
least 15 records. A step at a restart would show
as a break between two lines; near the ends of a stretch the line rests on fewer records and wanders by
about ±0.01 nats. Vertical lines: the restarts of Table C2 (solid; M1, M2: measurement pauses), the
switch to rows of 2,048 tokens at p = 0.67 (P1, dashed) and the tail at p = 0.92 (P2, dashed). (c, d) The
learning-rate factor and progress p as logged. (e) From p = 0.92 the loss covers the assistant tokens of
conversations (and base text) only; its 73 records are left out of (a) and (b) and drawn on their own scale,
not comparable with (a) and (b). The loss mixes 22 sources whose shares change with p and the data rules;
the stream changed at the graft, and from P1 its chat share rises linearly (Appendix C.1), which by itself is
expected to lower the loss of a step: levels on either side of a restart, of P1 or of P2 do not compare
configurations.

**Reading.** A step's loss falls from 4.858 at step 50; the average reads 2.736 at step 3,100 (where it is
first drawn), 2.690 at 3,800 (10⁹ tokens), 2.584 at 11,450, 2.511 at 22,200 and 2.493 at 29,000. Across
the graft: 2.487 at step 29,300 (last `final` point drawn) and 2.490 at 30,150 (first `v9` point); the last
21 logged `final` steps average 2.492 (sd 0.039), the first 21 `v9` steps 2.497 (sd 0.045). In the final
rebuild the 2,048-token stretch before the tail has 226 records (steps 36,900–48,150), from 2.404 (mean of
the first 5) to 1.815 (last 5), while the chat share of the stream rises (Appendix C.1), and panel (e) draws
the 73 records of the tail (48,200–51,800), from 1.778 to 1.733; at the 18:25 rebuild (190 records, to
46,350) the stretch's line was drawn from step 37,300 to 45,950. The scatter of one step's loss around the line, where the window is full, is 0.037 nats
(sd, 91 records). Largest centre offset of a drawn point: 0.000999 decades (15:17 build; the 18:25 log does
not print it).

**Changed in this pass.** The caption referred to panels "(c, d)", which had no letters; they are now
titled c and d. The white box of the label "4: p and the factor jump" hid the p curve between about 5.0
and 6.6 × 10⁹ tokens; the label now stands between the two curves without a box, and the label of
p = 0.67 moved below its line (above it, its box cut the factor curve). In (a) `v9`'s light dots have the
lightness of `final`'s (L* 79.5 against 78.3); the legend now says "from line 7" and the cloud is
named. The caption gives the number of records and the cut-off.

**After p = 0.92.** From p = 0.92 the trainer logs the loss of assistant tokens only (stage `tail`); those
records are left out of (a) and (b) and drawn in an extra panel (e) on their own scale, with P2 marked;
the caption says so ("(e) From p = 0.92 the loss covers the assistant tokens of conversations (and base
text) only …").

## Figure D1 (`F5_openings`). Stock openings and what the loader leaves out

**Script and inputs.** `fig5_openings.py`; `data/openings_scan.csv` (`N/openings.json`),
`data/deflection_check2.csv` (`N/deflection_check2.json`), `data/loader_drops_live.csv` (train logs, to
the cut-off), `data/cutoff.csv`. Shares are computed from the counts in the CSVs and rounded half up in
decimal (0.15% prints as 0.2).

**Caption** (as in `paper.md`, Figure D1; copied by `lead5_work/scripts/sync_figures_md.py`). (a) Share of each source's assistant turns whose opening (first 44 tokens) matches a stock
pattern, from the scan of the tokenised sources (24 November 08:20; all segments of small sources,
40,000 sampled otherwise); "Sure …" is a control that no rule removes; a turn can match several
patterns. (b) Share of each chat source's conversations that the stock-refusal rule (light gray, live
since step 6,647) and, in addition, the deflection rule (violet, live since step 25,280) leave out,
measured with the loader's functions on 30,000 sampled conversations per source (2,260 for
smoltalk_everyday, all it has). Black ticks and the column on the right: share of reads the live loader
actually skipped for either rule from step 25,280 to step 51,800 (18:33:30 UTC on 24 November, the last
step record; `final` and `v9` together), with the number of reads. In both
panels "<0.1" means above 0 and below 0.05%, and "0" means none.

**Reading** (18:25 rebuild, cut-off 16:27:21; the final rebuild's shares to step 51,800, e.g. 30.01% of
64,924 reads of the system chats, are in the section "Final pass" above). The live loader skipped 29.82% of
44,541 reads of the system chats (scan 29.77%), 9.33% of 77,492 of UltraChat (9.47%), 6.30% of 257,801 of
Tulu 3 (6.07%), 1.22% of 254,236 of SYNTH (1.17%); smoltalk_constraints 0.04% of 64,571 (printed "<0.1";
scan: 11 of 30,000 conversations, 0.04%); smoltalk_everyday 0.00% of 58 reads (scan 0.93%).

**Changed in this pass.** smoltalk_constraints was labelled "0" in (b) although the rule removes 11 of
30,000 of its conversations; small non-zero shares now print "<0.1" in both panels, "0" only for none.
The live ticks now carry their share and number of reads (right column). Shares are rounded from the
counts (`smoltalk_magpie_ultra`'s 45 of 30,000 printed as 0.1 from its binary float, now 0.2).

## Figure 6 (`F12_micrograft`). Graft or the page from the first step, at the micro width

**Script and inputs.** `fig12_micrograft.py`; the 15 finished records of the micro graft experiment
(`R/docs/MICRO_GRAFT.md`), `N/micro_graft/6M/results_all.jsonl` (`ref`, `v9_pages`, `graft_25`,
`graft_50`, `graft_75`, seeds 0–2; equal to the newest finished record in each arm's own
`results.jsonl` beside it, 15 of 15). Every number is recomputed from the records. `mix` is the held-out
loss of web and chat weighted by the record's training row shares (`arm.mixture`: 0.75 web, 0.25 chat,
normalised over the evaluated domains other than assistant), as `R/scripts/micro_graft_report.py`
(`losses`) and `R/bes/eval/microbench.py` (`domain_shares`) define it; a domain's loss is the mean
cross-entropy per token over its 32 held-out rows (web 16,354 tokens, chat 13,637; the same rows in
every record and checkpoint). The experiment's own report, `N/micro_graft/6M/graft_report.json`, is only
compared with. Writes `data/micro_graft_curve.csv` (one row per record and checkpoint, and the loss right
after each graft from the record's check) and `data/micro_graft_summary.csv` (what is drawn: mean, sd,
min, max and each seed, with the report's value beside it), then draws from the summary as read back.
When `N/micro_graft/3M/results.tar.gz` exists, it is unpacked into `data/_src/micro_graft_3M` and the
three finished `v9_pages` records with a budget of 3,000,000 tokens (configuration checked against the 6M
`v9_pages`; all seeds of `ref` required) are added to (a); otherwise the script prints that they are left
out and why.

**Caption** (as in `paper.md`, Figure 6; copied by `lead5_work/scripts/sync_figures_md.py`). The graft at the micro width (Table 6; width 128, 6 × 10⁶ tokens, three seeds). (a) Each arm's
held-out loss (0.75 × web + 0.25 × chat) minus `ref`'s with the same seed: large dots, the mean of the seeds;
small dots, each seed; a graft arm leaves the zero line at its diamond. Squares, above a break in the axis:
the restart at its own end (3 × 10⁶ tokens) minus `ref` at 6 × 10⁶, +0.1985, which pairs unequal token
counts. (b) Share of the page gain recovered at the end: large dots, the ratio of the mean differences (1.06,
0.69, 0.24); small dots, each seed. No interval is drawn: three seeds are too few.

**Reading** (records of 24 November, 14:12:51–16:41:01 UTC; nats per token, mean ± sd of the three
paired differences). (a) `v9_pages` − `ref`: −0.0164 ± 0.0062 after 1.50 million tokens, −0.0329 ±
0.0091 after 3.00, −0.0494 ± 0.0100 after 4.50, −0.0519 ± 0.0107 at the end (−1.24%; seeds −0.0628,
−0.0414, −0.0515). One checkpoint after the graft: `graft_25` −0.0192 ± 0.0019, `graft_50` −0.0161 ±
0.0019, `graft_75` −0.0123 ± 0.0004; `graft_25` then −0.0465 ± 0.0032 after 4.50. At the end: `graft_25`
−0.0550 ± 0.0010 (seeds −0.0539, −0.0551, −0.0558), `graft_50` −0.0358 ± 0.0030, `graft_75` −0.0123 ±
0.0004; against `v9_pages` −0.0031 ± 0.0113, +0.0161 ± 0.0096 and +0.0396 ± 0.0103. (b) Shares, ratio of
the mean differences (seeds 0, 1, 2): 1.060 (0.860, 1.333, 1.084), 0.689 (0.616, 0.869, 0.635), 0.237
(0.203, 0.289, 0.238); the means of the per-seed shares, 1.092, 0.707 and 0.243, are in the CSV and not
drawn. Losses at the end: `ref` 4.1912 ± 0.0076, `v9_pages` 4.1394 ± 0.0083, `graft_25` 4.1363 ±
0.0075, `graft_50` 4.1555 ± 0.0091, `graft_75` 4.1789 ± 0.0074. Grafts at steps 366, 732 and 1,098
(1,499,136, 2,998,272 and 4,497,408 tokens); the last checkpoint is at 5,996,544 tokens.

**Checks** (printed on every rebuild). Before and at their graft, all 18 evaluations of the graft arms
equal `ref`'s of the same seed exactly (loss and every row's cross-entropy); at each of the nine grafts
the held-out loss after equals the loss before, row by row, and the function check's difference is 0.
`graft_report.json`: its 1,013 numbers and 128 names and flags equal the recomputation exactly (largest
difference 0), so the figure and the report agree in every digit.

**How it is drawn.** Before its graft a graft arm is `ref`; drawing that stretch as its own line would put
three lines on the zero line, so the zero line stands for it and says so in a note, and each graft arm's
line starts at its graft point (diamond, at the measured loss right after the graft). There is no
evaluation before 1.5 million tokens, so `v9_pages`'s line starts there. The seeds are drawn beside the
mean (left for `v9_pages`, right for the grafts), so that the seeds of `v9_pages` and `graft_25`, which
overlap at 4.5 and 6.0 million tokens, stay apart. Orange = the page, darker the longer it trained; the
four steps are #494949, #696969, #8a8a8a and #adadad in grey and the zero line is black, so the figure
reads in grey; the arm names at the line ends carry identity as well.

**Assumptions and limits.** As in the caption. (b) uses the ratio of the mean differences, the report's
steadier estimate (its denominator is averaged before dividing); the mean of the per-seed shares is
larger by 0.032, 0.017 and 0.006. The restart arms' marks compare a 3M-token run at its own
end with `ref` at the end of 6M tokens: a different comparison from the rest of (a), which pairs equal
token counts. They are drawn inside (a) if they fall in its range, else in a strip above it at the same
scale, with the axis broken (tested on archives built from the 6M records: both placements, a torn
archive and a missing seed, the last two left out with a message).

---

## Known discrepancies of the second draft, and how the third draft resolved them

Checked against `paper.md` on 24 November, 18:45 UTC (after `REVIEW_2`); the wording of each item as
raised is kept in `lead4_work/FIGURES_v3.md`.

1. **Table 4, FLOP cell** ("20% (66%)", an artefact of a fit that is undefined without `final`).
   Resolved: Table 4 reads "not defined without `final`; 10% (21%) on the five `v9`". `make_data.py` still
   prints this as a `MISMATCH`, because `lead2_work/predictor_test.json` keeps the artefact; it is the one
   expected `MISMATCH` line of a rebuild.
2. **"112–117 k (monitor readings)".** Resolved: Table 2's first row reads "115,394 and 116,718 (two
   monitor readings; runs discarded)" with the start-up benchmark 111,893 beside it, as do §5.2 and Table
   H1. Appendix F.3 quotes `docs/TPU_NOTES.md`'s own "112–117 thousand" and says what is on disk.
3. **§2.2 "the step's own rate".** Resolved: §2.2 defines the logged rate as the trainer computes it and
   gives the achieved-rate gap as at most 0.13% (0.124% at the cut-off 46,350).
4. **§5.1 "we report medians".** Resolved: §5.1 says single pieces are medians and whole steps single
   timings.
5. **The cut-off of the text.** Resolved: the text and the figures use the records to step 46,350 (16:27:21
   UTC), and `paths.py` now takes the cut-off from `lead3_work/numbers.json`.
6. **"±0.025 at 592.5 minutes".** Resolved: withdrawn (CLAIMS C294); not cited.
7. **Table 5's intervals use minutes rounded to 0.1** (±0.0554 and ±0.0657 at 34,166 and 35,294 against
   ±0.0551 and ±0.0654 unrounded; ±0.1714 against ±0.1706 at 46,178). Unchanged and immaterial; the figure
   and the table use the same rounding.
8. **`R/docs/MICRO_GRAFT.md`'s "4.6 to 6.2 s per step".** Not quoted in the paper.

## Sentences of paper.md that should cite the figures

Resolved: every figure of the main text is cited in the running text (`lead3_work/check_refs.py
paper.md`, run again at 18:40 UTC). The list below, written for the second draft's numbering, is kept as
history.

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
- **Figure F12 (micro graft).** `paper.md` (15:12) does not describe the experiment yet (`FACTS_2.md`,
  "Added 16:50 UTC"); cite the figure where its results are given: the paired differences at the end and
  over training (F12a) and the shares recovered (F12b).

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
