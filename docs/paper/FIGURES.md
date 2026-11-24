# Figures: captions, readings and sources

All figures are in `figures/` as SVG (text outlined, so they look the same everywhere) and PNG
(300 dpi), each made by its own script; `bash figures/make_all.sh` rebuilds the data, every figure
and `tables.md` from the sources and re-extracts the newest snapshot of the live run, so the
figures grow with the logs. The series behind each figure are in `data/` (`data/README.md`), which
also serves as the table view of every chart. Numbers below are those of the newest snapshot at the
time of writing (last step record 24 Nov 11:01:59 UTC, step 32,100); a re-run changes the live-run
numbers, not the method. Times are UTC.

**Conventions.** One font (Inter; ≥ 6.5 pt at the printed size), one palette: blue `#2a78d6` is
always `configs/final.py`, orange `#eb6834` always `configs/v9.py`, gray is context; the pair was
checked with the palette validator of the dataviz method against a white surface (adjacent
colour-vision-deficiency ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1). F5 uses violet `#6250d6`
against gray for the one rule it is about. "Tokens" are token positions of the training batches,
262,144 per optimiser step (16 micro-batches of 16 × 1,024 or 8 × 2,048, padding included).
Intended print widths: F1, F2, F5, F6, F7 span both columns (7.0 in); F3, F4, F8 one column
(3.4 in).

**The interventions** (numbered alike in F1, F2 and the tables; stop and restore times from the
run's own train log, `data/interventions.csv`):

| # | step | tokens | stopped → restored | what |
|---|---|---|---|---|
| 1 | 5,075 | 1.33 × 10⁹ | runtime 2 lost about 19:29 (23 Nov) → 01:54:14 (24 Nov) | resume on runtime 3 from the step-5,075 checkpoint; the steps after 5,075 trained on runtime 2 (about 2–3 min) were lost |
| 2 | 6,647 | 1.74 × 10⁹ | 02:26:02 → 02:28:59 | stock-refusal rule (commit 566517a) |
| M1 | 8,659 | | 03:09:43 → 03:15:37 | pause: micro-batch pieces timed |
| 3 | 9,332 | 2.45 × 10⁹ | 03:29:17 → 03:31:24 | each micro-batch cut into 4 pieces on the device (b05bfe7) |
| M2 | 10,242 | | 03:48:04 → 03:57:28 | pause: VMEM-limit flags timed |
| 4 | 12,062 | 3.16 × 10⁹ | 04:30:39 → 04:32:03 | post-training plan cut to 118 min, training end moved from 16:51:48 to 18:33:49 (782be85) |
| 5 | 22,613 | 5.93 × 10⁹ | 07:44:14 → 07:53:35 | one-tile attention kernel + hand-written ReLU² backward (6bf9b6e, d9f0ce4) |
| 6 | 25,280 | 6.63 × 10⁹ | 08:35:05 → 08:36:31 | deflection rule (4f4ef45) |
| 7 | 29,722 | 7.79 × 10⁹ | 09:45:34 → 09:58:45 | graft to v9 (9a4d652) |
| 8 | 31,912 | 8.37 × 10⁹ | 10:52:36 → 10:57:50 | v9 speed switches flat, rows, tokn (769f240) |

---

## F1. Training loss, interventions and schedule (`F1_loss.svg`, `fig1_loss.py`)

**Caption.** Training loss of run main16 against training tokens. (a) Whole run on logarithmic
tokens; (b) from 10⁹ tokens on, linear. Dots: the loss of one optimiser step (weighted
cross-entropy of the output law on its 262,144-token batch), logged every 50th step; lines: a
Gaussian kernel average over log₁₀(tokens) with σ = 0.03 decades, computed separately for final
(blue) and v9 (orange) and drawn where the kernel holds at least four effective points. Vertical
lines and numbers: the interventions of the table above. (c, d) The learning-rate factor (black)
and the wall-clock progress p = T/(T + R) (gray) as logged; horizontal hairlines mark p = 0.67
(rows of 2,048 tokens) and p = 0.92 (chat-heavy tail), both still to come.

**Reading.** The loss of a single step falls from 4.86 at step 50 to about 2.5: the smoothed line
reads 2.89 at 3 × 10⁸ tokens, 2.69 at 10⁹, 2.58 at 3 × 10⁹, 2.52 at 5.9 × 10⁹ and 2.497 at the
graft (7.79 × 10⁹). None of the eight interventions shows as a step in the training loss at this
resolution; one step's loss scatters with a standard deviation of about 0.04 nats, so steps
smaller than about 0.01 cannot be seen here. Across the graft the training loss is level: the last
21 logged final steps average 2.492 (sd 0.039), the first 21 v9 steps 2.497 (sd 0.045). The
training loss cannot show whether v9 helps, because the data stream changed at the graft (new seed,
no library rows); the held-out loss of F8 is the measure for that. The schedule panels show the two places where p is not continuous: at 4 the training end
was moved 1 h 42 min later, so p dropped from 0.2413 to 0.2196 and the factor rose from 0.783 to
0.802; at the graft p moved from 0.5045 (last final record) to 0.5118 (first v9 record) because
the 13-minute pause shortened the remaining time R while T stood still.

**Sources.** `data/train_steps.csv` (train logs of `snap_final_main16` and `snap_main`),
`data/interventions.csv`, `data/f1_loss_smoothed.csv` (plotted line). Factor definition:
`bes/train/schedule.py` `lr_factor` (warm-up 0.002, linear decay to 0.1 over all of p).

**Assumptions and limits.** The smoothing width is a choice; the trainer's own average (`loss_ema`,
decay 0.99 per step, in the CSV) lags and is not used. Early steps are sparse on a log axis, so
the line starts at step 850. The training loss mixes 22 sources whose shares change over p and
with the rules; it is not comparable across the graft. Near the graft the v9 line is a one-sided
average that already includes later, lower points, so it starts below the level of its first
records.

## F2. Throughput against wall-clock time (`F2_throughput.svg`, `fig2_throughput.py`)

**Caption.** Training throughput of run main16 against UTC time on the three runtimes, one time
scale, axis broken where no runtime was held (12 h 12 min 23 s and 5 h 35 min 55 s). Lines: the
achieved rate between consecutive step records (tokens ÷ wall time, checkpoint writes included,
intervals across a restart excluded); diamonds: start-up benchmarks; dots: single log lines of
runtime 1. Light bands: TPU held but not training (stops, restores, measurements); darker bands: no
TPU. Numbers on the lines: median rate of each stretch in 10³ tokens/s. Numbered tags: restore
times of the interventions (table above).

**Reading.** On runtime 1 the code before the rewrite trained at 111,893 tokens/s (start-up
benchmark; 115,394 and 116,718 in two monitor readings of its training), and the glue rewrite
brought it to 201,877–204,006 (start-up benchmarks) and 204,699 (its first training log line). The run itself went 221.9k (runtime 2, 16-row micro-batches) → 216.7k (runtime 3,
same program, 2.3% slower, not investigated) → 240.0k (4 pieces per micro-batch) → 281.2k (one-tile
kernel) → 178.0k (graft to v9, 0.633 of final) → 196.9k (v9 speed switches; 3 readings so far).
Before training resumed, runtime 3 spent 48 min 19 s restoring and fixing two data errors that only
a new machine shows; the stops after step 5,075 add up to 50 min 59 s (`tables.md` T5b).

**Sources.** `data/train_steps.csv` (`interval_tok_per_s`), `data/f2_segments.csv` (printed
medians), `data/throughput_log_points.csv` (runtime 1 and benchmarks: `v9.log`, `final.log`,
`colab_bes24h_final.log`, `docs/tpu_profile/final_b.run.log`, run.log of main16),
`data/run_events.csv` (stops and restores), `data/tpu_intervals.csv` (`tpu_intervals.txt`); runtime-1
measurement windows 03:34–03:57 and 04:08–04:17 from `docs/CHANGES.md` 19.1; runtime 2's training
start 17:46:31 = t0 of `train/run.json`.

**Assumptions and limits.** Runtime 2 is shaded from 17:13:00, the earliest time it can have been
assigned (the ledger's convention; it was found already assigned at 17:21). The achieved rate is
slightly below the trainer's own estimate where checkpoints are written; both are in the CSV. The
v9 rate with the switches rests on 3 interval readings at the time of writing.

## F3. Attention-kernel candidates on the chip (`F3_kernels.svg`, `fig3_kernels.py`)

**Caption.** Time of one value-and-gradient of a piece of 4,096 tokens (4 × 1,024 left, 2 × 2,048
right) of configs/final.py on the TPU v6e, eight kernel configurations measured in one pause of the
run (24 Nov 07:44–07:52). Dots: median of 30 timings; whiskers (mostly hidden by the dots): their
minimum and maximum; right column: speed-up against the configured splash kernel (vertical line).
Blue: the configuration adopted at step 22,613. save: forward kernel output kept through
rematerialisation; relu2, ce: hand-written backward of ReLU² and of the cross-entropy head; io:
splash accumulators in VMEM. Equality against the configured kernel (T3a): loss difference 0 at
1,024 and −9.5 × 10⁻⁷ at 2,048 for the one-tile variants; worst relative gradient difference over
parameter leaves 2.8 × 10⁻² (maximum) and 2.6 × 10⁻² (L2) at 1,024, 7.9 × 10⁻² and 5.7 × 10⁻² at
2,048 for the adopted configuration; splash + save, which changes only the rematerialisation of
the same kernel, gives 1.4 × 10⁻² / 1.2 × 10⁻² and 4.7 × 10⁻² / 3.3 × 10⁻²; only io is exact (0).

**Reading.** The one-tile kernel accounts for most of the gain (×1.16 / ×1.13), saving its output
through rematerialisation adds to it at 2,048 (×1.20), and the ReLU² backward brings the adopted
configuration to ×1.212 / ×1.219 (14.639 and 16.275 ms). The cross-entropy backward is slower at
1,024 and faster at 2,048 by less than the spread, so it stayed off. The gradient differences of
the new kernel lie in the band of graph changes that are exact in float32 on the CPU (relu2 + ce:
1.1 × 10⁻² / 3.9 × 10⁻²), i.e. bfloat16 rounding moves with the shape of the graph (`docs/CHANGES.md`
19.6); the logic was checked against float32 and float64 references on the CPU
(`tests/model/test_attention_onetile.py`), not on the chip.

**Sources.** `data/chip_kernels_0745.csv` from `wf/chip_kernels_0745.txt` (the `KERNELS` JSON line;
medians recomputed from the 30 timings agree with the reported `ms` to 10⁻³ ms, column
`median_check`).

## F4. Cost of the v9 step against final's (`F4_v9cost.svg`, `fig4_v9cost.py`)

**Caption.** Wall time of one accumulation step of 262,144 tokens on a synthetic batch, by
configuration, table-gradient mode and piece size, timed on the chip; ratios against final's
fastest step of the same length. Black ticks and "live": median step time of the run itself
(262,144 ÷ achieved rate), which includes the optimiser update and real data. final was timed
with the hand-written cross-entropy backward on (07:44), which the run does not use (its piece is
0.177 ms slower per piece at 1,024 with it; T3a). v9 rows at 09:45–09:58 (the switch) and 10:53–10:56
(the three v9 switches, each added to the previous).

**Reading.** At the switch a v9 step cost 1,433.1 ms against final's 884.5 ms (×1.62; live ×1.58,
1,472.6 against 932.2 ms). The table-gradient mode configured for v9 (`sorted`) was the slowest
(1,789.0 ms, ×2.02), `onehot` 1,484.8 ms; pieces of 8 rows (1,491.5) or, at 2,048, of 4 rows
(1,551.6 against 1,510.2) were slower, so `take` with 4- and 2-row pieces was kept. The three
switches brought the synthetic step to 1,406.4 (flat), 1,295.2 (+ rows) and 1,287.1 ms (+ tokn),
×1.46 of final; live 1,331.4 ms (3 readings). The cost model had predicted −64 ms for tokn; the chip
gave −8 ms (`docs/CHANGES.md` 19.7).

**Sources.** `data/chip_whole_steps.csv` (`wf/chip_kernels_0745.txt`, `v9_measA.txt`,
`v9_measB_sorted.txt`, `v9_measC_onehot.txt`, `v9_measD.txt`), `data/train_steps.csv` (live).

## F5. Stock openings in the sources and what the loader leaves out (`F5_openings.svg`, `fig5_openings.py`)

**Caption.** (a) Share of each source's assistant turns whose opening (first 44 tokens) matches a
stock pattern, from a scan of the tokenised sources on the runtime (24 Nov 08:20; all segments of
small sources, 40,000 sampled otherwise). Patterns are the scan's regular expressions
(`openings_scan.py`); "stock refusal" is the loader's own refusal function as it stood; "Sure …" is a
control that no rule removes. A turn can match several patterns. (b) Share of each chat source's
conversations that the stock-refusal rule (gray, live since step 6,647) and, in addition, the
deflection rule (violet, live since step 25,280) leave out, measured with the loader's functions
on 30,000 sampled conversations per source (2,260 for smoltalk_everyday, all it has). Black ticks:
share of reads the live loader actually skipped for either rule since 08:36 (final and v9 together).

**Reading.** Stock openings concentrate in a few sources: SmolTalk's system-prompt chats open 10.6%
of assistant turns with an apology and 2.1% with "As an AI …", UltraChat 2.8% with "I don't have …",
Tulu 3 4.1–4.7% with refusals and apologies; the retrieval sources glaive_rag and synth_rag also have
1–2% such openings but are not touched by the deflection rule, which applies to chat sources only
("the passage does not say" is the task there). The deflection rule removes 29.1% of the
system-prompt chats (29.8% of their tokens), 9.1% of UltraChat (7.1% of tokens), 1.2% of Tulu 3, and
under 1% of every other chat source. The live loader's skips agree with the scan within about one
percentage point (system chats 30.4% of 12,392 reads against 29.8% predicted; UltraChat 9.6%
against 9.5%; Tulu 3 6.3% against 6.1%). smoltalk_everyday has no tick: no logged loader block drew
from it (its epoch cap cut its planned share from 0.0033 to 0.0003; run.log).

**Sources.** `data/openings_scan.csv` (`openings.json`), `data/deflection_check2.csv`
(`deflection_check2.json`, which matches `deflection_out.txt` and the numbers of commit 4f4ef45),
`data/loader_drops_live.csv` (train logs).

**Assumptions and limits.** The check ran a copy of the rule uploaded to the runtime
(`/content/bes_cli/exact_score_new.py`); that it is byte-identical to `stock_deflection` of commit
4f4ef45 is not verified here, but the live loader (which runs the committed code) skips the same
shares. The comment above `DROP_STOCK_DEFLECTIONS` in `bes/data/loader.py` gives older figures (7%
of UltraChat, about a fifth of the system chats) than the commit message and the check (9%, 29%);
the check's numbers are used. The effect of the rule on the replies is not measured yet [TBD: decode
study and final blind evaluation].

## F6. The graft (`F6_graft.svg`, `fig6_graft.py`)

**Caption.** Stored numbers of the two configurations at step 29,722, drawn to scale (one bar
height = 10⁸ numbers ≈ 2.55 in). Blue: the 59 parameter leaves both configurations share
(34,531,720 numbers), copied with their optimiser slots; gray: final's written library and
retrieval index (63,768,391), which v9 does not have; orange: v9's five new leaves (63,504,384) at
their initialisation (gain/o and page/val zero, page/key normal, page/pin and page/pout uniform),
with fresh optimiser slots. Right: what else was carried, what is new, and the check.

**Reading.** The graft keeps every trained number of final's network and replaces the library's
63.8M stored numbers with 63.5M token-page numbers, so both configurations stay at about 98M stored
numbers. Because v9's values and gains start at zero, the grafted network computes the same function
as the source: the loss of the source and of the grafted parameters on a synthetic batch of 2 × 256
tokens was identical (difference 0.0) at steps 26,890 (11.2808027) and 28,501 (11.3348608), and,
per `docs/CHANGES.md` 19.7, at 29,722. The step, the optimiser's update count, the time ledger and
the loss guard continue, so the learning-rate schedule continues (p 0.5049 at the stop, 0.5112 at
the restore). New: a data stream with another seed and no library rows; the loader counts the
7.79 × 10⁹ tokens before the graft, so its horizon estimate is not 40 times too small and a source
may be read 8 × (1 − 0.5049) = 3.96 more times.

**Sources.** `data/ledger_leaves.csv` (manifests of `ckpt-000026-step-000029722` and
`ckpt-000001-step-000030742`), `data/ledger_final_export.csv`, `data/run_events.csv`; live run
manifest `graft` record (`_src/live/run/manifest.json`: checkpoint, p 0.504884, 7,791,443,968
tokens); `graft_test.log` (26,890) and `graft_dry_out.txt` (28,501) in the old scratchpad;
`runlog_peek.txt` (the loader's "at most 3.96093 reads"); `docs/CHANGES.md` 19.7.

**Assumptions and limits.** The raw output of the check at step 29,722 is not among the files read
for this figure: the graft output kept from 09:48 (`graft_out.txt`) was made without `--check`, into
a directory named `main16_v9_sorted` with configuration hash f1243cb29610103a, while the live run's
checkpoints carry f29bfeb3ebf353a8 (the hash of the 26,890 and 28,501 checks). The statement for
29,722 rests on `docs/CHANGES.md` 19.7.

## F7. The model at one token position (`F7_model.svg`, `fig7_model.py`)

**Caption.** Schematic of BES-100M-A20M at one position, with stored numbers / numbers read per
generated token (standard reading of `docs/ACCOUNTING.md`) for each part. Gray outlines: shared by
both configurations; blue: final only (written library); orange: v9 only (attention-output gains
and private token neurons in each block). The looped core applies two weight-shared blocks eight
times; the per-loop value embedding enters the first block; two hyper-connection lanes carry the
state between loops; the output law adds the masses of 16 candidate next tokens to the network's
distribution and renormalises in closed form.

**Reading.** About 19M of the numbers are read for every token in both configurations (the dense
core 14.1M and the head 4.6M), the rest of the 98M are rows of tables of which one token reads a
few: final spends 63.8M on a written library read only when retrieval fires (worst case 245,223
numbers), v9 spends 63.5M on token pages of which every token reads 597,504 numbers (its 1,536 gains,
its 6,144 neuron keys and values, and the 589,824 numbers of the two dense projections). Totals: final 98,300,111 stored, 19,278,959 read; v9 98,036,104 stored,
19,631,240 read.

**Sources.** Architecture: `docs/SPEC_MODEL_TRAIN.md` 4.1–4.3, `docs/V9.md` 4, `bes/model/model.py`
(`init_params`, `block`, `forward`), `bes/config.py` (d_model 768, 2 blocks, 8 loops, 6 heads of 128
with 2 bias dimensions, feed-forward 3,072, head rank 512, softcap 15, 2 lanes, `CAND_SLOTS` = 16,
15 register fields). Numbers: `data/ledger_leaves.csv`, `data/ledger_final_export.csv`.

**Assumptions and limits.** A schematic, not a data figure: the decoder (6 particles, exact
potentials) is noted, not drawn, and the setting used in the evaluation is chosen after training
[TBD].

## F8 (additional). Held-out loss across the graft (`F8_heldout.svg`, `fig8_heldout.py`)

**Caption.** Held-out loss of the checkpoints around the graft against training time of the run
(minutes of the kept trajectory): web 64 rows, chat 64 rows, proxy prompts 16 rows of the export
check, float32, dense attention, evaluated on the runtime's host cores while training ran
(`scripts/midrun_eval.py`). The gray line is the least-squares line through final's three points,
extended past the graft: an extrapolation, not a measurement.

**Reading.** final's held-out loss fell 0.0081 in the last 44 minutes before the graft (2.2679 →
2.2598); 53.8 minutes of v9 training later it is 2.2337, 0.0158 below final's extended line (0.016
in `docs/CHANGES.md` 19.8, which extrapolates the end-to-end slope). The data stream changed at the
graft as well, the comparison line is an extrapolation of three points, and the set has 144 rows;
the gain cannot be attributed to the token pages alone. More v9 checkpoints are evaluated
automatically (`heldout_v9.jsonl`) and enter the figure on a re-run.

**Sources.** `data/heldout_loss.csv` (`mideval_last.log`; `heldout_v9.jsonl`).

---

## Not made, or made differently from the request

- F3's "equality metrics in the caption" are summarised above and given in full in `tables.md` T3a.
- No dark-mode or interactive versions: these are print figures; the CSVs are the table view.
- F1 uses two x scales (logarithmic for the whole run, linear from 10⁹ tokens) because on the
  logarithmic axis the eight interventions fall within the last 0.8 decade.
- F8 was not requested; it is the only measure in the record that compares the configurations
  across the graft and is offered for the training section.
