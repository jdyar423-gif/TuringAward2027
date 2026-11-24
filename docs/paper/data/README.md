# paper/data: the series behind the figures and tables

Every CSV here is written by `paper/figures/make_data.py` from the sources named below; nothing is typed
by hand, and derived series (the smoothed loss, the live rates by stage, the held-out trend line, the
leave-one-out predictions) are computed there too, so the CSVs are the table view of every figure.
Rebuild everything, the figures included, with

    bash /tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/figures/make_all.sh

(it reads the newest snapshot of the live run first; `paper/FIGURES.md`, "How to rebuild", has the
options and the full list of inputs; source paths are in `figures/paths.py`). Every CSV holds the run's
records up to one cut-off, `cutoff.csv`: by default the newest step of `paper/lead2_work/numbers.json`
(the records the text's live numbers come from), so the figures and the text agree; newer records of the
snapshot, held-out evaluations of later checkpoints and later loader counters are left out
(`make_all.sh --cutoff newest|STEP` to move it). Times are UTC. "tokens" =
optimiser step × 262,144 (token positions of the training batches, padding included; the run's ledger:
7,791,443,968 tokens = 29,722 steps × 262,144).

`_src/` holds the record files extracted from the snapshots by `figures/extract_records.py` (train log,
run log, manifests, ledgers, the checkpoints' manifests; no weights): `_src/final` from `snap_final_main16`
(the stopped final run, step 29,722), `_src/live` from `snap_main` (the newest snapshot of the live v9 run;
`EXTRACTED.json` names the snapshot and its creation time), `_src/step5075` from `snap_keep_step5075` (not
used by the scripts).

| file | rows | source | columns worth knowing |
|---|---|---|---|
| `cutoff.csv` | one row: the newest record used | the train log; `paper/lead2_work/numbers.json` (`v9.newest_step`) | `step`, `utc`, `kind` (paper, newest, step), `step_records_after_cutoff`, the snapshot's newest record and creation time |
| `train_steps.csv` | one per logged optimiser step (every 50th) of final and v9, to the cut-off | `_src/*/run/train/train_log.jsonl` | `loss` (one step's weighted cross-entropy), `loss_ema` (the trainer's guard average), `lr`, `p`, `seq_len`, `phase` (short, long, tail), `assistant_only`, `tok_per_s` (the rate the trainer logs: 262,144 ÷ its running average of the time per step, weight 0.2 on the newest; time booked to checkpoint writes not counted), `interval_tok_per_s` (achieved: tokens between this and the previous record ÷ wall time, checkpoint writes included; empty after a restart) |
| `run_events.csv` | checkpoint, stop, restore, set-up, stage and end events | same | `phase`, `seq_len` (stage), `lost_s` (restore), `train_end_utc`, `deadline_utc` (set-up) |
| `interventions.csv` | the curated restarts (`figures/interventions.py`) with stop and restore times from the train log, the measurement pauses (M), restarts that match no entry (U), and the schedule's phase changes (P1, P2) from the stage events | same | `stopped_s` = restore − stop; `event_utc` = restore time, or stage time for a phase change |
| `throughput_stages.csv` | Table 2's stages in time order: two benchmark stages before the run, then the live stages of run main16, and Table 3's live window | `train_steps.csv`, `chip_whole_steps.csv`, `throughput_log_points.csv`; checked against `lead2_work/numbers.json` | `live_median`, `live_n` (median of `tok_per_s`), `achieved_median`, the window (`first_step`… `last_utc`), `from_id`, `from_step` (the change that starts the stage), `delta_live_pct`, `synthetic_tok_per_s` and its source |
| `f2_segments.csv` | achieved-rate medians per stage under the stretch names `make_tables.py` reads | `train_steps.csv` | |
| `f1_loss_smoothed.csv` | one per step record | `train_steps.csv`, `run_events.csv` | `stretch` (between restarts and phase changes), `loss_smoothed` (centred average), `kernel_records_effective`, `half_width_decades`, `centre_offset_decades`, `drawn` |
| `heldout_loss.csv` | held-out loss per checkpoint and evaluation, checkpoints to the cut-off | `N/midrun_eval_1.json`, `N/heldout_v9.jsonl` (runtime), `lead2_work/heldout_analysis.json` (local, pages off) | `web_chat` (Table 7's set, weighted by tokens), `all_rows`, weights and row counts per domain, `final_line`, `final_line_pi95_half`, `gap_to_final_line` |
| `heldout_trend.csv` | `final`'s line on training minutes and its 95% prediction interval, every 0.5 min | `heldout_loss.csv` | `fit_*` columns: points, slope, residual sd, degrees of freedom, t |
| `costmodel_steps.csv` | the six configurations of Table 4 | `lead2_work/predictor_test.json` | `loo_<count>_ms`, `loo_<count>_err` (recomputed), `json_loo_<count>_err` (as in the json) |
| `costmodel_differences.csv` | the seven changes of Table 5 | same | `predicted_ms`, `measured_ms`, `ratio`, `sign_agrees` |
| `blind_comparisons.csv` | one per blind-comparison report | `N/study/proxy/reports/ab/<name>.json` (the four dev sets and any `--blind` names) | `wins`, `ties_or_both_bad`, `losses`, `score`, `score_lo`, `score_hi`, `readings`, `our_step`, `our_config`, `split` |
| `graft_facts.csv` | the graft's checks and carried state | the three `--check` logs of Table 6, `lead2_work/graft_grad.json`, the 29,722 checkpoint manifest (guard), the live manifest (`graft`, `run.seed` in `overrides`), the live `run.log` (read cap before and after the graft, speed switches of the start after it) | `kind`: check (`check_tokens`), gradient, guard, manifest (`seed`), loader (`reads_cap`, `reads_cap_before`, `fast_at_graft`) |
| `tpu_intervals.csv` | the three runtimes | `S/tpu_intervals.txt` (the operator's TPU-time ledger) | an open interval is counted until the cut-off |
| `chip_kernels_0745.csv` | 8 kernel configurations × 2 lengths | `N/wf/chip_kernels_0745.txt` (KERNELS line) | `ms` = median of 30 timings, `min`, `max`, `q1`, `q3`, equality metrics |
| `chip_v9_pieces.csv` | v9 pieces | `S/v9_measA.txt`, `S/v9_measB_sorted.txt`, `S/v9_measC_onehot.txt`, `S/v9_measD.txt` | 6 timings each |
| `chip_whole_steps.csv` | whole accumulation steps of 262,144 tokens | the same files and `S/micro_out.txt` | `timing`: each row is one timing, the mean of a few back-to-back steps (`R/scripts/tpu_kernels.py --step-calls`, default 2; `tpu_micro.py --calls`, default 3; the count used is not in the output files) |
| `chip_vmem_flags.csv` | the VMEM-limit measurement | `S/flags_out.txt`, `S/flags_out2.txt` | |
| `chip_tune_2311.csv` | the tune of 23 Nov | `R/docs/tpu_profile/tpu_tune.v6e.txt` | |
| `throughput_log_points.csv` | start-up benchmarks and single log lines of runtimes 1 and 2 | `S/v9.log`, `S/final.log`, `S/colab_bes24h_final.log`, `R/docs/tpu_profile/final_b.run.log`, `_src/final/run.log` | |
| `openings_scan.csv` | one per source | `N/openings.json` (scan of 24 Nov, 08:20–08:24) | `<pattern>_pct_turns` = share of the source's assistant turns whose opening matches the pattern |
| `deflection_check2.csv` | one per chat source | `N/deflection_check2.json` (08:31, 30,000 sampled conversations per source) | conversations the loader's refusal and deflection functions skip, and their token shares |
| `loader_drops_live.csv` | per period and source, to the cut-off | loader-block counters in the train logs | `reads` = successful draws + every skipped read; per (config, block) only the record with the largest `last` is used |
| `ledger_leaves.csv` | one per parameter leaf | the two newest checkpoints' `manifest.json` + the row and read rules of `R/bes/budget.py` | `read_per_step` (standard reading), `optimiser_slots` |
| `ledger_final_export.csv` | ledger rows of final's export | `_src/final/run/final/export.json` (`accounting`) | library and index |

N = `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`, S = the old
scratchpad `/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`, R =
`/home/user/turingaward2027`.
