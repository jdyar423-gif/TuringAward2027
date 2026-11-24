# paper/data: the series behind the figures and tables

Every CSV here is written by `paper/figures/make_data.py` (or, where noted, by a figure script) from
the sources named below; nothing is typed by hand. Rebuild everything with
`bash paper/figures/make_all.sh` (it re-extracts the newest snapshot of the live run first). Times
are UTC. "tokens" = optimiser step × 262,144 (token positions of the training batches, padding
included; `optim.tokens_per_step`, and the run's ledger: 7,791,443,968 tokens = 29,722 steps ×
262,144).

`_src/` holds the record files extracted from the snapshots by `figures/extract_records.py` (train
log, run log, manifests, ledgers, the checkpoints' manifests; no weights): `_src/final` from
`snap_final_main16` (the stopped final run, step 29,722), `_src/live` from `snap_main` (the newest
snapshot of the live v9 run), `_src/step5075` from `snap_keep_step5075` (not used by the scripts).

| file | rows | source | columns worth knowing |
|---|---|---|---|
| `train_steps.csv` | one per logged optimiser step (every 50th) of final and v9 | `_src/*/run/train/train_log.jsonl` | `loss` (one step's weighted cross-entropy), `loss_ema` (the trainer's guard average, decay 0.99), `lr` (learning-rate factor), `p` (wall-clock progress), `tok_per_s` (trainer estimate from a 0.8/0.2 moving average of step times), `interval_tok_per_s` (achieved: tokens between this and the previous record ÷ wall time; empty after a restart) |
| `run_events.csv` | checkpoint, stop, restore, set-up and stage events | same | `lost_s` (restore), `compile_s`, `train_end_utc` (set-up) |
| `interventions.csv` | the curated interventions (`figures/interventions.py`) with stop and restore times from the train log; a restore that matches no curated entry appears as `U` | same | `stopped_s` = restore time − stop time |
| `tpu_intervals.csv` | the three runtimes | `tpu_intervals.txt` (the operator's TPU-time ledger) | an open interval is counted until the newest step record |
| `chip_kernels_0745.csv` | 8 kernel configurations × 2 lengths | `wf/chip_kernels_0745.txt` (KERNELS line) | `ms` = median of 30 timings, `min`, `max`, `q1`, `q3`, equality metrics |
| `chip_v9_pieces.csv` | v9 pieces | `v9_measA.txt`, `v9_measB_sorted.txt`, `v9_measC_onehot.txt`, `v9_measD.txt` | 6 timings each |
| `chip_whole_steps.csv` | whole accumulation steps of 262,144 tokens | the same files and `micro_out.txt` | |
| `chip_vmem_flags.csv` | the VMEM-limit measurement | `flags_out.txt`, `flags_out2.txt` | |
| `chip_tune_2311.csv` | the tune of 23 Nov | `docs/tpu_profile/tpu_tune.v6e.txt` | |
| `throughput_log_points.csv` | start-up benchmarks and single log lines of runtimes 1 and 2 | `v9.log`, `final.log`, `colab_bes24h_final.log` (old scratchpad), `docs/tpu_profile/final_b.run.log`, `_src/final/run.log` | |
| `openings_scan.csv` | one per source | `openings.json` (scan of 24 Nov, 08:20–08:24, `openings_scan.py`) | `<pattern>_pct_turns` = share of the source's assistant turns whose opening matches the pattern |
| `deflection_check2.csv` | one per chat source | `deflection_check2.json` (08:31, `deflection_check.py`, 30,000 sampled conversations per source) | conversations the loader's refusal and deflection functions skip, and their token shares |
| `loader_drops_live.csv` | per period and source | loader-block counters in the train logs | `reads` = successful draws + every skipped read; per (config, block) only the snapshot with the largest `last` is used |
| `ledger_leaves.csv` | one per parameter leaf | the two newest checkpoints' `manifest.json` + the row and read rules of `bes/budget.py` | `read_per_step` (standard reading), `optimiser_slots` |
| `ledger_final_export.csv` | ledger rows of final's export | `_src/final/run/final/export.json` (`accounting`) | library and index |
| `heldout_loss.csv` | held-out loss at checkpoints | `mideval_last.log` (old scratchpad), newer v9 checkpoints from `heldout_v9.jsonl` | `train_minutes` = training time of the kept trajectory |
| `f1_loss_smoothed.csv` | written by `fig1_loss.py` | `train_steps.csv` | the plotted smoothed series and its kernel |
| `f2_segments.csv` | written by `fig2_throughput.py` | `train_steps.csv` | the medians printed on F2 |

Full paths of the sources are in `figures/paths.py`.
