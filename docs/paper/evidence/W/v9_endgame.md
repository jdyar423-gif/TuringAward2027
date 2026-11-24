# v9 endgame: do the stages after training work for a v9 run?

## Verdict

**Yes, as they are. No code change was needed.** On CPU, a tiny model with the v9 features
(token neurons, output gains, no library) went through the whole `scripts/run_24h.py` pipeline:
train, sample, prefopt, select, export. Then `scripts/gen_test_replies.py` ran on the export.
Every stage finished, or was skipped because the post plan gave it 0 minutes, and the run exited
with code 0. The export decodes to the training model's next-token distribution, and the token
neurons and output gains are part of that comparison.

I found no blocker. The patch adds one test and changes no code. Two things are not covered by
this rehearsal: anything that only exists on the TPU, and the new run directory's per-run files
(section 4).

## 1. Rehearsal (CPU, cores 2 and 3, `HF_HUB_OFFLINE=1`, 17-minute budget)

The command:

`scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 17` with these
overrides:
- `model.tok_neurons=8 model.tok_dim=32 model.gain_o=true`
- `data.use_library=false data.library_frac_base=0 data.library_frac_chat=0 decode.use_library=false`
- `model.table_grad=take`
- `run.post_minutes=sample=1 critic=0 prefopt=3 select=4 judge=0 export=1.5 buffer=0.5`
- `run.post_reserve_hours=0.16667`
- `run.delete_raw_after_tokenize=false`

The post plan has the same shape as the live plan (critic and judge at 0). Run directory and log:
`wf/v9_endgame_work/run`, `wf/v9_endgame_work/rehearsal.log`.

The tiny model has 13,855,396 numbers, of which 2,097,152 are `gain` and 8,404,992 are `page`.

| stage | result | numbers (from the log, stages.json, manifest.json, export.json) |
|---|---|---|
| background | passed | library task wrote its marker with no library (path None); gguf, opponents, tokenize finished |
| benchmark / canary | passed | xla plans only (splash skipped on CPU); chosen xla-remat0-ms8-take; canary parity and determinism ok |
| train | passed | 38 steps in 171 s (loss 8.61 at step 5); 4 checkpoints, the last at step 38 |
| sample | passed | 13 prompts, about 13 pairs, 62 s; no passage (`library: null`) |
| critic | skipped | "planned with 0 minutes (run.post_minutes)" |
| prefopt | passed | DPO on 8 + 4 pairs; lr 1e-6 and 3e-6 both accepted, 1e-6 kept; post-DPO weights saved; 92 s. Plain Adam updates every leaf, page/* and gain/* included |
| select | passed | variants default, k1, noband (no `nolib`: no library); 4 dev prompts each; chosen default (scores 0.75 / 0.0 / 0.75); library "not shipped: no library was built"; no factual measurement; 24 s |
| judge | skipped | "planned with 0 minutes (run.post_minutes)" |
| export | passed | 13 s; held-out 16-bit check loss32 7.06487, loss16 7.06483, abs delta 3.7e-05 < 0.002; `errors: []`; library mode none; the accounting has gain 2,097,152 / 256 and page 8,404,992 / 17,408; the card names the pages and gains and says "No written library ships with it." |
| gen_test_replies | passed | 2 workers x 1 core, 3 test prompts, 3/3 complete, 57 s wall; elastic cache calibrated bitwise on [256, 512] slots; worker max RSS 613 MB; compile 41-51 s |

## 2. Does the export decode like the training model?

**Measured on the rehearsal's export.** The script is `wf/v9_endgame_work/check_export_decode.py`
and its log is `check_export_decode.log`. The export loaded with `bes.export.load_export` has
the switches `gain=1` and `page=1`.

- **Stored leaves.** Every stored page and gain leaf equals the float16 rounding of the prefopt
  checkpoint's value. These leaves are non-zero after training:
  - `page/key`: 4,194,304 of 4,194,304
  - `page/val`: 2,789,856 of 4,194,304
  - `gain/o`: 1,850,616 of 2,097,152
- **Incremental decode against the full forward, same exported weights.** `OurLM` over the
  exported weights (prefill, 6 greedy steps, one reorder, 3 particles, 7 rows each) against the
  full forward `bes.model.model.logits_fn` over the same weights: max |d log p| = 2.2e-06.
- **Against the float32 training forward.** The same decode against the training forward over
  the float32 checkpoint: max |d log p| = 4.1e-04, max KL = 2.0e-07 nats.
- **The pages and gains matter.** Turning the two switches off changes the logits by up to 0.50,
  so the comparison does exercise them.

**The export's own check covers pages and gains.** `check_16bit` compares
`bes.model.model.loss_fn` over the whole 16-bit tree against the float32 tree, using the
checkpoint's switches. That includes `page/*` and `gain/*`.

**Tests (all pass at HEAD 4f4ef45 plus my patch):**
- Already in the suite, all passing:
  - `tests/decode/test_decode_lm.py::test_ourlm_with_the_v9_paths_matches_the_full_forward`:
    every v9 path with non-zero pages and gains, incremental against full, with a resampling.
  - `tests/train/test_export.py::test_export_of_a_v9_style_model_states_the_page_ledger`.
  - `tests/model/test_generate.py -k v9`: 16 passed, including bfloat16 decoding with the v9
    paths and the configs/v9.py allocation.
- New test:
  `tests/train/test_export.py::test_a_v9_export_without_library_decodes_to_the_training_models_distribution`
  (39 s). A v9-style model with non-zero pages and gains and no library is exported and loaded
  back. The test asserts:
  - the export's decoder matches the full forward over the same weights (atol 1e-4);
  - it is within 0.1 of the float32 training forward;
  - it differs by more than 0.1 with the switches off;
  - the record has no failed accounting, read-set or consts part, ships no library, carries
    `extra.use_library=false`, and its generator replies with `library: None`.

## 3. Places that assume a library or touch pages and gains, by stage (code reading)

- **sample** (`run_24h.py` stage_sample): it retrieves only when `decode.use_library` is on. For
  v9 it uses no library and no passage. OK.
- **critic and judge:** 0 minutes in the plan, so they are skipped with a reason (rehearsed).
- **prefopt** (`bes/train/prefopt.py`): it uses one Adam group for every trainable leaf, and
  pages and gains are included. Nothing in it concerns the library.
  - With `optim.micro_seqs=16` at T=1024, a DPO micro-batch is 8 pairs = 16 x 1024 rows. That is
    the shape of v9's unsplit training micro-batch, which already ran on the chip. That it fits
    is my inference, not a measurement.
  - The extra HBM for the Adam moments is 2 x 392 MB = 784 MB for v9, against 276 MB for final.
    This is computed from the parameter counts.
  - `micro_split` applies to training only.
- **select:** `has_lib` is False because the library marker's path is None. So the stage has no
  `nolib` arm, no factual measurement, and records `ship: False`.
  - The reason it records is "no library was built". The branch "decode.use_library is off"
    comes after it and is never reached. This is cosmetic.
- **export** (`bes/export.py`): the marker's path is None and both `use_library` flags are off,
  so the library is None. The decision `ship: False` shows up in the card and the disclosures.
  - weights16 holds every page and gain leaf. For the real v9 tree that is about 196 MB, against
    69 MB for final (98,036,104 against 34,531,720 numbers, from `jax.eval_shape` of
    `init_params`). A float32 copy of 392 MB is added only if the check fails.
  - The card's "within both" caps line uses the standard reading, which is 19,631,240 for v9
    (docs/V9.md). The read-set table also prints the strict reading, which is above 20M for v9
    (V9.md: 24.5M at T=1024). That is disclosure, not a failure.
  - The fallback export's `_accounting` uses `bes.budget.count`, which has the gain and page
    rows.
- **decode** (`bes/model/generate.py`, `bes/decode/lm.py`): the token-neuron and gain rows are
  computed from the chunk's own tokens at every step, through the same `M.token_rows`,
  `M.gain_rows` and `M.block_gain` as the forward. There is no cache state of its own.
- **gen_test_replies.py:** it appends `extra.use_library=false` when the export has no library.
  - The entry is duplicated, because the decode choice already has it. This is harmless: the
    settings hash is d1f3ffb44e, the same as the select stage's default.
  - The elastic cache's `first_slots` slices only `k`, `v`, `seg`, `pos`, `bos` and `sel`. v9
    adds none of these.
  - Per-worker memory grows by about 4 x (98.0M - 34.5M) = 254 MB over final's float32 tree.
    This is an estimate; it was not measured with v9-sized weights.
- **snapshot** (`colab_tpu.snapshot_members`): it takes generic files and the newest `ckpt-*`.
  Nothing is v9-specific, but the checkpoint grows to about 392 MB of parameters plus their
  optimiser state. This is an estimate.
- **GGUF/opponents tasks:** these are independent of our model. With `tok_root` and `proxy_root`
  at "" (the final and v9 defaults), a new run directory on the same machine reuses
  `data_cache/tok` and `data_cache/proxy`, so the opponents' replies are already cached.

## 4. Not reached, and reminders

- **Not rehearsed (TPU only):**
  - the splash and onetile kernels and `relu2=1` during prefopt;
  - bfloat16 activations in sample and select on the chip;
  - HBM use of the DPO step;
  - export time for the 98M tree. My estimate is a minute or two, against the 5-minute slice:
    the CPU export of the 13.9M-number model took 13 s.
- **Not covered here:** the graft itself, the critic and judge stages (0 minutes in the plan),
  and the slow kill-and-resume rehearsal.
- **The new run directory starts without the per-run files.** `<run>/fast`,
  `<run>/micro_split` and `<run>/post_plan` are read from the new directory, or from its
  manifest. Write them before the start (a new run is refused only when `manifest.json`
  exists).
- **`post_plan` must name every stage.** It replaces only the entries it gives.
  - `sample=3 prefopt=60 select=35 export=5` alone keeps `critic=10 judge=30 buffer=60`, which
    is a 203-minute reserve.
  - With no post_plan at all, the reserve is the configuration's 4.5 hours.
  - Example: `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=<B>`.

## Files

- Patch: `wf/v9_endgame.patch` (git diff of the worktree). It adds one test to
  `tests/train/test_export.py` and changes no code.
- Rehearsal: `wf/v9_endgame_work/` (`rehearsal.log`, `run/`, `testgen/`, `testgen.log`,
  `check_export_decode.py`, `check_export_decode.log`).
