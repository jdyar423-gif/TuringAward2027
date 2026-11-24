# graft_run.py review (sceptic)

## Verdict: safe to use as planned. No blocker found; scripts/graft_run.py is unchanged.

This review and its tests cover the main checkout's current graft_run.py, including the late edit
that adds `graft.p`. That edit only adds a manifest field, and all 4 tests pass on it again (102 s).

The resume path accepts the grafted directory. Cell 8 runs `run_24h.py --out RUN_DIR --resume` with no
--config and no overrides, so the manifest checks pass. The hashes, the structure check, the step,
the update count, the slots, the ledger, the guard and cursor 0 all match, and a trainer built here
restores the graft and takes a real step (see the tests below). Two things for the operator first:

1. **MEDIUM, before the `mv`: final is left only on the runtime's local disk.** The grafted manifest
   keeps the source's `created`. The Drive mirror (MIRROR_CODE, started again by cell 8) uses that
   field to identify a run, so it treats v9 as the same run and mirrors it into the same Drive
   directory. On its first pass it copies the graft checkpoint, deletes final's last checkpoint from
   Drive, and overwrites train_log.jsonl and loader_journal.jsonl there with the new short files.
   After that, final exists only in /content/runs/main16_final. If you might ever go back to final,
   take a `snapshot` (ctl) of main16 BEFORE the swap, or copy
   main16_final/train/ckpt/<newest> to Drive under another name. Do not "fix" this by giving v9 a
   new `created`: the mirror would then refuse to mirror v9 at all ("holds another run").
2. **MEDIUM, data, silent (not a crash): the loader's epoch caps and horizon restart at p ~ 0.47.**
   The new journal starts at index 0, so `consumed` is 0: every source may be read
   `MixCfg.max_epochs` = 8 times AGAIN, which allows up to 16 reads over the whole run.
   `Loader._horizon` estimates the remaining tokens as `tokens0 / p - tokens0`, which assumes the
   journal began at p = 0. The first block plans over the benchmark's full-run horizon (1.31e10).
   From the second block on, the estimate is about 1.1 x the tokens drawn since the graft at
   p ~ 0.5, which is 2 to 7 times less than the real remaining ~8e9 tokens. Small sources that the
   cap was holding back are therefore drawn at their full target share, front-loaded, and once
   spent fall to the BUDGET_FLOOR share for the rest of the run, which is the later phase where
   chat counts most. Nothing fails; the mixture is not the one planned. A fix needs loader code: offset
   `tokens0` in `_horizon`, and ideally `consumed` too, by the ledger's tokens and the source
   journal's consumption at the graft (manifest `graft.tokens` = 7.05e9). That is not done here.
   Acceptable for this run if the small sources were not cap-bound; otherwise worth a follow-up.

## Findings by severity

* MEDIUM: 1 and 2 above.
* LOW, speed and memory not measured for v9: the carried benchmark plan (`splash, remat,
  micro_seqs=16, table_grad=take`; the command line also has `model.table_grad=take`) and
  `micro_split=4` were timed on final. v9 adds 63.5M stored numbers (about 0.76 GB more HBM for
  parameters and two fp32 slots) plus the activations of the page lookups. The run keeps
  `take`, not v9's `sorted`, unless you pass `--plan-override model.table_grad=sorted`.
  `run_overrides` puts the plan last, so the plan's value wins over the command line's `take`;
  the test checks this. `horizon_tokens` (1.31e10) is final's speed x time. It sets the matrix
  learning rate's data factor (`optim.horizon_tokens`), and keeping it is what keeps the
  learning rate continuous, so keep it.
  If v9 does not fit at micro_seqs=16, the train stage fails at compile time. It is essential,
  so the run would end after an emergency export. Watch the first set-up in the log.
* LOW, guard: the regime key is (shape, assistant_only, phase), and none of these change, so the
  carried guard EMA (final's loss with library rows) is kept and not re-based. Dropping the library rows moves
  the loss level by a few percent at most, far below the 1.3x spike threshold. Keeping the EMA is
  the better choice: it covers the new paths' first steps against a real spike. A rollback would
  go to the graft checkpoint (seq 0, reason "graft", with ledger and guard in its meta); the test
  checks that the trainer loads it under its own hash.
* LOW, numerics are sane:
  * The function is unchanged at the graft. The effective configurations differ only in
    `gain_o`, `tok_neurons`, the library fields, `run.name`, `out_dir` and `seed`. The batch keys
    and consts structure are the same, and the new consts (switches) come from the v9 template.
  * The learning rate is continuous. p comes from the carried ledger. `warmdown_frac=1` puts the
    factor at ~0.53, and there is no warm-up for the new groups (page 0.004, gain 0.01, the embed
    group for page/key 0.3, all x 0.53). The momentum schedule continues with the carried step,
    and the momentum warm-up exists only for "muon", not MuonH.
  * Adam on the new leaves uses the carried count (~26,890), so there is no bias correction. The
    first update is 0.894 lr sign(g), rising to ~1.38x the corrected size around updates 5 to 10
    and back to 1.0x by ~50. A row first touched late in a fresh run behaves exactly the same way.
  * The new MuonH matrices (page/pin, page/pout) get fresh slots with their own cached norm.
* NIT: `graft.from_run` and the checkpoint meta's `grafted_from` name /content/runs/main16, which
  after the swap is the v9 run itself (it should read main16_final). Only informative; nothing
  reads it.
* NIT: graft_run.py resolves the source's relative `config` (configs/final.py) against the CWD,
  through `SV.base_config`. Run it from the repository, as run_24h.py is; anywhere else it fails
  before writing anything.
* Checked and fine:
  * `Run.effective()` hash check, and `stage_train` with use_library False: no library wait, no
    journal shards to wait for, LibraryGate inactive, a new journal, seed 1.
  * `_micro_split` reads the carried manifest and file; `post_plan` and `fast` are carried.
  * Supervisor: the train stage is "interrupted", so it is pending and starts again. There is no
    attempt limit, and the slice is recomputed from the carried stages.json t0/deadline.
  * Checkpoint sequence numbers restart at 0, then `next_seq` = 1, and pruning keeps the newest.
  * The later stages pick `CK.latest` of the new train/ckpt (only v9 checkpoints) and read the
    new ledger.json. The export's config comes from `effective_config(man)`.
  * The snapshot and mirror lists do not include state/ or background.json, as for any run (the
    manifest carries micro_split, post_plan and fast).
  * jax_cache is per run directory and rebuilt (the first set-up compiles everything again).
  * tok_root and the library live in data_cache, outside the run directory, so they survive the
    `mv`.

## Tests (tests/train/test_graft_run.py, 4 tests, 183 s on 2 cores, all pass)

* A fabricated smoke run (complete benchmark, perturbed trained-looking state, count 455, step 456,
  cursor 77, ledger and guard in the meta, an older checkpoint too) is grafted into a v9-like smoke
  configuration written by the test (gain_o, tok_neurons=2, tok_dim=16, no library) with
  `--override run.seed=1 --check`. The test checks:
  * the check's loss difference is <= 1e-6;
  * every source leaf (parameters, slots, count, step, rng) is equal;
  * the 5 new parameter leaves and their slots equal the fresh initialisation, and page/val and
    gain/o are exactly 0;
  * meta: cursor 0, ledger, guard and regime carried;
  * manifest hashes equal `effective_config` / `base_config`, the overrides are merged, the
    horizon is kept, jax_cache is dropped and `created` is kept;
  * the carried files are there, and there is no old journal or log.
* `--plan-override model.table_grad=sorted` changes the plan and the effective hash consistently,
  and refuses other fields.
* The tool refuses, and writes nothing, for: an existing destination, no checkpoint, a checkpoint
  of another configuration hash, and a shape mismatch (ve_dim 64).
* A Trainer of the grafted directory (real model, hash check on, fresh v9-like template) restores
  the graft. It has step 456, cursor 0, the guard and ledger carried, and restarts = 1. It takes a
  finite step at the carried p, with count 456. page/val moves off zero, and checkpoint seq 1 is
  written.

## Not checked

* Anything on the TPU (memory and speed of v9 under final's plan, `fast` kernel switches traced
  with v9's token-neuron and gain paths).
* The real run's state/ markers and background.json contents (not in the local copy).
* The data effect of finding 2, in numbers per source (needs the real journal and mixture sizes).
* The CPU rehearsal in graft_rehearsal (left alone).
