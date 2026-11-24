# Audit: state that lives only on the machine (audit_machine_state)

Checkout `/home/user/turingaward2027`, branch `claude/modest-carson-yi2d9g`, HEAD `566517a`. Read-only:
nothing in the checkout was changed by this audit (the untracked `scripts/tpu_micro.py`, mtime 03:08:01,
is not from it). Every script, patched copy and diff is under
`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/audit_machine_state/`.

## What a snapshot holds, and what a recycled runtime rebuilds

`scripts/colab_tpu.py:140-144` (`SNAP_ROOT_FILES`, `SNAP_FILES`, `SNAP_TREES`, `SNAP_CKPT_ROOTS`):
`manifest.json`, `stages.json`, `stage_table.json`, `train/{ledger,run}.json`, `train/train_log.jsonl`,
`train/loader_journal.jsonl`, `benchmark/results.jsonl`, the trees `logs/ health/ post/ final/`, and the
newest `ckpt-*` of `train/ckpt`, `train/ckpt_params`, `post/ckpt`.

**Not in it:** `state/` (every ready marker: `library.json`, `opponents.json`, `opponents_progress.json`,
`downloads.json`), `background.json`, `jax_cache/`, and all of `data_cache/` (relative to the code, not
the run directory): `raw/` (downloads and their cached listings), `tok/` (shards), `library/lib_v2`,
`opponents/` (GGUF files) and `proxy/` (`run.proxy_root=""` gives `data_cache/proxy`: the opponents'
replies, every harness cache and every exact, factual and judged report).

On a recycled runtime `main()` restarts the background tasks and continues at the first pending stage
**at once**. Only `stage_train` waits for machine state (library, journal shards: `run_24h.py:572-578`).
No post-training stage waits for anything.

Times below use the live deadline 24 Nov 20:31:49 UTC (19.4) and the run's overrides
(`post_reserve_hours=3.6667`, `post_minutes=critic=0,buffer=20`): train_end 16:51:49; sample to 17:41;
prefopt to 18:41; select to 19:26; judge to 19:56; export to 20:11; buffer to 20:31. The opponents task
works until `end = deadline - (judge + export + buffer) = 19:26:49`. A new runtime moves the deadline by
its downtime, so the windows shift with it.

## Findings (most costly first)

### F1 [blocker] On a recycled runtime the export leaves out the library that the decode selection shipped, and replaces the snapshot's export that had it

- `scripts/run_24h.py:133-137` `library_path()` reads `state/library.json`, which is not in the snapshot.
- `scripts/run_24h.py:1298` passes `"library": self.library_path() or None` to `export_run`.
- `bes/export.py:1734-1736` falls back to `state/library.json` (also absent). `bes/export.py:1745-1748`
  only ever removes a library. `bes/export.py:1993-1999` swaps the new export over `final/`.

**Sequence.** The runtime is lost after select has recorded `decode.library.ship = True`, during
judge, export or the buffer before the export finished (about 19:27–20:12).
1. The new runtime restores the snapshot. The library task starts rebuilding: RUNBOOK 5 gives about
   10 min of host time plus downloads, and it is slower while it shares cores with the catch-up
   tokenizer (F6).
2. The judge skips itself within seconds (F5).
3. `stage_export` starts at once. `library_path()` is `""`, so `export_run(library=None)` runs.
4. Result: `final/export.json` has `library.mode = "none"`, and `final/library` is removed. The card
   says "not shipped; library decision: shipped: …". The snapshot's `final/`, which held the library,
   is replaced.

The weights are right, but the shipped model is not the selected one: the library is counted in the
model's stored numbers and is on in the chosen decode settings.

**Reproduced: yes.** `repro_recycled_post.py`, using the repo's own test helpers, the real
`snapshot_members`, `Run.stage_export` and `export_run`:

```
A: export library mode = copy | decode overrides: ['extra.use_library=true']
snapshot: 39 files, checkpoints {'train/ckpt': 'ckpt-000001-step-000000007'}, state/ in it: False
B: final/ from the snapshot holds a library: True
B: library_path() = ''
B: manifest decode.library.ship = True
B: export library mode = none | final/library exists: False | disclosure: not shipped; library decision: shipped: ...
```

**Fix:** `diffs/F1_export_library.diff`.
- It adds two helpers, `_busy` and `_await_tasks`: a bounded wait that does nothing when the tasks have
  ended.
- The export waits only when all three hold: `ship` is True, the library is not on this disk, and the
  library task of this process is still running. It waits at most until `deadline - export slice`.
- Verified on the patched copy (`verify_fixes.py`, check 2): the export waited for the rebuilding
  library task and shipped it (`library mode copy`).

### F2 [major] On a recycled runtime the decode selection records "no library was built" for good, and sampling builds its whole prompt pool without the library, the held-out conversations or the final raw file

- `scripts/run_24h.py:1068`: `has_lib = bool(self.library_path()) and …` is False while the library task
  rebuilds.
- `scripts/run_24h.py:1134-1135`: `_library_decision` then returns `ship: False, "no library was built"`.
  The "nolib" variant and the factual measurement are not run. Line 1115 writes the result to the
  manifest, so the export never ships the library, even after it is rebuilt.
- `scripts/run_24h.py:768`: sampling uses `lib = self.library_ready()`, which is None.
- `scripts/run_24h.py:783`: the pool is built once, at the stage start, from what is on disk.
  - `heldout_samples` sees only the shards committed so far: the phase-2 chat files arrive after about
    30 GB of ClimbMix.
  - The top-up reads `completed_files(...)[-1]`, the newest *present* file. During the catch-up that is
    file 0, not the last file. After the catch-up it is the same file as on the old machine, because
    `prune_raw` keeps the highest index (verified with the real `prune_raw`, see "checked").
  - There is also a millisecond race: the tokenize task's `prune_raw` can delete that file between
    `completed_files` and `iter_records` (line 733). The read is not guarded, so the stage fails.

**Sequence (select).** The runtime is lost during select (about 18:41–19:26), or late in prefopt.
Select restarts at once on the new runtime. The library task has not finished, so
`decode.library = {"ship": false, "reason": "no library was built"}` is recorded permanently.

**Sequence (sample).** The runtime is lost during sample (16:52–17:41). The restarted stage samples
without passages, from training rows of file 0, instead of held-out conversations.

**Reproduced: yes,** for the decision. `repro_recycled_post.py` on the restored directory:
`B: select's library decision: {'ship': False, 'reason': 'no library was built'}`. The pool composition
was not run on real data.

**Fix:** `diffs/F2_select_sample_library.diff`, on top of F1's helpers.
- Select waits at most 40% of its slice, until the library is on disk.
- Sample waits at most 30% of its slice, for the library and tokenize tasks.

### F3 [major] On a recycled runtime, preference optimisation runs without the UltraFeedback pairs

- `scripts/run_24h.py:967-971`: `uf = …/tok/ultrafeedback; if uf.exists(): pairs_from_shards(uf)`. Nothing
  waits for it.
- UltraFeedback is the last item of the download plan (`bes/train/supervisor.py:1557-1559`). Computed for
  the live config: about 46.4 GB comes before it (ClimbMix 30 GB, then the chat and rag sources,
  FineWeb-Edu 4 GB).
- It is not a loader source: `source_table` has 24 sources and no `ultrafeedback`. So the train stage's
  journal wait and `_catch_up_cores` never cover it.

**Sequence.** The runtime is lost during prefopt (about 17:41–18:41), or late in sample.
1. Prefopt restarts at the restore on the new runtime.
2. The `ultrafeedback` shard is not committed yet, so `groups == {"sampled": …}`: a few hundred pairs
   from about 50 min of sampling. The slice could take up to about 57k pairs
   (`0.6 * slice / 0.0378 s` per pair at 216k tok/s), almost all of them UltraFeedback.
3. The result is a DPO run on almost nothing (most likely rejected by the health checks), or
   `StageSkipped("not enough preference pairs")`.

**Reproduced: no.** This is the code path plus the computed plan order and source table.

**Fix:** `diffs/F3_prefopt_ultrafeedback.diff`, on top of F1. While the tokenize task of this process
runs, prefopt waits at most 25% of its slice.

### F4 [major] A runtime restored shortly before train_end fails its essential train stage instead of skipping the closed window

- `scripts/run_24h.py:155-157` and `183-185`: `_wait_for_library` and `_wait_for_shards` (e096dff)
  raise `RuntimeError` at `hard_deadline - 60 s`.
- `bes/train/supervisor.py:1485-1491`: train is essential, so the supervisor runs an emergency export
  and raises `FatalStageError`.
- `scripts/run_24h.py:1690`: on the way out, `run.close()` sends SIGTERM to every background task
  (library build, tokenizer, opponents).

**Sequence.**
1. The new runtime arrives less than the rebuild time (about 15–25 min, 19.4) before train_end
   (16:51:49 now).
2. The window closes while the train stage waits.
3. Train is recorded as failed. An emergency export is written, without the library (F1). run_24h
   exits with status 1 and the library build in progress is killed.
4. Only an automatic continuation (`BES_AUTO_RESUME`, 2 per runtime) gets the run going again. Train is
   then skipped by `allocate`, after a few minutes and with the library build restarting.

The cost is minutes, one continuation, and a later library on the machine (F1, F2). If no continuation
is left, the run halts.

**Reproduced: yes.** `repro_train_window.py` runs the real `Supervisor` with `Run._wait_for_shards`:

```
train FAILED: RuntimeError: 2 of the 2 shards the training journal used were not tokenised on this machine before the end of the training window
emergency export: train failed: ...
FatalStageError (run_24h exits with status 1)
ledger: {'train': 'failed', 'sample': 'pending', 'prefopt': 'pending', 'select': 'pending', 'export': 'pending'}
```

**Fix:** `diffs/F4_train_window_skip.diff`. Raise `SV.StageSkipped` for the closed window. The library
task failing and the "data differs" error stay `RuntimeError`. Verified (`verify_fixes.py`, check 1):
status `complete`, train `skipped`, and sample, prefopt, select and export ran in the same process.

### F5 [major] Judge: the opponents' replies do not survive a recycle, the judge gives up at once without them, and a restart after their end time erases their progress record

- The replies and their markers are outside the snapshot: `data_cache/proxy` and `state/opponents*.json`.
- `scripts/run_24h.py:1230-1234`: with no marker the judge raises `StageSkipped` at once, and it is
  never retried.
- `bes/train/supervisor.py:1748-1780`: the opponents task restarts from zero (about 105 min of CPU). It
  gives the first arm all of `end - now`, and it shares cores with the 24-worker catch-up tokenizer.
- `bes/train/supervisor.py:1770`: when started again after `end`, the task writes
  `opponents_progress` with `runs={}` before it returns.

**Sequence (a).** The runtime is lost after the opponents finished (about 03:00 on the third runtime).
On the new runtime the GGUF files are downloaded again and the opponents start over. The judge then
compares with whatever was redone by 19:26:49. If it starts before the first arm returns (an earlier
stage skipped, failed or short), it is skipped.

**Sequence (b).** A late-started runtime runs out of opponents' time, so only `opponents_progress` is
written. A supervisor restart after `end` on the same machine (an automatic continuation during judge
or export) rewrites the record as empty. The judge is skipped, although every reply is in the harness
cache on disk.

**Cost:** a report only; the export does not depend on it.

**Reproduced: yes.**
- (a), the immediate skip: `repro_recycled_post.py` gives
  `B: judge skipped: no opponent replies on the test split yet`.
- (b): `repro_opponents_marker.py` gives `orig: … test runs the judge sees now: none -> StageSkipped`,
  and with the fix `fixed: … {'falcon=recommended/test': …}`.

**Fix:**
- `diffs/F5_judge_wait.diff`, on top of F1: the judge waits, at most half its slice, while gguf or
  opponents run, until every arm has a test run. Verified in `verify_fixes.py`, check 3.
- `diffs/F5b_opponents_marker.diff`: the task starts from the earlier progress record.

### F6 [minor] The catch-up tokenizer shares the library build's cores

`scripts/run_24h.py:203-205` takes the top 24 cores. With the 44-core plan that is cores 19..42, which
include the library cores 37 and 38. A resumed training that used the library waits for the library
*first* (`run_24h.py:572-575`), so the library build is on the TPU's critical path at every machine
switch.

On this container's kernel (`sched_autogroup_enabled=1`; each task is started in its own session)
a process sharing one core got 50% of it whatever its nice value (measured: 2.00 s and 2.00 s of CPU in
4 s). So the build runs at about half speed while the catch-up runs.

**Reproduced: yes** for the core overlap and the 50% share. The effect on the TPU host is not measured.

**Fix:** `diffs/F6_catch_up_cores.diff`. Leave `cp.library` out of the spare cores. The tokenizer keeps
24 workers on cores 17..42 minus 37 and 38.

### F7 [minor] The snapshot can pack a record half old, half new

`scripts/colab_tpu.py:744-749`: `tar.add` stats a file, then opens it. Several records are rewritten
atomically:
- `stages.json` by the heartbeat every 10 s;
- `manifest.json`;
- `train/ledger.json` at each checkpoint;
- `health/latest.json` at each side-car report.

**What goes wrong.** A rewrite between the stat and the open packs N bytes of the new file, which is
truncated JSON. If the new file is shorter, `OSError` is swallowed after the header was written and
the archive is misaligned. The restored run then cannot read `stages.json` (`StageLedger` raises at
start), or later members are lost. The window is microseconds against 10 s, so it is unlikely, but the
snapshot is the only copy the continuation has.

**Reproduced: yes,** with the interleaving forced in `repro_snapshot_race.py`:
```
orig shrink: restored {'manifest.json': 'ok', 'stages.json': 'corrupt JSON (JSONDecodeError, 3140 bytes)', 'train/run.json': 'missing'}
orig grow:   restored {'manifest.json': 'ok', 'stages.json': 'corrupt JSON (JSONDecodeError, 3140 bytes)', 'train/run.json': 'ok'}
fixed shrink/grow: all 'ok'
```

**Fix:** `diffs/F7_snapshot_race.diff`. Open the file first, then build the header with
`tar.gettarinfo(fileobj=f)`, so header and data come from one inode. The quick `tests/train/test_colab_cli.py`
tests pass: 33 passed.

### F8 [minor] Three rag sources follow a moving ref on a new machine

`glaive_rag`, `rag_instruct` and `dolly` use `revision="refs/convert/parquet"` (`bes/data/sources.py:610`).
Only a machine's cached `raw/.../listing.*.json` pins the commit (`bes/data/download.py:162-195`), and
it is not in the snapshot. A new machine resolves the ref again. If the conversion moved,
`_missing_shards` raises "the data differs" and train fails on that machine; with equal counts, the
rows differ silently.

Checked today (public Hub metadata): all three refs still resolve to the registry's `verified_sha`.
So pinning to it changes nothing for this run.

**Reproduced: no** (the failure needs the ref to move).

**Fix:** `diffs/F8_convert_refs.diff`. `source_listing` lists `verified_sha` when the revision is not a
commit, and falls back to the ref when that commit is gone. A cached listing (the live runtime) is
read as before. The data tests pass: `tests/data/test_tokenize_worker.py` and `test_sources.py`,
109 passed.

## Checked and fine

- `prune_raw` keeps the highest-index file present.
  - Checked with the real function, files arriving out of order (`sim_prune_newest.py`): present `[0]`,
    then `[3]`, then `[2, 3]`, then `[3]`.
  - After the catch-up on a new machine, the file kept and read by `_sample_prompts` is the same index
    as on the old machine (pinned revisions, deterministic `select_files`). During the catch-up it is
    file 0 (F2).
- `synth` and `synth_rag` share their files. After 77305dc a file goes only when both have committed
  their shard, and the kept file stays. The smoltalk subsets have distinct file patterns, so they are
  not grouped.
- The held-out split is a function of (source, shard, segment). Once a machine has the shards back, the
  same conversations are held out.
- Deadline moves on a second and third resume (`sim_two_moves.py`, the live settings, two moves):
  - manifest, `stages.json` and `train/run.json` are rewritten together and agree; `StageLedger` refuses
    a mismatch;
  - `allocate` recomputes the slices from the moved deadline at each stage start;
  - downtime goes to `down_s` (stage ledger) and `lost` (trainer ledger);
  - p continues: 0.5998 at the checkpoint, 0.6050 at the restore (the 5 min between checkpoint and
    snapshot are lost);
  - `budget_s` and the stage specs stay; the opponents' `end` and `colab_tpu`'s overrun read the moved
    deadline.
- A resumed post-training stage gets its full planned slice again, not `planned - used`. Shedding then
  takes from buffer and judge first. That suits prefopt and select, which restart from zero, and is
  generous for sample, whose done prompts persist.
- `post/samples.jsonl`: the resumed stage skips done ids, and `_samples` ignores a torn last line. At
  worst one record is lost, and only if the snapshot copy hits a write.
- `post/ckpt` (the DPO weights) is used only when `manifest.prefopt.accepted`. A re-run starts from the
  trained weights (`prefer_post=False`) and replaces it (`keep=1`). A rejected re-run makes
  `latest_params` ignore a stale `post/ckpt`.
- `final/`: `export_run` writes a temporary directory and swaps it in, so `final/` is never a mix of
  two exports. The problem in F1 is what gets swapped in.
- Train checkpoints are written to `.tmp-*` and renamed. `snapshot_members` takes the newest `ckpt-*`,
  never a temporary directory. Only the newest survives a switch, so a guard rollback after a restore
  can reach only that checkpoint (by design).
- `jax_cache` is not in the snapshot. It costs compile time only (the benchmark's programs are not
  reused on a resume) and leaves no stale entries from another machine.
- With no `background.json` or `state/downloads.json` on a new machine, `_downloads_over` is False until
  the new downloads task ends (77305dc). The `downloads_phase1`, `tokenize` and `gguf` markers have no
  reader.
- GGUF judge: the judge stage fetches with `download=True` under the same file lock as the gguf task,
  so it waits instead of failing. The opponents task depends on gguf.
- The core plan comes from the manifest. The runtime type is the same 44 cores; a missing CPU id only
  makes `sched_setaffinity` fail silently.
- The library rebuild is deterministic (byte-identical rebuilds, `docs/E1_COVERAGE.md`), and the train
  stage waits for it.
- The factual questions and the select stage's harness caches are rebuilt deterministically (seeded).
  This costs time only.
- Export held-out rows are downloaded again at the pinned revision. On a machine still re-tokenising,
  `data_licences` marks some sources `tokenized_in_run: false` (cosmetic).
- The old machine's proxy reports are missing on a new one. `report_files` copies only what exists, and
  the manifest keeps the numbers.
- critic is planned at 0 minutes, so `allocate` skips it.
- The patched copy passes the quick tests `tests/train/test_supervisor.py` (39 passed),
  `test_move_deadline.py`, `test_colab_cli.py` (33 passed) and the data tests (109 passed). The
  per-finding diffs applied in sequence equal the tested file (`split_diffs.py`).

## Notes for the supervision scripts (not read; outside the repository)

- Section 19.4 says a snapshot is fetched when a new checkpoint appears. After train_end only
  `post/ckpt` changes, so samples, the prefopt record and the decode selection would not reach a
  snapshot unless the loop also fetches on stage changes.
- A snapshot older than the previous machine's move restores that older deadline. Pass an absolute
  `BES_MOVE_DEADLINE`: `+minutes` would be added to the old deadline.
- `nice` across sessions is ineffective under autogroup. Check
  `cat /proc/sys/kernel/sched_autogroup_enabled` on the runtime: if it is 1, the opponents (nice 15)
  share cores 50/50 with the loader workers.
- A continuation runs notebook cell 4 (llama-cpp-python built from source) before cell 8. A build
  failure there blocks training on the new machine, although only opponents and judge need `llama_cpp`.

## Files

- Reproductions:
  - `repro_recycled_post.py` (F1, F2, F5a)
  - `repro_train_window.py` (F4)
  - `repro_opponents_marker.py` (F5b)
  - `repro_snapshot_race.py` (F7)
- Simulations: `sim_two_moves.py`, `sim_prune_newest.py`.
- Checks of the fixes: `verify_fixes.py`.
- Patched copy: `checkout/`.
- Per-finding diffs, each against the original: `diffs/F1…F8*.diff`. F2, F3 and F5 need the helper that F1 adds; the others stand alone.
