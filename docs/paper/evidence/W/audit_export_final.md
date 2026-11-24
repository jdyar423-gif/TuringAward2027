# Audit: export, the final directory and resumed post stages (`audit_export_final`)

Checkout `/home/user/turingaward2027`, branch `claude/modest-carson-yi2d9g`. The task named HEAD 566517a; during the audit HEAD moved to **b05bfe7** ("micro_split"), which touches only `bes/train/train.py` and `stage_train`/`_micro_split` in `scripts/run_24h.py`. Line numbers below are at b05bfe7; in 566517a, `scripts/run_24h.py` lines are 31 lower between lines 192 and 640, and 32 lower after that. The audit was read-only. Experiments ran under the scratch directory (tiny `configs/smoke.py` model, CPU, `HF_HUB_OFFLINE=1`, `taskset -c 2,3 nice -n 10`). No `colab` command was run: the fetch was driven with a local stand-in class for `Colab`.

Scratch directory: `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/audit_export_final/`
- `exp1_export_chain.py` … `exp9_prefopt_stop.py`: the reproductions (outputs are quoted below).
- `patches/F*.diff`: one unified diff per finding against b05bfe7. Each diff applies on its own, and all of them apply in sequence (`git apply --check` / `git apply`). Applied together they give exactly the tested copy.
- `patched/`: a copy of the checkout with all the diffs applied. 153 targeted tests pass on it (`test_export.py`, `test_stages.py`, `test_supervisor.py` without the slow rehearsal, `test_move_deadline.py`, `test_prefopt.py`, and two `test_colab_cli.py` tests that use no colab binary). `exp6_patched.py`, `exp7b_patched.py` and `exp9` (with `AUDIT_REPO=…/patched`) show the fixed behaviour.
- `patches/on_staged/*.diff` and `patched_staged/`: the same fixes for the checkout's staged, uncommitted state of 04:04 UTC (see the next section). `exp10_on_staged.py` runs against it, and the same 153 tests pass.
- `make_patches.py` and `make_patches_staged.py` regenerate and cross-check every diff.

## The checkout changed while I worked

Another process edited the checkout while this audit ran. At 04:04 UTC the index held uncommitted, staged changes (`git diff --cached | sha256sum` → `333273d96869dd4f`) to `bes/data/download.py`, `bes/train/supervisor.py`, `scripts/colab_tpu.py` and `scripts/run_24h.py`. There were also unstaged edits to `_micro_split`, `bes/train/train.py` and a test.

The staged `run_24h.py` adds `Run._busy` / `_await_tasks` waits to `sample`, `prefopt`, `select`, `judge` and `export`. Those cover the **waits** of finding 3. They do not change what a recycled machine downloads, so UltraFeedback still comes after about 36 GB, and the 25 % prefopt wait ends without it. They also leave the 24-worker catch-up tokenisation on in the post phase.

I therefore provide each diff twice:
- `patches/*.diff` against b05bfe7;
- `patches/on_staged/*.diff` against that staged state. Here F3 is reduced to what the staged state lacks: the post-phase download plan, no catch-up cores after training, the library decision's reason, and the export's failed `library` part.

Checks on the staged variant:
- `F1` and `F3` of `patches/` conflict with the staged state; the five others apply to it unchanged.
- All seven `on_staged` diffs apply on their own and in sequence.
- The result (`patched_staged/`) passes the same 153 targeted tests, and `exp10_on_staged.py` shows F1, F2 and the reduced F3 working there.

Re-check with `git apply --check` before applying, because the checkout is still being edited.

## Summary

| # | severity | finding | patch |
|---|---|---|---|
| 1 | blocker | If `export_run` fails at the end, the step-5075 emergency export stays in `final/`. The run still reports complete (exit 0), `--resume` never retries the export, the runtime supervisor packages `final/`, and `fetch` delivers that old export with "SHA-256 verified" | `F1_stage_export_fallback.diff` |
| 2 | blocker | prefopt accepted but `post/ckpt` holds no loadable checkpoint → `select` and `judge` fail, and the export writes `EXPORT_NOTE.txt` *into* the old export (breaking its verification) instead of exporting the trained weights | `F2_latest_params_fallback.diff` |
| 3 | major | Post stages continued on a recycled machine act on inputs that machine does not have yet. `sample` skips itself. `prefopt` runs without UltraFeedback (the last of 43 downloads, after about 36 GB). `select` decides "no library was built". The export drops a library that `select` chose to ship. `judge` skips itself | `F3_recycled_machine_post_stages.diff` |
| 4 | minor | A stop (SIGTERM) during the hour of DPO throws away the finished, possibly accepted result (`ctx.check()` runs before the save), and the resume starts DPO again | `F7_prefopt_keeps_a_finished_result_on_a_stop.diff` |
| 5 | minor | `fetch` checks only the archive's SHA-256. It never says which step or weights it brought back, and it extracts over an earlier fetch (stale files stay) | `F4_fetch_says_what_it_fetched.diff` |
| 6 | minor | The Drive mirror (cell 1b) and cell 12 only ever add files: the Drive copy of the export becomes a union of all exports and fails verification | `F5_drive_copy_of_the_export_is_replaced.diff` |
| 7 | minor | `scripts/export.py --run` (the manual re-export) ignores `train/ckpt_params`. `run_24h` takes it when it is newer | `F6_manual_export_checkpoint.diff` |

## Answers to the questions

### 1. Does a later export overwrite a complete `final/`, skip it, or mix old and new parts?

- **`export_run` overwrites it, atomically and completely** (`bes/export.py:1899-2002`). It writes `.tmp-final-*`, verifies it, renames `final` to `.old-final-*`, renames the temporary directory to `final`, then removes the old one. It never skips. E1 shows this: the emergency export (step 7, train weights, default decode settings) is replaced by step 11 with the prefopt weights and the chosen decode overrides `['extra.particles=1', 'extra.use_library=false', …]`. Verification passes and no `.tmp-` or `.old-` directory is left behind.
- **Old and new parts get mixed** in four places:
  - `stage_export`'s "no checkpoint to export" branch writes `EXPORT_NOTE.txt` into an existing export (`run_24h.py:1317`). E1 then reports `verify: ['EXPORT_NOTE.txt: not listed in SHA256SUMS']` while `export.json` still names the old step.
  - The fallback path removes only `final/ckpt` and writes `config.json`, `accounting.json` and `EXPORT_NOTE.txt` over an `export_run` directory (`run_24h.py:1342-1363`).
  - The Drive mirror and cell 12 (finding 6).
  - `fetch` extracting over an earlier fetch (finding 5).

### 2. Which checkpoint does `latest_params` pick?

`Run.latest_params` is at `run_24h.py:288-299`. `train_checkpoint()` (`run_24h.py:273-286`) takes the newer of `train/ckpt` and `train/ckpt_params` by step; on a tie the full checkpoint wins.

| stage | call | weights |
|---|---|---|
| sample, critic, prefopt | `prefer_post=False` | `train_checkpoint()`. DPO always starts from the trained weights, never from an earlier DPO result |
| select, judge, export, emergency export | `prefer_post=True` | the newest `post/ckpt` if `manifest.prefopt.accepted` and `post/ckpt` exists, else `train_checkpoint()` |

- **Interrupted prefopt.** Nothing is written: the save comes after `ctx.check()`, which is finding 4. So select, judge and export use the trained weights, or an earlier attempt's accepted checkpoint if one exists; either way this is consistent with the manifest. The re-run starts from the trained weights.
- **Rejected prefopt** (also a rejected re-run after an earlier accepted attempt left `post/ckpt`): `accepted` is False, so the trained weights are used. E1 confirms this: the export takes the step-11 train checkpoint although `post/ckpt` holds a checkpoint.
- **After a resume.** `select`, `judge` and `export` reload the manifest (`self.effective()` / `self.reload()`), so they make the same choice as before the resume.
- **Accepted, but `post/ckpt` empty or broken** → `FileNotFoundError` / `ValueError`. This is finding 2.

### 3. Does the export's held-out check use the right weights?

Yes. `check_16bit` compares the loaded checkpoint's float32 tree with its 16-bit copy on the same held-out rows. Both are run with float32 activations, dense attention and highest precision (`bes/export.py:1752-1779`). In E1, recomputing the float32 held-out loss of each tree independently gives:
- A (step 7): 11.10166
- B (step 11, train): 11.35065
- C (post, exported): **11.39247**

`export.json` reports `loss32` 11.392469284700887, which is C. The source fingerprint equals the post checkpoint's. The stored 16-bit `head/out` is within 4.9e-4 of C and 1.5e-2 from B.

### 4. What does `python scripts/colab_cli.py fetch` bring back? Can it bring back a stale export silently?

It downloads `/content/bes_cli/export/parts.json` and its parts (`colab_cli.py:517-559`). That is a tar of `<run>/final` *as it was when* `colab_tpu.Supervisor.package()` (`colab_tpu.py:703-722`) ran. Packaging happens in three cases:
- in `finish()`, once the run is "complete" by the run's own rule. A **failed** export counts as over here (`colab_tpu.py:477-488`);
- when the run is incomplete and no automatic continuation is left. Then any `final/` is packaged, including an emergency export (`colab_tpu.py:941-947`);
- on the `package` command, at any time.

The fetch checks every part's SHA-256 and the joined archive's, then prints `export in <dir> (N MB, SHA-256 of the archive verified)`. **Yes, it can bring back a stale export without saying so.** E2 reproduces the whole chain:
- process 1: training fails and an emergency export of step 7 is made;
- process 2: training ends at step 15 and `export_run` fails, giving supervisor status `'complete'` (run_24h exits 0) with export stage `'failed'`;
- process 3 (`--resume`): pending stages `[]`;
- colab_tpu: `run_state 'complete'`, failed stages `['export']`, export packaged;
- fetch: exit 0. The fetched export is **step 7, train weights**, and it verifies, while the run ended at step 15.

A second fetch into the same directory leaves files of the first one behind (`verify: ['weights32.npz: not listed in SHA256SUMS']`).

### 5. The trainer's end of training

`Trainer.run` (`bes/train/train.py:880-960`):
- **Stop reasons.** `_stop_reason` (855-862) returns `requested`, `max_steps` or `deadline`; a guard halt gives `halt`.
- **Last step.** The pending step's loss is read before the end, and a final rollback keeps the restored state.
- **Final save.** It is blocking and checked whenever the state is not already saved, and is skipped after a halt that restored a trusted checkpoint. It tries the full state, retries once, then saves the parameters only (`train/ckpt_params`), then raises `CheckpointError`.
- `summary["checkpoint"]` names the checkpoint that holds the final state.

`stage_train` (`run_24h.py:661-676`):
- `requested` raises `Interrupted`. The stage stays pending; on resume it continues, or is skipped with "the training window closed" once `train_end` has passed.
- `deadline`, `max_steps` and `halt` finish the stage.
- There is no `complete` reason.
- A `CheckpointError` fails the essential stage. That triggers an emergency export and `FatalStageError` (exit 1), and a resume runs the stage again (a failed essential stage stays pending).
- `train_checkpoint` prefers a newer `ckpt_params`.

Five end-of-training tests pass on the real checkout: `test_stop_request_and_restart_after_the_window`, `test_interrupted_run_resumes_and_still_ends_on_time`, `test_a_failed_final_save_falls_back_to_the_parameters_then_fails_loudly`, `test_persistent_skipped_updates_roll_back_and_halt` and `test_run_24h_finds_a_parameters_only_final_checkpoint`.

### 6. A stop during a post stage, then a resume

**On the same machine it continues the stage.** It neither restarts training nor skips stages (E3, real `Supervisor`, the live run's plan: budget 1279.7 min, `post_reserve_hours` 3.6667, `post_minutes` critic=0,buffer=20):
- Process A is interrupted in `sample`: train finished (1 attempt), sample pending (1 interrupt), no emergency export.
- Process B, started 30 minutes later, has pending stages `['sample', 'critic', 'prefopt', 'select', 'judge', 'export']`. It called `['sample', 'prefopt', 'select', 'judge', 'export']`; critic was skipped (0 minutes) and train was never called.
- Shedding follows `SHED_ORDER`. With prefopt stopped after 55 minutes:
  - resumed 20 minutes later, the slices are prefopt 60, select 45, judge 24.9, export 15;
  - resumed 75 minutes later, they are prefopt 60, select 14.9, judge skipped, export 15.

**On a recycled machine the stage itself acts on missing inputs** (finding 3). For prefopt, the resume restarts the stage, and a stop during the DPO loses a finished result (finding 4).

## Findings

### 1 (blocker). A failed `export_run` leaves the stale emergency export, and it is packaged and fetched as the run's export

**Where.**
- `scripts/run_24h.py:1334`: `res = fn(self.paths.out, **kw)` has no handler.
- `bes/train/supervisor.py:1485-1492`: a failed non-essential stage is recorded and the loop continues.
- `bes/train/supervisor.py:1422-1425`: `pending()` drops failed non-essential stages, so `--resume` never retries the export.
- `scripts/run_24h.py:1725`: `EXIT_OK`.
- `scripts/colab_tpu.py:484-488`: the failed export counts as over, so `finish()` packages `final/`.
- `scripts/colab_cli.py:552-558`: the fetch checks the transport hash only.
- Also `run_24h.py:1317` and `1342-1363`: the note and fallback paths write into an earlier export.

**Sequence.** Today the supervisor made four complete, verifying emergency exports of step 5075 into `/content/runs/main16/final`. At the real end, any exception in `export_run` (for example ENOSPC while writing the 16-bit weights, the shipped library or the float32 copy next to the old export, or an exception in a path the emergency exports never ran) leads to:
1. the export stage is `failed`, the run is `complete`, and the exit status is 0;
2. a resume does nothing;
3. `colab_tpu` packages `final/`, which still holds the step-5075 export;
4. `fetch` reports success.

The user receives step-5075 weights as the 24-hour model.

**Reproduced** (E2, the real `Supervisor`, `Run.stage_export`, `colab_tpu.Supervisor.package`, and `colab_cli.cmd_fetch` with a local stand-in; output in answer 4).

**Fix** (tested, `exp6_patched.py` part 2).
- If `export_run` raises, `stage_export` logs it and falls through to the fallback export of the **final** weights (checkpoint, configuration, accounting, and a note with the error).
- The fallback first moves a non-empty `final/` aside to `final.replaced-<time>`, so old weights can neither stay mixed in nor be packaged.
- The "no checkpoint" note is written only into an empty `final/`.

Result on the patched copy: `by fallback, weights train, checkpoint ckpt-000001-step-000000015`. `final/` holds `['EXPORT_NOTE.txt', 'accounting.json', 'ckpt', 'config.json']` and the old export is in `final.replaced-…`. The fetch then says `NOT a full export: bes/export.py (export_run) did not export (export_run failed: OSError: [Errno 28] …)`.

```diff
--- a/scripts/run_24h.py
+++ b/scripts/run_24h.py
@@ -37,7 +37,6 @@ import inspect
 import json
 import math
 import os
-import shutil
 import sys
 import time
 from pathlib import Path
@@ -1314,7 +1313,10 @@ class Run:
             _, _, info, ckpt, source = self.latest_params(ecfg)
         except (FileNotFoundError, ValueError) as e:
             note = f"no checkpoint to export: {e}"
-            (final / "EXPORT_NOTE.txt").write_text(note + "\n")
+            if any(final.iterdir()):            # a note inside an earlier export breaks its SHA256SUMS
+                self.log(f"export: {note}; {final} keeps the earlier export it holds")
+            else:
+                (final / "EXPORT_NOTE.txt").write_text(note + "\n")
             return {"exported": False, "note": note}
@@ -1331,17 +1333,27 @@ class Run:
             params = inspect.signature(fn).parameters
             kw = {k: v for k, v in avail.items() if k in params}
-            res = fn(self.paths.out, **kw)
-            return {"exported": True, "by": "bes.export.export_run", "checkpoint": str(ckpt),
-                    "weights": source, "result": SV._jsonable(res) if isinstance(res, Mapping) else str(res)}
+            try:
+                res = fn(self.paths.out, **kw)
+            except Exception as e:  # noqa: BLE001 - the final weights leave the run all the same
+                why = f"export_run failed: {type(e).__name__}: {e}"
+                self.log(f"export: {why}; writing the fallback export of the {source} weights instead")
+            else:
+                return {"exported": True, "by": "bes.export.export_run", "checkpoint": str(ckpt),
+                        "weights": source, "result": SV._jsonable(res) if isinstance(res, Mapping) else str(res)}
         # fallback: ...
         from bes.train import checkpoint as CK
 
+        if any(final.iterdir()):
+            # an earlier export (an emergency one, of older weights) must neither stay mixed with
+            # these files nor be packaged and fetched in their place
+            aside = final.with_name(f"{final.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
+            os.replace(final, aside)
+            final.mkdir()
+            self.log(f"export: the earlier export moved to {aside}")
         params, consts, info, _, _ = self.latest_params(ecfg)
         root = final / "ckpt"
-        if root.exists():
-            shutil.rmtree(root)                 # an earlier (emergency) export is replaced
         saved = CK.save(root, ...
@@ -1356,7 +1368,7 @@ class Run:
-        note = (f"bes/export.py (export_run) is not available ({why}); this directory holds the "
+        note = (f"bes/export.py (export_run) did not export ({why}); this directory holds the "
```
(The exact diff is in `patches/F1_stage_export_fallback.diff`.)

### 2 (blocker). Accepted prefopt, `post/ckpt` without a loadable checkpoint: `select` and `judge` fail and the export writes a note into the old export

**Where.** `scripts/run_24h.py:293-299`: `respond.find_checkpoint(post)` is called whenever `post/ckpt` *exists*, and the result is loaded without a fallback. `stage_export` then reaches the note branch at line 1317.

**Sequence.**
1. A Drive mirror pass copies `manifest.json` (prefopt accepted) and then starts copying `post/ckpt` into `.part-ckpt-…`.
2. The runtime is reclaimed during that copy.
3. `restore_from_drive()` ignores `.part-*` but creates `post/ckpt`, so the new machine has an **empty** `post/ckpt` and an `accepted` manifest. E5(c) reproduces this with the notebook's own mirror and restore code.
4. `select` and `judge` fail with `FileNotFoundError: no checkpoint at …/post/ckpt`.
5. `stage_export` returns `exported: False` after adding `EXPORT_NOTE.txt` into the earlier export. The trained weights are never exported.

On the snapshot route (the live one) the window is negligible, because `manifest.json` is the first tar member and checkpoint directories are complete once named. Any other loss or damage of `post/ckpt` has the same effect.

**Reproduced** (E1 "post-missing": `stage result: {'exported': False, 'note': 'no checkpoint to export: …'}`, `files added: ['EXPORT_NOTE.txt']`, `verify: ['EXPORT_NOTE.txt: not listed in SHA256SUMS']`, `stage_select -> latest_params raises FileNotFoundError`. E5(c) shows the trigger).

**Fix** (tested, `exp6_patched.py` part 1). If the accepted post checkpoint does not load, log a WARNING and use the trained weights. All three stages then agree: `latest_params with an empty post/ckpt: train step 11`, export verifies.

```diff
--- a/scripts/run_24h.py
+++ b/scripts/run_24h.py
@@ -292,11 +292,16 @@ class Run:
         post = self.paths.post / "ckpt"
         if prefer_post and (self.man.get("prefopt") or {}).get("accepted") and post.exists():
-            path, source = respond.find_checkpoint(post), "prefopt"
-        else:
-            path, source = self.train_checkpoint(), "train"
+            try:
+                path = respond.find_checkpoint(post)
+                params, consts, info = respond.load_checkpoint(path, ecfg.model)
+                return params, consts, info, Path(path), "prefopt"
+            except Exception as e:  # noqa: BLE001 - lost or broken: the trained weights, said loudly
+                self.log(f"WARNING: the preference optimisation was accepted but {post} holds no loadable "
+                         f"checkpoint ({type(e).__name__}: {e}); the trained weights are used")
+        path = self.train_checkpoint()
         params, consts, info = respond.load_checkpoint(path, ecfg.model)
-        return params, consts, info, Path(path), source
+        return params, consts, info, Path(path), "train"
```

### 3 (major). Post stages continued on a recycled machine act on inputs that machine does not have yet

**Where.**
- `run_24h.py:815-817`: `sample` raises `StageSkipped("no chat prompts on disk")` on its first look.
- `run_24h.py:999-1002`: `prefopt` uses UltraFeedback only if `tok/ultrafeedback` exists.
- `run_24h.py:1166-1167`: `_library_decision` returns `ship False, "no library was built"` when `state/library.json` is absent.
- `run_24h.py:1330`: the export passes `library=self.library_path() or None`.
- `run_24h.py:1262-1266`: `judge` skips without the opponents' markers.
- `supervisor.py:1530-1566`: the downloads start again from the full plan, and UltraFeedback is entry 43 of 43, after ClimbMix's 30 GB cap.
- `run_24h.py:223-235`: the catch-up tokenisation (up to 24 workers) also starts in the post phase.

A machine change in the post phase (16:51–20:31 UTC) is plausible: runtime 2 lived 2 h 12 min, and runtime 3, assigned at 01:05:55 UTC, reaches the Pro limit of about 12 h around 13:06. A restored run directory has records, `post/`, `final/` and the newest checkpoints. It has no shards, raw files, library, `state/` markers or proxy outputs until the background tasks have made them again, and the resumed stage starts within seconds.

**Reproduced** (E4, E7):
- `sample` was continued on a fresh machine with `samples.jsonl` on disk. It skips itself with `no chat prompts on disk`, and its whole remaining slice is lost.
- The library decision is `ship False, reason 'no library was built'`, although the library is ready by export time. The export then says `library mode 'none', not_shipped "the decode selection's library rule did not pass: no library was built"`.
- With `ship True` from the old machine, the export on the new machine gives `library mode 'none'`. The `decode_choice` still says `extra.use_library=true`, the disclosure reads "not shipped; library decision: shipped: factual gain", and 0 parts are reported failed.
- `judge`: `no opponent replies on the test split yet`.
- The download order for the final config was printed: `ultrafeedback` is index 42, the last entry.

**Fix** (tested, `exp6_patched.py` part 3 and `exp7b_patched.py`). On the original machine nothing changes, because every readiness check holds immediately and the downloads task started with the full plan.
- **Download plan.** `bes/train/supervisor.py` gains `training_over(paths)` and `post_download_plan(cfg)`. When the ledger says training is over, the downloads task fetches only the first file of each chat, rag and math source, then the preference data. `_catch_up_cores` returns the plan's share once training is over.
- **Bounded waits.** `Run._wait_for_inputs(ctx, ready, what, task, frac)` polls until the input exists, its background task ends, a stop is requested, or a fraction of the slice is used:

  | stage | waits for | at most |
  |---|---|---|
  | sample | every sampling source's download manifest or shard directory | 25 % of the slice |
  | prefopt | `tok/ultrafeedback/manifest.jsonl` | 30 % |
  | select | the library marker (before its generator is built) | 30 % |
  | export | the library, when `decode.library.ship` | 60 % |

- **Reporting.** `_library_decision` says "the library was not ready on this machine (its task was still building it)" while the library task runs. `export_run` records a failed `library` part when the selection shipped a library that is not there, so the card says so at the top.

Patched results:
- `stage_sample on the fresh machine after 2 polls: 64 new prompts`;
- `library wait: ready True after 3 polls`;
- `[a] library task running: polls 2, library mode 'copy', errors []`;
- `[b] library task failed: polls 0, library mode 'none', errors ['library'], card warns: True`;
- post-phase plan: the 19 chat, rag and math sources and `ultrafeedback`.

The full diff is in `patches/F3_recycled_machine_post_stages.diff` (8 hunks: run_24h.py, supervisor.py, export.py). The judge's missing opponents are not addressed: the judge is shed first, and the opponents' replies on CPU take long.

On the staged state of 04:04, the waits are already there (`_await_tasks`, which also covers the judge). `patches/on_staged/F3_recycled_machine_post_stages.diff` keeps only:
- `training_over` and `post_download_plan`, plus the plan switch in `task_downloads`;
- the `_catch_up_cores` guard;
- the `_library_decision` reason;
- the `export_run` `library` error part.

Without the post-phase plan, the staged prefopt wait (25 % of the slice) ends before UltraFeedback has arrived.

### 4 (minor). A stop during the DPO discards the finished result

**Where.** `run_24h.py:1047-1050`: `run_prefopt` has no checks. `ctx.check()` then runs before the report, the checkpoint and the manifest are written.

**Sequence.**
1. SIGTERM (`send stop`, or the overrun stop) arrives during the roughly 60 minutes of DPO.
2. DPO finishes, accepted.
3. `Interrupted` is raised, and nothing is saved.
4. The resume runs DPO again from the trained weights. It costs up to 60 more minutes and, late in the run, `judge` and then `select` get shed (E3).

A kill (reclaim) loses the run anyway. This applies only to stop requests.

**Reproduced** (E9: `stage_prefopt raised Interrupted (stop requested during prefopt) after an accepted DPO`; `post/ckpt: False, prefopt_report.json: False, manifest prefopt: None`).

**Fix** (tested, E9 on the patched copy). Drop that check: the stop is honoured at the next stage boundary. Result: `supervisor interrupted at select: prefopt finished, post/ckpt True, manifest prefopt accepted True; later stages called: []`.

```diff
--- a/scripts/run_24h.py
+++ b/scripts/run_24h.py
@@ -1047,7 +1047,8 @@ class Run:
                                               tok=tok, log=self.log)
-        ctx.check()
+        # a stop requested during the DPO (run_prefopt has no checks) is honoured at the next
+        # stage boundary: the finished result is saved and recorded first, never thrown away
         out = {"accepted": bool(rep["accepted"]), ...
```

### 5 (minor). `fetch` does not say what it fetched, and it overlays an earlier fetch

**Where.** `scripts/colab_cli.py:554-556`.

**Sequence.** `fetch` extracts into `<to>/final` with `extractall`, over whatever is there, and prints only the size and the transport hash. A stale export (finding 1, or a `package` sent before the export stage) looks exactly like the real one. A file of an earlier fetch stays and breaks `--verify`.

**Reproduced** (E2: `[fetch again] step 19; verify problems ['weights32.npz: not listed in SHA256SUMS']`).

**Fix** (tested, `exp6_patched.py` part 2). Extract into `<to>/.incoming`, set an earlier `final` aside as `final.previous-<time>`, swap, then print the export's step, weights, export time and held-out check, or the fallback's note:
- first fetch: `fetch 1: NOT a full export: …`
- second fetch: `fetch 2: step 15, train weights, exported 2026-11-24 03:43:20 UTC, held-out check True`, and `final` verifies.

The diff is in `patches/F4_fetch_says_what_it_fetched.diff`.

### 6 (minor). The Drive copy of the export becomes a union of every export

**Where.** Notebook cell 1b `MIRROR_CODE`, `sync()`: it copies changed files and never removes any. Cell 12: `shutil.copytree(src, DRIVE_DIR + ".export", dirs_exist_ok=True)`.

**Sequence.**
1. An earlier export holds a file the final one lacks: `weights32.npz` when its check could not run, or library files, or reports.
2. The Drive copy keeps it, and the copy no longer verifies.
3. `restore_from_drive()` brings the mixed `final/` back to a recycled machine.

This only matters with Drive; the live route uses snapshots.

**Reproduced** (E5(b): `Drive copy of final/: step 15, verify ['weights32.npz: not listed in SHA256SUMS']`; with the patch, `verify []`).

**Fix.** `sync()` removes Drive files of `final/` that the local `final/` no longer has, and cell 12 removes `DRIVE_DIR.export` before copying. Both are applied to the three notebooks, with the v9 notebook regenerated by `scripts/notebook_variants.py`; the notebook tests pass. The diff is in `patches/F5_drive_copy_of_the_export_is_replaced.diff`.

### 7 (minor). The manual re-export picks an older checkpoint than the run

**Where.** `bes/export.py:1602-1608` `_latest_checkpoint`: `find_checkpoint(out / "train")` looks only in `train/ckpt`.

**Sequence.** The full final save fails and the parameters-only save at step 40 succeeds. `run_24h` exports step 40. `scripts/export.py --run` (the remedy for a failed export stage) exports `train/ckpt` at step 30.

**Reproduced** (E5(a): `Run.train_checkpoint: train/ckpt_params/ckpt-000001-step-000000040`, `_latest_checkpoint: train/ckpt/ckpt-000001-step-000000030`; patched: both step 40).

**Fix.** The same rule as `Run.train_checkpoint`. The diff is in `patches/F6_manual_export_checkpoint.diff`.

## Checked and fine

- `export_run` replaces an earlier export atomically and completely: it builds, verifies, then swaps (E1, and `test_export_replaces_an_earlier_export_atomically`).
- The end-of-run export uses the accepted post-DPO weights and the decode overrides chosen by `select`. A rejected prefopt gives the newest training checkpoint (E1).
- The held-out check runs on the exported weights: `loss32` equals the recomputed loss of the post tree, and the fingerprint is the post checkpoint's (E1).
- `export_run` handles the records a complete run leaves and the real library in copy mode (`data_cache/library/lib_v2`), which the emergency exports never exercised. E8 produced prefopt weights, library `copy`, `errors []`, `verify []`, five proxy reports copied, and card sections for prefopt, decode, the library decision, the judged table and a skipped arm.
- `latest_params` with `prefer_post=False` (sample, critic, prefopt) always uses the trained weights, so DPO never starts from an earlier DPO result.
- `train_checkpoint` prefers a newer `train/ckpt_params` and the full checkpoint on a tie (the test passes).
- The trainer's final save is blocking and checked, retried once, then falls back to parameters only, then fails loudly; `summary['checkpoint']` names the holder (tests pass).
- In `stage_train`, `requested` raises `Interrupted` (stage pending); `deadline`, `max_steps` and `halt` finish the stage; a `CheckpointError` fails the essential stage, and the resume retries it.
- A stop during a post stage followed by a resume on the same machine continues that stage. Training is not re-run, and later stages are shed only by the documented order when time is short (E3).
- The emergency export runs only on an essential failure, or on an interruption within twice the export's planned slice of the deadline; it never runs for the export stage itself.
- `manifest.json` is reloaded before every `latest_params(prefer_post=True)` call in `select`, `judge` and `export`.
- The snapshot of a run directory carries `final/`, `post/` and the newest complete checkpoint of `post/ckpt`, and skips temporary directories (`.tmp-final-*`, `.old-final-*`).
- `package()` empties `export/` before writing new parts, so a fetch never mixes parts of two packagings (parts are checked against the new `parts.json`).

## Other observations (not findings)

- `stage_export` silently uses the base configuration when `self.effective()` raises (`run_24h.py:1309-1312`). The same configuration-hash mismatch makes the other post stages fail loudly, but here it yields an export whose `config.json` describes the base configuration (no plan overrides, ladder switches or horizon). The weights and consts are still right.
- `export_run`'s swap (`os.replace(final, old)` then `os.replace(tmp, final)`, `bes/export.py:1993-1999`) under `except BaseException: rmtree(tmp)`: a second SIGTERM landing between the two renames would leave no `final/` (the old export survives in `.old-final-*`). The window is microseconds.
- On a recycled machine, `judge` skips itself until the opponents task has produced the test replies again; the proxy outputs under `data_dir/proxy` are neither mirrored nor in the snapshot.

## Deployment notes for the live run

- The running supervisor process imported `scripts/run_24h.py` and `bes/train/supervisor.py` at its start. `bes/export.py` is imported lazily inside `stage_export` and nowhere else in that process. So:
  - a corrected `bes/export.py` on the runtime's checkout before the export stage is used by the running process (F3's export hunk, F6);
  - the `run_24h.py` changes (findings 1–4) take effect only in a new process: a stop and resume, or the next machine, which clones the branch;
  - the download-plan change of finding 3 matters only in the downloads task of a resumed process, which loads the new code anyway.
- No change touches the model, the optimiser, the trainer, the configuration or the checkpoint format. Parameter tree, optimiser state and computed values are unchanged.
