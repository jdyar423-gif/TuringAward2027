# Micro graft experiment: graft arms for the micro bench

Worktree `/home/user/turingaward2027/.claude/worktrees/wf_ad9c3744-513-1`, base commit `8a86bcb`
(HEAD, unchanged; nothing committed). Patch: `scratchpad/wf/micro_graft.patch` (`git diff HEAD`
with the four new files marked intent-to-add; 6 files, +1,756 / -18 lines). Scratch, logs and
smoke records: `scratchpad/wf/micro_graft_work/` (420 KB after clean-up).

## Summary

* Graft arms exist in the micro bench (`bes.eval.microbench`: `Graft`, `graft_plan`,
  `find_trunk`, `run_graft_arm`; `run_arm` dispatches to it). A graft arm trains its base arm's
  computation up to a fraction g of the budget, grafts it into the target configuration with
  `scripts/graft_run.py` `graft_arrays` (imported, not copied), checks on held-out rows that the
  loss is unchanged (fails loudly otherwise), and continues to the end with the same schedule
  positions and the batches the base would have seen.
* The 12 new arms are in `scripts/micro_ablate.py` `GRAFT_ARMS` (`ref_s2`, `v9_pages_s1`,
  `v9_pages_s2`, `graft_{25,50,75}` with seeds 0, 1, 2). With `ref`, `ref_s1` and `v9_pages` they
  make the 15 runs of `GRAFT_EXPERIMENT`. They are selected with `--arms`. The default E2 queue
  and `docs/E2_ABLATIONS.md` are unchanged: the arm list is the same, and the spec hash, config
  hash and arm dict of all 31 E2 arms match HEAD at 1M and 120k tokens with two bench settings
  (124 entries).
* Parallel runs: one process per arm with its own `--out` sub-directory and CPU pair, then
  `scripts/micro_graft_report.py` (new) on the parent directory; `scripts/micro_graft_launch.sh`
  (new) does it for the 15 runs; `docs/MICRO_GRAFT.md` (new) documents the experiment.
* Smoke run at the real micro size, 120,000 tokens (29 steps), seed 0, 2 threads on cores 2,3:
  * The trained trunk of `graft_50` reproduces `ref` exactly: both held-out evaluations before
    the graft (per-row sums included) are bit-identical.
  * The graft check has a difference of exactly 0 (loss 6.72171688079834 before and after;
    held-out web, chat and assistant also 0).
  * A second `graft_50` that loaded `ref`'s kept state ends bit-identical to the one that
    trained its trunk.
  * Each arm costs about 2.2-2.5 s per step, 13-15 s for the first step (compilation) and
    1.7-2.3 GB peak RSS.
* **The chat training sample is not on this machine.** `data_cache/samples` holds only the two
  held-out files. `HuggingFaceTB__smol-smoltalk/data/train-00000-of-00004.parquet` (the bench's
  chat training source) exists nowhere on the machine. ClimbMix `shard_00000.parquet` (web
  training) exists only under `data_cache/raw/` (same pinned revision, sha256 verified).
  The smoke run therefore read its samples from a scratch directory of symlinks, in which the
  chat training file is replaced by the smol-smoltalk test file. Its chat training rows
  overlap the chat held-out rows. Its losses are not results, and its chat-training pool is
  not the real one. Costs and the equality checks do not depend on it.

## What was built

| file | change |
|---|---|
| `bes/eval/microbench.py` | `Arm.graft` (optional; serialised only when set, so existing arms' dicts and hashes are unchanged), `Graft` (base arm, fraction, target label; checks same seed, mixture and budget), kind `graft`; `run_arm(keep_states=...)` (opt-in, default off: keeps the state of every evaluation checkpoint before the last in `arms/<name>/states`); `graft_plan`, `find_trunk`, `run_graft_arm`, `GRAFT_TOL`, `STATES`, `TRUNK_MODES`; module docstring section "Graft arms" |
| `scripts/micro_ablate.py` | `GRAFT_ARMS`, `GRAFT_EXPERIMENT`, `ALL_ARMS` (`--arms` and `--run-arm` resolve against it, the default queue is still `E2_ARMS`); options `--prepare` (build the pools once, then exit), `--keep-states`, `--trunk {auto,train,load}`, `--trunk-from DIR`, `--pool-root`, `--samples-dir`; graft arms are checked before a queue starts (fraction must be an evaluation fraction, budget large enough); time estimate for graft arms; the queue's parent process stays without JAX (40 MB) |
| `scripts/micro_graft_report.py` (new) | the report: (a) table, (b) paired differences, (c) share recovered; text and `--json`; `--merge` writes the records used |
| `scripts/micro_graft_launch.sh` (new) | prepare, 15 runs in parallel (run i on CPUs `FIRST_CPU + 2i`, `+1`), wait, report; `DRY_RUN=1` prints the 17 commands |
| `tests/eval/test_micro_graft.py` (new) | 11 tests (below) |
| `docs/MICRO_GRAFT.md` (new) | the experiment: arms, mechanism, commands, cost, report, files needed |

`bes/config.py` is not touched; no existing arm changes (see Tests).

## Design decisions

**How the trunk is shared.** The trunk (steps `[0, g)`) is the base arm's own computation: the
same `arm_config` (the base arm's overrides, `run.seed`, horizon = budget), the same initial
parameters (`M.init_params(cfg.model, jax.random.key(seed))`, the line `run_arm` uses), the same
optimiser, the same compiled `make_train_step`, the same schedule calls and the same
`BatchMaker` (a pure function of pools, mixture, seed, step and shape). It comes from one of two
places, and `record["graft"]["trunk"]` records which (`source`, `checkpoint`, `spec_hash`,
`rejected` candidates):

* *loaded* (preferred): a state the base arm kept at step g (`--keep-states`), found in `--out`
  (`arms/<base>/states`) or a `--trunk-from` directory (also `<dir>/<base>/arms/<base>/states`,
  the one-sub-directory-per-arm layout). It is accepted only if its manifest's hash is the base
  arm's spec hash at this budget and bench, which covers configuration, seed, mixture, budget,
  bench settings and the bench config file text. The base's curve and train log up to g are
  taken from its metadata. A candidate with another hash is listed as rejected and the trunk is
  trained.
* *trained*: inside the graft arm. In the 15-way parallel layout every graft arm does this,
  because the reference runs are still running when the graft arms start.

`--trunk auto` (default) loads when it can; `train` and `load` force one. In both paths, curve
points up to g carry `phase: "trunk"` and no evaluation views (the trunk of a reference arm
would otherwise add three view evaluations per checkpoint).

**How optimiser state and schedule are carried.** `graft_arrays` fills a fresh train state of
the target configuration with every leaf of the trunk's state checkpoint that has the same
path, shape and dtype: the 59 parameter leaves, their 114 optimiser slot arrays (MuonH momentum
and cached initial norms, Adam moments), `opt_state/count` and `step`. The fresh state is built
exactly as `run_arm` builds the target arm's first state with the same seed, so the 5 new
parameter leaves (`gain/o`, `page/key`, `page/pin`, `page/pout`, `page/val`; 10,502,144 numbers
at the micro width) and their 10 slot arrays are those of `v9_pages` at its first step. The
consts are the target's.

* `graft_run.fresh_tree` is not used. It builds the 24-hour trainer's state, whose
  initialisation key is a split of the seed, while every micro arm is initialised from
  `jax.random.key(seed)`.
* `graft_run.function_check` is not used either. It reads run manifests and uses a synthetic
  batch. Its tolerance rule is kept.
* Before any training, `graft_plan` refuses a graft when:
  * the fraction is not one of the evaluation fractions;
  * the evaluation fractions do not fall on distinct steps;
  * base and target have different step shapes;
  * a base parameter is missing or reshaped in the target;
  * any optimiser setting differs (the rate and decay of a group the base has no leaf in are
    exempt);
  * the learning-rate factor, momentum or decay factor differs at any step.
* The schedule continues because the loop index continues (`p = s / n` with the same n).
* The data order continues because the batch of step s does not depend on the model.
* **Carried count.** The new Adam slots start at zero with the count at g, so their bias
  corrections are those of update g + k, about 1 for g of a few tens. With `adam_b1 = 0.8` and
  `adam_b2 = 0.95`, the page tables' updates 1, 2, 5 and 50 after the graft are 0.89, 1.15, 1.41
  and 1.04 times those of a fresh start with the same moments. The live graft did the same, so
  the experiment keeps it. It is part of what "graft" means here.

**What is checked at the graft point.**

* Leaves: `graft_arrays` must report no mismatched and no left-behind leaf.
* Loss: `bes.model.model.loss_fn` on the first `eval_chunk` (8) held-out web rows (8 x 512
  positions), with the trunk's parameters and consts and with the grafted ones. The held-out
  loss of every evaluated domain (web, chat, assistant; 32 rows each) is also re-evaluated on
  the grafted model.
* Tolerance: every difference must be within 1e-6 x max(1, |loss|), the rule of
  `graft_run.function_check`.
* Record: before, after, difference, tolerance, the largest per-row difference, the carried
  step, count, p, learning-rate factor and momentum, and the leaf report go into
  `record["graft"]`.
* On failure, a `status="failed"` record with these numbers is appended and `RuntimeError` is
  raised. A test checks this: a target that switches the tape's output masses off fails.
* Resume: a run killed after the trunk's save at g and before the graft is written grafts
  first on resume. This was a bug in my first version; the test of the stopped-at-the-graft
  case catches it.

**Kept apart from the E2 report.** Graft records carry `kind: "graft"`; the E2 report is not run
on the graft experiment's directories, and `docs/E2_ABLATIONS.md` was not regenerated.

## Commands for the full experiment (36-core Linux machine, CPU only)

From a checkout of `8a86bcb` with the patch applied (`git apply micro_graft.patch`), the four
sample files in `data_cache/samples/` (next section), `JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1`:

```
TOKENS=1000000 bash scripts/micro_graft_launch.sh          # or 2000000 / 3000000
DRY_RUN=1 TOKENS=1000000 bash scripts/micro_graft_launch.sh  # prints the commands below
```

which runs exactly:

```
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1
mkdir -p runs/graft_1000000
python scripts/micro_ablate.py --prepare --arms ref --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000 --threads 2 --cpus 0,1
# the 15 runs in parallel, arm i on CPUs 2i, 2i+1 (the launcher uses nohup)
python scripts/micro_ablate.py --arms ref --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/ref --threads 2 --cpus 0,1 --no-probe > runs/graft_1000000/ref.log 2>&1 &
python scripts/micro_ablate.py --arms ref_s1 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/ref_s1 --threads 2 --cpus 2,3 --no-probe > runs/graft_1000000/ref_s1.log 2>&1 &
python scripts/micro_ablate.py --arms ref_s2 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/ref_s2 --threads 2 --cpus 4,5 --no-probe > runs/graft_1000000/ref_s2.log 2>&1 &
python scripts/micro_ablate.py --arms v9_pages --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/v9_pages --threads 2 --cpus 6,7 --no-probe > runs/graft_1000000/v9_pages.log 2>&1 &
python scripts/micro_ablate.py --arms v9_pages_s1 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/v9_pages_s1 --threads 2 --cpus 8,9 --no-probe > runs/graft_1000000/v9_pages_s1.log 2>&1 &
python scripts/micro_ablate.py --arms v9_pages_s2 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/v9_pages_s2 --threads 2 --cpus 10,11 --no-probe > runs/graft_1000000/v9_pages_s2.log 2>&1 &
python scripts/micro_ablate.py --arms graft_25 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_25 --threads 2 --cpus 12,13 --no-probe > runs/graft_1000000/graft_25.log 2>&1 &
python scripts/micro_ablate.py --arms graft_25_s1 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_25_s1 --threads 2 --cpus 14,15 --no-probe > runs/graft_1000000/graft_25_s1.log 2>&1 &
python scripts/micro_ablate.py --arms graft_25_s2 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_25_s2 --threads 2 --cpus 16,17 --no-probe > runs/graft_1000000/graft_25_s2.log 2>&1 &
python scripts/micro_ablate.py --arms graft_50 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_50 --threads 2 --cpus 18,19 --no-probe > runs/graft_1000000/graft_50.log 2>&1 &
python scripts/micro_ablate.py --arms graft_50_s1 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_50_s1 --threads 2 --cpus 20,21 --no-probe > runs/graft_1000000/graft_50_s1.log 2>&1 &
python scripts/micro_ablate.py --arms graft_50_s2 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_50_s2 --threads 2 --cpus 22,23 --no-probe > runs/graft_1000000/graft_50_s2.log 2>&1 &
python scripts/micro_ablate.py --arms graft_75 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_75 --threads 2 --cpus 24,25 --no-probe > runs/graft_1000000/graft_75.log 2>&1 &
python scripts/micro_ablate.py --arms graft_75_s1 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_75_s1 --threads 2 --cpus 26,27 --no-probe > runs/graft_1000000/graft_75_s1.log 2>&1 &
python scripts/micro_ablate.py --arms graft_75_s2 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 --out runs/graft_1000000/graft_75_s2 --threads 2 --cpus 28,29 --no-probe > runs/graft_1000000/graft_75_s2.log 2>&1 &
wait
python scripts/micro_graft_report.py runs/graft_1000000 --json runs/graft_1000000/graft_report.json --merge runs/graft_1000000/results_all.jsonl > runs/graft_1000000/graft_report.txt
```

For 2M or 3M tokens replace `1000000` by `2000000` or `3000000` everywhere (a run directory
per budget: the budget is part of every arm's identity).

Notes:

* Every process writes only its own `runs/graft_1000000/<arm>/results.jsonl`. The report reads
  all of them, and `--merge` writes one combined file.
* Rerunning the same command after a kill skips finished arms and resumes killed ones from
  their last evaluation checkpoint.
* `EXTRA="--samples-dir /path"` passes options to every run. `FIRST_CPU` and `ARMS` select CPUs
  and a subset of arms, for running in two waves.
* Memory: 15 at once need about 32 GB (sum of the measured peaks: 3 x 1.72 + 3 x 1.99 +
  9 x 2.27 GB, plus 15 parents of 40 MB).
* Disk, without `--keep-states`: at most 69 MB (reference) or 237 MB (page and graft arms)
  per arm during a run. At the end: 17.6 MB per reference arm, 59.6 MB per page or graft arm,
  0.77 GB in total.
* Check `lscpu` first. If the 36 are hyperthreads, 15 x 2 threads may run slower per step than
  measured here.
* Alternative that saves 1.5 reference runs of compute per seed but takes about twice the wall
  time: run the three `ref` arms first with `--keep-states`, then the graft arms with
  `--trunk-from runs/graft_1000000`.

## Smoke run: what was run, measured costs

Commands, from the worktree (W = `scratchpad/wf/micro_graft_work`), all with
`JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 taskset -c 2,3 nice -n 5 <venv python>`:

```
scripts/micro_ablate.py --prepare --arms ref,v9_pages,graft_50 --tokens 120000 --eval-fracs 0.25,0.5,0.75,1.0 --out W/smoke --threads 2 --cpus 2,3 --samples-dir W/samples --pool-root W/pools
scripts/micro_ablate.py --arms ref,v9_pages,graft_50 --tokens 120000 --eval-fracs 0.25,0.5,0.75,1.0 --out W/smoke --threads 2 --cpus 2,3 --samples-dir W/samples --pool-root W/pools --keep-states --trunk train
scripts/micro_ablate.py --arms graft_50 --tokens 120000 --eval-fracs 0.25,0.5,0.75,1.0 --out W/smoke_loaded --threads 2 --cpus 2,3 --samples-dir W/samples --pool-root W/pools --no-probe --trunk load --trunk-from W/smoke
scripts/micro_graft_report.py W/smoke --json W/smoke/graft_report.json
```

* Budget: 120,000 tokens is 29 steps of 4,096, with checkpoints at steps 7, 14, 22 and 29
  (28,672 / 57,344 / 90,112 / 118,784 tokens). The graft step is 14, because Python rounds
  0.5 x 29 to 14, so p = 0.483 at the graft. At 1M tokens the four fractions fall on steps
  61, 122, 183 and 244 exactly.
* Machine: Intel Xeon @ 2.10 GHz, 4 CPUs shared with other jobs; one training process at a
  time.
* Software: Python 3.13.18, jax/jaxlib 0.11.1, numpy 2.5.5, pyarrow 26.0.1, tiktoken 0.14.0.

| arm | wall | s / step (timed steps) | first step (compile + 1 step) | evaluations | setup | peak RSS | parameters | train MFLOP/token |
|---|---|---|---|---|---|---|---|---|
| `ref` | 198 s | 2.368 | 13.26 s | 111.3 s (4 checkpoints x eval + 3 views) | 4.9 s | 1,716 MB | 4,402,372 | 36.55 |
| `v9_pages` | 130 s | 2.543 | 15.39 s | 29.9 s | 7.1 s | 1,987 MB | 14,904,516 | 37.67 |
| `graft_50`, trunk trained | 145 s | trunk 2.221, target 2.524 | 13.97 s, 14.90 s | 27.2 s; graft + check 17.2 s | 4.8 s | 2,271 MB | 14,904,516 | 37.67 |
| `graft_50`, trunk loaded | 86 s | target 2.463 | 14.74 s | 12.0 s; graft + check 16.6 s | 22.7 s (incl. graft) | 2,214 MB | 14,904,516 | 37.67 |

* **Compile.** About 11-13 s per training program: first step minus one step is 10.9 s for
  `ref`, 12.8 s for `v9_pages`, 11.8 / 12.4 s for the graft arm's two programs. The queue's
  timing probe measured 13 s for the reference's first step, 2.29 s per step and 1,562 MB.
  Evaluation programs compile inside the "evaluations" time.
* **Disk** (bytes on disk).

  | arm | parameters | one state checkpoint |
  |---|---|---|
  | `ref` | 17,634,823 | 51.25-51.28 MB |
  | `v9_pages` | 59,645,233 | 177.20-177.21 MB |
  | `graft_50` | 59,646,060 | |

  The graft arm's directory at the end held only its parameters: 59,646,060 bytes.
  Peak of one arm's directory without `--keep-states`, computed from these sizes: one state
  plus the parameters, 69 MB for `ref` and 237 MB for `v9_pages` and graft arms. The 5-second
  disk sampler saw 205 MB for `ref` and 709 MB for `v9_pages`, both with 3 kept states, and
  177 MB for `graft_50`; it misses transients. `--keep-states` adds 3 x 51 MB (`ref`) or
  3 x 177 MB (`v9_pages`).
* **Records.** `results.jsonl` holds about 50 KB per reference record (views) and 17-23 KB per
  other record.
* **Pools.** Built once with `--prepare` in 9.6 s wall: web train 3.25 s, chat train 5.47 s
  (substitute file), held-out pools 0.12 s and 0.14 s. A second build gave identical arrays.

**Equality and check results** (`W/logs/compare_smoke.json`, `compare_smoke_loaded.json`, script
`W/scripts/compare_smoke.py`):

* `graft_50` (trunk trained) against `ref`:
  * The held-out evaluations at 28,672 and 57,344 tokens are equal as whole dicts (losses,
    per-row cross-entropy and weight lists, tape and sink metrics).
  * The train losses at those checkpoints are equal.
  * The train-log entries at steps 0 and 10 are equal (the log keeps every 10th step).
* The graft at step 14:
  * Count 14 and step 14 carried; lr factor 0.5655; momentum 0.95.
  * Leaves: 175 copied (59 parameter leaves with 4,402,372 numbers, 114 slot arrays, the count
    and the step); 5 new parameter leaves (10,502,144 numbers) and 10 new slot arrays.
  * Check loss 6.72171688079834 before and after, difference 0.0 (tolerance 6.7e-6).
  * Held-out before = after: web 6.709457444112488, chat 6.12198586494349, assistant
    5.968481775262088. The largest per-row difference is 0.0.
* `graft_50` (trunk loaded from `W/smoke/arms/ref/states/ckpt-000002-step-000000014`, spec hash
  `7d9f1d1852d5aebc`) against `graft_50` (trunk trained): the evaluations at all four
  checkpoints are equal, the check record is equal, and the train log after the graft is equal.
* Realised token shares: chat 22.6 %, web 77.4 % (E2 at 1M: 22.1 / 77.9).

The report on the smoke records (`W/smoke/graft_report.txt`) runs. Its numbers are one seed at
120,000 tokens with the substitute chat file and are not results. For the record, at the
last checkpoint:

* mix loss: `ref` 6.2325, `v9_pages` 6.1347, `graft_50` 6.2001;
* share recovered: 0.33.

## How long one arm takes at 1M, 2M, 3M tokens (2 cores of this machine; extrapolated, not measured)

Method: the measured fixed parts plus steps x seconds per step. The fixed parts are setup,
first steps, evaluation of the 4 checkpoints, the graft and about 3 s of saving; they do not
grow with the budget. I used 2.37 s per step for the reference computation (graft trunks
measured 2.22) and 2.54 s with the page (2.46-2.54 measured).

| arm | 1M (244 steps) | 2M (488) | 3M (732) |
|---|---|---|---|
| `ref` (fixed 132 s) | 11.8 min | 21.4 min | 31.1 min |
| `v9_pages` (fixed 55 s) | 11.2 min | 21.6 min | 31.9 min |
| `graft_25` (fixed 81 s) | 11.4 min | 21.6 min | 31.7 min |
| `graft_50` | 11.3 min | 21.2 min | 31.2 min |
| `graft_75` | 11.1 min | 20.9 min | 30.7 min |

* Sequentially on these 2 cores: about 2.8 h for the 15 runs at 1M (170 min), 5.3 h at 2M
  (320 min), 7.8 h at 3M (470 min).
* On the 36-core machine with the 15 in parallel: the wall time is that of the slowest arm,
  about 12 / 22 / 32 min plus about 10 s for the pools. This assumes each pair of cores is as
  fast as here. How much 15 simultaneous runs slow each other (memory bandwidth, shared caches,
  hyperthreads) is not measured.
* Run-to-run variation of the seconds per step on this shared machine was about 7 %
  (2.22-2.37 for the same computation).

## Files a machine needs to run these arms offline

1. The repository at `8a86bcb` plus the patch. The tokenizer is in `assets/tokenizer/`
   (136 KB, sha256 `15d5b99b...cac59`) and needs `tiktoken`; the configurations are in
   `configs/micro/`. Python with the versions above (`pyproject.toml` pins jax/jaxlib 0.11.1),
   `pytest` for the tests.
2. Four sample files in `data_cache/samples/`, or anywhere with `--samples-dir`. Nothing else
   of `data_cache` is read: no `raw/`, `library/`, `opponents/`, wiki or probe sources.

| file | use | bytes | sha256 | here |
|---|---|---|---|---|
| `karpathy__climbmix-400b-shuffle/shard_00000.parquet` | web train | 92,291,918 | `054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b` | only as `data_cache/raw/karpathy__climbmix-400b-shuffle/915333b4.../shard_00000.parquet` (same revision; hash equals its download manifest); copy it |
| `karpathy__climbmix-400b-shuffle/shard_00997.parquet` | web held-out | 92,450,989 | `ddbfc6fb3b917ca33e1ebd52f5cb5d3830876ac7ed6c698f48b417da570c6079` | yes |
| `HuggingFaceTB__smol-smoltalk/data/train-00000-of-00004.parquet` | chat train | not known | not known | **missing**: download once where there is network (revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`) |
| `HuggingFaceTB__smol-smoltalk/data/test-00000-of-00001.parquet` | chat held-out | 48,177,022 | `be6773dcce145f3918ff14237b1f765affa427b0b13f6a02d397e665ac908b9a` | yes |

   Download with the repository's own helper:
   `python -c "from bes.tokenizer import samples as S; S.download(S.WEB_TRAIN[:1] + S.WEB_HELDOUT + S.CHAT_TRAIN + S.CHAT_HELDOUT)"`
   (`hf_hub_download` at the pinned revisions into `data_cache/samples/`).
3. Pools (`data_cache/micro/pools/`) are optional.
   * **Size and build time.** The four this experiment uses take 57 MB: web train 37.8 MB,
     chat train 18.9 MB (fixed array shapes, so the real one has the same size), held-out
     0.3 MB each. They build in about 9 s and are built once by `--prepare`.
   * **Copied pools are used as they are.** A pool's directory name is a digest of domain,
     split, row length, row count, pool version, tokenizer sha256 and the sample file's
     relative path, column, record range and byte size (no absolute paths). The build is
     deterministic (rebuilt identical here).
   * **Pool names, verified here.** The held-out pools built in scratch have the same names
     as computed from `data_cache/samples` (`web_heldout_T512_n64_2d6de9bf47`,
     `chat_heldout_T512_n64_6b0a2861ff`). The web training pool is
     `web_train_T512_n8192_dd82fdd7db` wherever the 92,291,918-byte `shard_00000` sits in the
     samples directory.
   * **A copied pool needs its sample file.** The file's size enters the name, so a pool
     copied without its sample file is not found: here the name became `..._5cf6ddec97`
     (size -1) and the bench would rebuild it from the missing file and fail.
   * **Conclusion.** Copy the samples; copying the pools saves 9 s. The chat training pool
     built here (`chat_train_T512_n4096_aff20b6361`) came from the substitute file and must not
     be used. It was deleted, as were the other scratch pools.

## Tests and results

New: `tests/eval/test_micro_graft.py`, 11 tests. They use the tiny model of
`test_microbench.py` (d=32, 2 loops, rows of 64) on synthetic pools: 8 steps, checkpoints at
2/4/6/8, target with output gains and 2 token neurons, graft at 0.5.

| test | what it shows |
|---|---|
| `test_the_graft_arm_is_the_base_arm_up_to_the_graft_point` | train log of steps 0-3, held-out evaluations and train losses at 2 and 4 steps equal the base arm's exactly; batch digests of all 8 steps equal those of the base and of the target arm (data order continues) |
| `test_the_function_is_preserved_at_the_graft_point` | check within the tolerance, held-out before/after per domain, carried step/count/p/lr factor, new leaves listed |
| `test_the_grafted_state_is_the_trunk_with_a_fresh_target_page` | stopped right after the graft: every leaf of the base's kept state at step 4 is carried bit for bit, the new leaves and their slots equal a fresh target tree of the same seed, consts are the target's; resumed, it ends equal to the uninterrupted run (curves and final parameters) |
| `test_a_run_stopped_at_the_graft_resumes_with_the_graft` | killed between the trunk's save and the graft: the rerun grafts first and ends equal to the uninterrupted run |
| `test_a_kept_trunk_is_loaded_and_gives_the_same_run` | trunk loaded from the base's states: same curves, check and final parameters as the trained trunk; only the target's steps run; the base keeps states at 2, 4, 6 |
| `test_trunk_lookup_and_its_refusals` | another hash is rejected and listed; the per-arm sub-directory layout is found; `trunk="load"` without a state fails before writing anything |
| `test_a_graft_fraction_must_be_an_evaluation_fraction` | refusals: fraction not an evaluation fraction, budget too small, optimiser difference, a base leaf missing in the target, other seed, fraction 1, kind without spec; round trip and hash of a graft arm |
| `test_a_graft_that_changes_the_function_fails_loudly` | a target with the tape output masses off: `RuntimeError`, a `failed` record with the numbers, not counted as finished |
| `test_the_e2_arms_keep_their_hashes_and_their_queue` | names and order of `E2_ARMS`, spec and config hashes of all 31 E2 arms equal the values computed at HEAD before the change (and 3 at the experiment's fractions); pool and sample roots do not enter the hash |
| `test_the_graft_experiment_arms` | the 15 arms, seeds, bases, targets, configurations, graft steps 61/122/183 at 1M; `graft_75` refused with the default fractions; 120,000 tokens gives 4 distinct checkpoints, 12,288 does not |
| `test_the_report_on_synthetic_results` | five families x three seeds with chosen losses in one sub-directory each: table means and standard deviations, paired differences only after the graft, share recovered 0.75 / 0.5 / 0.25, mixed budgets refused unless `--tokens`, JSON and merge written |

Results:

| command | result |
|---|---|
| `pytest tests/eval/test_micro_graft.py`, worktree | 11 passed in 65 s (2 cores) |
| same, in a clean export of HEAD with only the patch applied (no `data_cache`) | 11 passed in 67 s |
| `pytest tests/eval/test_microbench.py tests/eval/test_probes.py tests/train/test_graft_run.py` | 56 passed in 103 s |

Further checks:

* Hashes before/after: `W/scripts/head_hashes.py` on HEAD and on the final worktree gives the
  identical file `W/head_hashes.json` = `W/after_hashes.json` (spec hash, config hash and arm
  dict of the 31 E2 arms at 1M and 120k tokens, default bench and 4 fractions).
* Results before/after: `W/scripts/tiny_arms.py` trained the tiny reference and `no_tape` arms
  with the HEAD code (a `git archive` export) and with the patched code. The records were
  identical (`W/logs/tiny_head_vs_patched.txt`): arm dict, spec hash, config hash, curves
  with per-row sums, train log, skipped steps, SHA-256 of the final parameters.
* Patch: `git apply --check` on the HEAD export passes. Applied to a copy, the six files
  equal the worktree's.

## What was not done, and limits

* **The experiment itself.** Not run: the 15 runs at 1M, 2M or 3M tokens. Only the 3-arm smoke
  at 120,000 tokens with one seed ran.
* **The real chat training sample.** Not available here, so the smoke's chat training rows came
  from the held-out file. The real chat training pool was never built here, and its directory
  name is not known.
* **Extrapolation.** The 1M-3M durations are extrapolated from 29-step runs on a shared 4-core
  machine. The behaviour of 15 simultaneous runs (memory bandwidth, RSS, hyperthreads) is not
  measured.
* **Graft step at small budgets.** It is `round(frac x steps)` with Python's round-half-even: at
  29 steps the 0.5 graft is at p = 0.483. At the 1M/2M/3M budgets all graft points are exact.
* **Kept states and `--keep-states`.** Kept states are written by `run_arm` only when asked; an
  E2 run made earlier has none. `--keep-states` applies to every arm of a queue, page arms
  included (3 x 177 MB each).
* **Evaluation views.** Not computed on graft trunks; the reference arms in the experiment
  still compute their three views.
* **How this graft differs from the live one.**
  * The data stream continues unchanged here; the live run also changed seed and dropped the
    library rows at the switch.
  * The schedule is in tokens, not wall-clock.
  * The micro model is 6x narrower and sees 1-3M tokens. `docs/V9.md` section 8 explains why a
    micro run can show harm more easily than the benefit of a token table (about 120 visits per
    row at 1M), and E2 measured the page at -0.61 % at 1M against a seed-replicate noise of
    0.42 %.
  * Three seeds give a 2-degree-of-freedom standard deviation. The share recovered is a ratio
    with that gain in the denominator, so it can be unstable per seed. The report therefore also
    gives the ratio of the mean differences.
* **What was left untouched.** No commits, no `colab`, no network. `bes/config.py`,
  `docs/E2_ABLATIONS.md` and `docs/CHANGES.md` are unchanged.
* **Left in the worktree.** The intent-to-add index entries of the four new files and the
  untracked `data_cache` symlink, which is not in the patch.
