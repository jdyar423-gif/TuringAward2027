# Graft or from the start? A controlled comparison at the micro width

The 24-hour run trained `configs/final.py` to 0.505 of its budget and was then grafted into
`configs/v9.py` (`scripts/graft_run.py`, `docs/CHANGES.md` 19.7): every leaf the two share was
carried with its optimiser state, update count, step and schedule; the token pages (`page/*`)
and output gains (`gain/o`) started at their initial values, where their paths add exactly zero.
That run cannot say how a graft compares with training the pages from the first step, or with
not adding them. This experiment asks that question with the repository's own model and micro
bench (`bes/eval/microbench.py`, `scripts/micro_ablate.py`; d=128, 8 loops, 4,096 tokens per
step), with everything else held equal.

## Arms

| family | what it trains | seeds (arm names) |
|---|---|---|
| `ref` | the micro reference (`configs/micro/bench.py`) | 0 `ref`, 1 `ref_s1`, 2 `ref_s2` |
| `v9_pages` | the reference plus the v9 page from the first step (`model.gain_o=true`, `model.tok_neurons=8`, `model.tok_dim=32`) | `v9_pages`, `v9_pages_s1`, `v9_pages_s2` |
| `graft_25`, `graft_50`, `graft_75` | `ref` up to 25 / 50 / 75 % of the budget, then grafted into `v9_pages` and trained to the end | `graft_g`, `graft_g_s1`, `graft_g_s2` |

A seed fixes the data order and the initialisation; arms with one seed see bit-identical
batches at every step, and the leaves two arms share start equal. The arms are
`micro_ablate.GRAFT_ARMS` (15 runs: `micro_ablate.GRAFT_EXPERIMENT`); they are not in the E2
queue and do not change `docs/E2_ABLATIONS.md`. The spec hash of every E2 arm is the one it had
before graft arms existed (tested).

## What a graft arm does (`bes.eval.microbench.run_graft_arm`)

* **Trunk** (steps `[0, g)`, `g` = the graft step): the base arm's own computation: its
  configuration, initial parameters (`jax.random.key(seed)`), optimiser, compiled step, schedule
  and batches. It is either *loaded*, a state the base arm kept at step `g`
  (`--keep-states`; looked up in `--out` and `--trunk-from` directories, accepted only under the
  base arm's spec hash, which covers its configuration, seed, budget and bench), together with
  the base's curve and train log up to `g`, or *trained* inside the graft arm. `--trunk auto`
  (default) loads when a kept state exists and trains otherwise; `train` and `load` force one.
  `record["graft"]["trunk"]["source"]` says which.
* **Graft** (after the evaluation at step `g`): `scripts/graft_run.py` `graft_arrays` (imported,
  not copied) fills a fresh train state of the target configuration of the same seed with every
  leaf of the trunk's state checkpoint: parameters, optimiser slots (MuonH momentum and cached
  initial norms, Adam moments), the update count and the step. The target-only leaves
  (`gain/o`, `page/key`, `page/pin`, `page/pout`, `page/val`) and their slots keep the values of
  that fresh state, which are those of the `v9_pages` arm of the same seed at its first step;
  the consts are the target's. `graft_run.fresh_tree` is not used: it builds the 24-hour
  trainer's state, whose initialisation key is a split of the seed, while the micro bench
  initialises every arm from `jax.random.key(seed)`. A graft is refused before any training
  when a base leaf is missing or reshaped in the target, when the optimiser settings differ
  (except the rate and decay of a group the base has no leaf in), when the schedule differs at
  any step, or when the graft fraction is not an evaluation fraction.
* **Function check**: the training loss (`bes.model.model.loss_fn`) of the first 8 held-out web
  rows (8 x 512 positions) under the trunk and under the grafted model, and the held-out loss of
  every evaluated domain (web, chat, assistant) before and after, must agree within
  1e-6 x max(1, |loss|) (the rule of `graft_run.function_check`). Both numbers and the
  difference are recorded; a failure appends a `status="failed"` record and raises.
* **Target** (steps `[g, n)`): the target configuration at the same schedule positions
  (`p = s / n`: learning-rate factor, momentum) with the batches the base would have seen next.
  Curve points carry `phase` (`trunk` or `target`). A killed run resumes before or after the
  graft; a run stopped between the trunk's save at `g` and the graft grafts first.
* **Carried count**: the Adam slots of the new leaves start at zero while the update count is
  `g`, so their k-th update is bias-corrected with the factors of update `g + k` instead of `k`,
  which are about 1 once `g` is a few tens (`0.95^61 = 0.04`). With the bench's
  `adam_b1 = 0.8`, `adam_b2 = 0.95` and a large `g`, the page tables' updates 1, 2, 5 and 50
  after the graft are 0.89, 1.15, 1.41 and 1.04 times those of a fresh start with the same
  moments. The live graft did the same (count 29,722); the experiment keeps it.
* **What differs from the live graft**: here the data stream continues unchanged (the live run
  also changed its seed and dropped the library rows at the graft), the schedule is in tokens
  (the live run's is in wall-clock time, grafted at p = 0.511), and the graft points are 0.25,
  0.5 and 0.75.

## Running it

The graft fraction must be an evaluation fraction: all 15 runs use
`--eval-fracs 0.25,0.5,0.75,1.0` (checkpoints at 61, 122, 183 and 244 steps of a 1M-token run;
the default fractions 0.25, 0.5, 1.0 refuse `graft_75`). One process per arm, each with its own
`--out` sub-directory and CPU pair, so no two processes write one `results.jsonl`:

```
# once: the pools (about 10 s; parallel runs would otherwise all build them at the same time)
python scripts/micro_ablate.py --prepare --arms ref --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 \
    --out runs/graft_1000000 --threads 2 --cpus 0,1
# the 15 runs in parallel (arm i on CPUs 2i, 2i+1), e.g. the seventh:
python scripts/micro_ablate.py --arms graft_25 --tokens 1000000 --eval-fracs 0.25,0.5,0.75,1.0 \
    --out runs/graft_1000000/graft_25 --threads 2 --cpus 12,13 --no-probe > runs/graft_1000000/graft_25.log 2>&1 &
# when all have finished
python scripts/micro_graft_report.py runs/graft_1000000 --json runs/graft_1000000/graft_report.json \
    --merge runs/graft_1000000/results_all.jsonl
```

`scripts/micro_graft_launch.sh` does the three steps (`TOKENS=1000000 bash
scripts/micro_graft_launch.sh`; `DRY_RUN=1` prints the 17 commands). A run that is killed is
continued by the same command (finished arms are skipped, a killed arm resumes from its last
evaluation checkpoint). In this layout every graft arm trains its own trunk: the three
reference runs are still running when the graft arms start. To reuse the references' trunks
instead, run the three `ref` arms first with `--keep-states` and start the graft arms afterwards
with `--trunk-from runs/graft_1000000` (it costs a longer wall time and saves 1.5 reference runs
of compute per seed). Interrupting the launcher (Ctrl-C, a dropped session) stops its runs; it refuses to start an arm whose
`--out` still has a live process (stop the experiment with
`pkill -f 'micro_ablate.py .*--out runs/graft_<TOKENS>/'` and launch again once `pgrep` shows none).

## Cost (measured on the 4-core development container, 2 threads per run, 24 Nov 2026)

A smoke run of `ref`, `v9_pages` and `graft_50` (seed 0) at 120,000 tokens (29 steps,
checkpoints 7, 14, 22, 29):

| arm | s per step | first step (compile + 1 step) | evaluations, all checkpoints | peak RSS | arm directory (end / peak) |
|---|---|---|---|---|---|
| `ref` | 2.37 | 13.3 s | 111 s (16 evaluations: 3 views) | 1,716 MB | 17.6 MB / 69 MB |
| `v9_pages` | 2.54 | 15.4 s | 30 s | 1,987 MB | 59.6 MB / 237 MB |
| `graft_50`, trunk trained | 2.22 (trunk), 2.52 (target) | 14.0 s and 14.9 s | 27 s, graft and check 17 s | 2,271 MB | 59.6 MB / 237 MB |
| `graft_50`, trunk loaded | 2.46 (target) | 14.7 s | 12 s (2 checkpoints), graft and check 17 s | 2,214 MB | 59.6 MB / 237 MB |

The directory peak (computed from the checkpoint sizes) is one state checkpoint (51.3 MB for the
reference, 177.2 MB with the page) plus the parameters; `--keep-states` adds one state per
checkpoint before the last. The trunk-loaded run took 86 s instead of 145 s. Extrapolated
from these numbers (not measured): about 11-12 minutes per arm at 1M tokens, 21-22 at 2M and
30-32 at 3M on two such cores. Fifteen runs at once need about 32 GB of memory (the sum of the
peaks: 3 x 1.7 + 3 x 2.0 + 9 x 2.3 GB); with less, run them in two waves (`ARMS=... FIRST_CPU=...`).
How much fifteen simultaneous runs slow each other down (memory bandwidth) was not measured.

## Results (24 Nov 2026, 6,000,000 tokens, three seeds)

Run next to the 24-hour training, on the idle host cores of its runtime (AMD EPYC 9B14; one
physical core, two threads, per arm; CPU only; a separate clone of the repository), 14:12:51 to
16:41:01 UTC; 4.6 to 6.2 s per step with 15 arms at once. Budget 6,000,000 tokens = 1,464 steps of
4,096 tokens; training pools of 4.20M (web) and 1.79M (chat) tokens, so about one pass; held-out:
32 web rows and 32 chat rows. The records are in `docs/paper/evidence/N/micro_graft/6M/`
(`results_all.jsonl`, `graft_report.txt`, `graft_report.json`, the logs).

Inside the run: every graft arm's evaluations before its graft equal those of `ref` with the same
seed exactly, and at each of the nine grafts the held-out loss after equals the loss before, with
a difference of exactly 0 (per row as well).

Held-out loss at the end (5,996,544 tokens; mix of web and chat; nats per token; mean ± standard
deviation over the three seeds), and the differences paired by seed:

| arm | loss | arm - `ref` | arm - `v9_pages` | share of the page gain recovered |
|---|---|---|---|---|
| `ref` | 4.1912 ± 0.0076 | | +0.0519 ± 0.0107 | 0 |
| `v9_pages` (page from the first step) | 4.1394 ± 0.0083 | -0.0519 ± 0.0107 (-1.24%) | | 1 |
| `graft_25` (page grafted at 0.25) | 4.1363 ± 0.0075 | -0.0550 ± 0.0010 | -0.0031 ± 0.0113 | 1.06 (per seed 0.86, 1.33, 1.08) |
| `graft_50` (page grafted at 0.5) | 4.1555 ± 0.0091 | -0.0358 ± 0.0030 | +0.0161 ± 0.0096 | 0.69 (0.62, 0.87, 0.64) |
| `graft_75` (page grafted at 0.75) | 4.1789 ± 0.0074 | -0.0123 ± 0.0004 | +0.0396 ± 0.0103 | 0.24 (0.20, 0.29, 0.24) |

The share is (`ref` - graft) / (`ref` - `v9_pages`), given as the ratio of the mean differences;
per domain (web, chat, assistant turns) it is the same within 0.04. `v9_pages` - `ref` over
training: -0.0164 at 1.5M tokens, -0.0329 at 3.0M, -0.0494 at 4.5M, -0.0519 at the end; by domain
at the end: web -0.0464 ± 0.0118, chat -0.0683 ± 0.0089, assistant turns -0.0689 ± 0.0107.
One checkpoint (1.5M tokens) after its graft an arm is -0.0192 (`graft_25`), -0.0161 (`graft_50`)
and -0.0123 (`graft_75`) from `ref`; the page from the first step was -0.0164 after its first
1.5M tokens.

Reading. At this width a graft never did worse than not grafting; grafted at a quarter of the
budget the page ended where the page trained from the first step ended, grafted at half it
recovered about 0.7 of that gain, grafted at three quarters about a quarter. In its first 1.5M
tokens a grafted page gained about what the from-scratch page gained in its first 1.5M tokens:
the page learns at about the same rate whenever it is added, and what a late graft loses is the
time it did not have. A graft's difference from `ref` varies little between seeds (0.0004 to
0.0030) because the two share their trunk; the difference between two arms that never shared a
state varies by 0.010.

### Restart arms: the page configuration from scratch on the half that is left

Three more runs (16:42:47 to 17:29:56 UTC, same machine and layout): `v9_pages` from the first
step on a budget of 3,000,000 tokens (732 steps, with the schedule of that budget), seeds 0, 1, 2:
what starting again with the page would give when half of a 6M budget is already spent. Records:
`docs/paper/evidence/N/micro_graft/3M/` (`python scripts/micro_ablate.py --arms v9_pages --tokens
3000000 --eval-fracs 0.25,0.5,0.75,1.0 --out ...` per seed).

| arm, at the end of its budget | held-out loss (mix) | arm - restart, paired by seed |
|---|---|---|
| restart: `v9_pages`, 3M tokens | 4.3897 ± 0.0017 | |
| `graft_50`: `ref` for 3M tokens, then the page for 3M | 4.1555 ± 0.0091 | -0.2342 ± 0.0075 |
| `ref`, 6M tokens, no page | 4.1912 ± 0.0076 | -0.1985 ± 0.0062 |
| `v9_pages`, 6M tokens | 4.1394 ± 0.0083 | -0.2504 ± 0.0074 |

The restart's own schedule is already counted: it ends 0.0509 ± 0.0086 below `v9_pages` at the
same token count inside the 6M-budget run. With half of the budget spent without the page,
continuing with a graft ended 0.234 nats below starting the page configuration again, and 0.016
above having had the page from the first step; the tokens already trained were worth about four
times the page. The restart read the first half of the data order again rather than the unseen
second half (about one pass over the pools in all); that this does not matter was not tested.

Limits. d_model 128 against the 768 of the live run; 6M tokens against 1.36e10; one pass over
small pools; three seeds, so two degrees of freedom, and a share whose denominator is itself a
difference of two runs (the per-seed shares of `graft_25` range from 0.86 to 1.33); the schedule
counts tokens, not wall-clock time; the data stream does not change at the graft, as it did in
the live run; the page is worth 1.2% here. This is evidence about the mechanism of a graft, not
about the size of its effect in the live run.

## Report (`scripts/micro_graft_report.py`)

Reads `results.jsonl` in the given directories and their sub-directories (newest finished
record per arm; one budget) and prints: (a) per arm and checkpoint the held-out loss of `mix`
(web and chat weighted by the training row shares, as in E2), web, chat and assistant, mean and
standard deviation over seeds; (b) paired by seed, `v9_pages - ref` at every checkpoint,
`graft_g - ref` and `graft_g - v9_pages` at every checkpoint after the graft point: mean,
standard deviation of the paired differences, number of seeds and the per-seed values;
(c) the share of the page gain recovered at the end, `(ref - graft_g) / (ref - v9_pages)`, per
seed, their mean and the ratio of the mean differences. `--json` writes the same numbers.

## Files a machine needs (offline)

The repository (tokenizer in `assets/tokenizer/`, configurations in `configs/`) and four sample
files under `data_cache/samples/`, read with pyarrow (no network access):

| file | use | bytes | sha256 |
|---|---|---|---|
| `karpathy__climbmix-400b-shuffle/shard_00000.parquet` | web training rows | 92,291,918 | `054ddbd9...25bf9b` |
| `karpathy__climbmix-400b-shuffle/shard_00997.parquet` | web held-out rows | 92,450,989 | `ddbfc6fb...0c6079` |
| `HuggingFaceTB__smol-smoltalk/data/train-00000-of-00004.parquet` | chat training rows | (not measured) | |
| `HuggingFaceTB__smol-smoltalk/data/test-00000-of-00001.parquet` | chat held-out rows | 48,177,022 | `be6773dc...ac908b9a` |

They are files of pinned revisions (`bes/tokenizer/samples.py`: ClimbMix `915333b4`,
smol-smoltalk `f73fe857`). On a machine with network access,
`python -c "from bes.tokenizer import samples as S; S.download(S.WEB_TRAIN[:1] + S.WEB_HELDOUT + S.CHAT_TRAIN + S.CHAT_HELDOUT)"`
puts them in `data_cache/samples/`; `--samples-dir` reads them from elsewhere. Nothing else of
`data_cache` is used (no library, no `raw/`, no wiki or probe sources).

The pools under `data_cache/micro/pools/` are optional: the four this experiment uses (web and
chat training, 8,192 and 4,096 rows, 37.8 and 18.9 MB; web and chat held-out, 64 rows, 0.3 MB
each) were built in 9 s in total here. A pool's directory name is a digest of its domain, split,
row length, row count, pool version, tokenizer hash and the sample file's relative path, column,
record range and size in bytes (`microbench.pool_key`), and the build is deterministic, so a pool
built on another machine from the same files is found and used as it is (for instance
`web_train_T512_n8192_dd82fdd7db`, `web_heldout_T512_n64_2d6de9bf47`,
`chat_heldout_T512_n64_6b0a2861ff`). The size is read from the sample file, so a pool copied
without its sample file is not found: the name changes and the bench tries to build it again
from the missing file and fails.
