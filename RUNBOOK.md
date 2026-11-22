# RUNBOOK: the 24-hour run on one TPU v6e

What to do before, during and after the single 24-hour run of `docs/PLAN.md` section 6. Read it
once end to end before you start. **Nothing in this repository has been measured on a TPU**: the
code was developed and rehearsed on a 4-core CPU container, and the TPU facts below are
compile-only estimates (`docs/TPU_NOTES.md`). The first 40 minutes of the run are where those
estimates meet the chip; section 3 says what you must see there and what each failure means.

Contents: 1 before the run, 2 starting it, 3 the first 40 minutes, 4 during the run,
5 resuming, 6 the fallback ladder, 7 after the run (export, evaluation, arena), 8 reproducing
the accounting, 9 known unknowns, 10 honest expectations, 11 where things are.

## 1. Before the run

### 1.1 Accounts and resources

| what | why | notes |
|---|---|---|
| Google account with Colab access to a **TPU v6e-1** runtime | the run | a paid tier (compute units); TPU availability varies by region and time |
| about **72 GB of local disk** on the runtime (`/content`) | downloads (44 GB at the registry caps), token shards (61 GB), the library, checkpoints, the compilation cache | the estimate and its rule are in `manifest.json` (`disk_need`); a raw download is deleted once its whole token shard is committed (without that: 114 GB). The preflight stops the run on a TPU when the free disk under any data path is below the estimate, or when a data path is on Drive |
| Google Drive with about **10 GB free**, or less if you empty its trash during the run | the mirror: the run's records, logs, the newest checkpoint and the export | optional but strongly advised: a recycled runtime loses the local disk. Drive holds at most about 1.3 GB at a time for `configs/final.py`: records, ledgers, the loader journal, samples and logs (under 0.3 GB), the newest training checkpoint (0.35 GB: parameters 138 MB, optimiser moments 216 MB; two while one replaces the other), the post-DPO weights (0.14 GB) and the export (about 0.25 GB with the library). Files replaced or deleted through the Colab Drive mount go to the Drive trash and keep counting against the quota until it is emptied (reported by users, not tested here): with one checkpoint copy per hour that is about 20 x 0.35 GB = 7 GB over 19 hours of training, plus rewritten logs and ledgers. Empty the trash (drive.google.com, Trash) every few hours, or keep about 10 GB free. **No data goes to Drive** (no downloads, shards, library or models) |
| Hugging Face account and a read token | downloads of data, library sources and GGUF files | not strictly required (every file used is public and ungated), but it raises rate limits; entered at run time, never stored |
| access to this repository | the notebook clones it (`REPO_URL`, `BRANCH`) | a private repository: put a copy on Drive and give it as `CODE_COPY` in cell 1. Cell 3 copies the code to the local `REPO_DIR` without `data_cache/` and `runs/` and runs it there. Never point `REPO_DIR` at Drive (cell 3 refuses): every data path of the run is relative to the code |
| about 25 hours of attention budget | the run itself (24 h) plus installation (up to 20 min) and the evaluation afterwards | the run needs no attention between the first 40 minutes and the end, except a look every few hours |

**Colab session limits.** The 24-hour budget is wall time from the moment `scripts/run_24h.py`
creates the run; installation happens before. Colab runtimes have a maximum lifetime that depends on
the tier and the accelerator; check yours before you start (this was not verified for TPU v6e
runtimes). If the runtime may not outlive install + 24 h, set `BUDGET_MINUTES = 1380` (23 hours) in cell 1:
the post-training reserve and the export keep their shares, training gets shorter. A runtime that
dies anyway is handled by section 5 (resume on a new runtime; the downtime counts against the
budget).

### 1.2 Rehearse

1. **Tests** (CPU, any machine): `python -m pytest tests/<area>` for the areas you changed; the whole
   suite takes several minutes per area (`-m slow` adds the long ones).
2. **CPU rehearsal of the whole pipeline** (12 minutes, about 2 cores):

       python scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 12 --out runs/smoke

   It must exit 0 with every stage finished or skipped with a reason (`runs/smoke/stage_table.json`)
   and an export in `runs/smoke/final/` whose `MODEL_CARD.md` reports a passed held-out check and
   the library decision. Kill it with SIGKILL in the middle of training and continue with
   `--resume` once, to see the resume. It uses `data_cache/library/lib_v2` when that library was
   built with the run's settings (a library built by hand with `scripts/build_library.py` is used
   when what it records itself matches; `README.md`), else it builds it again in the background.
3. **Free-tier smoke notebook** (optional, not part of the budget): `notebooks/colab_v5e_smoke.ipynb`
   runs the same pipeline with `configs/smoke.py` and a 30-minute budget on whatever TPU the free
   tier gives (v5e expected). It shows that installation, downloads, compilation on a real TPU,
   the background tasks and the export work on Colab.
4. **One-hour canary on the real chip** (optional, strongly advised if you have the compute units):
   `python scripts/run_24h.py --config configs/canary.py --out runs/canary` runs the final model
   and every stage in one hour with the first files of the main sources only. It is the only way to
   see the benchmark, the canary and real step times before committing 24 hours. Its library is the
   default (not permissive-only) one; a later run with other library settings moves it aside and
   builds its own (section 3).

### 1.3 Decisions to make before you start

- **Configuration.** `configs/final.py` (default; standard accounting convention),
  `configs/strict.py` (both accounting conventions under 20M at T = 2048 for windows of up to
  about 918 distinct tokens by the budget's upper bound, real 2048-token text held at most 837:
  feed-forward width 3072 -> 2688 and critic head 256 -> 128, 1.28M fewer numbers read per token,
  6.7% of the dense read; `docs/ACCOUNTING.md`), `configs/no_library.py` (no written library),
  `configs/safe.py` (every feature path switched off: the most conservative use of the 24 hours).
- **Licences.** ClimbMix (the main web source) is CC BY-NC 4.0 upstream; `--override
  data.permissive_only=true` replaces it by FineWeb-Edu (`docs/DATA_SOURCES.md` section 4) and
  builds the library without the question-answer sources whose licence is unknown (TriviaQA,
  WebQuestions; NQ-open, CC BY-SA 3.0, stays). A library on disk built with other settings (by
  the canary, for example) is moved aside and rebuilt; the model card lists the question-answer
  sources the shipped library stores, with their licences.
- **Budget** (see 1.1) and the **Drive mirror** (`USE_DRIVE`).

Overrides and the budget are fixed when the run starts: a resume refuses different ones.

## 2. Starting the run

Open `notebooks/colab_v6e_24h.ipynb` in Colab on a TPU v6e-1 runtime and run cells 1, 1b and 2 to
5, then **cell 6 to start a new run** (or cell 8 to continue one: section 5):

| cell | what it does | what you must see |
|---|---|---|
| 1 | parameters: `REPO_URL`, `BRANCH` (or `CODE_COPY`, a copy of the repository, e.g. on Drive), `REPO_DIR` (local), `RUN_DIR`, `USE_DRIVE`, `DRIVE_DIR`, `CONFIG`, `BUDGET_MINUTES` | |
| 1b | helper functions (start in the background, the Drive mirror and its status, progress) | nothing |
| 2 | runtime check in a separate process | `backend: tpu`, one `TPU v6 lite` device; a warning if jax is not 0.11.1 (it is not changed) or the local disk has under 80 GB free |
| 3 | mounts Drive (when `USE_DRIVE` or `CODE_COPY` is on Drive); copies `CODE_COPY` to `REPO_DIR` without `data_cache/` and `runs/`, or clones `REPO_URL`; refuses a `REPO_DIR` on Drive | the commit hash |
| 4 | installs `requirements-colab.txt` with jax, jaxlib and libtpu held at their installed versions; checks they did not change; stops when the imports fail | `imports ok` (llama-cpp-python builds from source: 5 to 15 minutes) |
| 5 | Hugging Face token: a Colab secret `HF_TOKEN` if you granted the notebook access, else a hidden prompt; kept in the kernel's memory only | `HF_TOKEN set` |
| 6 | starts a **new** run: refuses when `RUN_DIR` or `DRIVE_DIR` already holds one (continue that with cell 8). Starts `scripts/run_24h.py --config configs/final.py --out RUN_DIR` in its own session (it survives a kernel restart) with its output in `RUN_DIR.log`, and the Drive mirror (a pass every 10 minutes) | `started pid ...` |
| 7 | progress: stage table, benchmark and canary, the last training steps, background tasks, the Drive mirror (last good pass, errors, the checkpoint on Drive, a warning when it is stale), the log tail | run it whenever you like |
| 8 | continues a run (section 5) | `resumed: pid ...` |
| 9 | stops the run, only after you set `STOP_THE_RUN = True` in the cell: Runtime > Run all never stops a run | |

The notebook never imports JAX in its own kernel before the chat and arena cells: whichever
process imports JAX first on the TPU host owns the TPU.

Without the notebook (any TPU VM with this repository on a local disk and the requirements
installed), from the repository's directory:

    mkdir -p runs && nohup python scripts/run_24h.py --config configs/final.py --out runs/final > runs/final.log 2>&1 &

## 3. The first 40 minutes

The plan's timeline (`docs/PLAN.md` section 6) puts installation, downloads, the benchmark and the
canary in the first 40 minutes. Every line below appears in `RUN_DIR.log` (prefixed with the
minutes since the start); cell 7 summarises them. The benchmark's numbers are also in
`RUN_DIR/manifest.json` (key `benchmark`) and `RUN_DIR/benchmark/results.jsonl`.

| when | log line (abridged) | expected | if not, what it means and what to do |
|---|---|---|---|
| 0 min | `preflight: jax 0.11.1 backend tpu devices ['TPU v6 lite']; N cores, R GB RAM, D GB free disk, HF_TOKEN set` | backend `tpu`, one `TPU v6 lite`; free disk under every data path (`manifest.json` `host.data_free_gb`) at least `disk_need_gb` (about 72 GB for the default plan) | `PreflightError: no TPU`: wrong runtime type. `data paths on a network drive`: the code runs from Drive; use `CODE_COPY` (section 2). Disk below the estimate: the run stops before anything starts; free disk or lower the download caps (`--override run.download_sources="climbmix:200 ..."`, see `bes/train/supervisor.py` `download_plan`) |
| 0 to 1 min | `background: started downloads / tokenize / library / gguf / opponents (pid ...)` | five background tasks running (`RUN_DIR/background.json`, logs in `RUN_DIR/logs/<task>.log`) | a task that fails is restarted with back-off; `FAILED after N failures` in the log means it gave up: read its log. Downloads failing: network or rate limits (set `HF_TOKEN`) |
| 1 to 10 min | `benchmark: training shards found; building the benchmark loader` | the first ClimbMix file (92 MB) and the first file of each chat source downloaded and tokenised (one ClimbMix file tokenised in about 35 s per core in the development container) | `no training shards under ... after N s`: the benchmark fails, an emergency export runs and the run stops (exit 1). Read `logs/downloads.log` and `logs/tokenize.log`, fix, then `--resume` |
| | `benchmark: loader R rows/s with W workers` | at least the rows the device consumes: `A x B / step time`, about 390 rows/s of T=1024 at 400k tokens/s (gate E6 target: 350 sequences/s) | below the need: the benchmark raises the loader workers (up to the core plan's maximum) and the horizon uses the slower of device and loader. A host-bound run trains on fewer tokens |
| 2 to 20 min | one measurement per execution plan (also `benchmark/results.jsonl`) | the configured plan `splash-remat1-ms32-onehot` compiles and runs at T=1024 and T=2048. Compile-only estimates: peak 4.21 GiB with remat, 17.03 GiB for the no-remat plan with half the micro-batch; the limit is `min(28 GiB, 90% of HBM)`. Throughput: the plan expects 300k to 450k tokens/s (MFU 25-40%) | a plan with `RESOURCE_EXHAUSTED` or another error is skipped with its reason. `BenchmarkError: no execution plan works`: the run stops; compare jax/libtpu with `docs/TPU_NOTES.md`, try `--override model.attn_impl=xla` (dense attention) in a new run directory. Much lower throughput than expected is not an error: the horizon (below) adapts, the model gets fewer tokens |
| | `canary: determinism True, parity {...}` | parity: the chosen plan's loss within 2e-2 (relative, bfloat16) of the dense attention's; determinism: the same batch twice gives the same loss bit for bit | parity failed: `PARITY FAILED ... falling back to the dense attention` (slower, more memory: compile-only peaks 5.33 / 7.13 GiB), the run continues. Determinism failed: a warning; a restart will not reproduce the trajectory bit for bit, nothing else changes |
| | `canary: safe configuration: first X last Y ok=True` and `canary: full configuration: ... ok=True` | both runs from the same initial parameters lower the loss (the mean of the last fifth at most 0.98 x the mean of the first fifth), on **every compiled shape of the run** (T=1024 and T=2048), with no skipped update | full fails and safe passes: `FALLBACK LADDER rung k: [...] switched off` (section 6), the run continues on that rung. Safe fails: `CanaryError`, the run stops: the plain looped transformer does not train on this machine (data, optimiser or numerics); do not spend the 24 hours before understanding why |
| 15 to 25 min | `benchmark: chosen plan ... (T tok/s expected)` and `benchmark: horizon H tokens (T tok/s x S s); loader workers W` | the horizon is the expected number of training tokens; at 300k to 450k tokens/s over about 19 hours, 2.0e10 to 3.1e10 | it sets the learning-rate transfer rule and the epoch caps once, for the whole run (also after a resume). A tiny horizon means a slow chip or loader: the run still works, the model will be weaker |
| 25 to 40 min | `train: set-up (compile 2 shapes, 2 reused from the benchmark, restore) took N s; step 0` | the trainer reuses the steps the benchmark compiled in the same process (the learning rates are run-time inputs of the compiled step); after a restart it compiles them again, from the persistent compilation cache under `RUN_DIR/jax_cache` when the backend uses it (`manifest.json` `jax_cache`) | a compile failure here is a failure of an essential stage: emergency export, exit 1 |
| from 40 min | `train: step S p P loss L T 1024 tok/s R` (every 500 steps) | the loss starts near ln 8192 = 9.0 nats (lower with the candidate tape) and falls; tokens/s close to the benchmark's | `train: rollback ...` events: a loss spike, a non-finite loss or an update skipped for a non-finite gradient rolled back to a checkpoint and cut the learning rate by 0.7 (at most 3 times; a checkpoint more than three intervals old is not restored, the cut still applies). Repeated rollbacks: section 4 |
| 10 to 30 min | `library: ready (...)`, then `library ready: loader.set_library at data index ...` | the library task found `data_cache/library/lib_v2` built with the run's settings, or built it, and it passed its self-check (the build took about 9 minutes of host time in the development container, plus downloads) | `library: ... not usable ([...]); building it`: the library on disk was broken or built with other settings (permissive flag, question-answer sources, tokenizer, `LibraryCfg`); it was moved to `lib_v2.aside-<time>-<id>` and is being rebuilt. `background: library FAILED`: training continues without library passages (their fractions stay 0); the export ships without a library |

## 4. During the run

- **Look every few hours** with cell 7: the stage table, the newest step (loss, tokens/s), the
  training ledger (tokens, hours, lost time, restarts, rollbacks), the background tasks, the Drive
  mirror and the side-car health report (`RUN_DIR/health/latest.json`, every 2 hours: probe loss
  per source and two sample replies with exact scores; the chat, rag and math probes come from the
  loader's held-out split, the others from the training universe, so read those as a trend).
- **The Drive mirror** writes `RUN_DIR.mirror.json` after every pass. Cell 7 prints its last good
  pass, the errors (total and in the last pass, with the last message), the checkpoint on Drive and
  the free space it sees, and `WARNING: the Drive copy is stale or the mirror stopped` when no good
  pass happened for two periods. Then read `RUN_DIR.mirror.log`: a full quota (empty the Drive
  trash), a stale mount (remount: `drive.mount("/content/drive", force_remount=True)`), or a Drive
  directory that holds another run (the mirror never touches it). A mirror that died is started
  again with `start_mirror()` in a cell.
- **Checkpoints**: atomic, every 25 minutes, the newest three kept on the local disk
  (`RUN_DIR/train/ckpt/`); free space is checked before each save and the oldest are pruned first.
  Drive gets the newest complete one: copied into a temporary directory, renamed when complete,
  and only then is the older one on Drive removed; at most once an hour while training runs, and at
  once when training is over.
- **Rollbacks**: the guard rolls back after 3 consecutive bad steps (spike above 1.3 x the moving
  average, non-finite loss, or a skipped update) and cuts the learning rate by 0.7, at most 3 times.
  More than that, or a loss that climbs for hours: within the first 2 hours you may restart
  training from scratch once (`python scripts/run_24h.py --out RUN_DIR --resume --restart-training`,
  after stopping the run); later only rollbacks are allowed (plan section 6).
- **Stopping**: set `STOP_THE_RUN = True` in cell 9 and run it: it sends SIGTERM once, the trainer
  stops after its step and writes a final checkpoint, the run exits with status 3, and `--resume`
  (cell 8) continues it. A second SIGTERM exits at once.
- **After training** (from about 19:30 of the budget) the run samples (prompts from the loader's
  held-out split first), fits the critic, runs one epoch of DPO (kept only if its paired, noise-aware
  health checks on 64 held-out prompts pass), runs the decode selection, the judged comparison on
  the held-out prompts and the export. The decode selection chooses the decoding settings by exact
  scores on the dev prompts and makes the **library decision**: in the same stage (about 40% of its
  slice) factual questions are answered with the library on and off, the questions on which the
  gate fires first, replies capped at 256 tokens (`bes.eval.proxy.factual`); the library ships only
  if the sign test of helped against hurt gives p < 0.05 with more helped, and the dev exact score
  does not drop with it. Otherwise the decoding settings switch the library off and the export does
  not ship or count it (`manifest.json` `decode.library`; the report in
  `data_cache/proxy/reports/factual/`). When time runs short, stages are shrunk and then skipped in
  the order buffer, judge, critic, sample, select, prefopt; the export always runs
  (`bes/train/stages.py`).

## 5. Resuming

After a crash, a kill, a stop, a disconnected or recycled runtime:

1. On a recycled runtime, run cells 1, 1b and 2 to 5 again (code, dependencies, token). Do not run
   cell 6: it starts a new run (it refuses when Drive holds the run).
2. Run cell 8. When the local run directory is gone it copies the mirrored run back from Drive
   (the records, the ledgers, the loader journal, the samples and reports, the logs and the newest
   checkpoint), then starts `python scripts/run_24h.py --out RUN_DIR --resume` in the background.

The resumed run keeps its budget (the downtime counted against it), its configuration, overrides
and benchmark result (the horizon is reused, not re-measured), continues at the first unfinished
stage, restarts the background tasks and restores the newest loadable training checkpoint. On a
recycled machine the data is not on Drive: the background tasks download, tokenise and build it
again, the library included (its build is deterministic, so it is the same library). A training
that already used library passages waits for the library before it goes on (about 10 minutes of
host time plus downloads); the checkpoint on Drive may be up to an hour older than the last local
one (the hourly copy). A second supervisor in the same run directory is refused (`RunLocked`).

## 6. The fallback ladder

Every new mechanism is zero-initialised and switched by a runtime constant, so one compiled
program covers the ladder from the full model down to the plain looped transformer
(`bes/train/supervisor.py` `LADDER`, plan section 6). The canary decides the rung before training:

| rung | switched off (cumulative) | switches |
|---|---|---|
| 0 | nothing: the full configuration | |
| 1 | attention sink | `sink:0` |
| 2 | + confidence channel | `conf:0` |
| 3 | + n-gram count heads (candidate-tape input features) | `cand_in:0` |
| 4 | + library passages in training rows | `data.library_frac_base=0`, `data.library_frac_chat=0` |
| 5 | + candidate tilt (candidate masses in the output law) | `cand_out:0` |
| 6 | + registers and structural clocks | `reg:0 clock:0` |
| 7 | + calculator spans (no model switch: recorded only) | |
| safe | every feature path off (`configs/safe.py`) | `reg:0 cand_in:0 cand_out:0 conf:0 sink:0 clock:0 kind:0` |

Which rung the run is on:

    python -c "import json; print(json.load(open('RUN_DIR/manifest.json'))['benchmark']['ladder'])"

prints `{"rung": k, "groups": [...], "switches": "...", "overrides": [...]}` (rung 0: `groups` is
empty). The log says `benchmark: FALLBACK LADDER rung k: [...] switched off` when it is not 0, cell
7 prints it, and the export's `MODEL_CARD.md` states it in its first paragraph. To choose a rung
yourself, start a new run with `--override run.switches="sink:0 conf:0"` (plus the data overrides
of rung 4 and above). Whether the particle decoder and the library ship is decided after training
by the decode selection (section 4) and does not touch the weights.

## 7. After the run

### 7.1 The export

`RUN_DIR/final/` (`bes/export.py`; again by hand: `python scripts/export.py --run RUN_DIR`):

- `weights16.npz`: every parameter as a 16-bit number (float16 or bfloat16 per leaf, whichever
  reproduces the float32 values with the smaller error; `weights.json` lists both errors per leaf);
- the **held-out check**: the loss of the 16-bit weights differs from the float32 loss by less than
  0.002 nats on held-out rows (ClimbMix shard 997, beyond the download cap; the smol-smoltalk test
  split; the 400 proxy prompts). If it does not hold, the model card says so at the top and
  `weights32.npz` is kept next to the 16-bit file;
- `config.py` / `config.json` (the effective configuration), `tokenizer/`, `library/` and
  `SHA256SUMS` (`python scripts/export.py --verify RUN_DIR/final`). `library/` holds exactly the
  library's counted tables (a 32-bit table counts two numbers per element), the JSON files that
  configure them and its attribution text; its provenance arrays, build reports and the build
  settings record stay behind, and the export checks that the shipped tables add up to the
  accounted total. There is no `library/` when the decode selection's library rule did not pass;
- `MODEL_CARD.md`, generated from measured facts only: a first paragraph built from the record
  (the feature paths on and off, the library, K), the accounting table with the real library, the
  read-set audit, the training ledger and stage table, the post-training and evaluation records
  that exist with the library decision and its numbers, licences and attributions (the
  question-answer sources the library stores, with their licences) and the disclosures of plan
  section 2 (library on or off, particles K, external models for scoring only).

The mirror copies it to Drive (`DRIVE_DIR/final`); notebook cell 12 copies it once more, with the
arena votes, and verifies it, before the runtime goes away.

### 7.2 Evaluation

The run itself measured the decoding variants on the dev half by exact scores and the library on
and off on factual questions (`manifest.json` `decode`), and judged our chosen setting against
each opponent arm on the held-out half, partially if time ran out (`manifest.json` `judged`;
reports in `data_cache/proxy/reports/`). With no time limit any more, complete and extend it
(`docs/EVAL_PROTOCOL.md` section 9 for every command):

    G="--generator bes.export:export_generator --generator-arg export=RUN_DIR/final"
    python -m bes.eval.proxy run --model ours $G --split test
    python -m bes.eval.proxy run --model ours $G --split test --preset k1        # K = 1 (disclosure)
    python -m bes.eval.proxy run --model ours $G --split test --preset nolib     # library off
    python -m bes.eval.proxy runs                                                # run ids
    python -m bes.eval.proxy exact <our run id> --split test
    python -m bes.eval.proxy pairs --x <our run id> --y <opponent run id> --name final-vs-falcon --split test
    python -m bes.eval.proxy judge --pairs final-vs-falcon --judge qwen3-4b-2507 --threads 16
    python -m bes.eval.proxy judge --pairs final-vs-falcon --judge gemma4-e4b --threads 16  # second family
    python -m bes.eval.proxy score --pairs final-vs-falcon
    python -m bes.eval.proxy rate --pairs final-vs-falcon --rater <you>          # human ratings

The `ours` presets of the harness take `bes.config.DecodeCfg` defaults; to reproduce the run's own
setting add its decode overrides with `--set` (they are in `RUN_DIR/final/config.json` under
`decode_choice.overrides`). Report the control arms too (each opponent with the same K, keeping its
best candidate by its own log-probability: `--samples K --select logprob_mean`). Read exact scores
first, then judged scores with their intervals; a difference under about 5 points is not resolvable
on 400 prompts (`docs/EVAL_PROTOCOL.md` section 8). The whole factual measurement (every
question, without the run's time limit) needs the library, which an export whose library rule
did not pass does not ship; run it on the run's checkpoint with the built library (add the run's
`model.*` / `decode.*` overrides of `manifest.json` `effective.overrides` with
`--generator-arg overrides=...`):

    python -m bes.eval.proxy factual --generator bes.decode.respond:our_generator \
        --generator-arg checkpoint=RUN_DIR --generator-arg config=configs/final.py \
        --generator-arg library=data_cache/library/lib_v2

### 7.3 The blind arena

    python scripts/arena.py --export RUN_DIR/final                      # you type the prompts
    python scripts/arena.py --export RUN_DIR/final --prompts proxy:test --name test-split
    python scripts/arena.py --export RUN_DIR/final --reveal             # identities after each vote

(notebook cell 11 runs the same in a cell). Every prompt is answered by our model and by the
opponents (`falcon=recommended`, `mobilellm=recommended` by default: the harness's GGUF files and
settings); the replies appear only when all are ready, in a new random order under A, B, C; you
vote `a`/`b`/`c`, `t` (tie), `x` (all bad), `s` (skip), `u` (undo), `q` (quit). Hidden reasoning is
never shown. Votes and tallies with 95% Wilson intervals are saved after every vote in
`data_cache/arena/<name>/`; the same command continues the arena. Do not open `key.jsonl`,
`arena.json`, `tallies.*` or `cache/` until you have finished voting: they name the models. Style can
still give a model away (`\boxed{}` answers); that cannot be hidden.

### 7.4 Chat

    python scripts/chat.py --export RUN_DIR/final --show-diagnostics
    python scripts/chat.py --export RUN_DIR/final --particles 1 --no-library --temperature 0.7

`--show-diagnostics` prints, after each reply, the retrieved library passage, the constraints
compiled from your message, the particles kept and dropped and the calculator spans. While a run
still trains, add `--device cpu`.

## 8. Reproducing the accounting numbers

The definitions are in `docs/ACCOUNTING.md`: a parameter is a stored number of at most 16 bits; the
total counts every shipped number (weights, tables, the library's compressed bytes / 2, the
retrieval index); the per-token count is the standard convention (numbers read to produce one
token's distribution given the cache), with the strict convention (union over the context window)
reported next to it.

- The export's numbers: `RUN_DIR/final/export.json` keys `accounting`, `readset`, `weights` and
  `library` (`numbers`: every shipped table against the accounting), and the tables of
  `MODEL_CARD.md`.
- Recompute them from the export:

      python - <<'EOF'
      from bes.export import load_export, accounting_report
      ex = load_export("RUN_DIR/final")
      acc = accounting_report(ex.params, ex.cfg.model, ex.library)
      print(acc["total"], acc["active_step"], acc["within_caps"])
      for row in acc["rows"]:
          print(row)
      EOF

- The tests that enforce the budget on the shipped configurations and check the read-set audit
  against a NumPy reference that sees only the read rows:
  `python -m pytest tests/model/test_config_budgets.py tests/model/test_roles_budget.py tests/model/test_readset.py`.
- The library's own numbers: `docs/E1_COVERAGE.md` (63,768,391 stored numbers for `lib_v2`, worst-case
  read of one retrieval 245,223).

## 9. Known unknowns

Everything here could only be estimated without the hardware; the first hour of the run (or the
one-hour canary) measures most of it.

- **The chip**: step time, MFU and tokens per second; real HBM use against the compile-only peaks;
  compile time on the TPU VM host (minutes per program on two CPU cores of the compile host); the
  flash kernel's numerics and speed against the dense attention; bit-for-bit determinism; whether
  the persistent compilation cache is hit after a restart (`docs/TPU_NOTES.md`).
- **The host**: the loader's rows per second with 4 to 6 workers at the real batch size; the
  number of host cores of the Colab TPU VM (the core plan adapts to it); the side-car's cost at full
  model size; the decoding speed of the particle sampler (prompts are sampled one after another, so
  the sampling stage and the factual library measurement will be partial: with few questions
  answered the library is not shipped); the 4B judge on the host CPU (the judge stage will be
  partial).
- **Colab and Drive**: download speed from Hugging Face, the disk (about 72 GB needed by the
  default plan), the build time of llama-cpp-python, whether install + 24 hours fits one runtime,
  and whether files replaced or deleted through the Drive mount fill the Drive trash as reported
  (section 1.1).
- **The recipe**: the learning-rate transfer rule is extrapolated from its authors' sweep to about
  2e10 tokens (`docs/CHANGES.md` section 6); the evidence gates E2 and E4 had not been run when
  this was written (`docs/E2_ABLATIONS.md`, `docs/E4_PROBES.md`); E5 was measured on a 1M-token
  micro model only (`docs/E5_SELF_GENERATION.md`); the particle decoder was measured only on an
  opponent model as a stand-in (`docs/E9_DECODING.md`).
- **The export**: the held-out check of the 16-bit weights has only been run on smoke-sized models.

## 10. Honest expectations

From `docs/PLAN.md` section 10, unchanged by anything measured since:

- Against **MobileLLM-R1.5-140M** the plan estimates winning odds of about 65 to 85%; against
  **Falcon-H1-Tiny-90M-Instruct** about **30 to 50%** (an adversarial model gave 41% at baseline and
  48 to 58% with every item of plan section 5; the plan calls that optimistic: the opponent has a 74M
  dense core and about 22 times the training compute).
- The gap is decided mostly by post-training, data and decoding (plan section 5), not by the
  architecture: every known way of rearranging 20M active parameters is worth single-digit percent
  (`docs/NEGATIVE_RESULTS.md`).
- The blind comparison cannot resolve differences below about 5 points on 400 prompts; treat a
  smaller difference as no evidence either way.
- The reviewers scored the novelty of the mathematics 2 to 3 out of 10: rigorous, but not a
  breakthrough, and nothing here should be presented as one.

## 11. Where things are

| path | content |
|---|---|
| `RUN_DIR/manifest.json` | configuration, overrides, budget, code commit, host and devices (free disk under every data path), core plan, benchmark (plans, canary, ladder, horizon), decode choice with the library decision, judged results |
| `RUN_DIR/stages.json`, `stage_table.json` | the stage ledger (status, planned, allotted and used seconds) |
| `RUN_DIR/train/` | `train_log.jsonl` (step records), `ledger.json` (time per phase, tokens, steps, restarts, rollbacks), `ckpt/`, the loader journal |
| `RUN_DIR/benchmark/results.jsonl` | every plan measurement |
| `RUN_DIR/post/` | samples (labelled on the whole text of their tokens), critic report, DPO report and the post-DPO checkpoint when accepted |
| `RUN_DIR/health/` | side-car reports |
| `RUN_DIR/logs/`, `RUN_DIR/background.json`, `RUN_DIR/state/` | background tasks: logs, states, finished markers (this machine's: not mirrored) |
| `RUN_DIR/jax_cache/` | the persistent compilation cache (not mirrored) |
| `RUN_DIR/final/` | the export (section 7.1) |
| `RUN_DIR.log`, `RUN_DIR.mirror.log`, `RUN_DIR.mirror.json` | the run's output, the Drive mirror's output and its status (notebook) |
| `DRIVE_DIR/`, `DRIVE_DIR.log` | the mirror on Drive: records, logs, the newest checkpoint, the export |
| `data_cache/` | downloads, token shards, the library (`library/lib_v2`, with its `build_settings.json`), opponent GGUF files, the proxy harness's cache and reports (`proxy/reports/factual/` for the library measurement), the arena |

Code: `scripts/run_24h.py` (stages), `bes/train/supervisor.py` (manifest, preflight, background
tasks, library settings, stage loop), `bes/train/stages.py` (budget and shedding),
`bes/train/benchmark.py` (plans, canary, horizon), `bes/export.py` (export), `bes/eval/arena.py`
(arena), `bes/eval/proxy/` (evaluation harness).
