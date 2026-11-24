# Integration of the speed switches (24 Nov 2026)

Worktree `/home/user/turingaward2027/.claude/worktrees/wf_3d459e66-645-9` (HEAD `ba0db12`).
Patch: `wf/combined.patch`, made with `git add -N` on the new files and `git diff`. It holds
`ledger.patch`, then `speed_attention.patch`, `speed_mlp_block.patch`, `speed_loops.patch`
(empty) and `speed_io.patch`, in that order. All of them applied cleanly with
`git apply --3way` and none of them conflicted. On top of those, the patch adds this
integration's files:

- `tests/model/test_fast_switches.py`;
- `scripts/tpu_candidates.py`;
- a pin in `tests/model/test_model_loss.py` (section 6).

The constraints held:

- nothing is committed or pushed, and no branch changed;
- `bes/config.py` and `configs/` are untouched, and no configuration field was added;
- the parameter tree, the optimiser state and the configuration hash are the same whichever
  switches are on, because the switches are module-level Python constants read at trace time,
  not configuration.

## 0. Read first: the main checkout has moved past this base

This worktree is at `ba0db12`, the commit the four patches were made against. During this
integration, the main checkout `/home/user/turingaward2027` moved to **`6bf9b6e`** (branch
`claude/modest-carson-yi2d9g`, 07:37 UTC, "Speed candidates behind switches (all off)"). That
commit already holds the same three switches:

- `bes/model/head.py`, `bes/model/layers.py`, `tests/model/test_io_speed.py`,
  `tests/model/test_mlp_block_speed.py` and `tests/model/test_splash_io.py` are byte-identical
  to the integrated ones here;
- `bes/model/attention_splash.py` has the same FAST_SPLASH_IO code, plus an attention kernel
  switch `FAST_KERNEL`, whose default is `""` or the value of `BES_FAST_KERNEL`.
  FAST_SPLASH_IO applies only when `FAST_KERNEL == ""` and the `save` flag is not set;
- `bes/model/fast.py` and `scripts/run_24h.py:fast_switches_run` let a run take its switches
  from `<run>/fast`, with names `relu2`, `ce`, `io` and `kernel`;
- `scripts/tpu_kernels.py` is its own chip measurement.

I checked the traced trainer program with the jaxpr tool of section 3:
- with every switch off, `6bf9b6e` traces byte for byte the program of `ba0db12`, at both
  lengths, unsplit and in 4 pieces;
- with each switch on alone, it traces byte for byte the program of this worktree with that
  switch on (16 x 1024).

So the review and the tests below hold for `6bf9b6e` as they stand.

Two patches follow from this:

| patch | base | content |
|---|---|---|
| `wf/combined.patch` | `ba0db12`, as asked | the whole integration. It does **not** apply on `6bf9b6e`, which already has the switches |
| `wf/combined_on_6bf9b6e.patch` | `6bf9b6e` | only what `6bf9b6e` lacks: `scripts/glue_ledger.py`, `scripts/tpu_candidates.py`, `tests/model/test_fast_switches.py` and the pin in `tests/model/test_model_loss.py`. Checked on a pristine export of `6bf9b6e` (`git archive`): the patch applies cleanly; `test_fast_switches.py` and the pinned test give 6 passed (119 s); `tpu_candidates.py --smoke --lengths 128 --min-gain 0.5`, which forces the minus-one variants, ran its 8 variants (baseline, all, 3 singles, 3 minus-one) in 112 s, exit 0 |

## 1. Switches kept (all three; default `False`)

Expected effect, in ms per trainer micro-batch:
- **pieces** means the live run's 4 pieces per micro-batch (`micro_split` 4). That is 4 x 1024
  pieces of a 16 x 1024 micro-batch (66.5 ms on the chip, CHANGES 19.5) and 2 x 2048 pieces of an
  8 x 2048 one (76.1 ms).
- **unsplit** means one `value_and_grad` of the whole micro-batch.

These numbers are the engineers' estimates (compile-only TPU v6e, libtpu 0.0.50). They are not
chip timings.

| switch | what it does | exactness on CPU | ledger, 16 x 1024 (8 x 2048) | compiled for v6e (engineers) | expected effect |
|---|---|---|---|---|---|
| `bes.model.attention_splash.FAST_SPLASH_IO` | `attend_major` calls the library's own splash kernel bodies (`flash_attention_kernel`, `_flash_attention_dkv_kernel`, unchanged) through pallas_call set-ups. In those set-ups the m/l/o and dq/dk/dv accumulators are VMEM scratch instead of HBM outputs, and the log-sum-exp is written once as `f32[H,8,T]` (the backward kernel's layout) instead of `f32[H,T,128]` that is then sliced and re-broadcast. Separate backward kernels fall back to the library's backward | bit for bit | glue -57 MB (-57); Pallas kernel outputs 4,832 -> 2,114 MB (4,631 -> 2,517) | HBM traffic 70.0 -> 66.0 GB (-5.7%) at 16 x 1024; 71.9 -> 68.7 GB (-4.5%) at 8 x 2048; peak memory unchanged | about -1.5%: -1.0 ms at T=1024 (range -0.4 to -2.5) and about -0.9 ms at T=2048. The kernels' output DMA may already be hidden, so the size is uncertain |
| `bes.model.layers.FAST_RELU2` | `mlp()` uses `layers.relu2`, a `custom_vjp` with the same forward. Its residual is the stored bf16 pre-activation, and its backward is `ct * 2 relu(h)`. The `h > 0` select of autodiff, which only fixes the sign of an exact zero, goes away, and with it the u16 bit mask that XLA packs in the rematerialised `W_up` product and unpacks in the backward one | bit for bit (`-0.0 == 0.0`) | glue +-0 (the mask exists only in the optimised TPU HLO) | estimated cycles: 4 x 1024 piece 31.85 -> 31.05 M (-2.5%); 2 x 2048 piece 35.12 -> 33.30 M (-5.2%); 16 x 1024 140.73 -> 140.06 M (-0.5%) | pieces: -1.7 ms (T=1024), -4.0 ms (T=2048); unsplit 16 x 1024: -0.35 ms |
| `bes.model.head.FAST_CE` | `chunked_ce` goes through `_fast_ce`, a `custom_vjp`. Its log-sum-exp has no row maximum (the softcap bounds `abs(z) <= 15`), which XLA fuses into the logit product's epilogue. Its backward keeps the f32 logits and row sums, so no logit product or vocabulary reduction is repeated. Guards: the reference path runs if the softcap is not in (0, 60] or if `4nV > 2 GiB` | float32 rounding (not bit for bit) | glue -1,846 MB (-1,846); materialised -2,441 (-2,441); products -137 GFLOP | cycles: 4 x 1024 piece 31.846 -> 31.514 M (-1.0%); 2 x 2048 -0.95%; 16 x 1024 140.73 -> 137.21 M (-2.5%); peak memory unchanged | pieces: -0.7 ms at both lengths; unsplit 16 x 1024: -1.8 ms |

All three together, if the effects add up:

| | estimated change | step at 16 micro-batches |
|---|---|---|
| T=1024, pieces | about -3.4 ms of 66.5 ms per micro-batch (-5%) | about 246.5k -> 259k tok/s |
| T=2048, pieces | about -5.6 ms of 76.1 ms (-7%) | about 215k -> 232k tok/s |

Only the chip can say whether these estimates hold (section 5).

Ledger totals at the trainer's shapes. Each cell is the glue pass, then the materialised MB in
brackets. The kernel column is the Pallas kernel outputs in MB, and the last column is the
matrix-product GFLOP:

| variant | 16 x 1024 | 8 x 2048 | 4 x 1024 piece | 2 x 2048 piece | kernel out 16x1024 / 8x2048 | GFLOP |
|---|---|---|---|---|---|---|
| all off (= `wf/ledger_baseline_*.json`, +0) | 87,465 (49,104) | 89,086 (49,517) | 22,461 (12,458) | 22,873 (12,569) | 4,832 / 4,631 | 17,939.6 |
| FAST_SPLASH_IO | 87,408 (49,054) | 89,029 (49,467) | 22,447 (12,446) | 22,859 (12,556) | 2,114 / 2,517 | 17,939.6 |
| FAST_RELU2 | 87,465 (49,104) | 89,086 (49,517) | 22,461 (12,458) | 22,873 (12,569) | 4,832 / 4,631 | 17,939.6 |
| FAST_CE | 85,620 (46,663) | 87,240 (47,076) | 21,920 (11,774) | 22,332 (11,885) | 4,832 / 4,631 | 17,802.2 |
| all on | 85,563 (46,613) | 87,183 (47,025) | 21,906 (11,762) | 22,318 (11,872) | 2,114 / 2,517 | 17,802.2 |

The ledger is blind to two of the three gains:
- FAST_RELU2's packed mask exists only in the optimised TPU program;
- most of FAST_SPLASH_IO's saving sits in the kernel outputs, which the ledger books outside the
  glue total.

It also overstates FAST_CE at one-chunk pieces. There, XLA already removes the reference's
recomputation; only the second pass over the logits goes away (-1.0% compiled).

Files: `wf/integration_work/led_{off,splash,relu2,ce,all}_{1024,2048}.{json,txt}` and
`ledp_*` (the pieces).

## 2. Removed, or not kept

- **No delivered switch was removed.** Every one survived review (section 3).
- **`speed_loops`** delivered no switch. Its patch is empty.
- **The candidates the engineers dropped themselves** are not in the patch. Their reports give
  the reasons: FAST_ROPE_ROLL, FAST_ROPE_FLIP, FAST_QK_PAIR, FAST_PROJ_VJP, FAST_MM_LO, FAST_TAKE,
  FAST_CARRY_NEXT, and the sub-1% items.
- **`FAST_SPLASH_IO` with `model.remat_keep=kernel`** is not adoptable. `remat_keep` is a
  configuration field, so it changes the configuration hash (`config_hash` hashes the whole
  `Cfg`), and the live run cannot continue from its checkpoints with it. `tpu_candidates.py`
  times it only when asked (`--info`), last, as information only, and only if the budget is
  left over.

## 3. Review (sceptic's checks)

**The switches off give HEAD's program.**
- The jaxpr text of the trainer program is byte-identical to HEAD's (`git archive HEAD`) at
  16 x 1024 and at 8 x 2048, both unsplit and with `micro_split` 4. That program is
  `train.accumulate` over one micro-batch, `configs/final.py` with the live overrides, traced
  abstractly at the real shapes, with object addresses and the checkout path normalised. Tool:
  `wf/integration_work/jaxpr_dump.py`.
- The ledger totals equal `wf/ledger_baseline_1024.txt` and `_2048` exactly.
- Each switch alone changes the jaxpr.

**FAST_SPLASH_IO.** I read the set-ups against `splash_attention_kernel.py` of jax 0.11.1:
- the ref order of the kernel bodies, the index maps and the grid are the same;
- `grid_width` is the same, from the unshrunk fused-dkv mask info;
- the `di` einsum is the same, and so is the dq partial-sum.

The log-sum-exp tile is constant across its 128 lanes (m and l are lane-broadcast), so the
in-kernel transposed `[:8]` rows hold exactly the values the library's slice-and-broadcast gives.

The module the fork reads private names from is identical, comments stripped, in jax 0.11.1
(here) and 0.11.2 (the chip; `scratchpad/dl_jax/x`). So are `splash_attention_mask*.py`.

Tests: 6 kernel cases. They cover the dq accumulator with and without, a padded length, several
query tiles, bf16 and f32, and separate dq/dkv kernels. On top of those:
- the kernels' output counts;
- the model, bit for bit, in bf16 and f32, with every gradient leaf;
- `remat_keep=kernel`.

A gap in the engineer's own model test, closed by `test_fast_switches.py`: it used 128 tiles
only, the smoke defaults `use_conf=True` and `clock_planes=3`, and two loops.

**FAST_RELU2.** Autodiff of `square(relu(f32 h))` is `select(h > 0, ct * (2 r), 0)`. The custom
VJP computes `ct * (2 relu(h))` in the same order of operations and casts the result to the
activation dtype as the reference's convert transpose does.

Where `h <= 0`, both give a zero; only its sign can differ. A NaN can appear only where the
incoming cotangent is already non-finite, and such a step is skipped by the trainer.

The tests compare everything at tolerance zero: the values and the VJP with +0, -0 and negative
inputs, under jit and a checkpointed scan, `mlp()` with and without a gain, and the whole model
with live features in bf16 and in f32, including `gain_ffn` with PhaseGain and no remat.

I also checked the engineer's optimised TPU programs for the removed mask:
- the reference packs it with `shift-left` into u16 arrays at 16 x 1024 (32 of them) and into
  u8 at the 4 x 1024 and 2 x 2048 pieces;
- the FAST_RELU2 programs contain no `shift-left` and no u16 array.

**FAST_CE.** I checked the forward against `layers.softcap`: `tanh(z/cap)*cap` gives the same
logits bit for bit.

- The unshifted sum is safe for `cap <= 60` and V up to 2^24.
- Padded rows get zero cotangents.
- The backward is the closed form of autodiff's: `-g` at the target, plus `(g/s) exp(zc)`,
  times `(1-t)(1+t)`.
- `d e_out` and `d hp` are the same dot_generals as the transposes. With several chunks the
  reference rounds its bf16 running sum after every chunk, while the fast path sums in f32 and
  rounds once. That is the more accurate of the two, and the only bf16-level difference.
- The guards fall back to the reference path. A test covers it.

Tests: chunked_ce alone in f64 (1e-13), f32 and bf16, the softcap extremes, and the model with
live features (all 59 leaves). Not bit for bit, but equal to float rounding.

For scale, cutting the micro-batch into pieces, which the run accepted on 24 Nov
(`micro_split`), changes the smoke model's gradient norm by 7.5e-5 to 1.2e-4. Small leaves
(`lanes/mix`, `lanes/alpha`) move by 2e-2 of their largest entry. FAST_CE on the same model
changes the norm by 1.3e-9, and only `head/out`, by 3.3e-4 (`wf/integration_work/exp_split_scale.py`).

Side effects to know about:
- the preference optimisation's `token_ce` takes the fast path too when the switch is on. Its
  residual is bounded by `FAST_CE_MAX_RESIDUAL`, 2 GiB;
- `test_model_loss.py::test_chunked_ce_memory_stays_per_chunk` describes the reference path, so
  it is now pinned to the switch off (section 6).

## 4. The switches together (`tests/model/test_fast_switches.py`, new)

The test runs `value_and_grad(loss_fn)` (loss, every metric, all 59 gradient leaves) on a tiny
model with the live run's features:
- splash attention (interpret mode), bf16, remat, sink;
- the tape in and out, registers and value embeddings;
- clocks and confidence off, `table_grad=take`, 3 loops.

It uses tiles 128/256/128 at T=200, padded to 256, with packed segments. That covers what the
shipped 1024/1024/512 tiles do at T=1024: the fused backward's dq accumulator, two query blocks
and padding. The T=2048 case, a causally skipped block together with the accumulator, is the
kernel test below.

It checks that every switch reached the program:
- the `splash_mha_fwd_io` kernels;
- the custom VJP of relu2 inside the remat;
- one `reduce_max` fewer for FAST_CE.

Then, for three cases (bf16 with 1 loss chunk as in every training piece, bf16 with 2 chunks
as in an unsplit micro-batch, and f32):

| comparison | requirement | measured |
|---|---|---|
| FAST_SPLASH_IO + FAST_RELU2 against all off | **bit for bit** | bit for bit in all three cases |
| all three against FAST_CE alone | **bit for bit** | bit for bit in all three cases |
| all three against all off | loss and metrics to 1e-6; bf16: every leaf to 2^-7 of its largest entry, and `head/out` to 2^-7 per chunk; f32: 5e-6 | bf16, 1 chunk: loss and all 59 leaves bit for bit, metrics 1.4e-7. bf16, 2 chunks: `head/out` 2.4e-3, the other 58 bit for bit. f32: loss bit for bit, worst leaf 6.2e-7 |

The model test takes 3 x 55 s on idle cores. It is marked `slow`, and took 6.5 min under the
container's load.

The same file has a kernel test for the one combination the engineer's kernel cases did not
cover together: the shipped T=2048 tiles. Those are 1024/1024/512, with two query and two key
blocks, a causally skipped block and the dq accumulator, run here at a quarter size (128/256/128
at a padded length of 512). It checks the output, dq, dk and dv in bf16 and in f32, all bit for
bit, in 32 s.

## 5. Commands for the orchestrator

**Tests**, from the root of the checkout with the patch applied (CPU, cores 2-3, each well
under 9 minutes):

    PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
    taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_fast_switches.py -o addopts="-p no:cacheprovider" -q -x      # 5 tests, 2-7 min
    taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_splash_io.py -o addopts="-p no:cacheprovider" -q -x          # 10 tests, 1.6 min
    taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_mlp_block_speed.py -o addopts="-p no:cacheprovider" -q -x    # 7 tests, 2 min
    taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_io_speed.py -o addopts="-p no:cacheprovider" -q -x           # 14 tests, 2.5 min
    taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_model_loss.py -o addopts="-p no:cacheprovider" -q -x         # the pinned test
    taskset -c 2,3 nice -n 10 $PY scripts/tpu_candidates.py --smoke                                                       # CPU smoke, < 6 min

Results here, with every switch off (the defaults), CPU, under the container's load:

| test files | result |
|---|---|
| `test_fast_switches.py` (new) | 5 passed (124 s; the three model cases 46, 40 and 27 s, the two kernel cases 5 s each) |
| `test_splash_io.py` | 10 passed (98 s) |
| `test_mlp_block_speed.py` | 7 passed (121 s) |
| `test_io_speed.py` | 14 passed (153 s) |
| `test_splash.py`, `test_attention_major.py` | 39 passed (239 s) |
| `test_model_loss.py` (with the pin), `test_model_train_v9.py` | 9 passed (184 s) |
| `test_model_forward.py`, `test_model_grads.py`, `test_model_reference.py` (and `test_model_loss.py` before the pin) | 24 passed (246 s) |
| `test_model_gain.py` | 20 passed (421 s) |
| `test_model_gate.py`, `test_model_rope.py` | 15 passed (91 s) |
| `test_model_page.py`, `test_model_selective.py` | 27 passed (211 s) |
| `test_model_sink.py`, `test_model_sorted.py` | 15 passed (161 s) |
| `test_model_tables.py`, `test_predictive.py`, `test_readset.py`, `test_roles_budget.py`, `test_config_budgets.py`, `test_config_v9.py` | 104 passed, 1 skipped (`test_config_budgets.py`: "the built library is not present") (235 s) |
| `test_generate.py` | 37 passed, plus the 7 slow ones in a second run (111 s) |

That is every file under `tests/model/`, with no failure.

**On the chip** (`scripts/tpu_candidates.py`), in one pause of the live run:
- it needs the idle TPU and the patched checkout on the runtime;
- it writes nothing into the run;
- about 200 s of measurement is the default budget, start-up included. It never starts a
  variant it cannot finish within that budget.

On 23 Nov, `tpu_tune.py` compiled comparable programs on the chip in 10-13 s each, so 200 s
holds the ten core variants (both lengths) and a few minus-one ones.

The pause must also fit stopping (about 1 min) and restarting (about 1.5 min) the run, CHANGES
19.5. If all of that must fit in 4 minutes, pass `--budget-s 90`. The order still gives the
T=1024 baseline, all on and the singles first.

    python scripts/tpu_candidates.py --config configs/final.py \
        --override model.table_grad=take --override optim.micro_seqs=16 \
        --micro-split 4 --budget-s 200 --out /content/bes_cli/candidates.json

Set `--micro-split` to the live run's value, the content of `/content/runs/main16/micro_split`
or the `micro_split` key of its manifest; 4 is what CHANGES 19.5 reports. It may be a mapping
such as `1024:4,2048:8`.

What is timed is what the trainer computes for one micro-batch, `train.accumulate` over
`[1, B, T]`, cut on the device into pieces as the run cuts them, at the trainer's two shapes:
- 16 x 1024;
- 8 x 2048, which is the trainer's micro-batch at T=2048 (`schedule.micro_shape`; the brief's
  "16 x 2048" would be a shape the run never computes; `--rows 2048:16` gives it).

The batches are synthetic (`tpu_tune.synth_batch`) and the parameters freshly initialised.

**Order** (most promising first, so the budget cuts the least promising):
1. per length: the baseline (all off), all on, then each switch alone, by expected gain at that
   piece size (RELU2, SPLASH_IO, CE with pieces; CE first unsplit);
2. then all on without each switch that was less than 1% faster alone (or failed alone);
3. last, only with `--info`, `SPLASH_IO` + `remat_keep=kernel`, information only.

**Per variant**: the first program runs 5 discarded calls, every program 2 more, then the
median of 5 timed calls. It prints:
- compile s, ms per micro-batch, gain against the baseline and tok/s;
- the loss and its difference from the baseline. A difference above 1e-5 relative, or a
  gradient-norm difference above 1e-4 relative, marks the variant MISMATCH, and it is never
  recommended;
- the worst leaf's difference relative to that leaf's largest entry, for information.

Expect 0 for the exact switches, and up to about 1e-2 on small leaves with FAST_CE. Leaves with
cancelling sums, `sink_logit` and `lanes/*`, show that on CPU too, and `micro_split` itself does
the same.

**Last line**: `CANDIDATES {json}`. It has `variants` (each with `ms`, `tokens_per_s`,
`gain_pct`, `loss_diff`, `grad_norm_rel_diff`, `worst_leaf_rel_diff` and `compile_s`),
`skipped`, `recommend` (`{"1024": [...], "2048": [...], "overall": [...]}`) and `gain_pct`.

The **recommended set** is chosen as follows:
- among the switches that are at least 1% faster alone (`--min-gain`), take the fastest
  measured combination: a single switch, all on, or all on without the others;
- if no combination beats the baseline, recommend nothing;
- `overall` weights the two lengths by `--long-share` (default 0.6). By my estimate the run
  reaches progress 0.67 around 12:30 UTC and trains at T=2048 from then until 18:34, which is
  about 0.55-0.6 of what is left of its training as of 08:00.

The CPU smoke test (`--smoke`) uses `configs/smoke.py` with the live features: splash in
interpret mode, bf16, remat, `take`, clocks and confidence off, and pieces of 4 x 128 and
2 x 256. CPU timings mean nothing. Two runs here:

- an earlier version with `--budget-s 1200` and the information variant ran every variant and
  code path in 470 s: baseline, all, singles, the five minus-one variants, the information
  variant, the recommendation and the JSON;
- the final version with the defaults (300 s budget) ran the ten core variants (baseline, all
  and the three singles at both lengths) in 145 s with nothing skipped, printed the per-length
  and overall recommendation, and printed a CANDIDATES line that parses as JSON. All singles
  came out "faster" on the loaded CPU, so no minus-one variant ran; the minus-one path of the
  final code was run separately on the `6bf9b6e` export (section 0).

## 6. Making an accepted set permanent

**On `6bf9b6e`**, the main checkout's commit, no code changes. Stop the run and write the
accepted subset into `/content/runs/main16/fast`:

- write `relu2=1 ce=1 io=1` for all three;
- `io` is FAST_SPLASH_IO, `relu2` is FAST_RELU2 and `ce` is FAST_CE;
- leave `kernel` out, or set it to `""`.

Then restart the run. `fast_switches_run` validates the text, copies it into the manifest (so a
snapshot carries it to the next machine), and applies it before any stage traces a program.

To make it permanent in the code instead, flip the defaults as below.

**On `ba0db12`**, the base of `combined.patch`, which has no `fast.py`: for every accepted
switch, flip its default from `False` to `True` in its module, one line each:

| switch | file and line |
|---|---|
| `FAST_SPLASH_IO` | `bes/model/attention_splash.py`: `FAST_SPLASH_IO = False` -> `True` |
| `FAST_RELU2` | `bes/model/layers.py`: `FAST_RELU2 = False` -> `True` |
| `FAST_CE` | `bes/model/head.py`: `FAST_CE = False` -> `True` |

Both ways, what does not change:
- no configuration field, so `bes/config.py` and the configuration hash stay the same;
- no checkpoint conversion: the parameter tree and the optimiser state are unchanged.

What to expect:
- the live run takes the change at its next start, because the programs are traced when the
  trainer compiles them. Setting a module attribute in a process that has already traced has
  no effect. The run continues from its checkpoint;
- the persistent compilation cache misses once;
- the profile kernel names become `splash_mha_fwd_io` and `splash_mha_dkv_io`.

The tests keep working after a flip:
- the switch tests set both values explicitly;
- `tpu_candidates.py` turns every switch off for its baseline;
- `test_chunked_ce_memory_stays_per_chunk` is now pinned to the reference path.

I checked a flip here with all three defaults flipped to `True`, using a pytest plugin:

    PYTHONPATH=wf/integration_work/plugins python -m pytest -p fast_on_plugin <files>

`FAST_ON=` limits the plugin to some of the switches. Results:

| files | result |
|---|---|
| `test_model_loss.py` (with the pin), `test_model_forward.py`, `test_model_grads.py` | 21 passed |
| `test_splash.py`, `test_attention_major.py` | 39 passed |

Without the pin, `test_chunked_ce_memory_stays_per_chunk` fails with FAST_CE on. Its
temporaries are 202 MB against the 67 MB of all the logits, which is by design.

A jax upgrade must re-check the private splash names that `FAST_SPLASH_IO` uses:
`flash_attention_kernel`, `_flash_attention_dkv_kernel`, `_next_nonzero`, `_div`,
`_splash_attention_bwd`, `NUM_LANES` and `NUM_SUBLANES`. `tests/model/test_splash_io.py` fails
if they change.
