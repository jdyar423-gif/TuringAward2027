# Track A (key `kernel`): a faster causal attention kernel for one-tile pieces, behind a switch (reviewed)

Patch (reviewed version, applies cleanly to HEAD `ba0db12`):
`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/kernel.patch`
(review worktree `/home/user/turingaward2027/.claude/worktrees/wf_6717e710-a1c-4`; author's worktree
`.../wf_6717e710-a1c-1`; nothing committed). The author's original patch and report are kept as
`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/kernel/review/author_kernel.patch` and
`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/kernel/review/author_kernel.md`.
Author's scratch (tools, LLO dumps, private tokamax install): `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/kernel/`;
the reviewer's checks: `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/kernel/review/`.

## Verdict

* **Time these, in this order** (the script's default list; every kernel at 1024 before any at 2048):
  1. `""`: today's kernels, the reference;
  2. `onetile`: the new kernels, a pure kernel swap, no memory change. Estimate -16% per step at
     4 x 1024, about -15% at 2 x 2048;
  3. `onetile:save`: the same kernels, and the forward kernel's output is kept through the loop's
     rematerialisation. +108 MB of device memory per piece. The author estimates -18% / -19%.
     My estimate is at most 1-2% beyond `onetile`, possibly nothing: the one-tile forward
     kernel it saves is only about 0.4 ms per piece. Against that, the kept output is copied
     into the scan's residual stack and read back, about 0.2 ms of HBM traffic per piece that
     the static count does not see. `remat_keep=kernel`, which also kept the 128-lane
     log-sum-exp (about 3 times the data), was 2% slower on the chip;
  4. `blocks:save`: today's kernels with their forward kept, no new kernel code. The author
     estimates -8%; mine, with the residual copies, is -4% to -5%. This is the fallback if the
     one-tile kernels fail or differ on the chip;
  5. `blocks:kvc=1024`: a tile change only. This is the yardstick for "equal to rounding", not a
     candidate (by estimate it is 4% slower).
* **Command** on the TPU runtime, from the repository root, with the patched files in place and
  the trainer stopped (the chip must be idle):

      python scripts/tpu_kernels.py --config configs/final.py --override model.table_grad=take --override optim.micro_seqs=4 --lengths 1024,2048 --calls 5 --per-round 10 --budget-s 480

  That makes 10 compiles. On the TPU host one compile of this program should take about 15-40 s
  (`tpu_tune` compiled 32 x 1024 programs at about 20 s each), so roughly 3-7 minutes in all.
  The budget stops new runs after 8 minutes, and the last ones dropped are the 2048 runs.
  Read the `BEST @ 1024` and `BEST @ 2048` lines and the `KERNELS {json}` line. `BEST` only
  names a kernel whose loss and gradients match the reference: loss within 1e-4, gradient L2
  difference at most max(5%, 3 x the yardstick's). `mosaic_calls` must be lower for the `save`
  values; if it is not, the flag did not take.
  Shortest pause instead: `--lengths 1024 --kernels ";onetile;onetile:save" --budget-s 240`
  (3 compiles). Then start the trainer with `BES_FAST_KERNEL=onetile:max_t=1024` (or
  `onetile:save,max_t=1024`): both run exactly today's program at 2048 (checked on the lowered
  StableHLO). Time 2048 at a later pause.
* **To adopt the winner**, start the trainer process with `BES_FAST_KERNEL=<value>` in its
  environment and write the value into the run's log. It is not part of the configuration or of
  its hash. Parameters and optimizer state are unchanged, so the checkpoint resumes. The first
  start compiles both shapes again (the compilation cache misses on a new kernel). The switch
  applies to the whole process (every splash-path program in it: training, evaluation, the loss
  of preference optimisation). Decoding uses the XLA path and is not affected.
* **Install on the runtime first: nothing.** Copy the patched files: `bes/model/attention_splash.py`,
  `bes/model/attention_onetile.py`, `bes/model/attention_tokamax.py`, `bes/model/controller.py`,
  `scripts/tpu_kernels.py` (tests and docs are optional). With `BES_FAST_KERNEL` unset the
  trainer's program stays byte-identical. tokamax is not worth timing. If it were wanted anyway:
  `pip install --no-deps tokamax==0.0.17`.

## Review

### What was checked

1. **Default switch = HEAD, byte for byte.** `review/jaxpr_dump.py` traces the HEAD tree (via
   `git archive`) and the patched tree and compares the results with `cmp`. All are identical:
   * `make_jaxpr` of `loss_fn` and of `value_and_grad(loss_fn)` on `configs/smoke.py` (splash,
     remat, bf16, interpret);
   * `value_and_grad` on `configs/final.py` + `table_grad=take` at 4 x 1024 and at 2 x 2048;
   * the whole training step (`make_train_step`, `micro_split=4`, A=2, B=16, T=1024);
   * the StableHLO of `value_and_grad` lowered for TPU v6e at both lengths.
   With `onetile:max_t=1024` the 2048 programs (jaxpr and StableHLO) are those of HEAD. With
   `onetile:save,max_t=1024` the 2048 StableHLO is that of HEAD, after the fix below.
2. **Tests.**
   * The author's 26 tests pass: 21, plus the 5 tokamax tests when the author's private tokamax
     is put on `PYTHONPATH` (read-only use).
   * 21 cases added to `tests/model/test_attention_onetile.py`, all passing:
     * every row with its own segment layout (`_layouts`): one segment; boundaries on and one
       past every strip boundary; a one-token segment at the last position; many short
       segments with ids out of order and one id that comes back; one-token segments at
       positions 1 and T-2;
     * T = 256 (one strip), 512 (two strips), 1024 (the splash tile, 2 heads per step) and
       2048 (twice the tile, 1 head per step), each in f32 and bf16. Output and dq, dk, dv are
       compared with the configured kernels, and in f32 also with float64 and dense gradients;
     * the last position and one-token segments at T=1000 (padded): the output equals the
       token's own value, and a cotangent on the last position reaches only that position's v;
     * bf16 against f32 inputs holding the same values: the one-tile kernels lose no more to
       bf16 than the splash kernels do;
     * the custom VJP inside `controller.run_loops` (scan over a checkpointed step with the
       switch's policy), in f32, for `onetile`, `onetile:save`, `blocks:save` and
       `onetile:hp=1`. Value and gradients match, and the program holds the forward kernel 2
       times (1 with `save`);
     * `max_t` fallback, alone and with `save`: bit for bit today's kernels, with no names and
       no kept kernels;
     * the six compile-only orders of the kernel bodies (`FWD_ORDER`, `BWD_ORDER`,
       `DS_CAST_FIRST`): bit for bit the defaults;
     * switch typos are refused.
   * Unchanged existing tests: `test_splash.py` + `test_attention_major.py` 39 passed;
     `test_model_grads.py` + `test_model_gain.py` (not slow) 25 passed.
3. **Can each candidate run on the chip?** `review/tpu_probe.py` compiles `value_and_grad` for
   `v6e:1x1` with the read-only libtpu 0.0.50 copy. The attention sits inside
   `controller.run_loops`, as the model calls it (scan over loops, `jax.checkpoint` with the
   switch's policy, custom VJP inside), at B x 6 heads x T x 128 in bf16:

   | value | 4 x 1024 | 2 x 2048 | Mosaic calls |
   |---|---|---|---|
   | `""` | 20 s | 13 s | 3 |
   | `onetile` | 17 s | 23 s | 3 |
   | `onetile:save` | 9 s | 11 s | 2 |
   | `blocks:save` | 11 s | 9 s | 2 |
   | `onetile:hp=1` | 8 s | 20 s | 3 |
   | `onetile:bs=128,hp=1` | 17 s | 34 s | 3 |
   | `onetile:hp=2` | 23 s | 49 s | 3 |
   | `xlamap` | 12 s | 20 s | 0 |
   | `tokamax`, `tokamax:diag=1,grid=4` | 11-14 s | 13-14 s | 3 |

   * Every candidate compiles at both shapes. The forward-only one-tile kernel (an evaluation
     pass, no log-sum-exp) compiles too.
   * Memory spaces: VMEM blocks plus one VMEM scratch (dq), no SMEM. Block shapes: full
     trailing dimensions, with T a multiple of 128 enforced. Dtypes: bf16 operands, f32
     compute, int32 segment ids, and the f32 and bf16 transposes all compiled for v6e. It fits
     the default 32 MiB scoped VMEM (nothing sets `vmem_limit_bytes`).
   * I recounted the bundles from new dumps:
     * splash: 4435 fwd / 6664 dkv per slice at 1024;
     * one-tile at 1024: 3853 / 6755 per two-head step;
     * one-tile at 2048: 7080 / 12233 per head.
     These match the author's table (0.47 and 0.58 of today's work).
   * The static count has no DMA or stall model. The chip's profile of the splash kernel alone
     (`docs/tpu_profile/block_profile.v6e.txt`: 2.76 ms per block application at 32 x 1024 with
     1024/1024/512 tiles, including that test's layout transposes) is consistent with the count
     plus memory passes, which give about 2.3 ms without the transposes. So the estimates are
     plausible, not measured.
4. **Smoke.** `taskset -c 2,3 nice -n 10 $PY scripts/tpu_kernels.py --smoke` with the new
   defaults ran 10 of 10 (kernel, length) pairs in 196 s. All losses equal the reference, and
   the `BEST` and `KERNELS` lines are printed (`review/smoke_review.txt`). CPU times in interpret
   mode mean nothing.

### What I changed, and why

* **`save` with `max_t` ran another program than asked** (`bes/model/attention_splash.py`).
  Beyond `max_t`, the switch fell back to splash but kept the residual name, and the policy kept
  every `splash_mha_fwd` kernel. So `onetile:save,max_t=1024` silently became `blocks:save` at
  2048, a program nobody had timed. Now the fallback uses no name, and `remat_policy` keeps only
  the forward kernels of the switch's own kernel (`FORWARD_KERNELS` per kernel). The lengths
  beyond `max_t` lower to today's StableHLO, and there is a test for it.
* **Typos in the switch were silently ignored.** `onetile:svae` timed and ran plain `onetile`.
  `parse_switch` now refuses options and flags a kernel does not take (`SWITCH_KEYS`). The
  timing script checks all values before any work.
* The policy reads the kernel name from either `name` (jax 0.11) or `name_and_src_info`. If a
  jax release renamed it, `save` would quietly stop removing the forward kernel. The chip has
  jax 0.11.2.
* Docstring: "same values: the kernel is deterministic" was not exact. The backward pass reads
  the forward pass's output instead of one recomputed from rematerialised operands, so the
  values are equal to float rounding. The smoke shows `onetile` and `onetile:save` differ at
  the bf16 level.
* **`scripts/tpu_kernels.py`:**
  * Default list cut to the 4 values worth chip time plus the yardstick. Out of the list:
    * `tokamax:diag=1,grid=4`: needs an install, and at 0.80 by count it is behind
      `blocks:save` (0.71);
    * `onetile:hp=1` and `onetile:bs=128,hp=1`: weaker variants of the same kernel, so if
      `onetile` fails they fail too;
    * `blocks:kvc=1024` as a candidate, and `blocks:sep`: slower by count;
    * `xlamap`: cannot win.
  * Order by length (all kernels at 1024 first), so that a budget cut loses 2048 runs and not
    candidates.
  * Timing in rounds of `--per-round` calls sent back to back: device time without a host
    round trip per call; the median over rounds.
  * Default budget 600 s.
  * `BEST @ T` lines with the equality filter, and `NOT EQUAL` lines.
  * `mosaic_calls` per program, to confirm on the chip that `save` took effect.
* Tests: the 21 cases above. `_inputs` takes an optional `seg`.
* `docs/TPU_NOTES.md`: a "Review" paragraph in "Attention kernels for one tile".
* Not changed: the kernels (`bes/model/attention_onetile.py`), `bes/model/attention_tokamax.py`,
  the 5 lines in `bes/model/controller.py`. No configuration field; `bes/config.py` untouched.

### Findings that need no change

* **At 2048 tokens, bf16 dq differs from today's kernels in about 7.5% of elements.** The
  author's "at most 1% of elements differ" holds only up to 1024.
  * The difference is below 1.6e-4 of the scale. The one-tile dq is the closer one to a float32
    reference: RMS 3.629e-4 against 3.640e-4.
  * Cause: today's fused splash backward writes one bf16 dq per 1024-key tile
    (`dq_unreduced`, operand dtype) and sums them in XLA. That is one rounding more than the
    one-tile float32 sum.
  * The splash kernels with a single 2048 tile (`blocks:q=2048,kv=2048`, the "T=2048, tile
    2048" setting of the earlier chip tuning) differ from today's kernels in exactly the same
    way.
  * Out, dk and dv meet the 1% criterion at 2048.
  * Consequence for the chip: at 2048 the gradient differences printed against the reference
    are larger than at 1024. That is expected and is not a defect.
* tokamax, `xlamap` and the `hp`/`bs` variants are kept as switch values: they are tested and
  cost nothing unless selected. They are just not worth chip time.
* Not done: a whole-program compile at 2 x 2048 (330-410 s and several GB here, while other jobs
  were being killed for memory). The kernels were compiled at 2 x 2048 inside the loop scan. The
  rest of the program is today's, which the chip already runs at 2048.

### Commands run (from the review worktree; `PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python`, no installs)

* `taskset -c 2,3 nice -n 10 $PY .../review/jaxpr_dump.py <tree> <out> [--tpu-lower] [--switch=...]`, then `cmp`
* `taskset -c 2,3 nice -n 10 $PY -m pytest tests/model/test_attention_onetile.py -o addopts="-p no:cacheprovider" -q -x`
  in three parts (a whole run no longer fits 9 minutes on the loaded host): `-m "not slow"`
  author's tests 20 passed and 5 skipped; review tests 19 passed; `-m slow` 3 passed (model,
  T=2048 f32 and bf16); `-k tokamax` with `PYTHONPATH=.../wf/kernel/tkx_site` 5 passed
* `... -m pytest tests/model/test_splash.py tests/model/test_attention_major.py` 39 passed;
  `... tests/model/test_model_grads.py tests/model/test_model_gain.py -m "not slow"` 25 passed
* `taskset -c 2,3 nice -n 10 $PY .../review/tpu_probe.py <worktree> <switch> B T [loops]`
  (compile-only for v6e; `KDUMP=dir` for bundle dumps, `FWD_ONLY=1` for the forward-only kernel)
* `taskset -c 2,3 nice -n 10 $PY .../review/diag2048.py 2048 "onetile;onetile:hp=2;blocks:kvc=1024;blocks:q=2048,kv=2048,kvc=512"`
* `taskset -c 2,3 nice -n 10 $PY scripts/tpu_kernels.py --smoke`

---

## The author's report (updated where the review changed it)

### Result

* A switch `bes.model.attention_splash.FAST_KERNEL` selects the kernel. It is a module-level
  value, read at trace time, with its initial value from `BES_FAST_KERNEL`; the default `""`
  is today's path and the program is unchanged. Values:
  * `onetile[:bs=,hp=,max_t=]`: new Pallas kernels, `bes/model/attention_onetile.py`;
  * `blocks:...`: the existing splash kernels with other tiles, backward kernels or layouts;
  * `tokamax[:...]`: tokamax 0.0.17's splash kernel, `bes/model/attention_tokamax.py`;
  * `xlamap`: plain XLA, slice by slice;
  * the flag `save` on any of them: keep the forward kernel's output and log-sum-exp through
    the loop's rematerialisation, so the forward kernel runs once per block application
    instead of twice. With `onetile`, this applies only to the lengths the one-tile kernels
    take.
  Unknown options and flags are refused. No configuration field was added.
* All candidates compute today's output and gradients to float rounding (tests below).
* **Ranking by estimate**: executed static VLIW bundles from the chip's own compiler, per
  (row, head) slice, for 2 forward + 1 backward kernel (one block application under remat),
  relative to today's kernels:

| rank | `FAST_KERNEL` | T=1024 | T=2048 | est. saving per piece (4x1024) | notes |
|---|---|---|---|---|---|
| 1 | `onetile:save` | **0.34** | **0.42** | ~3.0 ms of ~16.6 ms (-18%); review: ~-17%, about 1% beyond `onetile` once the residual copies are counted | + ~7 MB residuals per block application (108 MB per piece) |
| 2 | `onetile` | **0.47** | **0.58** | ~2.6 ms (-16%) | pure kernel swap, same memory |
| 3 | `blocks:save` | 0.71 | 0.72 | ~1.3 ms (-8%); review: -4% to -5% with the residual copies | no new kernel code; residuals as rank 1 |
| 4 | `tokamax:diag=1,grid=4` | 0.80 | - | ~0.9 ms (-5%) | needs `pip install --no-deps tokamax==0.0.17`; not in the timing list |
| 5 | `onetile:hp=1` / `onetile:bs=128,hp=1` | 0.50 / 0.54 | | | variants of rank 2; not in the timing list |
| 6 | `blocks:klayout=seq` | 0.98 | | ~0 | plus an XLA transpose of k |
| - | `blocks:kvc=1024` / `blocks:kvc=256` / `blocks:sep` | 1.04 / 1.08 / 1.19 | | slower | `blocks:kvc=1024` is the timing yardstick |
| - | `tokamax` (1024/1024/1024, no diagonal skip) | 1.02 | | slower | |
| - | `xlamap` | not countable (XLA loop) | | expected much slower | |

* How the per-piece estimates are built:
  * bundle time at 1.75 GHz, plus the HBM passes around the kernel. Today's path writes the
    log-sum-exp 128 lanes wide and slices it, and writes the running max, sum and output as
    kernel outputs: about 70 us per block application. The one-tile path writes a lane-dense
    log-sum-exp and no statistics.
  * Calibration: for today's kernels this gives 3.4 ms of kernel + 1.1 ms of glue = 4.5 ms per
    piece.
  * Step at 64 x 4 x 1024: 1063 ms -> ~895 ms (`onetile`) or ~870 ms (`onetile:save`), i.e.
    ~290-300k tok/s instead of 246k, if nothing else shifts.
  * At 2 x 2048, by the same method: 1218 ms -> roughly 1030 ms (`onetile`).
  **Not timed; the chip decides.**

### What was built (files)

* `bes/model/attention_onetile.py` (new), the one-tile kernels.
  * Forward:
    * one grid step per (row, 2 heads) at T<=1024, 1 head at 2048; q, k and v of the slice in
      VMEM;
    * strips of 256 query rows cover only the causal triangle (62.5% of the score elements at
      T=1024), split into the diagonal block (causal + segment mask) and the keys before it
      (segment compare only);
    * the strip's softmax in one pass; `p @ v` with `p` in f32 and `v` cast to f32 as in the
      splash kernel; output `o * (1/l)`;
    * the log-sum-exp leaves as a lane-dense row `[B, H, 8, T]`.
  * Backward:
    * key-major strips;
    * `p^T = exp(s^T - lse)`, `dv = p^T.bf16 @ do`, `dp^T = v @ do^T`,
      `ds^T = (dp^T - di) p^T`, `dk = ds^T.bf16 @ q`;
    * `dq += ds.bf16 @ k`, accumulated in an f32 VMEM scratch;
    * `di` as in splash (an XLA einsum).
  * custom_vjp; residuals (q, k, v, seg, o, lse); `residual_name` names o and lse for `save`.
  * `attend_xla_slices` (`xlamap`).
* `bes/model/attention_tokamax.py` (new): loads tokamax 0.0.17's four kernel files from the
  installed wheel without running `tokamax/__init__`. Optional.
* `bes/model/attention_splash.py`: `FAST_KERNEL`, `parse_switch` (+ `SWITCH_KEYS`),
  `remat_policy` (+ `FORWARD_KERNELS` per kernel), `switched_block_sizes`,
  `ONETILE_MAX_T = 2048`, `make_kernel(..., residual_name)`, and the dispatch in `attend_major`.
* `bes/model/controller.py` (5 lines): when remat is on and `remat_keep == ""`, the policy comes
  from `attention_splash.remat_policy()`. That is None unless the switch carries `save`, so the
  program is unchanged.
* `scripts/tpu_kernels.py` (new): the timing script.
* `tests/model/test_attention_onetile.py` (new): 47 tests (3 slow; 5 need tokamax).
* `docs/TPU_NOTES.md`: section "Attention kernels for one tile" (method, bundle table, review).

### Method: static schedules from the chip's compiler (no chip needed)

* libtpu 0.0.50, the chip's version (a read-only copy in the scratchpad), compiles Pallas
  kernels for `v6e:1x1` on this CPU host. `LIBTPU_INIT_ARGS=--xla_jf_dump_to=DIR` writes each
  Mosaic kernel's final VLIW bundles and a per-bundle utilisation table.
* The kernels are straight-line code per grid step, so executed bundles per step give a cycle
  count without DMA or MXU stalls.
* The splash forward is bound by vector stores and VALU, the backward by MXU issue slots. MXU
  work scales with MACs whatever the tiles, so only skipping the upper triangle cuts it.
* Whole program on one piece, author's compile-only check at 4 x 1024 (`value_and_grad` of
  `loss_fn`):

  | setting | compile | peak | Mosaic calls |
  |---|---|---|---|
  | today | 334 s | 0.62 GiB | 6 |
  | `onetile` | 411 s | 0.61 GiB | 6 |
  | `onetile:save` | 375 s | 0.70 GiB | 4 |

### Equality

`tests/model/test_attention_onetile.py` runs in interpret mode on CPU against today's kernels at
the same shapes:

* **float32:** within 1e-5 of the scale.
* **bfloat16:** the largest difference is at most 2^-8 of the scale, at most 1% of elements
  differ, and the RMS error against an f32 reference is within 10% of today's kernel's. This
  holds for every array up to T = 1024. At T = 2048, dq differs in about 7.5% of elements (see
  the review): it is checked against the splash kernels with a single tile, and its RMS error
  must be no worse than today's.
* **Yardstick:** `blocks:kvc=128` gives the same sizes of difference.
* **Whole model:** the tiny model (remat, 2 loops) gives the same loss and parameter gradients
  with `onetile`, `onetile:save`, `blocks:save` and `xlamap`. Model-level bf16 (`--smoke
  --lengths 256`), largest relative gradient difference, max / L2:
  * `blocks:kvc=128`: 1.3e-2 / 1.1e-2;
  * `onetile`: 7.6e-3 / 5.1e-3;
  * `onetile:save`: 3.4e-4 / 1.5e-5.

### Dropped (author), unchanged

* bf16 `p` in the forward: changes values beyond rounding.
* Other block sizes of today's kernel: they cost more.
* `residual_checkpoint_name` alone: the forward kernel still reruns, because JAX's DCE never
  removes a `pallas_call`.
* tokamax without the diagonal skip: costs more.
* Several onetile body orders: they compiled worse; they are kept as module constants and are
  bit-identical, per the test.

### tokamax

* Apache-2.0. 0.0.17 is the last release for jax 0.11.
* A full install pulls in 69 packages. The loader needs only `pip install --no-deps
  tokamax==0.0.17` (one 2.7 MB wheel) and otherwise fails as `FAILED ImportError` at no cost.
* Not recommended for the chip.

### What could go wrong on the chip

* The chip runs jax 0.11.2, this host 0.11.1, and the Pallas -> Mosaic lowering is on the
  Python side: a small compile risk. It would show as a `FAILED` row.
* No kernel has run on a TPU yet. On the chip, the numbers are checked only by the script's loss
  and gradient comparison against the reference, which `BEST` requires.
* The estimate ignores DMA waits and MXU stalls (about 1-3 us per grid step of prefetch against
  2-4 us of compute). `save` adds residual copies through the loop scan that the static count
  does not see.
* VMEM: 2048 compiles under the default 32 MiB, even with 2 heads per step. If a later libtpu
  refuses it, `onetile:max_t=1024` is the fallback (today's program at 2048).
