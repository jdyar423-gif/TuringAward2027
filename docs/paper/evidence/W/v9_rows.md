# v9_rows: the token-neuron rows around the contractions (switch `rows`)

## Decision (start with this)

- **Switch:** `rows` = `bes.model.model.FAST_ROWS`. It is off by default and registered in
  `bes/model/fast.py` `FLAGS`.
  - For the run, `<run>/fast` becomes `kernel=onetile:save relu2=1 ce=0 flat=1 rows=1`.
  - It sits on top of `flat=1` (the patch contains `wf/v9_speed.patch`), but it does not depend
    on it.
- **Expected gain: about -119 ms per step (-8.1%).** Source: one whole-program compile-only
  estimate (XLA estimated cycles, libtpu 0.0.50, `v6e:1x1`). Program: `accumulate` of
  `configs/v9.py`, `take`, 2 pieces of 4 x 1024, `kernel=onetile:save relu2=1 flat=1`. Step time =
  64 pieces x per-piece cycles + once-per-step cycles, at 0.521 ms per M cycles.

  | program | M cycles/piece | once | step M cycles | predicted step | tok/s (synthetic) | peak GiB |
  |---|---|---|---|---|---|---|
  | baseline: v9, take, 4 x 1024, `flat=1` (colleague's compile) | 47.661 | 3.360 | 3,053.6 | 1,462.3 ms | 179.3k | 2.215 |
  | `rows` part (a) only (experiment) | 44.604 | 3.584 | 2,858.2 | 1,360.5 ms (-101.8) | 192.7k | 2.023 |
  | **`rows=1` (a)+(b), delivered** | **44.093** | **3.617** | **2,825.6** | **1,343.4 ms (-118.9)** | **195.1k** | **1.974** |

  - The cost model put v9 without `flat` at 1,508 ms; the chip measured 1,433 ms. So expect
    about -110 to -119 ms on the chip, i.e. the live rate goes from about 178.5k to about
    192-194k tok/s. This is an estimate; the chip command below measures it.
  - T = 2048 (2 x 2048 pieces) was not compiled. The pieces carry the same 4,096 tokens, so the
    saving should be of the same size.
- **Exactness: bit for bit.** The switch runs the same operations on the same values. No sum
  changes order.
  - (a) keeps the forward's residuals of the key RMS norm (the float32 gathered keys and their
    reciprocal RMS) instead of computing them again.
  - (b) is reshapes only. The scan still sums each row cotangent over the 8 loops in bfloat16,
    with the same adds in the same order (loop 7 first); only the layout of the accumulator
    differs.
  - `tests/model/test_fast_rows.py` runs on a tiny v9-like model (token neurons rho x dk = 4 x 32
    = 128 lanes, so part (b) applies, as it does for 8 x 192; output gains, value embeddings,
    remat, non-zero pages and gains). It compares `bes.train.train.accumulate` (loss, every
    gradient leaf, weight, metrics) with the switch off and on, at `micro_split` 1 and 2:
    - float32: largest difference **0**;
    - bfloat16: largest difference **0**;
    - bfloat16 with `flat=1` on as well: **0**.

    The comparison is exact (`== 0`), not a tolerance. Together with
    `test_supervisor.py::test_run_24h_applies_the_speed_switches_of_a_run` (updated for the new
    name), 5 passed in 156 s on cores 2 and 3. With `rows=1`, the traced program of the final
    code is byte-identical to the one that was compiled and tested.
  - One TPU-only remark. In today's program the forward key norm and its rematerialisation are
    two different fusions. If they reduce the 192 squares in different orders, today's backward
    can use a reciprocal RMS one ulp away from the forward's. With `rows` the backward uses the
    forward's own value. On CPU the two are identical, which the test shows.
- **Switch off:** the traced program is byte-identical to HEAD's. I compared the `make_jaxpr`
  text of `configs/v9.py` (`take`, `kernel=onetile:save relu2=1`) for three functions:
  `value_and_grad(loss_fn)` at 4 x 1024, `accumulate` with `micro_split` 4 over 2 x 16 x 1024,
  and `loss_fn`. Each was traced in an untouched `git archive HEAD` tree and in this worktree
  with `flat` and `rows` off, and all three are byte-identical (`cmp`). With `rows=1` all three
  differ, as expected.
- **Parameter tree, checkpoint, optimiser: unchanged.** Only how the forward and backward are
  computed changes. `generate.py` calls `token_rows` and is unaffected: (a) only matters under
  differentiation, and (b) lives in `forward`.

## What the switch does

`bes/model/model.py`:

- **(a) The key preparation keeps its residuals** (`token_rows`). Today,
  `jax.checkpoint(keys_of)` gathers, normalises and casts the keys in the forward, then gathers
  and normalises them again in the backward to get the norm's residuals. With `rows` there is
  no checkpoint, and the forward keeps the float32 gathered keys (50 MB per 4 x 1024 piece; the
  reciprocal RMS is tiny).
  - What the forward keeps now: these f32 keys, plus, as before, the bf16 prepared keys and
    values that the loop scan closes over (25 MB each), the gain rows and the tokens.
  - Peak memory still falls, from 2.215 to 1.974 GiB, because the buffers of the removed
    relayouts are gone.
  - What disappears per piece, in M cycles:
    - the relayout copies of the whole key table `f32[8192,3072]` for the gather (0.49 forward,
      0.49 remat);
    - the remat gather (0.21);
    - the forward gather's relayout to the tokens-in-lanes layout (0.21);
    - two `f32[4096,3072]` row copies (0.12 each);
    - the remat norm (0.10);
    - the backward copy of the bf16 keys (0.11).
  - XLA also fused the scan's cotangent sums differently: the four bf16 accumulations went into
    one fusion, and the f32 `gain/o` accumulation fell from 0.58 to 0.17.
  - Net for (a): -3.06 M per piece, +0.22 M once.
- **(b) The loop scan closes over lane-dense rows** (`forward`, `_lane_dense`, `tok_of`).
  - `forward` reshapes each array that `token_rows` returns from `[..., m, n]` to `[..., m*n]`,
    when `n` is not a multiple of 128 lanes and `m*n` is. For v9 that is `[B, T, 8, 192]` ->
    `[B, T, 1536]`; scalars and lane-dense arrays are left as they are. Each loop step reshapes
    them back before calling `block`.
  - The cotangents that the scan sums over the 8 loops are then carried as
    `bf16[4,1024,1536]{1,2,0}`, with no padding. Before, they were `bf16[4,1024,8,192]` in
    `(16, 256)`-padded tiles, 2.7 times the bytes.
  - Measured on top of (a): the scan's accumulation fusions fall from 2.22 to 1.47 M per piece.
    XLA also re-lays out the neighbouring contraction fusions (keys now tokens-in-sublanes,
    `{1,3,2,0}`), and the net is -0.51 M per piece.
  - No relayout copy of the reshape appears inside the loop.

**Merging with the other engineer's switch (`tokn`).** I did not edit `token_preact` or
`token_neurons`.
- (a) does not depend on any layout.
- (b) works on whatever arrays `token_rows` returns, and it is a no-op for rows that are
  already lane-dense. For example, `[B, T, 1536]` or tokens-in-lanes `[B, rho, dk, T]` are left
  as they are.
- If `tokn` keeps `[B, T, rho, dk]` and changes only the contractions, (b) still applies, and
  the combination needs one compile to check: (b) changed the layouts that XLA picked for his
  fusions, by -0.44 M per piece in the "page neurons" area here.

## What was tried, and what is left

- Two whole-program compiles, about 554 s each, run in parallel on cores 2 and 3:
  - (a) alone: `--set bes.model.model._ROWS_DENSE=0`, a temporary knob that has been removed;
  - (a)+(b).

  Both paid, so the switch carries both.
- HLOs, logs and tools are in `wf/v9_rows_work/`:
  - `hlo/v9_take_flat_rows_r4t1024.hlo` and `hlo/v9_take_flat_rowsA_r4t1024.hlo`, plus copies of
    the baselines;
  - `diffk.py`: a per-fusion diff keyed by frames outside `model.py`, whose line numbers moved;
  - `jaxpr_v9.py`: the HEAD comparison; the dumps are in `jx_head`, `jx_off` and `jx_on`.
- **Part (b) for the gain rows: not needed.** `gain/o` rows are 768 or 128 lanes wide, so they
  are already lane-dense. Their f32 accumulation fell from 0.58 to 0.17 M per piece as a side
  effect of (a).
- **(c) Not done: casts and relayouts between the table, the gathered rows and the block.**
  What remains per piece, in M cycles:
  - the backward copy `f32[4096,3072]` feeding the key scatter (0.25);
  - the copy `f32[4,1024,2,8,192]` feeding the value scatter (0.19);
  - one forward copy `f32[4096,1,3072]` (0.12);
  - the key norm's forward and backward fusions (about 0.9 together).

  A next step would be to normalise and cast in the flat `[B*T, 3072]` layout, with a segmented
  sum over the 192 lanes, so that the gather output and the scatter input need no relayout. Its
  value is at most about 0.5 M per piece (about 17 ms per step), and it did not fit in the time.

## Chip measurement (idle chip, about 3 minutes)

From the checkout with this patch:

    python scripts/tpu_kernels.py --config configs/v9.py \
      --override model.table_grad=take --override optim.micro_seqs=4 \
      --kernels "onetile:save+relu2+flat;onetile:save+relu2+flat+rows" --steps all \
      --step-rows 4 --lengths 1024 --calls 3 --step-calls 2 --budget-s 200

How to read the output:
- Compare the two `whole step 64 x 4 x 1024 ... ms ... tok/s` lines. The prediction is about
  1,462 -> 1,343 ms in the model; on the chip, expect about 110-119 ms less with `+rows`.
- The `value_and_grad` lines already include `rows`, because the switch acts inside the
  model. Expect the second entry to be about 3.5 M cycles, roughly 1.8 ms, faster per piece.
- Their loss difference must be 0.

If it holds, write `rows=1` into `<run>/fast` next to `flat=1`. The switch is read at trace
time, so it takes effect at the next program build (restart or graft), like `flat`.
