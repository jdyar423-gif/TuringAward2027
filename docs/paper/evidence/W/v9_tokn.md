# v9_tokn: the token neurons with the tokens in lanes (switch `tokn`)

## Decision

- **Switch:** `tokn` (`bes.model.layers.FAST_TOKN`, off by default, registered in `bes/model/fast.py`
  FLAGS). Turn it on with flat: `<run>/fast` = `kernel=onetile:save relu2=1 ce=0 flat=1 tokn=1`.
  The run must restart from its checkpoint to trace it, as for `flat`.
- **Expected gain: about -64 ms per step (-4.3%) at 4 x 1024 on top of `flat=1`.** Predicted step
  1,462.3 -> 1,398.7 ms. Synthetic rate 179.3k -> 187.4k tok/s; live about 170k -> 178k (x 281/296).
  - **How it was estimated:** one whole-program compile-only of `accumulate` over 2 pieces (v9,
    `table_grad=take`, 4 x 1024, `micro_seqs=16`, `flat=1`, `kernel=onetile:save relu2=1`, libtpu
    0.0.50 for v6e:1x1). XLA's estimated cycles per piece go 47.661 -> 45.755 M (-1.906 M, -4.0%).
    The once-per-step part is unchanged (3.360 M). The step change is 64 pieces x -1.906 M x
    0.521 ms per M cycles = -63.5 ms.
  - The baseline is the colleague's `v9_take_flat_r4t1024` compile (same tool, same settings).
  - T=2048 (2 x 2048 pieces) was not compiled. Scaled by 0.542/0.521, it is about -66 ms per step.
- **Of the -1.906 M per piece, -1.22 M sits in the token-neuron code itself (-41 ms).** The rest
  comes from XLA's changed fusion and layout choices around the block (feed-forward products
  -0.31 M, q/k/v projections -0.67 M). That part is the least certain; the chip settles it.
- **Peak memory** goes 2.215 -> 2.020 GiB per piece.
- **Exactness.** The products are the same and everything outside the token neurons is the same.
  Some float32 sums may add in another order:
  1. forward pre-activations `a[b,j,t] = sum_c z[b,c,t] key[b,j,c,t]`: the sum over `dk` (192);
  2. forward output `o[b,c,t] = sum_j s[b,j,t] val[b,j,c,t]`: the sum over `rho` (8);
  3. backward `dz = sum_j da_j key_j` (over `rho`) and `ds_j = sum_c do_c val_jc` (over `dk`).

  The outer products `dkey = da_j z`, `dval = s_j do` are single products, so they have the same
  values. Their sum over the loops is the scan's own, unchanged: loop order, in bfloat16.
  - The reference computes these sums with `einsum` (dot_general); the switch uses float32
    `multiply` + `reduce_sum`. The dtype path is the same: bfloat16 products are exact in float32,
    the accumulation is float32, and the casts to bfloat16 sit where they were.
  - With the switch off, the traced programs are byte-identical to HEAD's: `make_jaxpr` of v9
    `value_and_grad(loss_fn)` at 4 x 1024 and of `accumulate` over 2 x 4 x 1024 (take), compared
    with `cmp` against a `git archive HEAD` tree.
- **Measured differences** (`tests/model/test_tokn.py`, CPU, jax 0.11.1). Tiny v9 configuration:
  token neurons 4 x 16, `gain_o`, value embeddings, remat, non-zero pages and gains, `take`. The
  test compares `accumulate`'s loss, weight and all 64 gradient leaves, off against on:

  | act dtype | micro_split | loss | gradient leaves bit for bit | worst leaf (max abs diff / leaf max) |
  |---|---|---|---|---|
  | float32 | 1 | bit for bit | 10 / 64 | 5.6e-6 (`lanes/beta`) |
  | float32 | 2 | bit for bit | 10 / 64 | 6.1e-6 (`lanes/beta`) |
  | bfloat16 | 1 | bit for bit | 64 / 64 | 0 |
  | bfloat16 | 2 | bit for bit | 64 / 64 | 0 |

  - Tolerances in the test: float32 2e-5 per leaf, loss 1e-6. bfloat16 one rounding step (2^-7)
    per leaf, loss 1e-4.
  - The test also checks the layer alone in float32: output within 1e-6 of its largest entry,
    every input's gradient within 1e-5.
  - On the TPU, the reduction order of the reference fusions differs from the CPU's. A bfloat16
    run there can therefore show float32 rounding differences, which bfloat16 rounding steps carry
    through the backward pass, rather than bit-for-bit equality.
  - 4 passed in 260 s. `tests/train/test_supervisor.py`'s `fast.current()` assertion has the new
    name (like flat's), and it passes.
  - `tests/model/test_model_page.py` was also run with the switch forced on for the whole session
    (`wf/v9_tokn_work/run_on.py`): 15 passed. These include the numpy reference, the float64
    gradients (`onehot`, `sorted`), "zero values bit identical", "page switch off", key-scale
    invariance, remat + bfloat16, the block-hook interface and the splash path.
  - The other 4 failed by design. They assert the reference layout `[B, T, rho, dk]` of the rows,
    either directly or by calling `token_preact` with the rows of `token_rows`.

## What changed (behind the switch only)

- **`bes/model/model.py` `token_rows`.** With `FAST_TOKN`, the prepared rows (after the key norm
  and the casts, which are unchanged) are moved once per forward, outside the loop scan:
  `[B, T, rho, dk] -> [B, rho, dk, T]` (`jnp.moveaxis(r, 1, -1)`).
  - The gather, the key preparation, its remat and the casts are untouched, so this merges with
    the other engineer's part.
  - `generate.py` uses the same `token_rows` and `block`, so it stays consistent with the switch
    on.
- **`bes/model/layers.py` `token_neurons`.** With `FAST_TOKN` it calls `_token_neurons_lanes`:
  - `z = rmsnorm(x @ pin)`, computed as in `token_preact`, then moved to `[B, dk, T]`;
  - `a = sum_c(z * key) / sqrt(dk)`, a float32 sum over the sublane axis, giving `[B, rho, T]`;
  - `s = relu(a)^2`, cast to bfloat16;
  - `o = sum_j(s * val)`, a float32 sum over a major axis, giving `[B, dk, T]`;
  - `o` goes back to `[B, T, dk]` for `@ pout`.

  In this layout, `dk = 192` lies in sublanes (12 bfloat16 tiles of 16, 24 float32 tiles of 8) and
  `T` in lanes, so nothing is padded. In the reference, each token's `[8, 192]` block pads to
  `[16, 256]`.

## Where the cycles moved (M cycles per piece, flat=1 -> flat=1 + tokn)

| area (`v9_areas.py`) | flat | flat + tokn |
|---|---|---|
| page neurons (contractions, key norm, row transposes) | 8.640 | 7.421 |
| table: page/key+val (gathers, scatter) | 1.039 | 1.120 |
| accumulate (train.py) | 1.116 | 1.079 |
| rest of model | 36.146 | 35.416 |
| **total per piece** | **47.661** | **45.755** |

**The contraction fusions** (forward, rematerialised and backward: `a`, `o`, `dz`, `ds`, 2
blocks x 8 loops) fall from 5.875 to 4.702 M. Per fusion they now cost 42-54k cycles, against
56-76k before.
- The forward ones read float32 rows: XLA hoists the bfloat16 -> float32 converts of the
  closed-over rows out of the forward loop.
- In the cost model those are HBM-bound at 44k. The backward ones read bfloat16 and are
  compute-bound at 54k.

**New: the moves of the rows to lanes**, about +0.25 M per piece in the forward (`model.py`
`token_rows`). In the backward, the moves of the rows' cotangents back to the gather's layout add
+0.23 M (copies under `controller.py:109`).

**Under `controller.py:109`** (the scan's accumulation of the rows' cotangents; the other
engineer's code, not changed here), the outer-product accumulations go from 2.02 to 1.87 M. With
those copies, that frame goes from 3.80 to 4.01 M.

**Feed-forward products** (`mlp`, which shares `x` with the token neurons): -0.31 M.
**q/k/v projections** (`attention.py:373`): -0.67 M. These are XLA's fusion choices around the
block. The source of that code did not change.

## Tried / not tried

- **Tried:** the tokens-in-lanes layout with explicit float32 products and sums (this switch).
  One compile. It pays, though less than the 3-4 M per piece the colleague had guessed. In the
  cost model the dense fusions are compute-bound (bfloat16 -> float32 converts, multiply, add), not
  byte-bound.
- **Not tried: lane-dense `[B, T, 1536]` rows** with MXU segment sums. Exact float32 products need
  a hi/lo bfloat16 split (two MXU passes) and an 8-fold lane tiling of `z` across 192-lane
  segments. That is more VPU work per element than the lanes layout, so this variant was set aside
  without a compile.
- **Not tried: keep `a` from the forward, so the remat skips the `a` contraction** (about
  0.77 M per piece, about 26 ms per step). It needs a named-residual policy in the loop step's
  `jax.checkpoint` (`controller.py`). That file is in the other engineer's part (the summation of
  the cotangents over the loops), so it was left alone.
- **Not checked:** the 2 x 2048 compile, and a whole-program compile with `tokn` alone (no `flat`).
  The change is independent of the accumulation, so a similar gain is expected without `flat`.

## Chip measurement (idle chip, needs this patch; about 2-3 minutes)

From the checkout on the TPU host:

    python scripts/tpu_kernels.py --config configs/v9.py \
      --override model.table_grad=take --override optim.micro_seqs=4 \
      --kernels "onetile:save+relu2+flat;onetile:save+relu2+flat+tokn" --steps all \
      --step-rows 4 --lengths 1024 --calls 3 --step-calls 2 --budget-s 240

**How to read it.** Under each entry, the line `whole step 64 x 4 x 1024 ... ms ... tok/s` is the
decision. Predicted values:
- `flat`: about 1,462 ms;
- `flat+tokn`: about 1,399 ms.

The `value_and_grad` lines should differ by about 1 ms per piece (-1.9 M cycles of the 4 x 1024
piece). Their loss difference is float32 rounding, at most about 1e-6 relative, not necessarily 0.

**For T=2048 too:** use `--lengths 1024,2048`. The 2048 steps are 2 x 2048 pieces. This adds
2 compiles per entry, so about 4-5 minutes.

## Files

- Patch: `wf/v9_tokn.patch` (git diff against HEAD 9a4d652, the flat patch included; applies to
  HEAD with `git apply`).
- Test: `tests/model/test_tokn.py`.
- Work: `wf/v9_tokn_work/`:
  - `hlo/v9_take_flat_tokn_r4t1024.hlo(.json)`, the compile;
  - `jaxpr_v9.py` and `jx_head`/`jx_off`/`jx_on`, the jaxpr check;
  - `ctl.py`, the scan-frame breakdown;
  - `run_on.py`, pytest with the switch forced on;
  - `test_tokn.log`, `page_on.log`.
