# speed_mlp_block: the feed-forward sub-layer, the norms and the residual plumbing

The patch is at `wf/speed_mlp_block.patch`. It changes `bes/model/layers.py` and adds
`tests/model/test_mlp_block_speed.py`. `block()` in `bes/model/model.py` is untouched, as are
`bes/config.py` and every configuration field. Nothing is committed.

**Result: one switch, `bes.model.layers.FAST_RELU2` (default `False`).** It gives the same values,
bit for bit on CPU. The ledger sees no change, because a jaxpr-level pass count cannot see this one.
The TPU compiler's own cost model, run on the chip's libtpu, estimates these reductions in
training-step time:
- 4 x 1024 pieces (`micro_split=4`): -2.5%;
- 2 x 2048 pieces: -5.2%;
- 16 x 1024: -0.5%.

Every other candidate in my area was either already fused by XLA on the TPU or not exact (both
listed below).

## 1. FAST_RELU2: a hand-written backward for relu(h)^2

**The change.** With the switch on, `mlp()` calls `layers.relu2(h)` in place of
`jnp.square(jax.nn.relu(h.astype(f32)))`. `relu2` is a `jax.custom_vjp`:
- its forward is the same expression;
- its residual is the activation-dtype pre-activation `h` (bf16), which is already stored;
- its backward is `ct_h = (ct * (2 * relu(f32(h)))).astype(h.dtype)`.

Autodiff of the reference computes the same product `ct * (2 * relu(h))` (from `square`'s JVP,
`g * (2 * x)`) and then `select(h > 0, ., 0)` (from `relu`'s custom JVP). Where `h <= 0` the
product is already zero, so the select changes only the sign of an exact zero. It also needs the
predicate `h > 0` as an extra residual.

On TPU v6e that predicate is costly. In the compiled program (below), the rematerialised `W_up`
product's epilogue computes `compare`, `iota`, `shift-left` and a `reduce`, and packs the bits of
the 16 batch rows into a `u16[1024,3072]` mask. The backward `dh` product's epilogue then unpacks
it (`iota`, `shift-left`, `and`, `convert`) and applies the `select`. With the switch both disappear.

**Exactness: tolerance zero.** Every comparison uses `np.testing.assert_array_equal` on the loss,
every metric and every gradient leaf. That function treats -0.0 and +0.0 as equal, and the sign of
an exact zero is the only difference the switch can make. NaN propagation also differs, but only
where `h <= 0` and the incoming cotangent is already inf or NaN, and that step is skipped as
non-finite either way. The tests, in `tests/model/test_mlp_block_speed.py`:
- `test_relu2_has_the_values_and_the_gradient_of_the_reference_expression[bf16, f32]`: values and
  VJP against the reference, with +0, -0 and negative inputs, under `jit` and inside a
  checkpointed `scan`.
- `test_fast_relu2_feed_forward_values_and_gradients[bf16, f32]`: `mlp()` with and without a gain;
  value and gradients for x, W_up and W_down.
- `test_fast_relu2_model_loss_and_gradients_are_unchanged` (marked slow, about 100 s), switch off
  against on for `value_and_grad(loss_fn)` on the tiny config with the live run's features. These
  are splash in interpret mode, bf16, remat, sink, candidate tape in and out, registers, value
  embeddings, `table_grad="take"`, `clock_planes=0` and `use_conf=False`. Three cases:
  - (a) bf16;
  - (b) float32;
  - (c) `gain_ffn=8`, `gain_phases=2` (the PhaseGain path after the activation), `remat=False`.

  All are bit-identical. The existing `test_feed_forward_stores_pre_activations_in_the_activation_dtype`
  still passes, with the switch off.

Run: `taskset -c 2,3 nice -n 10 <venv>/bin/python -m pytest tests/model/test_mlp_block_speed.py -o addopts="-p no:cacheprovider" -q -x`
(7 passed in about 100 s).

### Ledger, before and after

Glue per micro-batch:

| shape | glue pass | mat |
|---|---|---|
| 16 x 1024 | 87,465 -> 87,465 MB (±0) | 49,104 -> 49,104 (±0) |
| 8 x 2048 | 89,086 -> 89,086 MB (±0) | 49,517 -> 49,517 (±0) |

By class: `select_n` -3,221 MB and `convert` +3,221 MB; the fused `elementwise` class rises by
2,416 MB. The new backward converts the bf16 `h` to float32 again, which the ledger counts as a new
pass. XLA puts that convert into the epilogue of the same backward product that already reads `h`
(next section). The ledger has no line for the packed mask, which XLA creates after the jaxpr. So the
ledger cannot show this change.

### TPU v6e compile-only, with the chip's own compiler

The libtpu 0.0.50 wheel is already in this session's scratchpad (`dl_libtpu0050/x`). jax 0.11.1
compiles for `v6e:1x1` with it, with no TPU attached. I compiled `train.accumulate` over one
micro-batch, the same program the ledger traces, with real splash kernels
(`splash_interpret=false`). Then I read the optimised HLO:
- HBM bytes per top-level op, where on-chip `S(1)` buffers count zero and while-loop trip counts
  are applied;
- element ops in fusions;
- XLA's own `estimated_cycles` from each fusion's backend_config.

**Calibration.** The program's total estimated cycles match the chip's measured times (commit
b05bfe7, `scripts/tpu_micro.py`) at both shapes, at the same rate:
- 140.73 M cycles against 73.36 ms per 16 x 1024 micro-batch;
- 31.85 M cycles against 16.61 ms per 4 x 1024 piece.

Both are 1.92 G cycles/s. So I convert cycles to milliseconds at 0.521 ms per M cycles.

| program (one micro-batch, remat on) | est. cycles | HBM (my count) | element ops | peak (compiler) |
|---|---|---|---|---|
| 16 x 1024 reference | 140.73 M | 68.54 GB | 100.2 G | 1.680 GiB |
| 16 x 1024 FAST_RELU2 | **140.06 M (-0.48%)** | 68.34 GB | 89.7 G | 1.604 GiB |
| 4 x 1024 reference | 31.85 M | 8.09 GB | 25.4 G | 0.624 GiB |
| 4 x 1024 FAST_RELU2 | **31.05 M (-2.5%)** | 7.77 GB | 22.8 G | 0.610 GiB |
| 2 x 2048 reference | 35.12 M | 8.20 GB | 25.4 G | 0.627 GiB |
| 2 x 2048 FAST_RELU2 | **33.30 M (-5.2%)** | 7.64 GB | 22.8 G | 0.610 GiB |
| 4 x 1024, remat off, reference | 36.46 M | 39.65 GB | 19.2 G | 4.313 GiB |
| 4 x 1024, remat off, FAST_RELU2 | 34.67 M (-4.9%) | 34.88 GB | 18.6 G | 3.750 GiB |

Per fusion at 4 x 1024, in thousands of cycles per application (reference -> switch on):

| fusion | block 0 | block 1 |
|---|---|---|
| rematerialised `W_up` product | 73.7 -> 48.8 | 76.3 -> 41.0 |
| backward `dh` product | 49.2 -> 45.0 | 139.2 -> 100.2 |
| `dW_down` | 48.1 -> 39.9 | 73.9 -> 48.1 |

The mlp area of the loop body drops from 0.929 to 0.802 M cycles per trip (-14%).

At 2 x 2048 the mask is dearer: the rematerialised `W_up` products cost 115.7 and 133.8 k cycles,
falling to 48.8 and 41.0 k. The mlp area there drops from 1.076 to 0.823 M cycles per trip (-24%).

At 16 x 1024, the rematerialised `W_up` product drops from 396.0 to 366.5 k cycles. The
compiler's estimate for the `dh` product barely moves there (488.9 to 487.7 k), although its element
ops halve (654 to 352 M).

Without remat the switch also saves memory. The reference keeps the float32 `2 relu(h)` and the
mask for every loop; FAST_RELU2 keeps the bf16 `h`. The peak at 4 x 1024 falls by 0.56 GiB,
which would be about 2.2 GiB at 16 x 1024.

### Expected effect per 32 x 1024 micro-batch

| derivation | estimate |
|---|---|
| Compiler cost model, 4 x 1024 pieces (`micro_split=4`, the faster setting from commit b05bfe7) | -0.80 M cycles per piece x 8 pieces x 0.521 ms/M = **-3.3 ms** (2.5% of 132.9 ms) |
| Compiler cost model, 16 x 1024 pieces | -0.67 M x 2 x 0.521 = **-0.7 ms** (0.5% of 146.7 ms) |
| Compiler cost model, 2 x 2048 pieces (last third of the run; calibration 35.12 M cycles against 19.03 ms, so 0.542 ms/M) | -1.82 M x 8 x 0.542 = **-7.9 ms per 32,768 tokens** (5.2% of 152.3 ms) |
| Your pass model alone (HBM only) | about -0.6 ms (below) |

For the pass model: the removed u16 mask is 1/16 of a pass over the bf16 hidden layer. It is written
once and read once in each of the 16 block applications. At about 0.32 ms per hidden-layer pass at
32k tokens (the d x 4d product), that gives 16 x 2 x 0.02 ms. This model leaves out the vector-unit
work of packing and unpacking the mask, which the compiler counts.

Caveats on the compiler numbers:
- They are XLA's own model, not a timing.
- The splash kernels get 0 cycles and async copies count as free.
- The single calibration rate held at two shapes 4 times apart (1.92 G cycles/s, and 1.85 at
  T=2048), but the effect of one change can still be off.

Because the change removes work and adds none, its sign is not in doubt; its size is.

**What to check on the chip:**
- the same loss and gradient norm, bit for bit;
- the step time with the switch off and on, at the run's `micro_split`.

## 2. My area on the TPU: what the ledger's top lines really are

Read from the compiled program (`wf/speed_mlp_block_work/tpuc/base_loop_ops.txt` lists every
loop-body op at 16 x 1024).

**Norms.** There are 5 norms per loop (2 per block plus the loop norm), so 40 per micro-batch plus
embedding and head.
- **Forward:** the sum of squares is fused into the epilogue of the product that made `u` (`W_o`,
  `W_down`, or the lane combine). The normalisation is fused into the operand of each consuming
  product. The exception is block 0's g1 norm, which is written once to VMEM because the
  value-embedding gate also slices `x`.
- **Backward:** the row reduction and the gain reduction sit in the epilogue of the `dx` product.
  One element-wise pass follows (about 24 k cycles per norm at 4 x 1024).
- No float32 `[N, d]` tensor of a norm reaches HBM. The isolated 0.23 ms per norm from
  `block_profile` F is not what a norm costs inside the fused program. The norms come to 0.27-0.30 M
  cycles per loop trip at 4 x 1024 (8%) and 0.99 M at 16 x 1024 (6%). The remaining structure, two
  reductions and one pass, is the minimum for this arithmetic.

**Residual adds and their casts** (`model.py:483`/`490`; ledger mat 805 MB each). In the forward
and remat they are fused into the `W_o` / `W_down` epilogues, which also compute the next norm's
statistics. In the backward, XLA rewrites the backward products to read the bf16 cotangent (for
example the `dh` product reads `bf16[16,1024,768]` directly), so no float32 cotangent of the
residual branch is written. The ledger's note that "every backward product has a float32 operand"
is true of the jaxpr, not of the TPU program.

**`mm()`'s `out_dtype`** (`layers.py:72`; 6,493 MB pass, 3,255 MB mat). The convert to bf16 is
folded into the product's output, and the backward float32 cotangent `dh` is never materialised:
the `dh` epilogue writes it in bf16 and both consumers read bf16.

**The feed-forward forward.** `relu^2` is computed inside the `W_down` product's operand fusion, so
`h6` is never written (ledger `layers.py:134`, 3,221 MB mat). At 4 x 1024 the products run at their
MXU time in the cost model (41 k cycles for 19.3 GFLOP).

## 3. Tried and dropped

**FAST_MM_LO** (`mm(..., out_dtype=bf16)` as one product with `preferred_element_type=bf16`).
- In the ledger it looks large: -6,997 MB of glue (-8.0%) together with FAST_RELU2.
- On the TPU the compiled program is unchanged: identical cycles (31.05 M) and HBM bytes. XLA
  already folds the convert.
- It is not exact on CPU: the gradients of 57 leaves differ, by up to 2.4% relative. XLA CPU's
  bf16-output dot is not "f32 sum rounded once".
- Dropped, and the code is removed.

**FAST_RELU2 with a float32 residual** (keep `f32(h)` instead of `h`).
- With remat on, the TPU program is identical.
- The ledger would show -3,221 MB (-3.7%), because the backward convert moves into the remat.
- Without remat it keeps twice the bytes per loop.
- Dropped in favour of the bf16 residual. The ledger's preference is an artefact of how it counts.

**Not implemented, because they are not exact or not mine:**
- **Square in bfloat16:** a variant of "feed-forward entirely in bfloat16", which lost on the chip.
- **Folding the norm gains into the next weights:** not exact, because `bf16(y*g)` and
  `bf16(g*W)` round differently.
- **Restructuring the norm backward:** already at the structural minimum; any reordering changes
  the float32 rounding.
- **Summing the two cotangents of `u` in float32:** changes values, because the reference rounds
  each to bf16 first.
- **Splitting the hidden layer along d_ffn so it fits VMEM:** changes `W_down`'s accumulation
  order. `micro_split=4` already puts it in VMEM.
- **A fused Pallas MLP:** not exact.
- **Keeping block 1's output across remat to skip one rematerialised `W_down`** (1 of the 16 MLP
  products per loop): needs a remat policy in `controller.py`, and a policy that keeps outputs lost
  on the chip before (`remat_keep=kernel`).

## 4. For the orchestrator

**The compile-only method is reusable by all four engineers:**
- `wf/speed_mlp_block_work/tpuc/tpu_compile.py`: run from a checkout with `--rows R`,
  `--seq-len T`, `--set module.FLAG=1` and `--override`; about 2 minutes and 2 GB per compile.
- `hlo_bytes.py`, `areas.py [--diff]` and `fusions.py PROG AREA`.

These read XLA's `estimated_cycles`, which tracked the chip's time at both shapes. For my area,
the ledger's largest lines are fusions on the TPU. I expect the same for the other areas' converts
and reductions next to products.

To reproduce, from a checkout, with `PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python`
and `T=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/speed_mlp_block_work/tpuc`:

    taskset -c 2,3 nice -n 10 $PY -B $T/tpu_compile.py --rows 4 --out /path/base.hlo
    taskset -c 2,3 nice -n 10 $PY -B $T/tpu_compile.py --rows 4 --set bes.model.layers.FAST_RELU2=1 --out /path/fast.hlo
    cd $T && $PY -B areas.py /path/base.hlo --diff /path/fast.hlo      # per loop trip, by area
    cd $T && $PY -B areas.py /path/base.hlo 1 --diff /path/fast.hlo    # outside the loops
    cd $T && $PY -B fusions.py /path/fast.hlo mlp                       # one area's fusions

**Seen outside my area at 4 x 1024 (cost model):**
- The attention projections' backward products cost 63-87 k cycles each for 4.8 GFLOP, about 7-9
  times their MXU time; the `[768, 6, 128]` weight-gradient layout is a likely cause.
- Non-product fusions are 30% of the loop body: attention 0.53 M, norms 0.30 M, controller 0.20 M
  cycles per trip.
- The memory-space copies (prefetches and evictions) are about 500 MB per trip. They are free in
  the cost model, but probably not entirely hidden on the chip.

## Files

- `wf/speed_mlp_block.patch`
- `wf/speed_mlp_block_work/` holds:
  - the ledger JSONs (`base_1024.json`, `base_2048.json`, `relu2_1024.json`);
  - `check_switch.py`, an ad hoc off/on comparison;
  - `tpuc/`: the compile tools, the HLO dumps of every variant, and their `.hlo.json` with memory
    and cost.
