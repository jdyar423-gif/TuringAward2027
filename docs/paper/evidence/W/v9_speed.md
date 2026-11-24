# v9_speed: how to run the v9 training step fastest on TPU v6e

## Decision (start with this)

1. **`model.table_grad=take`** (override, as for the live run), **not** v9's configured `sorted`, and
   not `onehot`. The compiler's cost model: `sorted` costs +3.89 M cycles per 4 x 1024 piece
   (+130 ms per step, +8.6%). `onehot` costs +2.26 M more than `take` on the page and gain tables alone
   (micro-program, +75 ms per step).
2. **Keep `micro_split=4` at both lengths** (pieces of 4 x 1024 and 2 x 2048, as live). Bigger pieces
   lose for v9 too: 8 x 1024 costs +50 ms per step, 16 x 1024 costs +189 ms. The per-piece table and
   gradient-accumulation costs they save are smaller than what the rest of the model loses at 8 and
   16 rows.
3. **Turn on the new switch `flat=1`** (patch below; bit for bit the same values) in `<run>/fast`:
   `kernel=onetile:save relu2=1 ce=0 flat=1`. Its gain in the whole-program compile is -1.39 M
   cycles per piece, +0.70 M once per step: **-46 ms per step (-3.0%)** at 4 x 1024.
   - With `flat=1`, 8 x 1024 pieces come out 1,499 ms in the model, against 1,462 ms for 4 x 1024.
   - The model overstates the 8-row penalty by about 36 ms (see below), so on the chip they may be
     tied.
   - The chip command below settles 4 against 8 rows.
4. **Expect v9 to train at about 0.59 of final's rate, not 0.7-0.8.** These are estimates: the cost
   model plus the chip's measured final step.

   | plan | step (262,144 tok) | synthetic tok/s | live tok/s (x 281/296) |
   |---|---|---|---|
   | final now (measured) | 884.5 ms | 296k | 281k |
   | v9, take, 4 x 1024 | 1,508 ms | 174k (0.59) | ~165k |
   | **v9, take, 4 x 1024, `flat=1` (recommended)** | **1,462 ms** | **179k (0.60)** | **~170k** |
   | v9, take, 8 x 1024, `flat=1` | 1,499 ms (about 1,463 after the 8-row correction) | 175k | ~166k |
   | v9, sorted (v9's own setting), 4 x 1024 | 1,638 ms | 160k (0.54) | ~152k |
   | v9 at T=2048, take, 2 x 2048 (scaled from 1024, no compile) | ~1,633 ms | ~161k | ~152k |
   | the same with `flat=1` | ~1,585 ms | ~165k | ~157k |

   An independent check is the chip measurement of 23 Nov, made on the old program before the
   speed-ups. It had final at 112-117k and v9 at 88-92k tok/s, i.e. v9 cost +0.62 s per 262k-token
   step. The cost model gives v9 +0.624 s per step today. The page work did not get faster with the
   core, so the ratio fell.

## The table (compile-only, XLA's estimated cycles; same kernel setting `onetile:save relu2=1`)

Program: `bes.train.train.accumulate` over 2 pieces of R x T (the loop body is one piece of the
trainer's step, accumulation included), `configs/<cfg>.py`, `optim.micro_seqs=16`, libtpu 0.0.50 for
`v6e:1x1`. The predicted step is final's measured whole step (884.5 ms at 64 x 4 x 1024) plus the cycle
difference at 0.521 ms per M cycles. The attention kernels are 0 cycles in the model; they are the
same in v9 and final.

| program | piece | pieces/step | M cycles/piece | peak GiB | step M cycles | vs final live | predicted step | tok/s |
|---|---|---|---|---|---|---|---|---|
| final, take | 4 x 1024 | 64 | 30.37 | 0.87 | 1,944.7 | 0 | 884.5 (measured) | 296,375 |
| final, take | 8 x 1024 | 32 | 63.96 | 1.29 | 2,047.8 | +53.7 ms | 938.2 | 279,401 |
| v9, take | 4 x 1024 | 64 | 49.05 | 1.84 | 3,141.8 | +623.7 ms | 1,508.2 | 173,814 |
| v9, sorted | 4 x 1024 | 64 | 52.94 | 2.00 | 3,390.7 | +753.4 ms | 1,637.9 | 160,050 |
| v9, take | 8 x 1024 | 32 | 101.09 | 2.46 | 3,237.6 | +673.6 ms | 1,558.1 | 168,244 |
| v9, take | 16 x 1024 | 16 | 218.85 | 3.73 | 3,504.2 | +812.5 ms | 1,697.0 | 154,475 |
| v9, take, `flat=1` | 4 x 1024 | 64 | 47.66 | 2.22 | 3,053.6 | +577.8 ms | **1,462.3** | **179,271** |
| v9, take, `flat=1` | 8 x 1024 | 32 | 97.54 | 2.97 | 3,124.7 | +614.8 ms | 1,499.3 | 174,847 |

Peak memory is far below the 31 GiB of the chip in every variant.

For calibration: the chip measured final with the configured kernels at 1,081.5 ms for 32 x 8 x 1024
and 1,063.3 ms for 64 x 4 x 1024 (docs/CHANGES.md 19.5), a gap of +18 ms. The cost model gives +54 ms
for the same change. So the model overstates the 8-row penalty by about 36 ms. Corrected, v9 at
8 x 1024 is about 1,522 ms, still not better than 4 x 1024 (1,508).

Micro-program, the page and gain tables of one piece only (`micro_tg.py`, `micro_flat.py`: lookup,
key norm, contraction, value_and_grad, `a + w g` accumulation):

| table gradient | 4 x 1024 | 16 x 1024 |
|---|---|---|
| take | 7.44 M | 16.20 M |
| sorted | 8.59 M | 18.96 M |
| onehot | 9.70 M | |
| take with `[V, W]` tables (`flat`) | **3.39 M** | |

## Where v9's extra 18.7 M cycles per 4 x 1024 piece go (v9 take against final take)

| part | M cycles/piece | ms/step (x 64 x 0.521) |
|---|---|---|
| token-neuron contractions: `token_preact` (`layers.py:192`) and `sum_j s_j val_j` (`layers.py:214`), forward, remat and backward | +5.66 | 189 |
| float32 accumulation `a + w g` (`train.py:243`): mostly relayout copies of the page and gain tables' gradients (`f32[8192,3072]{0,1}` copies of 0.49 M each, `f32[8192,1,2,768]` with `T(2,128)` tiles 0.43 M), not the adds | +2.88 | 96 |
| scan over the loops (`controller.py:109`): cotangents of the closed-over page and gain rows accumulated over 8 loops | +2.38 | 79 |
| feed-forward products (`mlp`, `layers.py:176`): the token neurons share `W_up`'s input and add to its output | +1.71 | 57 |
| relayout of the page tables for the gather (`model.py:397`: `[V,2,8,192] -> [V,3072]`, forward and remat) | +1.23 | 41 |
| page/key and page/val gathers and scatter (`tables.py`) | +0.92 | 31 |
| `page/pin`, `page/pout` products | +0.93 | 31 |
| key RMS norm and casts (`keys_of`, `rmsnorm`) | +1.1 | 37 |
| frames not attributed (`?`) | +1.09 | 36 |
| attention gain_o (gather of rows and the multiply) | +0.17 | 6 |
| q/k/v projections | +0.32 | 11 |
| **total** | **+18.68** | **624** |

What this means:
- The **gain table is nearly free** (0.17 M per piece). The **token neurons are the cost**.
- Their per-token contractions are `[8, 192]` blocks per token. These sit in padded tiles: 8 bf16
  sublanes are padded to 16, and 192 lanes to 256, so the fusions move 2.7 times the bytes
  (about 59-73 k cycles per fusion against 15 k for the bytes themselves).
- The float32 accumulation of the 98 M-parameter tree costs only 0.23 M per piece for final.
  For v9 the cost is the table **relayouts**, not the size of the tree.
- In `take` mode, the table gradient comes out of the scatter in a vocabulary-minor layout. It is
  transposed back to the parameter's tiled shape every piece. `flat=1` removes that.

## The change: `flat=1` (`bes.train.train.FLAT_TABLES`, off by default)

`wf/v9_speed.patch`:
- `bes/train/train.py`: `accumulate` hands the loss function its token tables as reshapes of
  `[V, W]` matrices and accumulates their gradients in that shape, reshaped back once per call.
  These are the leaves `[V, ...]` with 3 or more dimensions: `emb/ve`, `gain/o`, `page/key`,
  `page/val`. The model still sees its own shapes. Its own reshape to `[V, W]` for the lookup then
  cancels in XLA, and the relayouts leave the piece loop.
- `bes/model/fast.py`: `flat` is a name in the speed switches, so `<run>/fast` and
  `scripts/tpu_kernels.py --kernels "...+flat"` can set it.

**Exactness.** The change is reshapes only. `tests/model/test_flat_tables.py` compares accumulate
with the switch off and on, with `micro_split` 1 and 2, on the tiny v9 configuration. That
configuration has token neurons, output gains, value embeddings, remat, non-zero pages and gains.
The test covers `take`, `sorted` and `onehot`. Gradients, loss, weight and metrics are bit for bit
equal (`assert_array_equal`): 3 passed in 250 s.
`tests/train/test_supervisor.py::test_run_24h_applies_the_speed_switches_of_a_run` was updated
(`fast.current()` now has `flat`) and passes.

With the switch off, the program is the same as before: the same `jax.value_and_grad(loss_fn)` call.

**Gain.** The whole-program compile (v9, take, 4 x 1024) goes from 49.05 to 47.66 M cycles per
piece:
- the accumulation falls from 3.12 to 1.12 M per piece;
- the rest of the model rises by about 0.9 M (XLA lays out the gathered rows differently);
- the once-per-step part rises by 0.70 M (the relayouts now run once per step).

That is -46 ms per step (-3.0%), 1,508 -> 1,462 ms (estimate from the cost model).

The table micro-program had promised more (7.44 -> 3.39 M). The full program keeps part of the
relayout work inside the model.

For final it changes nothing: `emb/ve [8192, 8, 128]` reshapes as a bitcast.

## One chip measurement (idle chip, about 3 minutes of chip time)

This needs the patch, for `+flat`. Run on the TPU host from the checkout:

    python scripts/tpu_kernels.py --config configs/v9.py \
      --override model.table_grad=take --override optim.micro_seqs=4 \
      --kernels "onetile:save+relu2;onetile:save+relu2+flat" --steps all \
      --step-rows 4,8 --lengths 1024 --calls 3 --step-calls 2 --budget-s 200

What it runs: two entries, each with one `value_and_grad` compile, then the whole accumulation step
of 64 x 4 x 1024 and of 32 x 8 x 1024. That is 6 compiles of about 15-25 s plus about 15 s of
timing, so about 2.5-3 minutes.

How to read it:
- The lines `whole step 64 x 4 x 1024 ... ms ... tok/s` and `whole step 32 x 8 x 1024 ...` under
  each entry give the four numbers that decide the run: piece size times `flat` off or on.
  Predicted:
  - without `flat`: about 1,508 ms (4 x 1024) and 1,522-1,558 ms (8 x 1024);
  - with `flat`: about 1,462 ms (4 x 1024) and 1,463-1,499 ms (8 x 1024).

  Final on the same chip was 884.5 ms.
- Take the smallest. If 4 x 1024 wins, `micro_split=4` (or `1024:4,2048:4`). If 8 x 1024 wins,
  `micro_split=2`.
- A 4 x 1024 piece carries the same tokens as a 2 x 2048 piece, so the result carries over to
  T=2048 (2 x 2048 or 4 x 2048).
- The two entries' `value_and_grad` lines should agree within noise. `flat` only acts in the whole
  step, so they are a noise check. Their loss difference must be 0.
- The `KERNELS {json}` last line has every number: `results[i].steps[j].ms` and `tokens_per_s`.

If 1-2 more minutes are available, run the same command with `--override model.table_grad=sorted
--kernels "onetile:save+relu2+flat" --step-rows 4`. Its step should be about 130 ms slower than
`take` with `flat`. That settles `table_grad` on the chip too. `table_grad` is a configuration
field, so it needs its own process.

## Method

- **Compile-only.** libtpu 0.0.50 (`scratchpad/dl_libtpu0050/x`), jax 0.11.1, `v6e:1x1`, from this
  CPU host, with the switches set through `bes.model.fast.apply`.
- **Tools.** `wf/v9_speed_work/tpuc_v9.py` compiles `accumulate` over A pieces, or one
  `value_and_grad`, for any config. `hlo_bytes.py` is the reader from `speed_io`.
  `estimate.py`, `v9_areas.py` and `top.py` give the per piece, the once-per-program part and the
  step, by area. `micro_page.py`, `micro_tg.py` and `micro_flat.py` are the table micro-programs.
- **Data.** HLOs and json are in `wf/v9_speed_work/hlo/`.
- **Compile time.** Each full compile took 460-500 s on this loaded host.
- **Calibration.** 0.521 ms per M cycles at T=1024 (speed_mlp_block). The step prediction is final's
  measured whole step plus v9's cycle difference.
- **What the model cannot see.** Mosaic kernels count 0 cycles, so only differences are reported.
  The kernels are identical in v9 and final.

## Not done

- No compile of v9 or final at 2 x 2048, 4 x 2048 or 8 x 2048. The T=2048 numbers above are the
  T=1024 differences scaled by 0.542/0.521.
- No whole-program `onehot` compile; only the table micro-program.
- No whole-program compile of `sorted` or `onehot` with `flat=1`, and none at 16 x 1024 with it.
- **Described, not implemented: the token neurons' padded contractions.** These are 5.7 M cycles
  per piece, the largest item. Two ways to avoid the padding:
  - tokens in lanes (`[B, rho, dk, T]` rows, with `dk` and `rho` as major dims);
  - lane-dense `[B, T, 1536]` rows, with the segment sums as small matrix products at `HIGHEST`
    precision.

  Either could save on the order of 3-4 M cycles per piece (about 100-130 ms per step; estimate).
  Both change the order of float32 sums, so the values agree only to rounding. Each needs its own
  test and a compile check, which did not fit in the time.
