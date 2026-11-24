# speed_io: embedding, tables, tape, sink and the loss head

The patch is `wf/speed_io.patch`. It changes `bes/model/head.py` only (the head's
`chunked_ce`, plus a new private function) and adds `tests/model/test_io_speed.py`. Nothing else
in `bes/` or `configs/` changed, no configuration field was added, and nothing was committed.

**Result: one switch, `bes.model.head.FAST_CE` (default `False`).** It is the cross-entropy head
with a hand-written backward pass. It gives the same values to float32 rounding. It removes the
second pass over the float32 logits, and (with more than one loss chunk) the recomputation of
the logit product in the backward pass. The estimates come from the TPU compiler's own cost
model (libtpu 0.0.50, compile-only for `v6e:1x1`, calibrated at 0.521 ms per M cycles; method in
the speed_mlp_block report):

| micro-batch | before | after | change |
|---|---|---|---|
| 4 x 1024 piece (the live run, `micro_split=4`) | 31.846 M cycles | 31.514 M | -0.332 M (-1.04%), -0.17 ms per piece |
| 2 x 2048 piece (the live run at T=2048) | 35.123 M | 34.791 M | -0.332 M (-0.95%), -0.18 ms per piece |
| 16 x 1024 (no split) | 140.733 M | 137.210 M | -3.523 M (-2.50%), -1.84 ms |

**Expected effect per 32 x 1024 tokens:**
- about -1.4 ms with the live run's 4 x 1024 pieces (8 pieces x 0.173 ms), and about the same
  with its 2 x 2048 pieces (8 x 0.18 ms, at that shape's calibration of 0.542 ms per M cycles);
- about -3.7 ms with unsplit 16 x 1024 micro-batches (2 x 1.84 ms).
The measured head time in `docs/TPU_NOTES.md` (5.8 ms per 32 x 1024) agrees with the cost
model's reference head (2 x 5.72 M cycles x 0.521 = 5.96 ms), so the model's head estimates
are trustworthy at this scale. Peak memory is unchanged at all three shapes (670, 673 and
1,804 MB).

Everything else in my area costs less than 1% of a step, and is listed below with what I tried.

## 1. FAST_CE: the cross-entropy head with a hand-written backward pass

### What the reference does on the chip

These are the optimised TPU programs (`wf/speed_io_work/tpuc/`), cycles per micro-batch.

**One chunk** (n = 4,096 rows <= `loss_chunk` = 8,192: every training piece of the live run, at
both lengths). XLA removes the checkpointed recomputation of the chunk (one chunk,
`prevent_cse=False`):
- Forward: the logit product writes the float32 logits `z` (134 MB), with the row max in its
  epilogue (97k cycles).
- Forward: a second pass over `z` recomputes `tanh` and computes `exp`, the row sum and the target
  logit (363k cycles). It is bound by the transcendental unit, not by memory.
- Backward: two products, `dz^T hp` and `dz E`. Each one reads `z` and recomputes the softmax and
  softcap derivative in its operand fusion (513k).
- Head total: 0.977 M cycles of the piece's 31.85 M.

**Two chunks** (16 x 1024 at the default `loss_chunk`). The checkpointed scan body is recomputed
in the backward pass: the product, the max and the exp-sum pass all run a second time. The head
costs 5.72 M cycles of 140.7 M.

### The change

`chunked_ce` calls `_fast_ce(cap, n_chunks)`, a `jax.custom_vjp`, when `FAST_CE` is on and two
conditions hold: `0 < softcap <= 60`, and the float32 logit residual (`4 n V` bytes) is at most
`FAST_CE_MAX_RESIDUAL` (2 GiB, i.e. 65,536 rows at V = 8,192). Otherwise the reference path
runs. The reference keeps one chunk of logits live at a time; with the switch on, all chunks
are kept. So `test_model_loss.py::test_chunked_ce_memory_stays_per_chunk` describes the
switch-off path only (it runs with the default). In the training programs the compiler's peak
memory does not change (below).

**Forward, per chunk.** The same product and softcap as the reference, bit for bit:
- `z = hp e_out^T`, `t = tanh(z / cap)`, `zc = cap t`, and the target `zc[y]` by compare/select.
- Then `ce = log(sum_v exp(zc)) - zc[y]` **without the row maximum**. The softcap bounds
  `|zc| <= 15`, so `exp(zc)` lies in [3e-7, 3.3e6] and the sum over 8,192 entries stays below
  3e10: no overflow, no underflow, and the same relative precision as the shifted sum.
- Without the max, the log-sum-exp needs one pass instead of two, and XLA fuses it, with the
  target, into the epilogue of the logit product.

**Residuals.** `t` (float32 `[n, V]`) and the row sums `s`. XLA keeps the product's output `z`
itself and re-applies the `tanh` where it is needed.

**Backward.**
- `dzc = where(v = y, -g, 0) + (g / s) exp(zc)`, which is the form autodiff of the reference
  gives.
- `dz = dzc (1 - t)(1 + t)`.
- `d e_out = dz^T hp` and `d hp = dz e_out`, the same `dot_general`s autodiff emits, with
  float32 accumulation and a cast to the operand dtype.
- No product and no reduction over the vocabulary is repeated.

With more than one chunk, `d e_out` is summed over the chunks in float32 and cast once. The
reference's transposed scan rounds the running sum to bfloat16 after every chunk. With one chunk
(every training piece) the two are the same operation.

### On the TPU (compile-only, XLA's cost model)

**4 x 1024 piece**, head only:
- forward 97k + 363k -> 184k (one fused product);
- backward 513k -> 457k;
- head total 0.977 -> 0.641 M cycles.

**2 x 2048 piece**: the same 4,096-row head, 0.977 -> 0.641 M cycles.

**16 x 1024**: head 5.72 -> 2.34 M cycles (forward 0.67 M, backward 1.57 M).

**Peak memory** (compiler): unchanged, 670 MB at 4 x 1024, 673 MB at 2 x 2048 and 1,804 MB at
16 x 1024. The kept
logits (134 MB per 4,096 rows) are what the reference keeps too when it has one chunk. With two
chunks they sit below the peak of the loop scan.

### Ledger, before and after

| shape | glue pass | materialised | matrix products |
|---|---|---|---|
| 16 x 1024 | 87,465 -> 85,620 MB (-1,846, -2.1%) | 49,104 -> 46,663 MB (-2,441, -5.0%) | 17,939.6 -> 17,802.2 GFLOP (-137.4) |
| 8 x 2048 | 89,086 -> 87,240 MB (-1,846, -2.1%) | 49,517 -> 47,076 MB (-2,441, -4.9%) | same |

By class at 16 x 1024:
- reductions -1,611 pass / -2,148 mat (the max and the second exp-sum of the recomputed chunks);
- elementwise -3,222 pass;
- materialised broadcasts -277;
- one logit product fewer per chunk (-537 MB of product output).

By source:
- `head.py chunked_ce.<locals>.body`: -4,060 MB pass, -3,507 mat;
- `_fast_ce.chunk_fwd`: +1,611 / +1,074;
- `chunked_ce`, where the backward pass is charged: +587 / +512;
- `layers.py softcap`: mat -537.

The ledger traces 16 x 1024 with two loss chunks, so it sees the removed recomputation. At the
live run's 4 x 1024 pieces, XLA already removes it in the reference, and only the second pass
over the logits goes away. That is the -1.04% the compiler shows.

### Exactness

Tests are in `tests/model/test_io_speed.py`. Tolerances are relative to the largest entry of
each array; the measured differences are given with each.

**`test_fast_ce_has_the_values_and_gradients_of_the_reference[float64|float32 x chunk 32|80|1000]`**
compares `chunked_ce` values and the gradients of `h`, `proj` and `out`, with one or three
chunks.
- float64: tolerance 1e-13 (measured 3e-16).
- float32: tolerance 1e-6 (measured 2e-7).
- It also checks that the fast program is a custom VJP without `reduce_max`.

**`test_fast_ce_in_bfloat16_differs_by_rounding_steps_only[32|80]`**:
- The cross-entropy is within 1e-6 (measured 1.2e-7).
- The `h` and `proj` gradients agree to one bfloat16 rounding step on at most 1% of entries
  (measured: bit for bit).
- `out` with one chunk: the same bound (0.02% of entries one step apart).
- `out` with three chunks: at most 3 steps at the scale of its largest entry (measured 3.8e-3).
  This comes from the reference's bfloat16 running sum.

**`test_fast_ce_falls_back_where_its_assumptions_fail`**: with no softcap, or with a residual
above the limit, the reference path runs.

**`test_fast_ce_unshifted_sum_is_safe_at_the_softcap`**: every logit is saturated at ±15. Values
and gradients are finite and within 1e-5 of the reference.

**`test_fast_ce_model_loss_and_gradients_match[bfloat16|float32 x loss_chunk 128|64]`** (slow)
runs `value_and_grad(loss_fn)` on the tiny model with the live run's features: splash in
interpret mode, bf16, remat, sink, tape in and out, registers, value embeddings,
`table_grad=take`, clocks and confidence off. It checks the loss, every metric and all 59
gradient leaves.
- bfloat16, one chunk: tolerance 1e-6. Measured: the loss and 58 of 59 leaves bit for bit; the
  unembedding 1.6e-7 (0.01% of its entries one step apart).
- bfloat16, two chunks: the same, except the unembedding at 2 steps (measured 2.6e-3).
- float32: tolerance 5e-6. Measured: the loss bit for bit, the worst leaf 1.0e-6.

Run:

    taskset -c 2,3 nice -n 10 <venv>/bin/python -m pytest tests/model/test_io_speed.py -o addopts="-p no:cacheprovider" -q -x

All 14 pass (10 fast tests in 26 s; with the 4 slow ones, 185 s). The existing
`tests/model/test_model_loss.py` passes with the switch off (5 tests). Its four
value-and-gradient tests also pass with the switch on, against their own independent
references: a dense autodiff reference of the whole loss, the unchunked loss, the
independent numpy scores, and the bit-for-bit zero-mass identities
(`wf/speed_io_work/loss_tests_fast.py`).

## 2. My area on the chip

M cycles per micro-batch, from the TPU cost model, split by the source stack of each
instruction (`wf/speed_io_work/tpuc/areas_io.py`):

| part | 4 x 1024 ref | 4 x 1024 FAST_CE | 16 x 1024 ref | 16 x 1024 FAST_CE |
|---|---|---|---|---|
| cross-entropy head (`chunked_ce`, incl. `h @ P`) | 0.977 | **0.641** | 5.717 | **2.339** |
| tape: candidate tokens (`big_lookup` weighted sum) | 0.306 | 0.306 | 1.185 | 1.185 |
| tape: candidate features | 0.099 | 0.099 | 0.232 | 0.232 |
| tape: masses (`log_mass`, `masses`, `mix_ce`) | 0.168 | 0.170 | 0.788 | 0.786 |
| registers | 0.050 | 0.050 | 0.244 | 0.244 |
| value embeddings | 0.043 | 0.043 | 0.397 | 0.397 |
| token embedding and its norm | 0.077 | 0.077 | 0.356 | 0.356 |
| loss, weights, metrics | 0.010 | 0.012 | 0.019 | 0.019 |
| **my area** | **1.731** | **1.399** | **8.937** | **5.557** |
| whole micro-batch | 31.846 | 31.514 | 140.733 | 137.210 |

My area is 5.4% of a live piece (6.4% at 16 x 1024). After the head, the largest item is the
tape's candidate tokens at 1.0% of a piece (0.8% at 16 x 1024). The rest are below 0.6% each.
(The 16 x 1024 reference is the speed_mlp_block engineer's `base_trainer.hlo`. My own reference
compile at 4 x 1024 matches theirs to the cycle, 31.846 M.)

## 3. Tried and dropped

**FAST_TAKE: `jnp.take(..., mode="clip")` in `tables._gather_sum`.** The default `"fill"` mode
adds an in-bounds mask, and a select with a NaN row, to every gather and to its transpose.
`big_lookup` clips the indices first, so clip mode is bit for bit the same; this was tested with
tolerance zero (12 unit tests over the three `table_grad` modes, K = 1 and 16, bf16 and f32,
plus the tiny live model in bf16 and f32).
- Ledger: -1 MB.
- TPU cost model at 4 x 1024 with FAST_CE: 31.514 -> 31.412 M cycles (-0.102 M, -0.32%).
  - The tape's candidate tokens fell from 0.306 to 0.162 M; their weighted-sum fusion went from
    210.6k to 78.6k cycles.
  - The value embeddings rose from 0.043 to 0.099 M: their gather and transpose became a
    separate layout copy (86k).
- Clip mode only where K > 1 (the tape) would give about -0.15 M (0.5%); this is inferred from
  the two compiles, not compiled itself.
- Below the 1% bar, so dropped. The version is kept at `wf/speed_io_work/tables_fast_take.py`.
  It is exact, so it could be adopted later.

**Variants of FAST_CE's residual and backward** (the head alone, `tpuc/tpu_head.py`, 4,096
rows, M cycles; the whole-program numbers above are a little lower):

| variant | head | verdict |
|---|---|---|
| reference | 1.032 | |
| autodiff's derivative chain (`cap`, `a + a t`, `/ cap`) | 0.742 | |
| `(1 - t)(1 + t)` | 0.710 | kept |
| barrier on `t`, so `t` is kept rather than `z` | 0.739 | no gain |
| `t` and `exp(zc)` both kept | 1.072 | twice the bytes |
| `exp(zc)(1 - t^2)` kept | 0.744 | XLA recomputes it from `z` anyway |
| `dz` built once (barrier) for both products | 0.760 | the separate pass costs more than the duplicated operand fusion |

**A bfloat16 `dz`, or bfloat16 kept logits.** On the TPU, the backward products already round
their float32 `dz` operand to bfloat16 (default precision is a single bf16 pass). On CPU it is a
bfloat16-level change of the gradients, so it is not exact. Building `dz` once does not reduce
the bytes either: one read of the f32 logits, one write and two reads, against two reads now.
Not done.

**The candidate weighted sum as 16 separate per-slot gathers.** It is no better than clip mode
(0.190 M cycles each in a standalone compile).

**Log-masses as `ht @ table^T`, then a selection by feature value.** This would remove the 65-wide
`[n, K, 65]` rows: at most about 0.1 M cycles per piece (0.3%). It would also change which
operand the TPU's default-precision float32 products round to bfloat16 (`ht` instead of the table
rows), so it is not float rounding on the chip. Not implemented.

**Registers as gathers instead of the multi-hot product** (0.09 M per piece in total): no gain
expected. **The value-embedding transpose** (0.04 M per piece): too small. Not implemented.

**A Pallas kernel for the whole cross-entropy.** The kernel would keep the logits in VMEM, with no
`[n, V]` traffic. Bound by its four products, I estimate about 0.35 M cycles per piece against
0.64 M with FAST_CE. I did not attempt it, for two reasons:
- the compile-only cost model gives Mosaic kernels 0 cycles, so there is no evidence to be had
  here;
- the backward products inside the kernel would take a bfloat16 `dz`, which is not exact on CPU.
It is the next step if the head still matters after FAST_CE.

## 4. Files

All in `wf/speed_io_work/`:
- `base_1024.json`, `fastce_1024.json`, `base_2048.json`, `fastce_2048.json` (ledgers);
- `tpuc/full_*.hlo` (optimised TPU programs of the whole micro-batch);
- `tpuc/head_*.hlo` (the head alone);
- `tpuc/lk_*.hlo` (the lookups);
- `tpuc/myarea.py` (per-source cycles, using the speed_mlp_block engineer's `hlo_bytes.py`);
- `tpuc/tpu_head.py`, `tpuc/tpu_lookup.py`, `tpuc/tpu_compile.py` (compile-only drivers);
- `check_ce.py`, `measure_model.py` (the measured differences quoted above).

Run the compile-only drivers with `ALLOW_MULTIPLE_LIBTPU_LOAD=1`; otherwise two of them
cannot run at once, because of libtpu's lockfile.
