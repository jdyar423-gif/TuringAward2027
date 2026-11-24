# speed_attention: the attention sub-layer outside its matrix products

Worktree `/home/user/turingaward2027/.claude/worktrees/wf_3d459e66-645-5` (HEAD b05bfe7), patch
`wf/speed_attention.patch` (`bes/model/attention_splash.py` +253 lines, new
`tests/model/test_splash_io.py`). `bes/model/attention.py` is unchanged: every rewrite of the
projections I tried lost on the compiled program (section 4). Nothing committed, no config field,
parameter tree, optimiser state and values unchanged.

## 1. Result: one switch, bit-for-bit exact

`bes.model.attention_splash.FAST_SPLASH_IO` (default False, read at trace time).

The splash kernels of jax 0.11.1/0.11.2 (the two files differ only in type-checker comments)
waste most of their output bandwidth:

* the forward kernel declares its m, l and o accumulators as **outputs** (`TODO(sharadmv): convert
  m/l to be scratch`), so under `vmap` every example writes them to HBM (25 MB per call at
  16 x 1024), and writes the log-sum-exp **128-lane broadcast** as `f32[B, H, T, 128]` (50 MB per
  call) although one value per row is needed;
* the backward wrapper then slices lane 0 out (`copy`, a 50 MB read per call) and broadcasts it
  again to `[B, H, 8, T]`;
* the fused backward kernel declares its dq/dk/dv accumulators as outputs too (25 MB per call).

With the switch, `attend_major` calls **the library's own kernel bodies, unchanged**
(`flash_attention_kernel`, `_flash_attention_dkv_kernel`) through pallas_call set-ups in
`attention_splash.py` (`_io_forward`, `_io_dkv`, `_io_attention`) that put the accumulators in VMEM
scratch, and write the log-sum-exp once as `f32[B, H, 8, T]`: the `(bq, 128)` tile the library
computes is transposed in the kernel (`lse_scratch[...].T[:8, :]`), which is exactly the layout
the backward kernel reads. Same masks, tiles, mask value and arithmetic, so outputs and gradients
are identical bit for bit. The call sits inside a `jax.jit` like the library's (without it, XLA
fused the query prologue differently at 8 x 2048: 1.3 GB saved instead of 3.3 GB, 0.2 GB more peak
memory). The separate-backward-kernel case falls back to the library's backward on the `[H, T]`
log-sum-exp.

### Numbers (per micro-batch)

The ledger cannot show this saving: it books kernel outputs outside its glue total and counts a
slice by its output (0.4 MB), not by the 50 MB it reads. So I also measured the **real program
compiled for the chip**: `value_and_grad(loss_fn)` of `configs/final.py` (`table_grad=take`),
compiled for TPU v6e:1x1 on this CPU host with libtpu 0.0.50 (the chip's version; compile only, no
hardware). Then I summed the HBM bytes read and written by every scheduled instruction (tiled
layouts, VMEM-resident buffers excluded, async copies and slices counted on their HBM side, loop
trip counts applied). Tools are in section 5.

| | baseline | FAST_SPLASH_IO | change |
|---|---|---|---|
| **16 x 1024, compiled HBM traffic** | 70,000 MB | 66,024 MB | **-3,976 MB (-5.7%)** |
| ... forward / recompute / backward | 15,731 / 14,405 / 39,864 | 14,171 / 12,442 / 39,411 | -1,560 / -1,963 / -453 |
| ... by kind | | | kernels -2,517, copy -805 (the 16 lse slices), copy-start -805, broadcast -50, loop fusions +201 |
| **8 x 2048, compiled HBM traffic** | 71,932 MB | 68,660 MB | **-3,272 MB (-4.5%)** |
| peak memory (compiler) | 1.804 GB / 1.804 GB | 1.804 GB / 1.804 GB | 0 (identical bytes at 1024) |
| ledger 16 x 1024: "pallas kernel" outputs | 4,832 MB | 2,114 MB | -2,718 MB |
| ledger 16 x 1024: glue | 87,465 MB | 87,408 MB | -57 MB (-0.1%) |
| ledger 8 x 2048: "pallas kernel" outputs | 4,631 MB | 2,517 MB | -2,114 MB |
| ledger 8 x 2048: glue | 89,086 MB | 89,029 MB | -57 MB (-0.1%) |
| matrix-product FLOPs | 17.94 TFLOP | 17.94 TFLOP | 0 |

The compiled baseline also says where a step's bytes go: about 37 of the 70 GB per 16 x 1024
micro-batch are the attention sub-layer (its five projections and their transposes 20.3 GB, the
splash kernels 8.8 GB, the rest glue). The compiled program is strongly memory-bound: 70 GB is
43 ms at 1.64 TB/s, against 73.3 ms measured per 16-row micro-batch.

### Expected effect: about -2.5 ms per 32 x 1024 micro-batch (range -1 to -6 ms)

Derivation, at twice the 16 x 1024 bytes and the chip's measured pass speed (a pass over a 100 MB
f32 activation costs about one d x d product, 0.08 ms, i.e. about 1.25 TB/s):

* stand-alone ops removed (the lse slice copies and broadcasts, net of the new small ones):
  1.3 GB, fully exposed, about **1.0 ms**;
* the kernels' unwritten outputs (5.0 GB) and the async copies that go with them (1.6 GB):
  between 0 and 5.3 ms. In this kernel the output DMA is per grid step (about 1 MB against roughly
  3 us of VPU-bound compute per step), so most of it is probably hidden; I count a quarter of it.

So about -2.5 ms of 162 ms (1.5%), at T=2048 slightly less (-3.3 GB instead of -4.0 GB per
micro-batch).

### A combination to measure: FAST_SPLASH_IO with `model.remat_keep=kernel`

`remat_keep=kernel` (keep the kernels' outputs across the rematerialised loop step, so the
backward pass does not rerun the 16 forward kernels) lost 1.8% on the chip on 23 Nov. The compiled
program explains why: with the library's kernels the kept outputs are 100 MB per block application
(including the 128-lane lse and the m/l/o accumulators), and HBM traffic rises from 70.0 to
**81.3 GB** (peak 3.28 GB). With the switch the kept outputs are 28 MB, and traffic is **71.6 GB**
(+1.6 GB over the baseline and +5.6 GB over the switch alone; peak 2.18 GB), while 16
forward-kernel runs per micro-batch are skipped. From the 23 Nov numbers (+2.9 ms measured against
about +22.6 GB at 32 rows, 14-18 ms of traffic), a forward kernel is worth roughly 0.6-0.9 ms at
32 x 1024. On paper the
combination may now win, but the same reasoning predicted a gain on 23 Nov and it lost, so only
the chip can decide. It is bit-for-bit exact as well (test below). Please measure three plans:
switch off; `FAST_SPLASH_IO=1`; `FAST_SPLASH_IO=1` with `--override model.remat_keep=kernel`.

## 2. Tests (tolerance: none needed, every comparison is bit-for-bit)

`tests/model/test_splash_io.py`, 10 tests, about 3-4 min on two cores (interpret mode):

* `test_kernel_output_and_gradients_are_bitwise_equal` (6 cases): kernel output and dq, dk, dv,
  switch off and on, `assert_array_equal`. Covers the backward kernel with and without its dq
  accumulator (`block_kv == block_kv_compute` or not, as the shipped 1024/1024/512), a padded length
  (200), several query tiles (384 = 3 x 128), float32 and bfloat16, packed segments, and separate
  dq/dkv kernels.
* `test_the_kernels_write_only_their_results`: in the gradient jaxpr, the forward kernel has 2
  outputs (`o`, `f32[B,H,8,T]`) instead of 5, the backward kernel 3 instead of 5.
* `test_model_loss_and_every_gradient_are_bitwise_equal[bfloat16|float32]`: `loss_fn` loss,
  sink-mass metric and **every gradient leaf** of a small model, bit for bit. The model runs the
  live code path: splash, remat, sink, confidence, tape in and out, registers, value embeddings,
  `table_grad=take`, packed segments, length 40 padded to 128.
* `test_keeping_the_kernel_outputs_with_the_switch_changes_no_value`: switch on with
  `remat_keep=kernel` and with `""`, against the live program (switch off, `""`). Loss and all
  gradients are bit for bit equal, and the kept-outputs program has fewer forward kernels.

Existing tests with the switch off: `tests/model/test_splash.py` and
`tests/model/test_attention_major.py` pass, 34 fast and 5 slow.

Command, from the root of the checkout:
`taskset -c 2,3 nice -n 10 <venv>/bin/python -m pytest tests/model/test_splash_io.py -o addopts="-p no:cacheprovider" -q -x`

## 3. What the compiled program showed about the ledger's candidates in my area

* `attention.py:377/380` (L2 norm, q gain), the top ledger lines: in the compiled program the
  rotation, the sum of squares and the bf16 cast are fused into the second projection's output
  fusion. Per q or k and phase the cost is one 50 MB f32 write of `x @ W_a`, its read, and the
  25 MB bf16 write of the kernel operand. The q-gain gradient and the norm backward's row
  reduction are **already one multi-output fusion** over `dq`.
* "Every backward product has a float32 operand": on TPU at default precision that operand is fed
  to the MXU as bf16. XLA itself stores a stacked cotangent in bf16 (variant 4 below). So it is not
  a 2x pass in practice, and an explicit bf16 cast would gain nothing.
* "Transposes at a product": none survive in the compiled HLO; they are layouts.
* The f32 rotated `r` of q and k (50 MB each) is recomputed in the backward phase, kept in VMEM
  or evicted, and read by four transposed products. The five dx partials are written in bf16 and
  summed (about 200 MB per block application). Variants 3 and 4 below try to remove this and lose.

## 4. Tried and dropped (all compiled for v6e at 16 x 1024 against the 70.0 GB baseline)

| variant | idea | values | FLOPs | compiled HBM | verdict |
|---|---|---|---|---|---|
| FAST_ROPE_ROLL | q/k lane layout `[half1, reserved, half2, reserved]` (64 + 64), rotation `u*c + roll64(u)*s` of ONE product (slice + concatenate) | bitwise q, k, v (CPU) | -2.47 TFLOP (-13.8%) | **96.0 GB (+26.0)** | XLA writes each 64-lane half as a padded 128-lane f32 tensor and pads them back |
| FAST_ROPE_FLIP | layout `[half1, reserved, reversed half2]`, rotation `u*c + reverse(u)*s` | f32 rounding (norm sum order) | -2.47 TFLOP | **85.0 GB (+15.0)** | `reverse` is never fused: 106 stand-alone reverse ops (8 GB) in forward, recomputation and backward |
| FAST_QK_PAIR | each of q, k as one product with both halves in 256 lanes (128-aligned slices) | bitwise forward | 0 | **81.8 GB (+11.8)** | XLA writes the 256-lane f32 output and reads it back instead of fusing the slices |
| FAST_PROJ_VJP | custom VJP: the transposed products of all five projections as two products over the cotangents stacked on the head axis | dx rounding changes at bf16 level | 0 | **71.8 GB (+1.8)**; backward -0.1, forward +2.5 | XLA builds the stack with five in-place bf16 updates; the custom VJP costs the forward its fusions |
| segment-id broadcasts | kernel inputs `i32[B,T,128]` broadcast in every block application (XLA does not hoist them) | | | about 0.2 GB per micro-batch (0.3%) | below 1%; hoisting needs `model.make_ctx` (not my file) |
| value-embedding gate/add | `attention.py:388/389` | | | 0.38 GB (0.5%) | below 1% |
| T=2048 dq partial sum | the fused backward writes 2 bf16 dq partials summed outside | | | 0.8-1.2 GB per 8 x 2048 (1.1-1.7%) | fusing the sum into the five consumers makes each read both partials; accumulating in VMEM across kv blocks changes the rounding; a full-length backward kv tile computes the masked half (+33% kernel work) |

The pattern: XLA TPU's own fusion of the head-major projections (rotation as two products, norm
and cast fused into the second product's output) is close to a local optimum. Every formulation
that moves the half-swap off the MXU, or adds a boundary, loses. The remaining large item, about
20 GB per micro-batch in the projections and their transposes, would need a fused Pallas prologue
kernel, which cannot be tuned without the chip.

## 5. How the numbers were made (reusable)

All in `wf/speed_attention_work/` (my working files, HLO dumps of every variant, `*_hlo.json`).

* `libtpu/libtpu.so`: extracted from `scratchpad/dl_libtpu0050/libtpu-0.0.50-cp313.whl`, whose
  sha256 equals PyPI's published digest (`64a959441c0a...378671`).
  `tools/tpu_env.py` sets `TPU_LIBRARY_PATH`, the v6e-1 variables and
  `ALLOW_MULTIPLE_LIBTPU_LOAD=1`. Another engineer's compile-only process was holding
  `/tmp/libtpu_lockfile`; compile-only opens no device, so this lets both run.
* `tools/tpu_compile.py --repo CHECKOUT --out F.hlo [--rows 16 --seq 1024] [--set module.NAME=1]
  [--override model.field=value]` compiles the program. It takes 70 s to 4 min on cores 2-3 and
  about 1.7 GB RSS, never touches a TPU, and prints the compiler's peak memory.
* `tools/hlo_bytes.py F.hlo [--diff G_hlo.json] [--grep TEXT --list]` gives the HBM bytes per
  scheduled instruction, by kind and by `bes/` source line (from the HLO stack frames), and by
  phase. `tools/phase_sum.py`, `tools/phase_diff.py` and `tools/hlo_show.py` (fusion contents,
  operands and memory spaces) build on it.
* `tools/probe_transpose.py` checks that Mosaic lowers the in-kernel transpose and VMEM scratch
  under vmap for v6e, and the values in interpret mode.

## 6. Notes for the merge and the chip

* Only `bes/model/attention_splash.py` and a new test file are touched, so the patch does not
  overlap `speed_mlp_block.patch`.
* To flip the switch: set `bes.model.attention_splash.FAST_SPLASH_IO = True` before tracing.
  The program changes, so the persistent compilation cache misses once.
* Kernel names in a profile: `splash_mha_fwd_io` and `splash_mha_dkv_io` (instead of
  `splash_mha_fwd_segmented_residuals` and `splash_mha_dkv_segmented_no_residuals`).
* The fork uses names of the jax splash module `splash_attention_kernel`, some of them private:
  `flash_attention_kernel`, `_flash_attention_dkv_kernel`, `_next_nonzero`, `_div`,
  `_splash_attention_bwd` (only for separate backward kernels), `NUM_LANES` and `NUM_SUBLANES`.
  The file is identical in jax 0.11.1 and 0.11.2 apart from comments. A jax upgrade must re-check
  them; `tests/model/test_splash_io.py` would catch a change.
