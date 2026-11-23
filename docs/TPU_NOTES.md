# TPU notes (one TPU v6e chip)

What the compile-only checks in this container established, which settings matter for numerics
on the TPU, and what must be measured in the first 15 minutes on the real chip. Everything here
was produced without TPU hardware: libtpu's compiler driven from a CPU host (see "Compile-only
check"). Speed, MFU and real memory use are **not** measured.

## Target

- One TPU v6e chip (Colab `v6e-1`): 32 GiB HBM, bf16 matrix units. The plan's gate keeps the
  compiler's peak estimate below **28 GiB**.
- Our matmul dimensions are multiples of 256 except the attention head (128):
  `d_model = 768`, `d_ffn = 3072`, `head_rank = 512`, `V = 8192`, `H * head_dim = 768`.

## Precision settings

1. **bfloat16 activations, float32 everything else.** `configs/final.py` sets
   `act_dtype="bfloat16"`: matmul operands are cast to bf16 and accumulate in float32
   (`preferred_element_type`); normalisation statistics, lane mixing, rotary angles, logits,
   log-sum-exp and the loss are float32. Parameters stay float32 master copies.
2. **Float32 runs on the TPU need an explicit matmul precision.** The model passes
   `preferred_element_type` but no `precision`, and on a TPU the default precision of a float32
   matmul is a single bf16 pass. Any float32 run on the TPU (a parity check against the CPU
   tests, a `safe` configuration) must set
   `jax.config.update("jax_default_matmul_precision", "highest")` (or the environment variable
   `JAX_DEFAULT_MATMUL_PRECISION=highest`) before tracing; otherwise expect errors at the bf16
   level (around 1e-3 relative; not measured here), far above the 1e-4 tolerances of the CPU
   tests.
3. **Keep the element-wise float32 code element-wise.** The rotary angles
   (`rope_clocks.angles`), the lane combinations (`controller.combine/update`) and the head's
   lane sum are written as float32 products on purpose; rewriting them as einsums would route
   them through the matrix unit at bf16 precision (positions above 256 would lose integer
   precision).
4. **Splash vs dense numerics.** Inside the splash kernel `q.k` accumulates in float32, the
   softmax statistics are float32 and `p @ v` uses float32 `p` with `v` cast to float32; the
   output is returned in bf16. The dense path casts `p` to bf16 before `p @ v`. In bf16 the two
   paths therefore differ at the bf16 rounding level (the CPU test of bf16 inputs allows 0.05);
   in float32 interpret mode they agree within the test tolerances (1e-5 forward, 2e-4
   gradients).
5. **Masking.** The kernel masks with `-0.7 * float32_max`, the dense path with `float32_min`;
   both give exactly zero probability. Every query sees itself (causal, same segment), so no
   softmax row is empty; padded positions get their own segment id.
6. **No softmax scale.** The kernel computes the plain dot product; the learned gain lives in
   `q` (initialised to `sqrt(126) ~ 11.2`). The sink is the BOS key whose reserved dim carries
   `sink_logit`; the kernel's own `sinks` argument is a value-less logit and is not used.

## Splash attention as shipped in jax 0.11.1

Read from `jax/experimental/pallas/ops/tpu/splash_attention/` of the installed version:

| feature | present | used |
|---|---|---|
| `make_splash_mha_single_device(mask, block_sizes, interpret)` | yes | yes, one kernel per `(T, H, blocks)`, vmapped over the batch |
| static masks `CausalMask`, `MultiHeadMask`, `LocalMask`, `NumpyMask`, `FullMask` | yes | `CausalMask` (computed in-kernel, fully masked blocks skipped) |
| `SegmentIds(q, kv)` per example | yes | yes (packed segments) |
| `sinks` (one value-less logit per head) | yes | no (our sink is an ordinary key) |
| `attn_logits_soft_cap` | yes | no |
| `interpret=True` | yes, runs on CPU for tiny shapes, forward and backward | CPU tests only |
| `BlockSizes` with separate or fused (`use_fused_bwd_kernel`) backward | yes | separate by default, fused selectable |
| custom VJP (residual: log-sum-exp) | yes | yes |

Constraints: `block_kv_compute` a multiple of 128, `block_kv` a multiple of `block_kv_compute`,
the sequence a multiple of `block_q` and `block_kv`. `bes.model.attention_splash` pads the
length to a multiple of 128 (extra segment id) and shrinks each tile to the largest multiple of
128 that divides the padded length.

**Block sizes.** Defaults (`ModelCfg.splash_block_q / _kv / _kv_compute`) are 1024 / 1024 / 512
with the fused backward kernel (`splash_fused_bwd=True`), measured on TPU v6e at 32 x 1024
(section "Where a step spends its time"): against 512 / 512 / 512 with two backward kernels
(the first defaults, never measured before the run) the attention sub-layer takes 5.87 ms
instead of 7.04 ms per block application and a training step is 10% faster. 256 tiles are 2.2
times slower than 512 for the kernel alone. Override with `--override model.splash_block_q=...`
to sweep (`scripts/plan_sweep.py` times a plan with any override).

## Known pitfalls

1. **Import name of the splash module** (fixed). `bes/model/attention.py` imports
   `bes.model.attention_splash` lazily when `attn_impl="splash"`; the old alias
   (`attention_splash.register_alias()`) is no longer needed.
2. **Interpret mode on the compile host.** `splash_interpret="auto"` means "interpret unless the
   default backend is a TPU". On a CPU host that compiles for the TPU topology the default
   backend is the CPU, so the compile-only tool forces `splash_interpret="false"`; on the TPU VM
   `"auto"` is right.
3. **The dense `xla` attention materialises `[B, H, T, T]` float32 logits and probabilities** per
   block and loop. It compiles and fits at the training shapes with remat (table below) but is
   meant for tests; train with splash.
4. **Remat.** `configs/final.py`, `configs/strict.py` and `configs/no_library.py` set
   `remat=True`. With `remat=False` (the `ModelCfg` default) and the full micro-batch neither
   training shape compiles: the compiler needs 32.93 GB of temporaries (table below); half the
   micro-batch fits (addendum). Tune the policy on the chip.
5. **Decoding reads the whole cache every step.** The cache is
   `[n_loops, n_blocks, B, H, Tmax, head_dim]` for keys and values in bf16:
   `2 * 8 * 2 * 6 * 128 * 2 bytes = 48 KiB` per row and slot, i.e. 0.75 GiB for `B = 8`,
   `Tmax = 2048`. Static shapes mean every step attends over all `Tmax` slots (empty ones are
   masked). Choose `Tmax` per request bucket (the host loops round to multiples of 128).
6. **Float32 master weights are cast inside every decoding step.** `block_params` casts the
   float32 matrices to bf16 inside the jitted step, so each step reads 4 bytes per weight. For
   generation it is cheaper to pass a parameter tree whose matrices are already bf16 (the casts
   become no-ops); benchmark before relying on it.
7. **Donated caches.** `bes.model.generate.jitted(cfg)` donates the cache argument of
   `prefill` and `decode_step`; never reuse a cache object after passing it in (use
   `jitted(cfg, donate=False)` in tests that branch from one cache).
8. **`cost_analysis()` undercounts.** It counts the body of a `while` loop (the loop scan, the
   micro-batch scan, the loss chunks) once and does not count Mosaic kernels (table below).
   Compute MFU from measured step time and an analytic FLOP count.
9. **Building a splash kernel inside a trace.** The kernel factory turns its mask information
   into `jnp` arrays; called inside a `jit`/`scan` trace those are tracers. A kernel cached from
   inside one trace and reused by another program raises `UnexpectedTracerError`.
   `attention_splash.make_kernel` therefore builds kernels under
   `jax.ensure_compile_time_eval()` (regression test in `tests/model/test_splash.py`).
10. **Compile time.** One program of the final model takes minutes on two CPU cores (table). The
   group learning rates are run-time inputs of the compiled step, so the trainer reuses the
   steps the start-up benchmark compiled in the same process (`train: set-up (compile 2 shapes,
   2 reused from the benchmark, ...)`), and `scripts/run_24h.py` enables the persistent
   compilation cache under `RUN_DIR/jax_cache` (guarded: an option this JAX does not know, or a
   backend that does not use the cache, is logged and the run goes on without it;
   `manifest.json` `jax_cache` says which). Whether a restart actually hits the cache on the TPU
   VM is not known yet: check the set-up time of a resumed run.

## Compile-only check

`python -m bes.train.aot` (`bes/train/aot.py`) lowers and compiles for the topology returned by
`jax.experimental.topologies.get_topology_desc(platform="tpu", topology_name="v6e:1x1",
chips_per_host_bounds=(1, 1, 1), num_slices=1)` with the installed libtpu (device kind
`TPU v6 lite`). The call prints harmless warnings about the accelerator type and worker
hostnames on a host without TPU metadata; the tool sets `TPU_ACCELERATOR_TYPE=v6e-1`,
`TPU_WORKER_HOSTNAMES=localhost`, `TPU_SKIP_MDS_QUERY=1` if they are unset.

Measured in this container (jax 0.11.1, libtpu 0.0.46.3, two CPU cores, one program per
compile; the compiler reports 31.24 GB of available HBM for the target). "Peak" is
`memory_analysis().peak_memory_in_bytes`, the compiler's estimate, not a measurement on the
chip. `configs/final.py` (d=768, 2 blocks, 8 loops, bf16, splash) unless noted:

| program | attention | shape | remat | compile | peak | notes |
|---|---|---|---|---|---|---|
| train step: 8 micro-batches, MuonH update, donated state | splash | T=1024, B=32 | on | 240 s | 4.21 GiB | args 0.38, temp 5.06, aliased 0.33 GiB; 0 scatter; 8 Mosaic calls |
| same | splash | T=2048, B=16 | on | 223 s | 4.21 GiB | same |
| `value_and_grad(loss_fn)`, one micro-batch | splash | T=1024, B=32 | on | 198 s | 3.93 GiB | temp 4.76 GiB |
| same | splash | T=2048, B=16 | on | 171 s | 3.93 GiB | |
| same | splash | T=1024, B=32 | **off** | fails | - | `RESOURCE_EXHAUSTED: ... HLO temporaries (32.93G) exceeds available HBM (31.24G)` |
| same | splash | T=2048, B=16 | **off** | fails | - | same message |
| same | xla (dense) | T=1024, B=32 | on | 215 s | 5.33 GiB | |
| same | xla (dense) | T=2048, B=16 | on | 212 s | 7.13 GiB | |
| `decode_step` | cache | B=8, Tmax=2048 | - | 10 s | 1.66 GiB | the 0.75 GiB cache is aliased (updated in place) |
| `prefill`, one 256-token chunk | cache | B=1, Tmax=2048 | - | 14 s | 0.34 GiB | |

No optimised HLO above contains a scatter. Host memory of the compiling process stayed between
2.0 and 3.4 GB. After the kernel-cache fix (pitfall 9) the loss and the train step at
`T=1024, B=32` were compiled again in one process, the case the fix is about: same peaks
(3.93 and 4.21 GiB). Smoke configuration for comparison: loss T=128, B=8 in 68 s; train step
(A=2) with splash in 108 s.

Consequences:

- **`configs/final.py` enables remat** (`ModelCfg(remat=True, ...)`): without it neither
  training shape fits. With full remat of the loop body the step needs about 4.2 GiB of the
  31 GiB, so there is room for a cheaper policy (keeping some activations, or recomputing only
  every other loop) or larger micro-batches; that trade-off is for the benchmark.
- The splash path is the cheaper one in memory (3.93 vs 5.33 / 7.13 GiB) as expected; the dense
  fallback also fits.
- `cost_analysis()["flops"]` is not a FLOP count for these programs: it reports 4.2e12 for the
  splash micro-batch at both shapes (the Mosaic kernels are not counted), 5.06e12 for the dense
  model and 3.89e12 for the same dense model with `n_loops=1`, i.e. the body of the loop scan is
  counted once, not eight times.

## First 15 minutes on the real chip

1. `jax.devices()` lists one `TPU v6 lite`; record the jax, jaxlib and libtpu versions.
2. Run `python -m bes.train.aot --programs loss,train,decode` on the TPU VM: the compiler's
   peak estimate must agree with `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` after
   one real step (and stay below 28 GiB).
3. Step time and peak memory for both training shapes (`T=1024, B=32, A=8` and
   `T=2048, B=16, A=8`): splash vs dense, full remat vs cheaper remat policies that still fit
   (no remat does not fit), larger micro-batches, `table_grad` `"onehot"` vs `"take"`.
4. Splash block sizes: forward plus backward of one attention call at the real shapes for
   `block_q, block_kv` in {256, 512, 1024} and `block_kv_compute` in {256, 512}, fused backward
   on and off; then the full step with the best two settings.
5. Parity on the chip: one batch through `loss_fn` with `attn_impl` xla vs splash (bf16:
   |delta loss| below 1e-2); a float32 run with `highest` matmul precision against the CPU value.
6. Decoding: `decode_step` tokens per second for `B = 8`, `Tmax = 2048`, and prefill of a
   256-token chunk; one generation compared with the full-sequence forward at bf16 tolerance.
7. Determinism: the same batch twice gives the same loss bit for bit.
8. Host pipeline rate (exact layer, loader) against the measured step time.

## Where a step spends its time (measured on TPU v6e-1, 2026-11-23)

The first numbers from the real chip (Colab `TPU v6 lite`, jax 0.11.2, libtpu 0.0.50, 31.24 GiB
HBM) were three times under the plan's assumption: `configs/final.py` trained at 112-117k tokens
per second, `configs/v9.py` at 88-92k (the plan assumed 300-450k). What was ruled out, in the
order it was tried, each by a measurement on the idle chip:

| suspect | measurement | verdict |
|---|---|---|
| a fixed cost per micro-batch | `ms64` 110,855, `ms128` 104,576 tok/s against `ms32` 111,893 (`scripts/plan_sweep.py`) | no: larger is slower |
| sequence length | `T=2048` 105,912 against `T=1024` 111,893 | the quadratic part is about 6% |
| rematerialisation | without it (half micro-batch) 116,986 | 4.5% |
| the float32 softmax head | 5.8 ms per micro-batch of 289 ms | no |
| the optimiser (Polar Express, once per step) | 1.5 ms of a 2.34 s step | no |
| gradient accumulation | the scan of 8 micro-batches is 8.0 times one `value_and_grad` | no |
| candidate tape, registers, value embeddings, sink | each within 5% when removed (10% for the sink) | no |
| dense instead of flash attention | 303 ms against 289 ms | no |
| the loops | 1 loop instead of 8: 42.9 ms against 289 ms | **yes: 97% is the 16 block applications** |

Inside one block application at 32 x 1024 tokens (forward and backward with rematerialisation,
a scan of 8 steps of two applications; the first program timed in a process is 10 times slower
for a few calls and has to be discarded):

| part | ms | |
|---|---|---|
| the block | 15.4 | |
| attention sub-layer | 12.7 | |
| ... its kernel alone (512 tiles) | 3.35 | 2.8 with 1024 tiles or the fused backward kernel; dense: 7.0 |
| ... without the kernel | 5.97 | projections, rotation, normalisation, reserved dims |
| feed-forward sub-layer | 2.97 | 2.13 with the hidden layer stored in bfloat16 |
| the two feed-forward products alone | 1.52 | 814 TFLOP/s forward: the matrix units work |
| one `d x d` product | 0.23 | 0.08 forward; float32 accumulation costs nothing |
| `rmsnorm` | 0.23 | |

So the matrix products of a block take about 2.4 ms and the kernel 3.4 ms; **the other 9.6 ms
(62%) is glue**: slices and concatenations (the rotary halves, the reserved bias dims), transposes
between `[B, T, H, hd]` and the kernel's `[B, H, T, hd]`, and float32 tensors of the widest
shapes. Each of those is one or more passes over a 50-400 MB tensor at memory speed, and on this
chip a pass costs about as much as the `d x d` product that made the tensor.

What was changed, with the same values to float rounding (`tests/model/test_attention_major.py`;
loss 8.33353 for every variant on the chip), in tokens per second for one `value_and_grad` at
32 x 1024 (`configs/final.py`):

| implementation | ms per micro-batch | tok/s |
|---|---|---|
| reference | 289.3 | 113,264 |
| rotation as two products, reserved dims as zero columns and broadcast constants (no slice, no concatenation) | 215.0 | 152,379 |
| ... and operands built in the kernel's layout, `W_o` as one contraction over (head, dim) (no transpose) | 194.9 | 168,155 |
| ... and feed-forward pre-activations stored in bfloat16 | 181.8 | 180,248 |
| ... and lanes carried as a tuple | 179.3 | 182,779 |
| ... and tiles 1024 / 1024 / 512 with the fused backward kernel | 162.0 | **202,245** |

Variants that lost: one product per head stacked on the head axis (274.6 ms: 36 small products
are slower than 5 large ones plus a contraction); feed-forward entirely in bfloat16 (184.5 ms,
no better than stored bfloat16 with float32 arithmetic); no rematerialisation at half the
micro-batch (now 159-167k: with less glue the recomputed forward is cheaper than keeping the
residuals, so `remat=True` is also the faster plan); 256 tiles.

The scripts that made these tables are kept in `docs/tpu_profile/` (`step_profile.py`: parts of
a step and single features removed; `block_profile.py`: parts of a block; `proto_profile.py`:
the candidates, patched into the modules); they need a tokenised shard directory and a free
chip. What is left at 202k: 16 x about 8 ms of block (kernel 2.3, products 2.9, glue 2.8) and
34 ms outside the blocks (head 5.8; the per-loop controller and value embeddings; embedding and
tape).

## Addendum: the plan without rematerialisation

Compile-only, same target (v6e:1x1), after the output-law change:

| program | attention | shape | remat | compile | peak |
|---|---|---|---|---|---|
| train step | splash | A=16, T=1024, B=16 | off | 413 s | 17.03 GiB (temp 18.35) |
| train step | splash | A=8, T=1024, B=32 | on | 325 s | 4.21 GiB |

Halving the micro-batch (and doubling the number of micro-batches, so the tokens per step stay
262,144) fits in device memory without rematerialisation and avoids recomputing every loop body in
the backward pass. Which of the two plans is faster on the chip is unknown; the start-up benchmark
times both and keeps the faster one (`optim.micro_seqs=16, model.remat=False` against the default).

After the configuration was frozen (`configs/final.py`: clocks and confidence channel off), the
same compile-only check was repeated: train step at T=1024 (A=8, B=32) and at T=2048 (A=8, B=16),
both 4.21 GiB peak with rematerialisation, compile 320 s and 306 s; decode (8 streams, 2048
slots) 1.66 GiB. No scatter in any program.
