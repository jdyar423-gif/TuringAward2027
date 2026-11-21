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

**Block sizes.** Defaults (`ModelCfg.splash_block_q / _kv / _kv_compute`) are 512 for the
forward and both backward kernels: the per-example grid is `(6, 2, 2)` at `T = 1024` and
`(6, 4, 4)` at `T = 2048`. They are a starting point, not a measurement. Override with
`--override model.splash_block_q=1024` etc. and sweep on the chip (below).

## Known pitfalls

1. **Import name of the splash module.** `bes/model/attention.py` selects the splash path with
   `from bes.model import splash`; the module is `bes/model/attention_splash.py`. Until that line
   reads `from bes.model import attention_splash as splash`, anything that runs
   `attn_impl="splash"` must call `bes.model.attention_splash.register_alias()` first (the
   compile-only tool does; `tests/model/test_splash.py` holds a strict xfail that flips when the
   import is fixed).
2. **Interpret mode on the compile host.** `splash_interpret="auto"` means "interpret unless the
   default backend is a TPU". On a CPU host that compiles for the TPU topology the default
   backend is the CPU, so the compile-only tool forces `splash_interpret="false"`; on the TPU VM
   `"auto"` is right.
3. **The dense `xla` attention materialises `[B, H, T, T]` float32 logits and probabilities** per
   block and loop. It compiles and fits at the training shapes with remat (table below) but is
   meant for tests; train with splash.
4. **Remat.** `configs/final.py` sets `remat=True`. With `remat=False` (the `ModelCfg` default,
   still used by `configs/no_library.py`) neither training shape compiles: the compiler needs
   32.93 GB of temporaries (table below). Tune the policy on the chip.
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
   training process compiles both shapes up front; enable the persistent compilation cache
   (`jax.config.update("jax_compilation_cache_dir", ...)`) so that a restart does not pay again,
   and check on the chip that the cache is hit.

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
