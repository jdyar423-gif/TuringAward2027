# Knobs: Pallas TPU + splash attention (jax 0.11.2, one TPU v6e chip)

Angle: Pallas on TPU and the splash attention kernel (and its siblings). Everything below was read
from source or a public page; nothing was run on a TPU. "verified" = exact spelling quoted from a
source; "effect" claims are from the source unless marked as my inference.

## Sources actually read

- jax 0.11.2 wheel from PyPI (`pip download jax==0.11.2 --no-deps`), unpacked under
  `scratchpad/dl_jax/x/` and only read:
  - `jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py` (2645 lines),
    `splash_attention_mask.py`, `splash_attention_mask_info.py`, `__init__.py`
  - `jax/_src/pallas/mosaic/core.py` (CompilerParams), `jax/_src/tpu_info.py` (chip table),
    `jax/_src/tpu_custom_call.py`, `jax/_src/pallas/mosaic/lowering.py`,
    `jax/_src/pallas/mosaic/primitives.py`, `jax/_src/pallas/pallas_call.py` (vmap rule),
    `jax/_src/config.py`
  - siblings: `flash_attention.py`, `paged_attention/`, `ragged_paged_attention/`, `megablox/`
- docs at tag jax-v0.11.2 (raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/docs/pallas/...):
  `tpu/details.rst` ("Writing TPU kernels with Pallas"), `tpu/pipelining.md`, `tpu/sparse.md`,
  `tpu/matmul.md`, `tpu/hardware.ipynb`, `design/design.md`, `CHANGELOG.md`;
  test `tests/pallas/tpu_pallas_test.py` (same tag).
- tokamax 0.0.17 wheel (PyPI; `Requires-Dist: jax>=0.11.1`, bench extra `jax==0.11.2`), file
  `tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py` (3055 lines).
  tokamax 0.0.18 needs `jax>=0.12.0` (not usable with this run).
- MaxText main: `benchmarks/maxtext_trillium_model_configs.py` ("Shared Benchmark config for v6e
  orchestrations."), `benchmarks/xla_flags_library.py`, `src/maxtext/configs/base.yml`,
  `src/maxtext/layers/attention_op.py`, `src/maxtext/configs/post_train/distillation_qwen3_30b_base.yml`,
  `src/maxtext/trainers/pre_train/scripts/olmo/run_olmo3_7b_stage1.sh`.
- openxla.org/xla/flags_guidance, openxla.org/xla/errors/error_1001,
  docs.cloud.google.com/tpu/docs/tutorials/LLM/maxdiffusion-inference-v6e,
  arxiv.org/html/2611.01426v1 (FlyWheel, v6e attention kernels),
  github vllm-project/tpu-inference `.buildkite/xla_autotune/flags.txt` (v7x sweep list).

## Chip facts as jax 0.11.2 encodes them (`jax/_src/tpu_info.py`)

- `| v6e     | 1                             | Yes       | No               |`
  (TensorCores per chip / lite / Megacore support). One TensorCore: `dimension_semantics`
  "parallel" cannot split a kernel over cores on v6e.
- v6e entry: `mxu_column_size=MXU_COLUMN_SIZE_GEN_GE_6` (256), `num_mxus=2`,
  `vmem_capacity_bytes=128 * 1024 * 1024,  # 128 MiB per core`, `smem_capacity_bytes=1024 * 1024`,
  `mem_bw_bytes_per_second=int(1.64e12)`, `bf16_ops_per_second=int(9.20e14)`.
  Ridge point 920e12/1.64e12 = 561 flop/byte (my arithmetic).
- `# Caveat: before TPU7x, by default XLA does not use large 2nd minor tiling
  # but it can be enabled by setting the flag
  # xla_tpu_enable_large_2nd_minor_layout_for_x16.` (bf16 tiling (16,128) instead of (8,128)(2,1)).

## What the jax 0.11.2 splash kernel does (line numbers of splash_attention_kernel.py)

Factory: `make_splash_mha_single_device = partial(make_splash_mha, is_mqa=False, head_shards=1, q_seq_shards=1)`
-> `_make_splash_attention(mask, *, block_sizes=None, is_mqa, save_residuals=False,
mask_value=DEFAULT_MASK_VALUE, attn_logits_soft_cap=None, downcast_smem_data=True, head_shards,
q_seq_shards, residual_checkpoint_name=None, interpret=False)` (l.2545-2558).
Call: `kernel(q, k, v, segment_ids=None, sinks=None)`, q `[H,T,hd]`.

- Forward pallas_call (l.1144-1171): grid `(num_q_heads, q_seq_len // bq, grid_width)`,
  `compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary"))`.
  dq kernel (l.1655) and dkv kernel (l.2225) all "arbitrary". None of the three passes
  `vmem_limit_bytes`, `flags`, `allow_input_fusion`, `disable_bounds_checks` or a `cost_estimate`
  (jax's own `flash_attention.py` does pass `cost_estimate=_fwd_cost_estimate(...)`, l.777).
  => every pallas_call compiler option is unreachable through the splash API; changing one means
  patching `pltpu.CompilerParams` / `pl.pallas_call` as seen from the splash module (or forking
  the 3 files).
- m/l/o accumulators are OUTPUTS with a constant index map (l.1099-1111,
  `# TODO(sharadmv): convert m/l to be scratch`). Under `jax.vmap` every output is batched
  (`all_dims = list(dims) + [0] * grid_mapping.num_outputs`, pallas_call.py), so each example
  writes m, l ([bq,128] f32) and o ([bq,hd] f32) to HBM: 1.5 MiB per example per forward call at
  bq=1024, never read (my arithmetic).
- logsumexp residual is lane-broadcast: `jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)`
  then `logsumexp = logsumexp[..., 0]` (l.1113-1116, 1194-1196): 128x the needed bytes.
- `residual_checkpoint_name` (l.1198-1203): `out = ad_checkpoint.checkpoint_name(out, name=residual_checkpoint_name)`
  and the same for the already sliced `logsumexp`.
- Backward: `di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))`
  (l.2300) runs as XLA outside the kernel; lse/di are re-broadcast to `(num_q_heads, NUM_SUBLANES, q_seq_len)`
  (`# TODO(apaszke): Remove the sublane expansion once Mosaic has all retilings`, l.2123-2131).
- Fused backward: `dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)` (l.2011),
  `dq = dq_unreduced.sum(axis=0)` (l.2249): leading dim 1 at T=1024/bkv_dkv=1024 (free), 2 at
  T=2048/bkv_dkv=1024 (extra [H,T,hd] write+read, partial sums in bf16).
  `shrink_grid=not block_sizes.use_fused_bwd_kernel` (l.2614): fused dkv grid keeps fully masked tiles.
- Segment ids are broadcast in HBM before each kernel:
  `jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, NUM_LANES), (0,))` and `(NUM_SUBLANES, kv_seq_len)`
  (l.1056-1061, 1545-1550, 2079-2084): int32 [T,128] per example per kernel.
- Masking: `CausalMask` is a `_ComputableMask`; `_process_mask` then uses `mask_function`/`q_sequence`
  ("If the mask object defines a q_sequence and a mask_function, then make use of these in the kernel
  rather. This is preferable over loading the mask from memory.", mask_info.py l.672-675).
  In `_apply_mask_and_soft_cap` (l.597-694) the mask_function and the segment compare run on EVERY
  computed block (`should_not_mask` is only consulted on the `mask_ref` path), i.e. fully visible
  below-diagonal blocks still pay iota+tile+compare+and+where on the VPU (tokamax 0.0.17 gates the
  causal part with `has_partial_mask`).
- `MultiHeadMask([CausalMask((t,t))]*H)` is de-duplicated (`len(unique_masks) == 1`) -> mask_function
  path + grid shrinking apply. vmap over the batch with the kernel object closed over takes the cheap
  batching path (`grid=(new_grid_dim, *grid_mapping.grid)`), not `_batch_with_explicit_loop`. Keep.
- Precision: forward `v = v.astype(float32); o_curr = lax.dot_general(s_curr, v, sv_dims)` (f32 x f32);
  Mosaic lowering `if precision is None or precision == lax.Precision.DEFAULT: precision_attr = None  # That's the default in Mosaic.`
  / `lax.Precision.HIGHEST -> "#tpu.contract_precision<fp32>"` (lowering.py l.2958-2967);
  details.rst: "Pallas TPU lowering is aware of ``jax.default_matmul_precision``. For best performance
  (and lowest precision), use ``bfloat16``." The repo sets "highest" only in `bes/export.py:643`
  and `scripts/midrun_eval.py:86` (context managers), not in training. Keep it so.
- At hd=128 the kernel is softmax/VPU-bound, not MXU-bound: FlyWheel (arXiv 2611.01426, one v6e chip,
  JAX 0.11.0, libtpu 0.0.44): "At D=128 every kernel drops by roughly half, to 38–44%, 29–39%, and
  12–30%" (FlyWheel, Splash-Attention [tokamax], RPA-v3, of v6e bf16 peak); cause given: halving the
  head dimension halves the arithmetic intensity of the products "but not of the softmax between them".
  Same paper: "Enabling Mosaic's LP LLO scheduler, used by Splash-Attention by default, does not recover
  this overlap" (for their unpipelined online-softmax variant).

## Knob list (single chip), most promising first

1. `--xla_tpu_scoped_vmem_limit_kib=N` (LIBTPU_INIT_ARGS). Global per-op scratch VMEM; "Rest of the
   VMEM is used as buffer space." (openxla flags_guidance: default/suggested 16384, candidates
   `[4096, VMEM size of the architecture - 1024]`). tokamax 0.0.17 comment: "needs more than XLA's
   default 32 MiB". MaxText v6e: `_DENSE_VMEM_LIMIT = 98304` ("xla tpu scoped vmem defines the amount
   of vmem used for the current hlo op. The remaining vmem is used for prefetching latter op needs.
   These limits are experimentally recommended values for compute bound models."), used by every
   Trillium model in `maxtext_trillium_model_configs.py`; MaxDiffusion v6e-4 tutorial:
   `LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=65536"`.
   error_1001 lists v6e usable scoped VMEM 131008 KiB. For THIS memory-bound program sweep both
   directions (more scoped = less buffer/prefetch space): e.g. 16384, 32768, 65536, 98304.
2. Splash backward tiles decoupled from forward tiles (needs `bes/model/attention_splash.block_sizes`
   to read separate fields): MaxText v6e recipe at seq 2048, head_dim 128:
   `"sa_block_q": 1024, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048` (others default:
   `sa_block_kv: 512`, `sa_block_kv_compute: 512`, `sa_block_kv_dkv: 512`, `sa_block_kv_dkv_compute: 512`,
   `sa_block_kv_dq: 512`, `sa_use_fused_bwd_kernel: false`). Try at T=2048 against the current
   1024/1024/512 fused (and at T=1024: q_dkv=1024).
3. `residual_checkpoint_name="<name>"` on `make_splash_mha_single_device` + remat policy
   `jax.checkpoint_policies.save_only_these_names("<name>")`: keeps only `out` (bf16) and the
   sliced lse ([H,T] f32) so the backward of the remat'd loop step does not re-run the forward kernel.
   MaxText jax path: `residual_checkpoint_name="context",`. The repo's current `remat_keep=kernel`
   policy keeps every `pallas_call` output, i.e. also the 128-lane lse and the m/l/o "scratch"
   outputs (measured -2%: 161.2 vs 158.3 ms); the named variant is 4x smaller (my arithmetic).
4. Mosaic per-kernel scheduler flag: `pltpu.CompilerParams(..., flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True})`
   (tokamax 0.0.17 `use_experimental_scheduler` -> exactly this; "An experimental scheduler that
   sometimes produces better softmax overlap."). jax splash: patch needed. Global relatives seen
   only in a v7x sweep list: `--xla_tpu_consider_lp_llo_scheduler=true`,
   `--xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=true` (unverified effect).
5. tokamax 0.0.17 splash (`tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel`,
   `make_splash_mha_single_device(mask, config=SplashConfig(...))`, 2-D mask): fused bwd only;
   m/l/o are real VMEM scratch; causal mask computed only on partial blocks; per-call segment-range
   tile pruning (`_refine_mask_info_with_segments`, reduced over the vmap batch); cost estimates;
   `qk_diag_skip`/`sv_diag_skip`/`qk_diag_grid` ("so the result is bit-exact; the elementwise/softmax
   ops still run on the full assembled tile"; needs `block_q == block_kv == block_kv_compute`);
   `use_base2_exp=True` default multiplies q by LOG2E in-kernel (extra bf16 rounding: set False for
   numerics closest to now); `fuse_reciprocal=True`; `max_logit_const` (changes overflow behaviour:
   not for this run); `dq_reduction_steps` (3 or None; irrelevant with <= 3 kv steps).
6. `allow_input_fusion=[...]` (one bool per pallas_call operand incl. scalar prefetch). jax test
   `test_allow_input_fusion`: `compiler_params=pltpu.CompilerParams(allow_input_fusion=[True])`
   then `assert re.search(r'fusion.*kind=kCustom.*fused_computation', compiled)`: XLA fuses the
   producer into the kernel. Candidate operands: the segment-id broadcasts, elementwise producers of
   q/k/v. Patch needed; effect unmeasured.
7. `disable_bounds_checks=True` (CompilerParams: "Disable bounds checks in the kernel.");
   flags_guidance: `xla_mosaic_on_device_checks=bounds` is the default ("enables on-device checks for
   Mosaic codegen ... on bounds"). Patch needed; expected small (scalar work).
8. Layouts `q_layout/k_layout/v_layout=QKVLayout.SEQ_MINOR`: docstring "changing the layouts only
   influences the physical layout that the kernel will enforce"; wrapper does
   `k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2)` (an HBM transpose unless XLA
   folds it into the producer); dkv kernel transposes in-kernel (`return ref[:, slice_k].T`).
   MaxText uses `sa_q_layout=SEQ_MINOR sa_k_layout=SEQ_MINOR sa_v_layout=HEAD_DIM_MINOR` on TPU7x
   (tokamax), not on v6e. matmul.md: "Accelerators often support native matrix multiplication routine
   that fuse a RHS transpose. For instance TPU v5e, the MXU allows us to do `x @ y.T`". Low on v6e.
9. `block_kv_compute` / `block_kv_dkv_compute` = 256 with block_q = block_kv = 1024 (smaller f32
   live tile, more unrolled iterations) - unmeasured point of the existing sweep. Low.
10. `cost_estimate=pl.CostEstimate(flops=..., transcendentals=..., bytes_accessed=...)` on the splash
   pallas_calls (tokamax computes `_fwd_cost_estimate`; MaxText `cost_estimate_flops_fwd` says
   "to overlap for communication"). Single-chip effect unknown. Low.
11. `pltpu.with_memory_space_constraint(x, pltpu.VMEM)`: "constrains the memory space where it should
   be allocated ... As of now, this only operates on the inputs pallas_calls". error_1001 lists it.
   With 4-row micro-batches q/k/v are 6 MiB each (fits 128 MiB VMEM) - needs a kernel fork whose
   BlockSpecs take VMEM operands; speculative.
12. `pl.BlockSpec(..., pipeline_mode=pl.Buffered(buffer_count=N[, use_lookahead=True]))`
   ("The default buffer count is 2 for all inputs and outputs."). Not exposed by splash; the kernel
   is VPU-bound, so low.

Checks, not speed knobs: `interpret=False` on TPU (repo "auto" ok); `jax_pallas_enable_debug_checks`
(default False), `jax_pallas_poison_buffers` (default False) stay off; `jax_default_matmul_precision`
stays default in training; `dimension_semantics` irrelevant on v6e; `save_residuals=True` is
inference-only (`_splash_attention_fwd` raises "Higher-order AD not supported"); `downcast_smem_data`,
`mask_value`, `attn_logits_soft_cap=None`, `sinks=None`: no speed effect (soft cap would add tanh).

## Siblings in the same directory

- `flash_attention.py`: `BlockSizes(block_q, block_k_major, block_k, block_b, block_q_major_dkv,
  block_k_major_dkv, block_k_dkv, block_q_dkv, block_k_major_dq, block_k_dq, block_q_dq)`; takes
  `[B,H,T,D]` and `segment_ids` `[B,T]` natively (no vmap), `block_b` batches rows per grid step,
  `causal=True` in-kernel, `sm_scale=1.0`, passes a cost estimate; no fused backward. Low.
- `paged_attention/`, `ragged_paged_attention/` (has `vmem_limit_bytes` argument): decode/serving
  only. `megablox/` (grouped matmul), `matmul.py`, `all_gather.py`: not relevant.

## Multi-chip only (named, not pursued)
All-gather/reduce-scatter/all-reduce fusion and SparseCore offload flags from xla_flags_library.py
(CF_FOR_ALL_GATHER, CF_FOR_ALL_REDUCE, LAYOUT_FOR_ALL_REDUCE_SCATTER, REDUCE_SCATTER_FUSION,
DATA_PARALLEL_OVERLAP, ENABLE_SPARSECORE_OFFLOADING_*), `head_shards`/`q_seq_shards`,
`manual_sharding_spec`, `sa_bwd_dkv_megacore`, megacore `dimension_semantics`, ring/Ulysses attention,
host offload flags.

## Further compiler options of pallas_call (jax 0.11.2 `pltpu.CompilerParams`, core.py l.88-146)

Docstring quotes: "vmem_limit_bytes: Overrides the default VMEM limit for a kernel. Note that this
must be used in conjunction with the --xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib >
vmem_limit_bytes." (tokamax 0.0.17 nevertheless sets per-kernel limits alone: "setting it per kernel
keeps the kernel self-contained", `_VMEM_USAGE_FRACTION = 0.9`). "allow_input_fusion: A list of
booleans indicating whether input fusion is allowed for each argument." "flags: A dictionary of
command line flags for the kernel." "internal_scratch_in_bytes: The size of the internal scratch
space used by Mosaic." "disable_bounds_checks: Disable bounds checks in the kernel."
"fuse_transposed_lhs_in_matmul: Hint to compilers to attempt to fuse transposed LHS in MXU if users
specify the transposed layout of LHS in matmul operations, e.g., `jnp.einsum('km,kn->mn', lhs, rhs)`;
... When transposition is performed separately from multiplication (e.g. jnp.matmul(lhs.T, rhs)),
this flag does not affect the compiler's decision". Lowering: `transpose_lhs_hint=not
ctx.forward_compatible and ctx.lowering_context.fuse_transposed_lhs_in_matmul and implicit_transpose`.
Splash's dkv kernel computes `dq = lax.dot_general(ds.T.astype(k.dtype), k, NN_DIM_NUMBERS, ...)`
(explicit .T, l.1821-1824): rewriting it as a contraction over ds's dim 0 + this flag is a fork-level
idea (same math), effect unmeasured.
Defaults: `serialization_format=1`, `shape_invariant_numerics=True`, `needs_layout_passes=True`
(CHANGELOG 0.10.2: "now defaults ``needs_layout_passes`` to True").

## Other facts

- Persistent compilation cache key hashes `XLA_FLAGS` and `LIBTPU_INIT_ARGS` (`jax/_src/cache_key.py`
  `_hash_xla_flags`: `libtpu_init_args_env_var = os.getenv("LIBTPU_INIT_ARGS")`): every flag variant
  is a cache miss -> full compile on restart.
- jax appends `--xla_tpu_use_enhanced_launch_barrier=true` to LIBTPU_INIT_ARGS on Cloud TPU VMs
  (`cloud_tpu_init.py`), not a speed flag.
- Kernel names for profiles: `splash_mha_fwd_segmented_residuals` (and `_no_residuals`),
  `splash_mha_dq_segmented_no_residuals`, `splash_mha_dkv_segmented_no_residuals`
  (`get_kernel_name`); xprof metadata carries the BlockSizes as JSON.
- Docs on block size choice: details.rst "VMEM is fairly large for such a low-level memory hierarchy
  (16MB+), making it possible to use large window sizes. And, oftentimes, the larger the window size,
  the better the eventual hardware utilization will be."; sparse.md: "Using larger block sizes also
  helps ... However, a counter-argument to this in practice is that smaller block sizes also enables
  data to be more sparse"; BlockSizes docstring: "Those parameters have negligible effect on numerics,
  but affect performance greatly."; `get_default` = all 128 with
  `# TODO(apaszke,sharadmv): Select better parameters based on a heuristic.`
- XLA flag seen in a v6e (Trillium) public config, single-chip relevant (fusion cost model):
  `DISABLE_BUNDLE_AWARE_COST_MODEL = " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"`
  with comment "Some fusions in the backward pass of the model were 3x slower without this." (used by
  gpt-3-175b / gpt-3-175b-bf16 Trillium configs).
- Flag names seen only in vLLM tpu-inference v7x autotune candidate list (queue `tpu_v7x_8_queue`),
  spelling only, no effect claim: `--xla_tpu_consider_lp_llo_scheduler=true`,
  `--xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=true`, `--xla_tpu_vmem_scavenging_mode=NONE|SAFE_IFF_KERNEL_REQUIRED|SAFE|AGGRESSIVE`,
  `--xla_tpu_msa_reduce_scoped_vmem_limit=true`, `--xla_jf_experimental_vmem_for_hlo_outputs=65536`,
  `--xla_tpu_enable_explicit_prefetch_scheduler=true`, `--xla_msa_enable_window_prefetch=true`,
  `--xla_tpu_mosaic_fusion=false`, `--xla_tpu_enable_vmem_to_vmem_dmas=false`, `--xla_jf_tune_large_vmem=true`.
- `--xla_tpu_enable_large_2nd_minor_layout_for_x16` (jax tpu_info.py comment, v6e and older):
  bf16 (16,128) tiling instead of (8,128)(2,1); affects every bf16 buffer and the Mosaic boundary;
  effect unknown, risky.
- MaxText v7x (not v6e) splash layouts: `sa_q_layout: "SEQ_MINOR"`, `sa_k_layout: "SEQ_MINOR"`,
  `sa_v_layout: "HEAD_DIM_MINOR"` with tokamax; olmo3 Ironwood script flags
  `--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_dvfs_p_state=7`
  ("Ironwood-tuned XLA flags (loss-neutral; MFU 27% → 41% → 44.5%)") - v7x only evidence.

Status: DONE.
