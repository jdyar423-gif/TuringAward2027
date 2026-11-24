# MaxText angle: XLA / libtpu / splash knobs for ONE TPU v6e chip

Status: complete (2026-11-24). Every flag spelling below was found in the libtpu 0.0.50 binary (section 6).

Context of the run (from the task + repo docs/TPU_NOTES.md): one v6e chip, jitted step = lax.scan over
64 micro-batches of value_and_grad of a looped transformer (d=768, 16 block applications, remat scan of
8 loops), bf16 activations, splash attention (jax.experimental.pallas.ops.tpu.splash_attention) with
BlockSizes 1024/1024/512 + fused bwd, T=1024 now, 2048 later. ~half the step is memory-bound glue.

## 1. benchmarks/xla_flags_library.py (fetched verbatim)

Source: https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/benchmarks/xla_flags_library.py
(fetched 2026-11-24, 216 lines; local copy in scratchpad/dl_maxtext/xla_flags_library.py)

File notice (quoted): "These are potential optimization flags that are used within MaxText models.
They may or may not provide performance improvements in extended use cases but provide a
recommendation for potential grouping of xla flags and their usage."

### 1a. Every flag, grouped as the file groups them (comment quoted), with single-chip relevance

| Python constant | flag(s) exactly as spelled | file comment (quoted) | single chip? |
|---|---|---|---|
| DENSE_VMEM_LIMIT_FLAG | `--xla_tpu_scoped_vmem_limit_kib=98304` | "xla tpu scoped vmem defines the amount of vmem used for the current hlo op. The remaining vmem is used for prefetching latter op needs. These limits are experimentally recommended values for compute bound models." | YES (HIGH) |
| MOE_VMEM_LIMIT_FLAG | `--xla_tpu_scoped_vmem_limit_kib=81920` | same comment (MoE value) | YES (same flag) |
| CUSTOM_VMEM_LIMIT_FLAG | `--xla_tpu_scoped_vmem_limit_kib={vmem_limit}` (format string) | same | YES (same flag) |
| CF_FOR_ALL_GATHER | `--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true` | "Continuation Fusion (CF) for All Gather Collectives. Continuation Fusion is a form of parallelizing compute work with collectives." | no (multi-chip) |
| ASYNC_ALL_GATHER_OVERLAP | `--xla_tpu_enable_async_collective_fusion=false --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true` | "Async All Gather overlapped with TensorCore compute, without Continuation Fusion. Use on platforms where libtpu rejects CF (e.g. v5p rejects CF for both All Gather and All Reduce: "Continuation fusion for ... is enabled on platform viperfish ... not supported"), so CF is disabled globally as libtpu recommends." | no |
| CF_FOR_ALL_REDUCE | `--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_reduce=true` | "Continuation Fusion (CF) for All Reduce Collectives" | no |
| CF_FOR_ALL_REDUCE_AND_ALL_GATHER | `--xla_enable_async_all_reduce=true --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true` | "Continuation Fusion (CF) for All Gather and All Reduce Collectives." | no |
| ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS | `--xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false --xla_sc_disable_megacore_partitioning=true` | "Base Flags needed when enabling sparsecore offloading" | no (only used with SC collective offload) |
| ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR | `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_all_gather_offload_tracing=true --xla_tpu_enable_reduce_scatter_offload_tracing=true --xla_tpu_enable_all_reduce_offload_tracing=true` + base | "Enable SparseCore All Gather (1D), Reduce Scatter (1D) and All Reduce (ND). On Ironwood, by default: ...offload_all_gather as True ...reduce_scatter as True ...all_reduce as True" | no |
| ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER | `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_reduce_scatter_offload_tracing=true` + base | "Enable SparseCore Reduce Scatter (SC RS). Either one of CF or SC can be enabled at a time." | no |
| ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER | `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_all_gather_offload_tracing=true` + base | "Enable SparseCore All Gather (SC AG). Either one of CF or SC can be enabled at a time." | no |
| ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE | `--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_all_reduce_offload_tracing=true` + base | "Enable SparseCore All Reduce (SC AR) ... useful for reducing the gradient reduction all-reduce time with overlapping with compute during that time." | no |
| LAYOUT_FOR_ALL_REDUCE_SCATTER | `--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true` | "Better memory layout for all-reduce (AR)." | no |
| REDUCE_SCATTER_FUSION | `--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1` | "Enable Reduce Scatter (RS) Fusion in certain cases by merging All-reduce (AR) and Dynamic-Slice (DS)." | no |
| DATA_PARALLEL_OVERLAP | `--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true` | "xla_tpu_data_parallel_opt_different_sized_ops: enable pipelining of data parallel ops across multiple iterations; xla_tpu_enable_data_parallel_all_reduce_opt: optimize DCN all-reduces used for data parallel sharding" | no |
| ENHANCED_LAUNCH_BARRIER | `--xla_tpu_use_enhanced_launch_barrier=true` | "Gracefully handle dispatch failures on TPU workers. For Pathways is required for error propagation and out-of-order execution detection." | no (robustness, not speed) |
| HOST_OFFLOAD_FLAGS | `--xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2` | "Host offloading Flags. These are optimizations recommended when using host offloading." | MIXED: see 1b |
| PIPELINING_FLAGS | `--xla_tpu_iova_dma_chunk_size_bytes=16777216` | "Flags to optimize pipeline parallelism over DCN with large host offloading." / "breaks DMA to/from host into 16M chunks" | no (host DMA only) |
| DISABLE_BUNDLE_AWARE_COST_MODEL | `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false` | "Disable bundle-aware CostModel which was causing worse perf b/357103386. Some fusions in the backward pass of the model were 3x slower without this." | YES (fusion cost model; single-chip compile decision) |
| SDC_CHECKER (off by default) | `--xla_tpu_enable_sdc_checker=true --xla_tpu_sdc_check_halt_on_detection=true --xla_tpu_sdc_replicate_llo=true --xla_tpu_sdc_check_repeat_count=5` | "SDC Checker will check for chip / ici / hardware corruption events ... Enabling this will reduce performance by a factor of 1/(repeat_count + 1)." | single-chip but SLOWER: never for speed |
| DEBUG_LOGS (env vars, off) | `TPU_STDERR_LOG_LEVEL=0`, `TF_CPP_MIN_LOG_LEVEL=0`, `TPU_MIN_LOG_LEVEL=0`, `TPU_VMODULE=tpu_configuration_ops_impl=3` | "These are a set of debug logs recommended to be part of a workload." | diagnostic only |
| DISABLE_COLLECTIVE_MATMUL | `--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000` | "Disables collective matmul operations." | no (SPMD windowed einsum) |
| DISABLE_MEGACORE_FUSION_ALLOW_AGS | `--xla_tpu_megacore_fusion_allow_ags=false` | "Disable "megacore fusion allow ags"." | no (v6e has no megacore; all-gathers) |
| ENABLE_ASYNC_COLLECTIVE_PERMUTE | `--xla_enable_async_collective_permute=true` | "Enable async collective permute." | no |


### 1b. Single-chip reading of the HOST_OFFLOAD_FLAGS group (all spelled as in the file)

These are scheduler flags, not host-only, though MaxText only ships them as "recommended when using host offloading":
- `--xla_tpu_enable_all_experimental_scheduler_features=true` : scheduler feature bundle; single-chip possible effect on op ordering / overlap of async copies. Unverified gain on 1 chip.
- `--xla_tpu_enable_scheduler_memory_pressure_tracking=true` : scheduler tracks memory pressure. Single-chip possible.
- `--xla_tpu_host_transfer_overlap_limit=24` : host transfers only -> n/a (no host offload in our run).
- `--xla_tpu_aggressive_opt_barrier_removal=ENABLED` : removes optimization barriers. WARNING: remat uses optimization barriers;
  MaxText's own deepseek3-671b-lineage.yml says "Never add --xla_tpu_aggressive_opt_barrier_removal; it strips the optimization
  barriers Lineage's backward pass depends on." -> could undo remat (memory blow-up) on our scan-of-remat program.
- `--xla_lhs_prioritize_async_depth_over_stall=ENABLED` : latency-hiding-scheduler policy (async ops). Little async work on 1 chip.
- `--xla_tpu_enable_ag_backward_pipelining=true` : all-gather pipelining -> multi-chip.
- `--xla_should_allow_loop_variant_parameter_in_chain=ENABLED`, `--xla_should_add_loop_invariant_op_in_chain=ENABLED` : OpenXLA flags
  guidance: "These 3 flags should be used in conjunction to enable collective pipelining of ICI(Interchip-Interconnect) all-gather
  operations" (with xla_tpu_enable_ici_ag_pipelining) -> multi-chip only.
- `--xla_max_concurrent_host_send_recv=100` : host only -> n/a.
- `--xla_tpu_scheduler_percent_shared_memory_limit=100` (v6e llama3-70b uses 90; Ironwood recipes 150) : how much of memory the
  scheduler may plan to use (percent). Single-chip meaningful (it bounds how far the scheduler hoists/overlaps), mainly a memory-vs-overlap knob.
- `--xla_latency_hiding_scheduler_rerun=2` : rerun LHS passes. Single-chip possible.

### 1c. How the recipes compose LIBTPU_INIT_ARGS

- benchmarks/maxtext_ctk_runner.py line 560: `libtpu_flags = f"LIBTPU_INIT_ARGS='{wl_config.model.xla_flags}'"` then
  the command is `export {libtpu_flags} && ... python3 -m maxtext.trainers.pre_train.train .../base.yml <tuning params>`.
  i.e. the flag constants (each starts with a space) are string-concatenated, e.g.
  `xla_flags=(xla_flags_library.DENSE_VMEM_LIMIT_FLAG + xla_flags_library.CF_FOR_ALL_GATHER)`, and exported as ONE env var.
- HLO dump in the same runner: `XLA_FLAGS='--xla_dump_large_constants --xla_dump_to=/tmp/xla_dump'` (generic XLA flags go in XLA_FLAGS).
- docs/guides/optimization/shard_mode_performance.md: TPU flags "go in `LIBTPU_INIT_ARGS`, *not* `XLA_FLAGS` — libtpu silently
  ignores them in the latter, so a run that looks like it took them may not have". The same doc says the flags were "picked out of
  the 1,108 flags listed in the dump's `*.tpu_comp_env.txt`" -> an XLA dump (XLA_FLAGS=--xla_dump_to=DIR) of the run's own program
  writes a *.tpu_comp_env.txt that lists every libtpu compiler flag with its value for THAT libtpu: the way to verify a spelling and
  default on libtpu 0.0.50 without guessing.
- Per-program alternative used by MaxText (no env var): base.yml `compile_xla_flags: "" # Compiler options e.g.
  compile_xla_flags="--xla_tpu_num_sparse_cores_for_gather_offloading=1 --xla_tpu_scoped_vmem_limit_kib=65536"`; train_utils.py
  parses it with `max_utils.parse_libtpu_flags_to_dict` (strict `--key=value` tokens -> {key: value-string}) and passes
  `jax.jit(functional_train, ..., compiler_options=compiler_options or None)`; train_compile.py uses
  `lowered.compile(compiler_options=compiler_options)`. So `jax.jit(f, compiler_options={"xla_tpu_scoped_vmem_limit_kib": "98304"})`
  or `jax.jit(f).lower(*args).compile(compiler_options={...})` scopes a libtpu flag to ONE program (train step only).

### 1d. Values MaxText uses on v6e (Trillium) - benchmarks/maxtext_trillium_model_configs.py + src/maxtext/configs/tpu/v6e/*.sh

Scoped VMEM limit on v6e:
- 98304 KiB (96 MiB, DENSE_VMEM_LIMIT_FLAG) on every dense v6e recipe (llama2-7b/70b, llama3/3.1-8b/70b/405b, gpt3-175b, mistral-7b, default-*).
- 81920 KiB (80 MiB, MOE) for mixtral / deepseek / custom MoE on v6e.
- 114688 KiB (112 MiB) gemma2-9b-8192 (v6e-256); 122880 KiB (120 MiB) gemma2-27b-8192, gemma3-27b, gemma3-12b-32768 (v6e-256).
- shell recipes: src/maxtext/configs/tpu/v6e/llama2_7b_4096.sh line 39
  `export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true ..."`;
  gemma2_9b.sh `--xla_tpu_scoped_vmem_limit_kib=114688 ...`; gemma2_27b.sh / gemma3_27b.sh `--xla_tpu_scoped_vmem_limit_kib=122880 ...`;
  mixtral_8x7b.sh `--xla_tpu_scoped_vmem_limit_kib=81920`.
- docs/guides/optimization/benchmark_and_performance.md "Scoped Vmem tuning": "`scoped_vmem` can be tuned using
  `xla_tpu_scoped_vmem_limit_kib`. The hardware limitations for vmem are 64M for v5e, 128M for v6e, and 64M for v7x. This can be set
  via `LIBTPU_INIT_ARGS`, range can be 0 through hardware limit." (their example has a typo: `---xla_tpu_scoped_vmem_limit_kib=65536`).
- v5p MaxText recipes also use 98304/81920 (maxtext_v5p_model_configs.py); TPU7x recipes use 65472/65536.
- DISABLE_BUNDLE_AWARE_COST_MODEL (`--xla_tpu_use_bundle_aware_cost_model_for_fusions=false`) is used on v6e only in gpt-3-175b
  and gpt-3-175b-bf16 (with comment "Some fusions in the backward pass of the model were 3x slower without this.").

Splash attention on v6e (MaxText `attention: flash` on TPU = jax splash kernel; base.yml defaults all 512, fused bwd off):
- seq 8192 recipes (llama3_1-8b-8192, llama3_1-70b-8192*, mistral-7b, llama3_1_70b_131072, custom_moe_700b):
  `sa_block_q=sa_block_kv=sa_block_kv_compute=sa_block_q_dkv=sa_block_kv_dkv=sa_block_kv_dkv_compute=sa_block_q_dq=sa_block_kv_dq=2048`,
  `sa_use_fused_bwd_kernel: True` (18 v6e configs). i.e. block_kv_compute == block_kv (no inner kv split).
- older v6e recipes (llama2-7b-4096, llama2-70b-4096, llama3-8b-8192, gpt-3-175b, default-*): `sa_block_q=1024, sa_block_q_dkv=2048,
  sa_block_q_dq=2048`, the rest default 512, separate dq kernel.
- gemma3-12b-32768 on v6e-256: q 1024, kv 1024, kv_compute 1024, q_dkv 512, kv_dkv 2048, kv_dkv_compute 512, q_dq 1024, kv_dq 1024.
- MaxText passes q/k/v layouts into the jax BlockSizes: attention_op.py
  `splash_attention_kernel.BlockSizes(..., use_fused_bwd_kernel=self.use_fused_bwd_kernel, q_layout=splash_attention_kernel.QKVLayout[blocks.q_layout], k_layout=..., v_layout=...)`
  with base.yml `sa_q_layout: "HEAD_DIM_MINOR"`, `sa_k_layout`, `sa_v_layout` (alternative value SEQ_MINOR); and
  `make_splash_mha(mask=..., head_shards=..., q_seq_shards=cp_size, block_sizes=sa_config, attn_logits_soft_cap=..., residual_checkpoint_name="context")`.
- other remat-related v6e settings: `remat_policy: full` (older) or `custom` with `decoder_layer_input/query_proj/key_proj/value_proj/out_proj: offload`
  (host offload of activations: only helps when HBM capacity binds; our run is bandwidth-bound and uses 4.2 GiB of 31 GiB).


## 2. Splash / Pallas knobs (verified in jax-v0.11.2 source, used by MaxText)

Source: https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py
(identical size at jax-v0.11.1; local copy scratchpad/dl_jax/)

- `BlockSizes` docstring: "Those parameters have negligible effect on numerics, but affect performance greatly. Note that changing
  the layouts only influences the physical layout that the kernel will enforce. The logical interface to splash attention always
  takes the head dimension as the minormost one."
- fields: block_q, block_kv, block_kv_compute (None -> block_kv), block_q_dkv, block_kv_dkv, block_kv_dkv_compute (None -> block_kv_dkv),
  block_q_dq, block_kv_dq (must be None with fused bwd), use_fused_bwd_kernel=False,
  `q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR`, `k_layout`, `v_layout`;
  `class QKVLayout(enum.IntEnum): HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]; SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]`.
  With SEQ_MINOR the wrapper does `k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2)` before pallas_call and the
  kernel uses NN instead of NT dot numbers (`qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS`).
  XLA may absorb that swapaxes into the producer's layout (or emit a transpose pass): must be measured.
  MaxText uses SEQ_MINOR q/k only on TPU7x (deepseek3_671b_tpu7x config.py; distillation_qwen3_30b_base.yml: "Worth ~+10% MFU together
  with sa_q_layout=SEQ_MINOR on v7x" - tokamax splash, v7x). No v6e recipe sets a layout (all default HEAD_DIM_MINOR).
- factory (line 2545): `_make_splash_attention(mask, *, block_sizes=None, is_mqa, save_residuals=False, mask_value=DEFAULT_MASK_VALUE,
  attn_logits_soft_cap=None, downcast_smem_data=True, head_shards, q_seq_shards, residual_checkpoint_name=None, interpret=False)`;
  `make_splash_mha_single_device = partial(make_splash_mha, is_mqa=False, head_shards=1, q_seq_shards=1)`.
- `residual_checkpoint_name`: when set, `out = ad_checkpoint.checkpoint_name(out, name=residual_checkpoint_name)` and the logsumexp
  likewise (line 1198-1202). MaxText always passes `residual_checkpoint_name="context"` so its remat policies can SAVE (or offload)
  the attention output+LSE instead of re-running the splash forward kernel inside the rematerialised backward. Our repo calls
  `sa.make_splash_mha_single_device(mask, block_sizes=bs, interpret=interpret)` (no name) under a full-remat loop body, so the forward
  kernel runs again in the backward. Knob: pass `residual_checkpoint_name="context"` and use a policy like
  `jax.checkpoint(body, policy=jax.checkpoint_policies.save_only_these_names("context"))` (memory: out bf16 [B,H,T,hd] + f32 LSE per
  application; we have ~27 GiB headroom). Same math; trades HBM for one kernel forward per block application.
- pallas_call compiler params of the three splash kernels in jax 0.11.2: fwd `pltpu.CompilerParams(dimension_semantics=("parallel",
  "arbitrary", "arbitrary"))`, dq and dkv/fused `dimension_semantics=("arbitrary", "arbitrary", "arbitrary")` - nothing else (no
  vmem_limit_bytes, no allow_input_fusion). "parallel" only matters for megacore chips (v4/v5p); v6e has one TensorCore.

Pallas TPU `CompilerParams` (https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/jax/_src/pallas/mosaic/core.py, line 88-128):
- "vmem_limit_bytes: Overrides the default VMEM limit for a kernel. Note that this must be used in conjunction with the
  --xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes."
- "allow_input_fusion: A list of booleans indicating whether input fusion is allowed for each argument." (lets XLA fuse the producer
  of an operand - transpose/slice/convert - into the kernel's input instead of materialising it in HBM; splash does not set it, so
  using it means vendoring/patching the kernel's pallas_call).
- "disable_bounds_checks: Disable bounds checks in the kernel." (MaxText megablox v2 gmm: `vmem_limit_bytes=vmem_limit_bytes,
  disable_bounds_checks=True`).
- "fuse_transposed_lhs_in_matmul: Hint to compilers to attempt to fuse transposed LHS in MXU ..." ; "flags: A dictionary of command
  line flags for the kernel."; "internal_scratch_in_bytes"; "has_side_effects".
- MaxText idiom for the limit: `vmem_limit_bytes=int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)` (gdn_fp8_projections.py,
  megablox v2 gmm/tgmm default).

## 3. Compilation-cache and AOT facts relevant to a 3-minute restart

- jax-v0.11.2 jax/_src/cache_key.py `_hash_xla_flags`: XLA_FLAGS and LIBTPU_INIT_ARGS (split on spaces, sorted) are hashed into the
  persistent compilation cache key -> ANY flag change is a cache miss and a full recompile of every program on the restart
  (budget the compile time, not 3 minutes). Comment there: "all XLA flags that take an argument must use '=' and not a space".
- MaxText AOT doc (docs/guides/monitoring_and_debugging/features_and_diagnostics.md): LIBTPU_INIT_ARGS is exported before
  `train_compile` "because those affect the compiled object" -> the repo's compile-only check (bes/train/aot.py, CPU host,
  get_topology_desc v6e:1x1) sees the same flags: a flag spelling / VMEM-limit / memory effect can be tried there without the chip.
- MaxText train.py `initialize()`: appends `" --xla_tpu_spmd_rng_bit_generator_unsafe=true"` to os.environ["LIBTPU_INIT_ARGS"] in
  Python before the backend initialises (shows that setting the env var in-process before first jax device use works).


## 4. VMEM limit: hardware ceilings and the documented meaning

- OpenXLA flags guidance (https://openxla.org/xla/flags_guidance), row: "xla_tpu_scoped_vmem_limit_kib | This flag sets the amount of
  scratchpad VMEM available to per op for local usage in KiloBytes. Rest of the VMEM is used as buffer space. |
  xla_tpu_scoped_vmem_limit_kib=16384 | xla_tpu_scoped_vmem_limit_kib=16384 | xla_tpu_scoped_vmem_limit_kib=[4096, VMEM size of the architecture - 1024]"
- OpenXLA error E1001 (https://openxla.org/xla/errors/error_1001): limit "controlled globally for the entire program via the flag
  --xla_tpu_scoped_vmem_limit_kib and per custom kernel via vmem_limit_bytes param"; table "TPU v6e | 127.9MB | 131008 | 134152192"
  (v5e 131072 KiB, v5p 65472, TPU7x 65472) -> on v6e the flag may go up to 131008 KiB. Also: "Modify Memory Coloring: Explicitly
  color/constrain the kernel's inputs/outputs to Vmem using pallas.tpu.with_memory_space_constraint."
- Semantics trade-off (MaxText comment): "xla tpu scoped vmem defines the amount of vmem used for the current hlo op. The remaining
  vmem is used for prefetching latter op needs." -> bigger limit = bigger per-fusion tiles / kernel buffers, smaller pool for
  cross-op prefetch (memory space assignment keeping tensors resident in VMEM). For a memory-bound program the best value is
  NOT obviously the largest; MaxText's v6e values 98304 / 81920 / 114688 / 122880 were found "experimentally ... for compute bound models".
- v6e-1 single-chip use: tpu-recipes microbenchmarks/README.md ("Set up a v6e TPU VM for single-chip microbenchmarks ...
  --accelerator-type=v6e-1") runs `python benchmark_matmul.py --dim 8192 8192 8192 --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536`
  -> "TFLOPS: 827.474382048629"; HBM copy benchmark "bandwidth (GBps, median): 1359.5960438266143" for a 32 MiB tensor (no flag).
  The scripts do `os.environ["LIBTPU_INIT_ARGS"] = args.libtpu_args` after `import jax` and before the first compile.
- Trillium MaxDiffusion recipes (tpu-recipes/inference/trillium/MaxDiffusion/...): Wan2.1 `--xla_tpu_scoped_vmem_limit_kib=65536`,
  Wan2.2 `--xla_tpu_scoped_vmem_limit_kib=32768` + `--xla_tpu_memory_bound_loop_optimizer_options=enabled:true`, StableDiffusion
  `LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=131072"`.
  So on v6e the tuned value ranged 32768 .. 131072 depending on workload: sweep it.

## 5. MaxText's v6e autotuned compiler-flag set (single-chip compile knobs, from inference)

Source: https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/inference/mlperf/trillium/select_xla_flags.py
used by benchmarks_llama2-70b-trillium_2x4.sh: `TEST_FLAGS=$(python3 select_xla_flags.py)` / `export LIBTPU_INIT_ARGS=${TEST_FLAGS}`.
Commit a5057afb8 (2024-11-13) "[Inference Perf] Add autotuned xla flags to improve latency for v6e": "Use auto tune to find a better
set of xla flag settings that can improve inference latency. It gives ~10% latency reduction for generate step and doesn't hurt
prefill." Prefill 60.695 -> 60.451 ms ("0.5% better"); "AR step average time: 57.327 ms ... 9.8% better" (from 63.547).
The memory-bound decode step gained ~10%, the compute-bound prefill nothing: the profile most like our glue-heavy half.
Rendered as `--key=value`. Flags with a documented meaning (OpenXLA flags guidance) marked [doc]; values equal to the documented
default marked (=default).

Fusion:
- `--xla_tpu_enable_experimental_fusion_cost_model=false`
- `--xla_tpu_dot_dot_fusion_duplicated=false`
- `--xla_tpu_dot_dot_fusion=true` [doc "fuses a producer-dot operation with a consumer-dot operation ... producer-dot's output is not
  manifested in slow/main memory"] (=default true)
- `--xla_jf_conv_input_fusion=true`, `--xla_jf_conv_output_fusion=true`, `--xla_jf_conv_reshape_fusion=false`, `--xla_tpu_input_conv_multi_users=false`
- `--xla_tpu_rwb_fusion=false` [doc "enables reduce+broadcast type of fusions, and may decrease memory usage", default true; listed under Memory flags]
- `--xla_tpu_copy_fusion_pad_unpad_ratio=0`, `--xla_tpu_enable_copy_fusion=true`
- `--xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false`
- `--xla_tpu_vector_load_fusion_window=1024`, `--xla_tpu_vector_store_fusion_window=256`
- `--xla_tpu_enable_multi_level_input_dot_dot_fusion=false`, `--xla_tpu_enable_multi_level_output_dot_dot_fusion=false`,
  `--xla_tpu_dot_dot_fusion_separable_convs_only=false`, `--xla_tpu_enable_multi_level_nested_loop_fusion=true`,
  `--xla_tpu_nested_dot_fusion=true`, `--xla_tpu_enable_multi_level_nested_dot_fusion=false`
- `--xla_jf_enable_multi_output_fusion=true` [doc "enables fusions that fuse multiple consumers"] (=default true)
- `--xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=false`
- `--xla_tpu_enable_flash_attention=true` (XLA-internal attention rewrite; irrelevant when the Pallas splash kernel is used)
Memory space assignment (VMEM prefetch) / copies:
- `--xla_tpu_licm_size_inflation_ratio=1`, `--xla_tpu_copy_elision_analysis_allowance=150000`,
  `--xla_tpu_copy_insertion_use_region_analysis_limit=10000`
- `--xla_tpu_order_dot_after_layout=true`, `--xla_tpu_layout_use_dot_grouping=false`
- `--xla_jf_rematerialization_percent_shared_memory_limit=100`
- `--xla_tpu_use_repeated_instance_for_preferred_prefetch_time=true`, `--xla_tpu_enforce_prefetch_fifo_order=false`,
  `--xla_tpu_prefetch_interval_picker_size_override=6000000`
- `--xla_tpu_async_copy_bandwidth_scaling_factor=1` [doc "Scales effective bandwidth for async copies. This is used when making
  prefetch decisions and deciding which tensors should live in VMEM." default 1, range (0, 1]] (=default)
- `--xla_tpu_nd_short_transfer_max_chunks=-1`, `--xla_tpu_enable_aggressive_broadcast_priority_update=true`
- `--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=SQRT`
- `--xla_tpu_memory_bound_loop_optimizer_options=enabled:true` (also on Trillium Wan2.2 and Ironwood Wan recipes; undocumented)
- `--xla_tpu_enable_cross_program_prefetch_freeing=false`
- `--xla_tpu_msa_inefficient_use_to_copy_ratio=0.5` [doc "The ratio of use bytes to copy bytes for a given allocation site below which
  we consider the site to be inefficient. This is used while making VMEM placement decisions." default 0.5] (=default)
- `--xla_tpu_enable_dot_strength_reduction=true` [doc "rewrites non-compute intensive dots as multiply + reduce operations"] (=default true)
Multi-chip in the same file (skip): `--xla_jf_auto_cross_replica_sharding=false`, `--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false`,
`--xla_tpu_enable_windowed_einsum_for_all_gather=false`; `--xla_tpu_prefer_latch_optimized_rhs_layouts=true` (manually set; meaning unverified).
Caveat: tuned for int8 llama2-70b decode on v6e 2x4 with libtpu of Nov 2024; no MaxText training recipe uses it; several values equal
defaults. Treat as an A/B candidate set (all at once, then bisect) - apply via LIBTPU_INIT_ARGS, check the *.tpu_comp_env.txt dump that
each spelling still exists in libtpu 0.0.50.

v5p MaxText MLPerf GPT-3 recipe (src/maxtext/configs/tpu/v5p/gpt3_175b/gpt3_175b_base.sh line 28) uses opposite choices for several of the
same fusion flags: `--xla_tpu_enable_experimental_fusion_cost_model=false --xla_tpu_dot_dot_fusion_duplicated=false --xla_tpu_dot_dot_fusion=false
--xla_jf_conv_input_fusion=true --xla_jf_conv_output_fusion=false --xla_tpu_rwb_fusion=false --xla_tpu_copy_fusion_pad_unpad_ratio=300
--xla_tpu_enable_aggressive_loop_fusion_layout_opt=false --xla_tpu_enable_copy_fusion=false --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false
--xla_tpu_scavenge_vmem_for_fusions=false --xla_tpu_vector_load_fusion_window=256 --xla_tpu_vector_store_fusion_window=64` (+ collective flags).
-> these knobs are workload-specific; v5p (training, megacore) vs v6e (inference).


## 6. Spelling check against the run's own libtpu (0.0.50)

- Downloaded libtpu-0.0.50-cp313-cp313-manylinux_2_31_x86_64.whl from PyPI (sha256 64a959441c0ad4a278e9013024c8b542f40eed5b559902592303377242378671,
  matches PyPI JSON), unpacked, `strings -n 8 libtpu/libtpu.so` (no execution). 4617 distinct `xla_*` tokens.
- ALL 146 flag names quoted in this file (xla_flags_library.py, select_xla_flags.py, v5p gpt3 recipe, TPU7x/DeepSeek and distillation
  flag sets, Trillium MaxDiffusion recipes, OpenXLA guidance names) occur as whole tokens in libtpu 0.0.50 -> spellings exist in the
  exact library of the run (defaults/semantics are NOT in the strings). The 3rd occurrence of xla_tpu_scoped_vmem_limit_kib sits in the
  `xla.jellyfish.TpuCompilationEnvironment` proto descriptor = the list that the dump's *.tpu_comp_env.txt prints with values.
- Notable names that exist (semantics unverified): `xla_tpu_msa_tensor_overrides`, `xla_tpu_msa_sort_order_overrides`,
  `xla_msa_expanded_scoped_alternate_memory_mode`, `xla_tpu_msa_reduce_scoped_vmem_limit`, `xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments`,
  `xla_tpu_scoped_vmem_limit_sweep_profile_path`, `xla_msa_enable_window_prefetch`, `xla_tpu_vmem_scavenging_mode`, `xla_tpu_copy_fusion_threshold`.

## 7. v6e facts from openxla/xla (cost-model table) - answers "what is the default" and "does DVFS apply"

Source: https://github.com/openxla/xla/blob/main/xla/benchmarks/core/platform_info.py (HEAD 463a21521, 2026-11-23), PlatformInfo for
`chip_version=pltpu.ChipVersion.TPU_V6E` (lines ~399-454):
- `default_vmem_limit_kib=32 * 1024` ("default_vmem_limit_kib: The default VMEM limit in KiB for the TPU platform.") -> the v6e default
  scoped VMEM limit is 32 MiB (OpenXLA flags guidance's 16384 is the generic/older value). Unchanged since the file was published
  (2026-08-12); the 2026-11-09 "Update default vmem limits." commit only changed TPU_8T.
- "# Note that v6e doesn't support P states." with `clock_speed_ghz_by_p_state=immutabledict({None: 1.75})`, `default_p_state=None`
  -> `--xla_tpu_dvfs_p_state` (MaxText TPU7x recipes use 7, RL tutorial 6) does nothing useful on v6e.
- `hbm_to_vmem_bandwidth_gb_per_sec=1373` (cost-model figure; tpu-recipes measured 1359.6 GB/s copy on v6e-1),
  `default_internal_scratch_bytes=73728` (Pallas internal scratch default on v6e), `num_alus=4`, `num_xlus=2`.

## 8. Memory-bound-loop optimizer: what `xla_tpu_memory_bound_loop_optimizer_options` sets

Open-source XLA MSA proto (last version before 5b10102b3 "Remove unused memory_space_assignment implementation from open-source XLA."
2026-09-23; file xla/service/memory_space_assignment/memory_space_assignment.proto):
"// Options for memory-bound loop optimizations in memory space assignment. If enabled, this pass can optimize memory-bound unrolled
loops to maximize the bandwidth utilized and minimize the execution time. message MemoryBoundLoopOptimizerOptions { optional bool
enabled = 1; optional float desired_copy_ratio = 2; optional bool allow_unsatisfied_fully_pipelined_prefetch = 3; optional float
min_num_iterations = 4; }" ("min_num_iterations: The minimum number of iterations that the loop needs to be unrolled for the
memory-bound loop optimizer to kick in.") and memory_bound_loop_optimizer.h: "An optimizer for unrolled memory-bound loops. It keeps
track of alternate memory capacity and default memory bandwidth to decide the allocations of each tensor within a loop iteration."
-> text-proto value `enabled:true` (as MaxText/tpu-recipes spell it). It targets UNROLLED loops; our lax.scan bodies are large while
loops, so expect little unless the scan is unrolled (lax.scan(..., unroll=k)) - an A/B, not a promise.
Same proto: `MsaTensorOverride` ("Unified override rule for tensor and operand placement and timing. Overrides specified via
MsaTensorOverride are always strictly enforced; compilation fails if an override cannot be fulfilled.") with actions
`PinInAlternateMemoryAction` ("pin a buffer in alternate memory across its entire lifetime") / `KeepInDefaultMemoryAction` and an
`HloPositionMatcher` (instruction_regex, instruction_name_regex, size_gte, size_lte). libtpu 0.0.50 has `xla_tpu_msa_tensor_overrides`;
no public example of its value format was found -> unverified.

## 9. Diagnostics MaxText docs use that apply to one chip

- docs/guides/optimization/shard_mode_performance.md §4.9 (v5p-8): when a scanned stack's length is a multiple of 8 and the stack axis
  is not leading, layout assignment picked `f32[8,512,8192]{2,0,1:T(8,128)}` and per-iteration writes
  `dynamic-update-slice(..., f32[1,512,8192]{2,0,1:T(1,128)S(1)} %copy, ...)`: "`T(1,128)` is a degenerate tile — one row per tile
  instead of eight — so the write moves 8× the minimum number of tiles" (22% of the step); `param_scan_axis: 0` ([L, in, out]) fixed it
  (-23.6% at L8). Our loop scan has exactly 8 iterations: grep the optimized HLO for `T(1,128)` / dynamic-update-slice into [8,...]
  stacks (remat residuals, per-loop grads) - cheap check, fix = put the loop axis first.
- same doc §3: relayout copies are bandwidth-bound; "2.29–2.38 TB/s effective ... 83–86% of the v5p 2765 GB/s spec".
- MaxText custom_model.md: "Flash Attention. Use the largest possible block size to maximize throughput."; dims "multiples of 256 (for
  Trillium and Ironwood)" (our head_dim 128 cannot change: model maths).
- tune-jax (https://github.com/rdyro/tune-jax) is MaxText's recommended block-size sweeper: `@functools.partial(tune, hyperparams={'block_q': [256, 512, 1024], 'block_k': [8, 16]})`.

## 10. Tokamax splash (MaxText `use_tokamax_splash: true`) - extra knobs, but version-blocked

- tokamax main SplashConfig (https://raw.githubusercontent.com/openxla/tokamax/main/tokamax/_src/ops/splash_attention/splash_attention_kernel.py):
  `qk_diag_skip`: "Skip the wasted causal-diagonal work. Bit-exact: the output, dQ, dK and dV are identical to qk_diag_skip=False."
  (requires block_q == block_kv; backward square tiles; "pure CausalMask with offset 0"); `qk_diag_grid` (default 2), `sv_diag_skip`,
  `diag_band_skip`, `use_experimental_scheduler` ("An experimental scheduler that sometimes produces better softmax overlap."),
  `dq_reduction_steps`, `fuse_reciprocal=True`, `use_base2_exp=True`, `max_logit_const` (changes rounding/overflow behaviour).
  At T=1024 with 1024 tiles the jax kernel's only block is the diagonal one, so ~half of QK/SV is masked work -> qk_diag_skip targets
  exactly that, bit-exactly.
- BLOCKER: tokamax main pyproject: `"jax>=0.12.0"`, `"jaxlib>=0.12.0"` -> not installable next to jax 0.11.2; an older tokamax
  (MaxText base.yml says diag skip "requires tokamax>=0.0.15") would have to be matched to jax 0.11.x (not checked here).
- `use_max_logit_estimate` (MaxText) -> tokamax `max_logit_const`; Ironwood forum post: "We used this parameter to get a 4% reduction
  in step time." (TPU7x; numerics risk, excluded for this run).

## 11. Dead ends / negative results

- No MaxText or tpu-recipes TRAINING recipe for a single v6e chip; smallest is v6e-8 (Llama3.1-8B). Single-chip v6e material = tpu-recipes
  microbenchmarks only (matmul 827 TFLOPS with --xla_tpu_scoped_vmem_limit_kib=65536; HBM copy 1359.6 GB/s).
- DVFS P-state: v6e has none (openxla platform_info.py).
- Public docs for most libtpu fusion/MSA flags: none (web searches for xla_tpu_vector_load_fusion_window, xla_tpu_enable_copy_fusion,
  xla_tpu_prefetch_interval_picker_size_override, xla_tpu_enable_all_experimental_scheduler_features returned nothing); only ~12
  TPU flags are described on openxla.org/xla/flags_guidance.
- Open-source XLA MSA implementation was removed 2026-09-23; semantics only in git history.
- `xla_tpu_enable_flash_attention` (true in MaxText v6e inference set, false in PyTorch/XLA trillium recipes) is XLA's own attention
  rewrite; irrelevant with the Pallas splash custom call.
- `--xla_tpu_bf16_emission_mode=NATIVE_EMISSION` and `--xla_tpu_impure_enable_packed_bf16_math_ops=true` only seen on TPU7x/v5p-era
  recipes; "impure" packed-bf16 math changes elementwise rounding -> violates "same mathematics"; not recommended.


## 12. Suggested A/B order for the run (each = one restart; every flag change is a compile-cache miss)

1. Scoped VMEM sweep, one flag: `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=N"` with N in {16384, 32768 (= v6e default), 65536,
   98304 (MaxText dense v6e), 131008 (v6e max)}. Rationale: the step's win from small micro-batches says VMEM residency matters; the
   flag moves the split between per-op scratch and the cross-op VMEM pool, so the optimum may sit BELOW the default, unlike MaxText's
   compute-bound values. Try it first in the CPU compile-only check (bes/train/aot.py) to see the compile accepts it and the peak.
2. Splash tiles (code knob, no flag): T=1024 -> block_kv_compute=1024 (MaxText v6e uses kv_compute == kv); decouple dkv tiles from fwd
   tiles; T=2048 -> 2048/2048/2048 + fused bwd (all 18 MaxText v6e seq-8192 recipes). Optionally `residual_checkpoint_name="context"` +
   a remat policy that saves it.
3. The v6e autotuned bundle from select_xla_flags.py as one A/B; if it wins, bisect (start with memory_bound_loop_optimizer_options,
   rwb_fusion=false, copy_fusion, vector_load/store windows, prefetch_interval_picker_size_override).
4. `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false` alone (MaxText v6e gpt-3: backward fusions 3x slower without it).
5. MSA placement knobs: `--xla_tpu_msa_inefficient_use_to_copy_ratio` below 0.5 (e.g. 0.25), `--xla_tpu_async_copy_bandwidth_scaling_factor`
   (0,1] - documented by OpenXLA, never tuned by MaxText for training.
Avoid: `--xla_tpu_aggressive_opt_barrier_removal` (can strip remat barriers), DVFS (no P-states on v6e), SDC checker, numerics-changing bf16 flags.
