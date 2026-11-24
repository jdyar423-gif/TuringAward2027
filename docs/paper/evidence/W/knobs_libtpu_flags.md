# libtpu / XLA TPU flags outside MaxText: knobs for ONE TPU v6e chip

Status: COMPLETE (2026-11-24). Angle: public uses of LIBTPU_INIT_ARGS / xla_tpu_* / xla_jf_* / xla_msa_*
and jax.config knobs anywhere other than MaxText: openxla/xla docs + benchmarks, jax-ml/jax source/docs/tests,
jax-ml/jax-llm-examples, apple/axlearn, marin/levanter, google-pytorch/torch_tpu, pytorch/xla, vLLM tpu-inference,
vllm-torchtpu issues, google/ml-workload-debugging-wiki, GKE samples, google/saxml, google/qwix, DeepMind simply,
AI-Hypercomputer accelerator-microbenchmarks / accelerator-agents wiki, community v6e projects (nanocode,
tpu_performance_autoresearch_wiki), public tpu_comp_env dumps. (MaxText itself = sibling file knobs_maxtext.md.)

Method: GitHub code search (`gh search code`, public code), issue search (REST search API; bodies taken from
search results), raw.githubusercontent.com downloads (saved under scratchpad/wf/libtpu_dl/, search JSON under
scratchpad/wf/libtpu_gh/), web search. JAX facts read from the jax 0.11.2 wheel (scratchpad/dl_jax/x).
Spelling check: every flag below occurs as a whole token in the run's libtpu 0.0.50 (section 9).

Nothing below was run on a TPU. "verified" = exact spelling quoted from a public source AND present in libtpu
0.0.50; the semantics are only what the quoted source says.

-------------------------------------------------------------------------------------------------------------
## 0. How a libtpu flag reaches the compiler (and the cache) - all routes public

1. `LIBTPU_INIT_ARGS="--flag=value ..."` set before the TPU backend initialises (os.environ before `import jax`
   works; the ML workload debugging wiki: "In interactive environments like Google Colab or Jupyter notebooks, set
   the environment variable prior to importing JAX or initializing the runtime").
2. `XLA_FLAGS` - only for XLA DebugOptions flags (xla_dump_*). Do NOT put xla_tpu_* / xla_jf_* there:
   - google/ml-workload-debugging-wiki hlo-llo-dump-analysis.md: "The `--xla_jf_dump_to` flag controls LLO dumping
     and **MUST** be passed through `LIBTPU_INIT_ARGS`. Passing it via `XLA_FLAGS` will be ignored by the TPU runtime."
   - PSAL-POSTECH/PyTorchSim-Triton-Backend issue #69 (v6e AOT dump recipe): "The flags go in two environment
     variables. Mixing them aborts with `parse_flags_from_env.cc: Unknown flags`."
   - google-pytorch/torch_tpu issue #3444: "Not all TPU flags are possible to pass via XLA_FLAGS env, some are only
     accepted via LIBTPU_INIT_ARGS (e.g. `xla_tpu_scoped_vmem_limit_kib`."
   (AXLearn's docstring claims XLA_FLAGS "works on any platform including TPU" - contradicted by the three above.)
3. Per executable, no restart: `jax.jit(f, compiler_options={...})` or `jitted.lower(...).compile(compiler_options={...})`.
   JAX 0.11.2 jax/_src/compiler.py: the dict becomes `compile_options.env_option_overrides`; keys
   `optimization_level` / `memory_fitting_level` are moved to ExecutableBuildOptions. Public uses:
   - jax-ml/jax-llm-examples gemma4/main.py:
     `decode_options = dict(xla_tpu_scoped_vmem_limit_kib=200 * 1024) if g4jax.which_platform(cfg) == "tpu" else {}`
     `decode_step = jax.jit(g4jax.decode_step, donate_argnames=("cache",), compiler_options=decode_options)`
   - jax Pallas debugging doc: `jax.jit(pl.pallas_call(...), compiler_options={'xla_tpu_enable_log_recorder': 'true'})`
   - apple/axlearn compiler_options.py: "These options can be passed to `jitted_fn.lower(...).compile(compiler_options=...)`"
   - jax issue #40553: "the flag is honored per-executable via `jax.jit(f).lower(...).compile(compiler_options=
     {"xla_tpu_offload_gather_to_sparsecore": "false"})` - the offload forms disappear from that executable's
     optimized HLO only"
   - ayaka14732/tpu-v4-latency-numbers: `jax.jit(kernel, compiler_options={'xla_msa_enable': 'false',
     'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(x).compile()`
   - torch_tpu compilation.cc sets its TPU defaults the same way (`options.env_option_overrides`).
   => For the run: one process can compile the same step under N flag settings via compiler_options and time them
   back to back (one restart instead of N). Each setting is its own compile (minutes for this program).
4. Cache: jax 0.11.2 cache_key.py `_hash_xla_flags` hashes XLA_FLAGS and LIBTPU_INIT_ARGS (sorted), and
   `_hash_serialized_compile_options` hashes env_option_overrides -> every flag change = persistent-cache miss.
   JAX itself appends a flag on a real TPU: jax/_src/cloud_tpu_init.py (0.11.2):
   `if '--xla_tpu_use_enhanced_launch_barrier' not in os.environ.get('LIBTPU_INIT_ARGS', ''): os.environ['LIBTPU_INIT_ARGS'] = (... + ' --xla_tpu_use_enhanced_launch_barrier=true')`
   jax docs/aot.md (main, "Matching the cache key across hosts"): "Before `import jax`, set
   `os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_use_enhanced_launch_barrier=true"` on both hosts (JAX appends this
   flag automatically only when physical TPUs are present) ... Set `jax_persistent_cache_min_compile_time_secs=-1`
   and `jax_persistent_cache_min_entry_size_bytes=-1` on the CPU host so fast compiles are always written."
   => relevant to the repo's CPU compile host (bes/train/aot.py): a cache entry compiled on CPU only hits on the
   TPU VM if LIBTPU_INIT_ARGS (incl. this flag and any perf flag) is identical on both.

-------------------------------------------------------------------------------------------------------------
## 1. VMEM: the single most-used single-chip flag

### 1a. `--xla_tpu_scoped_vmem_limit_kib=N` (KiB)
- openxla docs/flags_guidance.md (Performance Flags, "Compute centric"): "This flag sets the amount of scratchpad
  VMEM available to per op for local usage in KiloBytes. Rest of the VMEM is used as buffer space." Default column
  `xla_tpu_scoped_vmem_limit_kib=16384`, candidate `[4096, VMEM size of the architecture - 1024]` (generic table).
- v6e default = 32 MiB: google/ml-workload-debugging-wiki numerics-error-triage.md: "`--xla_tpu_scoped_vmem_limit_kib`:
  Limits VMEM available for scoped allocations (default is 16 MiB on v4, v5e, v5p; 32 MiB on v6e+)."
  Public tpu_comp_env dump (patrick-toulme/justabyte, MaxText run): "# xla_tpu_scoped_vmem_limit_kib in the current
  tpu_comp_env has a non-default value. Default flag value: -1" (=> -1 means per-platform default).
- v6e maximum: openxla docs/errors/error_1001.md table: "| TPU v6e | 127.9MB | 131008 | 134152192 |".
  jax 0.11.2 tpu_info.py TPU_V6E: `vmem_capacity_bytes=128 * 1024 * 1024,  # 128 MiB per core`.
- error_1001.md: instruction-scoped vs program-scoped VMEM: "Instruction-scoped allocations: Temporary storage in Vmem
  while executing a single HLO instruction. This includes operand span buffer (e.g. for double buffering) and register
  spills. Program-scoped allocations: Allocations that live beyond the scope of a single HLO instruction. These are
  usually HLO temporaries and intermediate results that are inputs and/or outputs of HLO instructions." The limit is
  "controlled globally for the entire program via the flag --xla_tpu_scoped_vmem_limit_kib and per custom kernel via
  vmem_limit_bytes param".
- Public v6e values (all multi-chip runs, but the flag itself is per-chip):
  levanter docs/Getting-Started-TPU-VM.md: "v6e: # If you're lucky enough to have a v6e, you can set the following,
  which is pretty important for performance LIBTPU_INIT_ARGS: "--xla_tpu_scoped_vmem_limit_kib=98304"";
  levanter AGENTS.md "### TPU VMEM flags for kernel tuning ... `v6e`: `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`";
  marin lib/fray types.py `elif self.variant.startswith("v6e-"): defaults["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=98304"`;
  apple/axlearn v6e: `# Improved performance for v6e. xla_tpu_scoped_vmem_limit_kib=98304,`;
  salmanmohammadi/nanocode (pure-JAX nanochat on v6e-8 with splash): `export LIBTPU_INIT_ARGS="--xla_tpu_use_bundle_aware_cost_model_for_fusions=false --xla_tpu_scoped_vmem_limit_kib=65536"`;
  EasyDeL _linkup.py "v6e": "--xla_tpu_scoped_vmem_limit_kib=98304 ...";
  tpu_performance_autoresearch_wiki (Qwen3-8B JAX on v6e-8): "`xla_tpu_scoped_vmem_limit_kib=81920` | required for
  bs>=2 + splash (v005->v007) | default 32 MB < 33.84 MB dkv tile"; later frontier "vmem100352"; v270/v275 moved it
  by +-512 KiB with no gain ("Keep scoped VMEM `100352`").
- The OPPOSITE direction (lower is faster) is public too: vllm-project/vllm-torchtpu issue #3062 (2026-11-10):
  "#1790 ... On a tpu7x decode server, lowering `--xla_tpu_scoped_vmem_limit_kib` from 64 MiB to the XLA default of
  32 MiB cut TPOT by about 40% (8.99 to 5.03 ms at concurrency 1). The PR attributes this to XLA having VMEM left to
  prefetch weights and overlap work." Same issue: "a buffer that it prefetches into VMEM before such a call ... can
  only stay there across the call if it fits in the VMEM the call leaves free."
- Interaction with Pallas (conflicting public statements):
  jax 0.11.2 pallas/mosaic/core.py CompilerParams: "vmem_limit_bytes: Overrides the default VMEM limit for a kernel.
  Note that this must be used in conjunction with the --xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes."
  google/qwix fused_fp8_matmul/forward_kernel.py: "The `--xla_tpu_scoped_vmem_limit_kib` flag governs XLA's own fusions
  and does not reach a Mosaic custom call, which otherwise gets a 32 MiB default."
  jax 0.11.2 splash_attention_kernel.py passes only `pltpu.CompilerParams(dimension_semantics=(...))` (no
  vmem_limit_bytes, no cost_estimate) -> the run's splash calls carry no scoped_memory_configs (tpu_custom_call.py
  writes them only `if self.vmem_limit_bytes is not None`).
- RUN RELEVANCE: HIGH. The run's 1024/1024/512 splash tiles already fit the 32 MiB default. The flag moves the split
  between per-op scratch for XLA's glue fusions and the VMEM that MSA can use to keep/prefetch tensors across ops.
  The measured "4-row micro-batch 10% faster" fits the MSA story (at 4x1024x768 a bf16 activation is 6 MiB and can
  live in VMEM). Sweep BOTH directions: 16384, 24576, (32768 default), 65536, 98304. Check each on the CPU
  compile-only host first: a too-small limit fails at compile time ("Ran out of memory in memory space vmem").

### 1b. `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments` (true/false; default auto)
- libtpu strings: "TpuCustomCallScopedVmemAdjuster setting scoped vmem limit to", "We are lowering the scoped Vmem for".
- autoresearch wiki v072 (Qwen3 v6e-8, splash): "The flagfile leaves `--xla_tpu_enable_tpu_custom_call_scoped_vmem_
  adjustments=auto`; forcing it to `true` may let the scoped VMEM adjuster make stronger custom-call-specific
  scheduling choices" -> result 50,793 vs 50,790 tok/s ("tiny improvement", noise); v129 `=false` "refuted/tie".
  Also in tpu-recipes ironwood wan2.1 recipe `=true`. Relevance LOW-MEDIUM (one restart A/B with the VMEM sweep).

### 1c. `--xla_tpu_vmem_scavenging_mode` (NONE | SAFE_IFF_KERNEL_REQUIRED | SAFE | AGGRESSIVE)
- ml-workload-debugging-wiki: "`--xla_tpu_vmem_scavenging_mode`: Allows Fusion+MSA to recuperate excess scoped memory
  for deeply nested fusions (`"AGGRESSIVE"` or `"SAFE"`)."
- torch_tpu compilation.cc default env override: `{"xla_tpu_vmem_scavenging_mode", std::string("SAFE_IFF_KERNEL_REQUIRED")}`.
- enum values from vLLM tpu-inference .buildkite/xla_autotune/flags.txt: `=NONE`, `=SAFE_IFF_KERNEL_REQUIRED`, `=SAFE`,
  `=AGGRESSIVE`. Microbenchmarks pin `NONE` (AI-Hypercomputer/accelerator-microbenchmarks op_flags.yaml gemm,
  ayaka14732 latency scripts) to get clean timings. Default: unset message in the public dump (= platform auto).
  Relevance MEDIUM (directly about VMEM use by fusions; untested publicly for training speed).

### 1d. `--xla_tpu_control_large_2nd_minor_layout_for_x16=true` (AutoProto)
- openxla xla/benchmarks/README.md (v6e / 7x microbenchmarks): "It may also be necessary to turn on large 2nd minor for
  bf16 inputs in some cases, or the cost model will underestimate the required VMEM ...
  LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=32768 --xla_tpu_control_large_2nd_minor_layout_for_x16=true"".
  Used together with explicit `memory_utils.with_large_2nd_minor_layout(lhs)` in dense_matmul_lib.py and in
  jax tests/overlap_test.py compiler opts. Dump defaults: `xla_tpu_enable_large_2nd_minor_layout: true`,
  `xla_tpu_allow_large_2nd_minor_layout_for_x16: false`. Relevance LOW (layout control for user-requested layouts).

-------------------------------------------------------------------------------------------------------------
## 2. Memory-space assignment (MSA: what lives in VMEM vs HBM across ops) - single chip, all documented by OpenXLA

| flag | quote (source) | default | values |
|---|---|---|---|
| `--xla_tpu_msa_inefficient_use_to_copy_ratio` | "The ratio of use bytes to copy bytes for a given allocation site below which we consider the site to be inefficient. This is used while making VMEM placement decisions. A value of 0 would treat all sites as efficient and a value of 1 would require the amount of bytes used at the site to be at least as much as the async copy bytes." (openxla flags_guidance) | 0.5 | [0, 1] |
| `--xla_tpu_async_copy_bandwidth_scaling_factor` | "Scales effective bandwidth for async copies. This is used when making prefetch decisions and deciding which tensors should live in VMEM." (flags_guidance) | 1 | (0, 1]; tpu-recipes ironwood deepseek3 uses 0.5 |
| `--xla_msa_enable_cross_program_prefetch_freeing` | "Enables freeing optimization for cross-program-prefetched buffers." (flags_guidance) | enabled | enabled/disabled |
| `--xla_msa_enable=false` | "Disables Memory Space Assignment (MSA) specific optimizations." (ml-workload-debugging-wiki, numerics isolation) | AUTO (dump) | diagnostic only |
| `--xla_msa_enable_sync_copy_replacement=false` | "Forces all copies to use synchronous copy." (same wiki) | unset | diagnostic only |
| `--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers` | set `=SQRT` in GKE v6e JetStream sample | "SQRT" (dump) | other values unverified |
| `--xla_msa_max_outstanding_prefetches`, `--xla_msa_max_outstanding_evictions`, `--xla_msa_max_repacks`, `--xla_msa_max_retries`, `--xla_msa_preferred_overlap_to_async_copy_ratio`, `--xla_msa_cost_model_options` (ALL/OUTPUT_FUSION/CONV_LOWERABLE/LOOP_FUSION) | names+types only: vLLM tpu-inference xla_autotune/flags.txt ("Full OFAT candidate set generated from .../tpu_comp_env_tuning.json") | per-gen defaults in dump (`xla_vf_vmem_max_outstanding_prefetches: 40`, gf the same) | semantics unverified |

Relevance MEDIUM: for a step that is half memory-bound glue, MSA decides whether intermediates of a 4/16-row
micro-batch stay in VMEM. A/B ideas: inefficient_use_to_copy_ratio 0.25 or 0 (more sites eligible), and
async_copy_bandwidth_scaling_factor 0.5 (MSA assumes slower copies -> more conservative prefetching; opposite
direction). Expect small effects; one compile each via compiler_options.

-------------------------------------------------------------------------------------------------------------
## 3. Fusion knobs (single chip; documented)

| flag | quote | default | public v6e use |
|---|---|---|---|
| `--xla_tpu_rwb_fusion` | "This flag enables reduce+broadcast type of fusions, and may decrease memory usage." (flags_guidance; Suggested `false`) | true | false in: tpu-recipes inference/trillium/MaxDiffusion (`LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=131072"`), GKE v6e 2x4 JetStream sample, google/saxml lm_cloud.py `'xla_tpu_rwb_fusion': 'False'`; TPU agent suite lists it as an OOM-recovery knob |
| `--xla_tpu_enable_dot_strength_reduction` | "This flag rewrites non-compute intensive dots as multiply + reduce operations." | true | saxml gemma.py: "# Need to add --extra_flags=xla_tpu_enable_dot_strength_reduction=false" (Gemma7B) |
| `--xla_tpu_dot_dot_fusion` | "fuses a producer-dot operation with a consumer-dot operation. On doing so, the producer-dot's output is not manifested in slow/main memory" | true | GKE v6e sample true |
| `--xla_jf_enable_multi_output_fusion` | "enables fusions that fuse multiple consumers (i.e. the resultant fusion will have multiple outputs)" | true | |
| `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false` | (no doc) | AUTO (dump) | nanocode v6e-8 uses false; autoresearch v6e-8: llama3 exp42 "-2.6 %. Doesn't apply to our shape."; qwen3 v019 compile OOM ("HLO temp 33.98 GiB ... the flag change materially worsened scheduling/memory placement") |
| `--xla_tpu_enable_copy_fusion`, `--xla_tpu_copy_fusion_pad_unpad_ratio`, `--xla_tpu_vector_load_fusion_window`, `--xla_tpu_vector_store_fusion_window`, `--xla_tpu_enable_multi_level_nested_loop_fusion`, `--xla_tpu_enable_experimental_fusion_cost_model` | no description anywhere public; GKE v6e 2x4 sample values: copy_fusion=true, pad_unpad_ratio=0, load window 1024, store window 256, nested_loop_fusion=true, experimental_fusion_cost_model=false | dump: copy_fusion true, pad_unpad 300, windows 256/256, exp. cost model false | the Google wiki names `xla_tpu_dot_dot_fusion` and `xla_tpu_enable_multi_level_nested_loop_fusion` only as isolation knobs |

Relevance MEDIUM for rwb_fusion (rmsnorm/softmax-statistic style reduce+broadcast are exactly the run's glue);
LOW-MEDIUM for the undocumented window/copy-fusion knobs (only an OFAT A/B can tell).

-------------------------------------------------------------------------------------------------------------
## 4. Scheduler knobs (single chip but mostly about async collectives)

- `--xla_tpu_enable_latency_hiding_scheduler` (default true): "allows us to perform asynchronous collective instead of
  synchronous ones. Disabling it reduces memory usage at the cost of losing the performance gains" (flags_guidance).
- `--xla_latency_hiding_scheduler_rerun=N` (default 1): flags_guidance "incrementally reducing the memory limit
  allocated for scheduling with each "rerun""; AXLearn v6e sets 2 with comment "... Sometimes when the scheduler thinks
  it goes out memory, it may not actually happen ..."; pytorch/xla __init__.py sets 1: "rerun the latency hidding
  scheduler if the default shared memory limit 95% leads to OOM ... Shared memory limit refers to
  --xla_tpu_scheduler_percent_shared_memory_limit. Lower shared memory limit means less communiation and computation
  overlapping". The run uses ~4 GiB of 31 GiB -> reruns never trigger. Relevance NONE.
- `--xla_memory_scheduler` (kDefault/kList/kDfs/kPostOrder/kBrkga per flags_guidance; vLLM list spells DEFAULT, LIST,
  DFS, POST_ORDER, BRKGA, BFS, ILP, ...): "minimize memory consumption ... at the cost of longer compilation time".
  Memory-only. Relevance LOW.
- `--xla_tpu_enable_all_experimental_scheduler_features=true`: AXLearn v6e true (under "# Host offloading flags");
  accelerator-microbenchmarks gemm flags true; autoresearch v6e-8 Qwen3: "HOST_OFFLOAD scheduler bundle | refuted (v010):
  flat at seq2048/bs2 (+40% compile)". DeepMind simply decode flags set it 'false'. Relevance LOW on one chip (no
  collectives; +compile time).
- `--xla_tpu_enable_scheduler_memory_pressure_tracking` (AUTO default; AXLearn: "Flag to enable memory tracking
  scheduling. The default AUTO only enables it in some situations"), `--xla_lhs_prioritize_async_depth_over_stall`
  (AXLearn: "more aggressive scheduling for async ops, such as pushing the async start to the beginning of the loop
  body"), `--xla_tpu_aggressive_opt_barrier_removal` (AXLearn: "aggressive removal of opt-barriers"; RISK: remat
  relies on optimization barriers). Relevance LOW; avoid the opt-barrier one.

-------------------------------------------------------------------------------------------------------------
## 5. SparseCore gather offload (single chip! v6e has 2 SparseCores)

- jax issue #40553 (2026-09-08, v7x, jax 0.10.2 -> 0.11.1): "0.11.1's libtpu offloads gathers in the bs=1 decode graph
  to SparseCore, and for these shapes the offload is a large net loss ... Setting `LIBTPU_INIT_ARGS="--xla_tpu_offload_
  gather_to_sparsecore=false --xla_tpu_offload_all_supported_gathers_to_sparsecore=false"` makes both ops disappear from
  the trace and recovers decode step time completely" and "public IR dumps show `xla_tpu_offload_gather_to_sparsecore`
  defaults to `auto`"; the offload "is genuinely profitable for large gathers" (prefill +21% slower without it).
- jax 0.11.2 tpu_info.py v6e: `sparse_core=SparseCoreInfo(num_cores=2, num_subcores=16, ...)`; the wiki: "`2/chip` on v6e".
- libtpu 0.0.50 op names to grep in the run's profile / optimized HLO: `gather_offload_async_start`,
  `gather_offload_async_done`, `gather_offload_custom_fusion`. Related flags present: `xla_tpu_enable_offloading_
  gather_to_sparsecore`, `xla_tpu_enable_offloading_scatter_to_sparsecore`, `xla_tpu_offload_gather_to_sc_min_num_indices`.
- RUN RELEVANCE: MEDIUM - cheap check. In the repo every vocabulary-table lookup's FORWARD is a gather
  (bes/model/tables.py `big_lookup`: mode "onehot" (the default, bes/config.py `table_grad: str = "onehot"`) calls
  `_gather_sum(table, idx, w)` forward and uses the one-hot matmul only for the gradient); model.py calls big_lookup
  for the token embedding and three more tables, inside every micro-batch. If the profile/optimized HLO shows
  gather_offload ops, A/B `xla_tpu_offload_gather_to_sparsecore=false` (+ `xla_tpu_offload_all_supported_gathers_to_
  sparsecore=false`). Changes placement only, not values (a gather is exact).

-------------------------------------------------------------------------------------------------------------
## 6. Diagnostics that apply to one chip (no speed change; use them to find the glue)

- Profile inside the splash custom call: AI-Hypercomputer/accelerator-agents wiki (XProf "Custom Call Profiling"):
  `LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"` -> "a new
  **LLO utilization** line appears in the Trace Viewer per TPU core / device executing the custom call"; "Flags increase
  profile size and can slightly impact collection-time performance".
- `--xla_enable_transpose_trace` - jax Pallas debugging.md "Useful Command line flags": "Enable trace markers in XProf:
  `--xla_enable_transpose_trace`" (pltpu.trace markers inside kernels). Sits next to xla_enable_mxu_trace /
  xla_enable_hlo_trace in libtpu's trace-flag list; default false in dumps. NOT an optimisation.
- Same doc: "OOB Checks: `--xla_mosaic_on_device_checks=bounds`", "Poison VMEM allocations:
  `--xla_jf_poison_vmem_allocations=true`", "Dump Mosaic: `--xla_mosaic_dump_to=<directory>`"; `pl.debug_print` needs
  `compiler_options={'xla_tpu_enable_log_recorder': 'true'}`.
- Dumps: XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text" (HLO; the dump's `*tpu_comp_env.txt` = "Exact compiler
  flags, build targets, and environment settings" -> read the run's libtpu 0.0.50 v6e defaults there, also from the CPU
  compile-only host); LLO: LIBTPU_INIT_ARGS="--xla_jf_dump_to=DIR" (+ `--xla_jf_dump_category_filter=...`,
  `--xla_tpu_instructions_on_separate_lines=true`). PSAL issue #69 gives a full no-TPU v6e:1x1 dump recipe
  (`get_topology_desc(platform="tpu", topology_name="v6e:1x1", chips_per_host_bounds=(1, 1, 1))`).
- jax issue #41166: "TPU: `Compiled.memory_analysis().temp_size_in_bytes` counts HBM fragmentation twice" (closed).

-------------------------------------------------------------------------------------------------------------
## 7. JAX-level knobs (jax 0.11.2, read from the wheel)

- `jax_optimization_level` (enum UNKNOWN/O0/O1/O2/O3, default 'UNKNOWN'): help "The degree to which the compiler should
  optimize for execution time"; env `JAX_OPTIMIZATION_LEVEL`; goes to `build_options.optimization_level`. torch_tpu uses
  O1 for eager mode ("Use O1 for the eager mode and the default optimization level (O2) for the torch.compile mode").
  Whether O3 does anything more on TPU: unverified. Relevance LOW-MEDIUM (cheap A/B).
- `jax_memory_fitting_level` (default 'O2'): "The degree to which the compiler should attempt to make the program fit in memory".
- `jax_compiler_enable_remat_pass` (default True): "Config to enable / disable the rematerialization HLO pass ... likely to
  get better results manually with jax.checkpoint". Program is far below HBM -> no effect expected.
- `jax_enable_pgle` / `jax_pgle_profiling_runs` (default False / 3): "modules will be recompiled after running specified
  number times with collected data provided to the profile guided latency estimator." Documented for GPU; TPU effect
  unverified.
- Auto layouts (JAX API, not a flag): jax-ml/jax-llm-examples gemma4 model.py `jax.jit(forward,
  in_shardings=Format(Layout.AUTO), out_shardings=Format(Layout.AUTO), donate_argnames=("cache",))` then
  `.compile().input_formats` and loads weights/cache in those formats (`from jax.experimental.layout import Format,
  Layout`, present in 0.11.2). Lets XLA pick parameter/state layouts so it does not relayout (transpose) them inside
  every step. Same values. Relevance MEDIUM for a donated train state; needs code to place the state in those formats.

-------------------------------------------------------------------------------------------------------------
## 8. Library defaults worth knowing (what others ship for TPU)

- torch_tpu (Google, compilation.cc) env overrides: `xla_jf_program_hbm_alignment_in_kib=1` ("Aligning program HBM
  allocations to 1 KiB instead of 16 KiB avoids 16x alignment fragmentation across hundreds or thousands of async
  copies"), `xla_tpu_autofdo=false` ("Avoid overheads associated with fingerprinting XLA inputs to attempt looking up
  feedback-directed optimization profile data that is unlikely to be available." - compile-time only),
  `xla_tpu_enable_flash_attention=false` ("disable XLA's HLO FlashAttention/online-softmax pattern rewrite" - only matters
  for XLA-dense attention; the run uses the splash custom call), vmem_scavenging SAFE_IFF_KERNEL_REQUIRED.
- AXLearn (Apple) v6e: VMEM 98304, LHS rerun 2, host_transfer_overlap_limit 24 (host offload only), scheduler bundle,
  collective fusion (multi-chip), xla_tpu_sdc_replicate_llo for "TPU v6 lite" only in the SDC-checker path.
- pytorch/xla: xla_tpu_use_enhanced_launch_barrier=false (hack), xla_latency_hiding_scheduler_rerun=1,
  xla_tpu_prefer_async_allgather_to_allreduce=true.
- GKE kubernetes-engine-samples jetstream-pathways-disagg-llama-2-70b-2-2x4.yaml (tpu-v6e-slice 2x4, inference):
  a 40-flag single-chip compiler bundle incl. `--xla_tpu_copy_elision_analysis_allowance=150000`,
  `--xla_tpu_copy_insertion_use_region_analysis_limit=10000`, `--xla_tpu_licm_size_inflation_ratio=1`,
  `--xla_jf_rematerialization_percent_shared_memory_limit=100`, `--xla_tpu_prefetch_interval_picker_size_override=6000000`,
  `--xla_tpu_memory_bound_loop_optimizer_options=enabled:true`, `--xla_tpu_enable_flash_attention=true` ... (same family as
  MaxText's trillium select_xla_flags). Dump defaults for comparison: copy_insertion_use_region_analysis false,
  _limit 1000, copy_elision allowance 150000, licm_analysis_allowance 100000, prefetch_interval_picker -1,
  memory_bound_loop_optimizer_options {} (off). Undocumented; inference-tuned.
- vLLM tpu-inference autotuner: 562 flags / 854 OFAT trials (names + enum values only), incl. while-loop knobs
  `--xla_while_loop_unroll_count`, `--xla_tpu_enable_pipelined_loop_unrolling=true`,
  `--xla_tpu_memory_space_assignment_while_execution_count`, `--xla_tpu_sliced_prefetch_max_slices`.

-------------------------------------------------------------------------------------------------------------
## 9. Spelling check (libtpu 0.0.50)

All of these occur as whole tokens in scratchpad/dl_libtpu0050/xla_tokens_counts.txt (strings of libtpu.so 0.0.50):
xla_tpu_scoped_vmem_limit_kib, xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments, xla_tpu_vmem_scavenging_mode,
xla_tpu_control_large_2nd_minor_layout_for_x16, xla_tpu_msa_inefficient_use_to_copy_ratio,
xla_tpu_async_copy_bandwidth_scaling_factor, xla_msa_enable_cross_program_prefetch_freeing, xla_msa_enable,
xla_msa_enable_sync_copy_replacement, xla_msa_max_outstanding_prefetches, xla_msa_cost_model_options,
xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers, xla_tpu_rwb_fusion, xla_tpu_enable_dot_strength_reduction,
xla_tpu_dot_dot_fusion, xla_jf_enable_multi_output_fusion, xla_tpu_use_bundle_aware_cost_model_for_fusions,
xla_tpu_enable_copy_fusion, xla_tpu_copy_fusion_pad_unpad_ratio, xla_tpu_vector_load_fusion_window,
xla_tpu_vector_store_fusion_window, xla_tpu_enable_multi_level_nested_loop_fusion, xla_tpu_enable_experimental_fusion_cost_model,
xla_tpu_enable_latency_hiding_scheduler, xla_latency_hiding_scheduler_rerun, xla_memory_scheduler,
xla_tpu_enable_all_experimental_scheduler_features, xla_tpu_enable_scheduler_memory_pressure_tracking,
xla_tpu_aggressive_opt_barrier_removal, xla_lhs_prioritize_async_depth_over_stall, xla_tpu_scheduler_percent_shared_memory_limit,
xla_tpu_offload_gather_to_sparsecore, xla_tpu_offload_all_supported_gathers_to_sparsecore,
xla_tpu_enable_offloading_gather_to_sparsecore, xla_tpu_enable_offloading_scatter_to_sparsecore,
xla_enable_custom_call_region_trace, xla_xprof_register_llo_debug_info, xla_enable_transpose_trace, xla_tpu_enable_log_recorder,
xla_mosaic_on_device_checks, xla_jf_poison_vmem_allocations, xla_mosaic_dump_to, xla_jf_dump_to, xla_jf_dump_category_filter,
xla_tpu_instructions_on_separate_lines, xla_jf_bounds_check, xla_tpu_use_enhanced_launch_barrier, xla_tpu_autofdo,
xla_jf_program_hbm_alignment_in_kib, xla_tpu_enable_flash_attention, xla_tpu_memory_bound_loop_optimizer_options,
xla_tpu_copy_insertion_use_region_analysis, xla_tpu_copy_insertion_use_region_analysis_limit,
xla_tpu_copy_elision_analysis_allowance, xla_tpu_licm_analysis_allowance, xla_tpu_licm_size_inflation_ratio,
xla_tpu_prefetch_interval_picker_size_override, xla_jf_rematerialization_percent_shared_memory_limit,
xla_tpu_impure_enable_packed_bf16_math_ops, xla_allow_excess_precision, xla_tpu_enable_pipelined_loop_unrolling,
xla_while_loop_unroll_count, xla_tpu_scavenge_vmem_for_fusions, xla_jf_tune_large_vmem, xla_tpu_host_transfer_overlap_limit,
xla_tpu_enable_large_2nd_minor_layout, xla_tpu_allow_large_2nd_minor_layout_for_x16, xla_tpu_msa_reduce_scoped_vmem_limit,
xla_tpu_fusion_cost_model_options, xla_tpu_latency_hiding_scheduler_cost_model_options, xla_tpu_accumulate_into_mrb,
xla_tpu_dvfs_p_state, xla_tpu_enable_sdc_checker, xla_tpu_sdc_replicate_llo, xla_dump_to, xla_dump_hlo_as_text.
No help text is embedded (AbslFlagHelpGenFor* symbols exist, but phrases from the OpenXLA guide are absent).
Per-generation MSA families in 0.0.50: xla_jf_vmem_* (12), xla_vf_vmem_* (31), xla_gf_vmem_* (31); which family v6e
("Ghostlite" in libtpu strings) reads is not public.

-------------------------------------------------------------------------------------------------------------
## 10. Do NOT use (change numerics = change the mathematics)

- `--xla_tpu_impure_enable_packed_bf16_math_ops` (wiki: "Controls whether `fusion_emitter` lowers loop fusions with `bf16`
  math ops using hardware-native BF16 operations"), `--xla_allow_excess_precision` (wiki: "permits XLA to use higher
  precision for intermediate calculations"), `--xla_tpu_pre_fusion_remat=true` (openxla error_1000: "may **potentially
  impact numerical stability**"), `--xla_tpu_spmd_rng_bit_generator_unsafe` (different random draws), and the vLLM OFAT
  entries xla_tpu_auto_reduce_precision, xla_jf_accumulation_reassociation, xla_tpu_enable_mixed_precision_add_transform,
  xla_tpu_optimize_bf16_math, xla_tpu_bf16_emission_mode, xla_jf_bf16_propagation, xla_tpu_use_associative_reordering,
  xla_tpu_enable_int4_rewrite, xla_tpu_experimental_enable_dynamic_int8_quantization.
- `--xla_tpu_aggressive_opt_barrier_removal` (opt barriers implement remat; risk of different memory/schedule, maybe worse).

-------------------------------------------------------------------------------------------------------------
## 11. Multi-chip only (out of scope, names only)

xla_tpu_enable_async_collective_fusion(_fuse_all_gather/_fuse_all_reduce/_fuse_reduce_scatter/_multiple_steps),
xla_tpu_overlap_compute_collective_tc, xla_enable_async_all_gather, xla_enable_async_all_reduce,
xla_enable_async_collective_permute, xla_tpu_enable_async_all_to_all, xla_tpu_enable_data_parallel_all_reduce_opt,
xla_tpu_data_parallel_opt_different_sized_ops, xla_tpu_enable_ag_backward_pipelining, xla_tpu_enable_ici_ag_pipelining,
xla_should_allow_loop_variant_parameter_in_chain, xla_should_add_loop_invariant_op_in_chain,
xla_jf_spmd_threshold_for_windowed_einsum_mib, xla_tpu_enable_windowed_einsum_for_all_gather/_reduce_scatter,
xla_tpu_enable_sparse_core_collective_offload_*, xla_tpu_use_single_sparse_core_for_all_gather_offload,
xla_tpu_enable_concurrent_sparse_core_offloading, xla_tpu_use_tc_device_shape_on_sc, xla_sc_enable_instruction_fusion,
xla_sc_disable_megacore_partitioning, xla_tpu_use_minor_sharding_for_major_trivial_input,
xla_tpu_relayout_group_size_threshold_for_reduce_scatter, xla_tpu_assign_all_reduce_scatter_layout,
xla_tpu_enable_megascale_barrier, xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction,
xla_tpu_host_dcn_reduce_max_sinking_limit, megascale_*, xla_max_concurrent_host_send_recv,
xla_tpu_prefer_async_allgather_to_allreduce, xla_tpu_perform_spmd_cse_prevention, xla_tpu_megacore_fusion_allow_ags,
xla_*_latency_bound_threshold_in_bytes, xla_jf_auto_cross_replica_sharding, xla_tpu_spmd_threshold_for_allgather_cse,
xla_tpu_enable_net_router_in_all_gather, xla_tpu_sparse_core_*_latency_multiplier, TPU_MEGACORE (v4 only),
xla_tpu_use_enhanced_launch_barrier (launch barrier; JAX sets it anyway). Also not useful on v6e:
xla_tpu_dvfs_p_state (v6e has no P-states per openxla platform_info, see sibling file), xla_tpu_accumulate_into_mrb
(only seen in TPU7x-era microbenchmark flags).

-------------------------------------------------------------------------------------------------------------
## 12. Dead ends

- Cloud TPU docs (performance guide, v6e pages) do not document LIBTPU_INIT_ARGS or any xla_tpu_* flag; the guide only
  documents TPU_PREMAPPED_BUFFER_SIZE / TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES (host<->device DMA pool; irrelevant
  at the run's input size) and "MXU size is 256 x 256 for v6e".
- google-research (big_vision, t5x, scenic, vision_transformer), google-deepmind (except simply), keras-team, kithara,
  tensorflow, flax: no LIBTPU_INIT_ARGS / xla_tpu_ uses in public code search (swirl-lm only documents the mechanism;
  paxml/EasyLM use v4-era collective flags).
- Developers blog "A Developer's Guide to Debugging JAX on Cloud TPUs" (2026-01-05): only TPU_VMODULE / TPU_MIN_LOG_LEVEL /
  TPU_STDERR_LOG_LEVEL / TF_CPP_MIN_LOG_LEVEL.
- No public single-chip v6e TRAINING measurement of any MSA/fusion flag; all v6e A/Bs found are v6e-8 multi-chip
  (autoresearch wiki, nanocode) or inference.
- No help text in libtpu 0.0.50 strings; MSA large-buffer scaling alternatives to "SQRT" not found.
- `--xla_tpu_enable_pipelined_loop_unrolling=true` on v6e-8 (autoresearch v032): "The flag was a silent no-op on our
  loop structure: the optimized module has 6 `while(` calls - identical to v027"; the auto-unroll it relies on
  ("unrolls by a factor of 2 if there is any collective present within a while loop") needs collectives -> nothing to
  gain on one chip. Unroll the run's scans in JAX (lax.scan unroll=) if wanted, not via this flag.
- `--xla_tpu_enable_all_experimental_scheduler_features` / HOST_OFFLOAD bundle on v6e-8 Qwen3 at seq 2048: "flat ...
  (+40% compile)"; its gains elsewhere came from collective overlap at seq 8192.

-------------------------------------------------------------------------------------------------------------
## 13. Suggested order for the run (each needs a compile; use compiler_options to batch them in one process)

1. Read the run's own defaults: one HLO dump on the CPU compile host (`--xla_dump_to`) -> `*.tpu_comp_env.txt`.
   Grep the optimized HLO / a profile for `gather_offload` (section 5) and for big `copy`/`transpose` fusions.
2. VMEM split sweep: xla_tpu_scoped_vmem_limit_kib in {16384, 24576, 32768(default), 65536, 98304}, compile-check on CPU.
3. If gather_offload ops exist: xla_tpu_offload_gather_to_sparsecore=false (+ xla_tpu_offload_all_supported_gathers_to_sparsecore=false).
4. xla_tpu_rwb_fusion=false; xla_tpu_vmem_scavenging_mode=SAFE / AGGRESSIVE; MSA ratio 0.25 / 0.
5. jax_optimization_level=O3 (cheap), auto layouts for the donated state (needs code).
