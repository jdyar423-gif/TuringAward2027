# JAX-side and XLA-side knobs for one TPU v6e chip (JAX 0.11.2, libtpu 0.0.50)

Angle: settings that change how the same program is compiled or executed (not its math).
Primary sources are the JAX sources and docs at tag `jax-v0.11.2` (downloaded raw from
https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/...) and the OpenXLA docs/sources.
Quotes are copied verbatim from those files.

Repository context read (read-only): `bes/model/attention_splash.py`, `bes/model/controller.py`
(remat: `jax.checkpoint(step, prevent_cse=False)` in a `lax.scan` over loops, policy
`keep_kernel_outputs` = save `pallas_call` outputs), `bes/train/train.py` (`accumulate`: `lax.scan`
over micro-batches of `value_and_grad`; `jax.jit(train_step, donate_argnums=(0,))`),
`scripts/run_24h.py` (`jax_compilation_cache_dir` + `jax_persistent_cache_min_compile_time_secs=1.0`),
`scripts/tpu_flags.py` (LIBTPU_INIT_ARGS variants per process), `docs/TPU_NOTES.md`, `docs/CHANGES.md`
section 19 (measured: remat on is faster than off; `remat_keep=kernel` slower, 161.2 vs 158.3 ms;
micro-batch 16 rows +8%; split into 4-row pieces +10%).

---------------------------------------------------------------------------------------------------

## 1. jax.config options (exact definitions in jax/_src/config.py and jax/_src/compiler.py, 0.11.2)

### 1.1 `jax_default_matmul_precision` (env `JAX_DEFAULT_MATMUL_PRECISION`)
Source: https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/jax/_src/config.py
```
default_matmul_precision = optional_enum_state(
    name='jax_default_matmul_precision',
    enum_values=[
        # Legacy precision API values
        'default', 'high', 'highest', 'bfloat16', 'tensorfloat32', 'float32',
        # Dot algorithm presets
        'ANY_F8_ANY_F8_F32', 'ANY_F8_ANY_F8_F32_FAST_ACCUM', 'ANY_F8_ANY_F8_ANY',
        'ANY_F8_ANY_F8_ANY_FAST_ACCUM', 'F16_F16_F16', 'F16_F16_F32',
        'BF16_BF16_BF16', 'BF16_BF16_F32', 'BF16_BF16_F32_X3',
        'BF16_BF16_F32_X6', 'BF16_BF16_F32_X9', 'TF32_TF32_F32',
        'TF32_TF32_F32_X3', 'F32_F32_F32', 'F64_F64_F64',
    ],
    default=None,
    help=('Control the default matmul and conv precision for 32bit inputs.\n\n'
          'Some platforms, like TPU, offer configurable precision levels for '
          ...
          "precision at which scalar products are computed. The 'bfloat16' "
          "option is the fastest and least precise; 'float32' is similar to "
          "full float32 precision; 'tensorfloat32' is intermediate.\n\n"
```
TPU meaning, from `jax.lax.Precision` (jax/_src/lax/lax.py, 0.11.2):
```
  This only has an effect on float32 computations, and does not affect the
  input/output datatypes. Members are:

  DEFAULT:
    Fastest mode, but least accurate. On TPU: performs float32 computations in
    bfloat16. ...
  HIGH:
    Slower but more accurate. On TPU: performs float32 computations in 3
    bfloat16 passes. ...
  HIGHEST:
    Slowest but most accurate. On TPU: performs float32 computations in 6
    bfloat16. Aliases: ``'highest'``. On GPU: uses float32.
```
and the string aliases: `'highest'`/`'float32'` -> HIGHEST, `'high'`/`'bfloat16_3x'`/`'tensorfloat32'`
-> HIGH, `'default'`/`'bfloat16'`/`'fastest'` -> DEFAULT.
Single-chip verdict: the training run's matmul operands are already bf16 with f32 accumulation, so
the option only affects float32-operand dots (e.g. f32 optimizer matmuls, any f32 einsum). The
default (None == DEFAULT == one bf16 pass on TPU) is already the fastest. Changing it would CHANGE
numerics of f32 dots; it is not a speed knob for this run. Keep unset. (Generation: generic TPU text,
not v6e-specific.) Note `include_in_jit_key=True`: changing it recompiles and changes the
persistent-cache key.

### 1.2 `jax_enable_x64` (env `JAX_ENABLE_X64`), default False
Keep False. Turning it on changes dtypes (math) and makes 64-bit ops that TPU emulates. Not a knob.

### 1.3 `jax_disable_most_optimizations` (env `JAX_DISABLE_MOST_OPTIMIZATIONS`), default False
Defined in jax/_src/compiler.py (0.11.2), not config.py:
```
_DISABLE_MOST_OPTIMIZATIONS = config.bool_flag(
    'jax_disable_most_optimizations',
    config.bool_env('JAX_DISABLE_MOST_OPTIMIZATIONS', False),
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')
...
  if _DISABLE_MOST_OPTIMIZATIONS.value:
    debug_options.xla_backend_optimization_level = 0
    debug_options.xla_llvm_disable_expensive_passes = True
    debug_options.xla_test_all_input_layouts = False
```
Verdict: a compile-time-saving debug switch; makes the program slower. Never for training.

### 1.4 `jax_default_dtype_bits`
NOT PRESENT in jax 0.11.2: grep of jax/_src/config.py, jax/_src/dtypes.py and CHANGELOG.md at tag
jax-v0.11.2 finds no `jax_default_dtype_bits` (I believe it was an old x64-related transitional flag;
not verified). Status: does not exist in this version; nothing to set.

### 1.5 Compiler effort levels: `jax_optimization_level`, `jax_memory_fitting_level` (NEW, relevant)
Source: jax/_src/config.py (0.11.2):
```
optimization_level = enum_state(
    name='jax_optimization_level',
    enum_values=[ 'UNKNOWN', 'O0', 'O1', 'O2', 'O3', ],
    default='UNKNOWN',
    help='The degree to which the compiler should optimize for execution time',
    include_in_jit_key=True
)

memory_fitting_level = enum_state(
    name='jax_memory_fitting_level',
    enum_values=[ 'UNKNOWN', 'O0', 'O1', 'O2', 'O3', ],
    default='O2',
    help=(
        'The degree to which the compiler should attempt to make the program'
        ' fit in memory'
    ),
    include_in_jit_key=True,
)
```
jax/_src/compiler.py (0.11.2):
```
class CompilerEffortLevel(enum.Enum):
  """Effort level enumeration for XLA.

  Used to specify the degree to which the XLA compiler should optimize for
  runtime performance or memory fitting, as described in
  https://openxla.org/xla/effort_levels. Higher effort levels will expend
  more compile time but should yield better results in the respective
  dimension.
  """
  UNKNOWN = 0
  O0 = 9
  O1 = 19
  O2 = 29
  O3 = 39
...
  build_options.optimization_level = CompilerEffortLevel(
      config.optimization_level.value
  ).value
  build_options.memory_fitting_level = CompilerEffortLevel(
      config.memory_fitting_level.value
  ).value
...
    overrides_on_build_options = ["optimization_level", "memory_fitting_level"]
```
CHANGELOG 0.11.1: "The `exec_time_optimization_effort` and `memory_fitting_effort` flags have been
removed in favor of the `EffortLevel` enum." (so the older float flags
`jax_exec_time_optimization_effort` / `jax_memory_fitting_effort` from jax 0.4.36 no longer exist.)
How to set: `jax.config.update("jax_optimization_level", "O3")` / env `JAX_OPTIMIZATION_LEVEL=O3`
before the first compile, or per program `jax.jit(f, compiler_options={"optimization_level": "O3"})`
(compiler.py pops these two names out of `compiler_options` onto build options).
What it does on TPU: see section 3.2 (OpenXLA effort_levels page).

### 1.6 `jax_compiler_enable_remat_pass` (env `JAX_COMPILER_ENABLE_REMAT_PASS`), default True
```
enable_remat_opt_pass = bool_state(
    name='jax_compiler_enable_remat_pass',
    default=True,
    help=('Config to enable / disable the rematerialization HLO pass. '
          'Useful to allow XLA to automatically trade off memory and '
          'compute when encountering OOM errors. However, you are '
          'likely to get better results manually with jax.checkpoint'))
```
compiler.py: `debug_options.xla_disable_hlo_passes = _add_disabled_hlo_pass(debug_options.xla_disable_hlo_passes, "rematerialization")`.
Single chip: the step peaks at ~2 GiB of 31 GiB, so XLA's memory-driven remat pass should have
nothing to do; disabling it is at most a small compile-time saving. Low relevance. (Verify in a dump
that the pass did nothing: `--xla_dump_hlo_pass_re=rematerialization`.)

### 1.7 Persistent compilation cache
Options (config.py 0.11.2 + docs/persistent_compilation_cache.md):
- `jax_compilation_cache_dir` (env `JAX_COMPILATION_CACHE_DIR`), default None: enables the cache.
- `jax_persistent_cache_min_compile_time_secs`, default 1.0: "A computation will only be written to
  the persistent cache if the compilation time is longer than the specified value."
- `jax_persistent_cache_min_entry_size_bytes`, default 0: "-1: disable the size restriction and
  prevent overrides ... 0 to allow for overrides ... > 0: the actual minimum size desired".
- `jax_enable_compilation_cache`, default True.
- `jax_compilation_cache_max_size`, default -1 (unlimited, LRU eviction).
- `jax_persistent_cache_enable_xla_caches`, default `'xla_gpu_per_fusion_autotune_cache_dir'`
  (values `all`, `none`, `xla_gpu_kernel_cache_file`, `xla_gpu_per_fusion_autotune_cache_dir`): GPU
  only, nothing for TPU.
- `jax_explain_cache_misses` (default False), `jax_raise_persistent_cache_errors`,
  `jax_compilation_cache_check_contents`, `jax_compilation_cache_include_metadata_in_key`.
Does it work on TPU? YES. jax/_src/compilation_cache.py (0.11.2):
```
      # Persistent compilation cache only implemented on TPU and GPU and the
      # backend that supports serialization of executables.
      ...
      supported_platforms = ["tpu", "gpu", "cpu", "neuron"]
      ...
      elif (
          backend.platform in supported_platforms
          and getattr(backend, "supports_executable_serialization", True)
      ):
```
Cache key (jax/_src/cache_key.py 0.11.2): computation (unoptimized HLO), jaxlib version, backend
platform+platform_version (libtpu version), "XLA flags", compile_options (serialized, includes
`compiler_options` env overrides), accelerator_config, compression, custom_hook. The XLA-flags hash
reads BOTH env vars:
```
  xla_flags_env_var = os.getenv("XLA_FLAGS")
  if xla_flags_env_var:
    xla_flags.extend(xla_flags_env_var.split())
  libtpu_init_args_env_var = os.getenv("LIBTPU_INIT_ARGS")
  if libtpu_init_args_env_var:
    xla_flags.extend(libtpu_init_args_env_var.split())
```
and EXCLUDES the dump flags (`--xla_dump_to`, `--xla_dump_hlo_as_text`, `--xla_dump_hlo_pass_re`,
`--xla_dump_hlo_as_proto`, `--xla_dump_hlo_as_html`, ... and `--xla_tpu_compiler_variant`).
Consequences for the 3-minute restart:
 (a) any change of LIBTPU_INIT_ARGS / XLA_FLAGS (other than dump flags), of a jit-key config
     (matmul precision, optimization_level, memory_fitting_level), of libtpu or jaxlib version is a
     guaranteed cache miss -> full recompile (chip-host compile time not measured here; the
     compile-only check on 2 CPU cores took 240-413 s per program);
 (b) a restart with identical flags should hit; `jax_explain_cache_misses=True` and
     `JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.lru_cache` show "PERSISTENT COMPILATION CACHE
     MISS for '<module>' with key ..." lines to confirm;
 (c) the repo sets min_compile_time 1.0 (the default) and leaves min_entry_size at 0: fine.
Generation: generic (TPU listed), no v6e-specific statement.

### 1.8 Executable serialization (alternative to the cache): `jax.experimental.serialize_executable`
Source: https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/jax/experimental/serialize_executable.py
```
def serialize(compiled: jax.stages.Compiled):
  """Serializes a compiled binary.

  Because pytrees are not serializable, they are returned so that
  the user can handle them properly.
  """
...
def deserialize_and_load(serialized,
                         in_tree,
                         out_tree,
                         backend: str | xc.Client | None = None,
                         execution_devices: Sequence[xc.Device] | None = None):
```
Limits in 0.11.2: `raise NotImplementedError("serialize_executables with const_args")` (programs
that close over large constants cannot be serialized) and "can't serialize with a closed-over
mutable array ref". Only works with the same libtpu/jaxlib. Use: after the AOT compile of the
train step, `blob, in_tree, out_tree = serialize(compiled)`; pickle trees yourself; on restart
`deserialize_and_load(blob, in_tree, out_tree)`. Saves the full compile on restart independently of
the cache-key logic. Relevance: restart time only (not step time).

### 1.9 Other config options that must stay OFF for speed
`jax_debug_nans`, `jax_debug_infs`, `jax_enable_checks`, `jax_disable_jit`, `jax_log_compiles`
(only logging), `jax_pallas_enable_debug_checks` ("Makes `pl.debug_check` calls run at runtime"),
`jax_pallas_poison_buffers`. All default False.

### 1.10 Options that would change the math (out of bounds for this run)
`jax_default_prng_impl` (default `'threefry2x32'`; `'rbg'`/`'unsafe_rbg'` are faster on TPU but give
different random bits), `jax_threefry_partitionable` (default True; changes bits vs old behaviour),
`jax_enable_x64`, `jax_default_matmul_precision`, `jax_numpy_dtype_promotion`.

### 1.11 Remat/scan "future behaviour" flags present in 0.11.2 (experimental)
`jax_remat3` ("Opts into the future remat behavior"), `jax_scan3` ("future loop behavior"),
`jax_remat_barrier_no_cotangents` ("remat CSE-prevention barriers that don't pin cotangents"),
`jax_log_checkpoint_residuals` (logs saved residuals: diagnostic, useful to see what the
`jax.checkpoint` in the loop body saves). Not speed knobs per se; `jax_remat3`/`jax_scan3` change the
implementation and could change the compiled program; only for an A/B measurement, unverified gain.

---------------------------------------------------------------------------------------------------

## 2. Where flags go on TPU: XLA_FLAGS vs LIBTPU_INIT_ARGS vs per-program `compiler_options`

### 2.1 XLA_FLAGS only knows the open-source DebugOptions; unknown names abort the process
Source: https://raw.githubusercontent.com/openxla/xla/main/xla/debug_options_flags.cc
```
  MakeDebugOptionsFlags(flag_objects, flag_values);
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", *flag_objects);
```
The open-source DebugOptions registers only two `xla_tpu_*` names (grep of the same file):
`"xla_tpu_detect_nan"` ("Trigger error on execution on TPU if a NAN value is detected") and
`"xla_tpu_detect_inf"`. xla/xla.proto's "XLA:TPU options" section is empty
(`// go/keep-sorted start` / `// go/keep-sorted end`). So every other TPU flag
(`--xla_tpu_*`, `--xla_jf_*`, `--xla_msa_*`, `--xla_memory_scheduler`, ...) must go in
`LIBTPU_INIT_ARGS` (or per program in `compiler_options`, 2.3); putting one in XLA_FLAGS kills the
process at backend start. Generic DebugOptions flags (`--xla_dump_*`, `--xla_disable_hlo_passes`,
`--xla_allow_excess_precision`, `--xla_backend_optimization_level`) go in XLA_FLAGS and are forwarded
to libtpu in the compile options' debug_options.
Evidence that dump flags in XLA_FLAGS work on TPU: libtpu 0.0.50 itself (strings of
`libtpu/libtpu.so` from the PyPI wheel libtpu-0.0.50-cp313-cp313-manylinux_2_31_x86_64.whl) contains
"Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and
attach the results."; MaxText (TPU-first) sets `os.environ["XLA_FLAGS"] = raw_keys["dump_hlo_xla_flags"]`
with default `--xla_dump_to={dump_hlo_local_dir} --xla_dump_large_constants`
(src/maxtext/configs/pyconfig_deprecated.py), and its configs README says it "Sets various XLA compiler
flags ... as `LIBTPU_INIT_ARGS`".
JAX doc https://docs.jax.dev/en/latest/xla_flags.html: "Set XLA_FLAGS before importing Jax or other
relevant libraries" (changing it after backend initialisation has no effect). Same for
LIBTPU_INIT_ARGS (read when libtpu initialises; the repo's scripts/tpu_flags.py already starts one
process per variant).

### 2.2 Flag names verified to exist in libtpu 0.0.50 (exact strings in libtpu.so)
`xla_tpu_scoped_vmem_limit_kib`, `xla_tpu_msa_inefficient_use_to_copy_ratio`,
`xla_tpu_async_copy_bandwidth_scaling_factor`, `xla_msa_enable_cross_program_prefetch_freeing`,
`xla_tpu_dot_dot_fusion`, `xla_jf_enable_multi_output_fusion`, `xla_tpu_rwb_fusion`,
`xla_tpu_enable_dot_strength_reduction`, `xla_memory_scheduler`, `xla_latency_hiding_scheduler_rerun`,
`xla_tpu_use_bundle_aware_cost_model_for_fusions`, `xla_tpu_enable_latency_hiding_scheduler`,
`xla_tpu_aggressive_opt_barrier_removal`, `xla_tpu_enable_all_experimental_scheduler_features`,
`xla_tpu_enable_scheduler_memory_pressure_tracking`; and undocumented loop ones
`xla_while_loop_unroll_count`, `xla_tpu_enable_pipelined_loop_unrolling`, `xla_tpu_pre_fusion_remat`,
`xla_tpu_rematerialization_algo`, `xla_disable_while_loop_copies` (no public description found:
semantics UNVERIFIED). Method: `strings -n 6 libtpu.so | grep -x -F <name>` (binary only inspected,
never executed). Help text is not adjacent to the names in the binary, so meanings come from the
OpenXLA guide below.

### 2.3 Per-program TPU flags without a restart: `jax.jit(..., compiler_options={...})`
Source: https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.2/jax/_src/api.py
```
def jit(
  fun: Callable | NotSpecified = NotSpecified(), /, *,
  ...
  inline: bool | Inline = False,
  compiler_options: dict[str, Any] | None = None,
```
CHANGELOG (jax 0.4.36): "`jax.jit` got a new `compiler_options: dict[str, Any]` argument, for passing
compilation options to XLA. For the moment it's undocumented and may be in flux."
Also `Lowered.compile(self, compiler_options: CompilerOptions | None = None, ...)` (jax/_src/stages.py).
Plumbing (jax/_src/compiler.py 0.11.2): `compile_options.env_option_overrides = list(env_options_overrides.items())`
after popping `optimization_level` / `memory_fitting_level` onto build options.
TPU usage in public code, MaxText (https://github.com/AI-Hypercomputer/maxtext, commit be91f97, 2026-11-23),
src/maxtext/utils/train_utils.py:
```
    compiler_options = max_utils.parse_libtpu_flags_to_dict(getattr(config, "compile_xla_flags", ""))
    p_train_step = jax.jit(
        functional_train,
        ...
        donate_argnums=donate_argnums,
        compiler_options=compiler_options or None,
    )
```
with base.yml: `compile_xla_flags: "" # Compiler options e.g. compile_xla_flags="--xla_tpu_num_sparse_cores_for_gather_offloading=1 --xla_tpu_scoped_vmem_limit_kib=65536"`
and `parse_libtpu_flags_to_dict` maps `--key=value` to `{key: value}` (string values, no `--`).
So: `jax.jit(train_step, donate_argnums=(0,), compiler_options={"xla_tpu_scoped_vmem_limit_kib": "65536"})`.
Why it matters here: the start-up benchmark can compile 2-4 flag variants of the SAME step in ONE process
(no 3-minute restart, no TPU re-init) and keep the fastest; each variant gets its own persistent-cache
key because compile_options are hashed (cache_key.py `_hash_serialized_compile_options`).
Caveat: pxla.py (0.11.2) "Only cache executable into `self` if `.compile()` in AOT is called without
specifying compiler_options and device_assignment." (`use_cache = compiler_options is None and
device_assignment is None`): `Lowered.compile(compiler_options=...)` recompiles on each call (the
persistent disk cache still applies); prefer the `jax.jit(compiler_options=...)` form for the
production step. Splash kernels in 0.11.2 set only `dimension_semantics` in their
`pltpu.CompilerParams` (no `vmem_limit_bytes`), so they run under the global scoped-VMEM limit too. Seen on: MaxText generic TPU
(deepseek3 configs are tpu7x); not verified by me on v6e.

---------------------------------------------------------------------------------------------------

## 3. XLA-side knobs honoured by the TPU compiler (single chip)

### 3.1 Effort levels (OpenXLA https://openxla.org/xla/effort_levels, raw docs/effort_levels.md)
"| EFFORT_O0 | Fastest compilation, slowest runtime |
 | EFFORT_O1 | Faster compilation with reasonable runtime |
 | EFFORT_O2 | Strongly prioritize runtime (suitable default for production workloads) |
 | EFFORT_O3 | Expensive or experimental optimizations |"
Memory fitting: "for example, in XLA:TPU, this option controls the degree to which the compiler works
to keep the TPU's high-bandwidth memory (HBM) usage below the HBM capacity".
JAX spellings: `jax_optimization_level` (default 'UNKNOWN' = backend default) and
`jax_memory_fitting_level` (default 'O2'); or per program `compiler_options={"optimization_level": "O3"}`.
Single chip: the step uses ~2 GiB of 31 GiB, so memory fitting has nothing to do (O1 may shave compile
time; not verified). O3 on TPU: effect undocumented -> one cheap A/B (one extra compile).

### 3.2 TPU memory-space-assignment / fusion flags from the OpenXLA flags guide (LIBTPU_INIT_ARGS)
Source: https://openxla.org/xla/flags_guidance (raw https://raw.githubusercontent.com/openxla/xla/main/docs/flags_guidance.md), "Performance Flags", Compute centric:
- `xla_tpu_scoped_vmem_limit_kib`: "This flag sets the amount of scratchpad VMEM available to per op for
  local usage in KiloBytes. Rest of the VMEM is used as buffer space." default `16384`, candidates
  `[4096, VMEM size of the architecture - 1024]`.
- `xla_tpu_async_copy_bandwidth_scaling_factor`: "Scales effective bandwidth for async copies. This is
  used when making prefetch decisions and deciding which tensors should live in VMEM." default `1`,
  candidates `(0, 1]`.
- `xla_tpu_msa_inefficient_use_to_copy_ratio`: "The ratio of use bytes to copy bytes for a given
  allocation site below which we consider the site to be inefficient. This is used while making VMEM
  placement decisions. A value of 0 would treat all sites as efficient and a value of 1 would require
  the amount of bytes used at the site to be at least as much as the async copy bytes." default `0.5`.
- `xla_msa_enable_cross_program_prefetch_freeing`: "Enables freeing optimization for
  cross-program-prefetched buffers." default `enabled`.
- `xla_tpu_dot_dot_fusion` (default true): "fuses a producer-dot operation with a consumer-dot
  operation. On doing so, the producer-dot's output is not manifested in slow/main memory".
- `xla_jf_enable_multi_output_fusion` (default true), `xla_tpu_enable_dot_strength_reduction` (true).
Memory flags (guide says change only on OOM): `xla_tpu_rwb_fusion` (true; "reduce+broadcast type of
fusions"), `xla_memory_scheduler` (`kDefault/kList/kDfs/kPostOrder/kBrkga`),
`xla_latency_hiding_scheduler_rerun` (1), `xla_tpu_enable_latency_hiding_scheduler` (true).
VMEM sizes: JAX Pallas "TPU Hardware Reference" https://docs.jax.dev/en/latest/pallas/tpu/hardware.html:
"| V6E | TPU v6 | 1 | 128 MiB | N/A | 1024 KiB | 34 GB | 1640.0 GB/s |" (TensorCores/chip, VMEM,
CMEM, SMEM, HBM, HBM BW; "all specs listed in the table below are **per TensorCore**").
OpenXLA error page https://openxla.org/xla/errors/error_1001 (scoped VMEM OOM) table:
"| TPU v6e | 127.9MB | 131008 | 134152192 |" and "controlled globally for the entire program via the
flag `--xla_tpu_scoped_vmem_limit_kib`"; per kernel `vmem_limit_bytes`; also suggests
"`pallas.tpu.with_memory_space_constraint` to place kernel inputs or outputs in VMEM".
v6e usage: MaxText benchmarks/xla_flags_library.py `_DENSE_VMEM_LIMIT = 98304`,
`DENSE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_DENSE_VMEM_LIMIT}"`, used by every dense
model in benchmarks/maxtext_trillium_model_configs.py ("Shared Benchmark config for v6e orchestrations"),
and `DISABLE_BUNDLE_AWARE_COST_MODEL = " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"` in
the v6e gpt-3-175b configs.
Relevance to "4-row micro-batches are 10% faster": XLA:TPU's memory space assignment (MSA) decides
which HLO temporaries live in VMEM ("Program-scoped allocations hold HLO temporaries and intermediate
results that pass between instructions", error_1001). The scoped limit is carved out of the same
128 MiB, so RAISING `xla_tpu_scoped_vmem_limit_kib` gives each fusion more scratch but LEAVES LESS for
MSA buffers ("Rest of the VMEM is used as buffer space"); lowering it, or lowering
`xla_tpu_msa_inefficient_use_to_copy_ratio` (more sites treated as worth prefetching), are the
directions that favour keeping glue tensors in VMEM. Direction unverified on this model: A/B.

### 3.3 Generic DebugOptions in XLA_FLAGS (honoured by any XLA backend incl. TPU)
- `--xla_dump_to=DIR`, `--xla_dump_hlo_as_text`, `--xla_dump_hlo_module_re=<regex>`,
  `--xla_dump_hlo_pass_re=<regex>`, `--xla_dump_large_constants`: diagnostics; EXCLUDED from the JAX
  persistent cache key (cache_key.py `xla_flags_to_exclude_from_cache_key`), so a dump run still
  hits/creates the same cache entry. Help text (debug_options_flags.cc): `xla_dump_hlo_pass_re` "If
  specified, dumps HLO before and after optimization passes which match this regular expression".
- No-env alternative: `jax.jit(f).lower(*args).compile().as_text()` (Compiled.as_text in
  jax/_src/stages.py: "A human-readable text representation of this executable"). Scaling book
  https://jax-ml.github.io/scaling-book/profiling : "S(1) tells us this array lives in VMEM. S(0)
  (sometimes omitted) is HBM." -> grep the optimized HLO of the 4-row and 16-row programs for `S(1)`
  to see which glue tensors MSA put in VMEM (direct test of the VMEM hypothesis, no chip time).
- `--xla_disable_hlo_passes=<names>` ("Comma-separated list of hlo passes to be disabled ... a plain
  pass name ('algsimp' ...)"): A/B tool; JAX's `jax_compiler_enable_remat_pass=False` appends
  `rematerialization` to it.
- `--xla_allow_excess_precision` (default `opts.set_xla_allow_excess_precision(true);`): "Allow xla to
  increase the output precision of an instruction." Keep default (true): it lets XLA drop
  f32->bf16->f32 convert pairs inside fusions. JAX already guards remat residuals against it
  (ad_checkpoint.py: "To avoid precision mismatches in fwd and bwd passes due to XLA excess
  precision, insert explicit x = reduce_precision(x, **finfo(x.dtype)) calls on producers of any
  residuals").
- `--xla_backend_optimization_level` (default 3; jax_disable_most_optimizations sets 0): keep.
- `--xla_tpu_detect_nan` / `--xla_tpu_detect_inf` (DebugOptions, default false): debug only; keep off.
- `xla_step_marker_location` (STEP_MARK_AT_ENTRY default): profiler step markers only.
- `deduplicate_hlo` (xla.proto ExecutionOptions/compile option): "deduplicate hlo into function calls to
  reduce binary size. Only works on TPU." Binary size, not speed.

---------------------------------------------------------------------------------------------------

## 4. lax.scan options (jax/_src/lax/control_flow/loops.py, jax-v0.11.2)

Signature: `scan(f, init, xs=None, length=None, reverse=False, unroll: int | bool = 1, _split_transpose: bool = False)`.
- `unroll`: "optional non-negative int or bool specifying, in the underlying operation of the scan
  primitive, how many scan iterations to unroll within a single iteration of a loop. If an integer is
  provided, it determines how many unrolled loop iterations to run within a single rolled iteration of
  the loop. `unroll=0` unrolls the entire loop. If a boolean is provided, it will determine if the loop
  is completely unrolled (i.e. `unroll=True`) or left completely rolled (i.e. `unroll=False`)."
  Docstring also: scan "is lowered to a single WhileOp. That makes it useful for reducing compilation
  times for JIT-compiled functions, since native Python loop constructs in an jit function are
  unrolled, leading to large XLA computations."
  Gradient-checkpointing doc (docs/gradient-checkpointing.md): "Automatic compiler optimizations across
  multiple control flow primitives (for example, across a forward-pass `scan` and the corresponding
  backward-pass `scan`), typically aren't as thorough." and "This scan-over-layers version reduces
  compile times, but by foiling some compiler optimizations it can lead to inefficient computation of
  gradients."
  Single-chip relevance: the 8-step loop scan (2 block applications per step, remat) is the place to
  try `unroll=2` and `unroll=True`: removes the per-iteration while-loop boundary (dynamic-slice of
  stacked per-loop params/residuals, DUS of stacked outputs, carry copies) and lets XLA fuse/schedule
  across iterations. Cost: HLO of the loop body x unroll factor -> compile time roughly x factor for
  that part (measured compile here was 240-413 s on 2 CPU cores). NOT recommended for the 64-step
  micro-batch scan (huge program, little to gain). No public TPU measurement found (GitHub issue #7024
  is GPU: "I set unroll to 2 to get the time down 20% from there"); MaxText's own warning for scanned
  layers: "Scan layers can introduce overhead to checkpointed values that in some configurations is
  slower than not checkpointing at all" (src/maxtext/layers/nnx_decoders.py). Verdict: A/B, unverified.
- `reverse`: semantics only; no speed meaning.
- `_split_transpose`: in jax 0.11.2 it is ACCEPTED BUT UNUSED: the name occurs only in the two
  signatures (`grep -n split_transpose loops.py` -> lines 218 and 251 only). Setting it does nothing.
- `jax_scan3` ("If True, embrace the future of loops."; scan3 implements the carry with mutable refs:
  `carry_refs = [core.new_ref(x) for x in init_flat]`): experimental, changes lowering; unverified.

## 5. Remat (jax.checkpoint) policies and costs (jax/_src/ad_checkpoint.py, 0.11.2)

- `jax.checkpoint(fun, prevent_cse=..., policy=..., static_argnums=...)`; repo already uses
  `prevent_cse=False` inside scan, as the docstring advises: "But in some settings, like when used
  inside a `jax.lax.scan`, this CSE prevention mechanism is unnecessary, in which case `prevent_cse` can
  be set to False." (with True, `_remat_lowering` wraps inputs in `hlo.OptimizationBarrierOp`, and
  "CSE prevention has costs because it can foil other optimizations").
- Policies in `jax.checkpoint_policies` (exact names): `everything_saveable`, `nothing_saveable`,
  `dots_saveable` (= `checkpoint_dots`), `dots_with_no_batch_dims_saveable`
  (= `checkpoint_dots_with_no_batch_dims`), `offload_dot_with_no_batch_dims(src, dst)`,
  `save_anything_except_these_names`, `save_any_names_but_these(*names)`,
  `save_only_these_names(*names)`, `save_from_both_policies(p1, p2)`,
  `save_and_offload_only_these_names(*, names_which_can_be_saved, names_which_can_be_offloaded, offload_src, offload_dst)`,
  classes `SaveOnlyTheseNames`, `SaveAnyNamesButThese`, `SaveAndOffloadOnlyTheseNames`
  (0.11.0: "`jax.checkpoint_policies` is now a submodule"). Names come from
  `jax.ad_checkpoint.checkpoint_name(x, name)`.
- Note on `dots_with_no_batch_dims_saveable`: a projection written as `einsum('btd,de->bte')` has
  no batch dims in dot_general terms, so it IS saved; the saved value is the dot OUTPUT dtype
  (f32 when `preferred_element_type=f32`) -> large f32 residuals, which TPU_NOTES says are
  "several times larger" than recomputing. Prefer naming the bf16 tensors you want kept.
- Diagnostics: `jax.ad_checkpoint.print_saved_residuals(f, *args)` and config
  `jax_log_checkpoint_residuals=True` ("printing what residuals are saved").
- Measured in this repo: `remat_keep=kernel` (save pallas_call outputs) 161.2 ms vs 158.3 ms baseline:
  saving was slower; consistent with MaxText's scanned-layers warning above. Offload policies
  (`pinned_host`) are for HBM pressure, which this run does not have (2 GiB peak): no.
- `jax_compiler_enable_remat_pass` (XLA's own HLO remat; default True): no effect expected at 2 GiB.

## 6. dot_general precision / preferred_element_type / bf16 accumulation on TPU (lax.py 0.11.2)

- `precision` only affects float32 operands ("This only has an effect on float32 computations").
  For bf16 x bf16 the MXU takes bf16 operands directly; DEFAULT/HIGH/HIGHEST and
  `jax_default_matmul_precision` do not change bf16 dots.
- `preferred_element_type`: "Setting `preferred_element_type` to a specific dtype will mean that the
  operation returns that element type. When `precision` is not a DotAlgorithm or DotAlgorithmPreset,
  `preferred_element_type` provides a hint to the compiler to accumulate the dot product using this
  data type." DotAlgorithmPreset names include `BF16_BF16_F32`, `BF16_BF16_BF16`,
  `BF16_BF16_F32_X3`/`_X6`/`_X9`, `F32_F32_F32`, `DEFAULT`; "the return type of dot_general is always
  defined by the `accumulation_type` parameter of the input algorithm, if specified".
- Transpose rule (`_dot_general_transpose_lhs`): the backward dots reuse the forward
  `preferred_element_type`, then `if x_bar.dtype != x.aval.dtype: x_bar = _convert_element_type(...)`.
  So a forward `dot(bf16, bf16, preferred_element_type=f32)` gives f32 outputs AND f32 cotangents
  flowing backwards; the input-gradient dot also produces f32 that is then cast to bf16. That is
  where "float32 tensors of the widest shapes" in the glue come from. Changing it is a numerics
  change (forbidden) EXCEPT where the f32 output is immediately cast to bf16 (then
  `preferred_element_type=bf16` is the same rounding: MXU accumulates in f32 internally and rounds
  once). Whether XLA already fuses that convert into the dot output: check `as_text()`.

## 7. Donation, in-place updates, optimization barriers

- `jax.jit(..., donate_argnums=..., donate_argnames=...)` (api.py docstring: "In some cases XLA can
  make use of donated buffers to reduce the amount of memory needed to perform a computation, for
  example recycling one of your input buffers to store a result."). docs/buffer_donation.md: "Note
  that this currently does not work when calling your function with key-word arguments!" and "You
  will get a warning if the donated buffer is not used". The step already donates the state
  (`donate_argnums=(0,)`). Donating the batch too only saves its HBM (memory, not speed).
- `jax.lax.optimization_barrier(operand)` (lax.py 0.11.2): "Prevents the compiler from moving
  operations across the barrier." ... "An optimization barrier prevents common subexpression
  elimination. This is used by JAX to implement rematerialization." ... "Optimization barriers prevent
  compiler fusions. That is, operations before the barrier may not be fused into the same kernel as
  operations after the barrier by the compiler." Use only as a surgical A/B tool (e.g. to stop XLA
  from fusing a big f32 producer into a consumer that then re-reads it, or to force a schedule).
  libtpu has `xla_tpu_aggressive_opt_barrier_removal` (MaxText HOST_OFFLOAD_FLAGS
  `--xla_tpu_aggressive_opt_barrier_removal=ENABLED`): semantics undocumented.

## 8. Layout control (jax.experimental.layout, 0.11.2)

- `jax.jit(f, in_shardings=Format(Layout.AUTO), out_shardings=Format(Layout.AUTO))` ("`AUTO`: A
  special, static sentinel object that can be used with `jax.jit` to request that the compiler
  automatically determine a good layout for a compiled function's input or output arrays.",
  docs/notebooks/layout.md). Scaling book: "you can specify the input layouts as AUTO, and `jax.jit`
  will return a layout that the jitted program prefers. You can then explicitly load the tensor in
  that layout to avoid inducing copies within the program." For the train step: the parameters /
  optimizer state could be kept in compiler-chosen layouts across steps (state is donated and
  returned each step). Gain unverified (params are small; matters only if relayout copies of weights
  sit inside the 64x8 loops).
- `jax.experimental.layout.with_layout_constraint(x, Layout(major_to_minor=(...)))`: "enforce a
  specific layout on an intermediate value within a JIT-compiled function" (layout.md); differentiable
  (`ad.deflinear2(layout_constraint_p, ...)`) and vmappable. Candidate use: pin q/k/v (and dO/dQ...)
  to the row-major layout the splash Mosaic kernel needs, so XLA does not choose a different layout
  for the producer fusion and then insert a re-layout copy before the custom call. Evidence of the
  problem: Ragged Paged Attention paper (arXiv 2604.15464): "XLA typically respects a descending
  minor-to-major layout for the boundary inputs and outputs of a Mosaic custom kernel, it frequently
  reorders the layout of intermediate tensors—effectively performing implicit transpositions."
  Scaling book: "XLA will sometimes introduce copies that “retile” or “re-layout” a tensor inside a
  program ... sometimes at non-trivial overhead." Gain unverified; check `copy` ops adjacent to the
  `custom-call` in as_text().

## 9. Documented TPU pitfalls (generation noted)

- OpenXLA HLO passes doc (https://openxla.org/xla/hlo_passes): "Reshapes and transposes can be
  expensive, especially on TPU." (generic TPU)
- Cloud TPU performance guide (https://docs.cloud.google.com/tpu/docs/performance-guide): "MXU size is
  256 x 256 for v6e and TPU7x (Ironwood), and 128 x 128 for versions prior to v6e."; "To use TPU
  memory efficiently, structure your data so that it can be tiled into 128 x 8 chunks."; "Fusion ...
  reduces memory transfers by removing the need to store intermediate results in main memory, which
  is slow."; "forcing a broadcast may result in poor performance and increased memory usage."
  Scaling book TPU chapter: "Weight matrices need to be padded to size divisible by 256 (128 on
  generations prior to TPU v6e) in both dimensions" (v6e-specific) -> head_dim 128 contractions use
  half an MXU pass on v6e.
- v6e numbers: Cloud TPU v6e page: "usable HBM is 31.25 GiB out of 32 GiB", 1638 GBps, 918 TFLOPs bf16;
  scaling book row "| TPU v6e | 16x16 | 4x2 | 32GiB | 1.6e12 | 9.20e14 | 1.84e15 |" -> ~575 FLOP per
  HBM byte break-even: any op below that is HBM-bound, which is every elementwise/convert/copy.
- Precision: on TPU f32 dots default to one bf16 pass (Precision DEFAULT text above).
- Persistent cache pitfall (docs): custom_partitioning breaks cache keys (not used here).
- Gather/scatter: no TPU-specific statement found in docs.jax.dev for single-chip dense models; the repo
  already verified "No optimised HLO above contains a scatter".

## 10. Multi-chip only (out of scope; names only)
xla_tpu_enable_data_parallel_all_reduce_opt, xla_tpu_data_parallel_opt_different_sized_ops,
xla_tpu_enable_async_collective_fusion(_fuse_all_gather/_fuse_all_reduce/_multiple_steps),
xla_tpu_overlap_compute_collective_tc, xla_enable_async_all_gather, xla_enable_async_all_reduce,
xla_enable_async_collective_permute, xla_tpu_enable_async_all_to_all, xla_*_latency_bound_threshold_in_bytes,
xla_jf_spmd_threshold_for_windowed_einsum_mib, xla_tpu_spmd_rng_bit_generator_unsafe,
xla_tpu_megacore_fusion_allow_ags, xla_tpu_enable_ag_backward_pipelining, xla_tpu_enable_ici_ag_pipelining,
xla_should_allow_loop_variant_parameter_in_chain, xla_should_add_loop_invariant_op_in_chain,
xla_tpu_enable_sparse_core_collective_offload_*, xla_tpu_use_minor_sharding_for_major_trivial_input,
xla_tpu_relayout_group_size_threshold_for_reduce_scatter, xla_tpu_assign_all_reduce_scatter_layout,
xla_tpu_enable_windowed_einsum..., jax_use_shardy_partitioner, jax_threefry_partitionable (sharding),
jax_pallas_auto_assign_collective_ids, jax_mock_gpu_topology, jax_share_binary_between_hosts,
`TPU_MEGACORE` (v4/v5p megacore; v6e has 1 TensorCore per chip).

---------------------------------------------------------------------------------------------------

## 11. Extra cache facts that matter for restarts on a NEW Colab runtime
- The key strips debug info, including inside the Mosaic (splash) kernel body, since jax 0.11.0
  (cache_key.py 0.11.2 `_strip_mosaic_debug_info`: "The top-level strip-debuginfo pass does not reach
  into the serialized kernel MLIR embedded in backend_config, so source file paths leak into the cache
  key."; present at jax-v0.11.0/0.11.1/0.11.2, absent at jax-v0.10.2). So a different checkout path on
  the new VM does not change the key (default `jax_compilation_cache_include_metadata_in_key=False`).
- Nothing host-specific is hashed: computation, jaxlib version, `backend.platform` +
  `backend.platform_version`, XLA_FLAGS + LIBTPU_INIT_ARGS + `--xla*` argv, serialized compile
  options, topology fingerprint (`xla_client.get_topology_for_devices(...).fingerprint()`),
  compression, custom_hook. => Copying `RUN_DIR/jax_cache` along with the snapshot to the next v6e-1
  runtime (same jax/jaxlib/libtpu wheel versions, identical flag strings) should turn the ~minutes of
  recompilation into a load. UNVERIFIED on Colab (the repo notes "Whether a restart actually hits the
  cache on the TPU VM is not known yet"); check with `jax_explain_cache_misses=True` /
  `JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.lru_cache` ("PERSISTENT COMPILATION CACHE MISS for
  ..." lines).
- Idea (unverified): the repo's compile-only path (`bes.train.aot`, topology v6e:1x1 on a CPU host)
  could pre-populate the cache off-chip if jaxlib and libtpu versions match the chip's exactly and the
  topology fingerprint/platform_version of the compile-only client equal the real ones.

## 12. Ranked suggestions for this run (my angle), cheapest first
1. Zero chip time: dump optimized HLO of the 4-row and 16-row step (`compiled.as_text()`, or
   `XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_as_text --xla_dump_hlo_module_re=train_step`, which is
   excluded from the cache key) with the compile-only tool; grep for `S(1)` (VMEM), `copy`,
   `transpose`, `convert` fusions inside the loop bodies and next to `custom-call` (tpu_custom_call).
   This tells which glue tensors live in HBM and which knob below can help.
2. One restart, many variants: compile the same step with different `compiler_options` in ONE process
   (`{"xla_tpu_scoped_vmem_limit_kib": "...", "xla_tpu_msa_inefficient_use_to_copy_ratio": "...",
   "optimization_level": "O3"}`) and time each; keep the fastest. Saves a process restart per flag
   variant vs scripts/tpu_flags.py.
3. Code-level, same math: `lax.scan(..., unroll=2)` / `unroll=True` on the 8-step loop scan (not on the
   64-step micro-batch scan); expect longer compile; A/B.
4. Code-level, same math: `jax.experimental.layout.with_layout_constraint` on q/k/v and on the kernel
   output if step 1 shows re-layout copies around the splash custom call.
5. Restart hygiene: keep LIBTPU_INIT_ARGS/XLA_FLAGS strings byte-identical between restarts; copy the
   cache dir with snapshots; optionally `jax.experimental.serialize_executable` for the step.
6. Leave as is: `jax_default_matmul_precision` unset, `jax_enable_x64` off,
   `jax_disable_most_optimizations` off, `jax_compiler_enable_remat_pass` default, debug checks off,
   `prevent_cse=False` in the scanned remat (already), donation of the state (already).

## 13. Dead ends / non-knobs (checked)
- `lax.scan(_split_transpose=True)`: no-op in jax 0.11.2 (parameter unused).
- `jax_default_dtype_bits`: does not exist in jax 0.11.2 (not in config.py, dtypes.py, CHANGELOG).
- `jax_exec_time_optimization_effort` / `jax_memory_fitting_effort`: removed in 0.11.1 (use the
  `*_level` options).
- `jax_default_matmul_precision`: cannot speed up bf16 dots; DEFAULT is already 1 bf16 pass for f32.
- `jax_persistent_cache_enable_xla_caches`: GPU caches only.
- `jax_enable_pgle`: documented in the GPU performance tips; on one TPU chip there are no collectives
  for the latency-hiding scheduler to hide; not evaluated.
- Offload remat policies / `pinned_host`: HBM is not the constraint (peak ~2 GiB).
- `inline=` / `jax.Inline`: the model has no nested `jax.jit` inside the step.
- Putting `--xla_tpu_*` into XLA_FLAGS: process aborts (ParseFlagsFromEnvAndDieIfUnknown).
- `jax_default_prng_impl`, `jax_threefry_partitionable`: change random bits (math).
- No public TPU (any generation) measurement of `scan(unroll=...)` for a transformer-sized body was
  found (GitHub #7024 and #16611 are GPU; Krea-2-JAX PR #10 "scan copy storm" is B300/MI350X GPU:
  "remat-saved GEMM outputs dynamic-update-sliced into stacked residual buffers", step -4.5%,
  compile 30 -> 39 s; JAX PR #3076 is a 2020 polyval micro-benchmark).

## 14. Cross-angle notes (for the Pallas / LIBTPU agents)
- error_1001 suggests `pallas.tpu.with_memory_space_constraint` "to place kernel inputs or outputs in
  VMEM" and per-kernel `vmem_limit_bytes`; JAX docs: `vmem_limit_bytes` "must be used with the
  --xla_tpu_scoped_vmem_limit_kib=N flag, where N*1KiB is greater than vmem_limit_bytes" (search
  snippet of docs.jax.dev pltpu.CompilerParams; not quoted from the page itself).
- MaxText v6e llama3.1-8b-8192 uses splash blocks 2048 for all of q/kv/kv_compute and dkv/dq with
  `"sa_use_fused_bwd_kernel": True` (benchmarks/maxtext_trillium_model_configs.py).
- Arithmetic: a [4, 1024, 768] bf16 activation is 6 MiB (f32: 12 MiB); [16, 1024, 768] is 24/48 MiB.
  With 128 MiB VMEM per v6e TensorCore (minus the scoped limit, default 16 MiB per the OpenXLA guide),
  only the 4-row tensors can plausibly be kept in VMEM by MSA, matching the measured +10%.
