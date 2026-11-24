# TPU v6e (Trillium, device kind "TPU v6 lite") - hardware and software-stack facts, and the knobs they imply for ONE chip

Status: COMPLETE (2026-11-24). Angle: public facts about the v6e chip (HBM, bandwidth, VMEM, MXU, peak FLOP/s),
how XLA uses VMEM (scoped VMEM vs memory-space assignment), v5e/v5p differences that matter on one chip, Cloud TPU docs
(performance guide, v6e page, run JAX on TPU), Colab v6e-1 notes, libtpu release notes, published single-chip v6e
training throughput, the transparent-hugepages warning, and what jax/_src/cloud_tpu_init.py does.

Nothing here was run on a TPU. Sibling notes in this directory already cover the flag catalogue in depth
(knobs_libtpu_flags.md, knobs_maxtext.md, knobs_pallas.md, knobs_jax_side.md); this file cross-references them and
adds the hardware view.

-------------------------------------------------------------------------------------------------------------
## 1. jax 0.11.2 (the run's JAX) - what the library itself says about v6e

Source: the jax 0.11.2 wheel, file jax/_src/tpu_info.py (read locally from scratchpad/dl_jax/x/jax/_src/tpu_info.py;
same file on GitHub: https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py).

- Device-kind mapping: `case "TPU v6e" | "TPU v6 lite": return ChipVersion.TPU_V6E` -> our "TPU v6 lite" IS v6e.
- Table in the ChipVersion docstring: `| v6e | 1 | Yes | No |` (physical TensorCores per chip = 1, lite chip = yes,
  megacore = no). Consequence: Pallas `dimension_semantics=("parallel", ...)` cannot split a kernel over two cores on
  v6e (only v4/v5p megacore); grid order matters only for pipelining/revisiting.
- TPU_V6E TpuInfo (per TensorCore = per chip on v6e):
  ```
  num_lanes=NUM_LANES (128), num_sublanes=NUM_SUBLANES (8),
  mxu_column_size=MXU_COLUMN_SIZE_GEN_GE_6 (256),   # v5e/v5p/v4: MXU_COLUMN_SIZE_GEN_LT_6 = 128
  num_mxus=2,                                       # v5e: num_mxus=4 (of 128 columns)
  vmem_capacity_bytes=128 * 1024 * 1024,  # 128 MiB per core      (v5e 128 MiB, v5p 64 MiB, v4 16 MiB, TPU7x 64 MiB)
  smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
  hbm_capacity_bytes=34_400_000_000,                # 32 GiB  (the run sees 31.24 GiB usable)
  mem_bw_bytes_per_second=int(1.64e12),             # v5e 8.20e11, v5p 2.46e12 (2 cores)
  bf16_ops_per_second=int(9.20e14),                 # v5e 1.97e14
  int8_ops_per_second=int(1.84e15), fp8_ops_per_second=int(9.20e14), int4_ops_per_second=int(3.68e15),
  sparse_core=SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=8,
                             vmem_capacity_bytes=256 * 1024,  # 256 KiB per vector subcore
                             dma_granule_size_bytes=32)
  ```
- Ridge point (peak FLOP/s / HBM B/s) on v6e: 9.20e14 / 1.64e12 = 561 FLOP per byte of HBM traffic
  (v5e: 1.97e14/8.2e11 = 240). An element-wise bf16 pass (read 2 B + write 2 B per element) does ~1 FLOP per 4 B,
  i.e. ~2000x below the ridge: on v6e every glue pass is pure HBM time, and v6e's compute/bandwidth ratio is 2.3x
  worse than v5e's, which is why glue is relatively MORE expensive on v6e than on v5e.
- Sublane tiling comment in the same file (applies to v6e, generation 6 < 7):
  "Caveat: before TPU7x, by default XLA does not use large 2nd minor tiling but it can be enabled by setting the flag
  xla_tpu_enable_large_2nd_minor_layout_for_x16." -> knob K-layout below.

Measured numbers of the run against these peaks: 814 TFLOP/s forward FFN products = 88% of 9.20e14.

## 2. jax/_src/cloud_tpu_init.py (jax 0.11.2) - everything it sets up

Read locally (scratchpad/dl_jax/x/jax/_src/cloud_tpu_init.py); called before the TPU backend loads.
1. Finds libtpu: `TPU_LIBRARY_PATH` env (if the file exists) else `libtpu.get_library_path()`.
2. Counts TPU chips through PCI (`hardware_utils.num_available_tpu_chips_and_device_id()`: vendor 0x1ae0, device id
   `'0x006f': TpuVersion.v6e`). If 0 chips: `os.environ['TPU_SKIP_MDS_QUERY'] = '1'`.
3. THP warning: `if tpu_id is not None and tpu_id >= hardware_utils.TpuVersion.v5e and not
   hardware_utils.transparent_hugepages_enabled(): warnings.warn('Transparent hugepages are not enabled. TPU runtime
   startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need
   to enable transparent hugepages in your VM image (sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled")')`
   `transparent_hugepages_enabled()` is true only when /sys/kernel/mm/transparent_hugepage/enabled reads exactly
   `[always] madvise never` (so `madvise` mode also triggers the warning).
4. Returns early unless libtpu is found and >=1 chip (or `JAX_FORCE_TPU_INIT` is set).
5. `os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')`, `TPU_ML_PLATFORM='JAX'`, `TPU_ML_PLATFORM_VERSION=<jax version>`,
   `ENABLE_RUNTIME_UPTIME_TELEMETRY='1'`.
6. Appends ` --xla_tpu_use_enhanced_launch_barrier=true` to LIBTPU_INIT_ARGS unless already present (keeps any
   user flags: it appends, so a user's LIBTPU_INIT_ARGS set before `import jax` survives).
7. `TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60`, `TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=256` (checkpoint I/O only).
8. `jax_platforms` -> 'tpu,cpu' if unset; `jax_pjrt_client_create_options` ->
   `ml_framework_name:JAX;ml_framework_version:<ver>` if unset.
Nothing in it touches compiler optimisation, VMEM, or clocks. It sets no performance flag except the launch barrier
(a multi-host launch synchronisation feature).

(sections below are filled as the research proceeds)

## 3. Public hardware facts for v6e (with exact quotes)

### 3a. Google Cloud TPU docs, "TPU v6e" page (https://docs.cloud.google.com/tpu/docs/v6e; cloud.google.com/tpu/docs/v6e redirects there)
- "Peak compute per chip (bf16) | 918 TFLOPs"; "Peak compute per chip (Int8) | 1836 TOPs"
- "HBM capacity per chip | 32 GB"; "On TPU v6e, usable HBM is 31.25 GiB out of 32 GiB of physical HBM per chip."
  (the run's compiler reports 31.24 GB available: consistent)
- "HBM bandwidth per chip | 1638 GBps"
- "Bidirectional inter-chip interconnect (ICI) bandwidth (per chip) | 800 GBps" (irrelevant on one chip)
- "Each v6e chip contains one TensorCore"; "Each TensorCore contains two 256x256 matrix-multiply units (MXU)";
  "a 128x8 vector processing unit (VPU) with 128 lanes x 8 sublanes"; "a scalar processing unit (SPU)"; SparseCore listed.
- v6e-1 VM: "1-chip VM | 44 | 176 | 1" (44 vCPUs, 176 GB RAM, 1 NUMA node). Colab's v6e-1 host may differ.
- No VMEM figure, no flags on that page.

### 3b. JAX Pallas "TPU Hardware Reference" (https://docs.jax.dev/en/latest/pallas/tpu/hardware.html), per TensorCore:
  "| V5E | TPU v5 | 1 | 128 MiB | N/A | 1024 KiB | 17 GB | 820.0 GB/s | 197 TFLOPs/s | N/A | 394 TOPs/s | 788 TOPs/s | No |";
  v6e row (fetched, restated by the fetch tool): 1 TC, VMEM 128 MiB, CMEM N/A, SMEM 1024 KiB, HBM 34 GB, 1640.0 GB/s,
  920 TFLOPs/s bf16, FP8 920, INT8 1840 TOPs/s, INT4 3680, SparseCore "Yes (2 SCs, 16 subcores, 256 KiB VMEM)".
  v5p: 2 TC, 64 MiB VMEM per TC.

### 3c. The Scaling Book (https://jax-ml.github.io/scaling-book/tpus/)
- "| TPU v6e | 16x16 | 4x2 | 32GiB | 1.6e12 | 9.20e14 | 1.84e15 |" (v5e: "| TPU v5e | 16x16 | 4x2 | 16GiB | 8.2e11 | 1.97e14 | 3.94e14 |")
- "At the core of the TPU MXU is a `256x256` systolic array (`128x128` prior to TPU v6e)."
- "The VPU is a 2D vector arithmetic unit of shape `(8, 128)` where the 128 dimension is referred to as lane axis"
- bf16 on the VPU: "VPU operations are still performed in fp32 or bf16 (though generations prior to TPU v6e only support
  fp32 VPU operations)." and "TPU v6e and TPU7x additionally can perform elementwise arithmetic operations on packed
  bfloat16 VREGs."  -> NEW on v6e vs v5e: native packed-bf16 element-wise math (see knob K-bf16emit; numerics caveat).
- VMEM: "an on-chip scratchpad located in the TensorCore", "operates somewhat like an L1/L2 cache on CPUs but is much
  larger and programmer-controlled"; "if we can fit our weights into VMEM instead of HBM, our matrix multiplications can
  be FLOPs bound at much smaller batch sizes"; v5e worked example assumes "VMEM bandwidth to the MXU as 22 times the HBM";
  "VMEM bandwidth is ~7x higher than HBM bandwidth for TPU7x". No v6e VMEM-bandwidth figure is published.

### 3d. Cloud TPU performance guide (https://docs.cloud.google.com/tpu/docs/performance-guide)
- "MXU size is 256 x 256 for v6e and TPU7x (Ironwood), and 128 x 128 for versions prior to v6e."
- "To achieve peak FLOPs, dimensions of matrix multiplication should be larger than the MXU size for the TPU version you are using."
- "To use TPU memory efficiently, structure your data so that it can be tiled into 128 x 8 chunks." / "When the data for a
  matrix computation does not fill an entire 128 x 8 chunk, the XLA compiler pads tensors."
- Host-memory section (single-chip relevant, host<->device transfers only):
  "`TPU_PREMAPPED_BUFFER_SIZE` sets the size of the host memory buffer (in bytes) that is pre-mapped and pinned for use by
  the TPU runtime for data transfers (for ..."; "The default value is 4294967296 bytes."; "If the buffer is excessively
  large, TPU initialization can take much longer (potentially more than 15 seconds), making it seem as if the TPU is stuck.";
  "`TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES`: Sets the maximum size of a single buffer that can be allocated from
  the pre-mapped region." (no default stated); unpinned_host / np.asarray "stages DMA transfers through the pre-mapped
  buffer pool"; logs at "/tmp/tpu_logs/tpu_driver.INFO" or "TPU_STDERR_LOG_LEVEL=0".
- tcmalloc: "The `tcmalloc` library is used by default on Cloud TPU VMs to improve performance for models with sizable,
  frequent memory allocations. This is configured through the `LD_PRELOAD` environment variable." / "for some workloads
  (for example, DLRM with very large embedding table allocations), `tcmalloc` can cause a slowdown." -> host-side only.
- libtpu 0.0.50 contains `TPU_PREMAPPED_BUFFER_SIZE`, `TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES`,
  `TPU_PREMAPPED_BUFFER_PARTITION_COUNT` and "CreateTpuSystemState: using TPU host premapped buffer of size: " (strings).

### 3e. google/ml-workload-debugging-wiki (public Google wiki; commit a413eea9)
- entities/tensorcore.md: "**TPU v6e (Trillium)**: **1 TensorCore per chip**, containing **2 MXUs of `256x256`**
  multiply-accumulators and 32 GB HBM." / "**TPU v5e**: **1 TensorCore per chip** with **4 MXUs of `128x128`** and 16 GB HBM."
- entities/cloud-tpu.md table: v6e "MXU Array Size | 2 x `256x256` / TC", "Peak Compute / Chip (BF16) | 918 TFLOPs",
  "HBM Bandwidth / Chip | 1,638 GBps", "SparseCores per Chip | 2"; machine types `ct6e-standard-{1t,4t,8t}`.
- entities/xla.md: "XLA tiles matrix multiplications into blocks matching the [[tensorcore]] MXU systolic array (`128x128`
  on TPU `v4`, `v5e`, and `v5p`; `256x256` on `v6e` and `TPU7x`) and aligns memory subsystem accesses to multiples of `8`.
  Unaligned tensor layouts force memory-bound `reshape` operations before tiling."
- playbooks/numerics-error-triage.md: "`--xla_tpu_scoped_vmem_limit_kib`: Limits VMEM available for scoped allocations
  (default is 16 MiB on v4, v5e, v5p; 32 MiB on v6e+)."  Same list: "`--xla_msa_enable=false`: Disables Memory Space
  Assignment (MSA) specific optimizations." (diagnostic: lets one MEASURE how much the step owes to VMEM residency).

### 3f. OpenXLA cost-model table, xla/benchmarks/core/platform_info.py
(https://github.com/openxla/xla/blob/main/xla/benchmarks/core/platform_info.py, downloaded 2026-11-24), TPU_V6E entry:
  `# Note that v6e doesn't support P states.` `clock_speed_ghz_by_p_state=immutabledict({None: 1.75,})`,
  `hbm_to_vmem_bandwidth_gb_per_sec=1373`, `hbm_to_vmem_latency_ns=628`, `vmem_to_hbm_latency_ns=523`,
  `default_p_state=None`, `default_vmem_limit_kib=32 * 1024`, `num_alus=4`, `num_xlus=2`,
  `default_internal_scratch_bytes=73728` ("The default size of the internal scratch in bytes for Pallas kernels"),
  bf16 `matmul_latency_cycles` 192. v5e entry for comparison: 1.5 GHz, `default_vmem_limit_kib=16 * 1024`, `num_xlus=3`.
  Docstring: "default_vmem_limit_kib: The default VMEM limit in KiB for the TPU platform."
  -> v6e default scoped VMEM = 32 MiB of the 128 MiB; the clock is fixed (DVFS P-state flags are moot on v6e).
  Note: 1.64e12 B/s is the HBM peak; the cost model uses 1373 GB/s HBM->VMEM, i.e. ~84% is what a stream achieves.

### 3g. OpenXLA error page E1001 (https://openxla.org/xla/errors/error_1001) - scoped VMEM
- "Instruction-scoped": "Temporary storage in Vmem while executing a single HLO instruction." "This includes operand span
  buffer (e.g. for double buffering) and register spills."
- "Program-scoped": "Allocations that live beyond the scope of a single HLO instruction." "These are usually HLO
  temporaries and intermediate results that are inputs and/or outputs of HLO instructions."
- Usable limits table: "| TPU v5e | 128MB | 131072 | 134217728 |", "| TPU v5p | 63.9MB | 65472 | 67043328 |",
  "| TPU v6e | 127.9MB | 131008 | 134152192 |", "| TPU 7x | 63.9MB | 65472 | 67043328 |" (v2-v4: 16MB).
- Knobs named: "globally for the entire program via the flag --xla_tpu_scoped_vmem_limit_kib" and "per custom kernel via
  vmem_limit_bytes param"; "Explicitly color/constrain the kernel's inputs/outputs to Vmem using
  pallas.tpu.with_memory_space_constraint." ("Be careful not to color too many inputs outputs to Vmem, as that might cause
  an overall Vmem OOM.")

### 3h. What this means for the run's observations (inference, not measured)
- VMEM = 128 MiB; default per-op scoped scratch = 32 MiB; the remaining ~96 MiB is what memory-space assignment (MSA)
  can use to keep HLO temporaries ("program-scoped" allocations) in VMEM across ops instead of HBM.
- Activation sizes per micro-batch row-count R at T=1024, d=768: bf16 [R,T,d] = 1.5 MiB*R, f32 = 3 MiB*R;
  f32 [R,T,3072] = 12 MiB*R. R=4: 6 / 12 / 48 MiB -> several glue tensors can live in the ~96 MiB pool;
  R=16: 24 / 48 / 192 MiB -> almost nothing of the wide f32 tensors fits. This is consistent with "4 rows is 10%
  faster than 16 rows" and with the public Colab v6e-1 run below ("16 microbatches of size 4 pushed the MXU usage over 50%").
- Ridge point 561 FLOP/byte (v5e 240): glue passes are ~2.3x more expensive relative to matmuls on v6e than on v5e.
- MXU 256x256: head_dim = 128 contractions (q.k^T over hd=128; p.v producing hd=128 columns) cannot fill a 256-deep /
  256-wide array; public v6e measurements (3i) show head_dim 256 runs at higher MFU. Model maths fixed -> no knob, but it
  explains why the attention kernel's share stays high on v6e.

## 4. The transparent-hugepages (THP) warning: start-up/shutdown only

- Origin: jax commit 5cb29949d "Warn the user if transparent huge pages aren't enabled." (Michael Whittaker, 2025-03-10;
  files jax/_src/cloud_tpu_init.py, jax/_src/hardware_utils.py). No rationale beyond the message.
- JAX CHANGELOG, "jax 0.5.1 (Feb 24, 2025)", Bug fixes (https://github.com/jax-ml/jax/blob/main/CHANGELOG.md):
  "TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer (from around 17s to
  around 8s). If not already set, you may need to enable transparent hugepages in your VM image
  (`sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'`). We hope to improve this further in future
  releases."
- Mechanism consistent with the public docs: at start-up the runtime pre-maps and pins a host buffer
  (performance guide: `TPU_PREMAPPED_BUFFER_SIZE` default 4294967296 bytes; "If the buffer is excessively large, TPU
  initialization can take much longer (potentially more than 15 seconds)"; libtpu strings "Starting premapped memory
  manager initialization...", "Done premapping memory of size "). With 4 KiB pages that is ~1M pages to fault/pin; THP
  makes it ~2k. The same mapping is torn down at shutdown.
- Steady state: no public source claims an effect on step time; the training step runs on the chip from HBM/VMEM, and
  the per-step host->device traffic of this run (int32 token ids, a few MB) goes through the already-pinned buffer.
  -> Verdict: start-up/shutdown only (~9 s saved per process start per the CHANGELOG figures), irrelevant for tokens/s.
  Worth doing only because the run restarts (3 min each): `sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'`
  before Python starts (may be refused inside Colab's container; harmless to try). Note the JAX check is strict:
  `transparent_hugepages_enabled()` requires the file to read exactly `[always] madvise never`, so `madvise` mode also warns.

## 5. Published single-chip v6e measurements to compare with (217-247k tok/s here)

### 5a. tpuchat / "LLM pretraining on TPU v6e with a $50 budget" (A. Vorushin), https://vorushin.github.io/blog/llm-pretraining-tpu-v6e-50usd
- Colab TPUs ("Supported TPUs are v5e and v6e"; "v5e costs 3.14 credits per hour, v6e - 3.71"); the nanochat-jax
  write-up calls it "trains depth 8 / 164M ... on a single v6e-1 in Colab. 46.5% MFU."
  (https://github.com/tucan9389/nanochat-jax/discussions/1).
- "| Throughput (tok/s) | 433,606 | 148,829 |" (v6e | v5e); "For our final config: 46.5% MFU, 51.4% MXU."
- Model: 163.6M params (130.0M non-embedding), 8 layers, d_model 1024, 4 q heads + 1 KV head (MQA), head_dim 256,
  SwiGLU 3072, vocab 32K, seq 2048, batch 64 "(16 microbatches x 4)".
- "adding gradient accumulation (using 16 microbatches of size 4) pushed the MXU usage over 50%."  <- same direction as
  this run's 16->4 rows finding.
- "Chunked LM head computation helped to reduce HBM usage - otherwise I saw multi-GB tensors in the XProf."
- head_dim row: "| head_dim=128 vs 256 | 22 vs 38 | 39.7 vs 23.1 | 256 aligns with MXU blocks |" (column order is
  ambiguous on the page; the text's conclusion is that 256 is better aligned).
- "It shows MXU usage of 80.6% when run on TPU v5e vs 51.4% on v6e." (same code: v6e's bigger MXU and higher
  FLOP/byte ratio make it harder to keep busy).
- Comparison (inference): model FLOP per token for them ~ 433.6e3 tok/s * F = 0.465 * 918e12 -> F ~ 0.98 GFLOP/token.
  Ours: 16 block applications of 12*d^2 = 7.08M params -> 6*113M = 0.68 GFLOP + attention (~0.08) + head (~0.04)
  = ~0.79 GFLOP/token model FLOPs (without the remat recompute) -> 230k tok/s = 1.8e14 FLOP/s = ~20% MFU
  (~26% counting the recomputed forward). Their per-token work is 1.25x ours, their tokens/s 1.9x ours.
  Structural differences that are maths (not knobs): d_model 768 vs 1024 (smaller matrices on 256x256 MXUs), head_dim
  128 vs 256, f32 glue. Process differences that are knobs: micro-batch of 4 rows, no remat noted.

### 5b. tpu-recipes single-chip microbenchmarks (Google), https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/microbenchmarks/README.md
- "Set up a v6e TPU VM for single-chip microbenchmarks:" ... `--accelerator-type=v6e-1` `--version=v2-alpha-tpuv6e`
- `python benchmark_matmul.py --dim 8192 8192 8192 --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 --trace_matcher="jit_matmul.*"`
  -> "dtype: bfloat16, matrix dimensions: (8192, 8192, 8192), time taken (median, ms): 1.328756094, TFLOPS: 827.474382048629"
  (= 90% of 918; this run's 814 TFLOP/s FFN products are at the same ceiling).
- `python benchmark_hbm.py --num_elements=16777216 --trace_matcher="jit_my_copy.*"` -> "Tensor size (bytes): 33554432,
  time taken (ms, median): 0.049359414, bandwidth (GBps, median): 1359.5960438266143" (2 x 32 MiB / 49.4 us, i.e. read +
  write counted: 83% of the 1640 GB/s peak; matches the OpenXLA cost model's 1373 GB/s HBM->VMEM).
  -> Rule for this run: one read+write pass over an X-MB tensor costs ~X/680 ms (e.g. a 100 MB f32 [32k,768] pass ~0.15 ms).

### 5c. Others (multi-chip v6e-8, per-chip comparable only roughly)
- nanochat-jax (tucan9389) on "8 v6e chips (a single-host slice)": "MFU is about 24% (d24)"; splash tiles: "In an A/B
  that changed only the tiles, MFU went from 22.4% to 41.2%," (head_dim 256, seq 4096); head_dim table
  "| 12 heads × 128 — nanochat as-is (our setting) | 30.1% |" vs "| 6 heads × 256 — the v6e-recommended alignment | **35.2%** |";
  "That works out to about 17% faster at the same parameter count."; flags `--attn-impl=splash --splash-block-q=512
  --splash-block-kv=512 --splash-block-kv-compute=256`; set-up `sudo rm -f /tmp/libtpu_lockfile`.
- nor, modded-nanogpt in JAX on "a TPU v6e-8" (https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax): "the model
  was entirely HBM-bound, with a Model Flops Utilization (MFU) of around 15%", "HBM utilization was hovering around 50%",
  later "the MFU improved to 23%"; d_model 768 -> 1024 and "4 heads with a `d_head` of 256"; "flash attention kernels
  (were slow), splash attention kernels (were also slow)" (their choice: jax.nn.dot_product_attention).
  -> a d=768 / head_dim 128 GPT on v6e being HBM-bound at 15-23% MFU is the public norm; ours (~20-26%) is in that band.
- picodo (martin-marek): a mirror of its README says 39% MFU on TPU v6e-1 for GPT-2 124M (unverified: the current
  README at https://raw.githubusercontent.com/martin-marek/picodo/main/README.md says only "32% MFU" with no hardware);
  same README: "Note: using `model.unroll=True` significantly increases throughput at the cost of longer copilation time."

## 6. More v6e-vs-older-generation differences that matter on one chip

| | v5e | v5p (per chip) | v6e (ours) | source |
|---|---|---|---|---|
| TensorCores / megacore | 1 / no | 2 / yes | 1 / no | jax tpu_info.py |
| MXUs | 4 x 128x128 | 4 x 128x128 per TC | 2 x 256x256 (docs) | Cloud v6e page, wiki tensorcore.md |
| peak bf16 | 197 TFLOP/s | 459 | 918 | Cloud docs / wiki |
| HBM / BW | 16 GB / 819 GB/s | 95 GiB / 2765 GB/s | 32 GB (31.25 GiB usable) / 1638 GB/s | wiki cloud-tpu.md, Cloud v6e page |
| ridge (FLOP/B) | 240 | 166 | 561 | derived |
| VMEM | 128 MiB | 64 MiB per TC | 128 MiB | jax tpu_info.py, JAX hardware page |
| default scoped VMEM | 16 MiB | 16 MiB | 32 MiB | openxla platform_info.py, wiki |
| max scoped VMEM | 131072 KiB | 65472 KiB | 131008 KiB | openxla E1001 |
| clock / DVFS | 1.5 GHz / none | 1.75 GHz | 1.75 GHz / "v6e doesn't support P states" | openxla platform_info.py |
| XLUs (cross-lane/transpose units) | 3 | 3 | 2 | openxla platform_info.py `num_xlus` |
| bf16 VPU math | no ("prior to TPU v6e only support fp32 VPU operations") | no | yes (packed bf16) | scaling book |
| SparseCores | 0 | 4 | 2 | wiki sparsecore.md, jax tpu_info.py |

- xprof cost model (openxla/xprof docs/custom_call_profiling.md, "Cycle Estimates by Unit and Generation",
  (v5e/v5p) vs (v6e/v7x)): "| **Packed Transpose**   | All | 17 | 4 |", "| **Standard Transpose** | B32 Transpose | 9 | 4 |",
  "B16 Transpose | 17 | 4"; EUP "**Vector Math** (`tanh`, `exp`, etc.) | 2 | 1". (Modelling numbers used to lay out
  trace events, not a throughput guarantee; they say the per-instruction transpose and exp costs dropped on v6e,
  so the transposes this run sees are expensive because of HBM passes, not XLU issue rate.)
- MXU count is publicly inconsistent: 2 x 256x256 x 2 FLOP x 1.75 GHz = 459 TFLOP/s, half of 918; a dev.to analysis
  (https://dev.to/gde/serving-gemma-4-e2b-on-a-tpu-v6e-1-what-trillium-buys-and-what-it-doesnt-5691) says "Treat peak
  compute as reliable and the MXU count as unresolved." The 918 figure is confirmed by measurement (827 TFLOP/s
  tpu-recipes matmul; 814 TFLOP/s in this run). Irrelevant for knobs.
- Google launch blog (cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus, via search snippet):
  4.7X peak compute per chip vs v5e by "expanding the size of matrix multiply units (MXUs) and increasing the clock
  speed", and "doubled the High Bandwidth Memory (HBM) capacity and bandwidth" vs v5e. Cloud TPU release notes
  (https://docs.cloud.google.com/tpu/docs/release-notes), December 16, 2024: "This Release Note announces General
  Availability of Trillium AKA v6e."; "An impressive 4.7x increase in peak compute performance per chip."
- Instruction memory (IMEM): Sail Research, "Chasing Speed of Light on TPU v6e" (https://sailresearch.com/blog/tpu-v6e-gemma,
  4-chip v6e inference): "it appears to be around ~4 MiB for v6e according to reverse engineering efforts.";
  "Programs that outgrow it are split into instruction overlays and streamed in from HBM during execution.";
  "Those "SyncWait (sequencer_overlay)" events were instruction-fetch stalls"; "By using smaller block sizes, we were
  able to keep our kernels under the IMEM limit". Splash code size grows with tile size: Patrick Toulme
  (https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas, "All experiments were performed on a TPU V6e
  Trillium"): block 512 -> 1,651 bundles, 1024 -> 4,302, 2048 -> 15,153 (forward kernel). -> Our step program is big
  (64-step micro-batch scan x 8-loop remat scan x fused-bwd splash at 1024 tiles): check a trace for sequencer_overlay
  waits before assuming the remaining glue is all HBM time.

## 7. Colab v6e-1 and the software stack

- Colab announcement (https://x.com/GoogleColab/status/1920904699122720941, via search snippet): "Say "hello [emoji omitted]" to Colab's fastest deep learning
  accelerator: v6e-1 (Trillium) TPUs!! 2x the high bandwidth memory as v5e-1 (32GB) and a whopping peak rating of 918
  BF16 TFLOPS (nearly 3x A100)! ... Try them out with JAX & Flax in our example notebook below". The Colab front-end
  config lists `"user_visible_accelerator_models": ["T4", "A100", "G4", "H100", "L4", "V5E1", "V6E1"]`.
  No Colab page documents the v6e-1 runtime's software (JAX/libtpu versions, THP, tcmalloc) - dead end.
- libtpu releases (PyPI JSON, 2026-11-24): 0.0.50 (2026-10-07), 0.0.50.1 (2026-10-25), 0.0.51 (2026-10-21),
  0.0.52 (2026-11-05), 0.0.53 (2026-11-15), 0.0.53.1 (2026-11-17). Release notes on PyPI are compatibility-only
  (e.g. "JAX Compatibility: `libtpu` supports `JAX` 0.7.1 or newer."); no performance notes.
- jax 0.11.2 METADATA pins `libtpu==0.0.48.*; extra == "tpu"` (the run uses 0.0.50, newer than the pin);
  JAX 0.12.0: "Updated minimum pinned `libtpu` version to `0.0.50`."; JAX 0.12.1: "Bumped the pinned `libtpu` dependency
  for `jax[tpu]` from `0.0.50.*` to `0.0.53.*` ({jax-issue}`#42544`)."; older: jax 0.4.37 "Pinned a new libtpu release
  (0.0.6) that fixes a compiler bug on TPU v6e."
- Pallas CHANGELOG (JAX 0.12.1): "Added {func}`jax.pallas.tpu.request_p_state` to request a TensorCore P-state transition
  on TPU7x+ targets." -> clocks are not a knob on v6e.
- Cloud TPU software image for v6e: `v2-alpha-tpuv6e` (https://docs.cloud.google.com/tpu/docs/runtimes); no
  configuration details published.

## 8. Knobs from this angle (single chip), ranked

All LIBTPU flags: `LIBTPU_INIT_ARGS="--flag=value ..."` in os.environ before `import jax` (cloud_tpu_init only APPENDS
its launch-barrier flag, so user flags survive). Every flag spelling below occurs in libtpu 0.0.50
(scratchpad/dl_libtpu0050/libtpu_strings.txt). The CPU compile-only host (bes/train/aot.py, topology v6e:1x1) runs
the same libtpu compiler, so a flag's acceptance, its compile-time VMEM failures and its effect on VMEM placement can be
checked there without the chip.

K1 `--xla_tpu_scoped_vmem_limit_kib=N` (HIGH). v6e default 32768 (platform_info.py `default_vmem_limit_kib=32 * 1024`;
   wiki "32 MiB on v6e+"), max 131008 (E1001). Meaning (openxla flags_guidance): "This flag sets the amount of
   scratchpad VMEM available to per op for local usage in KiloBytes." "Rest of the VMEM is used as buffer space."
   For this memory-bound step the interesting direction is DOWN as well as up: less per-op scratch = larger pool for
   keeping glue tensors in VMEM. Sweep 16384 / 24576 / 32768 / 49152 / 65536 / 98304; reject values that fail to
   compile ("Ran out of memory in memory space vmem") on the CPU host first. v6e use: tpu-recipes v6e-1 matmul (65536),
   MaxText v6e dense (98304), openxla v6e/7x benchmarks (32768).
K2 micro-batch rows `--override optim.micro_seqs=N` (HIGH, already 4). Public v6e evidence for small micro-batches
   ("adding gradient accumulation (using 16 microbatches of size 4) pushed the MXU usage over 50%", Colab v6e).
   VMEM arithmetic (section 3h) predicts a further gain at 2 rows only if the f32 [R,T,3072] tensors start fitting
   (24 MiB at R=2); cost: one more read of the weights and one more f32 gradient accumulate per extra micro-batch
   (~0.2 ms each at 1.36 TB/s, i.e. ~13 ms more per 262k-token step). A/B 2 vs 4.
K3 MSA placement knobs (MEDIUM; documented by OpenXLA, never tuned publicly for v6e training):
   `--xla_tpu_msa_inefficient_use_to_copy_ratio=R` (default 0.5, range [0,1]; "This is used while making VMEM
   placement decisions.") and `--xla_tpu_async_copy_bandwidth_scaling_factor=F` (default 1, range (0,1]; "This is used
   when making prefetch decisions and deciding which tensors should live in VMEM.").
K4 Diagnostics for the VMEM hypothesis (no speed-up by themselves, MEDIUM-HIGH value):
   - compiled HLO: `jax.jit(step).lower(*args).compile().as_text()` and count `S(1)` buffers (scaling book: "S(1) tells
     us this array lives in VMEM. S(0) (sometimes omitted) is HBM.") at micro_seqs 4 vs 16 and per K1/K3 value - on the
     CPU compile host.
   - `--xla_msa_enable=false` (wiki numerics triage: "Disables Memory Space Assignment (MSA) specific optimizations."):
     one timed step with it says how much of the step VMEM residency is already buying.
K5 Profiling knobs that expose v6e-specific stalls (MEDIUM, diagnostic): `--xla_xprof_register_llo_debug_info=true`
   (xprof docs: supported from "`jax >= 0.10.2` (`libtpu >= 0.0.42`)"; Sail: "the waits are only labeled if you set
   --xla_xprof_register_llo_debug_info=true in LIBTPU_INIT_ARGS"), profiler option
   `options = jax.profiler.ProfileOptions(); options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}`
   (XProf JAX profiling guide https://openxla.org/xprof/jax_profiling: "This traces both compute operations and
   synchronization events on the device."; default "TRACE_ONLY_XLA"; add "tpu_num_sparse_cores_to_trace": 2 to see the
   2 SparseCores), and for time inside the splash kernel `--xla_xprof_enable_custom_call_tracing=true` (xprof docs: "requires
   `jax >= 0.11.0` (`libtpu >= 0.0.44`)"; measured on "TPU v6e-1" with Pallas FlashAttention, trace 19.8 MB -> 127 MB).
   Look for "SyncWait (sequencer_overlay)" (IMEM overlays, ~4 MiB IMEM on v6e) and SparseCore copy waits.
   Spelling-only (public dump default `--xla_tpu_show_overlay_waits_in_xprof=false`): `--xla_tpu_show_overlay_waits_in_xprof=true`,
   semantics inferred from the name (unverified).
K6 `--xla_tpu_enable_offloading_copy_to_sparsecore=false` (LOW-MEDIUM). v6e has 2 SparseCores; MaxText fsdp_prefetch.md
   (TPU7x): "Keeps data-formatting copies on the TensorCore. Copies offloaded to the SparseCores run on both
   SparseCores at once and block the FSDP collectives queued on either of them." On one chip there are no collectives
   to block; A/B only if the trace shows SparseCore copies on the critical path.
K7 `--xla_tpu_enable_large_2nd_minor_layout_for_x16=true` (LOW-MEDIUM, risky). jax tpu_info.py: "before TPU7x, by
   default XLA does not use large 2nd minor tiling but it can be enabled by setting the flag
   xla_tpu_enable_large_2nd_minor_layout_for_x16." flywheel-tpu: "Before v7x that is 8: XLA tiles a bf16 HBM buffer
   (8, 128), unless xla_tpu_enable_large_2nd_minor_layout_for_x16 is set." Changes every bf16 buffer's tiling and the
   layout at the splash custom-call boundary (may ADD relayout copies). Related, used on v6e/7x microbenchmarks:
   `--xla_tpu_control_large_2nd_minor_layout_for_x16=true` (openxla xla/benchmarks/README.md).
K8 `--xla_tpu_bf16_emission_mode=NATIVE_EMISSION` (LOW; values seen publicly PROMOTE_F32 / NATIVE_EMISSION /
   SLP_VECTORIZER). v6e is the first generation with packed-bf16 VPU math (scaling book), but public use is TPU7x only
   and native bf16 element-wise math can change rounding (violates "same mathematics"); the run's glue is f32 by design.
   libtpu: "--xla_tpu_impure_enable_packed_bf16_math_ops is deprecated. Please use --xla_tpu_bf16_emission_mode in
   TpuCompilationEnvironment."
K9 scan unrolling (JAX side, MEDIUM, generation unstated): picodo README "Note: using `model.unroll=True` significantly
   increases throughput at the cost of longer copilation time." where the model does
   `jax.lax.scan(block_forward, h, weights["blocks"], unroll=cfg.unroll)` (optionally `jax.remat`). For the 8-loop
   remat scan: `unroll=2/4/8` = same maths, bigger program (watch IMEM overlays and compile time).
K10 host/runtime (NONE-LOW for tokens/s): `TPU_PREMAPPED_BUFFER_SIZE` (default 4294967296; raises init time when
   large; wiki: in-memory checkpoint blocking up to 300 s fixed by `export TPU_PREMAPPED_BUFFER_SIZE=34359738368` and
   `export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=34359738368` - only if snapshots are slow), THP `always`
   (start-up/shutdown ~17 s -> ~8 s), `LD_PRELOAD` tcmalloc (default on Cloud TPU VMs).
K11 libtpu version (LOW-MEDIUM, unverified effect): 0.0.50 -> 0.0.53.x is a compiler change (JAX 0.12.1 pins 0.0.53.*);
   no published performance notes; may also change fusion decisions and therefore rounding.
Dead knob: `--xla_tpu_dvfs_p_state=N` - "Note that v6e doesn't support P states." (openxla platform_info.py).

## 9. Multi-chip only (not pursued)
xla_tpu_enable_async_collective_fusion(_fuse_all_gather/_fuse_all_reduce), xla_tpu_enable_ici_ag_pipelining,
xla_tpu_enable_ag_backward_pipelining, xla_tpu_enable_async_all_to_all, xla_tpu_enable_data_parallel_all_reduce_opt,
xla_tpu_data_parallel_opt_different_sized_ops, xla_tpu_megacore_fusion_allow_ags, xla_tpu_enable_megacore_fusion,
xla_tpu_enable_sparse_core_collective_offload_*, xla_tpu_use_single_sparse_core_for_all_gather_offload,
xla_tpu_prefer_async_allgather_to_allreduce, xla_tpu_num_sparse_cores_for_gather_offloading / _scatter_offloading (MoE
gathers), xla_jf_spmd_threshold_for_windowed_einsum_mib, MaxText CF_FOR_ALL_GATHER / CF_FOR_ALL_REDUCE,
megascale_grpc_premap_memory_bytes, xla_tpu_use_enhanced_launch_barrier (set by cloud_tpu_init; launch sync),
MTU 8896 / multi-NIC, dcn_data_parallelism, xla_tpu_host_transfer_overlap_limit (host offload).

## 10. Dead ends
- Cloud TPU v6e page, "Run JAX on TPU", "TPU software versions", "Train a model using TPU v6e": no single-chip flags.
- Cloud TPU release notes: only the Trillium GA note (2024-12-16). libtpu PyPI notes: compatibility only.
- Colab release-notes pages 404 to the fetcher; only the X announcement is quotable.
- No public VMEM-bandwidth figure for v6e (scaling book gives 22x for v5e in a worked example, ~7x for TPU7x).
- THP: no steady-state effect documented anywhere.
- DVFS / P-states: none on v6e; `request_p_state` is TPU7x+.
- Developers-blog microbenchmark article: TPU7x only. picodo "39% MFU on v6e-1": only on a mirror, not in the README.
- flywheel-tpu (Pallas attention library for v6e/v7x, jax[tpu]>=0.11.2): only forward/prefill benchmarks at head_dim
  256 published; swapping the kernel changes rounding -> not a knob for this run.
