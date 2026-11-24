# speed_loops: the per-loop controller, the loop scan and final_hidden

**Result: no switch delivered. The patch is empty.**

At the live run's shapes (16 x 1024, and 8 x 2048, both 16,384 tokens per micro-batch), XLA
already fuses this area close to what its dataflow allows. I evaluated one structural candidate
(carrying the next loop's input) with exact-equality tests and a compile-only TPU v6e build. The
build makes it 4.9% worse, so I dropped it. Every other candidate is bounded below 1% before
implementation. For one of them, the lane-coefficient gradients the ledger flagged, I measured
the bound directly: removing those gradients outright saves 0.54%.

Worktree: `/home/user/turingaward2027/.claude/worktrees/wf_3d459e66-645-7`. It is clean, so
`speed_loops.patch` is empty. Nothing was committed, no configuration field was added, and no
other engineer's file or function was touched.

## 1. What the ledger says about the area

The depth-2 ledger lines belonging to the area are `controller.py` (combine, update, run_loops),
the loop-end `rmsnorm(u, g_r)` called from `forward.<locals>.step`, and `head.final_hidden`.
Script: `speed_loops_work/tools/ledger_area.py`.

| | glue pass | share | materialised | share |
|---|---|---|---|---|
| 16 x 1024 | 14,024 MB | 16.0% of 87,465 | 5,621 MB | 11.4% of 49,104 |
| 8 x 2048 | 14,024 MB | 15.7% of 89,086 | 5,621 MB | 11.4% of 49,517 |

These lines do not depend on T. The largest are:
- `controller.py:55` lane converts: 2,013 MB.
- `controller.py:61` mix-gradient reductions: 1,611 MB.
- `controller.py:41/43` combine: 1,409 MB each.
- loop-end rmsnorm: about 3,400 MB in total.
- `controller.py:62` update outputs: 1,208 MB.

## 2. What the chip program actually does (compile-only, TPU v6e, libtpu 0.0.50)

The ledger cannot see fusion or memory placement, so I compiled the real `value_and_grad(loss_fn)`:
- `configs/final.py` with `table_grad=take`, `micro_seqs=16`, at 16 x 1024.
- Built with the libtpu 0.0.50 wheel already in the scratchpad (the chip's version), through
  `jax.experimental.topologies`, in the same way as `bes/train/aot.py`.
- 250 s per compile, 1.8 GB host memory, no TPU.

From the optimised HLO I read, per kernel:
- the compiler's own `estimated_cycles`;
- the HBM bytes, aware of memory spaces: buffers the compiler placed in alternate memory `S(1)`
  cost nothing, and async copies and slices count their HBM side once.

Tools are in `speed_loops_work/tools/`: `tpu_hlo.py`, `hlo_traffic.py`, `hlo_cycles.py`,
`hlo_diff.py` and `hlo_show.py`.

Program totals:
- 140.6 M estimated cycles, excluding the Mosaic kernels, which carry no estimate.
- 68.9 GB of HBM traffic.
- Forward loop body: 3.40 M cycles per trip. Backward body: 12.70 M per trip.

Findings:

1. **The two lanes live in alternate memory for the whole forward pass.** The `while` carry is
   `bf16[16,1024,768]{...S(1)}` twice.
   - The forward combine (fusion.1527: combine plus the block-0 norm statistics) and the
     loop-end norm plus update (fusion.1549) move **0 bytes of HBM**.
   - They cost 84.8 k and 88.8 k cycles per trip, bound by the vector unit.
   - The lanes' only forward HBM traffic is the remat residual write (50 MB per trip, 137.5 k
     cycles).
2. **The 8 wide lane-coefficient reductions the ledger counts as about 4.4 GB are not separate
   passes on the chip.** XLA puts them into two multi-output fusions:
   - fusion.1500: the 4 mix and 2 alpha gradients, together with the tail of block 0's norm
     backward and `ct_u`. It reads 151 MB per trip and costs 257 k cycles.
   - fusion.1457: the 2 beta gradients, together with the update transpose and the loop-end norm
     backward. It costs 117 k cycles.
3. **The remaining backward cost** is the recomputed combine (fusion.1429: 50 MB read, 135 k) and
   the lane cotangents (fusion.1503: 100 MB, 193 k). Per trip, the lanes are read twice and
   their cotangents three times. The minimum given XLA's VMEM decisions is two and two; the
   difference is 50 MB per loop.
4. **The rounding points are preserved on TPU.** The compiled lane-cotangent fusion keeps every
   f32 -> bf16 -> f32 round trip of the jaxpr, so the chip's numerics follow the jaxpr. This
   matters for section 4.

The whole area, per micro-batch of 16 x 1024 (compiler cycles):

| part | cycles | share of 140.6 M |
|---|---|---|
| forward, 8 trips (combine, norm and update, residual write, scalar slices) | 2.59 M | 1.8% |
| backward, 8 trips (remat combine, update transpose, coefficient reductions, lane cotangents, scalars) | 5.80 M | 4.1% |
| outside the loops: lane-1 init copy 257 k, backward carry init copy 124 k, ve-cotangent stack zero-init 378 k, final_hidden about 508 k | 1.27 M | 0.9% |
| **total** | **about 9.7 M** | **about 6.9%** |

How cycles map to milliseconds is approximate. The compiler's cycles overstate time: scaled to
the measured 73.3 ms of a 16 x 1024 micro-batch, minus about 18 ms of splash kernels, 1 M
cycles is roughly 0.39 ms. On that scale the whole area is about 3.8 ms per 16 x 1024
micro-batch. The brief's "3 ms x 8 per 32 x 1024" was measured at 32 x 1024, where 50 MB lanes
do not fit next to the blocks in alternate memory. It does not describe the live shapes.

**Calibration of the cost model.** I compiled `model.remat_keep=kernel` and diffed it against
the baseline. The estimate is +5.18%; the chip measured it 1.8% slower at 32 x 1024. The
direction agrees. The magnitude is overstated, partly because the estimate cannot see the
Mosaic kernel time that this variant saves. The candidate below changes no kernel, so the
estimate applies to it more directly.

## 3. Candidates

| # | candidate | exact? | effect (compile-only v6e, 16 x 1024) | verdict |
|---|---|---|---|---|
| A | `FAST_CARRY_NEXT`: each step ends by forming the next step's input `u_{r+1} = combine(alpha[r+1], lanes')` and carries it (scan carry `(lanes, u)`; a custom-VJP start state keeps the bf16 order in which `ct_e` is summed) | yes, with excess precision off (section 4): the loss is identical in every case. All 59 gradient leaves are bit-identical with xla attention and no remat. With splash + remat, 58 of 59 are bit-identical and `lanes/alpha` differs by 4e-8 relative (f32 reduction order) | **+6.84 M cycles (+4.87%)**, +4.1 GB HBM; ledger glue +604 MB pass, +1,611 MB materialised, scan xs/ys +403 MB | **dropped**: a third 25 MB tensor in the carry overflows alternate memory. Blocks spill (`mm` +0.93 M, block-0 norm +0.65 M), the residual write grows by 25 MB per loop, and XLA splits the merged transpose into three large fusions. Prototype kept in `speed_loops_work/prototype_carry_next.patch` |
| B | cheaper lane-coefficient gradients (the ledger's 4.4 GB of reductions on lines 41/43/59/61) | n/a | upper bound by **deleting them outright** (diagnostic `stop_gradient`, not a valid program): **-0.757 M cycles (-0.54%)**, -403 MB HBM | **dropped**: no exact reformulation can beat the bound. Turning them into MXU products would only move the 4.4 GB out of the ledger's glue column |
| C | merge fusions 1500 and 1503 so the lane cotangents are read once at the end of the backward step | yes | at most 50 MB per loop, about 0.4 M cycles (about 0.3%) | not attempted: needs XLA to fuse a consumer into a multi-output reduction, or a Pallas kernel, for 0.3% |
| D | write the remat residual from the update fusion (save the carry output instead of the input; needs a hand-written scan VJP) | needs care | at most the whole residual-write fusion: 1.10 M cycles (0.78%) | not attempted: below 1% even if free, and rewriting the scan transpose (bf16 accumulation order of about 30 closed-over weight cotangents) is a large exactness risk |
| E | pack alpha/beta/mix into one per-loop vector (21 tiny slice/pad/DUS kernels per loop at about 2.1 k cycles each) | yes | at most about 0.29 M cycles (0.2%) | dropped: below 1% |
| F | avoid the copy of `e` into the second lane buffer (forward) and of `bf16(ct_s)` into the second cotangent buffer (backward) | yes | 0.38 M cycles (0.27%) | dropped: XLA's CSE removes every way of writing it from JAX without a copy |
| G | ve-cotangent stack zero-init (`broadcast` of bf16[8,16,1024,128], once per micro-batch) | n/a | 0.38 M cycles (0.27%) | dropped: lowering artifact. JAX zero-fills scan output buffers; XLA turns the forward's into `AllocateBuffer` but not the reversed scan's. Feeding ve differently trades it for a worse accumulator |
| H | final_hidden in bf16 | no | n/a | not possible: `predictive.log_mass` uses the float32 `h` |
| - | block gains, loop phases, gate index fed per loop | - | - | off in `configs/final.py`, so nothing of them is in the live program |

Ledger numbers for A are a `--diff` against my baseline JSON
(`speed_loops_work/base_1024.json`, same totals as the shared baseline): glue 87,465 -> 88,069 MB
pass and 49,104 -> 50,715 MB materialised. The ledger and the compiler agree that A is worse.

**Expected effect: 0 ms**, because there is no switch. For scale, per 32 x 1024-token
equivalent (two 16 x 1024 micro-batches, 0.39 ms per M cycles):
- A would cost about +5.3 ms.
- The coefficient-gradient bound B is about -0.6 ms.
- C, D, E, F and G together are below about 2 ms. They would need custom kernels or a
  hand-written scan VJP.

## 4. A pitfall for every bf16 exactness test on CPU

With default XLA flags on CPU (`xla_allow_excess_precision=true`), two programs with identical
jaxpr rounding points can give different bf16 results when their fusion context differs. For
candidate A, loss and forward were identical, but 41 of 59 gradient leaves differed by up to
1.3e-2 of the leaf maximum in bf16. In float32 they were identical except `lanes/alpha`
(8e-8, reduction order).

Compiling with `jax.jit(f).lower(*args).compile(compiler_options={"xla_allow_excess_precision":
False})`, or setting `XLA_FLAGS=--xla_allow_excess_precision=false`, removes the effect:
- xla attention, no remat: **bit-identical, all 59 leaves**;
- splash + remat: 58 of 59 bit-identical, and `lanes/alpha` differs by 4e-8 relative, which is
  only the f32 summation order of its reductions.

The TPU program keeps the round trips (section 2, point 4). So bf16 equality
tests of restructurings should compare compiles with this flag off. Otherwise they measure CPU
fusion artefacts, not the change. Tool: `speed_loops_work/tools/check_switch.py`.

## 5. Tests

- No switch, so no new test file.
- The existing `tests/model/test_attention_major.py::test_lanes_as_a_tuple_and_as_a_stack_agree`
  passes on the unchanged checkout.
- Candidate A was checked with `check_switch.py` (configs from `tests/model/_model_testlib.py`
  `tiny_cfg`, with `clock_planes=0`, `use_conf=False`, sink, tape, registers and value
  embeddings on):
  - splash + remat + bf16: with excess precision off, loss identical and 58 of 59 leaves
    bit-identical (alpha 4e-8); with it on, 41 leaves differ by up to 1.3e-2;
  - xla + no remat + bf16: with excess precision off, all 59 leaves bit-identical; with it on,
    bf16-level differences;
  - float32 (splash + remat): equal except alpha at 8e-8.

## 6. Files

All paths are under `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/`.

- `speed_loops.patch`: empty.
- `speed_loops.md`: this report.
- `speed_loops_work/hlo/{base,carry_next,keep_kernel,diag_nocoefgrad}.hlo`: optimised TPU HLO of
  the baseline, candidate A, the calibration point and the diagnostic.
- `speed_loops_work/hlo/base_traffic.txt`: per-kernel HBM traffic of the loop bodies.
- `speed_loops_work/base_1024*.txt|json`, `base_2048_d2.txt`, `explain_controller.txt`: ledger
  runs.
- `speed_loops_work/prototype_carry_next.patch`: the rejected candidate A.
- `speed_loops_work/tools/`:
  - `tpu_hlo.py`: compile-only build, takes `--set` and `--override`; `--diag nocoefgrad` is a
    diagnostic only.
  - `hlo_traffic.py`, `hlo_cycles.py`, `hlo_diff.py`, `hlo_show.py`: reading the optimised HLO.
  - `check_switch.py`: off/on loss and gradient comparison.
  - `ledger_area.py`: sums this area's ledger lines.

  Other engineers can reuse these for their own candidates: one compile per variant, then
  `hlo_diff.py base.hlo new.hlo`.
