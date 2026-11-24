# What a Parameter Budget Does Not Count: An Audited One-Day Training Run of a 98M-Stored, 20M-Read Language Model on One TPU v6e

[TBD: author list and affiliations, at submission]

**Status of this draft (third pass).** Written while the run is in progress: 24 November 2026, pass of
16:50–18:15 UTC. The run trains `configs/v9.py`. Newest record read: the step record of step 46,350 at
16:27:21 UTC (p = 0.8794; the trainer's ledger: 46,351 steps, 12,150,636,544 tokens at 16:27:24), from the
run's snapshot of 16:28 UTC; newest held-out record: step 46,178 (`heldout_v9.jsonl`, evaluated at
16:25:50 UTC). Training ends at 18:33:49 UTC (the run's own setup event; the operator's notes say
18:33:45). A result that does not exist yet is written [TBD: what, when it will exist]. Every number has
an entry in `CLAIMS.md`; numbers derived from logs are recomputed by `lead3_work/make_numbers.py` into
`lead3_work/numbers.json`, and the figures are rebuilt from the same records (cut-off step 46,350).

## Abstract

We train a language model from scratch under a budget stated in parameters (at most 100 million stored
numbers, at most 20 million read per token, 24 hours of one TPU v6e chip) for a blind comparison with
Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M, and report what the budget does not count. In our
implementation, two configurations that differ by 3.1% in numbers read and 3.6% in analytic FLOP per
token differed by 1.62× in step time, and by 1.46× after layout changes that hold arithmetic and reads
fixed. On six configurations, the compiler's count of bytes moved predicts step times within 11%
(leave-one-out); FLOP and reads per token cannot tell five of them apart. A
function-preserving graft changed the configuration at 0.505 of the budget without discarding
7.79 × 10⁹ trained tokens; its loss and shared gradients equal the source's bit for bit on the CPU. In a
controlled version at small width (three seeds), every graft ended below the arm without the new paths,
and a graft at half the budget recovered 0.69 of the gain of training them from the start. On 60
development prompts, Falcon-H1-Tiny was preferred in 39 of 42 decisive pairs by Opus readers and in 16
of 17 by the pre-registered judge; against MobileLLM-R1.5, which Falcon-H1-Tiny beats in 144 of 150
decisive test pairs, our model won 20 of 27 and 7 of 9. The test comparison is [TBD: after the
evaluation of 24–25 November 2026].

## 1. Introduction

Small language models are described by parameter counts: the numbers they store and, for sparse
models, the numbers they read per token ("active" parameters) [32, 33]. Our task fixed a budget of
that kind: at most 100 million stored numbers of at most 16 bits, weights, tables and written knowledge
included; at most 20 million read to produce the distribution of one token; 24 hours of one TPU v6e
chip, from scratch; then a blind pairwise comparison on held-out prompts with
Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M [7, 9, 11]. A dense core that every token reads costs
about 19 million reads, so the other 80 million stored numbers can only be read sparsely: one row of a
table per token, or one passage of a library per retrieval. The run trained one configuration (`final`: dense looped core and a written library) for
the first half of the budget and, at the user's request, a second one (`v9`: the same core and
per-token *pages*, rows of lookup tables indexed by the token id) for the second half. In the
parameter ledger the two are nearly equal: 98.3 and 98.0 million stored, 19.03 and 19.63 million read
per token. This paper is an audited case study of that run, and its subject is what the parameter
ledger does not count. It makes four contributions.

1. **Where the training time goes on one TPU v6e** (Section 5): an intervention series that changes
   layout with arithmetic and reads fixed (1.62× to 1.46×), a post-hoc test of which static count
   predicts the step time (bytes moved, not FLOP or reads), and the design's largest time cost, 16 block
   applications per token, which the ledger counts once.
2. **A function-preserving graft inside a wall-clock budget, verified on the live run and controlled
   at small width** (Section 6). When the user asked for `v9` at p = 0.505, the run was grafted rather
   than restarted: shared leaves keep their optimiser state, the step and the time-tied schedule, and
   new leaves start where their paths add exactly zero. We verify the function and the shared gradients
   at the live source checkpoint, and, because one run cannot say whether a graft is a good way to add
   new paths, repeat it under control at width 128 with three seeds: every graft ended below the arm
   without the new paths, and the share of the from-scratch gain it recovered fell with the graft point
   (1.06, 0.69 and 0.24 at a quarter, a half and three quarters of the budget).
3. **Two ledgers and an audit** (Section 4): stored numbers and numbers read per token in a standard
   and a strict reading, audited by a forward pass that receives only the rows it reads.
4. **The standing so far, read three ways** (Section 7): exact scores, bits per byte, and blind
   comparisons on development prompts read by Opus agents on exchanged sides and by both pre-registered
   judges, with the two opponents compared with each other on the test prompts as a scale.

**What we do not claim.** That per-token pages improve the model at this scale: at width 128 they
lowered held-out loss by 1.24% at equal tokens, but at full scale there is no `v9` trained from scratch
and no `final` trained to the end, and the live data stream, its seed and the phase of the learning
rate changed at the graft. That the live graft was better or worse than restarting or continuing at
equal wall-clock time. That 1.46–1.62× is the price of per-token tables rather than of this
implementation, compiler and chip. That our model beats either opponent: against Falcon-H1-Tiny every
reader that decides pairs points the other way; against MobileLLM-R1.5 the Opus readers' development
lead (20 to 7) is not significant by the pre-registered judge (7 to 2, p = 0.18), is untested on the test
half, and is a lead over a model that Falcon-H1-Tiny beats in 144 of 150 decisive test pairs.
Mathematical novelty: the accounting and the estimators apply known results (an internal adversarial
review rated their novelty 2–3 out of 10, `docs/MATH.md`).

## 2. Setting

### 2.1 The budget

The rules were fixed before training (`docs/PLAN.md` §2, `docs/ACCOUNTING.md`). A **parameter** is one
stored number of at most 16 bits that ships with the model; the **stored total** counts dense weights,
lookup tables, and a written library as its compressed bytes divided by two, with its index, but not
the tokenizer. **Numbers read per token** (standard reading) are the distinct numbers read to produce
the next-token distribution of one position given the cached state: a dense weight counts once however
often it is applied, a lookup table counts the rows read, the convention behind active parameters
[32]. The **strict reading** is the union over the context window. Limits: at most 100 million stored and
20 million read per token in the standard reading; 24 hours of TPU v6e time on one chip, from scratch,
counted as the time a runtime is allocated and not the waits between runtimes (the operator's reading,
reported to the user on 23 November).

### 2.2 Hardware, software and terms

One TPU v6e chip ("TPU v6 lite" to JAX; 31.24 GiB of device memory available to the compiler) on a
44-core host; JAX and jaxlib 0.11.2, libtpu 0.0.50. The two feed-forward products of a block ran at 814
TFLOP/s in the forward pass when timed alone (`docs/TPU_NOTES.md`); we use that measured rate as the
ceiling. A **micro-batch** is 16 rows of 1,024 tokens (8 of 2,048 from p = 0.67); an optimiser **step**
accumulates 16 micro-batches, 262,144 tokens; on the device each micro-batch is cut into 4 **pieces**.
**Exact** means bit-identical; **equal to rounding** means equal in exact arithmetic, with the
floating-point difference stated. A **live rate** is the median, over the trainer's
step records (one per 50 steps) in a stretch, of the rate each record logs: 262,144 tokens ÷ the running
average of the time per step (weight 0.2 on the newest step) in the time ledger's training time, which
leaves out checkpoint writes (`bes/train/train.py`); n is the number of records. The rate achieved
between records (wall time) differs from it by at most 0.13% in every stretch. A **synthetic** time is a
whole step on a synthetic batch on an idle chip. Times of day are UTC.

### 2.3 The opponents

Falcon-H1-Tiny-90M-Instruct runs as its publisher's Q8_0 file [86] through llama.cpp at llama.cpp's
default sampling (the publisher gives none); MobileLLM-R1.5-140M runs from a community Q8_0 file
rewritten losslessly as a `qwen3` model, with its publisher's sampling and budget forcing [73]
(reasoning closed after 3,072 tokens, 1,024 answer tokens), in two arms (`docs/EVAL_PROTOCOL.md` §2).
Falcon-H1-Tiny-90M-Instruct was trained on 8 × 10¹¹ tokens [11], MobileLLM-R1's pre-training used
4.2 × 10¹² [7, 9]; our run will see about 1.36 × 10¹⁰ (Section 6.3).

### 2.4 The run

The 24 hours were spread over three runtimes (Appendix C, Table C1, Figure C2); the second was taken back
by the provider about 2 h 12 min after its allocation, for a reason we could not establish. Training
started on 23 November at 17:46:32 (`configs/final.py`), was grafted into `configs/v9.py` at step 29,722
(7.79 × 10⁹ tokens) and ends at 18:33:49 on 24 November; the 118 minutes before the run's deadline
(20:31:49) are reserved for preference optimisation, decoding selection, sampling and export. At the newest record (16:27:24), 19.68 of the 24 hours had been used, 15.36 of them
(78%) in training steps of the kept trajectory. From p = 0.67 (step 36,867) the rows are 2,048 tokens
long and the chat share of the stream rises linearly from 0.25 to 0.70 at p = 0.92 (Appendix C.1).

## 3. Related work

A fuller account is in Appendix N. **Training to a deadline.** Cramming [1] and Izsak et al. [2] tie the
learning-rate decay to a one-day budget, as budgeted training recommends [3]; speedruns minimise time to
a target on an 8 × H100 node [4, 5]. Our schedule is of this kind, for a chat model on one chip that had to
survive a reclaimed runtime and a change of configuration. **Small models and weight-shared depth.**
MobileLLM [6–9], Falcon-H1 [10, 11], SmolLM2 [12] and TinyStories [13] set the scale; re-applied blocks
[14–19] and our core's source, the iso-depth law of [21] with hyper-connections [20]; we loop because
under a read budget a re-applied weight counts once. **Vocabulary-indexed parameters
and two counts.** Stored-but-sparsely-read capacity includes hashed feed-forward weights [23],
key-value memories [24, 25], larger input vocabularies [26], off-accelerator n-gram embeddings [27, 28]
and L3's per-token keys and values [29], to which `v9`'s token neurons are closest; token-keyed
parameters helped most on values and outputs [30, 31]. Counting numbers read per token is the
mixture-of-experts convention [32, 33]; we found no published read-set audit. **Changing a model during
training.** Net2Net [45], network morphism [46], staged training [47] (which copies Adam
moments for existing layers, zeroes them for new ones, and asks that loss and training dynamics be
preserved), LLaMA Pro [48], sparse upcycling [49] and LoRA [50] start new paths at an identity or at
zero; our graft is a loss-preserving operator of this family. What is particular is narrow: vocabulary-indexed tables and a feed-forward
branch are added, the schedule is tied to the wall clock, the function and the shared gradients were
checked at the live checkpoint, and a controlled replication measures how much of the from-scratch
gain a graft at a given point recovers; the training-dynamics property of [47] is not tested.
**Kernels and judges.** Our kernels follow [58–62]; transformer training is known to be memory-bound on
GPUs [63]. Language-model judges favour one position, longer replies and their own style [78–81]; we
count a win only when both orders agree [78], report outcomes by length ratio [80], and read the same
pairs with judges of two other families.

## 4. The model and its two ledgers

### 4.1 The model

Figure 1 shows one position of the model; Appendix B gives the equations. A host-side NumPy layer
computes causal features of the tokens alone: integer *registers* (role, parities, depths, counts,
countdowns) and a *candidate tape* of up to 16 candidate next tokens (successors of earlier occurrences
of the current n-grams, with their counts). The dense core (d = 768, 6 heads of width 128, feed-forward
width 3,072, ReLU² [84]) applies two weight-shared blocks in each of 8 loops, with two hyper-connection
lanes [20] between loops and a per-loop value embedding. The output law combines the network's softmax
q with the tape in closed form, p(v) = (q(v) + Σ_k m_k [c_k = v]) / (1 + Σ_k m_k), with masses m_k ≥ 0
learned from the candidates' count features; with all masses zero it equals the network's
cross-entropy bit for bit. Activations are bfloat16; normalisation statistics, logits and the loss are
float32.

**Token pages (`v9`).** Every vocabulary id owns, in each block i, ρ = 8 private ReLU² neurons in a
192-dimensional space shared by all tokens, and a gain row on the 768 attention-output channels:

    z_t = RMSNorm(x_t P_in,i)             a_{t,j} = ⟨z_t, RMSNorm(K_i[x_t, j])⟩ / √192
    Π_i = ( Σ_j relu(a_{t,j})² U_i[x_t, j] ) P_out,i        attention output (y_t ⊙ (1 + G_i[x_t])) W_o,i

K, U and G are lookup tables; P_in (768 × 192) and P_out (192 × 768) are dense. At initialisation
U = 0 and G = 0, so Π_i = 0 and the gain multiplies by 1, exactly (Section 6.2).

![Figure 1](figures/F7_model.svg)

**Figure 1.** The model at one token position, with stored numbers / numbers read per generated token
(standard reading) for each part, from the two checkpoints' manifests and `final`'s export accounting
(`data/ledger_leaves.csv`, `data/ledger_final_export.csv`), as in Table 1. Gray: shared by both
configurations; blue: `final` only (the written library); orange: `v9` only (attention-output gains and
private token neurons in each block). The 15 small dense leaves (306,776 numbers) are counted in both
totals and not drawn. A schematic: only the counts are data.

### 4.2 The ledgers and their audit

`bes/budget.py` computes both ledgers from the real parameter tree, and tests enforce the limits
(Table 1; full ledger in Appendix A).

**Table 1. Stored numbers and numbers read per token** (standard reading; `docs/V9.md` §5,
`docs/ACCOUNTING.md`, the checkpoints' manifests).

| part | `final` stored | `final` read | `v9` stored | `v9` read |
|---|---|---|---|---|
| shared: looped core, head, norms and lanes, embedding, value embedding, register and tape tables | 34,531,720 | 19,033,736 | 34,531,720 | 19,033,736 |
| written library and its index | 63,768,391 | 245,223 on a retrieving step (worst case), else 0 | – | – |
| attention-output gains | – | – | 12,582,912 | 1,536 |
| token neurons: tables K, U | – | – | 50,331,648 | 6,144 |
| token neurons: projections P_in, P_out | – | – | 589,824 | 589,824 |
| **total** | **98,300,111** | **19,278,959** worst case | **98,036,104** | **19,631,240** |

The reads differ by 597,504 numbers (3.1%), of which 589,824 are the two dense projections. The ledger
is audited as a claim about a function: `bes/readset.py` is a NumPy forward pass for one position that
receives only the rows the ledger says are read; it agrees with the JAX model to 1e−4, and poisoning
every row it was not given leaves its output unchanged. On 40 real documents of at least 2,048 tokens `v9`
reads 19.43 million in the standard reading (19.63 million counting the critic head) and, in the strict
reading, a median of 23.8 million at 1,024 positions and 26.7 million at 2,048: every distinct token of
the window adds 9,536 numbers. So `v9` meets the standard reading and not the strict one
(`docs/ACCOUNTING.md`, `docs/V9.md` §10; Appendix A).

### 4.3 What the ledgers do not count

Both ledgers count numbers once, at inference. They do not count how often a number is applied, the
activations it produces, the layouts it is stored in, or anything about training (gradients, optimiser
state, recomputation). The core's 14.1 million weights are applied 16 times per token and counted once;
a token's row of each page table (3,072 numbers) is counted once, while its gathered copy is carried
through all 8 loops and its cotangent accumulated over them in training. Section 5 measures what the
clock charges for.

## 5. Where the time goes on one TPU v6e

### 5.1 Method

**On the chip**, the live run was stopped and programs were timed on synthetic batches on the idle chip,
after warm calls (five pauses, 43 min 04 s of TPU time; Appendix C). A single piece is the median of
repeated calls (30 for `final`, 6 for `v9`); a whole step was timed once, as the mean of a few steps run
back to back (2 by default; the count used was not recorded), so whole steps carry no spread.
**Compile-only**, libtpu 0.0.50 compiles for `v6e:1x1` on a CPU host (jax 0.11.1), and we read from the
optimised program the compiler's estimated cycles and the logical bytes of every instruction's operands
and outputs in device memory (HBM); Pallas kernels count 0 cycles; cycles were calibrated at 0.521 ms per
million on two shapes of one program family. Section 5.4 tests these estimates against the chip.

### 5.2 The dense core is bound by glue, not by its matrix products

The first numbers from the chip (23 November) were a third of the plan's assumption: `final` trained at
115–117 thousand tokens/s (two monitor readings; start-up benchmark 111,893), where the plan had assumed
300–450 thousand. Inside one block application (15.4 ms with recomputation and backward pass, at 32 × 1,024 tokens) the
matrix products took about 2.4 ms and the attention kernel 3.4 ms; **the other 9.6 ms (62%) was outside
both**: slices and concatenations, transposes between the model's and the kernel's layouts, float32
copies of the widest tensors, and element-wise work (`docs/TPU_NOTES.md`). One 768 × 768 product took
0.08 ms forward: one pass over an activation tensor costs about as much as the product that made it.

Three kinds of change followed (Table 2; Appendix F, Figure F2). Five rewrites of the glue, equal to the
reference up to rounding (28 tests) with one deliberate extra rounding (the feed-forward pre-activation
stored in bfloat16, 2⁻⁹ relative), took one `value_and_grad` of a 32 × 1,024 micro-batch from 289.3 to
162.0 ms. Cutting each micro-batch into 4 pieces on the device equals the original update up to
rounding. A one-tile attention kernel that holds a (row, head) slice in on-chip memory and covers only
the causal triangle made one piece 1.212× faster at 1,024 tokens and 1.219× at 2,048 than the library's
splash kernel [61]; its gradients on the chip differ from the configured kernel's by about twice what a
change of compiled graph alone gives, so its equality there is not established (Appendix F.2, Figure F1).

**Table 2. Training throughput of the run, by cause** (live: median of the logged step rates, with n;
synthetic and benchmark values are whole-step timings; `lead3_work/numbers.json`, `docs/TPU_NOTES.md`,
`docs/CHANGES.md` §19.1–19.7; monitor readings from the operator's log of runtime 1).

| stage | configuration | live tokens/s (n) | synthetic or benchmark tokens/s |
|---|---|---|---|
| reference implementation, runtime 1 (23 Nov) | `final`, 32-row micro-batches | 115,394 and 116,718 (two monitor readings; runs discarded) | 111,893 (start-up benchmark) |
| the rewrites above, runtime 1 | same | – (runs discarded) | 201,877 (start-up benchmark) |
| 16-row micro-batches, runtime 2 | `final` | 222,031 (101) | 216,681 (start-up benchmark) |
| the same program on runtime 3 | `final` | 216,548 (85); −2.5%, not explained | – |
| 4 pieces per micro-batch (step 9,332) | `final` | 240,131 (266) | 246,538 |
| one-tile kernel + ReLU² backward (step 22,613) | `final` | 281,389 (142) | 296,371 (with the `ce` backward on) |
| graft (step 29,722) | `v9`, `take` table gradient | 178,087 (44) | 182,919 |
| `flat`, `rows`, `tokn` (step 31,912) | `v9` | 196,874 (99) | 203,665 |
| rows of 2,048 tokens (p = 0.67, step 36,867) | `v9` | 189,074 (190) | – (not timed with the switches) |

From the reference to the one-tile kernel the live rate of `final` rose 2.41–2.44× (281,399 over steps
25,300–29,700, against the two monitor readings), including a change of configuration (32 to 16 rows per
micro-batch, +8% by itself); around the kernel switch it rose 17.2% (240,143, n = 32, to 281,387, n = 53).
The live rate is about 5% below the synthetic one; the trainer's main thread used 2.9% of a core. At 2,048
tokens `v9` runs 4.0% below its rate at 1,024.

### 5.3 The cost of the pages: an intervention series

Table 3 and Figure 3 compare the two configurations on the same chip, kernels, table-gradient mode
(`take`, a scatter-add) and pieces, then add three switches to `v9` that change only layout (commit
`769f240`): `flat` hands the loss function its
token tables as reshapes of `[V, W]` matrices (exact); `rows` keeps the residuals of the key
normalisation and carries the page rows through the loop scan in a lane-dense layout (exact); `tokn` lays
the token neurons out with tokens in lanes (equal to rounding).

**Table 3. `final` against `v9` on the chip** (whole steps 64 × 4 × 1,024 on synthetic batches;
`wf/chip_kernels_0745.txt`, `S/v9_measA.txt`, `S/v9_measD.txt`; live medians from the train logs).

| configuration | analytic GFLOP per token | reads per token | one piece (ms) | whole step (ms) | ratio to `final` |
|---|---|---|---|---|---|
| `final` (step measured with the `ce` backward on) | 0.781 | 19,033,736 | 14.639 | 884.5 | 1.000 |
| `v9` as grafted | 0.809 | 19,631,240 | 22.79 | 1,433.1 | 1.620 |
| … + `flat` | 0.809 | 19,631,240 | 22.59 | 1,406.4 | 1.590 |
| … + `rows` | 0.809 | 19,631,240 | 20.44 | 1,295.2 | 1.464 |
| … + `tokn` | 0.809 | 19,631,240 | 20.01 | 1,287.1 | 1.455 |
| live: `final` (steps 25,300–29,700) against `v9` before / after the switches | | | | 281,399 against 178,087 / 196,874 tokens/s | 1.580 / 1.429 |

So 3.1% more reads and 3.6% more analytic arithmetic cost 62% more time per step at the switch. The
three switches held the arithmetic and the reads fixed and removed 146 of the 549 extra milliseconds
(27%), `rows` alone 111; the remaining 403 ms per step have not been attributed by any measurement. The
compiler's cost model attributes `v9`'s extra cycles mostly to the token neurons' per-token `[8, 192]`
contractions, which the chip's tiles pad from 8 to 16 sublanes and from 192 to 256 lanes
(`wf/v9_speed.md`); but `tokn`, the switch aimed at them, was predicted to save 64 ms and saved 8 (Figure
4b). Before the core was rewritten `v9` already cost about 0.6 s more per step than `final`; at the
switch, 0.55 s: as the core became faster, the pages did not.

![Figure 3](figures/F4_v9cost.svg)

**Figure 3.** One accumulation step of 262,144 tokens on a synthetic batch, timed on the chip, by
configuration, table-gradient mode and piece size; ratios against `final`'s fastest step of the same
length. Each bar is one timing (Section 5.1); `final` timed at 07:44–07:52, `v9` at the switch and with the
three switches added one by one. Black ticks: 262,144 ÷ the live rate, with n: Table 3's windows and, at
2,048 tokens, steps 36,900–46,350 (to 16:27:21 UTC on 24 November, the cut-off of the paper's live
numbers; n = 190, 1,386.5 ms), not timed there on a synthetic batch. `final` was timed with the
hand-written cross-entropy backward on, which the run does not use (0.177 ms slower per piece).

### 5.4 Which count predicts the step time?

For six configurations both a compiled program and a chip time exist. Table 4 and Figure 4a ask how well
a straight line in each of four static counts predicts the measured step when that configuration is left
out of the fit: an exploratory test, made after the measurements, on six points, five of them layout
variants of one model.

**Table 4. Static counts against the measured step** (`lead2_work/predictor_test.py`; compiled
programs `wf/v9_*_work/hlo/` read by their `estimate.py`; chip times as in Table 3).

| configuration (whole step) | analytic GFLOP per token | reads per token (M) | estimated cycles (M per step) | estimated HBM bytes (GB per step) | measured (ms) |
|---|---|---|---|---|---|
| `final`, `take`, 64 × 4 × 1,024 | 0.781 | 19.03 | 1,944.7 | 515 | 884.5 |
| `v9`, `take`, 64 × 4 × 1,024 | 0.809 | 19.63 | 3,141.8 | 1,277 | 1,433.1 |
| `v9`, `take`, 32 × 8 × 1,024 | 0.809 | 19.63 | 3,237.6 | 1,527 | 1,491.5 |
| `v9`, `sorted`, 64 × 4 × 1,024 | 0.809 | 19.63 | 3,390.7 | 1,623 | 1,789.0 |
| `v9`, `take` + `flat` | 0.809 | 19.63 | 3,053.6 | 1,237 | 1,406.4 |
| `v9`, `take` + `flat` + `rows` | 0.809 | 19.63 | 2,825.6 | 1,130 | 1,295.2 |
| **leave-one-out error, mean (largest)** | not defined without `final`; 10% (21%) on the five `v9` | as FLOP | 12% (40%) | **6% (11%)** | |
| rank correlation with the measured time | 0.66 (ties) | 0.66 (ties) | 1.00 | 1.00 | |

Both of the compiler's counts order all six configurations as the chip does. FLOP and reads cannot
order the five `v9` variants (1,295–1,789 ms): on them a line in either count predicts the mean of the
other four steps. The byte count predicts best; a line through all six gives 492.5 ms plus 0.73 ms per
GB, with residuals from −110 to +118 ms. This is consistent with bytes moved, not parameters, setting the
step time in this implementation, on a test too small to establish it; the attention kernels, invisible
to the compiler's model, and the half-second intercept are explained by no count. Differences are harder than totals: for the seven changes with both
a compile-only prediction and a chip measurement, the cost model had the sign right every time and the
size wrong by factors from 0.36 to 7.9 (Figure 4b; Appendix F.7, Table F5).

![Figure 4](figures/F10_costmodel.svg)

**Figure 4.** The compiler's estimates against the chip. (a) The six configurations of Table 4, sorted by
the measured step (bar; one timing each). Markers: each static count's leave-one-out prediction of the
step, joined to the bar by its error; the key gives the mean and the largest absolute error of each
count. FLOP and reads per token give the same predictions; for `final` their fit is undefined and no
marker is drawn. (b) The seven changes of Table F5: predicted against measured change of the step,
absolute values on logarithmic axes; triangles up: the change made the step slower, down: faster; the
number after each label: predicted ÷ measured. Six configurations chosen after the measurements: the
figure shows how the counts did here, not that bytes moved predict step time in general.

### 5.5 What the read ledger hides in the design itself

The largest gap between ledger and clock in this model is not the pages but the loops: the read ledger
counts the 16 block applications per token once, the clock charges every one. In the reference
implementation one loop instead of eight took 42.9 against 289 ms per micro-batch, so 97% of the time was
in the 16 block applications; after the rewrites a block application took about 8 ms and the rest of the
micro-batch 34 ms. At equal wall clock, 4 loops would have trained on about 1.6–1.9 times the tokens. In the micro ablation, at equal tokens, 4 loops were 0.12% better
than 8 on held-out loss and 12 loops 0.26% worse, both within its noise (`docs/E2_ABLATIONS.md`); under
the iso-depth law of [21] with two lanes (φ = 0.65), halving the loops divides the effective recurrent
size by 2^0.65 = 1.57. Whether more tokens would have made up for it at full scale was neither predicted
with the law's fitted constants (fitted with a prelude and coda this core lacks) nor tested. The loop
count was set by compute per number read, and the clock was never asked. (At its live rate `final`
executed 27% of the 814 TFLOP/s by the analytic count and 41% counting what the chip executes, Appendix
F.6; thirteen changes measured on the chip were not adopted, Table F4.)

## 6. Changing the design mid-run: a function-preserving graft

### 6.1 The decision

At 09:16 UTC on 24 November the user asked to switch to `configs/v9.py`. The run was at p ≈ 0.50 with
7.79 × 10⁹ tokens and 30,969 s of training left, in which a `v9` from scratch would have seen about
5.4 × 10⁹ tokens at the rates measured at the switch, or 6.0 × 10⁹ with the later switches of Section 5.3
(a projection): fewer than the run already had. The run was therefore continued in the new parameter
tree. The switch was the user's decision: a pre-registered comparison of the two configurations was not
run (Appendix C.5), and the evidence for the pages was small-scale and unresolved (Appendix B.4).

### 6.2 Function-preserving continuation

Let the target configuration have parameters θ = (θ_s, θ_n), where θ_s are the leaves the source also
has (same path, shape and type) and θ_n the new ones. Call an initialisation θ_n⁰ *neutral* if
f_target( · ; θ_s, θ_n⁰ ) = f_source( · ; θ_s ) for every θ_s. For `v9`, U = 0 and G = 0 make θ_n⁰ neutral
whatever K, P_in and P_out are, and in floating point too: with U = 0 the token-neuron output is a sum of
exact zeros, and with G = 0 the gain multiplies by exactly 1. At the graft point (1) every loss equals the
source's; (2) in exact arithmetic the gradient with respect to θ_s equals the source's, so with the
carried optimiser state the first update of θ_s is the source's (there is no gradient clipping); (3) of
the new leaves only U and G receive a non-zero gradient at the first step, while those of K, P_in and
P_out vanish exactly; (4) Adam on the new leaves runs with the carried update count, so for a constant
gradient its update is 0.894 times a fresh run's at the first step and 1.44 times at the seventh, the
maximum (β = 0.8, 0.95; Appendix G).

### 6.3 What is carried and what is new

`scripts/graft_run.py` (commit `9a4d652`; Figure 5; Appendix G) carries the parameters and optimiser
slots of the 59 shared leaves (34,531,720 numbers), the update count, the step, the trainer's time ledger
and loss guard; the 5 new leaves (63,504,384 numbers) start at their initialisation with fresh slots.
Because the time ledger is carried, the schedule continues: p went from 0.5049 at the stop to 0.5112 at
the restore, and the learning-rate factor continued on 1 − 0.9p (0.5394 at step 29,750). The data stream
cannot be carried: the mixture changed (no library rows), it restarted with another seed, and the
loader's plan needed two corrections (Appendix G). At the newest record `v9` trains at 189,074 tokens/s
(2,048 tokens, n = 190). Projected from the trainer's ledger at 16:27:24 (1.2151 × 10¹⁰ tokens) at that
rate, the run will have seen about 1.358 × 10¹⁰ tokens at 18:33:49, about 2.33 × 10⁹ (14.6%) fewer than
`final` would have seen without the switch, by a projection in which `final`'s own 2,048-token rate is a
scaled estimate [TBD: tokens trained, from the trainer's ledger at 18:34 UTC].

![Figure 5](figures/F6_graft.svg)

**Figure 5.** Stored numbers of the two configurations at step 29,722, to scale; colours as in Figure 1.
Gray: the 59 shared leaves, carried with their optimiser slots; blue: `final`'s library and index, which
`v9` does not have; orange: `v9`'s five new leaves at initialisation, with fresh slots. Right: what else
was carried, what is new, and the checks of Section 6.4, read from the run's logs and manifests
(`data/graft_facts.csv`, `data/ledger_leaves.csv`).

### 6.4 Verification

`--check` compares the loss of the source parameters under the source configuration with that of the
grafted parameters under the target configuration on a synthetic batch of 2 × 256 tokens (CPU, float32,
dense attention) and refuses unless they agree to 1e−6 relative. At four source checkpoints, among them
the switch's own checkpoint 29,722 (re-run under the live configuration hash, because the live
invocation ran without `--check`), the two losses agree to every printed digit (Appendix G, Table G1).
On real held-out rows (2 web and 2 chat rows of 1,024 tokens) with the source checkpoint of step 29,722,
the two programs give the same loss to the last bit (2.9320383 and 1.5048378), bit-identical gradients
for all 59 shared leaves, exactly zero gradients for K, P_in and P_out, and non-zero gradients for U and G
(`lead2_work/graft_grad.py`). Both checks are on the CPU in float32 with dense attention; on the chip the
grafted program is another compiled graph in bfloat16, where equality of the shared gradients holds to
rounding and was not measured.

### 6.5 Held-out loss across the graft

The held-out rows are those of the export's check: 64 web rows (a ClimbMix file beyond the run's
download cap), 64 chat rows (the smol-smoltalk test split) and 12 rows of the user turns of the 400
evaluation prompts, 1,024 tokens each, in float32 with dense attention. The last 12 hold test-prompt
turns and are left out: the comparison set is the 128 web and chat rows (125,880 weighted tokens). The
runtime evaluated 3 checkpoints of `final` and 16 of `v9` (Figure 6; Table G2); we evaluated four of them
again on the CPU, keeping the loss of every position and switching the new paths off; the local values
reproduce the runtime's to within 10⁻⁷.

**Table 6. Held-out loss across the graft** (web + chat rows; `lead2_work/heldout_analysis.json`,
`lead3_work/heldout46.json`, `lead3_work/heldout3.json`; step 34,166 in Appendix G, Table G3).

| comparison (web + chat) | difference (nats) | 95% interval |
|---|---|---|
| `v9` 35,294 − `final` 29,722 (same rows, 128.8 more training minutes) | −0.0605 | [−0.0653, −0.0557] (bootstrap over rows) |
| `v9` 46,178 − `final` 29,722 (379.2 more minutes) | −0.1679 | [−0.1828, −0.1533] |
| … web rows / chat rows only | −0.1019 / −0.2393 | [−0.1106, −0.0937] / [−0.2560, −0.2235] |
| `v9` − `final`'s line at equal training time (line in minutes / in log tokens), at 35,294 | −0.033 / −0.038 | ±0.066 / ±0.052 (prediction, 1 degree of freedom) |
| … at 46,178 | −0.088 / −0.112 | ±0.171 / ±0.111 |
| `v9` − `final`'s line at equal tokens (not the budget's axis), at 35,294 / at 46,178 | −0.044 / −0.127 | ±0.040 / ±0.084 |
| pages off − on, at 35,294 / at 46,178 | +0.0643 / +0.2166 | [0.0594, 0.0695] / [0.2019, 0.2320] |
| `v9` with pages off − `final` 29,722, at 35,294 / at 46,178 | +0.0038 / +0.0487 | [−0.0009, 0.0086] / [0.0394, 0.0582] |

Four readings, in order of how much they assume. (i) On the same rows the loss fell by 0.061 nats in the
129 training minutes after the graft and by 0.168 in 379 (web 0.102, chat 0.239; before the graft, 0.009
in 44 minutes); but the learning rate decays, and from p = 0.67 (between steps 36,422 and 37,522) the chat
share of the stream rises from 0.25 toward 0.70 (Appendix C.1), which lowers the chat loss of any
configuration. (ii) The budget's question, equal training time against a `final` that was not continued,
has one estimate: `final`'s trend through three checkpoints spanning 44 minutes, extended (Figure 6).
`v9` lies 0.03 nats below it at step 35,294 and 0.09–0.11 at 46,178, inside 95% prediction intervals of
±0.05 to ±0.17; at equal tokens, not the budget's axis, the gap at 46,178 (−0.127 ± 0.084) excludes zero.
The line models neither the decaying learning rate nor the chat ramp, which a continued `final` would
also have had and which favour `v9` here, and it keeps one residual degree of freedom because earlier
`final` checkpoints were deleted to free disk space before they could be evaluated. The gap is not
resolved.
(iii) Switched off at evaluation, the pages cost 0.064 nats at step 35,294, about the whole improvement
since the graft, and 0.217 at 46,178, more than it: with them off, the network at 46,178 is 0.049 nats
worse than at the graft point (web +0.062, chat +0.034). The trained network depends on the pages; this
decomposes one trained model and is not a counterfactual, since the shared parameters adapted to them.
(iv) At 46,178 the pages carry 0.299 nats on the rarest quarter of held-out positions (1–16 occurrences
of the input token) and 0.159 on the most frequent quarter (497–4,425).

![Figure 6](figures/F8_heldout.svg)

**Figure 6.** Held-out loss on the comparison set of Table 6 against training minutes, checkpoints to
step 46,178 (917.9 minutes). Filled points: the runtime's evaluations; rings: evaluated again on the CPU;
open diamonds: `v9` with its pages switched off (steps 34,166 and 35,294; at 46,178, 2.2754, not drawn).
Gray line and band: the least-squares line through `final`'s three points, extended, with its 95%
prediction interval (one residual degree of freedom); arrow: `v9`'s distance from it at the newest
checkpoint, −0.088 ± 0.171 (label cut at the right edge in this build). Every `v9` point lies inside the
band; the line models neither the decaying learning rate nor the chat ramp from p = 0.67, and the data
stream changed at the graft: the figure does not show that `v9` is ahead of a continued `final`.

### 6.6 The graft under control: a replication at the micro width

One run cannot say how a graft compares with training the new paths from the first step, or with not
adding them, so we repeated the graft with the repository's own model and micro bench, everything else
held equal (`docs/MICRO_GRAFT.md`; Appendix G.4; code at commit `b04478e`, reviewed adversarially before
the runs, `wf/micro_graft.md`).

**Design.** The model is the live model's code at width 128 (2 weight-shared blocks × 8 loops, rows of 512
tokens, 4,096 tokens per step). Five arms, each with seeds 0, 1 and 2: `ref`, the micro reference
(4,402,372 numbers); `v9_pages`, the reference with the `v9` page from the first step (gains and 8
private token neurons per token and block in a 32-dimensional space; 14,904,516 numbers); and `graft_25`,
`graft_50`, `graft_75`: `ref` up to 0.25, 0.5 or 0.75 of the budget, then grafted into `v9_pages` by the
live tool's own function (`graft_run.graft_arrays`), which carries the shared leaves with their optimiser
slots, update count and step, while the 5 new leaves (10,502,144 numbers) take the values of a fresh
`v9_pages` tree of the same seed (U = G = 0). The budget is 6 × 10⁶ tokens (1,464
steps), with held-out evaluations at a quarter, a half, three quarters and the end; the grafts fall on
steps 366, 732 and 1,098. Within a seed all arms share the data order, the schedule (p = step/1,464) and
the initial values of the shared leaves. Held-out loss is measured on 32 web and 32 chat rows from other
files; "mix" weights them 3:1, like the training rows.

**Checks.** Each graft arm trained its own trunk, and its evaluations before the graft equal those of
`ref` of the same seed row for row, bit for bit. At each of the nine grafts the held-out loss of every
domain, and the training loss of 8 held-out web rows, are the same before and after: the difference is
exactly 0, per row as well.

**Table 7. The graft at the micro width** (held-out loss, mix of web and chat, nats per token, at
5,996,544 tokens; mean ± standard deviation over seeds 0, 1, 2; differences paired by seed; share
recovered = (`ref` − arm) / (`ref` − `v9_pages`) as the ratio of the mean differences, per seed in
brackets; `lead3_work/micrograft.py` on `micro_graft/6M/results_all.jsonl`, which reproduces the
experiment's own `graft_report.txt`; restart arms from `micro_graft/3M/`).

| arm | new paths from | loss | arm − `ref` | arm − `v9_pages` | share recovered |
|---|---|---|---|---|---|
| `ref` | never | 4.1912 ± 0.0076 | – | +0.0519 ± 0.0107 | 0 |
| `v9_pages` | step 0 | 4.1394 ± 0.0083 | −0.0519 ± 0.0107 (−1.24%) | – | 1 |
| `graft_25` | 0.25 of the budget | 4.1363 ± 0.0075 | −0.0550 ± 0.0010 | −0.0031 ± 0.0113 | 1.06 (0.86, 1.33, 1.08) |
| `graft_50` | 0.5 | 4.1555 ± 0.0091 | −0.0358 ± 0.0030 | +0.0161 ± 0.0096 | 0.69 (0.62, 0.87, 0.64) |
| `graft_75` | 0.75 | 4.1789 ± 0.0074 | −0.0123 ± 0.0004 | +0.0396 ± 0.0103 | 0.24 (0.20, 0.29, 0.24) |
| restart: `v9_pages` from scratch on half the budget (own schedule) | step 0 of the half | RESTART_LOSS | RESTART_REF | RESTART_V9 | RESTART_SHARE |

**Reading** (Figure 7). With the page from the first step, held-out loss ended 0.052 nats (1.24%) below
`ref` (web 0.046, chat 0.068), a gap that grew through training (0.016, 0.033, 0.049 and 0.052 at the four
evaluations). Every graft ended below `ref` in every seed (by 0.054–0.056, 0.033–0.039 and 0.012–0.013
nats for the grafts at 0.25, 0.5 and 0.75). Grafted at a quarter of the budget, the page ended where the
page from the first step ended (−0.0031 ± 0.0113, signs mixed across seeds); grafted at half, it recovered
0.69 of that gain and ended above it in all three seeds; at three quarters, 0.24. The shares differ by at
most 0.04 between web, chat and assistant-turn losses. In the 1.5 × 10⁶ tokens after its graft, a grafted
page gained 0.019, 0.016 and 0.012 nats over `ref` (at learning-rate factors 0.775, 0.55 and 0.325),
against 0.016 for the page in its first 1.5 × 10⁶ tokens from scratch: a late graft loses mainly the time
it does not have.
RESTART_READING

**Limits, stated here.** Width 128 against 768; 6 × 10⁶ tokens against about 1.36 × 10¹⁰, about one pass
over small pools; three seeds, so standard deviations with 2 degrees of freedom and shares whose
denominator is itself a difference of two runs (the per-seed shares of `graft_25` range from 0.86 to
1.33); a schedule in tokens, not wall-clock time, so every arm sees the same tokens and the page's cost
in step time (1.43–1.58× live on the chip) does not enter; a data stream that continues unchanged across
the graft, where the live one changed its seed and dropped the library rows; and a page worth 1.2% of
the loss here. The experiment is evidence about the mechanism of a graft, not about the size of its
effect in the live run.

![Figure 7](figures/F12_micrograft.png)

**Figure 7.** The graft at the micro width (Table 7): held-out loss (mix of web and chat rows) of `ref`,
`v9_pages` and the grafts at 0.25, 0.5 and 0.75 of the budget at 1.50, 3.00, 4.50 and 6.00 × 10⁶ tokens,
mean of seeds 0–2 (a graft arm is `ref` up to its graft, row for row), and the share of the
from-scratch gain each graft recovered, per seed (0.86, 1.33, 1.08; 0.62, 0.87, 0.64; 0.20, 0.29, 0.24).
Width 128, schedule in tokens: the mechanism of a graft, not the size of its effect in the live run.
[Caption written from the data; the figures agent's caption in `FIGURES.md` is to be reconciled with it.]

### 6.7 What one run cannot show, and what the controlled experiment settles

The graft lets a run change its design without discarding what it has trained. For the mechanism, the
controlled experiment settles three things at width 128: a graft is function-preserving in practice
(difference exactly 0 at nine grafts); continuing in the grafted configuration never ended worse than
not grafting (9 of 9 runs); and the share of the from-scratch gain a graft recovers falls with the graft
point, about 0.7 at half the budget, consistent with new paths that learn at about their from-scratch
rate and run out of time. RESTART_SETTLES It leaves open, for the live run: whether the pages help at width 768 and
1.36 × 10¹⁰ tokens; whether the graft beat a restart or a continuation at equal wall-clock time, where
`v9`'s steps cost 1.43–1.58× those of `final` and the run will see about 2.3 × 10⁹ fewer tokens than
`final` would have (a projection); what the changed data stream and seed did; how the new leaves behaved
training only in the low-rate half of the schedule (learning-rate factor 0.54 down to 0.1); and
run-to-run variation at full scale. Any comparison of the shipped model with the opponents measures all
of these together.

## 7. Evaluation: protocol, deviations and the standing so far

### 7.1 Protocol

The protocol (`docs/EVAL_PROTOCOL.md`) is in the branch's root commit (`5ff33c5`, 23 November 00:35 UTC)
and unchanged since: it predates every TPU run and every reply of the full-scale model, not the
micro-scale runs of gates E2, E4 and E5. **Prompts**: 400 original English prompts in five categories, a
quarter with a second user turn, 95 with machine-checkable constraints, split by the SHA-256 of their
identifier into `dev` and `test` (200 each); decisions use `dev`, and `test` is used once. Sides are seeded
and balanced, and the scorer refuses any rating whose commitment to the hidden key does not match. **Readers**: the registered
judge `qwen3-4b-2507`, read on its next-token distribution over A, B and C in both orders, with
`gemma4-e4b` of another family as a second judge; a side wins only if both orders choose it. **Exact
scores** come first and are trusted over a reader. **Statistics**: the score (wins + ties/2)/n with a
Wilson interval, an exact two-sided sign test, slices by category, turns and length ratio.

### 7.2 Deviations from the protocol

1. **The readers.** Both registered judges were run on the four development pair sets in both orders
   (Table 8). The same pairs, and the test pairs, are also read by Opus 5.5 agents, of the family of the
   agents that built the system, twice on exchanged sides: an addition reported next to the registered
   judges.
2. **Eleven test prompts were seen** by mid-run CPU samples (chat-002, chat-017, math-034, instruct-034,
   instruct-062, writing-100, code-026, math-001, code-002, writing-001, instruct-001; checked with
   `bes.eval.proxy.prompts.load()`); two of them motivated the data rules of Appendix D. Every test result
   will also be reported on the 189 untouched test prompts (R11).
3. **The proxy held-out rows contain test-split user turns** (Section 6.5); they are left out of
   comparisons.
4. **MobileLLM's two arms.** The protocol credits MobileLLM-R1.5 with the better of its two arms on `dev`;
   only the `recommended` arm has been compared blind [TBD: the `neutral` arm on `dev`, before the test
   split is opened].
5. **At training time** (Appendix B.5): an E2 loss reading chosen after seeing the data; registers kept
   despite failing their gate; the E5 copy criterion judged wrong after the fact; the pre-registered
   configuration comparison not run; the configuration changed mid-run.

### 7.3 What 200 comparisons can resolve

The protocol's power table assumed ties at up to 30%. On the development pairs of `v9`, ties (including
"both bad" and orders that disagree) were 18 and 33 of 60 for the Opus readers (against Falcon-H1-Tiny
and MobileLLM-R1.5) and 43 and 51 for the registered judge. At 30% and 55% ties, n = 200 detects with 80%
power (two-sided 5% sign test) a decisive win rate of 0.5 + 0.117 and 0.5 + 0.146; at the registered
judge's 72% and 85%, only 0.5 + 0.182 and 0.5 + 0.246; on the 189 untouched prompts, 0.121, 0.150, 0.187
and 0.252 (`bes.eval.proxy.score.n_required`; `lead3_work/numbers.json`). A non-significant test result
will therefore leave open differences of 12 to 25 points.

### 7.4 The standing so far

Tables 8 and 9 and Figure 8 collect what has been measured against the opponents; none of it is the
planned test.

**Table 8. Blind comparisons by reader** (W / L / T from the first-named model's side, T including "both
bad" and pairs whose two orders disagree; score (wins + ties/2)/n with 95% Wilson interval; exact
two-sided sign test; orders: pairs on which a reader's two orders agree; first: mean preference for the
first-shown reply; `lead3_work/readers.py` on `study/proxy/ratings/`, which reproduces every report in
`study/proxy/reports/ab/`; ours with default decoding; first 60 `dev` prompts).

| pair set | reader | W / L / T | score [95%] | sign test p | orders; first |
|---|---|---|---|---|---|
| `final` 25,280 (p = 0.44) vs Falcon-H1-Tiny | Opus, one reading | 2 / 43 / 15 | 0.158 [0.087, 0.271] | < 0.0001 | – |
| | `qwen3-4b-2507` (registered) | 2 / 16 / 42 | 0.383 [0.271, 0.510] | 0.0013 | 63.3%; −0.096 |
| | `gemma4-e4b` (registered second) | 0 / 4 / 56 | 0.467 [0.346, 0.591] | 0.13 | 80.0%; −0.087 |
| `final` 25,280 vs MobileLLM-R1.5 | Opus, one reading | 15 / 8 / 37 | 0.558 [0.433, 0.677] | 0.21 | – |
| | `qwen3-4b-2507` | 7 / 4 / 49 | 0.525 [0.401, 0.646] | 0.55 | 78.3%; −0.000 |
| | `gemma4-e4b` | 2 / 0 / 58 | 0.517 [0.393, 0.638] | 0.50 | 91.7%; −0.008 |
| `v9` 31,912 (p = 0.565) vs Falcon-H1-Tiny | Opus, two readings | 3 / 39 / 18 | 0.200 [0.118, 0.318] | < 0.0001 | 93.3%; +0.000 |
| | `qwen3-4b-2507` | 1 / 16 / 43 | 0.375 [0.264, 0.501] | 0.0003 | 60.0%; −0.114 |
| | `gemma4-e4b` | 0 / 3 / 57 | 0.475 [0.354, 0.599] | 0.25 | 83.3%; −0.065 |
| `v9` 31,912 vs MobileLLM-R1.5 | Opus, two readings | 20 / 7 / 33 | 0.608 [0.482, 0.722] | 0.019 | 98.3%; +0.008 |
| | `qwen3-4b-2507` | 7 / 2 / 51 | 0.542 [0.417, 0.661] | 0.18 | 80.0%; −0.000 |
| | `gemma4-e4b` | 2 / 1 / 57 | 0.508 [0.385, 0.630] | 1.0 | 93.3%; +0.008 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5, 200 `test` prompts | Opus, two readings | 144 / 6 / 50 | 0.845 [0.788, 0.889] | < 0.0001 | 93.5%; +0.013 |

**Three readers.** Against Falcon-H1-Tiny every reader that took sides took Falcon-H1-Tiny's in almost
every pair: the Opus readers in 39 of 42 decisive pairs (`v9`) and 43 of 45 (`final`), the registered judge
in 16 of 17 and 16 of 18, the second judge in all it decided (3 and 4). Pair by pair (`v9`), where the Opus
readers and the registered judge both chose a side they chose the same one in 14 of 16 pairs against
Falcon-H1-Tiny and in 6 of 6 against MobileLLM-R1.5; the second judge's 3 and 3 sides all agreed with the
Opus readers. The readers differ in how much they decide: the registered judge's two orders agreed on only 60–63% of
the Falcon-H1-Tiny pairs and it preferred the second-shown reply, so most pairs the Opus readers decided
were ties for it (26 of 42 against Falcon-H1-Tiny, 21 of 27 against MobileLLM-R1.5; Cohen's κ over win,
loss and tie 0.21 and 0.18); the second judge answered C (equal or equally poor) in 93–97% of the pairs
and has almost no resolution here. Against MobileLLM-R1.5 the Opus
readers found a lead for `v9` (20 / 7, p = 0.019), mostly on instruction-following (9 / 0 / 3) and
two-turn prompts (7 / 0 / 8), with every decisive maths pair lost (0 / 5 / 4); the registered judge found
7 / 2 (p = 0.18). Both automatic judges cut replies longer than 4,000 characters, in 30 of the 60
MobileLLM-R1.5 pairs. Agreement between orders is not validity: no human has read these pairs.

**A scale.** On the 200 test prompts, read the same way, Falcon-H1-Tiny beat MobileLLM-R1.5 in 144 of 150
decisive pairs, in every category (chat 37 / 1, code 17 / 2, instruct 31 / 0, maths 15 / 3, writing 44 /
0). This set holds no reply of ours and does not open the test split for our model. MobileLLM-R1.5, a
reasoning model run with its publisher's sampling and budget forcing, is weak on these prompts: on the
development prompts 42 of its 75 replies ran into the token limit (Table 9), and the readers' notes on
the reference pairs describe its reply as looping or repeating in 87 and 83 of the 200 pairs (two
readings) and as a bare boxed answer in 46 and 49 (keyword count, `lead3_work/ref_notes.py`). A lead
over MobileLLM-R1.5 on these prompts is a low bar; the informative comparison is the one with
Falcon-H1-Tiny, where our model is far behind. The development and test prompts are two halves of one
set, so the scale is indicative, not a calibration.

**In absolute terms.** In the first reading of the pair set in which both replies were ours (`v9` at step
31,912, the default decoding against 3-gram blocking), both replies were judged bad in 44 of 60 pairs;
the readers' notes on those 44 name wrong facts or arithmetic in 20 and incoherent text in 14 (keyword
count, `lead3_work/ours_notes.py`; for instance "Both wrong; correct is 12,600. A says 30,000, B says 12.").
Against MobileLLM-R1.5, 31 of 60 pairs were judged "both bad", against Falcon-H1-Tiny 14 of 60; for
`final` at step 25,280, 46 to 55 of 60 pairs per decoding comparison (Appendix E). At p = 0.56, most of the
model's replies to these prompts were judged bad, mainly for wrong content and incoherence.

**Table 9. Exact scores and bits per byte** (the same 60 `dev` prompts, 75 turns, default decoding;
`lead2_work/exact_dev.json`, `lead2_work/opp_bpb_web.json`, `lead3_work/heldout46.json`).

| measure | ours, `final` 25,280 | ours, `v9` 31,912 | Falcon-H1-Tiny | MobileLLM-R1.5 (`neutral` arm) |
|---|---|---|---|---|
| maths answers correct (9 prompts) | 0 | 1 | 4 | 4 (1) |
| constrained prompts with every constraint met (14) | 12 | 12 | 9 | 2 (1) |
| replies cut at the token limit (75) | 2 | 0 | 0 | 42 (54) |
| stock refusals / deflections (75) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| median words per reply | 93 | 96 | 188 | 532 (698) |
| bits per UTF-8 byte, 64 held-out web rows (our training distribution) | 1.017 (step 29,722) | 0.998 (35,294); 0.980 (46,178) | 1.570 | 1.500 |

The exact scores agree with the readers: both opponents answer more maths prompts correctly, and our
replies meet constraints more often and are shorter than the opponent's in 38 to 45 of the 60 pairs. On
held-out web text from our own training distribution (ClimbMix, 62% of `final`'s training tokens), our
model needs 0.98 bits per byte at step 46,178 and the opponents 1.57 and 1.50 on the same 258,814 bytes;
this favours us by construction (instruction-tuned opponents read on raw web text, training data
unknown), and next-token loss on the pre-training distribution said little about preference in chat
here. Two decoding studies on the same prompts could not separate any setting from the default
(Appendix E).

![Figure 8](figures/F11_blind.svg)

**Figure 8.** Blind pairwise comparisons on the first 60 development prompts, Opus readers (Table 8;
the registered judges' readings are in the table): `final` at step 25,280 read once per pair, `v9` at step
31,912 read twice on exchanged sides. Bars: wins, ties or "both bad", losses; dots and whiskers: the score
with its 95% Wilson interval (about ±12 points at n = 60); the two studies differ in checkpoint,
configuration and number of readings, so the change between them is not an effect of `v9`. The
test-split comparison is [TBD: after the evaluation of 24–25 November 2026].

### 7.5 Results to come

The quantities the evaluation will fill are table shells R1–R12 in Appendix M, each with the file it
will come from [TBD: after the evaluation of 24–25 November 2026].

## 8. Limitations and threats to validity

1. **One run, one seed at full scale**, with no full-scale baseline for the design change (no `v9` from
   scratch, no `final` continued, no graft at another point). The controlled experiment supplies these
   at width 128 and 6 × 10⁶ tokens only, at equal tokens rather than equal wall-clock time (§6.6). At
   micro scale a change of seed moved held-out loss by 0.42% (E2) and `ref`'s three seeds have a standard
   deviation of 0.18% (§6.6).
2. **Mid-run changes confound every curve**: the training stream changed at steps 6,647, 25,280 and
   29,722 and its chat share ramps up from step 36,867; the learning-rate factor jumped at step 12,062;
   the compiled graph changed at steps 9,332, 22,613 and 31,912 (equal to rounding).
3. **One implementation, one chip, one software stack.** The 1.46–1.62× and the predictor test hold
   for this code, jax 0.11.2 (0.11.1 for the compiles), libtpu 0.0.50 and one TPU v6e; the predictor test
   has six points chosen after the fact, and compile-only estimates of differences were wrong by up to a
   factor of 8 (Table F5).
4. **Equality on the chip is to rounding and partly unverified** for the adopted kernel (§5.2) and the
   graft's gradients (§6.4).
5. **The readers.** The registered judge decides few pairs and prefers the second-shown reply; the
   second judge decides almost none; the Opus readers decide most pairs but are of the builders' family;
   no human has read the pairs [78–81]. Two readings on exchanged sides remove pure position preference,
   not a bias both readings share.
6. **The prompts are ours, and eleven test prompts were seen** (§7.2); the resolution of the test is
   limited (§7.3); our model selects among six particles while the opponents draw one sample (no matched
   control is planned), and the reply budgets differ (768 of our tokens; 1,024 for
   Falcon-H1-Tiny; 1,024 answer tokens after up to 3,072 of reasoning for MobileLLM-R1.5).
7. **Opponent settings** (llama.cpp defaults for Falcon-H1-Tiny; a converted community file with budget
   forcing for MobileLLM-R1.5) may understate either opponent.
8. **Small laboratories** produced the evidence for the pages and the components (widths 48 and 128,
   1–12 million tokens; Appendix B) and the controlled graft (width 128, 6 million tokens).
9. **The data rules' effects are not measured**, and from step 6,647 the model is not trained to refuse
   anything (Appendix D).
10. **The token gap** to the opponents (about 59 times fewer tokens than Falcon-H1-Tiny and about 310
    times fewer than MobileLLM-R1's pre-training) cannot be separated from the design in any final
    comparison.

## 9. Conclusion

A budget in parameters is a statement about a function: how many numbers it stores and how many it
reads to produce a token. The cost of training that function on one accelerator for a day is a statement
about bytes, shapes and repetitions. In this run the two came apart three ways: two configurations that
the ledger calls nearly equal differed by 1.62× in step time, a quarter of it removed by layout alone;
the compiler's count of bytes moved predicted step times where FLOP and reads could not, on six
configurations and after the fact; and the loop count, 97% of the step time in the reference code, is
invisible to the read ledger. A function-preserving graft let the run change its design at half time
without discarding 7.79 × 10⁹ trained tokens. Under control at width 128, grafting never ended worse than
not grafting, and a graft at half the budget recovered about 0.7 of the gain of having the new paths from
the start; this supports the mechanism and says nothing about the size of the effect or the wall-clock
trade at full scale. On development prompts the model is far behind Falcon-H1-Tiny by every reader that
decides pairs, and ahead of MobileLLM-R1.5 by the Opus readers but not significantly by the registered
judge; Falcon-H1-Tiny is far ahead of MobileLLM-R1.5 on the test prompts, so the informative comparison
is the one the model loses. Whether the finished model is preferred to either opponent is decided by a
protocol fixed in advance, with the deviations stated here [TBD: Appendix M, R1, after the evaluation].

