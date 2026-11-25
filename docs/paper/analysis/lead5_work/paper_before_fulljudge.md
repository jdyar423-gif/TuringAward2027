# What a Parameter Budget Does Not Count: An Audited One-Day Training Run of a 98M-Stored, 20M-Read Language Model on One TPU v6e

[TBD: author list and affiliations, at submission]

**Status of this draft (final pass).** Written after the run ended: 24 November 2026, 21:10–23:00 UTC
(what changed: `RESPONSE_3.md`). Records: the run's last packed snapshot (`N/snap_final`, 20:16 UTC; train log
to the end event of step 51,812 at 18:33:54), the stage ledger, the official export, the test replies and
every blind reading of the test pairs finished by the time stated in Section 7 and Appendix M. Every number
has an entry in `CLAIMS.md`; numbers derived from the records are recomputed by the scripts of
`lead5_work/scripts/` (`numbers.json`, `final_results.json`, `exact_counts.json`, `heldout5.json`), and the
figures are rebuilt from the same records (last step record 51,800). Still [TBD]: the registered judge's full
pass and MobileLLM-R1.5's two K = 6 control arms, both in progress at the time of writing, and the author
list.

## Abstract

We train a language model from scratch under a budget stated in parameters (at most 100 million stored
numbers, at most 20 million read per token, 24 hours of one TPU v6e chip), compare it blind with
Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M, and report what the budget does not count. The run used
86,054 of the 86,400 TPU seconds, 62,893 of them in training steps, and trained 1.358 × 10¹⁰ tokens; the
official model is the run's own export after preference optimisation (98.0 million stored numbers, 19.6
million read per token). On the 200 held-out test prompts, read twice on exchanged sides by Opus 5.5 agents
(an addition to the pre-registered protocol, of the builders' family), it was preferred to MobileLLM-R1.5 in
116 of 131 decisive pairs (score 0.753 [0.688, 0.807]) and not to Falcon-H1-Tiny, which was preferred in 108
of 143 (our score 0.318 [0.257, 0.385]), and again when given six samples per reply, kept by its own
log-probability or by our selector (112 and 116 of 147 and 145); with one particle instead of six our model
loses 22 to 122 and still wins 99 to 16; the pre-registered judge, part-way through its pass, and the 189
prompts no mid-run sample touched point the same way. The lead is over replies of which
63% degenerate under MobileLLM-R1.5's recommended preset in our harness; our edge in meeting constraints (38
of 48 prompts against Falcon-H1-Tiny's 33) is our decoder's, since Falcon-H1-Tiny with our selector meets 42;
and it is one run. Preference optimisation moved blind preference its way (59 to 34 decisive pairs, p = 0.012). Two
configurations within 3.1% in numbers read and 3.6% in FLOP per token differed 1.62× in step time (1.46× after
layout changes); on six configurations the compiler's count of bytes moved predicted step times within 11%
(leave-one-out) where FLOP and reads could not tell five apart. A function-preserving graft changed the
configuration at half time without discarding 7.79 × 10⁹ trained tokens; at width 128 and 6 × 10⁶ tokens, at
equal tokens, every graft ended below the arm without the new paths and a graft at half the budget recovered
0.69 of the gain of having them from the start (0.62–0.87 by seed), sizes that do not transfer to the live
run.

## 1. Introduction

Small language models are described by parameter counts: the numbers they store and, for sparse
models, the numbers they read per token ("active" parameters) [32, 33]. Our task fixed a budget of
that kind: at most 100 million stored numbers of at most 16 bits, weights, tables and written knowledge
included; at most 20 million read to produce the distribution of one token; 24 hours of one TPU v6e
chip, from scratch; then a blind pairwise comparison on held-out prompts with
Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M [7, 9, 11]. A dense core that every token reads costs
about 19 million reads, so the other 80 million stored numbers can only be read sparsely. The run trained one
configuration (`final`: dense looped core and a written library) for the first half of the budget and, at the
user's request, a second one (`v9`: the same core and per-token *pages*, rows of lookup tables indexed by the
token id) for the second half, then spent 88 minutes on sampling, preference optimisation, decoding selection
and export by rules fixed before the end. In the parameter ledger the two configurations are nearly equal
(98.3 and 98.0 million stored, 19.03 and 19.63 million read per token). This paper is an audited case study of
that run: what the ledger does not count, and how the finished model compares. It makes four contributions.

1. **Where the training time goes on one TPU v6e** (Section 5): an intervention series that changes
   layout with arithmetic and reads fixed (1.62× to 1.46×), a post-hoc test of which static count
   predicts the step time (bytes moved, not FLOP or reads), and the design's largest time cost, 16 block
   applications per token, which the ledger counts once.
2. **A function-preserving graft inside a wall-clock budget, verified on the live run and controlled
   at small width** (Section 6). When the user asked for `v9` at p = 0.505, the run was grafted rather
   than restarted: shared leaves keep their optimiser state, step and schedule, and new leaves start where
   their paths add exactly zero. We verify the function and the shared gradients at the live checkpoint
   and repeat the graft under control at width 128 with three seeds: evidence about the mechanism, whose
   sizes do not transfer to the live run.
3. **Two ledgers and an audit** (Section 4): stored numbers and numbers read per token in a standard
   and a strict reading, audited by a forward pass that receives only the rows it reads.
4. **The outcome under a protocol fixed in advance, and what limits it** (Section 7): the official model
   on the 200 test prompts against both opponents, read by Opus 5.5 agents (an addition to the protocol)
   and, part-way, by the pre-registered judge; exact scores of every model, including two control arms that
   give Falcon-H1-Tiny six samples; the weights before and after preference optimisation compared blind;
   and the trajectory of mid-run checkpoints.

**What we do not claim.** That per-token pages improve the model at this scale: there is no full-scale `v9`
trained from scratch and no `final` trained to the end, and the data stream, its seed and the phase of the
learning rate changed at the graft. That the live graft was better or worse than restarting or continuing at
equal wall-clock time. That 1.46–1.62× is the price of per-token tables rather than of this implementation,
compiler and chip. That our model is better than Falcon-H1-Tiny: every reader that decides pairs prefers
Falcon-H1-Tiny. That the lead over MobileLLM-R1.5 measures our model more than that opponent's state in our
harness, where 63% of its test replies degenerate. That our constraint score is the model's (it is the
decoder's; Section 7.6). The
accounting and the estimators apply known results (an internal adversarial review rated their novelty 2–3
out of 10, `docs/MATH.md`).

## 2. Setting

### 2.1 The budget

The rules were fixed before training (`docs/PLAN.md` §2, `docs/ACCOUNTING.md`). A **parameter** is one
stored number of at most 16 bits that ships with the model; the **stored total** counts dense weights,
lookup tables, and a written library as its compressed bytes divided by two, with its index, but not
the tokenizer. **Numbers read per token** (standard reading) are the distinct numbers read to produce
the next-token distribution of one position given the cached state: a dense weight counts once however
often it is applied, a lookup table counts the rows read [32]. The **strict reading** is the union over the
context window. Limits: at most 100 million stored and 20 million read per token in the standard reading; 24
hours of TPU v6e time on one chip, from scratch, counted as the time a runtime is allocated (the operator's
reading, reported to the user on 23 November).

### 2.2 Hardware, software and terms

One TPU v6e chip ("TPU v6 lite" to JAX; 31.24 GiB of device memory available to the compiler) on a
44-core host; JAX and jaxlib 0.11.2, libtpu 0.0.50. The two feed-forward products of a block ran at 814
TFLOP/s in the forward pass when timed alone (`docs/TPU_NOTES.md`); we use that measured rate as the
ceiling. A **micro-batch** is 16 rows of 1,024 tokens (8 of 2,048 from p = 0.67); an optimiser **step**
accumulates 16 micro-batches, 262,144 tokens; on the device each micro-batch is cut into 4 **pieces**.
**Exact** means bit-identical; **equal to rounding** means equal in exact arithmetic, with the
floating-point difference stated. A **live rate** is the median, over the trainer's step records (one per
50 steps) in a stretch, of the rate each record logs (262,144 tokens ÷ a running average of the step time,
checkpoint writes left out; `bes/train/train.py`); n is the number of records, and the rate achieved between
records differs from it by at most 0.13% in every stretch. A **synthetic** time is a whole step on a
synthetic batch on an idle chip. Times of day are UTC.

### 2.3 The opponents

Falcon-H1-Tiny-90M-Instruct runs as its publisher's Q8_0 file [86] through llama.cpp at llama.cpp's
default sampling (the publisher gives none); MobileLLM-R1.5-140M runs from a community Q8_0 file
rewritten, without touching a quantised value, as a `qwen3` model (mean KL 0.003 nats per position
against the original tensors), with its publisher's sampling and budget forcing [73]
(reasoning closed after 3,072 tokens, 1,024 answer tokens), in two arms (`docs/EVAL_PROTOCOL.md` §2).
Falcon-H1-Tiny-90M-Instruct was trained on 8 × 10¹¹ tokens [11], MobileLLM-R1's pre-training used
4.2 × 10¹² [7, 9]; our run trained 1.358 × 10¹⁰ (Section 6.3), 59 and 309 times fewer.

### 2.4 The run

The 24 hours were spread over three runtimes (Appendix C, Table C1, Figure C2); the second was taken back
by the provider about 2 h 12 min after its allocation, for a reason we could not establish. Training
started on 23 November at 17:46:32 (`configs/final.py`), was grafted into `configs/v9.py` at step 29,722
(7.79 × 10⁹ tokens) and stopped at its deadline, 18:33:49 on 24 November, at step 51,812 (13,582,204,928
tokens). From p = 0.67 (step 36,867) the rows are 2,048 tokens long and the chat share of the stream rises
linearly from 0.25 to 0.70 at p = 0.92 (step 48,190), from where conversations train on their assistant tokens
only (Appendix C.1). The stages after training ran from 18:33:55 to 20:02:22 (Appendix C.4), and the TPU was
given back at 20:41:04: 86,054 of the 86,400 TPU seconds were used, 62,893 (73.1%) in training steps (Table M9).

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
training.** Net2Net [45], network morphism [46], staged training [47], LLaMA Pro [48],
sparse upcycling [49] and LoRA [50] start new paths at an identity or at zero; our graft is a
loss-preserving operator of this family, applied to vocabulary-indexed tables under a wall-clock schedule,
checked at the live checkpoint and replicated under control; the training-dynamics property that [47] also
asks for is not tested. **Kernels and judges.** Our kernels follow [58–62]; transformer training is
known to be memory-bound on GPUs [63]. Language-model judges favour one position, longer replies and their
own style [78–81]; we count a win only when both orders agree [78], report outcomes by length ratio [80], and
set the registered judge of another family beside our added readers.

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
(standard reading) for each part, as in Table 1 (`data/ledger_leaves.csv`, `data/ledger_final_export.csv`).
Gray: shared by both configurations; blue: `final` only (the written library); orange: `v9` only (gains and
token neurons). The 15 small dense leaves (306,776 numbers) are counted and not drawn. A schematic: only the
counts are data.

### 4.2 The ledgers and their audit

`bes/budget.py` computes both ledgers from the real parameter tree, and tests enforce the limits
(Table 1; full ledger in Appendix A, Table A1).

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

Both ledgers count numbers once, at inference: not how often a number is applied, the activations it
produces, its layout, or anything about training. The core's 14.1 million weights are applied 16 times
per token and counted once; a token's page row (3,072 numbers) is counted once, while its gathered copy is
carried through all 8 loops and its cotangent accumulated over them in training.

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

Three kinds of change followed (Table 2; Appendix F, Tables F1–F3 and Figure F2). Five rewrites of the glue,
equal to the reference up to rounding (28 tests) with one deliberate extra rounding (2⁻⁹ relative), took one
`value_and_grad` of a 32 × 1,024 micro-batch from 289.3 to 162.0 ms. Cutting each micro-batch into 4 pieces on
the device equals the original update up to rounding. A one-tile attention kernel that holds a (row, head)
slice in on-chip memory and covers only the causal triangle made one piece 1.212× faster at 1,024 tokens and
1.219× at 2,048 than the library's splash kernel [61]; its equality on the chip is not established (Appendix
F.2, Figure F1).

**Table 2. Training throughput of the run, by cause** (live: median of the logged step rates, with n;
synthetic and benchmark values are whole-step timings; `lead5_work/numbers.json`, `docs/TPU_NOTES.md`,
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
| rows of 2,048 tokens (p = 0.67, step 36,867) | `v9` | 189,067 (226) | – (not timed with the switches) |
| chat-heavy tail, loss on assistant tokens only (p = 0.92, step 48,190, to the end) | `v9` | 189,035 (73) | – |

From the reference to the one-tile kernel the live rate of `final` rose 2.41–2.44× (281,399 over steps
25,300–29,700, against the two monitor readings), including a change of configuration (32 to 16 rows per
micro-batch, +8% by itself). The live rate is about 5% below the synthetic one. At 2,048 tokens `v9` runs 4.0%
below its rate at 1,024.

### 5.3 The cost of the pages: an intervention series

Table 3 and Figure 2 compare the two configurations on the same chip, kernels, table-gradient mode
(`take`, a scatter-add) and pieces, then add three switches to `v9` that change only layout (commit
`769f240`): `flat` (token tables as reshapes of `[V, W]` matrices; exact), `rows` (residuals of the key
normalisation kept, page rows carried lane-dense; exact) and `tokn` (token neurons with tokens in lanes;
equal to rounding).

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
compiler's cost model attributes `v9`'s extra cycles mostly to the token neurons' `[8, 192]` contractions,
which the chip's tiles pad to 16 × 256 (`wf/v9_speed.md`); but `tokn`, the switch aimed at them, was predicted
to save 64 ms and saved 8 (Figure 3b). As the core became faster, the pages did not: they cost about 0.6 s
more per step before the core was rewritten and 0.55 s at the switch.

![Figure 2](figures/F4_v9cost.svg)

**Figure 2.** One accumulation step of 262,144 tokens on a synthetic batch, timed on the chip, by
configuration, table-gradient mode and piece size; ratios against `final`'s fastest step of the same length.
Each bar is one timing (Section 5.1). Black ticks: 262,144 ÷ the live rate, with n: Table 3's windows and, at
2,048 tokens, steps 36,900–48,150 (n = 226, 1,386.5 ms), not timed on a synthetic batch. `final` was timed
with the hand-written cross-entropy backward on (0.177 ms slower per piece; the run does not use it).

### 5.4 Which count predicts the step time?

For six configurations both a compiled program and a chip time exist. Table 4 and Figure 3a ask how well
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
GB, with residuals from −110 to +118 ms: consistent with bytes moved, not parameters, setting the step time
in this implementation, on a test too small to establish it. Differences are harder than totals: for seven
changes the cost model had the sign right every time and the size wrong by factors from 0.36 to 7.9
(Figure 3b; Appendix F.7, Table F5).

![Figure 3](figures/F10_costmodel.svg)

**Figure 3.** The compiler's estimates against the chip. (a) The six configurations of Table 4, sorted by
the measured step (bar). Markers: each static count's leave-one-out prediction, joined to the bar by its
error; FLOP and reads give the same predictions, undefined for `final`. (b) The seven changes of Table F5:
predicted against measured change of the step, absolute values on logarithmic axes; triangles up: slower,
down: faster; after each label, predicted ÷ measured. Six configurations chosen after the measurements: the
figure shows how the counts did here, not that bytes moved predict step time in general.

### 5.5 What the read ledger hides in the design itself

The largest gap between ledger and clock in this model is not the pages but the loops: the read ledger
counts the 16 block applications per token once, the clock charges every one (97% of the reference
implementation's time; Appendix F.6). At equal wall clock, 4 loops would have trained on about 1.6–1.9
times the tokens of 8; at equal tokens, in the micro ablation, 4 loops were 0.12% better than 8 and 12
loops 0.26% worse, within its noise (`docs/E2_ABLATIONS.md`); under the iso-depth law of [21] with two
lanes, halving the loops divides the effective recurrent size by 2^0.65 = 1.57. Whether the tokens would
have made up for it at full scale was neither predicted nor tested: the loop count was set by compute per
number read, and the clock was never asked (arithmetic efficiency: Appendix F.6; negative results: Table H1).

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

`scripts/graft_run.py` (commit `9a4d652`; Figure 4; Appendix G) carries the parameters and optimiser
slots of the 59 shared leaves (34,531,720 numbers), the update count, the step, the trainer's time ledger
and loss guard; the 5 new leaves (63,504,384 numbers) start at their initialisation with fresh slots.
Because the time ledger is carried, the schedule continues: p went from 0.5049 at the stop to 0.5112 at
the restore, and the learning-rate factor continued on 1 − 0.9p (0.5394 at step 29,750). The data stream
cannot be carried: the mixture changed (no library rows), it restarted with another seed, and the
loader's plan needed two corrections (Appendix G). After the graft `v9` trained 22,090 steps and
5,790,760,960 tokens (189,067 tokens/s at 2,048 tokens, n = 226); the run ended at step 51,812 with
13,582,204,928 tokens, about 2.33 × 10⁹ (14.6%) fewer than `final` would have trained without the switch by
a projection in which `final`'s own 2,048-token rate is a scaled estimate.

![Figure 4](figures/F6_graft.svg)

**Figure 4.** Stored numbers of the two configurations at step 29,722, to scale; colours as in Figure 1.
Gray: the 59 shared leaves, carried with their optimiser slots; blue: `final`'s library and index; orange:
`v9`'s five new leaves, with fresh slots. Right: what else was carried, what is new, and the checks of
Section 6.4 (`data/graft_facts.csv`, `data/ledger_leaves.csv`).

### 6.4 Verification

The tool's `--check` compares the source's loss with the grafted tree's on a synthetic batch and refuses
unless they agree to 1e−6 relative; at four source checkpoints, among them the switch's own 29,722 (re-run
under the live configuration hash, since the live invocation ran without the check), the losses agree to
every printed digit (Table G1). On 2 web and 2 chat held-out rows with the checkpoint of step 29,722, the two
programs give the same loss to the last bit, bit-identical gradients for all 59 shared leaves, exactly zero
gradients for K, P_in and P_out, and non-zero gradients for U and G (`lead2_work/graft_grad.py`). Both checks
ran on the CPU in float32; on the chip, in bfloat16, equality holds to rounding and was not measured.

### 6.5 Held-out loss across the graft and to the end

The runtime evaluated 3 checkpoints of `final` and 22 of `v9`, the last at step 51,812, on 64 web rows (a
ClimbMix file beyond the run's download cap), 64 chat rows (the smol-smoltalk test split) and 12 rows of user
turns of the evaluation prompts, test prompts among them, which are left out of comparisons (1,024 tokens
each, float32; Figure 5, Table G2). The comparison set is the 128 web and chat rows. Five checkpoints were
evaluated again on the CPU, also with the new paths switched off and, at 51,812, after preference
optimisation (Appendix G.5).

**Table 5. Held-out loss at the end of training** (step 51,812; web + chat rows; intervals by bootstrap over
rows; `final`'s line by `lead5_work/scripts/heldout5.py`, with `final`'s tokens at equal time at the rates of
Section 6.3; earlier steps in Table G3).

| comparison (web + chat) | difference (nats) | 95% interval |
|---|---|---|
| `v9` 51,812 − `final` 29,722 (same rows, 509.5 more training minutes) | −0.1829 | [−0.2002, −0.1663] |
| … web rows / chat rows only | −0.1196 / −0.2514 | [−0.1286, −0.1112] / [−0.2759, −0.2274] |
| `v9` − `final`'s line at equal training time (line in minutes / in log tokens) | −0.076 / −0.117 | ±0.227 / ±0.129 (prediction, 1 degree of freedom); both inside (log tokens: outside from 41,850 to 48,342) |
| `v9` − `final`'s line at equal tokens (not the budget's axis) | −0.131 | ±0.103; outside (from step 34,166 on) |
| pages off − on | +0.2631 | [0.2442, 0.2829] |
| `v9` with pages off − `final` 29,722 | +0.0802 | [0.0674, 0.0938] |
| after preference optimisation − trained weights, at 51,812 | +0.0124 | [0.0112, 0.0136] |

Four readings, in order of how much they assume. (i) On the same rows the loss fell by 0.061 nats in the
129 training minutes after the graft (Table G3) and by 0.183 by the end of training (web 0.120, chat 0.251; before the
graft, 0.009 in 44 minutes), only 0.015 [0.009, 0.021] of it after step 46,178. From p = 0.67 the chat rows
fell much faster than the web rows (Table G2), while the chat share rose toward 0.70, the rows were longer,
the learning rate decayed and some chat sources were read again: causes this run cannot separate.
(ii) The budget's question, equal training time against a `final` that was not continued, has one
estimate: `final`'s trend through three checkpoints spanning 44 minutes, extended (Figure 5). In training
minutes `v9` lies inside its 95% prediction interval at every checkpoint; in log tokens it lay outside
from step 41,850 to 48,342 and inside again from 49,424, as the
interval widened and the comparison set stopped falling; at equal tokens, not the budget's axis, outside
from 34,166 on (Table 5). The line is the weak part: one residual degree of freedom, 44 minutes of data
extended over 510 more, and no model of the decaying learning rate, the chat ramp, the tail or the change of
data stream. The gap at equal time is not resolved.
(iii) Switched off at evaluation, the pages cost 0.263 nats at 51,812, more than the whole improvement since
the graft (the network is then 0.080 worse than at the graft point), most on rare input tokens (0.356 on the
rarest quarter of positions, 0.193 on the most frequent; Appendix G.5): this decomposes one trained model
and is not a counterfactual.
(iv) What rose at the end. The all-row loss reached its minimum, 2.0840, at step 47,260 and rose to 2.1422
at 51,812 because the proxy rows, almost all user text (10,594 of 11,094 weighted tokens), rose from 2.4683
to 3.2580 while web and chat fell. From p = 0.92 (step 48,190) user and system tokens carry no loss weight
(Appendix C.1), and the loss on user text rose at once: 2.6651 on the proxy rows at the first checkpoint
after, 1.8841 to 2.1144 on the chat rows' user and system positions (46,178 to 51,812), while their assistant
tokens fell from 1.2651 to 1.1983. The model stopped being trained to predict users, as the schedule chose;
its fit to the assistant and web text it is trained and judged on did not get worse.

![Figure 5](figures/F8_heldout.svg)

**Figure 5.** Held-out loss on the comparison set of Table 5 against training minutes, to step 51,588 (the
figures stop at the last step record, 51,800; the final checkpoint 51,812, at the same 2.0438, is in Table 5).
Filled points: the runtime's evaluations; rings: CPU re-evaluations (29,722, 34,166, 35,294, 46,178); open
diamonds: `v9` with its pages switched off (34,166, 35,294, 46,178). Gray line and band: the least-squares
line through `final`'s three points, extended, with its 95% prediction interval (one residual degree of
freedom); arrow: `v9`'s distance from it at 51,588, −0.077 ± 0.225. Every `v9` point lies inside the band; in
log tokens or at equal tokens later checkpoints lie outside (Table 5). The figure does not show that `v9` is
ahead of a continued `final` (reading ii).

### 6.6 The graft under control: a replication at the micro width

One run cannot say how a graft compares with training the new paths from the first step, or with not
adding them, so we repeated the graft with the repository's own model and micro bench, everything else
held equal (`docs/MICRO_GRAFT.md`; code at `b04478e`, reviewed adversarially before the runs; Appendix G.4).
At width 128 and a budget of 6 × 10⁶ tokens, five arms ran with seeds 0, 1 and 2: `ref`, the micro
reference; `v9_pages`, the reference with the `v9` page from the first step; and `graft_25`, `graft_50`,
`graft_75`, `ref` up to 0.25, 0.5 or 0.75 of the budget, then grafted into `v9_pages` by the live tool's own
function. Within a seed all arms share data order, schedule and initial values, and each of the nine grafts
left the held-out loss exactly unchanged (CPU, float32).

**Table 6. The graft at the micro width** (held-out loss, mix of web and chat, nats per token, at
5,996,544 tokens, the restart at its own end, 2,998,272; mean ± sd over seeds 0, 1, 2; differences paired
by seed; share recovered = (`ref` − arm) / (`ref` − `v9_pages`), the ratio of the mean differences, per seed
in brackets; `lead3_work/micrograft.py` on `micro_graft/6M/results_all.jsonl` and
`micro_graft/3M/results.tar.gz`, checked by `lead4_work/scripts/micro4.py`).

| arm | new paths from | loss | arm − `ref` | arm − `v9_pages` | share recovered |
|---|---|---|---|---|---|
| `ref` | never | 4.1912 ± 0.0076 | – | +0.0519 ± 0.0107 | 0 |
| `v9_pages` | step 0 | 4.1394 ± 0.0083 | −0.0519 ± 0.0107 (−1.24%) | – | 1 |
| `graft_25` | 0.25 of the budget | 4.1363 ± 0.0075 | −0.0550 ± 0.0010 | −0.0031 ± 0.0113 | 1.06 (0.86, 1.33, 1.08) |
| `graft_50` | 0.5 | 4.1555 ± 0.0091 | −0.0358 ± 0.0030 | +0.0161 ± 0.0096 | 0.69 (0.62, 0.87, 0.64) |
| `graft_75` | 0.75 | 4.1789 ± 0.0074 | −0.0123 ± 0.0004 | +0.0396 ± 0.0103 | 0.24 (0.20, 0.29, 0.24) |
| restart: `v9_pages` from scratch on half the budget (3 × 10⁶ tokens, own schedule) | step 0 of the half | 4.3897 ± 0.0017 | +0.1985 ± 0.0062 | +0.2504 ± 0.0074 (against `graft_50`: +0.2342 ± 0.0075) | – |

**Reading** (Table 6, Figure 6). The page from the first step ended 0.052 nats (1.24%) below `ref`. Every
graft ended below `ref` in every seed. Grafted at a quarter of the budget, the page ended with no difference
from the page from the first step resolved (−0.0031 ± 0.0113, sd of three seeds; a 95% t interval of the
mean, ±0.028, is about half the page's whole gain); grafted at half, it recovered 0.69 of that gain
(0.62–0.87 by seed); at three quarters, 0.24, consistent with a late graft losing the time it does not have,
with the learning rate also falling (Appendix G.4). Restarting instead with the page from scratch on the
second half ended 0.234 nats above `graft_50` (+0.227 to +0.241 per seed) and 0.199 above `ref`: the tokens a
restart discards were worth four to five times the page's whole gain (0.199–0.250 against 0.052). That is
what a restart costs while the loss still falls steeply, at equal tokens, not what it would have cost the
live run at 7.79 × 10⁹ tokens and equal wall-clock time.

![Figure 6](figures/F12_micrograft.svg)

**Figure 6.** The graft at the micro width (Table 6; width 128, 6 × 10⁶ tokens, three seeds). (a) Each arm's
held-out loss (0.75 × web + 0.25 × chat) minus `ref`'s with the same seed: large dots, the mean of the seeds;
small dots, each seed; a graft arm leaves the zero line at its diamond. Squares, above a break in the axis:
the restart at its own end (3 × 10⁶ tokens) minus `ref` at 6 × 10⁶, +0.1985, which pairs unequal token
counts. (b) Share of the page gain recovered at the end: large dots, the ratio of the mean differences (1.06,
0.69, 0.24); small dots, each seed. No interval is drawn: three seeds are too few.

### 6.7 What one run cannot show, and what the controlled experiment shows

At width 128 the controlled experiment shows that a graft is function-preserving in practice (on the CPU in
float32; on the chip in bfloat16 this was not measured), that continuing in the grafted configuration never
ended worse than not grafting (9 of 9 runs), that the share of the from-scratch gain a graft recovers falls
with the graft point, and that a restart with the new paths on the remaining half ended far behind the
graft. Its limits: width 128 against 768; 6 × 10⁶ tokens against 1.36 × 10¹⁰, about one pass over small
pools; three seeds; a schedule in tokens, not wall-clock time, so the page's cost in step time (1.43–1.58×
live on the chip) does not enter; a data stream unchanged across the graft; a page worth 1.2% of the loss.
It is evidence about the mechanism of a graft, not about the size of its effect in the live run, and it
leaves open whether the pages help at width 768, whether the graft beat a restart or a continuation at
equal wall-clock time, what the changed data stream did, how the new leaves behaved training only in the
low-rate half of the schedule, and run-to-run variation. Any comparison of the shipped model with the
opponents measures all of these together.

## 7. Evaluation: protocol, deviations and results

### 7.1 Protocol

The protocol (`docs/EVAL_PROTOCOL.md`) is in the branch's root commit (`5ff33c5`, 23 November 00:35 UTC)
and unchanged since: it predates every TPU run and every reply of the full-scale model, not the
micro-scale runs of gates E2, E4 and E5. **Prompts**: 400 original English prompts in five categories, a
quarter with a second user turn, split by the SHA-256 of their identifier into `dev` and `test` (200 each;
`test`: 252 turns, 30 maths and 48 constrained prompts); decisions use `dev`, and `test` is used once. Sides
are seeded and balanced, and the scorer refuses any rating whose commitment to the hidden key does not
match. **Readers**: the registered judge `qwen3-4b-2507` in both orders, with `gemma4-e4b` as a second judge;
a side wins only if both orders choose it. **Exact scores** come first and are trusted over a reader.
**Statistics**: the score (wins + ties/2)/n with a Wilson interval, an exact two-sided sign test, slices by
category, turns and length ratio. **What a claim must disclose** (§3): results with K = 1 as well as with
our K particles, with the library on and off, and two control arms of each opponent with the same K, kept by
the opponent's own mean log-probability (K-matched) or by our selector (selector-matched); MobileLLM-R1.5 is
credited with the better of its two arms on `dev`, decided before `test` is looked at.

### 7.2 Deviations from the protocol

1. **The readers.** The registered judge read the development pairs in both orders and, when this was
   written, 155 and 92 of the 200 pairs of the two test sets (files last written at 21:34 UTC) [TBD: its full
   pass, running on CPU runtimes since 20:41 UTC at about 40 s per pair on the Falcon-H1-Tiny set and 178 s
   on the MobileLLM-R1.5 set, whose replies are long]; the second registered judge,
   which decided 3 of 60 development pairs per set (Appendix E.6), was not run on the test pairs. Opus 5.5
   agents, of the family of the agents that built the system, read every test pair twice on exchanged sides
   from written instructions (Appendix J): an addition, never averaged with the registered judge.
2. **Eleven test prompts were seen** by mid-run CPU samples (chat-002, chat-017, math-034, instruct-034,
   instruct-062, writing-100, code-026, math-001, code-002, writing-001, instruct-001). Every test result is
   also reported on the 189 untouched prompts.
3. **The proxy held-out rows contain test-split user turns** (Section 6.5); they are left out of
   comparisons.
4. **MobileLLM's two arms.** No blind comparison of the `neutral` arm was made on `dev`, so our model is
   compared with the `recommended` arm by default, not by the protocol's rule; its exact scores favoured
   `recommended` on `dev` (Table E3), and on `test` the two are close (Table 8).
5. **At training time** (Appendix B.5): an E2 loss reading chosen after seeing the data; registers kept
   despite failing their gate; the E5 copy criterion judged wrong after the fact; the pre-registered
   configuration comparison not run; the configuration changed mid-run.
6. **K = 1 and the matched controls, late and incomplete.** Our replies are chosen among K = 6 particles
   and the opponents draw one sample. Made after training, on `test`, and read blind (Tables 7, 8): our K = 1
   arm (the official export with one particle) and Falcon-H1-Tiny's two control arms. MobileLLM-R1.5's two
   arms do not exist yet [TBD: in generation on a CPU runtime since 19:17 UTC, 40 of 200 prompts by 21:13].
   Library on and off does not apply: `v9` ships no library.

### 7.3 What 200 comparisons can resolve

With the Opus readers' tie rates on `test` (28.5% against Falcon-H1-Tiny, 34.5% against MobileLLM-R1.5,
53.5% between our two sets of weights), n = 200 detects with 80% power (two-sided 5% sign test) a decisive
win rate of 0.5 ± 0.116, ± 0.121 and ± 0.143 (± 0.119, 0.125 and 0.147 on the 189 untouched prompts;
`bes.eval.proxy.score.n_required`, `lead5_work/numbers.json`). At the registered judge's tie rates on
its partial pass, 66% and 73%, the bounds would be ± 0.166 and ± 0.186. The observed decisive rates against the two
opponents, 0.245 and 0.885, lie far beyond these bounds.

### 7.4 The official model

The official model is what the run's own rules shipped (Appendix C.4; Tables C4, C5, M7, M8): the weights of
step 51,812 after preference optimisation, with the default decoding. Preference optimisation (DPO [74],
β = 0.1, one epoch of 551 steps of 64 pairs, almost all from UltraFeedback [76]) tried two learning rates;
both passed every check (held-out preference accuracy 0.688 and 0.694 against 0.500; held-out
assistant-token loss +0.86% and +1.10%, limit 3%), and the rule kept the higher accuracy, 3 × 10⁻⁶. Decoding
selection keeps the default unless a variant's exact score on the 200 development prompts is higher on
significantly more prompts than lower: one particle was worse on 18 and better on 1 (p = 7.6 × 10⁻⁵), no
length band better on 2 and worse on none (p = 0.50), so the default stayed (6 particles, temperature 0.6,
top-p 0.9, 768 tokens, no library). The export (98,036,104 numbers in float16) passed its 16-bit check; our
test replies come from an export with the same `weights16.npz` (SHA-256 `2da7cbd2…`) and decoding choice,
released as `release/main16-v9` (Appendix J).

### 7.5 Blind comparisons on the test split

Table 7 gives every reading of the test pairs; Figure 7 draws the Opus readers' with the development
studies. The three kinds of reader are kept apart: the exact scorer (Section 7.6), the registered judge (two
thirds or more of its pairs undecided) and the Opus readers (28.5% and 34.5% undecided), whose resolutions
differ too much to be averaged.

**Table 7. Blind comparisons on the 200 test prompts** (W / L / T from the first-named model's side, T
including "both bad" and pairs whose two orders disagree; score (wins + ties/2)/n and decisive win rate, with
95% Wilson intervals; exact two-sided sign test; orders: share of pairs whose two orders agree; first: mean
preference for the first-shown reply; last column: the untouched prompts among those rated; recomputed with
the key check by `lead5_work/scripts/final_results.py`, which reproduces every report; the registered judge's
pass is partial).

| comparison | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders; first | 189 prompts: W / L / T; score |
|---|---|---|---|---|---|---|---|---|
| official vs Falcon-H1-Tiny | Opus, two readings | 200 | 35 / 108 / 57 | 0.318 [0.257, 0.385] | 0.245 [0.182, 0.321] | < 0.0001 | 90.5%; +0.005 | 32 / 101 / 56; 0.317 |
|  | `qwen3-4b-2507` (registered) | 155 | 10 / 43 / 102 | 0.394 [0.320, 0.472] | 0.189 [0.106, 0.314] | < 0.0001 | 60.7%; −0.139 | 10 / 41 / 96; 0.395 |
| official vs Falcon-H1-Tiny, K = 6, own log-probability | Opus | 200 | 35 / 112 / 53 | 0.308 [0.248, 0.375] | 0.238 [0.176, 0.313] | < 0.0001 | 93.5%; +0.020 | 34 / 106 / 49; 0.310 |
| official vs Falcon-H1-Tiny, K = 6, our selector | Opus | 200 | 29 / 116 / 55 | 0.283 [0.225, 0.349] | 0.200 [0.143, 0.272] | < 0.0001 | 92.5%; +0.018 | 27 / 110 / 52; 0.280 |
| ours with one particle (K = 1) vs Falcon-H1-Tiny | Opus | 200 | 22 / 122 / 56 | 0.250 [0.195, 0.314] | 0.153 [0.103, 0.220] | < 0.0001 | 92.5%; +0.015 | 20 / 116 / 53; 0.246 |
| official vs MobileLLM-R1.5 | Opus, two readings | 200 | 116 / 15 / 69 | 0.753 [0.688, 0.807] | 0.885 [0.820, 0.929] | < 0.0001 | 96.0%; −0.005 | 109 / 14 / 66; 0.751 |
|  | `qwen3-4b-2507` (registered) | 92 | 22 / 3 / 67 | 0.603 [0.501, 0.697] | 0.880 [0.700, 0.958] | 0.0002 | 72.8%; +0.011 | 20 / 3 / 64; 0.598 |
| ours with one particle (K = 1) vs MobileLLM-R1.5 | Opus | 200 | 99 / 16 / 85 | 0.708 [0.641, 0.766] | 0.861 [0.786, 0.913] | < 0.0001 | 91.5%; +0.008 | 92 / 14 / 83; 0.706 |
| official vs trained weights (before preference optimisation) | Opus | 200 | 59 / 34 / 107 | 0.563 [0.493, 0.629] | 0.634 [0.533, 0.725] | 0.012 | 91.5%; +0.010 | 55 / 33 / 101; 0.558 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus | 200 | 144 / 6 / 50 | 0.845 [0.788, 0.889] | 0.960 [0.915, 0.982] | < 0.0001 | 93.5%; +0.013 | 137 / 5 / 47; 0.849 |

**Against Falcon-H1-Tiny** the official model loses in every category (code 10 / 13 with p = 0.68, the
others p ≤ 0.025; Table M2) and on two-turn prompts (7 / 30); its replies fared best where they were the
shorter of the pair (score 0.419) and worst where they were the longer (0.143). Given six samples per reply,
Falcon-H1-Tiny was preferred again: paired by prompt, our score against its released preset differs from that
against the K-matched arm by 0.010 (29 prompts better, 25 worse, p = 0.68) and against the selector-matched
arm by 0.035 (34 / 21, p = 0.10), so the loss is not an artefact of six particles against one sample. Our own
six particles are worth something: against the same opponents the official decode scores higher than one
particle on 31 prompts and lower on 12 against Falcon-H1-Tiny (mean 0.068, p = 0.0054) and 28 / 13 against
MobileLLM-R1.5 (0.045, p = 0.028); with one particle the model loses to Falcon-H1-Tiny 22 to 122 and still
beats MobileLLM-R1.5 99 to 16.
**Against MobileLLM-R1.5** the official model wins in every category but maths (7 / 12, p = 0.36). **The
registered judge**, part-way, points the same way with fewer decided pairs: where it and the Opus readers both
chose a side they agreed in 40 of 44 pairs against Falcon-H1-Tiny and 22 of 22 against MobileLLM-R1.5 (Cohen's
κ 0.22 and 0.33). The restriction to the 189 untouched prompts moves every score by at most 0.006 (Tables M1, M3, M11).

### 7.6 Exact scores and degenerate replies

**Table 8. Exact scores on the 200 test prompts (252 turns)** (`bes.eval.exact_score` on every reply,
recomputed and equal to the run's exact reports where one exists; in brackets, the 189 untouched prompts, 28
maths and 44 constrained; the trained weights and every kind of degenerate reply in Table M4).

| measure | ours, official (K = 6) | ours, K = 1 | Falcon-H1-Tiny | Falcon, K = 6, own log-probability | Falcon, K = 6, our selector | MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |
|---|---|---|---|---|---|---|---|
| maths answers correct, of 30 (of 28) | 6 (6) | 5 (5) | 15 (14) | 17 (16) | 18 (17) | 12 (11) | 11 (10) |
| prompts with every constraint met, of 48 (of 44) | 38 (34) | 37 (33) | 33 (29) | 35 (32) | 42 (38) | 8 (8) | 9 (8) |
| degenerate replies, of 252 (of them cut at the token limit) | 5 (5) | 17 (17) | 2 (1) | 1 (0) | 0 | 158 (150) | 148 (146) |
| words per reply, median | 101 | 105 | 155 | 147 | 145 | 395 | 517 |

Against the released Falcon-H1-Tiny our model meets every constraint on more prompts (38 against 33 of 48)
and answers far fewer maths prompts (6 against 15 of 30). **The selector-matched arm shows what that
constraint score is**: given six samples and our selection rule (constraints compiled from the prompt,
failure filter, length band, maths majority vote, consensus; not our exact potentials, which act during
sampling), Falcon-H1-Tiny meets every constraint on 42 prompts, four more than our model; with one particle,
where nothing is selected but the potentials act, our model meets 37. The lead over the released preset is
our decoder's, not our model's, and the maths gap does not close (5 and 6 against 15 to 18). MobileLLM-R1.5's
replies are degenerate in 158 of 252 turns under its recommended preset (150 cut at the token limit) and in
148 under the neutral one; its blind defeat is largely a defeat of those replies. No model gave a stock
refusal or deflection.

### 7.7 Before and after preference optimisation

Read the same way on the same prompts (4 of the 200 got identical replies), the official weights were
preferred to the trained weights in 59 pairs and the trained ones in 34 (p = 0.012; 189 prompts: 55 / 33,
p = 0.025), mostly in code (10 / 1) and chat (16 / 8), not in maths (5 / 5). The exact scores did not follow:
maths 6 and 6 of 30, constraints 38 against 41 of 48, degenerate replies 5 against 3; the replies grew longer
(mean 130 against 119 words), and the held-out loss of the comparison set rose by 0.012 nats (Section 6.5).
One epoch of DPO (424 s of training at the chosen rate) bought a modest blind preference, measured by one
kind of reader only (Table M5).

### 7.8 In absolute terms

Both replies were judged bad in 89 of 200 pairs in each reading when both were ours (official against
trained weights, 44.5%), against 44 of 60 development pairs (73%, one reading) for two decodings of the
checkpoint of step 31,912; against Falcon-H1-Tiny in 44 and 41 pairs, against MobileLLM-R1.5 in 66 and 64,
and in the reference set, where no reply is ours, in 38 and 38 (raw verdicts, `lead5_work/final_results.json`).
A little under half of the official model's replies were judged useless together with a sibling reply; the
prompt sets differ, so the fall from 73% is indicative only.

### 7.9 The trajectory

**Table 9. From mid-run checkpoints to the official model** (Opus readers, W / L / T and score; registered
judge, W / L; exact scores of the same replies; on `dev`, the first 60 prompts, 9 maths and 14 constrained,
Appendix E.6; the official model's `dev` replies are the run's own decoding-selection replies).

| model (split, n) | vs Falcon-H1-Tiny: Opus | vs MobileLLM-R1.5: Opus | registered judge, W / L (Falcon; MobileLLM) | maths correct | every constraint met |
|---|---|---|---|---|---|
| `final` 25,280, p = 0.44 (`dev`, 60; one reading) | 2 / 43 / 15, 0.158 | 15 / 8 / 37, 0.558 | 2 / 16; 7 / 4 | 0 of 9 | 12 of 14 |
| `v9` 31,912, p = 0.56 (`dev`, 60; two readings) | 3 / 39 / 18, 0.200 | 20 / 7 / 33, 0.608 | 1 / 16; 7 / 2 | 1 of 9 | 12 of 14 |
| official, 51,812 after preference optimisation (`dev`, 60) | – | – | – | 0 of 9 | 13 of 14 |
| official (`test`, 200; two readings) | 35 / 108 / 57, 0.318 | 116 / 15 / 69, 0.753 | 10 / 43; 22 / 3 (partial) | 6 of 30 | 38 of 48 |

The development and test prompts are two halves of one set, and the studies differ in checkpoint,
configuration and number of readings, so the rows are comparable in kind only. Against Falcon-H1-Tiny the
decisive win rate rose from 0.044 and 0.071 (`dev`) to 0.245 (`test`); against MobileLLM-R1.5 from 0.652 and
0.741 to 0.885. The opponents' own reference (Falcon-H1-Tiny over MobileLLM-R1.5 in 144 of 150 decisive test
pairs) is the scale: the comparison that informs is the one our model loses.

![Figure 7](figures/F11_blind.svg)

**Figure 7.** Blind pairwise comparisons by the Opus readers (Tables 7, 9): `final` at step 25,280 on the
first 60 development prompts, read once; `v9` at step 31,912 on the same prompts and the official model
(labelled `v9`, step 51,812) on the 200 test prompts, read twice. Bars: wins, ties or "both bad", losses;
dots and whiskers: the score with its 95% Wilson interval, printed as the plotting software rounds it (0.7525
as 0.752; Table 7 rounds exact halves up). Only this added reader is drawn, because it decides most pairs;
the registered judge points the same way where it decides (Tables 7, E2).

### 7.10 What limits each statement

**Preferred over MobileLLM-R1.5** (116 to 15): by readers of the builders' family who are an addition to the
registered protocol, with the registered judge part-way (22 to 3 on 92 pairs); over an opponent whose replies
are degenerate in 63% of turns under its recommended preset in our harness (a converted community file with
budget forcing), so that the lead measures that opponent's state here as much as our model; on prompts of
which 11 were seen (the 189 agree); with six particles against one sample (with one, 99 to 16; MobileLLM-R1.5's
control arms are not finished); for one run. **Not preferred over Falcon-H1-Tiny** (35 to
108): by every reader, in every category, on the 189 prompts, against its K = 6 arms and with our K = 1
decode (22 to 122); its default sampling may understate it. No human has read the pairs.

## 8. Limitations and threats to validity

1. **One run, one seed at full scale**, with no full-scale baseline for the design change (no `v9` from
   scratch, no `final` continued, no restart); the controlled experiment supplies these at width 128 and
   6 × 10⁶ tokens only, at equal tokens (§6.6). The test result is one model's.
2. **Mid-run changes confound every curve**: the stream changed at steps 6,647, 25,280 and 29,722, its chat
   share ramps up from step 36,867 and conversations train on assistant tokens only from step 48,190; the
   learning-rate factor jumped at step 12,062; the compiled graph changed at steps 9,332, 22,613 and 31,912.
3. **One implementation, one chip, one software stack** (jax 0.11.2, libtpu 0.0.50) for the 1.46–1.62× and
   the predictor test, which has six points chosen after the fact (§5).
4. **Equality on the chip is to rounding and partly unverified** for the adopted kernel and the graft's
   gradients (§5.2, §6.4).
5. **The readers** (§7.2, §7.10): the Opus readers are an addition and of the builders' family; the
   registered judge's test pass is partial and decides few pairs; the second registered judge was not run on
   the test pairs; no human has read the pairs [78–81]. Two readings on exchanged sides remove pure position
   preference, not a bias both share.
6. **The prompts are ours**, eleven test prompts were seen, and the test's resolution is limited (§7.3); our
   decoder uses six particles and exact potentials against the opponents' single samples, and MobileLLM-R1.5's
   control arms are not finished; the reply budgets differ (768 of our tokens;
   1,024 for Falcon-H1-Tiny; 1,024 answer tokens after up to 3,072 of reasoning for MobileLLM-R1.5).
7. **Opponent settings**: a converted community file with budget forcing for MobileLLM-R1.5, 63% of whose
   test replies are degenerate in our harness; llama.cpp's default sampling for Falcon-H1-Tiny. Either may
   understate its model.
8. **Small laboratories** produced the evidence for the pages and the components (widths 48 and 128,
   1–12 million tokens; Appendix B) and the controlled graft.
9. **The data rules' effects are not measured**, from step 6,647 the model is not trained to refuse
   anything, and it was not evaluated for harmful output (Appendices D, K).
10. **The token gap** to the opponents (59 and 309 times fewer tokens) cannot be separated from the design
    in any comparison here.

## 9. Conclusion

A budget in parameters is a statement about a function: how many numbers it stores and how many it reads
to produce a token. The cost of training that function on one accelerator for a day is a statement about
bytes, shapes and repetitions. In this run the two came apart three ways: two configurations that the ledger
calls nearly equal differed by 1.62× in step time, a quarter of it removed by layout alone; the compiler's
count of bytes moved predicted step times where FLOP and reads could not, on six configurations and after
the fact; and the loop count, 97% of the step time in the reference code, is invisible to the read ledger. A
function-preserving graft let the run change its design at half time without discarding 7.79 × 10⁹ trained
tokens; under control at width 128, grafting never ended worse than not grafting, which supports the
mechanism and says nothing about the size of the effect at full scale. The run met its ledger (98.0 of 100
million stored, 19.6 of 20 million read per token) and its clock (86,054 of 86,400 TPU seconds) and trained
1.358 × 10¹⁰ tokens. On the 200 test prompts, read by Opus 5.5 agents added to the registered protocol and,
part-way, by the registered judge, the official model is preferred to MobileLLM-R1.5 (116 of 131 decisive
pairs), whose replies are mostly degenerate in our harness, and not to Falcon-H1-Tiny (35 of 143), which keeps
its lead when it gets six samples and our selector; our model's edge in meeting constraints belongs to its
decoder. The informative comparison is the one the model loses, to a model trained on 59 times more tokens.

## References

Numbered as in `refs.md`, where each entry records what was read to verify the claim it supports;
[82]–[96] are that file's additional verified entries A1, A2, A4, A16, A19–A25 and A27–A30, in that order.

[1] J. Geiping, T. Goldstein. Cramming: Training a Language Model on a Single GPU in One Day. arXiv:2212.14034, 2022.
[2] P. Izsak, M. Berchansky, O. Levy. How to Train BERT with an Academic Budget. arXiv:2104.07705, 2021.
[3] M. Li, E. Yumer, D. Ramanan. Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints. ICLR 2020. arXiv:1905.04753.
[4] K. Jordan and contributors. modded-nanogpt: the NanoGPT speedrun. GitHub repository, 2024–2026. https://github.com/KellerJordan/modded-nanogpt
[5] A. Karpathy. nanochat: The best ChatGPT that $100 can buy. GitHub repository, 2025. https://github.com/karpathy/nanochat
[6] Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, et al. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. ICML 2024. arXiv:2402.14905.
[7] C. Zhao, E. Chang, Z. Liu, C.-J. Chang, W. Wen, C. Lai, et al. MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes. ICLR 2026. arXiv:2509.24945.
[8] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist, et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. arXiv:2306.13649.
[9] Meta. MobileLLM-R1.5-140M model card. Hugging Face, 2025. https://huggingface.co/facebook/MobileLLM-R1.5-140M
[10] J. Zuo, M. Velikanov, I. Chahed, Y. Belkada, D. E. Rhayem, G. Kunsch, et al. Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance. arXiv:2507.22448, 2025.
[11] Falcon-LLM Team (TII). Falcon-H1-Tiny: A series of extremely small, yet powerful language models redefining capabilities at small scale. Technical blog post, 15 January 2026. https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost
[12] L. Ben Allal, A. Lozhkov, E. Bakouch, G. Martín Blázquez, G. Penedo, L. Tunstall, et al. SmolLM2: When Smol Goes Big. arXiv:2502.02737, 2025.
[13] R. Eldan, Y. Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759, 2023.
[14] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, Ł. Kaiser. Universal Transformers. ICLR 2019. arXiv:1807.03819.
[15] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR 2020. arXiv:1909.11942.
[16] N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi. Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025. arXiv:2502.17416.
[17] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171, 2025.
[18] R.-J. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, H. Que, et al. Scaling Latent Reasoning via Looped Language Models. arXiv:2510.25741, 2025.
[19] H. Prairie, Z. Novack, T. Berg-Kirkpatrick, D. Y. Fu. Parcae: Scaling Laws For Stable Looped Language Models. arXiv:2604.12946, 2026.
[20] D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, et al. Hyper-Connections. arXiv:2409.19606, 2024.
[21] K. Schwethelm, D. Rueckert, G. Kaissis. How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. NeurIPS 2026. arXiv:2604.21106.
[22] Z. Zhou, T. Wu, Z. Jiang, F. Obeid, Z. Lan. Value Residual Learning. arXiv:2410.17897, 2024.
[23] S. Roller, S. Sukhbaatar, A. Szlam, J. Weston. Hash Layers For Large Sparse Models. arXiv:2106.04426, 2021.
[24] G. Lample, A. Sablayrolles, M. Ranzato, L. Denoyer, H. Jégou. Large Memory Layers with Product Keys. NeurIPS 2019. arXiv:1907.05242.
[25] V.-P. Berges, B. Oğuz, D. Haziza, W. Yih, L. Zettlemoyer, G. Ghosh. Memory Layers at Scale. arXiv:2412.09764, 2024.
[26] H. Huang, D. Zhu, B. Wu, Y. Zeng, Y. Wang, Q. Min, et al. Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling. ICML 2025. arXiv:2501.16975.
[27] D. Yu, E. Cohen, B. Ghazi, Y. Huang, P. Kamath, R. Kumar, et al. Scaling Embedding Layers in Language Models. NeurIPS 2025. arXiv:2502.01637.
[28] X. Cheng, R. Tian, W. Zeng, D. Dai, Q. Chen, B. Wang, et al. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372, 2026.
[29] A. Tseng, C. De Sa. L³: Large Lookup Layers. ICML 2026. arXiv:2601.21461.
[30] R. Csordás, P. Piękos, K. Irie, J. Schmidhuber. SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention. NeurIPS 2024. arXiv:2312.07987.
[31] J. Kang. Memory Attention. arXiv:2609.28399, 2026.
[32] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, et al. Mixtral of Experts. arXiv:2401.04088, 2024.
[33] Google. Gemma 3n model overview. Google AI for Developers, 2025–2026. https://ai.google.dev/gemma/docs/gemma-3n
[34] S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer Sentinel Mixture Models. arXiv:1609.07843, 2016.
[35] E. Grave, A. Joulin, N. Usunier. Improving Neural Language Models with a Continuous Cache. arXiv:1612.04426, 2016.
[36] U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, M. Lewis. Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020. arXiv:1911.00172.
[37] J. Liu, S. Min, L. Zettlemoyer, Y. Choi, H. Hajishirzi. Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. COLM 2024. arXiv:2401.17377.
[38] Y. W. Teh. A Hierarchical Bayesian Language Model Based On Pitman-Yor Processes. COLING/ACL 2006, 985–992.
[39] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, et al. In-context Learning and Induction Heads. arXiv:2209.11895, 2022.
[40] E. Akyürek, B. Wang, Y. Kim, J. Andreas. In-Context Language Learning: Architectures and Algorithms. arXiv:2401.12973, 2024.
[41] J. Xu, X. Liu, J. Yan, D. Cai, H. Li, J. Li. Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation. NeurIPS 2022. arXiv:2206.02369.
[42] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453.
[43] OpenAI. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925, 2025.
[44] S. Min, J. Boyd-Graber, C. Alberti, D. Chen, E. Choi, M. Collins, et al. NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned. PMLR, 2021. arXiv:2101.00133.
[45] T. Chen, I. Goodfellow, J. Shlens. Net2Net: Accelerating Learning via Knowledge Transfer. arXiv:1511.05641, 2015.
[46] T. Wei, C. Wang, Y. Rui, C. W. Chen. Network Morphism. arXiv:1603.01670, 2016.
[47] S. Shen, P. Walsh, K. Keutzer, J. Dodge, M. Peters, I. Beltagy. Staged Training for Transformer Language Models. arXiv:2203.06211, 2022.
[48] C. Wu, Y. Gan, Y. Ge, Z. Lu, J. Wang, Y. Feng, et al. LLaMA Pro: Progressive LLaMA with Block Expansion. ACL 2024. arXiv:2401.02415.
[49] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. Riquelme Ruiz, B. Mustafa, J. Ainslie, et al. Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints. arXiv:2212.05055, 2022.
[50] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021.
[51] K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein. Muon: An optimizer for hidden layers in neural networks. Blog post, 2024. https://kellerjordan.github.io/posts/muon/
[52] J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, et al. Muon is Scalable for LLM Training. arXiv:2502.16982, 2025.
[53] N. Amsel, D. Persson, C. Musco, R. M. Gower. The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm. arXiv:2505.16932, 2025.
[54] Z. Li, L. Liu, C. Liang, W. Chen, T. Zhao. NorMuon: Making Muon more efficient and scalable. arXiv:2510.05491, 2025.
[55] L. Chen, J. Li, K. Liang, B. Su, C. Xie, N. W. Pierse, et al. Cautious Weight Decay. arXiv:2510.12402, 2025.
[56] K. Wen, X. Dang, K. Lyu, T. Ma, P. Liang. Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization. arXiv:2606.16899, 2026.
[57] L. Ren, Y. Liu, Y. Shen, W. Chen. Rethinking Language Model Scaling under Transferable Hypersphere Optimization. arXiv:2603.28743, 2026.
[58] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135, 2022.
[59] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691, 2023.
[60] JAX authors. Pallas: a JAX kernel language. JAX documentation, 2026. https://docs.jax.dev/en/latest/pallas/index.html
[61] JAX authors. Splash attention (sparse flash attention) for TPU, `jax/experimental/pallas/ops/tpu/splash_attention`. JAX 0.11.1 source code.
[62] Google (openxla). Tokamax: a Pallas custom kernel library, version 0.0.18. https://github.com/openxla/tokamax
[63] A. Ivanov, N. Dryden, T. Ben-Nun, S. Li, T. Hoefler. Data Movement Is All You Need: A Case Study on Optimizing Transformers. MLSys 2021. arXiv:2007.00072.
[64] A. K. Lew, T. Zhi-Xuan, G. Grand, V. K. Mansinghka. Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs. arXiv:2306.03081, 2023.
[65] S. Zhao, R. Brekelmans, A. Makhzani, R. Grosse. Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo. arXiv:2404.17546, 2024.
[66] J. Loula, B. LeBrun, L. Du, B. Lipkin, C. Pasti, G. Grand, et al. Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo. arXiv:2504.13139, 2025.
[67] I. Puri, S. Sudalairaj, G. Xu, K. Xu, A. Srivastava. Rollout Roulette: A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods. arXiv:2502.01618, 2025.
[68] K. Park, J. Wang, T. Berg-Kirkpatrick, N. Polikarpova, L. D'Antoni. Grammar-Aligned Decoding. NeurIPS 2024. arXiv:2405.21047.
[69] A. Beirami, A. Agarwal, J. Berant, A. D'Amour, J. Eisenstein, C. Nagpal, et al. Theoretical guarantees on the best-of-n alignment policy. ICML 2025. arXiv:2401.01879.
[70] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. arXiv:2203.11171.
[71] J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, et al. Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911, 2023.
[72] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.
[73] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, et al. s1: Simple test-time scaling. arXiv:2501.19393, 2025.
[74] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290, 2023.
[75] L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, et al. Zephyr: Direct Distillation of LM Alignment. arXiv:2310.16944, 2023.
[76] G. Cui, L. Yuan, N. Ding, G. Yao, B. He, W. Zhu, et al. UltraFeedback: Boosting Language Models with Scaled AI Feedback. ICML 2024. arXiv:2310.01377.
[77] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, et al. Tulu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
[78] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.05685.
[79] P. Wang, L. Li, L. Chen, Z. Cai, D. Zhu, B. Lin, et al. Large Language Models are not Fair Evaluators. arXiv:2305.17926, 2023.
[80] Y. Dubois, B. Galambosi, P. Liang, T. B. Hashimoto. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. COLM 2024. arXiv:2404.04475.
[81] A. Panickssery, S. R. Bowman, S. Feng. LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076, 2024.
[82] S. Diao, Y. Yang, Y. Fu, X. Dong, D. Su, M. Kliegl, et al. Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training. NeurIPS 2025. arXiv:2504.13161.
[83] G. Penedo, H. Kydlíček, L. Ben Allal, A. Lozhkov, M. Mitchell, C. Raffel, et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv:2406.17557, 2024.
[84] D. R. So, W. Mańke, H. Liu, Z. Dai, N. Shazeer, Q. V. Le. Primer: Searching for Efficient Transformers for Language Modeling. arXiv:2109.08668, 2021.
[85] A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi. The Curious Case of Neural Text Degeneration. ICLR 2020. arXiv:1904.09751.
[86] TII. Falcon-H1-Tiny-90M-Instruct model card. Hugging Face, 2026. https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct
[87] Hugging Face (HuggingFaceTB). Smol-SmolTalk dataset card. https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
[88] Pleias and the AI Alliance. SYNTH dataset card. https://huggingface.co/datasets/PleIAs/SYNTH
[89] N. Ding, Y. Chen, B. Xu, Y. Qin, Z. Zheng, S. Hu, et al. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (UltraChat). arXiv:2305.14233, 2023.
[90] A. Mitra, H. Khanpour, C. Rosset, A. Awadallah. Orca-Math: Unlocking the potential of SLMs in Grade School Math. arXiv:2402.14830, 2024.
[91] P. Rajpurkar, R. Jia, P. Liang. Know What You Don't Know: Unanswerable Questions for SQuAD. ACL 2018. arXiv:1806.03822.
[92] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018. arXiv:1809.09600.
[93] F. Brahman, S. Kumar, V. Balachandran, P. Dasigi, V. Pyatkin, A. Ravichander, et al. The Art of Saying No: Contextual Noncompliance in Language Models (CoCoNot). NeurIPS 2024 Datasets and Benchmarks. arXiv:2407.12043.
[94] L. Jiang, K. Rao, S. Han, A. Ettinger, F. Brahman, S. Kumar, et al. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models. arXiv:2406.18510, 2024.
[95] S. Han, K. Rao, A. Ettinger, L. Jiang, B. Y. Lin, N. Lambert, et al. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs. NeurIPS 2024. arXiv:2406.18495.
[96] W. Liu, J. Chen, K. Ji, L. Zhou, W. Chen, B. Wang. RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions. arXiv:2501.00353, 2024.

## Appendix A. The ledgers in full

**Table A1. Parameter ledger of `final` and `v9`** (standard reading; leaves and shapes from the
checkpoints' manifests, `ckpt-000026-step-000029722` of `final` and `ckpt-000001-step-000030742` of
`v9`, counted with the row and read rules of `bes/budget.py`; library and index from `final`'s export
accounting; `tables.md` T1).

| row | what | `final` stored | `final` read per token | `v9` stored | `v9` read per token |
|---|---|---|---|---|---|
| core | W_q, W_k, W_v, W_o, W_up, W_down of 2 blocks | 14,118,912 | 14,118,912 | 14,118,912 | 14,118,912 |
| head | P (768 × 512), E_out (8,192 × 512) | 4,587,520 | 4,587,520 | 4,587,520 | 4,587,520 |
| small | norm gains, lanes, loop gains, sink, value gate, trust and candidate projections, critic | 306,776 | 306,776 | 306,776 | 306,776 |
| emb | input embedding (8,192 × 768) | 6,291,456 | 768 | 6,291,456 | 768 |
| ve | per-loop value embeddings (8,192 × 8 × 128) | 8,388,608 | 1,024 | 8,388,608 | 1,024 |
| tables | register (15 fields), candidate-input and trust tables | 838,448 | 18,736 | 838,448 | 18,736 |
| gain | attention-output gains `gain/o` (8,192 × 2 × 768) | – | – | 12,582,912 | 1,536 |
| page | `page/key`, `page/val` (8,192 × 2 × 8 × 192 each), `page/pin`, `page/pout` (2 × 768 × 192 each) | – | – | 50,921,472 | 595,968 |
| library | passages, question-answer layer, directory, titles, gate | 40,268,392 | 112,595 (worst case, retrieving step) | – | – |
| index | retrieval index (BM25) | 23,499,999 | 132,628 (worst case, retrieving step) | – | – |
| **total** | | **98,300,111** | **19,278,959** | **98,036,104** | **19,631,240** |

Notes. `final` reads 19,033,736 numbers on a step that does not retrieve; measured retrievals on 500
proxy turns read at most 105,826. The critic head (196,864 numbers in `small`) is counted as read
although the decoder does not use it. In the strict reading, `final` read 19.62–19.81 million at 1,024
positions and 20.07–20.49 million at 2,048 on 6 documents (`docs/ACCOUNTING.md`), so neither
configuration meets the strict reading at 2,048 positions; `v9` read 21.2 / 23.8 / 24.6 million
(minimum / median / maximum over 40 documents) at 1,024 positions and 22.4 / 26.7 / 28.2 million at
2,048 (`docs/ACCOUNTING.md`, `docs/V9.md` §10). The audit's forward pass also logs the distinct numbers
read for every generated token. The audit's three checks are described in
Section 4.2; training computes dense products over whole tables and masks them, which changes neither
the function nor the counts.

## Appendix B. The model in detail

**B.1 Core.** For the token x_t at position t, with d = 768, two weight-shared blocks i ∈ {0, 1}, R = 8
loops and N = 2 lanes:

    e_t = RMSNorm( E[x_t] + Σ_f R_f[reg_{t,f}] + W_c Σ_k μ_{t,k} ( C[c_{t,k}] + Σ_g F_g[φ_{t,k,g}] ) )
    ℓ_1 = ℓ_2 = e
    for r = 1 … R:
        u   = Σ_n α_{r,n} ℓ_n
        u   = B_1( B_0( u ; E_ve[x, r] ) )
        s   = RMSNorm(u) ⊙ g_r
        ℓ_n = Σ_m M_{r,nm} ℓ_m + β_{r,n} s                              (n = 1, 2)
    h_t = RMSNorm( ℓ_1 + ℓ_2 ) ⊙ g_head
    z_t = 15 · tanh( (h_t P E_outᵀ) / 15 )                              (float32; P: 768 × 512)

    B_i(u):  a = Attn_i( RMSNorm(u) ⊙ g_{1,i} ) ;   u ← u + a
             x = RMSNorm(u) ⊙ g_{2,i} ;   u ← u + W_down,i · relu(W_up,i · x)² + Π_i(x; x_t)

E, R_f, C, F_g are the token, register and candidate tables; μ_{t,k} is the tape's slot mask; the lane
coefficients start at α_r = onehot(r mod 2), M_r = I, β_r = 1. There are no biases; Π_i is the token-
neuron path (absent in `final`). The per-loop value embedding E_ve[x_j, r] (128 numbers per loop) is
added to the first 128 value dimensions of block 0 through the gate 3σ(W_g x_j[1:12]) [5, 22].
Parameters are float32 master copies.

**B.2 Attention with a sink.** Queries and keys are rotated (RoPE) and L2-normalised on their 126
content dimensions; the query is scaled by a learned per-head gain γ_h (initially √126). The two
reserved dimensions are set to q = (1, 0) and k_j = (b_j, 0), so the logit is γ_h cos(q_t, k_j) + b_j for
any attention kernel. At a `BOS` position the key's content is zeroed, b_j is a learned scalar and the
value a learned vector (the prior mean); elsewhere b_j = 0. The mask is causal and within the segment,
and every segment starts with `BOS`. With exact-match one-hot keys of infinite sharpness, a sink logit
β + log λ and value m₀, the output for a query with n_a matching keys is (Σ v_j + λ m₀)/(n_a + λ), the
posterior predictive of a Markov chain under a symmetric Dirichlet prior when the values are one-hot
successors and m₀ is uniform (`docs/MATH.md` B1). In the micro ablation the sink was neutral (−0.07%
when removed) and kept, as harmless and with outside evidence [42, 43].

**B.3 The output law.** The network's distribution q = softmax(z_t) is combined with the tape:

    p(v) = ( q(v) + Σ_k m_k [c_k = v] ) / ( 1 + Σ_k m_k ),
    m_k  = exp( softcap( Σ_f β_f[φ_{k,f}] + ⟨h_t A, c_k⟩ − 6 ) ) · μ_k · s(src_k),

where the trust tables start at zero (every active slot starts at mass e⁻⁶) and s(·) scales the
masses by the role of the candidate's source. The normalising constant is 1 + Σ m, so no second pass
over the vocabulary is needed; with m_v = (n_v − d t_v)/(θ + d T) it is the Pitman–Yor posterior
predictive with the network as base measure [38]; the loss can exceed the network's own by at most
log(1 + Σ m); and with all masses zero it equals the network's cross-entropy bit for bit (tested).
Decoding tempers the base measure only (Appendix E). The *tape gain* is the mean of
−log q(y) − (−log p(y)) over training tokens: what the masses buy the trained model.

**B.4 Token pages: the evidence before training, and the allocation question.** The evidence for
spending the 80 million stored numbers one way or the other was small-scale.

- *Written library* (`docs/E1_COVERAGE.md`): Simple English Wikipedia lead passages and
  question-answer pairs in independent xz blocks with a BM25 index (63,768,391 numbers), behind a
  calibrated gate. After a review found confident wrong retrievals, the gate was refitted to
  precision 0.8 on every development set. Coverage / precision before → after: NQ-open validation
  35.0 / 54.1 → 6.0 / 76.9; PopQA 6.5 / 40.3 → 0.5 / 75.0; head-entity probes 82.3 / 98.7 → 74.4 / 99.1;
  common-name probes (the new probe definition for both) 24.3 / 35.6 → 81.3 / 99.9; NQ-open validation
  questions without a near-duplicate stored question 18.5 / 37.3 → 1.0 / 43.5. The last row shows that
  the library answered mainly through stored near-duplicates. The gate opens on 8 of the 200 dev first
  turns of the prompt set, so the blind comparison could hardly have seen the library.
- *Token pages* (`docs/V9.md` §2): in a CPU laboratory (width 48, 2 shared blocks × 4 loops, 1,040
  token types, 12 million tokens, one seed per arm, paired noise about 0.01 nats), token-keyed
  parameters on the writing side of attention helped and on the matching side hurt (class-specific
  `W_o` −0.079 nats, `W_v` −0.039, `W_q` and `W_k` together +0.046); private feed-forward neurons,
  class experts and dense widening fell on one curve ΔL = −c · E_{x∼f}[ln(n(x)/n₀)], c = 0.070 at
  6.6 million tokens (ten arms, root-mean-square residual 0.008 nats); the complete page gave −0.252
  nats at 12 million tokens against a base of 3.734. In the micro ablation (width 128, 1 million tokens)
  the page gave −0.61% of held-out loss, below the pre-registered resolution (margin 0.84%, bar 1%): not
  resolved.
- *The design's own prediction at full scale* (`docs/V9.md` §7): the project's calibrated scaling-law
  estimate put `final` at 2.507 nats per token and Falcon-H1-Tiny-90M at 2.358, and `v9` at about 2.45
  (2.41–2.48) after published gains of token-keyed tables; "v9 therefore narrows the predicted gap to
  the 90M opponent and does not close it". Losses per token across different tokenizers are not
  comparable; Table E3 gives bits per byte instead.

**B.5 Components and the pre-registered micro ablation.** Table B1 gives the micro ablation (width 128,
8 loops, 1 million tokens per arm, seed replicate 0.42%, margin 0.84%) and the decisions.

**Table B1. Micro ablation and decisions** (`docs/E2_ABLATIONS.md`; `docs/CHANGES.md` §14; relative
change of held-out loss when the component is removed, positive: removal hurts; training mixture /
chat / assistant tokens).

| component | removal | pre-registered rule | decision |
|---|---|---|---|
| candidate tape (input features and output masses) | +11.0% / +16.4% / +19.3% | keep | on |
| … the output masses alone | +8.5% / +12.6% / +14.9% | keep | on |
| … the input features alone | +0.9% / +1.0% / +1.2% | off on the mixture, on by the chat reading | on (deviation) |
| register embeddings | +0.8% / +1.1% / +1.1% | off on the mixture, on by the chat reading | on (deviation) |
| per-loop value embeddings | +1.5% / +2.2% / +2.3% | keep | on |
| confidence channel | +0.7% / +0.8% / +0.8% | off | off |
| structural rotary clocks | +0.3% / +0.4% / +0.3% | off | off |
| attention sink | −0.1% / −0.1% / −0.1% | keep (outside evidence) | on |
| 4 loops instead of 8 / 12 instead of 8 | −0.12% / +0.26% (mixture) | within noise | 8 |

The rule asked new components for at least 1% without saying on which loss; the automatic report used
the training mixture, by which the candidate input features and the registers would be off. We kept
them on the chat and assistant losses, a reading chosen after seeing the data. The registers also
failed their own step-function gate (exact counting 79.2% with the registers against 92.7% for an arm
trained without them at 2 million tokens, `docs/E4_PROBES_2M.md`) and were kept. Both are deviations.

**B.6 Optimisers.** Matrices train with MuonH, a port of the looped-model reference (Nesterov momentum
0.95, five steps of a polar iteration [53], the update rescaled to and projected back on the sphere of
the matrix's initial Frobenius norm [56], no weight decay) [21]; all other leaves train with Adam
(β = 0.8, 0.95; ε = 1e−10; no weight decay) in groups with their own learning rates: embeddings 0.3 (the
token-neuron keys train in this group), unembedding 0.004, feature tables 0.01, token-neuron values
0.004, gains 0.01, all multiplied by (768/640)^−0.5; the matrix rate is 0.014 before the transfer rule.
A transfer rule of the reference multiplies every group by (tokens per step / 2¹⁸)^0.558 (here exactly 1)
and the matrix group by (horizon tokens / 9.83 × 10⁸)^−0.32 [57], with the horizon fixed once by the
start-up benchmark at 1.3117 × 10¹⁰ tokens. The rule was fitted over a smaller range than ours; at
micro scale it cost +14.4% of loss when applied at 4,096 tokens per step and was not applied there.

## Appendix C. Training under a wall clock

**C.1 A schedule that is a function of progress.** With T the training seconds of the kept trajectory
and R = max(0, train_end − now), the trainer defines p = T / (T + R) (`bes/train/schedule.py`; tying
the decay to the time budget follows [1–3]). While training, T grows and R shrinks at the same rate, so
p is linear in wall time and reaches 1 when the window closes, whatever the throughput. The
learning-rate factor (a linear warm-up over the first 0.2% of p, then 1 − 0.9p), the sequence length
(1,024 tokens; 2,048 from p = 0.67), the chat share of the mixture (0.25 until p = 0.67, then rising linearly
to 0.70 at p = 0.92, then 0.70; `chat_frac_at` in `bes/train/schedule.py`), loss on the assistant tokens
only of conversations from p = 0.92 (base text still counts), and `final`'s library shares are all functions
of p. The ramp shows in the loader: base-text rows were 75% of the rows of the loader block of step 36,850
(p = 0.670) and 38.4% in that of step 46,350 (p = 0.879), where the trainer logged a chat share of 0.627; the
tail began at step 48,190 (stage event 17:09:57), after which the logged micro-batches give user and system
tokens no weight (at step 48,150 they had 1,833 of 14,924 units). By itself the ramp is expected to lower
the loss of a training step and the held-out chat loss of any configuration; this run does not measure
that effect apart from the others. Two
consequences were observed. A pause leaves T fixed and shrinks R, so p advances while nothing trains:
the graft's pause moved p from 0.5049 (stop event) to 0.5112 (restore event). Moving the end of
training later raises R, so p falls: when the post-training reserve was cut from 220 to 118 minutes at
step 12,062 and the training end moved from 16:51:49 to 18:33:49, p fell from 0.2415 (stop event) to
0.2190 (restore event) and the learning-rate factor rose from 0.783 to 0.803 (Figure C1d, line 4). The schedule is
continuous in the parameters and the optimiser state, not monotone in p. (Step records, the nearest
logged steps, give 0.2413 → 0.2196 and 0.5045 → 0.5118; we quote the events.)

**C.2 Surviving the loss of a machine.** Checkpoints are written atomically every 25 minutes
(parameters, optimiser state, data cursor, random state, time ledger); after each one a snapshot is
copied off the runtime. On a new runtime the snapshot is restored, the deadline is moved by the
downtime (which keeps p continuous) and the trainer continues. A loss guard rolls back on three
consecutive spikes above 1.3 times the moving average; it never fired (0 rollbacks in the ledger). Restarting on a new machine
exposed two errors no rehearsal had: the loader replays its journal and needs every shard it lists,
which a new machine must download and tokenise again (the trainer now waits for them, with 24
tokenising workers: 363 shards in 9 minutes), and two sources that share raw files deleted them
before both had read them. Together they left the third runtime's TPU idle for 48 min 19 s before
training resumed (commits `e096dff`, `77305dc`; Figure C2).

**Table C1. TPU runtimes** (`docs/CHANGES.md` §19.1–19.4; the operator's TPU-time ledger).

| runtime | counted (UTC) | seconds | what it produced |
|---|---|---|---|
| 1 | 23 Nov 02:58:32 – 05:00:37 | 7,325 | profiling and the rewrites of Section 5.2; its training runs were discarded; returned at the user's request |
| 2 | 23 Nov 17:13:00 – 19:30:00 (from the earliest possible allocation) | 8,220 | training from step 0 to step 5,075 (kept through a snapshot); taken back by the provider at about 19:29 |
| 3 | 24 Nov 01:05:55 – 20:41:04 | 70,509 | training resumed at step 5,075 at 01:54:14 and stopped at its deadline, 18:33:49, at step 51,812; post-training stages 18:33:55–20:02:22; given back by hand at 20:41:04 |
| total | | 86,054 of 86,400 | 62,893 s (73.1%) in training steps of the kept trajectory (trainer's ledger) |

Gaps without a runtime: 12 h 12 min 23 s and 5 h 35 min 55 s. Held to its limit the TPU budget would have
ended at 20:46:50 on 24 November; 346 s were left unused. The run's deadline was 20:31:49.

**Table C2. Events of the run** (train log: stop and restore records; tokens are steps × 262,144; live
rates are the medians of Table 2).

| # | step | tokens | stopped → restored (UTC) | event |
|---|---|---|---|---|
| – | 0 | 0 | 23 Nov 17:46:32 | training starts (`configs/final.py`, commit `98598c2`, 16-row micro-batches) |
| 1 | 5,075 | 1.33 × 10⁹ | runtime 2 lost about 19:29 → 24 Nov 01:54:14 | resumed on runtime 3 from the step-5,075 checkpoint |
| 2 | 6,647 | 1.74 × 10⁹ | 02:26:02 → 02:28:59 | stock-refusal rule (commit `566517a`) |
| M1 | 8,659 | 2.27 × 10⁹ | 03:09:43 → 03:15:37 | pause: micro-batch pieces timed |
| 3 | 9,332 | 2.45 × 10⁹ | 03:29:17 → 03:31:24 | 4 pieces per micro-batch (`b05bfe7`) |
| M2 | 10,242 | 2.68 × 10⁹ | 03:48:04 → 03:57:28 | pause: on-chip memory flags timed |
| 4 | 12,062 | 3.16 × 10⁹ | 04:30:39 → 04:32:03 | post-training reserve cut to 118 min; training end 16:51:49 → 18:33:49 (`782be85`) |
| 5 | 22,613 | 5.93 × 10⁹ | 07:44:14 → 07:53:35 | kernels timed; one-tile kernel and ReLU² backward (`6bf9b6e`) |
| 6 | 25,280 | 6.63 × 10⁹ | 08:35:05 → 08:36:31 | stock-deflection rule (`4f4ef45`) |
| 7 | 29,722 | 7.79 × 10⁹ | 09:45:34 → 09:58:45 | `v9` timed; graft into `configs/v9.py` (`9a4d652`) |
| 8 | 31,912 | 8.37 × 10⁹ | 10:52:36 → 10:57:50 | `v9` switches timed and adopted (`769f240`) |
| P1 | 36,867 | 9.66 × 10⁹ | no stop (stage event 12:47:51) | rows of 2,048 tokens; the chat share starts to rise (schedule) |
| P2 | 48,190 | 1.26 × 10¹⁰ | no stop (stage event 17:09:57) | chat share 0.70; conversations train on assistant tokens only (schedule) |
| – | 51,812 | 13,582,204,928 | 18:33:49 (final checkpoint 18:33:54) | end of training at the deadline; the stages after training to 20:02:22 (C.4) |

Averaged over its 32,320.8 training seconds, the trajectory up to the graft trained at 241,065
tokens/s (7,791,443,968 tokens; trainer's ledger). At the end the ledger holds 51,812 steps, 13,582,204,928
tokens and 62,893.1 training seconds (17.47 h), with 23.2 s booked to checkpoint writes, 279.6 s to set-up,
10 restarts and no rollback (`lead5_work/numbers.json`).

![Figure C2](figures/F2_throughput.svg)

**Figure C2.** Training throughput against UTC time on the three runtimes (axis broken where no runtime
was held). Runtime 1 (23 November) holds runs before run main16, all discarded: start-up benchmarks
(diamonds) of `final` before and after the glue rewrite and of `v9` before it (orange), and single log
lines of `final`'s training (dots). Runtimes 2 and 3, run main16: lines, the rate achieved between
consecutive step records (tokens ÷ wall time, checkpoint writes included); numbers, the live rate of
each stretch in 10³ tokens/s with its number of step records n (Table 2, Figure F2); in runtime 2, the
start-up benchmark (diamond) and the first log line (dot). Light gray bands: TPU held but not training;
hatched: no TPU (its length printed above the axis). Tags: restore times of the restarts of Table C2; P1
and P2 (dashed): rows of 2,048 tokens from p = 0.67 and the assistant-only tail from p = 0.92, without a stop.
Records to step 51,800 (24 November, 18:33:30 UTC), the last step record; training stopped at 18:33:49 and
the stages after it (18:33:55–20:02:22) and the hold to 20:41:04 are not drawn. A change of level at a tag is
a before-and-after difference, not the effect of that change alone.


**C.3 The training curve.** Table C3 summarises the step losses logged every 50 steps (each the mean
over the step's 262,144 tokens) in windows of progress (the whole curve is in Figure C1); the role split and the tape gain are those of
the last micro-batch of each logged step (16,384 tokens), weighted.

**Table C3. Training loss of `final` by window of progress** (`review1_work/win.py` on the stopped
run's train log; sample standard deviations).

| p | logged steps | step loss: mean (sd) | positions with role "assistant" | positions outside conversations | tape gain (nats) |
|---|---|---|---|---|---|
| 0.00–0.01 | 10 | 3.613 (0.535) | 2.835 | 3.823 | 0.319 |
| 0.05–0.10 | 48 | 2.699 (0.047) | 1.717 | 2.936 | 0.035 |
| 0.15–0.20 | 48 | 2.613 (0.042) | 1.605 | 2.861 | 0.026 |
| 0.25–0.30 | 60 | 2.550 (0.041) | 1.551 | 2.792 | 0.024 |
| 0.35–0.40 | 56 | 2.530 (0.046) | 1.521 | 2.771 | 0.023 |
| 0.45–0.50 | 68 | 2.496 (0.037) | 1.491 | 2.738 | 0.022 |

These are training losses on a stream whose composition changed at steps 6,647 and 25,280, in windows
of progress, not of tokens (p fell once). After the graft the 93 logged steps to 34,350 average 2.4675
(sd 0.0399); the stream changed at the graft, so the two numbers are not comparable. The 226 logged
steps at 2,048 tokens before the tail (36,900–48,150) fall from 2.404 (mean of the first 5) to 1.815 (the
last 5) while the chat share of the stream rises from 0.25 to 0.70 and the learning-rate factor falls from
0.40 to 0.17; the 73 steps of the tail (48,200–51,800), in which conversations train on their assistant
tokens only, fall from 1.778 to 1.733 as the factor falls to 0.10. Neither level is comparable with earlier
ones. The tape gain fell
from 0.319 nats in the first 1% of training to 0.022 nats (0.9% of the loss) by p ≈ 0.5; in the micro
ablation the tape switched off at evaluation cost 10.5% of the loss at 1 million tokens. One reading,
which the micro report anticipated, is that the network learns at scale most of what the exact
in-context statistics offered it early; the tape gain measures how much the trained model leans on the
tape, not what training without it would give. On the held-out rows the tape gain is 0.013 nats on web
text and 0.029 on chat at step 29,722.

![Figure C1](figures/F1_loss.svg)

**Figure C1.** Training loss of run main16 against training tokens: (a) the whole run on logarithmic
tokens; (b) from 10⁹ tokens on, linear. Dots: the loss of one optimiser step (the mean over its 262,144
tokens), logged every 50th step: 594 records of `final` (steps 50–29,700) and 442 of `v9` (steps
29,750–51,800, the last step record, 18:33:30 UTC on 24 November).
Lines: a centred average over log₁₀(tokens), computed separately within each stretch between restarts
and phase changes: each record is weighted by a Gaussian (σ = 0.03 decades, cut at 3σ) times 1/tokens,
so that the window is centred on the record (to within 0.001 decades), and near the end of a stretch
the window shrinks symmetrically instead of becoming one-sided; the line is drawn where it averages at
least 15 records. A step at a restart would show
as a break between two lines; near the ends of a stretch the line rests on fewer records and wanders by
about ±0.01 nats. Vertical lines: the restarts of Table C2 (solid; M1, M2: measurement pauses), the
switch to rows of 2,048 tokens at p = 0.67 (P1, dashed) and the tail at p = 0.92 (P2, dashed). (c, d) The
learning-rate factor and progress p as logged. (e) From p = 0.92 the loss covers the assistant tokens of
conversations (and base text) only; its 73 records are left out of (a) and (b) and drawn on their own scale,
not comparable with (a) and (b). The loss mixes 22 sources whose shares change with p and the data rules;
the stream changed at the graft, and from P1 its chat share rises linearly (Appendix C.1), which by itself is
expected to lower the loss of a step: levels on either side of a restart, of P1 or of P2 do not compare
configurations.

**C.4 After training.** The 118 minutes after training were planned before the end (`docs/CHANGES.md`
§19.5; `post_plan` = `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=15`) and ran by the
rules fixed in `bes/train/prefopt.py` and `scripts/run_24h.py` (Table C4); the run finished at 20:02:22, 29
minutes before its deadline.

**Table C4. The stages after training** (the stage ledger `run/stages.json`, the run log and the manifest's
`decode` record in the last snapshot; UTC; `lead5_work/numbers.json` keys `stages`, `prefopt`, `select`).

| stage | UTC | slice; used (s) | what it did |
|---|---|---|---|
| train | to 18:33:55 | – | stopped at its deadline (18:33:49), step 51,812; final checkpoint 18:33:54 |
| sample | 18:33:55–18:37:07 | 180; 192.5 | sampled the trained model on 17 held-out prompts of 12 chat and maths sources (about 28 candidate pairs); 20 pairs labelled by the exact verifiers were kept |
| critic, judge | – | 0; 0 | skipped (planned with 0 minutes) |
| prefopt | 18:37:07–19:06:51 | 3,600; 1,783.8 | DPO (β = 0.1, plain Adam on every leaf, gradient-norm clip 1.0, one epoch of 551 steps of 64 pairs) on 35,201 pairs, 1,826 more held out by prompt: 37,007 UltraFeedback pairs [76] left after dropping 306 identical pairs, 8,366 whose chosen reply is over 3 times the rejected one and 7,995 too long, plus the 20 sampled; two learning rates from the trained weights, both accepted (Table C5); kept 3 × 10⁻⁶, the higher held-out accuracy |
| select | 19:06:51–20:01:47 | 3,897; 3,295.3 | with the preference-optimised weights, the default, one particle and no length band on the 200 `dev` prompts (19.6, 4.0 and 31.3 minutes); mean exact scores 0.910, 0.850 and 0.915; against the default, one particle worse on 18 prompts and better on 1 (p = 7.6 × 10⁻⁵), no band better on 2 and worse on none (p = 0.50); the rule keeps the default unless a variant is better on significantly more prompts (p < 0.05): default kept; library: none built |
| export | 20:01:47–20:02:22 | 300; 35.0 | the preference-optimised weights to `final/` (Appendix M, Table M10): 98,036,104 numbers in float16, 16-bit check 2.34440 against 2.34443 nats on 43,138 held-out tokens (tolerance 0.002), passed |

**Table C5. The two candidates of preference optimisation** (`run/post/prefopt_report.json`; held-out
preference accuracy on 1,826 pairs with its standard error; sampling statistics on 64 held-out prompts,
paired with the start's; the checks of `bes/train/prefopt.py`).

| quantity | start (trained weights) | 1 × 10⁻⁶ | 3 × 10⁻⁶ (kept) | the check |
|---|---|---|---|---|
| held-out preference accuracy (se) | 0.500 | 0.688 (0.011) | 0.694 (0.011) | gain above 1 × its standard error (0.011) |
| held-out assistant-token loss | 1.5043 | 1.5172 (+0.86%) | 1.5208 (+1.10%) | at most +3% |
| mean sampled length (tokens) | 284.9 | 282.7 (0.99 ×) | 270.0 (0.95 ×) | 0.7–1.6 × the start's, by 2 standard errors |
| repeated-4-gram rate | 0.217 | 0.186 (0.86 ×) | 0.189 (0.87 ×) | at most 1.2 × the start's + 0.01, by 2 standard errors |
| training steps; seconds; training accuracy | – | 551; 485; 0.677 | 551; 424; 0.685 | – |
| accepted | – | yes | yes | best accuracy among the accepted |

Our replies to the 200 test prompts were generated on the runtime's host cores next to these stages
(`scripts/gen_test_replies.py` through `endgame_testgen.sh`, 40 workers of one core each): from the trained
weights at 18:36:25–19:07:06 and from the preference-optimised weights at 19:31:22–20:06:32. The second
generation's export has the same source checkpoint, fingerprint, `weights16.npz` SHA-256 and decoding choice
as the run's own export (`testgen.json`), so its replies are the official model's; the first is the
before-and-after comparison of Section 7.7.

**C.5 The configuration comparison that was not run.** A comparison of `v9` and `final` had been
pre-registered on 23 November: 75 minutes of each on the chip, decided on held-out assistant loss at
equal training time (`docs/CHANGES.md` §19). It was not run: first both configurations measured three
times slower than planned and the time went into finding out why (§19.1); then the comparison was
judged not worth about a tenth of the training time, because by the design's own numbers the expected
gain of the pages and the tokens lost to their slower step roughly cancelled (§19.2); `final` was
trained.

## Appendix D. Data and its hygiene

**D.1 Sources and the realised mixture.** The registry (`bes/data/sources.py`, `docs/DATA_SOURCES.md`)
holds 26 entries, each checked against the Hub on 21 November 2026 and pinned. Weights are relative
within a kind. Base text: ClimbMix [82] 0.85, FineWeb-Edu [83] 0.10, Simple English Wikipedia 0.05.
Chat: smol-smoltalk [87] 0.30, SmolTalk Magpie-Ultra [12] 0.20, Tulu 3 SFT [77] 0.15 (without its
non-commercial, non-English and hard-coded identity subsets), SYNTH closed-book [88] 0.12, UltraChat
[89] 0.08, and five smaller SmolTalk subsets. Retrieval format: SYNTH open-book 0.30, glaive RAG 0.15,
RAG-Instruct [96] 0.15, SQuAD v2 [91] 0.15, HotpotQA [92] 0.10, Dolly 0.08, neural-bridge 0.07. Maths:
Orca-Math [90] 0.65, GSM8K [72] 0.35, with its annotations turned into calculator spans whose results
the host computes. Preferences: UltraFeedback [76] (binarised, ties dropped). Rows are 75% base text
and 25% conversations. Plain means over the 594 loader blocks logged by `final` give the realised token
shares: base text 75.04%, chat 14.46%, retrieval format 5.24%, maths 3.04%, library tasks inside chat
rows 1.32% and library rows 0.89% (`review1_work/mix.py`). Over the first 48 blocks after the graft the
library shares are 0 and the others range over 75.5–76.0% base text, 14.8–17.1% chat, 3.8–6.2%
retrieval format and 2.7–4.4% maths. 12% of the chat samples get a verified constraint written into
their prompt from what the reply actually does (hindsight relabelling; `bes/data/hindsight.py`).

**D.2 Stock refusals.** At step 5,075 (p ≈ 0.1) a CPU sample of the snapshot with six particles answered
"Write a product review of a pair of running shoes in under 50 words" with "I'm sorry, but I can't
assist with that. It's important to respect user privacy and confidentiality …"; all six particles
refused (`N/midrun_sample_step5075_p6.md`; the prompt is test prompt instruct-034). The Tulu 3 mixture
holds safety refusals from CoCoNot [93], WildJailbreak [94] and WildGuardMix [95], and a model of this
size does not learn which requests they belong to. From step 6,647 (restore at 02:28:59) the loader
skips any conversation in which an assistant turn opens with a stock refusal (commit `566517a`); the
exact scorer counts such a reply as failed, so the decoder's selection prefers a particle that answers;
and preference sampling skips prompts whose reference reply is a refusal. The rule was written from
this one sample; the rates below were measured six hours later, at 08:31, with the deflection scan:
an assistant turn opens with a stock refusal in 4.85% of 30,000 tokenised conversations of Tulu 3,
0.83% of SYNTH's, 0.67% of the SmolTalk system chats and 0.41% of UltraChat's
(`N/deflection_check2.json`).

**D.3 Stock deflections.** At step 22,613 (p = 0.40) the model answered "My phone fell into a sink full
of water. What should I do right now?" with "I'm sorry, but I don't have access to the latest updates
on the latest phone fires …" (`N/midrun_sample_step22613.md`; test prompt chat-017). A scan at 08:20 of
how assistant turns open in every tokenised source (`N/openings.json`, 40,000 segments per source) found
such openings concentrated in a few sources: in the SmolTalk system chats 10.60% of the turns open with
an apology, 2.14% with "As an AI …" and 1.51% with "I don't have access …"; in UltraChat 2.81% open with
"I don't have …" and 2.21% with "I don't have access …". A rule wider than the refusal rule
(`bes.eval.exact_score.stock_deflection`: the reply opens by saying what the assistant is or lacks, or
with "I'm sorry, but …" whatever follows) removes from the training stream, from step 25,280 (commit
`4f4ef45`), every chat conversation with such a turn. On 30,000 conversations per source it removes
29.10% of the system chats (29.81% of their tokens), 9.06% of UltraChat (7.10%), 1.23% of Tulu 3 beyond
its refusals (1.10%), and under 1% of every other chat source; retrieval, maths and text sources are
untouched, because "the passage does not say" is the task there. The live loader's skips agree with the
scan within about one percentage point (Figure D1).

**D.4 What the deflection rule removes.** We labelled by hand the 30 stored dropped openings of each of
the two sources the rule hits hardest (one annotator, the author of this section, not blind to the rule,
reading the stored first 150 characters; `lead2_work/deflection_labels.py`). Of 60 dropped
conversations, 8 open with a pure deflection and no help in the visible text; 45 open with a disclaimer
followed by help or a pointer ("I do not have access to location-specific information. However, here
are some general tips …"); 3 ask for input the user referred to but did not give ("I don't have access
to the entire code you are referring to. Can you please provide me with the code …"); 2 are
safety-relevant referrals, one of them a reply to distress ("I'm really sorry that you're feeling this
way, but I'm unable to provide the help that you need. It's really important to talk things over with
someone …"); 1 answers directly after a persona prefix; and 1 says correctly that the information is
not in the given text. All 8 pure deflections are in the system chats; none is in UltraChat. The rule
therefore removes what it targets, the opening, together with useful behaviour that follows it: the model
may learn to answer without the information instead of asking for it, and it loses examples of safe
completions for distress. The near misses the scan stored (40 per source, not a random sample, so they
cannot estimate recall) include persona-style deflections the rule keeps ("As a character in a medieval
fantasy setting, I don't have access to current real-world information.").

**D.5 What this does and does not show.** Both rules are functions of the data, so the stream stays
reproducible, and both were triggered by the model's own failures rather than by a list written in
advance. Their effect on the model is not measured: there is no ablation, the second rule acted only on
the last part of `final` and on `v9`, and the triggering anecdotes are single prompts. On the 60
development prompts the replies of `final` (step 25,280) and of `v9` (step 31,912) contain no stock
refusal or deflection (Table E3), and on the 252 test replies of every model compared (ours with six
particles, with one and before preference optimisation; Falcon-H1-Tiny and its two control arms; both arms of
MobileLLM-R1.5) the exact scorer flags none (Table 8). This counts unwanted refusals and deflections only; it
says nothing about behaviour where a refusal would be right.
From step 6,647 on, the model is not trained to refuse anything; this was the operator's decision,
reported to the user, who could have overruled it, and the evaluation does not test behaviour on
harmful requests. The preference pairs from UltraFeedback are not filtered by either rule (2.44% of its
scanned responses open with "As an AI …").

![Figure D1](figures/F5_openings.svg)

**Figure D1.** (a) Share of each source's assistant turns whose opening (first 44 tokens) matches a stock
pattern, from the scan of the tokenised sources (24 November 08:20; all segments of small sources,
40,000 sampled otherwise); "Sure …" is a control that no rule removes; a turn can match several
patterns. (b) Share of each chat source's conversations that the stock-refusal rule (light gray, live
since step 6,647) and, in addition, the deflection rule (violet, live since step 25,280) leave out,
measured with the loader's functions on 30,000 sampled conversations per source (2,260 for
smoltalk_everyday, all it has). Black ticks and the column on the right: share of reads the live loader
actually skipped for either rule from step 25,280 to step 51,800 (18:33:30 UTC on 24 November, the last
step record; `final` and `v9` together), with the number of reads. In both
panels "<0.1" means above 0 and below 0.05%, and "0" means none.

## Appendix E. Decoding

**E.1 Temperature on the base measure only.** With the output law of Appendix B.3, tempering the whole
mixture sharpens the tape's share M/(1 + M), a calibrated probability, and makes copy loops absorbing:
an exact Bayesian predictor endorses a repetition it has seen (after r repeats its continuation
probability is (r + λ/V)/(r + λ)), and below temperature 1 the probabilities of escaping form a
convergent series (`docs/NEGATIVE_RESULTS.md` §1). The decoder therefore tempers the network's
distribution only, p_T(v) = (softmax(z/T)(v) + Σ_k m_k[c_k = v])/(1 + Σ m), and multiplies the masses of
candidates whose source is the reply being written by a self-evidence weight of 0.25. On the micro
model (48 held-out chat prompts, `docs/E5_SELF_GENERATION.md`) at temperature 0.5: tempering the
mixture gave a repeated-4-gram rate of 0.326, loops in 13 of 48 replies and 61.4% of tokens copied from
the prompt; tempering the network only with self-evidence 1 gave 0.134, 1 of 48 and 6.8%; with
self-evidence 0.25, 0.033, 0 of 48 and 5.5%; with the tape off, 0.032, 0 of 48 and 0%. Training keeps the
self-evidence masses at weight 1: trained with 0.25 the model learns the weight back (+0.07% loss),
trained with 0 its chat loss is 8.6% worse. By the letter of the pre-registered rule every tape setting
failed E5, because its copy share (0.04–0.06) exceeded that of a model trained without the tape
(0.000); the dataset's own replies, measured the same way, copy 0.074 ± 0.008 of their tokens. We read
the rule's copy criterion as wrong rather than the mechanism as failed, a reading made after seeing
the data.

**E.2 Particles, exact potentials and selection.** Each reply is drawn as K = 6 particles of the same
model. The proposal is the network's distribution multiplied by token-local exact potentials (structure,
explicit counts compiled from the user's words, a budget that keeps every particle able to finish),
renormalised; the incremental weight of a particle is the probability mass its mask allowed, which makes
the particle average an unbiased estimate of the normalising constant (`docs/MATH.md` F1, F2) [64–68].
Particles are resampled at sentence boundaries when the effective sample size falls below K/2. The reply
is chosen by a satisficing rule: particles that failed (unfinished, a violated constraint, degenerate
output including stock refusals, a prose reply cut mid-sentence) are dropped, a length band is kept, and
the most mutually similar of the rest is chosen; maths answers go by majority [70]. Selecting one of K
samples by any rule changes the distribution by at most log K = 1.79 nats (K = 6) in KL divergence; the
smaller value log K − (K − 1)/K = 0.96 nats is the KL of best-of-K under a continuous score and only an
upper bound in general [69]. Defaults: temperature 0.6 and top-p 0.9 [85] on the network's
distribution, n-gram blocking of order 6 outside code and quotes, 768 new tokens; `v9` ships without a
library.

**E.3 A stand-in measurement.** Our decoder driving Falcon-H1-Tiny's own weights through llama.cpp on
63 development prompts, 47 of them with machine-checkable constraints (`docs/E9_DECODING.md`): every
constraint was met on 31 of 47 prompts with plain sampling (66% [52, 78]), 43 of 47 with six particles,
the failure filter and the consensus (91% [80, 97]), and 46 of 47 with the exact potentials as well
(98% [89, 100]); paired, the filter was better on 12 prompts and worse on none (sign test p = 0.0005),
the potentials better on 15 and worse on none (p = 0.0001), at 5.8 times the response tokens and 4.0
times the CPU time per turn. These prompts were also the development set of the constraint compiler,
which reads every annotated constraint of 96% of them and of 85% of the held-out half, so the gains are
measured where the compiler is at its best. It tests an implementation, not a method.

**E.4 The decoding studies.** Six settings on the first 60 development prompts, generated on the
runtime's idle host cores from the export of `final` at step 25,280, and on `v9` at step 31,912 the
default against 3-gram blocking; read blind by Opus 5.5 agents, one reading per pair
(`N/study/proxy/reports/ab/`; exact scores `lead2_work/exact_dev.json`):

**Table E1. Decoding settings against the default** (W / L / T from the variant's side; "both bad"
counted among the ties).

| setting | W / L / T | of which "both bad" | sign test p | constrained prompts met (of 14) | replies cut at the limit (of 75) |
|---|---|---|---|---|---|
| default (6 particles, T 0.6, top-p 0.9, 768 tokens) | – | – | – | 12 | 2 |
| 1 particle | 1 / 3 / 56 | 53 | 0.63 | 11 | 10 |
| 3-gram blocking | 6 / 3 / 51 | 50 | 0.51 | 12 | 1 |
| temperature 0.4 | 5 / 6 / 49 | 46 | 1.0 | 11 | 1 |
| temperature 0.8 | 2 / 7 / 51 | 49 | 0.18 | 13 | 2 |
| 320-token budget | 1 / 1 / 58 | 55 | 1.0 | 11 | 4 |
| `v9` step 31,912: 3-gram blocking against its default | 4 / 8 / 48 | 44 | 0.39 | 13 (default 12) | 1 (default 0) |

Neither study can separate the settings: most pairs were ties, most of them "both bad". The six
settings were tried on `final`; on `v9` only 3-gram blocking was tried, read by the same readers with
one reading per pair.

**E.5 Decoding is bound by the host.** On the chip one decoding step took 12.1 ms for 6 rows and 34.5 ms
for 24; a complete particle decode with 6 particles took 43.1 ms per position, of which about 31 ms was
host-side Python (sampling, constraints, exact features) (`docs/CHANGES.md` §19.3). On the CPU, the
official replies to the 200 test prompts (252 replies) took 2,110 s on 40 workers of one host core each,
export and cache calibration included: a median of 207 s and 156 new tokens per reply, and 0.567 s per
decoding step of six particles on one core (Appendix M, Table M12; `testgen.json`).

**E.6 The development studies in full.** Before the test split was opened, mid-run checkpoints were compared
blind with both opponents on the first 60 development prompts (Section 7.9 summarises them). Table E2 gives
every reader; Table E3 the exact scores of the same replies and bits per byte.

**Table E2. Blind comparisons on development prompts, by reader** (W / L / T from the first-named model's side, T including "both
bad" and pairs whose two orders disagree; score (wins + ties/2)/n with 95% Wilson interval; exact
two-sided sign test; orders: pairs on which a reader's two orders agree; first: mean preference for the
first-shown reply; `lead3_work/readers.py` on `study/proxy/ratings/`, which reproduces every report in
`study/proxy/reports/ab/`; ours with default decoding; first 60 `dev` prompts; p: progress when the
checkpoint was written, the trainer's stop event).

| pair set | reader | W / L / T | score [95%] | sign test p | orders; first |
|---|---|---|---|---|---|
| `final` 25,280 (p = 0.44) vs Falcon-H1-Tiny | Opus, one reading | 2 / 43 / 15 | 0.158 [0.087, 0.271] | < 0.0001 | – |
| | `qwen3-4b-2507` (registered) | 2 / 16 / 42 | 0.383 [0.271, 0.510] | 0.0013 | 63.3%; −0.096 |
| | `gemma4-e4b` (registered second) | 0 / 4 / 56 | 0.467 [0.346, 0.591] | 0.13 | 80.0%; −0.087 |
| `final` 25,280 vs MobileLLM-R1.5 | Opus, one reading | 15 / 8 / 37 | 0.558 [0.433, 0.677] | 0.21 | – |
| | `qwen3-4b-2507` | 7 / 4 / 49 | 0.525 [0.401, 0.646] | 0.55 | 78.3%; −0.000 |
| | `gemma4-e4b` | 2 / 0 / 58 | 0.517 [0.393, 0.638] | 0.50 | 91.7%; −0.008 |
| `v9` 31,912 (p = 0.56) vs Falcon-H1-Tiny | Opus, two readings | 3 / 39 / 18 | 0.200 [0.118, 0.318] | < 0.0001 | 93.3%; +0.000 |
| | `qwen3-4b-2507` | 1 / 16 / 43 | 0.375 [0.264, 0.502] | 0.0003 | 60.0%; −0.114 |
| | `gemma4-e4b` | 0 / 3 / 57 | 0.475 [0.354, 0.599] | 0.25 | 83.3%; −0.065 |
| `v9` 31,912 vs MobileLLM-R1.5 | Opus, two readings | 20 / 7 / 33 | 0.608 [0.482, 0.722] | 0.019 | 98.3%; +0.008 |
| | `qwen3-4b-2507` | 7 / 2 / 51 | 0.542 [0.417, 0.661] | 0.18 | 80.0%; −0.000 |
| | `gemma4-e4b` | 2 / 1 / 57 | 0.508 [0.385, 0.630] | 1.0 | 93.3%; +0.008 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5, 200 `test` prompts | Opus, two readings | 144 / 6 / 50 | 0.845 [0.788, 0.889] | < 0.0001 | 93.5%; +0.013 |

**Three readers.** Against Falcon-H1-Tiny every reader that took sides took Falcon-H1-Tiny's in almost
every pair: the Opus readers in 39 of 42 decisive pairs (`v9`) and 43 of 45 (`final`), the registered judge
in 16 of 17 and 16 of 18, the second judge in all it decided. Where the Opus readers and the registered
judge both chose a side (`v9`), they chose the same one in 14 of 16 pairs against Falcon-H1-Tiny and 6 of 6
against MobileLLM-R1.5. The readers differ in how much they decide: the registered judge's two orders agreed on only 60–63% of
the Falcon-H1-Tiny pairs and it preferred the second-shown reply, so most pairs the Opus readers decided
were ties for it (26 of 42 against Falcon-H1-Tiny, 21 of 27 against MobileLLM-R1.5; Cohen's κ over win,
loss and tie 0.21 and 0.18); the second judge gave no side (C, "equal or equally poor", or orders that
disagree) in 93–97% of the pairs and has almost no resolution here. Against MobileLLM-R1.5 the Opus
readers found a lead for `v9` (20 / 7, p = 0.019), mostly on instruction-following (9 / 0 / 3) and
two-turn prompts (7 / 0 / 8), with every decisive maths pair lost (0 / 5 / 4); the registered judge found
7 / 2 (p = 0.18). Both automatic judges cut replies over 4,000 characters, in 30 of the 60 MobileLLM-R1.5
pairs. Agreement between orders is not validity: no human has read these pairs.

**A scale.** On the 200 test prompts, read the same way, Falcon-H1-Tiny beat MobileLLM-R1.5 in 144 of 150
decisive pairs, in every category (chat 37 / 1, code 17 / 2, instruct 31 / 0, maths 15 / 3, writing 44 /
0). This set holds no reply of ours and does not open the test split for our model. MobileLLM-R1.5, a
reasoning model run with its publisher's sampling and budget forcing, is weak on these prompts: on `dev`
42 of its 75 replies ran into the token limit (Table E3), and the readers' notes on the reference pairs
describe its reply as looping or repeating in 87 and 83 of 200 (two readings) and mention a boxed answer
in 46 and 49, often nonsense or looping (in 32 and 40 of these the same clause also calls it looping,
repeated, nonsense, degenerate, fragmentary or incoherent; keyword counts, `lead3_work/ref_notes.py`,
`lead4_work/scripts/ref_notes4.py`). A lead over MobileLLM-R1.5 here is a low bar; the informative
comparison is with Falcon-H1-Tiny, where our model, at step 31,912, is far behind. The development and
test prompts are two halves of one set, so the scale is indicative, not a calibration.

**In absolute terms.** In the first reading of the pair set in which both replies were ours (`v9` at step
31,912, the default decoding against 3-gram blocking), both replies were judged bad in 44 of 60 pairs;
the readers' notes on those 44 name wrong facts or arithmetic in 20 and incoherent text in 14 (keyword
count, `lead3_work/ours_notes.py`; for instance "Both wrong; correct is 12,600. A says 30,000, B says 12.").
Against MobileLLM-R1.5, 31 of 60 pairs were judged "both bad", against Falcon-H1-Tiny 14 of 60; for
`final` at step 25,280, 46 to 55 of 60 pairs per decoding comparison (Table E1). At p = 0.56, most of the
model's replies to these prompts were judged bad, mainly for wrong content and incoherence.

**Table E3. Exact scores and bits per byte on development prompts** (the same 60 `dev` prompts, 75 turns,
default decoding; `lead2_work/exact_dev.json`, `lead2_work/opp_bpb_web.json`, `lead3_work/heldout46.json`;
the official model's column: the run's own decoding-selection replies at the end, prefopt weights of step
51,812, `lead5_work/exact_dev_final.json`, and its bits per byte from `lead5_work/heldout5.json`).

| measure | ours, `final` 25,280 | ours, `v9` 31,912 | ours, official (51,812, after DPO) | Falcon-H1-Tiny | MobileLLM-R1.5 (`neutral` arm) |
|---|---|---|---|---|---|
| maths answers correct (9 prompts) | 0 | 1 | 0 | 4 | 4 (1) |
| constrained prompts with every constraint met (14) | 12 | 12 | 13 | 9 | 2 (1) |
| replies cut at the token limit (75) | 2 | 0 | 1 | 0 | 42 (54) |
| stock refusals / deflections (75) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| median words per reply | 93 | 96 | 98 | 188 | 532 (698) |
| bits per UTF-8 byte, 64 held-out web rows (our training distribution) | 1.017 (step 29,722) | 0.998 (35,294); 0.980 (46,178) | 0.978 (0.974 before DPO) | 1.570 | 1.500 |

The exact scores agree with the readers: both opponents answer more maths prompts correctly, and our
replies meet constraints more often and are shorter than the opponent's in 38 to 45 of the 60 pairs. On
held-out web text from our own training distribution (ClimbMix, 62% of `final`'s training tokens), our
model needs 0.98 bits per byte at step 46,178 and the opponents 1.57 and 1.50 (the same rows: 258,814
bytes as the opponents' tokenizers cut them, 258,817 as ours does);
this favours us by construction (instruction-tuned opponents read on raw web text, training data
unknown), and next-token loss on the pre-training distribution said little about preference in chat
here. Two decoding studies on the same prompts could not separate any setting from the default
(Table E1).

## Appendix F. On-chip measurements in full

**F.1 The rewrites of the dense core.** One 768 × 768 product took 0.08 ms forward and 0.23 ms with its
recomputation and backward pass (`docs/TPU_NOTES.md`).

**Table F1. One `value_and_grad` of a 32 × 1,024 micro-batch, `configs/final.py`**
(`docs/TPU_NOTES.md`; loss 8.33353 for every variant).

| implementation | ms | tokens/s |
|---|---|---|
| reference | 289.3 | 113,264 |
| rotation as two products; reserved dimensions as zero columns and broadcast constants | 215.0 | 152,379 |
| … operands built in the kernel's layout, `W_o` as one contraction over (head, dimension) | 194.9 | 168,155 |
| … feed-forward pre-activations stored in bfloat16 | 181.8 | 180,248 |
| … lanes carried as a tuple | 179.3 | 182,779 |
| … kernel tiles 1,024 / 1,024 / 512 with the fused backward kernel | 162.0 | 202,245 |

On whole training steps the new code ran at 201,877 tokens/s against 111,893 for the old (1.299 s
against 2.343 s per step; start-up benchmarks). The suspects measured and cleared first, one at a time
on the idle chip with a 32 × 1,024 micro-batch (`docs/TPU_NOTES.md`): a fixed cost per micro-batch
(larger micro-batches were slower); the sequence length (the quadratic part is about 6%);
recomputation (4.5%); the float32 vocabulary head (5.8 ms of 289); the optimiser (1.5 ms of a 2.34 s
step); gradient accumulation (the scan of 8 micro-batches is 8.0 times one); the tape, the registers,
the value embeddings and the sink (each within 5% when removed, 10% for the sink); dense attention
instead of the flash kernel (303 ms against 289).

**The rewrites.** The rotation is written as `rotate(xW) = (xW)·(cos, cos) + (xW′)·(sin, −sin)`, with
W′ the weight whose two rotary halves are exchanged per head, so the rotated operands come out of
products instead of slices; the reserved dimensions become zero columns and broadcast constants;
operands are built in the kernel's layout and `W_o` is one contraction over (head, dimension); lanes are
carried as a tuple; kernel tiles are 1,024 / 1,024 / 512 with the fused backward kernel. Storing the
feed-forward pre-activation in bfloat16 adds one unbiased rounding (2⁻⁹ relative) before the square and
saves 28% of the feed-forward sub-layer (2.97 → 2.13 ms per application). Everything else equals the
reference to rounding in float32 and bfloat16 (28 tests, `tests/model/test_attention_major.py`).

**The one-tile kernel.** At pieces of 4 × 1,024 the splash kernel sees one or two 1,024-key tiles per
(row, head) slice but pays for a general block-sparse kernel: an online softmax, a causal mask computed
over whole tiles of which half are discarded, running maximum, sum and output written to device memory,
and a log-sum-exp written 128 lanes wide. The one-tile kernels (`bes/model/attention_onetile.py`) hold
the slice in on-chip memory, cover only the causal triangle in strips of 256 query rows, mask the
off-diagonal pieces by segment only, take each strip's softmax exactly in one pass and write the
log-sum-exp as one lane-dense row; the backward pass runs key-major strips. The flag `save` keeps the
forward kernel's output and log-sum-exp through the loop's recomputation, so the forward kernel runs
once per block application instead of twice. A hand-written backward pass of relu(h)² removes a bit
mask the compiler had packed into the recomputed product; it is bit-identical on the CPU.

**Table F2. One `value_and_grad` of one piece on the chip, `configs/final.py`, median of 30**
(`wf/chip_kernels_0745.txt`; equality against the configured kernel at freshly initialised parameters
on a synthetic batch: loss difference / worst relative gradient difference over parameter leaves,
maximum / L2; `tables.md` T3a).

| kernels and switches | 4 × 1,024 (ms) | speed-up | Δloss / max / L2 | 2 × 2,048 (ms) | speed-up | Δloss / max / L2 |
|---|---|---|---|---|---|---|
| configured splash kernels | 17.745 | – | – | 19.845 | – | – |
| splash + `io` | 17.177 | 1.033 | 0 / 0 / 0 | 19.762 | 1.004 | 0 / 0 / 0 |
| splash + `relu2` + `ce` | 16.974 | 1.045 | 0 / 1.1e−2 / 1.0e−2 | 19.512 | 1.017 | 0 / 3.9e−2 / 3.4e−2 |
| splash + `save` | 16.800 | 1.056 | 0 / 1.4e−2 / 1.2e−2 | 17.920 | 1.107 | 0 / 4.7e−2 / 3.3e−2 |
| `onetile` | 15.268 | 1.162 | 0 / 2.8e−2 / 2.3e−2 | 17.605 | 1.127 | −9.5e−7 / 5.1e−2 / 3.9e−2 |
| `onetile:save` | 15.130 | 1.173 | 0 / 1.4e−2 / 1.5e−2 | 16.498 | 1.203 | −9.5e−7 / 7.5e−2 / 5.4e−2 |
| **`onetile:save` + `relu2` (adopted)** | **14.639** | **1.212** | 0 / 2.8e−2 / 2.6e−2 | **16.275** | **1.219** | −9.5e−7 / 7.9e−2 / 5.7e−2 |
| `onetile:save` + `relu2` + `ce` | 14.816 | 1.198 | 0 / 1.2e−2 / 1.6e−2 | 16.176 | 1.227 | −9.5e−7 / 4.8e−2 / 4.0e−2 |

In executed bundles of the chip's compiler, `onetile:save` needs 0.34 of the configured kernels' work
at 1,024 tokens and 0.42 at 2,048 (`docs/TPU_NOTES.md`). The independent review of the kernel had not
returned when it went live; it later changed the switch's handling of long sequences and typos, not
the kernel (`wf/kernel.md`).

**F.2 Equality of the adopted kernel on the chip.** On the chip, with freshly initialised parameters and a synthetic batch,
the adopted kernel's loss differed from the configured kernel's by at most 9.5 × 10⁻⁷; its gradients
differed, for the worst parameter leaf, by 2.8 × 10⁻² (maximum) and 2.6 × 10⁻² (L2) relative at 1,024
tokens and by 7.9 × 10⁻² and 5.7 × 10⁻² at 2,048. Changes that leave the attention's arithmetic alone
also move gradients on the chip (1.1–1.4 × 10⁻² at 1,024, 3.9–4.7 × 10⁻² at 2,048, for a changed
recomputation policy and for backward passes that are bit-identical on the CPU), but the adopted
kernel's differences are about twice that band, so the band does not establish its correctness. It was
accepted on CPU interpret-mode tests (within 1e−5 of the configured kernels in float32, against a
float64 dense reference, with segment boundaries on and next to every strip boundary) and on the live
loss staying in its band. A distance of each variant to a float32 reference on the chip would settle
it; it was not measured.

![Figure F1](figures/F3_kernels.svg)

**Figure F1.** One value-and-gradient of a piece of 4,096 tokens (left 4 × 1,024, right 2 × 2,048) of
`configs/final.py` on the chip, eight kernel configurations measured in one pause (24 November
07:44–07:52; `wf/chip_kernels_0745.txt`). Dots: median of 30 timings. The 30 timings of a dot span
0.09–0.47 ms from minimum to maximum, less than the width of a dot, so the ranges are not drawn; they
overlap for the adopted configuration and the same with `ce` (both lengths) and, at 2,048 tokens, for
one-tile + save against both of these, for splash + save against one-tile and for the configured splash
kernel against splash + io: those orderings are not resolved by the measurement. Vertical line: median
of the configured splash kernel; right: speed-up against it. Blue, ringed: the configuration adopted at
step 22,613. `save`: forward kernel outputs kept through recomputation; `relu2`, `ce`: hand-written
backward passes of ReLU² and of the cross-entropy head; `io`: splash accumulators in on-chip memory.

**F.3 Whole steps and the `v9` switches.** `flat` also accumulates the table gradients in the `[V, W]`
shape; `rows` also skips one recomputation of a normalisation; `tokn`'s float32 sums run in another order.
The cost model attributes part of `v9`'s extra cycles to relayouts of the table gradients and almost
nothing to the attention gain table (`wf/v9_speed.md`). On runtime 1 `docs/TPU_NOTES.md` summarises the
first rates as `final` 112–117 thousand and `v9` 88–92 thousand tokens/s; on disk are the start-up
benchmarks 111,893 (`final`) and 88,146 (`v9`) and `final`'s two monitor readings 115,394 and 116,718.

**Table F3. Whole accumulation steps of 262,144 tokens on a synthetic batch** (`tables.md` T3b;
`final` rows of 07:44 timed with `ce` on; micro-batch rows of 03:09 with the splash kernel).

| measured (UTC, 24 Nov) | configuration | table gradient | kernel and switches | pieces × rows × tokens | ms | tokens/s |
|---|---|---|---|---|---|---|
| 03:09–03:15 | `final` | `take` | splash | 16 × 16 × 1,024 | 1,173.7 | 223,348 |
| 03:09–03:15 | `final` | `take` | splash | 32 × 8 × 1,024 | 1,081.5 | 242,393 |
| 03:09–03:15 | `final` | `take` | splash | 64 × 4 × 1,024 | 1,063.3 | 246,538 |
| 03:09–03:15 | `final` | `take` | splash | 16 × 8 × 2,048 | 1,348.8 | 194,351 |
| 03:09–03:15 | `final` | `take` | splash | 32 × 4 × 2,048 | 1,284.3 | 204,121 |
| 03:09–03:15 | `final` | `take` | splash | 64 × 2 × 2,048 | 1,218.2 | 215,192 |
| 07:44–07:52 | `final` | `take` | `onetile:save+relu2+ce` | 64 × 4 × 1,024 | 884.5 | 296,371 |
| 07:44–07:52 | `final` | `take` | `onetile:save+relu2+ce` | 128 × 2 × 1,024 | 958.9 | 273,378 |
| 07:44–07:52 | `final` | `take` | `onetile:save+relu2+ce` | 64 × 2 × 2,048 | 983.9 | 266,432 |
| 07:44–07:52 | `final` | `take` | `onetile:save+relu2+ce` | 128 × 1 × 2,048 | 1,013.1 | 258,767 |
| 09:45–09:58 | `v9` | `take` | `onetile:save+relu2` | 32 × 8 × 1,024 | 1,491.5 | 175,764 |
| 09:45–09:58 | `v9` | `take` | `onetile:save+relu2` | 64 × 4 × 1,024 | 1,433.1 | 182,919 |
| 09:45–09:58 | `v9` | `take` | `onetile:save+relu2` | 32 × 4 × 2,048 | 1,551.6 | 168,949 |
| 09:45–09:58 | `v9` | `take` | `onetile:save+relu2` | 64 × 2 × 2,048 | 1,510.2 | 173,584 |
| 09:45–09:58 | `v9` | `sorted` | `onetile:save+relu2` | 64 × 4 × 1,024 | 1,789.0 | 146,529 |
| 09:45–09:58 | `v9` | `sorted` | `onetile:save+relu2` | 64 × 2 × 2,048 | 1,856.5 | 141,205 |
| 09:45–09:58 | `v9` | `onehot` | `onetile:save+relu2` | 64 × 4 × 1,024 | 1,484.8 | 176,556 |
| 10:53–10:56 | `v9` | `take` | … + `flat` | 64 × 4 × 1,024 | 1,406.4 | 186,393 |
| 10:53–10:56 | `v9` | `take` | … + `flat` + `rows` | 64 × 4 × 1,024 | 1,295.2 | 202,403 |
| 10:53–10:56 | `v9` | `take` | … + `flat` + `rows` + `tokn` | 64 × 4 × 1,024 | 1,287.1 | 203,665 |

On the chip the three `v9` switches gave a loss difference of 0 against `flat` alone and gradient
differences of 9.5 × 10⁻³ (maximum) and 1.0 × 10⁻² (L2), the level a change of the compiled graph gives.
One `value_and_grad` of a `v9` piece (median of 6): 22.79 ms (`take`), 29.07 (`sorted`), 23.38
(`onehot`), 22.59 (+ `flat`), 20.44 (+ `rows`), 20.01 (+ `tokn`) at 4 × 1,024; 23.48 (`take`) and 29.65
(`sorted`) at 2 × 2,048. A compile-only reading of the program at 16 × 1,024 shows the compiler keeping
both lanes in on-chip memory for the whole forward pass, which is consistent with the gain of the
pieces but not itself a measurement (`wf/speed_loops.md`).

![Figure F2](figures/F9_waterfall.svg)

**Figure F2.** Training throughput by cause (Table 2), in the order the changes happened. Upper part,
runtime 1 (32-row micro-batches, training runs discarded): start-up benchmarks (hollow bars) and single
log lines of training (dots; the reference row's two dots are the two monitor readings, 116,718 and
115,394 tokens/s). Lower part, run main16: each bar spans the live rate (Section 2.2) before and after one
change, with the window of step records printed under the stage (n, first and last step, UTC); records
to step 51,800 (24 November, 18:33:30 UTC), the last step record, the last stage being the assistant-only
tail from p = 0.92. Diamonds: Table 2's synthetic or benchmark value of the same stage. A bar is a
before-and-after difference of medians, not the effect of one change alone: the step from runtime 2 to
runtime 3 changed no code, and the graft changed the configuration.

**F.4 What did not help.** Thirteen changes were measured on the chip and not adopted (Table F4). Two
were faster by less than the adoption bar or the spread (one loss chunk per micro-batch, 1.2%; the
hand-written cross-entropy backward, 0.6% at 2,048 tokens and 1.2% slower at 1,024), and two
contradicted our own written expectation (keeping the kernel's outputs through recomputation; raising
the compiler's scoped on-chip memory limit).

**Table F4. Measured on the chip and not adopted** (sources in `CLAIMS.md`).

| change | measurement | why it was expected to help |
|---|---|---|
| larger micro-batches (64, 128 rows; reference code) | 110,855 and 104,576 against 111,893 tokens/s | amortise a fixed cost per micro-batch |
| no recomputation (half micro-batch) | 159–167 thousand against 202 thousand tokens/s after the rewrite; 188,907 against 216,681 in the run's benchmark | skip the recomputed forward |
| keep the attention kernel's outputs through recomputation (splash) | 161.2 against 158.3 ms | skip the second forward kernel |
| one loss chunk per micro-batch | 1.2% faster, below the 1.5% adoption bar | fewer passes over the logits |
| one 2,048-key tile at 2,048 tokens | 177.5 against 176.1 ms | fewer tiles |
| one product per head, stacked on the head axis | 274.6 ms per micro-batch | avoid a contraction |
| feed-forward entirely in bfloat16 | 184.5 against 181.8 ms | halve the bytes |
| kernel tiles of 256 | 2.2 times slower than 512 for the kernel alone | finer causal skipping |
| scoped on-chip memory limit 8 / 16 / 64 / 96 MiB | 1,295.9 / 1,159.1 / 1,103.7 / 1,322.2 ms against 1,062.5 ms (default) | more room for the compiler's buffers |
| pieces of 2 rows at 1,024 / 1 row at 2,048 with the new kernel | 958.9 / 1,013.1 against 884.5 / 983.9 ms | the trend of Table F3 |
| hand-written cross-entropy backward | 1.2% slower at 1,024, 0.6% faster at 2,048 per piece (within the spread) | one pass less over the float32 logits |
| `v9`: the scatter-free `sorted` table gradient | 1,789.0 against 1,433.1 ms (`take`) per step | 16 times fewer multiply-adds than one-hot |
| `v9`: pieces of 8 rows | 1,491.5 against 1,433.1 ms | fewer per-piece table costs |

**F.5 The compile-only method.** Programs: `bes.train.train.accumulate` over 2 pieces (the loop body is
one piece of the trainer's step, accumulation included), libtpu 0.0.50 for `v6e:1x1`, jax 0.11.1, on a
CPU host; each full compile took 460–500 s. The step estimate is n_pieces × cycles per piece + the
once-per-step part, converted at 0.521 ms per million cycles (calibrated at 1,024 tokens on two shapes
of the MLP-block programs) and, for `v9`, added to `final`'s measured step. Mosaic kernels count 0
cycles, so only differences are meaningful for the cycles; the byte count of Table 4 is the sum over
the instructions that run (while bodies times their trip count) of the logical bytes of their
HBM-resident operands and outputs, with a sliced fusion parameter counting its slices and an in-place
update counting the update (`wf/v9_speed_work/hlo_bytes.py`). The line through all six configurations
of Table 4 has a slope of 0.73 ms per GB, a marginal 1.37 TB/s.

**F.6 How far from the arithmetic.** Counting multiply-adds from the configuration (`lead2_work/make_numbers.py`), `final` needs 0.781 GFLOP
per training token at 1,024 tokens (forward and backward, causal attention, no recomputation): at
281,399 tokens/s, 220 TFLOP/s, 27% of the 814 TFLOP/s the chip delivered on the feed-forward products
alone. Counting what the chip executes (the loop body's products four times: forward, recomputed
forward, two backward products; the rotation's q and k as two products each; attention over the 62.5%
of the score square the strips cover, its forward kernel once, its backward 2.0–2.5 times the forward;
the head three times) gives 1.18–1.19 GFLOP per token: 332–336 TFLOP/s, 41% of 814.

**The loop count.** In the reference implementation one loop instead of eight took 42.9 against 289 ms
per micro-batch (`docs/TPU_NOTES.md`): about 35 ms per loop and 8 ms outside the loops; after the
rewrites a block application took about 8 ms and the rest of the micro-batch 34 ms, so at equal wall clock
4 loops would have trained on about 1.65 (rewritten code) to 1.95 (reference) times the tokens of 8.

**F.7 The cost model against the chip, by change.** For every change with both a compile-only
prediction and a chip measurement, Table F5 compares the two: the cost model had the sign right every time and the size wrong
by factors from 0.36 to 7.9. Its prediction of `v9`'s step, anchored on `final`'s measured step, looks
5% high; on the difference it actually predicts, it was 14% high.

**Table F5. Compiler cost model against the chip, by change** (`wf/v9_speed.md`, `wf/v9_rows.md`,
`wf/v9_tokn.md`; chip files as in Table 3 and `docs/CHANGES.md` §19.5; Figure 3b).

| change | predicted (ms per step) | measured (ms per step) | predicted / measured |
|---|---|---|---|
| `v9` instead of `final` (`take`, 4-row pieces) | +623.7 | +548.6 | 1.14 |
| `sorted` instead of `take` table gradient (`v9`) | +129.7 | +355.9 | 0.36 |
| 8-row instead of 4-row pieces (`v9`) | +49.9 | +58.4 | 0.85 |
| 8-row instead of 4-row pieces (`final`; chip with the splash kernels) | +53.7 | +18.2 | 2.95 |
| `flat` | −45.9 | −26.7 | 1.72 |
| `rows` (on `flat`) | −118.9 | −111.2 | 1.07 |
| `tokn` (predicted on `flat`, measured on `flat` + `rows`) | −63.6 | −8.1 | 7.85 |

Two further cases have only one side: keeping the attention kernel's outputs through recomputation was
measured (1.8% slower; no compile-only estimate), and carrying the next loop's input through the scan
was estimated (+4.9%) and not measured. The design document's argument for `sorted`, 16 times fewer
multiply-adds than a one-hot gradient (`docs/V9.md` §3), was right as arithmetic and wrong as a
prediction of time: the scatter-add was the fastest of the three.


## Appendix G. The graft in full

**G.1 The tool.** `scripts/graft_run.py SRC DST --config configs/v9.py --override run.seed=1 --check --cpu` (commit
`9a4d652`): read the source manifest and newest checkpoint and refuse if its configuration hash is not
the run's; build the target trainer's fresh tree (parameters and optimiser state at the target's
initialisation, with seed 1); replace every leaf whose path, shape, dtype and storage match by the
checkpoint's (parameters and optimiser slots; for MuonH the cached initial norms are part of the
slots), refuse on any mismatch, report leaves left behind (none here: the library is not a parameter);
write the checkpoint under the target's hash with the source's step and ledger; carry `run.json`, the
stage ledger, the benchmark's records, `post_plan`, `micro_split` and `fast`; write the manifest with the
configuration replaced and a `graft` record (time, source run and hash, checkpoint, step, tokens, p,
new and copied numbers); with `--check`, compare the losses (Section 6.4); rename the temporary
directory to the destination. The live switch then stopped `final`, timed `v9` on the chip, grafted on
the runtime's CPU, swapped the directories (so that `v9` kept the run path the monitoring and snapshot
scripts use) and resumed with `--resume`. A CPU rehearsal at small size ran the whole sequence end to end
(train → stop → graft → swap → resume → post-training → export), with step, p and learning rate
continuous across the graft.

An independent review ran four tests on a fabricated run (the check; every carried leaf equal; every
new leaf equal to a fresh initialisation, `page/val` and `gain/o` exactly zero; the refusals; a trainer
that restores the grafted directory and takes a finite step that moves `page/val` off zero) and found
no blocker; of its two medium risks, one (the loader's horizon) was fixed in the committed tool and the
other (a mirror that would have overwritten the source's last checkpoint) was avoided by keeping the
stopped run (`wf/graft_review.md`; tests in `tests/train/test_graft_run.py`, commit `769f240`). The tool
refuses to write if a shared leaf differs, if the destination exists, if there is no checkpoint, or if
the checkpoint was written under another configuration hash.

**G.2 The function check and the gradients.** The check of Section 6.4 at four source checkpoints:

**Table G1. The graft's function check** (synthetic batch of 2 × 256 tokens; grafted tree under the live
run's configuration hash `f29bfeb3ebf353a8`).

| source checkpoint | where | source loss | grafted loss | difference |
|---|---|---|---|---|
| step 26,890 (snapshot) | development container | 11.2808027 | 11.2808027 | 0.0 |
| step 28,501 (live run, before the switch) | runtime host CPU, next to the trainer | 11.3348608 | 11.3348608 | 0.0 |
| step 29,722 (the switch; re-run of the same source checkpoint at HEAD `8a86bcb`) | development container | 11.3726320 | 11.3726320 | 0.0 |
| step 14 (smoke-size rehearsal) | development container | 9.2843390 | 9.2843390 | 0.0 |

The row for step 29,722 is a re-run, by the reviewer, from the local copy of the stopped run, not the
live invocation: the only saved output of a graft at 29,722 ran without `--check`, into a directory of
a `sorted` variant under another hash. The tool also writes the manifest's graft record before the
check and prints the check's result without storing it; we report both as defects of the tool.

On real held-out rows (Section 6.4) the
gradients of U reached 7.4 × 10⁻³ and 9.0 × 10⁻³ (largest entry; web and chat rows) and those of G
1.0 × 10⁻³ and 1.2 × 10⁻³ (`lead2_work/graft_grad.json`). For a constant gradient, Adam with the carried
update count makes the new leaves' update 0.894 times a fresh run's at the first step, 1.44 times at the
seventh (the maximum) and 1.04 times at the fiftieth (β = 0.8, 0.95; our arithmetic).

**G.3 The data stream after the graft.** The data stream cannot be carried: the mixture changed (no library rows or tasks) and it restarted with
another seed. A stream that starts mid-run needs two corrections. The loader's horizon estimate
tokens/p − tokens was 40 times too small one block after the graft (1.9 × 10⁸ against 8.2 × 10⁹, found
by the graft's review), so small sources would have been read out at once; told the tokens before the
graft, the loader planned over 7.45 × 10⁹ remaining tokens, of which 5.79 × 10⁹ were in fact trained:
about 1.3 times too long rather than right. And every source may be read
max_epochs × (1 − p) = 3.96 more times; small sources were capped as before the graft (GSM8K's planned
share 0.0126 became 0.0007; before the graft, 0.0114 became 0.0008).


**G.4 The controlled experiment in detail.** *Design.* The model is the live model's code at width 128 (2 weight-shared blocks × 8 loops, rows of 512
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

*Checks.* Each graft arm trained its own trunk, and its evaluations before the graft equal those of
`ref` of the same seed row for row, bit for bit. At each of the nine grafts the held-out loss of every
domain, and the training loss of 8 held-out web rows, are the same before and after: the difference is
exactly 0, on the held-out domains also per row (the training-loss check stores one aggregate value).
These checks ran on the CPU in float32, as the whole experiment did.

*Sources.* (`docs/MICRO_GRAFT.md`, `wf/micro_graft.md`,
`micro_graft/6M/`; recomputed by `lead3_work/micrograft.py`). Code: graft arms in `bes/eval/microbench.py`
(`run_graft_arm`), `scripts/micro_ablate.py` (`GRAFT_ARMS`), `scripts/micro_graft_report.py` and 11 tests in
`tests/eval/test_micro_graft.py` (commit `b04478e`; results in `b1e9140`); written by one agent and
reviewed by a second before the runs (trunk equality, function preservation, carried optimiser state,
schedule and data order, unchanged hashes of the 31 E2 arms, report arithmetic). The runs used a separate
clone of the repository with the patch, on the idle host cores of the training runtime (one physical
core, two threads, per run; CPU only), 14:12:51 to 16:41:01 UTC; the main run's live rate at 2,048 tokens
was 189,054 tokens/s over 73 step records before 14:12:51, 189,077 over the 128 during these runs and 189,050
over the 23 of the restart arms' window (16:42:47–17:29:56) before the tail began at 17:09:57
(`lead5_work/scripts/rate_during_micro5.py` on the final train log). A graft is refused
before training when a base leaf is missing or reshaped in the target, when an optimiser setting or the
schedule differs, or when the graft fraction is not an evaluation fraction. The carried update count
makes the page tables' updates 1, 2, 5 and 50 after a graft 0.89, 1.15, 1.41 and 1.04 times those of a
fresh start with the same moments, as in the live graft. Training pools of 8,192 web rows (ClimbMix) and
4,096 chat rows (smol-smoltalk) of 512 tokens; the mixture draws rows 3:1, 8,784 web and 2,928 chat rows
of 11,712 (1.07 and 0.71 passes), 78.0% and 22.0% of the weighted tokens. Held-out: 32 web and 32 chat
rows from other files. The learning-rate factor decays linearly to 0.1 over the steps, as 1 − 0.9p in the
live run.

Per checkpoint (mix, mean ± sd over three seeds), `v9_pages` − `ref`: −0.0164 ± 0.0062 at 1.50 × 10⁶
tokens, −0.0329 ± 0.0091 at 3.00, −0.0494 ± 0.0100 at 4.50, −0.0519 ± 0.0107 at 6.00; at the end by
domain, web −0.0464 ± 0.0118, chat −0.0683 ± 0.0089, assistant turns −0.0689 ± 0.0107. Graft − `v9_pages`
after each graft: `graft_25` +0.0137 ± 0.0086, +0.0029 ± 0.0101, −0.0031 ± 0.0113 (3.00, 4.50, 6.00 ×
10⁶); `graft_50` +0.0333 ± 0.0085, +0.0161 ± 0.0096 (4.50, 6.00); `graft_75` +0.0396 ± 0.0103 (6.00).
Per seed at the end (mix): graft − `ref` −0.0539, −0.0551, −0.0558 (`graft_25`), −0.0386, −0.0359, −0.0327
(`graft_50`), −0.0127, −0.0120, −0.0122 (`graft_75`); `graft_50` − `v9_pages` +0.0241, +0.0054, +0.0188.
Shares recovered by domain (ratio of the mean differences; web, chat, assistant turns): `graft_25` 1.07,
1.04, 1.04; `graft_50` 0.70, 0.66, 0.67; `graft_75` 0.24, 0.24, 0.23. The function check's loss on 8
held-out web rows, seeds 0, 1, 2, identical before and after each graft: 5.0309, 5.0072, 5.0118
(`graft_25`); 4.7603, 4.7661, 4.7617 (`graft_50`); 4.6126, 4.6162, 4.6532 (`graft_75`). A graft's
difference from `ref` varies little between seeds (sd 0.0004–0.0030), because the two share their trunk;
arms that never shared a state differ by about 0.01 between seeds. The restart arms
(`v9_pages` from scratch with a budget of 3,000,000 tokens and its own schedule, seeds 0–2; 732 steps,
2,998,272 tokens; the same host, 16:42:47–17:29:56 UTC; `micro_graft/3M/results.tar.gz`, unpacked into
`lead3_work/micro3M/`) end at 4.3891, 4.3885 and 4.3917 (mix; web 4.6317 ± 0.0043, chat 3.6637 ± 0.0074);
minus `graft_50` at 6 × 10⁶ tokens, +0.2348, +0.2415, +0.2265; minus `ref` at 6 × 10⁶, +0.1962, +0.2055,
+0.1938 (4.7% of `ref`'s loss); minus `v9_pages` at 6 × 10⁶, +0.2504 ± 0.0074; minus `ref`'s own evaluation
at 3 × 10⁶ tokens (mid-schedule, learning-rate factor 0.55), −0.0844, −0.0835, −0.0836; minus `v9_pages`
at 3 × 10⁶ tokens inside its 6 × 10⁶ run, −0.0509 ± 0.0086, the advantage of a schedule that ends there,
which the comparisons above include. The restart's batches are the first half of the 6 × 10⁶ runs' batches
of the same seed (the row plan is a prefix-stable round robin, `microbench.row_kinds`): it reads the first
half of the data again, not the unseen second half; with about one pass over the pools in all this should
matter little, and it was not tested.

**G.5 Held-out loss of every evaluated checkpoint.** The rows are built by `bes.export.heldout_batches` with
64 rows per domain (`scripts/midrun_eval.py`); the export's own 16-bit check uses 16 per domain
(`heldout_rows` = 16 in `bes/export.py`). We evaluated five checkpoints again on the CPU (29,722, 34,166,
35,294, 46,178 and 51,812, the last from the final snapshot by `lead5_work/scripts/heldout_eval5.py`),
keeping the loss of every position and switching the new paths off, and the weights after preference
optimisation at 51,812; the local values reproduce the runtime's to within 5 × 10⁻⁸ on the web and chat rows
(2.3 × 10⁻⁷ on the proxy rows at 51,812; `lead5_work/heldout5.json`).

**Table G2. Held-out loss of every evaluated checkpoint** (nats per token; "web + chat" is the comparison
set of 128 rows and 125,880 weighted tokens; local evaluations reproduce the runtime's values to within
5 × 10⁻⁸ on the web and chat rows, `lead4_work/heldout4.json`, `lead5_work/heldout5.json`;
`lead2_work/heldout_analysis.json`, `lead3_work/heldout3.json`, `lead3_work/heldout46.json`; checkpoints
from 37,522 on were trained while the stream's chat share rose (from step 36,867), and from 48,342 on after
conversations began to train on assistant tokens only (step 48,190), Appendix C.1; the proxy rows, left out
of every comparison, in the last column).

| checkpoint | tokens | training minutes | web | chat | chat, assistant tokens | web + chat | evaluated | proxy rows |
|---|---|---|---|---|---|---|---|---|
| `final` 26,890 | 7.05 × 10⁹ | 494.7 | 2.7952 | 1.6300 | 1.5269 | 2.2358 | runtime | 2.6332 |
| `final` 28,501 | 7.47 × 10⁹ | 519.7 | 2.7905 | 1.6217 | 1.5191 | 2.2293 | runtime | 2.6382 |
| `final` 29,722 (the graft) | 7.79 × 10⁹ | 538.7 | 2.7886 | 1.6183 | 1.5164 | 2.2267 | runtime and local | 2.6361 |
| `v9` 30,742 | 8.06 × 10⁹ | 563.7 | 2.7739 | 1.6022 | 1.4994 | 2.2113 | runtime | 2.6112 |
| `v9` 31,762 | 8.33 × 10⁹ | 588.8 | 2.7666 | 1.5950 | 1.4918 | 2.2041 | runtime | 2.6105 |
| `v9` 31,912 | 8.37 × 10⁹ | 592.5 | 2.7662 | 1.5865 | 1.4830 | 2.1998 | runtime | 2.6187 |
| `v9` 33,038 | 8.66 × 10⁹ | 617.5 | 2.7541 | 1.5752 | 1.4713 | 2.1881 | runtime | 2.5867 |
| `v9` 34,166 | 8.96 × 10⁹ | 642.5 | 2.7437 | 1.5630 | 1.4569 | 2.1768 | runtime and local | 2.6000 |
| `v9` 35,294 | 9.25 × 10⁹ | 667.5 | 2.7344 | 1.5508 | 1.4463 | 2.1661 | runtime and local | 2.5849 |
| `v9` 36,422 | 9.55 × 10⁹ | 692.6 | 2.7254 | 1.5413 | 1.4355 | 2.1569 | runtime | 2.5430 |
| `v9` 37,522 | 9.84 × 10⁹ | 717.6 | 2.7137 | 1.5225 | 1.4167 | 2.1418 | runtime | 2.5700 |
| `v9` 38,604 | 1.012 × 10¹⁰ | 742.6 | 2.7066 | 1.5042 | 1.3967 | 2.1293 | runtime | 2.5742 |
| `v9` 39,686 | 1.040 × 10¹⁰ | 767.7 | 2.7035 | 1.4838 | 1.3761 | 2.1179 | runtime | 2.5473 |
| `v9` 40,768 | 1.069 × 10¹⁰ | 792.7 | 2.6997 | 1.4627 | 1.3546 | 2.1058 | runtime | 2.5258 |
| `v9` 41,850 | 1.097 × 10¹⁰ | 817.7 | 2.6977 | 1.4443 | 1.3339 | 2.0959 | runtime | 2.5157 |
| `v9` 42,932 | 1.125 × 10¹⁰ | 842.8 | 2.6922 | 1.4230 | 1.3111 | 2.0828 | runtime | 2.5151 |
| `v9` 44,014 | 1.154 × 10¹⁰ | 867.8 | 2.6910 | 1.4112 | 1.2965 | 2.0765 | runtime | 2.4841 |
| `v9` 45,096 | 1.182 × 10¹⁰ | 892.8 | 2.6857 | 1.3942 | 1.2795 | 2.0656 | runtime | 2.4868 |
| `v9` 46,178 | 1.211 × 10¹⁰ | 917.9 | 2.6867 | 1.3790 | 1.2651 | 2.0588 | runtime and local | 2.4614 |
| `v9` 47,260 | 1.239 × 10¹⁰ | 942.9 | 2.6847 | 1.3630 | 1.2475 | 2.0501 | runtime | 2.4683 |
| `v9` 48,342 | 1.267 × 10¹⁰ | 967.9 | 2.6833 | 1.3612 | 1.2319 | 2.0485 | runtime | 2.6651 |
| `v9` 49,424 | 1.296 × 10¹⁰ | 992.9 | 2.6815 | 1.3671 | 1.2148 | 2.0504 | runtime | 3.0764 |
| `v9` 50,506 | 1.324 × 10¹⁰ | 1018.0 | 2.6762 | 1.3684 | 1.2071 | 2.0483 | runtime | 3.1946 |
| `v9` 51,588 | 1.352 × 10¹⁰ | 1043.0 | 2.6719 | 1.3638 | 1.1981 | 2.0438 | runtime | 3.2241 |
| `v9` 51,812 (the end of training) | 1.358 × 10¹⁰ | 1048.2 | 2.6690 | 1.3668 | 1.1983 | 2.0438 | runtime and local | 3.2580 |
| `v9` 34,166, pages off (token neurons and gains) |  |  | 2.7804 | 1.6218 | 1.5162 | 2.2241 | local |  |
| `v9` 34,166, token neurons off |  |  | 2.7597 | 1.5848 | 1.4780 | 2.1956 | local |  |
| `v9` 34,166, gains off |  |  | 2.7616 | 1.5944 | 1.4900 | 2.2012 | local |  |
| `v9` 35,294, pages off |  |  | 2.7849 | 1.6301 | 1.5248 | 2.2305 | local |  |
| `v9` 46,178, pages off |  |  | 2.8508 | 1.6523 | 1.5386 | 2.2754 | local | 2.6920 |
| `v9` 51,812, pages off | | | 2.8627 | 1.7051 | 1.5446 | 2.3069 | local | 3.5195 |
| `v9` 51,812 after preference optimisation (the official weights) | | | 2.6804 | 1.3803 | 1.2095 | 2.0562 | local | 3.2842 |

**Table G3. Comparisons at steps 34,166, 35,294 and 46,178** (web + chat; bootstrap over rows;
`lead2_work/heldout_analysis.json`, `lead3_work/heldout46.json`, `lead4_work/heldout4.json`; the lines
recomputed by `lead5_work/scripts/heldout5.py`).

| comparison | difference (nats) | 95% interval |
|---|---|---|
| `v9` 34,166 − `final` 29,722 (same rows, 103.8 more training minutes) | −0.0499 | [−0.0541, −0.0457] (bootstrap over rows) |
| `v9` 34,166 − `final`'s line at equal training time (line in minutes / in log tokens) | −0.028 / −0.031 | ±0.055 / ±0.045 (prediction, 1 degree of freedom) |
| pages off − on; token neurons only; gains only | +0.0473; +0.0188; +0.0244 | [0.0436, 0.0511]; [0.0168, 0.0209]; [0.0221, 0.0267] |
| `v9` with pages off − `final` 29,722 | −0.0026 | [−0.0064, 0.0014] |
| `v9` 35,294 − `final` 29,722 (128.8 more training minutes) | −0.0605 | [−0.0653, −0.0557] |
| `v9` 35,294 − `final`'s line at equal training time (minutes / log tokens); at equal tokens | −0.033 / −0.038; −0.044 | ±0.066 / ±0.052, both inside; ±0.040, outside |
| pages off − on, at 35,294; `v9` with pages off − `final` 29,722 | +0.0643; +0.0038 | [0.0594, 0.0695]; [−0.0009, 0.0086] |
| `v9` 46,178 − `final` 29,722 (379.2 more minutes) | −0.1679 | [−0.1828, −0.1533] |
| … web rows / chat rows only | −0.1019 / −0.2393 | [−0.1106, −0.0937] / [−0.2560, −0.2235] |
| `v9` 46,178 − `final`'s line at equal training time (minutes / log tokens) | −0.088 / −0.115 | ±0.171 / ±0.107; inside / outside |
| `v9` 46,178 − `final`'s line at equal tokens | −0.127 | ±0.084; outside |
| pages off − on, at 46,178; `v9` with pages off − `final` 29,722 | +0.2166; +0.0487 | [0.2019, 0.2320]; [0.0394, 0.0582] |

At 34,166 the pages carried 0.066 nats on the rarest quarter of the held-out positions (1–16 occurrences
of the input token) against 0.032 on the most frequent quarter (497–4,425), and the improvement since
the graft was the same in every quarter (−0.049 to −0.051). At 46,178 the pages carry 0.299, 0.241, 0.170
and 0.159 nats on the four quarters, from the rarest, and the improvement since the graft is −0.152,
−0.153, −0.167 and −0.199; with the pages off at 46,178 the network is worse than at the graft point by
0.0623 nats on the web rows and 0.0340 on the chat rows (`lead3_work/heldout46.json`). At 51,812 the pages
carry 0.356, 0.295, 0.213 and 0.193 nats on the four quarters, the improvement since the graft is −0.171,
−0.183, −0.140 and −0.236, and with the pages off the network is worse than at the graft point by 0.074 nats
on the web rows and 0.087 on the chat rows (`lead5_work/heldout5.json`).

## Appendix H. Negative results

**Table H1. What was tried and did not work, or did not resolve** (sources in `CLAIMS.md`).

| idea | what was measured | consequence |
|---|---|---|
| exact Gaussian-process attention ("Bayesian attention") | the linear algebra is exact (errors 1e−15) and robust to repeated decoys, but against softmax with a sink both are within 2e−4 nats of the Bayesian predictor on Dirichlet–Markov data; on real text its shrinkage form beats a fixed weight by only 0.003–0.008 nats; at temperature 0.6 it loops in 63.6% of samples against 5.0%; estimated (not measured on the chip) to cost 4.5–9 times a softmax head | not adopted; the sink and the counts of distinct successors remain |
| scale-recursive loops | the allocation law holds; the gain is 0.1–3% of the reducible loss | not adopted |
| coarse-to-fine output head | equivalent to SVD-softmax; +0.035 to +0.07 nats on the head | not adopted |
| confidence value channel; structural rotary clocks | +0.7% and +0.3% when removed, under the 1% bar | off |
| countdown registers as a step function | counting 79% with them against 93% without at 2 million tokens | gate failed; kept anyway (deviation) |
| fence-closing probe | 100% with and without the mechanism | the probe cannot discriminate |
| the first head-wise attention gate (logit x · w) | saturated within 244 steps; +0.80% loss (+1.22% on chat); right function and gradient, so no test or reviewer saw it | rescaled by 1/d; then neutral (−0.02%) and not shipped |
| decay 0.1 on the page tables | argued to hold rare tokens' rows short of their optimum; not resolvable in 244 steps | no decay |
| token-specific queries or keys | +0.046 nats in the laboratory | not adopted |
| per-token gains on feed-forward neurons | −0.035 nats at four times the storage of the output gain | off |
| hashed n-gram input tables | about 0.007 nats at this budget by the published slope | not implemented |
| the written library after its precision fix | coverage on NQ-open validation 35.0% → 6.0%; answers mainly through stored near-duplicates; the gate opens on 8 of 200 dev first turns | not in the shipped configuration |
| the plan's throughput | 300–450 thousand tokens/s assumed; 111,893 (start-up benchmark) and 115–117 thousand (two monitor readings) measured | planned for 16–25 × 10⁹ tokens; the run trained 1.358 × 10¹⁰ |
| the pre-registered `v9`-against-`final` comparison | not run (Appendix C.5) | `final` trained; `v9` adopted later at the user's request |
| the critic | its gate (within-prompt rank correlation ≥ 0.3) was never measurable; its stage was given 0 minutes | off |
| the learning-rate transfer rule at micro scale | +14.4% loss when applied at 4,096 tokens per step | not applied in the micro ablations; applied at full scale by extrapolation |
| on-chip speed candidates | Table F4 (13 changes measured and not adopted) | not adopted |
| the cost model as a predictor of differences | sign right in 7 of 7 changes, size off by factors 0.36–7.9 (Table F5) | used to rank, never as a result |

## Appendix I. Operations

A human user set the goals and took the decisions marked as theirs (stopping runtime 1, the switch to
`v9`, the instruction to use parallel agent workflows); the design, the code, the measurements and this
draft were produced by language-model agents working under the user's direction, and every change is
dated in `docs/CHANGES.md`. Two operational losses are recorded there (§19.4). Agent turns that blocked
on an approval card or let the development container sleep cost about 4 h 50 min of calendar time in
which no runtime was requested (not TPU time); the operator now keeps a tracked background task alive
and avoids tools that wait for an approval while the user is away. The loss of runtime 2 about
2 h 12 min after its allocation is unexplained: a test of the idle-timeout explanation on CPU
runtimes did not reproduce it.
At 12:10 on 24 November the development container's disk filled (about 20 GB written by the
verification agents of a research workflow); to free it, old snapshots were deleted, among them the
local copies of `final`'s checkpoints of steps 5,075, 22,613 and 26,890 that Section 6.5 could
otherwise have evaluated. The run was not affected. At the end of the run the operating agent's session
worker was restarted twice, 19:08–19:30 and 19:32–20:13 UTC, each time when a tool call waited for an
approval that nobody present could give; the container's local loops (the watch loop, the presence keepers
and the TPU-budget guard) were down for those 22 and 41 minutes and were restarted by hand, and the Opus
readings of the test pairs could start only after the TPU was given back. The run on the runtime was not
affected: its stages and the test-reply generations ran on the runtime itself. The registered judge's pass
over the test pairs moved between machines: it ran on the TPU host's idle cores until 20:40 (113 of the
Falcon-H1-Tiny pairs, 9.8 s per pair at 16 threads), then on a CPU runtime (63.8 s per pair with 4 threads
beside a second stream, 137.6 s with 8 threads alone), and was restarted at 21:22 as two 4-thread streams,
one per pair set (since then 39.9 s per pair on the Falcon-H1-Tiny set, pairs 143–155, and 177.7 s on the
MobileLLM-R1.5 set, pairs 90–92, whose replies are long); a fourth CPU runtime was refused by the provider
("too many active sessions"), and at 21:36 a third runtime was taken from MobileLLM-R1.5's control-arm
generation for two more judge streams on that set. The TPU
runtime was given back by hand at 20:41:04 (`colab_cli.py send release`, `down --yes`), closing the ledger at
86,054 s. None of this is evidence for a claim of the paper; it is reported because it shaped what could be
measured and when.

## Appendix J. Reproducibility

- **The released model.** `release/main16-v9/` of the project repository holds the run's official export
  as 12 parts of at most 16 MB (`official_export.tar.000`–`.011`, 196,956,160 bytes), committed in
  `99ba0e7`…`994b3a1` (24 November, 20:22:37–20:23:25 UTC), with `parts.json` (the SHA-256 of every part and
  of the whole tar, `f8278440…`) and a README giving the commands below. To rebuild and verify:
  `mkdir -p runs/main16; cat release/main16-v9/official_export.tar.* | tar -x -C runs/main16; python
  scripts/export.py --verify runs/main16/final`, which checks every file against `SHA256SUMS` and repeats the
  16-bit held-out check (Python 3.13, `jax` 0.11.1 on CPU suffices). We checked that every committed part
  matches `parts.json` and equals the part fetched from the runtime, that the parts' concatenation has the
  recorded SHA-256, and that the 21 files of the copy unpacked from the committed parts match `SHA256SUMS`
  (`lead5_work/scripts/verify_release.sh`); `weights16.npz` has SHA-256 `2da7cbd2e2f1fe49…`, as in the run's
  `export.json`. To use it: `python scripts/chat.py --export runs/main16/final --device cpu`; a blind arena
  against both opponents: `python scripts/arena.py --export runs/main16/final --device cpu --prompts
  proxy:test`.
- **Code.** Branch `claude/modest-carson-yi2d9g` of the project repository. The run was created at
  commit `98598c2` (clean tree, recorded in the run manifest) and continued with code-only changes that
  leave the configuration hash unchanged: `e096dff`, `77305dc` (continuation on a new machine),
  `566517a` (refusals), `b05bfe7` (pieces), `86bcd45`, `782be85` (post-training stages and plan),
  `6bf9b6e`, `641c4b6` (speed switches, the reviewed kernel switch, test-reply generation), `4f4ef45`
  (deflections), `9a4d652` (graft), `769f240` (the `v9` switches; tests of the graft and of a `v9`
  export); documentation commits `ba0db12`, `d9f0ce4`, `1b9d1b8`, `8a86bcb`. The controlled graft
  experiment adds `b04478e` (graft arms of the micro bench), `b1e9140` (the results of its 15 runs) and
  `190e0b2` (the restart arms' records, Table 6); then `11aa314` (third draft), the release commits and
  `34f12ac` (the operator's record of the results); the repository is at `34f12ac` for this draft.
- **Configurations.** `configs/final.py` (effective hash `0a7e9b33372a07ee` with the run's overrides)
  and `configs/v9.py` (`f29bfeb3ebf353a8`). The run's command line: `scripts/run_24h.py --config
  configs/final.py --out /content/runs/main16 --budget-minutes 1279.7 --override model.table_grad=take
  --override run.bench_plans=reduced --override run.post_minutes=critic=0,buffer=20 --override
  run.post_reserve_hours=3.6667 --override optim.micro_seqs=16`; continuations `scripts/run_24h.py --out
  RUN --resume`. Per-run files outside the hash: `micro_split` = `4`; `fast` = `kernel=onetile:save
  relu2=1 ce=0` for `final` and `kernel=onetile:save relu2=1 ce=0 flat=1 rows=1 tokn=1` for `v9`;
  `post_plan` = `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=15`.
- **The test replies.** On the runtime's host, the operator's `endgame_driver.py TAG MINUTES` ran
  `scripts/gen_test_replies.py /content/runs/main16 --make-export --no-download --split test --workers 40
  --cores-per-worker 1 --nice 5 --max-rss-gb 8 --label final-TAG` (CPUs 2–21 and 24–43), which exports the
  newest weights with the run's decoding choice and generates every reply (settings hash `9a89782942`:
  6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library); TAG `trained` at 18:36, `prefopt` at
  19:31. The K = 1 arm: the same script on a CPU runtime with the export rebuilt from the committed parts
  and `--set extra.particles=1` (settings hash `f93654208f`). Falcon-H1-Tiny's control arms: `python -m
  bes.eval.proxy run --model falcon --preset recommended --split test --samples 6 --select logprob_mean`
  (`2dcb1c92a6`) and `--select decoder_select` (`d1f2a63f82`). Exact scores: `python -m bes.eval.proxy exact
  RUN --split test`.
- **The pair sets and their readings.** `gen_test_replies.py OUT --merge-into N/study/proxy`, then `python -m
  bes.eval.proxy --root N/study/proxy pairs --x OURS --y OPPONENT --name final-TAG-vs-OPPONENT --split test`
  (seed 0; the operator's `final_pairs.sh TAG`); the chunk files of the two readings by `blind_eval.py
  export --chunk 15` (the second with `--swap`). Each Opus 5.5 reader, a background agent, received two
  chunk files and the written instructions `N/final_eval/JUDGE_INSTRUCTIONS.md` (judge what the user who
  asked would prefer, in this order: correctness checked by the reader, every instruction followed,
  usefulness; length is no merit; "tie" and "both_bad" when true; read no other file, no shell, no web) and
  wrote one verdict file and one notes file per chunk; `blind_eval.py import2 --source judge-opus` combines
  the two readings (a side wins only when both choose it) and `python -m bes.eval.proxy score --pairs SET
  --source judge-opus` writes the report. The registered judge: `python -m bes.eval.proxy --root ROOT judge
  --pairs SET --judge qwen3-4b-2507 --threads N` (the operator's `judge_remote.sh`), scored with `--source
  judge-qwen3-4b-2507-3605803b`.
- **Measurements.** `scripts/tpu_kernels.py`, `scripts/tpu_micro.py`, `scripts/tpu_tune.py` and
  `docs/tpu_profile/*` on an idle chip; raw outputs of 24 November: `wf/chip_kernels_0745.txt`,
  `S/v9_meas{A,B_sorted,C_onehot,D}.txt`. Compile-only programs: `wf/v9_*_work/`.
- **This paper's own analyses.** Second pass (`N/paper/lead2_work/`): `make_numbers.py`, `predictor_test.py`
  (Table 4, Table F5), `heldout_eval.py` and `heldout_analysis.py`, `graft_grad.py` (Section 6.4),
  `exact_dev.py` and `opp_bpb.py`, `deflection_labels.py` (Appendix D.4). Third pass (`lead3_work/`):
  `make_numbers.py`, `heldout3.py`, `heldout_eval.py`, `heldout46.py`, `micrograft.py` (Table 6),
  `readers.py` (Table E2), `ref_notes.py` and `ours_notes.py`. After the check (`lead4_work/scripts/`):
  `heldout4.py`, `micro4.py`, `readers4.py`, `ref_notes4.py`, `local46178.py`, `bpb_web.py`,
  `exact_split.py` and `restrict_report.py`. Final pass (`lead5_work/scripts/`): `extract_snap.py` (the
  records of the last snapshot), `make_numbers.py` (live rates, ledger, TPU time, stages, preference
  optimisation, decoding selection, power), `final_results.py` (every test reading with the repository's key
  check, the 189-prompt restriction, raw verdicts per reading, paired comparisons, reader agreement),
  `paper_values.py` (Table 7's values rounded once), `exact_check.py` and `exact_dev_final.py` (Tables 8,
  E3), `heldout_eval5.py` and `heldout5.py` (Section 6.5, Tables 5, G2, G3), `rate_during_micro5.py`,
  `verify_release.sh`, `wordcount.py`. The figures: `figures/make_all.sh --cutoff newest --blind
  final-prefopt-vs-falcon,final-prefopt-vs-mobilellm` (`FIGURES.md`; log `lead5_work/make_all_final.log`).
- **Determinism.** The same batch gives the same loss twice, bit for bit, on the chip (the start-up
  canary); a restart after 10 steps reproduces 20 uninterrupted steps bit for bit on the CPU (tests); the
  data position is a function of the step. Bitwise reproducibility across restarts on the chip was not
  tested. What cannot be reproduced is the provider's allocation of runtimes, and the Opus readers'
  verdicts are not deterministic.
- **Tests.** 1,367 test functions in `tests/` at `b1e9140`, `190e0b2` and `34f12ac` (static count of
  `def test_`; 1,356 at `8a86bcb`).

## Appendix K. Licences, safety and conduct

ClimbMix, the largest base source, is distributed under an MIT card but derives from a corpus under
CC BY-NC 4.0; we treat it as non-commercial, so the trained weights inherit that restriction. The weights
were committed to the public repository at the end of the run (Appendix J) without a separate licence
review; the model card lists every training source with its licence and marks ClimbMix as non-permissive,
and the release's README says the model is a research artefact not fit to deploy. Several chat and preference sources were
generated by third-party models whose terms set conditions on training with their outputs
(`docs/DATA_SOURCES.md` §4). The model is not trained to refuse (Appendix D.5) and has not been evaluated
for harmful output; it should not be deployed. The development container held the user's Colab
credential only in a file readable by its owner and never in the repository.

## Appendix L. How this paper was checked

The draft was written by one agent from a claims ledger (`CLAIMS.md`: every number with its source),
reviewed by another as a hostile programme-committee member against the raw files (`REVIEW_1.md`), and
revised by a third (`RESPONSE_1.md`). The review changed or withdrew numbers in the first draft: the
gradient differences quoted for the adopted kernel belonged to another configuration; seven seen test
prompts were eleven; the held-out set had 140 rows, not 144; a graft check claimed at the live switch had
no record (it was re-run); the cost model's "5% high" was 14% on what it predicts; the executed
arithmetic was 1.18–1.19, not 1.13 GFLOP per token; the headline time ratio was stale. Earlier
audits found a gate whose logit saturated (by an ablation, not by review) and the loader's horizon
estimate after the graft (by the graft's review). We report this because it is the evidence that the
numbers here were checked against their sources, and because the checks were not complete. A third pass
(this draft) added the controlled graft experiment, the registered judges' readings and the opponents'
reference comparison, moved the live numbers to the records of 16:27 UTC, recomputed every new number
from the raw records (`lead3_work/`), and corrected the first row of Table 2, Table 4's FLOP cell, the
definition of the live rate and the description of whole-step timings (`RESPONSE_2.md`). A second check
of that pass (`REVIEW_2.md`) found that Section 6.5 called a gap "inside 95% prediction intervals" that
the log-token line and the equal-token comparison put outside, that the protocol's K = 1 result and
matched controls had been dropped without a deviation, and that several result shells could not be
filled from the files they named; this draft corrects them, with the numbers recomputed from the raw
records (`lead4_work/`; `RESPONSE_2.md`, "After the check"). The final pass (`RESPONSE_3.md`)
recomputed every result from the run's last records and the rating files (`lead5_work/`), checked the Opus
reports, the exact reports and the release against their sources, and found three statements in the
orchestrator's notes that the records do not support as written: that the constraint lead over
Falcon-H1-Tiny is "the selector's doing" (with one particle, where nothing is selected, our model meets
nearly as many constraints: it is the decoder's potentials), that the all-row held-out loss rose "after
46,178" (it fell to step 47,260 and rose from the tail at step 48,190 on), and several scores that were
printed by rounding an exact half or a 4-decimal value a second time.

## Appendix M. Results on the test split and at the end of the run

The shells of the third draft, filled. Every value is recomputed from the records named for its table by
the scripts of `lead5_work/scripts/` (this appendix is written by `make_appendix_m.py` from their outputs; no
number is typed). Conventions as in Table 7: W / L / T from our side (T includes "both bad" and pairs whose
two orders disagree), score (wins + ties/2)/n with its 95% Wilson interval, the decisive win rate wins /
(wins + losses) with its interval, exact two-sided sign test; values rounded once from full precision, exact
halves upwards. R = `N/study/proxy/reports/ab/`. Our official run is
`ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0` (TAG = `prefopt`: its export has the
run's own `weights16.npz` and decoding choice), the opponents' runs `falcon-e60034381f4a/recommended-5ccec0259b-s0`
and `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`. The registered judge's pass is complete on the Falcon-H1-Tiny set and partial on the MobileLLM-R1.5 set: its rows give the pairs rated in its files as read at 22:19 UTC (final-prefopt-vs-falcon: 200 records, last written 22:11:18, final-prefopt-vs-mobilellm: 92 records, last written 21:34:37), and its full pass on the MobileLLM-R1.5 set is [TBD: running on CPU runtimes since 20:41 UTC, at about 178 s per pair on the MobileLLM-R1.5 set]. The second registered judge
(`gemma4-e4b`) was not run on the test pairs, so its rows of the shells are gone (Section 7.2, deviation 1).

**Table M1 (R1). The test split, 200 prompts: each opponent and each reader** (`R/<set>--<source>.json`, fields `overall` and `judge`, reproduced by `final_results.py`; sets `final-prefopt-vs-falcon`, `-vs-falcon-k6lp`, `-vs-falcon-k6ds`, `-vs-mobilellm`, `ref-falcon-vs-mobilellm-test`).

| opponent | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders agree | first-position preference |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | `qwen3-4b-2507` (registered), full pass | 200 | 12 / 61 / 127 | 0.378 [0.313, 0.446] | 0.164 [0.097, 0.266] | < 0.0001 | 60.5% | −0.134 |
| Falcon-H1-Tiny-90M-Instruct | Opus, two readings | 200 | 35 / 108 / 57 | 0.318 [0.257, 0.385] | 0.245 [0.182, 0.321] | < 0.0001 | 90.5% | +0.005 |
| Falcon-H1-Tiny, K = 6 kept by its own log-probability | Opus, two readings | 200 | 35 / 112 / 53 | 0.308 [0.248, 0.375] | 0.238 [0.176, 0.313] | < 0.0001 | 93.5% | +0.020 |
| Falcon-H1-Tiny, K = 6 kept by our selector | Opus, two readings | 200 | 29 / 116 / 55 | 0.283 [0.225, 0.349] | 0.200 [0.143, 0.272] | < 0.0001 | 92.5% | +0.018 |
| MobileLLM-R1.5-140M | `qwen3-4b-2507` (registered), partial | 92 of 200 | 22 / 3 / 67 | 0.603 [0.501, 0.697] | 0.880 [0.700, 0.958] | 0.0002 | 72.8% | +0.011 |
| MobileLLM-R1.5-140M | Opus, two readings | 200 | 116 / 15 / 69 | 0.753 [0.688, 0.807] | 0.885 [0.820, 0.929] | < 0.0001 | 96.0% | −0.005 |
| MobileLLM-R1.5, K = 6 (both rules) | – | [TBD: replies in generation since 19:17 UTC, 40 of 200 prompts by 21:13] | | | | | | |
| Falcon-H1-Tiny, against our K = 1 arm | Opus, two readings | 200 | 22 / 122 / 56 | 0.250 [0.195, 0.314] | 0.153 [0.103, 0.220] | < 0.0001 | 92.5% | +0.015 |
| MobileLLM-R1.5, against our K = 1 arm | Opus, two readings | 200 | 99 / 16 / 85 | 0.708 [0.641, 0.766] | 0.861 [0.786, 0.913] | < 0.0001 | 91.5% | +0.008 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 200 | 144 / 6 / 50 | 0.845 [0.788, 0.889] | 0.960 [0.915, 0.982] | < 0.0001 | 93.5% | +0.013 |

**Table M2 (R2). By category and by turns** (W / L / T; the same files, fields `by_category` and `by_turns`; prompts of the `test` split).

| slice (prompts) | Falcon-H1-Tiny: Opus | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon, K = 6 own log-prob.: Opus | Falcon, K = 6 our selector: Opus | MobileLLM-R1.5: Opus | MobileLLM-R1.5: `qwen3-4b-2507` (partial) | ours K = 1 vs Falcon: Opus | ours K = 1 vs MobileLLM: Opus |
|---|---|---|---|---|---|---|---|---|
| chat (50) | 10 / 24 / 16 | 3 / 10 / 37 | 8 / 24 / 18 | 7 / 25 / 18 | 28 / 1 / 21 | 5 / 0 / 17 | 9 / 31 / 10 | 28 / 2 / 20 |
| code (30) | 10 / 13 / 7 | 3 / 4 / 23 | 11 / 13 / 6 | 11 / 12 / 7 | 16 / 2 / 12 | 6 / 0 / 6 | 7 / 13 / 10 | 13 / 4 / 13 |
| instruction-following (40) | 7 / 21 / 12 | 4 / 17 / 19 | 9 / 21 / 10 | 7 / 22 / 11 | 29 / 0 / 11 | 6 / 0 / 11 | 2 / 25 / 13 | 24 / 0 / 16 |
| maths (30) | 4 / 18 / 8 | 0 / 12 / 18 | 4 / 19 / 7 | 3 / 20 / 7 | 7 / 12 / 11 | 3 / 3 / 13 | 3 / 19 / 8 | 6 / 10 / 14 |
| writing (50) | 4 / 32 / 14 | 2 / 18 / 30 | 3 / 35 / 12 | 1 / 37 / 12 | 36 / 0 / 14 | 2 / 0 / 20 | 1 / 34 / 15 | 28 / 0 / 22 |
| single-turn (148) | 28 / 78 / 42 | 10 / 45 / 93 | 26 / 80 / 42 | 21 / 84 / 43 | 83 / 11 / 54 | 16 / 3 / 49 | 16 / 85 / 47 | 67 / 13 / 68 |
| two-turn (52) | 7 / 30 / 15 | 2 / 16 / 34 | 9 / 32 / 11 | 8 / 32 / 12 | 33 / 4 / 15 | 6 / 0 / 18 | 6 / 37 / 9 | 32 / 3 / 17 |

**Table M3 (R3). Length buckets and reader diagnostics** (the same files, fields `by_length` (ratio of our words to the opponent's, +1 each: shorter below 2/3, longer above 3/2) and `judge`; the Opus records carry no probability mass and no clipping, since the readers read the chunk files).

| row | Falcon-H1-Tiny: Opus | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon, K = 6 own log-prob.: Opus | Falcon, K = 6 our selector: Opus | MobileLLM-R1.5: Opus | MobileLLM-R1.5: `qwen3-4b-2507` (partial) | ours K = 1 vs Falcon: Opus | ours K = 1 vs MobileLLM: Opus |
|---|---|---|---|---|---|---|---|---|
| ours shorter: n; W / L / T | 93; 26 / 41 / 26 | 93; 11 / 14 / 68 | 94; 26 / 38 / 30 | 89; 19 / 37 / 33 | 140; 77 / 9 / 54 | 63; 18 / 0 / 45 | 81; 15 / 45 / 21 | 140; 71 / 9 / 60 |
| similar: n; W / L / T | 72; 7 / 40 / 25 | 72; 1 / 25 / 46 | 76; 8 / 51 / 17 | 82; 10 / 55 / 17 | 28; 17 / 5 / 6 | 14; 1 / 2 / 11 | 79; 5 / 46 / 28 | 23; 6 / 5 / 12 |
| ours longer: n; W / L / T | 35; 2 / 27 / 6 | 35; 0 / 22 / 13 | 30; 1 / 23 / 6 | 29; 0 / 24 / 5 | 32; 22 / 1 / 9 | 15; 3 / 1 / 11 | 40; 2 / 31 / 7 | 37; 22 / 2 / 13 |
| orders agree | 90.5% | 60.5% | 93.5% | 92.5% | 96.0% | 72.8% | 92.5% | 91.5% |
| first-position preference | +0.005 | −0.134 | +0.020 | +0.018 | −0.005 | +0.011 | +0.015 | +0.008 |
| probability mass on A, B, C (field `judge.format_mass_mean`) | – | 0.983 | – | – | – | 0.992 | – | – |
| pairs with a reply cut for the judge (over 4,000 characters) | – | 1 | – | – | – | 38 | – | – |

**Table M4 (R4). Exact scores on the 200 test prompts (252 turns), in full** (`bes.eval.exact_score` on every reply, by `lead4_work/scripts/exact_split.py --split test` with every run below, `lead5_work/exact_test.json`; equal field by field to the run's report in `study/proxy/reports/exact/` where one exists, which is every run but MobileLLM-R1.5's `neutral` arm; counts from `lead5_work/exact_counts.json`).

| measure | ours, official | ours, K = 1 | ours, trained weights | Falcon-H1-Tiny | Falcon K = 6, own log-prob. | Falcon K = 6, our selector | MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |
|---|---|---|---|---|---|---|---|---|
| maths answers correct (of 30) | 6 | 5 | 6 | 15 | 17 | 18 | 12 | 11 |
| constrained prompts with every constraint met (of 48) | 38 | 37 | 41 | 33 | 35 | 42 | 8 | 9 |
| single constraints met (of 56) | 46 | 44 | 48 | 41 | 42 | 50 | 10 | 11 |
| degenerate replies (of 252) | 5 | 17 | 3 | 2 | 1 | 0 | 158 | 148 |
| … cut at the token limit | 5 | 17 | 3 | 1 | 0 | 0 | 150 | 146 |
| … looping tail; repeated lines | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 14; 54 | 17; 50 |
| … empty; unclosed code fence; role markers | 0; 0; 0 | 0; 0; 0 | 0; 0; 0 | 0; 1; 0 | 0; 1; 0 | 0; 0; 0 | 2; 7; 0 | 1; 10; 0 |
| stock refusals; stock deflections (of 252) | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| words per reply: median; mean | 101; 130.4 | 105; 149.7 | 99; 119.2 | 155; 173.4 | 147; 174.3 | 145; 168.4 | 395; 474.1 | 517; 533.7 |
| mean repeated-4-gram rate | 0.034 | 0.054 | 0.038 | 0.020 | 0.026 | 0.021 | 0.258 | 0.290 |

A reply can be degenerate in several ways at once, so the sub-rows do not add up.

**Table M5 (R5). The trained weights against the weights after preference optimisation** (the direct pair set `final-prefopt-vs-trained`, both readings, `R/final-prefopt-vs-trained--judge-opus.json`; exact scores from Table M4; held-out losses from `lead5_work/heldout5.json`; the in-run checks from Table C5). The pair sets `final-trained-vs-falcon` and `-vs-mobilellm` were published at 19:07 UTC and not read: the direct set answers the question on the same prompts.

| measure | after preference optimisation (official) | trained weights | difference |
|---|---|---|---|
| blind, Opus, two readings, 200 prompts: W / L / T (official's side) | 59 / 34 / 107 | | score 0.563 [0.493, 0.629], p = 0.012 |
| … the 189 untouched prompts | 55 / 33 / 101 | | score 0.558 [0.487, 0.627], p = 0.025 |
| … by category: chat; code; instruction-following; maths; writing | 16 / 8 / 26; 10 / 1 / 19; 14 / 8 / 18; 5 / 5 / 20; 14 / 12 / 24 | | |
| maths answers correct (of 30) | 6 | 6 | +0 |
| every constraint met (of 48) | 38 | 41 | -3 |
| single constraints met (of 56) | 46 | 48 | -2 |
| degenerate replies (of 252) | 5 | 3 | +2 |
| words per reply, mean | 130.4 | 119.2 | |
| held-out loss, web + chat (64 + 64 rows) | 2.0562 | 2.0438 | +0.0124 [0.0112, 0.0136] |
| held-out loss, chat assistant tokens | 1.2095 | 1.1983 | +0.0112 [0.0097, 0.0129] |
| in-run: held-out preference accuracy (1,826 pairs); assistant-token loss of the stage's own rows | 0.694; 1.5208 | 0.500; 1.5043 | accuracy +0.194; loss +1.10% |

**Table M6 (R6). Held-out losses at the end of training and of the export** (nats per token). (a) The comparison set of Section 6.5 (64 web and 64 chat rows): the runtime's evaluation of the last checkpoint (`N/heldout_v9.jsonl`) and our CPU evaluations of both weights of step 51,812 from `N/snap_final` (`heldout_eval5.py`; bits per byte as `lead4_work/scripts/bpb_web.py` computes them, in `heldout5.py`). (b) The export's own 16-bit check on 16 rows per domain (`check` of `final/export.json`). The proxy rows are in neither part (Section 7.2, deviation 3).

| (a) 64 rows per domain | trained weights, step 51,812 (runtime) | after preference optimisation (local) | `final` 29,722 (the graft) |
|---|---|---|---|
| web | 2.6690 | 2.6804 | 2.7886 |
| chat | 1.3668 | 1.3803 | 1.6183 |
| chat, assistant tokens | 1.1983 | 1.2095 | 1.5164 |
| web + chat (comparison set) | 2.0438 | 2.0562 | 2.2267 |
| bits per byte, web rows (opponents: 1.570 and 1.500) | 0.974 | 0.978 | 1.017 |

| (b) export check, 16 rows per domain | float32 weights | 16-bit weights | difference |
|---|---|---|---|
| web (16,354 weighted tokens) | 2.69236 | 2.69243 | +0.000075 |
| chat (15,690 weighted tokens) | 1.31725 | 1.31721 | -0.000039 |
| all three domains, proxy included (43,138) | 2.34440 | 2.34443 | +0.000022 (tolerance 0.002) |

**Table M7 (R7). What the preference-optimisation stage did** (`run/stages.json`, `run/post/prefopt_report.json` and the run log in `N/snap_final`; Table C5 gives the candidates in full).

| quantity | value |
|---|---|
| minutes used of 60 | 29.7 (18:37:07–19:06:51 UTC) |
| preference pairs used | 35,201 for training and 1,826 held out by prompt: 37,007 UltraFeedback pairs and 20 of our own sampled pairs |
| learning rates tried and steps run at each | 1 × 10⁻⁶ and 3 × 10⁻⁶, 551 steps each (one epoch), 485 s and 424 s of training |
| held-out preference accuracy before / after (standard error) | 0.500 / 0.688 (0.011) and 0.694 (0.011) |
| held-out assistant loss change (limit +3%) | +0.86% and +1.10% |
| sampled length and repetition against their bands | length 0.99 and 0.95 × the start's (band 0.7–1.6), repeated 4-grams 0.86 and 0.87 × (at most 1.2 × + 0.01) |
| accepted (weights kept) or rejected | both accepted; kept 3e-06, the higher accuracy ("lr=3e-06 accepted with held-out preference accuracy 0.6944 (start 0.5000)") |

**Table M8 (R8). What the decoding selection chose** (the manifest's `decode` record; `run/stages.json`; the export's `decode_choice`).

| quantity | value |
|---|---|
| settings compared with the default | one particle (`k1`); no length band (`noband`); no library (none was built) |
| development prompts used; paired sign test of each against the default | the 200 `dev` prompts; mean exact score default 0.910, `k1` 0.850, `noband` 0.915; `k1` better on 1, worse on 18 (p = 7.6e-05); `noband` better on 2, worse on 0 (p = 0.50) |
| setting kept | `default` (6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library) |
| minutes used of 35 | 54.9, in a slice of 65.0 (the stage was given the time preference optimisation left) |

**Table M9 (R9). Training totals** (the trainer's ledger `run/train/ledger.json` and train log in `N/snap_final`; TPU seconds from `S/tpu_intervals.txt`, all three intervals closed; `lead5_work/numbers.json`).

| quantity | value |
|---|---|
| final step; tokens trained (`v9` part since step 29,722) | 51,812; 13,582,204,928 (5,790,760,960 in 22,090 steps) |
| training seconds of the kept trajectory; checkpoint seconds; set-up seconds | 62,893.1; 23.2; 279.6 |
| TPU seconds: runtime 1, runtime 2, runtime 3, total | 7,325; 8,220; 70,509; 86,054 of 86,400 |
| share of TPU time in training steps of the kept trajectory | 73.1% |
| live rate in the last stretch (p ≥ 0.92, conversations train on assistant tokens only), n | 189,035 tokens/s, n = 73 (steps 48,200–51,800) |

**Table M10 (R10). The 16-bit check of the export** (`final/export.json`: `check`, `weights`, `main_files`, `readset`; `SHA256SUMS`).

| quantity | value |
|---|---|
| largest absolute loss difference, 16-bit against float32, web and chat (the proxy rows' difference enters only the check's pass or fail) | 7.5e-05 nats (web); passed |
| leaves stored in bfloat16 / float16 (rule: the smaller relative L2 error; ties float16) | 0 / 64 (98,036,104 numbers; largest relative L2 error of a leaf 2.84e-04) |
| export size (bytes) and the SHA-256 of the weights | `weights16.npz` 196,088,804 bytes, `2da7cbd2e2f1fe49e805cb9bde62b15751f41ae92e71f00a3b0ba69f241afcf4`; the packed export 196,956,160 bytes in 12 parts |
| read-set audit of the export: numbers read per token, standard reading | ledger 19,631,240; measured on held-out sequences 19,427,809 (web, mean over 2,048 positions) and 19,427,956 (chat), at most 19,429,996; strict reading 24,352,024 at 1,024 positions and 27,929,250 at 2,048 (web) |

**Table M11 (R11). Tables M1 and M4 on the 189 test prompts no mid-run sample touched** (240 turns, 28 maths and 44 constrained prompts; the 11 seen ids of Section 7.2 left out; readers from each report's `per_prompt_score` as `lead4_work/scripts/restrict_report.py` does, in `final_results.py`; exact scores by `exact_split.py --exclude`, `lead5_work/exact_test189.json`).

| opponent | reader | n | W / L / T | score [95%] | sign test p |
|---|---|---|---|---|---|
| Falcon-H1-Tiny | `qwen3-4b-2507` | 189 | 11 / 57 / 121 | 0.378 [0.312, 0.449] | < 0.0001 |
| Falcon-H1-Tiny | Opus, two readings | 189 | 32 / 101 / 56 | 0.317 [0.255, 0.387] | < 0.0001 |
| Falcon-H1-Tiny, K = 6 own log-probability | Opus, two readings | 189 | 34 / 106 / 49 | 0.310 [0.248, 0.379] | < 0.0001 |
| Falcon-H1-Tiny, K = 6 our selector | Opus, two readings | 189 | 27 / 110 / 52 | 0.280 [0.221, 0.348] | < 0.0001 |
| MobileLLM-R1.5 | `qwen3-4b-2507`, partial | 87 | 20 / 3 / 64 | 0.598 [0.493, 0.694] | 0.0005 |
| MobileLLM-R1.5 | Opus, two readings | 189 | 109 / 14 / 66 | 0.751 [0.685, 0.808] | < 0.0001 |
| our trained weights | Opus, two readings | 189 | 55 / 33 / 101 | 0.558 [0.487, 0.627] | 0.025 |
| Falcon-H1-Tiny, against our K = 1 arm | Opus, two readings | 189 | 20 / 116 / 53 | 0.246 [0.190, 0.312] | < 0.0001 |
| MobileLLM-R1.5, against our K = 1 arm | Opus, two readings | 189 | 92 / 14 / 83 | 0.706 [0.638, 0.767] | < 0.0001 |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 189 | 137 / 5 / 47 | 0.849 [0.791, 0.893] | < 0.0001 |
| exact scores: maths correct (of 28); constrained prompts met (of 44) | – | – | ours, official 6 and 34; ours, K = 1 5 and 33; ours, trained weights 6 and 37; Falcon-H1-Tiny 14 and 29; Falcon K = 6, own log-prob. 16 and 32; Falcon K = 6, our selector 17 and 38; MobileLLM-R1.5 `recommended` 11 and 8; MobileLLM-R1.5 `neutral` 10 and 8 | – | – |

**Table M12 (R12). Decoding throughput of the export on CPU** (the official test generation's summary `N/testgen/prefopt/out/testgen.json` and its log; the trained weights' generation for comparison, `N/testgen/trained/out/testgen.json`; on `dev`, 75 replies took 1,798 s with 16 workers, `FACTS_2.md`).

| quantity | official (preference-optimised) | trained weights |
|---|---|---|
| workers × cores per worker; host | 40 × 1; the TPU runtime's host (CPUs 2–21, 24–43 of 44) | 40 × 1; the same |
| replies; wall time, export and calibration included | 252; 2,109.8 s (19:31:22–20:06:32) | 252; 1,840.4 s (18:36:25–19:07:06) |
| seconds per reply (median); new tokens per reply (median) | 206.7; 155.5 | 159.4; 151.5 |
| seconds per decoding step, 6 particles, one core (median; mean) | 0.567; 0.595 | 0.550; 0.575 |

**Table M13. Paired comparisons of our per-prompt score on the same prompts** (`final_results.py`, `bes.eval.proxy.score.paired_sign_test`, as `python -m bes.eval.proxy compare --a A --b B --source judge-opus`; a prompt is better when our score against the opponent is higher in A than in B).

| A | B | prompts better in A; in B; same | mean difference | sign test p | on the 189 untouched prompts |
|---|---|---|---|---|---|
| official vs released Falcon | official vs Falcon K = 6, own log-probability | 29; 25; 146 | +0.0100 | 0.6835 | 26; 23; 140, p = 0.7754 |
| official vs released Falcon | official vs Falcon K = 6, our selector | 34; 21; 145 | +0.0350 | 0.1048 | 32; 19; 138, p = 0.0919 |
| official (K = 6) vs Falcon | ours K = 1 vs Falcon | 31; 12; 157 | +0.0675 | 0.0054 | 30; 11; 148, p = 0.0043 |
| official (K = 6) vs MobileLLM-R1.5 | ours K = 1 vs MobileLLM-R1.5 | 28; 13; 159 | +0.0450 | 0.0275 | 26; 11; 152, p = 0.0201 |


## Appendix N. Related work in detail

The second draft's related-work section, kept here in full; Section 3 is its summary.

**Training to a deadline.** Cramming trains a masked language model on one consumer GPU in one day
[1], Izsak et al. a BERT on one 8-GPU server in 24 hours [2]; both tie the learning-rate decay to the
time budget, as budgeted training recommends [3]. Speedruns minimise time to a target on an 8 × H100
node [4, 5]; nanochat runs the whole chat pipeline [5]. Our schedule is of this kind; what differs is
the product, a chat model trained on one chip and compared blind with released models, and the need
for the schedule to survive a reclaimed runtime and a change of configuration.

**Small open models.** MobileLLM found deep-and-thin shapes and shared embeddings to matter below 10⁹
parameters [6]; MobileLLM-R1 pre-trains on 4.2 × 10¹² tokens [7] and R1.5 adds on-policy distillation
[8] from a 4B teacher [9]. Falcon-H1 runs attention and state-space heads in parallel [10]; its 90M
instruct model saw 8 × 10¹¹ tokens with instruction data from the start [11]. SmolLM2 trains 135M and
360M models on 2 × 10¹² and 4 × 10¹² tokens [12]; TinyStories showed fluency below 10⁷ parameters in a
narrow domain [13].

**Weight-shared depth.** Re-applying one block goes back to the Universal Transformer [14] and ALBERT
[15]; looped models match deeper ones on synthetic reasoning [16] and have been scaled to 3.5 × 10⁹
parameters [17] and 7.7 × 10¹² tokens [18], and Parcae stabilises them [19]. Schwethelm et al. fit an iso-depth law L = E + A(N_once + r^φ N_rec)^(−α)
+ B D^(−β), in which shared recurrences lose to unique layers at equal compute and two-lane
hyper-connections [20] raise φ from 0.46 to 0.65 [21]; our core is ported from their code. Our reason to
loop differs: under a read budget a re-applied weight counts once, so looping buys compute per number
read, not per FLOP; and our core has no prelude or coda (N_once = 0), outside the law's fitted range.
The per-loop value embeddings follow [4, 5, 22].

**Vocabulary-indexed parameters and two counts.** Stored-but-sparsely-read capacity includes hashed
feed-forward weights [23], key-value memories [24, 25], larger input vocabularies [26], n-gram
embeddings off the accelerator [27, 28], and L3's per-token keys and values [29], to which `v9`'s token
neurons are closest. Token-keyed parameters were found most useful on values and outputs and least on
queries and keys [30, 31], a caution for our output gains. Counting the numbers read per token is the
mixture-of-experts convention [32, 33]; we found no published counterpart of a read-set audit.

**Copying and exact counts.** Mixing the network's distribution with a copy distribution goes back to
pointer-sentinel models and the continuous cache [34, 35], with neighbours and n-gram counts in kNN-LM
and infini-gram [36, 37]; our output law contains the Pitman–Yor predictive [38] with the network as
base measure, a known rule. Induction heads [39], n-gram heads [40] and self-reinforcing repetitions
[41] bear on the tape; the sink follows [42, 43]; `final`'s library was counted against the size
budget, as in EfficientQA [44].

**Changing a model during training.** Net2Net [45] and network morphism [46] grow networks by
function-preserving transformations. Staged training applies growth operators to the whole training
state, copying Adam moments for existing layers and zeroing them for new ones, and asks that both the
loss and the training dynamics be preserved [47]; LLaMA Pro [48], sparse upcycling [49] and LoRA [50]
start new paths at an identity or at zero. Our graft is a loss-preserving operator of this family. What
is particular is narrow: vocabulary-indexed tables and a feed-forward branch are added rather than width
or depth; the continued schedule and data plan are those of a wall-clock budget; and the function and
the shared gradients were checked at the live run's checkpoint. The training-dynamics property of [47]
is not tested.

**Optimisers, kernels, decoding, judging.** We use MuonH [51–57] as [21] does. Our kernels follow the
IO-aware pattern of FlashAttention [58, 59] and the Pallas kernels of JAX [60–62]; Ivanov et al. found
transformer training on GPUs memory-bound [63]. We add a measurement against a budget stated in
parameters, on one TPU v6e, not a new algorithm. The decoder builds on sequential Monte Carlo steering
[64–68], best-of-n bounds [69], self-consistency [70], IFEval-style constraints [71], calculator
annotations [72] and budget forcing [73]; DPO [74] with UltraFeedback is a standard last stage
[11, 12, 75–77]. Language-model judges favour one position, longer replies and their own style [78–81];
we count a win only when both orders agree, the rule of [78], and report outcomes by length ratio [80].
