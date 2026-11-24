# Parameters Read Are Not Bytes Moved: An Audited One-Day Training Run of a 98M-Stored, 20M-Read Language Model on One TPU v6e

[TBD: author list and affiliations, at submission]

**Status of this draft.** Written while the run is in progress. State of the facts: 24 November
2026, 10:00 UTC. Training is under way in the configuration `configs/v9.py` (step 29,901, progress
p = 0.515, 7.84 × 10⁹ training tokens at the last snapshot, 10:03 UTC). Every result that does not
exist yet is written as [TBD: what, when it will exist]. Every number in the text has an entry in
`CLAIMS.md` with its source.

## Abstract

We train a language model from scratch under a budget stated in parameters: at most 100 million
stored numbers of at most 16 bits, at most 20 million of them read to produce one token, and 24
hours of a single TPU v6e chip. The model is to be judged blind, by pairwise preference, against
Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M. The budget counts parameters; the chip charges
for moving bytes. We measure the difference. Two configurations of our model read 19.03 million and
19.63 million numbers per token and differ by 3.6% in arithmetic per token, yet on the chip the
second needs 1.54 to 1.62 times as long per training token: its per-token lookup pages are cheap in
the parameter ledger and expensive in the shapes the hardware moves them in. In the dense looped
core, 62% of a block's time in our first implementation was data movement outside matrix products;
rewrites that leave the function unchanged to floating-point rounding, smaller device-side pieces of
each micro-batch and an attention kernel that walks only the causal triangle raised the live
throughput of the same model from 112–117 thousand to 281 thousand tokens per second. When the
configuration was changed at progress 0.505 of the wall-clock budget, an exact graft carried the
34.5 million shared numbers, their optimiser state and the schedule into a parameter tree whose 63.5
million new numbers contribute exactly zero at the switch, so the 7.79 × 10⁹ tokens already trained
were kept; in rehearsals of the same graft on two checkpoints of the live run, source and grafted
parameters gave identical losses (difference 0.0) on a test batch. We report the two-ledger
accounting and its audit, the on-chip measurements including the changes that
did not help, the graft and what a single run cannot show about it, data hygiene driven by the
model's own failures, and a blind protocol with sealed pair sets and two readings on exchanged
sides. At 0.4 of training the model was clearly behind Falcon-H1-Tiny on the prompts we sampled; the
final comparison is [TBD: blind test scores against both opponents with 95% intervals, after the
evaluation on 24–25 November 2026].

## 1. Introduction

Small language models are increasingly described by parameter counts: the numbers they store, and
for sparse models the numbers they read per token ("active" parameters) [RW: active parameters of mixture-of-experts models; refs.md 32]. Our task fixed a budget of exactly that kind. The model
may store at most 100 million numbers of at most 16 bits each, weights, lookup tables and any
written knowledge included; it may read at most 20 million of them to produce the distribution of
one token given its cached state; and it must be trained from scratch within 24 hours of one TPU
v6e chip (a Colab `v6e-1` runtime). It is then compared blind, by pairwise preference on held-out
prompts, with two public models of similar size, Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M
[RW: the opponents; refs.md 6, 7, 9, 10, 11].

The budget turns a design question into an allocation question. A dense core that every token
reads costs about 19 million reads, so the other 80 million stored numbers can only be read
sparsely: one row of a table per token, or one passage of a library per retrieval. Our first design
spent them on a written library with retrieval; the configuration that is now training spends them
on per-token *pages*, rows of lookup tables indexed by the token id: eight private feed-forward
neurons per block in a shared 192-dimensional space and a gain on the attention output, next to the
per-loop value embedding that both configurations have. In the parameter ledger the pages are nearly free: 63.5 million stored numbers
for 0.60 million more reads per token.

The thesis of this paper is that, for a small model trained on one accelerator against a wall
clock, a budget in parameters, stored or read, does not predict what the training costs; bytes
moved, and the shapes they are moved in, do. That data movement rather than arithmetic bounds the
training of transformers is known on GPUs [RW: data movement in transformer training; refs.md 63];
what we add is the measurement against a budget stated in parameters, on one TPU v6e, with a design
that the parameter ledger calls nearly free. We show it with measurements on the chip, in three
ways. First, the two configurations above read 19.03 and 19.63 million numbers per token and differ
by 3.6% in analytic arithmetic per token, but the second needs 1.54 to 1.62 times as long per
training token (Section 7.8). Second, inside the dense core, 62% of a block's time in our first
implementation was spent outside matrix products and the attention kernel, in slices, transposes
and float32 round trips; removing that traffic without changing the function, cutting each
micro-batch into smaller pieces on the device, and replacing a general block-sparse attention
kernel with one that holds a (row, head) slice in on-chip memory and walks only the causal triangle,
took the live throughput of the same model on the same chip from 112–117 thousand to 281 thousand
tokens per second (Sections 7.2–7.5). Third, a list of changes that were expected to be faster,
some of them by the arithmetic, and that the chip measured to be slower or no faster (Section 7.6).

Because the budget is a wall clock, every hour is spent once, and a design decision cannot be
revisited by starting again. At progress 0.505 of the budget the configuration was changed at the
user's request from `configs/final.py` (dense core, written library) to `configs/v9.py` (dense core,
token pages, no library). Training the new configuration from scratch in the 8.6 hours left would
have seen fewer tokens than the run already had. We therefore *grafted* the run: every parameter
leaf that both configurations share is carried with its optimiser state, the update count, the step
and the schedule, and the new leaves start at an initialisation at which the paths they form
contribute exactly zero, so the function, and the gradient with respect to the shared parameters, are unchanged at the
switch (Section 8).

**Contributions.** Each is separately checkable; Section 12 and `CLAIMS.md` give the evidence.

1. **Two ledgers and an audit** (Section 4). A precise accounting of stored numbers and of numbers
   read per token, in a standard and a strict reading, enforced by tests on the real parameter
   tree and audited by a reference forward pass that receives only the rows it reads. Our
   configuration stores 98,036,104 numbers and reads 19,631,240 per token in the standard reading;
   it does not meet the strict reading (21.2–28.2 million on 40 real documents), and we say so.
2. **Where training time goes on one TPU v6e** (Section 7). On-chip measurements of a 2-block,
   8-loop core: the share of memory-bound glue, an exact path from 112–117 thousand to 281 thousand
   tokens per second, the measured cost of lookup parameters (1.54–1.62 times the time per token for
   3.6% more arithmetic), a calibration of the compiler's cost model against the chip, and the
   changes that did not help.
3. **An exact graft for a configuration that only adds parameters** (Section 8). A procedure and a
   check that carry a run into a larger parameter tree mid-budget without changing the function,
   used on the live run at p = 0.505; with what it carries, what it changes (step time, data
   stream, optimiser state of the new leaves) and what a single run cannot show about it. It is
   close to known function-preserving growth (Section 8.6); its use inside a wall-clock budget is
   what is particular.
4. **A protocol for a claim that cannot be tuned after the fact, and an honest standing**
   (Sections 9, 11, 12). A blind pairwise protocol with seeded side assignment, pair sets sealed by
   hash-chain commitments, two readings on exchanged sides and strict agreement, exact scores read
   before preferences, a pre-computed power analysis and controls matched in sample count and in
   selection rule; data hygiene driven by the model's own failures (stock refusals and deflections
   measured in the sources and removed); and the mid-run standing, which is behind Falcon-H1-Tiny.

**What we do not claim.** We do not claim that per-token pages improve the model at this scale: the
evidence for them is a small CPU laboratory and a micro ablation whose effect is below its
pre-registered resolution, and there is no from-scratch baseline of either configuration at full
scale. We do not claim that our model beats either opponent; that result does not exist yet, and
mid-run samples point the other way. We do not claim mathematical novelty for the accounting or
the estimators; an adversarial review of the design mathematics rated its novelty 2–3 out of 10
(`docs/MATH.md`, first paragraph).

## 2. Setting and constraints

### 2.1 The budget

The rules, as fixed before training (`docs/PLAN.md` §2, `docs/ACCOUNTING.md`):

- A **parameter** is one stored number of at most 16 bits that ships with the model. The **stored
  total** is every such number: dense weights, lookup tables, register and candidate tables, and a
  written library counted as its compressed bytes divided by two, with its index. The tokenizer's
  merge table and the tables derived from it deterministically are not counted.
- **Numbers read per token, standard reading**: the distinct numbers read to produce the full
  next-token distribution of one position given the cached state. Dense weights count once however
  often they are reused; a lookup table counts the rows read. This is the convention behind the
  "active parameters" of sparse models.
- **Strict reading**: the union over the whole context window of what the output depends on, so a
  token-indexed table counts one row per distinct token of the window.
- Limits: at most 100 million stored, at most 20 million read per token (standard reading, with the
  strict reading reported next to it).
- Compute: 24 hours of TPU v6e time in total, one chip, from scratch. The operator's reading of
  this rule, reported to the user on 23 November, counts the time a runtime is allocated and not
  the waits between runtimes (`docs/CHANGES.md` §19.2).

### 2.2 Hardware

One TPU v6e chip ("TPU v6 lite" as JAX reports it) with 31.24 GB of device memory available to the
compiler, on a host with 44 cores and 185.7 GB of memory; JAX and jaxlib 0.11.2, libtpu 0.0.50
(run manifest). Our matrix dimensions are multiples of 256 except the attention head (128)
(`docs/TPU_NOTES.md`, Target). On this chip the two feed-forward products of a block ran at 814
TFLOP/s in the forward pass when timed alone (`docs/TPU_NOTES.md`, block table); we use that
measured rate, not a vendor figure, as the ceiling in Section 7.7.

### 2.3 The opponents and how they are run

Falcon-H1-Tiny-90M-Instruct runs as the publisher's Q8_0 GGUF file through llama.cpp with
llama.cpp's defaults (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, no repetition penalty, at
most 1024 new tokens), since the publisher gives no sampling parameters. MobileLLM-R1.5-140M's
upstream checkpoint is gated; an ungated community Q8_0 file is rewritten, without touching a
quantised value, as an equivalent `qwen3` model and checked against an independent forward pass of
the original tensors (mean KL 0.003 nats per position, 98% equal argmaxes). It reasons before it
answers; it is run with its publisher's sampling (temperature 0.6, top-p 0.95) and budget forcing
(the reasoning is closed after 3072 tokens and the answer gets 1024 more), in two arms (its own
injected reasoning prompt, and a neutral system prompt), and is credited with the better arm on the
development split (`docs/EVAL_PROTOCOL.md` §2). Our replies have at most 768 of our tokens, about
430 words (`docs/EVAL_PROTOCOL.md` §2, "Our token budget"). Both opponents were trained on far more
tokens than our run can see: Falcon-H1-Tiny-90M-Instruct on 800 billion according to its publisher,
and MobileLLM-R1.5-140M continues MobileLLM-R1, whose pre-training used 4.2 trillion (refs.md 11, 7,
9, as verified by the related-work agent); our run will see about 1.3 × 10¹⁰ (Section 8.5).

### 2.4 The time that was available

The 24 hours were spread over three runtimes (Table 1). The first was given back at the user's
request after the profiling of Section 7 and produced no model that was kept; the second was taken
back by the provider after 2 hours 12 minutes for a reason we could not establish; the third has
run since 24 November 01:05:55 UTC. At 10:00 UTC on 24 November, 13.22 of the 24 hours had been
used. At the graft (09:45:35), 12.98 hours had been used, of which 8.98 hours (69%) were training
steps of the trajectory that was kept (the trainer's ledger).

**Table 1. TPU runtimes** (`docs/CHANGES.md` §19.1–19.4; the TPU-time ledger of the operator).

| runtime | allocated (UTC) | seconds | what it produced |
|---|---|---|---|
| 1 | 23 Nov 02:58:32 – 05:00:37 | 7,325 | profiling and the exact rewrites of Section 7.3; its training runs were discarded; returned at the user's request |
| 2 | 23 Nov 17:13:00 – 19:30:00 (counted from the earliest possible allocation) | 8,220 | training from step 0 to step 5,075 (kept through a snapshot); taken back by the provider at about 19:29 |
| 3 | from 24 Nov 01:05:55 | 32,045 at 10:00 | training resumed at step 5,075 at 01:54:10; in progress |

The run's deadline is 24 November 20:31:49 UTC; training ends at 18:33:49 and the remaining 118
minutes are reserved for preference optimisation (60), decoding selection (35), sampling (3),
export (5) and a buffer (15) (`docs/CHANGES.md` §19.5; the trainer's setup record).

### 2.5 How the work was done

A human user set the goals and took the decisions that are marked as theirs; the design, the code,
the measurements and this draft were produced by language-model agents working under the user's
direction, and every change is dated in `docs/CHANGES.md`. We state this because it bears on one
threat to validity: the readers of the blind evaluation are also language models (Section 14).

## 3. Related work

[Slot: this section is `related_work.md`, written by the related-work agent, with its references
numbered and verified in `refs.md`. The `[RW: topic; refs.md n]` markers of this draft name the
topic and the current number in that list; the final assembly replaces them.]

## 4. Two ledgers: what the budget counts

### 4.1 The ledgers of both configurations

`bes/budget.py` computes both ledgers from the real parameter tree (and, for `configs/final.py`,
from the built library), and tests enforce the limits (`tests/model/test_config_budgets.py`,
`tests/model/test_roles_budget.py`). Table 2 gives them side by side.

**Table 2. Stored numbers and numbers read per token** (standard reading; `docs/V9.md` §5,
`docs/ACCOUNTING.md`).

| row | `final`: stored | `final`: read per token | `v9`: stored | `v9`: read per token |
|---|---|---|---|---|
| looped core (2 blocks) | 14,118,912 | all | 14,118,912 | all |
| output head (8192 × 512 + 512 × 768) | 4,587,520 | all | 4,587,520 | all |
| norms, lanes, sink, trust head, critic | 306,776 | all | 306,776 | all |
| input embedding | 6,291,456 | 768 | 6,291,456 | 768 |
| per-loop value embedding | 8,388,608 | 1,024 | 8,388,608 | 1,024 |
| register, candidate and trust tables | 838,448 | 18,736 | 838,448 | 18,736 |
| attention output gains | – | – | 12,582,912 | 1,536 |
| token neurons: tables | – | – | 50,331,648 | 6,144 |
| token neurons: projections | – | – | 589,824 | all (589,824) |
| written library and its index | 63,768,391 | 245,223 on a retrieving step (worst case), else 0 | – | – |
| **total** | **98,300,111** | **19,278,959** (19,033,736 on a step that does not retrieve) | **98,036,104** | **19,631,240** |

The two networks share 34,531,720 numbers: the first six rows. `final` spends the other 63.8
million on text it reads only on the rare steps that retrieve (at most one passage of at most 160
tokens, placed before a user turn); `v9` spends 63.5 million on rows read at every position. The reads differ by 597,504 numbers (3.1%), of which 589,824 are the two dense
projections of the token neurons.

### 4.2 The audit

The ledger is a claim about a function, so it is audited as one. `bes/readset.py` is a NumPy
forward pass for one position that is given only the rows the ledger says are read. Three checks
(`docs/ACCOUNTING.md`, Audit): its log-probabilities agree with the JAX model to 1e-4; adding large
noise to every row it was not given leaves its output unchanged; and it logs the distinct numbers
read for every generated token, in both readings. Training computes dense products over whole
tables and masks them for speed; that changes neither the function nor the counts.

Measured with the audit on real text (ClimbMix documents with the real exact-layer features, the
critic excluded):

- `final`: 18.83 million per step in the standard reading; 19.62–19.81 million in the strict reading
  at 1,024 positions and 20.07–20.49 million at 2,048, on 6 documents (`docs/ACCOUNTING.md`).
- `v9`: 19.43 million in the standard reading (19.63 million with the critic); in the strict
  reading 21.2 / 23.8 / 24.6 million (minimum / median / maximum) at 1,024 positions and 22.4 / 26.7
  / 28.2 million at 2,048, on 40 documents of at least 2,048 tokens (`docs/ACCOUNTING.md`,
  `docs/V9.md` §10). Every distinct token of the window adds 9,536 numbers in the strict reading.

So `v9` meets the standard reading and not the strict one. The budget's primary reading is the
standard one; we report both.

### 4.3 What the ledgers do not count

Both ledgers count numbers, once, at inference. They do not count how often a number is moved, the
activations it produces, the layouts in which it is stored, or anything about training: gradients,
optimiser state, recomputation. The core's 14.1 million weights are applied 16 times per token (2
blocks × 8 loops) and counted once; a token's row of a page table (3,072 numbers) is counted once,
while its gathered copy is carried through all 8 loops and, in training, its cotangent accumulated
over them. Section 7 measures what the wall clock does charge for.

### 4.4 The allocation question as it stood before training

The evidence for spending the free 80 million one way or the other was small-scale, and we give it
as such.

- **Written library** (`docs/E1_COVERAGE.md`). Simple English Wikipedia, lead passages and
  question-answer pairs in independent xz blocks with a BM25 index, 63,768,391 numbers, behind a
  calibrated gate. After a review found confident wrong retrievals, the gate was refitted to
  precision 0.8 on every development set: coverage on NQ-open validation fell from 35.0% (precision
  54.1%) to 6.0% (76.9%), on PopQA to 0.5% (75.0%); entity look-ups stayed high (common-name
  probes 81.3% at 99.9%). The gate opens on 8 of the 200 development first turns of our prompt set
  (`docs/EVAL_PROTOCOL.md` §8), so the blind comparison can hardly see the library.
- **Token pages** (`docs/V9.md` §2). In a CPU laboratory (width 48, 2 shared blocks × 4 loops,
  1,040 token types, 12 million tokens, one seed per arm, paired noise about 0.01 nats), token-keyed
  parameters on the writing side of attention helped and on the matching side hurt (class-specific
  `W_o` −0.079, `W_v` −0.039, `W_q` and `W_k` together +0.046 nats at 12 million tokens); private
  feed-forward neurons, class experts and dense widening fell on one curve
  ΔL = −c · E_{x∼f}[ln(n(x)/n₀)], where n(x) counts the neurons available to token x, with
  c = 0.070 at 6.6 million tokens (ten arms, root-mean-square residual 0.008 nats); the complete
  page gave −0.252 nats at 12 million tokens against a base of 3.734. In the micro ablation of this
  repository (width 128, 1 million tokens) the page gave −0.61% of held-out loss, below the
  pre-registered resolution (margin 0.84%, bar 1%): by the rule, not resolved (`docs/V9.md` §8).
- **The prediction at full scale** (`docs/V9.md` §7): applying published gains of token-keyed tables
  at two to four times the dense storage (0.04–0.11 nats) to the project's own scaling-law estimate
  put `v9` at about 2.45 nats per token (2.41–2.48) against 2.507 for `final`. These are predictions
  and are not evaluated here [TBD: held-out loss of both configurations on the same rows,
  scripts/midrun_eval.py on the stopped `final` checkpoint and the last `v9` checkpoint, after
  training ends at 18:33 UTC].

## 5. The model

### 5.1 Tokens and the exact layer

Byte-level BPE with V = 8,192 ids (256 bytes, 7,920 merges, 16 special tokens). A *segment* is a
document or a conversation and starts with `BOS`; turns are `ROLE content END`. A host-side layer in
NumPy computes, for every position, causal and segment-local features of the tokens alone
(`docs/SPEC.md` §2): integer *registers* (role, code-fence and quote parity, bracket depth, word,
sentence, line and item counts, countdowns to explicit targets compiled from the user turn) and a
*candidate tape* of up to 16 candidate next tokens: the distinct successors of earlier occurrences
of the current 1-, 2- and 3-gram and of the longest earlier match, each with its exact count, the
number of distinct successors of its context, and the role of its source. The training loader and
the decoder use the same functions.

### 5.2 The dense looped core

Notation: d = 768, two weight-shared blocks i ∈ {0, 1} applied in each of R = 8 loops, H = 6 heads
of width 128 (126 content dimensions and 2 reserved), feed-forward width d_f = 3,072, N = 2 lanes.
RMSNorm(·) is root-mean-square normalisation, ⊙ the element-wise product. For the token x_t at
position t:

    e_t = RMSNorm( E[x_t] + Σ_f R_f[reg_{t,f}] + W_c Σ_k μ_{t,k} ( C[c_{t,k}] + Σ_g F_g[φ_{t,k,g}] ) )
    ℓ_1 = ℓ_2 = e                                                     (two lanes)
    for r = 1 … R:
        u   = Σ_n α_{r,n} ℓ_n
        u   = B_1( B_0( u ; E_ve[x, r] ) )
        s   = RMSNorm(u) ⊙ g_r
        ℓ_n = Σ_m M_{r,nm} ℓ_m + β_{r,n} s                             (n = 1, 2)
    h_t = RMSNorm( ℓ_1 + ℓ_2 ) ⊙ g_head
    z_t = 15 · tanh( (h_t P E_outᵀ) / 15 )                             (float32, P: 768 × 512)

Here E, R_f, C, F_g are the token, register and candidate tables, μ_{t,k} the slot mask of the tape,
and the lane coefficients start at α_r = onehot(r mod 2), M_r = I, β_r = 1 [RW: hyper-connections, looped and universal transformers, value embeddings; refs.md 14, 16–22, 5]. A block is

    B_i(u):  a = Attn_i( RMSNorm(u) ⊙ g_{1,i} ) ;   u ← u + a
             x = RMSNorm(u) ⊙ g_{2,i} ;   u ← u + W_down,i · relu(W_up,i · x)² + Π_i(x; x_t)

with no biases; Π_i is the token-neuron path of Section 5.5 (absent in `final`). The per-loop value
embedding E_ve[x_j, r] (128 numbers per loop) is added to the first 128 value dimensions of block 0
through the gate 3σ(W_g x_j[1:12]). Activations are bfloat16; normalisation statistics, the lane
mixing, the rotary angles, the logits and the loss are float32; parameters are float32 master
copies (`docs/TPU_NOTES.md`, Precision).

Figure 1: The model: the host-side exact layer, the two shared blocks applied in 8 loops with two
lanes, the head and the output law; the token pages of `v9` drawn as rows of tables beside the dense
path; each part annotated with its stored numbers and its numbers read per token from Table 2 [from
the figures agent].

### 5.3 Attention with an exact sink

Queries and keys are rotated (RoPE) and L2-normalised on their 126 content dimensions; the query is
scaled by a learned per-head gain γ_h (initially √126). The two reserved dimensions are set to
q = (1, 0) and k_j = (b_j, 0), so the logit is γ_h cos(q_t, k_j) + b_j with any attention kernel.
At a `BOS` position the content of the key is zeroed, b_j is a learned scalar σ_{i,h} and the value
is a learned vector m_{i,h} (the prior mean); elsewhere b_j = 0. The mask is causal and within the
segment, and every segment starts with `BOS`, so every query sees its segment's sink.

The sink is the design's reading of a known identity (`docs/MATH.md` B1): with exact-match one-hot
keys of infinite sharpness, a sink logit β + log λ and value m₀, the output for a query with n_a
matching keys is (Σ_{matching} v_j + λ m₀)/(n_a + λ); with one-hot successor values and a uniform
m₀ that is the posterior predictive of a Markov chain under a symmetric Dirichlet prior. In the
micro ablation the sink was neutral (−0.1% when removed) and is kept because it costs nothing and
has outside evidence [RW: attention sinks and learned sink logits; refs.md 42, 43].

### 5.4 The output law

The network's distribution q = softmax(z_t) is combined with the candidate tape in closed form
(`docs/MATH.md` T1b, `bes/model/predictive.py`):

    p(v) = ( q(v) + Σ_k m_k [c_k = v] ) / ( 1 + Σ_k m_k ),
    m_k  = exp( softcap( Σ_f β_f[φ_{k,f}] + ⟨h_t A, c_k⟩ − 6 ) ) · μ_k · s(src_k),

where the trust tables start at zero (every active slot starts at mass e⁻⁶) and s(·) scales the
masses by the role of the candidate's source. Four properties make it safe to train from step 0:
the normalising constant is 1 + Σ m, so no second pass over the vocabulary is needed; with
m_v = (n_v − d t_v)/(θ + d T) it is exactly the Pitman–Yor posterior predictive with the network as
base measure [RW: hierarchical Pitman–Yor language models; refs.md 38]; the loss can rise
by at most log(1 + Σ m) over the network's own; and with all masses zero it equals the network's
cross-entropy bit for bit (tested). Decoding applies the temperature to the base measure only,
p_T(v) = (softmax(z/T)(v) + Σ_k m_k[c_k = v])/(1 + Σ m), because tempering the whole mixture made
copy loops absorbing (Section 10).

### 5.5 Token pages (`configs/v9.py`)

Every vocabulary id owns, in each block i, ρ = 8 private ReLU² neurons in a d_k = 192-dimensional
space shared by all tokens, and a gain row on the 768 attention-output channels:

    z       = RMSNorm( x P_in,i )                                       (P_in,i: 768 × 192, shared)
    a_{t,j} = ⟨ z_t , RMSNorm( K_i[x_t, j] ) ⟩ / √192                   (j = 1 … 8)
    o_t     = Σ_j relu(a_{t,j})² · U_i[x_t, j]                          (U_i[x, j] ∈ R¹⁹²)
    Π_i     = o_t P_out,i                                               (P_out,i: 192 × 768, shared)
    attention output:  a_t = ( y_t ⊙ (1 + G_i[x_t]) ) W_o,i

K, U, G are lookup tables (8,192 × 2 × 8 × 192 numbers each for K and U, 8,192 × 2 × 768 for G);
P_in and P_out are dense and trained with the matrix optimiser. Initialisation: P_in, P_out uniform
(non-zero, as the matrix optimiser requires), K standard normal (used normalised), and **U = 0,
G = 0**. Both paths also have a run-time switch. The pages need no dispatch, no capacity factor
and drop no token: every page is one row of a table.

**Exact reduction.** With U = 0, o_t = 0 and Π_i = 0 exactly; with G = 0 the gain multiplies by
exactly 1. Hence, for every value of the shared parameters θ_s, the `v9` network with its new
parameters at initialisation computes the same function as the `final` network without its
library, in floating point and not only in exact arithmetic: a product by 1 and a sum with 0 are
exact. This is tested (`docs/V9.md` §10) and is what the graft of Section 8 relies on.

### 5.6 Parameters, groups and optimisers

Matrices train with MuonH, a port of the looped-model reference (Nesterov momentum 0.95, five steps
of a polar iteration, the update rescaled to and projected back on the sphere of the matrix's
initial Frobenius norm, no weight decay) [RW: Muon and its variants, and the looped-model reference; refs.md 51–54, 21]; all other leaves train with Adam (β = 0.8, 0.95; ε = 1e−10; no weight decay) in
groups with their own learning rates: embeddings 0.3 (the token-neuron keys train in this group),
unembedding 0.004, feature tables 0.01, token-neuron values 0.004, gains 0.01, all multiplied by
(768/640)^−0.5; the matrix rate is 0.014 before the transfer rule below (`bes/config.py`,
`bes/optim/build.py`, `configs/v9.py`).
A transfer rule of the reference multiplies every group by (tokens per step / 2¹⁸)^0.558 (here
exactly 1) and the matrix group by (horizon tokens / 9.83 × 10⁸)^−0.32, with the horizon estimated
once by the start-up benchmark: 1.3117 × 10¹⁰ tokens (run manifest; `docs/CHANGES.md` §6). The rule
was fitted by its authors over a smaller range than ours, so this is an extrapolation.

**Evidence behind each component.** Table 3 gives the pre-registered micro ablation (width 128, 8
loops, 1 million tokens per arm, seed replicate 0.42%, margin 0.84%) and what was done with it.

**Table 3. Micro ablation and decisions** (`docs/E2_ABLATIONS.md`; `docs/CHANGES.md` §14). Values
are the relative change of held-out loss when the component is removed (positive: removal hurts),
on the training mixture / chat / assistant tokens.

| component | removal | pre-registered rule | decision |
|---|---|---|---|
| candidate tape (input features and output masses) | +11.0% / +16.4% / +19.3% | keep | on |
| … the output masses alone | +8.5% / +12.6% / +14.9% | keep | on |
| … the input features alone | +0.9% / +1.0% / +1.2% | off on the mixture, on by the chat reading | on (deviation, below) |
| register embeddings | +0.8% / +1.1% / +1.1% | off on the mixture, on by the chat reading | on (deviation, below) |
| per-loop value embeddings | +1.5% / +2.2% / +2.3% | keep | on |
| confidence channel | +0.7% / +0.8% / +0.8% | off | off |
| structural rotary clocks | +0.3% / +0.4% / +0.3% | off | off |
| attention sink | −0.1% / −0.1% / −0.1% | keep (outside evidence, harmless) | on |
| 4 loops instead of 8 / 12 instead of 8 | −0.12% / +0.26% (mixture) | within noise | 8 |

Two of these decisions deviate from the pre-registered rule, and we report them as deviations. The
rule asked new components for at least 1% without saying on which loss; the automatic report used
the training mixture, by which the candidate input features and the registers would be off. We kept
them on the chat and assistant losses, a reading chosen after seeing the data. The registers also
failed their own step-function gate (exact counting: 79% with the registers against 93% for an arm
trained without them at 2 million tokens, `docs/E4_PROBES_2M.md`); by the rule they should be off,
and we kept them.

## 6. Training under a wall clock

### 6.1 A schedule that is a function of progress

The run has one absolute deadline; training must end at `train_end` = deadline − post-training
reserve. With T the training seconds of the kept trajectory and R = max(0, train_end − now), the
trainer defines progress as

    p = T / (T + R)

(`bes/train/schedule.py`; tying the decay to the time budget follows budgeted training [RW: refs.md
1–3]). While the trainer is training, T grows and R shrinks at the same rate, so
p is linear in wall time and reaches 1 exactly when the window closes, whatever the throughput.
Everything the trainer and the loader need is a function of p alone: the learning-rate factor
(a linear warm-up over the first 0.2% of p, then 1 − 0.9p, i.e. a linear decay to 0.1 at p = 1), the
sequence length (1,024 tokens, 2,048 from p = 0.67), the chat share of the mixture (0.25; 0.7 from
p = 0.92), loss on assistant tokens only from p = 0.92, and the library shares of `final`
(`configs/final.py`, `bes/config.py`). A step is 262,144 tokens.

Two consequences were observed in this run and are worth stating because they are not obvious.
A pause leaves T fixed and shrinks R, so p advances while nothing trains: the 789-second pause of
the graft moved p from 0.5049 to 0.5112 (train log). Moving the end of training later raises R, so
p falls: when the post-training reserve was cut from 220 to 118 minutes at step 12,062 and the
training end moved from 16:51:49 to 18:33:49, p fell from 0.2415 to 0.2190 and the learning-rate
factor rose from 0.783 to 0.803 (train log; `docs/CHANGES.md` §19.5). The schedule is continuous in
the parameters and the optimiser state, not monotone in p.

Figure 2: Progress p and the learning-rate factor against wall-clock time over the three runtimes,
with the gaps between runtimes, the measurement pauses, the move of the training end at step 12,062
and the graft marked [from the figures agent].

### 6.2 Surviving the loss of a machine

Checkpoints are written atomically every 25 minutes (parameters, optimiser state, data cursor,
random state, time ledger); after each one a snapshot is copied off the runtime. On a new runtime
the snapshot is restored, the deadline is moved by the downtime (which keeps p continuous, an
exception to "downtime counts against the budget" that exists only for a change of machine) and
the trainer continues. A loss guard rolls back on three consecutive spikes above 1.3 times the
moving average; it has not fired (0 rollbacks in the ledger). Restarting on a *new* machine exposed
two errors that no rehearsal had: the loader replays its journal and needs every shard the journal
lists, which a new machine must download and tokenise again (the trainer now waits for them, with
24 tokenising workers instead of 4: 363 shards in 9 minutes); and two sources that share raw files
deleted them before both had read them. Together they left the third runtime's TPU idle for about
45 of its first 48 minutes (`docs/CHANGES.md` §19.4; commits `e096dff`, `77305dc`).

### 6.3 The run as it happened

**Table 4. Events of the run** (train log, `docs/CHANGES.md` §19.3–19.6, operations notes; tokens
are steps × 262,144).

| UTC | step | tokens | p | event | live tokens/s |
|---|---|---|---|---|---|
| 23 Nov 17:46:32 | 0 | 0 | 0 | training starts (`configs/final.py`, commit `98598c2`; 16-row micro-batches) | 220,082 at step 50 |
| 23 Nov ≈19:27 | 5,075 | 1.33 × 10⁹ | 0.095 | last snapshot before the second runtime is taken back | ≈222k |
| 24 Nov 01:54:10 | 5,075 | 1.33 × 10⁹ | 0.100 | resumed on the third runtime | ≈217k |
| 02:29 | 6,647 | 1.74 × 10⁹ | 0.132 | stock-refusal rule in the loader (Section 9.2) | ≈217k |
| 03:31 | 9,332 | 2.45 × 10⁹ | 0.188 | micro-batches cut into 4 pieces on the device (Section 7.4) | 240–246k |
| 04:32 | 12,062 | 3.16 × 10⁹ | 0.242 → 0.219 | post-training time given to training; end 16:51:49 → 18:33:49 | 240k |
| 07:53 | 22,613 | 5.93 × 10⁹ | 0.401 | one-tile attention kernel and ReLU² backward (Section 7.5) | 281,374 at step 23,150 |
| 08:36 | 25,280 | 6.63 × 10⁹ | 0.440 | stock-deflection rule (Section 9.3) | ≈281k |
| 09:45:35 | 29,722 | 7.79 × 10⁹ | 0.505 | `final` stopped; graft into `configs/v9.py` (Section 8) | 280,370 at step 29,300 |
| 09:58 | 29,722 | 7.79 × 10⁹ | 0.511 | `v9` resumes | 178,523 at step 29,750 |
| 18:33:49 | [TBD: final step, at 18:34 UTC] | [TBD: final token count, at 18:34 UTC] | 1 | end of training | [TBD: live rate at 2,048 tokens, from p = 0.67 at about 12:46 UTC] |

Averaged over its 32,320.8 training seconds, the trajectory up to the graft trained at 241,000
tokens per second (7,791,443,968 tokens; trainer's ledger).

### 6.4 The training curve so far

Figure 3: Training loss against tokens for the whole run, windowed means of the logged step
losses, with the events of Table 4 marked, the 1,024 → 2,048 change at p = 0.67 and progress p on
a second axis; a second panel shows the tape gain (this section) on a logarithmic axis [from the
figures agent].

Table 5 summarises the step losses logged every 50 steps (each the mean over the step's 262,144
tokens) in windows of progress; the role split and the tape gain are those of the last micro-batch
of each logged step (16,384 tokens), pooled.

**Table 5. Training loss of `final` by window of progress** (computed by us from
`train/train_log.jsonl` of the stopped run; script `lead_work/scripts/windows.py`).

| p | logged steps | step loss: mean (sd) | positions with role "assistant" | positions outside conversations | tape gain (nats) |
|---|---|---|---|---|---|
| 0.00–0.01 | 10 | 3.613 (0.507) | 2.835 | 3.823 | 0.318 |
| 0.05–0.10 | 48 | 2.699 (0.047) | 1.717 | 2.936 | 0.035 |
| 0.15–0.20 | 48 | 2.613 (0.041) | 1.605 | 2.861 | 0.026 |
| 0.25–0.30 | 60 | 2.551 (0.041) | 1.551 | 2.792 | 0.024 |
| 0.35–0.40 | 56 | 2.530 (0.046) | 1.521 | 2.771 | 0.023 |
| 0.45–0.50 | 68 | 2.496 (0.037) | 1.491 | 2.738 | 0.022 |

These are training losses on a stream whose composition changed at steps 6,647 and 25,280, not
held-out losses, and windows are of progress, not of tokens (p fell once, at step 12,062). The
held-out measurement is [TBD: held-out loss by domain of the checkpoints of steps 5,075, 12,062,
22,613, 29,722 and of the final export on the export's held-out rows, scripts/midrun_eval.py, after
training].

One observation is robust to these caveats. The *tape gain*, the mean of −log q(y) − (−log p(y)) on
training tokens (what the output law's masses buy the trained model over its own network
distribution), fell from 0.318 nats in the first 1% of training to 0.022 nats (0.9% of the loss)
by p ≈ 0.5. In the micro ablation the same switch, turned off at evaluation, cost 10.5% of the loss
at 1 million tokens (`docs/E2_ABLATIONS.md`, evaluation-time switches). One reading, which the
micro report anticipated (its "Limits"), is that at scale the network learns most of what the exact
in-context statistics offered it early on. The tape gain measures how much the trained model leans
on the tape, not what training without it would give.

### 6.5 After training

The 118 minutes after training are planned as follows (`docs/CHANGES.md` §19.5; `bes/train/prefopt.py`,
`PrefCfg` in `bes/config.py`). Sampling (3 minutes) adds a few pairs of the model's own replies.
Preference optimisation (60 minutes) runs one epoch of DPO against the frozen trained model
(β = 0.1, 64 pairs per step, plain Adam on every leaf, gradient-norm clip 1.0), almost entirely on
UltraFeedback pairs, tries the learning rates 1e−6 and 3e−6 from the same start, and accepts a
result only if held-out preference accuracy improves beyond its standard error, held-out assistant
loss rises by at most 3%, and sampled length and repetition stay within fixed bands; otherwise the
trained weights ship [RW: direct preference optimisation; refs.md 74]. Decoding selection (35 minutes,
Section 10.3), export (5 minutes: 16-bit weights checked against float32 on held-out rows) and a
15-minute buffer follow. Our replies to the test prompts are then generated on the runtime's host
cores (`scripts/gen_test_replies.py`; planned at 25–35 minutes on 42 workers, `testgen.md`).
[TBD: what each stage did, from the stage ledger at 20:32 UTC.]

## 7. Where the time goes on one TPU v6e

### 7.1 Method

Two kinds of measurement, kept apart. **On the chip**: the live run was paused, the chip left idle,
and programs were timed on synthetic batches with the first calls discarded; we report medians
(30 calls for one piece in Section 7.5) and, for whole steps, the scan over all pieces of a step
with its float32 gradient sums (`scripts/tpu_micro.py`, `scripts/tpu_kernels.py`,
`scripts/tpu_tune.py`; raw outputs in `docs/tpu_profile/` and the workflow directory). The four
pauses cost 6.7, 10.8, 9.4 and 13.2 minutes of TPU time (`docs/CHANGES.md` §19.5, §19.6; the
operator's resume log). **Compile-only**: libtpu 0.0.50, the chip's version, compiles for
`v6e:1x1` on a CPU host; we read the compiler's estimated cycles per fusion and bytes per operation
from the optimised program, and the executed VLIW bundles of each Pallas kernel. On two shapes four
times apart the estimated cycles matched the chip at one rate, 0.521 ms per million cycles
(`speed_mlp_block.md`). We use compile-only numbers to rank and attribute, never as results;
Section 7.8 checks them against the chip.

Every change that was adopted computes the reference function to floating-point rounding: by tests
on the CPU (bit for bit where we say so) and on the chip by the loss (differences ≤ 1e−6) and by
gradient differences no larger than those of a change of the compiled graph that leaves the
arithmetic alone (Section 7.5).

### 7.2 The dense core is not bound by its matrix products

The first numbers from the chip (23 November) were a third of the plan's assumption: `final`
trained at 112–117 thousand tokens per second and `v9` at 88–92 thousand, where the plan had assumed
300–450 thousand (`docs/TPU_NOTES.md`, "Where a step spends its time"; `docs/PLAN.md` §6). The
suspects, measured one at a time on the idle chip with a 32 × 1,024 micro-batch: a fixed cost per
micro-batch (no: larger micro-batches were slower); the sequence length (the quadratic part is
about 6%); rematerialisation (4.5%); the float32 vocabulary head (5.8 ms of a 289 ms micro-batch);
the optimiser (1.5 ms of a 2.34 s step); gradient accumulation (the scan of 8 micro-batches is 8.0
times one); the candidate tape, the registers, the value embeddings and the sink (each within 5%
when removed, 10% for the sink); dense attention instead of the flash kernel (303 ms against 289
ms). One loop instead of eight took 42.9 ms against 289 ms: 97% of the time is in the 16 block
applications.

Inside one block application (15.4 ms for forward and rematerialised backward): the matrix products
took about 2.4 ms, the attention kernel 3.4 ms, and **the other 9.6 ms (62%) was glue**: slices and
concatenations of the rotary halves and of the reserved dimensions, transposes between the model's
`[B, T, H, d_h]` and the kernel's `[B, H, T, d_h]`, and float32 copies of the widest tensors. The two
feed-forward products alone ran at 814 TFLOP/s in the forward pass; a single 768 × 768 product took
0.23 ms. On this chip one pass over an activation tensor costs about as much as the d × d product
that made it.

Figure 4: Time of one block application on the chip, split into matrix products, attention kernel
and glue, for the reference implementation and after the rewrites of Table 6 [from the figures
agent].

### 7.3 Removing the glue without changing the function

**Table 6. One `value_and_grad` of a 32 × 1,024 micro-batch on the chip, `configs/final.py`**
(`docs/TPU_NOTES.md`; loss 8.33353 for every variant).

| implementation | ms | tokens/s |
|---|---|---|
| reference | 289.3 | 113,264 |
| rotation as two products; reserved dimensions as zero columns and broadcast constants (no slice, no concatenation) | 215.0 | 152,379 |
| … operands built in the kernel's layout, `W_o` as one contraction over (head, dimension) (no transpose) | 194.9 | 168,155 |
| … feed-forward pre-activations stored in bfloat16 | 181.8 | 180,248 |
| … lanes carried as a tuple | 179.3 | 182,779 |
| … kernel tiles 1,024 / 1,024 / 512 with the fused backward kernel | 162.0 | **202,245** |

The rotation is written as `rotate(xW) = (xW)·(cos, cos) + (xW′)·(sin, −sin)` with W′ the weight
whose two rotary halves are exchanged per head, so the rotated operands come out of products
instead of slices. All of it equals the reference to rounding in float32 and bfloat16 (28 tests,
`tests/model/test_attention_major.py`), with one stated exception: storing the feed-forward
pre-activation in bfloat16 adds one rounding (2⁻⁹ relative, unbiased) before the square, and saves
28% of the feed-forward sub-layer's time (2.97 → 2.13 ms per application; `docs/CHANGES.md` §19.1,
`bes/model/layers.py`). On whole training steps the new code ran at 201,877 tokens per second
against 111,893 for the old (1.299 s against 2.343 s per step; `docs/CHANGES.md` §19.2).

### 7.4 Smaller pieces of the same micro-batch

Halving the micro-batch from 32 to 16 rows was 8% faster (`docs/CHANGES.md` §19.3). The micro-batch
size, however, is fixed for a run in progress (it enters the configuration hash, the loader's
shapes and the preference optimisation), so the further cut was made on the device:
`accumulate(..., micro_split=k)` reshapes `[A, B, …]` to `[A·k, B/k, …]` before the scan. Rows are
independent and the accumulation is weighted by tokens, so the update equals the original to
rounding (tests with a stand-in model and the real loss; old checkpoints continue).

**Table 7. Whole accumulation step of 262,144 tokens on the chip** (`docs/CHANGES.md` §19.5).

| pieces × rows × tokens | ms | tokens/s | | pieces × rows × tokens | ms | tokens/s |
|---|---|---|---|---|---|---|
| 16 × 16 × 1,024 | 1,173.7 | 223,348 | | 16 × 8 × 2,048 | 1,348.8 | 194,351 |
| 32 × 8 × 1,024 | 1,081.5 | 242,393 | | 32 × 4 × 2,048 | 1,284.3 | 204,121 |
| 64 × 4 × 1,024 | 1,063.3 | 246,538 (+10.4%) | | 64 × 2 × 2,048 | 1,218.2 | 215,192 (+10.7%) |

The live rate went from 217 to 240–246 thousand tokens per second. A compile-only reading of the
program at 16 × 1,024 rows shows the compiler keeping both lanes in on-chip memory for the whole
forward pass, so that the lane arithmetic moves no bytes to device memory (`speed_loops.md`); at
32 × 1,024 the 50 MB lanes do not fit next to the blocks. This is consistent with the measurements
and is not itself a measurement; we did not establish which tensors account for the gain of the
4-row pieces. After the kernel change of Section 7.5, a further halving was
slower (128 × 2 × 1,024: 958.9 ms against 884.5; 128 × 1 × 2,048: 1,013.1 against 983.9), and the
pieces stayed at 4 × 1,024 and 2 × 2,048.

### 7.5 An attention kernel that walks only the triangle

At these pieces the library's splash kernel sees one or two 1,024-key tiles per (row, head) slice
but pays for the machinery of a general block-sparse kernel: an online softmax, the causal mask
computed over whole tiles of which half are discarded, running maximum, sum and output written to
device memory, and a log-sum-exp written 128 lanes wide [RW: flash and splash attention, Pallas; refs.md 58–61].
The one-tile kernels (`bes/model/attention_onetile.py`) hold the slice in on-chip memory, cover only
the causal triangle in strips of 256 query rows (62.5% of the score elements at 1,024 tokens),
mask the off-diagonal pieces by segment only, take each strip's softmax exactly in one pass and
write the log-sum-exp as one lane-dense row; the backward pass runs key-major strips. A flag `save`
keeps the forward kernel's output and log-sum-exp through the loop's rematerialisation, so the
forward kernel runs once per block application instead of twice. In executed bundles of the chip's
own compiler, `onetile:save` needs 0.34 of the configured kernels' work at 1,024 tokens and 0.42 at
2,048 (`docs/TPU_NOTES.md`, "Attention kernels for one tile"). A hand-written backward pass for
relu(h)², which removes a bit mask that the compiler packed into the rematerialised product, is
equal bit for bit on the CPU.

**Table 8. One `value_and_grad` of one piece on the chip, median of 30** (`docs/CHANGES.md` §19.6;
`chip_kernels_0745.txt`).

| kernels and switches | 4 × 1,024 (ms) | speed-up | 2 × 2,048 (ms) | speed-up |
|---|---|---|---|---|
| configured splash kernels | 17.745 | – | 19.845 | – |
| `onetile` | 15.268 | 1.162 | 17.605 | 1.127 |
| `onetile:save` | 15.130 | 1.173 | 16.498 | 1.203 |
| `onetile:save` + ReLU² backward (**adopted**) | 14.639 | 1.212 | 16.275 | 1.219 |
| … + cross-entropy backward | 14.816 | 1.198 | 16.176 | 1.227 |
| splash with `save` | 16.800 | 1.056 | 17.920 | 1.107 |
| splash with its I/O in on-chip memory | 17.177 | 1.033 | 19.762 | 1.004 |

The whole step with `onetile:save` and both backward passes took 884.5 ms at 64 × 4 × 1,024
(296,371 tokens/s) and 983.9 ms at 64 × 2 × 2,048. Live, from step 22,613: 281,374 tokens per second
at step 23,150 against 239.4–239.7 thousand before (+17.5%), with the loss in the same band.

On the chip the new kernels' losses differed from the reference by at most 1e−6. Their gradients
differed, for the worst parameter leaf, by 1.4e−2 (maximum) and 1.5e−2 (L2) relative at 1,024 tokens
and 7.5e−2 and 5.4e−2 at 2,048. That is not evidence of an error: the same kernel with only the
rematerialisation policy changed differs by 1.4e−2 and 1.2e−2 (and 4.7e−2, 3.3e−2), and the backward
passes that are bit for bit equal on the CPU differ by 1.1e−2 and 3.9e−2 on the chip; only the
variant that leaves the rest of the compiled graph untouched (splash I/O) differs by 0. On the chip,
changing the shape of the compiled graph moves the bfloat16 rounding points. Logical equality is
established on the CPU in interpret mode (within 1e−5 of the configured kernels in float32, against
a float64 dense reference, with segment boundaries on and next to every strip boundary;
`tests/model/test_attention_onetile.py`). The independent review of the kernel had not returned
when it went live; it later changed the switch's handling of long sequences and typos, not the
kernel (`kernel.md`).

Figure 5: Live training rate (tokens per second) against step over the whole run, with the changes
of Table 4 marked (pieces at step 9,332, kernel at step 22,613, graft at step 29,722) and the rate of
the reference implementation on 23 November as a reference line [from the figures agent].

### 7.6 What did not help

**Table 9. Measured on the chip and not adopted** (sources in `CLAIMS.md`).

| change | measurement | why it was expected to help |
|---|---|---|
| larger micro-batches (64, 128 rows) | 110,855 and 104,576 against 111,893 tokens/s | amortise a fixed cost per micro-batch |
| no rematerialisation (half micro-batch) | 159–167 thousand against 202 thousand tokens/s after the rewrite; 188,907 against 216,681 in the run's benchmark | skip the recomputed forward |
| keep the attention kernel's outputs through rematerialisation (`remat_keep=kernel`) | 161.2 against 158.3 ms | skip the second forward kernel |
| one loss chunk per micro-batch | 1.2% faster, below the 1.5% adoption bar | fewer passes over the logits |
| one 2,048-key tile at 2,048 tokens | 177.5 against 176.1 ms | fewer tiles |
| one product per head, stacked on the head axis | 274.6 ms per micro-batch | avoid a contraction |
| feed-forward entirely in bfloat16 | 184.5 against 181.8 ms | halve the bytes |
| kernel tiles of 256 | 2.2 times slower than 512 for the kernel alone | finer causal skipping |
| scoped on-chip memory limit 8 / 16 / 64 / 96 MiB | 1,295.9 / 1,159.1 / 1,103.7 / 1,322.2 ms against 1,062.5 ms (default) | more room for the compiler's buffers |
| pieces of 2 rows at 1,024 / 1 row at 2,048 with the new kernel | 958.9 / 1,013.1 against 884.5 / 983.9 ms | the trend of Table 7 |
| hand-written cross-entropy backward | 1.2% slower at 1,024, 0.6% faster at 2,048 per piece | one pass less over the float32 logits |
| for `v9`: the scatter-free `sorted` table gradient | 1,789.0 against 1,433.1 ms (`take`) per step | 16 times fewer multiply-adds than one-hot |
| for `v9`: pieces of 8 rows | 1,491.5 against 1,433.1 ms | fewer per-piece table costs |

Two items in this table contradicted our own prior written expectation (`remat_keep=kernel` and
the memory limit, `docs/CHANGES.md` §19.3, §19.5). Compile-only analyses rejected further candidates
before they cost chip time: carrying the next loop's input through the scan (+4.9% estimated cycles,
because a third 25 MB tensor in the carry overflows on-chip memory), folding output conversions
into the products (no change in the compiled program), and others below 1% (`speed_loops.md`,
`speed_mlp_block.md`, `speed_io.md`).

### 7.7 How far from the arithmetic

Counting multiply-adds from the configuration (`lead_work/scripts/flops.py`), `final` needs 0.781
GFLOP per training token at 1,024 tokens (forward and backward, causal attention, no recomputation)
and 1.132 GFLOP counting what the chip executes (the rematerialised forward of the loop body and
attention over whole tiles). At 281,374 tokens per second that is 220 and 318 TFLOP/s, 27% and 39% of
the 814 TFLOP/s the same chip delivered on the feed-forward products alone. At the earlier rate of
246 thousand tokens per second, matrix products and attention kernels together were about half of
a step (`docs/CHANGES.md` §19.5); the rest is data movement that no rewrite of ours removed.

### 7.8 The cost of lookup parameters

Table 10 compares the two configurations on the same chip with the same kernels (`onetile:save`
and the ReLU² backward), the same table-gradient mode (`take`, a scatter-add) and the same pieces.

**Table 10. `final` against `v9` on the chip** (`docs/CHANGES.md` §19.6; `v9_measA.txt`,
`v9_measB_sorted.txt`, `v9_measC_onehot.txt`; live rates from the train logs; arithmetic from
`lead_work/scripts/flops.py`).

| quantity | `final` | `v9` | ratio |
|---|---|---|---|
| numbers read per token (standard reading) | 19,033,736 | 19,631,240 | 1.031 |
| analytic training FLOP per token (1,024 tokens, no recomputation) | 0.781 G | 0.809 G | 1.036 |
| one `value_and_grad` of a 4 × 1,024 piece | 14.639 ms | 22.79 ms | 1.557 |
| whole step 64 × 4 × 1,024 | 884.5 ms (measured with the cross-entropy backward on) | 1,433.1 ms | 1.620 |
| whole step 64 × 2 × 2,048 | 983.9 ms (same) | 1,510.2 ms | 1.535 |
| live training rate | 280,370 tokens/s (step 29,300) | 178,523 tokens/s (step 29,750) | 1.570 |

So 3.1% more reads and 3.6% more arithmetic cost 54–62% more time per token. The page rows of a
piece are gathered once and read in every loop, and their cotangents accumulate over the 8 loops;
the compiler's cost model attributes `v9`'s extra 18.7 million cycles per piece to the token
neurons' per-token contractions (5.7 million), relayouts of the table gradients in the float32
accumulation (2.9), the accumulation of the closed-over rows' cotangents in the loop scan (2.4), the
feed-forward products that share their input (1.7), the relayout of the tables for the gather
(1.2), the gathers and scatter (0.9), the dense page projections (0.9), key normalisation and casts
(1.1), the query, key and value projections (0.3), unattributed (1.1), and the attention gain (0.2)
(`v9_speed.md`; compile-only estimates, not measurements). The contractions are
`[8, 192]` blocks per token, which the chip's tiles pad from 8 to 16 sublanes and from 192 to 256
lanes, so they move 2.7 times the bytes they hold. The gain table is nearly free; the token neurons
are the cost; and the cost is layout, not arithmetic.

Figure 6: Step time of `final` and `v9` on the chip for the three table-gradient modes, with the
compiler cost model's attribution of `v9`'s extra time by part [from the figures agent].

The same additive cost was visible before the core was rewritten: on 23 November `v9` ran at 79–82%
of `final` (88–92 against 112–117 thousand tokens per second), about 0.6 s more per 262,144-token
step; today it is 0.55 s more per step (1,433.1 against 884.5 ms). As the core became faster the
pages did not, and the ratio fell from about 0.8 to 0.62.

**The cost model against the chip.** The compiler's model predicted `v9` at 1,508 ms per step with
`take` (measured 1,433.1: 5% high); it ranked `take` below `sorted` (and, on the tables alone,
below `onehot`) and 4-row below 8-row pieces, as the chip did; it predicted `sorted` to cost 130 ms
more than `take`, and the chip charged 356 ms (2.7 times the prediction). The design document's argument for `sorted`, that it needs 16
times fewer multiply-adds than a one-hot gradient (`docs/V9.md` §3), was correct as arithmetic and
wrong as a prediction of time: the scatter-add was the fastest of the three. A further switch that
keeps the token tables as `[V, W]` matrices through the accumulation is predicted to save 46 ms per
step and is exact bit for bit on the CPU [TBD: on-chip measurement of the `flat` switch, at the next
pause if one is taken], and two layouts that avoid the padding are described and estimated at 100–130
ms per step, not implemented (`v9_speed.md`).

### 7.9 Decoding is bound by the host

On the chip, one decoding step took 12.1 ms for 6 rows and 34.5 ms for 24; a complete particle
decode with 6 particles took 43.1 ms per position, of which about 31 ms was host-side Python
(sampling, constraints, exact features) (`docs/CHANGES.md` §19.3). Here too the arithmetic is not
the cost. [TBD: decoding throughput of the final export on the runtime's host cores and on the chip,
measured during the test-reply generation after the export.]

## 8. Changing the design mid-run: an exact graft

### 8.1 The decision

At 09:16 UTC on 24 November the user asked to switch to `configs/v9.py`. The run was at p ≈ 0.50 with
7.79 × 10⁹ tokens trained; 30,969 seconds of training were left (the trainer's slice at the
restart). At `v9`'s measured rates (178.5 thousand tokens per second live at 1,024 tokens; on
synthetic batches its step is 5% slower at 2,048 tokens, 1,510.2 against 1,433.1 ms), a `v9` started
from scratch in that time would have seen about 5.4 × 10⁹ tokens (5.53 × 10⁹ if all of it ran at 1,024
tokens; `lead_work/scripts/projections.py`), fewer than the 7.79 × 10⁹ the run already had. We
therefore continued the run in the new parameter tree.

The switch was not decided by a pre-registered rule. A comparison of the two configurations had
been pre-registered on 23 November (75 minutes of each, decided on held-out assistant loss at equal
training time; `docs/CHANGES.md` §19). It was not run: first both configurations measured three times
slower than planned and the time went into finding out why (§19.1); then the comparison was judged
not worth about a tenth of the training time, because by the design's own numbers the expected gain
of the pages and the tokens lost to their slower step roughly cancel (§19.2), and `final` was
trained. The switch to `v9` at p = 0.505 was the user's decision.

### 8.2 Function-preserving continuation

Let the target configuration have parameters θ = (θ_s, θ_n), where θ_s are the leaves the source also
has (same path, shape and type) and θ_n the new ones. Call an initialisation θ_n⁰ *neutral* if

    f_target( · ; θ_s, θ_n⁰ ) = f_source( · ; θ_s )      for every θ_s.

For `v9`, U = 0 and G = 0 make θ_n⁰ neutral, whatever K, P_in and P_out are (Section 5.5). At the
graft point (θ_s*, θ_n⁰) four things follow.

1. Every loss equals the source's, on every batch.
2. The gradient with respect to the shared parameters equals the source's,
   ∇_{θ_s} L_target = ∇_{θ_s} L_source, because the identity holds for all θ_s and so does its
   derivative in θ_s. Given the carried optimiser state, the first update of θ_s on a given batch
   and learning rate is the update the source would have made (there is no gradient clipping).
3. Of the new leaves only U and G receive a non-zero gradient at the first step:
   ∂L/∂U_i[x, j] = relu(a_{t,j})² · (∂L/∂Π_i) P_out,iᵀ and ∂L/∂G_i[x] = y_t ⊙ ((∂L/∂a_t) W_o,iᵀ),
   summed over the positions holding x; while ∂L/∂K, ∂L/∂P_in and ∂L/∂P_out are exactly zero, since
   each is a product with U = 0 or with o = 0. The trajectories therefore separate only through U and
   G after the first update (and through the data stream).
4. Adam on the new leaves runs with the carried update count (about 29,700), so its bias corrections
   are 1. For a constant gradient its update after t steps is lr · (1 − β₁ᵗ)/√(1 − β₂ᵗ) · sign(g)
   where a fresh run takes lr · sign(g): 0.894 times as large at the first update, 1.44 times at the
   seventh (the maximum), 1.04 times at the fiftieth (β = 0.8, 0.95; our arithmetic).

Item 2 is what makes the graft a continuation and not a restart: until U and G move, the shared
parameters take the steps the source would have taken on the same batches.

### 8.3 What is carried and what is new

`scripts/graft_run.py` (commit `9a4d652`) matches leaves by path, shape and type and refuses to write
anything if a shared leaf differs, if the destination exists, if there is no checkpoint, or if the
checkpoint was written under another configuration hash. From the source it carries the parameters
and optimiser slots of the 59 shared leaves (34,531,720 numbers; for the matrix optimiser the cached
initial norms are part of the slots), the update count, the step, the trainer's time ledger and loss
guard, and the run's manifest (budget, deadline, the benchmark's execution plan and its token horizon
of 1.3117 × 10¹⁰, the post-training plan, the piece size and the speed switches) with the
configuration replaced. The 5 new leaves (63,504,384 numbers: `page/key`, `page/val`, `page/pin`,
`page/pout`, `gain/o`) start at their initialisation with fresh slots.

The data stream cannot be carried, because the mixture changed (no library rows, no library tasks).
It starts anew with another seed. A stream that starts in the middle of a run needs two corrections
that a stream started at p = 0 does not: the loader estimates the remaining horizon as
tokens / p − tokens, which for a new journal at p ≈ 0.5 badly underestimates what is left (the commit
records that without the correction the estimate was 40 times too small one block after the graft,
so small sources would have been read out at once), and every source may still be read
max_epochs × (1 − p) = 8 × 0.495 = 3.96 times in the part that is left. Both are passed to the loader
from the graft record (`Loader(tokens_before = 7,791,443,968)`; `scripts/run_24h.py`).

### 8.4 Verification

`--check` evaluates the loss of the source parameters under the source configuration and of the
grafted parameters under the target configuration on one synthetic batch of 2 × 256 tokens (dense
attention, float32, on the CPU) and refuses unless they agree to 1e−6 relative.

**Table 11. The graft's function check.**

| source checkpoint | where | source loss | grafted loss | difference |
|---|---|---|---|---|
| step 26,890 (snapshot) | development container | 11.2808027 | 11.2808027 | 0.0 |
| step 28,501 (live run) | runtime host CPU, next to the trainer | 11.3348608 | 11.3348608 | 0.0 |
| step 14 (smoke-size rehearsal) | development container | 9.2843390 | 9.2843390 | 0.0 |
| step 29,722 (the switch) | runtime | [TBD: saved output of the check of the live graft; the saved output of a graft of step 29,722 into a `sorted` variant carries no check record] | | |

Both checked grafts use the live run's configuration hash (`f29bfeb3ebf353a8`). An independent
review ran four tests on a fabricated run (the check within 1e−6; every carried leaf equal; every new
leaf equal to a fresh initialisation, `page/val` and `gain/o` exactly zero; the refusals; a trainer
that restores the grafted directory and takes a finite step that moves `page/val` off zero) and
found no blocker; it found two medium risks, of which one (the loader's horizon) was fixed in the
committed tool and the other (a mirror of the run directory that would have overwritten the source's
last checkpoint) was avoided by keeping the stopped run on the runtime and as a local snapshot
(`graft_review.md`). [TBD: the commit that adds `tests/train/test_graft_run.py`; at the time of
writing it is in the working tree only.]

### 8.5 What changed at the switch

- **Step time**: 1.57 times longer per token live (Table 10).
- **Progress and learning rate**: the pause lasted 789 s, booked as 727 s lost and 64 s of set-up;
  p went from 0.5049 to 0.5112 and the learning-rate factor continued on 1 − 0.9p (0.5394 at step
  29,750).
- **Data**: no library rows or library tasks; the tape mass per position (the mean over a
  micro-batch) fell from 14.7–15.7 (medians over the logged steps of p ∈ [0.40, 0.50)) to 0.89–1.31
  in the four readings after the graft. The drop coincides with the removal of the library rows,
  whose passages offer long verbatim matches; we have not measured how much each removed kind of row
  contributed. Four small sources were capped by the re-planned epoch budget (train log).
- **Loss**: the four step losses logged after the graft, 2.4118, 2.5704, 2.4896 and 2.4423 (mean
  2.479), lie inside the range of the 68 readings of `final` in p ∈ [0.45, 0.50) (mean 2.496, standard
  deviation 0.037, range 2.416–2.570). They show neither a gain nor a loss, and on a changed stream
  they could not.
- **Tokens**: projected over the 30,969 s that were left, with the last 67.5% of that time at 2,048
  tokens, `v9` will see about 5.3 × 10⁹ more tokens (1.31 × 10¹⁰ in all), where `final` at its
  measured rates would have seen about 8.1 × 10⁹ more (1.59 × 10¹⁰ in all): about 2.8 × 10⁹ fewer,
  17% of `final`'s projected total (`lead_work/scripts/projections.py`; a projection, [TBD: tokens
  actually trained by 18:33:49 UTC, from the trainer's ledger at the end of training]).

### 8.6 What one run cannot show

The graft lets a run change its design without discarding what it has learnt. It does not let us
say whether the change was right. We observe neither `final` continued to the end, nor `v9` from
scratch, nor `v9` grafted at another point, and the run differs from each of them in configuration,
in the tokens it will see (about 2.8 × 10⁹ fewer than `final`, by projection), in its data stream after the switch,
and in the early dynamics of its new leaves. Any comparison of the shipped model with the opponents
measures all of these together. A held-out comparison of the grafted checkpoints with the stopped
`final` checkpoint on the same rows [TBD: scripts/midrun_eval.py on the stopped `final` checkpoint
(step 29,722) and the `v9` checkpoints, after training] compares models after different amounts of
training and does not remove the confounding. With one seed we cannot separate any of it from noise.

The construction is close to function-preserving network growth [RW: Net2Net, network morphism,
staged training, block expansion, sparse upcycling; refs.md 45–49]; staged training in particular
already carries the whole training state, optimiser state and learning-rate schedule included, through
loss- and dynamics-preserving growth operators (refs.md 47). What is particular here is narrower: the
added parameters are of a different kind (per-token lookup pages) rather than more width or depth, the
continued schedule and data plan are those of a wall-clock budget (progress p, the loader's horizon and
epoch caps), and the function is checked numerically on the live run's checkpoint before the switch.

## 9. Data, and hygiene driven by the model's failures

### 9.1 Sources and the realised mixture

The registry (`bes/data/sources.py`, `docs/DATA_SOURCES.md`) holds 26 entries, each checked against
the Hub on 21 November 2026 (commit, file list, sizes, licence) and pinned. Weights are relative
within a kind. **Base text**: ClimbMix 0.85, FineWeb-Edu 0.10, Simple English Wikipedia 0.05.
**Chat**: smol-smoltalk 0.30, SmolTalk Magpie-Ultra 0.20, Tulu 3 SFT 0.15 (without its non-commercial,
non-English and hard-coded identity subsets), SYNTH closed-book 0.12, UltraChat 0.08, and five
smaller SmolTalk subsets (constraints, rewrite, summarise, everyday, system chats). **Retrieval
format**: SYNTH open-book 0.30, glaive RAG 0.15, RAG-Instruct 0.15, SQuAD v2 0.15, HotpotQA 0.10,
Dolly 0.08, neural-bridge 0.07. **Maths**: Orca-Math 0.65, GSM8K 0.35 with its annotations turned into
calculator spans whose results the host computes. **Preferences**: UltraFeedback (binarised, ties
dropped). [RW: the datasets; refs.md 12, 76, 77]

Rows are 75% base text and 25% conversations. Averaged over the logged loader blocks of `final`, the
realised token shares were: base text 75.0%, chat 14.5%, retrieval format 5.3%, maths 3.1%, library
tasks inside chat rows 1.3% and library rows 0.9% (train log, `loader_block.subkind_share`). After the
graft the library shares are 0 and the four logged blocks hold 75.5–75.9% base text, 14.9–16.0% chat,
4.2–5.9% retrieval format and 2.7–4.4% maths. 12% of the chat samples get a verified constraint
written into their prompt from what the reply actually does (hindsight relabelling with the exact
layer; `bes/data/hindsight.py`, `DataCfg.hindsight_frac`).

### 9.2 Stock refusals

At step 5,075 (p ≈ 0.1) a CPU sample of the snapshot with six particles answered "Write a product
review of a pair of running shoes in under 50 words" with "I'm sorry, but I can't assist with that.
It's important to respect user privacy and confidentiality …"; all six particles refused
(`midrun_sample_step5075_p6.md`; `docs/CHANGES.md` §19.5; the prompt is test prompt instruct-034, see
Section 11). The cause is in the data: the Tulu 3 mixture holds safety refusals from
CoCoNot, WildJailbreak and WildGuardMix, about one eighth of its conversations, and a model of this
size does not learn which requests they belong to. A scan of 30,000 tokenised conversations per chat
source found an assistant turn opening with a stock refusal in 4.85% of Tulu 3's conversations, 0.83%
of SYNTH's, 0.67% of the SmolTalk system chats and 0.41% of UltraChat's (`deflection_check2.json`,
summary `deflection_out.txt`). From step 6,647 the loader skips any conversation in which an
assistant turn opens with a stock refusal (commit `566517a`); the exact scorer counts such a reply as
failed, so the decoder's selection prefers a particle that answers; and preference sampling skips
prompts whose reference reply is a refusal.

### 9.3 Stock deflections

At step 22,613 (p = 0.40) the model answered "My phone fell into a sink full of water. What should I
do right now?" with "I'm sorry, but I don't have access to the latest updates on the latest phone
fires …" (`midrun_sample_step22613.md`; test prompt chat-017). A scan of how assistant turns open in
every tokenised source
(`openings.json`, 40,000 segments per source) located it: in the SmolTalk system chats 10.60% of the
turns open with an apology, 2.14% with "As an AI …" and 1.51% with "I don't have access …"; in
UltraChat 2.81% open with "I don't have …" and 2.21% with "I don't have access …". A rule wider than
the refusal rule, `bes.eval.exact_score.stock_deflection` (the reply opens by saying what the
assistant is or lacks, or with "I'm sorry, but …" whatever follows), removes from the training
stream, from step 25,280 (commit `4f4ef45`), every chat conversation with such a turn. On 30,000
conversations per source it removes 29.10% of the system chats (29.81% of their tokens), 9.06% of
UltraChat (7.10%), 1.23% of Tulu 3 beyond its refusals (1.10%), and under 1% of every other chat
source; retrieval, maths and text sources are untouched, because "the passage does not say" is the
task there (`deflection_out.txt`).

### 9.4 What this does and does not show

Both rules are functions of the data, so the stream stays reproducible, and both were triggered by
the model's own failures rather than by a list written in advance. Their effect on the model is not
measured: there is no ablation, the second rule acted only on the last part of `final` and on `v9`, and
the anecdotes that triggered them are single prompts. [TBD: the share of test replies of ours and of
both opponents flagged as stock refusal or deflection by `exact_score`, after the test replies exist.]
Two consequences must be stated plainly. From step 6,647 on the model is not trained to refuse
anything; this was the operator's decision, reported to the user, who could have overruled it
(`docs/CHANGES.md` §19.5), and the evaluation does not test the model's behaviour on harmful requests.
And the preference pairs from UltraFeedback are not filtered by either rule (2.44% of its scanned
responses open with "As an AI …", `openings.json`).

## 10. Decoding

### 10.1 Temperature on the base measure only

With the output law of Section 5.4, tempering the whole mixture sharpens the tape's share M/(1 + M),
which is a calibrated probability, and makes copy loops absorbing; this is the prediction the
negative result on exact Bayesian attention made (an exact Bayesian predictor endorses a repetition
it has seen: after r repeats its continuation probability is (r + λ/V)/(r + λ), and below
temperature 1 the probabilities of escaping form a convergent series; `docs/NEGATIVE_RESULTS.md` §1).
The decoder therefore tempers the network's distribution only, and multiplies the masses of
candidates whose source is the reply being written by a self-evidence weight of 0.25. On the micro
model (48 held-out chat prompts, `docs/E5_SELF_GENERATION.md`), at temperature 0.5: tempering the
mixture gave a repeated-4-gram rate of 0.326, loops in 13 of 48 replies and 61.4% of tokens copied
from the prompt; tempering the network only with self-evidence 1 gave 0.134, 1 of 48, 6.8%; with
self-evidence 0.25, 0.033, 0 of 48, 5.5%; with the tape off, 0.032, 0 of 48, 0%. Training itself keeps
the self-evidence masses at weight 1: trained with 0.25 the model learns the weight back (+0.07% loss,
no effect left for the decoder), trained with 0 its chat loss is 8.6% worse (`docs/E2_ABLATIONS.md`).

By the letter of the pre-registered rule every tape setting failed E5, because its copy share
(0.04–0.06) exceeded that of a model trained without the tape (0.000). The dataset's own replies,
measured the same way, copy 0.074 ± 0.008 of their tokens; we read the rule's copy criterion as wrong
rather than the mechanism as failed, a reading made after seeing the data (`docs/CHANGES.md` §12).

### 10.2 Particles, exact potentials and satisficing selection

Each reply is drawn as K = 6 particles of the same model. The proposal is the network's distribution
multiplied by token-local exact potentials (structure, explicit counts compiled from the user's
words, a budget that keeps every particle able to finish, with reachability pruning), renormalised;
the incremental weight of a particle is the probability mass its mask allowed, which makes the
particle average an unbiased estimate of the normalising constant (`docs/MATH.md` F1, F2) [RW: sequential Monte Carlo steering of language models; refs.md 64–67]. Particles are resampled at sentence boundaries
when the effective sample size falls below K/2. The reply is chosen by a satisficing rule, not by an
argmax over a weak scorer: particles that failed (unfinished, a violated constraint, degenerate output
including stock refusals, a prose reply cut mid-sentence) are dropped, a length band is kept, and the
most mutually similar of the rest is chosen; maths answers go by majority. Selection among K samples
can tilt the distribution by at most log K − (K − 1)/K nats per stage (0.96 nats for K = 6;
`docs/MATH.md` F4). Defaults: temperature 0.6 and top-p 0.9 on the network's distribution, n-gram
blocking of order 6 outside code and quotes, 768 new tokens (`bes/config.py`, `DecodeCfg`); `v9`
ships without a library.

The only measurement of this machinery on a real chat model is a stand-in: our decoder driving
Falcon-H1-Tiny's own weights through llama.cpp on 63 development prompts, 47 of them with
machine-checkable constraints (`docs/E9_DECODING.md`). Every constraint was satisfied on 31 of 47
prompts with plain sampling (66% [52, 78]), 43 of 47 with six particles, the failure filter and the
consensus (91% [80, 97]), and 46 of 47 with the exact potentials as well (98% [89, 100]); paired, the
filter was better on 12 prompts and worse on none (sign test p = 0.0005), the potentials better on 15
and worse on none (p = 0.0001). The fluency proxy moved by −0.025 ± 0.019 nats per token (not
significant), at 5.8 times the response tokens and 4.0 times the CPU time per turn; 9 of 63 replies of
the potentials arm ended exactly at the token budget. These prompts were also the development set of
the constraint compiler, which reads every annotated constraint of 96% of them and of 85% of the
held-out half, so the gains are measured where the compiler is at its best.

### 10.3 Choosing the decoding settings

After training, the decoding-selection stage (35 minutes) may replace the default only on a
significant paired sign test on development prompts (`docs/CHANGES.md` §19.5). Separately, a study of
six settings on the first 60 development prompts (the default; 3-gram blocking; temperature 0.4; one
particle; a 320-token budget; temperature 0.8) is under way, with replies generated on the runtime's
idle host cores from the export of step 25,280, which is a `final` checkpoint (operations notes). [TBD:
judged results of the decoding study, paired against the default, after its blind readings; whether
a result on `final` transfers to `v9` is itself not known.]

## 11. Evaluation protocol

The protocol was written, implemented and tested before any of our replies existed:
`docs/EVAL_PROTOCOL.md` is in the branch's first commit (`5ff33c5`, 23 November 00:35 UTC), before
any run of ours had trained (the first runtime was allocated at 02:58:32), and has not changed since
(code in `bes/eval/proxy/`).

**Prompts.** 400 original English prompts written for this project, no benchmark items: chat 100,
maths 60, code 60, writing 100, instruction-following 80; a quarter of each category has a second user
turn that depends on the model's own first answer; 95 prompts carry machine-checkable constraints,
every count stated in the prompt text; all 60 maths answers are recomputed with exact fractions in a
test.

**Split.** Inside each stratum (category × number of turns) the prompts are ordered by the SHA-256 of
their identifier and assigned alternately to `dev` and `test`, 200 each. The protocol requires every
decision to use `dev` and `test` to be used once, for the final claim, with every setting fixed on
`dev`. **One deviation.** The informal mid-run samples of Section 12.2 used eight fixed prompts, of
which seven (chat-002, chat-017, math-034, instruct-034, instruct-062, writing-100, code-026) are in
the `test` split (checked against `bes.eval.proxy.prompts`); two of them motivated the data rules of
Section 9 (instruct-034 the refusal rule, chat-017 the deflection rule). Those decisions changed the
training data, not the evaluation, but these seven test prompts are no longer unseen. We will report
every test result also on the 193 test prompts that no sample touched [TBD: R1–R4 on the 193
untouched test prompts, with the main results].

**Conversations and seeds.** Each model sees its own system prompt, the user turns and its own earlier
visible replies (hidden reasoning is never fed back). Candidate k of turn t of prompt p uses the same
derived seed for every model. Generations are cached under a hash of model, messages, settings, seed
and harness version, so a run resumes and stale outputs are never paired. The opponents' replies to
all 400 prompts were generated before the end of training (operations notes).

**Blinding and sealing.** Which reply is "A" is a seeded assignment balanced overall and within every
category; pair order is a second seeded shuffle; pair identifiers are salted hashes; readers see only
the pair files, never the key. A pair set only grows: the key commits to its versions as a hash chain,
published with the pairs, and every rating record carries the commitment it was made under and the
digest of the pair it rated. The scorer accepts a record only if its commitment is in the chain, its
pair was published under it and the digest matches; otherwise it refuses the whole source. A key or
pair file edited, or a pair set rebuilt behind the verdicts' back, is detected.

**Two readings on exchanged sides.** Every pair is read twice, in the blinded order and with A and B
exchanged. A side wins only if both readings choose the same reply; otherwise the pair is a tie. The
instructions put correctness first, then following every explicit instruction, then helpfulness; they
say that length is not a merit, that order and labels must be ignored, and that repetition,
unfinished text or writing the user's next message are serious flaws (the full prompt is in
`docs/EVAL_PROTOCOL.md`, appendix). The protocol's automatic judge is an instruction model read on its
next-token distribution over A, B, C; in this run the readings are made by language-model agents that
read only the blinded pair files and write verdicts that the scorer verifies against the commitments
(the operator's `blind_eval.py`). [TBD: the reader model and version, and whether a second reader of
another family reads the decisive comparisons, when the readings are made.]

**Exact scores first.** Maths answers are extracted by a fixed rule and compared exactly; constraints
are checked with the project's exact layer; degenerate output (empty, cut at the token limit, looping
tail, repeated lines, unclosed fence, leaked role markers, stock refusal) is counted for every reply of
every model by the same finish-reason signal. Where a reader and an exact score disagree, the exact
score is trusted.

**Statistics and resolution.** Outcomes are win, loss, tie or both bad, from our side. We report the
score (wins + ties/2)/n with a Wilson interval, the decisive win rate wins/(wins + losses) with its
interval, an exact two-sided sign test, slices by category, by one or two turns and by the length
ratio of the two replies, and the readers' diagnostics (how often the two orders agree, the preference
for the first position). With n = 200 the 95% interval of a score near one half is about ±7 points;
detecting a 10-point difference with 80% power needs 194 decisive comparisons, 277 at a 30% tie rate;
5 points need 783 and 1,118 (`docs/EVAL_PROTOCOL.md` §7). A lead smaller than about 7 points on the test
half is no evidence either way.

**Controls and disclosure.** Our model draws six samples per reply and selects among them; each
opponent can be run with six samples chosen by its own mean log-probability (matched in samples) and
with our selector (matched in selection), and our model with one particle. Token budgets differ: 768
of our tokens (about 430 words) against Falcon-H1-Tiny's 1,024 (about 600 words) and MobileLLM's 1,024
answer tokens after up to 3,072 of reasoning. [TBD: which controls are run in the final evaluation,
when it is planned after training.]

**A reference point.** Falcon-H1-Tiny with its recommended preset on the 200 development prompts (248
replies): 66.7% of the 30 maths prompts answered correctly (last answer), 78.7% of the 47 constrained
prompts fully satisfied, 1.6% of replies cut at 1,024 tokens, no empty, looping or leaked replies,
median length 174 words (`docs/EVAL_PROTOCOL.md` §11, every verdict checked by hand).

## 12. Results

### 12.1 Measured so far

- **Accounting.** 98,036,104 stored numbers, 19,631,240 read per token in the standard reading; the
  strict reading is 21.2–28.2 million on 40 documents (Section 4).
- **Systems.** Live throughput of the same model on the same chip from 112–117 thousand to 281
  thousand tokens per second through changes exact to rounding (Section 7); the token pages cost
  1.54–1.62 times the time per token for 3.6% more arithmetic (Table 10).
- **Graft.** Identical losses of source and grafted parameters on synthetic batches at steps 26,890 and
  28,501; the run continued at p = 0.511 with an unchanged learning-rate schedule (Section 8).
- **Training.** 7.79 × 10⁹ tokens in 29,722 steps of `final`, then `v9`; windowed training loss 2.699
  (p 0.05–0.10) → 2.496 (p 0.45–0.50); tape gain 0.318 → 0.022 nats (Section 6.4).

### 12.2 The mid-run standing

We looked at the model during training on CPU, with one particle and no library, next to
Falcon-H1-Tiny's cached replies on eight fixed prompts of our set (seven of them in the `test`
split, a deviation from the protocol described in Section 11). At step 22,613 (5.93 × 10⁹ tokens,
p = 0.40) our model
confused a virus with a bacterium, deflected the question about a phone in water, answered that a week
has 60 minutes and that 2¹⁰ is 10, produced malformed and self-repeating JSON, a repetitive product
review, a monologue instead of a dialogue, and a JavaScript function that returns its input
(`midrun_sample_step22613.md`). Falcon-H1-Tiny's replies contain errors too (a week of 420 minutes; a
`reduce` that returns the minimum) but are fluent and on task. On each of the seven prompts that have
a Falcon reply in the sample, ours is the worse reply by our reading. This is an informal reading by
the authors, not the protocol, and it was reported to the user at the time; at 0.4 of training the
model was clearly behind Falcon-H1-Tiny.

### 12.3 Final results

**Table 12. Quantities to be filled after the evaluation** (all on the `test` half, 200 prompts, read
under the protocol of Section 11).

| # | quantity | value |
|---|---|---|
| R1 | score against Falcon-H1-Tiny-90M-Instruct (recommended preset): (wins + ties/2)/n, 95% Wilson interval; decisive win rate and interval; sign test p; tie rate; n | [TBD: after the blind readings, 24–25 Nov] |
| R2 | the same against MobileLLM-R1.5-140M (the arm chosen on `dev`) | [TBD: same] |
| R3 | R1 and R2 by category, by one or two turns, by length ratio; readers' agreement between orders and first-position preference | [TBD: same] |
| R4 | exact scores of ours and both opponents: maths final answers correct (of 30), constrained prompts fully satisfied (of the constrained test prompts), degenerate rates by kind, stock refusals and deflections | [TBD: after the test replies are generated, right after the export] |
| R5 | our model with one particle against both opponents; opponents with six samples (matched in samples, and matched in selection) | [TBD: if run, after the main readings] |
| R6 | held-out loss of the export by domain (web text, chat, assistant tokens), and of the stopped `final` checkpoint on the same rows | [TBD: after the export, scripts/midrun_eval.py] |
| R7 | preference optimisation: accepted or not by its health checks, learning rate kept, number of pairs, held-out preference accuracy | [TBD: after the 60-minute stage, about 19:37 UTC] |
| R8 | decoding selection: the setting kept and the paired test behind it | [TBD: after the 35-minute stage] |
| R9 | training totals: final step, tokens, TPU hours used, share in training steps | [TBD: at the end of the run, 20:32 UTC] |
| R10 | the 16-bit export's held-out check (absolute loss difference against float32) | [TBD: after the export] |
| R11 | the decoding study on 60 development prompts (Section 10.3) | [TBD: after its readings] |
| R12 | R1–R4 restricted to the 193 test prompts that no mid-run sample touched (Section 11) | [TBD: with R1–R4] |

Figure 7: Blind scores against each opponent with 95% intervals, overall and by category, with the
resolution of the test (±7 points at n = 200) marked [from the figures agent, after R1–R3 exist].

## 13. Negative results

**Table 13. What was tried and did not work, or did not resolve** (sources in `CLAIMS.md`).

| idea | what was measured | consequence |
|---|---|---|
| exact Gaussian-process attention ("Bayesian attention") | the linear algebra is exact (errors 1e−15) and robust to repeated decoys, but against softmax with a sink both are within 2e−4 nats of the Bayesian predictor on Dirichlet–Markov data; on real text its shrinkage form beats a fixed weight by only 0.003–0.008 nats; at temperature 0.6 it loops in 63.6% of samples against 5.0%; it costs 4.5–9 times a softmax head on the TPU | not adopted; the sink and the counts of distinct successors remain |
| scale-recursive loops | the allocation law holds; the gain is 0.1–3% of the reducible loss | not adopted |
| coarse-to-fine output head | equivalent to SVD-softmax; +0.035 to +0.07 nats on the head | not adopted |
| confidence value channel; structural rotary clocks | +0.7% and +0.3% when removed, under the 1% bar | off |
| countdown registers as a step function | counting 79% with them against 93% without at 2 million tokens | gate failed; kept anyway (deviation, Section 5.6) |
| fence-closing probe | 100% with and without the mechanism | probe cannot discriminate |
| the first head-wise attention gate (logit x · w) | saturated within 244 steps; +0.80% loss (+1.22% on chat); right function and gradient, so no test or reviewer saw it | rescaled by 1/d; then neutral (−0.02%) and not shipped |
| decay 0.1 on the page tables | argued to hold rare tokens' rows short of their optimum; not resolvable in 244 steps | no decay |
| token-specific queries or keys | +0.046 nats in the laboratory | not adopted |
| per-token gains on feed-forward neurons | −0.035 nats at four times the storage of the output gain | off |
| hashed n-gram input tables | about 0.007 nats at this budget by the published slope | not implemented |
| the written library after its precision fix | coverage on NQ-open validation 35.0% → 6.0%; the gate opens on 8 of 200 development first turns | not in the shipped configuration |
| the plan's throughput | 300–450 thousand tokens per second assumed; 112–117 thousand measured | the budget was planned for 16–25 × 10⁹ tokens; the run will see about 1.3 × 10¹⁰ (projection) |
| the pre-registered `v9`-against-`final` comparison | not run (both measured three times slower than planned; it would have cost a tenth of the budget) | `final` trained; `v9` adopted later at the user's request, not by the rule |
| the critic | its gate (within-prompt rank correlation ≥ 0.3) was never measurable; its stage was given 0 minutes | off |
| the learning-rate transfer rule at micro scale | +14.4% loss when applied at 4,096 tokens per step | not applied in the micro ablations; applied at full scale by extrapolation |
| on-chip speed candidates | Table 9 (thirteen measured non-improvements) | not adopted |
| the `sorted` table gradient | 16 times fewer multiply-adds than one-hot; 25% slower than the scatter-add on the chip | not used |
| operations | agent turns that blocked on an approval or let the development container sleep cost about 4 h 50 min of calendar time without allocation requests (not TPU time); the loss of the second runtime is unexplained; data errors on a new machine idled the TPU for about 45 minutes | fixed where the cause was ours |

## 14. Limitations and threats to validity

1. **One run, one seed.** There is no estimate of run-to-run variation at full scale; in the micro
   ablations a change of seed and data order moved held-out loss by 0.42%.
2. **No baselines for the design change.** There is no from-scratch `v9`, no `final` continued to the
   end and no graft at another point; the effect of the pages, and of grafting rather than restarting,
   is not identified by this run (Section 8.6).
3. **Mid-run changes confound the curve.** The training stream changed at steps 6,647 (refusals),
   25,280 (deflections) and 29,722 (no library rows, new seed, new parameters); the schedule's
   learning-rate factor jumped up at step 12,062; the compiled graph changed at steps 9,332 and 22,613
   (equal to rounding, but rounding points moved). Table 5 is a training curve on a moving target, not
   a learning curve.
4. **Training loss is not held-out loss.** Held-out numbers are [TBD: Table 12, R6, after the export].
5. **The readers are language models.** They have known biases toward position, length, style and
   their own family's text [RW: biases of language-model judges; refs.md 78–81]. The protocol turns pure position
   preference into ties and reports the diagnostics, but cannot remove a shared bias of both readings.
   The agents that built our system, its readers and parts of our and the opponents' training data
   may come from related model families; that is a self-preference risk we can report, not remove.
6. **The prompts are ours, and seven test prompts were seen.** The 400 prompts were written by the
   project; the `dev` half was used for decisions (the constraint compiler, the decoding study). Seven
   `test` prompts were used in informal mid-run samples, and two of them motivated changes to the
   training data (Section 11, "One deviation"); results are also to be reported without them. The
   style of the whole set may suit our training mixture more than the opponents'.
7. **Resolution.** With 200 comparisons per opponent only differences of roughly 7–10 points are
   resolved; a smaller lead is no evidence either way.
8. **Unequal sampling.** We select among six particles; the opponents draw one sample unless the
   matched controls are run (R5). Token budgets differ (Section 11).
9. **Opponent settings.** Falcon-H1-Tiny runs at llama.cpp's defaults because its publisher gives no
   sampling settings; MobileLLM-R1.5 runs from a converted community file with budget forcing, not
   with its publisher's 30,000-token evaluation setting. Either may understate the opponent.
10. **Compile-only numbers are models.** The attribution of `v9`'s cost by part and every predicted
    saving are estimates; the cost model was 5% high on `v9`'s step and 2.7 times low on the cost of
    `sorted` (Section 7.8).
11. **One chip and one stack.** The conclusion that bytes, not parameters, set the cost is measured
    on one TPU v6e with jax 0.11.2 and libtpu 0.0.50 and on our implementation; a layout of the pages
    that avoids the padding could be cheaper (Section 7.8), and other accelerators move bytes at other
    prices.
12. **Small laboratories.** The laboratory behind the pages is tiny and in a regime where count-based
    n-gram models beat every neural arm, so token-keyed storage is worth more there than at full scale
    (`docs/V9.md` §8); the scaling-law predictions of Section 4.4 are unvalidated.
13. **Deviations from pre-registration.** The loss reading of the E2 rule chosen after seeing the
    data; the registers kept despite failing E4; the E5 copy criterion judged wrong after the fact; the
    pre-registered configuration comparison not run; the configuration changed mid-run at the user's
    request; seven test prompts used in mid-run samples (Sections 5.6, 8.1, 10.1, 11).

## 15. Reproducibility

- **Code.** Branch `claude/modest-carson-yi2d9g` of the project repository. The run was created at
  commit `98598c2` (clean tree, recorded in the run manifest) and continued with code-only changes
  that leave the configuration hash unchanged: `e096dff` and `77305dc` (continuation on a new machine),
  `566517a` (refusals), `b05bfe7` (pieces), `86bcd45` and `782be85` (post-training stages and plan),
  `6bf9b6e` and `641c4b6` (speed switches, reviewed kernel switch, test-reply generation), `4f4ef45`
  (deflections), `9a4d652` (graft). This draft describes `9a4d652`; at the time of writing the working
  tree also holds uncommitted changes by other agents (the `flat` switch and the graft tests), which are
  [TBD: committed or dropped, before submission].
- **Configurations.** `configs/final.py` (effective hash `0a7e9b33372a07ee` with the run's overrides)
  and `configs/v9.py` (effective hash `f29bfeb3ebf353a8`). The run's command line:
  `scripts/run_24h.py --config configs/final.py --out /content/runs/main16 --budget-minutes 1279.7
  --override model.table_grad=take --override run.bench_plans=reduced --override
  run.post_minutes=critic=0,buffer=20 --override run.post_reserve_hours=3.6667 --override
  optim.micro_seqs=16`; continuations `scripts/run_24h.py --out RUN --resume`. Per-run files that do
  not enter the hash: `micro_split` = `4`; `fast` = `kernel=onetile:save relu2=1 ce=0`; `post_plan` =
  `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=15`.
- **The graft.** `python scripts/graft_run.py SRC DST --config configs/v9.py --override run.seed=1
  --check --cpu`, then the directory swap and a resume.
- **Measurements.** `scripts/tpu_kernels.py`, `scripts/tpu_micro.py`, `scripts/tpu_tune.py`,
  `scripts/plan_sweep.py` and `docs/tpu_profile/*` on an idle chip; the exact commands and raw outputs
  of the 24 November measurements are kept with the workflow reports (`chip_kernels_0745.txt`,
  `v9_meas*.txt`).
- **Evaluation.** `python -m bes.eval.proxy` (`docs/EVAL_PROTOCOL.md` §9), `scripts/gen_test_replies.py`
  for our replies, the operator's `blind_eval.py` for the readings.
- **Determinism.** The same batch gives the same loss twice, bit for bit, on the chip (the start-up
  canary); a restart after 10 steps reproduces 20 uninterrupted steps bit for bit on the CPU (tests);
  the data position is a function of the step. Bitwise reproducibility across restarts on the chip was
  not tested. What cannot be reproduced is the provider's allocation of runtimes and with it the
  timing of the run's interruptions.
- **Tests.** 1,349 test functions in `tests/` (static count), including the exact references of the
  exact layer, the output law, the kernels and the graft.

## 16. Licences, safety and conduct

ClimbMix, the largest base source, is distributed under an MIT card but derives from a corpus under
CC BY-NC 4.0; we treat it as non-commercial, so the trained weights inherit that restriction [TBD:
licence review of the released weights, before release]. Several chat and preference sources were
generated by third-party models whose terms set conditions on training with their outputs
(`docs/DATA_SOURCES.md` §4). The model is not trained to refuse (Section 9.4) and has not been
evaluated for harmful output; it should not be deployed. The development container held the user's
Colab credential only in a file readable by its owner and never in the repository (`docs/CHANGES.md`
§19).

## 17. Conclusion

A budget in parameters is a statement about a function: how many numbers it stores and how many it
reads to produce a token. The cost of training that function on one accelerator for one day is a
statement about bytes: how many are moved, in which shapes, through which memories. In our run the
two came apart by a factor of 1.6 between two configurations that the parameter ledger calls nearly
equal, and by a factor of 2.4–2.5 between two implementations of the same function. Treating bytes as
the cost recovered that factor without changing what the model computes, and an exact graft let the
run change its design halfway without discarding what it had learnt. Whether the resulting model is
preferred to Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M is decided by a protocol fixed in
advance, whose results are [TBD: Table 12, after the evaluation].

## Appendices (to be assembled)

- **A. Ledgers and audit.** Full output of `bes.budget.count` for both configurations; the read-set
  audit tables (`docs/ACCOUNTING.md`).
- **B. Mathematics.** Statements and proof sketches of B1 (softmax with a sink as a Dirichlet
  posterior predictive), T1b (the output law as a Pitman–Yor predictive with the network as base),
  F1–F5 (particles, masks, twists, limits), the loop allocation law and the token-neuron
  water-filling, each with its test (`docs/MATH.md`, `docs/V9.md` §3, `tests/math/`).
- **C. On-chip measurements.** Raw outputs (`docs/tpu_profile/`, `chip_kernels_0745.txt`,
  `v9_measA.txt`, `v9_measB_sorted.txt`, `v9_measC_onehot.txt`), the compile-only method and its
  calibration (`speed_mlp_block.md`, `v9_speed.md`), and the kernel review (`kernel.md`).
- **D. The graft.** Algorithm, refusal conditions, the check, the review and its tests
  (`scripts/graft_run.py`, `graft_review.md`).
- **E. Data.** Sources, licences and pinned commits (`docs/DATA_SOURCES.md`); the scans
  (`openings.json`, `deflection_check2.json`).
- **F. Evaluation.** The full protocol, the reader instructions and the power table
  (`docs/EVAL_PROTOCOL.md`).
- **G. History.** The dated record of changes (`docs/CHANGES.md` §17–19.6 and the facts of 24 November
  after 08:12 UTC listed in this draft).
- **H. Negative results in full** (`docs/NEGATIVE_RESULTS.md`, `docs/V9.md` §9).
- **I. Claims.** `CLAIMS.md`: every number of this paper with its source.
