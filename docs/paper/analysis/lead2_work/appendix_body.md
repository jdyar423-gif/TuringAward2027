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
configuration meets the strict reading at 2,048 positions. The audit's three checks are described in
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
  comparable; Section 7.4 measures bits per byte instead.

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
(1,024 tokens; 2,048 from p = 0.67), the chat share of the mixture (0.25; 0.7 from p = 0.92), loss on
assistant tokens only from p = 0.92, and `final`'s library shares are all functions of p. Two
consequences were observed. A pause leaves T fixed and shrinks R, so p advances while nothing trains:
the graft's pause moved p from 0.5049 (stop event) to 0.5112 (restore event). Moving the end of
training later raises R, so p falls: when the post-training reserve was cut from 220 to 118 minutes at
step 12,062 and the training end moved from 16:51:49 to 18:33:49, p fell from 0.2415 (stop event) to
0.2190 (restore event) and the learning-rate factor rose from 0.783 to 0.803. The schedule is
continuous in the parameters and the optimiser state, not monotone in p. (Step records, the nearest
logged steps, give 0.2413 → 0.2196 and 0.5045 → 0.5118; we quote the events.)

**C.2 Surviving the loss of a machine.** Checkpoints are written atomically every 25 minutes
(parameters, optimiser state, data cursor, random state, time ledger); after each one a snapshot is
copied off the runtime. On a new runtime the snapshot is restored, the deadline is moved by the
downtime (which keeps p continuous) and the trainer continues. A loss guard rolls back on three
consecutive spikes above 1.3 times the moving average; it has not fired. Restarting on a new machine
exposed two errors no rehearsal had: the loader replays its journal and needs every shard it lists,
which a new machine must download and tokenise again (the trainer now waits for them, with 24
tokenising workers: 363 shards in 9 minutes), and two sources that share raw files deleted them
before both had read them. Together they left the third runtime's TPU idle for 48 min 19 s before
training resumed (commits `e096dff`, `77305dc`).

**Table C1. TPU runtimes** (`docs/CHANGES.md` §19.1–19.4; the operator's TPU-time ledger).

| runtime | counted (UTC) | seconds | what it produced |
|---|---|---|---|
| 1 | 23 Nov 02:58:32 – 05:00:37 | 7,325 | profiling and the rewrites of Section 5.2; its training runs were discarded; returned at the user's request |
| 2 | 23 Nov 17:13:00 – 19:30:00 (from the earliest possible allocation) | 8,220 | training from step 0 to step 5,075 (kept through a snapshot); taken back by the provider at about 19:29 |
| 3 | from 24 Nov 01:05:55 | 39,245 at 12:00:00 | training resumed at step 5,075 at 01:54:14; in progress |

Gaps without a runtime: 12 h 12 min 23 s and 5 h 35 min 55 s. The 24 hours end at 20:46:50 on
24 November; the run's deadline is 20:31:49.

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
| – | [TBD: final step, at 18:34 UTC] | [TBD] | 18:33:49 | end of training |

Averaged over its 32,320.8 training seconds, the trajectory up to the graft trained at 241,065
tokens/s (7,791,443,968 tokens; trainer's ledger).

**C.3 The training curve.** Table C3 summarises the step losses logged every 50 steps (each the mean
over the step's 262,144 tokens) in windows of progress; the role split and the tape gain are those of
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
(sd 0.0399); the stream changed at the graft, so the two numbers are not comparable. The tape gain fell
from 0.319 nats in the first 1% of training to 0.022 nats (0.9% of the loss) by p ≈ 0.5; in the micro
ablation the tape switched off at evaluation cost 10.5% of the loss at 1 million tokens. One reading,
which the micro report anticipated, is that the network learns at scale most of what the exact
in-context statistics offered it early; the tape gain measures how much the trained model leans on the
tape, not what training without it would give. On the held-out rows the tape gain is 0.013 nats on web
text and 0.030 on chat at step 29,722.

![Figure C1](figures/F1_loss.svg)

**Figure C1.** Training loss of run main16 against training tokens: (a) the whole run on logarithmic
tokens; (b) from 10⁹ tokens on, linear. Dots: the loss of one optimiser step, logged every 50th step;
lines: a Gaussian kernel average over log₁₀(tokens) (σ = 0.03 decades), separately for `final` (blue)
and `v9` (orange). Near the graft the `v9` line is a one-sided average that already includes later,
lower points; it must not be read as a level change across the graft, where the data stream changed.
Vertical lines: the events of Table C2. (c, d) The learning-rate factor and progress p as logged.

**C.4 After training.** The 118 minutes after training are planned as follows (`docs/CHANGES.md` §19.5;
`bes/train/prefopt.py`). Sampling (3 minutes) adds a few pairs of the model's own replies. Preference
optimisation (60 minutes) runs one epoch of DPO [74] against the frozen trained model (β = 0.1, 64
pairs per step, plain Adam on every leaf, gradient-norm clip 1.0), almost entirely on UltraFeedback
pairs [76], tries the learning rates 1e−6 and 3e−6 from the same start (Falcon-H1-Tiny's choice [11]),
and accepts a result only if held-out preference accuracy improves beyond its standard error,
held-out assistant loss rises by at most 3%, and sampled length and repetition stay within fixed bands;
otherwise the trained weights ship. Decoding selection (35 minutes) may replace the default decoding
only on a significant paired sign test on development prompts. Export (5 minutes) checks the 16-bit
weights against float32 on held-out rows. Our replies to the test prompts are then generated on the
runtime's host cores (`scripts/gen_test_replies.py`; about 41 minutes on 42 workers, measured on the
development prompts). [TBD: what each stage did, from the stage ledger at about 20:32 UTC.]

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
development prompts the replies of `final` at step 25,280 contain no stock refusal or deflection
(Table 8). [TBD: the share of test replies of ours and of both opponents flagged by `exact_score`.]
From step 6,647 on, the model is not trained to refuse anything; this was the operator's decision,
reported to the user, who could have overruled it, and the evaluation does not test behaviour on
harmful requests. The preference pairs from UltraFeedback are not filtered by either rule (2.44% of its
scanned responses open with "As an AI …").

![Figure D1](figures/F5_openings.svg)

**Figure D1.** (a) Share of each source's assistant turns whose opening (first 44 tokens) matches a stock
pattern, from the scan of the tokenised sources (24 November 08:20; all segments of small sources,
40,000 sampled otherwise); "Sure …" is a control that no rule removes; a turn can match several
patterns. (b) Share of each chat source's conversations that the stock-refusal rule (gray, live since
step 6,647) and, in addition, the deflection rule (violet, live since step 25,280) leave out, measured
with the loader's functions on 30,000 sampled conversations per source (2,260 for smoltalk_everyday,
all it has). Black ticks: share of reads the live loader actually skipped for either rule since 08:36.

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

**E.4 The decoding study on `final`.** Six settings on the first 60 development prompts, generated on the
runtime's idle host cores from the export of `final` at step 25,280 and read blind by Opus 5.5 agents,
one reading per pair (`N/study/proxy/reports/ab/`; exact scores `lead2_work/exact_dev.json`):

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

The study cannot separate the settings: most pairs were judged "both bad". It was made on `final`;
whether a result would transfer to `v9` is not known. [TBD: the study on `v9` (step 31,912; default and
3-gram blocking), whose replies were fetched at 12:10 UTC and are being read.]

**E.5 Decoding is bound by the host.** On the chip one decoding step took 12.1 ms for 6 rows and 34.5 ms
for 24; a complete particle decode with 6 particles took 43.1 ms per position, of which about 31 ms was
host-side Python (sampling, constraints, exact features) (`docs/CHANGES.md` §19.3). [TBD: decoding
throughput of the final export, measured during the test-reply generation.]

## Appendix F. On-chip measurements in full

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
update counting the update (`wf/v9_speed_work/hlo_bytes.py`).

## Appendix G. The graft in full

`scripts/graft_run.py SRC DST --config configs/v9.py --override run.seed=1 --check --cpu` (commit
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
| the plan's throughput | 300–450 thousand tokens/s assumed; 112–117 thousand measured | planned for 16–25 × 10⁹ tokens; the run will see about 1.36 × 10¹⁰ (projection) |
| the pre-registered `v9`-against-`final` comparison | not run (Appendix C.5) | `final` trained; `v9` adopted later at the user's request |
| the critic | its gate (within-prompt rank correlation ≥ 0.3) was never measurable; its stage was given 0 minutes | off |
| the learning-rate transfer rule at micro scale | +14.4% loss when applied at 4,096 tokens per step | not applied in the micro ablations; applied at full scale by extrapolation |
| on-chip speed candidates | Table F4 (13 changes measured and not adopted) | not adopted |
| the cost model as a predictor of differences | sign right in 7 of 7 changes, size off by factors 0.36–7.9 (Table 5) | used to rank, never as a result |

## Appendix I. Operations

A human user set the goals and took the decisions marked as theirs (stopping runtime 1, the switch to
`v9`, the instruction to use parallel agent workflows); the design, the code, the measurements and this
draft were produced by language-model agents working under the user's direction, and every change is
dated in `docs/CHANGES.md`. Two operational losses are recorded there (§19.4). Agent turns that blocked
on an approval card or let the development container sleep cost about 4 h 50 min of calendar time in
which no runtime was requested (not TPU time); the operator now keeps a tracked background task alive
and avoids tools that wait for an approval while the user is away. The loss of runtime 2 after
2 h 17 min is unexplained: a test of the idle-timeout explanation on CPU runtimes did not reproduce it.
At 12:10 on 24 November the development container's disk filled, written by verification agents of a
speed workflow; the workflow was stopped and the run was not affected. None of this is evidence for a
claim of the paper; it is reported because it shaped what could be measured.

## Appendix J. Reproducibility

- **Code.** Branch `claude/modest-carson-yi2d9g` of the project repository. The run was created at
  commit `98598c2` (clean tree, recorded in the run manifest) and continued with code-only changes that
  leave the configuration hash unchanged: `e096dff`, `77305dc` (continuation on a new machine),
  `566517a` (refusals), `b05bfe7` (pieces), `86bcd45`, `782be85` (post-training stages and plan),
  `6bf9b6e`, `641c4b6` (speed switches, the reviewed kernel switch, test-reply generation), `4f4ef45`
  (deflections), `9a4d652` (graft), `769f240` (the `v9` switches; tests of the graft and of a `v9`
  export); documentation commits `ba0db12`, `d9f0ce4`, `1b9d1b8`, `8a86bcb`. This draft describes
  `8a86bcb`.
- **Configurations.** `configs/final.py` (effective hash `0a7e9b33372a07ee` with the run's overrides)
  and `configs/v9.py` (`f29bfeb3ebf353a8`). The run's command line: `scripts/run_24h.py --config
  configs/final.py --out /content/runs/main16 --budget-minutes 1279.7 --override model.table_grad=take
  --override run.bench_plans=reduced --override run.post_minutes=critic=0,buffer=20 --override
  run.post_reserve_hours=3.6667 --override optim.micro_seqs=16`; continuations `scripts/run_24h.py --out
  RUN --resume`. Per-run files outside the hash: `micro_split` = `4`; `fast` = `kernel=onetile:save
  relu2=1 ce=0` for `final` and `kernel=onetile:save relu2=1 ce=0 flat=1 rows=1 tokn=1` for `v9`;
  `post_plan` = `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=15`.
- **Measurements.** `scripts/tpu_kernels.py`, `scripts/tpu_micro.py`, `scripts/tpu_tune.py` and
  `docs/tpu_profile/*` on an idle chip; raw outputs of 24 November: `wf/chip_kernels_0745.txt`,
  `S/v9_meas{A,B_sorted,C_onehot,D}.txt`. Compile-only programs: `wf/v9_*_work/`.
- **This paper's own analyses** (`N/paper/lead2_work/`): `make_numbers.py` (live medians, ratios,
  projections, power, FLOP), `predictor_test.py` (Tables 4 and 5), `heldout_eval.py` and
  `heldout_analysis.py` (Section 6.5), `graft_grad.py` (Section 6.4), `exact_dev.py` (Table 8),
  `opp_bpb.py` (Table 8), `deflection_labels.py` (Appendix D.4).
- **Evaluation.** `python -m bes.eval.proxy` (`docs/EVAL_PROTOCOL.md` §9), `scripts/gen_test_replies.py`
  for our replies, the operator's `blind_eval.py` for the readings.
- **Determinism.** The same batch gives the same loss twice, bit for bit, on the chip (the start-up
  canary); a restart after 10 steps reproduces 20 uninterrupted steps bit for bit on the CPU (tests); the
  data position is a function of the step. Bitwise reproducibility across restarts on the chip was not
  tested. What cannot be reproduced is the provider's allocation of runtimes.
- **Tests.** 1,356 test functions in `tests/` at `8a86bcb` (static count of `def test_`).

## Appendix K. Licences, safety and conduct

ClimbMix, the largest base source, is distributed under an MIT card but derives from a corpus under
CC BY-NC 4.0; we treat it as non-commercial, so the trained weights inherit that restriction [TBD:
licence review of the released weights, before release]. Several chat and preference sources were
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
numbers here were checked against their sources, and because the checks were not complete.
