## Appendix P. Six more hours: rarity as the axis of memory

*Added on 25 and 26 November 2026. This appendix is about a second task on the same model: continue the
weights of the one-day run for six more hours of TPU time as a learner that keeps learning after it ships, under
the same two ledgers (at most 100M stored numbers, at most 20M read per token), and feel better than
Falcon-H1-Tiny. The pre-registration is `docs/PLAN_M6V.md`, the log `docs/CHANGES.md` section 21, the
measurement programs and their raw outputs `lab/m6v/`. Sections P.1 to P.7 were written while the run trained;
P.8 to P.10 after it.*

### P.1 What was measured first, and what it overturned

Three plans for this task were rejected before a fourth was accepted. All three planned to drop the token tables
of the v9 allocation (Section 6) and spend their numbers elsewhere, on the strength of one number of this
paper: the 0.016 nats by which v9 led the extrapolated reference 53.8 minutes after the graft (Table 5). That
number was never the value of the tables. Measured directly on the exported weights of the one-day run (16-bit,
8 held-out rows of 1,024 tokens per domain: web 8,179 and chat 7,901 weighted tokens;
`lab/m6v/tables_switches.py`, `tables_writes_neighbours.py`):

**Table P1. What the token tables of the trained model are worth.** Held-out loss 2.179 (web 2.774, chat
1.564) with everything on.

| removed | loss, all | chat | numbers freed |
|---|---|---|---|
| token pages (`page/key`, `page/val`) | +0.146 | +0.176 | 50.3M |
| output gains (`gain/o`) | +0.061 | +0.074 | 12.6M |
| pages and gains | **+0.233** | **+0.284** | 62.9M |
| per-loop value embeddings (`emb/ve`) | **+0.577** | +0.593 | 8.4M |

The tables are worth fifteen times what the plans had assumed, and the fourth plan keeps the whole allocation.

**Table P2. The value of a token's page and gain rows rises with the token's rarity.** Rows of every token of
frequency rank K and rarer set to zero (ranks from 3.0M held-out tokens).

| K | share of occurrences | loss | per occurrence | numbers freed |
|---|---|---|---|---|
| (the 512 most frequent tokens' own rows) | 0.602 | 0.074 | 0.12 | |
| 1,024 | 0.298 | +0.125 | 0.42 | 55.0M |
| 2,048 | 0.191 | +0.088 | 0.46 | 47.2M |
| 3,072 | 0.128 | +0.055 | 0.43 | 39.3M |
| 4,096 | 0.084 | +0.038 | 0.46 | 31.5M |
| 5,120 | 0.051 | +0.025 | 0.50 | 23.6M |
| 6,144 | 0.025 | +0.015 | 0.61 | 15.7M |
| 7,168 | 0.007 | +0.007 | 1.06 | 7.9M |

One occurrence of a rare token gains four to nine times what one occurrence of a common token gains from the
same eight private neurons: the shared core gives a token capacity in proportion to its gradient mass, and a
rare token gets little. Per stored number the order is the reverse (a common token's row is read more often),
and it is the product of the two that decides which rows are worth their numbers (P.5).

**Table P3. Where a fact is written decides what the write destroys.** Eight invented facts written one
after the other, eight normalised gradient steps each, into token rows only; then the loss of the answer
under a paraphrased question (2.245 nats per token before), the loss change of 60 unrelated question-answer
texts, and whether a greedy reply to the paraphrase contains the fact (`lab/m6v/writes_clean.py`).

| rows written | rows per fact (their corpus frequency) | paraphrase: after its own write → after all eight | unrelated texts, mean drift (largest) | texts without a written token | greedy recall |
|---|---|---|---|---|---|
| rare tokens (rank 2,048 and rarer) | 3.8 (0.00105) | 1.004 → 1.004 | +0.010 (0.25) | 55 of 55 bit-identical | 1 / 8 |
| rank 512 and rarer | 6.2 (0.00369) | 0.582 → 0.581 | +0.026 (0.37) | 46 of 46 bit-identical | 3 / 8 |
| every content token of the fact | 19 (0.160) | 0.393 → 0.559 | +0.827 (3.30) | 0 of 0 | 3 / 8 |
| the dense matrices (one fact, step 1e-3) | | 0.32 | +0.63 | | |

Two further measurements were negative and shaped the design as much. *Learning from retrieved neighbours
before answering* (one normalised step per neighbour on the 16 nearest stored question-answer pairs; the loss
of the true teacher answer, 1.70 before): token rows only −0.007, −0.026, −0.048 nats with 1, 4, 16
neighbours and −0.047 after two passes; dense updates at their best −0.02 to −0.04 after four steps and worse
after sixteen; a naive dense step +0.64. Test-time training is not a lever at this scale. *Coverage*: the
highest index score of a development prompt (mean 0.515, 33% at 0.6 or more) is distributed like that of a
library question with its own entry left out (0.519, 31%), so training with leave-one-out retrieval trains on
the distribution of deployment.

### P.2 The paradigm

Tables P2 and P3 are two faces of one fact. The rarer a token, the more a private memory is worth to it and
the less a write to that memory touches anything else, and for one reason: the shared core serves what is
common, and what is rare occurs in few texts.

> **The dense core computes what is shared; the sparse rows hold what is particular. Awake, write only what
> is rare; asleep, move what is dense.**

Freezing a model at the end of training and fine-tuning all of its weights both break this rule; an allocation
that keys part of its memory by the vocabulary can follow it exactly. One step further out on the same axis
is the single occurrence, a whole exchange that happened once: that is written as text.

### P.3 Exact locality

**Theorem 1.** Let the parameters be the dense leaves and, for every vocabulary id `v`, a row `r_v` in each
token table (`emb/tok`, `emb/ve`, `gain/o`, `page/key`, `page/val`: 9,472 numbers per id). The forward pass
on an input `x` reads `r_v` only for the ids at the positions of `x`. Hence for every change `Δ` whose
support is the rows of a set `S`: if no token of `x` is in `S`, the output on `x` after the change equals the
output before it, bit for bit.

*Corollaries.* (a) A text of `|x|` tokens is touched with probability at most `|x| Σ_{v∈S} f_v`: with the
written rows of Table P3's first line (total frequency 0.00105) and texts of about 80 tokens the bound is
8.4%, and 5 of 60 texts (8.3%) contained a written token. (b) Two writes commute exactly when neither taught
text contains a token whose rows the other writes; the first two lines of Table P3 show the consequence (a
fact's loss does not move while seven others are written). (c) The rows to write for the least interference
are those of the taught text's rarest tokens: the selection rule of sparse memory fine-tuning is, for a
memory keyed by the vocabulary, a consequence and not a heuristic. The unembedding is dense in every token
and is not writable in this sense.

The theorem is enforced by tests (`tests/learn/test_row_write.py`): a write through `bes.learn.write` changes
only the rows it was given, the logits of an input without a written token are identical arrays before and
after, an input that shares only unwritten tokens with the taught text is identical too, and two writes with
disjoint texts give the same parameters in either order.

### P.4 Which table, and which direction

The registered plan stated a closed-form write (its Theorem 4): among the changes of a token's page values
with a given first-order gain on the taught text, the one that least moves the page's output on the token's
other contexts is the gradient whitened by the second moment `C = E[s sᵀ]` of the activations of the token's
eight private neurons. The experiment (`lab/m6v/theorem4.py`; three passes) recorded those activations on up
to 24 library entries per token, wrote the eight facts of Table P3 with and without the whitening, and added
two measurements Table P3 lacks: *own contexts* (held-out library entries that contain a written token, up
to six per token) and *natural contexts* (150 entries drawn at random among those that contain one, so that
tokens count as often as they occur).

*The table matters more than the direction.* Writing only the page values instead of all five token tables
(rows of rank 512 and rarer): paraphrase loss 0.65 against 0.58, drift of the 60 unrelated texts +0.002
against +0.026, drift on own contexts 0.027 against 0.080, greedy recall 3 of 8 for both; with every content
token written, +0.154 against +0.827. The page value is the one row that is both private (Theorem 1) and
gated by the context (the private neurons); an embedding or a value embedding, once changed, changes every
context of its token. `bes.learn.write` therefore writes page values by default.

**Table P4. Plain and preconditioned writes of page values** (means over the step sizes 0.12, 0.18 and 0.25;
eight facts in sequence). `C⁻¹`: the plan's whitening; `R⁻¹`: the output side, with `R` the second moment of
the page-value gradients over the 192 output directions of a block (so that `C ⊗ R` is a Kronecker-factored
curvature of a row); both: `C⁻¹ G R⁻¹` and `C⁻¹ G R⁻½`.

| rows written | write | paraphrase loss | natural contexts | own contexts | greedy recall (of 8) |
|---|---|---|---|---|---|
| rank 2,048 and rarer | gradient | 1.140 | +0.113 | +0.053 | 0.67 |
| | `C⁻¹` | 1.172 | +0.055 | +0.044 | 1.00 |
| | `R⁻¹` | 1.110 | +0.059 | +0.040 | 1.00 |
| | `C⁻¹`, `R⁻¹` | 1.126 | +0.059 | +0.034 | 1.00 |
| | `C⁻¹`, `R⁻½` | 1.137 | +0.042 | +0.038 | 1.00 |
| rank 512 and rarer | gradient | 0.685 | +0.034 | +0.024 | 3.00 |
| | `C⁻¹` | 0.742 | +0.016 | +0.018 | 1.67 |
| | `R⁻¹` | 0.724 | +0.031 | +0.026 | 2.67 |
| | `C⁻¹`, `R⁻¹` | 0.829 | +0.020 | +0.017 | 1.67 |
| | `C⁻¹`, `R⁻½` | 0.814 | +0.019 | +0.017 | 2.33 |
| every content token | gradient | 0.511 | +0.103 | +0.145 | 4.33 |
| | `C⁻¹` | 0.460 | +0.083 | +0.115 | 3.67 |
| | `R⁻¹` | 0.480 | +0.070 | +0.106 | 3.67 |
| | `C⁻¹`, `R⁻¹` | 0.673 | +0.091 | +0.103 | 2.00 |
| | `C⁻¹`, `R⁻½` | 0.555 | +0.071 | +0.098 | 3.00 |

The whitening of the plan halves the interference on natural contexts for rows of rare tokens, at a similar
or slightly worse fit; the output side helps only for the rarest rows, and both sides together are not
reliably better than one. The registered criterion was that the closed-form write must be no worse than the
gradient write in paraphrase loss *and* in greedy recall. **It did not pass** (rank 512: 0.742 against 0.685,
recall 1.67 against 3.00), so the gradient write stays the default and the preconditioners are an option of
`bes.learn.write`. What the whitening buys is half the interference, not a firmer memory. The first two
passes, which had the own-context test only, showed no consistent difference; the halving appeared when
contexts were weighted as they occur. Both measurements are in the table.

One thing no variant changed: in free generation a written fact is recalled in at most four or five of eight
cases, and only when the rows of common tokens are written too, which costs everything else. What must be
recalled word for word is written as text (P.6).

### P.5 The water line

The numbers a token never reads at a position are inactive and can hold either the rows of rare tokens
(Table P2) or written entries. `bes/evict.py` stores `gain/o`, `page/key` and `page/val` for the tokens that
keep their rows and a map of 8,192 entries (one more number read per token); an evicted row is a zero row,
which the model already treats as "no page, gain one", so nothing that runs the model changed. The freed
numbers go to an *answers library*: first exchanges of the training conversations, stored as compressed text
with an exact term index (`bes/library/answers.py`), from which a reply reads at most 360,000 numbers.

Which entries. Every entry is used once as a query with itself left out; the three best candidates of the
index are credited with the square of their score; entries are kept in the order of their credit
(`scripts/subset_answers.py`). Which rows. The 5,120 most frequent ids in the library's own text and the 16
special tokens of the chat format (`scripts/ship_m6v.py`).

**Table P5. Coverage of the 200 development prompts by the library that fits beside the rows of K tokens**
(first turns; `lab/m6v/coverage_k.py`). The credit rule keeps the coverage of the full library (399,699
entries: 13 / 79 / 108, mean top score 0.515) with 43% of its entries, and more entries add little.

| rows kept | library entries | stored answer is the reply / exemplar shown / none | with the loose mode at 0.35 | mean top score |
|---|---|---|---|---|
| 5,120 | 172,273 | 12 / 71 / 117 | 12 / 136 / 52 | 0.515 |
| 4,096 | 228,062 | 12 / 74 / 114 | 12 / 140 / 48 | 0.514 |
| 3,072 | 283,973 | 12 / 76 / 112 | 12 / 139 / 49 | 0.512 |

The shipped form of the one-day weights at K = 5,120: 99,943,020 stored numbers and 19,991,241 read per
token (19,631,240 of the model, one of the map, 360,000 of the library), held-out loss 2.0071 before the
eviction and 2.0337 after. The same inputs give the same `weights16.npz` and the same library files on three
machines.

### P.6 What the library does before any training

A reply is made in one of three ways (`bes/decode/respond.py`): the library holds the user's question (same
question type, at least 0.8 of the user's term mass and 0.55 of the entry's, no differing number or name) and
its stored answer is the reply; or it holds similar questions and one or two of them are shown to the model
in a `LIBRARY` turn before the user's turn; or neither. Entries taught after shipping live in an overlay
(`bes/library/overlay.py`, 60,000 numbers reserved in the ledger and taken off the read cap once): for a
message that shares no index term with a taught entry, the candidates, their scores, the numbers read and
the reply are exactly what they were before the entry was taught. Teaching eight invented facts through the
overlay of the shipped one-day weights: seven of eight paraphrased questions are answered with the fact, six
of six unrelated replies are identical, and seven of eight again after the process is restarted from the
memory file.

**Table P6. The one-day weights with the library, against Falcon-H1-Tiny-90M-Instruct** (recommended
settings; registered judge `qwen3-4b-2507`, both orders, a verdict only when they agree; W / L / T from our
side, score = (W + T/2)/n).

| prompts | arm | W / L / T | score [95%] | sign test p |
|---|---|---|---|---|
| development, 200 | library on | 33 / 42 / 125 | 0.477 [0.435, 0.520] | 0.36 |
| development, 200 | the same weights and prompts, library off | 10 / 52 / 138 | 0.395 | |
| development, the 85 prompts whose replies used it | on / off | 28 / 13 / 44 against 5 / 23 / 57 | 0.588 against 0.394 | < 0.0001 (paired: 37 better, 6 worse) |
| **test, 200** (pair set `m6v-before-vs-falcon`) | **library on** | **27 / 63 / 110** | **0.410 [0.344, 0.479]** | **0.0002** |
| test, 200 (Table M1) | the official one-day model: no library, after DPO | 12 / 61 / 127 | 0.378 [0.313, 0.446] | < 0.0001 |

**Table P7. The same comparisons by the way the reply was made.**

| | development | test |
|---|---|---|
| the stored answer is the reply | 8 / 1 / 3 (0.792), n = 12 | 4 / 2 / 7 (0.577), n = 13 |
| exemplars shown | 20 / 12 / 39 (0.556), n = 71 | 18 / 18 / 29 (0.500), n = 65 |
| no exemplar | 5 / 29 / 83 (0.397), n = 117 | 5 / 43 / 74 (0.344), n = 122 |

Attaching the library to weights that were never trained for it moves the development score from 0.395 to
0.477, level with Falcon there, and the gain is in the replies that used it. On the test prompts the same
model scores 0.410 and is still clearly behind: every way of replying scores lower on the test prompts,
including the one that does not touch the library, so the difference between the splits is mostly the
prompts (and each score carries about ±0.07), but the rules of the library were set while looking at
development prompts and that cannot be excluded. What both splits agree on is the structure: where the
library was used, wins and losses are about even or better (28 to 13, and 22 to 20); where it was not, they
are 5 to 29 and 5 to 43. Loosening the rule for a verbatim reply would gain nothing: on the 29 development
prompts with a same-type candidate of score 0.6 or more, the model's own reply written beside the exemplar
(0.759) is as good as the stored answer (0.724).

### P.7 Loose exemplars and match tags

The split of Table P7 set the last change before the run. The library's own floors (ranked score 0.5; 0.75
for an arithmetic message; any contrast in numbers, names, order, degree or polarity excludes an entry) give
83 of the 200 development prompts an exemplar and one of the 30 arithmetic prompts. Of the candidates between
0.35 and 0.5, about two thirds are the same task with other content (a discount on another price, a haiku on
another subject, the growth of another town); and the best candidates of arithmetic prompts are excluded
because their numbers differ, which is exactly what makes them worked examples.

*The loose mode* (`AnswerLibrary.exemplars(floor=…)`): the best exemplar may score down to the floor for every
kind of message, the contrast check reaches down to the floor, and an entry that differs from the message in
its numbers and in nothing else is admitted. With a floor of 0.35 it shows exemplars to 148 development
prompts (arithmetic 28, writing 23, instructions 30). Without a floor the library behaves bit for bit as
before, and the rule for a verbatim reply is untouched.

*Match tags.* Shown a jacket of $200 at 15% off for a question about a jacket of $120 at 35% off, the one-day
weights answer for $200; asked for a title about saving money in double angle brackets beside an exemplar
about productivity, they return the productivity title. They had learnt that a `LIBRARY` turn holds the
answer, and the reader of the turn cannot see the index's score. So every exemplar starts with one line that
the retrieval computes (`match_tag`): `Match: close` (score at least 0.7), `similar` (at least the library's
floor), `loose` (under it; the distractors of training carry this tag too) or `other numbers`.

*Measured on the one-day weights* (no tags, never trained for this; the 79 development prompts whose
exemplars change at a floor of 0.35, 77 judged): entries admitted by score move those prompts from 0.383 to
0.436 (7 better, 3 worse), entries that differ in their numbers from 0.383 to 0.283 (3 better, 7 worse:
the numbers of the example are copied). Over the whole development split: 0.4775 with the library's floors,
0.4900 with the loose mode by score, 0.4750 with other numbers too; none of the differences is significant.

*In training* (`configs/m6v.py`): each sample uses the loose mode at 0.35 with probability 0.6 and the
library's floors otherwise, always with tags. On 1,000 library questions (leave-one-out) exemplars reach 65%
of the samples instead of 48%, 11% of all samples carry one that scores under 0.5, and a sample grows from
420 to 467 tokens. Because both modes are in the training distribution, the floor of inference is a choice
made afterwards on the development prompts (P.9), with the library's own floors as the way back.

### P.8 The continuation

The run (`configs/m6v.py`) started from the 64 leaves of the one-day weights before their preference
optimisation, with a fresh optimiser state, the graph and the fast plan of the one-day run, and the learning
rate of a single schedule over both runs (0.29 of the old peak, falling linearly to 0.099). It trained 13,615
steps of 2,048 tokens, 3.57 × 10⁹ tokens in 19,004 s at 188,300 tokens per second, without a restart; nine
tenths of the stream is conversation, each sample with the exemplars the library would show for it with its own
entry left out (P.7). A preference optimisation of 578 s followed (accepted by the run's own check: accuracy on
1,823 held-out preference pairs 0.500 → 0.690) and the export. With the 865 s spent before, the task used
20,937 s of the 21,600 s it was given.

**Table P8. The weights before and after** (the measurements of Tables P1 to P3 repeated on the shipped
weights; 16-bit; `lab/m6v/raw/final_weights/`).

| | before the run | end of training | after preference optimisation (shipped) |
|---|---|---|---|
| held-out loss, chat | 1.564 | **1.528** | **1.542** |
| held-out loss, web | 2.774 | 2.846 | **2.857** |
| page and gain rows off | +0.233 | | +0.303 |
| value per occurrence, ranks 0–512 / 512–1,024 / 1,024–2,048 / ≥ 2,048 | 0.12 / 0.34 / 0.35 / 0.46 | | 0.16 / 0.44 / 0.47 / 0.60 |
| rows of the 5,136 kept tokens only | +0.025 | | +0.030 |
| write into rows of rank ≥ 2,048: paraphrase loss, own write → all eight | 1.004 → 1.004 | | 1.034 → 1.034 |
| the same: texts without a written token that are bit-identical | 55 of 55 | | 55 of 55 |
| the same: unrelated texts affected, measured / predicted | 8.3% / 8.4% | | 8.3% / 8.4% |
| write into rows of rank ≥ 512: bit-identical texts | 46 of 46 | | 46 of 46 |

Three of the registered claims are about these numbers. *The chat loss fell* (claim 4, first half), by 0.02 to
0.04 nats, less than I expected of 3.4 × 10⁹ more conversation tokens. *The web loss rose by 0.07 to 0.08
nats*, more than the 0.05 the claim allowed: the second half of claim 4 fails. The stream replayed 5% of the
old data; that did not hold the web text, and the whole loss was there after a quarter of the run (2.850 at
step 3,241), in the stretch with the highest learning rate. This is the silent forgetting the plan cited and
meant to avoid, on text whose tokens the new stream rarely shows; a higher replay share or a lower starting
rate are the untested remedies. *The token tables are worth more than before, and more per occurrence the rarer
the token* (claim 3): the law of P.2 survived the run and sharpened, by about three tenths in every band. The
exactness and the frequency bound of P.3 (claims 1 and 2) hold on the new weights as they must: they are
properties of the allocation, not of the values in it.

### P.9 What the continued reader does with the library, and the reader's own test

*The continued model stopped copying.* A snapshot at a quarter of the run (step 3,241), shipped and judged like
the one-day weights, scored 0.465 on the development prompts against 0.477 before the run: prompts without an
exemplar had improved (0.397 → 0.436; 16 better, 8 worse) and prompts with one had got worse (0.556 → 0.465;
5 better, 18 worse, p = 0.011; among conversational prompts with an exemplar 14 / 3 / 12 became 4 / 6 / 19).
The replies show why. Asked how to tell whether an egg is fresh, beside a stored answer to the same question
in other words, the one-day weights return the stored text (put it in water; flat on the bottom is fresh). The
continued model writes five numbered points of its own, and they are wrong. Leave-one-out retrieval (P.7)
never shows an entry beside its own question, so every exemplar of training was the answer to *another*
question, and the model learnt exactly that. It is the right lesson for the loss and the wrong one for a
reader: the teacher's text is better than what 20M active parameters write. Loose exemplars did not help
either (the 79 prompts they change: 0.405 → 0.411), though with match tags an exemplar that differs in its
numbers no longer hurts (3 better, 3 worse; before the run 3 and 5).

*So the question moved from the weights to the decision: when may a stored answer be given as it stands?* The
library's own rule (the same question, nearly word for word) takes 12 of 200 development prompts. The index's
overlap cannot tell "why does ice float on water" from "why does oil float on water", and does not recognise a
question asked in other words. The reader can be asked directly. For a stored entry (q_e, a) and the user's
question q, two teacher-forced passes of the model over the same answer in a bare two-turn conversation give,
for every token of a, d_t = log p(a_t | q_e, a_<t) − log p(a_t | q, a_<t): what the user's question costs that
token. The test has a global and a local part (`bes/decode/fitness.py`):

    fit    = − mean_t d_t                       the answer is as probable under q as under q_e      (fit ≥ −τ)
    window = max_t (d_t + d_{t+1} + d_{t+2})     no three tokens are impossible under q               (window ≤ w)

Subtracting the pass under the stored question is what makes answers comparable (a fact and a poem have very
different likelihoods; each is compared with itself); the likelihood under the user's question alone predicts
nothing (AUC 0.57). The local part is there because a mean cannot see one decisive word: beside "what does
*break the ice* mean" the stored explanation of *break a leg* has three impossible tokens in three hundred and a
mean of −0.05 to −0.07. This is Theorem 3 of the plan read from the other side: the kernel that decides what
is "near" can be word overlap, or it can be the reader's own conditional.

**Table P9. The reader's test against the judge** (development prompts, single turn; stored answers of the
shipped library against Falcon's replies: the 125 judged candidates of 57 prompts that the library's own rule
does not take; fit and window from the snapshot at step 3,241; `lab/m6v/fit_table.py` on `lab/m6v/raw/fit_mid/`).

| candidates | n | the stored answer | the model's own reply to the same prompt |
|---|---|---|---|
| fit ≥ −0.05 | 33 | 0.788 | 0.515 |
| −0.10 ≤ fit < −0.05 | 48 | 0.573 | 0.562 |
| −0.20 ≤ fit < −0.10 | 38 | 0.474 | 0.632 |
| fit < −0.20 | 6 | 0.583 | 0.500 |
| window ≤ 6 nats | 54 | 0.731 | 0.528 |
| 6 < window ≤ 8 | 23 | 0.609 | 0.630 |
| window > 8 | 48 | 0.448 | 0.583 |

The median window is 4.6 nats where the stored answer won, 8.0 where it tied and 8.3 where it lost. As
predictors of "the stored answer is at least as good as the model's own reply" the fit and the window have an
area under the curve of 0.66 each and the index's overlap 0.59.

The rule (`AnswerLibrary.verbatim(fit=…)`, settings `decode.verbatim_fit`, `verbatim_fit_rank`,
`verbatim_fit_window`): the library's own rule first, without asking the model; then, among at most three
candidates without a contrast whose answers meet the user's explicit constraints and whose overlap reaches a
floor, the one with the best fit, if its fit is at least −τ and its window at most w. It reads the weights a
reply reads and nothing else, at most six more passes of at most 320 tokens per reply; the candidates' text
was already counted against the library's read cap. The window threshold was fixed at 8 nats (about one
content word) from the snapshot's data before any reply of the final model was judged; τ and the overlap
floor were chosen on the final model's development verdicts by a rule written down before those existed
(`docs/CHANGES.md` 21.7, 21.9).

*End markers.* 5,069 of the 172,273 shipped entries hold the characters `</s>` followed by a made-up next
turn: the teacher's text carries the end-of-text marker of its source. The continued model writes it too (7
of 200 development replies of the snapshot, scoring 0.29). With `decode.cut_end_markers` a stored answer and
every candidate reply end at the first such marker before the selection sees them; the shipped model has it on.

*The final weights on the development prompts.* The same structure, sharper.

**Table P10. The final model on the 200 development prompts** (against Falcon, registered judge; paired with the
one-day weights in the same shipped form; `lab/m6v/raw/final_dev/`, `final_pool/`, `final_loose/`).

| | before the run | final weights | better / worse | p |
|---|---|---|---|---|
| all prompts, the library's own rules | 33 / 42 / 125 (0.477) | 28 / 35 / 137 (0.482) | 30 / 26 | 0.69 |
| prompts without an exemplar, 117 | 5 / 29 / 83 (0.397) | **11 / 18 / 88 (0.470)** | **22 / 6** | **0.004** |
| prompts with exemplars, 71 | 20 / 12 / 39 (0.556) | 10 / 16 / 45 (0.458) | 7 / 18 | 0.043 |
| of these, conversational, 29 | 14 / 3 / 12 (0.690) | 5 / 10 / 14 (0.414) | 2 / 15 | 0.002 |
| stored answer by the library's own rule, 12 | 8 / 1 / 3 (0.792) | 7 / 1 / 4 (0.750) | 1 / 2 | 1.00 |
| *composed from judged pieces, single-turn prompts only:* | | | *against the first row* | |
| + stored answers at overlap ≥ 0.6 | | 36 / 31 / 133 (0.5125) | 10 / 1 | 0.012 |
| + the reader's test, τ = 0.075, overlap ≥ 0.5 | | 37 / 31 / 132 (0.5150) | 13 / 3 | 0.021 |
| **+ the reader's test with window ≤ 8** | | **38 / 31 / 131 (0.5175)** | **13 / 2** | **0.007** |
| + the same and loose exemplars from 0.4, by score | | 39 / 31 / 130 (0.5200) | 16 / 4 | 0.012 |

What the run did to the weights is the second row: where the model answers on its own it is clearly better than
it was (instructions 0.400 → 0.475, arithmetic 0.333 → 0.400, code 0.517 → 0.583). What it undid is the third
and fourth: the copying that had carried the conversational prompts. With the library's own rules the two cancel
(0.482 against 0.477). The reader's test gives the teacher's text back where it fits: 21 prompts change to a
stored answer, 13 of them for the better and 2 for the worse. By the rule registered beforehand (highest
composed score; the prompts it changes must improve more often than not; fewest changes among equals) the
shipped model decodes with τ = 0.075, overlap floor 0.5 and a window of 8 nats; by the letter of the same rule
it also shows exemplars that score between 0.4 and 0.5 (0.5200 against 0.5175: three prompts better, two worse,
which is no visible effect, and exemplars that differ in their numbers are still harmful, 3 better and 6
worse). The real path agrees with the composition where it was checked: the entries taken, their fit and
their text were the same for the prompts run through the shipped decoder, and replies of prompts the rule
does not touch were identical, character for character, to those of the default decoder made on another
machine. These numbers were chosen on the prompts they are measured on; the test prompts were not.
