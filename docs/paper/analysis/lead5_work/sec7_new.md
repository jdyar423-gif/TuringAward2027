## 7. Evaluation: protocol, deviations and results

### 7.1 Protocol

The protocol (`docs/EVAL_PROTOCOL.md`) is in the branch's root commit (`5ff33c5`, 23 November 00:35 UTC)
and unchanged since: it predates every TPU run and every reply of the full-scale model, not the
micro-scale runs of gates E2, E4 and E5. **Prompts**: 400 original English prompts in five categories, a
quarter with a second user turn, 95 with machine-checkable constraints, split by the SHA-256 of their
identifier into `dev` and `test` (200 each; `test`: 252 turns, 30 maths and 48 constrained prompts);
decisions use `dev`, and `test` is used once. Sides are seeded and balanced, and the scorer refuses any
rating whose commitment to the hidden key does not match. **Readers**: the registered judge
`qwen3-4b-2507`, read on its next-token distribution over A, B and C in both orders, with `gemma4-e4b` of
another family as a second judge; a side wins only if both orders choose it. **Exact scores** come first
and are trusted over a reader. **Statistics**: the score (wins + ties/2)/n with a Wilson interval, an exact
two-sided sign test, slices by category, turns and length ratio. **What a claim must disclose** (§3 of the
protocol): results with K = 1 as well as with our K particles, with the library on and off, and two control
arms of each opponent run with the same K, one kept by the opponent's own mean log-probability (the
K-matched control) and one by our selector (the selector-matched control); MobileLLM-R1.5 is credited with
the better of its two arms on `dev`, decided before `test` is looked at (§2).

### 7.2 Deviations from the protocol

1. **The readers.** The registered judge read the four development pair sets in both orders and, at the
   time of writing, 136 and 89 of the 200 pairs of the two test sets [TBD: its full pass, running on a CPU
   runtime since 20:41 UTC, expected at about 22:00]; the second registered judge, which decided 3 of 60
   development pairs per set (Appendix E.6), was not run on the test pairs. Opus 5.5 agents, of the family of
   the agents that built the system, read the development pairs once (`final`) or twice on exchanged sides
   (`v9`), and every test pair twice on exchanged sides, from written instructions
   (`N/final_eval/JUDGE_INSTRUCTIONS.md`, Appendix J): an addition, reported next to the registered judge and
   never averaged with it.
2. **Eleven test prompts were seen** by mid-run CPU samples (chat-002, chat-017, math-034, instruct-034,
   instruct-062, writing-100, code-026, math-001, code-002, writing-001, instruct-001); two of them
   motivated the data rules of Appendix D. Every test result is also reported on the 189 untouched prompts.
3. **The proxy held-out rows contain test-split user turns** (Section 6.5); they are left out of
   comparisons.
4. **MobileLLM's two arms.** No blind comparison of the `neutral` arm was made on `dev`, so on `test` our
   model is compared with the `recommended` arm by default, not by the protocol's rule. The exact scores on
   `dev` favoured `recommended` (Table E3); on `test` the two are close (Table 8).
5. **At training time** (Appendix B.5): an E2 loss reading chosen after seeing the data; registers kept
   despite failing their gate; the E5 copy criterion judged wrong after the fact; the pre-registered
   configuration comparison not run; the configuration changed mid-run.
6. **K = 1 and the matched controls, late and incomplete.** Our replies are chosen among K = 6 particles
   and the opponents draw one sample. Made after training, on the test split: Falcon-H1-Tiny's two control
   arms, scored exactly (Table 8) and read blind against the official model (Table 7); and our K = 1 arm
   (the official export with one particle), scored exactly [TBD: its Opus readings against both opponents,
   started at 21:23 UTC]. MobileLLM-R1.5's two arms do not exist yet [TBD: generation in progress on two CPU
   runtimes since 19:17 UTC, 40 of 200 prompts done by 21:13]. On `dev` the run's own decoding selection
   compared one particle with six by exact score: one particle was worse on 18 prompts and better on 1 of
   200 (p = 7.6 × 10⁻⁵; Section 7.4). Library on and off does not apply: `v9` ships no library.

### 7.3 What 200 comparisons can resolve

With the Opus readers' tie rates on `test` (28.5% against Falcon-H1-Tiny, 34.5% against MobileLLM-R1.5,
53.5% between our two sets of weights), n = 200 detects with 80% power (two-sided 5% sign test) a decisive
win rate of 0.5 ± 0.116, ± 0.121 and ± 0.143 (± 0.119, 0.125 and 0.147 on the 189 untouched prompts;
`lead5_work/numbers.json`, `bes.eval.proxy.score.n_required`). The registered judge's tie rates on its
partial pass, 68% and 72%, give it about half that resolution. The observed decisive rates against the two
opponents, 0.245 and 0.885, are far outside these bounds; the 0.634 between our weights is not.

### 7.4 The official model

The official model is what the run's own rules shipped (Appendix C.4): the weights of step 51,812 after
preference optimisation, with the default decoding. Preference optimisation (DPO [74], β = 0.1, one epoch of
551 steps of 64 pairs, almost all from UltraFeedback [76]) tried two learning rates from the trained weights;
both passed every check (held-out preference accuracy 0.688 and 0.694 against 0.500 at the start, standard
error 0.011; held-out assistant-token loss +0.86% and +1.10%, limit 3%; sampled length 0.99 and 0.95 times
the start's and repeated 4-grams 0.86 and 0.87 times, inside their bands), and the rule kept the higher
accuracy, 3 × 10⁻⁶. Decoding selection keeps the default unless a variant's exact score on the 200
development prompts is higher on significantly more prompts than lower: one particle was worse on 18 and
better on 1 (p = 7.6 × 10⁻⁵), no length band better on 2 and worse on none (p = 0.50), so the default stayed
(6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library). The export stores all 98,036,104 numbers
in float16 and passed its 16-bit check (2.34440 against 2.34443 nats on 43,138 held-out tokens). Our test
replies were generated on 40 host cores from an export with the same `weights16.npz` (SHA-256
`2da7cbd2…`) and decoding choice as the run's own (19:31–20:06 UTC; Appendix J); the same weights are
released in the repository (`release/main16-v9`).

### 7.5 Blind comparisons on the test split

Table 7 gives every reading of the test pairs; Figure 7 draws the Opus readers' with the development
studies. The three kinds of reader are kept apart: the exact scorer (Section 7.6), the registered judge
(about two thirds of its pairs undecided) and the Opus readers (a third undecided), whose resolutions
differ too much to be averaged.

**Table 7. Blind comparisons on the 200 test prompts** (W / L / T from the first-named model's side, T
including "both bad" and pairs whose two orders disagree; score (wins + ties/2)/n with 95% Wilson interval;
decisive win rate with its interval; exact two-sided sign test; orders: pairs on which a reader's two orders
agree; first: mean preference for the first-shown reply; "189": the untouched prompts among the rated ones;
recomputed from `study/proxy/ratings/` with the repository's key check by `lead5_work/scripts/final_results.py`,
which reproduces every Opus report in `study/proxy/reports/ab/`; the registered judge's pass is partial).

| comparison | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders; first |
|---|---|---|---|---|---|---|---|
| official vs Falcon-H1-Tiny | Opus, two readings | 200 | 35 / 108 / 57 | 0.318 [0.257, 0.385] | 0.245 [0.182, 0.321] | < 0.0001 | 90.5%; +0.005 |
| | … on the 189 untouched prompts | 189 | 32 / 101 / 56 | 0.317 [0.255, 0.387] | 0.241 [0.176, 0.320] | < 0.0001 | |
| | `qwen3-4b-2507` (registered), partial | 136 | 9 / 34 / 93 | 0.408 [0.329, 0.492] | 0.209 [0.114, 0.352] | 0.0002 | 59.6%; −0.141 |
| | … untouched prompts among them | 129 | 9 / 33 / 87 | 0.407 [0.326, 0.493] | 0.214 [0.117, 0.359] | 0.0003 | |
| official vs Falcon-H1-Tiny, K = 6 kept by its own log-probability (K-matched) | Opus, two readings | 200 | 35 / 112 / 53 | 0.308 [0.248, 0.375] | 0.238 [0.176, 0.313] | < 0.0001 | 93.5%; +0.020 |
| | … on the 189 untouched prompts | 189 | 34 / 106 / 49 | 0.310 [0.248, 0.379] | 0.243 [0.179, 0.320] | < 0.0001 | |
| official vs Falcon-H1-Tiny, K = 6 kept by our selector (selector-matched) | Opus, two readings | 200 | 29 / 116 / 55 | 0.283 [0.225, 0.349] | 0.200 [0.143, 0.272] | < 0.0001 | 92.5%; +0.018 |
| | … on the 189 untouched prompts | 189 | 27 / 110 / 52 | 0.280 [0.221, 0.348] | 0.197 [0.139, 0.272] | < 0.0001 | |
| official vs MobileLLM-R1.5 | Opus, two readings | 200 | 116 / 15 / 69 | 0.753 [0.688, 0.807] | 0.885 [0.820, 0.929] | < 0.0001 | 96.0%; −0.005 |
| | … on the 189 untouched prompts | 189 | 109 / 14 / 66 | 0.751 [0.685, 0.808] | 0.886 [0.818, 0.931] | < 0.0001 | |
| | `qwen3-4b-2507` (registered), partial | 89 | 22 / 3 / 64 | 0.607 [0.503, 0.702] | 0.880 [0.700, 0.958] | 0.0002 | 73.0%; +0.006 |
| | … untouched prompts among them | 84 | 20 / 3 / 61 | 0.601 [0.494, 0.699] | 0.870 [0.679, 0.955] | 0.0005 | |
| official vs trained weights (before preference optimisation) | Opus, two readings | 200 | 59 / 34 / 107 | 0.563 [0.493, 0.629] | 0.634 [0.533, 0.725] | 0.012 | 91.5%; +0.010 |
| | … on the 189 untouched prompts | 189 | 55 / 33 / 101 | 0.558 [0.487, 0.627] | 0.625 [0.521, 0.719] | 0.025 | |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 200 | 144 / 6 / 50 | 0.845 [0.788, 0.889] | 0.960 [0.915, 0.982] | < 0.0001 | 93.5%; +0.013 |
| | … on the 189 untouched prompts | 189 | 137 / 5 / 47 | 0.849 [0.791, 0.893] | 0.965 [0.920, 0.985] | < 0.0001 | |

**Against Falcon-H1-Tiny** the official model loses in every category (chat 10 / 24, code 10 / 13 with
p = 0.68, instruction-following 7 / 21, maths 4 / 18, writing 4 / 32) and on two-turn prompts (7 / 30). Our
replies fared best where they were the shorter of the pair (93 pairs: score 0.419, 26 / 41, p = 0.086) and
worst where they were the longer (35 pairs: 0.143): the readers did not reward our length. **The control
arms** give Falcon-H1-Tiny six samples per reply, as our decoder has six particles; it was preferred again,
in 112 and 116 of 147 and 145 decisive pairs. Paired by prompt, our score against the released preset is
higher than against the K-matched arm on 29 prompts and lower on 25 (mean difference 0.010, p = 0.68), and
than against the selector-matched arm on 34 and lower on 21 (0.035, p = 0.10): the loss to Falcon-H1-Tiny is
not an artefact of six particles against one sample, and Falcon-H1-Tiny does not need our selector to win.
**Against MobileLLM-R1.5** the official model wins in every category but maths (chat 28 / 1, code 16 / 2,
instruction-following 29 / 0, writing 36 / 0; maths 7 / 12, p = 0.36) and in every length bucket (Appendix M,
Tables M2–M3). **The registered judge**, part-way, points the same way in both sets with fewer decided pairs:
where it and the Opus readers both chose a side, they chose the same one in 40 of 44 pairs against
Falcon-H1-Tiny and 22 of 22 against MobileLLM-R1.5 (Cohen's κ over win, loss and tie 0.22 and 0.33); as on
`dev`, its two orders often disagree (60.6% agree against Falcon-H1-Tiny) and it prefers the second-shown
reply (−0.139). The 11 seen prompts change no conclusion: every restriction to the untouched prompts moves a
score by at most 0.006.

### 7.6 Exact scores and degenerate replies

**Table 8. Exact scores on the 200 test prompts (252 turns)** (`bes.eval.exact_score` on every reply,
recomputed by `lead4_work/scripts/exact_split.py` from the replies and equal, field by field, to the run's
exact reports in `study/proxy/reports/exact/`, `lead5_work/exact_counts.json`; in brackets, the 189 untouched
prompts: 28 maths and 44 constrained prompts, 240 turns; ours K = 1: the official export with one particle;
Falcon K = 6: six samples per reply, kept by its own mean log-probability or by our selector).

| measure | ours, official (K = 6) | ours, K = 1 | ours, trained weights | Falcon-H1-Tiny | Falcon, K = 6, own log-probability | Falcon, K = 6, our selector | MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |
|---|---|---|---|---|---|---|---|---|
| maths answers correct, of 30 (of 28) | 6 (6) | 5 (5) | 6 (6) | 15 (14) | 17 (16) | 18 (17) | 12 (11) | 11 (10) |
| prompts with every constraint met, of 48 (of 44) | 38 (34) | 37 (33) | 41 (37) | 33 (29) | 35 (32) | 42 (38) | 8 (8) | 9 (8) |
| single constraints met, of 56 | 46 | 44 | 48 | 41 | 42 | 50 | 10 | 11 |
| degenerate replies, of 252 (share) | 5 (2.0%) | 17 (6.7%) | 3 (1.2%) | 2 (0.8%) | 1 (0.4%) | 0 | 158 (62.7%) | 148 (58.7%) |
| … cut at the token limit; looping tail; repeated lines | 5; 0; 0 | 17; 0; 0 | 3; 0; 0 | 1; 0; 0 | 0; 0; 0 | 0; 0; 0 | 150; 14; 54 | 146; 17; 50 |
| stock refusals; stock deflections | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| words per reply, median (mean) | 101 (130) | 105 (150) | 99 (119) | 155 (173) | 147 (174) | 145 (168) | 395 (474) | 517 (534) |

Against the released Falcon-H1-Tiny our model meets every constraint on more prompts (38 against 33 of 48)
and answers far fewer maths prompts (6 against 15 of 30). **What the selector-matched arm shows about that
constraint score**: given six samples per reply and our selection rule (the constraints compiled from the
prompt, the failure filter, the length band, the maths majority vote and the consensus; not our exact
potentials, which act during sampling), Falcon-H1-Tiny meets every constraint on 42 prompts, four more than
our model; kept by its own log-probability, 35. With one particle, where no selection happens but the
potentials still act, our model meets 37. So the constraint lead over the released Falcon-H1-Tiny is our
decoder's, mostly its exact potentials, not our model's, and the same selection gives Falcon-H1-Tiny more than
our decoder gives us; the maths gap does not close (5 and 6 against 15 to 18). One particle also cut 17 of
252 replies at the token limit, against 5 with six. MobileLLM-R1.5's replies are degenerate in 158 of 252
turns under its recommended preset (150 of them cut at the token limit) and in 148 under the neutral one; its
blind defeat is largely a defeat of those replies. No model gave a stock refusal or deflection.

### 7.7 Before and after preference optimisation

Read the same way on the same prompts (4 of the 200 got identical replies), the official weights were
preferred to the trained weights in 59 pairs and the trained ones in 34 (score 0.563 [0.493, 0.629],
p = 0.012; 189 prompts: 55 / 33, p = 0.025), mostly in code (10 / 1) and chat (16 / 8), not in maths (5 / 5).
The exact scores did not move with it: maths 6 and 6 of 30, constraints 38 against 41 of 48, degenerate
replies 5 against 3; the replies grew longer (mean 130 against 119 words). The stage moved the held-out
quantities it was built to move a little (preference accuracy 0.500 → 0.694; assistant-token loss +1.10%) and
the held-out language-model loss of the comparison set by +0.012 nats (Section 6.5). One epoch of DPO on
35,201 pairs (424 s of training at the chosen rate, in a stage of 30 minutes) bought a modest blind
preference, measured by one kind of reader only.

### 7.8 In absolute terms

Both replies were judged bad ("both bad") in 89 of 200 pairs in each reading of the set in which both replies
are ours (official against trained weights, 44.5%); at step 31,912 on development prompts the set of two
decodings of one checkpoint had 44 of 60 (73%, one reading). Against Falcon-H1-Tiny 44 and 41 of 200 pairs
were "both bad" (22.0% and 20.5%), against MobileLLM-R1.5 66 and 64 (33.0%, 32.0%), and in the reference set,
where no reply is ours, 38 and 38 (19.0%) (raw verdicts of the chunk files, `lead5_work/final_results.json`).
At the end of the run a little under half of the official model's replies to these prompts were judged
useless together with a sibling reply of the same model; the prompts and the kind of pair differ from the
development set, so the fall from 73% is indicative only.

### 7.9 The trajectory

**Table 9. From mid-run checkpoints to the official model** (Opus readers, W / L / T and score; registered
judge, decisive pairs W / L; exact scores of the same replies; development studies on the first 60 `dev`
prompts, 9 maths and 14 constrained, in full in Appendix E.6; the official model's `dev` replies are the run's
own decoding-selection replies, which no reader compared).

| model (split, n) | vs Falcon-H1-Tiny: Opus | vs MobileLLM-R1.5: Opus | registered judge, W / L (Falcon; MobileLLM) | maths correct | every constraint met |
|---|---|---|---|---|---|
| `final` 25,280, p = 0.44 (`dev`, 60; one reading) | 2 / 43 / 15, 0.158 | 15 / 8 / 37, 0.558 | 2 / 16; 7 / 4 | 0 of 9 | 12 of 14 |
| `v9` 31,912, p = 0.56 (`dev`, 60; two readings) | 3 / 39 / 18, 0.200 | 20 / 7 / 33, 0.608 | 1 / 16; 7 / 2 | 1 of 9 | 12 of 14 |
| official, 51,812 after preference optimisation (`dev`, 60) | – | – | – | 0 of 9 | 13 of 14 |
| official (`test`, 200; two readings) | 35 / 108 / 57, 0.318 | 116 / 15 / 69, 0.753 | 9 / 34; 22 / 3 (partial) | 6 of 30 | 38 of 48 |

The development and test prompts are two halves of one set, so the rows are comparable in kind, not
calibrated against each other, and the development studies differ in checkpoint, configuration and number
of readings. Against Falcon-H1-Tiny the official model's decisive win rate (0.245) is above that of the
mid-run checkpoints (0.044 and 0.071 on `dev`); against MobileLLM-R1.5 its lead is larger (0.885 against 0.652
and 0.741). The opponents' own reference (Falcon-H1-Tiny over MobileLLM-R1.5 in 144 of 150 decisive test
pairs) is the scale: the comparison that informs is the one our model loses.

![Figure 7](figures/F11_blind.svg)

**Figure 7.** Blind pairwise comparisons by the Opus readers (Table 7, Table 9): `final` at step 25,280 on
the first 60 development prompts, read once per pair; `v9` at step 31,912 on the same prompts, read twice on
exchanged sides; the official model on the 200 test prompts, read twice. Bars: wins, ties or "both bad",
losses; dots and whiskers: the score with its 95% Wilson interval (about ±12 points at n = 60, ±7 at
n = 200). The figure draws only this added reader, because it decides most pairs; the registered judge's
readings, which point the same way where it decides, are in Tables 7 and E2. The studies differ in
checkpoint, configuration, split and number of readings: the change between rows is not the effect of one
of them.

### 7.10 What limits each statement

**Preferred over MobileLLM-R1.5** (116 to 15): by readers of the builders' family who are an addition to the
registered protocol, with the registered judge part-way (22 to 3 on 89 pairs); over an opponent whose replies
are degenerate in 63% of turns under its recommended preset in our harness (a converted community file with
budget forcing, Section 2.3), so that the lead measures that opponent's state here as much as our model; on
prompts of which 11 were seen (the 189 give the same result); with six particles against one sample (the
K = 1 arm and MobileLLM-R1.5's control arms are not finished); for one run. **Not preferred over
Falcon-H1-Tiny** (35 to 108): by every reader, in every category, on the 189 prompts too; Falcon-H1-Tiny ran
at llama.cpp's default sampling, which may understate it, and with our selector over six samples it scores
higher still on exact measures. No human has read the pairs.
