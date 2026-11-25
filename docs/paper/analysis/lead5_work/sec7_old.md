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
Wilson interval, an exact two-sided sign test, slices by category, turns and length ratio. **What a
claim must disclose** (§3 of the protocol): results with K = 1 as well as with our K particles, with the
library on and off, and two control arms of each opponent run with the same K, one kept by the
opponent's own mean log-probability (the K-matched control) and one by our selector (the
selector-matched control); MobileLLM-R1.5 is credited with the better of its two arms on `dev`, decided
before `test` is looked at (§2).

### 7.2 Deviations from the protocol

1. **The readers.** Both registered judges read the four development pair sets in both orders (Table 7);
   the same pairs are also read by Opus 5.5 agents, of the family of the agents that built the system,
   once (the `final` sets) or twice on exchanged sides (the `v9` sets), and the test pairs will be read
   twice on exchanged sides: an addition, reported next to the registered judges.
2. **Eleven test prompts were seen** by mid-run CPU samples (chat-002, chat-017, math-034, instruct-034,
   instruct-062, writing-100, code-026, math-001, code-002, writing-001, instruct-001; checked with
   `bes.eval.proxy.prompts.load()`); two of them motivated the data rules of Appendix D. Every test result
   will also be reported on the 189 untouched test prompts (R11).
3. **The proxy held-out rows contain test-split user turns** (Section 6.5); they are left out of
   comparisons.
4. **MobileLLM's two arms.** The protocol credits MobileLLM-R1.5 with the better of its two arms on `dev`.
   Only the `recommended` arm has been compared blind with our model, and no blind comparison of the
   `neutral` arm on `dev` is scheduled before the test readings begin (about 19:30 UTC on 24 November),
   so on `test` our model is compared with the `recommended` arm by default, not by the protocol's rule. The
   exact scores on the 60 development prompts, which the protocol reads first, favour the
   `recommended` arm (Table 8: maths 4 against 1 of 9, constrained prompts met 2 against 1 of 14, replies
   cut at the limit 42 against 54 of 75); the `neutral` arm's exact scores on `test` will be reported
   (Table M4).
5. **At training time** (Appendix B.5): an E2 loss reading chosen after seeing the data; registers kept
   despite failing their gate; the E5 copy criterion judged wrong after the fact; the pre-registered
   configuration comparison not run; the configuration changed mid-run.
6. **No K = 1 result and no matched controls.** Our replies are chosen among K = 6 particles (Appendix
   E.2) and the opponents draw one sample. The protocol's K-matched and selector-matched opponent arms
   were not generated and are not scheduled, and no K = 1 test result of our model is planned (unless the
   decoding selection ships one particle, Table M8); library on and off does not apply, since `v9` ships
   no library. A result on `test` therefore cannot say how much of a difference comes from selecting
   among six samples rather than from the model. Selection changes the reply distribution by at most
   log 6 = 1.79 nats in KL divergence (Appendix E.2); on `dev`, one particle against six on `final` at
   step 25,280 was 1 / 3 / 56 for one particle (Table E1, p = 0.63), which separates nothing.

### 7.3 What 200 comparisons can resolve

The protocol's power table gives the comparisons needed with 30% and with 50% ties. On the development
pairs of `v9`, ties (including "both bad" and orders that disagree) were 18 and 33 of 60 for the Opus
readers (against Falcon-H1-Tiny and MobileLLM-R1.5) and 43 and 51 for the registered judge. At 30% and 55% ties, n = 200 detects with 80%
power (two-sided 5% sign test) a decisive win rate of 0.5 + 0.117 and 0.5 + 0.146; at the registered
judge's 72% and 85%, only 0.5 + 0.182 and 0.5 + 0.246; on the 189 untouched prompts, 0.121, 0.150, 0.187
and 0.252 (`bes.eval.proxy.score.n_required`; `lead3_work/numbers.json`). A non-significant test result
will therefore leave open differences of 12 to 25 points.

### 7.4 The standing so far

Tables 7 and 8 and Figure 7 collect what has been measured against the opponents; none of it is the
planned test.

**Table 7. Blind comparisons by reader** (W / L / T from the first-named model's side, T including "both
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
42 of its 75 replies ran into the token limit (Table 8), and the readers' notes on the reference pairs
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
`final` at step 25,280, 46 to 55 of 60 pairs per decoding comparison (Appendix E). At p = 0.56, most of the
model's replies to these prompts were judged bad, mainly for wrong content and incoherence.

**Table 8. Exact scores and bits per byte** (the same 60 `dev` prompts, 75 turns, default decoding;
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
model needs 0.98 bits per byte at step 46,178 and the opponents 1.57 and 1.50 (the same rows: 258,814
bytes as the opponents' tokenizers cut them, 258,817 as ours does);
this favours us by construction (instruction-tuned opponents read on raw web text, training data
unknown), and next-token loss on the pre-training distribution said little about preference in chat
here. Two decoding studies on the same prompts could not separate any setting from the default
(Appendix E, Table E1).

![Figure 7](figures/F11_blind.svg)

**Figure 7.** Blind pairwise comparisons on the first 60 development prompts, Opus readers (Table 7):
`final` at step 25,280 read once per pair, `v9` at step 31,912 read twice on exchanged sides. The figure
draws only this added reader, because it is the one that decides most pairs; the registered judge
decided 17 and 9 of the 60 `v9` pairs (against Falcon-H1-Tiny and MobileLLM-R1.5) and the second judge 3
and 3, so their bars would be almost all ties. Their readings, which point the same way where they
decide, are in Table 7. Bars: wins, ties or "both bad", losses; dots and whiskers: the score
with its 95% Wilson interval (about ±12 points at n = 60); the two studies differ in checkpoint,
configuration and number of readings, so the change between them is not an effect of `v9`. The
test-split comparison is [TBD: after the evaluation of 24–25 November 2026].

### 7.5 Results to come

The quantities the evaluation will fill are table shells R1–R12 in Appendix M (Tables M1–M12), each
with the file it will come from [TBD: after the evaluation of 24–25 November 2026].

