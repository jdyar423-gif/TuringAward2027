# REVIEW_1: programme-committee review of the first full draft

Reviewer role: hostile but fair member of the committee that decides best-paper nominations.
Reviewed: `paper.md` (state of facts 10:00 UTC), `CLAIMS.md`, `OUTLINE.md`, `related_work.md`,
`refs.md`, `tables.md`, `FIGURES.md`, the eight figures, and `FACTS_2.md`. Each was checked against the
raw sources, not against the claims ledger. Review written 24 Nov 2026, 11:00–12:10 UTC.

Abbreviations as in `CLAIMS.md`: `R/` = the repository (HEAD is now `8a86bcb`, not `9a4d652`),
`N/` = the session scratchpad, `S/` = the old scratchpad
(`/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/`), `W/` = `N/wf/`.
My recomputation scripts are in `N/paper/review1_work/`: `steps.py`, `win.py`, `mix.py`,
`flops_check.py`, `stats.py`; the graft re-check log is in `graftcheck/`. Unless a point says
"my arithmetic", every number in this review is copied from the file it names.

Severity: **must fix** (blocks acceptance), **should fix** (a reviewer would raise it), **optional**.

## Strengths (so that the criticism is read in proportion)

- The draft separates on-chip measurements from compile-only estimates and says which is which.
  It reports 13 changes that were not adopted, and it states the deviations from its
  pre-registered rules. Few papers do any of this.
- Most numbers I traced are right. Correct to the digit: the ledgers of both configurations,
  Tables 3, 5, 6, 7 and 8, the chip step times, the p and learning-rate values in the train logs,
  the opening and deflection scans, and the throughput history. I fetched three references from
  arXiv and they say what `refs.md` says: [21], [47], and [31] (abstract only; its loss numbers
  come from tables I did not read). [63] and [32] I checked from memory.
- The limitations section is candid, and so is the "what we do not claim" paragraph.

## (1) The three strongest reasons to reject the paper as it stands

**1. The thesis is stated as a general law, but the evidence is one pair of implementations
whose engineering effort differs. The positive half of the thesis is never tested, and the
paper's own data contradict it. Must fix.**
- The claim appears in the title, §1, §17 and the OUTLINE: parameter counts "do not predict
  training cost; bytes moved and the shapes they move in do".
- The negative half rests on a single comparison of `final` and `v9`: 3.1% more reads and 3.6%
  more analytic FLOP, but a time ratio of 1.557 per piece and 1.620 per step
  (`S/v9_measA.txt`). That gathers, scatters and small contractions cost more than their FLOP is
  already known; the paper itself cites [63].
- The ratio measures implementation maturity as much as design.
  - `final` received the rewrites of 23 Nov (289.3 → 162.0 ms per micro-batch, Table 6), the
    pieces (Table 7) and the kernel work of 24 Nov (×1.212 per piece, Table 8).
  - `v9` received about one hour of such work. In that hour its step fell from 1,433.1 to
    1,287.1 ms (`S/v9_measD.txt`; `R/docs/CHANGES.md` §19.7).
  - The ratio is now 1,287.1 / 884.5 = 1.455 on synthetic batches. Live it is
    281,399 / 196,864 = 1.43 (medians of the train-log rates, my arithmetic). Both values are
    already outside the "1.54–1.62" in the abstract, §1, §12.1 and §17.
  - `W/v9_speed.md` lists further layouts that were never implemented.
- The positive half (bytes and shapes *predict* the cost) is never tested as a prediction. No
  model of bytes moved is fitted or validated. The only model of bytes or cycles in the paper, the
  compiler's cost model, missed the measured *differences* by factors from 0.37 to 8:

  | change | predicted | measured on the chip | source |
  |---|---|---|---|
  | `sorted` instead of `take` | +130 ms | +356 ms | `W/v9_speed.md`; `S/v9_measB_sorted.txt` |
  | `tokn` | −64 ms | −8 ms | `S/v9_measD.txt` |
  | `flat` | −46 ms | −27 ms | `S/v9_measD.txt` |
  | `v9` − `final` | +624 ms | +549 ms | `W/v9_speed.md`; `S/v9_measA.txt` |

- §7.8 attributes v9's extra time to layout, but that attribution is compile-only. The measured
  evidence is the switch series, which removed 146 of the 549 extra milliseconds (27%; `rows`
  alone took 111 ms). The remaining 403 ms has not been attributed by any measurement.
- Fix: either narrow the claim ("in our implementation on one v6e, at the switch, …") or earn it
  with item 37.

**2. The paper has no outcome and identifies no effect. Its contributions are applications of
known methods. Must fix.**
- Every claim about the trained model is either [TBD] or confounded within the one run.
  - Pages: there is no `v9` trained from scratch and no `final` trained to the end. The data
    stream, its seed and the phase of the learning-rate schedule all changed at the graft.
  - Graft against restart: this cannot be known from this run.
  - Data rules: there is no ablation.
  - Decoding: the study had no power. Between 46 and 55 of its 60 pairs were judged "both bad"
    (`N/study/proxy/reports/ab/`).
- The draft omits the only blind comparison on record. It was made on `dev`, with `final` at
  step 25,280 (0.44 of training) and one reading per pair:

  | opponent | W / L / T | score [95% CI] | sign test p | "both bad" |
  |---|---|---|---|---|
  | Falcon-H1-Tiny | 2 / 43 / 15 | 0.158 [0.087, 0.271] | < 0.0001 | 14 of 60 |
  | MobileLLM-R1.5 | 15 / 8 / 37 | 0.558 [0.433, 0.677] | 0.21 | 37 of 60 |

  (`N/study/proxy/reports/ab/s1-default-vs-*--opus-1.md`; `R/docs/CHANGES.md` §19.8.) The paper
  still reports instead an informal, unblinded reading of 7 prompts (§12.2).
- The held-out lead of `v9`, "0.016 nats ahead at equal training time" (CHANGES §19.8, FACTS_2 §2,
  F8), does not survive its own uncertainty.
  - The baseline is a straight line through three `final` checkpoints spanning 44 minutes, which
    leaves one residual degree of freedom.
  - The 95% prediction interval of that line at 592.5 minutes is about ±0.026 nats (my
    arithmetic: `stats.py`, t with 1 degree of freedom), wider than the 0.016 gap.
  - Checkpoint-to-checkpoint jitter is of the same order. `v9` at steps 31,762 and 31,912, only
    150 steps apart, differ by −0.0085 on the chat rows and +0.0082 on the proxy rows
    (`N/midrun_eval_1.json`).
  - The web rows are the cleanest domain: −0.0224 in the 53.8 minutes after the graft against
    −0.0066 in the 44 minutes before. But this comparison is matched in time, not in tokens, and
    the data stream changed at the same moment.
- By the paper's own positioning, each contribution applies a known method:
  - the accounting convention: `R/docs/MATH.md` rates the novelty of the design mathematics 2–3
    out of 10;
  - the kernels: "These apply known principles; they are not new algorithms" (`related_work.md`);
  - the graft: "an instance of the growth operators of staged training [47]";
  - the reading rule (a win only when both orders agree): the rule of [78].
- With no result and no general, tested insight, the draft is a well-kept laboratory notebook,
  not a paper.

**3. The draft does not yet meet the standard it names as its own distinguishing quality: every
number traceable and every deviation declared. Must fix.**
- The graft check at step 29,722 was claimed as done with no record behind it.
  - `related_work.md` states "loss difference 0.0 on a synthetic batch at steps 26,890, 28,501
    and 29,722", and `R/docs/CHANGES.md` §19.7 claims it as well.
  - The only saved output of a graft at 29,722 is `S/graft_out.txt`. It ran without `--check`,
    into `main16_v9_sorted`, under configuration hash `f1243cb29610103a`, not the live
    `f29bfeb3ebf353a8`.
  - `R/scripts/graft_run.py` sets the manifest's `graft` record (l.275) and writes the manifest
    (l.281) before the check runs (l.287–291). The check's result is printed, never stored.
  - I have now re-run the check; see item 21.
- The gradient differences quoted for the adopted kernel belong to another configuration
  (items 13 and 22).
- Mid-run samples touched 11 test prompts, not 7 (item 14).
- The blind readers are not the judge that was pre-registered, and the paper does not list this
  as a deviation (item 9).
- The headline numbers are stale against committed sources: CHANGES §19.7–19.8 and commits
  `769f240`, `1b9d1b8` and `8a86bcb`. Affected: the time ratio, the `flat` switch, the "two
  layouts … not implemented", the held-out losses and the decoding study (both still marked
  [TBD]), the test count, the commit list and the "uncommitted files".

## (2) Claims that are unsupported, overstated, ambiguous or contradicted by a source

4. **Title, §1 thesis, §17 conclusion. Must fix.** "Bytes moved … do [predict]": see item 1.
   - Wording the evidence supports: "In our implementation on one TPU v6e, two configurations that
     differ by 3.1% in numbers read and 3.6% in analytic FLOP per token differed by 1.62× in step
     time at the switch, and by 1.46× after one hour of layout work. Layout changes that leave the
     arithmetic unchanged removed 27% of the difference."
   - The title "Parameters Read Are Not Bytes Moved" is a truism unless the paper adds a
     predictive result.
5. **"Without changing the function" / "what the model computes". Should fix.** The phrase
   appears in OUTLINE thesis and §17; §1 contribution 2 says "an exact path".
   - Source: `R/docs/CHANGES.md` §19.1 calls the bfloat16 storage of the feed-forward
     pre-activation "the only change that is not identical up to floating-point rounding" (2⁻⁹
     relative).
   - On the chip the adopted kernel changes gradients by up to 7.9e-2, relative, for the worst
     leaf (`W/chip_kernels_0745.txt`).
   - Write instead: "equal to the reference up to rounding, with one deliberate extra bfloat16
     rounding". Reserve "exact" for bit-identical results.
6. **§7.8 "the cost is layout, not arithmetic". Must fix.** Only 27% of the extra cost was
   *shown* to be layout, by removing it (item 1). The rest is the cost model's attribution, and
   that model was 8× wrong on `tokn`, the change aimed at those very contractions. State both
   parts.
7. **Graft equality "in floating point and not only in exact arithmetic" (§5.5), and "equal
   shared gradients" (OUTLINE; §8.2 item 2). Should fix.**
   - The forward equality was tested on the CPU in float32 with dense attention: the check sets
     `model.attn_impl=xla`, `act_dtype=float32`, `remat=false` (`R/scripts/graft_run.py` l.57).
   - The gradient equality is derived, not measured.
   - On the chip the grafted program is a different compiled graph. By §7.5's own evidence that
     moves the bfloat16 rounding points.
   - State the scope of the check, and measure the gradients (item 40).
8. **Abstract: "identical losses … on a test batch". Should fix.** The batch is 2 × 256
   synthetic tokens, and "test" collides with the test split.
9. **Contribution 4: "a protocol that cannot be tuned after the fact". Must fix.** Four defects:
   - (a) The pre-registered judge is `qwen3-4b-2507`, with `gemma4-e4b` from another family as a
     second judge (`R/docs/EVAL_PROTOCOL.md` judge table, l.225–226). The run instead uses Opus 5.5
     agents for the dev study and for the planned official score
     (`N/OPERATIONS.md`, "Endgame (v9)"). The agents that built the system are of the same family,
     which is the self-preference risk the protocol's second judge was meant to check. §11 mentions
     the change but §14.13 does not list it as a deviation.
   - (b) Eleven test prompts were seen (item 14), and two of them changed the training data.
   - (c) Every checkpoint comparison uses a held-out "proxy" domain built from every user turn of
     all 400 prompts, test half included. Source: `R/bes/export.py` `proxy_samples` calls
     `prompt_set.load()`; `N/midrun_eval_1.json` shows proxy rows: 12, 11,094 tokens.
   - (d) The protocol credits MobileLLM with the better of its two arms on `dev`. Only the
     `recommended` arm has been compared on `dev`, although replies of the `neutral` arm exist
     (`N/opponents_local.log`).
   - Repair (items 32–34) or soften the claim.
10. **§11: "written … before any of our replies existed … before any run of ours had trained".
    Should fix.**
    - The micro-model runs of E2, E4 and E5 trained before 23 Nov.
    - The branch history starts at `5ff33c5`, its root commit (25 commits in all). So the
      timestamp proves only that the protocol existed by 23 Nov 00:35 and has not changed since.
      I verified that `git diff 5ff33c5 HEAD -- docs/EVAL_PROTOCOL.md` is empty.
    - Write instead: "before any TPU run and before any reply of the full-scale model".
11. **§9.2 chronology. Should fix.** The scan (`N/deflection_check2.json`,
    `S/deflection_out.txt`) was made at 08:31 on 24 Nov. The refusal rule went live six hours
    earlier, at 02:31 (step 6,647). As written, the paragraph reads as if the scan motivated the
    rule. Date both.
12. **§9.3–9.4: the deflection rule's precision is unmeasured, and the rule removes more than
    deflections. Should fix.** The dropped samples stored in `N/deflection_check2.json` include:
    - clarification requests: "I don't have access to the entire code you are referring to. Can
      you please provide me with the code…";
    - hedged but helpful replies: "I do not have access to location-specific information.
      However, here are some general tips…";
    - a reply to distress: "I'm really sorry that you're feeling this way, but I'm unable to
      provide the help that you need. It's really important to talk things over with someone…".

    The consequences to state next to the existing one ("the model is not trained to refuse"):
    the model may learn to answer without the information instead of asking for it, and it loses
    its safe completions for distress. Measure the rule's precision (item 42).
13. **§7.1 and §7.5: the "noise-floor" argument for the adopted kernel. Must fix.**
    - The paper says the gradient differences of adopted changes are "no larger than those of a
      change of the compiled graph that leaves the arithmetic alone".
    - For the adopted configuration, `onetile:save+relu2`, the differences are 2.8e-2 (maximum) /
      2.6e-2 (L2) at 1,024 tokens and 7.9e-2 / 5.7e-2 at 2,048 (`W/chip_kernels_0745.txt`).
    - The graph-only examples the paper cites are 1.1e-2 to 1.4e-2 at 1,024 and 3.9e-2 to 4.7e-2
      at 2,048 (`blocks:save`, `+relu2+ce`). The adopted kernel is about twice the cited band.
    - The kernel was accepted on the CPU interpret-mode tests and the live loss band, not on the
      chip comparison. Say so.
    - A better floor: on the chip, measure each bfloat16 variant's distance to a float32
      reference, old kernel against new, at the next pause if one is taken.
14. **§11 "One deviation", §14.6 and R12: the seven test prompts and the 193 untouched ones.
    Must fix.**
    - The step-1,262 sample (`N/midrun_sample_step1262.md`) and the six-particle step-5,075 sample
      (`N/midrun_sample_step5075_p6.md`) also decoded `math-001`, `code-002`, `writing-001` and
      `instruct-001`.
    - All four are in the test split. I checked with `bes.eval.proxy.prompts.load()`, which
      also confirms `math-011` is `dev`.
    - So 11 test prompts were seen, and results must also be reported on the **189** test prompts
      no sample touched.
    - `N/OPERATIONS.md` lists "factual-008, reason-005". Neither is in the prompt set.
15. **§7.1: "On two shapes four times apart the estimated cycles matched the chip at one rate".
    Should fix.**
    - This is a two-point calibration on one family of programs (the MLP-block study,
      `W/speed_mlp_block.md` l.86–91).
    - The draft then applies it to v9, a different program, where the cost model missed the
      measured differences by factors from 0.37 to 8 (item 1).
    - Also note: the compile-only work used jax 0.11.1 (`W/v9_speed.md`, Method), while the chip
      ran jax 0.11.2.
16. **Table 13: GP attention "costs 4.5–9 times a softmax head on the TPU". Should fix.**
    `R/docs/NEGATIVE_RESULTS.md` (introduction and §1) takes its numbers from the adversarial
    review, real-text experiments and a cost model. Nothing in it was measured on the TPU. Write
    "estimated".
17. **§4.4: "entity look-ups stayed high (common-name probes 81.3% at 99.9%)". Should fix.**
    `R/docs/E1_COVERAGE.md` shows:

    | set | before (coverage / precision) | after |
    |---|---|---|
    | common-name probes (new probe definition for both) | 24.3 / 35.6 | 81.3 / 99.9 |
    | head-entity probes | 82.3 / 98.7 | 74.4 / 99.1 |
    | NQ-open val without a near-duplicate stored question | 18.5 / 37.3 | 1.0 / 43.5 |

    The common-name probes rose rather than "stayed"; head-entity coverage fell. The last row
    shows the library answers mainly through stored near-duplicates. Say so; it strengthens the
    case for dropping the library.
18. **§4.4 omits the project's own unfavourable prediction. Should fix.**
    - `R/docs/V9.md` §7: "The calibrated scaling law of this project puts the final configuration
      at 2.507 nats per token and Falcon-H1-Tiny-90M at 2.358", and "v9 therefore narrows the
      predicted gap to the 90M opponent and does not close it".
    - Report it. Note also that loss per token across different tokenizers is not comparable
      without converting to bits per byte (item 31).
19. **§8.3: the loader's remaining-horizon estimate. Optional.**
    - `S/runlog_peek.txt` l.875 shows the loader planned over 7.45e9 remaining tokens, that is
      tokens/p − tokens at the average rate before the graft.
    - At v9's rates about 5.3–5.8e9 tokens remained after the graft (`projections.py`; item 28).
      The estimate is no longer 40× too small, but it is about 1.3–1.4× too large rather than
      "right".
20. **Smaller overstatements. Optional.**
    - Table 13 and the OUTLINE speak of "13 measured non-improvements". Two of the 13 were faster: the
      single loss chunk by 1.2% (below the bar) and the hand-written cross-entropy backward by
      0.6% at 2,048 (within the spread). Write "13 changes not adopted".
    - The abstract says 62% was "data movement outside matrix products". The source counts the
      62% outside both the matrix products and the attention kernel (`R/docs/TPU_NOTES.md`
      l.219), and its "glue" includes element-wise work as well as movement.
    - §7.2 gives "a single 768 × 768 product took 0.23 ms". That is forward plus rematerialised
      forward plus backward; the forward alone took 0.08 ms (TPU_NOTES l.216).
    - §10.2 gives log K − (K−1)/K as a bound for any selection. It is the KL of best-of-K under a
      continuous score, and an upper bound for best-of-n [69]. State the class of selection rules
      for which `MATH.md` F4 proves it, or use the general bound, log K = 1.79 nats.
    - §12.2 says Falcon's replies are "on task". Its phone-in-water reply tells the user to
      unscrew the phone with a Phillips screwdriver (`N/midrun_sample_step22613.md`).

## (3) Numbers that do not match their source

21. **Table 11, step 29,722 ([TBD]), and related_work.md. Must fix. Now resolved by a re-run.**
    - I re-ran the check on CPU cores 2 and 3, from the local snapshot of the stopped run
      (`N/snap_final_main16`, checkpoint `ckpt-000026-step-000029722`), with the configuration of
      the live graft:
      `scripts/graft_run.py SRC DST --config configs/v9.py --override run.seed=1 --check --cpu`,
      at HEAD `8a86bcb`.
    - Result: configuration hash `f29bfeb3ebf353a8`, the live one. Source loss 11.3726320,
      grafted loss 11.3726320, difference 0.0. 59 leaves (34,531,720 numbers) carried, 5 new
      (63,504,384).
    - Log: `N/paper/review1_work/graftcheck/graft_check_29722.log`. The large outputs were
      deleted.
    - Since `9a4d652`, `graft_run.py`, `bes/config.py` and the configs are unchanged. `model.py`
      and `layers.py` changed only by switches that are off by default; CHANGES §19.7 states the
      default path is bit-identical, which I did not re-verify.
    - Cite it as "a re-run in the development container of the same source checkpoint under the
      same configuration hash". It is not the live invocation, whose output was not saved.
22. **§7.5 gradient numbers. Must fix.** The paper quotes 1.4e-2 / 1.5e-2 and 7.5e-2 / 5.4e-2.
    Those are the numbers of `onetile:save`. The adopted `onetile:save+relu2` has 2.8e-2 / 2.6e-2
    and 7.9e-2 / 5.7e-2 (`W/chip_kernels_0745.txt`). This also contradicts the caption of F3,
    which has the right numbers. The error comes from CHANGES §19.6.
23. **Held-out rows. Must fix where they are cited.**
    - FACTS_2 §2, CHANGES §19.8 and the F8 caption say "proxy 16" rows and "144 rows".
    - `N/midrun_eval_1.json` gives proxy **12** rows; in all 64 + 64 + 12 = **140** rows and
      136,974 weighted tokens. The "16" is the export check's default `rows=16`; every user turn of
      the prompt set fills only 12 rows of 1,024.
24. **§7.8 and §14.10: the cost model was "5% high on v9's step". Should fix.**
    - The prediction is `final`'s *measured* 884.5 ms plus the modelled difference
      (`W/v9_speed.md`, Method).
    - Judged on what it actually predicts, the difference, it was +623.7 against +548.6 ms:
      **14% high**.
25. **§7.7 FLOP "counting what the chip executes". Should fix.**
    - The paper's 1.132 GFLOP per token assumes attention over whole tiles and a forward kernel
      run twice.
    - At 281,374 tokens/s the adopted kernel covers 62.5% of the score square, runs its forward
      kernel once (`save`), and the rotation computes the q and k projections as two products
      each. The two products add 0.153 GFLOP per token by themselves.
    - With the conventions of `lead_work/scripts/flops.py`, my recount is 1.18–1.19 GFLOP per
      token: about 332–336 TFLOP/s, 41% of 814, instead of 318 TFLOP/s and 39%
      (`N/paper/review1_work/flops_check.py`; the range depends on the backward multiplier of
      attention, 2.0 or 2.5).
    - Recount with your script and define "executed".
26. **Table 10 live row and the +17.5%. Should fix.**
    - Both live rates in Table 10 are single log lines; 178,523 is v9's first reading after the
      restart.
    - Medians over the train logs give 281,399 for `final` (steps 25,300–29,700, n = 89) and
      178,087 for `v9` (n = 44). The ratio is **1.580**, not 1.570 (`steps.py`, `win.py`).
    - In the same way, the kernel's live gain is +17.2% on medians: 240,118 (n = 33) to 281,387
      (n = 53), which matches `tables.md` T2. Use medians with n everywhere.
27. **Stale against committed sources. Must fix.**
    - The headline ratio "1.54–1.62" is now 1.455 synthetic and 1.43 live (item 1).
    - §7.8 says the `flat` switch is [TBD]. It measured 1,433.1 → 1,406.4 ms: −26.7 ms against
      −46 predicted.
    - §7.8 says two layouts are "not implemented". They now are, as `rows` and `tokn`
      (−111.2 and −8.1 ms).
    - §8.5 gives four loss readings after the graft. There are 48 to step 32,100: mean 2.4776,
      sd 0.048.
    - §7.1 counts "four pauses". There were five; the fifth lasted 320 s at 10:52.
    - §15 lists 1,349 tests. At `8a86bcb`, `grep -c "def test_"` counts 1,356.
    - §15 says "describes `9a4d652`", with the `flat` and graft tests uncommitted. Both are in
      `769f240`.
    - §6.4 and §14.4 mark the held-out results [TBD]. They exist (CHANGES §19.8).
    - §10.3 and R11 mark the decoding study [TBD]. It exists (FACTS_2 §3).
28. **Token projections (§2.3, §8.5, Table 13). Should fix.** The draft projects 1.31e10 tokens
    in all, and 2.8e9 fewer than `final`.
    - Inputs for an update: v9 at 196,864 tokens/s, and the slice of 27,420 s at the 10:57
      restart (p = 0.5651).
    - Under the assumption that the switches gain as much at 2,048 tokens as at 1,024 (not
      measured), my arithmetic gives about 1.36e10 in all and a deficit of about 2.3e9 (15%)
      (`stats.py`).
    - Recompute with `projections.py`, and state the assumption.
29. **Inconsistent definitions of one event. Optional.**
    - Graft time: 09:45:35 in the paper; 09:45:17–09:58:26 in CHANGES §19.7; 09:45:34–09:58:45 in
      the train log and T5b. The manifest's `graft.time` is 09:47:08.
    - p across the graft: 0.5049 → 0.5112 in the paper (end and stage events) against
      0.5045 → 0.5118 in F1 (step records).
    - The p jump at step 12,062: 0.2415 → 0.2190 in the paper against 0.2413 → 0.2196 in F1.
    - The pause lengths: 6.7, 10.8, 9.4 and 13.2 minutes in the paper (the operator's log,
      overheads included) against 5:54, 9:25, 9:21 and 13:10 in T5b (train-log stop to restore).
    - Pick one definition per kind of event and state it.
30. **Smaller mismatches. Optional.**
    - §2.2 writes "31.24 GB". The value is GiB: `R/bes/train/supervisor.py` l.398 divides
      `bytes_limit` by 2³⁰.
    - The first window of Table 5 has sd 0.507. That is the population sd; the sample sd is 0.535
      (`win.py`). State the convention.
    - §9.1 gives retrieval 5.3% and maths 3.1%. Plain means over the 594 logged blocks give 5.24%
      and 3.04% (`mix.py`). State how the shares were averaged.
    - Table 4 gives "240–246k" live at step 9,332. Live was 240.0k; 246.5k is the synthetic rate.
    - The header of `N/midrun_sample_step22613.md` says 6.02e9 tokens. 22,613 × 262,144 is
      5.93e9, which the paper rightly uses; do not copy the header.

## (4) Missing baselines, controls and analyses that can still be obtained

Each can be obtained before about 20:30 UTC from files on disk, or on CPU, without chip time
unless the item says otherwise.

31. **A judge-free comparison with the opponents. Should fix; the strongest addition
    available.**
    - Measure bits per UTF-8 byte on the same held-out text (the 64 web rows and the 64 chat rows
      of `midrun_eval`).
    - For the opponents, run llama.cpp on CPU on `N/models/Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf`
      and `N/models/mobilellm-r1.5-140m-q8_0.qwen3.gguf`. For ours, use the `final` and `v9`
      checkpoints.
    - This is the only comparison with the opponents that needs no judge and no decoding. State
      the possible contamination: the opponents' training data are unknown.
32. **Statistical power at the observed tie rates. Must fix.**
    - The dev reports' own power lines (`…--opus-1.md`, last lines) say that a 10-point
      difference needs 259 comparisons at the Falcon tie rate (25%) and 506 at the MobileLLM tie
      rate (62%).
    - By the usual 1/n scaling (my arithmetic), n = 200 then resolves only about 11 and 16
      points, not the "±7 points" of §11 and §14.7.
    - Two readings with strict agreement will raise the tie rate further. Fix the numbers in §11
      and §14 now, before the test split is read.
33. **Readers. Must fix if the Opus agents remain the readers.**
    - Run the pre-registered judge (`qwen3-4b-2507`, plus `gemma4-e4b` from another family) on
      the dev pair sets already built (`N/study/proxy/pairs/s1-default-vs-falcon`,
      `…-vs-mobilellm`), in both orders.
    - Report agreement with the Opus readings (κ, and how often the two orders agree) and the
      first-position preference.
    - Also re-read the dev pairs with sides swapped (Opus). This measures position bias and how
      many decisive verdicts survive strict agreement.
    - Both must be done before the test split is opened.
34. **Protocol repairs on dev. Must fix.**
    - Compare MobileLLM's `neutral` arm on `dev` and pick the better arm, as the protocol
      requires; its replies exist.
    - Report our exact scores on dev now: maths answers, constraints and degenerate rates of the
      60 dev replies in `N/study/proxy/outputs/ours-export-step25280-*`, next to both opponents'.
    - Drop the test-split user turns from the "proxy" held-out domain, or declare them.
35. **Held-out comparison across the graft, with uncertainty. Must fix if the 0.016 claim
    stays.**
    - Save per-row losses, and bootstrap the paired differences over rows.
    - Fit `final`'s trend on more points. Local checkpoints: 5,075 (`N/snap_keep_step5075`),
      22,613 (`N/snap_view`), 26,890 (`N/snap_look`), 29,722 (`N/final_look`). For v9:
      `N/snap_main`, `N/snap_main.prev`, and the runtime loop's `heldout_v9.jsonl`.
    - Report a prediction interval, not a 4-digit gap.
36. **Pages-off decomposition. Should fix.**
    - Evaluate the v9 checkpoints on the same rows with the token-neuron and gain paths switched
      off (the run-time switches of §5.5). This gives a "page gain" comparable to the tape gain.
    - Then stratify the post-graft held-out improvement by the frequency of the input token: the
      pages are indexed by the input token.
    - Neither identifies the counterfactual, but both test whether the gain is carried by the
      pages.
37. **The thesis's positive half as a prediction. Must fix if the title stays.**
    - For every configuration timed on the chip, tabulate analytic FLOP, reads per token,
      compiler-estimated HBM bytes (`hlo_bytes.py`) and estimated cycles. Compile them
      compile-only with libtpu 0.0.50 on CPU; about 8 minutes per compile on this host
      (`W/v9_speed.md`).
    - Then show which of these predicts the measured time, with leave-one-out errors.
    - Configurations that change only XLA code come first: the rows of Table 6, the pieces of
      Table 7, `take` / `sorted` / `onehot` / 8-row for v9, and `flat` / `rows` / `tokn`. That is
      about 15 compiles, roughly 2 hours on cores 2–3.
    - Report whatever the result is.
38. **Cost-model calibration table from existing files. Should fix.** For every change with both
    a prediction and a chip value, tabulate predicted against measured: v9 − final, `sorted`,
    8-row pieces for both configurations, `flat`, `rows` and `tokn`. Measured on the chip with no
    compile-only prediction: `remat_keep`. Predicted only: the carry, at +4.9%. This replaces the
    single-number "5% high" claim.
39. **Throughput decomposition (a waterfall figure). Should fix.**

    | stage | tokens/s |
    |---|---|
    | reference (32-row micro-batches) | 113k synthetic; 112–117k live |
    | rewrites | 202k synthetic |
    | 16-row micro-batches | 223.4k synthetic, 221.9k live |
    | runtime 3, same program | 216.7k live (−2.3%, unexplained) |
    | pieces | 240.0k live |
    | one-tile kernel | 281.4k live |

    Label each value synthetic or live (`tables.md` T2 has the values). The 2.4–2.5× includes a
    change of configuration (32 → 16 rows); say so.
40. **Gradient equality at the graft. Should fix.**
    - On CPU in float32, with a real held-out batch from the 29,722 checkpoint, compute ∇θ_s under
      both configurations and report max |Δ|.
    - Also show that ∂L/∂K, ∂L/∂P_in and ∂L/∂P_out are exactly 0 and that ∂L/∂U and ∂L/∂G are not.
    - Rebuilding the graft takes about 12 s (item 21's log).
41. **The thesis applied to its own design: the loop count. Should fix.**
    - The read ledger counts the 16 block applications per token once. The clock charges all 16:
      one loop took 42.9 ms against 289 ms for eight (TPU_NOTES l.202), so about 35 ms per loop.
    - At equal tokens, E2 found 4 loops slightly better than 8 at half the compute: −0.12%
      (`R/docs/E2_ABLATIONS.md`).
    - From the block and non-block times measured before and after the rewrite (TPU_NOTES l.202
      and l.247–249), 4 loops would see about 1.6–1.9× the tokens. With that and the iso-depth law
      of [21] (φ = 0.65 with two lanes), predict 4 loops against 8 loops at equal wall clock.
      Label it a prediction.
    - This is the largest gap between the ledger and the clock in the model, and the paper never
      mentions it.
42. **Precision of the deflection rule. Should fix.** Hand-label the stored dropped samples (up to
    30 per source) and kept-near samples (up to 40 per source) in `N/deflection_check2.json`.
    Report the precision and recall of the rule.
43. **The 2,048-token rate of v9 with the switches. Should fix.** It has not been measured; only
    1,024 was timed (`S/v9_measD.txt`). It needs no pause: read it from the live log after the
    phase change at about 12:47 UTC, and update the projections (item 28).
44. **The v9 decoding study (s2, started 11:10). Optional.** Report it, and state that s1 tested
    `final` weights.

## (5) What one run cannot show, and how to word it

45. **Must fix throughout.** None of the following can be concluded. Write "in this run", "we
    observed", "consistent with"; never "the cost of lookup parameters is", "the graft keeps what
    was learnt", "the rule removed the failure".
    - That token pages improve the model at this scale. The held-out change after the graft is
      confounded with the new data stream and seed, the missing library rows, the epoch caps (the
      share of `gsm8k` fell from 0.0126 to 0.0007, `S/runlog_peek.txt` l.876) and the phase of
      the learning rate. And the pages train only from a learning-rate factor of 0.5394 (step
      29,750, train log) down to 0.1: they see only the low-rate half of the schedule (CHANGES
      §19.7).
    - That the graft was better or worse than restarting `v9` or continuing `final`.
    - That 1.43–1.62× is the price of per-token tables, rather than of this implementation, this
      compiler or this chip.
    - That the refusal and deflection rules changed the model's behaviour (no ablation; one
      anecdote each).
    - That the throughput changes left the training trajectory unchanged. The evidence is the
      loss band and rounding-level tests.
    - That the final standing against the opponents is due to the design rather than to the token
      gap: about 1.36e10 tokens against 8e11 for Falcon-H1-Tiny and 4.2e12 for MobileLLM-R1
      pre-training (refs [11], [7]). It also cannot show that the standing holds for another judge
      family.
    - Run-to-run variation at full scale. The only estimate is 0.42% at micro scale.
    - That the default decoding settings are good (the study had no power).
    - That the time-tied schedule p is better than alternatives (none was run).
    - Compliance with the strict reading beyond the 40 documents measured.

## (6) Structure and writing

46. **Length and focus. Must fix.**
    - The main text runs to about 11,000 words, with 13 tables. A top venue allows about 9 pages.
      Keep in the main text: introduction; setting; ledgers (short); model (one paragraph each
      for the looped core, the pages and the output law); where the time goes; the graft;
      evaluation and results; limitations.
    - Move to appendices:
      - §2.3 (opponent mechanics), §2.4 with Table 1, and §2.5 (operations);
      - §4.4 (the evidence for allocation before training) and Table 3;
      - §5.3 and §5.4 in full, and §6.2;
      - Table 5 and the tape-gain paragraph;
      - §9, down to one paragraph;
      - §10, which rests on a micro model and a stand-in, not on this run;
      - Table 13, §15 and §16.
    - Move the operations narrative (blocked approvals, the 4 h 50 min) to an "Operations"
      appendix. It is honest, but it is not evidence for any claim.
47. **Where the argument breaks. Must fix.**
    - §1 promises a predictor; §7 gives none (item 37).
    - §4.4 presents evidence for pages, §8.1 says the switch was the user's decision, and §1
      reads as a design argument ("We therefore grafted"). Decide which story the paper tells and
      tell it once.
    - §12.1 "Results" restates §4, §7 and §8. Replace it with one results table that includes
      the held-out comparison and the dev blind comparison, with their uncertainties.
    - Contribution 4 mixes a protocol with data hygiene and a standing. Split it, or demote it
      until items 32–34 are done.
48. **Figures. Should fix.**
    - F1: the v9 smoothed line starts below its first records. It is a one-sided kernel average,
      as `FIGURES.md` itself says, and it visually invites the very cross-graft comparison the
      text forbids. Use a causal window on both sides, or draw only points for the first hour.
    - F8: add an uncertainty band and more `final` points (item 35). Print "≈0.016", not "0.0158",
      next to an extrapolation.
    - Add the throughput waterfall (item 39) and the cost-model calibration plot (item 38).
    - F3, F4, F6 and F7 are clear. F6's "at 29,722 per the operator" can now cite the re-run.
49. **Words and units. Should fix.**
    - Define once "exact" (bit-identical) against "equal to rounding", and use them consistently:
      the "exact rewrites", the "exact path" and the "exact graft" (on the chip) are not exact.
    - Define "piece", "micro-batch" and "step" once in the main text. Only `tables.md` defines
      them now.
    - Use one unit system: GiB versus GB; "thousand tokens per second" versus "k tok/s"; "1.62
      times" versus "×1.62".
    - Shorten the abstract to about 200 words, with only numbers that survive this review.
    - Keep our own numbers out of related work; `related_work.md` restates several, among them
      the 27% loops and the 1.79×.
50. **One source of truth for numbers. Should fix.**
    - The ledger idea is good, but the mismatches above (items 21–30) arose because numbers were
      copied by hand from summaries (CHANGES §19.6–19.8, FACTS_2) rather than from the raw files.
    - Extend `figures/make_data.py` so that it also writes every number cited in the text
      (`numbers.json`), and have the text cite that file.

## (7) What would make this a best paper rather than an accepted one

Everything below can be done truthfully with the data, the chip schedule and the time that
remain.

51. **One sharp, tested systems claim instead of four loosely related contributions.**
    - Present the measurement study as a test of a predictive statement: across all timed
      configurations, padding-aware bytes predict step time to within X%, and FLOP and reads
      per token do not (item 37).
    - Present the v9 switch series as an intervention study: layout changed, arithmetic and
      reads fixed, 1,433 → 1,287 ms.
    - Apply the claim to the design itself through the loop count (item 41).
    - If the predictive test fails, report that. "No static count predicts the time of small
      contractions on this chip" is also a publishable finding.
52. **Turn the graft into a measured method.**
    - Show the function equality at the live source checkpoint (item 21, done) and the gradient
      equality on a real batch (item 40).
    - Include the Adam transient that §8.2 already derives.
    - Add the pages-off decomposition and an uncertainty band for the held-out change (items 35
      and 36).
    - Position it precisely against [47]: the kind of parameters added, a schedule tied to the
      wall clock, and a check made next to the live run.
53. **An honest, objective outcome.**
    - Report in order: exact scores, then bits per byte against both opponents on the same text
      (item 31), then the blind test.
    - Run the blind test on the 189 untouched test prompts, with the pre-registered judge next to
      the Opus readers and both orders. State the power at the observed tie rates.
    - A clear negative result is publishable and best-paper-worthy when it is quantified and
      explained, for example: "one TPU-day with this design is behind a 90M model trained on 60×
      more tokens by X [CI], and here is where the time went".
54. **Write it as a 9-page paper with appendices.** One figure per claim, and every number
    generated from raw logs (item 50).
55. **Optional.** The way the work was done (language-model agents, a claims ledger, adversarial
    reviews) is unusual, and it can be measured. Report how many claims the audits changed and
    what they caught. Examples: the gate's logit bug was found by an ablation, not by review; the
    loader's horizon bug was found by the graft review; this review found items 13, 14, 21, 22
    and 23.

## Verdict

As it stands: reject, and do not nominate. With items 1–3, 9, 13, 14, 21–23, 27, 32–35 and 45–47
fixed, it would be a solid accept as an audited systems case study. A best-paper case needs
items 37 and 53.
