# RESPONSE_1: what was done for each point of `REVIEW_1.md`

Second-pass lead author, 24 November 2026, 12:05–13:20 UTC. One line per point. "Text" = the
paper was changed; "computed" = new analysis from files on disk (scripts and outputs in
`N/paper/lead2_work/`, `L2/` below); "limit" = could not be done here and is stated in the paper.
Section, table and figure numbers are those of the new `paper.md`. Every new number has a `CLAIMS.md` id.

## (1) Reasons to reject

1. **Thesis as a law (must fix).** Text + computed. Title changed to "What a Parameter Budget Does Not
   Count"; the thesis is stated for this implementation on one v6e (§1, §9); ratios updated to 1.62×
   at the switch and 1.46× after the switches (live 1.58× → 1.43×, medians); "27% removed by layout" is
   given as the only measured attribution and the remaining 403 ms as unattributed (§5.3). The positive
   half is tested post hoc on the six configurations that already had both a compiled program and a
   chip time (Table 4, `L2/predictor_test.py`, no new compile): the compiler's byte count predicts the
   step with leave-one-out errors of mean 6%, largest 11%, cycles 12% (40%), FLOP 20% (66%), and FLOP
   and reads cannot order the five `v9` variants. Stated as exploratory, six points, five of one model.
   Limit: the ~15-configuration version needs about nine new compiles (460–500 s each on this host);
   not done.
2. **No outcome, no identified effect (must fix).** Text + computed. Both blind comparisons on dev
   (`final` before the graft, one reading; `v9` after it, two readings on exchanged sides) are now in the
   main text (Table 8), with exact scores on the same 60 prompts (`L2/exact_dev.py`), bits per byte
   against both opponents on held-out web text (`L2/opp_bpb.py`), and the held-out comparison across
   the graft with a prediction interval and a pages-off decomposition (§6.5). Contributions re-scoped:
   the measurement study first, the graft as a measured method, the audit, and the standing; novelty
   stated plainly (§1, §3). Limit: no effect of the pages or the graft is identified, and the paper says
   so (§1, §6.6).
3. **Traceability (must fix).** Text. The 29,722 check is cited as the reviewer's re-run with its hash
   and HEAD (Table 6), the tool's defect (record written before the check, check not stored) is stated;
   kernel gradient numbers corrected; eleven seen test prompts; the readers listed as a deviation;
   every headline number re-derived from raw files (`L2/make_numbers.py` → `numbers.json`).

## (2) Claims unsupported, overstated, ambiguous or contradicted

4. **Title, thesis, conclusion (must fix).** Text: the reviewer's wording adopted in §1 and §9; title
   changed.
5. **"Without changing the function" (should fix).** Text: "exact" (bit-identical) and "equal to
   rounding" defined once (§2.2) and used throughout; the rewrites are "equal to the reference up to
   rounding, with one deliberate extra bfloat16 rounding" (§5.2).
6. **"Layout, not arithmetic" (must fix).** Text: both parts stated (27% by removal; the rest the cost
   model's attribution, 8× wrong on `tokn`) (§5.3).
7. **Graft equality in floating point (should fix).** Text + computed: the scope of the check (CPU,
   float32, dense, no recomputation) is stated; the shared gradients were measured on real rows and are
   bit-identical on the CPU (point 40; §6.4); on the chip the equality holds to rounding and was not
   measured.
8. **"Test batch" (should fix).** Text: "a synthetic batch of 2 × 256 tokens" (§6.4); the abstract no
   longer uses "test".
9. **"A protocol that cannot be tuned after the fact" (must fix).** Text: the claim is removed; the
   four defects are deviations 1–4 of §7.2. (a) Limit: the pre-registered judges are not on disk and
   this role has no network access for model files → [TBD before the test split is opened]. (b) 189
   untouched prompts. (c) Proxy rows reported apart and left out of every comparison (§6.5). (d) The
   neutral arm's exact scores are computed (Table 8); its blind comparison is [TBD].
10. **"Before any run of ours had trained" (should fix).** Text: the commit proves that the protocol
    existed before every TPU run and every reply of the full-scale model, not before the micro-scale runs
    of gates E2, E4 and E5 (§7.1).
11. **Chronology of the rules (should fix).** Text: the refusal rule went live at 02:28:59 (step 6,647)
    from one sample; the rates were measured at 08:31 (Appendix D.2).
12. **Deflection rule removes more than deflections (should fix).** Text + computed: hand labels of 60
    dropped openings (Appendix D.4): 8 pure deflections, 45 disclaimer then help, 3 requests for missing
    input, 2 safety referrals, 1 direct answer, 1 correct "not in the text"; the consequences stated.
13. **Noise-floor argument (must fix).** Text: correct numbers; "about twice that band, so the band does
    not establish its correctness"; accepted on CPU interpret-mode tests and the live loss band; the
    parameters were fresh initialisations on a synthetic batch (§5.2). Limit: the on-chip float32
    reference needs a pause, which this role cannot take.
14. **Seven → eleven test prompts (must fix).** Text: the eleven listed and checked with
    `prompts.load()`; results on the 189 untouched prompts (R12); `factual-008` and `reason-005` are not
    in the prompt set (the operator's notes, not the paper, listed them).
15. **Two-point calibration; jax versions (should fix).** Text (§5.1, Appendix F.5).
16. **GP attention cost (should fix).** Text: "estimated (not measured on the chip)" (Table H1).
17. **Library coverage (should fix).** Text: the full before/after rows, including the near-duplicate
    row, and its reading (Appendix B.4).
18. **The project's unfavourable prediction (should fix).** Text: quoted in Appendix B.4, with the note
    that per-token losses across tokenizers are not comparable; bits per byte measured instead (§7.4).
19. **Loader horizon (optional).** Text: "about 1.3 times too long rather than right" (§6.3).
20. **Smaller overstatements (optional).** Text: "13 changes measured and not adopted" (two faster
    below the bar); 62% "outside both" incl. element-wise work; 0.08 vs 0.23 ms; selection bound
    log K = 1.79 in general, 0.96 only for best-of-K under a continuous score [69] (Appendix E.2); the
    informal "on task" reading is removed with the informal sample comparison it belonged to.

## (3) Numbers that did not match their source

21. **Step 29,722 (must fix).** Text: Table 6 row from `review1_work/graftcheck/graft_check_29722.log`,
    labelled a re-run of the same source checkpoint under the live hash.
22. **Kernel gradient numbers (must fix).** Text: 2.8e−2 / 2.6e−2 and 7.9e−2 / 5.7e−2 (§5.2, Table F2).
23. **Held-out rows (must fix).** Text: proxy 12 rows, 140 rows, 136,974 weighted tokens (§6.5, Figure 6
    caption). The F8 caption in `FIGURES.md` still says 16 / 144: flagged to the figures agent below.
24. **Cost model "5% high" (should fix).** Text: 14% high on the difference; Table 5 replaces the single
    number.
25. **FLOP "executed" (should fix).** Text + computed: "executed" defined; 1.18–1.19 GFLOP per token,
    332–336 TFLOP/s, 41% (§5.6; `L2/make_numbers.py` with the reviewer's conventions).
26. **Live rows and +17.5% (should fix).** Text + computed: medians with n everywhere (Table 2, Table 3):
    +17.2%; 1.580; definitions of live and achieved rate in §2.2.
27. **Stale numbers (must fix).** Text: 1.455 / 1.429; `flat` −26.7 ms measured; `rows` and `tokn`
    implemented and measured; 93 post-graft loss readings (mean 2.4675, sd 0.0399) to step 34,350;
    five pauses; 1,356 tests; `769f240` holds the `flat` switch and the graft tests; held-out and
    decoding results reported.
28. **Token projections (should fix).** Computed (`L2/make_numbers.py`), at the measured live rates at
    both lengths (point 43): about 1.36 × 10¹⁰ in all, about 2.3 × 10⁹ (14%) fewer than `final`, whose
    own 2,048-token rate remains a scaled estimate (§6.3).
29. **One definition per event (optional).** Text: events are the train log's stop and restore records;
    p from the stop and restore events (step-record values given once, Appendix C.1); pauses from stop
    to restore record (§5.1, Table C2).
30. **Smaller mismatches (optional).** Text: GiB; sample sd (0.535) stated; plain means over 594 blocks
    (5.24%, 3.04%); Table C2 gives events only and Table 2 the rates (live 240,131, synthetic 246,538);
    5.93 × 10⁹ is used, not the sample file's header.

## (4) Missing baselines, controls and analyses

31. **Judge-free comparison (should fix).** Computed: bits per byte on the 64 held-out web rows for
    ours and both opponents through llama.cpp on the same Q8_0 files: ours 1.017 (`final` 29,722) and
    0.998 (`v9` 35,294) bits per byte, Falcon-H1-Tiny 1.570, MobileLLM-R1.5 1.500, on the same 258,814
    bytes (Table 8, §7.4), with the caveats (our training distribution; opponents' data unknown). Limit: the chat rows need
    each model's chat template and byte alignment of the assistant turns; not done → R11 [TBD].
32. **Power at the observed tie rates (must fix).** Computed + text (§7.3): at the two-reading tie
    rates of the `v9` dev comparison (30%, 55%), n = 200 detects decisive-rate differences of 0.117 and
    0.146 (0.121 and 0.150 on the 189 untouched prompts); at the one-reading rates of `final` (25%, 62%),
    0.113 and 0.158.
33. **Readers (must fix if Opus remains).** Partly done by the evaluation agents, partly a limit. The
    `v9` development comparison (step 31,912) was read twice on exchanged sides: the orders agreed on
    93.3% (Falcon-H1-Tiny) and 98.3% (MobileLLM-R1.5) of pairs, the mean preference for the first-shown
    reply was +0.000 and +0.008, and the two-reading score had 42 decisive verdicts against
    Falcon-H1-Tiny where a single reading of the same pairs had 44 (Table 8). Limit: the pre-registered
    judges are not on disk and this role may not download model files → [TBD before the test split is
    opened] in §7.2.
34. **Protocol repairs on dev (must fix).** Computed: exact scores of ours, Falcon-H1-Tiny and both
    MobileLLM arms on the 60 dev prompts (Table 8). Text: the proxy domain is declared and excluded.
    Limit: the neutral arm's blind comparison is [TBD].
35. **Held-out across the graft with uncertainty (must fix).** Computed (`L2/heldout_eval.py`,
    `L2/heldout_analysis.py`): per-position losses on CPU for `final` 29,722 and `v9` 34,166 and 35,294;
    paired bootstrap over rows: −0.050 [−0.054, −0.046] nats since the graft at 34,166 and −0.061
    [−0.065, −0.056] at 35,294 on web + chat; against `final`'s trend through its three checkpoints,
    −0.03 nats at equal training time with a 95% prediction interval of ±0.05 to ±0.07: not resolved
    (§6.5, Table 7). The local values reproduce the runtime's to 10⁻⁸, so the runtime's values for
    26,890 and 28,501 are used in the trend. Limit: the local copies of the 5,075, 22,613 and 26,890
    checkpoints were deleted at 12:10–12:14 to free the disk before they could be evaluated, so the
    trend keeps one residual degree of freedom.
36. **Pages-off decomposition (should fix).** Computed: pages, token neurons and gains switched off at
    evaluation at steps 34,166 and 35,294, and a stratification by input-token frequency (§6.5).
    Switched off, the pages cost 0.047 [0.044, 0.051] nats at 34,166 and 0.064 [0.059, 0.070] at 35,294,
    about all of the improvement since the graft (with them off the network is within 0.004 nats of the
    graft point); token neurons alone 0.019, gains alone 0.024; the cost is 0.066 nats on the rarest
    quarter of input tokens and 0.032 on the most frequent, while the improvement since the graft is the
    same in every quarter. Stated as a decomposition of one trained model, not a counterfactual.
37. **The positive half as a prediction (must fix if the title stays).** The title no longer states it;
    computed on the six existing configurations (point 1). Limit: no new compiles.
38. **Cost-model calibration table (should fix).** Computed: Table 5, seven changes; signs right 7 of 7;
    sizes off by 0.36–7.9×; the two one-sided cases stated.
39. **Throughput decomposition (should fix).** Text: Table 2, each value labelled live, synthetic or
    benchmark, with the 32 → 16 rows change named. A waterfall figure is requested from the figures
    agent (below).
40. **Gradient equality at the graft (should fix).** Computed (`L2/graft_grad.py`): on 2 web and 2 chat
    held-out rows of 1,024 tokens with the step-29,722 checkpoint, CPU float32: equal losses to the last
    bit, the gradients of all 59 shared leaves bit-identical, the gradients of K, P_in and P_out exactly
    zero, those of U and G non-zero (§6.4; `L2/graft_grad.json`).
41. **The loop count (should fix).** Text + computed (§5.5): 16 applications counted once and charged
    16 times; 4 loops would see 1.6–1.9× the tokens at equal wall clock; the micro result at equal
    tokens; the 2^0.65 factor. Limit: no prediction with the law's fitted constants (not used; the law
    was fitted with a prelude and coda this core lacks); not tested.
42. **Precision of the deflection rule (should fix).** Computed: point 12; recall cannot be estimated
    from the stored near misses, which are not a random sample (Appendix D.4).
43. **The 2,048-token rate of `v9` (should fix).** Computed from the train log of the 13:08 snapshot
    (read without extracting it): 188,980 tokens/s live (median, n = 16, steps 36,900–37,650 from
    12:48:36), 4.0% below the rate at 1,024; Table 2 and §6.3 updated.
44. **The `v9` decoding study (optional).** Text: reported (3-gram blocking against the default on
    `v9`: 4 / 8 / 48, p = 0.39; Table E1), together with the blind comparison of `v9` against both
    opponents (Table 8); s1 is stated to have tested `final` weights (Appendix E.4).

## (5) What one run cannot show

45. **Wording throughout (must fix).** Text: "in this run", "we observed", "consistent with"; the list
    of what is not claimed (§1), §6.6 and §8 cover every bullet, including the pages training only in
    the low-rate half (0.54 → 0.1), the GSM8K cap (before and after the graft alike) and the token gap
    (59× and about 310×).

## (6) Structure and writing

46. **Length and focus (must fix).** Text: the opponents' mechanics, runtimes, operations, the
    pre-training evidence, the micro-ablation table, the training-loss table, data hygiene, decoding,
    negative results, reproducibility and licences are appendices (B–K). The main text is about 10,200 words, of which about 8,200 are prose and
    the rest tables and captions; the appendices hold about 9,900. This is still longer than nine pages.
47. **Where the argument breaks (must fix).** Text: one story (the switch was the user's decision; the
    paper studies the cost and the graft, not whether the pages help) told in §1 and §6.1; the results
    list replaced by Table 8 with uncertainties; contribution 4 split (protocol and deviations in §7,
    data hygiene in Appendix D).
48. **Figures (should fix).** Text: all eight embedded with corrected captions (F1 warns about the
    one-sided average; F8 carries the prediction interval and the 140 rows). Requests to the figures
    agent, who owns `figures/`: F1 with a causal window on both sides; F8 with the band, the local
    points and Table 7's line; a throughput waterfall from Table 2; a calibration plot from Table 5;
    F6's "per CHANGES 19.7" replaced by the re-run; F8's "16 rows / 144 rows" by 12 / 140.
49. **Words and units (should fix).** Text: §2.2 defines exact / equal to rounding, piece / micro-batch
    / step, live / synthetic, k tokens/s, GiB / GB, ratios as ×; abstract 221 words; related work
    stripped of our own numbers.
50. **One source of truth (should fix).** Computed: `L2/make_numbers.py` writes `L2/numbers.json`, which
    `CLAIMS.md` cites. Limit: `figures/make_data.py` belongs to the figures agent; extending it is
    requested.

## (7) Best paper

51. **One sharp, tested systems claim.** Text: the measurement study leads, with the intervention series
    (Table 3), the predictor test (Table 4) and the loop count (§5.5); the predictor test is reported as
    exploratory.
52. **The graft as a measured method.** Text + computed: points 21, 40, 35, 36; the Adam transient
    (§6.2); positioning against [47] (§3).
53. **An honest, objective outcome.** Text + computed: exact scores, bits per byte, then the blind
    comparison (Table 8); the test on the 189 untouched prompts with the power stated; the
    pre-registered judge [TBD].
54. **Nine pages with appendices.** Text: see point 46.
55. **How the work was done (optional).** Text: Appendix L.
