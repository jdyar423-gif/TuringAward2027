# REVIEW_2: programme-committee review of the third pass

24 November 2026, written after 17:40 UTC. Object: `paper.md` (third pass, 17:37), with `RESPONSE_2.md`,
`CLAIMS.md` (C400–C448), `FACTS_2.md`, `FIGURES.md` and `figures/*.png`. Base of the pass:
`lead3_work/paper_v2.md`, byte-identical to `docs/paper/paper.md` at `a316f8c` (checked with `cmp`).
Repository read only (HEAD `190e0b2`, clean). Nothing opened whose name starts with `final-`; the notes
of the reference set were read from `N/final_eval/ref/ref-falcon-vs-mobilellm-test/` (opponents only).

Every check below was recomputed from the raw files by scripts in `paper/review2_work/` (CPU 2,3,
`venv/bin/python -I`), not from CLAIMS or the reports:

| script | what it recomputes | log |
|---|---|---|
| `micro_check.py` | all 15 micro records (`micro_graft/6M/results_all.jsonl`, checked equal to each arm's own `results.jsonl`) and the 3 restart records: means, sd (n−1), paired differences, shares, per-seed values, checks, timing | `micro_check.log` |
| `live_check.py` | the v9 train log of the 16:28 snapshot: 2,048-token medians, achieved rate, train_end, chat share, loader rows, C.3 losses, rates around the micro runs | `live_check.log` |
| `heldout_check.py` | 46,178 on/off from the npz files against `heldout_v9.jsonl`; `final`'s three-point lines at every checkpoint, also with `final`'s 2,048-token rate | `heldout_check.log` |
| `readers_check.py` | every reader on s1, s2, ngram3 and the reference set: the repository's key check (`make_ab.load_key`, `invalid_records`), then my own tally, Wilson, exact sign test, κ, pair-by-pair agreement | `readers_check.log` |
| `newnums.py` | every number of `paper.md` absent from the base (`newnums.txt`), the list I worked through | – |

**Verdict: not ok.** 2 must-fix, 14 should-fix, 19 minor. The numbers are, with three rounding slips,
right: the micro experiment, the three readers, the reference set, the held-out evaluation at 46,178 and
the live rates reproduce to the printed digit. The defects are in what two sentences say about those
numbers, in what the paper says about the registered protocol, in what the abstract leaves out, and in
the result shells and figure files that the last pass will depend on.

---

## Must-fix

**1. §6.5 (ii), lines 460–466: "inside 95% prediction intervals" is false at 46,178.**
The text: "`v9` lies 0.03 nats below it at step 35,294 and 0.09–0.11 at 46,178, inside 95% prediction
intervals of ±0.05 to ±0.17 (at equal tokens, … −0.127 ± 0.084 at 46,178)". Table 5 itself prints
the log-token line at 46,178 as −0.112 / ±0.111. `lead3_work/heldout3.json`
(`extrapolation_final_three_points["46178"]`): gap −0.1124, half-width 0.1108, so it is outside by
0.0016. At equal tokens −0.127 ± 0.084 is outside by 0.043. My recomputation (`heldout_check.log`)
also has the log-token gap outside at 42,932 (−0.0969 ± 0.0951) and 45,096 (−0.1083 ± 0.1057). Only the
minutes line (−0.088 ± 0.171) keeps every point inside, which is what Figure 5 draws.
*Fix:* say that the minutes line has `v9` inside its interval, while the log-token line and the
equal-token comparison have it just and clearly outside theirs. Rest "the gap is not resolved" on what
the line cannot model (one residual degree of freedom, no learning-rate decay, no chat ramp, the stream
change at the graft), not on the intervals. See should-fix 7 for the token count behind the log-token
line.

**2. The registered protocol's K = 1 and K-matched control are dropped without a deviation entry
(§7.1–7.2, §8.6, §9; Appendix M R5).**
`docs/EVAL_PROTOCOL.md` §3 ("What any claim must disclose", lines 145–152) says "Results are reported
with the library on and off, and with K = 1", and under "Control arms" that "Each opponent is also run
with the same K … This is the plan's K-matched control". The second draft planned R5 for this:
"our model with one particle against both opponents; opponents with six samples, matched in samples
and in selection" (`paper_v2.md` l.1689; l.851 "the matched controls are run (R5)"). This pass
replaced R5 with trained against preference-optimised weights. §8.6 now says "no matched control is
planned", but §7.2's five deviations do not list it, §7.1 does not say the protocol required it, and §9
says the outcome "is decided by a protocol fixed in advance, with the deviations stated here".
*Fix:* add a deviation: no K = 1 test result, no K-matched or selector-matched opponent arm, and library
on/off does not apply because `v9` ships no library. Alternatively schedule them. Then amend §9.

---

## Should-fix

**3. The abstract (lines 27–30) and §9 (755–758) do not name the checkpoint or say that 7 of 9 is not
significant.** The abstract reads: "On 60 development prompts, Falcon-H1-Tiny was preferred in 39 of 42
… by Opus readers and in 16 of 17 by the pre-registered judge; … our model won 20 of 27 and 7 of 9."
These are `v9` at step 31,912 (p ≈ 0.56), not the trained model. The 7 of 9 has p = 0.18, with 51 of 60
pairs tied for that judge (`readers_check.log`). "Opus readers" are Opus 5.5 agents of the builders'
family, added to the protocol, and the abstract does not say so. §9's "the model is far behind" has the
same gap. *Fix:* "at step 31,912 (0.56 of training)", "20 of 27 (p = 0.019) and 7 of 9 (p = 0.18; 51
of 60 ties)", "Opus 5.5 agents, an addition to the protocol"; in §9, "a mid-run checkpoint".

**4. The abstract (24–27) and §1 contribution 2 (57–61) state the micro results without their reach.**
"In a controlled version at small width (three seeds), every graft ended below …, a graft at half the
budget recovered 0.69 …, and restarting … ended 0.23 nats worse" stands next to the live graft at 0.505
and invites the reader to transfer the sizes. The limits appear only in §6.6 and §9: 6 × 10⁶ tokens,
equal tokens rather than wall clock, a page worth 1.24% there, and per-seed shares of 0.62–0.87 at half.
*Fix:* "at width 128 and 6 × 10⁶ tokens, at equal tokens; the sizes do not transfer to the live run".

**5. §6.6, lines 526–527: an equivalence claim that three seeds cannot carry.** "Grafted at a quarter
of the budget, the page ended where the page from the first step ended (−0.0031 ± 0.0113, signs mixed
across seeds)". The per-seed values are +0.0088, −0.0138 and −0.0043. A t interval of the mean with
2 df is −0.0031 ± 0.028, about ±54% of the page's whole gain (0.052). *Fix:* "no difference was
resolved (−0.0031 ± 0.0113, sd of three seeds)".

**6. §6.6, lines 533–534: the 0.234 is not the worth of the discarded tokens.** The text says "the
tokens a restart discards were worth more than four times the page's whole gain (0.234 against
0.052)". But 0.234 is `graft_50` − restart, which includes `graft_50`'s own page gain (0.036 over
`ref`). The comparison that isolates the discarded tokens, net of the page, is `ref`(6M) − restart =
0.199, or 3.8×. That is the figure FACTS_2 (17:35) uses for "about four times". `v9_pages`(6M) − restart
= 0.250 gives 4.8× (`micro_check.log`). *Fix:* "four to five times (0.199–0.250 against 0.052,
depending on the comparison)", or use 0.199.

**7. The log-token line's equal-time token count does not match §6.3.** `heldout3.py` puts `final`'s
equal-time tokens at 46,178 at 1.419 × 10¹⁰, because it runs `final` at its 1,024-token achieved rate
(281,220) through the whole 2,048-token phase. §6.3's counterfactual for the same `final` uses a scaled
252,972 at 2,048 (`numbers.json` `proj.final_total`). With that rate the count is 1.383 × 10¹⁰ and the
gap is −0.115 ± 0.107, outside the interval (`heldout_check.log`). *Fix:* use one rate in both places.

**8. Appendix M: several shells cannot be filled from the file they name.**
(i) **M6.** `export.json`'s check record (`bes/export.py` `check_16bit`) is computed on
`heldout_batches(rows=16)`. `export_run` defaults to `heldout_rows: int = 16` (l.1706), and
`scripts/run_24h.py`'s export stage (l.1473) does not pass it. T is `data.seq_len` = 1,024, and only
the per-domain losses for web, chat and proxy are stored. So the record holds 16 rows per domain, no
assistant-token loss and no bits per byte. The rows "web (64 rows)", "chat (64 rows)", "chat, assistant
tokens", "web + chat" and "bits per byte" cannot be copied from it. §6.5 l.431 ("The held-out rows are
those of the export's check: 64 web rows …") has the same mismatch: the export uses the first 16 of
them.
(ii) **M4** names no output file. `lead2_work/exact_dev.py` keeps only `split == "dev"`, so a test
version and its JSON must be named.
(iii) **M5**'s last two rows have no file: "[TBD: if run; pair-set name not fixed]" and the `compare`
output.
The preamble's "Every cell marked [TBD] is copied from the file named for its table; nothing is to be
computed by hand" is therefore false for M4, M6 and M11. *Fix:* for M6, name `heldout_eval.py` on the
export's weights at 64 rows plus a bits-per-byte script, or relabel the export columns as the 16-row
check. Name the M4 script and its output, and the M5 pair sets and `compare` output.

**9. The `neutral`-arm [TBD] has no run behind it (lines 600–601, 1816).** "[TBD: the `neutral` arm on
`dev`, before the test split is opened]". `N/OPERATIONS.md` ("Endgame (v9)", "Final blind evaluation:
the mechanics") schedules no dev comparison against the `neutral` arm, and the Opus test readings start
at about 19:30. *Fix:* schedule it (who, which readers, by when) or record it as a deviation, with the
`recommended` arm used by default.

**10. Figure 6 has two captions, neither final.** The paper's caption (l.547–555) ends in an editorial
bracket ("[Caption from the data; to be reconciled …]"). It also does not say that no interval is drawn
or that the per-seed shares spread from 0.86 to 1.33 at 0.25; FIGURES.md says both. FIGURES.md's F12
caption (l.562–589) still carries "[TBD: the restart arms … expected at about 17:40 UTC]", yet the 17:31
rebuild draws them (`make_all.log`: "F12: restart arms with half the budget: drawn in (a)") and the PNG
shows the squares. The numbers in both captions agree with my recomputation. *Fix:* one caption, in
both files.

**11. FIGURES.md is stale.**
- It still uses second-draft numbering (l.161–176) and calls F12 "Figure [TBD]".
- Five captions keep the cut-off step 37,650 (l.260, 297, 327, 483, 529).
- F8 says "agree to 10⁻⁸" (l.432).
- Discrepancies 1–7 are not marked resolved.
- The default cut-off still comes from `lead2_work/numbers.json` (l.46; `make_all.log` prints MISMATCH
  v9_2048), so a plain `make_all.sh` rebuilds the figures at 37,650.

*Fix:* update FIGURES.md, and point `paths.py` at `lead3_work/numbers.json`.

**12. Figure 5 (`F8_heldout.png`) is incomplete.** The arrow label is cut ("−0.0 / ± 0.1"), and the
pages-off value at 46,178 (2.2754) is not drawn. There are rings only at 29,722, 34,166 and 35,294, but
§6.5 says four checkpoints were evaluated again on the CPU, and the caption's "rings: evaluated again on
the CPU" implies all four. *Fix:* redraw. Until then, name the three ringed steps in the caption.

**13. Figure C2 (`F2_throughput.png`): the runtime-3 note runs into the "178.1" label** and into the
lines of tags 7–8.

**14. §7.4, line 662: "as a bare boxed answer in 46 and 49" goes beyond the count.** `ref_notes.py`
counts clauses about MobileLLM-R1.5's reply that match `\bbox`. The notes themselves say "A is boxed
nonsense, no story", "B degenerate boxed loop, truncated", "A outputs boxed fragments", "B only boxes
'Santa Claus'". "Bare" is not counted, and many of these notes overlap the loop count. *Fix:* "mention a
boxed answer (often nonsense or looping)".

**15. §6.7, lines 559–565: "settles" is too strong in two places.** "About 0.7 at half the budget" has
per-seed values of 0.62–0.87 (sd 0.14). "A graft is function-preserving in practice (difference exactly
0 at nine grafts)" was shown on the CPU in float32; on the chip in bfloat16 it was not measured (§6.4,
§8.4). *Fix:* give the range, and add "on the CPU in float32".

**16. Table M6, line 1901: the proxy row reports our model's loss on test-split text.** The row is
"proxy (12 rows; reported apart) … 2.6361". The proxy rows are built from the user turns of all 400
prompts, test included (§6.5, deviation 3). A shell that will report this for the export says something
about our model on test-split material. *Fix:* drop the row, or rebuild the proxy rows from dev prompts
only.

---

## Minor

17. Lines 532 and 565 give "+0.227 to +0.242". The per-seed restart − `graft_50` values are +0.234790,
    +0.241451 and +0.226504, so the range is +0.227 to +0.241. The 0.242 comes from rounding the
    report's 0.2415 a second time.
18. Table 7, line 636: the `qwen3-4b-2507` interval for `v9` against Falcon is "[0.264, 0.501]". The
    Wilson upper bound is 0.501515, which rounds to 0.502 (the same double rounding of 0.5015). The
    same value is in C421 and FACTS_2.
19. §7.2 deviation 1 (590–592) says the development pairs "are also read twice on exchanged sides by
    Opus". The s1 sets (`final` 25,280) were read once (`opus-1`), as Table 7 says.
20. G.4, line 1607: "189,086 over 117 during" covers 14:12:51 to the cut-off 16:27:21, not the whole run
    to 16:41:01 (`rate_during_micro.py`). The restart arms (16:42–17:29) are not covered at all. Say so.
21. §6.5, line 437: "within 10⁻⁷" does not hold for the proxy rows at 46,178, which differ by
    2.20 × 10⁻⁷. Web differs by 2.96 × 10⁻⁸ and chat by 3.24 × 10⁻⁹. C419 omits the proxy figure too.
    Either restrict the claim to web and chat or state the proxy difference.
22. Table E1, line 1310: the `v9` row gives "of which both bad: – (ties)", while §7.4 (668) gives 44 of
    60, the `opus-1` raw count.
23. Step 31,912 has two p values: "p = 0.565" in Table 7 (the restore event, 0.5651) and "p = 0.56" in
    §7.4 (672). The checkpoint was written at p = 0.5623 (end event). Use one convention.
24. Table 6's caption says "at 5,996,544 tokens", but the restart row is at its own end, 2,998,272.
25. §6.6 Checks (503–506): "per row as well" holds for the held-out domains (`max_row_difference` 0). The
    8-row training-loss check records only one aggregate difference.
26. §8 item 1 (716): "three seeds of `ref` spread by 0.18%". 0.18% is the sd (0.0076 of 4.1912); the
    range, 4.1830–4.1979, is 0.36%. Say "sd".
27. C.1 (1017–1018): "By itself it lowers … the held-out chat loss of any configuration" was not
    measured. Write "is expected to lower".
28. Appendix J (1744–1745): the restart records are in commit `190e0b2` (HEAD), not `b1e9140`.
29. CLAIMS.md has four stale entries:
    - "Open items" still lists the registered judges on dev and the 2,048-token rate as open.
    - C418 cites "6.5 (iv)", a part that does not exist.
    - MAL is described as the 16:57:45 rebuild, but `make_all.log` was overwritten at 17:31–17:32.
    - C416's recompute command needs checkpoint 46,178 from `snap_main.prev`, but `snap_main.prev` now
      holds the 17:14 snapshot (checkpoint 48,342) and the 16:28 snapshot is gone. The saved `.npz`
      files are now the only local copy of that evaluation.
30. Figure 6 is embedded as `.png` (545); every other figure is `.svg`.
31. Figure 7 draws only the Opus readers, the added instrument; the registered judge appears only in
    Table 7.
32. §6.6 (529–531): "a late graft loses mainly the time it does not have". The gains after the graft fall
    with the learning-rate factor (0.0192, 0.0161, 0.0123 at 0.775, 0.55, 0.325). "Consistent with" fits
    better than "mainly".
33. M4 (1867): "(of [TBD])". The number of constrained test prompts is a property of the prompt file, it
    exists now, and it involves no reply.
34. M3 (1856): "probability mass on A, B, C". The reports carry `judge.format_mass_mean` (the total mass
    on the three letters); name that field.
35. Pre-existing, §7.3 (608): "The protocol's power table assumed ties at up to 30%". The table in
    `EVAL_PROTOCOL.md` §7 has 30% and 50% columns.

---

## The seven questions

**(a) Numbers.** These reproduce from the raw files to the printed digit, except items 17 and 18:
- **Micro graft:** every value in Table 6 and G.4, including the losses, paired differences, shares
  1.060/0.689/0.237 with their per-seed values, shares by domain, check losses, the 18 pre-graft
  evaluations equal row for row, 9 grafts with difference 0, and the restart values 4.3897 ± 0.0017,
  +0.2342 ± 0.0075, +0.1985 ± 0.0062, +0.2504 ± 0.0074, −0.0509 ± 0.0086 and −0.0844/−0.0835/−0.0836.
  The restart's prefix-stable data order was checked in `microbench.row_kinds`.
- **Readers:** all 13 reader × set rows of Table 7, κ 0.209/0.183, the agreement counts 14 of 16, 6 of 6,
  26 and 21, ties of 93–97%, 30 clipped pairs, and the 44/31/14 and 46–55 both-bad counts. Every report
  in `reports/ab/` equals my tally, and no rating failed the commitment check.
- **Reference set:** 144/6/50, 0.845 [0.788, 0.889], decisive 0.960 [0.915, 0.982], and the categories.
- **Held-out at 46,178:** −0.1679, pages off − on +0.2166 and pages off − `final` +0.0487 (bootstrap
  reproduced), web/chat, quartiles, bpb 0.9801 over 258,817 bytes, local against runtime, and every new
  row of Table G2.
- **Live numbers:** 189,074 (n = 190), achieved gap 0.123%, train_end 1795545229 = 18:33:49, the ledger
  (46,351; 12,150,636,544; 55,310 s), TPU time 70,835 s = 19.68 h (78%), the projection 1.358 × 10¹⁰,
  51,821 steps, 2.33 × 10⁹ (14.6%) against 1.591 × 10¹⁰, the chat ramp (75% to 38.4% base rows, 0.627),
  C.3 (2.404 to 1.954; 0.40 to 0.21), the rates before and during the micro runs, 2.411/2.439×, the
  power values, and 1,367 tests.

**(b) The micro interpretation.** §6.6 states its limits where it makes its claims. The overreach is in
items 5, 6 and 15 and, by omission, in item 4. §6.7's list of what stays open for the live run is
accurate.

**(c) Abstract, introduction and conclusion.** The numbers are consistent with the body, apart from
the rounding in item 17. The missing context is in items 3 and 4.

**(d) The standing against the opponents.** The three readers are reported apart and never averaged.
The low bar is stated plainly (§1, §7.4, §9). The protocol's status is incomplete (item 2, and item 9 on
the `neutral` arm). On the test split for our model, the only statement is the planned proxy row (item
16). The mid-run samples on two seen test prompts are pre-existing and disclosed.

**(e) The figure blocks.** All 12 exist and are cited (`check_refs.py`). Problems: items 10–13 and 30.
Figures 2, 3, C1, D1, F1 and F2 match their captions and the 17:31 log.

**(f) The [TBD] items.** All name what they wait for and when, except item 9. None hides a number that
FACTS_2 already has: the dev generation cost of 1,798 s is quoted in M12.

**(g) The shells.** M1, M2, M3, M7–M10 and M12 can be filled from their named files (fields checked in
the reports, `export.py`, `prefopt.py` and `gen_test_replies.py`). M4, M5 (in part) and M6 cannot (item
8). M11 is honestly marked as not yet scripted.
