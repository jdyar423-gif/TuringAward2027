# RESPONSE_2: the third pass of the lead author

24 November 2026, 16:47–17:45 UTC. Inputs read in full: `paper.md` (second pass, 15:12), `FACTS_2.md`
(every section, including those added at 16:52 and 17:35 while this pass ran), `FIGURES.md` (also its
update of 17:19 with the F12 section and known discrepancy 8), `REVIEW_1.md`, `RESPONSE_1.md`,
`R/docs/MICRO_GRAFT.md`, `W/micro_graft.md`, `R/docs/CHANGES.md` §19.9, `R/docs/EVAL_PROTOCOL.md` §5, 7, 8,
`N/OPERATIONS.md` ("Endgame (v9)", "Final blind evaluation: the mechanics", later state notes). New
work is in `N/paper/lead3_work/` (11 MB; `lead2_work` untouched). The second draft is kept as
`lead3_work/paper_v2.md`, its claims file as `lead3_work/CLAIMS_v2.md`. Abbreviations as in `CLAIMS.md`
(`L3/` = `lead3_work/`). The repository was only read (HEAD `b1e9140` at the start; the operator committed
`190e0b2`, the restart arms, during the pass); `git status` is clean.

## What changed, by section

- **Status line.** Third pass; newest record step 46,350 at 16:27:21 UTC (p = 0.8794; ledger 46,351 steps,
  12,150,636,544 tokens at 16:27:24) from the snapshot of 16:28; newest held-out record step 46,178;
  training end 18:33:49 (the run's setup event; the task text and the operator's notes say 18:33:45, a
  summary: the raw record is used and the difference is stated).
- **Abstract.** Adds the controlled graft (every graft below the arm without the new paths; 0.69 of the
  from-scratch gain recovered at half; a restart on the remaining half 0.23 nats worse than that graft)
  and the three kinds of reader (Opus 39 of 42 and the registered judge 16 of 17 decisive pairs for
  Falcon-H1-Tiny; against MobileLLM-R1.5 20 of 27 and 7 of 9 for us) with the scale (Falcon-H1-Tiny beats
  MobileLLM-R1.5 in 144 of 150 decisive test pairs). The development comparison of `final` (2 / 43) left
  the abstract; it stays in Table 7. Title unchanged: every claim in it still holds.
- **§1.** Contribution 1 shortened; contribution 2 now includes the controlled experiment; contribution
  4 is the standing read by three readers with the reference scale. "What we do not claim" adds the
  micro page gain (1.24% at equal tokens), equal wall-clock time, and the low bar of MobileLLM-R1.5.
- **§2.** §2.2 defines the logged rate as the trainer computes it (262,144 ÷ the running average of the
  step time, weight 0.2, training time of the ledger, checkpoint writes left out) and the achieved-rate
  gap as at most 0.13% (was 0.1%). §2.4 gives TPU time at the cut-off and the chat ramp from p = 0.67.
- **§3.** Condensed to a summary that keeps every cited reference's role; the full section is
  Appendix N.
- **§4.** Trimmed; the strict reading's minimum and maximum moved to Appendix A.
- **§5.** §5.1: single pieces are medians, whole steps single timings (FIGURES.md item 4). §5.2: the
  first row of Table 2 reads "115,394 and 116,718 (two monitor readings)" with the start-up benchmark
  111,893 beside it (FIGURES.md item 2; also in §5.2's text, F.3 and Table H1); the 2,048-token row is
  189,074 (n = 190); the ratio from the reference becomes 2.41–2.44×. The kernel equality paragraph and
  the kernel figure moved to Appendix F.2 / Figure F1; the waterfall to Figure F2. §5.4: Table 4's FLOP
  cell reads "not defined without `final`; 10% (21%) on the five `v9`" (FIGURES.md item 1); the cost model
  by change is Table F5 (F.7). §5.5 condensed; the FLOP arithmetic is F.6, "what did not help" F.4.
- **§6.** §6.3: projections from the newest record. §6.4: the function-check table is Table G1. §6.5:
  checkpoints to 46,178, the pages switched off at 46,178 (new CPU evaluation), the chat ramp and the
  faster fall of the chat rows (FACTS_2, 16:52) as confounds. **New §6.6**: the controlled graft at the
  micro width (design, checks, Table 6, reading, restart arms, limits, Figure 6). **§6.7** rewritten: what
  the micro experiment settles (function preservation in practice, never worse than not grafting, the
  share recovered falls with the graft point, a restart far behind) and what stays open for the live run.
- **§7.** §7.1 names the registered readers; §7.2 deviation 1 now says the registered judges were run on
  the development pairs and the Opus readers are an addition; §7.3 adds the registered judge's tie
  rates (72%, 85%) and what n = 200 resolves at them; §7.4 has Table 7 (every reader on the four
  development pair sets and the reference set), the pair-by-pair agreement, the scale, the readers'
  verdicts in absolute terms, Table 8 (exact scores and bits per byte, 0.980 at 46,178) and Figure 7;
  §7.5 points to the shells of Appendix M.
- **§8, §9.** Limitations revised (the controlled experiment's reach, the chat ramp, the three
  readers, no matched control planned); conclusion revised for the controlled experiment and the
  scale.
- **Appendices.** A: strict reading of `v9`. C: the chat ramp (C.1), Table C1 at the cut-off, P1 and the
  projected end in Table C2, Figure C2 (moved), the 2,048-token stretch in C.3, Figure C1's caption, the
  current plan for the test replies (C.4). D: Figure D1's cut-off. F: F.1–F.7 (moved material, `v9`
  switch details, the rates of runtime 1 as on disk). G: G.1 tool, G.2 function check (Table G1), G.3
  data stream after the graft, G.4 the controlled experiment in detail (with the restart arms), G.5
  held-out tables (Table G2 to 46,178, Table G3 the 34,166 comparisons). H, J, L updated. M: twelve table
  shells R1–R12, each naming the file it will be filled from. N: related work in detail (new).
- **Figures.** Rebuilt with `make_all.sh --cutoff 46350` at 16:57 and again at 17:31 (same cut-off) so
  that F12 draws the restart arms; no figure script changed. Captions in `paper.md` updated for the
  cut-off (Figures 2 [F4], C1 [F1], C2 [F2], D1 [F5], F2 [F9]) and for the new checkpoints (Figure 5 [F8]).
  Figure 6 [F12] has a caption written from the data, to be reconciled with the figures agent's.
- **Renumbering** (main text): Figures 3, 4, 5, 6, 7, 8, 9 of the second draft (F9, F2, F4, F10, F6, F8,
  F11) are now Figure F2, Figure C2, Figures 2, 3, 4, 5, 7; Figure 2 (F3) is Figure F1; Figure 6 is new
  (F12). Tables 5 → F5, 6 → G1, 7 → 5 (34,166 rows → G3), 8 → 7 (readers) and 8 (exact scores); Table 6
  (micro) is new. Every table and figure of the main text is cited in the main text
  (`L3/check_refs.py`).
- **Length.** Main text (abstract to conclusion) 9,536 words counted without table rules, of which 977
  in table rows (second draft: 10,685). Detail was moved, not cut: Appendix N, F.1–F.7, G.2–G.5, A.
- **CLAIMS.md.** Entries C400–C448 for every new or changed number, with the source and the command; the
  second-pass rows that were changed or withdrawn are marked in place and listed at the end.

## Every number that changed, and why

| what | second draft | now | why |
|---|---|---|---|
| newest record | step 37,650, 13:05:57 | step 46,350, 16:27:21 (ledger 46,351 steps, 12,150,636,544 tokens) | newer snapshot (16:28) |
| live rate at 2,048 tokens | 188,980 (n 16) | 189,074 (n 190), 4.0% below 1,024 | 174 more records |
| achieved vs live | ≤ 0.1% | ≤ 0.13% (0.124% at 2,048) | more records; FIGURES.md item 3 |
| reference's live rate | 112–117 k (monitor readings) | 115,394 and 116,718 (monitor); 111,893 benchmark | FIGURES.md item 2; `S/colab_bes24h_final.log` |
| reference → kernel | 2.40–2.51× | 2.41–2.44× | divided by the two monitor readings, not by 112k–117k |
| Table 4 FLOP leave-one-out | 20% (66%) | not defined; 10% (21%) on five `v9` | floating-point artefact (FIGURES.md item 1) |
| projected tokens at the end | 1.36 × 10¹⁰ (1.360e10) from the 10:57 restart; deficit 2.28e9 (14.4%) vs 1.588e10 | 1.358 × 10¹⁰ (51,821 steps) from the ledger at 16:27:24; deficit 2.33 × 10⁹ (14.6%) vs 1.591 × 10¹⁰ | projection from the newest record; `final`'s counterfactual recomputed from its live median (281,399) instead of a hard-coded 1.588e10 |
| TPU time | 15.22 h at 12:00 | 19.68 h at 16:27:24, 78% in training steps (runtime 3: 55,290 s) | cut-off |
| held-out | to 35,294 | to 46,178: since the graft −0.1679 [−0.1828, −0.1533]; line −0.088 ± 0.171 (minutes), −0.112 ± 0.111 (log tokens), −0.127 ± 0.084 (equal tokens); pages off − on +0.2166 [0.2019, 0.2320]; pages off − graft point +0.0487 | `heldout_v9.jsonl` (16:47) and a new CPU evaluation at 46,178 |
| "pages carry about all of the improvement" | true at 34,166 and 35,294 | at 46,178 they carry more than all of it (+0.217 against −0.168) | new checkpoint |
| local vs runtime | to 10⁻⁸ | within 10⁻⁷ (46,178 web differs by 3.0 × 10⁻⁸) | new evaluation |
| bits per byte (ours) | 0.998 (35,294) | also 0.980 (46,178) | new evaluation |
| tests | 1,356 at `8a86bcb` | 1,367 at `b1e9140` | the micro graft commit added 11 |
| three-point line ±0.025 at 592.5 min (CLAIMS C294) | in CLAIMS only | withdrawn | it belonged to a line through all 140 rows (FIGURES.md item 6); the comparison set's line gives ±0.035; not cited |
| Figure captions' cut-off | 37,650 (13:05:57); "159 of `v9`", "16 records … too few for a line", "steps 36,900–37,650" | 46,350 (16:27:21); 333 of `v9`; a line is drawn at 2,048; steps 36,900–46,350, 1,386.5 ms | rebuild |

New numbers (all in `CLAIMS.md` C400–C448): the micro experiment (Table 6, G.4, restart arms), the
readers (Table 7, κ, pair-by-pair agreement, clipping, the registered judge's tie rates and resolution),
the reference set, the keyword counts over the readers' notes, the chat ramp, the 2,048-token training
losses, the main run's rate during the micro runs.

Checks made in this pass: `L3/micrograft.py` reproduces `graft_report.txt` in every printed digit and
`readers.py` every report of `study/proxy/reports/ab/` for the sets used; the figures agent's
`fig12_micrograft.py` prints the same restart numbers; `make_numbers.py` matches the figures' live
rates at the cut-off (the figures' cross-check still reads `lead2_work/numbers.json`, so it prints a
MISMATCH for the 2,048-token rate, expected); the evaluation at 46,178 with pages on reproduces the
runtime's all-row loss to 2.2 × 10⁻⁹. Inline checks (not saved as scripts): the restart minus `v9_pages`
at 3 × 10⁶ tokens, −0.0509 ± 0.0086 (per seed −0.0411, −0.0555, −0.0563), and the local/runtime
differences at 46,178 (web 2.96 × 10⁻⁸, chat 3.2 × 10⁻⁹, all 2.2 × 10⁻⁹).

## Still [TBD]

Authors; the tokens and final step at 18:34 (§6.3, Table C2, M9); the `neutral` arm of MobileLLM-R1.5 on
`dev` (deviation 4); every cell of Tables M1–M12 (R1 test split × three readers; R2 slices; R3 length
and diagnostics; R4 exact scores; R5 trained against preference-optimised weights; R6 the export's
held-out losses; R7 preference optimisation; R8 decoding selection; R9 training and TPU totals; R10 the
16-bit check; R11 the 189 untouched prompts, whose restriction is not yet scripted; R12 decoding
throughput); what each post-training stage did (C.4); the share of flagged test replies (D.5); decoding
throughput of the export (E.5); the licence review (K); the test-split comparison in the abstract,
§7.4/Figure 7 and §9. The shells name future files (`final-TAG-vs-*` reports) as item 6 asks; none of
them exists and none was opened.

## Not used from FACTS_2, or used differently

- §2 (11:10) "`v9` is about 0.016 nats ahead at equal training time": superseded; the gap is inside the
  prediction interval of the comparison set's line (±0.035 at 592.5 minutes) and the paper reports
  intervals, not a 4-digit gap.
- §2 "the web rows improved 4.4 times faster per token … the mixture change does not touch them": not
  used. From p = 0.67 the base-text share of the stream falls from 0.75 to 0.38, so the mixture does touch
  the web rows' exposure, and the web rows' fall slowed (2.6977 at 41,850, 2.6867 at 46,178).
- §1 single log readings (177.8k, 196.7k) and the 16:44 log line (step 46,950, 189.0k): the paper uses
  medians over step records up to its cut-off (step 46,350); the cut-off was not moved for later lines.
- "Three quarters of our replies are judged unusable (fluent, on the topic, wrong or empty of
  content)": used narrowly (44 of 60 pairs both bad when both replies are ours; the notes name wrong
  facts or arithmetic in 20, incoherence in 14); "fluent" and "on the topic" are not supported by a
  count of the notes and are not used.
- "The Opus readers as the stronger reading": not used; the paper says what each reader decides and that
  agreement between orders is not validity.
- "Looping meta-reasoning with boxed answers" (MobileLLM-R1.5): used as a keyword count of the notes
  (loop or repeat 87 and 83, boxed 46 and 49 of 200); "meta-reasoning" matched only 3 and 8 notes and is
  not used.
- The 16:52 list of candidate causes for the chat rows' faster fall: used, with the chat ramp added,
  which FACTS_2 does not name and which is the schedule's design (`chat_frac_at`).
- `docs/MICRO_GRAFT.md`'s "4.6 to 6.2 s per step": not quoted (FIGURES.md item 8: `graft_75_s2` ran at
  3.48 s).
- "11.57 and 11.59 min per 500 steps before and during": replaced by the recomputed medians (189,054,
  n = 73, and 189,086, n = 117; 11.57 min both).
- The housekeeping note (stale `judge-opus.jsonl` files moved out): `readers.py` reads only the files in
  `ratings/<set>/`, which no longer contain them.

## For the figures agent and the reviewer

- `F8_heldout`: at the new cut-off the arrow label at 917.9 minutes is cut at the right edge ("−0.0 /
  ± 0.1"); the pages-off value at 46,178 (2.2754, `L3/heldout46.json`) is not drawn.
- `F2_throughput`: the runtime-3 note ("Records to step 46,350 …") runs into the 178.1 label.
- The figures' live-rate cross-check reads `lead2_work/numbers.json` (`paths.py`); the text now uses
  `lead3_work/numbers.json`, which equals the figures at the cut-off.
- `FIGURES.md` still numbers the figures as in the second draft and its captions carry the old cut-off;
  the mapping is under "Renumbering" above.
