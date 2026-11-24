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

## After the check

24 November 2026, 18:10–18:55 UTC, the lead author's answer to `REVIEW_2.md` (2 must-fix, 14
should-fix, 19 minor). The reviewer was right on every item; none is contested. Every changed number was
recomputed from the raw records by new scripts in `N/paper/lead4_work/scripts/` (CPU 2,3, `venv/bin/python
-I -B`), outputs beside them in `lead4_work/`; the third pass's files are kept as `lead4_work/paper_v3.md`,
`CLAIMS_v3.md`, `FIGURES_v3.md`, `RESPONSE_2_v3.md` and `orig_*.py`. New claims C449–C465 and the list
"Changed after the check" are at the end of `CLAIMS.md`. Repository read only (HEAD `190e0b2`); nothing
whose name starts with `final-` was opened. The restart arms were already in the paper (not [TBD]): the
archive `micro_graft/3M/results.tar.gz` was unpacked again into `lead4_work/micro3M_raw/` and found equal
to the copies beside it; FACTS_2 has no section after 17:35, which the paper already used.

| # | defect | what was done |
|---|---|---|
| 1 | §6.5 (ii): "inside 95% prediction intervals" false | `heldout4.py` recomputes the three forms of `final`'s line at all 16 `v9` checkpoints. Minutes line: inside at every checkpoint (−0.088 ± 0.171 at 46,178). Log tokens at equal time, with `final`'s tokens at the rates of §6.3 (item 7): −0.115 ± 0.107 at 46,178, outside from step 41,850 on. Equal tokens: −0.127 ± 0.084, outside from step 34,166 on (also at 35,294, −0.044 ± 0.040, which the review did not flag). §6.5 (ii), Table 5 (inside/outside per row) and the Figure 5 caption now say this; "not resolved" now rests on the line (one residual df, 44 minutes extended over 379, no learning-rate decay, no chat ramp, the stream change), not on the intervals. |
| 2 | K = 1 and matched controls dropped without a deviation | §7.1 now states what §3 of the protocol requires; new deviation 6 in §7.2: no K-matched or selector-matched opponent arm (the opponents' outputs hold one run per preset; none scheduled in `N/OPERATIONS.md`), no K = 1 test result unless the decoding selection ships one particle, library on/off not applicable (`v9` has no library); what this leaves open (selection among six, at most log 6 nats; Table E1's 1 / 3 / 56). §8.6, §9 ("the six deviations of Section 7.2; … cannot separate the model from its selection") and Appendix M (preamble; M5 says it replaces the second draft's R5) amended. |
| 3 | abstract and §9 omit checkpoint, significance, nature of readers | Abstract: "with our model at step 31,912 (0.56 of training)", "Opus 5.5 agents (of the builders' family; an addition to the protocol)", "20 of 27 (p = 0.019) and 7 of 9 (p = 0.18; 51 of 60 pairs tied)" (`readers4.py`: exact sign tests 0.019157 and 0.179688). §9: "a mid-run checkpoint (step 31,912, 0.56 of training)", both p values, "an addition to the protocol". §1 contribution 4 and "What we do not claim" likewise. |
| 4 | micro results without their reach (abstract, §1) | Both now say: width 128, 6 × 10⁶ tokens, equal tokens, new paths worth 1.24%, 0.62–0.87 by seed at half, and that the sizes do not transfer to the live run. |
| 5 | §6.6 equivalence claim at a quarter | "no difference … resolved (−0.0031 ± 0.0113, sd of three seeds; per seed +0.0088, −0.0138, −0.0043; a 95% t interval of the mean, ±0.028, is about half the page's whole gain)" (`micro4.py`: half-width 0.0282 = 0.54 of 0.0519). Figure 6's caption says the same. |
| 6 | 0.234 is not the worth of the discarded tokens | "four to five times the page's whole gain (0.199–0.250 against 0.052, depending on the arm it is set against)"; `ref` − restart 0.1985 (3.8×), `v9_pages` − restart 0.2504 (4.8×), `graft_50` − restart 0.2342 (4.5×). Added what the restart does and does not show (steep part of the curve, equal tokens, re-reads the first half of the data, its own annealing schedule; nothing about equal wall-clock time in the live run). |
| 7 | log-token line runs `final` at its 1,024 rate through the 2,048 phase | `heldout4.py` uses §6.3's assumption (`numbers.json` `proj.final_total`: 281,399 for 10,061 s after the graft restart, then 252,972): 1.383 × 10¹⁰ tokens at 46,178, gap −0.115 ± 0.107 (old: 1.419 × 10¹⁰, −0.112 ± 0.111). At 35,294 unchanged (−0.038 ± 0.052). For the record, `final`'s own tokens per training second between its evaluated checkpoints were 281,195 (26,890–29,722), 0.07% below the live median; this changes no printed digit. |
| 8 | shells M4, M5, M6 (and the preamble) cannot be filled from the files named | §6.5 now says the rows are built by `heldout_batches` with 64 per domain and that the export's check uses 16 per domain (`bes/export.py`, `heldout_rows` = 16; `run_24h.py` passes none). M6 split into (a) 64 rows: last checkpoint from `heldout_v9.jsonl` or `heldout_eval.py` on `N/snap_final`, bits per byte by the new `bpb_web.py` (reproduces 1.0172, 0.9975, 0.9801 over 258,817 bytes), and (b) the export's 16-row check with the fields it stores (no assistant-token loss, no bits per byte). M4 names `exact_split.py --split test` with every run id and its output `exact_test.json`; on `dev` it reproduces `lead2_work/exact_dev.json` field by field for the four runs of Table 8. M5: the direct pair set dropped; the `compare` rows name the command, fields and output files. M11: `restrict_report.py` (reproduces the `overall` of three `dev` reports from `per_prompt_score`) and `exact_split.py --exclude SEEN`; "not yet scripted" removed. Preamble amended. |
| 9 | the `neutral`-arm [TBD] has no run | Recorded as deviation 4: no blind `dev` comparison of the `neutral` arm is scheduled before the test readings (about 19:30), so the `recommended` arm is used by default, not by the protocol's rule; the `dev` exact scores favour it (maths 4 against 1 of 9, constraints 2 against 1 of 14, cut 42 against 54 of 75); the `neutral` arm's test exact scores stay in M4. Appendix M's [TBD] removed. |
| 10 | two captions for Figure 6 | One caption, written from the figure and the records, now in `paper.md` and copied verbatim into `FIGURES.md` (`sync_figures_md.py`); it keeps the restart sentence (+0.1985; seeds +0.1962, +0.2055, +0.1938; unequal token counts), says no interval is drawn (three seeds) and gives the per-seed shares (0.86–1.33, 0.62–0.87, 0.20–0.29). The editorial bracket and FIGURES.md's [TBD] are gone. |
| 11 | FIGURES.md stale; plain rebuild at 37,650 | `paths.py` reads `lead3_work/numbers.json` (a plain `make_all.sh` now uses 46,350; the 2,048-token MISMATCH is gone; the predictor-test MISMATCH stays by design). FIGURES.md: third-draft headings and numbering table, every caption copied from `paper.md`, cut-off paragraph, inputs table, readings of Figures F2, C2, 2, 5, C1, D1 updated from the 18:25 build (`lead4_work/make_all_1825.log`), "10⁻⁸" gone with the old caption, discrepancies 1–8 marked with how they were resolved, the citation list marked resolved (`check_refs.py`: every main-text figure cited). |
| 12 | Figure 5 incomplete | Redrawn (`make_all.sh --no-extract`, 18:25): `local46178.py` writes the 46,178 evaluation in the local-summary format and `make_data.py` reads it, so rings mark all four CPU evaluations (29,722, 34,166, 35,294, 46,178) and the pages-off diamond at 46,178 (2.2754) is drawn; the arrow moved right and the x axis widened, so the label (−0.09 ± 0.17) is whole. Caption names the four steps. |
| 13 | Figure C2: note over the 178.1 label | The note is set in narrow lines between tags 4 and 5 under the 240.1 line; checked on the PNG. |
| 14 | "a bare boxed answer" | "mention a boxed answer … often nonsense or looping"; `ref_notes4.py` (same clause attribution): of the 46 and 49 pairs, 32 and 40 have the same clause also calling it looping, repeated, nonsense, degenerate, fragmentary or incoherent. C427 changed. |
| 15 | §6.7 "settles" | Heading and text now "shows"; 0.69 (0.62–0.87 by seed); "function-preserving in practice on the CPU in float32 …; on the chip in bfloat16 this was not measured". |
| 16 | M6 proxy row reports test-split material | Dropped; M6 says the proxy rows are in neither part (deviation 3); M10 reports the 16-bit difference for web and chat only. |
| 17 | +0.242 | +0.227 to +0.241 in §6.6 and §6.7 (per seed +0.234790, +0.241451, +0.226504). |
| 18 | [0.264, 0.501] | [0.264, 0.502] in Table 7 and C421 (Wilson 0.263528–0.501515). `FACTS_2.md` (13:52) keeps 0.501; it is the orchestrator's file and was not edited. |
| 19 | s1 read twice? | Deviation 1: "once (the `final` sets) or twice on exchanged sides (the `v9` sets)"; the ratings directories hold `opus-1` only for s1. |
| 20 | G.4 rate "during" | "to the cut-off of the paper's live numbers at 16:27:21; the last 14 minutes of these runs and the restart arms (16:42:47–17:29:56) fall after the cut-off and are not covered". |
| 21 | "within 10⁻⁷" | §6.5 and Table G2: within 3 × 10⁻⁸ on web and chat (largest 2.96 × 10⁻⁸), 2.2 × 10⁻⁷ on the proxy rows at 46,178. C419 changed; C292's "below 1e−8" was also wrong (web 2.7 × 10⁻⁸ at 29,722) and is marked. |
| 22 | Table E1 "– (ties)" | 44 (raw `both_bad` of `s2-ngram3-vs-default/opus-1.jsonl`). |
| 23 | p = 0.565 against 0.56 | One convention, the stop event at which the checkpoint was written: 0.56 for 31,912 (0.5623) and 0.44 for 25,280 (0.4396); Table 7's caption says so. |
| 24 | Table 6 caption | "the restart row at its own end, 2,998,272 tokens". |
| 25 | "per row as well" | "on the held-out domains also per row (the training-loss check stores one aggregate value)"; also "on the CPU in float32". |
| 26 | "spread by 0.18%" | "a standard deviation of 0.18% (range 0.36%)"; C435. |
| 27 | C.1 "By itself it lowers" | "is expected to lower … this run does not measure that effect apart from the others"; Figure C1's caption likewise. |
| 28 | Appendix J at `b1e9140` | Names `190e0b2` (restart records) and the repository at `190e0b2`; tests 1,367 there too. |
| 29 | stale CLAIMS entries | Open items rewritten; C418 → "6.5 (iii)"; MAL now the saved copy of the 17:31 log (`lead4_work/make_all_1731.log`; the 16:57 log is lost), MAL4 the 18:25 log; C416 says the npz files are the only local copy (the snapshots have rotated again: at 18:29 `snap_main.prev` holds 17:39:40, checkpoint 49,424). |
| 30 | Figure 6 as .png | Embedded as `F12_micrograft.svg`. |
| 31 | Figure 7 draws only the Opus readers | Caption says why: they decide most pairs; the registered judge decided 17 and 9 of the 60 `v9` pairs and the second judge 3 and 3, so their bars would be almost all ties; their readings are in Table 7. Not redrawn. |
| 32 | "mainly the time it does not have" | "consistent with a late graft losing the time it does not have, with the gains also falling with the learning rate, which this design does not separate from time". |
| 33 | M4 "(of [TBD])" | "of 48" (test split: 200 prompts, 252 turns, 30 maths, 48 constrained; prompt metadata only, `exact_split.py --count-only`). M11 gets 189 prompts, 240 turns, 28 maths, 44 constrained. |
| 34 | M3 field | Names `judge.format_mass_mean`. |
| 35 | power table "up to 30%" | "gives the comparisons needed with 30% and with 50% ties". |

Also changed: the status line; Appendix J lists the after-check scripts; Appendix L records this check.
Still [TBD], unchanged: authors; tokens and final step at 18:34; every result shell M1–M12; C.4, D.5,
E.5, K; the test comparison in the abstract, §7.4/Figure 7 and §9.
