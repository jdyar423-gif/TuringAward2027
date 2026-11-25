# RESPONSE_3: the final pass of the lead author

24 November 2026, 21:10–22:55 UTC. Inputs read in full: `paper.md` (third pass, revised after the check),
`RESPONSE_2.md`, `REVIEW_2.md`, `CLAIMS.md`, `FIGURES.md` and `FACTS_2.md`, read again from "Added 21:12 UTC" to
its end at 21:38, 22:11, 22:22, 22:30 and 22:39: the sections of 21:27, 21:40, 22:13 and 22:38 (the registered
judge's full passes against Falcon and MobileLLM-R1.5), "CORRECTIONS (22:25 UTC)", 22:28 and 22:31 (the
neutral-title replication) are integrated, each recomputed from the files it names. Also read: the independent audit `N/audit_19_10.md` (as far as it concerns
the paper) and the coordinator's message of 22:25. Primary records opened and recomputed: `N/main16.final.log`;
`N/records/unpacked/run/` and `proxy/`; the last snapshot `N/snap_final` (its records extracted and checked
byte-identical to `records/unpacked`, the official export and `main16.final.log`); `N/heldout_v9.jsonl`;
`N/packs/official_export/x/final/`; `R/release/main16-v9/`; `N/testgen/{prefopt,trained}/out/`; every rating
file, report, key and pair set of the test split under `N/study/proxy/`; the readers' chunk files under
`N/final_eval/{prefopt,controls,k1,ref,neutral}/`; the exact reports; `S/tpu_intervals.txt`; the loop logs
`S/monitor.log`, `S/presence_A.log`, `S/presence_C.log`; `S/cpu4_up.log`. New work is in `N/paper/lead5_work/`
(scripts in `scripts/`; 32 MB). The third pass's files are kept as `lead5_work/paper_v4.md`, `CLAIMS_v4.md`,
`FIGURES_v4.md`, `figures_v4/`, `data_v4/`, `orig_fig2_throughput.py`; intermediate copies of this pass as
`paper_before_*.md`. The repository was only read (HEAD `71121cb`, committed by the operator at 22:21; nothing
edited, committed or pushed); the one command run against `N/study/proxy` that writes is the documented
re-scoring of the registered judge's reports (Falcon set at 22:19:43, MobileLLM set at 21:58 and 22:37:54).

## What changed, by section

- **Status line.** Final pass; records and scripts named; what stays [TBD] in one sentence; the audit
  corrections and the replication noted.
- **Abstract.** Rewritten around the outcome: budget used (86,054 of 86,400 TPU seconds, 62,893 in training
  steps, 1.358 × 10¹⁰ tokens), the official model (98.0 M stored, 19.6 M read), the readers (Opus 5.5 agents,
  two readings, an addition, of the builders' family, able to see which two systems a set compared, with a
  neutral-title replication that agrees), the outcome with n and intervals (MobileLLM-R1.5: 116 of 131
  decisive, 0.752 [0.688, 0.807]; Falcon-H1-Tiny: 108 of 143 against us, 0.318 [0.257, 0.385]; again against
  its K = 6 arms, 112 of 147 and 116 of 145; with one particle 22 to 122 and 99 to 16), the registered judge
  (12 to 61 and 59 to 5 over all 200 pairs of each set), the 189 prompts, one clause per caveat (63%
  degenerate MobileLLM-R1.5 replies; the constraint edge gone when Falcon gets six samples and our selector;
  one run); preference optimisation 59 to 34 (p = 0.012). Title unchanged.
- **§1.** The run to the end; contribution 4 is the outcome and its limits; "What we do not claim" rewritten.
- **§2.** §2.3 tokens and the gaps (59×, 309×); §2.4 the end (deadline 18:33:49, end recorded 18:33:54, step
  51,812; tail from 48,190; stages 18:33:55–20:02:22; TPU back 20:41:04; 86,054 s, 73.1% in steps).
- **§3, §5.** Shortened; Table 2 to the end of the 2,048-token phase (189,067, n = 226) and the tail (189,035,
  n = 73); a stale "Figure 4b" corrected to 3b.
- **§6.3–6.7.** Tokens from the ledger; §6.5 at the final checkpoint (Table 5; the line inside in minutes,
  outside in log tokens 41,850–48,342 and at equal tokens from 34,166); reading (iv) on the rise of the proxy
  rows, now as the records show it: they fell during the chat ramp (2.5430 → 2.4614) and rose from the
  assistant-only tail (step 48,190), while web fell and the chat rows stayed near 1.36. §6.6–6.7 merged and
  shortened; design and checks moved to G.4.
- **§7** (results): 7.1 protocol; 7.2 deviations (deviation 1 now also says that the Opus readers' chunk
  titles and directories named the pair set, with the two checks); 7.3 resolution, with the registered judge's
  bounds (± 0.161, ± 0.172); 7.4 the official model; 7.5 Table 7 with every reading, the neutral-title
  replication, the registered judge's full passes (agreement with the readers 58 of 62 and 55 of 56 where both
  decide), the 189-prompt column and the paired value of
  six particles; 7.6 Table 8 and what the selector-matched and K = 1 arms show (rewritten: the constraint lead
  is not the selection, does not survive matching, and model and potentials are not separated); 7.7 before
  and after preference optimisation; 7.8 "both bad" per reading and across readings; 7.9 trajectory (Table 9,
  Figure 7); 7.10 limits. Development tables moved to E.6.
- **§8, §9.** Final; the readers' knowledge of the set names and the constraint wording corrected.
- **Appendices.** C: tail, runtime 3 closed, the ledger, stages (Table C4: the sample stage's bound of 28
  pairs, 21 received, 20 kept) and preference optimisation (Table C5), end recorded 18:33:54. D.5 and E.5
  [TBD] closed; E.6 new. G.3–G.5 to 51,812 (Table G2 caption on the ramp and the tail corrected). I: the last
  day, rewritten after the audit (one loop outage of about 18–20 min, boot 20:12:56, the judge's speeds per
  range from its records, the refusal quoted in full). J: release verified, commands, `71121cb`,
  `scripts/blind_eval.py`, the chunk titles, the readers' model name, tests 1,367. K: licence. L: the audit.
  M: renamed and filled by `make_appendix_m.py` (no number typed), now with the judge's full passes, the
  neutral-title rows (M1, M11) and the named-against-neutral paired comparisons (M13).
- **Rounding.** One convention, said once (Table 7's caption, repeated in M's conventions): values rounded once
  from full precision, an exact half to even (`FACTS_2.md` CORRECTIONS item 4). Changed from this pass's
  earlier half-up: 0.753 → 0.752, 0.563 → 0.562, 0.283 → 0.282, reference first-position +0.013 → +0.012,
  three category scores in M2.
- **Figures.** Rebuilt at 21:34 and again at 22:29 with `make_all.sh --cutoff newest --blind
  final-prefopt-vs-falcon,final-prefopt-vs-mobilellm` (cut-off: step 51,800, the last step record; logs
  `lead5_work/make_all_final.log`, `make_all_final2.log`). **One figure script changed**, `fig2_throughput.py`
  (Figure C2): the runtime-3 note said "To come: training to 18:33:49 … TPU time end at 20:46:50", false on the
  final data, and a gap box hid "n = 101"; it now reads "Records to step 51,800 (18:33:30), the last step
  record. Training's deadline was 18:33:49; the TPU was given back at 20:41:04." (at 21:41 it said "Training
  stopped at its deadline", changed at 22:28 after correction 6), and the gap notes sit above the axes; checked
  on the PNG; original kept. Figure 7 now prints what Table 7 prints (0.752). Captions synced into `FIGURES.md`.
- **Length.** Main text 10,274 words by the third draft's own convention (whitespace tokens, table rules and
  pipes not counted; that count gives the third draft 9,570, which it reported as 9,536), 1,263 of them in
  table rows; 9,929 tokens with a letter or digit. After moving detail to the appendices the text stood at
  9,929 by the first count (22:20); the audit corrections and the replication added about 350 words, so it is
  7% above the third draft. `lead5_work/scripts/wordcount.py` prints both counts.
- **CLAIMS.md.** C500–C563 (place, claim, value, source, how to recompute) and "Withdrawn or changed in the
  final pass", which also lists what this pass itself changed after 22:20.

## Every [TBD] that remains, and why

1. **MobileLLM-R1.5's two K = 6 control arms** (§7.2 deviation 6, M1): in generation since 19:17 UTC (40 of 200
   prompts by 21:13; from 22:36 on three CPU runtimes and this container); after midnight by the operator's
   estimate (`FACTS_2.md` 22:38). To add them: pair sets, Opus readings, `final_results.py`,
   `paper_values.py`, `make_appendix_m.py --splice`, Table 7, Table 8 and §7.5–7.6.
2. **The author list** (at submission).

Closed in this pass: everything else, including the Falcon control arms, the K = 1 arm, the registered
judge's full passes (Falcon 22:11, MobileLLM-R1.5 22:35) and the neutral-title replication of both official
sets (22:28, 22:31).

## What in FACTS_2.md was wrong when recomputed

1. **(21:00) "the constraint score … is the selector's doing".** Unsupported: our K = 1 arm, with nothing to
   select, meets every constraint on 37 of 48 prompts (0.771) against 38 with six particles; the selector over
   six Falcon samples gives Falcon 42 (0.875). (CORRECTIONS item 2 now says the same. This pass had at first
   written "the lead is the decoder's (mostly its potentials)", which is not separated on `test` either; it is
   withdrawn, C537.)
2. **(20:32) "the total rose after 46,178 (2.0914 → 2.1422)".** It fell to 2.0840 at step 47,260 and rose from
   the assistant-only tail (step 48,190; user and system tokens 1,833 of 14,924 weight units at step 48,150,
   none from 48,200). CORRECTIONS item 1 agrees.
3. **(20:32) "UTC = 18:33:49 at 1499.01".** 1,499.01 minutes after t0 (17:34:54.713) is 18:33:55; the end event
   and final checkpoint are at 18:33:54; 18:33:49 is the planned deadline. CORRECTIONS item 6 agrees.
4. **(20:32) the official test generation "19:31:12 to 20:06:32, 2,110 s"**: its summary starts 19:31:22
   (2,109.8 s).
5. **Rounding.** Under the convention of 22:25 (exact halves to even), 0.752, 0.562 and 0.282 are right and
   only 0.307 is wrong (0.3075 → 0.308; item 4). Double rounding of 4-decimal values remains wrong in four
   intervals: selector-matched score [0.225, 0.348] → [0.225, 0.349] (0.34854; item 4) and its decisive rate
   [0.143, 0.273] → [0.143, 0.272] (0.27247); K = 1 against Falcon, decisive [0.103, 0.221] → [0.103, 0.220]
   (0.22049); K = 1 against MobileLLM-R1.5, decisive [0.786, 0.912] → [0.786, 0.913] (0.91251). The last three
   are not in the CORRECTIONS list.
6. **(21:27) the registered judge's speeds** "137.6 s with 8 threads (pairs 137 to 141)" and "resuming from
   142": the records give 144.4 s on pairs 137–142 (item 10 agrees). With the full pass, the restarted Falcon
   stream (pairs 143–200) averaged 49.9 s per pair, not the 39.9 s of its first 13 pairs.
7. **(21:40) K = 1 degenerate 6.75%**: 17 of 252 = 6.7% (item 4 agrees).

Not checkable from a record, used as the operator's account: the cause of the two worker restarts and the
agent's inactivity of about 23 and 41 minutes (CORRECTIONS item 9; only the 20:12:56 boot is on record), the
K = 1 generation's "1,356 s on 8 workers", the readers' model name `claude-opus-5-5` (item 13). The operator's
notes (`N/OPERATIONS.md`, not FACTS_2) give "18:38:56 prefopt started … run deadline 20:33:37"; the stage
ledger has 18:37:07 and 20:31:49, which the paper uses. Everything else in the sections of 21:27, 21:40, 22:13,
22:25, 22:28, 22:31 and 22:38 was reproduced exactly from the files they name (the MobileLLM-R1.5 judge's
"3 pairs judged twice" is taken from the section, not checked).

## Checks made in this pass

- Every Opus report of the test split (official against Falcon, MobileLLM and the trained weights; both Falcon
  control sets; both K = 1 sets; the reference set; both neutral-title sets) and both registered-judge reports
  equal the summary recomputed from the ratings with the key check, in every field; each first reading's chunk
  files give exactly the `order1` / `order2` records of `judge-opus.jsonl`; the neutral chunk files differ from
  the first readings' only in their first line (checked on three chunks of each of three directories).
- Every exact report on test equals `lead4_work/scripts/exact_split.py`'s recomputation in maths, constraints,
  degenerate replies and words (7 runs; MobileLLM-R1.5's neutral arm has no report); the run's own `dev` exact
  reports equal `exact_dev_final.py`'s.
- The CPU evaluation of step 51,812 reproduces the runtime's record (web −4.0 × 10⁻⁸, chat +4.7 × 10⁻⁸).
- The release: 12 parts match `parts.json` and the fetched parts; whole-tar SHA-256 `f8278440…`; the unpacked
  copy's 21 files match `SHA256SUMS`; `weights16.npz` SHA-256 `2da7cbd2…` recomputed.
- The figures' train log (from `N/snap_main`, 18:44) has the same SHA-256 as `N/snap_final`'s (`c4c8d3e8…`).
