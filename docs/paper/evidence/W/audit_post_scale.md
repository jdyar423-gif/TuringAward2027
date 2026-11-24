# audit_post_scale: the post-training stages at real scale and in real time

Scope: `scripts/run_24h.py` stages sample, critic, prefopt, select, judge, export and their slices
in `bes/train/stages.py`, for the live run (`run.post_minutes=critic=0,buffer=20`,
`run.post_reserve_hours=3.6667`: sample 3000 s, critic 0 (skipped), prefopt 3600 s, select 2700 s,
judge 1800 s, export 900 s, buffer 1200 s; `train_end` = deadline - 13 200 s, about 16:52 UTC).
Line numbers are those of HEAD 566517a. In b05bfe7, the branch head now, `scripts/run_24h.py`
lines from 189 on are shifted by +32.

Nothing was changed in `/home/user/turingaward2027`. The experiments and the patched checkout are
in `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/audit_post_scale/`.

## Verdict

* **No blocker found.** None of the stages throws on the counts it will meet. The prompt filter
  and the labels ran over 24 000 local SmolTalk conversations without an exception. Zero-count
  paths (no common prompt, no opponent, no pair, no time) are guarded. The weights that select,
  judge and export use are the right ones: post-DPO when accepted, otherwise the trainer's final
  checkpoint. The DPO update keeps every one of the 59 float32 leaves (eval_shape).
* **Three majors** cost the run real value:
  1. The decode selection decides on about 18 common dev prompts. Its rule is "highest mean, ties
     keep the default". With every variant equally good, that rule ships a non-default decoder
     (k1 = 1 particle, or no band) in **54 %** of simulated runs. Two of its four variants
     re-decode the default's particles: nolib on 100 % of the prompts it reaches, because the
     gate fires on 0 of the first 18 dev prompts.
  2. `allocate` never passes on the time a stage leaves unused. Prefopt will finish 600 to
     1 800 s early, the judge and the export early too. The run ends 40 to 60 minutes before its
     deadline while select and judge are starved.
  3. The sample stage gets through **about 110 to 150 of its 4 000 prompts** (3 %). That gives
     roughly 50 to 200 sampled pairs, at most 1 % of the DPO data, which is UltraFeedback.
     Batching prompts per device call would not help, because the cost is host Python per
     particle.
* Seven minor findings follow, each with a small code-only diff. The combined patch is tested and
  applies to the branch head: see the "Patch" section.

## Rates used (and what was measured here)

| quantity | value | source |
|---|---|---|
| OurLM.step, 6 rows / 24 rows (TPU) | 12.1 / 34.5 ms | `docs/tpu_profile/tpu_tune.v6e.txt` |
| smc.decode, 6 particles, untrained weights (TPU) | 43.1 ms per position, of which ~31 ms decoder host | tpu_tune |
| decoder host per step, this CPU, **flat rows** (untrained-like) | 9.8 ms (+2.8 ms features) | `exp_flat.py` |
| decoder host per step, this CPU, **peaked rows** (trained-like) | 2.0 to 4.4 ms (+0.8 to 1.5 ms features) | `exp_decode.py` |
| TPU-host / this-CPU slowdown for this Python | ~3.2x (31 / 9.8) | ratio of the two flat-row numbers |
| **estimate: one synchronous step, 6 particles, trained model, TPU** | **~19 to 26 ms** | 12.1 + 3.2 x (2.0..4.4) |
| one step, 1 particle | ~10 ms | 4.6 + 1.24 per row (tpu_tune fit) + host |
| synchronous steps per sampled prompt (sentence resampling) | 656 to 1 672, mean 1 033 = 1.42 x the longest particle | stand-in decode of 8 real prompts |
| exact features of one DPO batch (128 rows x 1 024) on one core | 0.42 s (~300 rows/s), regardless of row content | `exp_features.py`, `exp_features_long.py` |
| DPO reference pass / step (TPU) | 739 rows/s / 0.65 s per 64 pairs (cached batches only) | tpu_tune |
| prompt filter (`_prompt_item`) | 4.5 ms per conversation, 0 exceptions in 24 000 | `exp_prompt_items.py` |
| emergency export on the TPU | 5.8 to 9.1 s | live log (`runlog_now.txt`) |

The stand-in (`stub_lm.py`) does OurLM's host work exactly: an IncrementalFeatures provider per
particle, forks on reorder, and the same concatenations. The device call is replaced by a bigram
model of real SmolTalk replies, END included, so that sentence boundaries, END and peaked rows
look like text. Time spent inside the stand-in is subtracted. The 43 ms of tpu_tune overstates a
trained model, because flat rows send top-p to the full sort and keep all six particles running.

## Stage by stage

### sample (3 000 s)

* It wants 4 000 prompts: 12 chat and math sources, 333 each, round-robin. Building the pool is
  cheap (about 18 s of prompt filter for 4 000, plus the shard and parquet reads).
* Cost per prompt is about 1 000 synchronous steps x 19 to 26 ms, plus prefill (up to 20 chunk
  calls, and about 1 s of prompt features on the host for a 1 280-token prompt). That is
  **20 to 28 s per prompt, so about 110 to 150 prompts in the slice (3 % of 4 000)**. The plan
  (`docs/PLAN.md` timeline) expected "about 40 000" prompts with batched particles.
* **Batching prompts would not pay.** OurLM.step costs 34.5 ms for 24 rows against 12.1 ms for 6,
  i.e. 1.24 ms per extra row, which is mostly host features. The decoder's host work (tracker,
  n-gram blocker, top-p, draws) is per particle and is not shared across prompts. Four prompts
  per device call would give at most 15 to 20 % more prompts, for a multi-prompt SMC driver.
* **Pairs.** A pair needs two different scores. For chat prompts, scores differ only through
  degenerate particles; for math, through wrong answers against the correct reference; for
  constraint prompts, through violations. The new rule (a reply that opens with a stock refusal
  scores 0; a prompt whose reference reply is a refusal is skipped) adds pairs whenever some
  particles refuse and others do not. In the stand-in decode of 8 real prompts the stage
  labelled 3 pairs. 15 of its 48 particles ended at the 768-token budget and **were all labelled
  as complete** (finding 4); with them counted as cut, the same 8 prompts give about 10 pairs.
  **Expected: 0.4 to 1.3 pairs per prompt, about 50 to 200 sampled pairs.** That is below 200,
  which triggers finding 5.
* Rows that do not fit: DPO rows hold at most 1 025 tokens, while prompts may have 1 280 tokens and
  replies 768. A pair with a long rejected reply needs a prompt of at most 255 tokens. Measured
  prompt lengths per source: summarize 29 % over 511 tokens, rewrite 31 % over 255, longalign
  100 % at the 1 280 cap. The CPU smoke run dropped 5 of its 18 sampled pairs as `too_long`.
* It never idles. It stops between prompts at the end of its slice (an overrun of at most one
  prompt, about 30 s).

### critic (0 min)

Skipped: "planned with 0 minutes". Nothing downstream needs it; the decoder has no critic path.

### prefopt (3 600 s)

* **Data.** The sampled pairs are about 50 to 200. UltraFeedback (`tok/ultrafeedback`, 61 135
  rows, ties dropped, length guard 3.0, rows at most 1 025 tokens) gives an estimated 40 000 to
  50 000. This is an assumption: the download plan's phase 2 and the tokeniser include it, but I
  could not list the TPU disk. Please check `data_cache/tok/ultrafeedback/manifest*` there.
  Without UltraFeedback, prefopt has 2 to 3 DPO steps (64 pairs each) and is a no-op (see below).
* **Held-out split (finding 5).** Because fewer than 200 pairs were sampled, 25 % of all pairs,
  UltraFeedback included, are held out. That leaves about 33 800 training pairs instead of 42 700,
  i.e. 528 instead of 668 steps per epoch per learning rate.
* **Time.** The `n_max` formula (`0.6 * left / per_pair`) counts device time only, and it does not
  trim here (56 000 > 45 000). Real time, with `h` the host slowdown:
  * The reference pass (90 000 rows) is host-bound at about 300 rows/s per core: 300 s at h = 1,
    750 s at h = 2.5.
  * Two epochs: 2 x 528 x max(0.65 s device, 0.42 x h s host), i.e. 690 to 1 110 s.
  * Held-out scoring, twice: 150 to 375 s.
  * Health sampling: the start plus 2 candidates, about 190 to 300 s each (64 prompts, 1 particle,
    up to 768 tokens).
  * Loading, compiles and the rest: about 100 to 150 s.
  * **Total about 1 900 to 3 300 s.** It fits the slice and leaves 300 to 1 700 s unused, which is
    lost today (finding 2).
  * There is no deadline inside `run_prefopt`. If UltraFeedback holds more pairs or the host is
    slower, an overrun is absorbed by the 1 200-s buffer.
* **Features.** The 1 GB feature cache is all-or-nothing (38 batches = 2 432 pairs), so every
  pass recomputes the exact features of every row on one core (finding 6).
* **Acceptance with plenty of data.** There are 2 000 to 11 000 held-out pairs, so the standard
  error of the accuracy is about 0.005. With 528 to 668 steps at 1e-6 or 3e-6 one candidate will
  very likely clear it, and the loss and length checks on 64 paired prompts are lenient. A pass is
  likely.
* **Acceptance with little data.** This only happens if UltraFeedback is missing. The margins
  after 2 steps are tiny but systematic: they move together with the length difference. When all
  held-out margins share a sign, the standard error collapses to 0, so any accuracy above 0.5
  passes. The CPU smoke run shows exactly this (`runs/smoke_b`): accepted after 1 step on 8 pairs,
  held-out accuracy 1.000 on 5 pairs. The accepted weights are practically the trained ones, so
  this is harmless; it is not listed as a finding.

### select (2 700 s)

* There are four variants, each with (2 700 - 1 080) / 4 = 405 s. At about 22 s per dev prompt
  (6 particles, 1.24 generations per prompt) default, nolib and noband reach 18 to 19 prompts. k1
  (about 4 s per generation) reaches about 80, of which only the 18 common ones count.
  Reproduced with the real `stage_select` on a fake clock (`repro_select.py`):
  `{'default': 19, 'k1': 80, 'nolib': 18, 'noband': 18} | common: 18`.
* **Redundant decodes.**
  * nolib differs from default only where the library gate fires. It fires on **6 of 200 dev
    prompts and on 0 of the first 18** (measured with `lib_v2`). So nolib's 405 s reproduce the
    default's replies, and the dev "no loss with the library" check compares identical replies,
    which makes it vacuous.
  * noband re-decodes the default's particles: same seed and settings, only the band in the
    selection differs.
* **Rule.** "Highest mean, ties keep the default", on 18 prompts. In a null simulation
  (`repro_select.py`, 400 trials, all variants equally good, noband identical to default on 80 %
  of prompts) it ships k1 in 37.5 % and noband in 16 % of trials: **54 % non-default**
  (finding 1).
* **Factual library measurement**, 1 068 s. Locally the gate fires on **115 of the 500
  questions** (94 head-entity probes and 21 of the 400 QA questions), and those come first. At
  about 9 to 10 s per question for both arms, it answers essentially all the fired ones. The
  library decision therefore has about 100 informative pairs, mostly probes: enough power if the
  library really helps. Questions after the fired ones give identical replies in both arms
  (finding 9).
* Guards: an empty common set keeps the default; a failed or untimed factual measurement means
  not shipped.

### judge (1 800 s)

* Our test replies get 0.45 x 1 800 = 810 s, about 30 to 37 of the 200 prompts. Each of the 3
  opponents then gets (deadline - now) / remaining time. The judge is Qwen3-4B Q4_K_M on 42
  threads; its speed on the TPU host was never measured.
* Fake-clock run of the real method (`repro_judge.py`):

  | judge time per pair | judged per opponent | unused slice |
  |---|---|---|
  | 5 s | 30 | 536 s |
  | 9 s | 30 | 176 s |
  | 15 s | 22 | 0 s |

  When time runs out, `run_judge` stops starting pairs. The results are written with
  `partial: true` and the stage ends normally. A judge prompt is at most about 4 500 tokens, below
  n_ctx 8 192, so it never overflows.
* The rounds fix (finding 8) gives 40 / 33 / 22 judged per opponent for the same three speeds.

### export (900 s)

`export_run` takes seconds: 5.8 to 9.1 s for the live run's emergency exports. The slice is
reserve, not work. The weights exported are post-DPO when `manifest.prefopt.accepted`, else the
trainer's final checkpoint (`latest_params`), and they match what select and judge evaluated. The
library is left out only on an explicit `decode.library.ship == False` (finding 10).

### Overall timeline

Suppose sample uses its 3 000 s, prefopt 2 400 s, select 2 700 s, judge 1 350 s and export 90 s.
The run then ends **61 minutes before the deadline**, of which only 20 are the planned buffer
(`repro_allocate.py`). With finding 2's fix the same run gives select 3 420 s and judge 2 280 s.

## Findings, ranked by cost to the run

### 1. major: the decode selection is mostly noise, and two of its four variants re-decode the default

`scripts/run_24h.py:1102` (`best = max(cands, ...)`), `:1073` / `:1086` (nolib on every prompt).

* **What happens:**
  1. Four variants get 405 s each, so default, nolib and noband reach about 18 dev prompts and k1
     about 80.
  2. Only the 18 common prompts are scored.
  3. `best` is the highest mean, with ties going to the default. With equal variants, k1 or
     noband win 54 % of the time.
  4. The manifest `decode.overrides` then ships `extra.particles=1` or `extra.band=false` in the
     export, without any evidence.
  5. nolib spends its 405 s re-creating the default's replies. The gate fires on none of the
     common prompts, which also makes the dev no-loss part of the library rule vacuous.
* **Reproduced** with the real `Run.stage_select` on a fake clock and harness:
  `repro_select.py`, result `non-default shipped in 54%`. With the patch: 1 %; nolib answers only
  the 6 fired prompts; noband gets the freed time.
* **Fix:**
  1. A variant replaces the default only with a significant paired gain:
     `bes.eval.proxy.score.paired_sign_test`, more prompts better than worse, p < `FACTUAL_ALPHA`
     (the library's own rule).
  2. nolib is decoded only on the prompts where the gate fires, and takes the default's score
     elsewhere, because the replies there are identical (same prompt, same seed).
  3. This tightens the documented rule ("highest mean exact score"), so it is the owner's call.
  4. An option not in the diff: also put the 6 fired prompts first, so that the dev no-loss check
     can see the library. That would make the check real but noisy (6 prompts), and it is a
     protocol choice.

### 2. major: time an early stage leaves is never used

`bes/train/stages.py:235` (`allocate` gives every post stage at most `planned_s`).

* **What happens:**
  1. The slices are fixed shares.
  2. When prefopt finishes 1 200 s early, select still gets 2 700 s, and `available - needed`
     (1 200 s) stays unallocated.
  3. The same holds for the judge and the export.
  4. **Reproduced:** `repro_allocate.py`, real `stage_specs` and `allocate` with the live
     overrides. Result: `run done ... 3660 s (61.0 min) before the deadline, of which the buffer
     is 1200 s`.
* **Fix:** when the run is ahead (`available > needed`, nothing shed), share the spare time among
  the pending elastic stages (sample, select, judge) in proportion to their planned slices. Export
  and buffer keep theirs, and late runs are unchanged. With the fix, select gets 3 420 s and judge
  2 280 s. `tests/train/test_stages.py` passes (27).

### 3. major: the sample stage processes about 3 % of its pool, and its pairs are at most 1 % of the DPO data

`scripts/run_24h.py:783-841` (one prompt per `smc.decode`, 6 particles, 768 tokens).

* **What happens:**
  1. Each prompt costs about 1 000 synchronous steps x 19 to 26 ms, i.e. 20 to 28 s.
  2. That gives about 110 to 150 prompts in 3 000 s.
  3. At 0.4 to 1.3 pairs per prompt, that is 50 to 200 sampled pairs, next to about 40 000
     UltraFeedback pairs in the epoch.
* **Not reproduced on the chip.** This is an estimate: the stage's own code path was run with the
  stand-in device (`exp_decode.py`, steps per prompt measured) and scaled with the TPU rates.
* **Fix:** no small change recovers 30x. Batching prompts per device call gains at most 15 to 20 %
  (see the sample section).
  * The cheap gains are to label budget-cut replies (finding 4), which raised the stand-in's pair
    yield from 3 to about 10 per 8 prompts, and to skip prompts no DPO row can hold (finding 7).
  * With finding 2 in place, the owner may also stop sampling early and pass the time to select
    and judge. That is a plan choice, so no diff is given.
  * Verify on the chip: the stage logs `sample: N new prompts ..., about P pairs, R prompts/min`.

### 4. minor: replies cut by the budget are labelled as complete

`scripts/run_24h.py:822` (`_labelled_part(..., p.finish, ...)`).

* **What happens:**
  1. The budget-aware potentials force END into slot 768, so a cut particle finishes with
     `"stop"`.
  2. `label_sample(..., "stop")` therefore does not flag it as truncated.
  3. So it scores like a complete reply, and pairs against cut replies are lost.
  4. `Responder.reply` reports the same particle as `finish_reason="length"`
     (`at_budget`, respond.py step 8), which is how select and judge score it.
* **Reproduced:** `repro_at_budget.py`. Six particles of 768 tokens end mid-word
  (`'...the average househ'`). The stage labels them score 1.0 and not degenerate; with the
  evaluation's finish they score 0.0 and are degenerate.
* **Fix:** pass `"length"` when `p.finish == "stop" and len(p.tokens) >= d.max_new_tokens`. The test
  `test_run_24h_sampling_labels_every_response_on_the_text_of_its_tokens` used a fake particle of
  71 tokens under a 64-token budget, which the decoder cannot produce. It is adjusted to a fitting
  budget and gains an at-budget case.

### 5. minor: a quarter of UltraFeedback is held out because few pairs were sampled

`scripts/run_24h.py:972` (`heldout_frac=... if len(sampled) >= 200 else 0.25`).

* **What happens:**
  1. With fewer than 200 sampled pairs, 25 % of all pairs are held out by prompt hash.
  2. UltraFeedback's share goes with them: about 9 000 to 11 000 pairs leave the epoch.
  3. The held-out set is scored three times on the host-bound path.
* **Reproduced:** `repro_heldout.py`. With 150 sampled and 40 000 UltraFeedback pairs: train
  27 657, held out 9 232 (9 197 of them UltraFeedback), 433 steps per epoch. With 250 sampled:
  35 044 / 1 855, 548 steps.
* **Fix:** decide on the total number of pairs.

### 6. minor: the 1 GB feature cache never holds the UltraFeedback set, so features are recomputed in every pass

`scripts/run_24h.py:958` (`cache_gb=min(PrefCfg.cache_gb, 1.0)`) with `PairBatches`' all-or-nothing
cache (`bes/data/prefs.py:503`).

* **What happens:**
  1. One batch is 28 MB (128 rows x 219 140 B), so 1 GB holds 38 batches = 2 432 pairs.
  2. The cache is therefore off.
  3. Every pass computes `exact_features` again, on one core at about 300 rows/s.
  4. Those passes are the reference pass, the epochs, and each candidate's held-out scoring.
  5. The epochs become host-bound as soon as the TPU host core is slower than 0.65 s per batch,
     about 1.5x this container.
* **Reproduced** as a measurement: `exp_features.py` (rows/s, `cache on ... False` for a
  UltraFeedback-sized set).
* **Fix:** size the cache from `MemAvailable` (20 %, at most 32 GB). The TPU host has 185 GB and
  the set needs about 16 to 20 GB. The values do not change. It saves 0 to about 15 minutes,
  depending on the host's core speed, and only matters together with finding 2.

### 7. minor: prompts that leave no room in a DPO row are still sampled

`scripts/run_24h.py:805-808`.

* **What happens:**
  1. The prompt may have up to 1 280 tokens, but a DPO row holds prompt + reply + END in 1 025.
  2. A prompt over about 960 tokens can only use replies under 64 tokens.
  3. In practice all its pairs are dropped as `too_long`. About 2 % of the pool falls under this
     (longalign-type prompts), at about 25 s of TPU time each.
  4. The smoke run dropped 5 of its 18 pairs as `too_long`.
* **Partly reproduced:** prompt lengths measured (`exp_lengths.py`, `exp_prompt_items.py`).
* **Fix:** skip such prompts.

### 8. minor: the judged comparison idles when the judge is faster than the generation

`scripts/run_24h.py:1240`.

* **What happens:** there is one round of 45 % generation, then judging. With a judge at 5 to 9 s
  per pair, 176 to 536 s of the slice stay unused.
* **Reproduced:** `repro_judge.py` with the real method on a fake clock.
* **Fix:** repeat generation (45 % of what is left) and judging until the slice is used. The
  harness cache resumes the generation and `make_pairs` grows the pair sets without touching
  published pairs or verdicts. Result: 40 / 33 / 22 pairs judged per opponent, against 30 / 30 /
  22 today.

### 9. minor: the factual measurement keeps answering questions that cannot tip its test

`scripts/run_24h.py:1203`.

* **What happens:**
  1. After the 115 fired questions it answers the other 385 with both arms until the deadline.
  2. Those replies are identical in both arms (no passage, same seed), so they count as "same".
  3. Today the fired ones nearly fill the 1 068 s. With finding 2 (a larger slice) this becomes
     pure waste.
* **Partly reproduced:** the fired count was measured on the local `lib_v2`.
* **Fix:** keep the fired questions plus 10 others, which preserves the report's
  identical-replies check, so that select can end early and finding 2 hands the rest to the judge.
  The library-decision test still passes.

### 10. minor: the export ships the library when no decode decision exists

`scripts/run_24h.py:1299` and `bes/export.py:1745` (only an explicit `ship is False` drops it).

* **What happens:**
  1. If select fails, or is shed (more than about 130 minutes late), there is no
     `manifest.decode`.
  2. The library is then shipped and counted, against the rule "no measurement, no gain: not
     shipped".
  3. The same applies to every emergency export.
* **Not reproduced:** this is code reading.
* **Fix:** in `stage_export`, pass an explicit `{"ship": False, "reason": ...}` when the run has a
  library but no decision. `bes/export.py` and its tests stay unchanged.

## Checked and fine

* The `critic=0` override skips the critic with a reason; the `post_minutes` validation is fine.
* `allocate` on time sheds nothing. A training stop of up to one step early, or the final save
  running about 30 s late, is absorbed by the buffer.
* The sample stage's prompt filter and labels: no exception on 24 000 real conversations. Pool
  build takes about 18 s for 4 000 prompts.
* The sample stage stops between prompts at its deadline, and records persist with fsync before
  each next prompt. A restart resumes without repeating prompts (`done` ids).
* Raw top-up files for sampling: `prune_raw` keeps the newest file of each chat and math messages
  source, and `completed_files` lists only files that are present.
* One compiled decoder per (K, tmax) is shared by all OurLM instances (`G.jitted` lru_cache). The
  sample stage uses tmax 5 248; select and judge use 2 944. The KV cache of 6 rows at 5 248 is
  1.55 GB (eval_shape).
* UltraFeedback pair shards carry `pairs: True` from `tokenize_worker`, and the pipeline test
  passes (`tests/data/test_prefs.py -k pairs_from_shards`).
* DPO: ppm 8 x n_micro 8 = 64 pairs per step at `micro_seqs=16`. `optimizer.update` keeps the
  dtype and shape of all 59 float32 leaves, so the post-DPO checkpoint loads with
  `load_checkpoint`'s tree check.
* Post-DPO weights are saved only when accepted. `latest_params` uses them for select, judge and
  export, never for sample.
* Library loading is instant and retrieval takes 2.6 ms per query.
* `_library_decision` guards: no time, an error, or no library all mean not shipped.
* `make_ab.make_pairs` with partial runs pairs the common prompts and raises only when none are
  common, which is caught. A judge prompt is at most about 4 500 tokens, below n_ctx 8 192.
* The judge-stage generator and pair-set names are stable across a resume (same run_id, same
  pair-set base).
* Export: seconds, not minutes; `export_run` takes the post-DPO checkpoint and labels its source.

## Patch

`/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/audit_post_scale/proposed_fixes.diff`
is the combined diff of findings 1, 2 and 4 to 10, against 566517a. It touches `bes/train/stages.py`,
`scripts/run_24h.py` and one test.

* `git apply --check` passes on the current branch head b05bfe7, with offsets.
* In the patched checkout `.../audit_post_scale/checkout`, these pass:
  `tests/train/test_stages.py` (27) and the run_24h tests of `tests/train/test_supervisor.py`
  (41 selected: labels, stale labels, library decision, health samples, helpers, preflight,
  journal shards).
* Not run: `test_cpu_rehearsal_with_kill_and_resume`, the full smoke run, which takes more than
  9 minutes and needs data_cache. Run it before deploying.
* Nothing in it touches the parameter tree, the optimiser state, a configuration field or
  `bes/config.py`, and no computed value of training changes.

**Deployment:** the running supervisor imported `scripts/run_24h.py` and `bes/train/stages.py`
when it started. The patch therefore takes effect only after a supervisor restart (stop, then
`--resume`), for example before `train_end`, or between stages. The order of value is:

1. finding 1, sign-test rule (2 small hunks);
2. finding 2;
3. finding 5;
4. finding 4;
5. the rest.

## Experiment files

All in `.../wf/audit_post_scale/`:

| file | what it measures or reproduces |
|---|---|
| `stub_lm.py` | the host-only stand-in device used by the decode experiments |
| `exp_decode.py`, `exp_decode_f.py` | the sample stage per prompt |
| `exp_flat.py` | tpu_tune reconciliation (untrained-like rows) |
| `exp_setup.py` | one-off costs inside a decode call |
| `exp_select.py` | Responder steps per reply |
| `exp_features.py`, `exp_features_long.py` | DPO batch host cost |
| `exp_prompt_items.py`, `exp_lengths.py` | prompt filter robustness, prompt lengths |
| `repro_at_budget.py` | finding 4 |
| `repro_allocate.py`, `repro_allocate_patched.py` | finding 2 |
| `repro_heldout.py` | finding 5 |
| `repro_select.py`, `repro_select_patched.py` | finding 1 |
| `repro_judge.py CHECKOUT` | finding 8 |
| `proxy_root/factual/` | the factual question file built locally (fired count) |
