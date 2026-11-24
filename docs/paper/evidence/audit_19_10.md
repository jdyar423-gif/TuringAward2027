# Audit of `docs/CHANGES.md` §19.10 and the README "24-hour run" sections (commit 34f12ac)

N = `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`
S = `/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`

**Method.**
- **Blind comparisons.** I recomputed all 8 sets from the raw r1/r2 verdicts, the keys and the pair files with my own script (`N/audit_work/recompute_ab.py`, exact fractions, nothing imported from `bes`). The ratings files match the raw readings on 1,600 of 1,600 pairs. The second reading has A/B swapped on 200 of 200 pairs of every set. `bes.eval.proxy score` regenerated all 8 reports byte-identical, and `compare` reproduced the paired tests.
- **Exact scores.** `exact … --split test` regenerated identical summaries for all 7 runs. I recounted the numerators from `per_prompt`.
- **Release.** The SHA-256 of all 12 committed parts and of the whole tar match `parts.json`. I hashed `weights16.npz` from inside the tar and re-ran `scripts/export.py --verify`.
- **Run records.** I read `stages.json`, `stage_table.json`, `train/ledger.json`, `train/train_log.jsonl`, `main16.final.log`, `heldout_v9.jsonl`, `prefopt_report.json`, the model card, the judge records, the progress logs and the loop logs.
- **Rounding** is checked against the exact value, rounded once.

## 1. Discrepancies (27: wrong 8, imprecise 13, unsupported 6)

| # | Location | Text says | Primary record says | Record | Severity |
|---|---|---|---|---|---|
| 1 | CHANGES §19.10, paragraph after the held-out table ("要照實說的一點") | "p = 0.67 之後對話資料的比例從 0.25 升到 0.70，而對話列裡使用者輪不計損失，模型對使用者輪文字的預測因此變差" | User turns had loss weight 1 until the assistant-only "tail" stage started at **step 48,190 (p = 0.9200)**. The chat share rises linearly from 0.25 at p = 0.67 to 0.70 at p = 0.92 (reached at step 48,200). During that ramp the proxy loss **fell** (2.5430 at 36,422 → 2.4614 at 46,178). The rise starts at the assistant-only switch: 2.4683 (47,260) → 2.6651 (48,342) → 3.0764 (49,424). The text states a rule that held only for the last 8% of training and never mentions the switch that matches the timing. | `N/records/unpacked/run/train/train_log.jsonl` (event `stage`, step 48190, `phase: tail`, `assistant_only: true`; `chat_frac` per step); `bes/config.py:205-222` (`assistant_only_from = 0.92`); `bes/data/samples.py:30-38`; `N/heldout_v9.jsonl` | wrong |
| 2 | same paragraph | "網頁與對話則繼續下降" | Web falls monotonically. Chat (the 對話 column) does not: 1.3790 (46,178) → 1.3630 → 1.3612 (48,342) → **1.3671 → 1.3684** → 1.3638 → **1.3668**. It rises in two of the four rows of the text's own table and ends above its 48,342 value. Only the assistant-turn loss kept falling (1.2651 → 1.1981, then 1.1983). | `N/heldout_v9.jsonl` | wrong |
| 3 | README, CN ("完整數字（對照組、K = 1、精確計分、…）在 `docs/CHANGES.md` 第 19.10 節與 `docs/paper/paper.md`") and EN ("Every number, the control arms, exact scores, …: … and `docs/paper/paper.md`") | paper.md holds the full numbers | `docs/paper/paper.md` at HEAD is the "third pass (before the final results)" (commit 11aa314, 18:40 UTC). Every test-split cell is a `[TBD]` shell (Appendix M, Table M1 ff.). It also says "The protocol's K = 1 result and K-matched and selector-matched controls have no shell, because they are not run (deviation 6)", which contradicts §19.10. | `docs/paper/paper.md` lines 1895-1940; `git log -- docs/paper/paper.md` | wrong |
| 4 | README EN | "MobileLLM-R1.5's replies are degenerate on 62.7% of these **prompts**" | 62.7% = 158 of 252 **replies**. Prompts with at least one degenerate reply: 139/200 = **69.5%**. The CN README and §19.10 correctly say 回覆 (replies). | `N/study/proxy/reports/exact/mobilellm-f680f0533368-qwen3__recommended-d9f149b0d3-s0-test.json` (`per_prompt`) | wrong |
| 5 | CHANGES §19.10, 事先註冊的自動評審 | "（MobileLLM 的回覆多數超過評審的 4,000 字元上限）" | 94 of 252 MobileLLM replies (**37.3%**) are longer than `MAX_REPLY_CHARS = 4000`. 88 of 200 pairs contain an over-limit reply. The judge's own records flag 38 of 92 pairs as clipped (39 of 96 at 21:40); among the 23 pairs judged on CPU runtimes, 7 are clipped. Not a majority by any count. | `N/study/proxy/pairs/final-prefopt-vs-mobilellm/pairs.jsonl` + `keys/…json`; `N/study/proxy/ratings/final-prefopt-vs-mobilellm/judge-qwen3-4b-2507-3605803b.jsonl`; `bes/eval/proxy/judge.py:53` | wrong |
| 6 | CHANGES §19.10, 收尾時的事故（一） | "19:08 與 19:32 兩次 … 本機的監看、presence 與預算守衛迴圈因此停了 22 與 41 分鐘" | The loop logs show **one ≈18-20 min outage**: `monitor.log` 19:13:12 → 19:30:55, `presence_A` 19:10:57 → 19:31:15, `presence_C` 19:14:31 → 19:34:21. The loops then ran **from 19:31 to ≈20:05**: monitor entries every ≈3 min until 20:03:48 (it ended because the run had completed); presence_A at 19:34, 19:44, 19:48, 19:58, 20:02, 20:05; presence_C at 19:37-20:05:58. The second gap is ≈8-11 min (presence_A 20:05:36 → 20:13:41; presence_C 20:05:58 → 20:16:45), which fits the container boot at 20:12:56 UTC. There was no 41-minute stop, and the first stop began at ≈19:11-19:14, not 19:08. | `S/monitor.log`, `S/presence_A.log`, `S/presence_C.log`, `/proc/stat` btime | wrong |
| 7 | CHANGES §19.10, 事先註冊的自動評審 | "MobileLLM 那組改由四路同時判不同的區段" | From 21:36, **three** streams judged the MobileLLM set: the testq3 stream on bes-cpu2 (cores 4-7) and two shards, [146,173) and [173,200), on bes-cpu3. The fourth stream (testq3, cores 0-3) was judging the **Falcon** set; its records grew from 155 to 164 by 21:40. | `N/judge_remote/testq3/progress.log`; `N/judge_remote/testq4/ranges.txt`, `shard_*.log`; `N/peek/q_final-prefopt-vs-falcon.jsonl` | wrong |
| 8 | CHANGES §19.10, 偏離註冊協定之處（一） | "註冊的自動評審只在兩組正式比較上跑完全部 200 對" | Not finished: 155 and 92 of 200 in the ratings files (21:34), 164 and 96 when the commit was made (21:40:06). The same section says "寫本節時各判了 155 與 92 對". The text reports no test-split result from the registered judge. | `N/study/proxy/ratings/final-prefopt-vs-{falcon,mobilellm}/judge-qwen3-4b-2507-3605803b.jsonl`; `N/peek/q_*.jsonl`; `git log` 34f12ac | wrong |
| 9 | CHANGES §19.10, 測試集盲測 (189 prompts) | "三組主要比較的分數是 0.318、0.751、0.558" | Falcon on 189 prompts: 32/101/56, so 60/189 = 0.31746, which rounds to **0.317**. `restrict_report.py` prints 0.3175, and 0.318 comes from rounding that again. 0.751 (0.75132) and 0.558 (0.55820) are right, and "相差不超過 0.005" holds (0.00004 / 0.0012 / 0.0043). | `N/paper/lead4_work/scripts/restrict_report.py` on `N/study/proxy/reports/ab/final-prefopt-vs-falcon--judge-opus.json`; my recomputation | imprecise |
| 10 | CHANGES §19.10 main table, row "Falcon，K = 6、以它自己的平均對數機率挑一則", and paragraph 對照組說明了什麼 ("我方分數 0.307") | 0.307 | (35 + 53/2)/200 = 0.3075 exactly. Rounded to three decimals this is **0.308** under both round-half-up and round-half-even. "0.307" is a binary floating-point artefact of `f"{0.3075:.3f}"`. | `N/study/proxy/reports/ab/final-prefopt-vs-falcon-k6lp--judge-opus.json` (score 0.3075) | imprecise |
| 11 | CHANGES §19.10 main table, row "Falcon，K = 6、以我方解碼器的選擇器挑一則" | 95% interval [0.225, **0.348**] | The Wilson upper bound is 0.348539, so it should be **0.349**. The report first rounds to 0.3485 and then prints 0.348. | `…/final-prefopt-vs-falcon-k6ds--judge-opus.json` (`score_ci95` [0.2247, 0.3485]); recomputation | imprecise |
| 12 | CHANGES §19.10 精確計分 table, row "同一權重，K = 1" | 退化回覆 **6.8%** | 17 of 252 replies = 6.746%, so **6.7%**. 6.8% comes from rounding the stored 0.0675 a second time. | `N/study/proxy/reports/exact/ours-export-step51812-47fd99c151a50a58__final-k1-f93654208f-s0-test.json` (`per_prompt`) | imprecise |
| 13 | CHANGES §19.10, 誠實的結論 | "44.5% 的題目兩則回覆都被判為不可用" | 44.5% is the rate **per reading** (89 of 200 both_bad in r1, 89 in r2). Both readings say both_bad on 83 prompts (41.5%); at least one does on 95 (47.5%). | `N/final_eval/prefopt/final-prefopt-vs-trained/r1/` and `r2/chunk_*.verdicts.json` | imprecise |
| 14 | CHANGES §19.10, held-out paragraph | "總損失在 46,178 步之後回升（2.0914 → 2.1422）" | The total loss fell further, to its minimum of 2.0840 at 47,260, and rose only after that; 46,178 is not the turning point. The decomposition of the +0.0508 (proxy +0.0645, web −0.0085, chat −0.0054) does support "不是助理輪變差". | `N/heldout_v9.jsonl` | imprecise |
| 15 | CHANGES §19.10, 訓練結束 | "18:33:49 訓練階段在第 51,812 步因期限結束" | The `train_log` end event is at 89,242.46 s after t0, i.e. **18:33:54.4 UTC**. The run log's "train: end" is at 1498.99 min (18:33:54), and `stages.json` records the train stage finished at 18:33:55.4. 18:33:49 is the *planned* training deadline (20:31:49 − 118 min), not a recorded event. | `train_log.jsonl` (event `end`), `N/records/unpacked/run/stages.json`, `N/main16.final.log` line 995 | imprecise |
| 16 | CHANGES §19.10, 後段各階段 | "取樣 193 s（17 題、約 28 對）" | 28 is the stage's upper bound (`pairs_new_upper: 28`; the log says "about 28 pairs"). Preference optimisation received **21** sampled pairs and kept 20 (1 was too long). | `stages.json` (`sample.result`; `prefopt.result.pairs_sampled`, `pairs_by_source.sampled`) | imprecise |
| 17 | CHANGES §19.10, 事先註冊的自動評審 ("對 Falcon 那組每對約 40 s，對 MobileLLM 那組每對約 180 s") and 偏離（一） ("Falcon 類 … 約 40 s、MobileLLM 類約 180 s") | ≈40 s and ≈180 s per pair with two 4-thread streams | **180 s is the mean of 3 pairs** (#90-92, 177.7 s). In the same two-stream, 4-thread setup (testq, 20:41-21:06), 16 MobileLLM pairs averaged **91.9 s**. All 19 CPU-runtime MobileLLM pairs at the time of writing average 105.5 s (23 pairs at 21:40: 101.7 s). For Falcon, 39.9 s is 13 pairs (#143-155); the same setup gave 63.8 s on 23 pairs (#114-136), 55.2 s over all 36. | `N/peek/q_final-prefopt-vs-*.jsonl`, ratings files (`seconds`); `N/judge_remote/testq*/progress.log` | imprecise |
| 18 | CHANGES §19.10, 偏離（一） | "六組合計約 18 個機器小時" | The 18 h follows from 40 s and 180 s: 3 Falcon-type + 3 MobileLLM-type sets × 200 pairs ÷ 2 streams = 18.3 h. With the recorded means (55.2 s and 105.5 s) it is **≈13.4 machine-hours**. | same as #17 | imprecise |
| 19 | CHANGES §19.10, 收尾時的事故（二） | "8 執行緒單獨跑時每對 137.6 s，n = 5" | The 8-thread session (testq2, 21:06:46-21:22:25) produced Falcon pairs #137-**142**: 141 records at 21:20, 142 at 21:22, and the restart "resum[ed] from 142". With #142 (178.4 s) included, n = 6 and the mean is **144.4 s**. The conclusion (8 threads are slower) does not change. | `N/peek/q.jsonl`, `q2.jsonl`, `q3.jsonl`; `N/judge_remote/testq2/progress.log`, `testq3/progress.log` | imprecise |
| 20 | CHANGES §19.10, 誠實的結論 | "偏好最佳化有可量到的幫助（59 比 34）" | This holds only for blind preference (p = 0.0124). The exact scores of the same replies moved the other way: all constraints met 0.854 → 0.792 (41 → 38 of 48), degenerate replies 1.2% → 2.0%, maths 0.200 in both. The conclusion does not mention this. | `N/study/proxy/reports/exact/*final-trained*-test.json`, `*final-prefopt*-test.json` | imprecise |
| 21 | CHANGES §19.10, 偏離（一） | "K = 1 與各對照組只由 Opus 讀者判讀" | MobileLLM's two control arms have not been generated (50 of 200 prompts at 21:40), so nobody has judged them. `final-prefopt-vs-trained` and `ref-falcon-vs-mobilellm-test` are also judged by Opus only, which the text does not say. | `N/study/proxy/ratings/*/` (sources per set); `N/peek/ctlm.log` | imprecise |
| 22 | CHANGES §19.10, 精確計分 paragraph | "我方唯一領先 Falcon 的精確指標（限制滿足 0.792 對 0.688）是選擇器的功勞" | The same table's **K = 1 row** (one particle, so nothing to select) meets all constraints on **0.771** (37/48), against Falcon's 0.688 (33/48). Four of the five prompts of the lead exist without the selector; K = 6 adds one (38/48). Falcon with our selector at 0.875 shows the selector helps Falcon, not that our lead is the selector's doing. | exact reports (`per_prompt`) for K = 1, the official model and Falcon | unsupported |
| 23 | CHANGES §19.10, 測試集盲測 ("讀的是去識別、順序隨機的成對回覆"); README CN "去識別"; README EN "blinded pairs" | The readers saw de-identified pairs. | Which side was which was hidden. But every chunk file the readers got starts with `# Blind pairs … of 200 (set final-prefopt-vs-falcon)` (likewise `…-vs-mobilellm`, `…-vs-trained`, `final-k1-…`, `…-k6lp` / `-k6ds`, `ref-falcon-vs-mobilellm-test`), inside directories with the same names. So the readers knew which two systems each set compared. The instructions call them "two unnamed systems". | `N/final_eval/*/*/r1/` and `r2/chunk_*.md` line 1; `N/final_eval/JUDGE_INSTRUCTIONS.md`; `S/scripts/blind_eval.py` line 41 | unsupported |
| 24 | CHANGES §19.10, 對照組說明了什麼 | "（MobileLLM 的兩個對照組）結果見本節末的補記" | §19.10 ends with 偏離註冊協定之處; there is **no addendum (補記)**, and no result exists (generation at 50 of 200 at 21:40). | `docs/CHANGES.md` lines 458-514; `N/peek/ctlm.log` | unsupported |
| 25 | CHANGES §19.10, 事先註冊的自動評審 | "完整結果見本節末的補記" | There is no addendum, and the text gives no test-split result for the registered judge anywhere. | `docs/CHANGES.md` | unsupported |
| 26 | CHANGES §19.10, 收尾時的事故（三） | "Colab 這個帳號同時只能有 3 個執行階段" | The refusal reads: "Allocation refused (precondition failed). This can mean too many active sessions, or a temporary usage or capacity limit …". It states no limit of 3. | `S/cpu4_up.log` | unsupported |
| 27 | CHANGES §19.10, 測試集盲測 | "（`blind_eval.py export [--swap]`、`import2` …）" | `blind_eval.py` is not in the repository; it exists only as `S/scripts/blind_eval.py` in the dev container, so the procedure cannot be followed from the repo. The behaviour described is confirmed: the second reading is side-swapped and a disagreement counts as a tie. | `git ls-files`; `S/scripts/blind_eval.py` | unsupported |

### Notes (not counted as discrepancies)
- **Exact ties at the third decimal.** These were printed by Python float formatting: 0.7525 → "0.752" (table and both READMEs), 0.2825 → "0.282", 0.5625 → "0.562"; also 0.3175 → 0.318, 0.7075 → 0.708, 0.6875 → 0.688. This matches round-half-even, not round-half-up (which would give 0.753 / 0.283 / 0.563), and the text states no convention. Only 0.3075 → 0.307 (row 10) is wrong under both conventions.
- **Earlier sections contradicted by the records that §19.10 relies on:**
  - §19.8 says the held-out set has "代理題 16" rows; every record has 12 (§19.10's "12 列" and "同一批列" are right).
  - §19.9 puts the 2,048 switch at "第 36,9xx 步"; the stage event is at step 36,867 (§19.10 is right).
  - §19.5 gives the end of training as 18:33:45; the plan gives 18:33:49 and the record 18:33:54.
- The qwen dev numbers 1/16/43 and 7/2/51 are from the step-31,912 dev sets (`s2-…`). The step-25,280 sets (`s1-…`) gave 2/16/42 and 7/4/49. The text does not say which.
- **2,048-token phase:** the mean over the whole phase is 1.3889 s/step (188,746 tok/s) and the median 1.3881. "1.388 s" fits the median; "18.9 萬" fits both.
- **Test-generation export vs the run's own export:** they are separate exports. Their held-out checks printed float32 2.34441 and 2.34440 respectively (16-bit 2.34443 and |Δ| 2.19e-05 in both), so the bundles are not byte-identical. The text only claims the same weights hash and decoding, which holds (see section 2).
- 28 of 200 test prompts have identical K = 6 and K = 1 replies, which accounts for part of the "same" counts (157 and 159) in the paired tests.

## 2. Verified as correct
- **Main table, every row.** W/L/T, decisive rate, 95% intervals (except row 11), p-values (< 0.0001; 0.0124) and order agreement (90.5 / 96.0 / 93.5 / 92.5 / 92.5 / 91.5 / 91.5%) match my recomputation from the raw verdicts. The reference row 144/6/50 and 0.845 is also right.
- **Category and turn splits** against Falcon and against MobileLLM, all ten counts each.
  - Against Falcon, code (10/13/7, p = 0.68) is the only category not significantly behind.
  - Against MobileLLM, maths is 7/12/11 (p = 0.36).
  - First-position preference is +0.005 / −0.005.
- **Paired tests.**
  - K = 6 vs K = 1: against Falcon 31/12/157, +0.0675, p = 0.0054; against MobileLLM 28/13/159, +0.045, p = 0.0275.
  - Released Falcon vs the two control arms: +0.010 (p = 0.683) and +0.035 (p = 0.105).
- **Test set:** 200 prompts, 52 two-turn. All 11 excluded ids are in the test split. The 189-prompt values 0.751 and 0.558 are right, and every difference from the 200-prompt score is ≤ 0.005.
- **Exact-score table:** every cell except row 12. MobileLLM 59.5% truncated and 21.4% repeated lines; maths 6 of 30 correct.
- **Training:**
  - Step 51,812; 13,582,204,928 tokens (1.3582e10 / 1.358e10 / 1.36e10).
  - Train stage `used_s` 65,208.3, 15 attempts, `down_s` 20,488.1.
  - 2,048-token phase from step 36,867.
  - All 20 held-out values in the 4-row table are correctly rounded; the proxy rows are 12; the proxy loss rises monotonically from 2.4614 (46,178) to 3.2580; the total is 2.0914 → 2.1422.
- **Post-training stages:**
  - Sample 192.5 s on 17 prompts; critic and judge skipped (0 min).
  - DPO with β = 0.1 in 1,783.8 s: 35,201 training pairs and 1,826 held out; two candidates of 551 steps, both learning rates accepted, 3e-6 kept.
  - Select 3,295.3 s: 0.91 / 0.85 / 0.915 on 200 dev prompts, `default` chosen, "no library was built".
  - Export 35.0 s: 43,138 tokens, float32 2.34440 / 16-bit 2.34443 nats.
  - Run end 20:02:22, 29 min before the 20:31:49 deadline.
- **TPU ledger:** 7,325 + 8,220 + 70,509 = 86,054 s, leaving 346 s of 86,400. The third interval runs 01:05:55-20:41:04 (ledger file written 20:41:05).
- **Official model:**
  - Weights after preference optimisation, step 51,812. Decoding: 6 particles, temperature 0.6, top-p 0.9, 768 tokens, no library.
  - 98,036,104 numbers stored (all float16) and 19,631,240 read per token.
  - `weights16.npz` SHA-256 `2da7cbd2…41afcf4`. I recomputed it from the export and from inside the tar rebuilt from the 12 committed parts; every part hash and the whole-tar hash match `parts.json`.
  - `scripts/export.py --verify` passes on the rebuilt export (I re-ran it) and passed on the fresh CPU runtime (`S/k1.log`).
- **Test replies vs the official export:** the test replies share the backend key `ours-export-step51812-47fd99c151a50a58` with the K = 1 run made from the rebuilt official export. That key hashes the 16-bit weights' SHA-256, the config hash and the decoder code, so the weights hash is the same. The reply settings equal `decode_choice` in `export.json`. Each pair set contains exactly the replies of its two runs (200 of 200).
- **Control arms:** both Falcon control arms are `samples 6`, with `select logprob_mean` and `decoder_select` respectively.
- **Registered judge and other operations:**
  - The dev numbers are as stated, with order agreement of 60% and 80%. Gemma tied 93.3-96.7% of dev pairs and has no test ratings.
  - 155 and 92 test pairs had been judged at 21:34.
  - The stated per-pair times hold for their pairs: 63.8 s (#114-136) and 39.9 s (#143-155).
  - The 8-thread spell lasted 15.7 min.
  - MobileLLM's K = 6 control ran at 0.35 prompts/min (40 in 115.6 min).
  - A fourth CPU runtime was refused.
  - The Opus readers are not among the protocol's registered judges.
- **README numbers:** 86,054 s of 86,400 s; step 51,812; 1.358e10 tokens; 98,036,104 stored and 19,631,240 read; both table rows; 62.7% of replies (CN); 35/112/53, 29/116/55 and 22/122/56; the graft at step 29,722.

## 3. Could not verify (no primary record found)
- That the readers were Opus 5.5 agents.
- Which 11 test prompts were seen during development, and how; and that the test half informed no decision.
- The cause of the two incidents (an approval-gated workflow tool), the dev-container reboot at ≈19:08, and "之後改用不需核准的背景代理". Only the 20:12:56 boot is on record.
- "機器只有 4 顆實體核心" for the 8-vCPU CPU runtime. Only `nproc` = 8 is recorded there; the `lscpu` record I found is for the 44-vCPU TPU host.
- The budget guard's own downtime (it has no log).
- A direct comparison of the test-generation export with the official export: no `cmp_exports.py` output exists. The identity of the weights hash rests on the cache key, which I count as indirect verification.
- The dev-container run of `export.py --verify` (no log; I reproduced it).
- 20:41:04 as the return time, beyond the author's own ledger (no Colab-side record).
- "代價是 6 倍的解碼運算". It holds by construction but was never measured on the same machine.
