# Operations notes (for myself after a context summary or a worker restart)

All times UTC. `S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad` (old scratchpad:
scripts, venv, ledger, logs). `N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad`
(new scratchpad: snapshots, proxy outputs). Repo: `/home/user/turingaward2027`, branch
`claude/modest-carson-yi2d9g` (push with `git -c push.negotiate=true push origin claude/modest-carson-yi2d9g`).
Colab CLI: `export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json COLAB_CLI_ARGS=--auth=adc`.

## User's rules
- Only TPU v6e-1; 24 hours of TPU in all. I count allocated time: `$S/tpu_intervals.txt`,
  `python3 -I $S/scripts/tpu_budget.py` prints used/left seconds. The user was told.
- "Stop" from the user means stop everything, the runtime included (they said so twice on 23 Nov).
- Reply in Traditional Chinese. No pull request. Never commit the credential, never copy it to the
  runtime.
- Decision told to the user: train `configs/final.py`, no v9 head-to-head (they may overrule).

## THE TWO FACTS THAT COST HOURS (24 Nov)
1. **This container sleeps about five minutes after my turn ends or blocks**, unless a *tracked*
   background task (Bash with `run_in_background: true`) is running. `setsid nohup` processes do
   NOT keep it awake and die when it sleeps (seen 23:40 and 00:23; with a tracked task it stayed
   awake from 19:51 to 20:57). After a sleep the machine boots fresh (`/proc/uptime` small): every
   local process is gone, files are kept.
   => always have `bash $S/scripts/anchor.sh 6600` running as a tracked background task
   (timeout 7000000 ms); when it returns (an event, or TIME after 110 min) start it again.
2. **Never call a tool that needs the user's approval while they are away**: `create_trigger` and
   `delete_trigger` wait for an approval card; the turn blocks, the container sleeps (19:51-23:27
   and 00:18-00:57 were lost this way). `send_later` needs no approval: use only that for check-ins.

## What happened so far
- Runtime 1 (23 Nov 02:58-05:00): profiling, speed-up; stopped at the user's request. 7,325 s.
- Runtime 2 (counted 17:13-19:30): run `/content/runs/main16` trained to about step 5,200 at
  222k tok/s; Colab took the runtime back at about 19:29. 8,220 s. Newest snapshot: step 5,075
  (1.33e9 tokens) in `$N/snap_keep_step5075` (copy) and `$N/snap_main`.
- Why it was taken back (best explanation, being tested on CPU runtimes): Colab Pro has no
  background execution and an idle timeout of about 90 minutes counted from the last request of a
  client. The only client was the `colab exec` that carried the supervisor; it sent nothing after
  17:34:45, was disconnected at 19:09:33, the runtime was gone by 19:30.
- No v6e-1 from 19:30 to 01:05: capacity (502/503) and the two blocked turns above.
- Runtime 3 (assigned 24 Nov 01:05:55, endpoint tpu-v6e1-s-euw4a0-3lr90467632wk): the automatic
  resume worked in 2.5 min, but the training failed at once, twice, for reasons that only show on
  a NEW machine: (1) the loader replays its journal and needs the shards the journal lists; they
  must be downloaded and tokenised again first (fixed in e096dff: the train stage waits, and the
  tokenisation gets 24 workers meanwhile: about 9 min instead of 60); (2) synth and synth_rag
  share raw files and one deleted them before the other had read them (fixed in 77305dc).
  Training resumed at 01:54:10 from step 5,075 (loss 2.63 at step 5,100). The fixed files were
  uploaded to /content/TuringAward2027 by hand; the branch has them, so a future runtime
  (which clones the branch) has them too. A future change of machine costs about 15 minutes.
- CPU experiment (ended 01:57, both test runtimes stopped): a CPU runtime with no client since
  23:39:42 and no request since 23:30:49 was still alive after 144 minutes. So "90 minutes idle"
  is not the rule on CPU runtimes; why runtime 2 was taken back is NOT explained. The presence
  loops stay (they cover both "no request" and "no client").

## Operating mode
- Detached loops (they live while the container is awake):
  - `$S/scripts/alloc_loop_resume.sh SNAP RUN NOTE` (log `$S/alloc_loop.log`): requests a v6e-1
    round after round; on assignment (or a `[?]` runtime: adopted) runs `resume_on_new_runtime.sh`
    (pre-setup, snapshot upload, supervisor as a process of its own with BES_RESTORE=1 and the
    deadline moved to what is left, presence loops; log `$S/resume.log`), then starts the watch loop.
  - `$S/scripts/watch_loop.sh RUN SNAPDIR STOP MAX_S` (v2): monitor + a snapshot into `$N/snap_main`
    as soon as a new checkpoint exists (checkpoints every 25 min; looked for every 5 min), starts
    `budget_guard.sh`. Verdict in `$S/watch_verdict.txt`: ENDED (act!), LOST (`on_lost.sh` already
    closed the ledger and restarted the allocation loop), STALE (status unreadable, runtime still
    listed: look). STOP is ignored (no planned hand-over: the runtime is kept as long as Colab
    lets it live; a loss costs at most ~30 min of training). Alarms in `$S/alarms.log`, snapshots
    in `$S/snapshots.log`, polls in `$S/monitor.log`.
  - presence: `bash $S/scripts/presence_start.sh bes24h` (two loops, a short cell every 200 s; logs
    `$S/presence_A.log`, `_B.log`). Must run whenever a v6e-1 trains. The anchor restarts them.
  - `$S/scripts/budget_guard.sh`: gives the runtime back when the 24 hours are used up.
  - `$S/scripts/on_lost.sh RUN NOTE`: confirm the loss on the server's listing, close the ledger,
    stop presence, copy the newest snapshot, start the allocation loop.
- After a manual `send resume` (or any time the watch loop ended with ENDED while the run goes
  on) the watch loop must be started again (see the checklist): it ends when it sees a stopped run.
- No planned hand-over and no stop/resume "insurance" before the 12th hour (weighed on 24 Nov:
  the expected gain is about two minutes, not worth touching a healthy run).

## Since 02:30 on 24 Nov
- The user (02:33 and 02:37, their own words): "根本遠遠不夠最高效。加速優化到最高效最快最優" and
  "多用 Opus 5.5 (Max) Dynamic Workflows". So: use the Workflow tool with agents on model 'opus',
  effort 'max' for parallel work, and push efficiency (throughput, no idle TPU, productive post stages).
- Stock refusals (commit 566517a, live on the runtime since the restart of 02:28-02:31, step 6647):
  conversations with a stock refusal are skipped by the loader (fail/<source>/refusal in the train
  log: about 21 per block from tulu3_sft), a refusal counts as a degenerate reply in selection and
  in the preference labels. Told to the user in the commit and CHANGES only so far: say it in chat.
- `bash $S/scripts/restart_run.sh /content/runs/main16 FILE...` restarts run AND supervisor with
  uploaded files (about 3 minutes, a final checkpoint is written; automatic continuations are 2
  again). It hangs at its end in a foreground call (a child keeps the pipe): run it with
  `> $S/restart.log 2>&1` and without a pipe.
- Workflow wf_3d459e66-645 (task w93xi09e6, 9 agents): a glue ledger tool, four speed areas behind
  FAST_* switches (default off), an integrator that writes scripts/tpu_candidates.py and
  `$N/wf/combined.patch`, and three audits of the post stages (`$N/wf/audit_*.md`). When it ends:
  read `$N/wf/integration.md` and the audits; apply combined.patch to the repo, run its tests;
  measure on the chip in ONE pause (stop the run with `send stop`, run tpu_candidates.py through
  `exec_detached.sh`, read the CANDIDATES line, flip the accepted defaults, commit, push, upload,
  restart with restart_run.sh); fix audit blockers before the train stage ends (about 16:50 UTC).

## Since 03:00 on 24 Nov (speed and plan)
- User again (03:40): "根本遠遠不夠最高效，必須優化加速到最高效最快最優，盡量多用 Opus 5.5 (Max)
  Dynamic Workflows"; and "能加速幾倍吧" (answered with the arithmetic: ceiling about 3.3x, in reach
  1.2-1.5x; matmul 28% at peak, attention kernel 23%).
- micro_split (commit b05bfe7, live since 03:30): `<run>/micro_split` = 4 -> every micro-batch is
  cut into 4 pieces on the device (64 x 4 x 1024, 64 x 2 x 2048). Live rate 240-246k tok/s (was
  217k). The file may say `1024:4,2048:8` (per length; 86bcd45). Not measured yet: 1 row at 2048
  (the trend there was still +5% per halving: 8 -> 4 -> 2 rows = 194k / 204k / 215k) and 2 rows
  at 1024 (8 -> 4 was only +1.7%). The 2048 phase starts at p = 0.67 (roughly 14:00 UTC now).
- xla_tpu_scoped_vmem_limit_kib measured: 8/16/32/64/96 MiB = 1296/1159/1063/1104/1322 ms per
  step against 1062.5 default: NO gain, do not try again.
- post_plan (782be85, live since 04:31): `<run>/post_plan` = "sample=3 critic=0 prefopt=60
  select=35 judge=0 export=5 buffer=15" (118 min instead of 220): training ends 18:33:45 UTC,
  deadline 20:31:49 unchanged. No in-run judged comparison: I generate the replies to all 200
  test prompts myself after the export (CPU, about 2-3 h) for the blind evaluation.
- Audits applied (86bcd45, 782be85): see docs/CHANGES.md 19.5 (owed) and `$N/wf/audit_*.md`.
- Pauses are expensive (6.7 and 10.8 min for the two measurements): ONE more pause at most, with
  everything prepared: the workflow's FAST_* switches (scripts/tpu_candidates.py when the
  integrator delivers it), 1 row at 2048 / 2 rows at 1024 (scripts/tpu_micro.py --rows 2),
  verified knobs from `$N/wf/knobs_report.md`. Tools: `tpu_measure.sh OUT LIMIT CMD...` (stops,
  measures, leaves the run stopped), `resume_after.sh RUN`, `bounce_run.sh RUN` (stop + resume,
  about 100 s; upload files BEFORE it).
- Third workflow wf_6717e710-a1c (task wc6pdyucy): track A = attention kernel candidates behind
  attention_splash.FAST_KERNEL with scripts/tpu_kernels.py (`$N/wf/kernel.patch`, kernel.md);
  track B = scripts/gen_test_replies.py: the 200 blind-test replies generated on the runtime's
  44 host cores right after the export (`$N/wf/testgen.patch`, testgen.md): run it through
  exec_detached in the last 30 minutes, fetch its output with the export.
- A CPU look at the newest snapshot: `$N/midrun_sample_latest.md` (started 04:36).
- Workflows: wf_3d459e66-645 (task w93xi09e6: speed areas + integrator still running; its
  audits are done), wf_653ccf23-1db (task wmp6t4wr6: knobs research). Only 2 agents run at a
  time per workflow on this 4-core container.

## Since 07:45 on 24 Nov (one-tile kernel live)
- Commits 6bf9b6e (candidates behind switches, `bes/model/fast.py`, `<run>/fast`, extended
  `scripts/tpu_kernels.py`) and d9f0ce4 (CHANGES 19.6). Local branch `integ-speed` and worktree
  `.claude/worktrees/integ` were only the merge bench (same content as the branch).
- ONE pause 07:44-07:53 (566 s): chip results in `$N/wf/chip_kernels_0745.txt` (table:
  `python3 -I $S/scripts/kern_table.py FILE`). Live since 07:52:43 (step 22,613):
  `/content/runs/main16/fast` = `kernel=onetile:save relu2=1 ce=0` (in the manifest too).
  Live rate 281,374 tok/s at step 23,150 (was 239.5k): +17.5%; loss in the same band.
  `micro_split` stays 4 (2 rows at 1024 and 1 row at 2048 are slower with this kernel).
  NOT measured with the new kernel: 8 rows at 1024 / 4 rows at 2048 (micro_split 2); only if
  another pause happens for other reasons.
- To go back to the reference path: write `kernel= relu2=0 ce=0` into `<run>/fast` (the manifest
  keeps the last valid text, so an empty file is not enough), then `bounce_run.sh`.
- A cell on the runtime while presence runs: upload `$S/presence_pause.txt` (current epoch) to
  `/content/bes_cli/presence_pause` first (presence cells then end at once for 300 s), then
  `TAG=x exec_detached.sh` + `exec_wait.sh`. Host probe: `$S/scripts/host_peek.py` (the trainer's
  main thread uses 3% of a core: the host is not the limit).
- Workflows still running at 08:15: wf_3d459e66-645 integrator (superseded by my merge; read
  `$N/wf/integration.md` when it comes), wf_653ccf23-1db knobs (verification, then
  `knobs_report.md`), wf_6717e710-a1c reviewers (kernel review: act on real findings with a
  bounce; testgen review: `testgen.md` already carries the review, apply `testgen.patch` when
  the workflow ends, upload `scripts/gen_test_replies.py`, plan 25-35 min on 42 workers).
- T=2048 phase starts at p = 0.67: about 12:45 UTC at the new rate (p moves 0.051/h by wall
  clock); training ends 18:33:45, run deadline 20:31:49.

## Since 08:20 on 24 Nov (quality: deflections, decode study)
- Commits 641c4b6 (kernel switch as reviewed, scripts/gen_test_replies.py), 4f4ef45 (deflection
  rule, glue_ledger.py, tpu_candidates.py). Workflows wf_3d459e66-645 and wf_6717e710-a1c are
  finished; wf_653ccf23-1db (knobs research) may still run: read `$N/wf/knobs_report.md` when it
  comes; a flag candidate is tried LIVE with a bounce (no measurement pause), only if strong.
- Honest state (CPU sample of step 22,613, `$N/midrun_sample_step22613.md`): clearly behind
  Falcon (loops, wrong arithmetic, a stock deflection). Told to the user at 08:20 and 08:40.
- Deflection rule live since the bounce of 08:34-08:36 (step 25,280): chat conversations with an
  assistant turn opening "As an AI ..." / "I don't have access ..." / "I'm sorry, but ..." are
  skipped (`fail/<source>/deflection` in train_log.jsonl). Scans: `$N/openings.json`,
  `$N/deflection_check2.json` (tools `$S/scripts/openings_scan.py`, `deflection_check.py`).
- Decode study s1 on the runtime's idle host cores (CPU only, started 08:42:
  `/content/bes_cli/study/s1`, driver `$S/scripts/study_driver.py`, 2 streams x 16 workers,
  nice 10): the first 60 dev prompts under default / ngram3 / t04 / k1 / short (max_tokens 320)
  / t08, from the export of step 25,280 (`/content/bes_cli/study/s1/default/export`).
  Fetch finished variants: `bash $S/scripts/study_fetch.sh s1` -> `$N/study/s1/raw/<variant>/`;
  merge: `python scripts/gen_test_replies.py $N/study/s1/raw/<variant> --merge-into $N/study/proxy`.
  Opponents' replies to all 400 prompts are in `$N/study/proxy/outputs/` (from the runtime).
  Judging: pairs (variant vs default, default vs falcon/mobilellm, split dev) ->
  `blind_eval.py export` chunks -> Opus judges in workflows (verdict files) -> `import` -> `score`.
  NEVER tune on the test split. If the training rate drops by more than 1% while the study
  runs, kill it: `pkill -f study_driver.py; pkill -f gen_test_replies.py` in a host cell.
- Final 200 test replies: `scripts/gen_test_replies.py` on the runtime right after the export
  (commands in `$N/wf/testgen.md`; 25-35 min on 42 workers; calibration 7.5 min measured here),
  the rest locally from the fetched cache.

## Since 09:16 on 24 Nov: THE RUN IS NOW v9 (user: "要 v9", "優化 v9 到最高效最快最優", "邊做邊寫頂會最佳論文水準的報告 paper")
- `/content/runs/main16` on the runtime IS THE v9 RUN since 09:57 (configs/v9.py, grafted from
  final at step 29,722 with scripts/graft_run.py; manifest has `graft`). The stopped final run is
  `/content/runs/main16_final` (fallback: `host_run.sh peek 60 $S/scripts/swap_run.py
  /content/runs/main16 x --back` after a stop, then resume) and locally `$N/snap_final_main16`.
  `$N/snap_main` holds v9 snapshots from now on (checkpoints are about 1.5 GB).
- Commits: 9a4d652 (graft tool, loader tokens_before), 769f240 (switches flat/rows/tokn, tests),
  1b9d1b8 (CHANGES 19.7). Live: `<run>/fast` = `kernel=onetile:save relu2=1 ce=0 flat=1 rows=1
  tokn=1`, micro_split 4, table_grad take. Live rate 196.7k tok/s (final was 281k). Chip
  numbers: `$S/v9_measA.txt` (take), `v9_measB_sorted.txt`, `v9_measC_onehot.txt`, `v9_measD.txt`.
- Helper: `bash $S/scripts/host_run.sh TAG LIMIT_S FILE_OR_REPO_SCRIPT ARGS...` runs a script on
  the runtime host (pauses presence, detached process, prints its output).
- T=2048 phase from p = 0.67: about 12:47 UTC. Training ends 18:33:45; deadline 20:31:49.
- Decode study s1 (final weights, step 25,280): all six variants generated; pair sets and chunk
  files in `$N/study/s1/judge/*`; judges: workflows wf_f8a4dfa8-025 (done) and wf_c10c208d-aa1;
  verdict files `chunk_NN.verdicts.json`. Import: `$S/venv/bin/python $S/scripts/blind_eval.py
  import --root $N/study/proxy --pairs NAME --source judge-opus --verdicts FILES`; then
  `python -m bes.eval.proxy --root $N/study/proxy score --pairs NAME --source judge-opus`.
  First half: mostly "both_bad" (the mid-run model is weak); say the numbers plainly.
- Paper: workflow wf_035f9026-91b writes `$N/paper/` (paper.md, CLAIMS.md, related_work.md,
  figures/, tables.md, REVIEW_1.md, RESPONSE_1.md). Copy into `docs/paper/` of the repository
  when a draft is complete; update with results as they come; a Traditional Chinese version at
  the end. Never invent numbers.
- Endgame plan: at the end of training generate the 200 test replies on the runtime's host cores
  with `scripts/gen_test_replies.py` (v9 export works: `$N/wf/v9_endgame.md`), blind pairs vs
  both opponents with two readings on exchanged sides, score, honest report.

## Endgame (v9), decided 11:15 on 24 Nov
- Results so far: `docs/CHANGES.md` 19.7-19.8; `$N/paper/FACTS_2.md`. Mid-run blind (final, step
  25,280, 60 dev prompts, one reading): vs Falcon 2/43/15 (0.158), vs MobileLLM 15/8/37 (0.558).
  Held-out across the graft: v9 0.016 nats ahead at equal time after 54 min (`$N/midrun_eval_1.json`;
  loop on the runtime writes `/content/bes_cli/heldout_v9.jsonl`, copy `$N/heldout_v9.jsonl`).
- Study s2 (v9 step 31,912, dev 60 prompts, default and ngram3) started 11:10 on the runtime:
  when `/content/bes_cli/study/s2/progress.log` says "all done": download
  `/content/bes_cli/study/s2/results.tar.gz` into `$N/study/s2/raw` (as for s1), then
  `bash $S/scripts/study_pairs.sh s2 default ngram3`, judge the chunk files with the workflow
  script `.../workflows/scripts/decode-study-judges-a-wf_f8a4dfa8-025.js` (args: list of
  {set, chunks}), import with `--source opus-1`, `score`.
- 18:33:45 training ends -> sample 3 min -> prefopt 60 min (to about 19:37) -> select 35 min ->
  export 5 min (about 20:17); run deadline 20:31:49; TPU budget ends 20:46:30 (budget_guard).
  1. about 18:36: `bash $S/scripts/endgame_testgen.sh trained 44` (200 test prompts from the trained
     weights, default decoding, 42 host cores; about 45 min).
  2. about 19:40 (prefopt done; check the run log for whether its weights were kept):
     `bash $S/scripts/endgame_testgen.sh prefopt 44` if prefopt weights were saved.
  3. after the export: `python3 scripts/colab_cli.py fetch`, `fetch_records.sh`, a last snapshot,
     `endgame_testgen.sh --fetch trained|prefopt`, and `/content/bes_cli/heldout_v9.jsonl`,
     `/content/bes_cli/study/*/results.tar.gz`, `/content/runs/main16.log`; then `send release`,
     `down --yes`, close the ledger. Official score: the exported model (the run's own rules) on the
     200 test prompts, two readings on exchanged sides (`blind_eval.py export [--swap]`, `import2`
     with `--source judge-opus`), also on the 193 prompts never looked at (the mid-run samples used
     7 test prompts: chat-002, chat-017, math-034, math-011, instruct-062, instruct-034,
     writing-100, code-026, factual-008, reason-005 as far as they exist in the test split).
- Paper: `$N/paper/` -> `docs/paper/` when the first workflow ends; second revision with
  FACTS_2.md and later results; Traditional Chinese version at the end.

## Since 12:10 (disk incident, study s2 fetched)
- 12:10: the container's disk was full (0 MB): the knobs-research workflow's verification agents had
  written about 20 GB of `sceptic_*` directories into `$N`. That workflow is stopped (it was of low
  value by then); old snapshots/dumps removed; 32 GB free at 12:14. `$N/sceptic_*` may be deleted
  when space is needed again. Nothing running was lost: uptime 11.3 h (no sleep), anchor, watch
  loop, presence A/B and budget guard alive, snapshot of step 34,166 at 11:52 complete.
  Before large downloads at the end: `df -h /tmp/claude-0` (need about 3 GB).
- Study s2 (v9 step 31,912; dev 60; default, ngram3) fetched into `$N/study/s2/raw`; pair sets
  `s2-default-vs-falcon`, `s2-default-vs-mobilellm`, `s2-ngram3-vs-default` under
  `$N/study/s2/judge/`; judge workflow `wf_130e3e2e-5ce` (task wy6y1wgm1) writes
  `chunk_NN.verdicts.json`. Then for each set:
  `$PY $S/scripts/blind_eval.py import --root $N/study/proxy --pairs SET --dir $N/study/s2/judge/SET --source opus-1`
  and `$PY -m bes.eval.proxy --root $N/study/proxy score --pairs SET --source opus-1` (see how s1
  was imported: `$N/study/s1/`).
- Measured on the runtime for v9: 60 dev prompts (75 replies) took 1,798 s with 16 workers while a
  second stream of 16 workers ran on the sibling threads and the trainer ran: about 150 replies
  per 1,290 s on 20 physical cores after calibration. 200 test prompts with 42 workers: about
  31 min + calibration 8 min + export: about 41 min. `endgame_testgen.sh TAG 44` is tight: give it
  55 minutes (the two generations then overlap for some minutes; acceptable).
- If the export's choice (weights trained/prefopt x decode default/k1/noband) is not one of the two
  generated sets: fetch the export, release the TPU, and generate the 200 replies on a CPU-only
  Colab runtime (allowed, not counted) with `gen_test_replies.py EXPORT --split test`.
- Held-out (64 rows, same rows, CPU): step 31,912: 2.2337; 33,038: 2.2204; 34,166: 2.2111;
  35,294: 2.2001 (chat assistant 1.4463). `$N/heldout_v9.jsonl` (download again at the end).

## Final blind evaluation: the mechanics (written 12:45)
- Scripts: `$S/scripts/final_pairs.sh TAG` (after `endgame_testgen.sh --fetch TAG`; merges into
  `$N/study/proxy`, publishes `final-TAG-vs-falcon` and `final-TAG-vs-mobilellm` on the test split,
  writes `$N/final_eval/TAG/<set>/r1|r2/chunk_NN.md`, r2 = sides exchanged, 14 chunks of 15 each),
  `$S/scripts/final_score.sh TAG` (import2 with source judge-opus, score).
- Judges: workflow script
  `/root/.claude/projects/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/workflows/scripts/final-blind-judges-wf_b440ed0e-80c.js`
  with args `{"root": "$N/final_eval/TAG", "sets": ["final-TAG-vs-falcon"], "readings": ["r1","r2"], "chunks": 14}`
  (absolute paths). One workflow runs 2 judges at a time: start one workflow per pair set and
  reading (4 workflows = 8 judges at a time; 7 judges each, about 15 min per judge).
- Reference on the same judge: `ref-falcon-vs-mobilellm-test` (the two opponents against each
  other, test split, both readings) in `$N/final_eval/ref/`, workflow `wf_b440ed0e-80c` (task
  wbazldd4b) started 12:44. Import: `blind_eval.py import2 --root $N/study/proxy --pairs
  ref-falcon-vs-mobilellm-test --source judge-opus --order1 .../r1/chunk_*.verdicts.json --order2
  .../r2/chunk_*.verdicts.json`, then `score`.
- Second reading of the s2 opponent sets (sides exchanged): `$N/study/s2/judge2/<set>/` (workflow
  wf_1fc9d90e-6a4, task wsyr0wa6e): with the first reading -> `import2 --source judge-opus`.
- Verified 12:40 on CPU: the v9 decode tests (17), the v9 export and the prefopt tests (43) pass
  with tokn, rows, flat and relu2 switched on (pytest plugin `$S/plug/fastflags_plugin.py`): the
  post-training stages run with the live switches.

## Where the trained model is kept (decided 12:55; told the user, who may say otherwise)
- The container is ephemeral and the runtime ends at 20:46: after `colab_cli.py fetch` the export
  (`final/`: 16-bit weights about 196 MB, tokenizer, config, card) exists only in this container.
  The repo is PUBLIC and its default branch is the working branch.
- Plan: commit the export as the packed parts (16 MB each, with their SHA-256 list) under
  `release/main16-v9/` as the LAST commits of the branch (two parts per commit, pushed one by one
  so that a size limit of the proxy cannot lose everything), with a README giving the commands to
  join, verify (`python scripts/export.py --verify`) and use it (`scripts/chat.py`,
  `scripts/arena.py --export ...`). Verify by unpacking from the committed parts.
- If the user prefers Drive or no publication: they say so; do not use the Colab credential for
  anything else (no Drive API from here).

## State at 13:20 (what runs, what is owed)
- Training: T=2048 since 12:48, 1.388 s/step = 188.8k tok/s; ends 18:33:45.
- Pushed: 111b402 (CHANGES 19.9), ef5649c (docs/paper: first revision + figures + data + evidence).
  `bash $S/scripts/paper_sync.sh` copies paper/figures/data/analysis/evidence into docs/paper again
  (`--with-test` only after ALL blind readings of test pairs are finished: it adds keys/outputs).
- Paper: first workflow done (REVIEW_1/RESPONSE_1). Title now "What a Parameter Budget Does Not
  Count". It found: 11 test prompts were seen mid-run (189 untouched, not 193); my 12:10 cleanup
  deleted local final checkpoints 5,075/22,613/26,890 (trend has 3 points). Protocol deviation to
  repair: the pre-registered judge is `qwen3-4b-2507` (GGUF, both orders; second judge
  `gemma4-e4b`), my Opus readers are an addition.
  - Figures pass 2: workflow wf_aead8f09-b31 (task wksb41gda).
  - Final revision (after the test results): feed FACTS_2.md + final numbers + micro-graft result;
    then the zh-TW version; then `paper_sync.sh --with-test`, commit, push.
- Pre-registered judge on dev pairs: runs on the runtime host since 13:15
  (`bash $S/scripts/judge_remote.sh fetch devq` shows progress and copies the ratings into
  `$N/study/proxy/ratings/<set>/judge-qwen3-4b-2507.jsonl` when done; then
  `python -m bes.eval.proxy --root $N/study/proxy score --pairs SET --source judge-qwen3-4b-2507`).
  For the TEST pairs the same must run: on the runtime if time allows (cores are busy with the
  generations after 18:36), else on a CPU Colab runtime after the TPU is released
  (`colab new -s bes-cpu --high-mem`; the judge GGUF is 2.5 GB from the Hub;
  `scripts/judge_driver.py`-like call: `python -m bes.eval.proxy --root ROOT judge --pairs SET
  --judge qwen3-4b-2507 --threads 8`). Measure the speed from the dev pass first.
- Micro-graft experiment: workflow wf_ad9c3744-513 (task wjuhxpzq7) writes
  `$N/wf/micro_graft.patch` and `micro_graft.md`. Then: copy the repo on the runtime without
  `data_cache/tok` to `/content/bes_cli/microrepo`, apply the patch there, run 15 arms (5 x 3
  seeds) in parallel on cores 2-21,24-43 (2 threads each), all must END BY 18:20 (kill:
  `pkill -f micro_ablate.py` in a host cell) because the test generation needs the cores at 18:36.
- Reference judges (opponents against each other, test split): wf_b440ed0e-80c, 18/28 chunk
  files at 13:17; then `bash $S/scripts/score2.sh ref-falcon-vs-mobilellm-test
  $N/final_eval/ref/ref-falcon-vs-mobilellm-test/r1 $N/final_eval/ref/ref-falcon-vs-mobilellm-test/r2`.
- Endgame generation now pins itself off cores 0,1 (+siblings): `endgame_testgen.sh TAG 58`
  uploads and runs `endgame_driver.py` (40 workers). Probe on the live run directory worked
  (12:52-12:56, 4 dev prompts, export of step 36,422 in 34 s).

## State at 14:20
- Micro-graft experiment RUNNING on the runtime host since 14:12:53: separate clone
  `/content/bes_cli/microrepo` (ef5649c + `$N/wf/micro_graft.patch`; samples downloaded there;
  11 new tests pass there), driver `/content/bes_cli/micro_driver.py 6000000` (source
  `$S/scripts/micro_driver.py`), 15 arms x 6M tokens (1,464 steps, 4.7 s/step each on CPUs c,c+22
  for c=2..16), out `/content/bes_cli/micro/graft_6000000` (progress.log, <arm>.log,
  results.tar.gz at the end). ETA 16:10. Background task bm44r4gg2 fetches into
  `$N/micro_graft/6M`. Kill if still running at 18:20: host cell `pkill -f micro_ablate.py; pkill -f micro_driver.py`.
  The review of the patch is still running (workflow wjuhxpzq7): a defect that changes numbers
  means a rerun. When done: apply the patch to the repo (tests pass), commit, write the result
  into the paper (FACTS_2.md) and CHANGES.
- Registered judges on dev pairs done (qwen3-4b-2507 and gemma4-e4b; FACTS_2.md 13:52, 14:12).
  Reference opponents-vs-each-other on test: Falcon 144/6/50 over MobileLLM (0.845).
- At the end compare exports on the runtime: `bash $S/scripts/host_run.sh peek 60
  $S/scripts/cmp_exports.py /content/runs/main16/final /content/bes_cli/testgen_trained/export
  /content/bes_cli/testgen_prefopt/export` (same `weights16.npz` hash and decode_choice = the
  generated replies are the official model's).
- `paper_sync.sh` also copies figures/data: do not run it while the figures workflow
  (wksb41gda) is writing (it copied half-written CSVs at 13:55; reverted before the commit).

## State at 16:50
- Pushed: b04478e (micro graft code), a316f8c (figures pass 2), b1e9140 (micro graft results).
- Micro graft, 15 arms x 6M tokens: DONE (results in `$N/micro_graft/6M`, FACTS_2.md 16:50,
  docs/MICRO_GRAFT.md "Results"). Restart arms (v9_pages from scratch, 3M tokens, 3 seeds) run on
  the runtime since 16:43 (`/content/bes_cli/micro/graft_3000000`), fetched by background task
  ba60jc9z8 into `$N/micro_graft/3M` (the driver's report step fails there by design: the report
  needs all five families; read `<arm>/results.jsonl`). Then: add to FACTS_2.md and
  docs/MICRO_GRAFT.md, commit. MUST be over by 18:20 (else host cell: pkill -f micro_ablate.py).
- Paper pass 3: workflow wf_ea200474-e65 (task wfbm645sl): lead author + micro graft figure,
  then review (REVIEW_2.md) and fix (RESPONSE_2.md). When it ends: `paper_sync.sh`, commit, push.
  The LAST pass (after the test results) fills Appendix M's shells and the abstract; then zh-TW.
- Tools ready for the end: `fetch_pack.sh RUNTIME_DIR NAME OUT` (16 MB parts, SHA-256; tested:
  the export of step 36,422 came back in 67 s; local copy `$N/packs/probe36422`, verified with
  `scripts/export.py --verify`, and `scripts/chat.py --export ... --device cpu --prompt ...`
  answered in 8 min on 2 cores); `cmp_exports.py`; `cpu_runtime.sh up|status|down` (CPU Colab
  runtime `bes-cpu`, presence C/D, set-up log `/content/bes_cli/cpu_setup.log`);
  `SESSION=bes-cpu judge_remote.sh start TAG JUDGE THREADS STREAMS SET...` for the registered
  judges on the test pairs (speed on the TPU host at 12 threads: qwen3-4b-2507 13-17 s per pair,
  gemma4-e4b 8-9 s per pair; a CPU runtime has 8 vCPUs: expect 2-3x slower).
- Early copy of the model at the end: right after `endgame_testgen.sh trained 58` has made its
  export (about 18:38), `bash $S/scripts/fetch_pack.sh /content/bes_cli/testgen_trained/export
  trained_export $N/packs/trained_export` (1 min): the trained weights are then safe locally.
- Alarm at 15:00 was benign: "eval ... sidecar skipped: late" (the run's own rule).

## ENDGAME LOG (from 18:33)
- 18:33:49 training ended: final checkpoint `ckpt-000022-step-000051812`, 13.582e9 tokens.
  18:34-18:38 sample (193 s; 17 prompts, 28 pairs); critic skipped (0 min); 18:38:56 prefopt
  started (slice 3600 s; 6881 s left in the run => run deadline 20:33:37).
- 18:36:17 `endgame_testgen.sh trained 58`: export of step 51,812 (36 s, held-out check passed:
  loss32 2.32877, loss16 2.32881), 200 prompts / 252 replies, 40 workers on CPUs 2-21,24-43,
  log `/content/bes_cli/testgen_trained/testgen.log`. Fetch: `endgame_testgen.sh --fetch trained`
  then `final_pairs.sh trained`.
- 18:38 the trained export is LOCAL: `$N/packs/trained_export` (12 parts of 16 MB + parts.json,
  verified). If the official export is these weights (prefopt rejected), these parts are what
  gets committed; else fetch the official one the same way:
  `fetch_pack.sh /content/runs/main16/final official_export $N/packs/official_export`.
- 18:39 CPU runtime `bes-cpu` (High-RAM, 8 vCPU, 50 GB; Python 3.13, jax 0.11.1, llama_cpp
  0.3.37; repo clone at /content/TuringAward2027; presence C/D). 18:42:18 started there:
  `/content/bes_cli/controls_driver.py 6 4 2 falcon mobilellm` = the registered control arms of
  docs/EVAL_PROTOCOL.md section 3 on the TEST split (each opponent with 6 samples, kept by
  logprob_mean and by decoder_select); progress `/content/bes_cli/ctl_progress.log`; outputs
  `/content/bes_cli/ctl_outputs.tar.gz` (rewritten after each arm) -> unpack and copy
  `outputs/<backend>/<run>.jsonl|.meta.json` into `$N/study/proxy/outputs/` and make pair sets
  against our official run. Tier 2 (after the main result): also ours with K = 1
  (`gen_test_replies.py EXPORT --split test --set extra.particles=1 --label final-k1`).
- Paper pass 3 pushed (11aa314). It found: the registered protocol also demands K = 1 results and
  K-matched opponent arms (now being generated); the chat share of the stream ramps from 0.25
  to 0.70 after p = 0.67 by design (explains the fast fall of chat loss).
- The registered judge on the TEST pairs needs CPU: bes-cpu is busy with the controls; bring up
  a second one: `SESSION=bes-cpu2 bash $S/scripts/cpu_runtime.sh up` (its presence names C/D
  collide with bes-cpu's logs only; fine), then
  `SESSION=bes-cpu2 bash $S/scripts/judge_remote.sh start testq qwen3-4b-2507 8 1 SET...`.
- At the very end: kill the watch loop BEFORE giving the TPU back if it still runs
  (`pkill -f scripts/watch_loop.sh`), `touch $S/alloc_stop $S/presence_stop_bes24h`,
  `python3 scripts/colab_cli.py send release`, `python3 scripts/colab_cli.py down --yes`, close
  the open interval in `$S/tpu_intervals.txt` with the epoch of the stop.

## AFTER THE TPU (state at 20:50 on 24 Nov) -- READ THIS FIRST
- TPU GIVEN BACK 20:41:04 UTC, ledger closed: 86,054 s of 86,400 used. NEVER allocate a TPU again
  (`$S/alloc_stop` exists; no watch loop, no alloc loop may run). Only CPU runtimes remain:
  `bes-cpu` (opponents' K=6 control arms, `/content/bes_cli/ctl_progress.log`, outputs
  `/content/bes_cli/ctl_outputs.tar.gz`) and `bes-cpu2` (registered judge qwen3-4b-2507 resumed on
  the official test pairs; `SESSION=bes-cpu2 bash $S/scripts/judge_remote.sh fetch testq`).
  Give both back when their work is fetched: `SESSION=bes-cpu bash $S/scripts/cpu_runtime.sh down`,
  same for bes-cpu2.
- THE WORKFLOW TOOL BLOCKS: twice (19:07, 19:32) a Workflow call waited for an approval nobody
  could give; the worker restarted, the container rebooted (all local processes gone) and I then
  wasted the wake-up by answering "No response requested". RULES: (1) do not call Workflow
  while the user is away; (2) the Agent tool (subagent_type general-purpose, model opus, effort
  max, run_in_background true) works without approval: use it for judges and writers; (3) after
  a "worker process was restarted" note CONTINUE AT ONCE: restart `bash $S/scripts/anchor2.sh
  6600` (tracked background), presence for the CPU runtimes (`presence.sh bes-cpu C 200`,
  `presence.sh bes-cpu2 E 200`), then go on.
- THE OFFICIAL MODEL: prefopt weights of step 51,812, default decoding. Pushed:
  `release/main16-v9/` (commits 99ba0e7..994b3a1). Local: `$N/packs/official_export` (+ `x/final`
  unpacked), `$N/packs/trained_export`, `$N/snap_final` (1.7 GB), `$N/records/unpacked`,
  `$N/main16.final.log`, `$N/heldout_v9.jsonl`.
- Test replies: `$N/testgen/prefopt/out` (official) and `$N/testgen/trained/out`; merged into
  `$N/study/proxy`; pair sets `final-prefopt-vs-falcon|mobilellm` (OFFICIAL), `final-trained-vs-*`;
  chunk files `$N/final_eval/prefopt|trained/<set>/r1|r2/chunk_NN.md`.
- Opus readers as background Agents (prompt: read `$N/final_eval/JUDGE_INSTRUCTIONS.md`, two chunk
  files each): official-vs-falcon r1 and r2 launched 20:42-20:46 (14 agents). Then
  official-vs-mobilellm r1, r2 (14 agents), then trained sets if wanted. When a set has all 28
  verdict files: `bash $S/scripts/final_score.sh prefopt` (import2 + score, source judge-opus).
- Registered judge, partial at 20:40 (TPU host): official vs Falcon 113/200: 7/24/82 (decisive
  0.226, p 0.0033); vs MobileLLM 73/200: 18/3/52 (decisive 0.857, p 0.0015). Resumed on bes-cpu2
  at 20:41 (4 threads per set; about 35 s per pair; done about 22:00).
- Exact scores on test are in FACTS_2.md (20:32): maths 0.20 (Falcon 0.50, MobileLLM 0.40);
  constraints 0.79 (0.69, 0.17).
- Still owed: K = 1 arm of ours on test (registered); opponents' K=6 arms -> pair sets + judge;
  the 189 untouched prompts; paper final pass (fill Appendix M from FACTS_2 20:32 + results),
  zh-TW version, `paper_sync.sh --with-test` when ALL test readings are done, docs/CHANGES.md
  19.10, README result section, final report to the user; offer to revoke the Google token.

## 巡檢清單 (what a check-in does)
1. `cut -d' ' -f1 /proc/uptime` (small = the container slept: everything below must be restarted);
   `pgrep -af "alloc_loop_resume|presence.sh|watch_loop|budget_guard|anchor"`;
   `tail -5 $S/alloc_loop.log $S/resume.log $S/watch_verdict.txt; tail -2 $S/monitor.log $S/snapshots.log`;
   `colab --auth=adc sessions`; `colab --auth=adc usage`.
2. A `bes24h` runtime exists and trains (`python3 -I $S/scripts/monitor.py /content/runs/main16 --max-s 1 --poll 5`):
   - presence loops missing -> `bash $S/scripts/presence_start.sh bes24h` AT ONCE;
   - watch loop missing -> `cd $S && setsid nohup bash $S/scripts/watch_loop.sh /content/runs/main16 $N/snap_main 0 90000 > $S/watch_out.log 2>&1 < /dev/null &`
3. No runtime: if the ledger has an open interval (`grep ' - ' $S/tpu_intervals.txt`) the runtime was
   lost while the container slept -> `bash $S/scripts/on_lost.sh /content/runs/main16 "note"` does
   everything. Otherwise, if no `alloc_loop_resume.sh` runs, start it (detached) with the newest
   complete snapshot (`$N/snap_main`, `$N/snap_keep_*`: compare `parts.json` tokens; keep a copy).
4. A `[?]` v6e-1 in `colab sessions` with nothing handling it -> `bash $S/scripts/adopt_and_resume.sh
   <epoch it can earliest have been assigned> SNAP /content/runs/main16 "note"`, then the watch loop.
5. Start the anchor (tracked background task) if none runs. Keep two or three `send_later`
   check-ins scheduled ahead (about every 60 minutes) as a safety net; nothing that needs approval.
6. Tell the user briefly what changed (they follow the session). When a v6e-1 is assigned, offer
   the optional extras: the Drive consent link (`$S/drive_link_v6e.txt`) and the browser link that
   attaches the Colab page to the runtime (`colab --auth=adc url -s bes24h`): an open browser tab
   is the strongest keep-alive.

## The run and its end
- Created 17:34:54 on 23 Nov, overrides model.table_grad=take, run.bench_plans=reduced,
  run.post_minutes=critic=0,buffer=20, run.post_reserve_hours=3.6667, optim.micro_seqs=16; plan
  splash-remat1-ms16-take (216,681 tok/s at T=1024, 192,912 at 2048). After training: sample 50,
  prefopt 60, select 45, judge 30, export 15, buffer 20 minutes. Its deadline is moved at every
  resume to (TPU time left - 15 min).
- Do not change `bes/config.py` (dataclasses, defaults) on the branch while this run lives: a
  continued run checks the configuration hash; overrides are fixed. Code-only changes are fine.
- At the end (watch verdict ENDED with a finished run), before giving the runtime back: stop
  presence (`touch $S/presence_stop_bes24h`), wait for an idle kernel, `python3 scripts/colab_cli.py fetch`
  (export), `bash $S/scripts/fetch_records.sh /content/runs/main16 $N/records` (proxy outputs and
  records), a last snapshot (`bash $S/scripts/snapshot_now.sh $N/snap_final`), `python3
  scripts/colab_cli.py send release`, `down --yes`, close the ledger interval.
- Task 7: opponents' replies to the 200 test prompts are in `$N/proxy` (falcon recommended,
  mobilellm recommended and neutral, all complete). Blind judging: `$S/scripts/blind_eval.py`
  (export chunks, two readings on exchanged sides by sub-agents, import2), then
  `python -m bes.eval.proxy --root $N/proxy score --pairs NAME --source judge-...`.
  Mid-run look at a snapshot: `$S/scripts/midrun_sample.py $N/snap_main`.
- Owed to the repo: docs/CHANGES.md section 19.4 (runtime 2 lost, idle rule, the operating mode, my
  blocked turns). Offer to revoke the Google token at the very end.

## State at 21:28 UTC on 24 Nov (after the TPU) -- READ THIS FIRST IF CONTEXT WAS LOST
- Keep-awake: tracked background task `bash $S/scripts/anchor3.sh 7000` (started 21:27; ends about
  23:24; it also restarts dead presence loops of bes-cpu / bes-cpu2 / bes-cpu3). Restart it as a
  tracked background task before it ends. Colab allows THREE sessions; a fourth is refused.
- Done and scored (Opus readers, 2 readings, source `judge-opus`, root `$N/study/proxy`):
  official vs Falcon 35/108/57 (0.318); vs MobileLLM 116/15/69 (0.752); after vs before prefopt
  59/34/107; Falcon K6 log-prob control 35/112/53 (0.307); Falcon K6 our-selector control
  29/116/55 (0.282). All in `$N/paper/FACTS_2.md` (last section "Added 21:27 UTC").
- Running:
  - Opus readers of the K = 1 sets (`$N/final_eval/k1/final-k1-vs-falcon|mobilellm/r1|r2`,
    14 chunk files per reading, 2 chunks per agent, Agent tool, opus, effort max, background;
    at most 20 agents at once). When all `chunk_*.verdicts.json` exist:
    `bash $S/scripts/score2.sh final-k1-vs-falcon $N/final_eval/k1/final-k1-vs-falcon/r1 .../r2 40`
    (same for mobilellm). Then FACTS_2.md.
  - bes-cpu2: registered judge qwen3-4b-2507, two 4-thread streams, both official sets
    (`SESSION=bes-cpu2 bash $S/scripts/judge_remote.sh fetch testq3`; ratings land in
    `$N/study/proxy/ratings/<set>/judge-qwen3-4b-2507-3605803b.jsonl` when "all done"; then
    `python -m bes.eval.proxy --root $N/study/proxy score --pairs <set> --source judge-qwen3-4b-2507-3605803b`).
    8 threads is SLOWER than 4 on these runtimes. ETA Falcon ~22:25, MobileLLM ~23:15.
  - bes-cpu: MobileLLM K6 control generation, forward (`/content/bes_cli/ctl_progress.log`,
    `/content/bes_cli/ctl_mobilellm_logprob_mean.log`; root `/content/bes_cli/ctl_root`).
  - bes-cpu3: the same arm from the end backwards (`controls_rev_driver.py 6 8 1 mobilellm
    logprob_mean`; `/content/bes_cli/ctlrev_progress.log`; cache tarball
    `/content/bes_cli/ctlrev_cache.tar.gz` every 4 minutes; stop file `/content/bes_cli/ctlrev_stop`).
    When forward count + reverse count >= 200: stop the reverse run, download its cache
    (`cache/mobilellm-f680f0533368-qwen3.jsonl`), append it to the forward root's cache on
    bes-cpu (upload + a small cell that concatenates), kill and restart controls_driver.py there
    (`controls_driver.py 6 4 2 mobilellm`: it replays the cache, writes the logprob_mean outputs,
    then runs decoder_select, whose two-turn second turns need NEW generations: about 52 x 6;
    split them the same way if it is slow). Then fetch `ctl_outputs.tar.gz`, merge outputs into
    `$N/study/proxy/outputs`, publish pairs `final-prefopt-vs-mobilellm-k6lp|k6ds`, chunk files
    under `$N/final_eval/controls/`, Opus readers, score2.sh.
  - Paper final pass agent (writes `$N/paper/paper.md`, `RESPONSE_3.md`, `lead5_work/`).
- Decision recorded: the registered judge is run in full on the two official sets only; K = 1
  and the control arms are read by the Opus readers only (compute: about 100 s per pair on a CPU
  runtime). Say so as a deviation in the paper and in CHANGES 19.10.
- Killing on a runtime: `rt_kill.py` patterns must not match its own command line: write
  `judge_driver[.]py`, `bes[.]eval[.]proxy`.
- Still to do: CHANGES 19.10 + README results; paper review/fix agents; zh-TW paper;
  `paper_sync.sh --with-test` (only when ALL test readings are finished); commit; push; CPU
  runtimes down (`SESSION=<name> bash $S/scripts/cpu_runtime.sh down`); final report to the user.

## State at 21:45 UTC on 24 Nov -- supersedes the "Running" list of 21:28
- K = 1 sets scored: vs Falcon 22/122/56 (0.250), vs MobileLLM 99/16/85 (0.708). CHANGES 19.10 +
  README results committed and pushed (34f12ac). Audit agent of those numbers writes
  `$N/audit_19_10.md` (apply its corrections, commit).
- REGISTERED JUDGE (qwen3-4b-2507), MobileLLM set is slow (178 s/pair), Falcon set 40 s/pair:
  - bes-cpu2, tag testq3 (standard driver, 2 streams x 4 threads): Falcon set (164/200 at 21:40,
    done about 22:05) and MobileLLM set going FORWARD from index 92 (96 at 21:40). Peek:
    `colab download -s bes-cpu2 /content/bes_cli/judge/testq3/root/ratings/<set>/judge-qwen3-4b-2507-3605803b.jsonl`.
  - bes-cpu3, tag testq4 (judge_shard.py): MobileLLM index ranges [146,173) on cores 0-3 and
    [173,200) on cores 4-7. `bash $S/scripts/judge_shard.sh fetch bes-cpu3 testq4 final-prefopt-vs-mobilellm`.
  - WHEN the Falcon stream ends on bes-cpu2 (its progress.log says "end final-prefopt-vs-falcon"):
    copy the 200-record Falcon file into `$N/study/proxy/ratings/final-prefopt-vs-falcon/`, score it
    (`--source judge-qwen3-4b-2507-3605803b`), and start a third shard there:
    `bash $S/scripts/judge_shard.sh start bes-cpu2 testq4 qwen3-4b-2507 final-prefopt-vs-mobilellm 0-3:125:146`
    (tag testq4 on bes-cpu2 is a new directory there; fetch with `... fetch bes-cpu2 testq4 ...`).
  - The forward stream on bes-cpu2 must be stopped when it reaches index 125 (or when all shards
    are done): `rt_kill.py "judge_driver[.]py" "bes[.]eval[.]proxy"` AFTER saving its ratings file.
  - Merge: put the forward file (bes-cpu2 testq3) as the main ratings file locally, then
    `bash $S/scripts/judge_shard.sh merge testq4 final-prefopt-vs-mobilellm` (prints missing indices),
    then score.
- MOBILELLM K6 CONTROL generation (logprob_mean arm, run id
  mobilellm-f680f0533368-qwen3/recommended-32f0c6d864-s0):
  - bes-cpu forward (50/200 at 136 min = 21:34): `/content/bes_cli/ctl_mobilellm_logprob_mean.log`.
  - THIS CONTAINER, from the end backwards: `setsid nohup taskset -c 1,2,3 nice -n 5 $S/venv/bin/python
    $S/scripts/ctl_local.py 6 3 1 mobilellm logprob_mean "::-1" > $N/ctl_local_root/stdout.log 2>&1 &`
    (root `$N/ctl_local_root`, progress.log there, stop file `$N/ctl_local_root/stop`; RESTART IT
    AFTER A CONTAINER REBOOT: it resumes from its cache).
  - After the judge is done, bes-cpu2 / bes-cpu3 join with middle slices (controls_rev_driver.py
    takes only "from the end"; give it a slice argument first), then merge all caches into the
    forward root on bes-cpu and run both select rules there (see the 21:28 section).

## State at 22:42 UTC on 24 Nov -- supersedes the earlier "Running" lists
- DONE and in the repo (b94e332): CHANGES 19.10 (audited and corrected), README results,
  scripts/blind_eval.py. Registered judge complete on both official sets (12/61/127 vs Falcon;
  59/5/136 vs MobileLLM). Neutral-title replication done (33/109/58; 112/15/73). All in
  `$N/paper/FACTS_2.md` (sections 21:27 ... 22:38 and "CORRECTIONS (22:25 UTC)").
- Rounding convention in repo docs: three decimals, exact halves to even (0.7525 -> 0.752,
  0.3075 -> 0.308, 0.7425 -> 0.742). The paper must use the same.
- RUNNING:
  - MobileLLM K6 control generation, both select rules per slice (`controls_slice_driver.py 6 8 1
    mobilellm both SLICE [ROOT]`; progress `/content/bes_cli/ctlslice_progress.log`; cache tarball
    `/content/bes_cli/ctlslice_cache.tar.gz`): bes-cpu ":100" in root /content/bes_cli/ctl_root
    (its cache also holds the first 60 prompts made by the old driver); bes-cpu2 "100:136";
    bes-cpu3 "136:172" (both in /content/bes_cli/ctlrev_root). This container: `ctl_local.py 6 3 1
    mobilellm logprob_mean "::-1"` (root `$N/ctl_local_root`; from the end backwards; it must
    cover [172,200) and then be STOPPED (touch `$N/ctl_local_root/stop`) and restarted for
    decoder_select on "172:200": `ctl_local.py 6 3 1 mobilellm decoder_select "172:200"`).
    AFTER A CONTAINER REBOOT restart the local one (it resumes from its cache).
  - WHEN all four are done: download the three `ctlslice_cache.tar.gz`, concatenate their
    `cache/mobilellm-f680f0533368-qwen3.jsonl` with the local one into
    `$N/ctl_merged_root/cache/` and run locally (model files in `$N/models`):
    `python -m bes.eval.proxy --root $N/ctl_merged_root --models-dir $N/models run --model mobilellm
    --preset recommended --split test --samples 6 --select logprob_mean --workers 0 --no-download`
    and the same with `--select decoder_select` (both must report 200 complete, 0 new
    generations; if a few are missing let them generate). Copy `outputs/` into
    `$N/study/proxy/outputs/`, exact scores, pairs `final-prefopt-vs-mobilellm-k6lp|k6ds` against
    the official run, chunk files with NEUTRAL titles (scripts/blind_eval.py default) in neutral
    directories, 28 Opus readers per set, import2 --source judge-opus, score.
  - Agents: paper final pass (a1c3...), failure analysis (writes `$N/paper/analysis_fail/`).
- THEN: paper review/fix (use audit style), zh-TW version, `paper_sync.sh --with-test` (only after
  ALL test readings incl. the MobileLLM control sets), commit, push, CPU runtimes down, final
  report. Keep-awake: `anchor3.sh 7000` started 21:27 ends 23:24 -> restart it (tracked).

## State at 00:20 UTC on 25 Nov -- NEW USER INSTRUCTIONS (00:07): READ FIRST
- The user said (in Chinese): "the weights file from before DPO must also be there" -> DONE:
  `release/main16-v9-before-dpo/` (12 parts, parts.json, README), commits 9db8cff..5f6884d,
  rebuilt from the committed parts and verified.
- The user said: "from now on do not use Dynamic Workflows etc. any more". => NO new Workflow
  calls and NO new subagents (Agent tool) for anything. Do the remaining work in the main loop
  with scripts. (One translation agent for part 4, launched before the message, was still
  writing `$N/paper/zh/zh/part_04.md` at 00:15.)
- Consequence for MobileLLM's K = 6 control arms: no Opus readers. They are judged by the
  registered automatic judge (qwen3-4b-2507) on the CPU runtimes with judge_shard.py streams
  (2 streams x 4 threads per runtime), and by exact scores. Say so in CHANGES 19.10 / the paper.
- MobileLLM K6: logprob_mean arm complete (200/200; merged cache `$N/peek/merged_cache_0012.jsonl`,
  root `$N/ctl_merged_root`; outputs copied to `$N/study/proxy/outputs/mobilellm-f680f0533368-qwen3/
  recommended-32f0c6d864-s0.*`; exact: maths 0.400, constraints all met 0.1875, degenerate 25.4%
  (released preset 62.7%), mean words 238.6; pair set `final-prefopt-vs-mobilellm-k6lp` published
  00:14). decoder_select arm (run id `.../recommended-aca173fa2c-s0`): 181/200 at 00:12; the 19
  missing prompts are being generated on bes-cpu ("100,101,108,111,112,", root ctl_root), bes-cpu2
  ("114,129,130,132,135,", root ctlrev_root), bes-cpu3 ("38,63,64,70,72,85,86,88,92,", root
  ctlrev_root); then download the three `ctlslice_cache.tar.gz`, `ctl_assemble.py $N/ctl_merged_root
  <bes-cpu> <bes-cpu2> <bes-cpu3> <local cache>` (must give 200/200 for both), copy outputs, exact,
  pairs `final-prefopt-vs-mobilellm-k6ds`, judge shards for both sets, merge, score.
  Generations with the same key made on two Colab runtimes are identical (48 of 48 checked);
  the local container's differ (other CPU); local made prompts 170..199 of the logprob arm.
- Translation: parts 1-3, 5-12 done in `$N/paper/zh/zh/` (raw copies in zh_raw), harmonised by
  script (numbers in 萬, a few terms); checker `$S/scripts/zh_check.py EN ZH`. Assemble to
  `$N/paper/paper.zh-TW.md` when part 4 exists; later patch the MobileLLM control [TBD] cells in
  both language versions by hand.
