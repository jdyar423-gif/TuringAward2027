# Track B (testgen): the 200 blind-test replies of the final model on the runtime's CPU cores

Patch: `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/testgen.patch`
(two new files, nothing else touched: `scripts/gen_test_replies.py`, `tests/train/test_gen_test_replies.py`,
7 tests). No hashed source changed (bes/decode, bes/exact, bes/model, chat_format, retrieve, tokenizer,
config.py untouched), so the generator's cache key is the export's own `Exported.key()`. No pip install.
Author's working files: `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/testgen/`;
the review's: `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/wf/testgen_review/`.

`scripts/gen_test_replies.py EXPORT --out OUT --workers W` answers the test prompts with the export's
own generator (`bes.export.export_generator`) in W pinned CPU worker processes, every conversation
through `bes.eval.proxy.runner.converse` (multi-turn, `derive_seed(0, id, turn, 0)`, the harness's
cache keys), and writes the run in the proxy layout plus `replies.jsonl`. Its speed comes from an
elastic key-value cache; after the review that cache is exact by construction and by check (below).

## review

### Verdict

Usable after the review's fixes, **not as submitted**. The submitted elastic cache does not give the
export's bits once a reply's filled cache passes 1,024 slots (most writing/chat replies at 6
particles, which take ~1,000 synchronous steps, and long second turns), and its calibration cannot
see it. The reviewed elastic cache
prefills on the export's own length and decodes only on lengths that a real-state calibration on the
machine itself has checked bitwise for the spans used; each reply reports what it used and is kept
only if covered. My runs equal the repository's single-process path field for field (latency
aside): single-turn and two-turn, library on, 6 and 1 particle, after a killed worker and a killed
parent; the calibration's real-state checks cover filled spans up to 3,011 slots. The time
estimate was too optimistic: **plan 25-35 min for the 200 prompts on 42 workers** (not 12-20).

### F1 (critical): the submitted elastic cache is not exact

Newest snapshot (step 18,934, copied to `testgen_review/snap`, exported through the script's own
snapshot path: unpack, SHA-256 per part, `export_run`, held-out check passed, |Δ| 1.55e-5 nats).
Rows (log-probabilities) compared bitwise with the export's own 5,248-slot cache, one core:

| case | 6 particles | 1 particle |
|---|---|---|
| a 1,085-token prompt prefilled on 1,280 / 1,536 slots (the submitted `_tmax` prefills on the elastic length) | differs from the prefill row on, max abs 3.6e-5 / 4.2e-5 | differs, 5.1e-5 / 5.1e-5 |
| the same prompt prefilled on 5,248 slots, decoding steps on 1,280 / 1,536 / 1,792 slots (span 1,089) | differs from the first step, 2.7e-5 | differs, 2.9e-5 |
| the same, steps on 2,048 / 2,304 / 2,560 / 2,816 / 3,072 slots | identical | identical (2,048, 2,304) |
| synthetic one-step scan (random keys/values, spans every 64 slots) | 256-1,024 identical at every span; 1,280 and 1,536 identical up to 1,024, differ from 1,088 on | 256 and 512 differ at every span; 768, 1,024, 2,048, 3,072 identical; 1,280-1,792 differ from 1,216 on (real data: from 1,089); 2,304-2,816 differ from 2,176 on |

XLA:CPU processes the attention's slot axis in blocks of 1,024 slots: a step on `L` slots equals the
5,248-slot step while the filled span stays within `L`'s whole blocks (below one block: within `L`),
and a prefill chunk's result depends on the cache length too. The submitted calibration checks a
~20-token prompt (~70 filled slots): at 6 particles that passes every length, so it certifies 1,280-
1,792 slots that are wrong for spans above 1,024. In the test split that is most writing/chat replies
(writing-036 takes 1,039 synchronous steps at 6 particles: span 1,103) and the long second turns. Texts would
mostly survive (a token changes only when a uniform draw falls in a ~1e-5 sliver), but
`logprob_sum`, `logprob_sum_free`, the resampling weights differ: the cache lines would not be the
single-process path's. The author's proof could not see it: its 4 prompts had spans up to 832 slots
on lengths up to 1,024.

**Fix (in the patch).**
- `ElasticLM` prefills on the export's own length (the same jitted program on the same inputs) and
  copies only the used first slots into the 6 particle rows (`first_slots`, a pure copy); one prefill
  program per worker instead of one per start length. Cost about +1 s per 64-token prompt chunk
  (first prompts have 1-3 chunks, second ones up to ~16).
- Candidate lengths: 256, 512, 768 (multiples of `--cache-step` below one block) and the whole blocks
  1,024, 2,048, 3,072, each used for filled spans up to `length - 61`; beyond 3,011 slots the export's
  own 5,248. A table `{length: longest span}` drives the growth (`length_for`).
- `calibrate_elastic` (worker 0): one ~3,070-token prompt prefilled chunk by chunk on 5,248 slots; at
  filled spans every 256 slots up to `length - 64` the state is copied into the particle rows on 5,248
  slots and on the candidate's first slots, and both take 4 decoding steps (same tokens, one particle
  paused); all rows must be bitwise equal. A failing candidate is dropped (replies take the next
  passing length). Then the real class end to end on a short prompt (prefill copy, growth through every
  kept length) against the plain model. Here, 6 particles: all 6 candidates pass, 30 checks, 458 s at
  ~50% of a shared core (load 7), 4.4 GB peak. 1 particle: 256 and 512 fail (at span 192), 768-3,072
  pass, 29 checks, 194 s, 1.8 GB — the per-machine, per-K check is necessary.
- Every finished reply carries `usage` `{length: longest span stepped}` (beside the cache line, never in
  it). Speculative replies (decoded before the decision on every candidate) are kept only when the
  calibration covers all their pairs, else dropped and handed out again (covered earlier turns stay);
  after the decision every reply is checked the same way.
- The calibration is stored in `OUT/calibration.json` with a fingerprint (CPU vendor, model, stepping,
  cache size, flags; jax/jaxlib; `XLA_FLAGS`; threads; generator key; settings hash; this script's
  SHA-256); a restarted call on the same machine type reuses it (`--recalibrate` forces a new one).

### F2: the time estimate was too optimistic

At 6 particles a reply lasts as long as its slowest particle plus sentence-level waiting: writing-036
turn 1 took **1,039** synchronous steps (6 particles, 3,102 tokens in all, a 235-token chosen reply,
488 s at ~50% of a core), not the 500-768 the estimate assumed. Revised model below.

### F3-F5: smaller fixes

- **Exit status** (check 6): partial results exited 1; now 0 when at least one prompt is complete
  (the `TESTGEN` line says `complete` or `partial`), 1 when none is, 2 when nothing could start.
- **Failures** are all recorded in `testgen.json` with their error (was: the first 20); the `TESTGEN`
  line now has `failed`, `unstarted`, `decode` and `lengths`.
- **CPU only, hardened**: TPU flags in `XLA_FLAGS` and `LIBTPU_INIT_ARGS` were removed for workers only;
  now also for the parent, at the first statement (`_cpu_only_environment`). Workers refuse to start
  unless `jax.default_backend()` is `cpu` and `jax.devices()` holds CPU devices only; the parent aborts
  on any other report.

### The six checks

1. **No TPU/GPU backend.** `JAX_PLATFORMS=cpu` is set before any import in the parent and in every
   worker's environment: `/proc/<pid>/environ` of both R1 workers shows `JAX_PLATFORMS=cpu
   OMP_NUM_THREADS=1`, affinity 2 resp. 3, 6 threads each; every worker reports backend `cpu`, platforms
   `["cpu"]` (asserted in the end-to-end test); a non-CPU report aborts the run (unit test). There is no
   TPU here: on the runtime this rests on `JAX_PLATFORMS=cpu` (JAX creates no other backend).
2. **Identical to the single-process path** (reference: `testgen_review/tools/ref_single.py`, independent
   of the script: the CLI's `build_spec`, `make_backend` = the export's own generator and 5,248-slot
   cache, `runner.converse`, one core; same run id `ours-export-step18934-c37a4e3b4429e7db/final-default-bd03abdea2-s0`):


   | run (elastic, `gen_test_replies.py`) | prompt | turns, steps | library | vs reference | latency elastic / reference |
   |---|---|---|---|---|---|
   | R1, 2 workers, 6 particles | chat-082 | 1, 48 | "Haiku" | identical | 28.4 / 52.0 s |
   | R1 | instruct-058 | 1, 90 (2 resamplings) | none | identical | 24.8 / 186.5 s |
   | R1 | instruct-076 | 2, 100 + 24 | "Snowman" (turn 1) | identical, both turns | 43.1 + 8.1 / 213.0 + 42.2 s |
   | R1k1, 1 worker, 1 particle (`--set extra.particles=1`; 256/512 excluded by the calibration) | chat-068 | 2, 329 + 297 (spans to 681 on 768 slots) | none | identical, both turns | 68.3 + 81.5 / 290.1 + 220.6 s |

   "identical" = every field of the cache line's reply except `latency_s` (text, raw text, tokens,
   finish, `logprob_sum`, all of `extra`) and the assembled conversation records; same run ids
   (`.../final-default-bd03abdea2-s0`, at 1 particle `.../final-default-62e52dd65e-s0`). These replies
   are short (spans up to ~700); longer spans are covered by the calibration's real-state checks (up
   to 3,011 filled slots) and the experiments of F1, not by a full reply: a 1,000-step reply on the
   export's own cache takes ~40 min here.
3. **Resumable; a killed worker loses at most its prompt.** R2 = the R1 prompts in a fresh `OUT`
   (`testgen_review/tools/kill_test.sh`, only processes of this review signalled): SIGKILL of worker 1
   while busy on instruct-076 → the parent logs `worker 1 ended (its output closed, exit -9) during
   instruct-076: it is handed out again` and `restart 1 of 2`; then, with one reply in the cache, SIGKILL
   of the parent → both workers gone within 5 s (`PR_SET_PDEATHSIG`), the cache line kept. The same
   command again: `1 prompts are complete in the cache`, 3 replies generated, `complete`; R2 against R1:
   4 / 4 generations and 3 / 3 conversations identical (latency aside). A worker loses only the
   reply in flight (finished turns of a two-turn prompt are appended as they finish); speculative
   replies held in the parent are lost with the parent (at most worker 0's calibration time of work).
4. **Pairing and scoring with the opponents.** R1 merged with `--merge-into` into a copy of `$N/proxy`
   (`testgen_review/proxy_copy`; `$N/proxy` untouched); `pairs --split test` against
   `falcon-e60034381f4a/recommended-5ccec0259b-s0` and `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`:
   3 pairs each (instruct 2, chat 1; 1 two-turn); `judge --judge standin-longer` (3 verdicts each);
   `score` (key commitment chain checked by `score_report`): reports in `proxy_copy/reports/ab/review-vs-*`.
   A partial run pairs as well (`make_pairs` uses the prompts both runs hold; later ones are added as a
   new version of the pair set).
5. **Time and memory for 44 cores**: see the cost model. W = 42 x 1 core is right (2 threads per worker
   gave nothing; with SMT the 22 physical cores carry two workers each). What limits scaling: decoding
   is memory-bound once 42 workers run (each re-reads ~0.55 GB of float32 block weights per step, 8
   loops x 2 blocks, plus 6 x L x 96 KB of cache: ~150 GB/s for 42 workers; here one worker's weights fit
   the 260 MB L3, so the dev-container step times are optimistic); compile ~20-55 s per worker at start
   plus ~5 s per new length (at most 6 lengths, one prefill program); the library is memory-mapped
   (shared page cache; only 5 of the 200 test prompts retrieve a passage). Memory per worker: ~0.6-1.0 GB
   ready, ~1.3 GB on short replies (R1 max 1.33 GB), ~3-4.5 GB on 2,048-3,072 slots (cache plus a grow or
   gather copy); worker 0 ~4.4 GB while calibrating: ~85-130 GB of 176 at the peak; `--max-rss-gb 12`
   only guards a runaway.
6. **Failure handling.** A prompt that raises is recorded with its error and skipped; the worker goes
   on; exit 0 if anything was produced, 1 if nothing. End-to-end test with the hidden `--fail-ids` hook:
   one of two prompts raises → `partial`, exit 0, error in `testgen.json` and the log; the only prompt
   raises → `failed`, exit 1.

Tests: `taskset -c 2,3 nice -n 10 $PY -m pytest tests/train/test_gen_test_replies.py -o addopts="-p no:cacheprovider" -q -x`
→ 7 passed in 172 s (96 s at lower load). On the smoke export the calibration keeps 128-320 slots and
the 2-worker run (with speculation: 3 replies kept) equals `runner.run(workers=0)`.

### Cost model (revised)

Per decoding step at 6 particles, one core of this container (author's bench, `OurLM.step` alone):
117 / 152 / ~200 / 254 / 430 / 627 / 1,309 ms at 256 / 512 / 768 / 1,024 / 2,048 / 3,072 / 5,248 slots,
plus ~45 ms of decoder per step. Steps at 6 particles: writing/chat first turns ~1,040 (writing-036),
code ~770 (code-028), math ~215, instruct ~100-150, second turns shorter. With the reviewed lengths:

| replies | count | seconds each (one core) | core-seconds |
|---|---|---|---|
| writing + chat, turn 1 (~1,040 steps) | 100 | ~279 | 27,900 |
| writing + chat, turn 2 (prompt ~450, ~500 steps) | 26 | ~142 | 3,700 |
| code, turn 1 (~770) / turn 2 (~240) | 30 / 8 | ~173 / ~53 | 5,600 |
| instruct (~150 steps) and math (~215), both turns | 88 | 25-38 | 2,700 |
| compile (start-up + new lengths), 42 workers | | 40-70 | ~2,100 |
| total | 252 replies | | **~42,000** |

On 42 workers that is ~17 min if each ran at this container's single-core speed; on the runtime the 42
workers share the memory bandwidth and (very likely) 22 physical cores, a factor of ~1.3-2: **plan 25-35
min for all 200 prompts**, start-up (~1 min) and worker 0's calibration (~3-4 min on a full core; the
other 41 decode meanwhile) included. The longest single prompt (two-turn writing, ~7 min here) starts
first. Hand-out order and cumulative work: the 52 two-turn prompts (13,900 core-s), the 37 single-turn
writing (24,200), 22 code (28,000), 37 chat (38,400), 30 instruct (39,100), 22 math (39,900). A run cut
by `--minutes` therefore leaves the cheapest prompts, but chat ones are not cheap (~280 core-s each):
at `M` = 20 expect ~100-145 prompts done; finishing 60-100 leftovers on this container's 2 shared
cores takes hours (it needs the export locally and the same `bes/` code, the key hashes it, and runs
its own calibration first). Give the run 30-35 minutes if the window allows it.

### Runtime commands

```
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json
# 0. once, after the patch is in the main repository (the file is new on the runtime too)
colab --auth=adc upload -s bes24h /home/user/turingaward2027/scripts/gen_test_replies.py /content/TuringAward2027/scripts/gen_test_replies.py
# 1. as soon as /content/runs/main16/final/export.json exists; M = minutes left before release - 8
TAG=testgen bash $S/scripts/exec_detached.sh bes24h $S/testgen_nonce.txt 3 --env "BES_CMD=scripts/gen_test_replies.py /content/runs/main16/final --out /content/bes_cli/testgen --workers 42 --cores-per-worker 1 --minutes M --grace-minutes 3 --max-rss-gb 12"
# 2. progress at any time (first lines: workers ready, then "calibration: ... on [256, 512, 768, 1024, 2048, 3072] slots")
colab --auth=adc download -s bes24h /content/bes_cli/testgen/testgen.log $N/wf/testgen/runtime_progress.log
# 3. the end
TAG=testgen bash $S/scripts/exec_wait.sh bes24h $S/testgen_nonce.txt $N/wf/testgen/runtime_last.log 2400
# 4. fetch, unpack, merge next to the opponents, pair
mkdir -p $N/wf/testgen/runtime/out && colab --auth=adc download -s bes24h /content/bes_cli/testgen/testgen.tar.gz $N/wf/testgen/runtime/testgen.tar.gz
tar -xzf $N/wf/testgen/runtime/testgen.tar.gz -C $N/wf/testgen/runtime/out
python scripts/gen_test_replies.py $N/wf/testgen/runtime/out --merge-into $N/proxy        # prints RUN_ID
python -m bes.eval.proxy --root $N/proxy pairs --x RUN_ID --y falcon-e60034381f4a/recommended-5ccec0259b-s0 --name final-vs-falcon-test --split test
python -m bes.eval.proxy --root $N/proxy pairs --x RUN_ID --y mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0 --name final-vs-mobilellm-test --split test
```

`M`: with a release at 20:40 and a start at about 20:12, `M` = 20 gives ~100-145 of the 200 prompts
(two-turn, writing and code first); all 200 need `M` ≈ 27-32 (plus the 3 of grace). Read the ETA in the
progress log after ~6 minutes. The calibration verdict in the log: lengths `[256, 512, 768, 1024, 2048, 3072]` is the
expected one; fewer lengths is still exact (slower); "no elastic length" means the export's own cache for
every reply (exact, 5-10x slower): then restart with `--cache elastic-always` only if speed matters more
than exactness (own key `<key>-elastic`).

Files to fetch: `/content/bes_cli/testgen/testgen.tar.gz` (replies.jsonl, testgen.json, testgen.log,
calibration.json, `proxy/cache/<key>.jsonl`, `proxy/outputs/<key>/final-default-<hash>-s0.jsonl` +
`.meta.json`, `logs/worker-NN.log`) and `/content/bes_cli/testgen_last.log` (the command's output, last
line `TESTGEN {…}`). If the tarball is missing: `/content/bes_cli/testgen/proxy/cache/<key>.jsonl` (the
resumable store; key in the first log line), `testgen.log`, `calibration.json`. To finish locally:
also `/content/runs/main16/final/` (the export, ~240 MB with the library) and run the same command here
with `--workers 2` and `--out` the unpacked tarball (resumes from its cache; calibrates for this machine).

Fallbacks (unchanged): export missing → source `/content/runs/main16 --make-export --no-download`;
stop early → `pkill -TERM -f "gen_test_replies.py /content"` (the parent; outputs are written).

### What the review kept

Source resolution (export / run / snapshot with SHA-256 per part, tested here on the newest snapshot),
the loader (`load_export` once with `SHA256SUMS`, `export_generator` per worker, key check), the settings
(`ours`/default + the export's decode choice + `extra.use_library=false` without a library, as
`ours_contestant` and the judge stage build them), `runner.converse`, the SMT-aware core plan, the
per-reply fsync'd cache, the hand-out order (two-turn first, then writing, code, chat, instruct, math),
restarts, `--max-rss-gb`, `--minutes` / `--grace-minutes`, the signals, the outputs and the merge.
Not adopted: a shared persistent XLA compile cache (jax's file cache writes non-atomically without
eviction locks; ~5% to gain).

### Working files of the review (`testgen_review/`, about 1.4 GB; `snap/` and `out0/source/` can go)

`snap/` (verified snapshot copy), `out0/export` (the step-18,934 export), `calib_k6.json`,
`calib_k1.json` (calibrations), `rows_k*_p1100.jsonl`, `dec_k*.jsonl`, `span_k*.jsonl` (the exactness
experiments), `steps_w036.*` (the 6-particle writing reply step by step), `R1/`, `R1k1/`, `R2/` (runs),
`refA/`, `refB/`, `refk1/` (single-process references), `compare_*.json`, `proxy_copy/` (pairs and
scores), `tools/` (every script used).
