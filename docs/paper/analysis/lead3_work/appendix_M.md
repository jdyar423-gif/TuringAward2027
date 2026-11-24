## Appendix M. Results to come: table shells

The last pass fills these shells without restructuring them. Every cell marked [TBD] is copied from the
file named for its table; nothing is to be computed by hand. Conventions: W / L / T, score, interval,
sign test and the reader diagnostics are as in Table 8 (from our side; T includes "both bad" and pairs
whose two orders disagree). R = `N/study/proxy/reports/ab/`. TAG = the test generation whose weights
and decoding equal the run's own export (`trained` or `prefopt`): `S/scripts/cmp_exports.py` compares
`/content/runs/main16/final` with `/content/bes_cli/testgen_TAG/export` (same `weights16.npz` hash and
`decode_choice`); if neither matches, the replies are generated again on a CPU runtime and TAG names that
generation (OPERATIONS, "Since 12:10"). The pair sets are `final-TAG-vs-falcon` and
`final-TAG-vs-mobilellm` on the 200 `test` prompts (`S/scripts/final_pairs.sh TAG`); the Opus readings are
imported by `S/scripts/final_score.sh TAG` (source `judge-opus`, two readings on exchanged sides from
`N/final_eval/TAG/<set>/r1|r2/chunk_*.verdicts.json`); the registered judges' ratings come from
`S/scripts/judge_remote.sh` on a CPU runtime into `N/study/proxy/ratings/<set>/`, then
`python -m bes.eval.proxy score`. MobileLLM-R1.5 is its `recommended` arm unless the `neutral` arm wins
on `dev` (Section 7.2, deviation 4) [TBD: which arm, before the test split is opened].

**Table M1 (R1). The test split, 200 prompts: each opponent and each reader** (`R/final-TAG-vs-<opponent>--<source>.json`,
fields `overall` and `judge`; sources `judge-qwen3-4b-2507-3605803b`, `judge-gemma4-e4b-85a896a0`,
`judge-opus`) [TBD: Opus readings once the test replies exist, from about 19:30 UTC on 24 November;
registered judges on a CPU runtime after the TPU is released, 24–25 November].

| opponent | reader | n | W / L / T | score [95%] | decisive win rate [95%] | sign test p | orders agree | first-position preference |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | `qwen3-4b-2507` (registered) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Falcon-H1-Tiny-90M-Instruct | `gemma4-e4b` (registered second) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Falcon-H1-Tiny-90M-Instruct | Opus, two readings | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5-140M | `qwen3-4b-2507` (registered) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5-140M | `gemma4-e4b` (registered second) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5-140M | Opus, two readings | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| reference: Falcon-H1-Tiny vs MobileLLM-R1.5 | Opus, two readings | 200 | 144 / 6 / 50 | 0.845 [0.788, 0.889] | 0.960 [0.915, 0.982] | < 0.0001 | 93.5% | +0.013 |

**Table M2 (R2). By category and by turns** (W / L / T; the same files, fields `by_category` and
`by_turns`; the counts of prompts are those of the `test` split) [TBD: with R1].

| slice (prompts) | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon-H1-Tiny: `gemma4-e4b` | Falcon-H1-Tiny: Opus | MobileLLM-R1.5: `qwen3-4b-2507` | MobileLLM-R1.5: `gemma4-e4b` | MobileLLM-R1.5: Opus |
|---|---|---|---|---|---|---|
| chat (50) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| code (30) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| instruction-following (40) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| maths (30) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| writing (50) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| single-turn (148) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| two-turn (52) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Table M3 (R3). Length buckets and reader diagnostics** (the same files, fields `by_length` (ratio of
our words to the opponent's: shorter below 2/3, longer above 3/2) and `judge`) [TBD: with R1].

| row | Falcon-H1-Tiny: `qwen3-4b-2507` | Falcon-H1-Tiny: `gemma4-e4b` | Falcon-H1-Tiny: Opus | MobileLLM-R1.5: `qwen3-4b-2507` | MobileLLM-R1.5: `gemma4-e4b` | MobileLLM-R1.5: Opus |
|---|---|---|---|---|---|---|
| ours shorter: n; W / L / T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| similar: n; W / L / T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ours longer: n; W / L / T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| orders agree | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| first-position preference | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| probability mass on A, B, C (automatic judges) | [TBD] | [TBD] | – | [TBD] | [TBD] | – |
| pairs with a reply cut for the judge (over 4,000 characters) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Table M4 (R4). Exact scores on the 200 test prompts (252 turns)** (`bes.eval.exact_score` on every
reply, as `lead2_work/exact_dev.py` does for `dev`, over the outputs in `N/study/proxy/outputs/`: ours
from `N/testgen/TAG/out` merged by `final_pairs.sh`, the opponents' test replies already there) [TBD:
after the test replies are generated, from about 19:30 UTC on 24 November].

| measure | ours (TAG) | ours (the other weights, if generated) | Falcon-H1-Tiny | MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |
|---|---|---|---|---|---|
| maths answers correct (of 30 maths prompts) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| constrained prompts with every constraint met (of [TBD]) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| replies cut at the token limit (of 252) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| replies with a looping tail or repeated lines (of 252) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| empty replies (of 252) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| stock refusals (of 252) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| stock deflections (of 252) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| median words per reply | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Table M5 (R5). The trained weights against the weights after preference optimisation** (pair sets
`final-trained-vs-*` and `final-prefopt-vs-*`, both generated by `S/scripts/endgame_testgen.sh trained|prefopt`;
reports `R/final-trained-vs-<opponent>--<source>.json` and `R/final-prefopt-vs-<opponent>--<source>.json`)
[TBD: the prefopt generation exists only if the stage saved its weights, from about 19:40 UTC; the
second pair of sets is read only if time allows].

| our weights | opponent | reader | W / L / T | score [95%] | sign test p |
|---|---|---|---|---|---|
| trained (end of training) | Falcon-H1-Tiny | Opus, two readings | [TBD] | [TBD] | [TBD] |
| trained | MobileLLM-R1.5 | Opus, two readings | [TBD] | [TBD] | [TBD] |
| after preference optimisation | Falcon-H1-Tiny | Opus, two readings | [TBD] | [TBD] | [TBD] |
| after preference optimisation | MobileLLM-R1.5 | Opus, two readings | [TBD] | [TBD] | [TBD] |
| after preference optimisation against trained | – | Opus, two readings | [TBD: if run; pair-set name not fixed] | [TBD] | [TBD] |
| paired difference of the two arms against the same opponent (`bes.eval.proxy compare`) | each opponent | Opus | [TBD] | – | [TBD] |

**Table M6 (R6). Held-out losses of the export** (the rows of Section 6.5; nats per token; the export's
check record in `final/export.json` of the run's export, `/content/runs/main16/final`, fetched at the end
by `scripts/colab_cli.py fetch`; the last training checkpoint from `N/heldout_v9.jsonl`) [TBD: after the
export, about 20:17 UTC].

| domain | export (16-bit weights) | the same weights in float32 | last training checkpoint [TBD: step] | `final` 29,722 (the graft) |
|---|---|---|---|---|
| web (64 rows) | [TBD] | [TBD] | [TBD] | 2.7886 |
| chat (64 rows) | [TBD] | [TBD] | [TBD] | 1.6183 |
| chat, assistant tokens | [TBD] | [TBD] | [TBD] | 1.5164 |
| web + chat (comparison set) | [TBD] | [TBD] | [TBD] | 2.2267 |
| proxy (12 rows; reported apart) | [TBD] | [TBD] | [TBD] | 2.6361 |
| bits per byte, web rows (opponents: 1.570 and 1.500) | [TBD] | – | [TBD] | 1.017 |

**Table M7 (R7). What the preference-optimisation stage did** (the stage ledger `run/stages.json` and the
stage's records in the last snapshot `N/snap_final`, and the run log `/content/runs/main16.log`, both fetched
at the end; `bes/train/prefopt.py` defines the checks) [TBD: after the 60-minute stage, about 19:37 UTC].

| quantity | value |
|---|---|
| minutes used of 60 | [TBD] |
| preference pairs used (UltraFeedback; own sampled pairs) | [TBD] |
| learning rates tried (1e−6, 3e−6) and steps run at each | [TBD] |
| held-out preference accuracy before / after (standard error) | [TBD] |
| held-out assistant loss change (limit +3%) | [TBD] |
| sampled length and repetition against their bands | [TBD] |
| accepted (weights kept) or rejected (trained weights ship) | [TBD] |

**Table M8 (R8). What the decoding selection chose** (the same stage ledger and records; the export's
`config.json` field `decode_choice`, printed by `cmp_exports.py`) [TBD: after the 35-minute stage, about
20:12 UTC].

| quantity | value |
|---|---|
| settings compared with the default (e.g. one particle, no length band) | [TBD] |
| development prompts used; paired sign test of each against the default | [TBD] |
| setting kept | [TBD] |
| minutes used of 35 | [TBD] |

**Table M9 (R9). Training totals** (the trainer's ledger `run/train/ledger.json` and train log in
`N/snap_final`; TPU seconds from `S/tpu_intervals.txt` once the third interval is closed) [TBD: training
totals at 18:34 UTC; TPU totals after the runtime is released, about 20:46 UTC].

| quantity | value |
|---|---|
| final step; tokens trained (`v9` part since step 29,722) | [TBD] ([TBD]); projected about 51,800 steps and 1.358 × 10¹⁰ tokens |
| training seconds of the kept trajectory; checkpoint seconds; set-up seconds | [TBD] |
| TPU seconds: runtime 1, runtime 2, runtime 3, total | 7,325; 8,220; [TBD]; [TBD] |
| share of TPU time in training steps of the kept trajectory | [TBD] |
| live rate in the last stretch (p ≥ 0.92, loss on assistant tokens only), n | [TBD] |

**Table M10 (R10). The 16-bit check of the export** (`final/export.json`, check record, and the weight
format record written by `bes/export.py`) [TBD: after the export, about 20:17 UTC].

| quantity | value |
|---|---|
| largest absolute loss difference, 16-bit against float32, by domain | [TBD] |
| leaves stored in bfloat16 / float16 (rule: the smaller relative L2 error; ties float16) | [TBD] |
| export size (bytes) and the `SHA256SUMS` of the weights | [TBD] |
| read-set audit of the export: numbers read per token, standard reading | [TBD] |

**Table M11 (R11). Table M1 and Table M4 on the 189 test prompts no mid-run sample touched** (the
`per_prompt_score` field of the R1 reports and the exact scores of R4, restricted to the 189 prompt ids,
the 11 seen ones listed in Section 7.2; [TBD: the restriction is not yet scripted]) [TBD: with R1 and R4].

| opponent | reader | n | W / L / T | score [95%] | sign test p |
|---|---|---|---|---|---|
| Falcon-H1-Tiny | `qwen3-4b-2507` | 189 | [TBD] | [TBD] | [TBD] |
| Falcon-H1-Tiny | `gemma4-e4b` | 189 | [TBD] | [TBD] | [TBD] |
| Falcon-H1-Tiny | Opus, two readings | 189 | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5 | `qwen3-4b-2507` | 189 | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5 | `gemma4-e4b` | 189 | [TBD] | [TBD] | [TBD] |
| MobileLLM-R1.5 | Opus, two readings | 189 | [TBD] | [TBD] | [TBD] |
| exact scores of ours, Falcon-H1-Tiny, MobileLLM-R1.5: maths correct; constraints met | – | – | [TBD] | – | – |

**Table M12 (R12). Decoding throughput of the export on CPU** (the test generation's summary
`testgen.json` and `testgen.log` in `N/testgen/TAG/`, fetched by `S/scripts/endgame_testgen.sh --fetch TAG`;
for comparison, on `dev`: 75 replies in 1,798 s with 16 workers, export and cache calibration included,
`FACTS_2.md`) [TBD: after the test generation, about 19:30 UTC].

| quantity | value |
|---|---|
| workers × cores per worker; host | [TBD] |
| replies; wall time, set-up and calibration included | [TBD] |
| seconds per reply (median); new tokens per reply (median) | [TBD] |
| seconds per decoding step, 6 particles, one core | [TBD] |

