# Model card: BES-100M-A20M, export of run `m6v`

Written by `bes/export.py` (export format 1) on 2026-11-25 21:56:02 UTC from the files of this run and measurements made during the export. Nothing here is a projection; a fact the run did not record is marked as missing.

## What this is

A looped transformer: d_model 768, 2 shared blocks applied 8 times, 6 heads of 128, feed-forward width 3072, vocabulary 8192, context 2048 tokens (`docs/PLAN.md`). Every vocabulary id owns a page of 8 private feed-forward neurons per block in a 192-dimensional space shared by all tokens and gains on the attention outputs of the positions that hold it, read once per token (`docs/V9.md`). Feature paths on: registers, candidate-tape input features (n-gram count heads), candidate-tape masses in the output law, calibrated attention sinks, per-token gains, token neurons. Switched off: the confidence channel, structural clocks. A written answers library ships with it (questions with the answers teacher models wrote, stored as text, not trained): a question it holds is answered with the stored answer, and similar entries are shown to the model as exemplars before the user's turn; on by default. Replies are decoded with K = 6 particles steered by exact potentials and chosen by a satisficing rule.

- checkpoint: `/content/runs/m6v/post/ckpt/ckpt-000001-step-000013615` (step 13615, prefopt weights)
- configuration hash 528ce8bec57c797d; configuration file `configs/m6v.py`, overrides `['run.deadline_hours=5.605', 'model.attn_impl=splash', 'model.remat=true', 'optim.micro_seqs=16', 'model.table_grad=take']`
- code commit 7df4cca81920b3eef0f807abd7f2cec62897eea3 (uncommitted changes: False)

## Stored weights (at most 16 bits per number)

| format | leaves | numbers | largest relative L2 error of a leaf |
|---|---|---|---|
| float16 | 64 | 74,566,024 | 0.000325 |
| bfloat16 | 0 | 0 | 0.000000 |
| total | 64 | 74,574,216 | |

Each leaf is stored in the format with the smaller measured relative error (`weights.json` lists both errors per leaf); the loader restores float32.

### Held-out check

Mean cross-entropy on 43,138 held-out tokens (float32 activations, dense attention, highest matmul precision): float32 weights 2.31086 nats, 16-bit weights 2.31096 nats, difference +0.000098 nats (tolerance 0.002): **passed**.

| domain | weighted tokens | float32 | 16-bit | difference |
|---|---|---|---|---|
| chat | 15,690 | 1.26823 | 1.26818 | -0.000046 |
| proxy | 11,094 | 3.10833 | 3.10875 | +0.000427 |
| web | 16,354 | 2.77019 | 2.77020 | +0.000012 |

Held-out data:
- chat: HuggingFaceTB/smol-smoltalk/data/test-00000-of-00001.parquet, 16 rows of 1024 tokens (the test split; the registry reads the train files only)
- proxy: bes/eval/proxy/prompts.jsonl (user turns), 12 rows of 1024 tokens
- web: karpathy/climbmix-400b-shuffle/shard_00997.parquet, 16 rows of 1024 tokens (beyond the run's ClimbMix download cap (files are taken in listing order up to the cap))

## Accounting (`bes.budget`, real library sizes)

| row | stored numbers | numbers read per generated token |
|---|---|---|
| core | 14,118,912 | 14,118,912 |
| head | 4,587,520 | 4,587,520 |
| small | 306,776 | 306,776 |
| emb | 6,291,456 | 768 |
| ve | 8,388,608 | 1,024 |
| gain | 7,888,896 | 1,536 |
| page | 32,153,600 | 595,969 |
| tables | 838,448 | 18,736 |
| library | 21,797,551 | 96,248 |
| index | 3,571,253 | 263,752 |
| **total** | **99,943,020** | **19,991,241** |

Caps 100,000,000 stored / 20,000,000 read per token: within both. The per-token column is the standard convention of `docs/ACCOUNTING.md`: every dense leaf, one row per lookup read, and the worst-case read of one library retrieval charged to the step that retrieves.

Answers library: 172,273 question-answer entries; 25,368,804 stored numbers (compressed text and its directory 21,453,004, per-entry rows 344,547, index 3,571,253); one reply reads at most 360,000 of them (a cap the read log enforces).

## Read-set audit (`bes.readset.ReadLogger`)

Distinct stored numbers read for each generated token on held-out sequences with their real exact-layer features. *step*: the standard convention (given the cache); *strict*: the union over the visible context window.

| probe | positions | distinct tokens | step mean | step max | strict at the end | strict max |
|---|---|---|---|---|---|---|
| web (held-out record) | 2048 | 883 | 19,427,809 | 19,429,609 | 27,929,250 | 27,929,250 |
| chat (held-out record) | 1480 | 560 | 19,427,956 | 19,429,996 | 24,793,944 | 24,794,399 |
- web (held-out record), window of 1024 positions: strict 24,352,024 (512 distinct tokens)
- web (held-out record), window of 2048 positions: strict 27,929,250 (883 distinct tokens)
- chat (held-out record), window of 1024 positions: strict 23,363,294 (410 distinct tokens)
The critic head is not read (the decoder does not use it).

## Training run

- budget 5.61 h of wall time from 2026-11-25 16:25 UTC; 0 resumes
- devices: ['TPU v6 lite'] (jax 0.11.2, backend tpu)
- trained 3,569,090,560 tokens in 13,615 optimiser steps; 5.26 h in the step loop; set-up 28 s, checkpoints 19 s, lost to restarts and rollbacks 0 s; 0 restarts, 0 rollbacks
- execution plan splash-remat1-ms16-take (167,330 tokens/s expected), horizon 3.174e+09 tokens
- fallback ladder rung 0 (full configuration)
- canary: parity True, determinism True, safe run True, full run True

| stage | status | planned | used | note |
|---|---|---|---|---|
| preflight | finished | 1.0 min | 2 s |  |
| background | finished | 1.0 min | 0 s |  |
| benchmark | finished | 5.0 min | 3.6 min |  |
| train | finished | 5.22 h | 5.28 h |  |
| sample | skipped | - | 0 s | planned with 0 minutes (run.post_minutes) |
| critic | skipped | - | 0 s | planned with 0 minutes (run.post_minutes) |
| prefopt | finished | 12.0 min | 9.6 min |  |
| select | skipped | - | 0 s | planned with 0 minutes (run.post_minutes) |
| judge | skipped | - | 0 s | planned with 0 minutes (run.post_minutes) |
| export | started | 2.0 min | 0 s |  |

## Post-training and evaluation

- preference optimisation: accepted (lr 3e-06, 23172 training pairs, 1823 held out)
- decoding choice on the dev split: `None` by exact scores {} over None common prompts of None
- library decision (None): not shipped: the decode selection made no library decision (no measurement, no gain)
- judged comparison: no record

Reports copied into `reports/`: `reports/manifest.json`, `reports/stages.json`, `reports/train_ledger.json`, `reports/train_run.json`, `reports/prefopt_report.json`, `reports/health_latest.json`

## Disclosures (`docs/PLAN.md` section 2)

- **Library**: not shipped (the decode selection's library rule did not pass: the decode selection made no library decision (no measurement, no gain)); library decision: not shipped: the decode selection made no library decision (no measurement, no gain); decode selection variants measured on dev: []
- **Inference-time compute**: K = 6 samples of this model per reply (particle decoding with exact potentials, then satisficing selection); the decode selection compares K = 1 when K > 1
- **External models**: scoring only: the judge `` rates blinded pairs and the opponents () are only compared against; no training step reads another model's logits (no distillation); run.use_external_models = True
- **Preference data**: DPO pairs by source: {'heldout': 1823, 'sampled': {'duplicate': 0, 'identical': 0, 'invalid': 0, 'kept': 0, 'length_ratio': 0, 'too_long': 0}, 'train': 35184, 'ultrafeedback': {'duplicate': 0, 'identical': 306, 'invalid': 0, 'kept': 37007, 'length_ratio': 8366, 'too_long': 7995}}; sampled pairs are labelled by the exact verifiers; UltraFeedback pairs (when used) carry labels produced by third-party models (dataset card)

## Data, licences and attribution

| training source | dataset | licence (dataset card) | non-permissive | tokenised in this run |
|---|---|---|---|---|
| climbmix | karpathy/climbmix-400b-shuffle | card: mit; upstream nvidia/Nemotron-ClimbMix: cc-by-nc-4.0 | True | True |
| magpie_llama31 | Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1 | llama3.1 (card): outputs of Llama 3.1 70B under the Llama 3.1 Community License | False | True |
| smoltalk_magpie_ultra | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| tulu3_sft | allenai/tulu-3-sft-mixture | odc-by-1.0 (subsets carry their own licences; the CC BY-NC No Robots subset is excluded) | False | True |
| smol_smoltalk | HuggingFaceTB/smol-smoltalk | apache-2.0 | False | True |
| openhermes | teknium/OpenHermes-2.5 | no licence field on the card (a compilation of sets with their own terms) | True | True |
| magpie_phi3 | Magpie-Align/Magpie-Phi3-Pro-1M-v0.1 | no licence field on the card (outputs of Phi-3 medium, an MIT model) | True | True |
| synth | PleIAs/SYNTH | cc-by-4.0 (card); seed text CC BY-SA 4.0 / CC0 (column seed_license) | False | True |
| ultrachat | HuggingFaceH4/ultrachat_200k | mit | False | True |
| smoltalk_constraints | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_summarize | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_rewrite | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| glaive_code | glaiveai/glaive-code-assistant | apache-2.0 | False | True |
| smoltalk_systemchats | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| gpt4all | nomic-ai/gpt4all-j-prompt-generations | apache-2.0 | False | True |
| wizardlm | WizardLMTeam/WizardLM_evol_instruct_V2_196k | mit | False | True |
| smoltalk_everyday | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| alpaca | yahma/alpaca-cleaned | cc-by-4.0 | False | True |
| codealpaca | sahil2801/CodeAlpaca-20k | cc-by-4.0 | False | True |
| synth_rag | PleIAs/SYNTH | cc-by-4.0 (card); seed text CC BY-SA 4.0 / CC0 (column seed_license) | False | True |
| glaive_rag | glaiveai/RAG-v1 | apache-2.0 | False | True |
| rag_instruct | FreedomIntelligence/RAG-Instruct | apache-2.0 | False | True |
| squad_v2 | rajpurkar/squad_v2 | cc-by-sa-4.0 | False | True |
| hotpot_qa | hotpotqa/hotpot_qa | cc-by-sa-4.0 | False | True |
| dolly | databricks/databricks-dolly-15k | cc-by-sa-3.0 | False | True |
| neural_bridge_rag | neural-bridge/rag-dataset-12000 | apache-2.0 (the card also asks users to abide by the terms of use of the web corpus it was drawn from) | False | True |
| orca_math | microsoft/orca-math-word-problems-200k | mit | False | True |
| gsm8k | openai/gsm8k | mit | False | True |
| fineweb_edu | karpathy/fineweb-edu-100b-shuffle | odc-by | False | True |
| simplewiki | wikimedia/wikipedia | cc-by-sa-3.0 and gfdl (card) | False | True |
| ultrafeedback | HuggingFaceH4/ultrafeedback_binarized | mit | False | True |

Caveats per source (synthetic outputs of third-party models, upstream terms): `docs/DATA_SOURCES.md` section 4.

Tokenizer trained on: karpathy/climbmix-400b-shuffle ['shard_00000.parquet', 'shard_00001.parquet']; HuggingFaceTB/smol-smoltalk ['data/train-00000-of-00004.parquet']; wikimedia/wikipedia ['20231101.simple/train-00000-of-00001.parquet']

Answers library (copy, digest f81b5936aed288fe...): the question-answer pairs below, from datasets whose cards state a licence; attribution in `ATTRIBUTION.md` of the library directory (copied next to this card as `LIBRARY_ATTRIBUTION.md`).

| question-answer source (stored in the library) | dataset | config / split | licence (dataset card) | permissive | entries |
|---|---|---|---|---|---|
| alpaca | yahma/alpaca-cleaned | default/train/*.parquet / train | cc-by-4.0 | True | 7,618 |
| codealpaca | sahil2801/CodeAlpaca-20k | default/train/*.parquet / train | cc-by-4.0 | True | 4,295 |
| dolly | databricks/databricks-dolly-15k | default/train/*.parquet / train | cc-by-sa-3.0 | True | 2,748 |
| glaive_code | glaiveai/glaive-code-assistant | default/train/*.parquet / train | apache-2.0 | True | 27,176 |
| gpt4all | nomic-ai/gpt4all-j-prompt-generations | data/train-*.parquet / train | apache-2.0 | True | 56,621 |
| magpie_llama31 | Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1 | data/train-*.parquet / train | llama3.1 (Llama 3.1 Community License: outputs of Llama 3.1 70B) | True | 24,979 |
| magpie_ultra | HuggingFaceTB/smoltalk | data/smol-magpie-ultra/train-*.parquet / train | apache-2.0 | True | 14,781 |
| smol_smoltalk | HuggingFaceTB/smol-smoltalk | data/train-*.parquet / train | apache-2.0 | True | 10,022 |
| tulu3 | allenai/tulu-3-sft-mixture | data/train-*.parquet / train | odc-by-1.0 (subsets carry their own licences; No Robots, Aya and the hard-coded identity prompts are left out) | True | 13,733 |
| ultrachat | HuggingFaceH4/ultrachat_200k | data/train_sft-*.parquet / train | mit | True | 8,134 |
| wizardlm | WizardLMTeam/WizardLM_evol_instruct_V2_196k | default/train/*.parquet / train | mit | True | 2,166 |

No build-settings record came with the library (built outside the run's library task): its tokenizer and `LibraryCfg` were not checked against the run's.

## Evicted token rows

The page and gain rows (`gain/o`, `page/key`, `page/val`) of 3,056 of the 8,192 vocabulary ids are not stored: the least frequent ones (the 5120 most frequent ids in the questions and answers of the full answers library, and the 16 special tokens of the chat format; the evicted ids are 2.86% of those occurrences). `page/rows` (uint16, one entry per id) gives a token its row or none; a token without a row reads zeros, i.e. no gain and no private neurons, and the loader restores the tables of the configuration with those rows at zero. Stored numbers freed: 23,470,080 (the map costs 8,192). The input embedding and the per-loop value embedding are whole.

Held-out loss of the 16-bit weights before the eviction 2.03475, after 2.07070 nats per token (32044.0 tokens; by domain: {'chat': [1.26818, 1.3077], 'web': [2.7702, 2.80272]}).

Check made when the rows were evicted: the weights of this directory load to exactly the source's weights with those rows zeroed (64 leaves compared bit for bit). Source export: weights16.npz SHA-256 `0d6705884f9fb602fd56dd511187565cb9fa6030a31b723d57ea2da80417a5dc`.

## Files

SHA-256 of every file is in `SHA256SUMS` (`sha256sum -c SHA256SUMS` or `python scripts/export.py --verify <dir>`). Main files:

| file | bytes | sha256 |
|---|---|---|
| `weights16.npz` | 149,165,278 | `ed481dbc1dde1fcd...` |
| `weights.json` | 18,063 | `93cc9f98d667a846...` |
| `config.json` | 6,035 | `6de6a48eb12c68e3...` |
| `tokenizer/tokenizer.json` | 129,787 | `78a7392833c33f1e...` |

Load for inference: `from bes.export import load_export; m = load_export(dir)`; chat: `python scripts/chat.py --export <dir>`; blind arena: `python scripts/arena.py --export <dir>`.
