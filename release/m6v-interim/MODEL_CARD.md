# Model card: BES-100M-A20M, export of run `main16`

Written by `bes/export.py` (export format 1) on 2026-11-24 18:37:02 UTC from the files of this run and measurements made during the export. Nothing here is a projection; a fact the run did not record is marked as missing.

## What this is

A looped transformer: d_model 768, 2 shared blocks applied 8 times, 6 heads of 128, feed-forward width 3072, vocabulary 8192, context 2048 tokens (`docs/PLAN.md`). Every vocabulary id owns a page of 8 private feed-forward neurons per block in a 192-dimensional space shared by all tokens and gains on the attention outputs of the positions that hold it, read once per token (`docs/V9.md`). Feature paths on: registers, candidate-tape input features (n-gram count heads), candidate-tape masses in the output law, calibrated attention sinks, per-token gains, token neurons. Switched off: the confidence channel, structural clocks. A written answers library ships with it (questions with the answers teacher models wrote, stored as text, not trained): a question it holds is answered with the stored answer, and similar entries are shown to the model as exemplars before the user's turn; on by default. Replies are decoded with K = 6 particles steered by exact potentials and chosen by a satisficing rule.

- checkpoint: `/content/runs/main16/train/ckpt/ckpt-000022-step-000051812` (step 51812, train weights)
- configuration hash f29bfeb3ebf353a8; configuration file `configs/v9.py`, overrides `['model.table_grad=take', 'run.bench_plans=reduced', 'run.post_minutes=critic=0,buffer=20', 'run.post_reserve_hours=3.6667', 'optim.micro_seqs=16', 'run.deadline_hours=21.328333333333333', 'run.seed=1']`
- code commit 98598c212bca1cfd7d4108d4eba43f0cfba91f96 (uncommitted changes: False)

## Stored weights (at most 16 bits per number)

| format | leaves | numbers | largest relative L2 error of a leaf |
|---|---|---|---|
| float16 | 64 | 74,566,024 | 0.000276 |
| bfloat16 | 0 | 0 | 0.000000 |
| total | 64 | 74,574,216 | |

Each leaf is stored in the format with the smaller measured relative error (`weights.json` lists both errors per leaf); the loader restores float32.

### Held-out check

Mean cross-entropy on 43,138 held-out tokens (float32 activations, dense attention, highest matmul precision): float32 weights 2.32877 nats, 16-bit weights 2.32881 nats, difference +0.000043 nats (tolerance 0.002): **passed**.

| domain | weighted tokens | float32 | 16-bit | difference |
|---|---|---|---|---|
| chat | 15,690 | 1.30512 | 1.30508 | -0.000040 |
| proxy | 11,094 | 3.25803 | 3.25813 | +0.000099 |
| web | 16,354 | 2.68048 | 2.68057 | +0.000084 |

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

- budget 21.33 h of wall time from 2026-11-23 17:34 UTC; 14 resumes
- devices: ['TPU v6 lite'] (jax 0.11.2, backend tpu)
- trained 13,582,204,928 tokens in 51,812 optimiser steps; 17.47 h in the step loop; set-up 4.7 min, checkpoints 23 s, lost to restarts and rollbacks 7.24 h; 10 restarts, 0 rollbacks
- execution plan splash-remat1-ms16-take (208,837 tokens/s expected), horizon 1.312e+10 tokens
- fallback ladder rung 0 (full configuration)
- canary: parity True, determinism True, safe run True, full run True

| stage | status | planned | used | note |
|---|---|---|---|---|
| preflight | finished | 3.0 min | 0 s |  |
| background | finished | 1.0 min | 0 s |  |
| benchmark | finished | 15.0 min | 11.6 min |  |
| train | finished | 19.05 h | 18.11 h | interrupted 10x |
| sample | started | 3.0 min | 0 s |  |
| critic | pending | - | 0 s |  |
| prefopt | pending | - | 0 s |  |
| select | pending | - | 0 s |  |
| judge | pending | - | 0 s |  |
| export | pending | - | 0 s |  |

## Post-training and evaluation

- preference optimisation: no record
- decoding choice: no record (the run's own `DecodeCfg` is the default)
- judged comparison: no record

Reports copied into `reports/`: `reports/manifest.json`, `reports/stages.json`, `reports/stage_table.json`, `reports/train_ledger.json`, `reports/train_run.json`, `reports/health_latest.json`

## Disclosures (`docs/PLAN.md` section 2)

- **Library**: not shipped; no decode selection recorded
- **Inference-time compute**: K = 6 samples of this model per reply (particle decoding with exact potentials, then satisficing selection); the decode selection compares K = 1 when K > 1
- **External models**: scoring only: the judge `qwen3-4b-2507` rates blinded pairs and the opponents (falcon=recommended mobilellm=recommended mobilellm=neutral) are only compared against; no training step reads another model's logits (no distillation); run.use_external_models = True
- **Preference data**: no preference optimisation record

## Data, licences and attribution

| training source | dataset | licence (dataset card) | non-permissive | tokenised in this run |
|---|---|---|---|---|
| climbmix | karpathy/climbmix-400b-shuffle | card: mit; upstream nvidia/Nemotron-ClimbMix: cc-by-nc-4.0 | True | True |
| smol_smoltalk | HuggingFaceTB/smol-smoltalk | apache-2.0 | False | True |
| smoltalk_magpie_ultra | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| tulu3_sft | allenai/tulu-3-sft-mixture | odc-by-1.0 (subsets carry their own licences; the CC BY-NC No Robots subset is excluded) | False | True |
| synth | PleIAs/SYNTH | cc-by-4.0 (card); seed text CC BY-SA 4.0 / CC0 (column seed_license) | False | True |
| ultrachat | HuggingFaceH4/ultrachat_200k | mit | False | True |
| smoltalk_constraints | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_rewrite | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_summarize | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_everyday | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
| smoltalk_systemchats | HuggingFaceTB/smoltalk | apache-2.0 (card text; the YAML header has no licence field) | False | True |
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

Models used for evaluation only (never trained on, never shipped):
- falcon (opponent): tiiuae/Falcon-H1-Tiny-90M-Instruct, file tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF/Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf, licence Falcon-LLM License
- mobilellm (opponent): facebook/MobileLLM-R1.5-140M (gated), file sasa2000/MobileLLM-R1.5-140M-Q8_0-GGUF/mobilellm-r1.5-140m-q8_0.gguf, licence FAIR Noncommercial Research License
- qwen3-4b-2507 (judge): Qwen/Qwen3-4B-Instruct-2507, file unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf, licence Apache-2.0

## Evicted token rows

The page and gain rows (`gain/o`, `page/key`, `page/val`) of 3,056 of the 8,192 vocabulary ids are not stored: the least frequent ones (the 5120 most frequent ids in the questions and answers of the full answers library, and the 16 special tokens of the chat format; the evicted ids are 2.86% of those occurrences). `page/rows` (uint16, one entry per id) gives a token its row or none; a token without a row reads zeros, i.e. no gain and no private neurons, and the loader restores the tables of the configuration with those rows at zero. Stored numbers freed: 23,470,080 (the map costs 8,192). The input embedding and the per-loop value embedding are whole.

Held-out loss of the 16-bit weights before the eviction 2.00707, after 2.03375 nats per token (32044.0 tokens; by domain: {'chat': [1.30508, 1.33163], 'web': [2.68057, 2.70736]}).

Check made when the rows were evicted: the weights of this directory load to exactly the source's weights with those rows zeroed (64 leaves compared bit for bit). Source export: weights16.npz SHA-256 `634230225867f197ed75593680c6dce3a3828e3d404787573559f2a510a9073a`.

## Files

SHA-256 of every file is in `SHA256SUMS` (`sha256sum -c SHA256SUMS` or `python scripts/export.py --verify <dir>`). Main files:

| file | bytes | sha256 |
|---|---|---|
| `weights16.npz` | 149,165,278 | `a6eab1212a58613f...` |
| `weights.json` | 18,096 | `804f869c200fa862...` |
| `config.json` | 5,178 | `5122de345e17f168...` |
| `tokenizer/tokenizer.json` | 129,787 | `78a7392833c33f1e...` |

Load for inference: `from bes.export import load_export; m = load_export(dir)`; chat: `python scripts/chat.py --export <dir>`; blind arena: `python scripts/arena.py --export <dir>`.
