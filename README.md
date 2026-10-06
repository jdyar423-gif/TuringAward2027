# Lookup–Compute Factorization (LCF): training a WikiText‑2 language model from scratch under a hard 128‑TFLOP budget

**Task.** Train from scratch on WikiText‑2, with a total budget of **1.28 × 10¹⁴ FLOPs**, and minimise test **bits‑per‑byte (BPB)**.

**Result.**

| | value |
|---|---|
| test BPB | **1.2275** (≈ 77.7 word‑level perplexity) |
| total compute | **1.2763 × 10¹⁴ FLOPs**, including all test‑time training (99.7 % of the budget) |
| standard AdamW Transformer, same budget | 1.5283 |

All numbers below were produced by the code in this repository on a 4‑core CPU. Each component has its own ablation.

> **What this is and isn't.** This is a carefully engineered and carefully *measured* system. It is not a Turing‑award result.
> Most ingredients are known; the contribution is:
> 1. how they are combined under a FLOP budget, with conservative, complete accounting;
> 2. a leave‑one‑document‑out estimator that lets a count‑memory gate be trained on training text and transfer to new articles (§3.4);
> 3. an exact Gram‑space form of the Polar‑Express/Muon iteration that cuts the optimizer's FLOPs by 34–44 % (§3.2);
> 4. a set of measured findings about this regime, including negative ones (§5).

---

## 1. Idea

Under a FLOP budget, a language model does two very different jobs:

1. **Memorising local co‑occurrence statistics.** What follows "*the United*"? Which name keeps recurring in this article?
2. **Composing context.** Syntax, agreement, topic.

A dense Transformer pays matrix‑multiply FLOPs for both. Job (1), however, can be served by **table lookups and integer counts**, which cost essentially **zero FLOPs**. LCF routes each kind of knowledge to the cheapest substrate that can hold it:

| tier | holds | substrate | FLOPs |
|---|---|---|---|
| **L0** input memory | hashed bigram/trigram features, injected into every layer | learned sparse tables (134 M params) | ≈ 0 |
| **L1** compute core | contextual composition | 6‑layer d=256 Transformer (4.7 M matmul params) | ~all of them |
| **L2** global memory | exact n‑gram statistics of the train split, orders 1–8 | integer counts | ≈ 0 |
| **L3** episodic memory | the current article so far, orders 1–6 | causal document cache | ≈ 0 |
| **L4** weight memory | adaptation to the test stream | score‑first dynamic evaluation | charged (13.7 %) |
| **gate** | which memory to trust in this context | 2.9 k‑param MLP on count features | ≈ 0 |

The output is a **proper mixture**:

p(y | ctx) = Σₑ wₑ(ctx) · pₑ(y | ctx).

Every expert is normalised over the vocabulary, and the weights depend on the context only, never on y.

## 2. Rules and FLOP accounting (conservative)

* **Data.** Only the WikiText‑2 *train* split is used for training. Validation is for model and hyperparameter selection. Test was evaluated once for the final system; baseline test numbers were computed afterwards, for the table only.
  * `huggingface.co` was blocked by this environment's network policy, so the experiments use the **standard WikiText‑2 (v1) release** mirrored in `pytorch/examples`. Its line counts (36 718 / 3 760 / 4 358) and the 245 569 test word tokens match the official release.
  * `prepare.py` switches to the raw `wiki.*.raw` files automatically if they are placed in `data/wikitext-2-raw/`. Numbers on the raw variant would differ.
* **BPB** = Σ −log₂ p(token) ⁄ (UTF‑8 bytes of the split file).
  * The tokenizer is a lossless byte‑level BPE with V = 4096, trained on the train split; the round‑trip is asserted.
  * The denominator therefore does not depend on the tokenizer, and every byte is predicted, including spaces and newlines.
* **Charged to the budget** (`flops.py`, `run_system.py`):
  * forward + backward matmul FLOPs, measured with `torch.utils.flop_counter`; they match the analytic count exactly;
  * attention, added analytically at the **full T×T cost (no causal discount)**, because the CPU flash kernel is invisible to the counter;
  * elementwise work (norms, activations, softmax, rotary, gates, loss, table scatter‑adds), using a generous upper bound;
  * all optimizer FLOPs: Newton–Schulz/Polar‑Express matmuls, every Adam update, and sparse table rows;
  * every forward pass on held‑out articles used to fit the gate, and the gate fitting itself;
  * every **test‑time backward pass and update**, and the mixture arithmetic;
  * a generous bound on the integer hashing, sorting and searching of the count memory, charged as if it were FLOPs.
* **Not charged** (as in any benchmark): forward passes that only *score* the evaluated split, and hyperparameter search.

### Final budget audit (`runs/f15_lremb3_ng4/system_final.json`)

| item | FLOPs | share |
|---|---:|---:|
| pre‑training: 594 steps × 1.816e11 | 1.0786e14 | 84.3 % |
|  · per step: matmul + attention | 1.611e11 | |
|  · per step: elementwise (upper bound) | 2.6e9 | |
|  · per step: optimizer (Gram Polar‑Express Muon, Adam, sparse rows) | 1.79e10 | 9.9 % of a step |
| held‑out train articles: forward passes for gate data | 1.63e12 | 1.3 % |
| gate fitting | 4.8e11 | 0.4 % |
| test‑time training: backward + updates, every 128 tokens | 1.757e13 | 13.7 % |
| mixture arithmetic on test | 2.9e9 | 0.0 % |
| count‑memory integer work (upper bound) | 8.8e10 | 0.1 % |
| **total** | **1.2763e14** | **99.7 %** |

## 3. Components

### 3.1 Compute core (`model.py`)
Pre‑RMSNorm Transformer, d=256, 6 layers, 4 heads, context 256, ReLU² MLP (4×), untied head. It includes:
* RoPE and QK‑norm;
* zero‑initialised output projections;
* x₀ embedding skip and value‑residual;
* tanh logit soft‑cap 15;
* three cheap gates from the 2025–26 speedrun literature: a smear gate, a sparse 12‑dim head‑wise attention‑output gate, and gated exclusive self‑attention (XSA).

Hidden matrices are trained with **Muon** using the **Polar‑Express** coefficients. Embeddings, head and scalars use Adam (β=(0.8, 0.95)). The schedule has 10 warm‑up steps, a flat phase, and a linear cooldown over the last 60 % to 0.1×. Batch is 16×256 tokens.

### 3.2 Gram‑space Polar Express (exact, 34–44 % fewer optimizer FLOPs) (`optim.py`)
Under honest accounting the Muon orthogonalisation is a first‑order cost: 10–25 % of all FLOPs at these sizes. Each Polar‑Express step X ← (aI + bA + cA²)X with A = XXᵀ is a polynomial in A, so the whole iteration can run in m×m Gram space:

```
A₀ = X₀X₀ᵀ ;  q_k = a_k I + b_k A_k + c_k A_k² ;  A_{k+1} = q_k² A_k ;  X_K = (q_{K-1} ⋯ q_0) X₀
```

* **Cost.** 4m²n + (8K−6)m³ instead of K(4m²n + 2m³).
* **Measured.** 2.83 vs 5.10 GFLOP for a 384×1536 matrix; the iterates agree to ≈1e‑5 relative error in fp32.
* **Selection.** The code picks the cheaper form for each matrix shape.
* **Effect.** Same model, same budget: more optimizer steps for the same FLOPs; val BPB **1.3971 → 1.3922**.

Similar Gram‑matrix rewrites of Newton–Schulz have been discussed elsewhere. What matters here is that under a FLOP budget the saving converts directly into training steps.

### 3.3 Hashed n‑gram input memory, L0 (`model.py`, `optim.py`)
* One table per order (2 and 3), 262 139 rows × 256, zero‑initialised.
* Each row is multiplied by a hashed ±1 sign vector, so colliding n‑grams decorrelate.
* The sum is injected into **every** layer's residual stream with learned λ's (BigramHash / Engram style).
* Trained with **sparse row‑Adam**: β₁=0, one second moment per row, only touched rows updated and charged.
* **Effect.** −0.126 BPB at ¼ budget; −0.058 BPB at full budget. This is the largest single neural gain.

### 3.4 Count memory with leave‑one‑document‑out calibration, L2 + L3 (`countmem.py`, `mixture.py`)
For each target and each context length k there are two experts:
* a **global expert**, C(c_k, y)/C(c_k), with counts over the train split, k = 0..7;
* a **document‑cache expert**, D(c_k, y)/D(c_k), with counts over the article so far, k = 0..5.

A gate (MLP, 64 hidden units) mixes these experts with the neural LM. Its inputs depend on the context only:
* log counts;
* numbers of distinct continuations (global and in‑document);
* recency of the context in the article;
* position in the article;
* the network's entropy and max‑probability.

**How the gate can be trained on training text.** On the train stream, the global counts are computed **leave‑one‑document‑out (LODO)**. The current article's own contribution is subtracted, including continuation types that occur only inside it. Together with the strictly causal document cache, this reproduces the test situation *exactly* on training data: a new article, global statistics from other articles, and a cache of the article so far.

* Count‑memory‑only gate fitted purely on LODO train statistics: **1.3779** val BPB.
* Gate fitted on validation itself: 1.3736. The transfer gap is 0.004 BPB.
* Final system: the gate is fitted on **3 % of training articles that the network never saw** (19 articles, 122 k tokens).
  * The network trains for <1 epoch anyway, so this costs it nothing.
  * The held‑out forward passes and the fitting are charged.
* The count memory is worth **−0.134 BPB on test** (1.3761 → 1.2420), at ~0 FLOPs.

### 3.5 Score‑first dynamic evaluation, L4 (`dyneval.py`)
The test stream is processed left to right. Every token is **scored before any update that could depend on it**. After each 128‑token chunk, one plain‑SGD step (lr 0.15, all parameters, including sparse table rows) is taken on that already‑scored chunk.
* With lr = 0 the scores reproduce static evaluation bit‑for‑bit (checked).
* The backward passes and updates are charged: 53.4 M FLOPs per test token, 1.757e13 in total.
* **Effect on test:** 1.2420 → **1.2275**.

## 4. Results

### 4.1 Main table: WikiText‑2 test, every row trained from scratch under the same 1.28e14 budget

| system | test BPB | ≈ word ppl | val BPB |
|---|---:|---:|---:|
| Transformer d256 L6, AdamW, GELU (standard recipe) | 1.5283 | 225.9 | 1.5432 |
| same + count memory (L2/L3) | 1.2869 | 96.0 | 1.2953 |
| Muon (Polar‑Express, Gram) + modern core | 1.4408 | 165.6 | 1.4576 |
| same + count memory | 1.2575 | 86.5 | — |
| **LCF network** (core + hashed n‑gram memory + gates) | 1.3761 | 131.7 | 1.3876 |
| + count memory and gate (static) | 1.2420 | 81.8 | 1.2482 |
| **+ score‑first dynamic evaluation = final LCF system** | **1.2275** | **77.7** | **1.2355** |

*Word perplexity is exp(BPB · bytes · ln 2 / words) with 245 569 test words + eos. It is an upper‑bound conversion: the model also pays for spaces and newlines.*

For scale, AWD‑LSTM reaches 65.8 test perplexity on the same benchmark (44.3 with dynamic evaluation) using roughly three orders of magnitude more compute (≈ 750 epochs of a 33 M‑parameter model). The two models are not directly comparable: word vocabulary vs. bytes.

### 4.2 Ablation of the network (val BPB, network alone, full budget, d256 L6)

| step | val BPB | Δ |
|---|---:|---:|
| AdamW + GELU, no QK‑norm / x₀ / value‑residual / soft‑cap | 1.5432 | |
| Muon (Polar‑Express, Gram) + ReLU², QK‑norm, x₀‑mix, value‑residual, soft‑cap | 1.4576 | −0.086 |
| + hashed bigram/trigram memory (sparse row‑Adam) | 1.3992 | −0.058 |
| + smear gate, sparse attention gate, gated XSA | 1.3886 | −0.011 |
| + embedding/table learning rates ×2 | 1.3878 | −0.001 |

Notes:
* The AdamW baseline used one reasonable setting (lr 2e‑3, wd 0.1). It was not tuned as heavily as the Muon runs.
* All results are single‑seed. Treat differences below ~0.005 as noise.

### 4.3 Model size under a fixed budget (val BPB, network alone)

| d | L | batch | steps | epochs | optimizer share of FLOPs | val BPB |
|---|---|---|---:|---:|---:|---:|
| 256 | 4 | 16 | 789 | 1.10 | 13.9 % | 1.4786 |
| 256 | 4 | 32 | 423 | 1.18 | 7.4 % | 1.4939 |
| 320 | 4 | 16 | 518 | 0.72 | 17.7 % | 1.4082 |
| 384 | 4 | 16 | 361 | 0.50 | 21.4 % | 1.4496 |
| 256 | 6 | 16 | 562 | 0.78 | 14.8 % | 1.3971 (standard NS) |
| **256** | **6** | 16 | 594 | 0.83 | 9.9 % | **1.3922** (Gram NS) |
| 320 | 6 | 16 | 391 | 0.55 | 12.7 % | 1.4380 |
| 256 | 8 | 16 | 462 | 0.64 | 10.2 % | 1.4216 |

The optimum is the largest model that still sees **just under one epoch**. Two forces bound it:
* Larger models get too few tokens and steps.
* Crossing into a second epoch makes training loss collapse (3.6 → 2.65 nats) through memorisation, and validation gets worse.

### 4.4 Test‑time training inside the mixture (val BPB, final network)

| schedule | network alone | full mixture | charged FLOPs / token |
|---|---:|---:|---:|
| static | 1.3876 | 1.2482 | 0 |
| update every 128 tokens, lr 0.5 | 1.3265 | 1.2434 | 53.4 M |
| update every 128 tokens, lr 0.3 | 1.3270 | 1.2377 | 53.4 M |
| update every 128 tokens, lr 0.2 | 1.3312 | 1.2357 | 53.4 M |
| **update every 128 tokens, lr 0.15** | 1.3357 | **1.2355** | 53.4 M |
| update every 256 tokens (earlier model) | −0.043 vs static | only −0.0015 … −0.0045 | 26.7 M |

The mixture **compresses** neural gains by 3–7×, because the document cache already captures much of what adaptation provides. Only frequent updates survive the compression. The mixture also prefers gentler learning rates than the network alone does.

### 4.5 Count memory variants (val BPB, count memory only, gate fitted on LODO train statistics)

| variant | val | gate fitted on val (oracle) |
|---|---:|---:|
| orders ≤6 global / ≤4 document, basic features | 1.3862 | 1.3828 |
| orders ≤8 / ≤6, + in‑document distinct continuations + recency | 1.3779 | 1.3736 |
| same, BPE vocabulary 8192 | 1.3764 | 1.3719 |

## 5. What did not work (measured negative results)

* **Training the network jointly through the mixture**, so that it learns only the residual: 1.3684 vs 1.3158 post‑hoc (¼ budget). With an auxiliary CE loss it is merely equal (1.3167). The post‑hoc gate wins.
* **Going past one epoch.** The network memorises immediately. Freezing the n‑gram tables after epoch 1 does **not** prevent it (1.5126), so the dense core memorises, keyed by the very specific n‑gram features.
* **Bigger hashed tables** (1 M rows) or **adding 4‑grams**: no gain at this data size (1.5503 / 1.5511 vs 1.5498).
* **Value embeddings + U‑net skips**: −0.004 at ¼ budget, within noise. Not adopted.
* **Tail EMA / weight averaging**: 1.3930 vs 1.3922. No gain at ~600 steps.
* **Higher Muon learning rate** (0.045): 1.3981, worse. **Batch 32**: worse. **3 Polar‑Express steps**: worse (confounded with crossing one epoch).
* **A larger gate** (128 hidden, 800 steps) overfits: 1.3150 vs 1.3083. The cheaper final gate fit (150 steps on stride‑256 held‑out statistics) also beat 400 steps on stride‑128 statistics (1.2506 vs 1.2535 val) at less than half the FLOPs. Those two settings were changed together, so the cause is not isolated.
* **RMSprop‑style dynamic evaluation** was worse than plain SGD (3.859 vs 3.781 nats/token on a proxy). Adapting only the top block or the head is cheaper but loses most of the gain.
* **A longer scoring context** (stride 64 vs 128) changes nothing. The network does not exploit context beyond ~128 tokens; the document cache does.

## 6. Reproduce

```bash
pip install torch numpy
bash scripts/reproduce.sh      # trains the network (≈15 min on 4 idle cores; up to 40 min when sharing them), then runs the full system on valid+test
```

* The final run's metrics are in `runs/f15_lremb3_ng4/result.json` and `system_final.json`. Every experiment above has its `runs/*/result.json` and `logs/*.log`.
* The analysis scripts behind §4.4–4.5 and §5 are in `experiments/analysis/`.

## 7. Files

| file | purpose |
|---|---|
| `prepare.py` | data, lossless byte‑level BPE, token streams |
| `model.py` | compute core, hashed n‑gram memory, gates |
| `optim.py` | Muon with Polar Express (standard and Gram‑space), sparse row‑Adam |
| `flops.py` | FLOP accounting |
| `train.py` | budgeted training (stops exactly at the budget), held‑out articles, schedules, AdamW baseline |
| `countmem.py` | count memory: hashing, LODO statistics, document cache, features |
| `mixture.py` | gate and statistics extraction |
| `dyneval.py` | score‑first dynamic evaluation with FLOP charging |
| `run_system.py` | full system: gate fitting on held‑out articles, evaluation, final budget audit |

## 8. Limitations

* WikiText‑2 v1 (with `<unk>`) rather than the raw variant, for the network‑access reason above.
* Single seeds throughout.
* Hyperparameters were tuned on validation using many more FLOPs than the budget, which is standard practice and not counted.
* The 134 M‑parameter hashed tables and the count memory make the system **cheap in FLOPs but not in memory** (~0.6 GB of tables plus the count index). That trade is the point of the design, but it matters if memory, rather than FLOPs, is the binding constraint.
