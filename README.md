# Lookup–Compute Factorization: a language model trained from scratch on WikiText‑2 under a hard 128‑TFLOP budget

**Task.** Train from scratch on WikiText‑2 with a total budget of **1.28 × 10¹⁴ FLOPs**, and minimise test **bits‑per‑byte (BPB)**.

**Result (WikiText‑2 test, all compute included):** see [Results](#results). The final system reaches
**⟨TEST_BPB⟩ BPB** at **⟨TOTAL_FLOPS⟩ FLOPs**. A standard modern Transformer trained on the same budget reaches ⟨BASE_BPB⟩.

> Honesty note. This is a carefully engineered and measured system. It is not a "Turing‑award" result. What is
> new is mostly *how known pieces are combined and accounted for under a FLOP budget*, plus two small technical
> contributions (§3.2 and §3.4). Every number below comes from code in this repository.

---

## 1. The idea in one paragraph

Under a FLOP budget, a language model has two jobs that cost very differently:
1. **Memorising local co‑occurrence statistics** ("what usually follows *the United*").
2. **Composing context** (agreement, topic, long‑range structure).

A dense Transformer pays matrix‑multiply FLOPs for both. But (1) can be served by *table lookups and integer counts*,
which cost essentially **zero FLOPs**. **Lookup–Compute Factorization (LCF)** therefore routes every kind of
knowledge to the cheapest substrate that can hold it:

| tier | what it stores | substrate | FLOPs |
|---|---|---|---|
| L0 input memory | hashed bigram/trigram embeddings (Engram‑style), injected into every layer | learned sparse tables, 134 M params | ≈ 0 |
| L1 compute core | contextual composition | 6‑layer, d=256 Transformer, 4.7 M matmul params | almost all of them |
| L2 output memory | exact n‑gram statistics of the training set (orders 1–8) | integer counts | ≈ 0 |
| L3 episodic memory | the current article so far (orders 1–6) | causal document cache | ≈ 0 |
| L4 weight memory | test‑time adaptation | score‑first dynamic evaluation | charged |
| gate | which memory to trust, given the context | 2.9 k‑parameter MLP over count features | ≈ 0 |

The final distribution is a **proper mixture**:

```
p(y | ctx) = Σ_e w_e(ctx) · p_e(y | ctx).
```

Each expert p_e is normalised over the vocabulary. The weights w(ctx) depend only on the context, never on y.

## 2. Rules and FLOP accounting (conservative)

* **Data.** WikiText‑2 train split only. Validation is used for model selection, and test is reported once.
  The environment's network policy blocked `huggingface.co`, so the experiments use the standard WikiText‑2 (v1) release, mirrored in
  `pytorch/examples` (line counts 36 718 / 3 760 / 4 358 match the official splits). `prepare.py` automatically
  uses the raw `wiki.*.raw` files instead if they are placed in `data/wikitext-2-raw/`.
* **BPB.** BPB = Σ −log₂ p(token) / (UTF‑8 bytes of the split file). The tokenizer is a lossless byte‑level BPE
  (vocabulary 4096) trained on the train split, and the round‑trip is asserted. The denominator is therefore independent of the tokenizer.
* **Everything that trains is counted** (`flops.py`):
  * Matmul FLOPs of forward and backward are *measured* with `torch.utils.flop_counter`, and match the analytic count exactly.
  * Attention is added analytically at the full T×T cost, with no causal discount. The CPU flash kernel is invisible to the counter.
  * Elementwise operations (norms, activations, softmax, rotary, gates, loss, table scatter‑adds) use a generous upper bound.
  * Optimizer FLOPs are counted exactly. This covers Newton–Schulz/Polar‑Express matmuls, every Adam update, and sparse table rows.
  * The EMA is counted.
* **Charged after training:**
  * forward passes on held‑out train articles used to fit the gate, and the gate fitting itself;
  * every test‑time backward pass and update;
  * the mixture arithmetic;
  * a generous bound on the integer hashing, sorting and searching of the count memory, charged as if it were FLOPs.
* **Not charged (as in every benchmark):** the forward passes that only *score* the evaluated split, and hyperparameter search.

## 3. Components

### 3.1 Compute core (`model.py`, `optim.py`)
Pre‑RMSNorm Transformer with:
* RoPE and QK‑norm;
* ReLU² MLP;
* zero‑initialised output projections;
* embedding skip (x₀‑mix) and value‑residual;
* tanh logit soft‑cap 15;
* untied head;
* the cheap 2025–26 gates (smear gate, sparse 12‑dim head‑wise attention‑output gate, gated exclusive self‑attention).

Hidden matrices are trained with **Muon** using **Polar‑Express** coefficients. Embeddings, head and scalars use Adam.

### 3.2 Gram‑space Polar Express (new, exact, ~35–45 % cheaper)
At this scale the Muon orthogonalisation is a first‑order cost: **12–25 % of every step's FLOPs**. Each Polar‑Express step
X ← (aI + bA + cA²)X with A = XXᵀ is a polynomial in A, so it can be run entirely in m×m Gram space:

```
A₀ = X₀X₀ᵀ;   q_k = a_k I + b_k A_k + c_k A_k²;   A_{k+1} = q_k² A_k;   Q = Π q_k;   X_K = Q X₀
```

This costs 4m²n + (8K−6)m³ instead of K(4m²n + 2m³). It produces the same iterates (relative difference ≈ 1e‑5 in fp32).
Measured cost: 2.83 vs 5.10 GFLOP for a 384×1536 matrix. The code uses whichever form is cheaper for each shape.
At equal FLOPs this alone improved validation BPB by 0.005.

### 3.3 Hashed n‑gram input memory (L0)
* One 262 139‑row table per order (2, 3), 256 wide, zero‑initialised.
* Rows are decorrelated by a hashed ±1 sign vector.
* The result is added to the residual stream of every layer through learned λ's.
* Training uses **sparse row‑Adam**: β₁=0, one second moment per row, and only rows touched in the step are updated (and charged).

This is the single largest neural gain: −0.126 BPB at equal FLOPs (proxy scale).

### 3.4 Count memory with leave‑one‑document‑out calibration (L2 + L3; new estimator)
For each target and each context length k there are two experts:
* a global expert C(c_k, y)/C(c_k) over training counts;
* a document‑cache expert D(c_k, y)/D(c_k) over the article so far.

A gate combines them with the neural LM. Its inputs are features of the context only: log counts, the number of distinct
continuations (global and in‑document), recency within the document, position in the document, and the network's entropy and max‑probability.

**The key is how the gate is trained.** On training text the global counts must be computed **leave‑one‑document‑out
(LODO)**: the current article's own contribution is subtracted, including continuation types that occur only inside it.
Combined with the causal document cache, this reproduces *exactly* the test‑time situation on training data: a new
article, global statistics from other articles, and a cache of the article so far.

Evidence: a gate fitted only on LODO training statistics reaches 1.3779 BPB on validation. A gate fitted directly on
validation reaches 1.3736. The transfer gap is 0.004 BPB.

In the final system the gate is fitted on 3 % held‑out training articles that the network never saw. The network trains for less than one epoch anyway, so holding them out costs nothing.

### 3.5 Score‑first dynamic evaluation (L4)
The evaluated stream is processed left to right. Every token is scored before any update that could use it. One SGD step
is then taken on the already‑scored chunk (`dyneval.py`). All backward and update FLOPs are charged.
With lr = 0 the scores reproduce static evaluation exactly; this is tested.

## 4. Results
⟨RESULTS⟩

## 5. What did not work (negative results)
⟨NEGATIVE⟩

## 6. Reproduce
```bash
pip install torch numpy
bash scripts/reproduce.sh        # ≈ 1.5 h on 4 CPU cores; writes runs/final/{result,system}.json
```

## 7. Files
| file | purpose |
|---|---|
| `prepare.py` | data download, byte‑level BPE (lossless), token streams |
| `model.py` | compute core + hashed n‑gram memory + gates |
| `optim.py` | Muon with Polar Express (standard and Gram‑space), NorMuon/cautious WD options, sparse row‑Adam |
| `flops.py` | FLOP accounting |
| `train.py` | budgeted training (stops exactly at the budget), held‑out articles, EMA, schedules |
| `countmem.py` | count memory: hashing, LODO statistics, document cache, features |
| `mixture.py` | gate, statistics extraction |
| `dyneval.py` | score‑first dynamic evaluation with FLOP charging |
| `run_system.py` | full system: gate fitting on held‑out articles, evaluation, final budget audit |
