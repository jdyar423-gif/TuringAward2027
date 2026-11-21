# BES-100M-A20M — engineering specification (part 2: model, training, data, library, decoding, evaluation)

Part 1 is `docs/SPEC.md` (ground rules, token format, exact layer, batch format).

## 4. Model (`bes/model/`, JAX)

Reference to port (MIT): `kschwethelm/looped-lm-scaling` `nanochat/gpt.py` in "hyperconnect"
mode (blocks, lanes, initialisation, per-loop KV layout) and `karpathy/nanochat`
`nanochat/gpt.py` (value-embedding gate). Fetch them from raw.githubusercontent.com.

### 4.1 Forward pass

```
e      = rmsnorm(E_tok[x] + sum_f E_reg_f[reg_f] + W_cin @ sum_k m_k (E_ctok[cand_k] + sum_f E_cfeat_f[feat_kf]))
lanes  = (e, e)                                        # n_lanes copies
for r in 0..n_loops-1:                                 # lax.scan; optional jax.checkpoint
    u     = sum_n alpha[r,n] * lanes[n]
    u     = block_0(u; ve = E_ve[x, r])                # value embedding enters block 0 only
    u     = block_1(u)
    s     = rmsnorm(u) * g_loop[r]
    lanes = mix[r] @ lanes + beta[r][:,None] * s
h      = rmsnorm(sum_n lanes[n]) * g_head
z      = softcap * tanh(((h @ P) @ E_out.T) / softcap)          # float32
z     += sum_k b_k * onehot(cand_k),  b_k = 8 tanh(<h @ A_t, c_k> / 8),  c_k = sum_f E_tfeat_f[feat_kf]
```
`block(u) : u += attn(rmsnorm(u)*g1); u += mlp(rmsnorm(u)*g2)`, MLP = `W_down relu(W_up x)^2`,
no biases. Initial lane coefficients: `alpha[r] = onehot(r mod n_lanes)`, `mix[r] = I`,
`beta[r] = 1`. All matrices that the spectral optimiser trains must be **non-zero** at
initialisation (it keeps each matrix on the sphere of its initial Frobenius norm); new feature
paths are disabled through zero-initialised *tables or scalars* trained by Adam instead.

### 4.2 Attention with exact key biases, sink and confidence

Per head, `head_dim = 128` is split into `126` content dims and `bias_dims = 2` reserved dims.
- Content dims: RoPE, then L2-normalise q and k (QK-norm) and scale q by a learned per-head
  gain. Of the 63 rotary planes, `63 - clock_planes` use the token position within the row and
  the remaining `clock_planes` are split evenly between the `N_CLOCKS` structural clocks
  (geometric frequencies; angle = clock value × frequency). `clock_planes = 0` gives plain RoPE.
- Reserved dims: `q[..., 126:] = (1, 0)`, `k[..., 126:] = (bias_j, 0)`. The attention logit is
  therefore `gain * cos(q, k) + bias_j` with any attention kernel, including TPU flash kernels.
- **Sink.** At positions whose token is `BOS` the content part of the key is zeroed, `bias_j` is
  a learned scalar `sink_logit[block, head]` and the value is a learned vector
  `sink_value[block, head, 127]` (the prior mean). Everywhere else `bias_j = 0`.
  With `use_sink = False` BOS is an ordinary token.
- **Confidence.** With `use_conf = True` the last value dim of every head is overwritten by the
  constant 1 at ordinary positions and 0 at sink positions, so that dim of the head output is
  the attention mass outside the sink. It reaches the residual stream through `W_o` like any
  other value dim.
- Value embedding: `E_ve[x, r]` (`ve_dim` wide) is added to the first `ve_dim` value dims of
  block 0 through the gate `3*sigmoid(W_g x[..., :12])` (never to the confidence dim).
- Mask: causal and same-segment (`seg[i] == seg[j]`); every segment starts with `BOS`, so every
  query can see its segment's sink.
- Two interchangeable implementations selected by `cfg.attn_impl`: `"xla"` (dense masked
  softmax, any backend, used in all CPU tests) and `"splash"`
  (`jax.experimental.pallas.ops.tpu.splash_attention`, static causal mask + per-example segment
  ids, vmapped over the batch). They must agree to float tolerance; the splash path must be
  import-guarded so CPU runs never touch it.

### 4.3 Parameter tree, roles, budget

`init_params(cfg: ModelCfg, key) -> dict` returns a nested dict of arrays with stable names
(`emb/tok`, `emb/ve`, `emb/reg/<field>`, `cand/in_tok`, `cand/in_feat/<f>`, `cand/in_proj`,
`core/attn/{wq,wk,wv,wo}` stacked over blocks, `core/attn/{q_gain,sink_logit,sink_value,ve_gate}`,
`core/mlp/{w_up,w_down}`, `core/norm/{g1,g2}`, `loop/g`, `lanes/{alpha,mix,beta}`,
`head/{g,proj,out}`, `trust/{proj,feat/<f>}`, `critic/{w1,w2}`).
- `bes/roles.py`: `role_of(path) -> Role(kind, group)` with `kind in {"dense", "lookup"}` and
  optimiser `group in {"matrix", "embed", "unembed", "norm", "scalar", "frozen"}`; every leaf
  must be covered (test).
- `bes/budget.py`: `count(params, cfg, assets=None) -> Budget(total, active_step,
  active_strict(T, distinct_tokens))`. Lookup tables count one row per read. A test asserts
  `total <= 100e6` and `active_step <= 20e6` for `configs/final.py` and `configs/no_library.py`.
- `bes/readset.py`: a numpy reference forward for one position that receives only the rows that
  were read, plus a logger of distinct numbers read per token (audit of the active count).

### 4.4 Loss and generation

- `loss_fn(params, consts, batch, cfg) -> (loss, metrics)` on one micro-batch (`[B,T]` arrays);
  weighted mean cross-entropy over `w`; logits and log-sum-exp in float32; tokens processed in
  chunks so that the `[n, V]` logits never exceed about 0.6 GB; the candidate tilt is added
  without scatter (compare/one-hot product) and costs at most a few percent of the step.
  `consts["enable"]` holds 0/1 switches (`reg`, `cand_in`, `cand_out`, per candidate kind,
  `conf`, `sink`) so one compiled graph covers the whole fallback ladder.
- With every feature path disabled the model must equal the plain looped transformer (test);
  with `b_k = 0` the loss must be bit-identical to the loss without the tilt (test).
- Generation (`generate.py`): `init_cache`, `prefill(params, consts, cache, feats)`,
  `decode_step(params, consts, cache, tok, feats) -> (logits[B,V], hidden, cache)` with a KV
  cache of shape `[n_loops, n_blocks, B, H, Tmax, head_dim]`, static shapes, jit-compiled once
  per `(B, Tmax)`. Test: full-sequence forward equals prefill + stepwise decode (atol 1e-4),
  and perturbing token `j` leaves all logits before `j` unchanged.
- Critic: `critic(params, hidden) -> float32[B]`, a 2-layer head on the final hidden state.

## 5. Optimisers and training (`bes/optim/`, `bes/train/`)

- `muonh.py`: port of `looped-lm-scaling/nanochat/optim.py` (Nesterov momentum 0.95, Polar
  Express with 5 steps and the 1.02 Frobenius pre-normalisation, update rescaled to the cached
  initial norm, projection back to the sphere, no weight decay) for stacked matrices
  `[..., m, n]`. `muon_normuon.py`: the `karpathy/nanochat` `nanochat/optim.py` Muon (Polar
  Express, NorMuon second moment, cautious weight decay). Both are tested against an independent
  numpy implementation written in the test.
- Adam (betas 0.8/0.95, eps 1e-10, no weight decay) for all non-matrix groups, with the group
  learning rates of `OptimCfg` multiplied by `(d_model / lr_width_ref) ** -0.5`.
- `build.py`: `build(cfg, params) -> Optimizer` with `init(params) -> opt_state` and
  `update(grads, opt_state, params, lr_scale) -> (params, opt_state)`; pure functions.
- `schedule.py`: wall-clock driven progress
  `p = train_seconds / (train_seconds + max(0, deadline - post_reserve - now))`; learning-rate
  factor, sequence length, chat fraction, library fraction and loss masking are functions of `p`
  only. Tested with a fake clock (slow, fast and interrupted runs all finish on time).
- `train.py`: one jit-compiled step = `lax.scan` over `A` micro-batches accumulating gradients,
  then one optimiser update; buffers donated; the loss is read one step late; both sequence
  shapes are compiled up front. Only this process touches the accelerator.
- `checkpoint.py`: atomic (write to temp, fsync, rename) save/restore of params, optimiser
  state, data cursor, RNG and time ledger; keep the last 3. Test: 10 + 10 steps with a restart
  equals 20 steps bit-for-bit on CPU.
- `guard.py`: NaN / loss-spike detection (loss > 1.3 × EMA three times, or non-finite) →
  restore the last good checkpoint, skip 200 steps of data, learning rate × 0.7 (at most 3 times).
- `benchmark.py`: the 15-minute start-up checks of the plan (step time and peak memory for both
  shapes, attention implementation, host pipeline rate, determinism of a repeated batch).
- `aot.py`: compile-only check for the TPU topology via `jax.experimental.topologies`.
- `prefopt.py`: one epoch of DPO against a frozen reference (reference log-probabilities
  computed once), learning rate 1e-6..3e-6, health checks before accepting.
- `scripts/run_24h.py`: supervisor that starts downloads, tokenisation workers, the library
  build, the trainer, the CPU evaluation side-car and the post-training stages, all driven by one
  deadline.

## 6. Data (`bes/data/`)

- `sources.py`: registry of datasets (Hugging Face id, config, split, text or message fields,
  licence, kind in `base | chat | rag | math | prefs | library`, default weight). Every entry is
  verified against the Hub before it is committed.
- `download.py` + `tokenize_worker.py`: streaming parquet → token shards
  `tok/<source>/<n>.u16` (segments separated by `BOS`) with `.idx` segment offsets and, for
  conversations, a `.w` loss-weight mask; `manifest.jsonl` appended only after fsync + rename.
- `chat_format.py`: conversations → ids, weights and truncation at turn boundaries.
- `hindsight.py`: adds verifiable constraints to existing prompts using `bes.exact.constraints`.
- `rag_format.py`: retrieval-aware sequences (library passage or distractor before the user
  turn / after a text prefix; a document never retrieves itself; "not in the passage" samples).
- `loader.py`: deterministic packing of segments into rows (every row starts with `BOS`),
  mixture by progress `p`, exact-layer features computed in worker processes, fixed-shape
  batches in a ring buffer; the data position is a pure function of the step so that a restart
  reproduces the same batches.
- `prefs.py`: preference pairs (existing datasets, verifier-labelled samples, particle-decoder
  winners versus rejected samples).

## 7. Library (`bes/library/`)

`clean.py` (strip templates, IPA, reference marks), `select.py` (cost-benefit greedy under a
budget of stored numbers), `pack.py` (independent 64 kB xz blocks, title list, CRC per block),
`index.py` (truncated inverted index counted in the budget: per term bucket the top postings
with 8-bit impacts, plus document frequencies), `retrieve.py` (BM25 with a fixed read budget
and a gate on score margin and title match; returns at most one passage of <= 160 tokens).
One stored number = 2 bytes; `budget.count` receives the library and index sizes.

## 8. Decoding (`bes/decode/`)

`sampling.py` (temperature, top-p, n-gram blocking outside code and quotes with an exemption for
library copies, cap on verbatim copy runs), `potentials.py` (token-local potentials from the
registers and the compiled constraints with reachability pruning; forced calculator results),
`smc.py` (K batched particles, locally optimal proposal, incremental weights = allowed mass,
adaptive resampling at sentence boundaries when ESS < K/2, KV-cache particle copy), `select.py`
(drop failures, length band, lexical consensus, majority vote for final numbers). Tests compare
against exhaustive enumeration on a tiny vocabulary (unbiased normalising constant, convergence
to the target distribution as K grows, mask weight = allowed mass).

## 9. Evaluation (`bes/eval/`)

`val.py` (held-out loss by source and by position), `probes.py` (step-function probes: copying,
exact counts, fence closing, arithmetic), `proxy/` (blind A/B harness: prompt set, our model,
opponents through GGUF on CPU, judge with side swap and length control, Wilson intervals).
