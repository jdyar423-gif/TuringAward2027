# CLAIMS: every quantitative or factual claim of `paper.md`, with its source

State of the sources: 24 November 2026, about 10:15 UTC; repository at commit `9a4d652`.

Abbreviations used in the source column:

- `R/` = `/home/user/turingaward2027/` (the repository; `docs/` is `R/docs/`).
- `N/` = `/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/` (snapshots, scans, samples).
- `S/` = `/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/` (operator logs, chip outputs, TPU-time ledger).
- `W/` = `N/wf/` (workflow reports). `L/` = `N/paper/lead_work/` (this author's extracts and scripts).
- `FLOG` = `L/final_snap/run/train/train_log.jsonl` (the stopped `final` run, extracted from
  `N/snap_final_main16/run.tar.*`); `VLOG` = `L/v9_snap/run/train/train_log.jsonl` (the `v9` run,
  extracted from `N/snap_main/run.tar.*`, snapshot of 10:03 UTC); `FMAN`, `VMAN` = their `run/manifest.json`;
  `FLED`, `VLED` = their `run/train/ledger.json`.
- "computed" = arithmetic by this author on the cited inputs; the script is named where there is one.
- Line numbers refer to the files as they are at `9a4d652`.

Claims marked **[TBD]** have no value yet. A claim without a source is not in the paper.

| id | § | claim | value | source |
|---|---|---|---|---|
| C001 | status | last snapshot of the live run: step, tokens, time | step 29,901 (checkpoint step 29,722); 7,838,367,744 tokens; created 1795514634 = 10:03:54 UTC | `N/snap_main/parts.json` |
| C002 | status | progress at the last logged step | p = 0.5153 at step 29,900 | VLOG, step 29,900 |
| C003 | abs, 2.1 | the budget: ≤100M stored numbers of ≤16 bits, ≤20M read per token, 24 h of one TPU v6e, from scratch, blind pairwise comparison with the two opponents | as stated | `docs/PLAN.md` l.5; `docs/ACCOUNTING.md` l.5–10 |
| C004 | abs, 1, 7.8 | numbers read per token: `final` on a step that does not retrieve; `v9` | 19,033,736 (19.03M); 19,631,240 (19.63M) | `docs/ACCOUNTING.md` l.26 ("不檢索的步為 19.03M"; exact value = 19,278,959 − 245,223, computed); `docs/V9.md` l.163 |
| C005 | abs, 1, 7.7, 7.8 | analytic training FLOP per token (1,024 tokens, forward + backward, causal attention, no recomputation) | `final` 0.781 G, `v9` 0.809 G: +3.6% | computed, `L/scripts/flops.py` (shapes from `R/bes/config.py` ModelCfg and `R/configs/v9.py`) |
| C006 | abs, 1, 7.8 | `v9` time per token relative to `final` on the chip | 1.557 (one piece: 22.79 / 14.639 ms); 1.620 (whole step 1,433.1 / 884.5 ms); 1.535 (2,048 tokens: 1,510.2 / 983.9 ms); 1.570 live (280,370 / 178,523 tokens/s) | `S/v9_measA.txt`; `docs/CHANGES.md` l.351, l.357; FLOG step 29,300; VLOG step 29,750 |
| C007 | abs, 1, 7.2 | share of a block application's time outside matrix products and the attention kernel, reference implementation | 9.6 of 15.4 ms (62%) | `docs/TPU_NOTES.md` l.208–223 |
| C008 | abs, 1, 7.2 | live rate of the reference implementation on 23 Nov (`final`; `v9`) and the plan's assumption | 112–117k; 88–92k; plan 300–450k tokens/s | `docs/TPU_NOTES.md` l.187–189; `docs/PLAN.md` l.149 |
| C009 | abs, 1, 7.5 | live rate after the kernel switch | 281,374 tokens/s at step 23,150 | `docs/CHANGES.md` l.363 |
| C010 | abs, 1, 8 | progress at the graft | p = 0.504884 (stop at step 29,722: p 0.50489) | VMAN `graft.p`; FLOG end event of step 29,722 |
| C011 | abs, 1, 8.3 | numbers carried by the graft (59 leaves) and new numbers (5 leaves) | 34,531,720; 63,504,384 | `S/graft_out.txt`; `S/graft_dry_out.txt`; `S/graft_test.log` |
| C012 | abs, 6.3, 8 | tokens trained at the graft | 7,791,443,968 (= 29,722 × 262,144) | FLED `tokens`; `N/snap_final_main16/parts.json` |
| C013 | abs, 8.4 | graft rehearsals: source and grafted loss identical on a synthetic batch | difference 0.0 at steps 26,890 and 28,501 | `S/graft_test.log`; `S/graft_dry_out.txt` |
| C014 | abs, 12.2 | at 0.4 of training the model was clearly behind Falcon-H1-Tiny on the sampled prompts | qualitative, 7 prompts with a Falcon reply | `N/midrun_sample_step22613.md`; `N/OPERATIONS.md` ("Honest state") |
| C015 | 1, 5.5 | the page: 8 private ReLU² neurons per block in a 192-dimensional shared space, output gains, per-loop value embedding | as stated | `R/configs/v9.py` docstring; `docs/V9.md` §4 |
| C016 | 1, 4.1 | extra reads of `v9` | 597,504 (3.1%), of which 589,824 dense projections | computed from `docs/V9.md` l.160–161 (gain 1,536; page 595,968 incl. projections 589,824) |
| C017 | 1, 8.1 | training time left at the restart after the switch | 30,969 s (8.60 h) | `S/runlog_peek.txt` ("start train: slice 30969 s") |
| C018 | 1, 4.1 | `v9` stored total and reads | 98,036,104; 19,631,240 | `docs/V9.md` l.163 |
| C019 | 1, 4.2 | `v9` strict reading on 40 documents | 21.2–28.2M | `docs/ACCOUNTING.md` l.45–46 |
| C020 | 1 | adversarial review's novelty rating of the design mathematics | 2–3 out of 10 | `docs/MATH.md` l.5 |
| C021 | 2.1 | definitions of stored total, standard and strict readings; tokenizer not counted | as stated | `docs/ACCOUNTING.md` l.7–10 |
| C022 | 2.1 | the 24 h counts allocated runtime time, not waits; the operator's reading, reported to the user | as stated | `docs/CHANGES.md` l.258 |
| C023 | 2.2 | device and host of the run | TPU v6 lite; memory limit 31.24 GB; 44 cores; 185.7 GB RAM; jax/jaxlib 0.11.2; libtpu 0.0.50 | FMAN `devices`, `host` |
| C024 | 2.2 | matrix dimensions are multiples of 256 except the head (128) | as stated | `docs/TPU_NOTES.md` l.12–13 |
| C025 | 2.2, 7.2 | the two feed-forward products alone, forward | 814 TFLOP/s (1.52 ms) | `docs/TPU_NOTES.md` l.215 |
| C026 | 2.3 | Falcon-H1-Tiny: publisher's Q8_0 GGUF, llama.cpp defaults (T 0.8, top-k 40, top-p 0.95, min-p 0.05, no repetition penalty), ≤1024 tokens | as stated | `docs/EVAL_PROTOCOL.md` l.71–79 |
| C027 | 2.3 | MobileLLM-R1.5: community file rewritten as qwen3; mean KL 0.003 nats; 98% equal argmaxes | as stated | `docs/EVAL_PROTOCOL.md` l.80–92 |
| C028 | 2.3 | MobileLLM sampling and budget forcing; two arms; better arm on dev | T 0.6, top-p 0.95; reasoning closed after 3072 tokens, 1024 answer tokens | `docs/EVAL_PROTOCOL.md` l.97–110 |
| C029 | 2.3, 11 | our reply budget | 768 of our tokens ≈ 430 words; Falcon 1024 tokens ≈ 600 words | `docs/EVAL_PROTOCOL.md` l.122–126 |
| C030 | 2.4 | runtime 1 interval and seconds; returned at the user's request; no usable model | 02:58:32–05:00:37 on 23 Nov; 7,325 s | `S/tpu_intervals.txt` l.1; `docs/CHANGES.md` l.230, l.253 |
| C031 | 2.4 | runtime 2 counted interval and seconds; taken back about 19:29, 2 h 12 min after allocation; cause unknown | 17:13:00–19:30:00; 8,220 s | `S/tpu_intervals.txt` l.2; `docs/CHANGES.md` l.266, l.287–289 |
| C032 | 2.4 | runtime 3 start | 24 Nov 01:05:55 | `S/tpu_intervals.txt` l.3; `docs/CHANGES.md` l.297 |
| C033 | 2.4 | TPU time used at 10:00 UTC | 47,590 s = 13.22 h (7,325 + 8,220 + 32,045) | computed from C030–C032 |
| C034 | 2.4 | TPU time used at the graft (09:45:35) and the share in training steps | 46,725 s = 12.98 h; training 32,320.76 s = 8.98 h (69%) | computed; FLED (`mark` 1795513534.68 = 09:45:34.7; `seconds.train`) |
| C035 | 2.4 | resumption on runtime 3 | step 5,075 at 01:54:10 | `docs/CHANGES.md` l.302 |
| C036 | 2.4 | deadline, end of training, post-training minutes | 20:31:49; 18:33:49; 118 = 60 + 35 + 3 + 5 + 15 (critic and judge 0) | `docs/CHANGES.md` l.304, l.333 (gives 18:33:45); FLOG setup events (`train_end` 1795545229 = 18:33:49); VMAN `post_plan` |
| C037 | 2.5 | design, code, measurements and draft by language-model agents under the user's direction | as stated | `docs/CHANGES.md` §19 (narrative); `N/OPERATIONS.md` |
| C038 | 4.1 | rows of Table 2 | as in the table | `docs/V9.md` l.153–163; `docs/ACCOUNTING.md` l.16–26 |
| C039 | 4.1 | `final` stored total and worst-case reads; library numbers | 98,300,111; 19,278,959; library + index 63,768,391; worst retrieval read 245,223 | `docs/ACCOUNTING.md` l.24–26; `docs/E1_COVERAGE.md` l.105, l.244 |
| C040 | 4.1 | the library returns at most one passage of at most 160 tokens, before the user turn | as stated | `docs/SPEC_MODEL_TRAIN.md` l.176–180 (`retrieve.py`); `docs/PLAN.md` §4D |
| C041 | 4.1 | shared numbers = first six rows | 34,531,720 | computed (14,118,912 + 4,587,520 + 306,776 + 6,291,456 + 8,388,608 + 838,448); equals C011 |
| C042 | 4.2 | audit checks: \|Δ log p\| ≤ 1e-4; poisoned unread rows leave output unchanged; reads logged | as stated | `docs/ACCOUNTING.md` l.48–55 |
| C043 | 4.2 | `final` audit: standard 18.83M; strict 19.62–19.81M (T 1024), 20.07–20.49M (T 2048), 6 documents | as stated | `docs/ACCOUNTING.md` l.30–35 |
| C044 | 4.2 | `v9` audit: standard 19.43M (19.63M with critic); strict 21.2/23.8/24.6M and 22.4/26.7/28.2M on 40 documents | as stated | `docs/ACCOUNTING.md` l.41–46; `docs/V9.md` l.307–310 |
| C045 | 4.2 | strict reading adds per distinct token | 9,536 numbers | `docs/V9.md` l.166 |
| C046 | 4.3 | core applied 16 times per token; a page-table row per token | 2 blocks × 8 loops; 3,072 numbers (2 × 8 × 192) | `R/bes/config.py` ModelCfg (`n_blocks`, `n_loops`); `R/configs/v9.py` |
| C047 | 4.4 | library coverage/precision before → after the fix | NQ-open 35.0/54.1 → 6.0/76.9; PopQA 6.5/40.3 → 0.5/75.0; common-name probes 81.3/99.9 after | `docs/E1_COVERAGE.md` l.24–33 |
| C048 | 4.4, 13 | the gate opens on 8 of 200 dev first turns | 8 | `docs/EVAL_PROTOCOL.md` l.324–326 |
| C049 | 4.4 | CPU laboratory setup | width 48, 2 blocks × 4 loops, FFN 192, 1,040 types, 12M tokens, base 3.734 nats | `docs/V9.md` l.27–29 |
| C050 | 4.4 | laboratory arms at 12M tokens | `W_o` −0.079; `W_v` −0.039; `W_q`+`W_k` +0.046 | `docs/V9.md` l.33–37 |
| C051 | 4.4 | one-curve law for feed-forward storage | c = 0.070 at 6.6M tokens, ten arms, rms residual 0.008 | `docs/V9.md` l.69–79 |
| C052 | 4.4 | complete page in the laboratory | −0.252 nats at 12M | `docs/V9.md` l.51 |
| C053 | 4.4 | laboratory noise and seeds | one seed per arm; paired noise ≈ 0.01 nats | `docs/V9.md` l.12, l.209 |
| C054 | 4.4 | micro ablation of the page; rule margin and bar | −0.61%; margin 0.84%; bar 1%; "not resolved" | `docs/V9.md` l.220, l.227–230 |
| C055 | 4.4 | predictions at full scale | `v9` ≈ 2.45 (2.41–2.48) vs `final` 2.507 nats/token; published gains 0.04–0.11 nats | `docs/V9.md` l.191–196 |
| C056 | 5.1 | tokenizer | V = 8,192: ids 0–255 bytes, 256–8175 merges (7,920), last 16 special | `docs/SPEC.md` l.37–39 |
| C057 | 5.1 | registers and candidate tape (16 slots; orders 1–3 and longest match; counts, distinct successors, source role) | as stated | `docs/SPEC.md` §2.1–2.3 (l.55–134) |
| C058 | 5.2 | core shapes | d 768; 2 shared blocks; 8 loops; 6 heads × 128 (126 + 2 reserved); d_f 3,072; 2 lanes; head rank 512; softcap 15 | `R/bes/config.py` ModelCfg; `docs/SPEC_MODEL_TRAIN.md` l.53–55 |
| C059 | 5.2 | forward equations, block, no biases, lane initialisation | as stated | `docs/SPEC_MODEL_TRAIN.md` l.13–31; `R/bes/model/model.py` `forward`, `block`; `R/bes/model/controller.py` `init_lanes` |
| C060 | 5.2 | value embedding added to the first 128 value dims of block 0 through 3σ(W_g x[:12]) | as stated | `docs/SPEC_MODEL_TRAIN.md` l.68–69 |
| C061 | 5.2 | precision of activations, statistics, lanes, angles, logits, loss, master weights | as stated | `docs/TPU_NOTES.md` l.17–20, l.29–33 |
| C062 | 5.3 | attention: RoPE, QK-norm on 126 dims, gain init √126, reserved dims (1,0)/(b_j,0), BOS sink with learned logit and value, causal same-segment mask | as stated | `docs/SPEC_MODEL_TRAIN.md` l.51–70; `docs/TPU_NOTES.md` l.43–45 |
| C063 | 5.3 | B1: softmax with a sink as a Dirichlet posterior predictive | as stated | `docs/MATH.md` l.9–12 |
| C064 | 5.3 | sink removal in the micro ablation | −0.07% (mixture), −0.1% rounded | `docs/E2_ABLATIONS.md` l.51; `docs/CHANGES.md` l.127 |
| C065 | 5.4 | output law, its four properties, Pitman–Yor identity | as stated | `docs/MATH.md` l.35–46; `docs/SPEC_MODEL_TRAIN.md` l.25–49, l.112–114 |
| C066 | 5.4 | initial mass per active slot | e⁻⁶ (mass_offset 6) | `docs/SPEC_MODEL_TRAIN.md` l.43–44 |
| C067 | 5.4, 10.1 | temperature on the base measure only | as stated | `docs/MATH.md` l.44; `docs/CHANGES.md` l.65–79 |
| C068 | 5.5 | token-neuron equations | as stated | `R/bes/model/layers.py` `token_preact`, `token_neurons`; `docs/V9.md` l.134–143 |
| C069 | 5.5 | table shapes | K, U: 8,192 × 2 × 8 × 192 = 25,165,824 each; G: 8,192 × 2 × 768 = 12,582,912 | `R/bes/model/model.py` `init_params`; `docs/V9.md` l.160–161 |
| C070 | 5.5 | initialisation: P uniform, K standard normal, U = 0, G = 0 | as stated | `R/bes/model/model.py` `init_params` docstring |
| C071 | 5.5 | exact reduction tested | as stated | `docs/V9.md` l.106–109, l.295 |
| C072 | 5.6 | MuonH: Nesterov 0.95, 5 polar steps, sphere of initial Frobenius norm, no decay | as stated | `docs/SPEC_MODEL_TRAIN.md` l.124–129 |
| C073 | 5.6 | Adam settings and group learning rates; width factor; matrix rate; page keys in the embed group | β 0.8/0.95, ε 1e−10; embed 0.3, unembed 0.004, feature 0.01, page 0.004, gain 0.01, matrix 0.014; × (768/640)^−0.5 | `R/bes/config.py` OptimCfg; `docs/SPEC_MODEL_TRAIN.md` l.130–131; `R/configs/v9.py` docstring |
| C074 | 5.6 | transfer rule exponents and reference | (tokens/step / 2¹⁸)^0.558; (tokens / 9.83e8)^−0.32 | `docs/CHANGES.md` l.31 |
| C075 | 5.6, 8.3 | token horizon fixed by the benchmark | 13,116,792,508 | FMAN and VMAN `benchmark.horizon_tokens` |
| C076 | 5.6 | Table 3 values | as in the table | `docs/E2_ABLATIONS.md` l.45–54, l.84–91; `docs/CHANGES.md` l.116–127 |
| C077 | 5.6 | seed replicate and margin | 0.42%; 0.84% | `docs/E2_ABLATIONS.md` l.36, l.44 |
| C078 | 5.6 | deviation: chat reading of the 1% rule; registers kept | as stated | `docs/CHANGES.md` l.129, l.146 |
| C079 | 5.6, 13 | counting probe at 2M tokens | 79.2% with registers; 92.7% without | `docs/E4_PROBES_2M.md` l.20 |
| C080 | 6.1 | p = T/(T + R), linear in wall time while training | as stated | `R/bes/train/schedule.py` docstring l.1–20 |
| C081 | 6.1 | learning-rate factor: warm-up over 0.2% of p, then 1 − 0.9p | warmup_frac 0.002, warmdown_frac 1.0, final_lr_frac 0.1; checked: lr 0.462 at p 0.000924, 0.9821 at p 0.01987, 0.5459 at p 0.50453 | `R/bes/config.py` OptimCfg; FLOG steps 50, 1,050, 29,700 |
| C082 | 6.1 | curriculum functions of p | T 1024 → 2048 at p 0.67; chat share 0.25 → 0.7 at 0.92; assistant-only loss from 0.92 | `R/configs/final.py` (`late_from=0.67`); `R/bes/config.py` DataCfg |
| C083 | 6.1 | tokens per step | 262,144 | `R/bes/config.py` OptimCfg `tokens_per_step`; `docs/SPEC.md` l.150 |
| C084 | 6.1, 8.5 | the graft pause advanced p | 0.5049 → 0.5112 | FLOG end event (0.50489); VLOG stage event (0.51120) |
| C085 | 6.1 | the moved end: reserve 220 → 118 min; end 16:51:49 → 18:33:49; p 0.2415 → 0.2190; lr factor 0.783 → 0.803 | as stated | `docs/CHANGES.md` l.333; FLOG (end event p 0.24153 at step 12,062; stage p 0.218987; setup `train_end` 1795539108.88 → 1795545229.0; lr 0.7829 at step 12,050); 0.803 = 1 − 0.9 × 0.218987 (computed) |
| C086 | 6.2 | checkpoints every 25 min, atomic, with optimiser state, cursor, RNG, ledger | as stated | `R/bes/config.py` RunCfg `ckpt_every_min`; `docs/SPEC_MODEL_TRAIN.md` l.140–142 |
| C087 | 6.2 | loss guard rule; rollbacks so far | 3 steps above 1.3 × EMA; 0 rollbacks | `R/bes/config.py` RunCfg `guard_*`; FLED and VLED `rollbacks` |
| C088 | 6.2 | new-machine errors; 24 workers, 363 shards in 9 min; 45 of 48 min idle | as stated | `docs/CHANGES.md` l.299–302; commits `e096dff`, `77305dc` |
| C089 | 6.3 | Table 4 rows | as in the table | FLOG (events and step records); `docs/CHANGES.md` l.283, l.287, l.302, l.310, l.320, l.333, l.363; `N/OPERATIONS.md` (deflection bounce 08:34–08:36, step 25,280); `S/runlog_peek.txt` (29,722; 29,750) |
| C090 | 6.3 | first logged step of the run | loss 4.8584, 220,082 tokens/s at step 50 | `docs/CHANGES.md` l.283; FLOG step 50 |
| C091 | 6.3 | mean training rate to the graft | 241,065 tokens/s (7,791,443,968 / 32,320.76 s) | computed from FLED |
| C092 | 6.4 | Table 5 values | as in the table | computed, `L/scripts/windows.py` on FLOG |
| C093 | 6.4 | tape gain early and at p ≈ 0.5 | 0.318 → 0.022 nats (0.88% of 2.496) | computed, `L/scripts/windows.py` on FLOG |
| C094 | 6.4 | tape switched off at evaluation in the micro ablation, 1M tokens | +10.49% | `docs/E2_ABLATIONS.md` l.108 |
| C095 | 6.5 | DPO settings: β 0.1, 64 pairs per step, plain Adam, clip 1.0, lrs 1e−6 and 3e−6, health checks (accuracy gain > 1 SE, loss rise ≤ 3%, length 0.7–1.6×, repetition ≤ 1.2×) | as stated | `R/bes/config.py` PrefCfg; `R/bes/train/prefopt.py` docstring; `R/scripts/run_24h.py` l.1096 (`lrs = (1e-6, 3e-6)`) |
| C096 | 6.5 | sampled pairs are under 1% of the preference data; the rest UltraFeedback | as stated | `docs/CHANGES.md` l.330 |
| C097 | 6.5 | test replies on host cores | 25–35 min on 42 workers (plan) | `W/testgen.md` l.29 |
| C098 | 7.1 | the four pauses | 6.7, 10.8, 9.4 (566 s), 13.2 (789 s) min | `docs/CHANGES.md` l.324, l.344; `S/resume_v9.log` |
| C099 | 7.1 | cost-model calibration | 0.521 ms per M cycles (140.73 M ↔ 73.36 ms; 31.85 M ↔ 16.61 ms) | `W/speed_mlp_block.md` l.86–91 |
| C100 | 7.2 | the suspects and their measurements | as listed | `docs/TPU_NOTES.md` l.192–202 |
| C101 | 7.2 | inside a block: 15.4 ms; products ≈ 2.4; kernel 3.4; glue 9.6 (62%); d × d product 0.23 ms | as stated | `docs/TPU_NOTES.md` l.208–223 |
| C102 | 7.3 | Table 6 | as in the table | `docs/TPU_NOTES.md` l.229–236 |
| C103 | 7.3 | rotation as two products | as stated | `docs/CHANGES.md` l.238 |
| C104 | 7.3 | 28 equality tests | 28 | `docs/CHANGES.md` l.243 |
| C105 | 7.3 | bfloat16 pre-activation: one rounding 2⁻⁹; 2.97 → 2.13 ms (28%) | as stated | `docs/CHANGES.md` l.240; `R/bes/model/layers.py` `mlp` docstring |
| C106 | 7.3 | whole steps new vs old code | 201,877 vs 111,893 tokens/s; 1.299 vs 2.343 s | `docs/CHANGES.md` l.251 |
| C107 | 7.4 | 32 → 16 rows | +8% (73.3 ms per 16 rows, 223,417 vs 206,973 tokens/s) | `docs/CHANGES.md` l.272–275 |
| C108 | 7.4 | Table 7 | as in the table | `docs/CHANGES.md` l.314–318 |
| C109 | 7.4 | live 217 → 240–246k after the pieces | as stated | `docs/CHANGES.md` l.320 |
| C110 | 7.4 | lanes kept in on-chip memory at 16 × 1,024; 50 MB lanes do not fit at 32 × 1,024 (compile-only) | as stated | `W/speed_loops.md` l.57–62, l.87–91 |
| C111 | 7.4, 7.6 | further halving with the new kernel | 958.9 vs 884.5 ms; 1,013.1 vs 983.9 ms | `docs/CHANGES.md` l.357 |
| C112 | 7.5 | one-tile kernels: VMEM slice, strips of 256 rows, 62.5% of score elements, lane-dense LSE, `save` | as stated | `docs/TPU_NOTES.md` l.311–318; `docs/CHANGES.md` l.339 |
| C113 | 7.5 | executed-bundle ratios of `onetile:save` | 0.34 (1,024), 0.42 (2,048) | `docs/TPU_NOTES.md` l.296–309 |
| C114 | 7.5 | ReLU² backward bit for bit on CPU; removes the packed mask | as stated | `W/integration.md` l.68, l.147–161; `W/speed_mlp_block.md` l.19–35 |
| C115 | 7.5 | Table 8 | as in the table | `docs/CHANGES.md` l.346–355; `W/chip_kernels_0745.txt` |
| C116 | 7.5, 7.8 | whole step with `onetile:save` + ReLU² + CE backward | 884.5 ms (296,371 tokens/s) at 64 × 4 × 1,024; 983.9 ms at 64 × 2 × 2,048 | `docs/CHANGES.md` l.357; `W/chip_kernels_0745.txt` |
| C117 | 7.5 | live effect of the kernel switch | 281,374 at step 23,150 vs 239.4–239.7k (+17.5%) | `docs/CHANGES.md` l.363 |
| C118 | 7.5 | chip loss and gradient differences of the candidates | ≤ 1e−6; 1.4e−2/1.5e−2, 7.5e−2/5.4e−2 (`onetile:save`); 1.4e−2/1.2e−2, 4.7e−2/3.3e−2 (`blocks:save`); 1.1e−2, 3.9e−2 (`relu2`+`ce`); 0 (`io`) | `docs/CHANGES.md` l.361; `W/chip_kernels_0745.txt` |
| C119 | 7.5 | CPU interpret-mode checks of the kernels | float32 within 1e−5; float64 dense reference; segment layouts | `docs/CHANGES.md` l.361; `W/kernel.md` l.71–96 |
| C120 | 7.5 | the review changed `max_t` fallback and typo handling, not the kernels | as stated | `W/kernel.md` l.137–152, l.170 |
| C121 | 7.6 | Table 9 rows | as in the table | `docs/TPU_NOTES.md` l.71–72, l.194, l.238–242; `docs/CHANGES.md` l.273, l.274, l.277, l.283, l.322, l.357, l.359; `S/v9_measA.txt`; `S/v9_measB_sorted.txt` |
| C122 | 7.6 | two items contradicted the operator's written expectation | `remat_keep=kernel`; the memory limit | `docs/CHANGES.md` l.273, l.322 |
| C123 | 7.6 | compile-only rejections | carry of the next input +6.84 M cycles (+4.87%); FAST_MM_LO identical program | `W/speed_loops.md` l.103; `W/speed_mlp_block.md` l.186–192 |
| C124 | 7.7 | FLOP per token with recomputation and full tiles; utilisation | 1.132 G; 220 and 318 TFLOP/s at 281,374 tokens/s; 27% and 39% of 814 | computed, `L/scripts/flops.py`; C009; C025; consistent with "1.13 GFLOP" in `docs/CHANGES.md` l.324 |
| C125 | 7.7 | matrix products and attention kernels ≈ half of a step at 246k | as stated | `docs/CHANGES.md` l.324 |
| C126 | 7.8 | Table 10 rows | as in the table | C004–C006; `S/v9_measA.txt`; `docs/CHANGES.md` l.351, l.357 |
| C127 | 7.8 | `final`'s whole step was measured with the CE backward on | as stated | `docs/CHANGES.md` l.357 ("整步（`onetile:save+relu2+ce`…）") |
| C128 | 7.8 | attribution of `v9`'s extra cycles per piece | total +18.68 M; 5.66, 2.88, 2.38, 1.71, 1.23, 0.92, 0.93, 1.1, 0.32, 1.09, 0.17 | `W/v9_speed.md` l.74–89 (compile-only) |
| C129 | 7.8 | padding of the [8, 192] contractions | 8 → 16 sublanes, 192 → 256 lanes, 2.7× bytes | `W/v9_speed.md` l.94–96 |
| C130 | 7.8 | `v9`/`final` on 23 Nov; extra seconds per step then and now | 79–82%; ≈ +0.6 s (cost model +0.624 s); now +0.549 s; ratio 0.617 | `docs/TPU_NOTES.md` l.188–189; `docs/CHANGES.md` l.232; `W/v9_speed.md` l.33–36; computed 1,433.1 − 884.5 |
| C131 | 7.8 | cost model vs chip | predicted 1,508.2 vs 1,433.1 ms (+5%); `sorted` +129.7 predicted vs +355.9 measured (2.74×) | `W/v9_speed.md` l.50–51; `S/v9_measA.txt`; `S/v9_measB_sorted.txt` |
| C132 | 7.8 | `sorted` needs 16× fewer multiply-adds than one-hot | as stated | `docs/V9.md` l.111–117 |
| C133 | 7.8 | `flat` switch: −46 ms predicted, bit for bit on CPU; two padding-free layouts estimated at 100–130 ms, not implemented | as stated | `W/v9_speed.md` l.13–16, l.112–128, l.190–198 |
| C134 | 7.8 | on-chip effect of `flat` | **[TBD]** | — |
| C135 | 7.9 | decoding on the chip | 12.1 ms (6 rows), 34.5 ms (24 rows); 43.1 ms per position for 6 particles, ≈ 31 ms on the host | `docs/CHANGES.md` l.279 |
| C136 | 8.1 | the user asked for `v9` at 09:16 UTC (p ≈ 0.47–0.50) | 09:16 | the workflow brief of 24 Nov (operator's record, not a file read here); commit `9a4d652` message ("at 0.47 of the configs/final.py run") |
| C137 | 8.1 | from-scratch `v9` in the slice | ≈ 5.4e9 tokens (5.53e9 all at 1,024) | computed, `L/scripts/projections.py` |
| C138 | 8.1 | the pre-registered comparison, why it was not run | as stated | `docs/CHANGES.md` l.212–226 (design), l.232 (19.1), l.259 (19.2) |
| C139 | 8.2 | neutral initialisation; equal losses and shared gradients; zero gradient of K, P_in, P_out at the first step | derivation | `paper.md` §5.5, §8.2 (our derivation from `R/bes/model/layers.py` and `model.py`); consistent with `W/graft_review.md` l.107–110 (a step moves `page/val` off zero) |
| C140 | 8.2 | no gradient clipping in training | `grad_clip = 0.0` | `R/bes/config.py` l.171 (OptimCfg) |
| C141 | 8.2 | Adam without bias correction on the new leaves | 0.894 at t = 1; 1.439 at t = 7 (max); 1.041 at t = 50 | computed ((1 − 0.8ᵗ)/√(1 − 0.95ᵗ)); `W/graft_review.md` l.64–66 (0.894) |
| C142 | 8.3 | graft tool: matching, refusals, carried items | as stated | `R/scripts/graft_run.py` l.1–36, l.225–247; `W/graft_review.md` l.105–111 |
| C143 | 8.3 | loader corrections: `tokens_before`; reads left = 8 × (1 − p) | 7,791,443,968; 3.96 | VMAN `graft.tokens`; `S/runlog_peek.txt` ("at most 3.96 reads of a source"); `R/scripts/run_24h.py` `_grafted_stream` |
| C144 | 8.3 | without the correction the horizon estimate was 40 times too small one block after the graft | 40× | commit `9a4d652` message |
| C145 | 8.4 | check: 2 × 256 synthetic tokens, dense attention, float32, tolerance 1e−6 relative | as stated | `R/scripts/graft_run.py` l.57, l.191, l.206 |
| C146 | 8.4 | Table 11 values | 11.2808027 (26,890); 11.3348608 (28,501); 9.2843390 (rehearsal, step 14) | `S/graft_test.log`; `S/graft_dry_out.txt`; `N/graft_rehearsal/graft.log` |
| C147 | 8.4 | the saved output of the step-29,722 graft (into `main16_v9_sorted`) has no check record | as stated | `S/graft_out.txt` |
| C148 | 8.4 | check of the live graft at step 29,722 | **[TBD]** | — |
| C149 | 8.4 | configuration hash of both checked grafts and of the live run | `f29bfeb3ebf353a8` | `S/graft_test.log`; `S/graft_dry_out.txt`; VMAN `effective.config_hash` |
| C150 | 8.4 | review: four tests, no blocker, two medium risks | as stated | `W/graft_review.md` l.3–34, l.89–111 |
| C151 | 8.4 | `tests/train/test_graft_run.py` is untracked at the time of writing | as stated | `git status` in `R/` at about 10:15 UTC |
| C152 | 8.5 | pause length and its booking | 789 s; lost +727.3 s (25,061.96 → 25,789.25); set-up +64.3 s (154.35 → 218.61) | `S/resume_v9.log`; FLED, VLED; `S/runlog_peek.txt` ("took 64.3 s") |
| C153 | 8.5 | learning-rate factor after the graft | 0.5394 at step 29,750 | VLOG |
| C154 | 8.5 | tape mass before and after | 14.68 and 15.67 (medians, p 0.40–0.45 and 0.45–0.50); 1.240, 0.887, 1.279, 1.313 | computed, `L/scripts/windows.py` on FLOG; VLOG |
| C155 | 8.5 | sources capped by the re-planned epoch budget | dolly, gsm8k, smoltalk_constraints, smoltalk_everyday | `S/runlog_peek.txt` (loader lines after the restart) |
| C156 | 8.5 | losses after the graft vs the last window of `final` | 2.4118, 2.5704, 2.4896, 2.4423 (mean 2.479); 68 readings: mean 2.496, sd 0.037, range 2.416–2.570 | VLOG; computed on FLOG |
| C157 | 8.5 | projections of remaining and total tokens | `v9` +5.34e9 (1.313e10); `final` +8.09e9 (1.588e10); difference 2.75e9 (17.3%); 67.5% of the time at 2,048 | computed, `L/scripts/projections.py` |
| C158 | 8.5, 12.3 | tokens actually trained by 18:33:49 | **[TBD]** | — |
| C159 | 9.1 | registry: 26 entries verified on 21 Nov | as stated | `docs/DATA_SOURCES.md` l.6–17 |
| C160 | 9.1 | weights per kind | as listed | `docs/DATA_SOURCES.md` l.29–56 |
| C161 | 9.1 | Tulu 3 exclusions | non-commercial (No Robots), non-English (Aya), hard-coded identity | `docs/DATA_SOURCES.md` l.75–77 |
| C162 | 9.1 | rows 75% base / 25% chat-kind; realised shares of `final` | base 75.0%, chat 14.5%, rag 5.3%, maths 3.1%, library tasks 1.3%, library rows 0.9% | computed: mean of `loader_block.subkind_share` over FLOG; `row_share` |
| C163 | 9.1 | realised shares after the graft | base 75.5–75.9%, chat 14.9–16.0%, rag 4.2–5.9%, maths 2.7–4.4%, library 0 | VLOG `loader_block.subkind_share` |
| C164 | 9.1 | hindsight constraints | 12% of chat samples | `R/bes/config.py` l.221 (DataCfg `hindsight_frac`) |
| C165 | 9.2 | the step-5,075 refusal, six particles | as quoted | `N/midrun_sample_step5075_p6.md`; `docs/CHANGES.md` l.310 |
| C166 | 9.2 | Tulu 3 safety subsets ≈ 1/8 of its conversations | CoCoNot, WildJailbreak, WildGuardMix | `docs/CHANGES.md` l.310 |
| C167 | 9.2 | stock-refusal openings per source (30,000 conversations) | Tulu 3 4.85%; SYNTH 0.83%; system chats 0.67%; UltraChat 0.41% | `S/deflection_out.txt`; `N/deflection_check2.json` |
| C168 | 9.2 | refusal rule from step 6,647; scorer; preference sampling | commit `566517a` | `docs/CHANGES.md` l.310; `R/bes/data/loader.py` l.165–171; `R/scripts/run_24h.py` l.758 |
| C169 | 9.3 | the step-22,613 deflection | as quoted | `N/midrun_sample_step22613.md` |
| C170 | 9.3 | openings scan (40,000 segments per source) | system chats: apology 10.60%, "As an AI" 2.14%, "don't have access" 1.51%; UltraChat: "don't have" 2.81%, "don't have access" 2.21% | `S/openings_out.txt`; `N/openings.json` |
| C171 | 9.3 | deflection rule from step 25,280 | commit `4f4ef45` | commit `4f4ef45`; FLOG end/restore at step 25,280; `N/OPERATIONS.md` |
| C172 | 9.3 | conversations removed by the deflection rule | system chats 29.10% (tokens 29.81%); UltraChat 9.06% (7.10%); Tulu 3 1.23% (1.10%); others < 1% | `S/deflection_out.txt` |
| C173 | 9.4 | the model is not trained to refuse from step 6,647; operator's decision reported to the user | as stated | `docs/CHANGES.md` l.310 |
| C174 | 9.4 | UltraFeedback responses opening "As an AI" | 2.44% | `S/openings_out.txt` |
| C175 | 9.4, 12.3 | refusal and deflection rates of test replies | **[TBD]** | — |
| C176 | 10.1 | E5 values at T = 0.5 (chat) | mixture-tempered 0.326 / 13 of 48 / 0.614; network-only w1 0.134 / 1 / 0.068; w0.25 0.033 / 0 / 0.055; tape off 0.032 / 0 / 0.000 | `docs/E5_SELF_GENERATION.md` l.21–27 |
| C177 | 10.1 | the mechanism of absorbing loops | as stated | `docs/NEGATIVE_RESULTS.md` l.15 |
| C178 | 10.1 | self-evidence weight in training: 0.25 → +0.07%; 0 → chat +8.55% | as stated | `docs/E2_ABLATIONS.md` l.62–63, l.86–87 |
| C179 | 10.1 | E5 copy criterion; dataset replies copy 0.074 ± 0.008 | as stated | `docs/E5_SELF_GENERATION.md` l.48–51; `docs/CHANGES.md` l.99 |
| C180 | 10.2 | particles, potentials, resampling at ESS < K/2, satisficing selection | as stated | `docs/SPEC_MODEL_TRAIN.md` l.183–194; `docs/MATH.md` l.55–56 |
| C181 | 10.2 | tilt bound per stage | log K − (K − 1)/K = 0.958 nats for K = 6 | `docs/MATH.md` l.60 (formula); computed |
| C182 | 10.2 | decoding defaults | T 0.6, top-p 0.9, K 6, 768 tokens, n-gram blocking 6, self-evidence 0.25 | `R/bes/config.py` DecodeCfg |
| C183 | 10.2 | E9 stand-in results | 31/47, 43/47, 46/47; 12–0 (p 0.0005), 15–0 (p 0.0001); −0.025 ± 0.019; 5.8× tokens, 4.0× CPU; 9 of 63 at budget | `docs/E9_DECODING.md` l.26–46, l.70–74 |
| C184 | 10.2 | constraint-compiler coverage | dev 96%; held-out half 85% | `docs/E9_DECODING.md` l.10, l.76 |
| C185 | 10.3 | the selection stage changes the default only on a significant paired sign test | as stated | `docs/CHANGES.md` l.330 |
| C186 | 10.3 | the decoding study: 60 dev prompts, six settings, export of step 25,280 | as stated | `N/OPERATIONS.md` ("Since 08:20"); `S/study_progress.log` |
| C187 | 10.3 | its judged results | **[TBD]** | — |
| C188 | 11 | prompt set composition | 400; 100/60/60/100/80; a quarter two-turn; 95 constrained; maths recomputed | `docs/EVAL_PROTOCOL.md` l.19–36 |
| C189 | 11 | split procedure | as stated | `docs/EVAL_PROTOCOL.md` l.37–42 |
| C190 | 11 | seven of the eight mid-run sample prompts are in the test split | chat-002, chat-017, math-034, instruct-034, instruct-062, writing-100, code-026 (test); math-011 (dev) | checked by this author with `bes.eval.proxy.prompts.load()` (`split` field) on the ids of `S/scripts/midrun_sample.py` (`DEFAULT_IDS`) |
| C191 | 11 | seeds, conversations, cache | as stated | `docs/EVAL_PROTOCOL.md` l.51–69 |
| C192 | 11 | the opponents' replies to all 400 prompts exist | as stated | `N/OPERATIONS.md` ("Opponents' replies to all 400 prompts…") |
| C193 | 11 | blinding and sealed pair sets | as stated | `docs/EVAL_PROTOCOL.md` l.173–198 |
| C194 | 11 | two readings, strict agreement, instructions | as stated | `docs/EVAL_PROTOCOL.md` l.209–219, l.423–465 |
| C195 | 11 | readers in this run: language-model agents through `blind_eval.py` | as stated | `S/scripts/blind_eval.py` docstring; `N/OPERATIONS.md` |
| C196 | 11 | exact scores and degenerate detectors | as stated | `docs/EVAL_PROTOCOL.md` l.255–289 |
| C197 | 11 | statistics and power | ±7 points at n = 200; 10 points: 194 / 277; 5 points: 783 / 1,118 | `docs/EVAL_PROTOCOL.md` l.10–17, l.291–313 |
| C198 | 11 | controls | as stated | `docs/EVAL_PROTOCOL.md` l.148–164 |
| C199 | 11 | Falcon dev reference | maths 66.7% of 30; constraints 78.7% of 47; 1.6% cut; median 174 words | `docs/EVAL_PROTOCOL.md` l.404–421 |
| C200 | 12.2 | contents of the step-22,613 sample, ours and Falcon's | as described | `N/midrun_sample_step22613.md` |
| C201 | 12.2 | the standing was reported to the user | 08:20 and 08:40 UTC | `N/OPERATIONS.md` |
| C202 | 12.3 | R1–R12 | **[TBD]** | — |
| C203 | 13 | GP attention numbers | 1e−15; ≤ 2e−4 nats; 0.003–0.008 nats; 63.6% vs 5.0%; 4.5–9× | `docs/NEGATIVE_RESULTS.md` l.9–16 |
| C204 | 13 | scale-recursive loops | 0.1–3% | `docs/NEGATIVE_RESULTS.md` l.24 |
| C205 | 13 | coarse-to-fine head | +0.035 to +0.07 nats | `docs/NEGATIVE_RESULTS.md` l.30 |
| C206 | 13 | confidence channel; clocks | +0.7%; +0.3% | `docs/E2_ABLATIONS.md` l.45–46 |
| C207 | 13 | fence probe | 100% both | `docs/E4_PROBES.md` l.21 |
| C208 | 13 | the first gate | +0.80% (+1.22% chat); 9–26% of values beyond [0.2, 1.8]; rescaled: −0.02% | `docs/V9.md` l.222–224, l.236–244 |
| C209 | 13 | page decay argument; set to 0 | as stated | `docs/V9.md` l.245–258 |
| C210 | 13 | token-specific Q/K; FFN gains; hashed n-gram tables | +0.046; −0.035; ≈ 0.007 nats | `docs/V9.md` l.33, l.42, l.93–96 |
| C211 | 13 | the critic: gate never measurable; 0 minutes | as stated | `docs/E9_DECODING.md` l.116; `docs/CHANGES.md` l.260 |
| C212 | 13 | transfer rule at micro scale | +14.44% | `docs/E2_ABLATIONS.md` l.64 |
| C213 | 13 | `sorted` 25% slower than `take` | 1,789.0 / 1,433.1 = 1.248 | `S/v9_measB_sorted.txt`; `S/v9_measA.txt` |
| C214 | 13 | blocked or sleeping agent turns | ≈ 4 h 50 min without allocation requests | `docs/CHANGES.md` l.291–294 |
| C215 | 13 | the plan's token expectation | 16–25 × 10⁹ tokens | `docs/PLAN.md` l.149 |
| C216 | 14 | seed replicate | 0.42% | `docs/E2_ABLATIONS.md` l.36 |
| C217 | 14 | the laboratory's regime: count-based n-gram models beat every neural arm | as stated | `docs/V9.md` l.207–208 |
| C218 | 15 | run created at commit `98598c2`, clean tree | as stated | FMAN `code` |
| C219 | 15 | commits of the continuation | `e096dff`, `77305dc`, `566517a`, `b05bfe7`, `86bcd45`, `782be85`, `6bf9b6e`, `641c4b6`, `4f4ef45`, `9a4d652` | `git log` in `R/` |
| C220 | 15 | effective configuration hashes | `final` `0a7e9b33372a07ee`; `v9` `f29bfeb3ebf353a8` | FMAN, VMAN `effective.config_hash` |
| C221 | 15 | the run's command line | as quoted | FMAN `argv` |
| C222 | 15 | per-run files | `micro_split` 4; `fast` `kernel=onetile:save relu2=1 ce=0`; `post_plan` as quoted | VMAN `micro_split`, `fast`, `post_plan` |
| C223 | 15 | determinism canary on the chip | loss 8.475837707519531 twice | VMAN `benchmark.canary.determinism` |
| C224 | 15 | restart reproduces uninterrupted steps bit for bit on CPU (test) | as stated | `docs/SPEC_MODEL_TRAIN.md` l.140–142 |
| C225 | 15 | test functions | 1,349 (static count of `def test_`) | `grep -c "def test_" -r R/tests` (this author) |
| C226 | 15 | uncommitted changes in the working tree at the time of writing | `bes/model/fast.py`, `bes/train/train.py`, two tests modified; `tests/model/test_flat_tables.py`, `tests/train/test_graft_run.py` untracked | `git status` in `R/` |
| C227 | 16 | ClimbMix: MIT card, upstream CC BY-NC 4.0 | as stated | `docs/DATA_SOURCES.md` l.106–110 |
| C228 | 16 | third-party-model outputs in chat and preference sources | as stated | `docs/DATA_SOURCES.md` l.110–135 (items 2, 3, 5, 8) |
| C229 | 16 | the credential stayed in an owner-readable file in the container, never in the repository | as stated | `docs/CHANGES.md` l.189 |
| C230 | 17 | factor 1.6 between configurations; 2.4–2.5 between implementations | 1.535–1.620; 281,374 / 112–117k = 2.40–2.51 | C006; C008; C009 |
| C231 | 11 | the protocol document is in the branch's first commit, before any run trained, and unchanged since | `5ff33c5`, 23 Nov 00:35:15 UTC; first runtime allocated 02:58:32 | `git log -- docs/EVAL_PROTOCOL.md` (one commit); `git diff 5ff33c5 HEAD -- docs/EVAL_PROTOCOL.md` (empty); C030 |
| C232 | abs, 1, 12.1 | time ratio range used in the text | 1.54–1.62 (1.535 at 2,048 tokens to 1.620 for the whole step at 1,024) | C006 |
| C233 | 6.3 | the 2,048-token phase starts (p = 0.67) at about 12:46 UTC | 12:46 | computed: restart at 09:58:30 with p 0.5112, slice 30,969 s, p linear in time (`R/bes/train/schedule.py`) |
| C234 | 2.3 | opponents' training tokens | Falcon-H1-Tiny-90M-Instruct 800 G tokens (25% SFT data from the start); MobileLLM-R1 pre-training 4.2 T tokens; R1.5 continues R1 with on-policy distillation | `N/paper/refs.md` [11], [7], [9] (verification notes of the related-work agent; not re-read by this author) |
| C235 | 1 | data movement is known to bound transformer training on GPUs | as stated | `N/paper/refs.md` [63] (verified abstract: "data movement is the key bottleneck") |
| C236 | 8.6 | staged training carries the whole training state through loss- and dynamics-preserving growth operators | as stated | `N/paper/refs.md` [47] (verified: abstract and Sec. 3.3) |
| C237 | 6.1 | tying the learning-rate decay to the time budget follows budgeted training | as stated | `N/paper/refs.md` [1]–[3] |

## Numbers in the paper that are projections or derivations, not measurements

C005, C016, C033, C034, C041, C091, C124, C137, C139, C141, C157, C181 are arithmetic on measured or
configured inputs. C110, C123, C128, C129, C131 (the predicted side) and C133 are compile-only
estimates of the compiler's cost model. C055 is a prediction from the project's own estimate.

## Open items ([TBD] in the paper)

C134, C148, C158, C175, C187, C202 (R1–R12), and in the text: authors; held-out losses (§4.4, §6.4);
the final step and tokens (Table 4); what each post-training stage did (§6.5); decoding throughput
of the export (§7.9); the reader model (§11); which controls are run (§11); results on the 193
untouched test prompts (§11, R12); the licence review (§16); commits of the uncommitted files (§8.4, §15).
