# BES-100M-A20M

## 摘要（繁體中文）

這個倉庫是 BES-100M-A20M：一個總共至多 1 億個儲存數字（每個數字至多 16 位元，含權重、查表與藏書）、每產生一個 token 至多讀取 2,000 萬個數字的語言模型，設計成在單顆 TPU v6e 上於 24 小時內從零訓練完成。核心原理是「後驗 ∝ 先驗 × 證據」，在三個尺度上能精確算的就精確算：注意力頭帶校準匯點與信心通道；主機端精確計算的 n-gram 計數頭與候選帶進入輸出律；整段回覆以少量粒子加精確勢函數做 Feynman–Kac 解碼；另附一個門控、離線、凍結的書寫藏書。所有程式都在 CPU 容器上開發、測試與彩排，**目前沒有任何數字是在 TPU 上量到的**（`docs/TPU_NOTES.md`）。正式的 24 小時訓練請照 `RUNBOOK.md` 使用 `notebooks/colab_v6e_24h.ipynb`；計畫原文在 `docs/PLAN.md`，核准後的修訂在 `docs/CHANGES.md`。

## What this is

A language model with at most 100M stored numbers (each at most 16 bits: weights, lookup tables,
the compressed library and its index) and at most 20M numbers read per generated token, designed
to be trained from scratch on one TPU v6e within 24 hours and judged blind against two small
open models.

**The idea in one paragraph.** Every step of the model is "posterior proportional to prior times
evidence", computed exactly wherever that is possible: inside attention, each head's calibrated
sink makes softmax attention an exact Bayesian estimate and a constant value channel reads out its
confidence; at the output, exact in-context n-gram counts computed on the host (a candidate tape
with Pitman-Yor statistics) enter a closed-form output law next to the network's distribution;
over a whole reply, a few particles of the same model are steered by exact potentials (registers,
compiled constraints, a calculator) and chosen by a satisficing rule. A written library (Simple
English Wikipedia and question-answer pairs) is retrieved behind a calibrated gate and counts
against the 100M. The core is a looped transformer (2 shared blocks, 8 loops, d = 768). Plan,
mathematics and honest odds: `docs/PLAN.md`, `docs/MATH.md`, `docs/NEGATIVE_RESULTS.md`.

## Measured so far

Everything below was measured on a 4-core CPU container (or compiled for the TPU without running
on it); nothing has run on a TPU.

| fact | where |
|---|---|
| `configs/final.py`: 19.10M numbers read per generated token (standard convention); strict convention 19.62-19.81M at T = 1024 and 20.07-20.49M at T = 2048 on real text (`configs/strict.py` keeps both under 20M) | `docs/ACCOUNTING.md` |
| Library `lib_v2`: 63,720,178 stored numbers; NQ-open questions answered 35.0% at 54.1% precision (18.5% at 37.3% without a near-duplicate stored question), worst-case read of one retrieval 232,818 numbers | `docs/E1_COVERAGE.md` |
| Candidate tape on a micro model (d = 128, 4M tokens, one seed): held-out loss 4.488 against 4.963 nats without it | `docs/CHANGES.md` section 9 |
| Particle decoding, measured on an opponent model as a stand-in: prompts with every constraint satisfied 66% (plain) -> 91% (filter) -> 98% (exact potentials), about 4x the compute | `docs/E9_DECODING.md` |
| Compile-only for v6e: train step peak 4.21 GiB with rematerialisation; without it the full micro-batch does not fit (32.93 GB of temporaries), half of it does (17.03 GiB) | `docs/TPU_NOTES.md` |
| Proxy blind test harness; Falcon-H1-Tiny-90M-Instruct on the dev half: math 66.7%, constraint prompts fully satisfied 78.7% | `docs/EVAL_PROTOCOL.md` |
| Evidence gates E2, E4, E5: not run when this README was written (the reports say where they stand) | `docs/E2_ABLATIONS.md`, `docs/E4_PROBES.md`, `docs/E5_SELF_GENERATION.md` |
| Rejected after testing: exact Gaussian-process attention, scale-recursive loops and others | `docs/NEGATIVE_RESULTS.md` |

## Running it

Tests (CPU; run the area you work on, each takes up to a few minutes; `-m slow` adds the long
ones):

    python -m pytest tests/model
    python -m pytest tests/train tests/eval

CPU rehearsal of the whole 24-hour pipeline in 12 minutes (downloads from `data_cache/`, the
library, the start-up benchmark and canary, training, sampling, DPO, decode selection, the judged
comparison against a GGUF opponent and a stand-in, the export):

    python scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 12 --out runs/smoke
    python scripts/chat.py --export runs/smoke/final --show-diagnostics
    python scripts/arena.py --export runs/smoke/final --opponent falcon=recommended

The real run: `RUNBOOK.md` (what to do before, during and after the 24 hours) and
`notebooks/colab_v6e_24h.ipynb`; a 30-minute free-tier check: `notebooks/colab_v5e_smoke.ipynb`.

## Layout

- `bes/` the package: `exact/` (registers, constraints, calculator, candidate tape), `library/`,
  `data/` (sources, downloads, tokenisation, loader), `model/`, `optim/`, `train/` (trainer,
  supervisor, benchmark), `decode/` (particle decoder, chat entry point), `eval/` (proxy harness,
  blind arena), `export.py`.
- `configs/` run configurations; `scripts/` command-line entry points; `tests/` tests;
  `docs/` plan (`docs/PLAN.md`), contracts (`docs/SPEC.md`, `docs/SPEC_MODEL_TRAIN.md`) and reports.
