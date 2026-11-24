# BES-100M-A20M

## 摘要（繁體中文）

這個倉庫是 BES-100M-A20M：一個總共至多 1 億個儲存數字（每個數字至多 16 位元，含權重、查表與藏書）、每產生一個 token 至多讀取 2,000 萬個數字的語言模型，設計成在單顆 TPU v6e 上於 24 小時內從零訓練完成。核心原理是「後驗 ∝ 先驗 × 證據」，在三個尺度上能精確算的就精確算：注意力頭帶校準匯點（信心通道在微型消融中增益不足 1%，最終配置關閉）；主機端精確計算的 n-gram 計數頭與候選帶進入輸出律；整段回覆以少量粒子加精確勢函數做 Feynman–Kac 解碼；另附一個門控、離線、凍結的書寫藏書，是否隨模型出貨由訓練後的事實題對照（藏書開與關各答一次）決定。所有程式都在 CPU 容器上開發、測試與彩排，**目前沒有任何數字是在 TPU 上量到的**（`docs/TPU_NOTES.md`）。正式的 24 小時訓練請照 `RUNBOOK.md` 使用 `notebooks/colab_v6e_24h.ipynb`；計畫原文在 `docs/PLAN.md`，核准後的修訂在 `docs/CHANGES.md`。另有不含藏書的替代配置 `configs/v9.py`（每個詞元一頁：私有前饋神經元與注意力輸出增益，外加逐頭的注意力輸出閘門；設計見 `docs/V9.md`），啟動指令相同，只需改用 `--config configs/v9.py`（`RUNBOOK.md` 第 1.4 節）。

### 24 小時訓練的結果（2026-11-24）

`configs/v9.py`（逐詞頁、無藏書，`docs/V9.md`；使用者在訓練中途指定改做這個配置，進行中的 `final` 檢查點以保函數的方式接枝過去，`docs/CHANGES.md` 19.7 節）已在單顆 TPU v6e-1 上從零訓練完成（TPU 配置時間合計 86,054 s，上限 86,400 s；第 51,812 步，1.358×10¹⁰ 個訓練 token）。正式模型在 `release/main16-v9/`：98,036,104 個 16-bit 數字，每產生一個 token 讀取 19,631,240 個。200 題測試集的盲測（Opus 5.5 代理判讀去識別、順序隨機的成對回覆，每對兩次、第二次左右互換；勝／負／平從本模型看，「平」含兩者皆差）：

| 對手（各自的建議設定） | 勝／負／平 | 分數（95% 區間） |
|---|---|---|
| MobileLLM-R1.5-140M | 116／15／69 | 0.752 [0.688, 0.807] |
| Falcon-H1-Tiny-90M-Instruct | 35／108／57 | 0.318 [0.257, 0.385] |

結論照實說：盲測上優於 MobileLLM-R1.5，明顯不如 Falcon-H1-Tiny，「同時超過兩者」的目標只達成一半。MobileLLM-R1.5 在這個評測框架、它的建議設定下有 62.7% 的回覆是退化的，那一項領先的門檻不高；讓 Falcon 也取 6 個樣本再挑一則時結果不變（35／112／53 與 29／116／55）；本模型只用一個粒子時對 Falcon 是 22／122／56。評審知道每一組比較的是哪兩個系統（檔案標題帶有配對集名稱），但不知道哪一側是誰；事先註冊、不經過這些檔名的自動評審對 Falcon 判了 12／61／127，方向相同。完整數字（對照組、K = 1、精確計分、自動評審、偏離協定之處、過程中的事故與稽核改正）在 `docs/CHANGES.md` 第 19.10 節；論文 `docs/paper/paper.md` 的定稿尚在更新，目前倉庫裡的版本還沒有測試集的數字。模型的還原、對話與自己做盲測的指令在 `release/main16-v9/README.md`。

### 在 Colab 直接開啟

| Notebook | 用途 | 直連 |
|---|---|---|
| `notebooks/colab_v6e_24h.ipynb` | 正式的 24 小時訓練（TPU v6e-1） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdyar423-gif/TuringAward2027/blob/claude%2Fmodest-carson-yi2d9g/notebooks/colab_v6e_24h.ipynb) |
| `notebooks/colab_v5e_smoke.ipynb` | 30 分鐘的管線檢查（免費層 TPU，預期 v5e-1；不計入 24 小時） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdyar423-gif/TuringAward2027/blob/claude%2Fmodest-carson-yi2d9g/notebooks/colab_v5e_smoke.ipynb) |
| `notebooks/colab_v6e_24h_v9.ipynb` | 24 小時訓練的替代配置 `configs/v9.py`（逐詞頁、無藏書，`docs/V9.md`；TPU v6e-1） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdyar423-gif/TuringAward2027/blob/claude%2Fmodest-carson-yi2d9g/notebooks/colab_v6e_24h_v9.ipynb) |

三個連結都從本倉庫的 `claude/modest-carson-yi2d9g` 分支載入 notebook（Colab 的網址把分支名當成單一路徑段，所以分支名裡的斜線寫成 `%2F`）。打開後不必改任何參數就能開始：第 1 格已填好本倉庫與分支，notebook 的中繼資料已指定 TPU 型號（v6e-1；管線檢查為 v5e-1）。連線前仍請在「執行階段 → 變更執行階段類型」確認硬體（第 2 格遇到不符的裝置會停下），再依 `RUNBOOK.md` 第 2 節的順序執行：第 1、1b、2 到 5 格，然後第 6 格。v9 的 notebook 與 24 小時 notebook 只差第 1 格的三個參數（配置、執行目錄、Drive 目錄），由 `scripts/notebook_variants.py` 產生；正式訓練只跑其中一個。這三個 notebook 還沒有在 Colab 上實際執行過。

### 從終端機直連 TPU（Colab CLI）

[Colab CLI](https://github.com/googlecolab/google-colab-cli)（Linux、macOS，Python 3.12 以上）可以不開瀏覽器，直接從終端機建立 Colab 的 TPU 執行階段並在上面執行。在本倉庫的 clone 裡：

    pip install google-colab-cli
    python scripts/colab_cli.py start --smoke        # 先做：v5e-1 TPU 上 30 分鐘的管線檢查
    python scripts/colab_cli.py down --smoke --yes   # 檢查完把那個執行階段還回去
    python scripts/colab_cli.py start                # v6e-1 TPU 上的 24 小時訓練（會掛載 Drive 做鏡像）
    python scripts/colab_cli.py status               # 查看進度（要在執行 start 的那台機器上）
    python scripts/colab_cli.py fetch                # 訓練完成後下載匯出並校驗
    python scripts/colab_cli.py down --yes           # 歸還執行階段：這一步才會停止計費

`start` 會依序執行 `colab sessions`（第一次使用時登入）、`colab new -s bes24h --tpu v6e1`（建立執行階段，從這一刻起計費）、`colab drivemount`（24 小時訓練預設鏡像到 Drive，需要您在瀏覽器授權；`--no-drive` 可關閉，但那樣執行階段一被回收訓練就全部遺失）與 `colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py`。

- **登入**：CLI 會印出一個授權連結，在瀏覽器同意後 Google 會顯示一段授權碼，請把它貼回 CLI 等待輸入的那個終端機，不要貼到別處（它等同於密碼）。
- **守護程式**：CLI 建立的執行階段只在核心有東西在執行時才會存活，所以 `scripts/colab_tpu.py` 會留在核心裡守著訓練直到結束。`start` 印出「== starting」之後，按 Ctrl-C 或關掉終端機只是離線，訓練照常進行；在那之前中斷則什麼都還沒開始，而執行階段可能已經在計費（`start` 會提醒）。
- **不要對同一個工作階段下 `colab exec`、`colab repl` 或 `colab restart-kernel`**：前兩者會無聲地排在守護程式後面，第三者會把守護程式殺掉。
- **停止計費**：守護程式結束或送出 `release` 只是不再維持執行階段，仍會計費到 Colab 回收為止；`down --yes` 才是立刻歸還。
- 執行階段上跑的是 GitHub 上這個分支的程式，不是本機 clone；原本要在 notebook 第 1 格改的設定，改用 `start` 的選項（`--budget-minutes`、`--config` 等）。

其餘指令（`send stop|resume|package|release`、`url`）與注意事項見 `RUNBOOK.md` 第 2.1 節。這條路線只用替身的 `colab` 指令測試過，沒有在 Colab 上實際執行。

## Result of the 24-hour run (2026-11-24)

`configs/v9.py` (per-token pages, no written library, `docs/V9.md`; the run began as `configs/final.py`
and its checkpoint was grafted into v9 by a function-preserving map at step 29,722, `docs/CHANGES.md`
section 19.7) was trained from scratch on one TPU v6e-1 (86,054 s of allocated TPU time against a
limit of 86,400 s; step 51,812; 1.358e10 training tokens). The official model is in
`release/main16-v9/`: 98,036,104 stored 16-bit numbers, 19,631,240 read per generated token. Blind
comparison on the 200 held-out test prompts (Opus 5.5 agents read blinded pairs of replies in random
order, every pair twice with the sides exchanged; wins / losses / ties are ours, a tie includes
"both bad"):

| opponent (its recommended settings) | W / L / T | score (95% interval) |
|---|---|---|
| MobileLLM-R1.5-140M | 116 / 15 / 69 | 0.752 [0.688, 0.807] |
| Falcon-H1-Tiny-90M-Instruct | 35 / 108 / 57 | 0.318 [0.257, 0.385] |

Stated plainly: the model is preferred over MobileLLM-R1.5 and is clearly behind Falcon-H1-Tiny, so
the goal of beating both was half met. 62.7% of MobileLLM-R1.5's replies to these prompts are degenerate under its recommended preset in
this harness, so that lead clears a low bar. Giving Falcon
the same six samples per reply does not change the result (35 / 112 / 53 with its own log-probability
as the selector, 29 / 116 / 55 with our selector); with a single particle our model scores
22 / 122 / 56 against Falcon. The readers knew which two systems a set compared (the file titles carried the pair
set's name) but not which side was which; the pre-registered automatic judge, which never sees
those names, gives 12 / 61 / 127 against Falcon, the same direction. Every number, the control
arms, exact scores, the automatic judge, the deviations from the protocol, the incidents of the
run and the corrections of an independent audit: `docs/CHANGES.md` section 19.10 (Chinese). The
final revision of the paper (`docs/paper/paper.md`) is still being updated; the copy in the
repository does not have the test-split numbers yet. Restoring the model, chatting
with it and running the blind arena yourself: `release/main16-v9/README.md`.

## What this is

A language model with at most 100M stored numbers (each at most 16 bits: weights, lookup tables,
the compressed library and its index) and at most 20M numbers read per generated token, designed
to be trained from scratch on one TPU v6e within 24 hours and judged blind against two small
open models.

**The idea in one paragraph.** Every step of the model is "posterior proportional to prior times
evidence", computed exactly wherever that is possible: inside attention, each head's calibrated
sink makes softmax attention an exact Bayesian estimate (a constant value channel that reads out
its confidence exists but stayed under the 1% bar in the ablations and is off in the final
configuration); at the output, exact in-context n-gram counts computed on the host (a candidate tape
with Pitman-Yor statistics) enter a closed-form output law next to the network's distribution;
over a whole reply, a few particles of the same model are steered by exact potentials (registers,
compiled constraints, a calculator) and chosen by a satisficing rule. A written library (Simple
English Wikipedia and question-answer pairs) is retrieved behind a calibrated gate and counts
against the 100M. The core is a looped transformer (2 shared blocks, 8 loops, d = 768). Plan,
mathematics and honest odds: `docs/PLAN.md`, `docs/MATH.md`, `docs/NEGATIVE_RESULTS.md`.

**Decoding defaults** (`bes.config.DecodeCfg`): 6 particles; temperature 0.6 applied to the
network's distribution only, `p_T = (softmax(z / T) + masses) / (1 + M)`, because the candidate
masses are calibrated probabilities that tempering would sharpen into copy loops
(`docs/CHANGES.md` section 10); the masses of candidates from the reply being written are
weighted x0.25; replies of up to 768 tokens; no cap on verbatim copies (when set, it applies to
library passages only).

**Whether the library ships** is decided after training by the decode selection of
`scripts/run_24h.py`: factual questions (evaluation splits that the library's store excludes, and
head-entity probes) are answered with the library on and off under the same seeds
(`python -m bes.eval.proxy factual`, `docs/EVAL_PROTOCOL.md` section 10). The library ships only
if it helps significantly more questions than it hurts (exact sign test, p < 0.05) and the dev
prompts' exact scores do not drop with it; otherwise the export ships without it and the
stored-number total drops by its 63.8M.

## Measured so far

The rows below were measured before the TPU run, on a 4-core CPU container (or compiled for the TPU
without running on it). What was measured on the TPU and after the run is in the section "Result of
the 24-hour run" above, in `docs/CHANGES.md` section 19 and in `docs/paper/`.

| fact | where |
|---|---|
| `configs/final.py`: 98,300,111 stored numbers with the built library; 19.28M numbers read per generated token on a step that retrieves (worst case of one retrieval; 19.03M on every other step), standard convention. The strict convention (union over the context window) is above 20M at T = 2048 for `final`; `configs/strict.py` stays under 20M in both conventions at T = 2048 for windows of up to about 918 distinct tokens by the budget's upper bound (19.85M at 840; real 2048-token windows held 606-837) | `docs/ACCOUNTING.md` |
| Library `lib_v2`: 63,768,391 stored numbers, worst-case read of one retrieval 245,223; gate at precision 0.8 on every dev set: head-entity probes 74.4% answered at 99.1% precision, common names 81.3% at 99.9%, NQ-open questions only 6.0% at 76.9% | `docs/E1_COVERAGE.md` |
| Candidate tape on a micro model (d = 128, 4M tokens, one seed): held-out loss 4.488 against 4.963 nats without it | `docs/CHANGES.md` section 9 |
| Temperature on the network only: with the full decoder on a tape-trained micro model at T = 0.5, repeated 4-gram rate 0.717 -> 0.207 and prompt copying 22.5% -> 2.0%; gate E5 (1M-token reference arm, 48 held-out chat prompts): with the defaults the repetition and loop rates are within error of the model trained without the tape (no loops) | `docs/CHANGES.md` sections 10 and 12, `docs/E5_SELF_GENERATION.md` |
| Particle decoding, measured on an opponent model as a stand-in: prompts with every constraint satisfied 66% (plain) -> 91% (filter) -> 98% (exact potentials), about 4x the compute | `docs/E9_DECODING.md` |
| Compile-only for v6e: train step peak 4.21 GiB with rematerialisation; without it the full micro-batch does not fit (32.93 GB of temporaries), half of it does (17.03 GiB) | `docs/TPU_NOTES.md` |
| Compile-only for v6e, `configs/v9.py`: train step peak 7.01 GiB with the "sorted" table gradient, 7.17 GiB with "onehot" (T=1024 and T=2048, remat), no scatter in either; decode step 1.89 GiB | `RUNBOOK.md` section 1.4 |
| Proxy blind test harness; Falcon-H1-Tiny-90M-Instruct on the dev half: math 66.7%, constraint prompts fully satisfied 78.7% | `docs/EVAL_PROTOCOL.md` |
| Gate E2, paired micro ablations (21 arms, 1M tokens each): removing the candidate tape costs 11% of the loss (19% on assistant tokens); value embeddings 1.5%; registers and candidate input features about 1.1% each on chat; confidence channel 0.7% and structural clocks 0.3% (both under the pre-registered 1% bar and switched off in `configs/final.py`); other settings within noise | `docs/E2_ABLATIONS.md`, `docs/CHANGES.md` section 14 |
| Gate E4, step-function probes on the micro model: copying a 20-token string 100% with the tape and 0% without; arithmetic through a calculator span with the forced result 56% at 1M tokens and 99% at 2M, 0% without forcing; exact counts do not show the step expected of the countdown registers (79% with them, 93% in the arm trained without them at 2M tokens), so that mechanism fails its probe and the registers are kept only for their effect on chat loss (stated as a deviation) | `docs/E4_PROBES.md`, `docs/E4_PROBES_2M.md`, `docs/CHANGES.md` section 14 |
| Rejected after testing: exact Gaussian-process attention, scale-recursive loops and others | `docs/NEGATIVE_RESULTS.md` |

## Running it

Tests (CPU; run the area you work on, each takes up to a few minutes; `-m slow` adds the long
ones):

    python -m pytest tests/model
    python -m pytest tests/train tests/eval

CPU rehearsal of the whole 24-hour pipeline in 12 minutes (downloads into `data_cache/`, the
library, the start-up benchmark and canary, training, sampling, DPO, decode selection with the
factual library measurement, the judged comparison against a GGUF opponent and a stand-in, the
export):

    python scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 12 --out runs/smoke
    python scripts/chat.py --export runs/smoke/final --show-diagnostics
    python scripts/arena.py --export runs/smoke/final --opponent falcon=recommended

The run reuses a built library only when it was built with the run's settings (permissive flag,
question-answer sources, and, recorded in `build_settings.json` by the run's library task, the
tokenizer and `LibraryCfg`); otherwise it moves the library aside and builds it again (about 10
minutes of host time plus downloads). A library built by hand with `scripts/build_library.py`
has no such record: it is reused when what it records itself matches, unless
`--override run.library_reuse_unrecorded=false`.

The real run: `RUNBOOK.md` (what to do before, during and after the 24 hours) and
`notebooks/colab_v6e_24h.ipynb`; a 30-minute free-tier check: `notebooks/colab_v5e_smoke.ipynb`.
The alternative without the library, `configs/v9.py` (a page per vocabulary id: private
feed-forward neurons and gains on the attention outputs;
98.0M stored numbers, 19.6M read per token in the standard convention; the strict one is above
20M, on real documents a median of 23.8M at T = 1024 and 26.7M at T = 2048, at most 28.2M;
`docs/V9.md`), starts with the same commands and `--config configs/v9.py` (`RUNBOOK.md`
section 1.4).
Both open in Colab from the links near the top of this file, with this repository, its branch
and the TPU type already filled in.
From a terminal instead of a browser tab: `python scripts/colab_cli.py start` uses the Colab
CLI (https://github.com/googlecolab/google-colab-cli) to create the v6e-1 runtime, to mount Drive
for the mirror and to send `scripts/colab_tpu.py`, which runs the notebook's cells there and
stays in the kernel as the run's supervisor; `down --yes` gives the runtime back, which is what
ends the charges (`RUNBOOK.md` section 2.1).
The code and every data file stay on the runtime's local disk (about 72 GB for the default plan:
raw downloads are deleted once tokenised); Google Drive receives only the run's records, logs, the
newest checkpoint and the export. Along the way the run keeps a hashed 0.3% of every chat, rag
and math source out of training (the sampling prompts and the side-car's probes come from it),
caches compiled programs under `RUN_DIR/jax_cache`, and its canary runs every compiled shape
before training starts.

## Layout

- `bes/` the package: `exact/` (registers, constraints, calculator, candidate tape), `library/`,
  `data/` (sources, downloads, tokenisation, loader), `model/`, `optim/`, `train/` (trainer,
  supervisor, benchmark), `decode/` (particle decoder, chat entry point), `eval/` (proxy harness,
  blind arena), `export.py`.
- `configs/` run configurations; `scripts/` command-line entry points; `tests/` tests;
  `docs/` plan (`docs/PLAN.md`), contracts (`docs/SPEC.md`, `docs/SPEC_MODEL_TRAIN.md`) and reports.
