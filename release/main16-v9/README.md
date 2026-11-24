# release/main16-v9：24 小時訓練的正式匯出（BES-100M-A20M）

這個目錄是 2026-11-23／24 那次 24 小時 TPU v6e-1 訓練（run `main16`，配置 `configs/v9.py`）依 run 自己的規則匯出的模型，切成 16 MB 的分片存放（GitHub 單檔上限 100 MB）。

| 項目 | 值 |
|---|---|
| 來源檢查點 | 第 51,812 步，訓練 1.358×10¹⁰ tokens；偏好最佳化（DPO，學習率 3e-6，被 run 的健康檢查接受）之後的權重 |
| 解碼設定 | run 的解碼選擇階段選了 `default`（6 個粒子、溫度 0.6、top-p 0.9、每則回覆最多 768 個 token、不用藏書） |
| 儲存的數字 | 98,036,104 個（全部 16-bit）；每個 token 讀取 19,631,240 個（標準讀法，`docs/ACCOUNTING.md`） |
| 16-bit 檢查 | 保留集 43,138 個 token：float32 2.34440、16-bit 2.34443 nats，差 0.000022（容許 0.002）：通過 |
| 分片 | `official_export.tar.000` … `.011`（共 196,956,160 位元組），`parts.json` 內有每片與整個 tar 的 SHA-256 |

## 還原與驗證

```bash
mkdir -p runs/main16
cat release/main16-v9/official_export.tar.* | tar -x -C runs/main16
python scripts/export.py --verify runs/main16/final
```

最後一行應為 `every file matches SHA256SUMS; held-out check passed`。需要 Python 3.13、`jax==0.11.1`（CPU 即可）與 `requirements-colab.txt` 的套件。模型卡在 `runs/main16/final/MODEL_CARD.md`。

## 對話

```bash
python scripts/chat.py --export runs/main16/final --device cpu
python scripts/chat.py --export runs/main16/final --device cpu --prompt "Give me three tips for a good night's sleep."
```

在 2 顆 CPU 核心上，第一則回覆含編譯約需 8 分鐘（6 個粒子的解碼器每步約 0.6 秒）；加 `--particles 1` 會快很多。有 TPU／GPU 時去掉 `--device cpu`。

## 自己做盲測（對 Falcon-H1-Tiny-90M-Instruct 與 MobileLLM-R1.5-140M）

```bash
python scripts/arena.py --export runs/main16/final --device cpu                       # 自己輸入題目
python scripts/arena.py --export runs/main16/final --device cpu --prompts proxy:test  # 200 題測試集
```

每題由三個模型作答，全部生成完才以隨機的 A／B／C 順序顯示；按 `a`／`b`／`c` 投票、`t` 平手、`x` 全差。投票完之前不要打開 `data_cache/arena/<name>/` 裡的 `key.jsonl`、`tallies.*`（它們寫明了哪個是哪個）。細節見 `RUNBOOK.md` 7.3。

## 這個模型是什麼、不是什麼

- 架構、預算與量測：`runs/main16/final/MODEL_CARD.md`、`docs/V9.md`、`docs/paper/paper.md`。
- 評估結果（盲測、精確計分）：`docs/paper/` 與 `docs/CHANGES.md` 第 19 節；結論請以那裡的數字為準。
- 訓練資料是公開資料集（`docs/DATA_SOURCES.md`）。訓練中途起拒答與推託樣本被過濾掉，也沒有任何安全性評估：這是研究用的產物，不適合直接部署給使用者。
