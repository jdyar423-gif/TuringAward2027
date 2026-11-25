# release/m6v-before-dpo：六小時續訓在偏好最佳化（DPO）之前的權重

這個目錄是六小時續訓（run `m6v`，配置 `configs/m6v.py`，由 `release/main16-v9-before-dpo` 出發）在**偏好最佳化之前**的權重：訓練階段最後一個檢查點（第 13,615 步，本次 3.57×10⁹ 個訓練 token；連同 24 小時那次共 1.72×10¹⁰ 個）直接匯出，沒有經過 DPO。DPO 之後、實際出貨的權重在 `release/m6v/`。

| 項目 | 值 |
|---|---|
| 來源檢查點 | `train/ckpt/ckpt-000022-step-000013615`（訓練階段的最後一個檢查點；在 TPU 執行進行 DPO 時取回，之後在 CPU 上匯出） |
| 儲存的數字 | 98,036,104 個（全部 16-bit）；每個 token 讀取 19,631,240 個（v9 的完整匯出，沒有淘汰詞列、沒有藏書） |
| 16-bit 檢查 | 保留集 43,138 個 token：float32 2.29436、16-bit 2.29444 nats，差 0.000084（容許 0.002）：通過 |
| 保留集損失（各領域，16-bit） | 對話 1.2563、網頁 2.7578（DPO 之後是 1.2682、2.7702；續訓前是 1.3051、2.6805） |
| `weights16.npz` 的 SHA-256 | `40b0fdb658be1824cc0d1a6053b37c589aa1bc60282cf150272b33ac4cef241e`（DPO 之後是 `0d6705884f9fb602…`） |
| 分片 | `trained_export.tar.000` … `.011`（共 196,321,280 位元組），`parts.json` 內有每片與整個 tar 的 SHA-256 |

## 還原與驗證

```bash
mkdir -p runs/m6v-before-dpo
cat release/m6v-before-dpo/trained_export.tar.* | tar -x -C runs/m6v-before-dpo
python scripts/export.py --verify runs/m6v-before-dpo/export
```

解開後的目錄名是 `export`。要做成與出貨模型相同的形式（淘汰稀有詞的列、掛上藏書與使用者教的條目的空間），把 `release/m6v/README.md` 重建步驟裡的 `--export` 換成 `runs/m6v-before-dpo/export` 即可。

## 它與 DPO 之後的差別：量到了什麼

- **保留集損失**：DPO 讓對話與網頁各升高約 0.01 nats（上表；我固定的 8 列量法是對話 1.528 → 1.542、網頁 2.846 → 2.857）。
- **DPO 的驗收**（執行內）：保留的 1,823 組偏好對，準確率 0.500 → 0.690；學習率 3×10⁻⁶ 的候選通過長度、重複與損失的檢查。
- **這組權重沒有對 Falcon-H1-Tiny 做過盲評。** 測試集只用於 DPO 之後的出貨模型（`release/m6v/README.md`）。24 小時那次的同類比較（DPO 前後，Opus 讀者）是 DPO 之後 59 勝 34 負，見 `release/main16-v9-before-dpo/README.md`；這次沒有重做。
