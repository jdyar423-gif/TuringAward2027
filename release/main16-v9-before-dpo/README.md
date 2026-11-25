# release/main16-v9-before-dpo：偏好最佳化（DPO）之前的權重

這個目錄是同一次 24 小時訓練（run `main16`，配置 `configs/v9.py`）在**偏好最佳化之前**的權重匯出：訓練階段最後一個檢查點（第 51,812 步，1.358×10¹⁰ 個訓練 token）直接量化成 16-bit，沒有經過 DPO。**它不是正式模型**；正式模型是 DPO 之後的權重，在 `release/main16-v9/`。兩者的配置、分詞器、儲存與讀取的數字個數、解碼設定都相同，只有權重不同。

| 項目 | 值 |
|---|---|
| 來源檢查點 | `train/ckpt/ckpt-000022-step-000051812`（訓練階段的最後一個檢查點；匯出時間 2026-11-24 18:37:02 UTC，在偏好最佳化開始之前） |
| 解碼設定 | 與正式模型相同：6 個粒子、溫度 0.6、top-p 0.9、每則回覆最多 768 個 token、不用藏書 |
| 儲存的數字 | 98,036,104 個（全部 16-bit）；每個 token 讀取 19,631,240 個 |
| 16-bit 檢查 | 保留集 43,138 個 token：float32 2.32877、16-bit 2.32881 nats，差 0.000043（容許 0.002）：通過 |
| `weights16.npz` 的 SHA-256 | `634230225867f197ed75593680c6dce3a3828e3d404787573559f2a510a9073a`（正式模型是 `2da7cbd2…41afcf4`） |
| 分片 | `trained_export.tar.000` … `.011`（共 196,352,000 位元組），`parts.json` 內有每片與整個 tar 的 SHA-256 |

這份匯出就是測試集「偏好最佳化之前」那一組回覆（run id `ours-export-step51812-ef5599caa3974edc/final-trained-9a89782942-s0`）所用的匯出，於 18:37 在 runtime 上打包、取回，分片的雜湊與當時的 `parts.json` 相同。

## 還原與驗證

```bash
mkdir -p runs/main16-before-dpo
cat release/main16-v9-before-dpo/trained_export.tar.* | tar -x -C runs/main16-before-dpo
python scripts/export.py --verify runs/main16-before-dpo/export
```

最後一行應為 `every file matches SHA256SUMS; held-out check passed`（解開後的目錄名是 `export`，不是 `final`）。需要的環境與 `release/main16-v9/README.md` 相同。

## 對話與盲測

```bash
python scripts/chat.py --export runs/main16-before-dpo/export --device cpu
python scripts/arena.py --export runs/main16-before-dpo/export --device cpu --prompts proxy:test
```

## 它與正式模型（DPO 之後）的差別：量到了什麼

200 題測試集，同一個解碼設定（`docs/CHANGES.md` 第 19.10 節）：

- **盲測偏好**（Opus 讀者，每對兩次判讀、左右互換）：DPO 之後對 DPO 之前是 59 勝／34 負／107 平（p = 0.012），讀者偏好 DPO 之後的回覆。這一組的檔案標題帶有配對集名稱，也沒有中性標題複核與註冊評審的判讀。
- **精確計分**沒有跟著變好：限制全數滿足 DPO 之前 0.854（41／48 題）、之後 0.792（38／48）；退化回覆 1.2% 對 2.0%；數學兩者都是 0.200（6／30）。
- **保留集損失**：DPO 之前 2.3288、之後 2.3444 nats（16-bit，上表與 `release/main16-v9/README.md`）。
- DPO 之前的權重**沒有**對 Falcon-H1-Tiny 或 MobileLLM-R1.5 做過盲評（配對集已建立，但沒有判讀）；對兩個對手的結果都是正式模型的。

所以「哪一份比較好」要看用哪一把尺：讀者的偏好選 DPO 之後，精確計分與保留集損失略偏向 DPO 之前。run 的規則事先寫定以 DPO 之後、通過健康檢查的權重為正式模型。

訓練資料、授權與安全方面的說明同 `release/main16-v9/README.md`：研究用的產物，沒有做過安全性評估，不適合直接部署給使用者。
