# release/m6v-interim：24 小時權重＋藏書的出貨形式（續訓之前）

這不是新訓練出來的權重。它是 `release/main16-v9-before-dpo`（24 小時那次訓練在偏好最佳化之前的權重）做成六小時任務的**出貨形式**：淘汰稀有詞的頁列與增益列，把空出來的數字換成藏書（教師模型寫過的問答），並預留使用者教它新東西的空間。它存在的理由是一個量測結果：只做這件事、不動任何權重，對 Falcon-H1-Tiny 的結果就變好——開發集由 10 勝 52 負變成 33 勝 42 負，測試集是 27 勝 63 負（24 小時的正式模型是 12 勝 61 負）；**測試集上仍然明顯落後**（下表）。續訓後的模型在 `release/m6v/`（TPU 執行完成後提交）。

| 項目 | 值 |
|---|---|
| 權重 | `main16-v9-before-dpo` 的 64 個葉節點，其中 `gain/o`、`page/key`、`page/val` 只存 5,136 個詞的列（藏書文字裡最常見的 5,120 個詞＋對話格式的 16 個特殊詞），另有一張 8,192 項的列號表 |
| 藏書 | 全量藏書（399,699 則）中留一法貢獻度最高的 172,273 則；25,308,804 個數字 |
| 使用者教的條目 | 預留 60,000 個數字（`/teach 問題 || 答案`，可跨重啟） |
| 儲存的數字 | **99,943,020**（上限 100,000,000） |
| 每個 token 讀取的數字 | **19,991,241**（上限 20,000,000；其中藏書每則回覆最多讀 360,000） |
| 保留集損失（16-bit） | 淘汰前 2.0071、淘汰後 2.0337 nats（32,044 個 token） |
| `weights16.npz` 的 SHA-256 | `a6eab1212a58613f2de4526ead7f4c0cc7a931242fa367e5bff46478c3b48e99` |
| 藏書的摘要值（digest） | `f81b5936aed288fe91c4f9eda7a04b9740708775b68bb78ef2aba738f9dcf405` |

## 重建（不需要 TPU，約五分鐘）

這個目錄只放重建所需的兩個小檔案；權重與全量藏書已經在 `release/main16-v9-before-dpo/` 與 `release/m6/library/`。

```bash
export BES_PROFILE=m6
python scripts/build_answers.py --unpack release/m6/library --out data_cache/library/answers_v1
python -c "from bes.train import init_from as IF; print(IF.export_dir('release/main16-v9-before-dpo', 'data_cache', print))"
python scripts/ship_m6v.py --export data_cache/init/main16-v9-before-dpo/export --keep 5120 \
    --credit release/m6v-interim/credit_answers_v1.npy --counts release/m6v-interim/token_counts.npy \
    --overlay 60000 --read-cap 360000 --out runs/m6v_interim/shipped
( cd runs/m6v_interim/shipped && grep -v "^#" ../../../release/m6v-interim/SHA256SUMS.expected | sha256sum -c - )
```

- `credit_answers_v1.npy`：全量藏書每一則的留一法貢獻度（`scripts/subset_answers.py --credit`；SHA-256 `0ab936dbc8026b13…`）。
- `token_counts.npy`：每個詞在全量藏書問答裡出現的次數（決定哪些詞留列；SHA-256 `a2f1035739c6408f…`）。
- `SHA256SUMS.expected`：重建後應該相同的檔案（權重、設定、分詞器、藏書的每個檔案）。`export.json` 與 `MODEL_CARD.md` 帶有建立時間與路徑，不在其中；`MODEL_CARD.md` 是我這邊那一份的副本。
- 同一份輸入在三台機器上（開發容器與兩台 Colab CPU 執行階段）建出相同的 `weights16.npz`；藏書子集自 `7df4cca` 起不含時間與路徑，建出的檔案與摘要值在不同機器上相同（測試 `tests/library/test_subset_answers.py`）。

## 使用

```bash
python scripts/chat.py --export runs/m6v_interim/shipped --device cpu --memory my_memory.jsonl
# 對話中：/teach 我的狗叫什麼名字？ || 你的狗叫 Pemberly。      /forget 我的狗叫什麼名字？
```

解碼設定：6 個粒子、溫度 0.6、top-p 0.9、每則回覆最多 768 個 token、藏書開啟、藏書自己的門檻（這組權重沒有為「鬆散範例」與相近程度標籤訓練過，不要對它開 `exemplar_floor` 或 `exemplar_tag`）。

## 量到了什麼

評審是事先登記的 `qwen3-4b-2507`（每對兩個順序、兩次一致才算勝負），對手是 Falcon-H1-Tiny-90M-Instruct 的建議設定。原始紀錄在 `lab/m6v/raw/`，說明在 `docs/CHANGES.md` 21.6。

**開發集 200 題**

| | 勝／負／平 | 分數 [95% 區間] | 符號檢定 p |
|---|---|---|---|
| 這個出貨形式（藏書開） | 33／42／125 | 0.477 [0.435, 0.520] | 0.36 |
| 同一組權重、把藏書關掉 | 10／52／138 | 0.395 | |
| 回覆就是藏書答案的 12 題 | 8／1／3 | 0.792 | 0.039 |
| 語境裡有相近問答的 71 題 | 20／12／39 | 0.556 | 0.22 |
| 沒有範例的 117 題 | 5／29／83 | 0.397 | <0.0001 |
| 對話 50 題 | 19／6／25 | 0.630 | 0.015 |
| 數學 30 題 | 2／12／16 | 0.333 | 0.013 |

**測試集 200 題**（登記的配對集 `m6v-before-vs-falcon`，只判讀一次）

| | 勝／負／平 | 分數 [95% 區間] | 符號檢定 p |
|---|---|---|---|
| **這個出貨形式（藏書開）** | **27／63／110** | **0.410 [0.344, 0.479]** | **0.0002** |
| 回覆就是藏書答案的 13 題 | 4／2／7 | 0.577 | 0.69 |
| 語境裡有相近問答的 65 題 | 18／18／29 | 0.500 | 1.00 |
| 沒有範例的 122 題 | 5／43／74 | 0.344 | <0.0001 |
| 對話 50 題 | 14／11／25 | 0.530 | 0.69 |
| 程式 30 題 | 5／10／15 | 0.417 | 0.30 |
| 指令 40 題 | 1／16／23 | 0.312 | 0.0003 |
| 數學 30 題 | 0／14／16 | 0.267 | 0.0001 |
| 寫作 50 題 | 7／12／31 | 0.450 | 0.36 |

兩個順序一致 68.0%；對首位回覆的偏好 −0.123。run id `ours-export-step51812-6a8d2688b157308e/interim-bd03abdea2-s0`。

**測試集上它仍然明顯輸給 Falcon。** 開發集的 0.477 在測試集是 0.410：三層的分數在測試集都比較低（逐字層 0.79 → 0.58、範例層 0.56 → 0.50、沒有範例 0.40 → 0.34），連完全不經過藏書的那一層也是，所以主要是兩批題目的難度不同，加上各 200 題的抽樣誤差（單一分數的 95% 區間約 ±0.07）。能確定的只有方向：用到藏書的 78 題是 22 勝 20 負，沒用到的 122 題是 5 勝 43 負。

同一組權重經偏好最佳化、沒有藏書時（24 小時任務的正式模型），同一個測試集、同一位評審是 12／61／127（0.378）。

## 它不是什麼

- 沒有範例可看的題目它照舊輸（數學、指令遵循、長篇寫作），因為權重沒有變。
- 藏書不是知識庫：條目是訓練對話裡的第一輪問答，覆蓋的是「常被問到的事」。問題一旦不在附近，它就是原來那個 2,000 萬活躍參數的模型。
- 出貨帳本用的是標準口徑（每個 token 讀到的不同數字）；嚴格口徑（把先前位置讀過的列也算進來）v9 本來就不滿足，這裡也一樣（`docs/ACCOUNTING.md`）。
