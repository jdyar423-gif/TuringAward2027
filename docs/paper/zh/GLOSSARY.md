# 翻譯規範與術語表（英文 → 繁體中文，台灣用語）

## 規則
1. 完整、忠實：每一句、每一個表格列、每一個數字、每一個引用標記（如 [7, 9, 11]）、每一個程式碼片段與檔案路徑（反引號內的文字）都要在譯文裡，內容不增不減。不要摘要、不要改寫論點、不要加譯註。
2. Markdown 結構原樣保留：標題層級與編號（`## 7. …`、`### 7.5 …`、`## Appendix M. …` 的編號與字母不變，文字翻譯；「Appendix M」寫成「附錄 M」）、表格的欄數與列數、清單、粗體與斜體、區塊引言、數學式、空行。表格的對齊列（`|---|---|`）照抄。
3. 不翻譯：數字與單位、反引號裡的任何文字、檔名與路徑、設定名（`final`、`v9`）、模型與資料集的專有名稱（Falcon-H1-Tiny-90M-Instruct、Falcon-H1-Tiny、MobileLLM-R1.5-140M、MobileLLM-R1.5、Opus 5.5、qwen3-4b-2507、UltraFeedback、ClimbMix、TPU v6e、JAX、XLA 等）、人名、論文標題（參考文獻整段保留原文，只翻「References」這個標題為「參考文獻」）、`[TBD: …]` 的方括號與 TBD 字樣（冒號後的說明要翻譯）。
4. 數字的寫法照原文（86,054、1.358 × 10¹⁰、0.752 [0.688, 0.807]、35 / 108 / 57、p = 0.012）。百分號、乘號、區間的方括號不變。
5. 標點用全形（，。；：「」（）），數字與英文字母和中文之間留半形空白（例：「在 200 題上」）。引號用「」，書名與節名不加書名號。
6. 文體：精確、平實的學術中文，短句優先；主詞「we」譯為「我們」；被動句可改主動。不要用大陸用語（用「資料」不用「数据」、「程式」不用「程序」、「記憶體」不用「内存」、「預設」不用「默认」、「檔案」不用「文件」）。
7. 原文若有明顯的筆誤，照譯，不要自行更正數字。

## 術語
| English | 中文 |
|---|---|
| stored numbers / numbers stored | 儲存的數字 |
| numbers read per token / reads | 每個 token 讀取的數字／讀取數 |
| parameter budget | 參數預算 |
| ledger | 帳（兩本帳：two ledgers；TPU time ledger：TPU 時間帳） |
| standard / strict reading (convention) | 標準讀法／嚴格讀法 |
| token | token（不譯） |
| dense looped core | 稠密的迴圈核心 |
| block application | 區塊套用 |
| per-token pages | 逐詞頁 |
| written library | 藏書 |
| candidate tape | 候選帶 |
| function-preserving graft | 保函數接枝（graft：接枝；grafted：接枝後的） |
| leaves (parameter leaves) | 葉（參數葉） |
| optimiser state | 最佳化器狀態 |
| learning rate schedule | 學習率排程 |
| wall clock / wall-clock budget | 牆鐘時間／牆鐘預算 |
| step time | 每步時間 |
| throughput | 吞吐量 |
| bytes moved | 搬移的位元組數 |
| layout | 佈局 |
| compiler | 編譯器 |
| micro-batch | 微批次 |
| checkpoint / snapshot | 檢查點／快照 |
| runtime (Colab runtime) | 執行階段 |
| development container | 開發容器 |
| export / official export | 匯出／正式匯出 |
| official model | 正式模型 |
| held-out loss / held-out set | 保留集損失／保留集 |
| preference optimisation (DPO) | 偏好最佳化 |
| decoding / decoder / decoding selection | 解碼／解碼器／解碼選擇 |
| particle (K particles) | 粒子 |
| selector | 選擇器 |
| blind comparison / blind test | 盲測比較／盲測 |
| readers (Opus readers) | 讀者（Opus 讀者） |
| reading (two readings on exchanged sides) | 判讀（左右互換的兩次判讀） |
| pre-registered protocol / registered judge | 事先註冊的協定／註冊評審 |
| automatic judge | 自動評審 |
| pair / pair set | 配對／配對集 |
| wins / losses / ties | 勝／負／平 |
| both bad | 兩者皆差 |
| decisive pairs | 分出勝負的配對 |
| score | 分數 |
| interval (95% interval, Wilson interval) | 區間（95% 區間、Wilson 區間） |
| sign test | 符號檢定 |
| first-position preference | 對首位回覆的偏好 |
| control arm / K-matched / selector-matched | 對照組／同 K 對照／同選擇器對照 |
| arm | 組（實驗組別） |
| test split / dev split | 測試集／開發集 |
| prompt | 題目 |
| reply | 回覆 |
| two-turn / single-turn | 兩輪／單輪 |
| exact scores / exact scorer | 精確計分／精確計分器 |
| constraint | 限制 |
| degenerate reply | 退化回覆 |
| truncated | 截斷 |
| neutral titles / neutral-title replication | 中性標題／中性標題複核 |
| deviation (from the protocol) | 偏離（協定） |
| limitations | 侷限 |
| threats to validity | 效度威脅 |
| claims ledger | 主張帳 |
| audit / audited | 稽核／經稽核的 |
| ablation | 消融 |
| seed | 種子 |
| negative results | 負面結果 |
| reproducibility | 可重現性 |

## 交叉引用與其他固定寫法
- Section 7.5 → 第 7.5 節；Sections 5 and 6 → 第 5、6 節；Appendix M → 附錄 M；Table 7 → 表 7；Table M4 → 表 M4；Figure 3 → 圖 3；Figure C2 → 圖 C2；Equation (3) → 式 (3)；contribution 4 → 貢獻 4。
- 圖片語法 `![說明](figures/xxx.png)` 的路徑不變，只翻譯說明文字；圖說與表說（Figure 3. … / Table 7. …）翻譯為「圖 3：…」「表 7：…」。
- 論文標題（只出現在第 1 段）：「參數預算沒算到的事：在單顆 TPU v6e 上以一天訓練一個儲存 9,800 萬、每個 token 讀取 2,000 萬個數字的語言模型，並全程稽核」。
- 「our model」→「我們的模型」；「the opponent(s)」→「對手」；「the run」→「這次訓練」或「run」（指 24 小時的那一次時可用「這次 run」）；「the user」→「使用者」；「the operator」→「操作者」。
- 「Status of this draft」→「本稿狀態」；「What we do not claim」→「我們不主張的事」；「Reading:」→「解讀：」。
- 時間寫法照原文（24 November 2026, 21:10–22:55 UTC → 2026 年 11 月 24 日 21:10–22:55 UTC）。
