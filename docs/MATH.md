# 數學命題與檢驗（對應計畫 §3）

原理：模型的每一步都是「後驗 ∝ 先驗 × 證據」。先驗是 20M 活躍參數學到的平滑規律；證據來自語境、藏書與精確演算法。能精確算的就精確算，其餘才交給網路或粒子。

下列每條命題都有證明草稿，並在 `tests/math/test_claims.py` 以獨立的 NumPy 計算檢驗（測試名稱列在各條之後）。對抗審稿對這套數學的評語是「嚴謹、幾乎每條都有前例、新穎度 2–3/10」；這份文件是計帳語言與設計依據，不是新定理的宣稱。

## 尺度一：層內——注意力是語境內估計

**B1　帶匯點的 softmax 注意力是精確的貝氏估計。**
設鍵為 one-hot（精確匹配）、核銳度 β→∞，匯點 logit 為 β + log λ、匯點的值為先驗 m₀。對查詢 a，匹配的 n_a 個鍵各得權重 1/(n_a+λ)，匯點得 λ/(n_a+λ)，所以輸出為 (Σ匹配的值 + λ·m₀)/(n_a+λ)。值取後繼的 one-hot、m₀ 取均勻分佈、λ = Vα 時，恰為對稱 Dirichlet(α) 先驗下馬可夫鏈的後驗預測 (n_ab+α)/(n_a+Vα)。匯點以外的總權重 n_a/(n_a+λ) 就是「信心」。
前半已見於 2026 年文獻（歸納頭內插 n-gram）；我們把它當成設計依據：每個頭都有一個可學 logit 的匯點與一個常數 1 的值通道。
檢驗：`test_b1_softmax_with_sink_is_dirichlet_posterior`。

**B2　精確高斯過程注意力與它和 B1 的關係。**
令 A = κ(K,K) + λI = LLᵀ，U = L⁻¹κ(K,Q)，Ṽ = L⁻¹V。因為 L 的前 i×i 主子陣就是前綴的 Cholesky 因子，所有前綴的後驗均值與變異可一次算出：o_i = Σ_{j≤i} U_{ji} ṽ_j，s_i² = 1 − Σ_{j≤i} U_{ji}²。
與匯點 softmax 的權重 w_sink = c/(λ + 1ᵀc) 相比，有恆等式
 w_GP − w_sink = A⁻¹((1ᵀc)·c − G·c)/(λ + 1ᵀc)，
兩者相等當且僅當 G·c = (1ᵀc)·c。匯點 softmax 等價於「常數核（所有鍵完全相關）加上隨查詢而定的雜訊 λ/c_j」的高斯過程；精確 GP 的差別只在鍵與鍵之間的相關（不稀釋、不被重複鍵佔滿）。
檢驗：`test_b2_one_cholesky_gives_every_prefix_posterior`、`test_b2_gp_weights_differ_from_sink_weights_by_key_correlations`。

**B3　判決：不採用 GP 頭**（細節見 `docs/NEGATIVE_RESULTS.md`）。

**B4　語言是冪律的。** 語境內快取的最優收縮是 Pitman–Yor 折扣 (n_b − d)₊/(n+θ) + (θ + d·K)/(n+θ)·基底，需要後繼的種類數 K；Dirichlet 與 GP 的線性形式 n/(n+λ) 表達不了。真實文本實驗中前者比固定權重好 0.025–0.048 nats，後者只好 0.003–0.008。設計後果：n-gram 計數頭同時輸出 (n, n_b, K)。

## 尺度二：輸出——先驗 × 局部證據，精確歸一

**T1　候選的價值是粗化後的 KL。** 候選集 U，把詞表粗化為 {u₁},…,{u_m}, V∖U。對任意加性信任度 g，
 E_P[log q^g(Y) − log q(Y)] = KL(P_U‖Q_U) − KL(P_U‖Q^g_U)，
最大值 KL(P_U‖Q_U) 在 g_j = log(p_j/q_j) − log(p₀/q₀) 取得；增益對 g 為凹。信任度為零時與基線完全相同，所以每種候選都能零初始化並隨時關閉。
檢驗：`test_t1_value_of_candidates_is_a_coarsened_kl`。

**T2　複製是跑在網路之上的 LZ77。** 以「延續次數／目前仍匹配的位置數」預測的複製器，對一段在藏書中唯一的片段恰付 log₂(M/occ) 位元（逐步機率相乘後望遠鏡消去）。檢索的作用是預付 log₂(M/M_ret)。最優信任度對匹配片段上網路累積的驚奇為仿射，並在續抄機率處飽和。
檢驗：`test_t2_copy_pointer_cost_telescopes_to_log_m_over_occ`。

**T3　歸一化的數值形式。** 恆等式 lse(z+S) = lse(z) + log1p(Σ_v softmax(z)_v·expm1(S_v)) 在 fp64 成立到 1e-12，但當大部分質量落在被大幅下修的 token 上時，fp32 會嚴重抵消；直接對合併後的 logits 做「減最大值的 log-sum-exp」才穩定。實作只用後者，恆等式僅作測試。
檢驗：`test_t3_tilt_identity_holds_but_complement_form_is_the_stable_one`。

## 尺度三：整段回覆——Feynman–Kac 路徑測度

目標分佈 π(x) ∝ p(x)·∏_t g_t(x_{≤t})·Φ(x)。

**F1／F2　粒子法無偏；遮罩的權重是被允許的質量。** 對 token 局部的硬約束，用「網路分佈乘上遮罩後重新歸一」作提議時，每步的增量權重恰為被允許的機率質量；K 個粒子的平均權重是歸一化常數 Z 的無偏估計。只做遮罩而不記權重，取樣的不是條件分佈。
檢驗：`test_f1_f2_smc_normaliser_is_unbiased_and_mask_weight_is_allowed_mass`（對小詞表做窮舉比對）。

**F3　最優扭轉。** ψ_t* = E[∏未來勢函數 | 前綴]（Doob h-變換）；必須以相鄰兩步的比值 ψ_t/ψ_{t−1} 進入權重，否則會重複計值。權重的變異取決於 ψ 對兄弟 token 的相對誤差。

**F4　極限。** K 選一至多把分佈傾斜 log K − (K−1)/K nats（K=8：1.20；K=32：2.50）。粒子與偏好優化互補：後者縮小 KL(π‖p)，前者處理剩下的部分。
檢驗：`test_f4_best_of_k_tilt_is_bounded_by_log_k`。

**F5　約束的前瞻可精確計算。** 可由有限自動機表示的約束，其扭轉可在代理馬可夫模型上以後向動態規劃算出；n-gram 替身實驗中單一粒子即把偏離由 5.9 降到 0.4 nats。

## 時間與記憶

**全迴圈最優。** 在迴圈律 N_eff = N₁ + r^φ·N_R、探測預算 N₁+N_R ≤ t、計算預算 N₁ + r·N_R ≤ c 下，最優為 N₁=0、r=c/t，N_eff* = t^{1−φ}c^φ。
檢驗：`test_all_recurrent_allocation_is_optimal_under_the_loop_law`。

**兩價注水。** 參數場各層有 M_l 個格、每格 n_l 個數字；最小化 Σ E_l(1+n_l/n₀)^{−α}，受限於 Σn_l ≤ A 與 ΣM_l n_l ≤ N，最優解為 n_l = n₀[(αE_l/(n₀(λ_A+λ_N M_l)))^{1/(1+α)} − 1]₊。
檢驗：`test_two_price_water_filling_closed_form_is_the_optimum`。

**尺度配置律（負面結果）。** 若同一區塊跨尺度共享，尺度 s_b 的累積圈數最優為 R_b ∝ (E_b/w_b)^{1/(1+γ)}，γ = αφ。能量譜 E_b ∝ s_b^{−β} 且 β<1 時粗尺度應多繞圈；但在 γ≈0.1 下可約損失只降不到 5%。
檢驗：`test_loops_per_scale_follow_the_allocation_law`。

**暫存器代數。** 指令 x ↦ min(hi, max(lo, x+a)) 對合成封閉且可結合：(a₁,lo₁,hi₁) 之後接 (a₂,lo₂,hi₂) 等於 (a₁+a₂, f₂(lo₁), f₂(hi₁))。
檢驗：`test_clamp_shift_registers_form_a_monoid`。

**曝光下界。** 訓練得來的事實需要約 10²–10³ 次曝光；T_train ≈ 2.6×10¹⁰ tokens、事實佔 5%、Zipf(1) 時，任何架構都只學得會約 7 萬–68 萬條。這是把長尾知識「寫下來」的理由。
