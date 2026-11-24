# References (Related Work)

Numbered in order of first citation in `related_work.md`. Every entry was read for the claim it supports; the `verified` line says what was read and what was checked (arXiv versions as fetched on 24 Nov 2026 from export.arxiv.org; web pages fetched the same day). A venue is given only where the arXiv comment, journal reference or the document itself states it; otherwise the entry says 'arXiv preprint'.

[1] Jonas Geiping, Tom Goldstein. *Cramming: Training a Language Model on a Single GPU in One Day*. arXiv preprint. arXiv:2212.14034 (v1 2022-12-28). https://arxiv.org/abs/2212.14034
    verified: abstract (v1) and the learning-rate-schedule paragraph of the PDF: one consumer GPU, one day, masked-LM pretraining from scratch, performance close to BERT; 'we re-scale the learning rate schedule so that it is tied to our budget' (following Izsak et al.).

[2] Peter Izsak, Moshe Berchansky, Omer Levy. *How to Train BERT with an Academic Budget*. arXiv preprint. arXiv:2104.07705 (v1 2021-04-15). https://arxiv.org/abs/2104.07705
    verified: abstract (v2) and PDF: 24 hours on a single low-end server (8 Titan-V 12 GB); 'synchronizing learning rate warmup and decay schedules with our 24 hour budget greatly improves model performance'.

[3] Mengtian Li, Ersin Yumer, Deva Ramanan. *Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints*. ICLR 2020. arXiv:1905.04753 (v1 2019-05-12). https://arxiv.org/abs/1905.04753
    verified: abstract (v4): budget-aware schedules; simple linear decay found robust and high-performing.

[4] Keller Jordan and contributors. *modded-nanogpt: the NanoGPT speedrun*. GitHub repository, 2024-2026. https://github.com/KellerJordan/modded-nanogpt
    verified: README fetched 2026-11-24: task = fastest training to 3.28 FineWeb validation loss on 8 x H100; record table (record 14, 2024-12-04, 'Value Embeddings'); feature list includes 'extra embeddings which are mixed into the values in attention layers (inspired by Zhou et al. 2024)' and a hashed bigram + trigram embedding table of 84.6M rows.

[5] Andrej Karpathy. *nanochat: The best ChatGPT that $100 can buy*. GitHub repository, 2025. https://github.com/karpathy/nanochat
    verified: README fetched 2026-11-24: full pipeline (tokenizer, pretraining, finetuning, evaluation, inference) on one 8 x H100 node; 'time to GPT-2' leaderboard (CORE metric), about 1.5-2 hours; nanochat/gpt.py: value embeddings in alternating layers with gate 3*sigmoid(W x[..., :12]) (ResFormer-style).

[6] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*. ICML 2024. arXiv:2402.14905 (v1 2024-02-22). https://arxiv.org/abs/2402.14905
    verified: abstract (v2) and the weight-sharing section of the PDF: deep-and-thin, embedding sharing, GQA; immediate block-wise weight sharing (adjacent block computed twice), MobileLLM-LS +0.7/+0.8 accuracy points at 125M/350M.

[7] Changsheng Zhao, Ernie Chang, Zechun Liu, Chia-Jung Chang, Wei Wen, Chen Lai, et al. *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*. ICLR 2026. arXiv:2509.24945 (v1 2025-09-29). https://arxiv.org/abs/2509.24945
    verified: abstract (v3), Fig. 2 and Table 3 of the PDF: ~2T curated tokens, 4.2T pre-training tokens (two 2T phases) plus two 100B mid-training phases and SFT; 140M model = 15 layers, d = 576, 140.2M.

[8] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024. arXiv:2306.13649 (v1 2023-06-23). https://arxiv.org/abs/2306.13649
    verified: abstract (v3): generalized knowledge distillation trains the student on its own generated sequences with teacher feedback; cited by the MobileLLM-R1.5 card as its on-policy KD method.

[9] Meta (facebook). *MobileLLM-R1.5-140M model card*. Hugging Face model card, 2025. https://huggingface.co/facebook/MobileLLM-R1.5-140M
    verified: card text fetched 2026-11-24 (repository files are gated; API metadata: Llama4ForCausalLM, base model MobileLLM-R1-140M-base): 'resume training from the final MobileLLM-R1 model ... perform an additional on-policy knowledge distillation (KD) step', teacher nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1.

[10] Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, et al. *Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance*. technical report, arXiv. arXiv:2507.22448 (v1 2025-07-30). https://arxiv.org/abs/2507.22448
    verified: abstract (v1): parallel hybrid of attention and state-space (Mamba) heads; sizes 0.5B to 34B.

[11] Falcon-LLM Team (Technology Innovation Institute). *Falcon-H1-Tiny: A series of extremely small, yet powerful language models redefining capabilities at small scale*. technical blog post (Hugging Face Space), published 15 Jan 2026. https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost
    verified: article source app/src/content/chapters/demo/{introduction,training-approach,tiny-h1-sft}.mdx read: 90M Instruct 'pretrained from scratch using SFT data, followed by a lightweight DPO stage'; 25% SFT data, 800 GT total; Muon with learnable multipliers; table: IFEval 50.11 -> 66.08 and AlpacaEval win rate 2.96 -> 9.43 with DPO; 'more than 1 epoch ... considerable performance degradation'; final DPO runs with learning rates 1e-06 and 3e-06 on 1 epoch only.

[12] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. *SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*. arXiv preprint. arXiv:2502.02737 (v1 2025-02-04). https://arxiv.org/abs/2502.02737
    verified: abstract (v1), Sec. 6 and the post-training section of the PDF: 135M on 2T tokens, 360M on 4T tokens, WSD schedule; post-training SFT on filtered SmolTalk, then DPO on UltraFeedback; MagPie-Ultra generated with Llama-3.1-405B-Instruct-FP8.

[13] Ronen Eldan, Yuanzhi Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*. arXiv preprint. arXiv:2305.07759 (v1 2023-05-12). https://arxiv.org/abs/2305.07759
    verified: abstract (v2): models below 10 million total parameters produce fluent, consistent stories on a synthetic dataset of simple vocabulary.

[14] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser. *Universal Transformers*. ICLR 2019. arXiv:1807.03819 (v1 2018-07-10). https://arxiv.org/abs/1807.03819
    verified: abstract (v3): parallel-in-time recurrent self-attentive model (shared block applied repeatedly), dynamic halting.

[15] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR 2020. arXiv:1909.11942 (v1 2019-09-26). https://arxiv.org/abs/1909.11942
    verified: abstract (v6) and Sec. 1 of the PDF ('Published as a conference paper at ICLR 2020'): two parameter-reduction techniques, the second being cross-layer parameter sharing.

[16] Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers*. ICLR 2025. arXiv:2502.17416 (v1 2025-02-24). https://arxiv.org/abs/2502.17416
    verified: abstract (v1): a k-layer transformer looped L times nearly matches a kL-layer model on addition, p-hop induction and math; competitive on downstream reasoning in language modelling.

[17] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*. arXiv preprint. arXiv:2502.05171 (v1 2025-02-07). https://arxiv.org/abs/2502.05171
    verified: abstract (v2): recurrent-depth model scaled to 3.5B parameters and 800B tokens.

[18] Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, et al. *Scaling Latent Reasoning via Looped Language Models*. arXiv preprint. arXiv:2510.25741 (v1 2025-10-29). https://arxiv.org/abs/2510.25741
    verified: abstract (v6): looped language models (Ouro) 1.4B and 2.6B trained on 7.7T tokens, matching up to 12B models; advantage attributed to knowledge manipulation, not capacity.

[19] Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu. *Parcae: Scaling Laws For Stable Looped Language Models*. arXiv preprint. arXiv:2604.12946 (v1 2026-04-14). https://arxiv.org/abs/2604.12946
    verified: abstract (v1): instability traced to large spectral norms of the injection parameters; Parcae constrains them; up to 6.3% lower validation perplexity than prior large looped models.

[20] Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, et al. *Hyper-Connections*. arXiv preprint. arXiv:2409.19606 (v1 2024-09-29). https://arxiv.org/abs/2409.19606
    verified: abstract (v3): hyper-connections as an alternative to residual connections; gains in LLM pre-training.

[21] Kristian Schwethelm, Daniel Rueckert, Georgios Kaissis. *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*. NeurIPS 2026 (spotlight; arXiv comment). arXiv:2604.21106 (v1 2026-04-22). https://arxiv.org/abs/2604.21106
    verified: abstract (v4), the implementation and optimisation paragraphs and Sec. 5.2 of the PDF: L = E + A(N_once + r^phi N_rec)^-alpha + B D^-beta, phi = 0.46; at r = 4 a 410M looped model ~ 580M non-looped at the training cost of 1B; hyperconnections (K = 2 lanes) raise phi to 0.65; prelude-recur-coda (2 + 16/r + 2 layers); MuonH with HyperP; code github.com/kschwethelm/looped-lm-scaling (README: MIT licence, close to nanochat).

[22] Zhanchao Zhou, Tianyi Wu, Zhiyun Jiang, Fares Obeid, Zhenzhong Lan. *Value Residual Learning*. arXiv preprint. arXiv:2410.17897 (v1 2024-10-23). https://arxiv.org/abs/2410.17897
    verified: abstract (v5): value residual connections (ResFormer); equivalent loss with 16.11% fewer parameters.

[23] Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston. *Hash Layers For Large Sparse Models*. arXiv preprint. arXiv:2106.04426 (v1 2021-06-08). https://arxiv.org/abs/2106.04426
    verified: abstract (v3): feed-forward layer hashed to different weights by the current token; competitive with Switch and BASE layers without routing parameters.

[24] Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. *Large Memory Layers with Product Keys*. NeurIPS 2019. arXiv:1907.05242 (v1 2019-07-10). https://arxiv.org/abs/1907.05242
    verified: abstract (v2): product-key memory; a 12-layer memory model outperforms a 24-layer baseline.

[25] Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Ghosh. *Memory Layers at Scale*. arXiv preprint. arXiv:2412.09764 (v1 2024-12-12). https://arxiv.org/abs/2412.09764
    verified: abstract (v2): trainable key-value memory layers up to 128B memory parameters, 1T tokens; beat dense models with twice the compute.

[26] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, et al. *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling*. ICML 2025. arXiv:2501.16975 (v1 2025-01-28). https://arxiv.org/abs/2501.16975
    verified: abstract (v2) and Fig. 5 of the PDF: 'L = 2.6754 - 0.0256 x log10 m', 500B tokens, OLMoE-1.3B; 400M model with 128x input vocabulary matches a 1B baseline.

[27] Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, et al. *Scaling Embedding Layers in Language Models*. NeurIPS 2025. arXiv:2502.01637 (v1 2025-02-03). https://arxiv.org/abs/2502.01637
    verified: abstract (v3): n-gram embeddings precomputed and stored off-accelerator; 1B accelerator-resident model beats a 1.9B baseline.

[28] Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. arXiv preprint. arXiv:2601.07372 (v1 2026-01-12). https://arxiv.org/abs/2601.07372
    verified: abstract (v2): hashed N-gram 'conditional memory' (Engram), U-shaped allocation law against MoE, 27B parameters, deterministic addressing with prefetch from host memory.

[29] Albert Tseng, Christopher De Sa. *L³: Large Lookup Layers*. ICML 2026. arXiv:2601.21461 (v1 2026-01-29). https://arxiv.org/abs/2601.21461
    verified: abstract (v3) and Sec. 3 of the PDF: L3(x,t) = W_mix[LayerNorm(W_up(V_t^T Softmax(K_t x))); x] with per-token K_t, V_t; LZW-based allocation, at least 1 and at most k rows per token; CPU-offloaded inference; models up to 2.6B active parameters.

[30] Róbert Csordás, Piotr Piękos, Kazuki Irie, Jürgen Schmidhuber. *SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention*. NeurIPS 2024. arXiv:2312.07987 (v1 2023-12-13). https://arxiv.org/abs/2312.07987
    verified: abstract (v3) and Table 6 of the PDF (47M models, WikiText-103): experts on V and O 12.27, dense 10-head 12.31, all four 12.57, Q and K 12.90 perplexity.

[31] Jiale Kang. *Memory Attention*. arXiv preprint (September 2026). arXiv:2609.28399 (v1 2026-09-23). https://arxiv.org/abs/2609.28399
    verified: abstract (v2) and Tables 4-5 of the PDF: V = K + Norm(M) with token-indexed M; placement ablation with shared keys and values, final loss H 2.574, Q 2.615, K 2.612, V 2.553, O 2.650; standard V = xW 2.591 (24 layers, d = 1024, 10B tokens).

[32] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, et al. *Mixtral of Experts*. arXiv preprint. arXiv:2401.04088 (v1 2024-01-08). https://arxiv.org/abs/2401.04088
    verified: abstract (v1): each token has access to 47B parameters but uses 13B active parameters.

[33] Google. *Gemma 3n model overview*. Google AI for Developers documentation, 2025-2026. https://ai.google.dev/gemma/docs/gemma-3n
    verified: page fetched 2026-11-24: 'E' prefix = effective parameters; E2B loads over 5 billion parameters in standard execution, about 1.91B effective with parameter skipping and Per-Layer Embedding caching.

[34] Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher. *Pointer Sentinel Mixture Models*. arXiv preprint. arXiv:1609.07843 (v1 2016-09-26). https://arxiv.org/abs/1609.07843
    verified: abstract (v1): mixture of copying from recent context and a softmax over the vocabulary.

[35] Edouard Grave, Armand Joulin, Nicolas Usunier. *Improving Neural Language Models with a Continuous Cache*. arXiv preprint (submitted to ICLR 2017). arXiv:1612.04426 (v1 2016-12-13). https://arxiv.org/abs/1612.04426
    verified: abstract (v1): continuous cache over past hidden states; link to cache models of count-based LMs.

[36] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis. *Generalization through Memorization: Nearest Neighbor Language Models*. ICLR 2020. arXiv:1911.00172 (v1 2019-11-01). https://arxiv.org/abs/1911.00172
    verified: abstract (v2): linear interpolation of an LM with a k-nearest-neighbour model; 15.79 perplexity on WikiText-103.

[37] Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, Hannaneh Hajishirzi. *Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens*. COLM 2024. arXiv:2401.17377 (v1 2024-01-30). https://arxiv.org/abs/2401.17377
    verified: abstract (v4): unbounded-n n-gram LM over 5T tokens with suffix arrays; complements neural LMs.

[38] Yee Whye Teh. *A Hierarchical Bayesian Language Model Based On Pitman-Yor Processes*. COLING/ACL 2006, pages 985-992. https://aclanthology.org/P06-1124/
    verified: ACL Anthology page and PDF Sec. 2: discount d and strength theta; reuse of a draw with probability (c_k - d)/(theta + c), new draw from the base G0 with probability (theta + d t)/(theta + c); interpolated Kneser-Ney recovered as approximate inference.

[39] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, et al. *In-context Learning and Induction Heads*. arXiv preprint (Transformer Circuits). arXiv:2209.11895 (v1 2022-09-24). https://arxiv.org/abs/2209.11895
    verified: abstract (v1): induction heads complete [A][B] ... [A] -> [B]; evidence they underlie in-context learning.

[40] Ekin Akyürek, Bailin Wang, Yoon Kim, Jacob Andreas. *In-Context Language Learning: Architectures and Algorithms*. arXiv preprint. arXiv:2401.12973 (v1 2024-01-23). https://arxiv.org/abs/2401.12973
    verified: abstract (v2): n-gram heads; hard-wiring them improves 340M-parameter models by up to 1.14 perplexity points (6.7%) on SlimPajama.

[41] Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li. *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation*. NeurIPS 2022. arXiv:2206.02369 (v1 2022-06-06). https://arxiv.org/abs/2206.02369
    verified: abstract (v2): sentence-level repetitions have a self-reinforcement effect.

[42] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. arXiv:2309.17453 (v1 2023-09-29). https://arxiv.org/abs/2309.17453
    verified: abstract (v4): attention sinks on initial tokens; a dedicated sink token in pre-training helps streaming.

[43] OpenAI, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, et al. *gpt-oss-120b & gpt-oss-20b Model Card*. model card, arXiv. arXiv:2508.10925 (v1 2025-08-08). https://arxiv.org/abs/2508.10925
    verified: PDF Sec. 2: 'Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks'.

[44] Sewon Min, Jordan Boyd-Graber, Chris Alberti, Danqi Chen, Eunsol Choi, Michael Collins, et al. *NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned*. PMLR, NeurIPS 2020 Competition and Demonstration Track. arXiv:2101.00133 (v1 2021-01-01). https://arxiv.org/abs/2101.00133
    verified: abstract (v2) and Sec. 1-3 of the PDF: tracks unrestricted, < 6 GiB, < 500 MiB, smallest system at 25%; the 500 MiB and smallest-system winners stored question-answer pairs.

[45] Tianqi Chen, Ian Goodfellow, Jonathon Shlens. *Net2Net: Accelerating Learning via Knowledge Transfer*. arXiv preprint (ICLR 2016 submission). arXiv:1511.05641 (v1 2015-11-18). https://arxiv.org/abs/1511.05641
    verified: abstract (v4): function-preserving transformations to deeper or wider networks.

[46] Tao Wei, Changhu Wang, Yong Rui, Chang Wen Chen. *Network Morphism*. arXiv preprint. arXiv:1603.01670 (v1 2016-03-05). https://arxiv.org/abs/1603.01670
    verified: abstract (v2): morphing a trained network with its function completely preserved (depth, width, kernel size, subnet).

[47] Sheng Shen, Pete Walsh, Kurt Keutzer, Jesse Dodge, Matthew Peters, Iz Beltagy. *Staged Training for Transformer Language Models*. arXiv preprint. arXiv:2203.06211 (v1 2022-03-11). https://arxiv.org/abs/2203.06211
    verified: abstract (v1) and Sec. 3.3 of the PDF: growth operators take the entire training state (parameters, optimizer state, learning-rate schedule); loss- and training-dynamics-preserving; for depth growth 'we copy m and v for the original model layers and set m and v to zero for the identity layers'; up to 22% compute savings.

[48] Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, et al. *LLaMA Pro: Progressive LLaMA with Block Expansion*. ACL 2024. arXiv:2401.02415 (v1 2024-01-04). https://arxiv.org/abs/2401.02415
    verified: abstract (v2) and the block-expansion section of the PDF: expanded blocks with zero-initialised output linear layers act as identity blocks.

[49] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, et al. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*. arXiv preprint. arXiv:2212.05055 (v1 2022-12-09). https://arxiv.org/abs/2212.05055
    verified: abstract (v2): sparse MoE initialised from a dense checkpoint; ~50% of the dense sunk cost.

[50] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint. arXiv:2106.09685 (v1 2021-06-17). https://arxiv.org/abs/2106.09685
    verified: PDF Sec. 4.1: 'zero for B, so Delta W = BA is zero at the beginning of training'.

[51] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, Jeremy Bernstein. *Muon: An optimizer for hidden layers in neural networks*. blog post, 2024. https://kellerjordan.github.io/posts/muon/
    verified: post (8 Dec 2024) read: Newton-Schulz orthogonalisation of SGD-momentum updates of 2D hidden-layer parameters, AdamW for the rest. (Author list as cited by the iso-depth paper; the post is by Keller Jordan with a contributors section.)

[52] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, et al. *Muon is Scalable for LLM Training*. arXiv preprint. arXiv:2502.16982 (v1 2025-02-24). https://arxiv.org/abs/2502.16982
    verified: abstract (v1): weight decay and per-parameter update scaling make Muon scale; ~2x computational efficiency over AdamW in scaling-law experiments.

[53] Noah Amsel, David Persson, Christopher Musco, Robert M. Gower. *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*. arXiv preprint. arXiv:2505.16932 (v1 2025-05-22). https://arxiv.org/abs/2505.16932
    verified: abstract (v5): minimax-optimal polynomial iteration for the polar factor, bfloat16-safe, used inside Muon. (The repository's bes/optim/polar.py states that its coefficients are this paper's.)

[54] Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao. *NorMuon: Making Muon more efficient and scalable*. arXiv preprint. arXiv:2510.05491 (v1 2025-10-07). https://arxiv.org/abs/2510.05491
    verified: abstract (v1): neuron-wise second-moment normalisation after orthogonalisation; 11.31% over Muon at 1.1B.

[55] Lizhang Chen, Jonathan Li, Kaizhao Liang, Baiyu Su, Cong Xie, Nuo Wang Pierse, et al. *Cautious Weight Decay*. arXiv preprint. arXiv:2510.12402 (v1 2025-10-14). https://arxiv.org/abs/2510.12402
    verified: abstract (v2): weight decay only on coordinates whose sign agrees with the update.

[56] Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang. *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization*. arXiv preprint. arXiv:2606.16899 (v1 2026-06-15). https://arxiv.org/abs/2606.16899
    verified: abstract (v2): Hyperball fixes the Frobenius norms of weight matrices and of their updates; Muon Hyperball 20-30% token-equivalent speed-up over weight decay up to 1.2B.

[57] Liliang Ren, Yang Liu, Yelong Shen, Weizhu Chen. *Rethinking Language Model Scaling under Transferable Hypersphere Optimization*. arXiv preprint. arXiv:2603.28743 (v1 2026-03-30). https://arxiv.org/abs/2603.28743
    verified: abstract (v2): learning-rate transfer under the Frobenius-sphere constraint with Muon; weight decay a first-order no-op; optimal LR follows a data power law with exponent 0.32.

[58] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv preprint. arXiv:2205.14135 (v1 2022-05-27). https://arxiv.org/abs/2205.14135
    verified: abstract (v2): IO-aware exact attention by tiling between HBM and on-chip SRAM.

[59] Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv preprint. arXiv:2307.08691 (v1 2023-07-17). https://arxiv.org/abs/2307.08691
    verified: abstract (v1): better work partitioning; 50-73% of peak FLOP/s on A100.

[60] JAX authors. *Pallas: a JAX kernel language*. JAX documentation, 2026. https://docs.jax.dev/en/latest/pallas/index.html
    verified: page fetched 2026-11-24: 'Pallas is an extension to JAX for writing custom kernels for GPU and TPU'.

[61] JAX authors. *Splash attention (Sparse Flash Attention) for TPU, jax/experimental/pallas/ops/tpu/splash_attention*. JAX 0.11.1 source code, 2023-2026. https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention
    verified: installed jax 0.11.1 source read: module docstring 'Implementation of Sparse Flash Attention, a.k.a. Splash attention'.

[62] Google (openxla). *Tokamax: a Pallas custom kernel library*. software, version 0.0.18, 2025-2026. https://github.com/openxla/tokamax
    verified: package metadata of tokamax 0.0.18 read: 'a library of custom accelerator kernels ... built on top of JAX and Pallas', including splash attention.

[63] Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, Torsten Hoefler. *Data Movement Is All You Need: A Case Study on Optimizing Transformers*. MLSys 2021. arXiv:2007.00072 (v1 2020-06-30). https://arxiv.org/abs/2007.00072
    verified: abstract (v3): data movement is the key bottleneck, training memory-bound; data movement reduced by up to 22.91%, 1.30x on a BERT encoder layer.

[64] Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka. *Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs*. arXiv preprint. arXiv:2306.03081 (v1 2023-06-05). https://arxiv.org/abs/2306.03081
    verified: abstract (v2): SMC steering of LLMs via probabilistic programs (LLaMPPL).

[65] Stephen Zhao, Rob Brekelmans, Alireza Makhzani, Roger Grosse. *Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo*. arXiv preprint. arXiv:2404.17546 (v1 2024-04-26). https://arxiv.org/abs/2404.17546
    verified: abstract (v1): learned twist functions estimating future potential; bidirectional SMC bounds.

[66] João Loula, Benjamin LeBrun, Li Du, Ben Lipkin, Clemente Pasti, Gabriel Grand, et al. *Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo*. arXiv preprint. arXiv:2504.13139 (v1 2025-04-17). https://arxiv.org/abs/2504.13139
    verified: abstract (v2): SMC with syntactic and semantic constraints lets small open models outperform models over 8x larger.

[67] Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. *Rollout Roulette: A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods*. arXiv preprint. arXiv:2502.01618 (v1 2025-02-03). https://arxiv.org/abs/2502.01618
    verified: abstract (v5): particle-based Monte Carlo for inference-time scaling; 4-16x better scaling rate than deterministic search.

[68] Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, Loris D'Antoni. *Grammar-Aligned Decoding*. NeurIPS 2024. arXiv:2405.21047 (v1 2024-05-31). https://arxiv.org/abs/2405.21047
    verified: abstract (v4): greedy constrained decoding distorts the LM distribution; ASAp samples the conditional.

[69] Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D'Amour, Jacob Eisenstein, Chirag Nagpal, et al. *Theoretical guarantees on the best-of-n alignment policy*. ICML 2025. arXiv:2401.01879 (v1 2024-01-03). https://arxiv.org/abs/2401.01879
    verified: abstract (v3): the common expression log(n) - (n-1)/n for the KL of best-of-n is an upper bound, not an equality.

[70] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023. arXiv:2203.11171 (v1 2022-03-21). https://arxiv.org/abs/2203.11171
    verified: abstract (v4): sample several reasoning paths and take the most consistent (majority) answer.

[71] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, et al. *Instruction-Following Evaluation for Large Language Models*. arXiv preprint. arXiv:2311.07911 (v1 2023-11-14). https://arxiv.org/abs/2311.07911
    verified: abstract (v1): 25 types of verifiable instructions, about 500 prompts.

[72] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, et al. *Training Verifiers to Solve Math Word Problems*. arXiv preprint. arXiv:2110.14168 (v1 2021-10-27). https://arxiv.org/abs/2110.14168
    verified: PDF Sec. 4 and App. C: calculator annotations <<...>> injected into training data; at test time the calculator overrides sampling after '='.

[73] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, et al. *s1: Simple test-time scaling*. arXiv preprint. arXiv:2501.19393 (v1 2025-01-31). https://arxiv.org/abs/2501.19393
    verified: abstract (v3): budget forcing ends the thinking phase (or extends it with 'Wait').

[74] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv preprint. arXiv:2305.18290 (v1 2023-05-29). https://arxiv.org/abs/2305.18290
    verified: abstract (v3): preference optimisation as a classification loss without an explicit reward model.

[75] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, et al. *Zephyr: Direct Distillation of LM Alignment*. arXiv preprint. arXiv:2310.16944 (v1 2023-10-25). https://arxiv.org/abs/2310.16944
    verified: abstract (v1) and Sec. 3-4 of the PDF: distilled DPO for a 7B model on binary preferences built from UltraFeedback.

[76] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, et al. *UltraFeedback: Boosting Language Models with Scaled AI Feedback*. ICML 2024. arXiv:2310.01377 (v1 2023-10-02). https://arxiv.org/abs/2310.01377
    verified: abstract (v2): over 1 million GPT-4 feedback annotations for 250k conversations.

[77] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, et al. *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv preprint. arXiv:2411.15124 (v1 2024-11-22). https://arxiv.org/abs/2411.15124
    verified: abstract (v5) and Table of SFT data in the PDF: SFT, DPO, RLVR; SFT mixture includes Tulu 3 CoCoNot (10,983), WildJailbreak and WildGuardMix safety subsets.

[78] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.05685 (v1 2023-06-09). https://arxiv.org/abs/2306.05685
    verified: abstract (v4), Table 2 and the 'Swapping positions' paragraph of the PDF: position, verbosity and self-enhancement biases; GPT-4 consistent under swapping in 65.0% of cases (default prompt); 'call a judge twice by swapping the order ... only declare a win when an answer is preferred in both orders'.

[79] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, et al. *Large Language Models are not Fair Evaluators*. arXiv preprint. arXiv:2305.17926 (v1 2023-05-29). https://arxiv.org/abs/2305.17926
    verified: abstract (v2): order of appearance changes LLM-judge rankings (Vicuna-13B beats ChatGPT on 66 of 80 queries); balanced position calibration.

[80] Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*. COLM 2024. arXiv:2404.04475 (v1 2024-04-06). https://arxiv.org/abs/2404.04475
    verified: abstract (v2): regression-based length control of AlpacaEval; Spearman correlation with Chatbot Arena 0.94 -> 0.98.

[81] Arjun Panickssery, Samuel R. Bowman, Shi Feng. *LLM Evaluators Recognize and Favor Their Own Generations*. arXiv preprint. arXiv:2404.13076 (v1 2024-04-15). https://arxiv.org/abs/2404.13076
    verified: abstract (v1): LLM evaluators recognise and favour their own generations.

## Also verified, not cited in Related Work

Offered to the other sections (data, method, kernels); same standard of verification.

[A1] Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, et al. *Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training*. NeurIPS 2025. arXiv:2504.13161 (v1 2025-04-17). https://arxiv.org/abs/2504.13161
    verified: abstract (v2): Nemotron-ClimbMix, a 400-billion-token pre-training dataset.

[A2] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. arXiv preprint. arXiv:2406.17557 (v1 2024-06-25). https://arxiv.org/abs/2406.17557
    verified: abstract (v2): FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens).

[A3] Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, et al. *TinyGSM: achieving >80% on GSM8k with small language models*. arXiv preprint. arXiv:2312.09241 (v1 2023-12-14). https://arxiv.org/abs/2312.09241
    verified: abstract (v1): 12.3M synthetic grade-school problems with Python solutions; 1.3B generator + 1.3B verifier 81.5% on GSM8K.

[A4] David R. So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le. *Primer: Searching for Efficient Transformers for Language Modeling*. arXiv preprint (NeurIPS camera ready per comment). arXiv:2109.08668 (v1 2021-09-17). https://arxiv.org/abs/2109.08668
    verified: abstract (v2): squared ReLU and depthwise convolution after Q, K, V.

[A5] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*. arXiv preprint. arXiv:2505.06708 (v1 2025-05-10). https://arxiv.org/abs/2505.06708
    verified: abstract (v1): head-specific sigmoid gate after SDPA; mitigates attention sinks (15B MoE and 1.7B dense, 3.5T tokens).

[A6] Yaniv Leviathan, Matan Kalman, Yossi Matias. *Selective Attention Improves Transformer*. ICLR 2025. arXiv:2410.02703 (v1 2024-10-03). https://arxiv.org/abs/2410.02703
    verified: abstract (v2): parameter-free selective attention.

[A7] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. arXiv preprint. arXiv:2407.08608 (v1 2024-07-11). https://arxiv.org/abs/2407.08608
    verified: abstract (v2): asynchrony and FP8 on H100; 740 TFLOP/s FP16.

[A8] Lvmin Zhang, Anyi Rao, Maneesh Agrawala. *Adding Conditional Control to Text-to-Image Diffusion Models*. arXiv preprint. arXiv:2302.05543 (v1 2023-02-10). https://arxiv.org/abs/2302.05543
    verified: abstract (v3): 'zero convolutions' (zero-initialised layers) attach new conditioning paths.

[A9] Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, et al. *bert2BERT: Towards Reusable Pretrained Language Models*. arXiv preprint. arXiv:2110.07143 (v1 2021-10-14). https://arxiv.org/abs/2110.07143
    verified: abstract (v1): function-preserving initialisation of a larger BERT/GPT; ~45%/47% compute saved.

[A10] Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid Karlinsky, Rogerio Feris, et al. *Learning to Grow Pretrained Models for Efficient Transformer Training*. ICLR 2023. arXiv:2303.00980 (v1 2023-03-02). https://arxiv.org/abs/2303.00980
    verified: abstract (v1): learned linear growth operator; up to 50% compute saved.

[A11] Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen. *Progressive Growing of GANs for Improved Quality, Stability, and Variation*. ICLR 2018. arXiv:1710.10196 (v1 2017-10-27). https://arxiv.org/abs/1710.10196
    verified: abstract (v3): layers added progressively during GAN training.

[A12] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*. arXiv preprint. arXiv:2404.06395 (v1 2024-04-09). https://arxiv.org/abs/2404.06395
    verified: abstract (v3): Warmup-Stable-Decay schedule.

[A13] Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, Martin Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*. NeurIPS 2024 (spotlight). arXiv:2405.18392 (v1 2024-05-28). https://arxiv.org/abs/2405.18392
    verified: abstract (v3): constant learning rate plus cooldown scales like cosine.

[A14] Xu Owen He. *Mixture of A Million Experts*. arXiv preprint. arXiv:2407.04153 (v1 2024-07-04). https://arxiv.org/abs/2407.04153
    verified: abstract (v1): product-key retrieval over a million tiny experts.

[A15] Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu, Zhiyu Li, Bo Tang, et al. *Memory³: Language Modeling with Explicit Memory*. Journal of Machine Learning 3 (2024). arXiv:2407.01178 (v1 2024-07-01). https://arxiv.org/abs/2407.01178
    verified: abstract (v1): explicit memory as a third memory form; 2.4B model trained from scratch.

[A16] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. *The Curious Case of Neural Text Degeneration*. ICLR 2020. arXiv:1904.09751 (v1 2019-04-22). https://arxiv.org/abs/1904.09751
    verified: abstract (v2): likelihood decoding gives repetitive text; nucleus (top-p) sampling.

[A17] Martin Marek, Sanae Lotfi, Aditya Somasundaram, Andrew Gordon Wilson, Micah Goldblum. *Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful*. NeurIPS 2025. arXiv:2507.07101 (v1 2025-07-09). https://arxiv.org/abs/2507.07101
    verified: abstract (v4): small batches train stably; recommends against gradient accumulation on one device.

[A18] William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR. arXiv:2101.03961 (v1 2021-01-11). https://arxiv.org/abs/2101.03961
    verified: abstract (v3): sparsely-activated MoE with constant computational cost.

[A19] Technology Innovation Institute (tiiuae). *Falcon-H1-Tiny-90M-Instruct model card*. Hugging Face model card, 2026. https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct
    verified: README fetched 2026-11-24: causal decoder-only, hybrid Transformers + Mamba, English, 90M parameters, Falcon-LLM License; training details deferred to the Falcon-H1-Tiny technical blog post (entry above); GGUF Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf in tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF.

[A20] Hugging Face (HuggingFaceTB). *Smol-SmolTalk dataset card*. Hugging Face dataset card, 2024-2025. https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
    verified: README fetched 2026-11-24: Apache-2.0; tagged synthetic; 'a subset of SmolTalk adapted for smol models with less than 1B parameters', used for SmolLM2-360M-Instruct.

[A21] Pleias and the AI Alliance. *SYNTH dataset card*. Hugging Face dataset card, 2025. https://huggingface.co/datasets/PleIAs/SYNTH
    verified: README fetched 2026-11-24: CC-BY-4.0; 79,648,272 synthetic samples (over 41 billion words) amplified from 58,698 Wikipedia articles.

[A22] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, et al. *Enhancing Chat Language Models by Scaling High-quality Instructional Conversations*. arXiv preprint. arXiv:2305.14233 (v1 2023-05-23). https://arxiv.org/abs/2305.14233
    verified: abstract (v1): UltraChat, large-scale multi-turn instructional conversations generated without human queries.

[A23] Arindam Mitra, Hamed Khanpour, Corby Rosset, Ahmed Awadallah. *Orca-Math: Unlocking the potential of SLMs in Grade School Math*. arXiv preprint. arXiv:2402.14830 (v1 2024-02-16). https://arxiv.org/abs/2402.14830
    verified: abstract (v1): a 7B small language model trained on a synthetic dataset of 200K math problems reaches 86.81% on GSM8K without verifiers, code or ensembling.

[A24] Pranav Rajpurkar, Robin Jia, Percy Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD*. ACL 2018. arXiv:1806.03822 (v1 2018-06-11). https://arxiv.org/abs/1806.03822
    verified: abstract (v1): SQuAD 2.0 adds over 50,000 adversarial unanswerable questions.

[A25] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018. arXiv:1809.09600 (v1 2018-09-25). https://arxiv.org/abs/1809.09600
    verified: abstract (v1): 113k Wikipedia-based multi-hop question-answer pairs with supporting facts.

[A26] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, et al. *Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing*. arXiv preprint. arXiv:2406.08464 (v1 2024-06-12). https://arxiv.org/abs/2406.08464
    verified: abstract (v2): alignment data synthesised by prompting aligned LLMs with only the pre-query template.

[A27] Faeze Brahman, Sachin Kumar, Vidhisha Balachandran, Pradeep Dasigi, Valentina Pyatkin, Abhilasha Ravichander, et al. *The Art of Saying No: Contextual Noncompliance in Language Models*. NeurIPS 2024 Datasets and Benchmarks (arXiv comment). arXiv:2407.12043 (v1 2024-07-02). https://arxiv.org/abs/2407.12043
    verified: abstract (v2): taxonomy of contextual noncompliance (incomplete, unsupported, indeterminate, humanizing and unsafe requests) and training data for it.

[A28] Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, et al. *WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models*. arXiv preprint. arXiv:2406.18510 (v1 2024-06-26). https://arxiv.org/abs/2406.18510
    verified: abstract (v1): WildTeaming and the WildJailbreak safety training resource.

[A29] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, et al. *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs*. NeurIPS 2024 (arXiv comment). arXiv:2406.18495 (v1 2024-06-26). https://arxiv.org/abs/2406.18495
    verified: abstract (v3): WildGuard moderation tool and the WildGuardMix data (prompt harm, response harm, refusal).

[A30] Wanlong Liu, Junying Chen, Ke Ji, Li Zhou, Wenyu Chen, Benyou Wang. *RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions*. arXiv preprint. arXiv:2501.00353 (v1 2024-12-31). https://arxiv.org/abs/2501.00353
    verified: abstract (v1): synthesised retrieval-augmented instruction data over five RAG paradigms.

[A31] Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA*. ICLR 2025. arXiv:2410.20672 (v1 2024-10-28). https://arxiv.org/abs/2410.20672
    verified: abstract (v3): Recursive Transformers initialised from pre-trained models, relaxed with depth-wise LoRA.

[A32] Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, et al. *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. arXiv preprint. arXiv:2507.10524 (v1 2025-07-14). https://arxiv.org/abs/2507.10524
    verified: abstract (v3): shared stack with routers assigning per-token recursion depth, 135M-1.7B.

[A33] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, et al. *Scaling Laws for Neural Language Models*. arXiv preprint. arXiv:2001.08361 (v1 2020-01-23). https://arxiv.org/abs/2001.08361
    verified: PDF Sec. 1-2: model size N defined as the number of non-embedding parameters.

[A34] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, et al. *Improving language models by retrieving from trillions of tokens*. arXiv preprint. arXiv:2112.04426 (v1 2021-12-08). https://arxiv.org/abs/2112.04426
    verified: abstract (v3): retrieval from a 2 trillion token database; comparable to GPT-3 on the Pile with 25x fewer parameters.

[A35] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, et al. *In-Context Retrieval-Augmented Language Models*. TACL (arXiv comment). arXiv:2302.00083 (v1 2023-01-31). https://arxiv.org/abs/2302.00083
    verified: abstract (v3): retrieved documents prepended to the input of an unchanged LM.

[A36] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. arXiv preprint. arXiv:2403.04132 (v1 2024-03-07). https://arxiv.org/abs/2403.04132
    verified: abstract (v1): crowdsourced pairwise comparisons, over 240K votes.
