---
type: concept
aliases: [Foundation Model, 基盤モデル, foundation models]
tags: [paradigm, scaling, pretraining]
related: [[self-supervised-learning]], [[weakly-supervised-pretraining]], [[vision-transformer]], [[zero-shot-transfer]], [[contrastive-learning]], [[promptable-segmentation]], [[promptable-concept-segmentation]], [[alignment-tuning]]
sources: [[sources/clip]], [[sources/dinov2-learning-robust-visual-features-without-supervision]], [[sources/segment-anything]], [[sources/sam-3]], [[sources/siglip]], [[sources/siglip-2]], [[sources/perception-encoder]], [[sources/foundational-models-vision-survey]]
updated: 2026-05-31
---

# Foundation Model（基盤モデル）

## 一言で

**「大規模かつ多様なデータで自己教師あり的に事前学習され、ファインチューニングや少数例適応で多様な下流タスクに転用できる単一の大規模モデル」**。Stanford CRFM（Center for Research on Foundation Models）が 2021 年の論文 "On the Opportunities and Risks of Foundation Models"（Bommasani ら）で名付けた概念。BERT、GPT、CLIP、DINOv2、SAM などが典型例。

> **補足: なぜ「foundation（基礎）」と呼ばれるか** — 「あらゆる応用がこのモデルの上に建てられる」という比喩。建築の基礎工事のように、応用ごとに最初から作り直すのではなく、一度作った基礎モデルに乗せていくという発想。日本語では「基盤モデル」と訳されることが多い。

## 構成要件（CRFM の定義から）

1. **大規模な事前学習**: 通常、数億〜数千億パラメータ、数十億〜数兆トークン／画像
2. **広範なデータ**: ドメイン・スタイル・分布が多様
3. **自己教師ありまたは弱教師ありの目的関数**: ラベル付きデータでは到底スケールできない
4. **下流タスクへの汎用性**: ゼロショット、線形プローブ、ファインチューン、プロンプトなどで多様な応用に転用
5. **emergent capabilities**: スケールに伴って事前に予測しなかった能力が現れる（ICL, chain-of-thought, dense correspondence など）

## NLP における基盤モデルの系譜

| 年 | モデル | 規模 | 特徴 |
|---|---|---|---|
| 2018 | BERT | 110M〜340M | masked language modeling、ファインチューン前提 |
| 2018 | GPT-1 | 117M | causal LM、生成 |
| 2019 | GPT-2 | 1.5B | スケール則の最初の示唆 |
| 2020 | GPT-3 | 175B | in-context learning が emergence |
| 2022 | PaLM, Chinchilla | 540B / 70B | scaling law の精緻化 |
| 2023-24 | GPT-4, Claude, Gemini | 非公開 | マルチモーダル、ツール使用 |

## CV における基盤モデルの系譜

CV では NLP より遅れたが、2021〜2024 にかけて本格化。

### A. 弱教師あり系（テキスト誘導）
- **CLIP**（OpenAI, 2021, [[sources/clip]] / [[entities/clip]]）: WIT-400M（[[entities/wit-400m]]）の画像-テキスト対で対比学習。**CV における初の本格的視覚基盤モデル**で、ゼロショット転移（[[concepts/zero-shot-transfer]]）を CV に持ち込んだ起点
- **ALIGN**（Google, 2021）: 18 億の Web 画像-alt text 対で学習
- **OpenCLIP**: LAION-2B/5B で CLIP を再現・拡張、公開
- **SigLIP** ([[entities/siglip]] / [[sources/siglip]]) (Google DeepMind, 2023): CLIP の **softmax → sigmoid 損失** で根本的効率化、4 TPU で 1 日訓練可能、**32k バッチで飽和** という発見。SO-400M で 5B EVA-CLIP を超える 83.2% IN-0
- **SigLIP 2** ([[sources/siglip-2]]) (Google DeepMind, 2025): SigLIP に LocCa decoder + 自己蒸留＋マスク予測 + ACID 蒸留 + 多言語＋de-bias + NaFlex を統合した「全部入りレシピ」。**CLIP 4 年分の独立改善を 1 モデルに凝集**。RefCOCO で +20pt、ADE20k で +4.2pt、representation bias 35.5%→7.3%。g/16 (1B) 新サイズ追加、85.0% IN-0
- **PE** ([[sources/perception-encoder]] / [[entities/perception-encoder]]) (Meta, NeurIPS 2025): 5.4B unique pairs を 86B samples seen まで訓練 + 22M videos。**「対比学習を頑健にスケールすると中間層に多目的な一般特徴量が育つ」発見と、[[concepts/alignment-tuning]] でそれを末端に引き出す戦略**で、PEcore（ゼロショット SOTA）/ PElang（MLLM 専門）/ PEspatial（dense 予測 SOTA）の 3 バリアントを構築。COCO 検出 66.0 box AP で SOTA
- **DeepSeek-OCR** ([[sources/deepseek-ocr]] / [[entities/deepseek-ocr]]) (Wei et al., DeepSeek-AI, 2025 Oct, arXiv:2510.18234): **DeepSeek-AI 初の wiki 登録、OCR / 文書理解特化型 MLLM、「視覚をテキスト圧縮媒体として活用する」新パラダイム**。**DeepEncoder**: **[[entities/sam\|SAM-base]] (80M, window attention) + 2 層 ConvNet (16× ダウンサンプリング) + [[entities/clip\|CLIP-large]] (300M, dense global attention) = 約 380M**。**DeepSeek-3B-MoE-A570M デコーダ**: 64 routed expert + 6 活性化 + 2 shared = 推論時 570M 活性。**6 解像度モード**: Tiny 64 / Small 100 / Base 256 / Large 400 / Gundam 動的 / Gundam-M。**Fox ベンチで 10× 圧縮 97% / 20× 圧縮 60%**。**OmniDocBench Gundam 795 トークンで 0.083 SOTA**（MinerU2.0 6790 トークン 0.133 を 8.5× 効率で凌駕、InternVL3-78B 0.218 / Qwen2.5-VL-72B 0.214 を上回る）。**100 トークンで GOT-OCR2.0 (256 トークン) を凌駕**。**1 台 A100-40G で 200K+ ページ/日**。**Memory Forgetting Mechanism**: 古い対話を画像化・縮小して LLM 長文脈問題を解決する応用提案。**訓練**: OCR 1.0 70% + OCR 2.0（10M チャート + 5M 化学式 SMILES + 1M 平面幾何 Slow Perception）+ 一般視覚 20% + テキスト 10%、20 ノード × 8× A100-40G = 160 GPU、PP=4 + DP=40、70-90B トークン/日。**オープンソース** (GitHub)。同著者 Haoran Wei の系譜: Vary → GOT-OCR2.0 → DeepSeek-OCR。**Qwen 系 / InternVL 系 / Google 系 (Gemma) に続く第 4 の中国系 AI ラボ系譜 (DeepSeek-AI)**。**弱点**: OCR 特化で汎用 MLLM ではない、MMMU 報告なし、SFT 段階なし、Newspapers で 0.645、Memory Forgetting は概念実証のみ、SAM+CLIP 直列の自然画像理解汎用性不明
- **Gemma 3** ([[sources/gemma-3]] / [[entities/gemma-3]]) (Gemma Team Google DeepMind, 2025 Mar, arXiv:2503.19786): **Google DeepMind の wiki 初のオープン MLLM、Qwen 系 / InternVL 系と並ぶ第 3 の主要系譜**。Gemini 2.0 と co-design された **1B/4B/12B/27B の 4 サイズ × PT/IT = 8 公開モデル**（1B は text-only、4B/12B/27B はマルチモーダル）。**[[entities/siglip\|SigLIP]] 400M variant を 4B/12B/27B で共有 + 凍結**、896² 固定 + **4×4 average pooling で 256 トークン圧縮**、**Pan & Scan (P&S)**（LLaVA 風タイル分割、推論時のみ）で **InfoVQA +17.0 / DocVQA +4.8**。**5:1 local:global attention + sliding window 1024**（KV キャッシュ・オーバーヘッド 60% → <15% 削減）。**128K 文脈**（global RoPE 10k→1M）。**QK-norm**（Gemma 2 soft-capping 置換）+ GQA + RMSNorm。**QAT**（per-channel int4 / per-block int4 / SFP8、27B Int4 で 14.1 GB）。**全モデル知識蒸留**（27B 14T トークン、Gemini 2.0 と同じ 262k 語彙）。**主要結果**: **LMSys Chatbot Arena Elo 1338（rank 9）** で DeepSeek-V3 (671B/37B MoE) / Llama-3.1-405B / Qwen2.5-72B を **より少ないパラメータで凌駕**、MMLU-Pro 67.5、**MATH 89.0**（Gemini 1.5 Pro 86.5 超え）、MMMU val 64.9、DocVQA w/ P&S 90.4。**PaliGemma 2 27B を文書理解で +4.4〜+14.4 凌駕** + 4B/12B は 10× 安価転送。**Gemma3-4B-IT が Gemma2-27B-IT 匹敵、27B-IT が Gemini-1.5-Pro 同等**。**[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形** + **KV キャッシュ最適化の新標準**。**Gemma License**（Apache 2.0 風、商用可）。**弱点**: 固定 896² + P&S の保守性、256 トークン圧縮の情報損失、128K で RULER 27B 66.0（Qwen3-VL 1M YaRN 99.5% に大きく劣る）、動画 16 frames 制約、**MoE 不採用**（InternVL 3.5 / Qwen3-VL と対照的）、訓練データ非公開
- **Qwen3-VL** ([[sources/qwen3-vl]] / [[entities/qwen3-vl]]) (Qwen Team Alibaba Group, 2025 Nov, arXiv:2511.21631): **Qwen-VL シリーズ第 4 世代、Qwen ファミリーが商用最先端と完全に肩を並べた決定的瞬間**。**3 主要構造革新**: (1) **Interleaved MRoPE**（temporal/height/width を埋め込み次元に**交互配置**、Qwen2/2.5-VL の塊状分割による周波数スペクトル不均衡を解消、長動画理解強化）、(2) **DeepStack**（ViT 中間 3 層の視覚特徴を LLM 最初 3 層に注入する多層融合、追加文脈長なし）、(3) **テキスト・ベース時間整合**（MRoPE absolute time を捨て、`<3.0 seconds>` のような明示的タイムスタンプ・トークンに置換、秒形式 + HMS 形式、Charades-STA mIoU 64.8 で Qwen2.5-VL から +13.9）。**6 サイズ × 2 バリアント = 12 公開モデル**: dense（**2B/4B/8B/32B**）+ MoE（**30B-A3B / 235B-A22B フラッグシップ**）× **Instruct（non-thinking）+ Thinking**（長 CoT 推論）。**Vision Encoder**: **SigLIP-2 から継続学習**（Qwen2.5-VL のゼロから路線放棄）、SigLIP2-SO-400M / SigLIP2-Large 300M。**256K ネイティブ文脈**（YaRN 拡張で **1M トークン = 2 時間動画**まで外挿、Needle-in-a-Haystack 256K 100% / 1M 99.5%）。**4 段階事前学習** ~2.2T トークン（S0 Merger 67B + S1 All 1T 8K + S2 All 1T 32K + **S3 Ultra-Long-Context 100B 262K**）+ 事後学習 SFT + Strong-to-Weak Distillation + SAPO RL + **Thinking with Images**（2 段階エージェント学習）。**多言語 OCR 10 → 39 言語**、9-DoF 3D bbox（Omni3D）、Multimodal Code（UI→HTML/SVG）、Agent（GUI + Function Calling + Search）。**主要結果**（235B-A22B）: MathVista **85.8 SOTA** / MathVision **74.6 SOTA** / MathVerse_mini **85.0 SOTA**、MMMU 80.6（GPT-5 84.2 -3.6）、**OCRBench 920 / OCRBench_v2 zh 63.5 SOTA**（GPT-5 +25.8 圧倒）、OmniDocBench en **0.143 SOTA**、MMLongBench-Doc **57.0 SOTA**、**MUIRBENCH 80.1 SOTA**、CountBench 93.7、ODinW-13 **48.6**（Qwen2.5-VL から +5.5）、VSI-Bench 62.7 / EmbSpatialBench 84.3 / RefSpatial 69.9、**Charades-STA mIoU 64.8 SOTA**（Qwen2.5-VL から +13.9）、V\* **93.7+ with tools**、**ScreenSpot Pro 62.0**（Qwen2.5-VL +18.4）、**OSWorld 38.1**（Qwen2.5-VL 8.83 から **4.3× 飛躍**）、**AndroidWorld 63.7**（Qwen2.5-VL +28.7）。**マルチモーダル化で言語が強くなる現象**を Qwen3-235B 純粋 LLM 比で AIME-25 +4.4 / HMMT-25 +2.0 で継続実証。**Apache 2.0**。**Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking モード**は MLLM 業界の新標準。**弱点**: MMMU で GPT-5 -3.6、We-Math で Gemini -5.8、OSWorld で Claude Opus 4 (44.4) に -6.3、訓練データ非公開、6×2=12 モデル管理コスト
- **Qwen2.5-VL** ([[sources/qwen2-5-vl]] / [[entities/qwen2-5-vl]]) (Bai et al., Qwen Team Alibaba Group, 2025 Feb, arXiv:2502.13923): **Qwen-VL シリーズ第 3 世代、視覚言語モデルから「視覚エージェント」への転換点**。**4 つの主要技術**: (1) **Window Attention 採用 ViT をゼロから学習**（32 層中 4 層のみ完全自己注意で計算量線形化、RMSNorm + SwiGLU で LLM 系統一）、(2) **MRoPE Aligned to Absolute Time**（temporal ID を絶対秒数に揃えて FPS 非依存の時間理解、Charades-STA mIoU 50.9 で GPT-4o 35.7 を +15.2）、(3) **動的 FPS サンプリング + 動画 3D パッチ分割**（連続 2 フレーム grouping、秒形式 + hmsf 形式タイムスタンプ）、(4) **事前学習 1.2T → 4.1T トークン**（QwenVL HTML フォーマットで文書統一表現 + ABC notation 楽譜 + SMILES 化学式、10,000+ カテゴリのオープン語彙検出、PixMo 点グラウンディング、多言語 OCR 10 言語）。**3B / 7B / 72B の 3 サイズ**（全サイズで同一 ViT 共有、LLM は Qwen2.5）。3 段階事前学習（Visual 1.5T + Multimodal 2T + Long-Context 0.6T、seq 8K → 32K）+ 2 段階事後学習（SFT 200 万 + DPO、ViT 凍結）。**主要結果**（72B）: MMMU **70.2 初突破**（GPT-4o 69.1 / InternVL2.5-78B 70.1 と並ぶ）/ MathVista **74.8**（GPT-4o +11.0）/ MATH-Vision **38.1** / MMBench-EN **88.6 SOTA** / MMStar **70.8 SOTA** / MMVet **76.2 SOTA** / DocVQA **96.4** / OCRBench **885** / **OCRBench_v2 en/zh 61.5/63.7**（Gemini 1.5-Pro 比 en +9.6 / zh +20.6 圧倒）/ CountBench **93.6 SOTA** / LVBench **47.3**（GPT-4o +16.5）/ EgoSchema **76.2** / MLVU **74.6**（GPT-4o +10.0）/ Charades-STA mIoU **50.9 SOTA**（時間グラウンディング）。**GUI エージェント**: **ScreenSpot Pro 43.6**（Qwen2-VL 1.6% から +42 ポイント = 27× 飛躍、Aguvis-72B 23.6 圧倒）/ Android Control High **67.36** / Low **93.7** / AndroidWorld **35%**（GPT-4o 超え）/ MobileMiniWob++ **68%**。**マルチモーダル化で言語が強くなる現象**を Qwen2.5-72B 純粋 LLM とほぼ同等性能で実証（[[entities/internvl-3|InternVL 3]] の発見の Qwen 系版での先行観察）。**Window Attention + MRoPE absolute time + QwenVL HTML format + GUI Agent パラダイム**は MLLM 業界の新標準。**弱点**: MMMU-Pro 51.1（GPT-4o -0.8）、Video-MME で Gemini 1.5-Pro に -1.7/-2.2、OSWorld デスクトップで Claude に -6.07、ViT ゼロから学習は計算コスト大、訓練データ非公開
- **Qwen2-VL** ([[sources/qwen2-vl]] / [[entities/qwen2-vl]]) (Wang et al., Qwen Team Alibaba Group, 2024 Sept, arXiv:2409.12191): **Qwen-VL シリーズ第 2 世代、「任意解像度・任意アスペクト比」を初めて本格的に実現したオープンソース MLLM**。**Naive Dynamic Resolution**: ViT の絶対位置埋め込みを 2D-RoPE に置換 + MLP で隣接 2×2 トークンを 1 トークンに圧縮、224² 画像が 66 トークン化。**M-RoPE（Multimodal Rotary Position Embedding）**: 回転位置埋め込みを **temporal / height / width の 3 成分**に分解、テキストは 1D-RoPE 等価 / 画像は temporal 固定 / 動画は temporal 増分。**位置 ID 値が小さく抑えられて長文脈外挿に有利、学習 16K → 推論 80K まで頑健**。**統一画像/動画処理**: 2 fps + 深さ 2 の 3D 畳み込み（3D チューブ）、動画あたり 16K トークン上限で **20 分以上の動画理解**。**2B / 7B / 72B の 3 サイズ**で全サイズが **675M ViT を共有**（DFN 初期化）、LLM は Qwen2。3 段階学習 + 累積 **1.4T トークン**、テキスト・トークンのみ監督。**主要結果**（72B）: DocVQA **96.5** / OCRBench **877** / RefCOCO val **93.2 SOTA**（汎用、専門 UNINEXT-H 92.6 を上回る、Qwen-VL 初代 89.4 から +3.8）/ MMBench-CN **86.6 SOTA** / MathVista **70.5**（GPT-4o +6.7）/ EgoSchema **77.9**（GPT-4o +5.7）/ 多言語 OCR 8 言語中 7 で GPT-4o 超え / UI 操作 AITZ EM 72.1（GPT-4o 35.3 を圧倒）。**Naive Dynamic Resolution + M-RoPE は MLLM 業界の事実上の標準**として広く採用、後の Qwen2.5-VL / Qwen3-VL の基盤。**弱点**: MMMU 64.5（GPT-4o -4.6）、VLN（R2R）51.7（専門モデル -27.3、3D 空間モデリングが弱点）
- **Qwen-VL** ([[sources/qwen-vl]] / [[entities/qwen-vl]]) (Bai et al., Alibaba Group, 2023 Aug, arXiv:2308.12966): **Alibaba 製 MLLM シリーズの初代、Qwen ファミリーの視覚言語基盤**。**Qwen-7B（中英バイリンガル LLM）+ OpenCLIP ViT-bigG（1.9B）+ Position-aware VL Adapter（256 学習可能クエリ + 2D 絶対位置エンコーディング、0.08B）= 9.6B**。**3 段階学習**（Stage1: 224² で LLM 凍結 + 1.4B 画像-テキスト対 → Stage2: 448² で全凍結解除 + 7 タスク同時 → Stage3: ViT 凍結 SFT で Qwen-VL-Chat）。**`<box></box>` で囲んだ [0, 1000) 正規化座標文字列**による細粒度グラウンディング、**`<ref></ref>` で参照対象を標識**。Flickr30K (0-shot) **85.8 CIDEr**（Flamingo-80B 67.2 を 9.6B で上回る）、VQAv2 79.5、TextVQA 63.8、RefCOCO val **89.36**、**TouchStone-Cn 401.2 で中国語マルチモーダル対話の事実上の標準**。**`<box>`/`<ref>` 特殊トークン設計が MLLM 業界の事実上の標準に**。InternVL と並ぶ **2023-2025 年オープンソース MLLM 2 大系譜の片翼**。後継: Qwen2-VL（2024.09, 動的解像度 + M-RoPE）/ Qwen2.5-VL（2025）。**弱点**: 解像度 448² 固定、256 トークン固定圧縮、Stage1 で LLM 凍結（後の InternVL 3 の Native Multimodal Pre-Training とは対照的な保守的設計）、単一モデル規模のみ
- **InternVL** ([[sources/internvl]] / [[entities/internvl]]) (OpenGVLab Shanghai AI Lab, CVPR 2024): **視覚エンコーダを 6B（InternViT-6B）にスケールアップ + 8B 言語ミドルウェア QLLaMA で LLM と整列** という設計を初めて本格化。4.98B image-text pairs での 3 段階訓練（contrastive → generative → SFT）で、対比 + 生成 + 対話を 1 モデルに統合。**ViT-22B（21.7B）を 1/3.7 のパラメータで ADE20K linear probe +12.6 mIoU 上回る**。InternVL 1.5 / 2.0 / 2.5 / 3 シリーズの起点
- **InternVL 1.5** ([[sources/internvl-1-5]] / [[entities/internvl-1-5]]) (OpenGVLab Shanghai AI Lab, 2024 April): InternVL 1.0 から **QLLaMA（8B 言語ミドルウェア）を完全廃止し LLaVA 系の MLP プロジェクタに統一**、InternViT-6B-448px-V1.5（45 層・dynamic 448）+ InternLM2-20B-Chat の 26B 構成。**動的高解像度（1-40 タイル / 最大 4K）+ バイリンガル（英中）+ 継続事前学習**の 3 改善で、**18 マルチモーダルベンチマーク中 8 つで SoTA、ChartQA / OCRBench / MMBench-CN / CCBench / HallusionBench / MathVista で全商用モデル（GPT-4V/Gemini Pro/Claude-3/Qwen-VL-Max）を上回る**。「How Far Are We to GPT-4V?」の問いに「OCR と中国語ではもう追いついた」と答えた論文
- **Mini-InternVL** ([[sources/mini-internvl]] / [[entities/mini-internvl]]) (OpenGVLab Shanghai AI Lab, 2024 Oct): InternVL シリーズの **「軽量 + ドメイン特化」分枝**。**[[entities/internvit-300m\|InternViT-300M]]**（CLIP-ViT-L で初期化 + InternViT-6B から negative cosine similarity 損失で蒸留）+ 軽量 LLM（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）で **1B/2B/4B クラス** を構築。**5% のパラメータで InternVL2-76B の 90% の性能**（Mini-InternVL-4B Avg 72.8 vs 76B Avg 81.4）。さらに **自律走行 / 医療画像 / リモートセンシング** を VQA 形式に統一する「ドメイン適応フレームワーク」を提案し、Mini-InternVL-DA-4B が **GMAI-MMBench で LLaVA-Med / Claude3-Opus 超え、DriveLM Challenge で 26B SOTA に 2B サイズで匹敵、MME-RealWorld 自律走行で GPT-4o を +24.78 圧倒**
- **InternVL 2.5** ([[sources/internvl-2-5]] / [[entities/internvl-2-5]]) (OpenGVLab Shanghai AI Lab, 2024 Dec): **MMMU 70.1% で 70% を超えた初のオープンソース MLLM**（GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 を凌駕）。**1B/2B/4B/8B/26B/38B/78B の 7 サイズスイート**で InternVL シリーズの軽量・大規模を統合。視覚は **InternViT-300M-V2.5 / InternViT-6B-V2.5** の 2 系統、LLM は **InternLM 2.5 / Qwen 2.5**。アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] と同じ（ViT-MLP-LLM）、**訓練戦略・データ品質・テスト時スケーリングのみを精緻化**。**Progressive Scaling Strategy 公式化**（小型 LLM で ViT 訓練 → 大型 LLM に転送、Qwen2-VL の 1/12 の訓練トークン）。**Test-Time Scaling**（CoT + Majority Voting）で MMMU +3.7 ポイント追加改善。MathVista 72.3（GPT-4o 63.8 +8.5）、RefCOCO 92.3 SOTA、Video-MME / MVBench / MLVU で複数 SOTA。InternViT の **「中間層特徴が線形分離性を失いつつ open-set 意味を捕捉」** という [[concepts/alignment-tuning\|PE と同等の発見]] を独立に実証
- **InternVL 3** ([[sources/internvl-3]] / [[entities/internvl-3]]) (OpenGVLab Shanghai AI Lab, 2025 Apr): InternVL シリーズの **「事後的 MLLM 適応 → Native Multimodal Pre-Training」の哲学転換**。LLM Chat 版から MLLM を改造する従来パラダイム（InternVL 1.0-2.5 含む）を捨て、**Qwen2.5 base からテキスト + マルチモーダルを共同事前学習**（言語 50B + マルチモーダル 150B トークン、1:3 比率）。**V2PE（Variable Visual Position Encoding）** で視覚トークンに小さな位置インクリメント δ（{1, 1/2, ..., 1/256} からランダム選択）、長文脈対応。**MPO（Mixed Preference Optimization）** で DPO + BCO + LM 損失の混合選好最適化、**+4.5 ポイント推論改善**（38B モデル）。**VisualPRM-8B** で test-time best-of-N（小型モデル +9.9 ポイント）。**MMMU 72.2 で再び SOTA 更新**、OCRBench **906 で史上初の 900 超え**、**MathVista 79.0**（GPT-4o 60.0 +19.0）。**Qwen2.5-Chat と同じ base から派生して言語能力が +1.6〜+8.9 強化**（「マルチモーダル化で言語が強くなる」の初実証）。**訓練データ + モデル重み完全公開**（open-science 強化）、7 サイズ + InternLM3-8B バリアント
- **MPO** ([[sources/mpo]] / [[entities/mpo]]) (Wang et al., OpenGVLab Shanghai AI Lab, 2024 Nov): **「MLLM の CoT 推論で性能が悪化する」現象を初めて本格的に解決した論文**。**Mixed Preference Optimization = DPO（相対選好）+ BCO（絶対品質）+ SFT loss（生成過程）** の 3 損失混合。**[[entities/mmpr\|MMPR データセット（3M 選好ペア）]]** + **DropoutNTP**（画像なしで応答を補完して rejected 生成、RLAIF-V の 57.5% コスト）を併用。**InternVL2-8B-MPO で MathVista +8.7（67.0、10× 大きい InternVL2-76B 67.2 と同等）、M3CoT +19.9（59.3 → 79.2）**。**マルチモーダル選好最適化が言語推論能力も強化**（TheoremQA +5.2、IFEval +4.1）という [[entities/internvl-3\|InternVL 3]] の「Native Multimodal Pre-Training で言語が強くなる」発見の **先例**。InternVL 3 の Stage 3 で正式採用、InternVL シリーズの永続的技術となった
- **InternVL 3.5** ([[sources/internvl-3-5]] / [[entities/internvl-3-5]]) (OpenGVLab Shanghai AI Lab, 2025 Aug): InternVL シリーズの **第 8 世代、商用フロンティアとの差をオープンソース最小に**。**Cascade Reinforcement Learning（offline RL = MPO で warm-up → online RL = GSPO で精緻化）** という coarse-to-fine 戦略で、**GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る**。**MoE スケーリング**: 20B-A4B（GPT-OSS）/ 30B-A3B（Qwen3）/ **241B-A28B**（最大、Qwen3-235B-A22B）。**ViR（Visual Resolution Router）**: patch ごとに 1/4 or 1/16 圧縮率を動的選択、視覚トークン 50% 削減で性能 99% 維持。**DvD（Decoupled Vision-Language Deployment）**: ViT と LLM を別 GPU に分離、非同期 3 段階パイプラインで **896 解像度で 4.05× 加速**。**MMMU 77.7（オープンソース新 SOTA）、MathVista 82.7（GPT-5 +0.8）、VSI-Bench 69.5（GPT-5 37.5 を +32 圧倒）、GPT-5 との Aggregate 差を 3.9% に**。**Qwen3-base 比で言語 16 ベンチ中 15 で上回る**（[[entities/internvl-3\|InternVL 3]] の発見継続）。**9 サイズ × dense + MoE × Flash 効率版 = 18+ モデル**
- **Florence**（Microsoft）, **BASIC**（Google）, **CoCa**（Google）, **EVA-CLIP**, **MetaCLIP**, **DFN**, **AIMv2**（Apple, キャプション化型）, **InternVideo2**（動画ネイティブ）など多数の関連モデル

### B. 自己教師あり系
- **MAE**（Meta, 2021/2022, [[entities/mae]]）: マスク再構成で ViT-H まで scale
- **DINO**（Meta, 2021, [[entities/dino]]）: 自己蒸留
- **iBOT**（2021/2022, [[entities/ibot]]）: DINO + MIM
- **DINOv2**（Meta, 2023, [[entities/dinov2]]）: 1B パラメータ、142M キュレーション画像で基盤モデル化
- **DINOv3**（Meta, 2025, [[entities/dinov3]]）: ViT-7B × 1.689B 画像、Gram anchoring で dense feature 劣化を解決

### C. 生成系
- **Stable Diffusion**（2022）: テキストから画像生成の基盤
- **Imagen, DALL-E 3, SDXL** など

### D. 領域汎用モデル（promptable / 教師あり）
- **SAM**（Segment Anything Model, Meta, 2023, [[sources/segment-anything]] / [[entities/sam]]）: **CV における初の本格的セグメンテーション基盤モデル**。promptable segmentation（[[concepts/promptable-segmentation]]）タスクで訓練、データエンジンで構築した SA-1B（[[entities/sa-1b]]、11M 画像 × 1.1B マスク）を使用。**SSL ではなく教師あり**でスケール（モデル支援アノテーションで実現）。
- **SAM 2**（Meta, 2024, [[sources/sam-2]] / [[entities/sam-2]]）: SAM の動画拡張版。**Hiera 画像エンコーダ**（[[entities/hiera]]）+ **streaming memory** で動画と画像を統一処理。SA-V（[[entities/sa-v]]、50.9K 動画 × 642.6K masklet、CC by 4.0）で訓練。画像でも SAM v1 比 6× 高速 + 高精度の上位互換。**動画 foundation model の de facto 標準**。
- **SAM 3**（Meta Superintelligence Labs, 2025, [[sources/sam-3]] / [[entities/sam-3]]）: **新タスク PCS（Promptable Concept Segmentation, [[concepts/promptable-concept-segmentation]]）を導入**。名詞句または画像 exemplar から **コンセプトの全インスタンス** を検出・セグメント・追跡。Perception Encoder backbone + DETR detector（[[sources/detr]] / [[entities/detr]] 系統）+ **presence head**（認識と位置特定を分離）+ SAM 2 tracker。SA-Co（[[entities/sa-co]]、4M unique NP、benchmark 207K concepts）で訓練・評価。**foundation model の第 4 軸（コンセプト指定）** を確立。
- **DALL·E, Stable Diffusion 系**: テキスト → 画像生成の基盤として CLIP/SigLIP を構成要素に持つ
- **Florence-2**（Microsoft, 2024）: 検出/セグメンテーション/キャプション統合 vision foundation

## サーベイによる体系的分類（Awais et al., TPAMI 2024）

**[[sources/foundational-models-vision-survey|Awais et al., TPAMI 2024]]**（arXiv:2307.13721, MBZUAI ら）が、CV における基盤モデルを **4 つの直交軸** で整理した分類体系（taxonomy）を提供している。本 wiki のすべての視覚基盤モデル entity はこの体系のいずれかに位置付けられる。

### 軸 1: テキスト・プロンプト型（Textually Prompted）

「言語を主要監督源にするモデル」。**訓練目的**でさらに 4 サブカテゴリに分割：

1. **対比型（Contrastive Learning, CL）**: [[entities/clip|CLIP]], ALIGN, [[entities/siglip|SigLIP]], OpenCLIP, EVA-CLIP, [[entities/perception-encoder|Perception Encoder]] — **画像-テキスト対比損失（ITC）** が主目的、Dual-Encoder
2. **生成型（Generative）**: Frozen, Flamingo, KOSMOS-1/2, SimVLM, PaLI — **言語モデリング（LM）/ PrefixLM / Captioning** 等、Encoder-Decoder / Adapter LLM
3. **ハイブリッド（CL + 生成）**: FLAVA, BLIP, BLIP-2, InstructBLIP, CoCa, UNITER, X-FM — **両目的を併用**、Fusion / Adapter LLM
4. **対話型 VLM（Conversational）**: GPT-4V, MiniGPT-4, [[entities/qwen-vl|Qwen-VL]] 系列, [[entities/internvl|InternVL]] 系列, LLaVA, [[entities/gemma-3|Gemma 3]], [[entities/deepseek-ocr|DeepSeek-OCR]] — **人間らしい指示追従と対話**、ほぼ全て Adapter LLM

**さらに「汎用 vs 視覚グラウンディング」で 2 分割**:
- **汎用**: 分類・検索が得意（CLIP 系）
- **視覚グラウンディング**: [[entities/glip|GLIP]], [[entities/grounding-dino|Grounding-DINO]], [[entities/yolo-world|YOLO-World]], OWL-ViT, OpenSeg, RegionCLIP — **領域-単語対応**を学習、open-vocabulary 検出/参照セグメンテーション

### 軸 2: 視覚プロンプト型（Visually Prompted）

「点・ボックス・マスク・テキストなど多様なプロンプトを受け付け、セグメンテーション等を返す」モデル。

- **[[entities/sam|SAM]]**（Meta, 2023）: PVS（[[concepts/promptable-segmentation|Promptable Visual Segmentation]]）タスクを確立、SA-1B で訓練、CLIP に依存せず
- **[[entities/sam-2|SAM 2]]**（Meta, 2024）: SAM の動画拡張、Hiera 採用
- **[[entities/sam-3|SAM 3]]**（Meta, 2025）: PCS（[[concepts/promptable-concept-segmentation|Promptable Concept Segmentation]]）タスクを追加、PE backbone
- CLIPSeg / SegGPT / SEEM: CLIP 流用や in-context 学習
- **汎用主義（Generalist）モデル**: Painter, VisionLLM, Prismer — 複数タスクを単一モデルで実行

### 軸 3: 異種モダリティ型（Heterogeneous Modalities）

「画像-テキスト以外のモダリティ対（動画-音声・画像-深度・IMU など）も整列する」モデル。

- **AudioCLIP**: CLIP に音声追加、3 モダリティ
- **ImageBind**: 6 モダリティを「画像を結合点」として整列
- **MACAW-LLM**: 画像・動画・音声・テキストの LLM 統合
- **[[entities/qwen3-5-omni|Qwen3.5-Omni]]**: Thinker-Talker 構造で**音声出力まで含めた end-to-end omni model**、軸 3 の現代的到達点

### 軸 4: 身体性型（Embodied Foundational Agents）

「視覚・言語を実世界の物理センサと結びつけ、ロボティクス・ナビゲーションに使う」モデル。

- **PaLM-E**（Google）: ViT + PaLM、ロボット操作
- **ViMA**: マルチモーダル・プロンプトでロボット制御
- **MineDojo / VOYAGER**: Minecraft 駆動の終身学習
- **LM-Nav**: CLIP + GPT-3 + ViNG でゼロショット・ナビゲーション
- **仮想環境エージェントとしての MLLM**: [[entities/qwen2-5-vl|Qwen2.5-VL]] の **ScreenSpot Pro 43.6**、[[entities/qwen3-vl|Qwen3-VL]] の **OSWorld 38.1** など、軸 1（対話型 VLM）が軸 4（GUI / 仮想環境エージェント）へ侵食している現代の流れ

### アーキテクチャ・スタイル（軸 1-4 とは直交する分類）

実装の観点では基盤モデルは **4 種のアーキテクチャ・スタイル**を取る（軸 1-4 と直交）：

| スタイル | 説明 | 代表例 |
|---|---|---|
| **Dual-Encoder** | 視覚エンコーダとテキスト・エンコーダが並列、出力を整列 | CLIP, ALIGN, SigLIP |
| **Fusion** | 視覚・テキスト表現を融合デコーダで結合 | GLIP, BLIP, CRIS |
| **Encoder-Decoder** | 共同特徴符号化と復号を逐次 | SimVLM, MetaLM, PaLI |
| **Adapter LLM** | 視覚エンコーダを LLM のアダプタとして統合 | Flamingo, KOSMOS, BLIP-2, mPLUG-OWL, LLaVA, MiniGPT-4, Qwen-VL 系列, InternVL 系列, Gemma 3 |

> **歴史的観察**: 2023-2025 で Adapter LLM パターンが事実上の標準となり、現代の対話型 MLLM はほぼすべて Adapter LLM。BLIP-2 の Q-Former、Qwen-VL の Position-aware VL Adapter、InternVL の MLP projector、Gemma 3 の SigLIP + P&S + average pooling はすべて **Adapter LLM カテゴリ内の異なる実装**として理解できる。

### サーベイのスコープ外（補完すべき軸）

- **純粋画像 SSL**: [[entities/dinov2|DINOv2]] / [[entities/dinov3|DINOv3]] / [[entities/mae|MAE]] / [[entities/ibot|iBOT]] / [[entities/byol|BYOL]] は Awais ら survey の対象外（VL に限定）。本 wiki ではこれらを別軸として並列管理
- **画像生成系**: GAN / VAE / 拡散モデル（[[entities/sdxl|SDXL]] 等）も対象外。専用 survey が並列で存在

## 基盤モデルがもたらす変化

### 開発パラダイムの変化

```
[従来]                          [基盤モデル時代]
タスクA用に                      汎用基盤モデルを 1 つ作る
モデルAを訓練                          ↓
タスクB用に              ──→  タスクA, B, C ... を
モデルBを訓練                    線形プローブ／プロンプト／
タスクC用に                      LoRA／ファインチューンで適応
モデルCを訓練
```

### 集中化と複製困難性

- 訓練コストが**数千 GPU × 数週間**規模なので、大学・中小企業が事前学習を行えない
- 「事前学習は数社が行い、世界中が下流応用する」という構造に
- DINOv2 論文 §9 では、ViT-g 訓練に 22k GPU 時間、プロジェクト全体で 200k GPU-days を要したと開示

### 評価方法のシフト

- 線形プローブ・k-NN 評価（[[concepts/knn-evaluation-protocol]]）が標準化
- ゼロショット評価（CLIP 系）
- 大規模ベンチマーク群（COCO, ADE20K, NYUd, ... を横断的に評価）

## 批判と未解決問題

1. **公平性とバイアス**: 訓練データの偏りが下流すべてに伝播する。DINOv2 自身が §8 で地理的バイアスを開示。
2. **環境コスト**: 訓練時の CO₂ 排出が無視できない（DINOv2 ViT-g 単体で 3.7t CO₂eq）
3. **再現性**: モデルや重みは公開されても、訓練データやレシピは非公開のことが多い（LVD-142M 自体は非公開）
4. **homogenization のリスク**: 全員が同じ基盤モデルに依存することで、その欠陥や偏向が全社会に波及する
5. **emergence の予測困難性**: スケールに伴い何が emerge するか／しないかが事前に分からない

## CV の foundation model に求められる特性（DINOv2 論文より）

DINOv2 著者らが暗黙に置いている要件：

- **ファインチューンなしで使える**（線形プローブで十分強い）
- **画像レベルとピクセルレベルの両方で機能**（分類と segmentation/depth の両方）
- **ドメイン汎化**（自然画像 → 絵画 → 衛星画像 などへの転移）
- **インスタンスレベルとカテゴリレベルの両方**（検索と分類の両方）
- **テキスト不要**（純粋画像で十分）

DINOv2 はこれらを多くの軸で達成し、「CV の基盤モデルは SSL でも作れる」ことを示した。

## VFM 時代の SeSL（半教師あり学習）への波及

[[sources/revisiting-ssl-foundation-models]]（NeurIPS 2025）が示した重要な発見：**「VFM をバックボーンとして使う場合、MixMatch/FixMatch/FlexMatch などのスクラッチ前提 SeSL 手法は驚くほど効果が薄い」**。

理由は単純で、VFM は既に膨大な事前学習で強力な汎化能力を獲得しているため、SeSL がラベルなしデータから抽出する追加情報の限界利得が小さい。代わりに以下が支配的な戦略となる：

1. **[[concepts/parameter-efficient-fine-tuning]]（PEFT）でラベルあきデータのみ fine-tune** —— LoRA/AdaptFormer で十分に高精度
2. **アンサンブル疑似ラベリング** —— 複数の (VFM, PEFT) ペアの予測を統合する [[entities/v-pet]] 等の手法

これは「**基盤モデルが SSL/SeSL という研究領域そのものを再定義した**」象徴的な事例。VFM × PEFT × アンサンブルという 3 軸が次世代の SeSL 設計空間を形成しつつある。

## 関連ページ

- [[sources/foundational-models-vision-survey]]: **Awais et al., TPAMI 2024** — 視覚における基盤モデルの 4 軸分類体系（テキスト/視覚/異種/身体性）を提供する重要なサーベイ。本 wiki の VL 基盤モデルすべての位置付けを与える「鳥瞰図」
- [[sources/clip]]: CV における初の本格的基盤モデル CLIP の原典
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] / [[sources/dinov3]]: 純粋 SSL 基盤モデルの代表例
- [[sources/segment-anything]]: セグメンテーション基盤モデル SAM の原典（教師ありデータをスケールできれば SSL でなくてよいという主張）
- [[sources/sam-2]]: 動画への拡張、Hiera 採用で画像でも SAM v1 を凌駕
- [[sources/sam-3]]: PCS タスク導入、コンセプト指定型 foundation model の確立、PE 採用
- [[entities/clip]] / [[entities/dinov2]] / [[entities/dinov3]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/sam]] / [[entities/sam-2]] / [[entities/sam-3]] / [[entities/hiera]]
- [[entities/sa-1b]] / [[entities/sa-v]] / [[entities/sa-co]]: SAM/SAM 2/SAM 3 の訓練データ（モデル支援アノテーションでスケール、SAM 3 では AI verifier も活用）
- [[concepts/self-supervised-learning]] / [[concepts/weakly-supervised-pretraining]] / [[concepts/promptable-segmentation]]: 基盤モデルを生む 3 大事前学習パラダイム
- [[concepts/zero-shot-transfer]] / [[concepts/contrastive-learning]]: 基盤モデルの中核能力と学習手法
- [[concepts/parameter-efficient-fine-tuning]]: VFM を下流タスクに適応させる主要手法（LoRA, AdaptFormer 等）
- [[concepts/semi-supervised-learning]]: SeSL は VFM 時代に再検討された（[[sources/revisiting-ssl-foundation-models]]）
- [[sources/revisiting-ssl-foundation-models]]: NeurIPS 2025 — VFM 時代における SeSL の再検討、V-PET 提案
- [[entities/v-pet]]: VFM × PEFT アンサンブル疑似ラベリングによる SeSL 手法
