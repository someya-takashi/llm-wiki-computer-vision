---
type: concept
aliases: [WSL, Weakly-Supervised Pretraining, Text-Guided Pretraining, 弱教師あり事前学習]
tags: [paradigm, pretraining, vision-language]
related: [[self-supervised-learning]], [[foundation-model]], [[contrastive-learning]], [[zero-shot-transfer]], [[alignment-tuning]]
sources: [[sources/clip]], [[sources/dinov2-learning-robust-visual-features-without-supervision]], [[sources/siglip]], [[sources/siglip-2]], [[sources/perception-encoder]]
updated: 2026-05-28
---

# Weakly-Supervised Pretraining（弱教師あり事前学習）

## 一言で

**人手の明示的なラベル付け（"これは犬"）ではなく、Web 上に自然に存在する「ノイジーで間接的な教師信号」を使って大規模事前学習を行う方法**。CV では主に **画像 - テキスト（キャプション、alt 属性、ハッシュタグ）対** を使うアプローチを指し、CLIP がその代表。「弱教師あり（weakly-supervised）」は「教師信号はあるが完全ではない・ノイズが多い・間接的」という意味。

> **補足: 「教師あり」「弱教師あり」「自己教師あり」の区別**
> - **教師あり（supervised）**: 人手で正確にラベル付けされたデータ（ImageNet など）。スケールしにくい。
> - **自己教師あり（self-supervised, SSL）**: 教師信号を**データ自身**から作る（DINO, MAE 等）。詳細: [[concepts/self-supervised-learning]]
> - **弱教師あり（weakly-supervised, WSL）**: 教師信号は**外部にあるが不完全**（Web のキャプション、ハッシュタグ、画像のクリック数など）。WSL は SSL と教師ありの中間。

## CV における WSL の主要パターン

### A. 画像 - テキスト対比学習（CLIP 系）

- **CLIP**（OpenAI, 2021, [[sources/clip]] / [[entities/clip]]）: Web から集めた **4 億組**の画像-キャプション対（[[entities/wit-400m]]）。**「対応する画像とテキストの埋め込みが近く、非対応は遠くなるよう対比学習（[[concepts/contrastive-learning]]）」**する。「a photo of a {class}」プロンプトでゼロショット分類を実現
- **ALIGN**（Google, 2021）: 18 億組のノイジーな画像-alt text 対。CLIP より大規模
- **OpenCLIP**: LAION-2B / LAION-5B を使った CLIP の公開再現＋拡張版
- **SigLIP** ([[entities/siglip]] / [[sources/siglip]]) (Zhai et al., Google DeepMind 2023): CLIP の **softmax 対比損失を sigmoid 損失に置き換え**。小バッチで圧倒的に勝つ、メモリ効率改善、ノイズ頑健、**32k バッチで飽和** という発見。4 TPU で 1 日訓練可能。SO-400M で 83.2% IN-0、5B EVA-CLIP より高精度
- **SigLIP 2** ([[sources/siglip-2]]) (Tschannen et al., Google DeepMind 2025): SigLIP に **LocCa decoder + 自己蒸留＋マスク予測 + ACID 蒸留 + 多言語＋de-bias + NaFlex** を統合した「全部入りレシピ」。**WSL の集大成**。RefCOCO で +20pt、dense seg で +4-5pt、representation bias を 35.5%→7.3% に削減。g/16 (1B) 新サイズで 85.0% IN-0
- **PE** ([[sources/perception-encoder]] / [[entities/perception-encoder]]) (Bolya et al., Meta, NeurIPS 2025): **5.4B unique image-text pairs を 86B samples seen まで訓練** + 22M videos ファインチューン。**「対比学習を頑健化＋大規模化すると中間層に多目的特徴量が育つ」発見**、それを [[concepts/alignment-tuning]] で末端に引き出す PEcore / PElang / PEspatial の 3 バリアント設計。SigLIP 2 が「全部入りレシピ」で多目的化したのと対照的に、**対比学習をピュアに保ったまま** 中間層活用で同じことを実現
- **Qwen3-VL** ([[sources/qwen3-vl]] / [[entities/qwen3-vl]]) (Qwen Team Alibaba Group, 2025 Nov, arXiv:2511.21631): **Qwen-VL シリーズ第 4 世代、4 段階事前学習で 256K ネイティブ文脈を獲得**。**4 段階事前学習**: **S0 Vision-Language Alignment**（**Merger のみ**学習、67B トークン、seq 8K）→ **S1 Multimodal Pre-Training**（全パラメータ、1T、seq 8K）→ **S2 Long-Context Pre-Training**（全パラメータ、1T、**seq 32K**）→ **S3 Ultra-Long-Context Adaptation**（全パラメータ、100B、**seq 262K**）= 累積 **~2.2T トークン**。**Vision Encoder**: **SigLIP-2 から継続学習**（[[entities/qwen2-5-vl|Qwen2.5-VL]] のゼロから路線放棄）。**8 カテゴリ事前学習データ**: Image Caption + Interleaved（書籍規模で 256K トークン系列にマージ）/ Knowledge（12+ カテゴリ、importance-based sampling）/ OCR 多言語 **39 言語**（Qwen2.5-VL の 10 言語から 29 言語追加）/ Document Parsing（QwenVL-HTML + QwenVL-Markdown 2 表現）/ Grounding & Counting（[0,1000] 正規化座標、Grounding DINO 合成）/ Spatial Understanding & 3D（9-DoF 3D bbox Omni3D 形式）/ Multimodal Code（UI→HTML/CSS、画像→SVG）/ Video（Length-Adaptive Sampling、Dense Caption）/ STEM（6000 万 K-12 + 1200 万 CoT、**Multimodal Necessity Filtering**）/ Agent（GUI + Function Calling + Search）。**3 構造革新**（Interleaved MRoPE + DeepStack + テキスト・ベース時間整合）+ **平方根再重み付け**で text/multimodal バランス改善。**256K ネイティブ + YaRN 1M 外挿**、Needle-in-a-Haystack 256K 100% / 1M 99.5%。**マルチモーダル化で言語が強くなる現象**を Qwen3-235B 純粋 LLM 比で AIME-25 +4.4 / HMMT-25 +2.0 / LiveCodeBench v6 +2.5 で継続実証
- **Qwen2.5-VL** ([[sources/qwen2-5-vl]] / [[entities/qwen2-5-vl]]) (Bai et al., Qwen Team Alibaba Group, 2025 Feb, arXiv:2502.13923): **Qwen-VL シリーズ第 3 世代、事前学習データを 1.2T → 4.1T トークンへ大幅拡張**。**3 段階事前学習**: Visual Pre-Training（**1.5T トークン**、ViT のみ学習、seq 8K、Image Caption + Knowledge + OCR）→ Multimodal Pre-Training（**2T トークン**、ViT & LLM、seq 8K、+ Pure text / Interleaved / VQA / Video / Grounding / Agent）→ Long-Context Pre-Training（**0.6T トークン**、ViT & LLM、seq 32K、+ Long Video / Long Agent / Long Document）。**Window Attention 採用 ViT をゼロから学習**（DataComp + 社内データで初期化、32 層中 4 層のみ完全自己注意で計算量線形化）。**QwenVL HTML フォーマット**で文書解析データを統一表現（ABC notation 楽譜 + SMILES 化学式）。**絶対座標グラウンディング**で 10,000+ カテゴリ + 存在しないカテゴリ合成 + PixMo 点グラウンディング。**多言語 OCR**（仏・独・伊・西・葡・露・日・韓・越・アラ）。**主要結果**: OCRBench_v2 zh **63.7**（Gemini 1.5-Pro 比 +20.6 圧倒）/ DocVQA **96.4** / OCRBench **885** / CountBench **93.6 SOTA** / Charades-STA mIoU **50.9 SOTA** / MMMU **70.2 初突破**。**マルチモーダル化で言語が強くなる現象**を Qwen2.5-72B 純粋 LLM とほぼ同等性能で実証
- **Qwen2-VL** ([[sources/qwen2-vl]] / [[entities/qwen2-vl]]) (Wang et al., Qwen Team Alibaba Group, 2024 Sept, arXiv:2409.12191): **Qwen-VL シリーズ第 2 世代、累積 1.4T トークンの弱教師あり事前学習**。Stage1: 約 **600B トークン**（画像-テキスト対 + OCR + 画像分類、LLM 凍結 + ViT のみ学習）→ Stage2: 追加 **800B トークン**（画像関連データ全凍結解除）→ Stage3: ChatML SFT（ViT 凍結）。**Naive Dynamic Resolution + M-RoPE** で「任意解像度・任意アスペクト比」の Web 画像を本来の解像度のまま学習に投入できる点が特徴（[[entities/qwen-vl|Qwen-VL]] 初代の 224² → 448² 固定解像度とは対照的）。ViT は **DFN（Apple の Data Filtering Networks）** で初期化し、絶対位置埋め込みを 2D-RoPE に置換。**監督はテキスト・トークンにのみ与える**方針は Qwen-VL の伝統踏襲。**2B / 7B / 72B の 3 サイズ**（全サイズで 675M ViT を共有）。**主要結果**: DocVQA 96.5（GPT-4o +3.7）、OCRBench 877（GPT-4o +141）、多言語 OCR 8 言語中 7 で GPT-4o 超え（多言語弱教師事前学習の効果）。**学習 16K トークン上限なのに推論 80K まで頑健**という長文脈外挿は M-RoPE の位置 ID 値抑制の効果
- **Qwen-VL** ([[sources/qwen-vl]] / [[entities/qwen-vl]]) (Bai et al., Alibaba Group, 2023 Aug, arXiv:2308.12966): **「Web クロール画像-テキスト対 50 億 → 浄化後 14 億」の弱教師あり事前学習を MLLM の Stage1 として活用**。Qwen-7B + OpenCLIP ViT-bigG + Position-aware VL Adapter（**256 学習可能クエリ + 2D 絶対位置エンコーディング**）= 9.6B。**LLM を凍結**し ViT + Adapter のみを 224² で 50K ステップ・1.5B サンプル消費して訓練するという保守的設計（cf. InternVL 3 の Native Multimodal Pre-Training は対照的）。データ内訳: 英語 77.3% / 中国語 22.7%（LAION-en/zh、LAION-COCO、DataComp、Coyo、CC12M/3M、SBU、COCO Caption + 社内中国語データ）。**[[concepts/zero-shot-transfer\|ゼロショット転移]]**で Flickr30K (0-shot CIDEr) **85.8**（Flamingo-80B 67.2 を 9.6B で超え）、ScienceQA-Img (0-shot) 67.1。**`<box>` / `<ref>` 特殊トークン**による細粒度グラウンディングという「弱教師から細粒度能力を引き出す」設計が、後の Qwen2-VL / Qwen2.5-VL 系列に継承される。InternVL と並ぶ **2023-2025 年オープンソース MLLM 2 大系譜の片翼**
- **InternVL** ([[sources/internvl]] / [[entities/internvl]]) (Chen et al., OpenGVLab Shanghai AI Lab, CVPR 2024): **「視覚エンコーダを LLM 規模にスケールアップする」最初の本格的試み**。InternViT-6B（5.9B 視覚）+ QLLaMA（8B 多言語 LLaMA 初期化の言語ミドルウェア）+ 4.98B image-text pairs を **3 段階（contrastive → generative → SFT）** で訓練。**ViT-22B（21.7B）を 1/3.7 のパラメータで ADE20K linear probe +12.6 mIoU 上回る**。SigLIP 2 / PE が「対比学習自体を改良」する路線なのに対し、InternVL は **「対比 + 生成 + LLM 接続を 1 モデルで統合し、glue layer を 188M → 8B にスケール（42×）」** という路線で WSL を発展させる
- **InternVL 1.5** ([[sources/internvl-1-5]] / [[entities/internvl-1-5]]) (Chen et al., OpenGVLab Shanghai AI Lab, 2024 April): InternVL 1.0 の **MLLM 特化版**。**QLLaMA を廃止**して LLaVA 系の MLP プロジェクタに統一、InternViT-6B-448px-V1.5（dynamic 448、45 層）+ InternLM2-20B-Chat の 26B。**動的高解像度（1-40 タイル/最大 4K）+ バイリンガル OCR（英中）+ 継続事前学習** で **18 ベンチ中 8 SoTA**、ChartQA / OCRBench / MMBench-CN / CCBench / MathVista で **全商用モデル（GPT-4V / Gemini / Claude-3 / Qwen-VL-Max）超え**。InternVL シリーズが「対比学習基盤モデル」から「MLLM」へ完全シフトした転換点
- **Mini-InternVL** ([[sources/mini-internvl]] / [[entities/mini-internvl]]) (Gao et al., OpenGVLab Shanghai AI Lab, 2024 Oct): InternVL の **「軽量化 + ドメイン特化」分枝**。**[[entities/internvit-300m\|InternViT-300M]]**（CLIP-ViT-L で初期化 + InternViT-6B から知識蒸留）+ 軽量 LLM（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）で 1B/2B/4B。**「VFM の知識蒸留で WSL の軽量化問題を解く」** という方向性を確立。CLIP-ViT-L 単体より OCR / Document / Chart で大きく上回ることをアブレーションで実証
- **InternVL 2.5** ([[sources/internvl-2-5]] / [[entities/internvl-2-5]]) (Chen et al., OpenGVLab Shanghai AI Lab, 2024 Dec): **MMMU 70% を超えた初の OS MLLM**（78B モデルで 70.1%、GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 超え）。1B-78B の 7 サイズスイート、視覚は InternViT-V2.5 系（5.5B + 300M）、LLM は InternLM 2.5 / Qwen 2.5。**「アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] と完全に同じ、データ + 訓練戦略 + テスト時スケーリングのみで性能境界を拡張」** という哲学。**Progressive Scaling Strategy**（小型 LLM で ViT 訓練 → 大型 LLM に転送）で **Qwen2-VL の 1/12 の訓練トークン**（120B vs 1.4T）。**InternViT のバージョン更新で「最終層の線形分離性が低下、attention pooling 維持」** という [[entities/perception-encoder\|PE]] の中間層特徴発見と同等の現象を独立に実証
- **InternVL 3** ([[sources/internvl-3]] / [[entities/internvl-3]]) (Zhu et al., OpenGVLab Shanghai AI Lab, 2025 Apr): InternVL シリーズの **「事後的 MLLM 適応 → Native Multimodal Pre-Training」哲学転換**。WSL の文脈では特筆すべきは **「マルチモーダル化が純粋言語能力を強化する」** という初実証（Qwen2.5-Chat より InternVL 3 が言語ベンチで強い、小型モデルで +8.9）。Qwen2.5 base + テキスト 50B + マルチモーダル 150B トークン（1:3 比率）の共同事前学習 + V2PE + MPO + VisualPRM。**MMMU 72.2 で再び SOTA**、**OCRBench 906** で史上初の 900 超え、**MathVista 79.0** で GPT-4o 60.0 を +19.0 圧倒
- **EVA-CLIP**, **DFN**, **MetaCLIP** など多数の関連モデル

### B. ハッシュタグ・タグ予測

- **WSL（Mahajan et al., 2018）**: Instagram の 35 億画像とハッシュタグ（17k クラス）で訓練。SEER 系の祖先
- **Facebook SWAG**（2022）: 3.6B 枚と関連ハッシュタグで訓練、ViT で実行

### C. メタデータ駆動の検索／フィルタリング

- 検索ログ、クリックデータ、共起情報など

## CLIP が変えたもの

CLIP の登場（2021）は CV パラダイムシフト的な出来事だった：

1. **ゼロショット分類が可能に**: 「a photo of a {label}」というテキストプロンプトで、ファインチューン不要で任意のクラスを分類
2. **テキストと共通の埋め込み空間**: 画像検索（テキスト → 画像）、画像クラスタリング（意味的に）、画像生成のガイダンス（Stable Diffusion 等）が容易に
3. **「ラベル」を人手で定義する必要が消えた**: タスクをテキストで記述できれば、専用の訓練不要

## WSL の長所

- **データが取りやすい**: Web には alt text 付き画像が無尽蔵にある
- **言語と接続済み**: VLM（[[concepts/foundation-model]] 参照）への展開が自然
- **ゼロショット汎用性**: ImageNet 以外のラベル系にも対応可能
- **テキストでタスクを指定できる**: prompting で柔軟な利用が可能

## WSL の弱点（DINOv2 論文が指摘するもの）

[[sources/dinov2-learning-robust-visual-features-without-supervision]] §1, §7 で繰り返し指摘されている点：

1. **キャプションは画像の豊かさを近似するだけ**: 「a photo of a dog」というキャプションは、犬の品種・姿勢・背景・テクスチャ・3D 構造などの大半を捨てる。**ピクセルレベル情報が失われる**ので、深度推定やセグメンテーションのような密予測タスクが弱い。
2. **テキスト - 画像対のアラインメント済みデータが必要**: 純粋画像のみのデータで学習できない（NLP の LM が生テキストだけで学べるのと対照的）。
3. **キャプションのバイアスを引き継ぐ**: Web 上の alt text は「人間がどう描写したか」のバイアスを持つ。細粒度カテゴリ（鳥の種類など）や非典型的物体に弱い傾向。
4. **計算コストが大きい**: テキストエンコーダも並行訓練するので、SSL より重い（DINOv2 §9 では OpenCLIP-G の訓練は DINOv2-g の 10 倍の CO₂ 排出と試算）。

DINOv2 論文の主張は要するに「**ピクセルの細部や密予測が大事なら、SSL の方が WSL より良い**」というもの。実際 DINOv2-g は深度推定と segmentation で OpenCLIP-G を大きく上回る。

> **2025 年のアップデート: SigLIP 2 は WSL の弱点を狭めにかかっている** — [[sources/siglip-2]] は SILC/TIPS から自己蒸留＋マスク予測を借用し、LocCa から decoder ベース事前学習を借用することで、SigLIP v1 比で ADE20k seg +4.2pt、PASCAL seg +5.1pt、RefCOCO +20pt と dense/位置特定で劇的改善。ただし純粋 SSL の [[entities/dinov3]] にはまだ届かない（ADE20k linear で SigLIP 2 So/14 41.8 vs DINOv3 7B 55.9）。「SSL vs WSL」の差は確実に縮まりつつあるが、**完全には埋まっていない**。

## SSL vs WSL（実用上の使い分け）

| 用途 | 推奨 |
|---|---|
| ゼロショット分類 | WSL（CLIP 系） |
| 画像 - テキスト検索 / 生成 | WSL（CLIP 系） |
| 密予測（segmentation, depth） | SSL（DINOv2 系）|
| 細粒度分類（鳥、車、花） | SSL（DINOv2 系）|
| 凍結特徴量で k-NN / 線形 | SSL（DINOv2 系）|
| 大規模 VLM の vision tower | 現状は WSL 優勢、SSL も追いつき中 |
| 産業ドメイン（医療・衛星） | SSL（DINOv2 系がよく転移する） |

実際には「**CLIP（or SigLIP）+ DINOv2 を併用**」が広がりつつある。例: LLaVA-NeXT は CLIP + DINOv2 の特徴を結合。

## 関連ページ

- [[sources/clip]]: WSL の最も影響力ある代表論文
- [[entities/clip]] / [[entities/wit-400m]]: CLIP モデル群と訓練データ
- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: SSL が WSL に対し dense prediction で勝つことを示した論文
- [[sources/siglip-2]]: SSL 技法（自己蒸留＋マスク予測）を WSL に統合し dense 改善を実現
- [[entities/siglip]] / [[entities/perception-encoder]]: 後続の WSL 代表モデル
- [[concepts/contrastive-learning]]: CLIP 系の学習手法
- [[concepts/zero-shot-transfer]]: WSL が可能にする推論パラダイム
- [[concepts/self-supervised-learning]]: 対比される対の概念（SigLIP 2 は両者をブレンド）
- [[concepts/foundation-model]]: 両者を包括する上位概念
