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
