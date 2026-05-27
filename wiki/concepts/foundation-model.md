---
type: concept
aliases: [Foundation Model, 基盤モデル, foundation models]
tags: [paradigm, scaling, pretraining]
related: [[self-supervised-learning]], [[weakly-supervised-pretraining]], [[vision-transformer]], [[zero-shot-transfer]], [[contrastive-learning]], [[promptable-segmentation]], [[promptable-concept-segmentation]]
sources: [[sources/clip]], [[sources/dinov2-learning-robust-visual-features-without-supervision]], [[sources/segment-anything]], [[sources/sam-3]], [[sources/siglip]], [[sources/siglip-2]]
updated: 2026-05-27
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
- **PE** ([[entities/perception-encoder]]) / **Florence**（Microsoft）, **BASIC**（Google）, **CoCa**（Google）, **EVA-CLIP**, **MetaCLIP**, **DFN** など多数の後継

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
- **SAM 3**（Meta Superintelligence Labs, 2025, [[sources/sam-3]] / [[entities/sam-3]]）: **新タスク PCS（Promptable Concept Segmentation, [[concepts/promptable-concept-segmentation]]）を導入**。名詞句または画像 exemplar から **コンセプトの全インスタンス** を検出・セグメント・追跡。Perception Encoder backbone + DETR detector + **presence head**（認識と位置特定を分離）+ SAM 2 tracker。SA-Co（[[entities/sa-co]]、4M unique NP、benchmark 207K concepts）で訓練・評価。**foundation model の第 4 軸（コンセプト指定）** を確立。
- **DALL·E, Stable Diffusion 系**: テキスト → 画像生成の基盤として CLIP/SigLIP を構成要素に持つ
- **Florence-2**（Microsoft, 2024）: 検出/セグメンテーション/キャプション統合 vision foundation

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
- [[sources/eva-x]] / [[entities/eva-x]]: 胸部 X 線専用基盤モデル（npj Digital Medicine 2025）。EVA-02 系統の医療応用
- [[sources/i-synmed]] / [[entities/i-synmed]]: DDPM 合成 + DINO による医療基盤モデル候補（IEEE Access 2025）
