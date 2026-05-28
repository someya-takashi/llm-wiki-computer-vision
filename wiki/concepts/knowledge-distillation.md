---
type: concept
aliases: [KD, Knowledge Distillation, 知識蒸留]
tags: [training-technique, distillation, transfer]
related: [[self-supervised-learning]], [[vision-transformer]], [[alignment-tuning]]
sources: [[sources/dino-emerging-properties-in-self-supervised-vit]], [[sources/siglip-2]], [[sources/perception-encoder]]
updated: 2026-05-28
---

# Knowledge Distillation（KD, 知識蒸留）

## 一言で

**学習済みの「教師（teacher）モデル」の出力を、別の「生徒（student）モデル」に真似させる学習手法**。Hinton ら（2015, "Distilling the Knowledge in a Neural Network"）で広く知られるようになった。元々の動機は「大きな教師モデルの知識を、デプロイしやすい小さな生徒モデルに圧縮する」というモデル圧縮だった。

## なぜ「ハードラベル」より「ソフトラベル」が良いのか

教師モデルがある画像を「犬: 0.8, 狼: 0.15, 猫: 0.04, ...」と予測しているとき、これを生徒に真似させると、

- 正解クラス（犬）が一番高いという情報だけでなく、
- 「狼が次に近い」「猫も少し似ている」という**クラス間の類似構造**まで伝わる。

この「類似構造の情報」を Hinton らは **dark knowledge（暗黒知識）** と呼んだ。正解ラベル（one-hot, "犬=1, それ以外=0"）には含まれない情報が、教師の確率分布には含まれている。

## 基本式

教師の出力分布 $P_t$ と生徒の出力分布 $P_s$ の間のクロスエントロピーを最小化する：

$$
\mathcal{L} = H(P_t, P_s) = -\sum_k P_t^{(k)} \log P_s^{(k)}
$$

ここで両者の確率は**温度付き softmax** で計算する：

$$
P^{(i)} = \frac{\exp(z^{(i)}/\tau)}{\sum_k \exp(z^{(k)}/\tau)}
$$

- $\tau$ が高い → 分布が滑らか（差が縮まる）= soft target が強調される
- $\tau$ が低い → 分布が鋭くなる（argmax に近づく）= sharpening

> **補足: なぜ温度が必要か** — 高温で分布をなめらかにすると、クラス間の微妙な比率（dark knowledge）が生徒に伝わりやすくなる。Hinton 論文では通常 $\tau = 2 \sim 8$ 程度。

## KD の主要ユースケース

### 1. モデル圧縮（オリジナルの用途）

巨大な教師モデル → 小型の生徒モデル。デプロイ向け。例: BERT → DistilBERT、ResNet-152 → ResNet-18。

### 2. データ拡張・擬似ラベリング（noisy student, self-training）

教師モデルが大量の未ラベルデータに擬似ラベル（ソフトラベル）を付け、生徒がそれで学習する。Xie ら（2020）の Noisy Student は、ImageNet で SOTA を更新した。

### 3. 自己蒸留（self-distillation）

教師と生徒が**同じアーキテクチャ**で、生徒の重みから何らかの形で教師を構築する。本論文 DINO はその極端な形で、教師は学習中に生徒の EMA で動的に作られる。詳細: [[sources/dino-emerging-properties-in-self-supervised-vit]]。

### 4. 出力以外の中間特徴量の蒸留

中間層の活性値や attention map を一致させる FitNet / AT / SP など多数の派生がある。

### 5. データを通じた暗黙的蒸留（ACID / ACED）

[[sources/siglip-2]] が小型モデル最適化に使う近年の手法。Udandarao et al. 2024 が提案：

- **ACID**（Active Curation Implicit Distillation）: 各訓練ステップで teacher と learner がサンプルの「learnability」をスコアリングし、super-batch（64k）から最適な 32k バッチを共同選択。**soft label を渡さずに、選別の偏りだけで知識転移する**
- **ACED**（ACID + Explicit Distillation）: ACID に加え 2 つ目の teacher で明示的 softmax 蒸留

SigLIP 2 の工夫: ACED の 2-teacher 設定を、**単一 teacher（SigLIP 2 So400m）を高品質キュレーションデータで 1B ファインチューンして使う ACID 単独** に置き換え、計算を半減しつつ同等の性能を達成。

> **補足: なぜ「暗黙的」と呼ぶか** — 古典的 KD は teacher の出力分布を直接 student に渡す（**明示的**）。ACID は teacher を「データセレクタ」として使うだけで、出力分布は渡さない。しかし teacher が「student が苦手なサンプル」を選び続けることで、結果的に teacher の知識が student に「データの偏りを通じて」伝わる。これが「**暗黙的（implicit）**」蒸留と呼ばれる理由。

## 「教師あり KD」と「DINO 流 self-distillation」の違い

| | 標準的な KD | DINO 流 self-distillation |
|---|---|---|
| **教師の出所** | 事前に訓練済み、固定 | 訓練中に生徒の EMA で動的に作る |
| **教師と生徒のアーキ** | 通常は別物（大→小） | 完全に同じ |
| **ラベル使用** | 正解ラベルもあり、生徒は両方で学ぶ | **使わない**（だから "no labels"） |
| **入力** | 同じ画像を両者に流す | **異なる拡張を施した画像**を両者に流す |
| **目的** | モデル圧縮 | 表現学習（SSL） |

DINO の重要な観察は、**「教師 = 生徒の EMA」かつ「異なる拡張のクロスエントロピー」だけで、ラベルなしでも豊かな表現が学べる**という点。これは「自己教師あり学習を mean teacher 自己蒸留として再解釈する」という見方（BYOL から続く流れ）の完成形。

## 関連する歴史的線

- **Hinton et al. 2015**: 古典的 KD の確立。
- **Mean Teacher** (Tarvainen & Valpola, 2017): 半教師あり学習で「EMA で更新される教師」を導入。
- **BYOL** (Grill et al., 2020): SSL でモメンタム teacher + predictor を導入。
- **DINO** (Caron et al., 2021): self-distillation という視点で SSL を再解釈、predictor 不要を示す。
- **codistillation** (Anil et al., 2018): student も teacher もお互いに蒸留し合う双方向版。DINO とは「teacher が student の EMA」という非対称性で区別される。

## KD と Alignment Tuning の対比（**世界観が逆**）

[[concepts/alignment-tuning]]（Perception Encoder, NeurIPS 2025）は技術的には KD の損失（cosine 類似度、特徴量マッチング）を借用するが、**目的が真逆**：

| 軸 | 古典的 KD | Alignment Tuning |
|---|---|---|
| 目的 | **教師の知識を student に注入** | **student に既にある特徴を引き出す** |
| 教師の役割 | 答えを教える（情報の提供者） | アンカーを与える（情報の整列者） |
| データ規模 | 大きい（教師の知識ほぼ全部を移植） | 小さい（事前学習の 1〜10%） |
| 教師の必要性 | 強い教師モデルが必須 | **自分自身の凍結コピーで十分**（PE の自己層 41 蒸留） |
| 適用文脈 | モデル圧縮、SSL 再解釈 | 「中間層に眠る一般特徴」を最終層に上げる |

Alignment tuning は「**持っていないものを受け取る**（KD）」ではなく「**持っているものを表に出す**」という新世代の発想。PE では SAM 2.1 mask logits を外部教師として併用するが（古典的 KD）、**自己の凍結層 41 特徴を教師として併用する**（alignment 固有）点が決定的に違う。

## 関連ページ

- [[sources/dino-emerging-properties-in-self-supervised-vit]]: 知識蒸留を SSL に拡張した代表例
- [[sources/siglip-2]]: ACID / ACED の現代的応用例（暗黙的蒸留 + 明示的蒸留の組み合わせ）
- [[sources/perception-encoder]]: KD の損失を借用しつつ世界観を反転した alignment tuning を導入
- [[concepts/alignment-tuning]]: PE 由来の新ファインチューニング戦略、KD との対比
- [[concepts/self-supervised-learning]]: 多くの SSL 手法が KD として再解釈される文脈
