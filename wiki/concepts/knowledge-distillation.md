---
type: concept
aliases: [KD, Knowledge Distillation, 知識蒸留]
tags: [training-technique, distillation, transfer]
related: [[self-supervised-learning]], [[vision-transformer]]
sources: [[sources/dino-emerging-properties-in-self-supervised-vit]]
updated: 2026-05-24
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

## 関連ページ

- [[sources/dino-emerging-properties-in-self-supervised-vit]]: 知識蒸留を SSL に拡張した代表例
- [[concepts/self-supervised-learning]]: 多くの SSL 手法が KD として再解釈される文脈
