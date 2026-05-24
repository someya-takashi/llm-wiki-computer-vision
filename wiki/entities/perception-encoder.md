---
type: entity
entity_kind: model
aliases: [Perception Encoder, PE, PEcore, PEspatial, PE-Core, PE-Spatial]
tags: [weakly-supervised, vision-language, foundation-model, meta-ai, distillation]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]], [[concepts/vision-transformer]]
sources: [[sources/dinov3]]
updated: 2026-05-24
---

# Perception Encoder（PE）

## 概要

**Perception Encoder（PE）** は Meta AI Research が 2024 年に発表した視覚基盤モデル群。**画像-テキスト対比学習（CLIP スタイル）の規模を極限まで押し上げ**、global と dense 両方の特徴量で SOTA を狙うシリーズ。DINOv3（[[entities/dinov3]]）論文において、**主要な比較相手**として頻繁に登場する。

- 論文: "Perception Encoder: The best visual embeddings are not at the output of the network"
- arXiv: 2504.13181（2024）
- コード: <https://github.com/facebookresearch/perception_models>
- メーカー: Meta AI Research

PE は 2 つの主要バリアントで構成される：

| バリアント | 用途 | 強み |
|---|---|---|
| **PEcore** | global タスク（分類・検索） | グローバル画像-テキスト整合性 |
| **PEspatial** | dense タスク（segmentation・depth） | SAM v2 からの蒸留で空間構造が強い |

---

## PEcore（global 版）

CLIP スタイルの contrastive 学習を、巨大データ + 巨大モデルで実行：

- **訓練データ**: 86B（860 億）画像-テキスト対（Meta 内部キュレーション）
- **グローバルバッチサイズ**: 131K
- **モデルサイズ**: ViT-B/14, L/14, **G/14（主力）**
- **損失**: CLIP の対比損失 + 追加の中間層整合性損失

PE のキー観察は **"the best visual embeddings are not at the output of the network"**（最良の視覚埋め込みはネットワークの出力にはない）。つまり：

- 最終層の出力（CLS トークン）は分類向けに「圧縮されすぎ」ている
- **中間層の特徴量**の方が下流タスクに有用な情報を保持している
- これを活用する追加の loss term と pooling 戦略を採用

### 主要結果（ImageNet-1k 線形プローブ）

| Model | Arch | Linear |
|---|---|---|
| OpenCLIP G/14 | LAION-2B | 86.2 |
| EVA-CLIP g/14 | custom | 86.4 |
| DINOv2 g/14 | LVD-142M | 87.3 |
| DINOv3 7B/16 | LVD-1689M | 88.4 |
| **PEcore G/14** | 86B pairs | **89.3** |

DINOv3 と並ぶ最強クラス。ImageNet-A（adversarial）では PEcore が DINOv3 を上回る（89.0 vs 86.9）。

---

## PEspatial（dense 版）

PEcore をベースに、**SAM v2（Segment Anything Model v2）から蒸留**して dense 特徴量を強化した派生：

- **教師モデル**: SAM v2（マスク注釈で訓練された強力なセグメンテーションモデル）
- **蒸留損失**: student（PEcore）と teacher（SAM v2）のパッチ特徴量間のコサイン類似度を高める
- 結果: dense タスクでの大幅な性能向上、ただし global 性能は若干犠牲

### DINOv3 との関係

DINOv3 の Gram anchoring（[[concepts/gram-anchoring]]）も「student と teacher のパッチ特徴量関係を保つ」という同じ系統の発想だが、決定的な違いがある：

| | PEspatial | DINOv3 Gram Anchoring |
|---|---|---|
| Teacher の出所 | **SAM v2**（教師あり、マスク注釈で訓練） | **SSL モデル自体の早期スナップショット** |
| 訓練の独立性 | 別ジョブで蒸留 | 同じ SSL ジョブ内で同時最適化 |
| 必要な supervision | SAM v2 の作成にマスク注釈が必要 | **完全にラベルフリー** |
| 特徴量の制約 | 直接マッチング（コサイン） | Gram 行列のみ（特徴量自体は自由） |

DINOv3 の主張は「**SAM のような supervised teacher なしで、純粋に SSL で同等以上の dense 性能が出る**」というもの。

---

## DINOv3 比較表での PE の位置

[[sources/dinov3]] §6 のベンチマークでの PE の振る舞い：

### 強い領域
- **ImageNet 系列の分類**（特に難しい OOD: ImageNet-A, ObjectNet）
- **細粒度分類**（PEcore が iNaturalist 21 で 87.0 と高い）
- **動画分類** Kinetics-400（PEcore TTA 88.8）

### 弱い領域
- **Dense linear probing**: ADE20k で PEspatial 49.3 < DINOv3 55.9
- **3D correspondence**: NAVI で PEcore 39.9 < DINOv3 64.4
- **Video tracking**（PEcore は低性能、PEspatial は SAM v2 蒸留で SAM v2 が動画モデルなので比較的健闘）
- **Instance retrieval**: Oxford-Hard で PEcore 28.8 / PEspatial 25.1 << DINOv3 60.7

DINOv3 著者らの主張: **「PE のような巨大 WSL モデルでも、dense feature の質と instance retrieval では純粋 SSL に勝てない」**

---

## PE が依然強い理由

- **86B の画像-テキスト対比学習はテキスト情報を埋め込みに直接組み込める**
- DINOv3 dino.txt のような後付けテキスト整合性ではなく、native に integrated
- ゼロショット分類はテキスト整合性 native の方が強い
- グローバルな概念理解（猫 vs 犬の区別など）はテキスト誘導が依然有利

---

## 弱教師あり vs 自己教師あり: PE と DINOv3 の対比

DINOv3 と PE は「**同じ Meta が同時期に出した、対照的なアプローチ**」と見ることもできる：

| 軸 | PE | DINOv3 |
|---|---|---|
| 学習信号 | テキスト + 画像 | 画像のみ |
| データキュレーション | テキストキャプションでフィルタ | 視覚的類似度でフィルタ |
| 強み | global 分類・ゼロショット | dense 予測・instance 検索・3D |
| データ規模 | 86B image-text pairs | 1.689B images |
| 公開モデル | PE-Core L/G, PE-Spatial G | ViT-S〜7B + ConvNeXt |

両者は補完的で、実際の応用では**併用**されることも多い（例: VLM の vision tower に両方をマージする手法）。

---

## 関連ページ

- [[sources/dinov3]]: PE を比較対象として頻繁に登場する論文
- [[entities/dinov3]]: 純粋 SSL 側の対抗馬
- [[entities/clip]]: PE の元祖
- [[entities/siglip]]: 同系統の Google 競合
- [[concepts/weakly-supervised-pretraining]]: PE が属するパラダイム
- [[concepts/foundation-model]]
