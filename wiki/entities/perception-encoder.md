---
type: entity
entity_kind: model
aliases: [Perception Encoder, PE, PEcore, PEspatial, PE-Core, PE-Spatial]
tags: [weakly-supervised, vision-language, foundation-model, meta-ai, distillation]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]], [[concepts/vision-transformer]], [[concepts/promptable-concept-segmentation]]
sources: [[sources/dinov3]], [[sources/sam-3]], [[sources/siglip-2]]
updated: 2026-05-27
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

## SigLIP / SigLIP 2 との関係

PE と SigLIP は WSL 内で **2 つの対照的戦略** を取る：

| 軸 | CLIP | **SigLIP** | **SigLIP 2** | **PE** |
|---|---|---|---|---|
| 損失 | softmax InfoNCE | **sigmoid loss** | sigmoid + **decoder + 自己蒸留 + マスク予測** | PE 独自（詳細別途） |
| データ規模 | 4 億（WIT, 2021） | WebLI 大規模 | WebLI 10B / 109 言語 | **86B**（最大） |
| バッチサイズ | 32k 必須 | **小〜中で OK** | 32k | 大（131K） |
| アーキテクチャ | ViT-B/L/H | ViT-B/L/SO-400M | + **g/16 (1B), NaFlex** | PE-Core L/G + PE-Spatial G |
| Dense 予測対応 | 弱 | 弱 | **改善**（SILC/TIPS 統合） | △（PEspatial は SAM v2 蒸留で改善） |
| 効率 | 中 | **高** | 中（多目的化のコスト） | 中 |
| 多言語 | 英語のみ | 英語のみ | **109 言語** | 英語のみ |

**戦略の違い**:
- **SigLIP** = 損失関数改革（sigmoid で大バッチ不要、メモリ効率）
- **SigLIP 2** = 既存改善の統合（decoder + 自己蒸留 + マスク予測 + 多言語 + de-bias + NaFlex）
- **PE** = データ規模化（CLIP の 200 倍、SigLIP 系の数倍）

両者は補完的で、PE がデータ駆動でスケールを追う一方、SigLIP 2 は損失とアーキテクチャ多様化で性能を引き出す。**2025 年時点では SigLIP 2 のレシピが「全部入り」として後発の利を発揮**、PE は分類・global タスクの絶対精度で先行。

> **補足: SigLIP 2 の登場で PE の位置づけは変化** — SigLIP 2 が SILC/TIPS 統合で dense feature を改善したため、「PE は global、SigLIP は global＋dense 改善」という構図になりつつある。PE が global タスクで保ち続けるか、SigLIP 2 が dense でも追いつくか、2025 年後半の WSL 競争の焦点。

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

## SAM 3 の backbone として採用

[[sources/sam-3]] / [[entities/sam-3]]（Meta Superintelligence Labs, 2025）は **PE を SAM 3 の共有 backbone として採用**。これは PE が大規模 foundation model で本格採用された初の事例である：

- **共有 PE**: 検出器とトラッカーで同じ PE インスタンスを使用
- **テキスト ↔ 画像 aligned** な性質が PCS タスク（テキスト名詞句で全インスタンスを得る）に適合
- DETR ベース detector に PE 出力を fusion encoder で融合
- DINOv3 が「PE は dense task で弱い」と主張した一方、SAM 3 は **fusion encoder + DETR decoder + presence head** という追加構造で PE の弱点を補い、最終的に segmentation で SOTA を達成
- PE が **テキスト条件付き open-vocab セグメンテーションの基盤** として再評価される契機に

これにより PE は「分類用 backbone」から「**PCS タスクの基盤**」へと役割が拡大。

## 関連ページ

- [[sources/dinov3]]: PE を比較対象として頻繁に登場する論文
- [[sources/sam-3]]: PE を backbone として本格採用した foundation model
- [[entities/sam-3]]: PE を採用、DETR 構造で dense 性能を補強
- [[entities/dinov3]]: 純粋 SSL 側の対抗馬
- [[entities/clip]]: PE の元祖
- [[entities/siglip]] / [[sources/siglip]] / [[sources/siglip-2]]: 同系統の Google 競合（sigmoid 損失 + decoder + 自己蒸留 + マスク予測の統合、PE のデータ規模化と対照的）
- [[concepts/promptable-concept-segmentation]]: PE が SAM 3 で支える新タスク
- [[concepts/weakly-supervised-pretraining]]: PE が属するパラダイム
- [[concepts/foundation-model]]
