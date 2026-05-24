---
type: entity
entity_kind: model
aliases: [iBOT, image BERT pre-Training with Online Tokenizer]
tags: [self-supervised, vision-transformer, masked-image-modeling, distillation]
related: [[concepts/self-supervised-learning]], [[concepts/masked-image-modeling]], [[concepts/vision-transformer]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# iBOT（手法・モデル）

## 概要

**iBOT** = **image BERT pre-Training with Online Tokenizer**。Zhou ら（ByteDance, Johns Hopkins）が 2022 年（ICLR 2022）に発表した自己教師あり手法。**DINO（[[entities/dino]]）に Masked Image Modeling（MIM, [[concepts/masked-image-modeling]]）を統合した**ハイブリッド。

- 論文: "iBOT: Image BERT Pre-Training with Online Tokenizer"
- arXiv: 2111.07832
- コード: <https://github.com/bytedance/ibot>

DINOv2（[[entities/dinov2]]）の直接の祖先で、DINOv2 の学習目的関数はほぼ iBOT のものを継承している。

## なぜ iBOT が重要か

DINOv2 以前の SSL では、2 つの目的関数のどちらかしか選べなかった：

| | 画像レベル目的（DINO 等） | パッチレベル目的（MAE 等） |
|---|---|---|
| 強み | 凍結特徴量、k-NN、画像分類 | dense prediction、fine-tune |
| 弱み | パッチレベル表現がやや弱い | 凍結特徴量が弱い、k-NN が弱い |

**iBOT はこれを同時にやる**ことで両方の長所を取った。これが DINOv2 が「凍結特徴量が線形分類でも密予測でも強い」という驚異的バランスを実現する基礎になった。

## アーキテクチャと学習目的関数

iBOT は **DINO + パッチ単位の self-distillation MIM** という構成。

### 損失の 2 つの項

1. **[CLS] レベル損失（DINO 由来）**: 同じ画像の異なる 2 つの crop を student と teacher（EMA）に通し、[CLS] トークンの出力分布をクロスエントロピーで一致させる。詳細: [[entities/dino]]
2. **パッチレベル損失（iBOT の新規性）**: student に渡す入力パッチの一部をランダムに `[MASK]` トークンに置換。teacher にはマスクなしで渡す。student のマスクトークン出力を、teacher の対応する**可視**パッチトークン出力に一致させる。

### "Online Tokenizer" とは

iBOT のキーアイデアは「**マスクされたパッチを画素ではなく teacher の出力分布で予測する**」という点。

- **BEiT** は事前訓練された dVAE が出力する離散コードを予測（オフライン tokenizer）
- **MAE** は画素を直接予測（tokenizer 不要だが画素は意味的に粗い）
- **iBOT** は teacher network 自体を tokenizer として使う（**オンライン** = 学習中に動的に変わる）

これにより：
- 事前 tokenizer 学習が不要
- ターゲットが意味的（画素より高次）
- DINO と同じ自己蒸留枠組みに統一できる

### 元の iBOT の設計選択

| 項目 | 値 |
|---|---|
| backbone | ViT-S/B/L |
| マスク率 | 10〜50%（block-wise masking, BEiT に倣う） |
| teacher | student の EMA（DINO と同じ） |
| 損失 | CE for both [CLS] and patch |
| projection head | DINO/iBOT で**共有**（DINOv2 では分離） |
| 評価 | k-NN, linear, fine-tune, segmentation |

## 主要結果（iBOT 原論文）

| Arch | k-NN | Linear | Fine-tune |
|---|---|---|---|
| ViT-S/16 | 75.2 | 77.9 | 82.3 |
| ViT-B/16 | 77.1 | 79.5 | 84.0 |
| ViT-L/16 (INet-22k) | 72.9* | 82.3 | 86.6 |

\* k-NN は ImageNet-22k pre-training では低めに見えるが、ImageNet-1k pre-training は ViT-S/B で報告。

### 派生・後続

- **DINOv2**（[[entities/dinov2]]）: iBOT の学習目的関数を継承し、データとモデルを大規模化。ヘッドを分離、KoLeo regularizer を追加、SK centering を採用、SwiGLU や Sequence packing 等の最適化を加えた
- **MaskFeat**（Wei et al., 2022）: 似た発想で HOG 特徴を予測

## DINOv2 との違い

[[sources/dinov2-learning-robust-visual-features-without-supervision]] の §4 アブレーション（表 1）が iBOT → DINOv2 の改善を細かく示している。要点：

| 改善 | 効果（INet-1k k-NN） |
|---|---|
| ベースライン iBOT | 72.9 |
| 再現（DINOv2 著者ら） | 74.5 |
| + LayerScale + Stochastic Depth | 75.4 |
| + 128k prototypes（DINO ヘッド出力次元拡大） | 76.6 |
| + KoLeo 正則化 | 78.9 |
| + SwiGLU FFN | 78.7 |
| + Patch size 14（16→14） | 78.9 |
| + Teacher momentum 0.994（更新を遅く） | 79.4 |
| + warmup スケジュール調整 | 80.5 |
| + Batch size 3k | 81.7 |
| + Sinkhorn-Knopp centering | 81.7 |
| + ヘッド分離（DINO と iBOT で別 MLP）= **DINOv2** | 82.0 |

つまり DINOv2 は **iBOT + 10 個ほどの実装テクニック**の組み合わせとも言える。

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: iBOT を継承・拡大した後継論文
- [[entities/dino]]: iBOT の [CLS] レベル損失の元
- [[entities/dinov2]]: iBOT の正統後継
- [[concepts/masked-image-modeling]]: iBOT が属する系統の解説
- [[concepts/self-supervised-learning]]
