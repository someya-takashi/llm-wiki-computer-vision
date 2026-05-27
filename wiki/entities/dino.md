---
type: entity
entity_kind: model
aliases: [DINO, self-DIstillation with NO labels]
tags: [self-supervised, vision-transformer, fair]
related: [[concepts/self-supervised-learning]], [[concepts/vision-transformer]], [[concepts/knowledge-distillation]]
sources: [[sources/dino-emerging-properties-in-self-supervised-vit]]
updated: 2026-05-24
---

# DINO（手法・モデル）

## 概要

**DINO** = **self-DIstillation with NO labels**。Facebook AI Research（FAIR）と Inria の共同チーム（Caron, Touvron ら）が 2021 年に発表した、Vision Transformer（[[concepts/vision-transformer]]）向け自己教師あり学習手法。コード・モデル公開済み: <https://github.com/facebookresearch/dino>

論文の詳細解説: [[sources/dino-emerging-properties-in-self-supervised-vit]] / 翻訳: [[translations/dino-emerging-properties-in-self-supervised-vit]]

## アーキテクチャと学習設定（要点）

| 項目 | 値 / 説明 |
|---|---|
| 学習パラダイム | 自己教師あり（ラベル不要） |
| 損失 | クロスエントロピー（CE） |
| 教師ネット | student の EMA（モメンタム・エンコーダ）。$\lambda$ は 0.996 → 1 のコサインスケジュール |
| 崩壊回避 | teacher 出力の **centering**（バッチ平均減算, momentum $m$）+ **sharpening**（低温度 softmax, $\tau_t = 0.04 \to 0.07$ ウォームアップ） |
| データ拡張 | BYOL 系（color jittering, Gaussian blur, solarization）+ **multi-crop**（2 × global $224^2$ + 数本 × local $96^2$）|
| projection head | 3 層 MLP（hidden=2048, GELU）→ $\ell_2$ 正規化 → 重み正規化全結合層（$K=65536$） |
| 正規化 | **BN フリー**（projection head も含む） |
| predictor | **なし**（BYOL/SimSiam とは異なる） |
| 学習データ | ImageNet（ラベルなし） |
| optimizer | AdamW、lr = 0.0005 × batchsize/256、コサイン減衰、重み減衰 0.04→0.4 |
| バッチサイズ | 1024（16 GPU）/ 8〜128 でも動く |
| 学習時間 | ViT-S/16, 300 epoch で 16 GPU × ~3 日 |

## 主要結果（ImageNet）

| 設定 | Linear top-1 | k-NN top-1 |
|---|---|---|
| DINO + ResNet-50 | 75.3 | 67.5 |
| DINO + ViT-S/16 | 77.0 | 74.5 |
| DINO + ViT-S/8 | 79.7 | 78.3 |
| DINO + ViT-B/16 | 78.2 | 76.1 |
| DINO + **ViT-B/8** | **80.1** | **77.4** |

参考（教師あり）:
- Supervised ResNet-50: 79.3
- Supervised ViT-S/16: 79.8

## 創発的特性（emergent properties）

DINO で学習した ViT には、教師あり学習でも他 SSL でも現れない以下の性質が観察される：

1. **自己注意マップに教師なしでオブジェクト境界が現れる**。最終層の [CLS] トークンの attention map を可視化すると、犬・鳥・人などの輪郭が浮かび上がる。詳細: [[sources/dino-emerging-properties-in-self-supervised-vit]] の「創発する自己注意マップ」セクション。
2. **k-NN 精度が線形評価にほぼ匹敵**（74.5% vs 77.0% on ViT-S）。詳細: [[concepts/knn-evaluation-protocol]]。
3. **DAVIS 動画セグメンテーションで、訓練なしの単純な特徴量最近傍だけで SSL convnet を上回る**（ViT-B/8 で $(\mathcal{J}\&\mathcal{F})_m = 71.4$）。

## 系譜と派生

### 由来

- **MoCo** からモメンタム・エンコーダを継承
- **BYOL** から「対比損失なしで teacher-student を一致させる」発想を継承
- **SwAV** からマルチクロップ拡張を継承
- **Hinton 流の知識蒸留** をラベルなしへ拡張する解釈

### 後続

- **iBOT** (Zhou et al., 2021/ICLR 2022, [[entities/ibot]], [[sources/ibot]]): DINO + Masked Image Modeling のハイブリッド。DINO の [CLS] 自己蒸留損失を継承しつつ、**online tokenizer**（[[concepts/online-tokenizer]]）として teacher 自身を MIM の教師信号源に流用するという新発想で MIM と self-distillation を統合。DINOv2/DINOv3 の損失関数の直接の源。
- **DINOv2** (Oquab et al., 2023, [[entities/dinov2]]): iBOT を 1.42 億枚規模のキュレーションデータ（[[entities/lvd-142m]]）で拡大学習。**汎用視覚基盤モデルとして提案され、凍結特徴量のまま OpenCLIP に勝つ**。詳細: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
- **DINOv2 with registers** (Darcet et al., 2023): attention artifact 解消
- **DINOv3** など継続的に改良が続く系譜

### 応用領域：医療画像

[[sources/i-synmed]]（IEEE Access 2025）は DINO + ViT-16 を**胸部 X 線画像**に適用した代表的応用例。特筆すべきは「DDPM で生成した合成 X 線画像で事前学習しても実画像と同等性能」を統計的に実証した点。データプライバシーが厳しい医療領域では、DINO 等の SSL と合成データの組み合わせが現実的な解決策として注目されている。Ablation で DINO + ViT-16 が MoCo + ResNet18 を大幅に上回ったことも、DINO の医療画像での優位性を裏付ける。

## 略称・固有名対応

- **DINO**: 本手法
- **DINOv2**: 続編（別エンティティとして将来追加予定）
- **EMA**: Exponential Moving Average
- **CE**: Cross Entropy

## 関連ページ

- [[sources/dino-emerging-properties-in-self-supervised-vit]]: 原論文の詳細解説
- [[translations/dino-emerging-properties-in-self-supervised-vit]]: 原論文の全訳
- [[concepts/self-supervised-learning]]
- [[concepts/vision-transformer]]
- [[concepts/knowledge-distillation]]
- [[concepts/knn-evaluation-protocol]]
- [[entities/imagenet]]
- [[sources/i-synmed]] / [[entities/i-synmed]]: DINO の医療画像応用（DDPM 合成データでの事前学習）
- [[concepts/diffusion-model]]: DDPM 等。合成データ生成で SSL と組み合わされる
