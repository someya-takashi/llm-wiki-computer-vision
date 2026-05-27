---
type: entity
entity_kind: model
aliases: [I-SynMed, Image-Synthetic-Medical]
tags: [medical-imaging, x-ray, chest-radiography, synthetic-data, diffusion-model, ddpm, dino, self-supervised-learning, weill-cornell-qatar]
related: [[concepts/diffusion-model]], [[entities/dino]], [[concepts/self-supervised-learning]], [[concepts/vision-transformer]], [[sources/i-synmed]]
sources: [[sources/i-synmed]]
updated: 2026-05-28
---

# I-SynMed

## 概要

**I-SynMed**（推定: Image-Synthetic-Medical）は Weill Cornell Medicine-Qatar の Abdullah Hosseini・Ahmed Serag が IEEE Access 2025 で提案した医療画像 SSL パイプライン。「DDPM で合成 X 線画像 100K 枚を生成し、それを DINO で事前学習する」というシンプルな組み合わせで、実画像事前学習と同等性能を達成。

- 論文: "Self-Supervised Learning Powered by Synthetic Data From Diffusion Models: Application to X-Ray Images"
- 出典: IEEE Access 2025
- 所属: Weill Cornell Medicine-Qatar
- コードと合成データ: https://github.com/serag-ai/I-SynMed

---

## パイプライン構成

```
[Phase 1: DDPM 訓練]
  実 X 線画像（NIH 112K + COVIDx 85K = ~200K）
    ↓
  DDPM 訓練（UNet, 1000 timesteps, batch 32）

[Phase 2: 合成データ生成]
  訓練済み DDPM → 合成 X 線画像 100K 枚

[Phase 3: SSL 事前学習（並列）]
  ┌─ DINO + ViT-16（実データで pretrain）   ─→ Model A
  └─ DINO + ViT-16（合成データのみで pretrain）─→ Model B
  （A100 GPU 1 枚、500 epochs、LightlySSL ベース）

[Phase 4: 下流タスク評価]
  バックボーン凍結
  分類: 4 層線形（肺炎、気胸）
  セグメンテーション: 4 層転置畳み込み（肺、気胸）
```

---

## 主要ハイパーパラメータ

| コンポーネント | パラメータ | 値 |
|---|---|---|
| DDPM | アーキテクチャ | UNet |
| DDPM | timesteps | 1000 |
| DDPM | batch size | 32 |
| DINO | backbone | ViT-16 |
| DINO | pretraining epochs | 500 |
| DINO | image size | 224×224（center crop, ImageNet 正規化） |
| 分類ヘッド | アーキテクチャ | 4 層線形 |
| セグメンテーションヘッド | アーキテクチャ | 4 層転置畳み込み |
| Hardware | GPU | NVIDIA A100 × 1 |
| 実装ベース | ライブラリ | LightlySSL |

---

## 主要な実験結果

### 合成データ品質

| 指標 | 値 |
|---|---|
| Inception Score (IS) | 4.33 |
| FID | 12.2 |
| Mean-ABS | 3.74 |
| 最近接 SSIM | 0.64 |

### 分類タスク

| タスク | 実 pretrain | **合成 pretrain** | p 値 |
|---|---|---|---|
| 肺炎 AUC | ~98.6 | **99.1** | p=0.999 |
| 肺炎正解率 | ~94.5% | **95.0%** | p=0.999 |
| 気胸正解率 | ベースライン | わずかに優位 | p=0.562 |
| 気胸 F1 | ベースライン | わずかに優位 | p=0.562 |

### ベースライン比較（肺炎分類）

| 手法 | AUC | 正解率 |
|---|---|---|
| Google AutoML Vision | 99.1 | 94.6% |
| UMAC | 96.6 | 90.3% |
| **I-SynMed（合成）** | **99.1** | **95.0%** |

### セグメンテーションタスク

| タスク | 指標 | 実 pretrain | **合成 pretrain** | p 値 |
|---|---|---|---|---|
| 肺 | IoU | 0.72 | **0.74** | p=0.107 |
| 肺 | Dice | 0.83 | **0.85** | p=0.107 |
| 肺 | mAP | 0.50 | **0.55** | p=0.107 |
| 気胸 | IoU | 0.13 | **0.14** | p=0.127 |
| 気胸 | Dice | 0.25 | **0.26** | p=0.127 |

### 混合データ比率（実：合成 = 3:1, 1:1, 1:3）

統計的に有意な差なし（p=0.977、FDR 補正後）。「**50/50 がわずかに最良**」。

---

## アブレーション：SSL アルゴリズムとバックボーン

| 構成 | 分類性能 |
|---|---|
| MoCo + ResNet18（実） | ベースライン |
| MoCo + ResNet18（合成） | 同程度 |
| **DINO + ViT-16（実）** | **最良** |
| **DINO + ViT-16（合成）** | **最良に近い** |

DINO + ViT-16 が MoCo + ResNet18 を「**大幅に上回り、より速く収束**」した（論文記述）。

---

## ノイズ耐性（DINO 特徴の頑健性、肺炎分類 KNN）

| ノイズタイプ | 結果 |
|---|---|
| ガウシアンノイズ | 性能大きく低下 |
| 塩胡椒ノイズ | 性能大きく低下 |
| **ぼかし（カーネルサイズ 25）** | **KNN 正解率 ~0.9 維持**（DINO の augmentation に blur が含まれるため）|

---

## データセット

| データセット | 用途 | サイズ |
|---|---|---|
| **COVIDx CXR-4** | DDPM/DINO 訓練 | 85,000 枚（COVID-19 二値分類）|
| **NIH Chest X-ray** | DDPM/DINO 訓練 | 112,120 枚（15 疾病カテゴリ）|
| **合成（本研究生成）** | DINO 事前学習（合成版） | 100,000 枚 |
| **Pneumonia Dataset** | 評価（分類） | — |
| **SIIM-ACR Pneumothorax** | 評価（セグメンテーション） | — |
| **Pulmonary Datasets** | 評価（セグメンテーション） | — |

---

## 限界

1. **DDPM の memorization 問題未検証**: 合成データのプライバシー保護効果が定量的に示されていない
2. **稀少疾患での未検証**: 一般的肺疾患のみで評価、稀少疾患では合成データの限界が明確化する可能性
3. **計算コスト**: DDPM の 1000 timesteps + 200K 画像訓練は resource-constrained environment では困難
4. **既存手法の組み合わせ**: 手法的新規性は低い（DDPM + DINO の既存手法組み合わせ）
5. **ベースライン選択が古い**: MAE、SimCLR v2、医療画像基盤モデル（MedSAM, RadFM 等）との比較がない

---

## EVA-X との対比

[[entities/eva-x]]（npj Digital Medicine 2025）は同じ「胸部 X 線 + SSL」だが対照的なアプローチを取る：

| 観点 | **I-SynMed** | **EVA-X** |
|---|---|---|
| ジャーナルの質 | IEEE Access（中堅） | **npj Digital Medicine（最高峰）** |
| データ規模 | 100K（合成） | **520K（実画像）** |
| データ性質 | DDPM 合成 | NIH+CheXpert+MIMIC 実 |
| SSL アルゴリズム | DINO（自己蒸留）| MIM + 凍結 CLIP トークナイザ |
| バックボーン | ViT-16（DINO 標準）| EVA-02 改良 ViT |
| 主張 | 「合成 = 実」を統計実証 | 医療基盤モデルを構築 |
| 規模感 | 1 GPU 500 epoch | 大規模事前学習 + 多タスク SOTA |

両者は「**胸部 X 線 SSL の二つの異なる道**」を代表する：応用研究 vs システム研究。

## 関連ページ

- [[sources/i-synmed]]: 詳細な論文要約
- [[translations/i-synmed]]: 日本語全文翻訳
- [[concepts/diffusion-model]]: DDPM の概念解説
- [[entities/dino]]: SSL バックボーンとして使用
- [[entities/eva-x]]: 対照される胸部 X 線基盤モデル（実画像 + MIM）
- [[concepts/self-supervised-learning]]: SSL の全体像と医療応用
- [[concepts/vision-transformer]]: バックボーンアーキテクチャ
