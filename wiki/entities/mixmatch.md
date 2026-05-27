---
type: entity
entity_kind: model
aliases: [MixMatch]
tags: [semi-supervised-learning, consistency-regularization, entropy-minimization, mixup, label-guessing, sharpening, wide-resnet, cifar, svhn, stl-10, google-research]
related: [[concepts/semi-supervised-learning]], [[concepts/self-supervised-learning]], [[entities/simclr]], [[entities/byol]], [[entities/dino]]
sources: [[sources/mixmatch]]
updated: 2026-05-27
---

# MixMatch

## 概要

**MixMatch** は Google Research の David Berthelot, Nicholas Carlini, Ian Goodfellow らが 2019 年に発表した半教師あり学習アルゴリズム。「一貫性正則化」「エントロピー最小化」「MixUp 正則化」という 3 つの主要アプローチを 1 本のアルゴリズムに統合し、CIFAR-10（250 ラベル）のエラー率を従来最良（VAT: 36.03%）の **3.3 倍改善**（11.08%）した。

- 論文: "MixMatch: A Holistic Approach to Semi-Supervised Learning"
- arXiv: 1905.02249（2019）
- 会議: NeurIPS 2019
- 所属: Google Research

---

## アルゴリズム構成

```
入力: ラベルあきバッチ X + ラベルなしバッチ U
   │
   ├── X: 1 回データ拡張 → X̂
   │
   └── U: K 回データ拡張 → {û_{b,1}, ..., û_{b,K}}
           ↓
       各 k について p_model(y|û_{b,k}) を計算
           ↓
       K 個の予測を平均 → q̄_b （一貫性確認）
           ↓
       Sharpen(q̄_b, T) → q_b （エントロピー最小化）
           ↓
   [X̂, Û] をシャッフル → W
   MixUp(X̂, W[:B]) → X'  ←── 損失: クロスエントロピー
   MixUp(Û, W[B:]) → U'  ←── 損失: Brier スコア（L2）
           ↓
   L = L_X + λ_U * L_U
```

---

## 主要ハイパーパラメータ

| パラメータ | デフォルト | 意味 |
|---|---|---|
| $T$ | 0.5 | シャープニング温度（0 → one-hot, 1 → 変化なし） |
| $K$ | 2 | 拡張数 |
| $\alpha$ | 0.75 | MixUp Beta 分布パラメータ |
| $\lambda_{\mathcal{U}}$ | データセット依存 | ラベルなし損失の重み（CIFAR: 75, SVHN: 250, STL: 50） |

---

## 主要な実験結果

### CIFAR-10 エラー率（Wide ResNet-28, 150 万パラメータ）

| 手法 | 250 ラベル | 4000 ラベル |
|---|---|---|
| Π-Model | 53.02% | 17.41% |
| Pseudo-Label | 49.98% | 16.21% |
| VAT | 36.03% | 11.05% |
| Mean Teacher | 47.32% | 10.36% |
| **MixMatch** | **11.08%** | **6.24%** |
| 完全教師あり（50K 枚） | — | 4.13% |

### 他データセット（250 ラベル）

| データセット | MixMatch | 次点 |
|---|---|---|
| SVHN | 3.78% | Mean Teacher: 6.45% |
| SVHN+Extra | 2.22% | Mean Teacher: 2.77% |
| STL-10（1000 ラベル） | 10.18% | IIC（5000 ラベル使用）: 11.20% |

### 差分プライバシー（SVHN + PATE フレームワーク）

| 手法 | 精度 | $\varepsilon$（小さいほどプライバシー強） |
|---|---|---|
| VAT | 91.6% | 4.96 |
| MixMatch + PATE | **95.21%** | **0.97** |

---

## アブレーション（CIFAR-10, 250 ラベル）

| 条件 | エラー率 |
|---|---|
| MixMatch（完全版） | **11.08%** |
| MixUp 除去 | 39.11%（最大の退化） |
| シャープニング除去（T=1） | 27.83% |
| K=1（分布平均なし） | 17.09% |
| ラベルあきのみ MixUp | 32.16% |
| ラベルなしのみ MixUp | 12.35% |

---

## 後継手法

| 手法 | MixMatch との関係 |
|---|---|
| **ReMixMatch**（Berthelot et al., 2020） | 分布アライメント + 拡張アンカリングを追加 |
| **FixMatch**（Sohn et al., 2020） | 「高信頼ラベルなし例のみ使用（閾値 0.95）+ RandAugment」でシンプル化。CIFAR-10 / 250 ラベルで 5.07% |
| **FlexMatch, FreeMatch** | 動的閾値による FixMatch の改善 |
| **UDA**（Unsupervised Data Augmentation） | 強力な拡張（AutoAugment + CTAugment）を一貫性正則化に適用 |

---

## 関連ページ

- [[sources/mixmatch]]: 詳細な論文要約
- [[translations/mixmatch]]: 日本語全文翻訳
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像
- [[concepts/self-supervised-learning]]: MixMatch と比較される自己教師あり学習
- [[entities/simclr]]: 一貫性正則化の思想を自己教師あり学習に発展させた手法
- [[entities/byol]]: 非対比 SSL の先駆け（自己教師あり学習）
