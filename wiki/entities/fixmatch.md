---
type: entity
entity_kind: model
aliases: [FixMatch]
tags: [semi-supervised-learning, pseudo-label, confidence-threshold, weak-strong-augmentation, consistency-regularization, randaugment, ctaugment, cifar, svhn, stl-10, imagenet, google-research]
related: [[concepts/semi-supervised-learning]], [[entities/mixmatch]], [[sources/fixmatch]]
sources: [[sources/fixmatch]]
updated: 2026-05-27
---

# FixMatch

## 概要

**FixMatch** は Google Research の Kihyuk Sohn, David Berthelot, Nicholas Carlini らが 2020 年に発表した半教師あり学習アルゴリズム。「弱い拡張で疑似ラベルを生成し、強い拡張でその疑似ラベルを学習させる」という **弱→強の非対称性** と、**信頼度閾値 τ=0.95** の組み合わせだけで MixMatch を大幅に上回り、半教師あり学習のシンプルなベースラインを確立した。

- 論文: "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
- arXiv: 2001.07685（2020）
- 会議: NeurIPS 2020
- 所属: Google Research

---

## アルゴリズム（擬似コード）

```
Input: ラベルあきバッチ X（B 枚）, ラベルなしバッチ U（μB 枚）, 閾値 τ, 比率 μ, 損失係数 λ_u

# ラベルあき損失
ℓ_s = (1/B) Σ_b H(p_b, p_m(y | α(x_b)))

# 疑似ラベル生成
for b = 1 to μB:
    q_b = p_m(y | α(u_b))   ← 弱い拡張で予測

# ラベルなし損失（高信頼のみ）
ℓ_u = (1/μB) Σ_b 1{max(q_b) ≥ τ} · H(argmax(q_b), p_m(y | A(u_b)))
                                                           ↑ 強い拡張で評価

return ℓ_s + λ_u · ℓ_u
```

- `α(·)`: 弱い拡張（水平フリップ + 微小ランダムクロップ）
- `A(·)`: 強い拡張（RandAugment または CTAugment + Cutout）
- `H(p, q)`: クロスエントロピー損失
- `1{·}`: 閾値を超えた場合 1、超えなかった場合 0

---

## デフォルトハイパーパラメータ（ImageNet 以外）

| パラメータ | 値 | 意味 |
|---|---|---|
| τ | 0.95 | 信頼度閾値 |
| λ_u | 1 | ラベルなし損失の重み |
| μ | 7 | ラベルなし / ラベルあきのバッチ比 |
| B | 64 | ラベルあきバッチサイズ |
| lr | 0.03 | 学習率 |
| β | 0.9 | SGD Nesterov momentum |
| weight decay | 0.0005 / 0.001 | CIFAR-10/SVHN/STL: 5e-4、CIFAR-100: 1e-3 |
| 学習スケジュール | コサイン減衰 | 1,000,000 ステップ |
| EMA | 0.999 | モデルの指数移動平均（評価用） |

---

## 主要な実験結果（エラー率 %）

**表2: ベンチマーク比較（WideResNet-28-2 または 28-8）**

| データセット | ラベル数 | FixMatch (RA) | FixMatch (CTA) | 次点 (MixMatch) |
|---|---|---|---|---|
| CIFAR-10 | 40 | 13.81 | 11.39 | — |
| CIFAR-10 | 250 | 5.07 | 5.07 | 11.08 |
| CIFAR-10 | 4000 | 4.26 | 4.31 | 6.24 |
| CIFAR-100 | 400 | 48.85 | 49.95 | — |
| CIFAR-100 | 2500 | 28.29 | 28.64 | — |
| CIFAR-100 | 10000 | 22.60 | 23.18 | — |
| SVHN | 40 | 3.96 | 7.65 | — |
| SVHN | 250 | 2.48 | 2.64 | 3.78 |
| SVHN | 1000 | 2.28 | 2.36 | — |
| STL-10 | 1000 | 7.98 | 5.17 | 10.18 |

- **RA**: RandAugment を強い拡張として使用
- **CTA**: CTAugment を強い拡張として使用

**ほぼ無ラベル設定（CIFAR-10、1 クラスあたり 1 枚 = 計 10 ラベル）**

| 指標 | 値 |
|---|---|
| 中央値（正解率） | **64.28%** |
| 最良値（正解率） | 78.30% |
| 最悪値（正解率） | 26.29% |

**ImageNet（10% ラベル = 128,000 枚、WideResNet-50-2）**

| 手法 | Top-1 エラー率 |
|---|---|
| UDA | 31.22% |
| **FixMatch** | **28.54%** |

---

## 信頼度閾値 τ のアブレーション（CIFAR-10 / 250 ラベル）

| τ | マスク率（捨てる割合） | 疑似ラベル不純度 | エラー率 |
|---|---|---|---|
| 0.25 | 0.00% | 6.39% | 6.40% |
| 0.50 | 80.02% | 4.30% | 5.75% |
| 0.75 | 92.56% | 3.57% | 5.03% |
| **0.95** | **98.13%** | **3.47%** | **4.84%** |
| 0.99 | 92.14% | 2.06% | 5.05% |

τ=0.95 で 98% を捨てるが最良。「質 > 量」の原則。

---

## 後継手法

| 手法 | FixMatch との関係 |
|---|---|
| **FlexMatch**（Zhang et al., 2021） | 全クラス固定閾値 → クラス別動的閾値（CPL）に変更。CIFAR-10/40L で 4.97%（FixMatch: 7.47%）を達成。ただし SVHN（クラス不均衡）では悪化 |
| **FreeMatch**（Wang et al., 2023） | クラス別＋局所適応の 2 レベル自由閾値。FlexMatch の SVHN 問題を改善 |
| **SoftMatch**（Chen et al., 2023） | ハード閾値をガウス重み付けに変え、低信頼サンプルもゼロでなく使用 |

## 関連ページ

- [[sources/fixmatch]]: 詳細な論文要約（弱→強非対称性の解説・CIFAR-100 異常の考察を含む）
- [[translations/fixmatch]]: 日本語全文翻訳
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像
- [[entities/mixmatch]]: FixMatch の直前の先行手法
- [[entities/flexmatch]]: FlexMatch（直接の後継手法）
