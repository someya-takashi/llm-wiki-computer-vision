---
type: entity
entity_kind: model
aliases: [FlexMatch]
tags: [semi-supervised-learning, pseudo-label, dynamic-threshold, curriculum-learning, fixmatch, cifar, svhn, stl-10, imagenet, titech, microsoft-research-asia]
related: [[concepts/semi-supervised-learning]], [[entities/fixmatch]], [[entities/mixmatch]], [[sources/flexmatch]]
sources: [[sources/flexmatch]]
updated: 2026-05-27
---

# FlexMatch

## 概要

**FlexMatch** は東京工業大学の Bowen Zhang・Yidong Wang ら（Microsoft Research Asia との共著）が 2021 年に発表した半教師あり学習アルゴリズム。FixMatch の「全クラス固定閾値」を「**クラス別動的閾値**（CPL: Curriculum Pseudo Labeling）」に置き換えることで、追加パラメータ・追加推論コストなしに大幅な性能改善と収束加速を実現した。

- 論文: "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"
- arXiv: 2110.08263（2021）
- 会議: NeurIPS 2021
- 所属: 東京工業大学・Microsoft Research Asia
- コード: https://github.com/TorchSSL/TorchSSL

---

## CPL の核心式

```
各反復で:

1. 各クラス c について推定学習効果を計算:
   σ_t(c) = Σ 1[max(q_b) > τ] × 1[argmax(q_b) = c]
   （τ を超えてクラス c に予測されたラベルなし例の数）

2. 正規化:
   ウォームアップ中: β_t(c) = σ_t(c) / max{ max_c σ_t, N - Σσ_t }
   通常時:          β_t(c) = σ_t(c) / max_c σ_t

3. クラス別閾値:
   T_t(c) = M(β_t(c)) × τ    where M(x) = x/(2-x) [凸型]

4. 損失（FixMatch と同形式、閾値のみ変更）:
   L_u,t = (1/μB) Σ 1[max(q_b) > T_t(argmax(q_b))] · H(q̂_b, p_m(y|A(u_b)))
```

---

## デフォルトハイパーパラメータ（FixMatch から変更なし）

| パラメータ | CIFAR-10 | CIFAR-100 | STL-10 | SVHN | ImageNet |
|---|---|---|---|---|---|
| モデル | WRN-28-2 | WRN-28-8 | WRN-37-2 | WRN-28-2 | ResNet-50 |
| τ（上限） | 0.95 | 0.95 | 0.95 | 0.95 | 0.70 |
| μ（ラベルなし比） | 7 | 7 | 7 | 7 | 1 |
| B（バッチサイズ） | 64 | 64 | 64 | 64 | 128 |
| lr | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| weight decay | 5e-4 | 1e-3 | 5e-4 | 5e-4 | 3e-4 |
| EMA | 0.999 | | | | |

---

## 主要な実験結果（最良エラー率 %）

| データセット | ラベル数 | FlexMatch | FixMatch | 差 |
|---|---|---|---|---|
| CIFAR-10 | 40 | **4.97** | 7.47 | −2.50 |
| CIFAR-10 | 250 | 4.98 | 4.86 | +0.12 |
| CIFAR-10 | 4000 | **4.19** | 4.21 | −0.02 |
| CIFAR-100 | 400 | **39.94** | 46.42 | −6.48 |
| CIFAR-100 | 2500 | **26.49** | 28.03 | −1.54 |
| CIFAR-100 | 10000 | **21.90** | 22.20 | −0.30 |
| STL-10 | 40 | **29.15** | 35.97 | −6.82 |
| STL-10 | 250 | **8.23** | 9.81 | −1.58 |
| STL-10 | 1000 | **5.77** | 6.25 | −0.48 |
| SVHN | 40 | 8.19 | **3.81** | +4.38（悪化） |
| SVHN | 1000 | 6.72 | **1.96** | +4.76（悪化） |
| ImageNet (100K) | — | **41.85** top-1 | 43.66 | −1.81 |

**収束速度（CIFAR-100 / 400 ラベル）**

| 手法 | 200K 反復時の精度 | 最終精度（≒1M 反復） |
|---|---|---|
| FixMatch | 56.35% | ~53% |
| **FlexMatch** | **94.29%** | **60.06%** |

FlexMatch の 200K 反復時の精度がすでに FixMatch の最終精度を大幅に上回る。訓練時間 1/5 未満で同等以上の性能。

---

## クラスごとの精度比較（CIFAR-10 / 40 ラベル）

| クラス | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| FixMatch | 0.964 | 0.982 | 0.697 | 0.852 | 0.974 | 0.890 | 0.987 | 0.970 | 0.982 | 0.981 |
| **FlexMatch** | 0.967 | 0.980 | **0.921** | **0.866** | 0.957 | 0.883 | 0.988 | 0.975 | 0.982 | 0.968 |

クラス 2（+0.224）・3（+0.014）が特に改善。苦手クラスを重点的に学習する CPL の効果が見える。

---

## 後継・関連手法

| 手法 | FlexMatch との関係 |
|---|---|
| **FreeMatch**（Wang et al., 2023） | クラス全体・局所の 2 段階自由閾値。SVHN 不均衡問題への対処を含む |
| **SoftMatch**（Chen et al., 2023） | ハード one-hot 閾値をガウス重み付けに変える（低信頼サンプルもゼロではなく使う） |
| **SemiReward**（Zhang et al., 2024） | 閾値の代わりに品質報酬関数で疑似ラベルを選別 |

---

## 関連ページ

- [[sources/flexmatch]]: 詳細な論文要約（CPL の設計詳細、SVHN 失敗事例の考察）
- [[translations/flexmatch]]: 日本語全文翻訳
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像
- [[entities/fixmatch]]: FlexMatch の直接の出発点（固定閾値版）
- [[entities/mixmatch]]: FixMatch の前身
