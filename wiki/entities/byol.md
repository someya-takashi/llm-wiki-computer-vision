---
type: entity
entity_kind: model
aliases: [BYOL, "Bootstrap Your Own Latent"]
tags: [ssl, non-contrastive, self-supervised-learning, momentum-encoder, representation-learning, resnet, imagenet, predictor]
related: [[concepts/self-supervised-learning]], [[concepts/contrastive-learning]], [[entities/simclr]], [[entities/dino]], [[entities/moco]]
sources: [[sources/byol]]
updated: 2026-05-27
---

# BYOL（Bootstrap Your Own Latent）

## 概要

**BYOL**（Bootstrap Your Own Latent、自分自身の潜在表現をブートストラップする）は DeepMind の Jean-Bastien Grill, Florian Strub らが 2020 年に発表した自己教師あり画像表現学習フレームワーク。「**負例ペアを使わずに対比学習を上回れる**」ことを初めて説得力ある形で示し、SSL コミュニティのパラダイムシフトを起こした。

- 論文: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
- arXiv: 2006.07733（2020）
- 会議: NeurIPS 2020
- 所属: DeepMind（+ Imperial College）

---

## アーキテクチャ

2 つのネットワーク（オンライン・ターゲット）と非対称設計が核心：

| ネットワーク | 構成 | 更新方法 |
|---|---|---|
| オンライン（θ） | f_θ（エンコーダ）→ g_θ（プロジェクタ）→ **q_θ（プレディクタ）** | 勾配降下 |
| ターゲット（ξ） | f_ξ（エンコーダ）→ g_ξ（プロジェクタ） | EMA: ξ ← τξ + (1-τ)θ |

**プレディクタ** q_θ はオンライン側にだけ存在する追加 MLP。これがターゲット側との非対称性を生み出し、崩壊防止の鍵となる。

---

## 主要な結果

### ImageNet 線形評価 top-1

| アーキテクチャ | Top-1 |
|---|---|
| ResNet-50 (1×) | 74.3% |
| ResNet-50 (2×) | 77.4% |
| ResNet-50 (4×) | 78.6% |
| **ResNet-200 (2×)** | **79.6%** |

比較: SimCLR (ResNet-50 1×) = 69.3%、SimCLR (4×) = 76.5%

### 半教師あり学習（ResNet-50）

| ラベル率 | Top-1 | Top-5 |
|---|---|---|
| 1% | 53.2% | 78.4% |
| 10% | 68.8% | 89.0% |

---

## BYOL が明らかにした知見

1. **負例は必須ではない**: 適切な設計（predictor + EMA target）があれば崩壊なしに強力な表現が学習できる
2. **predictor と EMA target は両方必要**: 一方だけでは崩壊する（Table 5(b) アブレーション）
3. **バッチサイズ依存を解消**: 256〜4096 の範囲でほぼ同等の性能（SimCLR は急落）
4. **拡張頑健性の改善**: 「クロップのみ」で BYOL 59.4%（SimCLR 40.3%）
5. **EMA target を加えるだけで SimCLR も改善**: 単純に target network を SimCLR に加えると +1.6pt（安定化効果）

---

## 後継・発展

| 手法 | BYOL との関係 |
|---|---|
| **SimSiam** (Chen & He, 2021) | BYOL から EMA も除去。stop-gradient + predictor だけで崩壊を防ぐことに成功 |
| **DINO** (Caron et al., 2021, [[entities/dino]]) | cross-entropy + centering + sharpening + EMA で ViT 特化の自己蒸留へ。predictor なし |
| **iBOT** ([[entities/ibot]]) | DINO + MIM（masked image modeling）のハイブリッド |
| **DINOv2** ([[entities/dinov2]]) | iBOT を大規模化した汎用基盤モデル |
| **BYOL-A / BYOL-Audio** | 音声ドメインへの BYOL 拡張 |

---

## 関連ページ

- [[sources/byol]]: 詳細な論文要約
- [[translations/byol]]: 日本語全文翻訳
- [[concepts/self-supervised-learning]]: SSL 系統における BYOL の位置づけ（非対比型）
- [[concepts/contrastive-learning]]: BYOL が「超えた」対比学習パラダイム
- [[entities/simclr]]: BYOL の主要比較対象（対比 SSL の代表）
- [[entities/dino]]: BYOL の発想を発展させた自己蒸留 SSL
