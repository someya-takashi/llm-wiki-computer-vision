---
type: source
source_path: raw/papers/Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.md
source_kind: paper
title: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
authors: [Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko]
year: 2020
venue: NeurIPS 2020
ingested: 2026-05-27
tags: [ssl, self-supervised-learning, non-contrastive, momentum-encoder, representation-learning, resnet, imagenet, byol, predictor, ema]
translation: [[translations/byol]]
---

# Bootstrap Your Own Latent（BYOL）

> 原典: [[translations/byol]] ・ `raw/papers/Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.md`
> 著者: Jean-Bastien Grill et al.（DeepMind）・ arXiv: 2006.07733 ・ NeurIPS 2020

## 一言まとめ

**「負例なしで対比学習を超えた」**最初の説得力ある実証。オンラインネットワークとターゲットネットワーク（EMA 更新）の 2 ネットワーク構造と、オンライン側だけに付いた追加 MLP（プレディクタ）の組み合わせで崩壊を防ぎ、ResNet-50 で ImageNet 74.3%（SimCLR の 69.3% を 5 ポイント超過）を達成した。

## 背景と問題意識

2020 年頃の自己教師あり学習（SSL）の主流は [[concepts/contrastive-learning]]（対比学習）、特に [[sources/simclr]]（SimCLR）や [[entities/moco]]（MoCo）だった。対比学習の本質は「**正例ペアを近づけ、負例ペアを遠ざける**」ことで自明解（崩壊）を防ぐ設計にある。しかしこの構造は必然的に以下の問題をはらんでいた：

- **大きなバッチサイズが必要**: SimCLR はバッチ内の他サンプルを負例として使うため、バッチが小さいと負例が貧困になり性能が落ちる
- **拡張選択への脆弱性**: 色歪みを取り除くだけで大幅に性能低下（SimCLR: -22 ポイント）。色ヒストグラムで近似できてしまうショートカットが生じるため
- **メモリバンクや特殊設計**: MoCo はメモリバンクと momentum encoder を要し、実装が複雑

BYOL の問い：**「負例ペアは本当に必要なのか？」** DeepMind チームはこの問いに「必要ない」という答えを実験で示した。

## アーキテクチャ：オンライン × ターゲット の 2 ネットワーク構造

<figure>

![](../../raw/assets/byol/fig2.png)

<figcaption>図2（再掲）: BYOL のアーキテクチャ。オンライン（上段）とターゲット（下段）の 2 ネットワーク構造。オンラインにだけ predictor q_θ が付く。ターゲットへは勾配が流れない（stop-gradient）。</figcaption>
</figure>

```
画像 x
├── 拡張 v  → オンライン: f_θ → y_θ → g_θ → z_θ → q_θ → 予測
└── 拡張 v' → ターゲット: f_ξ → y'_ξ → g_ξ → z'_ξ ← ターゲット（stop-grad）

損失: MSE(ℓ₂-正規化(q_θ(z_θ)), ℓ₂-正規化(z'_ξ))
（+ v と v' を入れ替えた対称損失）

パラメータ更新:
  θ ← optimizer(θ, ∇_θ L^BYOL)        ← 通常の勾配降下
  ξ ← τ·ξ + (1-τ)·θ                  ← EMA（τ ≈ 0.996〜0.999）
```

| ネットワーク | コンポーネント | 訓練後の扱い |
|---|---|---|
| オンライン | f_θ（エンコーダ）+ g_θ（プロジェクタ）+ **q_θ（プレディクタ）** | f_θ のみ残す |
| ターゲット | f_ξ（エンコーダ）+ g_ξ（プロジェクタ） | 全て破棄 |

**プレディクタ（predictor）** は BYOL の最大の発明の一つ。SimCLR や MoCo にはこの層は存在しない。オンライン側にだけ付けることでアーキテクチャを非対称にし、崩壊回避の鍵となる。

### なぜ崩壊しないか

BYOL が崩壊しないのは**プレディクタ × EMA ターゲットの組み合わせ**が必須であることがアブレーションで示されている：

| 設定 | 結果 |
|---|---|
| BYOL（predictor + EMA target、β=0） | **72.5%** |
| predictor のみ（target=online、β=0） | 0.3%（崩壊） |
| EMA target のみ（predictor なし、β=0） | 0.2%（崩壊） |
| predictor + EMA target + 負例（β=1） | 70.9%（逆に悪化） |

直観的解釈：プレディクタ $q_\theta$ が「最適」に近い状態に保たれると、BYOL のオンライン更新は **条件付き分散** $\mathbb{E}[\sum_i \text{Var}(z'_{\xi,i} | z_\theta)]$ の最小化に対応する。定数な $z_\theta$（= 崩壊）では $\text{Var}(z'_\xi | z_\theta)$ が最小化できないため、崩壊均衡は不安定になる。EMA ターゲットの役割は「プレディクタを常に最適に近い状態に保つこと」。

> **Mean Teacher との比較**: BYOL からプレディクタを取り除くと、分類損失なしの Mean Teacher（半教師あり学習手法）の教師なし版と等価になる。これは崩壊する（表 5 の 7 行目）。つまり predictor が教師なし設定での崩壊を防ぐ固有の役割を担っている。

## 実装の詳細

### ネットワーク構成

| ハイパーパラメータ | 値 |
|---|---|
| エンコーダ | ResNet-50 (1×〜4×) または ResNet-101/152/200 |
| プロジェクタ g | Linear(2048→4096) + BN + ReLU + Linear(4096→256) |
| プレディクタ q | Linear(256→4096) + BN + ReLU + Linear(4096→256) |
| 損失 | 対称 MSE（ℓ₂ 正規化後） |
| τ_base | 0.996（訓練に沿って 1 まで増加） |

SimCLR との大きな違い：
- プロジェクタの出力はバッチ正規化**しない**（SimCLR はする）
- プレディクタ q が追加される

### 訓練設定

- **オプティマイザ**: LARS（Large-scale Learning Rate Scaling）+ コサイン減衰 LR
- **エポック数**: 1000 epoch（アブレーションは 300 epoch）
- **バッチサイズ**: 4096（512 TPU v3 コア）
- **学習率**: 0.2 × BatchSize/256 → 約 3.2（base LR × linear scaling）
- **訓練時間**: ResNet-50 (1×) で約 8 時間

## 主要な実験結果

### ImageNet 線形評価（top-1 / top-5）

| モデル | Top-1 | Top-5 | 比較 |
|---|---|---|---|
| BYOL (ResNet-50 1×) | **74.3%** | 91.6% | SimCLR +5.0pt |
| BYOL (ResNet-50 4×) | 78.6% | 94.2% | SimCLR (4×) +2.1pt |
| BYOL (ResNet-200 2×) | **79.6%** | 94.8% | 250M params |

### 半教師あり学習（ImageNet、ResNet-50）

| ラベル率 | Top-1 | Top-5 |
|---|---|---|
| 1% | 53.2%（SimCLR: 48.3%） | 78.4%（SimCLR: 75.5%） |
| 10% | 68.8%（SimCLR: 65.6%） | 89.0%（SimCLR: 87.8%） |

### 転移学習（線形評価、ResNet-50）

12 データセット中 12 で SimCLR を上回る（Food101, CIFAR10, CIFAR100, Birdsnap, SUN397, Cars, Aircraft, VOC2007, DTD, Pets, Caltech-101, Flowers）。12 中 7 で Supervised-IN ベースラインも上回る。

### 他タスクへの転移（ResNet-50）

| タスク | BYOL | SimCLR repro | Supervised-IN |
|---|---|---|---|
| 物体検出 AP50 | **77.5** | 75.2 | 74.4 |
| 意味的セグメンテーション mIoU | **76.3** | 75.2 | 74.4 |
| 深度推定 pct<1.25 | **84.6%** | 83.3% | 81.1% |

## アブレーションから見えた知見

### バッチサイズ頑健性

<figure>

![](../../raw/assets/byol/fig3a.png)

<figcaption>図3（a）（再掲）: バッチサイズ vs. 性能。BYOL は 256〜4096 でほぼ水平、SimCLR は 256 以下で急落する。</figcaption>
</figure>

| バッチサイズ | BYOL top-1 | SimCLR top-1 |
|---|---|---|
| 4096 | **72.5%** | 67.9% |
| 512 | 72.2% | 66.5% |
| 256 | 71.8% | 64.3%±2.1 |
| 128 | 69.6%±0.5 | 63.6% |
| 64 | 59.7%±1.5 | 59.2%±2.9 |

バッチサイズ 64 での BYOL の性能低下は BN（バッチ正規化）の挙動が小バッチで不安定になるため。

### 拡張頑健性

| 拡張設定 | BYOL top-1 | SimCLR top-1 |
|---|---|---|
| ベースライン | **72.5%** | 67.9% |
| クロップのみ | 59.4% | 40.3% |
| カラー（ジッタ+グレースケール）除去 | 63.4% | 45.7% |
| ブラー除去 | 71.2% | 65.2% |

カラー拡張がない場合でも BYOL は一定の性能を維持する。これは BYOL が「ターゲット表現がキャプチャした情報を保持するインセンティブ」を持つため、カラーヒストグラムショートカットに依存しないから。

### EMA 減衰率

| τ_base | Top-1 |
|---|---|
| 1（固定ランダム） | 18.8%±0.7 |
| 0.999 | 69.8% |
| **0.99** | **72.5%** |
| 0.9 | 68.4% |
| 0（直接コピー） | 0.3%（崩壊） |

τ=1（ランダム固定ネットワーク）でも 18.8% と無意味でない結果が出る。これが BYOL の核心的動機：「固定ランダム表現を改善できるならば、自分自身の改善した表現を追いかける繰り返しによってどんどん良くなれる」という観察。

## 限界・批判的視点

1. **拡張依存**: 依然として視覚特化の拡張（ランダムクロップ、カラー歪み等）に依存。音声・テキストへの一般化には新たな拡張設計が必要
2. **BN 依存**: バッチサイズ 64 以下での性能低下は BN の挙動に起因。後継の SimSiam ではこの問題が扱われる
3. **なぜ崩壊しないかの完全な理論的説明がない**: 著者自身が「GAN と同様に共同最適化の視点では説明できない」と認めており、プレディクタ最適性の仮定を置いた間接的な論法に依拠している。後に BatchNorm の implicit normalization が重要との主張も出た
4. **大規模計算資源**: 512 TPU v3 コアで学習。アカデミア再現は困難
5. **密予測への弱さ（相対的）**: 画像レベル特徴は強いが、パッチレベルの密予測はこの時点では MIM 系（MAE, iBOT）に比べて設計上不利
6. **後継の SimSiam はさらにシンプル化**: Chen & He (2021) は EMA ターゲットも取り除き、stop-gradient + predictor だけで動かすことに成功

## SimCLR との詳細比較

| 観点 | SimCLR | BYOL |
|---|---|---|
| 崩壊防止 | 負例ペア（InfoNCE 損失） | predictor + EMA ターゲット |
| バッチサイズ依存 | 高い（負例数がバッチに依存） | 低い（256〜4096 で安定） |
| 拡張依存 | 高い（カラー歪み必須） | 中（カラーなしでも -9pt） |
| 余分なパラメータ | なし | predictor（小） |
| 推論時コスト | f のみ | f のみ（同じ） |
| ResNet-50 top-1 | 69.3% | 74.3% |

## 用語と略称

| 用語 | 説明 |
|---|---|
| BYOL | Bootstrap Your Own Latent。自分自身の潜在表現をブートストラップするという命名 |
| オンラインネットワーク | 勾配降下で直接更新されるネットワーク（θ）。f_θ + g_θ + q_θ |
| ターゲットネットワーク | オンラインの EMA で間接更新されるネットワーク（ξ）。f_ξ + g_ξ のみ |
| プレディクタ（predictor） | オンラインにだけ付く追加 MLP。BYOL 最大の設計上の特徴 |
| プロジェクタ（projector） | エンコーダ出力 y を z に変換する MLP。訓練後は破棄 |
| EMA（指数移動平均） | Exponential Moving Average。ξ ← τ·ξ + (1-τ)·θ |
| stop-gradient | 勾配をターゲット側に流さない操作。ターゲットは θ の勾配から独立 |
| LARS | Layer-wise Adaptive Rate Scaling。大バッチ訓練の安定化オプティマイザ（SimCLR も使用） |
| τ（タウ） | EMA 減衰率。0.99〜0.999 が最良。訓練中に 1 に向けて増加 |
| ブートストラッピング | 以前の出力を次の学習ターゲットとして使う手続き。強化学習（DQN）から着想 |
| 崩壊（collapse） | 全サンプルに同じ表現を出力する自明解。BYOL ではなぜ起きないか理論的に完全には説明できていない |
| Mean Teacher（MT） | 半教師あり学習手法。BYOL からプレディクタを除いて分類損失なしにすると MT の教師なし版となり崩壊する |

## 関連ページ

- [[translations/byol]]: 全文和訳
- [[entities/byol]]: BYOL モデルエンティティページ
- [[sources/simclr]]: BYOL が直接比較している対比学習のベースライン
- [[concepts/self-supervised-learning]]: SSL 全体の系統（BYOL = 非対比型に位置づけ）
- [[concepts/contrastive-learning]]: 対比学習パラダイム（BYOL はこれを「負例なしで超える」）
- [[concepts/knowledge-distillation]]: BYOL は自己蒸留として再解釈できる
- [[entities/simclr]]: SimCLR エンティティページ
- [[entities/dino]]: BYOL の後継として発展した自己蒸留 SSL の代表
