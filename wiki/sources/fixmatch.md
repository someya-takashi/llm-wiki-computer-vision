---
type: source
source_path: raw/papers/FixMatch- Simplifying Semi-Supervised Learning with Consistency and Confidence.pdf
source_kind: paper
title: "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
authors: [Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A. Raffel, Ekin Dogus Cubuk, Alexey Kurakin, Chun-Liang Li]
year: 2020
venue: NeurIPS 2020
ingested: 2026-05-27
tags: [semi-supervised-learning, pseudo-label, consistency-regularization, confidence-threshold, randaugment, ctaugment, weak-strong-augmentation, cifar, svhn, stl-10, imagenet, google-research]
translation: [[translations/fixmatch]]
---

# FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

> 原典: [[translations/fixmatch]] ・ `raw/papers/FixMatch- Simplifying Semi-Supervised Learning with Consistency and Confidence.pdf`
> 著者: Kihyuk Sohn, David Berthelot, Nicholas Carlini, et al.（Google Research）
> 年・会議: 2020 / NeurIPS 2020

## 一言まとめ

「弱い拡張で疑似ラベルを生成し、強い拡張に対してその疑似ラベルを押しつける」というシンプルな 2 ステップと、高信頼度予測（τ=0.95）のみを使う閾値機構の組み合わせだけで、MixMatch を大きく上回る半教師あり学習の性能を達成した論文。CIFAR-10（250 ラベル）で 5.07%（MixMatch: 11.08%）を記録し、「複雑さは敵だ」ことを証明した。

## 背景と問題意識

[[entities/mixmatch]] は 2019 年時点で半教師あり学習の SOTA（State-Of-The-Art, 最先端性能）だったが、その構成は複雑だった。「K 回拡張の予測平均」「温度シャープニング」「ラベルあき・なし横断の MixUp」「Brier スコアによるラベルなし損失」「λ_U の段階的ウォームアップ」といった要素が絡み合い、どれが効いているのかが見えにくかった。

FixMatch は「もっとシンプルにできるはず」という発想から出発する。

> **何が本質だったのか？** 既存手法の共通要素を整理すると、「一貫性正則化（同じ入力に対して予測を安定させる）」と「エントロピー最小化（確信ある予測を促す）」の 2 つが効いていた。FixMatch はこの 2 つだけを最小の形で実装し、他のすべてを削る。

MixMatch との設計比較（論文 Table 1 より）：

| アルゴリズム | ラベルあき拡張 | ラベルなし予測拡張 | 後処理 |
|---|---|---|---|
| Π-Model / Teacher-Student | 弱 | 弱 | なし |
| Mean Teacher | 弱 | 弱 | なし（EMA） |
| UDA | 弱 | **強** | シャープニング |
| MixMatch | 弱 | 弱 | シャープニング |
| ReMixMatch | 弱 | **強** | シャープニング |
| **FixMatch** | **弱** | **強** | **疑似ラベル（閾値）** |

UDA と ReMixMatch も「強い拡張」を使っていたが、FixMatch は「強い拡張」の役割を "**疑似ラベルを当てる問題を難しくすることで、consistency regularization をより効果的にする**" と明確に位置づけた点が新しい。

## 提案手法：弱→強の非対称性

### コアアイデア：2 種類の拡張の分業

FixMatch の中心的な発想は「**弱い拡張 → 疑似ラベル生成**、**強い拡張 → 疑似ラベルで学習**」という役割分担だ。

```
ラベルなし画像 u_b
  │
  ├── 弱い拡張 α(u_b): 水平フリップ + 微小クロップ
  │     ↓
  │   q_b = p_model(y | α(u_b))  ← クリーンな予測
  │     ↓
  │   max(q_b) ≥ τ(=0.95)?
  │     ├── NO  → この画像はスキップ
  │     └── YES → q̂_b = argmax(q_b) として one-hot 疑似ラベルを作成
  │
  └── 強い拡張 A(u_b): RandAugment または CTAugment + Cutout
        ↓
      p_model(y | A(u_b))
        ↓
      H(q̂_b, p_model(y | A(u_b)))  ← クロスエントロピー損失
```

総合損失は：

$$\ell_s = \frac{1}{B}\sum_{b=1}^{B} H(p_b,\, p_m(y \mid \alpha(x_b)))$$

$$\ell_u = \frac{1}{\mu B}\sum_{b=1}^{\mu B} \mathbf{1}\bigl[\max(q_b) \geq \tau\bigr] \cdot H(\hat{q}_b,\, p_m(y \mid \mathcal{A}(u_b)))$$

$$\mathcal{L} = \ell_s + \lambda_u \ell_u$$

- $B$: ラベルあきバッチサイズ（デフォルト 64）
- $\mu$: ラベルなし比率（デフォルト 7、つまりラベルなし = 7B 枚）
- $\lambda_u$: ラベルなし損失の重み（デフォルト 1、ウォームアップなし）

### なぜ「弱→強」が効くのか

直感的には「難問（強拡張）に対して、簡単な問題（弱拡張）で得た答えを教師とする」という構造。

- **弱い拡張**でモデルがまだ正しく分類できるものについてだけ、疑似ラベルを信用する
- そのラベルを「わざと歪めた」入力に対して学習させることで、モデルは拡張不変性を強く獲得する
- MixMatch の「弱→弱」では、2つの予測が同程度に不安定で、学習シグナルが弱い

### 信頼度閾値 τ = 0.95：量より質

MixMatch のシャープニング（T=0.5）は「ソフトに」エントロピーを下げたのに対し、FixMatch は「ハードに」絞る。τ=0.95 を超える予測だけを one-hot 疑似ラベルとして採用し、残りは完全に捨てる。

**確証バイアス（confirmation bias）** という落とし穴がある：閾値が低い（τ=0.25 など）とモデルが間違えている例も疑似ラベルとして採用されてしまい、間違いが自己強化されてループに入る。閾値が高いほどバイアスを防げるが、使える疑似ラベルが減る。τ=0.95 はこのトレードオフのスイートスポット。

論文の τ アブレーション（CIFAR-10 / 250 ラベル）：

| τ | マスク率（使わない割合） | 疑似ラベルの不純度 | エラー率 |
|---|---|---|---|
| 0.25 | 0.00%（全部使用） | 6.39% | 6.40% |
| 0.95 | 98.13%（98% 捨てる） | 3.47% | **4.84%** |
| 0.99 | 92.14% | 2.06% | 5.05% |

τ=0.95 のとき実に **98% の疑似ラベルを捨てている**にもかかわらず最良性能が出る。「2% しか使わない疑似ラベルでも、質が高ければ十分」という強いメッセージだ。

### 自然なカリキュラム学習

閾値機構には副次効果がある。訓練初期は max(q_b) < τ の割合が高く、ほとんどの疑似ラベルが使われない。モデルが成長するにつれて徐々に閾値を超える例が増え、疑似ラベルの利用量が自動的に増加する。これは明示的な「カリキュラム」設定なしに、**難易度を段階的に増やす自然なカリキュラム学習**になっている。

### MixMatch との設計差分

| 設計要素 | MixMatch | FixMatch |
|---|---|---|
| 疑似ラベルの生成 | K=2 拡張の平均 → ソフトラベル（シャープニング） | 弱拡張 1 回 → ハード one-hot（閾値） |
| ラベルなしの拡張 | 弱 (weak) のみ | 弱（疑似ラベル用） + 強（学習用） |
| ラベルなし損失 | Brier スコア（有界 L2） | クロスエントロピー |
| MixUp | ラベルあき・なし横断で使用 | **使わない** |
| λ_u の制御 | 線形ウォームアップ | **即時有効（ウォームアップなし）** |
| シャープニング | あり（T=0.5） | **なし**（代わりに閾値） |

MixMatch がなぜ Brier スコアを使っていたかというと、ソフトラベルに対してクロスエントロピーを使うと、ラベルが少しでも間違っていたとき損失が無限大になるリスクがあるから。FixMatch はハードラベル（one-hot）にしているのでクロスエントロピーでも安全。

## 強い拡張：RandAugment と CTAugment

FixMatch の強さの一端は「強い拡張」の質に依存する。

- **RandAugment**（Cubuk et al., 2019）: 固定された拡張集合から M 個をランダムに選んで適用（magnitude N も指定）。データセット依存の探索不要で使いやすい。
- **CTAugment**（Control Theory Augment, Berthelot et al., 2020 ReMixMatch）: ラベルあきデータを使って各拡張の強度を動的に調整する。制御理論的アプローチで、最適な拡張強度をオンラインで学習。
- **Cutout**: 画像のランダムな矩形領域をゼロでマスクする正則化。RandAugment / CTAugment の後に適用される。

アブレーションでは、RandAugment 単体でも CTAugment 単体でも弱い拡張に比べて大幅に改善し、Cutout との組み合わせが必須。強い拡張が効かないモデルに（間違っていない）疑似ラベルを当てることで、モデルが拡張不変性を強制的に学ぶ。

## 実験結果

### 主要ベンチマーク（エラー率 %、WideResNet-28-2 または 28-8）

| データセット | ラベル数 | RandAugment | CTAugment | 次点手法 |
|---|---|---|---|---|
| CIFAR-10 | 40 | 13.81 | 11.39 | — |
| CIFAR-10 | 250 | 5.07 | 5.07 | ReMixMatch: 5.44 |
| CIFAR-10 | 4000 | 4.26 | 4.31 | MixMatch: 6.24 |
| CIFAR-100 | 400 | 48.85 | 49.95 | ReMixMatch: 44.28 |
| CIFAR-100 | 2500 | 28.29 | 28.64 | ReMixMatch: 27.43 |
| SVHN | 40 | 3.96 | 7.65 | — |
| SVHN | 250 | 2.48 | 2.64 | — |
| STL-10 | 1000 | 7.98 | 5.17 | — |

### ほぼ無ラベル設定（barely supervised）

1 クラスあたり 1 枚（10 クラスなら計 10 ラベル）という極端な設定での CIFAR-10 中央値：**64.28%**（正解率）。完全教師あり（50,000 ラベル）の 93.65% には及ばないが、10 枚だけのラベルでここまで動くことを実証。

### ImageNet（10% ラベル = 128,000 枚）

WideResNet-50-2 で top-1 エラー率 28.54%（UDA 比 2.68 pt 改善）。自然言語でのプレトレーニング知識なしに純粋な SeSL でここまで届く。

## CIFAR-100 での異常現象：FixMatch は ReMixMatch に負ける

FixMatch の主張とは逆説的だが、CIFAR-100 低ラベル条件（400 ラベル）では ReMixMatch（44.28%）が FixMatch（48.85%）を上回る。なぜか？

ReMixMatch が持つ「**分布アライメント（Distribution Alignment, DA）**」という機構が鍵だ。DA は「モデルの周辺予測分布 p_model(y) が訓練データのラベル分布に近づくよう」バイアスをかける。CIFAR-100 のように 100 クラスある場合、ラベルなし損失のみ使っていると特定クラスへの偏りが強くなる。DA はこれを補正する。

実証：FixMatch に DA を追加すると **40.14%**（ReMixMatch の 44.28% より良い）。つまり「FixMatch の設計思想は正しいが、クラス多数設定では DA が必須」。

これは FixMatch の Appendix D で「Augmentation Anchoring」「Distribution Alignment」「他データタイプへの応用」として拡張版が紹介されている。

## 実装上の重要ポイント：重みの減衰

論文のアブレーション（§5.2）で特に強調されているのが **Weight decay（重み減衰、L2 正則化）** の重要性だ。デフォルトの $5 \times 10^{-4}$ から 1 桁小さくした $5 \times 10^{-5}$ にするだけで CIFAR-10 エラー率が約 10 pt 悪化する。

> 「weight decay は一般的に重要とされているが、SeSL では特に大きな影響を持つ」— 論文より

さらに **SGD + Nesterov momentum**（β=0.9）が Adam より有効で、コサイン学習率減衰との組み合わせが安定した収束をもたらす。

## 限界・批判的視点

1. **CIFAR-100 問題**: 前述の通り、クラス数が多い場合は分布アライメントなしでは性能が落ちる。DA は FixMatch の本体に入っておらず、別途実装が必要。
2. **強い拡張への依存**: 弱→強の非対称性は「強い拡張でも正解できること」を要求する。強すぎる拡張（画像の意味が変わるほど）は逆効果。
3. **τ=0.95 の固定**: データセットや初期化によっては最適値が異なる可能性。後続の **FlexMatch**（2021）はクラスごとに動的に閾値を調整することでこの問題を緩和。
4. **画像分類中心**: 検出・セグメンテーションへの直接応用は設計変更が必要。後のピクセルレベル SeSL 研究へのステップ。

## 研究上の位置づけ

FixMatch が CV の SeSL にとって重要なのは、「シンプルな設計でも複雑な設計に勝てる」という実証だけではない。「弱→強の非対称性」という設計原則を明示化したことで、後の研究の出発点になった：

- **FlexMatch** (2021): クラス別動的閾値
- **FreeMatch** (2023): 全体的・局所的の 2 レベル自由閾値
- **SoftMatch** (2023): ハード閾値をソフト重み付けに変える

また、弱→強の非対称性は自己教師あり学習（SSL）における「multi-crop augmentation」（小さい crop → 局所特徴；大きい crop → 大域特徴）や「teacher-student の非対称性」（BYOL, DINO）と思想的に通底する。

## 用語と略称

- **SeSL** = Semi-Supervised Learning（半教師あり学習）。本 wiki では SSL（自己教師あり学習）との混同を避けるためこの略称を使用
- **Pseudo-label（疑似ラベル）** = モデルが推定した予測を訓練ラベル代わりに使うこと
- **Confidence threshold τ（信頼度閾値）** = モデルの最大予測確率がこの値を超えないと疑似ラベルを採用しない（τ=0.95）
- **Confirmation bias（確証バイアス）** = 誤った疑似ラベルでモデルを学習させると、その間違いが自己強化されてループに入る現象
- **RandAugment** = Randomly selected augmentation（自動選択拡張）。事前探索なしに強い拡張を適用
- **CTAugment** = Control Theory Augment。ラベルあきデータを使って拡張強度をオンライン調整
- **Cutout** = 画像の矩形領域をゼロでマスクする正則化
- **Barely supervised** = 1 クラスあたり数枚程度という極端に少ないラベル設定
- **Distribution Alignment（DA）** = モデルの周辺予測分布をラベル分布に合わせるバイアス補正
- **WRN** = Wide ResNet（Wide Residual Network）。深さより幅（チャネル数）を増やした ResNet 亜種
- **SVHN** = Street View House Numbers。Googleのストリートビューから切り出した番地認識データセット
- **STL-10** = Stanford 96×96 画像分類ベンチマーク（ImageNet のサブセットに補足した 10 クラス）
- **UDA** = Unsupervised Data Augmentation。強い拡張（AutoAugment）で一貫性正則化を行う手法
- **MixMatch** = FixMatch の前身。一貫性正則化・シャープニング・MixUp を統合した手法

## 関連ページ

- [[translations/fixmatch]]: 日本語全文翻訳
- [[entities/fixmatch]]: スペックシート・実験数値・アルゴリズム詳細
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像と FixMatch の系譜上の位置づけ
- [[sources/mixmatch]]: 直前の先行手法 MixMatch との詳細比較
- [[entities/mixmatch]]: MixMatch エンティティページ
- [[concepts/self-supervised-learning]]: FixMatch の弱→強非対称性と思想的に近い SSL 手法群（BYOL, DINO, multi-crop）
