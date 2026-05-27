---
type: concept
aliases: [SeSL, 半教師あり学習, Semi-Supervised Learning]
tags: [paradigm, semi-supervised, label-efficiency, consistency-regularization, entropy-minimization, mixup, pseudo-label, foundation-model, peft]
related: [[concepts/self-supervised-learning]], [[concepts/contrastive-learning]], [[concepts/knowledge-distillation]], [[concepts/foundation-model]], [[concepts/parameter-efficient-fine-tuning]]
sources: [[sources/mixmatch]], [[sources/fixmatch]], [[sources/flexmatch]], [[sources/revisiting-ssl-foundation-models]]
updated: 2026-05-28
---

# 半教師あり学習（Semi-Supervised Learning）

## 一言で

**少量のラベルあきデータと大量のラベルなしデータを組み合わせ、ラベルの収集コストを抑えながら高精度な分類器を学習させる枠組み**。ラベルあきデータだけで学習する「教師あり学習」と、ラベルを一切使わない「自己教師あり学習（SSL）」の中間に位置する。

> **用語上の注意**: 半教師あり学習は英語で "semi-supervised learning" であり、文献によっては "SSL" と略称されることがある。しかし当 wiki では **SSL = 自己教師あり学習（Self-Supervised Learning）** という用法を採用している。混同を避けるため本ページでは「半教師あり学習」または "SeSL" と表記する。

## 3 つのパラダイムの違い

| パラダイム | ラベルの使い方 | 代表手法 |
|---|---|---|
| **教師あり学習** | 全データにラベルあき | ResNet, ViT（ImageNet 訓練） |
| **半教師あり学習（SeSL）** | 少量ラベルあき + 大量ラベルなし | Π-Model, Mean Teacher, MixMatch, FixMatch |
| **自己教師あり学習（SSL）** | ラベルなしのみ（事前学習） | SimCLR, BYOL, DINO, MAE |

半教師あり学習は「タスクが固まっていてラベルが高コスト」という状況に最適。自己教師あり学習は「汎用表現を先に作り、後でどんなタスクにも転移する」という発想で異なる。

## 主要なアプローチ

### 1. 一貫性正則化（Consistency Regularization）

「同じ入力に対して、拡張後も同じ予測を返すべき」という制約。モデルの決定境界がデータの高密度領域に入り込まないよう押し出す役割を果たす。

- **Π-Model**（Laine & Aila, 2017）: 2 回の異なる拡張の予測を一致させる（MSE 損失）
- **Mean Teacher**（Tarvainen & Valpola, 2017）: 一方の出力を EMA（指数移動平均）モデルの予測に合わせる。より安定したターゲット
- **VAT**（Virtual Adversarial Training, Miyato et al.）: 出力分布を最も大きく変える方向の摂動に対して一貫性を保つ
- **MixMatch**（Berthelot et al., 2019）: K 回拡張の平均予測をターゲットとして一貫性正則化

Mean Teacher の「EMA ターゲット」という発想は、後に自己教師あり学習の BYOL（EMA teacher + student）や DINO（momentum encoder）に受け継がれた（ただし、目的は「ラベルなしでの表現学習」に変化している）。

### 2. エントロピー最小化（Entropy Minimization）

「分類器の決定境界は、データの密集していない低密度領域を通るべき」という仮定に基づく。ラベルなしデータに対して確信ある（低エントロピーな）予測を出力するよう直接最適化する。

- **Grandvalet & Bengio（2005）**: エントロピー損失の直接最小化
- **Pseudo-Label**（Lee, 2013）: 高信頼の予測をハード one-hot ラベルとして再利用
- **MixMatch**: 温度シャープニング（$T=0.5$）で平均予測のエントロピーを下げる
- **FixMatch**（Sohn et al., 2020）: 信頼度閾値（0.95 以上）を超える予測のみ pseudo-label として使用

### 3. 生成的手法（Generative Methods）

VAE や GAN を使いラベルなしデータの密度を明示的にモデル化する。実装が複雑で、近年は一貫性正則化ベースの手法に性能で劣ることが多い。

### 4. MixUp による正則化

例と例の「間」を補間して決定境界を平滑化する。MixMatch はこれをラベルあき・なし横断で適用。後の UDA, ReMixMatch, FixMatch でも使用される。

## 代表的な半教師あり学習手法の系譜

```
Π-Model (2017)
    ↓ EMA ターゲット
Mean Teacher (2017) ──── VAT (2018)
    ↓ MixUp 統合              ↓
MixMatch (2019) ←────────────┘
    ↓ 拡張強化 + 閾値
ReMixMatch / FixMatch (2020)
    ↓ 大規模化
FlexMatch, FreeMatch, SoftMatch ...
```

## MixMatch のポジション

[[entities/mixmatch]] は2019年時点での半教師あり学習の「総合」手法として、3つのアプローチを1本に統一した。CIFAR-10（250 ラベル）で 11.08%（VAT: 36.03%）という劇的な改善を示し、「少量ラベルでもラベルなし大量データを活用すれば教師あり学習の 1/16 以下のラベルで同等性能が出る」ことを実証した。

## FixMatch のポジション

[[entities/fixmatch]] は2020年に「シンプルさによる勝利」を証明した。「弱い拡張 → 疑似ラベル生成、強い拡張 → 疑似ラベルで学習」という **弱→強の非対称性** と、**信頼度閾値 τ=0.95**（98% の疑似ラベルを捨てて 2% の高品質ラベルだけ使う）の組み合わせだけで、CIFAR-10（250 ラベル）エラー率 5.07%（MixMatch: 11.08%）を達成した。

FixMatch が示した「**確証バイアス（confirmation bias）** を避けるには閾値を高くすべき」という原則は、後継の FlexMatch・FreeMatch・SoftMatch に引き継がれ、現代 SeSL の標準設計指針となっている。

弱→強の非対称性という考え方は、自己教師あり学習（SSL）の multi-crop（局所→大域）や BYOL/DINO の teacher-student 非対称性と思想的に通底する。SeSL と SSL は独立して発展しているように見えるが、データ拡張による不変性の学習という点で同じ根を持つ。

## FlexMatch のポジション

[[entities/flexmatch]]（2021）は「全クラスに同じ閾値を使うべきか？」という FixMatch の設計前提を問い直した。**CPL（Curriculum Pseudo Labeling）** というクラス別動的閾値機構を導入し、「学習が難しいクラスの閾値を下げてデータを活用しやすくし、十分学習されたクラスの閾値を上げて品質を確保する」という**カリキュラム学習**の発想を SeSL に持ち込んだ。

CIFAR-10 40 ラベルで 4.97%（FixMatch: 7.47%）、CIFAR-100 400 ラベルで 39.94%（FixMatch: 46.42%）を達成し、収束速度は FixMatch の **1/5**。ただし SVHN（クラス不均衡データセット）では逆に悪化（8.19% vs 3.81%）—— CPL は「クラス別サンプル数がほぼ均等」という仮定が崩れると裏目に出る。

FlexMatch が確立した「**固定閾値 → クラス別適応閾値**」という転換点は、後の FreeMatch（2 レベル自由閾値）・SoftMatch（ガウス重み付け）に直接引き継がれ、2022 年以降の SeSL 論文では「どのように閾値を適応させるか」が主要な設計軸の一つとなった。

## 2025 年の転換点：VFM 時代の SeSL の再検討

[[sources/revisiting-ssl-foundation-models]]（NeurIPS 2025）は、Wide ResNet を**スクラッチから訓練する**前提で設計されてきた既存 SeSL 手法（MixMatch/FixMatch/FlexMatch/SoftMatch）を、**Vision Foundation Models（VFM, CLIP/DINOv2 等）の時代**で再評価し、衝撃的な発見を報告した：

### 発見 1: VFM 上では Labeled-only PEFT が SSL を凌駕

**[[concepts/parameter-efficient-fine-tuning]]（PEFT, LoRA/AdaptFormer 等）でラベルあきデータのみを使って fine-tune する**だけで、ラベルなしを使う FixMatch/FlexMatch/SoftMatch を凌駕する。DINOv2 上では既存 SSL がしばしば**Labeled-only より悪化**する。

> **解釈**: VFM は膨大な事前学習で既に強力な汎化能力を獲得しているので、SSL 手法が導入する「ノイズの多い教師信号」が VFM の汎化能力を**むしろ損なう**可能性がある。

### 発見 2: PEFT 済みモデルの疑似ラベルは多様

異なる **(VFM, PEFT) ペア**——CLIP×LoRA、CLIP×AdaptFormer、DINOv2×LoRA、DINOv2×AdaptFormer——は同じデータに対しても**多様な予測**を生む。全体精度は似ているが、サンプル単位の誤りパターンは補完的。

### 提案手法 V-PET

この多様性をアンサンブルする [[entities/v-pet]]（VFM-PEFT Ensemble Training）は、すべての疑似ラベルを **one-hot 化して平均（Mean Labels）**し、**閾値を使わず（τ=0）**、**1 ラウンド**の self-training で完了する。FixMatch/FlexMatch/SoftMatch/FineSSL を 12 設定全体で凌駕した。

### 系譜上の位置づけ

```
MixMatch (2019) ─→ FixMatch (2020) ─→ FlexMatch (2021) ─→ ... ─→ V-PET (2025)
[スクラッチ + 一貫性正則化]              [スクラッチ + 動的閾値]    [VFM + PEFT + アンサンブル]
```

「**スクラッチ前提 → 基盤モデル前提**」というパラダイムシフトを SeSL でも体現した重要研究。今後の SeSL 研究は VFM × PEFT × アンサンブルという 3 軸の設計空間で探索される可能性が高い。

## 半教師あり学習 vs 自己教師あり学習（現代の視点）

2020 年以降、自己教師あり学習（SSL）が成熟するにつれ、両者の関係が変化した：

| 観点 | 半教師あり学習（SeSL） | 自己教師あり学習（SSL） |
|---|---|---|
| **目的** | 特定タスクをラベル効率よく解く | 汎用表現を作り複数タスクに転移する |
| **ラベルなしの役割** | 汎化を助ける補助データ | 学習の主信号 |
| **現在の主流用途** | 医療画像、希少クラス認識 | 大規模視覚基盤モデルの事前学習 |
| **ラベルの必要時期** | 訓練中（少量） | 訓練中は不要（fine-tune 時に少量） |

現代の実用的な高精度システムは「SSL 事前学習 → 少量ラベルでファインチューン」という組み合わせが主流であり、これは半教師あり学習と自己教師あり学習の合わせ技とも言える。

## 参考

- [[sources/mixmatch]]: MixMatch 論文詳細（NeurIPS 2019）— 半教師あり学習の 3 アプローチ統合
- [[sources/fixmatch]]: FixMatch 論文詳細（NeurIPS 2020）— 弱→強非対称性と閾値機構
- [[sources/flexmatch]]: FlexMatch 論文詳細（NeurIPS 2021）— CPL クラス別動的閾値・SVHN 失敗事例の考察
- [[sources/revisiting-ssl-foundation-models]]: NeurIPS 2025 — VFM 時代における既存 SSL の再評価、V-PET 提案
- [[entities/mixmatch]]: MixMatch エンティティページ
- [[entities/fixmatch]]: FixMatch エンティティページ
- [[entities/flexmatch]]: FlexMatch エンティティページ
- [[entities/v-pet]]: V-PET エンティティページ（VFM × PEFT アンサンブル）
- [[concepts/parameter-efficient-fine-tuning]]: PEFT 概念（VFM × SeSL の中心ツール）
- [[concepts/foundation-model]]: 基盤モデル全般。VFM × SeSL の関係
- [[concepts/self-supervised-learning]]: ラベルなし事前学習の系譜（SimCLR/BYOL/DINO/MAE）
- [[concepts/contrastive-learning]]: 半教師あり学習の一貫性正則化と思想的に近い
- [[concepts/knowledge-distillation]]: Mean Teacher / BYOL の EMA ターゲットが使う考え方
