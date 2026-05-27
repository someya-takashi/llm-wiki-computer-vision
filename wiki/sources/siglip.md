---
type: source
source_path: raw/papers/Sigmoid Loss for Language Image Pre-Training.pdf
source_kind: paper
title: "Sigmoid Loss for Language Image Pre-Training"
authors: [Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer]
year: 2023
venue: ICCV 2023
ingested: 2026-05-26
tags: [siglip, contrastive-learning, weakly-supervised, vision-language, foundation-model, google-deepmind, sigmoid-loss, efficient-training]
translation: [[translations/siglip]]
---

# SigLIP: Sigmoid Loss for Language Image Pre-Training

> 原典: [[translations/siglip]] ・ `raw/papers/Sigmoid Loss for Language Image Pre-Training.pdf`
> 著者: Zhai, Mustafa, Kolesnikov, Beyer（Google DeepMind, Zürich）
> 発表: arXiv:2303.15343（2023 年 3 月）→ ICCV 2023
> コード: <https://github.com/google-research/big_vision>

## 一言まとめ

**CLIP の softmax 対比学習を sigmoid 損失に置き換えるだけ** で、(1) 小バッチで圧倒的に勝つ、(2) 大バッチでも僅かに勝つ、(3) メモリ効率が劇的に良い、という三拍子そろった改善を実現。さらに **バッチサイズが 32k で飽和する**（CLIP 以来「対比学習は大バッチが命」とされた常識を覆す）、**学習可能 bias term + 適切初期化**（b=-10, t'=log10）が決定的に重要、**β₂=0.95 で大バッチ訓練を安定化**、などの実用知見を提供。**4 TPU で 1 日訓練して 79.7% ImageNet 0-shot**（SigLiT）、**SO/400M で 83.2%**（SigLIP）。

## 背景と問題意識

### CLIP の softmax 対比学習の不便さ

CLIP（[[sources/clip]] / [[entities/clip]]）と ALIGN が確立したパターン：

1. 画像とテキストをそれぞれエンコーダで埋め込み
2. バッチ内の **全画像 × 全テキスト** の類似度行列を作成
3. **softmax で正規化**（画像→テキスト方向 + テキスト→画像方向の 2 回）
4. 対角を正例として cross-entropy 損失（**InfoNCE**, [[concepts/contrastive-learning]]）

この標準レシピには 3 つの厄介な問題があった：

1. **メモリ集約**: $|\mathcal{B}|^2$ の類似度行列をメモリに実体化する必要 → バッチサイズ上限
2. **数値不安定**: softmax のナイブ実装は不安定、最大値を引いて安定化する必要（もう 1 度バッチ全体パス）
3. **タスク定義とバッチサイズの結合**: 「正しい 1 個を見分ける」というタスク自体がバッチ構成に依存（小バッチでは選択肢が少なすぎる、大バッチでは膨大な candidate のうち 1 つ）

これらが原因で CLIP は 256+ TPU で何日もかけて訓練する必要があり、研究の参入障壁が高かった。

### 既存の効率化アプローチの限界

- **LiT**（Zhai 2022）: 事前訓練 frozen 画像 backbone を使ってテキストのみ訓練 → 強い backbone が必要
- **FLIP**（Li 2023）: 視覚トークンをランダムドロップで効率化 → 品質を犠牲
- **BASIC, LAION-2B/5B**: バッチサイズを 16k/160k までスケール → 何百チップも必要

「**もっと根本的に効率化できないか**」が SigLIP の出発点。

## 提案手法：Sigmoid 損失

### 数式比較

**Softmax（CLIP）**:

$$-\frac{1}{2|\mathcal{B}|} \sum_i \left( \log \frac{e^{t \mathbf{x}_i \cdot \mathbf{y}_i}}{\sum_j e^{t \mathbf{x}_i \cdot \mathbf{y}_j}} + \log \frac{e^{t \mathbf{x}_i \cdot \mathbf{y}_i}}{\sum_j e^{t \mathbf{x}_j \cdot \mathbf{y}_i}} \right)$$

→ 各 $i$ について **全バッチ横断で正規化** が必要、2 方向

**Sigmoid（SigLIP）**:

$$-\frac{1}{|\mathcal{B}|} \sum_i \sum_j \log \frac{1}{1 + e^{z_{ij}(-t \mathbf{x}_i \cdot \mathbf{y}_j + b)}}$$

ここで $z_{ij} \in \{+1, -1\}$（対角が +1、それ以外 -1）

→ 各 (i, j) ペアが **独立** に二値分類問題に変わる。グローバル正規化不要。

### 何が嬉しいか

| 観点 | Softmax (CLIP) | **Sigmoid (SigLIP)** |
|---|---|---|
| **タスクの解釈** | 「正しいペアを選ぶ」（バッチ依存） | 「このペアは正しいか？」（**ペア独立**） |
| **数値安定化** | 最大値を引くため追加パス | 不要、`log_sigmoid` で安定 |
| **メモリ** | $\mathcal{O}(\|\mathcal{B}\|^2)$ 行列を実体化 | **chunked 実装で $\mathcal{O}(b^2)$** のみ |
| **対称性** | 非対称（2 方向別に正規化） | **対称的** |
| **小バッチ性能** | 弱い | **大幅優位** |
| **大バッチ性能** | 同等 | 僅かに優位 |
| **ノイズ頑健性** | 弱い | **強い**（§4.10） |

### 学習可能 bias term（隠れた重要発明）

シグモイド損失をそのまま使うと、初期化時に **負例の重い不均衡**（バッチサイズ 16k なら正例 16k vs 負例 268M）が損失を支配し、大きな初期最適化ステップでこのバイアスを修正しようとして訓練が不安定になる。

解決策：温度 $t$ と類似の **学習可能 bias term $b$** を導入し、$b = -10$、$t' = \log 10$ で初期化。

**これにより訓練が事前分布に近く始まり、大規模な過剰修正を回避**。

アブレーション（表 4）で：
- bias なし: ImageNet 0-shot 62.0
- **b=-10, t'=log 10**（推奨）: **63.0**
- b=0, t'=log 10: 61.7
- b=0, t'=log 1: 53.7（崩壊）

**bias term の有無で 10 ポイント以上の差**が出る場合がある。SigLIP を実装する際は **絶対に忘れてはいけない設計**。

### Chunked Efficient 実装（メモリ削減の核心）

データ並列で $D$ デバイスに分散させたとき、シグモイド損失は次のように分解できる：

1. **各デバイス** が自分のバッチ $b$ で正例 + ローカル負例 ($b-1$ 個) の損失を計算（per-device 損失）
2. **テキスト表現を隣のデバイスに collective_permute** → 各デバイスが次の負例セットを取得
3. これを $D$ 回繰り返して、すべてのペア組み合わせをカバー
4. 最後に全デバイスの損失を合計

これにより **各デバイスの最大メモリは $b^2$**（$|\mathcal{B}|^2$ ではない）に削減。

> **補足: collective_permute と all-gather の比較** — softmax では全埋め込みを集める all-gather が必要だが、sigmoid は permute だけで済む。$D$ 回の collective_permute は 2 回の all-gather より高速（典型的に）。これにより比較的少数のデバイスで **100 万バッチサイズ訓練に成功**（§4.1）。

## 主要結果

### バッチサイズ研究（最大の貢献）

**§4.1 図 2 左**: SigLiT setup (B/8, frozen ViT-g 画像)、ImageNet 0-shot

| Batch Size | Softmax | **Sigmoid** | 差 |
|---|---|---|---|
| 2k | ~81 | ~82 | **+1** |
| 8k | ~82.5 | ~83.5 | **+1** |
| **16k** | ~83.5 | ~84 | **+0.5** |
| 32k | ~84 | ~84 | **0** |
| 262k | ~84.5 | ~84.5 | ~0 |
| 1M | ~84.5 | ~84.5 | ~0（飽和） |

→ **小バッチで決定的に sigmoid が勝つ**、大バッチで両者飽和。

### 驚き発見: 32k で飽和

CLIP 時代以来「対比学習は大バッチが正義」とされ、研究では 64k, 160k と上っていたが：

- **SigLiT も SigLIP も 32k で飽和**
- それ以上拡大しても僅かなブースト
- 1M バッチ（SigLiT の極限実験）でも 256k より良くない
- **mSigLIP（多言語）でも 32k で十分**

→ **「対比学習を大バッチで訓練するために巨大インフラを買う必要はもうない」** という重要な発見。研究のアクセス可能性を根本的に変える。

### 4 TPU での驚異的効率

**SigLiT**（B/8 frozen, LiT データ）:
- **1 日 4 TPUv4 で 79.7% IN-0**
- 2 日で **84.5% IN-0**（g/14, LiT より僅かに低い 85.2% に近い）

**SigLIP**（B/16 from-scratch, WebLI）:
- 32 TPUv4 × 2 日で **72.1% IN-0**
- 32 TPUv4 × 5 日で **73.4% IN-0**
- 比較: CLIP は 256 TPUv3 × 10 日で 72.6%

→ **桁違いの効率改善**。

### スケールアップ（表 3）

ImageNet 0-shot で SigLIP は CLIP/OpenCLIP/EVA-CLIP を凌駕：

| Model | ViT size | # Patches | IN-1k Val |
|---|---|---|---|
| CLIP | B | 196 | 68.3 |
| OpenCLIP | B | 196 | 70.2 |
| EVA-CLIP | B | 196 | 74.7 |
| **SigLIP** | **B** | **196** | **76.2** |
| **SigLIP** | **B** | **1024** | **79.2** |
| **SigLIP** | **L** | **576** | **82.1** |
| OpenCLIP | G (2B) | 256 | 80.1 |
| EVA-CLIP | E (5B) | 256 | 82.0 |
| **SigLIP** | **SO (400M)** | **729** | **83.2** ★ |

**Shape-Optimized 400M ViT が 5B EVA-CLIP より高精度** — モデルアーキテクチャ最適化の威力を実証。

### mSigLIP（多言語版）

- 100 言語の WebLI で訓練
- **XM3600 text-to-image retrieval で 34.9%**（先行 LiT の 28.5% を 6 ポイント超え）
- ボトルネック化トークン埋め込み（$N \times K \to K \times W$）で大語彙のメモリ節約
- 大規模 mSigLIP ViT-B でも **42.6% IR recall@1 + 54.1% TR recall@1** で SOTA

### ノイズ頑健性（§4.10）

画像/テキスト/バッチアラインメントを意図的に破損させた実験で、sigmoid 訓練は softmax より一貫してノイズに頑健（図 7）。Web crawl の noisy データに対する重要な実用利点。

### Bias term の決定的重要性（§4.9）

| b | t' | INet-0 |
|---|---|---|
| n/a | log 10 | 62.0 |
| **-10** | **log 10** | **63.0** ★ |
| 0 | log 1 | 53.7（崩壊） |

→ **b=-10, t'=log 10 が SigLIP の標準** として確立。

### 大バッチ訓練の安定化（§4.7）

- 大バッチで勾配ノルムのスパイクが発生
- **Adam/AdaFactor の β₂ を 0.999 → 0.95 に下げる** だけで安定化
- これは [Lion 論文] [GIT 論文] でも独立に提案された

→ **β₂=0.95 が SigLIP（と大バッチ vision transformer 訓練全般）の標準** に。

### Negative ratio 研究（§4.8）

- バッチサイズ 16k で正例 16k vs 負例 268M = 1:16k の極端な不均衡
- マスキング実験：
  - **Random masking**: 性能劣化（多様性損失）
  - **Hardest negatives 保持**: ほぼ品質維持
  - **Easiest negatives 保持**: 完全崩壊
- 結論: **不均衡は問題でない、hard negative mining はある程度効くが trivial でない**

CLIP/SigLIP では **特別な negative mining なし** で十分動作することが確認された。

## なぜ重要か

### CV foundation model の系譜での位置

**WSL（弱教師あり事前学習）系の進化**：

| 年 | モデル | 主要貢献 |
|---|---|---|
| 2021 | **CLIP**（OpenAI） | 4 億 web 画像-テキスト対 + softmax 対比学習で zero-shot 分類革命 |
| 2021 | **ALIGN**（Google） | 18 億 noisy ペア、CLIP より大規模 |
| 2022 | **LiT**（Google） | 事前訓練 frozen 画像 backbone + テキスト訓練で効率化 |
| 2022 | **OpenCLIP**（LAION） | CLIP の公開再現、LAION-2B/5B |
| 2022 | **BASIC, CoCa, FLIP** | 各種効率化、生成的事前学習 |
| **2023** | **SigLIP**（Google DeepMind） | **Sigmoid 損失で根本的効率化、bias term、batch size 飽和の発見** |
| 2024 | **SigLIP 2** | SigLIP の改良版（multi-resolution, dense prediction 強化等） |
| 2024 | **EVA-CLIP, MetaCLIP, DFN** | データキュレーション + masking で更なる効率化 |
| 2024 | **Perception Encoder**（Meta） | 86B 対で SigLIP 系の極限 |

SigLIP は **CLIP 以来最大のレシピ革新**として位置づけられる：

1. **Sigmoid 損失** を標準化（その後 SAM 3 の hard negative 訓練、PE 等で採用）
2. **大バッチ依存を解消**、研究のアクセス可能性を根本的に改善
3. **bias term, β₂=0.95, chunked impl** など実装テクニックを確立
4. **ノイズ頑健性** を WSL の標準的価値に押し上げる

### Practical Impact

- **DINOv3 のベンチマーク**（[[sources/dinov3]]）で頻繁な比較対象、SSL vs WSL の主要競合
- **SAM 3**（[[sources/sam-3]]）の比較対象、特に open-vocabulary 検出で
- **VLM の vision tower 候補**: CLIP/SigLIP/SigLIP 2/PE/DINOv2 の中で実用的に重要
- **コミュニティへの貢献**: `big_vision` コードベース公開、Apache 2.0 モデル公開

## 限界

1. **Web データ依存**: WebLI（Google 内部、非公開）で訓練、公開版は OpenSigLIP まで待つ必要
2. **dense prediction が弱い傾向**: DINOv3 論文で指摘されるように、密予測（segmentation, depth）では純粋 SSL に劣る場面あり（SigLIP 2 で改善）
3. **テキストは英語中心**: mSigLIP で改善するが、cross-modal retrieval にとどまる
4. **モデルアーキテクチャ**: 標準 ViT、別ベース（CNN, Hiera 等）の探索なし
5. **画像-テキスト対以外の signal なし**: マルチモーダル（音声、3D 等）への拡張は別研究

## 用語と略称

- **SigLIP** = **Sig**moid **L**oss for Language-**I**mage **P**re-training
- **SigLiT** = SigLIP の LiT 版（**Lock**ed-image **T**uning、frozen 画像 backbone + テキスト訓練）
- **mSigLIP** = multilingual SigLIP（100 言語）
- **WebLI** = Google 内部の Web 由来画像-テキストペアデータセット（PaLI で初出）
- **XM3600** = Crossmodal-3600、36 言語の多言語画像-テキスト検索ベンチマーク
- **SO-400M** = Shape-Optimized 400M parameter ViT（Alabdulmohsin 2023）
- **InfoNCE** = Information Noise Contrastive Estimation（softmax 対比学習の標準損失）
- **InfoNCE vs sigmoid loss** = 本論文の核心対比
- **chunked implementation** = メモリ効率的な分散実装（per-device $b^2$ メモリ）
- **bias term $b$, temperature $t'$** = SigLIP の重要なハイパーパラメータ（推奨初期化 $b=-10$, $t'=\log 10$）
- **β₂ = 0.95** = Adam/AdaFactor のモメンタム係数、大バッチ安定化の鍵
- **big_vision** = Google の公開コードベース
- **ScalingViT-Adafactor** = 大規模 ViT 訓練向け Adafactor 派生
- **Lion optimizer** = symbolic discovery で発見された効率的オプティマイザ（[12]）
- **CapFilt** = BLIP の caption filter 手法
- **EVA-CLIP / CLIPA-v2** = CLIP の効率改善派生

## 関連ページ

- [[translations/siglip]] — 全文和訳（本文 §1-5）
- [[entities/siglip]] — SigLIP モデル群のスペック
- [[sources/clip]] / [[entities/clip]] — 対比される元（softmax 対比学習）
- [[concepts/contrastive-learning]] — sigmoid 損失も対比学習の一形態
- [[concepts/weakly-supervised-pretraining]] — SigLIP が属するパラダイム
- [[concepts/zero-shot-transfer]] — SigLIP の主要評価軸
- [[concepts/foundation-model]] — WSL foundation model の代表
- [[entities/perception-encoder]] — SigLIP 系の最新拡張
- [[sources/dinov3]] — SigLIP を比較対象に持つ純粋 SSL 側論文
- [[sources/sam-3]] — SigLIP を比較対象に持つ promptable concept seg 論文
- [[overview]] — CV 全体俯瞰での位置づけ
