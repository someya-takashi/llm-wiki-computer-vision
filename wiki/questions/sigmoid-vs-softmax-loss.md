---
type: question
asked: 2026-06-10
question: "SigLIP で softmax と sigmoid の損失関数の比較が説明されているが、メモリ効率が良い理由と分散学習で利点がある理由をわかりやすく解説してほしい。"
sources_used: ["[[sources/siglip]]", "[[sources/clip]]", "[[sources/siglip-2]]", "[[entities/siglip]]", "[[entities/clip]]", "[[concepts/contrastive-learning]]", "[[concepts/weakly-supervised-pretraining]]", "[[entities/qwen3-vl]]"]
---

# SigLIP の sigmoid 損失はなぜ memory 効率と分散学習で有利なのか

## 結論を先に

**鍵は「セル間の独立性」**：

- **CLIP の softmax 損失**: 各行に **グローバル正規化** が必要 → **$N \times N$ 行列全体を同時に材化（materialize）** しなければ計算できない
- **[[sources/siglip|SigLIP]] の sigmoid 損失**: 各 $(i, j)$ セルが **独立な二値分類** → **ブロック単位で計算・破棄** できる

具体的な利得：

| 指標 | CLIP (softmax) | SigLIP (sigmoid) | 比率 |
|---|---|---|---|
| ピーク・メモリ（$N=32768$ で 1024² ブロック計算）| ~4 GB（行全体）| ~4 MB（1 ブロック）| **1000×** |
| 分散学習の通信 | 全 GPU 埋め込みの **all-gather** | テキスト・チャンクの **順送り** | 大幅減 |
| 訓練計算予算（IN-0 79.7%）| 数十〜数百 TPU | **4 TPUv4 × 1 日** | 大幅減 |

直感的な対比：

> **softmax = 競争（試験の偏差値）**: 同じバッチの他の例との「相対順位」が損失に影響、**全員の点数が必要**
>
> **sigmoid = 個別判定（合否判定）**: 各ペアを独立に「合うか合わないか」判定、**他のペアの情報は不要**

この「**独立性**」が、メモリ効率と分散効率の両方の源泉。

---

## ステップ 1: CLIP の softmax 損失の構造

### バッチ単位の類似度行列

バッチサイズ $N$ の画像-テキスト対 $\{({\boldsymbol v}_i, {\boldsymbol t}_i)\}_{i=1}^{N}$ を考える。**$N \times N$ の類似度行列** $S$ を作る：

```
            テキスト 1   テキスト 2   ...   テキスト N
画像 1     [ s_11       s_12       ...    s_1N      ]
画像 2     [ s_21       s_22       ...    s_2N      ]
  :        [  :          :         ...     :        ]
画像 N     [ s_N1       s_N2       ...    s_NN      ]
```

ここで $s_{ij} = {\boldsymbol v}_i^{\intercal} {\boldsymbol t}_j / \tau$ はコサイン類似度（$\tau$ は温度）。

### CLIP の損失（[[sources/clip|InfoNCE]] の対称版）

各画像 $i$ について「**正解はテキスト $i$、他は不正解**」として softmax + cross-entropy：

$$L_{i \to t} = -\dfrac{1}{N}\sum_{i=1}^{N}\log\dfrac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}$$

テキスト → 画像 方向も対称に計算し、両者の平均を取る。

### **決定的な観察**: 分母にバッチ全体が必要

$$\text{分母} = \sum_{j=1}^{N}\exp(s_{ij})$$

これは **行 $i$ の全エントリの和**。つまり「画像 1 の損失を計算するには、画像 1 と **すべての** テキストの類似度 $s_{1,1}, s_{1,2}, \ldots, s_{1,N}$ が必要」。

これが [[concepts/contrastive-learning|softmax]] の **「グローバル正規化」** で、性能の源泉でもあり制約の源泉でもある。

---

## ステップ 2: なぜ softmax はメモリを食うのか

### 完全な $N \times N$ 行列を materialize する必要

softmax は **行ごとに正規化** するため、その行のすべての値が **同時にメモリに乗っている** 必要があります。

具体的な数値（SigLIP の主張する **バッチサイズ 32k** で）:

| 項目 | 数値 |
|---|---|
| バッチサイズ $N$ | 32,768 |
| 類似度行列 $S$ のサイズ | $32768 \times 32768$ ≈ **10.7 億エントリ** |
| FP32 で保持 | $10.7\text{B} \times 4\text{ B}$ = **約 4.3 GB** |
| FP16 で保持 | 約 2.1 GB |
| $\exp(s_{ij})$ も同サイズ | さらに +2-4 GB |

しかも softmax の **数値安定性** のため、$\exp$ の計算は通常 FP32 で行う必要があります（FP16 では $\exp(20)$ で既にオーバーフロー）。

**問題**: GPU の活性化メモリの大部分がこの 1 つの行列で消費される。

### なぜ「行を分割」できないか

「行ごとに分割して計算すれば？」と考えるかもしれません。しかし：

```
画像 1 の損失 = -log(exp(s_11)) + log(exp(s_11) + exp(s_12) + ... + exp(s_1N))
                                  ─────────────────────────────────────────
                                  この sum を出すには行全体が必要
```

**分母が「行の全成分の和」なので、行を分割できない**。これが softmax の **本質的な制約**。

数値的にはオンライン softmax（[[sources/siglip|SigLIP]] §2.3 で言及）でメモリ削減も可能だが、それでも **2 パス必要**（最大値の計算 → 正規化）で複雑化する。

---

## ステップ 3: SigLIP の sigmoid 損失の構造

### アイデア: 各セルを独立な「二値分類」として扱う

[[sources/siglip|SigLIP]] は softmax の代わりに **各 $(i,j)$ ペアを独立にバイナリ分類**：

> 「画像 $i$ とテキスト $j$ は対応するペアか、しないか？」

正解ラベル $z_{ij}$:
- $z_{ij} = +1$ if $i = j$（正例ペア）
- $z_{ij} = -1$ if $i \neq j$（負例ペア）

損失：

$$L = -\dfrac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma(z_{ij}(t \cdot s_{ij} + b))$$

ここで $\sigma$ は sigmoid 関数、$t$ は **学習可能な温度**、$b$ は **学習可能なバイアス**（後述の初期化が重要）。

### 鍵となる違い: 各セルの完全独立

```
L_11 の計算: s_11 のみ必要
L_12 の計算: s_12 のみ必要
L_13 の計算: s_13 のみ必要
...
L_NN の計算: s_NN のみ必要
```

softmax と違い、**「行の合計」も「列の合計」も不要**。**全 $N \times N$ セルが互いに完全独立**。

### 数値安定性も改善

sigmoid は $\sigma(x) = 1/(1+e^{-x})$ で、log-sigmoid は **numerical stable な形式 $-\log(1+e^{-x}) = -\text{softplus}(-x)$** で計算でき、**FP16/BF16 でも安定**。

これに対し softmax は $\exp$ → 和 → 割り算の 3 段階で誤差が累積。

---

## ステップ 4: なぜ sigmoid はメモリ効率的か

### チャンク化計算が可能

各セルが独立なので、**$N \times N$ 行列をブロック単位で計算・破棄**できます：

```
chunk_size = 1024 とすると：

ステップ 1:
  最初の 1024 × 1024 ブロックを計算
       ↓
  そのブロックの sigmoid 損失を計算して累積
       ↓
  ブロックを破棄（メモリ解放）

ステップ 2:
  次の 1024 × 1024 ブロックを計算
       ↓
  ...
  
合計 32 × 32 = 1024 ブロックを処理
```

**ピーク・メモリ使用量**:

| 戦略 | ピーク・メモリ |
|---|---|
| softmax: 行全体（$32768 \times 32768$、FP32）| **4 GB** |
| sigmoid: 1 ブロック（$1024 \times 1024$、FP32）| **4 MB** |
| sigmoid: 1 ブロック（$1024 \times 1024$、BF16）| **2 MB** |

**約 1000-2000 倍の削減**。これが **バッチサイズ 32k を 4 つの TPUv4 で 1 日訓練できる根本理由**。

### 直感的な例え

> **softmax は「クラス全員のテストの点数を見て偏差値を出す」**。クラス全員の点数が手元になければ偏差値は計算できない。
>
> **sigmoid は「個別の答案を一人ずつ採点する」**。一人ずつ独立に「合格/不合格」を判定できる。採点済みの答案は破棄してよい。

---

## ステップ 5: なぜ分散学習で有利か

### softmax の分散学習の問題

複数 GPU で訓練する場合（典型的に 64-256 GPU、SigLIP は最大 256 TPUv4）：

- 各 GPU が **バッチの一部** $N/G$ の画像とテキストを処理（$G$ は GPU 数）
- しかし softmax の分母 $\sum_j \exp(s_{ij})$ は **バッチ全体のテキスト埋め込み** を要求

**結果**: 全 GPU の埋め込みを **all-gather**（全通信）する必要がある：

```
GPU 0:  [ v_0 ... v_{N/G-1} ]  + 自分のテキスト t_0 ... t_{N/G-1}
GPU 1:  [ v_{N/G} ... ]
   :
GPU G-1:[ ... ]

         ↓ all-gather (テキスト全部を全 GPU に配布)

GPU 0:  [ t_0 ... t_N ]（全テキスト埋め込み）← 通信が爆発、メモリも爆発
GPU 1:  [ t_0 ... t_N ]
   :
```

**コスト**:
- **通信**: $O(N \cdot d \cdot G)$（$d$ は埋め込み次元、$G$ は GPU 数）。GPU 数 × バッチサイズで線形に増大
- **メモリ**: 各 GPU が **全バッチ埋め込み** を保持 → スケールしない

### sigmoid の分散学習の優雅さ

各セルが独立なので、**「ペアごとに必要な埋め込みだけ」を該当 GPU に転送**すれば十分：

```
GPU 0 が担当する (i, j) ペア：
  - 画像埋め込み v_i は GPU 0 にある
  - テキスト埋め込み t_j は **必要なときだけ** 別 GPU から取得

行列を「ブロック対角」で分散：
  GPU 0: 自分の画像 v_0..v_{N/G} × 全テキスト t_0..t_N を順次処理
       Step 1: 自分のテキスト t_0..t_{N/G} と内積 → 損失累積
       Step 2: GPU 1 のテキスト t_{N/G}..t_{2N/G} を受信 → 損失累積
       Step 3: GPU 2 のテキストを受信 → 損失累積
       ...
       Step G: GPU G-1 のテキストを受信 → 損失累積
```

**[[sources/siglip|SigLIP]] §3 の通信戦略**: **テキストの順送り（ring all-reduce 風）**

```
各ステップで GPU $k$ は：
  1. 現在保持しているテキスト・チャンクとの内積を計算
  2. 損失を累積
  3. 自分のテキスト・チャンクを次の GPU に送信
  4. 前の GPU からテキスト・チャンクを受信

G ステップで全ブロックを処理完了
```

**コスト削減**:
- **通信**: 一度に転送するのは **テキスト・チャンク 1 個分**（$N/G \cdot d$）。**全 GPU 同時の all-gather** ではなく **隣接 GPU 間の順送り**
- **メモリ**: 各 GPU は常に **自分の画像 + 自分のテキスト + 受信中の 1 テキスト・チャンク** のみ。スケールしても増えない

### 図解で対比

```
■ softmax (CLIP):
  ┌──────┐ ┌──────┐ ┌──────┐
  │GPU 0 │ │GPU 1 │ │GPU 2 │
  │      │ │      │ │      │
  │ all  │←│ all  │→│ all  │  all-gather: 全 GPU の埋め込みが
  │gather│ │gather│ │gather│   全 GPU に必要
  └──────┘ └──────┘ └──────┘
  通信 O(N·d·G)、メモリ O(N·d) per GPU

■ sigmoid (SigLIP):
  ┌──────┐    ┌──────┐    ┌──────┐
  │GPU 0 │ →  │GPU 1 │ →  │GPU 2 │ → GPU 0 へ循環
  │  t_0 │    │  t_1 │    │  t_2 │
  └──────┘    └──────┘    └──────┘
  各ステップで隣接 GPU 間でテキスト・チャンク 1 個転送
  通信 O(N·d) total、メモリ O(N/G·d) per GPU
```

---

## ステップ 6: 学習可能バイアス $b$ の役割（小バッチでも有利な理由）

### sigmoid の「正例 vs 負例」の不均衡

各 $i$ について **正例ペアは 1 個（$j = i$）、負例ペアは $N-1$ 個**。

例：$N = 32$ なら正例 32 個、負例 992 個。**負例が圧倒的多数**。

そのまま訓練すると、モデルは **「全部負例」と予測すれば 96.9% 正解** という自明解に落ちる（[[concepts/contrastive-learning|崩壊]] に近い現象）。

### SigLIP の対策: $b$ をマイナス初期化

[[sources/siglip|SigLIP]] §2.1 の重要な工夫：

$$\sigma(z_{ij}(t \cdot s_{ij} + b))$$

の **$b$ を $b_0 = -10$（強い負バイアス）で初期化**。

これにより：
- 初期では **すべてのペアを「負例」と予測** が自然な状態
- 訓練が進むにつれ、**正例ペアだけ「正例」と分類する方向にバイアスが減少**

### バッチサイズの効果

[[sources/siglip|SigLIP]] §4.3 の重要な発見:

> **バッチサイズ 32k で性能が飽和する**

- それまでの常識（[[sources/clip|CLIP]] 派）: 「大バッチほど良い、無限にスケールできる」。OpenAI CLIP は 32k で訓練、その後の研究は 64k, 128k... を試みた
- **SigLIP の発見**: **32k 以上は無駄**

直感的説明:
- CLIP softmax: 負例の数 = $N-1$ に直接依存 → **大バッチで負例多数 = 学習信号強化**
- SigLIP sigmoid: 各セルが独立な二値分類 → **「負例の数」ではなく「比率」が重要、バイアス $b$ で補正**

→ **小バッチでも sigmoid は弱くない**、これが計算予算の限られた研究にも開かれる利点。

---

## ステップ 7: 具体的な数値比較（[[sources/siglip|SigLIP]] §4 より）

### 訓練コストの比較

| 項目 | CLIP (softmax) | SigLIP (sigmoid) |
|---|---|---|
| 16k バッチで IN-0 79.7% | 多数の TPU × 数日 | **4 TPUv4 × 1 日** |
| 推奨バッチサイズ | 32k 以上が望ましい（精度のため）| **32k で飽和、それ以上不要** |
| 100k+ バッチ | 事実上不可能（メモリ・通信） | 訓練可能（性能改善なし） |
| 最大規模モデル | ViT-G/14, EVA-CLIP-E | **SO-400M（83.2% IN-0）が 5B EVA-CLIP を凌駕** |

### バッチサイズ別の比較（[[sources/siglip]] §4.3 図 4）

| バッチサイズ | SigLIP | CLIP |
|---|---|---|
| 4k | sigmoid 圧勝 | softmax 大幅劣化 |
| 16k | sigmoid 優位 | softmax 同等近く |
| 32k | 同等（**飽和点**）| 同等 |
| 64k+ | 改善なし | 改善なし |

**この発見が「小バッチでも sigmoid が強い」という SigLIP の主張の根拠**。

---

## ステップ 8: 後続研究での影響

### SigLIP 2（[[sources/siglip-2]]、Google DeepMind 2025）

SigLIP の損失関数を基盤にしつつ、**5 系統融合**:
- 対比（sigmoid 損失、SigLIP 直接継承）
- LocCa decoder
- 自己蒸留（SILC 由来）
- マスク予測（TIPS 由来）
- ACID 蒸留

メモリ効率を保ちつつ多様な目的を追加。**g/16（1B）サイズで 85.0% IN-0**。

### Qwen-VL 系の vision encoder 採用

- [[entities/qwen3-vl|Qwen3-VL]]: **SigLIP-2 から継続学習**
- [[entities/gemma-3|Gemma 3]]: **SigLIP 400M variant を共有・凍結**
- 多くの MLLM の vision encoder として SigLIP/SigLIP-2 が事実上の標準

### Perception Encoder（[[sources/perception-encoder|PE]]、Meta 2025）

「**対比学習をピュアに保ったまま大規模化**」する路線で、**sigmoid 損失は採用せず softmax-based 対比を保持**しつつ、**慎重なキュレーション + 大バッチ（131k）で頑健化**。SigLIP の補完的対立軸。

---

## まとめ — ユーザの問いへの直接的回答

### Q1: なぜ sigmoid 損失はメモリ効率が良いのか？

**A: softmax が「行ごとのグローバル正規化」を要求するのに対し、sigmoid は「各 $(i,j)$ セルが独立な二値分類」だから**。

- **softmax**: $N \times N$ 行列全体を materialize して行正規化 → バッチ 32k で **4 GB+**
- **sigmoid**: **ブロック単位（例 $1024 \times 1024$）でチャンク化計算** → ピーク **4 MB**
- **約 1000 倍の削減** → 4 TPU で 1 日訓練可能

### Q2: なぜ分散学習で有利か？

**A: softmax の分母計算が「バッチ全体の埋め込み」を要求するのに対し、sigmoid は「ペアごとに必要な埋め込みだけ」あれば計算できるから**。

- **softmax**: 全 GPU の埋め込みを **all-gather** 必要 → 通信爆発、メモリ爆発
- **sigmoid**: **テキスト・チャンクを順送り** するだけで分散計算可能 → 通信小、メモリ小

### 直感的な要約

> **softmax = 競争（試験の偏差値）**: 同じバッチの他の例と「相対的に」勝たねばならない。全員の点数が必要。
>
> **sigmoid = 個別判定（合否判定）**: 各ペアを独立に「合うか合わないか」判定。他のペアの情報は不要。

この「**独立性**」が、メモリ効率と分散効率の両方の源泉。

### CV / MLLM での影響

- [[entities/siglip|SigLIP]] が **SO-400M で 5B EVA-CLIP を凌駕**（83.2% IN-0）
- [[sources/siglip-2|SigLIP 2]] の 5 系統融合への発展
- **MLLM の vision encoder の事実上の標準**: Qwen3-VL（SigLIP-2 継続学習）、Gemma 3（SigLIP 400M 共有）、その他多数
- **データセット規模を上げやすい**（同計算予算でより多くのサンプルを処理）

---

## 関連ページ

### 直接の出典

- [[sources/siglip]] — SigLIP 原典（Zhai et al., Google DeepMind, ICCV 2023）。本ページの理論的基盤
- [[entities/siglip]] — SigLIP モデルのスペック
- [[sources/clip]] — CLIP 原典（Radford et al., 2021）、比較対象の softmax 対比学習

### 発展形

- [[sources/siglip-2]] — SigLIP 2（Tschannen et al., 2025）、5 系統融合の集大成
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — Meta PE、softmax 系を保ちつつ大バッチで頑健化（対立軸）

### MLLM での採用

- [[entities/qwen3-vl]] / [[sources/qwen3-vl]] — SigLIP-2 から継続学習
- [[entities/gemma-3]] / [[sources/gemma-3]] — SigLIP 400M variant 共有・凍結

### 上位概念

- [[concepts/contrastive-learning]] — 対比学習全般、softmax/sigmoid の位置付け
- [[concepts/weakly-supervised-pretraining]] — WSL 全般、CLIP/SigLIP 系の系譜
- [[concepts/foundation-model]] — 基盤モデル全体

### 関連 question

- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列、SigLIP の位置付け
- [[questions/vit-dynamic-resolution-evolution]] — Vision encoder の進化、SigLIP の役割
