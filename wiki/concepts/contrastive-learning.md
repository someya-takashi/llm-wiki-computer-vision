---
type: concept
aliases: [Contrastive Learning, 対比学習, contrastive representation learning]
tags: [paradigm, ssl, training-technique, representation-learning]
related: [[self-supervised-learning]], [[weakly-supervised-pretraining]], [[knn-evaluation-protocol]]
sources: [[sources/simclr]], [[sources/clip]], [[sources/dino-emerging-properties-in-self-supervised-vit]], [[sources/siglip]], [[sources/siglip-2]]
updated: 2026-05-27
---

# Contrastive Learning（対比学習）

## 一言で

**「正例ペア（類似すべきもの）の埋め込みを近づけ、負例ペア（異なるもの）の埋め込みを遠ざける」**ことで表現を学習する枠組み。距離学習（metric learning）から発展し、SimCLR / MoCo / CLIP の登場（2020-2021）で SSL と弱教師あり学習の中核技術となった。**「何を正例と見なすか」**で多様な変種を生む万能的な学習パラダイム。

---

## 基本構造

```
[サンプル x]
   ↓
データ変換（augmentation, ペア化, etc.）
   ↓ ↓
正例 x⁺ ─────→ 埋め込み f(x⁺)
負例 x⁻ ─────→ 埋め込み f(x⁻)
   ↓
類似度関数（コサイン類似度）
   ↓
sim(f(x), f(x⁺)) を最大化、sim(f(x), f(x⁻)) を最小化
```

学習目的（**InfoNCE 損失**、対比学習の標準）:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(f(x), f(x^+)) / \tau)}{\exp(\text{sim}(f(x), f(x^+)) / \tau) + \sum_{x^-} \exp(\text{sim}(f(x), f(x^-)) / \tau)}
$$

- $\tau$: 温度パラメータ（学習可能または固定）
- 分子: 正例との類似度
- 分母: 正例 + すべての負例との類似度

> **補足: InfoNCE の直観** — 多項ロジスティック回帰として読むと分かりやすい。「正例を当てるクラス分類」を、バッチ内のすべてのペアの中で行っている。「正例 1 つ vs 負例 N-1 個」を当てれば、$\log_2 N$ ビットの相互情報量を学習したことになる、というのが論文 [van den Oord et al. 2018] の解釈。

---

## 何を正例とするかで変種が生まれる

### 1. インスタンス識別（instance discrimination）

**SimCLR / MoCo 系**:
- **正例**: 同じ画像 $x$ の 2 つの異なるデータ拡張版 $\text{aug}_1(x), \text{aug}_2(x)$
- **負例**: バッチ内の他の画像

ここでは「**augmentation 不変性**」を学習する。color jitter, crop, blur 等を変えても同じ意味になるように。

> **補足: なぜ強い拡張が必要か** — 軽い拡張だと正例ペアが「ほぼ同じ」になり学習信号が弱い。SimCLR は意図的に強い random crop, color jitter, Gaussian blur, solarization 等を使う。これが対比学習の最大の特徴で、MAE 等の MIM が augmentation 不要なのと対照的。

#### SimCLR の核心設計（[[sources/simclr]]）

SimCLR（Chen et al., ICML 2020）は「何が対比学習を効かせるか」を体系的なアブレーションで解明した。3 つの重要な発見：

**① 拡張の構成が決定的** — 単一の拡張（クロップのみ、カラー歪みのみ）では性能が低い。**ランダムクロップ×カラー歪みの組み合わせ**が突出して重要。理由: クロップのみだと同じ画像の 2 パッチはほぼ同じカラーヒストグラムを共有し、ショートカットを使える。カラー歪みを加えると、モデルは色に頼らず意味的一致を見つけなければならなくなる。

**② 非線形射影ヘッドが最大の単独改善要因**

```
エンコーダ出力 h → [射影ヘッド g] → z
                              ↑
                    ここで NT-Xent 損失を計算（訓練後は捨てる）
                    下流タスクには h を使う
```

射影なし: ~57%、線形射影: ~60%、**非線形 MLP: ~63-64%**（+10% 以上）。
*なぜ機能するか*: z は変換不変になるよう訓練されるため、色・向きなどの情報が消える。g がそのバッファ役を担い、h に豊かな情報を維持させる。

**③ 対比学習は教師あり学習より強い拡張を必要とする** — カラー歪みを強くすると SSL の性能は向上するが、同じ拡張を教師あり学習に適用すると性能が下がる。対比学習はショートカット特徴（カラー統計など）を使えなくすることを学習信号として活用するため。

### 2. クロスモーダル対比（CLIP）

詳細: [[sources/clip]] / [[entities/clip]]

- **正例**: 対になっている画像とテキスト $(I, T)$
- **負例**: バッチ内の他の画像とテキストのすべての組み合わせ

CLIP のような **画像-テキスト** の対比は、結合埋め込み空間を学習し、ゼロショット分類を可能にする。

### 3. クラスタリングベース対比

**SwAV / DeepCluster**:
- **正例**: 同じクラスタに割り当てられたサンプル
- 負例の明示なし、クラスタ割当の均等化制約で代用

### 4. 動画・時系列対比

**Time-Contrastive Networks**:
- **正例**: 動画内の時間的近接フレーム
- **負例**: 別動画または離れたフレーム

---

## 崩壊（collapse）回避

対比学習も SSL の他の系統と同じく崩壊問題を持つ：

| 崩壊の種類 | 例 |
|---|---|
| 定数崩壊 | すべての入力に同じ埋め込み → 類似度がすべて同じ |
| 次元崩壊 | 一部の次元だけ使う、表現の自由度低下 |

**対比学習の崩壊回避策**:
- **負例の存在自体**: 異なるサンプルが「離れる」よう学習されるため、自動的に崩壊しない
- **大バッチが必要**: SimCLR は 8192 など。負例の多様性が学習信号の質を決める

これが「対比学習」の弱点でもあり、**大バッチ → 大量メモリ要求** につながる。

### 「非対比型」の出現

BYOL / SimSiam / DINO は **負例なしで動かす方法**を発見した：

- **BYOL**: predictor + momentum encoder で崩壊を防ぐ
- **DINO**: centering + sharpening で崩壊を防ぐ
- **SimSiam**: stop-gradient だけで動く

これらは厳密には「対比学習」ではないが、対比学習との対比で「**非対比型 SSL**」と呼ばれる。詳細: [[concepts/self-supervised-learning]]

---

## CLIP の対比損失（特殊形）

CLIP（[[sources/clip]]）の対比損失は他と少し違う：

- **対称的 InfoNCE**: 行方向（画像→テキスト）と列方向（テキスト→画像）の両方で softmax を取る
- バッチサイズ **32,768** という巨大スケール
- 温度 $\tau$ は **学習可能**（log-parameterized、100 倍以下にクリップ）

```python
# CLIP の擬似コード
logits = (I @ T.T) * exp(τ)
labels = arange(N)
loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
```

これは N-pair 損失 [Sohn 2016] と InfoNCE [van den Oord 2018] を画像-テキストドメインに適用したもの。**ConVIRT** [Zhang et al., 2020] が医療画像で先行したアイデアを、4 億画像規模に拡張した。

## Sigmoid 損失（SigLIP の代替）

CLIP の softmax 対比損失には実用上の問題があった：

1. **メモリ集約**: バッチ全体の $|\mathcal{B}|^2$ 類似度行列を実体化
2. **数値不安定**: softmax のナイブ実装は不安定で、最大値減算で追加バッチパスが必要
3. **タスク定義とバッチサイズの結合**: 「正しい 1 個を選ぶ」というタスク自体がバッチ構成に依存

**SigLIP**（[[sources/siglip]] / [[entities/siglip]]）は **sigmoid 損失** で根本的に解決：

$$-\frac{1}{|\mathcal{B}|} \sum_i \sum_j \log \frac{1}{1 + e^{z_{ij}(-t \mathbf{x}_i \cdot \mathbf{y}_j + b)}}$$

ここで $z_{ij} \in \{+1, -1\}$（対角が +1、それ以外 -1）。

```python
# SigLIP の擬似コード
t = exp(t_prime)
zimg, ztxt = l2_normalize(I), l2_normalize(T)
logits = zimg @ ztxt.T * t + b  # b は学習可能 bias
labels = 2 * eye(n) - ones(n)   # diagonal=+1, off=-1
loss = -sum(log_sigmoid(labels * logits)) / n
```

**InfoNCE (CLIP) vs sigmoid (SigLIP) の対比**：

| 観点 | **softmax InfoNCE** | **sigmoid (SigLIP)** |
|---|---|---|
| **タスクの解釈** | 「正しいペアを選ぶ」（バッチ依存） | 「このペアは正しいか？」（**ペア独立**） |
| **正規化** | バッチ全体 softmax（2 方向） | 不要 |
| **メモリ** | $\mathcal{O}(\|\mathcal{B}\|^2)$ | **chunked で $\mathcal{O}(b^2)$** |
| **数値安定化** | 最大値減算で追加パス | `log_sigmoid` で安定 |
| **対称性** | 非対称（2 方向別正規化） | **対称的** |
| **小バッチ性能** | 弱い | **大幅優位** |
| **バッチサイズ要求** | 大（32k+） | **小〜中で十分**（32k で飽和） |
| **ノイズ頑健性** | 弱い | **強い** |
| **学習可能 bias** | なし | **必須**（b=-10, t'=log10 初期化） |

> **補足: sigmoid が「対比学習」と言えるか** — sigmoid 損失は形式上、各ペアを独立に二値分類しているように見えるが、**バッチ内の他ペアの relative ranking を学習する** という対比学習の核心思想は維持されている。違いは「バッチ全体で確率正規化するか」だけ。SigLIP 著者は「sigmoid と softmax は同じ contrastive 学習目的への異なる近似」と位置づける。

## SigLIP 2: 対比学習を超えた統合レシピ

[[sources/siglip-2]]（Tschannen et al., 2025）は **対比学習の枠を超え、CLIP 4 年分の独立改善を統合**：

1. **対比損失（sigmoid loss）**: SigLIP v1 の継承
2. **decoder ベース事前学習（LocCa）**: pool 前視覚特徴に transformer decoder を接続し、キャプション生成 + 参照表現予測 + grounded captioning を同時訓練
3. **自己蒸留（SILC）**: student（局所ビュー）と teacher（global ビュー）の MLP 後特徴の一貫性損失（[[entities/dino]] と同系統）
4. **マスク予測（TIPS）**: 50% パッチをマスクし teacher の特徴にマッチ（[[entities/ibot]] / [[concepts/masked-image-modeling]] と同系統）
5. **能動的データキュレーション（ACID）**: teacher/learner で learnability を計算しバッチを共同選択

> **補足: 対比学習はもう「単独で十分」ではない** — SigLIP 2 の貢献は、対比学習を中核に据えつつ「キャプション化／自己蒸留／マスク予測／データキュレーション」を段階的に統合することで、対比単独では届かなかった dense 予測・位置特定・OCR の性能を達成した点。対比学習は **必要条件だが十分条件ではない**、というのが 2025 年時点の WSL の到達点。

これにより：
- **RefCOCO（参照表現理解）+20pt**（decoder の効果）
- **ADE20k seg +4.2pt, PASCAL seg +5.1pt**（自己蒸留＋マスク予測の効果）
- **小型モデル +2.4pt**（ACID 蒸留の効果）

対比学習を **基礎レイヤ** として、その上に他系統の技法を積層する流れが 2024〜2025 の主流。

---

## 歴史的系譜

```
[1992] Becker & Hinton: Self-organizing networks（対比学習の原点）
   ↓
[2006] Hadsell, Chopra, LeCun: 次元削減のためのコントラスト損失
   ↓
[2010s] Triplet loss (FaceNet 2015), N-pair loss (Sohn 2016)
   ↓
[2018] van den Oord: CPC（InfoNCE 損失の提唱）
   ↓
[2019] MoCo (He et al.): momentum encoder + queue で大規模化
[2020] SimCLR (Chen et al.): 大バッチ + 強い拡張で SOTA
[2020] MoCo v2 / SwAV: 改良競争
   ↓
[2020] BYOL: 負例なしでも動くと発見
[2021] SimSiam: predictor + stop-gradient のみ
[2021] DINO: 自己蒸留として再解釈
[2021] CLIP: 画像-テキスト対比でゼロショット転移
   ↓
[2022] BLIP / FLIP: CLIP の拡張
[2023] SigLIP: softmax → sigmoid で効率化
[2024] PE: 86B ペアで超大規模化
[2024-25] LocCa / SILC / TIPS: CLIP に decoder / 自己蒸留 / マスク予測を追加
[2025] SigLIP 2: 上記すべてを統合した「全部入りレシピ」
```

---

## 対比学習の長所と短所

### 長所
- **理論的にきれいな枠組み**: InfoNCE は相互情報量の下界を最適化
- **正例の定義次第で柔軟**: 画像-画像、画像-テキスト、画像-音声など多様な応用
- **強い表現を学習**: 凍結特徴量で linear probing/k-NN が強い
- **大バッチで動く**: 大規模分散訓練と相性が良い

### 短所
- **大バッチ要求**（softmax InfoNCE の場合）: 小バッチだと負例多様性不足。ただし sigmoid 損失（SigLIP）は小バッチでも動く
- **augmentation 設計が肝**: 弱すぎる/強すぎると性能崩壊
- **dense prediction が弱い**: 画像レベルの表現に偏り、パッチレベル意味性は MIM ほど強くない
- **負例選びの問題**: バッチ内負例には実は「正例」（同じクラス）が混入することがあり、性能を阻害

---

## 関連ページ

- [[sources/clip]] / [[entities/clip]] — クロスモーダル対比学習の代表
- [[sources/simclr]] / [[entities/simclr]] — 対比学習の 4 コンポーネントを解明した基礎論文（ICML 2020）
- [[sources/clip]] / [[entities/clip]] — クロスモーダル対比学習の代表
- [[sources/dino-emerging-properties-in-self-supervised-vit]] / [[entities/dino]] — 非対比型 SSL の代表、DINO は「自己蒸留」枠組みで対比学習を超える
- [[concepts/self-supervised-learning]] — SSL 全般の中での対比学習の位置づけ
- [[concepts/weakly-supervised-pretraining]] — 画像-テキスト対比は WSL の代表手法
- [[concepts/masked-image-modeling]] — MIM は対比学習と対照的な系統
