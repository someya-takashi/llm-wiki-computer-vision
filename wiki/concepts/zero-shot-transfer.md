---
type: concept
aliases: [Zero-Shot Transfer, ゼロショット転移, zero-shot classification, zero-shot learning]
tags: [paradigm, evaluation, transfer-learning]
related: [[weakly-supervised-pretraining]], [[foundation-model]], [[knn-evaluation-protocol]], [[promptable-segmentation]]
sources: [[sources/clip]], [[sources/segment-anything]]
updated: 2026-05-26
---

# Zero-Shot Transfer（ゼロショット転移）

## 一言で

**訓練時に一度も見たことがないタスク・データセット・クラスに対して、追加訓練なしで予測を行う**こと。CV ではもともと「未見の物体カテゴリへの汎化」を意味する狭い用語だったが、**CLIP（[[sources/clip]]）**が「未見のデータセットへの汎化」というより広い意味でこの概念を持ち込み、現代の CV 基盤モデル評価の中核プロトコルとなった。NLP の **GPT-2/GPT-3** が示した「事前学習中にタスク学習が起こる」という発見の CV 版。

---

## ゼロショットの 3 つの異なる意味

「zero-shot」という用語は文脈で意味が大きく違うので注意：

### 1. 古典的な意味: 未見のクラスへの汎化（zero-shot learning）

[Lampert et al., 2009] の伝統的なゼロショット学習：

- 訓練時に「シマウマ」を見ていない
- でも「シマウマは縞模様の馬」という**属性情報**は知っている
- → 既知クラス（馬、縞模様の物体）の組み合わせとしてシマウマを推論

属性ベースの推論で、補助知識（attribute, word embedding 等）を必要とする。

### 2. CLIP 流の意味: 未見のデータセットへの汎化

[[sources/clip]] が拡張した定義:

- 訓練データセット（CLIP の場合 WIT）でクラス名と画像のペアを大量に見ている
- 推論時、ImageNet 等のクラス名をテキストで提示するだけ
- → 「ImageNet で 1 度も訓練していないのに ImageNet 分類器が即座に動く」

実質的に「**新しいタスクへの転移**」を意味する。CLIP は「タスクを評価する代理として、データセットへの転移を使う」と論文で明示。

### 3. プロンプトベースの意味（NLP）

GPT-2/GPT-3 が示した:
- 「翻訳しろ: Hello → ?」とプロンプトを書く
- LLM はファインチューニングなしで翻訳する
- → 「prompt」で任意のタスクを指示できる

CLIP の「a photo of a {class}」プロンプトは、この NLP の流れを CV に持ち込んだ。

> **補足: 3 つの意味の関係** — CLIP のゼロショットは 2 と 3 の融合。「テキストで指定できる任意のタスクに、追加訓練なしで対応」が CV 基盤モデルの目指す姿となった。

---

## CLIP のゼロショット転移の仕組み

```
ステップ 1: クラス名にプロンプトを付ける
  classes = ["dog", "cat", "bird"]
  prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

ステップ 2: テキストエンコーダで埋め込み
  text_embeddings = TextEncoder(prompts)  # [3, D]

ステップ 3: 画像をエンコード
  image_embedding = ImageEncoder(image)  # [D]

ステップ 4: コサイン類似度で予測
  logits = image_embedding @ text_embeddings.T  # [3]
  prediction = argmax(softmax(logits))
```

これは「**ハイパーネットワーク** = テキストエンコーダ」という解釈ができる：

> テキストエンコーダがテキストから線形分類器の重みを生成し、画像エンコーダの特徴量に対する分類器として機能する。

CLIP 論文 §3.1.2 より。

---

## プロンプトエンジニアリング（Prompt Engineering）

CLIP 論文が CV に導入した重要な発見（NLP 由来）：

| クラス指定の方法 | ImageNet zero-shot 精度 |
|---|---|
| クラス名のみ（`"dog"`） | ベースライン |
| `"a photo of a dog"` プロンプト | **+1.3%** |
| タスク固有プロンプト（例: `"a photo of a {label}, a type of pet"`）| 追加で改善 |
| 80 プロンプトのアンサンブル平均 | **+3.5%** |
| プロンプト+アンサンブル合計 | **約 +5%** |

> **補足: なぜプロンプトが効くか** — 訓練データ WIT で、画像と組になっているテキストはほぼ常に文（「a photo of...」「a picture of...」等）。クラス名単独は分布外の入力で、テキストエンコーダが上手く処理できない。プロンプトで「実際の訓練分布に近い形式」に揃えることで性能が上がる。

### プロンプトアンサンブル

複数のプロンプト変種の埋め込みを**平均化**し、それを単一の分類器として使う：

```python
templates = ["a photo of a {}", "a picture of a {}", "an image of a {}", ...]  # 80 種
embeddings = []
for template in templates:
    text = template.format(class_name)
    embeddings.append(TextEncoder(text))
classifier = mean(embeddings)  # 平均化
```

**計算コスト: 1 プロンプト分**（一度平均すれば使い回せる）。

---

## ゼロショット転移の評価指標

### 精度

通常の top-1, top-5 accuracy。

### Effective Robustness（[Taori et al., 2020]）

**「ID 性能から予測される OOD 性能を超える改善」**。

ImageNet 訓練モデルの ID 精度と OOD 精度には**強い線形関係**がある。これを基準として、その線から「上に外れる」モデルが effective robustness を持つと言える。

CLIP 論文の §3.3 の最重要発見:
- ゼロショット CLIP は effective robustness が圧倒的に高い
- ImageNet 精度に応じて他のモデルが OOD で苦戦する中、CLIP は **robustness gap を最大 75% 削減**

### Relative Robustness

OOD 性能の絶対値そのものの改善。

---

## 何故 CLIP のゼロショットが効くのか

複数の要因の組み合わせ:

1. **対比学習で意味的に整理された埋め込み空間**: 類似画像と類似テキストが近くにある
2. **テキストの汎用性**: 任意のタスクをテキストで記述できる
3. **大規模ウェブデータ**: WIT 4 億ペアに ImageNet 概念は当然含まれる（だから「ゼロショット」と言えるか議論の余地あり）
4. **ハイパーネットワーク的解釈**: テキストエンコーダが分類器を生成する柔軟性

> **補足: 「真のゼロショット」議論** — CLIP の事前学習データに ImageNet クラスの画像が含まれている可能性は当然ある（dog, cat の写真は無数に web にある）。なので「ImageNet ゼロショット」は厳密にはタスク非依存の事前学習からの転移であり、未見クラスを学んだわけではない。CLIP 論文も §6 でこの点を認めている。

---

## ゼロショット転移と他の評価プロトコルの比較

| プロトコル | 説明 | 強み | 弱み |
|---|---|---|---|
| **Zero-Shot** | プロンプトのみ、追加訓練なし | 即座に使える、汎用性 | タスク学習に上限 |
| **k-NN** (詳細: [[knn-evaluation-protocol]]) | 凍結特徴量 + 最近傍 | ハイパラ不要、表現の素質を測れる | データセット保持必要 |
| **Linear Probing** | 凍結特徴量 + 線形分類器 | 表現の線形分離性を測る | LR 等のチューニング必要 |
| **Few-Shot** | 数例から学習 | 実用に近い | サンプル選びでバラつく |
| **Fine-tuning** | 全層を再学習 | 最終性能が高い | 大量のラベル付きデータと計算が必要 |

CLIP は **zero-shot, linear probing, fine-tuning すべてで強い**ことを実証した。

---

## CLIP 以降のゼロショット転移の展開

CLIP がパラダイムを確立した後、ゼロショット転移は CV 基盤モデルの標準評価に：

- **OpenCLIP** (2022): 公開データで CLIP 再現、より大きなモデル
- **SigLIP / SigLIP 2** (Google, 2023/2024): 効率化、多言語化
- **PE** (Meta, 2024): 86B ペアで超大規模ゼロショット
- **DINOv3 + dino.txt** (2025): DINO 系統に後付けでテキスト整合性を追加し、ゼロショットを獲得
- **SAM** (Meta, 2023, [[entities/sam]] / [[sources/segment-anything]]): **セグメンテーションへのゼロショット転移**を確立。プロンプト（点・ボックス・テキスト）の設計でエッジ検出・オブジェクト提案・インスタンスセグメンテーション・テキスト → マスク等を解く。CLIP の「テキストプロンプト → 分類」と並列で、「画像/テキストプロンプト → マスク」というインターフェイス（[[concepts/promptable-segmentation]]）

「**ゼロショット ImageNet 精度**」は今や CV 基盤モデルの de facto 基本指標。同様に「**ゼロショット 23 データセット mIoU**」が SAM 系統の標準指標となった。

---

## ゼロショット転移が苦手なこと

CLIP 論文と後続研究で明らかになった限界：

1. **抽象的・体系的タスク**: 物体カウント、距離推定、形状認識
2. **新規ドメイン**: 医療画像、衛星画像（CLIP は分布外）
3. **手書き文字**（MNIST）: ウェブ画像にほぼ存在しない
4. **細粒度識別**: 似た車種、似た花種（注釈が必要なレベル）

これらでは追加の few-shot や fine-tuning が必要。

---

## 関連ページ

- [[sources/clip]] — ゼロショット転移を CV に持ち込んだ論文
- [[sources/segment-anything]] — セグメンテーションへゼロショット転移を拡張した論文
- [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] — 分類でのゼロショット転移ができる主要モデル
- [[entities/sam]] — セグメンテーションでのゼロショット転移ができるモデル
- [[concepts/contrastive-learning]] — CLIP のゼロショットを可能にする学習方式
- [[concepts/promptable-segmentation]] — SAM のゼロショットを可能にするタスクパラダイム
- [[concepts/weakly-supervised-pretraining]] — ゼロショット転移を可能にする事前学習の枠組み
- [[concepts/foundation-model]] — ゼロショット転移は基盤モデルの中核能力
- [[concepts/knn-evaluation-protocol]] — 代替的な評価プロトコル
