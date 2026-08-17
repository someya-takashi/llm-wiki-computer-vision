---
type: question
asked: 2026-06-17
question: "ConvNeXt と ConvNeXt V2 にはアーキテクチャの違いはあるでしょうか？それとも単に V1 に FCMAE を行ったものを V2 としているのでしょうか？"
sources_used: ["[[sources/convnext]]", "[[sources/convnext-v2]]", "[[entities/convnext]]", "[[entities/convnext-v2]]", "[[concepts/masked-image-modeling]]", "[[entities/mae]]", "[[entities/hiera]]"]
---

# ConvNeXt V1 と V2 のアーキテクチャの違い — 「V2 = V1 + FCMAE」なのか？

## 結論を先に

**アーキテクチャの違いはあります。** ただし変更は極小（GRN 層を 1 つ足して LayerScale を消すだけ）です。

そして重要なのは、**「単に V1 に FCMAE を行ったもの」という理解は、[[sources/convnext-v2|論文が明示的に否定した仮説]]**だという点です。著者らはまずそれを試して**失敗**を報告し、原因を突き止め、アーキテクチャ側を修正することで解決しています。

$$\text{V2} = \underbrace{\text{V1} + \text{GRN}}_{\text{アーキテクチャ変更}} \;\;\text{を}\;\; \underbrace{\text{FCMAE}}_{\text{事前学習}} \;\text{で訓練したもの}$$

---

## 1. アーキテクチャの差分は 2 点だけ

<figure>

![](../../raw/assets/convnext-v2/fig5.png)

<figcaption>図5（[[sources/convnext-v2]] より再掲）: ConvNeXt V1 ブロック（左）と V2 ブロック（右）。V2 は次元拡大 MLP（1×1, 384）の GELU 直後に GRN を追加し、冗長になった LayerScale を削除している。それ以外は完全に同一。</figcaption>
</figure>

```
V1: d7×7 → LN → 1×1(C→4C) → GELU ──────────→ 1×1(4C→C) → ×LayerScale → +残差
V2: d7×7 → LN → 1×1(C→4C) → GELU → ★GRN ──→ 1×1(4C→C) ─────────────→ +残差
```

| 項目 | 変更 |
|---|---|
| **GRN 層** | **追加**（次元拡大 MLP の GELU 直後） |
| **LayerScale** | **削除**（GRN があると冗長になるため） |
| stem（patchify 4×4 s4） | 同一 |
| stage 構成・独立ダウンサンプリング層 | 同一 |
| depthwise conv 7×7 の位置 | 同一 |
| LN 1 個・GELU 1 個という部品数 | 同一 |

**GRN は実装 3 行、追加パラメータ・FLOPs は実質ゼロ**なので、「ほぼ同じアーキテクチャ」という理解自体は間違っていません。詳細は [[entities/convnext-v2]]。

---

## 2. しかし「V1 + FCMAE」では性能が出ない — ここが論文の核心

[[sources/convnext-v2]] の表 3 が論文の主張そのものです。

**表**: co-design の効果（ImageNet-1K fine-tuning 精度、800 epoch FCMAE 事前学習）

| Backbone | 事前学習 | IN-1K | 読み方 |
|---|---|---|---|
| ConvNeXt V1-B | 教師あり 300ep | 83.8 | ベースライン |
| **ConvNeXt V1-B** | **FCMAE** | **83.7** | ← **質問の仮説がこれ。教師ありにすら届かない** |
| ConvNeXt V2-B | 教師あり | 84.3 (+0.5) | GRN だけでは効果小 |
| **ConvNeXt V2-B** | **FCMAE** | **84.6 (+0.8)** | 両方揃って初めて効く |
| ConvNeXt V1-L | 教師あり | 84.3 | |
| ConvNeXt V1-L | FCMAE | 84.4 | ← やはりほぼ効かない |
| ConvNeXt V2-L | 教師あり | 84.5 (+0.2) | |
| **ConvNeXt V2-L** | **FCMAE** | **85.6 (+1.3)** | **モデルが大きいほど差が開く** |

読み取るべきは 3 点です。

1. **FCMAE だけ（V1 + FCMAE）では効かない**: 83.8 → 83.7 と**むしろ微減**。MAE が ViT で教師ありを大きく上回るのとは対照的
2. **GRN だけ（V2 + 教師あり）でも効きが小さい**: +0.5 / +0.2
3. **両方揃うと大きく効く**: +0.8 / +1.3。しかもスケールするほど差が開く

著者らはこれを **co-design（共設計）** と呼び、「**自己教師あり手法は既存アーキテクチャに後から載せるもの**」という業界の慣行そのものへの異議申し立てとして位置づけています。

> 自己教師あり学習における一般的な慣行は、教師あり学習のために設計された*あらかじめ決められた*アーキテクチャを用い、その設計を固定されたものとみなすことである。（[[translations/convnext-v2]] §1）

---

## 3. なぜ GRN が必要だったのか — 特徴崩壊

V1 に FCMAE をかけると **特徴崩壊（feature collapse）** が起きます。

- **現象**: 次元拡大 MLP 層（1×1 で 4 倍に広げる箇所）で、**死んだ／飽和したチャネルが増え、チャネル間で活性化が冗長になる**
- **定量化**: チャネル間の平均対ごとコサイン距離を層ごとに測ると、V1+FCMAE が全層で最も低い（ViT+MAE や V1 教師ありより顕著に低い）
- **なぜ問題か**: すべてのチャネルが似た反応をするなら実質的なチャネル数が減っているのと同じで、表現容量が無駄になる

> **補足: SSL の「表現崩壊」とは別概念** — [[concepts/self-supervised-learning]] で扱う collapse は「入力によらず同じ表現を返す」自明解のこと。こちらは**チャネル方向の多様性が失われる**現象で、別物です。

**GRN（Global Response Normalization）** はチャネル間に競合を持ち込んでこれを防ぎます。

1. 各チャネルを空間方向の **L2 ノルム**でスカラー化（大域平均プーリングではだめ: 83.7 vs 84.6）
2. **除法正規化** $||X_i|| / \sum_j ||X_j||$ で「他チャネルと比べた相対的重要度」を出す
3. 元の応答に掛け戻す。$\gamma, \beta$ はゼロ初期化 + 残差なので、**最初は恒等写像として振る舞い訓練とともに効いてくる**

---

## 4. GRN は「FCMAE 用の一時的な仕掛け」ではない

質問の「単に FCMAE を行ったもの」という理解が成り立たない、もう一つの決定的な証拠がこれです。

**表**: GRN を事前学習／fine-tuning のどちらに入れるか（[[sources/convnext-v2]] 表2f）

| 設定 | IN-1K |
|---|---|
| ベースライン（GRN なし） | 83.7 |
| **fine-tuning で GRN を外す** | **78.8** |
| **fine-tuning でのみ GRN を足す** | **80.6** |
| **両方で使う** | **84.6** |

片方だけだと**ベースラインより大幅に悪化**します。つまり GRN は事前学習を助ける補助輪ではなく、**モデルの恒久的な一部**です。「事前学習が済んだら外せる」ものではありません。

---

## 5. 副次的な違い: モデルラインナップ

アーキテクチャ本体とは別に、提供されるサイズも変わっています。

| 帯域 | V1 | V2 |
|---|---|---|
| 超軽量 | — | **Atto 3.7M / Femto 5.2M / Pico 9.1M / Nano 15.6M** |
| 中核 | T (29M) / **S (50M)** / B (89M) / L (198M) | T (28.6M) / B (89M) / L (198M) |
| 最大 | **XL (350M, C=256)** | **Huge (659M, C=352)** |

- **T / B / L の $C$・$B$ 構成は V1 と完全に同一**
- **V2 に Small と XLarge は存在しない**。代わりに超軽量帯 4 種と Huge が追加された
- 超軽量帯は V1 論文由来ではなく別研究（Ross Wightman の小型 ConvNeXt 変種）から

これにより **MIM の恩恵が 3.7M〜660M という広範囲で実証された**（従来 MIM は Base 以上でしか恩恵が確認されていなかった）というのが V2 の副次的な貢献です。

---

## 6. なぜこの区別が重要か — wiki 全体の文脈で

本 wiki には「**MIM に合わせてアーキテクチャを直す**」という同じ発想の研究がもう 1 つあります。

| | アーキテクチャ側の変更 | 目的 |
|---|---|---|
| **[[entities/hiera]]**（ICML 2023） | 階層型 ViT から shifted window・RPB を**削除** | MAE 互換にする |
| **[[entities/convnext-v2]]**（CVPR 2023） | ConvNeXt に GRN を**追加** | FCMAE 互換にする |

**同年に、Transformer 側と ConvNet 側から同じ結論に到達している**のが興味深い点です。詳細は [[concepts/masked-image-modeling]] の「MIM とアーキテクチャの相性」表を参照。

またこの区別は、[[entities/convnext|V1]] のページに本 wiki が書いていた「**ConvNeXt は MIM と相性が悪い**」という記述の精度にも関わります。正確には:

- **V1 が MIM に向かないのは事実**（密なスライディングウィンドウがマスク領域を跨ぎ、情報漏洩でショートカット学習が起きる）
- **ただしそれは CNN の本質的限界ではなく V1 の設計上の制約**（疎畳み込み + GRN で解決できた）
- **その後も ViT が主流であり続けた理由は MIM 適性ではなく、言語との接続性**（CLIP / MLLM の視覚エンコーダ採用状況）

---

## 補足: FCMAE 側にも工夫がある

質問の焦点はアーキテクチャですが、FCMAE も「MAE をそのまま適用したもの」ではありません。

- **疎畳み込み（submanifold sparse convolution）で可視画素のみ処理** — これがないと 79.3、あると 83.7 で **+4.4**。情報漏洩の遮断がすべての前提
- **マスク率 0.6**（[[entities/mae]] の 75% より低い）、マスク単位 32×32
- **デコーダは単一の ConvNeXt ブロック**（UNet 比 1.7× 高速で精度同等）
- fine-tuning 時は特別な扱いなしに標準の密畳み込みへ戻せる

つまり **V1 → V2 は「アーキテクチャ側の最小変更」と「MAE 側の CNN 向け再設計」の両方**からなっています。

## 関連ページ

- [[sources/convnext]] / [[entities/convnext]] — V1。近代化ロードマップと 13 段階の対照実験
- [[sources/convnext-v2]] / [[entities/convnext-v2]] — V2。本ページの主要な出典
- [[translations/convnext-v2]] — V2 の全文和訳（Appendix A〜D 込み）
- [[concepts/masked-image-modeling]] — MIM とアーキテクチャの相性
- [[entities/mae]] — FCMAE の下敷き
- [[entities/hiera]] — 同じ co-design 発想の Transformer 側
- [[concepts/convolutional-neural-network]] — CNN の帰納バイアスと部品
