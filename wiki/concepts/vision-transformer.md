---
type: concept
aliases: [ViT, Vision Transformer]
tags: [architecture, transformer, image-classification]
related: ["[[self-supervised-learning]]", "[[knowledge-distillation]]", "[[rotary-position-embeddings]]"]
sources: ["[[sources/vision-transformer]]", "[[sources/dino-emerging-properties-in-self-supervised-vit]]"]
updated: 2026-06-03
---

# Vision Transformer（ViT）

## 一言で

**画像を 16×16 のパッチに切り分け、各パッチを「単語」のように扱って Transformer に流し込むだけの画像分類モデル**。Dosovitskiy ら（Google Brain）が 2020 年に発表（**[[sources/vision-transformer|"An Image is Worth 16x16 Words", ICLR 2021]]** — 原典）。それまで CV を支配していた CNN（畳み込みニューラルネット）の代替として、十分な事前学習データがあれば CNN 並みかそれ以上の精度を出せることを示した。**ImageNet 88.55% / VTAB 77.63% を達成し、CNN の 1/4 以下の計算（2.5k vs 9.9k TPUv3-core-days）で SOTA**。本論文以降、CV の主要モデルはほぼすべて ViT バックボーンに移行した。

## モデル構造

```
入力画像 (224×224×3)
    │
    ├─→ 16×16 パッチに分割 (14×14 = 196 パッチ)
    │
    ├─→ 各パッチを線形射影して埋め込み (パッチ埋め込み, dim=D)
    │
    ├─→ 先頭に [CLS] トークン（学習可能ベクトル）を連結 → 197 トークン
    │
    ├─→ 各トークンに位置埋め込み（learnable）を加算
    │
    ├─→ Transformer エンコーダ × N 層
    │     各層 = LayerNorm → Multi-Head Self-Attention → 残差
    │            LayerNorm → MLP → 残差
    │
    └─→ [CLS] トークンの出力ベクトル
          ↓
        分類ヘッド（線形）
```

### 各構成要素の補足

- **パッチ分割（patchify）**: 画像 $H \times W \times 3$ を $N \times N$ のグリッド（$N=16$ なら 14×14=196 個）に切る。実装は stride $N$ の畳み込み 1 層と等価。
- **[CLS] トークン**: BERT 由来の特殊トークン。系列全体の情報を集約させる「読み出し口」として学習されるベクトル。最終層の [CLS] トークンの出力を分類ヘッドに入れる。
- **位置埋め込み（positional embedding）**: Transformer は系列の順序情報を持たないので、各位置に学習可能なベクトルを足す。1D（パッチを並べた順）でも 2D（縦横の座標）でも経験的にあまり差はないと報告されている。
- **Multi-Head Self-Attention（MHSA, 多頭自己注意）**: トークン同士の関連度（attention）を計算して情報を混ぜる。複数の「ヘッド」を並列に持ち、それぞれ異なる関係性に着目できる。
- **pre-norm**: LayerNorm を残差ブロックの**前**に置く構造。学習の安定性が高い。

## ViT の変種命名規約

論文・実装で `ViT-S/16` `ViT-B/8` のような表記が頻出する。意味は:

- **S / B / L / H**: モデルサイズ。Small / Base / Large / Huge。表で比較:

| Model | Layers | Hidden dim | Heads | #params |
|---|---|---|---|---|
| ViT-S | 12 | 384 | 6 | 21M |
| ViT-B | 12 | 768 | 12 | 86M |
| ViT-L | 24 | 1024 | 16 | 307M |
| ViT-H | 32 | 1280 | 16 | 632M |

- **/16, /8, /14**: パッチサイズ。`/8` だと 224×224 入力で 28×28 = 784 トークン、`/16` だと 14×14 = 196 トークン。**パッチが小さいほどトークン列が長くなり、計算量はトークン数の 2 乗で増える**が、細かい構造を捉えられる。

## CNN との比較（直観）

| | CNN | ViT |
|---|---|---|
| 帰納バイアス | 局所性・平行移動不変性が組み込み済み | ほぼなし（位置埋め込みのみ） |
| データ要求量 | 中（ImageNet 程度で OK） | **大**（JFT-300M 級が望ましい） |
| 受容野 | 層を重ねて徐々に広がる | 第 1 層から大域 |
| ピクセル間相互作用 | 局所の畳み込みカーネル | 全トークン間の attention |

> **補足: 帰納バイアスとデータ要求量** — CNN は「画像の近くのピクセルが関係する」という仮定（局所性）と「位置がずれても物体は同じ」という仮定（平行移動不変性）を**構造として持っている**。ViT はそれらをほぼ持たないので、その仮定を**データから学ぶ**必要があり、結果として大量のデータが必要になる。これが「ViT は事前学習が肝心」となる理由。

## ViT の弱点と工夫

- **データ飢餓**: ImageNet 単独学習では CNN に劣る。JFT-300M（Google 内部の 3 億枚データセット）級で事前学習するか、強い拡張・蒸留（DeiT）、あるいは SSL（DINO, MAE）で補う。
- **計算量**: トークン数 $n$ に対し self-attention は $O(n^2)$。高解像度や小パッチで急速に重くなる。
- **空間的階層性の欠如**: CNN のような U-Net 的階層を持たないので、密予測（セグメンテーション・検出）には Swin Transformer 等の派生が好まれる。

## なぜ ViT が SSL とこんなに相性が良いのか

DINO 論文 [[sources/dino-emerging-properties-in-self-supervised-vit]] の最大の発見は、ViT を SSL で事前学習すると **CNN にも教師あり ViT にも現れない創発的特性**（自己注意マップへのセグメンテーション情報の出現、k-NN 性能の異常な良さ）が出るという点。直観的な説明候補:

1. **帰納バイアスの弱さ**: CNN が「持っている」前提を ViT は持っていない分、SSL の信号からデータの構造をより自由に学べる。
2. **トークン構造の柔軟性**: マスキング（MAE）、トークン取捨選択（DINOv2）など、トークン単位の操作が自然に組み込める。これは画素を扱う CNN では難しい。
3. **[CLS] トークンの集約性**: 学習可能な「読み出し口」が、教師信号なしでもオブジェクト中心の情報を集めるよう自己組織化する（DINO の図 1 の現象）。

## 関連ページ

- [[sources/vision-transformer]] — **ViT 原典論文**（Dosovitskiy et al., Google Brain, ICLR 2021）。本概念ページの原典
- [[translations/vision-transformer]] — 原典の本文和訳
- [[sources/dino-emerging-properties-in-self-supervised-vit]]: ViT × SSL の代表論文
- [[concepts/self-supervised-learning]]: SSL の枠組み全般
- [[concepts/rotary-position-embeddings]]: ViT の学習可能 1D 位置埋め込みを置き換える RoPE（CV では DINOv3 / SAM 2 / Qwen2-VL 系で採用）
- [[entities/dino]]: DINO 手法のスペック
