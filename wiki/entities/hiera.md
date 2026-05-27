---
type: entity
entity_kind: model
aliases: [Hiera, Hierarchical Vision Transformer]
tags: [model, architecture, vision-transformer, hierarchical, mae, meta-fair]
related: [[concepts/vision-transformer]], [[concepts/masked-image-modeling]], [[entities/mae]]
sources: [[sources/sam-2]]
updated: 2026-05-26
---

# Hiera（Hierarchical Vision Transformer）

## 概要

**Hiera** = **Hierarchical Vision Transformer**。Meta AI FAIR の Ryali ら（2023）が提案した、**「bells-and-whistles なし」のシンプルな階層型 vision transformer**。Swin Transformer などの先行階層型 ViT が様々な工夫（shifted window, relative position bias 等）を加えていたのを、**MAE 事前学習の力で削ぎ落とす**ことに成功した。SAM 2（[[entities/sam-2]] / [[sources/sam-2]]）の画像エンコーダとして採用されたことで、CV 基盤モデル分野で再評価されている。

- 論文: "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles"
- 出典: ICML 2023
- arXiv: 2306.00989
- コード: <https://github.com/facebookresearch/hiera>
- 採用: SAM 2 の画像エンコーダ

> **補足: なぜ「bells-and-whistles なし」が重要か** — Swin Transformer や Pyramid ViT などの階層型 ViT は、画像分類精度を上げるために shifted window attention・relative position bias・cross-shape window 等の多くの工夫を積み重ねていた。これらは実装が複雑で速度を遅くする。Hiera は「**MAE で適切に事前学習すれば、これらの工夫はすべて不要**」を実証した。

## なぜ階層型 ViT が必要か

[[concepts/vision-transformer]]（ViT）原典の plain ViT は**全層が同じ解像度・チャネル次元**。一方 CNN（ResNet 等）は階層的：

- 浅い層: 高解像度・低チャネル（細部の特徴）
- 深い層: 低解像度・高チャネル（意味的な特徴）

この階層性は **密予測タスク**（セグメンテーション、検出、深度推定）に重要で、**多スケール特徴量を FPN（Feature Pyramid Network）で融合**することで性能が大幅に上がる。

しかし、plain ViT は階層的でないため：
- そのままでは多スケール特徴が出ない
- SAM v1 は ViT-H/16 を windowed attention で部分的に修正していた

Hiera は ViT-MAE の力を活かしながら階層性を持つ設計。

## アーキテクチャ

### 4 ステージ構成

| Stage | 解像度（入力 224×224 の場合） | チャネル次元 | Attention |
|---|---|---|---|
| 1 | 56×56（stride 4） | 96 | Pool attention（局所） |
| 2 | 28×28（stride 8） | 192 | Pool attention |
| 3 | 14×14（stride 16） | 384 | **Global attention**（一部） |
| 4 | 7×7（stride 32） | 768 | **Global attention** |

各ステージ間で **空間 pooling**（2×2）+ チャネル次元倍増（PatchEmbed と類似だが学習可能）。

### Pool Attention

Hiera の中核設計：

1. 入力トークン: `(N, L, C)`
2. **Q をそのまま保持**（解像度は L のまま）
3. **K, V を pool で縮小**（max pool 2×2）してから attention
4. 注意計算の計算量 O(L²) → O(L · L/4) に削減

これにより Stage 1, 2 の高解像度でも attention が現実的に計算可能。

### Shifted window や relative position bias は **不要**

Swin と異なり：
- ❌ Shifted window attention
- ❌ Relative position bias（RPB）
- ❌ Cross-shape window
- ✅ シンプルな pool attention のみ
- ✅ Absolute positional encoding

**MAE 事前学習で十分に強い表現を学習できるので、アーキテクチャの工夫が不要になった** というのが核心主張。

## モデルファミリー

| 名前 | パラメータ数 | 用途 |
|---|---|---|
| **Hiera-T**（Tiny） | 27M | 軽量モデル、SAM 2 T |
| **Hiera-S**（Small） | 35M | SAM 2 S |
| **Hiera-B**（Base） | 51M | 画像分類ベンチマーク |
| **Hiera-B+**（Base Plus） | 70M | **SAM 2 デフォルト** |
| **Hiera-L**（Large） | 213M | SAM 2 L |
| **Hiera-H**（Huge） | 672M | 最大、論文の主要モデル |

## 主要結果（Hiera 原論文）

ImageNet-1k 分類で、**MAE 事前学習 → ファインチューン** の設定で：

| モデル | パラメータ | IN1k Top-1 | 速度 |
|---|---|---|---|
| ViT-B（MAE） | 86M | 83.6 | baseline |
| Swin-B | 88M | 83.5 | やや遅い |
| **Hiera-B** | 51M | **84.3** | **より速い** |
| ViT-L（MAE） | 304M | 85.9 | - |
| **Hiera-L** | 213M | **86.1** | **より速い** |
| **Hiera-H** | 672M | **86.9** | - |

**より少ないパラメータでより高精度かつ高速**。動画分類（K400, K600, K700）でも SOTA。

## SAM 2 での利用

SAM 2（[[entities/sam-2]]）は Hiera を画像エンコーダに採用：

1. **MAE で事前学習**: ImageNet-1k での 1600 epoch MAE 訓練
2. **FPN で多スケール特徴融合**: Stage 3 (stride 16) と Stage 4 (stride 32) を融合して画像埋め込みを生成
3. **Stage 1, 2 の高解像度特徴**（stride 4, 8）はマスクデコーダのアップサンプリング層に skip 接続として使用 → 高解像度マスク予測に寄与
4. **Global attention は一部の層のみ**: 計算効率のため
5. **RPB を削除**: SAM v1 の ViT 風 RPB → Hiera 論文の「absolute-win」位置エンコーディング → SAM 2 はさらにシンプルにグローバル位置埋め込みの補間を採用

> **補足: SAM 2 で plain ViT を捨てた理由** — SAM v1 は ViT-H で運用していたが、動画ストリーミングには計算量が重すぎる（1 フレームあたり 21.7 FPS）。Hiera-B+ なら 130 FPS、しかも同じ SA-1B で訓練しても精度が上回る（58.1 → 58.9）。**「Hiera は plain ViT の上位互換」** という発見が SAM 2 の重要な技術選択。

## 他の階層型 ViT との関係

| モデル | 階層化方式 | 工夫 | MAE 互換 |
|---|---|---|---|
| **Swin Transformer**（2021） | Window attention + shift | Shifted window, RPB, hierarchical | ❌（窓構造が MAE と相性悪い） |
| **PVT / PVTv2** | Spatial Reduction Attention | RPB, MIT 風設計 | △ |
| **MViT v1/v2** | Pool attention の先駆 | Decomposed RPB | ✅ |
| **Hiera**（2023） | **Pool attention のみ** | **工夫なし** | ✅（MAE 前提） |

Hiera は MViT v2 の発展形と見るのが正確。「**MViT v2 から余計な工夫を全部削除して MAE に賭けたもの**」が Hiera。

## なぜ重要か

1. **MAE 事前学習の力の実証**: アーキテクチャ工夫より事前学習の質が重要、という現代 SSL の主張をシンプルに示した
2. **SAM 2 の画像エンコーダ採用**: 動画 foundation model 時代の標準アーキテクチャになりつつある
3. **多スケール特徴の階層型 ViT**: 密予測タスクで plain ViT を凌駕する一方、MAE 互換性も持つ稀有な設計
4. **シンプルさ**: 実装が容易（pool + attention のみ）、デバッグしやすい

## 限界

1. **MAE 事前学習に強く依存**: 教師あり事前学習だけだと先行階層型 ViT に劣ることが多い
2. **動画専用設計ではない**: Hiera 原論文は動画分類も扱うが、SAM 2 のような時間メモリは持たない（メモリは SAM 2 側の追加）
3. **Pool attention のメモリ非局所性**: max pool は K, V の情報を失う面もある
4. **MViT v2 との差別化が微妙**: Pool attention 自体は MViT 由来

## 関連ページ

- [[sources/sam-2]] — Hiera を採用した SAM 2 の原典
- [[entities/sam-2]] — Hiera を画像エンコーダとして使う
- [[concepts/vision-transformer]] — Hiera が拡張する元の ViT
- [[concepts/masked-image-modeling]] — Hiera を活かす MAE 系の事前学習
- [[entities/mae]] — Hiera の事前学習に使用
- 比較: Swin Transformer / MViT v2 / PVTv2（独立ページなし）
