---
type: entity
entity_kind: model
aliases: [MAE, Masked Autoencoder]
tags: [self-supervised, vision-transformer, masked-image-modeling, fair, meta-ai]
related: ["[[concepts/masked-image-modeling]]", "[[concepts/self-supervised-learning]]", "[[concepts/vision-transformer]]", "[[concepts/denoising-autoencoder]]"]
sources: ["[[sources/mae]]"]
updated: 2026-05-25
---

# MAE（Masked Autoencoder）

## 概要

**MAE** = **Masked Autoencoder**。Kaiming He ら（Facebook AI Research, 現 Meta AI Research）が 2021 年 11 月に発表した、Vision Transformer（[[concepts/vision-transformer]]）向けのシンプルかつスケーラブルな自己教師あり学習法。

- 論文: "Masked Autoencoders Are Scalable Vision Learners"
- arXiv: 2111.06377（2021）→ CVPR 2022
- コード: <https://github.com/facebookresearch/mae>
- 詳細解説: [[sources/mae]] / 翻訳: [[translations/mae]]

「**画像の 75% をマスクして残り 25% から欠落画素を再構成する**」というシンプルな目的関数で、ViT-Huge を ImageNet-1K のみで 87.8% 精度まで押し上げ、CV における Masked Image Modeling（MIM, [[concepts/masked-image-modeling]]）の中核手法となった。

---

## アーキテクチャ

非対称な encoder-decoder。これが MAE の最大の特徴。

```
[入力画像 224×224]
   ↓ patchify (16×16 パッチ、計 196)
[196 トークン]
   ↓ ランダムシャッフル + 75% 切り捨て
[49 可視トークン]
   ↓ encoder（ViT-B/L/H、フル ViT）
[49 エンコード済みトークン]
   ↓ + mask token × 147、unshuffle、位置埋め込み追加
[196 トークン]
   ↓ decoder（軽量 Transformer、8 ブロック × 512-d）
[196 再構成パッチ]
   ↓ MSE 損失（マスク位置のみ、正規化画素）
```

### Encoder

- 標準的な ViT（B/L/H/14 or 16）をそのまま使用
- **入力は可視パッチ（25%）のみ**。mask token は入力しない
- これにより計算とメモリが約 4 分の 1 に削減され、ViT-H クラスのスケールを実現

### Decoder

- 軽量 Transformer（既定: 8 ブロック × 幅 512-d）
- ViT-L に対して**トークンあたり FLOPs は 9% 未満**
- 入力: encoded visible patches + learnable mask tokens
- 事前学習後は**完全に破棄**される

### マスクトークン

- 単一の学習可能な D 次元ベクトル
- すべてのマスク位置で同じものを使用
- 位置埋め込みでどこのマスクかを区別

---

## 学習設定（事前学習）

| 項目 | 値 |
|---|---|
| 学習データ | ImageNet-1K（ラベルなし） |
| マスク率 | **75%** |
| マスクサンプリング | ランダム（uniform）|
| 再構成ターゲット | **正規化画素**（パッチ内 mean 0, std 1）|
| 損失 | MSE（マスク位置のみ）|
| Encoder | ViT-B/16 / L/16 / H/14 |
| Decoder | 8 ブロック, 512-d, 16 heads |
| Optimizer | AdamW（β₂ = 0.95）|
| Base learning rate | 1.5e-4（linear scaling: lr = base_lr × bs / 256）|
| Weight decay | 0.05 |
| Batch size | 4096 |
| LR schedule | cosine decay, 40 warmup epochs |
| 事前学習エポック | **1600**（多くの実験は 800） |
| データ拡張 | RandomResizedCrop のみ（色ジッタ不要）|

---

## 主要結果

### ImageNet-1K Fine-tuning（IN1K のみ）

| Model | Encoder Params | Fine-tune Acc |
|---|---|---|
| MAE ViT-B/16 | 86M | 83.6 |
| MAE ViT-L/16 | 304M | 85.9 |
| MAE ViT-H/14 | 632M | 86.9 |
| **MAE ViT-H/14 @ 448** | 632M | **87.8** |

→ IN1K のみで当時の最高精度。BEiT, MoCo v3, DINO すべてを上回る。

### Linear Probing（凍結 + 線形）

| Model | Linear Probing |
|---|---|
| iGPT-XL (6.8B) | 72.0 |
| BEiT ViT-L | 52.1 |
| MoCo v3 ViT-L | 77.6 |
| **MAE ViT-H** | **76.6** |

→ 対比学習 MoCo v3 にはやや劣るが、MIM 手法の中ではトップ。

### Partial Fine-tuning（重要な発見）

| Tuned blocks | MAE ViT-L | MoCo v3 ViT-L |
|---|---|---|
| 0（linear） | 73.5 | 77.6 |
| 1 | **81.0** | 79.5 |
| 4 | **83.7** | 81.1 |
| 24（full） | 85.9 | 84.1 |

→ **1 ブロック fine-tune するだけで linear probing から +7.5% 跳ね上がる**。MAE の表現は「線形には整理されていないが非線形に強い」性質を実証。

### Transfer Learning

| Task | MAE ViT-L | supervised ViT-L | Δ |
|---|---|---|---|
| COCO AP^box | 53.3 | 49.3 | **+4.0** |
| COCO AP^mask | 47.2 | 43.9 | **+3.3** |
| ADE20K mIoU | 53.6 | 49.9 | **+3.7** |
| iNat 2018 (ViT-H₄₄₈) | 86.8 | – | prev best 81.2 |

### Robustness

ViT-H/448 で:
- ImageNet-A: **76.7%** (supervised 33.1%)
- ImageNet-R: **66.5%** (supervised 50.3%)
- ImageNet-Sketch: **50.9%** (supervised 38.0%)

SSL の OOD ロバストネスの強さを最初に明示。

---

## 公開モデル

GitHub <https://github.com/facebookresearch/mae> で以下が公開:

| Model | Encoder | Decoder | Fine-tune Acc | URL |
|---|---|---|---|---|
| MAE ViT-B/16 | 86M | 8 blocks × 512-d | 83.6 | 公式リポジトリ |
| MAE ViT-L/16 | 304M | 8 blocks × 512-d | 85.9 | 公式リポジトリ |
| MAE ViT-H/14 | 632M | 8 blocks × 512-d | 86.9 | 公式リポジトリ |

各モデルとも:
- **事前学習済み重み**（encoder のみ）
- **fine-tuned 重み**（IN1K 分類ヘッド付き）

下流応用では encoder 重みのみをロードして使う。

---

## 系譜と派生

### 先行研究（MAE が乗り越えた手法）
- **Context Encoder**（Pathak 2016）: CNN ベースの inpainting
- **iGPT**（Chen 2020）: 画素を 1 次元系列として自己回帰生成
- **ViT MJP**（Dosovitskiy 2020）: ViT 論文内で masked patch prediction を簡易検証
- **BEiT**（Bao 2021）: dVAE 離散トークン予測
- **MoCo v3**（Chen 2021）: ViT 対応の対比学習

### 後続・派生
- **SimMIM**（Xie 2022）: MAE をさらに単純化、tokenizer も lightweight decoder も使わない
- **MAE Video / VideoMAE**: 動画への拡張
- **MAE 系統が iBOT に統合される**: [[entities/ibot]] が DINO + MIM ハイブリッドとして MAE のアイデアを継承
- **MaskFeat**（Wei 2022）: HOG 特徴量を再構成ターゲットに
- **AudioMAE**（Huang 2022）: 音声スペクトログラム版

---

## DINO 系統との関係（補完性）

| | MAE | DINO / iBOT |
|---|---|---|
| 学習信号 | 画像内マスク再構成 | 画像間の自己蒸留（DINO）+ MIM（iBOT）|
| 強い領域 | Fine-tuning、密予測 | 凍結特徴量、k-NN、画像検索 |
| 弱い領域 | Linear probing、k-NN | Linear / fine-tune の両立 |
| 再構成ターゲット | 画素 | teacher 出力（"online tokenizer"） |
| データ拡張 | ほぼ不要（masking が拡張役）| 重要（multi-crop, color jit など）|
| Emergent property | なし（特殊な性質は持たない） | 自己注意マップにオブジェクト境界が現れる |

両者は本質的に**補完的**で、iBOT 以降の研究はこの 2 系統を統合する方向に進んだ。

---

## なぜ重要か

1. **CV における MIM の中核手法**: MAE 以降、MIM は CV SSL の主要 2 系統（識別型 vs MIM）の片翼に確立。後の iBOT/DINOv2/DINOv3 にも MIM 損失として組み込まれる
2. **「シンプル + スケール」の典型**: NLP の BERT/GPT 流の思想を CV に持ち込んだ最も影響力ある論文
3. **75% マスク率という反直観的発見**: 後続のすべての MIM 研究のデフォルト値
4. **非対称設計のエンジニアリング**: 大規模 SSL を可能にした計算効率の改善
5. **後続 foundation model の画像エンコーダ初期化に活用**: [[entities/sam]]（SAM, Meta 2023）は MAE 事前学習済み ViT-H/16 を画像エンコーダの初期値として採用（[[sources/segment-anything]] §A）。[[entities/sam-2]]（SAM 2, Meta 2024）は **MAE 事前学習済み Hiera**（[[entities/hiera]]）を採用し、SAM v1 比 6× 高速化を達成（[[sources/sam-2]] §C.2.1）。MAE は識別タスクで CLIP より弱いとされたが、SAM/SAM 2 のような **密予測 foundation model の出発点としては最良** という評価が確立した

---

## 関連ページ

- [[sources/mae]] — 詳細解説
- [[translations/mae]] — 全文和訳（Appendix 込み）
- [[concepts/masked-image-modeling]] — MAE が代表する系統
- [[concepts/denoising-autoencoder]] — 理論的祖先
- [[concepts/self-supervised-learning]] — SSL 全般
- [[concepts/vision-transformer]] — バックボーン
- 系譜的関連: [[entities/dino]] / [[entities/ibot]] / [[entities/dinov2]] / [[entities/dinov3]]
- データセット: [[entities/imagenet]]
- 応用先: [[entities/sam]] — SAM が MAE 事前学習済み ViT-H/16 を画像エンコーダ初期化に採用
- 応用先: [[entities/sam-2]] / [[entities/hiera]] — SAM 2 が MAE 事前学習済み Hiera を画像エンコーダに採用（plain ViT より 6× 高速化）
