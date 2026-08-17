---
type: entity
entity_kind: model
aliases: [MaxViT, Max-SA, multi-axis attention, grid attention]
tags: [maxvit, vision-transformer, hybrid, window-attention, grid-attention, backbone, google-research]
related: ["[[entities/swin-transformer]]", "[[concepts/vision-transformer]]", "[[concepts/convolutional-neural-network]]", "[[entities/convnext]]", "[[entities/hiera]]"]
sources: ["[[sources/maxvit]]"]
updated: 2026-06-17
---

# MaxViT

Google Research + UT Austin の **Zhengzhong Tu, Yinxiao Li ら**によるハイブリッド視覚バックボーン（arXiv:2204.01697, **ECCV 2022**）。詳細解説: [[sources/maxvit]] / 翻訳: [[translations/maxvit]]。

## 一言で

**「窓の中で attention（局所）」+「等間隔に間引いた画素どうしで attention（大域）」+ MBConv を 1 ブロックに統合し、線形計算量のまま第 1 層から大域的視野を持つ**バックボーン。名前は **M**ulti-**ax**is **Vi**sion **T**ransformer の略。[[entities/swin-transformer|Swin]] の attention と**パラメータ数・FLOPs が完全に同一の drop-in 置換**でありながら、cyclic shift もマスキングも不要。

## アーキテクチャ

### Max-SA（多軸自己注意）— 中核

同じ「空間軸を分解する」発想の 2 通りの切り方を組み合わせる。$P=G=7$（Swin の窓サイズに合わせる）が既定。

| | テンソル形状 | attention する軸 | 役割 |
|---|---|---|---|
| **block attention** | $(\frac{H}{P}\times\frac{W}{P},\ P\times P,\ C)$ | 窓の中（$P^2$） | 局所（Swin の W-MSA 相当） |
| **grid attention** | $(G\times G,\ \frac{H}{G}\times\frac{W}{G},\ C)$ | 格子点間（$G^2$） | **大域（dilated）** |

$P$、$G$ を固定するため**どちらも入力サイズに対して線形**。両者の計算量が完全に釣り合う。

### MaxViT ブロック

```
入力
  ├─ MBConv          ← Conv1×1 拡大 → DWConv3×3 → SE → Conv1×1 縮小 + 残差
  │                     （depthwise conv が CPE として働き、位置埋め込みが不要になる）
  ├─ Block Attention ← Block-SA + 残差 → FFN + 残差   （局所）
  └─ Grid Attention  ← Grid-SA + 残差 → FFN + 残差    （大域）
```

- **順序が重要**: C-BA-GA（畳み込み → 局所 → 大域）が最良。MBConv を後ろに置くと -0.54〜-0.60。**ただし生成タスクでは逆順 GA-BA-C が良い**（FID 30.77 vs 31.40）
- **逐次であることが重要**: 並列設計にすると **-0.98〜-1.63** と大きく劣化する
- relative attention（相対位置バイアス付き）、head 次元 32、MBConv 拡大率 4、SE 縮小率 0.25
- **[CLS] トークンなし**（最終段階の大域平均プーリング）

### 全体構造

ResNet 流の 4 段階階層。S0 Conv ステム（Conv3×3 stride 2 + Conv3×3）→ S1（1/4）→ S2（1/8）→ S3（1/16）→ S4（1/32）。各段階の最初の MBConv でダウンサンプリング。

## バリアント

**表**: MaxViT の構成（$B$ = ブロック数、$C$ = チャネル数）と ImageNet 性能（[[sources/maxvit]] 表1 / 表2 / 表3 より）

| モデル | S1 | S2 | S3 | S4 | params | FLOPs (224²) | IN-1K 224² | IN-1K 512² | IN-21K→1K | JFT→1K |
|---|---|---|---|---|---|---|---|---|---|---|
| **MaxViT-T** | B2 C64 | B2 C128 | B5 C256 | B2 C512 | 31M | 5.6G | 83.62 | 85.72 | — | — |
| **MaxViT-S** | B2 C96 | B2 C192 | B5 C384 | B2 C768 | 69M | 11.7G | 84.45 | 86.19 | — | — |
| **MaxViT-B** | B2 C96 | B6 C192 | B14 C384 | B2 C768 | 120M | 23.4G | 84.95 | 86.66 | **88.38** | 88.82 |
| **MaxViT-L** | B2 C128 | B6 C256 | B14 C512 | B2 C1024 | 212M | 43.9G | 85.17 | **86.70** | 88.46 | 89.41 |
| **MaxViT-XL** | B2 C192 | B6 C384 | B14 C768 | B2 C1536 | 475M | — | — | — | **88.70** | **89.53** |

- **段階配分が Swin と違う**: Swin/ConvNeXt の {2,2,18,2} に対し MaxViT-B/L/XL は {2,6,14,2}。**小モデルでは差はないが、大モデルで MaxViT のレイアウトが大きく優位**（図5）
- **XL は IN-21K / JFT 事前学習のみ**で提供

## 主要結果

- **ImageNet-1K（追加データなし）**: MaxViT-L 512² で **86.70%** — 当時の通常訓練 SOTA。224² でも MaxViT-L 85.17（CoAtNet-3 に +0.67）
- **ImageNet-21K → 1K**: MaxViT-B **88.38%** が **CoAtNet-4 をパラメータ 43% / FLOPs 38% で +0.28 上回る**。MaxViT-XL 512² で **88.70%**
- **JFT-300M → 1K**: MaxViT-XL **89.53%**
- **COCO 検出**（Cascade Mask R-CNN, 896²）: MaxViT-B **53.4 AP / 45.7 AP^m**。**MaxViT-S が約 40% 少ない計算量で Swin-B / UViT-B を上回る**
- **画像美観評価（AVA）**: MaxViT-T 512² で PLCC **0.745** / SRCC **0.708**（多重解像度の MUSIQ 0.720/0.706 超え）
- **無条件画像生成（GAN）**: FID **30.77** / IS **22.58** を **18.6M**（HiT 32.9M の約半分）で達成

## 系譜・位置づけ

- **直接の比較対象は [[entities/swin-transformer]]**。Max-SA は Swin の attention の drop-in 置換（同パラメータ・同 FLOPs）で、**cyclic shift とマスクという Swin の実装上の複雑さを不要にした**。
- **もう一つの比較対象は CoAtNet**（Google の先行ハイブリッド、本 wiki 未取り込み）。MaxViT は同等以上の精度をより少ないパラメータで達成した。
- **[[entities/convnext]] とは同時期の別解**。ConvNeXt が「ConvNet を近代化して Transformer に追いつく」なら、MaxViT は「**畳み込みと attention を 1 ブロックに統合する**」ハイブリッド路線。ConvNeXt V2（[[sources/convnext-v2]] 表5）は MaxViT-XL を 88.5/88.7 として比較対象に挙げている。
- **[[entities/hiera]] とは設計思想が対極**。Hiera が「MAE で事前学習すれば工夫は全部要らない」と削ぐ方向なのに対し、MaxViT は「**工夫を足して 1 ブロックに統合する**」方向。
- **その後の主流にはならなかった**。CLIP / DINOv2 / MLLM の視覚エンコーダは plain ViT に収束しており、ハイブリッド路線は分類ベンチマークでは強かったが、**言語との接続性や SSL との相性という軸では選ばれなかった**。

## 弱点

- **スループットが遅い**: MaxViT-T 349.6 img/s に対し Swin-T 755.2 / ConvNeXt-T 774.7 と**2 倍以上遅い**。MBConv + 2 種 attention + FFN×2 という 3 段構えのブロックは重い
- **パラメータ効率**: IN-1K のみの領域では ConvNeXt / Swin の方が良い（MaxViT-B 120M/84.95 vs ConvNeXt-B 89M/83.8）。効率が逆転するのは IN-21K 以降
- **最高スコアは非公開データ依存**: 89.53% は JFT-300M（Google 内部）で再現不可
- **「単純さ」の主張は相対的**: 1 ブロックに MBConv + SE + block attention + grid attention + FFN×2 が入る

## 公開

- コード / 重み: <https://github.com/google-research/maxvit>（Apache 2.0）
- `timm` に `maxvit_tiny_tf_224` 等として収録
- 派生: **MaxViT-GAN**（生成器版、ブロック順序を GA-BA-C に反転）、**Swin-Mixer 同様に MLP 版への一般化も可能**と論文は示唆

## 関連ページ

- [[sources/maxvit]] — 論文要約（最重要）
- [[translations/maxvit]] — 全文和訳（Appendix 0.A〜0.C 込み）
- [[entities/swin-transformer]] / [[sources/swin-transformer]] — 最大の比較対象
- [[entities/convnext]] / [[sources/convnext]] — 同時期の ConvNet 側の回答
- [[concepts/vision-transformer]] — 出発点の ViT
- [[concepts/convolutional-neural-network]] — MBConv / SE / inverted bottleneck の背景
- [[entities/hiera]] — 設計思想が対極（削ぐ vs 統合する）
