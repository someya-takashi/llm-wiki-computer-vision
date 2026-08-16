---
type: entity
entity_kind: model
aliases: [ConvNeXt, CNX, Convolutional Next]
tags: [convnext, convolutional-neural-network, architecture, backbone, fair, meta-ai]
related: ["[[concepts/convolutional-neural-network]]", "[[concepts/vision-transformer]]", "[[entities/dinov3]]", "[[entities/hiera]]", "[[entities/mae]]", "[[entities/convnext-v2]]", "[[entities/swin-transformer]]"]
sources: ["[[sources/convnext]]", "[[sources/convnext-v2]]"]
updated: 2026-06-17
---

# ConvNeXt

Facebook AI Research（FAIR）+ UC Berkeley の **Zhuang Liu, Saining Xie ら**による純粋 ConvNet アーキテクチャ（arXiv:2201.03545, **CVPR 2022**）。詳細解説: [[sources/convnext]] / 翻訳: [[translations/convnext]]。

## 一言で

**ResNet-50 を Transformer 流の訓練レシピと設計選択で「近代化」して得られた、attention を持たない ConvNet**。同 FLOPs の Swin Transformer を分類・検出・セグメンテーションのすべてで上回りつつ、特殊モジュール（shifted window attention、相対位置バイアス）を一切持たない。

## アーキテクチャ

4 段階（res2〜res5）の階層構造。各段階でチャネル数が倍増する。

**ConvNeXt ブロック**（ResNet ブロックより部品が少ない）:

```
入力 (C ch)
  ├─ d7×7, C            ← depthwise conv（7×7、空間方向のみ混合）
  ├─ LayerNorm          ← 正規化はブロックあたり 1 個だけ
  ├─ 1×1, C → 4C        ← inverted bottleneck（4 倍に拡大）
  ├─ GELU               ← 活性化もブロックあたり 1 個だけ
  ├─ 1×1, 4C → C
  └─ + 残差接続（layer scale 1e-6 付き）
```

**その他の構造上の決定**:

- **patchify stem**: 4×4・stride 4 の非重複畳み込み（ResNet の 7×7 s2 conv + maxpool を置換）
- **stage compute ratio**: (3, 3, 9, 3) または (3, 3, 27, 3)（ResNet-50 の (3,4,6,3) から変更、Swin の 1:1:3:1 / 1:1:9:1 に倣う）
- **独立ダウンサンプリング層**: 段階間に 2×2・stride 2 の conv。**解像度が変わる箇所すべて（stem 後・各ダウンサンプリング前・最終 GAP 後）に LN を入れないと訓練が発散する**
- **完全畳み込み**: 位置埋め込みを持たないため、解像度変更時に補間が不要（ViT/Swin より fine-tune が容易）

## バリアント

$C$ = 各段階のチャネル数、$B$ = 各段階のブロック数。

**表**: ConvNeXt のバリアント構成と ImageNet 性能（[[sources/convnext]] 表1 より）

| モデル | $C$ | $B$ | params | FLOPs (224²) | IN-1K only | IN-22K→1K (384²) |
|---|---|---|---|---|---|---|
| ConvNeXt-T | (96, 192, 384, 768) | (3, 3, 9, 3) | 29M | 4.5G | 82.1 | 84.1 |
| ConvNeXt-S | (96, 192, 384, 768) | (3, 3, 27, 3) | 50M | 8.7G | 83.1 | 85.8 |
| ConvNeXt-B | (128, 256, 512, 1024) | (3, 3, 27, 3) | 89M | 15.4G | 83.8 | 86.8 |
| ConvNeXt-L | (192, 384, 768, 1536) | (3, 3, 27, 3) | 198M | 34.4G | 84.3 | 87.5 |
| ConvNeXt-XL | (256, 512, 1024, 2048) | (3, 3, 27, 3) | 350M | 60.9G | — | **87.8** |

> **パラメータ数の丸めについて** — 上表は原典の表1 の表記（ConvNeXt-T = 29M）に従っている。正確には **T = 28.6M / B = 88.6M / L = 197.8M / XL = 350.2M**（表8・表9）。[[entities/dinov3]] のモデル表は同じモデルを **T = 28M / B = 88M** と切り捨て表記しているが、**指しているモデルは同一**である。

- **ConvNeXt-T / -B** はそれぞれ ResNet-50 / ResNet-200 領域の近代化手続きの最終成果物。**-S / -L / -XL** はそれをスケールしたもの。
- **isotropic 版**（ダウンサンプリングなし、ViT と同形）も S/B/L が構築され、ViT と同等性能かつ訓練メモリはより少ない。

## 主要結果

- **ImageNet-1K**: ConvNeXt-XL で **87.8%**（IN-22K 事前学習、384²）。同規模の Swin を全サイズで上回る。
- **COCO**（Cascade Mask R-CNN）: ConvNeXt-XL ‡ で **55.2 AP^box / 47.7 AP^mask**（Swin-L ‡ 53.9 / 46.7）
- **ADE20K**（UperNet）: ConvNeXt-XL ‡ で **54.0 mIoU**（Swin-L ‡ 53.5）
- **A100 スループット**: Swin 比 **最大 +49%**（TF32 + channel-last）。V100 では差はわずかで、**新しいハードウェアほど ConvNeXt が有利**になる
- **頑健性**: ConvNeXt-XL ‡ で ImageNet-A 69.3 / ImageNet-R 68.2 / Sketch 55.0、mCE 38.8

## 系譜・位置づけ

- **出発点は ResNet**（He et al., 2015）。ResNeXt の grouped/depthwise conv、MobileNetV2 の inverted bottleneck、ViT の patchify、Swin の stage ratio と LN と独立ダウンサンプリングを取り込んでいる。**個々の要素はすべて既存**で、新規性は組み合わせと対照実験にある。
- **比較対象は [[entities/swin-transformer]]**（ICCV 2021, [[sources/swin-transformer]]）。同じ「特殊モジュールを削ぐ」方向の Transformer 側の研究が [[entities/hiera]]。
- **その後の主流にはならなかった**。CLIP / DINOv2 / MLLM の視覚エンコーダはほぼ ViT に収束した。ConvNeXt の完全畳み込み性は解像度変更には有利だが、**素の V1 は MIM（[[concepts/masked-image-modeling]] / [[entities/mae]]）と相性が悪い**——密なスライディングウィンドウゆえマスク領域からの情報漏洩を防げず、実際に FCMAE で事前学習しても教師あり 300ep（83.8）を超えられなかった（83.7）。**ただしこれは翌年 [[entities/convnext-v2]] が疎畳み込みと GRN で解決する**ので、CNN の本質的限界ではなく V1 の設計上の制約だったことになる。
- **ただし「効率が要る場所」で生存**。**[[entities/dinov3]]（2025）は ViT-7B から ConvNeXt-T/S/B/L を蒸留**し、量子化・エッジ展開向けのモデルファミリーとして配布している。本論文の「現代ハードウェアで ConvNeXt は実際上効率的」という主張が、基盤モデルの配布形態として実を結んだ形。
- **後継: [[entities/convnext-v2]]**（Woo et al., CVPR 2023, [[sources/convnext-v2]]）が **FCMAE + GRN** で MIM 非互換問題を正面から解決した。**V1 ブロックに GRN 層を 1 つ足し LayerScale を消しただけ**の最小変更で、疎畳み込みベースのマスク事前学習と組み合わせると IN-1K **88.9%**（公開データのみで当時 SOTA）に達する。V1 の T/B/L はそのまま引き継がれ、**超軽量帯（Atto 3.7M 〜 Nano 15.6M）と Huge 659M が追加**された（V1 の XL は V2 に存在しない）。

## 公開

- コード / 重み: <https://github.com/facebookresearch/ConvNeXt>（MIT ライセンス）
- IN-1K 訓練版と IN-22K 事前学習版の双方、224² / 384² の重みが公開されている
- `timm` に `convnext_tiny` ほかとして収録

## 関連ページ

- [[sources/convnext]] — 論文要約（最重要。近代化ロードマップの全数値を含む）
- [[translations/convnext]] — 全文和訳（Appendix A〜G 込み）
- [[concepts/convolutional-neural-network]] — CNN の帰納バイアス・部品・系譜
- [[concepts/vision-transformer]] — 比較対象のアーキテクチャ
- [[entities/dinov3]] — ConvNeXt バリアントを蒸留で提供している後年の基盤モデル
- [[entities/hiera]] — 「特殊モジュールを削ぐ」simplicity 論の Transformer 側
- [[entities/swin-transformer]] / [[sources/swin-transformer]] — 近代化の到達目標かつ最大の比較対象
- [[entities/convnext-v2]] / [[sources/convnext-v2]] — 後継。GRN + FCMAE で自己教師あり学習と共設計（CVPR 2023）
- [[entities/mae]] — ConvNeXt V2 が取り込んだマスク再構成の源流
