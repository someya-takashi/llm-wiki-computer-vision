---
type: entity
entity_kind: model
aliases: [ConvNeXt, CNX, Convolutional Next]
tags: [convnext, convolutional-neural-network, architecture, backbone, fair, meta-ai]
related: ["[[concepts/convolutional-neural-network]]", "[[concepts/vision-transformer]]", "[[entities/dinov3]]", "[[entities/hiera]]", "[[entities/mae]]", "[[entities/convnext-v2]]", "[[entities/swin-transformer]]", "[[entities/maxvit]]", "[[entities/rdnet]]", "[[entities/inceptionnext]]", "[[concepts/skip-connection]]"]
sources: ["[[sources/convnext]]", "[[sources/convnext-v2]]"]
updated: 2026-09-01
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
- [[entities/maxvit]] / [[sources/maxvit]] — 同時期のハイブリッド路線（畳み込みと attention を 1 ブロックに統合）
- [[entities/convnext-v2]] / [[sources/convnext-v2]] — 後継。GRN + FCMAE で自己教師あり学習と共設計（CVPR 2023）
- [[entities/mae]] — ConvNeXt V2 が取り込んだマスク再構成の源流

## 対をなす論文: RDNet（DenseNets Reloaded）

**[[entities/rdnet]]**（NAVER AI Lab, ECCV 2024, [[sources/rdnet]]）は、**ConvNeXt とまったく同じ手続きを DenseNet に適用した**論文である。

| | **ConvNeXt**（2022） | **RDNet**（2024） |
|---|---|---|
| **出発点** | ResNet-50（レシピ更新で 76.1 → 78.8） | DenseNet-201（現代レシピで 79.7） |
| **到達点** | ConvNeXt-T **82.1** | RDNet-T **82.8** |
| **保持したもの** | 畳み込み（attention を使わない） | **連結ショートカット** |
| **輸入したもの** | Transformer の訓練レシピと設計 | **ConvNeXt のブロック設計** + 訓練レシピ |
| **最大の単一要因** | **訓練レシピ**（総改善の約 46%） | **遷移層を 3 ブロックごとに挟む**（+1.5%p） |
| **主張** | ViT が CNN を置き換えたわけではない | **加算が連結を置き換えたわけではない** |

**RDNet は ConvNeXt のブロックをそのまま借りている**（LayerNorm・depthwise・7×7 カーネル・活性化を減らす）。つまり「ConvNeXt の成果を土台にして、**ConvNeXt が触らなかった軸——ショートカットの型——を変えた**」という構図になる。

結果は **RDNet-T 82.8 > ConvNeXt-T 82.1**、**RDNet-B 84.4 > ConvNeXt-B 83.8**、ADE20K **49.6 > 49.1**、COCO **47.5 > 46.2**（しかも 43M vs 48M）。ただし**メモリでは ConvNeXt が勝つ**（RDNet-T 4.1GB vs ConvNeXt-T 2.7GB）——連結の代償は緩和できても消えない。また b1 レイテンシでは RDNet が圧倒的に速い（7.4ms）一方、b128 では差が縮む。

**ConvNeXt が本 wiki に持ち込んだ「近代化ロードマップ」という作法が別のアーキテクチャで再現された**点で、方法論としての一般性を示す事例でもある。

- [[entities/rdnet]] / [[sources/rdnet]] — 詳細
- [[concepts/skip-connection]] — 加算 vs 連結という軸の整理

## もう 1 つの批判: InceptionNeXt（7×7 は実機で遅い）

**[[entities/inceptionnext]]**（NUS / Sea AI Lab, CVPR 2024, [[sources/inceptionnext]]）は、RDNet とは別の角度から ConvNeXt を刺した。狙いは**ショートカットではなく 7×7 depthwise 畳み込みそのもの**である。

> **ConvNeXt-T は ResNet-50 と同程度の FLOPs を持つが、A100 上の訓練スループットは約 60% しかない**（575 vs 969 img/s）。

原因は **メモリアクセスコスト**。depthwise 畳み込みは演算量に対して読み書きするデータ量の比が悪く、カーネルが大きいほど GPU がメモリ帯域に律速される。**ConvNeXt の近代化ロードマップは「7×7 で精度が飽和する」ことは示したが、7×7 が実機で払っているコストは評価していない**（ConvNeXt 論文が出す A100 スループットは Swin との比較のみで、同 FLOPs の ResNet-50 とは並べていない）。

InceptionNeXt の処方は**大カーネルの分解**である。`DWConv 7×7` を **`DWConv 3×3` + `DWConv 1×11` + `DWConv 11×1` + 恒等写像**の 4 分岐に置き換え、畳み込みを通すのは全チャネルの 3/8 だけにする。

| Model | Params | MACs | A100 訓練 | Top-1 |
|---|---|---|---|---|
| **ConvNeXt-T** | 29 | 4.5 | **575** | 82.1 |
| **InceptionNeXt-T** | 28 | 4.2 | **901 (+57%)** | **82.3 (+0.2)** |
| ConvNeXt-A | 3.7 | 0.55 | 835 | 75.7 |
| **InceptionNeXt-A** | 4.2 | 0.51 | **2661 (+219%)** | 75.3 (−0.4) |

**ブロック構造そのものは ConvNeXt をほぼそのまま踏襲している**（4 ステージ、ブロック数 [3,3,9,3]、MLP 比 4）。違いはトークン混合器と、**速度優先で LayerNorm ではなく BatchNorm を採る**点、第 4 ステージだけ MLP 比を 3 にする点の 3 つだけ。**LN の方が精度は 0.1 高いが訓練スループットが 20% 落ちる**——ConvNeXt が「近代化を済ませた後なら LN がわずかに良い」と結論したのと、目的関数が違えば判断が逆になる例である。

**RDNet と InceptionNeXt は同じ診断・別の処方**として並べられる。

| | **[[entities/rdnet\|RDNet]]** | **[[entities/inceptionnext\|InceptionNeXt]]** |
|---|---|---|
| 変えた軸 | **ショートカットの型**（加算 → 連結） | **トークン混合器**（7×7 → 4 分岐） |
| ConvNeXt-T 比 | **82.8**（+0.7） | 82.3（+0.2）／**訓練 1.6 倍** |
| 効きどころ | 小バッチ・低レイテンシ | **軽量モデル**（大きいと MLP が支配し効かない） |

RDNet の比較表には InceptionNeXt-T が **132ms と表中最速のレイテンシ**で載っており（RDNet-T は 175ms で 82.8）、**精度で RDNet、速度で InceptionNeXt**という住み分けになる。両者は直交しうるが、組み合わせは誰も試していない ⚠。
