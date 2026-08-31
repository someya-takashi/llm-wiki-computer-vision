---
type: entity
entity_kind: model
aliases: [HRNet, HRNetV1, HRNetV2, HRNetV2p, High-Resolution Network, HRNet-W32, HRNet-W48]
tags: [hrnet, convolutional-neural-network, backbone, pose-estimation, semantic-segmentation, object-detection, high-resolution, microsoft]
related: ["[[concepts/convolutional-neural-network]]", "[[concepts/object-detection]]", "[[entities/swin-transformer]]", "[[entities/convnext]]", "[[entities/nfnet]]", "[[concepts/skip-connection]]"]
sources: ["[[sources/hrnet]]"]
updated: 2026-08-31
---

# HRNet（High-Resolution Network）

Microsoft Research Asia の **Jingdong Wang, Ke Sun** ら 12 名による、**過程全体を通して高解像度表現を維持する** CNN バックボーン（arXiv:1908.07919, CVPR 2019 → **IEEE TPAMI 2021**）。詳細解説: [[sources/hrnet]] / 翻訳: [[translations/hrnet]]。

## 一言で

**「低解像度に落としてから復元する」という LeNet-5 以来の設計規則を捨てたバックボーン**。高解像度から低解像度までの畳み込みストリームを**並列に**走らせ続け、その間で**双方向の情報交換を 8 回繰り返す**。姿勢推定・セマンティックセグメンテーション・物体検出という**位置に敏感なタスク**で ResNet 系を一貫して上回り、しかも**セグメンテーションでは PSPNet の約 1/3 の GFLOPs**で済む。

## 構造

```
ステム: 3×3 s2 → 3×3 s2  （1/4 解像度へ）
        ↓
Stage 1: 1/4                                    ← bottleneck ×4、幅を C に
Stage 2: 1/4  1/8                    ×1 ブロック
Stage 3: 1/4  1/8  1/16              ×4 ブロック
Stage 4: 1/4  1/8  1/16  1/32        ×3 ブロック
         C    2C    4C    8C          ← 各解像度の幅
```

- 各ブロックの各分岐 = **$3\times3$ 畳み込み 2 つの残差ユニット ×4** + BN + ReLU
- **4 つの残差ユニットごとに多解像度融合ユニット**（合計 8 回）
- 融合: 高→低は **stride 2 の $3\times3$ 畳み込み**、低→高は**双線形アップサンプル + $1\times1$**、同解像度は恒等。**すべての出力解像度がすべての入力解像度の和**
- **$1/4$ 解像度のストリームは最初から最後まで途切れない**

## 3 つのヘッド

| ヘッド | 出力の作り方 | 論文での適用先 |
|---|---|---|
| **HRNetV1** | 高解像度ストリームの出力のみ（他 3 つは捨てる） | 人物姿勢推定 |
| **HRNetV2** | 4 解像度をアップサンプルして連結 → $1\times1$ で混合 | セマンティックセグメンテーション |
| **HRNetV2p** | HRNetV2 の出力をダウンサンプルして特徴ピラミッド化 | 物体検出（FPN 系の枠組みに差し込む） |
| （HRNetV1h） | V1 に $1\times1$ を足して次元だけ V2 に揃えた**対照実験用**の変種 | アブレーション専用 |

**V1 = CVPR 2019 会議版、V2 / V2p = TPAMI 版で追加。** 姿勢推定では V1 と V2 の性能はほぼ同じ（73.4 vs 73.6 AP）だが、セグメンテーションと検出では **V2 が V1 を大幅に上回る**。HRNetV1h がわずかしか改善しないことから、**差は出力次元ではなく低解像度表現を実際に集約していること自体から来る**と確認された。

## バリアント

幅 $C$（高解像度ストリームのチャネル数）でサイズが決まる。他の解像度は $2C, 4C, 8C$。

| 名称 | $C$ | 主な用途と代表スコア |
|---|---|---|
| **HRNet-W18** | 18 | ImageNet top-1 err. 23.1%（21.3M, 3.99 GFLOPs） |
| **HRNetV1-W32** | 32 | COCO pose val **73.4 AP**（事前学習なし）/ **74.4 AP**（IN 事前学習）、28.5M, 7.10 GFLOPs |
| **HRNet-W30 / W40** | 30 / 40 | ImageNet 21.9% / 21.1% err.。**HRNetV2-W40** で Cityscapes val **80.2 mIoU**（493.2 GFLOPs） |
| **HRNetV1-W48** | 48 | COCO pose val **76.3 AP**（384×288）/ test-dev **75.5 AP**、63.6M |
| **HRNetV2-W48** | 48 | Cityscapes val **81.1 mIoU**（696.2 GFLOPs）、PASCAL-Context 54.0 / LIP で SOTA |
| **HRNetV2-W48 + OCR** | 48 | Cityscapes val **81.6**、PASCAL-Context **56.2 / 50.1**（論文中の最高値） |
| HRNet-W44 / W76 / W96 | — | ImageNet 分類用の bottleneck 版（ResNet-50 / 101 / 152 と対応） |

分類ヘッドの代替方式として **HRNet-W$x$-Ci**（各解像度を大域プーリングして連結、$15C$ 次元）と **HRNet-W$x$-Cii**（会議版で使用、2048 次元）も評価されているが、**採用された -C 方式（bottleneck で昇圧し順次加算して 1024ch）が最良**（表XVI）。

## 主要結果

| タスク | データセット | スコア | 比較 |
|---|---|---|---|
| 人物姿勢推定 | COCO val | **76.3 AP**（V1-W48, 384×288） | SimpleBaseline ResNet-152 の 74.3 を **92.4% の計算量で** +2.0 |
| 人物姿勢推定 | COCO test-dev | **75.5 AP**（V1-W48） | 全トップダウン手法を上回る |
| セグメンテーション | Cityscapes val | **81.1 mIoU**（V2-W48） | PSPNet 79.7 を**同パラメータ・約 1/3 の GFLOPs**で上回る |
| セグメンテーション | PASCAL-Context | **56.2 / 50.1 mIoU**（+OCR） | SOTA |
| セグメンテーション | LIP | 最良（追加情報なし） | 姿勢・エッジ情報を使わずに |
| 物体検出 | COCO | Faster/Cascade R-CNN, FCOS, CenterNet, Mask R-CNN, HTC 等で ResNet/ResNeXt 超え | **特に AP_S（小物体）で顕著** |
| 分類 | ImageNet | top-1 err. **21.0%**（W96-C） | ResNet-152 の 21.2% とほぼ同等 |

**分類での優位が小さく（0.1〜1.5pt）、位置に敏感なタスクでの優位が大きい（1.4〜6.5pt）**という非対称性は、論文の主張そのものを裏付けている。

## 設計上の非自明な知見（アブレーションより）

- **融合の回数が効く**: 1 回のみ 70.8 AP → 3 回 71.9 → **8 回 73.4**。「並列にする」だけでは足りない
- **融合は和でなければならない**: 乗算に置換すると **54.7 AP / 66.0 mIoU** と壊滅
- **ダウンサンプルは学習可能でなければならない**: 双線形ダウンサンプルに置換すると −0.8 AP / −2.2 mIoU
- **低解像度ストリームは最初から並走させない方がよい**: 4 本すべてを最初から走らせると 72.5 AP（提案 73.4 より低い）。「低解像度ストリームの初期段階の低レベル特徴はあまり有用でない」
- **かといって高解像度ストリームだけでも駄目**: 同パラメータでもはるかに低い性能。**両方が要る**
- **最低解像度から推定したヒートマップは AP 10 未満**: 解像度がキーポイント予測品質を直接支配する

## 理論的な位置づけ

TPAMI 版で追加された §III-E の議論。

- **並列多解像度畳み込み ≒ group convolution**（チャネルを分けて別々に畳み込む。違いは解像度が異なること）
- **多解像度融合 ≒ 通常の畳み込みの多分岐全結合形式**（入出力チャネルを部分集合に分け、全対全の小畳み込みで結んで和を取る）

この等価性が **V2 / V2p の根拠**になっている。「1 ブロックが通常の畳み込み 1 層と同型なら、高解像度だけ使う V1 は出力チャネルの一部を捨てている」——実験がこの予測を裏付けた。

## 系譜・位置づけ

- **「解像度をどう扱うか」という軸の到達点**。ResNet（直列に落とす）→ U-Net / Hourglass（落として復元）→ DeepLab / PSPNet（dilated で $1/8$ で止める）→ **HRNet（落とさず並列維持 + 反復融合）**
- **先行する「並列」の試みを明示的に批判している**: convolutional neural fabrics / interlinked CNNs（設計の作り込みを欠き、BN も残差もない）、**GridNet**（情報交換が高→低・低→高の 2 段階に分離し双方向の反復がない）、Multi-scale DenseNet（低→高の情報が返らない）。**新規性は「並列」ではなく「双方向の反復融合」**
- **[[entities/swin-transformer]] とは同じ問題への別解**。Swin（2021）が「CNN の階層性を Transformer に輸入する」ことで密 prediction を解いたのに対し、HRNet（2019）は「CNN の階層性を並列化する」ことで解いた。Swin の ADE20K 比較表に **OCRNet + HRNet-w48（45.7 mIoU）** が競合として載っており、**Swin が乗り越えるべき CNN 側 SOTA が HRNet だった**（[[translations/swin-transformer]]）
- **[[entities/convnext]] とは対照的**。ConvNeXt は「部品を減らす」方向で成功したが、HRNet の多分岐 + 全対全融合は実装・最適化のコストが高い
- **コミュニティによる採用が最も直接的な評価**: ICCV 2019 の COCO キーポイント検出チャレンジで**ほぼ全参加者が採用**、DensePose チャレンジ優勝者、OpenImage インスタンスセグメンテーション優勝者も使用。Mapillary パノプティックセグメンテーションでは ASPP と組み合わせた変種が単一モデル SOTA
- **本 wiki 内での登場**: [[sources/segment-anything|SAM]] の対話的セグメンテーション比較で RITM の **HRNet32** がベースライン、[[sources/dino-x|DINO-X]] の姿勢推定比較表にも HRNet が並ぶ。**2023〜2025 年の論文にまだ比較対象として出てくる 2019 年のアーキテクチャ**

## 弱点

- **多分岐構造は実装・最適化コストが高い**。姿勢推定の推論時間は PyTorch 1.0（動的グラフ）では SimpleBaseline より遅く、静的グラフの MXNet 1.5.1 でようやく同等になる。フレームワーク依存性が大きい
- **分類での優位が小さい** → 大規模事前学習によるスケール則の恩恵を受けにくいことを示唆。2019 年以降 CV が向かった「大規模事前学習 → 汎用表現」の方向とは哲学が違う
- **自己教師あり学習との接続が未検証**。[[concepts/masked-image-modeling]] との組み合わせは扱われていない（[[entities/convnext-v2]] が CNN 側で行ったような作業に相当するものがない）
- **言語との接続がない**。出力が 4 解像度の特徴マップで、パッチ列として [[entities/clip]] 系や MLLM の視覚エンコーダには馴染みにくい
- **解像度は $1/4$ で止まっている**。$1/2$・全解像度への拡張は論文自身が今後の課題としている
- PASCAL-Context では **OCR なしだと APCN に負ける**（論文も明記）

## 応用範囲

論文本体（姿勢推定・セグメンテーション・検出）に加え、**顔ランドマーク検出**（Appendix D: WFLW / AFLW / COFW / 300W）を評価。論文が挙げる後続研究には画像スタイル化、インペインティング、画像強調、画像除霧、時間的姿勢推定、ドローン物体検出がある。**位置に敏感な問題全般**が守備範囲。

## 公開

- コード / 重み: <https://github.com/HRNet>（タスク別に HRNet-Human-Pose-Estimation / HRNet-Semantic-Segmentation / HRNet-Object-Detection / HRNet-Facial-Landmark-Detection 等のリポジトリ群）
- `timm` に `hrnet_w18` 〜 `hrnet_w64` として分類版を収録。MMPose / MMSegmentation / MMDetection にも標準バックボーンとして統合されている

## 関連ページ

- [[sources/hrnet]] — 論文要約（最重要）
- [[translations/hrnet]] — 全文和訳（Appendix A〜E 込み）
- [[concepts/convolutional-neural-network]] — CNN の系譜。本モデルはその「解像度」軸の到達点
- [[concepts/object-detection]] — HRNetV2p が差し込まれる枠組み
- [[concepts/skip-connection]] — HRNet の多解像度融合を「解像度を運ぶ skip の極端な形」として位置づけた概念ページ
- [[entities/swin-transformer]] / [[sources/swin-transformer]] — 同じ問題への Transformer 側からの回答
- [[entities/convnext]] — 「部品を減らす」対照的な設計哲学
- [[entities/nfnet]] — 同時期の別方向の CNN（正規化の排除）
- [[sources/segment-anything]] / [[sources/dino-x]] — 後年の論文で比較対象として登場する箇所
