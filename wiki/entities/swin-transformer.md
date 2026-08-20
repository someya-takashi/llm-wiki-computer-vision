---
type: entity
entity_kind: model
aliases: [Swin, Swin Transformer, Swin-T, Swin-S, Swin-B, Swin-L, SwinL]
tags: [swin-transformer, vision-transformer, hierarchical, window-attention, backbone, microsoft]
related: ["[[concepts/vision-transformer]]", "[[entities/hiera]]", "[[entities/convnext]]", "[[concepts/object-detection]]", "[[concepts/masked-image-modeling]]", "[[entities/maxvit]]", "[[entities/hrnet]]"]
sources: ["[[sources/swin-transformer]]"]
updated: 2026-08-21
---

# Swin Transformer

Microsoft Research Asia の **Ze Liu, Han Hu, Stephen Lin, Baining Guo ら**による階層型 Vision Transformer（arXiv:2103.14030, **ICCV 2021 最優秀論文賞 / Marr Prize**）。詳細解説: [[sources/swin-transformer]] / 翻訳: [[translations/swin-transformer]]。

## 一言で

**「局所窓の中だけで self-attention し、次の層で窓を半分ずらす」ことで、ViT を検出・セグメンテーションにも使える汎用視覚バックボーンに変えたモデル**。名前の Swin は **S**hifted **win**dows の略。2021〜2023 年の検出・セグメンテーション系研究の事実上の標準バックボーンになった。

## アーキテクチャ

### 4 段階の階層構造

CNN と同じ解像度系列（H/4 → H/8 → H/16 → H/32）を持つため、**FPN や UperNet といった既存の密予測ヘッドにそのまま差し込める**。

| 段階 | 解像度（224² 入力） | ダウンサンプリング | チャネル |
|---|---|---|---|
| Stage 1 | 56×56 | Patch Partition 4×4 + Linear Embedding | $C$ |
| Stage 2 | 28×28 | Patch Merging（2×2 連結 → 線形） | $2C$ |
| Stage 3 | 14×14 | Patch Merging | $4C$ |
| Stage 4 | 7×7 | Patch Merging | $8C$ |

- **パッチサイズ 4×4**（ViT の 16×16 より細かい）。初期特徴は生 RGB 連結の 48 次元
- **[CLS] トークンなし**。分類は最終段階の大域平均プーリング
- **絶対位置埋め込みなし**。代わりに窓内の**相対位置バイアス**を使う

### Swin Transformer ブロック（必ず 2 個 1 組）

```
z^(l-1) ─→ LN ─→ W-MSA  ─→ + ─→ LN ─→ MLP ─→ +  = z^l      ← 通常窓
z^l     ─→ LN ─→ SW-MSA ─→ + ─→ LN ─→ MLP ─→ +  = z^(l+1)  ← シフト窓
```

**W-MSA と SW-MSA は必ずペア**で使うため、各段階のブロック数は必ず偶数（{2,2,6,2} 等）。pre-norm 構成、MLP は GELU 付き 2 層で拡大率 $\alpha=4$。

### 3 つの中核技法

1. **窓内 self-attention（W-MSA）**: 窓サイズ $M=7$ 固定。計算量が $\Omega=4hwC^2+2M^2hwC$ となり、**画像サイズに対して線形**（大域 MSA は $2(hw)^2C$ で二次）
2. **シフトウィンドウ（SW-MSA）**: 次のブロックで窓を $(\lfloor M/2\rfloor, \lfloor M/2\rfloor)$ ずらし、窓をまたぐ結合を作る。**sliding window より 4.1×（naive）/ 1.5×（カーネル最適化）高速で精度は同等**
3. **cyclic shift + マスク**: 窓をずらすと窓数が増える問題を、特徴マップ自体を巡回シフトさせて回避。**13-18% 高速化**

## バリアント

$C$ = Stage 1 のチャネル数、layer numbers = 各段階のブロック数。窓サイズ $M=7$、head 次元 $d=32$ は全バリアント共通。

**表**: Swin のバリアントと ImageNet 性能（[[sources/swin-transformer]] 表1 / 表7 より）

| モデル | $C$ | layer numbers | params | FLOPs (224²) | IN-1K only | IN-22K → 1K |
|---|---|---|---|---|---|---|
| **Swin-T** | 96 | {2, 2, 6, 2} | 29M | 4.5G | 81.3 | — |
| **Swin-S** | 96 | {2, 2, 18, 2} | 50M | 8.7G | 83.0 | — |
| **Swin-B** | 128 | {2, 2, 18, 2} | 88M | 15.4G | 83.5（384² で 84.5） | 85.2（384² で **86.4**） |
| **Swin-L** | 192 | {2, 2, 18, 2} | 197M | 103.9G (384²) | — | **87.3**（384²） |

- **設計の対応関係**: Swin-T ≈ ResNet-50 / DeiT-S、Swin-S ≈ ResNet-101、Swin-B ≈ ViT-B / DeiT-B、Swin-L ≈ その 2 倍
- **Swin-L は IN-22K 事前学習のみ**で提供される（IN-1K からのスクラッチ訓練は報告なし）
- **派生**: **Swin-Mixer**（階層設計 + シフト窓を MLP-Mixer に移植、Appendix A3.3。MLP-Mixer 76.4 → 81.3）、**Swin V2**（後続研究。本 wiki 未取り込みだが [[entities/dino-detector]] の SwinV2-G、[[entities/convnext-v2]] の Swin V2-H として言及あり）

## 主要結果

- **ImageNet-1K**: Swin-T **81.3**（DeiT-S 79.8 に +1.5）/ Swin-B **83.5** / Swin-L ‡ 384² で **87.3**
- **COCO 検出**: **58.7 box AP / 51.1 mask AP**（Swin-L + HTC++ + マルチスケールテスト、test-dev）。従来 SOTA を **+2.7 box AP / +2.6 mask AP** 更新
  - 4 つの検出フレームワーク（Cascade Mask R-CNN / ATSS / RepPointsV2 / Sparse R-CNN）すべてで ResNet-50 → Swin-T の差し替えだけで **+3.4〜+4.2 box AP**
- **ADE20K セグメンテーション**: UperNet + Swin-L ‡ で **53.5 mIoU**（SETR 50.3 に **+3.2**）
- **アブレーション**: シフト窓が分類 +1.1 / 検出 +2.8 AP / セグ +2.8 mIoU、相対位置バイアスが検出 +1.3 AP / セグ +2.3 mIoU

## 本 wiki における Swin の位置づけ

Swin は本 wiki の多くのページで**バックボーンとして言及されてきた**。その一覧:

| 参照元 | Swin の役割 |
|---|---|
| [[entities/glip]] | Swin-T / Swin-L をバックボーンに採用（GLIP-L = Swin-L） |
| [[entities/grounding-dino]] | Swin-T（172M）/ Swin-L（341M）の 2 バリアント |
| [[entities/grounding-dino-1-5]] | Swin ベースから ViT-L への移行の比較対象 |
| [[entities/dino-detector]] | **DINO-SwinL で COCO test-dev 63.3 AP**（SwinV2-G の 1/15 パラメータで上回る） |
| [[entities/yolo-world]] | 速度比較の対象（Grounding-DINO-T = Swin-T の 35× 高速を主張） |
| [[entities/dino-x]] | 比較表の Swin-L 系列 |
| [[entities/hiera]] | **「Swin の工夫を MAE で削ぎ落とす」対象**（shifted window / RPB を削除） |
| [[entities/convnext]] / [[sources/convnext]] | **近代化の到達目標かつ主要比較対象**（ConvNeXt-T 82.1 > Swin-T 81.3） |
| [[entities/convnext-v2]] | SimMIM で事前学習した Swin が MIM 比較のベースライン |
| [[concepts/object-detection]] | 検出バックボーンの世代交代の中心 |
| [[concepts/rotary-position-embeddings]] | 相対位置バイアスの代表例（RoPE 以前の主流） |
| [[concepts/masked-image-modeling]] | 窓構造が MIM と干渉する例 |

## 系譜・後の評価

- **出発点は [[concepts/vision-transformer]]**（ViT, 2020）。Swin はそこに **CNN の階層性と局所性を輸入**した。
- **2019: [[entities/hrnet|HRNet]] が同じ問題に CNN 側から答えていた** — 「密 prediction には高解像度が要る」という問題意識は Swin と共通で、HRNet の解は**解像度を落とさず並列ストリームを維持する**ことだった（[[sources/hrnet]]）。Swin が乗り越えるべき当時の CNN 側 SOTA がまさにこれで、[[translations/swin-transformer]] の ADE20K 比較表には **OCRNet + HRNet-w48 が 45.7 mIoU** で載っている（Swin-L ‡ は 53.5）。**階層性を輸入した Swin と、階層性を並列化した HRNet** という対比で読むと分かりやすい。
- **2022: [[sources/convnext|ConvNeXt]] からの反論** — 「ConvNet の性質を苦労して輸入するくらいなら ConvNet を近代化せよ」。実際 ConvNeXt-T 82.1 > Swin-T 81.3 で、A100 では最大 +49% 高速。**「Swin の特殊モジュールは性能の源泉ではなかった」**という主張。
- **2022: [[entities/maxvit]] からの置き換え提案** — 「窓をずらして間接的に大域性を得るより、**grid attention で直接得た方が良い**」。Max-SA は **Swin の attention と同パラメータ・同 FLOPs の drop-in 置換**でありながら cyclic shift もマスクも不要で、IN-21K 以降で大きく上回る（[[sources/maxvit]]）。
- **2023: [[entities/hiera]] からの簡素化** — 「MAE で適切に事前学習すれば shifted window も相対位置バイアスも全部要らない」。Hiera-B 51M で 84.3 > Swin-B 88M で 83.5。
- **MIM との相性の悪さ**が弱点として残り、SimMIM という専用手法が必要だった。この制約が [[entities/hiera]] と [[entities/convnext-v2]] を生む動機になった（[[concepts/masked-image-modeling]] の「MIM とアーキテクチャの相性」表を参照）。
- **それでも検出・グラウンディング領域では長く標準**であり続けた。[[entities/glip]]（2022）→ [[entities/grounding-dino]]（2024）と、open-vocabulary 検出の主要系譜が Swin バックボーンの上に築かれている。

## 公開

- コード / 重み: <https://github.com/microsoft/Swin-Transformer>（MIT ライセンス）
- IN-1K 訓練版（T/S/B）と IN-22K 事前学習版（B/L）、224² / 384² の重みが公開
- `timm` に `swin_tiny_patch4_window7_224` 等として収録。`mmdetection` / `mmsegmentation` も公式対応

## 関連ページ

- [[sources/swin-transformer]] — 論文要約（最重要）
- [[translations/swin-transformer]] — 全文和訳（Appendix A1〜A3 込み）
- [[concepts/vision-transformer]] — 出発点の ViT
- [[sources/convnext]] / [[entities/convnext]] — 直接の反論（CVPR 2022）
- [[entities/hiera]] — Swin の工夫を削ぎ落とした簡素化版（ICML 2023）
- [[concepts/object-detection]] — Swin が塗り替えた検出バックボーン事情
- [[entities/maxvit]] / [[sources/maxvit]] — Swin の attention の drop-in 置換を提案したハイブリッド系（ECCV 2022）
- [[concepts/convolutional-neural-network]] — Swin が輸入した階層性・局所性の出どころ
- [[entities/hrnet]] / [[sources/hrnet]] — 同じ密 prediction 問題への CNN 側からの回答（2019）。ADE20K の比較対象
