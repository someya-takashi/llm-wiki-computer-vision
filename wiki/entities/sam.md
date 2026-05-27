---
type: entity
entity_kind: model
aliases: [SAM, Segment Anything Model, SAM v1]
tags: [model, segmentation, foundation-model, meta-fair, promptable]
related: [[concepts/promptable-segmentation]], [[concepts/foundation-model]], [[concepts/zero-shot-transfer]], [[concepts/vision-transformer]]
sources: [[sources/segment-anything]]
updated: 2026-05-26
---

# SAM（Segment Anything Model）

## 概要

**SAM** = **Segment Anything Model**。Meta AI FAIR チーム（Kirillov ら, 2023）が発表した、**CV における初の本格的セグメンテーション基盤モデル**。promptable segmentation（[[concepts/promptable-segmentation]]）タスクで訓練され、点・ボックス・粗マスク・テキストの任意のプロンプトから即座にマスクを返す。

- 論文: "Segment Anything"
- arXiv: 2304.02643（2023 年 4 月）→ ICCV 2023
- コード: <https://github.com/facebookresearch/segment-anything>
- デモ: <https://segment-anything.com/demo>
- ライセンス: **Apache 2.0**（モデル）/ Research-only（SA-1B データセット）
- 詳細解説: [[sources/segment-anything]] / 翻訳: [[translations/segment-anything]]

## アーキテクチャ

3 コンポーネント設計：

| コンポーネント | 構成 | 出力 |
|---|---|---|
| **画像エンコーダ** | MAE 事前学習済み **ViT-H/16**（[[entities/mae]] / [[concepts/vision-transformer]]）、14×14 windowed attention + 4 つの global attention block、1024×1024 入力 | 64×64×256 画像埋め込み |
| **プロンプトエンコーダ** | 疎: 位置エンコーディング + 学習埋め込み（点・ボックス）、CLIP テキストエンコーダ（[[entities/clip]]、テキスト）。密: 畳み込みで埋め込み（マスク） | 256 次元埋め込みの集合 |
| **マスクデコーダ** | 修正 Transformer デコーダ × 2 層、出力トークン + 動的線形分類器 | 3 マスク + 信頼度（推定 IoU） |

**画像エンコーダは画像 1 枚あたり 1 回だけ走る**ため、その後のプロンプト処理は Web ブラウザの CPU 上で **約 50ms**。

## モデルファミリー

公開された SAM v1 は 3 種類の画像エンコーダ：

| 名前 | 画像エンコーダ | パラメータ数 | 訓練 GPU | チェックポイント |
|---|---|---|---|---|
| **SAM ViT-B** | ViT-B/16 | 94M | 128 GPU × 180k iter | `sam_vit_b_01ec64.pth` |
| **SAM ViT-L** | ViT-L/16 | 312M | 128 GPU × 180k iter | `sam_vit_l_0b3195.pth` |
| **SAM ViT-H**（デフォルト） | ViT-H/16 | 636M | **256 A100 × 68 時間 ≈ 17,408 GPU 時間** | `sam_vit_h_4b8939.pth` |

すべて 1024×1024 入力、AdamW、focal:dice = 20:1 損失、SA-1B（[[entities/sa-1b]]）で訓練。

> **補足: ViT-B → ViT-L → ViT-H のスケーリング** — 著者は §7.6（図 13 右）で「ViT-H は ViT-B を実質的に改善するが、ViT-L → ViT-H は限界的。これ以上の画像エンコーダスケーリングは実りある様子ではない」と明言。これは MAE/DINOv2/DINOv3 が ViT-g/7B までスケールしたのとは異なる挙動で、**promptable segmentation タスクは画像エンコーダ容量よりデータ多様性で決まる**ことを示唆する。

## 訓練レシピ（ViT-H デフォルト）

- **データ**: SA-1B（11M 画像 / 1.1B マスク、完全自動段階のみ）
- **最適化**: AdamW（β₁=0.9, β₂=0.999）
- **学習率**: 8e-4（250 イテレーションのウォームアップ後）、60k と 86666 で 10 分の 1 ずつ減衰
- **イテレーション数**: 90k（約 2 SA-1B エポック）
- **バッチサイズ**: 256 画像（256 GPU 分散、GPU あたり 64 マスク）
- **損失**: focal loss + dice loss を 20:1 で結合 + IoU 予測 MSE（係数 1.0）
- **正則化**: weight decay 0.1, drop path 0.4, 層別 lr decay 0.8
- **データ拡張**: なし（SA-1B が十分大きいため）
- **対話シミュレーション**: マスクあたり **11 ラウンド**（1 初期 + 8 反復点 + 2 マスクのみ）
- **初期化**: MAE 事前学習済み ViT-H

## SA-1B 構築用の特別版

完全自動段階で SA-1B を生成するためには、デフォルトと異なる特別版 SAM が使われた（§B）：

- 手動 + 半自動データのみで訓練
- **177656 イテレーション**（90k より長い）
- large-scale jitter データ拡張
- 点とマスクプロンプトのみ（ボックスなし）、マスクあたり 4 点サンプル
- **マスクデコーダ 3 層**（デフォルト 2 層）

## 推論パイプライン（公開コード）

```python
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

predictor.set_image(image)  # 画像エンコーダ実行 ← 重い
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),  # 1=foreground, 0=background
    multimask_output=True,  # 3 マスク返す
)
# masks: (3, H, W), scores: (3,) ＝ 推定 IoU
```

自動マスク生成（SA-1B 生成スタイル）：

```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# 各 mask: {"segmentation": ..., "area": ..., "bbox": ..., "predicted_iou": ..., "stability_score": ..., "crop_box": ...}
```

## 主要結果

ゼロショット転移実験（詳細は [[sources/segment-anything]]）：

| タスク | 評価 | 結果 |
|---|---|---|
| 単一点 → マスク | 23 データセット mIoU | 16/23 で RITM を上回り、最大 +47 IoU |
| 単一点 → マスク（人手評価） | 7 データセット平均評価 | 7〜9 点（"小さなエラーのみ"）で RITM を有意に上回る |
| エッジ検出 | BSDS500 ODS | 0.768（HED 0.788 と同等、Canny/Felz-Hutt を圧倒） |
| オブジェクト提案 | LVIS v1 AR@1000 | 59.3（ViTDet-H 63.0 に近い、中/大/希少で上回る） |
| インスタンスセグメンテーション | COCO mask AP | 46.5 vs ViTDet-H 51.0（人手評価では SAM が勝つ） |
| テキスト → マスク | 定性 | "a wheel", "beaver tooth grille" 等で機能 |

## 後継・派生モデル

| モデル | 提案 | 特徴 |
|---|---|---|
| **SAM 2**（Meta, 2024, [[entities/sam-2]] / [[sources/sam-2]]） | Ravi et al. | 動画拡張、Hiera エンコーダ + streaming memory、SA-V データセット。**画像でも SAM v1 比 6× 高速 + 高精度** で SAM v1 の事実上の上位互換 |
| **SAM 3**（Meta Superintelligence Labs, 2025, [[entities/sam-3]] / [[sources/sam-3]]） | Carion et al. | **新タスク PCS（Promptable Concept Segmentation）導入** — 名詞句/画像 exemplar で全インスタンスを検出・セグメント・追跡。Perception Encoder backbone + DETR detector + presence head + SAM 2 tracker。PVS でも SAM 2 を凌駕 |
| **MobileSAM**（2023） | Zhang et al. | ViT-H → TinyViT 蒸留、軽量化 |
| **FastSAM**（2023） | Zhao et al. | YOLOv8-seg ベースで高速化、SA-1B の 2% で訓練 |
| **EfficientSAM**（Meta, 2023） | Xiong et al. | MAE → SAM 蒸留 |
| **HQ-SAM**（2023） | Ke et al. | 高品質マスク用の追加トークン |
| **MedSAM**（2024） | Ma et al. | 医療画像ファインチューン |
| **Grounded-SAM**（2023） | IDEA Research | Grounding DINO + SAM のテキスト → ボックス → マスクパイプライン |
| **SAM HQ / Semantic-SAM / Personalized SAM** | 多数 | ドメイン特化派生 |

> **補足: SAM 2 とは何か** — Meta が 2024 年に発表した SAM の動画拡張版（[[sources/sam-2]] / [[entities/sam-2]] で ingest 済み）。Streaming memory（空間メモリ + プロンプトメモリ + object pointer）を追加し、動画フレーム間でオブジェクトを追跡できる。画像エンコーダを ViT-H から **Hiera**（[[entities/hiera]]）に変更したことで、画像セグメンテーションでも SAM v1 比 **6× 高速 + 高精度**（同じ SA-1B で訓練しても勝つ）。SA-V データセット（[[entities/sa-v]]、35.5M マスク、642.6K masklet、50.9K 動画、**CC by 4.0**）も同時公開。新規プロジェクトでは SAM v1 ではなく SAM 2 を使うことが推奨される。

## なぜ重要か

CLIP と並ぶ **CV foundation model の双璧**：

| | CLIP | SAM | DINOv2/v3 |
|---|---|---|---|
| 学習データ | 4 億画像-テキスト対（Web crawl） | 1.1B マスク（モデル支援アノテーション） | 142M / 1689M キュレーション画像 |
| 学習目的 | 対比学習（[[concepts/contrastive-learning]]） | promptable segmentation（[[concepts/promptable-segmentation]]） | 自己蒸留 + MIM |
| 主タスク | 分類・検索 | セグメンテーション | 密予測・特徴量 |
| ゼロショット手法 | テキストプロンプト | プロンプトエンジニアリング（点/ボックス/テキスト） | 線形プローブ |
| 教師あり vs SSL | WSL | **教師あり**（SA-1B はモデル支援アノテーション） | 純粋 SSL |

SAM は **「データを生成できれば教師ありで十分」** という第 3 のパラダイムを示した。CRFM の "foundation model = SSL" 前提を §8 で明示的に修正している。

## 限界（論文 §8 + コミュニティ知見）

1. **微細構造の見逃し** / 小さな切断成分の幻覚
2. **境界が鮮明でない**: FocalClick のようにズームインする手法には敵わない
3. **多数点プロンプトでは専用対話的セグメンタに劣る**
4. **画像エンコーダ重い**: 全体で実時間ではない（MobileSAM/EfficientSAM が解決を試みる）
5. **テキスト → マスクは proof-of-concept**: 完全に頑健ではない
6. **セマンティック/パノプティックの簡単なプロンプト設計が不明**
7. **modal vs amodal の選択不可** — データセット規約は zero-shot で学習できない
8. **特定ドメインでは専門ツールが勝つ**: ilastik（バイオイメージング）等

## 関連ページ

- [[sources/segment-anything]] — 原典の要約
- [[translations/segment-anything]] — 全文和訳
- [[entities/sam-2]] — SAM の上位互換（動画対応、Hiera エンコーダで画像も 6× 高速）
- [[entities/sam-3]] — さらなる後継（PCS タスク追加、PE backbone、SA-Co データセット）
- [[concepts/promptable-segmentation]] — SAM が定義したタスクパラダイム
- [[entities/sa-1b]] — 訓練データ
- [[entities/mae]] — 画像エンコーダ初期化に使用
- [[entities/clip]] — テキストプロンプトに使用
- [[concepts/vision-transformer]] — ViT-H/16 がバックボーン
- [[concepts/foundation-model]] — 第 3 の foundation model パラダイム
- [[concepts/zero-shot-transfer]] — ゼロショット転移をセグメンテーションで実現
