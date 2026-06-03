---
type: concept
aliases: [Object Detection, 物体検出, 物体検知]
tags: [task, dense-prediction, localization]
related: ["[[vision-transformer]]", "[[foundation-model]]", "[[promptable-segmentation]]", "[[promptable-concept-segmentation]]"]
sources: ["[[sources/detr]]", "[[sources/segment-anything]]", "[[sources/sam-3]]", "[[sources/perception-encoder]]"]
updated: 2026-05-28
---

# Object Detection（物体検出）

## 一言で

**「画像内の各物体について、バウンディングボックス（位置）とクラスラベルを **集合として** 予測する」タスク**。CV の中核タスクの 1 つで、ImageNet 分類が「画像 1 枚に 1 つのラベル」を返すのに対し、物体検出は「**画像 1 枚に複数の物体の (class, bbox)**」を返す。**集合予測（set prediction）** という性質が、深層学習の標準的な「ベクトル出力」設計と相性が悪く、長年 hand-designed なパイプラインに依存してきた。

## なぜ難しいか

- **物体数が画像ごとに変動**: 1 個から 100 個以上まで
- **重複予測の問題**: 同じ物体に複数の候補が出やすい → **NMS（非最大抑制）** が必須に
- **スケールの多様性**: 同じ画像内で「画素 10 個の鳥」と「画面全体の家」が共存
- **集合の順序不変性**: 出力に順序がない → 損失関数の設計が非自明
- **クラス不均衡**: 背景ピクセルが圧倒的多数

## 評価指標

| 指標 | 説明 |
|---|---|
| **AP**（Average Precision）| Precision-Recall 曲線下の面積 |
| **mAP**（mean AP）| 全クラス平均 AP |
| **AP_50 / AP_75** | IoU 閾値 0.5 / 0.75 での AP |
| **AP_S / AP_M / AP_L** | Small / Medium / Large 物体に対する AP（COCO 標準）|
| **IoU**（Intersection over Union）| 予測ボックスと ground truth の重なり面積 / 和集合面積 |

主要ベンチマーク: **COCO**（80 クラス、~118K 訓練画像）、**LVIS**（1203 クラスで長尾分布）、**Objects365**（365 クラス）、**Open Images**（600 クラス）。

## 歴史的系譜

### 前史（深層学習以前）

- **HOG + SVM**（[Dalal & Triggs, 2005]）: 歩行者検出
- **DPM**（Deformable Part Models, [Felzenszwalb et al., 2010]）: 構造ベースの古典的手法

### 1. R-CNN ファミリー（Two-stage）

```
2014  R-CNN (Girshick et al.)         — Selective Search で proposal、CNN で各 proposal を分類
2015  Fast R-CNN (Girshick)            — proposal 生成は分離、RoI Pooling で 1 回 CNN
2015  Faster R-CNN (Ren et al.)        — RPN で proposal も学習、完全 end-to-end（に近い）
2017  Mask R-CNN (He et al.)           — Faster R-CNN + マスク branch、インスタンスセグ
2017  FPN (Lin et al.)                 — Multi-scale 特徴ピラミッド、小物体性能の根本解決
2020  Cascade R-CNN, HTC               — IoU 閾値段階化で精度向上
```

**特徴**: 高精度だが複雑、多段パイプライン、NMS 依存。

### 2. Single-stage / Anchor-based

```
2016  YOLO (Redmon et al.)             — 画像をグリッドに分け、各セルで直接予測
2016  SSD (Liu et al.)                 — Multi-scale anchor で one-shot 検出
2017  RetinaNet (Lin et al.)           — Focal loss でクラス不均衡を解決
2018  YOLOv3, 2020 YOLOv4/5            — 実用性能トップ、リアルタイム
```

**特徴**: 高速、エッジデバイス向き。NMS と anchor 設計に依存。

### 3. Anchor-free

```
2018  CornerNet (Law & Deng)           — コーナーキーポイントとして検出
2019  CenterNet (Zhou et al.)          — 物体中心の keypoint heatmap
2019  FCOS (Tian et al.)               — fully convolutional anchor-free
```

**特徴**: anchor ハイパラを排除。NMS は依然必要。

### 4. DETR ファミリー（Transformer + Set Prediction）

```
2020 May    DETR (Carion et al., ECCV 2020, [[sources/detr]]) ★転換点★
              — 集合予測損失 + Hungarian + Transformer + 並列デコード
              — NMS, anchor, proposal をすべて排除
              — 大物体 +7.8 AP、小物体 -5.5 AP
              │
2020 Oct    Deformable DETR (Zhu et al., ICLR 2021)
              — Sparse attention、訓練速度 10×、小物体改善
              │
2022        DAB-DETR / DN-DETR
              — Object queries の解釈と訓練の安定化
              │
2023        DINO (detector) (Zhang et al., ICLR 2023, [[sources/dino-detector]] / [[entities/dino-detector]])
              — DETR ファミリーの集大成、CDN + Mixed QS + LFT
              — COCO test-dev 63.3 AP、初の end-to-end Transformer SOTA
              │
2022-2023   DETA / CoDETR
              — 1 対多マッチング / 協調訓練、収束加速・性能改善
              │
2024-2025   PEspatial × DETA (Bolya et al., [[sources/perception-encoder]])
              — COCO 66.0 box AP、シンプル DETR-style で SOTA
2025        SAM 3 × DETR-style detector ([[sources/sam-3]])
              — PCS タスク（テキスト条件付きセグメンテーション）の中核
```

**特徴**: hand-designed components を最小化、Transformer の表現力で検出を end-to-end 化。

### 5. Open-Vocabulary 検出（言語条件付き）

```
2018  Bansal et al. — Glove embedding でゼロショット検出（祖）
2021  ViLD — CLIP を two-stage 検出器に蒸留（CVPR 2022）
2021 Oct  MDETR — DETR にテキスト条件付け（ICCV 2021）
2021 Dec  ★ GLIP ([[sources/glip]] / [[entities/glip]]) ★ — 検出 = phrase grounding 統一、CVPR 2022
            — Self-training で 27M grounding data、COCO ゼロショット 49.8 AP
            — open-vocab 検出パラダイムを確立
2022  OWL-ViT — ViT + DETR、open-vocab 検出（Google）
2023  ★ Grounding DINO ([[sources/grounding-dino]] / [[entities/grounding-dino]]) ★
            — GLIP × [[entities/dino-detector|DINO 検出器]]、Swin-L で COCO ZS 52.5 AP / FT 63.0 AP / ODinW ZS 26.1 SOTA
            — Tight 3-phase fusion (A+B+C) + sub-sentence text representation
2024  MM-Grounding-DINO — マルチモーダル拡張
2024 May  ★ Grounding DINO 1.5 ([[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]]) ★
            — Pro (ViT-L + Grounding-20M, COCO ZS 54.3 / LVIS-mv 55.7 / ODinW35 30.2 SOTA)
            — Edge (EfficientViT-L1 + efficient feature enhancer, Orin NX 10.7 FPS)
            — 「精度と速度の両極を 1 モデルスイートで統合」
2024 Nov  ★ DINO-X ([[sources/dino-x]] / [[entities/dino-x]]) ★
            — IDEA Research の unified perception model（box-only 検出から進化）
            — 3 プロンプト (Text/Visual/Customized) + 4 ヘッド (Box/Mask/Keypoint/Language)
            — Grounding-100M、CLIP text encoder への切替
            — LVIS-mv 59.8 / LVIS rare APr 63.3 (+5.8 over GD 1.6 Pro) SOTA
            — Edge は KD + FP16 で Orin NX 20.1 FPS、YOLO-Worldv2-L を LVIS で +15.3 AP 凌駕
            — Universal Object Prompt で prompt-free 検出という新タスク
```

### 6. Real-Time Open-Vocabulary 検出

```
2024  ★ YOLO-World ([[sources/yolo-world]] / [[entities/yolo-world]]) ★
            — YOLOv8 + CLIP text encoder + RepVL-PAN、52 FPS で LVIS 35.4 AP
            — Prompt-Then-Detect で text encoder を推論時に削除
            — GLIP の精度・Grounding DINO の精度を 30 倍以上の速度で凌駕
            — エッジデバイスでの open-vocab 検出を可能に
2024 May  ★ Grounding DINO 1.5 Edge ★ — EfficientViT-L1 + P5 のみ cross-modal 融合
            — Orin NX 10.7 FPS で LVIS 36.2 AP (YOLO-Worldv2-L 32.9 超え)
            — YOLO-World と並ぶリアルタイム open-vocab の双璧
2024 Nov  ★ DINO-X Edge ★ — EfficientViT-L2 + Knowledge Distillation + FP16 量子化
            — Orin NX 20.1 FPS (GD 1.5 Edge から +87%) で LVIS-mv 48.3 AP
            — YOLO-Worldv2-L を +15.3 AP 凌駕、real-time open-vocab の新 SOTA
```

「高精度・大型」志向の GLIP / Grounding DINO に対し、「実用・軽量」志向の対抗路線。

### 7. Promptable / Foundation Model 時代

```
2023  SAM ([[sources/segment-anything]])           — promptable segmentation、検出への汎化
2024  SAM 2 ([[sources/sam-2]])                    — 動画 promptable segmentation
2025  SAM 3 ([[sources/sam-3]])                    — テキスト名詞句で全インスタンス検出（PCS、GLIP の精神的後継）
2025  PE ([[sources/perception-encoder]])          — alignment tuning + DETA で検出 SOTA
```

物体検出は **「クローズドセット検出」→「DETR の集合予測」→「open-vocab 検出（GLIP）」→「real-time open-vocab 検出（YOLO-World）」→「promptable 検出（SAM）」** という抽象化階段を上っている。同時に **「精度志向（GLIP / Grounding DINO / SAM 3）」と「実用志向（YOLO-World）」** という直交軸も存在。

## DETR の位置づけ — パラダイムシフト

DETR が他のすべての手法と決定的に異なる点：

| 項目 | 従来手法（R-CNN, YOLO 等） | **DETR** |
|---|---|---|
| 検出の見方 | ボックス回帰 + 分類の代理タスク | **集合予測** |
| 重複処理 | **NMS**（後処理）| 訓練レベルで Hungarian 1 対 1 マッチング |
| Anchor | 必須（手設計か anchor-free でも grid 必要） | **不要**（学習可能 object queries）|
| Proposal | Two-stage で必須 | **不要** |
| グローバル推論 | 限定的（FPN, RoIPool）| **Self-attention でグローバル** |
| 大物体性能 | 中程度 | **大幅改善**（COCO で +7.8 AP）|
| 小物体性能 | 強い（FPN 等） | 当初は弱い（Deformable DETR で解決） |
| 訓練時間 | 標準 | **10 倍**（後の研究で短縮） |
| パノプティック化 | 別モデル（PanopticFPN）| マスクヘッド追加で容易 |

> **補足: なぜ DETR が「パラダイムシフト」か** — 物体検出を「単なるベクトル回帰」から「集合の最適輸送（optimal transport）問題」に置き直したことが本質。これにより、(1) hand-designed components が消える、(2) 他の集合予測タスク（パノプティック、視覚 grounding、video tracking）への自然な拡張が可能になる、(3) Transformer の表現力が直接効く、という波及効果が生まれた。

## 現代 CV foundation model における物体検出

2025 年時点で、物体検出は **foundation model 上の「タスクヘッド」** として実装されることが主流：

| Backbone | 検出ヘッド | 代表モデル |
|---|---|---|
| ResNet | Faster R-CNN / Mask R-CNN | 古典的選択 |
| ViT | ViTDet (Mask R-CNN-like) | [[sources/dinov2-learning-robust-visual-features-without-supervision]] / [[sources/dinov3]] |
| ViT | DETR / DETA / CoDETR | [[entities/perception-encoder]] PEspatial |
| ViT + memory | DETR + presence head + tracker | [[entities/sam-3]] |
| Hiera | SAM mask decoder | [[entities/sam-2]] |

**潮流**: foundation model（CLIP, DINOv3, PE）が「凍結された一般特徴」を提供し、その上に DETR スタイルの検出ヘッドを乗せるのが 2025 年の主流。

## 関連する重要技術

- **NMS（Non-Maximum Suppression）**: 古典的な重複除去後処理。DETR で不要に
- **GIoU（Generalized IoU）** [Rezatofighi et al., 2019]: スケール不変ボックス類似度、DETR の核
- **DIoU / CIoU**: GIoU の派生（中心距離・アスペクト比も考慮）
- **Hungarian アルゴリズム**: 二部マッチングを O(N³) で最適解、DETR の損失計算
- **Focal Loss** [Lin et al., 2017]: クラス不均衡対応。RetinaNet の核、SAM でも採用
- **Soft-NMS** [Bodla et al., 2017]: NMS の confidence を IoU に応じて減衰

## 関連ページ

- [[sources/detr]] / [[entities/detr]] — 集合予測パラダイムの祖
- [[concepts/vision-transformer]] — DETR は CNN backbone + Transformer、ViT は完全 Transformer
- [[concepts/foundation-model]] — 現代の検出は foundation model 上のヘッドとして実装
- [[concepts/promptable-segmentation]] — SAM の任意プロンプトベース検出
- [[concepts/promptable-concept-segmentation]] — SAM 3 のテキスト条件検出
- [[sources/segment-anything]] / [[entities/sam]] — Mask R-CNN 系統の現代的後継
- [[sources/sam-3]] / [[entities/sam-3]] — DETR スタイル detector + PCS
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — PEspatial + DETA で COCO 66.0 SOTA
