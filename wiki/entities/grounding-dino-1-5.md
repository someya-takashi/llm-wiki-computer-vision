---
type: entity
entity_kind: model
aliases: [Grounding DINO 1.5, GD 1.5, Grounding DINO 1.5 Pro, Grounding DINO 1.5 Edge, GroundingDINO 1.5]
tags: [object-detection, open-set-detection, grounding-dino, edge-deployment, vit-l, efficientvit, idea-research]
related: [[concepts/object-detection]], [[concepts/zero-shot-transfer]], [[concepts/foundation-model]], [[concepts/vision-transformer]]
sources: [[sources/grounding-dino-1-5]]
updated: 2026-05-28
---

# Grounding DINO 1.5

## 概要

**Grounding DINO 1.5** は IDEA Research（Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Lei Zhang ら）が 2024 年 5 月に公開した **[[entities/grounding-dino|Grounding DINO]] の Pro/Edge 双子拡張**。論文タイトルの "Edge" は **「分野の先端（edge）を進化させる」と「エッジコンピューティング（edge）に持っていく」の二重の意味**。

- **Grounding DINO 1.5 Pro**: ViT-L backbone + Grounding-20M で **COCO ZS 54.3 AP / LVIS-mv 55.7 AP / ODinW35 30.2** の精度 SOTA
- **Grounding DINO 1.5 Edge**: EfficientViT-L1 + 新 efficient feature enhancer で **Orin NX 10+ FPS / A100 TRT 75.2 FPS、LVIS-mv 36.2 AP**（YOLO-Worldv2-L 32.9 を超える）

---

## 論文情報

- 論文: "Grounding DINO 1.5: Advance the 'Edge' of Open-Set Object Detection"
- arXiv: 2405.10300 / preprint May 2024
- API リポジトリ: <https://github.com/IDEA-Research/Grounding-DINO-1.5-API>
- プロジェクト: <https://deepdataspace.com/home>
- メーカー: International Digital Economy Academy (IDEA) Research（北京）
- 著作権・ライセンス: API は無料利用可（商用利用は要確認）

---

## モデルバリアント

### Grounding DINO 1.5 Pro

| 項目 | 値 |
|---|---|
| 画像 backbone | **ViT-L**（事前学習済み） |
| テキスト backbone | BERT-base |
| 訓練データ | **Grounding-20M**（20M+ grounding 画像、IDEA 独自データエンジン） |
| Fusion | Deep early fusion + 改善された負例サンプリング |
| 用途 | 最高精度 open-vocab 検出 |

### Grounding DINO 1.5 Edge

| 項目 | 値 |
|---|---|
| 画像 backbone | **EfficientViT-L1**（[Cai et al., 2023]） |
| テキスト backbone | BERT-base |
| 訓練データ | **Grounding-20M**（Pro と同じ） |
| Feature enhancer | **Efficient Feature Enhancer**（新ネック） |
| 用途 | エッジデバイス（NVIDIA Orin NX 等）でのリアルタイム open-vocab 検出 |

### Efficient Feature Enhancer の技術詳細

[[entities/grounding-dino|Grounding DINO]] の重い feature enhancer の 3 つの最適化：

1. **P5 のみ Cross-Modality 融合**: 低レベル P3/P4 特徴は意味情報に欠ける → cross-modality 融合を **P5 のみ** に制限（トークン数を大幅削減）
2. **Deformable → Vanilla Self-Attention**: deformable attention はエッジ GPU で実装複雑 → vanilla self-attn に置換（TensorRT で簡単に最適化可能）
3. **Cross-Scale Feature Fusion**: P5 で融合した特徴と P3/P4 の低レベル特徴を [Zhao et al., RT-DETR, 2023] のモジュールで統合 → マルチスケール検出能力を維持

---

## 主要結果

### COCO ゼロショット転送（表 1）

| Model | Backbone | Pre-train | COCO ZS AP |
|---|---|---|---|
| Grounding DINO | Swin-L | O365+OID+GoldG | 52.5 |
| OmDet-Turbo-B | ConvNeXt-B | O365+GoldG+ | 53.4 |
| YOLO-World-L | YOLOv8-L | O365+GoldG+CC3M | 45.1 |
| T-Rex2 (text) | Swin-L | O365+OID+GoldG+CC3M+SBU+LAION | 52.2 |
| **Grounding DINO 1.5 Pro** | **ViT-L** | **Grounding-20M** | **54.3** |

**+1.8 AP over Grounding DINO Swin-L**。

### LVIS ゼロショット転送（表 1）

| Model | Backbone | LVIS-minival AP | APr | APc | APf | LVIS-val AP |
|---|---|---|---|---|---|---|
| MDETR | RN101 | 22.5 | 7.4 | 22.7 | 25.0 | - |
| GLIP-L | Swin-L | 37.3 | 28.2 | 34.3 | 41.5 | 26.9 |
| Grounding DINO-T | Swin-T | 27.4 | 18.1 | 23.3 | 32.7 | - |
| YOLO-World-L | YOLOv8-L | 35.4 | 27.6 | 34.1 | 38.0 | - |
| DetCLIPv2 | Swin-L | 44.7 | 43.1 | 46.3 | 43.7 | 36.6 |
| DetCLIPv3 | Swin-L | 48.8 | 49.9 | 49.7 | 47.8 | 41.4 |
| T-Rex2 (text) | Swin-L | 54.9 | 49.2 | 54.8 | 56.1 | 45.8 |
| **Grounding DINO 1.5 Pro** | **ViT-L** | **55.7** | **56.1** | **57.5** | **54.1** | **47.6** |

**LVIS-minival で +6.9 AP over DetCLIPv3 SOTA**, **LVIS-val で +6.2 AP**。

### ODinW（表 1, 2, 4）

| Model | ODinW35 (35 sets) | ODinW13 (13 sets) |
|---|---|---|
| GLIP-L | - | 52.1 |
| Grounding DINO Swin-L | 26.1 | 56.9 |
| Florence | 25.8 | - |
| OmDet-Turbo-B | 30.1 | 54.7 |
| **GD 1.5 Pro (ZS)** | **30.2** | **58.7** |
| **GD 1.5 Pro (fine-tune)** | **70.6 (+40.4)** | **72.4 (+13.7)** |

**ODinW35 ZS で 30.2** の新記録 + GD Swin-L の 26.1 から +4.2 AP。Fine-tune で **70.6 AP** という巨大ジャンプ。

### Edge モデル（表 5）

| Model | Backbone | Input | COCO | LVIS-mv | A100 (PyT/TRT FPS) | Orin NX (FPS) |
|---|---|---|---|---|---|---|
| Grounding DINO-T | Swin-T | 800×1333 | 48.4 | 27.4 | 9.4 / 42.6 | **1.1** |
| YOLO-Worldv2-S | YOLOv8-S | 640 | - | 22.7 | 47.4 / - | - |
| YOLO-Worldv2-M | YOLOv8-M | 640 | - | 30.0 | 42.7 / - | - |
| YOLO-Worldv2-L | YOLOv8-L | 640 | - | **32.9** | 37.4 / - | - |
| OmDet-Turbo-T (lang cache) | Swin-T | 640 | 42.5 | 30.3 | 21.5 / **140.0** | - |
| **GD 1.5 Edge** | **EfficientViT-L1** | **640** | **42.9** | **33.5** | **21.7 / 111.6** | **10.7** |
| **GD 1.5 Edge** | **EfficientViT-L1** | **800×1333** | **45.0** | **36.2** | **18.5 / 75.2** | **5.5** |

**重要**:
- **640×640 で LVIS-mv 33.5 AP at Orin NX 10.7 FPS** — エッジで実用的
- **800×1333 で LVIS-mv 36.2 AP**（YOLO-Worldv2-L 32.9 を超える + Orin NX 5.5 FPS）
- **Grounding DINO-T (1.1 FPS Orin) を 10 倍高速化** + 精度も向上

### Fine-tuning 結果（表 3）

| Setting | LVIS-mv | LVIS-val | ODinW35 | ODinW13 |
|---|---|---|---|---|
| **GD 1.5 Pro (zero-shot)** | 55.7 | 47.6 | 30.2 | 58.7 |
| **GD 1.5 Pro (fine-tune)** | **68.1 (+12.4)** | **63.5 (+15.9)** | **70.6 (+40.4)** | **72.4 (+13.7)** |
| DetCLIPv3 (fine-tune) | 60.5 | - | - | 72.1 |
| GLIPv2 (fine-tune) | 59.8 | - | - | 70.4 |

**Fine-tune 後**:
- **LVIS-mv 68.1 AP**: DetCLIPv3 fine-tune (60.5) から **+7.6 AP**
- **LVIS-val 63.5 AP**: 同等規模で SOTA
- **ODinW35 70.6 AP**: **未達成領域への進出**（ゼロショット 30.2 → fine-tune 70.6 という巨大ジャンプ）

---

## アーキテクチャ比較

### Grounding DINO vs Grounding DINO 1.5 Pro

| 軸 | Grounding DINO | Grounding DINO 1.5 Pro |
|---|---|---|
| 画像 backbone | Swin-T/L | **ViT-L** |
| Fusion phases | A+B+C（tight 3-phase fusion） | **A+B+C を保持** |
| Sub-sentence text | ✓ | ✓ |
| 訓練データ | O365+OID+GoldG+Cap4M+COCO+RefC（~27M） | **Grounding-20M**（質を上げた選別データ） |
| 負例サンプリング | 標準 | **改善された戦略**（hallucination 抑制） |
| COCO ZS | 52.5 (L) | **54.3** |
| LVIS-mv ZS | - | **55.7** |
| ODinW35 | 26.1 (L) | **30.2** |

### Grounding DINO 1.5 Edge vs YOLO-World

| 軸 | YOLO-Worldv2-L | Grounding DINO 1.5 Edge |
|---|---|---|
| Backbone | YOLOv8-L (CNN) | EfficientViT-L1 (Transformer) |
| Text encoder | CLIP（凍結） | BERT-base |
| Fusion | RepVL-PAN (max-sigmoid attention) | Efficient Feature Enhancer (P5 のみ) |
| Inference paradigm | Prompt-Then-Detect (re-param) | 標準 (language cache はオプション) |
| Input size | 640×640 | 640×640 or 800×1333 |
| LVIS-mv ZS | 32.9 | **33.5 (640) / 36.2 (800×1333)** |
| A100 TRT FPS | - | 111.6 / 75.2 |
| Orin NX FPS | - | **10.7 / 5.5** |

**両モデルともリアルタイム open-vocab を実現**しているが、GD 1.5 Edge は **EfficientViT + efficient feature enhancer** という Transformer 系の最適化、YOLO-World は **YOLOv8 + RepVL-PAN + prompt-then-detect** という CNN 系の最適化。

---

## 設計判断の重要ポイント

### 1. Early Fusion vs Late Fusion のトレードオフ

論文 §2.1.1 で明示的に議論：

| 軸 | Early Fusion | Late Fusion |
|---|---|---|
| 検出再現率 | **高** | 低 |
| Box 精度 | **高** | 低 |
| Hallucination | **多い** ← 欠点 | 少 |
| 訓練の難しさ | 容易 | 難（vision-language alignment） |

**Pro モデルの解**: early fusion を保持 + **改善された負例サンプリング戦略**（訓練中の負例の割合を増やす）→ early の長所を保ちつつ hallucination を緩和。

### 2. Edge モデルでの 3 段階効率化

```
Step 1: Cross-modality fusion を P5 のみに制限
        → Lite-DETR の知見（低レベル特徴は意味情報に欠ける）
        → トークン数大幅削減

Step 2: Deformable self-attn → Vanilla self-attn
        → エッジ GPU での実装複雑さを解消
        → TensorRT 等の標準最適化ツール適用可

Step 3: Cross-Scale Feature Fusion で P3/P4 統合
        → マルチスケール検出能力を維持
        → 小物体検出も対応
```

### 3. EfficientViT-L1 の採用

[Cai et al., MIT, 2023] の **EfficientViT-L1**：
- 通常の ViT-L より大幅に高速
- マルチスケール特徴抽出に最適化
- エッジ GPU での実行効率が良い
- Transformer 系の利点（事前学習リッチ、柔軟）も享受

---

## 主要な貢献まとめ

1. **「精度（Pro）と速度（Edge）の両極を 1 モデルスイートで統合」**: GLIP / Grounding DINO 系統と YOLO-World 系統の双方の領分を吸収
2. **ViT-L backbone への切り替え**: Grounding DINO の Swin-L から、より下流タスクに強い純粋 Transformer へ
3. **Grounding-20M データセット**: 20M+ 高品質 grounding 画像（IDEA 独自データエンジン）
4. **Early fusion + 改善された負例サンプリング**: hallucination 問題を緩和
5. **Efficient Feature Enhancer**: P5 のみ融合 + vanilla self-attn + cross-scale fusion でエッジ展開を可能に
6. **EfficientViT-L1 の採用**: Edge モデルでのリアルタイム性能の基盤
7. **新ベンチマーク ODinW35**: ODinW を 13 → 35 データセットに拡張
8. **COCO/LVIS/ODinW すべてで新記録**: 精度の幅広い領域で SOTA
9. **Orin NX で 10+ FPS の open-vocab 検出**: 自律走行・ロボティクス応用への扉

---

## 限界と批判

- **Grounding-20M の詳細非公開**: IDEA の商用データエンジンが背景、完全再現困難
- **モデル重みは API 経由のみ**: 完全オープンソースではない、商用利用に制約
- **Pro モデルの訓練コスト**: ViT-L + 20M データの再現には大量の GPU
- **Edge モデルでも一般用途では不十分**: Orin NX 10.7 FPS、リアルタイム要求が厳しい応用には届かない
- **早期融合の hallucination 問題は完全解決ではない**: 緩和のみ
- **DetCLIPv3 との比較条件の差**: ViT-L vs Swin-L というベース差もあり、純粋なアルゴリズム改善の寄与度は不明

---

## 関連ページ

- [[sources/grounding-dino-1-5]] — 原典の要約
- [[translations/grounding-dino-1-5]] — 原典の和訳
- [[entities/grounding-dino]] / [[sources/grounding-dino]] — 直接の前身
- [[entities/yolo-world]] / [[sources/yolo-world]] — 速度志向の競合（Edge と直接対比）
- [[entities/dino-detector]] / [[sources/dino-detector]] — DETR ファミリーの集大成
- [[entities/glip]] / [[sources/glip]] — phrase grounding パラダイムの祖
- [[entities/sam-3]] / [[sources/sam-3]] — より広い PCS タスク
- [[entities/dino-x]] / [[sources/dino-x]] — **直接の後継**（2024 Nov、IDEA Research の unified perception model、3 プロンプト + 4 ヘッド）
- [[concepts/object-detection]] — 物体検出全体
- [[concepts/zero-shot-transfer]] — ゼロショット転送
- [[concepts/vision-transformer]] — ViT-L backbone の基盤
