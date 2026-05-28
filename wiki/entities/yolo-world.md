---
type: entity
entity_kind: model
aliases: [YOLO-World, YOLOWorld, YOLO World]
tags: [yolo, real-time, open-vocabulary, object-detection, tencent, clip-text-encoder, repvl-pan]
related: [[concepts/object-detection]], [[concepts/zero-shot-transfer]], [[concepts/foundation-model]]
sources: [[sources/yolo-world]]
updated: 2026-05-28
---

# YOLO-World

## 概要

**YOLO-World** は Tencent AI Lab + 華中科技大学（Tianheng Cheng, Lin Song, Yixiao Ge ら）が CVPR 2024 で発表した、**リアルタイム open-vocabulary 物体検出器**。**YOLOv8 を CLIP text encoder + 新ネック RepVL-PAN で open-vocabulary 化**し、**Prompt-Then-Detect パラダイム** で **52 FPS V100 で LVIS 35.4 AP** を達成（[[entities/grounding-dino|Grounding DINO-T]] の **35×、[[entities/glip|GLIP-T]] の 433×** の高速化）。**実用デプロイメント向けの open-vocab 検出を切り拓いた研究**。

- 論文: "YOLO-World: Real-Time Open-Vocabulary Object Detection"
- arXiv: 2401.17270 / CVPR 2024
- コード: <https://github.com/AILab-CVC/YOLO-World>
- メーカー: Tencent AI Lab + ARC Lab Tencent PCG + 華中科技大学
- ライセンス: GPL-3.0（コード）、Apache 2.0（モデル重み）

---

## モデルバリアント

| Model | Backbone | Params (re-param/orig) | FPS (re-param/orig) | LVIS ZS AP |
|---|---|---|---|---|
| **YOLO-World-S** | YOLOv8-S | **13M** / 77M | **74.1** / 19.9 | 26.2 |
| **YOLO-World-M** | YOLOv8-M | **29M** / 92M | **58.1** / 18.5 | 31.0 |
| **YOLO-World-L** | YOLOv8-L | **48M** / 110M | **52.0** / 17.6 | 35.0 |
| **YOLO-World-L** | YOLOv8-L (+CC3M) | **48M** / 110M | **52.0** / 17.6 | **35.4** |

- **括弧内が re-parameterize 版**（推論時のサイズ・速度）
- **括弧外がオリジナル版**（text encoder 込み）
- FPS は **NVIDIA V100、TensorRT なし** で測定

### アーキテクチャ詳細

| コンポーネント | 内容 |
|---|---|
| 画像 backbone | **YOLOv8 Darknet**（S/M/L） |
| ネック | **RepVL-PAN**（YOLOv8 PAN + T-CSPLayer + I-Pooling Attention） |
| Multi-scale 特徴 | 3 スケール（C3/C4/C5 → P3/P4/P5） |
| Text encoder | **CLIP-base (ViT-base) text tower、凍結** |
| Text 表現 | $W \in \mathbb{R}^{C \times D}$（$C$ = 名詞数、$D$ = 埋め込み次元） |
| Head | **Decoupled head**（2× 3×3 conv で bbox $b_k$ と object embedding $e_k$ を回帰） |
| Text Contrastive Head | $s_{k,j} = \alpha \cdot \texttt{L2}(e_k) \cdot \texttt{L2}(w_j)^{\top} + \beta$ |
| Online vocabulary 最大数 | 80 名詞/サンプル |
| 損失 | region-text contrastive + IoU + DFL |

---

## RepVL-PAN の詳細

### T-CSPLayer（Text-guided CSPLayer）

YOLOv8 の C2f（CSPLayer）に **max-sigmoid attention** を追加：

$$
X_l^{\prime} = X_l \cdot \delta\left(\max_{j \in \{1..C\}}(X_l W_j^{\top})\right)^{\top}
$$

- 各画像位置で最も関連するテキスト token のスコアを取り、sigmoid で 0-1 にしてチャネル方向で画像特徴をスケーリング
- **softmax ではなく max**: 検出では複数カテゴリが同時存在しうるため、token 間の競合を避ける

### I-Pooling Attention（Image-Pooling Attention）

軽量な cross-attention でテキスト埋め込みを画像で更新：

```
{X_3, X_4, X_5} → 各々 3×3 max pooling → 9 patches each → 27 patch tokens X̃ ∈ R^{27 × D}
W ← W + MultiHead-Attention(W, X̃, X̃)
```

**わずか 27 tokens** で画像の概略をテキスト側に注入する効率的設計。

### Re-parameterization

推論時、$W$ は事前計算されているので：

- **T-CSPLayer の $X_l W_j^{\top}$**: $W$ を 1×1 conv の重みとして見なし、**通常の conv 演算**に変換
- **I-Pooling Attention**: $\tilde{X}$ は画像のみから計算、$W$ との attention は事前計算分の重み和に

→ **text encoder を完全に削除でき、通常の YOLO ネットワークと同じ計算グラフ** になる。TensorRT 等の高速化ツールが完全に適用可能。

---

## 訓練設定

| 項目 | 値 |
|---|---|
| Optimizer | AdamW |
| 初期学習率 | 0.002 |
| Weight decay | 0.05 |
| Epoch | 100（事前学習）/ 80（fine-tune） |
| バッチサイズ | 512（事前学習）/ 標準（fine-tune） |
| GPU | 32× V100 |
| データ拡張 | color augmentation, random affine, random flip, mosaic（4 images） |
| Text encoder | **凍結**（事前学習中） |
| Mosaic 最大語彙 | 80 名詞 |
| Label assignment | task-aligned label assignment（TOOD） |

---

## 訓練データ（表 1）

| Dataset | Type | Vocab | Images | Annotations |
|---|---|---|---|---|
| **Objects365 V1** | Detection | 365 | 609k | 9,621k |
| **GQA** | Grounding | open | 621k | 3,681k |
| **Flickr30k** | Grounding | open | 149k | 641k |
| **CC3M† (sub)** | Image-Text | open | **246k** | **821k** |

**CC3M†** = GLIP-L で疑似ラベリングした 246K subset（**GLIP の自己学習パイプラインを継承**）。
**GoldG** = GQA + Flickr30k = MDETR キュレーション（COCO 除外）

---

## 主要結果

### LVIS ゼロショット（表 2）— 最重要

| Method | Backbone | Params | FPS | AP | APr |
|---|---|---|---|---|---|
| GLIP-T | Swin-T | 232M | **0.12** | 24.9 | 17.7 |
| GLIPv2-T | Swin-T | 232M | **0.12** | 26.9 | - |
| Grounding-DINO-T | Swin-T | 172M | **1.5** | 25.6 | 14.4 |
| DetCLIP-T | Swin-T | 155M | **2.3** | 34.4 | 26.9 |
| **YOLO-World-S** | YOLOv8-S | **13M** | **74.1** | 26.2 | 19.1 |
| **YOLO-World-M** | YOLOv8-M | **29M** | **58.1** | 31.0 | 23.8 |
| **YOLO-World-L** | YOLOv8-L | **48M** | **52.0** | 35.0 | 27.1 |
| **YOLO-World-L (+CC3M†)** | YOLOv8-L | **48M** | **52.0** | **35.4** | **27.6** |

**速度の比較**:
- YOLO-World-L (52 FPS) vs Grounding-DINO-T (1.5 FPS) = **34.7× 高速化**
- YOLO-World-L (52 FPS) vs GLIP-T (0.12 FPS) = **433× 高速化**
- YOLO-World-L (52 FPS) vs DetCLIP-T (2.3 FPS) = **22.6× 高速化、AP は +1.0**

### COCO ゼロショット転送 + Fine-tune（表 6）

| Method | Setting | AP | FPS |
|---|---|---|---|
| YOLOv8-L (scratch) | Closed-set | 52.9 | 159 |
| **YOLO-World-L (ZS)** | Zero-shot | **45.1** | - |
| **YOLO-World-L (FT, w/o RepVL-PAN)** | Fine-tuned | **53.3** | **156** |

**Fine-tune 後は YOLOv8-L (scratch) を上回る**。

### LVIS Fine-tune（表 7）

LVIS-base で fine-tune → LVIS-novel（rare）で評価：

| Method | AP | APr |
|---|---|---|
| YOLOv8-L (full LVIS oracle) | 26.9 | 10.2 |
| ViLD | 27.8 | 16.7 |
| BARON | 29.5 | 23.2 |
| **YOLO-World-L** | **34.1** | **20.4** |

**先行 two-stage SOTA を one-stage で上回る**。

### OVIS（Open-Vocabulary Instance Segmentation、表 8）

LVIS-base のマスク注釈で fine-tune：

| Setting | Fine-tune | Mask AP | Mask APr |
|---|---|---|---|
| LVIS-base | Seg Head only | 19.1 | **14.2 (ZS 維持)** |
| LVIS-base | All modules | **28.7** | 15.0 |

「Seg head only」では事前学習の open-vocab 能力を保持、「All」では LVIS 適合は良いが ZS 性能 -0.6 APr 劣化。

### ablation: RepVL-PAN（表 4）

| GQA | T→I | I→T | AP | APr |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 22.4 | 14.5 |
| ✗ | ✓ | ✓ | 23.5 (+1.1) | 16.2 (+1.7) |
| ✓ | ✗ | ✗ | 29.7 | 21.0 |
| ✓ | ✓ | ✓ | **31.9 (+2.2)** | **22.5 (+1.5)** |

**GQA（豊富なテキスト）があるほど RepVL-PAN の効果が増す**。

### ablation: Text Encoder（表 5）— 重要

| Text Encoder | Frozen? | AP | APr |
|---|---|---|---|
| BERT-base | Frozen | 14.6 | 3.4 |
| BERT-base | Fine-tune | 18.3 | 6.6 |
| **CLIP-base** | **Frozen** | **22.4** | **14.5 (+10.1)** |
| CLIP-base | Fine-tune | 19.3 | 8.6 |

**CLIP は BERT を APr で +10.1 圧倒**、ただし fine-tune すると CLIP は事前学習の汎化能力を破壊して -3.1 AP。

---

## 主要な貢献まとめ

1. **YOLO 系を open-vocab 化した初の本格的研究**: YOLOv8 を base に CLIP text encoder + RepVL-PAN
2. **RepVL-PAN**: T-CSPLayer（max-sigmoid attention）+ I-Pooling Attention（27 patch tokens）の **軽量で効果的な vision-language 融合**
3. **Prompt-Then-Detect パラダイム**: テキスト encoder を推論時に削除、テキスト埋め込みを **モデル重みに re-parameterize**
4. **Region-Text Contrastive Loss**: 検出 / grounding / image-text データを region-text ペアに統一
5. **LVIS ゼロショット 35.4 AP at 52 FPS**: Grounding-DINO-T の 34.7× 速、AP も上回る
6. **小型モデルでの open-vocab を実証**: YOLO-World-S 13M params で GLIP-T 232M と同等 AP
7. **CLIP text encoder の決定的優位を実証**: BERT 比 APr +10.1
8. **OVIS への拡張**: Open-Vocabulary Instance Segmentation の新ベンチマーク提示
9. **エッジデバイスでの open-vocab 検出を可能に**: 自律走行、ロボティクス、AR/VR の応用を切り拓く

---

## GLIP / Grounding DINO との比較（精度 vs 速度）

| 軸 | GLIP-T | Grounding-DINO-T | **YOLO-World-L** |
|---|---|---|---|
| 系統 | DyHead + BERT | DINO 検出器 + BERT | **YOLOv8 + CLIP** |
| Backbone | Swin-T | Swin-T | **YOLOv8-L Darknet** |
| Params | 232M | 172M | **48M** |
| FPS (V100) | 0.12 | 1.5 | **52** |
| LVIS ZS AP | 24.9 | 25.6 | **35.4** |
| LVIS ZS APr | 17.7 | 14.4 | 27.6 |
| デプロイ | 困難（>1 sec/frame） | 困難 | **エッジ可** |
| 用途 | アノテーション・サーバ | 高精度推論 | **実用検出** |

**「YOLO-World は速度と精度の両方で GLIP / Grounding DINO を上回った稀有な例」** — 通常 DETR 系 vs YOLO 系は精度 vs 速度のトレードオフだが、YOLO-World は両得。

---

## 既存 wiki への接続

### CLIP の object-level 拡張系統

[[entities/clip]] → [[entities/glip]] → [[entities/grounding-dino]] → **YOLO-World** という流れの中で：

| 軸 | CLIP | GLIP | Grounding DINO | **YOLO-World** |
|---|---|---|---|---|
| タスク | 分類 | 検出 + grounding | 検出 + grounding + REC | 検出 + REC |
| Text encoder | CLIP（自前） | BERT | BERT | **CLIP（流用）** |
| 検出器 | n/a | DyHead | DINO 検出器 | **YOLOv8** |
| FPS | n/a | 0.12 | 1.5 | **52** |

**YOLO-World は CLIP text encoder への原点回帰** をしつつ、検出器側を最も軽量化（DyHead/Swin-L → DINO/Swin-L → YOLOv8/Darknet）。

### GLIP の self-training パイプラインの継承

YOLO-World は **GLIP-L を疑似ラベル生成 teacher として使用**：

```
GLIP-T (teacher) → CC4M に疑似ラベル → GLIP-L 訓練（[[sources/glip]] の self-training）
                                       ↓
                            ★ YOLO-World が GLIP-L をさらに teacher として使用 ★
                                       ↓
                            CC3M に疑似ラベル → YOLO-World 訓練
```

これは「**高精度モデルから高速モデルへの実質的蒸留**」の重要な実用パターン。

### YOLO 系統での open-vocab 化

[[concepts/object-detection]] の YOLO 系統で、YOLO-World は **YOLOv8 → open-vocab YOLOv8** という拡張：

- YOLOv1（2016）→ YOLOv8（2023）: closed-set でリアルタイム標準
- **YOLO-World（2024）: YOLOv8 を open-vocab 化**
- 推論時は YOLOv8 と同じ速度で実行可能

### SAM 3 / PE との対照

[[entities/sam-3]] / [[entities/perception-encoder]] の「高精度・大型」と対照的な「実用・軽量」路線。**用途別の使い分け**:

- 自律走行・ロボティクス・AR/VR: **YOLO-World**
- アノテーション支援・コンテンツ理解: SAM 3 / PE

### Foundation Model 時代のリアルタイム検出

[[concepts/foundation-model]] 時代の「**高精度 foundation model から軽量モデルへの知識蒸留**」の典型例。GLIP-L の知識が CC3M 疑似ラベルを介して YOLOv8 に転移。

---

## 限界と批判

- **APr は依然として DetCLIP-T の特化能力に競合できない局面あり**: 27.6 vs 26.9 は僅差
- **疑似ラベル品質依存**: GLIP-L の box 品質に上限
- **COCO fine-tune で RepVL-PAN を除去**: 80 カテゴリでは言語融合の意義が薄い、しかし open-vocab 検出器としての本質を失う
- **CLIP text encoder の fine-tune が逆効果**: 大規模・多様データでないと汎化を破壊
- **テキスト依存**: 推論前にプロンプト定義必要（ただし offline vocabulary で 1 回のみ）
- **動画対応なし**: SAM 3 のような時間追跡なし
- **マスク生成は OVIS で対応**: マスク head を fine-tune する必要

---

## 関連ページ

- [[sources/yolo-world]] — 原典の要約
- [[translations/yolo-world]] — 原典の和訳
- [[concepts/object-detection]] — 物体検出全体（YOLO 系統と open-vocab 系統の交差点）
- [[concepts/zero-shot-transfer]] — YOLO-World のゼロショット転送
- [[concepts/foundation-model]] — open-vocab 検出は CV foundation model の重要分野
- [[entities/clip]] / [[sources/clip]] — YOLO-World は CLIP text encoder を流用
- [[entities/glip]] / [[sources/glip]] — YOLO-World は GLIP-L を疑似ラベリング teacher として活用
- [[entities/grounding-dino]] / [[sources/grounding-dino]] — 同じ open-vocab 系統、精度 vs 速度の対比
- [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — Grounding DINO の Pro/Edge スイート（2024 May）、Edge モデルが YOLO-World と直接対比
- [[entities/dino-x]] / [[sources/dino-x]] — Grounding DINO 系統の到達点（2024 Nov）、Edge モデルが YOLO-Worldv2-L を精度・速度両方で凌駕
- [[entities/sam-3]] / [[sources/sam-3]] — 高精度・大型路線との対照
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — 同じく対照的な大型路線
