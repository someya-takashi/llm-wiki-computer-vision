---
type: entity
entity_kind: model
aliases: [Grounding DINO, Grounding-DINO, GroundingDINO, GD]
tags: [object-detection, open-set-detection, dino-detector, glip, tight-fusion, idea, microsoft, vision-language]
related: [[concepts/object-detection]], [[concepts/zero-shot-transfer]], [[concepts/promptable-concept-segmentation]], [[concepts/foundation-model]]
sources: [[sources/grounding-dino]]
updated: 2026-05-28
---

# Grounding DINO

## 概要

**Grounding DINO** は IDEA + 清華大学 + HKUST + Microsoft の共同チーム（Shilong Liu, Zhaoyang Zeng ら、本論文の DINO 検出器グループ + GLIP グループの直接協力）が ECCV 2024 で発表した、**open-set 物体検出器**。**[[entities/glip|GLIP]] の grounded pre-training を、最強の DETR 系検出器 [[entities/dino-detector|DINO 検出器]] に移植** した「正統な統合論文」。3 つの新工夫（**Tight 3-Phase Fusion / Language-Guided Query Selection / Sub-Sentence Level Text Representation**）と **REC タスクへの評価拡張**を提案し、COCO ゼロショット 52.5 AP、ODinW ゼロショット 26.1 AP で SOTA。**SAM 3、MM-Grounding-DINO の直接の祖**。

- 論文: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
- arXiv: 2303.05499 / ECCV 2024
- コード: <https://github.com/IDEA-Research/GroundingDINO>
- メーカー: IDEA + 清華大学 + HKUST + 香港中文大学（深圳）+ Microsoft Research
- ライセンス: Apache 2.0

> **重要な注記**: Grounding DINO の "DINO" は **物体検出器の DINO**（[[entities/dino-detector]]、Zhang et al., ICLR 2023）であり、SSL の DINO（[[entities/dino]]、Caron et al., 2021）とは別物。命名の経緯: **D**ETR with **I**mproved de**N**oising anch**O**r boxes。

---

## モデルバリアント

| Model | Image Backbone | Text Backbone | Model Size | Pre-Training Data |
|---|---|---|---|---|
| **Grounding-DINO-T** | Swin-T | BERT-base | 172M | O365v1（または O365 + GoldG, +Cap4M） |
| **Grounding-DINO-L** | Swin-L | BERT-base | 341M | O365v2 + OI + GoldG（または + Cap4M + COCO + RefC） |

### アーキテクチャ詳細

| コンポーネント | 内容 |
|---|---|
| 画像 backbone | **Swin Transformer**（Tiny or Large、ImageNet-22K 事前学習） |
| テキスト backbone | **BERT-base**（Hugging Face、BPE トークナイザ） |
| Multi-scale 特徴 | 3 スケール（Swin-T: 4× downsample）/ 4 スケール（Swin-L: 4× to 32×） |
| Feature enhancer 層数 | **6** |
| Decoder 層数 | **6** |
| Queries 数 | **900**（DINO 検出器と同じ） |
| 最大テキスト token | **256**（BERT 制限） |
| 隠れ次元 | 256 |
| Attention heads | 8 |
| FFN 次元 | 2048 |

### Feature Enhancer 層の構成

```
画像側: [Deformable self-attn] → [image-to-text cross-attn]
テキスト側: [vanilla self-attn] → [text-to-image cross-attn]
```

### Cross-Modality Decoder 層の構成（DINO 検出器 vs Grounding DINO）

```
DINO 検出器:           [self-attn] → [image cross-attn] → [FFN]
Grounding DINO:        [self-attn] → [image cross-attn] → [TEXT cross-attn] → [FFN]
                                                          ↑ DINO 検出器に対する追加点
```

---

## 3 つの中核技術

### 1. Tight 3-Phase Fusion

| Phase | 名前 | 実装 |
|---|---|---|
| **A** | Neck（feature enhancement） | Feature Enhancer（6 層、X-MHA + Deformable self-attn） |
| **B** | Query Initialization | Language-Guided Query Selection（top-K image-text alignment） |
| **C** | Head（decoder） | Cross-Modality Decoder（image + text cross-attn 両方を毎層） |

**3 フェーズすべてで融合する初の open-set 検出器**。GLIP は Phase A のみ、OV-DETR は Phase B のみだった。

### 2. Language-Guided Query Selection（Phase B、PyTorch スタイル）

```python
# image_features: (bs, num_img_tokens, ndim)
# text_features: (bs, num_text_tokens, ndim)
# num_query: 900

logits = torch.einsum("bic,btc->bit", image_features, text_features)
# (bs, num_img_tokens, num_text_tokens)

logits_per_img_feat = logits.max(-1)[0]
# 各 image token につき最も近い text token のスコア

topk_proposals_idx = torch.topk(logits_per_img_feat, num_query, dim=1)[1]
# テキストに最も関連する K=900 の image positions を選択
```

DINO 検出器の Mixed Query Selection の **言語誘導版**: 物体性スコアの代わりに **テキスト関連性** を選択基準にする。

### 3. Sub-Sentence Level Text Representation

3 つの表現レベルの比較：

| 方式 | 長所 | 短所 | 採用研究 |
|---|---|---|---|
| **Sentence-level** | カテゴリ独立 | 単語粒度を失う | DetCLIP, OWL-ViT |
| **Word-level** | 単語粒度を保つ | カテゴリ間で意図しない attention | **GLIP**, MDETR |
| **Sub-sentence**（提案）| カテゴリ独立 + 単語粒度を両得 | - | **Grounding DINO** |

**実装**: BERT への attention mask で、異なるカテゴリ名の単語間の attention を遮断。"person" の文脈で "bicycle" や "car" の attention は立たないが、"traffic light" 内の "traffic" と "light" の関係は保持。

---

## 訓練設定

| 項目 | 値 |
|---|---|
| optimizer | AdamW |
| 学習率 | 1e-4 |
| 画像 backbone lr | 1e-5 |
| テキスト backbone lr | 1e-5 |
| weight decay | 1e-4 |
| gradient clip max norm | 0.1 |
| 損失係数 (class / bbox / GIoU) | 1.0 / 5.0 / 2.0 |
| Hungarian マッチコスト | 2.0 / 5.0 / 2.0 |
| GPU | 16× V100 (T) / 64× A100 (L) |
| バッチサイズ | 32 (T) / 64 (L) |

---

## 主要結果

### COCO ゼロショット → fine-tune（表 2）

| Model | Pre-train | Zero-Shot | Fine-Tune |
|---|---|---|---|
| GLIP-L | FourODs + GoldG + Cap24M | 49.8 | 60.8 / 61.0 |
| DINO(Swin-L) | O365 | n/a（closed-set のみ） | 62.5 |
| **Grounding-DINO-T** | O365 + GoldG | 48.1 | 57.1 |
| **Grounding-DINO-T** | O365 + GoldG + Cap4M | 48.4 | 57.2 |
| **Grounding-DINO-L** | O365 + OI + GoldG | **52.5** | **62.6 / 62.7** (63.0 with TTA) |
| **Grounding-DINO-L** | + Cap4M + COCO + RefC | **60.7** | 62.6 / - |

**ゼロショット 52.5 AP**: GLIP-L (49.8) を **+2.7 AP** 超え、純粋 DINO Swin-L の closed-set 62.5 にゼロショットで近接。

### LVIS ゼロショット（表 3）

| Model | Pre-train | MiniVal AP | APr | APc | APf |
|---|---|---|---|---|---|
| GLIP-T | O365 + GoldG + Cap4M | 26.0 | 20.8 | 21.4 | 31.0 |
| **Grounding-DINO-T** | O365 + GoldG + Cap4M | **27.4** | 18.1 | 23.3 | 32.7 |
| **Grounding-DINO-L** | フル | **33.9** | 22.2 | 30.7 | 38.8 |

**重要な発見**: APf (frequent) では GLIP を上回るが、**APr (rare) では GLIP に劣る**。論文の解釈: 900 queries 固定が長尾分布に不利。

### ODinW（表 4）

35 データセット平均：

| Model | Backbone | Size | Zero-Shot | Few-Shot | Full-Shot |
|---|---|---|---|---|---|
| GLIP-T | Swin-T | 232M | 19.6 | 38.9 | 62.6 |
| GLIPv2-T | Swin-T | 232M | 22.3 | - | - |
| DetCLIP | Swin-L | 267M | 24.9 | - | - |
| Florence | CoSwinH | ≈841M | 25.8 | - | - |
| DINO-Swin-L | Swin-L | 218M | n/a | - | 68.8 |
| **Grounding-DINO-T** | Swin-T | 172M | 22.3 | **46.4** | **70.7** |
| **Grounding-DINO-L** | Swin-L | 341M | **26.1** | - | - |

- **ゼロショット 26.1**: Florence (841M) を **341M で上回る**
- **Full-shot 70.7**: 純粋 DINO Swin-L (68.8) すらも Swin-T で上回る

### RefCOCO/+/g（表 5、REC）

| Model | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---|---|---|
| MDETR (fine-tune) | 86.75 | 79.52 | 81.64 |
| **Grounding-DINO-T** (+ RefC pre-train, fine-tune) | 89.19 | 81.09 | 84.15 |
| **Grounding-DINO-L** (fine-tune) | **90.56** | **82.75** | **86.13** |

### ablation（表 6）

| 構成 | COCO ZS | LVIS ZS |
|---|---|---|
| Full Model | 46.7 | 16.1 |
| - encoder fusion（Phase A） | 45.8 | **13.1 (-3.0)** |
| - language-guided query selection（Phase B） | 46.3 | **13.6 (-2.5)** |
| - text cross-attention（Phase C） | 46.1 | 14.3 (-1.8) |
| - sub-sentence（word-level） | 46.4 | 15.6 (-0.5) |

**LVIS では Phase A と B が決定的**（合わせて -5.5 AP）。

### DINO 事前学習からの転送（表 7）

| 設定 | COCO ZS | LVIS ZS | ODinW ZS |
|---|---|---|---|
| Grounding-DINO-T from scratch (O365+GoldG) | 48.1 | 25.6 | 20.0 |
| Grounding-DINO-T from pre-trained DINO (O365+GoldG) | 46.4 | **26.1** | 18.5 |

**DINO 事前学習からの転送が LVIS で標準訓練を上回る**。Text/fusion ブロックだけ訓練すれば良いので **計算コスト大幅削減**。

### Closed-set COCO（表 9、Appendix C.1）

| Model | Epochs | AP | AP_S |
|---|---|---|---|
| DINO-4scale | 12 | **49.0** | **32.0** |
| **Grounding DINO (4scale)** | 12 | 48.1 | 30.4 |

**Closed-set では DINO 検出器に -0.9 AP 劣る** — 言語側の追加コンポーネントが最適化を難しくしている。

---

## 訓練データ

| データセット | 種別 | サイズ | 用途 |
|---|---|---|---|
| **Objects365 v1** | 検出（365 クラス） | ~0.6M | Grounding-DINO-T 検出 |
| **Objects365 v2** | 検出（365 クラス） | ~1.7M | Grounding-DINO-L 検出 |
| **OpenImages (OI)** | 検出 | ~2M | Grounding-DINO-L 追加 |
| **COCO** | 検出 | 0.11M | Grounding-DINO-L 追加 fine-tune |
| **GoldG** | grounding | 0.8M | MDETR 由来、Flickr30K + VG |
| **RefC** | REC | RefCOCO + RefCOCO+ + RefCOCOg | Grounding-DINO-L REC 強化 |
| **Cap4M** | 画像-テキスト（GLIP 生成）| 4M | T 用 caption データ |
| **Cap24M** | 画像-テキスト（GLIP 生成）| - | （Grounding DINO は使わず、GLIP-L 比較用） |

**GLIP-L の 27M データ規模には届かない**（論文 §5 で限界として明記）。

---

## Open-Vocabulary 検出ファミリーの中で

```
2018  Bansal et al. — Glove ベース（祖）
2021  CLIP                — image-level open-vocab
2021  ViLD                — CLIP 蒸留型
2021 Oct  MDETR           — DETR + テキスト条件（A+C）
2021 Dec  GLIP            — DyHead ベース、Phase A のみ、検出 = grounding
2022 Apr  OWL-ViT         — ViT + DETR
2022  OV-DETR             — Deformable DETR、Phase B のみ
2022 Nov  GLIPv2          — GLIP + segmentation 拡張
2023 Mar  ★ Grounding DINO（本論文）★ — DINO 検出器 + 3-phase fusion
            │
2024      MM-Grounding-DINO  — マルチモーダル拡張、open-vocab タスク統合
2024      DINO-X             — Grounding DINO 後継、IDEA
2025      ★ SAM 3 ★          — Grounding DINO + presence head + マスク + 動画
2025      PE PEspatial       — DETA decoder（DINO 検出器系統）採用、COCO 66.0 AP
```

**Grounding DINO の位置**: **「open-vocab 検出 + DETR 系最強検出器」の最初の本格的成功例**。SAM 3 や PE PEspatial に直接影響を与えた。

---

## 主要な貢献まとめ

1. **DINO 検出器（[[entities/dino-detector]]）に GLIP の grounded pre-training を移植**: open-vocab 検出と DETR 系統の最強検出器の正統な統合
2. **Tight 3-Phase Fusion**: 既存の partial fusion（A only / B only）を全フェーズ融合に拡張
3. **Language-Guided Query Selection**: テキスト誘導で decoder queries を初期化（DINO 検出器の Mixed QS の言語版）
4. **Cross-Modality Decoder**: DINO 検出器の decoder に text cross-attention を追加
5. **Sub-Sentence Level Text Representation**: word-level と sentence-level の長所を結合
6. **REC タスクへの open-set 評価拡張**: 属性付き物体指示を評価軸に追加
7. **COCO ゼロショット 52.5 AP / ODinW 26.1 AP**: 両方とも SOTA
8. **DINO 事前学習からの転送**: 計算コスト削減と LVIS 改善
9. **SAM 3 と MM-Grounding-DINO の直接の祖**: 後続研究への決定的影響

---

## 関連ページ

- [[sources/grounding-dino]] — 原典の要約
- [[translations/grounding-dino]] — 原典の和訳
- [[entities/glip]] / [[sources/glip]] — GLIP（grounded pre-training の祖）
- [[entities/dino-detector]] / [[sources/dino-detector]] — DINO 検出器（DETR 系統の集大成）
- [[entities/detr]] / [[sources/detr]] — DETR の祖
- [[concepts/object-detection]] — 物体検出全体（Grounding DINO は open-vocab DETR 系の決定版）
- [[concepts/zero-shot-transfer]] — Grounding DINO のゼロショット転送
- [[concepts/promptable-concept-segmentation]] — SAM 3 PCS の直接の前駆
- [[entities/sam-3]] / [[sources/sam-3]] — Grounding DINO + presence head + マスク + 動画
- [[entities/perception-encoder]] — DETA decoder で COCO 66.0 box AP、DINO 検出器系統の後継
- [[entities/dino]] — **SSL の DINO（完全に別物）**
- [[entities/yolo-world]] / [[sources/yolo-world]] — 同じ open-vocab 路線の対照軸（精度志向 vs 速度志向、CVPR 2024）
- [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — **直接の後継**（2024 May、Pro/Edge 双子スイート、ViT-L + Grounding-20M で SOTA 更新）
- [[entities/dino-x]] / [[sources/dino-x]] — IDEA 系譜の現時点最新版（2024 Nov、unified perception、4 ヘッド + 3 プロンプト、Grounding-100M）
