---
type: entity
entity_kind: model
aliases: [GLIP, Grounded Language-Image Pre-training, GLIP-T, GLIP-L]
tags: [object-detection, phrase-grounding, open-vocabulary, microsoft, vision-language, foundation-model]
related: [[concepts/object-detection]], [[concepts/zero-shot-transfer]], [[concepts/promptable-concept-segmentation]], [[concepts/foundation-model]]
sources: [[sources/glip]]
updated: 2026-05-28
---

# GLIP（Grounded Language-Image Pre-training）

## 概要

**GLIP** は UCLA + Microsoft + 共同チーム（Liunian Harold Li, Pengchuan Zhang ら）が CVPR 2022 で発表した、**物体検出と phrase grounding を統一する事前学習モデル**。box 分類のロジットを「region 視覚特徴と単語 token 言語特徴の内積」に置き換えるだけで、**任意の検出モデルを grounding モデルに変換** できる定式化を提案。**language-aware deep fusion** + **self-training で 24M の web 画像-テキストペアに自動 box 注釈** を組み合わせて、open-vocabulary 物体検出のパラダイムを確立。

- 論文: "Grounded Language-Image Pre-training"
- arXiv: 2112.03857 / CVPR 2022
- コード: <https://github.com/microsoft/GLIP>
- メーカー: Microsoft Research + UCLA + Univ. of Washington + Univ. of Wisconsin + IDEA
- ライセンス: 公開（リポジトリ参照）

---

## モデルバリアント

GLIP には 5 つの主要バリアントがあり、3 つの中核技術（unified formulation / deep fusion / 両データ事前学習）の効果を分離する：

| Model | Backbone | Deep Fusion | Detection データ | Grounding データ | Caption データ | 計データ量 |
|---|---|---|---|---|---|---|
| **GLIP-T (A)** | Swin-T | ✗ | Objects365 (0.66M) | - | - | 0.66M |
| **GLIP-T (B)** | Swin-T | ✓ | Objects365 (0.66M) | - | - | 0.66M |
| **GLIP-T (C)** | Swin-T | ✓ | Objects365 (0.66M) | GoldG (0.8M) | - | 1.46M |
| **GLIP-T** | Swin-T | ✓ | Objects365 (0.66M) | GoldG (0.8M) | Cap4M (4M) | 5.46M |
| **GLIP-L** | **Swin-L** | ✓ | **FourODs (2.66M)** | GoldG (0.8M) | **Cap24M (24M)** | **27M** |

- **GLIP-T (A)** = 純粋な「再定式化」のみ
- **GLIP-T (B)** = + deep fusion
- **GLIP-T (C)** = + gold grounding データ
- **GLIP-T** = + web 画像-テキスト疑似 grounding
- **GLIP-L** = フルスケール

### アーキテクチャ詳細

| コンポーネント | 内容 |
|---|---|
| 画像 backbone | Swin-Transformer（Tiny: GLIP-T、Large: GLIP-L） |
| 検出ヘッド | **DyHead（Dynamic Head, Dai et al., CVPR 2021）** |
| テキスト encoder | **BERT（base-uncased）** |
| Deep fusion | DyHead の各層と BERT の追加層を **X-MHA** で接続 |
| 損失 | Focal binary sigmoid loss + box regression loss |
| プロンプト形式 | `"person. bicycle. car. ... toothbrush"`（"Detect:" よりも良い） |
| 推論 | サブワード token 確率を平均化して phrase 確率を計算 |

### Cross-Modality Multi-Head Attention (X-MHA) の数式

```
O^{(q)} = O W^{(q,I)},   P^{(q)} = P W^{(q,L)}
Attn = O^{(q)} (P^{(q)})^T / sqrt(d)

P^{(v)} = P W^{(v,L)},   O_t2i = SoftMax(Attn) P^{(v)} W^{(out,I)}
O^{(v)} = O W^{(v,I)},   P_i2t = SoftMax(Attn^T) O^{(v)} W^{(out,L)}

O^{i+1} = DyHeadModule(O^i + O_t2i)
P^{i+1} = BERTLayer(P^i + P_i2t)
```

---

## 主要結果

### COCO ゼロショット → fine-tune（表 2）

| Model | Backbone | Pre-train | Zero-Shot | Fine-Tune (test-dev) |
|---|---|---|---|---|
| Faster RCNN-FPN（参考） | RN50 | - | - | 40.2（教師あり） |
| DyHead-T | Swin-T | O365 | 43.6 | 53.3 |
| GLIP-T (A) | Swin-T | O365 | 42.9 | 52.9 |
| GLIP-T (B) | Swin-T | O365 | **44.9 (+2.0)** | 53.8 |
| GLIP-T (C) | Swin-T | O365 + GoldG | **46.7 (+1.8)** | 55.1 |
| GLIP-T | Swin-T | O365 + GoldG + Cap4M | 46.3 | 54.9 |
| **GLIP-L** | **Swin-L** | **FourODs + GoldG + Cap24M** | **49.8** | **61.0** |
| GLIP-L (+ COCO) | Swin-L | FourODs + GoldG+ + COCO | - | **61.5（SOTA）** |

**ゼロショットで Faster RCNN 教師あり 42.0 を上回る + fine-tune で SOTA**。

### LVIS ゼロショット（表 3、長尾分布）

| Model | Backbone | MiniVal APr | APc | APf | AP |
|---|---|---|---|---|---|
| MaskRCNN（教師あり） | RN101 | 26.3 | 34.0 | 33.9 | 33.3 |
| GLIP-T (B) | Swin-T | 13.5 | 12.8 | 22.2 | 17.8 |
| GLIP-T (C) | Swin-T | **17.7 (+4.2)** | 19.5 | 31.0 | 24.9 |
| GLIP-T | Swin-T | **20.8 (+3.1)** | 21.4 | 31.0 | 26.0 |
| **GLIP-L** | Swin-L | **28.2** | **34.3** | **41.5** | **37.3** |

**GLIP-L が rare クラス APr で教師あり MaskRCNN を超える**。

### Flickr30K phrase grounding（表 4）

| Model | Pre-train | Test R@1 |
|---|---|---|
| MDETR-ENB5 | GoldG+ | 84.3 |
| GLIP-T | O365 + GoldG + Cap4M | 85.7 |
| **GLIP-L** | FourODs + GoldG + Cap24M | **87.1** |

MDETR を 2.8 ポイント上回り phrase grounding SOTA。

### Object Detection in the Wild (ODinW, 13 データセット)

| 比較 | 結果 |
|---|---|
| ゼロショット GLIP-T vs 5-shot DyHead-T | **GLIP-T > DyHead-T** |
| 1-shot GLIP-L vs fully-supervised DyHead-T | **同等** |
| GLIP-L で prompt tuning vs full-tuning | **ほぼ同じ性能** |

「**1 つのモデルで 13 タスクを prompt 変更のみで対応**」が実用化された。

### Manual Prompt Tuning（図 6）

stingray を直接検出できない → プロンプトに「flat and round」を追加 → **AP50 4.6 → 9.7**。**言語による domain knowledge の注入** が可能。

---

## 訓練データの内訳

GLIP は **多種多様なデータソースを組み合わせる** ことが特徴：

| データセット | 種別 | サイズ | 用途 |
|---|---|---|---|
| **Objects365 (O365)** | 検出（365 クラス） | 0.66M | 全 GLIP の検出 backbone |
| **OpenImages** | 検出 | ~2M | GLIP-L の FourODs に含む |
| **Visual Genome (VG)** | 検出（COCO 除外）| 0.1M | GLIP-L の FourODs に含む |
| **ImageNetBoxes** | 検出 | ~1M | GLIP-L の FourODs に含む |
| **GoldG** | grounding（人手注釈）| 0.8M | Flickr30K + VG Caption + GQA、MDETR キュレーション、COCO 除外 |
| **GoldG+** | grounding（COCO 含む）| 1.3M | GLIP-L の COCO fine-tuning 時に使用 |
| **Cap4M** | 画像-テキスト（GLIP 生成 box）| 4M | GLIP-T 用、teacher GLIP-T(C) で box 生成 |
| **Cap24M = CC12M + SBU** | 画像-テキスト（GLIP 生成 box）| 24M | GLIP-L 用、78.1M phrase-box 疑似注釈、58.4M ユニーク名詞句 |
| **FourODs** | 検出（4 統合）| 2.66M | GLIP-L 用、1,500+ カテゴリ |

> **データ規模の比較**: GLIP-L の 27M grounding データ vs CLIP の 400M 画像-テキストペア。GLIP は **桁違いに少ないデータで object-level 性能を実現**。これは grounding データの「**意味的豊かさ**」が画像レベル CLIP データの量を補う証拠。

---

## Open-Vocabulary 検出ファミリーの祖

GLIP は open-vocabulary 物体検出系統の中核的な研究：

```
2018      Bansal et al. — Glove embedding でゼロショット検出
2021 Mar  CLIP — 画像レベル open-vocab
2021      ViLD — CLIP を two-stage 検出器に蒸留
2021 Oct  MDETR — DETR にテキスト条件追加、grounding
2021 Dec  ★ GLIP（本論文）★ — 検出 = grounding、deep fusion、self-training scale
            │
2022 Apr  OWL-ViT (Google) — ViT + DETR、open-vocab 検出
2023 Mar  ★ Grounding DINO ([[sources/grounding-dino]] / [[entities/grounding-dino]]) ★ — IDEA × Microsoft 共同、GLIP × DINO 検出器、Swin-L で COCO ZS 52.5 AP / FT 63.0 AP / ODinW ZS 26.1 AP
2024      MM-Grounding-DINO (OpenMMLab) — マルチモーダル拡張
2025      ★ SAM 3 (Meta) ★ — PCS タスク（GLIP+SAM の融合）
```

**GLIP の革新**:
1. **DETR ベースではなく DyHead ベース**で grounding を実現（並列研究の MDETR が DETR ベース）
2. **deep fusion を導入**して語彙の拡張性と prompt 認識性を両立
3. **self-training scale up**（27M grounding data）で実用性能を達成

**[[entities/grounding-dino|Grounding DINO]]** は GLIP の「検出 = grounding」を **DETR + 改良 DINO 検出器**（[[entities/dino-detector]]）の枠組みに移植 + tight 3-phase fusion + sub-sentence text representation で発展。SAM 3 は GLIP のテキスト条件付き検出を **promptable segmentation** に拡張（[[concepts/promptable-concept-segmentation]]）。

---

## SAM 3 / PE との関係（重要）

[[sources/sam-3]] / [[entities/sam-3]] の **PCS タスク** は GLIP のパラダイムを segmentation に拡張：

| 軸 | **GLIP（2022）** | **SAM 3（2025）** |
|---|---|---|
| 入力 | 名詞句プロンプト | 名詞句 or 画像 exemplar |
| 画像 encoder | Swin + DyHead | **Perception Encoder**（[[entities/perception-encoder]]） |
| テキスト encoder | BERT | PE のテキスト塔 |
| 融合 | X-MHA（DyHead と BERT 間） | Fusion encoder（PE 画像 ↔ プロンプト） |
| 検出 | DyHead box（古典的）| **DETR-based**（[[entities/detr]] / [[entities/dino-detector]] の系譜） |
| マスク | なし | MaskFormer 適応 |
| 認識/位置分離 | なし | **Presence head**（SAM 3 の核心改良） |
| データ規模 | 27M | 207K concepts benchmark + SA-Co |

**SAM 3 の "presence head" は GLIP の弱点を解決**: GLIP は「文脈なしの絶対的な存在判断」が苦手（"vaccine が画像にあるか"の判断が単に "vaccine" を localize しようとして強引に位置を出す）。Presence head はこれを **全 region に対するグローバルな existence score** として分離。

---

## 限界と批判

- **DyHead 検出器に依存**: Grounding DINO 等の後継は DETR ベースに移行
- **deep fusion の計算コスト**: late-fusion CLIP より重い
- **疑似ラベル品質依存**: teacher の grounding 間違いが student に伝播
- **言語 token 数制限**: BERT の 256 token 制限により Objects365 365 クラスは複数プロンプトに分割
- **テキストプロンプト必須**: 推論時に必ずテキスト入力が必要、image-only 使用不可
- **prompt design への敏感性**: "Detect:" を入れるか、"." vs ", " 等で性能変動

---

## YOLO-World との関係（実は深い）

[[entities/yolo-world]]（CVPR 2024）は、**GLIP-L を疑似ラベル生成 teacher として活用**:

```
GLIP-T → CC4M に擬似ラベル → GLIP-L 訓練（GLIP 自体の self-training）
                                ↓
              ★ YOLO-World が GLIP-L をさらに teacher として使用 ★
                                ↓
              CC3M (246K subset) に擬似ラベル → YOLO-World 訓練（246K → 821K 注釈）
```

これは **GLIP の self-training パイプラインの直接の継承** で、「**高精度 GLIP の知識を YOLOv8 に蒸留**」する間接的な蒸留パターンとして機能する。

## 主要な貢献まとめ

1. **物体検出と phrase grounding の統一定式化**: box 分類ロジットを region-word alignment スコアに置換
2. **検出と grounding の理論的等価性**: 同じ DyHead モデルが COCO で同じ AP（49.4）を達成
3. **Language-Aware Deep Fusion (X-MHA)**: 複数層での cross-modality 融合により言語認識視覚特徴を学習
4. **Self-training で grounding データを 27M にスケール**: NLP parser で名詞句抽出 + teacher GLIP で疑似 box 注釈
5. **COCO ゼロショット 49.8 AP / fine-tune 61.5 AP**: 教師ありを上回るゼロショット、SOTA fine-tune
6. **LVIS rare クラスで教師ありを上回る**: grounding データの語彙豊富性の実証
7. **ODinW ベンチマーク導入**: 13 多様タスクで 1-shot fully-supervised に匹敵
8. **Prompt tuning が物体検出で初めて実用化**: deep fusion モデルでのみ full-tuning に並ぶ
9. **Open-Vocabulary 検出パラダイムを確立**: Grounding DINO / SAM 3 / OWL-ViT の源流

---

## 関連ページ

- [[sources/glip]] — 原典の要約
- [[translations/glip]] — 原典の和訳
- [[concepts/object-detection]] — 物体検出全体（GLIP は open-vocab 転換点）
- [[concepts/zero-shot-transfer]] — GLIP のゼロショット転送（CLIP の object-level 拡張）
- [[concepts/promptable-concept-segmentation]] — SAM 3 PCS の祖
- [[concepts/foundation-model]]
- [[concepts/parameter-efficient-fine-tuning]] — prompt tuning は PEFT の一形態
- [[entities/clip]] / [[sources/clip]] — GLIP は CLIP の object-level 拡張
- [[entities/sam-3]] / [[sources/sam-3]] — GLIP パラダイムを segmentation に拡張
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — SAM 3 backbone（GLIP の精神的後継の基盤）
- [[entities/detr]] / [[sources/detr]] — DETR ベース後継（Grounding DINO）への流れ
- [[entities/dino-detector]] / [[sources/dino-detector]] — Grounding DINO は GLIP × DINO 検出器
- [[entities/yolo-world]] / [[sources/yolo-world]] — GLIP-L を疑似ラベル teacher として活用、real-time open-vocab 路線
