---
type: entity
entity_kind: model
aliases: [InternViT-300M, InternViT-300M-448px]
tags: [vision-foundation-model, vfm, knowledge-distillation, lightweight, mllm, opengvlab]
related: [[concepts/knowledge-distillation]], [[concepts/vision-transformer]], [[concepts/foundation-model]], [[entities/internvl]], [[entities/internvl-1-5]], [[entities/mini-internvl]], [[entities/clip]]
sources: [[sources/mini-internvl]]
updated: 2026-05-29
---

# InternViT-300M — InternViT-6B の知識を蒸留した 300M VFM

## 概要

**InternViT-300M** = [[entities/mini-internvl|Mini-InternVL]] 系列の視覚エンコーダ。**[[entities/internvl|InternVL 1.0]] の InternViT-6B（5.9B）を 300M に蒸留した軽量 VFM**。CLIP-ViT-L (300M) と同じパラメータ数でありながら、**多ドメインの視覚知識を継承** している。

- 論文: [[sources/mini-internvl]]（arXiv:2410.16261, 2024 Oct）
- HuggingFace: <https://huggingface.co/OpenGVLab/InternViT-300M-448px>
- ライセンス: MIT

---

## 設計の核心: 「CLIP 初期化 + InternViT-6B 蒸留」

```
[初期化フェーズ]
   CLIP-ViT-L-336px (OpenAI, 304M)
       ↓ 重みをコピー
   InternViT-300M (Initial)

[蒸留フェーズ]
   入力画像（448×448 固定）
       ↓
   ┌─ InternViT-6B (5.9B, 教師)
   │   ↓ 最後の K 層の隠れ状態 h_t
   │
   └─ InternViT-300M (生徒)
       ↓ 最後の K 層の隠れ状態 h_s
       ↓
   Loss = -cos(h_t, h_s)  ← 負のコサイン類似度
       ↓ 逆伝播
   InternViT-300M を更新
```

**KD 損失の特徴**:
- **最後の K 層のみ** で蒸留（全層整列ではない）
- **negative cosine similarity**（feature-based distillation）
- 教師 InternViT-6B は凍結

### なぜ「CLIP 初期化 + 蒸留」なのか

| 選択肢 | 結果 |
|---|---|
| ランダム初期化 + 蒸留 | 訓練コスト膨大、収束遅い |
| CLIP-ViT-L のまま使う | 自然画像のみで、OCR/medical/RS に弱い |
| **CLIP 初期化 + InternViT-6B 蒸留**（本論文） | **CLIP の良い初期表現 + InternViT-6B の多ドメイン知識** |

[[concepts/knowledge-distillation|知識蒸留]] の典型的応用。

---

## アーキテクチャ

| 項目 | 値 |
|---|---|
| ベース | vanilla [[concepts/vision-transformer\|ViT]] |
| パラメータ | **300M** |
| 解像度 | 448×448（蒸留時固定、推論時 dynamic） |
| Patch size | 14（CLIP-L-336 と同じ） |
| 初期化 | CLIP-ViT-L-336px（304M、OpenAI MIT） |
| 蒸留教師 | InternViT-6B（[[entities/internvl]] 由来） |
| 蒸留損失 | negative cosine similarity（最後の K 層） |
| パッチ表現 | 1024 token / 448 tile（Pixel Unshuffle 前） |

---

## 訓練データ

[[sources/mini-internvl]] 表 1 の **5 カテゴリの多様な画像** で蒸留：

| カテゴリ | 主要データセット |
|---|---|
| **Natural images** | LAION, COYO, GRIT, COCO, LVIS, Objects365, Flickr30K, VG, All-Seeing, MMInstruct, LRV-Instruction |
| **OCR** | TextCaps, Wukong-OCR, CTW, MMC-Inst, LSVT, ST-VQA, RCTW-17, ReCTs, ArT, SynthDoG, LaionCOCO-OCR, COCO-Text, DocVQA, TextOCR, LLaVAR, TQA, SynthText, DocReason25K, Common Crawl PDF |
| **Chart** | AI2D, PlotQA, InfoVQA, ChartQA, MapQA, FigureQA, IconQA, MMC-Instruction |
| **Multidisciplinary** | CLEVR-Math/Super, GeoQA+, UniChart, ScienceQA, Inter-GPS, UniGeo, PMC-VQA, TabMWP, MetaMathQA |
| **Other** | Stanford40, GQA, MovieNet, KonIQ-10K, ART500K, ViQuAE |

**意義**:
- CLIP は **自然画像 + 短いキャプション** しか見ていない
- InternViT-300M は **OCR / chart / multidisciplinary / 医療（PMC-VQA）** までカバー
- これが Mini-InternVL のドメイン適応能力の基礎

---

## 性能比較

### vs CLIP-ViT-L-336px（同サイズ 300M）

Mini-InternVL-2B 構成での比較（[[sources/mini-internvl]] 表 12）:

| ベンチ | CLIP-L | **InternViT-300M** | 差 |
|---|---|---|---|
| MMB-EN | 70.3 | **73.2** | +2.9 |
| MMB-CN | 68.1 | **70.9** | +2.8 |
| ChartQA | 70.9 | **76.2** | +5.3 |
| **DocVQA** | 77.5 | **85.9** | **+8.4** |
| InfoVQA | 49.6 | **57.7** | +8.1 |
| MMMU | 32.9 | **34.3** | +1.4 |
| MME-RW (AD) | 43.7 | **48.0** | +4.3 |
| DriveLM | 0.580 | 0.578 | -0.002 |

**OCR / Document / Chart で蒸留の優位性が圧倒的**。

### vs InternViT-6B（教師）

| 項目 | InternViT-6B | InternViT-300M | 比率 |
|---|---|---|---|
| パラメータ | 5.9B | 300M | **1/20** |
| 推論速度 | 1× | ~20× 高速 | 大幅向上 |
| Avg MLLM スコア（Mini-InternVL-4B vs InternVL2-76B） | 81.4 | 72.8 | **90%** |

---

## 用途

### 1. Mini-InternVL のバックボーン（主用途）

[[entities/mini-internvl|Mini-InternVL-1B/2B/4B]] の視覚エンコーダとして使用。

### 2. 単体での視覚タスク（潜在用途）

CLIP-L の代替として：
- 線形プローブ（OCR / chart で CLIP より強い可能性）
- 検出 / セグメンテーションの backbone
- 医療画像・衛星画像の特徴抽出

**ただし論文は MLLM 用途以外の評価を行っていない**ため、これらは未検証。

---

## 系譜上の位置

```
[CLIP-ViT-L-336px (2021, OpenAI)]
       ↓ 重みコピー（初期化）
[[entities/internvl|InternViT-6B (2023, OpenGVLab)]]
       ↓ 知識蒸留（negative cosine similarity, 最後 K 層）
[InternViT-300M (2024, OpenGVLab) ← 本ページ]
       ↓ Mini-InternVL の VFM として使用
[Mini-InternVL-1B/2B/4B (2024, OpenGVLab)]
       ↓ さらに多様データで段階的事前学習（V2.5）
[InternViT-300M-V2.5 (2024-12)]
       ↓ InternVL 2.5 の 1B/2B/4B/8B モデルの VFM として使用
[[entities/internvl-2-5|InternVL 2.5 (2024-12)]]
```

### V2.5 への進化（InternVL 2.5 と同期）

| バージョン | 用途 |
|---|---|
| InternViT-300M-448px-Distill | CLIP-L 初期化 + InternViT-6B-V1.5 から cosine 蒸留 |
| InternViT-300M-448px | + dynamic 448 + NTP 損失で訓練（[[entities/mini-internvl\|Mini-InternVL]] と InternVL 2.0 が使用） |
| **InternViT-300M-448px-V2.5** | **多様データで段階的事前学習を継続**（[[entities/internvl-2-5\|InternVL 2.5]] の 1B-8B + [[entities/internvl-3\|InternVL 3]] の 1B-14B で使用） |

> **InternVL 3 でも V2.5 を継承**: [[entities/internvl-3|InternVL 3]]（2025-04）は **InternVL 2.5 と同じ InternViT-V2.5 を再利用** し、**LLM 側のみ Qwen2.5 base に切り替え + Native Multimodal Pre-Training**。視覚エンコーダの更新なし、InternViT-V2.5 が **2 世代にわたって使われる成熟版** となった。

**「Vision Foundation Model の蒸留」** という方向性は、PE / DINOv2 の軽量化の参考になる。同時期の **CLIP 蒸留系列**:
- CLIPA-v2: 訓練効率改善
- MobileCLIP（Apple, 2024）: モバイル向け CLIP
- **InternViT-300M (本ページ)**: CLIP より広いドメイン知識を持つ蒸留 VFM

---

## 関連ページ

- 提案論文: [[sources/mini-internvl]] / [[translations/mini-internvl]]
- 主用途: [[entities/mini-internvl]]
- 教師モデル: [[entities/internvl]]（InternViT-6B が教師）
- 初期化元: [[entities/clip]]（CLIP-ViT-L-336px）
- 関連概念: [[concepts/knowledge-distillation]]、[[concepts/vision-transformer]]、[[concepts/foundation-model]]
- 関連 VFM: [[entities/perception-encoder]] / [[entities/siglip]] / [[entities/dinov2]] — CLIP 系の競合
