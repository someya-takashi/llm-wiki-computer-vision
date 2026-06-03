---
type: entity
entity_kind: model
aliases: [InternVL 3, InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B, InternVL3]
tags: [mllm, vllm, internvl-3, native-multimodal-pretraining, v2pe, mpo, mmmu-72-2, opengvlab, internvl-series]
related: ["[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/vision-transformer]]", "[[concepts/zero-shot-transfer]]", "[[concepts/alignment-tuning]]", "[[entities/internvl]]", "[[entities/internvl-1-5]]", "[[entities/mini-internvl]]", "[[entities/internvl-2-5]]", "[[entities/internvit-300m]]", "[[entities/clip]]"]
sources: ["[[sources/internvl-3]]"]
updated: 2026-05-29
---

# InternVL 3 — Native Multimodal Pre-Training を確立、MMMU 72.2 で SOTA

## 概要

**InternVL 3** = OpenGVLab Shanghai AI Lab の **InternVL シリーズ第 7 世代**（2025 Apr）。**「LLM Chat 版を base に事後的にマルチモーダル化する」** という InternVL 1.0-2.5 の伝統を捨て、**Native Multimodal Pre-Training**（テキスト + マルチモーダル共同事前学習）へ哲学転換した記念碑的モデル。

- 論文: "InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models"
- arXiv: 2504.10479（2025 Apr、technical report）
- HuggingFace: <https://huggingface.co/OpenGVLab/InternVL3-78B>
- データ: <https://huggingface.co/datasets/OpenGVLab/InternVL-Data>（**訓練データ完全公開**）
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- 詳細: [[sources/internvl-3]] / 翻訳: [[translations/internvl-3]]

7 サイズの基本モデル。**訓練データ + モデル重み完全公開**、open-science 原則を強調。

---

## アーキテクチャ（ViT-MLP-LLM、InternVL 2.5 と同じ）

```
画像入力（任意解像度）
   ↓ 動的高解像度（35 種アスペクト比 × 1-12 タイル、test 40 タイル / 4K）
448×448 タイル × N + 448×448 サムネイル
   ↓ InternViT-300M-V2.5 (300M) or InternViT-6B-V2.5 (5.5B、45 層)
1024 visual tokens / tile
   ↓ Pixel Unshuffle（1/4 圧縮）
256 visual tokens / tile + V2PE 位置エンコーディング
   ↓ 2 層 MLP プロジェクタ
LLM 埋め込み空間
   ↓ Qwen2.5 base or InternLM3-8B
出力テキスト
```

**重要変更（InternVL 2.5 から）**:
1. **LLM は base モデルから初期化**（InternVL 2.5 は Chat/Instruct 版）
2. **V2PE**（Variable Visual Position Encoding）追加
3. **訓練パラダイム = Native Multimodal Pre-Training**（共同事前学習）
4. **MPO**（Mixed Preference Optimization）追加

---

## モデルファミリー（7 サイズ）

| モデル | 視覚 | LLM Base | LLM Provider | 合計 | OpenCompass |
|---|---|---|---|---|---|
| **InternVL3-1B** | InternViT-300M-V2.5 | Qwen2.5-0.5B (base) | Alibaba | 0.9B | **57.4** |
| **InternVL3-2B** | InternViT-300M-V2.5 | Qwen2.5-1.5B (base) | Alibaba | 1.9B | **63.9** |
| **InternVL3-8B** | InternViT-300M-V2.5 | Qwen2.5-7B (base) | Alibaba | 8.1B | **73.3** |
| **InternVL3-9B** | InternViT-300M-V2.5 | InternLM3-8B (instruct) | Shanghai AI Lab | 9.2B | **72.4** |
| **InternVL3-14B** | InternViT-300M-V2.5 | Qwen2.5-14B (base) | Alibaba | 15.1B | **75.5** |
| **InternVL3-38B** | InternViT-6B-V2.5 | Qwen2.5-32B (base) | Alibaba | 38.4B | **77.3** |
| **InternVL3-78B** | InternViT-6B-V2.5 | Qwen2.5-72B (base) | Alibaba | 78.4B | **79.5** |

**特徴**:
- **InternVL 2.5 と同じ InternViT-V2.5** を再利用（視覚エンコーダは継続事前学習で強化された V2.5 を継承）
- **LLM はほぼ Qwen2.5 ベース**（InternLM3-8B 1 例のみ）
- **9B 以外はすべて base モデル初期化**（9B は InternLM3-8B-Instruct）

---

## 訓練パイプライン（3 段階、InternVL 2.5 から再設計）

### Stage 1: Native Multimodal Pre-Training（最大の変更点）

| 項目 | 内容 |
|---|---|
| 訓練対象 | **ViT + MLP + LLM 全層共同訓練** |
| データ | 言語 ~50B + マルチモーダル ~150B = **総 ~200B トークン** |
| 比率 | **言語 : マルチモーダル = 1 : 3** |
| 言語データ | InternLM2.5 pretraining data + オープンソース |
| マルチモーダル | [[entities/internvl-2-5\|InternVL 2.5]] 由来 + 新規（**GUI / Tool / 3D / Video**） |
| 損失 | text-only loss（視覚トークンは予測しない）+ square averaging (w = 1/l^0.5) |

### Stage 2: Supervised Fine-Tuning (SFT)

| 項目 | 内容 |
|---|---|
| 訓練対象 | 全モデル |
| データ | **21.7M サンプル**（[[entities/internvl-2-5\|InternVL 2.5]] の 16.3M から +5.4M） |
| 新規データ | ツール使用、3D シーン理解、GUI 操作、長文脈、動画、科学図解、創造的書きもの、推論 |
| 技術 | Random JPEG Compression + Square Averaging + Multimodal Data Packing（[[entities/internvl-2-5\|2.5]] から継承） |

### Stage 3: Mixed Preference Optimization (MPO)

| 項目 | 内容 |
|---|---|
| 訓練対象 | 全モデル（選好調整） |
| データ | **[[entities/mmpr\|MMPR v1.2]]、約 300K 選好ペア** |
| 損失 | $\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$ |
| $\mathcal{L}_p$ | DPO loss（chosen/rejected 相対選好） |
| $\mathcal{L}_q$ | BCO loss（個別応答の絶対品質） |
| $\mathcal{L}_g$ | LM loss（選好応答の生成プロセス） |

**重要**: MPO データは SFT データのサブセット → **性能改善はアルゴリズムによる**（データではない）。

> **MPO の起源**: MPO アルゴリズムと MMPR データセットは **[[sources/mpo|MPO 論文（Wang et al., 2024 Nov）]] で初提案**、InternVL2-8B-MPO（[[entities/mpo]] 参照）が MathVista で +8.7 ポイント（10× 大きい InternVL2-76B と同等）を達成。InternVL 3 で **正式採用された永続的技術**。アルゴリズム詳細: [[entities/mpo]]、データセット詳細: [[entities/mmpr]]。

---

## V2PE（Variable Visual Position Encoding）

**長いマルチモーダル文脈対応の位置エンコーディング**。

```
従来:
   p_i = p_{i-1} + 1（テキスト・視覚同じ）

V2PE:
   p_i = p_{i-1} + 1   （テキストトークン）
   p_i = p_{i-1} + δ   （視覚トークン、δ < 1）
```

**δ の選択肢**: $\{1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256\}$、訓練中は各画像でランダム選択。

**画像内では δ 一定**（相対位置保持）。$\delta = 1$ で [[entities/internvl-2-5|InternVL 2.5]] の従来エンコーディングに帰着。

**アブレーション**（表 12、InternVL3-8B 事前学習版）:

| V2PE | δ | Overall |
|---|---|---|
| ✗ | – | 75.2 |
| ✓ | 1/256 | 75.0 |
| ✓ | 1/16 | 75.6 |
| ✓ | **1/4** | **75.9** |
| ✓ | 1/1 | 75.7 |

**$\delta = 1/4$ で最良**（短文脈タスクでも）。

---

## Test-Time Scaling: VisualPRM-8B + Best-of-N

**Visual Process Reward Model**（VisualPRM-8B、InternVL3-8B ベース）が各推論ステップに +/- スコアを付与、平均してソリューション全体スコア。

**Best-of-N (N=8) 戦略の効果**（特に小型モデル）:

| Model | w/o Bo8 | w/ Bo8 | Δ |
|---|---|---|---|
| InternVL3-1B | 25.1 | **35.0** | **+9.9** |
| InternVL3-2B | 32.4 | **41.7** | **+9.3** |
| InternVL3-8B | 44.3 | **50.2** | **+5.9** |
| InternVL3-78B | 54.6 | **56.5** | **+1.9** |

**「計算リソースをモデルサイズで使うか、推論回数で使うか」** という新しいトレードオフを提示。

---

## 主要結果（vs InternVL 2.5 + 商用フロンティア）

### MMMU で SOTA 更新（72.2）

| Model | MMMU(val) |
|---|---|
| InternVL2.5-78B | 70.0 |
| Qwen2.5-VL-72B | 68.2 |
| GPT-4o-20241120 | 70.7 |
| Gemini-2.0-Pro | 69.9 |
| **InternVL3-78B** | **72.2** (+2.2 from 2.5) |
| Claude-3.7-Sonnet | 75.0 |

**Claude-3.7-Sonnet には届かないが、GPT-4o / Gemini-2.0-Pro / Qwen2.5-VL-72B 超え**。

### 数学推論（MathVista で SOTA 79.0）

| Model | MathVista | MathVision | MathVerse |
|---|---|---|---|
| GPT-4o | 60.0 | 31.2 | 40.6 |
| Claude-3.7-Sonnet | 66.8 | 41.9 | 46.7 |
| Gemini-2.0-Pro | 71.3 | 48.1 | **67.3** |
| Qwen2.5-VL-72B | 74.2 | 39.3 | 47.3 |
| InternVL2.5-78B | 72.3 | 32.2 | 39.2 |
| **InternVL3-78B** | **79.0** | 43.1 | 51.0 |
| **w/ VisualPRM-Bo8** | **80.5** | 40.8 | 54.2 |

### OCR / 文書（OCRBench 906 で史上初の 900 超え）

| Model | OCRBench | DocVQA | ChartQA |
|---|---|---|---|
| Qwen2.5-VL-72B | 885 | 96.4 | 89.5 |
| InternVL2.5-78B | 854 | 95.1 | 88.3 |
| **InternVL3-78B** | **906** | 95.4 | 89.7 |

**OCRBench 906 で史上初の 900 超え**。

### 言語能力（同 Base から派生した Qwen2.5-Chat を超える）

| Base LLM | Qwen2.5-Chat | **InternVL3** | Δ |
|---|---|---|---|
| 0.5B → 1B | 33.5 | **42.4** | **+8.9** |
| 1.5B → 2B | 51.6 | **59.2** | **+7.6** |
| 7B → 8B | 69.4 | **72.9** | **+3.5** |
| 14B → 14B | 73.4 | **76.6** | **+3.2** |
| 72B → 78B | 78.9 | **80.5** | **+1.6** |

**「マルチモーダル化で言語能力が強くなる」** という InternVL シリーズ初の本格的実証。**小型ほど顕著**。

### 空間推論で GPT-4o を圧倒（VSI-Bench）

| Model | VSI Overall | Obj.Count | Abs.Dist | Rel.Dist |
|---|---|---|---|---|
| GPT-4o | 34.0 | 46.2 | 5.3 | 37.0 |
| Gemini-1.5 Pro | 45.4 | 56.2 | 30.9 | 51.3 |
| **InternVL3-8B** | **42.1** | 68.1 | 39.0 | 48.3 |
| **InternVL3-38B** | **48.9** | **71.7** | **50.2** | **53.5** |
| **InternVL3-78B** | **48.4** | 71.2 | **53.7** | **55.9** |

**InternVL3-8B (8B) が GPT-4o を +8.1 圧倒**、3D シーン理解の新水準。

### GUI Grounding（ScreenSpot で UI-TARS 同等）

| Model | ScreenSpot | ScreenSpot-V2 |
|---|---|---|
| GPT-4o | 18.1 | – |
| Aguvis-72B | **89.2** | – |
| UI-TARS-72B | 88.4 | 90.3 |
| Qwen2.5-VL-72B | 87.1 | – |
| **InternVL3-72B** | 88.7 | **90.9** |

### 動画理解（複数 SOTA）

| Model | MVBench | MLVU | CG-Bench(long/clue) |
|---|---|---|---|
| Qwen2.5-VL-72B | 70.4 | 74.6 | – |
| InternVL2.5-78B | 76.4 | 75.7 | 42.2 / 58.5 |
| **InternVL3-78B** | **78.7** | **79.5** | **48.4 / 65.3** |

### Visual Grounding で InternVL 2.5 にわずかに退行

| Model | RefCOCO/+/g Overall |
|---|---|
| InternVL2.5-78B | **92.3** |
| **InternVL3-78B** | 91.4 (-0.9) |

**「訓練データ拡張で grounding 特化データが含まれず、相対的に grounding データ比率が低下したため」**（著者分析）。シリーズ初の「先代に劣る」ケース。

---

## InternVL シリーズ全体での位置づけ

```
[2023-12] InternVL 1.0 (CVPR 2024, [[entities/internvl]])
   └─ InternViT-6B + QLLaMA-8B + Vicuna、対比 + 生成 + 対話統合（3 段階）

[2024-04] InternVL 1.5 ([[entities/internvl-1-5]])
   └─ QLLaMA 廃止 + MLP、動的 4K、26B、GPT-4V との差を埋める

[2024-07] InternVL 2.0
   └─ 1B-76B、複数画像 + 動画対応

[2024-10] Mini-InternVL ([[entities/mini-internvl]])
   └─ InternViT-300M 蒸留、軽量 + ドメイン適応

[2024-12] InternVL 2.5 ([[entities/internvl-2-5]])
   ├─ 1B-78B 7 サイズ、Progressive Scaling、Test-Time Scaling
   └─ MMMU 70.1% で 70% 突破の初 OS MLLM

[2025-04] InternVL 3 ←（本ページ）
   ├─ 1B-78B 7 サイズ、**LLM は base モデル初期化**
   ├─ **Native Multimodal Pre-Training**（哲学転換）
   ├─ V2PE + MPO + VisualPRM
   └─ **MMMU 72.2 で再び SOTA 更新**、Qwen2.5-Chat より強い言語能力

[2025-08] InternVL 3.5 ([[entities/internvl-3-5]])
   ├─ **Cascade RL（MPO + GSPO の 2 段階強化学習、本論文の MPO が Stage 1 で再採用）**
   ├─ **MoE 導入**（20B-A4B / 30B-A3B / 241B-A28B）
   ├─ ViR（視覚トークン 50% 削減）+ DvD（4.05× 推論加速）
   ├─ Qwen3 base + GPT-OSS、9 サイズ × dense+MoE × Flash
   └─ **MMMU 77.7、GPT-5 との差 3.9%**、VSI-Bench で GPT-5 を +32 圧倒
```

**「事後的 MLLM 適応」から「Native Multimodal Pre-Training」への哲学転換**。InternVL シリーズの根本的方針変更。

---

## アーキテクチャ比較

### vs InternVL 2.5

| 項目 | InternVL 2.5 | InternVL 3 |
|---|---|---|
| アーキテクチャ | ViT-MLP-LLM | **完全に同じ** |
| 視覚 | InternViT-300M-V2.5 / InternViT-6B-V2.5 | **完全に同じ** |
| **LLM 初期化** | **InternLM 2.5 Chat / Qwen 2.5 Instruct** | **Qwen2.5 base / InternLM3-8B** |
| **訓練パラダイム** | 3 段階（MLP warmup → ViT incremental → Full instruction tuning） | **Native Multimodal Pre-Training + SFT + MPO** |
| 位置エンコーディング | 標準（1 per token） | **V2PE（視覚 δ < 1）** |
| 訓練データ規模 | SFT 16.3M | SFT **21.7M** + MPO **300K** |
| サイズ | 1B/2B/4B/8B/26B/38B/78B（7 サイズ） | 1B/2B/8B/**9B**/14B/38B/78B（7 サイズ） |
| MMMU | 70.1 | **72.2** (+2.1) |
| OCRBench | 854 | **906** (+52) |
| 言語能力 | Chat 版より +0.5〜+1.4 改善 | **Chat 版より +1.6〜+8.9 改善** |

### vs Qwen2.5-VL-72B（直接の競合）

| 項目 | Qwen2.5-VL-72B | InternVL3-78B |
|---|---|---|
| 視覚 | 600M | **5.5B（InternViT-6B-V2.5）** |
| LLM | Qwen2.5-72B | Qwen2.5-72B base |
| MMMU | 68.2 | **72.2** (+4.0) |
| MathVista | 74.2 | **79.0** (+4.8) |
| DocVQA | **96.4** | 95.4 |
| OCRBench | 885 | **906** |
| InfoVQA | **87.3** | 86.5 |
| MVBench | 70.4 | **78.7** |
| MLVU | 74.6 | **79.5** |

**同じ Qwen2.5-72B base から派生しているが、InternVL3 が大半で優位**。

### vs 商用フロンティア（2025-04 時点）

| 項目 | GPT-4o | Claude-3.7-Sonnet | Gemini-2.0-Pro | InternVL3-78B |
|---|---|---|---|---|
| MMMU | 70.7 | **75.0** | 69.9 | **72.2** |
| MathVista | 60.0 | 66.8 | 71.3 | **79.0** |
| MathVision | 31.2 | 41.9 | **48.1** | 43.1 |
| MathVerse | 40.6 | 46.7 | **67.3** | 51.0 |
| LogicVista | 52.8 | 58.2 | 53.2 | **55.9** |
| Reasoning Overall | 47.9 | 53.9 | **58.5** | **54.6** |
| WildVision | **80.6** | – | – | 73.6 |
| OCRBench | 736 | 788 | – | **906** |

**強み**: MathVista / OCRBench / MMStar / MMVet / Video
**弱み**: MMMU (Claude-3.7 に -2.8) / MathVision (Gemini-2.0-Pro に -5.0) / WildVision (GPT-4o に -7.0)

---

## InternVL 3 の哲学的意義

### 「事後的（post-hoc）→ Native」パラダイム転換

```
[従来 (InternVL 1.0-2.5、LLaVA、Qwen2-VL)]
   テキスト専用 LLM 事前学習 → Chat 後訓練 → MLLM 適応
   ↑
   「真っ白なキャンバスではなく、既存の絵に追加描画する」

[InternVL 3 (Native Multimodal Pre-Training)]
   Base LLM + ViT + マルチモーダル + テキスト → 共同で事前学習
   ↑
   「真っ白なキャンバスから絵を描く」
```

### マルチモーダル化が言語能力を強化する

**Qwen2.5-Chat（言語特化）を InternVL3（マルチモーダル統合）が上回る** という結果は、「マルチモーダル化は言語能力にとってコスト」という従来の常識への重要な反論。

### Test-Time Compute の MLLM 適用本格化

OpenAI o1 (2024 Sep) → DeepSeek R1 (2025 Jan) → Claude-3.7-Sonnet (2025 Feb) の「reasoning + test-time scaling」流れを MLLM へ本格適用。**VisualPRM** という process reward model を MLLM 向けに導入したのは本論文が最初の本格事例。

---

## 公開リソース

公開済み（HuggingFace `OpenGVLab/` 配下）:

| モデル | サイズ | ライセンス |
|---|---|---|
| `InternVL3-1B/2B/8B/9B/14B/38B/78B` | 各サイズ | MIT + Qwen2.5 / InternLM3 |
| `VisualPRM-8B` | 8B | MIT |
| `InternVL-Data` | – | 公開（訓練データ完全公開） |

**ライセンス注意**: Qwen2.5 base 依存 → Tongyi Qianwen LICENSE の制約あり（特に大規模商用利用）。

---

## 関連ページ

- 詳細解説: [[sources/internvl-3]]
- 翻訳: [[translations/internvl-3]]
- 直接の祖: [[entities/internvl-2-5]] / [[sources/internvl-2-5]]（InternVL 2.5、同じ視覚エンコーダ V2.5、訓練パラダイムが異なる）
- 軽量分岐の起源: [[entities/mini-internvl]] / [[entities/internvit-300m]]
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]]
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/vision-transformer]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]
