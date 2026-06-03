---
type: entity
entity_kind: model
aliases: [InternVL 2.5, InternVL2.5-1B, InternVL2.5-2B, InternVL2.5-4B, InternVL2.5-8B, InternVL2.5-26B, InternVL2.5-38B, InternVL2.5-78B, InternVL2_5, InternViT-6B-V2.5, InternViT-300M-V2.5]
tags: [mllm, vllm, internvl-2-5, mmmu-70-percent, opengvlab, internvl-series, test-time-scaling]
related: ["[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/vision-transformer]]", "[[concepts/zero-shot-transfer]]", "[[concepts/knowledge-distillation]]", "[[concepts/alignment-tuning]]", "[[entities/internvl]]", "[[entities/internvl-1-5]]", "[[entities/mini-internvl]]", "[[entities/internvit-300m]]", "[[entities/clip]]"]
sources: ["[[sources/internvl-2-5]]"]
updated: 2026-05-29
---

# InternVL 2.5 — MMMU 70% を超えた最初のオープンソース MLLM スイート（1B-78B）

## 概要

**InternVL 2.5** = OpenGVLab Shanghai AI Lab の **InternVL シリーズ第 6 世代**（2024 Dec）。**MMMU で 70.1% を達成し、商用 GPT-4o (69.1) / Claude-3.5-Sonnet (68.3) を凌ぎ、70% を超えた初のオープンソース MLLM** となった記念碑的モデル。

- 論文: "Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling"
- arXiv: 2412.05271（2024 Dec、technical report）
- HuggingFace: <https://huggingface.co/OpenGVLab/InternVL2_5-78B>
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- Demo: <https://huggingface.co/spaces/OpenGVLab/InternVL>
- 詳細: [[sources/internvl-2-5]] / 翻訳: [[translations/internvl-2-5]]

7 サイズの基本モデル + InternVL2.5-Pro（非公開）= **8 モデルのスイート**。

---

## アーキテクチャ（ViT-MLP-LLM、InternVL 1.5 / 2.0 と同じ）

```
画像入力（任意解像度）
   ↓ 動的高解像度（35 種アスペクト比 × 1-12 タイル、test 40 タイル / 4K）
448×448 タイル × N + 448×448 サムネイル
   ↓ InternViT-300M-V2.5 (300M) or InternViT-6B-V2.5 (5.5B、45 層)
1024 visual tokens / tile
   ↓ Pixel Unshuffle（1/4 圧縮）
256 visual tokens / tile
   ↓ 2 層 MLP プロジェクタ（GELU、ランダム初期化 → 訓練）
LLM 埋め込み空間
   ↓ InternLM 2.5 系 or Qwen 2.5 系 LLM
出力テキスト
```

**重要**: アーキテクチャは [[entities/internvl-1-5|InternVL 1.5]] / 2.0 と **完全に同じ**。「**アーキテクチャは変えず、訓練戦略・データ・テスト時スケーリングを精緻化する**」哲学。

---

## モデルファミリー（7 サイズ）

| モデル | 視覚エンコーダ | LLM | LLM provider | 合計 | OpenCompass | HF |
|---|---|---|---|---|---|---|
| **InternVL2.5-1B** | InternViT-300M-V2.5 | Qwen2.5-0.5B-Instruct | Alibaba | 0.9B | **54.5** | `OpenGVLab/InternVL2_5-1B` |
| **InternVL2.5-2B** | InternViT-300M-V2.5 | InternLM2.5-1.8B-Chat | Shanghai AI Lab | 2.2B | **59.8** | `OpenGVLab/InternVL2_5-2B` |
| **InternVL2.5-4B** | InternViT-300M-V2.5 | Qwen2.5-3B-Instruct | Alibaba | 3.7B | **65.1** | `OpenGVLab/InternVL2_5-4B` |
| **InternVL2.5-8B** | InternViT-300M-V2.5 | InternLM2.5-7B-Chat | Shanghai AI Lab | 8.1B | **68.1** | `OpenGVLab/InternVL2_5-8B` |
| **InternVL2.5-26B** | InternViT-6B-V2.5 | InternLM2.5-20B-Chat | Shanghai AI Lab | 25.5B | **71.3** | `OpenGVLab/InternVL2_5-26B` |
| **InternVL2.5-38B** | InternViT-6B-V2.5 | Qwen2.5-32B-Instruct | Alibaba | 38.4B | **73.3** | `OpenGVLab/InternVL2_5-38B` |
| **InternVL2.5-78B** | InternViT-6B-V2.5 | Qwen2.5-72B-Instruct | Alibaba | 78.4B | **75.5** | `OpenGVLab/InternVL2_5-78B` |
| InternVL2.5-Pro | InternViT-6B-V2.5 | – | – | – | – | 非公開 |

**特徴**:
- **InternViT-300M-V2.5**: 1B-8B の軽量モデル用（[[entities/mini-internvl|Mini-InternVL]] と同じ）
- **InternViT-6B-V2.5**: 26B+ の大規模モデル用（45 層、5.5B、[[entities/internvl-1-5|InternVL 1.5]] と同じ）
- **LLM ベンダー**: 偶数番号（2B, 8B, 26B）が InternLM 2.5、奇数番号（1B, 4B, 38B, 78B）が Qwen 2.5
- **`OpenCompass`**: 8 学術 VQA ベンチマークの平均、サイズに対して綺麗な単調増加

---

## 視覚エンコーダの系譜（InternViT 系統）

InternVL シリーズで継続的に進化してきた InternViT：

| Model | Train Res | Layers | Loss | #Param | 用途 |
|---|---|---|---|---|---|
| InternViT-6B-224px | fixed 224 | 48 | CLIP | 5.9B | InternVL 1.0 |
| InternViT-6B-448px-V1.0 | fixed 448 | 48 | NTP | 5.9B | (中間版) |
| InternViT-6B-448px-V1.2 | fixed 448 | **45**（-3） | NTP | 5.5B | InternVL 1.2 |
| InternViT-6B-448px-V1.5 | dynamic 448 | 45 | NTP | 5.5B | InternVL 1.5 / 2.0 |
| **InternViT-6B-448px-V2.5** | dynamic 448 | 45 | NTP | 5.5B | **InternVL 2.5** |
| InternViT-300M-448px-Distill | fixed 448 | 24 | Cosine 蒸留 | 0.3B | (中間版) |
| InternViT-300M-448px | dynamic 448 | 24 | NTP | 0.3B | InternVL 2.0 / Mini-InternVL |
| **InternViT-300M-448px-V2.5** | dynamic 448 | 24 | NTP | 0.3B | **InternVL 2.5** |

> **InternViT-300M-V2.5 の生成経路**: CLIP-ViT-L-336px → cosine 蒸留（InternViT-6B-V1.5 教師）→ NTP 損失で段階訓練 → V2.5 でさらに多様データで段階的事前学習

---

## 訓練パイプライン（3 段階）

| Stage | 訓練対象 | データ | 学習率 | 目的 |
|---|---|---|---|---|
| **1. MLP Warmup** | MLP のみ（ViT + LLM 凍結） | Pre-train mixture | 2e-4 | クロスモーダル整列 |
| **1.5. ViT Incremental Learning（任意）** | **ViT + MLP** | Pre-train mixture | 1e-5 | 視覚特徴抽出強化（多言語 OCR、数学チャート等） |
| **2. Full Model Instruction Tuning** | **全モデル** | Fine-tune mixture | 4e-5 (8B 以下) / 2e-5 (8B 超) | 指示追従 + 全体最適化 |

**訓練トークン量**（表 3）:
- InternVL2.5-1B: ~367B（Stage 1: 191B + Stage 2: 176B）
- InternVL2.5-8B: ~142B（Stage 1: 22B + Stage 1.5: 76B + Stage 2: 44B）
- InternVL2.5-26B: ~221B（Stage 1: 31B + Stage 1.5: 146B + Stage 2: 44B）
- InternVL2.5-78B: ~120B（Stage 1: 76B + Stage 2: 44B、**Stage 1.5 スキップ**）

**Qwen2-VL: 1.4T トークン vs InternVL2.5-78B: ~120B トークン = 1/12**。

---

## 段階的スケーリング戦略（Progressive Scaling Strategy）

本論文の独自貢献。「ViT + LLM の NTP 共同訓練が **汎用視覚特徴** を生む」という観察に基づく：

```
[Stage 1.5: 小型 LLM (20B) で ViT 訓練]
   InternViT-6B-V2.5 + InternLM2.5-20B-Chat
       ↓ NTP 損失で共同訓練、低学習率
   訓練済み InternViT-6B-V2.5

[Stage 2: 大型 LLM へ転送]
   訓練済み InternViT-6B-V2.5 → Qwen2.5-72B-Instruct（再訓練なし）
       ↓ Stage 1 + Stage 2 のみ（Stage 1.5 スキップ）
   InternVL2.5-78B
```

**効果**: 大型モデル訓練時に Stage 1.5 をスキップでき、訓練コストを大幅削減。

---

## 主要結果

### MMMU 70% の壁を突破

| Model | MMMU(val) | MMMU(test) | MMMU-Pro(overall) |
|---|---|---|---|
| GPT-4V | 63.1 | – | – |
| Gemini-1.5-Pro | 62.2 | – | 46.9 |
| Claude-3.5-Sonnet | 68.3 | – | 51.5 |
| **GPT-4o (2024-05)** | **69.1** | – | **51.9** |
| Qwen2-VL-72B | 64.5 | – | 46.2 |
| InternVL2-Llama3-76B | 62.7 | 55.1 | 40.0 |
| **InternVL2.5-78B** | **70.1** | **61.8** | **48.6** |
| InternVL2.5-38B | 63.9 | 57.6 | 46.0 |
| InternVL2.5-26B | 60.0 | 51.8 | 37.1 |
| InternVL2.5-8B | 56.0 | 48.9 | 34.3 |

**MMMU 70.1% は GPT-4o (69.1) を超える、当時のオープン MLLM 最高記録**。

### 数学推論（GPT-4o を圧倒）

| Model | MathVista | MATH-Vision (mini/full) | MathVerse | OlympiadBench |
|---|---|---|---|---|
| GPT-4V | 58.1 | – / 24.0 | 32.8 | 18.0 |
| **GPT-4o** | **63.8** | – / 30.4 | 50.2 | **25.9** |
| Claude-3.5-Sonnet | 67.7 | – | – | – |
| Qwen2-VL-72B | 70.5 | – / 25.9 | – | – |
| **InternVL2.5-78B** | **72.3** | **34.9 / 32.2** | **51.7** | 11.6 |
| **InternVL2.5-38B** | **71.9** | **32.2 / 31.8** | 49.4 | **12.1** |

**MathVista で GPT-4o を +8.5 圧倒**。MATH-Vision でも GPT-4o (30.4) 超え 32.2。

### OCR / 文書（Claude-3.5-Sonnet と接戦）

| Model | DocVQA | ChartQA | OCRBench | CharXiv (RQ/DQ) |
|---|---|---|---|---|
| GPT-4o | 92.8 | 85.7 | 736 | 47.1 / 84.5 |
| Claude-3.5-Sonnet | 95.2 | **90.8** | 788 | **60.2 / 84.3** |
| **Qwen2-VL-72B** | **96.5** | 88.3 | **877** | – |
| InternVL2.5-78B | 95.1 | 88.3 | 854 | 42.4 / 82.3 |
| InternVL2.5-38B | 95.3 | 88.2 | 842 | 42.4 / 79.6 |

VCR-EN-Easy: InternVL2-2B 32.9 → **InternVL2.5-2B 93.2**（+60.3 ポイント、22K サンプル追加効果）。

### 視覚的グラウンディング（RefCOCO SOTA）

| Model | RefCOCO avg | RefCOCO+ avg | RefCOCOg avg | overall |
|---|---|---|---|---|
| Grounding-DINO-L | 90.7 | 82.6 | 86.6 | 86.6 |
| UNINEXT-H | 92.8 | 84.9 | 89.1 | 88.9 |
| CogVLM-Grounding-17B | 92.2 | 88.3 | 90.3 | 90.3 |
| Qwen2-VL-72B | 93.1 | 89.8 | 90.2 | 91.1 |
| **InternVL2.5-78B** | **93.9** | **90.7** | **92.5** | **92.3** |

**RefCOCO 8 サブセット平均 92.3 で SOTA**、grounding 特化モデルも超える。

### 動画理解（複数で SOTA）

| Model | Video-MME(wo/w sub) | MVBench | MMBench-Video | MLVU |
|---|---|---|---|---|
| Gemini-1.5-Pro | **75.0 / 81.3** | – | 1.30 | – |
| GPT-4o | 71.9 / 77.2 | – | 1.63 | 64.6 |
| Qwen2-VL-72B | 71.2 / 77.8 | 73.6 | 1.70 | – |
| **InternVL2.5-78B** | 72.1 / 74.0 | **76.4** | **1.97** | **75.7** |

**入力フレーム数増加に強いスケーラビリティ**（4-24 → 8-32 訓練の効果）。

### 純粋言語能力の改善

| Avg over 17 LLM 用ベンチ | InternVL 2.0 | InternVL 2.5 | vs 基盤 LLM |
|---|---|---|---|
| 2B | 39.2 | **48.4** | **+0.8**（vs InternLM2.5-1.8B-Chat） |
| 8B | 67.2 | **70.0** | **+0.5**（vs InternLM2.5-7B-Chat） |
| 26B | 64.2 | **72.9** | **+1.4**（vs InternLM2.5-20B-Chat） |

**「MLLM 訓練が純粋言語能力を低下させる問題」を高品質言語データ + フィルタリングで解決**。

---

## アーキテクチャ比較

### vs InternVL 1.5

| 項目 | InternVL 1.5 | InternVL 2.5 |
|---|---|---|
| アーキテクチャ | ViT-MLP-LLM | **完全に同じ** |
| 視覚 | InternViT-6B-V1.5 (45 層) | **InternViT-6B-V2.5 / 300M-V2.5** |
| LLM | InternLM2-20B-Chat | **InternLM 2.5 / Qwen 2.5** |
| サイズ | 26B（単一） | **1B/2B/4B/8B/26B/38B/78B（7 サイズ）** |
| 訓練段階 | 2 段階 | **3 段階（Stage 1.5 = ViT incremental）** |
| データ量（FT） | 5.1M | **16.3M** (3.2×) |
| MMMU | 45.2 | **70.1** (+24.9) |
| OCRBench | 724 | **854** (+130) |
| 訓練トークン | 不明 | ~120B（78B モデル） |

### vs InternVL 2.0（同シリーズ）

| 項目 | InternVL 2.0 | InternVL 2.5 |
|---|---|---|
| 視覚 | InternViT-6B-V1.5 / 300M | **InternViT-6B-V2.5 / 300M-V2.5** |
| 訓練データ | 7.3M | **16.3M** (2.2×) |
| データフィルタリング | 限定的 | **3 戦略 + 異常除去** |
| 純粋言語 | -2.3（vs LLM）| **+0.5**（vs LLM） |
| MMMU (76B) | 62.7 | **70.1** (+7.4) |
| CoT 効果 | 低（しばしばループ） | **高（+3.7 改善）** |

### vs GPT-4o / Claude-3.5-Sonnet（商用フロンティア）

| 項目 | GPT-4o | Claude-3.5-Sonnet | InternVL2.5-78B |
|---|---|---|---|
| パラメータ | 非公開（推定 ~200B） | 非公開 | **78B** |
| 公開 | クローズド | クローズド | **オープン MIT/InternLM/Qwen** |
| MMMU | 69.1 | 68.3 | **70.1** |
| MathVista | 63.8 | 67.7 | **72.3** |
| MMBench-EN | 83.4 | 82.6 | **88.3** |
| MMStar | 64.7 | 65.1 | **69.5** |
| DocVQA | 92.8 | **95.2** | 95.1 |
| ChartQA | 85.7 | **90.8** | 88.3 |
| Video-MME (wo sub) | **71.9** | – | 72.1 |
| RefCOCO avg | – | – | **93.9** |
| WildVision | **80.6** | – | 71.4 (-9.2) |
| MMVet v2 | 71.0 | **71.8** | 65.5 (-5.5) |

**強み**: MMMU / 数学 / MMBench / MMStar / RefCOCO / 動画 で商用超え
**弱み**: WildVision（長応答品質）/ MMVet v2 / ChartQA（Claude-3.5）

---

## InternVL シリーズ全体での位置づけ

```
[2023-12] InternVL 1.0 (CVPR 2024, [[entities/internvl]])
   └─ 対比 + 生成 + 対話統合、3 段階訓練、6B + QLLaMA-8B

[2024-04] InternVL 1.5 ([[entities/internvl-1-5]])
   └─ QLLaMA 廃止 + MLP、動的 4K、26B、18 ベンチ中 8 SoTA

[2024-07] InternVL 2.0
   └─ 1B/2B/4B/8B/26B/40B/76B、複数画像 + 動画対応

[2024-10] Mini-InternVL ([[entities/mini-internvl]])
   └─ InternViT-300M 蒸留、軽量 + ドメイン適応

[2024-12] InternVL 2.5 ←（本ページ）
   ├─ 1B/2B/4B/8B/26B/38B/78B、視覚 + LLM 両方最新化
   ├─ Progressive Scaling Strategy 公式化
   ├─ Test-Time Scaling（CoT + Majority Voting）
   └─ **MMMU 70.1% で 70% を超えた初の OS MLLM**

[2025-04] InternVL 3 ([[entities/internvl-3]])
   ├─ **Native Multimodal Pre-Training**（Qwen2.5 base + テキスト + マルチモーダル共同事前学習）
   ├─ V2PE + MPO + VisualPRM、訓練データ完全公開
   └─ **MMMU 72.2% で再び SOTA 更新、Qwen2.5-Chat より言語が強い**
```

**InternVL 2.5 は「商用追従ライン」の最終到達点**。次の [[entities/internvl-3|InternVL 3]] で **「事後的 MLLM 適応 → Native Multimodal Pre-Training」** という哲学転換を実行。

---

## 公開リソース

公開済み（HuggingFace `OpenGVLab/` 配下）:

| モデル | サイズ | ライセンス |
|---|---|---|
| `InternVL2_5-1B/2B/4B/8B/26B/38B/78B` | 各サイズ | MIT + InternLM2 / Qwen2.5 |
| `InternViT-6B-448px-V2_5` | 5.5B | MIT |
| `InternViT-300M-448px-V2_5` | 300M | MIT |
| `InternVL2_5-MPO` | 後の DPO 版 | （別論文） |

InternVL2.5-Pro はクローズド。

---

## 関連ページ

- 詳細解説: [[sources/internvl-2-5]]
- 翻訳: [[translations/internvl-2-5]]
- 直接の祖: [[entities/internvl-1-5]] / [[sources/internvl-1-5]]（同アーキテクチャ）
- 軽量分岐: [[entities/mini-internvl]] / [[sources/mini-internvl]] / [[entities/internvit-300m]]
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]]
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/vision-transformer]], [[concepts/zero-shot-transfer]], [[concepts/knowledge-distillation]], [[concepts/alignment-tuning]]（PE の中間層特徴発見との対比）
