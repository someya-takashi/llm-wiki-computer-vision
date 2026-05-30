---
type: entity
entity_kind: model
aliases: [Mini-InternVL, Mini-InternVL-1B, Mini-InternVL-2B, Mini-InternVL-4B, Mini-InternVL-DA]
tags: [mllm, lightweight, edge-deployment, opengvlab, internvl-series, domain-adaptation, knowledge-distillation]
related: [[concepts/foundation-model]], [[concepts/knowledge-distillation]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]], [[entities/internvit-300m]], [[entities/internvl]], [[entities/internvl-1-5]], [[entities/clip]]
sources: [[sources/mini-internvl]]
updated: 2026-05-29
---

# Mini-InternVL — 5% パラメータ × 90% 性能のポケット MLLM 系列

## 概要

**Mini-InternVL** = OpenGVLab Shanghai AI Lab の **軽量 MLLM スイート**。InternVL シリーズが「**大規模 + 商用追従**」（[[entities/internvl-1-5|InternVL 1.5]] = 26B）と「**軽量 + ドメイン特化**」（本ページ = 1B/2B/4B）に二分岐したことを示す。

- 論文: "Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance"
- arXiv: 2410.16261（2024 年 10 月、technical report）
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- 詳細: [[sources/mini-internvl]] / 翻訳: [[translations/mini-internvl]]

3 種のサイズ + 各サイズのドメイン適応版 = **計 12 モデルのスイート**。

---

## アーキテクチャ（ViT-MLP-LLM、InternVL 1.5 と同じ）

```
画像入力（任意解像度）
   ↓ 動的高解像度（35 種アスペクト比 × 1-12 タイル、test 40 タイル / 4K）
448×448 タイル × N + 448×448 サムネイル
   ↓ InternViT-300M（CLIP-L 初期化 + InternViT-6B 蒸留、300M）
1024 visual tokens / tile
   ↓ Pixel Unshuffle（1/4 圧縮）
256 visual tokens / tile
   ↓ MLP プロジェクタ（ランダム初期化 → 訓練）
LLM 埋め込み空間
   ↓ 軽量 LLM（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）
出力テキスト
```

[[entities/internvl-1-5|InternVL 1.5]] と完全に同じ構造で、**視覚エンコーダと LLM を軽量化** したもの。

---

## モデルスイート

### 基本 3 サイズ

| モデル | LLM | LLM provider | 視覚 | 計 | HF |
|---|---|---|---|---|---|
| **Mini-InternVL-1B** | Qwen2-0.5B-Instruct | Alibaba | InternViT-300M | **1B** | `OpenGVLab/Mini-InternVL-Chat-1B-V1-5` |
| **Mini-InternVL-2B** | InternLM2-Chat-1.8B | Shanghai AI Lab | InternViT-300M | **2B** | `OpenGVLab/Mini-InternVL-Chat-2B-V1-5` |
| **Mini-InternVL-4B** | Phi-3-Mini-128K-Instruct | Microsoft | InternViT-300M | **4B** | `OpenGVLab/Mini-InternVL-Chat-4B-V1-5` |

**LLM ベンダーが異なる**点に注目: OpenGVLab は **「LLM は外部 + 視覚は内製で蒸留」** という分業を採用。

### Domain-Adapted（DA）版

論文では各サイズに対し、3 つのドメイン適応版を構築：

| DA バージョン | ドメイン | 訓練データ | 汎用:特化 比率 |
|---|---|---|---|
| **Mini-InternVL-DA-* (Driving)** | 自律走行 | DriveLM-nuScenes v1.1（317K）+ BDD-X（26K 動画） | **1:4** (DriveLM), **1:1** (BDD-X) |
| **Mini-InternVL-DA-* (Medical)** | 医療画像 | PMC-OA + MedICaT + PMC-Image + Open-i + MedPix + Quilt-1M + RP3D + MIMIC-CXR + Retina Image Bank（500K） | **1:1** |
| **Mini-InternVL-DA-* (RemoteSensing)** | リモートセンシング | GeoChat + RSVQA-HR + FIT-RS + DIOR-RSVG | **20% 汎用** |

各 DA 版は **8 A100 GPU × 1 epoch × 1e-5 学習率** で全パラメータ微調整。

---

## 視覚エンコーダ詳細: InternViT-300M

詳細は [[entities/internvit-300m]] を参照。要点：

| 項目 | 値 |
|---|---|
| 教師 | InternViT-6B（[[entities/internvl]] の視覚エンコーダ） |
| 初期化 | **CLIP-ViT-L-336px** の重み |
| パラメータ | 300M（教師の 1/20） |
| 蒸留損失 | **最後の K 層の隠れ状態間で negative cosine similarity** |
| 訓練データ | 多様な image data（自然 + OCR + chart + multidisciplinary） |
| 解像度 | 蒸留時 448 固定、推論時 dynamic 448（1-40 タイル） |
| Pixel Unshuffle | 256 token / tile（1024 → 1/4） |

**「CLIP の初期化 + InternViT-6B からの知識蒸留」** という 2 段階で、CLIP 単体より大幅に強い 300M VFM を構築。

---

## 訓練の 2 段階（基本版）

| Stage | 訓練対象 | データ | 目的 |
|---|---|---|---|
| **1. Language-Image Alignment** | **MLP プロジェクタのみ**（InternViT-300M + LLM 凍結） | キャプション + 検出 + grounding + OCR（[[sources/internvl-1-5\|InternVL 1.5]] と同じ） | 視覚と言語の整列 |
| **2. Visual Instruction Tuning** | **全パラメータ** | 多様な指示データ | 指示追従 + 世界知識注入 |

[[entities/internvl-1-5|InternVL 1.5]] の 2 段階訓練と同じパターン。

### ドメイン適応版（Stage 3 相当）

| Stage | 訓練対象 | データ | 目的 |
|---|---|---|---|
| **3. Domain Adaptation** | **全パラメータ** | ドメイン特化 + 汎用ミックス | 特定ドメインへの転移 |

---

## 統一ドメイン適応フレームワーク

**5 種類のタスクを VQA 形式に統一**:

| タスクタイプ | 形式 |
|---|---|
| 画像分類 | 多肢選択 "Classify ... Answer with one word." |
| 視覚的グラウンディング | `<ref>名前</ref><box>[[x1,y1,x2,y2]]</box>`（座標 0-1000） |
| 領域知覚 | 画像 box 描画 or 質問内 `<box>` 表記 |
| 複数視点（6 視点自律走行） | 6 画像を 896×448 + 結合 → 2688×896 → 12 タイル + サムネ + "CAM_FRON" 注釈 |
| 動画フレーム | "Frame1: <img>... Frame2: <img>..." 形式、最大 40 フレーム |

**訓練戦略**: 全パラメータ微調整、汎用:特化 = 1:1 〜 1:4。

---

## 主要結果

### 汎用マルチモーダル（表 2）

| モデル | パラメータ | Avg Score | vs InternVL2-76B |
|---|---|---|---|
| InternVL2-Llama3-76B | 76B | 81.4 | **100% (基準)** |
| **Mini-InternVL-4B** | **4B (5%)** | **72.8** | **90%** |
| Mini-InternVL-2B | 2B (3%) | 66.8 | 82% |
| Mini-InternVL-1B | 1B (1%) | 60.6 | 74% |
| Gemini-Pro-1.5 | – | 73.6 | – |
| GPT-4V-0409 | – | 75.4 | – |
| MiniCPM-V 2.0 | 3B | 58.3 | 72% |
| Qwen2-VL-2B | 2B | 68.5 | 84% |

### ドメイン適応 — 自律走行（表 4, 5）

DriveLM Challenge Leaderboard:

| Method | #Param | Final Score |
|---|---|---|
| InternVL4Drive-v2 (SOTA) | 26B | 0.6002 |
| **Mini-InternVL-DA-2B** | **2B** | **0.5958（1/13 サイズで匹敵）** |
| Mini-InternVL-DA-4B | 4B | 0.5821 |
| Mini-InternVL-DA-1B | 1B | 0.5686 |
| **Mini-InternVL-4B (DA なし)** | **4B** | **0.3051**（DA で +0.27） |

MME-RealWorld 自律走行:

| Method | Avg |
|---|---|
| **Mini-InternVL-DA-4B** | **49.38** |
| InternVL2-Llama3-76B | 44.30 |
| LLaVA-OneVision-7B | 41.75 |
| Claude 3.5 Sonnet | 32.10 |
| **GPT-4o** | **24.60** |

### ドメイン適応 — 医療画像（GMAI-MMBench、表 9）

| Model | 2D Seg C | 2D Seg M | 2D Cls |
|---|---|---|---|
| **GPT-4V** | **47.87** | **46.58** | **42.24** |
| LLaVA-Med（医療特化） | 18.45 | 18.97 | 21.15 |
| RadFM 14B（医療特化） | 20.43 | 20.27 | 25.71 |
| Claude3-Opus | 33.56 | 33.36 | 32.17 |
| **Mini-InternVL-DA-4B** | **41.41** | **40.45** | **41.34** |

### ドメイン適応 — リモートセンシング（表 11）

| Method | RSVQA-LR Avg | RSVQA-HR-T2 Avg | DIOR-RSVG @0.5 |
|---|---|---|---|
| GeoChat (7B) | 90.70 | 83.19 | 72.30 |
| SkyEyeGPT | 84.19 | 81.89 | 88.59 |
| SkySenseGPT | 92.69 | 76.64 | – |
| **Mini-InternVL-DA-1B** | 87.37 | 90.24 | 89.73 |
| **Mini-InternVL-DA-2B** | 89.87 | 90.30 | 89.24 |
| **Mini-InternVL-DA-4B** | 89.46 | 90.18 | **92.04** |

---

## アブレーション

### 1. 知識蒸留の効果（表 12）

**InternViT-300M vs CLIP-ViT-L-336px (同サイズ 300M)**:

| ベンチ | CLIP-L | InternViT-300M | 差 |
|---|---|---|---|
| MMB-EN | 70.3 | 73.2 | +2.9 |
| ChartQA | 70.9 | 76.2 | +5.3 |
| **DocVQA** | 77.5 | **85.9** | **+8.4** |
| InfoVQA | 49.6 | 57.7 | +8.1 |
| MME-RW (AD) | 43.7 | 48.0 | +4.3 |

**OCR / Document / Chart で蒸留の優位性が顕著**。

### 2. データ比率（図 4(a)）

- ドメイン特化のみ: 性能最適にならず
- **r=0.25（汎用:特化 = 1:4）で最適**
- r > 0.25: 性能低下

### 3. 訓練サンプル数（図 4(b)）

- **全データの 1/4 で性能ほぼ同等**
- 計算負荷を 4× 削減可能

### 4. 適応手法比較（表 13, 図 5）

| 手法 | GPU memory | Speed | 性能上限 | 特性 |
|---|---|---|---|---|
| **Full-parameter** | 33.11 GB | 185 iter/h | **最高** | 重い |
| Freezing ViT | 29.87 GB | 260 iter/h | 中 | バランス |
| LoRA | **23.40 GB** | **263 iter/h** | 低 | 軽い、grounding 弱い |

すべて **1500 ステップで収束**。

---

## アーキテクチャ比較

### vs InternVL 1.5（[[entities/internvl-1-5]]）

| 項目 | InternVL 1.5 | Mini-InternVL-4B |
|---|---|---|
| 視覚 | InternViT-6B-V1.5 (5.9B) | **InternViT-300M (300M, 1/20)** |
| LLM | InternLM2-20B-Chat | **Phi-3-Mini (3.8B, 1/5)** |
| 計 | 26B | **4B (1/6.5)** |
| 動的解像度 | 1-40 タイル | 1-40 タイル（同じ） |
| OCRBench | 724 | 788（**+64**） |
| DocVQA | 90.9 | 89.2 (-1.7) |
| MMBench-EN | 82.2 | 78.6 (-3.6) |
| MMMU | 45.2 | **48.3** (+3.1) |

**Mini-InternVL-4B は 1/6.5 のサイズで OCRBench を上回り、MMMU でも勝つ**。Phi-3-Mini が「Microsoft の教科書データで訓練された強い軽量 LLM」だったことが効いている。

### vs Qwen2-VL-2B（同時期軽量 MLLM）

| 項目 | Qwen2-VL-2B | Mini-InternVL-2B |
|---|---|---|
| パラメータ | 2B | 2B |
| Avg Score | **68.5** | 66.8 |
| MMMU | 42.2 | 36.3 |
| MathVista | 43.0 | **46.3** |
| AI2D | 74.7 | 74.1 |
| ChartQA | 73.5 | **76.2** |
| DocVQA | **90.1** | 86.9 |
| OCRBench | **794** | 784 |
| MMB-EN | **74.9** | 73.2 |

Qwen2-VL-2B にやや劣るが、**ドメイン適応フレームワークが Mini-InternVL の付加価値**。

### vs MiniCPM-V 2.0（直接の軽量競合）

| 項目 | MiniCPM-V 2.0 (3B) | Mini-InternVL-2B | Mini-InternVL-4B |
|---|---|---|---|
| 平均スコア | 58.3 | **66.8** | **72.8** |

**全方位で Mini-InternVL が MiniCPM-V を凌駕**。

---

## InternVL シリーズ全体での位置づけ

```
[2023-12] InternVL 1.0 (CVPR 2024, [[entities/internvl]])
   └─ InternViT-6B + QLLaMA-8B + Vicuna-7B/13B（対比 + 生成 + 対話統合）

[2024-02] InternVL 1.2 / 1.2-Plus
   └─ InternViT-6B-V1.2 + MLP + Yi-34B（QLLaMA 廃止、40B）

[2024-04] InternVL 1.5 ([[entities/internvl-1-5]])
   ├─ InternViT-6B-V1.5（dynamic 448）+ MLP + InternLM2-20B-Chat（26B）
   └─ 18 ベンチ中 8 SoTA、GPT-4V との差を縮める

[2024-07] InternVL2 (1B 〜 76B レンジ)
   └─ プログレッシブ整列、複数 LLM family サポート

[2024-10] Mini-InternVL ← 本ページ
   ├─ InternViT-300M（**InternViT-6B からの蒸留**）+ MLP + 軽量 LLM
   ├─ Mini-InternVL-1B/2B/4B + DA バリアント（自律走行/医療/リモセン）
   └─ **5% パラメータで 90% 性能、ドメイン適応フレームワーク確立**

[2024-12] InternVL 2.5 ([[entities/internvl-2-5]])
   ├─ InternViT-300M-V2.5 / InternViT-6B-V2.5 + InternLM 2.5 / Qwen 2.5
   ├─ 1B/2B/4B/8B/26B/38B/78B の 7 サイズ、Mini-InternVL の軽量化路線を吸収
   └─ **MMMU 70.1% で 70% を超えた初のオープン MLLM**

[2025-04] InternVL 3 ([[entities/internvl-3]])
   ├─ **Native Multimodal Pre-Training** へ哲学転換
   ├─ V2PE + MPO + VisualPRM
   └─ **MMMU 72.2% で SOTA 更新**
```

**Mini-InternVL は「軽量化 + ドメイン特化」という InternVL シリーズの **第 2 の柱** を確立した論文**。

---

## モデルカタログ

公開済み（HuggingFace `OpenGVLab/` 配下）:

| モデル | 用途 | パラメータ |
|---|---|---|
| `Mini-InternVL-Chat-1B-V1-5` | 軽量 MLLM（1B） | 1B |
| `Mini-InternVL-Chat-2B-V1-5` | 軽量 MLLM（2B） | 2B |
| `Mini-InternVL-Chat-4B-V1-5` | 軽量 MLLM（4B） | 4B |
| `InternViT-300M-448px` | 視覚エンコーダのみ（line probe / 知覚タスク用） | 300M |
| 各 DA 版 | 自律走行 / 医療 / リモセンの fine-tuned 版 | 1B / 2B / 4B |

**ライセンス**: MIT（コード）+ 各 LLM の独立ライセンス（Qwen2: Apache 2.0、InternLM2: 商用制限、Phi-3: MIT）

---

## 関連ページ

- 詳細解説: [[sources/mini-internvl]]
- 翻訳: [[translations/mini-internvl]]
- 視覚エンコーダ: [[entities/internvit-300m]]（蒸留先の VFM 単独ページ）
- 直接の祖: [[entities/internvl-1-5]] / [[sources/internvl-1-5]]（動的解像度の系譜）/ [[entities/internvl]] / [[sources/internvl]]（InternViT-6B 教師の系譜）
- 関連視覚エンコーダ: [[entities/clip]]（CLIP-ViT-L-336px が InternViT-300M の初期化に使用）
- 関連概念: [[concepts/knowledge-distillation]]（中核技法）, [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]]
