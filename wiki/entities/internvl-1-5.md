---
type: entity
entity_kind: model
aliases: [InternVL 1.5, InternVL-Chat-V1-5, InternViT-6B-448px-V1.5, InternViT-6B-448px-V1.2, InternVL 1.2]
tags: [mllm, vllm, vision-language, opengvlab, internvl-series, gpt-4v-competitor]
related: ["[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/vision-transformer]]", "[[concepts/zero-shot-transfer]]", "[[entities/internvl]]", "[[entities/clip]]", "[[entities/siglip]]", "[[entities/perception-encoder]]"]
sources: ["[[sources/internvl-1-5]]"]
updated: 2026-05-29
---

# InternVL 1.5 — 26B のオープンソース MLLM、GPT-4V との差を 8/18 ベンチマークで埋めた到達点

## 概要

**InternVL 1.5** = OpenGVLab Shanghai AI Lab の **InternVL シリーズ第 4 世代**（1.0 → 1.2 → 1.5）。論文タイトル「**How Far Are We to GPT-4V?**」が示すように、当時のオープンソース MLLM と商用商品（GPT-4V / Gemini / Claude-3 / Qwen-VL-Max）の差を明確に縮めることを目的とした技術報告。

- 論文: "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites"
- arXiv: 2404.16821（2024 年 4 月、technical report）
- HuggingFace: <https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5>
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- Demo: <https://internvl.opengvlab.com>
- 詳細: [[sources/internvl-1-5]] / 翻訳: [[translations/internvl-1-5]]

---

## アーキテクチャ（ViT-MLP-LLM）

```
画像入力（任意解像度）
   ↓ 動的アスペクト比マッチング（35 通り）
448×448 タイル × N (N=1〜40) + 448×448 サムネイル
   ↓ InternViT-6B-448px-V1.5（5.9B、45 層、凍結 or 訓練）
1024 visual tokens / tile
   ↓ Pixel Shuffle（1/4 圧縮、4 patches → 1 token）
256 visual tokens / tile
   ↓ MLP プロジェクタ（ランダム初期化 → 訓練）
LLM 埋め込み空間
   ↓ InternLM2-20B-Chat（20B、ベース版ではなく Chat 版）
出力テキスト
```

**重要設計判断**: [[entities/internvl|InternVL 1.0]] の **QLLaMA（8B 言語ミドルウェア）を完全廃止**し、LLaVA 系の標準アーキテクチャに統一。シンプルな MLP プロジェクタで接続。

---

## コンポーネント詳細

### 1. InternViT-6B-448px-V1.5（視覚エンコーダ）

| 項目 | V1.0（InternVL 1.0） | V1.2（InternVL 1.2） | **V1.5（本モデル）** |
|---|---|---|---|
| パラメータ | 5.9B | 5.9B | **5.9B** |
| アーキテクチャ | vanilla ViT | vanilla ViT | vanilla ViT |
| 層数 | 48 | **45**（後ろ 3 層削除） | **45** |
| 解像度 | 224 | 448 固定 | **448 dynamic（1-12 タイル）** |
| 連携 LLM | LLaMA-7B（QLLaMA 経由） | Nous-Hermes-2-Yi-34B | **InternLM2-20B-Chat** |
| 訓練データ | 4.98B 対 (Stage 1) | + OCR・キャプションデータ | + バイリンガル OCR + Common Crawl PDF |

**「後ろから 4 層目」発見**: V1.2 開発時、48 層中 **第 45 層の出力が MLLM タスクで最良** であることを発見。最終 3 層を削除（理由は推測: 最終層は対比学習用に過度に特化、中間層に多目的特徴が育つ）。これは [[entities/perception-encoder|PE]] の「最良特徴は中間層に育つ」発見の先取り。

### 2. Pixel Shuffle（視覚 token 圧縮）

```
[448×448 tile] → InternViT → [32×32 = 1024 tokens of 3200-d]
                                ↓ Pixel Shuffle (空間 2×2 → チャンネル 4×)
                              [16×16 = 256 tokens of 12800-d]
                                ↓ MLP プロジェクタ
                              [256 tokens of LLM_dim]
```

token 数を **1/4** に削減。これにより 40 タイル + サムネイル = 41 × 256 = **10,496 visual tokens** でも LLM の文脈長 4096 を超えるが、long context inference では対応可能（推論時のみ）。

### 3. InternLM2-20B-Chat（LLM）

- Shanghai AI Lab 独自の中国語強化 LLM
- **Chat 版**（指示追従済み）を使用、base 版ではない
- 文脈長: 訓練時 4096、推論時はそれ以上に拡張可能
- 中国語と英語の双方で強い

### 4. MLP プロジェクタ

- 2 層 MLP（GELU 活性化）
- 入力: 12800 次元（Pixel Shuffle 後）
- 出力: InternLM2-20B 埋め込み次元
- ランダム初期化、Stage 1 + Stage 2 で訓練

---

## 動的高解像度（Dynamic High-Resolution）

### 35 種のアスペクト比

タイル数 N=1〜12 から構成可能な全 35 通り：
```
{1:1}, {1:2, 2:1}, {1:3, 3:1}, {1:4, 4:1, 2:2}, {1:5, 5:1},
{1:6, 6:1, 2:3, 3:2}, {1:7, 7:1}, {1:8, 8:1, 2:4, 4:2},
{1:9, 9:1, 3:3}, {1:10, 10:1, 2:5, 5:2}, {1:11, 11:1},
{1:12, 12:1, 2:6, 6:2, 3:4, 4:3}
```

### マッチング手順

```python
def dynamic_aspect_ratio_matching(image):
    input_ar = image.width / image.height
    best_ratio = min(PREDEFINED_35, key=lambda r: abs(r - input_ar))
    # 多候補時は元画像の 2 倍面積以下を優先
    if multiple_matches and best_ratio.area > 2 * image.area:
        best_ratio = smaller_ratio
    return best_ratio
```

### 訓練 vs テスト

| 段階 | タイル数上限 | 視覚トークン数 | 用途 |
|---|---|---|---|
| 訓練 | **12** | 256-3,328 | データに含まれる最大解像度 |
| テスト | **40**（zero-shot 拡張） | 256-10,496 | 4K 解像度文書対応 |

ゼロショットで 12 → 40 タイル拡張可能なのは、InternViT が ViT であり位置埋め込みが柔軟だから。

---

## 訓練の 2 段階

### Stage 1: Pre-training

| 項目 | 内容 |
|---|---|
| 訓練対象 | **InternViT-6B + MLP プロジェクタ**（LLM は凍結） |
| データ | 表 1(a)（事前学習データ） |
| データ量 | 大規模（具体的サイズは論文未記載） |
| 主要内訳 | Captioning 53.9% / Detection 5.2% / OCR (大) 32.0% / OCR (小) 8.9% |
| 目的 | 視覚特徴抽出最適化、特に OCR 強化 |

### Stage 2: Fine-tuning

| 項目 | 内容 |
|---|---|
| 訓練対象 | **全 26B パラメータ**（InternViT + MLP + LLM すべて） |
| データ | 表 1(b)（指示データ） |
| データ量 | 約 4.2M 指示サンプル |
| 文脈長 | 4096 |
| プロンプト | LLaVA 1.5 と同じ応答整形 |
| 目的 | マルチモーダル能力統合 + 指示追従 |

InternVL 1.0 の **3 段階訓練（contrastive 4.98B → ITC/ITM/ITG 1.03B → SFT 4M）** から **2 段階に簡素化**。MLLM 時代の標準パターン（LLaVA 流）に追随。

---

## モデルカタログ

| モデル名 | 視覚 | LLM | 計 | 公開リソース |
|---|---|---|---|---|
| **InternVL-Chat-V1-2** | InternViT-6B-V1.2 (5.9B) | Nous-Hermes-2-Yi-34B (34B) | **40B** | HF: `OpenGVLab/InternVL-Chat-V1-2` |
| **InternVL-Chat-V1-2-Plus** | InternViT-6B-V1.2 (5.9B) | Yi-34B-Chat (34B) | **40B** | HF: `OpenGVLab/InternVL-Chat-V1-2-Plus` |
| **InternVL-Chat-V1-5**（本モデル） | InternViT-6B-V1.5 (5.9B) | InternLM2-20B-Chat (20B) | **26B** | HF: `OpenGVLab/InternVL-Chat-V1-5` |
| **InternViT-6B-448px-V1.5**（VFM のみ） | InternViT-6B-V1.5 | – | 5.9B | HF: `OpenGVLab/InternViT-6B-448px-V1-5` |

**ライセンス**: MIT（コード）+ 各 LLM ライセンス（InternLM2 は Apache 2.0 + Commercial 制限）

---

## 主要結果

### 18 ベンチマーク（表 2）

OCR 関連（**5/5 健闘、2 つで SoTA**）:

| ベンチマーク | InternVL 1.5 | 最強商用 | 差 |
|---|---|---|---|
| DocVQA | 90.9 | Qwen-VL-Max 93.1 | -2.2 |
| **ChartQA** | **83.8** | Claude-3 Haiku 81.7 | **+2.1 SoTA** |
| InfoVQA | 72.5 | Gemini Ultra 80.3 | -7.8 |
| TextVQA | 80.6 | Gemini Ultra 82.3 | -1.7 |
| **OCRBench** | **724** | Qwen-VL-Max 723 | **+1 SoTA** |

汎用マルチモーダル（**9 ベンチマーク**）:

| ベンチマーク | InternVL 1.5 | GPT-4V | 結果 |
|---|---|---|---|
| MME | 2187.8 | 1926.6 | +261 |
| RealWorldQA | 66.0 | 61.4 | +4.6 |
| AI2D | 80.7 | 78.2 | +2.5 |
| MMMU | 45.2 | 56.8 | **-11.6** |
| MMBench-EN | 82.2 | 77.0 | +5.2 |
| **MMBench-CN** | **82.0** | 74.4 | **+7.6 SoTA** |
| **CCBench** | **69.8** | 46.5 | **+23.3 SoTA** |
| MMVet | 62.8 | 67.6 | -4.8 |
| SEED | 76.0 | 71.6 | +4.4 |
| **HallusionBench** | **49.3** | 46.5 | **+2.8 SoTA** |

数学（**MathVista で GPT-4V 超え**）:

| モデル | MathVista |
|---|---|
| **InternVL 1.5** | **53.5 SoTA** |
| Gemini Ultra 1.0 | 53.0 |
| Grok-1.5V | 52.8 |
| GPT-4V | 49.9 |

マルチターン対話（**ConvBench で GPT-4V との差大**）:

| モデル | ConvBench Pairwise R₁ |
|---|---|
| GPT-4V | 39.51 |
| Claude-3 Opus | 36.60 |
| **InternVL 1.5** | **17.65**（オープンソース中で首位だが商用には大敗） |

**SoTA 達成数**: 18 中 **8** （ChartQA, OCRBench, MMBench-CN, CCBench, HallusionBench, MathVista, InfoVQA は最強オープン, AI2D 等は同等競合）

### MMT-Bench

| モデル | Overall | Overall\* |
|---|---|---|
| Qwen-VL-Plus | 62.3 | 56.6 |
| GPT-4V | 62.0 | 55.5 |
| Gemini Pro 1.0 | 61.6 | 55.1 |
| LLaVA-NeXT (35B) | 60.8 | 56.3 |
| **InternVL 1.2 (40B)** | **63.4** | **58.2** |
| **InternVL 1.5 (26B)** | **59.0** | **56.2** |

**InternVL 1.2 (40B) が MMT-Bench で最強**、ただし 1.5 で軽量化を取って若干低下。

---

## アーキテクチャ比較

### vs InternVL 1.0（[[entities/internvl]]）

| 項目 | InternVL 1.0 | InternVL 1.5 |
|---|---|---|
| 視覚 | InternViT-6B 224 (48 層) | **InternViT-6B-448px-V1.5 (45 層、dynamic 448)** |
| Glue | **QLLaMA (8B)** | **MLP (~150M)** |
| LLM | Vicuna-7B/13B | **InternLM2-20B-Chat** |
| 計 | 13-19B | **26B** |
| 訓練段階 | 3 段階 | **2 段階** |
| 主軸 | 対比 + 生成 + 対話統合 | **MLLM 専用、OCR + 多言語特化** |
| 主要結果 | IN-1K ZS 83.2 / Flickr30K I2T 95.7 | **18 ベンチ中 8 SoTA、GPT-4V 並み** |

### vs LLaVA-NeXT-35B（同時期競合）

| 項目 | LLaVA-NeXT (35B) | InternVL 1.5 (26B) |
|---|---|---|
| 視覚 | CLIP-L 336 (300M) | **InternViT-6B (5.9B、~20×)** |
| Glue | MLP | MLP（同じ） |
| LLM | LLaMA2-Yi-34B | InternLM2-20B-Chat |
| 解像度戦略 | 672×672 dynamic | **448 dynamic 1-40 tiles** |
| OCRBench | 574 | **724（+150）** |
| ChartQA | 68.7 | **83.8（+15.1）** |
| MMBench-CN | 79.0 | **82.0（+3.0）** |
| CCBench | 49.2 | **69.8（+20.6）** |

**LLaVA-NeXT に対し OCR と中国語で圧倒的優位**。VFM スケール（300M vs 5.9B）と動的解像度の効果。

### vs GPT-4V（商用フロンティア）

| 項目 | GPT-4V | InternVL 1.5 (26B) |
|---|---|---|
| パラメータ | ≥ 100B 推定 | 26B |
| 公開 | クローズド | **オープン MIT/InternLM** |
| OCR 5 ベンチ平均 | ~76.5 | **~76.5（同等）** |
| 中国文化（CCBench） | 46.5 | **69.8（+23.3）** |
| 数学（MathVista） | 49.9 | **53.5（+3.6）** |
| MMMU（多分野知識） | 56.8 | **45.2（-11.6）** |
| ConvBench マルチターン | 39.51 | 17.65（-21.86） |

**強み**: OCR / 中国語 / 数学
**弱み**: MMMU（多分野知識）/ マルチターン対話

---

## InternVL シリーズ全体（参考）

```
[2023-12] InternVL 1.0 (CVPR 2024, [[entities/internvl]])
   ├─ InternViT-6B 224 (48 layers) + QLLaMA-8B + Vicuna-7B/13B
   └─ 対比 + 生成 + 対話統合、3 段階訓練

[2024-02] InternVL 1.2 (Plus)
   ├─ InternViT-6B-448px-V1.2 (45 layers, fixed 448) + MLP + Nous-Hermes-2-Yi-34B
   ├─ QLLaMA を廃止、MLP に簡素化
   ├─ 「後ろから 4 層目」発見、最終 3 層削除
   └─ 40B 計、MMT-Bench 強い

[2024-04] InternVL 1.5 ←（本ページ）
   ├─ InternViT-6B-448px-V1.5 (45 layers, dynamic 448) + MLP + InternLM2-20B-Chat
   ├─ 動的高解像度 1-12 タイル（test 40 タイル / 4K）
   ├─ バイリンガル OCR データ + データ翻訳パイプライン
   ├─ 26B、18 ベンチ中 8 SoTA
   └─ "How Far Are We to GPT-4V?" - 商用との差を初めて明確に縮める

[2024-07] InternVL 2.0
   └─ プログレッシブ整列、Sonnet 系の 1B〜108B レンジ

[2024-10] Mini-InternVL ([[entities/mini-internvl]])
   ├─ InternViT-300M（**InternViT-6B からの蒸留**）+ MLP + 軽量 LLM
   ├─ Mini-InternVL-1B/2B/4B + DA バリアント（自律走行/医療/リモセン）
   └─ **5% パラメータで InternVL2-76B の 90% 性能**、ドメイン適応フレームワーク確立

[2024-12] InternVL 2.5 ([[entities/internvl-2-5]])
   ├─ InternViT V2.5（6B + 300M）+ InternLM 2.5 / Qwen 2.5
   ├─ 1B/2B/4B/8B/26B/38B/78B の 7 サイズ
   ├─ Progressive Scaling Strategy 公式化、Test-Time Scaling（CoT + Majority Voting）
   └─ **MMMU 70.1% で 70% を超えた初のオープン MLLM**、商用 GPT-4o / Claude-3.5-Sonnet 超え

[2025-04] InternVL 3 ([[entities/internvl-3]])
   ├─ **Native Multimodal Pre-Training**（テキスト + マルチモーダル共同事前学習、哲学転換）
   ├─ V2PE + MPO + VisualPRM（test-time scaling）
   ├─ Qwen2.5 base 起点（Chat 版ではない）、Qwen2.5-Chat より純粋言語が強い
   └─ **MMMU 72.2% で再び SOTA 更新**

[2025-04] InternVL 3
   └─ Native multimodal pretraining（テキストと画像を最初から共訓練）
```

InternVL 1.5 は **シリーズ中で最も「GPT-4V を意識した」設計**であり、OCR・中国語・MLLM ベンチマークに特化したマイルストーンモデル。

---

## 関連ページ

- 詳細解説: [[sources/internvl-1-5]]
- 翻訳: [[translations/internvl-1-5]]
- 直接の祖: [[entities/internvl]] / [[sources/internvl]] (InternVL 1.0)
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]] — MLLM 用 VFM の選択肢
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/vision-transformer]], [[concepts/zero-shot-transfer]]
