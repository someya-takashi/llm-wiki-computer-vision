---
type: entity
entity_kind: model
aliases: [Gemma 3, Gemma3, Gemma-3, Gemma 3 PT, Gemma 3 IT, Gemma-3-1B, Gemma-3-4B, Gemma-3-12B, Gemma-3-27B, Gemma-3-27B-IT]
related: ["[[entities/siglip]]", "[[entities/qwen3-vl]]", "[[entities/qwen2-vl]]", "[[entities/internvl-1-5]]", "[[entities/internvl-3-5]]", "[[concepts/vision-transformer]]", "[[concepts/rotary-position-embeddings]]", "[[concepts/foundation-model]]", "[[concepts/knowledge-distillation]]"]
sources: ["[[sources/gemma-3]]"]
updated: 2026-05-31
---

# Gemma 3 / Gemma-3-1B / Gemma-3-4B / Gemma-3-12B / Gemma-3-27B

**Gemma 3** は **Google DeepMind** が開発するオープン軽量 MLLM 系第 3 世代（2025 年 3 月公開、arXiv:2503.19786）。**Gemini 2.0 と co-design** され、**1B/4B/12B/27B の 4 サイズ × PT/IT 2 形態**で提供される。**4B 以上は SigLIP 400M variant でマルチモーダル化**、**5:1 local:global attention で 128K 文脈**を実現。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2025 年 3 月（arXiv:2503.19786） |
| 開発元 | Google DeepMind |
| サイズ展開 | **1B / 4B / 12B / 27B** の 4 サイズ |
| 形態 | **PT (pre-trained) / IT (instruction-tuned)** の 2 形態 = **計 8 公開モデル** |
| ライセンス | **Gemma License**（Apache 2.0 風、商用可） |
| リポジトリ | https://github.com/google-deepmind/gemma |
| HF Hub | `google/gemma-3-{1b,4b,12b,27b}-{pt,it}` |
| 文脈長 | **128K**（1B のみ 32K） |
| マルチモーダル | **4B/12B/27B のみ**（1B は text-only） |
| 量子化 | **QAT 提供**: per-channel int4 / per-block int4 / SFP8 |

## モデル詳細

| Model | Vision Encoder | Embedding | Non-embedding | Total | Context |
| --- | --- | --- | --- | --- | --- |
| **1B** | なし | 302M | 698M | **1.0B** | **32K** |
| **4B** | SigLIP 400M | 675M | 3,209M | **3.9B** | 128K |
| **12B** | SigLIP 400M | 1,012M | 10,759M | **11.8B** | 128K |
| **27B** | SigLIP 400M | 1,416M | 25,600M | **27.0B** | 128K |

**重要**:
- 全マルチモーダル・サイズ（4B/12B/27B）で **同じ SigLIP 400M variant を共有**
- 視覚エンコーダは **訓練中ずっと凍結**（言語モデルのみ訓練）
- 視覚埋め込みは **事前計算**して訓練コスト 0 で言語モデル訓練

## アーキテクチャの 4 つの主要技術

### 1. Vision Modality（SigLIP 400M + Pan & Scan）

- **Vision Encoder**: [[entities/siglip|SigLIP]] の 400M variant
- **入力**: **896 × 896 固定正方形**
- **出力圧縮**: 4×4 average pooling → **256 トークン固定**
- **Pan & Scan (P&S)**:
  - 推論時のみの適応的ウィンドウ・アルゴリズム
  - 非正方形・高解像度画像を **重複しない等サイズクロップ**に分割
  - 各クロップを 896×896 にリサイズ
  - 最大クロップ数を制御、推論高速化のため無効化可
  - **LLaVA 風タイル分割**

### 2. 5:1 Local:Global Attention（KV キャッシュ最適化）

- **local sliding window self-attention** と **global self-attention** を交互配置
- **5 local 層 : 1 global 層**（Gemma 2 は 1:1）
- **local の sliding window 1024 トークン**のみ
- **long context は global 層のみ**が対応
- 結果: 32K 文脈で KV キャッシュ・メモリ・オーバーヘッドを **60% → <15%** に削減

### 3. RoPE スケーリング（128K 文脈の実現）

- **global self-attention**: RoPE base **10k → 1M**
- **local self-attention**: RoPE base 10k（変更なし）
- **32K 系列で事前学習 → 128K へスケーリング係数 8 で拡張**

### 4. QK-norm + GQA

- Gemma 2 の **soft-capping** を **QK-norm** に置換（Chameleon、Olmo 2 着想）
- **Grouped-Query Attention (GQA)** + RMSNorm（post-norm + pre-norm）

## Quantization Aware Training（QAT）

3 つの量子化形式を提供（27B 例）：

| 形式 | 27B 単体 | 27B + KV (32k) | 用途 |
| --- | --- | --- | --- |
| bf16 (Raw) | 54.0 GB | 72.7 GB | サーバ |
| **Int4 per-channel** | **14.1 GB** | 32.8 GB | 消費者 GPU |
| Int4 blocks=32 | 15.3 GB | 34.0 GB | バランス |
| SFP8 | 27.4 GB | 46.1 GB | 高精度量子化 |

- **5,000 ステップの微調整**で量子化（QAT）
- llama.cpp 互換

## 訓練

### 事前学習データ

| Model | トークン数 |
| --- | --- |
| 1B | 2T |
| 4B | 4T |
| 12B | 12T |
| **27B** | **14T** |

- **画像 + テキスト + 多言語**の混合
- **Gemini 2.0 と同じ SentencePiece トークナイザ**（262k 語彙、数字分割、空白保持、バイトレベル）
- **知識蒸留**: 1 トークンあたり 256 ロジット・サンプリング

### 訓練インフラ

| Model | TPU | #Chips | Data Shards | Seq. Shards | Replica |
| --- | --- | --- | --- | --- | --- |
| 1B | TPUv5e | 512 | 16 | 16 | 2 |
| 4B | TPUv5e | 2048 | 16 | 16 | 8 |
| 12B | TPUv4 | 6144 | 16 | 16 | 24 |
| 27B | TPUv5p | 6144 | 24 | 8 | 32 |

- **ZeRO-3** + Pathways + GSPMD + MegaScale XLA + Jax

### 事後学習（IT モデル）

- 大規模 IT 教師からの**改善された知識蒸留**
- **BOND + WARM + WARP の改善版**による RL 微調整
- 報酬: 重み平均報酬モデル + 人間フィードバック + コード実行 + 数学 ground-truth
- ChatML 風フォーマット: `<start_of_turn>` / `<end_of_turn>` / `[BOS]`

## 主要ベンチマーク結果

### LMSys Chatbot Arena Elo（2025 年 3 月 8 日）

**Gemma-3-27B-IT Elo 1338（rank 9）** — DeepSeek-V3 (671B/37B MoE, 1318)、Llama-3.1-405B (1269)、Qwen2.5-72B (1257) を上回る。Gemma-2-27B-it (1220) から **+118 Elo の飛躍**。

### IT モデル（表 6 vs 商用フロンティア）

| ベンチ | Gemini 1.5 Pro | Gemini 2.0 Pro | **Gemma 3 27B** |
| --- | --- | --- | --- |
| MMLU-Pro | 75.8 | 79.1 | **67.5** |
| **MATH** | 86.5 | 91.8 | **89.0** |
| GPQA Diamond | 59.1 | 64.7 | 42.4 |
| MMMU val | 65.9 | 72.7 | **64.9** |
| FACTS Grounding | 80.0 | 82.8 | **74.9** |

### マルチモーダル（表 16、IT with P&S）

| ベンチ | 4B | 12B | **27B** |
| --- | --- | --- | --- |
| MMMU val | 48.8 | 59.6 | **64.9** |
| **DocVQA** | 75.8 | 87.1 | **86.6** |
| **InfoVQA** | 50.0 | 64.9 | **70.6** |
| **MathVista** | 50.0 | 62.9 | **67.6** |
| TextVQA | 57.8 | 67.7 | 65.1 |
| AI2D | 74.8 | 84.2 | 84.5 |
| ChartQA | 68.8 | 75.7 | 78.0 |

### P&S アブレーション（表 8）

| | DocVQA | **InfoVQA** | TextVQA |
| --- | --- | --- | --- |
| 27B w/o P&S | 85.6 | 59.4 | 68.6 |
| **27B w/ P&S** | **90.4** (+4.8) | **76.4** (+17.0) | **70.2** (+1.6) |

### PaliGemma 2 比較（表 12、転送学習）

Gemma 3 27B は **DocVQA +4.4 / InfoVQA +14.4 / TextVQA +8.1 / ChartQA +12.1** で PaliGemma 2 27B を圧倒。さらに **4B/12B は同 896² 解像度の PaliGemma 2 9B/27B より 10× 安価に転送**。

### 長文脈（表 15、IT）

| ベンチ | Context | 27B IT |
| --- | --- | --- |
| **RULER** | 32K | **91.1** |
| RULER | 128K | 66.0 |
| MRCR | 32K | 63.2 |
| MRCR | 128K | 59.3 |

32K では SOTA レベル、128K への外挿で劣化（**Qwen3-VL の 1M YaRN 99.5% には大きく及ばず**）

### 動画理解（表 17、IT、16 frames）

| ベンチ | 4B | 12B | 27B |
| --- | --- | --- | --- |
| Perception Test MCVQA | 50.6 | 54.9 | **58.1** |
| ActivityNet-QA | 46.3 | 50.4 | **52.8** |

16 フレーム制約のため Qwen2-VL/3-VL 系の動的 FPS には届かず

## 限界

- **固定 896² + P&S**: タイル分割の限界（境界での意味的連続性損失）
- **256 トークン固定圧縮**: 情報密度の高い画像で損失
- **128K で RULER 27B 66.0**: Qwen3-VL の 1M 99.5% に大きく劣る
- **MMMU で Gemini 1.5 Pro -1.0 / Gemini 2.0 Pro -7.8**
- **動画 16 frames 制約**
- **MoE 採用なし**: InternVL 3.5 (241B-A28B) / Qwen3-VL (235B-A22B) と対照的
- **訓練データ非公開**（cf. InternVL 3 は OpenGVLab/InternVL-Data 公開）
- **PaliGemma 2 との分業不明確**

## 系譜・後継

- **Gemma 1**（2024 Feb, 2B/7B, テキスト専用）
- **Gemma 2**（2024 Jul, 2B/9B/27B, テキスト専用, distillation 中心, 1:1 local:global, soft-capping）
- **Gemma 3**（2025 Mar, 1B/4B/12B/27B, **マルチモーダル + 128K + 5:1 attention + QK-norm + QAT**）← 本ページ
- 並列発展: **PaliGemma 2**（より大規模なオープン VLM、Google の別系列）

## 関連ページ

- [[sources/gemma-3]] — 原典の要約
- [[translations/gemma-3]] — 原典の翻訳
- [[entities/siglip]] — Gemma 3 が 400M variant で活用した視覚エンコーダ
- [[concepts/vision-transformer]] — SigLIP の基盤
- [[concepts/rotary-position-embeddings]] — RoPE 10k→1M スケーリング
- [[concepts/foundation-model]] — Google 系 MLLM の代表
- [[concepts/knowledge-distillation]] — 14T トークン蒸留学習
- [[questions/vit-dynamic-resolution-evolution]] — Pan & Scan = タイル分割路線の最新形
- [[entities/qwen3-vl]] — Qwen 系の対比（VL 純化 + 動的解像度 + MoE）
- [[entities/qwen2-vl]] — Naive Dynamic Resolution の対比
- [[entities/internvl-1-5]] — タイル分割路線の前任者
- [[entities/internvl-3-5]] — InternVL 系の対比（Cascade RL + MoE + ViR）
