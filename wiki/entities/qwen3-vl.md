---
type: entity
entity_kind: model
aliases: [Qwen3-VL, Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B, Qwen3-VL-32B, Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B, Qwen3VL, qwen3-vl]
related: [[entities/qwen2-5-vl]], [[entities/qwen2-vl]], [[entities/qwen-vl]], [[entities/siglip]], [[entities/internvl-3]], [[entities/internvl-3-5]], [[concepts/rotary-position-embeddings]], [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]]
sources: [[sources/qwen3-vl]]
updated: 2026-05-30
---

# Qwen3-VL / 6 サイズ × 2 バリアント = 12 公開モデル

**Qwen3-VL シリーズ**は Alibaba Group の Qwen Team が開発する MLLM 系譜の第 4 世代（2025 年 11 月 27 日公開、arXiv:2511.21631）。**Interleaved MRoPE + DeepStack + テキスト・ベース時間整合** という 3 つの構造革新と、**6 サイズ × 2 バリアント = 12 公開モデル**で、Qwen ファミリーが商用最先端と完全に肩を並べた決定的瞬間。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2025 年 11 月 27 日（arXiv:2511.21631 v2）、2025 年 12 月 1 日報告 |
| 開発元 | Alibaba Group / Qwen Team |
| サイズ展開 | **2B / 4B / 8B / 32B（dense）+ 30B-A3B / 235B-A22B（MoE）** = 6 サイズ |
| バリアント | **Instruct（non-thinking）+ Thinking** = 各サイズに 2 バリアント |
| 公開モデル数 | **12 モデル** |
| ライセンス | **Apache 2.0** |
| リポジトリ | https://github.com/QwenLM/Qwen3-VL |
| HF Hub | `Qwen/Qwen3-VL-{2B/4B/8B/32B/30B-A3B/235B-A22B}-{Instruct/Thinking}` |
| デモ | https://chat.qwen.ai |
| 文脈長 | **256K ネイティブ**（YaRN 拡張で 1M トークン = 2 時間動画まで） |
| 多言語 OCR | **39 言語**（Qwen2.5-VL の 10 言語から大幅拡張） |

## モデル詳細

### Dense モデル

| モデル | LLM | Vision Encoder | 用途 |
| --- | --- | --- | --- |
| **Qwen3-VL-2B** | Qwen3 2B | **SigLIP2-Large（300M）** | エッジ AI |
| **Qwen3-VL-4B** | Qwen3 4B | **SigLIP2-Large（300M）** | エッジ AI |
| **Qwen3-VL-8B** | Qwen3 8B | **SigLIP2-SO-400M** | バランス |
| **Qwen3-VL-32B** | Qwen3 32B | **SigLIP2-SO-400M** | 中規模 |

### MoE モデル

| モデル | LLM 総 | 活性 | Vision Encoder |
| --- | --- | --- | --- |
| **Qwen3-VL-30B-A3B** | 30B | **3B 活性** | SigLIP2-SO-400M |
| **Qwen3-VL-235B-A22B** | **235B** | **22B 活性** | SigLIP2-SO-400M |

**重要**:
- 全サイズで **SigLIP-2 から初期化して継続学習**（[[entities/qwen2-5-vl|Qwen2.5-VL]] のゼロから学習路線を放棄）
- 動的解像度は CoMP の方法論に基づく **2D-RoPE + 絶対位置補間**
- Vision-Language Merger: 2 層 MLP で **2×2 視覚特徴を 1 トークンに圧縮**
- DeepStack 用に **専用 Vision-Language Merger** を別途展開

## アーキテクチャの 3 つの主要構造革新

### 1. Interleaved MRoPE

- [[entities/qwen2-vl|Qwen2-VL]] の MRoPE は temporal/height/width を埋め込み次元の「**塊**」に分割していた → **周波数スペクトル不均衡**で長動画理解性能が劣化
- Qwen3-VL は t / h / w 成分を埋め込み次元にわたって **均一に交互配置**（interleaved）
- **各時空間軸が低周波帯と高周波帯の双方で均一に表現される**
- 動画の長距離位置モデリングを著しく改善

### 2. DeepStack（Meng et al., 2024）

- ViT の **中間 3 層から視覚トークンを抽出**（低・中・高レベル特徴）
- 各層用の **専用 Vision-Language Merger** で投影
- **最初の 3 つの LLM 層の隠れ状態に直接加算**
- 追加文脈長を導入しない多層融合
- アブレーション（表 12、200B トークン学習）: 平均 +1.3 ポイント、InfoVQA +2.3 / DocVQA +1.6 / OCRB +2.6

### 3. テキスト・ベース時間整合（最重要の方針転換）

- [[entities/qwen2-5-vl|Qwen2.5-VL]] の **MRoPE absolute time の 2 つの限界**:
  - 長動画で **temporal ID が過度に大きく疎**になり、長文脈理解を劣化
  - 多様な FPS で均一に分布したサンプリングが必要で **学習データ構築コストが膨大**
- Qwen3-VL は MRoPE absolute time を**捨て**、**明示的テキスト・タイムスタンプ・トークン**に置換
- 各動画時間パッチの前に `<3.0 seconds>` のようなフォーマットされたテキストを付加
- **秒形式 + HMS（時:分:秒）形式の双方**でタイムスタンプを生成
- 文脈長のわずかな増加と引き換えに精密な時間グラウンディングを実現
- **Charades-STA mIoU 63.5（Qwen2.5-VL 50.9 から +12.6 ポイント）**

## 4 段階事前学習（合計 ~2.2T トークン）

| Stage | 目的 | 学習対象 | トークン | Seq |
| --- | --- | --- | --- | --- |
| **S0** | Vision-Language Alignment | **Merger のみ** | 67B | 8,192 |
| **S1** | Multimodal Pre-Training | All | ~1T | 8,192 |
| **S2** | Long-Context Pre-Training | All | ~1T | **32,768** |
| **S3** | Ultra-Long-Context Adaptation | All | 100B | **262,144** |

**S0 でマージャのみを学習**するアライメント優先フェーズと、**S3 で seq 262K へ拡張**するウルトラ長文脈フェーズが Qwen2.5-VL からの主要な追加。

## 事前学習データ（8 カテゴリ）

1. **Image Caption + Interleaved Text-Image**: 微調整 Qwen2.5-VL-32B で再キャプション付け、書籍規模で 256K トークン系列にマージ
2. **Knowledge**: 12+ 意味カテゴリ、importance-based sampling
3. **OCR / Document Parsing / Long Document**: 3000 万社内 OCR + **多言語 29 言語追加**、QwenVL-HTML + QwenVL-Markdown（表は LaTeX 符号化）
4. **Grounding + Counting**: Grounding DINO で合成、**[0, 1000] 正規化座標系**（Qwen2.5-VL の絶対座標から復帰）
5. **Spatial Understanding + 3D**: 関係注釈・アフォーダンス・行動条件付き、**9-DoF 3D bbox（Omni3D 形式）**
6. **Code**: Text-Only Coding + Multimodal Coding（UI→HTML/CSS、画像→SVG、StackOverflow QA）
7. **Video**: Dense Caption Synthesis + Spatio-Temporal Grounding、Length-Adaptive Sampling
8. **STEM**: 100 万点グラウンディング + 200 万 VQA + 600 万図解キャプション + 6000 万 K-12 演習 + 1200 万 CoT
9. **Agent**: GUI（デスクトップ / モバイル / Web）+ Function Calling + Search

## 事後学習：3 段階 + Thinking with Images

| 段階 | 内容 |
| --- | --- |
| **SFT** | 約 120 万サンプル、2 フェーズ（32K → 256K）、non-thinking と thinking バリアントに分岐 |
| **Strong-to-Weak Distillation** | テキスト専用データで LLM バックボーン蒸留、Off-policy → On-policy（KL 最小化） |
| **Reinforcement Learning** | Reasoning RL + General RL、**SAPO アルゴリズム**、約 30K クエリ |

**Long-CoT Cold Start**: thinking モデル用、視覚言語と純粋テキストの 1:1 比率、**Multimodal Necessity Filtering**（視覚なしで解けるサンプルを除外）

**Thinking with Images**: 2 段階（Qwen2.5-VL-32B で 10K → 蒸留して 120K）、3 報酬（Answer Accuracy + Multi-Turn Reasoning + Tool-Calling）

## 主要ベンチマーク結果（ハイライト）

### 235B-A22B vs 商用フラッグシップ（表 2）

| ベンチ | 235B Thinking | 235B Instruct | Gemini 2.5-Pro Thinking | GPT-5 high | Claude Opus 4.1 |
| --- | --- | --- | --- | --- | --- |
| MMMU | 80.6 | 78.7 | 81.7 | **84.2** | 78.4 |
| **MathVista** | **85.8** | 84.9 | 82.7 | 81.3 | 75.5 |
| **MathVision** | **74.6** | 66.5 | 73.3 | 70.9 | 64.3 |
| **MathVerse_mini** | **85.0** | 72.5 | 82.9 | 84.1 | 70.6 |
| MMBench-EN | 88.8 | **89.3** | 90.1 | 83.8 | 79.4 |
| MMStar | **78.7** | 78.4 | 77.5 | 76.4 | 72.1 |
| HallusionBench | **66.7** | 63.2 | 63.7 | 65.7 | 60.4 |
| MIA-Bench | **92.7** | 91.3 | 92.3 | 92.4 | 91.2 |
| DocVQA | 96.5 | **97.1** | 92.6 | 91.5 | 92.5 |
| InfoVQA | 89.5 | **89.2** | 84.2 | 79.0 | 69.4 |
| **OCRBench** | 875 | **920** | 866 | 810 | 764 |
| **OCRBench_v2 en** | 66.8 | **67.1** | 54.3 | 50.4 | 48.4 |
| **OCRBench_v2 zh** | **63.5** | 61.8 | 48.5 | 37.7 | 43.7 |
| **OmniDocBench en** ↓ | 0.155 | **0.143** | 0.206 | 0.356 | 0.194 |
| **MMLongBench-Doc** | 56.2 | **57.0** | 55.6 | 51.5 | 54.5 |
| RefCOCO-avg | **92.1** | 91.9 | 74.6 | 66.8 | - |
| **CountBench** | 93.7 | 93.0 | 91.0 | 91.7 | 93.1 |
| **ODinW-13** | 43.2 | **48.6** | 33.7 | - | - |
| **VSI-Bench** | **60.0** | 62.7 | - | - | - |
| **EmbSpatialBench** | **84.3** | 83.1 | 79.1 | 82.9 | 69.2 |
| **MUIRBENCH** | **80.1** | 73.0 | 77.2 | 77.5 | 64.1 |
| **Charades-STA mIoU** | 63.5 | **64.8** | - | - | - |
| **V\*** | 85.9 | **93.7+**(tool) | 83.8 | 72.8 | 34.8 |
| **ScreenSpot Pro** | **61.8** | 62.0 | - | - | - |
| **AndroidWorld** | 62.0 | 63.7 | - | - | - |
| **OSWorld** | **38.1** | 31.6 | - | - | 44.4 |

**Qwen2.5-VL からの主な進化**:
- ScreenSpot Pro: 43.6 → **62.0**（+18.4）
- OSWorld: 8.83 → **38.1**（4.3× 飛躍）
- AndroidWorld: 35% → **63.7%**（+28.7）
- Charades-STA mIoU: 50.9 → **63.5**（+12.6）
- ODinW-13: 43.1 → **48.6**（+5.5）

### 純粋テキスト性能（表 5、Instruct）

| ベンチ | Qwen3-VL-235B-A22B Instruct | Qwen3-235B-A22B-Instruct-2507（純粋 LLM） |
| --- | --- | --- |
| **AIME-25** | **74.7** | 70.3 |
| **HMMT-25** | **57.4** | 55.4 |
| **LiveCodeBench v6** | **54.3** | 51.8 |

「マルチモーダル化で言語が強くなる」現象を継続実証（[[entities/internvl-3|InternVL 3]] / [[entities/qwen2-5-vl|Qwen2.5-VL]] の発見の継続）

### Needle-in-a-Haystack（図 3）

- **30 分動画（256K トークン）完全 100% 精度**
- **YaRN 拡張で 1M トークン（2 時間動画）まで 99.5% 精度**

### アブレーション

#### Vision Encoder（表 11）

| | OmniBench (CLIP) | OmniBench (VLM) |
| --- | --- | --- |
| SigLIP-2 | 36.9 | 50.1 |
| **Qwen3-ViT** | **45.5** | **53.0** |

#### DeepStack（表 12）

| | AVG | OCRB | InfoVQA | DocVQA |
| --- | --- | --- | --- | --- |
| Baseline | 74.7 | 81.0 | 71.9 | 89.5 |
| **DeepStack** | **76.0** | **83.6** | **74.2** | **91.1** |

## インフラ

- **Alibaba Cloud PAI-Lingjun**、最大 **10,000 GPU 規模**
- **Megatron-LM** + 5D 並列（TP + PP + CP + EP + ZeRO-1 DP）
- 推論: **vLLM**（PagedAttention）or **SGLang**（構造化生成）

## 限界

- **MMMU で GPT-5 high (84.2) に -3.6**（80.6）
- **We-Math で Gemini 2.5-Pro Thinking (80.6) に -5.8**
- **Video-MME / MLVU で Gemini 2.5-Pro / GPT-5 にやや劣る**
- **OSWorld で Claude Opus 4 (44.4) に -6.3**
- **訓練データ非公開**（cf. InternVL 3 は完全公開）
- **MoE 235B モデルの推論コスト**: 活性 22B でも複数 GPU 必須
- **6 サイズ × 2 バリアント = 12 モデル**の管理コスト

## 系譜・後継

- **[[entities/qwen-vl|Qwen-VL]]**（2023.08, 9.6B）← 初代、`<box>`/`<ref>` + 256 トークン固定 + 448² 固定
- **[[entities/qwen2-vl|Qwen2-VL]]**（2024.09, 2B/7B/72B）← Naive Dynamic Resolution + M-RoPE
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（2025.02, 3B/7B/72B）← Window Attention + MRoPE absolute time + 4.1T + QwenVL HTML + GUI Agent
- **Qwen3-VL**（2025.11, 2B/4B/8B/32B + 30B-A3B/235B-A22B）← 本ページ、Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + 256K + thinking モード

## 関連ページ

- [[sources/qwen3-vl]] — 原典の要約
- [[translations/qwen3-vl]] — 原典の翻訳
- [[entities/qwen2-5-vl]] — 前世代
- [[entities/qwen2-vl]] — 第 2 世代
- [[entities/qwen-vl]] — 初代
- [[entities/siglip]] — Vision Encoder の初期化
- [[entities/qwen3-5-omni]] — 並列発展する Qwen ファミリーの **Omni 統合路線**（テキスト + 画像 + 音声 + 動画 + 音声生成、Thinker-Talker + Hybrid MoE + AuT + ARIA、リアルタイム音声対話エージェント）
- [[entities/internvl-3]] — 同時期の Native Multimodal Pre-Training InternVL 系
- [[entities/internvl-3-5]] — Cascade RL + MoE 241B-A28B の InternVL 系（同サイズスケール）
- [[entities/grounding-dino]] — オープン語彙検出専門モデル（Qwen3-VL のグラウンディング合成に活用）
- [[entities/sam]] / [[entities/sam-2]] — グラウンディング・データ合成
- [[concepts/rotary-position-embeddings]] — Interleaved MRoPE の前提
- [[concepts/foundation-model]]
- [[concepts/weakly-supervised-pretraining]]
- [[concepts/zero-shot-transfer]]
- [[concepts/alignment-tuning]]
