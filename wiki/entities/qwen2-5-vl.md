---
type: entity
entity_kind: model
aliases: [Qwen2.5-VL, Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen2.5-VL-72B, Qwen25VL, qwen2-5-vl, Qwen2.5VL]
related: [[entities/qwen2-vl]], [[entities/qwen-vl]], [[entities/internvl-2-5]], [[entities/internvl-3]], [[concepts/rotary-position-embeddings]], [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]]
sources: [[sources/qwen2-5-vl]]
updated: 2026-05-30
---

# Qwen2.5-VL / Qwen2.5-VL-3B / Qwen2.5-VL-7B / Qwen2.5-VL-72B

**Qwen2.5-VL シリーズ**は Alibaba Group の Qwen Team が開発する MLLM 系譜の第 3 世代（2025 年 2 月 19 日公開、arXiv:2502.13923）。**Window Attention を組み込んだ ViT をゼロから学習** + **MRoPE を絶対時間に整合** + **動的 FPS サンプリング** + **QwenVL HTML フォーマット** + **4.1T トークンの事前学習**で、視覚エージェントとしての能力を商用級に引き上げた MLLM。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2025 年 2 月 19 日（arXiv:2502.13923 v1）、2025 年 3 月 5 日報告 |
| 開発元 | Alibaba Group / Qwen Team |
| サイズ展開 | **3B / 7B / 72B** の 3 サイズ（Qwen2-VL の 2B → 3B に変化） |
| ライセンス | Apache 2.0（3B/7B）/ Qwen License（72B） |
| リポジトリ | https://github.com/QwenLM/Qwen2.5-VL |
| HF Hub | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct` |
| デモ | https://chat.qwenlm.ai |
| 言語サポート | 中国語・英語・欧州主要言語・日本語・韓国語・ベトナム語・アラビア語など多言語強化 |
| 文脈長 | 32K トークン（Stage 3 で 8K → 32K に拡張） |

## モデル詳細（表 1）

| Configuration | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
| --- | --- | --- | --- |
| **Vision Transformer (ViT)** | | | |
| Hidden Size | 1280 | 1280 | 1280 |
| # Layers | 32 | 32 | 32 |
| # Heads | 16 | 16 | 16 |
| Intermediate Size | 3456 | 3456 | 3456 |
| Patch Size | 14 | 14 | 14 |
| **Window Size** | **112** | **112** | **112** |
| **Full Attention Block Indexes** | **{7, 15, 23, 31}** | **{7, 15, 23, 31}** | **{7, 15, 23, 31}** |
| **Vision-Language Merger** | | | |
| In Channel | 1280 | 1280 | 1280 |
| Out Channel | 2048 | 3584 | 8192 |
| **Large Language Model (LLM)** | | | |
| Hidden Size | 2048 | 3584 | 8192 |
| # Layers | 36 | 28 | 80 |
| # KV Heads | 2 | 4 | 8 |
| Head Size | 128 | 128 | 128 |
| Intermediate Size | 4864 | 18944 | 29568 |
| Embedding Tying | ✓ | ✗ | ✗ |
| Vocabulary Size | 151646 | 151646 | 151646 |
| # Trained Tokens | 4.1T | 4.1T | 4.1T |

**重要**: 全サイズで **同一の ViT 構造を共有**（[[entities/qwen2-vl|Qwen2-VL]] と同じ思想）、ただし **DFN 初期化から「ゼロから学習」に変更**。最小モデルは Qwen2-VL の 2B から 3B に大型化（Qwen2.5 ファミリーに 2B が存在しないため）。

## アーキテクチャの 4 つの主要技術

### 1. Fast and Efficient Vision Encoder

- **ViT 32 層中、4 層のみ完全自己注意**（インデックス {7, 15, 23, 31}）、残り 28 層は **112×112 ウィンドウ・アテンション**
- 計算量がパッチ数に **線形にスケール**（従来は 2 次）
- 112×112 未満の領域は **パディングなしに処理**して元の解像度を保持
- LLM 系統一構造: **RMSNorm + SwiGLU**
- **ViT をゼロから学習**（DataComp + 社内データで初期化、CLIP 事前学習 → 視覚言語整合 → エンドツーエンド微調整）
- 動画は **連続 2 フレームをグループ化（3D パッチ分割）**でトークン数半減

### 2. Native Dynamic Resolution and Frame Rate

- 空間: バウンディング・ボックス・点を **入力画像の実際の次元で直接表現**（座標正規化を捨てる）→ スケール情報を本質的に学習
- 時間: **動的 FPS サンプリング**で学習時に異なる FPS（5/10/30 fps 等）を均等混在 → 推論時に様々な FPS の動画に頑健
- 半時間超の動画には **複数フレーム・キャプション**を targeted 合成で構築
- タイムスタンプを **秒形式 + hmsf（時-分-秒-フレーム）形式**で扱う

### 3. MRoPE Aligned to Absolute Time（最重要の新規性）

- **Qwen2-VL の MRoPE**: temporal ID をフレーム番号に結びつけていた → 異なる FPS の動画で「同じ秒数」を表現できない
- **Qwen2.5-VL の改善**: temporal ID 間の間隔を **絶対時間（秒数）に揃える**
- 効果: 異なる FPS の動画で **一貫した時間整合を学習**、追加テキスト・タイムスタンプ注入や追加ヘッド不要
- これが **動画グラウンディング Charades-STA mIoU 50.9（GPT-4o 35.7 を +15.2）** と LVBench / MLVU SOTA の決め手

### 4. Pre-Training Data: 1.2T → 4.1T tokens

- **Interleaved image-text**: 4 基準で採点（テキスト品質 / 関連性 / 相補性 / 情報密度バランス）
- **Grounding with absolute coordinates**: 10,000+ カテゴリ、存在しないカテゴリも合成、PixMo + 自動化パイプラインで点グラウンディング
- **Document Omni-Parsing Data**: QwenVL HTML フォーマットで統一表現
- **OCR**: 多言語（仏・独・伊・西・葡・露・日・韓・越・アラ）、チャート 100 万合成、表 600 万 E2E 処理
- **Video**: 動的 FPS、長動画キャプション、秒形式 + hmsf
- **Agent**: モバイル・Web・デスクトップ・スクリーンショット + UI 要素グラウンディング + 共有行動空間関数呼び出し + 各ステップ推論

## QwenVL HTML フォーマット（独自貢献）

```html
<p data-bbox="x1 y1 x2 y2">paragraph</p>
<table data-bbox="x1 y1 x2 y2">table</table>
<div class="chart" data-bbox="..."><img/><table>chart</table></div>
<div class="formula" data-bbox="..."><img/><div>formula</div></div>
<div class="music sheet" format="abc notation" data-bbox="..."><img/><div>music</div></div>
<div class="chemical formula" format="smile" data-bbox="..."><img/><div>chemical</div></div>
```

**レイアウト・テキスト・表・チャート・数式・楽譜・化学式を 1 つの HTML 文書として表現**。楽譜は **ABC notation**、化学式は **SMILES** を採用。OCRBench_v2 で英語 +9.6 / 中国語 +20.6（vs Gemini-1.5-Pro）の決定打。

## 学習パイプライン

### 3 段階事前学習（表 2、累積 4.1T トークン）

| Stage | データ | トークン | Seq | 学習対象 |
| --- | --- | --- | --- | --- |
| **Visual Pre-Training** | Image Caption / Knowledge / OCR | 1.5T | 8192 | ViT のみ |
| **Multimodal Pre-Training** | + Pure text / Interleaved / VQA / Video / Grounding / Agent | 2T | 8192 | ViT & LLM |
| **Long-Context Pre-Training** | + Long Video / Long Agent / Long Document | 0.6T | **32768** | ViT & LLM |

### 2 段階事後学習（ViT 凍結）

- **SFT**: 200 万エントリ（純粋テキスト 50% / マルチモーダル 50%）、ChatML 形式
- **データ・フィルタリング**: 
  - Stage 1: 専用分類器 *Qwen2-VL-Instag* で **8 主要ドメイン × 30 サブカテゴリ**に階層分類
  - Stage 2: ルールベース + モデルベース（複雑性 / 関連性 / 完全性 / 視覚情報利用検証）
- **棄却サンプリング**: ground truth 一致のみ保持、CoT 推論を重視
- **DPO**: 画像-テキスト + 純粋テキストのみ、各サンプル 1 回処理

## 主要ベンチマーク結果（ハイライト）

### 一般 VQA / 大学レベル推論 / 数学（表 3）

| ベンチマーク | 3B | 7B | 72B | GPT-4o | Claude-3.5 | Qwen2-VL-72B |
| --- | --- | --- | --- | --- | --- | --- |
| **MMMU val** | 53.1 | 58.6 | **70.2** | 69.1 | 68.3 | 64.5 |
| MMMU-Pro | 31.56 | 38.3 | 51.1 | **51.9** | 51.5 | 46.2 |
| **MathVista mini** | 62.3 | 68.2 | **74.8** | 63.8 | 67.7 | 70.5 |
| **MATH-Vision** | 21.2 | 25.1 | **38.1** | 30.4 | - | 25.9 |
| **MathVerse mini** | 47.6 | 49.2 | **57.6** | 50.2 | - | - |
| **MMBench-EN test** | 79.1 | 83.5 | **88.6** | 83.4 | 82.6 | 86.9 |
| **MMBench-V1.1-EN** | 77.4 | 82.6 | **88.4** | 83.1 | 80.9 | 86.1 |
| **MMStar** | 55.9 | 63.9 | **70.8** | 64.7 | 65.1 | 68.3 |
| **MuirBench** | 47.7 | 59.6 | **70.7** | 68.0 | - | - |
| BLINK | 47.6 | 56.4 | 64.4 | **68.0** | - | - |
| **CRPE relation** | 73.6 | 76.4 | **79.2** | 76.6 | - | - |
| **MMVet turbo** | 61.8 | 67.1 | **76.2** | 69.1 | 70.1 | 74.0 |
| **MME-RealWorld en** | 53.1 | 57.4 | **63.2** | 45.2 | 51.6 | - |
| MME sum | 2157 | 2347 | 2448 | 2328 | 1920 | 2483 |

**ハイライト**: MMMU で初の 70.2（Qwen2-VL から +5.7、GPT-4o 超え）、MathVista +4.3、MathVision +12.2。

### 純粋テキスト（表 4、72B 比較）

| ベンチマーク | Qwen2.5-VL-72B | Qwen2.5-72B | Llama-3.1-405B |
| --- | --- | --- | --- |
| MMLU-Pro | 71.2 | 71.1 | **73.3** |
| MMLU-redux | 85.9 | **86.8** | 86.2 |
| **LiveBench-0831** | **57.0** | 52.3 | 53.2 |
| GPQA | 49.0 | 49.0 | **51.1** |
| MATH | 83.0 | **83.1** | 73.8 |
| GSM8K | 95.3 | 95.8 | **96.8** |
| HumanEval | 87.8 | 86.6 | **89.0** |
| **MultiPL-E** | **79.5** | 75.1 | 73.5 |
| **IFEval** | **86.3** | 84.1 | 86.0 |

**マルチモーダル化による純粋テキスト性能の劣化なし**、いくつかのベンチで Qwen2.5-72B を上回る。

### 文書理解 / OCR（表 5）

| ベンチマーク | 72B | GPT-4o | Gemini 1.5-Pro | Claude-3.5 | InternVL2.5-78B |
| --- | --- | --- | --- | --- | --- |
| **CC-OCR** | **79.8** | 66.9 | 73.0 | 62.5 | 64.7 |
| **OmniDocBench en** ↓ | **0.226** | 0.265 | 0.230 | 0.330 | 0.275 |
| AI2D | 88.7 | 84.6 | 88.4 | 81.2 | **89.1** |
| **DocVQA test** | **96.4** | 91.1 | 93.1 | 95.2 | 95.1 |
| **InfoVQA test** | **87.3** | 80.7 | 81.0 | 74.3 | 84.1 |
| ChartQA | 89.5 | 86.7 | 87.2 | **90.8** | 88.3 |
| **CharXiv DQ** | **87.4** | 84.5 | 72.0 | 84.3 | 82.3 |
| **OCRBench** | **885** | 736 | 754 | 788 | 854 |
| **OCRBench_v2 en/zh** | **61.5/63.7** | 46.5/32.2 | 51.9/43.1 | 45.2/39.6 | 49.8/52.1 |

**OCRBench_v2 で Gemini 1.5-Pro を英語 +9.6 / 中国語 +20.6 ポイントで圧倒**。

### グラウンディング（表 6, 7）

| ベンチマーク | 72B | InternVL2.5-78B | Grounding DINO | GPT-4o |
| --- | --- | --- | --- | --- |
| RefCOCO val | 92.7 | **93.7** | 90.6 | - |
| RefCOCO+ val | 88.9 | **90.4** | 88.2 | - |
| RefCOCOg val | 89.9 | **92.7** | 86.1 | - |
| **ODinW** | **43.1** | 31.7 | 55.0 (専門) | - |
| **PointGrounding** | 67.5 | - | - | - |
| **CountBench** | **93.6** | 72.1 | - | 87.9 |

### 動画理解（表 8）

| ベンチマーク | 72B | Gemini 1.5-Pro | GPT-4o |
| --- | --- | --- | --- |
| Video-MME w/o sub. | 73.3 | **75.0** | 71.9 |
| Video-MME w sub. | 79.1 | **81.3** | 77.2 |
| **MVBench** | **70.4** | 60.5 | 64.6 |
| **MMBench-Video** | **2.02** | 1.30 | 1.63 |
| **LVBench** | **47.3** | 33.1 | 30.8 |
| **EgoSchema test** | **76.2** | 71.2 | 72.2 |
| **MLVU M-Avg** | **74.6** | - | 64.6 |
| **TempCompass Avg** | **74.8** | 67.1 | 73.8 |
| **Charades-STA mIoU** | **50.9** | - | 35.7 |

**長動画（LVBench / MLVU）で GPT-4o を圧倒、Charades-STA で時間グラウンディング SOTA**。

### GUI エージェント（表 9）

| ベンチマーク | 72B | Aguvis-72B | Qwen2-VL-72B | GPT-4o | Gemini 2.0 | Claude |
| --- | --- | --- | --- | --- | --- | --- |
| ScreenSpot | 87.1 | **89.2** | - | 18.1 | 84.0 | 83.0 |
| **ScreenSpot Pro** | **43.6** | 23.6 | **1.6** | - | - | 17.1 |
| **Android Control High EM** | **67.36** | 66.4 | 59.1 | 20.8 | 28.5 | 12.5 |
| **Android Control Low EM** | **93.7** | 84.4 | 59.2 | 19.4 | 60.2 | 19.4 |
| **AndroidWorld SR** | **35%** | 26.1% | 6% (SoM) | 34.5% (SoM) | 26% (SoM) | 27.9% |
| **MobileMiniWob++ SR** | **68%** | 66% | 50% (SoM) | 61% | 42% (SoM) | 61% (SoM) |
| OSWorld | 8.83 | 10.26 | 2.42 | 5.03 | 4.70 | **14.90** |

**ScreenSpot Pro で Qwen2-VL の 1.6% から 43.6% へ +42 ポイント（27×）の飛躍**。GUI エージェントを商用級に。

## アブレーション要点

- **Window Attention** 採用で ViT 計算量が線形化、高解像度画像で効率改善
- **MRoPE absolute time** で動画時間グラウンディング Charades-STA +15.2（vs GPT-4o）
- **QwenVL HTML format** で OCRBench_v2 zh +20.6（vs Gemini）
- **3 段階事前学習**（1.5T + 2T + 0.6T = 4.1T）で系列長段階的拡張（8K → 8K → 32K）

## 限界

- **MMMU-Pro で GPT-4o に -0.8**（51.1 vs 51.9）
- **Video-MME で Gemini 1.5-Pro に -1.7/-2.2**
- **OSWorld（デスクトップ）で Claude 14.90 に -6.07**
- 動画フレーム上限 768 / トークン上限 24,576
- ViT をゼロから学習するため、既存 VFM 蒸留より計算コスト大
- 訓練データ完全公開なし（cf. [[entities/internvl-3|InternVL 3]] は公開）
- 「Native Multimodal Pre-Training」未採用（依然として「LLM 事前学習 → MLLM 適応」）

## 系譜・後継

- **[[entities/qwen-vl|Qwen-VL]]**（2023.08, 9.6B）← 初代、固定 448² + 256 トークン + `<box>`/`<ref>`
- **[[entities/qwen2-vl|Qwen2-VL]]**（2024.09, 2B/7B/72B）← Naive Dynamic Resolution + M-RoPE + DFN ViT
- **Qwen2.5-VL**（2025.02, 3B/7B/72B）← 本ページ、Window Attention + MRoPE absolute time + 4.1T + QwenVL HTML + GUI Agent
- **[[entities/qwen3-vl|Qwen3-VL]]**（2025.11, 2B/4B/8B/32B Dense + 30B-A3B/235B-A22B MoE × Instruct/Thinking = 12 モデル）: Interleaved MRoPE + DeepStack + テキスト・ベース時間整合（MRoPE absolute time を捨て `<3.0 seconds>` 明示的タイムスタンプへ）+ SigLIP-2 継続学習 + 256K ネイティブ → 1M YaRN 外挿 + Strong-to-Weak Distillation + SAPO RL + Thinking with Images。MathVista 85.8 / OCRBench_v2 zh 63.5 SOTA / Charades-STA 64.8 / ScreenSpot Pro 62.0 / OSWorld 38.1（本モデル 8.83 から **4.3× 飛躍**）/ AndroidWorld 63.7、Apache 2.0、Qwen ファミリーが商用最先端と完全互角に
- **[[entities/qwen3-5-omni|Qwen3.5-Omni]]**（2026, Plus/Flash 2 バリアント）: 並列発展する Omni 統合路線、Thinker-Talker + Hybrid MoE + AuT + ARIA + 256K + Audio-Visual Vibe Coding、テキスト + 画像 + 音声 + 動画の完全 omnimodal 統合、API のみ公開

## 関連ページ

- [[sources/qwen2-5-vl]] — 原典の要約
- [[translations/qwen2-5-vl]] — 原典の翻訳
- [[entities/qwen2-vl]] — 前世代
- [[entities/qwen-vl]] — 初代
- [[entities/qwen3-vl]] — 直接の後継、Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking モード
- [[entities/internvl-2-5]] — 同時期 InternVL 系（Progressive Scaling）
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系
- [[entities/internvl-3-5]] — Cascade RL + MoE の InternVL 系最新版
- [[entities/grounding-dino]] — オープン語彙物体検出専門モデル
- [[entities/sam]] / [[entities/sam-2]] — グラウンディング・データ合成に活用
- [[concepts/rotary-position-embeddings]] — MRoPE absolute time の前提
- [[concepts/foundation-model]]
- [[concepts/weakly-supervised-pretraining]]
- [[concepts/zero-shot-transfer]]
- [[concepts/alignment-tuning]]
