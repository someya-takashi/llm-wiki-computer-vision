---
type: source
source_path: raw/papers/DeepSeek-OCR_ Contexts Optical Compression.md
source_kind: paper
title: "DeepSeek-OCR: Contexts Optical Compression"
authors: [Haoran Wei, Yaofeng Sun, Yukun Li]
year: 2025
venue: arXiv:2510.18234
ingested: 2026-05-31
tags: [deepseek, deepseek-ocr, ocr, optical-compression, mllm, vision-language, sam, clip, moe, memory-forgetting, omnidocbench, fox-benchmark, document-understanding]
translation: "[[translations/deepseek-ocr]]"
---

# DeepSeek-OCR — 「LLM 中心視点」から VLM を再定義し、視覚モダリティをテキスト圧縮媒体として使う新パラダイム

> 原典: [[translations/deepseek-ocr]] ・ `raw/papers/DeepSeek-OCR_ Contexts Optical Compression.md`
> 著者・年・会議: Haoran Wei, Yaofeng Sun, Yukun Li（DeepSeek-AI）, 2025 Oct, arXiv:2510.18234

## 一言まとめ

**DeepSeek-AI 初の wiki 登録モデル**。**「視覚モダリティをテキスト情報の効率的圧縮媒体として活用する」**という新パラダイムを提唱し、**LLM 中心視点で VLM を再定義**。**DeepEncoder**（SAM-base 80M + 2 層 ConvNet で 16× ダウンサンプリング + CLIP-large 300M = 380M）+ **DeepSeek-3B-MoE-A570M デコーダ**（64 routed expert、6 活性化）の 2 段構成で、**5 つの解像度モード**（Tiny 64 トークン / Small 100 / Base 256 / Large 400 / Gundam 動的）をサポート。**Fox ベンチマークで 10× 圧縮時 97% 精度、20× 圧縮時でも 60% 精度**を達成。**OmniDocBench で 100 視覚トークンで GOT-OCR2.0（256 トークン）を凌駕、800 トークン未満で MinerU2.0（6790 トークン）を上回り SOTA**（編集距離 0.083）。**1 台の A100-40G で 1 日 20 万ページ以上**の生成能力。**「記憶忘却機構（Memory Forgetting Mechanism）」**として、古い文脈を低解像度画像にレンダリングして自然な忘却曲線を模倣する応用を提案。**Computer Vision wiki 内では、Qwen3-VL/InternVL 系列のような汎用 MLLM とは異なる「OCR/文書理解特化路線」の代表**として、また **SAM ([[entities/sam]]) と CLIP ([[entities/clip]]) を直列に組み合わせる新しい視覚エンコーダ設計**として位置付けられる。**GitHub オープンソース公開**。

## 背景と問題意識

2024-2025 年の MLLM/VLM 競争は **任意解像度処理** に向かって進化してきた（[[questions/vit-dynamic-resolution-evolution]] 参照）。しかし、**現代の主要 VLM の視覚エンコーダは依然として深刻な問題**を抱えている：

1. **Vary / DeepSeekVL 系（VITDet + VIT 二重前処理）**: パイプライン並列非対応、極端な解像度非対応、2 つの前処理プロセス、展開困難
2. **InternVL 系（タイル分割、384² パッチ多数）**: 通常 15 タイル超、低 native 解像度、視覚トークン過多、小グローバル・ビュー
3. **Qwen2/2.5/3-VL 系（NaViT 風 ViT を 2D-RoPE で任意解像度化）**: 視覚トークン過多、大活性化メモリ、長系列必要、推論遅い

これら全ての問題は **「視覚トークン数の爆発」**に起因する。1 ページの文書を扱うのに **6000+ トークン**（MinerU2.0、Qwen2.5-VL）必要というのは、**LLM の文脈長を視覚側が食い尽くす**ことを意味する。

### DeepSeek の哲学的転換

DeepSeek-OCR は **「LLM 中心の視点」**から VLM を再定式化する：
- 視覚エンコーダは「画像を理解する」のではなく、**「テキスト情報を効率的に圧縮する媒体」**
- 単一の文書画像は、同等のデジタル・テキストよりも **実質的に少ないトークンで情報を符号化できる**
- **OCR が理想的な検証台**：視覚↔テキストの **圧縮-解凍マッピング** + **定量評価可能**

これは Qwen / InternVL / Gemma 系の「**視覚は視覚として扱う**」哲学とは正反対の **「視覚はテキストの効率的符号化方式である**」という主張。

## 提案手法 / 主張

### DeepEncoder アーキテクチャ：SAM + 16× ConvNet 圧縮 + CLIP の直列構成

**5 つの要件**を同時に満たす視覚エンコーダ設計：
1. **高解像度処理能力**
2. **低活性化メモリ**
3. **最小視覚トークン数**
4. **複数解像度サポート**
5. **適度なパラメータ数**

**コンポーネント 1: Visual Perception**（視覚知覚）
- **[[entities/sam|SAM]]-base（80M パラメータ、patch size 16）**
- **Window Attention** 支配 → 局所ウィンドウ化で高解像度入力の活性化メモリを抑制
- 1024×1024 入力 → 4096 パッチ・トークン

**コンポーネント 2: Compression Bridge**（圧縮ブリッジ）
- **2 層畳み込みモジュール**
- 各層 kernel=3, stride=2, padding=1
- チャンネル数 256→1024
- **16× ダウンサンプリング**（4096 → 256 トークン）

**コンポーネント 3: Visual Knowledge**（視覚知識）
- **[[entities/clip|CLIP]]-large（300M パラメータ）**
- **Dense Global Attention** → 意味抽出
- 圧縮された 256 トークンから視覚意味を抽出

**全体**: 約 380M パラメータ

### 5 つの解像度モード

| Mode | Resolution | Tokens | Process |
| --- | --- | --- | --- |
| **Tiny** | 512×512 | **64** | Resize |
| **Small** | 640×640 | **100** | Resize |
| **Base** | 1024×1024 | **256** | Padding |
| **Large** | 1280×1280 | **400** | Padding |
| **Gundam** | n×640² + 1024² | **n×100+256**（典型 n=2-9） | 動的 |
| **Gundam-M** | n×1024² + 1280² | （継続訓練版） | 動的 |

**パディング・モードの Valid Token 計算**:
$$N_{\text{valid}} = \lceil N_{\text{actual}} \times [1 - \frac{\max(w,h)-\min(w,h)}{\max(w,h)}] \rceil$$
アスペクト比を保持するパディングを考慮。

### DeepSeek-3B-MoE-A570M デコーダ

**MoE 構造**:
- **64 個の routed expert**、**推論ごとに 6 個活性化**
- **2 個の shared expert**
- **推論時の活性パラメータ: 約 570M**（A570M）
- 非常にコンパクトな MoE 設計

**機能**: 圧縮された視覚トークン（n 個）をテキスト・トークン列（N 個、n ≤ N）にマッピング

### 訓練データ構成（合計 70% OCR 1.0 + 20% 一般視覚 + 10% テキスト）

**OCR 1.0**（30M+ ページ）:
- 100 言語の PDF 30M ページ（中国語 25M + 他言語 5M）
- 粗注釈（fitz）+ 精細注釈（PP-DocLayout + MinerU/GOT-OCR2.0）
- 少数言語 600K（GOT-OCR2.0 モデル・フライホイール）
- Word 文書 3M（HTML 表/数式）
- シーン OCR 中英各 10M（LAION/Wukong）

**OCR 2.0**（合成データ）:
- **チャート解析**: 10M 合成画像（pyecharts/matplotlib）→ HTML 表
- **化学式**: 5M SMILES + RDKit レンダリング
- **平面幾何**: 1M（Slow Perception フレームワーク）

**一般視覚 20%**: DeepSeek-VL2 風（キャプショニング + 検出 + グラウンディング）

**テキスト 10%**: 社内事前学習データ、系列長 8192

### 訓練パイプライン

**Stage 1: DeepEncoder 訓練**
- OCR 1.0 + 2.0 + 100M LAION
- バッチサイズ 1280、エポック 2、AdamW、lr 5e-5、cosine annealing、系列長 4096

**Stage 2: 全訓練**
- **HAI-LLM プラットフォーム**、**20 ノード × 8× A100-40G = 160 GPU**
- **パイプライン並列（PP=4）+ データ並列（DP=40）**
- PP0: SAM + 圧縮器（凍結）/ PP1: CLIP（凍結解除）/ PP2-PP3: DeepSeek-3B-MoE 各 6 層
- グローバル・バッチサイズ 640、AdamW、lr 3e-5、系列長 8192
- 速度: **テキスト専用 90B トークン/日、マルチモーダル 70B トークン/日**

## 実験結果と知見

### Fox ベンチマークでの視覚-テキスト圧縮研究（最重要結果）

**Tiny モード（64 視覚トークン）**:
| Text Tokens | Precision | Compression |
| --- | --- | --- |
| 600-700 | **96.5%** | 10.5× |
| 800-900 | 83.8% | 13.2× |
| 1000-1100 | 79.3% | 16.5× |
| 1200-1300 | **59.1%** | **19.7×** |

**Small モード（100 視覚トークン）**:
| Text Tokens | Precision | Compression |
| --- | --- | --- |
| 600-700 | **98.5%** | 6.7× |
| 800-900 | **96.8%** | 8.5× |
| 900-1000 | **96.8%** | 9.7× |
| 1100-1200 | 89.8% | 11.3× |
| 1200-1300 | 87.1% | 12.6× |

**主要発見**:
- **10× 圧縮で 97% 以上の精度** → 近似ロスレス
- **20× 圧縮でも 60% 精度** → 大幅な圧縮でも実用可能

### OmniDocBench 比較（編集距離↓、低いほど良い）

| Model | Vision Tokens | Overall Edit Distance |
| --- | --- | --- |
| GOT-OCR2.0 | 256 | 0.287 |
| Qwen2.5-VL-7B | 3949 | 0.316 |
| OCRFlux-3B | 3949 | 0.238 |
| Qwen2.5-VL-72B | 3949 | 0.214 |
| InternVL3-78B | 6790 | 0.218 |
| MinerU2.0 | 6790 | 0.133 |
| dots.ocr (200dpi) | 5545 | 0.125 |
| **DeepSeek-OCR (Small)** | **100** | **0.205**（GOT-OCR2.0 超え） |
| **DeepSeek-OCR (Base)** | **256 (182*)** | **0.156** |
| **DeepSeek-OCR (Large)** | **400 (285*)** | **0.117** |
| **DeepSeek-OCR (Gundam)** | **795** | **0.083 SOTA** |
| **DeepSeek-OCR (Gundam-M)** | **1853** | **0.085 SOTA** |

**衝撃的結果**:
- **100 視覚トークン**で **GOT-OCR2.0（256 トークン）を上回る**（0.205 vs 0.287）
- **800 トークン未満**で **MinerU2.0（6790 トークン）を凌駕**（0.083 vs 0.133）
- **8.5× 効率向上**（MinerU2.0 比、6790÷795=8.5×）

### 文書タイプ別性能（Base 256 トークン）

| Category | Edit Distance | 必要モード |
| --- | --- | --- |
| **Financial Reports** | **0.027** | Tiny 64 でも十分（卓越） |
| **Books** | **0.037** | 100 で優秀 |
| **Slides** | 0.08 | 64 トークン十分 |
| Exam Papers | 0.13 | 100 で良好 |
| **Newspapers** | 0.645 | Gundam/Gundam-M 必須 |

### 実用的価値

**スループット**:
- 1 台 A100-40G: **200K+ ページ/日**
- 20 ノード（160 GPU）: **3300 万ページ/日**

これにより **LLM/VLM の事前学習用合成データを本番グレードの速度で生成可能**。

### 質的能力

- **深層解析（Deep Parsing）**: チャート → HTML 表、化学式 → SMILES、幾何 → 線分座標
- **多言語**: 約 100 言語サポート（アラビア語、シンハラ語等を実証）
- **一般視覚理解**: OCR 専門化 70% にもかかわらず、キャプショニング/検出/グラウンディング能力を保持

## 記憶忘却機構（Memory Forgetting Mechanism）— 核心的応用提案

DeepSeek-OCR の **最も革新的な応用提案**は **「人間の記憶減衰と視覚知覚劣化の並行性」**を活用した文脈圧縮：

**実装**:
1. 歴史的会話テキストを **画像にレンダリング**
2. 古い画像を **段階的に縮小**
3. **多レベル圧縮**:
   - 最近の文脈: 高解像度（全トークン）
   - 古い文脈: 削減解像度（少ないトークン）
   - 非常に古い文脈: 極度に圧縮（60% 精度を維持）

**応用**: 多ターン対話システムで **自然な忘却曲線**に従い、古いやり取りは少ないトークン消費。これは **LLM の文脈長制限への新しい解決策**となる可能性。

## 限界・批判的視点

- **OCR 特化のため汎用 MLLM ではない**: Qwen3-VL/InternVL 3.5 のような汎用視覚理解能力では劣る（MMMU 等のベンチマーク報告なし）
- **SFT 段階なし**: 一部タスクには補完プロンプトが必要
- **Newspapers で 0.645 編集距離**: Gundam/Gundam-M が必要、Base モードでは不十分
- **20× 圧縮で 60% 精度**: 「忘却機構」の根拠だが、**40% 損失は実用上どこまで許容できるか不明**
- **記憶忘却機構は概念実証のみ**: 実際の LLM 対話システムへの統合実験はなし
- **OmniDocBench に評価が偏重**: MMMU や視覚一般理解での評価が欠如
- **SAM + CLIP 直列構成の汎用性**: 文書 OCR には強いが、自然画像理解（物体検出、視覚推論）への適用性が不明
- **DeepSeek-VL2 系列との関係**: 同じ DeepSeek-AI による汎用 VLM 系列との分業が不明確
- **SFT/RLHF なし**: 命令調整版がなく、対話用途への直接利用は困難
- **Memory Forgetting Mechanism の検証なし**: 多ターン対話での実証実験は将来課題として残る

## CV 分野における意義

DeepSeek-OCR は、CV/MLLM 分野で以下の独自性を持つ：

1. **「LLM 中心の VLM 視点」という新パラダイム**: 視覚を「理解対象」ではなく「テキスト圧縮媒体」として再定義
2. **[[entities/sam|SAM]] + [[entities/clip|CLIP]] の直列構成という新エンコーダ設計**: 「窓注意で高解像度処理 → ConvNet で 16× 圧縮 → 大域注意で意味抽出」の 3 段アーキテクチャ
3. **視覚トークン数の劇的削減を SOTA で達成**: MinerU2.0 比 8.5× 効率、GOT-OCR2.0 比 2.56× 効率
4. **「Optical Compression」+ 「Memory Forgetting」の新概念**: LLM の長文脈問題に視覚モダリティで応える革新的方向性
5. **DeepSeek-AI 初の wiki 登録モデル**: DeepSeek-V3 / R1 は Qwen3-VL や Gemma 3 の比較対象として頻出していたが、DeepSeek-* の独立 ingest は初めて。**Qwen 系 / InternVL 系 / Google 系（Gemma）に続く第 4 の中国系 AI ラボ系譜**
6. **文書 OCR 特化の MLLM 分野**: Qwen3-VL/InternVL 3.5/Gemma 3 が「汎用 MLLM」なのに対し、DeepSeek-OCR は **「OCR/文書理解専門」**として明確な分業を提案
7. **GOT-OCR2.0 系の正統な後継**: 同じ著者 Haoran Wei による Vary → GOT-OCR2.0 → DeepSeek-OCR の系譜
8. **オープンソース公開**: GitHub と Hugging Face で公開、再現可能

Computer Vision wiki 内では：
- **[[entities/sam]] の主要応用**: SAM の window attention 設計が「文書 OCR」という新用途で活用
- **[[entities/clip]] の主要応用**: CLIP-large が「視覚知識成分」として組み込まれる
- **[[questions/vit-dynamic-resolution-evolution]] への新視点**: 「解像度を任意化」ではなく「視覚トークン数を最小化」というアプローチ
- **[[concepts/foundation-model]] における OCR 特化系の代表**

## 用語と略称

- **DeepSeek-OCR** = DeepSeek-AI が開発した OCR/文書理解特化型 MLLM
- **DeepSeek-AI** = 中国の AI 研究機関、DeepSeek-V3 / R1 / VL2 等の LLM/VLM 系列
- **DeepEncoder** = 本論文の独自視覚エンコーダ（SAM + 16× ConvNet + CLIP）
- **DeepSeek-3B-MoE-A570M** = 3B 総パラメータ、推論時 570M 活性の MoE デコーダ
- **MoE (Mixture-of-Experts)** = 混合エキスパート（routed + shared expert の組み合わせ）
- **A570M** = Activated 570M（活性化パラメータ数）
- **Optical Compression** = 光学的圧縮（テキストを画像化して視覚トークンで圧縮）
- **Optical 2D Mapping** = 光学的 2D マッピング（テキスト→画像→視覚トークンの変換）
- **Visual Perception Component** = SAM が担う「視覚知覚」成分（高解像度処理）
- **Visual Knowledge Component** = CLIP が担う「視覚知識」成分（意味抽出）
- **Compression Bridge** = 2 層 ConvNet による 16× ダウンサンプリング
- **Window Attention** = 局所ウィンドウ・アテンション（[[concepts/vision-transformer|ViT]] の高解像度対応技術）
- **Dense Global Attention** = 密大域アテンション（CLIP-large）
- **Tiny/Small/Base/Large/Gundam/Gundam-M** = DeepSeek-OCR の 6 解像度モード
- **Gundam** = 動的解像度モード（タイル + グローバル・ビュー）の独自呼称
- **Valid Token** = パディング・モードでアスペクト比を考慮した実効トークン数
- **Memory Forgetting Mechanism** = 記憶忘却機構（古い文脈を低解像度画像化）
- **OmniDocBench** = 文書解析評価ベンチマーク（PDF 27 ドメイン）
- **Fox Benchmark** = 文書理解評価ベンチマーク（600-1300 テキスト・トークン文書）
- **GOT-OCR2.0** = 同著者 Haoran Wei による前作（General OCR Theory）
- **MinerU2.0** = 文書解析パイプライン（6790 トークン平均）
- **dots.ocr** = Rednote の文書解析モデル
- **Marker / Mathpix / OCRFlux** = 商用/OSS 文書解析ツール
- **PaddleOCR** = Baidu のオープン OCR
- **SmolDocling** = ultra-compact VLM for document conversion
- **Nougat** = Neural OCR for academic documents (Meta)
- **Vary** = Wei et al. の前々作、視覚語彙拡大
- **Slow Perception** = 同著者の幾何図形知覚フレームワーク（Wei et al., 2024d）
- **HAI-LLM** = High-flyer / DeepSeek-AI の訓練インフラ
- **Edit Distance** = 編集距離（文字列の類似度指標、低いほど良い）
- **ANLS / CIDEr** = 文書 OCR の他の標準指標

## 関連ページ

- [[entities/deepseek-ocr]] — DeepSeek-OCR のエンティティ詳細
- [[translations/deepseek-ocr]] — 本論文の翻訳（ar5iv 本文ベース、Web Clipper の不完全 markdown を補完）
- [[entities/sam]] — DeepEncoder の Visual Perception 成分として活用される SAM-base
- [[entities/clip]] — DeepEncoder の Visual Knowledge 成分として活用される CLIP-large
- [[concepts/vision-transformer]] — SAM と CLIP の基盤
- [[concepts/foundation-model]] — OCR 特化 MLLM の代表
- [[concepts/weakly-supervised-pretraining]] — 30M ページの OCR 1.0 + 10M シーン OCR
- [[questions/vit-dynamic-resolution-evolution]] — 「視覚トークン数最小化」という新視点
- [[entities/qwen2-5-vl]] — 比較対象（3949 トークンで 0.214）
- [[entities/qwen3-vl]] — 同時期の汎用 MLLM 競合
- [[entities/internvl-3]] — 比較対象（InternVL3-78B 6790 トークンで 0.218）
- [[entities/internvl-3-5]] — 同時期の MoE MLLM 競合（Cascade RL + 241B-A28B）
- [[entities/gemma-3]] — 同時期の保守路線（タイル分割）との対比
