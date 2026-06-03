---
type: entity
entity_kind: model
aliases: [DeepSeek-OCR, DeepSeek OCR, DeepEncoder, DeepSeek3B-MoE-A570M, deepseek-ocr]
related: ["[[entities/sam]]", "[[entities/clip]]", "[[entities/qwen2-5-vl]]", "[[entities/qwen3-vl]]", "[[entities/internvl-3]]", "[[entities/internvl-3-5]]", "[[concepts/vision-transformer]]", "[[concepts/foundation-model]]"]
sources: ["[[sources/deepseek-ocr]]"]
updated: 2026-05-31
---

# DeepSeek-OCR

**DeepSeek-OCR** は **DeepSeek-AI**（中国の AI 研究機関）が開発した **OCR / 文書理解特化型 MLLM**（2025 年 10 月公開、arXiv:2510.18234）。**「視覚モダリティをテキスト圧縮媒体として活用する」**新パラダイムを提唱し、**SAM + 16× ConvNet + CLIP の直列構成 DeepEncoder** + **DeepSeek-3B-MoE-A570M デコーダ**で **少ない視覚トークンで高精度 OCR** を実現。**wiki 初の DeepSeek-AI モデル**。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2025 年 10 月（arXiv:2510.18234） |
| 開発元 | **DeepSeek-AI** |
| 著者 | Haoran Wei（GOT-OCR2.0 の著者）, Yaofeng Sun, Yukun Li |
| 焦点 | **OCR / 文書理解特化**（汎用 MLLM ではない） |
| ライセンス | **オープンソース**（MIT 系、商用可） |
| リポジトリ | https://github.com/deepseek-ai/DeepSeek-OCR |
| モデル | Hugging Face で公開 |

## 全体構成

DeepSeek-OCR は **2 つの主要構成要素**から成る：

1. **DeepEncoder** （視覚エンコーダ）: 約 380M パラメータ
   - SAM-base + 2 層 ConvNet 圧縮 + CLIP-large
2. **DeepSeek-3B-MoE-A570M デコーダ** （MoE 言語モデル）: 推論時 570M 活性

## DeepEncoder アーキテクチャ

### コンポーネント詳細

| Component | Model | Parameters | Role |
| --- | --- | --- | --- |
| **Visual Perception** | **[[entities/sam\|SAM-base]]** | 80M | Window attention で高解像度処理（patch size 16） |
| **Compression Bridge** | 2 層 ConvNet | （少量） | **16× ダウンサンプリング**（kernel=3, stride=2, padding=1、ch 256→1024） |
| **Visual Knowledge** | **[[entities/clip\|CLIP-large]]** | 300M | Dense global attention で意味抽出 |

**合計: 約 380M パラメータ**

### トークン処理例（1024×1024 入力）

```
入力: 1024×1024 画像
    ↓ SAM (patch=16)
4096 パッチ・トークン
    ↓ 2 層 ConvNet (16× 圧縮)
256 トークン
    ↓ CLIP-large (dense global attention)
256 視覚トークン → デコーダへ
```

## 6 つの解像度モード

| Mode | Resolution | Vision Tokens | Process |
| --- | --- | --- | --- |
| **Tiny** | 512×512 | **64** | Resize |
| **Small** | 640×640 | **100** | Resize |
| **Base** | 1024×1024 | **256** | Padding |
| **Large** | 1280×1280 | **400** | Padding |
| **Gundam** | n×(640²) + 1024² | **n×100 + 256**（n=2-9 典型） | 動的（タイル + グローバル） |
| **Gundam-M** | n×(1024²) + 1280² | （継続訓練版） | 動的（高解像度版） |

**Valid Token 計算式（パディング・モード）**:
$$N_{\text{valid}} = \lceil N_{\text{actual}} \times [1 - \frac{\max(w,h)-\min(w,h)}{\max(w,h)}] \rceil$$

## デコーダ: DeepSeek-3B-MoE-A570M

| 項目 | 値 |
| --- | --- |
| 総パラメータ | 3B |
| 推論時活性パラメータ | **570M (A570M)** |
| Routed Experts | **64 個** |
| 推論ごとの活性 expert | **6 個** |
| Shared Experts | **2 個** |

**機能**: 圧縮された視覚トークン（n 個）をテキスト・トークン列（N 個、n ≤ N）にマッピング。**非常にコンパクトな MoE 設計**で、コンパクトな言語モデルが学習できる非線形マッピングを実現。

## 訓練データ構成

| カテゴリ | 比率 | 内容 |
| --- | --- | --- |
| **OCR 1.0** | **70%** | 30M+ PDF ページ、100 言語、シーン OCR 中英各 10M |
| **OCR 2.0** | （上記内） | 10M チャート + 5M 化学式 + 1M 平面幾何 |
| **一般視覚** | **20%** | DeepSeek-VL2 風（キャプショニング + 検出 + グラウンディング） |
| **テキスト** | **10%** | 社内事前学習データ、系列長 8192 |

## 訓練パイプライン

### Stage 1: DeepEncoder 訓練
- OCR 1.0/2.0 + 100M LAION サンプル
- Batch 1280、Epoch 2、AdamW、lr 5e-5、cosine annealing、seq 4096

### Stage 2: 全訓練
- **インフラ**: HAI-LLM、**20 ノード × 8× A100-40G = 160 GPU**
- **並列**: パイプライン並列 PP=4 + データ並列 DP=40
- **コンポーネント配置**:
  - PP0: SAM + 圧縮器（凍結）
  - PP1: CLIP（凍結解除、入力埋め込み）
  - PP2-PP3: DeepSeek-3B-MoE 各 6 層
- **速度**: テキスト専用 90B トークン/日、マルチモーダル **70B トークン/日**

## 主要ベンチマーク結果

### Fox ベンチマーク（視覚-テキスト圧縮）

**Tiny 64 トークン**:
- 10-12× 圧縮: **96.5%** 精度
- 17× 圧縮: 76.4%
- **20× 圧縮: 59.1%**（記憶忘却機構の根拠）

**Small 100 トークン**:
- 6.7× 圧縮: **98.5%** 精度
- 9.7× 圧縮: **96.8%**
- 12.6× 圧縮: 87.1%

### OmniDocBench（編集距離↓、低いほど良い）

| Model | Vision Tokens | Edit Distance |
| --- | --- | --- |
| GOT-OCR2.0 | 256 | 0.287 |
| Qwen2.5-VL-7B | 3949 | 0.316 |
| **DeepSeek-OCR Small** | **100** | **0.205**（GOT-OCR2.0 比 2.56× 効率） |
| **DeepSeek-OCR Base** | **256** | **0.156** |
| OCRFlux-3B | 3949 | 0.238 |
| Qwen2.5-VL-72B | 3949 | 0.214 |
| InternVL3-78B | 6790 | 0.218 |
| MinerU2.0 | 6790 | 0.133 |
| **DeepSeek-OCR Large** | **400** | **0.117** |
| dots.ocr | 5545 | 0.125 |
| **DeepSeek-OCR Gundam** | **795** | **0.083 SOTA**（MinerU2.0 比 8.5× 効率） |
| **DeepSeek-OCR Gundam-M** | 1853 | 0.085 SOTA |

### 文書タイプ別性能（Base 256 トークン）

| Category | Edit Distance |
| --- | --- |
| Financial Reports | **0.027** |
| Books | **0.037** |
| Slides | 0.08 |
| Exam Papers | 0.13 |
| Newspapers | 0.645（Gundam 必須） |

## 実用的価値

| 構成 | スループット |
| --- | --- |
| 1 台 A100-40G | **200K+ ページ/日** |
| 20 ノード（160 GPU） | **3300 万ページ/日** |

→ **LLM/VLM の事前学習用合成データ生成**を本番グレードの速度で。

## 質的能力

- **深層解析**: チャート → HTML 表、化学式 → SMILES、幾何 → 座標
- **多言語**: 約 100 言語（アラビア語、シンハラ語等を実証）
- **一般視覚理解**: OCR 専門化（70%）でもキャプショニング/検出/グラウンディング能力を保持

## 記憶忘却機構（独自貢献）

**人間の記憶減衰と視覚知覚劣化の並行性**を活用した文脈圧縮提案：

1. 歴史的会話テキストを画像にレンダリング
2. 古い画像を段階的に縮小
3. 多レベル圧縮:
   - 最近: 高解像度（全トークン）
   - 古い: 削減解像度
   - 非常に古い: 極度に圧縮（60% 精度を維持）

→ **LLM の長文脈問題に対する新解決策**として提案

## 限界

- **OCR 特化のため汎用 MLLM ではない**（Qwen3-VL/InternVL 3.5 のような MMMU 報告なし）
- **SFT 段階なし**: 一部タスクで補完プロンプトが必要
- **Newspapers で 0.645 編集距離**: Gundam/Gundam-M 必須
- **20× 圧縮で 60% 精度**: 40% 損失の実用上の許容度が不明
- **記憶忘却機構は概念実証のみ**: 実 LLM 対話への統合実験なし
- **OmniDocBench に評価偏重**: MMMU や視覚一般理解の評価欠如
- **SAM + CLIP 直列の汎用性**: 自然画像理解での適用性が不明

## 系譜・関連モデル

### 同著者の前作
- **Vary** (Wei et al., ECCV 2024): 視覚語彙拡大
- **Vary-tiny** (2024): 小規模言語モデル + 強化視覚語彙
- **Focus Anywhere** (2024): 細粒度多ページ文書理解
- **Slow Perception** (2024): 幾何図形の段階的知覚
- **GOT-OCR2.0** (2024): General OCR Theory、本論文の直接の前任
- **DeepSeek-OCR** (2025) ← 本ページ

### DeepSeek-AI の他系譜
- **DeepSeek-V2/V3**: 汎用 LLM（MoE）
- **DeepSeek-R1**: 推論モデル
- **DeepSeek-VL / VL2**: 汎用 VLM

DeepSeek-OCR は **DeepSeek-AI の OCR 特化分枝**で、汎用 VLM 系列とは別系統。

## 関連ページ

- [[sources/deepseek-ocr]] — 原典の要約
- [[translations/deepseek-ocr]] — 原典の翻訳（ar5iv 本文ベース）
- [[entities/sam]] — Visual Perception 成分として使用される SAM-base
- [[entities/clip]] — Visual Knowledge 成分として使用される CLIP-large
- [[concepts/vision-transformer]] — SAM/CLIP の基盤
- [[concepts/foundation-model]] — OCR 特化 MLLM の代表
- [[questions/vit-dynamic-resolution-evolution]] — 「視覚トークン数最小化」という新視点
- [[entities/qwen2-5-vl]] — 同時期の汎用 MLLM、OmniDocBench で比較
- [[entities/qwen3-vl]] — 同時期の汎用 MLLM 競合
- [[entities/internvl-3]] / [[entities/internvl-3-5]] — InternVL 系の比較対象
- [[entities/gemma-3]] — 同時期の保守路線（タイル分割）との対比
