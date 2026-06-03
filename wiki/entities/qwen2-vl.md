---
type: entity
entity_kind: model
aliases: [Qwen2-VL, Qwen2-VL-2B, Qwen2-VL-7B, Qwen2-VL-72B, Qwen2VL, qwen2-vl]
related: ["[[entities/qwen-vl]]", "[[entities/internvl-1-5]]", "[[entities/internvl-2-5]]", "[[entities/internvl-3]]", "[[concepts/rotary-position-embeddings]]", "[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]"]
sources: ["[[sources/qwen2-vl]]"]
updated: 2026-05-30
---

# Qwen2-VL / Qwen2-VL-2B / Qwen2-VL-7B / Qwen2-VL-72B

**Qwen2-VL シリーズ**は Alibaba Group の Qwen Team が開発する MLLM 系譜の第 2 世代（2024 年 9 月公開、arXiv:2409.12191）。**Naive Dynamic Resolution** と **M-RoPE** で「任意解像度・任意アスペクト比」を初めて本格的に実現したオープンソース MLLM。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2024 年 9 月、arXiv:2409.12191 |
| 開発元 | Alibaba Group / Qwen Team |
| サイズ展開 | **2B / 7B / 72B** の 3 サイズ |
| ライセンス | Apache 2.0（72B のみ Tongyi Qianwen License） |
| リポジトリ | https://github.com/QwenLM/Qwen2-VL |
| HF Hub | `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct` |
| 言語サポート | 英語・中国語・欧州主要言語・日本語・韓国語・アラビア語・ベトナム語など |
| 文脈長 | **32K トークン**（80K まで外挿可能） |

## モデル詳細（表 1）

| モデル | Vision Encoder | LLM | 用途 |
| --- | --- | --- | --- |
| **Qwen2-VL-2B** | 675M ViT（DFN init + 2D-RoPE） | Qwen2 1.5B | オンデバイス、リソース限定環境 |
| **Qwen2-VL-7B** | 675M ViT（同上） | Qwen2 7.6B | コスト最適化、テキスト認識・動画理解強化 |
| **Qwen2-VL-72B** | 675M ViT（同上） | Qwen2 72B | 最高性能、複雑なタスク・エージェント |

**重要**: 全サイズで **同一の 675M ViT を共有**することで、LLM 規模に関係なく視覚処理コストを一定に保つ設計（cf. [[entities/mini-internvl|Mini-InternVL]] の [[entities/internvit-300m|InternViT-300M]] 共有戦略と同思想）。

## アーキテクチャの 3 つの主要技術

### 1. Naive Dynamic Resolution

- ViT の**絶対位置埋め込みを除去**し **2D-RoPE** を導入
- 任意解像度の画像を可変数の視覚トークンへ動的変換
- 推論時、複数解像度の画像を**単一系列にパック**（GPU メモリで長さ制御）
- ViT 後に **MLP で隣接 2×2 トークンを 1 トークンに圧縮**（4× 圧縮）
- 224×224 画像は `(224/14)²=256` パッチ → 圧縮で **66 トークン**化
- 設定: `min_pixels = 100 × 28 × 28`、`max_pixels = 16384 × 28 × 28`

### 2. Multimodal Rotary Position Embedding (M-RoPE)

回転位置埋め込みを **temporal / height / width の 3 成分**に分解：

| モダリティ | temporal | height | width |
| --- | --- | --- | --- |
| Text | 通常 ID | 同上 | 同上 |
| Image | 一定 | パッチ行 index | パッチ列 index |
| Video | フレームごとに増分 | パッチ行 index | パッチ列 index |

副次効果: 位置 ID 値が小さく抑えられ、**学習 16K → 推論 80K まで外挿可能**。

### 3. Unified Image and Video Understanding

- 動画: **2 fps サンプリング** + **深さ 2 の 3D 畳み込み**で 3D チューブ化（系列長を増やさず 2 倍のフレーム処理）
- 画像は **2 つの同一フレーム**として扱う
- 動画あたり最大 **16,384 視覚トークン**で制限

## 特殊トークン

| 用途 | トークン |
| --- | --- |
| 視覚入力境界 | `<|vision_start|>`, `<|vision_end|>` |
| バウンディング・ボックス | `<|box_start|>`, `<|box_end|>` |
| 参照対象 | `<|object_ref_start|>`, `<|object_ref_end|>` |
| 対話境界 | `<|im_start|>`, `<|im_end|>` |

座標は **[0, 1000) 正規化文字列**（[[entities/qwen-vl|Qwen-VL]] 初代を踏襲）。

## 学習パイプライン（3 段階、累積 1.4T トークン）

| Stage | 凍結対象 | データ | トークン |
| --- | --- | --- | --- |
| Stage 1 | LLM 凍結 | 画像-テキスト対、OCR、画像分類 | 約 600B |
| Stage 2 | 全パラメータ凍結解除 | 画像関連データ追加 | 追加 800B |
| Stage 3 | ViT 凍結 | ChatML 命令データ | SFT |

- LLM 初期化: **Qwen2 シリーズ**
- ViT 初期化: **[[entities/dfn|DFN（Data Filtering Networks）]]の ViT**（絶対位置埋め込みを 2D-RoPE に置換）。詳細: [[sources/dfn]]
- データカットオフ: 2023 年 6 月
- **テキスト・トークンにのみ監督を与える**（視覚トークンは予測対象としない）

## 主要ベンチマーク結果（ハイライト）

### 一般 VQA / 文書 / OCR（表 2）

| ベンチマーク | 2B | 7B | 72B | GPT-4o | Claude-3.5 |
| --- | --- | --- | --- | --- | --- |
| MMMU val | 41.1 | 54.1 | 64.5 | 69.1 | 68.3 |
| **DocVQA test** | 90.1 | 94.5 | **96.5** | 92.8 | 95.2 |
| **InfoVQA test** | 65.5 | 76.5 | **84.5** | - | - |
| **AI2D** | 74.7 | 83.0 | **88.1** | 84.6 | 80.2 |
| ChartQA test | 73.5 | 83.0 | 88.3 | 85.7 | 90.8 |
| **TextVQA val** | 79.7 | 84.3 | **85.5** | - | - |
| **OCRBench** | 809 | 866 | **877** | 736 | 788 |
| **MTVQA** | 18.1 | 25.6 | **30.9** | 27.8 | 25.7 |
| **VCR zh easy** | 46.2 | 59.9 | **65.4** | 14.9 | 1.0 |
| **RealWorldQA** | 62.9 | 70.1 | **77.8** | 75.4 | 60.1 |
| MME sum | 1872.0 | 2326.8 | **2482.7** | 2328.7 | 1920.0 |
| MMBench-EN | 74.9 | 83.0 | 86.5 | 83.4 | 79.7 |
| **MMBench-CN** | 73.5 | 80.5 | **86.6** | 82.1 | 80.7 |
| MMT-Bench | 54.5 | 63.7 | **71.7** | 65.5 | - |
| MMStar | 48.0 | 60.7 | **68.3** | 63.9 | 62.2 |
| MMVet | 49.5 | 62.0 | **74.0** | 69.1 | 66.0 |
| HallBench | 41.7 | 50.6 | **58.1** | 55.0 | 49.9 |
| **MathVista** | 43.0 | 58.2 | **70.5** | 63.8 | 67.7 |
| MathVision | 12.4 | 16.3 | 25.9 | 30.4 | - |

### 参照表現理解（表 6）

| モデル | RefCOCO val | RefCOCO+ val | RefCOCOg val |
| --- | --- | --- | --- |
| Qwen-VL（初代） | 89.4 | 83.1 | 85.6 |
| Qwen2-VL-2B | 87.6 | 79.0 | 81.2 |
| Qwen2-VL-7B | 91.7 | 85.8 | 87.3 |
| **Qwen2-VL-72B** | **93.2** | **90.1** | **89.9** |

**Qwen-VL 初代から RefCOCO val で +3.8 改善**、動的解像度の効果が顕著。

### 動画理解（表 4）

| ベンチマーク | 2B | 7B | 72B | GPT-4o |
| --- | --- | --- | --- | --- |
| MVBench | 63.2 | 67.0 | **73.6** | - |
| PerceptionTest | 53.9 | 62.3 | **68.0** | - |
| EgoSchema | 54.9 | 66.7 | **77.9** | 72.2 |
| Video-MME wo/w | 55.6/60.4 | 63.3/69.0 | 71.2/77.8 | 71.9/77.2 |

### エージェント能力（表 5、Qwen2-VL-72B）

| Benchmark | 従来 SoTA | GPT-4o | Qwen2-VL-72B |
| --- | --- | --- | --- |
| Function Calling TM | - | 90.2 | **93.1** |
| Function Calling EM | - | 50.0 | **53.2** |
| AITZ TM | 83.0 | 70.0 | **89.6** |
| AITZ EM | 47.7 | 35.3 | **72.1** |
| Number Line | 89.4 | 91.5 | **100.0** |
| EZPoint | 50.0 | 85.5 | **100.0** |
| ALFRED SR | 67.7 | - | **67.8** |
| R2R SR | **79.0** | 43.7 | 51.7 |

UI 操作で GPT-4o を圧倒、VLN（R2R）では専門モデルに大きく後れ。

### 多言語 OCR（表 3、社内ベンチ）

| 言語 | GPT-4o | Qwen2-VL-72B |
| --- | --- | --- |
| Korean | 87.8 | **94.5** |
| Japanese | 88.3 | **93.4** |
| French | 89.7 | **94.1** |
| German | 88.3 | **91.5** |
| Italian | 74.1 | **89.8** |
| Russian | 96.8 | **97.2** |
| Vietnamese | 72.0 | **73.0** |
| Arabic | **75.9** | 70.7 |

## アブレーション要点

- **動的解像度**: 平均 1924 トークンの動的版が固定 3136 版を OCRBench で上回る（866 vs 786）
- **M-RoPE**: 1D-RoPE 比で MathVista +4.2、動画 NextQA +2.1、STAR +2.4
- **長さ外挿**: 学習 16K で推論 80K まで頑健

## 限界

- **MMMU で GPT-4o に -4.6**（大学レベル推論で後れ）
- **MathVision で GPT-4o に -4.5**（複雑数学）
- **R2R で -27.3**（3D 空間モデリングが弱点）
- **アラビア語 OCR で -5.2**
- **動画 16K トークン上限**で 1 時間級の長動画は精度低下
- 依然として「LLM 事前学習 → MLLM 適応」の伝統パラダイム（cf. InternVL 3 の Native Multimodal Pre-Training）

## システム実装

- 学習基盤: **Alibaba Cloud PAI-Lingjun**
- ストレージ: **CPFS（テキスト）+ OSS（視覚）**の分離、動画キャッシュデコード
- 並列化: **3D 並列（DP + TP + PP）+ ZeRO-1 + 系列並列**
- 72B: **1F1B PP**（インターリーブ版より速い）
- ソフトウェア: PyTorch 2.1.2 + CUDA 11.8 + flash-attention + 融合演算子

## 系譜・後継

- **[[entities/qwen-vl|Qwen-VL]]**（2023.08, 9.6B）← 初代、固定 448² + 256 トークン
- **Qwen2-VL**（2024.09, 2B/7B/72B）← 本ページ、Naive Dynamic Resolution + M-RoPE
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（2025.02, 3B/7B/72B）: Window Attention + ViT from scratch + MRoPE Aligned to Absolute Time + 動的 FPS + QwenVL HTML フォーマット + 4.1T トークン、MMMU 70.2 初突破、OCRBench_v2 zh で Gemini +20.6 圧倒、Charades-STA mIoU 50.9 SOTA、ScreenSpot Pro で本モデル 1.6% から 43.6%（27× 飛躍）、視覚エージェント本格化
- **Qwen3-VL**（2025-）← Qwen3 LLM ベースの後継（将来追加予定）

## 関連ページ

- [[sources/qwen2-vl]] — 原典の要約
- [[translations/qwen2-vl]] — 原典の翻訳
- [[entities/qwen-vl]] — 前世代 Qwen-VL
- [[entities/qwen2-5-vl]] — 直接の後継、Window Attention + MRoPE absolute time + 4.1T + QwenVL HTML + GUI Agent
- [[entities/internvl-1-5]] — タイル分割で動的解像度を先行実装した競合系
- [[entities/internvl-2-5]] — Progressive Scaling を提唱、Qwen2-VL の 1/12 訓練トークンで超えると主張
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系後継
- [[entities/internvl-3-5]] — Cascade RL + MoE の InternVL 系最新版
- [[concepts/rotary-position-embeddings]] — RoPE の基礎
- [[concepts/foundation-model]]
- [[concepts/weakly-supervised-pretraining]]
- [[concepts/zero-shot-transfer]]
