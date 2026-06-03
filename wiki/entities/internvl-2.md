---
type: entity
entity_kind: model
aliases: [InternVL 2.0, InternVL2, InternVL-2, InternVL2-Llama3-76B]
related: ["[[entities/internvl]]", "[[entities/internvl-1-5]]", "[[entities/internvl-2-5]]", "[[entities/internvl-3]]", "[[entities/internvit-300m]]", "[[entities/mini-internvl]]"]
sources: []
updated: 2026-05-30
---

# InternVL 2.0（InternVL2）

**InternVL 2.0**（InternVL2 とも）は OpenGVLab Shanghai AI Lab が 2024 年中頃にリリースした InternVL シリーズ **第 5 世代**。**[[entities/internvl-1-5|InternVL 1.5]] と [[entities/internvl-2-5|InternVL 2.5]] の橋渡し**となる中間世代で、**1B〜76B の幅広いサイズスイート**を初めて確立。本 wiki ではまだ専用 source ページ未取り込み（**ingest 候補**）、本ページは複数の後続論文（[[sources/mini-internvl]], [[sources/internvl-2-5]], [[sources/internvl-3]]）から参照される概要エンティティ。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2024 年 7 月頃（公式ブログ・モデル公開ベース、論文形式の Technical Report はリリースなし） |
| 開発元 | OpenGVLab Shanghai AI Lab |
| サイズ展開 | **InternVL2-1B / 2B / 4B / 8B / 26B / 40B / 76B** の 7 サイズ |
| フラッグシップ | **InternVL2-Llama3-76B**（Llama3-70B-Instruct + InternViT-6B） |
| ライセンス | MIT + LLM 各種ライセンス |
| HF Hub | `OpenGVLab/InternVL2-*` |
| シリーズ位置 | 1.0 → 1.5 → **2.0**（本ページ）→ Mini → 2.5 → 3 → 3.5 |

## アーキテクチャ

[[entities/internvl-1-5|InternVL 1.5]] と本質的に同じ **ViT-MLP-LLM 構造**を継承：

- **Vision Encoder**: InternViT-6B-448px（InternVL 1.5 から継承） or **InternViT-300M-448px**（軽量版、Mini-InternVL の InternViT-300M と同系統、[[entities/internvit-300m]] 参照）
- **MLP プロジェクタ**: 2 層 MLP + Pixel Shuffle 1/4 圧縮
- **LLM**: モデルサイズに応じて **InternLM2 / Qwen2 / Llama3** の各種ベースを混用
- **動的解像度**: 1.5 と同じ 35 通りのアスペクト比 × 1-40 タイル（4K 相当）

主な進化は **モデルサイズ・スイート化とデータ拡張**であり、構造的革新は限定的（次世代 InternVL 2.5 が「アーキテクチャを保持しデータと訓練戦略のみで MMMU 70% 突破」と明示するのは、本世代までの構造的固定化の上に成り立つ）。

## 主要結果（フラッグシップ InternVL2-Llama3-76B）

| ベンチマーク | スコア | 比較 |
| --- | --- | --- |
| **MMMU val** | **62.7** | GPT-4V 56.8 を上回るが GPT-4o 69.1 には届かず |
| MMBench-EN | 86.5 | |
| MMBench-CN | 86.5 | |
| MMStar | 67.4 | |
| **MathVista** | **67.0** | |
| **RefCOCO val** | **92.2** | |
| **DocVQA** | **94.1** | |
| AI2D | 87.6 | |
| OCRBench | 851 | |
| VideoMME w/o sub | 64.8 | |

**当時 (2024.07) のオープンソース MLLM として最強クラス**、Llama-3-V-405B（同年 Meta）と並び称される。

## シリーズ系譜上の位置

```
InternVL 1.5 (2024.04)
  │ QLLaMA 廃止 → MLP プロジェクタ、26B、動的高解像度
  │
  ↓
InternVL 2.0 (2024.07) ← 本ページ
  │ 1B〜76B の 7 サイズスイート確立、ViT-MLP-LLM 構造を保持
  │ Mini-InternVL (2024.10) と並行リリース
  │
  ↓
InternVL 2.5 (2024.12)
  │ アーキテクチャ保持、データ + 訓練戦略のみで MMMU 70.1% 突破
  │ Progressive Scaling Strategy 公式化
  │
  ↓
InternVL 3 (2025.04) → InternVL 3.5 (2025.08)
```

## なぜ「ingest 候補」だが専用ページがないか

InternVL 2.0 は **公式 Technical Report がリリースされなかった**世代で、後続論文（[[sources/internvl-2-5]] / [[sources/internvl-3]] / [[sources/mini-internvl]]）でベースラインや比較対象として頻繁に言及されるが、独立した論文として ingest 対象になっていない。

主要な情報源:
- OpenGVLab 公式ブログ（リリース発表）
- HuggingFace モデルカード（`OpenGVLab/InternVL2-*`）
- 後続論文での **InternVL2-Llama3-76B / InternVL2-8B / InternVL2-2B** などの引用

## 後続論文での主要引用

| 引用元 | 文脈 |
| --- | --- |
| [[sources/mini-internvl]] | **InternVL2-Llama3-76B の 90% 性能を 4B で**達成という比較対象（Mini-InternVL-4B Avg 72.8 vs 76B 81.4） |
| [[sources/internvl-2-5]] | 「InternVL 2.0 で純粋言語が -2.1〜-2.3 低下」という前世代の弱点として明示 |
| [[sources/internvl-3]] | MMMU 62.7 (76B) → 70.1 (2.5) → 72.2 (3) の系譜表 |
| [[sources/qwen2-5-vl]] | RefCOCO で InternVL2-76B 92.2 を Qwen2.5-VL-72B 93.2 と比較 |

## 関連ページ

- [[entities/internvl-1-5]] — 前世代（QLLaMA 廃止、MLP プロジェクタ、動的高解像度）
- [[entities/internvl-2-5]] — 直接の後継、アーキテクチャ保持で MMMU 70% 突破
- [[entities/mini-internvl]] — 並行発展する軽量分枝（2024.10）
- [[entities/internvit-300m]] — 軽量視覚エンコーダ、InternVL2 と Mini-InternVL で共有
- [[entities/internvl-3]] — Native Multimodal Pre-Training の世代
- [[entities/internvl-3-5]] — Cascade RL + MoE の最新世代
