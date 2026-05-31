---
type: entity
entity_kind: model
aliases: [Qwen-VL, Qwen-VL-Chat, QwenVL, qwen-vl]
related: [[entities/internvl]], [[entities/internvit-300m]], [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]
sources: [[sources/qwen-vl]]
updated: 2026-05-30
---

# Qwen-VL / Qwen-VL-Chat

**Qwen-VL シリーズ**は Alibaba の Qwen チームが開発する大規模視覚言語モデル（LVLM / MLLM）系譜の総称。本ページではその **初代モデル**（2023 年 8 月公開、arXiv:2308.12966）である Qwen-VL と、命令調整版の Qwen-VL-Chat を扱う。後継の Qwen2-VL、Qwen2.5-VL は別エンティティとして扱う。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2023 年 8 月、arXiv:2308.12966 |
| 開発元 | Alibaba Group（Qwen チーム） |
| 総パラメータ | **9.6B** |
| ライセンス | オープンソース（Qwen-VL License、研究・商用利用可） |
| リポジトリ | https://github.com/QwenLM/Qwen-VL |
| 言語サポート | 英語・中国語（バイリンガル） |

## 構成

| コンポーネント | 中身 | パラメータ | 役割 |
| --- | --- | --- | --- |
| Visual Encoder | **OpenCLIP ViT-bigG**（パッチサイズ 14） | 1.9B | 画像のパッチ・トークン化 |
| Position-aware VL Adapter | 単層クロスアテンション + 学習可能 256 クエリ + 2D 絶対位置エンコーディング | 0.08B | ViT トークンを **固定長 256** に圧縮 |
| LLM | **Qwen-7B**（中英バイリンガル LLM、中間 ckpt で初期化） | 7.7B | 言語生成・理解 |

### 入出力仕様

- 通常画像入力：`<img>...</img>` で挟む
- バウンディング・ボックス：`<box>(x1,y1),(x2,y2)</box>`（座標は **[0, 1000) に正規化された文字列**）
- 領域記述：`<ref>...</ref>` で参照対象の語句を挟む
- 対話形式：ChatML（`<im_start>`, `<im_end>`）

## 学習パイプライン（3 段階）

| Stage | データ規模 | 解像度 | 凍結対象 | 学習率 | バッチ | ステップ |
| --- | --- | --- | --- | --- | --- | --- |
| **Stage 1** Pre-training | 1.4B 画像-テキスト対（En 77.3%、Zh 22.7%） | 224² | LLM 凍結 | 2e-4 | 30,720 | 50K |
| **Stage 2** Multi-task | 7 タスク同時、約 77M サンプル | 448² | 全モジュール学習 | 5e-5 | 4,096 | 19K |
| **Stage 3** SFT | 命令調整 350K | 448² | ViT 凍結、LLM+Adapter 学習 | 1e-5 | 128 | 8K |

Stage 2 の 7 タスク内訳：Captioning 19.7M / VQA 3.6M / Grounding 3.5M / Ref Grounding 8.7M / Grounded Captioning 8.7M / OCR 24.8M / Pure-text 7.8M。

## 主要ベンチマーク結果（ハイライト）

| ベンチマーク | Qwen-VL | Qwen-VL-Chat | 補足 |
| --- | --- | --- | --- |
| Flickr30K (0-shot CIDEr) | **85.8** | 81.0 | 当時 SOTA、Flamingo-80B 67.2 を上回る |
| Nocaps (0-shot CIDEr) | 121.4 | 120.2 | |
| VQAv2 | **79.5** | 78.2 | |
| OKVQA | **58.6** | 56.6 | |
| GQA | **59.3** | 57.5 | |
| ScienceQA-Img (0-shot) | 67.1 | 68.2 | |
| VizWiz (0-shot) | 35.2 | 38.9 | |
| TextVQA | **63.8** | 61.5 | OCR データ 24.8M の効果 |
| DocVQA | 65.1 | 62.6 | |
| ChartQA | 65.7 | **66.3** | |
| AI2D | **62.3** | 57.7 | |
| OCR-VQA | **75.7** | 70.5 | |
| RefCOCO val | **89.36** | 88.55 | Shikra-13B 87.83 を超える |
| RefCOCO+ val | **83.12** | 82.82 | |
| RefCOCOg val | **85.58** | 85.96 | |
| GRIT refexp | **78.22** | - | |
| TouchStone En | - | **645.2** | |
| TouchStone Cn | - | **401.2** | 中国語マルチモーダル対話の事実上の標準 |
| SEED-Bench All | 56.3 | **58.2** | |
| MME Perception | - | **1487.58** | |
| MME Cognition | - | **360.71** | |

## 設計上の特徴

1. **位置認識アダプタ**：BLIP-2 の Q-Former 系と異なり、クロスアテンションのクエリ-キー対に **2D 絶対位置エンコーディング**を入れることで、圧縮後も位置情報を保持
2. **座標を数値文字列化**：位置語彙を追加せず、`<box></box>` 内に純粋な文字列として座標を埋め込む。これにより通常のテキスト生成と同じ仕組みでグラウンディングを実現
3. **中英バイリンガル**：Qwen-7B の中英バイリンガル能力を継承し、SFT で中国語マルチモーダル対話を強化
4. **3 段階学習**：(凍結事前学習) → (全パラメータ・マルチタスク) → (ViT 凍結 SFT) という構造は後続の Qwen2-VL でも踏襲される

## 限界

- 解像度が **448×448 固定**（高解像度向け Window Attention は採用せず）
- ViT 出力を **256 トークンに固定圧縮**。高情報密度の画像では情報損失あり
- Stage1 で LLM 凍結のため、LLM 側で視覚との接続を学習できない（cf. [[entities/internvl-3]] の Native Multimodal Pre-Training）
- 単一モデル規模のみ（複数サイズ展開は後の Qwen2-VL から）
- 一部学習データ（社内中国語 220M、SFT 350K）は非公開

## 系譜・後継

- **Qwen-VL**（2023.08, 9.6B）← 本ページ
- **[[entities/qwen2-vl|Qwen2-VL]]**（2024.09, 2B/7B/72B）: Naive Dynamic Resolution（ViT を 2D-RoPE で任意解像度対応）+ M-RoPE（temporal/height/width 3 成分）+ 統一画像/動画処理（3D 畳み込み）、675M ViT 共有、学習 16K → 推論 80K 外挿、20+ 分動画理解、DocVQA 96.5 / OCRBench 877 / RefCOCO 93.2 SOTA
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（2025.02, 3B/7B/72B）: Window Attention + ViT from scratch + MRoPE Aligned to Absolute Time + QwenVL HTML フォーマット + 4.1T トークン、MMMU 70.2 初突破、Charades-STA mIoU 50.9 SOTA、ScreenSpot Pro 43.6（本モデル 1.6% から 27× 飛躍）、視覚エージェント本格化
- **[[entities/qwen3-vl|Qwen3-VL]]**（2025.11, 2B/4B/8B/32B Dense + 30B-A3B/235B-A22B MoE × Instruct/Thinking = 12 モデル）: Interleaved MRoPE + DeepStack + テキスト・ベース時間整合、SigLIP-2 継続学習、256K ネイティブ → 1M YaRN 外挿、Apache 2.0、Qwen ファミリーが商用最先端と完全互角
- **[[entities/qwen3-5-omni|Qwen3.5-Omni]]**（2026, Plus/Flash 2 バリアント）: 並列発展する Omni 統合路線、Thinker-Talker + Hybrid MoE + AuT + ARIA + Audio-Visual Vibe Coding、テキスト + 画像 + 音声 + 動画の完全 omnimodal 統合

## 関連ページ

- [[sources/qwen-vl]] — 原典の要約ページ
- [[translations/qwen-vl]] — 原典の翻訳
- [[entities/qwen2-vl]] — 直接の後継、Naive Dynamic Resolution + M-RoPE + 統一画像/動画処理を導入
- [[entities/internvl]] — 同時期の対抗系譜 InternVL シリーズ初代
- [[entities/internvl-1-5]] — InternVL 系の高解像度対応版（動的アスペクト比、Qwen-VL より後発）
- [[entities/internvl-2-5]] — InternVL 系のスケーリング論文
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系
- [[entities/internvl-3-5]] — Cascade RL を導入した最新 InternVL 系
- [[entities/mpo]] — Mixed Preference Optimization
- [[concepts/foundation-model]]
- [[concepts/weakly-supervised-pretraining]]
- [[concepts/zero-shot-transfer]]
- [[concepts/alignment-tuning]]
