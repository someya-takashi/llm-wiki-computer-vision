---
type: source
source_path: raw/papers/Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling.md
source_kind: paper
title: "Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling"
authors: [Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang]
year: 2024
venue: "arXiv:2412.05271 (technical report)"
ingested: 2026-05-29
tags: [mllm, vllm, internvl-2-5, internvit-v2-5, mmmu-70-percent, cot-reasoning, test-time-scaling, progressive-scaling, opengvlab]
translation: "[[translations/internvl-2-5]]"
---

# InternVL 2.5 — MMMU で 70% を超えた初のオープンソース MLLM + 段階的スケーリング戦略を公式化

> 原典: [[translations/internvl-2-5]] ・ `raw/papers/Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling.md`
> 著者: Zhe Chen, Weiyun Wang, Yue Cao ら（Shanghai AI Lab + SenseTime + 清華大学 + 南京大学 + 復旦大学 + 香港中文大学 + Shanghai Jiao Tong University）
> 投稿: arXiv:2412.05271（2024 年 12 月、technical report）
> リポジトリ: <https://github.com/OpenGVLab/InternVL>
> モデル: <https://huggingface.co/OpenGVLab/InternVL2_5-78B>

---

## 一言まとめ

**「[[entities/internvl-1-5|InternVL 1.5]] / 2.0 のアーキテクチャを保ちつつ、**モデル・データ・テスト時の 3 軸でスケーリング** することで、**MMMU で 70.1% という、商用 GPT-4o (69.1) / Claude-3.5-Sonnet (68.3) / Gemini-1.5-Pro (62.2) を凌ぎ、**70% を超えた初のオープンソース MLLM** を実現した論文」**。InternVL 2.5 シリーズは **1B / 2B / 4B / 8B / 26B / 38B / 78B の 7 サイズ** で展開、視覚は **InternViT-300M-V2.5（軽量）/ InternViT-6B-V2.5（大規模、45 層）** の 2 系統、LLM は **InternLM 2.5 / Qwen 2.5 系**を採用。**段階的スケーリング戦略（Progressive Scaling Strategy）を初めて公式化** し、Qwen2-VL の 1/10 の訓練トークン（120B vs 1.4T）で 78B モデルを訓練。**Chain-of-Thought (CoT) + majority voting** によるテスト時スケーリングで MMMU +3.7 ポイント追加改善。

---

## 背景と問題意識

### InternVL 1.5 から 2.0 へ、そして 2.5 へ — シリーズが直面した壁

[[sources/internvl-1-5|InternVL 1.5]] (2024-04) は **「OCR と中国語で GPT-4V に追いついた」** が、以下の壁が残っていた:

| 壁 | 状況 |
|---|---|
| **MMMU 70% の壁** | InternVL 1.5: 45.2 / InternVL2-Llama3-76B: 62.7 / GPT-4o: 69.1 / Claude-3.5-Sonnet: 68.3 — 主要 OS MLLM は **MMMU 70% を超えていなかった** |
| **CoT 推論で性能低下** | 既存 OS MLLM は CoT で逆に性能低下することが多く、テスト時スケーリングが効かない |
| **訓練データの非効率なスケーリング** | Qwen2-VL は **1.4T トークン** を必要、コスト爆発 |
| **データ品質と CoT の関係** | 数千の繰り返しパターンサンプルだけで CoT が「無限ループ」に陥る |

> **補足: 「test-time scaling（テスト時スケーリング）」とは** — 推論時に計算を増やすことで性能向上を狙う手法群。OpenAI o1 が 2024 年 9 月に「推論段階での RL + CoT」で注目を集めて以来、LLM/MLLM 界での重要トピック。**Chain-of-Thought**（明示的に "Let's think step by step" させる）と **Majority Voting**（複数応答の多数決）が代表的手法。InternVL 2.5 は MLLM 向けに本格的に取り組んだ初期論文の 1 つ。

### この論文の位置づけ

InternVL シリーズの **「商用追従ライン」の最終到達点**（次の InternVL 3 へ続く）。発表時期（2024 Dec）は **OpenAI o1 (2024 Sep) と Claude-3.5-Sonnet (2024 Jun) のテスト時推論ブーム** の直後で、OS コミュニティが「**MMMU 70%**」という心理的目標を達成した瞬間。

---

## 提案手法

### モデル: ViT-MLP-LLM 構造（InternVL 1.5 / 2.0 と同じ）

**重要**: モデルアーキテクチャは [[entities/internvl-1-5|InternVL 1.5]] と完全に同じ。**「アーキテクチャは変えず、訓練戦略・データ・テスト時スケーリングを精緻化する」** という哲学。

```
画像入力（任意解像度）
   ↓ 動的高解像度（1-40 タイル）
448×448 タイル × N + サムネイル
   ↓ InternViT-300M-V2.5 or InternViT-6B-V2.5
1024 visual tokens / tile
   ↓ Pixel Unshuffle (1/4 圧縮)
256 visual tokens / tile
   ↓ 2 層 MLP プロジェクタ
LLM 埋め込み空間
   ↓ InternLM 2.5 / Qwen 2.5（1B 〜 72B レンジ）
出力テキスト
```

### モデルファミリー（7 サイズ）

| モデル | 視覚 | LLM | 合計 | OpenCompass |
|---|---|---|---|---|
| **InternVL2.5-1B** | InternViT-300M-V2.5 | Qwen2.5-0.5B-Instruct | 0.9B | **54.5** |
| **InternVL2.5-2B** | InternViT-300M-V2.5 | InternLM2.5-1.8B-Chat | 2.2B | **59.8** |
| **InternVL2.5-4B** | InternViT-300M-V2.5 | Qwen2.5-3B-Instruct | 3.7B | **65.1** |
| **InternVL2.5-8B** | InternViT-300M-V2.5 | InternLM2.5-7B-Chat | 8.1B | **68.1** |
| **InternVL2.5-26B** | InternViT-6B-V2.5 | InternLM2.5-20B-Chat | 25.5B | **71.3** |
| **InternVL2.5-38B** | InternViT-6B-V2.5 | Qwen2.5-32B-Instruct | 38.4B | **73.3** |
| **InternVL2.5-78B** | InternViT-6B-V2.5 | Qwen2.5-72B-Instruct | 78.4B | **75.5** |
| InternVL2.5-Pro | InternViT-6B-V2.5 | – (非公開) | – | – |

**InternViT-6B-V2.5** は **45 層（48 - 3、[[entities/internvl-1-5|InternVL 1.5]] と同じ）、5.5B**。**InternViT-300M-V2.5** は CLIP-ViT-L で初期化 + InternViT-6B-V1.5 から cosine 蒸留 + NTP 損失で段階訓練。

### 3 段階訓練パイプライン

**Stage 1: MLP Warmup** — MLP のみ訓練可、ViT + LLM 凍結。動的高解像度をこの段階から使用。NTP 損失、高学習率。

**Stage 1.5: ViT Incremental Learning（任意）** — **ViT + MLP** 両方訓練可。Web 規模データで希少なドメイン（多言語 OCR、数学チャート等）の特徴抽出能力を強化。低学習率で catastrophic forgetting 防止。

**Stage 2: Full Model Instruction Tuning** — **全モデル** 訓練可。データ品質が極めて重要。

### 段階的スケーリング戦略（Progressive Scaling Strategy）の公式化 — 本論文の独自貢献

**観察**: 「**ViT と LLM を NTP 損失で共同訓練しても、得られる視覚特徴は他の LLM が容易理解できる汎用表現になる**」。

**手順**:
1. Stage 1.5 で InternViT を **小型 LLM（例: 20B）** と訓練 → 基本視覚能力 + クロスモーダル整列
2. 訓練済み InternViT を **大型 LLM（例: 72B）** に転送 → 再訓練不要
3. 大型モデル訓練時には Stage 1.5 を **スキップ**

**効果**: **Qwen2-VL の 1/10 未満の訓練トークン**（120B vs 1.4T）で同等以上の性能を実現。

> **補足: 「6B vision encoder の威力」** — Qwen2-VL-72B（600M 視覚 + 72B LLM）vs InternVL2.5-78B（**6B 視覚** + 72B LLM）。**視覚エンコーダ規模 10×** が訓練トークン要求量 1/10 に転換。**「大規模視覚エンコーダ = 訓練データ効率」**という主張。

### 訓練強化

- **Random JPEG Compression**: 品質 75-100 のランダム JPEG 圧縮で実世界画像劣化をシミュレート
- **Square Averaging Loss**: w_i = 1/x^0.5（token averaging と sample averaging の中間）で応答長バイアスを軽減

### データの厳密フィルタリング — 本論文のもう一つの中核貢献

**重要発見**: **「LLM は視覚エンコーダよりデータノイズに顕著に敏感。Stage 2 で全重み訓練可時、わずか数千の異常サンプルでも推論時の異常挙動を招く」**。

**繰り返し生成（repetitive generation）が最も有害**。CoT 推論で「無限ループ」を生み、テスト時スケーリングを完全に破壊する。

**パイプライン（図 8）**:
- テキスト: LLM スコアリング + 繰り返し検出 + ヒューリスティック規則
- マルチモーダル: 繰り返し検出 + ヒューリスティック規則のみ（スコアリング困難）

**データ規模**: InternVL 1.5: 5.1M → 2.0: 7.3M → **2.5: 16.3M サンプル**（2.0 の倍）。**動画 +30% / 複数画像 +5%** で長動画・複数画像理解強化。

### テスト時スケーリング（test-time scaling）

**Chain-of-Thought + Majority Voting** で MMMU を **直接応答 66.4 → CoT 70.1（+3.7）** に押し上げ。さらに majority voting で追加改善（InternVL2-76B で 62.7 → 65.3）。

---

## 実験結果と知見

### 5.1 MMMU 70% の壁を破る（多分野推論 + 数学）

| Model | MMMU(val) | MathVista | OlympiadBench |
|---|---|---|---|
| GPT-4V | 63.1 | 58.1 | 18.0 |
| Gemini-1.5-Pro | 62.2 | 63.9 | – |
| Claude-3.5-Sonnet | 68.3 | 67.7 | – |
| **GPT-4o (2024-05)** | **69.1** | 63.8 | **25.9** |
| Qwen2-VL-72B | 64.5 | 70.5 | – |
| InternVL2-Llama3-76B | 62.7 | 65.5 | 5.5 |
| **InternVL2.5-78B** | **70.1** | **72.3** | 11.6 |
| **InternVL2.5-38B** | **63.9** | **71.9** | **12.1** |

**「MMMU 70% を超えた初のオープン MLLM」** + **MathVista で GPT-4o を +8.5 圧倒**（72.3 vs 63.8）。

### 5.2 OCR / 文書: VCR で劇的改善

| Model | VCR-EN-Easy EM/Jaccard |
|---|---|
| InternVL2-2B | 32.9 / 59.2 |
| **InternVL2.5-2B** | **93.2 / 97.6** |
| InternVL2-26B | 74.5 / 86.7 |
| **InternVL2.5-26B** | **94.4 / 98.0** |
| Claude-3.5-Sonnet | 63.9 / 74.7 |
| GPT-4o | 91.6 / 96.4 |

**わずか 22K の VCR 訓練サンプル追加で 60+ ポイント改善**。「OCR 能力ではなく指示追従能力の不足」だった、と分析。

### 5.7 視覚的グラウンディング: RefCOCO 三部作で SOTA

InternVL2.5-78B が RefCOCO/+/g の **8 サブセット平均 92.3 で SOTA**、Qwen2-VL-72B（91.1）を +1.2 上回る。**Grounding 特化モデル**（Grounding-DINO-L 86.6 / UNINEXT-H 88.9 / CogVLM-Grounding-17B 90.3）も超える。

### 5.9 動画: 入力フレーム数スケーラビリティの発見

**InternVL 2.0 は 16/32 フレームで頭打ち、InternVL 2.5 はフレーム増加で改善継続**。訓練時のフレームサンプリングを 4-24 → 8-32 に拡張した効果。

InternVL2.5-78B: Video-MME 72.1/74.0、MVBench 76.4（Qwen2-VL-72B 73.6 超え）、MMBench-Video 1.97（GPT-4o 1.63 超え）、MLVU 75.7 SOTA。

### 6. 言語能力: 基盤 LLM を「保持 + 強化」

| Avg over 17 benchmarks | InternVL 2.0 | InternVL 2.5 | vs 基盤 LLM |
|---|---|---|---|
| 2B | 39.2 | **48.4** | +0.8（vs InternLM2.5-1.8B-Chat） |
| 8B | 67.2 | **70.0** | +0.5（vs InternLM2.5-7B-Chat） |
| 26B | 64.2 | **72.9** | +1.4（vs InternLM2.5-20B-Chat） |

**「MLLM 訓練が純粋言語能力を低下させる問題」** を高品質言語データ追加で解決。Mini-InternVL や InternVL 2.0 では純粋言語が劣化する課題があった。

### 7. InternViT 視覚能力: 「中間層特徴」発見の精緻化

**重要な観察**: InternViT のバージョン更新で **linear probing 性能が低下、attention pooling / head tuning は維持または改善**。Δ（差）が拡大。

| Version | Linear probe ImageNet avg | Attn pool ImageNet avg | Δ |
|---|---|---|---|
| InternViT-6B-224px | 82.5 | 85.5 | 3.0 |
| InternViT-6B-448px-V1.5 | 77.9 | 85.1 | 7.2 |
| **InternViT-6B-448px-V2.5** | **78.5** | **85.2** | **6.7** |

**意義**: 「最終層の特徴が線形分離性を失う一方、より複雑・非線形な意味表現を獲得」。これは [[entities/perception-encoder|PE]] の「中間層特徴」発見と本質的に同じ現象を、InternViT 系統で **2024 年中盤に独立に発見** していた、ということ。MLLM 用途には好ましい性質（catastrophic forgetting なし、open-set 意味捕捉）。

セグメンテーション（ADE20K + COCO-Stuff）でも同じ傾向。

---

## 限界・批判的視点

### MMVet v2 / MMMU では closed-source に依然劣る

- MMVet v2: GPT-4o 71.0 / Claude-3.5 71.8 / Qwen2-VL-72B 66.9 / **InternVL2.5-78B 65.5**
- 「マルチモーダル統合能力」では商用にまだ届かない
- MMMU でも 70.1 vs GPT-4o 69.1（+1.0）、Claude-3.5-Sonnet 68.3 と僅差

### WildVision で長応答品質が低い

- InternVL2.5-78B WildVision 71.4 vs GPT-4o 80.6 (-9.2)
- 「簡潔・正確だが、長応答での人間嗜好整合に課題」と著者自身が認める

### Stage 3（preference optimization 等の post-training）未実施

- DPO / RLHF / RLAIF が未統合 = ユーザ嗜好整合の余地大
- 著者は「将来の研究」と明記

### 視覚エンコーダ重視論への反論余地

- 「6B vision encoder が訓練トークンを 1/10 にする」主張に対し、**Qwen2-VL-2B の優位（600M 視覚）** が示すように、**小型モデル領域では小さい視覚エンコーダが OCR で勝つ** ケースもある
- InternVL2.5-2B が Qwen2-VL-2B に DocVQA / InfoVQA で劣るのは 300M vs 600M 視覚エンコーダの差

### 段階的スケーリング戦略の汎化性

- 段階的スケーリング戦略の効果は InternViT 系統で実証されたが、**Qwen-VL や LLaVA 系で同じ効果が出るかは不明**
- 「ViT と LLM の汎用特徴」という仮定の検証が必要

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **InternVL 2.5** | – | InternVL シリーズ第 6 世代（2024 Dec、本論文） |
| **InternVL 2.0** | – | InternVL シリーズ第 5 世代（2024 Jul、1B-76B レンジ、論文なし） |
| **InternVL2-1B/2B/4B/8B/26B/40B/76B** | – | InternVL 2.0 のサイズバリアント |
| **InternVL2.5-1B/2B/4B/8B/26B/38B/78B** | – | InternVL 2.5 の 7 サイズ |
| **InternViT-6B-448px-V2.5** | – | InternViT 第 4 版（45 層、dynamic 448、5.5B） |
| **InternViT-300M-448px-V2.5** | – | InternViT-300M 第 3 版（蒸留版の段階訓練強化版） |
| **NTP loss** | Next Token Prediction loss | 自己回帰言語モデルの標準損失 |
| **QK-Norm** | Query-Key Normalization | 大規模 ViT の安定化技法（InternViT-6B が採用） |
| **RMSNorm** | Root Mean Square Normalization | LayerNorm の効率版（LLaMA 系で標準） |
| **InternLM 2.5** | – | Shanghai AI Lab の最新 LLM 系列（1.8B/7B/20B-Chat） |
| **Qwen 2.5** | – | Alibaba の最新 LLM 系列（0.5B/3B/32B/72B-Instruct） |
| **Hermes-2-Theta-Llama-3-70B** | – | NousResearch の Llama-3 派生（InternVL2-76B が使用） |
| **Nous-Hermes-2-Yi-34B** | – | Yi-34B 派生（InternVL2-40B が使用） |
| **Pixel Unshuffle** | – | 空間 → チャンネル変換（[[entities/internvl-1-5\|InternVL 1.5]] の Pixel Shuffle と同じ） |
| **Stage 1 / 1.5 / 2** | – | InternVL 2.5 の訓練 3 段階（MLP warmup / ViT incremental / Full instruction tuning） |
| **Progressive Scaling Strategy** | – | 小型 LLM で ViT 訓練 → 大型 LLM へ転送する戦略（本論文で公式化） |
| **Test-Time Scaling** | – | 推論時計算を増やして性能向上を狙う手法群（CoT + Majority Voting 等） |
| **Chain-of-Thought (CoT)** | – | "Let's think step by step" 等で明示的推論を促す技法 |
| **Majority Voting** | – | 複数応答の多数決でロバスト化 |
| **Random JPEG Compression** | – | 品質 75-100 ランダム JPEG 圧縮の augmentation |
| **Square Averaging Loss** | – | w_i = 1/x^0.5 の NTP 損失重み付け |
| **Data Packing** | – | 複数サンプルを長系列に連結して GPU 利用率向上 |
| **Repetition Detection** | – | LLM ベースの繰り返しパターン検出フィルタ |
| **LLM-Based Quality Scoring** | – | LLM が 0-10 でデータ品質スコアリング |
| **MMMU** | Massive Multi-discipline Multimodal Understanding | 大学レベル 6 分野ベンチマーク |
| **MMMU-Pro** | – | MMMU の強化版（standard 10 options / vision / overall） |
| **MathVista** | – | 視覚文脈での数学推論ベンチマーク |
| **MATH-Vision** | – | 競技レベル数学（3,040 問） |
| **MathVerse** | – | 視覚数学（2,612 問、6 バージョン） |
| **OlympiadBench** | – | バイリンガル五輪・高考レベル数学・物理 |
| **CharXiv** | – | 科学論文 chart 理解（RQ: reasoning / DQ: descriptive） |
| **VCR** | Visual Caption Restoration | 画像内の隠されたテキストを復元するタスク |
| **SEED-Bench-2-Plus** | – | text-rich 視覚タスク（charts / maps / webs） |
| **Mantis-Eval** | – | 複数画像推論ベンチマーク（217 問） |
| **MMIU** | – | 7 種類の複数画像関係 × 52 タスク |
| **MuirBench** | – | 12 タスク × 10 種類の複数画像関係 |
| **BLINK** | – | 14 タスクの視覚知覚（半数以上が複数画像） |
| **MIRB** | – | 複数画像理解 4 カテゴリ（perception / world knowledge / reasoning / multi-hop） |
| **MMT-Bench** | – | 162 サブタスクの包括マルチモーダル |
| **MME-RealWorld** | – | 43 実世界シナリオ × 5 ドメイン |
| **WildVision-Bench** | – | 500 人手キュレーション、GPT-4o 採点（Claude-3-Sonnet 参照） |
| **R-Bench** | – | 実世界画像劣化への頑健性 |
| **MME** | – | 14 サブタスク総合認知 |
| **MMBench / MMBench v1.1 / MMBench-CN** | – | 20 次元 × 3000 多肢選択 |
| **MMVet / MMVet v2** | – | 6 コア能力 × 16 統合タスク（v2 は interleaved 追加） |
| **MMStar** | – | データリーク最小化 1500 問 |
| **HallusionBench** | – | Yes/No による hallusination 評価 |
| **MMHal-Bench** | – | 96 問の hallucination 評価（GPT-4o 採点） |
| **CRPE** | – | オブジェクト関係 hallucination（multiple-choice） |
| **POPE** | – | バイナリ object hallucination 評価 |
| **MMMB** | – | 6 言語の多言語マルチモーダル（en/zh/pt/ar/tr/ru） |
| **MTVQA** | – | 9 言語 text-centric VQA |
| **Video-MME** | – | 全スペクトル動画分析、subtitle あり/なし |
| **MVBench** | – | 20 動画タスク（perception → cognition、16 フレーム） |
| **MMBench-Video** | – | 動画理解 + 時間推論 |
| **MLVU** | – | 長動画理解（3 分 - 2 時間） |
| **LongVideoBench** | – | 長フレーム入力の referring reasoning |
| **CG-Bench** | – | clue-based 動画理解評価（1,219 動画 + 12,000 QA） |
| **OpenCompass** | – | Shanghai AI Lab の MLLM 評価フレームワーク |
| **VLMEvalKit** | – | InternVL の評価ツール |
| **MMLU / CMMLU / C-Eval** | – | LLM 5-shot 知識ベンチマーク |
| **GAOKAO-Bench** | – | 中国高考ベース LLM ベンチ |
| **TriviaQA / NaturalQuestions** | – | LLM 0-shot QA |
| **C3** | – | 中国語多肢選択読解 |
| **RACE** | – | 英語高校生試験読解 |
| **WinoGrande / HellaSwag** | – | 常識推論 / NLI |
| **BBH** | BigBench Hard | 23 難タスク推論 |
| **GSM8K / MATH / TheoremQA** | – | 小学・高校・STEM 数学 |
| **HumanEval / MBPP / MBPP-CN** | – | コード生成（Python） |
| **Linear Probing** | – | 凍結特徴の線形分類器評価 |
| **Attention Pooling Probing** | – | 凍結特徴の attention pooling 評価 |
| **Head Tuning** | – | 凍結 backbone + UperNet head 訓練（セグ） |
| **Full Tuning** | – | 全層訓練（セグ） |
| **UperNet** | – | セマンティックセグメンテーション標準 head |
| **ImageNet-1K / ReaL / V2 / A / R / Sketch** | – | ImageNet 派生（ロバストネス評価） |
| **ADE20K / COCO-Stuff-164K** | – | セマンティックセグ標準ベンチ |
| **Grounding-DINO-L / UNINEXT-H / ONE-PEACE** | – | 視覚的グラウンディング SOTA 比較相手 |
| **Shikra / Ferret-v2 / CogVLM-Grounding / TextHawk2** | – | grounding 機能付き MLLM |
| **NVLM / Molmo / LLaVA-OneVision / Qwen2-VL / MiniCPM-V 2.6 / Ovis 1.6 / Phi-3.5-Vision / Aquila-VL / Cambrian / VILA-1.5** | – | 同時期競合 MLLM |
| **InternVL4Drive-v2** | – | InternVL 派生の 26B 自律走行 SOTA |
| **OpenAI o1** | – | OpenAI の reasoning-focused モデル（2024 Sep） |
| **LAION-5B** | – | 大規模 Web 画像-テキスト対 |
| **AGI** | – | 汎用人工知能 |

---

## 関連ページ

- 翻訳: [[translations/internvl-2-5]]
- 主要エンティティ: [[entities/internvl-2-5]]
- 直接の祖: [[sources/internvl-1-5]] / [[entities/internvl-1-5]]（InternVL 1.5、同じアーキテクチャ）
- 軽量分岐: [[sources/mini-internvl]] / [[entities/mini-internvl]]（Mini-InternVL は同じ ViT-MLP-LLM 設計を軽量化）
- 視覚エンコーダ: [[entities/internvit-300m]]（InternViT-300M-V2.5 への進化）
- 関連視覚エンコーダ系統: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]]
- 関連概念: [[concepts/foundation-model]]、[[concepts/weakly-supervised-pretraining]]、[[concepts/zero-shot-transfer]]、[[concepts/knowledge-distillation]]（InternViT-300M の蒸留経路）、[[concepts/alignment-tuning]]（[[entities/perception-encoder|PE]] の同期発見との対比）
