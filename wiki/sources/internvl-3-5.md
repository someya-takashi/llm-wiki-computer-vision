---
type: source
source_path: raw/papers/InternVL3.5_ Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency.md
source_kind: paper
title: "InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency"
authors: [Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, "InternVL Team Shanghai AI Lab"]
year: 2025
venue: "arXiv:2508.18265 (technical report)"
ingested: 2026-05-29
tags: [mllm, vllm, internvl-3-5, cascade-rl, mpo, gspo, vir, dvd, moe, qwen3, gpt-oss, opengvlab, internvl-series]
translation: [[translations/internvl-3-5]]
---

# InternVL 3.5 — Cascade RL + ViR + DvD + MoE で 4.05× 推論加速、GPT-5 との差を 3.9% に縮める

> 原典: [[translations/internvl-3-5]] ・ `raw/papers/InternVL3.5_ Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency.md`
> 著者: Weiyun Wang, Zhangwei Gao, Lixin Gu ら（70+ 名、Shanghai AI Laboratory InternVL Team）
> 投稿: arXiv:2508.18265（2025 年 8 月、technical report）
> リポジトリ: <https://github.com/OpenGVLab/InternVL>
> モデル: <https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B>

---

## 一言まとめ

**「[[entities/internvl-3|InternVL 3]] の Native Multimodal Pre-Training 哲学を継承しつつ、**Cascade RL（MPO + GSPO の 2 段階強化学習）+ MoE スケーリング（最大 241B-A28B）+ 効率化技術（ViR + DvD で 4.05× 加速）** という 3 軸で前進させた論文」**。InternVL 3 の MMMU 72.2 → **InternVL 3.5 で MMMU 77.7（GPT-5 84.2 / Claude-3.7-Sonnet 75.0 と並ぶ水準）**。**1B〜241B の 9 サイズ × dense + MoE × Flash 効率版** という極めて広いスイート。**Cascade RL は offline RL（MPO）で warm-up → online RL（GSPO）で精緻化** という coarse-to-fine 戦略で、GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る。**ViR が視覚トークンを動的に 50% 削減して性能 99%維持、DvD が ViT/LLM を別 GPU で並列実行 → 高解像度で 4.05× 加速**。**LLM は Qwen3 base（と GPT-OSS）に統一**、3 ベンチ平均で **Qwen3-base 比 +6.7（1B）〜 +2.3（241B）** の言語能力向上を継続実証。**InternVL シリーズの最新版（第 8 世代）**、商用 GPT-5 との差をオープンソースとして最小化。

---

## 背景と問題意識

### [[entities/internvl-3|InternVL 3]] から残った 3 つの課題

[[sources/internvl-3|InternVL 3]]（2025 Apr）が **「Native Multimodal Pre-Training」** で MMMU 72.2 を達成したが、その時点で残っていた課題:

| 課題 | 状況 |
|---|---|
| **RL の不安定さ** | GSPO/GRPO 単独訓練は計算高・不安定、reward hacking の恐れ |
| **推論コスト** | 高解像度・複数画像・動画で **ViT が LLM をブロック**、実世界応用のボトルネック |
| **MoE 未活用** | InternVL 3 は全 dense モデル、**MoE のスケーラビリティ** を活用できていない |

### この論文の位置づけ

**InternVL シリーズ第 8 世代、商用フロンティアとの最終決戦**。発表時期（2025 Aug）は **GPT-5（2025 Aug 7）の直後**、**「GPT-5 との差をオープンソースで最小化する」** ことを明示的目標に設定。

```
[InternVL シリーズの哲学進化]
1.0 (2023-12): QLLaMA + 視覚エンコーダスケール
1.5 (2024-04): QLLaMA 廃止 + dynamic 4K + bilingual
Mini (2024-10): 軽量化 + ドメイン適応
2.5 (2024-12): Progressive Scaling + Test-Time Scaling、MMMU 70 突破
3 (2025-04): Native Multimodal Pre-Training + V2PE + MPO + VisualPRM、MMMU 72.2
3.5 (2025-08): ★ Cascade RL + MoE + ViR + DvD、MMMU 77.7、GPT-5 との差 3.9%
```

---

## 提案手法

### モデルファミリー（9 サイズ × dense + MoE + Flash）

```
[Dense Models]
   1B   = InternViT-300M + Qwen3-0.6B
   2B   = InternViT-300M + Qwen3-1.7B
   4B   = InternViT-300M + Qwen3-4B
   8B   = InternViT-300M + Qwen3-8B
   14B  = InternViT-300M + Qwen3-14B
   38B  = InternViT-6B   + Qwen3-32B

[MoE Models（新規）]
   20B-A4B  = InternViT-300M + GPT-OSS-20B    （activated 4B）
   30B-A3B  = InternViT-300M + Qwen3-30B-A3B  （activated 3B）
   241B-A28B = InternViT-6B  + Qwen3-235B-A22B（activated 28B、最大）

[Flash 効率版（各サイズに対応）]
   ViR (Visual Resolution Router) を統合、視覚トークン 50% 削減
```

**特徴**:
- **LLM は Qwen3 系に統一**（[[entities/internvl-3|InternVL 3]] は Qwen2.5、本作は Qwen3 へ）
- **OpenAI の GPT-OSS-20B を統合**（注目すべき選択）
- **3 つの MoE モデル** で MoE スケーラビリティを本格活用

### 1. Cascade RL（本論文の最重要貢献）

**「offline RL → online RL」の 2 段階 coarse-to-fine 戦略**。

```
Stage 1 (offline RL): MPO で warm-up
   - データ: MMPR-v1.2（200K ペア）
   - 損失: DPO + BCO + LM の混合
   - 利点: 安定、効率的、reward hacking なし
   - 効果: SFT から +3-6 ポイント

Stage 2 (online RL): GSPO で精緻化
   - データ: MMPR-Tiny（70K クエリ、accuracy 0.2-0.8 でフィルタリング）
   - reference model 制約なし
   - importance ratio: トークン単位の geometric mean
   - 利点: 性能上限を押し上げ
   - 効果: MPO から +2-5 ポイント
```

**Cascade RL の効果（表 15）**:

| Model | Instruct (SFT) | + MPO | + Cascade RL | SFT→Cascade |
|---|---|---|---|---|
| InternVL3.5-2B | 38.5 | 41.1 | **50.7** | **+12.2** |
| InternVL3.5-8B | 53.6 | 56.3 | **60.3** | +6.7 |
| InternVL3.5-38B | 57.7 | 61.4 | **66.0** | +8.3 |
| InternVL3.5-241B-A28B | 60.4 | 62.4 | **66.9** | +6.5 |

**Cascade RL vs GSPO 単独（表 16、InternVL3.5-8B）**:

| Method | GPU Hours | Overall |
|---|---|---|
| Instruct | – | 53.6 |
| + MPO | 0.3K | 56.3 |
| + GSPO (1 ep) | 5.5K | 57.3 |
| + GSPO (2 ep) | 11.0K | 58.2 |
| **+ Cascade RL** | **5.8K** | **60.3** |

**GSPO の半分の時間で +2.1 ポイント上回る**。

> **Cascade RL の天才的なアイデア**: 「**MPO は性能上限低いが安定・高速、GSPO は性能上限高いが不安定・遅い → 順番に適用**」。MPO で warm-up したモデルは **高品質 rollouts を生成** → GSPO の online RL を **少ない episode で収束** させられる。これは **「コスト × 性能」の Pareto 改善**。

### 2. Visual Resolution Router (ViR) + Visual Consistency Learning (ViCO)

**InternVL 3.5 のもう一つの中核貢献**: 視覚トークンを動的に削減して推論加速。

```
[従来の Dynamic High Resolution]
   入力画像 → 448×448 タイル × N → 各タイル 1024 token → Pixel Shuffle で 256 token
   ↑ すべての patch を同じ圧縮率で扱う

[ViR (Visual Resolution Router)]
   入力画像 → 448×448 タイル × N → 各タイル 1024 token
   ↓ patch ごとに **意味的内容** を評価
   ├─ semantic-rich patch → 256 token（1/4 圧縮、Pixel Shuffle）
   └─ semantic-poor patch → 64 token（1/16 圧縮、higher rate Pixel Shuffle）
```

**Dynamic High Resolution が画像の幅と高さ（spatial）の観点で分割するのに対し、ViR は意味的内容（semantic）の観点から適応性を導入**。

#### ViR を訓練する ViCO の 2 ステージ

**Stage 1: Consistency training（一貫性訓練）**。
$$\mathcal{L}_{\text{ViCO}} = \mathbb{E}_{\xi \sim \mathcal{R}}\left[\frac{1}{N}\sum_i \text{KL}(\pi_{\theta_{ref}}(y_i | y_{<i}, I) \| \pi_{\theta_{policy}}(y_i | y_{<i}, I_\xi))\right]$$
- 凍結された InternVL3.5 を reference として、policy の出力分布を整列
- $\xi \in \{1/4, 1/16\}$ から uniform sampling
- $\xi=1/4$ で 256 tokens（high res）、$\xi=1/16$ で 64 tokens（low res）
- reference は常に $\xi=1/4$ → 1/16 でも 1/4 と同じ応答を生成できるよう学習

**Stage 2: Router training（ルータ訓練）**。
- patch ごとに loss ratio を計算: $r_i = \mathcal{L}_{\text{ViCO}}(y_i | I_{1/16}) / \mathcal{L}_{\text{ViCO}}(y_i | I_{1/4})$
- $r_i$ が大 → 圧縮の影響大 → high res（256 token）を選択
- $r_i$ が小 → 圧縮の影響小 → low res（64 token）でも OK
- バイナリ分類器（ViR）を cross-entropy で訓練、**target 分布をバランス**（k-th percentile による動的閾値 $\tau$）
- **MLLM 凍結、ViR のみ訓練**

**結果**: **視覚トークン 50% 削減で性能 99% 維持**（表 17）

| Model | Original | Flash (ViR) | 維持率 |
|---|---|---|---|
| InternVL3.5-8B | 80.2 | 79.8 | 99.5% |
| InternVL3.5-38B | 83.9 | 83.4 | 99.4% |
| InternVL3.5-235B-MoE | 85.0 | 84.5 | 99.4% |

### 3. Decoupled Vision-Language Deployment (DvD)

**Vision と Language を別 GPU に展開する推論時最適化**。

```
[従来の展開]
   1 つの GPU 上で ViT → MLP → LLM を逐次実行
   ↑ ViT と LLM のサイズ・計算パターンが大きく違うため、互いをブロック

[DvD]
   Vision Server: ViT + MLP + (ViR for Flash)
   Language Server: LLM のみ
   ↓
   非同期 3 段階パイプライン:
      1. Vision Server で ViT 計算
      2. BF16 視覚特徴を TCP（or RDMA）で転送
      3. Language Server で LLM の prefilling + decoding
   ↓
   ViT の計算が LLM の prefilling と decoding に隠れる
```

**DvD + ViR の推論加速（表 18、InternVL3.5-38B、Request Throughput）**:

| Resolution | Baseline | +DvD | +DvD+ViR |
|---|---|---|---|
| 448 | 12.39 | 14.69 (1.19×) | 18.62 (1.50×) |
| **896** | 2.71 | 5.06 (1.87×) | **10.97 (4.05×)** |
| 1344 | 1.48 | 2.92 (1.97×) | 5.14 (3.47×) |

**解像度が高いほど DvD の効果が大**（ViT の計算ボトルネックが目立つため）。**896 解像度で 4.05× 加速**。

> **DvD の系統的意義**: これは **MLLM のシステムレベル最適化** であり、**「アルゴリズムではなくインフラで性能を上げる」** という新方向。MoE 時代の大規模 MLLM 展開で必須技術になる可能性。

### 4. 訓練データの拡張

InternVL 3 から:
- **SFT**: 21.7M → **56M サンプル**（+2.6×）、130B トークン、テキスト:マルチモーダル = 1:3.5
- **3 つの新スキル**: **GUI 対話、embodied 対話、SVG 理解・生成**
- **Thinking モードのマルチモーダル推論データ** を追加（大規模 reasoning model から rollouts、厳格フィルタリング）
- 事前学習: 116M サンプル、250B トークン、テキスト:マルチモーダル = **1:2.5**（InternVL 3 の 1:3 より若干テキスト寄り）

---

## 実験結果と知見

### 全体比較（表 2）

InternVL3.5-241B-A28B が GPT-5 と互角:

| Category | InternVL3.5-241B-A28B | GPT-5 | Δ |
|---|---|---|---|
| General Overall | **74.1** | 74.0 | **+0.1** |
| Reasoning Overall | 67.1 | **74.3** | -7.2 |
| Text Overall | 85.3 | **91.3** | -6.0 |
| Agentic Overall | **66.2** | – | – |
| **Aggregate** | **74.0**（推定）| **78.0** | **3.9% gap** |

### MMMU で 77.7（オープンソース新 SOTA）

| Model | MMMU |
|---|---|
| InternVL3-78B | 72.2 |
| **InternVL3.5-30B-A3B** | **75.6** |
| **InternVL3.5-38B** | **76.9** |
| **InternVL3.5-241B-A28B** | **77.7** |
| Claude-3.7-Sonnet | 75.0 |
| Gemini-2.5-Pro | 74.7 |
| GPT-5-nano | 72.6 |
| **GPT-5** | **84.2** |

**InternVL3.5-30B-A3B (30B 活性 3B) が GPT-5-nano 超え**。

### MathVista で 82.7（GPT-5 81.9 超え）

InternVL3.5-241B-A28B **MathVista 82.7** が **GPT-5 81.9 を +0.8 超え**。

### VSI-Bench 空間推論で GPT-5 を圧倒

| Model | VSI-Bench |
|---|---|
| **InternVL3.5-241B-A28B** | **69.5** |
| **InternVL3.5-38B** | **66.3** |
| Gemini-2.5-Pro | 47.8 |
| Claude-3.7-Sonnet | 47.0 |
| GPT-5 | 37.5 |

**3D 空間推論で +20-30 ポイント圧倒**。embodied AI への大きな前進。

### WindowsAgentArena / WebArena-Lite-v2 で GPT-4o を圧倒

| Benchmark | InternVL3.5-241B-A28B | GPT-4o |
|---|---|---|
| WindowsAgentArena | – | **3.5** |
| WebArena-Lite-v2 | **11.7** | 1.9 |

**GPT-4o より 6× 高いスコア**、GUI agent としての実用性を実証。

### 言語能力: Qwen3 base を 16 ベンチ中 15 で上回る

| Base LLM | Qwen3 | InternVL3.5 | Δ |
|---|---|---|---|
| 0.6B → 1B | 38.1 | **44.8** | **+6.7** |
| 1.7B → 2B | 52.5 | **61.0** | **+8.5** |
| 8B → 8B | 74.6 | **77.8** | **+3.2** |
| 14B → 14B | 77.8 | **80.7** | **+2.9** |
| 32B → 38B | 79.0 | **83.7** | **+4.7** |
| 235B → 241B | 85.3 | **87.6** | **+2.3** |

**[[entities/internvl-3|InternVL 3]] の「マルチモーダル化で言語が強くなる」発見を InternVL 3.5 で継続実証**、大型モデルでも一貫した +2-9 ポイント改善。

### ViR の Flash 版は 50% トークン削減で性能 99% 維持

InternVL3.5-Flash は **実用展開での標準** となる可能性。

---

## 限界・批判的視点

### GPT-5 とのまだ 3.9% の差

- Aggregate で 3.9% 差は **オープンソースとしては最小** だが、**まだ商用フロンティアを超えてはいない**
- 特に **Reasoning Overall** で -7.2、**Text Overall** で -6.0 と差が残る

### MoE モデルでも 241B の計算コスト

- 「activated 28B」だが、**メモリには 241B 全部 load 必要** → 巨大 GPU クラスタが必要
- **個人利用は事実上不可能**、エンタープライズ向け

### GSPO の reference model なしの理論的根拠

- 論文は GSPO（Geometric mean Sequence-level PPO）の reference 制約なしを採用したが、**理論的安定性の証明は本論文外**
- 「empirically more effective」のみ、長期的安定性は未検証

### ViR の patch-aware compression の意味的妥当性

- 「semantic richness」の判定基準は **loss ratio による heuristic** で、人間の直感と一致するかは未検証
- 重要 patch を誤って 1/16 圧縮するリスク

### Cascade RL の汎化性

- MMPR-v1.2 + MMPR-Tiny データに依存
- 他ドメイン（医療、法律、コード等）へのスケーラビリティは未検証

### Flash の追加訓練コスト

- ViCO + Router training の追加段階が必要
- 既存 InternVL3.5 から Flash 化するコストは無視できない

---

## InternVL シリーズでの位置づけ

```
[2023-12] InternVL 1.0 ([[entities/internvl]])
   └─ QLLaMA + 6B 視覚

[2024-04] InternVL 1.5 ([[entities/internvl-1-5]])
   └─ QLLaMA 廃止 + MLP、26B

[2024-07] InternVL 2.0
   └─ 1B-76B、複数画像 + 動画

[2024-10] Mini-InternVL ([[entities/mini-internvl]])
   └─ InternViT-300M 蒸留、軽量

[2024-11] MPO ([[entities/mpo]])
   └─ Mixed Preference Optimization、CoT 問題解決

[2024-12] InternVL 2.5 ([[entities/internvl-2-5]])
   └─ 7 サイズ、Progressive Scaling、MMMU 70.1

[2025-04] InternVL 3 ([[entities/internvl-3]])
   ├─ Native Multimodal Pre-Training、V2PE、MPO、VisualPRM
   └─ MMMU 72.2、7 サイズ

[2025-08] InternVL 3.5 ←（本ページ）
   ├─ Cascade RL（MPO + GSPO）
   ├─ MoE（20B-A4B / 30B-A3B / 241B-A28B）
   ├─ ViR + ViCO（視覚トークン 50% 削減）
   ├─ DvD（4.05× 推論加速）
   ├─ 9 サイズ × dense + MoE × Flash
   └─ MMMU 77.7、GPT-5 との差 3.9%
```

**InternVL シリーズ第 8 世代、商用フロンティアとの差を最小化したオープンソース MLLM の最新版**。

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **InternVL 3.5** | – | InternVL シリーズ第 8 世代（2025 Aug、本論文） |
| **InternVL3.5-1B/2B/4B/8B/14B/38B** | – | Dense モデル（Qwen3 ベース） |
| **InternVL3.5-20B-A4B** | – | MoE（GPT-OSS-20B、activated 4B） |
| **InternVL3.5-30B-A3B** | – | MoE（Qwen3-30B-A3B、activated 3B） |
| **InternVL3.5-241B-A28B** | – | MoE（Qwen3-235B-A22B、activated 28B、最大モデル） |
| **InternVL3.5-Flash** | – | ViR を統合した効率版 |
| **InternViT-300M / InternViT-6B** | – | InternVL シリーズ共通の視覚エンコーダ |
| **Qwen3 series** | – | Alibaba の最新 LLM（0.6B/1.7B/4B/8B/14B/32B/30B-A3B/235B-A22B） |
| **GPT-OSS-20B** | – | OpenAI の公開 MoE LLM |
| **Cascade RL** | Cascade Reinforcement Learning | offline RL + online RL の 2 段階 RL（本論文の中核） |
| **MPO** | Mixed Preference Optimization | [[entities/mpo\|offline RL アルゴリズム]] (DPO + BCO + LM) |
| **GSPO** | Geometric mean Sequence-level PPO | online RL アルゴリズム（reference model 制約なし） |
| **GRPO** | Group Relative Policy Optimization | DeepSeek 系の online RL |
| **DPO / BCO** | – | [[entities/mpo\|MPO の構成要素]] |
| **MMPR-v1.2** | – | [[entities/mmpr\|MPO 訓練データ]]（200K ペア） |
| **MMPR-Tiny** | – | MMPR-v1.2 から accuracy 0.2-0.8 でフィルタリング（70K クエリ）、online RL 用 |
| **Reward Hacking** | – | RL モデルが reward を gaming する現象 |
| **Importance Sampling Ratio** | – | PPO 系の policy ratio、GSPO ではトークン単位の geometric mean |
| **Advantage** | – | RL の advantage function（正規化された報酬） |
| **ViR** | Visual Resolution Router | patch ごとに圧縮率を動的選択するルータ（本論文の中核） |
| **ViCO** | Visual Consistency Learning | ViR 訓練用の 2 段階手法 |
| **Pixel Shuffle / Unshuffle** | – | 空間 → チャンネル変換（[[entities/internvl-1-5\|InternVL 1.5 から継承]]、4 patches → 1 token） |
| **Loss Ratio $r_i$** | – | ViR の target 生成用、$\mathcal{L}_{1/16}/\mathcal{L}_{1/4}$ |
| **patch router** | – | ViR の binary classifier（cross-entropy 訓練） |
| **k-th percentile threshold $\tau$** | – | ViR の動的閾値（target 分布バランス） |
| **DvD** | Decoupled Vision-Language Deployment | ViT/MLP を vision server、LLM を language server に分離（本論文の中核） |
| **Vision Server / Language Server** | – | DvD の 2 つのサーバ |
| **BF16 視覚特徴 TCP/RDMA 通信** | – | DvD の特徴転送方法 |
| **非同期 3 段階パイプライン** | – | DvD の Vision → Transfer → Language の重ね合わせ実行 |
| **InternEVO** | – | InternVL 3.5 訓練フレームワーク（InternVL 3 から継承） |
| **XTuner** | – | InternVL 3.5 の主訓練フレームワーク（FSDP + FP8 + FlashAttention-3） |
| **DeepGEMM** | – | FP8 GEMM カーネル（DeepSeek 提案） |
| **liger-kernel** | – | fused cross-entropy operator |
| **FlashAttention-3** | – | 高速 attention 実装 |
| **TMA-Adaptive FP8 Grouped GEMM** | – | MoE 訓練用の特殊カーネル |
| **verl** | – | online RL 段階のコードベース |
| **window attention with sink** | – | GPT-OSS-20B の特殊 attention（InternVL3.5-20B-A4B で Triton 版実装） |
| **TTS** | Test-Time Scaling | Deep Thinking + Parallel Thinking |
| **Deep Thinking** | – | "Thinking" モード起動、step-by-step 推論 |
| **Parallel Thinking** | – | Best-of-N + VisualPRM critic |
| **VisualPRM-v1.1** | – | InternVL 3.5 の critic モデル（[[entities/internvl-3\|InternVL 3]] の VisualPRM 後継） |
| **Thinking Mode** | – | Qwen3 等の reasoning モード |
| **GPT-5** | – | OpenAI の最新フロンティア MLLM（2025 Aug） |
| **GPT-5-nano** | – | GPT-5 の小型版 |
| **Claude-3.7-Sonnet** | – | Anthropic Claude-3 系列の reasoning 強化版 |
| **Gemini-2.5-Pro** | – | Google Gemini-2.5 系列 |
| **GLM-4.5V** | – | 智谱 AI の最新マルチモーダル MLLM |
| **Step-3** | – | StepFun の MLLM |
| **Kimi-VL-A3B-2506** | – | Moonshot Kimi-VL の MoE 版 |
| **MiMo-VL-RL-8B** | – | Xiaomi の RL 強化 MLLM |
| **Keye-VL-8B** | – | 快手 Keye の MLLM |
| **Ovis-2B/4B/8B/2-16B/2-34B** | – | Ovis シリーズ MLLM |
| **MiniCPM-V-4 / MiniCPM-o-2.6** | – | Tsinghua の軽量 MLLM |
| **Skywork-R1V3-38B** | – | 昆仑万维 Skywork の reasoning MLLM |
| **QvQ-72B-Preview** | – | Alibaba の reasoning MLLM |
| **Doubao-1.5-Pro** | – | ByteDance Doubao |
| **WindowsAgentArena** | – | Windows GUI エージェントベンチ |
| **WebArena-Lite-v2** | – | Web エージェントベンチ |
| **OSWorld / OSWorld-G** | – | OS-level GUI エージェントベンチ |
| **Seed1.5-VL** | – | ByteDance の MLLM |
| **UI-TARS-72B** | – | ByteDance の GUI 特化 MLLM |
| **SGP-Bench** | – | SVG 理解ベンチ |
| **SArena-Icon** | – | SVG 生成ベンチ（Text2SVG / Img2SVG） |
| **VSI-Bench** | Visual-Spatial Intelligence | 3D 空間推論ベンチ |
| **ERQA** | – | embodied reasoning ベンチ |
| **SpaCE-10** | – | 空間理解ベンチ |
| **OmniSpatial** | – | 包括的空間ベンチ |
| **MTVQA** | – | 9 言語 text-centric VQA |
| **MathVision / MathVerse / DynaMath / WeMath / LogicVista / OlympiadBench** | – | 推論・数学ベンチ |
| **AIME24 / AIME25** | – | 数学オリンピックレベル |
| **MATH500 / GPQA / MMLU-Pro** | – | テキスト推論ベンチ |
| **CharXiv** | – | 科学論文 chart 理解（RQ/DQ） |
| **R-Bench** | – | 実世界画像劣化頑健性 |
| **MIRB** | – | 複数画像推論 |
| **MuirBench / MMIU / Mantis-Eval** | – | 複数画像ベンチ |
| **SEED-2-Plus** | – | text-rich 視覚タスク |
| **VCR-EN-Easy** | – | Visual Caption Restoration |
| **HallusionBench / POPE / CRPE** | – | hallucination ベンチ |
| **InternLM2.5** | – | Shanghai AI Lab の LLM（テキスト専用ベンチで比較） |
| **DeepSeek-Coder-V2-16B** | – | DeepSeek のコード MLLM |
| **DeepSeek-V3-671B-A37B** | – | DeepSeek の最大 MoE |
| **Llama-3.1-8B/70B/405B / Llama-4-Scout/Maverick** | – | Meta Llama 系 |
| **GPT-OSS-20B** | – | OpenAI の公開 MoE LLM（InternVL3.5-20B-A4B が採用） |
| **Activated Parameters (A4B/A3B/A28B)** | – | MoE モデルの活性パラメータ数 |
| **Bo8 (Best-of-8)** | – | Parallel Thinking の N=8 |

---

## 関連ページ

- 翻訳: [[translations/internvl-3-5]]
- 主要エンティティ: [[entities/internvl-3-5]]
- 直接の祖: [[sources/internvl-3]] / [[entities/internvl-3]]（Native Pre-Training 哲学を継承）
- 後訓練の基盤: [[sources/mpo]] / [[entities/mpo]]（MPO は Cascade RL の Stage 1）/ [[entities/mmpr]]（MMPR-v1.2 + MMPR-Tiny）
- InternVL シリーズ: [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]] / [[entities/internvl-3-5]]
- 視覚エンコーダ: [[entities/internvit-300m]]（共有）
- 関連視覚エンコーダ系統: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]]
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]
