---
type: entity
entity_kind: model
aliases: [InternVL 3.5, InternVL3.5, InternVL3.5-1B, InternVL3.5-2B, InternVL3.5-4B, InternVL3.5-8B, InternVL3.5-14B, InternVL3.5-38B, InternVL3.5-20B-A4B, InternVL3.5-30B-A3B, InternVL3.5-241B-A28B, InternVL3.5-Flash]
tags: [mllm, vllm, internvl-3-5, cascade-rl, mpo, gspo, vir, dvd, moe, qwen3, opengvlab, internvl-series]
related: ["[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/zero-shot-transfer]]", "[[concepts/alignment-tuning]]", "[[entities/internvl]]", "[[entities/internvl-1-5]]", "[[entities/mini-internvl]]", "[[entities/internvl-2-5]]", "[[entities/internvl-3]]", "[[entities/mpo]]", "[[entities/mmpr]]", "[[entities/internvit-300m]]"]
sources: ["[[sources/internvl-3-5]]"]
updated: 2026-05-29
---

# InternVL 3.5 — Cascade RL + MoE + ViR + DvD で 4.05× 加速、GPT-5 との差 3.9%

## 概要

**InternVL 3.5** = OpenGVLab Shanghai AI Lab の **InternVL シリーズ第 8 世代**（2025 Aug）。**Cascade RL + MoE スケーリング + 効率化（ViR + DvD）** という 3 軸で前進、**GPT-5 との差をオープンソース最小の 3.9%** まで縮める。

- 論文: "InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency"
- arXiv: 2508.18265（2025 Aug、technical report）
- HuggingFace: <https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B>
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- 詳細: [[sources/internvl-3-5]] / 翻訳: [[translations/internvl-3-5]]

**9 サイズ × dense + MoE × Flash 効率版** = **18+ モデルのスイート**。

---

## モデルファミリー（9 サイズ）

### Dense Models（6 サイズ）

| Model | Vision | LLM | Vision Param | LLM Param | Total |
|---|---|---|---|---|---|
| **InternVL3.5-1B** | InternViT-300M | Qwen3-0.6B | 0.3B | 0.8B | 1.1B |
| **InternVL3.5-2B** | InternViT-300M | Qwen3-1.7B | 0.3B | 2.0B | 2.3B |
| **InternVL3.5-4B** | InternViT-300M | Qwen3-4B | 0.3B | 4.4B | 4.7B |
| **InternVL3.5-8B** | InternViT-300M | Qwen3-8B | 0.3B | 8.2B | 8.5B |
| **InternVL3.5-14B** | InternViT-300M | Qwen3-14B | 0.3B | 14.8B | 15.1B |
| **InternVL3.5-38B** | InternViT-6B | Qwen3-32B | 5.5B | 32.8B | 38.4B |

### MoE Models（3 サイズ、本作の新規貢献）

| Model | Vision | LLM | Total | Activated |
|---|---|---|---|---|
| **InternVL3.5-20B-A4B** | InternViT-300M | **GPT-OSS-20B** | 21.2B | **4B** |
| **InternVL3.5-30B-A3B** | InternViT-300M | Qwen3-30B-A3B | 30.8B | **3B** |
| **InternVL3.5-241B-A28B** | InternViT-6B | Qwen3-235B-A22B | **240.7B** | **28B** |

**特徴**:
- **LLM が Qwen3 系に統一**（InternVL 3 は Qwen2.5）
- **OpenAI の GPT-OSS-20B を採用**（21.2B は OpenGVLab 系列で初の GPT-OSS 統合）
- **241B-A28B が最大、activated 28B で実用効率**

### Flash 効率版（各サイズに対応）

```
[Flash 版の構造]
   ViR (Visual Resolution Router) を統合
   ↓ patch ごとに 256 token (1/4) or 64 token (1/16) を動的選択
   視覚トークンを 50% 削減
   ↓
   性能 99% 維持、推論 4.05× 加速（DvD と組み合わせで）
```

---

## アーキテクチャ（ViT-MLP-LLM、InternVL シリーズ継承）

```
[InternVL3.5（基本版）]
画像入力（任意解像度）
   ↓ Dynamic High Resolution（35 種アスペクト比 × 1-12 タイル）
448×448 タイル × N + サムネイル
   ↓ InternViT-300M / InternViT-6B
1024 visual tokens / tile
   ↓ Pixel Shuffle（1/4 圧縮）
256 visual tokens / tile
   ↓ 2 層 MLP プロジェクタ
LLM 埋め込み空間
   ↓ Qwen3 / GPT-OSS
出力テキスト

[InternVL3.5-Flash（ViR 追加）]
画像入力
   ↓ Dynamic High Resolution + ViR（patch-aware semantic compression）
448×448 タイル × N + サムネイル
   ↓ InternViT
1024 visual tokens / tile
   ↓ Patch Router（ViR）が semantic richness 評価
   ├─ rich patch → Pixel Shuffle (1/4) → 256 tokens
   └─ poor patch → Pixel Shuffle (1/16) → 64 tokens
   ↓
平均 50% 削減した visual tokens
   ↓ MLP → LLM
出力テキスト
```

---

## 訓練パイプライン（3 段階 + Flash の追加段階）

### Stage 1: Native Multimodal Pre-Training

| 項目 | 内容 |
|---|---|
| 訓練対象 | **ViT + MLP + LLM 全層共同訓練** |
| データ | 116M サンプル、**250B トークン** |
| 比率 | テキスト : マルチモーダル = **1 : 2.5** |
| 損失 | text-only NTP loss + square averaging ($w_i = 1/N^{0.5}$) |
| 最大系列長 | **32K トークン** |
| Augmentation | Random JPEG compression |

[[entities/internvl-3|InternVL 3]] の Native Pre-Training パラダイムを継承。データ比率がやや変化（1:3 → **1:2.5**）。

### Stage 2: Supervised Fine-Tuning (SFT)

| 項目 | 内容 |
|---|---|
| 訓練対象 | 全モデル |
| データ | **56M サンプル、130B トークン**（[[entities/internvl-3\|InternVL 3]] の 21.7M から +2.6×） |
| 比率 | テキスト : マルチモーダル = 1 : 3.5 |
| 文脈長 | 32K |
| データソース | InternVL3 指示データ + Thinking モード推論 + 新スキル（GUI/embodied/SVG） |

**Thinking モードの推論データ**: 大規模 reasoning model から rollouts、**厳格フィルタリング**（思考の明確性、redundancy 除去、フォーマット一貫性）。

### Stage 3: Cascade RL（本作の中核）

#### Stage 3a: MPO（offline RL、warm-up）

| 項目 | 内容 |
|---|---|
| 損失 | $\mathcal{L}_{\text{MPO}} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$ |
| $\mathcal{L}_p$ | **DPO** loss |
| $\mathcal{L}_q$ | **BCO** loss |
| $\mathcal{L}_g$ | **LM** loss |
| データ | **MMPR-v1.2**（約 200K ペア） |
| GPU 時間（8B モデル） | ~0.3K hours |

詳細: [[entities/mpo]] / [[sources/mpo]]。

#### Stage 3b: GSPO（online RL、精緻化）

| 項目 | 内容 |
|---|---|
| アルゴリズム | **GSPO（Geometric mean Sequence-level PPO）** |
| reference model | **なし** |
| Advantage | normalized reward: $\widehat{A}_i = (r - \text{mean}) / \text{std}$ |
| Importance Ratio | **トークン単位の geometric mean**: $s_i = (\pi_\theta/\pi_{\text{old}})^{1/|y_i|}$ |
| 目的関数 | $\mathcal{L}_{\text{GSPO}}$ = clip 関数で min |
| データ | **MMPR-Tiny**（70K クエリ、accuracy 0.2-0.8 でフィルタリング） |
| GPU 時間（8B モデル） | ~5.5K hours（1 episode） |

#### Cascade RL（合計）

| Method | GPU Hours (8B) | Overall |
|---|---|---|
| SFT only | – | 53.6 |
| + MPO | 0.3K | 56.3 |
| + GSPO (1 ep) | 5.5K | 57.3 |
| + GSPO (2 ep) | 11.0K | 58.2 |
| **+ Cascade RL** | **5.8K** | **60.3** |

**Cascade RL は GSPO 単独の半分の時間で +2.1 ポイント上回る**。

### Stage 4: ViCO（Flash 版のみ）

#### Stage 4a: Consistency training

| 項目 | 内容 |
|---|---|
| 損失 | KL divergence between $\xi=1/4$ (256 tokens) と $\xi=1/16$ (64 tokens) |
| reference | 凍結 InternVL3.5（常に $\xi=1/4$） |
| データ | SFT データ同じ（性能保持） |

#### Stage 4b: Router training

| 項目 | 内容 |
|---|---|
| target | loss ratio $r_i = \mathcal{L}(I_{1/16}) / \mathcal{L}(I_{1/4})$ |
| threshold | **k-th percentile** で動的計算、balanced distribution |
| 訓練対象 | **ViR のみ**（MLLM 凍結） |
| データ | OCR + VQA の SFT サブセット |

---

## Cascade RL の効果

**全モデルサイズで SFT → Cascade RL の改善**（表 15）:

| Model | SFT | MPO | Cascade RL | SFT→Cascade |
|---|---|---|---|---|
| InternVL3-1B（前世代）| – | – | 25.1 | – |
| **InternVL3.5-1B** | 25.7 | 29.1 | **33.8** | +8.1 |
| **InternVL3.5-2B** | 38.5 | 41.1 | **50.7** | **+12.2** |
| **InternVL3.5-4B** | 49.4 | 52.3 | **57.4** | +8.0 |
| **InternVL3.5-8B** | 53.6 | 56.3 | **60.3** | +6.7 |
| **InternVL3.5-14B** | 54.9 | 56.6 | **62.0** | +7.1 |
| **InternVL3.5-30B-A3B** | 52.7 | 56.1 | **59.0** | +6.3 |
| **InternVL3.5-38B** | 57.7 | 61.4 | **66.0** | +8.3 |
| **InternVL3.5-241B-A28B** | 60.4 | 62.4 | **66.9** | +6.5 |

**Cascade RL は SFT 単独より平均 +7-8 ポイント改善、MPO 単独より +3-5 ポイント改善**。スケーラブル（小さいモデルほど効果大、+12.2 が最大）。

---

## ViR + DvD の推論加速

### Flash の性能維持（表 17）

| Model | Original Overall | Flash Overall | 維持率 |
|---|---|---|---|
| InternVL3.5-8B | 80.2 | 79.8 | **99.5%** |
| InternVL3.5-38B | 83.9 | 83.4 | **99.4%** |
| InternVL3.5-30B-MoE | 82.6 | 82.2 | **99.5%** |
| InternVL3.5-235B-MoE | 85.0 | 84.5 | **99.4%** |

**視覚トークン 50% 削減で性能 99%以上維持**。

### DvD + ViR の推論加速（表 18、Request Throughput）

**InternVL3.5-38B（解像度別）**:

| Resolution | Baseline | +DvD | +DvD+ViR |
|---|---|---|---|
| 448 | 12.39 | 14.69 (1.19×) | 18.62 (1.50×) |
| **896** | 2.71 | 5.06 (1.87×) | **10.97 (4.05×)** |
| 1344 | 1.48 | 2.92 (1.97×) | 5.14 (3.47×) |

**InternVL3.5-241B-A28B（解像度別）**:

| Resolution | Baseline | +DvD | +DvD+ViR |
|---|---|---|---|
| 448 | 11.29 | 14.05 (1.24×) | 20.84 (1.85×) |
| 896 | 2.54 | 4.73 (1.86×) | 8.81 (3.47×) |
| 1344 | 1.37 | 2.75 (2.01×) | 4.27 (3.12×) |

**解像度が高いほど DvD の効果が大**、最大 **896 解像度で 4.05×** 加速。

---

## 主要結果

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

**InternVL3.5-30B-A3B が GPT-5-nano 超え**（30B 活性 3B というコスト効率）。

### MathVista で GPT-5 を上回る

| Model | MathVista |
|---|---|
| GPT-5 | 81.9 |
| **InternVL3.5-241B-A28B** | **82.7** (+0.8) |
| **InternVL3.5-38B** | 81.9 (互角) |
| GLM-4.5V | 84.6 |

### Multimodal Reasoning Overall

| Model | Overall |
|---|---|
| InternVL3-78B | 54.6 |
| **InternVL3.5-241B-A28B** | **66.9** (+12.3) |
| **InternVL3.5-38B** | **66.0** |
| GPT-5 | **74.1** |

InternVL3 から **全モデルサイズで +10 ポイント以上の改善**。

### Parallel Thinking でさらに改善

| Model | w/o Bo8 | w/ Bo8 | Δ |
|---|---|---|---|
| InternVL3.5-1B | 33.8 | **43.3** | +9.5 |
| InternVL3.5-2B | 50.7 | **53.9** | +3.2 |
| InternVL3.5-4B | 57.4 | **60.0** | +2.6 |
| InternVL3.5-8B | 60.3 | **62.4** | +2.1 |
| InternVL3.5-38B | 66.0 | **67.1** | +1.1 |
| InternVL3.5-241B-A28B | 66.9 | **68.7** | +1.8 |

**小型モデルほど test-time scaling の効果大**（InternVL 3 の発見を継続）。

### VSI-Bench で GPT-5 を圧倒（空間推論）

| Model | VSI-Bench |
|---|---|
| **InternVL3.5-241B-A28B** | **69.5** |
| **InternVL3.5-38B** | **66.3** |
| **InternVL3.5-8B** | **56.3** |
| Gemini-2.5-Pro | 47.8 |
| Claude-3.7-Sonnet | 47.0 |
| GPT-5 | 37.5 |
| GPT-4o | 34.0 |

**InternVL3.5-8B (8B 活性) でも GPT-5 を +18.8 圧倒**。3D 空間推論で全商用モデルを大幅超え。

### GUI Agent: GPT-4o を圧倒

| Benchmark | InternVL3.5-241B-A28B | UI-TARS-72B | GPT-4o |
|---|---|---|---|
| ScreenSpot-v2 | 92.9 | 90.3 | – |
| WindowsAgentArena | – | – | **3.5** |
| WebArena-Lite-v2 | **11.7** | – | 1.9 |

### 言語能力: Qwen3 base を上回る（InternVL 3 の発見を継続）

| Base LLM | Qwen3 | **InternVL3.5** | Δ |
|---|---|---|---|
| Qwen3-0.6B → 1B | 38.1 | **44.8** | **+6.7** |
| Qwen3-8B → 8B | 74.6 | **77.8** | **+3.2** |
| Qwen3-235B-A22B → 241B | 85.3 | **87.6** | **+2.3** |

**16 ベンチ中 15 で Qwen3 base を上回る**（[[entities/internvl-3|InternVL 3]] と同じ「マルチモーダル化で言語が強くなる」現象）。

---

## アーキテクチャ比較

### vs InternVL 3

| 項目 | InternVL 3 | InternVL 3.5 |
|---|---|---|
| アーキテクチャ | ViT-MLP-LLM | **完全に同じ** |
| 視覚 | InternViT-V2.5 | **完全に同じ** |
| **LLM** | **Qwen2.5 base** | **Qwen3 base + GPT-OSS** |
| **MoE** | なし | **3 サイズ追加**（A4B/A3B/A28B） |
| サイズ範囲 | 1B-78B | **1B-241B** |
| 訓練後段階 | SFT + MPO | **SFT + Cascade RL (MPO + GSPO)** |
| Test-Time Scaling | CoT + VisualPRM | **Deep Thinking + Parallel Thinking + VisualPRM-v1.1** |
| 効率化 | なし | **ViR + DvD + Flash 版** |
| MMMU | 72.2 (78B) | **77.7 (241B)** |
| 推論 Overall | 54.6 | **66.9** (+12.3) |
| 推論速度 | 1× | **4.05× (896 res)** |

### vs GPT-5

| 項目 | GPT-5 | InternVL3.5-241B-A28B |
|---|---|---|
| パラメータ | 非公開 | 241B (activated 28B) |
| 公開 | クローズド | **オープン（MIT + Qwen3）** |
| General Overall | 74.0 | **74.1** (+0.1) |
| Reasoning Overall | **74.3** | 67.1 (-7.2) |
| Text Overall | **91.3** | 85.3 (-6.0) |
| MMMU | **84.2** | 77.7 (-6.5) |
| MathVista | 81.9 | **82.7** (+0.8) |
| VSI-Bench | 37.5 | **69.5** (+32.0) |
| WildVision | 77.4 | **82.8** (+5.4) |

**強み**: VSI-Bench / MathVista / WildVision / OCRBench
**弱み**: MMMU / Reasoning Overall / Text Overall

---

## InternVL シリーズ全体での位置づけ

```
[2023-12] InternVL 1.0 (CVPR 2024, [[entities/internvl]])
[2024-04] InternVL 1.5 ([[entities/internvl-1-5]])
[2024-07] InternVL 2.0
[2024-10] Mini-InternVL ([[entities/mini-internvl]])
[2024-11] MPO ([[entities/mpo]])
[2024-12] InternVL 2.5 ([[entities/internvl-2-5]])
[2025-04] InternVL 3 ([[entities/internvl-3]])
[2025-08] InternVL 3.5 ←（本ページ）
   ├─ Cascade RL（MPO + GSPO）
   ├─ MoE スケーリング（20B-A4B / 30B-A3B / 241B-A28B）
   ├─ ViR + ViCO（視覚トークン 50% 削減）
   ├─ DvD（4.05× 推論加速）
   ├─ 9 サイズ × dense + MoE × Flash = 18+ モデル
   └─ MMMU 77.7、GPT-5 との差 3.9%
```

**InternVL シリーズ第 8 世代、現時点最新版**。

---

## 公開リソース

公開済み（HuggingFace `OpenGVLab/` 配下）:

| モデル | サイズ | ライセンス |
|---|---|---|
| `InternVL3_5-1B/2B/4B/8B/14B/38B` | Dense 各サイズ | MIT + Qwen3 |
| `InternVL3_5-20B-A4B` | MoE（GPT-OSS-20B） | MIT + Apache 2.0 |
| `InternVL3_5-30B-A3B` | MoE（Qwen3-30B-A3B） | MIT + Qwen3 |
| `InternVL3_5-241B-A28B` | MoE（最大） | MIT + Qwen3 |
| 各サイズの `-Flash` 版 | ViR 統合効率版 | – |

**ライセンス**: MIT（コード）+ Qwen3 / GPT-OSS の各々のライセンス。

---

## 関連ページ

- 詳細解説: [[sources/internvl-3-5]]
- 翻訳: [[translations/internvl-3-5]]
- 直接の祖: [[entities/internvl-3]] / [[sources/internvl-3]]（Native Pre-Training 哲学を継承）
- 後訓練の基盤: [[entities/mpo]]（Cascade RL の Stage 1）/ [[entities/mmpr]]（MMPR-v1.2 + MMPR-Tiny）
- InternVL シリーズ: [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]]
- 視覚エンコーダ: [[entities/internvit-300m]]（共有）
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]]
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]
