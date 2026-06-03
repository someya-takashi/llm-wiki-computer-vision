---
type: entity
entity_kind: model
aliases: [InternVL, InternVL-1.0, InternViT-6B, QLLaMA, InternVL-C, InternVL-G, InternVL-Chat]
tags: [vision-language, foundation-model, llm-alignment, vit, transformer, opengvlab]
related: ["[[concepts/vision-transformer]]", "[[concepts/foundation-model]]", "[[concepts/contrastive-learning]]", "[[concepts/zero-shot-transfer]]", "[[concepts/weakly-supervised-pretraining]]", "[[entities/clip]]", "[[entities/perception-encoder]]", "[[entities/siglip]]", "[[entities/dinov2]]"]
sources: ["[[sources/internvl]]"]
updated: 2026-05-28
---

# InternVL — 6B 視覚基盤モデル + 8B 言語ミドルウェアの統合スイート

## 概要

**InternVL** = OpenGVLab Shanghai AI Lab を中心とする中国合同チームが 2023 年 12 月に発表した、**視覚エンコーダを 60 億パラメータにスケールアップし、80 億パラメータの言語ミドルウェアを介して LLM と整列させた視覚言語基盤モデル**。論文タイトルの「**Intern**」は所属の Shanghai AI Lab の Intern シリーズ（InternImage, InternVideo, InternLM）と統一されたブランド。

- 論文: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
- arXiv: 2312.14238（2023 年 12 月）/ CVPR 2024
- リポジトリ: <https://github.com/OpenGVLab/InternVL>
- 詳細: [[sources/internvl]] / 翻訳: [[translations/internvl]]

本論文は InternVL 1.0 として「InternViT-6B + QLLaMA + Vicuna」を提案する。後の InternVL 1.2 / 1.5 / 2.0 / 2.5 / 3 シリーズの起点。

---

## InternVL ファミリーの構成

InternVL は単一モデルではなく **3 つの異なる構成を持つスイート**。図 4 にあるとおり：

```
                  視覚エンコーダ        glue/middleware       LLM デコーダ
 ┌─ InternVL-C  ─ InternViT-6B  ─  (attention pooling)  ─  (text [EOS])
 │                                                              ↑
 │                                                      QLLaMA  ← text
 │
 ├─ InternVL-G  ─ InternViT-6B  ─  QLLaMA (96 queries)  ─  (text [EOS])
 │
 ├─ InternVL-Chat
 │  (w/o QLLaMA) ─ InternViT-6B  ─  MLP  ─  Vicuna-7B/13B  ← image+text
 │
 └─ InternVL-Chat
    (w/ QLLaMA)  ─ InternViT-6B  ─  QLLaMA  ─  Vicuna-13B  ← image+query+text
```

| 構成 | 主な用途 | 訓練段階 |
|---|---|---|
| **InternVL-C** | ゼロショット画像-テキスト検索、ゼロショット分類 | Stage 1 完了後 |
| **InternVL-G** | キャプショニング、対比 + 生成統合 | Stage 1 + 2 完了後 |
| **InternVL-Chat (w/o QLLaMA)** | 標準 VLLM 構成（LLaVA 互換）。MLP 1 層で接続 | Stage 1 + 3 |
| **InternVL-Chat (w/ QLLaMA)** | フル構成。QLLaMA で query を作って LLM に渡す | Stage 1 + 2 + 3 |

---

## 主要コンポーネント

### 1. InternViT-6B（視覚エンコーダ）

純粋な vanilla [[concepts/vision-transformer|ViT]]。設計はハイパーパラメータ網羅探索で決定（表 11 の variant 3）：

| 項目 | 値 |
|---|---|
| アーキテクチャ | vanilla ViT（特殊機構なし） |
| パラメータ | **5.9B**（ViT-22B の ~1/4、ViT-G の ~3×） |
| width | 3200 |
| depth | 48 |
| MLP hidden | 12800 |
| head 数 | 25 |
| head 次元 | 128 |
| patch size | 14（画像 224 → 16×16 patches） |
| 出力 | $F\in\mathbb{R}^{H/14\times W/14\times 3200}$ |
| 初期化 | ランダム（事前学習なし、Stage 1 で対比訓練） |
| 訓練データ | 4.98B image-text pairs（LAION-en + LAION-multi + LAION-COCO + COYO + Wukong + CC12M/3M + SBU） |

> **設計上の特徴**: 既存大型 ViT（ViT-G/e/22B, EVA-02-ViT-E, ViT-6.5B）と比べると **head 数が顕著に少ない（25 vs 16-48）** が、これは LAION 100M 上での 10K iter 対比訓練で「精度はほぼ同じだが安定性が高い」と判明したから。安定訓練重視の選択。

### 2. QLLaMA（言語ミドルウェア）

| 項目 | 値 |
|---|---|
| ベース | 多言語強化 LLaMA-7B（事前学習重みから初期化） |
| 新規追加 | **96 学習可能 query + cross-attention 層**（1B 追加） |
| 合計パラメータ | **8B** |
| 機能 | (1) 対比学習のテキスト encoder, (2) 生成学習の query encoder, (3) LLM 接続の glue |
| 初期化（query/cross-attn） | ランダム |
| 訓練段階 | Stage 1 で全パラメータ訓練、Stage 2 で新規 query+cross-attn のみ |

**QFormer との比較**:

| 項目 | QFormer (BLIP-2) | QLLaMA (InternVL) | 倍率 |
|---|---|---|---|
| パラメータ | 188M | 8B | **42×** |
| 初期化 | BERT-base | 多言語 LLaMA-7B | – |
| query 数 | 32 | 96 | 3× |
| 対比学習 | ❌（学習しない） | ✅ | – |
| 生成学習 | ✅ | ✅ | – |
| LLM 接続 | ✅ | ✅ | – |

---

## 訓練の 3 段階

| Stage | 目的 | 訓練可能パラメータ | データ量 | 損失 |
|---|---|---|---|---|
| **1: Contrastive** | InternViT を LLaMA-7B と対比整列 | InternViT-6B + LLaMA-7B（全可） | **4.98B 対** | 対称 InfoNCE（CLIP 流） |
| **2: Generative** | QLLaMA で生成能力 + リッチな画像特徴 | 96 query + cross-attn のみ（1B） | **1.03B 対** | ITC + ITM + ITG（BLIP-2 流） |
| **3: SFT** | LLM 接続 + 指示追従 | MLP + (optional QLLaMA) | **4M 指示** | autoregressive LM loss |

> **Stage 1 の規模感**: 4.98B 対比訓練は GPT-3 規模の計算予算。Shanghai AI Lab の InternLM クラスタを使い、InternLM の訓練インフラを流用したと考えられる。

---

## 主要結果

### 視覚知覚タスク（InternViT-6B 単体）

| タスク | データセット | 設定 | 結果 | 比較 |
|---|---|---|---|---|
| 画像分類 | ImageNet-1K | linear probe | **88.2** | EVA-01-CLIP-g 86.5 (1.1B) |
| 画像分類（IN 派生平均） | IN-V2/A/R/Sketch | linear probe | **82.5** | EVA-01-CLIP-g 79.1 |
| セマンティックセグ | ADE20K | linear probe | **47.2 mIoU** | ViT-22B 34.6 (+12.6) |
| セマンティックセグ | ADE20K | UperNet 凍結 | **54.9 mIoU** | ViT-22B 52.7 |
| セマンティックセグ | ADE20K | full tuning | **58.9 mIoU** | ViT-22B 55.3 |

### 視覚言語ゼロショットタスク（InternVL-C / G）

| タスク | データセット | InternVL-C/G | 既存 SoTA |
|---|---|---|---|
| 画像分類 | IN-1K (EN) | **83.2** | EVA-02-CLIP-E+ 82.0 |
| 画像分類（多言語） | IN-1K (avg 5 lang) | **64.0** | OpenCLIP-XLM-R-H 55.9 |
| 動画分類 (1F) | K400 top-1 | **65.9** | – |
| 動画分類 (8F) | K400 top-1 | **69.1** | ViCLIP 64.8 |
| 動画分類 (8F) | K700 top-1 | **60.6** | ViCLIP 54.3 |
| Flickr30K I2T R@1 (EN) | – | **95.7 (G)** | EVA-02-CLIP-E+ 93.9 |
| COCO I2T R@1 (EN) | – | **74.9 (G)** | EVA-02-CLIP-E+ 68.8 |
| Flickr30K-CN I2T R@1 (ZH) | – | **92.9 (G)** | AltCLIP-ViT-H 88.9 |
| **画像キャプショニング** | COCO Karpathy | CIDEr **128.2 (G)** | DreamLLM 115.4 |

### マルチモーダル対話（InternVL-Chat）

| 構成 | LLM | MME | POPE | VQAv2 | GQA | VizWiz | TextVQA |
|---|---|---|---|---|---|---|---|
| LLaVA-1.5 (CLIP-L 336) | Vicuna-7B | 1510.7 | 85.9 | 78.5 | 62.0 | 50.0 | 58.2 |
| LLaVA-1.5 (CLIP-L 336) | Vicuna-13B | 1531.3 | 85.9 | 80.0 | 63.3 | 53.6 | 61.3 |
| InternVL-Chat (IViT-6B + MLP) | Vicuna-7B | **1525.1** | **86.4** | **79.3** | **62.9** | **52.5** | 57.0 |
| InternVL-Chat (IViT-6B + MLP) | Vicuna-13B | **1546.9** | **87.1** | **80.2** | **63.9** | **54.6** | 58.7 |
| **InternVL-Chat (IViT-6B + QLLaMA)** | **Vicuna-13B** | **1586.4** | **87.6** | **81.2** | **66.6** | **58.5** | **61.5** |

**最重要観察**: 同じ LLaVA-1.5 学習レシピ（558K PT + 665K SFT）で、視覚エンコーダを **CLIP-L (304M) → InternViT-6B (5.9B)** に置換するだけで **MME +14 / POPE +0.5 / GQA +0.9 / VizWiz +2.5** の改善が出る。さらに QLLaMA を glue にすると **MME +50 / GQA +3.3 / VizWiz +4.9** の上乗せ。

---

## アーキテクチャ比較

### vs CLIP（[[entities/clip]]）

| 項目 | CLIP-L (OpenAI 2021) | InternVL-C |
|---|---|---|
| 視覚エンコーダ | ViT-L 304M | **InternViT-6B 5.9B**（19×） |
| テキストエンコーダ | 63M | **QLLaMA 8B**（127×） |
| 訓練データ | WIT 400M | **4.98B**（12×） |
| 訓練データ言語 | 英語のみ | **多言語** |
| 損失 | symmetric InfoNCE | symmetric InfoNCE（同じ） |
| IN-1K ゼロショット | 76.2 | **83.2** (+7.0) |

### vs BLIP-2（QFormer の祖）

| 項目 | BLIP-2 | InternVL-G |
|---|---|---|
| 視覚エンコーダ | ViT-g (EVA-g, 1B) | InternViT-6B (5.9B) |
| glue layer | QFormer (188M, BERT-base 初期化) | **QLLaMA (8B, 多言語 LLaMA-7B 初期化)** |
| query 数 | 32 | 96 |
| 生成損失 | ITC + ITM + ITG | ITC + ITM + ITG（同じ） |
| 対比能力 | ❌（QFormer は対比なし） | ✅（QLLaMA は対比可） |

### vs LLaVA-1.5

| 項目 | LLaVA-1.5-13B | InternVL-Chat-13B (QLLaMA) |
|---|---|---|
| 視覚エンコーダ | CLIP-L 336 (304M) | InternViT-6B (5.9B) |
| glue layer | MLP 2 層 | QLLaMA (8B) |
| LLM | Vicuna-13B | Vicuna-13B（同じ） |
| PT data | 558K | **1.0B** |
| SFT data | 665K | **4M** |
| MME | 1531.3 | **1586.4** (+55) |

### vs ViT-22B（Google, 2023）

| 項目 | ViT-22B | InternViT-6B |
|---|---|---|
| パラメータ | 21.7B | 5.9B (**1/3.7**) |
| 訓練データ | JFT-3B（非公開） | 4.98B Web pairs（公開） |
| IN-1K linear probe | 89.5 | 88.2 (-1.3) |
| ADE20K linear probe | 34.6 mIoU | **47.2** (**+12.6**) |
| ADE20K UperNet full | 55.3 mIoU | **58.9** (+3.6) |

**意義**: ViT-22B が「JFT-3B + 21.7B param」で達成した知覚性能を、**InternViT-6B が公開データ + 1/3.7 のパラメータで** 達成 or 上回る。CV foundation model の **「データ規模 vs パラメータ規模」のトレードオフ** の好例。

---

## モデルカタログ（公開リソース）

公開済み（HuggingFace `OpenGVLab/` 配下）：

| モデル | 用途 | パラメータ |
|---|---|---|
| InternViT-6B-224px | 画像エンコーダのみ（線形プローブ・知覚タスク用） | 5.9B |
| QLLaMA-7B（内部公開） | 言語ミドルウェアのみ | 8B |
| InternVL-14B-224px | InternViT + QLLaMA（対比・キャプショニング） | 14B |
| InternVL-Chat-Vicuna-7B/13B | フル対話モデル | 13B / 19B |
| InternVL-Chat-V1-1 / V1-2 / V1-5 | 後続シリーズの起点 | 増加中 |

**ライセンス**: MIT（コード）+ 各 LLM のライセンス（Vicuna は Apache, LLaMA は Meta ライセンス）

---

## 系譜上の位置

```
                     ┌─ [[entities/clip]] (2021)
                     │   ↓ Web 規模対比学習スケール
                     │
[Contrastive 系]    ─┼─ EVA-CLIP / OpenCLIP (2022-23)
                     │   ↓ さらにスケール
                     │
                     ├─ [[entities/internvl]] (2023 12月、本ページ) ───┐
                     │   ├─ 6B 視覚 + 8B LM middleware                  │
                     │   └─ 対比 + 生成 + 対話の統合                    │
                     ↓                                                  ↓
[VLLM 系]           ─┼─ Flamingo (2022) → BLIP-2 (2023) → LLaVA (2023) → InternVL-Chat (2023)
                     │                                                  │
                     │                                                  ├─ InternVL 1.5 (2024)
                     │                                                  ├─ InternVL 2.0 (2024)
                     │                                                  ├─ InternVL 2.5 (2024)
                     │                                                  └─ InternVL 3 (2025)
                     │
[Modern Continuation]
                     ├─ [[entities/siglip]] / SigLIP 2 (2023-25): sigmoid loss + 全部入り
                     ├─ [[entities/perception-encoder]] (2025): alignment tuning + 3 バリアント
                     └─ [[entities/dinov3]] (2025): 純粋 SSL の対抗軸
```

**「InternVL は『6B 視覚 + 8B 言語の整列』というアーキテクチャ路線を初めて本格化した。後の Perception Encoder, SigLIP 2, Qwen-VL シリーズなどの「視覚エンコーダのスケールアップ + LLM 整列」という流れに直接影響を与えた」**

---

## 後続バージョン（参考、非ingested）

| バージョン | 発表 | 主な変更 |
|---|---|---|
| **InternVL 1.0**（本論文） | 2023-12 | InternViT-6B + QLLaMA + Vicuna |
| **InternVL 1.2** | 2024-02 | InternViT-6B-448px-V1.2（45 層、fixed 448）+ MLP + Nous-Hermes-2-Yi-34B、**QLLaMA 廃止** → MLP に簡素化。40B 計 |
| **InternVL 1.5** | 2024-04 | InternViT-V1.5（dynamic 448）+ MLP + InternLM2-20B-Chat。**18 ベンチ中 8 SoTA、GPT-4V 並み**。詳細: [[entities/internvl-1-5]] / [[sources/internvl-1-5]] |
| InternVL 2.0 | 2024-07 | プログレッシブ整列 + 1B〜76B レンジ（論文なし、Mini-InternVL と同時期） |
| **Mini-InternVL** | 2024-10 | **InternViT-6B を 300M に蒸留 + 軽量 LLM**（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）。**5% パラメータで InternVL2-76B の 90% 性能**、統一ドメイン適応フレームワーク（自律走行/医療/リモセン）。詳細: [[entities/mini-internvl]] / [[sources/mini-internvl]] |
| **InternVL 2.5** | 2024-12 | **MMMU 70.1% で 70% を超えた初のオープン MLLM**。1B/2B/4B/8B/26B/38B/78B の 7 サイズ。InternViT V2.5（6B + 300M）+ InternLM 2.5 / Qwen 2.5。Progressive Scaling Strategy 公式化、Test-Time Scaling（CoT + Majority Voting）。詳細: [[entities/internvl-2-5]] / [[sources/internvl-2-5]] |
| **InternVL 3** | 2025-04 | **Native Multimodal Pre-Training**（テキスト + マルチモーダル共同事前学習）+ V2PE + MPO + VisualPRM。MMMU **72.2% で再び SOTA 更新**。Qwen2.5 base 起点（Chat 版ではない）+ InternLM3。Qwen2.5-Chat より純粋言語が強い。1B/2B/8B/9B/14B/38B/78B の 7 サイズ。詳細: [[entities/internvl-3]] / [[sources/internvl-3]] |
| **InternVL 3.5** | 2025-08 | **Cascade RL（MPO + GSPO の 2 段階強化学習）+ MoE スケーリング（20B-A4B / 30B-A3B / 241B-A28B）+ ViR（視覚トークン 50% 削減）+ DvD（4.05× 推論加速）**。9 サイズ × dense + MoE × Flash。**MMMU 77.7、MathVista で GPT-5 を +0.8、VSI-Bench で GPT-5 を +32 圧倒**、GPT-5 との差 3.9%。Qwen3 base + GPT-OSS。詳細: [[entities/internvl-3-5]] / [[sources/internvl-3-5]] |

これらは [[sources/internvl]] の発展形。**[[sources/internvl-1-5|InternVL 1.5]], [[sources/mini-internvl|Mini-InternVL]], [[sources/internvl-2-5|InternVL 2.5]], [[sources/internvl-3|InternVL 3]], [[sources/internvl-3-5|InternVL 3.5]] は ingest 済み**。それ以外（1.2, 2.0）は未 ingest。

> **「InternVL の二分岐 → 再統合 → パラダイム転換」**: シリーズは [[entities/internvl-1-5|InternVL 1.5]]（26B、商用追従）と [[entities/mini-internvl|Mini-InternVL]]（1B-4B、軽量+ドメイン特化）に枝分かれし、両者は **[[entities/internvl-2-5|InternVL 2.5]]**（1B-78B、7 サイズスイート）で **「軽量も大規模も全部入り」** として再統合された。その後 **[[entities/internvl-3|InternVL 3]]**（2025-04）で **「事後的 MLLM 適応 → Native Multimodal Pre-Training」** という哲学転換、Qwen2.5 base 起点でテキストと視覚を最初から共同事前学習。InternViT-6B / 300M が **教師・蒸留先として両系統の核** に位置する。

---

## 関連ページ

- 詳細解説: [[sources/internvl]]
- 翻訳: [[translations/internvl]]
- 関連エンティティ: [[entities/clip]]（直系の祖）、[[entities/perception-encoder]]（同時期の対比スケール路線）、[[entities/siglip]]（sigmoid loss 改良路線）、[[entities/dinov2]] / [[entities/dinov3]]（純粋 SSL 対抗路線）
- 関連概念: [[concepts/vision-transformer]]、[[concepts/foundation-model]]、[[concepts/contrastive-learning]]、[[concepts/weakly-supervised-pretraining]]、[[concepts/zero-shot-transfer]]
