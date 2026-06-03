---
type: entity
entity_kind: model
aliases: [DFN, Data Filtering Networks, DFN-CLIP, DFN-2B, DFN-5B]
related: ["[[entities/clip]]", "[[entities/siglip]]", "[[entities/qwen2-vl]]", "[[entities/qwen2-5-vl]]", "[[entities/qwen3-vl]]", "[[entities/perception-encoder]]"]
sources: ["[[sources/dfn]]"]
updated: 2026-06-03
---

# DFN（Data Filtering Networks）

## 概要

**Apple が 2023 年に提唱した「データ・フィルタリング・ネットワーク」パラダイムと、それで構築された CLIP モデル・データセット系列**。論文では「DFN」が手法名（小さな CLIP モデルでデータをフィルタリングするネットワーク）とそれが誘導するデータセット名（DFN-2B、DFN-5B）の両方を指す。本 wiki にとって最重要なのは、**Qwen2-VL / Qwen2.5-VL の ViT 初期化として DFN-CLIP（特に ViT-L/14、ViT-H/14）が採用されている点**。

## 命名規約

| 名前 | 意味 |
|---|---|
| **DFN（手法）** | データ・フィルタリング・ネットワーク = 小さな CLIP モデルでデータをフィルタリングするパラダイム |
| **DFN モデル** | フィルタリングに使う ViT-B/32 ベース・モデル（HQITP-350M で訓練 + ファインチューン） |
| **DFN-2B（データセット）** | DFN が DataComp 12.8B から上位 15% を選別した **2B サンプル**画像-テキスト・データセット |
| **DFN-5B（データセット）** | DataComp 12.8B + 30B 非 DataComp 画像（計 42B）から DFN で選別した **5B サンプル** |
| **DFN-CLIP（モデル）** | DFN-2B / DFN-5B で訓練された誘導 CLIP モデル（公開チェックポイント） |

## DFN モデル（フィルタとしての CLIP）

データのフィルタリングに使う「DFN」自体は **小さな CLIP モデル**：

- **アーキテクチャ**: ViT-B/32 ベースの CLIP
- **訓練データ**: **HQITP-350M**（**H**igh-**Q**uality **I**mage-**T**ext **P**airs 350M、Apple 社内、**357M、人間検証済みキャプション**）
- **初期化**: OpenAI CLIP ViT-B/32 の重みで初期化（**OAI-Init**）
- **ファインチューン**: MS COCO + Flickr30k + ImageNet（OpenAI テンプレートをキャプションとして）
- **重みアンサンブル**: 過学習防止
- **訓練詳細**（最終 DFN）: 5.12B サンプル、16,384 バッチサイズ、2,000 ステップ・ウォームアップ

### 公開可能版 DFN（§4.2）

**HQITP-350M は非公開**だが、**完全公開データだけ**で競争的 DFN を構築可能：
- 訓練データ: **CC12M + CC3M + Shutterstock 15M**（合計 30M）
- DataComp medium/large で OpenAI WIT-400M を上回るフィルタ性能

## DFN-CLIP モデル（公開チェックポイント）

DFN-2B / DFN-5B で訓練された CLIP モデル群（[Apple Hugging Face](https://huggingface.co/apple)）：

| モデル | データセット | 解像度 | ImageNet ZS | 38-task Average | URL |
|---|---|---|---|---|---|
| **DFN5B-CLIP-ViT-H-14-378** | DFN-5B | 378² | **0.844** | **0.710** | [HF](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) |
| DFN5B-CLIP-ViT-H-14 | DFN-5B | 224² | 0.834 | 0.698 | [HF](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14) |
| DFN2B-CLIP-ViT-L-14 | DFN-2B | 224² | 0.814 | 0.669 | [HF](https://huggingface.co/apple/DFN2B-CLIP-ViT-L-14) |
| DFN2B-CLIP-ViT-B-16 | DFN-2B | 224² | 0.762 | 0.609 | [HF](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16) |

**特筆すべきは**:
- **DFN5B-ViT-H-14-378 が 84.4% IN ZS** — 2023 年当時の SOTA、当時の LAION-2B / DataComp-1B / OpenAI WIT を上回る
- **DFN2B-ViT-L-14 が 81.4% IN ZS** — DataComp-1B の以前ベストを **+2.2 ポイント** 上回る、LAION-2B ViT-G/14 (34B samples) を **16× 少ない計算**で上回る
- **DFN2B-ViT-B-16** が **OpenAI ViT-L/14 と競争的、4× 少ない計算**

## DFN-2B / DFN-5B データセット（誘導データセット）

### DFN-2B

- **出自**: DataComp の 12.8B サンプル CommonPool
- **DFN フィルタリング**: 上位 15% を選別
- **サンプル数**: 2B
- **公開**: あり（[DataComp 経由](https://www.datacomp.ai/dcclip/index.html#dfn)）
- **DFN-2B から訓練できる SOTA モデル**: ViT-L/14 で IN 81.4%、ViT-B/16 で IN 76.2%

### DFN-5B

- **出自**: DataComp 12.8B + 30B 非 DataComp Web 画像 = **42B プール**
- **DFN フィルタリング**: 上位画像-テキスト対を選別
- **サンプル数**: 5B
- **公開**: なし（30B 非 DataComp 画像が Apple 内部）
- **DFN-5B から訓練できる SOTA モデル**: ViT-H/14 で IN 84.4%（378² 解像度）

## DFN の鍵となる発見

論文の §3.3 で実証された 3 つの非自明な発見：

### 発見 1: フィルタ性能 ≠ ImageNet 性能（図 3）

**OpenAI CLIP より 30% 低い ImageNet 性能のモデルが、フィルタとしては同じくらい良くなり得る**。CLIP モデルの ImageNet 性能とフィルタリング性能は相関しない。

### 発見 2: フィルタ訓練データの品質が決定的（図 4）

CC12M を Common Crawl で段階的に汚染すると：
- DFN の ImageNet 性能: 緩やかに低下
- **DFN のフィルタ性能: 少量の汚染で即座に崩壊**

つまり「**フィルタリング・モデルには高品質訓練データが特に重要**」。

### 発見 3: バイナリ分類器より CLIP フィルタ（表 1）

ResNet-34 バイナリ分類器、OpenAI CLIP バイナリ・フィルタ、M3AE 再構成損失と比較し、**CLIP フィルタが全代替を上回る**（IN top-1 0.289 vs 次点 M3AE 0.237）。理由：「**CLIP は柔軟な整列指標を提供、バイナリ分類器は明示的な分布仮定をする**」。

## ベンチマーク結果

### DataComp ベンチマーク（38 タスク）

DFN-2B / DFN-5B は SOTA：

| データセット | IN | IN Shifts | VTAB | Retrieval | Average |
|---|---|---|---|---|---|
| LAION-2B (xlarge) | 0.731 | 0.603 | 0.586 | 0.589 | 0.601 |
| OpenAI WIT-400M | 0.755 | 0.649 | 0.586 | 0.543 | 0.617 |
| DataComp-1B | 0.792 | 0.679 | 0.652 | 0.608 | 0.663 |
| **DFN-2B** | **0.814** | **0.688** | **0.656** | **0.649** | **0.669** |
| **DFN-5B (224²)** | **0.834** | 0.713 | **0.675** | 0.684 | **0.698** |
| **DFN-5B (378²)** | **0.844** | 0.738 | **0.685** | 0.695 | **0.710** |

### BLIP-2 VQA 性能（DFN モデルを視覚エンコーダに）

| 視覚エンコーダ | アーキテクチャ | VQAv2 | GQA | OKVQA |
|---|---|---|---|---|
| OAI-WIT-400M | ViT-L | 45.5 | 30.0 | 19.1 |
| **DFN-2B** | ViT-L | **48.3** | **31.3** | 21.9 |
| LAION-2B | ViT-g | 48.7 | 31.1 | **24.5** |

**DFN-2B ViT-L が OpenAI WIT-400M ViT-L を全 VQA タスクで上回り、LAION-2B ViT-g（より大きい）と競争的**。

## MLLM での採用

| MLLM | DFN の使い方 | 出典 |
|---|---|---|
| **[[entities/qwen2-vl|Qwen2-VL]]** (2024.09) | **675M ViT を DFN-CLIP の重みで初期化**、絶対位置埋め込みを 2D-RoPE に置換 | [[sources/qwen2-vl]] |
| **[[entities/qwen2-5-vl|Qwen2.5-VL]]** (2025.02) | **Window Attention ViT をゼロから訓練するが DataComp + 社内データで初期化**（DFN 系の流れを踏襲） | [[sources/qwen2-5-vl]] |
| **[[entities/qwen3-vl|Qwen3-VL]]** (2025.11) | **SigLIP-2 から継続学習に切り替え**（DFN を捨てる） | [[sources/qwen3-vl]] |

**Qwen2-VL / Qwen2.5-VL 世代の MLLM 視覚エンコーダは DFN の遺産**。Qwen3-VL で SigLIP-2 に切り替えた理由は、SigLIP 2 が **CLIP 5 系統融合（対比 + 自己蒸留 + MIM + decoder + 蒸留）** で更に高性能化したため。

## DFN と他の CLIP 系列の比較

| モデル | データ規模 | 公開度 | 革新 |
|---|---|---|---|
| OpenAI CLIP | 400M（非公開 WIT） | コード+重み公開、データ非公開 | ゼロショット転移の起点 |
| OpenCLIP | LAION 5B | 完全公開 | オープン再現 |
| **DFN-5B** | **42B プール → 5B 選別** | **モデル公開、HQITP-350M 非公開、DFN-2B 公開** | **データ・キュレーション革新** |
| SigLIP | WebLI（非公開） | コード+重み公開、データ非公開 | sigmoid 損失 |
| **SigLIP 2** | WebLI 10B（非公開） | コード+重み公開 | **5 系統融合（DFN 後継として現代的）** |
| MetaCLIP | CommonCrawl 400M/2.5B（公開） | **データ完全公開** | キュレーション基準公開 |
| Perception Encoder | 5.4B unique（非公開） | コード+重み公開 | 中間層活用 + alignment tuning |

**DFN の差別化**: 「**データ・キュレーション手法の革新**」に焦点。SigLIP は損失関数、PE は中間層活用、MetaCLIP はデータ公開、それぞれ別軸で CLIP を改善。

## 限界

1. **HQITP-350M が非公開** — Apple 内部の人間検証済みキャプション 357M。完全再現不可
2. **DFN-5B の 30B 非 DataComp 画像が非公開** — 出自・キュレーションが不透明
3. **DataComp 評価への偏重** — 38 タスクは多様だが、動画・3D・専門ドメインが薄い
4. **MLLM 評価が BLIP-2 のみ** — 現代的な LLaVA / Qwen-VL / InternVL での実証は本論文後の Qwen-VL 系が間接的に裏付けるのみ
5. **SigLIP 2 (2025) に追い抜かれている** — IN ZS 85.0%、5 系統融合で DFN-5B (84.4%) を超える
6. **「フィルタ性能 ≠ IN 性能」の説明が浅い** — 実証はあるが理論的説明欠如

## 関連ページ

### 直接の出典

- [[sources/dfn]] — DFN 論文（Fang et al., Apple, ICLR 2024）
- [[translations/dfn]] — 本論文の本文 + Appendix 和訳

### DFN を採用した MLLM

- [[entities/qwen2-vl]] — **DFN-CLIP で ViT 初期化**
- [[entities/qwen2-5-vl]] — Window Attention ViT + DFN 系の流れ
- [[entities/qwen3-vl]] — SigLIP-2 に切り替え（DFN を捨てる）

### CLIP 系列の他の主要モデル

- [[entities/clip]] — オリジナル CLIP
- [[entities/siglip]] / [[sources/siglip]] — sigmoid 損失改良
- [[sources/siglip-2]] — 5 系統融合の現代版
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — 中間層活用

### 上位概念

- [[concepts/weakly-supervised-pretraining]] — DFN は WSL CLIP 系列の重要分枝
- [[concepts/foundation-model]] — 基盤モデル全体
- [[concepts/contrastive-learning]] — CLIP 対比学習

### データセット研究の関連

- DataComp（Gadre et al. 2023）— 並行発表のベンチマーク・データセット
- MetaCLIP（Xu et al. 2024）— データ公開で相補的
- LAION-2B / LAION-5B — オープン CLIP 訓練データの基盤

### 関連 question

- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列、DFN の位置付け
