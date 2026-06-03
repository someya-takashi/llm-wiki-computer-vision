---
type: source
source_path: raw/papers/Data Filtering Networks.md
source_kind: paper
title: "Data Filtering Networks"
authors: [Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, Vaishaal Shankar]
year: 2023
venue: arXiv 2309.17425 / ICLR 2024
ingested: 2026-06-03
tags: [dfn, clip, data-curation, vision-encoder, dataset-construction, apple]
translation: "[[translations/dfn]]"
---

# Data Filtering Networks（データ・フィルタリング・ネットワーク）

> 原典: [[translations/dfn]] ・ `raw/papers/Data Filtering Networks.md`
> 著者: Alex Fang*¹², Albin Madappally Jose¹, Amit Jain¹, Ludwig Schmidt², Alexander Toshev¹, Vaishaal Shankar¹
> 所属: ¹ Apple, ² University of Washington（* Work done while at Apple）
> 出典: arXiv:2309.17425（2023 年 9 月）、ICLR 2024
> 公開モデル: https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378（および DFN5B-CLIP-ViT-H-14 / DFN2B-CLIP-ViT-L-14 / DFN2B-CLIP-ViT-B-16）

## 一言まとめ

**Apple が提唱した「データセットを直接訓練するのではなく、データを選別する小さな専用ネットワーク（DFN）を訓練し、その DFN で大規模未フィルタプールから高品質データセットを誘導する」というデータ・キュレーション・パラダイム**。本論文の DFN は **HQITP-350M（357M 人間検証済みキャプション）で小さな ViT-B/32 CLIP モデルを訓練 + 重みアンサンブル付きファインチューン**したもので、これを使って **DataComp の 12.8B プールから上位 15% を選別して DFN-2B を生成**。誘導された **DFN-5B（42B プールから生成）で訓練した ViT-H/14 が ImageNet ゼロショット 84.4%** で当時の SOTA。**鍵となる発見は「フィルタとしての性能は ImageNet 性能と相関しない」**で、小さな高品質モデルが大きな ImageNet 強モデルより良いフィルタになり得ることを示した。本 wiki にとっての重要性は、**[[entities/qwen2-vl|Qwen2-VL]] / [[entities/qwen2-5-vl|Qwen2.5-VL]] の ViT 初期化** に DFN の CLIP モデルが使われている点で、現代の主要 MLLM の視覚エンコーダの出自原典として最重要。

## 背景と問題意識

### CV データセット研究の長年の不透明さ

論文の §1 が明確に指摘する問題：

- **モデル研究**: 幅・深さ・正規化・ハイパラの順列を厳密に評価でき、年単位の改善が積み上がる
- **データセット研究**: ほとんどの大規模訓練セットは非公開、コミュニティのオープン再現はワンオフ努力（LAION, COYO）
- **結果**: 「データセットがどう作られたか」が不透明で、研究として再現性が低い

CLIP（OpenAI 2021）の WIT-400M、Google の WebLI、JFT-3B はすべて **非公開**。LAION-2B はオープンだが OpenAI WIT との性能差を埋めるのに **大規模計算が必要**だった。

### CLIP フィルタリングの限界

LAION の標準的構築方法は **CLIP フィルタリング**：

```python
def clip_filter(image, text, threshold=0.3):
    similarity = cosine_sim(clip.encode_image(image), clip.encode_text(text))
    return similarity > threshold
```

しかしこのアプローチは **既存の訓練済み CLIP（LAION の場合 OpenAI CLIP ViT-B/32）に依存**し、**そのフィルタが達成できる top-line 性能で制限される**。LAION-2B が OpenAI WIT より 5 倍大きいのに ImageNet ゼロショット性能で同等となるのに大計算を要したのはこのため。

### DataComp ベンチマーク

本論文と並行発表の **DataComp（Gadre et al. 2023）** は 12.8B 画像-テキスト対と固定計算予算による標準的データセット評価フレームワークを提供。本論文はこの DataComp ベンチマークでフィルタリング手法を競う。**DataComp-1B（DC-1B）** が以前の SOTA（CLIP フィルタリング + ImageNet クラスタリング）。

## 提案手法 / 主張：DFN パラダイム

### 中心的アイデア

```
従来：
  Web → CLIP フィルタ（既存モデル）→ 訓練データ → CLIP モデル訓練

DFN パラダイム：
  Web → 候補プール（DataComp 12.8B）
              ↓
       DFN（高品質データで訓練した小さな CLIP）でフィルタリング
              ↓
       誘導データセット（DFN-2B = 上位 15%）
              ↓
       CLIP モデル訓練（ViT-B/L/H）
```

**用語定義**（§3.1）:
- **filter dataset**: DFN を訓練するデータプール（本論文では HQITP-350M）
- **induced dataset**: DFN がフィルタリングして得たデータセット（DFN-2B、DFN-5B）
- **induced model**: 誘導データセットで訓練したモデル

### 鍵となる発見 1: フィルタ性能 ≠ ImageNet 性能（§3.3、図 3）

**従来の直感**: より良い CLIP モデル（ImageNet で強い）を CLIP フィルタリングに使えば、再帰的により良いデータセットとモデルが得られるはず。

**実証された反例**: **OpenAI CLIP より 30% 低い ImageNet 性能のモデルが、フィルタリング・モデルとして同じくらい良くなり得る**。**ImageNet 性能とフィルタリング性能は相関しない**。

これは「**フィルタリングは下流タスクと異なる能力**」という重要な洞察で、本論文の理論的貢献。

### 鍵となる発見 2: データ品質が最重要（§3.3、図 4）

CC12M（10M 高品質）を Common Crawl で段階的に汚染する実験：
- **DFN の ImageNet 性能**: 緩やかに低下
- **DFN のフィルタリング性能**: **少量の汚染で即座に崩壊**

つまり **「フィルタリング・モデルには高品質訓練データが決定的」**。Common Crawl で汚染された DFN は未フィルタ・データよりわずかに良いだけ。

### 鍵となる発見 3: バイナリ分類器より CLIP フィルタ（§3.3、表 1）

代替手法との比較：
- **ResNet-34 / OpenAI ViT-B/32 バイナリ・フィルタ**（ImageNet/CC12M 陽性 vs Common Crawl 陰性）
- **M3AE**（マルチモーダル MAE、再構成損失をフィルタ基準）
- **CLIP ViT-B/32（CC12M で訓練）**

結果：**CLIP フィルタが全代替を上回る**（IN 0.289 vs 次点 M3AE 0.237）。理由：**バイナリ分類器は「良い分布」の明示的仮定をするが CLIP はより柔軟、M3AE はテキスト再構成が CC12M の少データで困難**。

### 最高性能 DFN の作成（§4）

3 段階レシピ：

1. **HQITP-350M で ViT-B/32 を訓練** — 357M 画像-テキスト対、**人間検証済みキャプション**、Apple 社内データセット
2. **OpenAI ViT-B/32 で重み初期化** — OAI-Init
3. **MS COCO + Flickr30k + ImageNet（OpenAI テンプレート）でファインチューン + 重みアンサンブル**

この DFN を DataComp 12.8B に適用、**上位 15% で DFN-2B を抽出**。さらに **30B 非 DataComp 画像を加えた 42B プールから DFN-5B を抽出**。

### 標準的 ML 技法の効果（表 2）

DFN は通常の ML モデルと同じ技法で改善できる：
- **拡張（Augmentation）**: +0.006 average
- **サンプル/バッチサイズ増加**（2.56B → 5.12B、4096 → 8192）: +0.005
- **ファインチューン**（COCO/Flickr/IN）: +0.010（最も効く）
- **OAI-Init**（OpenAI 重みで初期化）: +0.012

組み合わせると DFN ベースライン 0.536 → 0.560（+0.024）。

## 実験結果と知見

### 表 3: DFN-5B 84.4% IN ゼロショット SOTA

xlarge スケール（ViT-L/14、12.8B サンプル）での各データセット比較：

| データセット | IN | IN Shifts | VTAB | Retrieval | Average |
|---|---|---|---|---|---|
| LAION-2B | 0.731 | 0.603 | 0.586 | 0.589 | 0.601 |
| OpenAI WIT-400M | 0.755 | 0.649 | 0.586 | 0.543 | 0.617 |
| DC-1B | 0.792 | 0.679 | 0.652 | 0.608 | 0.663 |
| **DFN-2B** | **0.814** | **0.688** | **0.656** | **0.649** | **0.669** |

**DFN-2B → ViT-L/14 が DC-1B より +2.2 IN、WIT-400M より +5.9 IN**。

最大規模では：
- **DFN-5B → ViT-H/14（224²）: IN 0.834, Average 0.698**
- **DFN-5B → ViT-H/14（378²）: IN 0.844, Average 0.710** ← **当時の SOTA**

WebLI (Google SigLIP の SO-400M) の 0.831 IN / 0.692 Avg を上回る。

### 計算効率の改善

- **DFN-2B ViT-L/14 が LAION-2B ViT-G/14（34B サンプル）を IN +1.5% で上回り、16 倍少ない計算**
- **DFN-2B ViT-B/16 が OpenAI ViT-L/14 と競争的、4 倍少ない計算**

つまり **「より良いデータセットでモデルサイズを小さくできる」**。

### 表 4: 高品質データは DFN 訓練に使うべき、終点モデル訓練に使うべきでない

| 戦略 | IN |
|---|---|
| OAI ViT-B/32 誘導データ + HQITP-350M で ViT-L 訓練 | 0.774 |
| **DFN-2B で ViT-L 訓練** | **0.814** |
| DFN-2B + HQITP-350M で ViT-L 訓練 | 0.813（変化小） |

**「高品質データを終点モデル訓練に直接使うより、フィルタリング・モデル訓練に使う方が遥かに効く」**。

### 表 5: VQA でも DFN-2B が優位

BLIP-2 の視覚エンコーダ比較：

| 訓練データ | アーキテクチャ | VQAv2 | GQA | OKVQA |
|---|---|---|---|---|
| OAI-WIT-400M | ViT-L | 45.5 | 30.0 | 19.1 |
| **DFN-2B** | ViT-L | **48.3** | **31.3** | 21.9 |
| LAION-2B | ViT-g | 48.7 | 31.1 | **24.5** |

**DFN-2B ViT-L が OAI WIT ViT-L を全 VQA タスクで上回り、より大きな LAION-2B ViT-g と競争的**。これが **DFN が MLLM の視覚エンコーダとして有効な根拠**。

### 表 6: 公開データだけで競争的 DFN を構築可能（§4.2）

公開データセット **CC12M + CC3M + Shutterstock 15M** で DFN を訓練：

| DFN 訓練データ | Scale | IN | Average |
|---|---|---|---|
| **CC12M + CC3M + SS15M** | medium | **0.307** | **0.349** |
| OpenAI WIT-400M | medium | 0.285 | 0.338 |
| **CC12M + CC3M + SS15M** | large | **0.591** | **0.532** |
| OpenAI WIT-400M | large | 0.578 | 0.527 |

**完全公開データで OpenAI WIT-400M を超える DFN を構築可能**。これは「大規模高品質データセット作成の民主化」への一歩。

### 図 5 + 表 8（Appendix C）: DFN ファインチューンは頑健性を保つ

ImageNet を「直接訓練データに加える」vs「DFN をファインチューンするのに使う」の比較：

| 戦略 | IN | IN-V2 | ObjectNet | IN-Sketch | IN-R | IN-A |
|---|---|---|---|---|---|---|
| Baseline DFN | 0.624 | 0.547 | 0.511 | 0.510 | 0.724 | 0.257 |
| **Baseline DFN FT on ImageNet** | **0.678** | **0.594** | **0.536** | **0.536** | **0.743** | **0.284** |
| Baseline DFN + IN（IN を訓練に直接追加） | 0.757 | 0.652 | 0.509 | 0.512 | 0.703 | 0.272 |

**直接 IN 訓練は IN/IN-V2 で高性能だが ObjectNet/Sketch/R で改善しない**。**DFN を IN でファインチューンすると IN とすべての分布シフトで改善**。**DFN は蒸留ではなく頑健性正則化として機能**（誘導モデルの方が元の DFN より高性能）。

## 限界・批判的視点

### 著者が認める限界（§5）

1. **データセット品質を直接最適化する方法が不明**: 「整列」を弱い代理として使うが、品質の正式定義はなし
2. **他ドメインへの拡張が不明**: 音声・テキスト・動画データで何が「フィルタリング・モデル」になるか
3. **DFN は alignment proxy にすぎない**: 真のデータ品質の代理として完璧ではない

### 本 wiki の視点からの追加批判

4. **HQITP-350M が非公開**: 「**人間検証済みキャプション 357M**」という最強の DFN 訓練データは Apple 内部資産。完全な再現は不可能（§4.2 で公開データ DFN を提示するが性能ギャップあり）
5. **DataComp 評価への偏重**: 評価が DataComp の 38 タスク中心。**動画、3D、専門ドメイン（医療、衛星）など DataComp 外の評価が薄い**
6. **MLLM での実証が BLIP-2 のみ（§4.1）**: VQA で DFN-2B が有効と示すが、現代的な LLaVA / Qwen-VL / InternVL での評価は本論文時点では未完。**Qwen2-VL/2.5-VL の ViT 初期化として DFN モデルが採用されたのは本論文後の発展**
7. **「フィルタ性能 ≠ IN 性能」の説明が浅い**: 図 3 で実証はするが「**なぜ ImageNet 性能が高くてもフィルタリングが下手になるか**」の理論的説明は不足。**fine-grained image-text alignment と coarse-grained 分類の能力が分離する**理由が深掘りされていない
8. **DFN-5B の構成不透明**: 30B 非 DataComp 画像の出自・キュレーションが不明。**完全再現が困難**
9. **DFN の再帰的改善は試されていない**: DFN-5B で訓練した ViT-H/14 を新しい DFN として使う実験はない。**理論上は再帰的改善が可能なはずだが、論文では検証されない**
10. **2024-2025 年の SigLIP 2 / PE との比較なし**: 本論文時点で SOTA だったが、**SigLIP 2 (85.0% IN) や Perception Encoder が後に追い抜いている**

### 本論文の DFN モデルの現在地

- **Qwen2-VL の ViT 初期化として採用**（[[entities/qwen2-vl]] technical report で明記、DFN-CLIP の重みを継続学習）
- **Qwen2.5-VL もこの路線を踏襲**（[[entities/qwen2-5-vl]]）
- ただし **Qwen3-VL は SigLIP-2 から継続学習に切り替え**（[[entities/qwen3-vl]]）— DFN は Qwen2-VL/2.5-VL 世代に固有

## 用語と略称

- **DFN**: **D**ata **F**iltering **N**etwork（本論文の中心概念）
- **DFN-2B**: DFN で DataComp 12.8B から上位 15% を選別した 2B サンプル・データセット
- **DFN-5B**: DataComp 12.8B + 30B 非 DataComp 画像（計 42B）から DFN で選別した 5B サンプル
- **HQITP-350M**: **H**igh-**Q**uality **I**mage-**T**ext **P**airs 350M、Apple 内部の人間検証済みキャプション 357M
- **HQITP-135M**: HQITP-350M の部分集合（[^38] で使用）
- **filter dataset**: DFN 訓練に使うデータプール
- **induced dataset**: DFN がフィルタリングして生成したデータセット
- **induced model**: 誘導データセットで訓練したモデル
- **DataComp**: 標準データセット評価ベンチマーク（Gadre et al. 2023, [^12]）
- **DC-1B / DataComp-1B**: DataComp が提供するベースライン・データセット（CLIP フィルタ + ImageNet クラスタリング）
- **CommonPool**: DataComp の未フィルタ 12.8B プール
- **WIT-400M**: OpenAI CLIP の訓練データ（非公開）
- **LAION-2B / LAION-5B**: LAION のオープン CLIP 訓練データ
- **WebLI**: Google SigLIP の訓練データ（非公開）
- **OAI-Init**: OpenAI CLIP 重みでの初期化
- **OpenClip**: LAION ベースのオープン CLIP 実装
- **AXlearn**: Apple の TPU 訓練フレームワーク
- **M3AE**: Multimodal Masked Autoencoder（[^14]、本論文の代替フィルタ候補）
- **BLIP-2**: VQA 評価で使った MLLM 基盤（[^28]）
- **CC12M / CC3M / SS15M**: Conceptual Captions 12M/3M、Shutterstock 15M（公開 DFN 訓練データ）

## CV / MLLM における意義（wiki 既存ページとの接続）

### Qwen-VL シリーズの ViT 初期化として採用

- **[[entities/qwen2-vl|Qwen2-VL]]**（Alibaba, 2024、[[sources/qwen2-vl]]）: **675M ViT を DFN で初期化** + 絶対位置埋め込みを 2D-RoPE に置換。**現在の Qwen 系 MLLM の視覚基盤の原典が DFN**
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（Alibaba, 2025、[[sources/qwen2-5-vl]]）: **Window Attention ViT をゼロから訓練するが DFN + 社内データで初期化**
- **[[entities/qwen3-vl|Qwen3-VL]]**（Alibaba, 2025、[[sources/qwen3-vl]]）: **SigLIP-2 に切り替え**（[[entities/siglip|SigLIP]] の系統）。DFN は Qwen2-VL/2.5-VL 世代の遺産に

### CLIP 系列での位置付け

[[concepts/weakly-supervised-pretraining]] / [[concepts/foundation-model]] の「**A. 弱教師あり系**」セクションで CLIP / SigLIP / OpenCLIP / EVA-CLIP / MetaCLIP の系譜の重要分枝。

- **CLIP 系の主要分枝**:
  - OpenAI CLIP（WIT-400M、非公開データ）
  - OpenCLIP（LAION、公開）
  - **DFN（Apple、データ・キュレーション革新）**
  - SigLIP / SigLIP 2（Google、損失関数革新）
  - MetaCLIP（Meta、データ公開）
  - Perception Encoder（Meta、中間層活用）

### [[questions/large-scale-pretraining-series]] における位置付け

本論文は **「系列 ①: WSL 対比型 CLIP 系列」** の重要分枝。「**データ品質が訓練の鍵**」という本論文の主張は、SigLIP 2 の SSL 借用や PE の慎重なキュレーションと併せて、**2024-2025 年の「データセット中心主義」の流れ**を象徴。

### データセット研究のパラダイム転換

本論文の最大の貢献は **「データセット設計はモデル設計と同じツールを使える」** という主張。これは：
- DataComp（同年）と並ぶ「データ中心 AI」ムーブメントの代表
- MetaCLIP（2024）の「データ公開」と相補的（DFN は手法公開、MetaCLIP はデータ公開）
- 後の SigLIP 2 / PE のキュレーション哲学に影響

## 関連ページ

### 直接の前提

- [[sources/clip]] / [[entities/clip]] — DFN は CLIP モデルを DFN として使う
- [[concepts/weakly-supervised-pretraining]] — WSL の中の CLIP 系列に DFN が属する
- [[translations/dfn]] — 原典の本文 + Appendix 和訳

### DFN を採用したモデル（最重要）

- [[entities/qwen2-vl]] / [[sources/qwen2-vl]] — **675M ViT を DFN で初期化**、現在の Qwen 系 MLLM の視覚基盤
- [[entities/qwen2-5-vl]] / [[sources/qwen2-5-vl]] — Window Attention ViT、DFN + 社内データで初期化
- [[entities/qwen3-vl]] / [[sources/qwen3-vl]] — SigLIP-2 に切り替えたため DFN は使わないが系統の流れを把握する上で重要

### 関連の CLIP 系列モデル

- [[entities/siglip]] / [[sources/siglip]] — Google の CLIP 改良（sigmoid 損失）
- [[sources/siglip-2]] — SigLIP 2、5 系統融合（DFN 後継として現代的）
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — Meta、中間層活用

### 関連 question

- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列、DFN の位置付け
- [[questions/vit-dynamic-resolution-evolution]] — Qwen-VL 系の解像度進化、DFN ViT 初期化との関係

### 上位概念

- [[concepts/foundation-model]] — 基盤モデル全体
- [[concepts/contrastive-learning]] — CLIP の対比学習
- [[concepts/vision-transformer]] — DFN モデルの視覚バックボーン
