---
type: source
source_path: raw/papers/An Image is Worth 16x16 Words_ Transformers for Image Recognition at Scale.md
source_kind: paper
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: [Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby]
year: 2020
venue: arXiv 2010.11929 / ICLR 2021
ingested: 2026-06-03
tags: [vit, vision-transformer, transformer, image-classification, foundation-model]
translation: "[[translations/vision-transformer]]"
---

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（一枚の画像は 16×16 単語に値する：大規模画像認識のための Transformer）

> 原典: [[translations/vision-transformer]] ・ `raw/papers/An Image is Worth 16x16 Words_ Transformers for Image Recognition at Scale.md`
> 著者: Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*, Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*（* equal technical contribution / equal advising、Google Research, Brain Team）
> 出典: arXiv:2010.11929（2020 年 10 月）、ICLR 2021
> 公開コード: https://github.com/google-research/vision_transformer

## 一言まとめ

**「画像を 16×16 のパッチに切り分け、各パッチを単語のように扱って標準的な Transformer に流し込むだけ」という驚くほどシンプルなアイデアで、十分な事前学習データ（JFT-300M 級）があれば CNN を上回ることを示した、現代 CV のパラダイム・シフトを起こした原典論文**。本論文は **ViT（Vision Transformer）** という名称を確立し、`ViT-B/16`、`ViT-L/14`、`ViT-H/14` などの命名規約も導入。本 wiki に登録されたほぼすべての視覚モデル（[[entities/dino|DINO]] / [[entities/dinov2|DINOv2]] / [[entities/dinov3|DINOv3]] / [[entities/mae|MAE]] / [[entities/ibot|iBOT]] / [[entities/clip|CLIP]] / [[entities/siglip|SigLIP]] / [[entities/sam|SAM]] / [[entities/perception-encoder|PE]] / [[entities/qwen-vl|Qwen-VL]] 系 / [[entities/internvl|InternVL]] 系 / [[entities/gemma-3|Gemma 3]] / [[entities/sdxl|SDXL]]）が ViT を視覚バックボーンとして採用しており、**本 wiki の最も基礎的な原典の 1 つ**。

## 背景と問題意識

### 2020 年の CV の状況

論文発表時（2020 年 10 月、arXiv 初版）、コンピュータ・ビジョンは以下の状況にあった：

- **CNN（畳み込みニューラルネット）が支配的**: ResNet（[^16]）、EfficientNet（[^55]）、ResNeXt などが画像分類の SOTA
- **NLP では Transformer がデファクト・スタンダード**: BERT（2018）、GPT-3（2020）が大成功
- **CV への self-attention 採用は限定的**: 部分的な CNN 置換（[^41]、[^48]）、特徴マップへの拡張（[^4]、[^7]）など、CNN とのハイブリッドが主流
- **「画像に Transformer を直接使う」試みは小スケール**: iGPT（[^8]、画素を直接 Transformer に）や [^12]（2×2 パッチ）など、いずれも小解像度・限定的成果

### 中心的問い

著者らの問いはシンプル：「**画像固有のアーキテクチャ修正を最小限にして、NLP の標準 Transformer をそのまま画像に適用したら何が起きるか？**」

### 仮説と展望

著者らの仮説は「**スケールの恩恵を CV にもたらせるか**」。NLP では Transformer の **データ・モデル・計算のスケーリング**が成功の鍵だった（BERT → GPT-3 で 1000 億パラメータまで）。CV でも同じスケール則が働くなら、CNN の帰納バイアス（局所性・平行移動同変性）を放棄しても、データで補えるはず。

## 提案手法 / 主張：ViT アーキテクチャの核心

### 4 ステップ・パイプライン（図 1）

```
入力画像 H×W×3
  ↓ ① パッチ分割: 16×16 のグリッド（224×224 なら 14×14 = 196 パッチ）
  ↓ ② 線形射影: 各パッチを D 次元ベクトルに埋め込み（実装は stride 16 の畳み込み 1 層と等価）
  ↓ ③ [CLS] トークン前置 + 学習可能 1D 位置埋め込み加算
  ↓ ④ Transformer Encoder × L 層（pre-norm + MSA + MLP + 残差接続）
  ↓ 出力: [CLS] トークンの最終状態
  ↓ 分類ヘッド（線形 1 層）
```

### モデル変種命名規約（本論文で確立）

| モデル | 層数 | 隠れ次元 $D$ | MLP | ヘッド | パラメータ |
|---|---|---|---|---|---|
| **ViT-Base (B)** | 12 | 768 | 3072 | 12 | 86M |
| **ViT-Large (L)** | 24 | 1024 | 4096 | 16 | 307M |
| **ViT-Huge (H)** | 32 | 1280 | 5120 | 16 | 632M |

`ViT-B/16` = Base + パッチサイズ 16。後の研究で `ViT-S`（Small）/ `ViT-g`（giant）/ `ViT-G`（Giant）/ `ViT-e`（enormous）など各種サイズが追加されたが、**B/L/H の命名と該当パラメータ数は本論文がオリジナル**。

### 3 つの重要な設計判断

1. **学習可能 1D 位置埋め込み**: 著者らは 2D-aware 位置埋め込み・相対位置埋め込み・正弦波エンコーディングを Appendix D.4 で比較したが、**ImageNet-21k 事前学習時には差がほぼなし**。「**モデルが 2D 構造を位置埋め込みの類似度から自動的に学習する**」（§4.5、図 7 中央）ことが実証された
2. **[CLS] トークン**: BERT 風の読み出しトークン。これも GAP（global average pooling）と比較されたが、適切な学習率設定で同等の性能（Appendix D.3）
3. **Pre-norm**: LayerNorm を各ブロックの**前**に配置（オリジナル Transformer の post-norm から変更）。深い Transformer の訓練安定性に重要

### Hybrid モデルとの比較

論文では**Hybrid モデル**（CNN 特徴マップを ViT に入力）も評価しているが、結論として：
- 小さなモデルではハイブリッドが純粋 ViT をわずかに上回る
- **大きなモデルでは差が消える**（§4.4、図 5）
- ViT は試した範囲内で**飽和しない**

## 実験結果と知見

### Table 2: ImageNet 系での SOTA 比較

| | ViT-H/14 (JFT) | ViT-L/16 (JFT) | ViT-L/16 (I21k) | BiT-L | Noisy Student |
|---|---|---|---|---|---|
| **ImageNet** | **88.55** | 87.76 | 85.30 | 87.54 | 88.4/88.5 |
| **ImageNet-ReaL** | **90.72** | 90.54 | 88.62 | 90.54 | 90.55 |
| **CIFAR-100** | **94.55** | 93.90 | 93.25 | 93.51 | - |
| **VTAB (19 tasks)** | **77.63** | 76.28 | 72.72 | 76.29 | - |
| **TPUv3-core-days** | **2.5k** | **0.68k** | **0.23k** | 9.9k | 12.3k |

**最重要結果**: ViT-H/14（JFT で事前学習）が ImageNet 88.55% で SOTA、しかも **2.5k TPUv3-core-days で済む**（BiT-L 9.9k、Noisy Student 12.3k の 1/4 以下の計算）。**「同じ計算で 2-4× 高性能」または「同じ性能で 1/4 計算」が ViT の決定的優位**。

### Figure 3-5: 3 つの普遍的スケーリング知見

論文の §4.3-4.4 が示した CV における**スケーリング則の発見**は、後続研究すべての基盤になった：

1. **データ・スケールに対する単調性**: ImageNet（1.3M）→ ImageNet-21k（14M）→ JFT-300M（300M）と進むにつれ、ViT-Large は ViT-Base を追い抜く。**小データセットでは ViT-L < ViT-B**、大データセットで初めて逆転
2. **データ・サイズが帰納バイアスの差を埋める**: ImageNet では BiT > ViT、JFT では ViT > BiT。**「大規模訓練は帰納バイアスに勝る」**（§4.3 の見出し）
3. **未飽和**: 試した最大のモデル（ViT-H/14）と最大のデータ（JFT-300M）でも、性能は飽和しない（§4.4）。これが「**ViT は無限にスケールできる**」という後続研究の動機（DINOv2 1B / DINOv3 7B / SigLIP SO-400M / EVA 1B 等）

### Figure 6-7: ViT の内部動作

- **第 1 層の埋め込みフィルタ（図 7 左）**: PCA で見ると、Gabor 風や低周波基底のような「**畳み込みフィルタに似た**」基底関数が自然に出現
- **位置埋め込みの 2D 構造（図 7 中央）**: 行・列構造が自動的に学習される。**手作りの 2D 位置埋め込みが不要な理由**
- **Attention 距離（図 7 右）**: 一部のヘッドは最低層から**画像全体に注意**、他のヘッドは局所的。**深さと共に attention 距離が増加**。CNN の受容野と類比的な現象が attention から創発

### §4.6: 自己教師あり予備実験

論文末で**マスク・パッチ予測**（BERT 風 MLM の画像版）を予備実験。ViT-B/16 で ImageNet 79.9%（ゼロから訓練比 +2%、教師あり事前学習比 -4%）。これが後の **[[entities/mae|MAE]]**（2021、本 wiki ingest 済み）/ **[[entities/ibot|iBOT]]**（2021）/ **BEiT** / **SimMIM** などの **Masked Image Modeling（MIM）** 系列の出発点となった。

## 限界・批判的視点

### 著者自身が認める限界（§5）

1. **データ要求が大きい**: JFT-300M（Google 内部の非公開 3 億画像）級が望ましく、ImageNet 単独では CNN に劣る
2. **検出・セグメンテーションへの適用は未検証**: 分類のみで評価、密予測タスクは future work
3. **自己教師あり vs 教師ありのギャップ**: SSL は予備的、教師あり事前学習に 4% 劣る

### この wiki の視点からの追加批判

4. **JFT-300M が非公開** — 再現性問題: ViT 論文の最良結果は Google 内部データ依存。OpenCLIP / LAION 等のオープン代替が必要になった理由
5. **位置エンコーディング選択の浅さ**: 「1D は 2D と同等」と結論したが、後の **DINOv3 の axial RoPE** や **Qwen2-VL の M-RoPE** は「**任意解像度への外挿**」という別の重要側面を示し、本論文の比較は限定的だった（[[concepts/rotary-position-embeddings]] / [[sources/roformer]] 参照）
6. **計算量問題は未解決**: $\mathbb{O}(N^2)$ の self-attention は高解像度や小パッチで急速に重くなる。後の **Swin Transformer**（階層型）、**Hiera**（[[entities/hiera]]、SAM 2 のバックボーン）、**Window Attention**（[[entities/qwen2-5-vl|Qwen2.5-VL]] が ViT 全層をゼロから訓練）など、本論文以降の主要研究はこの問題に取り組む
7. **セグメンテーション・検出への弱さ**: 階層性（U-Net 風マルチスケール特徴）の欠如により、**密予測タスクには Swin/Hiera 等の階層型 ViT が支配的**になった。本論文の純粋 ViT は分類用に最適化された設計
8. **小データへの脆弱性**: ImageNet 単独で CNN に勝てない問題は、後に **DeiT**（Touvron 2021、強い拡張 + 蒸留）と **MAE**（Masked Image Modeling、SSL）の 2 系統で解決された
9. **「シンプルすぎる」設計**: 著者らの「**意図的にシンプル**」という設計思想は美しいが、後の研究で 2D-aware 位置（axial RoPE）、relative position bias（Swin）、各種正則化（DeiT）、SSL（DINO/MAE/iBOT）など多数の改良が追加された。**本論文単体ではなく、その後のエコシステム全体が ViT パラダイムの成功**

## 用語と略称

- **ViT**: **V**ision **T**ransformer（本論文の中心モデル名）
- **MSA**: **M**ulti-head **S**elf-**A**ttention（多頭自己注意、Appendix A 参照）
- **MLP**: Multi-Layer Perceptron（多層パーセプトロン）
- **LN**: Layer Normalization（層正規化、各ブロック前に適用 = pre-norm）
- **GELU**: Gaussian Error Linear Unit（活性化関数、本論文の MLP で採用）
- **[CLS] token**: classification token（BERT 由来、画像全体を集約する学習可能トークン）
- **B/L/H**: Base/Large/Huge（モデルサイズ、本論文で命名）
- **/16, /14, /32**: パッチサイズ（`ViT-B/16` のように使う）
- **ImageNet (IN, IN-1k)**: ILSVRC-2012、1k クラス・1.3M 画像（[[entities/imagenet]]）
- **ImageNet-21k (I21k)**: 21k クラス・14M 画像の上位集合
- **JFT-300M**: Google 内部の 18k クラス・300M 画像（**非公開**）
- **BiT**: Big Transfer（[^25]、ResNet ベースの大規模転移学習比較対象）
- **Noisy Student**: 大規模 EfficientNet ベース半教師あり SOTA（本論文の比較対象）
- **VTAB**: Visual Task Adaptation Benchmark（19 タスク、Natural/Specialized/Structured の 3 グループ）
- **iGPT**: image GPT（[^8]、画素を Transformer に直接食わせた先行研究）
- **TPUv3-core-days**: TPU v3 コア × 訓練日数（事前学習計算コストの単位）
- **inductive bias**（帰納バイアス）: モデル構造に組み込まれた仮定（CNN の局所性・平行移動同変性など）
- **patch embedding**（パッチ埋め込み）: パッチを線形射影した D 次元ベクトル
- **position embedding**（位置埋め込み）: 1D 学習可能ベクトル（本論文）

## ViT 後の発展（wiki 既存ページとの接続）

ViT は **CV における基盤モデルの出発点**で、本 wiki にはその後の主要発展がすべて ingest 済み：

### 視覚バックボーンとしての継承

- **[[entities/dino|DINO]]**（Caron et al., Meta, 2021、[[sources/dino-emerging-properties-in-self-supervised-vit]]）: ViT × SSL の代表、自己蒸留で**訓練監督なしに segmentation がエンコードされる**創発を発見
- **[[entities/mae|MAE]]**（He et al., Meta, 2022、[[sources/mae]]）: 本論文 §4.6 の「マスク・パッチ予測」を本格化、75% マスクで ViT-H を ImageNet のみで 87.8% 達成。**Masked Image Modeling（MIM）パラダイムの確立**
- **[[entities/ibot|iBOT]]**（Zhou et al., 2022、[[sources/ibot]]）: DINO + MIM の統合、online tokenizer 概念を導入
- **[[entities/dinov2|DINOv2]]**（Oquab et al., Meta, 2023、[[sources/dinov2-learning-robust-visual-features-without-supervision]]）: ViT-g/14（1.1B パラメータ）× LVD-142M で純粋 SSL の基盤モデル化
- **[[entities/dinov3|DINOv3]]**（Siméoni et al., Meta, 2025、[[sources/dinov3]]）: **ViT-7B × LVD-1689M**、axial RoPE + Gram anchoring。本論文の ViT を CV における究極のスケールまで押し上げた

### マルチモーダル基盤としての継承

- **[[entities/clip|CLIP]]**（Radford et al., OpenAI, 2021、[[sources/clip]]）: ViT を視覚エンコーダとして採用、テキストとの対比学習で**ゼロショット転移**を実現
- **[[entities/siglip|SigLIP]]** / **SigLIP 2**: CLIP の softmax → sigmoid 改良、ViT バックボーン継承
- **[[entities/perception-encoder|PE]]**（Bolya et al., Meta, NeurIPS 2025、[[sources/perception-encoder]]）: 対比学習の頑健スケーリングで**中間層特徴の創発**を発見

### 検出・セグメンテーションへの応用

- **[[entities/sam|SAM]]** / **[[entities/sam-2|SAM 2]]** / **[[entities/sam-3|SAM 3]]**: ViT バックボーンを segmentation foundation model に変換
- **[[entities/detr|DETR]]** / **DINO-detector** / **Grounding-DINO**: CNN バックボーン → ViT バックボーンの流れ

### MLLM（マルチモーダル LLM）の視覚エンコーダ

すべての対話型 VLM が ViT 系を視覚エンコーダに採用：
- **[[entities/qwen-vl|Qwen-VL]]** / **[[entities/qwen2-vl|Qwen2-VL]]** / **[[entities/qwen2-5-vl|Qwen2.5-VL]]** / **[[entities/qwen3-vl|Qwen3-VL]]**
- **[[entities/internvl|InternVL]]** 系列（InternViT-6B、InternViT-300M）
- **[[entities/gemma-3|Gemma 3]]**（SigLIP 400M variant 採用）
- **[[entities/deepseek-ocr|DeepSeek-OCR]]**（SAM-base + CLIP-large、いずれも ViT）

### ViT が解決できなかった問題への後継研究

- **位置エンコーディング**: ViT の学習可能 1D 位置埋め込みは解像度固定 → **[[sources/roformer|RoPE]]** / **[[concepts/rotary-position-embeddings]]** が任意解像度対応を実現
- **計算量 $\mathbb{O}(N^2)$**: 高解像度で重い → **[[entities/hiera|Hiera]]**（階層型）、**Window Attention**（Qwen2.5-VL）、**Pixel Shuffle/Pooling**（Gemma 3、InternVL）
- **密予測の弱さ**: Swin Transformer（階層型）、Hiera、ViT-Det などが分類に最適化された純粋 ViT を改良

## 関連ページ

### 直接の派生・拡張概念

- [[concepts/vision-transformer]] — ViT の概念解説。本論文がその「原典」
- [[concepts/rotary-position-embeddings]] — ViT の学習可能 1D 位置埋め込みを置き換える RoPE（CV では DINOv3 / SAM 2 / Qwen2-VL 系などで採用）
- [[sources/roformer]] — RoPE 原典（NLP 発祥だが CV で開花）

### ViT を視覚バックボーンに採用したモデル（網羅的）

**SSL 系**:
- [[sources/dino-emerging-properties-in-self-supervised-vit]] / [[entities/dino]] — DINO（ViT × self-distillation）
- [[sources/mae]] / [[entities/mae]] — MAE（ViT × MIM、本論文 §4.6 の発展）
- [[sources/ibot]] / [[entities/ibot]] — iBOT（DINO + MIM）
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] / [[entities/dinov2]] — DINOv2（ViT-g/14、1.1B）
- [[sources/dinov3]] / [[entities/dinov3]] — DINOv3（ViT-7B、axial RoPE + Gram anchoring）
- [[sources/byol]] / [[entities/byol]] — BYOL（非対比 SSL、ResNet/ViT 両対応）
- [[sources/simclr]] — SimCLR（対比 SSL、ResNet ベース、ViT への拡張は後続）

**Vision-Language 系**:
- [[sources/clip]] / [[entities/clip]] — CLIP（ViT 視覚エンコーダ）
- [[sources/siglip]] / [[entities/siglip]] — SigLIP（CLIP 改良）
- [[sources/siglip-2]] — SigLIP 2
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — PE（対比学習の頑健スケーリング）

**検出・セグメンテーション系**:
- [[sources/segment-anything]] / [[entities/sam]] — SAM（ViT-H/16）
- [[sources/sam-2]] / [[entities/sam-2]] — SAM 2（Hiera = 階層型 ViT）
- [[sources/sam-3]] / [[entities/sam-3]] — SAM 3（PE backbone）
- [[sources/detr]] / [[entities/detr]] — DETR（後継で ViT バックボーン採用）
- [[sources/grounding-dino]] / [[entities/grounding-dino]] — Grounding-DINO

**対話型 VLM / MLLM 系**:
- Qwen 系: [[sources/qwen-vl]] / [[sources/qwen2-vl]] / [[sources/qwen2-5-vl]] / [[sources/qwen3-vl]]
- InternVL 系: [[sources/internvl]] / [[sources/internvl-1-5]] / [[sources/mini-internvl]] / [[sources/internvl-2-5]] / [[sources/internvl-3]] / [[sources/internvl-3-5]]
- [[sources/gemma-3]] — Gemma 3
- [[sources/deepseek-ocr]] — DeepSeek-OCR

**生成系（ViT を採用 / DiT への発展）**:
- [[sources/sdxl]] / [[entities/sdxl]] — SDXL（UNet 系だが、後継の SD3 / FLUX は DiT = Diffusion Transformer に移行、ViT の生成系版）

### 上位概念

- [[concepts/foundation-model]] — 基盤モデル全体。ViT はその視覚側基盤
- [[sources/foundational-models-vision-survey]] — Awais et al. survey で**ViT バックボーンは「Adapter LLM」を含む 4 アーキテクチャ・スタイルすべての視覚側で広く採用**と整理

### データセット

- [[entities/imagenet]] — ILSVRC-2012、ViT 評価の中心ベンチマーク
- ImageNet-21k（公開、14M）/ JFT-300M（Google 内部、非公開、300M）— ViT が**スケーリングの恩恵を受けるためのデータ規模**を示した

### 関連 question

- [[questions/vit-dynamic-resolution-evolution]] — ViT の学習可能 1D 位置埋め込みの解像度制約から、Qwen2-VL の Naive Dynamic Resolution、InternVL 1.5 / Gemma 3 のタイル分割、DeepSeek-OCR の視覚トークン数最小化までの「**任意解像度 ViT への進化**」を整理した question ページ
