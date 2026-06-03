---
type: source
source_path: raw/papers/Grounded Language-Image Pre-training.md
source_kind: paper
title: "Grounded Language-Image Pre-training"
authors: [Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao]
year: 2022
venue: CVPR 2022
ingested: 2026-05-28
tags: [glip, object-detection, phrase-grounding, vision-language, open-vocabulary, microsoft, ucla, prompt-tuning, deep-fusion, self-training]
translation: "[[translations/glip]]"
---

# GLIP: Grounded Language-Image Pre-training

> 原典: [[translations/glip]] ・ `raw/papers/Grounded Language-Image Pre-training.md`
> 著者: Liunian Harold Li, Pengchuan Zhang, Haotian Zhang ら（UCLA / Microsoft Research / U. Washington / U. Wisconsin / Microsoft Cloud and AI / IDEA）
> 出典: CVPR 2022 / arXiv 2112.03857
> arXiv: <https://arxiv.org/abs/2112.03857>
> コード: <https://github.com/microsoft/GLIP>

## 一言まとめ

**「物体検出 = phrase grounding として再定式化し、CLIP 流の対比学習を `region-word alignment` レベルで実現した最初の本格的成功例」**。box 分類器のロジットを「region 視覚特徴と text token 言語特徴の内積（alignment score）」に置き換えるだけで、**任意の検出モデルを grounding モデルに変換** できる統一定式化を提案。**language-aware deep fusion**（X-MHA で複数層で融合）と **self-training で 24M の web 画像-テキストペアに自動 box 注釈** を加えて、COCO ゼロショットで 49.8 AP、fine-tune で 61.5 AP（SOTA）、LVIS で rare クラスを大幅改善、13 downstream tasks（ODinW）で 1-shot fully-supervised に匹敵。**open-vocabulary 検出パラダイムを確立**し、Grounding DINO / SAM 3 / OWL-ViT 系の祖となった。

## 背景と問題意識

2021 年末時点の CV における 2 つの強力なトレンドの間にあったギャップ：

| パラダイム | 代表 | 強み | 弱み |
|---|---|---|---|
| **CLIP（画像レベル WSL）** | [[entities/clip]] | open-vocab 画像分類、ゼロショット | **オブジェクトレベル特徴が弱い**（[[concepts/zero-shot-transfer]]）、検出/セグに直接使えない |
| **古典的物体検出** | Faster RCNN, DyHead | 高精度 | **事前定義された固定クラス集合**（COCO 80, LVIS 1203）、open-vocab 不可 |
| **ゼロショット検出** | Bansal et al. 2018, ViLD 2021 | 既知クラスから未知へ汎化 | 訓練データに rare クラスを明示的に除外、scale が限定的 |
| **MDETR** | [Kamath et al., ICCV 2021] | テキスト条件付き検出（grounding） | 既存のマルチモーダル注釈データ（small scale）のみ |

**GLIP の問題意識**: 「CLIP の semantic richness を **オブジェクトレベル** で実現するには？」「画像-テキストデータを **検出スケール** にどう活用するか？」

> **補足: なぜ画像レベル CLIP では検出に不十分か** — CLIP は「画像全体 ↔ キャプション全体」の対応を学ぶので、各 region の細粒度な意味を学ばない。物体検出は本質的に「region ↔ 概念」のマッチングなので、CLIP の **dense / object-level 版** が必要だった。

## 提案手法 / 主張

### 全体構造

```
画像 → Image encoder（Swin-T or Swin-L + DyHead）─┐
                                                  │
                                            X-MHA で deep fusion（L 層）
                                                  │
テキスト → Text encoder（BERT + 追加 BERTLayer）──┘
            ("person. bicycle. car. ... toothbrush")
                                                  
   ↓ 最終層の region 特徴 O ∈ R^{N×d} と token 特徴 P ∈ R^{M×d}
   ↓
   Alignment score: S_ground = O · P^T  ∈ R^{N×M}
   ↓
   target T'（各 region がどの token に対応するか）と Focal/CE 損失
```

**3 つの中核技術**：

### 1. Unified Formulation（検出 = grounding として再定式化）

通常の物体検出の box 分類損失：

$$
S_{\text{cls}} = OW^T, \quad \mathcal{L}_{\text{cls}} = \text{loss}(S_{\text{cls}}; T)
$$

ここで $W \in \mathbb{R}^{c \times d}$ は **固定された $c$ 個のクラス重み**。

GLIP の grounding 損失：

$$
S_{\text{ground}} = OP^T, \quad \mathcal{L}_{\text{cls}} = \text{loss}(S_{\text{ground}}; T')
$$

ここで $P \in \mathbb{R}^{M \times d}$ は **テキストエンコーダから動的に計算される token 特徴**。クラス名を joined した文字列（"person. bicycle. car. ..."）をプロンプトとして与える。

**理論的等価性**: 単一プロンプトで全クラスが収まる場合、検出と grounding は **完全に等価**（COCO で DyHead と GLIP-T(A) が同じ 49.4 AP を出すことで実証）。

> **補足: prompt 形式の工夫** — "Detect: person, bicycle, ..." より "person. bicycle. car. ..." の方が良い性能を出す。BERT が "Detect:" を意味のあるトークンとして処理しないため。これは [[concepts/zero-shot-transfer]] の prompt engineering と同じ精神。

**sub-word 対応**: "toothbrush" → "tooth#" + "#brush" のような分割があるため、target を $T \in \{0,1\}^{N \times c}$ から $T' \in \{0,1\}^{N \times M}$ に拡張（phrase が positive なら全 sub-word を positive 扱い）。

### 2. Language-Aware Deep Fusion（X-MHA）

CLIP のような **late-fusion**（最後だけ dot product）では grounding 性能が不足する。GLIP は **複数層で X-MHA（cross-modality multi-head attention）** を使って融合：

```
O^{i+1} = DyHeadModule(O^i + O^i_{t2i})   ← text-to-image context
P^{i+1} = BERTLayer(P^i + P^i_{i2t})       ← image-to-text context

O^i_{t2i}, P^i_{i2t} = X-MHA(O^i, P^i)
```

**X-MHA**: 各モダリティから query、他方から key/value を計算して context vector を作る（典型的な cross-attention）。

**deep fusion の 2 つの効果**:
1. **phrase grounding 性能の向上**: 言語コンテキストが視覚特徴を guide
2. **言語認識視覚特徴**: モデルの出力がテキストプロンプトに条件付けされる → **prompt tuning が機能する基盤**（§5.2 で実証）

### 3. Self-Training で grounding データをスケール

```
Stage 1: gold data で teacher GLIP-T(C) を訓練
   ├─ Objects365（0.66M 画像、365 クラス、検出データ）
   └─ GoldG（0.8M 人手注釈 grounding、Flickr30K + VG + GQA、MDETR キュレーション）
            ↓
Stage 2: teacher で web 画像-テキストペアに疑似 box 注釈
   ├─ Cap4M（4M ペア、GLIP-T 用）
   └─ Cap24M = CC12M + SBU（24M ペア、GLIP-L 用）
   NLP parser で名詞句を抽出 → teacher が box を予測
            ↓
Stage 3: gold + 疑似データで student GLIP-T / GLIP-L を訓練
   GLIP-L: FourODs（2.66M 検出）+ GoldG + Cap24M = 計 27M grounding データ
```

**24M ペアから 78.1M phrase-box pseudo annotations**（信頼度 > 0.5）、**5840 万のユニーク名詞句**。

> **補足: なぜ student が teacher を超えるのか** — teacher は gold データに含まれない概念（例: vaccine, turquoise）を直接知らないが、**言語コンテキストから推測** できる: 「小さい vial を localize できれば、'vaccine' を localize できる」「'caribbean sea' を localize できれば、'turquoise' を localize できる」。teacher の「教育された推測」が student の supervised signal になる。

> **補足: 既存の検出データのスケールアップとの違い** — ViLD のような従来の self-training 検出スケールアップは「teacher の 2000 カテゴリ語彙の範囲内」しか拡張できない。GLIP は **言語パーサで生成された任意の名詞句**（5840 万種類）を扱えるので、語彙が事実上無限。

## 実験結果と知見

### バリアントと ablation（表 1）

| Model | Backbone | Deep Fusion | Detection | Grounding | Caption |
|---|---|---|---|---|---|
| GLIP-T (A) | Swin-T | ✗ | O365 | - | - |
| GLIP-T (B) | Swin-T | ✓ | O365 | - | - |
| GLIP-T (C) | Swin-T | ✓ | O365 | GoldG | - |
| GLIP-T | Swin-T | ✓ | O365 | GoldG | Cap4M |
| **GLIP-L** | **Swin-L** | ✓ | FourODs | GoldG | **Cap24M** |

### COCO ゼロショット（表 2）

| Model | Pre-train | Zero-Shot AP |
|---|---|---|
| DyHead-T | O365 | 43.6 |
| GLIP-T (A) | O365 | 42.9（再定式化で微減） |
| GLIP-T (B) | O365 | 44.9（+deep fusion で +2.0） |
| GLIP-T (C) | O365 + GoldG | **46.7（+gold grounding で +1.8）** |
| **GLIP-L** | FourODs + GoldG + Cap24M | **49.8** |

**Faster RCNN（教師あり 42.0 AP）を GLIP-L がゼロショットで上回る** — 物体検出のパラダイムシフト。

### COCO Fine-tune（表 2）

GLIP-L (Swin-L + FourODs + GoldG + COCO) で **test-dev 61.5 AP** — bells and whistles なしで先行 SOTA を超える。

### LVIS ゼロショット（表 3）

| Model | MiniVal APr | APc | APf | AP |
|---|---|---|---|---|
| MaskRCNN（教師あり） | 26.3 | 34.0 | 33.9 | 33.3 |
| GLIP-T (B)（O365 のみ） | 13.5 | 12.8 | 22.2 | 17.8 |
| GLIP-T (C)（+ GoldG） | 17.7 | 19.5 | 31.0 | 24.9 |
| GLIP-T（+ Cap4M） | **20.8** | 21.4 | 31.0 | 26.0 |
| **GLIP-L** | **28.2** | **34.3** | **41.5** | **37.3** |

- **GLIP-L (28.2 APr) は教師あり MaskRCNN (26.3 APr) を超える**
- grounding データ追加で APr が **+4.2 ポイント** 改善
- web grounding データでさらに **+3.1 ポイント** 改善

### Flickr30K phrase grounding（表 4）

GLIP-L で **R@1 test 87.1**（MDETR の 84.3 から +2.8 ポイント）。

### Object Detection in the Wild (ODinW)（13 データセット、図 4）

新規ベンチマーク: EgoHands, Pothole, ThermalDogsandPeople 等の **多様な実世界タスク**。

- **ゼロショット GLIP-T > 5-shot DyHead-T**
- **1-shot GLIP-L ≈ fully-supervised DyHead-T**
- **prompt tuning が full-model tuning とほぼ同じ性能を達成**（GLIP-L で）

### 検出データ vs grounding データのスケーリング比較（表 5）

| Pre-train | COCO AP | LVIS APr |
|---|---|---|
| O365（0.66M 検出） | 44.9 | 13.5 |
| O365 + GoldG（+0.8M grounding） | 46.7 | **17.7** |
| FourODs（2.66M 検出、人手） | 46.3 | 15.0 |

**0.66M + 0.8M grounding が 2.66M 検出データを APr で上回る** — 「grounding データの語彙の豊かさが検出の規模より重要」。

### Manual prompt tuning（図 6）

stingray を直接検出できない → プロンプトに "flat and round" を追加 → **AP50 が 4.6 → 9.7**。**ユーザーが言語で domain knowledge を注入できる**。

## 限界・批判的視点

- **DyHead 検出器に依存**: 後続研究は DETR ベース（Grounding DINO）に移行。GLIP の貢献は detection-grounding 統一定式化にあり、検出器自体は古典的
- **deep fusion の計算コスト**: late-fusion の CLIP と比べてメモリと FLOPS が大きい
- **疑似ラベルの品質依存**: teacher が間違った grounding をすると student に伝播
- **言語エンコーダの長さ制限**: BERT は 512 token、実装は 256 token に制限 → Objects365 の 365 カテゴリは複数プロンプトに分割する必要あり（わずかな性能低下）
- **テキスト依存の inference**: 検出時に必ずテキストプロンプトが必要、image-only の使い方ができない
- **prompt design への敏感性**: "Detect:" を入れるかどうかなど、表現で性能が変動

## 既存 wiki との接続

### Open-Vocabulary 検出パラダイムの確立

[[concepts/object-detection]] の系譜で、GLIP は **クローズドセット検出 → open-vocabulary 検出** への転換点：

```
[クローズドセット]                    [open-vocabulary]
Faster R-CNN, YOLO, DyHead    ──→    GLIP（2022 CVPR）★
                                       ├── ★ Grounding DINO（[[sources/grounding-dino]] / [[entities/grounding-dino]]、2023/ECCV 2024、DETR 版、[[entities/dino-detector]] の後継、ODinW ZS 26.1 SOTA）★
                                       ├── OWL-ViT（ViT + DETR、Google）
                                       ├── MDETR（GLIP より先、テキスト条件付き DETR）
                                       ├── ★ YOLO-World（[[sources/yolo-world]] / [[entities/yolo-world]]、CVPR 2024、YOLOv8 + CLIP + RepVL-PAN、52 FPS で LVIS 35.4 AP）★
                                       │     ※ GLIP-L を疑似ラベリング teacher として活用 — GLIP の self-training パイプラインを継承
                                       ├── [[entities/sam-3]]（2025、PCS、PE + DETR + presence head）
                                       └── X-Decoder, SEEM, etc.
```

GLIP は **MDETR と DETR を踏襲しつつ、CLIP の semantic richness と self-training を組み合わせた**。後続の **[[sources/grounding-dino|Grounding DINO]]** は GLIP の「検出 = grounding」を **DINO 検出器**（[[entities/dino-detector]]）の枠組みに移植 + **tight 3-phase fusion**（GLIP の Phase A のみ → A+B+C）+ **sub-sentence text representation** で発展させたもの。ODinW ゼロショット 26.1 AP で Florence (841M) を 341M で超える SOTA を達成。

### SAM 3 の前駆としての GLIP

[[sources/sam-3]] / [[entities/sam-3]] の **PCS タスク（Promptable Concept Segmentation）** は、GLIP の「テキスト名詞句から全インスタンス検出」を **セグメンテーションに拡張** したもの。SAM 3 の構造的継承：

| 要素 | GLIP（2022） | SAM 3（2025） |
|---|---|---|
| 入力 | 名詞句プロンプト | 名詞句または画像 exemplar |
| 画像 encoder | Swin + DyHead | **Perception Encoder**（[[entities/perception-encoder]]） |
| テキスト encoder | BERT | PE のテキスト塔 |
| 融合 | X-MHA（DyHead と BERT 間） | **Fusion encoder**（PE 画像 ↔ プロンプト） |
| 検出 | DyHead box | **DETR decoder**（[[entities/detr]] / [[entities/dino-detector]] の系譜） |
| マスク | なし | MaskFormer 適応 |
| 認識/位置分離 | なし | **Presence head**（[[concepts/promptable-concept-segmentation]] の核） |

SAM 3 の "presence head" は、GLIP の弱点だった「文脈なしの絶対的存在判断」を解決する革新。

### CLIP からの発展系統

[[entities/clip]] / [[sources/clip]] からの系統：

| 軸 | CLIP（2021） | **GLIP（2022, 本論文）** | SigLIP（2023） | PE（2025） |
|---|---|---|---|---|
| タスク | 画像分類 | **物体検出 + grounding** | 画像分類 | 検出 + MLLM + dense |
| 融合 | late（最後の dot product） | **deep（X-MHA 複数層）** | late | alignment tuning |
| ゼロショット | 画像分類 | **物体検出（49.8 AP）** | 画像分類 | 検出 + MLLM |
| 学習データ | 400M pairs | **27M grounding** | 10B + WebLI | 5.4B + 22M video |
| 主な貢献 | プロンプトベース zero-shot | **検出 = grounding 統一** | sigmoid loss | 中間層特徴 + alignment |

GLIP は **「CLIP を object-level に持ち込む」アプローチの最も成功した例の 1 つ**。ViLD（CLIP 蒸留型）と並ぶ open-vocab 検出のパイオニア。

### Prompt Tuning の物体検出への持ち込み

§5.2 の発見は重要: **物体検出で prompt tuning が full-tuning に並ぶのは GLIP のような deep fusion モデルでのみ**。shallow-fused モデル（CLIP）では prompt tuning は linear probing 程度。これは [[concepts/parameter-efficient-fine-tuning]] の文脈で重要な知見。

## 用語と略称

- **GLIP** = **G**rounded **L**anguage-**I**mage **P**re-training
- **phrase grounding** = 文中の phrase（単語または句）と画像内の物体/領域の対応を見つけるタスク
- **region-word alignment** = 画像 region と単語 token の埋め込み内積。GLIP の中核
- **unified formulation** = 物体検出を grounding として再定式化する GLIP の統一定式化
- **X-MHA** = Cross-Modality Multi-Head Attention。GLIP の deep fusion の核
- **late fusion** = 最後の層だけで vision-language を融合（CLIP, ALIGN）
- **deep fusion** = 複数層で vision-language を相互融合（GLIP, MDETR）
- **DyHead** = Dynamic Head（[Dai et al., CVPR 2021]）、GLIP の検出 backbone。注意機構ベースの検出ヘッド
- **BERT** = Bidirectional Encoder Representations from Transformers（[Devlin et al., 2019]）、GLIP のテキスト encoder
- **MDETR** = Modulated DETR（[Kamath et al., ICCV 2021]）、テキスト条件付き DETR。GLIP の先駆け
- **ViLD** = Vision-and-Language knowledge Distillation（[Gu et al., ICLR 2022]）、CLIP を two-stage 検出器に蒸留
- **Objects365 / O365** = 365 クラス、0.66M 画像の大規模検出データセット
- **GoldG** = MDETR キュレーションの 0.8M 人手注釈 grounding データ（Flickr30K + VG + GQA、COCO 除外）
- **GoldG+** = COCO 含む 1.3M grounding データ
- **Cap4M / Cap24M** = web からの 4M / 24M 画像-テキストペア（GLIP 生成 box 付き）
- **CC / CC12M** = Conceptual Captions（3M / 12M ペア、Google）
- **SBU** = SBU Captions（1M ペア、Ordonez et al., 2011）
- **FourODs** = 4 つの検出データセット結合（Objects365 + OpenImages + VG（除 COCO）+ ImageNetBoxes、2.66M）
- **LVIS** = Large Vocabulary Instance Segmentation（1203 クラス、長尾分布）
- **APr / APc / APf** = LVIS の rare / common / frequent カテゴリの AP
- **ODinW** = Object Detection in the Wild、GLIP が導入した 13 データセットベンチマーク
- **EgoHands / Pothole / ThermalDogsandPeople** = ODinW の代表例（自己中心、道路穴、赤外線）
- **Flickr30K Entities** = phrase grounding 標準ベンチマーク（Plummer et al., 2015）
- **Visual Genome (VG)** = 110,689 ユニーク phrase を持つ大規模グラフベース画像注釈
- **GQA** = Question Answering ベースの visual reasoning データセット
- **Glove** = 古典的単語埋め込み（Pennington et al., 2014）、Bansal et al. 2018 のゼロショット検出の祖
- **prompt tuning** = タスク特有のプロンプト embedding のみを学習する手法（モデル本体は固定）
- **linear probing** = 最終層の線形分類器のみを訓練する転送学習設定
- **manual prompt tuning** = ユーザーが言語で domain knowledge を追加する zero-shot 改善法

## 関連ページ

- [[translations/glip]] — 本文全文の和訳
- [[entities/glip]] — GLIP モデルの詳細スペック
- [[concepts/object-detection]] — 物体検出全体（GLIP は open-vocab 転換点）
- [[concepts/zero-shot-transfer]] — GLIP のゼロショット転送（CLIP の object-level 拡張）
- [[concepts/promptable-concept-segmentation]] — SAM 3 の PCS タスクの祖
- [[entities/clip]] / [[sources/clip]] — CLIP の object-level 拡張としての GLIP
- [[entities/sam-3]] / [[sources/sam-3]] — GLIP のパラダイムを segmentation に拡張
- [[entities/detr]] / [[sources/detr]] — 検出 Transformer の祖、Grounding DINO への流れ
- [[entities/dino-detector]] / [[sources/dino-detector]] — Grounding DINO の DETR 側祖（DINO 検出器）
- [[sources/grounding-dino]] / [[entities/grounding-dino]] — GLIP × DINO 検出器、高精度路線の発展
- [[sources/yolo-world]] / [[entities/yolo-world]] — GLIP-L を疑似ラベル teacher として活用、リアルタイム路線の発展
- [[concepts/parameter-efficient-fine-tuning]] — prompt tuning は PEFT の一形態
- [[concepts/contrastive-learning]] — region-word alignment は対比学習の object-level 版
