---
type: entity
entity_kind: model
aliases: [BiomedCLIP, BiomedCLIP-PubMedBERT_256-vit_base_patch16_224]
tags: [biomedclip, medical-imaging, vision-language, clip, contrastive-learning, foundation-model, open-data, microsoft]
related: ["[[entities/clip]]", "[[entities/pmc-15m]]", "[[entities/medsiglip]]", "[[concepts/medical-foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/contrastive-learning]]", "[[entities/eva-x]]"]
sources: ["[[sources/biomedclip]]"]
updated: 2026-08-28
---

# BiomedCLIP

Microsoft Research + University of Washington による**生物医学の視覚–言語基盤モデル**（arXiv:2303.00915, 2023）。詳細解説: [[sources/biomedclip]] / 翻訳: [[translations/biomedclip]]。

## 一言で

**論文の図とキャプションは、そのまま画像–テキスト対である。** PubMed Central の 440 万件の全文論文から [[entities/pmc-15m|PMC-15M]]（1,500 万対）を掻き集め、[[entities/clip]] の領域に合わない部品だけを取り替えて生物医学向けに作り直した。**訓練データが完全に公開されており再現可能**なのが最大の特徴。

## 構成

```
BiomedCLIP = ViT-B/16 (224px, ImageNet 初期化)  ←→  PubMedBERT (WordPiece 30k, 文脈 256)
                          InfoNCE 対比損失（CLIP と同一）
                       事前学習: PMC-15M（1,500 万対）
```

| 箇所 | CLIP | **BiomedCLIP** | 理由 |
|---|---|---|---|
| テキストエンコーダ | GPT-2（白紙から） | **PubMedBERT** | 生物医学の事前学習済み言語モデル |
| トークナイザ | Byte-Pair Encoding | **WordPiece**（30k 領域特有語彙） | BPE は専門用語を細断してしまう |
| 文脈長 | 77 | **256** | PMC キャプションの 90% を覆う |
| 視覚エンコーダ | ViT-B/16 or ResNet-50 | **ViT-B/16** | S→M→B で単調に改善 |
| 解像度 | 224 | **224** | **384 は検証で良くなるが下流で悪化**（後述） |
| 損失 | InfoNCE | **InfoNCE**（変更なし） | — |

## アブレーションの要点

**テキスト側の効き方**（検証 img2txt R@1）

| text encoder | vocab | context | R@1 |
|---|---|---|---|
| GPT | 50k 汎用 | 77 | 64.53 |
| PubMedBERT | 30k 領域特有 | 77 | 69.03 |
| **PubMedBERT** | **30k 領域特有** | **256** | **73.50** |

**ViT の規模**（同上）: ViT-S/16 22M → 69.45 / ViT-M/16 39M → 71.85 / **ViT-B/16 86M → 73.50**

### ⚠ 検証で良くなったのに下流で悪くなった 3 件

**本モデルのアブレーションで最も価値のある記録。**

| 選択肢 | 検証 R@1 | 下流 | 採用 |
|---|---|---|---|
| 解像度 **224** vs 384 | 82.90 → **84.63**（384 が勝ち） | ゼロショット分類平均 **75.52** → 70.37（**224 が勝ち**）、訓練時間は 1.92 倍 | **224** |
| バッチ **4k** vs 64k | 83.98 → **87.32**（64k が勝ち） | 「利得は下流に**転移しない**」 | **4k** |
| **ImageNet 初期化** vs ランダム | 83.15（ランダムがわずかに勝ち） | 「ImageNet 初期化の方が**より安定**」 | **ImageNet** |

解像度 384 では PCam 73.41 → 67.15、TCGA-TIL 67.04 → 57.00 と大きく落ちる。**検証 retrieval の R@1 を信じて選んでいたら 3 つとも間違えていた。**

64k バッチが効かない理由の著者の説明は「**極端に大きなバッチはより多くの訓練データを要する。CLIP は 4 億対だが PMC-15M は 1,500 万対**」。

## 主要結果

### クロスモーダル検索（PMC-15M テスト 725,739 対）

| Model | txt2img R@1 | R@5 | R@10 |
|---|---|---|---|
| **BiomedCLIP (BERT)** | **69.60** | **86.30** | **90.20** |
| BiomedCLIP (GPT) | 59.60 | 80.40 | 85.70 |
| CLIP | 8.48 | 16.20 | 20.10 |
| **PubMedCLIP** | **1.00** | 2.51 | 3.59 |

**PubMedCLIP は汎用 CLIP の 1/8。** 狭いデータ（ROCO 8 万の放射線対）で汎用 CLIP をファインチューニングした結果、**汎用性を失ったうえに新領域も広くカバーできない**という最悪の組み合わせになった。

### ゼロショット画像分類

| Dataset | BiomedCLIP (BERT) | (GPT) | 他の最良 |
|---|---|---|---|
| PCam | **~73.5** | ~70.1 | PubMedCLIP ~63.5 |
| LC25000 (Lung) | ~65.3 | ~60.6 | **PLIP ~81** |
| LC25000 (Colon) | ~93.0 | ~89.2 | **LLaVA-Med ~100** |
| RSNA | ~79.0 | **~81.0** | PubMedCLIP ~70.7 |
| TCGA-TIL (AUROC) | ~67.0 | **~70.5** | PLIP ~68 |

- **PLIP は自分の土俵（肺の病理）で 16pt 上回る**が RSNA では崩れる
- **Med-Flamingo は RSNA で ~32 と壊滅**（プロンプトによらず常に同じ答えを生成するため）
- **GPT 版が RSNA と TCGA-TIL で BERT 版を上回る**——検証 retrieval では BERT 版が 10pt 圧勝なのに逆転する（4 つ目の「検証と下流の乖離」）

### RSNA での正の転移（本モデルの目玉）

> **BiomedCLIP はラベル付きデータのわずか 10% を用いるだけで、完全教師ありの BioViL を既に上回る。**

BioViL は MIMIC-CXR で訓練された**放射線科特化**モデル。しかも論文は「BiomedCLIP の放射線関連画像は MIMIC-CXR より多くはなく、対はおそらくはるかにノイジー」と明記して、「放射線データを多く見たから勝った」という解釈を潰している。**多様性そのものが放射線科の性能を上げた＝正の転移**。

### 医療 VQA

| Model | VQA-RAD Overall | SLAKE Overall | VQA-RAD トークン F1 |
|---|---|---|---|
| **BiomedCLIP** | 72.70 | 86.10 | **73.13** |
| LLaVA-Med | **75.80** | **87.00** | – |
| BiomedGPT-B | 73.20 | 86.50 | – |
| PubMedCLIP | 72.10 | 80.10 | – |
| Med-PaLM M (562B) | – | – | 62.06 |

**トークンレベル F1 で 562B の Med-PaLM M に 11pt 差**をつける一方、**全体精度では最上位ではない**。明確に勝つのは **VQA-RAD の開放型（67.00、次点 LLaVA-Med 64.75）**とトークン F1。

なお **LLaVA-Med は BiomedCLIP を中核の視覚エンコーダとして使っている**ので、「BiomedCLIP に負けた」のではなく「BiomedCLIP の上に指示追従を足したものが勝った」。

## 弱点

- **PMC-15M の大半は医療画像ではない**。上位の画像型は統計図表・グラフ・チャート・表・フローチャートといった汎用の科学図解
- **論文の図は「見せたい所見」しか写さない**。臨床画像の実地の分布とは別物
- **画像の半分が複合図**で、そのまま 1 対として扱っている（分割版 PMC-Fine-Grained-46M は事前学習に未使用）
- **引用箇所（citances）を使っていない**
- **プライバシー保護の代理は論文が言うより弱い**。F1 は全条件で 12〜42%、**Cardiomegaly では汎用 CLIP に 5 倍負ける**（6.99 vs 36.24）。論文は「論文のキャプションは臨床レポートほど包括的でない」と説明
- 評価がゼロショット / linear probe / VQA に閉じており、密なタスク（セグメンテーション・検出）を扱っていない
- **本文と図の数値が食い違う**（本文「上位 5 件 77% 超 / 上位 1 件 56% 超」vs 図2A の R@5 86.30 / R@1 69.60）

## 系譜・位置づけ

- **[[entities/clip]] の直系の医療派生**。損失も骨格もそのままで、領域に合わない部品だけを取り替えた
- **[[entities/medsiglip]] とは正反対の設計**。BiomedCLIP は医療データ 100% でほぼゼロから、MedSigLIP は SigLIP に医療を 2% だけ混ぜる。**「汎用性を捨てて専門性を取る」か「汎用性を保ったまま専門性を足す」か**の両端（[[concepts/medical-foundation-model]]）
- **PubMedCLIP の失敗が両者をつなぐ教訓**。狭いデータでのファインチューニングは汎用性も専門性も失いうる。BiomedCLIP は「データを 2 桁増やして置き換える」、MedSigLIP は「元を残して薄く混ぜる」ことで回避している
- **下流で広く使われた**。**LLaVA-Med の視覚エンコーダ**が代表例で、本モデルの真の価値は基盤として使われることにある
- [[entities/eva-x]]（胸部 X 線特化 SSL）とは「特化の深さ vs 多様性の広さ」で対照的。RSNA の結果は多様性側に一本取った例

## 公開

- モデル・コード・PMC-15M 再現スクリプト: <https://aka.ms/biomedclip>
- Hugging Face: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- OpenCLIP ベースの実装。事前学習は A100 / V100 × 16 台

## 関連ページ

- [[sources/biomedclip]] — 論文要約（最重要）
- [[translations/biomedclip]] — 全文和訳（Supplementary Note 込み）
- [[entities/pmc-15m]] — 訓練データ
- [[entities/clip]] / [[sources/clip]] — 直接の祖
- [[entities/medsiglip]] — 正反対の設計の医療 CLIP 系エンコーダ
- [[concepts/medical-foundation-model]] — 医療基盤モデルの類型
- [[concepts/weakly-supervised-pretraining]] / [[concepts/contrastive-learning]] — 属するパラダイム
- [[entities/eva-x]] — 単一モダリティ特化との対比
