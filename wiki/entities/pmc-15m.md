---
type: entity
entity_kind: dataset
aliases: [PMC-15M, PMC-Fine-Grained-46M, PMC-OA]
tags: [dataset, medical-imaging, image-text-pairs, pubmed-central, open-data, microsoft]
related: ["[[entities/biomedclip]]", "[[entities/wit-400m]]", "[[entities/lvd-142m]]", "[[entities/sa-1b]]", "[[concepts/medical-foundation-model]]", "[[concepts/weakly-supervised-pretraining]]"]
sources: ["[[sources/biomedclip]]"]
updated: 2026-08-28
---

# PMC-15M

**PubMed Central Open Access Subset（PMC-OA）の全文論文から抽出した、1,500 万の図–キャプション対のデータセット**（Microsoft Research, arXiv:2303.00915）。[[entities/biomedclip]] の訓練データ。

## 一言で

**科学論文の図には必ずキャプションが付いている。それはそのまま画像–テキスト対である。** 440 万件の公開全文論文から機械的に抜くだけで、**15,282,336 対**——既存の医療画像–テキストデータセットより **2 桁大きい**ものが、**完全に公開された形で**手に入る。

## 規模の比較

| データセット | 対の数 | 領域 | 公開性 |
|---|---|---|---|
| ARCH | 7,562 | 病理 | 公開 |
| ROCO | 87,952 | 放射線 | 公開 |
| CheXpert | 224,316 | 胸部 X 線 | 公開 |
| MIMIC-CXR | 377,110 | 胸部 X 線 | 公開（要申請） |
| **PMC-15M** | **15,282,336** | **30 の画像型** | **完全公開・再現可能** |

MIMIC-CXR の**約 40 倍**。既存データセットが抱えていた 3 つの限界（**私的である / 小さい / 胸部 X 線に偏る**）を一度に解いている。

## 作り方

```
PMC-OA（440 万件の公開全文論文、2022-06-15 時点）
  ↓ 完全パッケージ（XML / PDF / メディア / 補足資料）を展開
  ↓ PubMed Parser で XML を解析
  ↓ 図ファイルと対応するキャプションを PMID / PMCID とともに抽出
  ↓ 図の参照を持たない論文・不正な XML は除外
PMC-15M（15,282,336 対、300 万件超の異なる論文から）
```

処理は **Azure Databricks**（Apache Spark）で並列に実行。

## 統計的な特徴

<figure>

![](../../raw/assets/biomedclip/fig1.png)

<figcaption>図1（再掲、A・B）: A は画像サイズとキャプション長の分布。画像の高さは 450 前後が最頻で 2000 まで裾を引き、キャプション長は 500 語超まで長い裾を持つ。B は上位 30 の画像型を円周方向の対数目盛の棒グラフで示したもの。</figcaption>
</figure>

**モデル設計を規定した 2 つの事実**（[[entities/biomedclip]] の設計はここから来ている）:

- **画像が大きい**: 汎用領域の標準 224×224 よりずっと大きい
- **キャプションが長い**: 標準 CLIP の既定の最大長 **77 トークンに全然収まらない** → BiomedCLIP は文脈を **256** に拡張（PMC キャプションの 90% を覆う）

## 何が写っているか — 重要な注意

上位 30 の画像型は、**放射線撮影**（磁気共鳴、CT、X 線、血管造影、PET）、**デジタル病理・顕微鏡**（光学顕微鏡、電子顕微鏡、透過顕微鏡、蛍光顕微鏡）、**生理信号**（心電図、脳波、筋電図）、**内視鏡・皮膚科・3D 再構成**などに及ぶ。

> ⚠ **ただし最も頻度が高いのは医療画像ではない。** 図1B の上位は **統計図表・グラフ・チャート、表とフォーム、フローチャート、システム概要図、遺伝子配列、化学構造、数式** といった**汎用の科学図解**であり、放射線・病理はその下に来る。「1,500 万の生物医学画像」という語感と実態にはずれがあり、**有効な医療画像の実数はもっと小さい**。
>
> それでも [[sources/biomedclip]] は **RSNA 肺炎検出で放射線科特化の BioViL を上回る**という結果を出しており、論文は「BiomedCLIP の放射線関連画像は BioViL の MIMIC-CXR より多くはない」と明記したうえで、**非放射線科の画像型を含む多様性そのものが放射線科の性能を上げた（正の転移）**と結論している。**雑多さは欠点ではなく効いている**、というのが本データセットの主張である。

**分布のバイアス**: 論文の図は**発表に値するほど典型的か珍しいものに偏る**。臨床画像が現場の分布からランダムに来るのとは別物である。また[[sources/biomedclip]] のプライバシー代理の実験で明らかになった通り、**論文のキャプションは臨床レポートほど網羅的に所見を書かない**（その論文が言いたい 1 点だけを書く）。

## PMC-Fine-Grained-46M

PMC-15M の**複合図（composite figure）をパネルに分割**し、**論文本文中の引用箇所（citances）**も画像–テキスト対の追加のデータ源として加えた拡張版。

パイプラインは **Data Ingestor → Citance Extractor → Caption Splitter → Citance Splitter → OCR → Label-To-Box Matcher → Figure Splitter → Label-to-Panel Matcher** の 8 段。

> **PMC-15M の画像の半分は複合図**である。分割すれば細粒度のモデリングとグラウンディングが可能になるが、**[[sources/biomedclip]] における現在の用途は画像型の分布の集計だけ**で、事前学習には使われていない。論文自身が今後の課題として挙げている。

## 本 wiki の他の大規模事前学習データとの位置づけ

| データセット | 出どころ | 規模 | 公開性 |
|---|---|---|---|
| [[entities/wit-400m]]（CLIP） | Web の画像 + alt テキスト | 4 億 | **非公開**（OpenAI 内部） |
| [[entities/lvd-142m]]（DINOv2） | キュレートされた画像（テキストなし） | 1.42 億 | **非公開**（Meta 内部） |
| [[entities/sa-1b]]（SAM） | **データエンジン**（モデル + アノテータのループ） | 1,100 万画像 / 11 億マスク | 公開 |
| **PMC-15M** | **科学論文の図–キャプション** | **1,528 万対** | **完全公開・再現可能** |
| MedGemma の内部データ | 病院の内部データ | 病理パッチ 3,260 万 ほか | **非公開** |

**医療では「データを外に出せない」制約が強い。** その中で PMC-15M は「**そもそも最初から公開されている場所を探す**」という解き方を取っており、[[concepts/medical-foundation-model]] の再現性の問題に対する数少ない実効的な回答になっている。[[entities/medgemma]] が重みだけを公開し訓練データの中核が非公開であるのと対照的である。

## 入手

- 再現スクリプト: <https://aka.ms/biomedclip>
- 元データの PMC-OA: <https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/#indart>
- **データセット自体の配布ではなく「PMC-OA から再現するスクリプト」の形で提供される**（元論文のライセンスが個別に異なるため）

## 関連ページ

- [[sources/biomedclip]] / [[translations/biomedclip]] — 原典
- [[entities/biomedclip]] — 本データセットで訓練されたモデル
- [[concepts/medical-foundation-model]] — 医療基盤モデルにおけるデータの再現性の問題
- [[concepts/weakly-supervised-pretraining]] — 画像–テキスト対から学ぶパラダイム
- [[entities/wit-400m]] / [[entities/lvd-142m]] / [[entities/sa-1b]] — 他の大規模事前学習データ
- [[entities/imagenet]] — 古典的なキュレートされたデータセット
