---
type: concept
aliases: [Foundation Model, 基盤モデル, foundation models]
tags: [paradigm, scaling, pretraining]
related: [[self-supervised-learning]], [[weakly-supervised-pretraining]], [[vision-transformer]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# Foundation Model（基盤モデル）

## 一言で

**「大規模かつ多様なデータで自己教師あり的に事前学習され、ファインチューニングや少数例適応で多様な下流タスクに転用できる単一の大規模モデル」**。Stanford CRFM（Center for Research on Foundation Models）が 2021 年の論文 "On the Opportunities and Risks of Foundation Models"（Bommasani ら）で名付けた概念。BERT、GPT、CLIP、DINOv2、SAM などが典型例。

> **補足: なぜ「foundation（基礎）」と呼ばれるか** — 「あらゆる応用がこのモデルの上に建てられる」という比喩。建築の基礎工事のように、応用ごとに最初から作り直すのではなく、一度作った基礎モデルに乗せていくという発想。日本語では「基盤モデル」と訳されることが多い。

## 構成要件（CRFM の定義から）

1. **大規模な事前学習**: 通常、数億〜数千億パラメータ、数十億〜数兆トークン／画像
2. **広範なデータ**: ドメイン・スタイル・分布が多様
3. **自己教師ありまたは弱教師ありの目的関数**: ラベル付きデータでは到底スケールできない
4. **下流タスクへの汎用性**: ゼロショット、線形プローブ、ファインチューン、プロンプトなどで多様な応用に転用
5. **emergent capabilities**: スケールに伴って事前に予測しなかった能力が現れる（ICL, chain-of-thought, dense correspondence など）

## NLP における基盤モデルの系譜

| 年 | モデル | 規模 | 特徴 |
|---|---|---|---|
| 2018 | BERT | 110M〜340M | masked language modeling、ファインチューン前提 |
| 2018 | GPT-1 | 117M | causal LM、生成 |
| 2019 | GPT-2 | 1.5B | スケール則の最初の示唆 |
| 2020 | GPT-3 | 175B | in-context learning が emergence |
| 2022 | PaLM, Chinchilla | 540B / 70B | scaling law の精緻化 |
| 2023-24 | GPT-4, Claude, Gemini | 非公開 | マルチモーダル、ツール使用 |

## CV における基盤モデルの系譜

CV では NLP より遅れたが、2021〜2024 にかけて本格化。

### A. 弱教師あり系（テキスト誘導）
- **CLIP**（OpenAI, 2021）: WIT-400M の画像-テキスト対で対比学習。最初の本格的視覚基盤モデル。詳細: [[entities/clip]]
- **ALIGN**（Google, 2021）: 18 億の Web 画像-alt text 対で学習
- **OpenCLIP**: LAION-2B/5B で CLIP を再現・拡張、公開
- **Florence**（Microsoft）, **BASIC**（Google）, **CoCa**（Google）など

### B. 自己教師あり系
- **MAE**（Meta, 2022）: マスク再構成で ViT-H まで scale
- **DINO**（Meta, 2021）: 自己蒸留。[[entities/dino]]
- **iBOT**（2022）: DINO + MIM。[[entities/ibot]]
- **DINOv2**（Meta, 2023）: 1B パラメータ、142M キュレーション画像で基盤モデル化。[[entities/dinov2]]

### C. 生成系
- **Stable Diffusion**（2022）: テキストから画像生成の基盤
- **Imagen, DALL-E 3, SDXL** など

### D. 領域汎用モデル
- **SAM**（Segment Anything Model, Meta, 2023）: セグメンテーションのプロンプタブル基盤モデル

## 基盤モデルがもたらす変化

### 開発パラダイムの変化

```
[従来]                          [基盤モデル時代]
タスクA用に                      汎用基盤モデルを 1 つ作る
モデルAを訓練                          ↓
タスクB用に              ──→  タスクA, B, C ... を
モデルBを訓練                    線形プローブ／プロンプト／
タスクC用に                      LoRA／ファインチューンで適応
モデルCを訓練
```

### 集中化と複製困難性

- 訓練コストが**数千 GPU × 数週間**規模なので、大学・中小企業が事前学習を行えない
- 「事前学習は数社が行い、世界中が下流応用する」という構造に
- DINOv2 論文 §9 では、ViT-g 訓練に 22k GPU 時間、プロジェクト全体で 200k GPU-days を要したと開示

### 評価方法のシフト

- 線形プローブ・k-NN 評価（[[concepts/knn-evaluation-protocol]]）が標準化
- ゼロショット評価（CLIP 系）
- 大規模ベンチマーク群（COCO, ADE20K, NYUd, ... を横断的に評価）

## 批判と未解決問題

1. **公平性とバイアス**: 訓練データの偏りが下流すべてに伝播する。DINOv2 自身が §8 で地理的バイアスを開示。
2. **環境コスト**: 訓練時の CO₂ 排出が無視できない（DINOv2 ViT-g 単体で 3.7t CO₂eq）
3. **再現性**: モデルや重みは公開されても、訓練データやレシピは非公開のことが多い（LVD-142M 自体は非公開）
4. **homogenization のリスク**: 全員が同じ基盤モデルに依存することで、その欠陥や偏向が全社会に波及する
5. **emergence の予測困難性**: スケールに伴い何が emerge するか／しないかが事前に分からない

## CV の foundation model に求められる特性（DINOv2 論文より）

DINOv2 著者らが暗黙に置いている要件：

- **ファインチューンなしで使える**（線形プローブで十分強い）
- **画像レベルとピクセルレベルの両方で機能**（分類と segmentation/depth の両方）
- **ドメイン汎化**（自然画像 → 絵画 → 衛星画像 などへの転移）
- **インスタンスレベルとカテゴリレベルの両方**（検索と分類の両方）
- **テキスト不要**（純粋画像で十分）

DINOv2 はこれらを多くの軸で達成し、「CV の基盤モデルは SSL でも作れる」ことを示した。

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: CV における純粋 SSL 基盤モデルの代表例
- [[entities/dinov2]], [[entities/clip]]
- [[concepts/self-supervised-learning]], [[concepts/weakly-supervised-pretraining]]
