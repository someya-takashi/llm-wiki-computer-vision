---
type: concept
aliases: [WSL, Weakly-Supervised Pretraining, Text-Guided Pretraining, 弱教師あり事前学習]
tags: [paradigm, pretraining, vision-language]
related: [[self-supervised-learning]], [[foundation-model]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# Weakly-Supervised Pretraining（弱教師あり事前学習）

## 一言で

**人手の明示的なラベル付け（"これは犬"）ではなく、Web 上に自然に存在する「ノイジーで間接的な教師信号」を使って大規模事前学習を行う方法**。CV では主に **画像 - テキスト（キャプション、alt 属性、ハッシュタグ）対** を使うアプローチを指し、CLIP がその代表。「弱教師あり（weakly-supervised）」は「教師信号はあるが完全ではない・ノイズが多い・間接的」という意味。

> **補足: 「教師あり」「弱教師あり」「自己教師あり」の区別**
> - **教師あり（supervised）**: 人手で正確にラベル付けされたデータ（ImageNet など）。スケールしにくい。
> - **自己教師あり（self-supervised, SSL）**: 教師信号を**データ自身**から作る（DINO, MAE 等）。詳細: [[concepts/self-supervised-learning]]
> - **弱教師あり（weakly-supervised, WSL）**: 教師信号は**外部にあるが不完全**（Web のキャプション、ハッシュタグ、画像のクリック数など）。WSL は SSL と教師ありの中間。

## CV における WSL の主要パターン

### A. 画像 - テキスト対比学習（CLIP 系）

- **CLIP**（OpenAI, 2021）: Web から集めた 4 億組の画像-キャプション対。**「対応する画像とテキストの埋め込みが近く、非対応は遠くなるよう対比学習**」する。詳細: [[entities/clip]]
- **ALIGN**（Google, 2021）: 18 億組のノイジーな画像-alt text 対。CLIP より大規模
- **OpenCLIP**: LAION-2B / LAION-5B を使った CLIP の公開再現＋拡張版
- **EVA-CLIP**, **SigLIP**, **DFN** など多数の後継

### B. ハッシュタグ・タグ予測

- **WSL（Mahajan et al., 2018）**: Instagram の 35 億画像とハッシュタグ（17k クラス）で訓練。SEER 系の祖先
- **Facebook SWAG**（2022）: 3.6B 枚と関連ハッシュタグで訓練、ViT で実行

### C. メタデータ駆動の検索／フィルタリング

- 検索ログ、クリックデータ、共起情報など

## CLIP が変えたもの

CLIP の登場（2021）は CV パラダイムシフト的な出来事だった：

1. **ゼロショット分類が可能に**: 「a photo of a {label}」というテキストプロンプトで、ファインチューン不要で任意のクラスを分類
2. **テキストと共通の埋め込み空間**: 画像検索（テキスト → 画像）、画像クラスタリング（意味的に）、画像生成のガイダンス（Stable Diffusion 等）が容易に
3. **「ラベル」を人手で定義する必要が消えた**: タスクをテキストで記述できれば、専用の訓練不要

## WSL の長所

- **データが取りやすい**: Web には alt text 付き画像が無尽蔵にある
- **言語と接続済み**: VLM（[[concepts/foundation-model]] 参照）への展開が自然
- **ゼロショット汎用性**: ImageNet 以外のラベル系にも対応可能
- **テキストでタスクを指定できる**: prompting で柔軟な利用が可能

## WSL の弱点（DINOv2 論文が指摘するもの）

[[sources/dinov2-learning-robust-visual-features-without-supervision]] §1, §7 で繰り返し指摘されている点：

1. **キャプションは画像の豊かさを近似するだけ**: 「a photo of a dog」というキャプションは、犬の品種・姿勢・背景・テクスチャ・3D 構造などの大半を捨てる。**ピクセルレベル情報が失われる**ので、深度推定やセグメンテーションのような密予測タスクが弱い。
2. **テキスト - 画像対のアラインメント済みデータが必要**: 純粋画像のみのデータで学習できない（NLP の LM が生テキストだけで学べるのと対照的）。
3. **キャプションのバイアスを引き継ぐ**: Web 上の alt text は「人間がどう描写したか」のバイアスを持つ。細粒度カテゴリ（鳥の種類など）や非典型的物体に弱い傾向。
4. **計算コストが大きい**: テキストエンコーダも並行訓練するので、SSL より重い（DINOv2 §9 では OpenCLIP-G の訓練は DINOv2-g の 10 倍の CO₂ 排出と試算）。

DINOv2 論文の主張は要するに「**ピクセルの細部や密予測が大事なら、SSL の方が WSL より良い**」というもの。実際 DINOv2-g は深度推定と segmentation で OpenCLIP-G を大きく上回る。

## SSL vs WSL（実用上の使い分け）

| 用途 | 推奨 |
|---|---|
| ゼロショット分類 | WSL（CLIP 系） |
| 画像 - テキスト検索 / 生成 | WSL（CLIP 系） |
| 密予測（segmentation, depth） | SSL（DINOv2 系）|
| 細粒度分類（鳥、車、花） | SSL（DINOv2 系）|
| 凍結特徴量で k-NN / 線形 | SSL（DINOv2 系）|
| 大規模 VLM の vision tower | 現状は WSL 優勢、SSL も追いつき中 |
| 産業ドメイン（医療・衛星） | SSL（DINOv2 系がよく転移する） |

実際には「**CLIP（or SigLIP）+ DINOv2 を併用**」が広がりつつある。例: LLaVA-NeXT は CLIP + DINOv2 の特徴を結合。

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: SSL が WSL に対し dense prediction で勝つことを示した論文
- [[entities/clip]]: WSL の代表例
- [[concepts/self-supervised-learning]]: 対比される対の概念
- [[concepts/foundation-model]]: 両者を包括する上位概念
