---
type: concept
aliases: [MIM, Masked Image Modeling, マスク画像モデリング]
tags: [paradigm, pretraining, ssl, transformer]
related: [[self-supervised-learning]], [[vision-transformer]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# Masked Image Modeling（MIM, マスク画像モデリング）

## 一言で

**画像の一部のパッチを隠し、隠した部分を残りから予測する**ことで表現を学習する自己教師あり学習の系統。NLP の **BERT** が「文中の単語を隠して当てる masked language modeling（MLM）」で大成功したアイデアを、**画像パッチに転用**したもの。Vision Transformer（[[concepts/vision-transformer]]）の登場で「画像 = パッチの系列」という統一的な見方が可能になり、MIM が急速に発展した。

## なぜ ViT 時代に流行ったのか

- **CNN ではマスキングが扱いにくい**: 畳み込みは局所的な計算で、「あるピクセルを隠す」と周囲の畳み込みカーネルが意味を成さない。
- **ViT は系列として処理する**: トークンを 1 つ取り除いたり、`[MASK]` トークンに置き換えたりすることが NLP の BERT と同じ感覚で自然にできる。
- **画素レベル予測との接続**: MIM は本質的に密予測なので、セグメンテーション・深度推定など pixel-level の下流タスクと相性が良い。

## 代表的な MIM 手法の系譜

### BEiT（Bao et al., 2021）

「Image as a Foreign Language」発想。

- パッチを離散コードブック（dVAE による visual tokenizer）でトークン化
- 一部のパッチを `[MASK]` トークンに置換
- マスクされた位置のコードブック ID を予測（分類タスク）

NLP の BERT と最も忠実に対応する MIM だが、別途 tokenizer の学習が必要。

### MAE（He et al., 2022）

**Masked Autoencoder**。MIM を一気に主流化した名作。

- **画素を 75% マスク**（高マスク率が肝）
- **encoder は可視パッチだけ処理**（軽量・高速）
- 軽量な **decoder が画素値を直接再構成**
- tokenizer 不要、純粋な MSE 損失

> **補足: なぜ 75% という極端なマスク率が効くのか** — MAE 論文の重要な観察は「画像は冗長性が高いので、軽くマスクしても周りの情報からほぼ補間できてしまい、学習信号が弱くなる」というもの。75% という攻めたマスク率にすることで、モデルは「文脈から大きく欠けた部分の意味」を推論せざるを得なくなる。NLP の 15% マスク率（BERT）よりはるかに高い。

MAE は ファインチューニングで強力。ただし**凍結特徴量の k-NN / 線形評価は弱い**（出力が画素再構成に最適化されているため、意味的にきれいな表現空間にはなりにくい）。

### SimMIM（Xie et al., 2022）

MAE をさらに単純化。tokenizer も lightweight decoder も使わず、ViT 全体に `[MASK]` トークンを通して画素を予測。

### iBOT（Zhou et al., 2022）

**[[entities/ibot]] = DINO + MIM**。詳細は別ページ。

- DINO の [CLS] トークン蒸留（画像レベル目的）
- + マスクされたパッチトークンの teacher-student マッチング（パッチレベル目的）
- **画素ではなく teacher の出力分布を予測する**（"online tokenizer" と呼ばれる）

これにより MAE の「凍結特徴量が弱い」問題と DINO の「パッチレベル表現が弱い」問題の両方を解決。

### DINOv2 における MIM

[[entities/dinov2]] は iBOT の MIM 損失をそのまま継承。アブレーション（[[sources/dinov2-learning-robust-visual-features-without-supervision]] §6.4）で「MIM 損失を外すと **ADE20k mIoU が -3% 落ちる**」と示し、**密予測性能の鍵**であることを示した。

## MIM vs 他の SSL 系統

| | MIM (BEiT, MAE, iBOT) | 識別型 (SimCLR, MoCo, BYOL, DINO) |
|---|---|---|
| 信号源 | 画像内のマスク予測 | 画像間/拡張間の関係 |
| 強み | 密予測タスク、fine-tune 性能 | 凍結特徴量、k-NN、検索 |
| 弱み | 凍結特徴量が弱い（MAE）、k-NN が弱い | パッチレベル表現がやや弱い |
| 必要なデータ拡張 | ほぼ不要 | crop, color jitter 等が肝 |
| 崩壊回避 | 不要（再構成ターゲットがある） | 必須（[[concepts/self-supervised-learning]] 参照） |

iBOT/DINOv2 は**ハイブリッド**で、両方の長所を取りに行く設計。これが「凍結特徴量が線形分類でも密予測でも強い」DINOv2 の性能の源泉。

## 関連する歴史的線

MIM の発想は実は古く、**Stacked Denoising Autoencoder**（Vincent et al., 2008）が「入力にノイズを加えて元を復元させる」自己教師あり学習を提唱していた。MIM はこれを Transformer 時代に再発明したものとも言える。

NLP 側の系譜:
- **word2vec**（Mikolov, 2013）の skip-gram = 単語の文脈予測
- **BERT**（Devlin et al., 2018）の MLM = マスクされた単語の予測

CV 側の系譜:
- **Context Prediction**（Doersch et al., 2015）= パッチ間の相対位置予測
- **Inpainting**（Pathak et al., 2016）= 矩形領域の補完
- **Jigsaw Puzzle**（Noroozi et al., 2016）= シャッフルされたパッチの並べ替え
- → ViT 時代に **BEiT → MAE → SimMIM → iBOT → MaskFeat → ...** と一気に発展

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: MIM を識別型 SSL と統合した代表例
- [[entities/ibot]]: DINO + MIM のハイブリッド
- [[concepts/self-supervised-learning]]: SSL 全般の中での位置づけ
- [[concepts/vision-transformer]]: MIM の主要ターゲットアーキテクチャ
