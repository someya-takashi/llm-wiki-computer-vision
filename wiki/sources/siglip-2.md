---
type: source
source_path: raw/papers/SigLIP 2_ Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.md
source_kind: paper
title: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features"
authors: [Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai]
year: 2025
venue: arXiv 2502.14786
ingested: 2026-05-27
tags: [vision-language, contrastive, sigmoid-loss, multilingual, self-distillation, masked-prediction, locca, dense-features, naflex, distillation, foundation-model, paligemma]
translation: "[[translations/siglip-2]]"
---

# SigLIP 2: 多言語・dense feature・位置特定を改善した SigLIP の後継

> 原典: [[translations/siglip-2]] ・ `raw/papers/SigLIP 2_ Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.md`
> 著者: Michael Tschannen, Xiaohua Zhai ら（Google DeepMind）
> 出典: arXiv 2502.14786（2025）
> arXiv: <https://arxiv.org/abs/2502.14786>
> モデル: HuggingFace `google/siglip2-*`、Apache 2.0
> コード: `big_vision`（<https://github.com/google-research/big_vision>）

## 一言まとめ

**[[sources/siglip]]（[[entities/siglip]]）の sigmoid 損失に、(1) LocCa の decoder ベース事前学習、(2) SILC/TIPS の自己蒸留＋マスク予測、(3) ACID の能動的データキュレーション、(4) NaFlex の native アスペクト比対応、(5) 多言語化＋ de-bias 化を統合した「全部入りレシピ」**。同サイズの SigLIP を全ベンチマークで上回り、特に **dense 予測** と **位置特定（RefCOCO）** で劇的な改善。**後方互換**（同じアーキテクチャ、重みだけ差し替え可能）。

## 背景と問題意識

[[entities/clip]]（CLIP, 2021）以降、CV 視覚-言語モデルは多数の改善提案が並行して進んできた：

| 改善の方向 | 代表研究 | 何が嬉しいか |
|---|---|---|
| 損失関数の改善 | [[entities/siglip]]（sigmoid loss） | 効率・小バッチ性能・ノイズ頑健性 |
| データ品質フィルタ | DFN, MetaCLIP, [^61] ACID | キュレーション質 |
| 再キャプション化 | VeCLIP, [^46] | キャプション質 |
| 自己教師あり損失追加 | SLIP, SILC, TIPS | dense feature の改善 |
| Decoder 追加（キャプション） | CoCa, BLIP-2, LocCa | 位置特定・OCR |
| 多言語化 | mSigLIP | 非英語圏対応 |
| 可変解像度 | FlexiViT, NaViT | アスペクト比歪み軽減 |

問題は **それぞれが独立に研究されていて、1 つの公開モデルに全部入っていない** こと。CLIP 以降のオープン重み公開モデル群（CLIP, OpenCLIP, EVA-CLIP, MetaCLIP, DFN, SigLIP）はどれも CLIP のアプローチに近いままだった。

> **補足: なぜ「全部入れ」がチャレンジングか** — 各損失は計算コストとメモリを消費し、ハイパーパラメータも増える。素朴に全部足すと 1 エポックも回せなくなる。SigLIP 2 の貢献は「**段階的アプローチで現実的に統合**」する具体的レシピを示したこと。

SigLIP 2 はこの状況に対し、「**過去 4 年の主要な独立改善をすべて統合した単一モデル**」を提供。さらに **後方互換性**（CLIP/SigLIP と同じアーキテクチャ、重みのみ差し替え可能）を保つ。

## 提案手法

### 段階的訓練レシピ（全体俯瞰）

```
[第 1 段階: §2.2] SigLIP sigmoid loss + LocCa decoder loss を等重みで同時最適化
   - WebLI 10B 画像 / 12B alt-text（109 言語）
   - 90% 英語 + 10% 非英語
   - de-bias 化フィルタ（[2])
   - バッチ 32k、合計 40B サンプル、最大 2048 TPUv5e
                          ↓
[第 2 段階: §2.3] 訓練 80% 完了時点で SILC/TIPS の self-distillation + masked prediction を追加
   - student（pool 前）と teacher（EMA）の local-to-global 一貫性
   - 50% パッチをマスクし、マスク位置で teacher にマッチさせる
                          ↓
[第 3 段階: §2.4.1] 95% 完了時点で位置埋め込みをリサイズし、目標解像度（256/384/512）で再開
                          ↓
[追加段階: §2.5] B/16, B/32 のみ ACID で 4B サンプル蒸留ファインチューン
[並行: §2.4.2] NaFlex バリアントは 90% 完了時点でアスペクト比保持＋可変系列長に切替
```

### 4 つの主要な追加技法

#### 1. LocCa（decoder ベース事前学習）

[[concepts/contrastive-learning]] の InfoNCE/sigmoid 損失だけでなく、**標準 transformer decoder を pool 前の視覚特徴量に cross-attention で接続** し、以下の 3 タスクで同時訓練：

- **画像キャプション化**（50% は parallel prediction、すなわち causal mask なしで全トークン並列予測）
- **automatic referring expression prediction**（バウンディングボックス座標を出力）
- **grounded captioning**（バウンディングボックスを入力に領域特化キャプションを生成）

領域-キャプション対は **alt-text から n-gram 抽出 + open-vocab 検出** で自動注釈。

> **補足: なぜ decoder を追加するのか** — sigmoid/InfoNCE は「画像とテキスト全体の対応」しか学べない。decoder で「画像のどこに何があるか」を逐次予測させると、**空間的に局所化された** 特徴量が学習される。これが OCR や RefCOCO（参照表現理解）の大幅改善（SigLIP の +20pt 以上）に直結。

**decoder はモデルリリースには含まれない**。表現学習のためだけに使われる「足場」。

#### 2. Self-distillation + masked prediction（SILC + TIPS 流）

[[entities/dino]] や [[entities/ibot]] と同じ系統の自己蒸留損失を追加：

- **Local-to-global consistency loss**（SILC [^45]）: student は局所ビュー（8 個）、teacher は global ビュー（1 個）を見て、MLP ヘッド経由で特徴量がマッチするよう訓練
- **Masked prediction loss**（TIPS [^38]）: 50% パッチをマスクし、teacher のパッチ特徴量にマッチするよう student を訓練

> **補足: SILC/TIPS とは何か** — SILC = Self-supervised + Image-Language Contrastive learning（[^45], ECCV 2024）, TIPS = Text-Image Pretraining with Spatial awareness（[^38], ICLR 2025）。両方とも「CLIP に DINO スタイル自己蒸留を加える」研究で、SigLIP 2 はその上位互換とも言える。

訓練の **最後の 20% でのみ追加** することで計算コストを抑制。teacher は student 重みで初期化、追加ヘッドはランダム初期化。

#### 3. ACID（能動的データキュレーションによる暗黙的蒸留）

B/16, B/32 の小型モデルのみ、追加の 40 億サンプル分のファインチューン段階で **ACID 法 [^61]** を適用：

1. 各ステップで teacher と learner がサンプルを「learnability」スコアで評価
2. **super-batch（64k）から最適な 32k バッチを共同選択**
3. SigLIP 2 だけの工夫: 単一 teacher（So400m）を高品質キュレーションデータで 1B サンプルファインチューンして使うことで、ACED の 2-teacher 設定を 1-teacher 化（計算半減）

> **補足: ACID/ACED とは** — ACID = Active Curation Implicit Distillation（[^61]）。「データを通じた暗黙的蒸留」と呼ばれる。teacher が learner の苦手なサンプルを選び続けることで、明示的に soft label を渡さずに知識転移する。ACED は ACID + 明示的 softmax 蒸留の組み合わせ。

#### 4. NaFlex（可変アスペクト比・可変解像度）

NaViT [^12]（native アスペクト比処理）と FlexiViT [^6]（複数系列長対応）を統合：

- パッチサイズと目標系列長が与えられたとき、アスペクト比歪みを最小化しつつパッチ整数倍にリサイズ
- 位置埋め込みは bilinear リサイズで非正方格子に適応
- ミニバッチごとに $\{128, 256, 576, 784, 1024\}$ から系列長を一様サンプリング
- 訓練 90% 完了時点で標準訓練から切り替え（学習率スケジュールを 3.75× 引き伸ばし）

> **補足: なぜ NaFlex が嬉しいか** — 文書画像（縦長）や Web スクリーンショット（横長）を正方形に強制リサイズすると OCR 性能が劣化する。NaFlex なら **単一チェックポイントで任意のアスペクト比・任意解像度** に対応可能。アスペクト感度が高い OCR/文書理解ベンチマーク（TextCaps, HierText, SciCap, Screen2Words）で標準版を上回る。

NaFlex には計算量を抑えるため自己蒸留・マスク予測は適用しない。

### アーキテクチャの注意点

- **後方互換**: SigLIP と同じ ViT、学習済み位置埋め込み、MAP ヘッド（attention pooling）。**重みとトークナイザを差し替えるだけで使える**
- 画像 tower とテキスト tower は同形（g サイズの ViT のみ So400m サイズのテキストエンコーダとペア）
- テキスト長 64、**多言語 Gemma トークナイザ（語彙 256k）**、トークン化前に小文字化
- すべてのモデルでパッチサイズ 16、初期解像度 256

## 主要結果

### 1. ゼロショット分類・検索（表 1）

**全モデルサイズ・全解像度で SigLIP を上回る**。主な比較（IN-1k 0-shot val）：

| Model | サイズ | 解像度 | SigLIP | **SigLIP 2** | 改善 |
|---|---|---|---|---|---|
| B/16 | 86M | 256 | 76.7 | **79.1** | **+2.4** |
| L/16 | 303M | 256 | 80.5 | **82.5** | **+2.0** |
| So400m/14 | 400M | 224 | 82.2 | **83.2** | **+1.0** |
| So400m/14 | 400M | 384 | 83.2 | **84.1** | **+0.9** |
| g/16 | 1B | 384 | (なし) | **85.0** | NEW |

- B/16 で **+2.4pt** は ACID 蒸留の恩恵が大きい
- **g/16 は新サイズ**（SigLIP v1 にはなかった、SigLIP 2 で 1B モデル投入）
- 多言語 **XM3600**: SigLIP 2 L/16@256 で 46.5（SigLIP 30.9 から +15.6pt）、mSigLIP 50.0 に肉薄しつつ英語性能は維持

### 2. Dense 予測（表 2）— 最大の改善領域

凍結バックボーンを線形プローブ or DPT decoder でセグメンテーション/深度/法線推定：

| タスク | 指標 | SigLIP So/14@224 | **SigLIP 2 So/14@224** | 改善 |
|---|---|---|---|---|
| PASCAL Seg | mIoU↑ | 72.0 | **77.1** | **+5.1** |
| ADE20k Seg | mIoU↑ | 37.6 | **41.8** | **+4.2** |
| NYUv2 Depth | RMSE↓ | 0.576 | **0.493** | **−14%** |
| NAVI Depth | RMSE↓ | 0.083 | **0.067** | **−19%** |

**SILC/TIPS の自己蒸留＋マスク予測が効果を発揮**。CLIP/OpenCLIP/SigLIP より顕著に上、ただし [[entities/dinov3]] には及ばない（次節参照）。

### 3. 位置特定（表 5）— 劇的な改善

参照表現理解（RefCOCO）でのプロービング:

| サイズ | seq | SigLIP val | **SigLIP 2 val** | 改善 |
|---|---|---|---|---|
| B | 256 | 64.05 | **83.76** | **+19.7** |
| L | 256 | 67.33 | **86.04** | **+18.7** |
| So | 256 | 64.68 | **86.42** | **+21.7** |

**+20pt 近い改善**。これは LocCa の decoder ベース事前学習の直接効果。SigLIP 2 は LocCa 本家（88.34）にだけ負ける（LocCa は英語専用、SigLIP 2 は多言語化のコストを払っているため）。

### 4. オープン語彙検出（表 4）

OWL-ViT 経由で COCO/LVIS 検出：

| ViT | Model | COCO AP | LVIS AP | **LVIS APr** |
|---|---|---|---|---|
| So/14 | SigLIP | 44.3 | 39.5 | 40.9 |
| So/14 | **SigLIP 2** | **45.2** | **40.5** | **42.3** |

**LVIS rare カテゴリで +1.4pt** が特筆。多言語＋多様データの恩恵。

### 5. VLM 視覚エンコーダとして（図 4・表 6）

PaliGemma 2 レシピで Gemma 2 2B と組み合わせて 35 個の VLM ベンチマーク：

- **全データセットでほぼ SigLIP を上回る**
- L サイズで **AIMv2**（Apple, 2024 マルチモーダル自己回帰モデル）も上回る
- 特に **OCR/文書系**（DocVQA, TextVQA, InfoVQA, ST-VQA, TextCaps）で大幅改善（LocCa decoder の恩恵）
- TextVQA: SigLIP So/14@384 54.5 → **SigLIP 2 So/14@384 59.4**（+4.9）

### 6. 公平性の劇的改善（表 9）

**representation bias（性別とランダムオブジェクトの偏った関連付け）**：

| サイズ | 解像度 | SigLIP | **SigLIP 2** | 削減 |
|---|---|---|---|---|
| L/16 | 256 | 35.5% | **7.3%** | **−28.2pt** |
| L/16 | 384 | 34.8% | **6.6%** | **−28.2pt** |
| So400m/14 | 224 | 33.3% | **7.4%** | **−25.9pt** |
| g-opt/16 | 384 | (なし) | **4.9%** | **最小** |

**de-bias フィルタ [^2] の効果が劇的**。SigLIP では「ランダムオブジェクトを 85.5% の確率で男性に関連付け」ていたものが、SigLIP 2 では 7.3% に。

文化的多様性タスク（Dollar Street, GeoDE, GLDv2）でも一貫して改善。

### 7. NaFlex の効果（図 3・表 7）

- **検索ベンチマークで NaFlex が標準版を上回る**（特に小系列長で）
- **OCR/文書系（HierText, SciCap, Screen2Words）で大幅優位**: 例えば HierText@1024 で標準版 25.2 vs NaFlex 25.1（同等）だが、@256 では 17.1 vs 19.7（NaFlex +2.6）
- ただし自然画像中心のベンチマークでは標準版が僅かに有利（特に B サイズ、ACID 蒸留の効果が NaFlex では適用されないため）

## DINOv3 系との位置関係

[[entities/dinov3]]（Meta, 2025）の主張: 「WSL は dense prediction で SSL に勝てない」。SigLIP 2 はこの主張への **WSL 側からの強力な反論** と見ることもできるが、決着は付いていない：

| タスク | DINOv3 7B/16 | SigLIP 2 g/16 | 勝者 |
|---|---|---|---|
| ImageNet 0-shot | 82.3（dino.txt） | **85.0** | SigLIP 2 |
| ADE20k linear | **55.9** | 41.8（So/14） | DINOv3 |
| NAVI 3D | **64.4** | 49.4 | DINOv3 |
| DAVIS tracking | **83.3** | 62.9 | DINOv3 |
| RefCOCO val | (基本なし) | **86.04**（L/16）| SigLIP 2 |

**SigLIP 2 は dense feature を SigLIP 比で大幅改善したが、純粋 SSL（DINOv3）にはまだ及ばない**。一方、**画像-テキスト系タスク（zero-shot 分類、検索、RefCOCO）では SigLIP 2 が勝つ**。両者は補完的で、実用では併用も普通。

## CLIP 系発展レシピの全体俯瞰

SigLIP 2 は CLIP 以降の主要な独立研究を統合した「CLIP 4 年分の進化を 1 モデルに凝集」：

| 改善 | 原典 | SigLIP 2 での実装位置 |
|---|---|---|
| sigmoid loss | [[sources/siglip]] | §2.2 第 1 損失 |
| decoder ベース事前学習 | CapPa, CoCa, LocCa | §2.2 第 2 損失 |
| 局所-大域自己蒸留 | DINO, SILC | §2.3 第 1 項 |
| マスク予測 | iBOT, MAE, TIPS | §2.3 第 2 項 |
| データキュレーション | DFN, ACID | §2.5 |
| 多言語化 | mSigLIP, PaLI | §2.1 |
| Native アスペクト比 | NaViT, FlexiViT | §2.4.2 NaFlex |
| de-bias 化 | [^2] | §2.1 |

> **補足: SigLIP 2 は「単発の新規アイデア」ではなく「系統的統合のレシピ」** — 各要素は他研究で既出だが、それらを段階的に組み合わせて単一のオープン重みモデルとして配布した点が価値。これは「研究の再現性と利用性」という観点で大きい。

## 限界

1. **WebLI 自体は非公開**: モデル重みは Apache 2.0 で公開だが、訓練データは Google 内部のみ。完全再現は困難
2. **Dense prediction で DINOv3 に劣る**: 大幅改善したが、純粋 SSL の壁を超えるには至らず
3. **NaFlex は外挿しない**: 訓練系列長を超えると性能劣化
4. **B サイズの NaFlex は標準版に負ける**: ACID 蒸留が NaFlex に適用されないため
5. **計算コストは依然大きい**: 最大 2048 TPUv5e × 多日数、研究機関規模が必要
6. **多言語化のトレードオフ**: 多言語データを 10% 入れる対価として、英語性能が mSigLIP と僅差で並ぶ（純粋英語版より良いが特化版には及ばず）
7. **LocCa decoder は公開せず**: VLM 構築で再利用したくても decoder 重みはない

## 用語と略称

- **SigLIP 2** = SigLIP version 2、多言語＋ dense feature 改善版
- **WebLI** = Web Language and Image、Google 内部の 10B 画像 / 12B alt-text データセット
- **LocCa** = Location-aware Captioner、decoder ベース事前学習レシピ
- **SILC** = Self-supervised + Image-Language Contrastive learning（CLIP + DINO スタイル自己蒸留）
- **TIPS** = Text-Image Pretraining with Spatial awareness（SILC + マスク予測）
- **ACID** = Active Curation Implicit Distillation（データキュレーション経由の暗黙的蒸留）
- **ACED** = ACID + Explicit Distillation
- **NaFlex** = Native aspect ratio + Flexible sequence length バリアント
- **NaViT** = Native-resolution ViT（[^12]）
- **FlexiViT** = Flexible patch-size ViT（[^6]）
- **MAP head** = Multi-head Attention Pooling、学習可能 query で可変長を pool するヘッド
- **Gemma トークナイザ** = Google Gemma の多言語 SentencePiece トークナイザ（256k 語彙）
- **PaliGemma / PaliGemma 2** = SigLIP + Gemma の組み合わせ VLM
- **AIMv2** = Apple の自己回帰画像モデル v2、SigLIP 2 L の主要比較相手
- **DPT** = Dense Prediction Transformer、dense 予測ヘッドの標準
- **OWL-ViT** = Open-vocabulary detection 手法
- **Cat-Seg** = オープン語彙セグメンテーション手法
- **GeoDE, Dollar Street, GLDv2** = 地理的多様性／所得多様性／ランドマーク評価データセット
- **XM3600** = Crossmodal-3600、36 言語の多言語検索ベンチマーク
- **RefCOCO / RefCOCO+ / RefCOCOg** = 参照表現理解ベンチマーク
- **representation bias** = ランダムオブジェクトを特定の性別グループに関連付けるモデルの偏り
- **EMA** = Exponential Moving Average、teacher パラメータ更新方式
- **FSDP** = Fully-Sharded Data Parallel
- **TPUv5e** = Google の第 5 世代 Tensor Processing Unit、最大 2048 チップで訓練

## 関連ページ

- [[translations/siglip-2]] — 本論文の全文和訳（§1-5 + Appendix A-C）
- [[sources/siglip]] / [[entities/siglip]] — 前身（v1）、sigmoid 損失の元
- [[entities/clip]] / [[sources/clip]] — 元祖
- [[entities/perception-encoder]] — Meta の対抗 WSL モデル
- [[entities/dinov3]] / [[sources/dinov3]] — 純粋 SSL 側、dense prediction で SigLIP 2 を上回る
- [[entities/dino]] / [[entities/ibot]] — 自己蒸留・マスク予測の源流
- [[concepts/contrastive-learning]] — sigmoid 損失の位置づけ
- [[concepts/weakly-supervised-pretraining]] — SigLIP 2 が属するパラダイム
- [[concepts/self-supervised-learning]] — 自己蒸留・マスク予測の概念
- [[concepts/foundation-model]] — WSL foundation model としての位置づけ
- [[concepts/masked-image-modeling]] — マスク予測の系統
- [[concepts/knowledge-distillation]] — ACID/ACED の文脈
- [[concepts/zero-shot-transfer]] — 主要評価軸
