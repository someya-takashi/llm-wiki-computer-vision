---
type: source
source_path: raw/papers/Revisiting Semi-Supervised Learning in the Era of Foundation Models.pdf
source_kind: paper
title: "Revisiting Semi-Supervised Learning in the Era of Foundation Models"
authors: [Ping Zhang, Zheda Mai, Quang-Huy Nguyen, Wei-Lun Chao]
year: 2025
venue: NeurIPS 2025
ingested: 2026-05-28
tags: [semi-supervised-learning, foundation-model, vision-foundation-model, parameter-efficient-fine-tuning, peft, lora, adaptformer, self-training, pseudo-label-ensemble, clip, dinov2, vtab, ohio-state]
translation: [[translations/revisiting-ssl-foundation-models]]
---

# Revisiting Semi-Supervised Learning in the Era of Foundation Models

> 原典: [[translations/revisiting-ssl-foundation-models]] ・ `raw/papers/Revisiting Semi-Supervised Learning in the Era of Foundation Models.pdf`
> 著者: Ping Zhang, Zheda Mai, Quang-Huy Nguyen, Wei-Lun Chao（Ohio State University）
> 年・会議: 2025 / NeurIPS 2025
> コード: https://github.com/OSU-MLB/SSL-Foundation-Models

## 一言まとめ

**「Vision Foundation Models（VFM, CLIP/DINOv2 等）の時代に、既存 SSL アルゴリズムは本当に必要か？」を体系的に問い直した論文**。衝撃的な結論は「**ラベルあきデータのみで VFM を PEFT（LoRA/AdaptFormer 等）すれば、ラベルなしを使う既存 SSL（FixMatch/FlexMatch/SoftMatch）を凌駕することが多い**」。これを踏まえ、複数の VFM × PEFT のアンサンブル疑似ラベリングという単純な手法 **V-PET** を提案し、既存 SSL を圧倒した。

## 背景と問題意識

[[entities/fixmatch]]、[[entities/flexmatch]]、SoftMatch、MixMatch といった現代 SeSL アルゴリズムは、**「スクラッチからニューラルネットを訓練する」前提**で設計された。一方、2020 年代に入って CV の主役は事前学習済みの **Vision Foundation Model（VFM）** ——大規模データで事前学習された [[entities/clip]]、[[entities/dinov2]] 等——に移行している。

> **本論文の問題提起**: 「スクラッチ前提の SSL アルゴリズム」を「VFM をバックボーンとして使う SSL」に当てはめても本当に機能するのか？

著者らは「凍結 VFM が苦戦するベンチマーク」を新規構築し（後述）、FixMatch/FlexMatch/SoftMatch の VFM 上での性能を**公正なハイパラチューニング**（教師ラベルを使わない 7 つの教師なし基準）で評価した。

## 衝撃的発見（3 つ）

### 発見 1: ラベルあきのみ Full Fine-Tuning が SSL に匹敵

「VFM をラベルあきデータのみで普通に fine-tune するだけ」で、ラベルなしデータを大量に使う SSL とほぼ同等の性能が出る（図 3）。

> **解釈**: VFM は既に膨大な事前学習で強力な汎化能力を獲得しているので、SeSL アルゴリズムが導入する「ノイズの多い教師信号」が VFM の汎化能力を**むしろ損なう**可能性がある。「SSL がラベルなしデータを活かす」という前提自体が VFM 時代では再検討を要する。

### 発見 2: PEFT は救いになるが、SSL を救えない

LoRA や AdaptFormer といった **Parameter-Efficient Fine-Tuning（PEFT）** ——VFM の大部分を凍結したまま小数のパラメータだけを更新——を使えば、SSL の性能も向上する。**しかし**、同じ PEFT を「ラベルあきのみ」で適用しても同様に向上するため、**PEFT は SSL 専用の利得ではない**。結果として PEFT × SSL ≈ PEFT × ラベルあきのみで、ラベルなしの貢献は依然として限定的。

### 発見 3: PEFT 済みモデルの疑似ラベルは高品質かつ多様

ここからが面白い。著者らは「ラベルあきのみで PEFT した VFM」が**そこそこ良い精度**を達成することに注目し、その**予測を疑似ラベルとして使う**ことを考えた。さらに重要なのは：

- 異なる **PEFT 手法**（LoRA vs AdaptFormer）は同じ VFM 上でも**異なる予測**を生む
- 異なる **VFM**（CLIP vs DINOv2）も**異なる予測**を生む
- 全体精度は似ているが、サンプル単位では「補完的」な誤りパターン

**図 1（ベン図）が示すこと**: CLIP+LoRA、CLIP+AdaptFormer、DINOv2+LoRA、DINOv2+AdaptFormer の 4 モデルの「高信頼度予測」を集合として見ると、共通部分（4 つ全てで一致：219 個）と独自予測（各モデル独自の上位 79〜103 個）が共存する。この**多様性**こそアンサンブルが効く根拠。

## 提案手法：V-PET（VFM-PEFT Ensemble Training）

設計は驚くほどシンプル。**4 フェーズで完結**：

```
フェーズ (a) Supervised PEFT
  各 (VFM, PEFT) ペアについて、ラベルあきデータ L で fine-tune
  → N × M 個の fine-tuned モデル θ̃_{n,m}

フェーズ (b) Pseudo-Label Generation
  各モデルでラベルなしデータ U の予測を出す（one-hot 化）
  → 各サンプル u に対して N×M 個の one-hot 疑似ラベル

フェーズ (c) Pseudo-Label Ensemble（Mean Labels）
  N×M 個の one-hot 疑似ラベルを平均 → ソフト疑似ラベル p̄_i
  （Mean Logits や Mean Probabilities ではなく Mean Labels が最良）

フェーズ (d) Self-Training
  1 つの (VFM, PEFT) ペアを選んで、ソフト疑似ラベルで再 fine-tune
  → 最終モデル θ*
```

**重要な設計判断**:

1. **τ=0 で全疑似ラベル使用**: 信頼度閾値で疑似ラベルを選別しない（FixMatch とは正反対）。「アンサンブルで品質を確保するから閾値要らない」という発想
2. **1 ラウンドの self-training**: 反復的な self-training（教師→生徒→新教師→...）ではなく、1 回で完了。シンプル化
3. **Mean Labels（One-hot 平均）の優位性**: ロジット平均/確率平均だと、異なるモデル間でスケールが揃わないため、強いモデルが支配的になる。One-hot 化することで全モデルを「投票」として等価に扱える

## なぜ Mean Labels が効くか：キャリブレーション問題

図 4 は **異なる fine-tuned VFM のエントロピー分布が大きく異なる**ことを示す。例えば DINOv2+AdaptFormer は高信頼な予測を多く出す（低エントロピー）が、CLIP+AdaptFormer は不確実な予測が多い（高エントロピー）。

確率を直接平均すると、高信頼モデルが他を圧倒してしまう（少数モデルの「意見」が反映されない）。**One-hot 化することで「argmax = 投票」に統一**し、すべてのモデルに等しい重みを与える。これは結果的に「**多数決による堅牢化**」になる。

## 新ベンチマーク：VTAB ベース 6 データセット

既存 SSL ベンチマーク（CIFAR-10/100, STL-10, Food101）の問題：**凍結 VFM が線形プローブだけで既に高精度を出してしまう**ため、SSL アルゴリズムが効くかを評価できない。そこで著者らは **VTAB（Visual Task Adaptation Benchmark）** から「凍結 VFM が苦戦する」6 データセットを選定：

| カテゴリ | データセット | ドメイン | クラス数 | ラベル/クラス |
|---|---|---|---|---|
| Natural | DTD | テクスチャ認識 | 47 | 3, 6 |
| Natural | SUN397 | 自然シーン | 397 | 3, 6 |
| Specialized | RESISC45 | リモートセンシング | 45 | 1, 2 |
| Specialized | Retinopathy | 医療画像（網膜症） | 5 | 4, 8 |
| Structured | CLEVR-C | 合成推論（カウント） | 8 | 1, 2 |
| Structured | KITTI | 自動運転（深度） | 4 | 5, 10 |

凍結 VFM の線形プロービング精度（CIFAR-10 で 85〜91% → 我々のベンチマークでは 30〜70%）。これにより SSL アルゴリズムの真の貢献が評価可能になる。

## 公正なハイパラチューニング：7 つの教師なし基準

SeSL ではラベルあきデータが希少なため、validation set を作るのが難しい。テストセットでチューニングするとデータリークが発生する（多くの過去研究の隠れた問題）。著者らは：

- **特徴量由来 5 基準**: AMI (Adjusted Mutual Information), ARI (Adjusted Rand Index), V-Measure, FMI (Fowlkes-Mallows Index), BNM (Batch Nuclear-norm Maximization)
- **ロジット由来 2 基準**: RankMe（行列ランクベース）, CHI (Calinski-Harabasz Index)

これら 7 つでハイパラ候補をランキングし、**平均ランクが最小**のものを採用。表 4 では、オラクルテスト精度との絶対誤差が単一基準（2.8〜12.0）より低い **2.6%** を達成。

## 実験結果

### 主要結果（表 3、12 設定平均）

| 手法 | CLIP/LoRA | CLIP/Adapt | DINOv2/LoRA | DINOv2/Adapt |
|---|---|---|---|---|
| Labeled Only | 55.7 | 55.6 | 56.0 | 51.6 |
| FixMatch | 53.7 | 53.7 | 47.6 | 47.7 |
| FlexMatch | 56.2 | 53.9 | 48.6 | 47.8 |
| SoftMatch | 59.7 | 56.3 | 50.2 | 48.5 |
| FineSSL | 53.9 | 51.6 | 49.8 | 46.8 |
| ST (self-training) | 54.7 | 56.7 | 52.9 | 52.5 |
| PET（PEFT アンサンブル） | 59.3 | 59.7 | 56.7 | 58.0 |
| **V-PET** | **60.5** | **61.0** | **61.2** | **61.5** |

**衝撃的観察**: DINOv2 では FixMatch/FlexMatch/SoftMatch が「**Labeled Only（51.6〜56.0%）よりも悪化**」（47.6〜50.2%）している。SSL がむしろ性能を**下げている**。

### ランキング頻度（図 5）

12 設定全体でのランキングを見ると、V-PET が最頻でトップを獲得。すべての設定で 1 位ではないが、平均ランクが最良。

### スケーラビリティ（図 6）

PEFT 手法やVFM の数を増やすにつれて V-PET の性能は向上するが、**収穫逓減**。著者推奨は **2〜4 個のアンサンブル**で性能と計算コストのバランスをとる。

### 計算コスト

「N×M モデルを fine-tune」と聞くと重そうだが、各 PEFT 訓練はラベルあきデータが少ないため軽い。実行時間は他の SSL ベースラインの **1.16 倍**程度。

## 限界・批判的視点

1. **アンサンブルの恣意性**: PEFT 手法と VFM の選択基準が明確でない。「LoRA と AdaptFormer」「CLIP と DINOv2」が最適な組み合わせかは未検証
2. **ベンチマークの非標準性**: VTAB ベースの新ベンチマークは妥当だが、コミュニティ標準ではない。今後の追試で安定するかは未知
3. **長尾分布への対処不足**: FlexMatch の SVHN 問題と同様、クラス不均衡の処理は議論されていない
4. **分類のみ**: セグメンテーション・検出への拡張は将来課題として明記される
5. **PEFT の理論的理解の欠如**: 「なぜ異なる PEFT が多様な疑似ラベルを生むか」のメカニズム解析は限定的（経験的観察のみ）

## 研究上の位置づけ

この論文は「**SeSL 研究のパラダイムシフト**」を実証した重要な研究だ。具体的には：

### 1. 「スクラッチ前提」から「基盤モデル前提」へ

[[entities/mixmatch]]（2019）、[[entities/fixmatch]]（2020）、[[entities/flexmatch]]（2021）は全て「Wide ResNet をスクラッチから訓練する」前提だった。本論文は「**事前学習済み VFM 上では SSL の貢献は驚くほど小さい**」ことを実証し、SeSL 研究の前提を根本から問い直した。

### 2. PEFT × SeSL という新分野

[[concepts/foundation-model]] × SeSL の交差点として、PEFT が中心的役割を果たすことが明確化された。LoRA / AdaptFormer / BitFit / VPT 等の PEFT 手法が SeSL の標準ツールキットに加わる。

### 3. アンサンブルの再評価

「複数モデルのアンサンブル」という古典的アイデアが、VFM × PEFT の組み合わせ多様性によって**新しい力**を得た。「Mean Labels（One-hot 平均）」というシンプルな技法が、複雑な閾値機構（FixMatch τ=0.95、FlexMatch CPL）よりも有効。

### 4. 既存 SeSL 手法への警鐘

FixMatch・FlexMatch・SoftMatch・FineSSL が DINOv2 上で **Labeled Only より悪化**するという結果は、これらの手法を「無条件に効くもの」として推す既存研究への警鐘。**ベンチマーク選択とハイパラチューニングの公正さ**が、SeSL 評価において決定的に重要。

## 用語と略称

- **VFM** = Vision Foundation Model（視覚基盤モデル）。CLIP, DINOv2, OpenCLIP, ImageNet21k 事前学習 ViT 等
- **PEFT** = Parameter-Efficient Fine-Tuning（パラメータ効率的ファインチューニング）。VFM 全体ではなく一部だけ更新。詳細: [[concepts/parameter-efficient-fine-tuning]]
- **LoRA** = Low-Rank Adaptation。Transformer 層の重み更新を低ランク行列で近似（Hu et al., 2021）
- **AdaptFormer** = アダプタベース PEFT。Transformer ブロック内に小さな MLP を挿入（Chen et al., 2022）
- **BitFit** = bias 項のみ更新する超軽量 PEFT（Zaken et al., 2022）
- **VPT-Deep / Visual Prompt Tuning** = 学習可能なプロンプトトークンを各層に挿入（Jia et al., 2022）
- **ConvPass** = 畳み込みベースの PEFT（Jie & Deng, 2022）
- **Fact-TT** = テンソル分解ベース PEFT（Jie & Deng, 2023）
- **V-PET** = VFM-PEFT Ensemble Training（本論文提案手法）
- **PET** = PEFT Ensemble Training（V-PET の単一 VFM 版）
- **ST** = Self-Training（自己訓練、本論文ではアンサンブルなしのベースライン）
- **Mean Labels** = one-hot 疑似ラベルの平均（V-PET のアンサンブル戦略）
- **Mean Logits / Mean Probabilities** = ロジットや確率の平均（比較対象、劣る）
- **FineSSL** = CLIP 視覚バックボーンと平衡マージン softmax を組み合わせる手法（Gan & Wei, 2024）
- **VTAB** = Visual Task Adaptation Benchmark（Zhai et al., 2020）。視覚表現を 19 タスクで評価
- **DTD** = Describable Textures Dataset（テクスチャ認識）
- **SUN397** = Scene Understanding Database 397 クラス
- **RESISC45** = Remote Sensing Image Scene Classification 45 クラス
- **CLEVR-C** = CLEVR Count（合成推論タスク）
- **Retinopathy** = 糖尿病性網膜症診断データセット
- **AMI / ARI** = Adjusted Mutual Information / Adjusted Rand Index（クラスタリング評価指標）
- **V-Measure** = エントロピーベースのクラスタリング評価
- **FMI** = Fowlkes-Mallows Index
- **BNM** = Batch Nuclear-norm Maximization（特徴行列の核ノルム）
- **RankMe** = 表現行列のランクで表現の質を測る（Garrido et al., 2023）
- **CHI** = Calinski-Harabasz Index
- **confirmation bias** = 誤疑似ラベルが自己強化されるループ（FixMatch でも論じられた）
- **SeSL** = Semi-Supervised Learning（半教師あり学習）。当 wiki では SSL（自己教師あり学習）との混同を避けるためこの略称を使用

## 関連ページ

- [[translations/revisiting-ssl-foundation-models]]: 日本語全文翻訳
- [[entities/v-pet]]: V-PET アルゴリズムのスペックシート
- [[concepts/parameter-efficient-fine-tuning]]: PEFT の概念解説（新規）
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像（本論文は系譜の現在地）
- [[concepts/foundation-model]]: 基盤モデルと SSL/SeSL の関係
- [[entities/fixmatch]]: 本論文の主要比較対象（VFM 上では性能低下）
- [[entities/flexmatch]]: 本論文の主要比較対象（同上）
- [[entities/clip]]: 本論文のバックボーン候補の一つ
- [[entities/dinov2]]: 本論文のバックボーン候補の一つ
- [[entities/mixmatch]]: スクラッチ前提時代の代表手法
