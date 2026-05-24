---
type: source
source_path: raw/papers/Emerging Properties in Self-Supervised Vision Transformers.md
source_kind: paper
title: Emerging Properties in Self-Supervised Vision Transformers
authors: [Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin]
year: 2021
venue: ICCV 2021 (arXiv:2104.14294)
ingested: 2026-05-24
tags: [dino, self-supervised-learning, vision-transformer, knowledge-distillation, segmentation]
translation: [[translations/dino-emerging-properties-in-self-supervised-vit]]
---

# DINO: 自己教師あり Vision Transformer に現れる新たな特性

> 原典: [[translations/dino-emerging-properties-in-self-supervised-vit]] ・ `raw/papers/Emerging Properties in Self-Supervised Vision Transformers.md`
> 著者: Mathilde Caron ほか（Facebook AI Research, Inria, Sorbonne University）
> 出典: arXiv:2104.14294（2021）→ ICCV 2021

---

## 一言まとめ

ViT（Vision Transformer, 画像をパッチに分割して Transformer に通す画像分類モデル）を**ラベルを一切使わず**に学習する手法 **DINO** を提案した論文。学習済み ViT の自己注意マップに**教師なしでオブジェクト境界が浮かび上がる**という驚くべき創発的性質を発見し、加えて単なる k-NN（k 近傍法）だけで ImageNet で 78.3% という当時 SOTA に肉薄する精度を出せることを示した。「自己教師あり学習は CV における BERT になり得る」というその後の流れを決定づけた重要論文。

---

## 背景と問題意識

### この論文以前の状況

- **ViT（Vision Transformer）** は 2020 年（Dosovitskiy ら）に登場した「画像を 16×16 のパッチに切って Transformer に流し込む」だけの単純なモデルで、CNN（畳み込みニューラルネットワーク）に匹敵する分類精度を達成した。ただし、当時の ViT は教師あり学習で訓練されており、CNN に対する明確な優位性（少ない計算量・少ないデータ・特徴量の固有の良さ）を示せていなかった。
- 一方 NLP（自然言語処理）では、**BERT** や **GPT** の成功は**自己教師あり事前学習**（ラベルなしの大量テキストでまず学習する）によって支えられていた。画像レベルの教師信号は「画像 1 枚 → ラベル 1 つ」と豊かな視覚情報を 1 概念に潰してしまう。だから視覚でも自己教師あり学習が鍵になるはず、という直観があった。
- 画像での自己教師あり学習（SSL, Self-Supervised Learning）には既に **SimCLR, MoCo, BYOL, SwAV** などの強力な手法が CNN ベースで存在していたが、ViT に SSL を持ち込んだらどうなるのか？という問いはまだ十分に答えられていなかった。

### この論文が問うたこと

> 「SSL は ViT に対して、CNN にはない**新しい性質**を引き出すのではないか？」

著者らは、SSL × ViT という組み合わせを丁寧に調べることで、単なる精度の向上にとどまらない**創発的特性（emerging properties）** を発見できると見立てた。タイトルの "Emerging Properties" はこれを指す。

> **補足: 自己教師あり学習（SSL）とは** — 人手の正解ラベルを使わず、データ自身から「正解」を作って学習させる枠組み。画像なら「同じ画像に対する 2 つの異なる切り抜き／変形は近い表現に、別画像とは遠い表現に」というように、データ拡張の対の相対関係をラベル代わりにする。詳しくは [[concepts/self-supervised-learning]]。

> **補足: なぜ ViT × SSL が興味深いか** — CNN は畳み込みという「局所性（locality）」と「平行移動不変性」が組み込まれた帰納バイアス（inductive bias, モデル構造に最初から仕込まれた仮定）を持つ。ViT はその帰納バイアスが圧倒的に弱い分、データから何でも学べる代わりに大量のラベルなりデータなりが必要になる。SSL はそのラベル不要側を埋める可能性を持つ。

---

## 提案手法: DINO

**DINO = self-DIstillation with NO labels**（ラベルなしの自己蒸留）。名前自体が手法を要約している。

<figure>

![](../../raw/assets/dino/x1.png)

<figcaption>図2: DINO の全体図。同じ画像の 2 つの異なる切り抜きを student と teacher（両者は同じアーキテクチャ）に流し、両者の出力分布が一致するよう student をクロスエントロピーで学習する。teacher は student の EMA で更新され、出力には centering と sharpening を施す。</figcaption>
</figure>

### コアアイデア

通常の**知識蒸留（knowledge distillation, KD）** は「教師モデル（大きな or 訓練済み）の出力を、生徒モデル（小さい）が真似るよう学習させる」枠組み。ふつうは教師があらかじめ存在する。

DINO はこれを逆手に取り、「**教師も生徒も同じ無学習のネットワークから始め、教師は生徒自身の過去の状態から作る**」という設定にする。これがタイトルにある "self-distillation"（自己蒸留）の意味。

> **補足: 略称まとめ（本ページ冒頭）**
> - **SSL** = Self-Supervised Learning（自己教師あり学習）
> - **KD** = Knowledge Distillation（知識蒸留）
> - **ViT** = Vision Transformer
> - **CNN / convnet** = Convolutional Neural Network（畳み込みニューラルネット）
> - **EMA** = Exponential Moving Average（指数移動平均）
> - **k-NN** = k-Nearest Neighbors classifier（k 近傍分類器）
> - **CE** = Cross Entropy（クロスエントロピー損失）
> - **MSE** = Mean Squared Error（二乗誤差）
> - **BN** = Batch Normalization（バッチ正規化）
> - **MLP** = Multi-Layer Perceptron（多層パーセプトロン）

### 学習の流れ（最小構成）

1. 入力画像 $x$ から 2 つの異なる**ビュー**（ランダム拡張による切り抜き）$x_1, x_2$ を作る。
2. 両ビューを student $g_{\theta_s}$ と teacher $g_{\theta_t}$ にそれぞれ通す。両者は同じアーキテクチャ、別パラメータ。
3. 各ネットワークは $K$ 次元の確率分布を出力する（最後に softmax）。teacher の温度 $\tau_t$ は低め（鋭い分布 ＝ sharpening）、student の温度 $\tau_s$ はやや高め。
4. **損失はクロスエントロピー**: teacher の分布に student の分布を近づける。式: $\min_{\theta_s} H(P_t(x_1), P_s(x_2)) + H(P_t(x_2), P_s(x_1))$。
5. **勾配は student にのみ流す**（teacher には `stop-gradient`）。
6. **teacher は EMA で更新**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$。$\lambda$ は 0.996 → 1 のコサインスケジュール。
7. **崩壊（collapse）回避**は teacher 出力の **centering**（バッチ平均を引く）と **sharpening**（低温度 softmax）の組合せのみ。対比損失も predictor も BN も不要。

> **補足: なぜ崩壊回避が必要か** — 「すべての入力に対し同じ出力を返せば損失は最小化できる」という自明解（trivial solution, モデルの「崩壊」）に陥らないようにする工夫が、すべての非対比型 SSL で必須。BYOL は predictor + BN、SwAV はクラスタ割当の Sinkhorn-Knopp 正規化で防ぐ。DINO は centering（1 次元に支配されるのを防ぐ）と sharpening（一様分布に潰れるのを防ぐ）が**逆方向の力**として釣り合う、という説明をしている（§5.3, 図 7）。

### 追加の重要要素

- **マルチクロップ（multi-crop）拡張**: SwAV 由来。大きな「global」ビュー 2 つ（$224^2$）に加えて、小さな「local」ビュー数本（$96^2$）も student に流す。teacher には global だけ流す。これで「**局所 → 大域**」の対応を学習することになり、性能を大きく押し上げる（§5.1 表 7 で +5 ポイント、§5.4 表 8 で 2 倍速）。
- **小さいパッチ**: ViT は通常 16×16 パッチだが、DINO では 8×8 にすると性能が大きく上がる（§5.1 図 5）。パラメータ数は変わらないが、トークン列が 4 倍に長くなり計算量は跳ね上がる。
- **モメンタム・エンコーダ（momentum encoder）**: EMA で更新される teacher。MoCo 由来。DINO では「single epoch を凍結した teacher」「前イテレーションの student のコピー」など他案も試したが、モメンタムが最良（§5.2 図 6）。著者は「これは Polyak-Ruppert 平均によるモデルアンサンブルが、訓練中ずっと続いているようなものだ」と解釈している。

### 教師あり知識蒸留との違い（要注意）

| | 通常の KD | DINO（self-distillation） |
|---|---|---|
| teacher | 事前学習済みで固定 | 学習中に student の EMA で動的に構築 |
| student | 小さなモデル（圧縮目的） | teacher と同じアーキテクチャ |
| ラベル | 必要 | **不要** |

---

## 実験結果と知見

### ImageNet 線形分類 (§4.1, 表 2)

- ViT-S/16 で **DINO は SwAV/BYOL/MoCov2 を線形評価で +3.5%, k-NN で +7.9% 上回る**。
- ViT-B/8 で **線形 80.1%, k-NN 77.4%** を達成（当時の SSL 系で最高水準）。
- **驚異的なのは k-NN が線形評価にほぼ匹敵すること**（74.5% vs 77.0% on ViT-S）。これは ResNet-50 や他 SSL では起きない、DINO × ViT 固有の現象。

### 創発する自己注意マップ (§4.2.2, 図 1, 3, 4)

<figure>

![](../../raw/assets/dino/attn6.png)

<figcaption>図1（再掲）: 教師なしで学習された ViT-S/8 の [CLS] トークンの自己注意マップ。最終層の異なるヘッドが、犬・鳥などのオブジェクトに自動的にフォーカスしている。</figcaption>
</figure>

- [CLS] トークン（クラス予測用に系列に追加される特殊トークン）が、ラベルなしで学習しただけなのに**オブジェクト境界を捉える**。
- 教師あり ViT ではこの性質は曖昧で、PASCAL VOC12 で Jaccard 類似度を比較すると **DINO の方が明確に上**（§4.2.2 図 4）。
- DAVIS-2017 動画オブジェクトセグメンテーションで、訓練もファインチューンも一切なしの単純な特徴量最近傍だけで、SSL convnet を上回る（§4.2.2 表 5）。

### その他の特性

- **画像検索**（§4.2.1 表 3）: 凍結 DINO 特徴量で Oxford/Paris 検索が教師あり ViT を上回る。
- **コピー検出**（表 4）: ViT-B/8 で mAP 85.5。
- **転移学習**（§4.2.3 表 6）: 7 つの下流データセット（CIFAR, iNat, Flowers, Cars, etc）で、教師あり事前学習を一貫して上回る。

### 計算コスト (§5.4 表 8)

- 8-GPU マシン × 2 台で 3 日、$2 \times 224^2 + 10 \times 96^2$ マルチクロップ・300 エポックで 76.1%。
- バッチサイズ 8 でも動く（性能は落ちるが）→ 大規模モデルへの拡張可能性。

---

## なぜこの研究が CV にとって重要か

- **「CV における BERT」探しの本命候補**を示した。著者が結論で書いている通り、この結果は「自己教師あり学習が ViT に基づく BERT 様モデルを開発する鍵となり得る」という証拠を提供する。これがその後の MAE（He et al., 2022）、DINOv2（Oquab et al., 2023）、SAM 等の流れに直結する。
- **「特徴量の品質」を測る新しい眼**を持ち込んだ: k-NN という極めて素朴な指標で良い性能が出ることは、特徴量空間そのものが意味的にきれいに整理されていることを意味する。線形プロービングよりも特徴量の「素の品質」を反映する。
- **教師なしセグメンテーション**という長年難しかった問題に、「単に SSL で ViT を学習するだけ」という意外な経路を示した。

---

## 限界・批判的視点

- **計算コスト**: パッチ「/8」と多数の local crop を併用すると ViT-B/8 で 8-GPU × 2 台 × 数日が必要。「小さいパッチが効く」という発見は美しいが、計算量が patch サイズの 2 乗で増える。
- **データセット**: 主実験は ImageNet（120 万枚、キュレーション済み）。著者自身が結論で「未キュレーション画像で大規模に試したい」と述べている。後の DINOv2（[[entities/dino]] を参照する後継研究）で実際に拡張される。
- **「emerging properties」の理論的説明はない**: なぜ ViT × SSL で注意マップが意味的にきれいになるのかについての厳密な説明はなく、経験的観察にとどまる。
- **k-NN の性能向上には特定の組合せが必須**: momentum encoder + multi-crop + 小さなパッチを全部揃えないと k-NN が伸びない。論文内のアブレーション（§5.1, §B）は丁寧だが、これらの組合せが本質なのかどうかは未だ完全には解明されていない。

---

## 用語と略称

| 略称 | 展開 | 短い意味 |
|---|---|---|
| **DINO** | self-**DI**stillation with **NO** labels | 本論文の提案手法 |
| **SSL** | Self-Supervised Learning | ラベルなし学習。詳細: [[concepts/self-supervised-learning]] |
| **KD** | Knowledge Distillation | 教師ネットの出力を生徒に真似させる学習。詳細: [[concepts/knowledge-distillation]] |
| **ViT** | Vision Transformer | 画像をパッチ分割して Transformer に通すモデル。詳細: [[concepts/vision-transformer]] |
| **CNN / convnet** | Convolutional Neural Network | 畳み込みベースの古典的視覚モデル |
| **EMA** | Exponential Moving Average | 移動平均で値を滑らかに更新する方法 |
| **k-NN** | k-Nearest Neighbors | 訓練データの k 個の近傍で多数決する分類器。詳細: [[concepts/knn-evaluation-protocol]] |
| **[CLS] token** | Classification token | 系列に付加されクラス予測等に使われる学習可能トークン |
| **BN** | Batch Normalization | バッチごとに正規化する層。ViT では既定で使わない |
| **CE** | Cross Entropy | 分布間の差を測る損失関数 |
| **SK** | Sinkhorn-Knopp | クラスタ割当を均等化する反復アルゴリズム（SwAV で使用） |
| **InfoNCE / INCE** | InfoNCE loss | 対比学習で使われる損失（MoCo, SimCLR 等） |
| **GeM pooling** | Generalized Mean pooling | 検索でよく使われるプーリング |
| **collapse** | 崩壊 | SSL で全入力が同じ表現に潰れてしまう自明解 |
| **multi-crop** | マルチクロップ拡張 | 大小複数の crop を作る SwAV 由来のデータ拡張 |
| **momentum encoder** | モメンタム・エンコーダ | EMA で更新される teacher ネット |
| **centering** | センタリング | バッチ平均を引いて偏りを取り除く操作 |
| **sharpening** | シャープニング | 低温度 softmax で分布を鋭くする操作 |
| **predictor** | 予測器 | BYOL/SimSiam で student の最後に付ける追加 MLP |
| **inductive bias** | 帰納バイアス | モデルアーキテクチャに最初から仕込まれた仮定 |

---

## 関連ページ

- 翻訳: [[translations/dino-emerging-properties-in-self-supervised-vit]]
- 概念:
  - [[concepts/self-supervised-learning]]（SSL 全般 / momentum encoder / multi-crop も含む）
  - [[concepts/vision-transformer]]
  - [[concepts/knowledge-distillation]]
  - [[concepts/knn-evaluation-protocol]]
- エンティティ:
  - [[entities/dino]]（本論文で提案された手法そのもののスペックシート）
  - [[entities/imagenet]]
