---
type: source
source_path: raw/papers/Masked Autoencoders Are Scalable Vision Learners.md
source_kind: paper
title: "Masked Autoencoders Are Scalable Vision Learners"
authors: [Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick]
year: 2021
venue: "arXiv:2111.06377 → CVPR 2022"
ingested: 2026-05-25
tags: [mae, self-supervised-learning, masked-image-modeling, vision-transformer, fair]
translation: [[translations/mae]]
---

# MAE: マスクオートエンコーダはスケーラブルな視覚学習器である

> 原典: [[translations/mae]] ・ `raw/papers/Masked Autoencoders Are Scalable Vision Learners.md`
> 著者: Kaiming He（プロジェクトリード）, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick（全員 Facebook AI Research, 旧 FAIR）
> 出典: arXiv:2111.06377（2021 年 11 月）→ CVPR 2022

> 翻訳メモ: 本要約は CLAUDE.md §4 の標準ルールと異なり、**Appendix も翻訳済み**（[[translations/mae]] 参照）。これはユーザーからの個別指示による。

---

## 一言まとめ

「**画像の 75% をランダムに隠して、隠した部分の画素を ViT で再構成させるだけ**」というシンプルな自己教師あり学習。BERT の MLM（masked language modeling）を画像に持ち込んだ最も素直な形だが、**「マスク率 75%」「非対称な encoder-decoder（encoder は可視 25% のパッチだけ処理）」「画素を直接予測」**という 3 つの設計判断で、ViT-Huge を ImageNet-1K のみで 87.8% top-1 までスケールさせた。**MAE の本当の強さは fine-tuning にあり、linear probing では弱い**という性質が、後の iBOT / DINOv2 / DINOv3 がハイブリッド設計に向かう原動力になった。CV における SSL を「**シンプル + スケール**」の方向に決定づけた歴史的論文。

---

## 背景と問題意識

### この論文以前の状況

- **NLP は BERT（2018）の MLM で爆発的に進歩**: 「文中の単語の 15% を `[MASK]` に置き換えて、当てさせる」だけで強力な事前学習が可能なことを示し、GPT 系の自己回帰モデリングと並んで NLP 基盤モデルの中核に
- **CV では SSL は対比学習（SimCLR, MoCo, BYOL）が中心**: 「同じ画像の異なる拡張」を正例として近づける手法。データ拡張の設計が肝
- **CV のマスク予測は長年存在したが大きく成功していなかった**: Context Encoder（2016, inpainting）、iGPT（2020, 画素を系列として予測）、BEiT（2021, dVAE 離散トークン予測）など。BERT の素直な「マスクして当てる」発想は CV でずっとうまく動かなかった

### この論文が問うた根本的な疑問

**「なぜビジョンで masked autoencoding が NLP の BERT ほどうまくいかないのか？」**

著者ら（He, Dollár, Girshick らという ResNet/Mask R-CNN チーム）は 3 つの仮説を立てる：

1. **アーキテクチャの違い**: CNN ではマスクトークンや位置埋め込みが自然に扱えなかった → **ViT の登場で解消済み**
2. **情報密度の違い**: 言語は意味的に密で、15% のマスクで十分挑戦的。画像は空間的冗長性が高く、**少しマスクしただけでは近傍から補間できてしまう** → **マスク率を 75% に上げて克服**
3. **デコーダの役割の違い**: 言語デコーダは「意味的単語」を予測するが、画像デコーダは「画素」を予測する → **デコーダ設計が学習表現の意味レベルを決める**ことを発見

この 3 つを正面から解決したのが MAE。

> **補足: なぜ BERT 流発想が CV で長年うまく動かなかったか** — NLP の単語と CV の画素は「情報の単位」として全く違う。「a [MASK] sat on the mat」の `[MASK]` は dog/cat/man など限られた語彙から選ばないと意味が通らず、強い意味理解が要求される。一方「画像のここのピクセルを隠した」は周りの色を見れば補完できてしまう。MAE はこのギャップを「**マスク率を BERT の 5 倍にする**」という雑だが効く方法で埋めた。

---

## 提案手法

<figure>

![](../../raw/assets/mae/x1.png)

<figcaption>図1: MAE の全体図。事前学習中、画像パッチの 75% がマスクされる。encoder は残りの 25% の可視パッチのみを処理（mask token を入力しない）。その後、エンコード済みパッチに mask token を挿入し、軽量な decoder が画素レベルで再構成する。事前学習後、decoder は破棄され、encoder のみが下流タスクに使われる。</figcaption>
</figure>

### コアアイデア 3 点

#### 1. 非対称な encoder-decoder

**この論文の最重要設計**。

- **Encoder**: ViT そのもの。ただし**可視パッチ（全体の 25%）のみ**を処理する。マスクトークンは入力しない。
- **Decoder**: 軽量。encoder の出力（25%）+ マスクトークン（75%, 学習可能ベクトル）を入力し、画素を再構成。Transformer ブロック ~8 層、幅 512-d の小型版で十分。
- **事前学習後は decoder を破棄**、encoder だけを下流タスクに使う。

> **補足: なぜ encoder にマスクトークンを入れないのか** — マスクトークン `[M]` は事前学習中にしか登場しない。これを encoder に入れて訓練すると、推論時（マスクなしの完全な画像）との分布ギャップが生まれ、linear probing 精度が 14% も落ちる（表 1(c)）。一方 encoder が常に「実際の画像パッチ」だけを見るようにすれば、このギャップが消える。**おまけに encoder の入力長が 4 分の 1 になるので訓練が 3 倍以上速くなる**という劇的な副次効果。

#### 2. マスク率 75%（驚異的に高い）

| | 典型的なマスク率 |
|---|---|
| BERT (NLP) | 15% |
| Context Encoder, BEiT 等 (CV 先行研究) | 20〜50% |
| **MAE** | **75%** |

著者らは図 5 のアブレーションで、**マスク率 40〜80% の広い範囲で fine-tuning が良く動き、75% が sweet spot** であることを示す。Linear probing もマスク率が高いほど良くなる（54.6% → 73.5% のジャンプ）。

#### 3. ターゲットは「画素」（しかもパッチごとに正規化した画素）

- **画素を直接 MSE で再構成**（BEiT のような事前訓練済み dVAE トークン予測ではない）
- ただし**「パッチ内画素を平均 0 分散 1 に正規化」**した値をターゲットにするとさらに良い（fine-tuning 84.9 → 85.4）

> **補足: なぜ画素直接が動くのか** — MAE 以前、BEiT は「画素は意味的でない」として dVAE で離散化していた。MAE はそれを覆し、「**画素ターゲットでも encoder には十分に意味的な表現が学ばれる**」と示した。decoder が pixel-specialization を吸収するため、encoder は抽象表現に集中できる、というのが論文の解釈。

### 学習の最小実装

```python
# 疑似コード
def mae_forward(x):
    tokens = patchify_and_embed(x)            # (B, 196, D)
    shuffled = random_shuffle(tokens)
    visible = shuffled[:, :49]                # 25% keep, 75% drop
    encoded = encoder(visible)                # ViT-L 等の重い encoder
    full = concat(encoded, mask_tokens)       # mask_tokens は学習可能ベクトル
    full = unshuffle(full, add_positional_emb=True)
    pred = decoder(full)                      # 軽量 Transformer
    loss = MSE(pred[masked_idx], target_pixels_normalized[masked_idx])
    return loss
```

ポイント: **疎演算は不要**。シャッフル → 切り捨て → unshuffle というシンプルな実装。

---

## 実験結果と知見

### ImageNet-1K で SOTA を更新（IN1K のみで）

| Method | ViT-B | ViT-L | ViT-H | ViT-H₄₄₈ |
|---|---|---|---|---|
| scratch (我々の実装) | 82.3 | 82.6 | 83.1 | – |
| DINO | 82.8 | – | – | – |
| MoCo v3 | 83.2 | 84.1 | – | – |
| BEiT | 83.2 | 85.2 | – | – |
| **MAE** | **83.6** | **85.9** | **86.9** | **87.8** |

- ViT-H/448 で **87.8% top-1**、ImageNet-1K のみを使う手法で当時の SOTA
- BEiT より速くシンプルで精度も高い
- MoCo v3 と比較して、**MAE は訓練時間が短い**（ViT-L で 1600 epoch / 31h vs MoCo v3 の 300 epoch / 36h）

### 「Linear probing は弱いが fine-tuning は強い」という重要な性質

<figure>

![](../../raw/assets/mae/x10.png)

<figcaption>図9（再掲）: ViT-L で fine-tune するブロック数による精度。0 ブロック = linear probing、24 = full fine-tuning。MoCo v3 は linear probing で MAE を上回るが、1 ブロック以上を fine-tune すると MAE が逆転する。MAE の表現は「線形分離性は低いが非線形に強い」。</figcaption>
</figure>

これは MAE の最大の特徴で、**「画素再構成は意味的に粗いタスクなので、表現が線形に整理されない」**ことの裏返し。詳細な発見：

- **Linear probing**: MAE ViT-L 75.8% vs MoCo v3 ViT-L 77.6% → MoCo 系（対比学習）が勝つ
- **1 block fine-tune**: 73.5% → **81.0%** に跳ね上がる（非線形 head を 1 つ載せるだけ）
- **Full fine-tune**: 85.9%、SOTA

> **補足: なぜこれが重要か** — DINO/iBOT/DINOv2/DINOv3 はこの「MAE は fine-tune には強いが linear probing は弱い」性質をはっきり対比軸として使い、**「我々の手法は凍結特徴量がそのまま強い」**ことを売りにする。MAE と DINO 系が**補完的に存在**するのはこの性質の違いによる。iBOT が MIM と DINO を統合したのも、まさにこの両者の長所を取りに行く設計。

### 転移学習でも教師あり事前学習を超える

| Task | ViT-L 改善 |
|---|---|
| COCO 検出 (AP^box) | supervised 49.3 → MAE 53.3（**+4.0**） |
| COCO segm. (AP^mask) | supervised 43.9 → MAE 47.2（**+3.3**） |
| ADE20K seg. (mIoU) | supervised 49.9 → MAE 53.6（**+3.7**） |
| iNat 2018 (ViT-H₄₄₈) | prev best 81.2 → MAE **86.8** |

特に detection/segmentation で大きく上回り、**密予測タスクへの強い適合性**を示した。

### ロバストネス（Appendix C）

ViT-H₄₄₈ で：

| Benchmark | MAE | 教師あり ViT-H |
|---|---|---|
| ImageNet-A (adversarial) | **76.7** | 33.1 |
| ImageNet-R (rendition) | **66.5** | 50.3 |
| ImageNet-Sketch | **50.9** | 38.0 |

教師ありを大きく上回り、後の DINOv2/DINOv3 でも観察される「**SSL は OOD ロバストネスが強い**」傾向を最初に明示。

### Decoder の役割という意外な発見

- **Linear probing には深い decoder（〜8 ブロック）が重要**: 浅い decoder だと encoder が画素再構成の specialization を吸収せざるを得ず、表現が低レベル化
- **Fine-tuning なら decoder 1 ブロックでも十分**: encoder の最後の層が認識タスクに適応できるため
- → 「decoder が**意味レベルの buffer** として働く」という洞察

---

## なぜこの研究が CV にとって重要か

1. **「シンプル + スケール」を CV SSL に持ち込んだ**: NLP の BERT/GPT が示した「シンプルなアルゴリズム + 大量データ + 大規模モデル = 強い基盤モデル」というレシピが、CV でも通用することを実証。コンピュータビジョンの SSL を「**マスクして当てる**」という単純化した方向に決定づけた。
2. **75% という極端なマスク率の発見**: CV と NLP の本質的違い（情報密度）を可視化し、後続研究（iBOT, MaskFeat, SimMIM, DINOv2/v3 の MIM 損失設計）すべての出発点になった。
3. **「非対称 encoder-decoder」設計**: encoder のメモリ・計算を 4 倍削減することで、ViT-Huge クラスのスケールを実現可能に。後の DINOv3 の sequence packing や FSDP と並ぶ「**大規模 SSL を可能にしたエンジニアリング**」の起点。
4. **linear probing vs fine-tuning の解離を浮き彫りに**: 「partial fine-tuning」というベンチマーク方式を提案し、SSL 評価の議論を深めた。後の iBOT/DINOv2/DINOv3 の評価指標の選び方に直接影響。
5. **教師あり事前学習を密予測タスクで明確に上回った**: detection/segmentation で MAE > supervised を実証し、「**事前学習は教師ありがデフォルト**」という常識を覆した。

---

## 限界・批判的視点

- **Linear probing と k-NN が弱い**: 凍結特徴量で軽い分類器を訓練する用途では DINO 系の方が強い。これは **「MAE の表現は線形に整理されていない」** という構造的問題で、根本的に MIM 単独では解決しにくい
- **再構成された画素は意味的にぼんやり**: 図 2/3 を見ると、再構成は「もっともらしい」が ground truth とは異なる。これは「**画素レベルでは複数の解がある**」という MIM の宿命
- **fine-tuning に強いハイパラ要求**: 表 9 の通り、mixup, cutmix, drop path, layer-wise lr decay 等のレシピを慎重に設計しないと潜在能力が出ない
- **対比学習に比べて表現の意味的構造化が弱い**: 例えば「semantic segmentation が attention map に教師なしで現れる」のような DINO の創発性質は MAE では起きない
- **モデル圧縮には不向き**: MAE の強さは fine-tune ありの大規模モデルで出るので、軽量化された下流応用には DINOv2 系の方が向く
- **動画への自然な拡張は別研究が必要**: 後の VideoMAE などで対応

これらの限界が **iBOT（DINO + MIM）→ DINOv2 → DINOv3** という「MAE の MIM 強みを DINO 系統に統合する」流れを生んだ。

---

## 用語と略称

| 略称 | 展開 | 短い意味 |
|---|---|---|
| **MAE** | Masked Autoencoder | 本論文の手法 |
| **MIM** | Masked Image Modeling | 画像版マスク予測の総称。[[concepts/masked-image-modeling]] |
| **MLM** | Masked Language Modeling | NLP 版（BERT が代表） |
| **DAE** | Denoising Autoencoder | MAE の理論的祖先。[[concepts/denoising-autoencoder]] |
| **ViT** | Vision Transformer | バックボーン。[[concepts/vision-transformer]] |
| **BERT** | Bidirectional Encoder Representations from Transformers | MAE の発想元 |
| **BEiT** | BERT pre-training of image Transformers | 直前の CV MIM 手法（dVAE トークン予測） |
| **iGPT** | image GPT | OpenAI の画素系列自己回帰 |
| **dVAE** | discrete VAE | 画像を離散トークン化する VAE、DALL-E 由来 |
| **MoCo v3** | Momentum Contrast v3 | ViT 対応版の MoCo、対比学習の代表 |
| **DINO** | self-DIstillation with NO labels | 同時期の SSL、対比軸として頻出。[[entities/dino]] |
| **SimCLR / BYOL** | 対比学習・非対比学習の代表 | 対比軸として頻出 |
| **MSE** | Mean Squared Error | MAE の再構成損失 |
| **MLP** | Multi-Layer Perceptron | decoder の活性化 |
| **LN** | Layer Normalization | ViT の正規化 |
| **CLS token** | classification token | ViT が画像全体の表現として持つ特殊トークン |
| **`[M]`** | Mask token | 学習可能ベクトル、欠落パッチの位置を示す |
| **FPN** | Feature Pyramid Network | 物体検出で多スケール特徴量を作る構造 |
| **Mask R-CNN** | — | 物体検出・インスタンスセグメンテーションの代表モデル |
| **UperNet** | — | セマンティックセグメンテーションのデコーダ |
| **FAIR** | Facebook AI Research | 著者所属（現 Meta AI Research） |
| **JFT-300M** | — | Google の 3 億画像教師ありデータセット（非公開） |
| **IN1K / IN22K** | ImageNet-1K / ImageNet-22K | 主要事前学習データセット。[[entities/imagenet]] |
| **iNat / Places** | iNaturalist / Places205 | 細粒度・シーン分類ベンチマーク |
| **AdamW** | Adam with decoupled Weight decay | 標準 optimizer |
| **LARS** | Layer-wise Adaptive Rate Scaling | 大バッチ optimizer、linear probing で使用 |
| **drop path** | stochastic depth | 正則化、過学習抑制 |
| **mixup / cutmix** | — | 画像レベルのデータ拡張 |
| **RandAug** | RandAugment | 自動データ拡張 |
| **EMA** | Exponential Moving Average | モデル重みの移動平均 |

---

## 関連ページ

- 翻訳: [[translations/mae]]
- 概念:
  - [[concepts/masked-image-modeling]] — MAE が代表する MIM 系統全般
  - [[concepts/denoising-autoencoder]] — MAE の理論的祖先（DAE）
  - [[concepts/self-supervised-learning]] — SSL 全般の中の位置づけ
  - [[concepts/vision-transformer]] — バックボーン
  - [[concepts/knn-evaluation-protocol]] — linear probing の議論の延長
- エンティティ:
  - [[entities/mae]] — MAE モデルのスペックシート
  - [[entities/imagenet]] — 事前学習データ
- 系譜的関連:
  - [[entities/dino]] — 同時期の対照的アプローチ
  - [[entities/ibot]] — MAE の MIM 損失を DINO に統合した後継
  - [[entities/dinov2]] — iBOT を基盤モデルに発展
  - [[entities/dinov3]] — Gram anchoring で long-training 問題を解決
