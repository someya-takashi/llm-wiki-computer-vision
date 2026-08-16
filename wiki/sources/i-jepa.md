---
type: source
source_path: raw/papers/Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.md
source_kind: paper
title: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
authors: [Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas]
year: 2023
venue: "arXiv:2301.08243 → CVPR 2023"
ingested: 2026-06-17
tags: [i-jepa, jepa, self-supervised-learning, vision-transformer, fair, lecun]
translation: "[[translations/i-jepa]]"
related: ["[[concepts/joint-embedding-predictive-architecture]]", "[[concepts/self-supervised-learning]]", "[[concepts/masked-image-modeling]]", "[[entities/i-jepa]]", "[[entities/mae]]"]
---

# I-JEPA: Joint-Embedding Predictive Architecture による画像の自己教師あり学習

> 原典: [[translations/i-jepa]] ・ `raw/papers/Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.md`
> 著者: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, **Yann LeCun**, Nicolas Ballas（全員 Meta AI / FAIR ほか）
> 出典: arXiv:2301.08243（2023 年 1 月）→ CVPR 2023

> 翻訳メモ: 本要約に対応する [[translations/i-jepa]] は CLAUDE.md §4 の標準ルールと異なり **Appendix A〜E も翻訳済み**（ユーザーからの個別指示による）。

---

## 一言まとめ

**「画像から 1 つのコンテキストブロックを取り、同じ画像の複数のターゲットブロックの *表現* を予測する」** だけの、データ拡張に依存しない自己教師あり学習（SSL, [[concepts/self-supervised-learning]]）。MAE が「マスクして **画素** を再構成する」のに対し、I-JEPA は「マスクして **抽象表現** を予測する」点が決定的に異なる。これにより MAE より意味的な凍結特徴量を、対比学習（DINO/iBOT）より圧倒的に少ない計算で獲得できる。Yann LeCun が提唱する **JEPA（Joint-Embedding Predictive Architecture, [[concepts/joint-embedding-predictive-architecture]]）** という世界モデル構想の、画像領域における最初の本格的実装。後の V-JEPA / V-JEPA 2（動画）へと続く系譜の起点。

---

## 背景と問題意識

この論文以前、画像 SSL には 2 つの大きな系統があり、それぞれに弱点があった。

- **不変性ベース（invariance-based）= 対比型・自己蒸留型**: SimCLR / MoCo / BYOL / DINO / iBOT など（[[concepts/contrastive-learning]] / [[concepts/self-supervised-learning]]）。「同じ画像の 2 つの **手作業データ拡張ビュー**（random crop, color jitter, blur 等）を似た埋め込みにする」。**高い意味レベルの凍結特徴量**を得られる（線形プロービング・k-NN が強い）が、拡張が **強いバイアス** を仕込むため、必要とする不変性が異なる下流タスク（分類 vs セグメンテーション）で問題になり、画像以外のモダリティへ一般化しにくい。さらに各画像の **複数ビューを処理** する必要がありスケールしにくい。
  - **補足: 不変性（invariance）** = 入力が多少変わっても出力（埋め込み）が変わらない性質。「色を変えても同じ犬」と教えることが、色が手がかりになるタスクには害になりうる、という話。
- **生成型（generative）= マスク再構成型（MIM, [[concepts/masked-image-modeling]]）**: MAE / BEiT / SimMIM など。「画像の一部を隠し **画素やトークンを再構成**」する。データ拡張バイアスが小さく他モダリティに一般化しやすいが、得られる表現は **意味レベルが低く**、線形プロービングでは対比型に負ける（[[entities/mae]] 参照）。フルファインチューニングのような手の込んだ適応が必要。

> **核心の問い**: 「**手作業データ拡張という追加の事前知識を使わずに、対比型なみに意味的な凍結表現を学べないか？**」

LeCun は以前から、生物の脳は「感覚入力をそのまま再構成する」のではなく「**内部モデルで感覚応答を予測する**」ことで世界を学ぶ、という認知学習理論を引き、**画素ではなく抽象表現空間で予測する** アーキテクチャ（JEPA）を提唱していた。I-JEPA はそれを画像 + ViT + マスキングで具体化した最初の本格実装である。これは [[concepts/masked-image-modeling]]（MIM）の「再構成」を「表現予測」へ置き換えた、SSL の第 3 の系統と言える。

---

## エネルギーベースモデル（EBM）による 3 アーキテクチャの整理

論文は 3 つの SSL アーキテクチャを **Energy-Based Model（EBM, 入力の「両立しなさ」をエネルギーで測る枠組み）** で統一的に並べる。これが本論文の概念的な背骨である。

<figure>

![](../../raw/assets/i-jepa/x2.png)

<figcaption>図2(a): Joint-Embedding Architecture（JEA）。x, y を別々のエンコーダに通し、両立するペアの埋め込み距離 D(sₓ, s_y) を最小化。対比型・自己蒸留型（SimCLR/DINO）がこれ。崩壊（collapse）を防ぐ仕掛けが必須。</figcaption>
</figure>

<figure>

![](../../raw/assets/i-jepa/x3.png)

<figcaption>図2(b): Generative Architecture（生成型）。追加変数 z（どこを埋めるかを指定するマスク・位置トークン）に条件付けたデコーダで、x から y を画素レベルで再構成。MAE/BEiT がこれ。z の情報容量が低ければ崩壊しない。</figcaption>
</figure>

<figure>

![](../../raw/assets/i-jepa/x4.png)

<figcaption>図2(c): Joint-Embedding Predictive Architecture（JEPA）。生成型と似るが、損失を入力（画素）空間ではなく埋め込み空間で取る。z に条件付けた予測器で、x から y の「表現」s_y を予測。I-JEPA はこれの画像・マスキング実装。</figcaption>
</figure>

- **(a) JEA**: 両立する入力に似た埋め込み。**崩壊**（= 何を入れても同じ定数を返す自明解）を防ぐため対比損失・冗長性削減・クラスタリング・非対称設計などが要る。
- **(b) 生成型**: x から y を **画素で直接再構成**。再構成ターゲットがあるので崩壊しない。
- **(c) JEPA**: 生成型と構造は似るが、**損失を埋め込み空間で取る**。「画素を当てる」のではなく「**相手の表現を当てる**」。崩壊は JEA 同様の懸念があり、I-JEPA は **x-エンコーダと y-エンコーダの非対称性（EMA ターゲット）** で防ぐ。
  - **補足: なぜ表現空間予測が「意味的」になるのか** — 画素再構成は背景の芝生 1 本まで合わせる必要があり、モデルは無関係な低レベル詳細に容量を割く。表現空間予測なら、ターゲットエンコーダが「予測不能で無意味な詳細」をあらかじめ捨てた抽象ターゲットを作るので、予測器は意味のある構造だけを学べばよい。

---

## 提案手法：I-JEPA

<figure>

![](../../raw/assets/i-jepa/x5.png)

<figcaption>図3: I-JEPA の全体像。コンテキストエンコーダ（ViT、可視パッチのみ処理）→ 予測器（狭い ViT、位置マスクトークンに条件付け）→ ターゲットブロックの表現を予測。ターゲットエンコーダ（右）はコンテキストエンコーダ重みの EMA で更新される。損失はターゲットエンコーダ出力との L2 距離。</figcaption>
</figure>

3 つの ViT（[[concepts/vision-transformer]]）で構成される。

1. **ターゲットエンコーダ** $f_{\bar\theta}$: 画像全体を $N$ パッチに分けて表現 $s_y$ を計算。その **出力から** $M=4$ 個のブロックをサンプリングしてターゲットとする。
2. **コンテキストエンコーダ** $f_\theta$: 1 個のコンテキストブロック（ターゲットと重なる部分は除去）の **可視パッチだけ** を処理して $s_x$ を出力。
3. **予測器** $g_\phi$: $s_x$ と、予測対象パッチの **マスクトークン（共有学習ベクトル + 位置埋め込み）** を入力し、各ターゲットブロックの表現 $\hat s_y(i)$ を出力。$M$ 個のブロックそれぞれに $M$ 回適用。

**損失**は予測表現とターゲット表現の平均 L2 距離のみ：

$$\frac{1}{M}\sum^{M}_{i=1}\sum_{j\in B_{i}}\lVert\hat{{\bm{s}}}_{y_{j}}-{\bm{s}}_{y_{j}}\rVert^{2}_{2}$$

- $\phi, \theta$ は勾配で学習、**ターゲットエンコーダ $\bar\theta$ はコンテキストエンコーダの EMA（指数移動平均）で更新**。この非対称性（[[concepts/self-supervised-learning]] の momentum encoder と同じ仕掛け）が崩壊を防ぐ鍵。
- **2 つの決定的な設計判断**:
  1. **ターゲットはエンコーダの *出力* をマスクして作る**（入力をマスクしない）。出力マスクにすると各ターゲットが画像全体の文脈を見た高意味表現になる。入力マスクにすると Top-1 が 67.3 → 56.1 に落ちる（表11）。
  2. **multi-block マスキング**（図4）: 大きめ（scale 0.15–0.2）の意味的ターゲットを 4 個 + 情報量の多い（scale 0.85–1.0）疎なコンテキスト 1 個。

<figure>

![](../../raw/assets/i-jepa/x6.png)

<figcaption>図4: multi-block マスキングの例。各行が 1 枚の画像。コンテキスト（大きな可視領域）から、複数の中サイズのターゲットブロック（白枠）の表現を予測する。ターゲットが意味的で、コンテキストが情報量豊富かつ疎、になるよう設計されている。</figcaption>
</figure>

---

## 実験結果と知見

### 意味的タスク（分類）— MAE 系を圧倒、対比型に肉薄

- **ImageNet-1K 線形プロービング（表1）**: I-JEPA ViT-H/14 で **79.3%**、ViT-H/16₄₄₈（解像度 448）で **81.1%**。同じく拡張なしの MAE ViT-H/14（77.2）・CAE・data2vec を上回り、**拡張ありの iBOT ViT-L/16（81.0）に並ぶ**。
- **半教師あり ImageNet-1%（表2）**: ViT-H/16₄₄₈ で **77.3%**。拡張なしの MAE を大きく上回り、拡張ありの MSN/DINO/iBOT を超える。
- **転移（表3）**: CIFAR100 **87.5** / Places205 **58.4** / iNat18 47.6。CIFAR100・Places205 では拡張ありの DINO すら上回る。

### 低レベル・密予測タスク — 対比型を大差で上回る

- **Clevr 物体カウント/深度（表4）**: I-JEPA ViT-H/14 は Clevr/Dist（深度）で **72.4** と、DINO（53.4）・iBOT（62.8）を大きく上回る。物体カウントでも DINO/iBOT 超え。
  - **意義**: 対比型はグローバルな意味に偏り局所情報を捨てがちだが、I-JEPA は局所的な特徴も保持する。「より硬直的でない帰納バイアス」ゆえに広いタスクに使える、という主張の核。

### スケーラビリティ — 本論文最大のセールスポイント

<figure>

![](../../raw/assets/i-jepa/x7.png)

<figcaption>図5: GPU 時間 vs ImageNet-1% 性能。I-JEPA は MAE/data2vec より少ないエポックで収束し、iBOT（拡張あり、複数ビュー処理）より圧倒的に速い。巨大な I-JEPA ViT-H/14 が小さな iBOT ViT-S/16 より少ない計算で済む。</figcaption>
</figure>

- ViT-H/14 を **16 基の A100 × 72 時間未満（< 1200 GPU 時間）** で事前学習。
- iBOT で訓練した ViT-S/16 より **2.5× 速く**、MAE で訓練した ViT-H/14 より **10× 効率的**。
- 表現空間ターゲット計算で 1 反復あたり約 7% 遅くなるが、**約 5× 少ない反復で収束**するため差し引き大幅節約。
- データ・モデル両方向にスケール（表5）: IN1k → IN22k、ViT-H/14 → ViT-G/16 で意味的タスクが改善（ただし ViT-G/16 はパッチが大きく低レベルタスクは伸びない）。

### 予測器は「位置的不確実性」を捉えている

<figure>

![](../../raw/assets/i-jepa/x8.png)

<figcaption>図6: 予測器出力の RCDM デコード可視化。凍結した予測器出力を拡散デコーダ（[[concepts/diffusion-model]]）で画素に戻すと、サンプル間で共通する部分（= 表現に含まれる情報）は正しいポーズの物体部位（鳥の背・車の上部）を示し、変動する部分（= 含まれない情報）は低レベル詳細・背景に対応。</figcaption>
</figure>

- 予測器を凍結し RCDM（拡散モデルベースの表現可視化、Appendix E）でデコードすると、**予測器が位置的不確実性を正しく扱い、高レベルな物体部位を正しいポーズで生成** している様子が見える。低レベル詳細・背景は捨てている。

### アブレーションの要点（第9節 + Appendix C）

- **表現空間 vs 画素空間（表7）**: 損失を画素空間にすると ImageNet-1% で 66.9 → **40.7** に激減。**「表現空間で予測する」ことが I-JEPA の生命線**。
- **multi-block が最良（表6）**: multi-block 54.2 に対し rasterized 15.5 / block 20.2 / random 17.6。
- **ターゲットは大きいほど良い（表8）/ コンテキストは大きいほど良い（表9）/ ターゲット数は多いほど良い（表10、1個 9.0 → 4個 54.2）**。
- **出力マスク > 入力マスク（表11、67.3 vs 56.1）**、**予測器は深い方が良い（表12）**、**予測器は幅にボトルネック（384ch）が良い（表14）**。
- **フル ImageNet ファインチューニング（表15）**: ViT-H/16₄₄₈ で 87.1%。MAE（87.8、1600 ep）に **5.3× 少ないエポックで 1% 未満差**。

---

## 限界・批判的視点

- **DINOv2/iBOT のような最強の凍結特徴量には届かない**: 本論文時点で I-JEPA の線形プロービングは iBOT に「並ぶ」レベルであり、その後 DINOv2（[[entities/dinov2]]）が iBOT 路線を 1B param × 142M 画像にスケールして大きく上回ったため、「凍結特徴量で SSL の頂点」を取ったわけではない。I-JEPA の主張はむしろ **計算効率と低レベルタスクでの汎用性** にある。
- **ViT-G/16 が低レベルタスクで伸びない**: 大きいパッチが局所予測に不利。スケールが万能でない。
- **マスキング設計への敏感さ**: ターゲット/コンテキストのスケール・数に性能が強く依存し（Appendix C）、ハイパラ調整が効く。
- **画像単体・静止画にとどまる**: 本論文は画像のみ。LeCun の世界モデル構想で本命とされる **時間方向の予測** は、後続の V-JEPA（動画）/ V-JEPA 2 に持ち越される。
- **「拡張なしで意味的」だが、EMA ターゲットという非対称性に依存**: 結局 momentum encoder（[[entities/byol]] / [[entities/dino]] と同じ）が崩壊回避に必須で、「設計の単純さ」は相対的なもの。

---

## 用語と略称

- **I-JEPA** = Image-based Joint-Embedding Predictive Architecture（画像版 JEPA）
- **JEPA** = Joint-Embedding Predictive Architecture（結合埋め込み予測アーキテクチャ）。詳細: [[concepts/joint-embedding-predictive-architecture]]
- **JEA** = Joint-Embedding Architecture（結合埋め込みアーキテクチャ、対比型・自己蒸留型の総称）
- **EBM** = Energy-Based Model（エネルギーベースモデル）
- **EMA** = Exponential Moving Average（指数移動平均）。ターゲットエンコーダ更新に使用
- **MIM** = Masked Image Modeling（マスク画像モデリング、[[concepts/masked-image-modeling]]）
- **MAE** = Masked Autoencoder（[[entities/mae]]）
- **SSL** = Self-Supervised Learning（自己教師あり学習、[[concepts/self-supervised-learning]]）
- **ViT** = Vision Transformer（[[concepts/vision-transformer]]）
- **RCDM** = Representation-Conditioned Diffusion Model（表現に条件付けた拡散デコーダ、表現可視化用）
- **CAE** = Context Autoencoders、**data2vec** = Meta のマルチモーダル自己蒸留 SSL（I-JEPA に最も近い先行研究、本 wiki 未取り込み）
- **multi-block マスキング** = 大きめの意味的ターゲット複数 + 情報量の多い疎なコンテキスト 1 個、という I-JEPA 固有のマスク戦略
- **線形プロービング（linear probing）** = 凍結特徴量の上に線形層だけ載せて評価。表現の「素の意味分離性」を測る
- **collapse（崩壊）** = 入力によらず一定の表現を返す自明解。SSL の中心的設計課題

## 関連ページ

- [[concepts/joint-embedding-predictive-architecture]] — 本論文が具体化した JEPA パラダイムの解説（LeCun の世界モデル構想）
- [[entities/i-jepa]] — I-JEPA モデルのスペック・系譜
- [[concepts/self-supervised-learning]] — SSL 全体の中での I-JEPA の位置づけ（対比型・MIM と並ぶ第 3 の系統）
- [[concepts/masked-image-modeling]] — 「再構成」する MIM と、「表現を予測」する I-JEPA の対比
- [[sources/mae]] / [[entities/mae]] — 構造が最も近い対照群（画素再構成 vs 表現予測）
- [[sources/ibot]] / [[entities/ibot]] — 対比型 + MIM ハイブリッドの主要比較対象
- [[sources/dino-emerging-properties-in-self-supervised-vit]] — momentum encoder / 自己蒸留の源流、低レベルタスク比較対象
- [[entities/byol]] — EMA ターゲット + predictor で負例なし学習という共通発想
