---
type: concept
aliases: [SSL, 自己教師あり学習, Self-Supervised Learning]
tags: [paradigm, pretraining, representation-learning]
related: [[knowledge-distillation]], [[vision-transformer]], [[knn-evaluation-protocol]], [[masked-image-modeling]], [[foundation-model]], [[weakly-supervised-pretraining]], [[gram-anchoring]], [[semi-supervised-learning]], [[diffusion-model]]
sources: [[sources/simclr]], [[sources/byol]], [[sources/dino-emerging-properties-in-self-supervised-vit]], [[sources/dinov2-learning-robust-visual-features-without-supervision]], [[sources/dinov3]], [[sources/siglip-2]], [[sources/i-synmed]], [[sources/eva-x]]
updated: 2026-05-28
---

# Self-Supervised Learning（SSL, 自己教師あり学習）

## 一言で

**人間がつけた正解ラベルを使わず、データ自身の構造から「正解」を作り出して学習させる枠組み**。画像なら「同じ画像の 2 つの異なる切り抜きは似た表現になるべき」、テキストなら「文の中の単語を 1 つ隠して当てる」というように、データ内部の関係をタスクに変える。

NLP では **BERT（Bidirectional Encoder Representations from Transformers, 2018）** や **GPT** の大成功で「事前学習はラベルなしが当たり前」という常識を確立した。Computer Vision でもこの流れを輸入する研究が 2018 年頃から本格化し、2020〜2022 年に SimCLR / MoCo / BYOL / SwAV / DINO / MAE などが次々と登場した。

> **補足: なぜラベルなしが嬉しいのか**
> - **データ量の壁を越えられる**: ImageNet（120 万枚にラベル付き）を作るには人手で何ヶ月もかかる。Web には数億〜数兆枚の画像があり、ラベルなしならスケールできる。
> - **「ラベル」が画像の豊かさを潰さない**: 1 枚の画像を「dog」「cat」など 1 単語にすると、テクスチャ・構図・3D 構造といった膨大な情報を捨てている。SSL はこの情報を表現にとどめられる。
> - **下流タスクへの転移性**: 一度学習した表現を、分類・検出・セグメンテーションなど多様な下流タスクにファインチューンや凍結特徴量で再利用できる。

## 大まかな系統

CV の SSL 手法は、自明解（崩壊 = collapse）の回避戦略の違いで大別できる。

### 1. 対比型（contrastive）

「**正例ペア**（同じ画像の 2 つの拡張）を引き寄せ、**負例**（別画像）を引き離す」という対比損失で表現を作る。負例があるため自然に崩壊が防げる。

- **SimCLR** (Chen et al., 2020): 大きなバッチ内の他画像を負例として使う InfoNCE 損失。バッチサイズが性能に直結。
- **MoCo / MoCov2** (He et al., 2019/2020): 負例をキュー（メモリバンク）で保持し、teacher は momentum encoder で更新。

### 2. 非対比型（non-contrastive）

負例を使わず、正例ペアだけで学習する。崩壊を別の仕掛けで防ぐ必要がある。

- **BYOL** (Grill et al., 2020, [[sources/byol]]): student の最後に **predictor**（追加 MLP）を付け、teacher（momentum encoder）の出力に MSE で近づける。predictor + EMA ターゲットの組み合わせが必須で崩壊を防ぐ。負例なしで SimCLR を 5 pt 上回り（74.3% vs 69.3%）、「負例は必須か」という問いに「否」と答えた先駆的論文。
- **SimSiam** (Chen & He, 2021): BYOL から momentum を取り、stop-gradient + predictor だけで動かす。
- **DINO** (Caron et al., 2021): cross entropy + centering + sharpening + momentum で崩壊を防ぐ。詳細: [[sources/dino-emerging-properties-in-self-supervised-vit]] および [[entities/dino]]。

### 3. クラスタリング型

特徴量を擬似クラスタに割り当て、その割当をラベル代わりにする。

- **DeepCluster, SeLa, SwAV** (Caron et al.): SwAV はオンラインでクラスタ割当を Sinkhorn-Knopp で均等化することで崩壊を防ぐ。

### 4. マスク再構成型（masked image modeling, MIM）

BERT の流儀を画像に持ち込む系統。入力の一部を隠し、隠した部分を再構成または予測させる。ViT との親和性が高い。詳細: [[masked-image-modeling]]。**理論的祖先は [[denoising-autoencoder]]**（DAE, Vincent 2008）。

- **BEiT** (Bao et al., 2021): 事前学習済み dVAE が出力する離散トークンを予測。
- **MAE** (He et al., 2021, [[entities/mae]]): **画素を 75% マスク**し、軽量 decoder で再構成。非対称設計で大規模化を実現、CV MIM の中核手法に。
- **SimMIM** (Xie et al., 2022): MAE の単純化版。
- **iBOT** (Zhou et al., 2022, [[entities/ibot]]): **DINO + MIM のハイブリッド**。マスクされたパッチを画素ではなく teacher 出力で予測（"online tokenizer"）。
- **DINOv2** (Oquab et al., 2023, [[entities/dinov2]]): iBOT を 1B param × 142M 画像に scale、汎用基盤モデルへ。

### 5. ハイブリッド型（識別 + MIM）

iBOT が代表。「**画像レベル目的（DINO 由来）+ パッチレベル MIM 目的**」を同時に最適化。識別型の凍結特徴量の良さと MIM の密予測性能を両立する。**DINOv2 はこの路線を完成させ、CV における基盤モデル化を達成**。さらに **DINOv3 (2025) が ViT-7B × LVD-1689M に scale し、Gram anchoring（[[gram-anchoring]]）で長時間学習による dense feature 劣化問題を解決**、PE/SigLIP 2 を dense タスクで圧倒した。

> **2025 年の逆流: WSL が SSL 技法を借用** — [[sources/siglip-2]] は CLIP/SigLIP 系統（WSL）でありながら、**SILC（Naeem et al., ECCV 2024）の local-to-global 自己蒸留** と **TIPS（Maninis et al., ICLR 2025）のマスク予測** を借用して dense feature を強化。これは SSL → WSL への技法逆流の代表例。「対比 vs 自己蒸留 vs MIM」という区分は、2025 年時点で **実用モデルでは融合** している（SigLIP 2 は対比＋自己蒸留＋MIM＋decoder＋蒸留の 5 系統融合）。

DINO はこの分類では「非対比型」に属し、知識蒸留の枠組みで再解釈した点に独自性がある。iBOT/DINOv2/DINOv3 はそれを更にハイブリッド化したもの。

## SSL の基本構造（多くの手法に共通）

```
入力画像 x
   ├── データ拡張 view1 ─→ encoder f_s ─→ projection h_s ─→ z1
   └── データ拡張 view2 ─→ encoder f_t ─→ projection h_t ─→ z2
                                     ↑
                              比較損失 L(z1, z2)
```

- **エンコーダ** $f$: ResNet や ViT などのバックボーン。**実際に下流で使う特徴量はここから取り出す**（projection head は捨てる）。
- **projection head** $h$: 通常 2〜3 層の MLP。表現学習を助けるが下流では使わない（DINO の場合は $\ell_2$ 正規化 + 重み正規化付き全結合層で末尾を構成）。
- **データ拡張**: random resize crop, color jittering, Gaussian blur, solarization 等。同じ画像の異なる「見え方」を作ることで、不変性を仕込む。

## 重要な構成要素（DINO 文脈で頻出するもの）

### momentum encoder

**student の重みを EMA（指数移動平均）で滑らかに追従するもう一方のエンコーダ**。

$$\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$$

- $\lambda$ は 0.99 〜 1 に近い値（更新が遅い ＝ teacher が安定する）。
- もともと MoCo で「対比学習における negative queue の代替」として導入されたが、BYOL/DINO では「**訓練ダイナミクスを安定化し、student に滑らかなターゲットを与える**」役割を果たす。
- DINO 論文では「これは指数減衰付き Polyak-Ruppert 平均によるモデルアンサンブルとみなせる」と解釈される。teacher は実際、訓練を通して student を一貫して上回る。

### multi-crop augmentation

SwAV で導入されたデータ拡張戦略。

- 通常: 1 画像 → 2 つの大きな crop（例 $224^2$）→ 互いに近づける
- multi-crop: 1 画像 → **2 つの「global」crop（$224^2$, 元画像の 50% 以上を覆う）** + **数本の「local」crop（$96^2$, 50% 未満）** を作る。
- student には全 crop、teacher には global crop だけを流す。**「局所 → 大域」の対応**を表現に強制でき、性能・収束速度ともに大きく改善する。
- 計算コストとメモリは増えるが、それ以上のリターンがある（DINO §5.4）。

### 崩壊（collapse）

**「全ての入力に対し同じ表現を出せば損失が最小化される」という自明解にモデルが落ちること**。SSL では避けられない設計課題。

- 一定値への崩壊 → 何の情報も保持していない
- 一様分布への崩壊（softmax 出力の場合）→ 区別能力ゼロ
- 防ぎ方は手法依存: contrastive 損失 / clustering 正規化 / predictor + BN / centering + sharpening 等

> **補足: なぜ「崩壊」が起きるのか** — 「同じ画像の 2 つの拡張を近づけよ」という制約は、すべての画像を同じ点に写せば最小化される。これを防ぐ「斥力（repulsion）」を何らかの形で組み込むか、学習ダイナミクスの非対称性（teacher と student の差）で間接的に防ぐ必要がある。

## SSL 表現を評価する代表的プロトコル

- **線形分類（linear probing）**: 凍結特徴量の上に線形分類器だけ載せて教師あり学習。表現の「線形分離可能性」を測る。
- **ファインチューニング**: 事前学習重みで初期化し全体を再学習。最終的な下流性能を測る。
- **k-NN 分類**: 凍結特徴量の最近傍で投票。最もハイパラに頑健で、表現の「素の品質」を測れる。詳細: [[concepts/knn-evaluation-protocol]]。
- **転移学習**: 別ドメインのデータセットで上記を行う。汎化性能を測る。
- **密予測タスクの線形プローブ**: セグメンテーション（ADE20K, Cityscapes, Pascal VOC）や深度推定（NYUd, KITTI）。**パッチレベル特徴量の品質**を測る。DINOv2 以降は重要視されるようになった。例えば DINOv2 ViT-g/14 は凍結 + 線形だけで ADE20K mIoU 49.0 を達成し、MAE のフル fine-tune に匹敵する。
- **インスタンス検索（image retrieval）**: Oxford / Paris / GLDv2。「**意味的に近い画像が特徴量空間でも近い**」かを測る。DINOv2 ViT-L が Oxford-Hard で mAP 54.0 と圧倒。
- **ゼロショット / few-shot 評価**: CLIP 系で標準。SSL でも近年テキストエンコーダと aligning することで適用される。

## 参考

- [[concepts/semi-supervised-learning]]: 半教師あり学習（少量ラベルあき + 大量ラベルなし）— 当 wiki の SSL（自己教師あり学習）とは別概念。文献によっては "SSL" と同じ略称を使うため注意
- [[concepts/diffusion-model]]: 拡散モデル（DDPM, Stable Diffusion 等）。合成データ生成で SSL 事前学習を補完する応用が登場
- [[sources/i-synmed]] / [[entities/i-synmed]]: DDPM 合成画像で DINO を事前学習する医療応用例（IEEE Access 2025）
- [[sources/eva-x]] / [[entities/eva-x]]: 胸部 X 線専用基盤モデル。EVA-02 系統（凍結 CLIP トークナイザ × MIM）の医療版（npj Digital Medicine 2025）
- [[sources/byol]]: BYOL 論文（「負例なし SSL」の先駆け、非対比型の代表）
- [[sources/dino-emerging-properties-in-self-supervised-vit]]: DINO 論文（本概念の主要文献の一つ）
- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: DINOv2 論文（SSL がスケールで基盤モデルになり得ることを実証）
- [[sources/dinov3]]: DINOv3 論文（SSL が dense タスクで弱教師ありを圧倒できることを実証、Gram anchoring 導入）
- [[concepts/vision-transformer]]: SSL の主要ターゲットアーキテクチャ
- [[concepts/knowledge-distillation]]: 多くの SSL 手法（BYOL, DINO 等）は自己蒸留として再解釈できる
- [[concepts/masked-image-modeling]]: MIM の系統解説
- [[concepts/foundation-model]]: SSL がスケールで到達する目的地
- [[concepts/weakly-supervised-pretraining]]: SSL と対比される CLIP 系のアプローチ
- [[concepts/gram-anchoring]]: SSL の長時間学習問題への解決法（DINOv3 提案）
