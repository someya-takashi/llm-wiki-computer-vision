---
type: concept
aliases: [MIM, Masked Image Modeling, マスク画像モデリング]
tags: [paradigm, pretraining, ssl, transformer]
related: ["[[self-supervised-learning]]", "[[vision-transformer]]", "[[denoising-autoencoder]]", "[[online-tokenizer]]"]
sources: ["[[sources/dinov2-learning-robust-visual-features-without-supervision]]", "[[sources/mae]]", "[[sources/ibot]]", "[[sources/siglip-2]]"]
updated: 2026-05-27
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

### MAE（He et al., 2021）— MIM を主流化した最重要論文

**Masked Autoencoder**。詳細: [[entities/mae]] / [[sources/mae]]。MIM を一気に主流化した名作で、後の iBOT/DINOv2/DINOv3 の MIM 損失の直接の源流。

**3 つの核心設計**:

1. **画素を 75% マスク**（高マスク率が肝）。BERT の 15% や先行 CV MIM の 20-50% より圧倒的に高い
2. **非対称 encoder-decoder**: encoder は可視 25% のパッチのみ処理（mask token を入力しない）。軽量 decoder が encoder 出力 + mask token から画素再構成
3. **画素を直接再構成**（パッチ内正規化）。BEiT のような dVAE トークン化は不要

**コアアブレーション結果**（[[sources/mae]] 表 1 より）:

| 設定 | fine-tune | linear |
|---|---|---|
| mask 比率 75%（既定） | 84.9 | 73.5 |
| mask 比率 15% | 大幅低下 | 大幅低下 |
| encoder に `[M]` を入れる | 84.2 | **59.6**（-14%） |
| decoder 深さ 1 ブロック | 84.8 | **65.5**（-8%） |
| ターゲット = dVAE token | 85.3 | 71.6（pixel 73.5 より低い） |

> **補足: なぜ 75% という極端なマスク率が効くのか** — MAE 論文の重要な観察は「画像は冗長性が高いので、軽くマスクしても周りの情報からほぼ補間できてしまい、学習信号が弱くなる」というもの。75% という攻めたマスク率にすることで、モデルは「文脈から大きく欠けた部分の意味」を推論せざるを得なくなる。NLP の 15% マスク率（BERT）よりはるかに高い。

> **補足: なぜ encoder に mask token を入れないと良いか** — 推論時の入力（完全画像）と訓練時の入力（mask token 混じり）に分布ギャップがあると linear probing が崩壊する。mask token を decoder 側に押し付けることで、encoder は常に実パッチだけを見る。**おまけに計算量が 4 分の 1 になる**ので大規模化が現実的になる。

**MAE の特徴と限界**:

- **Fine-tuning は強い**: ViT-H で IN1K のみ 87.8% top-1（当時の SOTA）
- **Linear probing は弱い**: ViT-L で 75.8%（対比学習 MoCo v3 ViT-L 77.6% に負ける）
- **「Partial fine-tuning」という新評価**: 1 ブロックだけ fine-tune すると 73.5% → 81.0% に跳ね上がる。MAE 表現は「**線形分離性は低いが非線形に強い**」
- **教師あり事前学習を密予測で大きく上回る**: COCO 検出 ViT-L で MAE 53.3 vs supervised 49.3（+4.0 AP^box）
- **Augmentation 不要**: random crop だけで動く（対比学習が color jitter 等に依存するのと対照的）

### SimMIM（Xie et al., 2022）

MAE をさらに単純化。tokenizer も lightweight decoder も使わず、ViT 全体に `[MASK]` トークンを通して画素を予測。

### iBOT（Zhou et al., 2021 / ICLR 2022）— MIM と self-distillation の統合

**[[entities/ibot]] = DINO + MIM with online tokenizer**。詳細: [[sources/ibot]] / [[entities/ibot]]。

iBOT の貢献は、MIM を**知識蒸留の枠組みで再解釈**したこと：

> BEiT の MIM 損失と DINO の self-distillation 損失は数式上**同じ形**（クロスエントロピーで分布を一致させる）。違うのは「teacher が何か」だけ。

これに基づき、**teacher を事前訓練 dVAE ではなく EMA で動的に更新される student のコピーにする**ことで、両者を単一の枠組みに統合した。これが [[concepts/online-tokenizer]]。

**2 つの損失の同時最適化**:

1. **[CLS] レベル損失（DINO 由来）**: 同じ画像の 2 ビューを student と teacher に通し、[CLS] トークン出力分布を一致させる
2. **パッチレベル MIM 損失（iBOT の新規）**: student のマスクされたパッチ出力を、teacher の対応する**非マスク**パッチ出力に近づける

**重要な実装ポイント**:

- **ブロック単位マスキング**（BEiT 由来）: ランダム単独パッチではなく、隣接パッチをまとめてマスク
- **予測比率はランダム**: 確率 0.5 で 0（DINO として動作）、0.5 で [0.1, 0.5]（MIM）。これでマスク有無の分布ギャップを吸収
- **[CLS] とパッチで射影ヘッドを共有**: [CLS] で獲得した意味性をパッチ MIM に転移
- **ソフトラベル使用**（hardmax にすると大幅低下）

**結果**:
- ImageNet-1K linear probing 81.0%（ViT-L）/ 82.3%（ViT-L, IN-22K 事前学習）
- COCO 検出 51.2 AP^b、ADE20K linear セグメンテーション 38.3 mIoU（BEiT の 27.4 を大きく超え）
- **創発する部位レベル意味性**: パッチトークンに「犬の耳」「車のヘッドライト」が教師なしで現れる（BEiT には現れない）

これにより MAE の「凍結特徴量が弱い」問題と DINO の「パッチレベル表現が弱い」問題の**両方を解決**。**後の DINOv2/DINOv3 の損失関数の直接の源**となった。

### DINOv2 における MIM

[[entities/dinov2]] は iBOT の MIM 損失をそのまま継承。アブレーション（[[sources/dinov2-learning-robust-visual-features-without-supervision]] §6.4）で「MIM 損失を外すと **ADE20k mIoU が -3% 落ちる**」と示し、**密予測性能の鍵**であることを示した。

### EVA / EVA-02 / EVA-X 系統：凍結 CLIP トークナイザによる MIM

**EVA-02**（Fang et al., 2024）は MIM の新潮流を確立した。「**EVA-CLIP を凍結トークナイザとして使う MIM**」というレシピで、自然画像の強力な基盤モデルを構築。

[[entities/eva-x]]（npj Digital Medicine 2025）はこの設計を**胸部 X 線医療画像**に適用：

- 学習可能 ViT + 凍結 EVA-CLIP（または MGCA-CLIP）の dual ViT
- マスク比 0.3、コサイン類似度損失
- Merged-520K（NIH+CheXpert+MIMIC）で事前学習
- 11 下流タスクで SOTA、EVA-X-Ti (6M) が 13× FLOPs の MGCA-B を上回る効率性

これは [[entities/ibot]] の「online tokenizer」と対照的な「**frozen external tokenizer**」の路線。両者の対比は [[concepts/online-tokenizer]] で詳述。MIM 設計の重要な分岐点。

### SigLIP 2 における MIM（WSL への流入）

[[sources/siglip-2]]（Tschannen et al., 2025）は WSL モデル（CLIP/SigLIP 系統）でありながら MIM を借用：

- 訓練の **最後の 20%** で、TIPS（Maninis et al., ICLR 2025）由来のマスク予測損失を追加
- パッチの **50% をマスク** し、student の対応位置の特徴を teacher の非マスク特徴にマッチさせる
- これにより SigLIP v1 比で **ADE20k seg +4.2pt, PASCAL seg +5.1pt** の dense 性能改善

つまり「**MIM は今や SSL 専売特許ではなく、WSL の dense 性能向上のための標準ツール**」になっている。CLIP/SigLIP のような対比学習だけでは捉えきれない局所的意味性を、MIM が補う役割。

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

MIM の発想は実は古く、**Stacked Denoising Autoencoder**（Vincent et al., 2008、詳細: [[concepts/denoising-autoencoder]]）が「入力にノイズを加えて元を復元させる」自己教師あり学習を提唱していた。MIM はこれを Transformer 時代に再発明したものとも言える。MAE 論文も §2 で「MAE はノイズ除去オートエンコーディングの一形態」と明示している。

NLP 側の系譜:
- **word2vec**（Mikolov, 2013）の skip-gram = 単語の文脈予測
- **BERT**（Devlin et al., 2018）の MLM = マスクされた単語の予測

CV 側の系譜:
- **Context Prediction**（Doersch et al., 2015）= パッチ間の相対位置予測
- **Inpainting**（Pathak et al., 2016）= 矩形領域の補完
- **Jigsaw Puzzle**（Noroozi et al., 2016）= シャッフルされたパッチの並べ替え
- → ViT 時代に **BEiT → MAE → SimMIM → iBOT → MaskFeat → ...** と一気に発展

## 関連ページ

- [[sources/mae]] / [[entities/mae]] — CV MIM の中核手法、本概念の最重要文献
- [[sources/ibot]] / [[entities/ibot]] — MIM と self-distillation を online tokenizer で統合
- [[concepts/online-tokenizer]] — iBOT の核心、MIM のトークナイザ問題を動的に解決
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] — iBOT を 1B param × 142M 画像に scale
- [[concepts/denoising-autoencoder]] — MIM の理論的祖先（DAE）
- [[concepts/self-supervised-learning]] — SSL 全般の中での位置づけ
- [[concepts/vision-transformer]] — MIM の主要ターゲットアーキテクチャ
- [[sources/eva-x]] / [[entities/eva-x]] — EVA-02 系統の医療版（npj Digital Medicine 2025）。凍結 EVA-CLIP トークナイザ × MIM で胸部 X 線基盤モデル構築
- [[sources/i-synmed]] / [[entities/i-synmed]] — 対比される医療 X 線 SSL（DDPM 合成 + DINO、MIM ではない）
