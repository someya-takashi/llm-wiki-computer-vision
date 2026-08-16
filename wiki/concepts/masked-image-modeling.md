---
type: concept
aliases: [MIM, Masked Image Modeling, マスク画像モデリング]
tags: [paradigm, pretraining, ssl, transformer]
related: ["[[self-supervised-learning]]", "[[vision-transformer]]", "[[denoising-autoencoder]]", "[[online-tokenizer]]", "[[joint-embedding-predictive-architecture]]", "[[convolutional-neural-network]]"]
sources: ["[[sources/dinov2-learning-robust-visual-features-without-supervision]]", "[[sources/mae]]", "[[sources/ibot]]", "[[sources/siglip-2]]", "[[sources/i-jepa]]", "[[sources/convnext-v2]]"]
updated: 2026-06-17
---

# Masked Image Modeling（MIM, マスク画像モデリング）

## 一言で

**画像の一部のパッチを隠し、隠した部分を残りから予測する**ことで表現を学習する自己教師あり学習の系統。NLP の **BERT** が「文中の単語を隠して当てる masked language modeling（MLM）」で大成功したアイデアを、**画像パッチに転用**したもの。Vision Transformer（[[concepts/vision-transformer]]）の登場で「画像 = パッチの系列」という統一的な見方が可能になり、MIM が急速に発展した。

## なぜ ViT 時代に流行ったのか

- **CNN ではマスキングが扱いにくい**: 畳み込みは局所的な計算で、「あるピクセルを隠す」と周囲の畳み込みカーネルが意味を成さない。より正確には、**密なスライディングウィンドウがマスク領域を跨いで計算してしまうため、モデルが「隠された部分を周囲からコピーしてくる」ショートカットを学習でき、学習信号が壊れる**（情報漏洩）。
- **ViT は系列として処理する**: トークンを 1 つ取り除いたり、`[MASK]` トークンに置き換えたりすることが NLP の BERT と同じ感覚で自然にできる。
- **画素レベル予測との接続**: MIM は本質的に密予測なので、セグメンテーション・深度推定など pixel-level の下流タスクと相性が良い。

> **ただしこの「CNN では無理」は 2023 年に覆される** — [[sources/convnext-v2|ConvNeXt V2]] が **疎畳み込み（sparse convolution）で可視画素のみを処理する**ことで情報漏洩を遮断し、CNN でも MIM が効くことを実証した（後述）。上の制約は「畳み込みの本質的限界」ではなく「素朴な実装の問題」だった。

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

### FCMAE / ConvNeXt V2（Woo et al., CVPR 2023）— MIM を ConvNet へ持ち込む

**[[entities/convnext-v2]] = 疎畳み込み MIM + GRN の共設計**。詳細: [[sources/convnext-v2]]。

MIM は長らく「ViT 専用の技法」とみなされていた。**ConvNeXt V2 はこれを CNN 側に開いた最初の成功例**である。

**なぜ CNN で MIM が難しかったか**: MAE の効率は「エンコーダが可視パッチだけを処理する」非対称設計に由来するが、CNN は密なスライディングウィンドウで動くのでマスク領域を跨いで計算してしまい、**モデルがマスク部分を周囲からコピーするショートカットを学習してしまう**。入力側に学習可能なマスクトークンを置く素朴な解（SimMIM 流）は、事前学習効率を下げ、テスト時にマスクトークンが存在しないため訓練・テストの分布不整合も生む。

**FCMAE の解法**: **マスク画像を「2D の疎な画素配列」とみなし、submanifold sparse convolution で可視データ点のみを処理する**（3D 点群処理からの着想）。ファインチューニング時は特別な扱いなしに標準の密畳み込みへ戻せる。

- **この 1 点の効果が決定的**: 疎畳み込みなし **79.3** → あり **83.7**（+4.4）
- デコーダは**単一の ConvNeXt ブロック**（次元 512）。UNet 比 1.7× 高速で精度同等
- **マスク率 0.6**（MAE の 75% より低い）、マスク単位 32×32
- 副産物として**スループット 1.3×、メモリ 1/2**

**ただし FCMAE だけでは足りなかった**——ここが本論文の肝である。V1 に FCMAE を適用しても 83.7 で、教師あり 300ep の 83.8 に届かない。原因は **特徴崩壊**（チャネル間で活性化が冗長になり、死んだ／飽和したチャネルが増える現象）で、主に次元拡大 MLP 層で起きていた。これを **GRN（Global Response Normalization）** で解決する。

**GRN**: 各チャネルを空間方向の L2 ノルムで集約 → 除法正規化 $||X_i||/\sum_j||X_j||$ で**相対的重要度**を出す → 元の応答に掛け戻す。ゼロ初期化の $\gamma,\beta$ と残差接続を伴い、**実装 3 行・追加パラメータ実質ゼロ**。生物の側方抑制に着想。

**co-design（共設計）が主張の中核**:

| | IN-1K ft（Base） |
|---|---|
| V1 + 教師あり 300ep | 83.8 |
| V1 + FCMAE（枠組みだけ） | 83.7 |
| V2 + 教師あり（GRN だけ） | 84.3 |
| **V2 + FCMAE（両方）** | **84.6**（1600ep で 84.9） |

**枠組みだけでもアーキテクチャだけでも効かず、両方揃って初めて効く**。「自己教師あり手法は既存アーキテクチャに後から載せるもの」という業界の慣行への異議申し立てになっている。**GRN は事前学習と FT の両方に入っていないと壊滅する**（片方だけだと 78.8 / 80.6）という実験が、この不可分性を最も直接的に示す。

**MIM 手法としての立ち位置**（同一条件の IN-1K 事前学習）:

- **SimMIM（Swin）は全サイズで上回る**
- **MAE（ViT）は Large まで互角**（198M で 85.8 vs ViT-L 307M で 85.9）
- **Huge では負ける**（86.3 vs ViT-H 86.9）
- **MoCo v3（対比学習）には勝つ**（84.9 vs 83.7）。「MIM > 対比学習」という ViT 側の知見が ConvNet でも成立

> **MIM 史における意味** — [[entities/hiera]] が「MAE 互換であること」を設計目標に階層型 ViT を作り直したのと同じ *co-design* の発想を、ConvNet 側で実行したもの。**MIM がアーキテクチャを選ぶのではなく、アーキテクチャを MIM に合わせて直せばよい**という方向転換を示した。

### I-JEPA（Assran et al., CVPR 2023）— MIM を「表現空間予測」へ拡張

**[[entities/i-jepa]] = マスクして *画素ではなく表現* を予測**。詳細: [[sources/i-jepa]] / [[concepts/joint-embedding-predictive-architecture]]。

厳密には I-JEPA は MIM（再構成型）ではなく **JEPA（予測型）** だが、「画像をマスクして欠落部を埋める」という骨格を MIM と共有するため、MIM の発展形として理解すると分かりやすい。

- **MAE との決定的な違い**: MAE が「マスク → 軽量デコーダで**画素を再構成**」なのに対し、I-JEPA は「マスク → 予測器で**ターゲットエンコーダ出力（表現）を予測**」。再構成先が画素か表現かが分岐点。
- **なぜ表現予測か**: 画素再構成は予測不能な低レベル詳細（背景・テクスチャ）まで当てさせるため意味性が下がる。表現空間なら、ターゲットエンコーダが無関係な詳細を捨てた抽象ターゲットを作るので、予測器は意味構造だけを学べる。**損失を画素空間に戻すと ImageNet-1% が 66.9 → 40.7 に激減**（[[sources/i-jepa]] 表7）し、この設計の重要性が裏づけられる。
- **崩壊回避**: MIM は再構成ターゲットがあるので崩壊しないが、I-JEPA は表現を予測するため JEA 同様に崩壊しうる。EMA ターゲットエンコーダ（[[entities/byol]] / [[entities/dino]] の momentum）で回避する。
- **結果**: 凍結特徴量で MAE を上回り、Clevr 深度などの低レベル・密予測で DINO/iBOT を大差で上回りつつ、iBOT 比 2.5×・MAE 比 10× の計算効率。

> **MIM 3 系統の整理**: ①画素再構成（MAE/SimMIM）、②離散/外部トークン予測（BEiT / EVA / online tokenizer の iBOT）、③**表現予測（I-JEPA）**。①②が「決まったターゲット（画素・固定トークン）」を当てるのに対し、③は「EMA で共進化する自分自身の表現」を当てる点で質的に異なる。

## MIM vs 他の SSL 系統

| | MIM (BEiT, MAE, iBOT) | 識別型 (SimCLR, MoCo, BYOL, DINO) |
|---|---|---|
| 信号源 | 画像内のマスク予測 | 画像間/拡張間の関係 |
| 強み | 密予測タスク、fine-tune 性能 | 凍結特徴量、k-NN、検索 |
| 弱み | 凍結特徴量が弱い（MAE）、k-NN が弱い | パッチレベル表現がやや弱い |
| 必要なデータ拡張 | ほぼ不要 | crop, color jitter 等が肝 |
| 崩壊回避 | 不要（再構成ターゲットがある） | 必須（[[concepts/self-supervised-learning]] 参照） |

iBOT/DINOv2 は**ハイブリッド**で、両方の長所を取りに行く設計。これが「凍結特徴量が線形分類でも密予測でも強い」DINOv2 の性能の源泉。

## MIM とアーキテクチャの相性

MIM は「どのバックボーンでも動く」わけではなく、**マスクをどう表現するか**でアーキテクチャを選ぶ。

| バックボーン | マスクの扱い | MIM 適性 | 代表 |
|---|---|---|---|
| **plain ViT** | トークンを列から取り除く | ◎ 最も自然 | [[entities/mae]] |
| **階層型 ViT（窓 attention）** | 窓構造がマスクと干渉 | △ 工夫が要る | Swin + SimMIM |
| **階層型 ViT（pool attention）** | MAE 互換を設計目標に据えた | ◎ | [[entities/hiera]] |
| **ConvNet（素朴）** | 密な畳み込みがマスクを跨ぐ → 情報漏洩 | ✗ | ConvNeXt V1 + MAE（83.7、教師あり以下） |
| **ConvNet（疎畳み込み）** | 可視画素のみ演算し漏洩を遮断 | ◎ | [[entities/convnext-v2]] + FCMAE |

**この表の含意**: [[entities/hiera]] と [[entities/convnext-v2]] はどちらも「**MIM に合わせてアーキテクチャを直す**」という同じ発想（co-design）に立っている。MIM が普及した結果、**アーキテクチャ設計の評価軸に「MIM と組めるか」が加わった**というのが 2022-2023 年の重要な変化。

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
- [[sources/i-jepa]] / [[entities/i-jepa]] — MIM の「再構成」を「表現予測」に置き換えた JEPA 系。MAE の直接の対照群
- [[concepts/joint-embedding-predictive-architecture]] — I-JEPA が確立した「表現空間予測」パラダイムの解説
- [[sources/convnext-v2]] / [[entities/convnext-v2]] — 疎畳み込み（FCMAE）+ GRN で **MIM を ConvNet へ開いた**研究
- [[concepts/convolutional-neural-network]] — CNN 側の背景。BN がマスク入力と相性が悪い理由など
- [[entities/hiera]] — 「MAE 互換であること」を設計目標に据えた階層型 ViT。ConvNeXt V2 と同じ co-design
