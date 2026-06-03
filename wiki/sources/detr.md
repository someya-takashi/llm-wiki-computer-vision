---
type: source
source_path: raw/papers/End-to-End Object Detection with Transformers.md
source_kind: paper
title: "End-to-End Object Detection with Transformers"
authors: [Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko]
year: 2020
venue: ECCV 2020
ingested: 2026-05-28
tags: [detr, object-detection, transformer, bipartite-matching, hungarian-loss, set-prediction, panoptic-segmentation, end-to-end, facebook-ai]
translation: "[[translations/detr]]"
---

# DETR: Transformer による end-to-end 物体検出

> 原典: [[translations/detr]] ・ `raw/papers/End-to-End Object Detection with Transformers.md`
> 著者: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko（Facebook AI Research, Paris）
> 出典: ECCV 2020 / arXiv 2005.12872
> arXiv: <https://arxiv.org/abs/2005.12872>
> コード: <https://github.com/facebookresearch/detr>

## 一言まとめ

**「物体検出を `集合予測（set prediction）問題` として再定義し、Hungarian アルゴリズムによる二部マッチング + Transformer のグローバル attention + 並列デコーディング **だけ** で、NMS（非最大抑制）・anchor・proposal という従来の hand-designed components をすべて排除した最初の本格的成功例」**。COCO で Faster R-CNN と同等の 42 AP を達成しつつ、**大物体で +7.8 AP** という顕著な改善を示した。**Transformer を物体検出に持ち込んだ転換点** であり、Deformable DETR / DINO / DETA / CoDETR / MDETR / OWL-DETR など **DETR ファミリー** という研究系統を切り拓いた。

## 背景と問題意識

2020 年時点の物体検出の主流：

| パラダイム | 代表手法 | 特徴 |
|---|---|---|
| **Two-stage** | Faster R-CNN, Mask R-CNN | **Region proposals** をまず生成し、各 proposal を分類・回帰 |
| **Single-stage** | YOLO, SSD, RetinaNet | 画像全体の **anchor グリッド** 上で直接予測 |
| **Anchor-free** | CenterNet, FCOS | 物体中心・キーポイント検出ベース |

これらすべてに共通する問題：

1. **大量の hand-designed components**: anchor 設計、proposal 生成、NMS（非最大抑制）、ground truth → anchor 割り当てヒューリスティック
2. **間接的な集合予測**: 物体検出は本質的に「**集合**」を返す問題なのに、すべての手法は「ボックス回帰 + 分類」という代理タスクに分解して解いていた
3. **重複予測の事後処理**: NMS で近接する重複を削除するという、深層学習の "end-to-end" 哲学に反する後処理が必須

> **補足: NMS（非最大抑制）とは** — 同じ物体について複数の検出ボックスが出てきたとき、IoU が高い（重なりが大きい）ペアでは confidence が低い方を捨てる、というアルゴリズム。シンプルだが、(1) 閾値ハイパラに敏感、(2) 重なった真の異なる物体（人混みなど）も誤って削除する、(3) GPU で並列化しにくい、という弱点がある。

機械翻訳・音声認識・テキスト生成では既に **end-to-end 化** が確立していた（Transformer + 並列デコーディング）。物体検出でも同じことができるはず — それが DETR の問題意識。

## 提案手法 / 主張

### 全体アーキテクチャ

```
画像 → CNN backbone → flatten + positional encoding
         ↓
    Transformer encoder（self-attention でグローバルにシーン理解、特に「インスタンスの分離」）
         ↓
    Transformer decoder（N=100 個の object queries が並列に attend）
         ↓
    各 query 出力を共有 FFN に通す → (class, bbox) を独立予測
         ↓
    Hungarian 二部マッチング損失で訓練（ground truth との一意な対応を強制）
```

3 つの構成要素：

1. **CNN backbone**（ResNet-50 / ResNet-101）: 画像 → コンパクトな特徴マップ（典型的に H/32 × W/32 × 2048）
2. **Transformer encoder-decoder**: 6 層 encoder + 6 層 decoder、幅 256、8 ヘッド attention
3. **共有 FFN（3 層 MLP）**: 各 decoder 出力から (class, normalized bbox) を予測。$\varnothing$（物体なし）クラスが背景の役割を果たす

### 鍵 1: 集合予測損失（二部マッチング + Hungarian loss）

DETR は固定数 $N$（=100）の予測を出力する。これを ground truth とどう対応付けるか？

**Hungarian アルゴリズムによる二部マッチング**:

$$
\hat{\sigma}=\operatorname*{arg\,min}_{\sigma\in\mathfrak{S}_{N}}\sum_{i}^{N}{\cal L}_{\rm match}(y_{i},\hat{y}_{\sigma(i)})
$$

- マッチングコスト ${\cal L}_{\rm match}$ は「**クラス確率 + ボックス類似度**」の組み合わせ
- 各 ground truth は **ちょうど 1 つの予測** と対応付けられる（**1 対 1 マッチング**）
- これにより、**重複予測は構造的に許されない**（同じ物体に複数のスロットが対応することはできない）

→ **NMS が不要になる根本理由**: 損失関数が「1 つの物体には 1 つのスロット」を訓練時から強制するため、推論時に重複は自然に出ない。

**ボックス損失**: $\ell_1$ + **generalized IoU（GIoU）損失** の線形和。Anchor 不要のため絶対座標で予測することによる「小ボックスと大ボックスでスケールが違う」問題を、スケール不変な GIoU で緩和。

> **補足: なぜ Hungarian アルゴリズムか** — N 個の予測 × N 個の ground truth の最適な対応付けを総当たり的に解くと N! 通りで爆発するが、Hungarian アルゴリズムは O(N³) で最適解を見つける。N=100 程度なら GPU 上でも実用的。

### 鍵 2: Transformer のグローバル attention

**Encoder の役割**: self-attention で **インスタンスを分離**。Encoder attention map の可視化（図 3）で、画像内の各物体が encoder の時点で既に空間的に分離されていることが見える。

**Decoder の役割**: 各 object query が **物体の末端（頭・脚など）に局所的に attend**（図 6）。「encoder が物体を分離し、decoder が境界を抽出」という分業構造。

**Encoder の重要性**: 表 2 で encoder 層を 0 にすると AP が **-3.9 ポイント**（特に大物体で -6.0 AP）。グローバル推論がなくても decoder だけで動くが、性能は大きく落ちる。

### 鍵 3: Object queries（学習可能な位置埋め込み）

100 個の object queries は **学習可能な位置埋め込み** で、訓練中に **異なる「役割」に特化** する：

- スロットによっては「画像の左下の小さな物体」担当
- スロットによっては「画像全体を覆う大きな水平ボックス」担当
- ほぼすべてのスロットが「画像全体ボックス」モードも持つ（COCO の物体分布を反映）

これは図 7 で可視化されている。「**スロットの自然な特化**」は、attention の力で hand-designed なクラスごとの anchor が不要になる仕組みの実証。

### 鍵 4: 並列デコーディング

Transformer による系列生成は、機械翻訳など多くの応用で **autoregressive（1 トークンずつ生成）** だが、DETR は **並列に N=100 個の予測を一度に出力**:
- 集合予測には順序が不要（permutation invariant）
- 並列化により高速・バッチ化が容易
- すべての物体について **グローバルにペアワイズ関係を考慮** できる（同じ物体への重複予測を抑制）

### 鍵 5: パノプティックセグメンテーションへの拡張

DETR を訓練後、decoder 出力の上に **マスクヘッド**（multi-head attention で encoder 出力に再 attend し、FPN で解像度復元）を追加するだけで、**パノプティックセグメンテーション** に拡張できる。Things（人、車）と stuff（空、道路）を **統一的に** 扱える初の手法。COCO で 46 PQ を達成し、UPSNet / PanopticFPN を上回る。

## 実験結果と知見

### COCO 物体検出（表 1）

| Model | #params | AP | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|
| Faster RCNN-FPN+ (9× schedule) | 42M | 42.0 | 26.6 | 45.4 | 53.4 |
| **DETR** | 41M | 42.0 | 20.5 | 45.8 | **61.1** |
| **DETR-DC5-R101** | 60M | **44.9** | 23.7 | 49.5 | **62.3** |

- **大物体（AP_L）で +7.8 ポイント**: グローバル attention の威力
- **小物体（AP_S）で -5.5 ポイント**: 弱点。「FPN が Faster R-CNN を救ったように、将来の研究が DETR を救うだろう」と論文自身が予言（実際 Deformable DETR がこれを解決）

### 訓練の特殊性

- **300 〜 500 エポック**（標準的検出器の 10 倍以上）
- **AdamW + Xavier 初期化**（Faster R-CNN は SGD + 標準初期化）
- **補助デコーディング損失**（各 decoder 層の後で同じ Hungarian 損失を適用）が訓練安定化に重要
- **Backbone と transformer で学習率を分ける**（backbone は 1 桁低い学習率）

### 重要なアブレーション

1. **Encoder 層は重要**（0 → 6 で +3.9 AP）
2. **FFN は重要**（削除すると -2.3 AP）
3. **位置エンコーディング**: 完全削除でも 32 AP（baseline -7.8 AP）まで落ちる。Encoder の空間エンコは消しても -1.3 AP（attention の柔軟性）。Decoder の output positional encoding（=object queries）は **必須**
4. **GIoU 損失が主役**（GIoU だけで -0.7 AP、$\ell_1$ だけで -4.8 AP）
5. **Decoder 層を深くすると NMS は不要**（最終層では NMS を入れると AP が下がる）
6. **OOD 汎化**: 訓練に 13 体超えのキリン画像がなくても、合成画像で 24 体を正しく検出（§4.3.2）

## 限界・批判的視点

論文自身が認める限界：

- **小物体（AP_S）の性能不足**: FPN のような multi-scale 特徴が必要。**Deformable DETR**（Zhu et al., 2020）が直後にこれを解決
- **訓練時間が極端に長い**: 16 GPU で 3-6 日。500 エポック必要。**DINO-detector**（Zhang et al., ICLR 2023, [[entities/dino-detector]] / [[sources/dino-detector]]）や **DETA**（Ouyang-Zhang et al., 2022）がこれを大幅に短縮
- **収束の遅さ**: hand-designed prior がない代わりに、すべてをデータから学ぶ必要があり、初期の訓練が不安定
- **N=100 の上限**: §0.A.5.1 の分析で、画像内に 100 物体近く存在すると検出率が急落

その後の研究の課題：
- **Deformable DETR**（2020）: 各 attention を sparse な小領域に限定して計算量と訓練速度を改善
- **DAB-DETR**（2022）: object queries を 4D ボックスとして解釈
- **DN-DETR**（2022）: ノイズ加えた ground truth を補助訓練に使い収束加速
- **DINO**（2023, [Zhang et al., ICLR 2023], [[entities/dino-detector]] / [[sources/dino-detector]]）: 上記すべてを統合。**SSL の DINO（[[entities/dino]]）とは別物**、検出向け DETR 改良。COCO test-dev 63.3 AP（end-to-end Transformer 初の SOTA）
- **DETA**（2022, Ouyang-Zhang et al.）: 1 対 1 二部マッチングを **1 対多** に緩和、収束を大幅加速
- **CoDETR**（2023, Zong et al.）: 複数の補助 head（ATSS, Faster R-CNN）と DETR を協調訓練し、SOTA 達成

## 既存 wiki との接続

### Transformer の vision への適用としての位置づけ

[[concepts/vision-transformer]] の ViT（2020）と DETR（2020）は **ほぼ同時期** に発表されたが、目的が異なる：

| | ViT（Dosovitskiy et al., 2020） | DETR（Carion et al., 2020） |
|---|---|---|
| Transformer の使い方 | 画像を **パッチ系列** に分解し、Transformer でグローバル分類 | CNN backbone の特徴を Transformer encoder-decoder で再処理 |
| 目的 | **画像分類**（global） | **物体検出**（dense / set） |
| Backbone | なし（純粋 ViT） | あり（ResNet-50/101） |
| Position encoding | 1D 学習 | 2D sine（固定）+ 学習 object queries |

両者は **「CV における Transformer 革命の出発点」** として並列に重要。DETR の方が ViT より先に Transformer を CV に持ち込んだという見方もある（5 月 vs 10 月）。

### SAM / SAM 3 / Perception Encoder の系統との接続

DETR は、その後の **CV foundation model の検出ヘッド** として広く採用される：

| モデル | DETR との関係 |
|---|---|
| [[entities/sam-3]] | **DETR スタイル detector** + presence head + SAM 2 tracker で PCS タスクを解く |
| [[entities/perception-encoder]]（PEspatial） | **DETA decoder**（DETR-with-Assignment）を採用、COCO 66.0 box AP で SOTA |
| MDETR（2021） | DETR にテキスト条件を追加、open-vocab 検出の先駆け。SAM 3 の前身 |
| OWL-ViT（[Minderer et al., 2022]） | ViT backbone + DETR-style head、open-vocab 検出 |

**DETR は現代 CV foundation model の「検出ヘッド」標準** になったと言える。

### Attention 機構の vision における役割

DETR の **encoder attention がインスタンスを分離** することの発見（図 3）は、後の以下の発見と通底する：

- [[entities/dino]] の self-attention map に **オブジェクト境界が現れる**（教師なしで！）
- [[concepts/promptable-segmentation]] の SAM が attention map を直接マスク生成に使う
- **Transformer は「画像構造を発見する能力」を本質的に持つ** という見方の系列

## 用語と略称

- **DETR** = DEtection TRansformer。Transformer ベースの end-to-end 検出
- **set prediction**（集合予測） = 出力が「順序のない集合」である問題。物体検出はその典型
- **bipartite matching**（二部マッチング） = 2 つの集合（予測 N 個 vs ground truth）の間で 1 対 1 の対応を見つける問題
- **Hungarian algorithm**（ハンガリアン・アルゴリズム） = 二部マッチングを O(N³) で最適に解く古典的組合せ最適化アルゴリズム
- **NMS**（Non-Maximum Suppression, 非最大抑制） = IoU 閾値ベースで重複検出を後処理的に削除する手法。DETR は **訓練レベルで重複を排除** するため NMS 不要
- **anchor**（アンカー） = 画像上のグリッド位置に事前定義された参照ボックス。Faster R-CNN / YOLO / RetinaNet などの基盤。DETR は anchor フリー
- **proposal**（提案） = 物体候補のバウンディングボックス。Two-stage 検出器（Faster R-CNN）の第 1 段階で生成される
- **object queries**（物体クエリ） = DETR の decoder 入力となる N 個の学習可能な位置埋め込み。各クエリが 1 つの予測スロットに対応
- **GIoU**（Generalized IoU） = [Rezatofighi et al., CVPR 2019] のスケール不変なボックス類似度。重なりがない場合も勾配が伝わる
- **DC5**（Dilated C5）= ResNet の最後のステージで stride を削減し dilation を追加して空間解像度を 2 倍にする変種
- **FPN**（Feature Pyramid Network） = [Lin et al., CVPR 2017] のマルチスケール特徴ピラミッド。Faster R-CNN の小物体性能を救った。DETR の弱点（小物体）にも適用される（後の Deformable DETR が事実上の FPN 統合）
- **Panoptic Quality (PQ)** = パノプティックセグメンテーション標準指標。SQ（Segmentation Quality, IoU 平均）× RQ（Recognition Quality, F1）の積
- **things / stuff** = 可算物体（人、車など、数えられる）/ 不可算領域（空、道路、草地など、領域として扱う）の区別
- **AP_S / AP_M / AP_L** = それぞれ Small（area < 32²）/ Medium（32² < area < 96²）/ Large（area > 96²）の物体に対する AP
- **auxiliary decoding losses**（補助デコーディング損失） = 各 decoder 層の出力に同じ Hungarian 損失を適用し、訓練を安定化させる工夫
- **AdamW** = Adam + decoupled weight decay [Loshchilov & Hutter, ICLR 2019]
- **Xavier 初期化** = 重みを fan-in / fan-out で正規化して初期化 [Glorot & Bengio, 2010]
- **MDETR** = Modulated DETR（テキスト条件付き DETR）、open-vocab 検出の先駆け
- **Deformable DETR** = 各 attention を学習可能な少数のサンプリング点に限定、DETR の遅さ・小物体問題を解決 [Zhu et al., ICLR 2021]
- **DAB-DETR** / **DN-DETR** / **DINO-detector** = 収束加速・性能改善を進めた DETR 系統
- **DETA** = Detection Transformer with Assignment、PEspatial の検出 decoder
- **CoDETR** = Collaborative DETR、複数 head 協調訓練で SOTA

## 関連ページ

- [[translations/detr]] — 本文全文の和訳
- [[entities/detr]] — DETR モデルの詳細スペック
- [[concepts/object-detection]] — 物体検出タスク全体の俯瞰（DETR の位置づけ）
- [[concepts/vision-transformer]] — 同時期に Transformer を CV に持ち込んだ並列研究
- [[concepts/foundation-model]] — DETR は現代 CV foundation model の検出ヘッド標準
- [[sources/sam-3]] / [[entities/sam-3]] — DETR スタイル detector で PCS タスクを解く
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — DETA decoder で COCO 66.0 SOTA
- [[entities/sam]] — Mask R-CNN の精神的後継としての関係
