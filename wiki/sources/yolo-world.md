---
type: source
source_path: raw/papers/YOLO-World_ Real-Time Open-Vocabulary Object Detection.md
source_kind: paper
title: "YOLO-World: Real-Time Open-Vocabulary Object Detection"
authors: [Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan]
year: 2024
venue: CVPR 2024
ingested: 2026-05-28
tags: [yolo-world, yolo, real-time, open-vocabulary, object-detection, vision-language, tencent, repvl-pan, prompt-then-detect, clip-text-encoder]
translation: "[[translations/yolo-world]]"
---

# YOLO-World: リアルタイム Open-Vocabulary 物体検出

> 原典: [[translations/yolo-world]] ・ `raw/papers/YOLO-World_ Real-Time Open-Vocabulary Object Detection.md`
> 著者: Tianheng Cheng, Lin Song, Yixiao Ge ら（Tencent AI Lab / ARC Lab, Tencent PCG / 華中科技大学）
> 出典: CVPR 2024 / arXiv 2401.17270
> arXiv: <https://arxiv.org/abs/2401.17270>
> コード: <https://github.com/AILab-CVC/YOLO-World>

## 一言まとめ

**「YOLO + open-vocabulary 検出 = リアルタイム open-vocab 検出器」**。GLIP / Grounding DINO は Swin-L + DETR で **0.12-1.5 FPS の遅さ**、YOLO-World は **YOLOv8 + CLIP text encoder + 新ネック RepVL-PAN** で **52 FPS V100 で LVIS 35.4 AP**（**Grounding-DINO-T の 35 倍速い**）。鍵は (1) **Re-parameterizable Vision-Language PAN**（T-CSPLayer で max-sigmoid attention によりテキストを画像に注入、I-Pooling Attention で 27 patch tokens によりテキストを更新）、(2) **prompt-then-detect パラダイム**（テキスト encoder を推論時に削除、テキスト埋め込みをモデル重みに再パラメータ化）、(3) **CLIP text encoder が決定的**（BERT vs CLIP で APr +10.1 AP の差）、(4) **GLIP-L で CC3M に疑似ラベル**（246K 画像 / 821K 注釈）。**実用デプロイメントの open-vocab 検出を切り拓いた**。

## 背景と問題意識

2024 年初頭時点の open-vocabulary 検出の現実：

| 手法 | Backbone | パラメータ | FPS (V100) | LVIS AP | 弱点 |
|---|---|---|---|---|---|
| **[[entities/glip\|GLIP-T]]** | Swin-T | 232M | **0.12** | 24.9 | 死ぬほど遅い |
| **[[entities/glip\|GLIP-L]]** | Swin-L | n/a | n/a | n/a | 巨大 |
| **GLIPv2-T** | Swin-T | 232M | **0.12** | 26.9 | 同じく遅い |
| **[[entities/grounding-dino\|Grounding-DINO-T]]** | Swin-T | 172M | **1.5** | 25.6 | DETR の重さ |
| **DetCLIP-T** | Swin-T | 155M | **2.3** | 34.4 | これでも遅い |

**問題意識**:
- **エッジデバイスでの実用デプロイメントが不可能**: 0.12-2.3 FPS では推論レイテンシが許容範囲外
- **小型検出器での open-vocab 化が未探求**: 全研究が「大きな backbone」前提
- **テキストと画像を毎フレーム両方エンコード**: 高解像度ストリーミング推論にとって計算の無駄

> **補足: なぜリアルタイムが大事か** — 自律走行、ロボティクス、AR/VR、警備カメラ等の応用では「30 FPS で物体検出」が前提。GLIP の 0.12 FPS では「1 フレーム検出に 8 秒」かかる。実用化のために **2-3 桁の高速化** が必要だった。

## 提案手法 / 主張

### 全体構造

```
画像 → YOLOv8 Backbone（Darknet）→ C3, C4, C5（multi-scale）─┐
                                                              │
                                                  ★ RepVL-PAN ★（top-down + bottom-up）
                                                  各 CSPLayer に：
                                                  ├── T-CSPLayer（テキスト → 画像）
                                                  └── I-Pooling Attention（画像 → テキスト）
                                                              │
テキスト → ★ CLIP Text Encoder ★（凍結）→ W ∈ R^{C×D} ────┘
            ("person. bicycle. car. ..." or 名詞句)
                                                              │
                                                  Decoupled Head（2 × 3x3 conv）
                                                  → bbox b_k、object embedding e_k
                                                              │
                                                  ★ Text Contrastive Head ★
                                                  s_{k,j} = α · L2(e_k) · L2(w_j)^T + β
                                                              │
                                                  損失: region-text contrastive (CE) + IoU + DFL
```

### 鍵 1: Re-parameterizable Vision-Language PAN (RepVL-PAN)

YOLOv8 の PAN ネックに **2 つの cross-modality 融合機構** を追加：

#### T-CSPLayer（Text → Image、Max-Sigmoid Attention）

YOLOv8 の **C2f（CSPLayer）** の最後の dark bottleneck block の後で、テキスト特徴を画像特徴に注入：

$$
X_l^{\prime} = X_l \cdot \delta\left(\max_{j \in \{1..C\}}(X_l W_j^{\top})\right)^{\top}
$$

- 各画像位置 × 各テキスト token の内積を取る（$X_l W_j^{\top}$、shape: H×W×C）
- テキスト軸で **max-pooling**（最も関連するテキスト token のスコア、shape: H×W）
- sigmoid で 0-1 に正規化
- 元の画像特徴 $X_l$ にチャネル方向でスケーリング

**直感**: 「画像のこの位置はテキストプロンプトのどれかに関連するか？」を 0-1 で評価し、関連する位置の特徴を強調。

> **補足: なぜ max-sigmoid なのか** — softmax を使うと「テキスト token 間の競合」が生じるが、検出ではプロンプト内の各カテゴリが独立に存在しうる（複数カテゴリが同時に画像内に存在）。max-sigmoid なら「最も近いテキストの強さ」を独立に評価できる。

#### I-Pooling Attention（Image → Text）

テキスト埋め込みを画像で更新するため、軽量な cross-attention：

1. マルチスケール特徴 $\{X_3, X_4, X_5\}$ それぞれに **3×3 max pooling**
2. → 各特徴マップから 9 patches、合計 **27 patch tokens** $\tilde{X} \in \mathbb{R}^{27 \times D}$
3. テキスト埋め込み $W$ を query、$\tilde{X}$ を key/value として **MultiHead-Attention**:

$$
W^{\prime} = W + \texttt{MultiHead-Attention}(W, \tilde{X}, \tilde{X})
$$

**画像の概略的内容にテキスト埋め込みを condition** する。

### 鍵 2: Prompt-Then-Detect パラダイム（核心の革新）

**伝統的 open-vocab 検出（GLIP / Grounding DINO）**:

```
[Inference]:
  画像 + テキスト → 両方エンコード（毎回）→ 検出
  → テキスト encoder の計算が毎フレーム発生（高コスト）
```

**YOLO-World（prompt-then-detect）**:

```
[Setup（1 回のみ）]:
  ユーザーがプロンプト T を定義
  → CLIP Text Encoder で W = TextEncoder(T) を計算
  → ★ W を RepVL-PAN の重みに re-parameterize ★
  → text encoder を破棄

[Inference（高速）]:
  画像 → 純粋な YOLO ネットワーク（テキスト encoder なし）→ 検出
  → テキスト計算がゼロ
```

これにより、**通常の YOLO 推論と同じ速度** で open-vocab 検出が可能に。

> **補足: re-parameterization の具体例** — T-CSPLayer の $X_l W_j^{\top}$ という演算は、**$W_j$ を 1×1 conv の重みとして見なせる**。事前に $W$ を計算しておけば、推論時は「画像と固定 1×1 conv の計算」だけ。これは **構造的に YOLO の標準 layer と同じ** で、TensorRT 等の高速化ツールも適用可能。

### 鍵 3: Region-Text Contrastive Loss

検出データ、grounding データ、image-text データを **region-text ペア** $\{B_i, t_i\}$ に統一：

- **検出データ** (Objects365): $t_i$ = カテゴリ名 ("person", "bicycle")
- **grounding データ** (GoldG = GQA + Flickr30K): $t_i$ = 名詞句 ("a red car", "standing man")
- **image-text データ** (CC3M): $t_i$ = NLP パーサ抽出の名詞句、box は GLIP-L で疑似生成

**損失関数**:

$$
\mathcal{L}(I) = \mathcal{L}_{\text{con}} + \lambda_I \cdot (\mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{dfl}})
$$

ここで $\lambda_I$ は **インジケータ**:
- 検出 / grounding データ: $\lambda_I = 1$（正確な box なので回帰損失を使う）
- image-text データ: $\lambda_I = 0$（**疑似 box はノイジーなので回帰損失を使わない**、分類のみ）

これは「**疑似ラベルの不確実性を損失設計で吸収する** 工夫」として重要。

### 鍵 4: 疑似ラベリングパイプライン（GLIP-L + CLIP filtering）

CC3M（246K 画像）から region-text ペアを自動生成：

```
1. n-gram で名詞句抽出: caption → 候補 phrases {t_k}
2. GLIP-L で box 生成: GLIP(image, phrases) → {B_i, t_i, c_i}（粗い擬似ラベル）
3. CLIP で再スコアリング・フィルタリング:
   - 画像-テキスト全体スコア s^img
   - region-text スコア s^r_i（box を crop して CLIP）
   - 再ラベル（optional）: 各 crop に対し全名詞句の中で最大の CLIP score の名詞を再割当
   - 再スコア: c̃_i = sqrt(c_i × s^r_i)
   - region-level NMS（IoU=0.5）+ confidence > 0.3 フィルタ
   - image-level: s = sqrt(s^img × s^region) > 0.3
4. 最終: 821K region-text 注釈
```

> **補足: なぜ GLIP-L を使うか** — [[entities/glip]] / [[sources/glip]] は本 wiki に ingest 済みで、GLIP の self-training 戦略の延長線上にある。YOLO-World は **GLIP の知識を YOLOv8 に蒸留** する形になる（直接の蒸留ではなく、データを介した間接的な転移）。

### 鍵 5: CLIP Text Encoder が決定的

ablation（表 5）で BERT vs CLIP の比較：

| Text Encoder | Frozen? | AP | APr |
|---|---|---|---|
| BERT-base | Frozen | 14.6 | 3.4 |
| BERT-base | Fine-tune | 18.3 | 6.6 |
| **CLIP-base** | **Frozen** | **22.4** | **14.5 (+10.1 vs BERT)** |
| CLIP-base | Fine-tune | 19.3 | 8.6 |

**観察**:
- **CLIP は BERT を rare クラスで +10.1 AP** 圧倒（事前学習で image-text 整列を学んでいるため）
- **CLIP は凍結が最良**: fine-tune すると汎化能力が劣化（特に 365 カテゴリの O365 だけだと事前学習の image-text 知識を破壊）
- **BERT は fine-tune が有効**: 元々 text-only なので画像との整列を学ぶ必要あり

これは [[sources/glip]] が BERT を使うのと対照的で、**「open-vocab の鍵は CLIP の image-text 整合性」** という GLIP からの洞察更新。

## 実験結果と知見

### LVIS ゼロショット（表 2、最重要結果）

| Method | Backbone | Params | Pre-train | FPS | AP | APr |
|---|---|---|---|---|---|---|
| GLIP-T | Swin-T | 232M | O365+GoldG | **0.12** | 24.9 | 17.7 |
| Grounding-DINO-T | Swin-T | 172M | O365+GoldG | **1.5** | 25.6 | 14.4 |
| DetCLIP-T | Swin-T | 155M | O365+GoldG | **2.3** | 34.4 | 26.9 |
| **YOLO-World-S** | YOLOv8-S | **13M** | O365+GoldG | **74.1** | 26.2 | 19.1 |
| **YOLO-World-M** | YOLOv8-M | **29M** | O365+GoldG | **58.1** | 31.0 | 23.8 |
| **YOLO-World-L** | YOLOv8-L | **48M** | O365+GoldG | **52.0** | 35.0 | 27.1 |
| **YOLO-World-L** | YOLOv8-L | **48M** | +CC3M† | **52.0** | **35.4** | 27.6 |

**衝撃の結果**:
- YOLO-World-S（**13M params**）が GLIP-T（232M）を **AP で同等 + FPS で 617×**
- YOLO-World-L（48M）が DetCLIP-T（155M）と **同等 AP + FPS で 22×**
- **APr でも competitive**（YOLO-World-L 27.6 vs DetCLIP-T 26.9）

### COCO ゼロショット転送 + Fine-tune（表 6）

| Method | Setting | AP | FPS |
|---|---|---|---|
| YOLOv8-L (scratch) | Closed-set scratch | 52.9 | 159 |
| **YOLO-World-L (ZS)** | **Zero-shot transfer** | **45.1** | - |
| **YOLO-World-L (FT)** | Fine-tuned | **53.3** | 156 |

**ゼロショット 45.1 AP** は YOLOv8-L scratch の 52.9 に近い。Fine-tune すると **53.3 AP** で YOLOv8-L (scratch) を超える。

### LVIS Fine-tune（表 7）

LVIS-base（866 クラス）で fine-tune、LVIS-novel（337 rare クラス）で評価：

| Method | AP | APr |
|---|---|---|
| YOLOv8-L (full LVIS, oracle) | 26.9 | 10.2 |
| ViLD | 27.8 | 16.7 |
| BARON | 29.5 | 23.2 |
| **YOLO-World-L** | **34.1** | **20.4** |

**先行 two-stage の SOTA を one-stage で上回る**。

### OVIS（Open-Vocabulary Instance Segmentation、表 8）

新たな OVIS タスクへの拡張：

| Setting | Fine-tune | Mask AP | Mask APr |
|---|---|---|---|
| LVIS-base | Seg Head only | 19.1 | 14.2 |
| LVIS-base | All | **28.7** | 15.0 |

「Seg head のみ fine-tune」だと **open-vocab 性能を保持**、「All fine-tune」だと **LVIS で +9.6 AP だが open-vocab 性能が -0.6 APr 劣化**。**用途に応じた trade-off**。

### ablation: RepVL-PAN の効果（表 4）

| GQA | T→I | I→T | AP | APr |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 22.4 | 14.5 |
| ✗ | ✓ | ✓ | 23.5 (+1.1) | 16.2 (+1.7) |
| ✓ | ✗ | ✗ | 29.7 | 21.0 |
| ✓ | ✓ | ✓ | **31.9 (+2.2)** | **22.5 (+1.5)** |

- **GQA（grounding データ、豊富なテキスト）があるほど RepVL-PAN の効果が大きい**
- T→I + I→T の組み合わせが効く

## 限界・批判的視点

- **APr は依然として DetCLIP-T に劣る**: rare カテゴリでの認識能力では特化型に負ける（APr 27.6 vs DetCLIP 26.9 は僅差だが他指標で逆転されることも）
- **疑似ラベルの品質依存**: GLIP-L の疑似 box の品質に YOLO-World の性能が依存
- **COCO fine-tune では RepVL-PAN 不要**: 80 カテゴリだと言語情報の意味が薄い → 推論加速のため除去（しかし「open-vocab 検出器」としての本質を失う）
- **CLIP text encoder の fine-tune が逆効果**: 365 カテゴリだけの O365 では事前学習の image-text 知識を破壊。**大規模・多様データでないと fine-tune できない**
- **YOLO 系特有の弱点を継承**: 小物体検出、長尾分布対応など（ただし RepVL-PAN で改善）
- **テキスト依存の推論**: 推論前にプロンプト定義が必要（ただし offline vocabulary で 1 回のみ）

## 既存 wiki との接続

### GLIP / Grounding DINO ファミリーへの「real-time 路線」

[[concepts/object-detection]] の系譜上の位置：

```
DETR 系（高精度・低速）              YOLO 系（高速・固定語彙）
─────────────────────────           ────────────────────
DETR (2020)                          YOLOv1-v8
  ↓                                    ↓
Deformable DETR / DAB-DETR /        YOLOv6 / YOLOv7 / YOLOv8
DN-DETR / DINO 検出器                  ↓
  ↓ + テキスト                         ↓ + open-vocab
GLIP / Grounding DINO                ★ YOLO-World（本論文）★
  ↓ + presence head + マスク + 動画
SAM 3
```

**YOLO-World は「YOLO 系 + open-vocab」という新軸を開いた**。GLIP / Grounding DINO の「精度志向」と並行する「実用志向」路線。

### CLIP の object-level 拡張系統で

| 軸 | CLIP（2021） | GLIP（2022） | Grounding DINO（2023） | **YOLO-World（2024）** |
|---|---|---|---|---|
| タスク | image 分類 | 検出 + grounding | 検出 + grounding + REC | 検出 + REC |
| 検出器 | n/a | DyHead | DINO 検出器 | **YOLOv8** |
| 融合 | late | deep (X-MHA) | tight 3-phase | **RepVL-PAN（軽量）** |
| FPS | n/a | 0.12 | 1.5 | **52** |
| パラメータ | n/a | 232M | 172M | **48M** |
| Text encoder | CLIP（自前） | BERT | BERT | **CLIP（流用）** |

**YOLO-World は CLIP text encoder を流用** することで、CLIP の image-text 整合性を効率的に YOLO に注入。

### GLIP との実は深い繋がり

YOLO-World は **GLIP-L を疑似ラベル生成 teacher として使用**（CC3M に GLIP-L で box 生成）。これは [[sources/glip]] の self-training パイプラインの直接の継承：

```
GLIP-T (teacher) → CC4M に擬似ラベル → GLIP-L 訓練
                          ↓
                  YOLO-World が GLIP-L をさらに teacher として活用
                          ↓
                  CC3M に擬似ラベル → YOLO-World 訓練
```

**GLIP → YOLO-World への知識転移** は、**「高精度モデルから高速モデルへの蒸留」** という重要な実用パターン。

### YOLOv8 の open-vocab 化

[[concepts/object-detection]] の YOLO 系統では、YOLOv8 は **closed-set 検出のリアルタイム標準** だった。YOLO-World はこれを **open-vocab に拡張した最初の本格的研究**:

- **アーキテクチャ的に最小限の変更**: YOLOv8 の PAN ネックを RepVL-PAN に置き換えただけ
- **推論時は YOLOv8 と同じ速度**: re-parameterization で text encoder を削除
- **既存 YOLO デプロイメント基盤（TensorRT, MMYOLO 等）が流用可能**

### SAM 3 / PE と対照的

[[entities/sam-3]] / [[entities/perception-encoder]] が「高精度・大型」志向なのに対し、YOLO-World は「実用・軽量」志向：

| 軸 | SAM 3 / PE | YOLO-World |
|---|---|---|
| 主な貢献 | 精度 SOTA | 実用性能 SOTA |
| Backbone | Perception Encoder（大型） | YOLOv8 Darknet（軽量） |
| 推論速度 | 低速 | 52 FPS V100 |
| デプロイメント | サーバ前提 | エッジデバイス可 |
| マスク生成 | あり | OVIS で対応 |
| 動画対応 | あり（SAM 3） | なし |

**用途別の使い分け**:
- 自律走行・ロボティクス・AR/VR: YOLO-World
- アノテーション支援・コンテンツ理解: SAM 3 / PE

## 用語と略称

- **YOLO-World** = YOLO + Open-Vocabulary World、本論文のモデル
- **YOLO** = You Only Look Once（[Redmon et al., 2016]）、リアルタイム one-stage 検出器の祖
- **YOLOv8** = Ultralytics 版（2023）、本論文の base architecture
- **RepVL-PAN** = Re-parameterizable Vision-Language Path Aggregation Network、本論文の核心
- **PAN** = Path Aggregation Network（[Liu et al., 2018]）、ネック構造
- **CSPLayer / C2f** = Cross-Stage Partial Layer（YOLOv8 の主要ブロック）
- **T-CSPLayer** = Text-guided CSPLayer、テキストを画像に注入
- **I-Pooling Attention** = Image-Pooling Attention、画像でテキストを更新
- **max-sigmoid attention** = T-CSPLayer の核心演算（softmax の代わりに max + sigmoid）
- **re-parameterization** = 訓練時と推論時で異なる計算グラフを使う技法（[Ding et al., RepVGG, 2021]）
- **prompt-then-detect** = テキストを事前エンコードしてオフライン語彙を作る推論パラダイム
- **online vocabulary** = 訓練時、各 mosaic サンプル用に動的に構築する語彙（最大 80 名詞）
- **offline vocabulary** = 推論時、事前計算されたテキスト埋め込みのセット
- **region-text pair** = $\{B_i, t_i\}$ のペア（box とテキスト）
- **region-text contrastive loss** = 各 region の object embedding と各 text token の内積を交差エントロピーで損失化
- **task-aligned label assignment** ([Feng et al., TOOD, 2021]) = YOLO の正解割当戦略
- **DFL** = Distribution Focal Loss（[Li et al., 2020]、bbox 回帰用）
- **decoupled head** = 分類と box 回帰のヘッドを分ける YOLOv8 設計
- **CLIP text encoder** = CLIP（[[entities/clip]]）の言語塔、本論文では凍結使用
- **n-gram algorithm** = キャプションから名詞句を抽出する古典的 NLP 手法
- **MDETR** = Modulated DETR、[Kamath et al., ICCV 2021]、テキスト条件付き DETR の祖
- **GLIP / GLIPv2** = [Li et al., CVPR 2022] / [Zhang et al., NeurIPS 2022]、phrase grounding ベース検出
- **Grounding DINO** = [[sources/grounding-dino]]、GLIP × DINO 検出器
- **DetCLIP** = [Yao et al., NeurIPS 2022]、ATSS + large-scale captioning による open-vocab
- **OWL-ViT / OWL-ViTv2** = [Minderer et al., 2022/2023]、ViT + DETR 版 open-vocab
- **ZSD-YOLO** = [Xie & Zheng, 2022]、YOLO 版 zero-shot detection（YOLO-World の前駆だが性能劣る）
- **MMYOLO / MMDetection** = OpenMMLab のフレームワーク
- **Objects365 V1 / V2** = 365 カテゴリの検出データ（0.6M / 1.7M images）
- **GQA** = Question Answering ベース visual reasoning データ（grounding 用）
- **Flickr30k Entities** = phrase grounding 標準ベンチマーク
- **GoldG** = MDETR キュレーションの 0.8M grounding データ（GQA + Flickr30k、COCO 除外）
- **CC3M** = Conceptual Captions（3M、Google）、YOLO-World では 246K subset を疑似ラベリング
- **LVIS** = Large Vocabulary Instance Segmentation、1203 カテゴリ、長尾分布
- **APr / APc / APf** = rare / common / frequent クラスの AP（LVIS）
- **Fixed AP** = LVIS 評価の公平比較プロトコル（[Dave et al., 2021]）
- **OVIS** = Open-Vocabulary Instance Segmentation
- **LVIS-base / LVIS-novel** = 866 クラス（common + frequent）/ 337 rare クラス
- **TensorRT** = NVIDIA の推論高速化ツール
- **TOOD** = Task-aligned One-stage Object Detection（[Feng et al., 2021]、label assignment）

## 関連ページ

- [[translations/yolo-world]] — 本文全文の和訳
- [[entities/yolo-world]] — YOLO-World モデルの詳細スペック
- [[concepts/object-detection]] — 物体検出全体（YOLO 系統と open-vocab 系統の交差点として）
- [[concepts/zero-shot-transfer]] — YOLO-World のゼロショット転送
- [[entities/clip]] / [[sources/clip]] — YOLO-World は CLIP text encoder を流用
- [[entities/glip]] / [[sources/glip]] — YOLO-World は GLIP-L を疑似ラベリング teacher として活用
- [[entities/grounding-dino]] / [[sources/grounding-dino]] — 同じ open-vocab 系統、精度 vs 速度の対比
- [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — Grounding DINO 1.5 Edge は同じ real-time open-vocab 路線の直接的競合（2024 May）
- [[entities/dino-x]] / [[sources/dino-x]] — DINO-X Edge が YOLO-Worldv2-L を LVIS-mv で +15.3 AP 上回りつつ Orin NX 20.1 FPS（2024 Nov）
- [[concepts/foundation-model]] — open-vocab 検出は CV foundation model の重要分野
