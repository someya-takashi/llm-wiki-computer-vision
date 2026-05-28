---
type: source
source_path: raw/papers/Grounding DINO 1.5_ Advance the "Edge" of Open-Set Object Detection.md
source_kind: paper
title: "Grounding DINO 1.5: Advance the \"Edge\" of Open-Set Object Detection"
authors: [Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, Yuda Xiong, Hao Zhang, Feng Li, Peijun Tang, Kent Yu, Lei Zhang]
year: 2024
venue: arXiv preprint (May 2024)
ingested: 2026-05-28
tags: [grounding-dino-1-5, grounding-dino, open-set-detection, edge-deployment, vit-l, efficientvit, idea-research, grounding-20m, real-time]
translation: [[translations/grounding-dino-1-5]]
---

# Grounding DINO 1.5: Open-Set 検出の「Edge」を進化させる

> 原典: [[translations/grounding-dino-1-5]] ・ `raw/papers/Grounding DINO 1.5_ Advance the "Edge" of Open-Set Object Detection.md`
> 著者: Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng ら（IDEA Research）
> 出典: arXiv preprint, May 2024 (arXiv 2405.10300)
> arXiv: <https://arxiv.org/abs/2405.10300>
> API: <https://github.com/IDEA-Research/Grounding-DINO-1.5-API>
> プロジェクト: <https://deepdataspace.com/home>

## 一言まとめ

**「[[entities/grounding-dino]] の Pro/Edge 双子拡張」**。論文タイトルの "Edge" は **二重の意味**: (1) **open-set 検出分野の「先端（edge）」を押し広げる**、(2) **エッジコンピューティング（edge）デバイスへ持っていく**。**Grounding DINO 1.5 Pro**（ViT-L backbone + Grounding-20M 訓練データ）で精度 SOTA を更新（**COCO ZS 54.3 AP / LVIS-minival 55.7 AP / ODinW35 30.2**、Grounding DINO 比 +1.8/+28.3/+4.2）。**Grounding DINO 1.5 Edge**（EfficientViT-L1 + 新 efficient feature enhancer、P5 のみ cross-modality 融合 + vanilla self-attn）で **A100 TensorRT 75.2 FPS / Orin NX 10+ FPS、LVIS-minival 36.2 AP**（YOLO-Worldv2-L の 32.9 を上回る）。**精度志向と実用志向の両極を 1 モデルスイートで統合**し、GLIP / Grounding DINO 系統と YOLO-World 系統の双方の領分を吸収した。

## 背景と問題意識

[[entities/grounding-dino]]（ECCV 2024）以降の open-set 検出の発展を踏まえると、2 つの需要が明確になった：

| 軸 | 状況 |
|---|---|
| **精度の限界** | Grounding-DINO-L は COCO ZS 52.5 AP、LVIS-minival では 30 台前半。DetCLIPv3 が LVIS-minival 48.8 AP で SOTA だったが、まだ上限がある |
| **エッジ展開** | 自律走行・医療・computational photography 等で open-set 検出が望まれるが、Grounding DINO は **Orin NX で 1.1 FPS** という事実上不可能な速度。[[entities/yolo-world|YOLO-World]] が登場したが、それでも 30 FPS には届かない |

**Grounding DINO 1.5 の問題意識**: 「精度と速度の **両方** で SOTA を取る、1 つのモデルスイート」。

> **補足: IDEA Research の戦略的位置** — Grounding DINO 1.5 は、Grounding DINO（[[entities/grounding-dino]]、ECCV 2024）の著者陣と同じ IDEA グループから 2024 年 5 月に出版（タイミング的には ECCV 2024 採択前後）。**Grounding DINO の正統な公式後継** であり、API ベースの商用展開も視野に入れた研究（DeepDataSpace プラットフォーム経由でモデル提供）。

## 提案手法 / 主張

### 全体構造（Pro と Edge の比較）

```
Grounding DINO 1.5 Pro:
画像 → ViT-L backbone → multi-scale features ────┐
                                                  ↓
                                       Feature Enhancer（重い、deep early fusion）
                                                  ↓
テキスト → BERT-base ────────────────────────────┘
            (改善された負例サンプリング)
                                                  ↓
                                       Cross-Modality Decoder
                                                  ↓
                                       (box, class) 予測


Grounding DINO 1.5 Edge:
画像 → EfficientViT-L1 backbone → multi-scale features ─┐
                                                         │
                                       ★ Efficient Feature Enhancer ★
                                       ├── P5 のみ cross-modality 融合
                                       ├── Deformable → Vanilla self-attn
                                       └── Cross-scale fusion で P3/P4 統合
                                                         │
テキスト → BERT-base ──────────────────────────────────┘
                                                         │
                                       Cross-Modality Decoder（軽量）
                                                         │
                                       (box, class) 予測
```

### Pro モデルの 3 つの改善点

#### 1. ViT-L Backbone への拡張

[[entities/grounding-dino|Grounding DINO]] の Swin-L から **事前学習された ViT-L** に置換：

- **下流タスクでの優れた性能**: ViT 系は MAE / DINOv2 / SigLIP / PE 等のリッチな事前学習済み重みが利用可能
- **純粋な Transformer 設計**: 訓練と推論プロセスの最適化基盤が堅固

#### 2. Grounding-20M データセット

**20M+ grounding 画像** を公開ソースから収集 → IDEA 独自のデータエンジンと注釈パイプラインで **grounding 注釈の高品質を保証**:

- 先行の Grounding DINO（27M、GoldG + Cap4M 等）を **質的に超える**（具体的なソース内訳は論文では非公開、IDEA のデータエンジンは商用基盤の一部）
- 元の Grounding DINO の "Grounded SAM" 系の流れの延長線上にある

#### 3. Early Fusion + 改善された負例サンプリング

**Early fusion vs Late fusion の trade-off** を分析：

| 軸 | Early Fusion | Late Fusion |
|---|---|---|
| 検出再現率 | **高い** | 低い |
| Box 精度 | **高い** | 低い |
| Hallucination（存在しない物体を予測） | **高い**（弱点） | 低い |
| 訓練の難しさ | 容易 | 難しい（vision-language アラインメント） |

**Grounding DINO 1.5 Pro**: **early fusion を保持** しながら、**訓練中の負例サンプリング戦略を改善**（負例の割合を増やす）して hallucination を抑制 → **両得**。

> **補足: なぜ negative sampling が hallucination を減らすか** — early fusion の hallucination の原因は「すべての名詞句に対して何か box を予測したくなる過剰な inductive bias」。訓練中に「**画像内に存在しない名詞句に対しては何も予測しない**」ことを明示的に学ばせる負例（カテゴリ名は与えるが画像にそのオブジェクトはない）を増やすことで、過剰予測を抑制できる。

### Edge モデルの 3 つの効率化技法

[[entities/grounding-dino|Grounding DINO]] は Orin NX で **1.1 FPS** という非実用的な速度。Edge モデルでこれを **10+ FPS** にする 3 つの工夫：

#### 1. P5 のみ Cross-Modality 融合（最重要）

**Lite-DETR** [Li et al., 2023] の洞察に従って：

- **低レベル画像特徴（P3, P4）は意味情報に欠ける**: cross-modality 融合に使ってもメリットが少ない
- **計算コストが膨大**: P3 や P4 のトークン数は P5 の数十倍
- **解決**: cross-modality 融合を **P5 レベルのみに制限** → 処理トークン数を **大幅削減**

#### 2. Deformable → Vanilla Self-Attention

Deformable attention は精度は良いが **エッジ GPU での実装が複雑**（カスタム CUDA kernel が必要）。Vanilla self-attention に置き換えることで：

- **TensorRT 等の標準ツールで簡単に最適化**
- **エッジデバイスでのデプロイメントが容易に**
- 精度は若干低下するが許容範囲

#### 3. Cross-Scale Feature Fusion で P3/P4 を統合

P5 のみ cross-modality 融合した特徴と、P3/P4 の低レベル特徴を **cross-scale feature fusion モジュール**（[37] Zhao et al.）で統合 → **マルチスケール検出能力を維持**。

#### 4. EfficientViT-L1 Backbone

[Cai et al., 2023] の **EfficientViT-L1** を採用：
- 通常の ViT-L より大幅に高速
- **高速マルチスケール特徴抽出** 用に最適化されたアーキテクチャ
- Edge GPU での実行効率が良い

### Grounding-20M データセット

「**公開可能なソースから収集した 20M+ grounding 画像**」とのみ公表。詳細な構成は非公開（IDEA Research のデータエンジン技術として商用化されている可能性）。

> **補足: Grounding DINO の訓練データとの関係** — Grounding DINO（オリジナル）は FourODs + GoldG + Cap4M + COCO + RefC で約 27M。Grounding DINO 1.5 の **Grounding-20M は質を上げた選別データ** で、サイズは下がっても性能は大幅に向上。**「データの量より質」** の典型例。

## 実験結果と知見

### COCO / LVIS ゼロショット（表 1）

| Model | Backbone | Pre-train | COCO | LVIS-mv | LVIS-mv APr | LVIS-val |
|---|---|---|---|---|---|---|
| Grounding DINO | Swin-T | O365+GoldG+Cap4M | 48.4 | 27.4 | 18.1 | - |
| Grounding DINO | Swin-L | O365+OID+GoldG | 52.5 | - | - | - |
| YOLO-World | YOLOv8-L | O365+GoldG+CC3M | 45.1 | 35.4 | 27.6 | - |
| DetCLIPv3 | Swin-L | O365+V3Det+GoldG+GranuCap50M | - | **48.8** | 49.9 | 41.4 |
| T-Rex2 (text) | Swin-L | O365+OID+GoldG+CC3M+SBU+LAION | 52.2 | 54.9 | 49.2 | 45.8 |
| **Grounding DINO 1.5 Pro** | **ViT-L** | **Grounding-20M** | **54.3** | **55.7** | **56.1** | **47.6** |

**重要な比較**:
- **COCO ZS 54.3**: GD Swin-L 52.5 から **+1.8 AP**
- **LVIS-mv 55.7**: DetCLIPv3 から **+6.9 AP**
- **LVIS-val 47.6**: DetCLIPv3 から **+6.2 AP**
- **GD Swin-T からは LVIS-mv で +28.3 AP** の大幅改善（27.4 → 55.7）

### ODinW ベンチマーク

| Model | ODinW35 | ODinW13 |
|---|---|---|
| GLIP-L | - | 52.1 |
| Grounding DINO Swin-L | 26.1 | 56.9 |
| Florence | 25.8 | - |
| OmDet-Turbo-B | 30.1 | 54.7 |
| **Grounding DINO 1.5 Pro** | **30.2** | **58.7** |

**ODinW35 ゼロショットで 30.2 AP の新記録**（GD Swin-L から +4.2 AP）。35 多様タスクで普遍的に強い。

### Pro Fine-tune 結果（表 3）

| Setting | LVIS-mv | LVIS-val | ODinW35 | ODinW13 |
|---|---|---|---|---|
| GD 1.5 Pro (zero-shot) | 55.7 | 47.6 | 30.2 | 58.7 |
| **GD 1.5 Pro (fine-tune)** | **68.1 (+12.4)** | **63.5 (+15.9)** | **70.6 (+40.4)** | **72.4 (+13.7)** |

**Fine-tune での飛躍的改善**:
- ODinW35: 30.2 → **70.6 AP**（+40.4!）
- LVIS-val: 47.6 → 63.5（+15.9）
- DetCLIPv3 fine-tune の 60.8（LVIS-mv）を超える

### Edge モデル（表 5）

| Model | Backbone | Input | COCO | LVIS-mv | A100 FPS (PyT/TRT) | Orin FPS |
|---|---|---|---|---|---|---|
| Grounding DINO-T | Swin-T | 800×1333 | 48.4 | 27.4 | 9.4 / 42.6 | **1.1** |
| YOLO-Worldv2-S | YOLOv8-S | 640 | - | 22.7 | 47.4 / - | - |
| YOLO-Worldv2-M | YOLOv8-M | 640 | - | 30.0 | 42.7 / - | - |
| YOLO-Worldv2-L | YOLOv8-L | 640 | - | 32.9 | 37.4 / - | - |
| OmDet-Turbo-T | Swin-T | 640 | 42.5 | 30.3 | 21.5 / 140.0 | - |
| **GD 1.5 Edge** | **EfficientViT-L1** | **640** | **42.9** | **33.5** | **21.7 / 111.6** | **10.7** |
| **GD 1.5 Edge** | **EfficientViT-L1** | **800×1333** | **45.0** | **36.2** | **18.5 / 75.2** | **5.5** |

**衝撃の結果**:
- **LVIS-mv で YOLO-Worldv2-L (32.9 AP) を 33.5/36.2 AP で超える** + 同等の Orin 速度
- **Grounding DINO-T (1.1 FPS Orin) を 10.7 FPS に高速化**
- A100 TensorRT で 75.2-111.6 FPS（YOLO-Worldv2 と同等）

## 限界・批判的視点

- **Grounding-20M の詳細非公開**: IDEA Research の商用データエンジンが背景にあり、コミュニティでの完全再現が難しい
- **モデル重みは API 経由のみ**: 完全なオープンソース化されていない（API は <https://github.com/IDEA-Research/Grounding-DINO-1.5-API>）。商用利用の制約あり
- **Pro モデルは重い**: ViT-L backbone + 20M データの訓練は計算リソース大、再現には大量の GPU が必要
- **Edge モデルでも一般用途では不十分**: Orin NX で 10.7 FPS は許容範囲だが、リアルタイム要求が厳しい応用（30 FPS）には届かない
- **早期融合の hallucination 問題は完全には解決されず**: 改善された負例サンプリングで緩和しているが、依然として late fusion より hallucination 多め
- **比較対象の選択**: 論文は DetCLIPv3 を主要比較相手として強調するが、SOTA 主張は **Pro モデルが ViT-L、DetCLIPv3 が Swin-L** という条件差もある

## 既存 wiki との接続

### Grounding DINO の正統な後継

[[entities/grounding-dino]] / [[sources/grounding-dino]] の **直接の後継** で、同じ IDEA Research / 一部の著者が継続。系譜：

```
2020  DETR                        — 集合予測パラダイム
2022  GLIP                        — phrase grounding
2023  DINO 検出器                 — DETR 系の集大成（CDN + Mixed QS + LFT）
2023  ★ Grounding DINO ★          — GLIP × DINO 検出器
            │
2024 May ★ Grounding DINO 1.5 ★   — Pro/Edge スイート（本論文）
            │
            ├── 後継候補: Grounding DINO 1.6, DINO-X
            └── 並列: SAM 3（より広い PCS タスク、[[entities/sam-3]]）
```

### YOLO-World との並列・統合

[[entities/yolo-world]]（CVPR 2024）と **同時期に登場した競合**:

| 軸 | **YOLO-World** | **Grounding DINO 1.5 Pro** | **Grounding DINO 1.5 Edge** |
|---|---|---|---|
| 戦略 | 「精度を犠牲にして速度」 | 「速度を犠牲にして精度」 | **「両方追求」** |
| Backbone | YOLOv8-L (48M) | ViT-L (重い) | EfficientViT-L1 (軽い) |
| Text encoder | CLIP（凍結） | BERT-base | BERT-base |
| LVIS-mv ZS | 35.4 | **55.7** | 36.2 |
| Orin NX FPS | - | - | **10.7** |
| 設計思想 | Prompt-Then-Detect | Deep early fusion | Efficient feature enhancer |

**Grounding DINO 1.5 Edge と YOLO-World は LVIS-mv で互角**（36.2 vs 35.4）、Orin NX 上のリアルタイム性能も Edge が達成。**「YOLO-World を超える real-time open-vocab 検出器」** の主張。

### SAM 3 / PE との関係

[[entities/sam-3]] / [[entities/perception-encoder]] は **より広い PCS（Promptable Concept Segmentation）タスク** を目指す（マスク + 動画 + presence head）。Grounding DINO 1.5 は **box 検出に集中**、SAM 3 への補完的位置付け。

PE PEspatial（DETA decoder）は DINO 検出器系統の精度志向、Grounding DINO 1.5 Pro は ViT-L + Grounding DINO 系統の精度志向 → **異なるアプローチで類似の精度を目指す**。

### 開発系譜の整理（IDEA Research 内部）

IDEA Research の open-vocab 検出系譜：

```
[[entities/dino-detector]]（DINO 検出器, 2023）
    ↓
[[entities/grounding-dino]]（Grounding DINO, 2024 ECCV）
    ↓
★ Grounding DINO 1.5 Pro/Edge（本論文, 2024 May）★
    ↓
DINO-X（2024、SOTA 競争継続）
    ↓
Grounded SAM 2、MM-Grounding-DINO（OpenMMLab 拡張版）
```

IDEA Research（清華大学卒業生中心の北京の研究機関）が **open-vocab 検出の中心的開発拠点** として確立。

## 用語と略称

- **Grounding DINO 1.5** = Grounding DINO の Pro/Edge スイート版
- **Pro** = 精度志向版（ViT-L + Grounding-20M）
- **Edge** = 速度志向版（EfficientViT-L1 + efficient feature enhancer）
- **Grounding-20M** = IDEA Research が構築した 20M+ grounding 画像データセット
- **early fusion** = vision-language を encoder 段階で融合（再現率と box 精度高い、hallucination 多い）
- **late fusion** = vision-language を loss 段階のみで融合（hallucination 低い、再現率低い）
- **hallucination** = 画像に存在しない物体を予測する誤検出
- **negative sampling** = 訓練時に「画像に存在しない名詞句」を意図的に与えてモデルに「何も予測しない」ことを学ばせる戦略
- **efficient feature enhancer** = Edge モデルの新ネック（P5 のみ + vanilla self-attn + cross-scale fusion）
- **P3 / P4 / P5** = backbone のマルチスケール特徴のレベル（解像度低い順）
- **Lite-DETR** = [Li et al., 2023] の効率的 DETR、低レベル特徴の有用性が低いことを実証
- **Cross-scale feature fusion** = 異なる解像度の特徴を統合するモジュール（[Zhao et al., RT-DETR, 2023]）
- **EfficientViT-L1** = [Cai et al., 2023] の高速 ViT、Edge backbone
- **Deformable self-attention** = [Zhu et al., Deformable DETR, 2021]、sparse な学習可能 attention
- **Vanilla self-attention** = 通常の Transformer の self-attention（エッジ GPU で実装しやすい）
- **TensorRT** = NVIDIA の推論高速化ツール
- **NVIDIA Orin NX** = NVIDIA のエッジ GPU（ロボティクス・自律走行用）
- **A100** = NVIDIA のデータセンター GPU
- **FPS** = Frames Per Second（推論速度の指標）
- **language cache** = テキスト encoder の出力をキャッシュして毎回再計算しない技法（YOLO-World の prompt-then-detect の派生）
- **ViT-L** = Vision Transformer Large（[[concepts/vision-transformer]]）
- **fixed AP** = [Dave et al., 2021]、LVIS 評価の公平比較プロトコル
- **AP_all / AP_r / AP_c / AP_f** = 全体 / rare / common / frequent カテゴリの AP（LVIS の長尾分布対応）
- **ODinW35** = Object Detection in the Wild の 35 データセット拡張版（Grounding DINO 1.5 で導入）
- **ODinW13** = ODinW の 13 データセット版（GLIP オリジナル）
- **DetCLIPv3** = [Yao et al., NeurIPS 2024]、Swin-L ベースの SOTA だった open-vocab 検出器
- **T-Rex2** = [Jiang et al., 2024]、visual prompt + text prompt のハイブリッド検出器
- **APE** = Aligning and Prompting Everything（[Shen et al., CVPR 2024]）
- **GLEE-Pro** = [Wu et al., CVPR 2024]、segmentation+ 統合モデル
- **MQ-GLIP** = Multi-Query GLIP（[Xu et al., NeurIPS 2023]）
- **OmDet-Turbo** = [Zhao et al., 2024]、language cache を使う高速 open-vocab 検出器
- **OWL-ST** = OWL-ViT の self-training 拡張
- **DINOv** = visual prompt 系 detector
- **V3Det** = [Wang et al., ICCV 2023]、超大規模 vocabulary 検出データセット
- **GranuCap50M** = DetCLIPv3 の grounding データ
- **WebLI** = Google の Web Large-scale Images-text（OWL-ST が使用）
- **SA-1B** = SAM の訓練データ（[[entities/sa-1b]]）

## 関連ページ

- [[translations/grounding-dino-1-5]] — 本文全文の和訳
- [[entities/grounding-dino-1-5]] — モデルの詳細スペック
- [[entities/grounding-dino]] / [[sources/grounding-dino]] — 直接の前身
- [[entities/yolo-world]] / [[sources/yolo-world]] — 速度志向の競合
- [[entities/dino-detector]] / [[sources/dino-detector]] — DETR ファミリーの集大成、Grounding DINO の祖
- [[entities/glip]] / [[sources/glip]] — phrase grounding パラダイムの祖
- [[entities/sam-3]] / [[sources/sam-3]] — より広い PCS タスク（マスク + 動画）
- [[entities/dino-x]] / [[sources/dino-x]] — **直接の後継**（2024 Nov、IDEA Research の unified perception model）
- [[concepts/object-detection]] — 物体検出全体
- [[concepts/zero-shot-transfer]] — Grounding DINO 1.5 のゼロショット転送
- [[concepts/foundation-model]] — open-vocab 検出は CV foundation model の重要分野
- [[concepts/vision-transformer]] — ViT-L backbone の基盤
