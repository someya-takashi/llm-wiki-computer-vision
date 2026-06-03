---
type: source
source_path: raw/papers/DINO-X_ A Unified Vision Model for Open-World Object Detection and Understanding.md
source_kind: paper
title: "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding"
authors: [IDEA Research Team]
year: 2024
venue: arXiv preprint (November 2024)
ingested: 2026-05-28
tags: [dino-x, grounding-dino, open-world-detection, multi-prompt, multi-head, idea-research, grounding-100m, edge-deployment, vit-l, efficientvit, knowledge-distillation, mllm-perception]
translation: "[[translations/dino-x]]"
---

# DINO-X: Open-World 物体検出と理解のための統一視覚モデル

> 原典: [[translations/dino-x]] ・ `raw/papers/DINO-X_ A Unified Vision Model for Open-World Object Detection and Understanding.md`
> 著者: IDEA Research Team（International Digital Economy Academy）
> 出典: arXiv preprint, November 2024 (arXiv 2411.14347)
> arXiv: <https://arxiv.org/abs/2411.14347>
> API: <https://github.com/IDEA-Research/DINO-X-API>
> プロジェクト: <https://deepdataspace.com/home>

## 一言まとめ

**「[[entities/grounding-dino-1-5|Grounding DINO 1.5]] の正統な後継、unified object-centric vision model」**。[[entities/grounding-dino|Grounding DINO]] → [[entities/grounding-dino-1-5|Grounding DINO 1.5]] → Grounding DINO 1.6（中間版）→ **DINO-X** の系譜で、IDEA Research の open-set 検出を **box 出力単一タスクから unified perception model に進化**。**3 種プロンプト**（Text/Visual/Customized）+ **4 種 head**（Box/Mask/Keypoint/Language）+ **CLIP text encoder への切り替え** + **Grounding-100M**（GD 1.5 の Grounding-20M から 5 倍） + **Pro/Edge 双子戦略**。**LVIS rare で APr 63.3**（GD 1.6 Pro から +5.8、長尾物体検出が劇的改善）、**LVIS-mv 59.8 / LVIS-val 52.4 SOTA**。**Edge** は **Knowledge Distillation** + **FP16 量子化**で **Orin NX 20.1 FPS**（GD 1.5 Edge 10.7 から +87%）かつ LVIS-mv 48.3 AP。**universal object prompt で prompt-free detection** という新タスクも実現。

## 背景と問題意識

[[entities/grounding-dino-1-5]]（2024 May）の後、IDEA Research は **Grounding DINO 1.6**（公開論文なし、API のみ）で性能を漸進的に改善していた。しかし以下の限界が残っていた：

| 軸 | 問題 |
|---|---|
| **テキストプロンプトの限界** | 「red shirt person」のように記述しづらいオブジェクトや、訓練データに含まれない長尾カテゴリでは不十分 |
| **単一タスク主義** | Box 出力のみ。マスク、keypoint、キャプションは別モデル（Grounded SAM など）と連携が必要 |
| **MLLM への接続性** | MLLM の hallucination 削減には object-level な認識が重要だが、検出 + 認識 + キャプションを 1 モデルで提供できる例がなかった |
| **prompt の柔軟性** | text-only に固定されていた |

**DINO-X の問題意識**: 「**1 つのモデルで open-world perception の全タスクを統一**」。SAM が **画像 perception 基盤** であるのと並列に、**object-centric な perception 基盤** を目指す。

> **補足: IDEA Research の open-set 検出系譜の整理**
>
> ```
> 2023 ICLR   [[entities/dino-detector|DINO 検出器]]  — DETR ファミリーの集大成（IDEA + HKUST）
>      ECCV   [[entities/grounding-dino|Grounding DINO]]  — GLIP × DINO 検出器（IDEA + Microsoft）
> 2024 May    [[entities/grounding-dino-1-5|Grounding DINO 1.5]]  — Pro/Edge 双子、Grounding-20M（IDEA Research）
>      ?      Grounding DINO 1.6  — 中間バージョン、API のみで公開（IDEA Research）
>      Nov    ★ DINO-X ★  — 統一 perception model、Grounding-100M（IDEA Research）
> ```

## 提案手法 / 主張

### 全体構造

```
                          ┌──────────────────────┐
                          │  3 種プロンプト入力   │
                          ├──────────────────────┤
                          │ Text (CLIP encoder)  │
                          │ Visual (T-Rex2 流)   │
                          │ Customized (tunable) │
                          │   └─ Universal Obj. │
                          │      → prompt-free   │
                          └──────────┬───────────┘
                                     ↓
画像 → ViT (Pro) / EfficientViT-L2 (Edge) → Deep Early Fusion ←┘
                                     ↓
                          Object-level 表現
                                     ↓
                  ┌──────┬──────┬──────┬──────┐
                  ↓      ↓      ↓      ↓
              Box      Mask  Keypoint Language
              Head     Head    Head    Head
                ↓        ↓       ↓       ↓
              (bbox)  (mask) (keypoint) (caption)
```

### 鍵 1: 3 種プロンプト

#### Text Prompt（CLIP encoder への切り替え）

[[entities/grounding-dino]] / [[entities/grounding-dino-1-5]] は BERT を使用していたが、DINO-X は **CLIP** を採用：

- **理由**: BERT は **テキストのみで訓練** → open-world 検出の multi-modal 整列に最適化されていない
- **CLIP は image-text contrastive で事前学習** → 視覚-意味的整列の能力が高い
- **[[entities/yolo-world]] が CLIP を採用したのと同じ流れ** — 「open-vocab 検出の鍵は CLIP の image-text 整合性」という観察を IDEA も追認

#### Visual Prompt（T-Rex2 由来）

box または point フォーマットの **ユーザー定義 visual prompt**:

- sine-cosine 位置埋め込み → 統一特徴空間に射影
- multi-scale deformable cross-attention で画像特徴と融合
- **テキストでは記述しづらい物体**（例: 「あの特殊な部品」）の検出に有効

#### Customized Prompt（prompt-tuning 可能、prompt-free の鍵）

**事前定義または学習可能な prompt embedding**:

- **Domain-customized**: 医療画像、衛星画像など特定ドメイン用
- **Function-specific**: カウント、計測など特定機能用
- ★ **Universal Object Prompt** ★: 「**画像内の任意の物体**」を表す学習済み prompt → **prompt-free 検出**

> **補足: prompt-free 検出の革新性** — GLIP / Grounding DINO / Grounding DINO 1.5 はすべて「テキストプロンプト必須」。DINO-X は universal object prompt を **事前学習済みの固定 embedding として埋め込む** ことで、推論時にプロンプトなしで「すべての物体を出力」が可能に。画面解析、汎用カメラなど **対話的でない応用** に重要。

### 鍵 2: 4 種 Perception Head

#### Box Head

[[entities/grounding-dino]] と同じ language-guided query selection + L1 + GIoU + contrastive loss。

#### Mask Head（Mask2Former 流）

- **1/4 解像度 backbone 特徴 + 1/8 解像度 encoder 特徴を融合**して pixel embedding map を構築
- object query との **dot-product でマスク予測**
- **Mask2Former** / **Mask DINO** のコア設計を継承
- 計算効率のため、サンプリングされた点でのみ mask 損失計算
- 訓練データは **SAM / SAM 2 で疑似マスク生成** された Grounding-100M のサブセット

#### Keypoint Head（ED-Pose 流の簡略版）

- **person 17 keypoints + hand 21 keypoints の 2 つ**
- 各検出ボックスをクエリとして扱い、deformable Transformer decoder で keypoint 位置と可視性を予測
- **ED-Pose** の物体検出ロジック部分を削除した簡略実装
- COCO + CrowdPose + Human-Art + HInt + OneHand10K で訓練

#### Language Head（OPT-125M、軽量で強力）

- **凍結された DINO-X backbone** から RoIAlign で領域特徴を抽出
- object tokens（query embedding と連結）→ 線形射影でテキスト埋め込み空間に整列
- **task tokens**（学習可能、タスク識別子）+ object tokens を OPT-125M に入力
- **自己回帰的にテキスト生成**：物体認識、領域キャプション、OCR、region-based VQA

**重要**: OPT-125M はわずか 125M パラメータ。Osprey の **Vicuna-7B (7B)** と比べて **1% の規模で同等以上の性能**（LVIS-val SS 71.25 vs Osprey 65.24）。**「軽量 language head + 強力な visual backbone」が region understanding の理想的構成** であることを実証。

### 鍵 3: Grounding-100M

**100M+ 画像**の grounding データセット：

- [[entities/grounding-dino-1-5|Grounding DINO 1.5]] の **Grounding-20M から 5 倍**
- web からの収集 + T-Rex2 訓練データ + 産業シナリオデータ
- **マスクヘッド訓練**: SAM / SAM 2 で疑似マスク注釈生成
- **Prompt-free 訓練**: 高品質サブセット
- **Language head 訓練**: 10M+ region understanding データ（物体認識、領域キャプション、OCR、region-level QA）

> **補足: 公開度** — Grounding-100M の詳細は非公開（IDEA Research の商用データエンジン）。Grounding-20M と同様、API/モデル重みは IDEA の DeepDataSpace プラットフォーム経由。

### 鍵 4: 2 段階訓練戦略

```
[Stage 1] 共同事前学習（grounding 中心）
  ├── text-prompt detection
  ├── visual-prompt detection
  └── object segmentation
  ※ COCO / LVIS / V3Det は意図的に除外（ゼロショット評価のため）

[Stage 2] backbone 凍結 + head 追加
  ├── pose head（person + hand）
  ├── language head
  └── universal object prompt（prompt-tuning で訓練）
```

**目的**:
1. コア grounding 能力を新機能追加で損なわない
2. 大規模 grounding 事前学習が **object-centric 基盤** として機能することを検証

### 鍵 5: DINO-X Edge の追加工夫

[[entities/grounding-dino-1-5|Grounding DINO 1.5 Edge]] からの 3 つの改善：

#### 1. CLIP Text Encoder（BERT → CLIP）

Pro と同じテキストエンコーダ → text embedding を事前計算可能なので推論速度に影響なし、精度向上。

#### 2. Knowledge Distillation（Pro → Edge）

- **Feature ベース蒸留**: Pro と Edge の中間特徴を整列
- **Response ベース蒸留**: Pro と Edge の予測 logits を整列
- Grounding DINO 1.6 Edge より強いゼロショット能力

#### 3. FP16 Inference（正規化技法）

- **浮動小数点乗算の正規化**で精度を保ったまま FP16 量子化
- **Orin NX 20.1 FPS**（GD 1.6 Edge 15.1 から +33%、GD 1.5 Edge 10.7 から +87%）

#### Backbone の強化

- [[entities/grounding-dino-1-5|GD 1.5/1.6 Edge]] は EfficientViT-**L1**
- DINO-X Edge は **EfficientViT-L2**（より強力）

## 実験結果と知見

### LVIS / COCO ゼロショット検出（表 1）

| Model | Backbone | COCO Box | LVIS-mv all | **LVIS-mv APr** | LVIS-val all | **LVIS-val APr** |
|---|---|---|---|---|---|---|
| DetCLIPv3 | Swin-L | - | 48.8 | 49.9 | 41.4 | 41.4 |
| Grounding DINO 1.5 Pro | ViT-L | 54.3 | 55.7 | 56.1 | 47.6 | 44.6 |
| Grounding DINO 1.6 Pro | ViT-L | 55.4 | 57.7 | 57.5 | 51.1 | 51.5 |
| **DINO-X Pro** | ViT-L | **56.0** | **59.8** | **63.3 (+5.8)** | **52.4** | **56.5 (+5.0)** |

**長尾改善が劇的**: LVIS-mv APr で **+5.8 AP over GD 1.6 Pro**、**+7.2 AP over GD 1.5 Pro**。

### Few-shot 物体カウント（表 2）

| Method | FSC147 MAE | FSCD-LVIS AP |
|---|---|---|
| T-Rex2 | 10.9 | 43.4 |
| **DINO-X Pro** | **5.6** | **44.8** |

T-Rex2 を MAE で **約半分**（visual prompt 検出の精度大幅向上）。

### Keypoint 検出（表 3, 4）

| Benchmark | Best baseline | **DINO-X Pro** |
|---|---|---|
| COCO AP | ED-Pose 75.8 | 74.4 |
| **CrowdPose AP** | ED-Pose 76.6 | **80.0 (+3.4)** |
| **Human-Art AP** | ED-Pose 72.3 | **74.1 (+1.8)** |
| **HInt All joints** | HaMeR 51.6 / 56.5 / 46.9 | **54.3 / 63.0 / 66.0** |

専門モデルを上回る性能（COCO のみ ED-Pose に劣るが head パラメータ制限のため）。

### Object Recognition（表 5、LVIS SS）

| Model | Language Decoder | LVIS SS | LVIS S-IoU |
|---|---|---|---|
| Osprey | Vicuna-7B | 65.24 | 38.19 |
| **DINO-X Pro** | **OPT-125M** | **71.25 (+6.01)** | **41.15** |

**7B → 125M で +6.01 SS** — 軽量 language head の優位性。

### Region Captioning（表 6、Visual Genome）

| Model | CIDEr |
|---|---|
| GRIT | 142.0 |
| AlphaCLIP | 160.3 |
| **DINO-X Pro (zero-shot)** | 143.2 |
| **DINO-X Pro (fine-tuned)** | **201.8** |

**fine-tune で 201.8 CIDEr（SOTA）**。GRIT、AlphaCLIP、Vicuna-7B 系すべてを大幅超え。

### Edge モデル（表 7）

| Model | Backbone | Test | COCO | LVIS-mv | Orin NX FP16 FPS |
|---|---|---|---|---|---|
| YOLO-Worldv2-L | YOLOv8-L | 640 | - | 33.0 | - |
| GD 1.5 Edge | EfficientViT-L1 | 640 | 42.9 | 33.5 | - |
| GD 1.5 Edge | EfficientViT-L1 | 800 | 45.0 | 36.2 | - |
| GD 1.6 Edge | EfficientViT-L1 | 800 | 44.8 | 36.9 | 15.1 |
| GD 1.6 Edge | EfficientViT-L1 | 1024 | 46.5 | 40.1 | 10.5 |
| **DINO-X Edge** | EfficientViT-L2 | 640 | **48.7** | **44.5** | **20.1** |
| **DINO-X Edge** | EfficientViT-L2 | 800 | **50.9** | **48.3** | 9.1 |

**Edge モデルが GD 1.5 Pro のテキスト性能（54.3 / 55.7）に近づきつつ、Orin NX 20 FPS のリアルタイム性**。Edge は **YOLO-Worldv2-L (33.0) や OmDet-Turbo-T (30.3) を大差で凌駕**。

## 限界・批判的視点

- **モデル重みは API 経由のみ**: GD 1.5 と同様、完全オープンソース化されていない（IDEA Research の商用化戦略）
- **Grounding-100M の詳細非公開**: 学術コミュニティでの完全再現困難
- **PACO で Osprey に劣る**: 訓練データに PACO 含まれず、part-level 認識は専用モデルに分がある
- **COCO keypoint で ED-Pose に劣る**: 軽量 pose head の制約
- **マスクは Grounded SAM (1.5 Pro + Huge) より低い**: LVIS-mv mask AP 43.8 vs Grounded SAM 47.7。「複雑な統一 vs 専門特化」の trade-off
- **Grounding DINO 1.6 が中間版で論文未公開**: 「+5.8 AP over GD 1.6 Pro」と比較するが、GD 1.6 の詳細が公開されていない
- **MLLM 連携の前提**: 「MLLM の hallucination 削減」は重要な動機だが、本論文では MLLM との具体的統合実験はない（応用提案のみ）

## 既存 wiki との接続

### IDEA Research の正統な系譜

[[entities/dino-detector]] → [[entities/grounding-dino]] → [[entities/grounding-dino-1-5]] → **DINO-X** という **IDEA Research の open-set 検出系譜の現時点最新版**。

### SAM 3 (PCS) との対比

[[entities/sam-3]]（Meta, 2025）の PCS（Promptable Concept Segmentation）と **目的が似ている**：

| 軸 | **DINO-X** | **[[entities/sam-3]]** |
|---|---|---|
| 著者 | IDEA Research | Meta Superintelligence Labs |
| タスク | unified perception（detection + seg + pose + caption） | PCS（concept segmentation） |
| Backbone | ViT-L | Perception Encoder（PE） |
| Text encoder | CLIP | PE のテキスト塔 |
| Visual prompt | box + point（T-Rex2 流） | 画像 exemplar |
| マスク head | Mask2Former 流 | MaskFormer 適応 |
| 認識/位置分離 | 別 head（box + language） | **presence head**（核心革新） |
| 動画対応 | なし | あり（SAM 2 tracker） |
| 公開度 | API のみ | 公開予定 |

**両者は概念的に並行進化**しているが、設計選択が異なる:
- DINO-X = **multi-head + multi-prompt 統合**
- SAM 3 = **presence head + 対話性 + 動画**

### YOLO-World との対比

[[entities/yolo-world]] の **real-time open-vocab** に対し、DINO-X Edge は **「Transformer 系の real-time open-vocab + multi-task」**:

| 軸 | YOLO-Worldv2-L | **DINO-X Edge** |
|---|---|---|
| Backbone | YOLOv8-L (CNN) | EfficientViT-L2 (Transformer) |
| Text encoder | CLIP（凍結） | CLIP |
| Heads | Box のみ | **Box + Mask + Keypoint + Language** |
| LVIS-mv ZS | 33.0 | **48.3 (+15.3)** |
| Edge GPU | - | **Orin NX 20.1 FPS FP16** |

DINO-X Edge は **YOLO-World を精度・機能・速度（Orin NX）すべてで上回る**。

### Grounded SAM への代替

[[entities/sam]] 系統では「**Grounding DINO で検出 → SAM でマスク**」というパイプライン（**Grounded SAM**）が定番だった。DINO-X は **これを単一モデル化** する：

| 軸 | Grounded SAM (1.5 Pro + Huge) | **DINO-X Pro** |
|---|---|---|
| 推論ステップ | 2（検出 → セグメント） | **1（統一）** |
| LVIS-mv mask AP | 47.7（SAM-Huge） | 43.8（差は残るが効率大） |
| LVIS-val mask AP | 41.8 | 38.5 |

**精度ではまだ Grounded SAM に劣る** が、**推論効率と統一性で実用優位**。

### GLIP / Grounding DINO 系統の到達点

[[concepts/object-detection]] の open-vocab 系譜：

```
2021  ViLD / MDETR — CLIP 蒸留 / DETR + テキスト条件
2022  GLIP        — 検出 = phrase grounding 統一
2023  Grounding DINO — DINO 検出器 + GLIP
2024  Grounding DINO 1.5 — Pro/Edge スイート
2024  Grounding DINO 1.6 — 中間版（API のみ）
2024  ★ DINO-X ★      — unified perception model（本論文）
2025  SAM 3            — PCS + 動画 + presence head
```

**DINO-X は「Grounding DINO 系統の到達点」** であり、IDEA Research が **box 出力単一タスクから unified perception model へ移行を完了**した節目。

## 用語と略称

- **DINO-X** = IDEA Research の unified object-centric vision model（2024 Nov）
- **Pro / Edge** = 精度志向 / エッジデプロイメント志向の双子モデル（GD 1.5 から継承）
- **Grounding-100M** = 100M+ grounding 画像、Grounding-20M から 5 倍
- **3 種プロンプト** = Text + Visual + Customized
- **Universal Object Prompt** = 「画像内の任意の物体」を表す事前学習 prompt embedding、prompt-free 検出の鍵
- **prompt-free detection** = ユーザー入力なしで全物体検出
- **prompt-tuning** = カスタマイズ可能な prompt embedding の学習手法（[Lester et al., 2021]）
- **4 種 perception head** = Box + Mask + Keypoint + Language
- **Box Head** = Grounding DINO 流の language-guided query selection + GIoU
- **Mask Head** = Mask2Former 流（pixel embedding × object query の内積）
- **Mask2Former / Mask DINO** = 統一マスクアーキテクチャ（Cheng et al., CVPR 2022 / Li et al., CVPR 2023）
- **Keypoint Head** = ED-Pose 流の簡略版（person 17 + hand 21 keypoints）
- **ED-Pose** = End-to-end Detection Pose（Yang et al., 2023）
- **Language Head** = OPT-125M ベースの軽量自己回帰生成
- **OPT** = Open Pre-trained Transformer（Zhang et al., 2022、Meta の軽量言語モデル）
- **task tokens** = タスク識別子として language decoder に与える学習可能 token
- **object tokens** = RoIAlign で抽出された領域特徴
- **CLIP text encoder** = BERT に代わる multimodal 整列済みテキストエンコーダ（YOLO-World と同じ選択）
- **T-Rex2** = visual prompt 検出の祖（[Jiang et al., 2024]）
- **deep early fusion** = encoder 段階での vision-language 融合（Grounding DINO 系統の伝統）
- **Knowledge Distillation** = Pro → Edge への蒸留（feature + response based）
- **feature-based distillation** = 中間特徴を Pro と Edge で整列
- **response-based distillation** = 予測 logits を Pro と Edge で整列
- **EfficientViT-L1 / L2** = MIT EfficientViT（Cai et al., 2023）、L2 は L1 の強化版
- **FP16 量子化** = 浮動小数点乗算の正規化技法による精度保持 FP16 推論
- **Grounding DINO 1.6** = GD 1.5 と DINO-X の中間版、論文未公開で API のみで存在
- **Grounded SAM** = Grounding DINO + SAM のパイプライン（IDEA, 2024）
- **Grounded SAM 2** = Grounded SAM + SAM 2 の動画版
- **Mean Absolute Error (MAE)** = カウントタスクの平均絶対誤差
- **RMSE** = Root Mean Squared Error
- **FSC147 / FSCD-LVIS** = few-shot counting ベンチマーク
- **CrowdPose / Human-Art** = 困難な姿勢推定ベンチマーク（混雑、芸術的表現）
- **HInt** = Hand In-the-wild ベンチマーク
- **PCK@0.05** = Percentage of Correctly Localized Keypoints @ 5% box size
- **OKS** = Object Keypoint Similarity（COCO 標準）
- **Semantic Similarity (SS)** / **Semantic-IoU (S-IoU)** = vocabulary-free 物体認識評価指標
- **CIDEr / METEOR** = 領域キャプション評価指標
- **Osprey** = ConvNeXt-L + Vicuna-7B の region understanding モデル
- **GPT4RoI / Shikra / Ferret / Kosmos-2 / AlphaCLIP / GRIT** = 関連 region MLLM
- **Visual Genome / RefCOCOg / PACO** = 領域 / part 認識データセット
- **V3Det** = 大規模 vocabulary 検出データセット（ICCV 2023）
- **RoIAlign** = 領域特徴抽出演算子（Mask R-CNN 由来）
- **DetCLIPv3** = DetCLIP 系の最新（Swin-L、NeurIPS 2024）
- **Hamba / HaMeR** = hand pose の SOTA 手法
- **Osprey の Vicuna-7B vs DINO-X の OPT-125M** = 7B → 125M で同等以上の性能、軽量化の好例

## 関連ページ

- [[translations/dino-x]] — 本文全文の和訳
- [[entities/dino-x]] — モデルの詳細スペック
- [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — 直接の前身
- [[entities/grounding-dino]] / [[sources/grounding-dino]] — IDEA Research 系譜の祖
- [[entities/dino-detector]] / [[sources/dino-detector]] — DETR ファミリーの集大成
- [[entities/yolo-world]] / [[sources/yolo-world]] — real-time open-vocab、Edge と直接対比
- [[entities/glip]] / [[sources/glip]] — phrase grounding パラダイムの祖
- [[entities/sam-3]] / [[sources/sam-3]] — PCS、平行進化する unified perception
- [[entities/sam]] / [[entities/sam-2]] — Grounded SAM の構成要素、DINO-X が代替
- [[entities/clip]] / [[sources/clip]] — DINO-X の text encoder
- [[concepts/object-detection]] — 物体検出全体
- [[concepts/zero-shot-transfer]] — DINO-X のゼロショット転送
- [[concepts/foundation-model]] — DINO-X は object-centric vision foundation model
- [[concepts/vision-transformer]] — ViT-L backbone の基盤
