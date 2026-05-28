---
type: entity
entity_kind: model
aliases: [DINO-X, DINO X, DINO-X Pro, DINO-X Edge, DinoX]
tags: [object-detection, open-world-detection, grounding-dino, unified-perception, multi-prompt, multi-head, idea-research, edge-deployment, vit-l, efficientvit, mllm-perception]
related: [[concepts/object-detection]], [[concepts/zero-shot-transfer]], [[concepts/foundation-model]], [[concepts/vision-transformer]]
sources: [[sources/dino-x]]
updated: 2026-05-28
---

# DINO-X

## 概要

**DINO-X** は IDEA Research が 2024 年 11 月に公開した **統一 object-centric vision model**。[[entities/grounding-dino]] → [[entities/grounding-dino-1-5]] → **DINO-X** という IDEA の系譜で、**box 出力単一タスクから unified perception model に進化**。**3 種プロンプト**（Text/Visual/Customized）+ **4 種 perception head**（Box/Mask/Keypoint/Language）+ **CLIP text encoder** + **Grounding-100M データセット** + **Pro/Edge 双子戦略**。**LVIS rare APr 63.3 (+5.8 over GD 1.6 Pro)** の長尾改善、**Orin NX 20.1 FPS FP16** の Edge デプロイメントを同時実現。

---

## 論文情報

- 論文: "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding"
- arXiv: 2411.14347 / preprint November 2024
- API リポジトリ: <https://github.com/IDEA-Research/DINO-X-API>
- プロジェクト: <https://deepdataspace.com/home>
- メーカー: International Digital Economy Academy (IDEA) Research（北京）
- 著作権・ライセンス: API は無料利用可（商用利用は要確認）

---

## モデルバリアント

### DINO-X Pro

| 項目 | 値 |
|---|---|
| 画像 backbone | **ViT-L**（事前学習済み） |
| テキスト encoder | **CLIP**（BERT から変更） |
| Visual prompt encoder | **T-Rex2 流**（box + point、multi-scale deformable cross-attn） |
| Customized prompt | **prompt-tuning 可能な学習可能 embedding** |
| 訓練データ | **Grounding-100M**（100M+ grounding 画像） |
| Perception heads | **Box + Mask + Keypoint + Language**（4 種） |
| 用途 | unified perception SOTA |

### DINO-X Edge

| 項目 | 値 |
|---|---|
| 画像 backbone | **EfficientViT-L2**（L1 から強化） |
| テキスト encoder | **CLIP**（Pro と同じ） |
| 訓練 | **Knowledge Distillation from Pro** + Grounding-100M |
| 推論最適化 | **FP16 量子化**（浮動小数点乗算正規化技法） |
| Perception heads | Box（メイン）、他は Pro と共有可能 |
| 用途 | エッジデバイス（Orin NX）でのリアルタイム unified perception |

---

## アーキテクチャ詳細

### 全体構造

```
                              ┌─────────────────────────────┐
                              │  3 種プロンプト入力          │
                              ├─────────────────────────────┤
                              │ Text (CLIP encoder)         │
                              │ Visual (T-Rex2 流、box/point)│
                              │ Customized (prompt-tuning)  │
                              │   └─ Universal Obj. Prompt  │
                              │      → prompt-free 検出      │
                              └─────────────┬───────────────┘
                                            ↓
画像 → ViT-L (Pro) / EfficientViT-L2 (Edge) → Deep Early Fusion ←┘
                                            ↓
                             Object-level 表現（Transformer encoder 出力）
                                            ↓
                             Language-Guided Query Selection
                                            ↓
                             Transformer Decoder（layer-by-layer 更新）
                                            ↓
                  ┌─────────┬─────────┬─────────┬─────────┐
                  ↓         ↓         ↓         ↓
              Box Head  Mask Head  Keypoint  Language
                                    Head      Head (OPT-125M)
                  ↓         ↓         ↓         ↓
                (bbox)    (mask)  (keypoint) (caption/QA)
```

### 3 種プロンプト詳細

#### Text Prompt（CLIP encoder）

- BERT（GD 1.5 まで使用）→ **CLIP** に切り替え
- **理由**: BERT はテキストのみ訓練、CLIP は image-text contrastive で multi-modal 整列済み
- [[entities/yolo-world]] と同じ流れ — open-vocab 検出には CLIP の image-text 整合性が重要

#### Visual Prompt（T-Rex2 由来）

```python
# box / point プロンプトの処理
prompts = (box_or_point_prompts)
position_emb = sine_cosine_layer(prompts)
unified_features = linear_projection(position_emb)
# box / point は異なる線形射影で区別
visual_prompt_features = multi_scale_deformable_cross_attention(
    queries=unified_features,
    keys=image_features  # multi-scale
)
```

#### Customized Prompt + Universal Object Prompt

```python
# 事前学習された universal object prompt
universal_obj_prompt = learned_embedding(shape=(num_queries, D))
# Prompt-free 検出: 入力なしで全物体検出
detections = dino_x(image, prompt=universal_obj_prompt)
```

**Universal Object Prompt の意義**: GLIP/Grounding DINO/GD 1.5 はテキストプロンプト必須だったが、DINO-X は **prompt-tuning で「すべての物体」を表す固定 embedding を学習** → 推論時にユーザー入力不要。**画面解析、汎用カメラ等の対話的でない応用** に重要。

### 4 種 Perception Head

#### Box Head

```
language-guided query selection（[[entities/grounding-dino]] 由来）
  → Transformer decoder（layer-by-layer 更新）
  → MLP → bbox coordinates
損失: L1 + GIoU + contrastive（query-prompt 整列）
```

#### Mask Head（Mask2Former / Mask DINO 流）

```
1/4 解像度 backbone 特徴 + 1/8 解像度 encoder 特徴
  → 融合して pixel embedding map
  → 各 object query との dot-product
  → mask 出力
```

**訓練効率化**: サンプリングされた点でのみ mask 損失計算（Mask R-CNN 流）。

**訓練データ**: SAM / SAM 2 で疑似マスク生成された Grounding-100M のサブセット。

#### Keypoint Head（ED-Pose 簡略版）

- **2 つの head**: person（17 keypoints）+ hand（21 keypoints）
- 各検出ボックスをクエリとして拡張
- **Deformable Transformer decoder** で keypoint 位置 + 可視性を予測
- 訓練: COCO + CrowdPose + Human-Art + HInt + OneHand10K

#### Language Head（OPT-125M、軽量自己回帰）

```
凍結 DINO-X backbone
  ↓ RoIAlign（領域特徴抽出）
object tokens（領域特徴 + query embedding）
  ↓ 線形射影（テキスト埋め込み次元に整列）
[task_tokens, object_tokens] → OPT-125M decoder
  ↓ 自己回帰生成
caption / recognition / VQA / OCR
```

**Task Tokens の役割**: タスク識別子として decoder に与え、1 つの decoder で複数タスクに対応。

**OPT-125M の特筆点**: わずか 125M パラメータで Osprey の Vicuna-7B（7B）と同等以上（LVIS-val SS 71.25 vs 65.24）。**軽量 language head + 強力な visual backbone** の理想構成。

---

## 訓練設定

### 2 段階訓練戦略

**Stage 1**: grounding 中心の共同事前学習
- text-prompt detection
- visual-prompt detection
- object segmentation
- **COCO / LVIS / V3Det は意図的に除外**（ゼロショット評価のため）

**Stage 2**: backbone 凍結 + head 追加
- pose head（person + hand、別々に訓練）
- language head（OPT-125M、別々に訓練）
- **universal object prompt**（prompt-tuning で訓練、prompt-free 検出を可能に）

**意義**:
1. コア grounding 能力を新機能追加で損なわない
2. 大規模 grounding 事前学習が object-centric 基盤として機能できることを検証

### DINO-X Edge の追加最適化

1. **Stronger Text Prompt Encoder**: BERT → CLIP（Pro と同じ）
2. **Knowledge Distillation**: Pro → Edge
   - Feature ベース蒸留（中間特徴整列）
   - Response ベース蒸留（予測 logits 整列）
3. **Improved FP16 Inference**: 浮動小数点乗算の正規化技法で精度保持 FP16

---

## 訓練データセット

### Grounding-100M（核心）

| 用途 | サイズ | 内容 |
|---|---|---|
| Box head 訓練 | 100M+ | web 収集 + T-Rex2 訓練データ + 産業シナリオ |
| Mask head 訓練 | サブセット | SAM / SAM 2 で疑似マスク生成 |
| Prompt-free 検出 | 高品質サブセット | universal object prompt 学習用 |
| Language head 訓練 | 10M+ | 物体認識、領域キャプション、OCR、region-level QA |

**[[entities/grounding-dino-1-5|Grounding DINO 1.5]] の Grounding-20M から 5 倍** に拡張。

---

## 主要結果

### LVIS / COCO ゼロショット検出（表 1）

| Model | Backbone | COCO Box | LVIS-mv all | **LVIS-mv APr** | LVIS-val all | **LVIS-val APr** |
|---|---|---|---|---|---|---|
| DetCLIPv3 | Swin-L | - | 48.8 | 49.9 | 41.4 | 41.4 |
| T-Rex2 (text) | Swin-L | 52.2 | 54.9 | 49.2 | 45.8 | 42.7 |
| Grounding DINO 1.5 Pro | ViT-L | 54.3 | 55.7 | 56.1 | 47.6 | 44.6 |
| Grounding DINO 1.6 Pro | ViT-L | 55.4 | 57.7 | 57.5 | 51.1 | 51.5 |
| **DINO-X Pro** | **ViT-L** | **56.0** | **59.8** | **63.3 (+5.8)** | **52.4** | **56.5 (+5.0)** |

**重要な観察**: 長尾改善が劇的（LVIS-mv APr +5.8 / LVIS-val APr +5.0 over GD 1.6 Pro、+7.2 / +11.9 over GD 1.5 Pro）。

### Mask AP（同じ表）

| Model | COCO mask | LVIS-mv mask | LVIS-val mask |
|---|---|---|---|
| Grounded SAM (1.5 Pro + Huge) | 44.3 | **47.7** | **41.8** |
| Grounded SAM 2 (1.5 Pro + Large) | 44.7 | 46.2 | 40.5 |
| **DINO-X Pro** | **37.9** | 43.8 | 38.5 |

**精度では Grounded SAM に劣る** が、**統一モデルの効率優位**（複数の複雑な推論ステップ不要）。

### 物体カウント（表 2）

| Method | FSC147 MAE | FSCD-LVIS AP |
|---|---|---|
| T-Rex2 | 10.9 | 43.4 |
| **DINO-X Pro** | **5.6** | **44.8** |

MAE で **T-Rex2 の半分**（5.6 vs 10.9）。

### Keypoint 検出（表 3, 4）

| Benchmark | Best baseline | **DINO-X Pro** |
|---|---|---|
| COCO AP | ED-Pose 75.8 | 74.4 |
| **CrowdPose AP** | ED-Pose 76.6 | **80.0 (+3.4)** |
| **Human-Art AP** | ED-Pose 72.3 | **74.1 (+1.8)** |
| **HInt All joints (Ego4D)** | HaMeR 46.9 | **66.0 (+19.1)** |

専門モデルを上回る（COCO のみ pose head パラメータ制約で僅差）。

### Object Recognition / Captioning（表 5, 6）

| Benchmark | Best baseline | **DINO-X Pro** |
|---|---|---|
| LVIS SS | Osprey (7B) 65.24 | **71.25 (+6.01)** |
| LVIS S-IoU | Osprey 38.19 | **41.15 (+2.96)** |
| Visual Genome CIDEr (ZS) | GRIT 142.0 | 143.2 |
| **Visual Genome CIDEr (FT)** | AlphaCLIP 160.3 | **201.8 SOTA** |

**OPT-125M で Vicuna-7B 系を上回る軽量化の成功例**。

### Edge モデル（表 7）

| Model | Backbone | Test | COCO | LVIS-mv | LVIS-mv APr | A100 TRT FPS | Orin NX FP16 FPS |
|---|---|---|---|---|---|---|---|
| YOLO-Worldv2-L | YOLOv8-L | 640 | - | 33.0 | 22.6 | - | - |
| GD 1.5 Edge | EfficientViT-L1 | 640 | 42.9 | 33.5 | 28.0 | 111.6 | 10.7 (FP32) |
| GD 1.5 Edge | EfficientViT-L1 | 800 | 45.0 | 36.2 | 33.2 | 75.2 | 5.5 (FP32) |
| GD 1.6 Edge | EfficientViT-L1 | 800 | 44.8 | 36.9 | 34.6 | 152.7 | 15.1 |
| GD 1.6 Edge | EfficientViT-L1 | 1024 | 46.5 | 40.1 | 36.8 | 108.1 | 10.5 |
| **DINO-X Edge** | EfficientViT-L2 | 640 | **48.7** | **44.5** | **41.4** | **138.6** | **20.1** |
| **DINO-X Edge** | EfficientViT-L2 | 800 | **50.9** | **48.3** | **47.6** | **74.5** | **9.1** |

**Edge モデルの圧倒的優位**:
- LVIS-mv で **YOLO-Worldv2-L (33.0) を +15.3 AP 上回る**
- LVIS-mv で **GD 1.6 Edge (36.9) を +7.6 AP 上回る**（640 解像度）
- LVIS rare APr で **+12.5 over GD 1.6 Edge (28.0 → 41.4)**
- Orin NX FP16 で **20.1 FPS**（GD 1.6 Edge 15.1 から +33%、GD 1.5 Edge 10.7 から +87%）

---

## アーキテクチャ比較

### Grounding DINO 1.5 → DINO-X の変更点

| 軸 | Grounding DINO 1.5 Pro | **DINO-X Pro** |
|---|---|---|
| 画像 backbone | ViT-L | ViT-L（同じ） |
| Text encoder | **BERT-base** | **CLIP**（変更） |
| Visual prompt | ❌ | ✅（T-Rex2 流） |
| Customized prompt | ❌ | ✅（prompt-tuning） |
| Prompt-free 検出 | ❌ | ✅（universal object prompt） |
| Heads | Box のみ | **Box + Mask + Keypoint + Language**（4 種） |
| 訓練データ | Grounding-20M | **Grounding-100M**（5×） |
| LVIS-mv ZS | 55.7 | **59.8 (+4.1)** |
| LVIS rare APr | 56.1 | **63.3 (+7.2)** |

### DINO-X Edge vs YOLO-Worldv2-L

| 軸 | YOLO-Worldv2-L | **DINO-X Edge** |
|---|---|---|
| Backbone | YOLOv8-L (CNN) | EfficientViT-L2 (Transformer) |
| Text encoder | CLIP（凍結） | CLIP |
| Knowledge Distillation | ❌ | ✅（Pro → Edge） |
| Heads | Box のみ | **Box + Mask + Keypoint + Language** |
| LVIS-mv ZS | 33.0 | **48.3 (+15.3)** |
| LVIS rare APr | 22.6 | **47.6 (+25.0)** |
| A100 TRT FPS | - | **74.5-138.6** |
| Orin NX FP16 | - | **9.1-20.1 FPS** |

**Edge モデルが YOLO-World を精度・機能・速度すべてで凌駕**。

### DINO-X vs SAM 3（並行進化する unified perception）

| 軸 | **DINO-X** | **[[entities/sam-3]]** |
|---|---|---|
| 著者 | IDEA Research | Meta Superintelligence Labs |
| 出版 | 2024 Nov | 2025 |
| タスク | unified perception | PCS（Promptable Concept Segmentation） |
| Backbone | ViT-L | Perception Encoder |
| Text encoder | CLIP | PE のテキスト塔 |
| Visual prompt | box + point | 画像 exemplar |
| Heads | 4 種（Box+Mask+Keypoint+Language） | Box + Mask + Tracker |
| 認識/位置分離 | 別 head | **presence head**（独自革新） |
| 動画対応 | ❌ | ✅（SAM 2 tracker） |
| Prompt-free | ✅（universal obj） | ❌ |
| Pose estimation | ✅（person + hand） | ❌ |
| Region understanding | ✅（language head） | ❌ |

**両者は概念的に並行進化**しているが設計選択が異なる:
- DINO-X = **multi-head + multi-prompt 統合**（より広い範囲）
- SAM 3 = **presence head + 対話性 + 動画**（より深い特化）

---

## 主要な貢献まとめ

1. **「Grounding DINO 系統の到達点」**: box 出力単一タスクから unified perception model への進化を完了
2. **3 種プロンプト + 4 種 head の統合設計**: text + visual + customized prompts、box + mask + keypoint + language heads
3. **Universal Object Prompt による prompt-free 検出**: GLIP 以来「プロンプト必須」だった open-vocab 検出に新オプション
4. **CLIP text encoder への切り替え**: GD 1.5 までの BERT から、YOLO-World と同じ流れで CLIP に
5. **Grounding-100M**: GD 1.5 の 20M から **5 倍**、長尾物体検出能力の劇的改善（LVIS APr +5.8-7.2）
6. **軽量 language head（OPT-125M）の強力さ**: Osprey の Vicuna-7B（7B）を **1/56 のサイズで上回る**
7. **2 段階訓練**: backbone を grounding 事前学習で固め、新 head を追加しても能力を損なわない
8. **DINO-X Edge の三重最適化**: CLIP encoder + KD + FP16 で Orin NX 20.1 FPS（GD 1.5 Edge から +87%）
9. **Region captioning SOTA**: Visual Genome CIDEr 201.8（fine-tuned、軽量 head で AlphaCLIP 160.3 を超える）
10. **MLLM 連携の基盤**: object-level な認識能力が MLLM の hallucination 削減に寄与する応用提案

---

## 限界と批判

- **モデル重みは API 経由のみ**: GD 1.5 と同様、完全オープンソースではない
- **Grounding-100M の詳細非公開**: IDEA Research の商用データエンジン
- **PACO で Osprey に劣る**: 訓練データに PACO 含まれず、part-level 認識は専用モデルに分がある
- **COCO keypoint で ED-Pose に劣る**: 軽量 pose head の制約（-1.6 AP）
- **マスクは Grounded SAM (1.5 Pro + SAM-Huge) より低い**: LVIS-mv mask AP 43.8 vs 47.7、複雑な統一の代償
- **Grounding DINO 1.6 が中間版で論文未公開**: 比較対象として頻繁に出るが詳細が不明
- **MLLM 連携の実証なし**: 「MLLM hallucination 削減」は動機だが具体的実験は本論文にはない
- **動画対応なし**: SAM 3 と異なる
- **依然として early fusion の hallucination 問題**: GD 1.5 と同じ、明示的な改善議論は本論文にない

---

## 関連ページ

- [[sources/dino-x]] — 原典の要約
- [[translations/dino-x]] — 原典の和訳
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
