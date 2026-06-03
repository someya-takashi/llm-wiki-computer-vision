---
type: entity
entity_kind: model
aliases: [DINOv3, DINO v3]
tags: [self-supervised, vision-transformer, foundation-model, meta-ai, convnext]
related: ["[[concepts/self-supervised-learning]]", "[[concepts/foundation-model]]", "[[concepts/masked-image-modeling]]", "[[concepts/vision-transformer]]", "[[concepts/gram-anchoring]]", "[[concepts/rotary-position-embeddings]]"]
sources: ["[[sources/dinov3]]"]
updated: 2026-05-24
---

# DINOv3（モデルファミリ）

## 概要

**DINOv3** = **DINO version 3**。Meta AI Research、Inria、WRI（World Resources Institute）が 2025 年に発表した、CV における汎用視覚基盤モデル（[[concepts/foundation-model]]）の現行最強候補。

- 論文: "DINOv3"
- arXiv: 2508.10104（2025）
- コード・モデル: <https://github.com/facebookresearch/dinov3>（モデルは Apache 2.0 ライセンスで公開）
- 詳細解説: [[sources/dinov3]]
- 翻訳: [[translations/dinov3]]

[[entities/dinov2]] の正統な後継。[[entities/ibot]] の損失関数を継承しつつ、**Gram anchoring**（[[concepts/gram-anchoring]]）で長時間学習の dense feature 劣化問題を解決し、ViT-7B（67 億パラメータ）まで scale。**凍結バックボーンのまま CV の主要タスクで SOTA を更新**した。

---

## モデルファミリ一覧

### Vision Transformer 系

| Model | Params | Patch | 蒸留元 | 用途 |
|---|---|---|---|---|
| ViT-S/16 | 21M | 16 | from 7B | エッジ・ラップトップ |
| ViT-S+/16 | 29M | 16 | from 7B | カスタムサイズ |
| ViT-B/16 | 86M | 16 | from 7B | 中規模応用 |
| **ViT-L/16** | 300M | 16 | from 7B | **主力（実用 sweet spot）** |
| ViT-H+/16 | 840M | 16 | from 7B | 高性能、ほぼ 7B 並み |
| **ViT-7B/16** | 6.7B | 16 | scratch | フラッグシップ、teacher |

### ConvNeXt 系（量子化向け）

| Model | Params | 蒸留元 |
|---|---|---|
| ConvNeXt-T | 28M | from 7B |
| ConvNeXt-S | 50M | from 7B |
| ConvNeXt-B | 88M | from 7B |
| ConvNeXt-L | 198M | from 7B |

### Text-aligned 版

- **dino.txt**: ViT-L をフリーズして上に 2 層 Transformer + テキストエンコーダを LiT スタイルで対比学習。ゼロショット分類・検索が可能。

### 衛星画像版

- **DINOv3 Sat ViT-7B**: SAT-493M（[[entities/sat-493m]]）で同レシピで学習
- **DINOv3 Sat ViT-L**: 衛星 7B からの蒸留

---

## アーキテクチャ詳細（ViT-7B）

DINOv2-g との比較：

| | DINOv2 ViT-g | **DINOv3 ViT-7B** |
|---|---|---|
| Backbone | ViT-giant | ViT-7B |
| #Params | 1.1B | **6.7B** |
| #Blocks | 40 | 40（同じ） |
| Patch size | 14 | **16** |
| Embed dim | 1536 | **4096** |
| FFN type | SwiGLU | SwiGLU（同じ） |
| FFN hidden | 4096 | **8192** |
| Attn heads | 24 | **32** |
| Head dim | 64 | **128** |
| Pos. embeddings | Learnable | **axial RoPE**（[[concepts/rotary-position-embeddings]]）|
| Registers | 4 | 4（同じ） |
| DINO head MLP | 4096-4096-256 | **8192-8192-512** |
| DINO prototypes | 128k | **256k** |
| iBOT head MLP | 4096-4096-256 | **8192-8192-384** |
| iBOT prototypes | 128k | 96k |

---

## 学習設定

| 項目 | 値 |
|---|---|
| 学習データ | LVD-1689M（[[entities/lvd-1689m]]）+ 10% ImageNet-1k 均質バッチ |
| 学習目的関数 | $\mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + 0.1 \mathcal{L}_{Koleo}$ |
| 崩壊回避 | Sinkhorn-Knopp centering（SwAV 由来）|
| projection head | DINO と iBOT で**分離**（DINOv2 から踏襲） |
| パッチサイズ | **16**（DINOv2 の 14 から変更） |
| Crop 戦略 | 2 global (256²) + 8 local (112²) multi-crop |
| Pos. embeddings | axial RoPE + box jittering $s \in [0.5, 2]$ |
| Register tokens | 4（[CLS] とは別） |
| Optimizer | AdamW |
| **スケジュール** | **constant**（学習率・WD・EMA momentum 全部定数） |
| Batch size | 4096（256 GPU 分散） |
| 主訓練長 | 1M iterations |
| 効率最適化 | FlashAttention 自作版 / FSDP / Sequence packing 等 |

### Gram anchoring フェーズ（§4）

- 主訓練 1M 反復後に追加で 10k 反復実施
- 損失: $w_D \mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + w_{DK} \mathcal{L}_{DKoleo} + w_{Gram} \mathcal{L}_{Gram}$
- **Gram teacher は早期スナップショット（〜200k 反復時点）**を使用
- Gram teacher は 2× 解像度で forward + bicubic ダウンサンプリングして高品質化
- 詳細: [[concepts/gram-anchoring]]

### 高解像度ポスト訓練（§5.1）

- mixed resolutions: global {512, 768}, local {112, 168, 224, 336}
- 10k 反復追加
- ここでも Gram anchoring が必須
- 結果: 4k 解像度を超えても安定

### Multi-student distillation（§5.2）

7B teacher の forward を**全 GPU で共有**して複数 student（ViT-S, B, L, S+, H+, ConvNeXt T/S/B/L）を**並列に**蒸留：

```
[全 GPU] → teacher 推論（1 回だけ）
    ↓
[all-gather] → 全 GPU に入力＋teacher 出力配信
    ↓
[student グループ A] [student グループ B] [グループ C] ... → 各自で student 学習
```

- 「student を 1 つ追加するコスト = その student の訓練コストのみ」
- 各 student の GPU 数を調整して反復時間を揃え、同期待ちを最小化
- これで多様なモデルファミリを単一ジョブで生産

---

## 主要結果

### Dense linear probing

| Method | ADE20k | Cityscapes | VOC | NYU↓ | KITTI↓ |
|---|---|---|---|---|---|
| DINOv2 ViT-g | 49.5 | 75.6 | 83.1 | 0.372 | 2.624 |
| AM-RADIOv2.5 g/14 | 53.0 | 78.4 | 85.4 | 0.340 | 2.918 |
| PEspatial G/14 | 49.3 | 73.2 | 82.7 | 0.362 | 3.082 |
| SigLIP 2 g/16 | 42.7 | 64.8 | 72.7 | 0.494 | 3.273 |
| **DINOv3 7B/16** | **55.9** | **81.1** | **86.6** | **0.309** | **2.346** |

### ImageNet 線形プローブ

| Method | IN-Val | IN-V2 | IN-ReaL | IN-R | IN-A | IN-C↓ | Obj. |
|---|---|---|---|---|---|---|---|
| SigLIP 2 g/16 | 89.1 | 81.6 | 90.5 | 92.2 | 84.6 | 30.0 | 78.6 |
| PEcore G/14 | 89.3 | 81.6 | 90.4 | 92.2 | **89.0** | 22.7 | **80.2** |
| DINOv2 g/14 | 87.3 | 79.5 | 89.9 | 81.1 | 81.7 | 24.1 | 66.4 |
| **DINOv3 7B/16** | 88.4 | 81.4 | 90.4 | 91.1 | 86.9 | **19.6** | 79.0 |

→ **SSL モデルとして初めて弱教師ありに肩を並べた**

### Complex pipelines (frozen backbone)

| タスク | DINOv3 結果 | 備考 |
|---|---|---|
| COCO 検出 mAP (Plain-DETR) | **66.1** | 凍結 backbone で初の SOTA、訓練可能 100M params |
| ADE20k seg. (Mask2Former+ViT-Adapter) | **63.0** mIoU | ONE-PEACE と同点 SOTA |
| NYUv2 depth RMSE (DAv2) | **4.3 ARel** | DAv2 ViT-g の 4.4 を超え、しかも凍結 |
| 3D 理解 (VGGT) | 全タスク SOTA | DINOv2 → DINOv3 差し替えで改善 |

### Instance retrieval

| Method | Oxford-H | Paris-H | Met (GAP) | AmsterTime |
|---|---|---|---|---|
| DINOv2 g/14 | 58.2 | 84.6 | 44.6 | 48.9 |
| OpenCLIP G/14 | 47.5 | 85.7 | 30.5 | 23.1 |
| **DINOv3 7B/16** | **60.7** | **87.1** | **55.4** | **56.5** |

---

## 創発的・固有の性質

- **4k 解像度を超えても feature map が安定**: RoPE と Gram anchoring の組合せ
- **動画タスクへの転用が容易**: 静止画モデルで DAVIS J&F 83.3、Kinetics-400 88.2（attentive probe）
- **3D 認識**: NAVI/SPair で SOTA、VGGT に組み込んで全 3D タスク改善
- **sim-to-real ギャップを埋める**: DINOv2 の特性を継承、Depth Anything V2 で合成データのみで実画像深度推定可能
- **ドメイン汎用性**: 同レシピで衛星画像版が Geo-Bench で SOTA

---

## 公開モデル使用例

```python
import torch

# Hub から ViT-L をロード（凍結バックボーンとして使用）
dinov3 = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')
dinov3.eval()

# パッチ特徴量と CLS トークンを取得
with torch.no_grad():
    features = dinov3.forward_features(image)
    cls_token = features['x_norm_clstoken']      # (B, 1024)
    patch_tokens = features['x_norm_patchtokens']  # (B, 256, 1024) for 512² input

# 下流タスクは凍結特徴量の上で軽い head を訓練する形が標準
```

---

## 系譜と派生

### 由来
- [[entities/dino]]（2021）→ [[entities/ibot]]（2022）→ [[entities/dinov2]]（2023）→ **DINOv3**（2025）

### 並列・競合
- [[entities/perception-encoder]] / [[sources/perception-encoder]]（PE / PEcore / PElang / PEspatial, NeurIPS 2025）
- [[entities/siglip]]（SigLIP / SigLIP 2）
- AM-RADIO（NVIDIA）, Franca, Web-DINO, EVA-CLIP-18B

---

## DINOv3 を選ぶべきとき / 選ばないとき

### DINOv3 が最良な選択肢

- 密予測（セグメンテーション・深度推定）が必要
- 高解像度（>1024 px）入力を扱う
- 凍結バックボーンで軽い head だけ訓練したい
- 細粒度分類（iNaturalist、植物・動物種）
- インスタンス検索・コピー検出
- 3D 認識・幾何タスク
- 衛星・医療・産業画像など特殊ドメイン
- 計算予算が中〜大（ViT-L 以上）

### 他のモデルを検討すべきとき

- **ゼロショット画像分類が必要** → CLIP 系（[[entities/clip]]）または PE
- **テキストとの直接整合が必要** → CLIP 系
- **超軽量モデルが必須**（モバイル）→ ConvNeXt-T 蒸留版か、別途軽量モデル
- **動画専用の高度な時間モデリング**が必要 → V-JEPA 2

---

## 関連ページ

- [[sources/dinov3]]: 原論文の詳細解説
- [[translations/dinov3]]: 原論文全訳
- 系譜: [[entities/dino]] → [[entities/ibot]] → [[entities/dinov2]] → **DINOv3**
- データ: [[entities/lvd-1689m]] / [[entities/sat-493m]]
- 競合: [[entities/clip]] / [[entities/perception-encoder]] / [[entities/siglip]]
- 概念: [[concepts/gram-anchoring]] / [[concepts/rotary-position-embeddings]] / [[concepts/foundation-model]] / [[concepts/self-supervised-learning]] / [[concepts/masked-image-modeling]] / [[concepts/knowledge-distillation]] / [[concepts/knn-evaluation-protocol]]
