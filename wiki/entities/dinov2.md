---
type: entity
entity_kind: model
aliases: [DINOv2, DINO v2]
tags: [self-supervised, vision-transformer, foundation-model, meta-ai]
related: [[concepts/self-supervised-learning]], [[concepts/foundation-model]], [[concepts/masked-image-modeling]], [[concepts/vision-transformer]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# DINOv2（手法・モデル群）

## 概要

**DINOv2** = **DINO version 2**。Meta AI Research（旧 FAIR）と Inria が 2023 年に発表した、CV における**汎用視覚基盤モデル**（[[concepts/foundation-model]]）として位置づけられる Vision Transformer（[[concepts/vision-transformer]]）の事前学習モデル群。

- 論文: "DINOv2: Learning Robust Visual Features without Supervision"
- arXiv: 2304.07193 → TMLR 2024
- コード・モデル: <https://github.com/facebookresearch/dinov2>
- 詳細解説: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
- 翻訳: [[translations/dinov2-learning-robust-visual-features-without-supervision]]

DINO（[[entities/dino]]）の正統な後継。**iBOT**（[[entities/ibot]]）の学習目的関数を継承し、データ・モデル・効率化の総力戦で「凍結特徴量のままで OpenCLIP（[[entities/clip]]）に勝つ」ことを実現した。

## 公開モデル群

| Model | Params | Patch | Tokens | Distilled? |
|---|---|---|---|---|
| ViT-S/14 | 21M | 14 | 256 | from ViT-g |
| ViT-B/14 | 86M | 14 | 256 | from ViT-g |
| ViT-L/14 | 300M | 14 | 256 | from ViT-g |
| ViT-g/14 | 1.1B | 14 | 256 | trained from scratch |

加えて以下の派生・拡張版もある（後続研究）：
- **DINOv2 with registers**: Darcet ら（2023）が追加した "register tokens" で attention artifact を解消
- **DINOv2 high-res**: 518×518 で短期間ファインチューン

## 学習設定（要点）

| 項目 | 値 |
|---|---|
| データセット | LVD-142M（142M 画像、自動キュレーション） |
| 学習目的関数 | iBOT 損失（DINO + MIM）+ KoLeo 正則化 |
| 崩壊回避 | Sinkhorn-Knopp centering（DINO の moving-average centering の置換） |
| projection head | **DINO と iBOT で分離**（iBOT 原論文は共有が良いと報告したが、スケール時は逆） |
| FFN | SwiGLU（Swish-Gated Linear Unit） |
| パッチサイズ | 14（DINO の 16 から変更） |
| Teacher momentum | 0.994（DINO の 0.996 より遅い）|
| Batch size | 3072 |
| 高解像度適応 | 訓練終盤に 518×518 で短期間訓練 |
| 主要最適化 | FlashAttention 自作版 / Sequence packing / Efficient stochastic depth / FSDP 混合精度 |
| 効率 | iBOT 比で 2× 速、メモリ 1/3 |
| 蒸留 | ViT-S/B/L は ViT-g からの蒸留で作成 |
| 学習コスト | ViT-g: 22k GPU 時間（A100-40GB）≒ 9.7 MWh ≒ 3.7 t CO₂eq |

## 主要結果

### ImageNet-1k

| Model | Linear (val) | k-NN |
|---|---|---|
| DINOv2 ViT-S/14 | 81.1 | 79.0 |
| DINOv2 ViT-B/14 | 84.5 | 82.1 |
| DINOv2 ViT-L/14 | 86.3 | 83.5 |
| **DINOv2 ViT-g/14** | **86.5** | **83.5** |
| OpenCLIP ViT-G/14（比較） | 86.2 | 83.2 |
| EVA-CLIP ViT-g/14（比較） | 86.4 | 83.5 |

**凍結 + 線形だけで OpenCLIP-G/EVA-CLIP-g を上回った**最初の SSL モデル。

### Robustness（ImageNet 派生）

ImageNet-A で **75.9%**（iBOT に対し +29.6%、OpenCLIP-G に対し +12.1%）。

### Dense prediction

| Task / Benchmark | DINOv2 ViT-g | OpenCLIP-G | iBOT-L |
|---|---|---|---|
| ADE20K seg. (lin+ms) | 53.0 | 46.0 | 47.5 |
| Cityscapes (lin+ms) | 81.0 | 70.3 | 74.5 |
| Pascal VOC (lin+ms) | 86.2 | 79.2 | 84.3 |
| NYUd depth RMSE (DPT) ↓ | 0.279 | 0.414 | 0.358 |
| KITTI depth RMSE (DPT) ↓ | 2.11 | 2.56 | 2.55 |

セグメンテーションも深度推定も、**凍結 backbone + 軽い head のみで OpenCLIP に大差**をつける。

### Instance retrieval

Oxford-Hard で mAP +41% vs iBOT、+34% vs OpenCLIP。

### 創発的性質

- **PCA で物体の部位がドメイン横断で対応**（[[sources/dinov2-learning-robust-visual-features-without-supervision]] §7.5 図 1 / 9 / 10）
- 教師信号なしで物体の部位構造とシーン幾何を表現

## なぜ DINO に比べて爆発的に強くなったか

5 つの要因の合成：

1. **iBOT 由来の MIM 損失**: dense prediction 性能の鍵
2. **LVD-142M**: ImageNet-22k より大きく多様
3. **ViT-g（1B パラメータ）への scale up**
4. **学習レシピの磨き込み**: KoLeo / SK / SwiGLU / patch=14 / momentum=0.994 / batch=3k / head 分離など、合計 +10% 程度の改善を積み上げ
5. **効率化**: FSDP / FlashAttention / sequence packing で 2× 速・メモリ 1/3、これで長時間 + 大バッチが可能に

## 産業応用への影響

DINOv2 の公開以降、**「画像特徴抽出は DINOv2 を凍結して使う」**が事実上の標準になった分野：

- **3D 再構成・SfM**（Structure from Motion）: 特徴量のロバスト性とパッチ対応の良さが有用
- **医療画像**: ファインチューンなしでも良く転移
- **衛星画像**
- **ロボティクスの perception**
- **VLM の vision tower**: LLaVA-NeXT などで CLIP と併用するケース

## 系譜と派生

### 由来
- [[entities/dino]]（2021）→ [[entities/ibot]]（2021/ICLR 2022, [[sources/ibot]]）→ **DINOv2**（2023）
- iBOT の MIM 損失と online tokenizer（[[concepts/online-tokenizer]]）の発想は [[entities/mae]]（He et al., 2021）の masked image modeling 系統と DINO の self-distillation 枠組みを統合したもの
- 各種実装技法は MoCo / BYOL / SwAV / SimMIM などからの吸収

### MAE との対比

[[entities/mae]] は DINOv2 と**対照的かつ補完的**:

| | MAE | DINOv2 |
|---|---|---|
| 学習信号 | 画像内マスク再構成のみ | 自己蒸留（DINO）+ MIM（iBOT 由来）|
| 凍結特徴量 | 弱い（linear 75.8%） | 強い（linear 86.3%、k-NN 83.5%）|
| Fine-tuning | 非常に強い（87.8% IN1K） | 強い（88.5%）|
| 用途 | Fine-tune ありの大規模応用 | 凍結特徴量での汎用利用 |
| Emergent | なし | 自己注意マップにオブジェクト境界 |

DINOv2 は MAE の MIM 強みを iBOT 経由で取り込みつつ、DINO の self-distillation で凍結特徴量を強化した「両取り」の設計。

### 後続
- **DINOv2 with registers**（Darcet et al., 2023）: artifact 解消、register tokens 導入
- **DINOv3**（2025, [[entities/dinov3]]）: ViT-7B × LVD-1689M に scale、**Gram anchoring**（[[concepts/gram-anchoring]]）で long-training dense feature 劣化を解決、**RoPE**（[[concepts/rotary-position-embeddings]]）採用、ConvNeXt/dino.txt まで含むモデルファミリを公開。詳細: [[sources/dinov3]]
- **DINOv3 satellite**: 同レシピを衛星画像に適用、地理空間ベンチマーク多数で SOTA。詳細: [[entities/sat-493m]]

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: 原論文の詳細解説
- [[translations/dinov2-learning-robust-visual-features-without-supervision]]: 原論文全訳
- 系譜: [[entities/dino]] → [[entities/ibot]] → **DINOv2**
- データ: [[entities/lvd-142m]]
- 比較対象: [[entities/clip]]
- 概念: [[concepts/foundation-model]] / [[concepts/self-supervised-learning]] / [[concepts/masked-image-modeling]] / [[concepts/knowledge-distillation]] / [[concepts/knn-evaluation-protocol]]
