---
type: entity
entity_kind: model
aliases: [MoCo, Momentum Contrast, MoCo v1, MoCo v2, MoCo v3]
related: ["[[concepts/contrastive-learning]]", "[[concepts/self-supervised-learning]]", "[[entities/simclr]]", "[[entities/byol]]", "[[entities/dino]]"]
sources: []
updated: 2026-05-30
---

# MoCo（Momentum Contrast）

**MoCo** は He et al.（Facebook AI Research, CVPR 2020）が発表した対比的自己教師あり学習（contrastive SSL）の代表的手法。**Momentum encoder（運動量エンコーダ）と queue（メモリバンク）**という 2 つの工夫で、対比学習を大バッチ依存から解放した重要な前駆体。本 wiki ではまだ専用 source ページ未取り込み（**ingest 候補**）、本ページは [[entities/byol|BYOL]] / [[entities/simclr|SimCLR]] / [[entities/dino|DINO]] からの参照を受ける概要エンティティ。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 論文 | "Momentum Contrast for Unsupervised Visual Representation Learning" |
| 著者 | Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick |
| 所属 | Facebook AI Research（現 Meta AI） |
| 発表 | CVPR 2020（arXiv:1911.05722, 2019 年 11 月） |
| バージョン | **MoCo v1**（2019）→ **MoCo v2**（2020）→ **MoCo v3**（2021、ViT 対応） |
| ライセンス | Apache 2.0 |
| 公式実装 | https://github.com/facebookresearch/moco |

## アーキテクチャの核心アイデア

### 1. Momentum Encoder（運動量エンコーダ）

対比学習では「**クエリ・エンコーダ $f_q$** と **キー・エンコーダ $f_k$**」の 2 つを使う：

- **クエリ側**: 通常の back-propagation で更新
- **キー側**: クエリ側の **指数移動平均（EMA）** で更新

$$
\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q, \quad m = 0.999
$$

これにより、キー・エンコーダが **緩やかに変化する一貫した表現**を提供。BYOL の target network や DINO の teacher network の **直接の源流**。

### 2. Queue（メモリバンク）

対比学習は大量の負例を必要とする。SimCLR は同バッチ内の他サンプルを負例にするため **大バッチ（4096 等）が必須**。MoCo は：

- 過去のバッチからのキー特徴量を **FIFO キュー**に保持（典型的 65,536 個）
- 各バッチで「クエリ vs キュー全体」で InfoNCE 損失を計算
- → **バッチサイズに依存しない大規模負例セット**

これにより 256 程度の小バッチでも対比学習を効果的に行える。

## バージョン進化

| Version | 年 | 主な変更 |
| --- | --- | --- |
| **MoCo v1** | 2019/CVPR 2020 | Momentum encoder + queue を初導入 |
| **MoCo v2** | 2020 | SimCLR の知見（MLP projection head + 強い拡張）を統合、ImageNet linear probe 71.1% |
| **MoCo v3** | 2021 | **ViT バックボーン対応**、queue を廃止して大バッチに戻る、ViT-B/16 で 76.7% |

## 主要結果（MoCo v2、ResNet-50）

| ベンチマーク | スコア |
| --- | --- |
| ImageNet linear probe | **71.1%** |
| ImageNet 1% labels | 38.9% |
| PASCAL VOC detection AP | 57.0 |
| COCO detection AP | 39.4 |

教師あり ResNet-50（76.5%）にギャップが残るが、後続の **[[entities/byol|BYOL]]（74.3%）** や **[[entities/simclr|SimCLR]]（76.5% with 4× width）** の比較対象として広く使われた。

## 系譜上の位置

### SSL 対比学習の進化

```
InstDisc (2018, memory bank)
    │
    ↓
MoCo v1 (2019) ← 本ページ
    │ momentum encoder + queue
    ↓
SimCLR (2020) ← 大バッチ + MLP head
    │
    ├→ MoCo v2 (2020) ← SimCLR の MLP head を取り込み
    │
    ├→ BYOL (2020) ← 負例なし、momentum target を継承
    │
    └→ SimSiam (2020) ← stop-gradient
        │
        ↓
DINO (2021) ← BYOL の teacher を ViT へ拡張、self-distillation 形式
    │
    ↓
DINOv2 (2023) / DINOv3 (2025) ← iBOT 統合、スケール化
```

### 何が「直接の源流」だったか

| 派生先 | 受け継いだ要素 |
| --- | --- |
| **[[entities/byol|BYOL]]** | **Momentum encoder（target network）** → 「負例なし SSL」の核心 |
| **[[entities/dino|DINO]]** | **Momentum teacher** + EMA 更新 → 自己蒸留 SSL の基盤 |
| **[[entities/simclr|SimCLR]]** | 対比学習の基本枠組み（ただし queue は使わず大バッチで代替） |
| **MoCo v3 / SimCLR v2 / DINOv2** | InfoNCE 損失の標準化 |

## なぜ「次の ingest 候補」か

本 wiki では **SSL 系の主要手法**（[[entities/simclr]] / [[entities/byol]] / [[entities/dino]] / [[entities/dinov2]] / [[entities/dinov3]] / [[entities/mae]] / [[entities/ibot]]）が ingest 済みだが、**MoCo はその起点**であり、3 つの ingest 済み手法（BYOL / SimCLR / DINO）から参照される。専用 source ページの ingest により：

- Momentum encoder の数学的詳細
- Queue（メモリバンク）のサイズ依存性
- v1 → v2 → v3 の改善史
- ViT への適用（v3）と DINO との比較

を完全に追跡できるようになる。

## 関連ページ

- [[entities/simclr]] — 同時期の対抗系譜（大バッチ路線）
- [[entities/byol]] — Momentum encoder を継承した非対比 SSL の先駆け
- [[entities/dino]] — Momentum teacher を ViT へ拡張、自己蒸留 SSL
- [[concepts/contrastive-learning]] — InfoNCE 損失、対比学習の枠組み
- [[concepts/self-supervised-learning]] — SSL 全体の俯瞰
