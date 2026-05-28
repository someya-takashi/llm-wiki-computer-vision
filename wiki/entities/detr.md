---
type: entity
entity_kind: model
aliases: [DETR, DEtection TRansformer]
tags: [object-detection, transformer, set-prediction, facebook-ai, end-to-end, panoptic-segmentation]
related: [[concepts/object-detection]], [[concepts/vision-transformer]], [[concepts/foundation-model]]
sources: [[sources/detr]]
updated: 2026-05-28
---

# DETR（DEtection TRansformer）

## 概要

**DETR** は Facebook AI Research Paris（Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko）が ECCV 2020 で発表した、**Transformer を物体検出に持ち込んだ最初の本格的成功例**。物体検出を **集合予測（set prediction）問題** として再定義し、Hungarian アルゴリズムによる二部マッチング + Transformer グローバル attention + 並列デコーディングだけで、**NMS / anchor / proposal という従来の hand-designed components をすべて排除**した。

- 論文: "End-to-End Object Detection with Transformers"
- arXiv: 2005.12872 / ECCV 2020
- コード: <https://github.com/facebookresearch/detr>
- メーカー: Facebook AI Research (Paris)
- ライセンス: Apache 2.0

---

## モデル仕様

### ベースモデル（表 1）

| Model | Backbone | #params | GFLOPS/FPS | AP | AP_S | AP_L |
|---|---|---|---|---|---|---|
| **DETR** | ResNet-50 | 41M | 86/28 | 42.0 | 20.5 | **61.1** |
| **DETR-DC5** | ResNet-50 (dilated C5) | 41M | 187/12 | 43.3 | 22.5 | 61.1 |
| **DETR-R101** | ResNet-101 | 60M | 152/20 | 43.5 | 21.9 | 61.8 |
| **DETR-DC5-R101** | ResNet-101 (dilated C5) | 60M | 253/10 | **44.9** | 23.7 | **62.3** |

### Transformer 構成

- **Encoder**: 6 層、幅 256、8 ヘッド attention、FFN 隠れ次元 2048
- **Decoder**: 6 層、同サイズ
- **Object queries**: N = 100（学習可能位置埋め込み）
- **位置エンコーディング**: 2D sine（固定、encoder の self-attention 各層で追加）
- **FFN**: 3 層 MLP（ReLU + 線形射影）で (class, bbox) を予測

### 主要設計判断

| 判断 | 影響 |
|---|---|
| **N = 100 個の固定スロット** | COCO の典型物体数 7 をはるかに上回るが、100 超では性能急落（§0.A.5.1） |
| **Hungarian 1 対 1 マッチング** | 重複予測を訓練レベルで排除、**NMS が不要** |
| **GIoU + ℓ₁ ボックス損失** | GIoU 単独でも -0.7 AP 程度、ℓ₁ 単独だと -4.8 AP |
| **補助デコーディング損失** | 各 decoder 層の後で同じ Hungarian 損失を適用、訓練安定化 |
| **Backbone と transformer の学習率分離** | backbone 1e-5、transformer 1e-4 — 訓練初期の安定化に重要 |
| **300〜500 エポック訓練** | 標準検出器の 10 倍以上、DETR の大きな弱点 |

---

## 主要結果

### COCO 物体検出

ResNet-50 ベースで Faster R-CNN-FPN+ と同等の 42.0 AP を達成しつつ：

- **大物体（AP_L）で +7.8 ポイント**（53.4 → 61.1）: グローバル attention の威力
- **小物体（AP_S）で -5.5 ポイント**（26.6 → 20.5）: 弱点、後の Deformable DETR が解決

### COCO パノプティックセグメンテーション（表 5）

| Model | PQ | PQ^th (things) | PQ^st (stuff) |
|---|---|---|---|
| PanopticFPN++ R101 | 44.1 | 51.0 | 33.6 |
| **DETR-R101** | **45.1** | 50.5 | **37.0** |

- **DETR は特に stuff（不可算領域）で支配的**: グローバル attention で「画像全体の意味的レイアウト」を捉える
- Things クラスでも競争的、テストセットで 46 PQ

---

## アーキテクチャの構造的特徴

### 1. Encoder の attention でインスタンスを分離

§4.2.1 の Encoder attention 可視化（図 3）: encoder の最終層で、画像内の各物体のピクセルが既に空間的に分離されている。**Encoder がシーン理解を完了、decoder は境界抽出だけに集中** という分業。

### 2. Decoder の attention は物体末端に局所化

§4.2.2 の Decoder attention 可視化（図 6）: 各 object query は予測物体の **頭・脚・手・耳** など末端部分に attend する。**「物体の輪郭は末端で決まる」** という直感に対応。

### 3. Object queries の自然なスロット特化

§4.3.1 のスロット可視化（図 7）:

- スロットによって「画像のどの位置・どのサイズの物体」を担当するかが学習で分化
- ほぼ全スロットが「画像全体ボックス」モードも持つ（COCO バイアスを反映）
- **クラス特化はしない**（特定スロットが「犬専用」にはならない）

これは、hand-designed な anchor 配置が **データから自然に学習される** ことの実証。

### 4. NMS を訓練で排除する仕組み

Hungarian 二部マッチングは **各 ground truth に対してちょうど 1 つの予測** しか正解扱いされない。同じ物体に複数のスロットが活性化することは構造的に許されない。

§4.2.2 で、各 decoder 層の出力に NMS を適用する実験：
- 第 1 層: NMS で +大幅改善（self-attention が機能していない）
- 第 6 層（最終）: NMS で **逆に AP 低下**（true positive を誤削除）

→ Decoder の self-attention がデコーディング過程で重複予測を抑制している。

---

## DETR ファミリーの系譜

DETR は ECCV 2020 で発表後、**「DETR を改善する」研究系統** を切り拓いた：

```
2020 May    DETR (Carion et al., ECCV 2020) — 原典
              │
2020 Oct    Deformable DETR (Zhu et al., ICLR 2021) — sparse attention で訓練速度・小物体を解決
              │
2022        DAB-DETR (Liu et al., ICLR 2022) — object queries を 4D ボックスとして解釈
              │
2022        DN-DETR (Li et al., CVPR 2022) — denoising 補助訓練で収束加速
              │
2022 Mar    MDETR (Kamath et al., ICCV 2021) — テキスト条件付き DETR、open-vocab 検出
              │
2022 Sep    DETA (Ouyang-Zhang et al., 2022) — 1 対 1 → 1 対多マッチングで収束加速
              │
2023        DINO (detector) (Zhang et al., ICLR 2023, [[entities/dino-detector]] / [[sources/dino-detector]]) — DAB + DN + Mixed query selection 統合、初の end-to-end Transformer COCO SOTA 63.3 AP
              │
2023        CoDETR (Zong et al., ICCV 2023) — DETR + ATSS + Faster R-CNN 協調訓練、COCO SOTA
              │
2024-2025   PEspatial が DETA を採用（[[sources/perception-encoder]]）、COCO 66.0 box AP
2025        SAM 3 が DETR スタイル detector + presence head（[[sources/sam-3]]）
```

**「DETR ファミリー」** は 2020 以降の検出研究の中心系統の 1 つ。

---

## 既存 wiki への接続

### Foundation Model の検出ヘッド標準

[[concepts/foundation-model]] 時代の CV foundation model は、検出タスクで以下のいずれかを採用：
- **Mask R-CNN 系**（[[entities/sam]] / [[entities/sam-2]] の SA-1B / SA-V データエンジン用）
- **DETR 系**（[[entities/sam-3]] の PCS、[[entities/perception-encoder]] の PEspatial）

PE PEspatial が **DETA + Objects365 で COCO 66.0 box AP** という SOTA を達成したのは、DETR ファミリーの集大成と言える。

### Promptable Segmentation との関係

[[concepts/promptable-segmentation]] の SAM は、DETR のような decoder + query アーキテクチャを採用するが、**クエリが「点・ボックス・テキスト」というユーザー入力** に置き換わる。DETR の「学習可能 query」と SAM の「ユーザー指定 prompt」は、設計思想として連続している。

[[concepts/promptable-concept-segmentation]] の SAM 3 は、**DETR スタイル detector を直接採用** し、テキスト条件付き化（MDETR の発展形）。

### ViT との並列性

[[concepts/vision-transformer]] の ViT（2020 Oct）と DETR（2020 May）は、**Transformer を CV に持ち込んだ 2 つの並列研究**：

| | ViT | DETR |
|---|---|---|
| 発表時期 | 2020 Oct（ICLR 2021）| 2020 May（ECCV 2020）|
| Transformer の使い方 | 画像をパッチに分解、CNN を完全置換 | CNN backbone の特徴を Transformer で再処理 |
| タスク | 画像分類 | 物体検出 |
| Position encoding | 学習 1D | 固定 2D sine + 学習 queries |

**DETR は ViT より先に Transformer を CV に持ち込んだ** とも言える。ただし「Transformer-only な CV」（CNN を完全に排除）の達成は ViT で、両者は相補的。

---

## 限界と批判

- **小物体での性能不足**: Deformable DETR が解決
- **訓練時間が極端に長い**: 16 V100 GPU で 3-6 日。DAB-DETR / DN-DETR / [[entities/dino-detector|DINO-detector]] / DETA が大幅に短縮
- **収束の不安定さ**: 補助損失 + Xavier 初期化 + 学習率分離が必須
- **N = 100 の物体数上限**: 100 物体近い画像で検出率急落（§0.A.5.1 図 12）
- **計算コスト**: encoder の self-attention が O((HW)²) で、大解像度では問題（Deformable DETR が解決）

---

## 主要な貢献まとめ

1. **物体検出を集合予測として再定義**: hand-designed components（NMS, anchor, proposal）すべて排除
2. **Hungarian 二部マッチング + Transformer 並列デコード** という新しい検出パラダイム
3. **大物体性能で +7.8 AP** という、グローバル attention の決定的優位を実証
4. **パノプティックセグメンテーションへの自然な拡張**: things と stuff を統一処理
5. **DETR ファミリー** という研究系統の創出（Deformable DETR, [[entities/dino-detector|DINO-detector]], DETA, CoDETR, MDETR, OWL-ViT...）
6. **CV における Transformer 革命の出発点**（ViT と並んで）

## 関連ページ

- [[sources/detr]] — 原典の要約
- [[translations/detr]] — 原典の和訳
- [[concepts/object-detection]] — 物体検出全体の系譜と DETR の位置づけ
- [[concepts/vision-transformer]] — 並列研究、Transformer を CV へ
- [[concepts/foundation-model]] — DETR は現代 CV foundation model の検出ヘッド標準
- [[concepts/promptable-segmentation]] — SAM が DETR スタイル decoder を採用
- [[concepts/promptable-concept-segmentation]] — SAM 3 が DETR-based detector + presence head
- [[entities/sam-3]] — DETR スタイル detector で PCS タスクを解く
- [[entities/perception-encoder]] — PEspatial が DETA decoder で COCO 66.0 box AP の SOTA
- [[entities/sam]] / [[entities/sam-2]] — Mask R-CNN 系統の対比（DETR 系 vs Mask R-CNN 系）
