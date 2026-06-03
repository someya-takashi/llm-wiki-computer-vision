---
type: entity
entity_kind: model
aliases: [DINO (detector), DINO detector, DINO-DETR, DETR with Improved DeNoising Anchor boxes]
tags: [object-detection, transformer, detr-family, denoising, contrastive-denoising, idea, hkust]
related: ["[[concepts/object-detection]]", "[[concepts/foundation-model]]"]
sources: ["[[sources/dino-detector]]"]
updated: 2026-05-28
---

# DINO（物体検出器）

> **重要な注記**: この **DINO は物体検出器**（Zhang et al., ICLR 2023）であり、自己教師あり学習の DINO（Caron et al., ICCV 2021、[[entities/dino]]）とは **完全に別物**。slug `dino-detector` で区別。

## 概要

**DINO（DETR with Improved deNoising anchOr boxes）** は、IDEA / HKUST / 清華大学（Hao Zhang, Feng Li, Shilong Liu, Lei Zhang ら）が 2022 年 3 月に発表（ICLR 2023 採択）した、**DETR ファミリーの集大成と言える物体検出器**。DAB-DETR + DN-DETR + Deformable DETR を統合し、さらに 3 つの新技法を追加して **初めて end-to-end Transformer 検出器が COCO leaderboard で SOTA**（test-dev 63.3 AP）を達成。

- 論文: "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection"
- arXiv: 2203.03605 / ICLR 2023
- コード: <https://github.com/IDEACVR/DINO>
- 開発元: IDEA（International Digital Economy Academy）+ HKUST + 清華大学
- ライセンス: 公開（リポジトリ参照）

---

## 主要設計

### アーキテクチャ（DETR ライク + 3 つの新技法）

| コンポーネント | 内容 |
|---|---|
| Backbone | **ResNet-50** または **SwinL**（ImageNet-22K 事前学習） |
| Multi-scale features | 4-scale または 5-scale（stages 2,3,4 + downsampled stage 4 + optional stage 1） |
| Encoder | 6 層 Transformer、deformable attention、隠れ次元 256、8 ヘッド |
| Decoder | 6 層 Transformer、deformable attention、**900 queries**（300 queries × 3 patterns 相当） |
| Position queries | **4D anchor box (x, y, w, h)** として明示的（DAB-DETR 由来） |
| Content queries | **学習可能（静的）**（Mixed Query Selection の核） |
| 予測ヘッド | **共有予測 FFN**（3 層 MLP + ReLU + 線形射影）→ (class, bbox)、各 decoder 層の後で auxiliary loss |

### 3 つの新技法

#### 1. Contrastive DeNoising (CDN)

DN-DETR を **正例 + 負例の対** に拡張：

- **正クエリ**: $\lambda_1$ より小さいノイズ → GT box を再構成
- **負クエリ**: $\lambda_1 < $ ノイズ $< \lambda_2$ → 「物体なし」を予測（focal loss で background クラスを学ぶ）

ハイパーパラメータ: $\lambda_1 = 1.0, \lambda_2 = 2.0$（4D anchor box の幅と高さに対する比）、100 CDN ペア（200 queries）。

**効果**: 重複予測の抑制（図 8 参照）、小物体で +1.3 AP。

#### 2. Mixed Query Selection

3 つのクエリ初期化方式の比較：

| 方式 | 位置 | コンテンツ | DINO の判断 |
|---|---|---|---|
| Static（DETR/DN-DETR/DAB-DETR）| 学習可能 | 学習可能 or 0 | ❌ encoder の知恵を使わない |
| Pure Selection（Deformable DETR two-stage）| encoder top-K | encoder top-K | ❌ content も未精錬で誤導しやすい |
| **Mixed**（DINO） | **encoder top-K** | **学習可能（静的）** | ✅ |

**論理**: 「encoder の top-K は位置として有用だがコンテンツとしては未精錬」→ 位置だけを使い、コンテンツは学習で精錬する。

#### 3. Look Forward Twice (LFT)

Deformable DETR の **Look Forward Once**（勾配を detach）を **Twice** に拡張：

- 層 $i+1$ の予測 box を、層 $i$ の **detach されていない** $b_i^{\prime}$ + 層 $i+1$ のオフセット $\Delta b_{i+1}$ で計算
- 結果として **層 $i+1$ の損失が層 $i$ のパラメータも更新**
- 後段層の知恵を前段に反映、box 予測の精度向上

---

## モデル仕様

### バリアント

| Model | Backbone | Pre-training | Epochs | val2017 AP | test-dev AP |
|---|---|---|---|---|---|
| DINO-4scale | ResNet-50 | ImageNet-1K | 12 | 49.0 | - |
| DINO-5scale | ResNet-50 | ImageNet-1K | 12 | 49.4 | - |
| DINO-4scale | ResNet-50 | ImageNet-1K | 24 | 50.4 | - |
| DINO-5scale | ResNet-50 | ImageNet-1K | 24 | 51.3 | - |
| DINO-4scale | ResNet-50 | ImageNet-1K | 36 | 50.9 | - |
| DINO-5scale | ResNet-50 | ImageNet-1K | 36 | 51.2 | - |
| **DINO-SwinL** | **SwinL** | **IN-22K → Objects365** | **26+18** | **63.1** (w/o TTA), **63.2** (w/ TTA) | **63.2** (w/o TTA), **63.3** (w/ TTA) |

### ハイパーパラメータ（表 8）

| 項目 | 値 |
|---|---|
| 学習率 | 1×10⁻⁴ |
| backbone 学習率 | 1×10⁻⁵ |
| weight decay | 1×10⁻⁴ |
| optimizer | AdamW |
| gradient clipping | max norm 0.1 |
| encoder/decoder 層数 | 6 / 6 |
| feedforward dim | 2048 |
| hidden dim | 256 |
| dropout | 0.0 |
| heads | 8 |
| num queries | 900 |
| encoder/decoder sampling points | 4 / 4 |
| 損失係数 | class 1.0, bbox 5.0, GIoU 2.0 |
| focal loss | $\alpha = 0.25, \gamma = 2$ |
| CDN ノイズ | box 0.4, label 0.5 |

---

## 主要結果

### COCO 検出 12 epoch（表 1）— ResNet-50 backbone での革命

| Model | AP | AP_S | AP_M | AP_L |
|---|---|---|---|---|
| Faster-RCNN | 37.9 | 22.4 | 41.1 | 49.1 |
| DETR(DC5) | 15.5 | 4.3 | 15.1 | 26.7 |
| Deformable DETR | 41.1 | - | - | - |
| DN-Deformable-DETR | 43.4 | 24.8 | 46.8 | 59.4 |
| **DINO-4scale** | **49.0 (+5.6)** | **32.0 (+7.2)** | 52.3 | 63.0 |
| **DINO-5scale** | **49.4 (+6.0)** | **32.3 (+7.5)** | 52.5 | 63.9 |

**12 epoch という短い訓練で先行 SOTA を圧倒**。特に **DETR の伝統的弱点 AP_S で +7.5 AP** という大きな改善。

### COCO SOTA 設定（表 3）— SwinL + Objects365

| Model | Params | val2017 AP | test-dev AP | End-to-end |
|---|---|---|---|---|
| SwinL（baseline） | 284M | 57.1 | 57.7 | ❌ |
| Soft Teacher+SwinL | 284M | 60.1 | - | ❌ |
| [[entities/glip|GLIP]] | ≥284M | - | - | ❌ |
| Florence-CoSwin-H | ≥637M | - | - | ❌ |
| SwinV2-G | **3.0B** | 61.9 | - | ❌ (HTC++) |
| **DINO-SwinL** | **218M** | **63.1** | **63.2** | **✅** |

- **end-to-end（NMS なし）で初めて COCO SOTA**
- **SwinV2-G の 1/15 のパラメータ**で SwinV2-G（63.1 test-dev w/ TTA）を上回る
- TTA なしで 63.2、TTA ありで 63.3

### ablation（表 4）

| 構成 | AP | 増分 |
|---|---|---|
| DN-DETR | 43.4 | baseline |
| Optimized DN-DETR | 44.9 | +1.5 |
| Strong baseline (+pure QS) | 46.5 | +1.6 |
| +Mixed QS | 47.0 | +0.5 |
| +Look Forward Twice | 47.4 | +0.4 |
| **+Contrastive DN（DINO 完成）** | **47.9** | **+0.5** |

3 つの新技法で合計 **+1.4 AP**。Pure QS による +1.6 AP もスケール上は大きい貢献。

### 訓練効率（表 5）

| Model | GPU Memory | 1 epoch | Total | AP |
|---|---|---|---|---|
| Faster RCNN | 13 GB | 60 min | 108 ep | 42.0 |
| DETR | 26 GB | 16 min | 300 ep | 41.2 |
| Deformable DETR | 16 GB | 55 min | 50 ep | 45.4 |
| **DINO** | **16 GB** | **55 min** | **12 ep** | **47.9** |

DETR の 300 epoch を 12 epoch に短縮。

### Encoder/Decoder 層数のアブレーション（表 6）

| Enc/Dec | 6/6 | 4/6 | 3/6 | 2/6 | 6/4 | 6/2 | 2/4 | 2/2 |
|---|---|---|---|---|---|---|---|---|
| AP | 47.4 | 46.2 | 45.8 | 45.4 | 46.0 | 44.4 | 44.1 | 41.2 |

- Decoder を 2 層に減らしても -3.0 AP のみ（Dynamic DETR の -13.8 AP と対照的）
- Mixed query selection が decoder layer refinement への依存を減らしている証拠

### DN クエリ数のアブレーション（表 7）

| #DN | 100 CDN | 1000 DN | 200 DN | 100 DN | 50 DN | 10 DN | No DN |
|---|---|---|---|---|---|---|---|
| AP | **47.9** | 47.6 | 47.4 | 47.4 | 46.7 | 46.0 | 45.1 |

- **100 CDN ペアが DN のどの設定（10〜1000）よりも良い**
- 100 を超えるとほぼ飽和（dynamic DN groups で実装、Appendix 0.D.1）

---

## DETR ファミリーの集大成としての位置

```
2020 May  DETR (Carion et al.) — 集合予測パラダイム、500 epoch 必要
            │
2020 Oct  Deformable DETR (Zhu et al., ICLR 2021) — sparse attention、50 epoch に
            │  ┌─ "Look Forward Once"（勾配 detach）→ DINO で "Twice" に拡張
            │  └─ Query Selection (two-stage バリアント) → DINO で "Mixed" に拡張
            │
2022 Jan  DAB-DETR (Liu et al., ICLR 2022) — 4D anchor box queries
            │  └─ Dynamic anchor box → DINO がそのまま採用
            │
2022 Mar  DN-DETR (Li et al., CVPR 2022) — denoising training
            │  └─ DN queries → DINO で "Contrastive DN" に拡張
            │
2022 Mar  ★ DINO-detector (Zhang et al., ICLR 2023) ★
            │  ├─ CDN（正例 + 負例）
            │  ├─ Mixed Query Selection
            │  └─ Look Forward Twice
            │  → COCO test-dev 63.3 AP、初の end-to-end Transformer SOTA
            │
2022 Sep  DETA (Ouyang-Zhang et al.) — 1 対多マッチング
            │  → PE PEspatial が採用、COCO 66.0 box AP
            │
2023      CoDETR (Zong et al., ICCV 2023) — 協調訓練
            │
2023      ★ Grounding DINO ([[entities/grounding-dino]] / [[sources/grounding-dino]], Liu et al., ECCV 2024) — open-vocab 拡張、GLIP × DINO 検出器、ODinW ZS 26.1 SOTA
2024      MM-Grounding-DINO — マルチモーダル拡張
2025      PE PEspatial / SAM 3 — DINO 系統技術の継承
```

**DINO 検出器の位置**: DETR を「研究用おもちゃ」から「実用 SOTA」に押し上げた決定的研究。

---

## SSL の DINO との関係（**重要な注記**）

[[entities/dino]] / [[sources/dino-emerging-properties-in-self-supervised-vit]] の DINO は **物体検出器の DINO とは完全に別物**：

| 軸 | SSL DINO（Caron et al., 2021） | **DINO 検出器**（Zhang et al., 2022） |
|---|---|---|
| 著者 | Meta AI Research（Caron, Touvron, Misra, Jégou ら） | HKUST/清華大/IDEA（Zhang, Li, Liu, Zhang ら） |
| 名前の由来 | self-**DI**stillation with **NO** labels | **D**ETR with **I**mproved de**N**oising anch**O**r box |
| タスク | 自己教師あり表現学習 | 物体検出 |
| 系譜 | [[concepts/self-supervised-learning]] | [[concepts/object-detection]] / DETR ファミリー |
| 出版 | ICCV 2021 | ICLR 2023 |
| 重要技法 | self-distillation, multi-crop, centering, sharpening | CDN, Mixed QS, LFT |
| 後継 | DINOv2, DINOv3, iBOT | DETA, CoDETR, Grounding DINO |
| 命名の重複 | 偶然 | 偶然 |

両者を併用する研究もある（例: DINOv2 backbone + DINO-detector head）。本 wiki では slug で完全分離：

- SSL DINO: `entities/dino` / `sources/dino-emerging-properties-in-self-supervised-vit`
- 検出器 DINO: `entities/dino-detector` / `sources/dino-detector`

---

## 主要な貢献まとめ

1. **DETR ファミリーが初めて COCO leaderboard SOTA に**（test-dev 63.3 AP）
2. **end-to-end Transformer 検出器がスケールすることを実証**: SwinV2-G の 1/15 パラメータ、Florence の 1/60 backbone データ
3. **3 つの新技法**:
   - **Contrastive DeNoising** — 重複予測抑制 + 小物体改善
   - **Mixed Query Selection** — encoder の位置情報のみ活用、コンテンツは学習可能
   - **Look Forward Twice** — 後段層の勾配で前段 box 予測を補正
4. **訓練の劇的効率化**: DETR 500 epoch → DINO 12 epoch（同 AP 比較で大幅改善）
5. **小物体性能の革新**: AP_S +7.5（FPN の革新と並ぶインパクト）
6. **DETR ファミリーの研究方向性を確定**: CoDETR / DETA / Grounding DINO / MM-Grounding-DINO の祖

---

## 関連ページ

- [[sources/dino-detector]] — 原典の要約
- [[translations/dino-detector]] — 原典の和訳
- [[sources/detr]] / [[entities/detr]] — DETR の祖
- [[concepts/object-detection]] — 物体検出全体（DINO は DETR ファミリーの集大成）
- [[concepts/foundation-model]]
- [[entities/dino]] / [[sources/dino-emerging-properties-in-self-supervised-vit]] — **同名の SSL 手法（完全に別物）**
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — DETA decoder で COCO 66.0 box AP、DINO 系統の後継
- [[sources/sam-3]] / [[entities/sam-3]] — DETR + DAC-DETR + Deformable DETR の DETR ファミリー技術を採用
- [[concepts/vision-transformer]] — Transformer の物体検出への適用
