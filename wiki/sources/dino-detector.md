---
type: source
source_path: raw/papers/DINO_ DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.md
source_kind: paper
title: "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection"
authors: [Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum]
year: 2022
venue: ICLR 2023
ingested: 2026-05-28
tags: [dino-detector, object-detection, transformer, detr-family, denoising, contrastive-denoising, query-selection, look-forward-twice, idea]
translation: [[translations/dino-detector]]
---

# DINO（検出器）: 改良された DeNoising Anchor Boxes を持つ DETR

> 原典: [[translations/dino-detector]] ・ `raw/papers/DINO_ DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.md`
> 著者: Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum（HKUST / Tsinghua / IDEA / HKUST(GZ)）
> 出典: ICLR 2023 / arXiv 2203.03605
> arXiv: <https://arxiv.org/abs/2203.03605>
> コード: <https://github.com/IDEACVR/DINO>

> **重要な注記（名前の衝突）**: この **DINO は物体検出器（Zhang et al., 2022）**。自己教師あり学習の DINO（Caron et al., ICCV 2021、[[entities/dino]] / [[sources/dino-emerging-properties-in-self-supervised-vit]]）とは **完全に別物** で、著者・所属（Meta vs HKUST/IDEA）・タスク（SSL 表現学習 vs 物体検出）・年（2021 vs 2022）すべて異なる。両方とも「DINO」と称するが偶然の重複。slug `dino-detector` で区別。

## 一言まとめ

**「DETR ファミリーの集大成」**。DAB-DETR の動的 anchor box、DN-DETR の denoising 訓練、Deformable DETR の deformable attention + query selection を統合し、さらに **3 つの新技法**（**Contrastive DeNoising / Mixed Query Selection / Look Forward Twice**）を追加。COCO で 12 epoch で **49.4 AP**（DN-DETR の 43.4 から +6.0）、SwinL + Objects365 事前学習で test-dev **63.3 AP**。**初めて end-to-end Transformer 検出器が COCO leaderboard の SOTA になった**転換点で、SwinV2-G の **1/15 のパラメータ** で上回る。Grounding DINO や MM-Grounding-DINO など多くの open-vocab 検出研究の基盤になった。

## 背景と問題意識

[[sources/detr]]（Carion et al., ECCV 2020）以降の DETR ファミリーの問題：

| モデル | 主な貢献 | 残った課題 |
|---|---|---|
| **DETR**（2020） | 集合予測パラダイム、NMS/anchor 排除 | 訓練 500 epoch 必要、小物体弱い |
| **Deformable DETR**（2021） | sparse attention、訓練 50 epoch に短縮 | 50 AP 未満で停滞 |
| **DAB-DETR**（2022 Jan） | queries を **4D anchor box** として明示的に解釈 | 二部マッチングの不安定性 |
| **DN-DETR**（2022 Mar） | **denoising 訓練**でマッチング不安定性を解決 | 「物体なし」予測能力が弱い、依然 50 AP 未満 |

2022 年時点の最重要課題：
1. **DETR ファミリーは依然 COCO で 50 AP 未満** — HTC++ や DyHead のような古典的検出器に負けていた
2. **DETR ファミリーは大規模化されていなかった** — Florence (900M) や SwinV2-G (3.0B) のようなスケール実験がない
3. **「物体なし」を学ぶ仕組みがない** — DN-DETR は近くに GT がある anchor を学ぶが、近くにない anchor を「拒絶」できない

→ DINO 検出器の目標: **DETR ファミリーを 50 AP の壁を越え、COCO leaderboard で SOTA を取る**。

## 提案手法 / 主張

### 全体構造（DETR ライク + 3 つの新技法）

```
画像 → backbone（ResNet-50 or SwinL）→ multi-scale 特徴
         ↓
   Transformer encoder（6 層、deformable attention）
         ↓
   ★ Mixed Query Selection ★ ← top-K encoder 特徴から位置クエリを初期化
                                   コンテンツクエリは静的・学習可能なまま
         ↓
   Transformer decoder（6 層、deformable attention）
         ├── 通常クエリ（900 個、4D anchor box として）
         └── ★ Contrastive DN クエリ ★ ← 正例（小ノイズ）+ 負例（大ノイズ）の対
         ↓
   ★ Look Forward Twice ★ ← 後段層の勾配で前段の box 予測を補正
         ↓
   各層の出力 → 共有予測 FFN → (class, bbox)
```

### 鍵 1: Contrastive DeNoising (CDN)

**動機**: DN-DETR は「近くに GT があるノイズ box を GT に戻す」訓練をする → 「近くに GT がない box は何も学んでいない」。重複予測（同じ物体に複数のスロットが活性化）を抑える能力に欠ける。

**仕組み**: 2 つのノイズスケール $\lambda_1 < \lambda_2$ を導入し、各 GT box について：

```
        外側正方形（λ₂）
           ┌─────────┐
           │ 負クエリ  │
           │   ┌───┐  │     正クエリ: GT を再構成する
           │   │ + │  │     負クエリ: 「物体なし」を予測
           │   └───┘  │
           │ 内側(λ₁) │
           └─────────┘
```

各 GT に対し **正クエリ + 負クエリのペア** を生成。負クエリには **focal loss で「背景」を予測** することを学習させる。

**効果**: 
- 重複予測の抑制（図 8: DN 単独だと 1 人の少年に 3 個のボックス、CDN だと 1 個）
- 小物体で **+1.3 AP**（ATD(k) 解析で実証、図 4）

> **補足: なぜ「contrastive」と呼ぶか** — 同じ GT に対して「positive（小ノイズ）vs negative（大ノイズ）」のペアを与えるため。対比学習の語法を借りているが、損失自体は CE/focal loss であり、SimCLR のような InfoNCE ではない。

### 鍵 2: Mixed Query Selection

**3 つのクエリ初期化方式の比較**（図 5）：

| 方式 | 位置クエリ | コンテンツクエリ | 代表モデル |
|---|---|---|---|
| **(a) Static**（両方静的） | 学習可能 | 学習可能 or 0 ベクトル | DETR / DN-DETR / DAB-DETR |
| **(b) Pure Selection** | encoder top-K から | encoder top-K から | Deformable DETR (two-stage) / Efficient DETR |
| **(c) Mixed**（提案）| **encoder top-K から** | **学習可能（静的）** | DINO |

**論理**: 「encoder の top-K 特徴は **位置情報としては有用**（『ここに物体がある可能性』）だが、**コンテンツとしては未精錬**（『複数物体を含む or 物体の一部』）」。だから位置だけを使い、コンテンツは学習可能のまま残す。

**効果**: ablation の +0.5 AP（行 3→4、Pure 46.5 → Mixed 47.0）。decoder 層を 2 層に削減しても性能低下が小さい副次効果あり（mixed QS が decoder layer refinement への依存を減らすため）。

### 鍵 3: Look Forward Twice (LFT)

**Deformable DETR の "Look Forward Once"**: 層 $i$ のパラメータは層 $i$ の損失のみで更新（勾配を $i-1$ から detach）。**訓練を安定化させるが、後段層の知恵を前段に伝えない**。

**Look Forward Twice**: 層 $i+1$ の予測 box を、層 $i$ の detach されていない $b_i^{\prime}$ + 層 $i+1$ のオフセット $\Delta b_{i+1}$ で計算 → **層 $i+1$ の損失が層 $i$ のパラメータも更新**。

```
従来 (LFO):  b_{i-1} → [Layer_i] → b_i^pred (loss_i のみが Layer_i を更新)
                                detach → b_i
                                          ↓
                                  [Layer_{i+1}] → b_{i+1}^pred (loss_{i+1} のみが Layer_{i+1} を更新)

DINO (LFT):  b_{i-1} → [Layer_i] → b_i^pred (loss_i が Layer_i を更新)
                                  b_i^prime (detach なし)
                                          ↓
                                  [Layer_{i+1}] → b_{i+1}^pred (loss_{i+1} が Layer_i, Layer_{i+1} を更新)
```

**効果**: ablation の +0.4 AP（行 4→5、47.0 → 47.4）。

## 実験結果と知見

### COCO 検出 12 epoch（表 1）— ResNet-50 backbone

| Model | AP | AP_S | AP_L |
|---|---|---|---|
| Faster-RCNN | 37.9 | 22.4 | 49.1 |
| DETR(DC5) | 15.5 | 4.3 | 26.7 |
| Deformable DETR | 41.1 | - | - |
| DN-Deformable-DETR | 43.4 | 24.8 | 59.4 |
| **DINO-4scale** | **49.0 (+5.6)** | **32.0 (+7.2)** | 63.0 |
| **DINO-5scale** | **49.4 (+6.0)** | **32.3 (+7.5)** | 63.9 |

**DETR の伝統的弱点だった小物体で +7.5 AP** という飛躍。

### COCO 検出 24 epoch（表 2）

DINO-5scale 24 epoch で 51.3 AP、Deformable DETR 50 epoch の 46.2 AP を +5.1 AP 上回る（しかも半分のエポック）。

### COCO SOTA 設定（表 3）— SwinL + Objects365

| Model | Params | Pre-train Detection | val2017 AP | test-dev AP | End-to-end |
|---|---|---|---|---|---|
| SwinV2-G | **3.0B** | IN-22K-ext-70M + O365 | 61.9 | 63.1 | ❌ (HTC++) |
| Florence-CoSwin-H | ≥637M | FLD-9M | - | 62.4 | ❌ |
| **DINO-SwinL** | **218M** | O365 | **63.1** | **63.3** | ✅ |

- **end-to-end（NMS なし）で初めて COCO SOTA**
- **SwinV2-G の 1/15 のパラメータ**で上回る
- **Florence の 1/60 の backbone 事前学習データ + 1/5 の検出事前学習データ**で上回る

### ablation（表 4）

ベースライン → 3 技法すべて適用までの段階：

| 構成 | AP |
|---|---|
| DN-DETR（baseline） | 43.4 |
| + Optimized DN-DETR（engineering） | 44.9 |
| + Pure Query Selection | 46.5 (+1.6) |
| + Mixed QS | 47.0 (+0.5) |
| + Look Forward Twice | 47.4 (+0.4) |
| **+ Contrastive DN（DINO 完成）** | **47.9 (+0.5)** |

CDN + Mixed QS + LFT で合計 **+1.4 AP**。最大の効果はエンジニアリング最適化と pure query selection から（baseline からの差分のほとんど）。

### 訓練効率（表 5、Appendix 0.B）

- 8× A100 で 12 epoch、**1 GPU あたり 2 画像のバッチサイズ**で 16 GB GPU メモリ
- 1 epoch あたり ~55 分（Deformable DETR と同じ）
- 結果として **DETR の 300 epoch を 12 epoch に短縮**

## 限界・批判的視点

- **依然として end-to-end 検出器のオーバーヘッド**: Faster-RCNN（13 GB / GPU）より多いメモリ（16 GB）
- **CDN の hard negative 設計の経験的性質**: $\lambda_1, \lambda_2$ の選び方は理論的根拠というよりも経験的調整
- **小物体性能の限界**: AP_S 32.0 は DETR (20.5) を大幅に超えるが、HTC++ のような mask 系には依然劣る場合あり
- **Mixed Query Selection の補助検出ヘッド**: pure two-stage と同じく追加 head が必要、純粋なシンプルさは失われている
- **「DINO」名の衝突**: SSL の DINO（Caron et al., 2021）と完全に同名で、コミュニティの混乱を招いている

## 既存 wiki との接続

### DETR ファミリーの集大成

[[entities/detr]] の系譜：

```
DETR (2020)
   └── Deformable DETR (2020 Oct) — sparse attention
        └── DAB-DETR (2022 Jan) — 4D anchor box queries
             └── DN-DETR (2022 Mar) — denoising training
                  └── ★ DINO-detector (2022, ICLR 2023) ★ — CDN + Mixed QS + LFT
                       ├── DETA (2022 Sep) — 1 対多マッチング（PE PEspatial が採用）
                       ├── CoDETR (2023) — 協調訓練
                       ├── Grounding DINO (2023, [[sources/grounding-dino]] / [[entities/grounding-dino]]) — open-vocab 拡張、ODinW 26.1 SOTA
                       └── MM-Grounding-DINO (2024) — マルチモーダル拡張
```

DINO-detector は **DETR ファミリーの最重要転換点** — ここで初めて DETR が SOTA に到達した。

### SSL の DINO との混同問題

[[entities/dino]] / [[sources/dino-emerging-properties-in-self-supervised-vit]] と本論文の DINO は **完全に別物**：

| 軸 | SSL DINO（Caron et al., 2021） | DINO 検出器（Zhang et al., 2022） |
|---|---|---|
| 著者 | Meta AI Research（Mathilde Caron ら） | HKUST / 清華大学 / IDEA（Hao Zhang ら） |
| タスク | 自己教師あり表現学習 | 物体検出 |
| 命名の由来 | self-**DI**stillation with **NO** labels | **D**ETR with **I**mproved de**N**oising anch**O**r boxes |
| 学術系統 | [[concepts/self-supervised-learning]] | [[concepts/object-detection]] / [[entities/detr]] |
| 出版年 | ICCV 2021 | ICLR 2023 |

両者は **完全な偶然の重複** であり、文献では文脈で区別される。本 wiki では slug `dino` vs `dino-detector` で完全に区別する。

### PE PEspatial / SAM 3 への影響

DINO-detector の主要発明（CDN, Mixed QS, LFT）の継承：

- [[entities/perception-encoder]] PEspatial が採用する **DETA**（Detection Transformer with Assignment, [Ouyang-Zhang et al., 2022]）は DINO 検出器の後継で、CDN と Mixed QS を含む
- [[entities/sam-3]] の DETR-based detector も **Deformable DETR + box-region-positional bias + DAC-DETR** という DETR ファミリーの累積技術を採用
- [[sources/sam-3]] の検出器設計説明で「DAC-DETR」「Plain-DETR」「Deformable DETR」が出てくる - これらは DINO 検出器以降の DETR 系統

### 物体検出 SOTA 競争への波及

DINO 検出器は **「end-to-end Transformer 検出器が SOTA を取れる」** ことを証明し、その後の研究方向を以下のように決定した：

1. **CoDETR**（2023）: DINO + 補助 head の協調訓練でさらに改善
2. **DETA**（2022）: 1 対多マッチング、PE PEspatial が採用
3. **[[entities/grounding-dino|Grounding DINO]]**（2023, ECCV 2024）: open-vocab 拡張、MDETR の後継、ODinW ZS 26.1 / COCO ZS 52.5 AP の SOTA
4. **DINOv2 with registers** + **DINO detector** など、複数の系譜の名前衝突問題が常態化

[[sources/dinov3]] §6.3.1 で「**Plain-DETR**」が DINOv3 + 凍結バックボーンで COCO 66.1 AP を達成したという結果は、本論文の系譜の延長線上。

## 用語と略称

- **DINO**（本論文）= **D**ETR with **I**mproved de**N**oising anch**O**r boxes — 物体検出 Transformer
- **DETR** = DEtection TRansformer（[[entities/detr]] / [[sources/detr]]、Carion et al., ECCV 2020）
- **DAB-DETR** = Dynamic Anchor Box DETR（[Liu et al., ICLR 2022]）。queries を 4D anchor box として明示的に解釈
- **DN-DETR** = DeNoising DETR（[Li et al., CVPR 2022]）。ノイズ加えた GT を補助訓練に使い収束加速
- **Deformable DETR** = [Zhu et al., ICLR 2021]。sparse attention で計算量と訓練速度を改善
- **CDN** = Contrastive De-Noising（本論文の主要貢献 1）。正例 + 負例の denoising
- **Mixed Query Selection** = 位置クエリのみ encoder から、コンテンツクエリは学習可能（本論文の主要貢献 2）
- **LFT** = Look Forward Twice（本論文の主要貢献 3）。後段層の勾配で前段を補正
- **LFO** = Look Forward Once（Deformable DETR の従来手法）。勾配 detach で訓練安定化するが情報損失
- **dynamic anchor box** = decoder で層ごとに動的更新される 4D box（DAB-DETR の核）
- **denoising training** = ノイズ加えた GT を補助訓練に使う（DN-DETR の核、本論文で contrastive 化）
- **query selection** = encoder の top-K 特徴で decoder queries を初期化（Deformable DETR の two-stage バリアント）
- **iterative box refinement** = 各 decoder 層で box を段階的に精錬（Deformable DETR）
- **positional queries / content queries** = 位置情報 / コンテンツ情報を担当する 2 種のクエリ（Conditional DETR で初導入）
- **DC5** = Dilated C5（ResNet の最終ステージで dilation により解像度倍化）
- **multi-scale features** = 4 スケール（stages 2,3,4 + downsampled stage 4）または 5 スケール（+ stage 1）の特徴ピラミッド
- **focal loss** = [Lin et al., 2017] のクラス不均衡対応損失、$\alpha=0.25, \gamma=2$
- **SwinL** = Swin Transformer Large、[[entities/dinov3]] でも比較対象
- **Objects365** = 365 クラスの大規模検出データセット（1.7M 画像）、DETR ファミリー事前学習の標準
- **ATD(k)** = Average Top-K Distance、anchor が GT からどれだけ離れているかの指標（本論文で導入）
- **GIOU** = Generalized IoU（[Rezatofighi et al., 2019]、スケール不変ボックス類似度）
- **HTC++** / **DyHead** = DINO 以前の COCO SOTA を保持していた古典的検出器
- **Florence** = Microsoft の VLM foundation model（900M 画像-テキスト + 9M 検出事前学習）
- **SwinV2-G** = Microsoft の 3.0B パラメータ Swin Transformer v2 + HTC++
- **TTA** = Test Time Augmentation（マルチスケール + 水平反転）。DETR ライクには NMS と相性が悪い

## 関連ページ

- [[translations/dino-detector]] — 本文全文の和訳
- [[entities/dino-detector]] — DINO 検出器モデルの詳細スペック
- [[sources/detr]] / [[entities/detr]] — DETR の祖、DINO は DETR ファミリーの集大成
- [[concepts/object-detection]] — 物体検出全体の系譜（DINO は DETR ファミリーの集大成）
- [[entities/dino]] / [[sources/dino-emerging-properties-in-self-supervised-vit]] — **同名の SSL 手法**（別物）
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — DETA decoder で COCO 66.0 box AP、DINO 検出器の後継
- [[sources/sam-3]] / [[entities/sam-3]] — DINO 検出器以降の DETR ファミリー（Deformable DETR + DAC-DETR）を採用
- [[concepts/vision-transformer]] — Transformer の CV への適用（DINO は ResNet/SwinL backbone + Transformer ヘッド）
