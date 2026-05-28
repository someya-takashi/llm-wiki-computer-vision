---
type: source
source_path: raw/papers/Grounding DINO_ Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.md
source_kind: paper
title: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
authors: [Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang]
year: 2023
venue: ECCV 2024
ingested: 2026-05-28
tags: [grounding-dino, open-set-detection, dino-detector, glip, phrase-grounding, referring-expression, tight-fusion, idea, tsinghua, microsoft]
translation: [[translations/grounding-dino]]
---

# Grounding DINO: DINO 検出器と Grounded Pre-Training を結婚させる

> 原典: [[translations/grounding-dino]] ・ `raw/papers/Grounding DINO_ Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.md`
> 著者: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang ら（清華大学 / IDEA / HKUST / Microsoft Research）
> 出典: ECCV 2024 / arXiv 2303.05499
> arXiv: <https://arxiv.org/abs/2303.05499>
> コード: <https://github.com/IDEA-Research/GroundingDINO>

## 一言まとめ

**「[[entities/glip]] × [[entities/dino-detector]] = Grounding DINO」**。GLIP の **「物体検出 = phrase grounding として再定式化」** の発想を、GLIP の DyHead 検出器ではなく **DETR ファミリーの集大成である DINO 検出器** に移植することで、open-set 物体検出を実用 SOTA に押し上げた。3 つの新しい工夫: (1) **tight 3-phase fusion**（GLIP は Phase A のみ、OV-DETR は Phase B のみだったところを **A+B+C 全フェーズで融合**）、(2) **Language-Guided Query Selection**（テキストに関連する画像 region を上位 K 個選んで decoder queries に）、(3) **Sub-sentence level text representation**（GLIP の word-level と他の sentence-level の中間）。COCO ゼロショット **52.5 AP**（GLIP-L 49.8 超え）、fine-tune **63.0 AP**、ODinW ゼロショット **26.1 mean AP**（Florence 25.8 超え）、RefCOCO/+/g で SOTA。**REC を open-set 評価に拡張** したのも重要貢献。**SAM 3 や MM-Grounding-DINO の直接の祖**。

## 背景と問題意識

2022 年末時点の open-set 検出研究の状況：

| 手法 | 検出器ベース | 融合フェーズ | 弱点 |
|---|---|---|---|
| **ViLD**（2021） | Mask R-CNN | なし（CLIP 蒸留） | CLIP の image-level 知識のみ |
| **MDETR**（2021） | DETR | A + C | テキスト条件を early fusion のみ |
| **GLIP**（2022, [[entities/glip]]） | **DyHead**（古典的 one-stage） | **A のみ** | DyHead の限界、scaling 不足 |
| **OV-DETR**（2022） | Deformable DETR | B のみ | query レベル融合のみ |
| **OWL-ViT**（2022） | ViT + DETR | late fusion | CLIP の two-tower 構造 |
| **DetCLIP**（2022） | ATSS | なし | 古典的検出器の限界 |

GLIP の問題意識:
- GLIP の発想は素晴らしいが、**DyHead 検出器ベース** だったため、(1) Transformer ベース言語モデルとの構造的不一致、(2) スケーリング能力の制約、(3) NMS が必要な古典的設計
- 一方で **DINO 検出器**（[[entities/dino-detector]]）は ICLR 2023 で「**初の end-to-end Transformer COCO SOTA**（63.3 AP）」を達成

→ Grounding DINO の問題意識: **「GLIP の grounded pre-training を、最強の DETR 系検出器 DINO に持ち込めば、open-set 検出の SOTA を取れるはず」**

> **補足: 著者の系譜** — 第一著者 Shilong Liu と Hao Zhang, Feng Li は [[entities/dino-detector]] と同じ IDEA / HKUST グループ。Chunyuan Li, Jianwei Yang は GLIP 著者と同じ Microsoft Research。**両研究の著者が直接協力** して作った正統な統合論文。

## 提案手法 / 主張

### 全体構造

```
画像 → Swin-T/L backbone → multi-scale image features ─┐
                                                       │
                                       Feature Enhancer（6 層）
                                       ├── Image: Deformable self-attention
                                       ├── Text: vanilla self-attention
                                       ├── X-MHA: image-to-text + text-to-image cross-attention
                                       │
テキスト → BERT-base ──────────────────┘
            ("person. bicycle. car. ...")
                                                       │
                                       ★ Language-Guided Query Selection ★
                                       （テキストと最も関連する top-K 画像 region を選択）
                                                       │
                                       Cross-Modality Decoder（6 層、900 queries）
                                       各層:
                                       ├── self-attention
                                       ├── image cross-attention (deformable)
                                       ├── text cross-attention（★DINO に対する追加点）
                                       └── FFN
                                                       │
                                                       ↓
                                       各 query → (box, テキスト token 確率)
                                       損失: L1 + GIoU + contrastive focal loss
```

### 鍵 1: Tight 3-Phase Fusion（A + B + C）

**閉集合検出器を 3 フェーズに分解**（図 2）:

```
[backbone] → [neck (Phase A)] → [query init (Phase B)] → [head (Phase C)]
              ↑                    ↑                       ↑
           特徴強化              query 生成               box/class 予測
```

各先行研究の融合フェーズ:

| 手法 | A（neck） | B（query init） | C（head） |
|---|---|---|---|
| GLIP | ✅ | ❌ | ❌ |
| OV-DETR | ❌ | ✅ | ❌ |
| MDETR | ✅ | ❌ | ✅ |
| **Grounding DINO** | **✅** | **✅** | **✅** |

**なぜ全フェーズが必要か**: 検索（retrieval）タスクは効率のため late-fusion（CLIP-like two-tower）が好まれるが、**open-set 検出では画像とテキストが最初から両方利用可能**。両者を最初から早期かつ複数フェーズで融合する方が、両モダリティの整列が深まる。

> **補足: なぜ古典的検出器では難しかったか** — Faster R-CNN のような検出器は RoI Pooling や RPN という Transformer と相性の悪いモジュールを持つ。一方 DINO 検出器（[[entities/dino-detector]]）は **層ごとに Transformer ブロックで構成** されているため、各層に言語 cross-attention を挿入しやすい。「**Transformer 検出器の構造が言語との融合に決定的に有利**」が本論文の重要な洞察。

### 鍵 2: Feature Enhancer（Phase A）

GLIP の X-MHA から継承して、**6 層の feature enhancer** を構築：

各層の構成:
- **画像側**: Deformable self-attention（DINO 検出器由来）→ image-to-text cross-attention
- **テキスト側**: vanilla self-attention → text-to-image cross-attention

これにより、**画像特徴がテキスト文脈に conditioned され、テキスト特徴が画像内容に conditioned される**。

### 鍵 3: Language-Guided Query Selection（Phase B）

DINO 検出器の **Mixed Query Selection**（[[entities/dino-detector]] の §3.4）を **言語誘導版** に拡張：

```python
# image_features: (bs, num_img_tokens, ndim)
# text_features: (bs, num_text_tokens, ndim)
# num_query: 900

# 各 image token × 各 text token の内積（alignment score）
logits = torch.einsum("bic,btc->bit", image_features, text_features)
# (bs, num_img_tokens, num_text_tokens)

# 各 image token で最も近い text token の score を取る
logits_per_img_feat = logits.max(-1)[0]  # (bs, num_img_tokens)

# top-K image tokens を選択 → これらの位置で decoder queries を初期化
topk_proposals_idx = torch.topk(logits_per_img_feat, num_query, dim=1)[1]
```

つまり「**入力テキストと最も関連する画像位置 K=900 個を選んで decoder queries の位置部分にする**」。content 部分は学習可能（DINO 検出器の Mixed Query Selection と同じ思想）。

> **補足: なぜ言語誘導か** — DINO 検出器の Mixed Query Selection は「物体性スコア（objectness）」で top-K を選ぶが、open-set では「テキストプロンプトに該当するか」が物体性の代わりになるべき。これを言語特徴との内積で実装。

### 鍵 4: Cross-Modality Decoder（Phase C）

DINO 検出器の decoder 層に **text cross-attention を追加**：

```
DINO decoder 層:           [self-attn] → [image cross-attn] → [FFN]
Grounding DINO decoder 層: [self-attn] → [image cross-attn] → [TEXT cross-attn] → [FFN]
                                                              ↑追加
```

各 query は layer ごとに、画像とテキスト両方から情報を取り込む。

### 鍵 5: Sub-Sentence Level Text Representation（§3.4）

GLIP の "person. bicycle. car. ..." プロンプトには問題があった：

**Word-level（GLIP の方式）**: 全カテゴリを 1 つの BERT forward でエンコード。だが、**異なるカテゴリ名の単語が attention で相互作用** してしまう（"person" の文脈で "bicycle" の attention が立つ等）。

**Sentence-level**: 各 phrase を独立にエンコード。**単語の細粒度情報を失う**。

**Sub-sentence level（提案）**:
- **Attention mask** で異なるカテゴリ間の attention を遮断
- 同じカテゴリ内の単語間 attention は許可（"traffic light" は "traffic" と "light" の関係を保つ）
- **カテゴリ独立性 + 単語粒度** の両得

### 鍵 6: REC 評価への拡張

先行 open-set 検出研究は **novel カテゴリ**（"new class names"）のみで評価。Grounding DINO は **指示表現理解 (REC)** を加える：

- 「**赤いシャツの男性**」「**左から 2 番目の犬**」のような属性付き指示
- データセット: RefCOCO, RefCOCO+, RefCOCOg
- 各テキスト入力に対して最高スコアの box を出力

これにより、open-set 検出の評価がより包括的になる。

## 実験結果と知見

### COCO ゼロショット → fine-tune（表 2）

| Model | Pre-train | Zero-Shot AP | Fine-Tune AP |
|---|---|---|---|
| DyHead-T† | O365 | 43.6 | 53.3 |
| DINO(Swin-T)† | O365 | 46.2 | 56.9 |
| GLIP-T (B) | O365 | 44.9 | 53.8 |
| GLIP-T (C) | O365 + GoldG | 46.7 | 55.1 |
| GLIP-L | FourODs + GoldG + Cap24M | 49.8 | 60.8 |
| **Grounding-DINO-T** | O365 + GoldG | **48.1 (+1.4 vs GLIP-T(C))** | 57.1 |
| **Grounding-DINO-L** | O365 + OI + GoldG | **52.5** | **62.7 (63.0 w/ TTA)** |

- **GLIP-L の 49.8 を 52.5 で上回り**、**DINO 検出器 SwinL の closed-set 62.5 もゼロショットで超える**
- fine-tune では DINO 検出器 SwinL の 63.3 にほぼ匹敵（62.7 / 63.0 with TTA）

### LVIS ゼロショット（表 3）

| Model | Pre-train | MiniVal AP | APr | APc | APf |
|---|---|---|---|---|---|
| GLIP-T (C) | O365 + GoldG | 24.9 | 17.7 | 19.5 | 31.0 |
| GLIP-T | O365 + GoldG + Cap4M | 26.0 | 20.8 | 21.4 | 31.0 |
| **Grounding-DINO-T** | O365 + GoldG | 25.6 | 14.4 | 19.6 | 32.2 |
| **Grounding-DINO-T** | O365 + GoldG + Cap4M | **27.4** | 18.1 | 23.3 | 32.7 |
| **Grounding-DINO-L** | O365 + OI + GoldG + Cap4M + COCO + RefC | **33.9** | 22.2 | 30.7 | 38.8 |

**重要な発見**: 「**Grounding DINO は common/frequent で GLIP より良いが、rare で悪い**」。論文の解釈：900 queries の固定数が長尾分布で限界。GLIP の DyHead は全 grid を使うので rare に強い。**Grounding DINO の固有の弱点**。

ただし、**Grounding DINO は Cap4M 追加で +1.8 AP（GLIP の +1.1 より大きい）** — **より良い scaling**。

### ODinW ベンチマーク（表 4）

35 データセットの平均 AP：

| Model | Backbone | Size | Zero-Shot Avg | Few-Shot Avg | Full-Shot Avg |
|---|---|---|---|---|---|
| MDETR | ENB5 | 169M | 10.7 | - | - |
| OWL-ViT | ViT-L/14 (CLIP) | >1243M | 18.8 | - | - |
| GLIP-T | Swin-T | 232M | 19.6 | 38.9 | 62.6 |
| GLIPv2-T | Swin-T | 232M | 22.3 | - | - |
| DetCLIP | Swin-L | 267M | 24.9 | - | - |
| Florence | CoSwinH | ≈841M | 25.8 | - | - |
| **Grounding-DINO-T** | Swin-T | 172M | **22.3** | **46.4** | **70.7** |
| **Grounding-DINO-L** | Swin-L | 341M | **26.1** | - | - |

- **ゼロショット**: Florence (841M params) を 341M で上回る **26.1 AP**
- **Few-shot**: GLIP の 38.9 を 46.4 で大幅超え
- **Full-shot**: GLIP の 62.6、**DINO 検出器 Swin-L の 68.8 すらも 70.7 で超える**

### RefCOCO/+/g (表 5)

GLIP-T と同条件（O365 + GoldG + Cap4M）で：

| Model | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---|---|---|
| GLIP-T | 50.42 | 49.50 | 66.09 |
| **Grounding-DINO-T** | 50.41 | 51.40 | 67.46 |

RefC を訓練に追加 + fine-tune した **Grounding-DINO-L で SOTA**（RefCOCO val 90.56）。

### ablation（表 6）

| 構成 | COCO ZS | LVIS ZS |
|---|---|---|
| Full Model | 46.7 | 16.1 |
| - encoder fusion（Phase A 削除） | 45.8 (-0.9) | 13.1 (-3.0) |
| - language-guided query selection（Phase B 削除） | 46.3 (-0.4) | 13.6 (-2.5) |
| - text cross-attention（Phase C 削除） | 46.1 (-0.6) | 14.3 (-1.8) |
| - sub-sentence（word-level に戻す） | 46.4 (-0.3) | 15.6 (-0.5) |

**Encoder fusion (Phase A) が最重要**、**LVIS では Phase B も決定的**。Sub-sentence は影響最小だが有用。

### DINO から Grounding DINO への転送（表 7）

事前学習された DINO 検出器の重みを使い、共有モジュールを凍結して text + fusion ブロックのみ訓練：

| 設定 | COCO ZS | LVIS ZS | ODinW ZS |
|---|---|---|---|
| Grounding-DINO-T (from scratch, O365 + GoldG) | 48.1 | 25.6 | 20.0 |
| Grounding-DINO-T (from pre-trained DINO, O365 + GoldG) | 46.4 | **26.1** | 18.5 |

**DINO 事前学習からの転送が標準訓練を LVIS で上回る** → モデル訓練に改善余地あり。**収束も DINO 事前学習の方が遥かに早い**（図 5）。実用的に重要な知見。

## 限界・批判的視点

- **Segmentation に使えない**: GLIPv2 のような mask head がない（論文の §5 限界で明言）
- **LVIS rare で GLIP に劣る**: 900 queries 固定が長尾分布に不利。1500+ にスケールすれば改善か
- **訓練データが GLIP-L より少ない**: Grounding-DINO-L は 27M データに到達せず、最終性能を制限している可能性
- **REC fine-tuning が必須**: ゼロショットだけでは GLIP 並み、RefC を訓練に入れて初めて飛躍
- **Closed-set で純粋 DINO に劣る**: 表 9 で DINO-4scale 49.0 vs Grounding DINO 48.1。追加コンポーネントが最適化を難しくしている
- **テキスト依存**: 推論時に必ずテキストプロンプトが必要、image-only 使用不可（GLIP と同じ）

## 既存 wiki との接続

### GLIP × DINO 検出器の正統な統合

[[entities/glip]] と [[entities/dino-detector]] の **構造的・著者的・思想的な統合**:

| 軸 | [[entities/glip]] | [[entities/dino-detector]] | **Grounding DINO** |
|---|---|---|---|
| 著者所属 | UCLA + Microsoft | IDEA + 清華 + HKUST | **両者共同**（GLIP 著者 Chunyuan Li, Jianwei Yang + DINO 検出器著者 Shilong Liu, Hao Zhang, Feng Li） |
| 検出器ベース | DyHead（古典） | 純 DINO 検出器（DETR 系） | **DINO 検出器** |
| 融合フェーズ | A のみ | n/a（言語なし） | **A + B + C** |
| Text prompt 設計 | word-level | n/a | **sub-sentence level** |
| COCO ZS | 49.8 (GLIP-L) | n/a（closed-set のみ 62.5） | **52.5 (GD-L)** |
| 訓練最適化 | 古典的 | CDN + Mixed QS + LFT | **DINO 流継承 + 言語追加** |

### Open-Vocabulary 検出系統での位置

[[concepts/object-detection]] の系譜上の位置：

```
2018  Bansal et al. — Glove ベース（祖）
2021  CLIP                — image-level open-vocab
2021  ViLD                — CLIP 蒸留
2021 Oct  MDETR           — DETR + テキスト条件
2021 Dec  ★ GLIP ★        — 検出 = grounding、DyHead ベース
2022  OWL-ViT             — ViT + DETR、Google
2022  GLIPv2              — GLIP + segmentation
2023 Mar  ★ Grounding DINO（本論文）★ — GLIP × DINO 検出器
            │
2024      MM-Grounding-DINO — マルチモーダル拡張、open-vocab タスク統合
2024      DINO-X            — Grounding DINO 後継
2025      ★ SAM 3 ★         — Grounding DINO + マスク生成 + presence head
```

### SAM 3 への流れ

[[entities/sam-3]] の **PCS タスク** は Grounding DINO の直接の発展：

| 軸 | **Grounding DINO** | **SAM 3** |
|---|---|---|
| 検出 | DINO 検出器ベース | DETR-based + presence head |
| 画像 encoder | Swin-T/L | **Perception Encoder**（[[entities/perception-encoder]]） |
| テキスト encoder | BERT | PE のテキスト塔 |
| 融合 | 3-phase tight fusion | Fusion encoder |
| マスク出力 | ❌（限界として論文に明記） | ✅（MaskFormer 適応） |
| 認識/位置分離 | ❌ | ✅ presence head |
| 動画対応 | ❌ | ✅（SAM 2 風 tracker） |

SAM 3 は本論文の限界（segmentation 不可、認識/位置混在）を解決した発展形。

### DETR ファミリーへの貢献

[[entities/detr]] / [[entities/dino-detector]] の系譜で、Grounding DINO は **「DETR 系で初めて open-set 検出 SOTA を取った」** 研究。後継の DINO-X, MM-Grounding-DINO, さらには PE PEspatial の DETA decoder（[[entities/perception-encoder]]）まで、本論文の思想が広く影響している。

## 用語と略称

- **Grounding DINO**（本論文）= DINO 検出器 + GLIP 流 grounded pre-training の統合
- **DINO**（検出器版）= [[entities/dino-detector]]（**SSL の DINO [[entities/dino]] とは別物**）
- **GLIP** = Grounded Language-Image Pre-training（[[entities/glip]]）
- **open-set object detection** = 訓練に含まれない novel カテゴリを言語で指定して検出するタスク
- **closed-set object detection** = 事前定義された固定カテゴリ集合のみを検出する古典的タスク
- **REC** = Referring Expression Comprehension（指示表現理解）。「赤いシャツの男性」のような属性付き指示で物体を localize
- **tight fusion** = 浅い late-fusion（CLIP のような末端での dot product）ではなく、複数フェーズで早期から深く融合する戦略
- **3-phase fusion** = neck (A) + query init (B) + head (C) の 3 フェーズでの融合
- **feature enhancer** = Phase A の融合モジュール（Deformable self-attn + X-MHA）
- **language-guided query selection** = Phase B、テキスト関連性で top-K 画像 region を選ぶ
- **cross-modality decoder** = Phase C、DINO 検出器の decoder に text cross-attn を追加
- **X-MHA** = Cross-Modality Multi-Head Attention（GLIP 由来、[[sources/glip]]）
- **sub-sentence level representation** = 異なるカテゴリ間の attention を mask で遮断、単語粒度を保つ
- **word level representation** = GLIP の方式、全単語を 1 つの BERT forward でエンコード（カテゴリ間相互作用あり）
- **sentence level representation** = 各 phrase を独立にエンコード（細粒度損失）
- **mixed query selection** = DINO 検出器の Phase B（[[sources/dino-detector]]）、位置のみ encoder から、content は学習可能
- **dynamic anchor box** = DAB-DETR 由来の 4D anchor（[[sources/dino-detector]]）
- **deformable attention** = Deformable DETR 由来の sparse attention（[[entities/dino-detector]]）
- **BERT-base** = Hugging Face の BERT、テキスト encoder
- **Swin Transformer** = 階層型 ViT（Liu et al., 2021）、画像 backbone
- **DyHead** = GLIP の検出器ベース、Grounding DINO が置き換えたもの
- **OV-DETR** = Open-Vocabulary DETR（CLIP queries を decoder 入力に）
- **MDETR** = Modulated DETR（GLIP より先、テキスト条件 DETR）
- **OWL-ViT** = ViT-based open-vocab 検出（Google）
- **DetCLIP** = ATSS-based open-vocab 検出（large-scale captioning）
- **GLIPv2** = GLIP + segmentation 拡張（Zhang et al., NeurIPS 2022）
- **Florence** = Microsoft の VLM foundation model（CoSwinH backbone, FLD-900M）
- **RefCOCO / RefCOCO+ / RefCOCOg** = REC 標準ベンチマーク（COCO に追加注釈）
- **RefC** = RefCOCO + RefCOCO+ + RefCOCOg の総称
- **GoldG** = MDETR キュレーションの 0.8M 人手注釈 grounding データ
- **Cap4M / Cap24M** = GLIP 生成の 4M / 24M 画像-テキストペア
- **FourODs** = GLIP の 2.66M 検出データ集約
- **O365v1 / O365v2** = Objects365 の v1（0.6M）と v2（1.7M）
- **OI** = OpenImages
- **VG** = Visual Genome
- **YFCC** = Yahoo Flickr Creative Commons 100M
- **PhraseCut** = phrase ベースのセグメンテーションデータセット

## 関連ページ

- [[translations/grounding-dino]] — 本文全文の和訳
- [[entities/grounding-dino]] — Grounding DINO モデルの詳細スペック
- [[entities/glip]] / [[sources/glip]] — GLIP（本研究の grounded pre-training 側の祖）
- [[entities/dino-detector]] / [[sources/dino-detector]] — DINO 検出器（本研究の検出器側の祖）
- [[entities/detr]] / [[sources/detr]] — DETR の祖
- [[concepts/object-detection]] — 物体検出全体（Grounding DINO は open-vocab 検出 SOTA）
- [[concepts/zero-shot-transfer]] — Grounding DINO のゼロショット転送
- [[concepts/promptable-concept-segmentation]] — SAM 3 PCS の前駆として Grounding DINO
- [[entities/sam-3]] / [[sources/sam-3]] — Grounding DINO + マスク生成 + presence head
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — PE PEspatial の検出 decoder（DETA）も DINO 検出器系統
- [[entities/yolo-world]] / [[sources/yolo-world]] — 同じ open-vocab 系統の対照軸（精度志向 vs 速度志向、CVPR 2024）
- [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — **直接の後継**（2024 May、Pro/Edge 双子スイート）
- [[entities/dino-x]] / [[sources/dino-x]] — IDEA 系譜の現時点最新版（2024 Nov、unified perception model）
