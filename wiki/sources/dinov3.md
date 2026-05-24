---
type: source
source_path: raw/papers/DINOv3.md
source_kind: paper
title: "DINOv3"
authors: [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, et al.]
year: 2025
venue: "arXiv:2508.10104"
ingested: 2026-05-24
tags: [dinov3, self-supervised-learning, vision-transformer, foundation-model, gram-anchoring, rope, distillation, geospatial]
translation: [[translations/dinov3]]
---

# DINOv3

> 原典: [[translations/dinov3]] ・ `raw/papers/DINOv3.md`
> 著者: Oriane Siméoni ほか（Meta AI Research, Inria, WRI）
> 出典: arXiv:2508.10104（2025）

---

## 一言まとめ

[[entities/dinov2]] の正統な後継。**「凍結 SSL バックボーンで弱教師ありモデルを dense タスクで圧倒し、global タスクでも肩を並べる」**を本格達成した CV 基盤モデル。データ・モデル・効率化の総力戦に加え、本論文の核心である **Gram anchoring**（パッチ間の Gram 行列を過去の teacher に合わせる新しい正則化）で「ViT-L 以上 × 長時間訓練で起きる dense feature 劣化問題」を解決した。結果として ViT-7B（DINOv2-g の 6 倍）を 1.689B 画像（DINOv2 の 12 倍）で訓練し、その上で multi-student distillation で ViT-S/B/L/H+ + ConvNeXt を一斉提供。**凍結バックボーンだけで COCO 検出 mAP 66.1、ADE20K セグメンテーション mIoU 63.0、NYUv2 深度 RMSE 0.279 など軒並み SOTA** を更新。さらに同じレシピで衛星画像版（[[entities/sat-493m]] で学習）を作り、地理空間ベンチマーク 12/15 で SOTA を獲得。

---

## 背景と問題意識

### この論文以前の状況

- **[[entities/dinov2]]（Oquab et al., 2023）** が「**CV における基盤モデル**（[[concepts/foundation-model]]）」として、純粋 SSL で OpenCLIP-G と並ぶ汎用視覚特徴量を作れることを示した。
- それから 2 年、競合は急成長していた：
  - **Perception Encoder (PE)**（[[entities/perception-encoder]]）: Meta の弱教師あり画像-テキストモデル。PEcore（global）と SAM2 蒸留版 PEspatial（dense）の 2 系統
  - **SigLIP / SigLIP 2**（[[entities/siglip]]）: Google が sigmoid 損失でスケールした CLIP 系
  - **AM-RADIO**（NVIDIA）: SAM + CLIP + DINOv2 の凝集蒸留（agglomerative distillation）
  - **EVA-CLIP-18B**: 巨大 CLIP 系
  - **V-JEPA 2**: 動画 SSL の最先端
  - **Web-DINO** / **Franca**: DINO のスケーリング試行
- いずれも DINOv2 を脅かす存在になり、特に **PEspatial と AM-RADIO は SAM から蒸留することで dense タスクを補強**していた。

### この論文が解決すべき問題

1. **データキュレーションのさらなるスケール**: LVD-142M（DINOv2）の 12 倍規模で「テキスト・メタデータ不使用」を維持できるか
2. **ViT-L 以上 × 長時間訓練で起きる dense feature 劣化問題**: classification 性能は単調に上がるが、segmentation 性能は 200k 反復で頭打ち→低下する。DINOv2 でもうっすら観察されていたが、ViT-7B では深刻化
3. **訓練の地平線が不明な問題**: コサインスケジュールは「いつ訓練を終えるか」を事前に決める必要があり、大規模では推測不可能
4. **小型モデルへの効率的蒸留**: 1 つの 7B teacher から複数 student をどう効率的に蒸留するか

> **補足: dense feature 劣化問題とは** — ViT に dense（パッチ単位の細かい意味）と global（[CLS] トークン的な画像全体の意味）を同時に学習させる場合、学習が長引くほど global が支配的になり、パッチ間のコサイン類似度が「どこを見ても無関係なパッチに似てしまう」状態に劣化する。視覚化すると、参照パッチに対して画像全体が均一に光るような無意味な類似度マップになる。これがセグメンテーションや深度推定のような dense 予測タスクの性能を直接押し下げる。

---

## 提案手法

### A. スケールアップ（§3）

#### データ: LVD-1689M

[[entities/lvd-1689m]] を参照。

- **17 億枚の Instagram 公開画像**（プラットフォームレベルのコンテンツモデレーション済み）から自動キュレーション
- 5 階層 k-means クラスタリング（200M → 8M → 800k → 100k → 25k クラスタ）+ バランスサンプリングで **16.89 億枚**を選別
- DINOv2 を画像埋め込みとして使用（**前世代モデルが次世代データキュレーションに利用される**循環構造）
- 各 iteration の **10% は ImageNet-1k のみの均質バッチ**、残り 90% は LVD-1689M 由来の異質バッチ

#### アーキテクチャ: ViT-7B（67 億パラメータ）

DINOv2-g と比較すると：

| | DINOv2 ViT-g | DINOv3 ViT-7B |
|---|---|---|
| パラメータ数 | 1.1B | **6.7B** |
| Embed dim | 1536 | **4096** |
| Heads | 24 | **32** |
| FFN hidden | 4096 | **8192** |
| Patch size | 14 | **16** |
| Pos. embeddings | Learnable | **RoPE**（[[concepts/rotary-position-embeddings]]）|
| Register tokens | 4 | 4（維持）|

> **補足: なぜ RoPE か** — 学習可能な位置埋め込み（learnable positional embedding）は固定解像度に縛られる。RoPE は座標を回転行列として attention に組み込むため、推論時に異なる解像度（4096×4096 まで！）でもそのまま使える。DINOv3 はさらに「RoPE-box jittering」（座標範囲を $s \in [0.5, 2]$ でランダムスケール）で解像度頑健性を強化。詳細: [[concepts/rotary-position-embeddings]]

> **補足: register tokens とは** — Darcet et al.（2023）が提唱した「入力画像に対応しない、純粋にモデル内部の計算用に追加する学習可能トークン」。これがないと ViT は dense feature 中に高ノルムの外れ値（artifact）を作りがちで、それが dense タスク性能を下げる。DINOv2 with registers から導入され、DINOv3 でも 4 つ採用される。

#### 訓練

- **constant スケジュール**: 学習率・重み減衰・teacher EMA モメンタムをすべて定数に。**「下流性能が向上し続ける限り訓練を続けられる」**設計。
- **multi-crop**: 2 global crops（256²）+ 8 local crops（112²）
- **損失**: DINO loss + iBOT loss + 0.1 × Koleo regularizer（[[entities/dinov2]] と同じ）

### B. Gram Anchoring（§4 — 本論文の核心）

詳細: [[concepts/gram-anchoring]]

**問題**: ViT-7B を 200k 反復以上訓練すると、パッチ間コサイン類似度マップが劣化する。

**解決アイデア**: 「特徴量そのもの」ではなく「**パッチ間の Gram 行列**（pairwise dot products の行列）」を、過去の teacher（"Gram teacher"）のそれに近づける。

$$
\mathcal{L}_{\text{Gram}} = \| X_S X_S^\top - X_G X_G^\top \|_F^2
$$

- **$X_S$**: student の L2 正規化された局所特徴量行列（P×d）
- **$X_G$**: Gram teacher（早い段階の teacher、dense feature が良かった頃のスナップショット）の同様の行列

**なぜ効くか**:
1. 個々の特徴量は自由に動ける（global タスクの学習を妨げない）
2. パッチ間の**相対的な類似度構造**だけが過去の teacher に合わせて保たれる
3. 結果として dense タスクの性能が回復し、global タスクも維持

**追加の工夫（§4.3）**:
- Gram teacher には**入力画像の 2 倍解像度**で forward させ、出力を 2× ダウンサンプリング
- これで「より滑らかで一貫性の高い」Gram 行列を取得
- ADE20k で追加 +2 mIoU の利得

> **補足: Gram 行列という発想の起源** — Gram 行列を比較する損失は、もともとスタイル転送（neural style transfer, Gatys et al. 2015）で使われた。「画像の "スタイル" は特徴量チャネル間の相関で表現できる」という発想。DINOv3 はこれを「**パッチ間の相関を時間方向で安定化させる**」という新文脈で再発明した。詳細: [[concepts/gram-anchoring]]

### C. Post-training（§5）

#### 解像度スケーリング (§5.1)

- 主訓練は 256² で行うが、ポスト訓練で **mixed resolutions**（global: {512, 768}, local: {112, 168, 224, 336}）を 10k 反復追加
- ここでも **Gram anchoring が必須**: なしでは dense 性能が劣化
- 結果: **4k 解像度を超える入力でも安定した feature map**（訓練時の上限 768 を遥かに超える汎化）

#### Multi-student distillation (§5.2)

7B teacher を一度推論して、その出力を**複数の student に同時に配信**することで teacher 推論コストを償却する並列パイプライン。

- 通常の single-teacher/single-student 蒸留: teacher 推論コストが各 student 訓練ジョブで重複
- DINOv3 のアプローチ: 全 GPU でグローバル teacher 推論 → all-gather → 各 student グループが個別訓練
- 「**student を 1 つ追加するコストは、その student の訓練コストだけ**」になる
- 各 student の GPU 数を調整して反復時間を揃え、同期待ち時間を最小化

これで ViT-S, B, L, S+, H+ + ConvNeXt T, S, B, L を**同一バッチデータ・同一 teacher 出力で並列に蒸留**。

#### Text alignment (§5.3)

凍結された ViT-L バックボーン上に 2 つの transformer 層を追加し、テキストエンコーダを LiT [234] スタイルで対比学習。**dino.txt** [100] の手法を踏襲。CLS トークンと mean-pooled patch token を**連結**してテキストとマッチさせるのが新規ポイント。

---

## 実験結果と知見

### Dense タスクで圧勝 (§6.1)

**ADE20k セグメンテーション（凍結 + 線形）**:
- DINOv2 ViT-g: 49.5 mIoU
- AM-RADIOv2.5 ViT-g: 53.0 mIoU（SAM 蒸留にも関わらず）
- **DINOv3 ViT-7B: 55.9 mIoU**（+6.4 vs DINOv2、+2.9 vs AM-RADIO）

**NYUv2 depth（凍結 + 線形）**:
- DINOv2 ViT-g: 0.372 RMSE
- **DINOv3 ViT-7B: 0.309 RMSE**（-0.063 改善）

**3D correspondence（NAVI、Probe3D プロトコル）**:
- DINOv2: 60.1% recall（既に強い）
- **DINOv3: 64.4%**（+4.3）

**DAVIS 動画トラッキング (L 解像度)**:
- DINOv2: 76.6 J&F
- **DINOv3: 83.3**（+6.7）

### Global タスクでも肩を並べる (§6.2)

**ImageNet-1k 線形プローブ**:
- SigLIP 2: 89.1, PEcore: 89.3
- **DINOv3 ViT-7B: 88.4**（弱教師ありとほぼ同等、SSL では初）

**ImageNet-A（adversarial）**:
- DINOv2: 81.7, PEcore: 89.0
- **DINOv3: 86.9**（SigLIP 2 の 84.6 を超え、PE に肉薄）

**Instance retrieval (Oxford-Hard)**:
- DINOv2: 58.2 mAP（既に強い）
- **DINOv3: 60.7**（+12.5 vs OpenCLIP, +29.6 vs SigLIP 2）

### Complex pipelines で SOTA (§6.3)

| タスク | DINOv3 結果 | 競合・備考 |
|---|---|---|
| COCO 検出 mAP (Plain-DETR) | **66.1** | 凍結 backbone で初の SOTA、訓練可能パラメータわずか 100M |
| ADE20k seg. (Mask2Former+ViT-Adapter) | **63.0** mIoU | ONE-PEACE と同点で SOTA |
| 相対深度 NYUv2 ARel (DAv2) | **4.3** | DAv2 ViT-g の 4.4 を超え、しかも DINOv3 は凍結 |
| 3D 理解 (VGGT) | 全タスク SOTA | DINOv2 を DINOv3 に差し替えるだけで上回る |

### DINOv3 ファミリ (§7)

- **ViT-S/14**: 21M params, ラップトップ向け
- **ViT-S+/14**: 29M, S と B の中間
- **ViT-B/14**: 86M
- **ViT-L/14**: 300M, 主力モデル（多くの応用で推奨）
- **ViT-H+/14**: 840M, **7B teacher にほぼ匹敵**（蒸留の恩恵を実証）
- **ViT-7B/16**: 6.7B, フラッグシップ
- **ConvNeXt T/S/B/L**: 量子化向け、エッジ展開向け
- **dino.txt (テキストアラインメント版)**: ゼロショット分類・検索

ViT-L は DINOv2-L 比で ADE20k +6.1 mIoU、DAVIS +6.5 J&F の大幅改善。

### 衛星画像でも汎用性を実証 (§8)

詳細: [[entities/sat-493m]]

- 完全に同じ DINOv3 レシピを **Maxar 衛星画像 493M 枚**に適用
- **Geo-Bench 15 タスク中 12 で SOTA**（Prithvi-v2 や DOFA など衛星専用モデルを上回る）
- 樹冠高度推定 Open-Canopy: MAE 2.42 → **2.02** に改善
- 興味深いことに、**LVD-1689M で訓練したウェブ版 DINOv3 も衛星タスクで競合的**（特に semantic segmentation タスク）

---

## なぜこの研究が CV にとって重要か

1. **「CV における BERT」の到達点**: DINOv2 が示唆した方向を完成形に。**凍結 SSL バックボーンが弱教師ありを dense タスクで圧倒**することを示した。これで「視覚タスクは DINOv3 を凍結して上に小型 head を載せれば終わり」が現実的選択肢に。
2. **Gram anchoring という汎用テクニックの発見**: 「特徴量の自由度を残しつつ、構造（pairwise relations）だけを過去の teacher に合わせる」発想は、SSL を超えて多様な long-training/scaling 問題に応用可能。後続の SSL 研究の標準ツールになりうる。
3. **マルチドメイン汎用性の実証**: 同じレシピが自然画像と衛星画像の両方で SOTA を獲得。「**SSL レシピは本当にドメイン非依存**」を強力に支持し、医療画像・天文・粒子物理など他ドメインへの展開を後押し。
4. **モデルファミリの実用性**: ViT-S/S+/B/L/H+/7B + ConvNeXt T/S/B/L + text-aligned 版が一斉公開。エッジから巨大サーバまで全レイヤーをカバー。
5. **物体検出・セグメンテーションでの「凍結バックボーン革命」**: 検出・セグメンテーションは伝統的にバックボーンも fine-tune するのが常識だった。DINOv3 はそれを覆し、凍結のままで SOTA を達成。

---

## 限界・批判的視点

- **巨大な計算コスト**: ViT-7B 単体で 61,440 H100-GPU 時間（47 MWh, 18 t CO₂eq）。プロジェクト全体で **9M GPU-hours, 2600 t CO₂eq**（パリ-NY 便 1 日分の半分）。著者らも §9 で正直に開示。
- **データ・モデルとも非公開部分が多い**:
  - LVD-1689M は非公開（Instagram データのため再配布不可）
  - SAT-493M は Maxar の商用画像で非公開
  - モデル重みは公開
- **地理的・社会的バイアス**: §8.1 と DINOv2 から継承する西洋・高所得地域偏重バイアス。著者らは §8.1 で言及するが本格的な mitigation は将来課題。
- **「Gram anchoring がなぜ正確に効くのか」の理論はまだ薄い**: 経験的に効くことは示されたが、global と dense の分離を理論的に説明する厳密な分析はない。
- **テキスト整合性は CLIP 系に劣る**: dino.txt 版のゼロショット ImageNet で DINOv3 82.3 vs SigLIP 2 83.1 / PE 83.5。global なテキスト理解では弱教師ありが依然有利。
- **ViT-7B のサイズに対する応用ハードル**: 7B 推論は GPU メモリと速度の制約が大きく、現実には ViT-L が「実用 sweet spot」となる。

---

## 用語と略称

| 略称 | 展開 | 短い意味 |
|---|---|---|
| **DINOv3** | DINO version 3 | 本論文のモデルファミリ |
| **DINOv2** | DINO version 2 | 前身。[[entities/dinov2]] |
| **iBOT** | image BERT pre-Training with Online Tokenizer | DINOv3 の損失関数の核。[[entities/ibot]] |
| **MIM** | Masked Image Modeling | パッチをマスクして予測する SSL 系統。[[concepts/masked-image-modeling]] |
| **SSL** | Self-Supervised Learning | 自己教師あり学習。[[concepts/self-supervised-learning]] |
| **WSL** | Weakly-Supervised Learning | 弱教師あり（CLIP 系）。[[concepts/weakly-supervised-pretraining]] |
| **CLIP** | Contrastive Language-Image Pre-training | 弱教師あり代表。[[entities/clip]] |
| **PE / PEcore / PEspatial** | Perception Encoder | Meta の弱教師ありモデル。[[entities/perception-encoder]] |
| **SigLIP / SigLIP 2** | Sigmoid Loss Image-Language Pre-training | Google の CLIP 系。[[entities/siglip]] |
| **AM-RADIO** | Agglomerative Models — RADIO | NVIDIA の凝集蒸留モデル（SAM + CLIP + DINOv2）|
| **Franca** | — | オープンデータ SSL モデル |
| **Web-DINO** | — | DINO の scaling 試行 |
| **AIMv2** | Autoregressive Image Models v2 | Apple の自己回帰 ViT |
| **EVA-CLIP** | — | 巨大 CLIP 系（最大 18B）|
| **SAM / SAM v2** | Segment Anything Model | Meta の汎用セグメンテーションモデル |
| **V-JEPA / V-JEPA 2** | Video Joint-Embedding Predictive Architecture | Meta の動画 SSL |
| **VGGT** | Visual Geometry Grounded Transformer | 3D 理解の汎用モデル |
| **DAv2** | Depth Anything V2 | 単眼深度推定の SOTA |
| **DPT** | Dense Prediction Transformer | 密予測用 Transformer デコーダ |
| **DETR / Plain-DETR** | DEtection TRansformer | Transformer ベース検出モデル |
| **Mask2Former** | — | セグメンテーションデコーダ |
| **ViT-Adapter** | — | ViT 上で密予測を強化するアダプタ |
| **LVD-1689M** | Large-scale Visual Dataset (1689M) | DINOv3 用キュレーションデータセット。[[entities/lvd-1689m]] |
| **LVD-142M** | Large-scale Visual Dataset (142M) | DINOv2 用、LVD-1689M の前身。[[entities/lvd-142m]] |
| **SAT-493M** | Satellite dataset (493M) | DINOv3 衛星版用。[[entities/sat-493m]] |
| **RoPE** | Rotary Position Embeddings | 回転位置埋め込み。[[concepts/rotary-position-embeddings]] |
| **Gram Anchoring** | Gram 行列によるアンカリング | 本論文の核となる正則化。[[concepts/gram-anchoring]] |
| **register tokens** | レジスタトークン | dense feature 改善のための内部用トークン |
| **EMA** | Exponential Moving Average | teacher の更新方式 |
| **SwiGLU** | Swish-Gated Linear Unit | Transformer FFN の活性化 |
| **FSDP** | Fully-Sharded Data Parallel | PyTorch の大規模分散学習 |
| **FlashAttention** | — | 高効率 attention 実装 |
| **TTA** | Test-Time Augmentation | テスト時拡張 |
| **OOD** | Out-Of-Distribution | 分布外、頑健性評価で頻出 |
| **JEPA** | Joint-Embedding Predictive Architecture | LeCun 提唱の latent 予測 SSL |
| **Sinkhorn-Knopp / SK** | — | クラスタ割当均等化アルゴリズム |
| **KoLeo** | Kozachenko-Leonenko | 微分エントロピー正則化 |
| **PUE** | Power Usage Effectiveness | データセンタ電力効率 |
| **CorLoc** | Correct Localization | 物体発見の評価指標 |
| **mIoU** | mean Intersection over Union | セグメンテーション指標 |
| **mAP** | mean Average Precision | 検出・検索の指標 |
| **RMSE** | Root Mean Squared Error | 深度推定で使う誤差 |
| **ARel** | Absolute Relative error | 相対深度推定の指標 |
| **CNX / ConvNeXt** | Convolutional Next | ConvNet 系の現代的アーキテクチャ |
| **GLDv2** | Google Landmarks Dataset v2 | ランドマーク画像データセット |
| **WRI** | World Resources Institute | 著者の一人 John Brandt の所属（衛星セクション関連）|

---

## 関連ページ

- 翻訳: [[translations/dinov3]]
- 系譜: [[entities/dino]] → [[entities/ibot]] → [[entities/dinov2]] → **[[entities/dinov3]]**
- データ: [[entities/lvd-1689m]] / [[entities/sat-493m]] / [[entities/lvd-142m]]（前身）
- 比較対象: [[entities/clip]] / [[entities/perception-encoder]] / [[entities/siglip]]
- 概念:
  - [[concepts/gram-anchoring]] — 本論文の核心
  - [[concepts/rotary-position-embeddings]] — RoPE
  - [[concepts/foundation-model]] — DINOv3 の位置づけ
  - [[concepts/self-supervised-learning]]
  - [[concepts/masked-image-modeling]]
  - [[concepts/weakly-supervised-pretraining]]
  - [[concepts/knowledge-distillation]] — multi-student distillation
  - [[concepts/knn-evaluation-protocol]]
  - [[concepts/vision-transformer]]
