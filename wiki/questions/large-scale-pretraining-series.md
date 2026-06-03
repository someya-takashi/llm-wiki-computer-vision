---
type: question
asked: 2026-06-03
question: "近年の大規模データによる自己教師あり学習には CLIP の系列（CLIP, SigLIP, LLM の Vision encoder など）と DINO の系列（DINOv2, v3 など）があると思っています。これはあっていますか？他にも系列がありますか？それぞれの系列の発展（アーキテクチャの進化、データの大規模化、革新的な工夫）を整理してください。"
sources_used: ["[[concepts/self-supervised-learning]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/contrastive-learning]]", "[[concepts/masked-image-modeling]]", "[[concepts/foundation-model]]", "[[concepts/vision-transformer]]", "[[concepts/rotary-position-embeddings]]", "[[sources/clip]]", "[[sources/siglip]]", "[[sources/siglip-2]]", "[[sources/perception-encoder]]", "[[sources/dino-emerging-properties-in-self-supervised-vit]]", "[[sources/dinov2-learning-robust-visual-features-without-supervision]]", "[[sources/dinov3]]", "[[sources/ibot]]", "[[sources/mae]]", "[[sources/simclr]]", "[[sources/byol]]", "[[sources/vision-transformer]]", "[[sources/roformer]]", "[[sources/foundational-models-vision-survey]]"]
---

# 大規模事前学習の系列 — WSL / SSL / MIM / 古典対比 / ハイブリッドの 5 系統

## 結論を先に

ユーザの認識は **「半分正しい」**：

- ✅ **正しい点**: CLIP 系列と DINO 系列が「現代の 2 大主流」という把握は実用上正しい。2025 年時点で、ほとんどのオープン視覚基盤モデルとほぼすべての MLLM の視覚エンコーダはこの 2 系列のいずれかに属する
- ⚠️ **修正すべき点（用語）**: CLIP 系列は厳密には **WSL（弱教師あり事前学習, Weakly-Supervised Learning, [[concepts/weakly-supervised-pretraining]]）** で、DINO 系列の **SSL（自己教師あり学習, Self-Supervised Learning, [[concepts/self-supervised-learning]]）** とは監督信号の出所が違う
- ⚠️ **欠落している系列**: 実際には **少なくとも 5 系列** が独立して存在し、wiki 上に明確な痕跡がある

### 5 系列の完全リスト

| # | 系列 | カテゴリ | 代表モデル | 監督信号 | バックボーン主流 |
|---|---|---|---|---|---|
| ① | **WSL 対比型** | WSL | CLIP, SigLIP, PE | 画像 - テキスト対 | ViT |
| ② | **自己蒸留 SSL** | SSL | DINO, DINOv2, DINOv3 | 画像のみ（拡張 + 自己蒸留） | ViT |
| ③ | **MIM 系** | SSL | MAE, BEiT, SimMIM | 画像のみ（マスク再構成） | ViT |
| ④ | **古典対比 SSL** | SSL | SimCLR, MoCo, BYOL, SwAV | 画像のみ（拡張対比） | ResNet（pre-ViT） |
| ⑤ | **ハイブリッド** | SSL | iBOT, DINOv2, DINOv3 | ② + ③ の融合 | ViT |

**さらに 2025 年は「系列の融合期」**：SigLIP 2（WSL ベース）が SSL 技法を借用、PE（WSL）が alignment tuning で中間層活用、DINOv3（SSL）が CLIP 由来の RoPE を採用するなど、系列の境界が曖昧化している。

---

## 用語の整理：SSL と WSL の決定的違い

ユーザの認識のズレを修正する最重要ポイント。

### 監督信号の出所の違い

| 観点 | SSL（自己教師あり） | WSL（弱教師あり） |
|---|---|---|
| **教師信号** | 画像のみから作る | 画像 - テキスト対の対応 |
| **増強の役割** | 「同じ画像の 2 つの拡張は類似」という制約を強制 | 主に正則化（拡張なしでも動く） |
| **テキストの役割** | 不要 | 必須（捨てられない） |
| **wiki 概念ページ** | [[concepts/self-supervised-learning]] | [[concepts/weakly-supervised-pretraining]] |
| **代表モデル** | DINO, MAE, BYOL, SimCLR | CLIP, SigLIP, PE, ALIGN |

### なぜユーザは混同するか

- **両者とも「人手ラベリングしたカテゴリラベル」を使わない**（ImageNet のような明示的なラベリングは不要）
- **両者とも Web スケールのデータが必要**
- **下流タスクでの使い方が似ている**（凍結特徴量 + 線形プローブ、ファインチューン）
- 一般的な技術記事でも「CLIP は SSL」と書かれることが多い（厳密ではないが、対比的に CV のラベル付きデータからの独立を強調する文脈で）

[[concepts/weakly-supervised-pretraining]] §「補足: 教師あり/弱教師あり/自己教師あり」より、wiki の公式立場：

> **教師あり**（supervised）: 人手で正確にラベル付けされたデータ（ImageNet など）
> **自己教師あり**（self-supervised, SSL）: 教師信号をデータ自身から作る
> **弱教師あり**（weakly-supervised, WSL）: 教師信号は外部にあるが不完全（Web キャプション、ハッシュタグ等）。**WSL は SSL と教師ありの中間**

### 並列議論する根拠

両者を本ページで「5 系列」として並列に扱う理由：

1. **共通の目的**: 「人手ラベリングなしで Web スケール訓練して凍結特徴量で多様な下流タスクに使う」
2. **共通の評価軸**: 線形プローブ精度・k-NN 精度・密予測精度
3. **2025 年の融合**: SigLIP 2 が SSL 技法を借用、PE が SSL 風の中間層活用、DINOv3 が WSL 由来 RoPE 採用など、技術的に相互輸入が進んでいる

---

## 系列 ①: WSL（画像 - テキスト対比）系 — CLIP 系列

### 系譜

```
CLIP (OpenAI 2021) ──── ALIGN (Google 2021) ──── OpenCLIP (LAION 2022)
                                                       │
                              FILIP / FLIP / EVA-CLIP / MetaCLIP / DFN ─┐
                                                       │                │
SigLIP (Google DeepMind 2023) ─── SigLIP 2 (Google DeepMind 2025)       │
                                                       │                │
                                  PE (Meta NeurIPS 2025) ────────────────┘
```

### アーキテクチャの進化

- **dual-encoder 構造は不変**: ViT 画像エンコーダ + Transformer テキストエンコーダ。CLIP からの基本骨格は変わらない
- **損失関数の進化**:
  - **CLIP（[[sources/clip]]）**: 対称 InfoNCE 損失（バッチ内 softmax 対比、N² 対対比）
  - **SigLIP（[[sources/siglip]]）**: softmax → **sigmoid 損失**。各対を独立に判定するためグローバル正規化不要、小バッチで圧倒的に勝つ
  - **PE（[[sources/perception-encoder]]）**: CLIP 風対比を頑健化（**LAMB optimizer + 大バッチ 131k + 慎重なデータキュレーション**）、**「対比学習を頑健にスケールすると中間層に多目的特徴量が育つ」** という創発を発見
- **マルチ目的化**:
  - **SigLIP 2（[[sources/siglip-2]]）**: 対比 + **LocCa decoder + 自己蒸留（SILC 由来）+ マスク予測（TIPS 由来）+ ACID 蒸留 + 多言語 + NaFlex** の **5 系統融合「全部入りレシピ」**
  - **PE（[[sources/perception-encoder]]）**: 純粋対比を保ったまま、**[[concepts/alignment-tuning|alignment tuning]]** で中間層特徴を末端に引き出し、PEcore（ゼロショット）/ PElang（MLLM 特化）/ PEspatial（dense 予測）の **3 バリアント設計**

### データの大規模化の推移

| 年 | データセット | サイズ | モデル |
|---|---|---|---|
| 2021 | WIT（OpenAI 内部）| 400M 対 | CLIP |
| 2021 | ALIGN | 1.8B 対 | ALIGN |
| 2022 | LAION-400M | 400M 対 | OpenCLIP |
| 2022 | LAION-5B | 5.85B 対 | OpenCLIP |
| 2023 | WebLI（Google 内部）| 数 B 対 | SigLIP, SigLIP 2 |
| 2024 | DataComp-1B | 1.4B 対 | DFN, MetaCLIP |
| 2025 | WebLI 10B | 10B 対 / 109 言語 | SigLIP 2 |
| 2025 | PE 訓練データ | 5.4B unique pairs / 86B samples seen + 22M videos | PE |

**桁の進化**: 4 億（CLIP 2021）→ 約 100 億（SigLIP 2 2025）= **約 25× の拡大**。並行してデータ品質のキュレーション（**MetaCLIP の選定基準公開、DFN の filtering networks、PE の慎重なキュレーション**）が同等以上に重要になった。

### 革新的工夫

- **CLIP（[[sources/clip]]）**: **ゼロショット分類を CV に持ち込んだ起点**。「a photo of a {class}」プロンプトでファインチューン不要、CV にテキスト誘導パラダイムを開拓
- **SigLIP（[[sources/siglip]]）**: **sigmoid 損失で 32k バッチで飽和**、4 TPU で 1 日訓練可能、ノイズ頑健、SO-400M で 83.2% IN-0（5B EVA-CLIP より高精度）
- **SigLIP 2（[[sources/siglip-2]]）**: **SSL 技法を借用した逆流の代表**。SILC（local-to-global 自己蒸留）+ TIPS（マスク予測）を統合、representation bias 35.5% → 7.3% に削減、RefCOCO で +20pt、dense seg で +4-5pt
- **PE（[[sources/perception-encoder]]）**: **「対比学習をピュアに保ったまま、中間層活用で多目的化を実現」** という SigLIP 2 と対照的な路線。**COCO 検出 66.0 box AP で SOTA**（DETR-style デコーダ）、dense タスクで SAM 2.1 蒸留と組み合わせて PEspatial で部分的に DINOv3 超え

### この系列の MLLM への継承（極めて重要）

**ほぼすべての商用級 MLLM の視覚エンコーダはこの系列**：

- **[[entities/qwen3-vl|Qwen3-VL]]**: **SigLIP-2 から継続学習**（[[entities/qwen2-5-vl|Qwen2.5-VL]] のゼロから路線放棄）
- **[[entities/qwen2-vl|Qwen2-VL]]**: DFN（Apple Data Filtering Networks、CLIP 系）で ViT を初期化
- **[[entities/gemma-3|Gemma 3]]**: **SigLIP 400M variant を共有・凍結**して 4B/12B/27B で再利用
- **[[entities/internvl|InternVL]] / [[entities/internvl-1-5|InternVL 1.5]]+**: InternViT-6B / 300M（**CLIP-ViT-L 初期化 + 大規模継続事前学習**）
- **[[entities/deepseek-ocr|DeepSeek-OCR]]**: SAM-base + **CLIP-large** の直列ハイブリッド

→ **ユーザが指摘した「LLM の Vision encoder」はこの系列の応用に他ならない**。独立系列ではなく、**CLIP 系の継承・改造**。

---

## 系列 ②: 自己蒸留 SSL 系 — DINO 系列

### 系譜

```
DINO (Meta 2021)
   │ + MIM
   ▼
iBOT (Zhou et al. 2022)
   │ + Web スケール + キュレーション
   ▼
DINOv2 (Meta 2023, ViT-g/14, LVD-142M)
   │ + 7B スケール + axial RoPE + Gram anchoring
   ▼
DINOv3 (Meta 2025, ViT-7B, LVD-1689M)
```

### アーキテクチャの進化

- **DINO（[[sources/dino-emerging-properties-in-self-supervised-vit]]）**: teacher-student の自己蒸留、**cross entropy + centering + sharpening + EMA teacher** で崩壊を防ぐ。projection head（学習可能 + 重み正規化 + bottleneck dim 256）が重要
- **iBOT（[[sources/ibot]]）**: **DINO + MIM のハイブリッド**。「**online tokenizer**」概念を導入し、teacher 自身を MIM の視覚トークナイザとして使う（dVAE 不要）
- **DINOv2（[[sources/dinov2-learning-robust-visual-features-without-supervision]]）**: ViT-g/14（1.1B パラメータ）。**iBOT を 142M キュレーション画像にスケール**、LayerScale + SwiGLU + stochastic depth + EMA teacher 等の安定化技法を集大成
- **DINOv3（[[sources/dinov3]]）**: ViT-7B にスケール。**axial RoPE（[[concepts/rotary-position-embeddings]] 由来、CLIP の学習可能 1D 位置埋め込みから乗り換え）+ Gram anchoring（[[concepts/gram-anchoring]]、長時間学習による dense feature 劣化を解決）+ register tokens**。**256² 訓練 → 4096² 推論で安定**

### データの大規模化の推移

| 年 | データセット | サイズ | モデル |
|---|---|---|---|
| 2021 | ImageNet-1k | 1.3M | DINO |
| 2022 | ImageNet-22k | 14M | iBOT |
| 2023 | LVD-142M（[[entities/lvd-142m]]、Meta 内部キュレーション）| 142M | DINOv2 |
| 2025 | LVD-1689M（[[entities/lvd-1689m]]、Web スケール拡張）| 1.689B | DINOv3 |

**桁の進化**: 130 万（DINO 2021）→ 17 億（DINOv3 2025）= **約 1300× の拡大**。SSL 系列で最大規模を達成、WSL 系列に追いつくスケール。

### 革新的工夫

- **DINO（[[sources/dino-emerging-properties-in-self-supervised-vit]]）**: **教師なし segmentation 創発の発見**。「自己注意マップが訓練監督なしにオブジェクト境界を捉える」という驚きの結果が CV を変えた
- **iBOT（[[sources/ibot]]）**: **online tokenizer** で MAE/BEiT の dVAE 依存を解消、DINO の image-level + iBOT の patch-level の両目的を統合
- **DINOv2（[[sources/dinov2-learning-robust-visual-features-without-supervision]]）**:
  - **dense feature の凍結線形プローブで MAE フル fine-tune に匹敵**（ADE20K 49.0 mIoU 凍結線形）
  - LVD-142M の **データキュレーション**（自動 deduplication + クラスタリング + ImageNet-22k seeded retrieval）
  - **distillation 蒸留パイプライン**で ViT-S/B/L へ知識転送
- **DINOv3（[[sources/dinov3]]）**:
  - **axial RoPE** で任意解像度
  - **Gram anchoring**（[[concepts/gram-anchoring]]）でパッチ間 Gram 行列を過去 teacher と整合させ、**長時間訓練による dense feature 劣化問題を解決**
  - **凍結バックボーン + Plain-DETR で COCO 検出 66.1 mAP SOTA**（教師あり SOTA 並み）

---

## 系列 ③: MIM 系 — MAE 系列

### 系譜

```
BEiT (Microsoft 2021)
   │ 画素直接予測へ単純化
   ▼
MAE (Meta 2021, He et al.) ───── SimMIM (Microsoft 2022, MAE 単純化版)
   │
   │ DINO と合流
   ▼
iBOT (DINO 系と統合) ──→ DINOv2 / DINOv3 へ吸収
```

### アーキテクチャの進化

- **BEiT（Bao et al. 2021）**: 事前学習済み dVAE（離散 VAE）が出力する離散トークンを予測（NLP の BERT に最も忠実）。**別途 tokenizer 学習が必要**
- **MAE（[[sources/mae]]、He et al. 2021）**: **3 つの核心設計**:
  1. **画素 75% マスク**（BERT の 15% より圧倒的に高い高マスク率）
  2. **非対称 encoder-decoder**: encoder は可視 25% のみ処理（mask token を入力しない）、軽量 decoder が encoder 出力 + mask token から画素再構成
  3. **画素を直接再構成**（パッチ内正規化）。dVAE 不要
- **SimMIM（Xie et al. 2022）**: MAE の単純化版（encoder にも mask token を入れる）

### データの大規模化の推移

| 年 | データセット | サイズ | モデル |
|---|---|---|---|
| 2021 | ImageNet-22k | 14M | BEiT |
| 2021 | ImageNet-1k | 1.3M | MAE |
| 2022 | ImageNet-22k + JFT-300M（実験的）| 14M-300M | 各種拡張 |

**重要観察**: MAE は **ImageNet-1k のみで ViT-H/14 を 87.8%** という驚異的な効率性を示した。**Web スケールへの拡張は DINOv2/v3 が iBOT 経由で MIM を統合する形で実現**。

### 革新的工夫

- **MAE（[[sources/mae]]）**:
  - **非対称設計で計算量 4× 削減**（encoder が可視 25% のみ処理）。これで ViT-H/L が現実的に訓練可能に
  - **ImageNet 1.3M のみで ViT-H 87.8%**（教師あり SOTA 並み、JFT 不要）
  - **下流転移性が強力**: COCO 検出 / ADE20K セグメンテーションでフル fine-tune が強い
- **後の発展（純粋 MIM の単独路線は衰退、ハイブリッドへ統合）**:
  - **iBOT** がオンライン・トークナイザで DINO と統合
  - **DINOv2/v3** が MIM 損失を共有エンコーダで併用
  - **V-JEPA**（Meta 2024）: 特徴量空間で MIM
  - **AIMv2**（Apple 2024）: autoregressive image modeling として継承

**MIM 系の現在地**: 純粋 MIM 単独でスケールする研究は減少。**MIM は単独より融合された形（iBOT, DINOv2, DINOv3）で生き続けている**のが 2025 年の状況。

---

## 系列 ④: 古典対比 SSL 系 — SimCLR / MoCo / BYOL / SwAV

### 系譜（pre-ViT 時代、ResNet ベース）

```
SimCLR (Chen et al. 2020) ──── MoCo (He et al. 2019) ──── MoCov2/v3
        │                              │
        └──────── BYOL (Grill 2020) ───┘
                          │
                          ▼
                    SwAV (Caron et al. 2020)
                          │ multi-crop 発明
                          ▼
                    DINO（系列②へ）
```

### アーキテクチャの進化

- **バックボーン**: 主に **ResNet-50/200**（pre-ViT 時代）。後に MoCov3 が ViT 移行を試みる
- 共通フレームワーク: 「画像の 2 つの拡張 view が encoder + projection head を通って類似性を計算」

### 各手法の革新

- **SimCLR（[[sources/simclr]]、Chen et al. 2020）**: **4 コンポーネント体系化**:
  1. 強い拡張（random resize crop + color jittering + Gaussian blur）
  2. encoder（ResNet）
  3. **非線形 projection head**（2 層 MLP、下流で捨てる重要原則）
  4. **NT-Xent 損失**（normalized temperature-scaled cross entropy = InfoNCE）
  - **ResNet-50 (4×) で 76.5% top-1**（教師あり ResNet-50 同等）、1% ラベルで top-5 85.8%
  - **対比 SSL の 2020 年標準ベースライン**
- **MoCo / MoCov2 / MoCov3（He et al. 2019/2020/2021）**: **momentum encoder + memory bank**:
  - SimCLR が大バッチ（4096）を要求するのに対し、**memory bank で小バッチでも大量負例を確保**
  - momentum encoder は後の **BYOL / DINO の EMA teacher** に直結
  - MoCov3 で初めて ViT 対応
- **BYOL（[[sources/byol]]、Grill et al. NeurIPS 2020）**: **「対比に負例は必須か」に「否」と答えた先駆け**:
  - online × target 2 ネットワーク + **predictor**（追加 MLP、崩壊防止の鍵）+ EMA で負例なし学習
  - SimCLR を 5pt 上回り（**74.3% vs 69.3%**）
  - **DINO の自己蒸留パラダイムの直接の前身**
- **SwAV（Caron et al. 2020）**: **クラスタリング + multi-crop の発明**:
  - **Sinkhorn-Knopp 正規化**でクラスタ均等化（崩壊防止）
  - **multi-crop**: 1 画像から 2 大 crop + 数本の小 crop を作り「**局所 → 大域**」対応を強制
  - **multi-crop は DINO に継承され決定的に重要**になる

### この系列の現在地

- **2025 年時点で独立路線としては終焉**: 古典対比 SSL の純粋系列は活発な研究対象ではない
- **理論的・実装的基礎としては不可欠**: 現代 SSL のすべて（DINO/MAE/iBOT/DINOv2/v3）が何らかの形で継承
  - DINO ← BYOL の EMA teacher + SwAV の multi-crop
  - MoCov3 → ViT への対比拡張
  - SimCLR の InfoNCE 損失は CLIP の基礎

---

## 系列 ⑤: ハイブリッド系（自己蒸留 + MIM）

### 概要

DINO 系の「画像レベル（image-level）目的」と MAE 系の「パッチレベル（patch-level）MIM 目的」を **同じエンコーダで同時最適化** する系列。

### 代表モデル

- **iBOT（[[sources/ibot]]、Zhou et al. 2022）**: ハイブリッド系の起点
- **DINOv2（[[sources/dinov2-learning-robust-visual-features-without-supervision]]、Meta 2023）**: ハイブリッド系の完成形
- **DINOv3（[[sources/dinov3]]、Meta 2025）**: ハイブリッド系を 7B スケールまで拡張

### 革新的工夫

- **iBOT online tokenizer**: 別途学習した tokenizer（BEiT の dVAE）が不要に。**teacher 自身を MIM の予測対象として使う**（teacher 出力 = 「動的な視覚 token」）
- **両目的の補完性**:
  - 識別型 SSL の **凍結特徴量の良さ**（線形分離性、画像レベル理解）
  - MIM の **密予測性能**（パッチレベル、セグメンテーション / 深度推定）
  - **両立**することで凍結バックボーンで多様な下流タスクに使える
- **DINOv2 がこの路線を完成させ、CV における純粋 SSL 基盤モデル化を達成**
- **DINOv3 で 7B スケールまで拡張**、PE / SigLIP 2 に対し dense タスクで広く優位

### この系列の位置付け

**2025 年時点の純粋 SSL 路線の主流**。系列 ②（自己蒸留）と系列 ③（MIM）の融合として理解すべき。

---

## 系列の融合：2024-2025 の逆流現象

**「対比 vs 自己蒸留 vs MIM」の区分は実用モデルでは融合している**。具体例:

### SSL 技法を借用する WSL 系統

- **[[sources/siglip-2|SigLIP 2]]** (2025): WSL でありながら **SILC（local-to-global 自己蒸留、DINO 系）+ TIPS（マスク予測、MIM 系）** を借用。**SigLIP 2 は対比 + 自己蒸留 + MIM + decoder + 蒸留の 5 系統融合**
- **[[sources/perception-encoder|PE]]** (2025): WSL でありながら **[[concepts/alignment-tuning|alignment tuning]]** で中間層特徴を活用、**PEspatial は SAM 2.1 蒸留 + 自己層 41 蒸留** で dense SOTA を一部奪還

### WSL 技法を借用する SSL 系統

- **[[sources/dinov3|DINOv3]]** (2025): SSL でありながら **axial RoPE**（[[sources/roformer]] 由来、NLP 発祥で CLIP/SigLIP 系で標準）を採用。**学習可能 1D 位置埋め込み（オリジナル ViT）から脱却**

### 系列横断技術の代表

- **[[concepts/rotary-position-embeddings|RoPE]]**: NLP（RoFormer 2021）→ Qwen-VL 系（M-RoPE, 2D-RoPE）→ DINOv3（axial RoPE）。**系列横断の最重要技法**
- **knowledge distillation**: SigLIP 2 の ACID, DINOv2/v3 の蒸留パイプライン, PE の SAM 2.1 蒸留, Mini-InternVL の InternViT 蒸留など、**ほぼ全系列で活用**

### 系列融合の意味

**「監督信号の出所」（SSL vs WSL）は依然として根本的な区別だが、「アーキテクチャ・損失関数・訓練レシピ」のレベルでは系列間で自由に技法が輸入される時代**。次世代の基盤モデルは「最良の対比 + 最良の自己蒸留 + 最良の MIM」を 1 モデルに統合する方向（SigLIP 2 が先駆け）。

---

## MLLM の Vision Encoder は新しい系列か？

**結論: 独立系列ではない。ほぼすべて CLIP/SigLIP 系の継承・改造**。

| MLLM | 視覚エンコーダの出自 | 系列 |
|---|---|---|
| [[entities/qwen-vl\|Qwen-VL]] (2023) | OpenCLIP ViT-bigG（1.9B）| ① WSL（OpenCLIP） |
| [[entities/qwen2-vl\|Qwen2-VL]] (2024) | DFN 初期化の ViT（675M、ゼロから学習）| ① WSL（DFN = CLIP 系） |
| [[entities/qwen2-5-vl\|Qwen2.5-VL]] (2025) | Window Attention ViT（ゼロから学習）| ① WSL 系の改造 |
| [[entities/qwen3-vl\|Qwen3-VL]] (2025) | **SigLIP-2 から継続学習** | ① WSL（SigLIP 2 系） |
| [[entities/internvl\|InternVL]] / [[entities/internvl-1-5\|InternVL 1.5]]+ | InternViT-6B / 300M（**CLIP-ViT-L 初期化** + 継続事前学習） | ① WSL（CLIP 系） |
| [[entities/gemma-3\|Gemma 3]] (2025) | **SigLIP 400M variant を共有・凍結** | ① WSL（SigLIP 系） |
| [[entities/deepseek-ocr\|DeepSeek-OCR]] (2025) | SAM-base + **CLIP-large** の直列 | ① WSL（CLIP 系）+ 領域汎用 |

### 例外的に純粋 SSL を使う MLLM

- **稀**: DINOv2/v3 を視覚エンコーダにする MLLM 研究は限定的（学術論文では存在、商用 MLLM の主流ではない）
- **理由**:
  1. CLIP 系のテキスト整列済み特徴量が **言語デコーダとの接続が自然**
  2. DINO 系の純粋画像特徴量は言語デコーダで使う前に追加の整列が必要
  3. CLIP 系は **ゼロショット転移能力が built-in**

### ユーザの認識への直接的回答

ユーザが「LLM の Vision encoder」を CLIP 系の枝として挙げた点は **概ね正しい**。**ただし「Vision encoder としてゼロから学習されたもの」（Qwen2-VL / Qwen2.5-VL）も、初期化やレシピは CLIP 系から派生** している点に注意。

---

## 5 系列の比較表

| 観点 | ① WSL 対比 | ② 自己蒸留 SSL | ③ MIM | ④ 古典対比 SSL | ⑤ ハイブリッド |
|---|---|---|---|---|---|
| **時代** | 2021〜現在 | 2021〜現在 | 2021-2023 | 2019-2021 | 2022〜現在 |
| **代表** | CLIP / SigLIP / PE | DINO / DINOv2 / v3 | MAE / BEiT | SimCLR / MoCo / BYOL | iBOT / DINOv2 / v3 |
| **監督** | 画像 - テキスト対 | 画像（自己蒸留）| 画像（マスク）| 画像（拡張対比）| 画像（自己蒸留 + マスク） |
| **バックボーン** | ViT | ViT | ViT | ResNet（pre-ViT）| ViT |
| **データ規模** | 4 億 → 100 億 | 1.3M → 17 億 | 1.3M → 14M | 1.3M | 142M → 17 億 |
| **強み** | ゼロショット転移、テキスト連携 | 凍結特徴量の良さ、教師なし segmentation 創発 | 密予測、計算効率 | 理論的明快さ | 識別 + 密予測の両立 |
| **弱み** | テキスト依存、学習可能位置埋め込み | テキスト連携が後付け | スケール限界（単独）| ViT 時代に陳腐化 | 複雑な訓練レシピ |
| **MLLM への活用** | **主流** | 限定的 | ほぼなし | なし | 限定的 |
| **2025 年の活発度** | 高（SigLIP 2, PE） | 高（DINOv3） | 低（融合系へ吸収）| 低（理論的基礎）| 高（DINOv3） |

---

## 訓練データの大規模化の俯瞰（時系列）

```
2019 ─ SimCLR / MoCo (ImageNet 1.3M)
2020 ─ BYOL / SwAV (ImageNet 1.3M)
2021 ─ CLIP (WIT 400M) ──── DINO / MAE / BEiT (ImageNet-22k 14M)
2022 ─ OpenCLIP (LAION-400M) ── iBOT (ImageNet-22k)
2023 ─ SigLIP (WebLI 数 B) ── DINOv2 (LVD-142M)
2024 ─ LAION-5B / DFN-1B / MetaCLIP-1B
2025 ─ SigLIP 2 (WebLI 10B / 109 言語) ── DINOv3 (LVD-1689M) ── PE (5.4B unique pairs + 22M videos)
```

### スケール則の系列差

- **WSL 系**: Web スケール（10 億+）が当然。**CLIP 以来 25× 拡大**
- **SSL 系**: DINOv2 (142M) → DINOv3 (1.689B) で **12× 拡大**。Web スケールに追いついた
- **MIM 系**: 単独路線は ImageNet スケール（〜14M）で停止、ハイブリッド系（DINOv2/v3）で間接的に Web スケール化
- **古典対比 SSL**: ImageNet スケール（1.3M）で停止。pre-ViT 時代の限界

---

## まとめ：ユーザの認識への直接的な回答

1. **「CLIP 系 + DINO 系」の 2 大主流の把握は実用上正しい** — 2025 年の活発な研究領域はこの 2 系列に集中
2. **ただし用語的には WSL vs SSL の区別あり** — CLIP は WSL、DINO は SSL（[[concepts/weakly-supervised-pretraining]] vs [[concepts/self-supervised-learning]]）
3. **他に存在する系列**:
   - **MIM 純粋 SSL 系**（MAE/BEiT、純粋路線は終焉、ハイブリッドへ吸収）
   - **古典対比 SSL 系**（SimCLR/MoCo/BYOL/SwAV、pre-ViT 時代、現代の理論的基礎）
   - **ハイブリッド系**（iBOT/DINOv2/v3、自己蒸留 + MIM 融合、純粋 SSL の現代主流）
4. **MLLM の Vision encoder はほぼ全部 CLIP/SigLIP 系の継承で独立系列ではない** — Qwen-VL 系 / Gemma 3 / InternVL 系 / DeepSeek-OCR すべて
5. **2024-2025 は「系列の融合期」** — SigLIP 2 / PE が SSL 技法を借用、DINOv3 が CLIP 由来 RoPE 採用

### 「次に何を読むべきか」の指針（系列補完の観点）

本 wiki の系列別カバレッジから見て、優先的に補完すべき原典:

- **系列 ① WSL**: ALIGN（Google 2021）、MetaCLIP（Meta 2024）、DFN（Apple 2023）、AIMv2（Apple 2024、autoregressive image modeling）
- **系列 ③ MIM**: BEiT（Microsoft 2021）、SimMIM（Microsoft 2022）、V-JEPA（Meta 2024）
- **系列 ④ 古典対比 SSL**: MoCo v1/v2/v3（He et al. 2019-2021）、SwAV（Caron et al. 2020）
- **系列 ⑤ ハイブリッド**: なし（DINOv2/v3 で十分カバー済み）

---

## 関連ページ

### 概念

- [[concepts/self-supervised-learning]] — SSL（系列 ②③④⑤）総論
- [[concepts/weakly-supervised-pretraining]] — WSL（系列 ①）総論
- [[concepts/contrastive-learning]] — 対比学習の基礎理論（系列 ①④ の損失関数の祖）
- [[concepts/masked-image-modeling]] — MIM（系列 ③）総論
- [[concepts/foundation-model]] — 基盤モデル全体の俯瞰
- [[concepts/vision-transformer]] — 系列 ①②③⑤ で共通の視覚バックボーン
- [[concepts/rotary-position-embeddings]] — 系列横断の位置埋め込み技法

### 系列 ①: WSL 対比系

- [[sources/clip]] / [[entities/clip]] — 起点
- [[sources/siglip]] / [[entities/siglip]] — sigmoid 損失で改良
- [[sources/siglip-2]] — 5 系統融合の集大成
- [[sources/perception-encoder]] / [[entities/perception-encoder]] — 中間層特徴量の活用

### 系列 ②: 自己蒸留 SSL 系

- [[sources/dino-emerging-properties-in-self-supervised-vit]] / [[entities/dino]] — 起点
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] / [[entities/dinov2]] — 基盤モデル化
- [[sources/dinov3]] / [[entities/dinov3]] — 7B スケール

### 系列 ③: MIM 系

- [[sources/mae]] / [[entities/mae]] — 起点
- [[sources/ibot]] / [[entities/ibot]] — DINO との合流点

### 系列 ④: 古典対比 SSL 系

- [[sources/simclr]] — 体系化の基盤
- [[sources/byol]] — 「負例なし」の先駆け

### 系列 ⑤: ハイブリッド系

- [[sources/ibot]] — 起点
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] — 完成形
- [[sources/dinov3]] — 7B スケール

### 系列融合論証

- [[sources/siglip-2]] — WSL での SSL 借用
- [[sources/perception-encoder]] — WSL での中間層活用
- [[sources/dinov3]] — SSL での RoPE 採用
- [[sources/roformer]] — RoPE の原典（NLP 発祥）
- [[sources/foundational-models-vision-survey]] — 4 軸（テキスト/視覚/異種/身体性）分類体系

### MLLM の Vision encoder（系列 ① の応用）

- [[entities/qwen-vl]] / [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]]
- [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]] / [[entities/internvl-3-5]]
- [[entities/gemma-3]] / [[entities/deepseek-ocr]]

### 関連 question

- [[questions/vit-dynamic-resolution-evolution]] — ViT の位置埋め込み制約から任意解像度への進化。本ページの系列 ①② の技術的詳細を補完
