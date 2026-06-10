---
type: question
asked: 2026-06-10
question: "DINOv3 や SigLIP 2 に至るまで、学習やデータの工夫は大幅に改善されてきたが、アーキテクチャの本質的な部分は初期の Vision Transformer モデルからあまり変化ないように思う。これは正しいか？アーキテクチャにおける変化を教えてほしい。"
sources_used: ["[[concepts/vision-transformer]]", "[[sources/vision-transformer]]", "[[concepts/rotary-position-embeddings]]", "[[sources/roformer]]", "[[sources/yarn]]", "[[concepts/alignment-tuning]]", "[[entities/dino]]", "[[sources/dino-emerging-properties-in-self-supervised-vit]]", "[[entities/dinov2]]", "[[sources/dinov2-learning-robust-visual-features-without-supervision]]", "[[entities/dinov3]]", "[[sources/dinov3]]", "[[entities/siglip]]", "[[sources/siglip]]", "[[sources/siglip-2]]", "[[entities/perception-encoder]]", "[[sources/perception-encoder]]", "[[entities/sam-2]]", "[[entities/hiera]]", "[[entities/qwen-vl]]", "[[entities/qwen2-vl]]", "[[entities/qwen2-5-vl]]", "[[entities/qwen3-vl]]", "[[entities/internvl-1-5]]", "[[entities/internvl-2-5]]", "[[entities/internvl-3-5]]", "[[entities/gemma-3]]", "[[entities/deepseek-ocr]]", "[[questions/vit-dynamic-resolution-evolution]]", "[[questions/rope-extrapolation-mechanism]]", "[[questions/vlm-vision-encoder-continued-pretraining]]", "[[questions/large-scale-pretraining-series]]"]
---

# ViT (2020) から DINOv3 / SigLIP 2 までのアーキテクチャ進化 — 8 軸の変化

## 結論を先に

**ユーザの認識は半分正しい**:

| 部分 | ViT (2020) からの変化 |
|---|---|
| **核心骨格** | ✅ **不変**（Patchify + Transformer Encoder × L + MSA + MLP + 残差） |
| **位置エンコーディング** | ⚠️ **根本的変化**（学習可能 1D → RoPE 系） |
| **Norm 細部** | ⚠️ 変化（LayerNorm → RMSNorm + QK-norm 追加） |
| **MLP 活性化関数** | ⚠️ 変化（GELU → SwiGLU） |
| **Register tokens** | 🆕 追加（[CLS] のみ → register tokens 追加） |
| **Attention 効率化** | ⚠️ 変化（完全 SA → Window/階層型 attention） |
| **出力読み出し** | ⚠️ 変化（最終 [CLS] のみ → 中間層活用） |
| **解像度処理** | ⚠️ 変化（固定 → 任意解像度・可変アスペクト比） |
| **トークン圧縮機構** | 🆕 追加（生 patch → Merger / Pixel Shuffle / ViR） |

つまり **「基本設計思想は不変だが、構成要素のほぼすべてが入れ替わっている」** が正確な答え。

メタ観察として、**「LLM 系の改良が CV に逆輸入された」** が変化の通底テーマ。

---

## 不変な部分 — ユーザの指摘が当たっている部分

[[concepts/vision-transformer|ViT]] ([[sources/vision-transformer]], 2020) の核心構造はそのまま：

```
入力画像 H×W×3
  ↓ ① Patchify: P×P パッチに分割（典型的に 14 or 16）
  ↓ ② 線形射影で D 次元埋め込み（実装は stride P の Conv 1 層）
  ↓ ③ 位置情報を加算（後述、ここが変化する）
  ↓ ④ Transformer Encoder × L 層
  │     各層 = LN → MSA → 残差
  │            LN → MLP → 残差
  ↓ ⑤ プーリング（[CLS] or GAP、後述、ここも変化）
  ↓ ⑥ 出力（分類なら線形ヘッド）
```

この **「画像をパッチ系列に変換 → Transformer で処理」** という発想は、[[entities/dinov3|DINOv3]] でも [[sources/siglip-2|SigLIP 2]] でも [[entities/qwen3-vl|Qwen3-VL]] でも本質的に変わっていません。

**ViT が確立した「画像 = トークン系列」というメンタルモデル自体が CV のパラダイムシフト** であり、これ自体が継続する遺産。具体的に不変な要素：

- **Patchify** + 線形射影
- **Transformer Encoder × L** の積み重ね（L = 12, 24, 32, ...）
- **Pre-norm** 構造（LN → MSA → 残差、LN → MLP → 残差）
- **Multi-Head Self-Attention**
- **MLP**（2 層 + 活性化関数）
- **残差接続**

ユーザが言う「**学習・データの工夫が大幅に改善**」というのも全くその通りで：

- [[entities/dino|DINO]] (2021): 自己蒸留 + multi-crop（学習の革新）
- [[entities/dinov2|DINOv2]] (2023): LVD-142M キュレーション + iBOT 統合（データと学習）
- [[entities/dinov3|DINOv3]] (2025): LVD-1689M + Gram anchoring（データと正則化）
- [[entities/siglip|SigLIP]] (2023): sigmoid 損失（学習の革新、cf. [[questions/sigmoid-vs-softmax-loss]]）
- [[sources/siglip-2|SigLIP 2]] (2025): 5 系統融合（学習の集大成）

**これらはアーキテクチャではなく学習・データ・損失関数の改革**。ユーザの観察は正鵠を射ています。

---

## 変化 ①: 位置エンコーディング — 最大かつ最重要の変化

### ViT の元設計
- **学習可能 1D 絶対位置埋め込み** $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$
- 訓練時の解像度に**強く依存**（196 パッチ向けの 196 ベクトルを訓練）
- 別解像度への転移は **2D 補間** が必要

### 現代の流れ
- **[[entities/dinov3|DINOv3]]**: **axial RoPE** + **RoPE-box jittering**（座標範囲をランダムスケール）
- **[[entities/sam-2|SAM 2]]**: memory cross-attention で **2D-RoPE**
- **[[entities/qwen2-vl|Qwen2-VL]]**: **2D-RoPE 化**（DFN-CLIP の絶対位置埋め込みを置換）
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**: **MRoPE Aligned to Absolute Time**
- **[[entities/qwen3-vl|Qwen3-VL]]**: **Interleaved MRoPE**（t/h/w 次元の均等配置）
- **[[sources/siglip-2|SigLIP 2]]**: **NaFlex** で可変解像度対応

### 変化の本質

**「離散辞書」から「連続関数」へ**:

- 元 ViT: 位置 1, 2, ..., 196 用の **辞書** → 任意解像度に弱い
- RoPE 系: 連続値 $m$ から角度 $m\theta$ を計算する **関数** → 任意解像度に形式的に対応（cf. [[questions/rope-extrapolation-mechanism]]）

これは **[[sources/roformer|RoFormer]]** (NLP, 2021) → **[[sources/yarn|YaRN]]** (2023) → DINOv3 / Qwen 系 (2024-2025) という **NLP からの輸入** が CV を変えた事例。詳細は [[concepts/rotary-position-embeddings]] / [[questions/vit-dynamic-resolution-evolution]] / [[questions/rope-extrapolation-mechanism]] 参照。

---

## 変化 ②: Norm の細部 — 安定化のための小さな変更の積み重ね

### ViT の元設計
- **LayerNorm** (pre-norm)
- 「LN → MSA → 残差、LN → MLP → 残差」が標準

### 現代の流れ

#### RMSNorm への移行

- 多くの MLLM ViT: **RMSNorm**（LLM と統一、計算コスト削減）
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**: ViT の中で **RMSNorm + SwiGLU** で LLM 系統一

**RMSNorm vs LayerNorm**:
- LayerNorm: 平均と分散の両方を計算（中央化 + スケーリング）
- RMSNorm: **二乗平均だけ計算**（スケーリングのみ）→ 計算約 30% 削減、性能は同等

#### QK-norm（追加）

- **[[entities/gemma-3|Gemma 3]]**: **attention 計算前に query/key を正規化**
- Gemma 2 の soft-capping を置換
- Chameleon / Olmo 2 で先行された技法

```
従来 attention: softmax(Q · K^T / sqrt(d))
QK-norm:        softmax(LN(Q) · LN(K)^T / sqrt(d))
```

attention スコアの暴走を防ぎ、訓練安定性を改善。

#### LayerScale（[[entities/dinov2|DINOv2]] で導入）

- 残差接続のスケール係数を **学習可能化**:
$$x_{l+1} = x_l + \text{LayerScale}_l \cdot \text{Block}_l(\text{LN}(x_l))$$
- 深い ViT（24 層以上）の訓練安定化に必須

これらは **どれも小さな修正だが、深い ViT を安定的に大規模化するために必須** になった改良。

---

## 変化 ③: MLP の活性化関数 — GELU → SwiGLU

### ViT の元設計
- **GELU**（Gaussian Error Linear Unit）
- MLP は 2 層 + GELU の単純構造:
$$\text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

### 現代の流れ

#### SwiGLU への移行

- **[[entities/dinov2|DINOv2]]**: **SwiGLU 採用**（Swish-gated Linear Unit）
- LLaMA / Qwen / Gemma など LLM 系で標準採用
- 多くの MLLM ViT も SwiGLU に移行

**SwiGLU の定義**:
$$\text{SwiGLU}(x) = \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

**SwiGLU の特徴**:
- **ゲート機構** を持つ MLP（GLU = Gated Linear Unit の Swish 版）
- パラメータ数を **1.5× 増やす** が性能向上
- 訓練安定性も改善

これは **「LLM 系の改良が CV に逆輸入」** された代表例。NLP では PaLM (2022) 以降標準。

---

## 変化 ④: Register Tokens の登場 — [CLS] 単独からの脱却

### ViT の元設計
- **[CLS] トークン**: BERT 由来の単一読み出しトークン
- 系列全体の情報を集約する「読み出し口」

### 現代の流れ

#### [CLS] + register tokens の組み合わせ

**"Vision Transformers Need Registers"** (Darcet et al., ICLR 2024) — wiki 未取り込みだが影響大

- [[entities/dinov2|DINOv2]] / [[entities/dinov3|DINOv3]]: **register tokens 採用**
- アイデア: ViT の attention map に「ノイズ」として現れる無関係パッチの代わりに、**専用の「待避所」トークン** を用意

```
ViT 元設計:        [CLS] | patch_1 | patch_2 | ... | patch_N
DINOv3 など:      [CLS] | reg_1 | reg_2 | reg_3 | reg_4 | patch_1 | patch_2 | ... | patch_N
                  └─読み出し─┘ └──「待避所」──┘  └────実際の画像情報────┘
```

**効果**:
- attention map が **クリーン** になり、segmentation 等の dense タスクで質が大幅向上
- 訓練安定性も改善
- **DINOv2/v3 の dense feature の質の良さ** に大きく貢献

これは ViT 元設計には存在しなかった **新コンポーネント** で、純粋に追加された変化。

---

## 変化 ⑤: Attention 機構の効率化 — 完全 SA から離れる

### ViT の元設計
- すべての層で **完全 Multi-Head Self-Attention**
- 計算量 $\mathbb{O}(N^2)$ — 高解像度で爆発（896² で 4096 トークン → $\sim$ 1670 万ペア）

### 現代の流れ

#### Window Attention の採用

- **Swin Transformer** (2021): 階層型 + シフトウィンドウ（wiki 未取り込み）
- **[[entities/hiera|Hiera]]** ([[entities/sam-2|SAM 2]] のバックボーン): Swin の単純化版、マルチスケール
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**: **32 層中 4 層のみ完全 SA、残り 28 層は 112² Window Attention** — 計算量 2 次 → 線形化

#### FlashAttention 等の実装最適化

- [[entities/dinov3|DINOv3]]: **FlashAttention** 採用（理論計算量は同じだが実装が高速・省メモリ）

#### 階層型 ViT

- [[entities/hiera|Hiera]]: マルチスケール特徴抽出
- ViT の「平坦」設計から **CNN 風階層性への部分回帰**

#### 純粋 ViT の保持

注: **純粋 ViT（[[entities/dinov3|DINOv3]] 等）は依然完全 SA を保持**。**「効率化が必要なタスクで階層型/Window 化を選ぶ」** という分岐が出てきた。

```
              ┌── 純粋 ViT 路線（DINOv3, SigLIP 2, CLIP 系統）
              │   完全 SA + RoPE で頑張る
ViT (2020)  ──┤
              └── 階層型路線（Hiera, Swin, Qwen2.5-VL ViT）
                  Window/階層化で計算効率優先
```

---

## 変化 ⑥: 出力読み出し — [CLS] のみから中間層活用へ

### ViT の元設計
- 最終層の **[CLS] トークン** を読み出して分類ヘッドへ
- 「最終層こそ最良の表現」という暗黙の前提

### 現代の流れ

#### 中間層活用の発見

- **[[entities/perception-encoder|Perception Encoder (PE)]]** ([[sources/perception-encoder]]): **「対比学習をスケールすると中間層に多目的特徴量が育つ」** という発見
- **[[concepts/alignment-tuning|alignment tuning]]** で中間層を末端に引き出す
- [[entities/internvl-2-5|InternVL 2.5]]: **独立に** **「最終層の線形分離性が低下、attention pooling 維持」** 発見

#### 多層融合

- **[[entities/qwen3-vl|Qwen3-VL]] の DeepStack**: **ViT 中間 3 層を LLM 最初 3 層に注入**
- [[entities/internvl-1-5|InternVL 1.5]]: **「48 → 45 層 shrink」**（最終 3 層を捨てる）

観察: **「最終層は実は MLLM タスクに最適でない」** が一般的な認識に。

#### Attention Pooling

- 多くの MLLM: [CLS] でなく **attention pooling** で集約
- InternViT-V2.5: 最終層線形分離性低下に対応して attention pooling 維持

これは **「[CLS] 単独で十分」という ViT の前提** が破られた事例。詳細は [[concepts/alignment-tuning]] / [[questions/vlm-vision-encoder-continued-pretraining]] 参照。

---

## 変化 ⑦: 解像度処理 — 構造的変化として

[[questions/vit-dynamic-resolution-evolution]] で詳述している通り、**ViT の元設計は固定解像度を強く前提** としていました。これを変えるには **構造的修正** が必要：

| 手法 | 採用モデル | 変更点 |
|---|---|---|
| **Naive Dynamic Resolution** | [[entities/qwen2-vl]] | 2D-RoPE + MLP で隣接 2×2 トークン圧縮 |
| **NaFlex** | [[sources/siglip-2]] | 可変解像度・可変アスペクト比対応 |
| **Pan & Scan** | [[entities/gemma-3]] | LLaVA 風タイル分割（推論時のみ）|
| **動的タイル分割** | [[entities/internvl-1-5]] | 1-40 タイル / 最大 4K |
| **6 解像度モード** | [[entities/deepseek-ocr]] | Tiny/Small/Base/Large/Gundam/Gundam-M |
| **RoPE-box jittering** | [[entities/dinov3]] | 訓練時に座標範囲をランダムスケール |

これらは **すべて元 ViT 設計の「位置埋め込みが固定解像度」前提を回避する** ための変更です。

詳細は [[questions/vit-dynamic-resolution-evolution]] で 4 フェーズの年表として整理：

```
Phase 1: 固定 224²/384² (ViT, 元 SigLIP, CLIP)
   ↓
Phase 2: タイル分割 (LLaVA → InternVL 1.5 → Gemma 3)
   ↓
Phase 3: ViT 構造改造 (Qwen2-VL Naive Dynamic Resolution)
   ↓
Phase 4: 可変解像度 + RoPE 系 (SigLIP 2 NaFlex, DINOv3 axial RoPE)
```

---

## 変化 ⑧: トークン圧縮機構 — MLLM 用の新コンポーネント

### ViT の元設計
- パッチ数 = トークン数（圧縮なし）
- 224² + 16² パッチ = 196 トークン

### 現代の流れ

MLLM の文脈長制約に対応するため、**ViT 出力をさらに圧縮する機構** が標準化：

| 圧縮機構 | 採用モデル | 圧縮率 |
|---|---|---|
| **Position-aware VL Adapter** | [[entities/qwen-vl]] | 256 学習可能クエリで cross-attention 圧縮 |
| **Pixel Shuffle** | [[entities/internvl-1-5]] | 視覚トークン **1/4 圧縮** |
| **Merger**（MLP 2×2）| [[entities/qwen2-vl]] 系 | MLP で隣接 2×2 → 1 トークン **1/4 圧縮** |
| **Average Pooling** | [[entities/gemma-3]] | 4×4 = **16× 圧縮**（256 トークン固定） |
| **ViR（Visual Resolution Router）** | [[entities/internvl-3-5]] | patch ごとに 1/4 or 1/16 動的選択 |
| **2 層 ConvNet** | [[entities/deepseek-ocr]] | SAM 出力を **16× ダウンサンプリング** |

これは **元 ViT になかった新コンポーネント**。MLLM 時代の要請から生まれた。

特筆すべきは [[entities/deepseek-ocr|DeepSeek-OCR]] の **DeepEncoder**:
```
入力画像
   ↓ SAM-base (80M, window attention で高解像度処理)
   ↓ 2 層 ConvNet (16× ダウンサンプリング)
   ↓ CLIP-large (300M, global attention で意味抽出)
   ↓ 圧縮された視覚トークン
```
これは **元 ViT の「単一パス処理」** から **「複数 vision モデルの直列ハイブリッド」** への大きな概念的変化。

---

## DINOv3 / SigLIP 2 を例にした「変化の積み重ね」の可視化

ユーザが特に挙げた 2 つのモデルで、元 ViT からの累積変化を整理：

### [[entities/dinov3|DINOv3]] (Meta, 2025) のアーキテクチャ

```
ViT (2020) ベース
+ SwiGLU              ← 変化 ③（DINOv2 で導入）
+ LayerScale          ← 変化 ②（DINOv2 で導入）
+ Register tokens     ← 変化 ④
+ axial RoPE          ← 変化 ①（最大の変化、DINOv3 で導入）
+ FlashAttention      ← 変化 ⑤
─────────────────────
+ Gram anchoring（損失関数で正則化、構造変化ではないが新概念）
+ RoPE-box jittering（訓練時拡張、構造変化ではない）
+ ViT-7B にスケール（規模拡大）
```

**結果**: 256² 訓練 → 4096² 推論で安定。**構成要素の多くが入れ替わった** からこそ実現できたスケール。

### [[sources/siglip-2|SigLIP 2]] (Google, 2025) のアーキテクチャ

```
ViT (2020) ベース
+ NaFlex（可変解像度・アスペクト比）  ← 変化 ⑦（新規）
─────────────────────
+ LocCa decoder（追加コンポーネント、構造拡張）
+ 補助損失（SILC 自己蒸留、TIPS マスク予測、ACID 蒸留）← 損失変化のみ
+ g/16 (1B) サイズに拡大
+ 多言語（109 言語）対応
+ 損失関数: sigmoid（SigLIP 1 で導入、cf. [[questions/sigmoid-vs-softmax-loss]]）
```

**SigLIP 2 は損失関数とデータの工夫が中心** でアーキテクチャ変更は控えめ（NaFlex のみ）。これが **「ユーザが『SigLIP 2 ではアーキテクチャ変化少ない』と感じる正当な根拠」**。

### 対比

| 比較軸 | DINOv3 | SigLIP 2 |
|---|---|---|
| 位置エンコ | **axial RoPE**（大変更）| NaFlex（小変更）|
| 活性化 | SwiGLU | （元 SigLIP のまま）|
| Register tokens | 採用 | 不採用 |
| 主要な革新 | **アーキテクチャ + 損失（Gram anchoring）** | **損失と補助タスクの 5 系統融合** |

**DINOv3 はアーキテクチャ革新型、SigLIP 2 は学習革新型** という対照的な路線を取っているのが見えます。

---

## メタ観察: 「LLM 系の改良が CV に逆輸入された」

これが ViT 以降の変化の **通底テーマ**:

| 技術 | NLP での起源 | CV への伝搬 |
|---|---|---|
| **RoPE** | [[sources/roformer|RoFormer]] (2021) | DINOv3, SAM 2, Qwen-VL 系 (2024-2025) |
| **RMSNorm** | T5, LLaMA | Qwen2.5-VL ViT, MLLM 系統一 |
| **SwiGLU** | PaLM (2022), LLaMA | DINOv2/v3 (2023-2025) |
| **YaRN** | LLM 長文脈 (2023) | Qwen3-VL の 1M YaRN 外挿 (2025) |
| **Pre-norm 安定化** | LLM 大規模化 | ViT 深層化 (DINOv2 の LayerScale) |
| **FlashAttention** | LLM 高速化 (2022) | DINOv3 採用 (2025) |
| **Window Attention** | Sparse Transformer (NLP, 2019) | Qwen2.5-VL ViT (2025) |

### ViT 時代の哲学転換

- **ViT (2020)**: 「**Vision のための独自設計**」(GELU、学習可能位置埋め込み、[CLS]、LayerNorm)
- **2025**: 「**LLM と同じアーキテクチャ要素を共有**」(SwiGLU、RoPE、register tokens 風、RMSNorm)

この **「Vision と Language が同じアーキテクチャ要素を共有する」方向への移行** が、本質的な変化の通底テーマです。背景には:

1. **MLLM の興隆**: 視覚モデルが LLM とインターフェース化される必要
2. **共通インフラ**: 同じ最適化技法（FlashAttention 等）を使える
3. **理論的洞察の共有**: RoPE の任意系列長対応など、両方の分野で同じ問題に有効
4. **エンジニアの共通スキル**: LLM 開発者が Vision にも進出

---

## まとめ: ユーザの問いへの直接的回答

### Q: アーキテクチャの本質的部分は ViT からあまり変化していない？

**A: 半分正しい**。

#### 正しい点
- ✅ 「Patchify + Transformer Encoder × L」という **基本骨格は不変**
- ✅ 「画像 = トークン系列」というパラダイムは ViT が確立した遺産
- ✅ MSA + MLP + 残差 + LN という **モジュール構造** は変わらない
- ✅ 特に SigLIP 2 のように「**学習・データの工夫中心**」のモデルもある

#### 変化している点（積み重ねで実は大きい）

1. **位置エンコーディング**: 学習可能 1D → RoPE 系（**最大の変化**）
2. **Norm**: LayerNorm → RMSNorm + QK-norm
3. **活性化**: GELU → SwiGLU
4. **Register tokens** の追加
5. **Attention 効率化**: 完全 SA → Window / 階層型（タスク依存）
6. **出力読み出し**: 最終 [CLS] → 中間層活用 / attention pooling
7. **解像度処理**: 固定 → 任意解像度（構造変更を伴う）
8. **トークン圧縮機構**: 新コンポーネント（Merger / Pixel Shuffle / ViR 等）

### 本質的な観察

**「ViT 時代の Vision 独自進化」** から **「Vision と Language が同じアーキテクチャ要素を共有」** への移行が、変化の通底テーマ。具体的には **NLP からの逆輸入**（RoPE / SwiGLU / RMSNorm / YaRN / FlashAttention 等）が CV のアーキテクチャを実質的に塗り替えています。

### 同時に正しいユーザの観察

ユーザが「**学習やデータの工夫は大幅に改善**」と感じる根拠も妥当です。**DINO 系（自己蒸留 + multi-crop）/ SigLIP 系（sigmoid + 5 系統融合）/ Web スケールデータ（LVD-1689M / WebLI 10B）** などは確かにアーキテクチャと独立した革新で、こちらも 2020-2025 で劇的に進化しています。

**両方が並行して進んだのが ViT 以降の 5 年間** という総括が最も正確。

### 系統的な俯瞰

[[questions/large-scale-pretraining-series]] / [[questions/vit-dynamic-resolution-evolution]] / [[questions/rope-extrapolation-mechanism]] / [[questions/vlm-vision-encoder-continued-pretraining]] と組み合わせて理解すると、ViT 以降の進化は **学習・データ・アーキテクチャの 3 軸並行進化** が見えてきます：

```
ViT (2020) ── 共通の出発点
      │
      ├── 学習軸: DINO → DINOv2 → DINOv3 / SigLIP → SigLIP 2
      ├── データ軸: ImageNet → LAION → LVD-142M → LVD-1689M / WebLI 10B
      └── アーキテクチャ軸: 8 つの変化（本ページ）
                              ↓
                         全部の積み重ね = 現代の DINOv3 / SigLIP 2
```

---

## 関連ページ

### ViT 原典と概念

- [[concepts/vision-transformer]] — ViT 概念
- [[sources/vision-transformer]] — ViT 原典（Dosovitskiy et al., ICLR 2021）

### 位置エンコーディング関連（変化 ①）

- [[concepts/rotary-position-embeddings]] — RoPE 概念
- [[sources/roformer]] — RoPE 原典（NLP）
- [[sources/yarn]] — RoPE 拡張（YaRN）

### Norm/活性化関連（変化 ②③）

- [[entities/dinov2]] / [[sources/dinov2-learning-robust-visual-features-without-supervision]] — SwiGLU + LayerScale + 大規模 SSL
- [[entities/gemma-3]] / [[sources/gemma-3]] — QK-norm 採用例

### Register tokens / 中間層活用（変化 ④⑥）

- [[entities/dinov3]] / [[sources/dinov3]] — Register tokens + axial RoPE + Gram anchoring
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — 中間層特徴の発見
- [[concepts/alignment-tuning]] — 中間層活用の方法論

### Attention 効率化（変化 ⑤）

- [[entities/sam-2]] / [[entities/hiera]] — 階層型 ViT
- [[entities/qwen2-5-vl]] — Window Attention

### 解像度処理（変化 ⑦）

- [[sources/siglip-2]] — NaFlex
- [[entities/dinov3]] — axial RoPE + box jittering
- [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]] — Naive Dynamic Resolution の発展
- [[entities/internvl-1-5]] — 動的タイル分割
- [[entities/gemma-3]] — Pan & Scan
- [[entities/deepseek-ocr]] — 6 解像度モード

### トークン圧縮機構（変化 ⑧）

- [[entities/qwen-vl]] — Position-aware VL Adapter
- [[entities/internvl-1-5]] — Pixel Shuffle
- [[entities/qwen2-vl]] — Merger
- [[entities/gemma-3]] — Average Pooling
- [[entities/internvl-3-5]] — ViR

### 関連 question

- [[questions/vit-dynamic-resolution-evolution]] — 変化 ⑦ の詳細
- [[questions/rope-extrapolation-mechanism]] — 変化 ① の数学的背景
- [[questions/vlm-vision-encoder-continued-pretraining]] — 変化 ①〜⑧ がなぜ起きるかの動機
- [[questions/large-scale-pretraining-series]] — 学習・データ軸の俯瞰
- [[questions/sigmoid-vs-softmax-loss]] — SigLIP の学習革新（アーキテクチャではない部分）
