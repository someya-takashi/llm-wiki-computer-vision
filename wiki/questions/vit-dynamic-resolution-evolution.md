---
type: question
asked: 2026-05-30
question: "ViT などでは固定解像度のものが多いと思うが、なぜ固定解像度が多いのか。任意解像度で入力できるようになったのはなぜか。ViT の登場からの解像度の扱いの変化は？"
sources_used: [[concepts/vision-transformer]], [[concepts/rotary-position-embeddings]], [[entities/qwen-vl]], [[entities/qwen2-vl]], [[entities/qwen2-5-vl]], [[entities/qwen3-vl]], [[entities/internvl-1-5]], [[entities/internvl-2-5]], [[entities/internvl-3]], [[entities/internvl-3-5]], [[entities/dinov3]], [[entities/dino]], [[entities/dinov2]]
---

# ViT における解像度処理の進化 — 固定解像度から任意解像度へ

## 結論を先に

**ViT は構造的には任意解像度を扱える**設計だったが、**「学習可能絶対位置埋め込み」という特定の設計選択**が解像度を固定していた。任意解像度シフトは以下 3 つの圧力で 2023-2025 年に一気に進んだ：

1. **位置埋め込みを RoPE 系へ置換**
2. **パッチ系列の可変長化（パッキング + 圧縮）**
3. **MLLM 時代のタスク要求**（文書理解 / 動画 / GUI エージェント）

---

## なぜ固定解像度が「多かった」のか（4 つの原因）

### 原因 1: 学習可能絶対位置埋め込み（オリジナル ViT, 2020）

[[concepts/vision-transformer|ViT]] 自体は CNN と違って構造的には任意の入力長を扱える（self-attention はトークン数に依存しない）。しかし **位置埋め込みが固定解像度の元凶**だった：

- 224×224 入力で `/16` パッチサイズなら **196 個の位置に対する 196 個の学習可能ベクトル**を訓練
- 推論時に 512×512 を入れたければ、位置埋め込みを **bilinear 補間**する必要があった → 性能劣化

[[concepts/rotary-position-embeddings]] 「既存の位置埋め込みとの比較」より：
> 学習時 224×224（14×14 = 196 パッチ）で訓練すると、推論時に 512×512 などを入れたければ補間が必要

### 原因 2: 事前学習データセットの規格化

- ImageNet（224×224 中心）、JFT-300M、LAION（256×256 リサイズ）など、**事前学習データが固定解像度**で前処理されていた
- アスペクト比は中央クロップで強制的に 1:1 に
- バッチ学習の効率（同じ shape でないと GPU メモリで揃わない）も固定化を後押し

### 原因 3: 計算コストの 2 次性

- self-attention は **トークン数 n に対して O(n²)**
- `/16` パッチで 224² なら 196 トークンだが、512² なら 1024 トークン → 計算量 **27 倍**
- 「とりあえず 224 か 384 で固定」が現実的だった（[[concepts/vision-transformer]] §「ViT の弱点」より）

### 原因 4: 帰納バイアスの弱さ

- CNN は局所性・平行移動不変性を **構造として**持つので、解像度変化に比較的頑健
- ViT は帰納バイアスがほぼなく、解像度変化を **データから学ぶ**必要があった
- → 「学習時と同じ解像度に揃える」が安全策に

---

## 任意解像度になった理由（3 つの圧力）

### 圧力 1: タスクの要求（MLLM の台頭）

[[entities/qwen-vl|Qwen-VL]] (2023.08) は **448² 固定 + 256 トークン固定圧縮**だったが、これでは：

- 文書・チャート・インフォグラフィックの **細部が潰れる**
- 縦長スクリーンショット・横長パノラマの **アスペクト比情報が消える**
- 1 時間動画の **時間軸を扱えない**

実用 MLLM には任意解像度が必須になった。

### 圧力 2: 位置埋め込みの数学的進化（RoPE）

**RoPE（Su et al., 2021）が決定的**だった。[[concepts/rotary-position-embeddings]] §RoPE の長所より：

> RoPE は「位置 m に対して角度 mθ 回転」する操作。m は連続値として扱えるので、訓練時に見たことがない位置（パッチ数）でも、回転角を計算するだけで対応できる。

つまり RoPE は **構造的に任意解像度を許容**する。学習可能位置埋め込みの「196 ベクトルしかない」制約を根本から解消。

### 圧力 3: ハードウェアと並列化技術の成熟

- **FlashAttention** で長系列のメモリ効率が改善
- 動的 shape を扱う PyTorch / vLLM の機能成熟
- 可変長系列を 1 バッチに詰める **packing 技術**

---

## 進化の年表

### Phase 1: 固定解像度時代（2020-2023）

| 年 | モデル | 解像度処理 |
| --- | --- | --- |
| 2020 | **ViT** (Dosovitskiy et al.) | **224² 固定 / 学習可能絶対位置埋め込み** |
| 2021 | CLIP ([[entities/clip]]) | 224² 固定（学習時）、推論時 336² 等は位置補間 |
| 2021 | DINO ([[entities/dino]]) | 224² 固定 / multi-crop で擬似的に多解像度 |
| 2023 | DINOv2 ([[entities/dinov2]]) | 224² 中心、518² で蒸留 |
| 2023.08 | **[[entities/qwen-vl|Qwen-VL]]** | **448² 固定 + 256 トークン固定圧縮** ← MLLM 初期は固定 |

### Phase 2: タイル分割で任意アスペクト比対応（2024）

「ViT 自体は固定 448²、ただし画像を **複数タイルに分割して各タイルを独立処理**」という工夫が登場：

| 年 | モデル | 解像度処理 |
| --- | --- | --- |
| 2024.04 | **[[entities/internvl-1-5|InternVL 1.5]]** | **35 通りのアスペクト比 × 1-12 タイル**（テスト時 40 タイル = 4K 相当）+ Pixel Shuffle 1/4 圧縮 |
| 2024.10 | [[entities/mini-internvl|Mini-InternVL]] | InternVL 1.5 と同じタイル方式 |
| 2024.12 | [[entities/internvl-2-5|InternVL 2.5]] | タイル方式継続 |
| 2025.03 | **[[entities/gemma-3\|Gemma 3]]** | **Pan & Scan (P&S)**: LLaVA 風適応的ウィンドウ、推論時のみ、非正方形・高解像度画像を 896×896 タイルに分割、推論高速化のため無効化可。**SigLIP 400M 凍結 + 256 トークン固定圧縮**との組み合わせで Google が選んだタイル分割路線の最新形 |
| 2025.10 | **[[entities/deepseek-ocr\|DeepSeek-OCR]]** | **「視覚トークン数最小化」という第 3 路線**: 動的解像度ではなく、**[[entities/sam\|SAM]] (window attention) + 2 層 ConvNet (16× 圧縮) + [[entities/clip\|CLIP]] (global attention) の直列構成** + 6 解像度モード（Tiny 64 / Small 100 / Base 256 / Large 400 / Gundam 動的）。OmniDocBench で **100 トークンで GOT-OCR2.0 (256 トークン) 凌駕**、**Gundam 795 トークンで MinerU2.0 (6790 トークン) 凌駕** = **8.5× 効率向上**。**「視覚は理解対象ではなくテキスト圧縮媒体」という LLM 中心の哲学的転換** |

**タイル分割の限界**: タイル境界で意味的連続性が切れる、メタ的なアスペクト比情報の損失。
**Gemma 3 P&S の独自性**: 推論時のみ適用、無効化可能なオプション機能として実装。**InfoVQA で +17.0 / DocVQA で +4.8** という劇的効果を示し、タイル分割の有効性を再確認。
**DeepSeek-OCR の新視点**: 「解像度を任意化する」競争に対し、**「視覚トークン数自体を最小化する」**という直交する解。Gundam で動的タイル + グローバル・ビューを採用しつつ、全体トークン数を 800 未満に抑える設計。

### Phase 3: ViT 自体を任意解像度対応にする（2024.09-）

[[entities/qwen2-vl|Qwen2-VL]] が **根源的解決**を提示：

> ViT の絶対位置埋め込みを除去し **2D-RoPE** を導入することで、任意の解像度のパッチ列に対して位置情報を付与できる（[[sources/qwen2-vl]]）

| 年 | モデル | 解像度処理 |
| --- | --- | --- |
| 2024.09 | **[[entities/qwen2-vl|Qwen2-VL]]** | **Naive Dynamic Resolution**: 2D-RoPE + MLP で隣接 2×2 トークンを 1 トークン圧縮 + 系列単位パッキング、**学習 16K → 推論 80K 外挿** |
| 2025.02 | [[entities/qwen2-5-vl|Qwen2.5-VL]] | + **Window Attention**（32 層中 4 層のみ完全自己注意）で計算量 **O(n²)→O(n) 線形化** + ViT をゼロから学習 |
| 2025.04 | [[entities/internvl-3|InternVL 3]] | **V2PE**（Variable Visual Position Encoding）、視覚トークンに位置インクリメント δ<1 |
| 2025.08 | [[entities/internvl-3-5|InternVL 3.5]] | **ViR**（Visual Resolution Router）で patch ごとに 1/4 or 1/16 圧縮率を動的選択 |
| 2025.11 | [[entities/qwen3-vl|Qwen3-VL]] | **Interleaved MRoPE**（t/h/w を埋め込み次元に交互配置で周波数スペクトル均衡化）+ **DeepStack**（ViT 中間 3 層を LLM 最初 3 層に注入）+ **256K ネイティブ → 1M YaRN 外挿** |

### Phase 4: 純粋 SSL でも任意解像度（並行進化、2025）

VL 文脈とは独立に、純粋 SSL の世界でも任意解像度化：

| 年 | モデル | 解像度処理 |
| --- | --- | --- |
| 2025 | [[entities/dinov3\|DINOv3]] | **axial RoPE + RoPE-box jittering**（座標 [-1,1] をランダムに [-s,s] にスケール）、**256² 訓練 → 4096² 推論で安定** |

---

## なぜ「ViT 自体を任意解像度対応」が時間がかかったか

[[entities/qwen2-vl|Qwen2-VL]] が 2024.09 にやっとこれを実現したのは、いくつかの非自明な工夫が必要だったから：

1. **位置埋め込みの置換**: 学習可能絶対位置 → 2D-RoPE（[[entities/qwen2-vl|Qwen2-VL]] / [[entities/dinov3|DINOv3]] 双方）
2. **可変長パッキング**: 異なる解像度の画像を 1 バッチに詰める実装が必要
3. **トークン数圧縮**: そのままだと文脈長が爆発するため、MLP 2×2 圧縮（Qwen2-VL）や Pixel Shuffle（InternVL 系）で 4× 圧縮
4. **計算複雑性対策**: 高解像度で 2 次計算量が爆発する問題を **Window Attention**（[[entities/qwen2-5-vl|Qwen2.5-VL]]）等で線形化

---

## 興味深い「揺り戻し」と分岐

### ViT 初期化戦略の揺れ動き

- **[[entities/qwen2-vl|Qwen2-VL]] (2024.09)**: DFN（Apple）の ViT で初期化 → 既存 VFM 活用路線
- **[[entities/qwen2-5-vl|Qwen2.5-VL]] (2025.02)**: ViT を **ゼロから学習**（自前で任意解像度に最適化）
- **[[entities/qwen3-vl|Qwen3-VL]] (2025.11)**: **SigLIP-2 から継続学習**に戻る（既存 VFM の活用 + 任意解像度継続学習）

→ 「ゼロから学習 vs 既存 VFM 継続学習」は現在も論争中

### 路線対立

| 路線 | 代表 | 特徴 |
| --- | --- | --- |
| **タイル分割** | InternVL 1.5 / 2.5 系 | ViT 自体は固定、画像をタイル化 |
| **ViT 改造** | Qwen2-VL / 2.5-VL / 3-VL 系 | 2D-RoPE で ViT を任意解像度対応 |
| **動的圧縮率** | InternVL 3.5 (ViR) | パッチごとに圧縮率を動的選択 |

---

## まとめ図解

```
ViT (2020) 固定 224²
    │ ← 学習可能絶対位置埋め込みが解像度を固定
    │
Qwen-VL (2023) 固定 448² + 256 トークン
    │
    ├─→ [タイル分割路線] InternVL 1.5 (2024.04) 1-40 タイル
    │        ↓
    │   InternVL 2.5, Mini-InternVL（タイル継続）
    │        ↓
    │   Gemma 3 (2025.03) Pan & Scan + SigLIP 400M 凍結（Google 系の最新形）
    │        ↓
    │   DeepSeek-OCR (2025.10) SAM + 16× ConvNet + CLIP の 3 段構成 + 6 モード
    │   = 「視覚トークン数最小化」という第 3 路線（OCR 特化）
    │        ↓
    │   InternVL 3.5 (2025.08) + ViR 動的圧縮率
    │
    └─→ [ViT 改造路線] Qwen2-VL (2024.09) 2D-RoPE で ViT 任意解像度化
             ↓
         Qwen2.5-VL (2025.02) + Window Attention で線形化
             ↓
         Qwen3-VL (2025.11) Interleaved MRoPE + DeepStack
             ↓
         256K ネイティブ / 1M YaRN 外挿（2 時間動画）

並行: DINOv3 (2025) 純粋 SSL でも axial RoPE で 256² → 4096² 対応
```

---

## 学べる教訓

1. **構造的可能性と実装上の制約は別物**: ViT は元から任意解像度を扱える設計だったが、**位置埋め込みという 1 つの設計選択**が長年制約していた
2. **数学的進化（RoPE）がパラダイムシフトを可能にする**: NLP で生まれた RoPE が CV に持ち込まれ、ViT 改造路線を開いた
3. **タスク要求が技術進化を駆動する**: 文書理解・動画・GUI エージェントという MLLM の実用要求が、任意解像度化を急速に進めた
4. **2 つの解決路線が並行発展する**: タイル分割（実装が簡単）vs ViT 改造（根源的解決）の対立は CV 分野の典型的パターン
5. **「ゼロから学習 vs 既存 VFM 継続学習」は循環する**: Qwen 系列が DFN → ゼロから → SigLIP-2 と揺り戻している

---

## 関連ページ

### 概念
- [[concepts/vision-transformer]] — ViT の基礎、固定解像度の根本理由
- [[concepts/rotary-position-embeddings]] — RoPE 系の進化（1D RoPE → 2D RoPE → M-RoPE → MRoPE absolute time → Interleaved MRoPE）

### モデル（時系列）
- [[entities/qwen-vl]] — 固定 448² + 256 トークン圧縮（MLLM 固定解像度の代表）
- [[entities/internvl-1-5]] — タイル分割路線の確立
- [[entities/gemma-3]] — Google が選んだタイル分割路線の最新形（Pan & Scan + SigLIP 400M 凍結）
- [[entities/deepseek-ocr]] — 「視覚トークン数最小化」という第 3 路線（SAM + 16× ConvNet + CLIP、6 解像度モード、OCR 特化）
- [[entities/qwen2-vl]] — Naive Dynamic Resolution（ViT 改造路線の出発点）
- [[entities/qwen2-5-vl]] — Window Attention で計算量線形化
- [[entities/internvl-3]] — V2PE
- [[entities/internvl-3-5]] — ViR で動的圧縮率
- [[entities/qwen3-vl]] — Interleaved MRoPE + DeepStack + 256K ネイティブ
- [[entities/dinov3]] — 純粋 SSL での axial RoPE + box jittering
