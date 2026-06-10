---
type: question
asked: 2026-06-10
question: "VLM の vision encoder は SigLIP などを継続事前学習しているとあるが、何のために継続事前学習するのか？"
sources_used: ["[[concepts/foundation-model]]", "[[concepts/weakly-supervised-pretraining]]", "[[concepts/rotary-position-embeddings]]", "[[concepts/alignment-tuning]]", "[[concepts/vision-transformer]]", "[[entities/siglip]]", "[[entities/clip]]", "[[entities/dfn]]", "[[entities/perception-encoder]]", "[[entities/qwen-vl]]", "[[entities/qwen2-vl]]", "[[entities/qwen2-5-vl]]", "[[entities/qwen3-vl]]", "[[entities/internvl]]", "[[entities/internvl-1-5]]", "[[entities/mini-internvl]]", "[[entities/internvit-300m]]", "[[entities/internvl-2-5]]", "[[entities/internvl-3]]", "[[entities/internvl-3-5]]", "[[entities/gemma-3]]", "[[entities/deepseek-ocr]]", "[[sources/qwen2-vl]]", "[[sources/qwen2-5-vl]]", "[[sources/qwen3-vl]]", "[[sources/gemma-3]]", "[[sources/internvl-1-5]]", "[[sources/mini-internvl]]", "[[sources/perception-encoder]]", "[[questions/vit-dynamic-resolution-evolution]]", "[[questions/rope-extrapolation-mechanism]]"]
---

# VLM の vision encoder を継続事前学習する 6 つの目的

## 結論を先に

VLM（MLLM）の vision encoder で **[[entities/siglip|SigLIP]] / [[entities/dfn|DFN-CLIP]] / [[entities/internvit-300m|InternViT]]** 等を継続事前学習する目的は、wiki の複数モデル事例から横断的に整理すると **6 つの軸** に集約できます：

1. **解像度・アスペクト比の柔軟化**（固定 224²/384² → 任意解像度）
2. **位置エンコーディングの変更**（学習可能 1D → 2D-RoPE / M-RoPE）
3. **OCR・文書・図表理解能力の強化**（自然画像 → 文書・表・チャート）
4. **計算効率の改善**（トークン圧縮、Window Attention 等）
5. **LLM と整合する特徴空間への整形**（projector を超えた本質的整合）
6. **追加モダリティ・タスクへの適応**（動画、3D、グラウンディング等）

**例外**: [[entities/gemma-3|Gemma 3]] のように **凍結戦略** を取る MLLM もあり、これは「訓練コスト vs カスタマイズ度」のトレードオフ。

本質的には **「汎用視覚モデル → MLLM 用にカスタマイズされた視覚モジュール」への変換** が継続事前学習の役割。

---

## 背景: なぜそもそも継続事前学習が必要なのか

[[concepts/weakly-supervised-pretraining|WSL 系の vision encoder]]（[[sources/siglip|SigLIP]] / [[entities/dfn|DFN]] / [[entities/clip|CLIP]] 等）は **汎用画像-テキスト対比学習** で事前学習されています:

- **訓練データ**: Web スケールの自然画像とキャプション（LAION, WebLI, WIT, HQITP-350M 等）
- **訓練目的**: 画像とテキスト埋め込みの整列（ゼロショット分類向け）
- **入力形式**: **固定解像度**（224² / 384² / 378²）、**1:1 アスペクト比中心**

しかし MLLM の実用タスクはこれと大きく異なります:

| 観点 | SigLIP/CLIP の前提 | MLLM の要求 |
|---|---|---|
| 解像度 | 224² 固定 | 高解像度・任意 (224²〜4K+) |
| アスペクト比 | 1:1 中心クロップ | 縦長・横長・パノラマ |
| データ分布 | 自然画像中心 | 文書・OCR・チャート・UI も |
| トークン数 | 数百固定 | 文脈長制約があるので圧縮したい |
| LLM 接続 | テキスト・エンコーダと整列 | LLM デコーダと整合必要 |
| モダリティ | 画像-テキスト 2 種 | 動画・3D・音声も |

**このギャップを埋めるのが継続事前学習**。SigLIP の良い汎用視覚表現を保ちつつ、MLLM 用に **カスタマイズ** します。

---

## 目的 1: 解像度・アスペクト比の柔軟化（最重要）

### なぜ必要か

SigLIP / CLIP / DFN は **固定解像度** で訓練されている:

- **[[entities/siglip|SigLIP]]**: 224²（base）/ 384²（large）
- **[[entities/dfn|DFN-CLIP]]**: 224² / 378²
- **OpenAI CLIP**: 224² / 336²

しかし MLLM のタスクは **任意解像度** を要求:

- 文書理解: 細部の文字を読むため高解像度（1024²+）が必要
- スクリーンショット: 縦長 / 横長アスペクト比
- 動画: 多様な解像度

### 具体例

**[[entities/qwen2-vl|Qwen2-VL]]** ([[sources/qwen2-vl]]):
- [[entities/dfn|DFN-CLIP]] の重みから始めて **絶対位置埋め込みを 2D-RoPE に置換**
- **Naive Dynamic Resolution** で任意解像度対応
- → これは継続事前学習で初めて可能（DFN-CLIP は元々 224² 固定）

**[[entities/internvl-1-5|InternVL 1.5]]** ([[sources/internvl-1-5]]):
- InternViT-6B-V1.5 を **動的高解像度（1-40 タイル / 最大 4K）** に継続事前学習
- 結果: ChartQA 83.8、OCRBench 724（当時の商用モデル超え）

**[[entities/qwen3-vl|Qwen3-VL]]** ([[sources/qwen3-vl]]):
- **[[entities/siglip|SigLIP-2]] から継続学習**で任意解像度を獲得

[[questions/vit-dynamic-resolution-evolution]] で詳述している通り、**固定解像度 → 任意解像度への移行** が継続事前学習の最も主要な動機です。

---

## 目的 2: 位置エンコーディングの変更

### なぜ必要か

- **元の SigLIP/CLIP**: 学習可能 1D 絶対位置埋め込み（固定解像度向け）
- **MLLM が必要とする位置エンコーディング**:
  - **2D-RoPE**: 縦横の幾何関係を表現（[[concepts/rotary-position-embeddings]] / [[sources/roformer]]）
  - **M-RoPE**: temporal / height / width の 3 軸分解（[[entities/qwen2-vl]] で導入）
  - **Interleaved MRoPE**: t/h/w を埋め込み次元に均等配置（[[entities/qwen3-vl]] で導入）
  - **YaRN 外挿**: 長文脈対応（[[sources/yarn]]）

これらは **学習可能 1D 位置埋め込み**（元の SigLIP/CLIP の方式）と互換性がないので、置き換える必要があり、**置き換え後に再訓練** が不可欠。

### 具体例

**[[entities/qwen2-vl|Qwen2-VL]]**:
- **DFN-CLIP の絶対位置埋め込みを 2D-RoPE に置換** してから継続事前学習
- 結果: 学習 16K → 推論 80K トークンまで頑健に外挿

**[[entities/qwen3-vl|Qwen3-VL]]**:
- **SigLIP-2 から継続学習** + **Interleaved MRoPE** + **テキスト・ベース時間整合**
- 結果: 256K ネイティブ + 1M YaRN 外挿（cf. [[questions/rope-extrapolation-mechanism]]）

詳細は [[concepts/rotary-position-embeddings]] / [[sources/roformer]] / [[sources/yarn]] 参照。

---

## 目的 3: OCR・文書・図表理解能力の強化

### なぜ必要か

SigLIP の WebLI と CLIP の WIT-400M は **自然画像中心**:

- **得意分野**: 「犬」「車」「風景」など物体・シーン認識
- **苦手分野**: 文字 / 表 / グラフ / 化学式 / 楽譜 / UI

しかし MLLM の実用タスクは:

- **文書 OCR**（DocVQA、InfoVQA）
- **多言語 OCR**（中国語、アラビア語、ベトナム語等）
- **チャート理解**（ChartQA、PlotQA）
- **画像内テキスト読み取り**（TextVQA、OCRBench、OCRBench_v2）

### 具体例

**[[entities/internvl-1-5|InternVL 1.5]]** ([[sources/internvl-1-5]]):
- InternViT-6B-V1.5 に **継続事前学習で OCR / バイリンガル能力を強化**
- 結果: OCRBench 724、ChartQA 83.8、MMBench-CN 82.0（**GPT-4V 超え**）

**[[entities/internvit-300m|InternViT-300M]]** ([[sources/mini-internvl]]):
- アブレーションで **CLIP-ViT-L 単体より OCR で +8.4、Chart で +5.3、InfoVQA で +8.1**
- これは「**多ドメイン知識の継承**」を蒸留で実現した実例

**[[entities/qwen2-5-vl|Qwen2.5-VL]]** ([[sources/qwen2-5-vl]]):
- **多言語 OCR を 10 言語に拡張**、4.1T トークン継続事前学習
- 結果: OCRBench_v2 中国語 63.7（**Gemini 1.5-Pro 比 +20.6**）

**[[entities/qwen3-vl|Qwen3-VL]]**:
- **多言語 OCR を 39 言語に拡張**
- 結果: OCRBench 920、OCRBench_v2 中国語 63.5 SOTA

**[[entities/deepseek-ocr|DeepSeek-OCR]]**:
- 訓練データの 70% を **OCR 1.0**（30M+ PDF・100 言語）に集中
- 「視覚をテキスト圧縮媒体として活用」という極端な OCR 特化路線

---

## 目的 4: 計算効率の改善

### なぜ必要か

元の SigLIP/CLIP は **MLLM 用に最適化されていない**:

- ViT-L で 224² なら 256 トークン / 384² なら 729 トークン
- 高解像度では爆発的にトークン増（512² なら 1296、896² なら 4096、1024² なら 5184）
- MLLM は LLM の文脈長制約があるので、視覚トークンを **圧縮** したい

[[concepts/vision-transformer|ViT]] の self-attention は $\mathbb{O}(N^2)$ なので、高解像度で計算量も爆発します。

### 具体例

**[[entities/qwen2-5-vl|Qwen2.5-VL]]** ([[sources/qwen2-5-vl]]):
- 32 層中 4 層のみ完全 self-attention、残り 28 層は **Window Attention**
- 計算量を 2 次 → 線形化
- これは継続事前学習で **構造変更** を伴う

**Pixel Shuffle / Average Pooling**:
- [[entities/internvl-1-5|InternVL 1.5]]: Pixel Shuffle で視覚トークン 1/4 圧縮
- [[entities/gemma-3|Gemma 3]]: 4×4 Average Pooling で 256 トークン固定圧縮
- これらは継続事前学習で「圧縮表現」を最適化する必要

**[[entities/internvl-3-5|InternVL 3.5]] の ViR（Visual Resolution Router）**:
- patch ごとに 1/4 or 1/16 の動的圧縮率を学習
- **視覚トークン 50% 削減で性能 99% 維持**

**[[entities/deepseek-ocr|DeepSeek-OCR]]**:
- SAM-base + 2 層 ConvNet（16× ダウンサンプリング）+ CLIP-large
- 高解像度入力を極端に圧縮する直列パイプライン

---

## 目的 5: LLM と整合する特徴空間への整形

### なぜ必要か

SigLIP/CLIP の出力特徴は **テキスト・エンコーダ** と整合するように訓練されており、**LLM デコーダ**（Qwen / InternLM / LLaMA）の言語空間とは異なる:

- 単純な MLP projector では本質的整合が不十分
- projector で投影しても、視覚特徴の **粒度** や **抽象度** が LLM の期待と合わない
- 特に **中間層の活用**（[[concepts/alignment-tuning]]）が継続事前学習で本領発揮

### 具体例

**[[entities/perception-encoder|Perception Encoder (PE)]]** ([[sources/perception-encoder]]):
- **中間層特徴の発見**: 「対比学習をスケールすると中間層に多目的特徴量が育つ」
- **[[concepts/alignment-tuning|alignment tuning]]** で中間層を末端に引き出す
- PEcore / PElang / PEspatial の **3 バリアントを継続事前学習で分化**

**[[entities/qwen3-vl|Qwen3-VL]] の DeepStack**:
- ViT 中間 3 層の視覚特徴を **LLM 最初 3 層に注入**
- これは SigLIP-2 を継続学習する中で組み込む必要

**[[entities/internvl-3|InternVL 3]] の Native Multimodal Pre-Training**:
- 「LLM Chat 版から事後改造」を捨て、**テキスト + マルチモーダルを共同事前学習**
- 視覚エンコーダも LLM と一緒に再訓練する立場

**[[entities/internvl-2-5|InternVL 2.5]] の発見**:
- **「最終層の線形分離性が低下、attention pooling 維持」** という観察を独立発見
- [[entities/perception-encoder|PE]] の中間層特徴発見と同等の現象

---

## 目的 6: 追加モダリティ・タスクへの適応

### なぜ必要か

SigLIP / DFN-CLIP は **画像-テキスト 2 モダリティ** のみ。MLLM はそれ以上を要求:

- **動画**（時間軸）
- **3D**（depth、point cloud、9-DoF bounding box）
- **音声-視覚**
- **グラウンディング**（`<box>` トークン対応）

### 具体例

**[[entities/qwen2-vl|Qwen2-VL]]** ([[sources/qwen2-vl]]):
- ViT を **動画対応** に継続事前学習
- 2 fps + 深さ 2 の 3D 畳み込み（3D チューブ化）
- **20 分以上の動画理解**

**[[entities/qwen3-vl|Qwen3-VL]]**:
- **9-DoF 3D bbox Omni3D 形式** に対応
- Spatial Understanding（VSI-Bench、EmbSpatialBench）でも SOTA

**[[entities/qwen-vl|Qwen-VL]] / 後継**:
- `<box></box>` / `<ref></ref>` 特殊トークン対応
- RefCOCO val 89.36 → Qwen2-VL で 93.2 SOTA に発展

**[[entities/qwen3-5-omni|Qwen3.5-Omni]]**:
- AuT（4000 万時間音声学習）+ 視覚エンコーダの音声-視覚統合
- 動画-音声共同学習

---

## 例外: なぜ Gemma 3 は SigLIP を凍結するのか？

すべての MLLM が継続事前学習するわけではありません。**[[entities/gemma-3|Gemma 3]]** ([[sources/gemma-3]]) は対照的な戦略を取ります:

### Gemma 3 の設計判断

- **SigLIP-400M variant を共有・凍結**（4B/12B/27B 全サイズで同一 ViT、訓練中変更なし）
- 896² 固定 + 4×4 Average Pooling で 256 トークン固定圧縮
- **Pan & Scan (P&S)**（推論時のみのタイル分割）で部分的に解像度問題に対応

### 凍結する利点

**Vision encoder の埋め込みを事前計算してキャッシュ** することで、言語モデル訓練コストをゼロに:

| 項目 | Gemma 3（凍結） | Qwen / InternVL（継続学習） |
|---|---|---|
| **訓練コスト** | 低（ViT 訓練なし）| 高（ViT も訓練）|
| **解像度** | 896² 固定（推論時 P&S） | 任意解像度 |
| **OCR 強化** | 限定的（Pan & Scan 部分対応） | 強化される |
| **能力ベース継承** | SigLIP の純粋な能力 | カスタマイズされる |
| **Pan & Scan 効果** | InfoVQA +17.0 / DocVQA +4.8 | 不要（任意解像度で対応） |

### Gemma 3 戦略の背景

- **Google が選んだ保守路線**: Qwen の根源的解決（2D-RoPE で ViT 改造）と対照的な実装的解決（LLaVA-inspired タイル分割）
- **コスト効率最優先**: Consumer-grade（スマホ・ラップトップ）で動くことを重視

これは [[questions/vit-dynamic-resolution-evolution]] の「**タイル分割路線**」（LLaVA → InternVL 1.5 → Gemma 3）の最新形。

---

## 各モデルの継続事前学習戦略まとめ

wiki に登録された主要 MLLM の vision encoder 戦略を一覧化:

| MLLM | Base | 継続事前学習の内容 |
|---|---|---|
| [[entities/qwen-vl\|Qwen-VL]] (2023.08) | OpenCLIP ViT-bigG | 224² 固定 + Position-aware VL Adapter（256 トークン圧縮）|
| [[entities/qwen2-vl\|Qwen2-VL]] (2024.09) | **[[entities/dfn\|DFN-CLIP]]** | 2D-RoPE 置換 + Naive Dynamic Resolution + 動画対応 + 1.4T トークン |
| [[entities/qwen2-5-vl\|Qwen2.5-VL]] (2025.02) | DFN + 社内データから**ゼロから**| Window Attention + MRoPE absolute time + 多言語 OCR + 4.1T トークン |
| [[entities/qwen3-vl\|Qwen3-VL]] (2025.11) | **[[entities/siglip\|SigLIP-2]]** | DeepStack + Interleaved MRoPE + テキスト・タイムスタンプ + 2.2T トークン |
| [[entities/internvl\|InternVL 1.0]] (2024) | (ゼロから 4.98B 対) | 3 段階訓練（対比 → 生成 → SFT）で InternViT-6B 構築 |
| [[entities/internvl-1-5\|InternVL 1.5]] (2024.04) | InternViT-6B | **48 → 45 層 shrink** + 動的高解像度 + 中英 OCR |
| [[entities/internvit-300m\|InternViT-300M]] (2024.10) | **CLIP-ViT-L** | InternViT-6B から **蒸留**（negative cosine sim、最後の K 層）|
| [[entities/internvl-2-5\|InternVL 2.5]]+（V2.5 系） | InternViT-V1.5 | データ品質 + Progressive Scaling で更新 |
| [[entities/internvl-3-5\|InternVL 3.5]] (2025.08) | InternViT-V2.5 | **継続学習なし**（InternVL 2.5 から完全継承）+ ViR で動的圧縮 |
| [[entities/gemma-3\|Gemma 3]] (2025.03) | SigLIP-400M | **継続学習なし**（凍結 + Pan & Scan で部分対応）|
| [[entities/deepseek-ocr\|DeepSeek-OCR]] (2025.10) | SAM-base + CLIP-large | 直列ハイブリッド + 高解像度 OCR 特化訓練 |

### 戦略の 3 分類

1. **積極的継続学習**（Qwen 系 / InternVL 1.5、Mini-InternVL）
   - 構造変更（位置 RoPE 置換、層削減）+ 大規模再訓練
   - 解像度・OCR・モダリティ拡張を一気に実現

2. **凍結戦略**（[[entities/gemma-3|Gemma 3]] / [[entities/internvl-3-5|InternVL 3.5]] の InternViT-V2.5）
   - 既存の優秀な vision encoder の能力を保持
   - 訓練コスト削減、推論最適化に集中

3. **ハイブリッド**（[[entities/deepseek-ocr|DeepSeek-OCR]]）
   - 複数の事前学習済みモデルを直列に組み合わせ
   - SAM（高解像度処理）+ CLIP（意味抽出）の利点を結合

---

## なぜ「ゼロから」と「継続事前学習」を選ぶのか

各 MLLM の選択を見ると、**訓練リソース vs 時間 vs 性能** のトレードオフが見える:

| 戦略 | 訓練コスト | 性能上限 | 開発期間 | 代表例 |
|---|---|---|---|---|
| **凍結** | 最低 | SigLIP の上限 | 最短 | Gemma 3 |
| **継続事前学習** | 中 | カスタマイズで上回る可能性 | 中 | Qwen2-VL, Qwen3-VL |
| **ゼロから訓練** | 最高 | 最大限カスタマイズ可 | 最長 | Qwen2.5-VL（DFN 風 + 社内データ）|

[[entities/qwen2-5-vl|Qwen2.5-VL]] が「ゼロから訓練」を選んだ理由は **Window Attention の構造改変**が継続学習では実現困難だったため。一方で [[entities/qwen3-vl|Qwen3-VL]] は SigLIP-2 が登場したことで「優秀な基盤からの継続学習」に戻りました。

---

## まとめ: 継続事前学習の本質

VLM の vision encoder の継続事前学習は、**「汎用視覚モデル → MLLM 用にカスタマイズされた視覚モジュール」への変換** が本質。

具体的な変換は **6 軸** で行われる:

1. **解像度・アスペクト比** （最重要、固定 → 任意）
2. **位置エンコーディング** （1D → 2D-RoPE / M-RoPE）
3. **データ分布** （自然画像 → 文書・OCR・図表）
4. **計算効率** （トークン圧縮、Window Attention）
5. **LLM との整合性** （projector を超えた本質的整合）
6. **新モダリティ** （動画、3D、グラウンディング）

そして **凍結戦略**（Gemma 3 型）もありえるトレードオフ。継続学習が「正解」とは限らず、**訓練コスト vs カスタマイズ度** の選択です。

### 系統的な俯瞰

[[questions/large-scale-pretraining-series]] と組み合わせて理解すると:
- **基盤 vision encoder**（系列 ① WSL: SigLIP / CLIP / DFN / PE）が「素材」を提供
- MLLM 各社が継続事前学習で「料理」する
- 例外として「素材のまま使う」（Gemma 3）戦略もある

ほぼすべての商用級 MLLM の vision encoder は **CLIP/SigLIP 系の継承・改造** であり、独立系列ではないという [[questions/large-scale-pretraining-series]] の結論を、継続事前学習の観点から具体的に裏付けます。

---

## 関連ページ

### 直接の前提

- [[concepts/foundation-model]] — 基盤モデル全体、CLIP/SigLIP 系の位置付け
- [[concepts/weakly-supervised-pretraining]] — WSL の系統（vision encoder の出発点）
- [[concepts/rotary-position-embeddings]] — 位置エンコーディング変更の前提
- [[concepts/alignment-tuning]] — PE の中間層活用の前提

### 主要な vision encoder

- [[entities/siglip]] / [[sources/siglip]] — Google の sigmoid 対比 vision encoder
- [[entities/clip]] / [[sources/clip]] — vision encoder の起点
- [[entities/dfn]] / [[sources/dfn]] — Apple のキュレーション特化 CLIP
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — 中間層活用パラダイム

### MLLM 各シリーズの継続事前学習

**Qwen-VL 系**:
- [[entities/qwen-vl]] / [[sources/qwen-vl]]
- [[entities/qwen2-vl]] / [[sources/qwen2-vl]]
- [[entities/qwen2-5-vl]] / [[sources/qwen2-5-vl]]
- [[entities/qwen3-vl]] / [[sources/qwen3-vl]]

**InternVL 系**:
- [[entities/internvl]] / [[sources/internvl]]
- [[entities/internvl-1-5]] / [[sources/internvl-1-5]]
- [[entities/mini-internvl]] / [[sources/mini-internvl]]
- [[entities/internvit-300m]] — 蒸留の代表例
- [[entities/internvl-2-5]] / [[sources/internvl-2-5]]
- [[entities/internvl-3]] / [[sources/internvl-3]]
- [[entities/internvl-3-5]] / [[sources/internvl-3-5]]

**凍結戦略・特殊系**:
- [[entities/gemma-3]] / [[sources/gemma-3]] — 凍結戦略の代表
- [[entities/deepseek-ocr]] / [[sources/deepseek-ocr]] — ハイブリッド戦略

### 関連 question

- [[questions/vit-dynamic-resolution-evolution]] — 解像度進化、本ページ目的 1 の詳細
- [[questions/rope-extrapolation-mechanism]] — 位置エンコーディング外挿、本ページ目的 2 の数学的背景
- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列、vision encoder の系統的位置付け
- [[questions/sigmoid-vs-softmax-loss]] — SigLIP の損失関数、なぜそれが継続学習の基盤として優秀か
