---
type: source
source_path: raw/papers/Gemma 3 Technical Report.md
source_kind: paper
title: "Gemma 3 Technical Report"
authors: [Gemma Team, Google DeepMind]
year: 2025
venue: arXiv:2503.19786
ingested: 2026-05-31
tags: [gemma, gemma-3, mllm, multimodal, vision-language, siglip, pan-and-scan, local-global-attention, qk-norm, qat, knowledge-distillation, google-deepmind, consumer-grade, open-source]
translation: [[translations/gemma-3]]
---

# Gemma 3 — Google DeepMind の SigLIP 400M + Pan&Scan + 5:1 local:global attention で「軽量オープン MLLM」を完成させた Gemma 系第 3 世代

> 原典: [[translations/gemma-3]] ・ `raw/papers/Gemma 3 Technical Report.md`
> 著者・年・会議: Gemma Team Google DeepMind, 2025 Mar, arXiv:2503.19786

## 一言まとめ

Google DeepMind が **Gemini 2.0 と co-design した軽量オープン MLLM ファミリー**（Gemma 系列第 3 世代）。**1B/4B/12B/27B の 4 サイズ**で、**1B は text-only、4B/12B/27B は SigLIP 400M variant をマルチモーダル化**。固定 896² + **LLaVA 風 Pan & Scan (P&S)** + **256 トークン固定圧縮**という保守的視覚処理に対し、**5:1 local:global attention 比 + sliding window 1024 + QK-norm + RoPE 10k→1M** で **128K 文脈の KV キャッシュ爆発を解決**。**14T/12T/4T/2T トークン事前学習**は全て**知識蒸留**で実施、**QAT（int4/sfp8）でスマホ展開可能**。**Gemma3-27B-IT が LMSys Chatbot Arena Elo 1338（rank 9）**で **DeepSeek-V3 (671B/37B MoE)、Llama-3.1-405B、Qwen2.5-72B を上回るオープン非推論モデル**となり、**Gemma3-4B-IT が Gemma2-27B-IT に匹敵**、**Gemma3-27B-IT が Gemini-1.5-Pro と同等**。Computer Vision wiki 内では **wiki 初の Google 系オープン MLLM** として、[[entities/qwen3-vl|Qwen3-VL]] / [[entities/internvl-3-5|InternVL 3.5]] の 2 大系譜と並ぶ第 3 の系列を確立。**Apache 2.0 風 Gemma License**（商用可）で公開。

## 背景と問題意識

[[entities/qwen3-vl|Qwen3-VL]]（Alibaba）と [[entities/internvl-3-5|InternVL 3.5]]（Shanghai AI Lab）が 2025 年のオープン MLLM 競争を牽引する中、**Google の Gemma 系列**は LLM として 1B-27B の軽量領域に注力してきた。Gemma 1（2024 Feb、テキスト専用）、Gemma 2（2024 Jul、テキスト専用、distillation 中心）の流れを継ぎ、**Gemma 3（2025 Mar）でついにマルチモーダル化**を実現。

**Gemma 3 の主要課題**:

1. **オープン軽量 MLLM の市場確立**: 1B-27B 範囲で **スマートフォン / ラップトップ / ハイエンド GPU** で動作する MLLM が必要。Qwen 系は 2B から始まり、InternVL 系も 1B/2B 等あるが、**Google ブランドのオープン軽量 MLLM** が欠落していた
2. **128K 文脈での KV キャッシュ爆発**: 標準 dense transformer の KV キャッシュは文脈長に比例して爆発し、軽量モデルでも消費者ハードウェアで OOM になる
3. **固定 vs 任意解像度のトレードオフ**: [[entities/qwen2-vl|Qwen2-VL]] の Native Dynamic Resolution（2D-RoPE で ViT 改造）は強力だが、**実装複雑性が高い**。Google は LLaVA 系の **シンプルなタイル分割 (Pan & Scan)** を選んだ
4. **PaliGemma 2 との分業**: Google 自身が **PaliGemma 2**（より大規模なオープン VLM、448×448 / 896×896 ベース）も持つ。Gemma 3 は **「軽量 + ChatBot 路線」**として差別化が必要

### Google DeepMind の戦略的判断

[[entities/qwen3-vl|Qwen3-VL]] の根源的解決（**Interleaved MRoPE + DeepStack + テキストタイムスタンプ**）に対し、Gemma 3 は **実装的解決（タイル分割 + KV キャッシュ最適化）** を選択。これは **「視覚エンコーダは凍結 + 推論時 P&S」** という保守的な分業設計で、**[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形**となる。

## 提案手法 / 主張

### モデル展開：4 サイズ（1B-27B）

| Model | Vision Encoder | Embedding | Non-embedding | 用途 |
| --- | --- | --- | --- | --- |
| **1B** | なし（text-only） | 302M | 698M | スマホ |
| **4B** | SigLIP 400M（共有） | 675M | 3,209M | ラップトップ |
| **12B** | SigLIP 400M（共有） | 1,012M | 10,759M | ハイエンド消費者 GPU |
| **27B** | SigLIP 400M（共有） | 1,416M | 25,600M | フラッグシップ |

語彙 256k（Gemini 2.0 と同じ SentencePiece、262k entries）。

### 3 つの主要構造革新

#### 1. Vision Modality — SigLIP 400M + Pan & Scan

- **Vision Encoder**: [[entities/siglip|SigLIP]] の 400M variant（CLIP 損失の sigmoid 変形版で訓練された [[concepts/vision-transformer|Vision Transformer]]）
- **共有設計**: 4B/12B/27B モデル間で同じエンコーダを共有、**訓練中は完全凍結**（言語モデルのみ訓練）
- **画像入力**: **896 × 896 の固定正方形**にリサイズ
- **256 トークン固定圧縮**: 視覚エンコーダ出力を **4×4 average pooling** で 256 ベクトルに圧縮（解像度に関係なく一定）
- **Pan & Scan (P&S)**: 推論時のみの適応的ウィンドウ・アルゴリズム
  - 非正方形・高解像度画像を **重複しない等サイズクロップ**に分割
  - 各クロップを 896×896 にリサイズしてエンコーダへ
  - 必要時のみ適用、最大クロップ数を制御
  - **LLaVA 風タイル分割**（[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形）

**P&S の効果**（表 8、27B IT）:
- **InfoVQA: 59.4 → 76.4（+17.0）**
- **DocVQA: 85.6 → 90.4（+4.8）**
- TextVQA: 68.6 → 70.2（+1.6）

→ **アスペクト比が変動する画像や画像内テキスト読み取り**で著しい改善

#### 2. 5:1 Local:Global Attention（KV キャッシュ最適化の革新）

**問題**: 標準 dense transformer は 128K 文脈で **KV キャッシュが 60% のメモリオーバーヘッド**を生む

**Gemma 3 の解決**:
- **local sliding window self-attention** と **global self-attention** を交互配置
- **5 local 層に対して 1 global 層**（Gemma 2 は 1:1、Gemma 3 は 5:1）
- **local の sliding window は 1024 トークン**のみ
- **long context は global 層のみ**が対応

**効果**（図 5、32k 文脈 2B モデル）:
- "global only" 構成: **KV キャッシュが 60% のメモリオーバーヘッド**
- "1:3, sw=4096"（Gemma 2）: 中程度のオーバーヘッド
- **"1:3, sw=1024"（Gemma 3）: 15% 未満に削減**

**RoPE 設定の工夫**:
- **global self-attention**: RoPE base 10k → **1M**（Gemma 2 から大幅増加）
- **local self-attention**: RoPE base 10k（変更なし）
- **位置補間 + スケーリング係数 8** で 32K 訓練 → 128K 推論を実現

#### 3. QK-norm（Gemma 2 からの構造改善）

- Gemma 2 の **soft-capping** を **QK-norm** に置換
- Chameleon、Olmo 2 などからの着想
- ([[concepts/vision-transformer|Vision Transformer]] 系で広く採用される正規化手法)

### Quantization Aware Training（QAT、スマホ展開の鍵）

未加工チェックポイントとともに **3 つの量子化形式**を提供：

| 形式 | 27B サイズ | 27B + KV (32k) | 用途 |
| --- | --- | --- | --- |
| **bf16 (Raw)** | 54.0 GB | 72.7 GB | サーバ |
| **Int4 per-channel** | 14.1 GB | 32.8 GB | ハイエンド消費者 |
| **Int4 blocks=32** | 15.3 GB | 34.0 GB | バランス型 |
| **SFP8 (Switched fp8)** | 27.4 GB | 46.1 GB | 高精度量子化 |

- **5,000 ステップの微調整**で量子化（QAT）
- 非量子化チェックポイントからの確率をターゲット
- **llama.cpp** などのオープンソース推論エンジンと互換

### 訓練レシピ

**事前学習データ**:
- 27B: **14T トークン**、12B: 12T、4B: 4T、1B: 2T（Gemma 2 より大）
- **画像 + テキスト + 多言語** の混合
- **Gemini 2.0 と同じトークナイザ**（262k 語彙、SentencePiece）

**知識蒸留**:
- **1 トークンあたり 256 ロジット**をサンプリング
- 教師の確率で重み付け、生徒はクロスエントロピー損失で学習
- 教師の非サンプル・ロジットは 0 確率に設定、再正規化

**インフラ**: TPUv4 / TPUv5e / TPUv5p、**ZeRO-3** + Pathways + GSPMD + MegaScale XLA、**視覚エンコーダの埋め込みは事前計算**して訓練コスト削減

**事後学習**（IT モデル）:
- 大規模 IT 教師からの**改善された知識蒸留**
- **BOND + WARM + WARP の改善版**による RL 微調整
- **重み平均報酬モデル** + 人間フィードバック + コード実行フィードバック + 数学 ground-truth 報酬
- ChatML 風フォーマット（`<start_of_turn>` / `<end_of_turn>` / `[BOS]`）

## 実験結果と知見

### LMSys Chatbot Arena Elo（表 5、2025 年 3 月 8 日時点）

| Rank | Model | Elo | Open | Type | 規模 |
| --- | --- | --- | --- | --- | --- |
| 1 | Grok-3-Preview / GPT-4.5-Preview | 1411-1412 | - | - | - |
| 6 | DeepSeek-R1 | 1363 | yes | MoE | 671B/37B |
| 8 | o1-2024-12-17 | 1352 | - | - | - |
| **9** | **Gemma-3-27B-IT** | **1338** | **yes** | **Dense** | **27B** |
| 13 | DeepSeek-V3 | 1318 | yes | MoE | 671B/37B |
| 28 | Llama-3.1-405B-Instruct | 1269 | yes | Dense | 405B |
| 38 | Llama-3.3-70B-Instruct | 1257 | yes | Dense | 70B |
| 39 | Qwen2.5-72B-Instruct | 1257 | yes | Dense | 72B |
| 59 | Gemma-2-27B-it | 1220 | yes | Dense | 27B |

**ハイライト**:
- **27B Dense で Elo 1338 は驚異的**: 671B/37B MoE の DeepSeek-V3 (1318)、405B の Llama-3.1-405B (1269) を **より小さなパラメータ数で上回る**
- **Gemma 2 からの飛躍**: 1220 → 1338（+118 Elo、約 25 ランクアップ）
- **オープン非推論モデルのトップクラス**（DeepSeek-R1 のみ上のオープンモデル、ただし推論モデル）

### 標準ベンチマーク（表 6、IT モデル）

| ベンチ | Gemini 1.5 Pro | Gemini 2.0 Pro | Gemma 2 27B | **Gemma 3 27B** |
| --- | --- | --- | --- | --- |
| MMLU-Pro | 75.8 | 79.1 | 56.9 | **67.5** |
| **MATH** | 86.5 | 91.8 | 55.6 | **89.0** |
| HiddenMath | 52.0 | 65.2 | 14.8 | **60.3** |
| GPQA Diamond | 59.1 | 64.7 | 34.3 | **42.4** |
| **MMMU (val)** | 65.9 | 72.7 | - | **64.9** |
| FACTS Grounding | 80.0 | 82.8 | 62.4 | **74.9** |
| Global MMLU-Lite | 80.8 | 86.5 | 68.6 | **75.1** |
| LiveCodeBench | 34.2 | 36.0 | 20.4 | **29.7** |

**ハイライト**:
- **MATH 89.0**: Gemini 1.5 Pro (86.5) を上回り Gemini 2.0 Pro (91.8) に肉薄
- **MMMU 64.9**: マルチモーダル理解で Gemini 1.5 Pro 比 -1.0 ポイント
- **MMLU-Pro 67.5**: Gemma 2 27B の 56.9 から +10.6 ポイントの飛躍

### マルチモーダル評価（Table 16、IT、P&S 適用）

| ベンチ | Gemma 3 4B | Gemma 3 12B | Gemma 3 27B |
| --- | --- | --- | --- |
| MMMU (val) | 48.8 | 59.6 | **64.9** |
| **DocVQA** | 75.8 | 87.1 | **86.6** |
| **InfoVQA** | 50.0 | 64.9 | **70.6** |
| TextVQA | 57.8 | 67.7 | 65.1 |
| AI2D | 74.8 | 84.2 | 84.5 |
| ChartQA | 68.8 | 75.7 | 78.0 |
| **MathVista** | 50.0 | 62.9 | **67.6** |

### PaliGemma 2 との比較（Table 12、転送学習）

| ベンチ | PaliGemma 2 27B | **Gemma 3 27B** |
| --- | --- | --- |
| DocVQA | 85.1 | **89.5**（+4.4） |
| InfoVQA | 50.2 | **64.6**（+14.4） |
| TextVQA | 75.1 | **83.2**（+8.1） |
| ChartQA | 71.3 | **83.4**（+12.1） |
| AI2D | 84.6 | **86.5**（+1.9） |

**Gemma 3 は文書理解で PaliGemma 2 を凌駕**、しかも **average pooling のため 4B/12B は同 896² 解像度の PaliGemma 2 9B/27B より約 10× 安価に転送可能**

### 長文脈（Table 15、IT モデル）

| ベンチ | Context | Gemma 3 27B IT |
| --- | --- | --- |
| RULER | 32K | **91.1** |
| RULER | 128K | 66.0 |
| MRCR | 32K | 63.2 |
| MRCR | 128K | 59.3 |

→ **32K では極めて高性能、128K への外挿で劣化**（Qwen3-VL の 1M YaRN 外挿には及ばず）

### 動画理解（Table 17、IT、16 frames linspace）

| ベンチ | Gemma 3 4B | Gemma 3 12B | Gemma 3 27B |
| --- | --- | --- | --- |
| Perception Test MCVQA | 50.6 | 54.9 | 58.1 |
| ActivityNet-QA | 46.3 | 50.4 | 52.8 |

**16 フレーム制約**のため、Qwen2-VL/3-VL の動的 FPS には届かない

### 記憶化（Memorization）

- **長文形式テキストの記憶化率**: 全前世代より **桁違いに低い**（log y 軸）
- **個人情報**: 記憶化として特徴付けられた出力に **個人情報は観察されず**（全モデル）
- 訓練データ除染と quality reweighting の効果

## 限界・批判的視点

- **固定 896² + P&S の保守性**: [[entities/qwen2-vl|Qwen2-VL]] の Naive Dynamic Resolution（2D-RoPE で ViT 改造）に比べると、**根源的解決ではなくタイル分割という実装的解決**。アスペクト比情報がタイル境界で失われる
- **256 トークン固定圧縮**: 文書のような情報密度の高い画像で **情報損失**が発生。InternVL 3.5 の **ViR（動的圧縮率）** とは対照的
- **128K 文脈での RULER 性能劣化**: 27B IT で 32K 91.1 → 128K 66.0（-25.1）。Qwen3-VL の 256K ネイティブ / 1M YaRN 99.5% に大きく劣る
- **MMMU で Gemini 1.5 Pro に -1.0、Gemini 2.0 Pro に -7.8**: 商用 Gemini フロンティアにはまだ追いつけていない
- **動画は 16 フレーム制約**: Qwen3-VL の 2048 frames / 32K video tokens、Qwen3.5-Omni の 400 秒 720P に大きく劣る
- **PaliGemma 2 との分業の不明確さ**: Google 内に複数の VL モデル系列が並存し、ユーザーの選択を困難に
- **訓練データ非公開**: Apache 2.0 風ライセンスだが、訓練データの詳細は公開されず（[[entities/internvl-3|InternVL 3]] の OpenGVLab/InternVL-Data 公開と対照的）
- **MoE 構造を採用していない**: 27B Dense で Elo 1338 を達成しているが、[[entities/internvl-3-5|InternVL 3.5]] の 241B-A28B MoE や [[entities/qwen3-vl|Qwen3-VL]] の 235B-A22B MoE に対して根本的にスケーラビリティ不足
- **CBRN 知識**: Google 自身が「Gemma 3 のこれら領域での知識は低い」と評価、極度に安全性重視の設計

## CV 分野における意義

Gemma 3 は **「Google の選んだ保守路線」**として、CV/MLLM 史において以下の意義を持つ：

1. **Google 系オープン MLLM の代表**: Gemini は商用閉源だが、Gemma 3 が **Google のオープン MLLM ブランド**を確立。**Qwen 系 / InternVL 系と並ぶ第 3 の主要系譜**
2. **タイル分割路線の最新形 + LLaVA 系継承**: [[questions/vit-dynamic-resolution-evolution]] で整理した「固定 vs 任意解像度」の対立軸において、**LLaVA → InternVL 1.5 → Gemma 3** のタイル分割路線を完成。**Pan & Scan は実装が簡単で効果も大きい**ことを実証
3. **KV キャッシュ最適化の新標準**: **5:1 local:global attention + sliding window 1024** は、長文脈 MLLM の標準的 KV キャッシュ削減技術となる可能性が高い
4. **軽量 + 量子化の実用化**: **QAT による int4 / sfp8 形式提供**で、**27B モデルを 14 GB（int4）で消費者 GPU に展開可能**。Qwen / InternVL 系の MoE 路線（数十 GB の活性パラメータ）と対照的な「物理的軽量化」アプローチ
5. **共有凍結 SigLIP 視覚エンコーダ**: 4B/12B/27B で同じ視覚エンコーダを共有 + 凍結という設計は、**訓練コスト削減と転送可能性**で実用的価値が高い
6. **Distillation 重視**: 14T トークンを **すべて知識蒸留**で学習する徹底ぶり。**Strong-to-Weak Distillation の本格的応用例**として、後の MLLM 設計に影響

Computer Vision wiki 内では：
- **[[entities/siglip|SigLIP]] の主要利用例**: 400M variant という具体的なサイズ選択
- **[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形**
- **[[entities/qwen3-vl|Qwen3-VL]] / [[entities/internvl-3-5|InternVL 3.5]] と対照的な「保守的軽量路線」**
- **[[concepts/foundation-model]] における Google 系の代表モデル**

## 用語と略称

- **MLLM / VLM** = Multimodal LLM / Vision-Language Model
- **Gemma 3** = Google DeepMind のオープン軽量 MLLM 系第 3 世代（2025 Mar）
- **Gemini 2.0** = Google の商用フロンティア LLM、Gemma 3 と co-design
- **PaliGemma 2** = Google の別系列オープン VLM、Gemma 3 と並列発展
- **SigLIP** = Sigmoid Loss Image-Language Pre-training（[[entities/siglip]]、Google DeepMind）
- **P&S (Pan & Scan)** = 推論時の適応的ウィンドウ・アルゴリズム、LLaVA 風タイル分割
- **5:1 local:global attention** = local 層 5 つに対して global 層 1 つを交互配置する構造
- **sliding window** = local attention の対象範囲を限定するウィンドウ（Gemma 3 は 1024 トークン）
- **QK-norm** = Query/Key の正規化、Gemma 2 の soft-capping を置換
- **soft-capping** = Gemma 2 で使用された attention 出力のクリッピング手法
- **RoPE (Rotary Position Embedding)** = [[concepts/rotary-position-embeddings|回転位置埋め込み]]
- **RoPE base 10k → 1M** = global 層の RoPE 基本周波数増加（長文脈対応）
- **GQA (Grouped-Query Attention)** = グループ化クエリ注意（複数 query head が同じ key/value head を共有）
- **QAT (Quantization Aware Training)** = 量子化を考慮した訓練、量子化形式を本番形式と一致させる
- **per-channel int4 / per-block int4 / SFP8** = 3 つの量子化形式
- **Knowledge Distillation** = 知識蒸留、教師モデルの確率分布を生徒が学習
- **256 logits sampling** = 蒸留時の各トークン位置でのロジット・サンプル数
- **ZeRO-3** = Zero Redundancy Optimizer Stage 3（オプティマイザ状態シャーディング）
- **Pathways** = Google の分散 ML フレームワーク
- **BOND / WARM / WARP** = Gemini で使用される RL 微調整手法
- **TPUv4 / v5e / v5p** = Google の Tensor Processing Unit 第 4-5 世代
- **LMSys Chatbot Arena** = 人間評価による LLM 比較プラットフォーム、Elo レーティング
- **RULER** = 長文脈ベンチマーク
- **MRCR** = Multi-document Reading Comprehension（長文脈ベンチマーク）
- **ECLeKTic** = 多言語ベンチマーク
- **CBRN** = Chemical/Biological/Radiological/Nuclear（化学・生物・放射性・核）
- **SDP (Sensitive Data Protection)** = Google Cloud のセンシティブデータ検出サービス
- **discoverable extraction** = 訓練データの記憶化を測定する手法（プレフィックス 50 + サフィックス 50）

## 関連ページ

- [[entities/gemma-3]] — Gemma 3 ファミリー（1B/4B/12B/27B）のエンティティ・ページ
- [[translations/gemma-3]] — 本論文の本文 + 付録翻訳
- [[entities/siglip]] — Gemma 3 が 400M variant で活用した視覚エンコーダ
- [[concepts/vision-transformer]] — SigLIP の基盤
- [[concepts/rotary-position-embeddings]] — RoPE 10k→1M スケーリング
- [[concepts/foundation-model]] — Google 系 MLLM の代表
- [[concepts/knowledge-distillation]] — 14T トークンの蒸留学習
- [[concepts/weakly-supervised-pretraining]] — 大規模事前学習
- [[questions/vit-dynamic-resolution-evolution]] — Pan & Scan = タイル分割路線の最新形
- [[entities/qwen3-vl]] — Qwen 系の対比（VL 純化 + 動的解像度 + MoE）
- [[entities/qwen2-vl]] — Naive Dynamic Resolution の対比（2D-RoPE で ViT 改造）
- [[entities/internvl-1-5]] — タイル分割路線の前任者
- [[entities/internvl-3-5]] — InternVL 系の対比（Cascade RL + MoE + ViR）
