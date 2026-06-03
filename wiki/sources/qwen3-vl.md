---
type: source
source_path: raw/papers/Qwen3-VL Technical Report.pdf
source_kind: paper
title: "Qwen3-VL Technical Report"
authors: [Qwen Team, Alibaba Group]
year: 2025
venue: arXiv:2511.21631
ingested: 2026-05-30
tags: [qwen, qwen3-vl, mllm, multimodal, vision-language, interleaved-mrope, deepstack, text-based-time-alignment, moe, 256k-context, sapo, sthinking-mode, alibaba]
translation: "[[translations/qwen3-vl]]"
---

# Qwen3-VL — Interleaved MRoPE + DeepStack + テキスト・タイムスタンプで「思考する VLM」と「MoE スケーリング」を本格化した Qwen 系第 4 世代

> 原典: [[translations/qwen3-vl]] ・ `raw/papers/Qwen3-VL Technical Report.pdf`
> 著者・年・会議: Qwen Team（Alibaba Group）, 2025 Nov 27（arXiv:2511.21631v2）, テクニカル・レポート

## 一言まとめ

[[entities/qwen2-5-vl|Qwen2.5-VL]] の **Window Attention + MRoPE absolute time** を発展させ、(i) **Interleaved MRoPE**（temporal/height/width を埋め込み次元に**交互配置**して周波数スペクトルを均衡化、長動画理解を強化）、(ii) **DeepStack**（ViT の中間 3 層の視覚特徴を LLM の最初 3 層に注入する多層融合）、(iii) **テキスト・ベース時間整合**（MRoPE absolute time をやめて `<3.0 seconds>` のような **明示的タイムスタンプ・トークン**に切り替え）、(iv) **6 モデル展開**（**dense: 2B/4B/8B/32B + MoE: 30B-A3B/235B-A22B**）、(v) **256K ネイティブ文脈窓**（YaRN で 1M トークン = 2 時間動画まで外挿）、(vi) **non-thinking と thinking の 2 バリアント**（後者は OpenAI o1 系の長 CoT 推論）、(vii) **多言語 OCR 10 言語 → 39 言語**、(viii) **平方根再重み付け**でテキストとマルチモーダルのバランス改善—という 8 つの主要進化を導入。フラッグシップ **Qwen3-VL-235B-A22B-Thinking** は MathVista 85.8 / MathVision 74.6 / OCRBench_v2 zh 63.5（GPT-5 37.7 +25.8）/ ODinW-13 48.6（[[entities/qwen2-5-vl|Qwen2.5-VL]] 43.1 +5.5）/ Charades-STA mIoU 63.5（Qwen2.5-VL 50.9 +12.6）/ V* with tools 93.7 / ScreenSpot Pro **62.0**（Qwen2.5-VL 43.6 +18.4）/ OSWorld 38.1（Qwen2.5-VL 8.83 = **4.3× 飛躍**）と多数の新 SOTA。さらに **Needle-in-a-Haystack で 256K 完全 100% / 1M で 99.5%** という長文脈耐性を実証。**Apache 2.0**。「視覚エージェント + 思考モデル + MoE」の 3 軸で Qwen ファミリーが完全に商用最先端と肩を並べた決定的瞬間。

## 背景と問題意識

[[entities/qwen2-5-vl|Qwen2.5-VL]]（2025 Feb）が **Window Attention + MRoPE absolute time + QwenVL HTML フォーマット**で「視覚エージェントの夜明け」を告げたのに続き、2025 年中盤〜後半は MLLM 業界に 4 つの大きな変化が起きた：

1. **推論モデルの台頭**: OpenAI o1（2024 Sept）、DeepSeek R1（2025 Jan）、Claude Sonnet 3.7 thinking、Gemini 2.5 Pro thinking、GPT-5 thinking などが次々と「長 CoT で考えるモデル」を打ち出し、MLLM にもこの流れが波及
2. **MoE スケーリングの本格化**: [[entities/internvl-3-5|InternVL 3.5]]（2025 Aug）が 241B-A28B MoE を導入、Qwen3 自体も MoE 系列を確立、Mixtral / DeepSeek-V3 系の MoE が業界標準に
3. **長文脈の必要性**: 1 時間超動画、数百ページ PDF、複数画像の交互配置などの実用シナリオで 256K 級文脈窓が必須に。[[entities/qwen2-5-vl|Qwen2.5-VL]] の 32K 文脈では不十分
4. **MRoPE absolute time の限界露呈**: Qwen2.5-VL の MRoPE absolute time は temporal ID を絶対秒数に揃えたが、**長動画では temporal ID が過度に大きく疎**になり、長時間文脈理解が劣化する問題が判明

これら 4 つの課題に応えるため、Qwen3-VL は **Qwen3 LLM ベース** + **3 構造革新（Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ）** + **6 サイズ展開（dense + MoE）** + **256K ネイティブ + thinking/non-thinking 2 バリアント** という総合パッケージで応えた。

## 提案手法 / 主張

### モデル展開：6 サイズ × 2 バリアント = 12 公開モデル

| サイズ | LLM | 種類 | 用途 |
| --- | --- | --- | --- |
| **Qwen3-VL-2B** | Qwen3 2B | Dense | エッジ AI |
| **Qwen3-VL-4B** | Qwen3 4B | Dense | エッジ AI |
| **Qwen3-VL-8B** | Qwen3 8B | Dense | バランス |
| **Qwen3-VL-32B** | Qwen3 32B | Dense | 中規模 |
| **Qwen3-VL-30B-A3B** | Qwen3 30B-A3B | **MoE**（活性 3B） | 効率志向 |
| **Qwen3-VL-235B-A22B** | Qwen3 235B-A22B | **MoE**（活性 22B） | フラッグシップ |

各サイズが **Instruct（non-thinking）と Thinking の 2 バリアント**を持つため、合計 **12 公開モデル**。

**Vision Encoder**: **SigLIP-2** ベース（[[entities/siglip]]）。フラッグシップ系（235B/32B/30B-A3B/8B）は **SigLIP2-SO-400M**、小型系（2B/4B）は **SigLIP2-Large（300M）**。Qwen2.5-VL は ViT をゼロから学習だったが、Qwen3-VL は **SigLIP-2 の事前学習済みチェックポイントから初期化して継続学習**するハイブリッド戦略。動的解像度は CoMP の方法論に基づく 2D-RoPE + 絶対位置補間で実現。

**Vision-Language Merger**: 2 層 MLP で **2×2 視覚特徴を 1 トークンに圧縮**（Qwen2.5-VL と同じ）。加えて **DeepStack 専用マージャ**を別途展開。

### 3 つの主要構造革新

#### 1. Interleaved MRoPE

**Qwen2-VL / Qwen2.5-VL の MRoPE の問題**:
- 埋め込み次元を **temporal (t) / height (h) / width (w) の「塊」に分割**して別々の回転周波数を割り当てる構造
- 結果として **周波数スペクトルが不均衡**になり、後続研究で長動画理解性能が劣化することが判明

**Qwen3-VL の改善**:
- t / h / w 成分を埋め込み次元にわたって **均一に交互配置**（interleaved）
- **各時空間軸が低周波帯と高周波帯の双方で均一に表現される**ことを保証
- スペクトル偏向を軽減し、動画の長距離位置モデリングを改善

#### 2. DeepStack

DeepStack 論文（Meng et al., 2024）に着想を得て、**視覚トークンを LLM の複数層に注入**する：

- **ViT の中間 3 層（低・中・高レベル特徴）から視覚トークンを抽出**
- 各層用の **専用 Vision-Language Merger** で投影
- **最初の 3 つの LLM 層の隠れ状態に直接加算**

これにより、追加文脈長を導入することなく **多層融合**を実現。InfoVQA や DocVQA のような細粒度視覚理解で +2-3 ポイントの改善（アブレーション表 12）。

#### 3. テキスト・ベース時間整合（最重要の方針転換）

**Qwen2.5-VL の MRoPE absolute time の 2 つの限界**:
1. **長動画で temporal ID が過度に大きく疎**になる → 長文脈理解を劣化
2. **多様な FPS で均一に分布したサンプリング**が必要で学習データ構築コストが膨大

**Qwen3-VL の解決策**:
- MRoPE absolute time を**捨て**、**明示的テキスト・タイムスタンプ・トークン**に置換
- 各動画時間パッチの前に `<3.0 seconds>` のようなフォーマットされたテキストを付加
- **秒形式 + HMS（時:分:秒）形式の双方**でタイムスタンプを生成、モデルが多様な時間表現を学ぶ

文脈長のわずかな増加と引き換えに、より精密で柔軟な時間グラウンディング能力を獲得。**Charades-STA mIoU 63.5（Qwen2.5-VL 50.9 から +12.6 ポイント）**の決定打。

### 4 段階事前学習（合計 ~2.2T トークン）

| Stage | 目的 | 学習対象 | トークン | Seq |
| --- | --- | --- | --- | --- |
| **S0** | Vision-Language Alignment | **Merger のみ** | 67B | 8,192 |
| **S1** | Multimodal Pre-Training | All | ~1T | 8,192 |
| **S2** | Long-Context Pre-Training | All | ~1T | **32,768** |
| **S3** | Ultra-Long-Context Adaptation | All | 100B | **262,144** |

[[entities/qwen2-5-vl|Qwen2.5-VL]]（4.1T、3 段階）と比べて、(i) **S0 でマージャのみを 67B トークン学習**するという「アライメント優先」フェーズを追加、(ii) S3 で seq 32K → **262K へ 8× 拡張**するという「Ultra-Long-Context Adaptation」フェーズを追加。これが **256K ネイティブ文脈窓**を実現。

### 事前学習データ：8 カテゴリ

1. **Image Caption + Interleaved Text-Image**: 微調整 Qwen2.5-VL-32B で再キャプション付け、書籍規模で 256K トークン系列にマージ
2. **Knowledge**: 12+ 意味カテゴリ（動物・植物・ランドマーク・食物・車両等）、ロングテール対応に importance-based sampling
3. **OCR / Document Parsing / Long Document**: **3000 万社内 OCR + 多言語 29 言語追加**、Qwen2.5-VL-72B で文書解析、QwenVL-HTML + QwenVL-Markdown の 2 表現
4. **Grounding + Counting**: COCO + Objects365 + OpenImages + RefCOCO/+/g、Grounding DINO で合成、**[0, 1000] 正規化座標系**（Qwen2.5-VL の絶対座標から復帰）
5. **Spatial Understanding + 3D Recognition**: 関係注釈・アフォーダンス・行動条件付きクエリ、**9-DoF 3D バウンディング・ボックス（Omni3D 形式）**で 3D グラウンディング
6. **Code**: Qwen3-Coder の text-only コード + Multimodal Coding（UI→HTML/CSS、画像→SVG、StackOverflow QA、フローチャート→コード）
7. **Video**: Dense Caption Synthesis + Spatio-Temporal Grounding、教育用 / シネマ / 自己中心的動画の Length-Adaptive Sampling
8. **STEM**: 100 万点グラウンディング + 200 万知覚 VQA + 600 万図解キャプション、6000 万 K-12 / 学部演習、1200 万マルチモーダル CoT
9. **Agent**: GUI（デスクトップ / モバイル / Web）+ Function Calling + Search の 3 系統

### 平方根再重み付け

サンプル単位損失からトークン単位の **平方根正規化損失**へ移行し、テキストとマルチモーダル・データの貢献をより良くバランス。マルチモーダル学習時の純粋テキスト能力劣化を防ぐ。

### 事後学習：3 段階パイプライン + Thinking with Images

| 段階 | 内容 |
| --- | --- |
| **SFT** | 2 フェーズ（**32K → 256K** 文脈拡張）、non-thinking 用標準フォーマット + thinking 用 CoT フォーマット、約 120 万サンプル |
| **Strong-to-Weak Distillation** | **テキスト専用データ**で LLM バックボーンを微調整、Off-policy → On-policy（KL 最小化） |
| **Reinforcement Learning** | **Reasoning RL + General RL**、SAPO（Smooth and Adaptive Policy-gradient）アルゴリズム |

#### Long-CoT Cold Start Data（thinking モデル用）

- 視覚言語と純粋テキストの **約 1:1 比率**
- **Multimodal Necessity Filtering**: Qwen3-30B-*nothink* モデルが視覚なしで解けるサンプルを破棄 → 真にマルチモーダル理解が必要な問題のみ残す
- Difficulty Curation + Response Quality Control の厳格フィルタリング

#### Reasoning RL

- 約 30K の RL クエリ、各 16 応答サンプリング、合格率 90% 超は除外
- **SAPO アルゴリズム**: テキスト・マルチモーダル・モデル規模・構造にわたって一貫した改善
- 報酬: フォーマット報酬なし、ルール・ベース統一、コード切り替えペナルティ

#### General RL

- Instruction Following + Preference Alignment の 2 次元
- SFT で染み込んだ欠陥のある知識事前分布を学習解除する是正機構
- ハイブリッド報酬: Rule-Based（フォーマット遵守等）+ Model-Based（Qwen2.5-VL-72B-Instruct or Qwen3 を判定者）

#### Thinking with Images（視覚エージェント）

- **2 段階パラダイム**: (1) Qwen2.5-VL-32B で 10K 例の SFT（think → act → analyze feedback → answer）+ 多ターン RL、(2) 120K 例にスケール、Qwen3-VL に蒸留
- **3 つの相補的報酬**: Answer Accuracy + Multi-Turn Reasoning + Tool-Calling（タスク複雑性に応じた適応的ツール探索）

### インフラ

- **Alibaba Cloud PAI-Lingjun**、最大 **10,000 GPU 規模**
- **Megatron-LM** + 5D 並列（TP + PP + CP + EP + ZeRO-1 DP）
- 推論: **vLLM**（PagedAttention）or **SGLang**（構造化生成）

## 実験結果と知見

### フラッグシップ Qwen3-VL-235B-A22B vs 商用（表 2）

| ベンチマーク | 235B-A22B Thinking | 235B-A22B Instruct | Gemini 2.5-Pro Thinking | GPT-5 high | Claude Opus 4.1 Thinking |
| --- | --- | --- | --- | --- | --- |
| MMMU | 80.6 | 78.7 | 81.7 | **84.2** | 78.4 |
| MMMU-Pro | 69.3 | 68.1 | 68.8 | **78.4** | 64.8 |
| **MathVista** | **85.8** | 84.9 | 82.7 | 81.3 | 75.5 |
| **MathVision** | **74.6** | 66.5 | 73.3 | 70.9 | 64.3 |
| MathVision_WP | **63.8** | 57.0 | 63.2 | 62.8 | 54.0 |
| We-Math | 74.8 | 67.5 | **80.6** | 73.8 | 65.2 |
| **MathVerse_mini** | **85.0** | 72.5 | 82.9 | 84.1 | 70.6 |
| DynaMath | 82.8 | 79.4 | 80.0 | **85.4** | 75.1 |
| Math-VR | 66.8 | 65.0 | 64.7 | 58.1 | 54.3 |
| **ZeroBench** | **4** | 2 | 3 | 2 | 3 |
| VlmsAreBlind | 79.5 | 80.4 | **86.1** | 80.5 | 77.8 |
| LogicVista | **72.2** | 65.8 | 72.0 | 68.7 | 67.3 |
| **VisuLogic** | **34.4** | 29.9 | 31.6 | 28.5 | 27.9 |
| **MMBench-EN** | 88.8 | **89.3** | 90.1 | 83.8 | 79.4 |
| **MMBench-CN** | 88.6 | **88.9** | **89.7** | 83.5 | 84.9 |
| **RealWorldQA** | 81.3 | 79.2 | 78.0 | **82.8** | 69.9 |
| MMStar | **78.7** | 78.4 | 77.5 | 76.4 | 72.1 |
| **HallusionBench** | **66.7** | 63.2 | 63.7 | 65.7 | 60.4 |
| **MIA-Bench** | **92.7** | 91.3 | 92.3 | 92.4 | 91.2 |
| **DocVQA test** | 96.5 | **97.1** | 92.6 | 91.5 | 92.5 |
| **InfoVQA test** | 89.5 | **89.2** | 84.2 | 79.0 | 69.4 |
| AI2D | 89.2 | **89.7** | 90.9 | 89.7 | 86.4 |
| ChartQA | 90.3 | 90.3 | 83.3 | 59.7 | 86.2 |
| **OCRBench** | 875 | **920** | 866 | 810 | 764 |
| **OCRBench_v2 en** | 66.8 | **67.1** | 54.3 | 50.4 | 48.4 |
| **OCRBench_v2 zh** | 63.5 | 61.8 | 48.5 | 37.7 | 43.7 |
| CC-OCR | 81.5 | **82.2** | 77.2 | 68.3 | 69.1 |
| **OmniDocBench en** ↓ | 0.155 | **0.143** | 0.206 | 0.356 | 0.194 |
| **OmniDocBench zh** ↓ | 0.207 | **0.007** | 0.238 | 0.472 | 0.293 |
| CharXiv DQ | 90.5 | **89.4** | **94.4** | 89.2 | 88.5 |
| CharXiv RQ | 66.1 | 62.1 | 67.9 | **81.1** | 63.6 |
| **MMLongBench-Doc** | 56.2 | **57.0** | 55.6 | 51.5 | 54.5 |
| RefCOCO-avg | **92.1** | 91.9 | 74.6 | 66.8 | - |
| **CountBench** | 93.7 | 93.0 | 91.0 | 91.7 | 93.1 |
| **ODinW-13** | 43.2 | **48.6** | 33.7 | - | - |
| **ERQA** | 52.5 | 51.3 | 55.3 | **65.7** | 34.8 |
| **VSI-Bench** | **60.0** | 62.7 | - | - | - |
| **EmbSpatialBench** | **84.3** | 83.1 | 79.1 | 82.9 | 69.2 |
| **RefSpatial** | **69.9** | 65.5 | 36.5 | 23.8 | - |
| **BLINK** | 67.1 | 70.7 | 70.6 | **71.0** | 64.1 |
| **MUIRBENCH** | **80.1** | 73.0 | 77.2 | 77.5 | 64.1 |
| MVBench | 75.2 | 76.5 | 69.9 | 73.5 | 61.4 |
| Video-MME w/o sub | 79.0 | 79.2 | **85.1** | 84.7 | 75.6 |
| MLVU M-Avg | 83.8 | 84.3 | 85.6 | **86.2** | 73.5 |
| **LVBench** | 63.6 | 67.7 | **73.0** | 69.0 | - |
| **Charades-STA mIoU** | 63.5 | **64.8** | - | - | - |
| **VideoMMMU** | 80.0 | 74.7 | 83.6 | **84.6** | 76.2 |
| **V*** | 85.9 | **93.7+**(tool) | 83.8 | 72.8 | 34.8 |
| **HRBench4K** | 84.3 | **85.4+**(tool) | 87.3 | - | - |
| **HRBench8K** | 76.6 | **82.4+**(tool) | 85.4 | - | - |
| **Design2Code** | **93.4** | 92.0 | 89.2 | 92.5 | 88.5 |
| **ChartMimic** | 78.4 | 80.5 | 83.9 | 62.1 | **85.2** |
| **ScreenSpot Pro** | **61.8** | 62.0 | - | - | - |
| **OSWorldG** | 68.3 | 66.7 | 45.2 | - | - |
| **AndroidWorld** | 62.0 | 63.7 | - | - | - |
| **OSWorld** | **38.1** | 31.6 | - | - | 44.4 |
| **WindowsAA** | **32.1** | 28.9 | - | - | - |

**ハイライト**:
- **MathVista 85.8 / MathVision 74.6** で数学推論 SOTA、GPT-5 high を上回る
- **OCRBench_v2 zh 63.5** で GPT-5 を **+25.8 ポイント圧倒**
- **OmniDocBench で SOTA**（en 0.143 / zh 0.207 ↓ いずれも最低エラー）
- **VSI-Bench 60.0 / ScreenSpot Pro 62.0 / OSWorld 38.1 / OSWorldG 68.3**: 空間推論と GUI エージェントで例外的な強さ
- **MUIRBENCH 80.1** で複数画像理解 SOTA、Gemini 2.5-Pro 77.2 を超え
- **V* with tools 93.7+** で細粒度知覚 SOTA（GPT-5 72.8 を +20.9 圧倒）
- **AndroidWorld 63.7** で Qwen2.5-VL の 35% から **+28.7 ポイント飛躍**
- **OSWorld 38.1** で Qwen2.5-VL の 8.83 から **4.3× 飛躍**、Claude Opus 4 (44.4) に肉薄

### 純粋テキスト（表 5、Instruct 系）

| ベンチマーク | Qwen3-VL-235B-A22B Instruct | Qwen3-235B-A22B Instruct-2507（純粋 LLM） |
| --- | --- | --- |
| MMLU-Pro | 81.8 | **83.0** |
| MMLU-Redux | 92.2 | **93.1** |
| GPQA | 74.3 | **77.5** |
| **AIME-25** | **74.7** | 70.3 |
| **HMMT-25** | **57.4** | 55.4 |
| **LiveCodeBench v6** | **54.3** | 51.8 |
| BFCL-v3 | 67.7 | **70.9** |
| MultiIF | 76.3 | **77.5** |
| **MMLU-ProX** | 77.8 | **79.4** |
| **PolyMATH** | 45.1 | **50.2** |

**マルチモーダル化で言語が強くなる現象**を継続実証: AIME-25 +4.4、HMMT-25 +2.0、LiveCodeBench v6 +2.5。InternVL 3 / 3.5 の発見が Qwen 系でも再確認。

### Needle-in-a-Haystack（図 3）

- **30 分動画（256K トークン）まで完全 100% 精度**
- **YaRN ベース位置拡張で 1M トークン（約 2 時間動画）まで 99.5% 精度**を維持
- 視覚 needle 検索の 80 分 / 60% 深さあたりで僅かに劣化が見えるが、全体的に堅牢

### アブレーション

#### Vision Encoder（表 11）

| | OmniBench (CLIP) | OCRB (VLM) | InfoVQA (VLM) | OmniBench (VLM) |
| --- | --- | --- | --- | --- |
| SigLIP-2 | 36.9 | 77.2 | 65.3 | 50.1 |
| **Qwen3-ViT**（SigLIP-2 継続学習版） | **45.5** | **78.7** | **67.0** | **53.0** |

**SigLIP-2 を継続学習した Qwen3-ViT が OmniBench で +8.6 / +2.9 ポイント上回る**。継続学習戦略の有効性を実証。

#### DeepStack（表 12、15B-A2B 内部 LLM で 200B トークン事前学習）

| | AVG | OCRB | InfoVQA | DocVQA | MMMU |
| --- | --- | --- | --- | --- | --- |
| Baseline | 74.7 | 81.0 | 71.9 | 89.5 | 52.9 |
| **DeepStack** | **76.0** | **83.6** | **74.2** | **91.1** | **54.1** |

**全ベンチで一貫した改善（平均 +1.3 ポイント）**、特に **InfoVQA +2.3 / DocVQA +1.6 / OCRB +2.6 で細粒度視覚理解で大きな効果**。

### Qwen3-VL-30B-A3B（中規模 MoE）vs Gemini 2.5 Flash / GPT-5 mini

中規模 dense Qwen3-VL-32B と中規模 MoE Qwen3-VL-30B-A3B の双方が、表 3 でほとんどの評価指標で Gemini 2.5 Flash と GPT-5 mini を一貫して凌駕。MoE 効率モデルが中規模商用フロンティアと完全に互角になった証拠。

### 小型モデル（8B/4B/2B、表 4 と 9/10）

- **Qwen3-VL-8B** は GPT-5-Nano と多くのベンチで競合
- **Qwen3-VL-4B-Thinking** は DynaMath / VisuLogic で最高スコア
- **Qwen3-VL-2B** ですら強力な推論能力を示し、Qwen3-VL-8B から段階的にスケールアップが可能

## 限界・批判的視点

- **MMMU で GPT-5 high (84.2) に -3.6 ポイント**（235B-A22B Thinking 80.6）。最難関の大学レベル推論ではまだ GPT-5 に届かない
- **We-Math で Gemini 2.5-Pro Thinking (80.6) に -5.8**。マルチホップ数学推論で Gemini に後れ
- **Video-MME / MLVU で Gemini 2.5-Pro と GPT-5 にやや劣る**（85.1 / 86.2 vs 79.0 / 84.3）
- **OSWorld で Claude Opus 4 (44.4) に -6.3**（38.1）。デスクトップ・エージェントで Claude が依然優位（[[entities/qwen2-5-vl|Qwen2.5-VL]] と同じ傾向）
- **CharXiv 推論サブセットで GPT-5 (81.1) と Gemini 2.5-Pro (67.9) に後れ**（66.1 thinking）
- **訓練データ非公開**（cf. [[entities/internvl-3|InternVL 3]] は完全公開）。オープン・サイエンスの観点で InternVL 系に劣後
- **6 サイズ × 2 バリアント = 12 モデルの管理コスト**: 適切なバリアント選択が現場で複雑化
- **MoE 235B モデルの推論コスト**: 活性 22B でも複数 GPU 必須、消費者環境では実行困難
- **Long-Context Pre-Training Stage 2 で 32K のまま 1T トークン**は意外と小さい（cf. Qwen2-VL の Stage2 が 800B）。ultra-long-context は S3 の 100B のみ
- **ViT を SigLIP-2 から継続学習**: [[entities/qwen2-5-vl|Qwen2.5-VL]] のゼロから学習路線を放棄、戦略の揺れ動きが見える
- **3D 物体理解（Omni3D）の閾値 IoU 0.15** は実用上の信頼性に疑問。閾値 0.5 での結果は未公開

## CV 分野における意義

Qwen3-VL は **「Qwen ファミリーが商用最先端と完全に肩を並べた決定的瞬間」** として位置付けられる。これは、(i) **Interleaved MRoPE** で動画の長距離位置モデリングを業界標準化、(ii) **DeepStack** で多層 ViT 特徴の LLM 注入を MLLM 設計の選択肢に加え、(iii) **テキスト・ベース時間整合**で「位置エンコーディングで時間を表現する」という MRoPE 路線（[[entities/qwen2-vl|Qwen2-VL]] / [[entities/qwen2-5-vl|Qwen2.5-VL]]）から「テキスト・トークンで明示的に時間を表現する」というよりシンプルなパラダイムへ転換、(iv) **6 サイズ × dense + MoE × thinking/non-thinking** の組み合わせで「実用 MLLM スイート」の標準を確立、(v) **256K ネイティブ + 1M 外挿**で長文脈 MLLM の新ベンチマークを設定、(vi) **ScreenSpot Pro 62.0 / OSWorld 38.1 / AndroidWorld 63.7** で GUI エージェントを商用最先端と互角に押し上げた、という 6 つの軸で MLLM 史に新たな節目を刻んだ。

[[entities/qwen-vl|Qwen-VL]] が `<box>`/`<ref>` で位置語彙を不要にし、[[entities/qwen2-vl|Qwen2-VL]] が Naive Dynamic Resolution + M-RoPE を業界標準にし、[[entities/qwen2-5-vl|Qwen2.5-VL]] が Window Attention + MRoPE absolute time + QwenVL HTML format で視覚エージェントを商用級に引き上げたのに続き、Qwen3-VL は **Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE スケーリング + thinking モード** で「商用フロンティアと完全に互角になった汎用 MLLM 基盤」を実現。InternVL シリーズ（[[entities/internvl-3|InternVL 3]] / [[entities/internvl-3-5|InternVL 3.5]]）と並ぶ現代オープンソース MLLM の双璧として、embodied AI の基盤エンジンを目指す。

## 用語と略称

- **LVLM / VLM / MLLM** = Large Vision-Language Model / Vision-Language Model / Multimodal LLM
- **2B / 4B / 8B / 32B / 30B-A3B / 235B-A22B** = 20 億 / 40 億 / 80 億 / 320 億 / 30B 活性 3B MoE / 235B 活性 22B MoE
- **Interleaved MRoPE** = temporal/height/width を埋め込み次元に交互配置する MRoPE の改良版（[[entities/qwen2-vl|Qwen2-VL]] の塊状分割を改善）
- **DeepStack** = ViT の中間 3 層の視覚特徴を LLM の最初 3 層に注入する多層融合機構（Meng et al., 2024）
- **テキスト・ベース時間整合** = MRoPE absolute time の代わりに `<3.0 seconds>` のような明示的タイムスタンプ・トークンを使う方式
- **MRoPE / M-RoPE** = Multimodal Rotary Position Embedding（[[concepts/rotary-position-embeddings]]）
- **SigLIP-2** = Google の改良 CLIP（[[entities/siglip]]）。Qwen3-VL の ViT 初期化に使用
- **SigLIP2-SO-400M / SigLIP2-Large（300M）** = SigLIP-2 の 2 サイズ
- **CoMP** = Continual Multimodal Pre-training for Vision Foundation Models（Chen et al., 2025）
- **2D-RoPE** = 2 次元回転位置埋め込み（[[concepts/rotary-position-embeddings]]）
- **256K context window** = ネイティブ・サポートする最大入力長
- **YaRN** = 長文脈外挿手法、Qwen3-VL では 1M トークン（2 時間動画）まで外挿
- **Thinking モード** = 長 CoT 推論を生成するバリアント（OpenAI o1 / DeepSeek R1 系）
- **Non-thinking モード（Instruct）** = 通常の応答生成バリアント
- **MoE** = Mixture-of-Experts（混合エキスパート）、入力ごとに専門家を切り替える疎モデル
- **A3B / A22B** = Activated 3B / 22B（MoE で各トークン処理時の活性パラメータ数）
- **Strong-to-Weak Distillation** = 強力な教師モデルから軽量生徒モデルへの知識蒸留
- **SAPO** = Smooth and Adaptive Policy-gradient method（Gao et al., 2025）、Qwen3-VL の RL アルゴリズム
- **Reasoning RL / General RL** = 推論能力向上の RL / 汎化能力向上の RL
- **Thinking with Images** = 視覚エージェント能力（think → act → analyze feedback → answer）
- **Long-CoT Cold Start** = 長 Chain-of-Thought データでの SFT 開始
- **Multimodal Necessity Filtering** = 視覚なしで解けるサンプルを除外するフィルタ（thinking データ整備）
- **QwenVL-HTML / QwenVL-Markdown** = Qwen3-VL の文書解析用 2 表現（前者は要素レベル bbox、後者は表を LaTeX で符号化）
- **9-DoF 3D bounding box** = 9 自由度（位置 3 + 回転 3 + サイズ 3）の 3D バウンディング・ボックス
- **Omni3D** = 統一 3D 物体検出ベンチマーク・形式（Brazil et al., 2023）
- **PixMo** = Molmo（Allen AI）が公開したポインティング・カウント・データセット
- **Megatron-LM** = NVIDIA の大規模 LLM 学習フレームワーク
- **TP / PP / CP / EP / ZeRO-1 DP** = Tensor / Pipeline / Context / Expert Parallelism + ZeRO-1 Data Parallelism
- **vLLM / SGLang** = LLM 推論サーバ
- **PagedAttention** = vLLM のメモリ管理機構
- **平方根再重み付け** = テキストとマルチモーダル・データの貢献を平方根でバランスする損失正規化
- **MMMU / MMMU-Pro / MathVista / MathVision / MathVerse / DynaMath / We-Math / Math-VR / LogicVista / VisuLogic / ZeroBench / VisualPuzzles / VLMsAreBlind** = 大学レベル・数学・論理推論ベンチマーク群
- **MMBench / MMStar / SimpleVQA / RealWorldQA / MM-MT-Bench** = 汎用 VQA ベンチマーク群
- **HallusionBench / MIA-Bench** = 幻覚 / 命令追従ベンチマーク
- **DocVQA / InfoVQA / AI2D / ChartQA / OCRBench / OCRBench_v2 / CC-OCR / OmniDocBench / CharXiv / MMLongBench-Doc** = 文書・OCR ベンチマーク群
- **RefCOCO / CountBench / ODinW** = グラウンディング・カウント・オープン語彙検出ベンチマーク
- **ARKitScenes / Hypersim / SUN RGB-D / ERQA / VSI-Bench / EmbSpatialBench / RefSpatial / RoboSpatialHome** = 3D・embodied 空間理解ベンチマーク
- **BLINK / MUIRBENCH** = 複数画像理解ベンチマーク
- **MVBench / Video-MME / MLVU / LVBench / Charades-STA / VideoMMMU / MMVU / TempCompass / EgoSchema / PerceptionTest** = 動画理解ベンチマーク
- **V\* / HRBench-4k / HRBench-8k** = 細粒度知覚ベンチマーク
- **Design2Code / ChartMimic / UniSVG** = マルチモーダル・コーディング・ベンチマーク
- **ScreenSpot / ScreenSpot Pro / OSWorldG / AndroidWorld / MobileMiniWob++ / OSWorld / WindowsAA** = GUI エージェント・ベンチマーク

## 関連ページ

- [[entities/qwen3-vl]] — Qwen3-VL 6 サイズ × 2 バリアント = 12 モデルのエンティティ・ページ
- [[translations/qwen3-vl]] — 本論文の本文翻訳
- [[entities/qwen2-5-vl]] — 前世代（Window Attention + MRoPE absolute time + QwenVL HTML format）
- [[entities/qwen2-vl]] — 第 2 世代（Naive Dynamic Resolution + M-RoPE）
- [[entities/qwen-vl]] — 初代（`<box>`/`<ref>` 特殊トークン）
- [[entities/siglip]] — Vision Encoder の初期化に使用
- [[concepts/rotary-position-embeddings]] — Interleaved MRoPE の前提
- [[concepts/foundation-model]] — 視覚言語基盤モデル
- [[concepts/weakly-supervised-pretraining]] — 4 段階事前学習
- [[concepts/zero-shot-transfer]] — 多言語 OCR、オープン語彙検出、3D グラウンディング、GUI エージェントでのゼロショット転移
- [[concepts/alignment-tuning]] — SFT + Strong-to-Weak Distillation + RL
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系
- [[entities/internvl-3-5]] — Cascade RL + MoE の InternVL 系（同時期の対抗系譜）
- [[entities/grounding-dino]] — オープン語彙検出専門モデル（Qwen3-VL のグラウンディング合成に活用）
