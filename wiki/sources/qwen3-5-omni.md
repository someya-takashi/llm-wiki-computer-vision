---
type: source
source_path: raw/papers/Qwen3.5-Omni Technical Report.md
source_kind: paper
title: "Qwen3.5-Omni Technical Report"
authors: [Qwen Team, Alibaba Group]
year: 2026
venue: arXiv:2604.15804（テクニカル・レポート）
ingested: 2026-05-30
tags: [qwen, qwen3-5-omni, omnimodal, mllm, thinker-talker, hybrid-moe, aria, aut, rvq, audio-visual, speech-generation, audio-visual-vibe-coding, alibaba]
translation: "[[translations/qwen3-5-omni]]"
---

# Qwen3.5-Omni — Hybrid MoE Thinker-Talker + AuT + ARIA で「真の omnimodal エージェント」を実現した Qwen-Omni シリーズ最新版

> 原典: [[translations/qwen3-5-omni]] ・ `raw/papers/Qwen3.5-Omni Technical Report.md`
> 著者・年・会議: Qwen Team（Alibaba Group）, 2026, arXiv:2604.15804（テクニカル・レポート）

## 一言まとめ

Qwen ファミリーの **Qwen-Omni 系譜**（Qwen2.5-Omni → Qwen3-Omni → **Qwen3.5-Omni**）の最新版。**Thinker-Talker 構造**を継承しつつ、(i) **Thinker と Talker の双方が Hybrid Attention MoE**（Qwen3.5 ベース、Gated Delta Net 採用）、(ii) **256K ネイティブ文脈**（10 時間音声 / 400 秒の 720P 動画 / 1 億時間音声-視覚データで事前学習）、(iii) **AuT (Audio Transformer)** をゼロから 4000 万時間音声で学習（6.25Hz トークンレート、20+ 言語）、(iv) **マルチコードブック・コーデック表現**による単一フレーム即時音声合成、(v) **ARIA (Adaptive Rate Interleave Alignment)** で MFA や固定交互配置レートを置換しテキスト-音声整合を動的化、(vi) 多言語 **音声入力 113 言語+方言 / 音声出力 36 / テキスト 201**、(vii) **Audio-Visual Vibe Coding**（音声-視覚命令から直接実行可能コードを生成する出現能力）—という 7 つの主要進化を実現。**Plus と Flash の 2 バリアント**で、**215 音声・音声-視覚サブタスクで SOTA**、主要音声タスクで Gemini-3.1 Pro を凌駕、AV 理解で同等。**ARIA + Hybrid MoE で first-packet 遅延 235ms（Flash, 音声入力）/ 426ms（動画入力）** という低遅延を実現し、リアルタイム omnimodal 対話の実用域に到達。**Computer Vision wiki 内では、視覚モダリティを音声・テキストと統合する omnimodal モデル**として、Qwen3-VL（純粋 VL）と並列する Qwen ファミリーの 2 つの主軸の片翼に位置付けられる。

## 背景と問題意識

**[[entities/qwen3-vl|Qwen3-VL]]（2025 Nov）が「VL モデルとして商用最先端と並ぶ」を達成**したのに対し、Qwen ファミリーには **音声・音声-視覚モダリティを統合した Qwen-Omni 系列**が並行して存在する：

- **Qwen2.5-Omni**（2025 初頭）: Thinker-Talker 構造を初提唱、TM-RoPE で音声-動画同期
- **Qwen3-Omni-30B-A3B**（2025 中頃）: MoE スケーリング初導入、RVQ ベース音声表現、dual-track Talker
- **Qwen3.5-Omni**（2026）: 本ページ、Hybrid MoE + 256K + ARIA で次世代 omnimodal を実現

2025 年後半時点で、GPT-4o（OpenAI）と Gemini-3.1 Pro（Google）の 2 系列が omnimodal 商用フロンティアを形成。Qwen-Omni 系列はこれら商用モデルに **オープンソースで肩を並べる**ことを目指す。**Qwen3.5-Omni の主要課題**は：

1. **長文脈の omnimodal 処理**: 10 時間音声、400 秒 720P 動画を扱う 256K 文脈窓
2. **超低遅延ストリーミング音声合成**: テキストと音声のトークン化レート不一致による不安定性（単語スキップ、誤発音、数字レンダリング曖昧化）を ARIA で解消
3. **音声・音声-視覚タスクでの商用追従**: 215 サブタスクで Gemini-3.1 Pro と並走または凌駕
4. **テキスト・視覚能力の劣化なし**: 同サイズ単一モデル Qwen と同等の純粋テキスト・視覚性能を維持
5. **エージェント的行動のネイティブ化**: WebSearch / FunctionCall / 音声生成 / Audio-Visual Vibe Coding（音声-視覚命令からの直接コード生成）

これらに応えるため、Qwen3.5-Omni は **Hybrid MoE + AuT + ARIA + 256K** の総合パッケージで設計された。

## 提案手法 / 主張

### Thinker-Talker 構造の継承と進化

| 構成要素 | アーキテクチャ | ストリーミング |
| --- | --- | --- |
| **Audio Encoder** | **AuT**（ゼロから学習、6.25Hz、4000 万時間） | ✓ |
| **Vision Encoder** | **SigLIP2**（[[entities/siglip]] からの拡張、Qwen3.5 共有） | – |
| **Thinker** | **Hybrid MoE Transformer**（Qwen3.5 ベース） | ✓ |
| **Talker** | **Hybrid MoE Transformer** | ✓ |
| **MTP**（Multi-Token Prediction） | Dense Transformer | ✓ |
| **Code2Wav** | 因果的 ConvNet | ✓ |

**Thinker**: テキスト生成 + 高レベル表現を Talker に伝達
**Talker**: マルチコードブック RVQ トークン列を自己回帰的に予測 → MTP モジュールが現フレーム残差コードブックを出力 → Code2Wav が波形を段階的に合成（**フレーム単位ストリーミング生成**）

### 7 つの主要進化

#### 1. Hybrid MoE（Thinker と Talker 双方）

- **Qwen3.5 で導入された Hybrid MoE 構造**を Thinker と Talker の双方が採用
- **Gated Delta Net (GDN) モジュール**を含み、長音声-動画系列モデリングを加速
- **長文脈推論で KV キャッシュ I/O オーバーヘッド削減**、サービング並行性向上

#### 2. AuT (Audio Transformer)

- attention-encoder-decoder モデル
- **Qwen3-ASR が生成した 4000 万時間音声-テキスト対**でゼロから学習
- フィルタバンク特徴量を **4 つの Conv2D ブロックで 16× ダウンサンプリング**
- 自己注意層で **6.25Hz トークンレート**（フレームあたり 160ms）
- 20+ 言語の多言語データ採用、**中英多言語比率 3.5:3.5:3**
- **動的注意ウィンドウサイズ学習**でリアルタイム prefill キャッシングとオフライン理解のバランス

#### 3. 256K ネイティブ文脈

- **10 時間超の音声理解**
- **400 秒超の 720P 動画**（1 FPS）
- 4000 万時間音声 + **1 億時間音声-視覚コンテンツ**で事前学習

#### 4. マルチコードブック・コーデック表現

- 単一フレーム即時音声合成
- **RVQ (Residual Vector Quantization)** ベース、Qwen3-Omni 由来
- MTP モジュールで残差コードブックの細粒度モデリング

#### 5. ARIA (Adaptive Rate Interleave Alignment)

**従来のテキスト-音声整合の問題**:
- MFA (Montreal Forced Aligner) 由来の整合や固定交互配置レート → テキストと音声のトークン化レート不一致で不安定
- 単語スキップ、誤発音、数字の曖昧なレンダリング

**ARIA の解決**:
- Qwen3-Omni の dual-channel 生成を **統一交互配置単一ストリーム定式化**に再構成
- **適応的レート制約**: 生成系列の任意の prefix で、累積音声-テキスト・トークン比率は対応する項目レベル・グローバル比率を超えてはならない
- 低符号化効率言語にも柔軟、任意のテキスト prefix に続く首尾一貫した音声継続をサポート

#### 6. 多言語サポート大幅拡大

| Modality | # Varieties | 主要言語 |
| --- | --- | --- |
| **Text** | **201** | Qwen3.5 互換 |
| **Speech Input** | **113** | 74 言語 + **39 中国方言**（東北マンダリン、上海語、四川語、客家語、閩南語、等） |
| **Speech Output** | **36** | 29 言語 + **7 中国方言** |

#### 7. Audio-Visual Vibe Coding（出現能力）

- 音声-視覚命令から **直接実行可能コードを生成**
- omnimodal モデルにおける **新規出現能力**
- 外部オーケストレーションなしにリアルタイム・クエリへ応答

### Audio-Visual Timestamp 設計

[[entities/qwen2-5-vl|Qwen2.5-VL]] の TM-RoPE → [[entities/qwen3-vl|Qwen3-VL]] の **テキスト・ベース時間整合**の系譜を継承：

- 各動画/音声-動画時間パッチの前に **秒単位フォーマット文字列のタイムスタンプ**を付加（例: `<3.0 seconds>`）
- 音声系列にもランダム間隔でタイムスタンプ挿入
- 音声成分: **160ms ごとの時間 ID**
- 動画: **ID あたり 160ms の一貫した時間解像度**
- 位置番号は連続的、モダリティ間で衝突なし

### 学習：3 段階事前学習 + 後学習

#### 事前学習（3 段階）

| Stage | 目的 | 学習対象 | 系列長 |
| --- | --- | --- | --- |
| **S1** Encoder Alignment | アダプタ + エンコーダ整合 | Vision / Audio Encoder（LLM 凍結） | - |
| **S2** General | **~4T トークン**（テキスト 0.92T + 音声 1.99T + 画像 0.95T + 動画 0.14T + 動画-音声 0.29T） | All | **32,768** |
| **S3** Long Context | 長音声・長動画比率増加 | All | **262,144** |

#### Thinker 事後学習（3 段階）

1. **Specialist Distillation**: ドメイン特化教師モデル（テキスト・視覚・音声）→ SFT + RL で蒸留
2. **On-Policy Distillation (OPD)**: テキスト条件付き応答を音声条件付きクエリの蒸留ターゲットに → モダリティ一貫した生成
3. **Interaction-Aligned RL**: 複数ターン相互作用軌跡で、コード切り替え・ペルソナ不整合・命令追従劣化を緩和

#### Talker 事後学習（4 段階）

1. **General Stage**: 2000 万時間超の多言語音声 + マルチモーダル文脈
2. **Long-Context Stage**: 高品質サブセットで CPT、Qwen3-Omni-Captioner で増強、**64k トークンへ拡張**
3. **RL Stage**: **DPO** で多言語選好対 + **GSPO** でルールベース報酬統合
4. **Speaker Fine-tuning Stage**: 軽量話者微調整で話者特性を忠実に捕捉

### ストリーミング・並行性

| 指標 | Plus | Flash |
| --- | --- | --- |
| **First-Packet Latency (音声入力)** | 435ms | **235ms** |
| **First-Packet Latency (動画入力)** | 651ms | **426ms** |

- **Chunked Prefilling**: 音声・視覚エンコーダが時間次元に沿ってチャンク出力 → Thinker/Talker の TTFT を著しく削減
- **vLLM + torch.compile + CUDA Graph**（MTP モジュールとコーデック・デコーダ用）

## 実験結果と知見

### 音声 → テキスト（表 5）vs Gemini-3.1 Pro

| カテゴリ | ベンチマーク | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
| --- | --- | --- | --- |
| **Audio Understanding** | MMAU | 81.1 | **82.2** |
| | MMSU | 81.3 | **82.8** |
| | RUL-MuchoMusic | 59.6 | **72.4**（+12.8）|
| **Dialogue** | VoiceBench | 88.9 | **93.1**（+4.2）|
| **S2TT** | Fleurs xx↔zh/en | 32.1 | **32.8** |
| **ASR** ↓ | Fleurs (top60) | 7.32 | **6.55** |
| | Librispeech clean | 3.36 | **1.11** |
| | Kespeech（方言）| 23.67 | **3.46**（**6.8× 改善**）|

**Gemini-3.1 Pro を音声理解・対話・ASR の主要タスクで凌駕**、特に音楽理解（RUL-MuchoMusic）と中国方言（Kespeech 6.8× 改善）で劇的差。

### 視覚 → テキスト（表 6）vs Qwen3.5-Plus-Instruct（純粋テキスト VL）

| カテゴリ | ベンチマーク | Plus-Instruct | Qwen3.5-Omni-Plus |
| --- | --- | --- | --- |
| MMMU | 81.0 | 80.1 |
| MathVision | 73.6 | 73.0 |
| **RealWorldQA** | 79.1 | **84.1**（+5.0）|
| **VideoMME w/o sub** | 81.0 | **81.9** |
| **MLVU M-Avg** | 85.1 | **86.8** |
| **MVBench** | 76.7 | **79.0** |
| **LVBench** | 68.6 | **71.2** |
| **MMVU** | 67.1 | **67.5** |
| **MME-VideoOCR** | 74.2 | **77.0** |
| **SLAKE**（医療 VQA） | 82.8 | **84.7** |
| **EmbSpatialBench** | 83.4 | **85.4** |

**動画理解で 6/6 ベンチで純粋 VL を上回る** → 共同 video-audio 学習パラダイムの効果実証。**RealWorldQA で +5.0** は実世界知覚での音声付き学習の優位性を示唆。

### 音声-視覚 → テキスト（表 7）vs Gemini-3.1 Pro

| ベンチマーク | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
| --- | --- | --- |
| **DailyOmni** | 82.7 | **84.6**（SOTA）|
| WorldSense | **65.5** | 62.8 |
| AVUT | **85.6** | 85.0 |
| AV-SpeakerBench | **75.1** | 71.3 |
| VideoMME w/ audio | **89.0** | 83.7 |
| **Qualcomm IVD** | 66.2 | **68.5**（+2.3、実世界 AV 対話で凌駕）|
| **Omni-Cloze**（キャプション） | 57.2 | **64.8**（+7.6）|
| OmniGAIA（ツール使用） | **68.9** | 57.2 |

**実世界音声-視覚対話（Qualcomm IVD）と AV キャプショニングで凌駕**、一般理解で同等、ツール使用で劣後（改善余地）。

### テキスト → テキスト（表 4）vs Qwen3.5-Plus-Instruct（純粋 LLM）

Qwen3.5-Omni-Plus の純粋テキスト性能は **同サイズ Qwen3.5-Plus-Instruct とほぼ同等**（MMLU-Pro 85.9 vs 86.8、IFEval 89.7 vs 89.7、HMMT Nov 25 84.4 vs 86.2）。**IFBench では 52.6 vs 51.1 で純粋 LLM を上回る** → **OPD + Interaction-Aligned RL が命令追従に正の効果**。

### ゼロショット音声生成（表 8、SEED-TTS、WER ↓）

| Model | test-zh | test-en |
| --- | --- | --- |
| CosyVoice 3 | **0.71** | 1.45 |
| MiniMax-Speech | 0.83 | 1.65 |
| Qwen3-Omni-30B-A3B | 1.07 | 1.39 |
| **Qwen3.5-Omni-Plus** | 0.99 | **1.26 SOTA** |

### 多言語音声生成（表 9-12）

- **29 言語中 22 で最低 WER**（vs MiniMax-Speech, ElevenLabs）
- **クロス言語音声生成 12 方向中 10 で SOTA**（vs Qwen3-Omni, CosyVoice）
- **zh-to-ko で CosyVoice3 から 72% 相対削減**（14.4 → 4.03）
- カスタム音声では **29 言語中 10 で最低 WER** vs ElevenLabs / Gemini-2.5 Pro / GPT-Audio / MiniMax

### 多言語 ASR（表 13）- 60 言語平均

| Model | 平均 WER (60 言語) |
| --- | --- |
| **Qwen3.5-Omni-Plus** | **6.6** |
| Gemini-3.1 Pro | 7.3 |
| GPT-4o-Transcribe | 10.4 |
| Qwen3.5-Omni-Flash | 10.8 |
| Gemini-3-Flash | 10.5 |

特に **広東語 2.2%（Gemini 6.3%）、タイ語、ベトナム語、日本語、韓国語**で著しい優位。

### 多言語翻訳（表 14-15、BLEU ↑）

| 方向 | Qwen3.5-Omni-Plus | Gemini-3.1 Pro |
| --- | --- | --- |
| en2xx 平均 | **33.8** | 31.8 |
| zh2xx 平均 | **21.4** | 19.6 |
| xx2en 平均 | 37.0 | **37.4** |
| xx2zh 平均 | 38.9 | **39.4** |

- en2xx/zh2xx で Gemini-3.1 Pro を凌駕
- **広東語で +15.6 BLEU**（en2xx）と圧倒的優位
- Asian languages（韓・日・広東語）で明確な強さ

## 限界・批判的視点

- **OmniGAIA（ツール使用）で Gemini-3.1 Pro 68.9 に対し 57.2** と大きく後れる、エージェント・ツール統合は今後の課題
- **VideoMME w/ audio で Gemini 89.0 に対し 83.7**、AV-SpeakerBench で 75.1 vs 71.3 → 一部 AV ベンチで商用に劣後
- **WorldSense で Gemini 65.5 vs 62.8** → 世界知識統合に改善余地
- **テキスト・タスクで純粋 Qwen3.5-Plus-Instruct より劣る**（MMLU-Pro 86.8 vs 85.9、SuperGPQA 67.4 vs 66.4 等）→ omnimodal 統合のトレードオフ
- **Plus と Flash 2 バリアントのみ** → [[entities/qwen3-vl|Qwen3-VL]] の 6 サイズ × 2 = 12 モデル展開と比べて、エッジ AI 用途のサイズ多様性が少ない
- **arXiv 番号 2604.15804 は 2026 年 4 月（仮想的）**、Gemini-3.1 Pro / GPT-5 等のベースラインも同様に時間的に整合する必要あり
- **訓練データ完全非公開**（cf. [[entities/internvl-3|InternVL 3]] は公開）
- **API のみ公開**でモデル重みのオープン化なし（Qwen3-VL は Apache 2.0 で重み公開）
- **AuT のゼロから学習に 4000 万時間音声**は再現困難な規模、独自データへの依存

## CV 分野における意義

Qwen3.5-Omni は **「視覚 + 音声 + テキストの完全 omnimodal 統合」** という、純粋 VL モデル（[[entities/qwen3-vl|Qwen3-VL]]）とは異なる進化軸を示す。これは：

1. **動画 + 音声の同時処理**: 動画理解で同サイズ純粋 VL モデル（Qwen3.5-Plus-Instruct）を 6/6 動画ベンチで上回る → **「視覚と聴覚は本質的に結合している」** という人間的知覚モデルを実証
2. **音声-視覚エージェント**: Audio-Visual Vibe Coding（音声-視覚命令からの直接コード生成）という新カテゴリの能力
3. **長文脈 omnimodal**: 10 時間音声 / 400 秒動画 / 256K トークンで実世界の長尺シナリオに対応
4. **ARIA による低遅延ストリーミング**: 235ms first-packet 遅延でリアルタイム対話の実用域
5. **多言語の網羅**: 音声入力 113 言語 + 39 中国方言、音声出力 36、テキスト 201 言語 → 文化的多様性への対応

Computer Vision wiki 内では、**Qwen ファミリーの 2 つの主軸**として：
- **[[entities/qwen3-vl|Qwen3-VL]]**: 純粋 Vision-Language の最先端、画像・動画・文書・GUI エージェント
- **Qwen3.5-Omni**: 音声を統合した omnimodal、音声-視覚対話・リアルタイム相互作用

の双方を持つ。これは [[entities/internvl-3-5|InternVL 3.5]]（純粋 VL + 推論）とは異なる「音声を含む完全マルチモーダル」アプローチで、現代の MLLM の **「VL 純化路線 vs Omni 統合路線」** の対立軸を示す。

## 用語と略称

- **Omnimodal** = テキスト + 画像 + 音声 + 動画（音声付き）の 4 モダリティを統合
- **MLLM / LVLM** = Multimodal LLM / Large Vision-Language Model
- **Thinker-Talker 構造** = テキスト生成（Thinker）+ 音声生成（Talker）の 2 分岐構造、Qwen2.5-Omni で初導入
- **Hybrid MoE** = Hybrid Attention Mixture-of-Experts、Qwen3.5 で導入
- **Gated Delta Net (GDN)** = 長系列モデリング加速モジュール、Hybrid MoE に含まれる
- **AuT (Audio Transformer)** = Qwen3.5-Omni の音声エンコーダ、ゼロから 4000 万時間で学習、6.25Hz
- **SigLIP2** = 視覚エンコーダ、Qwen3.5 から共有（[[entities/siglip]] 系列）
- **MTP (Multi-Token Prediction)** = Talker が残差コードブックを予測するモジュール
- **Code2Wav** = 因果的 ConvNet による波形再構成
- **RVQ (Residual Vector Quantization)** = 残差ベクトル量子化、マルチコードブック音声表現
- **ARIA (Adaptive Rate Interleave Alignment)** = 本論文の新規貢献、テキスト-音声単位の動的整合
- **MFA (Montreal Forced Aligner)** = 従来のテキスト-音声強制整合ツール、ARIA で置換される
- **TM-RoPE** = Time-aware Multimodal RoPE、Qwen2.5-Omni で導入された音声-動画同期用 RoPE
- **TMRoPE** = TM-RoPE と同義
- **OPD (On-Policy Distillation)** = テキスト条件付き応答を音声条件付きクエリの蒸留ターゲットに使う Stage 2 事後学習
- **Interaction-Aligned RL** = 複数ターン相互作用品質を最適化する Stage 3 事後学習
- **Specialist Distillation** = ドメイン特化教師モデルから単一統一モデルへの蒸留、Stage 1 事後学習
- **DPO (Direct Preference Optimization)** = 直接選好最適化、Talker RL Stage で使用
- **GSPO** = Talker RL Stage で DPO と併用される、ルールベース報酬統合
- **CPT (Continual Pre-Training)** = 継続的事前学習、Talker Stage 2 で使用
- **Audio-Visual Vibe Coding** = 音声-視覚命令から直接実行可能コードを生成する出現能力、Qwen3.5-Omni の新規発見
- **FunctionCall** = エージェント的関数呼び出し能力
- **WebSearch** = 自律的 Web 検索能力
- **TTFT (Time-To-First-Token)** = 入力受信から最初のトークン生成までの時間
- **TTFC (Time-To-First-Chunk)** = Talker の最初の音声チャンク生成までの時間
- **TPOP (Time-Per-Output-Token)** = 定常状態デコード中のトークンあたり遅延
- **TPS (Tokens Per Second)** = 生成スループット
- **Generation RTF (Real-Time Factor)** = リアルタイム係数、< 1 でリアルタイム再生可能
- **First-Packet Latency** = 最初の音声パケット生成までの遅延、ストリーミング体験の決定的要因
- **vLLM** = LLM 推論サーバ（PagedAttention 機構）
- **SGLang** = 構造化生成・複雑プロンプト処理に優れた推論サーバ
- **PAI-Lingjun** = Alibaba Cloud の AI 計算サービス（Qwen3-VL でも使用）
- **S2TT** = Speech-to-Text Translation（音声-テキスト翻訳）
- **ASR** = Automatic Speech Recognition（自動音声認識）
- **WER** = Word Error Rate（単語誤り率、↓ が良い）
- **CER** = Character Error Rate（文字誤り率、↓ が良い）
- **BLEU** = 機械翻訳評価指標（↑ が良い）
- **SIM** = Speaker Similarity（話者類似度、cosine 類似度、↑ が良い）
- **VoiceBench / MMAU / MMAR / MMSU / RUL-MuchoMusic / SongFormBench** = 音声理解ベンチマーク
- **DailyOmni / WorldSense / AVUT / AV-SpeakerBench / Qualcomm IVD / Omni-Cloze / OmniGAIA** = 音声-視覚ベンチマーク
- **VideoMME w/ audio** = VideoMME を音声付きで評価（use_audio_in_video=True）
- **SEED-TTS** = ゼロショット TTS ベンチマーク
- **FLEURS / Common Voice / LibriSpeech / WenetSpeech / KeSpeech / Opencpop / MIR-1K** = ASR ベンチマーク
- **CV3-Eval** = クロス言語音声生成ベンチマーク
- **SLAKE / PMC-VQA / MedXpertQA-MM** = 医療 VQA ベンチマーク

## 関連ページ

- [[entities/qwen3-5-omni]] — Qwen3.5-Omni-Plus / Qwen3.5-Omni-Flash のエンティティ・ページ
- [[translations/qwen3-5-omni]] — 本論文の本文 + 付録翻訳
- [[entities/qwen3-vl]] — 純粋 VL の Qwen 系最新版（並列発展、2025 Nov）
- [[entities/qwen2-5-vl]] — Qwen2.5-VL（[[concepts/rotary-position-embeddings]] / MRoPE absolute time → テキスト・タイムスタンプの源流）
- [[entities/siglip]] — SigLIP2 視覚エンコーダ
- [[concepts/rotary-position-embeddings]] — TM-RoPE / Interleaved MRoPE の前提
- [[concepts/foundation-model]] — omnimodal 基盤モデル
- [[concepts/weakly-supervised-pretraining]] — 1 億時間音声-視覚データの弱教師あり学習
- [[entities/internvl-3-5]] — 同時期の対抗系譜（純粋 VL + MoE + Cascade RL）
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系
