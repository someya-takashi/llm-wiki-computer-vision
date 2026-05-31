---
type: entity
entity_kind: model
aliases: [Qwen3.5-Omni, Qwen3.5-Omni-Plus, Qwen3.5-Omni-Flash, qwen3-5-omni]
related: [[entities/qwen3-vl]], [[entities/qwen2-5-vl]], [[entities/siglip]], [[concepts/rotary-position-embeddings]], [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]]
sources: [[sources/qwen3-5-omni]]
updated: 2026-05-30
---

# Qwen3.5-Omni / Qwen3.5-Omni-Plus / Qwen3.5-Omni-Flash

**Qwen3.5-Omni** は Alibaba Group の Qwen Team が開発する **omnimodal モデル系譜（Qwen-Omni シリーズ）の最新版**（2026 年公開、arXiv:2604.15804）。**Thinker-Talker 構造 + Hybrid MoE + AuT + ARIA** という独自設計で、テキスト・画像・音声・音声-視覚の 4 モダリティを完全統合し、リアルタイム音声-視覚エージェントを実現。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2026 年（arXiv:2604.15804、テクニカル・レポート） |
| 開発元 | Alibaba Group / Qwen Team |
| バリアント | **Qwen3.5-Omni-Plus**（フラッグシップ、数千億パラメータ）+ **Qwen3.5-Omni-Flash**（高効率版） |
| 公開形態 | **API のみ**（モデル重みのオープン化なし、cf. Qwen3-VL は Apache 2.0） |
| デモ | https://chat.qwen.ai |
| 文脈長 | **256K ネイティブ** |
| 音声サポート時間 | **10 時間超** |
| 動画サポート時間 | **400 秒超**（720P @ 1 FPS） |
| 多言語 | 音声入力 **113 言語+方言** / 音声出力 **36** / テキスト **201** |

## モジュール構成（Thinker-Talker 構造）

| モジュール | アーキテクチャ | ストリーミング | 役割 |
| --- | --- | --- | --- |
| **Audio Encoder** | **AuT**（6.25Hz、4000 万時間で学習） | ✓ | 音声→トークン |
| **Vision Encoder** | **SigLIP2**（Qwen3.5 共有） | – | 画像/動画→トークン |
| **Thinker** | **Hybrid MoE Transformer**（Qwen3.5 ベース + GDN） | ✓ | テキスト生成 + マルチモーダル理解 |
| **Talker** | **Hybrid MoE Transformer** | ✓ | ストリーミング音声トークン生成（RVQ） |
| **MTP** | Dense Transformer | ✓ | 残差コードブック予測 |
| **Code2Wav** | 因果的 ConvNet | ✓ | 波形再構成 |

## 7 つの主要進化（vs Qwen3-Omni）

### 1. Hybrid Attention MoE（Thinker + Talker 双方）
- **Qwen3.5 で導入された Hybrid MoE 構造**を双方が採用
- **Gated Delta Net (GDN) モジュール**で長音声-動画系列モデリング加速
- **KV キャッシュ I/O オーバーヘッド削減**、サービング並行性向上

### 2. AuT (Audio Transformer)
- ゼロから学習、Qwen3-ASR が生成した **4000 万時間音声-テキスト対**を消費
- フィルタバンク特徴を 4× Conv2D ブロックで **16× ダウンサンプリング**
- **6.25Hz トークンレート**（フレームあたり 160ms）
- **20+ 言語、中英多言語比率 3.5:3.5:3**
- **動的注意ウィンドウサイズ学習**でリアルタイムとオフラインのバランス

### 3. 256K ネイティブ文脈
- **10 時間超の音声理解**
- **400 秒超の 720P 動画**（1 FPS）
- **1 億時間音声-視覚コンテンツ**で事前学習

### 4. マルチコードブック・コーデック表現
- 単一フレーム即時音声合成
- **RVQ (Residual Vector Quantization)** ベース
- MTP モジュールで残差コードブック細粒度モデリング

### 5. ARIA (Adaptive Rate Interleave Alignment)
- Qwen3-Omni の dual-channel 生成を **統一交互配置単一ストリーム**に再構成
- **適応的レート制約**: 生成系列の任意 prefix で累積音声-テキスト比率はグローバル比率を超えない
- MFA や固定交互配置を不要に、テキスト-音声整合を動的化
- **単語スキップ、誤発音、数字レンダリング曖昧化の問題を解消**

### 6. 多言語サポート大幅拡大

| Modality | 数 | カバレッジ |
| --- | --- | --- |
| Text | **201** | Qwen3.5 互換 |
| Speech Input | **113** | 74 言語 + **39 中国方言**（東北マンダリン、上海語、四川語、客家語、閩南語等） |
| Speech Output | **36** | 29 言語 + **7 中国方言**（北京、天津、南京、陝西、四川、広東、閩南） |

### 7. Audio-Visual Vibe Coding（出現能力）
- 音声-視覚命令から **直接実行可能コード**を生成
- omnimodal モデルにおける新規出現能力
- 外部オーケストレーション不要

## 3 段階事前学習

| Stage | 目的 | 学習対象 | 系列長 | データ |
| --- | --- | --- | --- | --- |
| **S1** Encoder Alignment | アダプタ + エンコーダ整合 | Vision/Audio Encoder（LLM 凍結） | - | 高品質音声-テキスト + 画像-テキスト |
| **S2** General | 全パラメータ学習 | All | **32,768** | **~4T トークン**（テキスト 0.92T + 音声 1.99T + 画像 0.95T + 動画 0.14T + 動画-音声 0.29T） |
| **S3** Long Context | 長系列適応 | All | **262,144** | 長音声・長動画比率増加 |

## 事後学習

### Thinker（3 段階）

1. **Specialist Distillation**: ドメイン特化教師モデル（テキスト/視覚/音声）→ SFT + RL → 単一統一モデルへ蒸留
2. **On-Policy Distillation (OPD)**: テキスト条件付き応答を音声条件付きクエリの蒸留ターゲットに → モダリティ一貫した生成
3. **Interaction-Aligned RL**: 複数ターン相互作用で言語コード切り替え・ペルソナ不整合・命令追従劣化を緩和

### Talker（4 段階）

1. **General Stage**: **2000 万時間超の多言語音声** + マルチモーダル文脈
2. **Long-Context Stage**: CPT + Qwen3-Omni-Captioner 増強、**64k トークンへ拡張**
3. **RL Stage**: **DPO** で多言語選好対 + **GSPO** でルールベース報酬統合
4. **Speaker Fine-tuning Stage**: 軽量話者微調整で話者特性を忠実に捕捉

## First-Packet 遅延（実用上の重要指標）

| 入力 | Plus | Flash |
| --- | --- | --- |
| **音声入力** | 435ms | **235ms** |
| **動画入力** | 651ms | **426ms** |

**235ms** はリアルタイム音声対話の実用域に到達。

## 主要ベンチマーク結果

### 音声 → テキスト（vs Gemini-3.1 Pro）

| ベンチマーク | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
| --- | --- | --- |
| MMAU | 81.1 | **82.2** |
| MMSU | 81.3 | **82.8** |
| RUL-MuchoMusic | 59.6 | **72.4**（+12.8）|
| VoiceBench | 88.9 | **93.1**（+4.2）|
| Fleurs S2TT (top59) | 32.1 | **32.8** |
| Fleurs ASR (top60, WER ↓) | 7.32 | **6.55** |
| CV15 zh (WER ↓) | 8.59 | **3.46** |
| Librispeech clean (WER ↓) | 3.36 | **1.11** |
| **Kespeech（方言、WER ↓）** | 23.67 | **3.46**（**6.8× 改善**）|

### 視覚 → テキスト（vs Qwen3.5-Plus-Instruct）

| カテゴリ | ベンチ | Plus-Instruct | Qwen3.5-Omni-Plus |
| --- | --- | --- | --- |
| RealWorldQA | 79.1 | **84.1**（+5.0）|
| VideoMME | 81.0 | **81.9** |
| MLVU | 85.1 | **86.8** |
| MVBench | 76.7 | **79.0** |
| LVBench | 68.6 | **71.2** |
| MME-VideoOCR | 74.2 | **77.0** |
| SLAKE（医療 VQA）| 82.8 | **84.7** |
| EmbSpatialBench | 83.4 | **85.4** |

**動画理解 6/6 ベンチで純粋 VL を上回る** → 共同 video-audio 学習の効果。

### 音声-視覚 → テキスト（vs Gemini-3.1 Pro）

| ベンチマーク | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
| --- | --- | --- |
| **DailyOmni** | 82.7 | **84.6**（SOTA）|
| **Qualcomm IVD**（実世界 AV 対話）| 66.2 | **68.5**（+2.3）|
| **Omni-Cloze**（AV キャプション）| 57.2 | **64.8**（+7.6）|
| AVUT | **85.6** | 85.0 |
| AV-SpeakerBench | **75.1** | 71.3 |
| VideoMME w/ audio | **89.0** | 83.7 |
| WorldSense | **65.5** | 62.8 |
| OmniGAIA（ツール使用）| **68.9** | 57.2 |

### 音声生成（SEED-TTS、WER ↓）

| Model | test-zh | test-en |
| --- | --- | --- |
| CosyVoice 3 | **0.71** | 1.45 |
| Qwen3-Omni-30B-A3B | 1.07 | 1.39 |
| **Qwen3.5-Omni-Plus** | 0.99 | **1.26 SOTA** |

### クロス言語音声生成

- **12 方向中 10 で SOTA** vs Qwen3-Omni / CosyVoice
- **zh-to-ko で CosyVoice3 から 72% 相対削減**（14.4 → 4.03）

### 多言語 ASR（FLEURS 平均、WER ↓）

| Model | 60 言語平均 |
| --- | --- |
| **Qwen3.5-Omni-Plus** | **6.6** |
| Gemini-3.1 Pro | 7.3 |
| GPT-4o-Transcribe | 10.4 |

## 限界

- **OmniGAIA（ツール使用）で Gemini に -11.7**（57.2 vs 68.9）→ エージェント・ツール統合に改善余地
- **VideoMME w/ audio で Gemini に -5.3**
- **AV-SpeakerBench で Gemini に -3.8**
- **WorldSense で Gemini に -2.7**
- 純粋テキスト・タスクで同サイズ Qwen3.5-Plus-Instruct より僅か劣る（MMLU-Pro 86.8 vs 85.9 等）
- **2 バリアントのみ**（cf. Qwen3-VL は 6 サイズ × 2 = 12 モデル）
- **API のみ公開**（モデル重み非公開）
- **訓練データ完全非公開**
- **AuT のゼロから学習に 4000 万時間音声**は再現困難な独自データ依存
- ベースラインの Gemini-3.1 Pro / GPT-5 は仮想的、論文は時間的に未来寄り

## 系譜・後継

- **Qwen2.5-Omni-7B**（2025 初頭）: Thinker-Talker 構造を初提唱、TM-RoPE で音声-動画同期、純 dense
- **Qwen3-Omni-30B-A3B**（2025 中頃）: MoE スケーリング初導入、RVQ ベース音声表現、dual-track Talker
- **Qwen3.5-Omni**（2026）← 本ページ、Hybrid MoE Thinker-Talker + ARIA + 256K + Audio-Visual Vibe Coding

## Qwen ファミリーの 2 つの主軸

| 軸 | 代表 | 焦点 |
| --- | --- | --- |
| **VL 純化** | [[entities/qwen3-vl|Qwen3-VL]] | 画像・動画・文書・GUI エージェント（音声なし） |
| **Omni 統合** | Qwen3.5-Omni（本ページ）| テキスト + 画像 + 音声 + 動画 + 音声-視覚 + 音声生成 |

両軸は並列発展し、Qwen3-VL の Interleaved MRoPE / DeepStack / テキスト・ベース時間整合と、Qwen3.5-Omni の Hybrid MoE Thinker-Talker / AuT / ARIA は **異なる技術選択で異なるユースケースに対応**。

## 関連ページ

- [[sources/qwen3-5-omni]] — 原典の要約
- [[translations/qwen3-5-omni]] — 原典の翻訳
- [[entities/qwen3-vl]] — 並列発展する純粋 VL の Qwen 系最新版
- [[entities/qwen2-5-vl]] — テキスト・ベース時間整合の前駆体
- [[entities/siglip]] — SigLIP2 視覚エンコーダ
- [[concepts/rotary-position-embeddings]] — TM-RoPE / 時間整合の基盤
- [[concepts/foundation-model]] — omnimodal 基盤モデル
- [[concepts/weakly-supervised-pretraining]] — 1 億時間音声-視覚データ事前学習
- [[entities/internvl-3-5]] — 同時期対抗系譜（純粋 VL + MoE + Cascade RL）
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系
