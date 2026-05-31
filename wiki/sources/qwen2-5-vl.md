---
type: source
source_path: raw/papers/Qwen2.5-VL Technical Report.pdf
source_kind: paper
title: "Qwen2.5-VL Technical Report"
authors: [Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin]
year: 2025
venue: arXiv:2502.13923
ingested: 2026-05-30
tags: [qwen, qwen2-5-vl, mllm, multimodal, vision-language, window-attention, mrope-absolute-time, dynamic-fps, native-dynamic-resolution, omni-document-parsing, gui-agent, alibaba]
translation: [[translations/qwen2-5-vl]]
---

# Qwen2.5-VL — ViT from scratch + Window Attention + MRoPE-aligned-to-absolute-time で「視覚エージェント」を本格化した Qwen 系第 3 世代

> 原典: [[translations/qwen2-5-vl]] ・ `raw/papers/Qwen2.5-VL Technical Report.pdf`
> 著者・年・会議: Shuai Bai ら（Qwen Team, Alibaba Group）, 2025 Feb 19（arXiv:2502.13923）, テクニカル・レポート

## 一言まとめ

[[entities/qwen2-vl|Qwen2-VL]]（2024 年 9 月）の **Naive Dynamic Resolution + M-RoPE** をベースに、(i) **Window Attention を導入した ViT をゼロから学習**して計算コストを線形化、(ii) **MRoPE を絶対時間に整合**させて FPS が異なる動画でも一貫した時間理解を実現、(iii) **動的 FPS サンプリング**で数秒〜数時間の動画を統一処理、(iv) 事前学習データを **1.2T → 4.1T トークン**に拡大、(v) **omni-document parsing**（HTML タグでレイアウト・表・チャート・数式・楽譜・化学式を統一表現）と **絶対座標グラウンディング**で文書理解と細粒度位置特定を商用級に。**3B / 7B / 72B の 3 サイズ**で、Qwen2.5-VL-72B は MMMU 70.2 / MathVista 74.8 / DocVQA 96.4 / OCRBench 885 / RefCOCO 92.7 / Charades-STA mIoU 50.9 / ScreenSpot Pro 43.6（Aguvis-72B 23.6 を圧倒）など多数の新 SOTA を達成、GPT-4o / Claude 3.5 Sonnet に肩を並べる。**「視覚言語モデルから視覚エージェントへ」** の転換を Qwen ファミリーで本格化した世代。

## 背景と問題意識

[[entities/qwen2-vl|Qwen2-VL]] は **Naive Dynamic Resolution + M-RoPE** で「任意解像度」のオープンソース MLLM の標準を確立したが、3 つの未解決課題が残っていた：

1. **計算複雑性**: ネイティブ解像度入力に伴う ViT の 2 次計算複雑性。Qwen2-VL は ViT 全層で完全自己注意を使うため、高解像度画像で計算負荷が爆発
2. **動画の時間動態**: Qwen2-VL の MRoPE は **時間 ID をフレーム番号に直接結びつけていた**ため、FPS が異なる動画（10 秒 30fps と 10 秒 5fps）で同じ「秒数」を表現できなかった。秒レベルのイベント位置特定（時間グラウンディング）が苦手
3. **データのスケールと品質**: Qwen2-VL 段階は 1.4T トークン（うち画像-テキスト 1.2T 強）と、当時の Gemini / GPT-4o との「データ規模差」が残っていた

加えて、2024 年後半に台頭した **GUI エージェント**（スマートフォン操作、Web ブラウジング、デスクトップ操作）の流れに対し、Qwen2-VL は AITZ 等で GPT-4o を上回るものの、**ScreenSpot Pro のような高解像度プロ向け UI ベンチで 1.6%** と完全に未対応の状態だった。

Qwen2.5-VL はこれら 4 つの課題を一気に解決し、**「マルチモーダル基盤モデル」と「視覚エージェント」の 2 軸**で同時に商用フロンティアに追いつくことを目指した。

## 提案手法 / 主張

### アーキテクチャ：3 サイズ × 統一 ViT 設計

| モデル | Vision Encoder | Vision-Language Merger | LLM | 学習トークン |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 共通 ViT（次節参照） | In 1280 → Out 2048 | Qwen2.5 2048H / 36L / KV2 / Embedding Tying ✓ | 4.1T |
| Qwen2.5-VL-7B | 共通 ViT | In 1280 → Out 3584 | Qwen2.5 3584H / 28L / KV4 | 4.1T |
| Qwen2.5-VL-72B | 共通 ViT | In 1280 → Out 8192 | Qwen2.5 8192H / 80L / KV8 | 4.1T |

**Qwen2-VL からの変化点（重要）**:
- サイズ展開が **2B → 3B**（最小モデルがやや大型化）。Qwen2.5 ファミリーに最も小型な 2B が存在しないため
- 全サイズで **同一 ViT を共有**する設計は [[entities/qwen2-vl|Qwen2-VL]] と同じだが、**ViT は今回ゼロから学習**（DFN 初期化ではなくなった）
- Vision-Language Merger は **MLP 2 層で 4 パッチを 1 トークンに圧縮**（Qwen2-VL と同じ 4× 圧縮）

### 1. Fast and Efficient Vision Encoder（再設計 ViT）

| ViT 構成 | 値 |
| --- | --- |
| Hidden Size | 1280 |
| # Layers | **32** |
| # Heads | 16 |
| Intermediate Size | 3456 |
| Patch Size | 14 |
| **Window Size** | **112**（= 8×8 パッチ） |
| **Full Attention Block Indexes** | **{7, 15, 23, 31}** |
| Normalization | **RMSNorm** |
| Activation | **SwiGLU** |
| Position Embedding | **2D RoPE**（動画は 3D パッチ分割） |

**Window Attention の効果**:
- 32 層中、**完全自己注意は 4 層のみ**（インデックス 7/15/23/31）。残り 28 層は **112×112 ウィンドウ内アテンション**
- これにより計算量は **パッチ数に対して線形にスケール**（従来は 2 次）
- 112×112 より小さい領域は **パディングなしに処理**して元の解像度を保持
- LLM 系の構造（RMSNorm + SwiGLU）に揃えることで視覚と言語の整合性も改善

**ViT をゼロから学習**:
- 初期化に DataComp + 社内データを使用（Qwen2-VL の DFN 初期化と異なる）
- 学習段階: CLIP 事前学習 → 視覚言語整合 → エンドツーエンド微調整
- ネイティブ解像度で動的サンプリング、画像は元のアスペクト比に従ってランダム選択

**動画への 3D パッチ分割**: 画像と同じ 14×14 パッチを基本単位とし、**連続 2 フレームをグループ化**して言語モデルへのトークン数を半減（Qwen2-VL の 2D conv depth=2 と同思想）

### 2. Native Dynamic Resolution and Frame Rate

**空間領域の新規性**:
- バウンディング・ボックス、点、その他の空間特徴を **入力画像の実際の次元で直接表現**（座標を 0-1000 等に正規化せず）
- これによりモデルは **スケール情報を本質的に学習**できる

**時間領域の新規性**:
- 学習時に **FPS を動的にサンプリング**（5 fps、10 fps、30 fps などを混在させる）→ 推論時に様々な FPS の動画を均等に扱える
- 半時間超の動画には **複数フレーム・キャプション**を targeted 合成パイプラインで作成
- タイムスタンプを **秒ベース形式 + 時-分-秒-フレーム（hmsf）形式**の双方で定式化

### 3. MRoPE Aligned to Absolute Time（最重要の新規性）

**Qwen2-VL の MRoPE の弱点**:
- 時間 ID が **入力フレーム数に紐付け**られていた
- 30 fps で 10 秒撮影した 300 フレーム動画と、3 fps で 10 秒撮影した 30 フレーム動画は、**同じ「10 秒」なのに MRoPE では完全に異なる時間 ID 列**になっていた

**Qwen2.5-VL の解決策**:
- **時間 ID 間の間隔を絶対時間に揃える**（例: 1 秒経過ごとに temporal ID を +1）
- これにより異なる FPS の動画で **一貫した時間整合を学習**できる
- **追加のテキスト・タイムスタンプ注入や追加ヘッドが不要**（Qwen-Agent や他社 MLLM が採用するハック的アプローチを回避）

これが **動画グラウンディング Charades-STA mIoU 50.9（GPT-4o 35.7 を +15.2）** と LVBench / MLVU SOTA の決め手。

### 4. Pre-Training Data 1.2T → 4.1T tokens

データ拡張の内訳：
- **Interleaved image-text data**: 内部評価モデルで 4 基準採点（テキスト品質 / 画像-テキスト関連性 / 相補性 / 情報密度バランス）
- **Grounding data with absolute coordinates**: **10,000 以上の物体カテゴリ**でオープンボキャブラリ検出を強化、**存在しないカテゴリも合成**して頑健性向上、PixMo データセット + 自動化パイプラインで **点ベース・グラウンディング**専用データセット構築
- **Document Omni-Parsing Data**: 後述の **QwenVL HTML フォーマット**でレイアウト・表・チャート・数式・楽譜・化学式を統一表現
- **OCR Data**: フランス語・ドイツ語・イタリア語・スペイン語・ポルトガル語・アラビア語・ロシア語・日本語・韓国語・ベトナム語の多言語、チャート 100 万合成（matplotlib/seaborn/plotly）、表 600 万を E2E モデルで処理
- **Video Data**: 動的 FPS サンプリング、半時間超の長動画キャプション、秒形式と hmsf 形式
- **Agent Data**: モバイル・Web・デスクトップから収集、共有行動空間の関数呼び出し形式に統一、各ステップの推論過程を人間+モデル注釈者で生成

#### QwenVL HTML Format（重要な革新）

文書を **HTML タグ + data-bbox 属性**で統一表現する独自フォーマット：

```html
<p data-bbox="x1 y1 x2 y2">paragraph content</p>
<table data-bbox="x1 y1 x2 y2" class="table{id}">table content</table>
<div class="chart" data-bbox="x1 y1 x2 y2"><img/><table>chart content</table></div>
<div class="formula" data-bbox="x1 y1 x2 y2"><img/><div>formula content</div></div>
<div class="music sheet" format="abc notation" data-bbox="..."><img/><div>music sheet content</div></div>
<div class="chemical formula" format="smile" data-bbox="..."><img/><div>chemical formula content</div></div>
```

- **レイアウト・テキスト・チャート・図解を 1 つの HTML 文書として表現**できる初めての試み
- 楽譜は **ABC notation**、化学式は **SMILES** という業界標準を採用
- 文書を「読み順序」で構造化し、各モジュールに座標を付与
- これが **OCRBench_v2 で英語 +9.6 / 中国語 +20.6 ポイント**（Gemini-1.5-Pro 比）の決定打

### 学習：3 段階事前学習 + 2 段階事後学習

#### 3 段階事前学習（表 2、累積 4.1T トークン）

| Stage | データ | トークン | Seq | 学習対象 |
| --- | --- | --- | --- | --- |
| **Visual Pre-Training** | Image Caption / Knowledge / OCR | **1.5T** | 8192 | **ViT のみ** |
| **Multimodal Pre-Training** | + Pure text / Interleaved / VQA / Video / Grounding / Agent | **2T** | 8192 | ViT & LLM |
| **Long-Context Pre-Training** | + Long Video / Long Agent / Long Document | **0.6T** | **32768** | ViT & LLM |

Stage1 では **ViT のみ学習** で LLM との整合を取り（Qwen2-VL も同じ方針）、Stage2 で全凍結解除して相互作用を深め、Stage3 で **系列長を 4× に拡張**（8192 → 32768）して長文脈耐性を獲得。

#### 2 段階事後学習（SFT + DPO、両フェーズで ViT 凍結）

- **SFT**: 200 万エントリ（純粋テキスト 50% / マルチモーダル 50%）、ChatML 形式、中国語・英語中心 + 多言語補足
- **データ・フィルタリング・パイプライン**: 
  - Stage 1（ドメイン分類）: 専用分類器 *Qwen2-VL-Instag* で **8 主要ドメイン × 30 サブカテゴリ**に階層分類
  - Stage 2（ドメイン特化フィルタ）: ルールベース（反復パターン除去等）+ モデルベース（複雑性 / 関連性 / 完全性 / 視覚情報利用の検証）
- **棄却サンプリング**: ground truth と一致するサンプルのみ保持、CoT 推論を明示的に重視、コード切り替え / 過度の長さ / 反復パターンを除去
- **DPO**: 画像-テキスト + 純粋テキストのみ、各サンプル 1 回処理で効率最適化

## 実験結果と知見

### 一般 VQA / 大学レベル推論 / 数学（表 3、72B）

| ベンチマーク | Qwen2.5-VL-72B | GPT-4o | Claude-3.5 Sonnet | Qwen2-VL-72B |
| --- | --- | --- | --- | --- |
| **MMMU val** | **70.2** | 69.1 | 68.3 | 64.5 |
| MMMU-Pro overall | 51.1 | **51.9** | 51.5 | 46.2 |
| **MathVista mini** | **74.8** | 63.8 | 67.7 | 70.5 |
| **MATH-Vision** | **38.1** | 30.4 | - | 25.9 |
| **MathVerse mini** | **57.6** | 50.2 | - | - |
| **MMBench-EN test** | **88.6** | 83.4 | 82.6 | 86.9 |
| **MMBench-V1.1-EN** | **88.4** | 83.1 | 80.9 | 86.1 |
| **MMStar** | **70.8** | 64.7 | 65.1 | 68.3 |
| **MuirBench** | **70.7** | 68.0 | - | - |
| **CRPE relation** | **79.2** | 76.6 | - | - |
| **MMVet turbo** | **76.2** | 69.1 | 70.1 | 74.0 |
| MME-RealWorld en | **63.2** | 45.2 | 51.6 | - |

**ハイライト**:
- **MMMU で初めて 70.2 を達成**（[[entities/qwen2-vl|Qwen2-VL]] 64.5 → +5.7、GPT-4o 69.1 を超え）。Qwen2-VL 時代の弱点を解消
- **MathVista 74.8**（Qwen2-VL から +4.3、GPT-4o から +11.0）。数学推論で商用モデルを圧倒
- **MathVision 38.1**（Qwen2-VL 25.9 から +12.2、GPT-4o 30.4 から +7.7）。数学コンテスト問題で大幅進化
- MMBench 系・MMStar・MMVet で揃って新 SOTA

### 純粋テキスト・タスク（表 4、72B 同士の比較）

| ベンチマーク | Qwen2.5-VL-72B | Qwen2.5-72B | Llama-3.1-405B |
| --- | --- | --- | --- |
| MMLU-Pro | 71.2 | 71.1 | **73.3** |
| MMLU-redux | 85.9 | **86.8** | 86.2 |
| **LiveBench-0831** | **57.0** | 52.3 | 53.2 |
| GPQA | 49.0 | 49.0 | **51.1** |
| MATH | 83.0 | **83.1** | 73.8 |
| GSM8K | 95.3 | 95.8 | **96.8** |
| HumanEval | 87.8 | 86.6 | **89.0** |
| **MultiPL-E** | **79.5** | 75.1 | 73.5 |
| **IFEval** | **86.3** | 84.1 | 86.0 |

**Qwen2.5-72B（純粋 LLM）とほぼ同等の純粋テキスト性能を保持**（マルチモーダル化による劣化なし）。LiveBench / MultiPL-E / IFEval では Qwen2.5-72B を上回る項目もあり、[[entities/internvl-3|InternVL 3]] が初実証した **「マルチモーダル化で言語が強くなる」現象**が Qwen 系でも観察される。

### 文書理解 / OCR（表 5）

| ベンチマーク | Qwen2.5-VL-72B | GPT-4o | Gemini 1.5-Pro | Claude-3.5 Sonnet | InternVL2.5-78B |
| --- | --- | --- | --- | --- | --- |
| **CC-OCR** | **79.8** | 66.9 | 73.0 | 62.5 | 64.7 |
| **OmniDocBench edit en** ↓ | **0.226** | 0.265 | 0.230 | 0.330 | 0.275 |
| AI2D w. M. | 88.7 | 84.6 | 88.4 | 81.2 | **89.1** |
| **DocVQA test** | **96.4** | 91.1 | 93.1 | 95.2 | 95.1 |
| **InfoVQA test** | **87.3** | 80.7 | 81.0 | 74.3 | 84.1 |
| **ChartQA test** | 89.5 | 86.7 | 87.2 | **90.8** | 88.3 |
| **CharXiv DQ** | **87.4** | 84.5 | 72.0 | 84.3 | 82.3 |
| **SEED-Bench-2-Plus** | **73.0** | 72.0 | 70.8 | 71.7 | 71.3 |
| **OCRBench** | **885** | 736 | 754 | 788 | 854 |
| **OCRBench_v2 en/zh** | **61.5/63.7** | 46.5/32.2 | 51.9/43.1 | 45.2/39.6 | 49.8/52.1 |

**OCRBench_v2 で Gemini-1.5-Pro を英語 +9.6 / 中国語 +20.6 ポイントで圧倒**。**OmniDocBench で文書解析の編集距離が業界最低**。QwenVL HTML フォーマットによる統一表現の効果。

### 視覚グラウンディング（表 6 & 7）

| ベンチマーク | Qwen2.5-VL-72B | InternVL2.5-78B | Grounding DINO | Molmo-72B | GPT-4o |
| --- | --- | --- | --- | --- | --- |
| RefCOCO val | 92.7 | **93.7** | 90.6 | - | - |
| RefCOCO testA | 94.6 | **95.6** | 93.2 | - | - |
| RefCOCO+ val | 88.9 | **90.4** | 88.2 | - | - |
| RefCOCOg val | 89.9 | **92.7** | 86.1 | - | - |
| **ODinW**（オープン語彙検出） | **43.1** | 31.7 | 55.0 (専門) | - | - |
| **PointGrounding** | 67.5 | - | - | **69.2** | - |
| **CountBench** | **93.6** | 72.1 | - | 91.2 | 87.9 |

**参照表現理解は InternVL2.5-78B にわずかに後れる**が、**ODinW でオープン語彙検出 43.1 mAP**（InternVL の +11.4、専門モデル Grounding DINO 55.0 にも追従）、**CountBench 93.6 SOTA**（GPT-4o 87.9 を +5.7）。Molmo に近い **点ベース・グラウンディング**で 67.5。

### 動画理解（表 8）

| ベンチマーク | Qwen2.5-VL-72B | Gemini 1.5-Pro | GPT-4o |
| --- | --- | --- | --- |
| Video-MME w/o sub. | 73.3 | **75.0** | 71.9 |
| Video-MME w sub. | 79.1 | **81.3** | 77.2 |
| Video-MMMU | 60.2 | 53.9 | **61.2** |
| **MVBench** | **70.4** | 60.5 | 64.6 |
| **MMBench-Video** | **2.02** | 1.30 | 1.63 |
| **LVBench** | **47.3** | 33.1 | 30.8 |
| **EgoSchema test** | **76.2** | 71.2 | 72.2 |
| **PerceptionTest** | **73.2** | - | - |
| **MLVU M-Avg** | **74.6** | - | 64.6 |
| **TempCompass Avg** | **74.8** | 67.1 | 73.8 |
| **Charades-STA mIoU**（時間グラウンディング） | **50.9** | - | 35.7 |

**長動画理解の LVBench / MLVU で GPT-4o を 16.5 / 10.0 ポイント圧倒**（M-RoPE 絶対時間整合の効果）。**Charades-STA mIoU 50.9 で時間グラウンディング SOTA**（GPT-4o 35.7 を +15.2）。Video-MME では Gemini-1.5-Pro にやや後れるが、商用フロンティアと完全に互角。

### GUI エージェント（表 9）

| ベンチマーク | Qwen2.5-VL-72B | Aguvis-72B | Qwen2-VL-72B | GPT-4o | Gemini 2.0 | Claude |
| --- | --- | --- | --- | --- | --- | --- |
| ScreenSpot | 87.1 | **89.2** | - | 18.1 | 84.0 | 83.0 |
| **ScreenSpot Pro** | **43.6** | 23.6 | **1.6** | - | - | 17.1 |
| **Android Control High EM** | **67.36** | 66.4 | 59.1 | 20.8 | 28.5 | 12.5 |
| **Android Control Low EM** | **93.7** | 84.4 | 59.2 | 19.4 | 60.2 | 19.4 |
| **AndroidWorld SR** | **35%** | 26.1% | 6% (SoM) | 34.5% (SoM) | 26% (SoM) | 27.9% |
| **MobileMiniWob++ SR** | **68%** | 66% | 50% (SoM) | 61% | 42% (SoM) | 61% (SoM) |
| OSWorld | 8.83 | 10.26 | 2.42 | 5.03 | 4.70 | **14.90** |

**ScreenSpot Pro で Qwen2-VL の 1.6% から 43.6% へ +42 ポイントの飛躍**。同じ Qwen-VL 系列で **27× の精度向上**は、UI グラウンディングの本格的進化を示す。**Android Control / AndroidWorld / MobileMiniWob++ でも GPT-4o / Gemini 2.0 / Claude を上回る**、世界トップクラスの GUI エージェント能力。OSWorld（デスクトップ）のみ Claude に後れ。

## 限界・批判的視点

- **MMMU-Pro で GPT-4o に -0.8**（51.1 vs 51.9）。複雑なマルチドメイン推論ではまだ商用に追いつけていない（後の [[entities/internvl-3-5|InternVL 3.5]] が 77.7 で SOTA を奪取）
- **Video-MME で Gemini-1.5-Pro に -1.7 / -2.2**（w/o subs / w/ subs）。Gemini の長文脈処理（10M トークン級）には及ばず
- **OSWorld（デスクトップ・エージェント）で Claude 14.90 に -6.07**。スマホ / Web は強いがデスクトップは弱点
- **動画フレーム上限 768 / 動画トークン上限 24,576**: Qwen2-VL の 16K → 24K へ拡張したものの、1 時間超動画の真の上限性能は未測定
- **InternVL 3 / 3.5 の Native Multimodal Pre-Training とは対照的**な、依然として「LLM 事前学習 → MLLM 適応」のパラダイム。**「マルチモーダル化で言語が強くなる」現象**は観察されるが、InternVL 3 のような明示的検証は行われていない
- **ViT をゼロから学習**するアプローチは、DFN や InternViT のような既存 VFM の活用（蒸留や継続学習）に比べてデータ効率が悪い可能性
- 訓練データの完全公開なし（[[entities/internvl-3|InternVL 3]] は完全公開）
- **数学コンテスト（MathVision）で 38.1**: 改善はしたが GPT-4o の MathVision を超える専門 LLM（Llama-3.1-405B）レベルにはまだ遠い

## CV 分野における意義

Qwen2.5-VL は **「視覚言語モデル」から「視覚エージェント」への転換点**として位置付けられる。これは、(i) **Window Attention + ViT from scratch** で MLLM の計算効率を線形化、(ii) **MRoPE-aligned-to-absolute-time** で異なる FPS の動画を統一的に扱う MLLM の標準を確立（後続の Qwen3-VL や多くの MLLM が踏襲）、(iii) **QwenVL HTML フォーマット**で文書解析を「複数モデルの組み合わせ」から「単一汎用モデル」へ転換、(iv) **ScreenSpot Pro 43.6 / Android Control 93.7 / AndroidWorld 35%** で GUI エージェントを商用レベルに引き上げ—という 4 つの軸で MLLM 史を進めた。

[[entities/qwen-vl|Qwen-VL]] が `<box>`/`<ref>` で位置語彙を不要にし、[[entities/qwen2-vl|Qwen2-VL]] が **Naive Dynamic Resolution + M-RoPE** を業界標準にしたのに続き、Qwen2.5-VL は **Window Attention + MRoPE absolute time + 統一文書 HTML** を業界に提供。InternVL シリーズ（[[entities/internvl-3|InternVL 3]] / [[entities/internvl-3-5|InternVL 3.5]]）と並ぶ 2 大オープンソース MLLM 系譜の片翼として、現代の MLLM 設計思想の中核を担う。

## 用語と略称

- **LVLM** = Large Vision-Language Model（大規模視覚言語モデル）、**MLLM**（Multimodal LLM）とも
- **3B / 7B / 72B** = 30 億 / 70 億 / 720 億パラメータの 3 サイズ（Qwen2-VL の 2B → 3B に変化）
- **Window Attention**（ウィンドウ・アテンション） = 全体ではなく局所ウィンドウ内でのみアテンションを行う手法。計算量を 2 次 → 線形に削減
- **Full Attention** = 通常の完全自己注意（全パッチ間でアテンション）
- **Full Attention Block Indexes** = ViT の中で完全自己注意を使う層のインデックス。Qwen2.5-VL は {7, 15, 23, 31}（32 層中 4 層のみ）
- **MRoPE / M-RoPE** = Multimodal Rotary Position Embedding（マルチモーダル回転位置埋め込み）。Qwen2-VL で導入、temporal/height/width 3 成分
- **MRoPE Aligned to Absolute Time** = 本論文の新規性。temporal ID をフレーム番号ではなく **絶対秒数**に対応させる
- **Native Dynamic Resolution** = ネイティブ解像度のまま動的トークン数で処理
- **Dynamic FPS Sampling** = 学習時に異なる FPS を均等に混在させるサンプリング戦略
- **2D RoPE / 3D RoPE** = 2 次元 / 3 次元の回転位置埋め込み。動画は 3D（時間 + 高さ + 幅）
- **RMSNorm** = Root Mean Square Normalization（LLM 系の標準正規化、LayerNorm より高速）
- **SwiGLU** = Swish Gated Linear Unit（LLaMA / Qwen 系で標準の活性化関数）
- **DataComp** = MLLM 視覚エンコーダ初期化向けの大規模画像-テキスト・データセット
- **Vision-Language Merger** = ViT 出力を LLM の埋め込み次元に整合させる MLP 2 層（4 パッチ → 1 トークン圧縮も担当）
- **QwenVL HTML Format** = 本論文が定義した文書表現フォーマット。data-bbox 属性で座標、HTML タグでレイアウト・表・チャート・数式・楽譜・化学式を統一
- **ABC notation** = 楽譜の標準テキスト表現
- **SMILES** = Simplified Molecular Input Line Entry System（化学式の標準テキスト表現）
- **omni-document parsing** = 文書のあらゆる要素（テキスト・レイアウト・表・チャート・数式等）を統一的に解析
- **PixMo** = Molmo 論文（Allen AI）が公開したポインティング・カウント・データセット
- **DPO** = Direct Preference Optimization（直接選好最適化、RLHF の代替）
- **ChatML** = OpenAI 由来の対話マークアップ形式
- **CoT** = Chain-of-Thought（思考の連鎖、推論ステップを明示する手法）
- **SoM** = Set-of-Mark（画面要素に番号マークを付ける GUI エージェント補助）
- **mIoU** = mean Intersection over Union（時間グラウンディング・セグメンテーション評価指標）
- **ScreenSpot / ScreenSpot Pro** = UI 要素グラウンディング・ベンチマーク（Pro はプロ向け高解像度版）
- **Android Control / AndroidWorld / MobileMiniWob++ / OSWorld** = モバイル・Web・デスクトップ操作のエージェント評価
- **AITZ** = Android In The Zoo（Qwen2-VL 時代のモバイル UI ベンチ）
- **Qwen2-VL-Instag** = Qwen2-VL-72B から派生した SFT データ分類モデル
- **Charades-STA** = 動画時間グラウンディング・ベンチマーク
- **LVBench / MLVU / EgoSchema / PerceptionTest / Video-MME / TempCompass** = 動画理解ベンチマーク群
- **OCRBench / OCRBench_v2 / OmniDocBench / CC-OCR / CharXiv** = OCR・文書理解ベンチマーク群
- **MMMU / MMMU-Pro / MathVista / MathVision / MathVerse** = 大学レベル・数学推論ベンチマーク群
- **MMBench / MMStar / MMVet / MM-MT-Bench / MuirBench / BLINK / CRPE / HallBench** = 汎用 VQA ベンチマーク群
- **CountBench** = 物体カウント・ベンチマーク
- **ODinW** = Object Detection in the Wild（オープン語彙物体検出）

## 関連ページ

- [[entities/qwen2-5-vl]] — Qwen2.5-VL 3B/7B/72B のエンティティ・ページ
- [[translations/qwen2-5-vl]] — 本論文の本文翻訳
- [[entities/qwen2-vl]] — 前世代（Naive Dynamic Resolution + M-RoPE）
- [[entities/qwen-vl]] — 初代（`<box>`/`<ref>` 特殊トークン）
- [[concepts/rotary-position-embeddings]] — RoPE の基礎、M-RoPE absolute time の前提
- [[concepts/foundation-model]] — 視覚言語基盤モデル
- [[concepts/weakly-supervised-pretraining]] — 4.1T トークン弱教師あり学習
- [[concepts/zero-shot-transfer]] — オープン語彙検出・点グラウンディング・多言語 OCR でのゼロショット転移
- [[concepts/alignment-tuning]] — SFT + DPO + ChatML
- [[entities/internvl-2-5]] — 同時期に MMMU 70% 突破した InternVL 系
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系後継
- [[entities/internvl-3-5]] — Cascade RL + MoE の InternVL 系最新版
- [[entities/grounding-dino]] — オープン語彙物体検出の専門モデル（Qwen2.5-VL の比較対象）
- [[entities/sam]] / [[entities/sam-2]] — グラウンディング・データ合成に活用
