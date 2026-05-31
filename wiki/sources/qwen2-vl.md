---
type: source
source_path: raw/papers/Qwen2-VL_ Enhancing Vision-Language Model's Perception of the World at Any Resolution.md
source_kind: paper
title: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
authors: [Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, Junyang Lin]
year: 2024
venue: arXiv:2409.12191
ingested: 2026-05-30
tags: [qwen, qwen2-vl, mllm, multimodal, vision-language, dynamic-resolution, m-rope, naive-dynamic-resolution, video-understanding, 3d-conv, dfn, alibaba]
translation: [[translations/qwen2-vl]]
---

# Qwen2-VL — Naive Dynamic Resolution + M-RoPE で「任意解像度」を実現した Qwen 系第 2 世代 MLLM

> 原典: [[translations/qwen2-vl]] ・ `raw/papers/Qwen2-VL_ Enhancing Vision-Language Model's Perception of the World at Any Resolution.md`
> 著者・年・会議: Peng Wang, Shuai Bai ら（Qwen Team, Alibaba Group）, 2024 Sept, arXiv:2409.12191

## 一言まとめ

[[entities/qwen-vl|Qwen-VL]] 初代の **256 固定トークン圧縮 + 448² 固定解像度**という制約を捨て、**Naive Dynamic Resolution（任意解像度を可変数の視覚トークンに動的変換）+ Multimodal Rotary Position Embedding（M-RoPE, 時間・高さ・幅の 3 成分回転位置埋め込み）+ 統一画像/動画処理（3D 畳み込み）** を導入した Qwen ファミリー第 2 世代。**2B / 7B / 72B の 3 サイズスイート** で展開し、72B は GPT-4o / Claude 3.5-Sonnet と並ぶ性能を多数ベンチで達成、DocVQA 96.5 / OCRBench 877 / MathVista 70.5 / MMBench-CN 86.6（新 SOTA）。**学習 16K トークンで推論 80K トークンまで外挿** という長文脈耐性も特徴的で、20 分以上の動画理解と 32K 文脈長によるエージェント能力（UI 操作・ロボット制御・ナビゲーション）の基盤を作った。

## 背景と問題意識

[[entities/qwen-vl|Qwen-VL]] 初代（2023 Aug）は「**`<box>`/`<ref>` 特殊トークン**による細粒度グラウンディング」と「**位置認識アダプタ（256 学習可能クエリ + 2D 絶対位置エンコーディング）**」で Alibaba MLLM の起点を作ったが、3 つの構造的弱点を抱えていた：

1. **解像度 448×448 固定**: 文書・チャート・インフォグラフィックのような情報密度の高い画像で詳細が失われる
2. **視覚トークン 256 固定**: アスペクト比が極端な画像（縦長スクリーンショット、横長パノラマ）でも同じトークン数に圧縮するため、解像度に応じた柔軟な表現ができない
3. **動画は 2D 画像の系列として扱う**: 時間軸の位置情報が 1D-RoPE では正確にモデル化できない

一方で、ライバル系列 [[entities/internvl-1-5|InternVL 1.5]]（2024 Apr）は既に **35 通りのアスペクト比 × 1-40 タイルの動的高解像度** + Pixel Shuffle で「動的解像度の MLLM」を実現していた。InternVL のタイル分割アプローチに対し、Qwen2-VL はより直接的に **「ViT 自体を任意解像度対応に改造する」** という設計判断を採り、**2D-RoPE + パッチ系列を解像度に応じて可変長化** という方針で実装した。

加えて、**動画理解能力**は MLLM の次のフロンティアであり、20 分以上の長動画を扱える MLLM はオープンソースでは存在しなかった。Qwen2-VL は **時間軸を持つ M-RoPE + 3D 畳み込み + 2 fps サンプリング + 動画あたり 16K トークン上限** という設計で、長動画処理を MLLM の標準機能に押し上げた。

## 提案手法 / 主張

### アーキテクチャ：3 サイズ × 675M ViT 共通

| モデル | Vision Encoder | LLM | 総パラメータ |
| --- | --- | --- | --- |
| Qwen2-VL-2B | **675M ViT**（DFN 初期化） | Qwen2 1.5B | **約 2.2B** |
| Qwen2-VL-7B | 675M ViT | Qwen2 7.6B | **約 8.3B** |
| Qwen2-VL-72B | 675M ViT | Qwen2 72B | **約 72.7B** |

**設計のポイント**: 全サイズで **同一の 675M ViT を共有**することで、LLM のサイズに関係なく視覚処理の計算負荷を一定に保つ。これは Mini-InternVL（[[entities/mini-internvl]]）の InternViT-300M 共有戦略に近い思想。

**LLM の初期化**: Qwen2 シリーズ（中英バイリンガル + 多言語強化版）。Qwen2 は Qwen-7B（[[entities/qwen-vl|Qwen-VL]] 初代の LLM）の正統な後継で、より広範な多言語コーパスで学習されている。

**Vision Encoder の初期化**: **DFN（Data Filtering Networks）由来の ViT** を使用。**元の絶対位置埋め込みを 2D-RoPE に置き換える**ことで、任意解像度入力に対応させた点が肝。

### 3 つの主要技術

#### 1. Naive Dynamic Resolution（素朴な動的解像度）

任意解像度の画像を可変数の視覚トークンに動的変換する機構。実装は驚くほどシンプル：

1. ViT の**絶対位置埋め込みを除去**し **2D-RoPE** を導入（[[concepts/rotary-position-embeddings]]）。これで任意解像度のパッチ列に対して位置情報を付与できる
2. 推論時、複数の様々な解像度の画像を**単一の系列にパック**（パック長は GPU メモリで制御）
3. ViT 後に**単純な MLP 層で隣接 2×2 トークンを 1 トークンに圧縮**（4× 圧縮）
4. 圧縮された視覚トークンの先頭と末尾に **`<|vision_start|>` / `<|vision_end|>`** を配置

**具体例**: 224×224 画像を patch_size=14 で ViT 処理すると (224/14)²=256 パッチ → 2×2 MLP 圧縮で **66 トークン**（256/4 + 視覚境界トークン分）に圧縮されて LLM へ入力。

**設定**: `min_pixels = 100 × 28 × 28`、`max_pixels = 16384 × 28 × 28`（28 = patch 14 × 圧縮 2）で実用範囲を制御。

#### 2. Multimodal Rotary Position Embedding (M-RoPE)

回転位置埋め込み（[[concepts/rotary-position-embeddings|RoPE]]）を **時間（temporal）/ 高さ（height）/ 幅（width）の 3 成分** に分解：

| モダリティ | temporal ID | height ID | width ID |
| --- | --- | --- | --- |
| **テキスト** | 通常の位置 ID | テキスト ID と同じ | テキスト ID と同じ |
| **画像** | 一定（フレーム時刻なし） | パッチの行 index | パッチの列 index |
| **動画** | フレームごとに増分 | パッチの行 index | パッチの列 index |

テキスト処理時は機能的に 1D-RoPE と等価。マルチモーダル系列では「先行モダリティの最大位置 ID + 1」から次モダリティの位置番号付けを開始することで、整合性を保つ。

**重要な副次効果**: M-RoPE は画像と動画の位置 ID 値を相対的に小さく抑える（時間軸と空間軸を分けるため）ので、**推論時のより長い系列への外挿が容易**になる。実際、**学習時 16K トークン上限で、推論時 80K トークンまで頑健**という実験結果（図 5）。

#### 3. Unified Image and Video Understanding（統一画像・動画処理）

- 動画は **毎秒 2 フレーム**でサンプリング
- 動画入力には **深さ 2 の 3D 畳み込み**を統合して 2D パッチではなく **3D チューブ**を扱う → 系列長を増加させずに 2 倍のフレームを処理可能
- 画像は **2 つの同一フレーム** として扱われる（一貫性のため）
- 動画あたり最大 **16,384 視覚トークン**で制限（長動画でも学習効率を確保）

### 学習：3 段階パイプライン（Qwen-VL の踏襲）

| Stage | 凍結対象 | データ | トークン数 |
| --- | --- | --- | --- |
| **Stage 1** | LLM 凍結、ViT のみ学習 | 画像-テキスト対、OCR、画像分類 | 約 **600B** |
| **Stage 2** | 全パラメータ凍結解除 | 画像関連データの追加投入 | 追加 **800B** |
| **Stage 3** | ViT 凍結、LLM のみ SFT | ChatML 命令データ | - |

**累積 1.4 兆トークン**（[[entities/qwen-vl|Qwen-VL]] 初代の Stage 1 が 1.5B サンプル ≒ 500B トークン だったので、約 3 倍）。**監督はテキスト・トークンにのみ与える**（視覚トークンは予測対象としない）という Qwen-VL の方針を踏襲。

データソース：Web ページ、オープンソース・データセット、合成データ。**カットオフ 2023 年 6 月**。

### 拡張された特殊トークン

| 用途 | トークン |
| --- | --- |
| 視覚入力境界 | `<|vision_start|>`, `<|vision_end|>` |
| バウンディング・ボックス | `<|box_start|>`, `<|box_end|>` |
| 参照対象 | `<|object_ref_start|>`, `<|object_ref_end|>` |
| 対話境界 | `<|im_start|>`, `<|im_end|>` |

座標は **[0, 1000) 正規化文字列**（[[entities/qwen-vl|Qwen-VL]] 初代の `<box>`/`<ref>` を踏襲、命名は `vision_start` 等にアップデート）。

### システム実装

- **PAI-Lingjun Intelligent Computing Service**（Alibaba Cloud）で学習
- **CPFS + OSS** の分離ストレージ + **動画キャッシュ・デコード**で I/O ボトルネック解消
- **3D 並列化**（DP + TP + PP）+ **DeepSpeed ZeRO-1** + **系列並列**
- **72B モデルには 1F1B PP**（インターリーブ版より速い）
- **動的系列長**を 1F1B プロセス開始前にブロードキャストする工夫

## 実験結果と知見

### 一般 VQA・OCR・文書理解（表 2、Qwen2-VL-72B vs 商用）

| ベンチマーク | Qwen2-VL-72B | GPT-4o | Claude-3.5 Sonnet |
| --- | --- | --- | --- |
| MMMU val | 64.5 | 69.1 | 68.3 |
| **DocVQA test** | **96.5** | 92.8 | 95.2 |
| **InfoVQA test** | **84.5** | - | - |
| **AI2D** | **88.1** | 84.6 | 80.2 |
| ChartQA test | 88.3 | 85.7 | 90.8 |
| **TextVQA val** | **85.5** | - | - |
| **OCRBench** | **877** | 736 | 788 |
| **MTVQA**（多言語 OCR） | **30.9** | 27.8 | 25.7 |
| **VCR zh easy** | **65.4** | 14.9 | 1.0 |
| **RealWorldQA** | **77.8** | 75.4 | 60.1 |
| MME sum | **2482.7** | 2328.7 | 1920.0 |
| MMBench-EN test | 86.5 | 83.4 | 79.7 |
| **MMBench-CN test** | **86.6** | 82.1 | 80.7 |
| MMT-Bench | **71.7** | 65.5 | - |
| MMStar | **68.3** | 63.9 | 62.2 |
| MMVet | **74.0** | 69.1 | 66.0 |
| HallBench avg | **58.1** | 55.0 | 49.9 |
| **MathVista testmini** | **70.5** | 63.8 | 67.7 |

**ハイライト**: 文書系（DocVQA、InfoVQA、AI2D、OCRBench）で商用フロンティアを上回り、動的解像度の効果が定量的に実証された。中国語 OCR（VCR zh easy 65.4 vs Claude 1.0）と中国語マルチモーダル（MMBench-CN 86.6 SOTA）は圧勝。**MMMU では GPT-4o に -4.6 と未だ後れ**（複雑な大学レベル推論で改善余地）。

### 多言語 OCR（表 3、社内ベンチマーク）

| 言語 | GPT-4o | Qwen2-VL-72B |
| --- | --- | --- |
| Korean | 87.8 | **94.5** |
| Japanese | 88.3 | **93.4** |
| French | 89.7 | **94.1** |
| German | 88.3 | **91.5** |
| Italian | 74.1 | **89.8** |
| Russian | 96.8 | **97.2** |
| Vietnamese | 72.0 | **73.0** |
| Arabic | **75.9** | 70.7 |

**アラビア語以外**で GPT-4o を上回る。欧州主要言語 + 韓国語・日本語・ベトナム語の MLLM 標準として位置付けられる。

### 動画理解（表 4）

| ベンチマーク | Gemini 1.5-Pro | GPT-4o | Qwen2-VL-72B |
| --- | --- | --- | --- |
| **MVBench** | - | - | **73.6** |
| **PerceptionTest** | - | - | **68.0** |
| **EgoSchema test** | 63.2 | 72.2 | **77.9** |
| Video-MME (wo/w subs) | 75.0/81.3 | 71.9/77.2 | 71.2/77.8 |

**EgoSchema で +5.7 ポイント GPT-4o 超え**、Video-MME では Gemini 1.5-Pro に若干後れるが商用クラスに到達。

### 参照表現理解（表 6）

| モデル | RefCOCO val | RefCOCO+ val | RefCOCOg val |
| --- | --- | --- | --- |
| Qwen-VL（初代） | 89.4 | 83.1 | 85.6 |
| Shikra-13B | 87.0 | 81.6 | 82.3 |
| CogVLM | 92.8 | 88.7 | 89.8 |
| InternVL2-76B | 92.2 | 88.8 | 89.5 |
| **Qwen2-VL-72B** | **93.2** | **90.1** | **89.9** |
| Specialist UNINEXT-H | 92.6 | 85.2 | 88.7 |

**Qwen-VL 初代から RefCOCO val で +3.8 ポイント改善**（89.4 → 93.2）、汎用 MLLM のグラウンディングで新 SOTA、専門モデル UNINEXT-H すら上回る。動的解像度による「高解像度画像内の細部知覚」の効果が顕著。

### エージェント能力（表 5）

| Benchmark | 従来 SoTA | GPT-4o | Qwen2-VL-72B |
| --- | --- | --- | --- |
| Function Calling TM | - | 90.2 | **93.1** |
| Function Calling EM | - | 50.0 | **53.2** |
| AITZ（UI 操作）TM | 83.0 | 70.0 | **89.6** |
| AITZ EM | 47.7 | 35.3 | **72.1** |
| Number Line | 89.4 | 91.5 | **100.0** |
| EZPoint | 50.0 | 85.5 | **100.0** |
| ALFRED SR（家事ロボット） | 67.7 | - | **67.8** |
| R2R（ナビ） | **79.0** | 43.7 | 51.7 |

**UI 操作で GPT-4o を圧倒**（AITZ EM 72.1 vs 35.3）、これは UI 上の細粒度グラウンディング能力の効果。VLN（R2R）では専門モデルに大きく後れ（51.7 vs 79.0）、3D 空間モデリングが弱点として残る。

### アブレーション

#### 動的解像度（表 7）

| Strategy | Avg Tokens | InfoVQA | OCRBench | MMMU |
| --- | --- | --- | --- | --- |
| Fixed 64 | 64 | 28.85 | 572 | 53.33 |
| Fixed 576 | 576 | 65.72 | 828 | 52.78 |
| Fixed 1600 | 1600 | 74.99 | 824 | 52.89 |
| Fixed 3136 | 3136 | 77.27 | 786 | 53.44 |
| **Dynamic** | **1924** | **75.89** | **866** | **53.44** |

**動的解像度は固定 3136 トークン版より少ないトークン（1924）で OCRBench で大きく上回る**。「タスクごとに最適解像度が異なる」ことを実証。

**重要な発見**: 「単に画像サイズを大きくすれば良いわけではない」。OCRBench の小画像を過度にアップスケールすると out-of-distribution に陥り性能低下。MMMU の性能上限は解像度ではなく推論能力。

#### M-RoPE（表 8）

| | MathVista | MMB | DocVQA | NextQA | STAR |
| --- | --- | --- | --- | --- | --- |
| 1D-RoPE | 39.2 | 58.6 | 82.5 | 43.9 | 55.5 |
| **M-RoPE** | **43.4** | **60.6** | **82.8** | **46.0** | **57.9** |

画像・動画ベンチマークで M-RoPE が一貫して 1D-RoPE を上回る。**動画ベンチで特に効果大**（NextQA +2.1、STAR +2.4）。

#### 長さ外挿（図 5）

学習時 16K トークン上限なのに **推論時 80K トークンまで Qwen2-VL-72B が頑健**な性能を維持。M-RoPE の位置 ID 値を小さく抑える設計の効果が直接実証された。

## 限界・批判的視点

- **MMMU で GPT-4o に -4.6 ポイント後れ**（64.5 vs 69.1）。大学レベルの推論集合体では商用に追いつけていない（[[entities/internvl-2-5|InternVL 2.5]] 70.1 / [[entities/internvl-3|InternVL 3]] 72.2 が後に超える）
- **MathVision で 25.9**（GPT-4o 30.4 に -4.5）。複雑な数学コンテスト問題でも後れ
- **VLN（R2R）で 51.7**（専門モデル 79.0 に大敗）。3D 空間モデリング・マップ理解は依然として弱点
- **アラビア語 OCR で GPT-4o に -5.2** （唯一の言語で GPT-4o 優位）。右から左への書字系の特殊性が学習データに反映されていない可能性
- **Video-MME 1 時間級動画**で評価時の最大フレーム抽出を 768 に制限したため、長動画の真の性能上限は不明
- **動画あたり 16K トークン上限**は 20 分動画なら扱えるが、それ以上の長尺コンテンツ（映画レベル）には不十分
- **依然として動画は frame-based 2 fps サンプリング**で時間解像度に上限がある。Qwen2.5-VL ではこれが動的フレームレートに拡張される
- **InternVL 3 が後に提唱する Native Multimodal Pre-Training とは対照的**な、伝統的「LLM 事前学習 → MLLM 適応」のパラダイムを継続。これは保守的設計で安定性と引き換え

## CV 分野における意義

Qwen2-VL は **「MLLM で『任意解像度・任意アスペクト比』を初めて本格的に実現したオープンソース MLLM」** として位置付けられる。これは、(i) 動的解像度のアプローチを「タイル分割（InternVL 1.5 系）」から「ViT 自体の任意解像度対応（2D-RoPE）」へと進化させ、(ii) **M-RoPE による時間・空間の分離**を MLLM 業界に持ち込み（後の Qwen2.5-VL や多くの後続モデルが採用）、(iii) **長文脈外挿（16K 学習 → 80K 推論）**という大規模 MLLM の実用性を担保したという意味で、CV 史における重要な転換点である。**Naive Dynamic Resolution と M-RoPE は MLLM 業界の事実上の標準として広く採用**された。

[[entities/qwen-vl|Qwen-VL]] 初代の `<box>`/`<ref>` 特殊トークンが「位置語彙不要のグラウンディング」を業界標準にしたのと同様に、Qwen2-VL は「**ViT-2D-RoPE + M-RoPE による動的解像度**」を業界標準にした。InternVL シリーズと並ぶ Alibaba 系の MLLM 系譜として、その後の Qwen2.5-VL、Qwen3-VL（将来）に直接繋がる基盤を作った。

## 用語と略称

- **LVLM** = Large Vision-Language Model（大規模視覚言語モデル）。**MLLM**（Multimodal LLM）とも
- **2B / 7B / 72B** = 20 億 / 80 億 / 720 億パラメータの 3 サイズ
- **M-RoPE** = Multimodal Rotary Position Embedding（マルチモーダル回転位置埋め込み、温度・高さ・幅の 3 成分）
- **2D-RoPE** = 2 次元回転位置埋め込み（縦横 2 軸に分解した [[concepts/rotary-position-embeddings|RoPE]]）
- **1D-RoPE** = 通常の 1 次元 RoPE（LLM での標準）
- **Naive Dynamic Resolution** = 素朴な動的解像度。「タイル分割せず、ViT 自体を任意解像度対応にする」設計
- **DFN** = Data Filtering Networks（Apple の高品質 CLIP 系視覚エンコーダ）。Qwen2-VL の ViT は DFN 系で初期化
- **3D 畳み込み** = 動画の時間軸 + 空間 2 軸の 3 次元に対する畳み込み。Qwen2-VL では深さ 2（連続 2 フレーム）
- **3D チューブ** = 2D パッチを時間方向に拡張した立体的トークン単位
- **fps** = frames per second（動画の単位時間あたりフレーム数）。Qwen2-VL は 2 fps でサンプリング
- **ChatML** = OpenAI 由来の対話マークアップ形式（`<|im_start|>`, `<|im_end|>` で発言を区切る）
- **AITZ** = Android In The Zoo（モバイル UI 操作のベンチマーク）
- **ALFRED** = Action Learning From Realistic Environments and Directives（家事ロボット・ベンチマーク）
- **R2R / REVERIE** = Room-to-Room / 視覚言語ナビゲーション・ベンチマーク
- **AI2THOR** = Allen Institute の家庭内シミュレータ
- **SAM** = Segment Anything Model（[[entities/sam]]）
- **ChartQA / DocVQA / InfoVQA / TextVQA / OCRBench / MTVQA** = 文書・チャート・テキスト系 VQA ベンチマーク群
- **MMMU** = Massive Multi-discipline Multimodal Understanding（大学レベル多分野マルチモーダル）
- **MMBench-EN/CN/V1.1** = 中英バイリンガル + 改良版を含む MLLM 評価
- **MME** = Multimodal Evaluation（知覚 + 認知の 14 サブタスク）
- **MMStar / MMVet / MMT-Bench / HallBench** = MLLM 評価ベンチマーク群
- **VCR** = Visual Commonsense Reasoning（en easy / zh easy 含む）
- **RealWorldQA** = 実世界空間理解の VQA ベンチ（X 社が公開）
- **MathVista / MathVision** = 数学推論ベンチマーク
- **MVBench / PerceptionTest / EgoSchema / Video-MME / NextQA / STAR** = 動画理解ベンチマーク群
- **TM / EM** = Type Match / Exact Match（関数呼び出し評価指標）
- **SR / GC** = Success Rate / Goal-Condition success rate（ALFRED 等エージェント指標）
- **DP / TP / PP / SP** = Data / Tensor / Pipeline / Sequence Parallelism
- **1F1B** = 1 forward 1 backward（パイプライン並列の標準スケジューリング）
- **CPFS / OSS** = Alibaba Cloud の並列ファイルストレージ / オブジェクトストレージ
- **PAI-Lingjun** = Alibaba Cloud の AI 訓練プラットフォーム

## 関連ページ

- [[entities/qwen2-vl]] — Qwen2-VL 2B/7B/72B のエンティティ・ページ
- [[translations/qwen2-vl]] — 本論文の本文翻訳
- [[entities/qwen-vl]] — 前世代 Qwen-VL（直接の系譜）
- [[concepts/rotary-position-embeddings]] — RoPE の基礎（M-RoPE の前提）
- [[concepts/foundation-model]] — 視覚言語基盤モデル
- [[concepts/weakly-supervised-pretraining]] — Web クロール画像-テキスト対学習
- [[concepts/zero-shot-transfer]] — 多言語 OCR・参照表現でのゼロショット転移
- [[concepts/alignment-tuning]] — 動的解像度時代の SFT
- [[entities/internvl-1-5]] — タイル分割で動的解像度を先行実装した競合系
- [[entities/internvl-2-5]] — InternVL 系の Progressive Scaling（Qwen2-VL の 1/12 訓練トークンで超えると主張）
- [[entities/internvl-3]] — Native Multimodal Pre-Training の InternVL 系後継
- [[entities/internvl-3-5]] — Cascade RL + MoE の InternVL 系最新版
