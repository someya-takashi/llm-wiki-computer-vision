---
type: source
source_path: raw/papers/Foundational Models Defining a New Era in Vision_ A Survey and Outlook.md
source_kind: paper
title: "Foundational Models Defining a New Era in Vision: A Survey and Outlook"
authors: [Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Fahad Shahbaz Khan]
year: 2023
venue: arXiv 2307.13721 / IEEE TPAMI
ingested: 2026-05-31
tags: [survey, foundation-model, vision-language, vlm, sam, clip, taxonomy, embodied-ai]
translation: "[[translations/foundational-models-vision-survey]]"
---

# Foundational Models Defining a New Era in Vision: A Survey and Outlook（視覚における新時代を定義する基盤モデル：サーベイと展望）

> 原典: [[translations/foundational-models-vision-survey]] ・ `raw/papers/Foundational Models Defining a New Era in Vision_ A Survey and Outlook.md`
> 著者: Muhammad Awais, Muzammal Naseer, Salman Khan, Rao M. Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Fahad Shahbaz Khan
> 所属: MBZUAI / ANU / Linköping / UCF / UC Merced・Yonsei・Google Research
> 出典: arXiv:2307.13721（2023 年 7 月 v1）、IEEE TPAMI 掲載
> リポジトリ: https://github.com/awaisrauf/Awesome-CV-Foundational-Models

## 一言まとめ

**CV における基盤モデルを「テキスト・プロンプト型」「視覚プロンプト型」「異種モダリティ型」「身体性型」の 4 軸で体系的に分類し、CLIP・SAM・PaLM-E など 100+ モデルを共通の分類体系（taxonomy）に位置付けた、視覚-言語基盤モデル領域の重要なサーベイ**。本 wiki に既に登録済みの CLIP / SAM / SAM 2 / SAM 3 / DINOv2 / SigLIP / Grounding-DINO / GLIP などすべてが**この体系のどこに位置するかを明示**することで、wiki 全体に「鳥瞰図」を与える。

## 背景と問題意識

### サーベイが書かれた時点での状況（2023 年中期）

2023 年中期は CV 基盤モデルの「**カンブリア爆発**」の真っ只中だった：

- **CLIP**（OpenAI, 2021）が CV にゼロショット転移を持ち込んでから 2 年
- **SAM**（Meta, 2023 Apr）が登場してわずか 3 ヶ月
- **GPT-4V**（OpenAI, 2023 Sep）が登場する直前
- **LLaVA / MiniGPT-4 / InstructBLIP** などの対話型 VLM が一斉に出現
- **PaLM-E**（Google, 2023 Mar）が embodied AI を foundation model 時代に持ち込んだばかり

この爆発的な多様化により、「**何がどう違うのか、何と何が比較可能か**」が研究者にとって追えなくなった。本サーベイはこの混乱期に**統一的な分類体系**を提供することで「地図」を作る試み。

### 既存サーベイとの差別化

- **[180]**: VL 事前学習モデルの予備的レビュー（タスク定義と一般アーキテクチャ）
- **[73]**: 画像・テキスト埋め込み符号化と事前学習アーキテクチャ
- **[299]**: 幾何学的・トポロジカル視点からの Transformer
- **[364]**: 自己教師ありマルチモーダル学習
- **[132][84]**: VL 事前学習フレームワーク分類
- **[331]**: SAM とその下流タスク

これら既存サーベイは個別側面に焦点を当てる。本サーベイの独自性は **3 つの基盤モデル・クラス**（テキスト・プロンプト型・視覚プロンプト型・異種モダリティ型）を **包括的に**扱い、さらに**身体性エージェント**まで地続きで論じる点。

## 提案手法 / 主張：4 軸分類体系（最重要貢献）

本サーベイの中核貢献は **「foundation model in vision」の分類体系**。図 3 の taxonomy が体系の鳥瞰図。

### 軸 1: テキスト・プロンプト型（Sec. 3-4）

**「言語を主要監督源とするモデル」**。3 サブカテゴリ：

1. **対比型（Contrastive Learning, CL）**: [[entities/clip|CLIP]], ALIGN, [[entities/siglip|SigLIP]], OpenCLIP, EVA-CLIP, FLIP, FILIP, SLIP, MaskCLIP（2 種類あり混在しやすい）
2. **生成型（Generative）**: Frozen, Flamingo, KOSMOS-1/2, SimVLM, PaLI, mPLUG-OWL
3. **ハイブリッド（CL + 生成）**: FLAVA, BLIP, BLIP-2, InstructBLIP, CoCa, X-FM, UNITER
4. **対話型 VLM**（Sec. 4 で独立扱い）: **GPT-4V**, MiniGPT-4, LLaVA, Video-ChatGPT, LLaMA-Adapter V2, XrayGPT

**テキスト・プロンプト型はさらに「汎用 vs 視覚グラウンディング」で 2 分割**:
- **汎用**: CLIP, ALIGN, BLIP 系（分類・検索が得意、局所化が弱い）
- **視覚グラウンディング**: [[entities/glip|GLIP]], [[entities/grounding-dino|Grounding-DINO]], RegionCLIP, CRIS, OWL-ViT, OpenSeg, GroupViT, [[entities/yolo-world|YOLO-World]]（フレーズ・領域対応が得意）

> **wiki への含意**: 既に登録済みの [[entities/clip]] / [[entities/glip]] / [[entities/grounding-dino]] / [[entities/yolo-world]] / [[entities/siglip]] / [[entities/qwen-vl]] / [[entities/qwen2-vl]] / [[entities/internvl]] / [[entities/gemma-3]] などはすべてこの軸 1 の下位カテゴリに属する。**Qwen-VL / InternVL / Gemma 3 / DeepSeek-OCR は本サーベイ後の世代だが、軸 1-4「対話型 VLM」の正統な後継**として位置づけられる。

### 軸 2: 視覚プロンプト型（Sec. 5）

**「点・ボックス・マスク・テキストなど多様なプロンプトを受け付け、セグメンテーションを返すモデル」**。代表例：
- **[[entities/sam|SAM]]**（中核）
- **CLIPSeg**（CLIP 流用、ゼロショット）
- **SegGPT**（in-context 学習で多タスク統一）
- **SEEM**（マルチプロンプト統一）

**SAM の下流応用も体系的に整理**:
- 医療: MedSAM, AutoSAM, 3DSAM-adapter, MSA, DeSAM, MedLAM, SAMM
- 追跡: TAM, SAM-Track, SAM-PT, SAM-DA
- リモートセンシング: RsPrompter
- モバイル: MobileSAM, FastSAM, RefSAM
- キャプション: CAT

**汎用主義（Generalist）モデル**として Painter, VisionLLM, Prismer も視覚プロンプト型の延長で扱う。

> **wiki への含意**: 軸 2 は本 wiki の **[[entities/sam]] / [[entities/sam-2]] / [[entities/sam-3]]** 系列、および [[concepts/promptable-segmentation]] / [[concepts/promptable-concept-segmentation]] の理論的位置付けを与える。サーベイ時点では SAM 2 / SAM 3 はまだ存在しないが、SAM の「PVS（Promptable Visual Segmentation）タスク」は軸 2 の代表的 task formulation として確立されている。

### 軸 3: 異種モダリティ型（Sec. 6）

**「画像-テキスト以外のモダリティ対（動画-音声、画像-深度、画像-IMU など）も整列する」モデル**。代表例：
- **CLIP2Video**（CLIP 拡張、動画用）
- **AudioCLIP**（音声追加、3 モダリティ）
- **ImageBind**（6 モダリティを画像を中心軸として整列）
- **MACAW-LLM**（画像・動画・音声・テキストの LLM 統合）
- **COSA**（画像-テキストから動画-段落を動的合成）
- **Valley**（video assistant + LLM）

> **wiki への含意**: 軸 3 は **[[entities/qwen3-5-omni|Qwen3.5-Omni]]**（音声・視覚・テキスト統合）の系譜的位置付けに直結。Qwen3.5-Omni は ImageBind / MACAW-LLM の正統な後継として「**Thinker-Talker 構造で音声出力まで含めた end-to-end omni model**」を実現した。

### 軸 4: 身体性型（Sec. 7）

**「視覚・言語を実世界の物理センサと結びつけ、ロボティクス・ナビゲーションに使う」モデル**。代表例：
- **PaLM-E**（Google, ロボット操作）
- **ViMA**（マルチモーダル・プロンプトでロボット制御）
- **MineDojo**（Minecraft API、データ収集）
- **VOYAGER**（GPT-4 駆動の終身学習エージェント）
- **LM-Nav**（CLIP + GPT-3 + ViNG でゼロショット・ナビゲーション）

> **wiki への含意**: 軸 4 は本 wiki ではまだ未開拓だが、**[[entities/qwen3-vl|Qwen3-VL]] や [[entities/internvl-3|InternVL 3]] が示し始めた「VLM が GUI / OSWorld でエージェントとして機能する」流れは、Sec. 7 の身体性型の論理的延長**。Qwen2.5-VL の ScreenSpot Pro 43.6、Qwen3-VL の OSWorld 38.1 などは「**仮想環境における身体性**」と読める。

### 4 つのアーキテクチャ・スタイル（直交する分類）

軸 1-4 とは別に、**実装アーキテクチャ**で 4 種を区別（図 2）：

| スタイル | 説明 | 代表例 |
|---|---|---|
| **Dual-Encoder** | 視覚エンコーダとテキスト・エンコーダが並列、出力を整列 | CLIP, ALIGN, SigLIP |
| **Fusion** | 視覚・テキスト表現を融合デコーダで結合 | GLIP, BLIP, CRIS |
| **Encoder-Decoder** | 共同特徴符号化と復号を逐次 | SimVLM, MetaLM, PaLI |
| **Adapter LLM** | 視覚エンコーダを LLM のアダプタとして統合 | Flamingo, KOSMOS, BLIP-2, mPLUG-OWL, LLaVA, MiniGPT-4 |

> **補足**: **Adapter LLM** が wiki 既存の Qwen-VL / InternVL / Gemma 3 / LLaVA 系の主要パターン。BLIP-2 の Q-Former、Qwen-VL の Position-aware VL Adapter、InternVL の MLP projector、Gemma 3 の SigLIP+P&S+pooling などはすべて **Adapter LLM** カテゴリ内の異なる実装。

## 実験結果と知見

サーベイ論文のため独自の数値実験はないが、以下の**経験的観察**が散りばめられている：

### CL モデルのスケーリング知見
- **CLIP**: 400M 対 → 強力なゼロショット
- **OpenCLIP × LAION-5B**: 58 億対までデータ・モデル・計算を線形にスケールできる
- **CLIPA**: **逆スケーリング則** — より大きなモデルはより小さなトークン・サイズで訓練しても精度低下が少ない（後の研究で示された）
- **FLIP**: 50-75% マスキングで 2-4× 高速化、精度向上

### グラウンディング・モデル知見
- 単純な CLIP は局所化に弱い（図 5）— OpenSeg は「グルーピング→整列」の順序が重要と指摘
- GLIP: phrase grounding を物体検出として再定式化することでスケーラブル事前学習が可能に
- Grounding-DINO: closed-set DETR を言語で grounding することで open-set 検出を達成

### SAM の汎化性
- 11M 画像 + 11 億マスクで訓練、**CLIP に依存せず**ゼロショット・セグメンテーション
- 自然画像 → 医療画像のドメイン・ギャップは大きく、Adapter / 追加プロンプト / decoder 修正など多様な適応戦略が必要

### Adapter LLM の利便性
- **Frozen** が示した「LLM を凍結したまま視覚エンコーダだけ訓練」パターンが、その後のすべての対話型 VLM の基本設計を確立
- BLIP-2 の Q-Former、Flamingo の Perceiver Resampler はこの系譜

## 限界・批判的視点

### 1. 時期的限界（2023 年中期執筆）

サーベイ後に登場した重要モデルが含まれていない：
- **GPT-4V**（2023 Sep）の本格的議論なし
- **[[entities/qwen-vl|Qwen-VL]]**（2023 Aug、ぎりぎり同時期）以降の中国系 MLLM の爆発（[[entities/internvl|InternVL]] / [[entities/qwen2-vl|Qwen2-VL]] / [[entities/gemma-3|Gemma 3]] / [[entities/deepseek-ocr|DeepSeek-OCR]]）
- **[[entities/dinov2|DINOv2]]**（2023 Apr）/ **[[entities/dinov3|DINOv3]]**（2025）など **純粋 SSL 系基盤モデル**は対象外（本サーベイは VL モデルに限定）
- **[[entities/sam-2|SAM 2]] / [[entities/sam-3|SAM 3]]** はまだ存在しない
- **[[entities/sdxl|SDXL]]** など拡散モデル系はスコープ外（Sec. 2.1 で明示的に除外）

### 2. 「マルチモーダル＝ VL」の暗黙の前提

「foundation models in vision」と銘打つが、実態は **vision-language foundation models** に強く偏重。**純粋画像 SSL**（DINOv2, MAE 等）は事実上扱われていない（Sec. 2.1 で生成モデルと共に明示的にスコープ外と宣言）。これは本 wiki が並列で扱う **[[entities/dinov2]] / [[entities/dinov3]] / [[entities/mae]] / [[entities/ibot]]** などの SSL 系統と相補的な関係。

### 3. 評価方法論への批判が浅い

「LLM-as-a-judge」が議論的とは認めるが、より深い代替提案（例：人間評価コスト・GPT-4 バイアス・自動評価との整合）は薄い。後の研究（[[sources/perception-encoder|Perception Encoder]] の独立評価設計、[[sources/revisiting-ssl-foundation-models|Revisiting SSL]] の VFM 時代評価）の方が実践的。

### 4. 分類の重なり

軸 1-4 は必ずしも排他的でなく、たとえば：
- **LLaVA / Qwen-VL** は軸 1（テキスト・プロンプト対話型）と軸 4（GUI エージェント）の両方
- **ImageBind** は軸 3 だが、CLIP の拡張なので軸 1 にも近い
- **PaLM-E** は軸 4 だが、入力はマルチモーダルなので軸 3 にも掛かる

実装的には「Adapter LLM + マルチ目的訓練」がほぼ普遍化したことで、純粋な軸の区別は曖昧化している。

### 5. データセット記述の偏り

事前学習データセットは表 II/IV にまとまっているが、SA-1B（11M 画像 × 1.1B マスク）の異質性（**画像数より遥かに多いマスク数**）など、データ・スケールの非対称性についての分析が浅い。

## 用語と略称

- **FM**: Foundation Model
- **VLM**: Vision-Language Model
- **MLLM**: Multimodal Large Language Model
- **VL**: Vision-Language
- **CL**: Contrastive Learning（対比学習）
- **ITC**: Image-Text Contrastive（画像-テキスト対比損失）
- **ITM**: Image-Text Matching（画像-テキスト一致損失）
- **MLM**: Masked Language Modeling
- **LM**: Language Modeling（自己回帰）
- **MIM**: Masked Image Modeling
- **PrefixLM**: Prefix Language Modeling
- **ITG**: Image-grounded Text Generation
- **Cap**: Captioning loss
- **MMM**: Masked Multimodal Modeling
- **CapPa**: Captioning with Parallel prediction
- **TPC**: Text-to-Pixel Contrastive（CRIS）
- **RWA / RWC**: Region-Word Alignment / Contrastive（GLIP, GLIPv2）
- **MITC**: Multi-label Image-Text Contrastive（GroupViT）
- **UniCL**: Unified Contrastive Learning（Florence）
- **PVS**: Promptable Visual Segmentation（SAM タスク、本 wiki [[concepts/promptable-segmentation]]）
- **PCS**: Promptable Concept Segmentation（SAM 3 で定義、本 wiki [[concepts/promptable-concept-segmentation]]）
- **RPN**: Region Proposal Network
- **RoI**: Region of Interest
- **VOS / VIS**: Video Object Segmentation / Video Instance Segmentation
- **RVOS**: Referring Video Object Segmentation
- **PEFT**: Parameter-Efficient Fine-Tuning（[[concepts/parameter-efficient-fine-tuning]]）
- **LoRA / QLoRA / GLoRA**: Low-Rank Adaptation（PEFT 手法）
- **RLHF**: Reinforcement Learning from Human Feedback
- **CoT**: Chain-of-Thought（思考連鎖プロンプティング）
- **IMU**: Inertial Measurement Unit（慣性計測装置、ImageBind の対象モダリティの一つ）

## 関連ページ

### この survey が位置付ける wiki 既存のモデル

**軸 1: テキスト・プロンプト型**
- [[entities/clip]] — CL 汎用、survey の中核モデル
- [[entities/siglip]] — CL 汎用、CLIP 後継、softmax → sigmoid 改良
- [[entities/perception-encoder]] — CL の頑健スケーリング、survey 後の SOTA
- [[entities/glip]] — CL 視覚グラウンディング、phrase grounding を物体検出に転換
- [[entities/grounding-dino]] / [[entities/grounding-dino-1-5]] / [[entities/dino-x]] — CL 視覚グラウンディング、DETR 系言語拡張
- [[entities/yolo-world]] — CL 視覚グラウンディング、YOLO 系言語拡張（survey 後）
- [[entities/qwen-vl]] / [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]] — 対話型 VLM（Sec. 4）の正統な後継
- [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]] / [[entities/internvl-3-5]] — Adapter LLM 系の継続発展
- [[entities/gemma-3]] — 対話型 VLM、Google 系 Adapter LLM
- [[entities/deepseek-ocr]] — OCR 特化 Adapter LLM（survey 後）

**軸 2: 視覚プロンプト型**
- [[entities/sam]] — survey の中核、PVS タスクを確立
- [[entities/sam-2]] / [[entities/sam-3]] — survey 後、動画拡張と PCS 追加

**軸 3: 異種モダリティ型**
- [[entities/qwen3-5-omni]] — ImageBind / MACAW-LLM の正統な後継、音声出力まで含む

**軸 4: 身体性型**（wiki ではまだ entity ページなし、本 survey で参照される PaLM-E / ViMA / MineDojo / VOYAGER / LM-Nav）

### survey の分類体系を補強する concepts

- [[concepts/foundation-model]] — 基盤モデルの定義、本 survey の分類体系を統合
- [[concepts/contrastive-learning]] — 軸 1.1 の理論的基盤
- [[concepts/weakly-supervised-pretraining]] — CLIP / SigLIP 系の事前学習パラダイム
- [[concepts/zero-shot-transfer]] — CLIP が CV に持ち込んだプロンプト推論
- [[concepts/promptable-segmentation]] — SAM の PVS、軸 2 の中核
- [[concepts/promptable-concept-segmentation]] — SAM 3 の PCS、軸 2 の拡張（survey 後）
- [[concepts/alignment-tuning]] — Perception Encoder が定式化（survey 後）
- [[concepts/parameter-efficient-fine-tuning]] — FM 適応の中核技法、Sec. 8 で言及

### 関連 question / 比較ページ

- [[questions/vit-dynamic-resolution-evolution]] — タイル分割路線（LLaVA → InternVL 1.5 → Gemma 3 → DeepSeek-OCR）の系譜整理。survey の Adapter LLM 系の実装的進化を追跡する補助線

### survey のスコープ外（純粋画像 SSL）

- [[entities/dino]] / [[entities/dinov2]] / [[entities/dinov3]] — DINO 系列、純粋 SSL
- [[entities/mae]] / [[entities/ibot]] — MIM 系列
- [[entities/byol]] / [[entities/moco]] — 対比 SSL の先駆け
- [[sources/revisiting-ssl-foundation-models]] — VFM 時代における SSL/SeSL 再検討（survey 後の発展）
- [[entities/sdxl]] — 拡散モデル系（survey が明示的にスコープ外と宣言した領域）
