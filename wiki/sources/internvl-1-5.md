---
type: source
source_path: raw/papers/How Far Are We to GPT-4V_ Closing the Gap to Commercial Multimodal Models with Open-Source Suites.md
source_kind: paper
title: "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites"
authors: [Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang]
year: 2024
venue: "arXiv:2404.16821 (technical report)"
ingested: 2026-05-29
tags: [vision-language, mllm, vllm, gpt-4v, internvl-1-5, internvit, dynamic-high-resolution, bilingual, opengvlab, openai-gap]
translation: "[[translations/internvl-1-5]]"
---

# InternVL 1.5 — オープンソース MLLM で GPT-4V との差を初めて明確に縮めた論文

> 原典: [[translations/internvl-1-5]] ・ `raw/papers/How Far Are We to GPT-4V_ Closing the Gap to Commercial Multimodal Models with Open-Source Suites.md`
> 著者: Zhe Chen, Weiyun Wang, Hao Tian ら（Shanghai AI Lab + SenseTime + 清華大学 + 南京大学 + 復旦大学 + 香港中文大学）
> 投稿: arXiv:2404.16821（2024 年 4 月、technical report）
> Demo: <https://internvl.opengvlab.com>
> リポジトリ: <https://github.com/OpenGVLab/InternVL>
> モデル: <https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5>

---

## 一言まとめ

**「[[entities/internvl|InternVL 1.0]] (CVPR 2024) を MLLM 専用に再設計し、GPT-4V / Gemini Pro / Claude-3 / Qwen-VL-Max との差を OCR・中国語タスクで初めて埋めたオープンソース 26B MLLM」**。3 つの改善（**継続学習で InternViT-6B を強化** + **動的高解像度 4K** + **高品質バイリンガル（英中）データセット**）だけで、26B（InternViT-6B + InternLM2-20B）という比較的小サイズで **18 ベンチマーク中 8 つで SoTA**。**ChartQA / OCRBench で全商用モデル超え**。「**Closing the Gap to GPT-4V**」のキャッチコピーで MLLM 競争の風向きを変えた、現在の InternVL シリーズ（2.0/2.5/3）の直接の出発点。

---

## 背景と問題意識

### オープンソース MLLM と商用 MLLM の 3 つの隔たり

2024 年初頭時点、**GPT-4V / Gemini Pro / Claude-3 Opus / Qwen-VL-Max** といった独自商用 MLLM は、オープンソースモデルを **明確に上回って** いた。著者らはその原因を 3 つに分解：

| 隔たり | 商用 MLLM | オープンソース MLLM（〜2024 春） |
|---|---|---|
| **パラメータ規模** | LLM ≥ 100B、視覚 unknown | VFM 300M（CLIP-L 級）+ LLM 7B-13B |
| **画像解像度** | dynamic（GPT-4V の "low/high" 等、原アスペクト比保持） | 固定 336² or 448²（[[entities/internvl\|InternVL 1.0]] は 224²/336²） |
| **多言語性** | 多言語データで訓練 | 主に英語、他言語は LLM のゼロショット頼み |

> **補足: MLLM（Multimodal Large Language Model）と VLLM** — 用語は微妙に違うが本論文では interchangeable に使われる。両者とも「LLM + 画像入力」を指し、本論文は主に **MLLM** を採用。GPT-4V / Gemini / Claude-3V のような **商用フロンティアモデル** がベンチマークの上限を形成し、オープンソース陣営はそれを追う構図。

### この論文の位置づけ

**「How Far Are We to GPT-4V?」** という挑発的タイトルは、当時の MLLM コミュニティの空気を端的に表す。LLaVA-NeXT（35B, 2024 早春）/ Mini-Gemini（35B）/ DocOwl-1.5（8B）等が「商用に追いつこう」と次々に登場した時期。本論文は **「26B というモデルサイズで、特定の重要タスク（OCR・中国語）では商用を上回れる」** ことを実証し、論争に明確な答えを出した。

---

## 提案手法

### 全体アーキテクチャ: ViT-MLP-LLM（LLaVA 系標準）

```
入力画像（任意解像度）
   ↓ 動的解像度マッチング（35 通り）
448×448 タイル × N (N=1〜40) + サムネイル
   ↓ InternViT-6B（凍結 or 訓練、5.9B）
1024 visual tokens / tile
   ↓ Pixel Shuffle（1/4 圧縮）
256 visual tokens / tile
   ↓ MLP プロジェクタ（ランダム初期化）
LLM トークン空間
   ↓ InternLM2-20B-Chat
出力テキスト
```

[[entities/internvl|InternVL 1.0]] が持っていた **QLLaMA（8B 言語ミドルウェア）を完全に廃止**し、LLaVA 系の標準アーキテクチャ「ViT-MLP-LLM」に統一。これは「**glue layer を巨大化する InternVL 1.0 路線**」と「**シンプルな MLP でいい LLaVA 路線**」の論争に対し、InternVL チーム自身が後者に転向したことを意味する重要な設計選択。

> **補足: なぜ QLLaMA を捨てたか** — InternVL 1.0 の QLLaMA（8B）は「対比 + 生成 + LLM 接続を 1 モジュールで」という野心的設計だったが、**(a) 訓練コストが高い、(b) LLM を交換するたびに QLLaMA も再訓練が必要、(c) MLLM タスクでは MLP で十分** という事情から、1.5 では削除された。論文は「**InternViT が学習する視覚特徴は特定 LLM に強く束縛されない**」と明示し、可搬性を強調する。

### 3 つの主要改善

#### 1. 強力な視覚エンコーダ: InternViT-6B-448px-V1.5（継続事前学習）

InternViT-6B はバージョンごとに改良されてきた：

| バージョン | 解像度 | 層数 | LLM 連携 | 主要変更 |
|---|---|---|---|---|
| **V1.0**（InternVL 1.0 / CVPR 2024） | 224 | 48 | LLaMA-7B（QLLaMA 経由） | 元のオリジナル |
| **V1.2**（InternVL 1.2） | 448 | **45**（後ろ 3 層削除） | Nous-Hermes-2-Yi-34B | 「後ろから 4 層目の特徴が MLLM タスクで最良」と発見、最終 3 層を削除 |
| **V1.5**（InternVL 1.5 / 本論文） | **dynamic 448** | 45 | InternLM2-20B-Chat | 動的解像度 + バイリンガル + データ拡張 |

**V1.5 の key insight**: 
- 訓練解像度を **fixed 448 → dynamic 448**（1-12 タイル）に拡張
- LLM が **Nous-Hermes-2-Yi-34B → InternLM2-20B-Chat** に変更されても InternViT は無修正で適応 → **「視覚特徴は LLM に依存しない」** を実証

#### 2. 動的高解像度（Dynamic High-Resolution、本論文の中心的貢献）

UReader（2023）に着想を得た **タイル分割 + 動的アスペクト比マッチング**：

| ステップ | 内容 |
|---|---|
| **1. アスペクト比マッチング** | 1-12 タイルから構成される **35 通り** の事前定義アスペクト比から、入力画像に最も近いものを選ぶ |
| **2. リサイズ** | 例: 800×1300 → 896×1344（2:3 比率にマッチ） |
| **3. タイル分割** | 448×448 タイル × N 枚 |
| **4. サムネイル追加** | 全画像を 448×448 にダウンサンプル、グローバル文脈用 |
| **5. Pixel Shuffle** | 各タイル 1024 token → **256 token** に圧縮（1/4） |

**訓練時**: 1-12 タイル → **256-3,328 visual tokens**
**テスト時（ゼロショット）**: 最大 40 タイル → **10,496 visual tokens**（4K 解像度）

> **補足: なぜ Pixel Shuffle？** — Pixel Shuffle は超解像で使われる技法で、空間情報をチャンネル方向に詰め直す（H×W×C → H/2×W/2×4C）。InternVL 1.5 では「4 つの隣接 patch を 1 つの token に合体」させて token 数を 1/4 にする。1024 → 256 への削減により、LLM への入力負荷を 4 倍緩和。

#### 3. 高品質バイリンガルデータセット（英語 + 中国語）

事前学習データ（**ファインチューンではなく Vision Encoder 強化用**）：

| タスク | 比率 | 主要データセット |
|---|---|---|
| Captioning | 53.9% | LAION-EN, LAION-ZH, COYO (zh), GRIT (zh), COCO, TextCaps |
| Detection / Grounding | 5.2% | Objects365, GRIT, All-Seeing |
| OCR（大規模） | 32.0% | **Wukong-OCR (zh)**, **LaionCOCO-OCR (en)**, Common Crawl PDF |
| OCR（小規模） | 8.9% | MMC-Inst, LSVT, ST-VQA, RCTW-17, ReCTs, ArT, SynthDoG, ChartQA, CTW, DocVQA, TextOCR, PlotQA, InfoVQA |

特筆: **PaddleOCR で大規模に擬似 OCR ラベリング**（Wukong の中国語画像 + LaionCOCO の英語画像）。これがバイリンガル OCR 強化の核心。

**データ翻訳パイプライン**: OpenLLM（InternLM2 / Qwen-VL / Yi-34B）または GPT-3.5 で英語データを中国語化。プロンプトで「固有名詞は英語のまま」「専門用語は英語 + 中国語解説」「慣用句は文化適応」「略語は full form + 括弧内英語」を指示。

### 訓練の 2 段階

| Stage | 訓練対象 | データ | 目的 |
|---|---|---|---|
| **1. Pre-training** | InternViT-6B + MLP（LLM 凍結） | 表 1(a) | 視覚特徴抽出最適化 |
| **2. Fine-tuning** | **全 26B パラメータ** | 表 1(b)、約 4.2M 指示データ | マルチモーダル能力強化 |

InternVL 1.0 の 3 段階（contrastive → generative → SFT）から **2 段階に簡素化**。これは GPT-4V 後の MLLM 競争で「対比学習段階を独立させる必要は薄い、LLaVA 流の 2 段階で十分」という業界トレンドの追認。

---

## 実験結果と知見

### 主要結果: 18 ベンチマーク中 8 つで SoTA

#### OCR 関連（5 ベンチマーク、全勝 / 大健闘）

| ベンチマーク | InternVL 1.5 (26B) | 最強商用 | 結果 |
|---|---|---|---|
| **DocVQA** | **90.9** | Qwen-VL-Max 93.1 | -2.2（クロース） |
| **ChartQA** | **83.8** | Claude-3 Haiku 81.7 | **+2.1 SoTA** |
| **InfoVQA** | **72.5** | Gemini Ultra 1.0 80.3 | -7.8 |
| **TextVQA** | **80.6** | Gemini Ultra 1.0 82.3 | -1.7 |
| **OCRBench** | **724** | Qwen-VL-Max 723 | **+1 SoTA** |

ChartQA と OCRBench で **全商用モデルを上回る** という、26B サイズのオープンソースモデルとしては破格の結果。

#### 中国語関連（CCBench, MMBench-CN）

| ベンチマーク | InternVL 1.5 (26B) | GPT-4V | 結果 |
|---|---|---|---|
| **CCBench** | **69.8** | 46.5 | **+23.3** |
| **MMBench-CN** | **82.0** | 74.4 | **+7.6** |

**中国文化理解で GPT-4V を圧倒**。バイリンガルデータと中国語チーム発のメリットが最も発揮される領域。

#### 数学（MathVista）

| モデル | MathVista |
|---|---|
| Gemini Ultra 1.0 | 53.0 |
| **InternVL 1.5 (26B)** | **53.5** |
| Grok-1.5V | 52.8 |
| Gemini Pro 1.5 | 52.1 |
| Qwen-VL-Max | 51.0 |
| Claude-3 Opus | 50.5 |
| **GPT-4V** | **49.9** |

**GPT-4V を +3.6 ポイント上回る**。数学推論で初めてオープンソースが商用を抜いた瞬間。

#### Hallucination 制御（HallusionBench）

| モデル | HallB |
|---|---|
| **InternVL 1.5 (26B)** | **49.3** |
| Step-1V | 48.4 |
| **GPT-4V** | **46.5** |

物体ハルシネーション制御で SoTA。InternVL チームが特に重視した領域。

### アブレーション 1: 大きな LLM には大きな VFM が必要

| モデル | VFM | LLM | 計 |
|---|---|---|---|
| LLaVA-NeXT-34B | CLIP-L (300M) | LLaMA2-Yi-34B | 34.3B |
| **InternVL 1.2** | InternViT-6B | Nous-Hermes-2-Yi-34B | 40B |

OCR 関連 / ConvBench / RealWorldQA を除外した 11 ベンチマーク中、**InternVL 1.2 が 9 つで LLaVA-NeXT-34B を上回る**。条件は完全同一ではない（解像度・データ）が、**「34B LLM に対し 6B VFM が 300M VFM より明確に効く」** ことを示唆。

> **補足: LLM/VFM スケーリング則** — 過去の LLaVA 系研究は「LLM だけスケールすればよく VFM は 300M で十分」と主張していた。InternVL チームはこれに異を唱え、「LLM スケールに合わせて VFM もスケールせよ」と主張。これは [[entities/internvl|InternVL 1.0]] の主張の継続。

### アブレーション 2: 動的解像度は **タスク依存**

図 6 の知見：

- **OCR 系（DocVQA, InfoVQA, TextVQA, OCRBench）**: 解像度上昇とともに性能向上、12-24 タイル付近で頭打ち
- **AI2D, MMMU, MMBench, HallusionBench**: 高解像度で **わずかに低下**（理由: 高解像度で画像のグローバル context が薄まる）

→ **「OCR は高解像度を必要とするが、シーン推論には不要」** という重要な観察。これは後の MLLM 設計（NVLM, Qwen-VL2 等の dynamic resolution）に影響を与えた。

### 定性比較: vs GPT-4V

論文の §4.3.1 では 6 シナリオで詳細比較：

| シナリオ | InternVL 1.5 vs GPT-4V |
|---|---|
| **汎用 QA** | 同等。GPT-4V はプライバシー懸念で過剰拒否することがある |
| **中国語 OCR** | **InternVL 1.5 圧勝**（GPT-4V は中国語シーンの情報抽出に失敗） |
| **チャート理解** | 同等 |
| **科学的理解** | やや InternVL 1.5 優位（GPT-4V は推測に逃げる傾向） |
| **中国伝統文化** | **InternVL 1.5 優位**（より深い文化理解、布袋戏 vs 提线木偶の区別等） |
| **物体位置特定** | 同等 |
| **複数画像対話** | 同等（InternVL 1.5 は単一画像訓練のみだがゼロショットで対応） |

---

## 限界・批判的視点

### MMMU と MMT-Bench での退行

- InternVL 1.5 (26B, LLM=20B) は InternVL 1.2 (40B, LLM=34B) より **MMMU で -6.4 (51.6 → 45.2)、MMT-Bench で -4.4 (63.4 → 59.0)** 低下
- 著者自身がこれを **「LLM サイズ縮小に起因」** と認める
- **多分野知識タスクでは LLM パラメータが効く** という当然の事実。InternVL 1.5 は 26B 軽量化を取って MMMU を捨てた

### マルチターン対話で GPT-4V との差は残る

- ConvBench で GPT-4V (39.51) vs InternVL 1.5 (17.65) → **大差で負け**
- 「single-image 訓練だけで multi-image にゼロショット対応」と言いつつ、**真の多ターン対話はまだ限界**

### Closed-source 商用モデルとの真の比較は不可能

- GPT-4V / Gemini Pro 1.5 のパラメータ数・訓練データは未公開
- 18 ベンチマーク中 8 SoTA も、**ベンチマークの選び方** で印象が変わる（OCR と中国語を重視している）
- 「How Far」という問いに完全な答えはまだない

### 動的解像度の計算コスト

- 40 タイル + サムネイル = 41 × 256 = **10,496 visual tokens** を 4K テスト時に LLM に流す
- InternLM2-20B の文脈長制約と GPU メモリで実用上の制約大きい

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **InternVL 1.5** | – | InternVL シリーズの 4 代目（1.0 → 1.2 → 1.5）、CVPR 2024 でも内部世代 |
| **MLLM** | Multimodal Large Language Model | 画像入力可能な LLM、本論文での主呼称 |
| **VLLM** | Vision Large Language Model | MLLM と同義（混用） |
| **VFM** | Vision Foundation Model | CLIP-ViT / SigLIP / InternViT / DINOv2 等の総称 |
| **InternViT-6B-448px-V1.2** | – | InternViT 第 2 版（45 層、448 固定、Yi-34B 連携） |
| **InternViT-6B-448px-V1.5** | – | InternViT 第 3 版（45 層、448 dynamic、InternLM2-20B 連携） |
| **InternLM2-20B-Chat** | – | Shanghai AI Lab の独自 LLM、20B Chat 版（base 版ではなく） |
| **Pixel Shuffle** | – | 超解像由来の空間 → チャンネル変換。InternVL 1.5 では visual token を 1/4 圧縮 |
| **Dynamic High-Resolution** | – | 動的解像度。35 通りのアスペクト比 × 1-12 タイルから最適選択 |
| **35 種アスペクト比** | – | 1:1, 1:2, 2:1, 2:2, 1:3, 3:1, ..., 2:6 の 35 通り（1-12 タイル組合せ） |
| **Thumbnail** | – | グローバル文脈用に追加する 448×448 サムネイル |
| **Aspect Ratio Matching** | – | 入力画像のアスペクト比を 35 種から最近傍で選ぶ |
| **Continuous Learning** | – | InternViT を継続事前学習で強化する戦略（V1.0 → V1.2 → V1.5） |
| **ViT-MLP-LLM** | – | LLaVA 系の標準アーキテクチャ（QFormer ではなく MLP） |
| **GPT-4V** | – | OpenAI の GPT-4 with Vision（2023 秋） |
| **Gemini Ultra/Pro 1.0/1.5** | – | Google の Gemini 系列 |
| **Claude-3 Opus/Sonnet/Haiku** | – | Anthropic の Claude-3 ファミリー |
| **Qwen-VL-Max/Plus** | – | Alibaba の Qwen-VL 商用版 |
| **Grok-1.5V** | – | xAI の MLLM |
| **MM1** | – | Apple の 30B MLLM（2024 春） |
| **Step-1V** | – | StepFun の 100B MLLM |
| **HPT Pro** | – | HyperGAI の MLLM |
| **LLaVA-NeXT** | LLaVA Next | LLaVA-1.5 の後継（高解像度対応、2024 早春） |
| **Mini-Gemini** | – | 35B MLLM（2024） |
| **DocOwl-1.5** | – | 8B 文書 MLLM |
| **Text-Monkey** | – | 10B OCR MLLM |
| **MiniGPT-4** | – | LLaVA 系初期 MLLM |
| **CogVLM** | – | 智谱 AI の VLM |
| **Shikra** | – | grounding 機能付き VLLM |
| **VisionLLM** | – | grounding 機能付き VLLM |
| **ShareGPT4V** | – | GPT-4V で生成したバイリンガル指示データ |
| **DeepSeek-VL** | – | DeepSeek 系列（SigLIP-L + SAM-B の dual encoder） |
| **LLaVA-HR** | LLaVA High-Res | CLIP-ViT (low) + CLIP-ConvNext (high) の dual encoder |
| **mixture-of-features** | – | Tong et al. が提案した CLIP + DINOv2 結合（[[entities/clip\|CLIP]] + [[entities/dinov2\|DINOv2]] を合体） |
| **UReader** | – | タイル分割の祖（Ye et al., 2023） |
| **PaddleOCR** | – | Baidu の OCR エンジン、本論文の OCR データ生成に使用 |
| **DocVQA** | Document VQA | 文書理解ベンチマーク |
| **ChartQA** | Chart VQA | 図表理解 |
| **InfoVQA** | Infographic VQA | インフォグラフィック理解 |
| **TextVQA** | – | シーンテキスト理解 |
| **OCRBench** | – | OCR 総合評価 |
| **MMBench-EN/CN** | – | マルチモーダル総合ベンチマーク（英語/中国語） |
| **CCBench** | Chinese Culture Bench | 中国文化理解 |
| **MMMU** | Massive Multi-discipline Multimodal Understanding | 多分野理解 |
| **AI2D** | Allen Institute 2D | 科学図解理解 |
| **MathVista** | – | 数学的視覚推論 |
| **MME** | – | 14 サブタスクの総合認知 |
| **MMVet** | – | 視覚汎用能力ベンチマーク |
| **SEED** | – | 視覚生成・理解の総合評価 |
| **RealWorldQA** | – | 実世界の空間理解（Grok チームが公開） |
| **HallusionBench** | – | 視覚ハルシネーション評価 |
| **ConvBench** | – | マルチターン対話評価（Liu et al., 2024） |
| **MMT-Bench** | – | 162 subtask の包括ベンチマーク |
| **VLMEvalKit** | – | InternVL 1.5 評価用フレームワーク（OpenCompass 系） |
| **OpenCompass** | – | Shanghai AI Lab の LLM 評価フレームワーク |
| **Wukong-OCR** | – | Wukong（華為の中国語データ）+ PaddleOCR で生成 |
| **LaionCOCO-OCR** | – | LaionCOCO + PaddleOCR で生成 |
| **Common Crawl PDF** | – | Common Crawl の PDF 文書から抽出 |
| **GRIT** | – | Grounded Image-Text dataset（KOSMOS-2 系） |
| **All-Seeing** | – | Wang et al. 2023 の universal perception データ |
| **MMC-Inst** | MultiModal Chart Instruction | チャート指示データ |
| **LSVT** | Large-scale Street View Text | 街中文字データ（中国語） |
| **RCTW-17** | Reading Chinese Text in the Wild | 中国語シーンテキスト |
| **ReCTs** | Reading Chinese Text on Signboard | 看板中国語 |
| **ArT** | Arbitrary-Shaped Text | 任意形状テキスト |
| **SynthDoG** | Synthetic Document Generator | 合成文書 OCR データ |
| **COCO-Text** | – | COCO 内のテキスト |
| **CTW** | Chinese Text in the Wild | 中国語自然画像テキスト |
| **TextOCR** | – | OCR ベンチマーク |
| **PlotQA** | – | プロット VQA |
| **VSR** | Visual Spatial Reasoning | 空間推論 |
| **VisualDialog** | – | 視覚対話 |
| **ScienceQA** | – | 科学 QA |
| **TQA** | TextbookQA | 教科書 QA |
| **DVQA** | Diagram VQA | 図表 VQA |
| **LRV-Instruction** | – | ハルシネーション対策指示データ |
| **GeoQA+** | – | 幾何 QA |
| **TabMWP** | Tabular Math Word Problem | 表数学問題 |
| **MathQA** | – | 数学 QA |
| **CLEVR-Math/Super** | – | CLEVR 数学版 |
| **Geometry3K** | – | 幾何 3000 問題 |
| **KVQA** | Knowledge VQA | 知識 VQA |
| **A-OKVQA** | Augmented OK-VQA | 拡張 OK-VQA |
| **ViQuAE** | – | 質問 → エンティティ視覚 QA |
| **OCRVQA** | – | OCR + VQA |
| **RefCOCO/+/g** | – | Referring Expression Comprehension |
| **Visual Genome** | – | 領域注釈データ |
| **LLaVA-150K** | – | LLaVA の合成指示データ |
| **LVIS-Instruct4V** | – | LVIS 指示データ |
| **ALLaVA** | – | バイリンガル LLaVA 指示データ |
| **Laion-GPT4V** | – | LAION + GPT-4V 合成 |
| **TextOCR-GPT4V** | – | TextOCR + GPT-4V 合成 |
| **SVIT** | Scaling up Visual Instruction Tuning | 大規模指示データ |
| **OpenHermes2.5** | – | テキスト指示データ |
| **Alpaca-GPT4** | – | Alpaca + GPT-4 合成 |
| **ShareGPT** | – | ChatGPT 会話ダンプ |
| **COIG-CQIA** | – | 中国語指示データ |
| **Nous-Hermes-2-Yi-34B** | – | Yi-34B ベースのチャット LLM（InternVL 1.2 が使用） |
| **InternVL 1.0** | – | CVPR 2024 ([[entities/internvl]]) |
| **InternVL 1.2** | – | InternVL 1.0 と 1.5 の中間版（fixed 448、Yi-34B、40B 計） |
| **MLP プロジェクタ** | – | ViT 出力 → LLM 入力空間への線形変換（LLaVA 流） |
| **AGI** | Artificial General Intelligence | 汎用人工知能 |

---

## 関連ページ

- 翻訳: [[translations/internvl-1-5]]
- 主要エンティティ: [[entities/internvl-1-5]]
- 直接の祖: [[sources/internvl]] / [[entities/internvl]]（InternVL 1.0、CVPR 2024）
- 後続バージョン: InternVL 2.0 / 2.5 / 3（未 ingest、現時点最新は InternVL 3）
- 関連 MLLM: LLaVA / LLaVA-NeXT / Mini-Gemini / DocOwl-1.5 / DeepSeek-VL（未 ingest、必要なら別途）
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]] — MLLM 用 VFM の選択肢
- 関連概念: [[concepts/foundation-model]]、[[concepts/weakly-supervised-pretraining]]、[[concepts/zero-shot-transfer]]
