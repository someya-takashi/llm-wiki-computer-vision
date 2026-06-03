---
type: source
source_path: raw/papers/Mini-InternVL_ A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance.md
source_kind: paper
title: "Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance"
authors: [Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Jifeng Dai, Wenhai Wang]
year: 2024
venue: "arXiv:2410.16261 (technical report)"
ingested: 2026-05-29
tags: [mllm, vllm, mini-internvl, internvit-300m, knowledge-distillation, edge-deployment, domain-adaptation, autonomous-driving, medical, remote-sensing, opengvlab]
translation: "[[translations/mini-internvl]]"
---

# Mini-InternVL — 5% パラメータ × 90% 性能のポケット MLLM + 統一ドメイン適応フレームワーク

> 原典: [[translations/mini-internvl]] ・ `raw/papers/Mini-InternVL_ A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance.md`
> 著者: Zhangwei Gao, Zhe Chen ら（Shanghai AI Lab + 清華大学 + 南京大学 + 復旦大学 + 香港中文大学 + SenseTime + Shanghai Jiao Tong University）
> 投稿: arXiv:2410.16261（2024 年 10 月、technical report）
> リポジトリ: <https://github.com/OpenGVLab/InternVL>

---

## 一言まとめ

**「[[entities/internvl-1-5|InternVL 1.5]] / InternVL2 の 76B 路線を補完する軽量 MLLM スイート。InternViT-6B（5.9B）を CLIP-ViT-L (300M) に **知識蒸留**して `InternViT-300M` を作り、1B/2B/4B クラスの LLM（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）と組み合わせて、**5% のパラメータで InternVL2-76B の 90% の性能** を達成。さらに **自律走行 / 医療画像 / リモートセンシング** を VQA 形式に統一する **「ドメイン適応フレームワーク」を初めて MLLM スイートとして提案**、各専門ドメインで商用フロンティアモデル（GPT-4o, Claude 3.5 Sonnet, GPT-4V）を上回る Mini-InternVL-DA 系列まで構築」**。InternVL シリーズが「**大規模 + 商用追従**」（[[entities/internvl-1-5|InternVL 1.5]]）から「**軽量 + ドメイン特化**」（本ページ）へ二分岐したことを示す重要論文。

---

## 背景と問題意識

### 商用追従だけでは行けない場所がある

[[sources/internvl-1-5|InternVL 1.5]]（2024 April）は **26B の OS MLLM で GPT-4V との差を 18 ベンチ中 8 SoTA で埋めた** が、それでも以下の課題が残っていた：

| 制約 | 影響 |
|---|---|
| **モデルサイズ** | 26B-76B は**消費者 GPU**（RTX 3090, 4090）でも **推論メモリが足りない** ことが多い |
| **長尾ドメイン性能** | 汎用 MLLM は **自律走行 / 医療画像 / リモートセンシング** などで非常に弱い |
| **ドメイン適応の方法論** | GeoChat（リモセン）/ LLaVA-Med（医療）/ DriveGPT4（自律走行）が **それぞれ独自設計**、相互運用性なし |

この論文の問題提起：「How can MLLMs be deployable on consumer GPUs/edge AND transferable to specialized domains?」（消費者 GPU/エッジで展開可能、かつ特殊ドメインに転移可能な MLLM は作れるか？）

### 既存軽量 MLLM の弱点

2024 年中盤の軽量 MLLM（MiniCPM-V 2.0, DeepSeek-VL-1.3B, Qwen2-VL-2B）はすべて **CLIP-ViT-L (300M)** を視覚エンコーダに採用。しかし CLIP-ViT-L は：

- **Internet ドメインの自然画像のみで訓練** → OCR・医療・衛星画像で性能劣化
- **BERT 系と整列されている** → LLM の埋め込み空間と不整合

この論文は「**InternViT-6B が獲得した多ドメイン視覚知識を 300M に圧縮する**」という解を提案。

---

## 提案手法

### 全体構造: ViT-MLP-LLM（[[entities/internvl-1-5|InternVL 1.5]] と同じ）

```
画像入力（任意解像度）
   ↓ 動的高解像度（35 種アスペクト比 × 1-12 タイル、test 40 タイル）
448×448 タイル × N + サムネイル
   ↓ InternViT-300M（CLIP-L で初期化 + InternViT-6B から蒸留）
1024 visual tokens / tile
   ↓ Pixel Unshuffle（1/4 圧縮、4 patches → 1 token）
256 visual tokens / tile
   ↓ MLP プロジェクタ
LLM 埋め込み空間
   ↓ Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini
出力テキスト
```

> **補足: Pixel Unshuffle vs Pixel Shuffle** — [[entities/internvl-1-5|InternVL 1.5]] では "Pixel Shuffle" と呼んでいたが、本論文は "Pixel Unshuffle" と呼ぶ。実体は同じ「空間 → チャンネル」変換（4 patches → 1 token, 12800-d）。命名の不統一に注意。

### 1. InternViT-300M（知識蒸留による軽量 VFM）

**重要な発見**: 強力な VFM の能力は **知識蒸留で軽量モデルに転送可能**。

| 項目 | InternViT-6B (教師) | **InternViT-300M (生徒)** |
|---|---|---|
| パラメータ | 5.9B | **300M** |
| 初期化 | ランダム | **CLIP-ViT-L-336px** |
| 訓練 | 4.98B image-text の対比 + 段階的整列 | **InternViT-6B からの蒸留のみ** |
| 蒸留損失 | – | **最後の K 層の隠れ状態間で negative cosine similarity** |
| 訓練データ | – | 表 1 の多様な image data（natural / OCR / chart / multidisciplinary） |
| 解像度 | dynamic 448 | 固定 448（蒸留時のみ、推論時は dynamic） |

#### 蒸留データのドメイン分布

| Type | 内容 |
|---|---|
| **Natural images** | LAION, COYO, GRIT, COCO, LVIS, Objects365, Flickr30K, VG, All-Seeing, MMInstruct, LRV-Instruction |
| **OCR** | TextCaps, Wukong-OCR, CTW, MMC-Inst, LSVT, ST-VQA, RCTW-17, ReCTs, ArT, SynthDoG, LaionCOCO-OCR, COCO-Text, DocVQA, TextOCR, LLaVAR, TQA, SynthText, DocReason25K, Common Crawl PDF |
| **Chart** | AI2D, PlotQA, InfoVQA, ChartQA, MapQA, FigureQA, IconQA, MMC-Instruction |
| **Multidisciplinary** | CLEVR-Math/Super, GeoQA+, UniChart, ScienceQA, Inter-GPS, UniGeo, PMC-VQA, TabMWP, MetaMathQA |
| **Other** | Stanford40, GQA, MovieNet, KonIQ-10K, ART500K, ViQuAE |

**意義**: CLIP は **自然画像 + 短いキャプション** しか見ていないが、InternViT-300M は **OCR / chart / multidisciplinary / PMC-VQA（医療）** までカバー。これが「ドメイン適応の基礎」を作る。

### 2. Mini-InternVL シリーズ 3 種

| モデル | LLM | LLM サイズ | 視覚 | 合計 |
|---|---|---|---|---|
| **Mini-InternVL-1B** | Qwen2-0.5B | 0.5B | InternViT-300M | **1B (1%)** |
| **Mini-InternVL-2B** | InternLM2-1.8B | 1.8B | InternViT-300M | **2B (3%)** |
| **Mini-InternVL-4B** | Phi-3-Mini | 3.8B | InternViT-300M | **4B (5%)** |

LLM ベンダーが **OpenGVLab 内製（InternLM2）/ Alibaba (Qwen2) / Microsoft (Phi-3)** と異なる点に注目。**「LLM の選び方は外部、視覚側は内製で蒸留」** という分業。

### 3. 統一ドメイン適応フレームワーク（本論文の独自貢献）

すべての下流タスクを **VQA + 対話形式** に統一する。**5 つのタスクタイプ** の変換方法を提示：

| タスクタイプ | 変換方法 |
|---|---|
| **画像分類** | 多肢選択問題 "Classify the image within one of the given classes: ... Answer with one word." |
| **視覚的グラウンディング** | `<ref>名前</ref>` で対象、`<box>[[x1,y1,x2,y2]]</box>` で位置（0-1000 正規化） |
| **領域知覚** | 画像上に box/mask 描画 or 質問内 `<box>` 表記でモデル注目を誘導 |
| **複数視点画像**（自律走行） | 6 視点を 896×448 にリサイズ + 固定順序結合 = 2688×896 → 12 タイル + サムネイル + "CAM_FRON" 注釈 |
| **動画フレーム** | "Frame1: <img><IMG_CONTEXT></img> Frame2: ..." の interleaved 形式、最大 40 フレーム |

**訓練戦略**:
- **全パラメータ微調整**
- **汎用データを 1:1 〜 1:4 比率で混合**（ドメインによって最適比率が異なる）
- 訓練 1 epoch、学習率 1e-5、8 A100 GPU

### 訓練の 2 段階（Mini-InternVL の基本訓練）

| Stage | 訓練対象 | データ | 目的 |
|---|---|---|---|
| **1. Language-Image Alignment** | MLP のみ | キャプション + 検出 + grounding + OCR | 視覚と言語の整列 |
| **2. Visual Instruction Tuning** | **全パラメータ** | 指示データ（[[entities/internvl-1-5\|InternVL 1.5]] 由来）| 指示追従 + 世界知識注入 |

---

## 実験結果と知見

### 汎用マルチモーダル: **5% パラメータで 90% 性能**

| モデル | パラメータ | Avg Score | 相対性能 |
|---|---|---|---|
| InternVL2-Llama3-76B | 76B | 81.4 | 100% |
| **Mini-InternVL-4B** | **4B** | **72.8** | **90%** |
| Mini-InternVL-2B | 2B | 66.8 | 82% |
| Mini-InternVL-1B | 1B | 60.6 | 74% |
| Gemini-Pro-1.5 | – | 73.6 | – |
| GPT-4V-0409 | – | 75.4 | – |
| MiniCPM-V 2.0 | 3B | 58.3 | 72% |
| DeepSeek-VL-1.3B | 2B | 47.3 | 58% |
| Qwen2-VL-2B | 2B | 68.5 | 84% |

**ハイライト**:
- **Mini-InternVL-4B (4B) は Gemini-Pro-1.5 とほぼ同じ平均スコア（72.8 vs 73.6）**
- **Qwen2-VL-2B (2B) より弱いベンチが多い**（特に DocVQA / OCRBench）が、**統一ドメイン適応フレームワーク** という追加価値
- **MiniCPM-V 2.0 を全方位で凌駕**

### 自律走行: ドメイン適応で **76B モデルを 1/10 のサイズで超える**

**DriveLM Challenge Leaderboard**:

| Method | #Param | Final Score |
|---|---|---|
| InternVL4Drive-v2 (CVPR 2024 SOTA) | 26B | 0.6002 |
| **Mini-InternVL-DA-2B** | **2B** | **0.5958**（**1/13 のサイズで匹敵**） |
| Mini-InternVL-4B (DA なし) | 4B | 0.3051 |
| InternVL2-Llama3-76B (DA なし) | 76B | 0.2963 |

**MME-RealWorld 自律走行 ドメイン**:

| Method | Avg |
|---|---|
| **Mini-InternVL-DA-4B** | **49.38** |
| InternVL2-Llama3-76B | 44.30 |
| LLaVA-OneVision-7B | 41.75 |
| Claude 3.5 Sonnet | 32.10 |
| GPT-4o | **24.60** |

**Mini-InternVL-DA-4B が GPT-4o を +24.78 ポイント圧倒**。

### 医療画像: 商用商品超え（GMAI-MMBench）

| Model | 2D Seg C | 2D Seg M | 2D Cls |
|---|---|---|---|
| GPT-4V | 47.87 | 46.58 | 42.24 |
| LLaVA-Med（医療特化） | 18.45 | 18.97 | 21.15 |
| RadFM (14B、医療特化) | 20.43 | 20.27 | 25.71 |
| Claude3-Opus | 33.56 | 33.36 | 32.17 |
| **Mini-InternVL-DA-4B** | **41.41** | **40.45** | **41.34** |

**4B の Mini-InternVL-DA が LLaVA-Med / RadFM / Claude3-Opus を凌駕**。GPT-4V には届かないが、**専門 MLLM より強い汎用ベースの転移モデル** を実証。

### リモートセンシング: SkySenseGPT / GeoChat 超え

| Method | DIOR-RSVG acc@0.5 |
|---|---|
| SkyEyeGPT | 88.59 |
| GeoChat (7B) | 72.30 |
| Mini-InternVL-4B (DA なし) | 16.89 |
| **Mini-InternVL-DA-4B** | **92.04** |

**4B のドメイン適応で SkyEyeGPT を +3.45 ポイント超え**。

### アブレーション: 知識蒸留 vs CLIP-ViT-L

**Mini-InternVL-2B vs Mini-InternVL-CLIP-2B（CLIP-L 置換版）**:

| ベンチ | CLIP-2B | Mini-InternVL-2B | 差 |
|---|---|---|---|
| MMB-EN | 70.3 | **73.2** | +2.9 |
| MMB-CN | 68.1 | **70.9** | +2.8 |
| ChartQA | 70.9 | **76.2** | +5.3 |
| **DocVQA** | 77.5 | **85.9** | **+8.4** |
| InfoVQA | 49.6 | **57.7** | +8.1 |
| MMMU | 32.9 | **34.3** | +1.4 |
| MME-RW (AD) | 43.7 | **48.0** | +4.3 |
| DriveLM | 0.580 | 0.578 | -0.002 |

**InternViT-300M（蒸留）が CLIP-ViT-L (300M) を、特に OCR / Document / Chart で大きく上回る**。これは **「視覚エンコーダのドメイン多様性が MLLM の上流から下流まで効く」** ことの実証。

### アブレーション: データ比率と訓練手法

- **汎用:特化 = 1:4（r=0.25）が自律走行で最適**。それ以上汎用を入れても効果薄
- **訓練データの 1/4 でほぼ同じ性能**（計算大幅削減可能）
- **適応手法ランキング**: Full-parameter > Freezing ViT > LoRA
- LoRA は **視覚的グラウンディングが弱い**（座標出力に向かない）
- **すべての手法で 1500 ステップで収束**

---

## 限界・批判的視点

### MMMU で同期サイズ MLLM に劣る

- Mini-InternVL-2B: MMMU 36.3 vs Qwen2-VL-2B: 42.2 (-5.9)
- Mini-InternVL-4B: MMMU 48.3 vs Cambrian-1 (~7B): 50.4 (-2.1)
- **多分野知識タスクでは LLM サイズが効く**（[[sources/internvl-1-5|InternVL 1.5]] でも同じ観察）。Mini-InternVL は **LLM が 0.5B-3.8B と小さい**ため不利

### DriveLM で 4B が 2B より弱い

- Mini-InternVL-DA-2B: 0.5958 vs Mini-InternVL-DA-4B: 0.5821
- 著者自身が「現有 DriveLM 訓練データ・評価基準では大きいモデルが性能向上を出しにくい」と分析
- **DriveLM 自体のデータ規模が小さく、Phi-3-Mini の表現力をフル活用できない**

### ドメイン適応の汎化能力に限界

- 各ドメイン（自律走行 / 医療 / リモセン）の DA モデルは **その分野でのみ強い**
- **複数ドメインを 1 モデルで扱う統合 MLLM ではない**（各ドメインで個別 DA 必要）
- API mode で「医療 + 自律走行同時クエリ」のような複合シナリオには対応していない

### 視覚エンコーダのライセンス問題

- InternViT-6B が教師 → 知識蒸留で InternViT-300M を作る = **モデル蒸留**
- CLIP-ViT-L の重みで初期化 → OpenAI MIT ライセンス互換だが、**蒸留後のモデル系統はやや曖昧**
- 公開時のライセンスは MIT + InternLM2 のミックスで、商用利用に注意が必要

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **Mini-InternVL** | – | OpenGVLab の軽量 MLLM 系列（2024 Oct、本論文） |
| **Mini-InternVL-1B/2B/4B** | – | Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini ベース |
| **Mini-InternVL-DA** | – | Domain-Adapted、特定ドメインに fine-tuned したバリアント |
| **InternViT-300M** | – | 300M parameter ViT。**CLIP-ViT-L で初期化 + InternViT-6B から蒸留** |
| **InternViT-6B** | – | InternVL シリーズの 6B 教師 VFM（[[entities/internvl]] 由来） |
| **CLIP-ViT-L-336px** | – | OpenAI CLIP-L（304M）の 336×336 解像度版、InternViT-300M の初期重み |
| **Pixel Unshuffle** | – | [[entities/internvl-1-5\|InternVL 1.5]] の Pixel Shuffle と同じ（呼称違い）、4 patches → 1 token |
| **Qwen2-0.5B** | – | Alibaba の超軽量 LLM |
| **InternLM2-1.8B** | – | Shanghai AI Lab の超軽量 LLM |
| **Phi-3-Mini** | Phi-3-Mini-3.8B | Microsoft の軽量 LLM（3.8B） |
| **Negative Cosine Similarity Loss** | – | InternViT-300M の蒸留損失、最後の K 層の隠れ状態を整列 |
| **ViT-MLP-LLM** | – | LLaVA 系の標準構造（[[sources/internvl-1-5]] と同じ） |
| **Knowledge Distillation (KD)** | – | 教師の知識を生徒に転送する戦略 |
| **MLLM** | Multimodal Large Language Model | 画像入力可能な LLM、本論文の主呼称 |
| **VFM** | Vision Foundation Model | 視覚エンコーダの総称 |
| **Domain Adaptation (DA)** | – | 特定ドメインへの転移（本論文の独自フレームワーク） |
| **VQA Format** | Visual Question Answering | すべてのドメインタスクを定式化する統一形式 |
| **`<ref></ref>` / `<box></box>`** | – | Mini-InternVL の特殊トークン（grounding 用） |
| **`<image>` / `<IMG_CONTEXT>`** | – | InternVL 系の画像プレースホルダ |
| **Multi-view Images** | – | 6 視点（CAM_FRONT, CAM_BACK, CAM_LEFT 等）の自律走行画像 |
| **Interleaved Image Format** | – | "Frame1: <img>... Frame2: <img>..." の動画表現形式 |
| **DriveLM-nuScenes v1.1** | – | 自律走行ベンチマーク（DriveLM Challenge CVPR 2024）、317K サンプル |
| **CVPR 2024 Autonomous Driving Challenge** | – | DriveLM 公式リーダーボード |
| **InternVL4Drive-v2** | – | InternVL 1.5 派生の 26B 自律走行 SOTA モデル |
| **DriveGPT4** | – | BDD-X ベースの自律走行 VLM、Mini-InternVL の比較相手 |
| **DriveLM / DriveMLM / DriveVLM** | – | 自律走行 VLM 系列 |
| **BDD-X** | Berkeley Deep Drive eXplanation | 動画駆動の自律走行データ（DriveGPT4 が使用） |
| **MME-Realworld (AD)** | – | MME の自律走行サブドメイン |
| **CAM_FRONT / CAM_BACK / CAM_LEFT** | – | nuScenes の 6 視点カメラ位置 |
| **Action Description / Justification** | – | 自律走行タスクの動作記述と理由付け |
| **Speed Signal / Turning Angle** | – | 自律走行の制御信号予測 |
| **ADAPT** | – | 自律走行説明モデル（比較相手） |
| **GMAI-MMBench** | – | 医療 AI 総合 MLLM ベンチマーク |
| **PMC-OA / PMC-VQA / PMC-Image** | PubMed Central | 医療画像-テキスト対 |
| **MedICaT / MedPix** | – | 医療画像データセット |
| **MIMIC-CXR** | – | MIT の胸部 X 線データ |
| **Open-i** | – | NIH の医療画像データ |
| **Quilt-1M** | – | 組織病理学画像-テキスト対 |
| **RP3D** | – | X 線画像-テキスト対 |
| **Retina Image Bank** | – | 網膜画像 |
| **LLaVA-Med / Qilin-Med-VL / RadFM** | – | 医療特化 MLLM（比較相手） |
| **2D Seg C / Seg M** | – | 2D セグメンテーション（コアース / マイクロ）の GMAI 指標 |
| **2D Cls / 2D Det / 2D Mcls** | – | 2D 分類 / 検出 / 多クラス分類 |
| **RSVQA / RSVQA-LR / RSVQA-HR** | Remote Sensing VQA | リモートセンシング VQA 標準ベンチマーク（Low/High Res） |
| **GeoChat** | – | リモートセンシング MLLM（LLaVA-1.5 + 7B、本論文の比較相手） |
| **EarthGPT / SkyEyeGPT / SkySenseGPT** | – | リモートセンシング MLLM 系列 |
| **FIT-RS** | – | リモートセンシング VQA データ |
| **DIOR-RSVG** | – | リモートセンシング視覚的グラウンディングデータ |
| **Presence / Comparison / Rural-Urban** | – | RSVQA の評価サブタスク |
| **ChemVLM** | – | 化学特化 MLLM |
| **Cambrian-1** | – | NYU の 7B MLLM、本論文の比較相手 |
| **Qwen2-VL** | – | Alibaba の MLLM 系列 |
| **Qwen-VL-Chat** | – | Qwen-VL の対話版 |
| **LLaVA-OneVision** | – | LLaVA の動画拡張版（2024 Aug） |
| **LLaVA-NeXT-mistral-7B** | – | LLaVA-NeXT の mistral 版 |
| **MiniCPM-V 2.0** | – | Tsinghua の軽量 MLLM |
| **DeepSeek-VL-1.3B** | – | DeepSeek の軽量 MLLM |
| **Fuyu / MoMa / Chameleon** | – | 視覚エンコーダ不要の MLLM 系（参考） |
| **OpenCompass** | – | Shanghai AI Lab の MLLM 評価フレームワーク |
| **VLMEvalKit** | – | InternVL の評価ツール |
| **ZeRO1** | Zero Redundancy Optimizer Stage 1 | DeepSpeed のメモリ最適化技法 |
| **A100** | – | NVIDIA の訓練 GPU |
| **r=0.25** | – | Mini-InternVL の自律走行最適汎用データ比率 |

---

## 関連ページ

- 翻訳: [[translations/mini-internvl]]
- 主要エンティティ: [[entities/mini-internvl]]、[[entities/internvit-300m]]
- 直接の祖: [[sources/internvl-1-5]] / [[entities/internvl-1-5]]（InternVL 1.5、動的解像度の系譜）/ [[sources/internvl]] / [[entities/internvl]]（InternVL 1.0、InternViT-6B の系譜）
- 関連 MLLM: MiniCPM-V / DeepSeek-VL / Qwen2-VL / LLaVA-NeXT / LLaVA-OneVision（未 ingest）
- 関連視覚エンコーダ: [[entities/clip]]（CLIP-ViT-L-336px が初期化に使用）
- 関連概念: [[concepts/knowledge-distillation]]（中核技法）、[[concepts/foundation-model]]、[[concepts/weakly-supervised-pretraining]]、[[concepts/zero-shot-transfer]]
