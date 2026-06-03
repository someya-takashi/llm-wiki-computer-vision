---
type: source
source_path: raw/papers/InternVL_ Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks.md
source_kind: paper
title: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
authors: [Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, Jifeng Dai]
year: 2023
venue: CVPR 2024
ingested: 2026-05-28
tags: [vision-language, foundation-model, llm-alignment, internvit, qllama, opengvlab, vllm, multimodal]
translation: "[[translations/internvl]]"
---

# InternVL — 6B 視覚基盤モデルを LLM と整列させる初の本格的試み

> 原典: [[translations/internvl]] ・ `raw/papers/InternVL_ Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks.md`
> 著者: Zhe Chen, Jiannan Wu, Wenhai Wang ら（OpenGVLab Shanghai AI Lab + 南京大学 + 香港大学 + 清華大学 + USTC + SenseTime）
> 投稿: arXiv:2312.14238（2023 年 12 月）/ CVPR 2024
> リポジトリ: <https://github.com/OpenGVLab/InternVL>

---

## 一言まとめ

**「[[entities/clip|CLIP]] の視覚エンコーダ（~1B）を LLM 規模（6B）にスケールアップし、80 億パラメータの言語ミドルウェア QLLaMA で LLM と整列させた、初の本格的視覚言語基盤モデル」**。InternViT-6B + QLLaMA という 2 つの新規大規模モジュールを **3 段階の段階的整列訓練（contrastive → generative → SFT）** で接続し、視覚知覚・対比・生成・対話のすべてを 1 モデルでこなす「スイスアーミーナイフ」。視覚エンコーダのスケールアップという軸で **[[sources/perception-encoder|Perception Encoder]] や [[sources/siglip-2|SigLIP 2]] と並ぶ「対比学習の頂点」を作ったオリジンの 1 つ** であり、後の **[[sources/internvl-1-5|InternVL 1.5（2024 April）]]** / 2.0 / 2.5 / 3 のシリーズ起点となった重要論文。

---

## 背景と問題意識

### LLM と視覚モデルの隔たり

2023 年時点、NLP では LLM が **1000B（兆）パラメータ規模** に到達していたのに対し、VLLM（vision large language models, 画像入力可能な LLM）の視覚エンコーダは **依然として ~1B パラメータ** に留まっていた。著者らは [[concepts/vision-transformer|ViT]] を例に「視覚基盤モデルの規模が LLM の能力を制約している」と問題提起する：

| 制約 | 問題 |
|---|---|
| **パラメータ規模の乖離（Disparity）** | LLM 1000B vs 視覚 1B → LLM の能力が活かしきれない |
| **表現の不整合（Inconsistent representation）** | 視覚モデルは ImageNet / JFT で純粋訓練か BERT 系と整列、LLM とは別空間 |
| **非効率な接続（Inefficient connection）** | QFormer / 線形射影は軽量で random init、cross-modal な深い相互作用を捉えきれない |

> **補足: VLLM とは** — LLaVA / Mini-GPT4 / Flamingo / BLIP-2 のような、テキスト LLM に画像入力を「足した」マルチモーダルモデル。視覚エンコーダ（CLIP-L 等）の出力を、QFormer や線形層（**glue layer**）で LLM の埋め込み空間に投影する構造が標準。この **glue layer の貧弱さこそが性能のボトルネック** だと著者らは主張する。

### 既存 VLLM のスケール構図

論文の図 1 が「視覚基盤モデルの 3 世代」を端的に示す：

| 世代 | 構造 | 例 |
|---|---|---|
| (a) **教師あり事前学習** | 視覚エンコーダ単独 + 分類ヘッド | ResNet, ViT |
| (b) **対比事前学習** | 視覚エンコーダ + テキストエンコーダ（dual-tower） | [[entities/clip\|CLIP]], OpenCLIP, EVA-CLIP |
| (c) **InternVL（本論文）** | **6B 視覚エンコーダ + 8B 言語ミドルウェア（共有）+ LLM** | InternVL-C/G/Chat |

InternVL は (c) を提案し、**「対比学習 + 生成学習 + LLM 接続をすべて 1 モデルでカバーする」** という新世代を提唱する。

---

## 提案手法

### 主要な 3 つの設計

#### 1. パラメータが均衡した視覚・言語コンポーネント

- **InternViT-6B**（vision encoder, 5.9B）
- **QLLaMA**（language middleware, 8B = 7B LLaMA + 1B 新規 cross-attention 層）

両者を合わせて **14B** のスケールに達し、これは [[entities/clip|CLIP]] の総パラメータ数（~1B）の 14×、QFormer（188M）の 42×。これにより：

- **対比タスク** では InternViT のみ使用（dual-tower 互換）
- **生成タスク** では InternViT + QLLaMA で前置文脈を作る（BLIP-2 互換）
- **対話タスク** では InternViT + QLLaMA + LLM デコーダ（**Vicuna-7B/13B**, **InternLM**）の 3 段構成

#### 2. 整合した表現（Consistent representations）

- QLLaMA は **多言語強化 LLaMA で初期化**（テキスト encoder と LLM decoder で言語空間を共有）
- 視覚 encoder を **LLM 系の表現空間と整列させる** ことが鍵。BERT 系（CLIP, SigLIP の text encoder）ではなく LLaMA 系に揃えるのが新規

#### 3. 段階的画像-テキスト整列（Progressive image-text alignment）

3 段階の訓練戦略：

```
Stage 1: 対比学習
  InternViT-6B（学習可） ↔ LLaMA-7B（学習可）
  データ: 4.98B image-text pairs（[[entities/clip|CLIP]] 流の symmetric InfoNCE）
  ↓
Stage 2: 生成学習
  InternViT-6B（凍結）→ QLLaMA（query+cross-attn のみ学習）
  データ: 1.03B（高品質）
  損失: ITC + ITM + ITG（BLIP-2 流の 3 損失）
  ↓
Stage 3: 教師あり微調整（SFT）
  InternVL → MLP → Vicuna/InternLM（LLM デコーダは凍結 or 学習可）
  データ: 4M 指示データ（COCO Cap / VQAv2 / OKVQA / RefCOCO / LLaVA / etc.）
```

> **補足: なぜ段階的か** — 大規模対比学習（Stage 1）はノイズ耐性が高くデータ規模が物を言うのに対し、生成学習（Stage 2）は高品質データを少量与えるほうが効く。InternVL は両者の良いところを段階的に取ることで、Web 規模データの異種性に対応している。

### InternViT-6B のハイパーパラメータ探索

論文では深さ・head 次元・MLP 比を網羅探索し、**16 構成から 6 構成に絞り → 100M LAION-en で 10K iter 対比訓練** という手間をかけて最終構成を決定。最終的に variant 3（**width 3200 / depth 48 / MLP 12800 / 25 heads**）を採用。これは [[concepts/vision-transformer|ViT]] 系のスケール則を強く意識した設計：

- 既存 ViT-G (1.8B), ViT-e (3.9B), EVA-02-ViT-E (4.4B), ViT-6.5B, ViT-22B などと比較
- 深さ 48 を維持しつつ MLP 比 4 で計算を節約、head 数を 25 と比較的少なく設定

### QLLaMA: BLIP-2 QFormer の 42 倍版

| 項目 | QFormer ([[sources/clip|BLIP-2]]) | QLLaMA (InternVL) |
|---|---|---|
| パラメータ | 188M | 8B（7B LLaMA 継承 + 1B 新規） |
| 初期化 | BERT-base | 多言語 LLaMA-7B |
| 学習可能 query 数 | 32 | 96 |
| 機能 | vision → LLM 投影のみ | **対比 + 生成 + LLM 接続のすべて** |

QLLaMA は **「glue layer のスケーリング」** を初めて本格的に試した設計と言える。これにより LLM デコーダを凍結したままでも、InternVL は対話で SoTA を出せる。

---

## 実験結果と知見

### 視覚知覚タスク（InternViT-6B 単体評価）

| タスク | データセット | 指標 | InternViT-6B | 既存 SoTA | 差 |
|---|---|---|---|---|---|
| **画像分類（linear probe）** | ImageNet-1K | top-1 | **88.2** | EVA-01-CLIP-g 86.5 (1.1B) | +1.7 |
| 画像分類（IN 派生平均） | IN-A/R/V2/Sketch | avg | **82.5** | EVA-01-CLIP-g 79.1 | +3.4 |
| **セマンティックセグメンテーション** | ADE20K linear probe | mIoU | **47.2** | ViT-22B 34.6 (21.7B) | **+12.6** |
| ADE20K UperNet 凍結 | ADE20K | mIoU | **54.9** | ViT-22B 52.7 | +2.2 |
| **ADE20K full tuning** | ADE20K | mIoU | **58.9** | ViT-22B 55.3 | +3.6 |

**特筆**: InternViT-6B（5.9B）は **3.7× パラメータ少ない ViT-22B（21.7B）を ADE20K linear probe で +12.6 mIoU という驚異的な差で上回る**。これは「JFT-3B を使わない純公開データ + 適切な設計で、Google の ViT-22B に対抗できる」ことの実証となった。

### 視覚言語タスク（対比 + 生成、InternVL-C/G）

| タスク | データセット | InternVL-C/G | 既存 SoTA | 差 |
|---|---|---|---|---|
| **ゼロショット画像分類** | IN-1K | **83.2** | EVA-02-CLIP-E+ 82.0 | +1.2 |
| ゼロショット画像分類（多言語） | IN-1K JP | **61.5** | Japanese-CLIP 54.6 | +6.9 |
| ゼロショット画像分類（多言語） | IN-1K AR | **44.9** | OpenCLIP-XLM-R-H 37.0 | +7.9 |
| **ゼロショット動画分類** | Kinetics-400 (8F) | top-1 **69.1** | ViCLIP 64.8 | +4.3 |
| ゼロショット動画分類 | Kinetics-700 (8F) | top-1 **60.6** | ViCLIP 54.3 | +6.3 |
| **ゼロショット画像-テキスト検索** | Flickr30K I2T R@1 (EN) | **95.7 (G)** | EVA-02-CLIP-E+ 93.9 | +1.8 |
| ゼロショット画像-テキスト検索 | Flickr30K-CN I2T R@1 | **92.9 (G)** | AltCLIP-ViT-H 88.9 | +4.0 |
| **ゼロショット画像キャプショニング** | COCO Karpathy test | CIDEr **128.2 (G)** | DreamLLM 115.4 | +12.8 |

**特筆**: InternVL-G は **対比 + 生成の両方を 1 モデルで** こなし、Flickr30K キャプショニングで CIDEr 128.2 という強力な結果を得る。これは QLLaMA が 8B 規模の生成能力を持っているおかげ。

### マルチモーダル対話タスク（InternVL-Chat）

| タスク | データセット | InternVL-Chat-13B | LLaVA-1.5-13B | 差 |
|---|---|---|---|---|
| **総合認知** | MME | **1586.4** | 1531.3 | +55 |
| **物体ハルシネーション** | POPE | **87.6** | 85.9 | +1.7 |
| VQA | VQAv2 test-dev | **81.2** | 80.0 | +1.2 |
| VQA | GQA test-balanced | **66.6** | 63.3 | +3.3 |
| VQA | TextVQA val | **61.5** | 61.3 | +0.2 |
| キャプショニング | Flickr30K | **92.2** | – | – |

**特筆**: 同じ Vicuna-13B + 224 res 構成でも、視覚側を CLIP-L (304M) から InternViT-6B (5.9B) に置換するだけで **MME +50 / POPE +1.7 / GQA +3.3** の改善が出る。アブレーション（表 12）では InternViT-6B + QLLaMA glue で **MME 970 → 1227（+257）** という大幅向上が確認され、視覚エンコーダのスケールアップと LLM 系 glue の両方が効くことが立証された。

---

## 限界・批判的視点

### 訓練コストの重さ

- **Stage 1 だけで 4.98B 画像-テキスト対を InternViT-6B + LLaMA-7B でフル訓練**。これは当時最大級の計算予算（Shanghai AI Lab の InternLM クラスタを利用）
- 後の [[sources/perception-encoder|PE]]（5.4B unique pairs）や [[sources/siglip-2|SigLIP 2]]（WebLI 10B）と同じ「Web 規模対比学習スケール」軍備拡張競争の一翼

### 公開モデルの後退

- 論文時点では InternViT-6B / QLLaMA / InternVL-Chat 全部公開、Vicuna も Apache 系
- ただし論文時点の最大構成（Vicuna-13B + InternViT-6B + QLLaMA + 336 res）は推論コストも高く、エッジ展開には不適。後続の InternVL シリーズで InternVL-Mini 等の軽量化が進む

### 視覚エンコーダ規模の意義

- **InternViT-6B vs ViT-22B**: 6B でも 22B の linear probe（IN-1K）+ ADE20K で勝つ。ViT-22B の問題は「JFT-3B 非公開データへの依存」と「規模に見合わない linear probe 性能」だった、と暗に示唆
- **ただし、InternViT-6B は JFT を使わない代わりに 4.98B の Web ペアを必要とする** — 結局は「データ規模 vs パラメータ規模」のトレードオフで、データ側を膨らませる路線
- 後の [[sources/dinov3|DINOv3]] が「テキストなし純粋画像 SSL で 7B モデル × 1.689B 画像」を可能にしたことで、**「対比 vs 純 SSL のスケーリング戦略」** という議論軸が改めて立つ

### LLM への単方向依存

- QLLaMA を LLaMA で初期化することで「LLaMA 系 LLM とは整合する」が、**Qwen / GPT-4 / Claude 系の埋め込みとは異なる空間** になる
- 後の **[[sources/internvl-1-5|InternVL 1.5]]** で QLLaMA は完全に廃止され、LLaVA 系の MLP プロジェクタに統一された（「軽量 glue で十分」という業界トレンドの追認）。InternVL 1.5 では LLM が **InternLM2-20B-Chat** に変更され、解像度も **dynamic 448（最大 4K）** に拡張、OCR・中国語に特化した path に進化

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **InternVL** | Internal Vision-Language model | OpenGVLab Shanghai AI Lab 発の 6B+8B 視覚言語基盤モデル |
| **InternViT-6B** | Internal Vision Transformer 6B | 視覚エンコーダ（5.9B, w=3200, d=48, MLP=12800, h=25, vanilla ViT, /14 patch） |
| **QLLaMA** | Query LLaMA | 言語ミドルウェア（8B = 7B LLaMA 継承 + 1B 新規 96 queries + cross-attn） |
| **InternVL-C** | InternVL Contrastive | 対比モード（InternViT のみで $I_f$ 計算、QLLaMA [EOS] で $T_f$） |
| **InternVL-G** | InternVL Generative | 生成 + 対比モード（InternViT + QLLaMA で $I_f$ をリッチに） |
| **InternVL-Chat** | InternVL with Chat LLM | LLM 接続版（InternViT + QLLaMA + Vicuna / InternLM） |
| **VLLM** | Vision Large Language Model | 画像入力可能な LLM（LLaVA, MiniGPT-4, Flamingo, BLIP-2 等） |
| **glue layer** | – | 視覚エンコーダと LLM をつなぐ層（QFormer, MLP, Linear, Cross-Attn, VL-Adapter, QLLaMA 等） |
| **QFormer** | Querying Transformer | BLIP-2 の glue layer（188M, BERT-base 初期化、32 queries） |
| **ITC** | Image-Text Contrastive | BLIP-2 の対比損失 |
| **ITM** | Image-Text Matching | BLIP-2 のマッチング 2 値分類損失 |
| **ITG** | Image-Grounded Text Generation | BLIP-2 の生成損失（autoregressive language modeling） |
| **SFT** | Supervised Fine-Tuning | 指示データでの教師あり微調整 |
| **PT** | Pre-Training | 事前学習データ量 |
| **LAION-en / -multi** | Large-scale Artificial Intelligence Open Network | 公開 Web 画像-テキスト対（LAION-5B 系、英語版/多言語版） |
| **LAION-COCO** | LAION + COCO caption style | LAION-en から BLIP で高品質キャプション生成した合成データ |
| **COYO** | COYO-700M | Kakao の Web 画像-テキストデータ |
| **Wukong** | – | 華為が公開した 100M 中国語画像-テキストデータ |
| **CC3M / CC12M** | Conceptual Captions 3M/12M | Google の Web alt-text データセット |
| **SBU** | SBU Captions | Stony Brook の Web alt-text データ（1M, Ordonez 2011） |
| **WIT-400M** | WebImageText 400M | [[entities/wit-400m\|CLIP の非公開訓練データ]] |
| **JFT-3B** | – | Google 非公開の 3B 弱教師あり分類データ |
| **ViT-22B** | Vision Transformer 22B | Google が 2023 年に発表した 21.7B パラメータ ViT、JFT-3B で訓練 |
| **EVA / EVA-02** | – | BAAI の MIM + CLIP 蒸留系列（EVA-CLIP-g/E+ など） |
| **MAWS** | Masked Autoencoders are Weakly-Supervised | Meta の MIM + 弱教師 ViT-6.5B |
| **CoCa** | Contrastive Captioner | Google の対比 + 生成 dual-loss モデル |
| **LiT-22B** | Locked-image Tuning 22B | Google の ViT-22B + locked image LiT 法 |
| **DINOv2** | – | [[entities/dinov2\|純粋 SSL の 1B モデル]] |
| **Vicuna-7B/13B** | – | LLaMA fine-tuned chat LLM（lmsys） |
| **InternLM** | – | Shanghai AI Lab の独自 LLM 系列 |
| **LLaVA / LLaVA-1.5** | Large Language and Vision Assistant | Visual Instruction Tuning の代表作（Liu et al., 2023） |
| **InstructBLIP** | – | BLIP-2 + Instruction Tuning |
| **Shikra** | – | grounding 機能付き VLLM |
| **IDEFICS** | – | HuggingFace の Flamingo 互換 OS VLLM |
| **Qwen-VL** | – | Alibaba の VLLM |
| **KOSMOS-2** | – | Microsoft の grounding 機能付き VLLM |
| **Flamingo** | – | DeepMind の Few-Shot VLLM 先駆け（Chinchilla LLM + Cross-Attn glue） |
| **PaLI-X-55B** | – | Google の vision-language UL2-32B + 55B モデル |
| **Emu / Emu-I** | – | Baidu の生成 VLLM |
| **DreamLLM** | – | Pku の生成 VLLM |
| **MME** | Multimodal Evaluation | 14 サブタスクの総合認知ベンチマーク（Fu et al., 2023） |
| **POPE** | Polling-based Object Probing Evaluation | 物体ハルシネーション評価（Li et al., 2023） |
| **Tiny LVLM** | – | LVLM 用軽量ベンチマーク |
| **VQAv2 / GQA / VizWiz / TextVQA** | – | VQA 標準ベンチマーク群 |
| **OK-VQA / A-OK-VQA** | Outside Knowledge VQA | 外部知識を必要とする VQA |
| **IconQA / AI2D** | – | アイコン理解 / 図表理解 VQA |
| **OCR-VQA / ChartQA / DocVQA / ST-VQA / EST-VQA / InfoVQA / LLaVAR** | – | OCR/文書 VQA 系 |
| **RefCOCO/+/g** | – | 指示表現理解（REC）標準ベンチマーク |
| **Toloka** | – | grounding 用クラウドソーシングデータ |
| **LLaVA-150K / LLaVA-Mix-665K** | – | LLaVA の指示データ |
| **SVIT / VisDial / LRV-Instruction** | – | 視覚指示・対話データ |
| **NoCaps / Flickr30K / COCO Caption** | – | 画像キャプショニングベンチマーク |
| **Flickr30K-CN / COCO-CN** | – | 中国語版検索ベンチマーク |
| **XTD** | – | 多言語検索評価（8 言語） |
| **MSR-VTT** | – | 動画-テキスト検索ベンチマーク |
| **Kinetics-400/600/700** | – | 動画分類標準ベンチマーク |
| **IN-1K / IN-A / IN-R / IN-V2 / IN-Sketch / IN-ReaL** | ImageNet variants | ImageNet 派生分布シフトベンチマーク |
| **ObjectNet** | – | 制御された分布シフトベンチマーク |
| **AltCLIP / WuKong-ViT / Taiyi-CLIP / CN-CLIP / Japanese-CLIP / R2D2** | – | 多言語 CLIP 系列 |
| **OpenGVLab** | – | Shanghai AI Lab の vision-language 研究グループ |
| **AGI** | Artificial General Intelligence | 汎用人工知能 |
| **AlexNet / ResNet** | – | CNN 系の代表（AlexNet 2012 / ResNet 2015） |
| **JFT** | – | Google 非公開分類データセット（3B 規模） |

---

## 関連ページ

- 翻訳: [[translations/internvl]]
- 主要エンティティ: [[entities/internvl]]（モデルファミリーの仕様、benchmark 詳細、InternViT-6B / QLLaMA / InternVL-C/G/Chat 内部含む）
- 視覚基盤モデル系譜: [[concepts/foundation-model]]、[[concepts/weakly-supervised-pretraining]]、[[overview]]
- 直系祖: [[sources/clip]] / [[entities/clip]]（対比学習 + zero-shot transfer の起源）
- 直接の比較対象: [[entities/perception-encoder]]（Meta の対比スケール路線、2025）、[[entities/siglip]]（Google の対比改良路線、2023-25）、[[entities/dinov2]] / [[entities/dinov3]]（Meta の純粋 SSL 路線）
- glue layer 設計: BLIP-2 の QFormer（QLLaMA の直接の比較相手）、LLaVA の MLP、Flamingo の Cross-Attn、Qwen-VL の VL-Adapter
- 関連概念: [[concepts/zero-shot-transfer]]、[[concepts/contrastive-learning]]、[[concepts/vision-transformer]]、[[concepts/foundation-model]]
