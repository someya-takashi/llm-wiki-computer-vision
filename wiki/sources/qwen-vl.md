---
type: source
source_path: raw/papers/Qwen-VL_ A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.md
source_kind: paper
title: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
authors: [Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou]
year: 2023
venue: arXiv:2308.12966
ingested: 2026-05-30
tags: [qwen, qwen-vl, mllm, multimodal, vision-language, grounding, ocr, openclip, vit-bigg, cross-attention-adapter, alibaba]
translation: [[translations/qwen-vl]]
---

# Qwen-VL — Alibaba 製の細粒度グラウンディング対応 MLLM 第 1 世代

> 原典: [[translations/qwen-vl]] ・ `raw/papers/Qwen-VL_ A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.md`
> 著者・年・会議: Jinze Bai, Shuai Bai ら（Alibaba Group）, 2023, arXiv:2308.12966

## 一言まとめ

Alibaba の汎用言語モデル Qwen-7B に **OpenCLIP ViT-bigG** と **256 クエリのクロスアテンション・アダプタ**を接続し、**3 段階学習**（低解像度フリーズ事前学習 → 高解像度マルチタスク → SFT）と **`<box>` / `<ref>` 特殊トークン**による細粒度グラウンディングを組み合わせた、Qwen-VL シリーズの最初の論文。総パラメータ 9.6B で当時の同規模 LVLM を画像キャプション・VQA・グラウンディングの広範なベンチマークで上回り、後の Qwen2-VL / Qwen2.5-VL の基礎を築いた。

## 背景と問題意識

2023 年中頃、大規模言語モデル（LLM, Large Language Model：テキストのみで学習された汎用文生成・理解モデル）に視覚信号を扱う能力を付加した **LVLM**（Large Vision-Language Model, 大規模視覚言語モデル）の研究が一気に加速していた。代表例として Flamingo、BLIP-2、LLaVA、MiniGPT-4、Kosmos-2、Shikra などが立て続けに発表されていたが、オープンソース LVLM には以下の弱点が残されていた：

1. **学習が不十分でプロプライエタリ・モデル（GPT-4V など）に大きく後れを取っている**。
2. **細粒度視覚理解（fine-grained visual understanding）が弱い**。具体的には、画像中の特定領域を矩形（バウンディング・ボックス）で指し示す **グラウンディング**（grounding, 言葉と画像領域の対応付け）能力、および画像内のテキストを読む **OCR**（Optical Character Recognition, 光学文字認識）能力が、ほとんどのオープンソース LVLM で実用水準に達していなかった。
3. **中国語などの非英語マルチモーダル能力が脆弱**であり、特に中国語コミュニティのニーズに応えられていなかった。

Alibaba の Qwen チームは、自社の LLM である **Qwen-7B**（2023 年 8 月公開、約 7B パラメータの中英バイリンガル LLM）を土台に、この 3 つの弱点を一気に解決する LVLM を目指して Qwen-VL を設計した。これが Qwen-VL シリーズの初代であり、後に Qwen2-VL（2024）、Qwen2.5-VL（2025）へと発展する系譜の出発点である。

## 提案手法 / 主張

### アーキテクチャ：3 コンポーネントで合計 9.6B

| コンポーネント | 中身 | 役割 | パラメータ |
| --- | --- | --- | --- |
| **Visual Encoder** | OpenCLIP **ViT-bigG**（パッチサイズ 14） | 画像をパッチ分割しトークン化 | 1.9B |
| **Position-aware VL Adapter** | 単層クロスアテンション + 学習可能な **256 個のクエリ** + 2D 絶対位置エンコーディング | 可変長の ViT トークンを **固定長 256** に圧縮 | 0.08B |
| **LLM** | **Qwen-7B**（中英バイリンガル LLM） | テキスト生成・理解 | 7.7B |
| 合計 | | | **9.6B** |

**位置認識アダプタ（Position-aware VL Adapter）の肝**は、(i) BLIP-2 や Q-Former と同じく学習可能なクエリで ViT 出力を圧縮するが、(ii) クエリ-キー対に **2D 絶対位置エンコーディング**を加えることで「アダプタで圧縮しても位置情報が消えない」ことを保証している点。これが後段のグラウンディングを支える。

### 入出力仕様：4 種類の特殊トークン

| トークン対 | 用途 |
| --- | --- |
| `<img></img>` | 画像特徴系列の開始・終了 |
| `<box></box>` | バウンディング・ボックス座標 `(x1,y1),(x2,y2)` を囲む。座標は **画像サイズに関係なく [0, 1000) の文字列**として正規化 |
| `<ref></ref>` | バウンディング・ボックスが指す対象の記述語・句を囲む |

座標を専用語彙ではなく **数値文字列**として扱うのが特徴で、語彙拡張なしでグラウンディング能力を獲得できる。`[0, 1000)` 正規化は後の Qwen2-VL / Qwen2.5-VL でも踏襲される設計。

### 学習：3 段階パイプライン

<figure>

![](../../raw/assets/qwen-vl/fig2.png)

<figcaption>図3（再掲）: Qwen-VL シリーズの 3 段階学習パイプライン。Stage1 は LLM を凍結し低解像度（224²）の画像-テキスト対で ViT とアダプタを学習。Stage2 は全モジュールを高解像度（448²）でマルチタスク学習。Stage3 は ViT を凍結し SFT を行い Qwen-VL-Chat を得る。</figcaption>
</figure>

**Stage 1: Pre-training（事前学習）**
- データ：Web クロールの **画像-テキスト対 50 億 → 浄化後 14 億**（英語 77.3%、中国語 22.7%）。LAION-en/zh、LAION-COCO、DataComp、Coyo、CC12M/3M、SBU、COCO Caption + 社内中国語データ
- 解像度 224×224、LLM を**凍結**し ViT + アダプタのみ学習
- 学習率 2e-4、バッチサイズ 30,720、50,000 ステップ、約 15 億サンプル、約 5,000 億トークン消費
- 目的関数：テキスト・トークンの交差エントロピー（弱教師あり事前学習, [[weakly-supervised-pretraining]]）

**Stage 2: Multi-task Pre-training（マルチタスク事前学習）**
- 解像度を **448×448** に引き上げ、**LLM の凍結を解除**し全パラメータを学習
- **7 タスクを同時学習**（合計約 77M サンプル）：
  - Captioning 19.7M、VQA 3.6M、Grounding 3.5M、Refer Grounding 8.7M、Grounded Captioning 8.7M、OCR 24.8M、Pure-text 7.8M
- グラウンディング・データには **GRIT**（Kosmos-2 が公開）を採用、参照グラウンディングには Visual Genome、RefCOCO/+/g を使用
- OCR データには Common Crawl の pdf/HTML と、自然風景背景の合成 OCR データ（SynthDoG-en/zh）を含む
- 交互配置（interleaved）の系列長 2048 にパック

**Stage 3: Supervised Fine-tuning（教師あり微調整）**
- 命令調整データ **350K**（手動アノテーション、モデル生成、戦略的連結で構築）
- 視覚エンコーダを凍結、LLM + アダプタを学習
- 単一画像チャットだけでなく、位置特定・複数画像理解、多言語対話、純テキスト対話を混合
- ChatML フォーマット（`<im_start>`, `<im_end>` トークン）で会話を構造化
- 出力された対話モデルが **Qwen-VL-Chat**

### 重要な設計判断（アブレーションから）

- **クエリ数 = 256**：64/128/256/512 で比較した結果、少なすぎると情報損失、多すぎると収束困難。Stage2 で 448² 入力時の ViT 出力長は (448/14)²=1024 なので、256 は約 1/4 圧縮となる
- **Window Attention は採用せず Global Attention**：448² ではほぼ同速で損失が悪化するため。896² では Window でも遅すぎる
- **Qwen-7B の中間チェックポイント**で初期化：Qwen-VL と Qwen-7B がほぼ同時期に開発されたため

## 実験結果と知見

### 画像キャプション・一般 VQA（表4）

| ベンチマーク | Qwen-VL | 競合 |
| --- | --- | --- |
| Flickr30K (0-shot, CIDEr) | **85.8** | Flamingo-80B 67.2、InstructBLIP-13B 82.8 |
| Nocaps (0-shot, CIDEr) | 121.4 | InstructBLIP-13B 121.9 |
| VQAv2 | **79.5** | Flamingo-80B 56.3、Shikra-13B 77.36 |
| OKVQA | **58.6** | Flamingo-80B 50.6 |
| GQA | **59.3** | InstructBLIP-13B 49.5 |
| ScienceQA-Img (0-shot) | **67.1** | InstructBLIP-13B 63.1 |
| VizWiz (0-shot) | 35.2 | InstructBLIP-13B 33.4 |

**Flickr30K で 80B の Flamingo を 18.6 ポイント上回り**、9.6B モデルとして当時のゼロショット最強クラスを記録。

### テキスト指向 VQA（表5）

| ベンチマーク | Qwen-VL | 競合 |
| --- | --- | --- |
| TextVQA | **63.8** | mPLUG-DocOwl 52.6、InstructBLIP-13B 50.7 |
| DocVQA | 65.1 | Pix2Struct-Large 76.6（専門モデル） |
| ChartQA | 65.7 / Chat 66.3 | Pix2Struct-Large 58.6 |
| AI2D | **62.3** | Pix2Struct-Large 42.1 |
| OCR-VQA | **75.7** | Pix2Struct-Large 71.3 |

**汎用 MLLM として初めて TextVQA で 60 を超えた**世代のモデル。OCR データ 24.8M の効果が顕著。

### 参照表現理解（表6）

| ベンチマーク | Qwen-VL-7B | Shikra-13B |
| --- | --- | --- |
| RefCOCO val | **89.36** | 87.83 |
| RefCOCO+ val | **83.12** | 82.89 |
| RefCOCOg val | **85.58** | 82.64 |
| GRIT refexp | **78.22** | 69.34 |

汎用モデルの中では Shikra-13B を上回り、専門モデル（G-DINO-L, UNINEXT-H, ONE-PEACE）に肉薄。**`<box>`/`<ref>` トークンと座標文字列正規化だけで、位置語彙なしにグラウンディングを成立させた**点が技術的に重要。

### フューショット学習（図4）

OKVQA / VizWiz / TextVQA / Flickr30K の 4 ベンチマークで、Flamingo-9B、OpenFlamingo-9B、IDEFICS-9B を上回り、Flamingo-80B / IDEFICS-80B に匹敵。RICES 等の洗練手法ではなく単純なランダム例選択での結果。

### 命令追従ベンチマーク（表7）

| ベンチマーク | Qwen-VL-Chat | 競合最良 |
| --- | --- | --- |
| TouchStone En | **645.2** | mPLUG-Owl 605.4 |
| TouchStone Cn | **401.2** | （他モデルは中国語未対応） |
| SEED-Bench All | **58.2** | InstructBLIP 53.4 |
| MME Perception | **1487.58** | InstructBLIP 1212.82 |
| MME Cognition | **360.71** | InstructBLIP 291.79 |

**中国語マルチモーダル対話の事実上の標準**となり、その後 1 年以上にわたって VisualGLM、CogVLM などの中国語 MLLM の比較基準となった。

## 限界・批判的視点

- **固定解像度 448×448**：高解像度（896² など）は計算コストの観点で見送られたが、後の DocVQA や高解像度画像理解では他モデル（Pix2Struct, mPLUG-DocOwl 系列など）に劣後する場面が残った。後継 Qwen2-VL の **Naive Dynamic Resolution** と **M-RoPE**、Qwen2.5-VL のさらなる動的解像度対応で本質的に解決される
- **256 クエリ圧縮**：14M ピクセル相当の情報を 256 トークンに圧縮するため、文書のような情報密度の高い画像では情報損失が懸念される。後の InternVL シリーズが採用する **Pixel Shuffle**（[[entities/internvit-300m]]）や Qwen2-VL の動的トークン数とは設計思想が異なる
- **Stage1 で LLM 凍結**：純粋なテキスト能力との競合を避ける設計だが、LLM 側で視覚との接続を学習できない。InternVL 3 が提唱する [[entities/internvl-3|Native Multimodal Pre-Training]]（最初から LLM と Vision を共学習）とは対照的な保守的設計
- **学習データの公開度**：社内データ（中国語 220M、Pure-text 7.8M、SFT 350K）が含まれており完全な再現は困難
- **モデル規模が 1 種類のみ**：Mini-InternVL ([[entities/mini-internvl]]) や InternVL 2.5 ([[entities/internvl-2-5]]) のような複数サイズ展開がない。後継 Qwen2-VL で 2B / 7B / 72B の 3 サイズに拡張される

## CV 分野における意義

Qwen-VL は、当時急速に発展していた MLLM 分野で **「中国語＋細粒度グラウンディング＋OCR」の 3 つを同時に解決した最初のオープンソース MLLM** として位置付けられる。これは、(i) 中国語コミュニティに本格的な MLLM 基盤を提供したという意味でも、(ii) `<box>` / `<ref>` の特殊トークン設計を MLLM 業界の事実上の標準に押し上げたという意味でも、また (iii) 後続の Qwen2-VL、Qwen2.5-VL を経て今日の Qwen ファミリーの基盤を作ったという意味でも、CV 史における重要な分岐点である。InternVL ([[entities/internvl]]) と並ぶ、2023〜2025 年のオープンソース MLLM 2 大系譜の片翼を成す。

## 用語と略称

- **LVLM** = Large Vision-Language Model（大規模視覚言語モデル）。LLM に視覚入力を扱う能力を付加したもの。**MLLM**（Multimodal LLM）と呼ばれることも多い
- **LLM** = Large Language Model（大規模言語モデル）
- **ViT** = Vision Transformer。画像をパッチに分割し Transformer で処理する画像分類モデル
- **ViT-bigG** = OpenCLIP が公開した約 1.9B パラメータの巨大 ViT
- **OpenCLIP** = LAION による OSS の CLIP 再実装プロジェクト
- **CLIP** = Contrastive Language-Image Pre-training（画像とテキストを対照学習で揃える基盤モデル）
- **VQA** = Visual Question Answering（画像に対する質問応答）
- **OCR** = Optical Character Recognition（光学文字認識、画像中のテキストを読み取る）
- **SFT** = Supervised Fine-Tuning（教師あり微調整）。命令調整の第一段階
- **ChatML** = OpenAI が提案した対話マークアップ言語形式
- **CIDEr** = Consensus-based Image Description Evaluation（画像キャプション評価指標、人手アノテーションとの一致度を測る）
- **ANLS** = Average Normalized Levenshtein Similarity（編集距離ベースの DocVQA 用評価指標）
- **EM** = Exact Match（厳密一致）
- **MME** = Multimodal Evaluation（知覚 14 + 認知 4 のサブタスクで MLLM を Yes/No 評価するベンチマーク）
- **TouchStone** = Alibaba 提案のオープンエンド VL 命令追従ベンチマーク
- **SEED-Bench** = MLLM 評価用の 19K 多肢選択質問データセット（12 次元の空間・時間理解）
- **GRIT** = Grounded Image-Text dataset（Kosmos-2 公開、画像領域とフレーズの対応付きデータ）
- **RefCOCO/+/g** = 参照表現理解の 3 大ベンチマーク（特定領域を自然言語で指す問題）
- **Grounding** = 自然言語で記述された対象を画像領域（バウンディング・ボックス）として特定するタスク
- **MoE** = Mixture-of-Experts（混合エキスパート、入力ごとに専門家を切り替える疎モデル）
- **Q-Former** = BLIP-2 の Querying Transformer（凍結 ViT と凍結 LLM を橋渡しする学習可能なクエリ）

## 関連ページ

- [[entities/qwen-vl]] — Qwen-VL / Qwen-VL-Chat のエンティティ・ページ
- [[translations/qwen-vl]] — 本論文の本文翻訳
- [[concepts/foundation-model]] — 視覚言語基盤モデルとしての位置付け
- [[concepts/weakly-supervised-pretraining]] — Stage1 の弱教師あり画像-テキスト対学習
- [[concepts/zero-shot-transfer]] — Flickr30K / Nocaps / VizWiz でのゼロショット評価
- [[concepts/alignment-tuning]] — Stage3 SFT の位置付け
- [[entities/internvl]] — 同時期の対抗系譜 InternVL シリーズ初代
- [[entities/internvl-3]] — Native Multimodal Pre-Training を提唱した InternVL 系の後継
- [[entities/internvl-3-5]] — 2025 年時点の InternVL 系最新版（Cascade RL）
- [[entities/internvit-300m]] — InternVL 系の軽量 ViT（Pixel Shuffle 採用、設計思想の対比）
