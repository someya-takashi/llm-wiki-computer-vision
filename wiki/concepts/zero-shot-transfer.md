---
type: concept
aliases: [Zero-Shot Transfer, ゼロショット転移, zero-shot classification, zero-shot learning]
tags: [paradigm, evaluation, transfer-learning]
related: ["[[weakly-supervised-pretraining]]", "[[foundation-model]]", "[[knn-evaluation-protocol]]", "[[promptable-segmentation]]"]
sources: ["[[sources/clip]]", "[[sources/segment-anything]]"]
updated: 2026-05-26
---

# Zero-Shot Transfer（ゼロショット転移）

## 一言で

**訓練時に一度も見たことがないタスク・データセット・クラスに対して、追加訓練なしで予測を行う**こと。CV ではもともと「未見の物体カテゴリへの汎化」を意味する狭い用語だったが、**CLIP（[[sources/clip]]）**が「未見のデータセットへの汎化」というより広い意味でこの概念を持ち込み、現代の CV 基盤モデル評価の中核プロトコルとなった。NLP の **GPT-2/GPT-3** が示した「事前学習中にタスク学習が起こる」という発見の CV 版。

---

## ゼロショットの 3 つの異なる意味

「zero-shot」という用語は文脈で意味が大きく違うので注意：

### 1. 古典的な意味: 未見のクラスへの汎化（zero-shot learning）

[Lampert et al., 2009] の伝統的なゼロショット学習：

- 訓練時に「シマウマ」を見ていない
- でも「シマウマは縞模様の馬」という**属性情報**は知っている
- → 既知クラス（馬、縞模様の物体）の組み合わせとしてシマウマを推論

属性ベースの推論で、補助知識（attribute, word embedding 等）を必要とする。

### 2. CLIP 流の意味: 未見のデータセットへの汎化

[[sources/clip]] が拡張した定義:

- 訓練データセット（CLIP の場合 WIT）でクラス名と画像のペアを大量に見ている
- 推論時、ImageNet 等のクラス名をテキストで提示するだけ
- → 「ImageNet で 1 度も訓練していないのに ImageNet 分類器が即座に動く」

実質的に「**新しいタスクへの転移**」を意味する。CLIP は「タスクを評価する代理として、データセットへの転移を使う」と論文で明示。

### 3. プロンプトベースの意味（NLP）

GPT-2/GPT-3 が示した:
- 「翻訳しろ: Hello → ?」とプロンプトを書く
- LLM はファインチューニングなしで翻訳する
- → 「prompt」で任意のタスクを指示できる

CLIP の「a photo of a {class}」プロンプトは、この NLP の流れを CV に持ち込んだ。

> **補足: 3 つの意味の関係** — CLIP のゼロショットは 2 と 3 の融合。「テキストで指定できる任意のタスクに、追加訓練なしで対応」が CV 基盤モデルの目指す姿となった。

---

## CLIP のゼロショット転移の仕組み

```
ステップ 1: クラス名にプロンプトを付ける
  classes = ["dog", "cat", "bird"]
  prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

ステップ 2: テキストエンコーダで埋め込み
  text_embeddings = TextEncoder(prompts)  # [3, D]

ステップ 3: 画像をエンコード
  image_embedding = ImageEncoder(image)  # [D]

ステップ 4: コサイン類似度で予測
  logits = image_embedding @ text_embeddings.T  # [3]
  prediction = argmax(softmax(logits))
```

これは「**ハイパーネットワーク** = テキストエンコーダ」という解釈ができる：

> テキストエンコーダがテキストから線形分類器の重みを生成し、画像エンコーダの特徴量に対する分類器として機能する。

CLIP 論文 §3.1.2 より。

---

## プロンプトエンジニアリング（Prompt Engineering）

CLIP 論文が CV に導入した重要な発見（NLP 由来）：

| クラス指定の方法 | ImageNet zero-shot 精度 |
|---|---|
| クラス名のみ（`"dog"`） | ベースライン |
| `"a photo of a dog"` プロンプト | **+1.3%** |
| タスク固有プロンプト（例: `"a photo of a {label}, a type of pet"`）| 追加で改善 |
| 80 プロンプトのアンサンブル平均 | **+3.5%** |
| プロンプト+アンサンブル合計 | **約 +5%** |

> **補足: なぜプロンプトが効くか** — 訓練データ WIT で、画像と組になっているテキストはほぼ常に文（「a photo of...」「a picture of...」等）。クラス名単独は分布外の入力で、テキストエンコーダが上手く処理できない。プロンプトで「実際の訓練分布に近い形式」に揃えることで性能が上がる。

### プロンプトアンサンブル

複数のプロンプト変種の埋め込みを**平均化**し、それを単一の分類器として使う：

```python
templates = ["a photo of a {}", "a picture of a {}", "an image of a {}", ...]  # 80 種
embeddings = []
for template in templates:
    text = template.format(class_name)
    embeddings.append(TextEncoder(text))
classifier = mean(embeddings)  # 平均化
```

**計算コスト: 1 プロンプト分**（一度平均すれば使い回せる）。

---

## ゼロショット転移の評価指標

### 精度

通常の top-1, top-5 accuracy。

### Effective Robustness（[Taori et al., 2020]）

**「ID 性能から予測される OOD 性能を超える改善」**。

ImageNet 訓練モデルの ID 精度と OOD 精度には**強い線形関係**がある。これを基準として、その線から「上に外れる」モデルが effective robustness を持つと言える。

CLIP 論文の §3.3 の最重要発見:
- ゼロショット CLIP は effective robustness が圧倒的に高い
- ImageNet 精度に応じて他のモデルが OOD で苦戦する中、CLIP は **robustness gap を最大 75% 削減**

### Relative Robustness

OOD 性能の絶対値そのものの改善。

---

## 何故 CLIP のゼロショットが効くのか

複数の要因の組み合わせ:

1. **対比学習で意味的に整理された埋め込み空間**: 類似画像と類似テキストが近くにある
2. **テキストの汎用性**: 任意のタスクをテキストで記述できる
3. **大規模ウェブデータ**: WIT 4 億ペアに ImageNet 概念は当然含まれる（だから「ゼロショット」と言えるか議論の余地あり）
4. **ハイパーネットワーク的解釈**: テキストエンコーダが分類器を生成する柔軟性

> **補足: 「真のゼロショット」議論** — CLIP の事前学習データに ImageNet クラスの画像が含まれている可能性は当然ある（dog, cat の写真は無数に web にある）。なので「ImageNet ゼロショット」は厳密にはタスク非依存の事前学習からの転移であり、未見クラスを学んだわけではない。CLIP 論文も §6 でこの点を認めている。

---

## ゼロショット転移と他の評価プロトコルの比較

| プロトコル | 説明 | 強み | 弱み |
|---|---|---|---|
| **Zero-Shot** | プロンプトのみ、追加訓練なし | 即座に使える、汎用性 | タスク学習に上限 |
| **k-NN** (詳細: [[knn-evaluation-protocol]]) | 凍結特徴量 + 最近傍 | ハイパラ不要、表現の素質を測れる | データセット保持必要 |
| **Linear Probing** | 凍結特徴量 + 線形分類器 | 表現の線形分離性を測る | LR 等のチューニング必要 |
| **Few-Shot** | 数例から学習 | 実用に近い | サンプル選びでバラつく |
| **Fine-tuning** | 全層を再学習 | 最終性能が高い | 大量のラベル付きデータと計算が必要 |

CLIP は **zero-shot, linear probing, fine-tuning すべてで強い**ことを実証した。

---

## CLIP 以降のゼロショット転移の展開

CLIP がパラダイムを確立した後、ゼロショット転移は CV 基盤モデルの標準評価に：

- **OpenCLIP** (2022): 公開データで CLIP 再現、より大きなモデル
- **SigLIP / SigLIP 2** (Google, 2023/2024): 効率化、多言語化
- **PE** (Meta, NeurIPS 2025, [[sources/perception-encoder]]): 5.4B unique image-text pairs を 86B samples seen まで訓練。**JFT-3B / WebLI なしで 3 年ぶりに対比的ゼロショット SOTA を奪還**
- **Qwen3-VL** (Qwen Team Alibaba Group, 2025 Nov, [[sources/qwen3-vl]] / [[entities/qwen3-vl]]): **256K ネイティブ → 1M YaRN 外挿（2 時間動画）でのゼロショット転移、多言語 OCR 39 言語、9-DoF 3D グラウンディング、GUI エージェントの商用級到達**。**Needle-in-a-Haystack で 256K 完全 100% / 1M で 99.5%** という長系列モデリング能力の新ベンチマークを設定（YaRN ベース位置拡張）。**多言語 OCR 10 → 39 言語**（仏・独・伊・西・葡・露・日・韓・越・アラに加え、スワヒリ・ヘブライ・チェコ・ウルドゥー・タイ・スウェーデン・セルビア・デンマークなど 29 言語）、32/39 言語で 70% 超精度を達成しゼロショット多言語転移を実用化。**ODinW-13 で 48.6 mAP**（Qwen2.5-VL 43.1 から +5.5、Gemini 2.5-Pro 33.7 から +14.9）でオープン語彙物体検出の SOTA。**CountBench 93.7** でゼロショット物体カウント、**RefCOCO-avg 92.1** で参照表現理解 SOTA。**9-DoF 3D グラウンディング**（Omni3D 形式）で単眼画像から物体の 3D 空間位置をゼロショット推定。**GUI エージェントの劇的進化**: ScreenSpot Pro **62.0**（Qwen2.5-VL から +18.4）、OSWorld **38.1**（Qwen2.5-VL 8.83 から 4.3× 飛躍、Claude Opus 4 44.4 に肉薄）、AndroidWorld **63.7**（Qwen2.5-VL 35% から +28.7）。**Charades-STA mIoU 64.8** で時間グラウンディング SOTA（テキスト・ベース時間整合の効果）
- **Qwen2.5-VL** (Bai et al., Qwen Team Alibaba Group, 2025 Feb, [[sources/qwen2-5-vl]] / [[entities/qwen2-5-vl]]): **GUI エージェントへのゼロショット転移を一気に商用級に**。**ScreenSpot Pro で 43.6%（Qwen2-VL 1.6% の +42 ポイント、27× 飛躍）** で、これまで MLLM がほぼ不可能だった「プロ向け高解像度 UI でのゼロショット要素グラウンディング」を実現。**ODinW でオープン語彙物体検出 43.1 mAP**（InternVL2.5-78B 31.7 を +11.4）で 10,000+ カテゴリへのゼロショット転移。**CountBench で 93.6 SOTA** によりゼロショット物体カウント（Molmo-72B 91.2 を +2.4、GPT-4o 87.9 を +5.7）。**MRoPE Aligned to Absolute Time** によって異なる FPS の動画への時間グラウンディング転移を実現、**Charades-STA mIoU 50.9 SOTA**（GPT-4o 35.7 を +15.2）。**点ベース・グラウンディング（PointGrounding 67.5）** で過去にはバウンディング・ボックスでは表現困難だった物体の細部位置特定を可能に。多言語 OCR（仏・独・伊・西・葡・露・日・韓・越・アラ）で **MTVQA 31.7** を達成し、多言語ゼロショット転移を継続
- **Qwen2-VL** (Wang et al., Qwen Team Alibaba Group, 2024 Sept, [[sources/qwen2-vl]] / [[entities/qwen2-vl]]): **「任意解像度・任意アスペクト比」へのゼロショット転移**。Naive Dynamic Resolution と M-RoPE の組み合わせで、**学習時 16K トークン上限なのに推論時 80K トークンまで頑健**という長さ外挿能力を実証。これは「学習分布の外への系列長外挿」というゼロショット転移の新次元。多言語 OCR でも 8 言語中 7 言語で GPT-4o を上回り、特に欧州主要言語と中・日・韓・越での **ゼロショット多言語転移**を実現。動画理解では学習時 16K トークン上限ながら 1 時間動画も処理可能（評価時 768 フレーム制限）。**ゼロショット EgoSchema 77.9 で GPT-4o 72.2 を +5.7 上回る**
- **Qwen-VL** (Bai et al., Alibaba Group, 2023 Aug, [[sources/qwen-vl]] / [[entities/qwen-vl]]): **MLLM タスクへのゼロショット転移を `<box>`/`<ref>` 特殊トークンで細粒度化**。**Flickr30K (0-shot CIDEr) で 85.8** を達成（Flamingo-80B 67.2 を 9.6B で上回る当時 SOTA）、ScienceQA-Img (0-shot) 67.1、VizWiz (0-shot) 35.2、Nocaps (0-shot CIDEr) 121.4。**ゼロショット参照表現理解**でも RefCOCO val 89.36 など Shikra-13B を 9.6B で超える。**バウンディング・ボックス座標を [0, 1000) の文字列として表現するだけで、グラウンディング能力を獲得できる**ことを実証 — 位置語彙の追加なし、通常のテキスト生成と同じ仕組みでゼロショット転移可能。後の Qwen2-VL / Qwen2.5-VL に継承される設計の出発点
- **InternVL 1.5** (OpenGVLab Shanghai AI Lab, 2024 April, [[sources/internvl-1-5]] / [[entities/internvl-1-5]]): **MLLM タスクへの「ゼロショット転移」**。InternViT-V1.5 は **dynamic 448 で訓練したが、テスト時に zero-shot で 4K（40 タイル）まで拡張可能**。さらに **single-image 訓練のみで multi-image 対話に zero-shot 対応**。OCR・中国語・MathVista で GPT-4V を上回り、「商用フロンティアに対する zero-shot」の新次元を示した
- **InternVL 2.5** (OpenGVLab Shanghai AI Lab, 2024 Dec, [[sources/internvl-2-5]] / [[entities/internvl-2-5]]): **MMMU 70% を超えた初のオープン MLLM**（70.1%、GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 超え）。**Test-Time Scaling**（Chain-of-Thought + Majority Voting）で **直接応答 → CoT で MMMU +3.7 改善**、その後 majority voting でさらに改善。**「OS MLLM もテスト時スケーリングの恩恵を受けられる」** ことを初めて実証。Qwen2-VL の 1/12 の訓練トークン（120B vs 1.4T）で同等以上の性能、**Progressive Scaling Strategy** で学習効率を抜本改善
- **InternVL 3** (OpenGVLab Shanghai AI Lab, 2025 Apr, [[sources/internvl-3]] / [[entities/internvl-3]]): **MMMU 72.2 で再び SOTA**（GPT-4o 70.7 / Qwen2.5-VL-72B 68.2 超え）。Test-Time Scaling では **VisualPRM-8B（Visual Process Reward Model）+ Best-of-8** を導入、各推論ステップに +/- スコアを付与する process reward model 方式。**小型モデルで顕著な改善**（InternVL3-1B Overall 25.1 → 35.0 で +9.9）。「**計算リソースをモデルサイズで使うか、推論回数で使うか**」という新トレードオフを提示。**OpenAI o1 / DeepSeek R1 系の test-time scaling を MLLM へ本格適用した初の事例**
- **MPO** (Wang et al., OpenGVLab Shanghai AI Lab, 2024 Nov, [[sources/mpo]] / [[entities/mpo]]): **「MLLM の CoT 推論で性能が悪化する」分布シフト問題を初めて解決した論文**。Mixed Preference Optimization = **DPO + BCO + SFT loss** の 3 損失混合で、CoT 性能を Direct 性能より良くする。InternVL2-8B-MPO で **MathVista +8.7 / M3CoT +19.9**、CoT が Direct より +2.0 という「正しい挙動」を実現。InternVL 3 の Stage 3 で正式採用、+4.1～+4.5 ポイント推論改善（38B/78B）。**「マルチモーダル推論能力強化」 用の PO** という MLLM コミュニティの新方向性を開いた
- **InternVL 3.5** (OpenGVLab Shanghai AI Lab, 2025 Aug, [[sources/internvl-3-5]] / [[entities/internvl-3-5]]): **Cascade Reinforcement Learning + Test-Time Scaling の集大成**。Test-Time Scaling では **Deep Thinking（Thinking モード起動の step-by-step 推論）+ Parallel Thinking（VisualPRM-v1.1 critic で Best-of-N）** を同時実装。**小型モデルで効果大**（InternVL3.5-1B が Bo8 で +9.5 ポイント、最大）。さらに **Cascade RL（MPO offline + GSPO online）** で SFT 単独から +12 ポイント改善（2B モデル）。**MMMU 77.7、MathVista で GPT-5 を +0.8、VSI-Bench で GPT-5 を +32 圧倒、商用との差 3.9% に縮める**
- **DINOv3 + dino.txt** (2025): DINO 系統に後付けでテキスト整合性を追加し、ゼロショットを獲得
- **SAM** (Meta, 2023, [[entities/sam]] / [[sources/segment-anything]]): **セグメンテーションへのゼロショット転移**を確立。プロンプト（点・ボックス・テキスト）の設計でエッジ検出・オブジェクト提案・インスタンスセグメンテーション・テキスト → マスク等を解く。CLIP の「テキストプロンプト → 分類」と並列で、「画像/テキストプロンプト → マスク」というインターフェイス（[[concepts/promptable-segmentation]]）
- **GLIP** (Microsoft, CVPR 2022, [[sources/glip]] / [[entities/glip]]): **物体検出へのゼロショット転移を確立**。物体検出を phrase grounding として再定式化し、box 分類のロジットを region-word alignment スコアに置換することで、テキストプロンプトのみで任意のクラスを検出可能に。COCO ゼロショット 49.8 AP（Faster RCNN 教師あり 42.0 を超える）。**open-vocabulary 検出**パラダイムの祖、Grounding DINO / SAM 3 の源流
- **Grounding DINO** (IDEA + Microsoft, ECCV 2024, [[sources/grounding-dino]] / [[entities/grounding-dino]]): **GLIP × DINO 検出器**。GLIP の grounded pre-training を [[entities/dino-detector|DINO 検出器]] に移植し、tight 3-phase fusion + sub-sentence text representation で発展。COCO ゼロショット **52.5 AP**（GLIP-L 49.8 超え）、ODinW ゼロショット **26.1 mean AP**（Florence 841M を 341M で上回る SOTA）。**SAM 3 の直接の祖**
- **YOLO-World** (Tencent + 華中科技大学, CVPR 2024, [[sources/yolo-world]] / [[entities/yolo-world]]): **YOLO + open-vocabulary 検出**。YOLOv8 を CLIP text encoder + RepVL-PAN で open-vocab 化、**Prompt-Then-Detect パラダイム**でテキスト encoder を推論時に削除、テキスト埋め込みをモデル重みに re-parameterize。**LVIS ゼロショット 35.4 AP at 52 FPS V100**（Grounding DINO の 34.7× / GLIP の 433× 速度）。「精度志向の GLIP / Grounding DINO」に対する「実用志向」の対抗路線、エッジデバイスでの open-vocab 検出を可能に
- **Grounding DINO 1.5** (IDEA Research, 2024 May, [[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]]): **Pro/Edge 双子スイート**。**Pro**: ViT-L + Grounding-20M で **COCO ZS 54.3 / LVIS-mv 55.7 / ODinW35 30.2 SOTA**（DetCLIPv3 +6.9 AP）。**Edge**: EfficientViT-L1 + efficient feature enhancer（P5 のみ cross-modal 融合）で **Orin NX 10.7 FPS / A100 TRT 75.2 FPS、LVIS-mv 36.2 AP**（YOLO-Worldv2-L 32.9 超え）。**「精度志向と実用志向を 1 モデルスイートで統合」** した IDEA Research の戦略的後継
- **DINO-X** (IDEA Research, 2024 Nov, [[sources/dino-x]] / [[entities/dino-x]]): **Grounding DINO 系統の到達点 = unified perception model**。3 種プロンプト（Text/Visual/Customized）+ 4 種 head（Box/Mask/Keypoint/Language）+ CLIP text encoder + **Grounding-100M**（GD 1.5 の 5×）。**LVIS-mv 59.8 / LVIS rare APr 63.3**（GD 1.6 Pro から +5.8、長尾劇的改善）/ Visual Genome region caption CIDEr **201.8 SOTA**。**Edge**: Knowledge Distillation + FP16 で **Orin NX 20.1 FPS**（GD 1.5 Edge から +87%）かつ LVIS-mv 48.3 AP。**Universal Object Prompt で prompt-free 検出**という新タスク

「**ゼロショット ImageNet 精度**」は今や CV 基盤モデルの de facto 基本指標。同様に「**ゼロショット 23 データセット mIoU**」が SAM 系統の標準指標となった。

---

## ゼロショット転移が苦手なこと

CLIP 論文と後続研究で明らかになった限界：

1. **抽象的・体系的タスク**: 物体カウント、距離推定、形状認識
2. **新規ドメイン**: 医療画像、衛星画像（CLIP は分布外）
3. **手書き文字**（MNIST）: ウェブ画像にほぼ存在しない
4. **細粒度識別**: 似た車種、似た花種（注釈が必要なレベル）

これらでは追加の few-shot や fine-tuning が必要。

---

## 関連ページ

- [[sources/clip]] — ゼロショット転移を CV に持ち込んだ論文
- [[sources/segment-anything]] — セグメンテーションへゼロショット転移を拡張した論文
- [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/qwen-vl]] / [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]] / [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/internvl-2-5]] / [[entities/internvl-3]] / [[entities/internvl-3-5]] — 分類・MLLM でのゼロショット転移ができる主要モデル
- [[entities/sam]] — セグメンテーションでのゼロショット転移ができるモデル
- [[concepts/contrastive-learning]] — CLIP のゼロショットを可能にする学習方式
- [[concepts/promptable-segmentation]] — SAM のゼロショットを可能にするタスクパラダイム
- [[concepts/weakly-supervised-pretraining]] — ゼロショット転移を可能にする事前学習の枠組み
- [[concepts/foundation-model]] — ゼロショット転移は基盤モデルの中核能力
- [[concepts/knn-evaluation-protocol]] — 代替的な評価プロトコル
