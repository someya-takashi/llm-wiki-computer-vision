---
type: entity
entity_kind: model
aliases: [CLIP, OpenCLIP, Contrastive Language-Image Pre-training]
tags: [weakly-supervised, vision-language, foundation-model, openai, contrastive-learning]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/contrastive-learning]], [[concepts/zero-shot-transfer]], [[concepts/foundation-model]], [[concepts/vision-transformer]], [[concepts/promptable-segmentation]]
sources: [[sources/clip]]
updated: 2026-05-26
---

# CLIP / OpenCLIP

## 概要

**CLIP** = **Contrastive Language-Image Pre-training**。OpenAI の Radford ら（2021）が発表した、**画像-テキスト対の対比学習**による事前学習法。CV における**現代マルチモーダル AI の起点**となった歴史的論文。

- 論文: "Learning Transferable Visual Models From Natural Language Supervision"
- arXiv: 2103.00020（2021 年 2 月）→ ICML 2021
- OpenAI コード・モデル: <https://github.com/OpenAI/CLIP>
- 詳細解説: [[sources/clip]] / 翻訳: [[translations/clip]]

**OpenCLIP** は LAION コミュニティが公開データ（LAION-2B / LAION-5B）で CLIP を再現・拡張した版で、現在は実質的な「公開 CLIP」として広く使われている：

- コード: <https://github.com/mlfoundations/open_clip>
- 主要メンバー: Ilharco, Schuhmann ら

CLIP は後の **DALL-E, Stable Diffusion, BLIP, LLaVA, GPT-4V, Claude Vision, SigLIP, PE** など、ほぼすべての現代マルチモーダル AI の出発点。

---

## なぜ重要か

CLIP は CV に複数の paradigm shift をもたらした：

1. **「自然言語の教師信号 + 対比学習」**による事前学習の有効性を実証
2. **ゼロショット転移**を CV の標準評価プロトコルに（[[concepts/zero-shot-transfer]]）
3. **プロンプトエンジニアリング**を CV に持ち込み
4. **画像とテキストの結合埋め込み空間**を実用化（後のマルチモーダル AI の基盤）
5. **頑健性**: ゼロショット CLIP は ImageNet 派生分布で標準モデルを大幅に上回り、robustness gap を最大 75% 削減

---

## 学習設定（オリジナル CLIP）

| 項目 | 値 |
|---|---|
| 学習データ | **WIT**（4 億画像-テキスト対、OpenAI 内部）[[entities/wit-400m]] |
| 学習目的 | **対称的 InfoNCE**（画像 → テキスト + テキスト → 画像） |
| バッチサイズ | **32,768**（巨大バッチが性能の鍵） |
| エポック | 32 |
| 温度 τ | **学習可能**（log-parameterized、100 倍以下にクリップ） |
| Optimizer | Adam + decoupled weight decay |
| Mixed precision | あり |

### 損失関数

```python
# CLIP の擬似コード（論文 図3）
I_e = l2_normalize(image_encoder(image), axis=-1)
T_e = l2_normalize(text_encoder(text), axis=-1)
logits = I_e @ T_e.T * exp(τ)
labels = arange(n)
loss = (CE(logits, labels, axis=0) + CE(logits, labels, axis=1)) / 2
```

詳細: [[concepts/contrastive-learning]]

---

## モデルファミリ（CLIP）

### 画像エンコーダ: ResNet 系（5 モデル）

| Model | Embed dim | Input res | Notes |
|---|---|---|---|
| RN50 | 1024 | 224 | ベース |
| RN101 | 512 | 224 | 深い |
| RN50x4 | 640 | 288 | EfficientNet 風 4x scale |
| RN50x16 | 768 | 384 | 16x scale |
| RN50x64 | 1024 | 448 | 64x scale、最大 |

ResNet には注意プーリング（attention pooling）と ResNet-D 改善、blur pooling を導入。

### 画像エンコーダ: ViT 系（4 モデル）

| Model | Embed dim | Input res | Layers | Width | Heads |
|---|---|---|---|---|---|
| ViT-B/32 | 512 | 224 | 12 | 768 | 12 |
| ViT-B/16 | 512 | 224 | 12 | 768 | 12 |
| ViT-L/14 | 768 | 224 | 24 | 1024 | 16 |
| **ViT-L/14@336px** | **768** | **336** | **24** | **1024** | **16** |

論文の「CLIP」と呼ぶときは通常 **ViT-L/14@336px**（最良モデル）を指す。

### テキストエンコーダ

- 12 層 × 512 幅 × 8 ヘッド Transformer（**6300 万パラメータ**）
- BPE トークナイザ（語彙 49,152）、最大系列長 76
- `[EOS]` トークンの最終層出力をテキスト表現として使用
- 結合埋め込み空間に**線形射影**

---

## 主要結果

### ImageNet ゼロショット（§3.1）

| Method | ImageNet zero-shot top-1 |
|---|---|
| Visual N-Grams (2017) | 11.5% |
| ResNet-50 (教師あり、128 万訓練例) | 76.2% |
| **CLIP ViT-L/14@336px (ゼロショット)** | **76.2%** |

**訓練例 0 で ResNet-50 と同等** ― これが CLIP の最も印象的な結果。

### 27 データセット平均（§3.2）

CLIP ViT-L/14@336px は線形プローブで **Noisy Student EfficientNet-L2 を平均 +5% 上回り**、当時の SOTA。CLIP ViT は CLIP ResNet より **3 倍計算効率が良い**。

### 分布シフトへの頑健性（§3.3）

| Model | ImageNet | IN-V2 | IN-A | IN-R | ObjectNet | IN-Sketch |
|---|---|---|---|---|---|---|
| NS EfficientNet-L2 | 88.3 | 80.2 | 84.9 | 74.7 | 68.5 | 47.6 |
| ResNet-101 | 77.4 | 65.5 | 6.7 | 37.6 | 32.6 | 25.3 |
| **Zero-Shot CLIP** | 76.2 | **70.1** | **77.2** | **88.9** | **72.3** | **60.2** |

**ImageNet-A で CLIP 77.2 vs RN-101 6.7** という圧倒的な差。CLIP は ImageNet で訓練していないため、ImageNet 固有の偽相関を学ばず汎化する。

### Few-shot との比較

**ゼロショット CLIP ≈ 4-shot ロジスティック回帰**（同じ CLIP 特徴量空間で）。自然言語による概念の明示的指定が、画像例からの暗黙的推論より効率的。

---

## OpenCLIP（公開再現）

OpenCLIP は LAION コミュニティが行った CLIP の公開データ再現：

| Model | Arch | Data | ImageNet zero-shot | ImageNet linear |
|---|---|---|---|---|
| OpenCLIP ViT-B/32 | ViT-B/32 | LAION-400M | 62.9 | 76.6 |
| OpenCLIP ViT-L/14 | ViT-L/14 | LAION-2B | 75.5 | 84.4 |
| OpenCLIP ViT-H/14 | ViT-H/14 | LAION-2B | 78.0 | 84.4 |
| **OpenCLIP ViT-G/14** | ViT-G/14 | LAION-2B | 80.1 | **86.2** |
| **OpenCLIP ViT-bigG/14** | ViT-bigG/14 | LAION-2B | 82.0 | – |

CLIP と OpenCLIP の違い：

| | CLIP (OpenAI) | OpenCLIP (LAION) |
|---|---|---|
| 訓練データ | WIT（非公開、4 億）[[entities/wit-400m]] | LAION-400M / LAION-2B / LAION-5B |
| データ公開 | × | ✓ |
| モデル公開 | ✓ | ✓ |
| 最大モデル | ViT-L/14 (300M) | ViT-bigG/14 (2B+) |
| コミュニティ | OpenAI 内部開発 | オープンコミュニティ |

DINOv3 論文（[[sources/dinov3]]）における**主要ベースライン**は OpenCLIP（特に ViT-G/14, LAION-2B 訓練）。

---

## 系譜と派生

### 先行研究（CLIP が乗り越えた）
- **Visual N-Grams** (Li et al., 2017): ImageNet zero-shot 11.5%
- **VirTex** (Desai 2021): 画像キャプション予測 + SSL
- **ICMLM** (Sariyildiz 2020): inline classification with masked language modeling
- **ConVIRT** (Zhang et al., 2020): CLIP の最も近い祖先、医療画像で対比目的

### 後続・派生
- **DALL-E** (OpenAI, 2021): CLIP と同時期、text-to-image 生成
- **ALIGN** (Google, 2021): 18 億ペア、CLIP より大規模
- **OpenCLIP** (LAION, 2022): 公開データで CLIP 再現＋拡大
- **SLIP** (Mu et al., 2021): CLIP + SSL
- **FLIP** (Li et al., 2023): masked CLIP
- **LiT** (Zhai et al., 2022): テキストエンコーダのみ訓練
- **EVA-CLIP** (Sun et al., 2023): より良いビジョン initialization
- **SigLIP / SigLIP 2** ([[entities/siglip]] / [[sources/siglip]]) (Zhai et al., 2023/2025): **sigmoid 損失で効率化**。CLIP の softmax を pair-wise sigmoid に置き換え、小バッチで圧倒的に勝つ + メモリ効率改善 + ノイズ頑健。**32k バッチで飽和** という発見で「対比学習 = 大バッチ」常識を覆す。4 TPU で 1 日訓練可能（SigLiT）、SO-400M で 83.2% IN-0
- **MetaCLIP / DFN** (Meta, 2023): データキュレーションの改善
- **Perception Encoder (PE)** ([[entities/perception-encoder]] / [[sources/perception-encoder]]) (Meta, NeurIPS 2025): 5.4B unique pairs / 86B samples seen。alignment tuning で 3 バリアント（PEcore / PElang / PEspatial）化
- **CoCa, BLIP, LLaVA** など多数

### マルチモーダル AI への発展

CLIP の image encoder は、その後の **VLM（Vision-Language Models）** の標準的な vision tower になった：

- **BLIP / BLIP-2**: CLIP encoder + Q-Former + LLM
- **LLaVA**: CLIP ViT-L/14 + LLaMA
- **LLaVA-NeXT**: CLIP + **DINOv2** の特徴連結
- **Flamingo, IDEFICS** など
- **GPT-4V, Gemini, Claude Vision** 等の商用 VLM（実装は非公開だが CLIP 系を使用）
- **InternVL** ([[entities/internvl]] / [[sources/internvl]], OpenGVLab Shanghai AI Lab, CVPR 2024): **CLIP の対比学習を 6B 視覚 + 8B 言語ミドルウェアにスケールアップ**。CLIP-L (304M) を InternViT-6B (5.9B) に、CLIP-text (63M) を QLLaMA (8B) に置き換える発想。CLIP と同じ symmetric InfoNCE 損失で **4.98B image-text pairs を Stage 1 で対比訓練**、その後 Stage 2 で生成、Stage 3 で LLM 接続。CLIP の「対比 + zero-shot」を 14B 規模で完成形に

### セグメンテーション基盤モデルへの組み込み

CLIP のテキストエンコーダは、CLIP 自体が分類用でないモデルにも構成要素として組み込まれている：

- **SAM**（[[entities/sam]] / [[sources/segment-anything]] §7.5, §D.5）: SAM のテキストプロンプト処理は CLIP ViT-L/14@336px のテキストエンコーダを使う。訓練時は CLIP **画像**埋め込みをプロンプトに使い、推論時に CLIP **テキスト**埋め込みに置き換える（CLIP の画像-テキスト埋め込みアラインメントを利用）。"a wheel" や "beaver tooth grille" のような自由形式テキストで SAM のマスクを引き出せる。
- **Grounded-SAM**: Grounding DINO のテキスト → ボックス検出器（これも CLIP 系を使う）→ SAM のパイプライン
- **OWL-ViT / OWLv2**: CLIP ベースのオープン語彙検出

---

## CLIP の限界（論文 §6）

- **タスク学習の限界**: 細粒度分類、抽象タスク（カウント, 距離推定）、新規タスク（医療画像）でほぼランダム
- **本当の OOD には弱い**: 手書き MNIST 88%（生ピクセルロジスティック回帰に負ける）
- **生成不可**: 固定クラス集合内の選択のみ、キャプション生成不可
- **データ効率の悪さ**: 128 億画像を 1 秒 1 枚で見ると 405 年
- **少ショットでむしろ精度低下**: ゼロショット → 1-2 shot で性能が下がる（人間と逆）
- **社会的バイアス**: ウェブデータの偏りを継承（[[sources/clip]] §7 で詳細）

---

## 産業応用への影響

CLIP の登場以降、CV/AI 界は劇的に変化：

- **画像検索**: テキストから画像、画像から画像の検索が一気に実用化
- **テキストから画像生成**: Stable Diffusion 等が CLIP テキストエンコーダを使う
- **画像キャプション/VQA**: VLM の vision tower として
- **コンテンツモデレーション**: ゼロショット分類で任意のカテゴリ
- **ロボティクス**: CLIP 特徴量がロボットの perception に使われる
- **マルチモーダル LLM**: GPT-4V, Claude Vision 等の基盤
- **OCR / 文書理解の効率的圧縮**: **[[entities/deepseek-ocr\|DeepSeek-OCR]]**（DeepSeek-AI, 2025, [[sources/deepseek-ocr]]）が **CLIP-large (300M)** を「Visual Knowledge 成分」として組み込み、SAM-base + 2 層 ConvNet（16× 圧縮）+ CLIP-large の **DeepEncoder (約 380M)** で OmniDocBench SOTA（編集距離 0.083）を達成。CLIP の dense global attention で意味抽出を行う新しい応用パターン
- **text-to-image 拡散モデルのテキスト・エンコーダ**: **[[entities/sdxl\|SDXL]]**（Stability AI, 2023, [[sources/sdxl]]）が **CLIP ViT-L + OpenCLIP ViT-bigG の二重エンコーダ**（合計 817M、context dim 2048）を採用し、Stable Diffusion 系統で最大の品質を実現。Midjourney v5.1 を 54.9% で凌駕。CLIP/OpenCLIP の **最大の実用例の 1 つ**で、後の Stable Diffusion 3 / FLUX で T5-XXL 追加へ発展

---

## CLIP を選ぶべきとき / 選ばないとき

### CLIP が最良な選択肢
- **ゼロショット画像分類** が必要
- **テキストから画像検索 / 画像からテキスト検索**
- **VLM の vision tower**
- **画像生成のテキスト条件付け**（Stable Diffusion）
- **オープン語彙の分類タスク**
- **計算予算限定**（CLIP は比較的軽量）

### 他のモデルを検討すべきとき
- **密予測（segmentation, depth）が必要** → **DINOv2/v3**（[[entities/dinov2]], [[entities/dinov3]]）
- **細粒度分類** → DINOv2/v3 や教師ありモデル
- **インスタンス検索** → DINOv2/v3
- **多言語対応が必要** → **SigLIP 2**（[[entities/siglip]]）
- **より大きなモデル/データ** → **PE**（[[entities/perception-encoder]]）

---

## 関連ページ

- [[sources/clip]] — 原論文の詳細解説
- [[translations/clip]] — 原論文全訳（Appendix 込み）
- [[sources/foundational-models-vision-survey]] — Awais et al. の CV 基盤モデル survey で CLIP を「軸 1.1 対比型・汎用」の中核として位置付け
- 概念:
  - [[concepts/contrastive-learning]] — CLIP の学習目的関数
  - [[concepts/zero-shot-transfer]] — CLIP が CV に持ち込んだ評価パラダイム
  - [[concepts/weakly-supervised-pretraining]] — CLIP が代表する系統
  - [[concepts/foundation-model]] — CLIP は CV 初の本格的基盤モデル
  - [[concepts/vision-transformer]] — CLIP-ViT のバックボーン
- データ: [[entities/wit-400m]]
- 競合・派生: [[entities/siglip]] / [[entities/perception-encoder]]
- 対抗系統: [[entities/dinov2]] / [[entities/dinov3]]（純粋 SSL）
