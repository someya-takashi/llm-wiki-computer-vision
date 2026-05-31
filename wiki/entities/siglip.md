---
type: entity
entity_kind: model
aliases: [SigLIP, SigLiT, SigLIP 2, mSigLIP, Sigmoid Loss Image-Language Pre-training]
tags: [weakly-supervised, vision-language, foundation-model, google-deepmind, sigmoid-loss, efficient-training, contrastive-learning, naflex, multilingual]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]], [[concepts/contrastive-learning]], [[concepts/vision-transformer]], [[concepts/zero-shot-transfer]], [[concepts/self-supervised-learning]], [[concepts/masked-image-modeling]], [[concepts/knowledge-distillation]]
sources: [[sources/siglip]], [[sources/siglip-2]]
updated: 2026-05-27
---

# SigLIP / SigLiT / mSigLIP / SigLIP 2

## 概要

**SigLIP** = **Sig**moid **L**oss for Language-**I**mage **P**re-training。**Google DeepMind**（Zhai, Mustafa, Kolesnikov, Beyer, 2023, ICCV）が発表した CLIP 系の根本的効率改善。CLIP の **softmax-based InfoNCE 損失を sigmoid 損失に置き換える** だけで、(1) 小バッチで圧倒的に勝つ、(2) 大バッチでも僅かに勝つ、(3) メモリ効率が劇的に良い、を同時に達成。**SigLIP 2**（2025, Tschannen ら）は多言語化 + dense feature 改善版。

- SigLIP 論文: "Sigmoid Loss for Language Image Pre-Training"（arXiv:2303.15343, ICCV 2023）
- SigLIP 2 論文: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features"（arXiv:2502.14786, 2025）
- 詳細解説: [[sources/siglip]] / 翻訳: [[translations/siglip]]（v1）, [[sources/siglip-2]] / [[translations/siglip-2]]（v2）— **両方 ingest 済み**
- コード: <https://github.com/google-research/big_vision>
- HuggingFace: `google/siglip-base-patch16-224`, `google/siglip2-*` 等

## モデルファミリー

### SigLiT vs SigLIP の区別

| | **SigLiT** | **SigLIP** |
|---|---|---|
| **画像 backbone** | 事前訓練済み + **frozen**（"Locked-image Tuning"）| **ゼロから訓練** |
| **テキスト encoder** | ゼロから訓練 | ゼロから訓練 |
| **アーキテクチャ思想** | LiT（Zhai 2022）の sigmoid 版 | CLIP の sigmoid 版 |
| **訓練コスト** | 極めて低い（4 TPUv4 × 1 日で 79.7% IN-0） | 中（32 TPUv4 × 2 日で 72.1%）|
| **使い道** | 強い既存 backbone を再利用 | 完全 from-scratch |

### SigLIP 初版（2023）の主要モデル

| Model | 画像 ViT | パッチ数（224 解像度） | IN-1k 0-shot |
|---|---|---|---|
| SigLIP B/16 | B | 196 (14×14) | **76.2** |
| SigLIP B/16 (high-res) | B | 1024 | **79.2** |
| SigLIP L/16 | L | 576 | **82.1** |
| **SigLIP SO/400M** ★ | Shape-Optimized 400M | 729 | **83.2** |

**Shape-Optimized 400M ViT** が SigLIP の最強モデル。**5B パラメータ EVA-CLIP より高精度**。

### mSigLIP（多言語版, 2023）

- **WebLI の 100 言語全て** を保持
- 250k トークンの大語彙 + **ボトルネック化トークン埋め込み**（$N \times K$ + $K \times W$, $K \ll W$）でメモリ節約
- **XM3600 text-to-image retrieval で 34.9%**（先行 LiT 28.5% を 6 ポイント超え）
- スケールアップ版 mSigLIP ViT-B で **42.6% IR / 54.1% TR @1** で SOTA

### SigLIP 2（2025, [[sources/siglip-2]] / [[translations/siglip-2]] で原典 ingest 済み）

**「過去 4 年の CLIP/SigLIP 改善を 1 モデルに統合した全部入りレシピ」**。同サイズの SigLIP を全ベンチで上回り、特に dense 予測と RefCOCO で劇的改善。

#### 段階的訓練レシピ

```
[第 1 段階] SigLIP sigmoid loss + LocCa decoder loss を等重みで同時最適化
   - WebLI 10B 画像 / 12B alt-text、109 言語
   - 90% 英語 + 10% 非英語、de-bias 化フィルタ
   - バッチ 32k、合計 40B サンプル、最大 2048 TPUv5e
                          ↓
[第 2 段階, 80% 完了時] SILC/TIPS の self-distillation + masked prediction を追加
                          ↓
[第 3 段階, 95% 完了時] 位置埋め込みリサイズで目標解像度に適応
                          ↓
[追加, B/16・B/32 のみ] ACID 蒸留で 4B サンプル
[並行, NaFlex] 90% 完了時にアスペクト比保持＋可変系列長に切替
```

#### 4 つの主要追加技法

| 技法 | 元論文 | 何が嬉しいか |
|---|---|---|
| **LocCa**（decoder ベース事前学習） | LocCa [NeurIPS 2024] | 位置特定・OCR の劇的改善（RefCOCO +20pt） |
| **SILC + TIPS**（自己蒸留 + マスク予測） | SILC [ECCV 2024], TIPS [ICLR 2025] | dense 予測（segmentation, depth）の改善 |
| **ACID**（能動的データキュレーション蒸留） | Udandarao et al. 2024 | 小型モデル（B/16, B/32）の性能最大化 |
| **NaFlex**（可変アスペクト・可変解像度） | NaViT + FlexiViT 統合 | OCR/文書理解のアスペクト感度問題を解決 |

これらは元々独立に研究されていたが、SigLIP 2 が **段階的アプローチで現実的に統合**。

#### 主要モデルラインナップ

| サイズ | パラメータ | パッチ | 解像度バリアント | NaFlex | ACID |
|---|---|---|---|---|---|
| **B/32** | 86M | 32 | 256 | × | ✓ |
| **B/16** | 86M | 16 | 224 / 256 / 384 / 512 | ✓ | ✓ |
| **L/16** | 303M | 16 | 256 / 384 / 512 | – | × |
| **So400m/14** | 400M | 14 | 224 / 384 | – | × |
| **So400m/16** | 400M | 16 | 256 / 384 / 512 | ✓ | × |
| **g/16** ★ NEW | 1B | 16 | 256 / 384 | – | × |

**g/16（1B パラメータ）が新登場**。SigLIP v1 は最大 So400m まで、v2 で 1B 投入。g サイズのみテキストエンコーダは So400m（非対称）。

#### 主要結果

**ゼロショット分類 IN-1k val**:
- B/16@256: 76.7 → **79.1**（+2.4, ACID 効果大）
- L/16@256: 80.5 → **82.5**（+2.0）
- So/14@384: 83.2 → **84.1**（+0.9）
- **g/16@384: 85.0**（新規）

**Dense 予測**（So/14@224 PASCAL Seg mIoU）: 72.0 → **77.1**（+5.1）

**参照表現理解**（RefCOCO val）:
- B/16: 64.05 → **83.76**（+19.7）
- L/16: 67.33 → **86.04**（+18.7）
- So/16: 64.68 → **86.42**（+21.7）
- これは LocCa decoder の直接効果

**多言語 XM3600**（L/16@256 T→I R@1）: 30.9 → **46.5**（+15.6）、mSigLIP（50.0）に肉薄

**Representation bias 削減**（L/16@256, 性別偏向, 低いほど良い）:
- SigLIP: 35.5% → **SigLIP 2: 7.3%**（−28.2pt）
- de-bias フィルタ [^2] の劇的効果

**VLM ベンチマーク**（PaliGemma 2 レシピ）: 35 個のタスクでほぼ全勝、特に OCR/文書系（DocVQA, TextVQA, InfoVQA）で +4〜10pt

#### LocCa / SILC / TIPS / ACID とは

- **LocCa**（Wan et al., NeurIPS 2024）: Location-aware Captioner、pool 前視覚特徴に decoder を cross-attention で接続、キャプション化 + automatic referring expression prediction + grounded captioning を同時訓練。**decoder はモデルリリースに含まれない**（足場として使う）
- **SILC**（Naeem et al., ECCV 2024）: Self-supervised + Image-Language Contrastive。CLIP に DINO スタイル local-to-global 一貫性損失を追加
- **TIPS**（Maninis et al., ICLR 2025）: Text-Image Pretraining with Spatial awareness。SILC + マスク予測損失で dense feature を強化
- **ACID**（Udandarao et al., 2024）: Active Curation Implicit Distillation。teacher/learner で「learnability」スコアを計算し super-batch から最適バッチを共同選択

これらは [[concepts/self-supervised-learning]] と [[concepts/knowledge-distillation]] の現代的応用。

#### NaFlex の挙動

- **検索ベンチで標準版を上回る**（特に小系列長）
- **OCR/文書系で大幅優位**（HierText, SciCap, Screen2Words）
- **自然画像では B サイズで標準版が僅かに有利**（ACID 蒸留が NaFlex に適用されないため）
- 訓練系列長 $\{128, 256, 576, 784, 1024\}$、**外挿は失敗**

#### 限界

1. **WebLI 自体は非公開**: モデルは Apache 2.0 だがデータは Google 内部のみ
2. **Dense 予測で DINOv3 にまだ劣る**: 改善したが純粋 SSL の壁を越えられず
3. **LocCa decoder は公開せず**: VLM 構築で再利用したくても不可
4. **NaFlex は外挿しない**: 訓練系列長を超えると性能劣化

## CLIP との違い: sigmoid loss

### Softmax (CLIP) vs Sigmoid (SigLIP)

**CLIP の InfoNCE**:

$$-\frac{1}{2|\mathcal{B}|} \sum_i \left( \log \frac{e^{t \mathbf{x}_i \cdot \mathbf{y}_i}}{\sum_j e^{t \mathbf{x}_i \cdot \mathbf{y}_j}} + \log \frac{e^{t \mathbf{x}_i \cdot \mathbf{y}_i}}{\sum_j e^{t \mathbf{x}_j \cdot \mathbf{y}_i}} \right)$$

→ バッチ全体の **softmax 正規化** が 2 方向必要、$|\mathcal{B}|^2$ 行列を実体化

**SigLIP の sigmoid**:

$$-\frac{1}{|\mathcal{B}|} \sum_i \sum_j \log \frac{1}{1 + e^{z_{ij}(-t \mathbf{x}_i \cdot \mathbf{y}_j + b)}}$$

→ 各 (i, j) ペアが独立に **二値分類**、正規化不要

### 何が嬉しいか

| 観点 | CLIP softmax | **SigLIP sigmoid** |
|---|---|---|
| **タスク解釈** | 「正しいペアを選ぶ」（バッチ依存） | 「このペアは正しいか？」（**ペア独立**） |
| **数値安定化** | 最大値減算で追加パス | 不要、`log_sigmoid` で安定 |
| **メモリ** | $\mathcal{O}(\|\mathcal{B}\|^2)$ 行列 | **chunked で $\mathcal{O}(b^2)$** |
| **対称性** | 非対称（2 方向別正規化） | **対称的** |
| **小バッチ性能** | 弱い | **大幅優位** |
| **大バッチ性能** | 同等 | 僅かに優位 |
| **ノイズ頑健性** | 弱い | **強い** |

## 重要な実装テクニック（SigLIP の隠れた貢献）

### 1. 学習可能 bias term（決定的に重要）

シグモイドをそのまま使うと、初期化時の **負例の重い不均衡**（16k バッチで正例 16k vs 負例 268M）が損失を支配 → 大きな初期最適化ステップで訓練不安定化。

解決: **学習可能 bias term $b$** を導入、$b = -10$、温度 $t' = \log 10$ で初期化。

アブレーション（[[sources/siglip]] 表 4）：
- bias なし: IN-0 62.0
- **b=-10, t'=log 10**（推奨）: **63.0** ★
- b=0, t'=log 1: 53.7（崩壊）

**bias term の有無で 10 ポイント差** が出る場合あり。SigLIP 実装時は **絶対忘れてはいけない**。

### 2. Chunked Efficient 実装

データ並列で $D$ デバイス分散時：

1. 各デバイス: ローカル正例 + 隣接負例で per-device 損失計算
2. テキスト表現を **collective_permute** で次デバイスに送る
3. $D$ 回繰り返してすべてのペアをカバー
4. 全デバイスで損失合計

→ **各デバイス最大メモリ $b^2$**（$|\mathcal{B}|^2$ ではない）。

これにより **比較的少数のデバイスで 100 万バッチサイズ訓練に成功**（§4.1）。

### 3. β₂ = 0.95 で大バッチ安定化

大バッチで勾配ノルムのスパイク発生 → 重みの大きな変化 → 訓練不安定化。

**Adam/AdaFactor の β₂ をデフォルト 0.999 → 0.95** に下げるだけで安定化（[Lion 論文] [GIT 論文] でも独立提案）。

SigLIP は **β₂=0.95 をすべての実験で使用**、これが大バッチ vision transformer 訓練全般の標準に。

### 4. 事前訓練 backbone の weight decay 無効化

SigLIP ファインチューニング時：事前訓練 backbone の weight decay を 0 にする（学習率乗数 0.1）→ 性能大幅改善（§4.5 図 4）。

## 主要結果

### バッチサイズ研究（最大の発見）

[[sources/siglip]] §4.1 図 2 左、SigLiT setup:

| Batch Size | Softmax | **Sigmoid** | 差 |
|---|---|---|---|
| 2k | ~81 | **~82** | **+1** |
| 8k | ~82.5 | **~83.5** | **+1** |
| 16k | ~83.5 | ~84 | +0.5 |
| 32k | ~84 | ~84 | 0 |
| 1M | ~84.5 | ~84.5 | ~0（飽和） |

**驚き発見: 32k で飽和**。それ以上拡大しても僅かなブースト、1M でも 256k より良くない。**mSigLIP（多言語）でも 32k で十分**。

→ 「**対比学習を大バッチで訓練するために巨大インフラを買う必要はもうない**」という重要な発見、研究のアクセス可能性を根本的に変える。

### 4 TPU での驚異的効率

| 設定 | 訓練リソース | IN-0 |
|---|---|---|
| **SigLiT B/8 frozen** | **4 TPUv4 × 1 日** | **79.7%** |
| **SigLiT g/14 frozen** | **4 TPUv4 × 2 日** | **84.5%** |
| SigLIP B/16 unlocked | 16 TPUv4 × 3 日 | 71.0% |
| SigLIP B/16 from-scratch | 32 TPUv4 × 2 日 | **72.1%** |
| SigLIP B/16 from-scratch | 32 TPUv4 × 5 日 | **73.4%** |
| 比較: CLIP | 256 TPUv3 × 10 日 | 72.6% |

**SigLIP は CLIP より桁違いに効率的**。

### 大規模スケール結果（表 3）

| Model | ViT | パッチ | IN-1k Val |
|---|---|---|---|
| CLIP | B | 196 | 68.3 |
| OpenCLIP | B | 196 | 70.2 |
| EVA-CLIP | B | 196 | 74.7 |
| **SigLIP** | **B** | **196** | **76.2** |
| **SigLIP** | **B** | **1024** | **79.2** |
| **SigLIP** | **L** | **576** | **82.1** |
| OpenCLIP | G (2B) | 256 | 80.1 |
| EVA-CLIP | E (5B) | 256 | 82.0 |
| **SigLIP** | **SO (400M)** | **729** | **83.2** ★ |

**SO/400M（4 億パラメータ）が 5B EVA-CLIP より高精度** — モデルアーキテクチャ最適化の威力。

### ラベルノイズ頑健性（§4.10）

画像/テキスト/バッチアラインメントを破損させる実験で、sigmoid は softmax より一貫してノイズに頑健（[[sources/siglip]] 図 7）。Web crawl の noisy データに対する重要な実用利点。

## DINOv3 ベンチマークでの SigLIP 2

[[sources/dinov3]] §6 のベンチマーク（SigLIP 2 g/16）：

### 強い領域

- **ImageNet 分類**: 89.1 linear（DINOv3 88.4 を僅差で上回る）
- **OOD 分類** ImageNet-R: 92.2（DINOv3 91.1 を上回る）
- **ゼロショット分類**: 83.1 IN1k（DINOv3 dino.txt 82.3 を上回る）
- **画像-テキスト検索**: COCO で I→T 71.4 / T→I 55.3

### 弱い領域

- **Dense linear probing**: ADE20k 42.7 vs DINOv3 55.9（大差）
- **3D correspondence**: NAVI 49.4 vs DINOv3 64.4
- **Video tracking**: DAVIS-L 62.9 vs DINOv3 83.3（大差）
- **Instance retrieval**: Oxford-H 32.7 vs DINOv3 60.7

DINOv3 著者が指摘する通り「**キャプションでは捉えきれないピクセルレベル情報は SigLIP でも掴めない**」という弱教師あり全般の限界。

## SigLIP vs CLIP vs Perception Encoder vs DINOv3

| 軸 | CLIP / OpenCLIP | **SigLIP / SigLIP 2** | PE | DINOv3 |
|---|---|---|---|---|
| 教師信号 | テキスト（softmax） | テキスト（**sigmoid**） | テキスト（PE 独自） | 画像のみ |
| バッチ要件 | 大（32K+）必須 | **小〜中で OK** | 大 | 中〜大 |
| 効率 | 中 | **高** | 中 | 中 |
| データ規模 | 4 億（WIT） | WebLI 大規模 | **5.4B pairs / 86B samples seen** | 1.7B（純画像） |
| ゼロショット分類 | ◎ | ◎ | ◎ | △（dino.txt 経由）|
| 画像-テキスト検索 | ◎ | ◎ | ◎ | △ |
| Dense 予測 | △ | △ | △〜○ | **◎** |
| 細粒度分類 | ○ | ○ | ○ | **◎** |
| Instance 検索 | △ | △ | △ | **◎** |
| 3D 認識 | △ | △ | △ | **◎** |
| アクセス性 | 公開（OpenCLIP） | **完全公開** | 公開 | 公開 |

各々得意分野が異なる。実用では **用途で使い分け** または併用（CLIP/SigLIP + DINOv2 等）。

## 派生・応用

### 直接派生

- **SigLIP 2**（Google, 2025, [[sources/siglip-2]]）: LocCa decoder + SILC/TIPS 自己蒸留 + ACID 蒸留 + 多言語化 + de-bias + NaFlex 統合
- **SigLIP 2 NaFlex**: 可変解像度・アスペクト比対応バリアント、OCR/文書理解に強い
- **mSigLIP**: 100 言語版（SigLIP v1 ベース）

### SigLIP を使う応用モデル

- **PaliGemma / PaliGemma 2**（Google, 2024）: SigLIP vision tower + Gemma LLM、オープン VLM。SigLIP 2 論文は PaliGemma 2 レシピを評価に使用
- **[[entities/gemma-3|Gemma 3]]**（Google DeepMind, 2025 Mar, [[sources/gemma-3]]）: **SigLIP 400M variant** を 4B/12B/27B モデルで **共有 + 訓練中凍結**、896² 固定 + 4×4 average pooling で **256 トークン圧縮**、推論時の **Pan & Scan** で非正方形・高解像度画像対応。**Gemma 3 27B が PaliGemma 2 27B を文書理解で凌駕**（DocVQA +4.4 / InfoVQA +14.4 / TextVQA +8.1 / ChartQA +12.1）、しかも 4B/12B は 10× 安価に転送。**LMSys Chatbot Arena Elo 1338** で DeepSeek-V3 / Llama-3.1-405B / Qwen2.5-72B を凌駕する Google 系オープン MLLM の代表
- **PaLI-X / PaLI-3**: Google の独自マルチモーダルモデル
- **LLaVA-OneVision**: SigLIP backbone 採用
- **MiniCPM-V**: SigLIP 系を vision tower に
- **Idefics 系**: HuggingFace のオープン VLM

### コミュニティ採用

- HuggingFace `transformers` ライブラリで第一級サポート
- `big_vision` コードベース（Google 公式）が活発
- WebLI 自体は非公開だが、モデルチェックポイントは Apache 2.0 で公開

## なぜ重要か

1. **CLIP 以来最大のレシピ革新**: sigmoid 損失で対比学習を根本的に効率化
2. **研究の民主化**: 4 TPU でも実用的に訓練可能、大規模インフラ不要
3. **bias term, β₂=0.95, chunked impl** など実装テクニックを確立
4. **「32k で飽和」発見**: 「対比学習 = 大バッチ」常識を覆す
5. **ノイズ頑健性**: WSL の標準的価値に追加
6. **多言語化 (mSigLIP)**: 非英語圏での実用化
7. **公開モデルとコードの貢献**: `big_vision` + Apache 2.0 モデルでコミュニティ巨大利益
8. **後続の WSL モデルの基礎**: PE（[[entities/perception-encoder]]）, EVA-CLIP, MetaCLIP, DFN 等の起点

## 限界

1. **WebLI 自体は非公開**: モデルは公開だが訓練データは Google 内部のみ
2. **Dense prediction が依然弱い**: DINOv3 系には密予測で大差負け（SigLIP 2 で改善するが完全ではない）
3. **テキスト encoder は英語中心**（mSigLIP/SigLIP 2 で改善）
4. **モデルアーキテクチャは標準 ViT のみ**: Hiera, ConvNet 系の探索なし
5. **画像-テキスト対以外の signal なし**: 3D, 動画, 音声マルチモーダル対応は別研究
6. **対比学習の本質的限界を継承**: 「キャプションが捉えない情報は学べない」（DINOv3 系の主張）

## 関連ページ

- [[sources/siglip]] — SigLIP v1 原典の要約
- [[translations/siglip]] — SigLIP v1 原典の全文和訳
- [[sources/siglip-2]] — SigLIP 2 原典の要約
- [[translations/siglip-2]] — SigLIP 2 原典の全文和訳（Appendix A-C 含む）
- [[sources/clip]] / [[entities/clip]] — 比較対象の元（softmax 対比学習）
- [[entities/perception-encoder]] / [[sources/perception-encoder]] — 同系統 Meta 競合（NeurIPS 2025）。5.4B unique pairs を 86B samples seen まで訓練 + alignment tuning で PEcore / PElang / PEspatial の 3 バリアント化。SigLIP 2 の「全部入りレシピ」とは対照的に **対比学習をピュアに保つ** 戦略
- [[entities/dinov3]] / [[sources/dinov3]] — 純粋 SSL 側の対抗馬、dense 予測で SigLIP 2 を上回る
- [[entities/dino]] / [[entities/ibot]] — SigLIP 2 が借用した自己蒸留・マスク予測の源流
- [[entities/sam-3]] — SigLIP を比較対象に持つ promptable concept seg
- [[concepts/contrastive-learning]] — sigmoid 損失も対比学習の一形態
- [[concepts/weakly-supervised-pretraining]] — SigLIP が属するパラダイム
- [[concepts/self-supervised-learning]] — SigLIP 2 が借用した自己蒸留・マスク予測
- [[concepts/masked-image-modeling]] — SigLIP 2 のマスク予測損失
- [[concepts/knowledge-distillation]] — SigLIP 2 の ACID/ACED
- [[concepts/zero-shot-transfer]] — SigLIP の主要評価軸
- [[concepts/foundation-model]] — WSL foundation model の代表
