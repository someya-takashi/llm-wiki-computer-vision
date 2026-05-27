# Index — Computer Vision Wiki

このページは wiki 全体のカタログです。ingest / query で新しいページを作成するたびに、対応するセクションに 1 行追記してください。

書式: `- [[<slug>]] — <一行の説明>`

---

## Overview

- [[overview]] — Computer Vision 分野全体の俯瞰（随時更新）

## Sources

### Papers

- [[sources/dino-emerging-properties-in-self-supervised-vit]] — DINO（Caron et al., 2021/ICCV）。ラベルなしの自己蒸留として ViT を学習。教師なしで自己注意マップにオブジェクト境界が現れることを発見。
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] — DINOv2（Oquab et al., 2023/TMLR 2024）。iBOT を 1B パラメータ × 142M キュレーション画像に scale。凍結特徴量のまま OpenCLIP に勝つ純粋 SSL 基盤モデル。
- [[sources/dinov3]] — DINOv3（Siméoni et al., 2025）。ViT-7B × LVD-1689M に scale、Gram anchoring で dense feature 劣化を解決。凍結バックボーンで PE / SigLIP 2 を dense タスクで圧倒し、SOTA 多数更新。
- [[sources/mae]] — MAE（He et al., 2021/CVPR 2022）。「画像の 75% をマスクして残り 25% から画素を再構成」というシンプルな SSL。非対称 encoder-decoder で ViT-H を IN1K のみで 87.8% 達成。CV MIM の中核手法。
- [[sources/ibot]] — iBOT（Zhou et al., 2021/ICLR 2022）。**online tokenizer** を提案し、DINO の自己蒸留と MIM を統合。DINOv2/DINOv3 の損失関数の直接の源。
- [[sources/clip]] — CLIP（Radford et al., 2021/ICML 2021）。4 億画像-テキスト対 + 対比学習で「a photo of a {class}」プロンプトによるゼロショット分類を実現。現代マルチモーダル AI の起点。
- [[sources/segment-anything]] — SAM（Kirillov et al., 2023/ICCV 2023）。promptable segmentation という新タスクを定義し、データエンジンで 1.1B マスクを構築。CV における初の本格的セグメンテーション基盤モデル。
- [[sources/sam-2]] — SAM 2（Ravi et al., 2024/ICLR 2025）。SAM を動画に拡張、streaming memory + Hiera 画像エンコーダ。SA-V（35.5M マスク / 642.6K masklet / 50.9K 動画、CC by 4.0）を構築。画像でも SAM v1 比 6× 高速。
- [[sources/sam-3]] — SAM 3（Carion et al., 2025）。新タスク PCS（Promptable Concept Segmentation）を導入、名詞句/画像 exemplar で全インスタンスを検出・セグメント・追跡。Perception Encoder backbone + DETR ベース detector + presence head + SAM 2 風 tracker。SA-Co（4M unique NP / 52M マスク、benchmark 207K concepts）を構築。
- [[sources/siglip]] — SigLIP（Zhai et al., 2023/ICCV 2023）。CLIP の softmax 対比損失を sigmoid 損失に置き換え、小バッチで圧倒的に勝つ + メモリ効率改善。4 TPU で 1 日訓練して 79.7% IN-0、SO/400M で 83.2%。バッチサイズが 32k で飽和することを発見。bias term + β₂=0.95 で大バッチ安定化。
- [[sources/siglip-2]] — SigLIP 2（Tschannen et al., 2025）。SigLIP に **LocCa decoder + SILC/TIPS 自己蒸留＋マスク予測 + ACID 蒸留 + 多言語＋de-bias + NaFlex** を統合した「全部入りレシピ」。RefCOCO で +20pt、ADE20k seg で +4.2pt、representation bias を 35.5%→7.3% に削減。g/16 (1B) 新サイズで 85.0% IN-0。WebLI 10B 画像 / 109 言語で訓練。
- [[sources/simclr]] — SimCLR（Chen et al., ICML 2020）。4 コンポーネント（データ拡張 + エンコーダ + 非線形射影ヘッド + NT-Xent 損失）を体系的なアブレーションで解明。ResNet-50 (4×) で 76.5% top-1（教師あり ResNet-50 と同等）、1% ラベルで top-5 85.8%。対比 SSL の 2020 年標準ベースライン。
- [[sources/byol]] — BYOL（Grill et al., NeurIPS 2020）。オンライン×ターゲット 2 ネットワーク + predictor で負例なしに 74.3% top-1 達成（SimCLR+5pt）。「負例は必須か？」に「否」と答えた非対比 SSL の先駆け。
- [[sources/mixmatch]] — MixMatch（Berthelot et al., NeurIPS 2019）。一貫性正則化 + エントロピー最小化 + MixUp を統合した半教師あり学習。CIFAR-10（250 ラベル）で 11.08%（VAT の 3.3 倍改善）。
- [[sources/fixmatch]] — FixMatch（Sohn et al., NeurIPS 2020）。弱→強の非対称拡張 + 信頼度閾値 τ=0.95 だけで MixMatch を大幅更新。CIFAR-10（250 ラベル）で 5.07%。半教師あり学習の設計原則を確立。
- [[sources/flexmatch]] — FlexMatch（Zhang et al., NeurIPS 2021）。CPL（クラス別動的閾値）で FixMatch を拡張。CIFAR-10（40 ラベル）で 4.97%、収束速度 1/5。ただし SVHN（クラス不均衡）では逆に悪化。

### Articles

（まだありません）

## Translations

- [[translations/dino-emerging-properties-in-self-supervised-vit]] — DINO 原論文（§1-6 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov2-learning-robust-visual-features-without-supervision]] — DINOv2 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov3]] — DINOv3 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/mae]] — MAE 原論文の全文和訳。**Appendix A〜C を含む**（References のみ除外）。
- [[translations/ibot]] — iBOT 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/clip]] — CLIP 原論文の全文和訳。**Appendix A〜F を含む**（References のみ除外）。
- [[translations/segment-anything]] — SAM 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/sam-2]] — SAM 2 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/sam-3]] — SAM 3 原論文の全文和訳。Abstract + §1-9 + Appendix A.1 部分のみ（原典ファイルに残部の Appendix が含まれないため、ユーザー判断で本文のみ ingest）。
- [[translations/siglip]] — SigLIP 原論文の全文和訳。本文 §1-5 + Acknowledgements（独立 Appendix なし、PDF は本文 10 ページ）。
- [[translations/siglip-2]] — SigLIP 2 原論文の全文和訳。**Appendix A-C（PaliGemma 全結果 / NaFlex 全結果 / 文化的多様性・公平性全結果）を含む**（References のみ除外）。
- [[translations/simclr]] — SimCLR 原論文の全文和訳。Abstract + §1-8 + Appendix A-C（拡張詳細 / 追加実験 / 関連手法詳細比較）を含む（References のみ除外）。
- [[translations/byol]] — BYOL 原論文の全文和訳。Abstract + §1-6 + Appendix A-B（アルゴリズム / 画像拡張詳細）を含む（References のみ除外）。
- [[translations/mixmatch]] — MixMatch 原論文の全文和訳。Abstract + §1-5 + Appendix A-C（記号定義 / 全数値表 / 13 層 ConvNet 結果）を含む（References のみ除外）。
- [[translations/fixmatch]] — FixMatch 原論文の全文和訳。Abstract + §1-6 + Broader Impact + Appendix A-E（アルゴリズム擬似コード / 全数値表・ハイパーパラメータ / ImageNet 実装 / 拡張手法 / データ変換詳細）を含む（References のみ除外）。
- [[translations/flexmatch]] — FlexMatch 原論文の全文和訳。Abstract + §1-6 + Broader Impact + Appendix A-B（ハイパーパラメータ / クラスごと精度 / 中央値エラー率 / TorchSSL ベンチマーク全 4 データセット）を含む（References のみ除外）。

## Concepts

- [[concepts/self-supervised-learning]] — SSL（自己教師あり学習）の全体像。対比型・非対比型・クラスタリング型・マスク再構成型・ハイブリッド型、momentum encoder と multi-crop の解説を含む。
- [[concepts/vision-transformer]] — ViT のアーキテクチャ。パッチ分割、[CLS] トークン、命名規約（ViT-S/B/L、/16・/8）、CNN との比較。
- [[concepts/knowledge-distillation]] — 知識蒸留の基礎（dark knowledge, soft target, 温度付き softmax）と DINO 流 self-distillation との対比。
- [[concepts/knn-evaluation-protocol]] — SSL の表現品質を測る k-NN 評価。なぜ DINO で特に注目されたか。
- [[concepts/masked-image-modeling]] — MIM（BERT 流マスク予測の画像版）。BEiT, MAE, iBOT, SimMIM の系譜。
- [[concepts/denoising-autoencoder]] — DAE。MAE / BERT MLM / 拡散モデルなど現代 SSL の理論的祖先。「破損 → 再構成」枠組み。
- [[concepts/foundation-model]] — 基盤モデルの定義と歴史。NLP（BERT/GPT）から CV（CLIP, DINOv2, DINOv3）への系譜、必要要件と批判。
- [[concepts/weakly-supervised-pretraining]] — 弱教師あり事前学習（CLIP, ALIGN, OpenCLIP, SigLIP, PE）。SSL との対比、長所と弱点。
- [[concepts/gram-anchoring]] — DINOv3 が提案した、パッチ間 Gram 行列を過去の teacher に合わせる正則化。dense feature 劣化問題の解決法。
- [[concepts/rotary-position-embeddings]] — RoPE。NLP で標準の位置埋め込みを vision に持ち込み。可変解像度対応に必須。
- [[concepts/online-tokenizer]] — iBOT が提案した、teacher 自身を MIM の視覚トークナイザとして動的に使う発想。DINOv2/v3 の損失設計の中核。
- [[concepts/contrastive-learning]] — 対比学習。正例ペアを近づけ負例を遠ざける。InfoNCE 損失、SimCLR/MoCo/CLIP/DINO 系統の基盤。
- [[concepts/semi-supervised-learning]] — 半教師あり学習（SeSL）。少量ラベルあき + 大量ラベルなしを組み合わせる。一貫性正則化 / エントロピー最小化 / MixUp の 3 系統。自己教師あり学習（SSL）との違いを解説。
- [[concepts/zero-shot-transfer]] — ゼロショット転移。CLIP が CV に持ち込んだ「プロンプトのみで追加訓練なしの推論」というパラダイム。
- [[concepts/promptable-segmentation]] — SAM が定義した「任意プロンプトから妥当マスクを返す」新タスク（PVS）。セグメンテーション基盤モデルの事前学習目的兼ゼロショット転移インターフェイス。
- [[concepts/promptable-concept-segmentation]] — SAM 3 が定義した「名詞句/画像 exemplar からコンセプトの全インスタンスを返す」新タスク（PCS）。PVS と互補的な open-vocabulary な軸。

## Entities

### Models

- [[entities/dino]] — DINO 手法のスペック・主要結果・系譜（DINOv2/DINOv3 等への発展）
- [[entities/dinov2]] — DINOv2 モデル群（ViT-S/B/L/g, LVD-142M で学習, ViT-g からの蒸留）
- [[entities/dinov3]] — DINOv3 モデル群（ViT-S〜7B + ConvNeXt + dino.txt, LVD-1689M で学習, Gram anchoring, RoPE）
- [[entities/ibot]] — iBOT（Zhou et al., 2021/ICLR 2022）。DINO + MIM の online tokenizer 統合。DINOv2/DINOv3 の損失関数の直接の源。
- [[entities/mae]] — MAE（He et al., 2021）。マスクオートエンコーダ。CV MIM の中核手法、iBOT/DINOv2/DINOv3 の MIM 損失の源流。
- [[entities/clip]] — CLIP（OpenAI, 2021）と OpenCLIP（LAION）。弱教師あり代表、DINOv2 の主要ベースライン。
- [[entities/perception-encoder]] — Meta の Perception Encoder（PEcore / PEspatial）。DINOv3 の主要競合（弱教師あり）。
- [[entities/siglip]] — Google の SigLIP / SigLIP 2。CLIP の sigmoid 損失改良（v1）＋ 全部入りレシピ統合（v2: LocCa decoder + 自己蒸留＋マスク予測 + ACID + 多言語 + NaFlex）。DINOv3 の主要競合（弱教師あり）。
- [[entities/sam]] — SAM（Meta, 2023）。CV における初の本格的セグメンテーション基盤モデル。ViT-B/L/H 3 種、Apache 2.0、データエンジンで 1.1B マスクを使い訓練。
- [[entities/sam-2]] — SAM 2（Meta, 2024）。SAM の動画拡張版。Hiera 画像エンコーダ + streaming memory、Hiera-T/S/B+/L 4 種、Apache 2.0。画像でも SAM v1 比 6× 高速。
- [[entities/sam-3]] — SAM 3（Meta Superintelligence Labs, 2025）。PCS タスクを導入、PE backbone + DETR detector + presence head + SAM 2 tracker。Apache 2.0 想定。SA-Co/Gold で人間性能の 74% 達成。
- [[entities/hiera]] — Hiera（Meta, 2023）。MAE 互換の階層型 ViT、shifted window や RPB を削除したシンプル設計。SAM 2 の画像エンコーダ。
- [[entities/simclr]] — SimCLR（Google Brain, 2020）。対比 SSL の標準フレームワーク。4 コンポーネント設計と体系的アブレーションで 2020 年 SOTA を確立。
- [[entities/byol]] — BYOL（DeepMind, 2020）。オンライン×ターゲット 2 ネットワーク構造 + predictor で負例なし SSL。ResNet-200(2×) で 79.6% top-1、SimSiam/DINO 系の源流。
- [[entities/mixmatch]] — MixMatch（Google Research, NeurIPS 2019）。半教師あり学習。一貫性正則化 + シャープニング + ラベルあき・なし横断 MixUp。CIFAR-10 (250 ラベル) で 11.08%（VAT 比 3.3 倍改善）。
- [[entities/fixmatch]] — FixMatch（Google Research, NeurIPS 2020）。半教師あり学習。弱→強の非対称拡張 + τ=0.95 閾値。CIFAR-10（250 ラベル）で 5.07%（MixMatch 比 2.2× 改善）。
- [[entities/flexmatch]] — FlexMatch（東工大・Microsoft, NeurIPS 2021）。半教師あり学習。CPL クラス別動的閾値。CIFAR-10（40 ラベル）で 4.97%、収束速度 FixMatch の 1/5。

### Datasets

- [[entities/imagenet]] — ImageNet（特に ImageNet-1k と ImageNet-22k）。SSL の事前学習・線形評価・k-NN 評価のデファクト基盤
- [[entities/lvd-142m]] — DINOv2 用に自動キュレーションされた 142M 画像コレクション（非公開）
- [[entities/lvd-1689m]] — DINOv3 用に階層 k-means でキュレーションされた 1.689B 画像（Instagram 由来、非公開）
- [[entities/sat-493m]] — DINOv3 衛星版用の Maxar 衛星画像 493M（0.6m/pixel, 512², 非公開）
- [[entities/wit-400m]] — CLIP の訓練データ。OpenAI が web から構築した 4 億画像-テキスト対（非公開）
- [[entities/sa-1b]] — SAM の訓練データ。データエンジンで構築された 11M 画像 × 1.1B マスク（既存最大の 400 倍）。研究用途で公開。
- [[entities/sa-v]] — SAM 2 の訓練データ。50.9K 動画 × 642.6K masklet × 35.5M マスク（既存 VOS の 53 倍）。**CC by 4.0** で公開。
- [[entities/sa-co]] — SAM 3 の訓練・評価データ。SA-Co/HQ（5.2M 画像 + 4M unique NP + 52M マスク）+ SA-Co Benchmark（**207K concepts**、既存の 50 倍以上）+ SA-Co/SYN（合成 38M 句）+ SA-Co/VIDEO（52.5K 動画）。

### People

（まだありません）

### Organizations / Benchmarks

（まだありません）

## Questions

（まだありません）

---

## 略称対応表（既出のもののみ追加）

| 略称 | 展開 | 主たる解説ページ |
|---|---|---|
| ViT | Vision Transformer | [[concepts/vision-transformer]] |
| SSL | Self-Supervised Learning | [[concepts/self-supervised-learning]] |
| WSL | Weakly-Supervised Learning | [[concepts/weakly-supervised-pretraining]] |
| KD | Knowledge Distillation | [[concepts/knowledge-distillation]] |
| MIM | Masked Image Modeling | [[concepts/masked-image-modeling]] |
| MAE | Masked Autoencoder | [[entities/mae]] |
| DAE | Denoising Autoencoder | [[concepts/denoising-autoencoder]] |
| MLM | Masked Language Modeling | [[concepts/denoising-autoencoder]] |
| online tokenizer | オンライントークナイザ | [[concepts/online-tokenizer]] |
| dVAE | discrete VAE | [[concepts/online-tokenizer]]（BEiT の文脈で言及） |
| InfoNCE | Information Noise Contrastive Estimation | [[concepts/contrastive-learning]] |
| WIT | WebImageText (400M) | [[entities/wit-400m]] |
| BPE | Byte Pair Encoding | [[entities/clip]]（テキストトークナイザ） |
| zero-shot | ゼロショット（追加訓練なしの推論） | [[concepts/zero-shot-transfer]] |
| prompt engineering | プロンプトエンジニアリング | [[concepts/zero-shot-transfer]] |
| BEiT | BERT pre-training of image transformers | [[concepts/masked-image-modeling]] |
| DINO | self-DIstillation with NO labels | [[entities/dino]] |
| DINOv2 | DINO version 2 | [[entities/dinov2]] |
| DINOv3 | DINO version 3 | [[entities/dinov3]] |
| iBOT | image BERT pre-Training with Online Tokenizer | [[entities/ibot]] |
| CLIP | Contrastive Language-Image Pre-training | [[entities/clip]] |
| OpenCLIP | open-source CLIP reproduction | [[entities/clip]] |
| PE / PEcore / PEspatial | Perception Encoder | [[entities/perception-encoder]] |
| SigLIP / SigLIP 2 | Sigmoid Loss Image-Language Pre-training | [[entities/siglip]] |
| AM-RADIO | Agglomerative Models — RADIO | [[entities/dinov3]]（短い言及） |
| EVA-CLIP | enhanced CLIP scaling | [[entities/dinov3]]（短い言及） |
| Franca | open-data SSL model | [[entities/dinov3]]（短い言及） |
| Web-DINO | web-scale DINO | [[entities/dinov3]]（短い言及） |
| V-JEPA / V-JEPA 2 | Video JEPA | [[entities/dinov3]]（短い言及） |
| JEPA | Joint-Embedding Predictive Architecture | [[entities/dinov3]]（短い言及） |
| SAM | Segment Anything Model | [[entities/sam]] |
| SAM 2 | Segment Anything Model 2（動画版） | [[entities/sam-2]] |
| SAM 3 | Segment Anything Model 3（コンセプト版） | [[entities/sam-3]] |
| SA-1B | Segment Anything 1 Billion (dataset) | [[entities/sa-1b]] |
| SA-V | Segment Anything Video (dataset) | [[entities/sa-v]] |
| SA-Co | Segment Anything with Concepts (dataset/benchmark) | [[entities/sa-co]] |
| Hiera | Hierarchical Vision Transformer | [[entities/hiera]] |
| PVS | Promptable Visual Segmentation（SAM 1/2 のタスク） | [[concepts/promptable-segmentation]] |
| PCS | Promptable Concept Segmentation（SAM 3 のタスク） | [[concepts/promptable-concept-segmentation]] |
| promptable segmentation | プロンプト可能セグメンテーション | [[concepts/promptable-segmentation]] |
| masklet | 動画全体の時空間マスク | [[entities/sa-v]] / [[entities/sam-2]] |
| NP | Noun Phrase（短い名詞句、SAM 3 のテキストプロンプト形式） | [[concepts/promptable-concept-segmentation]] |
| hard negative | 敵対的に困難な negative ラベル | [[entities/sa-co]] |
| exemplar | 画像 exemplar（SAM 3 の画像プロンプト） | [[entities/sam-3]] |
| presence token / head | SAM 3 の認識と位置特定を分離するトークン | [[entities/sam-3]] |
| AI verifier | Llama 3.2 でファインチューンされた自動検証者 | [[entities/sa-co]] |
| MV / EV | Mask Verification / Exhaustivity Verification | [[entities/sa-co]] |
| cgF₁ | classification-gated F1（SAM 3 主要メトリクス） | [[sources/sam-3]] |
| pmF₁ | positive media-phrase F1 | [[sources/sam-3]] |
| IL_MCC | Image-Level Matthews Correlation Coefficient | [[sources/sam-3]] |
| pHOTA | positive HOTA（動画 PCS の主要メトリクス） | [[sources/sam-3]] |
| HOTA | Higher Order Tracking Accuracy | [[sources/sam-3]] |
| DETR | DEtection TRansformer | [[entities/sam-3]] |
| MDETR | Modulated DETR（テキスト条件付き DETR） | [[entities/sam-3]] |
| MaskFormer | マスクヘッドアーキテクチャ | [[entities/sam-3]] |
| SAM 3 Agent | SAM 3 を MLLM ツールとして使うパターン | [[entities/sam-3]] |
| SigLIP | Sigmoid Loss for Language-Image Pre-training | [[entities/siglip]] |
| SigLiT | Sigmoid LiT（frozen 画像 backbone 版 SigLIP） | [[entities/siglip]] |
| mSigLIP | multilingual SigLIP（100 言語版） | [[entities/siglip]] |
| sigmoid loss | ペア独立の二値分類損失（SigLIP の核心） | [[sources/siglip]] / [[concepts/contrastive-learning]] |
| WebLI | Google 内部の Web 由来画像-テキストデータ | [[entities/siglip]]（PaLI 由来） |
| SO-400M | Shape-Optimized 400M ViT | [[entities/siglip]] |
| XM3600 | Crossmodal-3600（36 言語の多言語検索ベンチマーク） | [[entities/siglip]] |
| LiT | Locked-image Tuning（frozen backbone + テキスト訓練） | [[entities/siglip]]（SigLiT の元） |
| InfoNCE | Info Noise Contrastive Estimation（softmax 対比損失） | [[concepts/contrastive-learning]] |
| big_vision | Google の公開コードベース | [[entities/siglip]] |
| β₂ | Adam/AdaFactor のモメンタム係数（SigLIP では 0.95） | [[entities/siglip]] |
| chunked implementation | デバイス間で per-device $b^2$ メモリの効率実装 | [[entities/siglip]] |
| SigLIP 2 | SigLIP version 2（全部入り統合レシピ） | [[entities/siglip]] / [[sources/siglip-2]] |
| NaFlex | Native aspect ratio + Flexible sequence length バリアント | [[entities/siglip]] / [[sources/siglip-2]] |
| NaViT | Native-resolution ViT（[Dehghani et al. 2024]） | [[sources/siglip-2]]（言及） |
| FlexiViT | Flexible patch-size ViT（[Beyer et al. 2023]） | [[sources/siglip-2]]（言及） |
| LocCa | Location-aware Captioner、decoder ベース事前学習レシピ | [[sources/siglip-2]] / [[entities/siglip]] |
| SILC | Self-supervised + Image-Language Contrastive learning | [[sources/siglip-2]] |
| TIPS | Text-Image Pretraining with Spatial awareness | [[sources/siglip-2]] |
| ACID | Active Curation Implicit Distillation | [[sources/siglip-2]] / [[concepts/knowledge-distillation]] |
| ACED | ACID + Explicit Distillation | [[sources/siglip-2]] / [[concepts/knowledge-distillation]] |
| MAP head | Multi-head Attention Pooling | [[entities/siglip]] |
| Gemma / Gemma 2 | Google の LLM ファミリー、SigLIP 2 のトークナイザに採用 | [[entities/siglip]] |
| PaliGemma / PaliGemma 2 | SigLIP + Gemma の VLM | [[entities/siglip]] |
| representation bias | ランダムオブジェクトと性別の偏った関連付け度 | [[sources/siglip-2]] |
| GeoDE | Geographically Diverse Evaluation データセット | [[sources/siglip-2]] |
| Dollar Street | 所得別多様性評価データセット | [[sources/siglip-2]] |
| RefCOCO / RefCOCO+ / RefCOCOg | 参照表現理解ベンチマーク | [[sources/siglip-2]] |
| DPT | Dense Prediction Transformer | [[sources/siglip-2]] |
| OWL-ViT | Open-vocab detection 手法 | [[sources/siglip-2]] |
| Cat-Seg | Open-vocab segmentation 手法 | [[sources/siglip-2]] |
| HierText / SciCap / Screen2Words / TextCaps | OCR/文書/スクリーン系ベンチマーク | [[sources/siglip-2]] |
| TPUv5e | Google 第 5 世代 TPU | [[sources/siglip-2]]（最大 2048 チップ） |
| VOS | Video Object Segmentation | [[entities/sam-2]] |
| iVOS | interactive VOS | [[entities/sam-2]] |
| 𝒥&ℱ | VOS 標準精度指標（領域類似度 𝒥 + 境界精度 ℱ） | [[sources/sam-2]] |
| 2D-RoPE | 2 次元 Rotary Positional Embedding | [[concepts/rotary-position-embeddings]] |
| FlashAttention | 効率的 attention カーネル | [[entities/sam-2]] |
| MOSE | 困難 VOS ベンチマーク | [[entities/sam-2]] |
| LVOS / LVOSv2 | 長期 VOS ベンチマーク | [[entities/sam-2]] |
| DAVIS | 古典的 VOS ベンチマーク | [[entities/sam-2]] |
| YouTube-VOS / YTVOS | 大規模 VOS ベンチマーク | [[entities/sam-2]] |
| XMem / XMem++ | 強力な対話的 VOS ベースライン | [[entities/sam-2]] |
| Cutie | 強力な対話的 VOS ベースライン | [[entities/sam-2]] |
| FPN | Feature Pyramid Network | [[entities/hiera]] |
| GRU | Gated Recurrent Unit（SAM 2 で試したが不採用） | [[entities/sam-2]] |
| RITM | Reviving Iterative Training with Mask guidance | [[sources/segment-anything]]（対話的セグメンテーション主要ベースライン） |
| ViTDet | Vision Transformer Detector | [[sources/segment-anything]]（インスタンスセグメンテーションのボックス供給元） |
| LVIS | Large Vocabulary Instance Segmentation | [[sources/segment-anything]] |
| COCO | Common Objects in Context | [[sources/segment-anything]] |
| NMS | Non-Maximum Suppression | [[sources/segment-anything]] |
| IoU / mIoU | Intersection over Union / mean IoU | [[sources/segment-anything]] |
| focal loss | 不均衡分類用損失 | [[sources/segment-anything]] |
| dice loss | セグメンテーション用領域重なり損失 | [[sources/segment-anything]] |
| MIAP | More Inclusive Annotations for People | [[sources/segment-anything]]（公平性評価データセット） |
| RAI | Responsible AI | [[sources/segment-anything]] |
| stuff / things | 不可算/可算オブジェクト区別 | [[sources/segment-anything]] |
| modal / amodal mask | 可視部分のみ/遮蔽部含むマスク | [[sources/segment-anything]] |
| BSDS500 | Berkeley Segmentation Dataset 500 | [[sources/segment-anything]]（エッジ検出ベンチマーク） |
| ODS / OIS / AP / R50 | エッジ検出標準指標 | [[sources/segment-anything]] |
| VGGT | Visual Geometry Grounded Transformer | [[entities/dinov3]] |
| DAv2 | Depth Anything V2 | [[entities/dinov3]] |
| SimCLR | A Simple Framework for Contrastive Learning of Visual Representations | [[entities/simclr]] |
| NT-Xent | Normalized Temperature-scaled Cross Entropy（SimCLR の損失関数, InfoNCE の実装） | [[sources/simclr]] |
| LARS | Layer-wise Adaptive Rate Scaling（大バッチ学習用オプティマイザ） | [[sources/simclr]] |
| MoCo | Momentum Contrast（momentum encoder + memory bank 版対比 SSL） | [[concepts/contrastive-learning]] |
| PIRL | Pretext-Invariant Representations Learning | [[concepts/contrastive-learning]] |
| Global BN | Global Batch Normalization（分散訓練での情報リーク防止） | [[sources/simclr]] |
| BYOL | Bootstrap Your Own Latent（負例なし SSL の先駆け） | [[entities/byol]] |
| predictor | BYOL のオンライン側追加 MLP（崩壊防止の鍵） | [[sources/byol]] |
| MixMatch | 半教師あり学習アルゴリズム（一貫性正則化 + シャープニング + MixUp 統合） | [[entities/mixmatch]] |
| MixUp | 2 例を凸補間する正則化手法（Zhang et al., 2017） | [[sources/mixmatch]] |
| Label Guessing | K 回拡張の予測平均で疑似ラベルを生成 | [[sources/mixmatch]] |
| Sharpening（半教師あり）| 温度で確率分布を先鋭化してエントロピーを下げる操作 | [[sources/mixmatch]] |
| SeSL / semi-SSL | Semi-Supervised Learning（半教師あり学習） | [[concepts/semi-supervised-learning]] |
| Mean Teacher | EMA ターゲット教師による一貫性正則化 | [[concepts/semi-supervised-learning]] |
| VAT | Virtual Adversarial Training（仮想敵対的訓練） | [[concepts/semi-supervised-learning]] |
| FixMatch | 高信頼閾値 τ=0.95 + 弱→強拡張での半教師あり学習（MixMatch 後継） | [[entities/fixmatch]] |
| RandAugment | Randomly-selected Augmentation（事前探索なしに M 種の拡張を適用） | [[sources/fixmatch]] |
| CTAugment | Control Theory Augment（ラベルあきデータで拡張強度をオンライン調整） | [[sources/fixmatch]] |
| Cutout | 画像の矩形領域をゼロでマスクする正則化 | [[sources/fixmatch]] |
| confirmation bias / 確証バイアス | 誤疑似ラベルが自己強化されるループ | [[sources/fixmatch]] |
| DA / Distribution Alignment | モデルの周辺予測分布をラベル分布に合わせるバイアス補正 | [[sources/fixmatch]] |
| barely supervised | 1 クラスあたり数枚程度という極端な少ラベル設定 | [[sources/fixmatch]] |
| τ (tau) | 信頼度閾値。FixMatch ではデフォルト 0.95（FlexMatch ではクラス別動的閾値の上限） | [[entities/fixmatch]] |
| FlexMatch | FixMatch にクラス別動的閾値（CPL）を導入した後継手法 | [[entities/flexmatch]] |
| CPL | Curriculum Pseudo Labeling（カリキュラム疑似ラベリング） | [[sources/flexmatch]] |
| σ_t(c) / β_t(c) | FlexMatch の推定学習効果 / 正規化学習効果（クラス別） | [[sources/flexmatch]] |
| カリキュラム学習 | 簡単→難しいサンプルの段階的導入戦略（Bengio 2009） | [[sources/flexmatch]] |
| TorchSSL | FlexMatch と同時公開された PyTorch ベース SSL 統合コードベース | [[entities/flexmatch]] |
| FreeMatch | 2 レベル自由閾値による FlexMatch の後継（Wang et al., 2023） | [[entities/flexmatch]] |
| SoftMatch | ガウス重み付きソフト閾値（FlexMatch の後継, Chen et al., 2023） | [[entities/flexmatch]] |
| Brier スコア | 多クラス予測確率と正解の L2 二乗距離（有界な損失） | [[sources/mixmatch]] |
| PATE | Private Aggregation of Teachers' Ensembles（差分プライバシー学習） | [[sources/mixmatch]] |
| Wide ResNet / WRN | Wide Residual Network（幅広い ResNet） | [[sources/mixmatch]] |
| DPT | Dense Prediction Transformer | [[entities/dinov3]] |
| LVD-142M | Large-scale Visual Dataset (142M) | [[entities/lvd-142m]] |
| LVD-1689M | Large-scale Visual Dataset (1689M) | [[entities/lvd-1689m]] |
| SAT-493M | Satellite Dataset (493M) | [[entities/sat-493m]] |
| Gram Anchoring | パッチ間 Gram 行列正則化 | [[concepts/gram-anchoring]] |
| RoPE | Rotary Position Embeddings | [[concepts/rotary-position-embeddings]] |
| register tokens | レジスタトークン | [[entities/dinov3]] |
| EMA | Exponential Moving Average | [[concepts/self-supervised-learning]] |
| k-NN | k-Nearest Neighbors classifier | [[concepts/knn-evaluation-protocol]] |
| ILSVRC | ImageNet Large Scale Visual Recognition Challenge | [[entities/imagenet]] |
| SK | Sinkhorn-Knopp | [[entities/dinov2]] |
| KoLeo | Kozachenko-Leonenko regularizer | [[entities/dinov2]] |
| SwiGLU | Swish-Gated Linear Unit | [[entities/dinov2]] |
| FFN | Feed-Forward Network | [[entities/dinov2]] |
| FSDP | Fully-Sharded Data Parallel | [[entities/dinov2]] |
| DDP | Distributed Data Parallel | [[entities/dinov2]] |
| FlashAttention | 高効率 attention 実装 | [[entities/dinov3]] |
| PUE | Power Usage Effectiveness | [[entities/dinov2]] |
| GLDv2 | Google Landmarks Dataset v2 | [[sources/dino-emerging-properties-in-self-supervised-vit]] |
| TTA | Test-Time Augmentation | [[sources/dinov3]] |
| OOD | Out-Of-Distribution | [[sources/dinov3]] |
| CNX / ConvNeXt | Convolutional Next | [[entities/dinov3]] |
| WRI | World Resources Institute | [[sources/dinov3]] |
| AIMv2 | Autoregressive Image Models v2 | [[sources/dinov3]] |
| LiT | Locked-image Tuning | [[entities/dinov3]] |
| dino.txt | DINO + text alignment | [[entities/dinov3]] |
| CNN / convnet | Convolutional Neural Network | （未作成） |
| MLP | Multi-Layer Perceptron | （未作成） |
| BN | Batch Normalization | （未作成） |
| CE | Cross Entropy | （未作成） |
| BYOL | Bootstrap Your Own Latent | （未作成） |
| MoCo | Momentum Contrast | （未作成） |
| SwAV | Swapping Assignments between multiple Views | （未作成） |
| BERT | Bidirectional Encoder Representations from Transformers | （未作成） |
| VLM | Vision-Language Model | （未作成） |
