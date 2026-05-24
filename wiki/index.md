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

### Articles

（まだありません）

## Translations

- [[translations/dino-emerging-properties-in-self-supervised-vit]] — DINO 原論文（§1-6 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov2-learning-robust-visual-features-without-supervision]] — DINOv2 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov3]] — DINOv3 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。

## Concepts

- [[concepts/self-supervised-learning]] — SSL（自己教師あり学習）の全体像。対比型・非対比型・クラスタリング型・マスク再構成型・ハイブリッド型、momentum encoder と multi-crop の解説を含む。
- [[concepts/vision-transformer]] — ViT のアーキテクチャ。パッチ分割、[CLS] トークン、命名規約（ViT-S/B/L、/16・/8）、CNN との比較。
- [[concepts/knowledge-distillation]] — 知識蒸留の基礎（dark knowledge, soft target, 温度付き softmax）と DINO 流 self-distillation との対比。
- [[concepts/knn-evaluation-protocol]] — SSL の表現品質を測る k-NN 評価。なぜ DINO で特に注目されたか。
- [[concepts/masked-image-modeling]] — MIM（BERT 流マスク予測の画像版）。BEiT, MAE, iBOT, SimMIM の系譜。
- [[concepts/foundation-model]] — 基盤モデルの定義と歴史。NLP（BERT/GPT）から CV（CLIP, DINOv2, DINOv3）への系譜、必要要件と批判。
- [[concepts/weakly-supervised-pretraining]] — 弱教師あり事前学習（CLIP, ALIGN, OpenCLIP, SigLIP, PE）。SSL との対比、長所と弱点。
- [[concepts/gram-anchoring]] — DINOv3 が提案した、パッチ間 Gram 行列を過去の teacher に合わせる正則化。dense feature 劣化問題の解決法。
- [[concepts/rotary-position-embeddings]] — RoPE。NLP で標準の位置埋め込みを vision に持ち込み。可変解像度対応に必須。

## Entities

### Models

- [[entities/dino]] — DINO 手法のスペック・主要結果・系譜（DINOv2/DINOv3 等への発展）
- [[entities/dinov2]] — DINOv2 モデル群（ViT-S/B/L/g, LVD-142M で学習, ViT-g からの蒸留）
- [[entities/dinov3]] — DINOv3 モデル群（ViT-S〜7B + ConvNeXt + dino.txt, LVD-1689M で学習, Gram anchoring, RoPE）
- [[entities/ibot]] — iBOT（Zhou et al., 2022）。DINO + MIM。DINOv2/DINOv3 の祖先。
- [[entities/clip]] — CLIP（OpenAI, 2021）と OpenCLIP（LAION）。弱教師あり代表、DINOv2 の主要ベースライン。
- [[entities/perception-encoder]] — Meta の Perception Encoder（PEcore / PEspatial）。DINOv3 の主要競合（弱教師あり）。
- [[entities/siglip]] — Google の SigLIP / SigLIP 2。CLIP の sigmoid 損失改良版、DINOv3 の主要競合（弱教師あり）。

### Datasets

- [[entities/imagenet]] — ImageNet（特に ImageNet-1k と ImageNet-22k）。SSL の事前学習・線形評価・k-NN 評価のデファクト基盤
- [[entities/lvd-142m]] — DINOv2 用に自動キュレーションされた 142M 画像コレクション（非公開）
- [[entities/lvd-1689m]] — DINOv3 用に階層 k-means でキュレーションされた 1.689B 画像（Instagram 由来、非公開）
- [[entities/sat-493m]] — DINOv3 衛星版用の Maxar 衛星画像 493M（0.6m/pixel, 512², 非公開）

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
| MAE | Masked Autoencoder | [[concepts/masked-image-modeling]] |
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
| SAM / SAM v2 | Segment Anything Model | [[entities/perception-encoder]] |
| VGGT | Visual Geometry Grounded Transformer | [[entities/dinov3]] |
| DAv2 | Depth Anything V2 | [[entities/dinov3]] |
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
