---
type: entity
entity_kind: model
aliases: [SDXL, Stable Diffusion XL, SD-XL, SDXL base, SDXL refiner, SDXL 1.0]
related: [[entities/clip]], [[concepts/diffusion-model]], [[concepts/vision-transformer]], [[concepts/foundation-model]], [[concepts/contrastive-learning]]
sources: [[sources/sdxl]]
updated: 2026-05-31
---

# SDXL（Stable Diffusion XL）

**SDXL** は **Stability AI** の Applied Research チーム（Rombach lab 中心）が開発した **オープン text-to-image 潜在拡散モデル**（2023 年 7 月公開、arXiv:2307.01952）。**Stable Diffusion 系統の第 3 世代**（SD 1.x → SD 2.x → **SDXL**）として、**3 倍大きい UNet + 二重テキスト・エンコーダ + 3 つの新規条件付け + 改良 VAE + Refinement Model** の 5 つの主要改善で **商用 Midjourney v5.1 を 54.9% で凌駕**し、**オープンソース text-to-image の品質を商用フロンティアと互角に押し上げた決定版**。

## 基本情報

| 項目 | 値 |
| --- | --- |
| 発表 | 2023 年 7 月（arXiv:2307.01952） |
| 開発元 | **Stability AI**（Applied Research、Robin Rombach ら） |
| モデル系統 | **Stable Diffusion** 第 3 世代 |
| ライセンス | **CreativeML Open RAIL++-M License**（商用可、Apache 2.0 風） |
| リポジトリ | https://github.com/Stability-AI/generative-models |
| HF Hub | https://huggingface.co/stabilityai/ |
| パイプライン | 2 段階（**base + refiner**） |
| 出力解像度 | 1024×1024 を中心とした 40 アスペクト比 |
| 生成タイプ | text-to-image、image-to-image（refiner） |

## モデル構成

### Base Model（SDXL UNet）

| パラメータ | 値 | SD 1.x との比較 |
| --- | --- | --- |
| **UNet パラメータ** | **2.6B** | 860M（3.0× 増） |
| Transformer Blocks | **[0, 2, 10]** | [1, 1, 1, 1] |
| Channel Multipliers | **[1, 2, 4]** | [1, 2, 4, 4] |
| Lowest level | 削除（8× ダウンサンプリングなし） | あり |
| Context Dimension | **2048** | 768 |
| 訓練ステップ | 256² で 600k + 512² で 200k + multi-aspect | - |
| バッチサイズ | 2048（事前学習） | - |
| ノイズ・スケジュール | 1000 ステップ離散時間 | - |

### テキスト・エンコーダ（合計 817M）

| Encoder | Variant | Params | Output |
| --- | --- | --- | --- |
| **[[entities/clip\|CLIP]] ViT-L** | OpenAI | 60M | ペナルティメイト出力（チャンネル軸で連結） |
| **OpenCLIP ViT-bigG** | LAION | 1.4B (の一部) | ペナルティメイト出力 + **pooled 埋め込み** |
| **合計** | | **~817M** | context dim **2048**（連結） |

- **pooled OpenCLIP 埋め込み**は **timestep 埋め込みに追加**される（GLIDE 風）

### Autoencoder (VAE)

| 指標 | SDXL-VAE | SD 1.x VAE | SD 2.x VAE |
| --- | --- | --- | --- |
| PSNR ↑ | **24.7** | 23.4 | 24.5 |
| SSIM ↑ | **0.73** | 0.69 | 0.71 |
| LPIPS ↓ | **0.88** | 0.96 | 0.92 |
| rFID ↓ | **4.4** | 5.0 | 4.7 |

- **ゼロから訓練**、バッチサイズ 256（vs SD 1.x の 9）
- **EMA で重み追跡**
- すべての再構成指標で SD 1.x/2.x VAE を凌駕

### Refinement Model

- **同じ潜在空間**で動作する別個の LDM
- **高品質・高解像度データに特化**
- **最初の 200 ノイズ・スケール**に特化（高ノイズ領域）
- **SDEdit ベースの noising-denoising** で base モデル出力を精緻化
- 任意の段階（オフにしても動作するが品質低下）

## 3 つの新規条件付け技術

すべて **フーリエ特徴埋め込み**で表現し、**timestep 埋め込みに加算**される。

### 1. Size-Conditioning（サイズ条件付け）

- $\mathbf{c}_{\text{size}} = (h_{\text{original}}, w_{\text{original}})$
- 訓練データの **元の解像度**を条件として与える
- **39% の訓練データ廃棄問題を解決**（512² 未満を廃棄せずに済む）
- 推論時に **「見かけの解像度」を制御**可能
- アブレーション: ImageNet 512² で **FID 36.53**（vs nocond 39.76、512-only 43.84）

### 2. Crop-Conditioning（クロップ条件付け）

- $\mathbf{c}_{\text{crop}} = (c_{\text{top}}, c_{\text{left}})$
- 訓練中のランダム・クロッピング座標を条件として与える
- **物体切れ問題**を解決（SD 1.5/2.1 の典型的失敗）
- 推論時 $(0, 0)$ で **物体中心**サンプル取得
- データ・バケッティング（NovelAI）と異なり **クロッピングが生成に漏れない**

### 3. Multi-Aspect Training（マルチ・アスペクト訓練）

- $\mathbf{c}_{\text{ar}} = (h_{\text{tgt}}, w_{\text{tgt}})$
- **40 種類のアスペクト比**（0.25 から 4.0、付録 I）、ピクセル数 ~1024² 維持
- **同じバケット内画像でバッチ構成**、ステップごとに交替
- 固定アスペクト比事前学習後の **微調整段階**として適用

## 訓練レシピ

**多段階訓練**:
1. **256² 事前学習**: 600,000 ステップ、バッチサイズ 2048、size + crop conditioning
2. **512² 継続**: 200,000 ステップ
3. **マルチ・アスペクト微調整**: ~1024² 面積、**offset-noise レベル 0.05**

**訓練手法**:
- 離散時間拡散（1000 ステップ）
- Classifier-free guidance（10% で null embedding 置換）
- $\lambda_\sigma = \sigma^{-2}$ 重み付け
- Offset-noise（aesthetic 品質向上のため）

## 主要結果

### ユーザ研究（メイン、表 6 vs SD 系統）

| Model | Win Rate |
| --- | --- |
| **SDXL w/ refinement** | **48.44%** |
| **SDXL base** | **36.93%** |
| Stable Diffusion 1.5 | 7.91% |
| Stable Diffusion 2.1 | 6.71% |

- **SDXL + refiner は SD 2.1 比で 7.2× の選好**
- **SDXL base 単体でも SD 1.5/2.1 を 4-5× 凌駕**

### Midjourney v5.1 との比較

- **17,153 ユーザ選好比較**で **SDXL 54.9%、Midjourney V5.1 45.1%**
- PartiPrompts (P2) ベンチマーク、AWS GroundTruth で評価
- **6 カテゴリ中 4 で SDXL 凌駕**
- **10 チャレンジ中 7 で SDXL 凌駕または同等**

### Autoencoder（表 3）

SDXL-VAE が PSNR / SSIM / LPIPS / rFID の **4 指標すべてで SD 1.x/2.x VAE を凌駕**。

### FID/CLIP の信頼性問題（付録 F）

**SDXL の FID は SD 1.5/2.1 より悪い**にもかかわらず、人間評価者は明らかに SDXL を好む。**古典指標は基盤的 text-to-image モデルに不適合**という重要な発見。Pick-a-pic（Kirstain et al., 2023）の発見を裏付け。

## 限界

- **人間の手**: 複雑な解剖学（指、ポーズ）でしばしば失敗
- **完全な写真現実性**: 微妙な照明・テクスチャ変化に不完全
- **社会的・人種的バイアス**: 大規模訓練データの不可避的問題
- **Concept bleeding**: 属性バインディング失敗（CLIP の単一トークン圧縮特性に起因）
- **長く判読可能なテキスト**: ランダム文字や不整合
- **二段階パイプライン**: メモリ・推論コスト増加
- **推論速度**: 2.6B UNet + 2 テキスト・エンコーダ + refiner で遅い
- **離散時間拡散**: offset-noise 必須
- **古典 CLIP 系テキスト・エンコーダ**: T5 のような大型 LLM エンコーダに比べてプロンプト追従性で劣る可能性
- **FID/CLIP 指標の不適合**: 評価が難しい
- **訓練データ非公開**

## 系譜・後継

### Stable Diffusion 系統
- **Latent Diffusion Models (LDM)**（Rombach et al., 2022）: 潜在拡散の原論文
- **Stable Diffusion 1.x**（Stability AI, 2022 Aug、860M UNet, CLIP ViT-L, 512²）
- **Stable Diffusion 2.x**（Stability AI, 2022 Nov、865M UNet, OpenCLIP ViT-H, 768²）
- **SDXL**（Stability AI, 2023 Jul、2.6B UNet, dual encoders, 1024²）← 本ページ
- **SDXL Turbo**（2023 Nov）: 1 ステップ推論版（蒸留）
- **Stable Cascade**（2024 Feb）: Würstchen アーキテクチャ
- **Stable Diffusion 3 (SD3)**（2024 Feb）: **DiT (Diffusion Transformer) 系へ移行、T5-XXL テキスト・エンコーダ追加**
- **SD3.5**（2024）: SD3 改良版

### 同チーム派生
- **FLUX.1**（Black Forest Labs, 2024、Rombach ら元 Stability AI が独立）: 12B Hybrid MM-DiT、SDXL/SD3 の正統な後継的存在

### 商用競合（SDXL が比較した相手）
- DALL-E 2 / DALL-E 3（OpenAI）
- Midjourney v5.1 / v5.2 / v6 / v7
- DeepFloyd IF
- Bing Image Creator
- Imagen 1/2/3（Google）
- eDiff-I（NVIDIA）

## 関連ページ

- [[sources/sdxl]] — 原典の要約
- [[translations/sdxl]] — 原典の翻訳（本文 + 付録）
- [[concepts/diffusion-model]] — LDM パラダイムの理論的基盤
- [[entities/clip]] — SDXL の主要テキスト・エンコーダ（ViT-L）
- [[concepts/vision-transformer]] — テキスト・エンコーダの基盤
- [[concepts/foundation-model]] — 生成系基盤モデルの代表
- [[concepts/weakly-supervised-pretraining]] — 大規模 text-image 対による事前学習
- [[concepts/contrastive-learning]] — CLIP の基盤
