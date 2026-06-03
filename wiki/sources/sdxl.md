---
type: source
source_path: raw/papers/SDXL_ Improving Latent Diffusion Models for High-Resolution Image Synthesis.md
source_kind: paper
title: "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
authors: [Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach]
year: 2023
venue: arXiv:2307.01952
ingested: 2026-05-31
tags: [sdxl, stable-diffusion, latent-diffusion, ldm, diffusion-model, text-to-image, generative-model, stability-ai, refinement-model, micro-conditioning, multi-aspect-training, clip, openclip]
translation: "[[translations/sdxl]]"
---

# SDXL — Stability AI のオープン LDM 第 3 世代、Stable Diffusion 系統を商用フロンティアと互角に押し上げた決定版

> 原典: [[translations/sdxl]] ・ `raw/papers/SDXL_ Improving Latent Diffusion Models for High-Resolution Image Synthesis.md`
> 著者・年・会議: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach（Stability AI Applied Research）, 2023 Jul, arXiv:2307.01952

## 一言まとめ

**wiki 初の Stable Diffusion / Latent Diffusion Model 系統のページ**。Stability AI の **オープン text-to-image 拡散モデル第 3 世代**（SD 1.x/2.x → SDXL）。**3 倍大きい UNet（2.6B vs 860M）**、**2 つのテキスト・エンコーダ（[[entities/clip|CLIP]] ViT-L + OpenCLIP ViT-bigG, context dim 2048）**、**3 つの新規条件付け技術**（size-conditioning、crop-conditioning、multi-aspect conditioning。すべてフーリエ特徴で timestep 埋め込みに加算）、**改良オートエンコーダ**（ゼロから訓練、batch 256 vs 9、EMA）、**Refinement Model**（同じ潜在空間の別 LDM で SDEdit ベース二段階生成）の 5 主要改善で実現。**ユーザ研究で SDXL+refiner が SD 1.5/2.1 を 5-7× 凌駕**（48.44% vs 7.91%/6.71%）、**Midjourney v5.1 を 54.9% で凌駕**、**DALL-E 2 / Bing Image Creator / DeepFloyd IF を質的比較で凌駕**。**事前学習**: 256² (600k steps) → 512² (200k steps) → multi-aspect (~1024² 面積、40 アスペクト比)。**コードと重みを Apache 2.0 ベースで完全公開**し、画像生成 AI の民主化を推進。**Computer Vision wiki 内では Stable Diffusion 系統の代表として、拡散モデル概念 [[concepts/diffusion-model]] の主要実装例**となる。

## 背景と問題意識

2023 年中頃時点で、**テキスト-画像生成の世界はクローズドソース vs オープンソースの激しい対立**にあった：

- **クローズドソース最先端**: DALL-E 2（OpenAI）、Midjourney、Imagen（Google）、eDiff-I（NVIDIA）—各社が独自の手法と訓練データで圧倒的品質を実現するが、**アーキテクチャと訓練が完全に不透明**
- **オープンソース**: Stable Diffusion 1.x/2.x（Stability AI、Rombach lab）—LDM パラダイムでオープン公開の旗手だが、**商用モデルとの品質ギャップが顕著**

問題提起:
1. **品質ギャップ**: 透明性を犠牲にしないとフロンティアに到達できない状態
2. **再現性**: ブラックボックスでは性能評価とバイアス分析が困難
3. **構造的限界**: SD 1.x/2.x は **絶対サイズ 512² 中心**で、**多様アスペクト比**や **訓練データ廃棄問題**（解像度 < 512 で 39% 廃棄）に悩む
4. **訓練時のランダム・クロッピング漏れ**: 生成画像の物体が **切り取られる**問題（図 4 で示される SD 1.5/2.1 の典型的失敗モード）
5. **古典指標 FID/CLIP の信頼性**: 基盤的 text-to-image モデルでは **人間の選好と相関しない**ことが Kirstain et al. (Pick-a-pic, 2023) で示されつつあった

SDXL は **「すべてオープンで、しかも商用フロンティアと互角」**という挑戦に応える。

## 提案手法 / 主張

### 5 つの主要改善

#### 1. UNet の規模拡大（2.6B、3× 増）

| Model | UNet Params | Transformer Blocks | Channel Mult. | Text Encoder | Context Dim |
| --- | --- | --- | --- | --- | --- |
| **SDXL** | **2.6B** | **[0, 2, 10]** | **[1, 2, 4]** | **CLIP ViT-L + OpenCLIP ViT-bigG** | **2048** |
| SD 1.4/1.5 | 860M | [1, 1, 1, 1] | [1, 2, 4, 4] | CLIP ViT-L | 768 |
| SD 2.0/2.1 | 865M | [1, 1, 1, 1] | [1, 2, 4, 4] | OpenCLIP ViT-H | 1024 |

**主要な変更**:
- **異種 transformer ブロック分布 [0, 2, 10]**: 最高特徴レベルで transformer 省略、低レベル（中解像度）で 2 と 10 ブロック、最低レベル（8× ダウンサンプリング）を**完全削除**
- **理由**: transformer 計算の大部分を低レベル特徴にシフトすることで効率化（simple diffusion からの着想）
- **2 つのテキスト・エンコーダ**: **[[entities/clip|CLIP]] ViT-L (60M) + OpenCLIP ViT-bigG (1.4B) = 817M**、ペナルティメイト出力をチャンネル軸で連結 → **context dim 2048**
- **Pooled OpenCLIP 埋め込み**を timestep 埋め込みに追加（GLIDE 風の追加条件付け）

#### 2. Size-Conditioning（サイズ条件付け）

**問題**: LDM 訓練は最小画像サイズを要求 → SD 1.4/1.5 は < 512px データを廃棄、結果として **39% の訓練データ廃棄**

**解決**: 元の高さ・幅 $\mathbf{c}_{\text{size}} = (h_{\text{original}}, w_{\text{original}})$ を **フーリエ特徴**で埋め込み、timestep 埋め込みに加算

**効果（ImageNet 512²、表 2）**:
| Model | FID-5k ↓ | IS-5k ↑ |
| --- | --- | --- |
| CIN-512-only（< 512 廃棄） | 43.84 | 110.64 |
| CIN-nocond | 39.76 | 211.50 |
| **CIN-size-cond** | **36.53** | **215.34** |

**推論時の活用**: ユーザは $c_{\text{size}}$ を介して **「見かけの解像度」**を制御可能（図 3）。大きな画像サイズを指定すると画質が明らかに向上

#### 3. Crop-Conditioning（クロップ条件付け）

**問題**: 訓練中の **ランダム・クロッピングが生成画像に漏れ**、物体が切れる（猫の頭が切れる典型例、図 4）

**解決**: クロップ座標 $\mathbf{c}_{\text{crop}} = (c_{\text{top}}, c_{\text{left}})$ をフーリエ特徴で埋め込み、追加条件付けとして使用。**Algorithm 1** で size- と crop-conditioning を同時サンプリング

**推論時**: $(c_{\text{top}}, c_{\text{left}}) = (0, 0)$ で **物体中心**サンプル取得。$(c_{\text{top}}, c_{\text{left}})$ を変化させると **クロップ量を制御**可能（図 5）

**利点**: データ・バケッティング（NovelAI）と比較して、**クロッピング誘発データ拡張の恩恵を維持しつつ生成プロセスへの漏れを防ぐ**。実装が単純、オンライン適用可

#### 4. Multi-Aspect Training（マルチ・アスペクト訓練）

**問題**: 共通出力解像度 512² または 1024² は **ランドスケープ（16:9）やポートレート**の使用を考慮していない

**解決**: 
- データを **アスペクト比のバケットに分割**（高さ・幅を 64 の倍数で変化、ピクセル数 ~1024² 維持）
- **40 種類のアスペクト比**（付録 I、0.25 から 4.0）
- バッチは **同じバケット内画像で構成**、ステップごとにバケット交替
- **target size** $\mathbf{c}_{\text{ar}} = (h_{\text{tgt}}, w_{\text{tgt}})$ もフーリエ埋め込みで条件付け
- 固定アスペクト比事前学習後の **微調整段階**として適用

#### 5. Improved Autoencoder & Refinement Model

**改良オートエンコーダ**（表 3）:
- ゼロから訓練、**バッチサイズ 256（vs SD 1.x の 9）**
- **EMA で重み追跡**
- **SDXL-VAE: PSNR 24.7 / SSIM 0.73 / LPIPS 0.88 / rFID 4.4**（全指標で SD 1.x/2.x VAE を凌駕）

**Refinement Model**:
- 同じ潜在空間で別個の LDM を**高品質・高解像度データに特化**
- **最初の 200 ノイズ・スケール**に特化（高ノイズ領域）
- ベース SDXL の潜在変数に **SDEdit ベースの noising-denoising** を適用
- 同じテキスト入力で潜在空間で直接拡散・ノイズ除去（**任意の段階**だが品質が改善）

### 訓練レシピ

**多段階訓練**（1000 ステップの離散時間拡散スケジュール）:
1. **256² 事前学習**: 600,000 ステップ、バッチサイズ 2048、size+crop conditioning
2. **512² 継続**: 200,000 ステップ
3. **マルチ・アスペクト微調整**: ~1024² 面積、**offset-noise レベル 0.05**

## 実験結果と知見

### ユーザ研究（メイン結果、図 1）

| モデル | 勝率 |
| --- | --- |
| **SDXL w/ refinement** | **48.44%** |
| **SDXL base** | **36.93%** |
| Stable Diffusion 1.5 | 7.91% |
| Stable Diffusion 2.1 | 6.71% |

**SDXL + refiner は SD 2.1 比で 7.2× の選好**。SDXL base 単体でも SD 1.5/2.1 を 4-5× 凌駕。

### Midjourney v5.1 との比較（17,153 ユーザ選好）

**SDXL v0.9 は Midjourney V5.1 に対して 54.9% で好まれた**（プロンプト追従性、PartiPrompts P2 ベンチマーク）。

- **6 カテゴリ中 4 で SDXL 凌駕**
- **10 チャレンジ中 7 で SDXL 凌駕または同等**

### Size-Conditioning ImageNet（表 2、ablation）

CIN-size-cond は **FID 36.53**（vs CIN-nocond 39.76、CIN-512-only 43.84）、IS 215.34 で**両指標を改善**。

### Autoencoder（表 3）

SDXL-VAE が 4 指標すべてで SD 1.x/2.x VAE を凌駕。

### FID/CLIP の信頼性問題（付録 F、図 12）

**SDXL の FID は SD 1.5/2.1 より悪い**、CLIP-score もわずかに改善するだけ—**人間評価者は明らかに SDXL を好むにもかかわらず**。

→ Kirstain et al. (Pick-a-pic, 2023) の **「COCO ゼロショット FID は視覚美学と負の相関」**という発見を再確認。**基盤的 text-to-image モデルには新しい評価指標が必要**

## 限界・批判的視点

論文の付録 B + 批判的視点で挙げられる弱点：

- **人間の手**: 複雑な解剖学（指の数、ポーズ）でしばしば失敗（図 7 上左）
- **完全な写真現実性**: 微妙な照明・テクスチャ変化に不完全
- **社会的・人種的バイアス**: 大規模訓練データに不可避的に含まれる
- **Concept bleeding（概念漏れ）**: 「青い帽子」と「赤い手袋」のペンギンが「赤い帽子」と「青い手袋」になるような、属性バインディング失敗。**CLIP の単一トークン圧縮と対比損失の特性**に起因
- **長く判読可能なテキストのレンダリング**: ランダム文字や不整合
- **二段階パイプライン**: 2 モデルをメモリにロードする必要、推論コスト増加
- **推論速度**: 2.6B UNet + 2 テキスト・エンコーダ + refiner = SD 1.x/2.x より遅い
- **離散時間拡散**: offset-noise 必須、EDM の連続時間フレームワークの方が将来性あり
- **Transformer ベース構造の試行**: UViT、DiT を実験したが SDXL 段階では即座の利益なし（後の SD3 / FLUX が DiT 系に移行）
- **FID/CLIP-score の不適合**: 古典指標では性能向上が反映されない
- **訓練データ非公開**: モデルとコードは公開だが訓練データの詳細は不透明
- **古典 CLIP 系テキスト・エンコーダ**: T5 のような大型 LLM テキスト・エンコーダの方がプロンプト追従性で優れる可能性（後の SD3 / FLUX で確認）

## CV 分野における意義

SDXL は CV/生成モデル史において以下の意義を持つ：

1. **wiki 初の拡散モデル (Diffusion Model) 系統の ingest**: これまで Qwen/InternVL/Gemma 等の MLLM、SAM/DINO 等の認識系を ingest してきたが、**生成モデル系の代表**を欠いていた。SDXL は [[concepts/diffusion-model]] の主要実装例として位置付けられる
2. **オープン text-to-image の品質ジャンプ**: Midjourney v5.1 を 54.9% で凌駕することで、**オープンソースが商用と肩を並べる**ことを実証
3. **「Open Foundation Model」の代表**: Stability AI の代表的成果として、[[concepts/foundation-model]] の生成系基盤モデル代表
4. **CLIP の主要応用例**: [[entities/clip|CLIP]] ViT-L + OpenCLIP ViT-bigG の二重テキスト・エンコーダは、**CLIP/OpenCLIP の最大の実用例の 1 つ**
5. **3 つの新規条件付け技術の影響**: Size-/Crop-/Multi-aspect conditioning は **後続の拡散モデル（SD3、FLUX、Imagen 2、Stable Cascade 等）で標準化**
6. **Refinement Model パラダイム**: SDEdit ベースの 2 段階生成は、後の eDiff-I、IDM-VTON、AYS（Align Your Steps）等で発展
7. **古典指標 FID/CLIP の信頼性問題**: 基盤的 text-to-image モデルでの **人間評価の重要性**を学術的に確認した重要な発見
8. **Stability AI の Rombach lab**: LDM の原論文（Rombach et al., 2022）の正統な後継、後の SD3（2024）/ FLUX（2024、同チームの分派）の前身

Computer Vision wiki 内では：
- **[[concepts/diffusion-model]] の主要実装例**: LDM パラダイムの代表
- **[[entities/clip|CLIP]] の主要応用**: ViT-L のテキスト・エンコーダ部分が SDXL に使われる
- **[[concepts/foundation-model]] の生成系代表**: 認識系（SAM、DINOv3、CLIP）と並ぶ生成系基盤モデル
- **[[concepts/vision-transformer]] との対比**: SDXL は ViT 系を採用せず UNet ベースを維持（後の DiT で逆転）

## 用語と略称

- **SDXL** = Stable Diffusion XL（Stability AI の拡張版 Stable Diffusion）
- **Stable Diffusion (SD)** = Stability AI のオープンソース text-to-image LDM 系統
- **LDM (Latent Diffusion Model)** = 潜在拡散モデル、画素空間ではなく **VAE で圧縮された潜在空間で拡散**を行う方式（Rombach et al., 2022）
- **VAE (Variational Autoencoder)** = 変分オートエンコーダ、LDM の潜在空間を提供
- **UNet** = U 字型の畳み込み + 自己注意ネットワーク、拡散モデルの denoiser 標準アーキテクチャ（Ronneberger et al., 2015）
- **DM (Diffusion Model)** = 拡散モデル一般（DDPM, DDIM 等を含む）
- **DDPM** = Denoising Diffusion Probabilistic Model（Ho et al., 2020、拡散モデルの祖）
- **DDIM** = Denoising Diffusion Implicit Model（決定論的サンプラー、Song et al., 2020）
- **SDEdit** = Stochastic Differential Editing（潜在空間の image-to-image 編集、Meng et al., 2021）
- **EDM** = Elucidating the Design space of Diffusion-based generative Models（連続時間 DM フレームワーク、Karras et al., 2022）
- **DiT** = Diffusion Transformer（拡散モデルの transformer 化、Peebles & Xie, 2022、後の SD3/FLUX の基盤）
- **UViT** = U-ViT（拡散用 ViT、Hoogeboom et al., 2023）
- **DSM** = Denoising Score Matching（拡散モデルの標準訓練目的）
- **Classifier-free guidance (CFG)** = 分類器なしガイダンス（Ho & Salimans, 2022、条件付けを強化する標準技法）
- **cfg-scale / guidance scale** = CFG の強度パラメータ $w$
- **Fourier feature embedding** = フーリエ特徴埋め込み（正弦波 timestep 埋め込みと同じ手法）
- **Offset-noise** = 拡散モデルで美的品質を改善するノイズ・スケジュール補正（Guttenberg, 2023）
- **EMA** = Exponential Moving Average（重みの指数移動平均）
- **CLIP** = Contrastive Language-Image Pre-training（[[entities/clip]]）
- **OpenCLIP** = LAION による CLIP の公開再現（[[entities/clip]]）
- **CLIP ViT-L** = CLIP の Large variant（60M パラメータ、context dim 768）
- **OpenCLIP ViT-bigG** = OpenCLIP の bigG variant（1.4B パラメータ、Stability AI 訓練）
- **OpenCLIP ViT-H** = OpenCLIP の Huge variant（SD 2.x で使用）
- **PSNR / SSIM / LPIPS / rFID** = オートエンコーダ評価指標（再構成品質）
- **Pooled text embedding** = テキスト・エンコーダの pooling 後の単一ベクトル
- **PartiPrompts (P2)** = Parti 論文のベンチマーク・プロンプト集（カテゴリとチャレンジ）
- **AWS GroundTruth** = ユーザ研究のクラウドソーシング・プラットフォーム
- **Pick-a-pic** = テキスト-画像生成のユーザ選好データセット（Kirstain et al., 2023）
- **FID (Fréchet Inception Distance)** = 生成画像評価指標（Inception ネットの特徴量分布距離）
- **IS (Inception Score)** = 生成画像評価指標（Inception ネットの分類確率）
- **CLIP-score** = テキストと画像の CLIP 埋め込み類似度
- **COCO** = Microsoft Common Objects in Context（評価データセット）
- **ImageNet CIN** = Class-conditional ImageNet（クラス条件付き生成評価）
- **Probability Flow ODE** = 拡散の決定論的サンプリング ODE
- **Wiener process** = ウィーナー過程（標準ブラウン運動）
- **Score function** = スコア関数 $\nabla_\mathbf{x} \log p(\mathbf{x})$
- **Score matching** = スコア・マッチング（拡散モデルの理論的基礎、Hyvärinen, 2005）
- **Multi-aspect bucketing** = アスペクト比バケッティング（NovelAI 由来）
- **Refinement model / refiner** = SDXL の二段階生成の第 2 段階モデル
- **Two-stage pipeline** = SDXL の base → refiner の 2 段階生成
- **Concept bleeding** = 異なる視覚概念の意図しない融合（例: 帽子と手袋の色が入れ替わる）

## 関連ページ

- [[entities/sdxl]] — SDXL ファミリー（base + refiner + VAE）のエンティティ詳細
- [[translations/sdxl]] — 本論文の本文 + 付録翻訳（Acknowledgements と References を除外）
- [[concepts/diffusion-model]] — LDM パラダイムの理論的基盤
- [[entities/clip]] — SDXL の主要テキスト・エンコーダ（ViT-L）
- [[concepts/vision-transformer]] — テキスト・エンコーダの基盤
- [[concepts/foundation-model]] — 生成系基盤モデルの代表
- [[concepts/weakly-supervised-pretraining]] — 大規模 text-image 対による事前学習
- [[concepts/contrastive-learning]] — CLIP の基盤
