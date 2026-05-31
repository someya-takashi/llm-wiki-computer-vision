---
type: concept
aliases: [Diffusion Model, DDPM, Denoising Diffusion Probabilistic Model, 拡散モデル]
tags: [generative-model, ddpm, unet, score-based, latent-diffusion, stable-diffusion, synthetic-data]
related: [[concepts/denoising-autoencoder]], [[concepts/foundation-model]], [[concepts/self-supervised-learning]]
sources: [[sources/i-synmed]]
updated: 2026-05-28
---

# Diffusion Model（拡散モデル）

## 一言で

**画像を段階的にノイズに破壊する「順方向プロセス」と、ノイズから画像を段階的に復元する「逆方向プロセス」を学習する生成モデル**。学習時はモデルに「各時刻のノイズ量を予測する」タスクを与え、推論時はランダムノイズから始めて学習したデノイザを反復適用することで新しい画像を生成する。

2020 年の **DDPM（Denoising Diffusion Probabilistic Models, Ho et al.）** の登場後、GAN を上回る生成品質と訓練の安定性で生成モデル分野を席巻。2022 年の **Stable Diffusion（Rombach et al.）** で画像生成 AI が一般化、Midjourney・DALL-E 2・Sora などすべての主流生成モデルが拡散モデル系。

## なぜ拡散モデルが革命的だったか

それ以前の生成モデルには以下の問題があった：

| モデル | 問題 |
|---|---|
| GAN | mode collapse、訓練不安定、評価困難 |
| VAE | 出力がぼやけがち、表現力不足 |
| Flow-based | 計算コスト過大、表現力に制限 |
| Autoregressive (PixelCNN 等) | 推論が逐次的で遅い |

拡散モデルは「**段階的ノイズ除去**」というシンプルな目的により、

- **訓練が安定**（GAN のような adversarial 訓練不要）
- **mode collapse なし**（明示的な尤度ベース目的）
- **高品質**（FID で GAN を凌駕）
- **多様性が高い**（モードカバレッジ良好）

を同時に達成した。

## 数学的フレームワーク

### 順方向拡散（forward diffusion）

データ $x_0$ にガウシアンノイズを徐々に加える**マルコフ過程**：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)$$

ここで $\beta_t \in (0, 1)$ はノイズスケジュール。$t$ が大きくなるにつれてデータは純粋なガウシアンノイズ $\mathcal{N}(0, \mathbf{I})$ に近づく。

**重要な性質**: 累積ノイズ係数 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ を使うと、任意の $t$ で $x_0$ から $x_t$ を一発でサンプリングできる：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

これが**再パラメータ化トリック**で、訓練を劇的に効率化する。

### 逆方向拡散（reverse diffusion）

学習目標は「**各時刻 $t$ で加えられたノイズ $\epsilon$ を予測するネットワーク $\epsilon_\theta$**」：

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

推論時は $x_T \sim \mathcal{N}(0, \mathbf{I})$ から始めて、各 $t$ で：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

を反復適用（典型的に 1000 ステップ、後の DDIM では 50 ステップに削減）。

## 標準アーキテクチャ：UNet

$\epsilon_\theta(x_t, t)$ の実装には **UNet**（Ronneberger et al., 2015）が標準的に使われる：

- エンコーダ-デコーダ構造（解像度を下げて上げる）
- skip connection（細部情報を保持）
- 時刻 $t$ を Sinusoidal embedding または学習可能 embedding で注入
- self-attention 層を中間に挿入（特に Stable Diffusion 系）

最近の傾向は **Diffusion Transformer (DiT)**（Peebles & Xie, 2023）で UNet を ViT に置き換える流れ。Sora 等の動画生成モデルが採用。

## 拡散モデルの系譜

```
DPM (Sohl-Dickstein, 2015) ── 理論的起源（非平衡熱力学）
     ↓
DDPM (Ho et al., 2020) ──── 実用化、UNet 採用、FID で GAN を凌駕
     ↓
DDIM (Song et al., 2021) ── 決定論的サンプリング、推論を 1000→50 ステップに
     ↓
Score-based SDE (Song et al., 2021) ── SDE フレームワークで統一理論化
     ↓
Classifier-free Guidance (Ho & Salimans, 2022) ── 条件付き生成の標準
     ↓
LDM / Stable Diffusion (Rombach et al., 2022) ── 潜在空間で拡散、計算量激減
     ↓
[[entities/sdxl|SDXL]] (Podell et al., Stability AI, 2023) ── オープン SD 系第 3 世代、3× UNet (2.6B) + 二重テキストエンコーダ + 3 新規条件付け + refinement、Midjourney v5.1 を 54.9% で凌駕
     ↓
DiT (Peebles & Xie, 2023) ── UNet を Transformer に
     ↓
Sora (OpenAI, 2024) ── 動画生成、Diffusion Transformer
     ↓
SD3 (Stability AI, 2024) / FLUX.1 (Black Forest Labs, 2024) ── DiT + T5 大規模テキストエンコーダ
```

### Stable Diffusion 系統の詳細

- **LDM (Rombach et al., 2022)**: 潜在拡散の原論文。VAE で 4-8× 空間圧縮した潜在空間で拡散、画素空間より計算量を劇的に削減
- **Stable Diffusion 1.x** (2022 Aug): 860M UNet、CLIP ViT-L、512²
- **Stable Diffusion 2.x** (2022 Nov): 865M UNet、OpenCLIP ViT-H、768²
- **[[entities/sdxl|SDXL]]** (2023 Jul、[[sources/sdxl]]): **wiki で詳細 ingest 済み**。2.6B UNet、CLIP ViT-L + OpenCLIP ViT-bigG (817M)、context dim 2048、size/crop/multi-aspect 条件付け、改良 VAE、refinement model
- **SDXL Turbo** (2023 Nov): 蒸留版、1 ステップ推論
- **Stable Cascade** (2024 Feb): Würstchen アーキテクチャ
- **SD3** (2024 Feb): **DiT 系に移行、T5-XXL テキストエンコーダ追加**
- **FLUX.1** (Black Forest Labs, 2024): Rombach ら独立、12B Hybrid MM-DiT、SDXL/SD3 の正統な後継的存在

## 拡散モデルと既存概念との接続

### [[concepts/denoising-autoencoder]] との関係

拡散モデルは「**ノイズレベルで条件付けされた連続的なデノイジングオートエンコーダ**」と解釈できる。DAE が単一のノイズレベルでの再構成だったのに対し、拡散モデルは全ノイズレベル $t \in [0, T]$ で訓練する。理論的にはこれを連続化したものが **score-based generative model**。

### 自己教師あり学習との関係

拡散モデル訓練自体が「ノイズ予測」という自己教師あり目的。最近では拡散モデルの中間特徴を [[concepts/self-supervised-learning]] の表現として使う研究も活発（**Diffusion Representation Learning**）。

### [[entities/mae]] との比較

| 観点 | MAE | DDPM |
|---|---|---|
| 破損方法 | パッチ単位のマスク | 連続的ガウシアンノイズ |
| 復元目的 | 隠したパッチ | 加えられたノイズ |
| 主用途 | 表現学習 | 生成 |
| 共通点 | 破損 → 復元という [[concepts/denoising-autoencoder]] の枠組み |  |

## 拡散モデルの応用領域

| 分野 | 代表モデル |
|---|---|
| **テキスト→画像** | DALL-E 2, Stable Diffusion, Midjourney, Imagen |
| **画像編集** | InstructPix2Pix, ControlNet |
| **画像→画像変換** | Palette, SR3（超解像） |
| **動画生成** | Sora, Stable Video Diffusion, VideoLDM |
| **3D 生成** | DreamFusion, Magic3D |
| **医療画像合成** | [[sources/i-synmed]] 等（プライバシー保護目的）|
| **音声生成** | DiffWave, Grad-TTS |
| **タンパク質構造予測** | RFDiffusion |

## SSL における拡散モデル：合成データ事前学習

[[sources/i-synmed]]（IEEE Access 2025）は「**DDPM 生成画像で DINO を事前学習しても実画像と同等性能**」を医療画像で実証。これは：

- 大規模実データが集めにくい領域（医療、衛星、産業検査）への適用
- プライバシー保護目的での合成データ訓練
- Web スケール合成データ（**Synthetic Data Scaling Laws**）の前段階研究

として位置づけられる。

> **批判的視点**: 拡散モデルが訓練データを memorization する問題（Carlini et al., 2023）があり、「合成データだからプライバシー保護」は単純には成立しない。生成画像が訓練画像とどれだけ独立かは慎重に評価する必要がある。

## 限界と未解決問題

1. **推論コスト**: 1000 ステップの反復が必要（DDIM/Consistency Models で改善中）
2. **訓練データ依存**: 大規模・高品質データが必須
3. **記憶問題**: 訓練画像の memorization リスク（プライバシー / 著作権問題）
4. **コントロール性**: 細かい意図の反映は ControlNet 等の追加が必要
5. **評価指標の限界**: FID だけでは「意味的な妥当性」を測れない

## 参考

- [[sources/i-synmed]]: DDPM 合成画像で DINO 事前学習する医療応用例（IEEE Access 2025）
- [[concepts/denoising-autoencoder]]: 拡散モデルの理論的祖先
- [[concepts/self-supervised-learning]]: 拡散モデル訓練自体が自己教師あり、表現学習にも転用可能
- [[concepts/foundation-model]]: 拡散モデルは生成系基盤モデルの中核（Sora, Stable Diffusion）
- [[entities/dino]]: I-SynMed で DDPM 合成画像を使って事前学習された SSL アルゴリズム
- [[entities/mae]]: DDPM と理論的に近い denoising 系（マスク復元）
