---
type: entity
entity_kind: model
aliases: [NFNet, NF-ResNet, Normalizer-Free Network, AGC, Adaptive Gradient Clipping]
tags: [nfnet, normalizer-free, agc, convolutional-neural-network, backbone, deepmind]
related: ["[[concepts/convolutional-neural-network]]", "[[entities/convnext]]", "[[entities/convnext-v2]]", "[[entities/maxvit]]", "[[concepts/self-supervised-learning]]"]
sources: ["[[sources/nfnet]]"]
updated: 2026-06-17
---

# NFNet（Normalizer-Free Network）

DeepMind の **Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan** による、BatchNorm を一切使わない CNN ファミリー（arXiv:2102.06171, **ICML 2021**）。詳細解説: [[sources/nfnet]] / 翻訳: [[translations/nfnet]]。

## 一言で

**BatchNorm を完全に排除しても、むしろ速く・強くなることを実証した純粋 CNN**。BN が担う 4 つの役割を個別の仕掛けで代替し、大バッチ訓練の不安定さを **AGC** で解決。追加データなしの ImageNet で **86.5%**（当時 SOTA）、JFT-300M 事前学習で **89.2%**。**EfficientNet-B7 と同精度を 8.7× 速い訓練で**達成する。

## 3 つの構成要素

### 1. Normalizer-Free の骨格（先行研究 NF-ResNet を継承）

$$h_{i+1}=h_{i}+\alpha f_{i}(h_{i}/\beta_{i})$$

- $f_i$ は初期化時に**分散保存的**、$\alpha$（既定 0.2）が分散の増加率を決める
- $\beta_i=\sqrt{\text{Var}(h_i)}$ は $\text{Var}(h_{i+1})=\text{Var}(h_i)+\alpha^2$ から**解析的に予測**
- 遷移ブロックで分散を $1+\alpha^2$ に**リセット**（BN 版の挙動を模倣）
- **SkipInit**: ゼロ初期化の学習可能スカラーで各ブロックを恒等写像から開始
- **Scaled Weight Standardization** $\hat{W_{ij}}=(W_{ij}-\mu_i)/(\sqrt{N}\sigma_i)$ で**平均シフト**を除去。活性化も非線形性ごとのゲイン $\gamma$ でスケール

### 2. AGC（Adaptive Gradient Clipping）— 本モデルの中核

**勾配ノルムそのものではなく「勾配ノルム / 重みノルム」の比**でクリップする。これは「1 ステップで重みがどれだけ変わるか」の尺度。

$$
G^{\ell}_{i}\rightarrow\lambda\frac{\|W^{\ell}_{i}\|^{\star}_{F}}{\|G^{\ell}_{i}\|_{F}}G^{\ell}_{i} \quad \text{if}\ \frac{\|G^{\ell}_{i}\|_{F}}{\|W^{\ell}_{i}\|^{\star}_{F}}>\lambda
$$

- **層単位ではなくユニット単位**（重み行列の行ごと）に適用するのが経験的に良い
- $\|W_i\|^\star_F=\max(\|W_i\|_F,\epsilon)$、$\epsilon=10^{-3}$
- **最終線形層には適用しない**（常にその方が良い）。ただし 4 ステージすべての重みはクリップしないとバッチ 4096 で不安定
- **バッチが大きいほど $\lambda$ は小さく**。バッチ 1024〜4096 では $\lambda=0.01$
- LARS 等の「正規化オプティマイザ」の**緩和版**（上限だけ課し下限は課さない）

### 3. アーキテクチャ（SE-ResNeXt-D からの 4 変更）

| 変更 | 理由 |
|---|---|
| **グループ幅を 128 に固定** | 小さいグループ幅は FLOPS を減らすが計算密度が落ち**実速度が上がらない** |
| **深さ $[1,2,6,3]\times N$** | ResNet の非一様な深さ増加は最適でない。全ステージを一律 $N$ 倍 |
| **幅 $[256,512,1536,1536]$** | 既定 $[256,512,1024,2048]$ より良い唯一の候補。**第 3 ステージに容量を足す** |
| **$3\times3$ グループ化 conv を追加** | FLOPS と訓練時間への影響が最小で精度が上がる |

- **設計指針が独特**: 理論 FLOPS でも推論レイテンシでもなく、**実機（TPUv3 / V100）の訓練レイテンシ**を最適化対象にした
- **スケーリングは深さと解像度のみ**（幅はスケールしない）。推論は訓練解像度の約 1.33 倍、**その解像度での再訓練はしない**
- ステム: 3×3 s2 (16ch) → 3×3 s1 (32ch) → 3×3 s1 (64ch) → 3×3 s2 (128ch)
- bottleneck 比 0.5、S&E 層あり（出力を **2×** して分散維持）、GELU

## バリアント

**表**: NFNet ファミリー（[[sources/nfnet]] 表1 / 表3 より）

| Variant | Depth | Train / Test 解像度 | params | FLOPs | IN-1K Top-1 | TPUv3 訓練 |
|---|---|---|---|---|---|---|
| **F0** | [1, 2, 6, 3] | 192 / 256px | 71.5M | 12.38B | **83.6** | 73.3 ms |
| **F1** | [2, 4, 12, 6] | 224 / 320px | 132.6M | 35.54B | **84.7** | 158.5 ms |
| **F2** | [3, 6, 18, 9] | 256 / 352px | 193.8M | 62.59B | **85.1** | 295.8 ms |
| **F3** | [4, 8, 24, 12] | 320 / 416px | 254.9M | 114.76B | **85.7** | 532.2 ms |
| **F4** | [5, 10, 30, 15] | 384 / 512px | 316.1M | 215.24B | **85.9** | 1033.3 ms |
| **F5** | [6, 12, 36, 18] | 416 / 544px | 377.2M | 289.76B | **86.0**（+SAM **86.3**） | 1398.5 ms |
| **F6** | [7, 14, 42, 21] | 448 / 576px | 438.4M | 377.28B | +SAM **86.5** | 2774.1 ms |
| **F4+** | F4 の幅広版 | — | 527M | 367B | **89.2**（JFT 事前学習） | — |

- **F4+ は幅を $[384,768,2048,2048]$ に変えた変種**（他は F4 と同じ）。JFT 事前学習専用で、F4 の 40 エポックに対し **20 エポックで同性能**
- Dropout 率はモデルが大きいほど上げる（0.2 → 0.5）。Stochastic Depth は全変種 0.25

## 主要結果

- **ImageNet（追加データなし）**: **NFNet-F6 + SAM で 86.5%** — 当時の SOTA
- **NFNet-F1 が EffNet-B7 と同じ 84.7% を 8.7× 速い訓練で達成**（158.5ms vs 1397.0ms）
- **JFT-300M → ImageNet**: **NFNet-F4+ で 89.2%**。転移学習で達成された当時の最高精度（全体では EffNet-L2 + Meta Pseudo Labels の 90.2% に次ぐ 2 位）
  - **計算効率も高い**: 1.86k TPUv3-core-days（Meta Pseudo Labels は 22.5k で 12 倍）
- **NF > BN が転移で一貫**: NF-ResNet が BN-ResNet を 50/101/152/200 の全サイズ・全解像度で**約 1 ポイント上回る**
- **同一アーキテクチャの BN 版との比較**: 精度はわずかに低く、**訓練は 20-40% 遅い**。さらに **F4/F5 は BN では訓練が不安定**（bfloat16 との相互作用）

## 系譜・位置づけ

- **BN 排除という問題設定の到達点**。先行研究の NF-ResNet が「BN 版に匹敵するが大バッチで不安定・EfficientNet に届かない」で止まっていたのを、AGC とアーキテクチャ設計で突破した。
- **[[entities/convnext]] とは「正規化問題への別解」の関係**。ConvNeXt（2022）も BN を捨てたが、**採用したのは LayerNorm** であって Normalizer-Free ではない。NFNet の Scaled WS + AGC + SkipInit は部品と調整項目が多く（$\alpha,\beta,\lambda,\epsilon,\gamma$）、LN に置き換える方が単純だった、というのが後年の評価。
- **[[entities/convnext-v2]] は別角度から BN を批判**: GRN のアブレーションで **BN がマスク入力と相性が悪く 80.5 まで落ちる**ことを示した（LN は 83.8）。
- **本 wiki が最初に NFNet を知った経路は [[sources/maxvit]] の比較表**（NFNet-F4+ 89.20 として登場）。[[questions/cnn-beyond-convnext-v2]] で「ConvNeXt V2 の 88.9% を超える純粋 CNN」として引用した数字の原典が本論文。
- **attention を一切試していない**。著者自ら「採用すれば結果は改善しうる」と述べており、実際その後 CV は attention 側へ動いた（[[entities/swin-transformer]] / [[entities/maxvit]]）。
- **BN 排除は主流にならなかった**が、**BN の欠点の整理（特に対比学習での情報漏洩）は広く引用され続けている**（[[concepts/self-supervised-learning]] / [[entities/moco]]）。

## 弱点

- **パラメータ効率が悪い**: NFNet-F1 132.6M で 84.7% に対し EffNet-B7 は 66.0M で同じ 84.7%（**2 倍**）。著者も「FLOPs 対精度では競争力があるがパラメータ対精度ではそうでない」と明記
- **最高値は非公開データ依存**: 89.2% は JFT-300M。[[sources/convnext-v2|ConvNeXt V2]] の 88.9% が公開データのみであることと直接比較できない
- **AGC はハイパラフリーではない**: $\lambda$ はバッチサイズ・学習率・オプティマイザに依存
- **部品が多い**: $\alpha$, $\beta$, SkipInit, Scaled WS, $\gamma$ ゲイン, AGC の $\lambda$/$\epsilon$ と調整項目が多く、LN 1 個に置き換える ConvNeXt の単純さに対して不利
- **訓練レイテンシ最適化という前提の賞味期限**: 2021 年の TPUv3/V100 が前提

## 公開

- コード / 重み: <https://github.com/deepmind/deepmind-research/tree/master/nfnets>（Apache 2.0、JAX / Haiku 実装）
- `timm` に `nfnet_f0` 〜 `nfnet_f6`、`nf_resnet50`、`eca_nfnet_*` 等として収録（PyTorch 移植は Ross Wightman による）

## 関連ページ

- [[sources/nfnet]] — 論文要約（最重要）
- [[translations/nfnet]] — 全文和訳（Appendix A〜E 込み）
- [[concepts/convolutional-neural-network]] — CNN の系譜と BN vs LN の議論
- [[entities/convnext]] / [[sources/convnext]] — 正規化問題への別解（BN → LN）
- [[entities/convnext-v2]] — BN がマスク入力と相性が悪いという別角度の批判
- [[questions/cnn-beyond-convnext-v2]] — 本モデルの 89.2% を引用した query
- [[entities/maxvit]] / [[sources/maxvit]] — NFNet-F4+ を比較表に載せている
- [[concepts/self-supervised-learning]] / [[entities/moco]] — BN の情報漏洩問題の文脈
