---
type: translation
source_path: raw/papers/InceptionNeXt_ When Inception Meets ConvNeXt.md
source_page: "[[sources/inceptionnext]]"
original_language: en
translated_to: ja
translated_at: 2026-08-31
---

# InceptionNeXt: Inception が ConvNeXt に出会うとき

> 原題: InceptionNeXt: When Inception Meets ConvNeXt
> 著者: Weihao Yu, Pan Zhou, Shuicheng Yan, **Xinchao Wang**（責任著者）
> 所属: National University of Singapore / Singapore Management University / Sea AI Lab / Skywork AI
> 出典: arXiv:2303.16900 → **CVPR 2024**
> コード: <https://github.com/sail-sg/inceptionnext>

> 翻訳メモ: 本翻訳は CLAUDE.md §4 の標準ルールと異なり、**Appendix A・B も翻訳対象に含めている**（本論文の Appendix は実験のハイパーパラメータ表と Grad-CAM の定性結果のみで分量が小さいため）。References のみ除外。原典 markdown（ar5iv 由来）には**画像が 1 枚（Grad-CAM の 1 コマ）しか埋め込まれておらず、図1（精度 vs スループットのトレードオフ）・図2（ブロックの図解）・図3（FLOPs の比較）がすべて欠落している**ため、図は arXiv e-print（arXiv:2303.16900）に同梱された PDF から生成した。巨大な HTML テーブルは markdown テーブルに変換している。

## Abstract（要旨）

ViT の長距離モデリング能力に着想を得て、受容野を広げモデルの性能を改善するために大カーネル畳み込みが近年広く研究され採用されてきた。$7\times 7$ の depthwise 畳み込みを採用する注目すべき研究である ConvNeXt がその例である。**そのような depthwise 演算子はわずかな FLOPs しか消費しないにもかかわらず、高いメモリアクセスコストのために強力な計算装置上でのモデルの効率を大きく損なう。** 例えば **ConvNeXt-T は ResNet-50 と同程度の FLOPs を持つが、A100 GPU 上で完全精度で訓練したときのスループットは約 60% しか達成しない。** ConvNeXt のカーネルサイズを減らせば速度は改善できるが、それは有意な性能の劣化をもたらす。これは困難な問題を提起する。**大カーネルに基づく CNN モデルを、その性能を保ちながらどう高速化するか。**

この問題に取り組むため、Inception に着想を得て、我々は**大カーネルの depthwise 畳み込みをチャネル次元に沿って 4 つの並列な分岐——小さな正方カーネル、2 つの直交する帯状カーネル、そして恒等写像——に分解する**ことを提案する。この新しい **Inception depthwise 畳み込み**を用いて、我々は **InceptionNeXt** と名付けた一連のネットワークを構築する。それは高いスループットを享受するだけでなく、競争力のある性能も維持する。例えば **InceptionNeXt-T は ConvNeXt-T より 1.6 倍高い訓練スループットを達成し、同時に ImageNet-1K で 0.2% の top-1 精度の改善を得る。** 我々は InceptionNeXt が、炭素排出量を削減するための今後のアーキテクチャ設計の**経済的なベースライン**として役立つことを期待する。

## 1 Introduction（はじめに）

<figure>

![](../../raw/assets/inceptionnext/fig1.png)

<figcaption>図1: 精度と訓練スループットのトレードオフ。すべてのモデルは DeiT の訓練ハイパーパラメータのもとで訓練されている。訓練スループットはバッチサイズ 128 で A100 GPU 上で測定。ConvNeXt-T/kn はカーネルサイズ n×n の派生を意味する。InceptionNeXt-T は ResNet-50 の速度と ConvNeXt-T の精度の両方を享受する。図中では ConvNeXt-T/k7 が約 575 img/s で 82.1%、k5 が約 675 で 82.0%、k3 が約 800 で 81.5%、Swin-T が約 535 で 81.3%、ResNet-50 が約 970 で 78.4%、InceptionNeXt-T が約 900 で 82.3% に位置し、ConvNeXt-T/k7 から InceptionNeXt-T へ「1.6× speedup」の矢印が引かれている。</figcaption>
</figure>

深層学習の歴史を振り返ると、**畳み込みニューラルネットワーク（CNN）**は間違いなくコンピュータビジョンで最も人気のあるモデルである。分水嶺の瞬間は 2012 年に AlexNet が ImageNet のコンテストで勝利を主張したときに訪れ、コンピュータビジョンにおける CNN の新時代を開いた。それ以来、Network In Network、VGG、Inception Nets、ResNe(X)t、DenseNet、その他の効率的なモデルといった無数の影響力ある CNN が現れた。

<figure>

![](../../raw/assets/inceptionnext/fig2.png)

<figcaption>図2: MetaFormer、MetaNeXt、ConvNeXt、InceptionNeXt のブロックの図解。MetaFormer ブロックと同様に、MetaNeXt は ConvNeXt から抽象化された一般的なブロックである。MetaNeXt は MetaFormer から 2 つの残差サブブロックを 1 つに統合して得られる、より単純な版と見なせる。注目すべきは、MetaNeXt で用いるトークン混合器は複雑すぎてはならない（例: 自己注意）ということである。さもなくば収束するよう訓練できないかもしれない。トークン混合器を depthwise 畳み込みまたは Inception depthwise 畳み込みとして特定することで、モデルはそれぞれ ConvNeXt ブロックまたは InceptionNeXt ブロックとして具体化される。ConvNeXt と比べて、InceptionNeXt は高価な大カーネル depthwise 畳み込みを 4 つの効率的な並列分岐（DWConv 3×3 / DWConv 1×11 / DWConv 11×1 / Identity）に分解するのでより効率的である。</figcaption>
</figure>

## 2 Related work（関連研究）

### 2.1 Transformer v.s. CNN

（Transformer と CNN の比較の系譜が記述されている。）

### 2.2 Convolution with large kernels.（大カーネルの畳み込み）

（大カーネル畳み込みの研究の系譜が記述されている。）

## 3 Formulation and Method（定式化と手法）

**Algorithm 1**: Inception Depthwise Convolution（PyTorch 風のコード）

```python
import torch.nn as nn

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3,
                 band_kernel_size=11, branch_ratio=1/8):
        super().__init__()
        gc = int(in_channels * branch_ratio)  # 各畳み込み分岐のチャネル数
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size,
                                   padding=square_kernel_size//2, groups=gc)
        self.dwconv_w  = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size),
                                   padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h  = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1),
                                   padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        # B, C, H, W = x.shape
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (self.dwconv_hw(x_hw),
             self.dwconv_w(x_w),
             self.dwconv_h(x_h),
             x_id),
            dim=1)
```

### 3.1 MetaNeXt

#### MetaNeXt ブロックの定式化

ConvNeXt では、各 ConvNeXt ブロックについて、入力 $X$ はまず空間次元に沿って情報を伝播させるために depthwise 畳み込みで処理される。我々は **MetaFormer** に従い、depthwise 畳み込みを**空間的な情報の相互作用を担うトークン混合器（token mixer）**として抽象化する。それに応じて、図2 の 2 番目の部分図に示すように、**ConvNeXt は MetaNeXt ブロックとして抽象化される**。形式的には、MetaNeXt ブロックにおいて入力 $X$ はまず次のように処理される。

$$
X^{\prime}=\mathrm{TokenMixer}(X),
$$

ここで $X,X^{\prime}\in\mathbb{R}^{B\times C\times H\times W}$ であり、$B$、$C$、$H$、$W$ はそれぞれバッチサイズ、チャネル数、高さ、幅を表す。次にトークン混合器からの出力が正規化される。

$$
Y=\mathrm{Norm}(X^{\prime}).
$$

正規化の後、特徴は 2 つの全結合層とその間の活性化関数からなる MLP モジュールに供給される。これは Transformer のフィードフォワードネットワークと同じである。2 つの全結合層は $1\times 1$ 畳み込みでも実装できる。またショートカット接続も採用される。このプロセスは次のように表せる。

$$
Y=\mathrm{Conv}_{1\times 1}^{rC\rightarrow C}\{\sigma[\mathrm{Conv}_{1\times 1}^{C\rightarrow rC}(Y)]\}+X,
$$

ここで $\mathrm{Conv}_{k\times k}^{C_{i}\rightarrow C_{o}}$ はカーネルサイズ $k\times k$、入力チャネル $C_{i}$、出力チャネル $C_{o}$ の畳み込みを意味し、$r$ は拡張率、$\sigma$ は活性化関数を表す。

#### MetaFormer ブロックとの比較

図2 に示すように、MetaNeXt ブロックが MetaFormer ブロックと似たモジュール（トークン混合器と MLP など）を共有していることが分かる。**しかしながら、両モデルの決定的な違いはショートカット接続の数にある。MetaNeXt ブロックは単一のショートカット接続を実装するのに対し、MetaFormer ブロックは 2 つを組み込む——1 つはトークン混合器用、もう 1 つは MLP 用である。** この観点から、MetaNeXt ブロックは MetaFormer から 2 つの残差サブブロックを統合した結果と見なすことができ、それにより全体のアーキテクチャを単純化している。

#### ConvNeXt への具体化

図2 に示すように、ConvNeXt ではトークン混合器は単純に depthwise 畳み込みで実装される。

$$
X^{\prime}=\mathrm{TokenMixer}(X)=\mathrm{DWConv}_{k\times k}^{C\rightarrow C}(X),
$$

ConvNeXt では $k$ は既定で 7 に設定される。

### 3.2 Inception depthwise convolution（Inception depthwise 畳み込み）

**表1**: ConvNeXt-T に基づく予備実験。**Convolution ratio** は depthwise 畳み込みで処理されるチャネルの割合を意味し、残りのチャネルは変更されない。スループットは A100 GPU 上でバッチサイズ 128、TF32 で測定。\* は ConvNeXt 論文で報告された結果。

| DWConv のカーネルサイズ | 畳み込み比率 | Params (M) | MACs (G) | 訓練スループット | 推論スループット | Top-1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **7×7** | 1.0 | 28.6 | 4.5 | **575** | 2413 | **82.1\*** |
| 5×5 | 1.0 | 28.4 | 4.4 | 675 | 2704 | 82.0 |
| **3×3** | 1.0 | 28.3 | 4.4 | **798** | 2802 | **81.5** |
| 3×3 | 1/2 | 28.3 | 4.4 | 818 | 2740 | 81.4 |
| 3×3 | 3/8 | 28.3 | 4.4 | 847 | 2762 | 81.4 |
| 3×3 | 1/4 | 28.3 | 4.4 | 871 | 2808 | 81.3 |
| **3×3** | **1/8** | 28.3 | 4.4 | **901** | 2833 | 80.8 |
| 3×3 | 1/16 | 28.3 | 4.4 | 916 | 2846 | **80.1** |

#### ConvNeXt-T に基づく予備実験

まず ConvNeXt-T に基づく予備実験を行った。第一に、depthwise 畳み込みのカーネルサイズを $7\times 7$ から $3\times 3$ へ減らす。**カーネルサイズ $7\times 7$ のモデルと比べて、$3\times 3$ のものは 1.4 倍高い訓練スループットを享受するが、82.1% から 81.5% への有意な性能低下を被る。** 次に、ShuffleNet V2 に着想を得て、**入力チャネルの一部だけを depthwise 畳み込みに供給し、残りは変更しないままにする**。

> **これらの予備実験から得られた 2 つの発見（Finding 1・Finding 2）が手法の設計に直結する。**
> - **発見 1**: 大カーネルは精度に必要だが、正方の大カーネルは実速度が遅い
> - **発見 2**: 全チャネルを畳み込みに通す必要はない。**比率 1/4 までは性能低下がわずか**（82.1 → 81.3、ただしカーネルは 3×3）だが、1/16 まで下げると深刻に落ちる（80.1）

**表2**: 異なる型の畳み込みの計算量。簡単のため入力と出力のチャネル数は同じとし、バイアス項は省略する。$k$、$C$、$H$、$W$ はそれぞれカーネルサイズ、チャネル数、高さ、幅を表す。

| 畳み込みの型 | Params | FLOPs |
| --- | --- | --- |
| 通常の畳み込み | $k^{2}C^{2}$ | $2k^{2}C^{2}HW$ |
| Depthwise 畳み込み | $k^{2}C$ | $2k^{2}CHW$ |
| **Inception depthwise 畳み込み** | $(2k+9)C/8$ | $(2k+9)CHW/4$ |

**通常の畳み込みと depthwise 畳み込みのパラメータと FLOPs はカーネルサイズ $k$ の 2 次であるのに対し、Inception depthwise 畳み込みは $k$ の 1 次である。**

<figure>

![](../../raw/assets/inceptionnext/fig3.png)

<figcaption>図3: depthwise 畳み込みと Inception depthwise 畳み込みの FLOPs の比較。カーネルサイズが大きくなるにつれて、Inception depthwise 畳み込みは depthwise 畳み込みよりはるかに効率的になる。前者は k の 1 次で緩やかに増えるのに対し、後者は k の 2 次で急激に増える。</figcaption>
</figure>

#### 定式化

上記の発見に基づき、精度と効率の両方を保つ新しい型の畳み込みを提案する。**発見 2 に従い、一部のチャネルを変更しないままにし、それを恒等写像の分岐とする。発見 1 に動機づけられ、処理するチャネルについては depthwise の演算を Inception 風に分解することを提案する。** Inception は小さなカーネル（例: $3\times 3$）と大きなカーネル（例: $5\times 5$）の複数の分岐を利用する。同様に我々も $3\times 3$ を分岐の 1 つとして採用するが、**実用上の速度が遅いために大きな正方カーネルの使用は取りやめる。代わりに大カーネル $k_{h}\times k_{w}$ を $1\times k_{b}$ と $k_{b}\times 1$ に分解する。**

具体的には、入力 $X$ をチャネル次元に沿って 4 つのグループに分割する。

$$
X_{\mathrm{hw}},X_{\mathrm{w}},X_{\mathrm{h}},X_{\mathrm{id}}=\mathrm{Split}(X)=X_{:,:g},X_{:,g:2g},X_{:,2g:3g},X_{:,3g:},
$$

ここで $g$ は畳み込み分岐のチャネル数である。比率 $r_{g}$ を設定して $g=r_{g}C$ で分岐のチャネル数を決められる。次に分割された入力が異なる並列分岐に供給される。

$$
\begin{split}X^{\prime}_{\mathrm{hw}}&=\mathrm{DWConv}_{k_{s}\times k_{s}}^{g\rightarrow g}(X_{\mathrm{hw}}),\\
X^{\prime}_{\mathrm{w}}&=\mathrm{DWConv}_{1\times k_{b}}^{g\rightarrow g}(X_{\mathrm{w}}),\\
X^{\prime}_{\mathrm{h}}&=\mathrm{DWConv}_{k_{b}\times 1}^{g\rightarrow g}(X_{\mathrm{h}}),\\
X^{\prime}_{\mathrm{id}}&=X_{\mathrm{id}},\end{split}
$$

ここで $k_{s}$ は小さな正方カーネルのサイズで既定は 3、$k_{b}$ は帯状カーネルのサイズで既定は 11 である。最後に各分岐からの出力が連結される。

$$
X^{\prime}=\mathrm{Concat}(X^{\prime}_{\mathrm{hw}},X^{\prime}_{\mathrm{w}},X^{\prime}_{\mathrm{h}},X^{\prime}_{\mathrm{id}}).
$$

### 3.3 InceptionNeXt

InceptionNeXt ブロックに基づき、InceptionNeXt と名付けた一連のモデルを構築できる。**ConvNeXt が我々の主要な比較ベースラインである**ため、主にそれに従って複数のサイズのモデルを構築する。ResNet や ConvNeXt と同様に、InceptionNeXt も **4 ステージの枠組み**を採用する。ConvNeXt と同じく、4 ステージのブロック数は atto サイズで \[2, 2, 6, 2\]、small サイズで \[3, 3, 9, 3\]、base サイズで \[3, 3, 27, 3\] である。**本論文は速度を重視するため Batch Normalization を採用する。** ConvNeXt とのもう 1 つの違いは、**InceptionNeXt が第 4 ステージで MLP 比 3 を用い、節約したパラメータを分類器に移す**ことである。これは FLOPs をわずかに減らす助けになる。

**表3**（抜粋）: ConvNeXt と同様のモデル構成を持つ InceptionNeXt モデルの構成。「A」「T」「S」「B」はそれぞれ Atto、Tiny、Small、Base を表す。

| Stage | 層の仕様 | A | T | S | B |
| --- | --- | --- | --- | --- | --- |
| 1（H/4×W/4） | ダウンサンプリング | 4×4, stride 4 | | | |
| | 埋め込み次元 | 40 | 96 | 96 | 128 |
| | 帯状カーネルサイズ | 9 | 11 | 11 | 11 |
| | 畳み込みグループ比 | 1/4 | 1/8 | 1/8 | 1/8 |
| | MLP 比 | 4 | 4 | 4 | 4 |
| | ブロック数 | 2 | 3 | 3 | 3 |
| 2（H/8×W/8） | ダウンサンプリング | 2×2, stride 2 | | | |
| | 埋め込み次元 | 90 | 192 | 192 | 256 |
| | ブロック数 | 2 | 3 | 3 | 3 |
| 3（H/16×W/16） | 埋め込み次元 | 180 | 384 | 384 | 512 |
| | ブロック数 | 6 | 9 | 27 | 27 |
| 4（H/32×W/32） | 埋め込み次元 | 320 | 768 | 768 | 1024 |
| | **MLP 比** | **3** | **3** | **3** | **3** |
| | ブロック数 | 2 | 3 | 3 | 3 |

## 4 Experiment（実験）

### 4.1 Image classification（画像分類）

#### 設定

**ImageNet-1K** で評価する。Swin や ConvNeXt のような広く使われるベースラインと公平に比較するため、主に **DeiT の訓練ハイパーパラメータ**（蒸留なし）に従う。具体的には AdamW オプティマイザで学習率 $lr=0.001\times\mathrm{batchsize}/1024$（$lr=4e{-3}$、バッチサイズ 4096）で訓練する。

**表4**（抜粋）: ImageNet-1K で訓練されたモデルの性能。スループットは A100 GPU（PyTorch 1.13.0, CUDA 11.7.1）で TF32、および 2080Ti（PyTorch 1.8.1, CUDA 10.2）で FP32 で測定。スループットのベンチマークのバッチサイズは初め 128 に設定し、GPU が収容できるまで減らす。「Channel First」と「Channel Last」のメモリレイアウトのうち良い方の結果を報告する。

| Model | 混合の型 | Params (M) | MACs (G) | A100 訓練 | A100 推論 | 2080Ti 訓練 | 2080Ti 推論 | Top-1 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV2 (1.4) | Conv | 6.1 | 0.60 | 1001 | 5190 | 471 | 1859 | 74.7 |
| EfficientNet-B0 | Conv | 5.3 | 0.40 | 954 | 5502 | 464 | 1944 | 77.1 |
| ConvNeXt-A | Conv | 3.7 | 0.55 | 835 | 4539 | 345 | 1568 | 75.7 |
| **InceptionNeXt-A** | Conv | 4.2 | 0.51 | **2661 (+219%)** | **9876 (+118%)** | **992 (+188%)** | **3595 (+129%)** | 75.3 (−0.4) |
| DeiT-S | Attn | 22 | 4.6 | 1227 | 3781 | 276 | 784 | 79.8 |
| Swin-T | Attn | 29 | 4.5 | 564 | 1768 | 184 | 554 | 81.3 |
| ResNet-50 | Conv | 26 | 4.1 | 969 | 3149 | 278 | 977 | 78.4 |
| RSB-ResNet-50 | Conv | 26 | 4.1 | 969 | 3149 | 278 | 977 | 79.8 |
| RegNetY-4G | Conv | 21 | 4.0 | 670 | 2694 | 222 | 859 | 81.3 |
| FocalNet-T | Conv | 29 | 4.5 | – | – | – | – | 82.3 |
| **ConvNeXt-T** | Conv | 29 | 4.5 | **575** | 2413 | 177 | 590 | 82.1 |
| **InceptionNeXt-T** | Conv | 28 | 4.2 | **901 (+57%)** | **2900 (+20%)** | **254 (+44%)** | **822 (+39%)** | **82.3 (+0.2)** |
| Swin-S | Attn | 50 | 8.7 | 359 | 1131 | 109 | 328 | 83.0 |
| RSB-ResNet-101 | Conv | 45 | 7.9 | 620 | 2057 | 168 | 592 | 81.3 |
| FocalNet-S | Conv | 50 | 8.7 | – | – | – | – | 83.5 |

#### 結果

**InceptionNeXt は ConvNeXt より一貫して良い精度–速度のトレードオフを享受する。** 例えば **InceptionNeXt-T は ConvNeXt-T を 0.2% 上回るだけでなく、A100 上で訓練/推論スループットが 1.6 倍 / 1.2 倍**であり、これは ResNet-50 のものと同程度である。つまり **InceptionNeXt-T は ResNet-50 の速度と ConvNeXt-T の精度の両方を享受する。**

> **速度の改善は軽量なモデルサイズで遥かに顕著であり、モデルサイズが大きくなるにつれて改善は徐々に小さくなる**ことが観察される。**理由は、depthwise 畳み込みと Inception depthwise 畳み込みの計算量がチャネル数について線形 $\mathcal{O}(C)$ であるのに対し、MLP の計算量は $\mathcal{O}(C^{2})$ であるためである。** より大きなモデル（より大きな $C$）では計算が MLP に支配される。depthwise 畳み込みのみを改善しても速度の改善は小さくなる。

**表5**: ViT、等方的 ConvNeXt、InceptionNeXt の比較。**MetaNeXt-Attn は MetaNeXt のトークン混合器を自己注意として具体化したもの**である。スループットは 2080Ti（PyTorch 1.8.1, CUDA 10.2）で FP32 で測定。

| Model | Params (M) | MACs (G) | 訓練 | 推論 | Top-1 (%) |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | 22 | 4.6 | 276 | 784 | 79.8 |
| **MetaNeXt-Attn** | 22 | 4.6 | 288 | 816 | **3.9** |
| ConvNeXt-S (iso.) | 22 | 4.3 | 270 | 879 | 79.7 |
| **InceptionNeXt-S (iso.)** | 22 | 4.2 | **310** | **998** | 79.7 |

4 ステージの枠組みのほかに、もう 1 つの注目すべきものは **ViT 型の等方的（isotropic）アーキテクチャ**で、これは 1 ステージしか持たない。DeiT-S のパラメータと MACs に合わせるため、ConvNeXt-S (iso.) に従って InceptionNeXt-S (iso.) を構築する。具体的には埋め込み次元を 384、ブロック数を 18 に設定する。加えて、**MetaNeXt ブロックのトークン混合器を自己注意として具体化した MetaNeXt-Attn というモデル**を構築する。このモデルの目的は、**MetaFormer の 2 つの残差サブブロックを統合することが可能かを調べる**ことである。

> **MetaNeXt-Attn は top-1 精度 3.9% しか達成しない——すなわち収束するよう訓練できない。** これは **MetaNeXt ブロックのトークン混合器が複雑すぎてはならない**ことを示す。

### 4.2 Semantic segmentation（セマンティックセグメンテーション）

#### 設定

**ADE20K** を用いてセマンティックセグメンテーションのタスクで評価する。ADE20K は 150 の細粒度の意味カテゴリを含み、訓練セットに 2 万枚、検証セットに 2 千枚の画像を持つ。ImageNet-1K で解像度 $224^{2}$ で訓練したチェックポイントでバックボーンを初期化する。Swin と ConvNeXt に従い、まず **UperNet** で評価する。

**表6**: ADE20K 検証セットにおける **UperNet** でのセマンティックセグメンテーションの性能。画像は訓練のため $512\times 512$ に切り出される。MACs は入力サイズ $512\times 2048$ で測定。FPS は 2080Ti でベンチマーク。

| Backbone | Params (M) | MACs (G) | FPS | mIoU (%) |
| --- | --- | --- | --- | --- |
| Swin-T | 60 | 945 | 20.6 | 45.8 |
| ConvNeXt-T | 60 | 939 | 20.6 | 46.7 |
| **InceptionNeXt-T** | **56** | **933** | **22.7** | **47.9** |
| Swin-S | 81 | 1038 | 16.2 | 49.5 |
| ConvNeXt-S | 82 | 1027 | 16.8 | 49.6 |
| **InceptionNeXt-S** | **78** | **1020** | **17.6** | **50.0** |
| Swin-B | 121 | 1188 | 16.2 | 49.7 |
| ConvNeXt-B | 122 | 1170 | 15.7 | 49.9 |
| **InceptionNeXt-B** | **115** | **1159** | **17.5** | **50.6** |

**表7**: ADE20K 検証セットにおける **Semantic FPN** でのセマンティックセグメンテーションの性能。MACs は入力サイズ $512\times 512$ で測定。

| Backbone | Params (M) | MACs (G) | FPS | mIoU (%) |
| --- | --- | --- | --- | --- |
| ResNet-50 | 29 | 46 | 30.2 | 36.7 |
| PVT-Small | 28 | 45 | 27.2 | 39.8 |
| PoolFormer-S24 | 23 | 39 | 28.8 | 40.3 |
| **InceptionNeXt-T** | 28 | 44 | **31.4** | **43.1** |
| ResNet-101 | 48 | 65 | 22.2 | 38.8 |
| PVT-Medium | 48 | 61 | 20.0 | 41.6 |
| PoolFormer-S36 | 35 | 48 | 21.6 | 42.0 |
| PoolFormer-M36 | 60 | 68 | 15.4 | 42.4 |
| **InceptionNeXt-S** | 50 | 65 | 20.7 | **45.6** |
| PVT-Large | 65 | 80 | 16.0 | 42.1 |
| PoolFormer-M48 | 77 | 82 | 12.1 | 42.7 |
| **InceptionNeXt-B** | 85 | 100 | **20.2** | **46.4** |

#### 結果

**UperNet では InceptionNeXt が異なるモデルサイズにわたって一貫して Swin と ConvNeXt を上回る。** Semantic FPN では **InceptionNeXt が PVT や PoolFormer のような他のバックボーンを大幅に上回る。** これらの結果は InceptionNeXt が**密予測タスクにも高い潜在能力を持つ**ことを示す。

### 4.3 Ablation studies（アブレーション研究）

**表8**: ImageNet-1K 分類ベンチマークにおける InceptionNeXt のアブレーション。InceptionNeXt-T をベースラインとして利用。スループットは A100 GPU（TF32、バッチサイズ 128）で測定。

| アブレーション | 派生 | Params (M) | MACs (G) | 訓練 | 推論 | Top-1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **ベースライン** | **なし（InceptionNeXt-T）** | 28.1 | 4.2 | 901 | 2900 | **82.3** |
| 分岐 | 水平の帯状カーネルを除去 | 28.0 | 4.2 | 947 | 3093 | **81.9** |
| | 垂直の帯状カーネルを除去 | 28.0 | 4.2 | 954 | 3173 | **81.9** |
| | 小さな正方カーネルを除去 | 28.0 | 4.2 | 940 | 3004 | 82.0 |
| | 水平と垂直の帯状カーネルを並列 → 逐次に | 28.1 | 4.2 | 903 | 2971 | 82.1 |
| 帯状カーネルサイズ | 11 → 7 | 28.0 | 4.2 | 905 | 2946 | 82.1 |
| | 11 → 9 | 28.1 | 4.2 | 904 | 2916 | 82.1 |
| | **11 → 13** | 28.1 | 4.2 | 896 | 2895 | **82.0** |
| 畳み込み分岐比 | 1/8 → 1/4 | 28.1 | 4.2 | **834** | **2499** | 82.2 |
| | **1/8 → 1/16** | 28.0 | 4.2 | 936 | 3097 | **81.8** |
| 正規化 | **BatchNorm → LayerNorm** | 28.1 | 4.2 | **721** | 2646 | **82.4** |

#### 分岐

Inception depthwise 畳み込みは 4 つの分岐（3 つの畳み込みと恒等写像）を含む。**水平または垂直の帯状カーネルのいずれかの分岐を除去すると、性能は 82.3% から 81.9% へ有意に落ちる。これはこれら 2 つの分岐の重要性を実証する。この 2 つの帯状カーネルの分岐がモデルの受容野を広げられるためである。** $3\times 3$ の小さな正方カーネルの分岐については、除去しても 82.0% の top-1 精度を達成でき、より高いスループットをもたらす。

#### 帯状カーネルサイズ

**カーネルサイズ 7 から 11 へは性能が改善するが、13 に増やすと落ちる**ことが分かる。この現象は最適化に起因するかもしれず、構造的な再パラメータ化のような手法で解決できる可能性がある。簡単のため既定でカーネルサイズを 11 に設定する（atto サイズを除く）。

#### 畳み込み分岐比

比率を $1/8$ から $1/4$ へ増やしても性能の改善は観察できない。**Ma らも全チャネルで畳み込みを行う必要はないと指摘している。** しかし比率を $1/16$ まで下げると深刻な性能低下をもたらす。**これはより小さな比率がトークン混合の程度を制限するためである。** したがって既定で畳み込み分岐比を $1/8$ に設定する。

#### 正規化

**Batch Normalization を Layer Normalization に置き換えると性能は 0.1% 改善するが、訓練と推論の両方でスループットが低下する**（訓練 901 → 721）。**本論文は効率に焦点を当てるため、InceptionNeXt には Batch Normalization を採用する。**

## 5 Conclusion（結論）

本研究では、従来のネットワークアーキテクチャより**実用上の速度と性能のより良いトレードオフ**を享受する効果的かつ効率的な CNN アーキテクチャ InceptionNeXt を提案した。InceptionNeXt は大カーネルの depthwise 畳み込みをチャネル次元に沿って 4 つの並列分岐（恒等写像、小さな正方カーネル、2 つの直交する帯状カーネル）に分解する。**これら 4 つの分岐はすべて大カーネルの depthwise 畳み込みより実用上はるかに計算効率が良く、同時に協働して大きな受容野を持つことができる。**

## Appendix A Hyper-parameters（ハイパーパラメータ）

### A.1 ImageNet-1K image classification（ImageNet-1K 画像分類）

ImageNet-1K の分類ベンチマークにおいて、ConvNeXt と timm で訓練された ConvNeXt-A に従い、表9 に示すハイパーパラメータを採用して入力解像度 $224^{2}$ で InceptionNeXt を訓練し、$384^{2}$ でファインチューニングする。我々のコードは **timm** ライブラリに基づき PyTorch で実装されている。

**表9**（抜粋）: ImageNet-1K 画像分類における InceptionNeXt のハイパーパラメータ。

| | A（訓練） | T / S / B（訓練） | B（ファインチューン） |
| --- | --- | --- | --- |
| 入力解像度 | 224² | 224² | 384² |
| エポック | 450 | 300 | 30 |
| バッチサイズ | 1280 | 4096 | 1024 |
| オプティマイザ | AdamW | AdamW | AdamW |
| Adam $\epsilon$ | 1e-8 | 1e-8 | 1e-8 |
| Adam ($\beta_1,\beta_2$) | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) |
| 学習率 | 1e-3 | 4e-3 | 5e-5 |
| 学習率の減衰 | Cosine | Cosine | Cosine |
| 勾配クリッピング | なし | なし | なし |
| ウォームアップエポック | 5 | 20 | なし |
| Weight decay | 0.05 | 0.05 | 0.05 |
| RandAugment | 5/uniform | 9/0.5 | 9/0.5 |
| Repeated Augmentation | off | off | off |
| CutMix | 1.0 | 1.0 | 1.0 |
| Mixup | 0.2 | 0.8 | 0.8 |
| CutMix-Mixup 切替確率 | 0.5 | 0.5 | 0.5 |
| Random erasing 確率 | 0.1 | 0.25 | 0.25 |
| Label smoothing | 0.1 | 0.1 | 0.1 |
| Peak stochastic depth 率 | 0.1 | T: 0.1 / S: 0.3 / B: 0.4 | 0.7 |
| 分類器の Dropout | 0.0 | 0.0 | 0.5 |
| LayerScale 初期化 | 1e-6 | 1e-6 | 事前学習済み |
| EMA 減衰率 | なし | なし | 0.9999 |

### A.2 Semantic segmentation（セマンティックセグメンテーション）

ADE20K のセマンティックセグメンテーションについては、Swin の設定に従って **UperNet** をバックボーンとともに利用し、PVT と PoolFormer の設定に従って **FPN** を利用する。バックボーンは ImageNet-1K で解像度 $224^{2}$ で事前学習されたチェックポイントで初期化される。

**表10**: ADE20K セマンティックセグメンテーションにおける UperNet と FPN での InceptionNeXt バックボーンの stochastic depth 率。

| Method | T | S | B |
| --- | --- | --- | --- |
| UperNet | 0.2 | 0.3 | 0.4 |
| FPN | 0.1 | 0.2 | 0.2 |

## Appendix B Qualitative results（定性的な結果）

**Grad-CAM** を用いて、ImageNet-1K で訓練された異なるモデル——RSB-ResNet-50、Swin-T、ConvNeXt-T、そして我々の InceptionNeXt-T——の活性化マップを可視化する。結果は図4 に示されている。**他のモデルと比べて、InceptionNeXt-T はより小さな活性化領域で重要な部分をより正確に位置特定する。**

<figure>

![](../../raw/assets/inceptionnext/fig4.png)

<figcaption>図4: ImageNet-1K で訓練された異なるモデルの Grad-CAM 活性化マップ。上から順に、入力画像、RSB-ResNet-50、Swin-T、ConvNeXt-T、InceptionNeXt-T。左から順に 4 つの検証画像。InceptionNeXt-T は他のモデルより小さな活性化領域で重要な部分をより正確に位置特定している。</figcaption>
</figure>
